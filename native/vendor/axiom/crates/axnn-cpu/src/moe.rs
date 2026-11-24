//! Mixture-of-experts dispatch over a borrowed expert store: sigmoid-style
//! routing with a selection-only correction bias, then each token's picked
//! experts, summed in ascending expert order.
//!
//! The store is a table of per-expert byte slices and scales rather than
//! one packed blob, so it can point into memory-mapped checkpoint tensors
//! as stored, or into whichever experts a residency cache holds.

mod continuation;
pub use continuation::{
    DEFAULT_WAVE_EXPERTS, MAX_WAVE_EXPERTS, MOE_PHASE_NAMES, MOE_PHASES, MoeBatchRow,
    MoeInvocation, MoePhases, MoeProgress, ResumableMoeDelegate, moe_prepared_scratch_elements,
    native_wave,
};

use crate::fork_join::{ForkJoin, for_rows};
use crate::neural::bf16;
use crate::planner::{typed_input_matches, typed_output_matches};
use crate::tensor_ops::select_top_k;
use crate::{CpuCapabilities, CpuError, CpuImplementation, CpuKernelPlan};
use axnn::DType;
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, ExpertActivation, KernelAccuracy, KernelCandidate,
    KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelRequest, KernelShapePolicy,
    MAX_MOE_TOP_K, MoeParams, SemanticOp,
};

/// One expert's projections, stored `[out, in]` in the store's dtype, each
/// with its F32 dequantization scale (1 for BF16).
#[derive(Clone, Copy, Debug)]
pub struct ExpertWeights<'a> {
    /// `[intermediate, hidden]`.
    pub up: &'a [u8],
    /// `[hidden, intermediate]`.
    pub down: &'a [u8],
    /// E4M3 scales, one per sixteen logical `up` weights (NVFP4 only).
    pub up_block_scales: &'a [u8],
    /// E4M3 scales, one per sixteen logical `down` weights (NVFP4 only).
    pub down_block_scales: &'a [u8],
    pub up_scale: f32,
    pub down_scale: f32,
}

impl ExpertWeights<'_> {
    /// No weights: a placeholder a source overwrites.
    pub const EMPTY: Self = Self {
        up: &[],
        down: &[],
        up_block_scales: &[],
        down_block_scales: &[],
        up_scale: 0.0,
        down_scale: 0.0,
    };

    /// Whether these projections fit `params`.
    fn fits(&self, params: MoeParams) -> bool {
        let bytes = params.projection_bytes();
        let scale_bytes = params.projection_scale_bytes();
        Some(self.up.len()) == bytes
            && Some(self.down.len()) == bytes
            && Some(self.up_block_scales.len()) == scale_bytes
            && Some(self.down_block_scales.len()) == scale_bytes
            && self.up_scale.is_finite()
            && self.down_scale.is_finite()
    }
}

/// Where weights live, as far as residency goes.
///
/// A kernel names the bytes it is about to read, as soon as it knows them,
/// so the backend can start bringing them in: page-cache readahead for a
/// mapped file, a transfer into device memory, or nothing at all for
/// resident weights. A hint never changes a result, and may be ignored.
pub trait WeightResidency: Sync {
    fn will_read(&self, bytes: &[u8]);
}

/// A layer's experts supplied on demand rather than all addressable at
/// once: a cache over a checkpoint larger than memory, or device memory a
/// transfer fills.
///
/// # Safety
///
/// Every weight `load` writes must stay valid and unchanged until the next
/// `load` on this source begins. (Callers, the expert kernels, never use
/// them past that point.)
pub unsafe trait ExpertSource: Sync {
    /// How many experts the layer has.
    fn experts(&self) -> usize;

    /// Make `picks` (distinct, ascending) readable and write their weights
    /// to `out`, in the same order. `hidden` is the token's input to this
    /// layer, for a source that anticipates later layers' picks. When an
    /// expert is shared by a group of tokens, it is the first selected
    /// token's input; this hint must not specialize the returned weights.
    fn load<'s>(
        &'s self,
        picks: &[u32],
        hidden: &[f32],
        out: &mut [ExpertWeights<'s>],
    ) -> Result<(), CpuError>;

    /// A later `load` of this kernel invocation will ask for `picks`
    /// (distinct, ascending): a source may start reading them now, while
    /// earlier tokens compute. Calls come in the order the loads will. A
    /// hint never changes a result, and may be ignored.
    fn will_load(&self, picks: &[u32]) {
        let _ = picks;
    }

    /// The most distinct experts one `load` can hold at once, for a source
    /// sized beyond one token's picks: an expert wave loads up to this many
    /// together. `None`, the default, is the model's `top_k`.
    fn max_picks(&self) -> Option<usize> {
        None
    }

    /// Every expert of the layer, in expert order, without a load: `true`
    /// only when each is readable now and stays readable and unmoved for the
    /// rest of this source's life, as a source holding the whole layer
    /// resident and never evicting can promise. A device that routes the
    /// layer itself then reads the weights in place, possibly after later
    /// loads of other layers. `out` holds [`Self::experts`] entries. `false`,
    /// the default, leaves `out` unspecified and the experts to `load`.
    fn resident<'s>(&'s self, out: &mut [ExpertWeights<'s>]) -> bool {
        let _ = out;
        false
    }
}

#[derive(Clone, Copy)]
enum Experts<'a> {
    Table(&'a [ExpertWeights<'a>]),
    Source(&'a dyn ExpertSource),
}

/// Every expert of one layer, borrowed for one invocation: a table in
/// expert order, or a source that supplies each token's picks.
#[derive(Clone, Copy)]
pub struct ExpertStoreRef<'a> {
    experts: Experts<'a>,
    residency: Option<&'a dyn WeightResidency>,
}

impl core::fmt::Debug for ExpertStoreRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("ExpertStoreRef");
        match self.experts {
            Experts::Table(table) => f.field("table", &table),
            Experts::Source(source) => f.field("source", &source.experts()),
        };
        f.field("residency", &self.residency.is_some()).finish()
    }
}

impl<'a> ExpertStoreRef<'a> {
    /// Every expert, in expert order, when the store holds them all in place
    /// for good: a table always does, a source when [`ExpertSource::resident`]
    /// says so. `out` must hold every expert; `false` leaves it unspecified.
    pub fn resident(&self, out: &mut [ExpertWeights<'a>]) -> bool {
        match self.experts {
            Experts::Table(table) => match out.get_mut(..table.len()) {
                Some(out) => {
                    out.copy_from_slice(table);
                    true
                }
                None => false,
            },
            Experts::Source(source) => match out.get_mut(..source.experts()) {
                Some(out) => source.resident(out),
                None => false,
            },
        }
    }

    /// A store of exactly `params.experts` experts whose projections are
    /// exactly `params.projection_bytes()` long, with finite scales.
    pub fn new(experts: &'a [ExpertWeights<'a>], params: MoeParams) -> Result<Self, CpuError> {
        let store = Self {
            experts: Experts::Table(experts),
            residency: None,
        };
        store.check(params)?;
        Ok(store)
    }

    /// A store whose experts `source` supplies per token. Their sizes and
    /// scales are checked as each load returns them.
    pub fn from_source(source: &'a dyn ExpertSource, params: MoeParams) -> Result<Self, CpuError> {
        let store = Self {
            experts: Experts::Source(source),
            residency: None,
        };
        store.check(params)?;
        Ok(store)
    }

    /// Tell `residency` which experts each token routes to before reading
    /// them, so all of a token's picks can load at once.
    pub fn with_residency(self, residency: &'a dyn WeightResidency) -> Self {
        Self {
            residency: Some(residency),
            ..self
        }
    }

    pub fn len(self) -> usize {
        match self.experts {
            Experts::Table(table) => table.len(),
            Experts::Source(source) => source.experts(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Whether this store fits `params`: count, projection sizes, scales.
    pub fn check(self, params: MoeParams) -> Result<(), CpuError> {
        let table_fits = match self.experts {
            Experts::Table(table) => table.iter().all(|expert| expert.fits(params)),
            Experts::Source(_) => true,
        };
        if !params.is_valid() || self.len() != params.experts as usize || !table_fits {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed expert store",
            });
        }
        Ok(())
    }
}

const fn q8_scratch_elements(elements: usize) -> usize {
    elements.div_ceil(4) + elements.div_ceil(16)
}

/// Scratch F32 elements `MOE_EXPERTS` needs: every pick's activations and,
/// for native NVFP4, reusable W4A8 activation codes and block scales.
pub fn moe_scratch_elements(params: MoeParams) -> usize {
    let activations = params.top_k as usize * params.intermediate as usize;
    if params.weights == DType::U8 {
        let hidden = q8_scratch_elements(params.hidden as usize);
        let experts = params.top_k as usize * q8_scratch_elements(params.intermediate as usize);
        activations + if hidden > experts { hidden } else { experts }
    } else {
        activations
    }
}

/// Execute an already-routed subset of resident NVFP4 experts for one token.
///
/// `mixture_weights` is aligned with `experts` and already includes routing
/// normalization and [`MoeParams::routed_scale`]. This is the common CPU
/// half of heterogeneous MoE execution: routing and residency remain owned
/// by the caller, while a scheduler may divide the loaded experts between
/// this kernel and an accelerator. The output is replaced with this subset's
/// weighted sum.
pub fn invoke_loaded_nvfp4_experts(
    plan: &CpuKernelPlan,
    params: MoeParams,
    hidden: &[f32],
    experts: &[ExpertWeights<'_>],
    mixture_weights: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let invalid = || CpuError::ShapeMismatch {
        op: "loaded NVFP4 expert batch",
    };
    let (width, inner) = (params.hidden as usize, params.intermediate as usize);
    let count = experts.len();
    if plan.semantic != SemanticOp::MoeExperts
        || !matches!(
            plan.implementation,
            CpuImplementation::Scalar | CpuImplementation::PortableSimd | CpuImplementation::Qdot
        )
        || params.weights != DType::U8
        || hidden.len() != width
        || output.len() != width
        || count == 0
        || count > params.top_k as usize
        || mixture_weights.len() != count
        || scratch.len()
            < count * inner + q8_scratch_elements(width).max(count * q8_scratch_elements(inner))
        || experts.iter().any(|expert| !expert.fits(params))
    {
        return Err(invalid());
    }

    let scalar = plan.implementation == CpuImplementation::Scalar;
    let qdot = plan.implementation == CpuImplementation::Qdot;
    let (activations, quantized_scratch) = scratch.split_at_mut(count * inner);
    let row_bytes = width / 2;
    let row_scales = width / 16;
    let column_bytes = inner / 2;
    let column_scales = inner / 16;

    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    let hidden_q =
        qdot.then(|| quantize_nvfp4(hidden, &mut quantized_scratch[..q8_scratch_elements(width)]));
    for_rows(parallel, activations, 1, width, |first, rows| {
        let (mut at, mut rows) = (first, rows);
        while !rows.is_empty() {
            let (e, r) = (at / inner, at % inner);
            let (here, later) = rows.split_at_mut((inner - r).min(rows.len()));
            let expert = &experts[e];
            let packed = &expert.up[r * row_bytes..(r + here.len()) * row_bytes];
            let scales = &expert.up_block_scales[r * row_scales..(r + here.len()) * row_scales];
            #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
            if let Some(hidden_q) = &hidden_q {
                project_qdot(packed, scales, expert.up_scale, hidden_q, here);
            } else {
                project(
                    DType::U8,
                    scalar,
                    packed,
                    scales,
                    expert.up_scale,
                    hidden,
                    here,
                );
            }
            #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
            project(
                DType::U8,
                scalar,
                packed,
                scales,
                expert.up_scale,
                hidden,
                here,
            );
            // The projection applied the tensor scale already.
            for activation in here.iter_mut() {
                let rectified = activation.max(0.0);
                *activation = rectified * rectified;
            }
            at += here.len();
            rows = later;
        }
    });

    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    if qdot {
        let stride = q8_scratch_elements(inner);
        for (expert, values) in activations.chunks_exact(inner).enumerate() {
            quantize_nvfp4(
                values,
                &mut quantized_scratch[expert * stride..(expert + 1) * stride],
            );
        }
    }
    output.fill(0.0);
    let activations = &*activations;
    let quantized_scratch = &*quantized_scratch;
    for_rows(parallel, output, 1, count * inner, |first, rows| {
        let mut down = [0.0_f32; DOWN_ROWS];
        for (c, rows) in (first..).step_by(DOWN_ROWS).zip(rows.chunks_mut(DOWN_ROWS)) {
            rows.fill(0.0);
            let down = &mut down[..rows.len()];
            for (e, expert) in experts.iter().enumerate() {
                let packed = &expert.down[c * column_bytes..(c + rows.len()) * column_bytes];
                let scales =
                    &expert.down_block_scales[c * column_scales..(c + rows.len()) * column_scales];
                #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
                if qdot {
                    let stride = q8_scratch_elements(inner);
                    let quantized =
                        borrowed_nvfp4(inner, &quantized_scratch[e * stride..(e + 1) * stride]);
                    project_qdot(packed, scales, expert.down_scale, &quantized, down);
                } else {
                    project(
                        DType::U8,
                        scalar,
                        packed,
                        scales,
                        expert.down_scale,
                        &activations[e * inner..(e + 1) * inner],
                        down,
                    );
                }
                #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
                project(
                    DType::U8,
                    scalar,
                    packed,
                    scales,
                    expert.down_scale,
                    &activations[e * inner..(e + 1) * inner],
                    down,
                );
                for (value, &projected) in rows.iter_mut().zip(&*down) {
                    *value += mixture_weights[e] * projected;
                }
            }
        }
    });
    Ok(())
}

/// Offer scalar and SIMD expert dispatch, like the float-weight matmuls.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    if request.semantic != SemanticOp::MoeExperts {
        return false;
    }
    let KernelParams::MoeExperts(params) = request.params() else {
        return true;
    };
    if !params.is_valid() {
        return true;
    }
    let tokens = request
        .input(0)
        .map(|input| input.shape.num_elements() / params.hidden as usize);
    let Some(bytes) = tokens
        .and_then(|tokens| moe_prepared_scratch_elements(params, tokens))
        .and_then(|elements| elements.checked_mul(size_of::<f32>()))
    else {
        return true;
    };
    let constraints = KernelConstraints {
        input_dtypes: [DTypeSet::one(DType::F32); 3],
        output_dtypes: DTypeSet::one(DType::F32),
        max_rank: 2,
        shape_policy: KernelShapePolicy::Typed,
        ..KernelConstraints::elementwise(DType::F32)
    };
    for (implementation, required_capabilities, cost) in [
        (
            CpuImplementation::Scalar,
            CapabilityMask::NONE,
            KernelCost::new(0, 12),
        ),
        (
            CpuImplementation::PortableSimd,
            CpuCapabilities::PORTABLE_SIMD.mask(),
            KernelCost::new(32, 2),
        ),
    ] {
        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation,
            required_capabilities,
            constraints,
            // Sums run in lanes and the expert order is fixed.
            accuracy: KernelAccuracy::Faithful,
            cost: cost.with_temporary_bytes(bytes),
        });
    }
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    if params.weights == DType::U8 {
        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation: CpuImplementation::Qdot,
            required_capabilities: CpuCapabilities::QDOT.mask(),
            constraints,
            accuracy: KernelAccuracy::Approximate,
            cost: KernelCost::new(64, 0).with_temporary_bytes(bytes),
        });
    }
    true
}

#[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
fn quantize_nvfp4<'a>(values: &[f32], scratch: &'a mut [f32]) -> crate::arch::qf4::Quantized<'a> {
    let biased_words = values.len().div_ceil(4);
    let (biased, scales) = scratch.split_at_mut(biased_words);
    // SAFETY: the first `ceil(len / 4)` F32 words exclusively own at
    // least `len` writable bytes and retain F32 alignment for `scales`.
    let biased =
        unsafe { core::slice::from_raw_parts_mut(biased.as_mut_ptr().cast::<u8>(), values.len()) };
    let scales = &mut scales[..values.len().div_ceil(16)];
    crate::arch::qf4::quantize(values, biased, scales);
    crate::arch::qf4::Quantized { biased, scales }
}

#[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
fn borrowed_nvfp4<'a>(elements: usize, scratch: &'a [f32]) -> crate::arch::qf4::Quantized<'a> {
    let biased_words = elements.div_ceil(4);
    let (biased, scales) = scratch.split_at(biased_words);
    // SAFETY: this is the immutable view of storage initialized by
    // `quantize_nvfp4` with the same element count.
    let biased = unsafe { core::slice::from_raw_parts(biased.as_ptr().cast::<u8>(), elements) };
    crate::arch::qf4::Quantized {
        biased,
        scales: &scales[..elements.div_ceil(16)],
    }
}

#[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
fn project_qdot(
    rows: &[u8],
    block_scales: &[u8],
    tensor_scale: f32,
    x: &crate::arch::qf4::Quantized<'_>,
    out: &mut [f32],
) {
    crate::arch::qf4::nvfp4_q8_f32(rows, block_scales, x, out);
    for value in out {
        *value *= tensor_scale;
    }
}

/// `out[r] = dot(decode(rows[r]), x)` for consecutive weight rows, as the
/// float-weight matmul computes them.
#[inline]
fn project(
    dtype: DType,
    scalar: bool,
    rows: &[u8],
    block_scales: &[u8],
    tensor_scale: f32,
    x: &[f32],
    out: &mut [f32],
) {
    match (dtype, scalar) {
        (DType::U8, _) => {
            crate::arch::f4::nvfp4_f32(rows, block_scales, x, out);
            for value in out {
                *value *= tensor_scale;
            }
        }
        (DType::B8, false) => crate::arch::gemv::e4m3_f32(rows, x, out),
        (_, false) => crate::arch::gemv::bf16_f32(rows, x, out),
        (DType::B8, true) => {
            for (w, y) in rows.chunks_exact(x.len()).zip(out) {
                *y = w
                    .iter()
                    .zip(x)
                    .fold(0.0, |sum, (&b, &v)| sum + crate::fp8::e4m3_to_f32(b) * v);
            }
        }
        (_, true) => {
            for (w, y) in rows.chunks_exact(2 * x.len()).zip(out) {
                *y = w
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(x)
                    .fold(0.0, |sum, (&b, &v)| sum + bf16(b) * v);
            }
        }
    }
}

/// Output rows a down-projection task sums at a time, on its stack: wide
/// enough that a transposed row's slice spans several cache lines.
const DOWN_ROWS: usize = 256;

/// One token's routing: its top-`k` picks by biased score, their weights
/// (unbiased scores, normalized if asked), and the picks in ascending
/// expert order, as a per-expert loop would apply them.
struct Route {
    weights: [f32; MAX_MOE_TOP_K as usize],
    picked: [i32; MAX_MOE_TOP_K as usize],
    /// Pick slots in ascending expert order, and those experts.
    order: [usize; MAX_MOE_TOP_K as usize],
    ascending: [u32; MAX_MOE_TOP_K as usize],
}

impl Route {
    fn new(row_scores: &[f32], correction_bias: &[f32], k: usize, normalize: bool) -> Self {
        let mut choice = [0.0_f32; MAX_MOE_TOP_K as usize];
        let mut picked = [0_i32; MAX_MOE_TOP_K as usize];
        select_top_k(
            row_scores
                .iter()
                .zip(correction_bias)
                .map(|(score, bias)| score + bias),
            &mut choice[..k],
            &mut picked[..k],
        );
        let mut weights = [0.0_f32; MAX_MOE_TOP_K as usize];
        for (weight, &expert) in weights.iter_mut().zip(&picked[..k]) {
            *weight = row_scores[expert as usize];
        }
        if normalize {
            let denominator = weights[..k].iter().sum::<f32>() + 1e-20;
            for weight in &mut weights[..k] {
                *weight /= denominator;
            }
        }
        let mut order = [0_usize; MAX_MOE_TOP_K as usize];
        for (slot, position) in order[..k].iter_mut().enumerate() {
            *position = slot;
        }
        order[..k].sort_unstable_by_key(|&slot| picked[slot]);
        let mut ascending = [0_u32; MAX_MOE_TOP_K as usize];
        for (expert, &slot) in ascending.iter_mut().zip(&order[..k]) {
            *expert = picked[slot] as u32;
        }
        Self {
            weights,
            picked,
            order,
            ascending,
        }
    }
}

/// Route every token of `hidden` and sum its experts into `output`.
///
/// Validates the plan, inputs and store before writing. Per token: pick
/// `top_k` experts by `scores + correction_bias`; weight each by its
/// unbiased score, divided by the picks' sum plus `1e-20` when
/// `normalize`, times `routed_scale`; then, in ascending expert order,
/// add `weight * down_scale * down(act(up_scale * up(x)))`.
#[allow(clippy::too_many_arguments)]
pub fn invoke_moe_experts(
    plan: &CpuKernelPlan,
    store: ExpertStoreRef<'_>,
    hidden: &[f32],
    scores: &[f32],
    correction_bias: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    invoke_moe_experts_with(
        plan,
        store,
        hidden,
        scores,
        correction_bias,
        output,
        scratch,
        parallel,
        &mut |_, _, _, _, _, _, _| Ok(false),
    )
}

/// What another executor may take from one token of the MoE kernel.
///
/// Two grains.  [`token`](Self::token) takes the whole token — the up
/// projections, the activation, the down projections and the mixture —
/// and suits experts whose down weights are resident.  [`up`](Self::up)
/// takes only the up projections of the token's chosen experts, writing
/// their pre-activation outputs (`experts.len() × intermediate`, expert
/// by expert in the order given). NVFP4 includes block and tensor scales;
/// FP8/BF16 leaves the tensor scale to the native activation. It leaves
/// the activation, the down rows and the mixture to
/// this kernel — the grain the streamed-expert path needs, whose down
/// rows are sparse and fetched by the activation, and the one an
/// accelerator's expert-up projection package computes.  Either returns
/// `false` to keep the CPU path for that token.
pub trait MoeDelegate<E> {
    #[allow(clippy::too_many_arguments)]
    fn token(
        &mut self,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        experts: &[ExpertWeights<'_>],
        mixture_weights: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<bool, E>;

    fn up(
        &mut self,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        experts: &[ExpertWeights<'_>],
        activations: &mut [f32],
    ) -> Result<bool, E> {
        let _ = (plan, params, hidden, experts, activations);
        Ok(false)
    }
}

/// The closure form of [`MoeDelegate`]: a whole-token delegate and no
/// up-projection one — what [`invoke_moe_experts_with`] accepts.
struct TokenDelegate<'f, F>(&'f mut F);

impl<E, F> MoeDelegate<E> for TokenDelegate<'_, F>
where
    F: FnMut(
        &CpuKernelPlan,
        MoeParams,
        &[f32],
        &[ExpertWeights<'_>],
        &[f32],
        &mut [f32],
        &mut [f32],
    ) -> Result<bool, E>,
{
    fn token(
        &mut self,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        experts: &[ExpertWeights<'_>],
        mixture_weights: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<bool, E> {
        (self.0)(
            plan,
            params,
            hidden,
            experts,
            mixture_weights,
            output,
            scratch,
        )
    }
}

/// [`invoke_moe_experts`] with an optional delegate for each routed, loaded
/// expert batch. The delegate sees experts in ascending expert-number order
/// and mixture weights in the same order. Returning `false` executes the
/// selected CPU kernel; returning `true` publishes the delegate's output.
///
/// This boundary deliberately follows routing and residency. It permits a
/// heterogeneous scheduler to operate on complete experts without copying
/// checkpoint ownership, duplicating cache policy, or changing routing.
#[allow(clippy::too_many_arguments)]
pub fn invoke_moe_experts_with<E, F>(
    plan: &CpuKernelPlan,
    store: ExpertStoreRef<'_>,
    hidden: &[f32],
    scores: &[f32],
    correction_bias: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
    delegate: &mut F,
) -> Result<(), E>
where
    E: From<CpuError>,
    F: FnMut(
        &CpuKernelPlan,
        MoeParams,
        &[f32],
        &[ExpertWeights<'_>],
        &[f32],
        &mut [f32],
        &mut [f32],
    ) -> Result<bool, E>,
{
    invoke_moe_experts_delegated(
        plan,
        store,
        hidden,
        scores,
        correction_bias,
        output,
        scratch,
        parallel,
        &mut TokenDelegate(delegate),
    )
}

/// [`invoke_moe_experts_with`] over a [`MoeDelegate`], which may take a
/// whole token or only its up projections.
#[allow(clippy::too_many_arguments)]
pub fn invoke_moe_experts_delegated<E>(
    plan: &CpuKernelPlan,
    store: ExpertStoreRef<'_>,
    hidden: &[f32],
    scores: &[f32],
    correction_bias: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
    delegate: &mut dyn MoeDelegate<E>,
) -> Result<(), E>
where
    E: From<CpuError>,
{
    let params = validate_invocation(
        plan,
        store,
        hidden,
        scores,
        correction_bias,
        output,
        scratch,
    )?;
    let (width, k) = (params.hidden as usize, params.top_k as usize);
    let (experts, inner) = (params.experts as usize, params.intermediate as usize);
    // Read-ahead hints, once, in the order the loads will come.
    if let Experts::Source(source) = store.experts
        && scores.len() > experts
    {
        for row in scores.chunks_exact(experts) {
            let route = Route::new(row, correction_bias, k, params.normalize);
            source.will_load(&route.ascending[..k]);
        }
    }
    let mut chosen = [ExpertWeights::EMPTY; MAX_MOE_TOP_K as usize];
    let mut mixture = [0.0_f32; MAX_MOE_TOP_K as usize];
    for (token, (x, out)) in hidden
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
        .enumerate()
    {
        let route = Route::new(
            &scores[token * experts..(token + 1) * experts],
            correction_bias,
            k,
            params.normalize,
        );
        // Discard the previous token's borrows before a load may evict.
        chosen.fill(ExpertWeights::EMPTY);
        match store.experts {
            Experts::Table(table) => {
                for (weights, &slot) in chosen.iter_mut().zip(&route.order[..k]) {
                    *weights = table[route.picked[slot] as usize];
                }
            }
            Experts::Source(source) => source.load(&route.ascending[..k], x, &mut chosen[..k])?,
        }
        if chosen[..k].iter().any(|expert| !expert.fits(params)) {
            return Err(CpuError::ShapeMismatch {
                op: "loaded expert weights",
            }
            .into());
        }
        if let Some(residency) = store.residency {
            for expert in &chosen[..k] {
                residency.will_read(expert.up);
                residency.will_read(expert.down);
            }
        }
        for (weight, &slot) in mixture.iter_mut().zip(&route.order[..k]) {
            *weight = route.weights[slot] * params.routed_scale.to_f32();
        }
        if delegate.token(plan, params, x, &chosen[..k], &mixture[..k], out, scratch)? {
            continue;
        }
        let up_taken = delegate.up(plan, params, x, &chosen[..k], &mut scratch[..k * inner])?;
        invoke_loaded_experts(
            plan,
            params,
            x,
            &chosen[..k],
            &mixture[..k],
            out,
            scratch,
            parallel,
            up_taken,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_invocation(
    plan: &CpuKernelPlan,
    store: ExpertStoreRef<'_>,
    hidden: &[f32],
    scores: &[f32],
    correction_bias: &[f32],
    output: &[f32],
    scratch: &[f32],
) -> Result<MoeParams, CpuError> {
    let invalid = || CpuError::ShapeMismatch {
        op: "planned expert dispatch",
    };
    let KernelParams::MoeExperts(params) = plan.params() else {
        return Err(invalid());
    };
    if !params.is_valid() {
        return Err(invalid());
    }
    let width = params.hidden as usize;
    let experts = params.experts as usize;
    if plan.semantic != SemanticOp::MoeExperts
        || !matches!(
            plan.implementation,
            CpuImplementation::Scalar | CpuImplementation::PortableSimd | CpuImplementation::Qdot
        )
        || plan.effects() != KernelEffects::READS_RESOURCE
        || Some(plan.temporary_bytes)
            != moe_prepared_scratch_elements(params, hidden.len() / params.hidden as usize)
                .and_then(|elements| elements.checked_mul(size_of::<f32>()))
        || scratch.len() < moe_scratch_elements(params)
        || plan.result_count() != 1
        || !typed_input_matches(plan, 0, DType::F32, hidden.len())
        || !typed_input_matches(plan, 1, DType::F32, scores.len())
        || !typed_input_matches(plan, 2, DType::F32, correction_bias.len())
        || (0..3).any(|i| plan.input(i).is_none_or(|d| d.layout.offset() != 0))
        || !typed_output_matches(plan, 0, DType::F32, plan.shape, output.len())
        || plan.result_tensor(0).is_none_or(|d| d.layout.offset() != 0)
        || hidden.len() / width != scores.len() / experts
    {
        return Err(invalid());
    }
    store.check(params)?;
    Ok(params)
}

/// Shared native tail for synchronous and suspended expert execution.
#[allow(clippy::too_many_arguments)]
fn invoke_loaded_experts(
    plan: &CpuKernelPlan,
    params: MoeParams,
    x: &[f32],
    chosen: &[ExpertWeights<'_>],
    mixture_weights: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
    up_taken: bool,
) -> Result<(), CpuError> {
    let inner = params.intermediate as usize;
    let k = chosen.len();
    if !up_taken && params.weights == DType::U8 {
        return invoke_loaded_nvfp4_experts(
            plan,
            params,
            x,
            chosen,
            mixture_weights,
            out,
            scratch,
            parallel,
        );
    }
    let (activations, quantized_scratch) = scratch.split_at_mut(k * inner);
    activate_loaded_experts(
        plan,
        params,
        x,
        chosen,
        activations,
        quantized_scratch,
        parallel,
        up_taken,
        true,
    );
    mix_loaded_experts(
        plan,
        params,
        chosen,
        activations,
        mixture_weights,
        out,
        quantized_scratch,
        parallel,
    )
}

/// The squared-ReLU activation of one expert's up projection, in place:
/// NVFP4 results carry their tensor scale already, FP8/BF16 ones apply it
/// here.
pub(crate) fn activate_expert(params: MoeParams, expert: &ExpertWeights<'_>, values: &mut [f32]) {
    let ExpertActivation::SquaredRelu = params.activation;
    let scale = if params.weights == DType::U8 {
        1.0
    } else {
        expert.up_scale
    };
    for activation in values.iter_mut() {
        let rectified = (*activation * scale).max(0.0);
        *activation = rectified * rectified;
    }
}

/// Fill `activations` (`chosen.len()` rows of `intermediate`) for token `x`:
/// the up projections unless a delegate took them (`up_taken`), then, when
/// `activate`, the squared ReLU. Without `activate` the rows keep the
/// delegate's pre-activation convention (NVFP4 scaled, FP8/BF16 not).
#[allow(clippy::too_many_arguments)]
pub(crate) fn activate_loaded_experts(
    plan: &CpuKernelPlan,
    params: MoeParams,
    x: &[f32],
    chosen: &[ExpertWeights<'_>],
    activations: &mut [f32],
    quantized_scratch: &mut [f32],
    parallel: &dyn ForkJoin,
    up_taken: bool,
    activate: bool,
) {
    let (width, inner) = (params.hidden as usize, params.intermediate as usize);
    let scalar = plan.implementation == CpuImplementation::Scalar;
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    let qdot = plan.implementation == CpuImplementation::Qdot;
    let row_bytes = if params.weights == DType::U8 {
        width / 2
    } else {
        params.weights.storage_bytes(width)
    };
    let row_scales = if params.weights == DType::U8 {
        width / 16
    } else {
        0
    };
    // The up projections may be taken by the delegate (an accelerator's
    // expert-up package, NVFP4 scales applied, no epilogue): then
    // only the activation is ours here, and the rest of the token goes
    // on unchanged.
    if up_taken {
        if activate {
            for (expert, values) in chosen.iter().zip(activations.chunks_exact_mut(inner)) {
                activate_expert(params, expert, values);
            }
        }
        return;
    }
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    let hidden_q =
        qdot.then(|| quantize_nvfp4(x, &mut quantized_scratch[..q8_scratch_elements(width)]));
    #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
    let _ = &quantized_scratch;
    for_rows(parallel, activations, 1, width, |first, rows| {
        let (mut at, mut rows) = (first, rows);
        while !rows.is_empty() {
            let (e, r) = (at / inner, at % inner);
            let (here, later) = rows.split_at_mut((inner - r).min(rows.len()));
            let expert = &chosen[e];
            let up = &expert.up[r * row_bytes..(r + here.len()) * row_bytes];
            let scales = if row_scales == 0 {
                &[][..]
            } else {
                &expert.up_block_scales[r * row_scales..(r + here.len()) * row_scales]
            };
            #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
            if let Some(hidden_q) = &hidden_q {
                project_qdot(up, scales, expert.up_scale, hidden_q, here);
            } else {
                project(params.weights, scalar, up, scales, expert.up_scale, x, here);
            }
            #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
            project(params.weights, scalar, up, scales, expert.up_scale, x, here);
            if activate {
                activate_expert(params, expert, here);
            }
            at += here.len();
            rows = later;
        }
    });
}

/// Replace `out` with the mixture of `chosen`'s down projections of their
/// (already activated) `activations`, each output row summing its picks in
/// `chosen` order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix_loaded_experts(
    plan: &CpuKernelPlan,
    params: MoeParams,
    chosen: &[ExpertWeights<'_>],
    activations: &[f32],
    mixture_weights: &[f32],
    out: &mut [f32],
    quantized_scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let (width, inner) = (params.hidden as usize, params.intermediate as usize);
    let k = chosen.len();
    let scalar = plan.implementation == CpuImplementation::Scalar;
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    let qdot = plan.implementation == CpuImplementation::Qdot;
    let column_bytes = if params.weights == DType::U8 {
        inner / 2
    } else {
        params.weights.storage_bytes(inner)
    };
    let column_scales = if params.weights == DType::U8 {
        inner / 16
    } else {
        0
    };
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    if qdot {
        let stride = q8_scratch_elements(inner);
        for (expert, values) in activations.chunks_exact(inner).enumerate() {
            quantize_nvfp4(
                values,
                &mut quantized_scratch[expert * stride..(expert + 1) * stride],
            );
        }
    }
    let quantized_scratch = &*quantized_scratch;
    #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
    let _ = quantized_scratch;
    // Then every output row sums its picks' down projections, in read
    // (ascending expert) order.
    for_rows(parallel, out, 1, k * inner, |first, rows| {
        let mut down = [0.0_f32; DOWN_ROWS];
        for (c, rows) in (first..).step_by(DOWN_ROWS).zip(rows.chunks_mut(DOWN_ROWS)) {
            rows.fill(0.0);
            let down = &mut down[..rows.len()];
            for (e, (&weight, expert)) in mixture_weights.iter().zip(chosen).enumerate() {
                let input = &activations[e * inner..(e + 1) * inner];
                let columns = &expert.down[c * column_bytes..(c + rows.len()) * column_bytes];
                let scales = if column_scales == 0 {
                    &[][..]
                } else {
                    &expert.down_block_scales[c * column_scales..(c + rows.len()) * column_scales]
                };
                #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
                if qdot {
                    let stride = q8_scratch_elements(inner);
                    let quantized =
                        borrowed_nvfp4(inner, &quantized_scratch[e * stride..(e + 1) * stride]);
                    project_qdot(columns, scales, expert.down_scale, &quantized, down);
                } else {
                    project(
                        params.weights,
                        scalar,
                        columns,
                        scales,
                        expert.down_scale,
                        input,
                        down,
                    );
                }
                #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
                project(
                    params.weights,
                    scalar,
                    columns,
                    scales,
                    expert.down_scale,
                    input,
                    down,
                );
                for (value, &d) in rows.iter_mut().zip(&*down) {
                    *value += weight
                        * if params.weights == DType::U8 {
                            d
                        } else {
                            d * expert.down_scale
                        };
                }
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::{e4m3_to_f32, f32_to_e4m3_rne};
    use crate::plan_cpu_kernel;
    use alloc::vec;
    use alloc::vec::Vec;
    use axnn::lowering::ResourceKey;
    use axnn::{Shape, TensorDesc};

    const KEY: ResourceKey = ResourceKey {
        group: 5,
        binding: 1,
    };
    const EXPERTS: usize = 6;
    const HIDDEN: usize = 16;
    const INNER: usize = 8;

    fn desc(dims: &[usize]) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), DType::F32)
    }

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 23 + seed * 7) % 31) as f32 - 15.0) / 10.0)
            .collect()
    }

    fn params(top_k: u32, normalize: bool) -> MoeParams {
        MoeParams::new(
            KEY,
            EXPERTS as u32,
            top_k,
            HIDDEN as u32,
            INNER as u32,
            DType::B8,
            ExpertActivation::SquaredRelu,
            normalize,
            2.5,
        )
        .unwrap()
    }

    fn plan(params: MoeParams, tokens: usize, capabilities: CpuCapabilities) -> CpuKernelPlan {
        let request = KernelRequest::moe_experts(
            &desc(&[tokens, HIDDEN]),
            &desc(&[tokens, EXPERTS]),
            &desc(&[EXPERTS]),
            params,
        )
        .unwrap();
        plan_cpu_kernel(&request, capabilities).unwrap()
    }

    /// Packed up and down projections with their scales.
    type Packed = (Vec<u8>, Vec<u8>, f32, f32);
    /// Scaled F64 up and down projections.
    type Decoded = (Vec<f64>, Vec<f64>);

    /// FP8 experts with distinct scales, and the F64 decode of each.
    fn experts() -> (Vec<Packed>, Vec<Decoded>) {
        let mut packed = Vec::new();
        let mut decoded = Vec::new();
        for e in 0..EXPERTS {
            let up: Vec<u8> = values(INNER * HIDDEN, 2 * e + 1)
                .into_iter()
                .map(f32_to_e4m3_rne)
                .collect();
            let down: Vec<u8> = values(HIDDEN * INNER, 2 * e + 2)
                .into_iter()
                .map(f32_to_e4m3_rne)
                .collect();
            let (up_scale, down_scale) = (0.5 + e as f32 / 8.0, 0.25 * (e + 1) as f32);
            let decode = |bytes: &[u8], scale: f32| -> Vec<f64> {
                bytes
                    .iter()
                    .map(|&b| f64::from(e4m3_to_f32(b)) * f64::from(scale))
                    .collect()
            };
            decoded.push((decode(&up, up_scale), decode(&down, down_scale)));
            packed.push((up, down, up_scale, down_scale));
        }
        (packed, decoded)
    }

    fn reference(
        params: MoeParams,
        decoded: &[(Vec<f64>, Vec<f64>)],
        x: &[f32],
        scores: &[f32],
        bias: &[f32],
    ) -> Vec<f64> {
        let k = params.top_k as usize;
        let mut ranked: Vec<usize> = (0..EXPERTS).collect();
        let choice = |e: usize| scores[e] + bias[e];
        ranked.sort_by(|&a, &b| choice(b).total_cmp(&choice(a)).then(a.cmp(&b)));
        let picked = &ranked[..k];
        let total: f64 = picked.iter().map(|&e| f64::from(scores[e])).sum::<f64>() + 1e-20;
        let mut out = vec![0.0; HIDDEN];
        for &e in picked {
            let mut weight = f64::from(scores[e]);
            if params.normalize {
                weight /= total;
            }
            weight *= 2.5;
            let (up, down) = &decoded[e];
            let act: Vec<f64> = (0..INNER)
                .map(|r| {
                    let v: f64 = (0..HIDDEN)
                        .map(|i| up[r * HIDDEN + i] * f64::from(x[i]))
                        .sum();
                    v.max(0.0).powi(2)
                })
                .collect();
            for c in 0..HIDDEN {
                out[c] += weight
                    * (0..INNER)
                        .map(|r| down[c * INNER + r] * act[r])
                        .sum::<f64>();
            }
        }
        out
    }

    #[test]
    fn experts_match_a_direct_routing_and_sum_in_both_implementations() {
        let (packed, decoded) = experts();
        let table: Vec<ExpertWeights<'_>> = packed
            .iter()
            .map(|(up, down, up_scale, down_scale)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *up_scale,
                down_scale: *down_scale,
            })
            .collect();
        let tokens = 2;
        let x = values(tokens * HIDDEN, 20);
        // Sigmoid-like scores; the bias promotes expert 5 into selection.
        let scores: Vec<f32> = values(tokens * EXPERTS, 21)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        let bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8];
        for (top_k, normalize) in [(2, true), (3, false), (6, true)] {
            let params = params(top_k, normalize);
            let store = ExpertStoreRef::new(&table, params).unwrap();
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
                let plan = plan(params, tokens, capabilities);
                let mut out = vec![f32::NAN; tokens * HIDDEN];
                let mut scratch = vec![0.0; moe_scratch_elements(params)];
                invoke_moe_experts(
                    &plan,
                    store,
                    &x,
                    &scores,
                    &bias,
                    &mut out,
                    &mut scratch,
                    &crate::Serial,
                )
                .unwrap();
                for t in 0..tokens {
                    let expected = reference(
                        params,
                        &decoded,
                        &x[t * HIDDEN..(t + 1) * HIDDEN],
                        &scores[t * EXPERTS..(t + 1) * EXPERTS],
                        &bias,
                    );
                    for (actual, expected) in
                        out[t * HIDDEN..(t + 1) * HIDDEN].iter().zip(&expected)
                    {
                        assert!(
                            (f64::from(*actual) - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                            "k {top_k} {:?}: {actual} vs {expected}",
                            plan.implementation
                        );
                    }
                }
            }
        }
    }

    /// Serves every expert from `packed`, at most `max_picks` a load, and
    /// records every hint and load (`true` for a load) in order.
    struct TableSource<'a> {
        packed: &'a [Packed],
        max_picks: Option<usize>,
        calls: core::cell::RefCell<Vec<(bool, Vec<u32>)>>,
    }

    // SAFETY: tests only; nothing loaded ever changes.
    unsafe impl Sync for TableSource<'_> {}
    // SAFETY: every slice handed out borrows `packed`, which outlives the
    // source and never changes.
    unsafe impl ExpertSource for TableSource<'_> {
        fn experts(&self) -> usize {
            EXPERTS
        }
        fn load<'s>(
            &'s self,
            picks: &[u32],
            _: &[f32],
            out: &mut [ExpertWeights<'s>],
        ) -> Result<(), CpuError> {
            assert!(picks.len() <= self.max_picks.unwrap_or(usize::MAX));
            self.calls.borrow_mut().push((true, picks.to_vec()));
            for (weights, &e) in out.iter_mut().zip(picks) {
                let (up, down, up_scale, down_scale) = &self.packed[e as usize];
                *weights = ExpertWeights {
                    up,
                    down,
                    up_block_scales: &[],
                    down_block_scales: &[],
                    up_scale: *up_scale,
                    down_scale: *down_scale,
                };
            }
            Ok(())
        }
        fn will_load(&self, picks: &[u32]) {
            self.calls.borrow_mut().push((false, picks.to_vec()));
        }
        fn max_picks(&self) -> Option<usize> {
            self.max_picks
        }
    }

    /// What a device does with a wave, computed as the native kernel does
    /// it: every route through its expert, added to its token's output in
    /// route order.
    fn device_wave(
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        routes: &[MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        output: &mut [f32],
    ) {
        let width = params.hidden as usize;
        let mut contribution = vec![0.0; width];
        let mut scratch = vec![0.0; moe_scratch_elements(params)];
        let mut pick = 0;
        for (index, route) in routes.iter().enumerate() {
            if index > 0 && route.expert() != routes[index - 1].expert() {
                pick += 1;
            }
            let token = route.token();
            invoke_loaded_experts(
                plan,
                params,
                &hidden[token * width..(token + 1) * width],
                &experts[pick..pick + 1],
                &[route.mixture()],
                &mut contribution,
                &mut scratch,
                &crate::Serial,
                false,
            )
            .unwrap();
            for (out, &value) in output[token * width..(token + 1) * width]
                .iter_mut()
                .zip(&contribution)
            {
                *out += value;
            }
        }
    }

    /// An up-projection delegate that computes exactly what the kernel would
    /// (the crate's own `project`, all rows at once) so the token it takes
    /// must come out identical; a delegate that declines changes nothing.
    struct UpDelegate {
        take: bool,
        taken: core::cell::Cell<usize>,
    }

    impl MoeDelegate<CpuError> for UpDelegate {
        fn token(
            &mut self,
            _: &CpuKernelPlan,
            _: MoeParams,
            _: &[f32],
            _: &[ExpertWeights<'_>],
            _: &[f32],
            _: &mut [f32],
            _: &mut [f32],
        ) -> Result<bool, CpuError> {
            Ok(false)
        }

        fn up(
            &mut self,
            plan: &CpuKernelPlan,
            params: MoeParams,
            hidden: &[f32],
            experts: &[ExpertWeights<'_>],
            activations: &mut [f32],
        ) -> Result<bool, CpuError> {
            if !self.take {
                return Ok(false);
            }
            let inner = params.intermediate as usize;
            let scalar = plan.implementation == CpuImplementation::Scalar;
            for (expert, out) in experts.iter().zip(activations.chunks_exact_mut(inner)) {
                project(
                    params.weights,
                    scalar,
                    expert.up,
                    expert.up_block_scales,
                    expert.up_scale,
                    hidden,
                    out,
                );
            }
            self.taken.set(self.taken.get() + 1);
            Ok(true)
        }
    }

    #[test]
    fn a_delegate_taking_only_the_up_projections_matches_the_cpu_path() {
        let (packed, _) = experts();
        let resident: Vec<ExpertWeights<'_>> = packed
            .iter()
            .map(|(up, down, up_scale, down_scale)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *up_scale,
                down_scale: *down_scale,
            })
            .collect();
        let tokens = 2;
        let x = values(tokens * HIDDEN, 20);
        let scores: Vec<f32> = values(tokens * EXPERTS, 21)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        let bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8];
        let params = params(3, true);
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
            let plan = plan(params, tokens, capabilities);
            let run = |delegate: &mut dyn MoeDelegate<CpuError>| {
                let mut out = vec![f32::NAN; tokens * HIDDEN];
                let mut scratch = vec![0.0; moe_scratch_elements(params)];
                invoke_moe_experts_delegated(
                    &plan,
                    ExpertStoreRef::new(&resident, params).unwrap(),
                    &x,
                    &scores,
                    &bias,
                    &mut out,
                    &mut scratch,
                    &crate::Serial,
                    delegate,
                )
                .unwrap();
                out
            };
            let mut declining = UpDelegate {
                take: false,
                taken: Default::default(),
            };
            let cpu = run(&mut declining);
            assert_eq!(declining.taken.get(), 0);
            let mut taking = UpDelegate {
                take: true,
                taken: Default::default(),
            };
            let delegated = run(&mut taking);
            assert_eq!(taking.taken.get(), tokens, "{:?}", plan.implementation);
            assert_eq!(delegated, cpu, "{:?}", plan.implementation);
        }
    }

    #[test]
    fn nvfp4_experts_match_their_decoded_weights() {
        const E2M1: [f64; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let (experts, hidden, inner, top_k) = (4, 32, 16, 2);
        let mut state = 0x9e37_79b9_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        // Packed codes, block scales between 1/4 and 2, and the tensor
        // scale; with the F64 matrix they decode to.
        let mut matrix = |rows: usize, cols: usize, tensor: f32| {
            let packed: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
            let scales: Vec<u8> = (0..rows * cols / 16)
                .map(|_| 0x28 + (next() % 0x18) as u8)
                .collect();
            let decoded: Vec<f64> = (0..rows * cols)
                .map(|i| {
                    let code = (packed[i / 2] >> (4 * (i % 2))) & 15;
                    E2M1[code as usize] * f64::from(e4m3_to_f32(scales[i / 16])) * f64::from(tensor)
                })
                .collect();
            (packed, scales, tensor, decoded)
        };
        // Small tensor scales, as NVFP4 checkpoints have: applying one twice
        // shrinks an expert's output a thousandfold.
        let weights: Vec<_> = (0..experts)
            .map(|e| {
                (
                    matrix(inner, hidden, 0.002 * (e + 1) as f32),
                    matrix(hidden, inner, 0.003 * (e + 2) as f32),
                )
            })
            .collect();
        let table: Vec<ExpertWeights<'_>> = weights
            .iter()
            .map(|(up, down)| ExpertWeights {
                up: &up.0,
                down: &down.0,
                up_block_scales: &up.1,
                down_block_scales: &down.1,
                up_scale: up.2,
                down_scale: down.2,
            })
            .collect();
        let params = MoeParams::new(
            KEY,
            experts as u32,
            top_k as u32,
            hidden as u32,
            inner as u32,
            DType::U8,
            ExpertActivation::SquaredRelu,
            true,
            2.5,
        )
        .unwrap();
        let store = ExpertStoreRef::new(&table, params).unwrap();
        let x: Vec<f32> = (0..hidden)
            .map(|i| ((i * 37 % 29) as f32 - 14.0) / 7.0)
            .collect();
        let scores = [0.3_f32, 0.8, 0.1, 0.6];
        let bias = [0.0_f32; 4];
        // Experts 1 and 3, normalized and scaled.
        let mut expected = vec![0.0_f64; hidden];
        for e in [1, 3] {
            let weight = f64::from(scores[e]) / f64::from(scores[1] + scores[3]) * 2.5;
            let (up, down) = (&weights[e].0.3, &weights[e].1.3);
            let act: Vec<f64> = (0..inner)
                .map(|r| {
                    let v: f64 = (0..hidden)
                        .map(|i| up[r * hidden + i] * f64::from(x[i]))
                        .sum();
                    v.max(0.0).powi(2)
                })
                .collect();
            for c in 0..hidden {
                let d: f64 = (0..inner).map(|r| down[c * inner + r] * act[r]).sum();
                expected[c] += weight * d;
            }
        }
        let peak = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(peak > 1e-3, "the experts contribute");
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
            let request = KernelRequest::moe_experts(
                &desc(&[1, hidden]),
                &desc(&[1, experts]),
                &desc(&[experts]),
                params,
            )
            .unwrap();
            let plan = plan_cpu_kernel(&request, capabilities).unwrap();
            let mut out = vec![f32::NAN; hidden];
            let mut scratch = vec![0.0; moe_scratch_elements(params)];
            invoke_moe_experts(
                &plan,
                store,
                &x,
                &scores,
                &bias,
                &mut out,
                &mut scratch,
                &crate::Serial,
            )
            .unwrap();
            // Resume an independently completed up phase against the same F64
            // reference. This catches omission or duplication of either NVFP4
            // scale across the new suspension boundary.
            struct DeferredWave(bool);
            impl ResumableMoeDelegate<CpuError> for DeferredWave {
                fn experts(
                    &mut self,
                    _: usize,
                    plan: &CpuKernelPlan,
                    params: MoeParams,
                    hidden: &[f32],
                    routes: &[MoeBatchRow],
                    experts: &[ExpertWeights<'_>],
                    output: &mut [f32],
                ) -> Result<MoeProgress, CpuError> {
                    if !core::mem::replace(&mut self.0, true) {
                        return Ok(MoeProgress::Pending);
                    }
                    device_wave(plan, params, hidden, routes, experts, output);
                    Ok(MoeProgress::Complete)
                }
            }
            let mut resumed = vec![f32::NAN; hidden];
            let mut scratch = vec![0.0; moe_prepared_scratch_elements(params, 1).unwrap()];
            let mut run =
                MoeInvocation::new(&plan, store, &x, &scores, &bias, &resumed, &scratch).unwrap();
            let mut delegate = DeferredWave(false);
            for expected_progress in [MoeProgress::Pending, MoeProgress::Complete] {
                assert_eq!(
                    run.advance(
                        &x,
                        &scores,
                        &bias,
                        &mut resumed,
                        &mut scratch,
                        &crate::Serial,
                        &mut delegate,
                    )
                    .unwrap(),
                    expected_progress,
                );
            }
            // 8-bit activations cost the W4A8 tier its last few digits; a
            // scale applied twice would cost every digit.
            let tolerance = if plan.implementation == CpuImplementation::Qdot {
                1e-2
            } else {
                1e-4
            };
            for (actual, expected) in out.iter().chain(&resumed).zip(expected.iter().cycle()) {
                assert!(
                    (f64::from(*actual) - expected).abs() <= tolerance * peak,
                    "{:?}: {actual} vs {expected}",
                    plan.implementation
                );
            }
            // The same quantized/tensor scales must survive grouping and
            // suspension when multiple tokens share the expert residency.
            let tokens = 3;
            let request = KernelRequest::moe_experts(
                &desc(&[tokens, hidden]),
                &desc(&[tokens, experts]),
                &desc(&[experts]),
                params,
            )
            .unwrap();
            let plan = plan_cpu_kernel(&request, capabilities).unwrap();
            let batch_x = x.repeat(tokens);
            let batch_scores = scores.repeat(tokens);
            let mut scratch = vec![0.0; moe_prepared_scratch_elements(params, tokens).unwrap()];
            for take in [false, true] {
                struct WaveNvfp4 {
                    take: bool,
                    pending: bool,
                }
                impl ResumableMoeDelegate<CpuError> for WaveNvfp4 {
                    fn experts(
                        &mut self,
                        _: usize,
                        plan: &CpuKernelPlan,
                        params: MoeParams,
                        hidden: &[f32],
                        routes: &[MoeBatchRow],
                        experts: &[ExpertWeights<'_>],
                        output: &mut [f32],
                    ) -> Result<MoeProgress, CpuError> {
                        if !self.take {
                            return Ok(MoeProgress::Unsupported);
                        }
                        self.pending = !self.pending;
                        if self.pending {
                            return Ok(MoeProgress::Pending);
                        }
                        device_wave(plan, params, hidden, routes, experts, output);
                        Ok(MoeProgress::Complete)
                    }
                }
                let mut output = vec![f32::NAN; batch_x.len()];
                let mut delegate = WaveNvfp4 {
                    take,
                    pending: false,
                };
                let mut run = MoeInvocation::new(
                    &plan,
                    store,
                    &batch_x,
                    &batch_scores,
                    &bias,
                    &output,
                    &scratch,
                )
                .unwrap();
                let mut done = false;
                for _ in 0..tokens * top_k + 1 {
                    if run
                        .advance(
                            &batch_x,
                            &batch_scores,
                            &bias,
                            &mut output,
                            &mut scratch,
                            &crate::Serial,
                            &mut delegate,
                        )
                        .unwrap()
                        == MoeProgress::Complete
                    {
                        done = true;
                        break;
                    }
                }
                assert!(done);
                for (&actual, &expected) in output.iter().zip(expected.iter().cycle()) {
                    assert!(
                        (f64::from(actual) - expected).abs() <= tolerance * peak,
                        "grouped {:?} take={take}: {actual} vs {expected}",
                        plan.implementation
                    );
                }
            }
        }
    }

    /// Waves: a source that holds two experts a load splits the routes into
    /// waves of distinct experts, each loaded once, in route order, after
    /// every hint. A device that goes pending once per wave and one that
    /// declines every wave both give the synchronous kernel's exact bits.
    #[test]
    fn waves_keep_the_synchronous_bits_whether_a_device_takes_them_or_not() {
        static TICKS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
        fn tick() -> u64 {
            TICKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
        }
        struct Device {
            take: bool,
            pending: bool,
            waves: Vec<usize>,
        }
        impl ResumableMoeDelegate<CpuError> for Device {
            fn wave_capacity(&self, _: MoeParams, _: usize, _: &[MoeBatchRow]) -> usize {
                // The source's two-expert residency bound wins even if the
                // device advertises an unbounded capacity.
                usize::MAX
            }
            fn experts(
                &mut self,
                wave: usize,
                plan: &CpuKernelPlan,
                params: MoeParams,
                hidden: &[f32],
                routes: &[MoeBatchRow],
                experts: &[ExpertWeights<'_>],
                output: &mut [f32],
            ) -> Result<MoeProgress, CpuError> {
                if !self.take {
                    return Ok(MoeProgress::Unsupported);
                }
                self.pending = !self.pending;
                if self.pending {
                    self.waves.push(wave);
                    return Ok(MoeProgress::Pending);
                }
                device_wave(plan, params, hidden, routes, experts, output);
                Ok(MoeProgress::Complete)
            }
        }
        let (packed, _) = experts();
        let table: Vec<ExpertWeights<'_>> = packed
            .iter()
            .map(|(up, down, up_scale, down_scale)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *up_scale,
                down_scale: *down_scale,
            })
            .collect();
        let tokens = 4;
        let params = params(3, true);
        let x = values(tokens * HIDDEN, 41);
        let scores: Vec<f32> = values(tokens * EXPERTS, 42)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        let bias = [0.0, 0.3, 0.0, 0.0, 0.0, 0.8];
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
            let plan = plan(params, tokens, capabilities);
            let mut expected = vec![0.0; tokens * HIDDEN];
            let mut native_scratch = vec![0.0; moe_scratch_elements(params)];
            invoke_moe_experts(
                &plan,
                ExpertStoreRef::new(&table, params).unwrap(),
                &x,
                &scores,
                &bias,
                &mut expected,
                &mut native_scratch,
                &crate::Serial,
            )
            .unwrap();
            for take in [false, true] {
                let source = TableSource {
                    packed: &packed,
                    max_picks: Some(2),
                    calls: Default::default(),
                };
                let store = ExpertStoreRef::from_source(&source, params).unwrap();
                let mut scratch = vec![0.0; moe_prepared_scratch_elements(params, tokens).unwrap()];
                let mut output = vec![f32::NAN; tokens * HIDDEN];
                let mut run =
                    MoeInvocation::new(&plan, store, &x, &scores, &bias, &output, &scratch)
                        .unwrap()
                        .with_clock(tick);
                let mut device = Device {
                    take,
                    pending: false,
                    waves: Vec::new(),
                };
                let mut pendings = 0;
                loop {
                    match run
                        .advance(
                            &x,
                            &scores,
                            &bias,
                            &mut output,
                            &mut scratch,
                            &crate::Serial,
                            &mut device,
                        )
                        .unwrap()
                    {
                        MoeProgress::Complete => break,
                        MoeProgress::Pending => pendings += 1,
                        MoeProgress::Unsupported => panic!("an invocation never declines"),
                    }
                }
                assert_eq!(
                    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "take {take} {:?}",
                    plan.implementation
                );
                let calls = source.calls.borrow();
                let hints: Vec<u32> = calls
                    .iter()
                    .take_while(|(load, _)| !load)
                    .flat_map(|(_, picks)| picks.clone())
                    .collect();
                let loads: Vec<Vec<u32>> = calls
                    .iter()
                    .filter(|(load, _)| *load)
                    .map(|(_, picks)| picks.clone())
                    .collect();
                assert!(hints.windows(2).all(|w| w[0] < w[1]), "{hints:?}");
                assert_eq!(loads.concat(), hints, "each expert loaded once, in order");
                assert!(loads.len() > 1 && loads.iter().all(|picks| picks.len() <= 2));
                let phases = run.phases().ns;
                assert!(phases[0] > 0 && phases[1] > 0, "route and load charged");
                if take {
                    assert_eq!(pendings, loads.len());
                    assert!(device.waves.windows(2).all(|w| w[0] < w[1]));
                    assert!(phases[2] > 0 && phases[3] == 0, "{phases:?}");
                } else {
                    assert_eq!(pendings, 0);
                    assert!(phases[3] > 0, "the native waves charged");
                }
            }
        }
    }

    /// Device sizing changes wave boundaries, never route order or bits.
    /// Include more than the metadata ceiling and clamp a hostile answer;
    /// no capacity callback is allowed while a wave is still Pending.
    #[test]
    fn wave_capacity_is_bounded_and_frozen_until_retirement() {
        use core::cell::Cell;
        struct Device {
            capacity: usize,
            sizing: Cell<usize>,
            pending: Option<(usize, usize, usize)>,
            sizes: Vec<usize>,
        }
        impl ResumableMoeDelegate<CpuError> for Device {
            fn wave_capacity(&self, _: MoeParams, _: usize, _: &[MoeBatchRow]) -> usize {
                assert!(self.pending.is_none());
                self.sizing.set(self.sizing.get() + 1);
                self.capacity
            }
            fn experts(
                &mut self,
                wave: usize,
                plan: &CpuKernelPlan,
                params: MoeParams,
                hidden: &[f32],
                routes: &[MoeBatchRow],
                experts: &[ExpertWeights<'_>],
                output: &mut [f32],
            ) -> Result<MoeProgress, CpuError> {
                let key = (wave, routes.len(), experts.len());
                if let Some(previous) = self.pending.take() {
                    assert_eq!(previous, key);
                    device_wave(plan, params, hidden, routes, experts, output);
                    Ok(MoeProgress::Complete)
                } else {
                    self.pending = Some(key);
                    self.sizes.push(experts.len());
                    Ok(MoeProgress::Pending)
                }
            }
        }
        let count = MAX_WAVE_EXPERTS + 3;
        let mut p = params(1, true);
        p.experts = count as u32;
        let (packed, _) = experts();
        let table: Vec<_> = (0..count)
            .map(|i| {
                let (up, down, a, b) = &packed[i % packed.len()];
                ExpertWeights {
                    up,
                    down,
                    up_block_scales: &[],
                    down_block_scales: &[],
                    up_scale: *a,
                    down_scale: *b,
                }
            })
            .collect();
        let request = KernelRequest::moe_experts(
            &desc(&[count, HIDDEN]),
            &desc(&[count, count]),
            &desc(&[count]),
            p,
        )
        .unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let x = values(count * HIDDEN, 41);
        let mut scores = vec![0.0; count * count];
        for i in 0..count {
            scores[i * count + i] = 1.0;
        }
        let bias = vec![0.0; count];
        let store = ExpertStoreRef::new(&table, p).unwrap();
        let mut expected = vec![0.0; x.len()];
        invoke_moe_experts(
            &plan,
            store,
            &x,
            &scores,
            &bias,
            &mut expected,
            &mut vec![0.0; moe_scratch_elements(p)],
            &crate::Serial,
        )
        .unwrap();
        for capacity in [0, 3, DEFAULT_WAVE_EXPERTS, 64, MAX_WAVE_EXPERTS, usize::MAX] {
            let mut device = Device {
                capacity,
                sizing: Cell::new(0),
                pending: None,
                sizes: Vec::new(),
            };
            let mut output = vec![f32::NAN; x.len()];
            let mut scratch = vec![0.0; moe_prepared_scratch_elements(p, count).unwrap()];
            let mut run =
                MoeInvocation::new(&plan, store, &x, &scores, &bias, &output, &scratch).unwrap();
            let mut done = false;
            for _ in 0..=count {
                if run
                    .advance(
                        &x,
                        &scores,
                        &bias,
                        &mut output,
                        &mut scratch,
                        &crate::Serial,
                        &mut device,
                    )
                    .unwrap()
                    == MoeProgress::Complete
                {
                    done = true;
                    break;
                }
            }
            assert!(done);
            let limit = capacity.clamp(1, MAX_WAVE_EXPERTS);
            assert_eq!(device.sizes.len(), count.div_ceil(limit));
            assert_eq!(device.sizing.get(), device.sizes.len());
            assert_eq!(device.sizes.iter().sum::<usize>(), count);
            assert!(device.sizes.iter().all(|&n| n <= limit));
            assert_eq!(run.phases().native_routes, 0);
            assert_eq!(run.phases().delegated_routes, count as u64);
            assert_eq!(
                output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    /// Scratch is sized and checked before anything is written, and a wave
    /// that went pending may not decline: the CPU never replays a wave a
    /// device may still be running.
    #[test]
    fn wave_scratch_and_pending_refusal_are_checked() {
        let (packed, _) = experts();
        let table: Vec<_> = packed
            .iter()
            .map(|(up, down, a, b)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *a,
                down_scale: *b,
            })
            .collect();
        let p = params(3, true);
        assert_eq!(moe_prepared_scratch_elements(p, 0), None);
        assert_eq!(moe_prepared_scratch_elements(p, usize::MAX), None);
        assert_eq!(
            moe_prepared_scratch_elements(p, 2),
            Some(2 * 3 * 3 + HIDDEN + moe_scratch_elements(p))
        );
        let plan = plan(p, 2, CpuCapabilities::SCALAR);
        assert_eq!(
            plan.temporary_bytes,
            moe_prepared_scratch_elements(p, 2).unwrap() * size_of::<f32>()
        );
        let store = ExpertStoreRef::new(&table, p).unwrap();
        let x = values(2 * HIDDEN, 17);
        let scores = vec![0.5; 2 * EXPERTS];
        let bias = [0.0; EXPERTS];
        let mut out = vec![77.0; x.len()];
        let mut scratch = vec![0.0; moe_prepared_scratch_elements(p, 2).unwrap()];
        assert!(
            MoeInvocation::new(
                &plan,
                store,
                &x,
                &scores,
                &bias,
                &out,
                &scratch[..scratch.len() - 1]
            )
            .is_err()
        );
        assert!(out.iter().all(|&v| v == 77.0));
        struct DeclineAfterWait(bool);
        impl ResumableMoeDelegate<CpuError> for DeclineAfterWait {
            fn experts(
                &mut self,
                _: usize,
                _: &CpuKernelPlan,
                _: MoeParams,
                _: &[f32],
                _: &[MoeBatchRow],
                _: &[ExpertWeights<'_>],
                _: &mut [f32],
            ) -> Result<MoeProgress, CpuError> {
                Ok(if core::mem::replace(&mut self.0, true) {
                    MoeProgress::Unsupported
                } else {
                    MoeProgress::Pending
                })
            }
        }
        let mut delegate = DeclineAfterWait(false);
        let mut run = MoeInvocation::new(&plan, store, &x, &scores, &bias, &out, &scratch).unwrap();
        assert_eq!(
            run.advance(
                &x,
                &scores,
                &bias,
                &mut out,
                &mut scratch,
                &crate::Serial,
                &mut delegate
            )
            .unwrap(),
            MoeProgress::Pending
        );
        for _ in 0..2 {
            assert!(
                run.advance(
                    &x,
                    &scores,
                    &bias,
                    &mut out,
                    &mut scratch,
                    &crate::Serial,
                    &mut delegate
                )
                .is_err()
            );
            assert!(
                out.iter().all(|&v| v == 0.0),
                "no CPU replay after device admission"
            );
        }
    }

    #[test]
    fn the_bias_selects_but_never_weights() {
        let (packed, _) = experts();
        let table: Vec<ExpertWeights<'_>> = packed
            .iter()
            .map(|(up, down, a, b)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *a,
                down_scale: *b,
            })
            .collect();
        let params = params(1, false);
        let store = ExpertStoreRef::new(&table, params).unwrap();
        let plan = plan(params, 1, CpuCapabilities::SCALAR);
        let x = values(HIDDEN, 3);
        let run = |scores: &[f32], bias: &[f32]| {
            let mut out = vec![0.0; HIDDEN];
            invoke_moe_experts(
                &plan,
                store,
                &x,
                scores,
                bias,
                &mut out,
                &mut [0.0; INNER],
                &crate::Serial,
            )
            .unwrap();
            out
        };
        // Expert 2 wins on score alone; a bias makes expert 4 win instead,
        // weighted by its own unbiased score, not the biased one.
        let scores = [0.1, 0.1, 0.9, 0.1, 0.2, 0.1];
        let biased = run(&scores, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let doubled = run(
            &[0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        for (a, b) in biased.iter().zip(&doubled) {
            assert!((2.0 * a - b).abs() <= 1e-5 * b.abs().max(1.0));
        }
        assert_ne!(biased, run(&scores, &[0.0; 6]));
    }

    #[test]
    fn stores_and_parameters_are_checked() {
        let (packed, _) = experts();
        let table: Vec<ExpertWeights<'_>> = packed
            .iter()
            .map(|(up, down, a, b)| ExpertWeights {
                up,
                down,
                up_block_scales: &[],
                down_block_scales: &[],
                up_scale: *a,
                down_scale: *b,
            })
            .collect();
        assert!(ExpertStoreRef::new(&table[..5], params(2, true)).is_err());
        let mut broken = table.clone();
        broken[3].down = &broken[3].down[1..];
        assert!(ExpertStoreRef::new(&broken, params(2, true)).is_err());
        let mut broken = table.clone();
        broken[0].up_scale = f32::NAN;
        assert!(ExpertStoreRef::new(&broken, params(2, true)).is_err());
        let make = |experts: u32, top_k: u32, weights: DType, scale: f32| {
            MoeParams::new(
                KEY,
                experts,
                top_k,
                4,
                4,
                weights,
                ExpertActivation::SquaredRelu,
                true,
                scale,
            )
        };
        assert!(make(128, 6, DType::B8, 2.5).is_some());
        assert!(make(128, 6, DType::B16, 1.0).is_some());
        assert!(make(128, 17, DType::B8, 1.0).is_none(), "at most 16 picks");
        assert!(make(4, 5, DType::B8, 1.0).is_none(), "picks exceed experts");
        assert!(make(4, 0, DType::B8, 1.0).is_none());
        assert!(make(4, 2, DType::F32, 1.0).is_none());
        assert!(make(4, 2, DType::B8, 0.0).is_none());
    }
}
