//! Checked invocation of the existing packed-dot and normalization kernels.
//! No allocation, unpacked weight mirror, host probe, or implicit quantization.
use crate::fork_join::{ForkJoin, for_row_ranges, for_rows};
use crate::{CpuCapabilities, CpuError, CpuImplementation, CpuKernelPlan, arch};
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, EpilogueActivation, KernelAccuracy,
    KernelCandidate, KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelRequest,
    KernelShapePolicy, SemanticOp,
};
use axnn::{DType, RopeMode, TensorDesc};

/// Apply a fused epilogue activation (#107) to one F32 value, through the
/// same `axmath` kernels the standalone `SILU` and `sigmoid` nodes use, so a
/// fused result matches an unfused one bit for bit.
pub(crate) fn apply_epilogue(activation: EpilogueActivation, value: f32) -> f32 {
    match activation {
        EpilogueActivation::None => value,
        EpilogueActivation::Silu => axmath::silu_f32(value),
        EpilogueActivation::Sigmoid => axmath::sigmoid_f32(value),
    }
}

pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &axnn::lowering::KernelRequest,
    visitor: &mut V,
) -> bool {
    if request.semantic == SemanticOp::Cast
        && matches!(
            (request.input_dtypes[0], request.output_dtype),
            (Some(DType::F32), DType::Q4 | DType::Q6 | DType::Q8)
                | (Some(DType::Q4 | DType::Q6 | DType::Q8), DType::F32)
        )
    {
        let Some(input) = request.input(0) else {
            return true;
        };
        if input.layout.offset() != 0 {
            return true;
        }
        let mut constraints = KernelConstraints::elementwise(request.output_dtype);
        constraints.input_dtypes[0] = DTypeSet::one(input.dtype);
        constraints.shape_policy = KernelShapePolicy::Typed;
        visitor.visit(KernelCandidate {
            semantic: SemanticOp::Cast,
            implementation: CpuImplementation::Scalar,
            required_capabilities: CapabilityMask::NONE,
            constraints,
            accuracy: KernelAccuracy::Faithful,
            cost: KernelCost::new(0, 12),
        });
        return true;
    }
    if request.semantic == SemanticOp::MatMulNvfp4Transposed {
        if request.input_dtypes != [Some(DType::F32), Some(DType::U8), Some(DType::B8)]
            || request.output_dtype != DType::F32
            || request
                .inputs()
                .iter()
                .flatten()
                .any(|d| d.layout.offset() != 0)
        {
            return true;
        }
        let mut constraints = KernelConstraints::elementwise(DType::F32);
        constraints.shape_policy = KernelShapePolicy::Typed;
        constraints.max_rank = 2;
        constraints.input_dtypes = [
            DTypeSet::one(DType::F32),
            DTypeSet::one(DType::U8),
            DTypeSet::one(DType::B8),
        ];
        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation: CpuImplementation::PortableSimd,
            required_capabilities: CpuCapabilities::PORTABLE_SIMD.mask(),
            constraints,
            accuracy: KernelAccuracy::Faithful,
            cost: KernelCost::new(32, 1),
        });
        #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation: CpuImplementation::Qdot,
            required_capabilities: CpuCapabilities::QDOT.mask(),
            constraints,
            accuracy: KernelAccuracy::Approximate,
            cost: KernelCost::new(64, 0).with_temporary_bytes({
                // Every token's quantized row at once, so several tokens
                // share each pass over the weights.
                let x = request.input(0).unwrap().shape;
                nvfp4_q8_scratch_elements(x.dim(1)) * x.dim(0).max(1) * 4
            }),
        });
        return true;
    }
    let packed = request.semantic == SemanticOp::MatMulTransposed;
    if !packed
        && !matches!(
            request.semantic,
            SemanticOp::Softmax | SemanticOp::RmsNorm | SemanticOp::LayerNorm | SemanticOp::Rope(_)
        )
    {
        return false;
    }
    if request
        .inputs()
        .iter()
        .flatten()
        .any(|d| d.layout.offset() != 0)
    {
        return true;
    }
    let mut constraints = KernelConstraints {
        shape_policy: KernelShapePolicy::Typed,
        ..KernelConstraints::elementwise(DType::F32)
    };
    let float_weights = packed
        && request.input_dtypes[0] == Some(DType::F32)
        && matches!(
            request.input_dtypes[1],
            Some(DType::B8 | DType::B16 | DType::F32)
        );
    if float_weights {
        // F32 activations times FP8, BF16 or F32 weights stored [out, in],
        // as a checkpoint keeps them: `[tokens, in] x [out, in]^T`.
        constraints.max_rank = 2;
        constraints.input_dtypes[1] = DTypeSet::one(DType::B8)
            .union(DTypeSet::one(DType::B16))
            .union(DTypeSet::one(DType::F32));
    } else if packed {
        if !matches!(
            request.input_dtypes[0],
            Some(DType::Q4 | DType::Q6 | DType::Q8)
        ) || request.input_dtypes[1] != Some(DType::Q8)
        {
            return true;
        }
        constraints.max_rank = 2;
        constraints.input_dtypes[0] = DTypeSet::one(DType::Q4)
            .union(DTypeSet::one(DType::Q6))
            .union(DTypeSet::one(DType::Q8));
        constraints.input_dtypes[1] = DTypeSet::one(DType::Q8);
    } else if let KernelParams::Axis(axis) = request.params() {
        if request.shape.rank() != axis as usize + 1 {
            return true;
        }
    } else if let KernelParams::Normalization { epsilon, axis } = request.params() {
        let e = epsilon.to_f32();
        // Layer norm keeps its whole-tensor contract.
        if !e.is_finite() || e <= 0.0 || (request.semantic == SemanticOp::LayerNorm && axis != 0) {
            return true;
        }
    }
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints,
        accuracy: KernelAccuracy::Faithful,
        cost: KernelCost::new(0, 12),
    });
    let quantized = packed && !float_weights;
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: if quantized {
            CpuImplementation::Qdot
        } else {
            CpuImplementation::PortableSimd
        },
        required_capabilities: if quantized {
            CpuCapabilities::QDOT.mask()
        } else {
            CpuCapabilities::PORTABLE_SIMD.mask()
        },
        constraints,
        accuracy: KernelAccuracy::Faithful,
        cost: KernelCost::new(32, 2),
    });
    true
}

/// Reusable F32 scratch elements for one row of W4A8 activation state.
pub const fn nvfp4_q8_scratch_elements(k: usize) -> usize {
    k.div_ceil(4) + k.div_ceil(16)
}

/// `[m,k]` F32 activations times native NVFP4 `[n,k]` checkpoint rows.
/// Packed weights and E4M3 block scales remain in their mapped storage.
pub fn invoke_matmul_nvfp4_transposed(
    plan: &CpuKernelPlan,
    activations: &[f32],
    packed: &[u8],
    block_scales: &[u8],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let rows = plan.input(1).ok_or_else(invalid)?.shape.dim(0);
    invoke_matmul_nvfp4_transposed_rows(
        plan,
        activations,
        packed,
        block_scales,
        0,
        rows,
        output,
        scratch,
        parallel,
    )
}

/// Execute a contiguous interval of output rows from an NVFP4 projection. The activation and
/// checkpoint tensors retain their full-plan layout; `output` is compact `[tokens, rows]`.
/// This is the CPU half of a tensor-parallel projection and uses the exact same selected kernel
/// and fused epilogue as whole-matrix execution.
#[allow(clippy::too_many_arguments)]
pub fn invoke_matmul_nvfp4_transposed_rows(
    plan: &CpuKernelPlan,
    activations: &[f32],
    packed: &[u8],
    block_scales: &[u8],
    first_row: usize,
    rows: usize,
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let x = plan.input(0).ok_or_else(invalid)?;
    let w = plan.input(1).ok_or_else(invalid)?;
    let s = plan.input(2).ok_or_else(invalid)?;
    let result = plan.result_tensor(0).ok_or_else(invalid)?;
    let (scale, activation) = match plan.params() {
        KernelParams::MatmulEpilogue { scale, activation } => (scale.to_f32(), activation),
        _ => return Err(invalid()),
    };
    if plan.semantic != SemanticOp::MatMulNvfp4Transposed
        || KernelRequest::matmul_nvfp4_transposed(x, w, s, result, scale, activation).is_none()
        || plan.effects() != KernelEffects::PURE
        || !matches!(
            plan.implementation,
            CpuImplementation::PortableSimd | CpuImplementation::Qdot
        )
        || !x.is_contiguous()
        || !w.is_contiguous()
        || !s.is_contiguous()
        || !result.is_contiguous()
    {
        return Err(invalid());
    }
    let (m, k, n) = (x.shape.dim(0), x.shape.dim(1), w.shape.dim(0));
    let qdot = plan.implementation == CpuImplementation::Qdot;
    if qdot && scratch.len() < nvfp4_q8_scratch_elements(k) {
        return Err(invalid());
    }
    if activations.len() != m * k
        || packed.len() != n * k / 2
        || block_scales.len() != n * k / 16
        || first_row.checked_add(rows).is_none_or(|end| end > n)
        || output.len() != m * rows
    {
        return Err(invalid());
    }
    if m > 1 && (!qdot || scratch.len() >= m * nvfp4_q8_scratch_elements(k)) {
        nvfp4_tokens(
            qdot,
            activations,
            (packed, block_scales),
            (first_row, rows),
            (scale, activation),
            output,
            scratch,
            parallel,
        );
        return Ok(());
    }
    for token in 0..m {
        let row = &activations[token * k..(token + 1) * k];
        let out = &mut output[token * rows..(token + 1) * rows];
        #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
        let quantized = if qdot {
            let (bytes, rest) = scratch.split_at_mut(k.div_ceil(4));
            // SAFETY: `bytes` owns at least `k.div_ceil(4) * 4 >= k`
            // writable bytes and is exclusively borrowed for this call.
            let biased =
                unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u8>(), k) };
            let scales = &mut rest[..k.div_ceil(16)];
            crate::arch::qf4::quantize(row, biased, scales);
            Some(crate::arch::qf4::Quantized { biased, scales })
        } else {
            None
        };
        for_rows(parallel, out, 1, k, |first, out| {
            let first = first_row + first;
            let weights = &packed[first * k / 2..][..out.len() * k / 2];
            let scales = &block_scales[first * k / 16..][..out.len() * k / 16];
            #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
            if let Some(quantized) = &quantized {
                crate::arch::qf4::nvfp4_q8_f32(weights, scales, quantized, out);
            } else {
                arch::f4::nvfp4_f32(weights, scales, row, out);
            }
            #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
            arch::f4::nvfp4_f32(weights, scales, row, out);
            for value in out {
                *value = apply_epilogue(activation, *value * scale);
            }
        });
    }
    Ok(())
}

/// Several tokens through output rows `first_row..first_row + rows` of an
/// NVFP4 matrix: each task owns a range of weight rows and walks it in
/// chunks small enough to stay in cache, running every token's product on a
/// chunk before the next, so the weights stream from memory once rather than
/// once per token. Each row is the one-token product, bit for bit.
#[allow(clippy::too_many_arguments)]
fn nvfp4_tokens(
    qdot: bool,
    activations: &[f32],
    (packed, block_scales): (&[u8], &[u8]),
    (first_row, rows): (usize, usize),
    (scale, activation): (f32, EpilogueActivation),
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) {
    let k = activations.len() / (output.len() / rows);
    let per_token = nvfp4_q8_scratch_elements(k);
    #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
    if qdot {
        for (row, scratch) in activations
            .chunks_exact(k)
            .zip(scratch.chunks_exact_mut(per_token))
        {
            let (bytes, rest) = scratch.split_at_mut(k.div_ceil(4));
            // SAFETY: `bytes` owns at least `k.div_ceil(4) * 4 >= k`
            // writable bytes and is exclusively borrowed here.
            let biased =
                unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u8>(), k) };
            crate::arch::qf4::quantize(row, biased, &mut rest[..k.div_ceil(16)]);
        }
    }
    let scratch: &[f32] = scratch;
    let row_bytes = k / 2 + k / 16;
    // Whole blocks of the kernel's row tile, so a chunk takes the same path
    // through it a full range would.
    let chunk = (TOKEN_CHUNK_BYTES / row_bytes).max(4) / 4 * 4;
    let base = SharedOutput(output.as_mut_ptr());
    let base = &base;
    let tokens = activations.len() / k;
    for_row_ranges(parallel, rows, k * tokens, |first, count| {
        for start in (first..first + count).step_by(chunk) {
            let len = chunk.min(first + count - start);
            let at = first_row + start;
            let weights = &packed[at * k / 2..][..len * k / 2];
            let scales = &block_scales[at * k / 16..][..len * k / 16];
            // Widening the codes and their block scales costs more than the
            // multiply-adds, so where the tier can, a few tokens share each
            // widened step.
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            if !qdot {
                for group in (0..tokens).step_by(arch::f4::TOKENS) {
                    let n = arch::f4::TOKENS.min(tokens - group);
                    let xs: [&[f32]; arch::f4::TOKENS] = core::array::from_fn(|i| {
                        if i < n {
                            &activations[(group + i) * k..][..k]
                        } else {
                            &[]
                        }
                    });
                    // SAFETY: as below, for each of this group's tokens.
                    let mut ys: [&mut [f32]; arch::f4::TOKENS] = core::array::from_fn(|i| {
                        if i < n {
                            unsafe {
                                core::slice::from_raw_parts_mut(
                                    base.0.add((group + i) * rows + start),
                                    len,
                                )
                            }
                        } else {
                            &mut []
                        }
                    });
                    arch::f4::nvfp4_f32_tokens(weights, scales, &xs[..n], &mut ys[..n]);
                    for value in ys[..n].iter_mut().flat_map(|y| y.iter_mut()) {
                        *value = apply_epilogue(activation, *value * scale);
                    }
                }
                continue;
            }
            for (t, x) in activations.chunks_exact(k).enumerate() {
                // SAFETY: this task alone owns output columns
                // `first..first + count` of every token's row, and they lie
                // inside `output` (checked by the caller).
                let y =
                    unsafe { core::slice::from_raw_parts_mut(base.0.add(t * rows + start), len) };
                #[cfg(all(feature = "nvfp4_vnni", target_arch = "x86_64"))]
                if qdot {
                    let state = &scratch[t * per_token..][..per_token];
                    let (bytes, rest) = state.split_at(k.div_ceil(4));
                    // SAFETY: quantization above wrote `k` bytes at the start
                    // of this token's region, which nothing writes now.
                    let biased =
                        unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), k) };
                    let quantized = crate::arch::qf4::Quantized {
                        biased,
                        scales: &rest[..k.div_ceil(16)],
                    };
                    crate::arch::qf4::nvfp4_q8_f32(weights, scales, &quantized, y);
                } else {
                    arch::f4::nvfp4_f32(weights, scales, x, y);
                }
                #[cfg(not(all(feature = "nvfp4_vnni", target_arch = "x86_64")))]
                arch::f4::nvfp4_f32(weights, scales, x, y);
                for value in y {
                    *value = apply_epilogue(activation, *value * scale);
                }
            }
        }
    });
}

fn invalid() -> CpuError {
    CpuError::ShapeMismatch {
        op: "neural kernel",
    }
}
fn packed_len(desc: &TensorDesc) -> Result<usize, CpuError> {
    if !desc.is_contiguous() || desc.layout.offset() != 0 {
        return Err(invalid());
    }
    desc.checked_storage_bytes().ok_or_else(invalid)
}

/// Pure `[m,k] x transpose([n,k]) -> [m,n]`. Inputs remain packed.
/// Validation precedes output mutation. The scalar policy never enters the ISA tier.
pub fn invoke_matmul_transposed(
    plan: &CpuKernelPlan,
    weights: &[u8],
    activations: &[u8],
    output: &mut [f32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::MatMulTransposed {
        return Err(invalid());
    }
    let lhs = plan.input(0).ok_or_else(invalid)?;
    let rhs = plan.input(1).ok_or_else(invalid)?;
    let result = plan.result_tensor(0).ok_or_else(invalid)?;
    if axnn::lowering::KernelRequest::matmul_transposed(lhs, rhs, result).is_none()
        || plan.params() != KernelParams::None
        || plan.effects() != KernelEffects::PURE
        || plan.input(2).is_some()
        || plan.output_dtype != result.dtype
        || plan.shape != result.shape
        || lhs.shape.rank() != 2
        || result.dtype != DType::F32
        || !result.is_contiguous()
        || result.layout.offset() != 0
        || !matches!(lhs.dtype, DType::Q4 | DType::Q6 | DType::Q8)
        || rhs.dtype != DType::Q8
        || weights.len() != packed_len(lhs)?
        || activations.len() != packed_len(rhs)?
        || output.len() != result.shape.checked_num_elements().ok_or_else(invalid)?
        || !matches!(
            plan.implementation,
            CpuImplementation::Scalar | CpuImplementation::Qdot
        )
    {
        return Err(invalid());
    }
    let (m, n, k) = (lhs.shape.dim(0), rhs.shape.dim(0), lhs.shape.dim(1));
    // Use allocation-free dot entry points: the legacy GEMM functions may
    // allocate preprocessing scratch under the `alloc` feature. A prepared
    // graph must retain the same allocation contract in every feature build.
    let wb = lhs.dtype.storage_bytes(k);
    let xb = rhs.dtype.storage_bytes(k);
    for i in 0..m {
        for j in 0..n {
            let w = &weights[i * wb..(i + 1) * wb];
            let x = &activations[j * xb..(j + 1) * xb];
            output[i * n + j] = match (plan.implementation, lhs.dtype) {
                (CpuImplementation::Scalar, DType::Q4) => {
                    arch::scalar::qdot::vec_dot_q4_q8(w, x, k)
                }
                (CpuImplementation::Scalar, DType::Q6) => {
                    arch::scalar::qdot::vec_dot_q6_q8(w, x, k)
                }
                (CpuImplementation::Scalar, _) => arch::scalar::qdot::vec_dot_q8_q8(w, x, k),
                (_, DType::Q4) => arch::qdot::vec_dot_q4_q8(w, x, k),
                (_, DType::Q6) => arch::qdot::vec_dot_q6_q8(w, x, k),
                _ => arch::qdot::vec_dot_q8_q8(w, x, k),
            };
        }
    }
    Ok(())
}

pub(crate) use crate::arch::portable::gemv::{bf16, dot_bf16_f32};

/// Weight bytes a multi-token product runs every token over before moving
/// on, so they are read from memory once and from cache after.
const TOKEN_CHUNK_BYTES: usize = 128 << 10;

/// The output of a product split by weight rows across tasks.
struct SharedOutput(*mut f32);
// SAFETY: tasks write disjoint column ranges (see the use).
unsafe impl Sync for SharedOutput {}

/// F32 activations `[tokens, in]` times FP8 (E4M3, unscaled) or BF16
/// weights `[out, in]`, transposed: `[tokens, out]` in F32.
///
/// The per-tensor FP8 scale and a layer's closing residual add are fused
/// into the write-out (#107): `KernelParams::Scalar` supplies the scale,
/// and a `KernelResult::Accumulate { input: 2 }` result accumulates into
/// the residual stream instead of writing a fresh tensor. `KernelParams::None`
/// with a plain tensor result keeps the unscaled, non-accumulating form.
///
/// The scalar implementation decodes each weight and accumulates in index
/// order; the SIMD one uses lane-parallel sums. Both are `Faithful`.
pub fn invoke_matmul_transposed_f32(
    plan: &CpuKernelPlan,
    activations: &[f32],
    weights: &[u8],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let rhs = check_float_weights(plan, activations, output)?;
    if !matches!(rhs.dtype, DType::B8 | DType::B16) || weights.len() != packed_len(rhs)? {
        return Err(invalid());
    }
    let (scale, activation, accumulate) = float_epilogue(plan);
    let (out, k) = (rhs.shape.dim(0), rhs.shape.dim(1));
    if k == 0 || out == 0 {
        if !accumulate {
            output.fill(0.0);
        }
        return Ok(());
    }
    let row_bytes = rhs.dtype.storage_bytes(k);
    let scalar = plan.implementation == CpuImplementation::Scalar;
    let dtype = rhs.dtype;
    // `out[j] = w[j] . x` for weight rows `first..first + y.len()`.
    let product = |first: usize, x: &[f32], y: &mut [f32]| {
        let w = &weights[first * row_bytes..(first + y.len()) * row_bytes];
        match (dtype, scalar) {
            (DType::B8, false) => arch::gemv::e4m3_f32(w, x, y),
            (_, false) => arch::gemv::bf16_f32(w, x, y),
            (DType::B8, true) => {
                for (w, y) in w.chunks_exact(row_bytes).zip(y) {
                    *y = w
                        .iter()
                        .zip(x)
                        .fold(0.0, |sum, (&b, &v)| sum + crate::fp8::e4m3_to_f32(b) * v);
                }
            }
            (_, true) => {
                for (w, y) in w.chunks_exact(row_bytes).zip(y) {
                    *y = w
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(x)
                        .fold(0.0, |sum, (&b, &v)| sum + bf16(b) * v);
                }
            }
        }
    };
    let rows = |first: usize, x: &[f32], y: &mut [f32]| {
        if accumulate {
            // Bounded task-local scratch preserves the residual while the
            // existing GEMV writes its product, without a weight mirror.
            for (chunk, dst) in y.chunks_mut(32).enumerate() {
                let mut scratch = [0.0; 32];
                product(first + chunk * 32, x, &mut scratch[..dst.len()]);
                for (dst, value) in dst.iter_mut().zip(scratch) {
                    *dst += apply_epilogue(activation, value * scale);
                }
            }
        } else {
            product(first, x, y);
            if plan.params() != KernelParams::None {
                for value in y {
                    *value = apply_epilogue(activation, *value * scale);
                }
            }
        }
    };
    let tokens = activations.len() / k;
    if tokens == 1 {
        // Each task owns a range of weight rows: one output element per row.
        for_rows(parallel, output, 1, k, |first, y| {
            rows(first, activations, y)
        });
        return Ok(());
    }
    // Several tokens: each task owns a range of weight rows and walks it in
    // chunks small enough to stay in cache, running every token's product on
    // a chunk before the next, so the weights stream from memory once. Each
    // row is still computed as a one-token product computes it.
    let chunk = (TOKEN_CHUNK_BYTES / row_bytes).max(1);
    let base = SharedOutput(output.as_mut_ptr());
    let base = &base;
    // FP8 widening costs more than its multiply-adds, so where the tier
    // can, a few tokens share each widened step.
    let grouped = cfg!(all(feature = "avx2", target_arch = "x86_64"))
        && dtype == DType::B8
        && !scalar
        && !accumulate;
    for_row_ranges(parallel, out, k * tokens, |first, count| {
        for start in (first..first + count).step_by(chunk) {
            let len = chunk.min(first + count - start);
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            if grouped {
                let w = &weights[start * row_bytes..(start + len) * row_bytes];
                for group in (0..tokens).step_by(arch::gemv::TOKENS) {
                    let n = arch::gemv::TOKENS.min(tokens - group);
                    let xs: [&[f32]; arch::gemv::TOKENS] = core::array::from_fn(|i| {
                        if i < n {
                            &activations[(group + i) * k..][..k]
                        } else {
                            &[]
                        }
                    });
                    // SAFETY: as below, for each of this group's tokens.
                    let mut ys: [&mut [f32]; arch::gemv::TOKENS] = core::array::from_fn(|i| {
                        if i < n {
                            unsafe {
                                core::slice::from_raw_parts_mut(
                                    base.0.add((group + i) * out + start),
                                    len,
                                )
                            }
                        } else {
                            &mut []
                        }
                    });
                    arch::gemv::e4m3_f32_tokens(w, &xs[..n], &mut ys[..n]);
                    if plan.params() != KernelParams::None {
                        for value in ys[..n].iter_mut().flat_map(|y| y.iter_mut()) {
                            *value = apply_epilogue(activation, *value * scale);
                        }
                    }
                }
                continue;
            }
            for (t, x) in activations.chunks_exact(k).enumerate() {
                // SAFETY: this task alone owns weight rows
                // `first..first + count`, so columns `start..start + len` of
                // every token's output row are written by it only, and they
                // lie inside `output` (checked against the plan above).
                let y =
                    unsafe { core::slice::from_raw_parts_mut(base.0.add(t * out + start), len) };
                rows(start, x, y);
            }
        }
    });
    Ok(())
}

/// F32 activations `[tokens, in]` times F32 weights `[out, in]`, transposed:
/// `[tokens, out]`. For the few F32 matrices a checkpoint keeps, such as a
/// mixture-of-experts router. Accumulation follows
/// [`invoke_matmul_transposed_f32`].
pub fn invoke_matmul_transposed_f32_weights(
    plan: &CpuKernelPlan,
    activations: &[f32],
    weights: &[f32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let rhs = check_float_weights(plan, activations, output)?;
    if rhs.dtype != DType::F32 || Some(weights.len()) != rhs.shape.checked_num_elements() {
        return Err(invalid());
    }
    let (scale, activation, accumulate) = float_epilogue(plan);
    let (out, k) = (rhs.shape.dim(0), rhs.shape.dim(1));
    if k == 0 || out == 0 {
        if !accumulate {
            output.fill(0.0);
        }
        return Ok(());
    }
    let scalar = plan.implementation == CpuImplementation::Scalar;
    for (x, dst) in activations
        .chunks_exact(k)
        .zip(output.chunks_exact_mut(out))
    {
        for_rows(parallel, dst, 1, k, |first, dst| {
            for (w, y) in weights[first * k..].chunks_exact(k).zip(dst) {
                let dot = if scalar {
                    w.iter().zip(x).fold(0.0, |sum, (w, v)| sum + w * v)
                } else {
                    dot_f32(w, x)
                };
                let value = apply_epilogue(activation, dot * scale);
                if accumulate {
                    *y += value;
                } else {
                    *y = value;
                }
            }
        });
    }
    Ok(())
}

fn float_epilogue(plan: &CpuKernelPlan) -> (f32, EpilogueActivation, bool) {
    let (scale, activation) = match plan.params() {
        KernelParams::Scalar(bits) => (bits.to_f32(), EpilogueActivation::None),
        KernelParams::MatmulEpilogue { scale, activation } => (scale.to_f32(), activation),
        _ => (1.0, EpilogueActivation::None),
    };
    (
        scale,
        activation,
        matches!(
            plan.result(),
            axnn::lowering::KernelResult::Accumulate { input: 2 }
        ),
    )
}

/// The checks every float-weight product shares; returns the weight
/// descriptor.
fn check_float_weights<'p>(
    plan: &'p CpuKernelPlan,
    activations: &[f32],
    output: &[f32],
) -> Result<&'p TensorDesc, CpuError> {
    let (Some(lhs), Some(rhs), Some(result)) =
        (plan.input(0), plan.input(1), plan.result_tensor(0))
    else {
        return Err(invalid());
    };
    // A fused residual (#107) accumulates into input 2 rather than writing a
    // fresh result, so it carries the accumulate effect instead of PURE.
    let accumulate = matches!(
        plan.result(),
        axnn::lowering::KernelResult::Accumulate { input: 2 }
    );
    let expected_effects = if accumulate {
        KernelEffects::ACCUMULATES
    } else {
        KernelEffects::PURE
    };
    if plan.semantic != SemanticOp::MatMulTransposed
        || axnn::lowering::KernelRequest::matmul_transposed(lhs, rhs, result).is_none()
        || !matches!(
            plan.params(),
            KernelParams::None | KernelParams::Scalar(_) | KernelParams::MatmulEpilogue { .. }
        )
        || plan.effects() != expected_effects
        || lhs.dtype != DType::F32
        || result.dtype != DType::F32
        || !lhs.is_contiguous()
        || lhs.layout.offset() != 0
        || !rhs.is_contiguous()
        || rhs.layout.offset() != 0
        || !result.is_contiguous()
        || result.layout.offset() != 0
        || Some(activations.len()) != lhs.shape.checked_num_elements()
        || Some(output.len()) != result.shape.checked_num_elements()
        || !matches!(
            plan.implementation,
            CpuImplementation::Scalar | CpuImplementation::PortableSimd
        )
    {
        return Err(invalid());
    }
    Ok(rhs)
}

/// `sum_i w[i] * x[i]` with eight F32 lanes, then the tail.
fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    use core::simd::f32x8;
    use core::simd::num::SimdFloat;
    let (lanes, tail) = w.as_chunks::<8>();
    let (x_lanes, x_tail) = x.as_chunks::<8>();
    let mut sum = f32x8::splat(0.0);
    for (w, x) in lanes.iter().zip(x_lanes) {
        sum += f32x8::from_array(*w) * f32x8::from_array(*x);
    }
    let mut total = sum.reduce_sum();
    for (w, x) in tail.iter().zip(x_tail) {
        total += w * x;
    }
    total
}

/// Existing full-tensor normalization and last-axis softmax kernels, with
/// caller-owned destination storage and no temporary allocations.
pub fn invoke_neural_f32(
    plan: &CpuKernelPlan,
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), CpuError> {
    let desc = plan.input(0).ok_or_else(invalid)?;
    let request = match (plan.semantic, plan.params()) {
        (SemanticOp::Softmax, KernelParams::Axis(axis)) => {
            KernelRequest::softmax(desc, axis as isize)
        }
        (SemanticOp::RmsNorm, KernelParams::Normalization { epsilon, axis }) => {
            KernelRequest::rms_norm_over(
                desc,
                plan.input(1).ok_or_else(invalid)?,
                epsilon.to_f32(),
                axis as isize,
            )
        }
        (SemanticOp::LayerNorm, KernelParams::Normalization { epsilon, axis: 0 }) => {
            KernelRequest::layer_norm(
                desc,
                plan.input(1).ok_or_else(invalid)?,
                plan.input(2).ok_or_else(invalid)?,
                epsilon.to_f32(),
            )
        }
        (SemanticOp::Rope(_), KernelParams::Rope { mode, rotary_dims }) => {
            KernelRequest::rope_partial(
                desc,
                plan.input(1).ok_or_else(invalid)?,
                plan.input(2).ok_or_else(invalid)?,
                mode,
                rotary_dims,
            )
        }
        _ => None,
    }
    .ok_or_else(invalid)?;
    if request.result() != plan.result()
        || request.inputs() != plan.inputs()
        || plan.shape != desc.shape
        || plan.effects() != KernelEffects::PURE
    {
        return Err(invalid());
    }
    let count = desc.shape.checked_num_elements().ok_or_else(invalid)?;
    if input.len() != count
        || output.len() != count
        || plan.output_dtype != DType::F32
        || plan
            .inputs()
            .iter()
            .flatten()
            .any(|d| d.dtype != DType::F32 || !d.is_contiguous() || d.layout.offset() != 0)
        || !matches!(
            plan.implementation,
            CpuImplementation::Scalar | CpuImplementation::PortableSimd
        )
    {
        return Err(invalid());
    }
    let scalar = plan.implementation == CpuImplementation::Scalar;
    match (plan.semantic, plan.params()) {
        (SemanticOp::Softmax, KernelParams::Axis(axis))
            if axis as usize + 1 == plan.shape.rank() =>
        {
            let width = plan.shape.dim(-1);
            if width == 0 {
                return Ok(());
            }
            for (src, dst) in input
                .chunks_exact(width)
                .zip(output.chunks_exact_mut(width))
            {
                if scalar {
                    arch::scalar::reduce::softmax_f32(src, dst);
                } else {
                    arch::portable::reduce::softmax_f32(src, dst);
                }
            }
        }
        (
            SemanticOp::RmsNorm | SemanticOp::LayerNorm,
            KernelParams::Normalization { epsilon, axis },
        ) => {
            let eps = epsilon.to_f32();
            let weight_len = plan
                .input(1)
                .and_then(|d| d.shape.checked_num_elements())
                .ok_or_else(invalid)?;
            let w = weight
                .filter(|w| w.len() == weight_len)
                .ok_or_else(invalid)?;
            if !eps.is_finite() || eps <= 0.0 {
                return Err(invalid());
            }
            if plan.semantic == SemanticOp::RmsNorm {
                // Each leading index is a row over the normalized trailing
                // axes; the weight repeats every `weight_len` elements.
                let width = plan.shape.dims()[axis as usize..].iter().product::<usize>();
                if count == 0 {
                    return Ok(());
                }
                for (row, (src, dst)) in input
                    .chunks_exact(width)
                    .zip(output.chunks_exact_mut(width))
                    .enumerate()
                {
                    let offset = (row * width) % weight_len;
                    let w = &w[offset..offset + width];
                    if scalar {
                        arch::scalar::reduce::rms_norm_f32(src, dst, w, eps);
                    } else {
                        arch::portable::reduce::rms_norm_f32(src, dst, w, eps);
                    }
                }
            } else {
                let w = &w[..count];
                let b = bias.filter(|b| b.len() == count).ok_or_else(invalid)?;
                if scalar {
                    arch::scalar::reduce::layer_norm_f32(input, output, w, b, eps);
                } else {
                    arch::portable::reduce::layer_norm_f32(input, output, w, b, eps);
                }
            }
        }
        (SemanticOp::Rope(_), KernelParams::Rope { mode, rotary_dims }) => {
            let (tokens, heads, dim) = match desc.shape.rank() {
                3 => (
                    desc.shape.dim(0) as usize,
                    desc.shape.dim(1) as usize,
                    desc.shape.dim(2) as usize,
                ),
                2 => (desc.shape.dim(0) as usize, 1, desc.shape.dim(1) as usize),
                _ => return Err(invalid()),
            };
            let rotary = rotary_dims as usize;
            if rotary == 0 || rotary % 2 != 0 || rotary > dim {
                return Err(invalid());
            }
            let half = rotary / 2;
            let cos = weight.ok_or_else(invalid)?;
            let sin = bias.ok_or_else(invalid)?;
            if cos.len() != tokens * half || sin.len() != cos.len() {
                return Err(invalid());
            }
            let rotate = |qk: &mut [f32], cos: &[f32], sin: &[f32], tokens, heads, width| match (
                mode, scalar,
            ) {
                (RopeMode::Interleaved, true) => {
                    arch::scalar::rope::rope_interleaved_f32(qk, cos, sin, tokens, heads, width)
                }
                (RopeMode::NeoX, true) => {
                    arch::scalar::rope::rope_neox_f32(qk, cos, sin, tokens, heads, width)
                }
                (RopeMode::Interleaved, false) => {
                    arch::portable::rope::rope_interleaved_f32(qk, cos, sin, tokens, heads, width)
                }
                (RopeMode::NeoX, false) => {
                    arch::portable::rope::rope_neox_f32(qk, cos, sin, tokens, heads, width)
                }
            };
            // Rotate a fresh output: the graph's separate result tensor.
            output.copy_from_slice(input);
            if rotary == dim {
                rotate(output, cos, sin, tokens, heads, dim);
            } else {
                // A partial rotation turns each head's leading span on its
                // own, as a one-head row of that width; the copied rest of
                // the head passes through.
                for (t, rows) in output.chunks_exact_mut(heads * dim).enumerate() {
                    let (cos, sin) = (&cos[t * half..][..half], &sin[t * half..][..half]);
                    for head in rows.chunks_exact_mut(dim) {
                        rotate(&mut head[..rotary], cos, sin, 1, 1, rotary);
                    }
                }
            }
        }
        _ => return Err(invalid()),
    }
    Ok(())
}

fn validate_packed_cast(
    plan: &CpuKernelPlan,
    float_len: usize,
    packed_len: usize,
    quantize: bool,
) -> Result<DType, CpuError> {
    let input = plan.input(0).ok_or_else(invalid)?;
    let result = plan.result_tensor(0).ok_or_else(invalid)?;
    let request = KernelRequest::cast(input, result.dtype).ok_or_else(invalid)?;
    let (float, packed) = if quantize {
        (input, result)
    } else {
        (result, input)
    };
    if plan.semantic != SemanticOp::Cast
        || request.result() != plan.result()
        || request.inputs() != plan.inputs()
        || plan.effects() != KernelEffects::PURE
        || plan.params() != KernelParams::None
        || plan.implementation != CpuImplementation::Scalar
        || float.dtype != DType::F32
        || !matches!(packed.dtype, DType::Q4 | DType::Q6 | DType::Q8)
        || !input.is_contiguous()
        || input.layout.offset() != 0
        || float_len != float.shape.checked_num_elements().ok_or_else(invalid)?
        || packed_len != packed.checked_storage_bytes().ok_or_else(invalid)?
    {
        return Err(invalid());
    }
    Ok(packed.dtype)
}
/// Execute an explicitly authored packed CAST. No hidden quantization in dot
/// lowering and no full-size unpacked weight mirror or temporary allocation.
pub fn invoke_quantize_f32(
    plan: &CpuKernelPlan,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), CpuError> {
    let dtype = validate_packed_cast(plan, input.len(), output.len(), true)?;
    match dtype {
        DType::Q4 => crate::quant::quantize_q4(input, output),
        DType::Q6 => crate::quant::quantize_q6(input, output),
        _ => crate::quant::quantize_q8(input, output),
    }
    Ok(())
}
/// Decode an explicitly authored packed CAST into caller-owned F32 storage.
pub fn invoke_dequantize_f32(
    plan: &CpuKernelPlan,
    input: &[u8],
    output: &mut [f32],
) -> Result<(), CpuError> {
    let dtype = validate_packed_cast(plan, output.len(), input.len(), false)?;
    match dtype {
        DType::Q4 => crate::quant::dequantize_q4(input, output),
        DType::Q6 => crate::quant::dequantize_q6(input, output),
        _ => crate::quant::dequantize_q8(input, output),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axnn::lowering::{KernelAccuracy, KernelPolicy};
    use axnn::{TensorDesc, shape};

    /// A partial rotation turns only each head's leading span, NeoX-paired
    /// within it, and copies the rest; the full form still rotates whole.
    #[test]
    fn a_partial_rope_rotates_only_its_span() {
        // One token, two heads of four values, the leading two rotating:
        // cos = 0 and sin = 1 turn (a, b) into (-b, a).
        let qk = TensorDesc::new(shape![1, 2, 4], DType::F32);
        let phase = TensorDesc::new(shape![1, 1], DType::F32);
        let request = KernelRequest::rope_partial(&qk, &phase, &phase, RopeMode::NeoX, 2).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            let plan = crate::plan_cpu_kernel(&request, capabilities).unwrap();
            let mut output = [0.0_f32; 8];
            invoke_neural_f32(&plan, &input, Some(&[0.0]), Some(&[1.0]), &mut output).unwrap();
            assert_eq!(output, [-2.0, 1.0, 3.0, 4.0, -6.0, 5.0, 7.0, 8.0]);
        }
        // Phase rows that do not match the span are refused at planning.
        let wide = TensorDesc::new(shape![1, 2], DType::F32);
        assert!(KernelRequest::rope_partial(&qk, &wide, &wide, RopeMode::NeoX, 2).is_none());
    }

    #[test]
    fn rope_requests_plan_on_the_cpu() {
        let qk = TensorDesc::new(shape![1, 2, 4], DType::F32);
        let phase = TensorDesc::new(shape![1, 2], DType::F32);
        let request = KernelRequest::rope(&qk, &phase, &phase, RopeMode::NeoX).unwrap();
        let plan = crate::plan_cpu_kernel(&request, CpuCapabilities::compiled());
        assert!(plan.is_some(), "a rope request should plan");
        let plan = plan.unwrap();
        assert_eq!(plan.semantic, SemanticOp::Rope(RopeMode::NeoX));
    }

    #[test]
    fn nvfp4_request_consumes_checkpoint_storage_directly() {
        let x_desc = TensorDesc::new(shape![1, 16], DType::F32);
        let w_desc = TensorDesc::new(shape![2, 8], DType::U8);
        let s_desc = TensorDesc::new(shape![2, 1], DType::B8);
        let out_desc = TensorDesc::new(shape![1, 2], DType::F32);
        let request = KernelRequest::matmul_nvfp4_transposed(
            &x_desc,
            &w_desc,
            &s_desc,
            &out_desc,
            0.5,
            EpilogueActivation::None,
        )
        .unwrap();
        let plan = crate::plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
        let x: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
        let weights = [0x22; 8]
            .into_iter()
            .chain([0xaa; 8])
            .collect::<alloc::vec::Vec<_>>();
        let scales = [0x38, 0x38]; // E4M3 1.0
        let mut output = [0.0; 2];
        invoke_matmul_nvfp4_transposed(
            &plan,
            &x,
            &weights,
            &scales,
            &mut output,
            &mut [],
            &crate::fork_join::Serial,
        )
        .unwrap();
        assert_eq!(output, [68.0, -68.0]);
        let mut second = [0.0];
        invoke_matmul_nvfp4_transposed_rows(
            &plan,
            &x,
            &weights,
            &scales,
            1,
            1,
            &mut second,
            &mut [],
            &crate::fork_join::Serial,
        )
        .unwrap();
        assert_eq!(second, [-68.0]);
    }

    /// Several tokens through FP8 and BF16 weights in one call give every
    /// token the bits a one-token call gives it, with a fused scale and
    /// activation, split across workers.
    #[test]
    fn float_tokens_match_one_token_products_bit_for_bit() {
        use alloc::vec;
        use alloc::vec::Vec;
        let (m, k, n) = (6, 200, 333);
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
            .collect();
        for dtype in [DType::B8, DType::B16] {
            let weights: Vec<u8> = (0..dtype.storage_bytes(n * k))
                .map(|i| (i * 53 % 251) as u8 & 0x77)
                .collect();
            let desc = |dims: [usize; 2], dtype| TensorDesc::new(axnn::Shape::from(dims), dtype);
            let plan = |tokens: usize| {
                let request = KernelRequest::matmul_transposed_activated(
                    &desc([tokens, k], DType::F32),
                    &desc([n, k], dtype),
                    &desc([tokens, n], DType::F32),
                    0.375,
                    EpilogueActivation::Silu,
                )
                .unwrap();
                crate::plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
            };
            let mut expected = vec![0.0; m * n];
            for (x, y) in x.chunks_exact(k).zip(expected.chunks_exact_mut(n)) {
                invoke_matmul_transposed_f32(&plan(1), x, &weights, y, &crate::fork_join::Serial)
                    .unwrap();
            }
            for parallel in [
                &crate::fork_join::Serial as &dyn ForkJoin,
                &crate::fork_join::Reversed(3),
            ] {
                let mut output = vec![f32::NAN; m * n];
                invoke_matmul_transposed_f32(&plan(m), &x, &weights, &mut output, parallel)
                    .unwrap();
                let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&output), bits(&expected), "{dtype:?}");
            }
        }
    }

    /// Several tokens in one call, sharing each pass over the weights, give
    /// every token the bits a one-token call gives it: for each admitted
    /// kernel, for a row interval, and split across workers.
    #[test]
    fn nvfp4_tokens_match_one_token_products_bit_for_bit() {
        use alloc::vec;
        use alloc::vec::Vec;
        let (m, k, n) = (5, 96, 700);
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let x: Vec<f32> = (0..m * k)
            .map(|_| (next() >> 8) as f32 / (1 << 23) as f32 - 1.0)
            .collect();
        let packed: Vec<u8> = (0..n * k / 2).map(|_| next() as u8).collect();
        // Finite E4M3 scales between 2^-3 and 2^3.
        let scales: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x20 + (next() % 0x30) as u8)
            .collect();
        let desc = |dims: [usize; 2], dtype| TensorDesc::new(axnn::Shape::from(dims), dtype);
        let plan = |tokens: usize, accuracy| {
            let request = KernelRequest::matmul_nvfp4_transposed(
                &desc([tokens, k], DType::F32),
                &desc([n, k / 2], DType::U8),
                &desc([n, k / 16], DType::B8),
                &desc([tokens, n], DType::F32),
                0.75,
                EpilogueActivation::Silu,
            )
            .unwrap();
            crate::select_cpu_kernel(
                &request,
                KernelPolicy::new(CpuCapabilities::compiled().mask())
                    .with_minimum_accuracy(accuracy),
            )
            .unwrap()
        };
        for accuracy in [KernelAccuracy::Faithful, KernelAccuracy::Approximate] {
            let one = plan(1, accuracy);
            let many = plan(m, accuracy);
            assert_eq!(one.implementation, many.implementation);
            let (first, rows) = (13, 611);
            let mut expected = vec![0.0; m * rows];
            let mut scratch = vec![0.0; nvfp4_q8_scratch_elements(k)];
            for (t, x) in x.chunks_exact(k).enumerate() {
                invoke_matmul_nvfp4_transposed_rows(
                    &one,
                    x,
                    &packed,
                    &scales,
                    first,
                    rows,
                    &mut expected[t * rows..(t + 1) * rows],
                    &mut scratch,
                    &crate::fork_join::Serial,
                )
                .unwrap();
            }
            let mut scratch = vec![0.0; m * nvfp4_q8_scratch_elements(k)];
            for parallel in [
                &crate::fork_join::Serial as &dyn ForkJoin,
                &crate::fork_join::Reversed(3),
            ] {
                let mut output = vec![f32::NAN; m * rows];
                invoke_matmul_nvfp4_transposed_rows(
                    &many,
                    &x,
                    &packed,
                    &scales,
                    first,
                    rows,
                    &mut output,
                    &mut scratch,
                    parallel,
                )
                .unwrap();
                let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&output), bits(&expected), "{accuracy:?}");
            }
        }
    }

    #[test]
    fn packed_q8_full_signed_domain_and_policy_admission() {
        for k in [32, 64, 96] {
            let shape = axnn::Shape::from([1, k]);
            let input = TensorDesc::new(shape, DType::Q8);
            let output = TensorDesc::new(shape![1, 1], DType::F32);
            let request = KernelRequest::matmul_transposed(&input, &input, &output).unwrap();
            let mut a = [0u8; 102];
            let mut b = [0u8; 102];
            let len = k / 32 * 34;
            let mut expected = 0i32;
            for block in 0..k / 32 {
                a[block * 34..block * 34 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                b[block * 34..block * 34 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                for lane in 0..32 {
                    let x = [-128i8, -127, 0, 127][(lane + block) % 4];
                    let y = [-128i8, 127, -127, -128][lane % 4];
                    a[block * 34 + 2 + lane] = x as u8;
                    b[block * 34 + 2 + lane] = y as u8;
                    expected += x as i32 * y as i32;
                }
            }
            for caps in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
                let plan = crate::plan_cpu_kernel(&request, caps).unwrap();
                assert_eq!(
                    plan.implementation,
                    if caps == CpuCapabilities::SCALAR {
                        CpuImplementation::Scalar
                    } else {
                        CpuImplementation::Qdot
                    }
                );
                let mut dst = [0f32];
                invoke_matmul_transposed(&plan, &a[..len], &b[..len], &mut dst).unwrap();
                assert_eq!(dst, [expected as f32]);
                let mut corrupted = plan;
                corrupted.shape = shape![2, 1];
                assert!(
                    invoke_matmul_transposed(&corrupted, &a[..len], &b[..len], &mut dst).is_err()
                );
                assert_eq!(dst, [expected as f32]);
            }
            assert!(
                crate::select_cpu_kernel(
                    &request,
                    KernelPolicy::new(CpuCapabilities::compiled().mask())
                        .with_minimum_accuracy(KernelAccuracy::Strict)
                )
                .is_none()
            );
        }
    }
}

#[cfg(test)]
mod float_weight_tests {
    use super::*;
    use crate::fp8::{e4m3_to_f32, f32_to_e4m3_rne};
    use crate::plan_cpu_kernel;
    use alloc::vec::Vec;
    use axnn::Shape;

    fn desc(dims: &[usize], dtype: DType) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), dtype)
    }

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 37 + seed * 11) % 41) as f32 - 20.0) / 8.0)
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        // Truncation keeps the test's weights exactly representable.
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// `[tokens, in] x [out, in]^T` for FP8 and BF16 weights, through both
    /// the scalar and SIMD candidates, against an F64 sum of decoded weights.
    #[test]
    fn float_weight_matmuls_match_a_direct_sum_in_both_implementations() {
        let (tokens, inputs, outputs) = (3, 45, 7);
        let x = values(tokens * inputs, 1);
        let dense = values(outputs * inputs, 2);
        for dtype in [DType::B8, DType::B16] {
            let weights: Vec<u8> = match dtype {
                DType::B8 => dense.iter().map(|&v| f32_to_e4m3_rne(v)).collect(),
                _ => bf16_bytes(&dense),
            };
            let decoded: Vec<f64> = match dtype {
                DType::B8 => weights.iter().map(|&b| f64::from(e4m3_to_f32(b))).collect(),
                _ => weights
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&pair| f64::from(bf16(pair)))
                    .collect(),
            };
            let request = KernelRequest::matmul_transposed(
                &desc(&[tokens, inputs], DType::F32),
                &desc(&[outputs, inputs], dtype),
                &desc(&[tokens, outputs], DType::F32),
            )
            .unwrap();
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
                let plan = plan_cpu_kernel(&request, capabilities).unwrap();
                let mut out = [0.0_f32; 21];
                invoke_matmul_transposed_f32(&plan, &x, &weights, &mut out, &crate::Serial)
                    .unwrap();
                for t in 0..tokens {
                    for o in 0..outputs {
                        let expected: f64 = (0..inputs)
                            .map(|i| f64::from(x[t * inputs + i]) * decoded[o * inputs + i])
                            .sum();
                        let actual = f64::from(out[t * outputs + o]);
                        assert!(
                            (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                            "{dtype:?} {:?}: {actual} vs {expected}",
                            plan.implementation
                        );
                    }
                }
            }
            // The packed weights must be exactly the planned size.
            let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
            let mut out = [0.0_f32; 21];
            assert!(
                invoke_matmul_transposed_f32(&plan, &x, &weights[1..], &mut out, &crate::Serial)
                    .is_err()
            );
        }
        // Quantized activations are not a float-weight product.
        assert!(
            plan_cpu_kernel(
                &KernelRequest::matmul_transposed(
                    &desc(&[1, 32], DType::Q8),
                    &desc(&[2, 32], DType::B8),
                    &desc(&[1, 2], DType::F32),
                )
                .unwrap(),
                CpuCapabilities::compiled(),
            )
            .is_none()
        );
    }

    /// A fused activation (#107) is applied after the scale, matching the
    /// standalone kernel the separate node would have used.
    #[test]
    fn float_weight_matmul_applies_the_fused_activation() {
        let (tokens, inputs, outputs) = (2, 32, 3);
        let x = values(tokens * inputs, 6);
        let dense = values(outputs * inputs, 7);
        let weights: Vec<u8> = dense.iter().map(|&v| f32_to_e4m3_rne(v)).collect();
        let scale = 0.25_f32;
        for activation in [EpilogueActivation::Silu, EpilogueActivation::Sigmoid] {
            let unfused = KernelRequest::matmul_transposed_scaled(
                &desc(&[tokens, inputs], DType::F32),
                &desc(&[outputs, inputs], DType::B8),
                &desc(&[tokens, outputs], DType::F32),
                scale,
            )
            .unwrap();
            let fused = KernelRequest::matmul_transposed_activated(
                &desc(&[tokens, inputs], DType::F32),
                &desc(&[outputs, inputs], DType::B8),
                &desc(&[tokens, outputs], DType::F32),
                scale,
                activation,
            )
            .unwrap();
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
                let mut scaled = [0.0_f32; 6];
                invoke_matmul_transposed_f32(
                    &plan_cpu_kernel(&unfused, capabilities).unwrap(),
                    &x,
                    &weights,
                    &mut scaled,
                    &crate::Serial,
                )
                .unwrap();
                let mut out = [0.0_f32; 6];
                invoke_matmul_transposed_f32(
                    &plan_cpu_kernel(&fused, capabilities).unwrap(),
                    &x,
                    &weights,
                    &mut out,
                    &crate::Serial,
                )
                .unwrap();
                // Fusing must equal applying the activation afterwards, bit
                // for bit: both go through the same `axmath` kernel.
                for (fused_value, scaled_value) in out.iter().zip(&scaled) {
                    assert_eq!(*fused_value, apply_epilogue(activation, *scaled_value));
                }
            }
        }
    }

    /// The fused residual (#107) accumulates the scaled product into the
    /// residual stream, leaving `residual + product · scale`.
    #[test]
    fn float_weight_matmul_accumulates_the_fused_residual() {
        let (tokens, inputs, outputs) = (2, 32, 3);
        let x = values(tokens * inputs, 3);
        let dense = values(outputs * inputs, 4);
        let weights: Vec<u8> = dense.iter().map(|&v| f32_to_e4m3_rne(v)).collect();
        let decoded: Vec<f64> = weights.iter().map(|&b| f64::from(e4m3_to_f32(b))).collect();
        let scale = 0.5_f32;
        let residual = values(tokens * outputs, 5);
        let request = KernelRequest::matmul_transposed_scaled_accumulate(
            &desc(&[tokens, inputs], DType::F32),
            &desc(&[outputs, inputs], DType::B8),
            &desc(&[tokens, outputs], DType::F32),
            scale,
        )
        .unwrap();
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
            let plan = plan_cpu_kernel(&request, capabilities).unwrap();
            let mut out = [0.0_f32; 6];
            out.copy_from_slice(&residual);
            invoke_matmul_transposed_f32(&plan, &x, &weights, &mut out, &crate::Serial).unwrap();
            for t in 0..tokens {
                for o in 0..outputs {
                    let product: f64 = (0..inputs)
                        .map(|i| f64::from(x[t * inputs + i]) * decoded[o * inputs + i])
                        .sum();
                    let expected =
                        f64::from(residual[t * outputs + o]) + f64::from(scale) * product;
                    let actual = f64::from(out[t * outputs + o]);
                    assert!(
                        (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                        "{:?}: {actual} vs {expected}",
                        plan.implementation
                    );
                }
            }
        }
    }

    /// The fused per-tensor scale (#107) multiplies the unscaled result, in
    /// both the scalar and SIMD candidates.
    #[test]
    fn float_weight_matmul_applies_the_fused_scale() {
        let (tokens, inputs, outputs) = (2, 32, 3);
        let x = values(tokens * inputs, 1);
        let dense = values(outputs * inputs, 2);
        let weights: Vec<u8> = dense.iter().map(|&v| f32_to_e4m3_rne(v)).collect();
        let decoded: Vec<f64> = weights.iter().map(|&b| f64::from(e4m3_to_f32(b))).collect();
        let scale = 0.125_f32;
        let request = KernelRequest::matmul_transposed_scaled(
            &desc(&[tokens, inputs], DType::F32),
            &desc(&[outputs, inputs], DType::B8),
            &desc(&[tokens, outputs], DType::F32),
            scale,
        )
        .unwrap();
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
            let plan = plan_cpu_kernel(&request, capabilities).unwrap();
            let mut out = [0.0_f32; 6];
            invoke_matmul_transposed_f32(&plan, &x, &weights, &mut out, &crate::Serial).unwrap();
            for t in 0..tokens {
                for o in 0..outputs {
                    let expected: f64 = f64::from(scale)
                        * (0..inputs)
                            .map(|i| f64::from(x[t * inputs + i]) * decoded[o * inputs + i])
                            .sum::<f64>();
                    let actual = f64::from(out[t * outputs + o]);
                    assert!(
                        (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                        "{:?}: {actual} vs {expected}",
                        plan.implementation
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_rows_gather_exactly() {
        let table = values(4 * 6, 3);
        let bytes = bf16_bytes(&table);
        let request =
            KernelRequest::gather(&desc(&[4, 6], DType::B16), &desc(&[2], DType::I32)).unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
        let mut out = [0.0_f32; 12];
        crate::invoke_gather(&plan, crate::GatherTable::B16(&bytes), &[3, 1], &mut out).unwrap();
        let truncated = |v: f32| f32::from_bits(v.to_bits() & 0xffff_0000);
        let expected: Vec<f32> = table[18..24]
            .iter()
            .chain(&table[6..12])
            .map(|&v| truncated(v))
            .collect();
        assert_eq!(out[..], expected[..]);
    }
}
