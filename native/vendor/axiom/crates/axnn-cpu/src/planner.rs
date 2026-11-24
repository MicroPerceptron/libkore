//! Static CPU kernel catalog and semantic lowering.
//!
//! This module does not inspect the host. The final runtime supplies a
//! [`CpuCapabilities`] policy; the catalog only enumerates implementations
//! compiled into this artifact. Selection is allocation-free and uses the
//! generic planner in `axnn`.

use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, KernelAccuracy, KernelCandidate, KernelCatalog,
    KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelPlan, KernelPolicy,
    KernelRequest, KernelResult, KernelScalar, KernelShapePolicy, LogicalOp, SemanticOp,
    select_kernel,
};
use axnn::{BinaryOp, CompareOp, DType, ReduceOp, Shape, TernaryOp, UnaryOp};

use crate::arch;
use crate::error::CpuError;

const CAP_PORTABLE_SIMD: CapabilityMask = CapabilityMask::from_bits(1 << 0);
const CAP_QDOT: CapabilityMask = CapabilityMask::from_bits(1 << 2);
const CAP_AVX2: CapabilityMask = CapabilityMask::from_bits(1 << 1);

/// CPU implementation capabilities admitted by the final runtime.
///
/// These are policy values, not host probes. `SCALAR` is universally safe;
/// `compiled()` admits the best tiers this artifact was built to contain.
/// The policy only governs implementations enumerated by this catalog; it
/// does not downgrade the artifact-wide target baseline selected by
/// `arch::*` at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuCapabilities(CapabilityMask);

impl CpuCapabilities {
    /// Baseline scalar kernels only.
    pub const SCALAR: Self = Self(CapabilityMask::NONE);
    /// Scalar plus portable-SIMD kernels.
    pub const PORTABLE_SIMD: Self = Self(CAP_PORTABLE_SIMD);
    /// Admits the artifact-selected quantized ISA tier (see build_info::qdot_isa).
    /// The final runtime must authorize the artifact CPU contract.
    pub const QDOT: Self = Self(CAP_QDOT);
    /// Scalar, portable SIMD, and the AVX2/FMA/F16C tier.
    ///
    /// The AVX2 candidate is enumerated only when that tier was compiled in,
    /// so supplying this policy to another artifact remains safe.
    pub const AVX2: Self = Self(CAP_PORTABLE_SIMD.union(CAP_AVX2));

    /// Capabilities represented by the kernels compiled into this artifact.
    /// No runtime host detection is performed.
    pub const fn compiled() -> Self {
        let mut bits = CAP_PORTABLE_SIMD.bits() | CAP_QDOT.bits();
        if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            bits |= CAP_AVX2.bits();
        }
        Self(CapabilityMask::from_bits(bits))
    }

    pub const fn mask(self) -> CapabilityMask {
        self.0
    }

    /// Preserve a runtime's admission mask when crossing adapter boundaries.
    /// This is not an ISA probe: only compiled catalog candidates can be selected.
    pub const fn from_mask(mask: CapabilityMask) -> Self {
        Self(mask)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0.union(other.0))
    }
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self::compiled()
    }
}

/// Executable CPU implementation family. The semantic operation remains in
/// [`KernelPlan::semantic`] rather than being multiplied into implementation
/// identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuImplementation {
    Scalar,
    PortableSimd,
    /// Packed integer-dot tier selected by the artifact CPU contract.
    Qdot,
    #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
    Avx2,
}

impl CpuImplementation {
    /// Exact implementation label for logs and calibration. Qdot labels its
    /// allocation-free vector-dot entry point, including AMX's portable
    /// fallback (AMX's separate GEMM entry point is not used by this catalog).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::PortableSimd => "portable-simd",
            Self::Qdot => {
                if cfg!(all(feature = "amx", target_arch = "x86_64")) {
                    "portable-qdot"
                } else {
                    crate::build_info::qdot_isa()
                }
            }
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            Self::Avx2 => "avx2+fma+f16c",
        }
    }
}

pub type CpuKernelPlan = KernelPlan<CpuImplementation>;

/// Zero-sized, statically enumerated CPU kernel catalog.
///
/// The automatic surface covers graph elementwise families, dense F32 matrix
/// multiplication, ray packing, typed borrowed brick-grid queries, and RGB
/// volume sampling.
/// Additional CPU kernel families become available to graph execution by
/// extending this catalog and its typed invoker, without changing `.ax` opcodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuKernelCatalog;

impl KernelCatalog for CpuKernelCatalog {
    type Implementation = CpuImplementation;

    fn visit_candidates<V: CandidateVisitor<Self::Implementation>>(
        &self,
        request: &KernelRequest,
        visitor: &mut V,
    ) {
        if crate::neural::visit_candidates(request, visitor) {
            return;
        }
        if crate::integer::visit_candidates(request, visitor) {
            return;
        }
        if crate::raster_program::visit_candidates(request, visitor) {
            return;
        }
        if crate::kv_cache::visit_candidates(request, visitor) {
            return;
        }
        if crate::tensor_ops::visit_candidates(request, visitor) {
            return;
        }
        if crate::recurrent::visit_candidates(request, visitor) {
            return;
        }
        if crate::moe::visit_candidates(request, visitor) {
            return;
        }
        if crate::routed::visit_candidates(request, visitor) {
            return;
        }
        if request.semantic == SemanticOp::Interleave3F32 {
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints: KernelConstraints {
                    max_rank: 3,
                    shape_policy: KernelShapePolicy::Typed,
                    ..KernelConstraints::elementwise(DType::F32)
                },
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(0, 1),
            });
            return;
        }
        if request.semantic == SemanticOp::VoxelBrickTrace {
            let Some(scratch_bytes) = request.elements().checked_mul(core::mem::size_of::<u32>())
            else {
                return;
            };
            let constraints = KernelConstraints {
                output_dtypes: DTypeSet::one(DType::F32).union(DTypeSet::one(DType::I32)),
                shape_policy: KernelShapePolicy::Typed,
                ..KernelConstraints::elementwise(DType::F32)
            };
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints,
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(0, 12).with_temporary_bytes(scratch_bytes),
            });
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::PortableSimd,
                required_capabilities: CAP_PORTABLE_SIMD,
                constraints: KernelConstraints {
                    min_elements: 4,
                    ..constraints
                },
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(32, 4).with_temporary_bytes(scratch_bytes),
            });
            return;
        }
        if request.semantic == SemanticOp::VolumeTrilinearRgb {
            // Typed slice invokers take already selected Rust subslices, not
            // a larger allocation plus an offset stored in the descriptor.
            if (0..3).any(|index| {
                request
                    .input(index)
                    .is_none_or(|desc| desc.layout.offset() != 0)
            }) {
                return;
            }
            let constraints = KernelConstraints {
                max_rank: 2,
                shape_policy: KernelShapePolicy::Typed,
                ..KernelConstraints::elementwise(DType::F32)
            };
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints,
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(0, 12),
            });
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::PortableSimd,
                required_capabilities: CAP_PORTABLE_SIMD,
                constraints: KernelConstraints {
                    min_elements: 4,
                    ..constraints
                },
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(8, 8),
            });
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            if matches!(request.params(), KernelParams::VolumeTrilinearRgb(params)
                if params.storage == axnn::lowering::VolumeRgbStorage::Uf8E5M3)
            {
                visitor.visit(KernelCandidate {
                    semantic: request.semantic,
                    implementation: CpuImplementation::Avx2,
                    required_capabilities: CAP_AVX2,
                    constraints: KernelConstraints {
                        min_elements: 4,
                        ..constraints
                    },
                    accuracy: KernelAccuracy::Strict,
                    cost: KernelCost::new(8, 6),
                });
            }
            return;
        }
        if request.semantic == SemanticOp::MatMul {
            let constraints = KernelConstraints {
                input_dtypes: [
                    DTypeSet::one(DType::F32),
                    DTypeSet::one(DType::F32),
                    DTypeSet::one(DType::F32),
                ],
                output_dtypes: DTypeSet::one(DType::F32),
                min_elements: 0,
                max_elements: usize::MAX,
                element_multiple: 1,
                min_work_items: 0,
                max_work_items: u64::MAX,
                work_item_multiple: 1,
                max_rank: 2,
                contiguous_only: true,
                shape_policy: KernelShapePolicy::Typed,
            };
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints,
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(0, 10),
            });
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::PortableSimd,
                required_capabilities: CAP_PORTABLE_SIMD,
                constraints: KernelConstraints {
                    min_work_items: 8,
                    ..constraints
                },
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(64, 2),
            });
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Avx2,
                required_capabilities: CAP_AVX2,
                constraints: KernelConstraints {
                    min_work_items: 16,
                    ..constraints
                },
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(48, 1),
            });
            return;
        }
        if let SemanticOp::Reduce(op) = request.semantic {
            let constraints = KernelConstraints {
                input_dtypes: [DTypeSet::one(DType::F32), DTypeSet::NONE, DTypeSet::NONE],
                output_dtypes: DTypeSet::one(DType::F32),
                min_elements: 1,
                max_elements: 1,
                element_multiple: 1,
                min_work_items: usize::from(op == ReduceOp::Max) as u64,
                max_work_items: u64::MAX,
                work_item_multiple: 1,
                max_rank: axnn::MAX_RANK as u8,
                contiguous_only: true,
                shape_policy: KernelShapePolicy::Typed,
            };
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints,
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(0, 12),
            });
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::PortableSimd,
                required_capabilities: CAP_PORTABLE_SIMD,
                constraints: KernelConstraints {
                    min_work_items: 8,
                    ..constraints
                },
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(16, 2),
            });
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Avx2,
                required_capabilities: CAP_AVX2,
                constraints: KernelConstraints {
                    min_work_items: 16,
                    ..constraints
                },
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(12, 1),
            });
            return;
        }
        if !supports_scalar(request.semantic) {
            return;
        }

        let mut scalar_constraints = match request.semantic {
            SemanticOp::Identity => match request.primary_input_dtype() {
                dtype @ (DType::F32 | DType::I32 | DType::U8) => {
                    KernelConstraints::elementwise(dtype)
                }
                _ => return,
            },
            SemanticOp::Compare(_) => match request.primary_input_dtype() {
                dtype @ (DType::F32 | DType::I32) => KernelConstraints::compare(dtype),
                _ => return,
            },
            SemanticOp::Logical(_) => KernelConstraints::elementwise(DType::U8),
            SemanticOp::Select => KernelConstraints::select(DType::F32),
            _ => KernelConstraints::elementwise(DType::F32),
        };
        scalar_constraints.shape_policy = if request.semantic == SemanticOp::Identity {
            KernelShapePolicy::Exact
        } else {
            KernelShapePolicy::Broadcast
        };

        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation: CpuImplementation::Scalar,
            required_capabilities: CapabilityMask::NONE,
            constraints: scalar_constraints,
            accuracy: accuracy(request.semantic),
            cost: KernelCost::new(0, 12),
        });

        if supports_vector(request.semantic) {
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::PortableSimd,
                required_capabilities: CAP_PORTABLE_SIMD,
                constraints: KernelConstraints {
                    min_elements: 8,
                    ..KernelConstraints::elementwise(DType::F32)
                },
                accuracy: accuracy(request.semantic),
                cost: KernelCost::new(32, 2),
            });

            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Avx2,
                required_capabilities: CAP_AVX2,
                constraints: KernelConstraints {
                    min_elements: 16,
                    ..KernelConstraints::elementwise(DType::F32)
                },
                accuracy: accuracy(request.semantic),
                cost: KernelCost::new(24, 1),
            });
        }
    }
}

/// Select a CPU implementation using caller-owned policy.
pub fn select_cpu_kernel(request: &KernelRequest, policy: KernelPolicy) -> Option<CpuKernelPlan> {
    select_kernel(&CpuKernelCatalog, request, policy)
}

/// Select using the common CPU capability policy and default cost/accuracy
/// admission.
pub fn plan_cpu_kernel(
    request: &KernelRequest,
    capabilities: CpuCapabilities,
) -> Option<CpuKernelPlan> {
    select_cpu_kernel(request, KernelPolicy::new(capabilities.mask()))
}

const fn supports_scalar(semantic: SemanticOp) -> bool {
    matches!(
        semantic,
        SemanticOp::Identity
            | SemanticOp::Unary(_)
            | SemanticOp::Binary(_)
            | SemanticOp::Ternary(TernaryOp::Clamp | TernaryOp::Fma)
            | SemanticOp::Compare(_)
            | SemanticOp::Logical(_)
            | SemanticOp::Select
    )
}

const fn supports_vector(semantic: SemanticOp) -> bool {
    matches!(
        semantic,
        SemanticOp::Unary(
            UnaryOp::Relu
                | UnaryOp::Neg
                | UnaryOp::Sqrt
                | UnaryOp::Exp
                | UnaryOp::Tanh
                | UnaryOp::Sigmoid
                | UnaryOp::Silu
                | UnaryOp::Gelu
        ) | SemanticOp::Binary(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div)
    )
}

const fn accuracy(semantic: SemanticOp) -> KernelAccuracy {
    match semantic {
        SemanticOp::Identity
        | SemanticOp::Unary(
            UnaryOp::Abs
            | UnaryOp::Ceil
            | UnaryOp::Floor
            | UnaryOp::Fract
            | UnaryOp::Neg
            | UnaryOp::Reciprocal
            | UnaryOp::Round
            | UnaryOp::Sign
            | UnaryOp::Trunc
            | UnaryOp::Relu,
        )
        | SemanticOp::Binary(
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Min
            | BinaryOp::Max
            | BinaryOp::Rem,
        )
        | SemanticOp::Ternary(TernaryOp::Clamp | TernaryOp::Fma)
        | SemanticOp::Compare(_)
        | SemanticOp::Logical(_)
        | SemanticOp::Select => KernelAccuracy::Strict,
        _ => KernelAccuracy::Faithful,
    }
}

/// Invoke a selected F32 elementwise kernel over caller-owned slices.
///
/// Inputs that are not part of the semantic arity must be `None`. The helper
/// validates dtype, shape, and slice lengths before entering a microkernel.
pub fn invoke_f32(
    plan: &CpuKernelPlan,
    first: &[f32],
    second: Option<&[f32]>,
    third: Option<&[f32]>,
    dst: &mut [f32],
) -> Result<(), CpuError> {
    if plan.output_dtype != DType::F32
        || plan.semantic == SemanticOp::Select
        || matches!(plan.semantic, SemanticOp::Compare(_))
        || plan
            .input_dtypes
            .iter()
            .flatten()
            .any(|&dtype| dtype != DType::F32)
    {
        return Err(CpuError::UnsupportedOp {
            op: "planned elementwise",
            dtype: plan.input_dtypes[0].unwrap_or(plan.output_dtype),
        });
    }
    let expected = plan.shape.num_elements();
    if dst.len() != expected || !input_len_matches(plan, 0, first.len()) {
        return Err(CpuError::ShapeMismatch {
            op: "planned elementwise",
        });
    }
    match plan.semantic.arity() {
        1 if second.is_none() && third.is_none() => {}
        2 if second.is_some_and(|v| input_len_matches(plan, 1, v.len())) && third.is_none() => {}
        3 if second.is_some_and(|v| input_len_matches(plan, 1, v.len()))
            && third.is_some_and(|v| input_len_matches(plan, 2, v.len())) => {}
        _ => {
            return Err(CpuError::ShapeMismatch {
                op: "planned elementwise arity",
            });
        }
    }

    match plan.implementation {
        CpuImplementation::Qdot => {
            return Err(CpuError::UnsupportedOp {
                op: "kernel",
                dtype: plan.output_dtype,
            });
        }
        CpuImplementation::Scalar => invoke_scalar(plan, first, second, third, dst),
        CpuImplementation::PortableSimd => {
            invoke_portable(plan.semantic, first, second, third, dst)
        }
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        CpuImplementation::Avx2 => invoke_avx2(plan.semantic, first, second, third, dst),
    }
}

/// Invoke a selected full-tensor F32 reduction into one scalar output.
///
/// The selected implementation remains explicit so a runtime that admits
/// only the scalar tier never reaches a wider kernel compiled into the same
/// artifact. Empty sums produce zero; empty maxima have no candidate.
pub fn invoke_reduce_f32(
    plan: &CpuKernelPlan,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), CpuError> {
    let SemanticOp::Reduce(op) = plan.semantic else {
        return Err(CpuError::UnsupportedOp {
            op: "planned reduction",
            dtype: plan.output_dtype,
        });
    };
    let Some(desc) = plan.input(0) else {
        return Err(CpuError::ShapeMismatch {
            op: "planned reduction",
        });
    };
    let expected = KernelRequest::reduce(op, desc);
    if plan.input(1).is_some()
        || plan.input(2).is_some()
        || plan.output_dtype != DType::F32
        || plan.shape != Shape::SCALAR
        || plan.result() != KernelResult::Scalar(KernelScalar::F32)
        || plan.params() != expected.params()
        || plan.effects() != KernelEffects::PURE
        || desc.dtype != DType::F32
        || !desc.is_contiguous()
        || input.len() != desc.shape.num_elements()
        || output.len() != 1
        || (op == ReduceOp::Max && input.is_empty())
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned reduction",
        });
    }

    output[0] = match (op, plan.implementation) {
        (ReduceOp::Sum, CpuImplementation::Scalar) => arch::scalar::reduce::sum_f32(input),
        (ReduceOp::Max, CpuImplementation::Scalar) => arch::scalar::reduce::max_f32(input),
        (ReduceOp::Sum, CpuImplementation::PortableSimd) => arch::portable::reduce::sum_f32(input),
        (ReduceOp::Max, CpuImplementation::PortableSimd) => arch::portable::reduce::max_f32(input),
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        (ReduceOp::Sum, CpuImplementation::Avx2) => arch::avx2::reduce::sum_f32(input),
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        (ReduceOp::Max, CpuImplementation::Avx2) => arch::avx2::reduce::max_f32(input),
        (_, CpuImplementation::Qdot) => {
            return Err(CpuError::UnsupportedOp {
                op: "planned reduction",
                dtype: DType::F32,
            });
        }
    };
    Ok(())
}

/// Invoke a selected U8 identity or boolean-mask kernel.
///
/// Boolean inputs treat zero as false and every non-zero value as true. Every
/// logical result is normalized to `0` or `1`; identity preserves its bytes.
pub fn invoke_u8(
    plan: &CpuKernelPlan,
    first: &[u8],
    second: Option<&[u8]>,
    dst: &mut [u8],
) -> Result<(), CpuError> {
    if plan.output_dtype != DType::U8
        || plan.implementation != CpuImplementation::Scalar
        || plan
            .input_dtypes
            .iter()
            .flatten()
            .any(|&dtype| dtype != DType::U8)
        || !matches!(plan.semantic, SemanticOp::Identity | SemanticOp::Logical(_))
        || dst.len() != plan.shape.num_elements()
        || !input_len_matches(plan, 0, first.len())
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned U8 elementwise",
        });
    }

    match plan.semantic {
        SemanticOp::Identity if second.is_none() => {
            dst.copy_from_slice(first);
            Ok(())
        }
        SemanticOp::Logical(LogicalOp::Not) if second.is_none() => {
            let first_shape = plan.input_shapes[0].unwrap();
            for (index, output) in dst.iter_mut().enumerate() {
                *output = u8::from(broadcast_at(first, first_shape, plan.shape, index) == 0);
            }
            Ok(())
        }
        SemanticOp::Logical(op @ (LogicalOp::And | LogicalOp::Or | LogicalOp::Xor)) => {
            let second = second
                .filter(|value| input_len_matches(plan, 1, value.len()))
                .ok_or(CpuError::ShapeMismatch {
                    op: "planned U8 binary",
                })?;
            let first_shape = plan.input_shapes[0].unwrap();
            let second_shape = plan.input_shapes[1].unwrap();
            for (index, output) in dst.iter_mut().enumerate() {
                let lhs = broadcast_at(first, first_shape, plan.shape, index) != 0;
                let rhs = broadcast_at(second, second_shape, plan.shape, index) != 0;
                *output = u8::from(match op {
                    LogicalOp::And => lhs && rhs,
                    LogicalOp::Or => lhs || rhs,
                    LogicalOp::Xor => lhs ^ rhs,
                    LogicalOp::Not => unreachable!(),
                });
            }
            Ok(())
        }
        _ => Err(CpuError::ShapeMismatch {
            op: "planned U8 elementwise arity",
        }),
    }
}

fn input_len_matches(plan: &CpuKernelPlan, index: usize, actual: usize) -> bool {
    plan.input_shapes[index].is_some_and(|shape| shape.num_elements() == actual)
}

pub(crate) fn typed_input_matches(
    plan: &CpuKernelPlan,
    index: usize,
    dtype: DType,
    actual: usize,
) -> bool {
    plan.input(index).is_some_and(|desc| {
        desc.dtype == dtype
            && desc.is_contiguous()
            && desc.shape.checked_num_elements() == Some(actual)
            && plan.input_dtypes.get(index) == Some(&Some(dtype))
            && plan.input_shapes.get(index) == Some(&Some(desc.shape))
    })
}

pub(crate) fn typed_output_matches(
    plan: &CpuKernelPlan,
    index: usize,
    dtype: DType,
    shape: Shape,
    actual: usize,
) -> bool {
    plan.result_tensor(index).is_some_and(|desc| {
        desc.dtype == dtype
            && desc.shape == shape
            && desc.is_contiguous()
            && shape.checked_num_elements() == Some(actual)
    })
}

/// Pack three broadcast F32 planes into the explicitly planned `H × W × 3`
/// xyz-interleaved output. Scalar-only inputs may expand to the whole field.
/// Copies preserve NaN payloads and signed zero; no arithmetic or allocation
/// is performed.
pub fn invoke_interleave3_f32(
    plan: &CpuKernelPlan,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    dst: &mut [f32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Interleave3F32
        || plan.implementation != CpuImplementation::Scalar
        || plan.params() != KernelParams::None
        || plan.effects() != KernelEffects::PURE
        || plan.result_count() != 1
        || plan.output_dtype != DType::F32
        || plan.shape.rank() != 3
        || plan.shape.dim(2) != 3
        || !typed_output_matches(plan, 0, DType::F32, plan.shape, dst.len())
        || !typed_input_matches(plan, 0, DType::F32, x.len())
        || !typed_input_matches(plan, 1, DType::F32, y.len())
        || !typed_input_matches(plan, 2, DType::F32, z.len())
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned interleave3 F32",
        });
    }
    let field = axnn::shape![plan.shape.dim(0), plan.shape.dim(1)];
    let shapes = plan.input_shapes.map(Option::unwrap);
    if shapes
        .iter()
        .any(|shape| shape.broadcast_with(&field) != Some(field))
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned interleave3 broadcast",
        });
    }
    let elements = dst.len() / 3;
    if [x.len(), y.len(), z.len()]
        .iter()
        .all(|&len| len == 1 || len == elements)
    {
        // A validated broadcast with the full element count preserves flat
        // indexing even when leading singleton dimensions differ. Resolve
        // each scalar/dense choice once, not three times per output point.
        match (x.len() == 1, y.len() == 1, z.len() == 1) {
            (false, false, false) => interleave3_scalar_dense::<false, false, false>(x, y, z, dst),
            (false, false, true) => interleave3_scalar_dense::<false, false, true>(x, y, z, dst),
            (false, true, false) => interleave3_scalar_dense::<false, true, false>(x, y, z, dst),
            (false, true, true) => interleave3_scalar_dense::<false, true, true>(x, y, z, dst),
            (true, false, false) => interleave3_scalar_dense::<true, false, false>(x, y, z, dst),
            (true, false, true) => interleave3_scalar_dense::<true, false, true>(x, y, z, dst),
            (true, true, false) => interleave3_scalar_dense::<true, true, false>(x, y, z, dst),
            (true, true, true) => interleave3_scalar_dense::<true, true, true>(x, y, z, dst),
        }
        return Ok(());
    }
    for (index, xyz) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        xyz[0] = broadcast_at(x, shapes[0], field, index);
        xyz[1] = broadcast_at(y, shapes[1], field, index);
        xyz[2] = broadcast_at(z, shapes[2], field, index);
    }
    Ok(())
}

fn interleave3_scalar_dense<const X: bool, const Y: bool, const Z: bool>(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    dst: &mut [f32],
) {
    for (index, xyz) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        xyz[0] = x[if X { 0 } else { index }];
        xyz[1] = y[if Y { 0 } else { index }];
        xyz[2] = z[if Z { 0 } else { index }];
    }
}

/// Invoke an I32 identity plan without routing integer values through F32.
pub fn invoke_i32(plan: &CpuKernelPlan, input: &[i32], dst: &mut [i32]) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Identity
        || plan.implementation != CpuImplementation::Scalar
        || plan.params() != KernelParams::None
        || plan.effects() != KernelEffects::PURE
        || plan.result_count() != 1
        || plan.input_dtypes != [Some(DType::I32), None, None]
        || plan.input(1).is_some()
        || plan.input(2).is_some()
        || plan.output_dtype != DType::I32
        || !typed_input_matches(plan, 0, DType::I32, input.len())
        || plan.input_shapes[0] != Some(plan.shape)
        || !typed_output_matches(plan, 0, DType::I32, plan.shape, dst.len())
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned I32 identity",
        });
    }
    dst.copy_from_slice(input);
    Ok(())
}

/// Compare I32 values exactly into a caller-owned U8 mask (`0` or `1`).
/// Inputs right-align and broadcast like the F32 comparison contract.
pub fn invoke_compare_i32(
    plan: &CpuKernelPlan,
    lhs: &[i32],
    rhs: &[i32],
    dst: &mut [u8],
) -> Result<(), CpuError> {
    let SemanticOp::Compare(op) = plan.semantic else {
        return Err(CpuError::UnsupportedOp {
            op: "planned I32 compare",
            dtype: plan.output_dtype,
        });
    };
    if plan.implementation != CpuImplementation::Scalar
        || plan.params() != KernelParams::None
        || plan.effects() != KernelEffects::PURE
        || plan.result_count() != 1
        || plan.input_dtypes != [Some(DType::I32), Some(DType::I32), None]
        || plan.input(2).is_some()
        || plan.output_dtype != DType::U8
        || !typed_input_matches(plan, 0, DType::I32, lhs.len())
        || !typed_input_matches(plan, 1, DType::I32, rhs.len())
        || !typed_output_matches(plan, 0, DType::U8, plan.shape, dst.len())
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned I32 compare",
        });
    }
    let lhs_shape = plan.input_shapes[0].unwrap();
    let rhs_shape = plan.input_shapes[1].unwrap();
    if lhs_shape.broadcast_with(&rhs_shape) != Some(plan.shape) {
        return Err(CpuError::ShapeMismatch {
            op: "planned I32 compare broadcast",
        });
    }
    macro_rules! compare {
        ($operation:expr) => {
            map_binary_broadcast(lhs, lhs_shape, rhs, rhs_shape, plan.shape, dst, $operation)
        };
    }
    match op {
        CompareOp::Equal => compare!(|lhs, rhs| u8::from(lhs == rhs)),
        CompareOp::Greater => compare!(|lhs, rhs| u8::from(lhs > rhs)),
        CompareOp::GreaterEqual => compare!(|lhs, rhs| u8::from(lhs >= rhs)),
    }
    Ok(())
}

/// Invoke an F32 comparison plan into a caller-owned U8 mask (`0` or `1`).
pub fn invoke_compare_f32(
    plan: &CpuKernelPlan,
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [u8],
) -> Result<(), CpuError> {
    let SemanticOp::Compare(op) = plan.semantic else {
        return Err(CpuError::UnsupportedOp {
            op: "planned compare",
            dtype: plan.output_dtype,
        });
    };
    if plan.input_dtypes != [Some(DType::F32), Some(DType::F32), None]
        || plan.output_dtype != DType::U8
        || plan.implementation != CpuImplementation::Scalar
        || !input_len_matches(plan, 0, lhs.len())
        || !input_len_matches(plan, 1, rhs.len())
        || dst.len() != plan.shape.num_elements()
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned compare",
        });
    }

    let lhs_shape = plan.input_shapes[0].unwrap();
    let rhs_shape = plan.input_shapes[1].unwrap();
    // Each arm instantiates a distinct loop: there is no operation switch or
    // broadcast-shape inspection at every element in the scalar/dense cases.
    macro_rules! compare {
        ($operation:expr) => {
            map_binary_broadcast(lhs, lhs_shape, rhs, rhs_shape, plan.shape, dst, $operation)
        };
    }
    match op {
        CompareOp::Equal => compare!(|lhs, rhs| u8::from(lhs == rhs)),
        CompareOp::Greater => compare!(|lhs, rhs| u8::from(lhs > rhs)),
        CompareOp::GreaterEqual => compare!(|lhs, rhs| u8::from(lhs >= rhs)),
    }
    Ok(())
}

/// Invoke `U8 × F32 × F32 → F32` selection over caller-owned slices.
pub fn invoke_select_f32(
    plan: &CpuKernelPlan,
    mask: &[u8],
    on_true: &[f32],
    on_false: &[f32],
    dst: &mut [f32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Select
        || plan.input_dtypes != [Some(DType::U8), Some(DType::F32), Some(DType::F32)]
        || plan.output_dtype != DType::F32
        || plan.implementation != CpuImplementation::Scalar
        || !input_len_matches(plan, 0, mask.len())
        || !input_len_matches(plan, 1, on_true.len())
        || !input_len_matches(plan, 2, on_false.len())
        || dst.len() != plan.shape.num_elements()
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned select",
        });
    }

    let mask_shape = plan.input_shapes[0].unwrap();
    let true_shape = plan.input_shapes[1].unwrap();
    let false_shape = plan.input_shapes[2].unwrap();
    if mask.len() == 1 {
        // A uniform condition chooses one entire plane. Copying/filling keeps
        // NaN payloads and signed zero intact, including non-Boolean mask bytes.
        let (source, shape) = if mask[0] != 0 {
            (on_true, true_shape)
        } else {
            (on_false, false_shape)
        };
        if source.len() == dst.len() {
            dst.copy_from_slice(source);
        } else if source.len() == 1 {
            dst.fill(source[0]);
        } else {
            for (index, output) in dst.iter_mut().enumerate() {
                *output = broadcast_at(source, shape, plan.shape, index);
            }
        }
    } else {
        map_ternary_broadcast(
            (mask, mask_shape),
            (on_true, true_shape),
            (on_false, false_shape),
            plan.shape,
            dst,
            |mask, on_true, on_false| if mask != 0 { on_true } else { on_false },
        );
    }
    Ok(())
}

/// Invoke a selected rank-2 F32 matrix multiplication kernel.
///
/// Pure `MATMUL` plans expect a zero-initialized destination. GEMM plans may
/// instead name the destination as their third, accumulating input; both use
/// the same tuned implementation family.
pub fn invoke_gemm_f32(
    plan: &CpuKernelPlan,
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
) -> Result<(), CpuError> {
    let left = plan.input(0).ok_or(CpuError::ShapeMismatch {
        op: "planned matmul",
    })?;
    let right = plan.input(1).ok_or(CpuError::ShapeMismatch {
        op: "planned matmul",
    })?;
    let result_is_valid = match plan.result() {
        KernelResult::Tensor(desc) => {
            desc.shape == plan.shape && desc.dtype == DType::F32 && desc.is_contiguous()
        }
        KernelResult::Accumulate { input: 2 } => plan.input(2).is_some_and(|desc| {
            desc.shape == plan.shape && desc.dtype == DType::F32 && desc.is_contiguous()
        }),
        _ => false,
    };
    if plan.semantic != SemanticOp::MatMul
        || left.dtype != DType::F32
        || right.dtype != DType::F32
        || left.shape.rank() != 2
        || right.shape.rank() != 2
        || !left.is_contiguous()
        || !right.is_contiguous()
        || plan.shape.rank() != 2
        || left.shape.dim(1) != right.shape.dim(0)
        || plan.shape.dim(0) != left.shape.dim(0)
        || plan.shape.dim(1) != right.shape.dim(1)
        || lhs.len() != left.shape.num_elements()
        || rhs.len() != right.shape.num_elements()
        || dst.len() != plan.shape.num_elements()
        || !result_is_valid
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned matmul",
        });
    }

    let m = left.shape.dim(0);
    let k = left.shape.dim(1);
    let n = right.shape.dim(1);
    match plan.implementation {
        CpuImplementation::Qdot => {
            return Err(CpuError::UnsupportedOp {
                op: "kernel",
                dtype: plan.output_dtype,
            });
        }
        CpuImplementation::Scalar => crate::arch::scalar::gemm::gemm_f32(lhs, rhs, dst, m, n, k),
        // `arch::gemm` preserves the artifact's compile-time target tier:
        // portable SIMD elsewhere, NEON/SME on AArch64. The explicit AVX2
        // candidate remains separately priced below on x86_64.
        CpuImplementation::PortableSimd => crate::arch::gemm::gemm_f32(lhs, rhs, dst, m, n, k),
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        CpuImplementation::Avx2 => crate::arch::avx2::gemm::gemm_f32(lhs, rhs, dst, m, n, k),
    }
    Ok(())
}

fn broadcast_at<T: Copy>(
    input: &[T],
    input_shape: Shape,
    output_shape: Shape,
    output_index: usize,
) -> T {
    // Private, post-admission path. The public checked index helper remains
    // fail-closed for arbitrary shapes; invocation already checked these.
    if input.len() == 1 {
        return input[0];
    }
    if input_shape == output_shape {
        return input[output_index];
    }
    if input_shape.rank() == 2 && output_shape.rank() == 2 {
        let width = output_shape.dim(1);
        if input_shape.dim(0) == 1 && input_shape.dim(1) == width {
            return input[output_index % width];
        }
        if input_shape.dim(1) == 1 && input_shape.dim(0) == output_shape.dim(0) {
            return input[output_index / width];
        }
    }
    let index = broadcast_index(input_shape, output_shape, output_index)
        .expect("kernel plan contains a validated broadcast");
    *input
        .get(index)
        .expect("kernel plan input length was validated")
}

/// Map a flat output index to a right-aligned broadcast input index.
///
/// Returns `None` for rank inversion, incompatible shapes, an out-of-range
/// output index, zero-sized indexing, or index arithmetic overflow.
pub(crate) fn broadcast_index(
    input: Shape,
    output: Shape,
    mut output_index: usize,
) -> Option<usize> {
    if input.rank() > output.rank()
        || input.broadcast_with(&output) != Some(output)
        || output_index >= output.checked_num_elements()?
    {
        return None;
    }
    if input == output {
        return Some(output_index);
    }

    let rank_offset = output.rank() - input.rank();
    let mut input_index = 0usize;
    let mut input_stride = 1usize;
    for output_axis in (0..output.rank()).rev() {
        let output_dim = output.dims()[output_axis];
        let coordinate = output_index % output_dim;
        output_index /= output_dim;
        if output_axis >= rank_offset {
            let input_dim = input.dims()[output_axis - rank_offset];
            if input_dim != 1 {
                input_index = input_index.checked_add(coordinate.checked_mul(input_stride)?)?;
            }
            input_stride = input_stride.checked_mul(input_dim)?;
        }
    }
    Some(input_index)
}

fn invoke_scalar(
    plan: &CpuKernelPlan,
    first: &[f32],
    second: Option<&[f32]>,
    third: Option<&[f32]>,
    dst: &mut [f32],
) -> Result<(), CpuError> {
    match plan.semantic {
        SemanticOp::Identity => {
            dst.copy_from_slice(first);
            Ok(())
        }
        SemanticOp::Unary(op) => {
            match op {
                UnaryOp::Relu => arch::scalar::vec::relu_f32(first, dst),
                UnaryOp::Neg => arch::scalar::vec::neg_f32(first, dst),
                UnaryOp::Sqrt => arch::scalar::vec::sqrt_f32(first, dst),
                UnaryOp::Exp => arch::scalar::vec::exp_f32(first, dst),
                UnaryOp::Tanh => arch::scalar::vec::tanh_f32(first, dst),
                UnaryOp::Sigmoid => arch::scalar::vec::sigmoid_f32(first, dst),
                UnaryOp::Silu => arch::scalar::vec::silu_f32(first, dst),
                UnaryOp::Gelu => arch::scalar::vec::gelu_f32(first, dst),
                _ => map_unary_f32(op, first, dst),
            }
            Ok(())
        }
        SemanticOp::Binary(op) => {
            let second = second.ok_or(CpuError::ShapeMismatch {
                op: "planned binary",
            })?;
            if first.len() == dst.len() && second.len() == dst.len() {
                match op {
                    BinaryOp::Add => arch::scalar::vec::add_f32(first, second, dst),
                    BinaryOp::Sub => arch::scalar::vec::sub_f32(first, second, dst),
                    BinaryOp::Mul => arch::scalar::vec::mul_f32(first, second, dst),
                    BinaryOp::Div => arch::scalar::vec::div_f32(first, second, dst),
                    _ => map_binary_f32(
                        op,
                        first,
                        plan.input_shapes[0].unwrap(),
                        second,
                        plan.input_shapes[1].unwrap(),
                        plan.shape,
                        dst,
                    ),
                }
            } else {
                map_binary_f32(
                    op,
                    first,
                    plan.input_shapes[0].unwrap(),
                    second,
                    plan.input_shapes[1].unwrap(),
                    plan.shape,
                    dst,
                );
            }
            Ok(())
        }
        SemanticOp::Ternary(op) => {
            let second = second.ok_or(CpuError::ShapeMismatch {
                op: "planned ternary",
            })?;
            let third = third.ok_or(CpuError::ShapeMismatch {
                op: "planned ternary",
            })?;
            map_ternary_f32(
                op,
                first,
                plan.input_shapes[0].unwrap(),
                second,
                plan.input_shapes[1].unwrap(),
                third,
                plan.input_shapes[2].unwrap(),
                plan.shape,
                dst,
            );
            Ok(())
        }
        _ => Err(CpuError::UnsupportedOp {
            op: "planned elementwise",
            dtype: DType::F32,
        }),
    }
}

fn invoke_portable(
    semantic: SemanticOp,
    first: &[f32],
    second: Option<&[f32]>,
    _third: Option<&[f32]>,
    dst: &mut [f32],
) -> Result<(), CpuError> {
    match semantic {
        SemanticOp::Unary(op) => {
            match op {
                UnaryOp::Relu => arch::portable::vec::relu_f32(first, dst),
                UnaryOp::Neg => arch::portable::vec::neg_f32(first, dst),
                UnaryOp::Sqrt => arch::portable::vec::sqrt_f32(first, dst),
                UnaryOp::Exp => arch::portable::vec::exp_f32(first, dst),
                UnaryOp::Tanh => arch::portable::vec::tanh_f32(first, dst),
                UnaryOp::Sigmoid => arch::portable::vec::sigmoid_f32(first, dst),
                UnaryOp::Silu => arch::portable::vec::silu_f32(first, dst),
                UnaryOp::Gelu => arch::portable::vec::gelu_f32(first, dst),
                _ => return unsupported(semantic),
            }
            Ok(())
        }
        SemanticOp::Binary(op) => {
            let second = second.ok_or(CpuError::ShapeMismatch {
                op: "planned binary",
            })?;
            match op {
                BinaryOp::Add => arch::portable::vec::add_f32(first, second, dst),
                BinaryOp::Sub => arch::portable::vec::sub_f32(first, second, dst),
                BinaryOp::Mul => arch::portable::vec::mul_f32(first, second, dst),
                BinaryOp::Div => arch::portable::vec::div_f32(first, second, dst),
                _ => return unsupported(semantic),
            }
            Ok(())
        }
        _ => unsupported(semantic),
    }
}

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
fn invoke_avx2(
    semantic: SemanticOp,
    first: &[f32],
    second: Option<&[f32]>,
    _third: Option<&[f32]>,
    dst: &mut [f32],
) -> Result<(), CpuError> {
    match semantic {
        SemanticOp::Unary(op) => {
            match op {
                UnaryOp::Relu => arch::avx2::vec::relu_f32(first, dst),
                UnaryOp::Neg => arch::avx2::vec::neg_f32(first, dst),
                UnaryOp::Sqrt => arch::avx2::vec::sqrt_f32(first, dst),
                UnaryOp::Exp => arch::avx2::vec::exp_f32(first, dst),
                UnaryOp::Tanh => arch::avx2::vec::tanh_f32(first, dst),
                UnaryOp::Sigmoid => arch::avx2::vec::sigmoid_f32(first, dst),
                UnaryOp::Silu => arch::avx2::vec::silu_f32(first, dst),
                UnaryOp::Gelu => arch::avx2::vec::gelu_f32(first, dst),
                _ => return unsupported(semantic),
            }
            Ok(())
        }
        SemanticOp::Binary(op) => {
            let second = second.ok_or(CpuError::ShapeMismatch {
                op: "planned binary",
            })?;
            match op {
                BinaryOp::Add => arch::avx2::vec::add_f32(first, second, dst),
                BinaryOp::Sub => arch::avx2::vec::sub_f32(first, second, dst),
                BinaryOp::Mul => arch::avx2::vec::mul_f32(first, second, dst),
                BinaryOp::Div => arch::avx2::vec::div_f32(first, second, dst),
                _ => return unsupported(semantic),
            }
            Ok(())
        }
        _ => unsupported(semantic),
    }
}

fn unsupported<T>(semantic: SemanticOp) -> Result<T, CpuError> {
    let op = match semantic {
        SemanticOp::Unary(_) => "planned unary",
        SemanticOp::Binary(_) => "planned binary",
        SemanticOp::Ternary(_) => "planned ternary",
        _ => "planned kernel",
    };
    Err(CpuError::UnsupportedOp {
        op,
        dtype: DType::F32,
    })
}

fn map_unary_f32(op: UnaryOp, src: &[f32], dst: &mut [f32]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        *out = unary_value_f32(op, x);
    }
}

fn map_binary_f32(
    op: BinaryOp,
    lhs: &[f32],
    lhs_shape: Shape,
    rhs: &[f32],
    rhs_shape: Shape,
    output_shape: Shape,
    dst: &mut [f32],
) {
    macro_rules! binary {
        ($operation:expr) => {
            map_binary_broadcast(
                lhs,
                lhs_shape,
                rhs,
                rhs_shape,
                output_shape,
                dst,
                $operation,
            )
        };
    }
    // Function items and closures are monomorphized, not selected as a
    // function pointer. Preserve the original operation and operand order.
    match op {
        BinaryOp::Add => binary!(|lhs, rhs| lhs + rhs),
        BinaryOp::Sub => binary!(|lhs, rhs| lhs - rhs),
        BinaryOp::Mul => binary!(|lhs, rhs| lhs * rhs),
        BinaryOp::Div => binary!(|lhs, rhs| lhs / rhs),
        BinaryOp::Min => binary!(minimum_f32),
        BinaryOp::Max => binary!(maximum_f32),
        BinaryOp::Atan2 => binary!(axmath::atan2_f32),
        BinaryOp::Pow => binary!(pow_f32),
        BinaryOp::Rem => binary!(remainder_f32),
    }
}

pub(crate) fn map_binary_broadcast<L: Copy, R: Copy, O>(
    lhs: &[L],
    lhs_shape: Shape,
    rhs: &[R],
    rhs_shape: Shape,
    output_shape: Shape,
    dst: &mut [O],
    operation: impl Fn(L, R) -> O,
) {
    // Admission and invoke_f32 already validated shapes/lengths. Keep the
    // overwhelmingly common field/scalar and dense cases out of the generic
    // per-element rank walk (which revalidated the same broadcast every time).
    if lhs.len() == 1 && rhs.len() == dst.len() {
        let left = lhs[0];
        for (out, &right) in dst.iter_mut().zip(rhs) {
            *out = operation(left, right);
        }
        return;
    }
    if rhs.len() == 1 && lhs.len() == dst.len() {
        let right = rhs[0];
        for (out, &left) in dst.iter_mut().zip(lhs) {
            *out = operation(left, right);
        }
        return;
    }
    if lhs.len() == dst.len() && rhs.len() == dst.len() {
        for ((out, &left), &right) in dst.iter_mut().zip(lhs).zip(rhs) {
            *out = operation(left, right);
        }
        return;
    }
    if let Some((rows, width)) = field_2d(output_shape, dst.len())
        && let (Some(left), Some(right)) = (
            Lane2::of(lhs_shape, rows, width),
            Lane2::of(rhs_shape, rows, width),
        )
    {
        for (row, out) in dst.chunks_exact_mut(width).enumerate() {
            let (a, b) = (left.row(lhs, row, width), right.row(rhs, row, width));
            match (a.len() == 1, b.len() == 1) {
                (false, false) => map_binary_contiguous::<false, false, _, _, _>(a, b, out, &operation),
                (false, true) => map_binary_contiguous::<false, true, _, _, _>(a, b, out, &operation),
                (true, false) => map_binary_contiguous::<true, false, _, _, _>(a, b, out, &operation),
                (true, true) => map_binary_contiguous::<true, true, _, _, _>(a, b, out, &operation),
            }
        }
        return;
    }
    for (index, out) in dst.iter_mut().enumerate() {
        *out = operation(
            broadcast_at(lhs, lhs_shape, output_shape, index),
            broadcast_at(rhs, rhs_shape, output_shape, index),
        );
    }
}

/// The `[rows, width]` of a rank-2 output field, when it has one.
fn field_2d(output: Shape, len: usize) -> Option<(usize, usize)> {
    (output.rank() == 2 && output.dim(1) != 0 && output.dim(0) * output.dim(1) == len)
        .then(|| (output.dim(0), output.dim(1)))
}

/// How one operand of a rank-2 `[rows, width]` field is laid out. Shader
/// graphs mix full fields with the tile's position axes (`[1, width]` and
/// `[rows, 1]`) and with constants; each row of such an operation then reads
/// either a contiguous slice or one repeated value, so the element loop stays
/// contiguous instead of walking broadcast indices per element.
#[derive(Clone, Copy)]
enum Lane2 {
    Scalar,
    Dense,
    Row,
    Column,
}

impl Lane2 {
    fn of(shape: Shape, rows: usize, width: usize) -> Option<Self> {
        let dims = shape.dims();
        match *dims {
            [] => Some(Self::Scalar),
            [w] if w == width => Some(Self::Row),
            [1] => Some(Self::Scalar),
            [r, w] if r == rows && w == width => Some(Self::Dense),
            [1, w] if w == width => Some(Self::Row),
            [r, 1] if r == rows => Some(Self::Column),
            [1, 1] => Some(Self::Scalar),
            _ => None,
        }
    }

    /// The operand's values for output row `row`: `width` elements, or one
    /// element repeated across the row.
    #[inline]
    fn row<T>(self, source: &[T], row: usize, width: usize) -> &[T] {
        match self {
            Self::Scalar => &source[..1],
            Self::Dense => &source[row * width..(row + 1) * width],
            Self::Row => &source[..width],
            Self::Column => &source[row..row + 1],
        }
    }
}

#[inline]
fn map_binary_contiguous<const LEFT_SCALAR: bool, const RIGHT_SCALAR: bool, L: Copy, R: Copy, O>(
    lhs: &[L],
    rhs: &[R],
    dst: &mut [O],
    operation: &impl Fn(L, R) -> O,
) {
    for (index, out) in dst.iter_mut().enumerate() {
        *out = operation(
            lhs[if LEFT_SCALAR { 0 } else { index }],
            rhs[if RIGHT_SCALAR { 0 } else { index }],
        );
    }
}

fn map_ternary_f32(
    op: TernaryOp,
    first: &[f32],
    first_shape: Shape,
    second: &[f32],
    second_shape: Shape,
    third: &[f32],
    third_shape: Shape,
    output_shape: Shape,
    dst: &mut [f32],
) {
    macro_rules! ternary {
        ($operation:expr) => {
            map_ternary_broadcast(
                (first, first_shape),
                (second, second_shape),
                (third, third_shape),
                output_shape,
                dst,
                $operation,
            )
        };
    }
    match op {
        TernaryOp::Clamp => ternary!(|value, low, high| minimum_f32(maximum_f32(value, low), high)),
        // Keep the multiply/add rounding boundary. This is not a fused kernel.
        TernaryOp::Fma => ternary!(|first, second, third| first * second + third),
    }
}

pub(crate) fn map_ternary_broadcast<A: Copy, B: Copy, C: Copy, O>(
    first: (&[A], Shape),
    second: (&[B], Shape),
    third: (&[C], Shape),
    output_shape: Shape,
    dst: &mut [O],
    operation: impl Fn(A, B, C) -> O,
) {
    let (first, first_shape) = first;
    let (second, second_shape) = second;
    let (third, third_shape) = third;
    if [first.len(), second.len(), third.len()]
        .iter()
        .all(|&len| len == 1 || len == dst.len())
    {
        // Const choices keep scalar-vs-dense indexing outside the element
        // loop while retaining a direct, monomorphized operation in it.
        macro_rules! contiguous {
            ($a:literal, $b:literal, $c:literal) => {
                map_ternary_contiguous::<$a, $b, $c, _, _, _, _>(
                    first, second, third, dst, operation,
                )
            };
        }
        match (first.len() == 1, second.len() == 1, third.len() == 1) {
            (false, false, false) => contiguous!(false, false, false),
            (false, false, true) => contiguous!(false, false, true),
            (false, true, false) => contiguous!(false, true, false),
            (false, true, true) => contiguous!(false, true, true),
            (true, false, false) => contiguous!(true, false, false),
            (true, false, true) => contiguous!(true, false, true),
            (true, true, false) => contiguous!(true, true, false),
            (true, true, true) => contiguous!(true, true, true),
        }
        return;
    }
    if let Some((rows, width)) = field_2d(output_shape, dst.len())
        && let (Some(a), Some(b), Some(c)) = (
            Lane2::of(first_shape, rows, width),
            Lane2::of(second_shape, rows, width),
            Lane2::of(third_shape, rows, width),
        )
    {
        for (row, out) in dst.chunks_exact_mut(width).enumerate() {
            let (first, second, third) = (a.row(first, row, width), b.row(second, row, width), c.row(third, row, width));
            macro_rules! per_row {
                ($a:literal, $b:literal, $c:literal) => {
                    map_ternary_contiguous::<$a, $b, $c, _, _, _, _>(first, second, third, out, &operation)
                };
            }
            match (first.len() == 1, second.len() == 1, third.len() == 1) {
                (false, false, false) => per_row!(false, false, false),
                (false, false, true) => per_row!(false, false, true),
                (false, true, false) => per_row!(false, true, false),
                (false, true, true) => per_row!(false, true, true),
                (true, false, false) => per_row!(true, false, false),
                (true, false, true) => per_row!(true, false, true),
                (true, true, false) => per_row!(true, true, false),
                (true, true, true) => per_row!(true, true, true),
            }
        }
        return;
    }
    for (index, out) in dst.iter_mut().enumerate() {
        *out = operation(
            broadcast_at(first, first_shape, output_shape, index),
            broadcast_at(second, second_shape, output_shape, index),
            broadcast_at(third, third_shape, output_shape, index),
        );
    }
}

#[inline]
fn map_ternary_contiguous<
    const FIRST_SCALAR: bool,
    const SECOND_SCALAR: bool,
    const THIRD_SCALAR: bool,
    A: Copy,
    B: Copy,
    C: Copy,
    O,
>(
    first: &[A],
    second: &[B],
    third: &[C],
    dst: &mut [O],
    operation: impl Fn(A, B, C) -> O,
) {
    for (index, out) in dst.iter_mut().enumerate() {
        *out = operation(
            first[if FIRST_SCALAR { 0 } else { index }],
            second[if SECOND_SCALAR { 0 } else { index }],
            third[if THIRD_SCALAR { 0 } else { index }],
        );
    }
}

#[cfg(test)]
mod broadcast_fast_path_tests {
    use super::*;
    use axnn::{TensorDesc, shape};

    const EDGE_VALUES: [f32; 16] = [
        -0.0,
        0.0,
        -1.0,
        1.0,
        -2.5,
        3.25,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffc0_5678),
        f32::from_bits(0x7f80_1234),
        f32::from_bits(0xff80_5678),
    ];

    fn binary_shapes(width: usize) -> [(Shape, Shape); 7] {
        [
            (shape![], shape![width]),
            (shape![width], shape![]),
            (shape![width], shape![width]),
            (shape![1, width], shape![2, 1]),
            (shape![2, 1], shape![1, width]),
            // Equal lengths with different ranks are still contiguous.
            (shape![width], shape![1, width]),
            (shape![2, 1, width], shape![1, 3, 1]),
        ]
    }

    fn ternary_shapes(width: usize) -> [(Shape, Shape, Shape); 12] {
        let scalar = shape![];
        let dense = shape![width];
        [
            (dense, dense, dense),
            (dense, dense, scalar),
            (dense, scalar, dense),
            (dense, scalar, scalar),
            (scalar, dense, dense),
            (scalar, dense, scalar),
            (scalar, scalar, dense),
            (scalar, scalar, scalar),
            (shape![2, 1], shape![1, width], scalar),
            // A uniform mask can choose a row while the other arm establishes
            // the full output height, or choose the column instead.
            (scalar, shape![1, width], shape![2, 1]),
            (shape![width], shape![1, width], shape![1, 1, width]),
            (shape![2, 1, width], shape![1, 3, 1], scalar),
        ]
    }

    fn assert_same_float(actual: f32, expected: f32, preserve_nan_bits: bool) {
        if expected.is_nan() && !preserve_nan_bits {
            // Arithmetic NaN payloads are not specified. Min/max and selection
            // explicitly preserve a chosen operand, so those are bit-exact.
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn scalar_dense_and_two_axis_paths_match_checked_indexing() {
        let output = shape![3, 4];
        for (left, right) in [
            (shape![1, 1], output),
            (output, shape![1, 1]),
            (output, output),
            (shape![1, 4], shape![3, 1]),
            (shape![3, 1], shape![1, 4]),
        ] {
            let mut lhs = [0.0; 12];
            let mut rhs = [0.0; 12];
            for i in 0..12 {
                lhs[i] = i as f32 - 5.0;
                rhs[i] = 0.25 - i as f32;
            }
            lhs[0] = -0.0;
            rhs[0] = f32::NAN;
            let lhs = &lhs[..left.num_elements()];
            let rhs = &rhs[..right.num_elements()];
            for op in [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Min,
                BinaryOp::Max,
            ] {
                let mut actual = [0.0; 12];
                map_binary_f32(op, lhs, left, rhs, right, output, &mut actual);
                for (i, &value) in actual.iter().enumerate() {
                    let expected = binary_value_f32(
                        op,
                        lhs[broadcast_index(left, output, i).unwrap()],
                        rhs[broadcast_index(right, output, i).unwrap()],
                    );
                    assert_eq!(value.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn binary_and_compare_specializations_match_checked_broadcast() {
        let lhs: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[i % EDGE_VALUES.len()]);
        let rhs: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[(i + 5) % EDGE_VALUES.len()]);
        // Empty fields, vector-width boundaries, and tails use the same ABI.
        for width in [0, 1, 3, 4, 7, 16, 17, 31, 32, 65] {
            for (left, right) in binary_shapes(width) {
                let left_desc = TensorDesc::new(left, DType::F32);
                let right_desc = TensorDesc::new(right, DType::F32);
                let lhs = &lhs[..left.num_elements()];
                let rhs = &rhs[..right.num_elements()];
                for op in [
                    BinaryOp::Add,
                    BinaryOp::Sub,
                    BinaryOp::Mul,
                    BinaryOp::Div,
                    BinaryOp::Min,
                    BinaryOp::Max,
                    BinaryOp::Atan2,
                    BinaryOp::Pow,
                    BinaryOp::Rem,
                ] {
                    let request = KernelRequest::binary(op, &left_desc, &right_desc).unwrap();
                    let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
                    let mut actual = [42.0; 512];
                    let actual = &mut actual[..plan.shape.num_elements()];
                    invoke_f32(&plan, lhs, Some(rhs), None, actual).unwrap();
                    for (i, &value) in actual.iter().enumerate() {
                        let expected = binary_value_f32(
                            op,
                            lhs[broadcast_index(left, plan.shape, i).unwrap()],
                            rhs[broadcast_index(right, plan.shape, i).unwrap()],
                        );
                        assert_same_float(
                            value,
                            expected,
                            matches!(op, BinaryOp::Min | BinaryOp::Max),
                        );
                    }
                }
                for op in [
                    CompareOp::Equal,
                    CompareOp::Greater,
                    CompareOp::GreaterEqual,
                ] {
                    let request = KernelRequest::compare(op, &left_desc, &right_desc).unwrap();
                    let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
                    let mut actual = [255; 512];
                    let actual = &mut actual[..plan.shape.num_elements()];
                    invoke_compare_f32(&plan, lhs, rhs, actual).unwrap();
                    for (i, &value) in actual.iter().enumerate() {
                        let left = lhs[broadcast_index(left, plan.shape, i).unwrap()];
                        let right = rhs[broadcast_index(right, plan.shape, i).unwrap()];
                        let expected = match op {
                            CompareOp::Equal => left == right,
                            CompareOp::Greater => left > right,
                            CompareOp::GreaterEqual => left >= right,
                        };
                        assert_eq!(value, u8::from(expected));
                    }
                }
            }
        }
    }

    #[test]
    fn min_max_preserve_operand_order_nan_payloads_and_signed_zero() {
        let dense: [f32; 65] = core::array::from_fn(|i| EDGE_VALUES[i % EDGE_VALUES.len()]);
        for scalar in EDGE_VALUES {
            let scalar = [scalar];
            for (lhs, left, rhs, right) in [
                (&scalar[..], shape![], &dense[..], shape![65]),
                (&dense[..], shape![65], &scalar[..], shape![]),
            ] {
                for op in [BinaryOp::Min, BinaryOp::Max] {
                    let mut actual = [42.0; 65];
                    map_binary_f32(op, lhs, left, rhs, right, shape![65], &mut actual);
                    for (i, &value) in actual.iter().enumerate() {
                        let expected = binary_value_f32(
                            op,
                            lhs[broadcast_index(left, shape![65], i).unwrap()],
                            rhs[broadcast_index(right, shape![65], i).unwrap()],
                        );
                        assert_same_float(value, expected, true);
                    }
                }
            }
        }
    }

    #[test]
    fn ternary_specializations_match_checked_broadcast() {
        let first: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[i % EDGE_VALUES.len()]);
        let second: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[(i + 5) % EDGE_VALUES.len()]);
        let third: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[(i + 9) % EDGE_VALUES.len()]);
        for width in [0, 1, 3, 4, 7, 16, 17, 31, 32, 65] {
            for (a, b, c) in ternary_shapes(width) {
                let first = &first[..a.num_elements()];
                let second = &second[..b.num_elements()];
                let third = &third[..c.num_elements()];
                for op in [TernaryOp::Clamp, TernaryOp::Fma] {
                    let request = KernelRequest::ternary(
                        op,
                        &TensorDesc::new(a, DType::F32),
                        &TensorDesc::new(b, DType::F32),
                        &TensorDesc::new(c, DType::F32),
                    )
                    .unwrap();
                    let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
                    let mut actual = [42.0; 512];
                    let actual = &mut actual[..plan.shape.num_elements()];
                    invoke_f32(&plan, first, Some(second), Some(third), actual).unwrap();
                    for (i, &value) in actual.iter().enumerate() {
                        let expected = ternary_value_f32(
                            op,
                            first[broadcast_index(a, plan.shape, i).unwrap()],
                            second[broadcast_index(b, plan.shape, i).unwrap()],
                            third[broadcast_index(c, plan.shape, i).unwrap()],
                        );
                        assert_same_float(value, expected, op == TernaryOp::Clamp);
                    }
                }
            }
        }
    }

    #[test]
    fn multiply_add_specialization_keeps_two_roundings() {
        // (1 + 2^-23) * (1 - 2^-23) rounds to 1 before subtracting 1.
        // Fusing the operations would instead produce -2^-46.
        let first = [core::hint::black_box(f32::from_bits(0x3f80_0001)); 33];
        let second = [core::hint::black_box(f32::from_bits(0x3f7f_fffe)); 33];
        let third = [-1.0; 33];
        let mut actual = [42.0; 33];
        for (a, b, c) in ternary_shapes(33).into_iter().take(8) {
            let shape = a.broadcast_with(&b).unwrap().broadcast_with(&c).unwrap();
            let actual = &mut actual[..shape.num_elements()];
            map_ternary_f32(
                TernaryOp::Fma,
                &first[..a.num_elements()],
                a,
                &second[..b.num_elements()],
                b,
                &third[..c.num_elements()],
                c,
                shape,
                actual,
            );
            assert!(actual.iter().all(|value| value.to_bits() == 0));
        }
    }

    #[test]
    fn select_specializations_preserve_selected_bits_and_non_boolean_masks() {
        let on_true: [f32; 512] = core::array::from_fn(|i| EDGE_VALUES[i % EDGE_VALUES.len()]);
        let on_false: [f32; 512] =
            core::array::from_fn(|i| EDGE_VALUES[(i + 7) % EDGE_VALUES.len()]);
        for mask_offset in 0..4 {
            let mask: [u8; 512] = core::array::from_fn(|i| [0, 1, 2, 255][(i + mask_offset) % 4]);
            for width in [0, 1, 3, 4, 7, 16, 17, 31, 32, 65] {
                for (m, t, f) in ternary_shapes(width) {
                    let request = KernelRequest::select(
                        &TensorDesc::new(m, DType::U8),
                        &TensorDesc::new(t, DType::F32),
                        &TensorDesc::new(f, DType::F32),
                    )
                    .unwrap();
                    let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
                    let mut actual = [42.0; 512];
                    let actual = &mut actual[..plan.shape.num_elements()];
                    invoke_select_f32(
                        &plan,
                        &mask[..m.num_elements()],
                        &on_true[..t.num_elements()],
                        &on_false[..f.num_elements()],
                        actual,
                    )
                    .unwrap();
                    for (i, &value) in actual.iter().enumerate() {
                        let expected = if mask[broadcast_index(m, plan.shape, i).unwrap()] != 0 {
                            on_true[broadcast_index(t, plan.shape, i).unwrap()]
                        } else {
                            on_false[broadcast_index(f, plan.shape, i).unwrap()]
                        };
                        assert_same_float(value, expected, true);
                    }
                }
            }
        }
    }
}

pub(crate) fn unary_value_f32(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Abs => axmath::abs_f32(x),
        UnaryOp::Acos => axmath::acos_f32(x),
        UnaryOp::Asin => axmath::asin_f32(x),
        UnaryOp::Atan => axmath::atan_f32(x),
        UnaryOp::Ceil => axmath::ceil_f32(x),
        UnaryOp::Cos => axmath::cos_f32(x),
        UnaryOp::Relu => x.max(0.0),
        UnaryOp::Neg => -x,
        UnaryOp::Sqrt => axmath::sqrt_f32(x),
        UnaryOp::Exp => axmath::exp_f32(x),
        UnaryOp::Exp2 => axmath::exp2_f32(x),
        UnaryOp::Floor => axmath::floor_f32(x),
        UnaryOp::Fract => x - axmath::floor_f32(x),
        UnaryOp::InverseSqrt => 1.0 / axmath::sqrt_f32(x),
        UnaryOp::Log => axmath::log_f32(x),
        UnaryOp::Log2 => axmath::log2_f32(x),
        UnaryOp::Reciprocal => 1.0 / x,
        UnaryOp::Round => round_ties_even_f32(x),
        UnaryOp::Sign => sign_f32(x),
        UnaryOp::Sin => axmath::sin_f32(x),
        UnaryOp::Tan => axmath::tan_f32(x),
        UnaryOp::Tanh => axmath::tanh_f32(x),
        UnaryOp::Trunc => axmath::trunc_f32(x),
        UnaryOp::Sigmoid => axmath::sigmoid_f32(x),
        UnaryOp::Silu => axmath::silu_f32(x),
        UnaryOp::Gelu => axmath::gelu_f32(x),
        UnaryOp::Softplus => axmath::softplus_f32(x),
    }
}

pub(crate) fn unary_value_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Abs => axmath::abs_f64(x),
        UnaryOp::Acos => axmath::acos_f64(x),
        UnaryOp::Asin => axmath::asin_f64(x),
        UnaryOp::Atan => axmath::atan_f64(x),
        UnaryOp::Ceil => axmath::ceil_f64(x),
        UnaryOp::Cos => axmath::cos_f64(x),
        UnaryOp::Relu => x.max(0.0),
        UnaryOp::Neg => -x,
        UnaryOp::Sqrt => axmath::sqrt_f64(x),
        UnaryOp::Exp => axmath::exp_f64(x),
        UnaryOp::Exp2 => axmath::exp2_f64(x),
        UnaryOp::Floor => axmath::floor_f64(x),
        UnaryOp::Fract => x - axmath::floor_f64(x),
        UnaryOp::InverseSqrt => 1.0 / axmath::sqrt_f64(x),
        UnaryOp::Log => axmath::log_f64(x),
        UnaryOp::Log2 => axmath::log2_f64(x),
        UnaryOp::Reciprocal => 1.0 / x,
        UnaryOp::Round => round_ties_even_f64(x),
        UnaryOp::Sign => sign_f64(x),
        UnaryOp::Sin => axmath::sin_f64(x),
        UnaryOp::Tan => axmath::tan_f64(x),
        UnaryOp::Tanh => axmath::tanh_f64(x),
        UnaryOp::Trunc => axmath::trunc_f64(x),
        UnaryOp::Sigmoid => axmath::sigmoid_f64(x),
        UnaryOp::Silu => axmath::silu_f64(x),
        UnaryOp::Gelu => axmath::gelu_f64(x),
        UnaryOp::Softplus => axmath::softplus_f64(x),
    }
}

pub(crate) fn binary_value_f32(op: BinaryOp, lhs: f32, rhs: f32) -> f32 {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Min => minimum_f32(lhs, rhs),
        BinaryOp::Max => maximum_f32(lhs, rhs),
        BinaryOp::Atan2 => axmath::atan2_f32(lhs, rhs),
        BinaryOp::Pow => pow_f32(lhs, rhs),
        BinaryOp::Rem => remainder_f32(lhs, rhs),
    }
}

pub(crate) fn binary_value_f64(op: BinaryOp, lhs: f64, rhs: f64) -> f64 {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Min => minimum_f64(lhs, rhs),
        BinaryOp::Max => maximum_f64(lhs, rhs),
        BinaryOp::Atan2 => axmath::atan2_f64(lhs, rhs),
        BinaryOp::Pow => pow_f64(lhs, rhs),
        BinaryOp::Rem => remainder_f64(lhs, rhs),
    }
}

pub(crate) fn ternary_value_f32(op: TernaryOp, first: f32, second: f32, third: f32) -> f32 {
    match op {
        TernaryOp::Clamp => minimum_f32(maximum_f32(first, second), third),
        // WGSL permits either fused or unfused evaluation. The scalar baseline
        // keeps the multiply/add rounding boundary; a future FMA candidate can
        // advertise a different implementation under the same semantic op.
        TernaryOp::Fma => first * second + third,
    }
}

pub(crate) fn ternary_value_f64(op: TernaryOp, first: f64, second: f64, third: f64) -> f64 {
    match op {
        TernaryOp::Clamp => minimum_f64(maximum_f64(first, second), third),
        TernaryOp::Fma => first * second + third,
    }
}

fn sign_f32(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn sign_f64(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// WGSL-compatible nearest-even rounding. `axmath::round_*` intentionally has
/// ties-away semantics, so it cannot be used for this operation.
fn round_ties_even_f32(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 || axmath::abs_f32(x) >= 8_388_608.0 {
        return x;
    }
    let floor = axmath::floor_f32(x);
    let fraction = x - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || (floor as i64) & 1 != 0 {
        floor + 1.0
    } else {
        floor
    };
    if rounded == 0.0 {
        axmath::copysign_f32(0.0, x)
    } else {
        rounded
    }
}

fn round_ties_even_f64(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 || axmath::abs_f64(x) >= 4_503_599_627_370_496.0 {
        return x;
    }
    let floor = axmath::floor_f64(x);
    let fraction = x - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || (floor as i64) & 1 != 0 {
        floor + 1.0
    } else {
        floor
    };
    if rounded == 0.0 {
        axmath::copysign_f64(0.0, x)
    } else {
        rounded
    }
}

fn minimum_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.min(rhs)
    }
}

fn maximum_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.max(rhs)
    }
}

fn minimum_f64(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.min(rhs)
    }
}

fn maximum_f64(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.max(rhs)
    }
}

/// FP-profile power. Small integral exponents preserve negative-base behavior;
/// the general real-valued case lowers through the existing `axmath` log/exp
/// substrate until a dedicated tuned candidate is registered.
fn pow_f32(base: f32, exponent: f32) -> f32 {
    let integer = exponent as i32;
    if integer as f32 == exponent && integer.unsigned_abs() <= 64 {
        let mut result = 1.0;
        for _ in 0..integer.unsigned_abs() {
            result *= base;
        }
        if integer < 0 { 1.0 / result } else { result }
    } else {
        axmath::exp_f32(exponent * axmath::log_f32(base))
    }
}

fn pow_f64(base: f64, exponent: f64) -> f64 {
    let integer = exponent as i32;
    if integer as f64 == exponent && integer.unsigned_abs() <= 64 {
        let mut result = 1.0;
        for _ in 0..integer.unsigned_abs() {
            result *= base;
        }
        if integer < 0 { 1.0 / result } else { result }
    } else {
        axmath::exp_f64(exponent * axmath::log_f64(base))
    }
}

fn remainder_f32(lhs: f32, rhs: f32) -> f32 {
    let value = lhs - rhs * axmath::trunc_f32(lhs / rhs);
    if value == 0.0 {
        axmath::copysign_f32(0.0, lhs)
    } else {
        value
    }
}

fn remainder_f64(lhs: f64, rhs: f64) -> f64 {
    let value = lhs - rhs * axmath::trunc_f64(lhs / rhs);
    if value == 0.0 {
        axmath::copysign_f64(0.0, lhs)
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axnn::lowering::{KernelPolicy, KernelRequest, SemanticOp};
    use axnn::{TensorDesc, shape};

    fn request(semantic: SemanticOp, elements: usize) -> KernelRequest {
        KernelRequest::new(semantic, &TensorDesc::new(shape![elements], DType::F32))
    }

    #[test]
    fn runtime_policy_changes_selected_variant() {
        let request = request(SemanticOp::Binary(BinaryOp::Add), 64);
        let scalar = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        assert_eq!(scalar.implementation, CpuImplementation::Scalar);

        let portable = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(portable.implementation, CpuImplementation::PortableSimd);
        assert!(portable.estimated_cost < scalar.estimated_cost);
    }

    #[test]
    fn scalar_and_portable_plans_invoke_existing_kernels() {
        let src = [0.0f32, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let request = request(SemanticOp::Unary(UnaryOp::Sqrt), src.len());
        let scalar = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let portable = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();

        let mut scalar_out = [0.0; 8];
        let mut portable_out = [0.0; 8];
        invoke_f32(&scalar, &src, None, None, &mut scalar_out).unwrap();
        invoke_f32(&portable, &src, None, None, &mut portable_out).unwrap();
        let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for ((scalar, portable), expected) in scalar_out.into_iter().zip(portable_out).zip(expected)
        {
            assert!((scalar - expected).abs() < 1e-6);
            assert!((portable - expected).abs() < 3e-5);
        }
    }

    #[test]
    fn reductions_select_the_admitted_tier_and_write_one_scalar() {
        let input = [1.0_f32, -4.0, 8.0, 2.0, 3.0, -1.0, 5.0, 6.0];
        for (op, expected) in [(ReduceOp::Sum, 20.0), (ReduceOp::Max, 8.0)] {
            let request =
                KernelRequest::reduce(op, &TensorDesc::new(shape![input.len()], DType::F32));
            let scalar = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
            let portable = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
            assert_eq!(scalar.implementation, CpuImplementation::Scalar);
            assert_eq!(portable.implementation, CpuImplementation::PortableSimd);
            assert!(portable.estimated_cost < scalar.estimated_cost);

            for plan in [scalar, portable] {
                let mut output = [f32::NAN];
                invoke_reduce_f32(&plan, &input, &mut output).unwrap();
                assert_eq!(output, [expected]);
            }
        }
    }

    #[test]
    fn empty_sum_is_defined_but_empty_max_is_not_admitted() {
        let desc = TensorDesc::new(shape![0], DType::F32);
        let sum = plan_cpu_kernel(
            &KernelRequest::reduce(ReduceOp::Sum, &desc),
            CpuCapabilities::SCALAR,
        )
        .unwrap();
        let mut output = [f32::NAN];
        invoke_reduce_f32(&sum, &[], &mut output).unwrap();
        assert_eq!(output, [0.0]);
        assert!(
            plan_cpu_kernel(
                &KernelRequest::reduce(ReduceOp::Max, &desc),
                CpuCapabilities::SCALAR,
            )
            .is_none()
        );
    }

    #[test]
    fn planned_exp_preserves_underflow_overflow_and_nan_across_simd_and_tail() {
        let input = [
            -10000.0,
            -100.0,
            -89.0,
            -88.0,
            -87.0,
            -1.0,
            0.0,
            1.0,
            87.0,
            88.0,
            88.5,
            90.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -104.0,
            -1000.0,
        ];
        let request = request(SemanticOp::Unary(UnaryOp::Exp), input.len());
        // The AVX2 policy falls back to portable on other builds; on x86 with
        // that explicit feature it exercises both wide implementations.
        for capabilities in [CpuCapabilities::PORTABLE_SIMD, CpuCapabilities::AVX2] {
            let plan = plan_cpu_kernel(&request, capabilities).unwrap();
            let mut output = [0.0; 17];
            invoke_f32(&plan, &input, None, None, &mut output).unwrap();
            for (&x, &actual) in input.iter().zip(&output) {
                let expected = axmath::exp_f32(x);
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else if !expected.is_normal() {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "exp({x})");
                } else {
                    assert!(actual.is_finite() && actual > 0.0, "exp({x}) = {actual}");
                    assert!(
                        (actual / expected - 1.0).abs() < 1e-5,
                        "exp({x}) = {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_catalog_selects_and_invokes_tuned_gemm() {
        let lhs_desc = TensorDesc::new(shape![2, 3], DType::F32);
        let rhs_desc = TensorDesc::new(shape![3, 4], DType::F32);
        let output_desc = TensorDesc::new(shape![2, 4], DType::F32);
        let request = KernelRequest::matmul(&lhs_desc, &rhs_desc, &output_desc).unwrap();
        let scalar = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let portable = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(scalar.implementation, CpuImplementation::Scalar);
        assert_eq!(portable.implementation, CpuImplementation::PortableSimd);
        assert_eq!(portable.work_items, 24);
        assert!(portable.estimated_cost < scalar.estimated_cost);

        let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let expected = [38.0_f32, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        for plan in [scalar, portable] {
            let mut output = [0.0_f32; 8];
            invoke_gemm_f32(&plan, &lhs, &rhs, &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn matmul_catalog_uses_reduction_work_for_narrow_outputs() {
        let lhs_desc = TensorDesc::new(shape![1, 16], DType::F32);
        let rhs_desc = TensorDesc::new(shape![16, 1], DType::F32);
        let output_desc = TensorDesc::new(shape![1, 1], DType::F32);
        let request = KernelRequest::matmul(&lhs_desc, &rhs_desc, &output_desc).unwrap();
        assert_eq!(request.elements(), 1);
        assert_eq!(request.work_items(), 16);

        let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(plan.implementation, CpuImplementation::PortableSimd);

        let lhs = [1.0_f32; 16];
        let rhs = [2.0_f32; 16];
        let mut output = [0.0_f32; 1];
        invoke_gemm_f32(&plan, &lhs, &rhs, &mut output).unwrap();
        assert_eq!(output, [32.0]);
    }

    #[test]
    fn new_scalar_semantics_are_selected_and_invoked() {
        let values = [-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5, 3.25, -3.25];
        let round_request = request(SemanticOp::Unary(UnaryOp::Round), values.len());
        let plan = select_cpu_kernel(
            &round_request,
            KernelPolicy::new(CpuCapabilities::PORTABLE_SIMD.mask())
                .with_minimum_accuracy(KernelAccuracy::Strict),
        )
        .unwrap();
        assert_eq!(plan.implementation, CpuImplementation::Scalar);

        let mut rounded = [0.0; 8];
        invoke_f32(&plan, &values, None, None, &mut rounded).unwrap();
        assert_eq!(rounded, [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 3.0, -3.0]);
        assert!(rounded[2].is_sign_negative());

        let rhs = [2.0f32; 8];
        let rem_request = request(SemanticOp::Binary(BinaryOp::Rem), values.len());
        let plan = plan_cpu_kernel(&rem_request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(plan.implementation, CpuImplementation::Scalar);
        let mut remainder = [0.0; 8];
        invoke_f32(&plan, &values, Some(&rhs), None, &mut remainder).unwrap();
        assert_eq!(remainder, [-0.5, -1.5, -0.5, 0.5, 1.5, 0.5, 1.25, -1.25]);

        let reciprocal_request = request(SemanticOp::Unary(UnaryOp::Reciprocal), values.len());
        let plan = plan_cpu_kernel(&reciprocal_request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(plan.implementation, CpuImplementation::Scalar);
        let positive = [0.5f32, 1.0, 2.0, 4.0, 8.0, 10.0, 0.25, 5.0];
        let mut reciprocal = [0.0; 8];
        invoke_f32(&plan, &positive, None, None, &mut reciprocal).unwrap();
        assert_eq!(reciprocal, [2.0, 1.0, 0.5, 0.25, 0.125, 0.1, 4.0, 0.2]);
    }

    #[test]
    fn ternary_plan_invokes_clamp() {
        let values = [-2.0f32, -1.0, 0.0, 2.0];
        let low = [-1.0; 4];
        let high = [1.0; 4];
        let request = request(SemanticOp::Ternary(TernaryOp::Clamp), values.len());
        let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(plan.implementation, CpuImplementation::Scalar);
        let mut output = [0.0; 4];
        invoke_f32(&plan, &values, Some(&low), Some(&high), &mut output).unwrap();
        assert_eq!(output, [-1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn compute_backend_routes_new_semantics_through_the_planner() {
        let backend = crate::CpuBackend::<4096>::with_kernel_capabilities(CpuCapabilities::SCALAR);
        let input = axnn::tensor!(&backend, [-2.5f32, -1.5, 1.5, 2.5]).unwrap();
        let output = axnn::round(&input, &backend).unwrap();
        assert_eq!(
            axnn::get(&output, &[0], &backend).unwrap(),
            axnn::Scalar::F32(-2.0)
        );
        assert_eq!(
            axnn::get(&output, &[1], &backend).unwrap(),
            axnn::Scalar::F32(-2.0)
        );
        assert_eq!(
            axnn::get(&output, &[2], &backend).unwrap(),
            axnn::Scalar::F32(2.0)
        );
        assert_eq!(
            axnn::get(&output, &[3], &backend).unwrap(),
            axnn::Scalar::F32(2.0)
        );

        let field = axnn::tensor!(&backend, [1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let scalar = axnn::tensor!(&backend, [2.0f32]).unwrap();
        let scaled = axnn::mul(&field, &scalar, &backend).unwrap();
        assert_eq!(scaled.shape().dims(), &[4]);
        assert_eq!(
            axnn::get(&scaled, &[3], &backend).unwrap(),
            axnn::Scalar::F32(8.0)
        );
    }

    #[test]
    fn scalar_field_broadcast_is_planned_and_invoked() {
        let field_desc = TensorDesc::new(shape![4], DType::F32);
        let scalar_desc = TensorDesc::new(shape![1], DType::F32);
        let request = KernelRequest::binary(BinaryOp::Mul, &field_desc, &scalar_desc).unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        // Portable kernels require equal-length slices; the planner preserves
        // the scalar broadcast and chooses the scalar implementation.
        assert_eq!(plan.implementation, CpuImplementation::Scalar);

        let field = [1.0f32, 2.0, 3.0, 4.0];
        let scalar = [2.5f32];
        let mut output = [0.0; 4];
        invoke_f32(&plan, &field, Some(&scalar), None, &mut output).unwrap();
        assert_eq!(output, [2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn scalar_kernel_broadcasts_coordinate_fields_across_both_axes() {
        // axshader supplies the coordinate fields in these complementary
        // shapes: x = [1, W], y = [H, 1]. The scalar fallback must evaluate
        // their full [H, W] broadcast rather than merely accepting len = 1.
        let y_desc = TensorDesc::new(shape![2, 1], DType::F32);
        let x_desc = TensorDesc::new(shape![1, 3], DType::F32);
        let request = KernelRequest::binary(BinaryOp::Atan2, &y_desc, &x_desc).unwrap();
        assert_eq!(request.shape, shape![2, 3]);

        let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(plan.implementation, CpuImplementation::Scalar);

        let y = [0.0f32, 1.0];
        let x = [1.0f32, 2.0, 4.0];
        let mut output = [0.0; 6];
        invoke_f32(&plan, &y, Some(&x), None, &mut output).unwrap();

        let expected = [
            0.0f32,
            0.0,
            0.0,
            core::f32::consts::FRAC_PI_4,
            0.463_647_6,
            0.244_978_67,
        ];
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 5e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn comparison_and_select_keep_typed_mask_semantics() {
        let field_desc = TensorDesc::new(shape![4], DType::F32);
        let scalar_desc = TensorDesc::new(shape![1], DType::F32);
        let compare_request =
            KernelRequest::compare(CompareOp::GreaterEqual, &field_desc, &scalar_desc).unwrap();
        let compare_plan =
            plan_cpu_kernel(&compare_request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        assert_eq!(compare_plan.output_dtype, DType::U8);
        assert_eq!(compare_plan.implementation, CpuImplementation::Scalar);

        let field = [-1.0f32, 0.0, 1.0, 2.0];
        let threshold = [1.0f32];
        let mut mask = [0u8; 4];
        invoke_compare_f32(&compare_plan, &field, &threshold, &mut mask).unwrap();
        assert_eq!(mask, [0, 0, 1, 1]);

        let mask_desc = TensorDesc::new(shape![4], DType::U8);
        let false_desc = TensorDesc::new(shape![4], DType::F32);
        let select_request = KernelRequest::select(&mask_desc, &scalar_desc, &false_desc).unwrap();
        let select_plan = plan_cpu_kernel(&select_request, CpuCapabilities::PORTABLE_SIMD).unwrap();
        let selected = [10.0f32];
        let alternative = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        invoke_select_f32(&select_plan, &mask, &selected, &alternative, &mut output).unwrap();
        assert_eq!(output, [1.0, 2.0, 10.0, 10.0]);
    }

    #[test]
    fn identity_plans_preserve_f32_and_u8_values() {
        let float_desc = TensorDesc::new(shape![4], DType::F32);
        let float_plan = plan_cpu_kernel(
            &KernelRequest::identity(&float_desc),
            CpuCapabilities::PORTABLE_SIMD,
        )
        .unwrap();
        assert_eq!(float_plan.implementation, CpuImplementation::Scalar);
        let float_input = [-0.0f32, 1.25, f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let mut float_output = [0.0; 4];
        invoke_f32(&float_plan, &float_input, None, None, &mut float_output).unwrap();
        assert_eq!(
            float_output.map(f32::to_bits),
            float_input.map(f32::to_bits)
        );

        let mask_desc = TensorDesc::new(shape![4], DType::U8);
        let mask_plan = plan_cpu_kernel(
            &KernelRequest::identity(&mask_desc),
            CpuCapabilities::PORTABLE_SIMD,
        )
        .unwrap();
        assert_eq!(mask_plan.implementation, CpuImplementation::Scalar);
        let mask_input = [0u8, 1, 2, u8::MAX];
        let mut mask_output = [0u8; 4];
        invoke_u8(&mask_plan, &mask_input, None, &mut mask_output).unwrap();
        assert_eq!(mask_output, mask_input);
    }

    #[test]
    fn interleave3_expands_scalar_row_column_and_dense_planes_exactly() {
        let values = [
            -0.0,
            0.0,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0x7f80_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            -2.5,
        ];
        let x: [f32; 258] = core::array::from_fn(|index| values[index % values.len()]);
        let y: [f32; 258] = core::array::from_fn(|index| values[(index + 1) % values.len()]);
        let z: [f32; 258] = core::array::from_fn(|index| values[(index + 2) % values.len()]);
        for [height, width] in [
            [2, 0],
            [1, 1],
            [2, 3],
            [2, 4],
            [2, 5],
            [1, 127],
            [1, 128],
            [2, 129],
        ] {
            let field = shape![height, width];
            for shapes in [
                [shape![height, 1], shape![1, width], field],
                [field, Shape::SCALAR, shape![width]],
            ]
            .into_iter()
            .chain((0..8).map(|mask| {
                core::array::from_fn(|axis| {
                    if mask & (1 << axis) == 0 {
                        field
                    } else {
                        Shape::SCALAR
                    }
                })
            })) {
                let inputs = shapes.map(|shape| TensorDesc::new(shape, DType::F32));
                let output = TensorDesc::new(shape![height, width, 3], DType::F32);
                let request =
                    KernelRequest::interleave3_f32(&inputs[0], &inputs[1], &inputs[2], &output)
                        .unwrap();
                let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
                assert_eq!(plan.implementation, CpuImplementation::Scalar);
                assert_eq!(plan.accuracy, KernelAccuracy::Strict);
                assert_eq!(plan.shape, output.shape);
                let mut actual = [42.0; 777];
                invoke_interleave3_f32(
                    &plan,
                    &x[..shapes[0].num_elements()],
                    &y[..shapes[1].num_elements()],
                    &z[..shapes[2].num_elements()],
                    &mut actual[..output.num_elements()],
                )
                .unwrap();
                for index in 0..field.num_elements() {
                    for (axis, source) in [&x, &y, &z].into_iter().enumerate() {
                        let expected = source[broadcast_index(shapes[axis], field, index).unwrap()];
                        assert_eq!(actual[index * 3 + axis].to_bits(), expected.to_bits());
                    }
                }
                assert!(
                    actual[output.num_elements()..]
                        .iter()
                        .all(|&value| value == 42.0)
                );
            }
        }
    }

    #[test]
    fn interleave3_rejects_layouts_and_bad_invocation_lengths_without_writing() {
        let scalar = TensorDesc::new(Shape::SCALAR, DType::F32);
        let output = TensorDesc::new(shape![2, 3, 3], DType::F32);
        let request = KernelRequest::interleave3_f32(&scalar, &scalar, &scalar, &output).unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let mut actual = [42.0; 18];
        assert!(invoke_interleave3_f32(&plan, &[], &[1.0], &[2.0], &mut actual).is_err());
        assert!(invoke_interleave3_f32(&plan, &[0.0], &[1.0], &[2.0], &mut actual[..17]).is_err());
        assert_eq!(actual, [42.0; 18]);

        let mut bad_plan = plan;
        bad_plan.input_shapes[0] = Some(shape![1]);
        assert!(invoke_interleave3_f32(&bad_plan, &[0.0], &[1.0], &[2.0], &mut actual).is_err());
        assert_eq!(actual, [42.0; 18]);

        let strided_output = TensorDesc {
            layout: axnn::Layout::from_raw(&[96, 16, 4], 0),
            ..output
        };
        let request =
            KernelRequest::interleave3_f32(&scalar, &scalar, &scalar, &strided_output).unwrap();
        assert!(plan_cpu_kernel(&request, CpuCapabilities::compiled()).is_none());
    }

    #[test]
    fn i32_identity_and_comparisons_preserve_large_ids_and_signed_extremes() {
        let values = [-1, 0, 16_777_216, 16_777_217, i32::MIN, i32::MAX];
        let field = TensorDesc::new(shape![2, 3], DType::I32);
        let identity = plan_cpu_kernel(
            &KernelRequest::identity(&field),
            CpuCapabilities::PORTABLE_SIMD,
        )
        .unwrap();
        assert_eq!(identity.implementation, CpuImplementation::Scalar);
        let mut copied = [0; 6];
        invoke_i32(&identity, &values, &mut copied).unwrap();
        assert_eq!(copied, values);
        assert!(invoke_i32(&identity, &values[..5], &mut copied).is_err());
        assert_eq!(copied, values);

        let rhs_values = [16_777_216, 16_777_217, -1, i32::MAX, i32::MIN, 0];
        for (left, right) in [
            (shape![2, 3], Shape::SCALAR),
            (Shape::SCALAR, shape![2, 3]),
            (shape![2, 3], shape![2, 3]),
            (shape![2, 1], shape![1, 3]),
        ] {
            for op in [
                CompareOp::Equal,
                CompareOp::Greater,
                CompareOp::GreaterEqual,
            ] {
                let request = KernelRequest::compare(
                    op,
                    &TensorDesc::new(left, DType::I32),
                    &TensorDesc::new(right, DType::I32),
                )
                .unwrap();
                let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
                assert_eq!(plan.accuracy, KernelAccuracy::Strict);
                let mut mask = [255; 6];
                invoke_compare_i32(
                    &plan,
                    &values[..left.num_elements()],
                    &rhs_values[..right.num_elements()],
                    &mut mask,
                )
                .unwrap();
                for (index, &actual) in mask.iter().enumerate() {
                    let lhs = values[broadcast_index(left, plan.shape, index).unwrap()];
                    let rhs = rhs_values[broadcast_index(right, plan.shape, index).unwrap()];
                    let expected = match op {
                        CompareOp::Equal => lhs == rhs,
                        CompareOp::Greater => lhs > rhs,
                        CompareOp::GreaterEqual => lhs >= rhs,
                    };
                    assert_eq!(actual, u8::from(expected));
                }
                let before = mask;
                assert!(
                    invoke_compare_i32(&plan, &[], &rhs_values[..right.num_elements()], &mut mask)
                        .is_err()
                );
                assert_eq!(mask, before);
                assert!(invoke_compare_f32(&plan, &[0.0; 6], &[0.0; 6], &mut mask).is_err());
            }
        }
        let select =
            KernelRequest::select(&TensorDesc::new(field.shape, DType::U8), &field, &field)
                .unwrap();
        let select = plan_cpu_kernel(&select, CpuCapabilities::compiled()).unwrap();
        crate::invoke_select_i32(
            &select,
            &[0, 1, 2, 0, 255, 0],
            &values,
            &rhs_values,
            &mut copied,
        )
        .unwrap();
        assert_eq!(copied, [16_777_216, 0, 16_777_216, i32::MAX, i32::MIN, 0]);
        let before = copied;
        assert!(
            crate::invoke_select_i32(&select, &[1], &values, &rhs_values, &mut copied).is_err()
        );
        assert_eq!(copied, before);
    }

    #[test]
    fn u8_logical_kernels_broadcast_and_normalize_masks() {
        let column_desc = TensorDesc::new(shape![2, 1], DType::U8);
        let row_desc = TensorDesc::new(shape![1, 3], DType::U8);
        let column = [0u8, 2];
        let row = [0u8, 3, 0];

        let not_plan = plan_cpu_kernel(
            &KernelRequest::logical_not(&column_desc).unwrap(),
            CpuCapabilities::PORTABLE_SIMD,
        )
        .unwrap();
        let mut not_output = [0u8; 2];
        invoke_u8(&not_plan, &column, None, &mut not_output).unwrap();
        assert_eq!(not_output, [1, 0]);

        let cases = [
            (LogicalOp::And, [0, 0, 0, 0, 1, 0]),
            (LogicalOp::Or, [0, 1, 0, 1, 1, 1]),
            (LogicalOp::Xor, [0, 1, 0, 1, 0, 1]),
        ];
        for (op, expected) in cases {
            let request = KernelRequest::logical_binary(op, &column_desc, &row_desc).unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
            assert_eq!(plan.implementation, CpuImplementation::Scalar);
            let mut output = [0u8; 6];
            invoke_u8(&plan, &column, Some(&row), &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn sign_propagates_nan_and_handles_zero_and_infinities() {
        let nan = f32::from_bits(0x7fc0_1234);
        let values = [nan, -0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY];
        let plan = plan_cpu_kernel(
            &request(SemanticOp::Unary(UnaryOp::Sign), values.len()),
            CpuCapabilities::SCALAR,
        )
        .unwrap();
        let mut output = [42.0; 5];
        invoke_f32(&plan, &values, None, None, &mut output).unwrap();

        assert_eq!(output[0].to_bits(), nan.to_bits());
        assert_eq!(output[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(output[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(output[3], 1.0);
        assert_eq!(output[4], -1.0);

        let nan64 = f64::from_bits(0x7ff8_0000_0000_1234);
        assert_eq!(
            unary_value_f64(UnaryOp::Sign, nan64).to_bits(),
            nan64.to_bits()
        );
        assert_eq!(
            unary_value_f64(UnaryOp::Sign, -0.0).to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(unary_value_f64(UnaryOp::Sign, f64::INFINITY), 1.0);
        assert_eq!(unary_value_f64(UnaryOp::Sign, f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn broadcast_index_checks_rank_shape_and_bounds() {
        assert_eq!(broadcast_index(shape![3], shape![2, 3], 4), Some(1));
        assert_eq!(broadcast_index(shape![1, 3], shape![2, 3], 5), Some(2));
        assert_eq!(broadcast_index(shape![2, 3], shape![3], 0), None);
        assert_eq!(broadcast_index(shape![2], shape![2, 3], 0), None);
        assert_eq!(broadcast_index(shape![3], shape![2, 3], 6), None);
        assert_eq!(broadcast_index(shape![0], shape![0], 0), None);
    }
}
