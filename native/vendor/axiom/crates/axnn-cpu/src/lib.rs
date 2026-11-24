#![no_std]
#![feature(portable_simd)]
#![feature(f16)]
#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![allow(unused)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_prefetch))]

// Production x86 targets must carry `-mmx` in their target feature contract.
// Rust does not expose LLVM's MMX switch through `cfg(target_feature)`, so it
// cannot be truthfully checked here. This no_std crate neither names MMX/x87
// intrinsics nor emits inline assembly; production artifact audits remain the
// authoritative verification that code generation respected `-mmx`.

#[cfg(any(test, feature = "std", feature = "alloc"))]
extern crate alloc;

mod access;
pub mod arch;
pub use arch::{FLOAT_TIER, FloatTier};
mod backend;
mod buffer_resource;
mod cast;
mod error;
mod floats;
mod fork_join;
pub mod gguf;
mod integer;
mod kv_cache;
mod moe;
mod neural;
pub use fork_join::{ForkJoin, Serial};
pub use neural::{
    invoke_dequantize_f32, invoke_matmul_nvfp4_transposed, invoke_matmul_nvfp4_transposed_rows,
    invoke_matmul_transposed, invoke_matmul_transposed_f32, invoke_matmul_transposed_f32_weights,
    invoke_neural_f32, invoke_quantize_f32, nvfp4_q8_scratch_elements,
};
pub mod planner;
mod pool;
mod quant;
pub mod raster_program;
mod recurrent;
mod routed;
pub mod rounded;
mod shapes;
pub mod sink_attention;
pub mod sparse;
mod storage;
mod stride;
mod tensor_ops;
mod volume_resource;
mod voxel_resource;

pub use backend::CpuBackend;
pub use buffer_resource::{U8BufferRef, invoke_buffer_load_u8};
pub use error::CpuError;
pub use integer::{
    cast_shift_or3_f32_i32, invoke_cast_f32_i32, invoke_cast_i32_f32, invoke_integer_i32,
    invoke_select_i32,
};
pub use kv_cache::{
    E8M0_NAN, KvCacheRef, attention_scratch_elements, e8m0_to_f32, invoke_attention_decode,
    invoke_kv_append,
};
pub use moe::{
    DEFAULT_WAVE_EXPERTS, ExpertSource, ExpertStoreRef, ExpertWeights, MAX_WAVE_EXPERTS,
    MOE_PHASE_NAMES, MOE_PHASES, MoeBatchRow, MoeDelegate, MoeInvocation, MoePhases, MoeProgress,
    ResumableMoeDelegate, WeightResidency, invoke_loaded_nvfp4_experts, invoke_moe_experts,
    invoke_moe_experts_delegated, invoke_moe_experts_with, moe_prepared_scratch_elements,
    moe_scratch_elements, native_wave,
};
pub use planner::{
    CpuCapabilities, CpuImplementation, CpuKernelCatalog, CpuKernelPlan, invoke_compare_f32,
    invoke_compare_i32, invoke_f32, invoke_gemm_f32, invoke_i32, invoke_interleave3_f32,
    invoke_reduce_f32, invoke_select_f32, invoke_u8, plan_cpu_kernel, select_cpu_kernel,
};
pub use raster_program::{
    CoverageAtlasRef, GlyphTableRef, XrgbSurfaceRef, XrgbTargetRef, invoke_raster, paint_rows,
};
pub use recurrent::{
    StateSourceRef, StateTargetRef, invoke_causal_conv1d, invoke_gated_delta_rule, invoke_ssm_scan,
};
pub use routed::{invoke_moe_down_nvfp4, invoke_moe_gate_up_nvfp4, invoke_moe_route};
pub use storage::CpuStorage;
pub use tensor_ops::{GatherTable, invoke_gather, invoke_top_k};
pub use volume_resource::{RgbVolumeRef, invoke_volume_trilinear_rgb};
pub use voxel_resource::{BrickGridRef, invoke_voxel_brick_trace};

/// Compile-time kernel tiers selected for this build.
///
/// This is intentionally exposed so benchmark output can prove which
/// capacity it measured. On x86, a plain generic `x86_64` build normally
/// reports portable floats and SSE2 qdot; use `-C target-cpu=native` to make
/// AVX2/FMA/F16C and VNNI available when the build host supports them.
pub mod build_info {
    /// Legacy aggregate floating-point label, retained for API compatibility.
    ///
    /// Historically this tracked dense GEMM, so its value remains identical to
    /// [`dense_gemm_isa`]. New benchmark output should use the family-specific
    /// accessors below.
    pub const fn float_isa() -> &'static str {
        dense_gemm_isa()
    }

    /// Dense GEMM kernel tier.
    pub const fn dense_gemm_isa() -> &'static str {
        if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "avx2+fma+f16c"
        } else if cfg!(all(target_arch = "aarch64", feature = "sme")) {
            "sme"
        } else if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "portable"
        }
    }

    /// Signed F8 expansion, dot-product, and mixed-GEMM tier.
    pub const fn signed_f8_isa() -> &'static str {
        if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "avx2+fma+f16c"
        } else if cfg!(all(target_arch = "aarch64", target_feature = "fhm")) {
            "neon+fhm"
        } else {
            "portable-simd"
        }
    }

    /// Unsigned-float storage expansion tier.
    pub const fn uf_isa() -> &'static str {
        if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "avx2+fma+f16c"
        } else if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "portable-simd"
        }
    }

    /// Software-raster kernel tier.
    pub const fn raster_isa() -> &'static str {
        if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "avx2"
        } else if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "portable-simd"
        }
    }

    /// Gaussian-splat compositor tier.
    ///
    /// The implementation is portable SIMD on every architecture; target
    /// codegen selects the available vector instructions.
    pub const fn splat_isa() -> &'static str {
        "portable-simd"
    }

    /// Default single-ray voxel traversal and volume-marching tier.
    pub const fn voxel_isa() -> &'static str {
        "portable-scalar"
    }

    /// Opt-in brick-packet traversal: four SIMD lanes with scalar fallbacks.
    pub const fn voxel_packet_isa() -> &'static str {
        "portable-simd4+scalar-fallback"
    }

    /// Quantized dot-product kernel tier, independently selected from GEMM.
    pub const fn qdot_isa() -> &'static str {
        if cfg!(all(feature = "amx", target_arch = "x86_64")) {
            "amx"
        } else if cfg!(all(feature = "avx512", target_arch = "x86_64")) {
            "avx512"
        } else if cfg!(all(feature = "avx_vnni", target_arch = "x86_64")) {
            "avx-vnni"
        } else if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "avx2"
        } else if cfg!(all(feature = "sse42", target_arch = "x86_64")) {
            "sse4.2"
        } else if cfg!(all(feature = "sse2", target_arch = "x86_64")) {
            "sse2"
        } else if cfg!(all(feature = "neon", target_arch = "aarch64")) {
            "neon"
        } else {
            "portable"
        }
    }

    /// NVFP4 activation tier selected by the compiled artifact.
    pub const fn nvfp4_isa() -> &'static str {
        if cfg!(all(
            feature = "avx512",
            feature = "nvfp4_vnni",
            target_feature = "avx512vnni",
            target_arch = "x86_64"
        )) {
            "w4a8-avx512-vnni"
        } else if cfg!(all(feature = "avx_vnni", target_arch = "x86_64")) {
            "w4a8-avx-vnni"
        } else if cfg!(all(feature = "avx512", target_arch = "x86_64")) {
            "w4a16-avx512"
        } else if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
            "w4a16-avx2"
        } else {
            "w4a16-portable"
        }
    }
}

/// Q1 bit-packed row and matrix operations (SIMD-accelerated).
///
/// Re-exported for direct use by crates that need raw bit vector
/// operations without going through the `ComputeBackend` trait
/// (e.g., `axfs` filesystem bitmaps).
pub use arch::portable::bitmat;

/// Thermal-radiation kernels with no_std Planck and Stefan-Boltzmann
/// evaluation.
pub use arch::portable::radiation;

/// Voxel ray tracing — uniform-grid DDA, sparse voxel octree, brick
/// grid, and mixed-precision density marching.
///
/// Re-exported for direct use by rendering crates (e.g. `axrd`); these
/// don't fit the `ComputeBackend` tensor-op surface. Single-ray traversal is
/// scalar; the opt-in brick packet API uses portable SIMD for its coarse walk.
pub use arch::portable::voxel;

/// Dense matrix multiply micro-kernels (`C += A × B`, row-major) —
/// re-exported for graph executors (e.g. `axtosa`).
/// Elementwise activation kernels (SIMD, in place).
pub mod activation {
    /// In-place tanh: minimax rational, libm-accurate to ~2 ulp.
    pub fn tanh_f32_inplace(v: &mut [f32]) {
        crate::arch::portable::act::tanh_f32_inplace(v)
    }

    /// In-place logistic sigmoid via `σ(x) = ½ + ½·tanh(x/2)`.
    pub fn sigmoid_f32_inplace(v: &mut [f32]) {
        crate::arch::portable::act::sigmoid_f32_inplace(v)
    }
}

/// Strict elementwise fusion kernels used by static graph executors.
///
/// These kernels deliberately spell multiply and add as distinct operations:
/// graph fusion removes an intermediate memory plane, but does not change the
/// rounding boundary into a fused multiply-add operation.
pub mod elementwise {
    /// Exact `(a - b) * scale + addend`, with each input either scalar or a
    /// complete output-sized plane. Keeps all three rounding boundaries.
    /// Uses portable SIMD explicitly; policy-aware callers must admit it.
    /// Panics on incompatible lengths before writing output.
    pub fn sub_mul_add_f32(inputs: [&[f32]; 4], output: &mut [f32]) {
        assert!(
            inputs
                .iter()
                .all(|p| p.len() == 1 || p.len() == output.len())
        );
        crate::arch::portable::vec::sub_mul_add_f32(inputs, output);
    }

    /// Original operand order for a fused strict plane `MUL` followed by
    /// `ADD`. The multiplication retains its source operand order; this enum
    /// records which side of the addition held the product.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MulAddOrder {
        /// `(left * right) + addend`.
        ProductAddend,
        /// `addend + (left * right)`.
        AddendProduct,
    }

    /// Original operand order for a fused strict `MUL` followed by `ADD`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ScaleAddOrder {
        /// `(input * scale) + addend`.
        InputScaleAddend,
        /// `(scale * input) + addend`.
        ScaleInputAddend,
        /// `addend + (input * scale)`.
        AddendInputScale,
        /// `addend + (scale * input)`.
        AddendScaleInput,
    }

    /// Multiply three equally sized dense planes by a shared scalar, then
    /// store xyz-interleaved F32 values without intermediate product planes.
    /// `scale_first[axis]` preserves whether that MUL was `scale * component`
    /// rather than `component * scale`. Arithmetic NaN payload/sign choice
    /// follows Rust's floating-point contract; packing adds no arithmetic.
    ///
    /// Panics for unequal input lengths or an output length other than 3N.
    /// Uses the selected architecture vector tier; policy-aware callers must
    /// admit that tier.
    pub fn scale_interleave3_f32(
        planes: [&[f32]; 3],
        scale: f32,
        dst: &mut [f32],
        scale_first: [bool; 3],
    ) {
        let elements = planes[0].len();
        assert!(planes.iter().all(|plane| plane.len() == elements));
        assert_eq!(elements.checked_mul(3), Some(dst.len()));
        crate::arch::vec::scale_interleave3_f32(planes, scale, dst, scale_first)
    }

    /// Compute an ordered strict scale/add over a vector addend in SIMD
    /// registers.
    pub fn scale_add_f32(
        input: &[f32],
        scale: f32,
        addend: &[f32],
        dst: &mut [f32],
        order: ScaleAddOrder,
    ) {
        assert_eq!(input.len(), addend.len());
        assert_eq!(input.len(), dst.len());
        crate::arch::vec::scale_add_f32(input, scale, addend, dst, order)
    }

    /// Compute an ordered strict scale/add over a scalar addend in SIMD
    /// registers.
    pub fn scale_add_scalar_f32(
        input: &[f32],
        scale: f32,
        addend: f32,
        dst: &mut [f32],
        order: ScaleAddOrder,
    ) {
        assert_eq!(input.len(), dst.len());
        crate::arch::vec::scale_add_scalar_f32(input, scale, addend, dst, order)
    }

    /// Apply an ordered strict scale/add to `input` in place. Static graph
    /// executors use this form when liveness proves the input plane has no
    /// remaining consumers, so the fused region needs neither an intermediate
    /// multiplication plane nor a fresh output allocation.
    pub fn scale_add_f32_inplace(
        input: &mut [f32],
        scale: f32,
        addend: &[f32],
        order: ScaleAddOrder,
    ) {
        assert_eq!(input.len(), addend.len());
        crate::arch::vec::scale_add_f32_inplace(input, scale, addend, order)
    }

    /// Scalar-addend form of [`scale_add_f32_inplace`].
    pub fn scale_add_scalar_f32_inplace(
        input: &mut [f32],
        scale: f32,
        addend: f32,
        order: ScaleAddOrder,
    ) {
        crate::arch::vec::scale_add_scalar_f32_inplace(input, scale, addend, order)
    }

    /// Compute an ordered strict elementwise product/add over three planes.
    ///
    /// This is the reusable compositor form of `MUL -> ADD`: it removes the
    /// full-size product allocation while retaining both IEEE arithmetic
    /// boundaries.
    pub fn mul_add_f32(
        left: &[f32],
        right: &[f32],
        addend: &[f32],
        output: &mut [f32],
        order: MulAddOrder,
    ) {
        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), addend.len());
        assert_eq!(left.len(), output.len());
        crate::arch::vec::mul_add_f32(left, right, addend, output, order)
    }

    /// In-place form of [`mul_add_f32`]. Static executors use this after
    /// liveness proves `left` can become the output plane.
    pub fn mul_add_f32_inplace(
        left: &mut [f32],
        right: &[f32],
        addend: &[f32],
        order: MulAddOrder,
    ) {
        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), addend.len());
        crate::arch::vec::mul_add_f32_inplace(left, right, addend, order)
    }

    /// Apply `current + (target - current) * weight` in place with each
    /// arithmetic boundary retained. Static graph executors use this for a
    /// single-consumer SUB → MUL → ADD region, removing two full-size
    /// intermediate planes without inventing a backend-specific graph op.
    pub fn lerp_scalar_f32_inplace(current: &mut [f32], target: f32, weight: &[f32]) {
        assert_eq!(current.len(), weight.len());
        crate::arch::vec::lerp_scalar_f32_inplace(current, target, weight)
    }

    /// Apply a strict two-axis linearized translation followed by a
    /// propagating-NaN scalar clamp in one plane pass.
    ///
    /// The arithmetic order is
    /// `clamp(value + derivative_x * offset_x + derivative_y * offset_y,
    /// lower, upper)`. This is the shared architecture-vector kernel for
    /// shallow graphics parallax regions; it removes five framebuffer-sized
    /// intermediates without changing graph-visible rounding boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn linearized_clamp_f32_inplace(
        value: &mut [f32],
        derivative_x: &[f32],
        offset_x: f32,
        derivative_y: &[f32],
        offset_y: f32,
        lower: f32,
        upper: f32,
    ) {
        assert_eq!(value.len(), derivative_x.len());
        assert_eq!(value.len(), derivative_y.len());
        crate::arch::vec::linearized_clamp_f32_inplace(
            value,
            derivative_x,
            offset_x,
            derivative_y,
            offset_y,
            lower,
            upper,
        )
    }

    /// Apply `input * rsqrt(maximum(input, floor))` in place. This is the
    /// selected vector-tier form of the guarded distance-root region emitted
    /// by geometry graphs, with propagating-NaN and strict arithmetic semantics.
    pub fn guarded_sqrt_f32_inplace(input: &mut [f32], floor: f32) {
        crate::arch::vec::guarded_sqrt_f32_inplace(input, floor)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn scale_add_covers_simd_chunks_and_tail() {
            let input = [
                -8.0, -3.0, -1.0, -0.0, 0.25, 1.0, 2.0, 9.0, 12.0, 16.0, 32.0,
            ];
            let addend = [0.5, -0.5, 4.0, 2.0, -3.0, 0.25, 8.0, -9.0, 1.0, 3.0, -7.0];
            let mut vector = [0.0; 11];
            super::scale_add_f32(
                &input,
                0.125,
                &addend,
                &mut vector,
                super::ScaleAddOrder::InputScaleAddend,
            );
            let expected = core::array::from_fn::<_, 11, _>(|i| input[i] * 0.125 + addend[i]);
            assert_eq!(vector, expected);

            let mut scalar = [0.0; 11];
            super::scale_add_scalar_f32(
                &input,
                -0.25,
                1.5,
                &mut scalar,
                super::ScaleAddOrder::InputScaleAddend,
            );
            let expected = core::array::from_fn::<_, 11, _>(|i| input[i] * -0.25 + 1.5);
            assert_eq!(scalar, expected);

            for order in [
                super::ScaleAddOrder::InputScaleAddend,
                super::ScaleAddOrder::ScaleInputAddend,
                super::ScaleAddOrder::AddendInputScale,
                super::ScaleAddOrder::AddendScaleInput,
            ] {
                let mut vector = [0.0; 11];
                super::scale_add_f32(&input, 0.125, &addend, &mut vector, order);
                let mut vector_inplace = input;
                super::scale_add_f32_inplace(&mut vector_inplace, 0.125, &addend, order);
                assert_eq!(vector_inplace, vector, "vector order {order:?}");

                let mut scalar = [0.0; 11];
                super::scale_add_scalar_f32(&input, -0.25, 1.5, &mut scalar, order);
                let mut scalar_inplace = input;
                super::scale_add_scalar_f32_inplace(&mut scalar_inplace, -0.25, 1.5, order);
                assert_eq!(scalar_inplace, scalar, "scalar order {order:?}");
            }
        }

        #[test]
        fn plane_mul_add_covers_simd_chunks_tail_and_operand_order() {
            let left = [
                -8.0, -3.0, -1.0, -0.0, 0.25, 1.0, 2.0, 9.0, 12.0, 16.0, 32.0,
            ];
            let right = [0.5, -0.5, 4.0, 2.0, -3.0, 0.25, 8.0, -9.0, 1.0, 3.0, -7.0];
            let addend = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
            for order in [
                super::MulAddOrder::ProductAddend,
                super::MulAddOrder::AddendProduct,
            ] {
                let mut output = [0.0; 11];
                super::mul_add_f32(&left, &right, &addend, &mut output, order);
                let expected = core::array::from_fn::<_, 11, _>(|index| match order {
                    super::MulAddOrder::ProductAddend => left[index] * right[index] + addend[index],
                    super::MulAddOrder::AddendProduct => addend[index] + left[index] * right[index],
                });
                assert_eq!(output, expected);

                let mut inplace = left;
                super::mul_add_f32_inplace(&mut inplace, &right, &addend, order);
                assert_eq!(inplace, expected);
            }
        }

        #[test]
        fn lerp_covers_simd_chunks_and_tail() {
            let mut current = [
                -8.0, -3.0, -1.0, -0.0, 0.25, 1.0, 2.0, 9.0, 12.0, 16.0, 32.0,
            ];
            let original = current;
            let weight = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
            super::lerp_scalar_f32_inplace(&mut current, 4.0, &weight);
            let expected =
                core::array::from_fn::<_, 11, _>(|i| original[i] + (4.0 - original[i]) * weight[i]);
            assert_eq!(current, expected);
        }

        #[test]
        fn guarded_sqrt_covers_simd_chunks_and_tail() {
            let mut input = [0.0, 0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
            let original = input;
            super::guarded_sqrt_f32_inplace(&mut input, f32::EPSILON);
            let expected =
                original.map(|value| value * (1.0 / axmath::sqrt_f32(value.max(f32::EPSILON))));
            assert_eq!(input.map(f32::to_bits), expected.map(f32::to_bits));
        }

        #[test]
        fn linearized_clamp_preserves_strict_order_and_nan() {
            let mut value = [0.0, 0.25, 0.75, 1.0, f32::NAN, -1.0, 2.0];
            let derivative_x = [1.0, -1.0, 0.5, 1.0, 1.0, -2.0, 2.0];
            let derivative_y = [-0.5, 0.5, 1.0, -1.0, 2.0, 3.0, -3.0];
            let expected = core::array::from_fn::<_, 7, _>(|index| {
                let x = derivative_x[index] * 0.25;
                let y = derivative_y[index] * -0.5;
                let shifted = (value[index] + x) + y;
                if shifted.is_nan() {
                    shifted
                } else {
                    shifted.clamp(0.0, 1.0)
                }
            });
            super::linearized_clamp_f32_inplace(
                &mut value,
                &derivative_x,
                0.25,
                &derivative_y,
                -0.5,
                0.0,
                1.0,
            );
            for (actual, expected) in value.into_iter().zip(expected) {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

pub mod linalg {
    #[inline(always)]
    pub fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::gemm::gemm_f32(a, b, c, m, n, k)
    }
    /// Packed little-endian f16 inputs, f32 accumulation.
    #[inline(always)]
    pub fn gemm_f16(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::gemm::gemm_f16(a, b, c, m, n, k)
    }
}

/// SH → RGB evaluation (3DGS rendering Kernel A) — per-item spherical
/// harmonics with f32 / f16 / F8 coefficient tables.
pub mod sh {
    #[inline(always)]
    pub fn sh_eval_e4m3(dirs: &[f32], coeffs: &[u8], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_e4m3(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_e5m2(dirs: &[f32], coeffs: &[u8], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_e5m2(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_f16(dirs: &[f32], coeffs: &[f16], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_f16(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_f32(dirs: &[f32], coeffs: &[f32], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_f32(dirs, coeffs, out, n, deg)
    }
}

/// Gaussian-splat tile compositing (3DGS rendering Kernel B).
///
/// Every `id` must address complete entries in the center (2 values), conic
/// (3), opacity (1), and color (3) tables. `rgb` must contain `3 * PW`
/// planar values and `t` at least `PW` values. Invalid IDs or undersized
/// slices panic; validation is intentionally outside the per-splat hot loop.
pub mod splat {
    /// Pixels per compositor strip.
    pub const PW: usize = crate::arch::splat::PW;
    #[inline(always)]
    pub fn composite_strip_f32(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[f32],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_f32(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    #[inline(always)]
    pub fn composite_strip_f16(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_f16(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    #[inline(always)]
    pub fn composite_strip_e4m3(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_e4m3(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    /// Composite with post-activation RGB stored as unsigned E4M4.
    #[inline(always)]
    pub fn composite_strip_uf8e4m4(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_uf8e4m4(
            center, conic, opacity, color, ids, x0, y, rgb, t,
        )
    }
    /// Composite with post-activation RGB stored as unsigned E5M3.
    #[inline(always)]
    pub fn composite_strip_uf8e5m3(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_uf8e5m3(
            center, conic, opacity, color, ids, x0, y, rgb, t,
        )
    }
}

/// Software rasterization primitives (ISA-dispatched) — the coverage /
/// depth / interpolation / packing pipeline, re-exported for rendering
/// crates that drive triangle rasterization directly (e.g. `axrd`
/// examples). See `arch::portable::raster` for kernel documentation.
pub mod raster {
    #[inline(always)]
    pub fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
        crate::arch::raster::cmpge_f32(a, b, dst)
    }
    #[inline(always)]
    pub fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
        crate::arch::raster::and_masks(a, b, dst)
    }
    #[inline(always)]
    pub fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
        crate::arch::raster::maskstore_u32(mask, src, dst)
    }
    #[inline(always)]
    pub fn depth_test(coverage: &[u32], frag_z: &[f32], zbuf: &mut [f32], out_mask: &mut [u32]) {
        crate::arch::raster::depth_test(coverage, frag_z, zbuf, out_mask)
    }
    /// Experimental depth test for planes already stored as unsigned E5M11.
    /// Raw ordering is numeric for this non-negative format; NaNs fail as in
    /// the f32 kernel. Encoding a freshly interpolated F32 plane is not part
    /// of this kernel and can outweigh the bandwidth saving; benchmark that
    /// full producer/consumer pipeline before selecting this storage.
    #[inline(always)]
    pub fn depth_test_uf16e5m11(
        coverage: &[u32],
        frag_z: &[u16],
        zbuf: &mut [u16],
        out_mask: &mut [u32],
    ) {
        assert_eq!(frag_z.len(), coverage.len());
        assert_eq!(zbuf.len(), coverage.len());
        assert_eq!(out_mask.len(), coverage.len());
        crate::arch::raster::depth_test_uf16e5m11(coverage, frag_z, zbuf, out_mask)
    }
    #[inline(always)]
    pub fn bary_interp(
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        a0: f32,
        a1: f32,
        a2: f32,
        dst: &mut [f32],
    ) {
        crate::arch::raster::bary_interp(w0, w1, w2, a0, a1, a2, dst)
    }
    #[inline(always)]
    pub fn xrgb_pack(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
        crate::arch::raster::xrgb_pack(r, g, b, packed)
    }
    /// Pack normalized channels using round-to-nearest quantization.
    #[inline(always)]
    pub fn xrgb_pack_rounded(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
        assert!(r.len() >= packed.len() && g.len() >= packed.len() && b.len() >= packed.len());
        crate::arch::raster::xrgb_pack_rounded(r, g, b, packed)
    }
    /// Pack three biased-INT8 display planes (`stored = display - 128`,
    /// wrapping) into XRGB pixels: each channel is widened and wrapping-added
    /// 128, the X byte is forced opaque.
    #[inline(always)]
    pub fn xrgb_pack_u8_biased(r: &[u8], g: &[u8], b: &[u8], packed: &mut [u32]) {
        crate::arch::raster::xrgb_pack_u8_biased(r, g, b, packed)
    }
    #[inline(always)]
    pub fn alpha_blend(mask: &[u32], src: &[u32], dst: &mut [u32]) {
        crate::arch::raster::alpha_blend(mask, src, dst)
    }

    // ── Shape coverage rows (see `RasterBackend::rounded_fill` & co.) ──
    //
    // Every row kernel is allocation-free and asserts its slice lengths;
    // the SIMD tiers are bit-exact with the scalar tier.

    /// Signed distance from a rounded-rectangle edge for one row of pixel
    /// centres: `out[i]` at `(px0 + i, py)` in shape-centred coordinates,
    /// negative inside. `radius` is the radius of the corner whose
    /// quadrant the whole row segment lies in.
    #[inline(always)]
    pub fn rounded_sdf_row_f32(
        out: &mut [f32],
        px0: f32,
        py: f32,
        half_w: f32,
        half_h: f32,
        radius: f32,
    ) {
        crate::arch::raster::rounded_sdf_row_f32(out, px0, py, half_w, half_h, radius)
    }
    /// `cov = round(255 · clamp(0.5 − d, 0, 1))`.
    #[inline(always)]
    pub fn sdf_coverage_u8(d: &[f32], cov: &mut [u8]) {
        crate::arch::raster::sdf_coverage_u8(d, cov)
    }
    /// `cov = sat(cov(outer) − cov(inner))`.
    #[inline(always)]
    pub fn sdf_ring_coverage_u8(outer: &[f32], inner: &[f32], cov: &mut [u8]) {
        crate::arch::raster::sdf_ring_coverage_u8(outer, inner, cov)
    }
    /// Table coverage: `falloff[i]` is the coverage at distance `i − len/2`,
    /// linearly interpolated and clamped at both ends; `1..=32` entries.
    #[inline(always)]
    pub fn sdf_falloff_u8(d: &[f32], falloff: &[u8], cov: &mut [u8]) {
        crate::arch::raster::sdf_falloff_u8(d, falloff, cov)
    }
    /// Blend one premultiplied pixel over a row through 8-bit coverage.
    #[inline(always)]
    pub fn blend_row_alpha(dst: &mut [u32], pixel: u32, coverage: &[u8]) {
        crate::arch::raster::blend_row_alpha(dst, pixel, coverage)
    }
    /// `dst = over(src · mask, dst)` for premultiplied `src`.
    #[inline(always)]
    pub fn blend_row_masked(src: &[u32], mask: &[u8], dst: &mut [u32]) {
        crate::arch::raster::blend_row_masked(src, mask, dst)
    }
    /// Tint 8-bit coverage with straight `color` at `opacity`.
    #[inline(always)]
    pub fn coverage_blend_u8(coverage: &[u8], color: u32, opacity: u8, dst: &mut [u32]) {
        crate::arch::raster::coverage_blend_u8(coverage, color, opacity, dst)
    }
    /// Tint 16-bit coverage (quantized as `min((c + 128) >> 8, 255)`).
    #[inline(always)]
    pub fn coverage_blend_u16(coverage: &[u16], color: u32, opacity: u8, dst: &mut [u32]) {
        crate::arch::raster::coverage_blend_u16(coverage, color, opacity, dst)
    }
    /// Tint packed-alpha coverage (top byte of each word).
    #[inline(always)]
    pub fn coverage_blend_a32(coverage: &[u32], color: u32, opacity: u8, dst: &mut [u32]) {
        crate::arch::raster::coverage_blend_a32(coverage, color, opacity, dst)
    }
    /// Premultiply straight `0xXXRRGGBB` by `opacity` (exact ÷255 rounding).
    #[inline(always)]
    pub fn premultiply_u32(color: u32, opacity: u8) -> u32 {
        crate::arch::scalar::raster::premultiply_u32(color, opacity)
    }
    /// Scale every channel of a premultiplied pixel by `factor` (0..=255).
    #[inline(always)]
    pub fn scale_pixel_u8(pixel: u32, factor: u8) -> u32 {
        crate::arch::scalar::raster::scale_pixel_u8(pixel, u32::from(factor))
    }
}

/// Scalar B8/F8 (OCP E4M3/E5M2) codecs — for crates that store 8-bit floats
/// assets (voxel densities, SH coefficients, splat color tables).
pub mod fp8 {
    /// Decode one E4M3 byte (exact, all 256 encodings).
    #[inline]
    pub fn e4m3_to_f32(b: u8) -> f32 {
        crate::cast::e4m3_to_f32(b)
    }
    /// Decode one E5M2 byte (exact, all 256 encodings).
    #[inline]
    pub fn e5m2_to_f32(b: u8) -> f32 {
        crate::cast::e5m2_to_f32(b)
    }
    /// Narrow one f32 to E4M3, saturating at ±448 (not NaN-on-overflow: these are offline
    /// constants, where saturation is unambiguous). Round to nearest, ties to the lower pattern.
    #[inline]
    pub fn quant_e4m3(value: f32) -> u8 {
        crate::cast::f32_to_e4m3(value)
    }
    /// Narrow one f32 to E5M2, saturating at the largest finite value rather than producing an
    /// infinity. Round to nearest, ties to the lower pattern.
    #[inline]
    pub fn quant_e5m2(value: f32) -> u8 {
        crate::cast::f32_to_e5m2(value)
    }
    /// Dequantize a packed E4M3 slice to f32 (SIMD widen; exact).
    #[inline]
    pub fn dequant_e4m3(src: &[u8], dst: &mut [f32], n: usize) {
        crate::arch::f8::e4m3_to_f32(src, dst, n)
    }
    /// Dequantize a packed E5M2 slice to f32 (SIMD widen; exact).
    #[inline]
    pub fn dequant_e5m2(src: &[u8], dst: &mut [f32], n: usize) {
        crate::arch::f8::e5m2_to_f32(src, dst, n)
    }
    /// Dequantize a packed E4M3 slice to packed LE f16 (exact; on
    /// FP16-native targets this never touches f32).
    #[inline]
    pub fn dequant_e4m3_f16(src: &[u8], dst: &mut [u8], n: usize) {
        crate::arch::f8::e4m3_to_f16(src, dst, n)
    }
    /// Dequantize a packed E5M2 slice to packed LE f16 (a pure integer
    /// relayout — `byte << 8` is already the f16 pattern).
    #[inline]
    pub fn dequant_e5m2_f16(src: &[u8], dst: &mut [u8], n: usize) {
        crate::arch::f8::e5m2_to_f16(src, dst, n)
    }
    /// Fused E4M3 × F32 dot product with F32 accumulation.
    #[inline(always)]
    pub fn vec_dot_e4m3_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::f8::vec_dot_e4m3_f32(w, x, n)
    }
    /// Fused E5M2 × F32 dot product with F32 accumulation.
    #[inline(always)]
    pub fn vec_dot_e5m2_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::f8::vec_dot_e5m2_f32(w, x, n)
    }
    /// Fused E4M3 × E4M3 dot product with F32 accumulation.
    #[inline(always)]
    pub fn dot_e4m3_e4m3(w: &[u8], x: &[u8], n: usize) -> f32 {
        crate::arch::f8::dot_e4m3_e4m3(w, x, n)
    }
    /// Fused E5M2 × E5M2 dot product with F32 accumulation.
    #[inline(always)]
    pub fn dot_e5m2_e5m2(w: &[u8], x: &[u8], n: usize) -> f32 {
        crate::arch::f8::dot_e5m2_e5m2(w, x, n)
    }
    /// Double-F8 GEMM with F16 bias:
    /// `D = decode(A[m,k])·decode(B[n,k]) + widen(C[m,n])` — the FMLAL fast
    /// path on FEAT_FHM targets.
    #[inline]
    pub fn gemm_e4m3_e4m3_f16(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        d: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        crate::arch::f8::gemm_e4m3_e4m3_f16(a, b, c, d, m, n, k)
    }
    /// Double-F8 GEMM with F16 bias (E5M2 operands).
    #[inline]
    pub fn gemm_e5m2_e5m2_f16(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        d: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        crate::arch::f8::gemm_e5m2_e5m2_f16(a, b, c, d, m, n, k)
    }
    /// Nearest-value E4M3 encode by exhaustive search over the 256
    /// encodings — setup/offline quality, not a hot-path quantizer.
    pub fn f32_to_e4m3_nearest(v: f32) -> u8 {
        let mut best = (f32::INFINITY, 0u8);
        let mut b = 0u8;
        loop {
            let d = e4m3_to_f32(b);
            if !d.is_nan() {
                let err = (d - v).abs();
                if err < best.0 {
                    best = (err, b);
                }
            }
            if b == 255 {
                return best.1;
            }
            b += 1;
        }
    }

    /// Round one finite value to OCP E4M3 (ties-to-even), saturating
    /// overflow to the largest finite encoding. This is the hot-path
    /// counterpart to [`f32_to_e4m3_nearest`].
    ///
    /// Rounding happens once, on the F32 bits: an intermediate F16 would
    /// round first and can turn an above-midpoint value into a tie.
    #[inline]
    pub fn f32_to_e4m3_rne(v: f32) -> u8 {
        if v.is_nan() {
            return 0x7f;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 24) & 0x80) as u8;
        let magnitude = bits & 0x7fff_ffff;
        // 448.0, the largest finite E4M3; larger values and infinities
        // saturate.
        if magnitude >= 0x43e0_0000 {
            return sign | 0x7e;
        }
        let exponent = (magnitude >> 23) as i32 - 127;
        let rounded = if exponent >= -6 {
            // Normal E4M3: rebias the exponent from 127 to 7 and round the
            // 20 dropped mantissa bits to even. A carry into the exponent is
            // the correct next binade; `magnitude < 448` bounds it at 0x7e.
            let rebiased = magnitude - (120 << 23);
            (rebiased + 0x7_ffff + ((rebiased >> 20) & 1)) >> 20
        } else {
            // Subnormal E4M3 counts 2^-9 steps: round the F32 significand
            // times 2^(exponent + 9). A carry to 8 is the smallest normal.
            let shift = (14 - exponent) as u32;
            if shift >= 32 {
                0
            } else {
                let significand = (magnitude & 0x7f_ffff) | 0x80_0000;
                let half = 1 << (shift - 1);
                (significand + half - 1 + ((significand >> shift) & 1)) >> shift
            }
        };
        sign | rounded as u8
    }

    /// Round one finite value to OCP E5M2 (ties-to-even), saturating
    /// overflow and infinities to the largest finite encoding.
    ///
    /// Rounding happens once, on the F32 bits: an intermediate F16 would
    /// round first and can turn an above-midpoint value into a tie.
    #[inline]
    pub fn f32_to_e5m2_rne(v: f32) -> u8 {
        if v.is_nan() {
            return 0x7e;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 24) & 0x80) as u8;
        let magnitude = bits & 0x7fff_ffff;
        // 57344.0, the largest finite E5M2; larger values and infinities
        // saturate.
        if magnitude >= 0x4760_0000 {
            return sign | 0x7b;
        }
        let exponent = (magnitude >> 23) as i32 - 127;
        let rounded = if exponent >= -14 {
            // Normal E5M2: rebias the exponent from 127 to 15 and round the
            // 21 dropped mantissa bits to even. A carry into the exponent is
            // the correct next binade; `magnitude < 57344` bounds it at 0x7b.
            let rebiased = magnitude - (112 << 23);
            (rebiased + 0xf_ffff + ((rebiased >> 21) & 1)) >> 21
        } else {
            // Subnormal E5M2 counts 2^-16 steps: round the F32 significand
            // times 2^(exponent + 16). A carry to 4 is the smallest normal.
            let shift = (7 - exponent) as u32;
            if shift >= 32 {
                0
            } else {
                let significand = (magnitude & 0x7f_ffff) | 0x80_0000;
                let half = 1 << (shift - 1);
                (significand + half - 1 + ((significand >> shift) & 1)) >> shift
            }
        };
        sign | rounded as u8
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hot_e4m3_encoder_roundtrips_every_finite_value() {
            for bits in u8::MIN..=u8::MAX {
                let value = e4m3_to_f32(bits);
                if value.is_finite() {
                    assert_eq!(f32_to_e4m3_rne(value), bits, "encoding {bits:#04x}");
                }
            }
        }

        /// Every rounding decision sits at a midpoint between adjacent
        /// encodings: check each midpoint and its F32 neighbours on both
        /// sides, both signs, the saturation edge and non-finite inputs.
        #[test]
        fn hot_e4m3_encoder_rounds_once_to_nearest_even() {
            let next_up = |v: f32| f32::from_bits(v.to_bits() + 1);
            let next_down = |v: f32| f32::from_bits(v.to_bits() - 1);
            for low in 0_u8..0x7e {
                let (a, b) = (e4m3_to_f32(low), e4m3_to_f32(low + 1));
                let midpoint = (a + b) * 0.5;
                // Ties go to the even encoding.
                let even = if low & 1 == 0 { low } else { low + 1 };
                for (value, expected) in [
                    (next_down(midpoint), low),
                    (midpoint, even),
                    (next_up(midpoint), low + 1),
                ] {
                    for (value, expected) in [(value, expected), (-value, expected | 0x80)] {
                        assert_eq!(
                            f32_to_e4m3_rne(value),
                            expected,
                            "{value:e} between {low:#04x} and {:#04x}",
                            low + 1
                        );
                    }
                }
            }
            // The double-rounding counterexample: F16 rounds 1.0625 + 2^-20
            // to the 1.0625 tie, which then rounds down to even.
            assert_eq!(f32_to_e4m3_rne(1.0625 + 1.0 / 1048576.0), 0x39);
            for value in [448.0, 464.0, 1.0e30, f32::MAX, f32::INFINITY] {
                assert_eq!(f32_to_e4m3_rne(value), 0x7e);
                assert_eq!(f32_to_e4m3_rne(-value), 0xfe);
            }
            assert_eq!(f32_to_e4m3_rne(f32::NAN), 0x7f);
            assert_eq!(f32_to_e4m3_rne(0.0), 0x00);
            assert_eq!(f32_to_e4m3_rne(-0.0), 0x80);
            assert_eq!(f32_to_e4m3_rne(f32::from_bits(1)), 0x00);
            assert_eq!(f32_to_e4m3_rne(-f32::MIN_POSITIVE), 0x80);
        }

        #[test]
        fn hot_e5m2_encoder_roundtrips_every_finite_value() {
            for bits in u8::MIN..=u8::MAX {
                let value = e5m2_to_f32(bits);
                if value.is_finite() {
                    assert_eq!(f32_to_e5m2_rne(value), bits, "encoding {bits:#04x}");
                }
            }
        }

        /// Every rounding decision sits at a midpoint between adjacent
        /// encodings: check each midpoint and its F32 neighbours on both
        /// sides, both signs, the saturation edge and non-finite inputs.
        #[test]
        fn hot_e5m2_encoder_rounds_once_to_nearest_even() {
            let next_up = |v: f32| f32::from_bits(v.to_bits() + 1);
            let next_down = |v: f32| f32::from_bits(v.to_bits() - 1);
            for low in 0_u8..0x7b {
                let (a, b) = (e5m2_to_f32(low), e5m2_to_f32(low + 1));
                let midpoint = (a + b) * 0.5;
                // Ties go to the even encoding.
                let even = if low & 1 == 0 { low } else { low + 1 };
                for (value, expected) in [
                    (next_down(midpoint), low),
                    (midpoint, even),
                    (next_up(midpoint), low + 1),
                ] {
                    for (value, expected) in [(value, expected), (-value, expected | 0x80)] {
                        assert_eq!(
                            f32_to_e5m2_rne(value),
                            expected,
                            "{value:e} between {low:#04x} and {:#04x}",
                            low + 1
                        );
                    }
                }
            }
            // The double-rounding counterexample: F16 rounds 1.125 + 2^-20
            // to the 1.125 tie, which then rounds down to even.
            assert_eq!(f32_to_e5m2_rne(1.125 + 1.0 / 1048576.0), 0x3d);
            // 61440 is the tie between 57344 and 2^16, which would overflow.
            for value in [57344.0, 57352.0, 61440.0, 1.0e30, f32::MAX, f32::INFINITY] {
                assert_eq!(f32_to_e5m2_rne(value), 0x7b);
                assert_eq!(f32_to_e5m2_rne(-value), 0xfb);
            }
            assert_eq!(f32_to_e5m2_rne(f32::NAN), 0x7e);
            assert_eq!(f32_to_e5m2_rne(0.0), 0x00);
            assert_eq!(f32_to_e5m2_rne(-0.0), 0x80);
            assert_eq!(f32_to_e5m2_rne(f32::from_bits(1)), 0x00);
            assert_eq!(f32_to_e5m2_rne(-f32::MIN_POSITIVE), 0x80);
        }
    }
}

/// Unsigned-float storage codecs and SIMD kernels.
///
/// These formats are reserved for quantities whose physical domain is
/// non-negative (density, extinction, emitted/post-activation color, depth).
/// They intentionally are not `axnn::DType` variants: current graph and TOSA
/// device ABIs cannot identify them. Promote UF8 to F16 or UF16 to F32 at an
/// accelerator boundary rather than relabelling the raw bits as F8/U8.
pub mod uf {
    /// Decode one unsigned E4M4 byte exactly.
    #[inline]
    pub fn uf8e4m4_to_f32(bits: u8) -> f32 {
        crate::cast::uf8e4m4_to_f32(bits)
    }

    /// Decode one unsigned E5M3 byte exactly.
    #[inline]
    pub fn uf8e5m3_to_f32(bits: u8) -> f32 {
        crate::cast::uf8e5m3_to_f32(bits)
    }

    /// Decode one unsigned E5M11 word exactly.
    #[inline]
    pub fn uf16e5m11_to_f32(bits: u16) -> f32 {
        crate::cast::uf16e5m11_to_f32(bits)
    }

    /// Encode E4M4 with round-to-nearest, ties-to-even. Negative nonzero
    /// values and NaNs become canonical NaN, `-0.0` becomes `+0`, and
    /// positive overflow becomes infinity.
    #[inline]
    pub fn f32_to_uf8e4m4_rne(value: f32) -> u8 {
        crate::cast::f32_to_uf8e4m4_rne(value)
    }

    /// Encode E5M3 with round-to-nearest, ties-to-even. Negative nonzero
    /// values and NaNs become canonical NaN, `-0.0` becomes `+0`, and
    /// positive overflow becomes infinity.
    #[inline]
    pub fn f32_to_uf8e5m3_rne(value: f32) -> u8 {
        crate::cast::f32_to_uf8e5m3_rne(value)
    }

    /// Encode E5M11 with round-to-nearest, ties-to-even. Negative nonzero
    /// values and NaNs become canonical NaN, `-0.0` becomes `+0`, and
    /// positive overflow becomes infinity.
    #[inline]
    pub fn f32_to_uf16e5m11_rne(value: f32) -> u16 {
        crate::cast::f32_to_uf16e5m11_rne(value)
    }

    /// Decode packed E4M4 storage to f32 through the selected SIMD tier.
    /// Every encoding is exact, including signaling/quiet NaN payload bits.
    #[inline]
    pub fn dequant_uf8e4m4(src: &[u8], dst: &mut [f32], n: usize) {
        assert!(src.len() >= n && dst.len() >= n);
        crate::arch::uf::uf8e4m4_to_f32(src, dst, n)
    }

    /// Decode packed E5M3 storage to f32 through the selected SIMD tier.
    /// Every encoding is exact, including signaling/quiet NaN payload bits.
    #[inline]
    pub fn dequant_uf8e5m3(src: &[u8], dst: &mut [f32], n: usize) {
        assert!(src.len() >= n && dst.len() >= n);
        crate::arch::uf::uf8e5m3_to_f32(src, dst, n)
    }

    /// Promote native-endian, in-memory unsigned E5M11 words to F32 without
    /// losing their extra fraction bit. Serialized/device bytes should use
    /// [`promote_uf16e5m11_le_to_f32`] instead.
    #[inline]
    pub fn promote_uf16e5m11_to_f32(src: &[u16], dst: &mut [f32], n: usize) {
        assert!(src.len() >= n && dst.len() >= n);
        crate::arch::uf::uf16e5m11_to_f32(src, dst, n)
    }

    /// Promote packed little-endian unsigned E5M11 to F32. This is the
    /// lossless host expansion for serialized assets and F32 device inputs.
    #[inline]
    pub fn promote_uf16e5m11_le_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
        let bytes = n.checked_mul(2).expect("UF16 promotion length overflow");
        assert!(src.len() >= bytes && dst.len() >= n);
        crate::arch::uf::uf16e5m11_le_to_f32(src, dst, n)
    }

    /// Promote packed E4M4 to packed little-endian F16. This is the lawful
    /// lossless UF8 boundary expansion for an F16 accelerator input.
    #[inline]
    pub fn promote_uf8e4m4_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
        let bytes = n.checked_mul(2).expect("UF8 promotion length overflow");
        assert!(src.len() >= n && dst.len() >= bytes);
        crate::arch::uf::uf8e4m4_to_f16(src, dst, n)
    }

    /// Promote packed E5M3 to packed little-endian F16, losslessly.
    #[inline]
    pub fn promote_uf8e5m3_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
        let bytes = n.checked_mul(2).expect("UF8 promotion length overflow");
        assert!(src.len() >= n && dst.len() >= bytes);
        crate::arch::uf::uf8e5m3_to_f16(src, dst, n)
    }

    #[cfg(test)]
    mod tests {
        use alloc::vec;
        use alloc::vec::Vec;

        use super::*;

        #[derive(Clone, Copy)]
        struct Error {
            squared: f64,
            max: f32,
        }

        fn measure(
            limit: f32,
            signed_encode: impl Fn(f32) -> f32,
            unsigned_encode: impl Fn(f32) -> f32,
        ) -> (Error, Error) {
            let (mut signed, mut unsigned) = (
                Error {
                    squared: 0.0,
                    max: 0.0,
                },
                Error {
                    squared: 0.0,
                    max: 0.0,
                },
            );
            for i in 0..=65_536 {
                let value = limit * (i as f32 / 65_536.0);
                let se = (signed_encode(value) - value).abs();
                let ue = (unsigned_encode(value) - value).abs();
                signed.squared += (se as f64) * (se as f64);
                unsigned.squared += (ue as f64) * (ue as f64);
                signed.max = signed.max.max(se);
                unsigned.max = unsigned.max.max(ue);
            }
            (signed, unsigned)
        }

        fn assert_same_float(got: f32, want: f32, context: &str) {
            if want.is_nan() {
                assert!(got.is_nan(), "{context}");
            } else {
                assert_eq!(got, want, "{context}");
            }
        }

        #[test]
        fn public_bulk_wrappers_cover_chunk_boundaries_and_preserve_sentinels() {
            // Covers both current lane widths (portable W=4, AVX2/NEON W=8)
            // and the four-way unroll boundaries used by the conversion tiers.
            const LENGTHS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];
            const F32_SENTINEL: f32 = -12_345.5;
            const BYTE_SENTINEL: u8 = 0xA5;

            for &n in LENGTHS {
                let src8: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
                for (kernel, decode, name) in [
                    (
                        dequant_uf8e4m4 as fn(&[u8], &mut [f32], usize),
                        uf8e4m4_to_f32 as fn(u8) -> f32,
                        "UF8 E4M4",
                    ),
                    (
                        dequant_uf8e5m3 as fn(&[u8], &mut [f32], usize),
                        uf8e5m3_to_f32 as fn(u8) -> f32,
                        "UF8 E5M3",
                    ),
                ] {
                    let mut dst = vec![F32_SENTINEL; n + 3];
                    kernel(&src8, &mut dst, n);
                    for i in 0..n {
                        assert_same_float(dst[i], decode(src8[i]), name);
                    }
                    assert!(dst[n..].iter().all(|&value| value == F32_SENTINEL));
                }

                let src16: Vec<u16> = (0..n).map(|i| (i * 8_191 + 17) as u16).collect();
                let mut native_dst = vec![F32_SENTINEL; n + 3];
                promote_uf16e5m11_to_f32(&src16, &mut native_dst, n);
                for i in 0..n {
                    assert_same_float(
                        native_dst[i],
                        uf16e5m11_to_f32(src16[i]),
                        "native UF16 E5M11",
                    );
                }
                assert!(native_dst[n..].iter().all(|&value| value == F32_SENTINEL));

                let packed16: Vec<u8> = src16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let mut packed_dst = vec![F32_SENTINEL; n + 3];
                promote_uf16e5m11_le_to_f32(&packed16, &mut packed_dst, n);
                for i in 0..n {
                    assert_same_float(
                        packed_dst[i],
                        uf16e5m11_to_f32(src16[i]),
                        "packed UF16 E5M11",
                    );
                }
                assert!(packed_dst[n..].iter().all(|&value| value == F32_SENTINEL));

                let mut e4_f16 = vec![BYTE_SENTINEL; n * 2 + 5];
                promote_uf8e4m4_to_f16(&src8, &mut e4_f16, n);
                for i in 0..n {
                    assert_eq!(
                        u16::from_le_bytes([e4_f16[i * 2], e4_f16[i * 2 + 1]]),
                        crate::cast::uf8e4m4_to_f16_bits(src8[i]),
                    );
                }
                assert!(e4_f16[n * 2..].iter().all(|&byte| byte == BYTE_SENTINEL));

                let mut e5_f16 = vec![BYTE_SENTINEL; n * 2 + 5];
                promote_uf8e5m3_to_f16(&src8, &mut e5_f16, n);
                for i in 0..n {
                    assert_eq!(
                        u16::from_le_bytes([e5_f16[i * 2], e5_f16[i * 2 + 1]]),
                        (src8[i] as u16) << 7,
                    );
                }
                assert!(e5_f16[n * 2..].iter().all(|&byte| byte == BYTE_SENTINEL));
            }
        }

        #[repr(C, align(32))]
        struct Aligned<T>(T);

        #[test]
        fn public_bulk_wrappers_accept_element_aligned_nonvector_offsets() {
            const N: usize = 33;
            const SENTINEL: f32 = -12_345.5;

            let mut src8 = Aligned([0u8; N + 1]);
            for i in 0..N {
                src8.0[i + 1] = (i * 37 + 11) as u8;
            }
            let src8 = &src8.0[1..];
            assert_ne!((src8.as_ptr() as usize) & 31, 0);
            for (kernel, decode, name) in [
                (
                    dequant_uf8e4m4 as fn(&[u8], &mut [f32], usize),
                    uf8e4m4_to_f32 as fn(u8) -> f32,
                    "UF8 E4M4",
                ),
                (
                    dequant_uf8e5m3 as fn(&[u8], &mut [f32], usize),
                    uf8e5m3_to_f32 as fn(u8) -> f32,
                    "UF8 E5M3",
                ),
            ] {
                let mut storage = Aligned([SENTINEL; N + 2]);
                let dst = &mut storage.0[1..];
                assert_ne!((dst.as_ptr() as usize) & 31, 0);
                kernel(src8, dst, N);
                for i in 0..N {
                    assert_same_float(dst[i], decode(src8[i]), name);
                }
                assert_eq!(dst[N], SENTINEL, "{name} wrote past n");
            }

            for (kernel, encode, name) in [
                (
                    promote_uf8e4m4_to_f16 as fn(&[u8], &mut [u8], usize),
                    crate::cast::uf8e4m4_to_f16_bits as fn(u8) -> u16,
                    "UF8 E4M4 to F16",
                ),
                (
                    promote_uf8e5m3_to_f16 as fn(&[u8], &mut [u8], usize),
                    (|raw| (raw as u16) << 7) as fn(u8) -> u16,
                    "UF8 E5M3 to F16",
                ),
            ] {
                let mut storage = Aligned([0xA5u8; N * 2 + 2]);
                let dst = &mut storage.0[1..];
                assert_ne!((dst.as_ptr() as usize) & 31, 0);
                kernel(src8, dst, N);
                for i in 0..N {
                    assert_eq!(
                        u16::from_le_bytes([dst[i * 2], dst[i * 2 + 1]]),
                        encode(src8[i]),
                        "{name}",
                    );
                }
                assert_eq!(dst[N * 2], 0xA5, "{name} wrote past n");
            }

            let mut src16 = Aligned([0u16; N + 1]);
            for i in 0..N {
                src16.0[i + 1] = (i * 8_191 + 17) as u16;
            }
            let src16 = &src16.0[1..];
            let mut native_storage = Aligned([SENTINEL; N + 2]);
            let native_dst = &mut native_storage.0[1..];
            promote_uf16e5m11_to_f32(src16, native_dst, N);
            for i in 0..N {
                assert_same_float(
                    native_dst[i],
                    uf16e5m11_to_f32(src16[i]),
                    "native UF16 E5M11",
                );
            }
            assert_eq!(native_dst[N], SENTINEL);

            let mut packed = Aligned([0u8; N * 2 + 1]);
            for i in 0..N {
                let bytes = src16[i].to_le_bytes();
                packed.0[i * 2 + 1] = bytes[0];
                packed.0[i * 2 + 2] = bytes[1];
            }
            let packed = &packed.0[1..];
            let mut packed_storage = Aligned([SENTINEL; N + 2]);
            let packed_dst = &mut packed_storage.0[1..];
            promote_uf16e5m11_le_to_f32(packed, packed_dst, N);
            for i in 0..N {
                assert_same_float(packed_dst[i], native_dst[i], "packed UF16 E5M11");
            }
            assert_eq!(packed_dst[N], SENTINEL);
        }

        #[test]
        fn extra_fraction_bit_strictly_improves_nonnegative_precision() {
            let (e4, u4) = measure(
                150.0,
                |v| crate::fp8::e4m3_to_f32(crate::fp8::f32_to_e4m3_rne(v)),
                |v| uf8e4m4_to_f32(f32_to_uf8e4m4_rne(v)),
            );
            let (e5, u5) = measure(
                10_000.0,
                |v| crate::fp8::e5m2_to_f32(crate::fp8::f32_to_e5m2_rne(v)),
                |v| uf8e5m3_to_f32(f32_to_uf8e5m3_rne(v)),
            );
            let (f16e, u16e) = measure(
                1.0,
                |v| (v as f16) as f32,
                |v| uf16e5m11_to_f32(f32_to_uf16e5m11_rne(v)),
            );

            for (name, signed, unsigned) in [
                ("E4M3 vs UF8 E4M4", e4, u4),
                ("E5M2 vs UF8 E5M3", e5, u5),
                ("F16 vs UF16 E5M11", f16e, u16e),
            ] {
                assert!(
                    unsigned.squared < signed.squared * 0.3,
                    "{name}: {} vs {}",
                    unsigned.squared,
                    signed.squared
                );
                assert!(unsigned.max < signed.max, "{name} max error");
            }
        }
    }
}

/// Benchmark access to internal kernels.
///
/// **Not part of the public API** — only available with `--features bench`.
/// Thin inline wrappers around `pub(crate)` functions so the `benches/`
/// binary can call them directly for micro-benchmarking.
#[cfg(feature = "bench")]
pub mod _bench {
    // ── Block constants ──
    pub const Q8_BLOCK_SIZE: usize = crate::quant::Q8_BLOCK_SIZE;
    pub const Q8_BLOCK_BYTES: usize = crate::quant::Q8_BLOCK_BYTES;
    pub const Q4_BLOCK_SIZE: usize = crate::quant::Q4_BLOCK_SIZE;
    pub const Q4_BLOCK_BYTES: usize = crate::quant::Q4_BLOCK_BYTES;
    pub const Q6_BLOCK_SIZE: usize = crate::quant::Q6_BLOCK_SIZE;
    pub const Q6_BLOCK_BYTES: usize = crate::quant::Q6_BLOCK_BYTES;

    // ── Quantize helpers (for building benchmark inputs) ──
    #[inline(always)]
    pub fn quantize_q8(s: &[f32], d: &mut [u8]) {
        crate::quant::quantize_q8(s, d)
    }
    #[inline(always)]
    pub fn quantize_q4(s: &[f32], d: &mut [u8]) {
        crate::quant::quantize_q4(s, d)
    }
    #[inline(always)]
    pub fn quantize_q6(s: &[f32], d: &mut [u8]) {
        crate::quant::quantize_q6(s, d)
    }

    // ── qdot kernels (hot path) ──
    #[inline(always)]
    pub fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
        crate::arch::qdot::vec_dot_q8_q8(x, y, n)
    }
    #[inline(always)]
    pub fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
        crate::arch::qdot::vec_dot_q4_q8(x, y, n)
    }
    #[inline(always)]
    pub fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
        crate::arch::qdot::vec_dot_q6_q8(x, y, n)
    }
    #[inline(always)]
    pub fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::qdot::gemm_q8_q8(w, x, c, m, n, k)
    }
    #[inline(always)]
    pub fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::qdot::gemm_q4_q8(w, x, c, m, n, k)
    }
    #[inline(always)]
    pub fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::qdot::gemm_q6_q8(w, x, c, m, n, k)
    }

    // ── Vec raster primitives ──
    #[inline(always)]
    pub fn sub_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
        crate::arch::vec::sub_f32(a, b, dst)
    }
    #[inline(always)]
    pub fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
        crate::arch::raster::cmpge_f32(a, b, dst)
    }
    #[inline(always)]
    pub fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
        crate::arch::raster::and_masks(a, b, dst)
    }
    #[inline(always)]
    pub fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
        crate::arch::raster::maskstore_u32(mask, src, dst)
    }
    #[inline(always)]
    pub fn fused_edge_coverage(
        e0: &[f32],
        e1: &[f32],
        e2: &[f32],
        thresh: f32,
        color: &[u32],
        fb: &mut [u32],
    ) {
        crate::arch::raster::fused_edge_coverage(e0, e1, e2, thresh, color, fb)
    }
    #[inline(always)]
    pub fn depth_test(coverage: &[u32], frag_z: &[f32], zbuf: &mut [f32], out_mask: &mut [u32]) {
        crate::arch::raster::depth_test(coverage, frag_z, zbuf, out_mask)
    }
    #[inline(always)]
    pub fn bary_interp(
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        a0: f32,
        a1: f32,
        a2: f32,
        dst: &mut [f32],
    ) {
        crate::arch::raster::bary_interp(w0, w1, w2, a0, a1, a2, dst)
    }
    #[inline(always)]
    pub fn alpha_blend(mask: &[u32], src: &[u32], dst: &mut [u32]) {
        crate::arch::raster::alpha_blend(mask, src, dst)
    }
    #[inline(always)]
    pub fn xrgb_unpack(packed: &[u32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
        crate::arch::raster::xrgb_unpack(packed, r, g, b)
    }
    #[inline(always)]
    pub fn xrgb_pack(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
        crate::arch::raster::xrgb_pack(r, g, b, packed)
    }
    #[inline(always)]
    pub fn xrgb_pack_rounded(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
        crate::arch::raster::xrgb_pack_rounded(r, g, b, packed)
    }
    #[inline(always)]
    pub fn xrgb_pack_u8_biased(r: &[u8], g: &[u8], b: &[u8], packed: &mut [u32]) {
        crate::arch::raster::xrgb_pack_u8_biased(r, g, b, packed)
    }
    #[inline(always)]
    pub fn rounded_sdf_row_f32(out: &mut [f32], px0: f32, py: f32, hw: f32, hh: f32, r: f32) {
        crate::arch::raster::rounded_sdf_row_f32(out, px0, py, hw, hh, r)
    }
    #[inline(always)]
    pub fn sdf_coverage_u8(d: &[f32], cov: &mut [u8]) {
        crate::arch::raster::sdf_coverage_u8(d, cov)
    }
    #[inline(always)]
    pub fn blend_row_alpha(dst: &mut [u32], pixel: u32, coverage: &[u8]) {
        crate::arch::raster::blend_row_alpha(dst, pixel, coverage)
    }
    #[inline(always)]
    pub fn coverage_blend_u16(coverage: &[u16], color: u32, opacity: u8, dst: &mut [u32]) {
        crate::arch::raster::coverage_blend_u16(coverage, color, opacity, dst)
    }
    #[inline(always)]
    pub fn blit_opaque(
        src: &[u32],
        src_pitch: usize,
        dst: &mut [u32],
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) {
        crate::arch::raster::blit_opaque(src, src_pitch, dst, dst_pitch, width, height)
    }
    #[inline(always)]
    pub fn blit_blend(
        src: &[u32],
        src_pitch: usize,
        dst: &mut [u32],
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) {
        crate::arch::raster::blit_blend(src, src_pitch, dst, dst_pitch, width, height)
    }

    // ── Scalar baseline (always portable/scalar, for A/B comparison) ──
    pub mod baseline {
        // Raster ops: scalar tier (no core::simd)
        #[inline(always)]
        pub fn sub_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
            crate::arch::scalar::vec::sub_f32(a, b, dst)
        }
        #[inline(always)]
        pub fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
            crate::arch::scalar::raster::cmpge_f32(a, b, dst)
        }
        #[inline(always)]
        pub fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
            crate::arch::scalar::raster::and_masks(a, b, dst)
        }
        #[inline(always)]
        pub fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
            crate::arch::scalar::raster::maskstore_u32(mask, src, dst)
        }
        #[inline(always)]
        pub fn fused_edge_coverage(
            e0: &[f32],
            e1: &[f32],
            e2: &[f32],
            thresh: f32,
            color: &[u32],
            fb: &mut [u32],
        ) {
            crate::arch::scalar::raster::fused_edge_coverage(e0, e1, e2, thresh, color, fb)
        }
        #[inline(always)]
        pub fn depth_test(
            coverage: &[u32],
            frag_z: &[f32],
            zbuf: &mut [f32],
            out_mask: &mut [u32],
        ) {
            crate::arch::scalar::raster::depth_test(coverage, frag_z, zbuf, out_mask)
        }
        #[inline(always)]
        pub fn bary_interp(
            w0: &[f32],
            w1: &[f32],
            w2: &[f32],
            a0: f32,
            a1: f32,
            a2: f32,
            dst: &mut [f32],
        ) {
            crate::arch::scalar::raster::bary_interp(w0, w1, w2, a0, a1, a2, dst)
        }
        #[inline(always)]
        pub fn alpha_blend(mask: &[u32], src: &[u32], dst: &mut [u32]) {
            crate::arch::scalar::raster::alpha_blend(mask, src, dst)
        }
        #[inline(always)]
        pub fn xrgb_unpack(packed: &[u32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
            crate::arch::scalar::raster::xrgb_unpack(packed, r, g, b)
        }
        #[inline(always)]
        pub fn xrgb_pack(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
            crate::arch::scalar::raster::xrgb_pack(r, g, b, packed)
        }
        #[inline(always)]
        pub fn xrgb_pack_rounded(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
            crate::arch::scalar::raster::xrgb_pack_rounded(r, g, b, packed)
        }
        #[inline(always)]
        pub fn xrgb_pack_u8_biased(r: &[u8], g: &[u8], b: &[u8], packed: &mut [u32]) {
            crate::arch::scalar::raster::xrgb_pack_u8_biased(r, g, b, packed)
        }
        #[inline(always)]
        pub fn rounded_sdf_row_f32(out: &mut [f32], px0: f32, py: f32, hw: f32, hh: f32, r: f32) {
            crate::arch::scalar::raster::rounded_sdf_row_f32(out, px0, py, hw, hh, r)
        }
        #[inline(always)]
        pub fn sdf_coverage_u8(d: &[f32], cov: &mut [u8]) {
            crate::arch::scalar::raster::sdf_coverage_u8(d, cov)
        }
        #[inline(always)]
        pub fn blend_row_alpha(dst: &mut [u32], pixel: u32, coverage: &[u8]) {
            crate::arch::scalar::raster::blend_row_alpha(dst, pixel, coverage)
        }
        #[inline(always)]
        pub fn coverage_blend_u16(coverage: &[u16], color: u32, opacity: u8, dst: &mut [u32]) {
            crate::arch::scalar::raster::coverage_blend_u16(coverage, color, opacity, dst)
        }
        #[inline(always)]
        pub fn blit_opaque(
            src: &[u32],
            src_pitch: usize,
            dst: &mut [u32],
            dst_pitch: usize,
            width: usize,
            height: usize,
        ) {
            crate::arch::scalar::raster::blit_opaque(src, src_pitch, dst, dst_pitch, width, height)
        }
        #[inline(always)]
        pub fn blit_blend(
            src: &[u32],
            src_pitch: usize,
            dst: &mut [u32],
            dst_pitch: usize,
            width: usize,
            height: usize,
        ) {
            crate::arch::scalar::raster::blit_blend(src, src_pitch, dst, dst_pitch, width, height)
        }
        #[inline(always)]
        pub fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
            crate::arch::portable::qdot::vec_dot_q8_q8(x, y, n)
        }
        #[inline(always)]
        pub fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
            crate::arch::portable::qdot::vec_dot_q4_q8(x, y, n)
        }
        #[inline(always)]
        pub fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
            crate::arch::portable::qdot::vec_dot_q6_q8(x, y, n)
        }
        #[inline(always)]
        pub fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
            crate::arch::portable::qdot::gemm_q8_q8(w, x, c, m, n, k)
        }
        #[inline(always)]
        pub fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
            crate::arch::portable::qdot::gemm_q4_q8(w, x, c, m, n, k)
        }
        #[inline(always)]
        pub fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
            crate::arch::portable::qdot::gemm_q6_q8(w, x, c, m, n, k)
        }
    }

    // ── Bitmat (Q1) kernels ──
    #[inline(always)]
    pub fn bitmat_or_row_into(dst: &mut [u8], src: &[u8]) {
        crate::arch::bitmat::or_row_into(dst, src)
    }
    #[inline(always)]
    pub fn bitmat_row_popcount(row: &[u8]) -> u32 {
        crate::arch::bitmat::row_popcount(row)
    }
    #[inline(always)]
    pub fn bitmat_col_popcounts(
        data: &[u8],
        row_bytes: usize,
        n: usize,
    ) -> heapless::Vec<u32, { crate::arch::bitmat::MAX_NODES }> {
        crate::arch::bitmat::col_popcounts(data, row_bytes, n)
    }
    #[inline(always)]
    pub fn bitmat_topo_sort(
        data: &[u8],
        row_bytes: usize,
        n: usize,
    ) -> Option<heapless::Vec<u16, { crate::arch::bitmat::MAX_NODES }>> {
        crate::arch::bitmat::topo_sort(data, row_bytes, n)
    }

    // ── Vec elementwise kernels ──
    #[inline(always)]
    pub fn relu_f32(src: &[f32], dst: &mut [f32]) {
        crate::arch::vec::relu_f32(src, dst)
    }
    #[inline(always)]
    pub fn neg_f32(src: &[f32], dst: &mut [f32]) {
        crate::arch::vec::neg_f32(src, dst)
    }
    #[inline(always)]
    pub fn add_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
        crate::arch::vec::add_f32(a, b, dst)
    }
    #[inline(always)]
    pub fn mul_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
        crate::arch::vec::mul_f32(a, b, dst)
    }
    #[inline(always)]
    pub fn scale_f32(src: &[f32], dst: &mut [f32], s: f32) {
        crate::arch::vec::scale_f32(src, dst, s)
    }
    #[inline(always)]
    pub fn silu_f32(src: &[f32], dst: &mut [f32]) {
        crate::arch::vec::silu_f32(src, dst)
    }
    #[inline(always)]
    pub fn gelu_f32(src: &[f32], dst: &mut [f32]) {
        crate::arch::vec::gelu_f32(src, dst)
    }

    // ── Cast kernels ──
    #[inline(always)]
    pub fn quant_f32_q8(s: &[f32], d: &mut [u8], n: usize) {
        crate::arch::cast::quant_f32_q8(s, d, n)
    }
    #[inline(always)]
    pub fn quant_f32_q4(s: &[f32], d: &mut [u8], n: usize) {
        crate::arch::cast::quant_f32_q4(s, d, n)
    }
    #[inline(always)]
    pub fn dequant_q8_f32(s: &[u8], d: &mut [f32], n: usize) {
        crate::arch::cast::dequant_q8_f32(s, d, n)
    }
    #[inline(always)]
    pub fn dequant_q4_f32(s: &[u8], d: &mut [f32], n: usize) {
        crate::arch::cast::dequant_q4_f32(s, d, n)
    }

    // ── F8 kernels (dequant + fused dot) ──
    #[inline(always)]
    pub fn dequant_e4m3_f32(s: &[u8], d: &mut [f32], n: usize) {
        crate::arch::f8::e4m3_to_f32(s, d, n)
    }
    #[inline(always)]
    pub fn dequant_e5m2_f32(s: &[u8], d: &mut [f32], n: usize) {
        crate::arch::f8::e5m2_to_f32(s, d, n)
    }
    #[inline(always)]
    pub fn dequant_e4m3_f16(s: &[u8], d: &mut [u8], n: usize) {
        crate::arch::f8::e4m3_to_f16(s, d, n)
    }
    #[inline(always)]
    pub fn dequant_e5m2_f16(s: &[u8], d: &mut [u8], n: usize) {
        crate::arch::f8::e5m2_to_f16(s, d, n)
    }
    #[inline(always)]
    pub fn vec_dot_e4m3_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::f8::vec_dot_e4m3_f32(w, x, n)
    }
    #[inline(always)]
    pub fn vec_dot_e5m2_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::f8::vec_dot_e5m2_f32(w, x, n)
    }

    /// One NVFP4 row: packed E2M1 weights, one E4M3 scale per sixteen.
    #[inline(always)]
    pub fn vec_dot_nvfp4_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::portable::f4::vec_dot_nvfp4_f32(w, scales, x, n)
    }

    /// One NVFP4 row against `T` activation rows, widening each block once.
    #[inline(always)]
    pub fn vec_dot_nvfp4_tokens<const T: usize>(
        w: &[u8],
        scales: &[u8],
        xs: &[&[f32]; T],
        n: usize,
        out: &mut [f32; T],
    ) {
        crate::arch::portable::f4::vec_dot_nvfp4_tokens::<T>(w, scales, xs, n, out)
    }

    /// [`vec_dot_nvfp4_f32`] through the NEON tier's explicit `vfmaq_f32`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn vec_dot_nvfp4_neon_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::neon::f4::vec_dot_nvfp4_f32(w, scales, x, n)
    }
    /// NVFP4 weight rows through the AVX2 tier.
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[inline(always)]
    pub fn gemv_nvfp4_avx2_f32(w: &[u8], scales: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::avx2::f4::nvfp4_f32(w, scales, x, out)
    }
    /// Explicit AVX-512 NVFP4 row kernel for native benchmarks.
    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    pub fn gemv_nvfp4_avx512_f32(w: &[u8], scales: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::f4::nvfp4_f32(w, scales, x, out)
    }
    /// NVFP4 rows against activations quantized to U8, through AVX-VNNI.
    #[cfg(all(target_arch = "x86_64", feature = "nvfp4_vnni"))]
    pub mod nvfp4_q8 {
        pub use crate::arch::portable::f4::Quantized;
        pub const fn quantized_len(k: usize) -> (usize, usize) {
            crate::arch::qf4::quantized_len(k)
        }
        pub fn quantize(x: &[f32], biased: &mut [u8], scales: &mut [f32]) {
            crate::arch::qf4::quantize(x, biased, scales)
        }
        pub fn nvfp4_q8_f32(w: &[u8], scales: &[u8], x: &Quantized<'_>, out: &mut [f32]) {
            crate::arch::qf4::nvfp4_q8_f32(w, scales, x, out)
        }
    }
    /// The 256-bit VNNI reference, even when 512-bit VNNI is selected.
    #[cfg(all(target_arch = "x86_64", feature = "avx_vnni"))]
    pub mod nvfp4_q8_avx_vnni {
        pub fn nvfp4_q8_f32(
            w: &[u8],
            scales: &[u8],
            x: &super::nvfp4_q8::Quantized<'_>,
            out: &mut [f32],
        ) {
            crate::arch::avx_vnni::f4::nvfp4_q8_f32(w, scales, x, out)
        }
    }
    /// [`vec_dot_nvfp4_f32`] with the table-free decoder.
    #[inline(always)]
    pub fn vec_dot_nvfp4_bits_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
        crate::arch::portable::f4::vec_dot_nvfp4_bits_f32(w, scales, x, n)
    }
    /// [`vec_dot_nvfp4_tokens`] over either decoder: `BITS` builds the F32
    /// patterns outright instead of looking them up.
    #[inline(always)]
    pub fn nvfp4_tokens<const T: usize, const BITS: bool>(
        w: &[u8],
        scales: &[u8],
        xs: &[&[f32]; T],
        n: usize,
        out: &mut [f32; T],
    ) {
        crate::arch::portable::f4::tokens::<T, BITS>(w, scales, xs, n, out)
    }

    // ── GEMV: weight rows against one activation row ──
    #[inline(always)]
    pub fn gemv_e4m3_f32(w: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::gemv::e4m3_f32(w, x, out)
    }
    /// Explicit 256-bit reference for native AVX-512 E4M3 benchmarks.
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    pub fn gemv_e4m3_avx2_f32(w: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::avx2::gemv::e4m3_f32(w, x, out)
    }
    /// Selected routed expert E4M3 update.
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    pub fn axpy_e4m3_selected_f32(w: &[u8], a: f32, out: &mut [f32]) {
        crate::arch::gemv::axpy_e4m3_f32(w, a, out)
    }
    /// Explicit 256-bit routed expert reference.
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    pub fn axpy_e4m3_avx2_f32(w: &[u8], a: f32, out: &mut [f32]) {
        crate::arch::avx2::gemv::axpy_e4m3_f32(w, a, out)
    }
    #[inline(always)]
    pub fn gemv_bf16_f32(w: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::gemv::bf16_f32(w, x, out)
    }
    /// Explicit 256-bit reference for native AVX-512 BF16 benchmarks.
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    pub fn gemv_bf16_avx2_f32(w: &[u8], x: &[f32], out: &mut [f32]) {
        crate::arch::avx2::gemv::bf16_f32(w, x, out)
    }
    #[inline(always)]
    pub fn vec_dot_bf16_f32(w: &[u8], x: &[f32]) -> f32 {
        crate::arch::portable::gemv::dot_bf16_f32(w, x)
    }

    // ── F8 mixed-precision MAC ──
    #[inline(always)]
    pub fn mad_e4m3_e4m3_f16(a: &[u8], b: &[u8], c: &[u8], d: &mut [f32], n: usize) {
        crate::arch::f8::mad_e4m3_e4m3_f16(a, b, c, d, n)
    }
    #[inline(always)]
    pub fn mad_e4m3_e4m3_bf16(a: &[u8], b: &[u8], c: &[u8], d: &mut [f32], n: usize) {
        crate::arch::f8::mad_e4m3_e4m3_bf16(a, b, c, d, n)
    }
    #[inline(always)]
    pub fn mul_e4m3_f32(a: &[u8], b: &[f32], d: &mut [f32], n: usize) {
        crate::arch::f8::mul_e4m3_f32(a, b, d, n)
    }
    #[inline(always)]
    pub fn dot_e4m3_e4m3(a: &[u8], b: &[u8], n: usize) -> f32 {
        crate::arch::f8::dot_e4m3_e4m3(a, b, n)
    }
    #[inline(always)]
    pub fn gemm_e4m3_e4m3_f16(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        d: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        crate::arch::f8::gemm_e4m3_e4m3_f16(a, b, c, d, m, n, k)
    }
    #[inline(always)]
    pub fn gemm_e4m3_e4m3_bf16(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        d: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        crate::arch::f8::gemm_e4m3_e4m3_bf16(a, b, c, d, m, n, k)
    }
    #[inline(always)]
    pub fn gemm_e4m3_f32(w: &[u8], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::f8::gemm_e4m3_f32(w, x, c, m, n, k)
    }
    #[inline(always)]
    pub fn gemm_e5m2_f32(w: &[u8], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        crate::arch::f8::gemm_e5m2_f32(w, x, c, m, n, k)
    }

    // ── SH → RGB (rendering Kernel A) ──
    #[inline(always)]
    pub fn sh_eval_e4m3(dirs: &[f32], coeffs: &[u8], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_e4m3(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_e5m2(dirs: &[f32], coeffs: &[u8], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_e5m2(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_f32(dirs: &[f32], coeffs: &[f32], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_f32(dirs, coeffs, out, n, deg)
    }
    #[inline(always)]
    pub fn sh_eval_f16(dirs: &[f32], coeffs: &[f16], out: &mut [f32], n: usize, deg: usize) {
        crate::arch::sh::sh_eval_f16(dirs, coeffs, out, n, deg)
    }

    // ── Splat compositing (rendering Kernel B) ──
    /// Pixels per compositor strip (see `arch::splat::PW`).
    pub const SPLAT_PW: usize = crate::arch::splat::PW;
    #[inline(always)]
    pub fn composite_strip_f32(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[f32],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_f32(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    #[inline(always)]
    pub fn composite_strip_f16(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_f16(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    #[inline(always)]
    pub fn composite_strip_e4m3(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[u8],
        ids: &[u32],
        x0: f32,
        y: f32,
        rgb: &mut [f32],
        t: &mut [f32],
    ) -> usize {
        crate::arch::splat::composite_strip_e4m3(center, conic, opacity, color, ids, x0, y, rgb, t)
    }
    /// Pack an f32 to a 2-byte LE f16 (bench setup helper).
    #[inline(always)]
    pub fn pack_f16(val: f32) -> [u8; 2] {
        crate::cast::f32_to_f16(val).to_le_bytes()
    }
    /// Decode one E4M3 byte (bench setup helper).
    #[inline(always)]
    pub fn decode_e4m3(b: u8) -> f32 {
        crate::cast::e4m3_to_f32(b)
    }

    // ── Voxel ray tracing (DDA + SVO) ──
    pub use crate::arch::voxel::SvoNode;
    #[inline(always)]
    pub fn dda_trace_n(
        bits: &[u64],
        dims: [u32; 3],
        origins: &[f32],
        dirs: &[f32],
        n: usize,
        t_max: f32,
        out_t: &mut [f32],
        out_idx: &mut [u32],
    ) {
        crate::arch::voxel::dda_trace_n(bits, dims, origins, dirs, n, t_max, out_t, out_idx)
    }
    #[inline(always)]
    pub fn svo_build(bits: &[u64], levels: u32, nodes: &mut [SvoNode]) -> Option<usize> {
        crate::arch::voxel::svo_build(bits, levels, nodes)
    }
    #[inline(always)]
    pub fn svo_trace_n(
        nodes: &[SvoNode],
        levels: u32,
        origins: &[f32],
        dirs: &[f32],
        n: usize,
        t_max: f32,
        out_t: &mut [f32],
        out_idx: &mut [u32],
    ) {
        crate::arch::voxel::svo_trace_n(nodes, levels, origins, dirs, n, t_max, out_t, out_idx)
    }
    pub const VOXEL_BRICK: u32 = crate::arch::voxel::BRICK;
    #[inline(always)]
    pub fn brick_pack(bits: &[u64], dims: [u32; 3], bricks: &mut [u64]) {
        crate::arch::voxel::brick_pack(bits, dims, bricks)
    }
    #[inline(always)]
    pub fn brick_trace_n(
        bricks: &[u64],
        dims: [u32; 3],
        origins: &[f32],
        dirs: &[f32],
        n: usize,
        t_max: f32,
        out_t: &mut [f32],
        out_idx: &mut [u32],
    ) {
        crate::arch::voxel::brick_trace_n(bricks, dims, origins, dirs, n, t_max, out_t, out_idx)
    }
    #[inline(always)]
    pub fn march_density_f32(
        density: &[f32],
        dims: [u32; 3],
        origin: [f32; 3],
        dir: [f32; 3],
        t_max: f32,
        tau_max: f32,
    ) -> (f32, f32) {
        crate::arch::voxel::march_density_f32(density, dims, origin, dir, t_max, tau_max)
    }
    #[inline(always)]
    pub fn march_density_f16(
        density: &[u8],
        dims: [u32; 3],
        origin: [f32; 3],
        dir: [f32; 3],
        t_max: f32,
        tau_max: f32,
    ) -> (f32, f32) {
        crate::arch::voxel::march_density_f16(density, dims, origin, dir, t_max, tau_max)
    }
    #[inline(always)]
    pub fn march_density_e4m3(
        density: &[u8],
        dims: [u32; 3],
        origin: [f32; 3],
        dir: [f32; 3],
        t_max: f32,
        tau_max: f32,
    ) -> (f32, f32) {
        crate::arch::voxel::march_density_e4m3(density, dims, origin, dir, t_max, tau_max)
    }
    #[inline(always)]
    pub fn march_density_e5m2(
        density: &[u8],
        dims: [u32; 3],
        origin: [f32; 3],
        dir: [f32; 3],
        t_max: f32,
        tau_max: f32,
    ) -> (f32, f32) {
        crate::arch::voxel::march_density_e5m2(density, dims, origin, dir, t_max, tau_max)
    }
}
