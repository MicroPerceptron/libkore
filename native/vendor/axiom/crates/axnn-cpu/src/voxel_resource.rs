//! Checked borrowed occupancy resources for planned voxel kernels.

use axnn::DType;
use axnn::lowering::{KernelEffects, KernelParams, KernelResult, SemanticOp};

use crate::CpuError;
use crate::arch::portable::voxel;
use crate::planner::{CpuImplementation, CpuKernelPlan, typed_input_matches, typed_output_matches};

/// An immutable, caller-owned brick grid validated for signed voxel IDs.
///
/// Fine dimensions are positive multiples of four and their product fits in
/// `i32`. One `u64` stores each 4³ brick, with x-major brick ordering and child
/// bit `x + 4*y + 16*z`. The slice contains exactly the required brick words.
/// Validation runs in release builds and performs no allocation or copy.
///
/// This view is supplied at invocation time; neither it nor its pointer is
/// retained in a kernel plan. Construct a fresh borrow after occupancy edits.
#[derive(Clone, Copy, Debug)]
pub struct BrickGridRef<'a> {
    bricks: &'a [u64],
    dims: [u32; 3],
}

impl<'a> BrickGridRef<'a> {
    pub fn new(bricks: &'a [u64], dims: [u32; 3]) -> Result<Self, CpuError> {
        let cells = dims
            .iter()
            .try_fold(1_u32, |count, &extent| count.checked_mul(extent));
        let cells = cells
            .filter(|&count| count <= i32::MAX as u32)
            .filter(|_| {
                dims.iter()
                    .all(|&extent| extent > 0 && extent.is_multiple_of(4))
            })
            .ok_or(CpuError::ShapeMismatch {
                op: "borrowed voxel brick grid",
            })?;
        let expected = cells as usize / 64;
        if bricks.len() != expected {
            return Err(CpuError::SizeMismatch {
                expected: expected * core::mem::size_of::<u64>(),
                actual: core::mem::size_of_val(bricks),
            });
        }
        Ok(Self { bricks, dims })
    }

    pub const fn dims(self) -> [u32; 3] {
        self.dims
    }

    pub const fn bricks(self) -> &'a [u64] {
        self.bricks
    }
}

/// Invoke a scalar or four-lane packet brick query over xyz-interleaved rays.
///
/// The plan consumes F32 `H × W × 3` origins/directions and one F32 limit;
/// results are F32 `H × W` distances followed by exact I32 cell IDs. Misses
/// are `(+infinity, -1)`. Directions are not normalized. Finite valid rays
/// preserve the existing scalar brick tracer's hit bits and tie ordering.
/// Nonfinite origins/directions and negative or NaN limits produce misses;
/// positive infinite limits and either sign of zero remain supported.
///
/// Inputs/outputs must have their exact planned lengths. Caller-owned U32
/// scratch must cover `H*W` IDs; any extra scratch tail remains untouched.
/// Shape/resource errors are returned before outputs or scratch are changed.
/// No allocation, occupancy copy, or resource pointer retention occurs.
pub fn invoke_voxel_brick_trace(
    plan: &CpuKernelPlan,
    grid: BrickGridRef<'_>,
    origins: &[f32],
    dirs: &[f32],
    limit: &[f32],
    out_t: &mut [f32],
    out_id: &mut [i32],
    scratch_u32: &mut [u32],
) -> Result<(), CpuError> {
    let KernelParams::VoxelBrickTrace(params) = plan.params() else {
        return Err(CpuError::UnsupportedOp {
            op: "planned voxel brick trace",
            dtype: plan.output_dtype,
        });
    };
    if plan.semantic != SemanticOp::VoxelBrickTrace
        || !params.is_valid()
        || params.dims != grid.dims
        || plan.effects() != KernelEffects::READS_RESOURCE
        || !matches!(plan.result(), KernelResult::TensorPair(_))
        || plan.result_count() != 2
        || plan.output_dtype != DType::F32
        || plan.shape.rank() != 2
        || !typed_output_matches(plan, 0, DType::F32, plan.shape, out_t.len())
        || !typed_output_matches(plan, 1, DType::I32, plan.shape, out_id.len())
        || !typed_input_matches(plan, 0, DType::F32, origins.len())
        || !typed_input_matches(plan, 1, DType::F32, dirs.len())
        || !typed_input_matches(plan, 2, DType::F32, limit.len())
        || limit.len() != 1
        || scratch_u32.len() < out_t.len()
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned voxel brick trace",
        });
    }
    let packed = axnn::shape![plan.shape.dim(0), plan.shape.dim(1), 3];
    if plan.input_shapes[0] != Some(packed) || plan.input_shapes[1] != Some(packed) {
        return Err(CpuError::ShapeMismatch {
            op: "planned voxel ray packing",
        });
    }
    if !matches!(
        plan.implementation,
        CpuImplementation::Scalar | CpuImplementation::PortableSimd
    ) {
        return Err(CpuError::UnsupportedOp {
            op: "planned voxel brick implementation",
            dtype: DType::F32,
        });
    }

    let n = out_t.len();
    let scratch = &mut scratch_u32[..n];
    let t_max = limit[0];
    if t_max.is_nan() || t_max < 0.0 {
        out_t.fill(voxel::MISS.0);
        scratch.fill(voxel::MISS.1);
    } else {
        match plan.implementation {
            CpuImplementation::Scalar => {
                for (index, (&origin, &dir)) in origins
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(dirs.as_chunks::<3>().0)
                    .enumerate()
                {
                    let hit = if origin.iter().chain(&dir).all(|value| value.is_finite()) {
                        voxel::brick_trace(grid.bricks, grid.dims, origin, dir, t_max)
                    } else {
                        voxel::MISS
                    };
                    (out_t[index], scratch[index]) = hit;
                }
            }
            CpuImplementation::Qdot => unreachable!("implementation checked before writing"),
            CpuImplementation::PortableSimd => voxel::brick_trace_packet_n(
                grid.bricks,
                grid.dims,
                origins,
                dirs,
                n,
                t_max,
                out_t,
                scratch,
            ),
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            CpuImplementation::Avx2 => unreachable!("implementation checked before writing"),
        }
    }
    for (output, &id) in out_id.iter_mut().zip(scratch.iter()) {
        // The validated fine-cell product bounds every hit by i32::MAX.
        // Integer conversion preserves IDs above F32's exact-integer range.
        *output = if id == voxel::MISS.1 {
            -1
        } else {
            debug_assert!(id <= i32::MAX as u32);
            id as i32
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel, select_cpu_kernel};
    use alloc::vec;
    use axnn::lowering::{
        KernelAccuracy, KernelPolicy, KernelRequest, ResourceKey, VoxelBrickTraceParams,
    };
    use axnn::{Shape, TensorDesc, shape};

    fn request(dims: [u32; 3], width: usize) -> KernelRequest {
        let rays = TensorDesc::new(shape![1, width, 3], DType::F32);
        let params = VoxelBrickTraceParams::new(
            ResourceKey {
                group: 2,
                binding: 7,
            },
            dims,
        )
        .unwrap();
        KernelRequest::voxel_brick_trace(
            &rays,
            &rays,
            &TensorDesc::new(Shape::SCALAR, DType::F32),
            params,
        )
        .unwrap()
    }

    fn set_cell(bricks: &mut [u64], dims: [u32; 3], cell: [u32; 3]) {
        let [x, y, z] = cell;
        let index = x / 4 + (dims[0] / 4) * (y / 4 + (dims[1] / 4) * (z / 4));
        let bit = x % 4 + 4 * (y % 4) + 16 * (z % 4);
        bricks[index as usize] |= 1 << bit;
    }

    #[test]
    fn voxel_catalog_selects_typed_results_and_accounts_for_required_scratch() {
        let request = request([12, 8, 4], 7);
        assert_eq!(request.result_count(), 2);
        for (capabilities, implementation) in [
            (CpuCapabilities::SCALAR, CpuImplementation::Scalar),
            (
                CpuCapabilities::PORTABLE_SIMD,
                CpuImplementation::PortableSimd,
            ),
        ] {
            let mut policy = KernelPolicy::new(capabilities.mask())
                .with_minimum_accuracy(KernelAccuracy::Strict);
            policy.max_temporary_bytes = 7 * core::mem::size_of::<u32>() - 1;
            assert!(select_cpu_kernel(&request, policy).is_none());
            policy.max_temporary_bytes += 1;
            let plan = select_cpu_kernel(&request, policy).unwrap();
            assert_eq!(plan.implementation, implementation);
            assert_eq!(plan.temporary_bytes, 28);
            assert_eq!(plan.shape, shape![1, 7]);
            assert_eq!(plan.result_count(), 2);
            assert_eq!(plan.result_tensor(0).unwrap().dtype, DType::F32);
            assert_eq!(plan.result_tensor(1).unwrap().dtype, DType::I32);
            assert_eq!(plan.result_tensor(1).unwrap().shape, plan.shape);
            assert!(plan.result_tensor(2).is_none());
            assert_eq!(plan.effects(), KernelEffects::READS_RESOURCE);
            assert_eq!(plan.params(), request.params());
        }
        let params = match request.params() {
            KernelParams::VoxelBrickTrace(params) => params,
            _ => unreachable!(),
        };
        let rays = TensorDesc::new(shape![1, 7, 3], DType::F32);
        let limit = TensorDesc::new(shape![1, 1, 1, 1], DType::F32);
        let rank_four_limit =
            KernelRequest::voxel_brick_trace(&rays, &rays, &limit, params).unwrap();
        assert!(plan_cpu_kernel(&rank_four_limit, CpuCapabilities::PORTABLE_SIMD).is_some());
        let strided = TensorDesc {
            layout: axnn::Layout::from_raw(&[112, 16, 4], 0),
            ..rays
        };
        let strided_request =
            KernelRequest::voxel_brick_trace(&strided, &rays, &limit, params).unwrap();
        assert!(plan_cpu_kernel(&strided_request, CpuCapabilities::compiled()).is_none());
    }

    #[test]
    fn borrowed_grid_validates_dimensions_and_exact_storage_without_copying() {
        let bricks = [0_u64; 6];
        let grid = BrickGridRef::new(&bricks, [12, 8, 4]).unwrap();
        assert_eq!(grid.dims(), [12, 8, 4]);
        assert_eq!(grid.bricks().as_ptr(), bricks.as_ptr());
        assert_eq!(grid.bricks().len(), bricks.len());
        let copied = grid;
        assert_eq!(copied.bricks().as_ptr(), grid.bricks().as_ptr());
        for dims in [
            [0, 4, 4],
            [4, 0, 4],
            [4, 4, 0],
            [6, 4, 4],
            [4, 4, 134_217_728],
            [u32::MAX; 3],
            [65_536, 65_536, 4],
        ] {
            assert!(matches!(
                BrickGridRef::new(&bricks, dims),
                Err(CpuError::ShapeMismatch { .. })
            ));
        }
        assert_eq!(
            BrickGridRef::new(&bricks[..5], [12, 8, 4]).unwrap_err(),
            CpuError::SizeMismatch {
                expected: 48,
                actual: 40
            },
        );
        assert_eq!(
            BrickGridRef::new(&bricks, [4, 4, 4]).unwrap_err(),
            CpuError::SizeMismatch {
                expected: 8,
                actual: 48
            },
        );
    }

    #[test]
    fn voxel_invokers_match_scalar_bits_for_tails_clips_and_invalid_rays() {
        const RAYS: [([f32; 3], [f32; 3]); 19] = [
            ([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]),
            ([-1.0, 0.5, 0.5], [2.0, 0.0, 0.0]),
            ([12.5, 7.5, 3.5], [-1.0, 0.0, 0.0]),
            ([4.0, 4.0, 1.0], [1.0, 1.0, 0.0]),
            ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, -0.0, 0.0]),
            ([1.5, 6.5, 2.5], [0.0, 0.0, 0.0]),
            ([-2.0, 10.0, 1.0], [-1.0, 0.0, 0.0]),
            ([8.0, 2.5, 1.5], [0.5, -0.0, 0.0]),
            ([f32::NAN, 0.5, 0.5], [1.0, 0.0, 0.0]),
            ([f32::INFINITY, 0.5, 0.5], [1.0, 0.0, 0.0]),
            ([0.5, f32::NEG_INFINITY, 0.5], [1.0, 0.0, 0.0]),
            ([0.5, 0.5, 0.5], [f32::NAN, 0.0, 0.0]),
            ([0.5, 0.5, 0.5], [1.0, f32::INFINITY, 0.0]),
            ([f32::MAX, -f32::MAX, 0.5], [-1.0, 1.0, 0.0]),
            ([0.5, 0.5, 0.5], [f32::from_bits(1), -0.0, 0.0]),
            ([3.0, 0.5, 0.5], [-1.0, -0.0, 0.0]),
            ([12.0, 8.0, 4.0], [-1.0, -1.0, -1.0]),
            ([-4.0, 4.0, 1.0], [4.0, 0.0, 0.0]),
        ];
        let dims = [12, 8, 4];
        let mut bricks = [0_u64; 6];
        for cell in [[0, 0, 0], [3, 0, 0], [4, 4, 1], [9, 2, 1], [11, 7, 3]] {
            set_cell(&mut bricks, dims, cell);
        }
        let mut origins = [0.0; 57];
        let mut dirs = [0.0; 57];
        for (index, (origin, dir)) in RAYS.iter().enumerate() {
            origins[index * 3..index * 3 + 3].copy_from_slice(origin);
            dirs[index * 3..index * 3 + 3].copy_from_slice(dir);
        }
        for n in [0, 1, 3, 5, 7, 19] {
            let request = request(dims, n);
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
                let plan = plan_cpu_kernel(&request, capabilities).unwrap();
                if n >= 5 && capabilities == CpuCapabilities::PORTABLE_SIMD {
                    assert_eq!(plan.implementation, CpuImplementation::PortableSimd);
                }
                for limit in [-1.0, f32::NAN, -0.0, 0.0, 0.5, 1.0, 10.0, f32::INFINITY] {
                    let mut out_t = [-99.0; 21];
                    let mut out_id = [-42; 21];
                    let mut scratch = [0x5555_5555; 21];
                    invoke_voxel_brick_trace(
                        &plan,
                        BrickGridRef::new(&bricks, dims).unwrap(),
                        &origins[..n * 3],
                        &dirs[..n * 3],
                        &[limit],
                        &mut out_t[..n],
                        &mut out_id[..n],
                        &mut scratch,
                    )
                    .unwrap();
                    for index in 0..n {
                        let (origin, dir) = RAYS[index];
                        let expected = if !limit.is_nan()
                            && limit >= 0.0
                            && origin.iter().chain(&dir).all(|value| value.is_finite())
                        {
                            voxel::brick_trace(&bricks, dims, origin, dir, limit)
                        } else {
                            voxel::MISS
                        };
                        assert_eq!(
                            out_t[index].to_bits(),
                            expected.0.to_bits(),
                            "n={n} ray={index} limit={limit}"
                        );
                        assert_eq!(out_id[index], expected.1 as i32);
                        assert_eq!(scratch[index], expected.1);
                    }
                    assert!(out_t[n..].iter().all(|&value| value == -99.0));
                    assert!(out_id[n..].iter().all(|&value| value == -42));
                    assert!(scratch[n..].iter().all(|&value| value == 0x5555_5555));
                }
            }
        }
    }

    #[test]
    fn voxel_invocation_rejects_mismatched_resources_and_lengths_before_writing() {
        let dims = [4; 3];
        let plan = plan_cpu_kernel(&request(dims, 5), CpuCapabilities::PORTABLE_SIMD).unwrap();
        let bricks = [u64::MAX];
        let grid = BrickGridRef::new(&bricks, dims).unwrap();
        let origins = [0.5; 15];
        let dirs = [1.0; 15];
        let limits = [10.0, 20.0];
        for [o, d, l, t, i, s] in [
            [14, 15, 1, 5, 5, 5],
            [15, 14, 1, 5, 5, 5],
            [15, 15, 0, 5, 5, 5],
            [15, 15, 2, 5, 5, 5],
            [15, 15, 1, 4, 5, 5],
            [15, 15, 1, 5, 4, 5],
            [15, 15, 1, 5, 5, 4],
        ] {
            let mut out_t = [42.0; 5];
            let mut out_id = [42; 5];
            let mut scratch = [42; 5];
            assert!(
                invoke_voxel_brick_trace(
                    &plan,
                    grid,
                    &origins[..o],
                    &dirs[..d],
                    &limits[..l],
                    &mut out_t[..t],
                    &mut out_id[..i],
                    &mut scratch[..s],
                )
                .is_err()
            );
            assert_eq!(out_t, [42.0; 5]);
            assert_eq!(out_id, [42; 5]);
            assert_eq!(scratch, [42; 5]);
        }
        let mut out_t = [42.0; 5];
        let mut out_id = [42; 5];
        let mut scratch = [42; 5];
        let other_bricks = [u64::MAX; 2];
        let wrong_grid = BrickGridRef::new(&other_bricks, [8, 4, 4]).unwrap();
        assert!(
            invoke_voxel_brick_trace(
                &plan,
                wrong_grid,
                &origins,
                &dirs,
                &limits[..1],
                &mut out_t,
                &mut out_id,
                &mut scratch,
            )
            .is_err()
        );
        let mut bad_plan = plan;
        bad_plan.shape = shape![5, 1];
        assert!(
            invoke_voxel_brick_trace(
                &bad_plan,
                grid,
                &origins,
                &dirs,
                &limits[..1],
                &mut out_t,
                &mut out_id,
                &mut scratch,
            )
            .is_err()
        );
        assert_eq!(out_t, [42.0; 5]);
        assert_eq!(out_id, [42; 5]);
        assert_eq!(scratch, [42; 5]);
    }

    #[test]
    fn repeated_queries_observe_edited_borrowed_bricks_with_reused_outputs() {
        let dims = [4; 3];
        let origins = [0.5; 15];
        let mut dirs = [0.0; 15];
        for ray in dirs.as_chunks_mut::<3>().0 {
            ray[0] = 1.0;
        }
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            let plan = plan_cpu_kernel(&request(dims, 5), capabilities).unwrap();
            let mut bricks = [0];
            let mut out_t = [42.0; 5];
            let mut out_id = [42; 5];
            let mut scratch = [42; 5];
            for (word, expected_t, expected_id) in
                [(1, 0.0, 0), (2, 0.5, 1), (0, f32::INFINITY, -1)]
            {
                bricks[0] = word;
                invoke_voxel_brick_trace(
                    &plan,
                    BrickGridRef::new(&bricks, dims).unwrap(),
                    &origins,
                    &dirs,
                    &[10.0],
                    &mut out_t,
                    &mut out_id,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(out_t.map(f32::to_bits), [expected_t.to_bits(); 5]);
                assert_eq!(out_id, [expected_id; 5]);
            }
        }
    }

    #[test]
    fn voxel_query_preserves_ids_above_f32_exact_integer_range() {
        let dims = [4, 4, 1_048_580];
        let mut bricks = vec![0; dims[2] as usize / 4];
        set_cell(&mut bricks, dims, [1, 0, 1_048_576]);
        set_cell(&mut bricks, dims, [2, 0, 1_048_576]);
        let mut origins = [0.0; 15];
        let mut dirs = [0.0; 15];
        for index in 0..5 {
            origins[index * 3..index * 3 + 3].copy_from_slice(&[
                1.5 + (index % 2) as f32,
                0.5,
                1_048_576.5,
            ]);
            dirs[index * 3 + 2] = 1.0;
        }
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            let plan = plan_cpu_kernel(&request(dims, 5), capabilities).unwrap();
            let mut out_t = [42.0; 5];
            let mut out_id = [42; 5];
            let mut scratch = [42; 5];
            invoke_voxel_brick_trace(
                &plan,
                BrickGridRef::new(&bricks, dims).unwrap(),
                &origins,
                &dirs,
                &[0.0],
                &mut out_t,
                &mut out_id,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(out_t, [0.0; 5]);
            assert_eq!(
                out_id,
                [16_777_217, 16_777_218, 16_777_217, 16_777_218, 16_777_217]
            );
        }
    }
}
