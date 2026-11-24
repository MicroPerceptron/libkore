use ax::{CostModel, DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn::lowering::{EpilogueActivation, KernelAccuracy};
use kore_axiom::{
    accelerator::Error,
    xe_graph::{Kernel, Target},
};
use kore_rt::{Heap, heap::StaticMemSource};
use kore_svc_accel::contract::{AccessMode, LaunchShape, Workgroups};
use kore_svc_accel::{client::LaunchBuilder, service::AccelRefusal};

#[repr(align(4096))]
struct Backing([u8; 64 * 1024]);

#[test]
fn semantic_admission_rejects_mutated_metadata_and_preserves_affine_ports() {
    use axnn::lowering::KernelRequest;
    use axnn::{Shape, TensorDesc, TernaryOp};
    let desc = TensorDesc::try_new(Shape::from([513]), axnn::DType::F32).unwrap();
    let request = KernelRequest::ternary(TernaryOp::Fma, &desc, &desc, &desc).unwrap();
    let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
    assert_eq!(kernel.input_bytes(), &[2052; 3]);
    assert_eq!(kernel.parameter_bytes(), block(513u32.to_le_bytes(), 0));
    let bindings = kernel
        .bindings(&[(1, 0), (2, 0), (3, 0)], (4, 0), (5, 0))
        .unwrap();
    assert_eq!(bindings.len(), 5);
    assert_eq!(bindings.iter().nth(3).unwrap().len, 8);
    let mut corrupt = request;
    corrupt.output_dtype = axnn::DType::F16;
    assert!(Kernel::select(&corrupt, KernelAccuracy::Approximate).is_none());
    corrupt = request;
    corrupt.shape = Shape::from([1024]);
    assert!(Kernel::select(&corrupt, KernelAccuracy::Approximate).is_none());
}

#[test]
fn retained_launch_rechecks_actual_import_ids_before_publication() {
    use axnn::lowering::KernelRequest;
    use axnn::{BinaryOp, Shape, TensorDesc};
    let desc = TensorDesc::try_new(Shape::from([257]), axnn::DType::F32).unwrap();
    let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
    let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
    // Offset metadata survives suspension. Bindings may share an
    // allocation only in disjoint regions, reads included: the provider
    // refuses any overlap within a launch.
    let plan = kernel
        .launch_builder(41, 42, [0, 1028, 0, 2056, 0], 100)
        .unwrap();
    let launch = plan.build(&[7, 7, 8, 7, 0]).unwrap();
    assert_eq!(
        (launch.queue, launch.program, launch.timeout_ns),
        (41, 42, 100)
    );
    assert_eq!(launch.launch, kernel.launch());
    assert_eq!(launch.bindings.len(), 4);
    let output = launch.bindings.iter().last().unwrap();
    assert_eq!((output.buffer, output.offset, output.len), (7, 2056, 1028));
    let reread = kernel
        .launch_builder(41, 42, [0, 0, 0, 1028, 0], 100)
        .unwrap();
    assert_eq!(
        reread.build(&[7, 7, 8, 7, 0]),
        Err(AccelRefusal::InvalidArgument)
    );
    // Local geometry was checked against distinct placeholders. A later
    // provider ID must not inherit that assumption or bypass overlap checks.
    let overlap = kernel.launch_builder(41, 42, [0; 5], 100).unwrap();
    assert_eq!(
        overlap.build(&[7, 8, 9, 7, 0]),
        Err(AccelRefusal::InvalidArgument)
    );
    assert_eq!(
        overlap.build(&[7, 8, 0, 10, 0]),
        Err(AccelRefusal::InvalidArgument)
    );
    assert!(kernel.launch_builder(0, 42, [0; 5], 100).is_err());
    assert!(kernel.launch_builder(41, 0, [0; 5], 100).is_err());
    assert!(kernel.launch_builder(41, 42, [0; 5], 0).is_err());
    assert!(kernel.launch_builder(41, 42, [0, 0, 0, 1, 0], 100).is_err());
    assert!(
        kernel
            .launch_builder(41, 42, [0, 0, 0, u64::MAX - 3, 0], 100)
            .is_err()
    );
}

#[test]
fn nvfp4_graph_preserves_scale_geometry_and_packed_storage() {
    let mut backing = Backing([0; 64 * 1024]);
    // SAFETY: the scoped heap and all its allocations drop before backing.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut builder = GraphBuilder::new_in(&heap);
    let a = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[8, 2688], &[])
        .unwrap();
    let b = builder
        .tensor(DType::U8, TensorFlags::INPUT, &[4096, 1344], &[])
        .unwrap();
    let scales = builder
        .tensor(DType::B8, TensorFlags::INPUT, &[4096, 168], &[])
        .unwrap();
    let c = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[8, 4096], &[])
        .unwrap();
    let attributes =
        ax::tensor_ops::encode_matmul_epilogue_attributes(0.375, EpilogueActivation::Silu).unwrap();
    builder
        .node(
            OpCode::MATMUL_NVFP4_TRANSPOSED,
            NodeFlags::default(),
            &[a, b, scales],
            &[c],
            &attributes,
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let node = graph.node(0).unwrap();
    let kernel = Kernel::from_node(&graph, node, KernelAccuracy::Approximate).unwrap();
    assert_eq!(kernel.parameter_bytes(), block(0.375f32.to_le_bytes(), 8));
    assert_eq!(kernel.input_bytes(), &[86016, 5505024, 688128]);
    assert_eq!(kernel.output_bytes(), 131072);
    assert_eq!(
        kernel.launch(),
        LaunchShape::Workgroups(Workgroups::new([256, 1, 1], [32, 1, 1]).unwrap())
    );
    assert_eq!(
        kernel.artifact().bytes(),
        kore_axiom::xe::nvfp4_projection(kore_axiom::xe::Nvfp4Projection::MambaGateSiluPrefill)
            .bytes()
    );
    assert!(Kernel::from_node(&graph, node, KernelAccuracy::Faithful).is_none());
    assert!(Kernel::from_node(&graph, node, KernelAccuracy::Strict).is_none());
    let bindings = kernel
        .bindings(&[(1, 0), (2, 4096), (3, 0)], (4, 0), (5, 0))
        .unwrap();
    assert_eq!(bindings.len(), 5);
    assert!(
        bindings
            .iter()
            .take(4)
            .all(|binding| binding.access == AccessMode::Read)
    );
    assert_eq!(bindings.iter().last().unwrap().access, AccessMode::Write);
    assert_eq!(
        kernel.bindings(&[(1, 0), (2, 4096), (3, 0)], (4, 0), (1, 4096)),
        Err(Error::Binding)
    );
    assert_eq!(
        kernel.bindings(&[(1, 0), (2, 4097), (3, 0)], (4, 0), (5, 0)),
        Err(Error::Binding)
    );
    assert_eq!(
        kernel.bindings(&[(1, 0), (2, 4096), (3, 0)], (4, 0), (5, u64::MAX - 3)),
        Err(Error::Binding)
    );
}

#[test]
fn vector_graph_requires_exact_shapes_and_loaded_package() {
    let mut backing = Backing([0; 64 * 1024]);
    // SAFETY: backing outlives the heap and all graph allocations.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut builder = GraphBuilder::new_in(&heap);
    let a = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[257], &[])
        .unwrap();
    let scalar = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[257], &[])
        .unwrap();
    let broadcast = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[257], &[])
        .unwrap();
    builder
        .node(OpCode::ADD, NodeFlags::default(), &[a, a], &[output], &[])
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[a, scalar],
            &[broadcast],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let node = graph.node(0).unwrap();
    let kernel = Kernel::from_node(&graph, node, KernelAccuracy::Approximate).unwrap();
    assert_eq!(kernel.parameter_bytes(), block(257u32.to_le_bytes(), 0));
    assert_eq!(
        kernel.launch(),
        LaunchShape::Workgroups(Workgroups::new([2, 1, 1], [256, 1, 1]).unwrap())
    );
    // One allocation may hold every binding, but only in disjoint regions:
    // the provider refuses overlap within a launch, reads included.
    assert!(
        kernel
            .bindings(&[(1, 0), (1, 1028)], (2, 0), (1, 2056))
            .is_ok()
    );
    assert_eq!(
        kernel.bindings(&[(1, 0), (1, 0)], (2, 0), (1, 1028)),
        Err(Error::Binding)
    );
    assert_eq!(
        kernel.bindings(&[(1, 0), (1, 0)], (2, 0), (2, 0)),
        Err(Error::Binding)
    );
    assert!(
        Kernel::from_node(&graph, graph.node(1).unwrap(), KernelAccuracy::Approximate).is_none()
    );
    let target = |admit| Target {
        minimum_accuracy: KernelAccuracy::Approximate,
        available: move |_: &Kernel| admit,
        execution_cost: |_: &Kernel| 7,
        transfer_cost: |_: ax::TensorView<'_>, _: Option<u16>, _: u16| 11,
    };
    let unavailable = target(false);
    assert!(!unavailable.supports(0, &graph, node));
    assert_eq!(unavailable.execute_cost(0, &graph, node), u64::MAX);
    let available = target(true);
    assert!(available.supports(0, &graph, node));
    assert_eq!(available.execute_cost(0, &graph, node), 7);
    assert!(!available.supports(1, &graph, node));
    assert_eq!(
        available.transfer_cost(graph.tensor(a).unwrap(), Some(0), 0),
        0
    );
    assert_eq!(
        available.transfer_cost(graph.tensor(a).unwrap(), None, 0),
        11
    );
    assert_eq!(
        available.transfer_cost(graph.tensor(a).unwrap(), Some(1), 0),
        u64::MAX
    );
}

#[test]
fn prepared_plan_admission_preserves_semantics_and_accuracy_policy() {
    use axnn::lowering::KernelRequest;
    use axnn::{BinaryOp, Shape, TensorDesc};
    use axnn_cpu::{CpuCapabilities, plan_cpu_kernel};
    let desc = TensorDesc::try_new(Shape::from([257]), axnn::DType::F32).unwrap();
    let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
    let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
    let admitted = Kernel::from_plan(&plan, KernelAccuracy::Approximate).unwrap();
    assert_eq!(admitted.input_bytes(), &[1028; 2]);
    assert_eq!(admitted.output_bytes(), 1028);
    assert!(Kernel::from_plan(&plan, KernelAccuracy::Faithful).is_none());
    assert!(Kernel::from_plan(&plan, KernelAccuracy::Strict).is_none());
    let mut corrupt = plan;
    corrupt.input_shapes[0] = Some(Shape::from([258]));
    assert!(Kernel::from_plan(&corrupt, KernelAccuracy::Approximate).is_none());
    corrupt = plan;
    corrupt.work_items += 1;
    assert!(Kernel::from_plan(&corrupt, KernelAccuracy::Approximate).is_none());
    corrupt = plan;
    corrupt.output_dtype = axnn::DType::F16;
    assert!(Kernel::from_plan(&corrupt, KernelAccuracy::Approximate).is_none());
    corrupt = plan;
    corrupt.semantic = axnn::lowering::SemanticOp::Unary(axnn::UnaryOp::Silu);
    assert!(Kernel::from_plan(&corrupt, KernelAccuracy::Approximate).is_none());
}

#[test]
fn preloaded_prefill_packages_are_reachable_from_prepared_plans() {
    use axnn::lowering::{EpilogueActivation as E, KernelRequest};
    use axnn::{Shape, TensorDesc};
    use axnn_cpu::{CpuCapabilities, plan_cpu_kernel};
    use kore_axiom::xe::Nvfp4Projection as P;
    // These are captured package contracts, not model node ordinals. Exercise
    // the public pre-load and prepared-plan entry points independently.
    for (kind, m, n, k, activation) in [
        (P::MambaGateSiluPrefill, 8, 4096, 2688, E::Silu),
        (P::MambaXbcPrefill, 8, 6144, 2688, E::None),
        (P::MambaDtPrefill, 8, 64, 2688, E::None),
        (P::HiddenOutPrefill, 8, 2688, 4096, E::None),
        (P::AttentionQPrefill, 8, 4096, 2688, E::None),
        (P::AttentionKvPrefill, 8, 256, 2688, E::None),
        (P::LmHeadDecode, 1, 131072, 2688, E::None),
        (P::LmHeadPrefill, 8, 131072, 2688, E::None),
        (P::LmHeadWide, 16, 131072, 2688, E::None),
    ] {
        let desc = |shape, dtype| TensorDesc::try_new(shape, dtype).unwrap();
        let request = KernelRequest::matmul_nvfp4_transposed(
            &desc(Shape::from([m, k]), axnn::DType::F32),
            &desc(Shape::from([n, k / 2]), axnn::DType::U8),
            &desc(Shape::from([n, k / 16]), axnn::DType::B8),
            &desc(Shape::from([m, n]), axnn::DType::F32),
            0.375,
            activation,
        )
        .unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD)
            .unwrap_or_else(|| panic!("no CPU plan for {kind:?}"));
        let preload = Kernel::nvfp4_projection(kind, 0.375, KernelAccuracy::Approximate).unwrap();
        let selected = Kernel::from_plan(&plan, KernelAccuracy::Approximate).unwrap();
        assert_eq!(
            preload.artifact().bytes(),
            kore_axiom::xe::nvfp4_projection(kind).bytes()
        );
        assert_eq!(selected.artifact().bytes(), preload.artifact().bytes());
        assert_eq!(selected.input_bytes(), preload.input_bytes());
        assert_eq!(selected.output_bytes(), preload.output_bytes());
        assert_eq!(selected.parameter_bytes(), preload.parameter_bytes());
        assert_eq!(selected.launch(), preload.launch());
        // One workgroup per 8x16 tile, as wide as the kind's package.
        assert_eq!(
            preload.launch(),
            LaunchShape::Workgroups(
                Workgroups::new(
                    [(n as u32).div_ceil(16), (m as u32).div_ceil(8), 1],
                    [kind.group_width(), 1, 1]
                )
                .unwrap()
            )
        );
        assert!(Kernel::nvfp4_projection(kind, 1.0, KernelAccuracy::Faithful).is_none());
        assert!(Kernel::nvfp4_projection(kind, 1.0, KernelAccuracy::Strict).is_none());
        assert!(Kernel::nvfp4_projection(kind, f32::NAN, KernelAccuracy::Approximate).is_none());
    }
    // Native availability does not imply a representable graph operation.
    for kind in [
        P::SharedUpSquaredReluPrefill,
        P::SharedDownAccumulatePrefill,
    ] {
        assert!(Kernel::nvfp4_projection(kind, 1.0, KernelAccuracy::Approximate).is_none());
    }
}

/// Prompt widths and ragged verification prefixes use the same checked
/// projection contract. Preserve fixed-width packages when they fit; wide
/// packages carry the actual row count and the same fused accumulator ports.
/// Even at M16, Nemotron's old stream package writes only eight rows per Y
/// workgroup; the dedicated Qwen M16 package has a separate row tile.
#[test]
fn wide_nvfp4_projections_admit_prompt_and_ragged_rows() {
    use ax_exec::FusedRun;
    use axnn::lowering::{EpilogueActivation as E, KernelRequest};
    use axnn::{Shape, TensorDesc};
    use axnn_cpu::{CpuCapabilities, plan_cpu_kernel};
    use kore_axiom::xe::Nvfp4Projection as P;
    use kore_axiom::xe_graph::Port;
    for m in [1usize, 2, 7, 8, 16, 17, 256, 512] {
        for (n, k, run, kinds) in [
            (
                3712,
                2688,
                Some(FusedRun::SquaredRelu),
                [
                    P::SharedUpSquaredReluDecode,
                    P::SharedUpSquaredReluPrefill,
                    P::SharedUpSquaredReluWide,
                ],
            ),
            (
                2688,
                3712,
                Some(FusedRun::Accumulate),
                [
                    P::SharedDownAccumulateDecode,
                    P::SharedDownAccumulatePrefill,
                    P::SharedDownAccumulateWide,
                ],
            ),
            (
                131072,
                2688,
                None,
                [P::LmHeadDecode, P::LmHeadPrefill, P::LmHeadWide],
            ),
        ] {
            let desc = |shape, dtype| TensorDesc::try_new(Shape::from(shape), dtype).unwrap();
            let request = KernelRequest::matmul_nvfp4_transposed(
                &desc([m, k], axnn::DType::F32),
                &desc([n, k / 2], axnn::DType::U8),
                &desc([n, k / 16], axnn::DType::B8),
                &desc([m, n], axnn::DType::F32),
                0.375,
                E::None,
            )
            .unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap();
            let admit = |plan: &_, accuracy| match run {
                Some(run) => Kernel::fused_run(plan, run, accuracy),
                None => Kernel::from_plan(plan, accuracy),
            };
            let kernel = admit(&plan, KernelAccuracy::Approximate).unwrap();
            let kind = kinds[match m {
                1 => 0,
                8 => 1,
                _ => 2,
            }];
            assert_eq!(
                kernel.artifact().bytes(),
                kore_axiom::xe::nvfp4_projection(kind).bytes()
            );
            assert_eq!(
                kernel.parameter_bytes(),
                block(0.375f32.to_le_bytes(), m as u32)
            );
            assert_eq!(
                kernel.input_bytes(),
                &[(m * k * 4) as u64, (n * k / 2) as u64, (n * k / 16) as u64]
            );
            assert_eq!(kernel.output_bytes(), (m * n * 4) as u64);
            assert_eq!(
                kernel.launch(),
                LaunchShape::Workgroups(
                    Workgroups::new(
                        [(n as u32).div_ceil(16), (m as u32).div_ceil(8), 1],
                        [32, 1, 1]
                    )
                    .unwrap()
                )
            );
            if run == Some(FusedRun::Accumulate) {
                assert_eq!(
                    kernel.ports(),
                    &[
                        Port::Input(0),
                        Port::Input(1),
                        Port::Input(2),
                        Port::Parameter,
                        Port::Accumulator,
                        Port::Ones(4)
                    ]
                );
                assert_eq!(kernel.port_bytes(Port::Accumulator), (m * n * 4) as u64);
            } else {
                assert_eq!(
                    kernel.ports(),
                    &[
                        Port::Input(0),
                        Port::Input(1),
                        Port::Input(2),
                        Port::Parameter,
                        Port::Output
                    ]
                );
            }
            assert!(admit(&plan, KernelAccuracy::Faithful).is_none());
            assert!(admit(&plan, KernelAccuracy::Strict).is_none());
            let mut altered = plan;
            altered.work_items += 1;
            assert!(admit(&altered, KernelAccuracy::Approximate).is_none());
        }
    }
}

/// RMS norm and the embedding gather admit only their captured geometry and
/// bind their package slots in order: the hidden norm takes x, weight, output;
/// the Mamba group norm puts a ones gate ahead of its weight (it normalizes
/// y * gate); the gather takes table, ids, output. None has a parameter port,
/// so the per-port single-job path refuses to bind them.
#[test]
fn norms_and_the_embedding_gather_bind_their_package_slots() {
    use axnn::lowering::KernelRequest;
    use axnn::{Shape, TensorDesc};
    use kore_axiom::xe_graph::Port;
    let tensor =
        |dims: &[usize], dtype| TensorDesc::try_new(Shape::try_from(dims).unwrap(), dtype).unwrap();
    let f32 = |dims: &[usize]| tensor(dims, axnn::DType::F32);
    let admit = |request: &KernelRequest| Kernel::select(request, KernelAccuracy::Approximate);
    let grid = |x| LaunchShape::Workgroups(Workgroups::new([x, 1, 1], [256, 1, 1]).unwrap());

    let hidden = KernelRequest::rms_norm_over(&f32(&[8, 2688]), &f32(&[2688]), 1e-5, 1).unwrap();
    let kernel = admit(&hidden).unwrap();
    assert_eq!(
        kernel.ports(),
        &[Port::Input(0), Port::Input(1), Port::Output]
    );
    assert_eq!(kernel.launch(), grid(8));
    assert!(!kernel.legacy_layout());
    assert!(matches!(
        kernel.launch_builder(1, 1, [0; 5], 1),
        Err(Error::Binding)
    ));
    let bindings = kernel
        .port_bindings(|port| match port {
            Port::Input(index) => (10 + u64::from(index), 0),
            _ => (20, 0),
        })
        .unwrap();
    assert_eq!(bindings.len(), 3);
    assert_eq!(bindings.iter().nth(2).unwrap().access, AccessMode::Write);
    assert_eq!(bindings.iter().nth(2).unwrap().len, 8 * 2688 * 4);
    // No binding may alias another.
    assert!(kernel.port_bindings(|_| (7, 0)).is_err());
    let other = KernelRequest::rms_norm_over(&f32(&[8, 2688]), &f32(&[2688]), 1e-6, 1).unwrap();
    assert!(admit(&other).is_none(), "the packages fold in epsilon 1e-5");
    let wide = KernelRequest::rms_norm_over(&f32(&[8, 4096]), &f32(&[4096]), 1e-5, 1).unwrap();
    assert!(admit(&wide).is_none());

    let group = KernelRequest::rms_norm_over(&f32(&[8, 8, 512]), &f32(&[8, 512]), 1e-5, 2).unwrap();
    let kernel = admit(&group).unwrap();
    assert_eq!(
        kernel.ports(),
        &[
            Port::Input(0),
            Port::Ones(8 * 4096 * 4),
            Port::Input(1),
            Port::Output
        ]
    );
    assert_eq!(kernel.launch(), grid(64));
    assert_eq!(kernel.port_bytes(Port::Input(1)), 4096 * 4);

    let ids = tensor(&[8], axnn::DType::I32);
    let gather = KernelRequest::gather(&tensor(&[131_072, 2688], axnn::DType::B16), &ids).unwrap();
    let kernel = admit(&gather).unwrap();
    assert_eq!(
        kernel.ports(),
        &[Port::Input(0), Port::Input(1), Port::Output]
    );
    assert_eq!(kernel.launch(), grid(8));
    assert_eq!(kernel.port_bytes(Port::Output), 8 * 2688 * 4);
    let f32_table = KernelRequest::gather(&f32(&[131_072, 2688]), &ids).unwrap();
    assert!(
        admit(&f32_table).is_none(),
        "only the BF16 table is captured"
    );
}

/// The shared expert's projections admit whole fused runs: the up projection
/// takes its squared-ReLU run (legacy slot layout, the e3 package), the down
/// projection its accumulate run (accumulator read-write in slot 4, the
/// route-scale ones read-write in the final slot 5, the e4 package). Other
/// shapes, epilogues, and accuracy policies refuse.
#[test]
fn shared_expert_projections_admit_their_fused_runs() {
    use ax_exec::FusedRun;
    use axnn::lowering::{EpilogueActivation as E, KernelRequest};
    use axnn::{Shape, TensorDesc};
    use axnn_cpu::{CpuCapabilities, plan_cpu_kernel};
    use kore_axiom::xe::Nvfp4Projection as P;
    use kore_axiom::xe_graph::Port;
    let plan = |n: usize, k: usize, activation| {
        let desc = |shape, dtype| TensorDesc::try_new(shape, dtype).unwrap();
        let request = KernelRequest::matmul_nvfp4_transposed(
            &desc(Shape::from([8, k]), axnn::DType::F32),
            &desc(Shape::from([n, k / 2]), axnn::DType::U8),
            &desc(Shape::from([n, k / 16]), axnn::DType::B8),
            &desc(Shape::from([8, n]), axnn::DType::F32),
            0.5,
            activation,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::PORTABLE_SIMD).unwrap()
    };
    let approximate = KernelAccuracy::Approximate;
    let up = plan(3712, 2688, E::None);
    let kernel = Kernel::fused_run(&up, FusedRun::SquaredRelu, approximate).unwrap();
    assert_eq!(
        kernel.artifact().bytes(),
        kore_axiom::xe::nvfp4_projection(P::SharedUpSquaredReluPrefill).bytes()
    );
    assert!(kernel.legacy_layout());
    assert_eq!(kernel.parameter_bytes(), block(0.5f32.to_le_bytes(), 8));
    assert_eq!(kernel.output_bytes(), 8 * 3712 * 4);
    assert!(Kernel::fused_run(&up, FusedRun::Accumulate, approximate).is_none());
    assert!(Kernel::fused_run(&up, FusedRun::SquaredRelu, KernelAccuracy::Faithful).is_none());
    let silu = plan(3712, 2688, E::Silu);
    assert!(Kernel::fused_run(&silu, FusedRun::SquaredRelu, approximate).is_none());
    assert!(
        Kernel::from_plan(&up, approximate).is_none(),
        "the plain catalog has no 3712-wide projection"
    );

    let down = plan(2688, 3712, E::None);
    let kernel = Kernel::fused_run(&down, FusedRun::Accumulate, approximate).unwrap();
    assert_eq!(
        kernel.artifact().bytes(),
        kore_axiom::xe::nvfp4_projection(P::SharedDownAccumulatePrefill).bytes()
    );
    assert_eq!(
        kernel.ports(),
        &[
            Port::Input(0),
            Port::Input(1),
            Port::Input(2),
            Port::Parameter,
            Port::Accumulator,
            Port::Ones(4)
        ]
    );
    assert!(!kernel.legacy_layout());
    let bindings = kernel
        .port_bindings(|port| match port {
            Port::Accumulator => (8, 0),
            Port::Ones(_) => (8, 1 << 20),
            Port::Parameter => (8, 1 << 19),
            Port::Input(index) => (20 + u64::from(index), 0),
            _ => (0, 0),
        })
        .unwrap();
    let access: Vec<_> = bindings.iter().map(|b| b.access).collect();
    assert_eq!(
        access,
        [
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::ReadWrite,
            AccessMode::ReadWrite
        ]
    );
    // A read may not overlap the accumulator it would see change.
    assert!(
        kernel
            .port_bindings(|port| match port {
                Port::Parameter => (8, 64),
                Port::Ones(_) => (8, 1 << 20),
                Port::Input(index) => (20 + u64::from(index), 0),
                _ => (8, 0),
            })
            .is_err()
    );
    assert!(
        Kernel::fused_run(
            &plan(2688, 1856, E::None),
            FusedRun::Accumulate,
            approximate
        )
        .is_none()
    );
}

#[test]
fn fused_group_rms_uses_the_actual_gate_and_cyclic_weight() {
    use ax_exec::FusedRun;
    use axnn::lowering::{F32Bits, KernelRequest};
    use axnn::{BinaryOp, Shape, TensorDesc};
    use axnn_cpu::{CpuCapabilities, plan_cpu_kernel};
    use kore_axiom::xe_graph::Port;
    let plan = |rows, width, op| {
        let desc = TensorDesc::try_new(Shape::from([rows, width]), axnn::DType::F32).unwrap();
        plan_cpu_kernel(
            &KernelRequest::binary(op, &desc, &desc).unwrap(),
            CpuCapabilities::PORTABLE_SIMD,
        )
        .unwrap()
    };
    let run = FusedRun::RmsNorm {
        width: 512,
        epsilon: F32Bits::from_f32(1e-5),
    };
    for rows in [1, 8, 64] {
        let kernel = Kernel::fused_run(
            &plan(rows, 4096, BinaryOp::Mul),
            run,
            KernelAccuracy::Approximate,
        )
        .unwrap();
        assert_eq!(
            kernel.input_bytes(),
            &[rows as u64 * 4096 * 4, rows as u64 * 4096 * 4, 4096 * 4]
        );
        assert_eq!(
            kernel.ports(),
            &[Port::Input(0), Port::Input(1), Port::Input(2), Port::Output]
        );
        assert_eq!(
            kernel.launch(),
            LaunchShape::Workgroups(Workgroups::new([rows as u32 * 8, 1, 1], [256, 1, 1]).unwrap())
        );
        let bindings = kernel
            .port_bindings(|port| match port {
                Port::Input(n) => (u64::from(n) + 1, 0),
                _ => (4, 0),
            })
            .unwrap();
        assert_eq!(bindings.iter().nth(2).unwrap().access, AccessMode::Read);
        assert_eq!(bindings.iter().nth(3).unwrap().len, rows as u64 * 4096 * 4);
    }
    assert!(
        Kernel::fused_run(&plan(8, 4096, BinaryOp::Mul), run, KernelAccuracy::Faithful).is_none()
    );
    assert!(
        Kernel::fused_run(
            &plan(8, 2688, BinaryOp::Mul),
            run,
            KernelAccuracy::Approximate
        )
        .is_none()
    );
    // ADD requires the separate residual-output package; it must not silently
    // dispatch the multiply specialization.
    assert!(
        Kernel::fused_run(
            &plan(8, 4096, BinaryOp::Add),
            run,
            KernelAccuracy::Approximate
        )
        .is_none()
    );
    let wrong_epsilon = FusedRun::RmsNorm {
        width: 512,
        epsilon: F32Bits::from_f32(1e-6),
    };
    assert!(
        Kernel::fused_run(
            &plan(8, 4096, BinaryOp::Mul),
            wrong_epsilon,
            KernelAccuracy::Approximate
        )
        .is_none()
    );
}

/// A parameter block: word 0, then the rows a runtime-M projection reads.
fn block(word: [u8; 4], rows: u32) -> [u8; 8] {
    let mut block = [0; 8];
    block[..4].copy_from_slice(&word);
    block[4..].copy_from_slice(&rows.to_le_bytes());
    block
}
