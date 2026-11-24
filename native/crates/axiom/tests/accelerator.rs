#![feature(allocator_api)]

use ax::ScheduleScratch;
use axnn::lowering::KernelAccuracy;
use kore_axiom::accelerator::{AdmittedGraph, ArtifactBuild, ArtifactPort, Error, Policy};
use kore_svc_accel::{contract::AccessMode, service::AccelRequest};

#[test]
fn fused_sink_o_profile_admits_six_weight_free_ports() {
    let authored = ax::sink_gqa_o_residual::graph_in(std::alloc::Global).unwrap();
    let graph = authored.as_bytes();
    let artifact = b"test-only fused sink O identity";
    let inputs = [0, 1, 2, 3, 4].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let outputs = [ArtifactPort {
        tensor: ax::TensorId(5),
        slot: 5,
    }];
    let build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_SINK_GQA_O_RESIDUAL_BLOCK,
        graph_blake3: *blake3::hash(graph).as_bytes(),
        artifact_blake3: *blake3::hash(artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &inputs,
        outputs: &outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 16 << 20,
        max_resident_bytes: 1 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 6],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    for (index, bytes) in [131_072, 2_129_920, 10_888, 11_010_048, 86_016]
        .into_iter()
        .enumerate()
    {
        assert_eq!(admitted.input_bytes(index), Ok(bytes));
    }
    assert_eq!(admitted.output_bytes(0), Ok(86_016));
    let bindings = admitted
        .bindings_with_offsets(&[(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)])
        .unwrap();
    assert_eq!(bindings.iter().count(), 6);
    assert_eq!(bindings.iter().nth(1).unwrap().len, 2_129_920);
    assert_eq!(bindings.iter().nth(3).unwrap().access, AccessMode::Read);
    assert_eq!(bindings.iter().nth(5).unwrap().access, AccessMode::Write);
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch,
        )
        .err(),
        Some(Error::Shape)
    );
}

#[test]
fn context_kv_admits_exact_six_ports_and_disjoint_output_views() {
    mod kv_bundle {
        include!("../../../drivers/ivpu/model/dspark_context_kv_40xx/native.rs");
    }
    let authored = ax::context_kv::graph_in(std::alloc::Global).unwrap();
    let graph = include_bytes!("../../../drivers/ivpu/model/dspark_context_kv_40xx/graph.ax");
    let artifact = include_bytes!("../../../drivers/ivpu/model/dspark_context_kv_40xx/model.blob");
    assert_eq!(authored.as_bytes(), graph);
    let build = kv_bundle::BUILD;
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 32 << 20,
        max_resident_bytes: 1 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 6],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(admitted.input_bytes(0), Ok(172_032));
    assert_eq!(admitted.input_bytes(1), Ok(5_505_024));
    assert_eq!(admitted.input_bytes(2), Ok(11_264));
    assert_eq!(admitted.input_bytes(3), Ok(4_096));
    assert_eq!(admitted.output_bytes(0), Ok(8_192));
    assert_eq!(admitted.output_bytes(1), Ok(8_192));
    let bindings = admitted
        .bindings_with_offsets(&[(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (14, 8_192)])
        .unwrap();
    assert_eq!(bindings.iter().count(), 6);
    assert_eq!(bindings.iter().nth(4).unwrap().buffer, 14);
    assert_eq!(bindings.iter().nth(5).unwrap().offset, 8_192);
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch,
        )
        .err(),
        Some(Error::UnsupportedOperation(ax::OpCode::CONTEXT_KV_BLOCK.0))
    );
}

#[test]
fn context_partial_admits_exact_six_dma_ports() {
    let authored = ax::context_partial_i8::graph_in(std::alloc::Global).unwrap();
    let graph = authored.as_bytes();
    let artifact = b"test-only context partial artifact identity";
    let inputs = [0, 1, 2, 3, 4].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let outputs = [ArtifactPort {
        tensor: ax::TensorId(5),
        slot: 5,
    }];
    let build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_CONTEXT_PARTIAL_I8_BLOCK,
        graph_blake3: *blake3::hash(graph).as_bytes(),
        artifact_blake3: *blake3::hash(artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &inputs,
        outputs: &outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 32 << 20,
        max_resident_bytes: 1 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 6],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(admitted.input_bytes(0), Ok(86_016));
    assert_eq!(admitted.input_bytes(1), Ok(86_016));
    assert_eq!(admitted.input_bytes(2), Ok(86_016));
    assert_eq!(admitted.input_bytes(3), Ok(21_676_032));
    assert_eq!(admitted.input_bytes(4), Ok(10_752));
    assert_eq!(admitted.output_bytes(0), Ok(86_016));
    assert_eq!(
        admitted
            .bindings(&[10, 11, 12, 13, 14, 15])
            .unwrap()
            .iter()
            .count(),
        6
    );
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch
        )
        .err(),
        Some(Error::Shape)
    );
}

#[test]
fn typed_i8_weight_projection_admits_only_its_own_profile() {
    let mut builder = ax::GraphBuilder::new_in(std::alloc::Global);
    let x = builder
        .tensor(ax::DType::F32, ax::TensorFlags::INPUT, &[8, 128], &[])
        .unwrap();
    let w = builder
        .tensor(ax::DType::I8, ax::TensorFlags::INPUT, &[96, 128], &[])
        .unwrap();
    let s = builder
        .tensor(ax::DType::F32, ax::TensorFlags::INPUT, &[1, 96], &[])
        .unwrap();
    let y = builder
        .tensor(ax::DType::F32, ax::TensorFlags::OUTPUT, &[8, 96], &[])
        .unwrap();
    builder
        .node(
            ax::OpCode::MATMUL_I8_PER_CHANNEL,
            ax::NodeFlags::default(),
            &[x, w, s],
            &[y],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let graph = graph.as_bytes();
    let artifact = b"test-only i8 artifact identity";
    let inputs = [x, w, s].map(|tensor| ArtifactPort {
        slot: tensor.0,
        tensor,
    });
    let outputs = [ArtifactPort { tensor: y, slot: 3 }];
    let build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_I8_PER_CHANNEL,
        graph_blake3: *blake3::hash(graph).as_bytes(),
        artifact_blake3: *blake3::hash(artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &inputs,
        outputs: &outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 1 << 20,
        max_resident_bytes: 1 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 4],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(admitted.input_bytes(0), Ok(8 * 128 * 4));
    assert_eq!(admitted.input_bytes(1), Ok(96 * 128));
    assert_eq!(admitted.input_bytes(2), Ok(96 * 4));
    assert_eq!(admitted.output_bytes(0), Ok(8 * 96 * 4));
    assert_eq!(
        admitted.check_input_f32(1, &[0.; 96 * 128 / 4]),
        Err(Error::Binding)
    );
    assert_eq!(
        admitted
            .bindings(&[10, 11, 12, 13])
            .unwrap()
            .iter()
            .nth(1)
            .unwrap()
            .len,
        96 * 128
    );
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch
        )
        .err(),
        Some(Error::Shape)
    );
}

#[test]
fn typed_sink_gqa_admits_six_ports_and_checks_i32_control_extent() {
    let authored = ax::sink_gqa::graph_in(std::alloc::Global).unwrap();
    let graph = authored.as_bytes();
    let artifact = b"test-only artifact identity";
    let inputs = [0, 1, 2, 3, 4].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let outputs = [ArtifactPort {
        tensor: ax::TensorId(5),
        slot: 5,
    }];
    let build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_SINK_GQA_BLOCK,
        graph_blake3: *blake3::hash(graph).as_bytes(),
        artifact_blake3: *blake3::hash(artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &inputs,
        outputs: &outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 2 << 20,
        max_resident_bytes: 2 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 6],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(admitted.input_bytes(1), Ok(2 << 20));
    assert_eq!(admitted.input_bytes(4), Ok(8));
    assert_eq!(admitted.output_bytes(0), Ok(128 << 10));
    assert_eq!(admitted.check_input_f32(4, &[0., 0.]), Err(Error::Binding));
    let bindings = admitted.bindings(&[10, 11, 12, 13, 14, 15]).unwrap();
    assert_eq!(bindings.iter().count(), 6);
    assert_eq!(bindings.iter().nth(4).unwrap().len, 8);
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch
        )
        .err(),
        Some(Error::Shape)
    );
}

#[test]
fn typed_qkv_prep_admits_two_direct_bind_outputs() {
    let authored = ax::qkv_prep::graph_in(std::alloc::Global).unwrap();
    let graph = authored.as_bytes();
    let artifact = b"test-only qkv prep artifact identity";
    let inputs = [0, 1, 2, 3].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let outputs = [4, 5].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_QKV_PREP_BLOCK,
        graph_blake3: *blake3::hash(graph).as_bytes(),
        artifact_blake3: *blake3::hash(artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &inputs,
        outputs: &outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 64 << 20,
        max_resident_bytes: 1 << 20,
    };
    let mut scratch = ScheduleScratch {
        adjacency: &mut [0; 1],
        producers: &mut [0; 6],
        indegree: &mut [0; 1],
        queue: &mut [0; 1],
        order: &mut [0; 1],
    };
    let admitted = AdmittedGraph::admit(
        graph,
        artifact,
        &build,
        policy(),
        artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(admitted.input_bytes(1), Ok(2688 * 4608 * 4));
    assert_eq!(admitted.input_bytes(2), Ok(2944 * 4));
    assert_eq!(admitted.output_bytes(0), Ok(8 * 32 * 128 * 4));
    assert_eq!(admitted.output_bytes(1), Ok(2 * 8 * 2 * 128 * 4));
    // The same already-imported allocation can hold disjoint outputs that
    // the attention graph reads from these exact offsets. No host repack.
    let bindings = admitted
        .bindings_with_offsets(&[
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 4096),
            (5, 4096 + 8 * 32 * 128 * 4),
        ])
        .unwrap();
    assert_eq!(bindings.iter().nth(4).unwrap().offset, 4096);
    assert_eq!(
        bindings.iter().nth(5).unwrap().offset,
        4096 + 8 * 32 * 128 * 4
    );
    let sink_graph = ax::sink_gqa::graph_in(std::alloc::Global).unwrap();
    let sink_artifact = b"test-only sink artifact identity";
    let sink_inputs = [0, 1, 2, 3, 4].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let sink_outputs = [ArtifactPort {
        tensor: ax::TensorId(5),
        slot: 5,
    }];
    let sink_build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_SINK_GQA_BLOCK,
        graph_blake3: *blake3::hash(sink_graph.as_bytes()).as_bytes(),
        artifact_blake3: *blake3::hash(sink_artifact).as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &sink_inputs,
        outputs: &sink_outputs,
    };
    let sink = AdmittedGraph::admit(
        sink_graph.as_bytes(),
        sink_artifact,
        &sink_build,
        policy(),
        sink_artifact.len() as u64,
        &mut scratch,
    )
    .unwrap();
    let consumed = sink
        .bindings_with_offsets(&[
            (5, 4096),
            (6, 0),
            (5, 4096 + 8 * 32 * 128 * 4),
            (7, 0),
            (8, 0),
            (9, 0),
        ])
        .unwrap();
    for (producer_port, consumer_port) in [(4, 0), (5, 2)] {
        let produced = bindings.iter().nth(producer_port).unwrap();
        let consumed = consumed.iter().nth(consumer_port).unwrap();
        assert_eq!(
            (produced.buffer, produced.offset, produced.len),
            (consumed.buffer, consumed.offset, consumed.len)
        );
    }
    assert_eq!(
        admitted
            .bindings_with_offsets(&[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 4),])
            .err(),
        Some(Error::Binding)
    );
    let legacy = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE,
        ..build
    };
    assert_eq!(
        AdmittedGraph::admit(
            graph,
            artifact,
            &legacy,
            policy(),
            artifact.len() as u64,
            &mut scratch,
        )
        .err(),
        Some(Error::UnsupportedOperation(82))
    );
}

mod fixture {
    include!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/native.rs");
}
const GRAPH: &[u8] = include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/graph.ax");
const MODEL: &[u8] = include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/model.blob");

#[test]
fn caller_budget_and_scheduler_scratch_are_enforced() {
    for (resident, tensor, expected) in [
        (1, 1024, Error::Budget),
        (1 << 20, 8, Error::Budget),
        (1 << 20, 1024, Error::Schedule),
    ] {
        let result = AdmittedGraph::admit(
            GRAPH,
            MODEL,
            &fixture::BUILD,
            Policy {
                minimum_accuracy: KernelAccuracy::Approximate,
                max_tensor_bytes: tensor,
                max_resident_bytes: resident,
            },
            1 << 20,
            &mut ScheduleScratch {
                adjacency: &mut [],
                producers: &mut [],
                indegree: &mut [],
                queue: &mut [],
                order: &mut [],
            },
        );
        assert_eq!(result.err(), Some(expected));
    }
}

fn admit<'a>(
    graph: &'a [u8],
    build: &'a ArtifactBuild<'a>,
    accuracy: KernelAccuracy,
) -> Result<AdmittedGraph<'a>, Error> {
    AdmittedGraph::admit(
        graph,
        MODEL,
        build,
        Policy {
            minimum_accuracy: accuracy,
            max_tensor_bytes: 1 << 20,
            max_resident_bytes: 4 << 20,
        },
        1 << 20,
        &mut ScheduleScratch {
            adjacency: &mut [0; 2],
            producers: &mut [0; 4],
            indegree: &mut [0; 2],
            queue: &mut [0; 2],
            order: &mut [0; 2],
        },
    )
}

#[test]
fn real_compilation_admits_and_derives_exact_accelerator_requests() {
    let graph = admit(GRAPH, &fixture::BUILD, KernelAccuracy::Approximate).unwrap();
    assert_eq!(graph.input_bytes(0), Ok(24));
    assert_eq!(graph.output_bytes(0), Ok(32));
    assert_eq!(graph.input_bytes(1), Err(Error::Ports));
    assert_eq!(graph.check_input_f32(0, &[1., 2., 3., -1., 0., 2.]), Ok(()));
    assert_eq!(graph.check_input_f32(0, &[0.; 5]), Err(Error::Binding));
    let bindings = graph.bindings(&[17, 21]).unwrap();
    let mut ports = bindings.iter();
    let input = ports.next().unwrap();
    assert_eq!(
        (input.buffer, input.slot, input.len, input.access),
        (17, 0, 24, AccessMode::Read)
    );
    let output = ports.next().unwrap();
    assert_eq!(
        (output.buffer, output.slot, output.len, output.access),
        (21, 1, 32, AccessMode::Write)
    );
    assert!(ports.next().is_none());
    assert!(
        matches!(graph.load_request(7, 1024), Ok(AccelRequest::LoadProgram {
        ctx: 7, stage: 1024, len: 61568, resident_bytes: 1048576, format: 1, target,
    }) if target == [0; 12])
    );
    assert_eq!(graph.load_request(7, u64::MAX).err(), Some(Error::Budget));
    assert!(graph.bindings(&[17]).is_err());
    assert!(graph.bindings(&[0, 21]).is_err());
    assert!(graph.bindings(&[17, 17]).is_err());
}

#[test]
fn imported_page_offsets_preserve_tensor_ranges_and_refuse_aliases() {
    let graph = admit(GRAPH, &fixture::BUILD, KernelAccuracy::Approximate).unwrap();
    let bindings = graph
        .bindings_with_offsets(&[(17, 128), (17, 152)])
        .unwrap();
    let mut ports = bindings.iter();
    let input = ports.next().unwrap();
    let output = ports.next().unwrap();
    assert_eq!((input.buffer, input.offset, input.len), (17, 128, 24));
    assert_eq!((output.buffer, output.offset, output.len), (17, 152, 32));
    assert_eq!(
        (input.access, output.access),
        (AccessMode::Read, AccessMode::Write)
    );
    assert!(ports.next().is_none());

    // Both sides of a page-rounded child route may carry interior data. The
    // hardware provider separately checks the imported allocation's extent.
    assert!(graph.bindings_with_offsets(&[(17, 4095), (21, 63)]).is_ok());
    assert!(graph.bindings_with_offsets(&[(17, 128), (17, 32)]).is_ok());
    for overlap in [127, 128, 151] {
        assert_eq!(
            graph
                .bindings_with_offsets(&[(17, 128), (17, overlap)])
                .err(),
            Some(Error::Binding)
        );
    }
    assert_eq!(
        graph
            .bindings_with_offsets(&[(17, u64::MAX - 22), (21, 0)])
            .err(),
        Some(Error::Binding)
    );
    assert_eq!(
        graph
            .bindings_with_offsets(&[(17, 0), (21, u64::MAX - 31)])
            .err(),
        Some(Error::Binding)
    );
    assert_eq!(
        graph.bindings_with_offsets(&[(17, 0)]).err(),
        Some(Error::Binding)
    );
}

#[test]
fn policy_identity_and_port_failures_happen_before_device_operations() {
    for accuracy in [KernelAccuracy::Faithful, KernelAccuracy::Strict] {
        assert_eq!(
            admit(GRAPH, &fixture::BUILD, accuracy).err(),
            Some(Error::Accuracy)
        );
    }
    let mut changed_graph = [0; 276];
    changed_graph.copy_from_slice(GRAPH);
    changed_graph[275] ^= 1;
    assert_eq!(
        admit(&changed_graph, &fixture::BUILD, KernelAccuracy::Approximate).err(),
        Some(Error::Identity)
    );
    let mut changed_build = fixture::BUILD;
    changed_build.artifact_blake3[0] ^= 1;
    assert_eq!(
        admit(GRAPH, &changed_build, KernelAccuracy::Approximate).err(),
        Some(Error::Identity)
    );
    let mut changed_build = fixture::BUILD;
    changed_build.outputs = &[ArtifactPort {
        tensor: ax::TensorId(3),
        slot: 0,
    }];
    assert_eq!(
        admit(GRAPH, &changed_build, KernelAccuracy::Approximate).err(),
        Some(Error::Ports)
    );
    let mut changed_build = fixture::BUILD;
    changed_build.inputs = &[ArtifactPort {
        tensor: ax::TensorId(1),
        slot: 0,
    }];
    assert_eq!(
        admit(GRAPH, &changed_build, KernelAccuracy::Approximate).err(),
        Some(Error::Ports)
    );
}

mod matmul_region {
    include!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/matmul/native.rs");
    include!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/matmul/region.rs");
    pub const GRAPH: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/matmul/graph.ax");
    pub const MODEL: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/matmul/model.blob");
}
mod relu_region {
    include!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/relu/native.rs");
    include!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/relu/region.rs");
    pub const GRAPH: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/relu/graph.ax");
    pub const MODEL: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/relu/model.blob");
}

fn compiled_region<'a>(
    bytes: &'a [u8],
    model: &'a [u8],
    build: &'a ArtifactBuild<'a>,
    parent: &'a [u8],
    region: &'a kore_axiom::accelerator::RegionBuild<'a>,
) -> Result<kore_axiom::accelerator::AdmittedRegion<'a>, Error> {
    AdmittedGraph::admit(
        bytes,
        model,
        build,
        Policy {
            minimum_accuracy: KernelAccuracy::Approximate,
            max_tensor_bytes: 1 << 20,
            max_resident_bytes: 4 << 20,
        },
        1 << 20,
        &mut ScheduleScratch {
            adjacency: &mut [0; 2],
            producers: &mut [0; 4],
            indegree: &mut [0; 2],
            queue: &mut [0; 2],
            order: &mut [0; 2],
        },
    )?
    .into_region(
        parent,
        region,
        &mut ScheduleScratch {
            adjacency: &mut [0; 2],
            producers: &mut [0; 4],
            indegree: &mut [0; 2],
            queue: &mut [0; 2],
            order: &mut [0; 2],
        },
        &mut [0; 4],
    )
}

#[test]
fn compiled_regions_bind_the_same_parent_intermediate() {
    let matmul = compiled_region(
        matmul_region::GRAPH,
        matmul_region::MODEL,
        &matmul_region::BUILD,
        GRAPH,
        &matmul_region::REGION,
    )
    .unwrap();
    let relu = compiled_region(
        relu_region::GRAPH,
        relu_region::MODEL,
        &relu_region::BUILD,
        GRAPH,
        &relu_region::REGION,
    )
    .unwrap();
    let residency = [17, 0, 21, 23];
    let a = matmul.bindings(&residency).unwrap();
    let b = relu.bindings(&residency).unwrap();
    assert_eq!(a.iter().nth(1).unwrap().buffer, 21);
    assert_eq!(b.iter().next().unwrap().buffer, 21);
    assert_eq!(b.iter().next().unwrap().len, 32);
    assert_eq!(b.iter().next().unwrap().access, AccessMode::Read);
    assert_eq!(b.iter().nth(1).unwrap().buffer, 23);
    assert!(relu.bindings(&[17, 0, 0, 23]).is_err());
    assert!(relu.bindings(&[17, 0, 21, 21]).is_err());
    assert!(relu.bindings(&[21, 23]).is_err());

    let shared = relu
        .bindings_with_offsets(&[(0, 0), (0, 0), (37, 64), (37, 96)])
        .unwrap();
    assert_eq!(shared.iter().next().unwrap().offset, 64);
    assert_eq!(shared.iter().nth(1).unwrap().offset, 96);
    assert_eq!(
        relu.bindings_with_offsets(&[(0, 0), (0, 0), (37, 64), (37, 80)])
            .err(),
        Some(Error::Binding)
    );
}

#[test]
fn region_metadata_cannot_substitute_tensor_identity_or_constant_contents() {
    use kore_axiom::accelerator::RegionBuild;
    let wrong = RegionBuild {
        tensors: &[ax::TensorId(1), ax::TensorId(3)],
        ..relu_region::REGION
    };
    assert_eq!(
        compiled_region(
            relu_region::GRAPH,
            relu_region::MODEL,
            &relu_region::BUILD,
            GRAPH,
            &wrong
        )
        .err(),
        Some(Error::Region)
    );
    let wrong = RegionBuild {
        order_start: 0,
        ..relu_region::REGION
    };
    assert_eq!(
        compiled_region(
            relu_region::GRAPH,
            relu_region::MODEL,
            &relu_region::BUILD,
            GRAPH,
            &wrong
        )
        .err(),
        Some(Error::Region)
    );
    let mut parent = *include_bytes!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/graph.ax");
    let constant = ax::Graph::parse(&parent)
        .unwrap()
        .tensor(ax::TensorId(1))
        .unwrap()
        .constant();
    let offset = constant.as_ptr() as usize - parent.as_ptr() as usize;
    parent[offset] ^= 1; // A different finite F32 weight in the same valid container.
    assert_eq!(
        compiled_region(
            matmul_region::GRAPH,
            matmul_region::MODEL,
            &matmul_region::BUILD,
            &parent,
            &matmul_region::REGION
        )
        .err(),
        Some(Error::Identity)
    );
    let rehashed = RegionBuild {
        source_graph_blake3: *blake3::hash(&parent).as_bytes(),
        ..matmul_region::REGION
    };
    assert_eq!(
        compiled_region(
            matmul_region::GRAPH,
            matmul_region::MODEL,
            &matmul_region::BUILD,
            &parent,
            &rehashed
        )
        .err(),
        Some(Error::Region)
    );
}

fn region_pair() -> [kore_axiom::accelerator::AdmittedRegion<'static>; 2] {
    [
        compiled_region(
            matmul_region::GRAPH,
            matmul_region::MODEL,
            &matmul_region::BUILD,
            GRAPH,
            &matmul_region::REGION,
        )
        .unwrap(),
        compiled_region(
            relu_region::GRAPH,
            relu_region::MODEL,
            &relu_region::BUILD,
            GRAPH,
            &relu_region::REGION,
        )
        .unwrap(),
    ]
}

#[test]
fn complete_sequence_preserves_intermediate_and_binds_capture_identity() {
    use kore_axiom::accelerator::AdmittedSequence;
    let regions = region_pair();
    let sequence = AdmittedSequence::new(&regions).unwrap();
    let mut assignments = [0; 4];
    let mut storage = [kore_axiom::accelerator::IntermediateStorage::default(); 4];
    let plan = sequence
        .plan_intermediates(
            4096,
            &mut assignments,
            &mut storage,
            &mut [kore_axiom::accelerator::ResidencyInterval::default(); 4],
        )
        .unwrap();
    assert_eq!(plan.bytes(), 4096);
    assert_eq!(plan.storage().len(), 1);
    assert!(
        matches!(plan.storage()[0].create_request(9), AccelRequest::CreateBuffer {
        ctx: 9, bytes: 4096, align: 4096, domain: 3, usage,
    } if usage == kore_svc_accel::contract::BufferUsage::MUTABLE_STATE.bits())
    );
    let mut residency = [17, 0, 0, 23];
    plan.assign_buffers(&[21], &mut residency).unwrap();
    assert_eq!(residency, [17, 0, 21, 23]);
    let programs = [31, 32];
    let mut launches = sequence
        .bind(7, &programs, &residency, 100, &mut [0; 4])
        .unwrap();
    let a = launches.next().unwrap();
    let b = launches.next().unwrap();
    assert!(launches.next().is_none());
    assert_eq!((a.program, b.program, a.queue, b.queue), (31, 32, 7, 7));
    assert_eq!(
        a.bindings.iter().nth(1).unwrap().buffer,
        b.bindings.iter().next().unwrap().buffer
    );
    assert_eq!(a.bindings.iter().nth(1).unwrap().access, AccessMode::Write);
    assert_eq!(b.bindings.iter().next().unwrap().access, AccessMode::Read);
    assert_eq!(
        blake3::Hash::from(sequence.identity()).to_hex().as_str(),
        include_str!("../../../drivers/ivpu/model/ax_matmul_relu_40xx/regions/sequence.blake3")
            .trim()
    );
}

#[test]
fn complete_sequence_binds_disjoint_offsets_without_tensor_copy() {
    use kore_axiom::accelerator::{AdmittedSequence, BufferRange};

    let regions = region_pair();
    let sequence = AdmittedSequence::new(&regions).unwrap();
    // Three live parent tensors occupy disjoint ranges of one imported DMA
    // allocation. Parent tensor 1 is internal to the extracted regions.
    let residency = [(17, 0), (0, 0), (17, 4096), (17, 8192)];
    let programs = [31, 32];
    let mut launches = sequence
        .bind_with_offsets(
            7,
            &programs,
            &residency,
            100,
            &mut [BufferRange::default(); 4],
        )
        .unwrap();
    let producer = launches.next().unwrap();
    let consumer = launches.next().unwrap();
    assert!(launches.next().is_none());
    let produced = producer.bindings.iter().nth(1).unwrap();
    let consumed = consumer.bindings.iter().next().unwrap();
    assert_eq!(
        (produced.buffer, produced.offset, produced.len),
        (consumed.buffer, consumed.offset, consumed.len)
    );
    assert_eq!(
        (produced.access, consumed.access),
        (AccessMode::Write, AccessMode::Read)
    );
    for bad in [
        [(17, 0), (0, 0), (17, 0), (17, 8192)],
        [(17, 0), (0, 0), (17, 4096), (17, 4096)],
        [(17, u64::MAX), (0, 0), (17, 4096), (17, 8192)],
    ] {
        assert!(
            sequence
                .bind_with_offsets(7, &programs, &bad, 100, &mut [BufferRange::default(); 4])
                .is_err()
        );
    }
}

#[test]
fn context_partials_exactly_cover_the_typed_kv_input() {
    use kore_axiom::accelerator::{
        AdmittedRegion, AdmittedSequence, BufferRange, PackedView, RegionBuild,
    };

    let partial = ax::context_partial_i8::graph_in(std::alloc::Global).unwrap();
    let kv = ax::context_kv::graph_in(std::alloc::Global).unwrap();
    let mut parent = ax::GraphBuilder::new_in(std::alloc::Global);
    let input = ax::TensorFlags::INPUT;
    let mut maps = [[ax::TensorId(0); 6]; 3];
    for map in maps.iter_mut().take(2) {
        for slot in map.iter_mut().take(3) {
            *slot = parent
                .tensor(ax::DType::F32, input, &[8, 2688], &[])
                .unwrap();
        }
        map[3] = parent
            .tensor(ax::DType::I8, input, &[2688, 8064], &[])
            .unwrap();
        map[4] = parent
            .tensor(ax::DType::F32, input, &[1, 2688], &[])
            .unwrap();
        map[5] = parent
            .tensor(ax::DType::F32, ax::TensorFlags::OUTPUT, &[8, 2688], &[])
            .unwrap();
        parent
            .node(
                ax::OpCode::CONTEXT_PARTIAL_I8_BLOCK,
                ax::NodeFlags::default(),
                &map[..5],
                &map[5..],
                &[],
            )
            .unwrap();
    }
    let map = &mut maps[2];
    map[0] = parent
        .tensor(ax::DType::F32, input, &[2, 8, 2688], &[])
        .unwrap();
    map[1] = parent
        .tensor(ax::DType::F32, input, &[2688, 512], &[])
        .unwrap();
    map[2] = parent.tensor(ax::DType::F32, input, &[2816], &[]).unwrap();
    map[3] = parent
        .tensor(ax::DType::F32, input, &[8, 128], &[])
        .unwrap();
    for slot in map.iter_mut().skip(4) {
        *slot = parent
            .tensor(ax::DType::F32, ax::TensorFlags::OUTPUT, &[8, 2, 128], &[])
            .unwrap();
    }
    parent
        .node(
            ax::OpCode::CONTEXT_KV_BLOCK,
            ax::NodeFlags::default(),
            &map[..4],
            &map[4..],
            &[],
        )
        .unwrap();
    let parent = parent.finish().unwrap();
    let parent_bytes = parent.as_bytes();
    let parent_digest = *blake3::hash(parent_bytes).as_bytes();
    let partial_inputs = [0, 1, 2, 3, 4].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let partial_outputs = [ArtifactPort {
        tensor: ax::TensorId(5),
        slot: 5,
    }];
    let kv_inputs = [0, 1, 2, 3].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let kv_outputs = [4, 5].map(|id| ArtifactPort {
        tensor: ax::TensorId(id),
        slot: id,
    });
    let partial_build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_CONTEXT_PARTIAL_I8_BLOCK,
        graph_blake3: *blake3::hash(partial.as_bytes()).as_bytes(),
        artifact_blake3: *blake3::hash(b"partial").as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &partial_inputs,
        outputs: &partial_outputs,
    };
    let kv_build = ArtifactBuild {
        profile: kore_axiom::accelerator::PROFILE_CONTEXT_KV_BLOCK,
        graph_blake3: *blake3::hash(kv.as_bytes()).as_bytes(),
        artifact_blake3: *blake3::hash(b"kv").as_bytes(),
        format: 1,
        target: [0; 12],
        inputs: &kv_inputs,
        outputs: &kv_outputs,
    };
    let policy = || Policy {
        minimum_accuracy: KernelAccuracy::Approximate,
        max_tensor_bytes: 32 << 20,
        max_resident_bytes: 1 << 20,
    };
    let region_builds: [RegionBuild<'_>; 3] = core::array::from_fn(|index| RegionBuild {
        source_graph_blake3: parent_digest,
        order_start: index as u32,
        node_count: 1,
        tensors: &maps[index],
    });
    let admitted: [AdmittedGraph<'_>; 3] = core::array::from_fn(|index| {
        let (graph, artifact, build) = if index == 2 {
            (kv.as_bytes(), &b"kv"[..], &kv_build)
        } else {
            (partial.as_bytes(), &b"partial"[..], &partial_build)
        };
        AdmittedGraph::admit(
            graph,
            artifact,
            build,
            policy(),
            artifact.len() as u64,
            &mut ScheduleScratch {
                adjacency: &mut [0; 1],
                producers: &mut [0; 6],
                indegree: &mut [0; 1],
                queue: &mut [0; 1],
                order: &mut [0; 1],
            },
        )
        .unwrap()
    });
    let mut admitted = admitted.map(Some);
    let extracted: [AdmittedRegion<'_>; 3] = core::array::from_fn(|index| {
        admitted[index]
            .take()
            .unwrap()
            .into_region(
                parent_bytes,
                &region_builds[index],
                &mut ScheduleScratch {
                    adjacency: &mut [0; 3],
                    producers: &mut [0; 18],
                    indegree: &mut [0; 3],
                    queue: &mut [0; 3],
                    order: &mut [0; 3],
                },
                &mut [0; 18],
            )
            .unwrap()
    });
    let sequence = AdmittedSequence::new(&extracted).unwrap();
    let mut residency = [(0, 0); 18];
    for (index, slot) in residency.iter_mut().enumerate() {
        *slot = (100 + index as u64, 0);
    }
    residency[maps[0][5].0 as usize] = (30, 0);
    residency[maps[1][5].0 as usize] = (30, 86_016);
    residency[maps[2][0].0 as usize] = (30, 0);
    residency[maps[2][4].0 as usize] = (40, 0);
    residency[maps[2][5].0 as usize] = (40, 8_192);
    let programs = [31, 31, 32];
    assert!(
        sequence
            .bind_with_offsets(
                7,
                &programs,
                &residency,
                100,
                &mut [BufferRange::default(); 18],
            )
            .is_err()
    );
    let producers = [maps[0][5], maps[1][5]];
    let view = [PackedView {
        consumer: maps[2][0],
        producers: &producers,
    }];
    let launches: Vec<_> = sequence
        .bind_with_packed_views(
            7,
            &programs,
            &residency,
            &view,
            100,
            &mut [BufferRange::default(); 18],
        )
        .unwrap()
        .collect();
    assert_eq!(launches.len(), 3);
    let first = launches[0].bindings.iter().nth(5).unwrap();
    let second = launches[1].bindings.iter().nth(5).unwrap();
    let consumed = launches[2].bindings.iter().next().unwrap();
    assert_eq!((first.buffer, first.offset, first.len), (30, 0, 86_016));
    assert_eq!(
        (second.buffer, second.offset, second.len),
        (30, 86_016, 86_016)
    );
    assert_eq!(
        (consumed.buffer, consumed.offset, consumed.len),
        (30, 0, 172_032)
    );
    for (index, replacement) in [
        (maps[1][5], (30, 86_020)),
        (maps[1][5], (29, 86_016)),
        (maps[0][0], (30, 0)),
    ] {
        let mut bad = residency;
        bad[index.0 as usize] = replacement;
        assert!(
            sequence
                .bind_with_packed_views(
                    7,
                    &programs,
                    &bad,
                    &view,
                    100,
                    &mut [BufferRange::default(); 18],
                )
                .is_err()
        );
    }
    let reversed = [producers[1], producers[0]];
    assert!(
        sequence
            .bind_with_packed_views(
                7,
                &programs,
                &residency,
                &[PackedView {
                    consumer: maps[2][0],
                    producers: &reversed,
                }],
                100,
                &mut [BufferRange::default(); 18],
            )
            .is_err()
    );
}

#[test]
fn native_qkv_to_sink_sequence_retains_exact_shared_dma_views() {
    use kore_axiom::accelerator::{AdmittedSequence, BufferRange, RegionBuild};

    mod qkv_bundle {
        include!("../../../drivers/ivpu/model/dspark_qkv_prep_40xx/native.rs");
    }
    mod sink_bundle {
        include!("../../../drivers/ivpu/model/dspark_sink_gqa_40xx/native.rs");
    }
    const QKV_GRAPH: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/dspark_qkv_prep_40xx/graph.ax");
    const QKV_ELF: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/dspark_qkv_prep_40xx/model.blob");
    const SINK_GRAPH: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/dspark_sink_gqa_40xx/graph.ax");
    const SINK_ELF: &[u8] =
        include_bytes!("../../../drivers/ivpu/model/dspark_sink_gqa_40xx/model.blob");

    let mut parent = ax::GraphBuilder::new_in(std::alloc::Global);
    let input = ax::TensorFlags::INPUT;
    let hidden = parent
        .tensor(ax::DType::F32, input, &[8, 2688], &[])
        .unwrap();
    let weights = parent
        .tensor(ax::DType::F32, input, &[2688, 4608], &[])
        .unwrap();
    let norms = parent.tensor(ax::DType::F32, input, &[2944], &[]).unwrap();
    let phase = parent
        .tensor(ax::DType::F32, input, &[8, 128], &[])
        .unwrap();
    let query = parent
        .tensor(
            ax::DType::F32,
            ax::TensorFlags::default(),
            &[8, 32, 128],
            &[],
        )
        .unwrap();
    let ring = parent
        .tensor(ax::DType::F32, input, &[2, 1024, 2, 128], &[])
        .unwrap();
    let block = parent
        .tensor(
            ax::DType::F32,
            ax::TensorFlags::default(),
            &[2, 8, 2, 128],
            &[],
        )
        .unwrap();
    let sinks = parent.tensor(ax::DType::F32, input, &[32], &[]).unwrap();
    let controls = parent.tensor(ax::DType::I32, input, &[2], &[]).unwrap();
    let output = parent
        .tensor(ax::DType::F32, ax::TensorFlags::OUTPUT, &[8, 32, 128], &[])
        .unwrap();
    parent
        .node(
            ax::OpCode::QKV_PREP_BLOCK,
            ax::NodeFlags::default(),
            &[hidden, weights, norms, phase],
            &[query, block],
            &[],
        )
        .unwrap();
    parent
        .node(
            ax::OpCode::SINK_GQA_BLOCK,
            ax::NodeFlags::default(),
            &[query, ring, block, sinks, controls],
            &[output],
            &[],
        )
        .unwrap();
    let parent = parent.finish().unwrap();
    let parent_bytes = parent.as_bytes();
    let parent_digest = *blake3::hash(parent_bytes).as_bytes();
    let qkv_map = [hidden, weights, norms, phase, query, block];
    let sink_map = [query, ring, block, sinks, controls, output];
    let qkv_region = RegionBuild {
        source_graph_blake3: parent_digest,
        order_start: 0,
        node_count: 1,
        tensors: &qkv_map,
    };
    let sink_region = RegionBuild {
        source_graph_blake3: parent_digest,
        order_start: 1,
        node_count: 1,
        tensors: &sink_map,
    };
    let admit = |graph: &'static [u8],
                 elf: &'static [u8],
                 build: &'static ArtifactBuild<'static>,
                 max_tensor_bytes: u64| {
        let resident = elf.len() as u64 + (4 << 20);
        AdmittedGraph::admit(
            graph,
            elf,
            build,
            Policy {
                minimum_accuracy: KernelAccuracy::Approximate,
                max_tensor_bytes,
                max_resident_bytes: resident,
            },
            resident,
            &mut ScheduleScratch {
                adjacency: &mut [0; 1],
                producers: &mut [0; 6],
                indegree: &mut [0; 1],
                queue: &mut [0; 1],
                order: &mut [0; 1],
            },
        )
        .unwrap()
    };
    let qkv = admit(QKV_GRAPH, QKV_ELF, &qkv_bundle::BUILD, 49_545_216)
        .into_region(
            parent_bytes,
            &qkv_region,
            &mut ScheduleScratch {
                adjacency: &mut [0; 2],
                producers: &mut [0; 10],
                indegree: &mut [0; 2],
                queue: &mut [0; 2],
                order: &mut [0; 2],
            },
            &mut [0; 10],
        )
        .unwrap();
    let sink = admit(SINK_GRAPH, SINK_ELF, &sink_bundle::BUILD, 2_097_152)
        .into_region(
            parent_bytes,
            &sink_region,
            &mut ScheduleScratch {
                adjacency: &mut [0; 2],
                producers: &mut [0; 10],
                indegree: &mut [0; 2],
                queue: &mut [0; 2],
                order: &mut [0; 2],
            },
            &mut [0; 10],
        )
        .unwrap();
    let regions = [qkv, sink];
    let sequence = AdmittedSequence::new(&regions).unwrap();
    // Q, block K/V, and final attention output occupy disjoint 4K-aligned
    // ranges in one imported buffer. No producer output is copied to a new
    // consumer input allocation between these two program submissions.
    let residency = [
        (10, 0),
        (11, 0),
        (12, 0),
        (13, 0),
        (20, 4096),
        (14, 0),
        (20, 135_168),
        (15, 0),
        (16, 0),
        (20, 151_552),
    ];
    let programs = [31, 32];
    let mut launches = sequence
        .bind_with_offsets(
            7,
            &programs,
            &residency,
            100,
            &mut [BufferRange::default(); 10],
        )
        .unwrap();
    let producer = launches.next().unwrap();
    let attention = launches.next().unwrap();
    assert!(launches.next().is_none());
    for (produced, consumed) in [(4, 0), (5, 2)] {
        let produced = producer.bindings.iter().nth(produced).unwrap();
        let consumed = attention.bindings.iter().nth(consumed).unwrap();
        assert_eq!(
            (produced.buffer, produced.offset, produced.len),
            (consumed.buffer, consumed.offset, consumed.len)
        );
        assert_eq!(
            (produced.access, consumed.access),
            (AccessMode::Write, AccessMode::Read)
        );
    }
}

#[test]
fn sequence_refuses_holes_overlap_order_and_bad_residency_before_iteration() {
    use kore_axiom::accelerator::AdmittedSequence;
    let mut regions = region_pair();
    assert!(AdmittedSequence::new(&regions[..1]).is_err());
    assert!(AdmittedSequence::new(&regions[1..]).is_err());
    assert!(AdmittedSequence::new(&[]).is_err());
    regions.swap(0, 1);
    assert!(AdmittedSequence::new(&regions).is_err());
    regions.swap(0, 1);
    let duplicate = core::array::from_fn::<_, 2, _>(|_| {
        compiled_region(
            matmul_region::GRAPH,
            matmul_region::MODEL,
            &matmul_region::BUILD,
            GRAPH,
            &matmul_region::REGION,
        )
        .unwrap()
    });
    assert!(AdmittedSequence::new(&duplicate).is_err());
    let sequence = AdmittedSequence::new(&regions).unwrap();
    for buffers in [[17, 0, 0, 23], [17, 0, 21, 21], [17, 0, 21, 17]] {
        assert!(
            sequence
                .bind(7, &[31, 32], &buffers, 100, &mut [0; 4])
                .is_err()
        );
    }
    assert!(
        sequence
            .bind(7, &[31], &[17, 0, 21, 23], 100, &mut [0; 4])
            .is_err()
    );
    assert!(
        sequence
            .bind(7, &[31, 0], &[17, 0, 21, 23], 100, &mut [0; 4])
            .is_err()
    );
    assert!(
        sequence
            .bind(0, &[31, 32], &[17, 0, 21, 23], 100, &mut [0; 4])
            .is_err()
    );
    assert!(
        sequence
            .bind(7, &[31, 32], &[17, 21, 23], 100, &mut [0; 4])
            .is_err()
    );
    assert!(
        sequence
            .bind(7, &[31, 32], &[17, 0, 21, 23], 100, &mut [0; 3])
            .is_err()
    );
}
