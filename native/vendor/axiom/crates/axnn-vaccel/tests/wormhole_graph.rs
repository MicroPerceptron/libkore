#![feature(allocator_api)]

use std::alloc::Global;
use std::{hint::black_box, time::Instant};

use ax::{DENSE_IMPLEMENTATION, Graph, Partition, Schedule, TensorId};
use axgeom::{
    graph::{WormholeGraphSpec, abi, build_wormhole_graph_in},
    wormhole::{DnegParams, PlanarRay, TraceConfig, velocity_verlet_step},
};
use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxFeedback, AxHetero, AxTosaAccelerator, Session, artifact,
};

const PLACEMENT_RAYS: usize = 8_192;
const STEPS: u32 = 8;

// The graph deliberately uses Rajan's inexpensive atan approximation while
// the scalar oracle uses axmath's atan. Keep each accumulated error budget
// explicit so a future approximation change cannot silently loosen coverage.
const ELL_ABS_TOLERANCE: f32 = 5.0e-4;
const PHI_ABS_TOLERANCE: f32 = 3.0e-3;
const P_ELL_ABS_TOLERANCE: f32 = 5.0e-4;

#[derive(Clone, Copy)]
struct Lane {
    ray: PlanarRay,
    active: u8,
}

const LANES: [Lane; 8] = [
    Lane {
        ray: PlanarRay::new(5.0, 0.2, -0.8, 0.6),
        active: 1,
    },
    Lane {
        ray: PlanarRay::new(-4.0, -0.4, 0.65, -0.8),
        active: 1,
    },
    Lane {
        ray: PlanarRay::new(0.5, 1.2, -0.4, 0.9),
        active: 1,
    },
    Lane {
        ray: PlanarRay::new(2.05, -0.7, -0.3, 0.75),
        active: 1,
    },
    // A caller-disabled lane must remain bitwise frozen.
    Lane {
        ray: PlanarRay::new(3.1, 0.3, -0.2, 0.8),
        active: 0,
    },
    // Initially escaped lanes are deactivated before the first step.
    Lane {
        ray: PlanarRay::new(10.25, 0.9, 0.5, 0.2),
        active: 1,
    },
    Lane {
        ray: PlanarRay::new(-10.25, -0.9, -0.5, -0.2),
        active: 1,
    },
    // This lane crosses the positive escape surface on its first step; the
    // remaining seven unrolled steps must see active=false and freeze it.
    Lane {
        ray: PlanarRay::new(9.98, -1.0, 0.5, 0.2),
        active: 1,
    },
];

struct InputPlanes {
    ell: Vec<u8>,
    phi: Vec<u8>,
    p_ell: Vec<u8>,
    impact: Vec<u8>,
    active: Vec<u8>,
}

impl InputPlanes {
    fn from_lanes(lanes: &[Lane]) -> Self {
        Self {
            ell: f32_bytes(lanes.iter().map(|lane| lane.ray.ell())),
            phi: f32_bytes(lanes.iter().map(|lane| lane.ray.phi())),
            p_ell: f32_bytes(lanes.iter().map(|lane| lane.ray.p_ell())),
            impact: f32_bytes(lanes.iter().map(|lane| lane.ray.impact_parameter)),
            active: lanes.iter().map(|lane| lane.active).collect(),
        }
    }

    fn for_tensor(&self, tensor: TensorId) -> &[u8] {
        match tensor.0 {
            0 => &self.ell,
            1 => &self.phi,
            2 => &self.p_ell,
            3 => &self.impact,
            4 => &self.active,
            _ => panic!("unexpected wormhole ABI input tensor {}", tensor.0),
        }
    }

    fn abi_order(&self) -> [&[u8]; 5] {
        [
            &self.ell,
            &self.phi,
            &self.p_ell,
            &self.impact,
            &self.active,
        ]
    }
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn escaped(config: &TraceConfig, ray: &PlanarRay) -> bool {
    (ray.ell() <= -config.escape_ell && ray.p_ell() < 0.0)
        || (ray.ell() >= config.escape_ell && ray.p_ell() > 0.0)
}

fn scalar_steps(params: &DnegParams, config: &TraceConfig, mut lane: Lane, steps: u32) -> Lane {
    let mut active = lane.active != 0 && !escaped(config, &lane.ray);
    for _ in 0..steps {
        if !active {
            break;
        }
        velocity_verlet_step(params, &mut lane.ray, config.step_size);
        active = !escaped(config, &lane.ray);
    }
    lane.active = u8::from(active);
    lane
}

fn execute_axtosa(
    graph: &Graph<'_>,
    inputs: &InputPlanes,
    expected_steps: u32,
) -> (Vec<TensorId>, Vec<Vec<u8>>) {
    let analysis = Schedule::analyze_in(graph, Global).unwrap();
    let partition = Partition {
        backend: ACCEL_BACKEND,
        implementation: DENSE_IMPLEMENTATION,
        order_start: 0,
        node_count: graph.node_count() as u32,
    };
    let lowered = artifact::ax_partition_in(
        graph,
        analysis.order(),
        analysis.producers(),
        partition,
        Global,
    )
    .unwrap();

    let mut input_ids = lowered.inputs.to_vec();
    input_ids.sort_by_key(|id| id.0);
    assert_eq!(input_ids, (0..5).map(TensorId).collect::<Vec<_>>());
    let mut output_ids = lowered.outputs.to_vec();
    output_ids.sort_by_key(|id| id.0);
    assert_eq!(output_ids, (5..9).map(TensorId).collect::<Vec<_>>());

    let input_refs: Vec<_> = lowered
        .inputs
        .iter()
        .map(|&tensor| inputs.for_tensor(tensor))
        .collect();
    let program = axtosa::Program::compile(&lowered.bytes).unwrap();
    let stats = program.stats();
    assert!(
        stats.scale_add_fusions >= expected_steps as usize * 6,
        "each wormhole step should expose its register-fused scale/add regions independent of topological adjacency: {stats:?}"
    );
    assert!(
        stats.constant_folds > 0,
        "serialized scalar constants should leave no warm CONST instructions: {stats:?}"
    );
    assert_eq!(
        stats.instructions
            + stats.scale_add_fusions
            + stats.mul_add_fusions
            + stats.lerp_fusions * 2
            + stats.linearized_clamp_fusions * 5
            + stats.guarded_sqrt_fusions * 2
            + stats.constant_folds,
        stats.operators
    );
    let outputs = program.execute(&input_refs).unwrap();
    assert_eq!(
        program.execute(&input_refs).unwrap(),
        outputs,
        "compiled program must be safely reusable across warm submissions"
    );
    (lowered.outputs.into_iter().collect(), outputs)
}

fn outputs_in_abi_order(ids: &[TensorId], outputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    (5..9)
        .map(|id| {
            let tensor = TensorId(id);
            let slot = ids
                .iter()
                .position(|candidate| *candidate == tensor)
                .unwrap();
            outputs[slot].clone()
        })
        .collect()
}

fn assert_close(label: &str, lane: usize, actual: f32, expected: f32, tolerance: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label} lane {lane}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
    );
}

#[test]
fn wormhole_graph_matches_scalar_and_forms_one_realistic_accelerator_region() {
    let params = DnegParams::REFERENCE;
    let trace = TraceConfig::new(0.1, STEPS, 10.0);
    let spec = WormholeGraphSpec {
        rays: PLACEMENT_RAYS as u32,
        steps: STEPS,
        params,
        trace,
    };
    let graph_bytes = build_wormhole_graph_in(spec, Global).unwrap();
    let graph = graph_bytes.parse().unwrap();

    let lanes: Vec<_> = (0..PLACEMENT_RAYS)
        .map(|index| LANES[index % LANES.len()])
        .collect();
    let inputs = InputPlanes::from_lanes(&lanes);
    let (direct_ids, direct_outputs) = execute_axtosa(&graph, &inputs, STEPS);
    let direct = outputs_in_abi_order(&direct_ids, &direct_outputs);

    let actual_ell = decode_f32(&direct[0]);
    let actual_phi = decode_f32(&direct[1]);
    let actual_p_ell = decode_f32(&direct[2]);
    let actual_active = &direct[3];
    for (index, &initial) in lanes.iter().enumerate() {
        let expected = scalar_steps(&params, &trace, initial, STEPS);
        assert_close(
            "ell",
            index,
            actual_ell[index],
            expected.ray.ell(),
            ELL_ABS_TOLERANCE,
        );
        assert_close(
            "phi",
            index,
            actual_phi[index],
            expected.ray.phi(),
            PHI_ABS_TOLERANCE,
        );
        assert_close(
            "p_ell",
            index,
            actual_p_ell[index],
            expected.ray.p_ell(),
            P_ELL_ABS_TOLERANCE,
        );
        assert_eq!(actual_active[index], expected.active, "active lane {index}");
    }

    for pattern in [4_usize, 5, 6] {
        assert_eq!(
            actual_ell[pattern].to_bits(),
            LANES[pattern].ray.ell().to_bits()
        );
        assert_eq!(
            actual_phi[pattern].to_bits(),
            LANES[pattern].ray.phi().to_bits()
        );
        assert_eq!(
            actual_p_ell[pattern].to_bits(),
            LANES[pattern].ray.p_ell().to_bits()
        );
        assert_eq!(actual_active[pattern], 0);
    }
    assert_eq!(actual_active[7], 0, "crossing lane was not frozen");

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let executable = runtime
        .compile_in(&graph, &AxDeviceModel::default(), Global)
        .unwrap();
    let placement = executable.placement_report();
    assert!(
        placement.nodes_reassigned > 0,
        "the core optimized-placement entry point must recover the launch-amortized wormhole region: {placement:?}"
    );
    assert_eq!(placement.aggregate_cost, executable.aggregate_cost());
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND),
        "the full 8K wormhole region should amortize one realistic accelerator launch"
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let runtime_inputs = inputs.abi_order();
    let scheduled = runtime
        .run_in(&executable, &runtime_inputs, Global)
        .unwrap();
    assert_eq!(
        scheduled, direct,
        "scheduled execution changed TOSA results"
    );
}

#[test]
#[ignore = "manual native CPU graph-executor performance probe"]
fn wormhole_cpu_register_fusion_probe() {
    let trace = TraceConfig::new(0.1, STEPS, 10.0);
    let spec = WormholeGraphSpec {
        rays: PLACEMENT_RAYS as u32,
        steps: STEPS,
        params: DnegParams::REFERENCE,
        trace,
    };
    let graph_bytes = build_wormhole_graph_in(spec, Global).unwrap();
    let graph = graph_bytes.parse().unwrap();
    let analysis = Schedule::analyze_in(&graph, Global).unwrap();
    let partition = Partition {
        backend: ACCEL_BACKEND,
        implementation: DENSE_IMPLEMENTATION,
        order_start: 0,
        node_count: graph.node_count() as u32,
    };
    let lowered = artifact::ax_partition_in(
        &graph,
        analysis.order(),
        analysis.producers(),
        partition,
        Global,
    )
    .unwrap();

    let lanes: Vec<_> = (0..PLACEMENT_RAYS)
        .map(|index| LANES[index % LANES.len()])
        .collect();
    let inputs = InputPlanes::from_lanes(&lanes);
    let input_refs: Vec<_> = lowered
        .inputs
        .iter()
        .map(|&tensor| inputs.for_tensor(tensor))
        .collect();
    let program = axtosa::Program::compile(&lowered.bytes).unwrap();

    for _ in 0..3 {
        black_box(program.execute(&input_refs).unwrap());
    }
    let repetitions = std::env::var("AXIOM_CPU_BENCH_REPS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(25);
    let started = Instant::now();
    for _ in 0..repetitions {
        black_box(program.execute(&input_refs).unwrap());
    }
    let elapsed = started.elapsed();
    let executions_per_second = f64::from(repetitions) / elapsed.as_secs_f64();
    let lane_steps_per_second = executions_per_second * (PLACEMENT_RAYS as f64) * f64::from(STEPS);
    println!(
        "wormhole CPU probe: {:?}; {:.3} ms/execute; {:.1} M lane-steps/s",
        program.stats(),
        elapsed.as_secs_f64() * 1_000.0 / f64::from(repetitions),
        lane_steps_per_second / 1_000_000.0,
    );
}

#[test]
fn full_wormhole_graph_coreml_profile_is_one_accelerator_partition() {
    let trace = TraceConfig::new(0.1, 64, 10.0);
    let graph_bytes = build_wormhole_graph_in(
        WormholeGraphSpec {
            rays: PLACEMENT_RAYS as u32,
            steps: 64,
            params: DnegParams::REFERENCE,
            trace,
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    assert_eq!(graph.node_count(), 4_010);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let executable = runtime
        .compile_in(&graph, &AxDeviceModel::coreml(), Global)
        .unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn resident_wormhole_preserves_coreml_boolean_bridge_without_intermediate_transfers() {
    const RAYS: usize = 64;
    let trace = TraceConfig::new(0.1, 2, 10.0);
    let graph_bytes = build_wormhole_graph_in(
        WormholeGraphSpec {
            rays: RAYS as u32,
            steps: 2,
            params: DnegParams::REFERENCE,
            trace,
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    let lanes: Vec<_> = (0..RAYS).map(|index| LANES[index % LANES.len()]).collect();
    let inputs = InputPlanes::from_lanes(&lanes);
    let initial = inputs.abi_order();

    let sessions = || {
        let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        AxHetero::new(cpu, Some(accel))
    };
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::coreml()
    };
    let mut reference_runtime = sessions();
    let reference_executable = reference_runtime
        .compile_in(&graph, &model, Global)
        .unwrap();
    let first = reference_runtime
        .run_in(&reference_executable, &initial, Global)
        .unwrap();
    let expected = reference_runtime
        .run_in(
            &reference_executable,
            &[&first[0], &first[1], &first[2], &inputs.impact, &first[3]],
            Global,
        )
        .unwrap();

    let mut resident_runtime = sessions();
    let resident_executable = resident_runtime.compile_in(&graph, &model, Global).unwrap();
    let provider = resident_runtime.accel.as_ref().unwrap().provider();
    let writes_before = provider.buffer_write_bytes();
    let reads_before = provider.buffer_read_bytes();
    let actual = resident_runtime
        .run_repeated_auto_in(
            &resident_executable,
            &initial,
            2,
            &[
                AxFeedback::new(abi::ELL_IN, abi::ELL_OUT),
                AxFeedback::new(abi::PHI_IN, abi::PHI_OUT),
                AxFeedback::new(abi::P_ELL_IN, abi::P_ELL_OUT),
                AxFeedback::new(abi::ACTIVE_IN, abi::ACTIVE_OUT),
            ],
            Global,
        )
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(
        resident_runtime
            .accel
            .as_ref()
            .unwrap()
            .provider()
            .buffer_write_bytes()
            - writes_before,
        (RAYS * 5 * 4) as u64,
        "resident recurrence uploaded an intermediate state"
    );
    assert_eq!(
        resident_runtime
            .accel
            .as_ref()
            .unwrap()
            .provider()
            .buffer_read_bytes()
            - reads_before,
        (RAYS * 4 * 4) as u64,
        "resident recurrence read an intermediate state back"
    );
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
#[test]
fn wormhole_graph_executes_as_one_openvino_partition() {
    use virtio_accel_openvino::{InitError, OPENVINO_TOSA_TARGET, OpenVinoAccelerator};

    let provider = match OpenVinoAccelerator::new() {
        Ok(provider) => provider,
        Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
            eprintln!("skipping: no OpenVINO inference device on this host");
            return;
        }
        Err(error) => panic!("OpenVINO init failed: {error:?}"),
    };

    let params = DnegParams::REFERENCE;
    let steps = 2;
    let trace = TraceConfig::new(0.1, steps, 10.0);
    let graph_bytes = build_wormhole_graph_in(
        WormholeGraphSpec {
            rays: 64,
            steps,
            params,
            trace,
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    let lanes: Vec<_> = (0..64).map(|index| LANES[index % LANES.len()]).collect();
    let inputs = InputPlanes::from_lanes(&lanes);
    let (direct_ids, direct_outputs) = execute_axtosa(&graph, &inputs, steps);
    let direct = outputs_in_abi_order(&direct_ids, &direct_outputs);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let openvino = Session::new(provider, OPENVINO_TOSA_TARGET).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(openvino));
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let runtime_inputs = inputs.abi_order();
    let actual = runtime
        .run_in(&executable, &runtime_inputs, Global)
        .unwrap();
    assert_eq!(actual[3], direct[3], "OpenVINO changed active masks");
    for (plane, tolerance) in [
        (0, ELL_ABS_TOLERANCE),
        (1, PHI_ABS_TOLERANCE),
        (2, P_ELL_ABS_TOLERANCE),
    ] {
        for (lane, (actual, expected)) in decode_f32(&actual[plane])
            .into_iter()
            .zip(decode_f32(&direct[plane]))
            .enumerate()
        {
            assert_close("OpenVINO", lane, actual, expected, tolerance);
        }
    }
}
