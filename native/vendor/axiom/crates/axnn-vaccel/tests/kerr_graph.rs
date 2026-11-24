#![feature(allocator_api)]

use std::alloc::Global;

use ax::{DENSE_IMPLEMENTATION, Graph, Partition, Schedule, TensorId};
use axgeom::{
    black_hole::{
        HamiltonianRay, KerrParams, TraceConfig, hamiltonian_rhs, horizon_radius, midpoint_step,
        radius,
    },
    kerr_graph::{KerrStepGraphSpec, abi, build_kerr_step_graph_in},
};
use axnn_vaccel::{ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, Session, artifact};

const NUMERIC_RAYS: usize = 7;
const PLACEMENT_RAYS: u32 = 8_192;
const POSITION_TOLERANCE: f32 = 1.0e-5;
const MOMENTUM_TOLERANCE: f32 = 1.0e-4;
const HIT_TOLERANCE: f32 = 1.0e-5;

#[derive(Clone, Copy)]
struct Lane {
    ray: HamiltonianRay,
    active: u8,
}

fn lanes() -> [Lane; NUMERIC_RAYS] {
    [
        // Ordinary active lane: one accepted step remains unresolved.
        Lane {
            ray: HamiltonianRay::new([0.0, 12.0, 1.0, 2.0], [-1.0, -0.9, 0.1, -0.05]),
            active: 1,
        },
        // A caller-disabled lane is bitwise frozen and emits no event.
        Lane {
            ray: HamiltonianRay::new([0.5, 9.0, -1.0, 1.5], [-0.75, -0.6, 0.05, -0.1]),
            active: 0,
        },
        // Already inside the horizon. Its z motion would otherwise cross the
        // disk, so this lane also locks capture-over-disk precedence.
        Lane {
            ray: HamiltonianRay::new([0.0, 1.6, 0.0, 0.03], [-1.0, -0.2, 0.0, -1.0]),
            active: 1,
        },
        // Already outside the escape sphere with positive radial velocity.
        Lane {
            ray: HamiltonianRay::new([0.0, 41.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 0.0]),
            active: 1,
        },
        // Crosses the equatorial thin disk during the accepted step at r~8.
        Lane {
            ray: HamiltonianRay::new([0.0, 8.0, 0.0, 0.03], [-1.0, 0.0, 0.0, -1.0]),
            active: 1,
        },
        // Starts just outside the horizon and advances inward.
        Lane {
            ray: HamiltonianRay::new([0.0, 1.94, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]),
            active: 1,
        },
        // Starts just inside the escape sphere and advances outward.
        Lane {
            ray: HamiltonianRay::new([0.0, 39.98, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]),
            active: 1,
        },
    ]
}

struct InputPlanes {
    position: [Vec<u8>; 4],
    momentum: [Vec<u8>; 4],
    active: Vec<u8>,
}

impl InputPlanes {
    fn from_lanes(lanes: &[Lane]) -> Self {
        Self {
            position: std::array::from_fn(|component| {
                f32_bytes(lanes.iter().map(|lane| lane.ray.position[component]))
            }),
            momentum: std::array::from_fn(|component| {
                f32_bytes(lanes.iter().map(|lane| lane.ray.momentum[component]))
            }),
            active: lanes.iter().map(|lane| lane.active).collect(),
        }
    }

    fn for_tensor(&self, tensor: TensorId) -> &[u8] {
        match tensor.0 {
            0..=3 => &self.position[tensor.0 as usize],
            4..=7 => &self.momentum[tensor.0 as usize - 4],
            8 => &self.active,
            _ => panic!("unexpected Kerr ABI input tensor {}", tensor.0),
        }
    }

    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    fn abi_order(&self) -> [&[u8]; 9] {
        [
            &self.position[0],
            &self.position[1],
            &self.position[2],
            &self.position[3],
            &self.momentum[0],
            &self.momentum[1],
            &self.momentum[2],
            &self.momentum[3],
            &self.active,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct Expected {
    ray: HamiltonianRay,
    active: u8,
    capture: u8,
    disk: u8,
    escape: u8,
    disk_position: [f32; 3],
    disk_radius: f32,
    disk_fraction: f32,
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "partial f32 output element");
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn outward(params: &KerrParams, ray: &HamiltonianRay) -> bool {
    let velocity = hamiltonian_rhs(params, ray, 1.0e-4).position;
    ray.position[1] * velocity[1] + ray.position[2] * velocity[2] + ray.position[3] * velocity[3]
        > 0.0
}

fn disk_hit(
    params: &KerrParams,
    trace: &TraceConfig,
    from: &HamiltonianRay,
    to: &HamiltonianRay,
) -> Option<([f32; 3], f32, f32)> {
    let (z0, z1) = (from.position[3], to.position[3]);
    if !((z0 > 0.0 && z1 <= 0.0) || (z0 < 0.0 && z1 >= 0.0)) {
        return None;
    }
    let fraction = z0 / (z0 - z1);
    let position = [
        from.position[1] + fraction * (to.position[1] - from.position[1]),
        from.position[2] + fraction * (to.position[2] - from.position[2]),
        0.0,
    ];
    let hit_radius = radius(params, position);
    (hit_radius >= trace.disk_inner_radius && hit_radius <= trace.disk_outer_radius)
        .then_some((position, hit_radius, fraction))
}

fn scalar_transition(params: &KerrParams, trace: &TraceConfig, lane: Lane) -> Expected {
    let unresolved = |ray| Expected {
        ray,
        active: 1,
        capture: 0,
        disk: 0,
        escape: 0,
        disk_position: [0.0; 3],
        disk_radius: 0.0,
        disk_fraction: 0.0,
    };
    if lane.active == 0 {
        let mut expected = unresolved(lane.ray);
        expected.active = 0;
        return expected;
    }

    let current_radius = radius(params, lane.ray.spatial_position());
    if current_radius <= horizon_radius(params) {
        let mut expected = unresolved(lane.ray);
        expected.active = 0;
        expected.capture = 1;
        return expected;
    }
    if current_radius >= trace.escape_radius && outward(params, &lane.ray) {
        let mut expected = unresolved(lane.ray);
        expected.active = 0;
        expected.escape = 1;
        return expected;
    }

    let previous = lane.ray;
    let mut candidate = lane.ray;
    midpoint_step(
        params,
        &mut candidate,
        trace.step_size,
        trace.gradient_epsilon,
    );
    let next_radius = radius(params, candidate.spatial_position());
    if next_radius <= horizon_radius(params) {
        let mut expected = unresolved(candidate);
        expected.active = 0;
        expected.capture = 1;
        return expected;
    }
    if let Some((position, hit_radius, fraction)) = disk_hit(params, trace, &previous, &candidate) {
        return Expected {
            ray: candidate,
            active: 0,
            capture: 0,
            disk: 1,
            escape: 0,
            disk_position: position,
            disk_radius: hit_radius,
            disk_fraction: fraction,
        };
    }
    if next_radius >= trace.escape_radius && outward(params, &candidate) {
        let mut expected = unresolved(candidate);
        expected.active = 0;
        expected.escape = 1;
        return expected;
    }
    unresolved(candidate)
}

fn execute_axtosa(graph: &Graph<'_>, inputs: &InputPlanes) -> (Vec<TensorId>, Vec<Vec<u8>>) {
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

    let mut input_ids = lowered.inputs.as_slice().to_vec();
    input_ids.sort_by_key(|id| id.0);
    assert_eq!(input_ids, (0..9).map(TensorId).collect::<Vec<_>>());
    let mut output_ids = lowered.outputs.as_slice().to_vec();
    output_ids.sort_by_key(|id| id.0);
    assert_eq!(output_ids, (9..26).map(TensorId).collect::<Vec<_>>());

    let input_refs: Vec<_> = lowered
        .inputs
        .iter()
        .map(|&tensor| inputs.for_tensor(tensor))
        .collect();
    let outputs = axtosa::execute(&lowered.bytes, &input_refs).unwrap();
    (lowered.outputs.into_iter().collect(), outputs)
}

fn output<'a>(ids: &[TensorId], outputs: &'a [Vec<u8>], tensor: TensorId) -> &'a [u8] {
    let slot = ids
        .iter()
        .position(|candidate| *candidate == tensor)
        .unwrap();
    &outputs[slot]
}

fn assert_close(label: &str, lane: usize, actual: f32, expected: f32, tolerance: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label} lane {lane}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
    );
}

fn assert_numeric_outputs(
    initial: &[Lane],
    expected: &[Expected],
    ids: &[TensorId],
    outputs: &[Vec<u8>],
) {
    let position = [abi::X_T_OUT, abi::X_X_OUT, abi::X_Y_OUT, abi::X_Z_OUT]
        .map(|id| decode_f32(output(ids, outputs, id)));
    let momentum = [abi::P_T_OUT, abi::P_X_OUT, abi::P_Y_OUT, abi::P_Z_OUT]
        .map(|id| decode_f32(output(ids, outputs, id)));
    let active = output(ids, outputs, abi::ACTIVE_OUT);
    let capture = output(ids, outputs, abi::CAPTURE_OUT);
    let disk = output(ids, outputs, abi::DISK_OUT);
    let escape = output(ids, outputs, abi::ESCAPE_OUT);
    let disk_x = decode_f32(output(ids, outputs, abi::DISK_X_OUT));
    let disk_y = decode_f32(output(ids, outputs, abi::DISK_Y_OUT));
    let disk_z = decode_f32(output(ids, outputs, abi::DISK_Z_OUT));
    let disk_radius = decode_f32(output(ids, outputs, abi::DISK_RADIUS_OUT));
    let disk_fraction = decode_f32(output(ids, outputs, abi::DISK_FRACTION_OUT));

    for lane in 0..initial.len() {
        assert_eq!(active[lane], expected[lane].active, "active lane {lane}");
        assert_eq!(capture[lane], expected[lane].capture, "capture lane {lane}");
        assert_eq!(disk[lane], expected[lane].disk, "disk lane {lane}");
        assert_eq!(escape[lane], expected[lane].escape, "escape lane {lane}");
        assert!(
            u16::from(active[lane])
                + u16::from(capture[lane])
                + u16::from(disk[lane])
                + u16::from(escape[lane])
                <= 1,
            "event masks overlap on lane {lane}"
        );

        for component in 0..4 {
            assert_close(
                "position",
                lane,
                position[component][lane],
                expected[lane].ray.position[component],
                POSITION_TOLERANCE,
            );
            assert_close(
                "momentum",
                lane,
                momentum[component][lane],
                expected[lane].ray.momentum[component],
                MOMENTUM_TOLERANCE,
            );
        }
        assert_eq!(
            momentum[0][lane].to_bits(),
            initial[lane].ray.momentum[0].to_bits(),
            "p_t changed bits on lane {lane}"
        );

        if expected[lane].active == 0 && expected[lane].ray == initial[lane].ray {
            for component in 0..4 {
                assert_eq!(
                    position[component][lane].to_bits(),
                    initial[lane].ray.position[component].to_bits(),
                    "pre-terminal position was not frozen on lane {lane} component {component}"
                );
                assert_eq!(
                    momentum[component][lane].to_bits(),
                    initial[lane].ray.momentum[component].to_bits(),
                    "pre-terminal momentum was not frozen on lane {lane} component {component}"
                );
            }
        }

        if expected[lane].disk != 0 {
            assert_close(
                "disk x",
                lane,
                disk_x[lane],
                expected[lane].disk_position[0],
                HIT_TOLERANCE,
            );
            assert_close(
                "disk y",
                lane,
                disk_y[lane],
                expected[lane].disk_position[1],
                HIT_TOLERANCE,
            );
            assert_close("disk z", lane, disk_z[lane], 0.0, 0.0);
            assert_close(
                "disk radius",
                lane,
                disk_radius[lane],
                expected[lane].disk_radius,
                HIT_TOLERANCE,
            );
            assert_close(
                "disk fraction",
                lane,
                disk_fraction[lane],
                expected[lane].disk_fraction,
                HIT_TOLERANCE,
            );
        } else {
            assert_eq!(disk_x[lane].to_bits(), 0.0_f32.to_bits());
            assert_eq!(disk_y[lane].to_bits(), 0.0_f32.to_bits());
            assert_eq!(disk_z[lane].to_bits(), 0.0_f32.to_bits());
            assert_eq!(disk_radius[lane].to_bits(), 0.0_f32.to_bits());
            assert_eq!(disk_fraction[lane].to_bits(), 0.0_f32.to_bits());
        }
    }
}

#[test]
fn kerr_step_matches_scalar_events_and_payloads_through_direct_tosa() {
    let params = KerrParams::REFERENCE;
    let trace = TraceConfig::REFERENCE;
    let lanes = lanes();
    let expected = lanes.map(|lane| scalar_transition(&params, &trace, lane));
    assert_eq!(
        expected.map(|lane| (lane.active, lane.capture, lane.disk, lane.escape)),
        [
            (1, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1),
        ],
        "the lane fixture stopped covering the intended event classes"
    );

    let graph_bytes = build_kerr_step_graph_in(
        KerrStepGraphSpec {
            rays: NUMERIC_RAYS as u32,
            params,
            trace,
            termination_radius: horizon_radius(&params),
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    let inputs = InputPlanes::from_lanes(&lanes);
    let (ids, outputs) = execute_axtosa(&graph, &inputs);
    assert_numeric_outputs(&lanes, &expected, &ids, &outputs);
}

#[test]
fn kerr_step_forms_one_region_under_the_realistic_device_model() {
    let graph_bytes = build_kerr_step_graph_in(
        KerrStepGraphSpec {
            rays: PLACEMENT_RAYS,
            params: KerrParams::REFERENCE,
            trace: TraceConfig::REFERENCE,
            termination_radius: horizon_radius(&KerrParams::REFERENCE),
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let executable = runtime
        .compile_in(&graph, &AxDeviceModel::default(), Global)
        .unwrap();
    let placement = executable.placement_report();
    assert!(
        placement.nodes_reassigned > 0,
        "the core optimized-placement entry point must recover the launch-amortized Kerr region: {placement:?}"
    );
    assert_eq!(placement.aggregate_cost, executable.aggregate_cost());
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND),
        "the 8K Kerr transition should amortize one realistic accelerator launch"
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
#[test]
fn kerr_step_matches_openvino_when_bool_boundaries_are_available() {
    use virtio_accel_openvino::{
        InitError, OPENVINO_TOSA_CAPABILITY, OPENVINO_TOSA_TARGET, OpenVinoAccelerator,
    };
    use virtio_accel_tosa::{DType as TosaDType, ValueRoles};

    if !OPENVINO_TOSA_CAPABILITY
        .supports_dtype(TosaDType::BOOL, ValueRoles::INPUT.union(ValueRoles::OUTPUT))
    {
        eprintln!("skipping: installed OpenVINO provider cannot expose BOOL model boundaries");
        return;
    }
    let provider = match OpenVinoAccelerator::new() {
        Ok(provider) => provider,
        Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
            eprintln!("skipping: no OpenVINO inference device on this host");
            return;
        }
        Err(error) => panic!("OpenVINO init failed: {error:?}"),
    };

    let params = KerrParams::REFERENCE;
    let trace = TraceConfig::REFERENCE;
    let lanes = lanes();
    let expected = lanes.map(|lane| scalar_transition(&params, &trace, lane));
    let graph_bytes = build_kerr_step_graph_in(
        KerrStepGraphSpec {
            rays: NUMERIC_RAYS as u32,
            params,
            trace,
            termination_radius: horizon_radius(&params),
        },
        Global,
    )
    .unwrap();
    let graph = graph_bytes.parse().unwrap();
    let inputs = InputPlanes::from_lanes(&lanes);

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
    let outputs = runtime
        .run_in(&executable, &inputs.abi_order(), Global)
        .unwrap();
    assert_numeric_outputs(&lanes, &expected, executable.output_ids(), &outputs);
}
