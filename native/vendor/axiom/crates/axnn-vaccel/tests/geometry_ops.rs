#![feature(allocator_api)]

use std::alloc::Global;

use ax::{
    DENSE_IMPLEMENTATION, DType, GraphBuilder, NodeFlags, OpCode, Partition, Schedule, TensorFlags,
    TensorId,
};
use axnn_vaccel::{ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, Session, artifact};

const SHAPE: [u32; 1] = [4];

fn tensor(builder: &mut GraphBuilder<Global>, dtype: DType, flags: TensorFlags) -> TensorId {
    builder.tensor(dtype, flags, &SHAPE, &[]).unwrap()
}

fn unary(
    builder: &mut GraphBuilder<Global>,
    op: OpCode,
    input: TensorId,
    dtype: DType,
) -> TensorId {
    let output = tensor(builder, dtype, TensorFlags::OUTPUT);
    builder
        .node(op, NodeFlags::default(), &[input], &[output], &[])
        .unwrap();
    output
}

fn binary(
    builder: &mut GraphBuilder<Global>,
    op: OpCode,
    left: TensorId,
    right: TensorId,
    dtype: DType,
) -> TensorId {
    let output = tensor(builder, dtype, TensorFlags::OUTPUT);
    builder
        .node(op, NodeFlags::default(), &[left, right], &[output], &[])
        .unwrap();
    output
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_output(outputs: &[Vec<u8>], ids: &[TensorId], id: TensorId) -> Vec<f32> {
    let slot = ids.iter().position(|candidate| *candidate == id).unwrap();
    outputs[slot]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn bool_output<'a>(outputs: &'a [Vec<u8>], ids: &[TensorId], id: TensorId) -> &'a [u8] {
    let slot = ids.iter().position(|candidate| *candidate == id).unwrap();
    &outputs[slot]
}

#[test]
fn geometric_standard_ops_roundtrip_and_execute_as_one_partition() {
    let mut builder = GraphBuilder::new_in(Global);
    let x = tensor(&mut builder, DType::F32, TensorFlags::INPUT);
    let y = tensor(&mut builder, DType::F32, TensorFlags::INPUT);

    let mul = binary(&mut builder, OpCode::MUL, x, y, DType::F32);
    let pow = binary(&mut builder, OpCode::POW, x, y, DType::F32);
    let maximum = binary(&mut builder, OpCode::MAXIMUM, x, y, DType::F32);
    let minimum = binary(&mut builder, OpCode::MINIMUM, x, y, DType::F32);
    let negate = unary(&mut builder, OpCode::NEGATE, x, DType::F32);
    let absolute = unary(&mut builder, OpCode::ABS, negate, DType::F32);
    let cosine = unary(&mut builder, OpCode::COS, x, DType::F32);
    let exponential = unary(&mut builder, OpCode::EXP, x, DType::F32);
    let logarithm = unary(&mut builder, OpCode::LOG, x, DType::F32);
    let reciprocal = unary(&mut builder, OpCode::RECIPROCAL, x, DType::F32);
    let reciprocal_root = unary(&mut builder, OpCode::RSQRT, x, DType::F32);
    let sine = unary(&mut builder, OpCode::SIN, x, DType::F32);

    let equal = binary(&mut builder, OpCode::EQUAL, x, y, DType::U8);
    let greater = binary(&mut builder, OpCode::GREATER, x, y, DType::U8);
    let greater_equal = binary(&mut builder, OpCode::GREATER_EQUAL, x, y, DType::U8);
    let logical_and = binary(
        &mut builder,
        OpCode::LOGICAL_AND,
        greater,
        greater_equal,
        DType::U8,
    );
    let logical_or = binary(&mut builder, OpCode::LOGICAL_OR, equal, greater, DType::U8);
    let logical_xor = binary(
        &mut builder,
        OpCode::LOGICAL_XOR,
        greater_equal,
        equal,
        DType::U8,
    );
    let logical_not = unary(&mut builder, OpCode::LOGICAL_NOT, equal, DType::U8);
    let selected = tensor(&mut builder, DType::F32, TensorFlags::OUTPUT);
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[logical_and, maximum, minimum],
            &[selected],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();

    let analysis = Schedule::analyze_in(&graph, Global).unwrap();
    let partition = Partition {
        backend: ACCEL_BACKEND,
        implementation: DENSE_IMPLEMENTATION,
        order_start: 0,
        node_count: graph.node_count() as u32,
    };
    let artifact = artifact::ax_partition_in(
        &graph,
        analysis.order(),
        analysis.producers(),
        partition,
        Global,
    )
    .unwrap();

    let x_values = [0.25_f32, 0.5, 1.0, 2.0];
    let y_values = [0.5_f32, 0.25, 1.0, 1.5];
    let x_bytes = f32_bytes(&x_values);
    let y_bytes = f32_bytes(&y_values);
    let expected = axtosa::execute(&artifact.bytes, &[&x_bytes, &y_bytes]).unwrap();

    // The standard-op subset is bidirectional. Provider TOSA can be ingested
    // back into `.ax` without losing any geometry operation.
    let roundtrip = axtosa::lower_ax_in(&artifact.bytes, Global).unwrap();
    let roundtrip = roundtrip.parse().unwrap();
    let mut original_ops: Vec<_> = (0..graph.node_count())
        .map(|index| graph.node(index).unwrap().op.0)
        .collect();
    let mut roundtrip_ops: Vec<_> = (0..roundtrip.node_count())
        .map(|index| roundtrip.node(index).unwrap().op.0)
        .collect();
    original_ops.sort_unstable();
    roundtrip_ops.sort_unstable();
    assert_eq!(roundtrip_ops, original_ops);

    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
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
            .all(|backend| *backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    let actual = runtime
        .run_in(&executable, &[&x_bytes, &y_bytes], Global)
        .unwrap();
    let expected_runtime_order: Vec<_> = executable
        .output_ids()
        .iter()
        .map(|id| {
            let slot = artifact
                .outputs
                .iter()
                .position(|candidate| candidate == id)
                .unwrap();
            expected[slot].clone()
        })
        .collect();
    assert_eq!(actual, expected_runtime_order);

    let ids = artifact.outputs.as_slice();
    let assert_close = |id, want: &[f32], tolerance: f32| {
        for (lane, (got, want)) in f32_output(&expected, ids, id)
            .into_iter()
            .zip(want.iter().copied())
            .enumerate()
        {
            assert!(
                (got - want).abs() <= tolerance,
                "tensor {} lane {lane}: {got} vs {want}",
                id.0
            );
        }
    };
    assert_close(
        mul,
        &std::array::from_fn::<_, 4, _>(|i| x_values[i] * y_values[i]),
        1.0e-6,
    );
    assert_close(absolute, &x_values, 1.0e-6);
    assert_close(
        reciprocal_root,
        &std::array::from_fn::<_, 4, _>(|i| 1.0 / x_values[i].sqrt()),
        1.0e-6,
    );
    assert_eq!(bool_output(&expected, ids, equal), &[0, 0, 1, 0]);
    assert_eq!(bool_output(&expected, ids, greater), &[0, 1, 0, 1]);
    assert_eq!(bool_output(&expected, ids, greater_equal), &[0, 1, 1, 1]);
    assert_eq!(bool_output(&expected, ids, logical_and), &[0, 1, 0, 1]);
    assert_eq!(bool_output(&expected, ids, logical_or), &[0, 1, 1, 1]);
    assert_eq!(bool_output(&expected, ids, logical_xor), &[0, 1, 0, 1]);
    assert_eq!(bool_output(&expected, ids, logical_not), &[1, 1, 0, 1]);
    assert_close(selected, &[0.25, 0.5, 1.0, 2.0], 1.0e-6);

    // Keep every constructed result live so the test locks the complete set.
    assert!(artifact.outputs.contains(&pow));
    assert!(artifact.outputs.contains(&cosine));
    assert!(artifact.outputs.contains(&exponential));
    assert!(artifact.outputs.contains(&logarithm));
    assert!(artifact.outputs.contains(&reciprocal));
    assert!(artifact.outputs.contains(&sine));
}

#[test]
fn guarded_distance_root_fuses_without_leaving_the_portable_graph_path() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = tensor(&mut builder, DType::F32, TensorFlags::INPUT);
    let floor = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &f32::EPSILON.to_le_bytes(),
        )
        .unwrap();
    let guarded = tensor(&mut builder, DType::F32, TensorFlags::default());
    builder
        .node(
            OpCode::MAXIMUM,
            NodeFlags::default(),
            &[input, floor],
            &[guarded],
            &[],
        )
        .unwrap();
    let inverse = tensor(&mut builder, DType::F32, TensorFlags::default());
    builder
        .node(
            OpCode::RSQRT,
            NodeFlags::default(),
            &[guarded],
            &[inverse],
            &[],
        )
        .unwrap();
    let output = tensor(&mut builder, DType::F32, TensorFlags::OUTPUT);
    // Keep the inverse first to lock commutative matching independently of
    // whichever operand order a graph emitter happens to choose.
    builder
        .node(
            OpCode::MUL,
            NodeFlags::default(),
            &[inverse, input],
            &[output],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let analysis = Schedule::analyze_in(&graph, Global).unwrap();
    let lowered = artifact::ax_partition_in(
        &graph,
        analysis.order(),
        analysis.producers(),
        Partition {
            backend: ACCEL_BACKEND,
            implementation: DENSE_IMPLEMENTATION,
            order_start: 0,
            node_count: graph.node_count() as u32,
        },
        Global,
    )
    .unwrap();

    let program = axtosa::Program::compile(&lowered.bytes).unwrap();
    assert_eq!(program.stats().guarded_sqrt_fusions, 1);
    let input_bytes = f32_bytes(&[0.0, 1.0, 4.0, 9.0]);
    let outputs = program.execute(&[&input_bytes]).unwrap();
    assert_eq!(outputs.len(), 1);
    for (actual, expected) in outputs[0]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .zip([0.0, 1.0, 2.0, 3.0])
    {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn integer_negate_zero_points_are_not_silently_elided() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = tensor(&mut builder, DType::I8, TensorFlags::INPUT);
    let _output = unary(&mut builder, OpCode::NEGATE, input, DType::I8);
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let analysis = Schedule::analyze_in(&graph, Global).unwrap();
    let partition = Partition {
        backend: ACCEL_BACKEND,
        implementation: DENSE_IMPLEMENTATION,
        order_start: 0,
        node_count: 1,
    };
    let artifact = artifact::ax_partition_in(
        &graph,
        analysis.order(),
        analysis.producers(),
        partition,
        Global,
    )
    .unwrap();

    match axtosa::lower_ax_in(&artifact.bytes, Global) {
        Err(axtosa::AxLowerError::UnsupportedOperator(op)) => {
            assert_eq!(op, virtio_accel_tosa::Op::NEGATE);
        }
        _ => panic!("integer NEGATE must retain zero-point semantics or be rejected"),
    }
}
