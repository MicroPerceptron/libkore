#![feature(allocator_api)]

use std::alloc::Global;

use ax::{
    DENSE_IMPLEMENTATION, DType, GraphBuilder, LinearizedField2d, NodeFlags, OpCode, Partition,
    PremultipliedRgba, Schedule, TensorFlags, TensorId,
};
use axnn_vaccel::{ACCEL_BACKEND, artifact};

const DIMENSIONS: [u32; 1] = [4];

fn input(builder: &mut GraphBuilder<Global>) -> TensorId {
    builder
        .tensor(DType::F32, TensorFlags::INPUT, &DIMENSIONS, &[])
        .unwrap()
}

fn bytes(values: &[f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn binary(
    builder: &mut GraphBuilder<Global>,
    op: OpCode,
    left: TensorId,
    right: TensorId,
    dimensions: &[u32],
) -> TensorId {
    let output = builder
        .tensor(DType::F32, TensorFlags::default(), dimensions, &[])
        .unwrap();
    builder
        .node(op, NodeFlags::default(), &[left, right], &[output], &[])
        .unwrap();
    output
}

#[test]
fn premultiplied_compositor_region_fuses_and_executes_portably() {
    let mut builder = GraphBuilder::new_in(Global);
    let source = PremultipliedRgba {
        rgb: core::array::from_fn(|_| input(&mut builder)),
        alpha: input(&mut builder),
    };
    let destination = core::array::from_fn(|_| input(&mut builder));
    let composed = builder
        .premultiplied_source_over_opaque_f32(source, destination, &DIMENSIONS)
        .unwrap();
    let outputs = composed.map(|channel| {
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &DIMENSIONS, &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[channel],
                &[output],
                &[],
            )
            .unwrap();
        output
    });
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let schedule = Schedule::analyze_in(&graph, Global).unwrap();
    let lowered = artifact::ax_partition_in(
        &graph,
        schedule.order(),
        schedule.producers(),
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
    let stats = program.stats();
    assert_eq!(stats.mul_add_fusions, 3, "compositor optimizer: {stats:?}");

    let source_values = [
        [0.0, 0.125, 0.25, 0.5],
        [0.0, 0.0625, 0.125, 0.25],
        [0.0, 0.03125, 0.0625, 0.125],
    ];
    let alpha = [0.0, 0.25, 0.5, 1.0];
    let destination_values = [
        [0.8, 0.8, 0.8, 0.8],
        [0.6, 0.6, 0.6, 0.6],
        [0.4, 0.4, 0.4, 0.4],
    ];
    let mut bindings = Vec::new();
    for &id in &lowered.inputs {
        let values = source
            .rgb
            .iter()
            .position(|candidate| *candidate == id)
            .map(|channel| &source_values[channel])
            .or_else(|| (id == source.alpha).then_some(&alpha))
            .or_else(|| {
                destination
                    .iter()
                    .position(|candidate| *candidate == id)
                    .map(|channel| &destination_values[channel])
            })
            .expect("lowered compositor input belongs to the source or destination");
        bindings.push(bytes(values));
    }
    let references = bindings.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let actual = program.execute(&references).unwrap();
    for (channel, output) in outputs.into_iter().enumerate() {
        let slot = lowered
            .outputs
            .iter()
            .position(|candidate| *candidate == output)
            .unwrap();
        let actual = actual[slot]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        let expected = core::array::from_fn::<_, 4, _>(|pixel| {
            source_values[channel][pixel]
                + destination_values[channel][pixel] * (1.0 - alpha[pixel])
        });
        assert_eq!(actual, expected);
    }
}

#[test]
fn shallow_solid_layer_fuses_translation_clamp_and_colour_lerps() {
    let mut builder = GraphBuilder::new_in(Global);
    let alpha_input = input(&mut builder);
    let derivative_x = input(&mut builder);
    let derivative_y = input(&mut builder);
    let offset_x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let offset_y = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let translated = builder
        .linearized_translate_2d_f32(
            LinearizedField2d {
                value: alpha_input,
                derivative_x,
                derivative_y,
            },
            [offset_x, offset_y],
            &DIMENSIONS,
        )
        .unwrap();
    let zero = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &0.0_f32.to_le_bytes(),
        )
        .unwrap();
    let one = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &1.0_f32.to_le_bytes(),
        )
        .unwrap();
    let alpha = binary(&mut builder, OpCode::MAXIMUM, translated, zero, &DIMENSIONS);
    let alpha = binary(&mut builder, OpCode::MINIMUM, alpha, one, &DIMENSIONS);
    let destination = core::array::from_fn(|_| input(&mut builder));
    let colour = [0.2, 0.4, 0.6];
    let composed = builder
        .solid_source_over_opaque_f32(alpha, colour, destination, &DIMENSIONS)
        .unwrap();
    let outputs = composed.map(|channel| {
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &DIMENSIONS, &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[channel],
                &[output],
                &[],
            )
            .unwrap();
        output
    });
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let schedule = Schedule::analyze_in(&graph, Global).unwrap();
    let lowered = artifact::ax_partition_in(
        &graph,
        schedule.order(),
        schedule.producers(),
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
    let stats = program.stats();
    assert_eq!(
        stats.linearized_clamp_fusions, 1,
        "parallax optimizer: {stats:?}"
    );
    assert_eq!(stats.lerp_fusions, 3, "solid compositor: {stats:?}");

    let alpha_values = [0.0, 0.25, 0.75, 1.0];
    let derivative_x_values = [-0.5, 0.5, -0.25, 0.25];
    let derivative_y_values = [0.25, -0.25, 0.5, -0.5];
    let offset_x_value = [0.5_f32];
    let offset_y_value = [-0.25_f32];
    let destination_values = [
        [0.8, 0.8, 0.8, 0.8],
        [0.6, 0.6, 0.6, 0.6],
        [0.4, 0.4, 0.4, 0.4],
    ];
    let mut bindings = Vec::new();
    for &id in &lowered.inputs {
        let values: &[f32] = if id == alpha_input {
            &alpha_values
        } else if id == derivative_x {
            &derivative_x_values
        } else if id == derivative_y {
            &derivative_y_values
        } else if id == offset_x {
            &offset_x_value
        } else if id == offset_y {
            &offset_y_value
        } else {
            let channel = destination
                .iter()
                .position(|candidate| *candidate == id)
                .expect("lowered input belongs to the parallax layer");
            &destination_values[channel]
        };
        bindings.push(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
    }
    let references = bindings.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let actual = program.execute(&references).unwrap();
    let shifted_alpha = core::array::from_fn::<_, 4, _>(|pixel| {
        ((alpha_values[pixel] + derivative_x_values[pixel] * offset_x_value[0])
            + derivative_y_values[pixel] * offset_y_value[0])
            .clamp(0.0, 1.0)
    });
    for (channel, output) in outputs.into_iter().enumerate() {
        let slot = lowered
            .outputs
            .iter()
            .position(|candidate| *candidate == output)
            .unwrap();
        let actual = actual[slot]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        let expected = core::array::from_fn::<_, 4, _>(|pixel| {
            destination_values[channel][pixel]
                + (colour[channel] - destination_values[channel][pixel]) * shifted_alpha[pixel]
        });
        assert_eq!(actual, expected);
    }
}
