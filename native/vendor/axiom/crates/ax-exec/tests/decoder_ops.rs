#![feature(allocator_api)]
use ax::tensor_ops::{encode_rms_norm_attributes, encode_top_k_attributes, encode_view_attributes};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeProgress, NativeRef};
use std::alloc::Global;

const ROWS: usize = 4;
const WIDTH: usize = 64;
const HALF: usize = WIDTH / 2;
const EPSILON: f32 = 1.0e-5;

/// Q8 rows with unit scales, so every entry decodes to its exact integer.
fn table() -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut values = Vec::new();
    for row in 0..ROWS {
        for block in 0..WIDTH / 32 {
            bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
            for i in 0..32 {
                let value = ((block * 32 + i) * 7 + row * 13) % 61;
                let value = value as i8 - 30;
                bytes.push(value as u8);
                values.push(f32::from(value));
            }
        }
    }
    (bytes, values)
}

fn weight() -> Vec<f32> {
    (0..WIDTH).map(|i| 0.5 + i as f32 / 64.0).collect()
}

/// Embed a token, normalize, view it as a gate and an up half, gate with
/// SiLU, and pick the three largest gated features: the stateless ops of a
/// decoder step between its matrix products.
fn fragment() -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let token = b.tensor(DType::I32, TensorFlags::INPUT, &[1], &[]).unwrap();
    let table = b
        .tensor(
            DType::Q8,
            TensorFlags::INPUT,
            &[ROWS as u32, WIDTH as u32],
            &[],
        )
        .unwrap();
    let weight_bytes: Vec<u8> = weight().iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight = b
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[WIDTH as u32],
            &weight_bytes,
        )
        .unwrap();
    let row = [1, WIDTH as u32];
    let half = [1, HALF as u32];
    let embedded = b
        .tensor(DType::F32, TensorFlags::default(), &row, &[])
        .unwrap();
    let normalized = b
        .tensor(DType::F32, TensorFlags::default(), &row, &[])
        .unwrap();
    let gate = b
        .tensor(DType::F32, TensorFlags::default(), &half, &[])
        .unwrap();
    let up = b
        .tensor(DType::F32, TensorFlags::default(), &half, &[])
        .unwrap();
    let activated = b
        .tensor(DType::F32, TensorFlags::default(), &half, &[])
        .unwrap();
    let gated = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &half, &[])
        .unwrap();
    let top_values = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 3], &[])
        .unwrap();
    let top_indices = b
        .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 3], &[])
        .unwrap();
    let none = NodeFlags::default();
    b.node(OpCode::GATHER, none, &[table, token], &[embedded], &[])
        .unwrap();
    b.node(
        OpCode::RMS_NORM,
        none,
        &[embedded, weight],
        &[normalized],
        &encode_rms_norm_attributes(EPSILON, -1).unwrap(),
    )
    .unwrap();
    // Gate and up are views of the normalized row: no bytes move.
    for (offset, output) in [(0, gate), (HALF as u64, up)] {
        b.node(
            OpCode::VIEW,
            none,
            &[normalized],
            &[output],
            &encode_view_attributes(offset),
        )
        .unwrap();
    }
    b.node(OpCode::SILU, none, &[gate], &[activated], &[])
        .unwrap();
    b.node(OpCode::MUL, none, &[activated, up], &[gated], &[])
        .unwrap();
    b.node(
        OpCode::TOP_K,
        none,
        &[gated],
        &[top_values, top_indices],
        &encode_top_k_attributes(3).unwrap(),
    )
    .unwrap();
    b.finish().unwrap()
}

fn reference(token: usize) -> Vec<f64> {
    let (_, values) = table();
    let row: Vec<f64> = values[token * WIDTH..(token + 1) * WIDTH]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let rms = (row.iter().map(|v| v * v).sum::<f64>() / WIDTH as f64 + f64::from(EPSILON)).sqrt();
    let normalized: Vec<f64> = row
        .iter()
        .zip(weight())
        .map(|(v, w)| v / rms * f64::from(w))
        .collect();
    (0..HALF)
        .map(|j| {
            let gate = normalized[j];
            gate / (1.0 + (-gate).exp()) * normalized[HALF + j]
        })
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn a_decoder_fragment_runs_natively_and_matches_a_direct_computation() {
    let graph = fragment();
    assert_eq!(&graph.as_bytes()[6..8], &5_u16.to_le_bytes());
    let program =
        NativeProgram::from_graph_in(graph, AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let (packed, _) = table();
    for token in [2_i32, 0, 3] {
        let token_bytes = token.to_le_bytes();
        let mut run = program.begin_in(&[&token_bytes, &packed], Global).unwrap();
        assert_eq!(run.advance(usize::MAX).unwrap(), NativeProgress::Ready);
        let outputs = run.take_retained_in().unwrap();
        let gated = f32s(outputs[0].as_bytes());
        for (actual, expected) in gated.iter().zip(reference(token as usize)) {
            assert!(
                (f64::from(*actual) - expected).abs() <= 1.0e-5 * (1.0 + expected.abs()),
                "token {token}: {actual} vs {expected}"
            );
        }
        // Top-k of the graph's own gated row: descending, ties to lower index.
        let mut order: Vec<usize> = (0..HALF).collect();
        order.sort_by(|&a, &b| gated[b].total_cmp(&gated[a]).then(a.cmp(&b)));
        let indices: Vec<i32> = outputs[2]
            .as_bytes()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            indices,
            order[..3].iter().map(|&i| i as i32).collect::<Vec<_>>()
        );
        assert_eq!(
            f32s(outputs[1].as_bytes()),
            order[..3].iter().map(|&i| gated[i]).collect::<Vec<_>>()
        );

        // The prepared arena runs the same instructions.
        arena
            .run_with_native_inputs_and_resources(
                &[
                    NativeRef::I32(&[token]),
                    NativeRef::Packed(DType::Q8, &packed),
                ],
                &[],
            )
            .unwrap();
        assert_eq!(arena.output_f32(0).unwrap(), gated);
        assert_eq!(arena.output_i32(2).unwrap(), indices);
    }

    // An out-of-range token fails without publishing any output.
    let bad = (ROWS as i32).to_le_bytes();
    assert!(matches!(
        arena.run(&[&bad, &packed]),
        Err(Error::Cpu(axnn_cpu::CpuError::ShapeMismatch { .. }))
    ));
    assert!(arena.output_f32(0).is_none());
}
