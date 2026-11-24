//! The aftershock lighting job as one `.ax` graph, shared by examples:
//! `MATMUL → NEGATE → MAXIMUM → NEGATE → EXP → MATMUL` over `probes` Gaussian probes and
//! `lights` lights, every tensor at `dtype` (F32 or F16).

#![allow(dead_code)]

use std::alloc::Global;

use ax::{DType, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags};

pub fn encode(values: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F16 => values
            .iter()
            .flat_map(|v| (*v as f16).to_bits().to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn decode(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])) as f32)
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

pub fn lighting_graph(dtype: DType, probes: u32, lights: u32) -> GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let features: Vec<f32> = (0..probes)
        .flat_map(|i| {
            let x = (i % 24) as f32 / 23.0 - 0.5;
            let y = (i / 24 % 12) as f32 / 11.0 - 0.5;
            let z = (i / 288) as f32 / 23.0 - 0.5;
            [x, y, z, x * x + y * y + z * z, 1.0]
        })
        .collect();
    let q = b
        .tensor(
            dtype,
            TensorFlags::CONSTANT,
            &[1, probes, 5],
            &encode(&features, dtype),
        )
        .unwrap();
    let w = b
        .tensor(dtype, TensorFlags::INPUT, &[1, 5, lights], &[])
        .unwrap();
    let zero = b
        .tensor(
            dtype,
            TensorFlags::CONSTANT,
            &[1, 1, 1],
            &encode(&[0.0], dtype),
        )
        .unwrap();
    let mut plane = |flags| b.tensor(dtype, flags, &[1, probes, lights], &[]).unwrap();
    let distance = plane(TensorFlags::default());
    // `-max(-d, 0)` rather than `min(d, 0)`: the spelling Intel's NPU compiler
    // gets right (see aftershock's lighting graph).
    let flipped = plane(TensorFlags::default());
    let clamped = plane(TensorFlags::default());
    let negative = plane(TensorFlags::default());
    let falloff = plane(TensorFlags::default());
    let rgb = b
        .tensor(dtype, TensorFlags::INPUT, &[1, lights, 3], &[])
        .unwrap();
    let out = b
        .tensor(dtype, TensorFlags::OUTPUT, &[1, probes, 3], &[])
        .unwrap();
    for (op, input, output) in [
        (OpCode::MATMUL, vec![q, w], distance),
        (OpCode::NEGATE, vec![distance], flipped),
        (OpCode::MAXIMUM, vec![flipped, zero], clamped),
        (OpCode::NEGATE, vec![clamped], negative),
        (OpCode::EXP, vec![negative], falloff),
        (OpCode::MATMUL, vec![falloff, rgb], out),
    ] {
        b.node(op, NodeFlags::default(), &input, &[output], &[])
            .unwrap();
    }
    b.finish().unwrap()
}

/// Deterministic light rows and colours for `lights` lights.
pub fn lighting_inputs(lights: u32) -> (Vec<f32>, Vec<f32>) {
    let w = (0..5 * lights)
        .map(|i| ((i * 7) % 11) as f32 / 8.0 - 0.75)
        .collect();
    let rgb = (0..lights * 3)
        .map(|i| ((i * 5) % 9) as f32 / 8.0)
        .collect();
    (w, rgb)
}
