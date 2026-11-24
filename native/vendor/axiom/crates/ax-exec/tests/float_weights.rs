#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, NativeProgram, NativeRef};
use axnn_cpu::CpuCapabilities;
use axnn_cpu::fp8::{e4m3_to_f32, f32_to_e4m3_rne};
use std::alloc::Global;

const VOCAB: usize = 6;
const HIDDEN: usize = 40;
const INTERMEDIATE: usize = 24;
const SCALE: f32 = 0.0625;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 29 + seed * 13) % 37) as f32 - 18.0) / 6.0)
        .collect()
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn from_bf16(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&p| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(p)) << 16)))
        .collect()
}

/// A checkpoint-shaped slice of a decoder: a BF16 embedding, an FP8 linear
/// layer with its per-tensor weight scale, and a BF16 output head. Weights
/// stay in their stored `[out, in]` layout.
fn graph(embedding: &[u8], up: &[u8], head: &[u8]) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let token = b.tensor(DType::I32, TensorFlags::INPUT, &[1], &[]).unwrap();
    let dims = |a: usize, c: usize| [a as u32, c as u32];
    let embedding = b
        .tensor(
            DType::B16,
            TensorFlags::CONSTANT,
            &dims(VOCAB, HIDDEN),
            embedding,
        )
        .unwrap();
    let up = b
        .tensor(
            DType::B8,
            TensorFlags::CONSTANT,
            &dims(INTERMEDIATE, HIDDEN),
            up,
        )
        .unwrap();
    let scale = b
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &SCALE.to_le_bytes(),
        )
        .unwrap();
    let head = b
        .tensor(
            DType::B16,
            TensorFlags::CONSTANT,
            &dims(VOCAB, INTERMEDIATE),
            head,
        )
        .unwrap();
    let f32_tensor = |b: &mut GraphBuilder<Global>, width: usize, flags| {
        b.tensor(DType::F32, flags, &dims(1, width), &[]).unwrap()
    };
    let hidden = f32_tensor(&mut b, HIDDEN, TensorFlags::default());
    let unscaled = f32_tensor(&mut b, INTERMEDIATE, TensorFlags::default());
    let projected = f32_tensor(&mut b, INTERMEDIATE, TensorFlags::OUTPUT);
    let logits = f32_tensor(&mut b, VOCAB, TensorFlags::OUTPUT);
    let none = NodeFlags::default();
    b.node(OpCode::GATHER, none, &[embedding, token], &[hidden], &[])
        .unwrap();
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        none,
        &[hidden, up],
        &[unscaled],
        &[],
    )
    .unwrap();
    b.node(OpCode::MUL, none, &[unscaled, scale], &[projected], &[])
        .unwrap();
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        none,
        &[projected, head],
        &[logits],
        &[],
    )
    .unwrap();
    b.finish().unwrap()
}

#[test]
fn fp8_and_bf16_weights_run_in_checkpoint_layout() {
    let embedding = bf16(&values(VOCAB * HIDDEN, 1));
    let up: Vec<u8> = values(INTERMEDIATE * HIDDEN, 2)
        .into_iter()
        .map(f32_to_e4m3_rne)
        .collect();
    let head = bf16(&values(VOCAB * INTERMEDIATE, 3));
    let program = NativeProgram::from_graph_in(
        graph(&embedding, &up, &head),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();

    let embedding_f64 = from_bf16(&embedding);
    let head_f64 = from_bf16(&head);
    for token in [4_i32, 0] {
        arena
            .run_with_native_inputs_and_resources(&[NativeRef::I32(&[token])], &[])
            .unwrap();
        let hidden = &embedding_f64[token as usize * HIDDEN..][..HIDDEN];
        let projected: Vec<f64> = (0..INTERMEDIATE)
            .map(|o| {
                let dot: f64 = (0..HIDDEN)
                    .map(|i| hidden[i] * f64::from(e4m3_to_f32(up[o * HIDDEN + i])))
                    .sum();
                dot * f64::from(SCALE)
            })
            .collect();
        let close = |actual: &[f32], expected: &[f64]| {
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(expected) {
                assert!(
                    (f64::from(*a) - e).abs() <= 1e-5 * (1.0 + e.abs()),
                    "{a} vs {e}"
                );
            }
        };
        close(arena.output_f32(0).unwrap(), &projected);
        let logits: Vec<f64> = (0..VOCAB)
            .map(|v| {
                (0..INTERMEDIATE)
                    .map(|o| projected[o] * head_f64[v * INTERMEDIATE + o])
                    .sum()
            })
            .collect();
        close(arena.output_f32(1).unwrap(), &logits);
    }
}

/// A router kept in F32, as the published Nemotron-H checkpoint keeps it:
/// `[2, HIDDEN] x [ROUTED, HIDDEN]^T` with the F32 weights a graph input.
const ROUTED: usize = 12;

fn router_graph() -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let dims = |a: usize, c: usize| [a as u32, c as u32];
    let hidden = b
        .tensor(DType::F32, TensorFlags::INPUT, &dims(2, HIDDEN), &[])
        .unwrap();
    let router = b
        .tensor(DType::F32, TensorFlags::INPUT, &dims(ROUTED, HIDDEN), &[])
        .unwrap();
    let logits = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &dims(2, ROUTED), &[])
        .unwrap();
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        NodeFlags::default(),
        &[hidden, router],
        &[logits],
        &[],
    )
    .unwrap();
    b.finish().unwrap()
}

/// F32 weights are bound in place, on the scalar and SIMD kernels.
#[test]
fn f32_weights_run_in_checkpoint_layout() {
    let x = values(2 * HIDDEN, 4);
    let w = values(ROUTED * HIDDEN, 5);
    let expected: Vec<f64> = x
        .as_chunks::<HIDDEN>()
        .0
        .iter()
        .flat_map(|x| {
            w.as_chunks::<HIDDEN>().0.iter().map(move |w| {
                x.iter()
                    .zip(w)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
            })
        })
        .collect();
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
        let program = NativeProgram::from_graph_in(
            router_graph(),
            AxCpuKernelPolicy {
                capabilities,
                ..AxCpuKernelPolicy::default()
            },
            true,
            Global,
        )
        .unwrap();
        let mut arena = program.prepare_borrowing_inputs_in(Global).unwrap();
        arena
            .run_with_native_inputs_and_resources(&[NativeRef::F32(&x), NativeRef::F32(&w)], &[])
            .unwrap();
        for (a, e) in arena.output_f32(0).unwrap().iter().zip(&expected) {
            assert!(
                (f64::from(*a) - e).abs() <= 1e-5 * (1.0 + e.abs()),
                "{a} vs {e}"
            );
        }
    }
}
