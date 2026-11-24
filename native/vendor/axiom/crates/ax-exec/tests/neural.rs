#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, NativeProgram, NativeProgress};
use axnn_cpu::CpuCapabilities;
use std::alloc::Global;

fn q8(rows: usize, width: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in 0..rows {
        for _ in 0..width / 32 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1
            bytes.extend((0..32).map(|i| ((i as i8 - 15) * (row as i8 + 1)) as u8));
        }
    }
    bytes
}

#[test]
fn packed_graph_preserves_bytes_and_reuses_prepared_storage() {
    let mut b = GraphBuilder::new_in(Global);
    let w = b
        .tensor(DType::Q8, TensorFlags::INPUT, &[2, 32], &[])
        .unwrap();
    let x = b
        .tensor(DType::Q8, TensorFlags::INPUT, &[3, 32], &[])
        .unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 3], &[])
        .unwrap();
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        NodeFlags::default(),
        &[w, x],
        &[y],
        &[],
    )
    .unwrap();
    let graph = b.finish().unwrap();
    assert_eq!(&graph.as_bytes()[6..8], &3u16.to_le_bytes());
    let mut downgraded = graph.as_bytes().to_vec();
    downgraded[6..8].copy_from_slice(&2u16.to_le_bytes());
    assert!(ax::Graph::parse(&downgraded).is_err());
    let w = q8(2, 32);
    let x = q8(3, 32);
    let dot: f32 = (0..32).map(|i| ((i - 15) * (i - 15)) as f32).sum();
    for caps in [CpuCapabilities::SCALAR, CpuCapabilities::compiled()] {
        let program = NativeProgram::compile_in(
            &graph.parse().unwrap(),
            AxCpuKernelPolicy {
                capabilities: caps,
                ..AxCpuKernelPolicy::default()
            },
            true,
            Global,
        )
        .unwrap();
        let mut run = program.begin_in(&[&w, &x], Global).unwrap();
        assert_eq!(run.advance(1).unwrap(), NativeProgress::Ready);
        assert_eq!(run.host_copy_stats().native_ingress_bytes, 0);
        let results = run.take_retained_in().unwrap();
        let actual: Vec<f32> = results[0]
            .as_bytes()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual,
            [dot, 2.0 * dot, 3.0 * dot, 2.0 * dot, 4.0 * dot, 6.0 * dot]
        );
        let mut arena = program.prepare_native_in(Global).unwrap();
        arena
            .run_with_native_inputs_and_resources(
                &[
                    ax_exec::NativeRef::Packed(DType::Q8, &w),
                    ax_exec::NativeRef::Packed(DType::Q8, &x),
                ],
                &[],
            )
            .unwrap();
        assert_eq!(arena.output_f32(0).unwrap(), actual);
        assert!(
            arena
                .run_with_native_inputs_and_resources(
                    &[
                        ax_exec::NativeRef::Packed(DType::Q4, &w),
                        ax_exec::NativeRef::Packed(DType::Q8, &x)
                    ],
                    &[]
                )
                .is_err()
        );
        assert!(arena.output_f32(0).is_none());
        for _ in 0..3 {
            arena.run(&[&w, &x]).unwrap();
            assert_eq!(arena.output_f32(0).unwrap(), actual);
        }
    }
}

#[test]
fn softmax_rows_and_full_tensor_normalization_execute_without_provider_fallback() {
    for op in [OpCode::SOFTMAX, OpCode::RMS_NORM, OpCode::LAYER_NORM] {
        let mut b = GraphBuilder::new_in(Global);
        let x = b
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 4], &[])
            .unwrap();
        let ones: Vec<u8> = [1.0f32; 8].into_iter().flat_map(f32::to_le_bytes).collect();
        let zeros = [0u8; 32];
        let w = b
            .tensor(DType::F32, TensorFlags::CONSTANT, &[2, 4], &ones)
            .unwrap();
        let bias = b
            .tensor(DType::F32, TensorFlags::CONSTANT, &[2, 4], &zeros)
            .unwrap();
        let y = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 4], &[])
            .unwrap();
        let inputs = match op {
            OpCode::SOFTMAX => vec![x],
            OpCode::RMS_NORM => vec![x, w],
            _ => vec![x, w, bias],
        };
        let attr = if op == OpCode::SOFTMAX {
            (-1i32).to_le_bytes()
        } else {
            1e-5f32.to_le_bytes()
        };
        b.node(op, NodeFlags::default(), &inputs, &[y], &attr)
            .unwrap();
        let graph = b.finish().unwrap();
        let program = NativeProgram::compile_auto_in(&graph.parse().unwrap(), Global).unwrap();
        let mut arena = program.prepare_native_in(Global).unwrap();
        let input: Vec<u8> = [1., 2., 3., 4., 5., 6., 7., 8.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        arena.run(&[&input]).unwrap();
        let out = arena.output_f32(0).unwrap();
        if op == OpCode::SOFTMAX {
            for row in out.chunks_exact(4) {
                assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
        } else if op == OpCode::RMS_NORM {
            assert!((out.iter().map(|x| x * x).sum::<f32>() / 8.0 - 1.0).abs() < 1e-5);
        } else {
            assert!(out.iter().sum::<f32>().abs() < 1e-5);
        }
    }
}

#[test]
fn authored_activation_quantization_feeds_packed_dot_and_dequantization() {
    for dtype in [DType::Q4, DType::Q6, DType::Q8] {
        let mut b = GraphBuilder::new_in(Global);
        let input = b
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 256], &[])
            .unwrap();
        let packed = b
            .tensor(dtype, TensorFlags::default(), &[1, 256], &[])
            .unwrap();
        let act = b
            .tensor(DType::Q8, TensorFlags::default(), &[1, 256], &[])
            .unwrap();
        let restored = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 256], &[])
            .unwrap();
        let dot = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1], &[])
            .unwrap();
        b.node(OpCode::CAST, NodeFlags::default(), &[input], &[packed], &[])
            .unwrap();
        b.node(OpCode::CAST, NodeFlags::default(), &[input], &[act], &[])
            .unwrap();
        b.node(
            OpCode::CAST,
            NodeFlags::default(),
            &[packed],
            &[restored],
            &[],
        )
        .unwrap();
        b.node(
            OpCode::MATMUL_TRANSPOSED,
            NodeFlags::default(),
            &[packed, act],
            &[dot],
            &[],
        )
        .unwrap();
        let graph = b.finish().unwrap();
        let p = NativeProgram::compile_auto_in(&graph.parse().unwrap(), Global).unwrap();
        let mut arena = p.prepare_native_in(Global).unwrap();
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.) / 128.).collect();
        let input: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        arena.run(&[&input]).unwrap();
        let restored = arena.output_f32(0).unwrap();
        assert!(
            restored
                .iter()
                .zip(&values)
                .all(|(a, b)| (a - b).abs() < 0.04)
        );
        let dot = arena.output_f32(1).unwrap()[0];
        let expected: f32 = values.iter().map(|x| x * x).sum();
        assert!((dot - expected).abs() / expected < 0.03);
    }
}
