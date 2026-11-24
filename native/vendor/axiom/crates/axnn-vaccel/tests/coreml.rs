//! Heterogeneous cross-validation against the Core ML / ANE provider.
//!
//! Runs only with `--features coreml`, and skips (passing) on hosts
//! without a Neural Engine — same convention as virtio-accel's own
//! Core ML tests. Asserts the two halves of the routing story: FP32
//! results agree across CPU and ANE within accelerator precision, and
//! F8 artifacts are rejected by Core ML and transparently served by
//! the CPU session.
#![cfg(all(feature = "coreml", target_os = "macos"))]
#![feature(allocator_api)]
#![feature(f16)]

use std::alloc::Global;

use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, CPU_BACKEND, Hetero, MatMulDType,
    Route, Session, artifact,
};
use virtio_accel_coreml::{COREML_TOSA_TARGET, CoreMlAccelerator, InitError};

fn ane_session() -> Option<Session<CoreMlAccelerator>> {
    let provider = match CoreMlAccelerator::new_tosa() {
        Ok(provider) => provider,
        Err(InitError::NeuralEngineUnavailable) => {
            eprintln!("skipping: no Neural Engine on this host");
            return None;
        }
        Err(error) => panic!("Core ML init failed: {error:?}"),
    };
    Some(Session::new(provider, COREML_TOSA_TARGET).expect("coreml session"))
}

fn f32_plane(len: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut state = seed | 1;
    let mut bytes = Vec::with_capacity(len * 4);
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((state >> 33) as f32 / u32::MAX as f32) - 0.5;
        bytes.extend_from_slice(&v.to_le_bytes());
        values.push(v);
    }
    (bytes, values)
}

fn scheduled_mask_graph() -> ax::GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let shape = [4_u32];
    let x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let active = builder
        .tensor(DType::U8, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let zero = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &0.0_f32.to_le_bytes(),
        )
        .unwrap();
    let magnitude = builder
        .tensor(DType::F32, TensorFlags::default(), &shape, &[])
        .unwrap();
    let positive = builder
        .tensor(DType::U8, TensorFlags::default(), &shape, &[])
        .unwrap();
    let live = builder
        .tensor(DType::U8, TensorFlags::default(), &shape, &[])
        .unwrap();
    let selected = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &shape, &[])
        .unwrap();
    let live_out = builder
        .tensor(DType::U8, TensorFlags::OUTPUT, &shape, &[])
        .unwrap();
    for (op, inputs, output) in [
        (OpCode::COPY, vec![x], magnitude),
        (OpCode::GREATER, vec![magnitude, zero], positive),
        (OpCode::LOGICAL_AND, vec![active, positive], live),
        (OpCode::SELECT, vec![live, magnitude, zero], selected),
        (OpCode::LOGICAL_OR, vec![live, live], live_out),
    ] {
        builder
            .node(op, NodeFlags::default(), &inputs, &[output], &[])
            .unwrap();
    }
    builder.finish().unwrap()
}

fn f16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| (value as f16).to_bits().to_le_bytes())
        .collect()
}

fn scheduled_coreml_to_cpu_graph(width: u32) -> ax::GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, width], &[])
        .unwrap();
    let weights = f16_bytes((0..width as usize * width as usize).map(|index| {
        if index % (width as usize + 1) == 0 {
            1.0
        } else {
            0.0
        }
    }));
    let weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, width, width],
            &weights,
        )
        .unwrap();
    let shared = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, width], &[])
        .unwrap();
    let absolute = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, width], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, weights],
            &[shared],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ABS,
            NodeFlags::default(),
            &[shared],
            &[absolute],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

fn ane_and_cpu_agree_on_f32_and_fp8_falls_back() {
    let Some(ane) = ane_session() else { return };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut hetero = Hetero::new(cpu, Some(ane));

    // Large FP32 matmul routes to the ANE and agrees with the CPU
    // within accelerator precision (Core ML may compute in reduced
    // precision internally).
    let (m, k, n) = (64, 64, 64);
    let (a_bytes, a_vals) = f32_plane(m * k, 7);
    let (b_bytes, b_vals) = f32_plane(k * n, 11);
    let (route, out) = hetero
        .matmul_bytes(MatMulDType::F32, m, k, n, &a_bytes, &b_bytes)
        .unwrap();
    assert_eq!(route, Route::Accel, "large f32 should route to the ANE");

    let got: Vec<f32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_vals[i * k + p] * b_vals[p * n + j];
            }
            worst = worst.max((got[i * n + j] - acc).abs());
        }
    }
    assert!(worst < 0.05, "ANE f32 divergence too large: {worst}");

    // F8 artifacts: Core ML rejects them at load; the router pins the
    // dtype to the CPU session and still returns correct results.
    let a: Vec<u8> = (0..16 * 16).map(|i| (i as u8) & 0x3f).collect();
    let (route, out) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(route, Route::Cpu, "fp8 must fall back to the CPU provider");
    assert!(hetero.is_rejected(MatMulDType::B8));
    assert_eq!(out.len(), 16 * 16 * 2, "f16 output plane");

    // Second call short-circuits the probe (no repeated load attempts).
    let (route, _) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(route, Route::Cpu);
}

fn scheduled_boolean_graph_executes_as_one_coreml_partition() {
    let Some(ane) = ane_session() else { return };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(ane));
    let bytes = scheduled_mask_graph();
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..runtime.automatic_device_model()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [ACCEL_BACKEND; 5]);
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let x: Vec<u8> = [-2.0_f32, -1.0, 2.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let active = [1_u8, 1, 1, 0];
    let output = runtime.run_in(&executable, &[&x, &active], Global).unwrap();
    let expected: Vec<u8> = [0.0_f32, 0.0, 2.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(output[0], expected);
    assert_eq!(output[1], [0_u8, 0, 1, 0]);
    // The provider-discovered Boolean bridge must also stay compatible with
    // automatic selection of the resident feedback loop on real CoreML.
    let repeated = runtime
        .run_repeated_auto_in(
            &executable,
            &[&x, &active],
            2,
            &[
                axnn_vaccel::AxFeedback::new(executable.input_ids()[0], executable.output_ids()[0]),
                axnn_vaccel::AxFeedback::new(executable.input_ids()[1], executable.output_ids()[1]),
            ],
            Global,
        )
        .unwrap();
    assert_eq!(repeated, output);
}

fn automatically_scheduled_matmul_executes_on_coreml() {
    let Some(ane) = ane_session() else { return };
    // Large enough to justify transfer/launch overhead under the default
    // estimates. No backend preset, cost override or manual CPU mask.
    let (m, k, n) = (64, 128, 128);
    let source = artifact::matmul(MatMulDType::F16, m, k, n).unwrap();
    let ax = axtosa::lower_ax_in(&source, Global).unwrap();
    let graph = ax.parse().unwrap();
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(ane));
    let executable = runtime.compile_auto_in(&graph, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|backend| *backend == ACCEL_BACKEND)
    );
    let left = f16_bytes((0..m * k).map(|i| (i % 33) as f32 / 16.0 - 1.0));
    let right = f16_bytes((0..k * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }));
    let row_bytes = n * 2;
    let stride = row_bytes + 16;
    let mut output = vec![0xa5; 9 + m * stride];
    let mut expected = output.clone();
    for y in 0..m {
        expected[9 + y * stride..9 + y * stride + row_bytes]
            .copy_from_slice(&left[y * row_bytes..(y + 1) * row_bytes]);
    }
    runtime
        .run_into_in(
            &executable,
            &[&left, &right],
            &mut [axnn_vaccel::AxOutput::pitched(&mut output, 9, row_bytes, m, stride).unwrap()],
            Global,
        )
        .unwrap();
    assert_eq!(
        output, expected,
        "automatic CoreML dispatch and pitched readback"
    );
}

fn scheduled_rank2_broadcast_executes_on_coreml() {
    const HEIGHT: u32 = 64;
    const WIDTH: u32 = 64;
    let Some(ane) = ane_session() else { return };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(ane));
    let mut builder = GraphBuilder::new_in(Global);
    let image = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[HEIGHT, WIDTH], &[])
        .unwrap();
    let row = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1, WIDTH], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[HEIGHT, WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[image, row],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..runtime.automatic_device_model()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.single_backend(), Some(ACCEL_BACKEND));
    let image = vec![0.25_f32.to_le_bytes(); (HEIGHT * WIDTH) as usize]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let row = vec![0.5_f32.to_le_bytes(); WIDTH as usize]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let output = runtime
        .run_in(&executable, &[&image, &row], Global)
        .unwrap();
    assert!(
        output[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|bytes| f32::from_le_bytes(*bytes) == 0.75)
    );
    // Exercise actual CoreML ByteSink readback directly into a pitched image,
    // not just the retained byte-vector compatibility path above.
    let row_bytes = WIDTH as usize * 4;
    let stride = row_bytes + 16;
    let mut backing = vec![0xa5; 7 + HEIGHT as usize * stride];
    let mut expected = backing.clone();
    for y in 0..HEIGHT as usize {
        expected[7 + y * stride..7 + y * stride + row_bytes]
            .copy_from_slice(&output[0][y * row_bytes..(y + 1) * row_bytes]);
    }
    runtime
        .run_into_in(
            &executable,
            &[&image, &row],
            &mut [axnn_vaccel::AxOutput::pitched(
                &mut backing,
                7,
                row_bytes,
                HEIGHT as usize,
                stride,
            )
            .unwrap()],
            Global,
        )
        .unwrap();
    assert_eq!(backing, expected);
}

fn scheduled_coreml_to_cpu_boundary_executes_on_both_real_backends() {
    const WIDTH: u32 = 64;
    let Some(ane) = ane_session() else { return };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(ane));
    let bytes = scheduled_coreml_to_cpu_graph(WIDTH);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 300,
        accelerator_speedup: 12,
        transfer_fixed_cost: 100,
        transfer_byte_cost: 0,
        ..runtime.automatic_device_model()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        [ACCEL_BACKEND, CPU_BACKEND]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 2);

    let values: Vec<f32> = (0..WIDTH as usize)
        .map(|index| index as f32 / 16.0 - 2.0)
        .collect();
    let input = f16_bytes(values.iter().copied());
    let expected_absolute = f16_bytes(values.iter().map(|value| value.abs()));
    let output = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(output[0], input, "Core ML identity MATMUL changed values");
    assert_eq!(
        output[1], expected_absolute,
        "CPU ABS received wrong values"
    );
}

#[test]
fn coreml_provider_scheduling_suite() {
    // Core ML's process-global compilation/execution lane is sensitive to
    // Rust assigning otherwise-serialized tests to different worker threads.
    // Keep the real-provider cases in one test so they execute sequentially
    // on one worker without weakening the rest of the test suite's parallelism.
    scheduled_rank2_broadcast_executes_on_coreml();
    ane_and_cpu_agree_on_f32_and_fp8_falls_back();
    scheduled_boolean_graph_executes_as_one_coreml_partition();
    scheduled_coreml_to_cpu_boundary_executes_on_both_real_backends();
    automatically_scheduled_matmul_executes_on_coreml();
}
