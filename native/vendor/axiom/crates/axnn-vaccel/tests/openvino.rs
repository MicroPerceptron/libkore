//! Heterogeneous cross-validation against the OpenVINO host provider.
//!
//! Runs with `--features openvino` and skips on hosts without an
//! OpenVINO inference device. It exercises both halves of dispatch:
//! supported FP32 work reaches OpenVINO, while rejected F8 work is
//! pinned to the axtosa CPU session.
#![cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
#![feature(allocator_api)]
#![feature(f16)]

use std::alloc::Global;

use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, CPU_BACKEND, Hetero, MatMulDType,
    Route, Session, artifact,
};
use virtio_accel_openvino::{InitError, OPENVINO_TOSA_TARGET, OpenVinoAccelerator};

fn openvino_session() -> Option<Session<OpenVinoAccelerator>> {
    let provider = match OpenVinoAccelerator::new() {
        Ok(provider) => provider,
        Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
            eprintln!("skipping: no OpenVINO inference device on this host");
            return None;
        }
        Err(error) => panic!("OpenVINO init failed: {error:?}"),
    };
    Some(Session::new(provider, OPENVINO_TOSA_TARGET).expect("openvino session"))
}

fn f32_plane(len: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut state = seed | 1;
    let mut bytes = Vec::with_capacity(len * 4);
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = ((state >> 33) as f32 / u32::MAX as f32) - 0.5;
        bytes.extend_from_slice(&value.to_le_bytes());
        values.push(value);
    }
    (bytes, values)
}

#[test]
fn openvino_and_cpu_agree_on_f32_and_fp8_falls_back() {
    let Some(openvino) = openvino_session() else {
        return;
    };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut hetero = Hetero::new(cpu, Some(openvino));

    let (m, k, n) = (64, 64, 64);
    let (a_bytes, a_values) = f32_plane(m * k, 7);
    let (b_bytes, b_values) = f32_plane(k * n, 11);
    let (route, out) = hetero
        .matmul_bytes(MatMulDType::F32, m, k, n, &a_bytes, &b_bytes)
        .unwrap();
    assert_eq!(route, Route::Accel, "f32 should route to OpenVINO");

    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut expected = 0.0f32;
            for p in 0..k {
                expected += a_values[i * k + p] * b_values[p * n + j];
            }
            worst = worst.max((got[i * n + j] - expected).abs());
        }
    }
    assert!(worst < 0.05, "OpenVINO f32 divergence too large: {worst}");

    let a: Vec<u8> = (0..16 * 16).map(|i| (i as u8) & 0x3f).collect();
    let (route, cpu_fp8_out) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(route, Route::Cpu, "fp8 must fall back to the CPU provider");
    assert!(hetero.is_rejected(MatMulDType::B8));
    assert_eq!(cpu_fp8_out.len(), 16 * 16 * 2, "f16 output plane");

    let (route, _) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(route, Route::Cpu, "rejected dtype should stay pinned");

    // Promotion is deliberately opt-in: once enabled, the exact F8 values
    // are represented losslessly as FP16 inputs and execute on the NPU.
    hetero.min_fp8_promotion_flops = 0;
    let (route, promoted_out) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(
        route,
        Route::Accel,
        "enabled F8 promotion should use FP16 NPU"
    );
    assert_eq!(
        promoted_out, cpu_fp8_out,
        "E4M3 promotion changed FP16 output"
    );
}

#[test]
fn ax_scheduler_compiles_and_reuses_fused_openvino_partition() {
    let Some(openvino) = openvino_session() else {
        return;
    };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(openvino));

    let (m, dims) = (64, [8_usize, 16, 3]);
    let weights = vec![vec![0.03125; 8 * 16], vec![0.0625; 16 * 3]];
    let tosa = artifact::mlp_f16(m, &dims, &weights).unwrap();
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).unwrap();
    let graph = ax_bytes.parse().unwrap();
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

    let input: Vec<u8> = (0..m * 8)
        .flat_map(|index| {
            let value = index as f32 / 512.0 - 0.5;
            (value as f16).to_bits().to_le_bytes()
        })
        .collect();
    let expected = axtosa::execute(&tosa, &[&input]).unwrap();
    let first = runtime.run_in(&executable, &[&input], Global).unwrap();
    let second = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(first, expected);
    assert_eq!(second, first);
}

#[test]
fn ax_scheduler_promotes_fp8_regions_and_preserves_cpu_narrowing() {
    let Some(openvino) = openvino_session() else {
        return;
    };
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(openvino));
    let (m, dims) = (64, [4_usize, 8, 6, 3]);
    let weights: Vec<Vec<u8>> = dims
        .windows(2)
        .enumerate()
        .map(|(layer, shape)| {
            (0..shape[0] * shape[1])
                .map(|index| {
                    let value = ((index * 17 + layer * 13) % 41) as f32 / 82.0 - 0.25;
                    axnn_cpu::fp8::f32_to_e4m3_rne(value)
                })
                .collect()
        })
        .collect();
    let tosa = artifact::mlp_b8(m, &dims, &weights).unwrap();
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).unwrap();
    let graph = ax_bytes.parse().unwrap();
    let model = AxDeviceModel {
        fp8_promotion: true,
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        &[
            ACCEL_BACKEND,
            ACCEL_BACKEND,
            CPU_BACKEND,
            ACCEL_BACKEND,
            ACCEL_BACKEND,
            CPU_BACKEND,
            ACCEL_BACKEND,
            ACCEL_BACKEND,
        ]
    );

    let input: Vec<u8> = (0..m * dims[0])
        .map(|index| axnn_cpu::fp8::f32_to_e4m3_rne(index as f32 / 128.0 - 0.5))
        .collect();
    let expected = axtosa::execute(&tosa, &[&input]).unwrap();
    let first = runtime.run_in(&executable, &[&input], Global).unwrap();
    let second = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(second, first, "warm promoted graph changed output");

    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])) as f32)
            .collect::<Vec<_>>()
    };
    let got = decode(&first[0]);
    let expected = decode(&expected[0]);
    let worst = got
        .iter()
        .zip(expected)
        .map(|(&actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst <= 0.02, "promoted graph divergence: {worst}");
}
