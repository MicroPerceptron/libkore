//! Heterogeneous cross-validation against the virtio-accel Vulkan provider.
//!
//! Runs with `--features vulkan` and skips on hosts without a Vulkan 1.3
//! loader or compute device (`VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1` turns
//! absence into a failure, as in the provider's own suite). It exercises
//! both advertised tiers through the ordinary `.ax` placement path: F32 and
//! F16 work reaches the device, rejected F8 work is pinned to the axtosa CPU
//! session, and opt-in F8 promotion rides the FP16 tier.
#![cfg(all(feature = "vulkan", axnn_vaccel_vulkan_runtime))]
#![feature(allocator_api)]
#![feature(f16)]

use std::alloc::Global;

use ax::{DType, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxHetero, AxTosaAccelerator, CPU_BACKEND, Hetero, MatMulDType,
    Route, Session, artifact, vulkan,
};
use virtio_accel_vulkan::VulkanAccelerator;

fn vulkan_session() -> Option<Session<VulkanAccelerator>> {
    let provider = match vulkan::open(None) {
        Ok(provider) => provider,
        Err(error) if vulkan::is_absent(error) => {
            assert!(
                std::env::var_os("VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE").is_none_or(|v| v == "0"),
                "VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE is set but no Vulkan compute device exists"
            );
            eprintln!("skipping: no Vulkan 1.3 compute device on this host");
            return None;
        }
        Err(error) => panic!("Vulkan init failed: {error:?}"),
    };
    eprintln!("Vulkan device: {}", provider.device_name());
    Some(vulkan::session(provider).expect("vulkan session"))
}

/// The provider's own descriptor with launch and transfer costs zeroed, so
/// these small graphs measure admission rather than the default cost model
/// (which correctly keeps a 64-row MLP on the CPU).
fn admission_model(session: &Session<VulkanAccelerator>) -> AxDeviceModel {
    AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    }
    .with_accelerator_tosa_capability(session.provider(), session.target())
}

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap()
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

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])) as f32)
        .collect()
}

fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "plane length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn vulkan_and_cpu_agree_on_f32_and_f16_and_fp8_falls_back() {
    let Some(vulkan) = vulkan_session() else {
        return;
    };
    let mut hetero = Hetero::new(cpu_session(), Some(vulkan));

    let (m, k, n) = (64, 64, 64);
    let (a_bytes, a_values) = f32_plane(m * k, 7);
    let (b_bytes, b_values) = f32_plane(k * n, 11);
    let (route, out) = hetero
        .matmul_bytes(MatMulDType::F32, m, k, n, &a_bytes, &b_bytes)
        .unwrap();
    assert_eq!(route, Route::Accel, "f32 should route to Vulkan");
    let got = f32s(&out);
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
    // The provider's MATMUL is the sequential ascending-k binary32 sum, so
    // only summation order separates it from this reference.
    assert!(worst < 1e-4, "Vulkan f32 divergence too large: {worst}");

    // The FP16 tier is advertised on every device: half operands execute on
    // the device and narrow back through the provider's RNE conversion.
    let half = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| (*v as f16).to_bits().to_le_bytes())
            .collect()
    };
    let (route, out) = hetero
        .matmul_bytes(
            MatMulDType::F16,
            m,
            k,
            n,
            &half(&a_values),
            &half(&b_values),
        )
        .unwrap();
    assert_eq!(route, Route::Accel, "f16 should route to Vulkan");
    let got = f16s(&out);
    assert_eq!(got.len(), m * n);
    let worst = got
        .iter()
        .zip(&f32s(&{
            let (_, cpu_out) = Hetero::new(cpu_session(), None::<Session<AxTosaAccelerator>>)
                .matmul_bytes(MatMulDType::F32, m, k, n, &a_bytes, &b_bytes)
                .unwrap();
            cpu_out
        }))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 0.02, "Vulkan f16 divergence too large: {worst}");

    let a: Vec<u8> = (0..16 * 16).map(|i| (i as u8) & 0x3f).collect();
    let (route, cpu_fp8_out) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(route, Route::Cpu, "fp8 must fall back to the CPU provider");
    assert!(hetero.is_rejected(MatMulDType::B8));
    assert_eq!(cpu_fp8_out.len(), 16 * 16 * 2, "f16 output plane");

    hetero.min_fp8_promotion_flops = 0;
    let (route, promoted_out) = hetero
        .matmul_bytes(MatMulDType::B8, 16, 16, 16, &a, &a)
        .unwrap();
    assert_eq!(
        route,
        Route::Accel,
        "enabled F8 promotion should use the Vulkan FP16 tier"
    );
    let worst = max_abs_difference(&f16s(&promoted_out), &f16s(&cpu_fp8_out));
    assert!(worst <= 0.02, "E4M3 promotion changed FP16 output: {worst}");
}

/// A two-layer MLP: the second weight matrix is a `CONST` first read after
/// the first layer's intermediate died. virtio-accel-vulkan 0.3.8 packed such
/// constants into the freed region and an earlier dispatch overwrote them at
/// run time; 0.3.9 allocates constants as written at load.
#[test]
fn ax_scheduler_places_f16_mlp_on_vulkan() {
    let Some(vulkan) = vulkan_session() else {
        return;
    };
    let model = admission_model(&vulkan);
    let mut runtime = AxHetero::new(cpu_session(), Some(vulkan));

    let (m, dims) = (64, [8_usize, 16, 3]);
    let weights = vec![vec![0.03125; 8 * 16], vec![0.0625; 16 * 3]];
    let tosa = artifact::mlp_f16(m, &dims, &weights).unwrap();
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).unwrap();
    let graph = ax_bytes.parse().unwrap();
    // Placement discovers the FP16 tier from the provider's own capability
    // descriptor; no Vulkan-specific lowering model is written here.
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|backend| *backend == ACCEL_BACKEND),
        "placements: {:?}",
        executable.schedule().placements()
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
    assert_eq!(second, first, "warm Vulkan graph changed output");
    let (got, want) = (f16s(&first[0]), f16s(&expected[0]));
    let worst = max_abs_difference(&got, &want);
    assert!(worst <= 0.02, "Vulkan f16 MLP divergence: {worst}");
}

/// The aftershock lighting shape: `MATMUL → MINIMUM → EXP → MATMUL`, once
/// in F32 and once in F16, both compiled through the generic placement path.
fn lighting_graph(dtype: DType, probes: u32, lights: u32) -> GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let encode = |values: &[f32]| -> Vec<u8> {
        match dtype {
            DType::F16 => values
                .iter()
                .flat_map(|v| (*v as f16).to_bits().to_le_bytes())
                .collect(),
            _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    };
    let features: Vec<f32> = (0..probes)
        .flat_map(|i| {
            let x = i as f32 / probes as f32 - 0.5;
            [x, -x, 0.25, x * x * 2.0 + 0.0625, 1.0]
        })
        .collect();
    let q = b
        .tensor(
            dtype,
            TensorFlags::CONSTANT,
            &[1, probes, 5],
            &encode(&features),
        )
        .unwrap();
    let w = b
        .tensor(dtype, TensorFlags::INPUT, &[1, 5, lights], &[])
        .unwrap();
    let zero = b
        .tensor(dtype, TensorFlags::CONSTANT, &[1, 1, 1], &encode(&[0.0]))
        .unwrap();
    let distance = b
        .tensor(dtype, TensorFlags::default(), &[1, probes, lights], &[])
        .unwrap();
    let negative = b
        .tensor(dtype, TensorFlags::default(), &[1, probes, lights], &[])
        .unwrap();
    let falloff = b
        .tensor(dtype, TensorFlags::default(), &[1, probes, lights], &[])
        .unwrap();
    let rgb = b
        .tensor(dtype, TensorFlags::INPUT, &[1, lights, 3], &[])
        .unwrap();
    let out = b
        .tensor(dtype, TensorFlags::OUTPUT, &[1, probes, 3], &[])
        .unwrap();
    for (op, input, output) in [
        (OpCode::MATMUL, vec![q, w], distance),
        (OpCode::MINIMUM, vec![distance, zero], negative),
        (OpCode::EXP, vec![negative], falloff),
        (OpCode::MATMUL, vec![falloff, rgb], out),
    ] {
        b.node(op, NodeFlags::default(), &input, &[output], &[])
            .unwrap();
    }
    b.finish().unwrap()
}

#[test]
fn ax_scheduler_runs_lighting_graph_on_vulkan_in_f32_and_f16() {
    let Some(vulkan) = vulkan_session() else {
        return;
    };
    let (probes, lights) = (96_u32, 32_u32);
    let w: Vec<f32> = (0..5 * lights)
        .map(|i| ((i * 7) % 11) as f32 / 8.0 - 0.75)
        .collect();
    let rgb: Vec<f32> = (0..lights * 3)
        .map(|i| ((i * 5) % 9) as f32 / 8.0)
        .collect();

    let model = admission_model(&vulkan);
    let mut runtime = AxHetero::new(cpu_session(), Some(vulkan));
    let mut reference = AxHetero::new(cpu_session(), None::<Session<AxTosaAccelerator>>);
    for dtype in [DType::F32, DType::F16] {
        let graph_bytes = lighting_graph(dtype, probes, lights);
        let graph = graph_bytes.parse().unwrap();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        let placements = executable.schedule().placements();
        assert!(
            placements.iter().all(|backend| *backend == ACCEL_BACKEND),
            "{dtype:?} placements: {placements:?}"
        );
        let cpu_executable = reference.compile_auto_in(&graph, Global).unwrap();
        assert!(
            cpu_executable
                .schedule()
                .placements()
                .iter()
                .all(|backend| *backend == CPU_BACKEND)
        );
        let (w_bytes, rgb_bytes): (Vec<u8>, Vec<u8>) = match dtype {
            DType::F16 => (
                w.iter()
                    .flat_map(|v| (*v as f16).to_bits().to_le_bytes())
                    .collect(),
                rgb.iter()
                    .flat_map(|v| (*v as f16).to_bits().to_le_bytes())
                    .collect(),
            ),
            _ => (
                w.iter().flat_map(|v| v.to_le_bytes()).collect(),
                rgb.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        };
        let inputs: [&[u8]; 2] = [&w_bytes, &rgb_bytes];
        let device = runtime.run_in(&executable, &inputs, Global).unwrap();
        let host = reference.run_in(&cpu_executable, &inputs, Global).unwrap();
        assert_eq!(device.len(), 1);
        let (device, host, tolerance) = match dtype {
            DType::F16 => (f16s(&device[0]), f16s(&host[0]), 0.05),
            _ => (f32s(&device[0]), f32s(&host[0]), 1e-4),
        };
        assert_eq!(device.len(), probes as usize * 3);
        let peak = host.iter().copied().fold(0.0_f32, f32::max);
        assert!(peak > 0.0, "reference light field is empty");
        let worst = max_abs_difference(&device, &host);
        assert!(
            worst <= tolerance * peak.max(1.0),
            "{dtype:?} lighting divergence {worst} (peak {peak})"
        );
    }
}

/// RELU lowers to `MAXIMUM(x, 0)` inside the provider partition, so a graph
/// spelled the way kore's NPU lighting is (`-relu(-d)`) places whole rather
/// than splitting around a CPU hop.
#[test]
fn relu_places_inside_the_vulkan_partition() {
    let Some(vulkan) = vulkan_session() else {
        return;
    };
    let model = admission_model(&vulkan);
    let mut runtime = AxHetero::new(cpu_session(), Some(vulkan));
    let mut reference = AxHetero::new(cpu_session(), None::<Session<AxTosaAccelerator>>);
    let mut b = GraphBuilder::new_in(Global);
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 8, 16], &[])
        .unwrap();
    let flipped = b
        .tensor(DType::F32, TensorFlags::default(), &[1, 8, 16], &[])
        .unwrap();
    let clamped = b
        .tensor(DType::F32, TensorFlags::default(), &[1, 8, 16], &[])
        .unwrap();
    let negative = b
        .tensor(DType::F32, TensorFlags::default(), &[1, 8, 16], &[])
        .unwrap();
    let out = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 8, 16], &[])
        .unwrap();
    for (op, input, output) in [
        (OpCode::NEGATE, vec![x], flipped),
        (OpCode::RELU, vec![flipped], clamped),
        (OpCode::NEGATE, vec![clamped], negative),
        (OpCode::EXP, vec![negative], out),
    ] {
        b.node(op, NodeFlags::default(), &input, &[output], &[])
            .unwrap();
    }
    let graph_bytes = b.finish().unwrap();
    let graph = graph_bytes.parse().unwrap();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|backend| *backend == ACCEL_BACKEND),
        "placements: {:?}",
        executable.schedule().placements()
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    let host = reference.compile_auto_in(&graph, Global).unwrap();
    let (input, _) = f32_plane(8 * 16, 3);
    let got = runtime.run_in(&executable, &[&input], Global).unwrap();
    let want = reference.run_in(&host, &[&input], Global).unwrap();
    let worst = max_abs_difference(&f32s(&got[0]), &f32s(&want[0]));
    assert!(worst <= 1e-6, "exp(min(x, 0)) divergence: {worst}");
}

/// Every recipe that synthesizes a scalar operand must author it at the
/// output's rank: TOSA 1.0 broadcasts per dimension but requires equal ranks,
/// so a rank-1 literal against a rank-3 result is not a valid graph and a
/// provider refuses the program outright.
#[test]
fn synthesized_scalar_operands_load_on_a_real_provider() {
    let Some(vulkan) = vulkan_session() else {
        return;
    };
    let model = admission_model(&vulkan);
    let mut runtime = AxHetero::new(cpu_session(), Some(vulkan));
    let mut reference = AxHetero::new(cpu_session(), None::<Session<AxTosaAccelerator>>);
    // EXP2 synthesizes the POW base, LOG2 the reciprocal-ln2 factor, and RELU
    // the zero bound; all three are rank-3 here.
    for op in [OpCode::EXP2, OpCode::LOG2, OpCode::RELU] {
        let mut b = GraphBuilder::new_in(Global);
        let x = b
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 4, 8], &[])
            .unwrap();
        let y = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 4, 8], &[])
            .unwrap();
        b.node(op, NodeFlags::default(), &[x], &[y], &[]).unwrap();
        let graph_bytes = b.finish().unwrap();
        let graph = graph_bytes.parse().unwrap();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        assert!(
            executable
                .schedule()
                .placements()
                .iter()
                .all(|backend| *backend == ACCEL_BACKEND),
            "{op:?} placements: {:?}",
            executable.schedule().placements()
        );
        // LOG2 needs positive inputs; the plane is shifted into (0, 1].
        let (_, values) = f32_plane(4 * 8, 5);
        let input: Vec<u8> = values
            .iter()
            .flat_map(|value| (value + 0.75).to_le_bytes())
            .collect();
        let host = reference.compile_auto_in(&graph, Global).unwrap();
        let got = runtime.run_in(&executable, &[&input], Global).unwrap();
        let want = reference.run_in(&host, &[&input], Global).unwrap();
        let worst = max_abs_difference(&f32s(&got[0]), &f32s(&want[0]));
        assert!(worst <= 1e-5, "{op:?} divergence: {worst}");
    }
}
