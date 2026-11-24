//! Heterogeneous MATMUL dispatch across virtio-accel providers.
//!
//!     cargo run --release -p axnn-vaccel --example hetero
//!     cargo run --release -p axnn-vaccel --example hetero --features coreml
//!     cargo run --release -p axnn-vaccel --example hetero --features openvino
//!     cargo run --release -p axnn-vaccel --example hetero --features vulkan
//!
//! Without a host-provider feature this sweeps the axtosa CPU provider.
//! With `coreml`, `openvino`, or `vulkan`, the same `Session` drives the
//! host accelerator next to the CPU. F8 artifacts pin to the CPU when the
//! accelerator rejects them, while large FP16/FP32 work is free to win.

#![cfg_attr(
    not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos"),
        all(feature = "vulkan", axnn_vaccel_vulkan_runtime)
    )),
    allow(unused_mut)
)]

use std::time::Instant;

use axnn_vaccel::{AxTosaAccelerator, MatMulDType, Session};

const SIZES: [usize; 6] = [64, 128, 192, 256, 384, 512];
const DTS: [(MatMulDType, &str); 4] = [
    (MatMulDType::F32, "f32"),
    (MatMulDType::F16, "f16"),
    (MatMulDType::B8, "b8"),
    (MatMulDType::F8, "f8"),
];

fn plane(dt: MatMulDType, len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut bytes = Vec::with_capacity(len * dt.input_size());
    for _ in 0..len {
        match dt {
            MatMulDType::F32 => {
                let v = (next() as f32 / u32::MAX as f32) - 0.5;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            MatMulDType::F16 => {
                // Exponent-limited bit patterns: finite, |v| < 2.
                let b = (next() as u16 & 0x2fff) | ((next() as u16 & 1) << 15);
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            MatMulDType::B8 | MatMulDType::F8 => {
                let raw = (next() & 0xff) as u8;
                bytes.push((raw & 0x3f) | (raw & 0x80));
            }
        }
    }
    bytes
}

/// Steady-state microseconds per call (programs and provider state warm).
fn time_matmul<P: axnn_vaccel::Accelerator>(
    session: &mut Session<P>,
    dt: MatMulDType,
    size: usize,
    a: &[u8],
    b: &[u8],
) -> Result<f64, axnn_vaccel::BackendError> {
    session.matmul_bytes(dt, size, size, size, a, b)?; // warm: build+load+run
    let flops = 2.0 * (size as f64).powi(3);
    let reps = ((2.0e8 / flops) as usize).clamp(3, 200);
    let start = Instant::now();
    for _ in 0..reps {
        session.matmul_bytes(dt, size, size, size, a, b)?;
    }
    Ok(start.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

fn gflops(size: usize, micros: f64) -> f64 {
    2.0 * (size as f64).powi(3) / (micros * 1e-6) / 1e9
}

fn main() {
    let mut cpu =
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("cpu session");

    // OpenVINO wins when several optional providers are enabled, then Core
    // ML, then Vulkan, so `cargo test --all-features` remains a meaningful
    // portable build.
    #[cfg(all(
        feature = "vulkan",
        axnn_vaccel_vulkan_runtime,
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime)),
        not(all(feature = "coreml", target_os = "macos"))
    ))]
    let mut accel = match axnn_vaccel::vulkan::open_session(None) {
        Ok(session) => {
            println!("(Vulkan device: {})\n", session.provider().device_name());
            Some(session)
        }
        Err(error) => {
            println!("(Vulkan unavailable: {error}; CPU only)\n");
            None
        }
    };

    #[cfg(all(
        feature = "coreml",
        target_os = "macos",
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
    ))]
    let mut accel = match virtio_accel_coreml::CoreMlAccelerator::new_tosa() {
        Ok(provider) => match Session::new(provider, virtio_accel_coreml::COREML_TOSA_TARGET) {
            Ok(session) => Some(session),
            Err(error) => {
                println!("(Core ML session unavailable: {error:?}; CPU only)\n");
                None
            }
        },
        Err(error) => {
            println!("(Neural Engine unavailable: {error:?}; CPU only)\n");
            None
        }
    };

    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    let mut accel = match virtio_accel_openvino::OpenVinoAccelerator::new() {
        Ok(provider) => match Session::new(provider, virtio_accel_openvino::OPENVINO_TOSA_TARGET) {
            Ok(session) => Some(session),
            Err(error) => {
                println!("(OpenVINO session unavailable: {error:?}; CPU only)\n");
                None
            }
        },
        Err(error) => {
            println!("(OpenVINO accelerator unavailable: {error:?}; CPU only)\n");
            None
        }
    };

    println!("MATMUL m=k=n, steady state, one submission per call");
    println!(
        "host kernel tiers: float={}, qdot={}",
        axnn_cpu::build_info::float_isa(),
        axnn_cpu::build_info::qdot_isa()
    );
    #[cfg(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos")
    ))]
    println!("providers: cpu = axtosa (axnn-cpu SIMD), accel = host provider");
    #[cfg(not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos")
    )))]
    println!("provider: cpu = axtosa (axnn-cpu SIMD kernels)");
    println!();
    println!(
        "{:>5} {:>5} {:>12} {:>9} {:>12} {:>9}  {}",
        "dtype", "size", "cpu µs", "GF/s", "accel µs", "GF/s", "route"
    );
    for (dt, name) in DTS {
        for size in SIZES {
            let a = plane(dt, size * size, 0xa11ce ^ size as u64);
            let b = plane(dt, size * size, 0xb0b ^ size as u64);
            let cpu_us = time_matmul(&mut cpu, dt, size, &a, &b).expect("cpu matmul");

            let mut accel_cell = String::from("           -");
            let mut accel_gf = String::from("        -");
            let mut route = "cpu (only provider)".to_string();

            #[cfg(any(
                all(feature = "openvino", axnn_vaccel_openvino_runtime),
                all(feature = "coreml", target_os = "macos")
            ))]
            if let Some(session) = accel.as_mut() {
                match session.ensure_program(dt, size, size, size) {
                    Ok(()) => match time_matmul(session, dt, size, &a, &b) {
                        Ok(us) => {
                            accel_cell = format!("{us:>12.1}");
                            accel_gf = format!("{:>9.1}", gflops(size, us));
                            route = if us < cpu_us {
                                "accel (faster)".to_string()
                            } else {
                                "cpu (faster)".to_string()
                            };
                        }
                        Err(error) => route = format!("cpu (accel run failed: {error:?})"),
                    },
                    Err(error) => {
                        route = format!("cpu (accel rejected: {error:?})");
                    }
                }
            }

            println!(
                "{name:>5} {size:>5} {cpu_us:>12.1} {:>9.1} {accel_cell} {accel_gf}  {route}",
                gflops(size, cpu_us),
            );
        }
        println!();
    }
}
