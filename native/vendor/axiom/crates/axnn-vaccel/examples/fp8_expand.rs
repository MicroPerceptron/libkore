//! F8 storage with host expansion into the OpenVINO NPU's FP16 compute tier.
//!
//!     RUSTFLAGS="-C target-cpu=native" \
//!       taskset -c 0 cargo run --release -p axnn-vaccel \
//!       --example fp8_expand --features openvino
//!
//! The NPU exposes FP16 and INT8, but not F8 or BF16. This benchmark
//! compares native CPU F8 MATMUL with three NPU costs: already-expanded
//! FP16 inputs, activation expansion with a cached/static weight matrix,
//! and expansion of both input matrices on every call.

#![cfg_attr(all(feature = "openvino", axnn_vaccel_openvino_runtime), feature(f16))]

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
use std::hint::black_box;
#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
use axnn_vaccel::{AxTosaAccelerator, MatMulDType, Session};
#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
use virtio_accel_openvino::{OPENVINO_TOSA_TARGET, OpenVinoAccelerator};

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
const SIZES: [usize; 5] = [128, 192, 256, 384, 512];
#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
const DTS: [(MatMulDType, &str); 2] = [(MatMulDType::B8, "b8"), (MatMulDType::F8, "f8")];

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn plane(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let raw = (state >> 33) as u8;
            (raw & 0x3f) | (raw & 0x80)
        })
        .collect()
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn expand(dt: MatMulDType, src: &[u8], dst: &mut [u8]) {
    match dt {
        MatMulDType::B8 => axnn_cpu::fp8::dequant_e4m3_f16(src, dst, src.len()),
        MatMulDType::F8 => axnn_cpu::fp8::dequant_e5m2_f16(src, dst, src.len()),
        _ => unreachable!(),
    }
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn micros(reps: usize, mut run: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        run();
    }
    start.elapsed().as_secs_f64() * 1e6 / reps as f64
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn preheat_cpu() {
    let src = vec![0x38u8; 512 * 512];
    let mut dst = vec![0u8; src.len() * 2];
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        axnn_cpu::fp8::dequant_e4m3_f16(&src, &mut dst, src.len());
        black_box(&dst);
    }
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn error(reference: &[u8], actual: &[u8]) -> (f32, f64) {
    assert_eq!(reference.len(), actual.len());
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    for (r, a) in reference.chunks_exact(2).zip(actual.chunks_exact(2)) {
        let r = f16::from_le_bytes([r[0], r[1]]) as f32;
        let a = f16::from_le_bytes([a[0], a[1]]) as f32;
        let delta = (r - a).abs();
        max_abs = max_abs.max(delta);
        sum_sq += f64::from(delta) * f64::from(delta);
    }
    (max_abs, (sum_sq / (reference.len() / 2) as f64).sqrt())
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn main() {
    let provider = OpenVinoAccelerator::new().expect("OpenVINO NPU");
    let mut npu = Session::new(provider, OPENVINO_TOSA_TARGET).expect("OpenVINO session");
    let mut cpu =
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("CPU session");

    println!(
        "F8 storage → FP16 compute, steady state (host float tier: {})",
        axnn_cpu::build_info::float_isa()
    );
    preheat_cpu();
    println!(
        "{:>5} {:>5} {:>10} {:>10} {:>9} {:>12} {:>12} {:>10} {:>10}",
        "dtype",
        "size",
        "cpu fp8",
        "npu f16",
        "expand A",
        "+ expand A",
        "+ expand A+B",
        "max |err|",
        "rms err"
    );
    println!(
        "{:>5} {:>5} {:>10} {:>10} {:>9} {:>12} {:>12}",
        "", "", "µs", "µs", "µs", "µs", "µs"
    );

    for (dt, name) in DTS {
        for size in SIZES {
            let len = size * size;
            let a = plane(len, 0xa11ce ^ size as u64);
            let b = plane(len, 0xb0b ^ size as u64);
            let mut ah = vec![0u8; len * 2];
            let mut bh = vec![0u8; len * 2];
            expand(dt, &a, &mut ah);
            expand(dt, &b, &mut bh);

            let cpu_out = cpu
                .matmul_bytes(dt, size, size, size, &a, &b)
                .expect("warm CPU");
            let npu_out = npu
                .matmul_bytes(MatMulDType::F16, size, size, size, &ah, &bh)
                .expect("warm NPU");
            let (max_abs, rms) = error(&cpu_out, &npu_out);
            let flops = 2.0 * (size as f64).powi(3);
            let reps = ((2.0e8 / flops) as usize).clamp(3, 100);

            let expand_reps = (50_000_000 / len).clamp(100, 10_000);
            let expand_a_only_us = micros(expand_reps, || {
                expand(dt, &a, &mut ah);
                black_box(&ah);
            });

            let cpu_us = micros(reps, || {
                cpu.matmul_bytes(dt, size, size, size, &a, &b).unwrap();
            });
            let npu_us = micros(reps, || {
                npu.matmul_bytes(MatMulDType::F16, size, size, size, &ah, &bh)
                    .unwrap();
            });
            let expand_a_us = micros(reps, || {
                expand(dt, &a, &mut ah);
                npu.matmul_bytes(MatMulDType::F16, size, size, size, &ah, &bh)
                    .unwrap();
            });
            let expand_both_us = micros(reps, || {
                expand(dt, &a, &mut ah);
                expand(dt, &b, &mut bh);
                npu.matmul_bytes(MatMulDType::F16, size, size, size, &ah, &bh)
                    .unwrap();
            });

            println!(
                "{name:>5} {size:>5} {cpu_us:>10.1} {npu_us:>10.1} {expand_a_only_us:>9.1} {expand_a_us:>12.1} {expand_both_us:>12.1} {max_abs:>10.4} {rms:>10.5}"
            );
        }
        println!();
    }
}

#[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
fn main() {
    eprintln!(
        "fp8_expand requires an OpenVINO C runtime; install it or set VIRTIO_ACCEL_OPENVINO_LIB_DIR"
    );
}
