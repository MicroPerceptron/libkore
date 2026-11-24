//! Heterogeneous dispatch showcase: one `Hetero` router serving a mixed
//! MATMUL queue across two live providers.
//!
//!     cargo run --release -p axnn-vaccel --example dispatch --features coreml
//!     cargo run --release -p axnn-vaccel --example dispatch --features openvino
//!     cargo run --release -p axnn-vaccel --example dispatch          # CPU only
//!
//! The router first *learns* capabilities (host NPUs reject F8 at
//! program load). OpenVINO then demonstrates explicit host F8→FP16
//! expansion for large jobs; other providers pin rejected dtypes to CPU.
//! then serves a queue shaped like real inference traffic: a few large
//! dense blocks, F8-quantized blocks, and hundreds of latency-sensitive
//! small heads. The same queue is timed under three policies:
//!
//!   cpu-only    — everything on the axtosa CPU provider
//!   accel-all   — everything offered to the host accelerator (subject
//!                 to its F8 fallback/promotion policy)
//!   hetero      — capability first, crossover threshold second
//!
//! Heterogeneous routing beats *both* homogeneous strategies — that gap
//! is the point of the `Accelerator` seam.

#![cfg_attr(
    not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos"),
        all(feature = "vulkan", axnn_vaccel_vulkan_runtime)
    )),
    allow(unused_mut, dead_code)
)]

use std::collections::HashMap;
use std::time::Instant;

use axnn_vaccel::{Accelerator, AxTosaAccelerator, Hetero, MatMulDType, Route, Session};

#[derive(Clone, Copy)]
struct Job {
    dt: MatMulDType,
    size: usize,
}

fn name(dt: MatMulDType) -> &'static str {
    match dt {
        MatMulDType::F32 => "f32",
        MatMulDType::F16 => "f16",
        MatMulDType::B8 => "b8",
        MatMulDType::F8 => "f8",
    }
}

fn plane(dt: MatMulDType, len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
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

/// The serving mix: 14 large blocks + 320 small heads, shuffled.
fn job_queue() -> Vec<Job> {
    let mut jobs = Vec::new();
    let push = |jobs: &mut Vec<Job>, dt, size, count| {
        for _ in 0..count {
            jobs.push(Job { dt, size });
        }
    };
    push(&mut jobs, MatMulDType::F32, 512, 6);
    push(&mut jobs, MatMulDType::F16, 384, 4);
    push(&mut jobs, MatMulDType::B8, 384, 4);
    push(&mut jobs, MatMulDType::F8, 512, 4);
    push(&mut jobs, MatMulDType::F32, 64, 320);
    // Deterministic Fisher–Yates so the trace reads like live traffic.
    let mut state = 0x5eedu64;
    for i in (1..jobs.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        jobs.swap(i, j);
    }
    jobs
}

type Planes = HashMap<(u8, usize), (Vec<u8>, Vec<u8>)>;

fn planes_for(jobs: &[Job]) -> Planes {
    let mut planes = Planes::new();
    for job in jobs {
        planes.entry((job.dt.tag(), job.size)).or_insert_with(|| {
            let len = job.size * job.size;
            (
                plane(job.dt, len, 0xa11ce ^ job.size as u64),
                plane(job.dt, len, 0xb0b ^ job.size as u64),
            )
        });
    }
    planes
}

fn flops(job: Job) -> u64 {
    2 * (job.size as u64).pow(3)
}

/// Run the queue under the router's current policy; returns total µs
/// and per-route counts. With `trace`, prints the decision for the
/// first job of each distinct shape as it streams past.
fn serve<C: Accelerator, A: Accelerator>(
    hetero: &mut Hetero<C, A>,
    jobs: &[Job],
    planes: &Planes,
    trace: bool,
) -> (f64, usize, usize) {
    let mut total_us = 0.0;
    let (mut on_accel, mut on_cpu) = (0usize, 0usize);
    let mut seen: Vec<(u8, usize)> = Vec::new();
    for (index, &job) in jobs.iter().enumerate() {
        let (a, b) = &planes[&(job.dt.tag(), job.size)];
        let start = Instant::now();
        let (route, _) = hetero
            .matmul_bytes(job.dt, job.size, job.size, job.size, a, b)
            .expect("matmul");
        let us = start.elapsed().as_secs_f64() * 1e6;
        total_us += us;
        match route {
            Route::Accel => on_accel += 1,
            Route::Cpu => on_cpu += 1,
        }
        let key = (job.dt.tag(), job.size);
        if trace && !seen.contains(&key) {
            seen.push(key);
            let reason = match route {
                Route::Accel if hetero.is_rejected(job.dt) => "  (host-expanded fp8 → fp16)",
                Route::Accel => "",
                Route::Cpu if hetero.is_rejected(job.dt) => "  (accel rejected fp8 → pinned)",
                Route::Cpu if flops(job) < hetero.min_accel_flops => "  (below crossover)",
                Route::Cpu => "",
            };
            println!(
                "  job {index:>3}: {:>4} {:>4}³ → {:<4} {:>8.1} µs{}",
                name(job.dt),
                job.size,
                match route {
                    Route::Accel => "accel",
                    Route::Cpu => "cpu",
                },
                us,
                reason
            );
        }
    }
    (total_us, on_accel, on_cpu)
}

fn main() {
    let jobs = job_queue();
    let planes = planes_for(&jobs);
    let shapes: Vec<Job> = {
        let mut seen = Vec::new();
        for &job in &jobs {
            if !seen
                .iter()
                .any(|s: &Job| s.dt == job.dt && s.size == job.size)
            {
                seen.push(job);
            }
        }
        seen
    };

    let cpu =
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("cpu session");

    #[cfg(all(
        feature = "coreml",
        target_os = "macos",
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
    ))]
    let accel = match virtio_accel_coreml::CoreMlAccelerator::new_tosa() {
        Ok(provider) => Session::new(provider, virtio_accel_coreml::COREML_TOSA_TARGET).ok(),
        Err(error) => {
            println!("(Neural Engine unavailable: {error:?}; running CPU-only)\n");
            None
        }
    };
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    let accel = match virtio_accel_openvino::OpenVinoAccelerator::new() {
        Ok(provider) => Session::new(provider, virtio_accel_openvino::OPENVINO_TOSA_TARGET).ok(),
        Err(error) => {
            println!("(OpenVINO accelerator unavailable: {error:?}; running CPU-only)\n");
            None
        }
    };
    #[cfg(all(
        feature = "vulkan",
        axnn_vaccel_vulkan_runtime,
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime)),
        not(all(feature = "coreml", target_os = "macos"))
    ))]
    let accel = match axnn_vaccel::vulkan::open_session(None) {
        Ok(session) => {
            println!("(Vulkan device: {})\n", session.provider().device_name());
            Some(session)
        }
        Err(error) => {
            println!("(Vulkan unavailable: {error}; running CPU-only)\n");
            None
        }
    };
    #[cfg(not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos"),
        all(feature = "vulkan", axnn_vaccel_vulkan_runtime)
    )))]
    let accel: Option<Session<AxTosaAccelerator>> = {
        println!("(no host provider enabled; run with --features coreml, openvino, or vulkan)\n");
        None
    };

    let has_accel = accel.is_some();
    let mut hetero = Hetero::new(cpu, accel);
    // Intel AI Boost advertises FP16 but not F8. At 256³ and above,
    // AVX2 host expansion is tiny relative to the saved compute time.
    // Enable this before warm-up so promoted FP16 shapes are compiled
    // outside the timed serving pass.
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    {
        hetero.min_fp8_promotion_flops = 32_000_000;
    }
    println!(
        "host kernel tiers: float={}, qdot={}",
        axnn_cpu::build_info::float_isa(),
        axnn_cpu::build_info::qdot_isa()
    );

    // ── Capability probe ─────────────────────────────────────────────
    // Load each distinct program on both providers: warms every cache
    // (host accelerators compile per shape) and teaches the router which dtypes
    // the accelerator rejects — before any timing.
    println!("capability probe ({} distinct programs):", shapes.len());
    for &job in &shapes {
        hetero
            .cpu
            .ensure_program(job.dt, job.size, job.size, job.size)
            .expect("cpu program");
        let verdict = match hetero.accel.as_mut() {
            None => "no accelerator".to_string(),
            Some(session) => match session.ensure_program(job.dt, job.size, job.size, job.size) {
                Ok(()) => "accel ok".to_string(),
                #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
                Err(error) if matches!(job.dt, MatMulDType::B8 | MatMulDType::F8) => {
                    format!("native accel rejected: {error:?}; fp16 promotion eligible")
                }
                Err(error) => format!("accel rejected: {error:?} → pinned to cpu"),
            },
        };
        println!("  {:>4} {:>4}³   {}", name(job.dt), job.size, verdict);
    }
    // One warm run per shape per side (steady-state from here on); this
    // also lets the router record the rejections the probe just saw.
    for &job in &shapes {
        let (a, b) = &planes[&(job.dt.tag(), job.size)];
        hetero.min_accel_flops = 0;
        hetero
            .matmul_bytes(job.dt, job.size, job.size, job.size, a, b)
            .expect("warm accel");
        hetero.min_accel_flops = u64::MAX;
        hetero
            .matmul_bytes(job.dt, job.size, job.size, job.size, a, b)
            .expect("warm cpu");
    }

    println!(
        "\njob queue: {} matmuls (6×f32·512³, 4×f16·384³, 4×b8·384³, 4×fp8·512³, 320×f32·64³)",
        jobs.len()
    );

    // ── The star: heterogeneous routing, decisions traced ────────────
    // Intel AVX2 wins decisively through 192³ and OpenVINO wins
    // decisively by 384³; keep the unstable 256³ boundary on the CPU.
    // Core ML retains the original Apple-host crossover.
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    let crossover = 64_000_000;
    #[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
    let crossover = 6_500_000;
    hetero.min_accel_flops = crossover;
    println!(
        "\nhetero routing (crossover = {:.1} MFLOP; first job of each shape shown):",
        crossover as f64 / 1e6
    );
    let (hetero_us, accel_jobs, cpu_jobs) = serve(&mut hetero, &jobs, &planes, true);
    println!("  routed {accel_jobs} jobs → accel, {cpu_jobs} jobs → cpu");

    // ── Homogeneous baselines over the identical queue ───────────────
    hetero.min_accel_flops = u64::MAX;
    let (cpu_us, _, _) = serve(&mut hetero, &jobs, &planes, false);

    let accel_us = if has_accel {
        hetero.min_accel_flops = 0;
        let (us, _, _) = serve(&mut hetero, &jobs, &planes, false);
        Some(us)
    } else {
        None
    };

    println!("\nstrategy totals over the same {}-job queue:", jobs.len());
    println!("  cpu-only   {:>8.1} ms", cpu_us / 1e3);
    if let Some(accel_us) = accel_us {
        #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
        let accel_note = "large fp8 expands to fp16; small jobs pay the submission floor";
        #[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
        let accel_note = "fp8 falls back; small jobs pay the submission floor";
        println!("  accel-all  {:>8.1} ms   ({accel_note})", accel_us / 1e3,);
        println!(
            "  hetero     {:>8.1} ms   {:.1}× vs cpu-only, {:.2}× vs accel-all",
            hetero_us / 1e3,
            cpu_us / hetero_us,
            accel_us / hetero_us
        );
    } else {
        println!(
            "  hetero     {:>8.1} ms   (no accelerator: identical routing to cpu-only)",
            hetero_us / 1e3
        );
    }
}
