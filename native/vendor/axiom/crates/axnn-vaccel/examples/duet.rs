//! duet — one 3D scene, two devices, live.
//!
//!     cargo run --release -p axnn-vaccel --example duet --features window,coreml
//!     cargo run --release -p axnn-vaccel --example duet --features window,openvino
//!     cargo run --release -p axnn-vaccel --example duet --features window
//!
//! A sphere-traced planet/moon scene with an orbiting camera, analytic
//! lighting, real depth compositing, and cast shadows — where both
//! surfaces are **neural fields** evaluated through the virtio-accel
//! `Accelerator` contract:
//!
//! - the **planet**'s surface field is a CPPN (8→192→192→3) represented
//!   as a backend-neutral FP16 `.ax` graph (MATMUL→TANH→…→SIGMOID,
//!   weights baked as CONSTs), scheduled into provider partitions once,
//!   and reused for every frame;
//! - the **moon**'s surface field is an **B8 (OCP E4M3)** CPPN the host NPU
//!   cannot execute natively. The same `.ax` graph automatically partitions
//!   F8→FP16-expanded MATMUL/activation regions onto the NPU and keeps the
//!   graph-visible F8 narrowing casts on the CPU provider.
//!
//! The main thread owns the 3D: rays, sphere intersections, normals,
//! shading, shadows, and compositing. The devices own the neural
//! fields, each on its own worker thread, computing simultaneously.
//! Without an accelerator the planet field right-sizes (8→96→96→3)
//! onto the CPU provider and the same binary runs everywhere.
//!
//! Env: `DUET_FRAMES=N` renders exactly N frames (deterministic t) and
//! exits; `DUET_HEADLESS=1` skips window creation for finite benchmark
//! runs; `DUET_SNAPSHOT=path.ppm` writes the final frame.

#![feature(allocator_api)]
#![feature(f16)]
#![cfg_attr(
    not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos")
    )),
    allow(unused_mut, dead_code, unused_variables)
)]

use std::alloc::Global;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
use axnn_vaccel::ACCEL_BACKEND;
use axnn_vaccel::{
    Accelerator, AxCompileError, AxDeviceModel, AxExecutable, AxHetero, AxTosaAccelerator, Session,
};
use minifb::{Key, Scale, Window, WindowOptions};

const W: usize = 640;
const H: usize = 360;
/// Internal render resolution (bilinear ×2 upscale to the window).
const RW: usize = 320;
const RH: usize = 180;
/// Fused-MLP batch per submission (within Level 8K).
const CHUNK: usize = 7200;
const MOON_CHUNK: usize = 3072;

const PLANET_R: f32 = 1.0;
const MOON_R: f32 = 0.30;
const SUN: [f32; 3] = [0.55, 0.45, -0.70];
const MOON_DIMS: [usize; 4] = [8, 40, 40, 3];

// ── Small vector helpers ────────────────────────────────────────────────

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn rot_y(p: [f32; 3], ang: f32) -> [f32; 3] {
    let (s, c) = ang.sin_cos();
    [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]
}

/// Ray/sphere: nearest positive hit distance, if any.
fn hit_sphere(orig: [f32; 3], dir: [f32; 3], center: [f32; 3], r: f32) -> Option<f32> {
    let oc = sub3(orig, center);
    let b = dot3(oc, dir);
    let c = dot3(oc, oc) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 1e-3).then_some(t)
}

// ── Neural surface fields ───────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next_f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / u32::MAX as f32
    }
    fn gauss(&mut self) -> f32 {
        (self.next_f() + self.next_f() + self.next_f() + self.next_f()) * 2.0 - 4.0
    }
}

fn weights(dims: &[usize], seed: u64, gains: &[f32]) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    dims.windows(2)
        .zip(gains)
        .map(|(w, gain)| {
            let scale = gain / (w[0] as f32).sqrt();
            (0..w[0] * w[1]).map(|_| rng.gauss() * scale).collect()
        })
        .collect()
}

fn fast_tanh(x: f32) -> f32 {
    if x > 3.0 {
        1.0
    } else if x < -3.0 {
        -1.0
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}

/// Surface-field feature vector for a unit-sphere point in object
/// space; `evo` slowly evolves the field.
fn surface_features(p: [f32; 3], evo: f32) -> [f32; 8] {
    [
        (5.1 * p[0] + 2.3 * p[2]).sin(),
        (4.6 * p[1] - 1.8 * p[0] + 0.20 * evo).sin(),
        (6.7 * (p[0] + 0.7 * p[1])).sin() * (4.9 * p[2]).cos(),
        (9.4 * p[2] + 3.3 * p[1] - 0.15 * evo).sin(),
        (11.7 * p[0] * p[1] + 4.8 * p[2]).sin(),
        p[1].abs(),
        p[1],
        1.0,
    ]
}

/// Restructure a wide CPPN trunk into one independent sub-network per
/// output channel (block-diagonal hidden layers + disjoint head), then
/// normalize each channel's pre-sigmoid spread over a sphere probe.
///
/// Why: at large widths every dense random projection CLT-averages
/// onto the layer's dominant component, so all outputs collapse to the
/// same map (measured: channels identical to 1e-4 at width 192 across
/// any gain). Independent blocks make the channels distinct functions
/// by construction while keeping the dense matmul shape the
/// accelerator wants.
fn shape_head(dims: &[usize], w: &mut [Vec<f32>], spread: f32) {
    let nout = dims[dims.len() - 1];
    let block_of = |unit: usize, width: usize| (unit / (width / nout)).min(nout - 1);
    // Hidden→hidden layers become block-diagonal.
    for layer in 1..w.len() - 1 {
        let (k, n) = (dims[layer], dims[layer + 1]);
        for p in 0..k {
            for j in 0..n {
                if block_of(p, k) != block_of(j, n) {
                    w[layer][p * n + j] = 0.0;
                }
            }
        }
    }
    // Head: channel j reads only its own block.
    let (k, n) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let head = w.last_mut().unwrap();
    for p in 0..k {
        for j in 0..n {
            if block_of(p, k) != j {
                head[p * n + j] = 0.0;
            }
        }
    }
    normalize_head(dims, w, spread);
}

/// Scale head-weight columns to a target pre-sigmoid spread over a
/// probe of sphere points (CPPN contrast independent of width/gain).
fn normalize_head(dims: &[usize], w: &mut [Vec<f32>], spread: f32) {
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let probe: Vec<[f32; 8]> = (0..256)
        .map(|_| {
            let p = norm3([
                rng.next_f() * 2.0 - 1.0,
                rng.next_f() * 2.0 - 1.0,
                rng.next_f() * 2.0 - 1.0,
            ]);
            surface_features(p, 0.7)
        })
        .collect();
    let mut act: Vec<Vec<f32>> = probe.iter().map(|f| f.to_vec()).collect();
    for layer in 0..w.len() - 1 {
        let (k, n) = (dims[layer], dims[layer + 1]);
        act = act
            .iter()
            .map(|a| {
                (0..n)
                    .map(|j| fast_tanh((0..k).map(|p| a[p] * w[layer][p * n + j]).sum::<f32>()))
                    .collect()
            })
            .collect();
    }
    let (k, n) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let head = w.last_mut().unwrap();
    for j in 0..n {
        let pre: Vec<f32> = act
            .iter()
            .map(|a| (0..k).map(|p| a[p] * head[p * n + j]).sum::<f32>())
            .collect();
        let mean = pre.iter().sum::<f32>() / pre.len() as f32;
        let std = (pre.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / pre.len() as f32)
            .sqrt()
            .max(1e-6);
        let scale = spread / (std + mean.abs() * 0.5);
        for p in 0..k {
            head[p * n + j] *= scale;
        }
    }
    if std::env::var("DUET_STATS").is_ok() {
        // Scalar forward over the probe with the final weights.
        for j in 0..n {
            let outs: Vec<f32> = act
                .iter()
                .map(|a| {
                    let pre: f32 = (0..k).map(|p| a[p] * head[p * n + j]).sum();
                    1.0 / (1.0 + (-pre).exp())
                })
                .collect();
            let mn = outs.iter().cloned().fold(f32::MAX, f32::min);
            let mx = outs.iter().cloned().fold(f32::MIN, f32::max);
            println!(
                "scalar head[{j}] over probe: min {mn:.3} max {mx:.3} first {:.4} {:.4} {:.4}",
                outs[0], outs[1], outs[2]
            );
        }
    }
}

fn dec_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])) as f32)
        .collect()
}

// ── Workers: pure neural-field evaluators over Sessions ────────────────

/// main → worker: feature plane (`count` × 8 f32).
struct FieldJob {
    feats: Vec<f32>,
    count: usize,
}
/// worker → main: albedo plane (`count` × 3 f32) + timing.
struct FieldOut {
    rgb: Vec<f32>,
    ms: f32,
    dev: &'static str,
}

enum FusedSession {
    #[cfg(all(
        feature = "coreml",
        target_os = "macos",
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
    ))]
    Ane(Session<virtio_accel_coreml::CoreMlAccelerator>),
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    OpenVinoAx {
        runtime: Box<AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>>,
        executables: HashMap<usize, AxExecutable<Global>>,
        model: AxDeviceModel,
        dev: &'static str,
    },
    Cpu(Session<AxTosaAccelerator>),
}

enum MoonSession {
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    OpenVinoAx {
        runtime: Box<AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>>,
        executables: HashMap<usize, AxExecutable<Global>>,
        model: AxDeviceModel,
    },
    CpuAx {
        runtime: Box<AxHetero<AxTosaAccelerator, AxTosaAccelerator>>,
        executables: HashMap<usize, AxExecutable<Global>>,
        model: AxDeviceModel,
    },
}

fn compile_moon_graph<C: Accelerator, A: Accelerator>(
    runtime: &mut AxHetero<C, A>,
    batch: usize,
    weights: &[Vec<u8>],
    model: &AxDeviceModel,
) -> Result<AxExecutable<Global>, AxCompileError> {
    let tosa =
        axnn_vaccel::artifact::mlp_b8(batch, &MOON_DIMS, weights).expect("fixed moon MLP shape");
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).expect("moon TOSA → ax");
    let graph = ax_bytes.parse().expect("moon ax graph");
    runtime.compile_in(&graph, model, Global)
}

fn compile_planet_graph<C: Accelerator, A: Accelerator>(
    runtime: &mut AxHetero<C, A>,
    batch: usize,
    dims: &[usize],
    weights: &[Vec<f32>],
    model: &AxDeviceModel,
) -> Result<AxExecutable<Global>, AxCompileError> {
    let tosa =
        axnn_vaccel::artifact::mlp_f16(batch, dims, weights).expect("fixed planet MLP shape");
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).expect("planet TOSA → ax");
    let graph = ax_bytes.parse().expect("planet ax graph");
    runtime.compile_in(&graph, model, Global)
}

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("cpu session")
}

/// Planet surface field: fused f16 MLP on the best provider.
fn planet_worker(rx: mpsc::Receiver<FieldJob>, tx: mpsc::Sender<FieldOut>) {
    let mut wide = weights(&[8, 192, 192, 3], 0xC0FFEE, &[1.9, 0.9, 0.32]);
    shape_head(&[8, 192, 192, 3], &mut wide, 1.8);
    #[cfg(all(
        feature = "coreml",
        target_os = "macos",
        not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
    ))]
    let mut session = match virtio_accel_coreml::CoreMlAccelerator::new_tosa()
        .ok()
        .and_then(|p| Session::new(p, virtio_accel_coreml::COREML_TOSA_TARGET).ok())
    {
        Some(mut ane) => {
            println!("planet: compiling fused f16 CPPN (8→192→192→3) for the Neural Engine…");
            match ane.ensure(0, || {
                axnn_vaccel::artifact::mlp_f16(CHUNK, &[8, 192, 192, 3], &wide)
                    .expect("fixed planet MLP shape")
            }) {
                Ok(()) => {
                    println!("planet: ane accepted the fused MATMUL+TANH+SIGMOID graph");
                    FusedSession::Ane(ane)
                }
                Err(error) => {
                    println!("planet: ane rejected the fused graph ({error:?}); cpu fallback");
                    FusedSession::Cpu(cpu_session())
                }
            }
        }
        None => {
            println!("planet: no Neural Engine; cpu fallback");
            FusedSession::Cpu(cpu_session())
        }
    };
    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    let mut session = match virtio_accel_openvino::OpenVinoAccelerator::new()
        .ok()
        .and_then(|p| Session::new(p, virtio_accel_openvino::OPENVINO_TOSA_TARGET).ok())
    {
        Some(openvino) => {
            println!("planet: scheduling .ax f16 CPPN (8→192→192→3) for CPU + OpenVINO…");
            let mut runtime = AxHetero::new(cpu_session(), Some(openvino));
            let model = AxDeviceModel::default();
            match compile_planet_graph(&mut runtime, CHUNK, &[8, 192, 192, 3], &wide, &model) {
                Ok(executable) => {
                    let dev = if executable
                        .schedule()
                        .placements()
                        .iter()
                        .all(|backend| *backend == ACCEL_BACKEND)
                    {
                        "openvino·ax"
                    } else {
                        "cpu+openvino·ax"
                    };
                    println!(
                        "planet: .ax placed {} nodes into {} {dev} partition(s)",
                        executable.schedule().info().nodes,
                        executable.schedule().plan().partitions.len()
                    );
                    let mut executables = HashMap::new();
                    executables.insert(CHUNK, executable);
                    FusedSession::OpenVinoAx {
                        runtime: Box::new(runtime),
                        executables,
                        model,
                        dev,
                    }
                }
                Err(error) => {
                    println!("planet: .ax/OpenVINO compile failed ({error:?}); cpu fallback");
                    FusedSession::Cpu(cpu_session())
                }
            }
        }
        None => {
            println!("planet: OpenVINO accelerator unavailable; cpu fallback");
            FusedSession::Cpu(cpu_session())
        }
    };
    #[cfg(not(any(
        all(feature = "openvino", axnn_vaccel_openvino_runtime),
        all(feature = "coreml", target_os = "macos")
    )))]
    let mut session = FusedSession::Cpu(cpu_session());

    let (dev, dims, netw): (&str, Vec<usize>, Vec<Vec<f32>>) = match &session {
        #[cfg(all(
            feature = "coreml",
            target_os = "macos",
            not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
        ))]
        FusedSession::Ane(_) => ("ane", vec![8, 192, 192, 3], wide),
        #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
        FusedSession::OpenVinoAx { dev, .. } => (*dev, vec![8, 192, 192, 3], wide),
        FusedSession::Cpu(_) => {
            let mut narrow = weights(&[8, 96, 96, 3], 0xC0FFEE, &[1.3, 0.55, 0.32]);
            shape_head(&[8, 96, 96, 3], &mut narrow, 1.8);
            ("cpu", vec![8, 96, 96, 3], narrow)
        }
    };

    let mut feat16 = vec![0u8; CHUNK * 8 * 2];
    while let Ok(job) = rx.recv() {
        let start = Instant::now();
        let mut rgb = vec![0.0f32; job.count * 3];
        for (chunk_i, chunk) in job.feats.chunks(CHUNK * 8).enumerate() {
            let rows = chunk.len() / 8;
            let batch = match &session {
                #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
                FusedSession::OpenVinoAx { .. } => rows.next_power_of_two().clamp(64, CHUNK),
                _ => CHUNK,
            };
            feat16.resize(batch * 8 * 2, 0);
            for (dst, v) in feat16.as_chunks_mut::<2>().0.iter_mut().zip(chunk) {
                dst.copy_from_slice(&(*v as f16).to_bits().to_le_bytes());
            }
            for dst in feat16[chunk.len() * 2..].as_chunks_mut::<2>().0 {
                dst.copy_from_slice(&[0, 0]);
            }
            let out = match &mut session {
                #[cfg(all(
                    feature = "coreml",
                    target_os = "macos",
                    not(all(feature = "openvino", axnn_vaccel_openvino_runtime))
                ))]
                FusedSession::Ane(s) => s
                    .run(0, || unreachable!("warm"), &[&feat16], CHUNK * 3 * 2)
                    .expect("planet ane"),
                #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
                FusedSession::OpenVinoAx {
                    runtime,
                    executables,
                    model,
                    ..
                } => {
                    let executable = executables.entry(batch).or_insert_with(|| {
                        println!("planet: compiling cached .ax specialization for batch {batch}");
                        compile_planet_graph(runtime, batch, &dims, &netw, model)
                            .expect("planet ax/OpenVINO specialization")
                    });
                    runtime
                        .run_in(executable, &[&feat16], Global)
                        .expect("planet ax/openvino")
                        .pop()
                        .expect("planet output")
                }
                FusedSession::Cpu(s) => s
                    .run(
                        0,
                        || {
                            axnn_vaccel::artifact::mlp_f16(CHUNK, &dims, &netw)
                                .expect("fixed planet MLP shape")
                        },
                        &[&feat16],
                        CHUNK * 3 * 2,
                    )
                    .expect("planet cpu"),
            };
            let vals = dec_f16(&out[..rows * 3 * 2]);
            rgb[chunk_i * CHUNK * 3..][..rows * 3].copy_from_slice(&vals);
        }
        let _ = tx.send(FieldOut {
            rgb,
            ms: start.elapsed().as_secs_f32() * 1e3,
            dev,
        });
    }
}

/// Moon surface field: B8 storage and per-layer requantization. OpenVINO
/// expands the exact F8 values to FP16 for NPU compute; other providers
/// use the CPU's native F8 GEMMs after their capability probe rejects F8.
fn moon_worker(rx: mpsc::Receiver<FieldJob>, tx: mpsc::Sender<FieldOut>) {
    let mut wf = weights(&MOON_DIMS, 0xB1A2E, &[2.0, 1.1, 0.55]);
    shape_head(&MOON_DIMS, &mut wf, 1.5);
    let wq: Vec<Vec<u8>> = wf
        .iter()
        .map(|w| {
            w.iter()
                .map(|v| axnn_cpu::fp8::f32_to_e4m3_nearest(*v))
                .collect()
        })
        .collect();
    let cpu_fallback = || {
        let mut runtime: AxHetero<AxTosaAccelerator, AxTosaAccelerator> =
            AxHetero::new(cpu_session(), None);
        let model = AxDeviceModel::default();
        let executable =
            compile_moon_graph(&mut runtime, MOON_CHUNK, &wq, &model).expect("moon CPU ax compile");
        println!(
            "moon: .ax placed {} nodes into {} cpu·ax·B8 partition(s)",
            executable.schedule().info().nodes,
            executable.schedule().plan().partitions.len()
        );
        let mut executables = HashMap::new();
        executables.insert(MOON_CHUNK, executable);
        MoonSession::CpuAx {
            runtime: Box::new(runtime),
            executables,
            model,
        }
    };

    #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
    let mut session = match virtio_accel_openvino::OpenVinoAccelerator::new()
        .ok()
        .and_then(|provider| {
            Session::new(provider, virtio_accel_openvino::OPENVINO_TOSA_TARGET).ok()
        }) {
        Some(openvino) => {
            println!("moon: scheduling .ax B8 CPPN (8→40→40→3) for CPU + OpenVINO…");
            let mut runtime = AxHetero::new(cpu_session(), Some(openvino));
            let model = AxDeviceModel {
                fp8_promotion: true,
                ..AxDeviceModel::default()
            };
            match compile_moon_graph(&mut runtime, MOON_CHUNK, &wq, &model) {
                Ok(executable) => {
                    let accelerator_nodes = executable
                        .schedule()
                        .placements()
                        .iter()
                        .filter(|&&backend| backend == ACCEL_BACKEND)
                        .count();
                    println!(
                        "moon: .ax placed {accelerator_nodes}/{} nodes on OpenVINO across {} partition(s); F8 narrowing stays on CPU",
                        executable.schedule().info().nodes,
                        executable.schedule().plan().partitions.len()
                    );
                    let mut executables = HashMap::new();
                    executables.insert(MOON_CHUNK, executable);
                    MoonSession::OpenVinoAx {
                        runtime: Box::new(runtime),
                        executables,
                        model,
                    }
                }
                Err(error) => {
                    println!("moon: .ax/OpenVINO compile failed ({error:?}); CPU graph fallback");
                    cpu_fallback()
                }
            }
        }
        None => {
            println!("moon: OpenVINO accelerator unavailable; CPU graph fallback");
            cpu_fallback()
        }
    };
    #[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
    let mut session = cpu_fallback();

    while let Ok(job) = rx.recv() {
        let start = Instant::now();
        let mut rgb = vec![0.0f32; job.count * 3];
        let mut output_row = 0usize;
        for chunk in job.feats.chunks(MOON_CHUNK * 8) {
            let rows = chunk.len() / 8;
            let batch = rows.next_power_of_two().clamp(64, MOON_CHUNK);
            let mut cur: Vec<u8> = chunk
                .iter()
                .map(|&value| axnn_cpu::fp8::f32_to_e4m3_rne(value))
                .collect();
            cur.resize(batch * 8, 0);
            let out = match &mut session {
                #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
                MoonSession::OpenVinoAx {
                    runtime,
                    executables,
                    model,
                } => {
                    let executable = executables.entry(batch).or_insert_with(|| {
                        println!("moon: compiling cached .ax specialization for batch {batch}");
                        compile_moon_graph(runtime, batch, &wq, model)
                            .expect("moon ax/OpenVINO specialization")
                    });
                    runtime
                        .run_in(executable, &[&cur], Global)
                        .expect("moon ax/openvino")
                        .pop()
                        .expect("moon output")
                }
                MoonSession::CpuAx {
                    runtime,
                    executables,
                    model,
                } => {
                    let executable = executables.entry(batch).or_insert_with(|| {
                        println!("moon: compiling cached .ax specialization for batch {batch}");
                        compile_moon_graph(runtime, batch, &wq, model)
                            .expect("moon ax/CPU specialization")
                    });
                    runtime
                        .run_in(executable, &[&cur], Global)
                        .expect("moon ax/cpu")
                        .pop()
                        .expect("moon output")
                }
            };
            let vals = dec_f16(&out[..rows * 3 * 2]);
            rgb[output_row * 3..][..rows * 3].copy_from_slice(&vals);
            output_row += rows;
        }
        let _ = tx.send(FieldOut {
            rgb,
            ms: start.elapsed().as_secs_f32() * 1e3,
            #[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
            dev: match &session {
                MoonSession::OpenVinoAx { .. } => "cpu+openvino·ax·b8→f16",
                MoonSession::CpuAx { .. } => "cpu·ax·B8",
            },
            #[cfg(not(all(feature = "openvino", axnn_vaccel_openvino_runtime)))]
            dev: "cpu·ax·B8",
        });
    }
}

// ── Scene ───────────────────────────────────────────────────────────────

/// Per-pixel hit record, shaded once the field results return.
struct Hit {
    pixel: usize,
    normal: [f32; 3],
    view: [f32; 3],
    shadow: f32,
}

fn pack(r: f32, g: f32, b: f32) -> u32 {
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0) as u32).min(255);
    0xff00_0000 | (q(r) << 16) | (q(g) << 8) | q(b)
}

fn upscale(fb: &mut [u32], src: &[u32]) {
    for y in 0..H {
        let fy = y as f32 * (RH - 1) as f32 / (H - 1) as f32;
        let (y0, ty) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(RH - 1);
        for x in 0..W {
            let fx = x as f32 * (RW - 1) as f32 / (W - 1) as f32;
            let (x0, tx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(RW - 1);
            let lerp = |c00: u32, c10: u32, c01: u32, c11: u32, shift: u32| {
                let f = |c: u32| ((c >> shift) & 0xff) as f32;
                let top = f(c00) + (f(c10) - f(c00)) * tx;
                let bot = f(c01) + (f(c11) - f(c01)) * tx;
                ((top + (bot - top) * ty) as u32).min(255)
            };
            let (c00, c10) = (src[y0 * RW + x0], src[y0 * RW + x1]);
            let (c01, c11) = (src[y1 * RW + x0], src[y1 * RW + x1]);
            fb[y * W + x] = 0xff00_0000
                | (lerp(c00, c10, c01, c11, 16) << 16)
                | (lerp(c00, c10, c01, c11, 8) << 8)
                | lerp(c00, c10, c01, c11, 0);
        }
    }
}

fn hash01(x: i32, y: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393)
        ^ (y as u32).wrapping_mul(668265263)
        ^ (z as u32).wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    ((h >> 8) & 0xffff) as f32 / 65535.0
}

fn write_ppm(path: &str, fb: &[u32]) {
    let mut out = format!("P6\n{W} {H}\n255\n").into_bytes();
    for px in fb {
        out.extend([(px >> 16) as u8, (px >> 8) as u8, *px as u8]);
    }
    std::fs::write(path, out).expect("snapshot");
    println!("snapshot → {path}");
}

fn main() {
    let max_frames: Option<u64> = std::env::var("DUET_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let snapshot = std::env::var("DUET_SNAPSHOT").ok();
    let headless = std::env::var_os("DUET_HEADLESS").is_some();

    let (planet_tx, planet_job_rx) = mpsc::channel();
    let (planet_out_tx, planet_rx) = mpsc::channel();
    let (moon_tx, moon_job_rx) = mpsc::channel();
    let (moon_out_tx, moon_rx) = mpsc::channel();
    std::thread::spawn(move || planet_worker(planet_job_rx, planet_out_tx));
    std::thread::spawn(move || moon_worker(moon_job_rx, moon_out_tx));

    let mut window = if headless {
        assert!(max_frames.is_some(), "DUET_HEADLESS requires DUET_FRAMES");
        None
    } else {
        Some(
            Window::new(
                "axnn-vaccel duet",
                W,
                H,
                WindowOptions {
                    scale: Scale::X2,
                    ..WindowOptions::default()
                },
            )
            .expect("window"),
        )
    };

    let sun = norm3(SUN);
    let mut fb = vec![0u32; W * H];
    let mut lowres = vec![0u32; RW * RH];
    let mut frame = 0u64;
    let mut fps = 0.0f32;
    let mut last = Instant::now();
    let epoch = Instant::now();

    while window
        .as_ref()
        .is_none_or(|window| window.is_open() && !window.is_key_down(Key::Escape))
    {
        // Interactive runs are uncapped, so animation is wall-clock
        // paced; DUET_FRAMES uses frame-count time for deterministic
        // snapshots.
        let t = if max_frames.is_some() {
            frame as f32 / 60.0
        } else {
            epoch.elapsed().as_secs_f32()
        };

        // ── Camera orbit + moon orbit (world space) ───────────────────
        let cam_ang = 0.12 * t;
        let eye = [
            2.6 * cam_ang.sin(),
            0.55 + 0.25 * (0.07 * t).sin(),
            -2.6 * cam_ang.cos(),
        ];
        let fwd = norm3(sub3([0.0; 3], eye));
        let right = norm3([fwd[2], 0.0, -fwd[0]]);
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];
        let moon_ang = 0.45 * t + 2.1;
        let moon_c = [
            1.85 * moon_ang.cos(),
            0.35 * (0.9 * moon_ang).sin(),
            1.85 * moon_ang.sin(),
        ];
        let spin = 0.25 * t;

        // ── Rays, intersections, sky; gather per-object hit batches ──
        let mut planet_hits: Vec<Hit> = Vec::with_capacity(14000);
        let mut planet_feats: Vec<f32> = Vec::with_capacity(14000 * 8);
        let mut moon_hits: Vec<Hit> = Vec::with_capacity(3000);
        let mut moon_feats: Vec<f32> = Vec::with_capacity(3000 * 8);
        let aspect = RW as f32 / RH as f32;
        let fov = 0.62; // tan(half-fov)

        for py in 0..RH {
            for px in 0..RW {
                let pixel = py * RW + px;
                let sx = (px as f32 / RW as f32 - 0.5) * 2.0 * fov * aspect;
                let sy = -(py as f32 / RH as f32 - 0.5) * 2.0 * fov;
                let dir = norm3([
                    fwd[0] + sx * right[0] + sy * up[0],
                    fwd[1] + sx * right[1] + sy * up[1],
                    fwd[2] + sx * right[2] + sy * up[2],
                ]);

                let tp = hit_sphere(eye, dir, [0.0; 3], PLANET_R);
                let tm = hit_sphere(eye, dir, moon_c, MOON_R);
                match (tp, tm) {
                    (Some(tp), tm) if tm.is_none_or(|tm| tp <= tm) => {
                        let p = [
                            eye[0] + tp * dir[0],
                            eye[1] + tp * dir[1],
                            eye[2] + tp * dir[2],
                        ];
                        let n = norm3(p);
                        // Cast shadow: does the sun ray hit the moon?
                        let shadow = match hit_sphere(p, sun, moon_c, MOON_R * 1.05) {
                            Some(_) => 0.25,
                            None => 1.0,
                        };
                        planet_feats.extend(surface_features(rot_y(n, -spin), t));
                        planet_hits.push(Hit {
                            pixel,
                            normal: n,
                            view: dir,
                            shadow,
                        });
                    }
                    (_, Some(tm)) => {
                        let p = [
                            eye[0] + tm * dir[0],
                            eye[1] + tm * dir[1],
                            eye[2] + tm * dir[2],
                        ];
                        let n = norm3(sub3(p, moon_c));
                        // Eclipse: planet shadow on the moon.
                        let shadow = match hit_sphere(p, sun, [0.0; 3], PLANET_R) {
                            Some(_) => 0.18,
                            None => 1.0,
                        };
                        moon_feats.extend(surface_features(rot_y(n, -0.6 * spin), t * 0.5));
                        moon_hits.push(Hit {
                            pixel,
                            normal: n,
                            view: dir,
                            shadow,
                        });
                    }
                    _ => {
                        // Sky: hashed starfield + planet atmosphere limb.
                        let q = 210.0;
                        let h = hash01(
                            (dir[0] * q).floor() as i32,
                            (dir[1] * q).floor() as i32,
                            (dir[2] * q).floor() as i32,
                        );
                        let mut r = 0.010f32;
                        let mut g = 0.012f32;
                        let mut b = 0.030f32;
                        if h > 0.9955 {
                            let tw = 0.55 + 0.45 * (t * 3.0 + h * 61.0).sin();
                            let s = ((h - 0.9955) / 0.0045 * tw).min(1.0);
                            r += s;
                            g += s;
                            b += s * 1.1;
                        }
                        // Closest approach of the ray to the planet.
                        let bq = -dot3(eye, dir);
                        if bq > 0.0 {
                            let closest = (dot3(eye, eye) - bq * bq).max(0.0).sqrt();
                            if closest < PLANET_R * 1.35 {
                                let glow = ((PLANET_R * 1.35 - closest) / (PLANET_R * 0.35))
                                    .clamp(0.0, 1.0)
                                    .powi(2);
                                r += glow * 0.05;
                                g += glow * 0.16;
                                b += glow * 0.28;
                            }
                        }
                        lowres[pixel] = pack(r, g, b);
                    }
                }
            }
        }

        // ── Dispatch both neural fields; devices run concurrently ────
        planet_tx
            .send(FieldJob {
                feats: std::mem::take(&mut planet_feats),
                count: planet_hits.len(),
            })
            .unwrap();
        moon_tx
            .send(FieldJob {
                feats: std::mem::take(&mut moon_feats),
                count: moon_hits.len(),
            })
            .unwrap();
        let planet = planet_rx.recv().unwrap();
        let moon = moon_rx.recv().unwrap();

        if std::env::var("DUET_STATS").is_ok() && frame == 1 {
            for ch in 0..3 {
                let vals: Vec<f32> = planet.rgb.chunks(3).map(|c| c[ch]).collect();
                let mn = vals.iter().cloned().fold(f32::MAX, f32::min);
                let mx = vals.iter().cloned().fold(f32::MIN, f32::max);
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                println!("planet ch{ch}: min {mn:.3} max {mx:.3} mean {mean:.3}");
            }
        }
        // ── Shade + composite ─────────────────────────────────────────
        let shade = |hit: &Hit, albedo: [f32; 3], warm: bool| -> u32 {
            let n = hit.normal;
            let diffuse = dot3(n, sun).max(0.0) * hit.shadow;
            let nv = -dot3(n, hit.view);
            let refl = [
                hit.view[0] - 2.0 * dot3(hit.view, n) * n[0],
                hit.view[1] - 2.0 * dot3(hit.view, n) * n[1],
                hit.view[2] - 2.0 * dot3(hit.view, n) * n[2],
            ];
            let spec = dot3(refl, sun).max(0.0).powi(32) * 0.30 * hit.shadow;
            let rim = (1.0 - nv.max(0.0)).powi(3);
            let l = 0.07 + 0.93 * diffuse;
            if warm {
                pack(
                    albedo[0] * l + spec + rim * 0.10,
                    albedo[1] * l + spec + rim * 0.16,
                    albedo[2] * l + spec + rim * 0.30,
                )
            } else {
                pack(
                    albedo[0] * l * 0.9 + spec + rim * 0.08,
                    albedo[1] * l * 0.9 + spec + rim * 0.09,
                    albedo[2] * l + spec + rim * 0.12,
                )
            }
        };
        for (i, hit) in planet_hits.iter().enumerate() {
            let (r, g, b) = (
                planet.rgb[i * 3],
                planet.rgb[i * 3 + 1],
                planet.rgb[i * 3 + 2],
            );
            // Planet grade: the zero-mean channel difference splits
            // ocean/land (robust to head bias), g drives vegetation,
            // b clouds; polar ice keys off true latitude.
            let land = ((r - b) * 2.2 + 0.45).clamp(0.0, 1.0);
            let ocean = [0.06, 0.22 + 0.20 * g, 0.46 + 0.28 * b];
            let terra = [0.38 + 0.34 * r, 0.33 + 0.38 * g, 0.16 + 0.12 * b];
            let mut alb = [
                ocean[0] + (terra[0] - ocean[0]) * land,
                ocean[1] + (terra[1] - ocean[1]) * land,
                ocean[2] + (terra[2] - ocean[2]) * land,
            ];
            let lat = hit.normal[1].abs();
            let ice = ((lat - 0.72) / 0.14).clamp(0.0, 1.0) * (0.55 + 0.45 * b);
            for (a, w) in alb.iter_mut().zip([0.90, 0.93, 0.98]) {
                *a += (w - *a) * ice;
            }
            let cloud = (((g + b) * 0.5 - r) * 2.6 - 0.55).clamp(0.0, 1.0) * 0.8;
            for (a, w) in alb.iter_mut().zip([0.92, 0.94, 0.97]) {
                *a += (w - *a) * cloud;
            }
            lowres[hit.pixel] = shade(hit, alb, true);
        }
        for (i, hit) in moon_hits.iter().enumerate() {
            let (r, g, b) = (moon.rgb[i * 3], moon.rgb[i * 3 + 1], moon.rgb[i * 3 + 2]);
            // Moon grade: cratered grey with a cool tint.
            let base = 0.16 + 0.66 * r + 0.18 * (g - b);
            let alb = [base, base * (0.95 + 0.1 * g), base * (1.05 + 0.1 * b)];
            lowres[hit.pixel] = shade(hit, alb, false);
        }

        upscale(&mut fb, &lowres);
        if let Some(window) = window.as_mut() {
            window.update_with_buffer(&fb, W, H).unwrap();
        }

        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();
        fps = if frame == 0 {
            1.0 / dt
        } else {
            fps * 0.92 + (1.0 / dt) * 0.08
        };
        if frame.is_multiple_of(15)
            && let Some(window) = window.as_mut()
        {
            window.set_title(&format!(
                "axnn-vaccel duet — planet[{}·f16 fused] {:.1} ms ∥ moon[{}] {:.1} ms · {:.0} fps",
                planet.dev, planet.ms, moon.dev, moon.ms, fps
            ));
        }
        if frame.is_multiple_of(120) {
            println!(
                "frame {frame:>5}: planet[{}] {:.2} ms ({} px) ∥ moon[{}] {:.2} ms ({} px) → {:.0} fps",
                planet.dev,
                planet.ms,
                planet_hits.len(),
                moon.dev,
                moon.ms,
                moon_hits.len(),
                fps
            );
        }
        frame += 1;
        if let Some(max) = max_frames
            && frame >= max
        {
            if let Some(path) = &snapshot {
                write_ppm(path, &fb);
            }
            println!("done: {frame} frames, {fps:.0} fps steady");
            return;
        }
    }
}
