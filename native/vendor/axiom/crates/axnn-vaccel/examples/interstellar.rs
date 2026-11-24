//! Backend-neutral relativistic scene demo built from `axgeom` `.ax` kernels.
//!
//! The first scene is a Dneg traversable wormhole. Its fixed-step ray map is
//! compiled once, scheduled across CPU + accelerator, and reused for every
//! tile. The completed ray map is then cached independently of procedural sky
//! shading, so an unchanged camera does not retrace geodesics every frame.
//! The two celestial environments rotate independently against that stationary
//! lens map, producing animation without pretending that a moving camera can
//! reuse the same geodesics.
//!
//! ```text
//! cargo +nightly run --release -p axnn-vaccel --example interstellar \
//!   --features window,openvino
//! ```
//! On Apple Silicon, replace `openvino` with `coreml` to use the Core ML host
//! backend. Automatic compilation discovers provider capabilities and selects
//! placement; Core ML Boolean ABI boundaries use lossless FP32 bridges.
//! Automatic recurrence keeps provider feedback resident when possible, or
//! reuses retained host tensors for native/mixed schedules.
//!
//! Quality is independent of window size: `AXGEOM_STEPS` and `AXGEOM_SPP`
//! control geodesic/image sampling, while `AXGEOM_WIDTH` and `AXGEOM_HEIGHT`
//! only control the output raster. Set `AXGEOM_HEADLESS=1`, optionally with
//! `AXGEOM_SNAPSHOT=wormhole.ppm`, for a finite run. `AXGEOM_CAMERA_ELL` moves
//! through the throat and `AXGEOM_CAMERA_YAW` turns in its local ray frame.
//! `AXGEOM_SKY_SPEED` controls celestial angular velocity (radians/second),
//! `AXGEOM_THREADS` controls host shading parallelism (default: available
//! parallelism), and `AXGEOM_UNCAP=1` removes the default 60 Hz presentation
//! cap. Set `AXGEOM_THREADS=1` for a directly comparable scalar baseline.
//! `AXGEOM_SKY_WIDTH` and `AXGEOM_SKY_HEIGHT` size the procedural linear-RGB
//! equirectangular atlases baked once before tracing (default: 2048x1024).

#![feature(allocator_api)]

use std::alloc::Global;
use std::time::Instant;

use axgeom::graph::{WormholeGraphSpec, abi, build_wormhole_graph_in};
use axgeom::wormhole::{
    DnegParams, PlanarFrame, TraceConfig, escaped_direction, initialize_planar_ray,
};
use axnn_vaccel::{ACCEL_BACKEND, Accelerator, AxFeedback, AxHetero, AxTosaAccelerator, Session};
use minifb::{Key, Scale, Window, WindowOptions};

#[path = "common/cinematic.rs"]
mod cinematic;
use cinematic::{SkyTexture, fbm, hash_u32, pack_aces};
use virtio_accel_tosa::TosaCapabilityProvider;

const DEFAULT_WIDTH: usize = 640;
const DEFAULT_HEIGHT: usize = 360;
const DEFAULT_TILE_RAYS: usize = 8_192;
const DEFAULT_STEPS: u32 = 256;
const DEFAULT_STEPS_PER_GRAPH: u32 = 64;
const DEFAULT_SKY_SPEED: f32 = 0.18;

#[derive(Clone, Copy, Debug)]
struct RenderConfig {
    width: usize,
    height: usize,
    tile_rays: usize,
    max_steps: u32,
    steps_per_graph: u32,
    samples_per_pixel: usize,
    step_size: f32,
    escape_ell: f32,
    camera_ell: f32,
    camera_yaw: f32,
    tan_half_fov: f32,
    sky_speed: f32,
}

impl RenderConfig {
    fn from_env() -> Self {
        let config = Self {
            width: env_or("AXGEOM_WIDTH", DEFAULT_WIDTH),
            height: env_or("AXGEOM_HEIGHT", DEFAULT_HEIGHT),
            tile_rays: env_or("AXGEOM_TILE_RAYS", DEFAULT_TILE_RAYS),
            max_steps: env_or("AXGEOM_STEPS", DEFAULT_STEPS),
            steps_per_graph: env_or("AXGEOM_STEPS_PER_GRAPH", DEFAULT_STEPS_PER_GRAPH),
            samples_per_pixel: env_or("AXGEOM_SPP", 1),
            step_size: env_or("AXGEOM_STEP_SIZE", 0.1),
            escape_ell: env_or("AXGEOM_ESCAPE_ELL", 10.0),
            camera_ell: env_or("AXGEOM_CAMERA_ELL", 5.0),
            camera_yaw: env_or("AXGEOM_CAMERA_YAW", 0.0),
            tan_half_fov: env_or("AXGEOM_TAN_HALF_FOV", 0.62),
            sky_speed: env_or("AXGEOM_SKY_SPEED", DEFAULT_SKY_SPEED),
        };
        assert!(config.width > 0 && config.height > 0);
        assert!((1..=8_192).contains(&config.tile_rays));
        assert!(config.max_steps > 0 && config.steps_per_graph > 0);
        assert_eq!(
            config.max_steps % config.steps_per_graph,
            0,
            "AXGEOM_STEPS must be exactly divisible by AXGEOM_STEPS_PER_GRAPH"
        );
        assert!(config.samples_per_pixel > 0);
        assert!(config.step_size > 0.0 && config.step_size.is_finite());
        assert!(
            config.camera_ell.is_finite()
                && config.camera_yaw.is_finite()
                && config.tan_half_fov > 0.0
                && config.sky_speed.is_finite()
        );
        config
    }

    fn sample_count(self) -> usize {
        self.width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.samples_per_pixel))
            .expect("requested image is too large")
    }

    fn dispatches(self) -> u32 {
        self.max_steps / self.steps_per_graph
    }
}

fn env_or<T: core::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

#[derive(Clone, Copy)]
struct RaySample {
    direction: [f32; 3],
    universe: u8,
}

const UNIVERSE_ORIGIN: u8 = 0;
const UNIVERSE_OTHER: u8 = 1;
const UNRESOLVED: u8 = 2;

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("axtosa CPU session")
}

#[cfg(all(feature = "openvino", axnn_vaccel_openvino_runtime))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>,
    String,
) {
    let (accelerator, label) = match virtio_accel_openvino::OpenVinoAccelerator::new() {
        Ok(provider) => {
            let label = provider
                .device_info()
                .map(|info| format!("OpenVINO {:?}", info.identity.class))
                .unwrap_or_else(|_| "OpenVINO".to_owned());
            match Session::new(provider, virtio_accel_openvino::OPENVINO_TOSA_TARGET) {
                Ok(session) => (Some(session), label),
                Err(_) => (None, "CPU (OpenVINO session unavailable)".to_owned()),
            }
        }
        Err(_) => (None, "CPU (OpenVINO unavailable)".to_owned()),
    };
    (AxHetero::new(cpu_session(), accelerator), label)
}

#[cfg(all(
    not(all(feature = "openvino", axnn_vaccel_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, virtio_accel_coreml::CoreMlAccelerator>,
    String,
) {
    let accelerator = virtio_accel_coreml::CoreMlAccelerator::new_tosa()
        .ok()
        .and_then(|provider| Session::new(provider, virtio_accel_coreml::COREML_TOSA_TARGET).ok());
    let label = if accelerator.is_some() {
        "Core ML / Neural Engine".to_owned()
    } else {
        "CPU (Core ML unavailable)".to_owned()
    };
    (AxHetero::new(cpu_session(), accelerator), label)
}

#[cfg(not(any(
    all(feature = "openvino", axnn_vaccel_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn runtime() -> (AxHetero<AxTosaAccelerator, AxTosaAccelerator>, String) {
    (AxHetero::new(cpu_session(), None), "CPU".to_owned())
}

fn write_f32_bytes(values: &[f32], bytes: &mut Vec<u8>) {
    bytes.clear();
    bytes.reserve(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("f32 plane"))
}

fn normalized(direction: [f32; 3]) -> [f32; 3] {
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt()
            .max(1.0e-12);
    [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ]
}

fn sample_jitter(sample: usize, pixel: usize) -> (f32, f32) {
    if sample == 0 {
        return (0.5, 0.5);
    }
    let seed = hash_u32(
        (pixel as u32)
            .wrapping_mul(0x9e37_79b9)
            .wrapping_add(sample as u32),
    );
    let x = ((seed & 0xffff) as f32 + 0.5) / 65_536.0;
    let y = ((seed >> 16) as f32 + 0.5) / 65_536.0;
    (x, y)
}

struct RayTile<'a> {
    ell: &'a mut [f32],
    phi: &'a mut [f32],
    p_ell: &'a mut [f32],
    impact: &'a mut [f32],
    active: &'a mut [u8],
    frames: &'a mut [PlanarFrame],
}

fn initialize_tile(
    config: RenderConfig,
    params: &DnegParams,
    first_sample: usize,
    tile: RayTile<'_>,
) -> usize {
    let RayTile {
        ell,
        phi,
        p_ell,
        impact,
        active,
        frames,
    } = tile;
    let remaining = config.sample_count() - first_sample;
    let count = remaining.min(config.tile_rays);
    let aspect = config.width as f32 / config.height as f32;
    let (yaw_sin, yaw_cos) = config.camera_yaw.sin_cos();
    let forward = [-yaw_cos, yaw_sin, 0.0];
    let right = [yaw_sin, yaw_cos, 0.0];
    for lane in 0..count {
        let sample_index = first_sample + lane;
        let pixel = sample_index / config.samples_per_pixel;
        let sub_sample = sample_index % config.samples_per_pixel;
        let x = pixel % config.width;
        let y = pixel / config.width;
        let (jitter_x, jitter_y) = sample_jitter(sub_sample, pixel);
        let screen_x = ((x as f32 + jitter_x) / config.width as f32 * 2.0 - 1.0)
            * config.tan_half_fov
            * aspect;
        let screen_y =
            -((y as f32 + jitter_y) / config.height as f32 * 2.0 - 1.0) * config.tan_half_fov;
        let local_direction = normalized([
            forward[0] + screen_x * right[0],
            forward[1] + screen_x * right[1],
            screen_y,
        ]);
        let (ray, frame) = initialize_planar_ray(params, config.camera_ell, local_direction);
        ell[lane] = ray.ell();
        phi[lane] = ray.phi();
        p_ell[lane] = ray.p_ell();
        impact[lane] = ray.impact_parameter;
        active[lane] = 1;
        frames[lane] = frame;
    }
    // Every live lane above is overwritten. Only the unused tail of the last
    // tile needs clearing; its floating-point state remains finite and the
    // graph's inactive mask prevents it from affecting cached results.
    active[count..].fill(0);
    count
}

fn trace_map<C: Accelerator, A: Accelerator + TosaCapabilityProvider>(
    runtime: &mut AxHetero<C, A>,
    backend_label: &str,
    config: RenderConfig,
    mut on_progress: impl FnMut(&[RaySample]) -> bool,
) -> Option<Vec<RaySample>> {
    let params = DnegParams::REFERENCE;
    let trace = TraceConfig::new(config.step_size, config.max_steps, config.escape_ell);
    let graph_bytes = build_wormhole_graph_in(
        WormholeGraphSpec {
            rays: config.tile_rays as u32,
            steps: config.steps_per_graph,
            params,
            trace,
        },
        Global,
    )
    .expect("build wormhole .ax graph");
    let graph = graph_bytes.parse().expect("parse wormhole .ax graph");
    let compile_start = Instant::now();
    let executable = runtime
        .compile_auto_in(&graph, Global)
        .expect("compile wormhole .ax graph");
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1_000.0;
    let accelerator_nodes = executable
        .schedule()
        .placements()
        .iter()
        .filter(|&&backend| backend == ACCEL_BACKEND)
        .count();
    println!(
        "wormhole: {} nodes, {} partition(s), {accelerator_nodes} accelerator node(s); runtime {backend_label}; compiled in {compile_ms:.1} ms",
        graph.node_count(),
        executable.schedule().plan().partitions.len(),
    );

    let mut ray_map = Vec::with_capacity(config.sample_count());
    let mut ell = vec![0.0; config.tile_rays];
    let mut phi = vec![0.0; config.tile_rays];
    let mut p_ell = vec![0.0; config.tile_rays];
    let mut impact = vec![0.0; config.tile_rays];
    let mut active = vec![0; config.tile_rays];
    let mut frames = vec![
        PlanarFrame {
            transverse_axis: [1.0, 0.0],
        };
        config.tile_rays
    ];
    let mut ell_bytes = Vec::with_capacity(config.tile_rays * 4);
    let mut phi_bytes = Vec::with_capacity(config.tile_rays * 4);
    let mut p_ell_bytes = Vec::with_capacity(config.tile_rays * 4);
    let mut impact_bytes = Vec::with_capacity(config.tile_rays * 4);
    let mut active_bytes = Vec::with_capacity(config.tile_rays);
    let trace_start = Instant::now();
    let mut submitted_lane_steps = 0_u64;
    while ray_map.len() < config.sample_count() {
        let first_sample = ray_map.len();
        let count = initialize_tile(
            config,
            &params,
            first_sample,
            RayTile {
                ell: &mut ell,
                phi: &mut phi,
                p_ell: &mut p_ell,
                impact: &mut impact,
                active: &mut active,
                frames: &mut frames,
            },
        );

        // Encode the graph ABI once per tile. Automatic recurrence retains
        // provider buffers when possible, or moves native host tensor backing
        // between dispatches. Only final outputs materialize byte vectors.
        write_f32_bytes(&ell, &mut ell_bytes);
        write_f32_bytes(&phi, &mut phi_bytes);
        write_f32_bytes(&p_ell, &mut p_ell_bytes);
        write_f32_bytes(&impact, &mut impact_bytes);
        active_bytes.clear();
        active_bytes.extend_from_slice(&active);
        let outputs = runtime
            .run_repeated_auto_in(
                &executable,
                &[
                    &ell_bytes,
                    &phi_bytes,
                    &p_ell_bytes,
                    &impact_bytes,
                    &active_bytes,
                ],
                config.dispatches() as usize,
                &[
                    AxFeedback::new(abi::ELL_IN, abi::ELL_OUT),
                    AxFeedback::new(abi::PHI_IN, abi::PHI_OUT),
                    AxFeedback::new(abi::P_ELL_IN, abi::P_ELL_OUT),
                    AxFeedback::new(abi::ACTIVE_IN, abi::ACTIVE_OUT),
                ],
                Global,
            )
            .expect("run recurrent wormhole .ax graph");
        submitted_lane_steps = submitted_lane_steps.saturating_add(
            config.tile_rays as u64 * config.steps_per_graph as u64 * config.dispatches() as u64,
        );
        let mut outputs = outputs.into_iter();
        ell_bytes = outputs.next().expect("ell output");
        phi_bytes = outputs.next().expect("phi output");
        p_ell_bytes = outputs.next().expect("p_ell output");
        active_bytes = outputs.next().expect("active output");
        assert!(outputs.next().is_none());

        for lane in 0..count {
            let final_ray = axgeom::wormhole::PlanarRay::new(
                f32_at(&ell_bytes, lane),
                f32_at(&phi_bytes, lane),
                f32_at(&p_ell_bytes, lane),
                impact[lane],
            );
            let universe = if active_bytes[lane] != 0 {
                UNRESOLVED
            } else if final_ray.ell() < 0.0 {
                UNIVERSE_OTHER
            } else {
                UNIVERSE_ORIGIN
            };
            ray_map.push(RaySample {
                direction: escaped_direction(&params, &final_ray, &frames[lane]),
                universe,
            });
        }

        let completed = ray_map.len();
        if completed == config.sample_count()
            || completed % (config.tile_rays * 16) < config.tile_rays
        {
            println!(
                "wormhole: traced {completed}/{} samples",
                config.sample_count()
            );
        }
        if !on_progress(&ray_map) {
            println!(
                "wormhole: trace cancelled at {completed}/{} samples",
                config.sample_count()
            );
            return None;
        }
    }
    let trace_ms = trace_start.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "wormhole: cached {} ray samples in {trace_ms:.1} ms; submitted {:.1} M lane-steps ({:.1} M lane-steps/s)",
        ray_map.len(),
        submitted_lane_steps as f64 / 1_000_000.0,
        submitted_lane_steps as f64 / trace_ms / 1_000.0,
    );
    Some(ray_map)
}

#[derive(Clone, Copy, Debug)]
struct Rotation3 {
    rows: [[f32; 3]; 3],
}

impl Rotation3 {
    const IDENTITY: Self = Self {
        rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    fn around_axis(axis: [f32; 3], angle: f32) -> Self {
        let [x, y, z] = normalized(axis);
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;
        Self {
            rows: [
                [
                    cos + x * x * one_minus_cos,
                    x * y * one_minus_cos - z * sin,
                    x * z * one_minus_cos + y * sin,
                ],
                [
                    y * x * one_minus_cos + z * sin,
                    cos + y * y * one_minus_cos,
                    y * z * one_minus_cos - x * sin,
                ],
                [
                    z * x * one_minus_cos - y * sin,
                    z * y * one_minus_cos + x * sin,
                    cos + z * z * one_minus_cos,
                ],
            ],
        }
    }

    fn apply(self, value: [f32; 3]) -> [f32; 3] {
        let dot = |row: [f32; 3]| row[0] * value[0] + row[1] * value[1] + row[2] * value[2];
        [dot(self.rows[0]), dot(self.rows[1]), dot(self.rows[2])]
    }
}

#[derive(Clone, Copy)]
struct SkyTransform {
    origin: Rotation3,
    other: Rotation3,
}

impl SkyTransform {
    const IDENTITY: Self = Self {
        origin: Rotation3::IDENTITY,
        other: Rotation3::IDENTITY,
    };

    fn at(time: f32, angular_speed: f32) -> Self {
        Self {
            origin: Rotation3::around_axis([0.16, 0.94, 0.30], time * angular_speed),
            other: Rotation3::around_axis([-0.28, 0.73, 0.62], time * angular_speed * -0.72),
        }
    }

    fn apply(self, direction: [f32; 3], universe: u8) -> [f32; 3] {
        if universe == UNIVERSE_OTHER {
            self.other.apply(direction)
        } else {
            self.origin.apply(direction)
        }
    }
}

fn procedural_sky(direction: [f32; 3], universe: u8, time: f32) -> [f32; 3] {
    if universe == UNRESOLVED {
        return [0.006, 0.018, 0.024];
    }

    let (normal, seed) = if universe == UNIVERSE_OTHER {
        ([-0.31, 0.16, 0.94], 0x8d26_4f31)
    } else {
        ([0.18, 0.88, 0.44], 0x3ac5_9b7d)
    };
    let cloud = fbm(
        3.4 * direction[0] + 1.2 * direction[2],
        3.4 * direction[1] - 1.6 * direction[2],
        seed,
    );
    let dust = fbm(
        12.0 * direction[0] - 3.0 * direction[1],
        12.0 * direction[2] + 4.0 * direction[1],
        seed ^ 0xa14f_2c91,
    );
    let band_coordinate = direction[0] * normal[0]
        + direction[1] * normal[1]
        + direction[2] * normal[2]
        + 0.085 * (cloud - 0.5);
    let band = (-38.0 * band_coordinate * band_coordinate).exp();
    let dust_lane = 1.0
        - 0.84
            * (-310.0 * (band_coordinate + 0.015 * (dust - 0.5)).powi(2)).exp()
            * (0.30 + 0.70 * dust);

    let cell_x = (direction[0] * 720.0).floor() as i32;
    let cell_y = (direction[1] * 720.0).floor() as i32;
    let cell_z = (direction[2] * 720.0).floor() as i32;
    let hash = hash_u32(
        (cell_x as u32).wrapping_mul(0x9e37_79b9)
            ^ (cell_y as u32).wrapping_mul(0x85eb_ca6b)
            ^ (cell_z as u32).wrapping_mul(0xc2b2_ae35)
            ^ (universe as u32).wrapping_mul(0x27d4_eb2f),
    );
    let random = hash as f32 / u32::MAX as f32;
    let star = if random > 0.9974 {
        let energy = ((random - 0.9974) / 0.0026).powi(6) * 5.5;
        energy * (0.96 + 0.04 * (time * 2.0 + random * 97.0).sin())
    } else {
        0.0
    };
    let star_color = if hash & 3 == 0 {
        [0.72, 0.84, 1.0]
    } else if hash & 7 == 1 {
        [1.0, 0.72, 0.48]
    } else {
        [1.0, 0.94, 0.84]
    };

    if universe == UNIVERSE_OTHER {
        [
            0.0012 + 0.075 * band * dust_lane * (0.30 + 0.70 * cloud) + star * star_color[0],
            0.0018 + 0.13 * band * dust_lane * (0.35 + 0.65 * cloud) + star * star_color[1],
            0.0045 + 0.22 * band * dust_lane * (0.42 + 0.58 * cloud) + star * star_color[2],
        ]
    } else {
        [
            0.0018 + 0.17 * band * dust_lane * (0.35 + 0.65 * cloud) + star * star_color[0],
            0.0013 + 0.12 * band * dust_lane * (0.30 + 0.70 * cloud) + star * star_color[1],
            0.0032 + 0.10 * band * dust_lane * (0.45 + 0.55 * cloud) + star * star_color[2],
        ]
    }
}

struct SkyAtlases {
    origin: SkyTexture,
    other: SkyTexture,
}

impl SkyAtlases {
    fn generate(width: usize, height: usize, threads: usize) -> Self {
        Self {
            origin: SkyTexture::generate(width, height, threads, |direction| {
                procedural_sky(direction, UNIVERSE_ORIGIN, 0.0)
            }),
            other: SkyTexture::generate(width, height, threads, |direction| {
                procedural_sky(direction, UNIVERSE_OTHER, 0.0)
            }),
        }
    }

    #[inline]
    fn sample(&self, direction: [f32; 3], universe: u8) -> [f32; 3] {
        if universe == UNRESOLVED {
            [0.006, 0.018, 0.024]
        } else if universe == UNIVERSE_OTHER {
            self.other.sample_equirectangular(direction)
        } else {
            self.origin.sample_equirectangular(direction)
        }
    }
}

fn shade_pixel_range(
    config: RenderConfig,
    map: &[RaySample],
    skies: &SkyAtlases,
    transform: SkyTransform,
    framebuffer: &mut [u32],
    pixels: core::ops::Range<usize>,
) {
    assert!(pixels.end <= framebuffer.len());
    assert!(pixels.end <= map.len() / config.samples_per_pixel);
    let sample_start = pixels.start * config.samples_per_pixel;
    let sample_end = pixels.end * config.samples_per_pixel;
    shade_pixels(
        config,
        &map[sample_start..sample_end],
        skies,
        transform,
        &mut framebuffer[pixels],
    );
}

/// Shade one contiguous pixel band from its corresponding cached ray samples.
///
/// The slices are deliberately paired before workers are spawned: each worker
/// borrows one immutable ray-map band and one disjoint framebuffer band. No
/// sample or pixel staging buffers are allocated or copied on the frame path.
fn shade_pixels(
    config: RenderConfig,
    map: &[RaySample],
    skies: &SkyAtlases,
    transform: SkyTransform,
    framebuffer: &mut [u32],
) {
    assert_eq!(map.len(), framebuffer.len() * config.samples_per_pixel);
    let inverse_samples = 1.0 / config.samples_per_pixel as f32;
    for (pixel, samples) in framebuffer
        .iter_mut()
        .zip(map.chunks_exact(config.samples_per_pixel))
    {
        let mut color = [0.0; 3];
        for sample in samples {
            let direction = transform.apply(sample.direction, sample.universe);
            let value = skies.sample(direction, sample.universe);
            color[0] += value[0];
            color[1] += value[1];
            color[2] += value[2];
        }
        color[0] *= inverse_samples;
        color[1] *= inverse_samples;
        color[2] *= inverse_samples;
        *pixel = pack_aces(color);
    }
}

fn shade_map(
    config: RenderConfig,
    map: &[RaySample],
    skies: &SkyAtlases,
    time: f32,
    framebuffer: &mut [u32],
    threads: usize,
) {
    assert!(threads > 0);
    assert_eq!(framebuffer.len(), config.width * config.height);
    assert_eq!(map.len(), framebuffer.len() * config.samples_per_pixel);
    let transform = SkyTransform::at(time, config.sky_speed);
    let workers = threads.min(config.height);
    if workers == 1 {
        shade_pixels(config, map, skies, transform, framebuffer);
        return;
    }

    // Match axrd's threaded rendering examples: scanline bands preserve
    // sequential access to the cached projection and make framebuffer writes
    // disjoint without locks. Scoped workers also keep all borrows local to a
    // frame, avoiding shared ownership or unsafe pointer publication.
    let rows_per_band = config.height.div_ceil(workers);
    let pixels_per_band = rows_per_band * config.width;
    let samples_per_band = pixels_per_band * config.samples_per_pixel;
    std::thread::scope(|scope| {
        for (pixel_band, sample_band) in framebuffer
            .chunks_mut(pixels_per_band)
            .zip(map.chunks(samples_per_band))
        {
            scope.spawn(move || shade_pixels(config, sample_band, skies, transform, pixel_band));
        }
    });
}

fn write_ppm(path: &str, config: RenderConfig, framebuffer: &[u32]) {
    let mut bytes = format!("P6\n{} {}\n255\n", config.width, config.height).into_bytes();
    bytes.reserve(framebuffer.len() * 3);
    for pixel in framebuffer {
        bytes.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]);
    }
    std::fs::write(path, bytes).expect("write snapshot");
    println!("snapshot -> {path}");
}

fn main() {
    let config = RenderConfig::from_env();
    let available_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let shade_threads = env_or("AXGEOM_THREADS", available_threads);
    assert!(
        shade_threads > 0,
        "AXGEOM_THREADS must be greater than zero"
    );
    let headless = std::env::var_os("AXGEOM_HEADLESS").is_some();
    let snapshot = std::env::var("AXGEOM_SNAPSHOT").ok();
    let max_frames: Option<u64> = std::env::var("AXGEOM_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok());
    println!(
        "wormhole: {}x{}, {} spp, {} steps ({}/graph), tile {}, sky {:.3} rad/s, shade {} thread(s)",
        config.width,
        config.height,
        config.samples_per_pixel,
        config.max_steps,
        config.steps_per_graph,
        config.tile_rays,
        config.sky_speed,
        shade_threads.min(config.height),
    );

    let sky_started = Instant::now();
    let sky_width = env_or("AXGEOM_SKY_WIDTH", 2_048_usize);
    let sky_height = env_or("AXGEOM_SKY_HEIGHT", 1_024_usize);
    let skies = SkyAtlases::generate(sky_width, sky_height, shade_threads);
    println!(
        "wormhole: two {}x{} sky atlases generated in {:.1} ms on {} thread(s)",
        sky_width,
        sky_height,
        sky_started.elapsed().as_secs_f64() * 1_000.0,
        shade_threads.min(sky_height),
    );
    let mut framebuffer = vec![pack_aces([0.0005, 0.001, 0.002]); config.width * config.height];
    let mut window = if headless {
        None
    } else {
        let mut window = Window::new(
            "axiom interstellar - initializing wormhole renderer",
            config.width,
            config.height,
            WindowOptions {
                resize: true,
                scale: Scale::X1,
                ..WindowOptions::default()
            },
        )
        .expect("open window");
        let uncap = std::env::var("AXGEOM_UNCAP").is_ok_and(|value| value != "0");
        window.set_target_fps(if max_frames.is_some() || uncap { 0 } else { 60 });
        window
            .update_with_buffer(&framebuffer, config.width, config.height)
            .expect("present initial frame");
        Some(window)
    };

    let (mut runtime, backend_label) = runtime();
    let ray_map = if let Some(window) = window.as_mut() {
        let trace_epoch = Instant::now();
        let mut shaded_pixels = 0;
        trace_map(&mut runtime, &backend_label, config, |partial_map| {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                return false;
            }
            let complete_pixels = partial_map.len() / config.samples_per_pixel;
            shade_pixel_range(
                config,
                partial_map,
                &skies,
                SkyTransform::IDENTITY,
                &mut framebuffer,
                shaded_pixels..complete_pixels,
            );
            shaded_pixels = complete_pixels;
            let percent = partial_map.len() as f64 / config.sample_count() as f64 * 100.0;
            window.set_title(&format!(
                "axiom interstellar - tracing {percent:.1}% ({:.1}s)",
                trace_epoch.elapsed().as_secs_f64()
            ));
            window
                .update_with_buffer(&framebuffer, config.width, config.height)
                .is_ok()
        })
    } else {
        trace_map(&mut runtime, &backend_label, config, |_| true)
    };
    let Some(ray_map) = ray_map else {
        return;
    };

    if headless {
        shade_map(
            config,
            &ray_map,
            &skies,
            0.0,
            &mut framebuffer,
            shade_threads,
        );
        if let Some(path) = snapshot.as_deref() {
            write_ppm(path, config, &framebuffer);
        }
        return;
    }

    let mut window = window.expect("interactive window");
    let epoch = Instant::now();
    let mut frame = 0_u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0_u64;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        shade_map(
            config,
            &ray_map,
            &skies,
            epoch.elapsed().as_secs_f32(),
            &mut framebuffer,
            shade_threads,
        );
        window
            .update_with_buffer(&framebuffer, config.width, config.height)
            .expect("present frame");
        frame += 1;
        if last_title.elapsed().as_secs_f32() >= 0.5 {
            let fps = (frame - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axiom interstellar - cached wormhole geodesics - {fps:.1} fps - {}t",
                shade_threads.min(config.height),
            ));
            last_title = Instant::now();
            frames_at_title = frame;
        }
        if max_frames.is_some_and(|limit| frame >= limit) {
            break;
        }
    }
    let seconds = epoch.elapsed().as_secs_f32();
    println!(
        "wormhole: {frame} animated frames in {seconds:.2}s ({:.2} fps, {} thread(s))",
        frame as f32 / seconds.max(f32::EPSILON),
        shade_threads.min(config.height),
    );
    if let Some(path) = snapshot.as_deref() {
        write_ppm(path, config, &framebuffer);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RaySample, RenderConfig, Rotation3, SkyAtlases, SkyTransform, UNIVERSE_ORIGIN, env_or,
        shade_map,
    };

    fn skies() -> SkyAtlases {
        SkyAtlases::generate(64, 32, 1)
    }

    fn squared_length(value: [f32; 3]) -> f32 {
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
    }

    #[test]
    fn sky_transform_starts_at_identity_and_then_moves() {
        let direction = [0.37, -0.51, 0.7765305];
        let initial = SkyTransform::at(0.0, 0.18).origin.apply(direction);
        assert_eq!(initial, direction);

        let moved = SkyTransform::at(1.0, 0.18).origin.apply(direction);
        assert!(
            moved
                .iter()
                .zip(direction)
                .any(|(actual, expected)| (actual - expected).abs() > 1.0e-3)
        );
    }

    #[test]
    fn axis_rotation_preserves_direction_length() {
        let direction = [0.37, -0.51, 0.7765305];
        let rotated = Rotation3::around_axis([0.16, 0.94, 0.30], 1.7).apply(direction);
        assert!((squared_length(rotated) - squared_length(direction)).abs() < 2.0e-6);
    }

    #[test]
    fn cached_ray_map_produces_distinct_animated_frames() {
        let config = RenderConfig {
            width: 1,
            height: 1,
            tile_rays: 1,
            max_steps: 1,
            steps_per_graph: 1,
            samples_per_pixel: 1,
            step_size: 0.1,
            escape_ell: 10.0,
            camera_ell: 5.0,
            camera_yaw: 0.0,
            tan_half_fov: 0.62,
            sky_speed: 0.18,
        };
        let map = [RaySample {
            direction: [0.37, -0.51, 0.7765305],
            universe: UNIVERSE_ORIGIN,
        }];
        let mut initial = [0];
        let mut animated = [0];
        let skies = skies();
        shade_map(config, &map, &skies, 0.0, &mut initial, 1);
        shade_map(config, &map, &skies, 8.0, &mut animated, 1);
        assert_ne!(initial, animated);
    }

    #[test]
    fn threaded_shading_is_pixel_identical_to_single_threaded() {
        let config = RenderConfig {
            width: 7,
            height: 5,
            tile_rays: 64,
            max_steps: 1,
            steps_per_graph: 1,
            samples_per_pixel: 1,
            step_size: 0.1,
            escape_ell: 10.0,
            camera_ell: 5.0,
            camera_yaw: 0.0,
            tan_half_fov: 0.62,
            sky_speed: 0.18,
        };
        let map: Vec<_> = (0..config.sample_count())
            .map(|index| RaySample {
                direction: super::normalized([
                    0.2 + index as f32 * 0.013,
                    -0.7 + index as f32 * 0.007,
                    0.6,
                ]),
                universe: if index % 3 == 0 {
                    super::UNIVERSE_OTHER
                } else {
                    super::UNIVERSE_ORIGIN
                },
            })
            .collect();
        let mut serial = vec![0; config.width * config.height];
        let mut threaded = vec![0; serial.len()];
        let skies = skies();

        shade_map(config, &map, &skies, 3.25, &mut serial, 1);
        shade_map(config, &map, &skies, 3.25, &mut threaded, 3);

        assert_eq!(threaded, serial);
    }

    /// Manual full-resolution probe for the host-only animated phase. This is
    /// ignored in normal test runs because it intentionally allocates a 2K ray
    /// map and measures wall-clock time. Keeping graph tracing out of the probe
    /// isolates the scaling provided by `AXGEOM_THREADS`.
    #[test]
    #[ignore = "manual 2K host-shading performance probe"]
    fn shading_thread_scaling_probe() {
        use std::hint::black_box;
        use std::time::Instant;

        let config = RenderConfig {
            width: 2560,
            height: 1440,
            tile_rays: 8192,
            max_steps: 512,
            steps_per_graph: 64,
            samples_per_pixel: 1,
            step_size: 0.1,
            escape_ell: 10.0,
            camera_ell: 5.0,
            camera_yaw: 0.0,
            tan_half_fov: 0.62,
            sky_speed: 0.18,
        };
        let map: Vec<_> = (0..config.sample_count())
            .map(|index| {
                let x = (index % config.width) as f32 / config.width as f32 * 2.0 - 1.0;
                let y = (index / config.width) as f32 / config.height as f32 * 2.0 - 1.0;
                RaySample {
                    direction: super::normalized([x, y, 1.0]),
                    universe: if index & 1 == 0 {
                        super::UNIVERSE_OTHER
                    } else {
                        super::UNIVERSE_ORIGIN
                    },
                }
            })
            .collect();
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let skies = SkyAtlases::generate(2_048, 1_024, available);
        let frames = env_or("AXGEOM_PROBE_FRAMES", 3_u32).max(1);

        for threads in [1, available] {
            let mut framebuffer = vec![0; config.width * config.height];
            let start = Instant::now();
            for frame in 0..frames {
                shade_map(
                    config,
                    &map,
                    &skies,
                    frame as f32 / 60.0,
                    &mut framebuffer,
                    threads,
                );
            }
            let seconds = start.elapsed().as_secs_f64();
            black_box(&framebuffer);
            println!(
                "2K host shade: {threads} thread(s), {:.2} fps, {:.1} ms/frame",
                frames as f64 / seconds,
                seconds * 1_000.0 / frames as f64,
            );
        }
    }
}
