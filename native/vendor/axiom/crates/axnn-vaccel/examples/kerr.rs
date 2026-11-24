//! Cached Kerr boundary transport driven by the backend-neutral `.ax` step
//! graph, with a mixed-precision emissive voxel model for the accretion flow.
//!
//! The one-step graph is compiled once and repeatedly submitted over ray
//! tiles. Capture, thin-disk, and escape masks terminate lanes independently.
//! The expensive geodesic result is retained as a boundary map; animation
//! transforms those records into a stationary material-space plasma volume
//! and never retraces an unchanged camera/metric.
//!
//! ```text
//! AXGEOM_SNAPSHOT=kerr.ppm \
//!   cargo +nightly run --release -p axnn-vaccel --example kerr --features openvino
//! ```
//! AMD XDNA is an explicit, narrower tier:
//!
//! ```text
//! AXGEOM_ACCELERATOR=xdna cargo +nightly run --release \
//!   -p axnn-vaccel --example kerr --features xdna
//! ```
//!
//! That route runs a measured BF16 identity probe over a real Kerr ray plane,
//! then keeps the unsupported FP32 Kerr step graph on the CPU. It does not
//! claim general FP32 XDNA execution.
//! Use a `{frame}` placeholder in `AXGEOM_SNAPSHOT` together with
//! `AXGEOM_FRAMES` to write a cinematic image sequence without retracing.
//! `AXGEOM_PLASMA_STORAGE` selects `uf8e4m4` (default), `uf8e5m3`, or the
//! legacy signed `B8` for a whole-example CPU material-storage/performance
//! A/B after tracing. It does not add a UF dtype to the vAccel graph/provider
//! ABI.
//! `AXGEOM_PLASMA_MODE=surface` is an explicit interactive quality tier that
//! retains the relativistic equatorial transport boundary and orbital heating
//! while omitting the exact atmospheric volume integral. It caches the equatorial
//! plasma projection, bounded spectral transfer, and retarded sky phase for
//! the animated host loop. The default `volume` tier remains the uncached
//! reference path used for final frames. `AXGEOM_SKY_WIDTH` and
//! `AXGEOM_SKY_HEIGHT` size the generated sky atlas (default: 2048x1024).
//! The volume tier builds a flared, pressure-supported flow (H/R about
//! 0.12..0.23) in a near-isotropic 96x96x48 voxel domain by default;
//! `AXGEOM_PLASMA_HALF_THICKNESS` controls its vertical enclosure.
//! `AXGEOM_BOUNDARY_SPP` (default 4) and `AXGEOM_BOUNDARY_STEP_SCALE`
//! (default 0.5) selectively retrace only terminal-class edge neighborhoods
//! for higher-precision cached reprojection. Set boundary SPP to 1 to disable.
//! On Apple Silicon, replace `openvino` with `coreml` to use the Core ML host
//! backend. Automatic compilation discovers provider capabilities and selects
//! placement; Core ML Boolean ABI boundaries use lossless FP32 bridges.

#![feature(allocator_api)]

use std::alloc::Global;
use std::time::Instant;

use axgeom::{
    black_hole::{
        CameraTetrad, HamiltonianRay, KerrDiskBoundary, KerrParams, OrbitDirection, TraceConfig,
        advance_infalling_observer, camera_null_ray, camera_past_null_ray,
        circular_equatorial_specific_kinetic_energy, disk_boundary_transport, hamiltonian_rhs,
        horizon_radius, midpoint_step, radius as kerr_radius, static_camera_tetrad,
        zero_angular_momentum_infalling_observer, zero_torque_thin_disk_flux,
    },
    kerr_graph::{KerrStepGraphSpec, abi, build_kerr_step_graph_in},
};
use axnn_cpu::{
    fp8,
    radiation::{self, BlackbodyRadiance},
    uf, voxel,
};
use axnn_vaccel::{
    ACCEL_BACKEND, Accelerator, AxDeviceModel, AxHetero, AxTosaAccelerator, Session,
};
#[cfg(feature = "window")]
use minifb::{Key, Scale, Window, WindowOptions};

#[path = "common/cinematic.rs"]
mod cinematic;
use cinematic::{SkyTexture, fbm, hash_u32, pack_aces};
use virtio_accel_tosa::TosaCapabilityProvider;

const DEFAULT_WIDTH: usize = 640;
const DEFAULT_HEIGHT: usize = 360;
const DEFAULT_TILE_RAYS: usize = 4_096;
const DEFAULT_STEPS: u32 = 320;
const DEFAULT_PLASMA_XY: u32 = 96;
const DEFAULT_PLASMA_Z: u32 = 48;

#[derive(Clone, Copy, Debug)]
struct RenderConfig {
    width: usize,
    height: usize,
    tile_rays: usize,
    max_steps: u32,
    samples_per_pixel: usize,
    step_size: f32,
    tan_half_fov: f32,
    threads: usize,
    plasma_xy: u32,
    plasma_z: u32,
    plasma_half_thickness: f32,
    plasma_extinction: f32,
    plasma_emission: f32,
    plasma_mode: PlasmaMode,
    plasma_time_scale: f32,
    sky_speed: f32,
    exposure: f32,
    boundary_spp: usize,
    boundary_step_scale: f32,
}

impl RenderConfig {
    fn from_env() -> Self {
        let config = Self {
            width: env_or("AXGEOM_WIDTH", DEFAULT_WIDTH),
            height: env_or("AXGEOM_HEIGHT", DEFAULT_HEIGHT),
            tile_rays: env_or("AXGEOM_TILE_RAYS", DEFAULT_TILE_RAYS),
            max_steps: env_or("AXGEOM_STEPS", DEFAULT_STEPS),
            samples_per_pixel: env_or("AXGEOM_SPP", 1),
            step_size: env_or("AXGEOM_STEP_SIZE", 0.18),
            tan_half_fov: env_or("AXGEOM_TAN_HALF_FOV", 0.58),
            threads: env_or(
                "AXGEOM_THREADS",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            ),
            plasma_xy: env_or("AXGEOM_PLASMA_XY", DEFAULT_PLASMA_XY),
            plasma_z: env_or("AXGEOM_PLASMA_Z", DEFAULT_PLASMA_Z),
            // 96x96x48 over a 30x30x15 domain keeps the default voxels nearly
            // isotropic. The old +/-1.45 slab clipped a substantial fraction
            // of the flared atmosphere and visually collapsed it into the
            // equatorial event surface.
            plasma_half_thickness: env_or("AXGEOM_PLASMA_HALF_THICKNESS", 7.5),
            plasma_extinction: env_or("AXGEOM_PLASMA_EXTINCTION", 0.32),
            plasma_emission: env_or("AXGEOM_PLASMA_EMISSION", 0.85),
            plasma_mode: PlasmaMode::from_env(),
            plasma_time_scale: env_or("AXGEOM_PLASMA_TIME_SCALE", 5.0),
            sky_speed: env_or("AXGEOM_SKY_SPEED", 0.035),
            exposure: env_or("AXGEOM_EXPOSURE", 0.45),
            boundary_spp: env_or("AXGEOM_BOUNDARY_SPP", 4),
            boundary_step_scale: env_or("AXGEOM_BOUNDARY_STEP_SCALE", 0.5),
        };
        assert!(config.width > 0 && config.height > 0);
        assert!((1..=8_192).contains(&config.tile_rays));
        assert!(config.max_steps > 0);
        assert!(config.samples_per_pixel > 0);
        assert!(config.step_size.is_finite() && config.step_size > 0.0);
        assert!(config.tan_half_fov.is_finite() && config.tan_half_fov > 0.0);
        assert!(config.threads > 0);
        assert!((8..=256).contains(&config.plasma_xy));
        assert!((4..=128).contains(&config.plasma_z));
        assert!(config.plasma_half_thickness.is_finite() && config.plasma_half_thickness > 0.0);
        assert!(config.plasma_extinction.is_finite() && config.plasma_extinction >= 0.0);
        assert!(config.plasma_emission.is_finite() && config.plasma_emission >= 0.0);
        assert!(config.plasma_time_scale.is_finite() && config.sky_speed.is_finite());
        assert!(config.exposure.is_finite() && config.exposure > 0.0);
        let boundary_side = (config.boundary_spp as f32).sqrt() as usize;
        assert!(
            config.boundary_spp == 1
                || boundary_side * boundary_side == config.boundary_spp
                    && config.boundary_spp <= 16,
            "AXGEOM_BOUNDARY_SPP must be 1, 4, 9, or 16",
        );
        assert!(
            config.boundary_step_scale.is_finite()
                && (0.0..=1.0).contains(&config.boundary_step_scale)
                && config.boundary_step_scale > 0.0,
        );
        config
    }

    fn sample_count(self) -> usize {
        self.width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(self.samples_per_pixel))
            .expect("requested image is too large")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlasmaMode {
    Volume,
    Surface,
}

impl PlasmaMode {
    fn parse(value: Option<&str>) -> Result<Self, &str> {
        match value {
            None | Some("volume") => Ok(Self::Volume),
            Some("surface") => Ok(Self::Surface),
            Some(value) => Err(value),
        }
    }

    fn from_env() -> Self {
        let requested = std::env::var("AXGEOM_PLASMA_MODE").ok();
        Self::parse(requested.as_deref()).unwrap_or_else(|value| {
            panic!("AXGEOM_PLASMA_MODE must be volume or surface; got {value:?}")
        })
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Volume => "exact volume",
            Self::Surface => "live surface",
        }
    }
}

fn env_or<T: core::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).expect("axtosa CPU session")
}

#[cfg(not(feature = "xdna"))]
fn reject_uncompiled_xdna_selection() {
    if std::env::var("AXGEOM_ACCELERATOR").is_ok_and(|value| value == "xdna") {
        panic!(
            "AXGEOM_ACCELERATOR=xdna requested, but axnn-vaccel was built without the xdna feature"
        );
    }
}

#[cfg(all(feature = "xdna", axnn_vaccel_xdna_runtime))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, virtio_accel_xdna::XdnaAccelerator>,
    String,
    AxDeviceModel,
) {
    let requested = std::env::var("AXGEOM_ACCELERATOR").unwrap_or_else(|_| "cpu".to_owned());
    match requested.as_str() {
        "cpu" => (
            AxHetero::new(cpu_session(), None),
            "CPU (XDNA feature available but not selected)".to_owned(),
            AxDeviceModel::default(),
        ),
        "xdna" => {
            let provider = virtio_accel_xdna::XdnaAccelerator::new().unwrap_or_else(|error| {
                panic!("AXGEOM_ACCELERATOR=xdna requested, but XDNA initialization failed: {error}")
            });
            let info = provider.device_info().unwrap_or_else(|error| {
                panic!("AXGEOM_ACCELERATOR=xdna requested, but device identity failed: {error:?}")
            });
            let session = Session::new_with_buffer_alignment(
                provider,
                virtio_accel_xdna::XDNA_TOSA_TARGET,
                4 * 1024,
            )
            .unwrap_or_else(|error| {
                panic!("AXGEOM_ACCELERATOR=xdna requested, but session creation failed: {error:?}")
            });
            let model = AxDeviceModel::default()
                .with_accelerator_tosa_capability(session.provider(), session.target());
            (
                AxHetero::new(cpu_session(), Some(session)),
                format!(
                    "CPU FP32 Kerr + XDNA BF16 probe ({:04x}:{:04x} {:?})",
                    info.identity.vendor_id, info.identity.device_id, info.identity.class
                ),
                model,
            )
        }
        value => panic!("AXGEOM_ACCELERATOR must be cpu or xdna; got {value:?}"),
    }
}

#[cfg(all(feature = "xdna", not(axnn_vaccel_xdna_runtime)))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, AxTosaAccelerator>,
    String,
    AxDeviceModel,
) {
    let requested = std::env::var("AXGEOM_ACCELERATOR").unwrap_or_else(|_| "cpu".to_owned());
    match requested.as_str() {
        "cpu" => (
            AxHetero::new(cpu_session(), None),
            "CPU (XDNA native HRX runtime not built)".to_owned(),
            AxDeviceModel::default(),
        ),
        "xdna" => panic!(
            "AXGEOM_ACCELERATOR=xdna requested, but axnn-vaccel was built without a detected amdxdna-native HRX runtime"
        ),
        value => panic!("AXGEOM_ACCELERATOR must be cpu or xdna; got {value:?}"),
    }
}

#[cfg(all(
    not(feature = "xdna"),
    feature = "openvino",
    axnn_vaccel_openvino_runtime
))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>,
    String,
    AxDeviceModel,
) {
    reject_uncompiled_xdna_selection();
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
    (
        AxHetero::new(cpu_session(), accelerator),
        label,
        AxDeviceModel::default(),
    )
}

#[cfg(all(
    not(feature = "xdna"),
    not(all(feature = "openvino", axnn_vaccel_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, virtio_accel_coreml::CoreMlAccelerator>,
    String,
    AxDeviceModel,
) {
    reject_uncompiled_xdna_selection();
    let accelerator = virtio_accel_coreml::CoreMlAccelerator::new_tosa()
        .ok()
        .and_then(|provider| Session::new(provider, virtio_accel_coreml::COREML_TOSA_TARGET).ok());
    let label = if accelerator.is_some() {
        "Core ML / Neural Engine".to_owned()
    } else {
        "CPU (Core ML unavailable)".to_owned()
    };
    (
        AxHetero::new(cpu_session(), accelerator),
        label,
        AxDeviceModel::coreml(),
    )
}

#[cfg(not(any(
    feature = "xdna",
    all(feature = "openvino", axnn_vaccel_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn runtime() -> (
    AxHetero<AxTosaAccelerator, AxTosaAccelerator>,
    String,
    AxDeviceModel,
) {
    reject_uncompiled_xdna_selection();
    (
        AxHetero::new(cpu_session(), None),
        "CPU".to_owned(),
        AxDeviceModel::default(),
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("f32 plane"))
}

/// Reconstruct the terminal equatorial crossing in host F64 before reducing
/// it to the cached F32 boundary record.
///
/// Providers classify the event in the all-F32 step graph, but the cached
/// hologram is reused for every later frame. Paying for one higher-precision
/// interpolation at that projection boundary prevents the graph's reciprocal
/// and multiply rounding from being baked permanently into the disk edge.
fn reproject_disk_crossing_f64(
    params: &KerrParams,
    trace: &TraceConfig,
    from: [f32; 4],
    to: [f32; 4],
) -> Option<([f32; 3], f32, f64)> {
    let z0 = f64::from(from[3]);
    let z1 = f64::from(to[3]);
    if !((z0 > 0.0 && z1 <= 0.0) || (z0 < 0.0 && z1 >= 0.0)) {
        return None;
    }
    let denominator = z0 - z1;
    if !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let fraction = (z0 / denominator).clamp(0.0, 1.0);
    let interpolate = |component: usize| {
        f64::from(from[component])
            + fraction * (f64::from(to[component]) - f64::from(from[component]))
    };
    let x = interpolate(1);
    let y = interpolate(2);
    // At z=0 the Cartesian Kerr-Schild radius reduces exactly to
    // sqrt(max(x^2+y^2-a^2, 0)). Keep that boundary decision in F64 too.
    let spin = f64::from(params.spin);
    let radius = (x * x + y * y - spin * spin).max(0.0).sqrt();
    if !radius.is_finite()
        || radius < f64::from(trace.disk_inner_radius)
        || radius > f64::from(trace.disk_outer_radius)
    {
        return None;
    }
    Some(([x as f32, y as f32, 0.0], radius as f32, fraction))
}

#[inline]
fn interpolate_f64_to_f32(from: f32, to: f32, fraction: f64) -> f32 {
    (f64::from(from) + fraction * (f64::from(to) - f64::from(from))) as f32
}

fn normalized(vector: [f32; 3]) -> [f32; 3] {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        .sqrt()
        .max(1.0e-12);
    [vector[0] / norm, vector[1] / norm, vector[2] / norm]
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
    (
        ((seed & 0xffff) as f32 + 0.5) / 65_536.0,
        ((seed >> 16) as f32 + 0.5) / 65_536.0,
    )
}

#[derive(Clone, Copy, Debug)]
enum BoundarySample {
    Capture,
    Disk(KerrDiskBoundary),
    Sky {
        direction: [f32; 3],
        retarded_sin: f32,
        retarded_cos: f32,
        retarded_uv: [f32; 2],
    },
    Unresolved,
    InteriorTerminated,
}

impl BoundarySample {
    fn sky(direction: [f32; 3], travel_time: f32, angular_speed: f32) -> Self {
        let (retarded_sin, retarded_cos) = (-0.05 * travel_time * angular_speed).sin_cos();
        let retarded_direction = rotate_z(direction, retarded_sin, retarded_cos);
        let retarded_uv = [
            retarded_direction[1].atan2(retarded_direction[0]) * (0.5 / core::f32::consts::PI)
                + 0.5,
            retarded_direction[2].clamp(-1.0, 1.0).acos() / core::f32::consts::PI,
        ];
        Self::Sky {
            direction,
            retarded_sin,
            retarded_cos,
            retarded_uv,
        }
    }
}

struct TileState {
    position: [Vec<u8>; 4],
    momentum: [Vec<u8>; 4],
    active: Vec<u8>,
}

impl TileState {
    fn inputs(&self) -> [&[u8]; 9] {
        [
            &self.position[0],
            &self.position[1],
            &self.position[2],
            &self.position[3],
            &self.momentum[0],
            &self.momentum[1],
            &self.momentum[2],
            &self.momentum[3],
            &self.active,
        ]
    }
}

fn camera_ray_for_pixel(
    config: RenderConfig,
    params: &KerrParams,
    camera: &CameraTetrad,
    past_directed: bool,
    pixel: usize,
    jitter: (f32, f32),
) -> HamiltonianRay {
    let x = pixel % config.width;
    let y = pixel / config.width;
    let aspect = config.width as f32 / config.height as f32;
    let screen = [
        ((x as f32 + jitter.0) / config.width as f32 * 2.0 - 1.0) * config.tan_half_fov * aspect,
        -((y as f32 + jitter.1) / config.height as f32 * 2.0 - 1.0) * config.tan_half_fov,
    ];
    if past_directed {
        camera_past_null_ray(params, camera, screen)
    } else {
        camera_null_ray(params, camera, screen)
    }
    .expect("valid camera ray")
}

fn initialize_tile(
    config: RenderConfig,
    params: &KerrParams,
    camera: &CameraTetrad,
    past_directed: bool,
    first_sample: usize,
) -> (TileState, usize) {
    let count = (config.sample_count() - first_sample).min(config.tile_rays);
    let mut position = std::array::from_fn(|_| vec![0.0; config.tile_rays]);
    let mut momentum = std::array::from_fn(|_| vec![0.0; config.tile_rays]);
    let mut active = vec![0; config.tile_rays];

    for lane in 0..count {
        let sample_index = first_sample + lane;
        let pixel = sample_index / config.samples_per_pixel;
        let sub_sample = sample_index % config.samples_per_pixel;
        let ray = camera_ray_for_pixel(
            config,
            params,
            camera,
            past_directed,
            pixel,
            sample_jitter(sub_sample, pixel),
        );
        for component in 0..4 {
            position[component][lane] = ray.position[component];
            momentum[component][lane] = ray.momentum[component];
        }
        active[lane] = 1;
    }

    (
        TileState {
            position: position.map(|plane| f32_bytes(&plane)),
            momentum: momentum.map(|plane| f32_bytes(&plane)),
            active,
        },
        count,
    )
}

#[cfg(all(feature = "xdna", axnn_vaccel_xdna_runtime))]
fn percentile(samples: &mut [std::time::Duration], percentile: usize) -> std::time::Duration {
    samples.sort_unstable();
    let index = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len().saturating_sub(1));
    samples[index]
}

#[cfg(all(feature = "xdna", axnn_vaccel_xdna_runtime))]
fn run_xdna_kerr_probe(
    runtime: &mut AxHetero<AxTosaAccelerator, virtio_accel_xdna::XdnaAccelerator>,
    config: RenderConfig,
    params: &KerrParams,
    camera: &CameraTetrad,
) {
    let Some(session) = runtime.accel.as_mut() else {
        return;
    };
    const FP32_NEGATIVE_KEY: u64 = u64::MAX - 1;
    const BF16_PROBE_KEY: u64 = u64::MAX;

    let (state, kerr_values) = initialize_tile(config, params, camera, false, 0);
    let elements = config
        .tile_rays
        .div_ceil(axnn_vaccel::xdna::BF16_IDENTITY_LINE_ELEMENTS)
        * axnn_vaccel::xdna::BF16_IDENTITY_LINE_ELEMENTS;
    // The camera origin is identical for every ray, while this momentum plane
    // varies across screen-space lanes and therefore exercises a meaningful
    // distribution of real Kerr input values.
    let mut fp32 = state.momentum[1].clone();
    fp32.resize(elements * 4, 0);
    let bf16 = axnn_vaccel::xdna::f32_bytes_to_bf16_rne(&fp32)
        .expect("narrow Kerr position plane to BF16");

    let unsupported = axnn_vaccel::xdna::unsupported_fp32_identity(elements)
        .expect("author FP32 XDNA negative control");
    let rejection = session
        .ensure(FP32_NEGATIVE_KEY, || unsupported)
        .expect_err("XDNA must reject FP32 identity");
    assert_eq!(
        rejection,
        axnn_vaccel::BackendError::Unsupported,
        "XDNA returned the wrong fail-loud class for FP32 arithmetic"
    );
    println!(
        "kerr/xdna: FP32 identity rejected as {rejection:?}; general FP32 Kerr arithmetic remains unsupported"
    );

    let authored = Instant::now();
    let artifact =
        axnn_vaccel::xdna::bf16_identity(elements).expect("author XDNA BF16 identity probe");
    let authoring = authored.elapsed();
    let loading = Instant::now();
    session
        .ensure(BF16_PROBE_KEY, || artifact.clone())
        .expect("compile/load XDNA BF16 identity probe");
    let program_load = loading.elapsed();

    let provider = session.provider();
    let direct_before = provider.direct_binding_admissions();
    let transfer_before = provider.explicit_transfer_bytes();
    let cold = session
        .run_many_profiled(BF16_PROBE_KEY, || artifact.clone(), &[&bf16], &[bf16.len()])
        .expect("run cold XDNA BF16 Kerr probe");
    assert_eq!(
        cold.outputs,
        [bf16.clone()],
        "cold BF16 identity changed bits"
    );

    let warmups = env_or("AXGEOM_XDNA_PROBE_WARMUPS", 5_usize);
    let measured = env_or("AXGEOM_XDNA_PROBE_SAMPLES", 20_usize).max(1);
    for _ in 0..warmups {
        let run = session
            .run_many_profiled(BF16_PROBE_KEY, || artifact.clone(), &[&bf16], &[bf16.len()])
            .expect("warm up XDNA BF16 Kerr probe");
        assert_eq!(
            run.outputs,
            [bf16.clone()],
            "warm BF16 identity changed bits"
        );
    }

    let mut submit = Vec::with_capacity(measured);
    let mut total = Vec::with_capacity(measured);
    for _ in 0..measured {
        let run = session
            .run_many_profiled(BF16_PROBE_KEY, || artifact.clone(), &[&bf16], &[bf16.len()])
            .expect("measure XDNA BF16 Kerr probe");
        assert_eq!(
            run.outputs,
            [bf16.clone()],
            "measured BF16 identity changed bits"
        );
        submit.push(run.timings.submit);
        total.push(run.timings.total());
    }

    let executions = 1 + warmups + measured;
    let direct = session
        .provider()
        .direct_binding_admissions()
        .saturating_sub(direct_before);
    let transfer = session
        .provider()
        .explicit_transfer_bytes()
        .saturating_sub(transfer_before);
    assert_eq!(direct, (executions * 2) as u64);
    assert_eq!(transfer, (executions * bf16.len() * 2) as u64);

    let original: Vec<f32> = fp32
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let narrowed =
        axnn_vaccel::xdna::bf16_bytes_to_f32(&bf16).expect("widen BF16 Kerr probe oracle");
    let worst_abs = original[..kerr_values]
        .iter()
        .zip(&narrowed[..kerr_values])
        .map(|(&fp32, &bf16)| (fp32 - bf16).abs())
        .fold(0.0_f32, f32::max);

    let mut submit_p50 = submit.clone();
    let mut submit_p95 = submit.clone();
    let mut total_p50 = total.clone();
    let mut total_p95 = total.clone();
    let submit_worst = *submit.iter().max().expect("measured samples");
    let total_worst = *total.iter().max().expect("measured samples");
    println!(
        "kerr/xdna: BF16 tier probe used {kerr_values} Kerr x-momentum values padded to {elements} ({:.1} KiB), bit-exact identity, max host FP32→BF16 abs error {worst_abs:.6}",
        bf16.len() as f64 / 1024.0,
    );
    println!(
        "kerr/xdna: artifact authoring {:.3} ms, program load/compile {:.3} ms; cold prepare/upload/submit/readback {:.3}/{:.3}/{:.3}/{:.3} ms",
        authoring.as_secs_f64() * 1_000.0,
        program_load.as_secs_f64() * 1_000.0,
        cold.timings.preparation.as_secs_f64() * 1_000.0,
        cold.timings.upload.as_secs_f64() * 1_000.0,
        cold.timings.submit.as_secs_f64() * 1_000.0,
        cold.timings.readback.as_secs_f64() * 1_000.0,
    );
    println!(
        "kerr/xdna: {measured} steady samples after {warmups} warmups; submit p50/p95/worst {:.3}/{:.3}/{:.3} ms, total p50/p95/worst {:.3}/{:.3}/{:.3} ms",
        percentile(&mut submit_p50, 50).as_secs_f64() * 1_000.0,
        percentile(&mut submit_p95, 95).as_secs_f64() * 1_000.0,
        submit_worst.as_secs_f64() * 1_000.0,
        percentile(&mut total_p50, 50).as_secs_f64() * 1_000.0,
        percentile(&mut total_p95, 95).as_secs_f64() * 1_000.0,
        total_worst.as_secs_f64() * 1_000.0,
    );
    println!(
        "kerr/xdna: {direct} direct bindings, zero submission-time staging, {transfer} explicit upload+readback bytes; device node /dev/accel/accel0={}",
        std::path::Path::new("/dev/accel/accel0").exists(),
    );
}

#[allow(clippy::too_many_arguments)]
fn trace_scene<C: Accelerator, A: Accelerator + TosaCapabilityProvider>(
    runtime: &mut AxHetero<C, A>,
    backend_label: &str,
    device_model: &AxDeviceModel,
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    camera: &CameraTetrad,
    termination_radius: f32,
    past_directed: bool,
    mut on_progress: impl FnMut(&[BoundarySample]) -> bool,
) -> Option<Vec<BoundarySample>> {
    let graph_bytes = build_kerr_step_graph_in(
        KerrStepGraphSpec {
            rays: config.tile_rays as u32,
            params: *params,
            trace: *trace,
            termination_radius,
        },
        Global,
    )
    .expect("build Kerr .ax graph");
    let graph = graph_bytes.parse().expect("parse Kerr .ax graph");
    let compile_start = Instant::now();
    let executable = runtime
        .compile_in(&graph, device_model, Global)
        .expect("compile Kerr .ax graph");
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1_000.0;
    assert_eq!(
        executable.output_ids(),
        [
            abi::X_T_OUT,
            abi::X_X_OUT,
            abi::X_Y_OUT,
            abi::X_Z_OUT,
            abi::P_T_OUT,
            abi::P_X_OUT,
            abi::P_Y_OUT,
            abi::P_Z_OUT,
            abi::ACTIVE_OUT,
            abi::CAPTURE_OUT,
            abi::DISK_OUT,
            abi::ESCAPE_OUT,
            abi::DISK_X_OUT,
            abi::DISK_Y_OUT,
            abi::DISK_Z_OUT,
            abi::DISK_RADIUS_OUT,
            abi::DISK_FRACTION_OUT,
        ]
    );
    let accelerator_nodes = executable
        .schedule()
        .placements()
        .iter()
        .filter(|&&backend| backend == ACCEL_BACKEND)
        .count();
    println!(
        "kerr: {} nodes, {} partition(s), {accelerator_nodes} accelerator node(s); runtime {backend_label}; compiled in {compile_ms:.1} ms",
        graph.node_count(),
        executable.schedule().plan().partitions.len(),
    );

    let start = Instant::now();
    let mut samples = Vec::with_capacity(config.sample_count());
    let mut submitted_lane_steps = 0_u64;
    while samples.len() < config.sample_count() {
        let first_sample = samples.len();
        let (mut state, count) =
            initialize_tile(config, params, camera, past_directed, first_sample);
        let mut tile_samples = vec![None; count];

        for _ in 0..config.max_steps {
            let outputs = runtime
                .run_in(&executable, &state.inputs(), Global)
                .expect("run Kerr .ax graph");
            submitted_lane_steps = submitted_lane_steps.saturating_add(config.tile_rays as u64);
            let mut outputs = outputs.into_iter();
            let position: [Vec<u8>; 4] =
                std::array::from_fn(|_| outputs.next().expect("position output"));
            let momentum: [Vec<u8>; 4] =
                std::array::from_fn(|_| outputs.next().expect("momentum output"));
            let active = outputs.next().expect("active output");
            let capture = outputs.next().expect("capture output");
            let disk = outputs.next().expect("disk output");
            let escape = outputs.next().expect("escape output");
            let disk_x = outputs.next().expect("disk x output");
            let disk_y = outputs.next().expect("disk y output");
            let _disk_z = outputs.next().expect("disk z output");
            let disk_radius = outputs.next().expect("disk radius output");
            let disk_fraction = outputs.next().expect("disk fraction output");
            assert!(outputs.next().is_none());

            for lane in 0..count {
                if tile_samples[lane].is_some() {
                    continue;
                }
                let ray = || {
                    HamiltonianRay::new(
                        std::array::from_fn(|component| f32_at(&position[component], lane)),
                        std::array::from_fn(|component| f32_at(&momentum[component], lane)),
                    )
                };
                tile_samples[lane] = if capture[lane] != 0 {
                    Some(BoundarySample::Capture)
                } else if disk[lane] != 0 {
                    let from_position =
                        std::array::from_fn(|component| f32_at(&state.position[component], lane));
                    let to_position =
                        std::array::from_fn(|component| f32_at(&position[component], lane));
                    let (point, radius, fraction) =
                        reproject_disk_crossing_f64(params, trace, from_position, to_position)
                            .unwrap_or_else(|| {
                                (
                                    [f32_at(&disk_x, lane), f32_at(&disk_y, lane), 0.0],
                                    f32_at(&disk_radius, lane),
                                    f64::from(f32_at(&disk_fraction, lane).clamp(0.0, 1.0)),
                                )
                            });
                    let momentum = std::array::from_fn(|component| {
                        let from = f32_at(&state.momentum[component], lane);
                        let to = f32_at(&momentum[component], lane);
                        interpolate_f64_to_f32(from, to, fraction)
                    });
                    let boundary_momentum = if past_directed {
                        momentum.map(|component| -component)
                    } else {
                        momentum
                    };
                    let coordinate_time =
                        interpolate_f64_to_f32(from_position[0], to_position[0], fraction);
                    disk_boundary_transport(
                        params,
                        trace,
                        camera,
                        point,
                        radius,
                        boundary_momentum,
                        coordinate_time,
                    )
                    .map(BoundarySample::Disk)
                    // A past-light-cone disk crossing can fail the exterior
                    // cache reducer's frequency-domain preconditions. Keep
                    // that explicit instead of fabricating an emissive hit.
                    .or(Some(if past_directed {
                        BoundarySample::InteriorTerminated
                    } else {
                        BoundarySample::Unresolved
                    }))
                } else if escape[lane] != 0 {
                    let ray = ray();
                    let flow = hamiltonian_rhs(params, &ray, trace.gradient_epsilon);
                    Some(BoundarySample::sky(
                        normalized([flow.position[1], flow.position[2], flow.position[3]]),
                        ray.position[0].abs(),
                        config.sky_speed,
                    ))
                } else {
                    None
                };
            }

            state = TileState {
                position,
                momentum,
                active,
            };
            if state.active[..count].iter().all(|&lane| lane == 0) {
                break;
            }
        }

        samples.extend(tile_samples.into_iter().map(|sample| {
            sample.unwrap_or(if past_directed {
                BoundarySample::InteriorTerminated
            } else {
                BoundarySample::Unresolved
            })
        }));
        let completed = samples.len();
        if completed == config.sample_count()
            || completed % (config.tile_rays * 16) < config.tile_rays
        {
            println!("kerr: traced {completed}/{} rays", config.sample_count());
            if !on_progress(&samples) {
                println!(
                    "kerr: trace cancelled at {completed}/{} rays",
                    config.sample_count()
                );
                return None;
            }
        }
    }
    let trace_ms = start.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "kerr: traced {} rays in {trace_ms:.1} ms; submitted {:.1} M lane-steps ({:.1} M lane-steps/s)",
        samples.len(),
        submitted_lane_steps as f64 / 1_000_000.0,
        submitted_lane_steps as f64 / trace_ms / 1_000.0,
    );
    Some(samples)
}

#[inline]
fn boundary_class(sample: BoundarySample) -> u8 {
    match sample {
        BoundarySample::Capture => 0,
        BoundarySample::Disk(_) => 1,
        BoundarySample::Sky { .. } => 2,
        BoundarySample::Unresolved => 3,
        BoundarySample::InteriorTerminated => 4,
    }
}

fn boundary_pixels(samples: &[BoundarySample], config: RenderConfig) -> Vec<usize> {
    if config.samples_per_pixel != 1 || config.boundary_spp == 1 {
        return Vec::new();
    }
    let mut pixels = Vec::new();
    for y in 0..config.height {
        for x in 0..config.width {
            let pixel = x + config.width * y;
            let class = boundary_class(samples[pixel]);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(config.width - 1);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(config.height - 1);
            let mut boundary = false;
            for neighbor_y in y0..=y1 {
                for neighbor_x in x0..=x1 {
                    let neighbor = neighbor_x + config.width * neighbor_y;
                    boundary |= boundary_class(samples[neighbor]) != class;
                }
            }
            if boundary {
                pixels.push(pixel);
            }
        }
    }
    pixels
}

fn ray_outward(params: &KerrParams, trace: &TraceConfig, ray: &HamiltonianRay) -> bool {
    let velocity = hamiltonian_rhs(params, ray, trace.gradient_epsilon).position;
    ray.position[1] * velocity[1] + ray.position[2] * velocity[2] + ray.position[3] * velocity[3]
        > 0.0
}

#[allow(clippy::too_many_arguments)]
fn trace_refined_boundary_ray(
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    camera: &CameraTetrad,
    termination_radius: f32,
    past_directed: bool,
    pixel: usize,
    jitter: (f32, f32),
) -> BoundarySample {
    let mut ray = camera_ray_for_pixel(config, params, camera, past_directed, pixel, jitter);
    let mut refined = *trace;
    refined.step_size = trace.step_size * config.boundary_step_scale;
    refined.max_steps =
        ((trace.max_steps as f32 / config.boundary_step_scale).ceil() as u32).max(1);
    for _ in 0..refined.max_steps {
        let current_radius = kerr_radius(params, ray.spatial_position());
        if current_radius <= termination_radius {
            return BoundarySample::Capture;
        }
        if current_radius >= refined.escape_radius && ray_outward(params, &refined, &ray) {
            let flow = hamiltonian_rhs(params, &ray, refined.gradient_epsilon);
            return BoundarySample::sky(
                normalized([flow.position[1], flow.position[2], flow.position[3]]),
                ray.position[0].abs(),
                config.sky_speed,
            );
        }

        let previous = ray;
        midpoint_step(
            params,
            &mut ray,
            refined.step_size,
            refined.gradient_epsilon,
        );
        if kerr_radius(params, ray.spatial_position()) <= termination_radius {
            return BoundarySample::Capture;
        }
        if let Some((point, radius, fraction)) =
            reproject_disk_crossing_f64(params, &refined, previous.position, ray.position)
        {
            let momentum = core::array::from_fn(|component| {
                interpolate_f64_to_f32(
                    previous.momentum[component],
                    ray.momentum[component],
                    fraction,
                )
            });
            let boundary_momentum = if past_directed {
                momentum.map(|component| -component)
            } else {
                momentum
            };
            let coordinate_time =
                interpolate_f64_to_f32(previous.position[0], ray.position[0], fraction);
            return disk_boundary_transport(
                params,
                &refined,
                camera,
                point,
                radius,
                boundary_momentum,
                coordinate_time,
            )
            .map(BoundarySample::Disk)
            .unwrap_or(if past_directed {
                BoundarySample::InteriorTerminated
            } else {
                BoundarySample::Unresolved
            });
        }
        if kerr_radius(params, ray.spatial_position()) >= refined.escape_radius
            && ray_outward(params, &refined, &ray)
        {
            let flow = hamiltonian_rhs(params, &ray, refined.gradient_epsilon);
            return BoundarySample::sky(
                normalized([flow.position[1], flow.position[2], flow.position[3]]),
                ray.position[0].abs(),
                config.sky_speed,
            );
        }
    }
    if past_directed {
        BoundarySample::InteriorTerminated
    } else {
        BoundarySample::Unresolved
    }
}

struct BoundaryRefinements {
    lookup: Vec<u32>,
    samples: Vec<BoundarySample>,
    samples_per_pixel: usize,
}

impl BoundaryRefinements {
    fn samples(&self, pixel: usize) -> Option<&[BoundarySample]> {
        let index = *self.lookup.get(pixel)?;
        (index != u32::MAX).then(|| {
            let start = index as usize * self.samples_per_pixel;
            &self.samples[start..start + self.samples_per_pixel]
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_projection_boundaries(
    samples: &[BoundarySample],
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    camera: &CameraTetrad,
    termination_radius: f32,
    past_directed: bool,
) -> Option<BoundaryRefinements> {
    let pixels = boundary_pixels(samples, config);
    if pixels.is_empty() {
        return None;
    }
    assert!(pixels.len() <= u32::MAX as usize);
    let side = (config.boundary_spp as f32).sqrt() as usize;
    let mut refined_samples = vec![BoundarySample::Unresolved; pixels.len() * config.boundary_spp];
    let workers = config.threads.min(pixels.len()).max(1);
    let pixels_per_worker = pixels.len().div_ceil(workers);
    let samples_per_worker = pixels_per_worker * config.boundary_spp;
    let started = Instant::now();
    std::thread::scope(|scope| {
        for (pixel_band, sample_band) in pixels
            .chunks(pixels_per_worker)
            .zip(refined_samples.chunks_mut(samples_per_worker))
        {
            scope.spawn(move || {
                for (local_pixel, &pixel) in pixel_band.iter().enumerate() {
                    for sub_sample in 0..config.boundary_spp {
                        let jitter = (
                            ((sub_sample % side) as f32 + 0.5) / side as f32,
                            ((sub_sample / side) as f32 + 0.5) / side as f32,
                        );
                        sample_band[local_pixel * config.boundary_spp + sub_sample] =
                            trace_refined_boundary_ray(
                                config,
                                params,
                                trace,
                                camera,
                                termination_radius,
                                past_directed,
                                pixel,
                                jitter,
                            );
                    }
                }
            });
        }
    });
    let mut lookup = vec![u32::MAX; config.width * config.height];
    for (index, &pixel) in pixels.iter().enumerate() {
        lookup[pixel] = index as u32;
    }
    println!(
        "kerr: refined {} projection-boundary pixel(s) at {} spp / {:.2}x step in {:.1} ms on {} thread(s)",
        pixels.len(),
        config.boundary_spp,
        config.boundary_step_scale,
        started.elapsed().as_secs_f64() * 1_000.0,
        workers,
    );
    Some(BoundaryRefinements {
        lookup,
        samples: refined_samples,
        samples_per_pixel: config.boundary_spp,
    })
}

fn load_sky_texture(path: &str) -> Result<SkyTexture, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    let mut cursor = 0;
    let magic = ppm_header_token(&bytes, &mut cursor).ok_or("missing PPM magic")?;
    if magic != b"P6" {
        return Err("sky texture must be a binary P6 PPM".to_owned());
    }
    let mut parse_dimension = |name| {
        ppm_header_token(&bytes, &mut cursor)
            .and_then(|token| core::str::from_utf8(token).ok())
            .and_then(|token| token.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("invalid PPM {name}"))
    };
    let width = parse_dimension("width")?;
    let height = parse_dimension("height")?;
    let maximum = ppm_header_token(&bytes, &mut cursor)
        .and_then(|token| core::str::from_utf8(token).ok())
        .and_then(|token| token.parse::<u16>().ok())
        .ok_or("invalid PPM component range")?;
    if maximum != 255 {
        return Err("sky texture must use an 8-bit PPM component range".to_owned());
    }
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return Err("PPM header is missing its binary payload separator".to_owned());
    }
    cursor += 1;
    if bytes.get(cursor - 1) == Some(&b'\r') && bytes.get(cursor) == Some(&b'\n') {
        cursor += 1;
    }
    let component_count = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("sky texture dimensions overflow")?;
    let payload = bytes
        .get(cursor..cursor + component_count)
        .ok_or("PPM payload is shorter than its header declares")?;
    let srgb_to_linear = |component: u8| {
        let value = component as f32 / 255.0;
        let linear = if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        };
        // Keep the survey below the disk's radiometric scale.
        linear * 0.70
    };
    Ok(SkyTexture::new(
        width,
        height,
        payload
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| {
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                ]
            })
            .collect(),
    ))
}

fn ppm_header_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    loop {
        while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
            *cursor += 1;
        }
        if bytes.get(*cursor) == Some(&b'#') {
            while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
                *cursor += 1;
            }
            continue;
        }
        break;
    }
    let start = *cursor;
    while bytes
        .get(*cursor)
        .is_some_and(|byte| !byte.is_ascii_whitespace())
    {
        *cursor += 1;
    }
    (start < *cursor).then_some(&bytes[start..*cursor])
}

fn sky(
    direction: [f32; 3],
    time: f32,
    travel_time: f32,
    angular_speed: f32,
    texture: Option<&SkyTexture>,
) -> [f32; 3] {
    let phase = (time - 0.05 * travel_time) * angular_speed;
    let (sin, cos) = phase.sin_cos();
    sky_rotated(direction, sin, cos, texture)
}

fn sky_rotated(direction: [f32; 3], sin: f32, cos: f32, texture: Option<&SkyTexture>) -> [f32; 3] {
    let direction = rotate_z(direction, sin, cos);
    if let Some(texture) = texture {
        // The lookup happens after geodesic integration. A background feature
        // is therefore deflected by the Kerr boundary map rather than pasted
        // in image space.
        return texture.sample_equirectangular(direction);
    }
    let cloud = fbm(
        3.2 * direction[0] + 1.7 * direction[2],
        3.2 * direction[1] - 1.3 * direction[2],
        0x51f1_5e3d,
    );
    let dust = fbm(
        11.0 * direction[0] - 4.0 * direction[1],
        11.0 * direction[2] + 3.0 * direction[1],
        0xa14f_2c91,
    );
    let band_coordinate =
        direction[2] * 0.79 + direction[0] * 0.24 - direction[1] * 0.18 + 0.10 * (cloud - 0.5);
    let band = (-34.0 * band_coordinate * band_coordinate).exp();
    let dust_lane = 1.0
        - 0.82
            * (-260.0 * (band_coordinate + 0.018 * (dust - 0.5)).powi(2)).exp()
            * (0.35 + 0.65 * dust);
    let cell = [
        (direction[0] * 900.0).floor() as i32,
        (direction[1] * 900.0).floor() as i32,
        (direction[2] * 900.0).floor() as i32,
    ];
    let random = hash_u32(
        (cell[0] as u32).wrapping_mul(0x9e37_79b9)
            ^ (cell[1] as u32).wrapping_mul(0x85eb_ca6b)
            ^ (cell[2] as u32).wrapping_mul(0xc2b2_ae35),
    ) as f32
        / u32::MAX as f32;
    let star = if random > 0.9972 {
        ((random - 0.9972) / 0.0028).powi(6) * 5.0
    } else {
        0.0
    };
    [
        0.0015 + 0.11 * band * dust_lane * (0.25 + 0.75 * cloud) + star,
        0.0022 + 0.15 * band * dust_lane * (0.35 + 0.65 * cloud) + star * 0.91,
        0.0050 + 0.23 * band * dust_lane * (0.45 + 0.55 * cloud) + star * 0.80,
    ]
}

#[inline]
fn rotate_z(direction: [f32; 3], sin: f32, cos: f32) -> [f32; 3] {
    [
        cos * direction[0] - sin * direction[1],
        sin * direction[0] + cos * direction[1],
        direction[2],
    ]
}

const BLACKBODY_LUT_BINS: usize = 256;
const BLACKBODY_MIN_K: f32 = 900.0;
const BLACKBODY_MAX_K: f32 = 30_000.0;
const BLACKBODY_REFERENCE_K: f32 = 10_000.0;
const SURFACE_TEMPERATURE_BINS: usize = 128;
const SURFACE_REDSHIFT_BINS: usize = 256;
const SURFACE_REDSHIFT_MIN: f32 = 0.12;
const SURFACE_REDSHIFT_MAX: f32 = 3.0;
const PLASMA_FLOOR_K: f32 = 250_000.0;
const PLASMA_PEAK_K: f32 = 1_500_000.0;
/// Ten-thousand-kelvin units map the supported thermal field to 25..150,
/// within the finite range of every selectable eight-bit plasma format.
const TEMPERATURE_FP8_SCALE_K: f32 = 10_000.0;
const PLASMA_RADIANCE_GAIN: f32 = 13.0;

struct BlackbodyLut {
    values: [BlackbodyRadiance; BLACKBODY_LUT_BINS],
}

impl BlackbodyLut {
    fn new() -> Self {
        Self {
            values: core::array::from_fn(|index| {
                let amount = index as f32 / (BLACKBODY_LUT_BINS - 1) as f32;
                radiation::blackbody_radiance_rgb(
                    BLACKBODY_MIN_K + amount * (BLACKBODY_MAX_K - BLACKBODY_MIN_K),
                    BLACKBODY_REFERENCE_K,
                )
            }),
        }
    }

    #[inline]
    fn sample(&self, temperature_kelvin: f32) -> BlackbodyRadiance {
        let position = ((temperature_kelvin.clamp(BLACKBODY_MIN_K, BLACKBODY_MAX_K)
            - BLACKBODY_MIN_K)
            / (BLACKBODY_MAX_K - BLACKBODY_MIN_K))
            * (BLACKBODY_LUT_BINS - 1) as f32;
        let low = position.floor() as usize;
        let high = (low + 1).min(BLACKBODY_LUT_BINS - 1);
        let amount = position - low as f32;
        BlackbodyRadiance {
            spectral_rgb: core::array::from_fn(|channel| {
                self.values[low].spectral_rgb[channel]
                    + amount
                        * (self.values[high].spectral_rgb[channel]
                            - self.values[low].spectral_rgb[channel])
            }),
            relative_bolometric_intensity: self.values[low].relative_bolometric_intensity
                + amount
                    * (self.values[high].relative_bolometric_intensity
                        - self.values[low].relative_bolometric_intensity),
        }
    }
}

/// Host-side transfer cache for the interactive surface tier.
///
/// The exact volume tier still evaluates its spectral transfer from the
/// physical temperature field. Surface animation repeatedly evaluates the
/// same bounded temperature/redshift domain for millions of cached disk
/// crossings, so a bilinear table removes logarithm and power calls from that
/// hot loop without changing the geodesic boundary or plasma field.
struct SurfaceTransferLut {
    values: Vec<[f32; 3]>,
}

impl SurfaceTransferLut {
    fn new(blackbody: &BlackbodyLut) -> Self {
        let mut values = Vec::with_capacity(SURFACE_TEMPERATURE_BINS * SURFACE_REDSHIFT_BINS);
        for redshift_index in 0..SURFACE_REDSHIFT_BINS {
            let redshift_amount = redshift_index as f32 / (SURFACE_REDSHIFT_BINS - 1) as f32;
            let redshift = SURFACE_REDSHIFT_MIN
                + redshift_amount * (SURFACE_REDSHIFT_MAX - SURFACE_REDSHIFT_MIN);
            for temperature_index in 0..SURFACE_TEMPERATURE_BINS {
                let temperature_amount =
                    temperature_index as f32 / (SURFACE_TEMPERATURE_BINS - 1) as f32;
                let temperature =
                    PLASMA_FLOOR_K + temperature_amount * (PLASMA_PEAK_K - PLASMA_FLOOR_K);
                let shifted = blackbody.sample(display_color_temperature(temperature * redshift));
                let gain = display_radiance_gain(temperature);
                values.push(shifted.spectral_rgb.map(|channel| channel * gain));
            }
        }
        Self { values }
    }

    #[inline]
    fn sample(&self, temperature: f32, redshift: f32) -> [f32; 3] {
        let temperature_position = ((temperature.clamp(PLASMA_FLOOR_K, PLASMA_PEAK_K)
            - PLASMA_FLOOR_K)
            / (PLASMA_PEAK_K - PLASMA_FLOOR_K))
            * (SURFACE_TEMPERATURE_BINS - 1) as f32;
        let redshift_position = ((redshift.clamp(SURFACE_REDSHIFT_MIN, SURFACE_REDSHIFT_MAX)
            - SURFACE_REDSHIFT_MIN)
            / (SURFACE_REDSHIFT_MAX - SURFACE_REDSHIFT_MIN))
            * (SURFACE_REDSHIFT_BINS - 1) as f32;
        let temperature_low = temperature_position.floor() as usize;
        let temperature_high = (temperature_low + 1).min(SURFACE_TEMPERATURE_BINS - 1);
        let redshift_low = redshift_position.floor() as usize;
        let redshift_high = (redshift_low + 1).min(SURFACE_REDSHIFT_BINS - 1);
        let temperature_amount = temperature_position - temperature_low as f32;
        let redshift_amount = redshift_position - redshift_low as f32;
        let at = |temperature_index, redshift_index| {
            self.values[temperature_index + SURFACE_TEMPERATURE_BINS * redshift_index]
        };
        let low = at(temperature_low, redshift_low);
        let low_next = at(temperature_high, redshift_low);
        let high = at(temperature_low, redshift_high);
        let high_next = at(temperature_high, redshift_high);
        core::array::from_fn(|channel| {
            let along_temperature_low =
                low[channel] + temperature_amount * (low_next[channel] - low[channel]);
            let along_temperature_high =
                high[channel] + temperature_amount * (high_next[channel] - high[channel]);
            along_temperature_low
                + redshift_amount * (along_temperature_high - along_temperature_low)
        })
    }
}

/// Exact equatorial slice of the mixed-precision plasma fields used by the
/// surface tier. A disk boundary always has material-space z=0, so sampling
/// this predecoded 2D projection is algebraically equivalent to trilinear
/// sampling the two central Z planes, while avoiding repeated FP8 decoding
/// and unused Z-coordinate work on every frame.
struct SurfaceField {
    dims: [u32; 2],
    temperature: Vec<f32>,
    density: Vec<f32>,
}

impl SurfaceField {
    fn from_volume(
        storage: PlasmaStorage,
        dims: [u32; 3],
        temperature_kk: &[u8],
        extinction: &[u8],
    ) -> Self {
        let count = dims[0] as usize * dims[1] as usize;
        let mut temperature = Vec::with_capacity(count);
        let mut density = Vec::with_capacity(count);
        let z = dims[2] as f32 * 0.5;
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let point = [x as f32 + 0.5, y as f32 + 0.5, z];
                temperature.push(
                    TEMPERATURE_FP8_SCALE_K * storage.sample_trilinear(temperature_kk, dims, point),
                );
                density.push(storage.sample_trilinear(extinction, dims, point));
            }
        }
        Self {
            dims: [dims[0], dims[1]],
            temperature,
            density,
        }
    }

    #[inline]
    fn sample(&self, point: [f32; 2]) -> (f32, f32) {
        let sample = |values: &[f32]| {
            if point.iter().zip(self.dims).any(|(coordinate, extent)| {
                !coordinate.is_finite() || *coordinate < 0.0 || *coordinate > extent as f32
            }) {
                return 0.0;
            }
            let centered_x = (point[0] - 0.5).clamp(0.0, self.dims[0] as f32 - 1.0);
            let centered_y = (point[1] - 0.5).clamp(0.0, self.dims[1] as f32 - 1.0);
            let x0 = centered_x.floor() as usize;
            let y0 = centered_y.floor() as usize;
            let x1 = (x0 + 1).min(self.dims[0] as usize - 1);
            let y1 = (y0 + 1).min(self.dims[1] as usize - 1);
            let fx = centered_x - x0 as f32;
            let fy = centered_y - y0 as f32;
            let width = self.dims[0] as usize;
            let top =
                values[x0 + width * y0] + fx * (values[x1 + width * y0] - values[x0 + width * y0]);
            let bottom =
                values[x0 + width * y1] + fx * (values[x1 + width * y1] - values[x0 + width * y1]);
            top + fy * (bottom - top)
        };
        (sample(&self.temperature), sample(&self.density))
    }
}

#[inline]
fn temperature_from_heating(heating: f32) -> f32 {
    let amount = heating.clamp(0.0, 1.0);
    let floor_squared = PLASMA_FLOOR_K * PLASMA_FLOOR_K;
    let peak_squared = PLASMA_PEAK_K * PLASMA_PEAK_K;
    let floor_fourth = floor_squared * floor_squared;
    let peak_fourth = peak_squared * peak_squared;
    (floor_fourth + amount * (peak_fourth - floor_fourth))
        .sqrt()
        .sqrt()
}

/// Map an X-ray-hot effective temperature to a bounded photographic color
/// temperature. At these temperatures a literal visible Planck spectrum is
/// nearly uniform blue-white; preserving a color-temperature response keeps
/// spatial heating and Doppler structure readable after tone mapping.
#[inline]
fn display_color_temperature(effective_temperature_kelvin: f32) -> f32 {
    let floor_log = PLASMA_FLOOR_K.ln();
    let peak_log = PLASMA_PEAK_K.ln();
    let amount = ((effective_temperature_kelvin.max(PLASMA_FLOOR_K).ln() - floor_log)
        / (peak_log - floor_log))
        .clamp(0.0, 1.0);
    1_600.0 + amount.powf(0.90) * 3_200.0
}

#[inline]
fn display_radiance_gain(effective_temperature_kelvin: f32) -> f32 {
    PLASMA_RADIANCE_GAIN * (effective_temperature_kelvin / 1_000_000.0).sqrt()
}

fn disk_turbulence(point: [f32; 3], radius: f32, inner_radius: f32) -> f32 {
    // Logarithmic rotation stretches isotropic fBm into shear-aligned eddies
    // without imposing the conspicuous concentric sine rings of the old scene.
    let winding = -1.7 * (radius / inner_radius).ln();
    let (sin_winding, cos_winding) = winding.sin_cos();
    let x = 0.52 * (cos_winding * point[0] - sin_winding * point[1]);
    let y = 0.52 * (sin_winding * point[0] + cos_winding * point[1]);
    let broad = fbm(x, y, 0xc13f_a9a9);
    let fine = fbm(2.7 * x + 5.0, 0.65 * y - 3.0, 0x91e1_0da5);
    0.62 + 0.58 * broad + 0.16 * (fine - 0.5)
}

#[derive(Clone, Copy, Debug)]
struct DiskRegion {
    local_flux: f32,
    specific_kinetic_energy: f32,
    scale_height: f32,
    inner_fade: f32,
    outer_fade: f32,
    turbulence: f32,
}

#[inline]
fn thick_disk_scale_height(orbital_radius: f32, radial: f32, turbulence: f32) -> f32 {
    let aspect_ratio = 0.11 + 0.10 * radial.clamp(0.0, 1.0).sqrt();
    orbital_radius * aspect_ratio * (0.85 + 0.30 * turbulence.clamp(0.0, 1.5))
}

/// Compact storage used by the CPU-side material kernels after the vAccel
/// boundary graph has finished. This is deliberately not an accelerator
/// `DType`: providers consume the graph's existing F32/F16 ABI, while these
/// unsigned formats are lawful only for the non-negative plasma fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlasmaStorage {
    B8,
    Uf8E4M4,
    Uf8E5M3,
}

impl PlasmaStorage {
    fn parse(value: Option<&str>) -> Result<Self, &str> {
        match value {
            None | Some("uf8e4m4") => Ok(Self::Uf8E4M4),
            Some("uf8e5m3") => Ok(Self::Uf8E5M3),
            Some("b8") => Ok(Self::B8),
            Some(value) => Err(value),
        }
    }

    fn from_env() -> Self {
        let requested = std::env::var("AXGEOM_PLASMA_STORAGE").ok();
        match Self::parse(requested.as_deref()) {
            Ok(storage) => storage,
            Err(value) => {
                panic!("AXGEOM_PLASMA_STORAGE must be B8, uf8e4m4, or uf8e5m3; got {value:?}")
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::B8 => "B8",
            Self::Uf8E4M4 => "UF8 E4M4",
            Self::Uf8E5M3 => "UF8 E5M3",
        }
    }

    #[inline]
    fn encode(self, value: f32) -> u8 {
        let maximum = match self {
            Self::B8 => 448.0,
            Self::Uf8E4M4 => 248.0,
            Self::Uf8E5M3 => 61_440.0,
        };
        assert!(
            value.is_finite() && (0.0..=maximum).contains(&value),
            "{} plasma value {value} is outside its finite non-negative range",
            self.label(),
        );
        match self {
            Self::B8 => fp8::f32_to_e4m3_rne(value),
            Self::Uf8E4M4 => uf::f32_to_uf8e4m4_rne(value),
            Self::Uf8E5M3 => uf::f32_to_uf8e5m3_rne(value),
        }
    }

    #[inline]
    fn sample_trilinear(self, values: &[u8], dims: [u32; 3], point: [f32; 3]) -> f32 {
        match self {
            Self::B8 => voxel::sample_trilinear_e4m3(values, dims, point),
            Self::Uf8E4M4 => voxel::sample_trilinear_uf8e4m4(values, dims, point),
            Self::Uf8E5M3 => voxel::sample_trilinear_uf8e5m3(values, dims, point),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn march_emission(
        self,
        extinction: &[u8],
        source_rgb: &[u8],
        dims: [u32; 3],
        origin: [f32; 3],
        direction: [f32; 3],
        t_max: f32,
        extinction_scale: f32,
        emission_scale: f32,
        transmittance_cutoff: f32,
    ) -> voxel::EmissionTransfer {
        match self {
            Self::B8 => voxel::march_emission_e4m3(
                extinction,
                source_rgb,
                dims,
                origin,
                direction,
                t_max,
                extinction_scale,
                emission_scale,
                transmittance_cutoff,
            ),
            Self::Uf8E4M4 => voxel::march_emission_uf8e4m4(
                extinction,
                source_rgb,
                dims,
                origin,
                direction,
                t_max,
                extinction_scale,
                emission_scale,
                transmittance_cutoff,
            ),
            Self::Uf8E5M3 => voxel::march_emission_uf8e5m3(
                extinction,
                source_rgb,
                dims,
                origin,
                direction,
                t_max,
                extinction_scale,
                emission_scale,
                transmittance_cutoff,
            ),
        }
    }
}

struct PlasmaVolume {
    dims: [u32; 3],
    storage: PlasmaStorage,
    extinction: Vec<u8>,
    source_rgb: Vec<u8>,
    temperature_kk: Vec<u8>,
    xy_half_extent: f32,
    half_thickness: f32,
    kinetic_reference: f32,
    blackbody: BlackbodyLut,
    surface_transfer: Option<SurfaceTransferLut>,
    surface_field: Option<SurfaceField>,
}

impl PlasmaVolume {
    fn build(config: RenderConfig, params: &KerrParams, trace: &TraceConfig) -> Self {
        Self::build_with_storage(config, params, trace, PlasmaStorage::from_env())
    }

    fn build_with_storage(
        config: RenderConfig,
        params: &KerrParams,
        trace: &TraceConfig,
        storage: PlasmaStorage,
    ) -> Self {
        let blackbody = BlackbodyLut::new();
        let dims = [config.plasma_xy, config.plasma_xy, config.plasma_z];
        let voxel_count = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        let mut extinction = vec![0; voxel_count];
        let mut source_rgb = vec![0; voxel_count * 3];
        let mut temperature_kk = vec![0; voxel_count];
        // Equatorial Cartesian Kerr-Schild coordinates obey rho^2=r^2+a^2.
        // The grid extent therefore encloses the requested Boyer-Lindquist
        // outer radius instead of clipping its azimuthal boundary.
        let xy_half_extent =
            (trace.disk_outer_radius * trace.disk_outer_radius + params.spin * params.spin).sqrt();
        let half_thickness = config.plasma_half_thickness;
        let xy_scale = 2.0 * xy_half_extent / dims[0] as f32;
        let z_scale = 2.0 * half_thickness / dims[2] as f32;
        let kinetic_reference = circular_equatorial_specific_kinetic_energy(
            params,
            trace.disk_inner_radius,
            OrbitDirection::Positive,
        )
        .expect("stable inner disk orbit has local kinetic energy");

        // Orbital energy, flux, and shear-aligned turbulence are invariant
        // through the disk thickness. Compute them once per XY region rather
        // than repeating metric and fBm work for every Z slice.
        let mut regions = Vec::with_capacity((dims[0] * dims[1]) as usize);
        let mut kinetic_minimum = f32::INFINITY;
        let mut kinetic_maximum = 0.0_f32;
        for y in 0..dims[1] {
            let world_y = (y as f32 + 0.5) * xy_scale - xy_half_extent;
            for x in 0..dims[0] {
                let world_x = (x as f32 + 0.5) * xy_scale - xy_half_extent;
                let orbital_radius = kerr_radius(params, [world_x, world_y, 0.0]);
                if orbital_radius < trace.disk_inner_radius
                    || orbital_radius > trace.disk_outer_radius
                {
                    regions.push(None);
                    continue;
                }
                let radial = ((orbital_radius - trace.disk_inner_radius)
                    / (trace.disk_outer_radius - trace.disk_inner_radius))
                    .clamp(0.0, 1.0);
                let specific_kinetic_energy = circular_equatorial_specific_kinetic_energy(
                    params,
                    orbital_radius,
                    OrbitDirection::Positive,
                )
                .unwrap_or(0.0);
                let turbulence = disk_turbulence(
                    [world_x, world_y, 0.0],
                    orbital_radius,
                    trace.disk_inner_radius,
                );
                // A radiatively inefficient, pressure-supported flow is
                // geometrically thick rather than an infinitesimal Novikov-
                // Thorne sheet. H/R flares from roughly 0.12 near the ISCO to
                // 0.23 at the outer boundary, with bounded turbulent support.
                kinetic_minimum = kinetic_minimum.min(specific_kinetic_energy);
                kinetic_maximum = kinetic_maximum.max(specific_kinetic_energy);
                regions.push(Some(DiskRegion {
                    local_flux: zero_torque_thin_disk_flux(orbital_radius, trace.disk_inner_radius)
                        .unwrap_or(0.0),
                    specific_kinetic_energy,
                    scale_height: thick_disk_scale_height(orbital_radius, radial, turbulence),
                    inner_fade: ((orbital_radius - trace.disk_inner_radius)
                        / (0.20 * trace.disk_inner_radius))
                        .clamp(0.0, 1.0),
                    outer_fade: ((trace.disk_outer_radius - orbital_radius)
                        / (0.12 * trace.disk_outer_radius))
                        .clamp(0.0, 1.0),
                    turbulence,
                }));
            }
        }

        let mut temperature_minimum = f32::INFINITY;
        let mut temperature_maximum = 0.0_f32;
        let mut bolometric_maximum = 0.0_f32;
        for z in 0..dims[2] {
            let world_z = (z as f32 + 0.5) * z_scale - half_thickness;
            for y in 0..dims[1] {
                let world_y = (y as f32 + 0.5) * xy_scale - xy_half_extent;
                for x in 0..dims[0] {
                    let Some(region) = regions[(x + dims[0] * y) as usize] else {
                        continue;
                    };
                    let world_x = (x as f32 + 0.5) * xy_scale - xy_half_extent;

                    // A flared, optically thinner atmosphere above a denser
                    // equatorial flow. Smooth radial tapers keep the voxel
                    // boundary from becoming a second artificial disk edge.
                    let vertical = (-0.5 * (world_z / region.scale_height).powi(2)).exp();
                    let vertical_detail = fbm(
                        world_x * 0.63 + world_z * 1.7,
                        world_y * 0.63 - world_z * 1.3,
                        0xe91a_7c53,
                    );
                    let filament = fbm(
                        world_x * 1.75 + world_z * 2.3,
                        world_y * 1.75 - world_z * 1.7,
                        0x7b83_4d11,
                    );
                    // Porosity grows into the flared atmosphere, preserving a
                    // turbulent, filamentary silhouette instead of a smooth
                    // optically thick upper-right rim. The dense equatorial
                    // core stays coherent rather than reading as white noise.
                    let edge_porosity = (0.15 + 1.10 * (filament - 0.24)).clamp(0.0, 1.0);
                    let atmosphere = (1.0 - vertical).clamp(0.0, 1.0);
                    let porous = 1.0 - atmosphere * 0.68 * (1.0 - edge_porosity);
                    let edge_distance = half_thickness - world_z.abs();
                    let edge_amount = (edge_distance / (0.20 * half_thickness)).clamp(0.0, 1.0);
                    let vertical_edge = edge_amount * edge_amount * (3.0 - 2.0 * edge_amount);
                    let density = (vertical
                        * region.inner_fade
                        * region.outer_fade
                        * porous
                        * vertical_edge
                        * (0.035
                            + 0.36 * region.turbulence * region.turbulence
                            + 0.13 * vertical_detail))
                        .clamp(0.0, 0.85);
                    let kinetic_fraction =
                        (region.specific_kinetic_energy / kinetic_reference).clamp(0.0, 1.5);
                    let kinetic_heating = 0.58 + 0.42 * kinetic_fraction;
                    let clump_heating = (0.30 + 0.70 * region.turbulence * region.turbulence)
                        * (0.78 + 0.22 * vertical_detail);
                    let height_heating = 0.38 + 0.62 * vertical.sqrt();
                    let heating = region.local_flux
                        * kinetic_heating
                        * clump_heating
                        * region.inner_fade
                        * region.outer_fade
                        * height_heating;
                    let temperature = temperature_from_heating(heating);
                    let radiance = blackbody.sample(display_color_temperature(temperature));
                    let radiance_gain = display_radiance_gain(temperature);
                    temperature_minimum = temperature_minimum.min(temperature);
                    temperature_maximum = temperature_maximum.max(temperature);
                    bolometric_maximum =
                        bolometric_maximum.max(radiance.relative_bolometric_intensity);
                    let index = (x + dims[0] * (y + dims[1] * z)) as usize;
                    extinction[index] = storage.encode(density);
                    temperature_kk[index] = storage.encode(temperature / TEMPERATURE_FP8_SCALE_K);
                    for channel in 0..3 {
                        source_rgb[index * 3 + channel] =
                            storage.encode(radiance.spectral_rgb[channel] * radiance_gain);
                    }
                }
            }
        }

        println!(
            "kerr: plasma volume {}x{}x{} ({} KiB {} extinction+temperature+RGB)",
            dims[0],
            dims[1],
            dims[2],
            (extinction.len() + temperature_kk.len() + source_rgb.len()) / 1_024,
            storage.label(),
        );
        println!(
            "kerr: regional orbital kinetic {:.4}..{:.4} mc^2, temperature {:.0}..{:.0} K, peak bolometric {:.3}x",
            kinetic_minimum,
            kinetic_maximum,
            temperature_minimum,
            temperature_maximum,
            bolometric_maximum,
        );
        let (surface_transfer, surface_field) = if config.plasma_mode == PlasmaMode::Surface {
            (
                Some(SurfaceTransferLut::new(&blackbody)),
                Some(SurfaceField::from_volume(
                    storage,
                    dims,
                    &temperature_kk,
                    &extinction,
                )),
            )
        } else {
            (None, None)
        };
        Self {
            dims,
            storage,
            extinction,
            source_rgb,
            temperature_kk,
            xy_half_extent,
            half_thickness,
            kinetic_reference,
            blackbody,
            surface_transfer,
            surface_field,
        }
    }

    #[inline]
    fn material_position(&self, point: [f32; 3]) -> [f32; 3] {
        [
            (point[0] + self.xy_half_extent) * self.dims[0] as f32 / (2.0 * self.xy_half_extent),
            (point[1] + self.xy_half_extent) * self.dims[1] as f32 / (2.0 * self.xy_half_extent),
            (point[2] + self.half_thickness) * self.dims[2] as f32 / (2.0 * self.half_thickness),
        ]
    }

    fn material_ray(&self, point: [f32; 3], direction: [f32; 3]) -> ([f32; 3], [f32; 3], f32) {
        let span = 2.5 * self.xy_half_extent + 2.0 * self.half_thickness;
        let world_origin = [
            point[0] - span * direction[0],
            point[1] - span * direction[1],
            point[2] - span * direction[2],
        ];
        let scale = [
            self.dims[0] as f32 / (2.0 * self.xy_half_extent),
            self.dims[1] as f32 / (2.0 * self.xy_half_extent),
            self.dims[2] as f32 / (2.0 * self.half_thickness),
        ];
        (
            [
                (world_origin[0] + self.xy_half_extent) * scale[0],
                (world_origin[1] + self.xy_half_extent) * scale[1],
                (world_origin[2] + self.half_thickness) * scale[2],
            ],
            [
                direction[0] * scale[0],
                direction[1] * scale[1],
                direction[2] * scale[2],
            ],
            2.0 * span,
        )
    }
}

fn disk(
    config: RenderConfig,
    trace: &TraceConfig,
    plasma: &PlasmaVolume,
    boundary: KerrDiskBoundary,
    time: f32,
) -> [f32; 3] {
    let (material_point, material_direction) =
        boundary.material_coordinates(time, config.plasma_time_scale);
    let redshift = boundary.redshift.clamp(0.12, 3.0);
    let material_position = plasma.material_position(material_point);
    let outer_distance =
        (trace.disk_outer_radius - boundary.radius) / (0.16 * trace.disk_outer_radius);
    let outer_amount = outer_distance.clamp(0.0, 1.0);
    let outer_taper = outer_amount * outer_amount * (3.0 - 2.0 * outer_amount);

    if config.plasma_mode == PlasmaMode::Surface {
        let (sampled_temperature, surface_density) = plasma
            .surface_field
            .as_ref()
            .expect("surface mode builds an equatorial field")
            .sample([material_position[0], material_position[1]]);
        let local_temperature = if sampled_temperature >= BLACKBODY_MIN_K {
            sampled_temperature
        } else {
            boundary_temperature_fallback(boundary, plasma)
        };
        let surface_fill = (surface_density / 0.30).clamp(0.0, 1.0);
        let surface_scale = outer_taper * (0.06 + 0.64 * surface_fill);
        let transfer = plasma
            .surface_transfer
            .as_ref()
            .expect("surface mode builds a spectral transfer cache")
            .sample(local_temperature, redshift);
        return transfer.map(|channel| channel * surface_scale);
    }

    let sampled_temperature = TEMPERATURE_FP8_SCALE_K
        * plasma
            .storage
            .sample_trilinear(&plasma.temperature_kk, plasma.dims, material_position);
    let local_temperature = if sampled_temperature >= BLACKBODY_MIN_K {
        sampled_temperature
    } else {
        boundary_temperature_fallback(boundary, plasma)
    };
    let surface_density =
        plasma
            .storage
            .sample_trilinear(&plasma.extinction, plasma.dims, material_position);
    let surface_fill = (surface_density / 0.30).clamp(0.0, 1.0);
    let surface_scale = outer_taper * (0.06 + 0.64 * surface_fill);
    let reference = plasma
        .blackbody
        .sample(display_color_temperature(local_temperature));
    let shifted = plasma
        .blackbody
        .sample(display_color_temperature(local_temperature * redshift));
    // For a Planck source, invariance of I_nu/nu^3 gives
    // g^3 B_(nu/g)(T) = B_nu(gT). The shifted/reference ratio therefore
    // carries both the spectral displacement and its intensity change; a
    // second g^4 multiplier would double-count the transfer.
    let spectral_shift = core::array::from_fn::<_, 3, _>(|channel| {
        shifted.spectral_rgb[channel] / reference.spectral_rgb[channel].max(1.0e-4)
    });
    let surface = core::array::from_fn::<_, 3, _>(|channel| {
        // The dense equatorial core is still an LTE surface, but the porous
        // atmosphere modulates its covering fraction. This avoids a perfectly
        // smooth, solid silhouette while preserving a bright inner disk.
        shifted.spectral_rgb[channel] * display_radiance_gain(local_temperature) * surface_scale
    });

    let (origin, direction, t_max) = plasma.material_ray(material_point, material_direction);
    let volume = plasma.storage.march_emission(
        &plasma.extinction,
        &plasma.source_rgb,
        plasma.dims,
        origin,
        direction,
        t_max,
        config.plasma_extinction,
        config.plasma_emission,
        0.01,
    );
    core::array::from_fn(|channel| {
        volume.radiance[channel] * spectral_shift[channel] + volume.transmittance * surface[channel]
    })
}

#[inline]
fn boundary_temperature_fallback(boundary: KerrDiskBoundary, plasma: &PlasmaVolume) -> f32 {
    let kinetic_fraction =
        (boundary.specific_kinetic_energy / plasma.kinetic_reference).clamp(0.0, 1.5);
    temperature_from_heating(boundary.local_flux * (0.58 + 0.42 * kinetic_fraction))
}

#[allow(clippy::too_many_arguments)]
fn shade_pixels(
    samples: &[BoundarySample],
    refinements: Option<&BoundaryRefinements>,
    first_pixel: usize,
    config: RenderConfig,
    trace: &TraceConfig,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
    time: f32,
    framebuffer: &mut [u32],
) {
    assert_eq!(samples.len(), framebuffer.len() * config.samples_per_pixel);
    let (frame_sin, frame_cos) = (time * config.sky_speed).sin_cos();
    let frame_u = time * config.sky_speed / core::f32::consts::TAU;
    for (pixel_offset, (pixel, base_samples)) in framebuffer
        .iter_mut()
        .zip(samples.chunks_exact(config.samples_per_pixel))
        .enumerate()
    {
        let pixel_samples = refinements
            .and_then(|refinements| refinements.samples(first_pixel + pixel_offset))
            .unwrap_or(base_samples);
        let inverse_samples = 1.0 / pixel_samples.len() as f32;
        let mut color = [0.0; 3];
        for sample in pixel_samples {
            let sample_color = match *sample {
                BoundarySample::Capture => [0.0; 3],
                BoundarySample::Disk(boundary) => disk(config, trace, plasma, boundary, time),
                BoundarySample::Sky {
                    direction,
                    retarded_sin,
                    retarded_cos,
                    retarded_uv,
                } => match sky_texture {
                    Some(texture) => {
                        texture.sample_equirectangular_uv(retarded_uv[0] + frame_u, retarded_uv[1])
                    }
                    None => {
                        let sin = frame_sin * retarded_cos + frame_cos * retarded_sin;
                        let cos = frame_cos * retarded_cos - frame_sin * retarded_sin;
                        sky_rotated(direction, sin, cos, None)
                    }
                },
                BoundarySample::Unresolved => [0.08, 0.012, 0.025],
                BoundarySample::InteriorTerminated => [0.0; 3],
            };
            for channel in 0..3 {
                color[channel] += sample_color[channel];
            }
        }
        *pixel = pack_aces(color.map(|channel| channel * inverse_samples * config.exposure));
    }
}

#[allow(clippy::too_many_arguments)]
fn shade(
    samples: &[BoundarySample],
    refinements: Option<&BoundaryRefinements>,
    config: RenderConfig,
    trace: &TraceConfig,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
    time: f32,
    framebuffer: &mut [u32],
) {
    assert_eq!(samples.len(), framebuffer.len() * config.samples_per_pixel);
    let workers = config.threads.min(framebuffer.len()).max(1);
    if workers == 1 || framebuffer.len() < config.width * 8 {
        shade_pixels(
            samples,
            refinements,
            0,
            config,
            trace,
            plasma,
            sky_texture,
            time,
            framebuffer,
        );
        return;
    }
    let pixels_per_worker = framebuffer.len().div_ceil(workers);
    let samples_per_worker = pixels_per_worker * config.samples_per_pixel;
    std::thread::scope(|scope| {
        for (band, (pixels, samples)) in framebuffer
            .chunks_mut(pixels_per_worker)
            .zip(samples.chunks(samples_per_worker))
            .enumerate()
        {
            scope.spawn(move || {
                shade_pixels(
                    samples,
                    refinements,
                    band * pixels_per_worker,
                    config,
                    trace,
                    plasma,
                    sky_texture,
                    time,
                    pixels,
                )
            });
        }
    });
}

#[cfg(feature = "window")]
fn shade_available(
    samples: &[BoundarySample],
    config: RenderConfig,
    trace: &TraceConfig,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
    time: f32,
    framebuffer: &mut [u32],
) {
    let pixels = (samples.len() / config.samples_per_pixel).min(framebuffer.len());
    let complete_samples = pixels * config.samples_per_pixel;
    shade(
        &samples[..complete_samples],
        None,
        config,
        trace,
        plasma,
        sky_texture,
        time,
        &mut framebuffer[..pixels],
    );
}

fn write_ppm(path: &str, config: RenderConfig, framebuffer: &[u32]) {
    let mut bytes = format!("P6\n{} {}\n255\n", config.width, config.height).into_bytes();
    bytes.reserve(framebuffer.len() * 3);
    for pixel in framebuffer {
        bytes.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]);
    }
    std::fs::write(path, bytes).expect("write Kerr snapshot");
    println!("snapshot -> {path}");
}

fn snapshot_path(pattern: &str, frame: u64) -> Option<String> {
    pattern
        .contains("{frame}")
        .then(|| pattern.replace("{frame}", &format!("{frame:04}")))
}

fn print_event_counts(samples: &[BoundarySample]) {
    let mut events = [0_usize; 5];
    for sample in samples {
        events[match sample {
            BoundarySample::Capture => 0,
            BoundarySample::Disk(_) => 1,
            BoundarySample::Sky { .. } => 2,
            BoundarySample::Unresolved => 3,
            BoundarySample::InteriorTerminated => 4,
        }] += 1;
    }
    println!(
        "kerr: events capture={} disk={} sky={} unresolved={} interior={}",
        events[0], events[1], events[2], events[3], events[4]
    );
}

#[allow(clippy::too_many_arguments)]
fn run_headless<C: Accelerator, A: Accelerator + TosaCapabilityProvider>(
    runtime: &mut AxHetero<C, A>,
    backend_label: &str,
    device_model: &AxDeviceModel,
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    camera: &CameraTetrad,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
) {
    let samples = trace_scene(
        runtime,
        backend_label,
        device_model,
        config,
        params,
        trace,
        camera,
        axgeom::black_hole::horizon_radius(params),
        false,
        |_| true,
    )
    .expect("headless trace cannot be cancelled");
    print_event_counts(&samples);
    let refinements = refine_projection_boundaries(
        &samples,
        config,
        params,
        trace,
        camera,
        horizon_radius(params),
        false,
    );
    let mut framebuffer = vec![0; config.width * config.height];
    let frames = env_or("AXGEOM_FRAMES", 1_u64).max(1);
    let frame_dt = env_or("AXGEOM_FRAME_DT", 1.0_f32 / 24.0);
    let initial_time = env_or("AXGEOM_TIME", 0.0_f32);
    assert!(frame_dt.is_finite() && initial_time.is_finite());
    let snapshot = std::env::var("AXGEOM_SNAPSHOT").unwrap_or_else(|_| "kerr.ppm".to_owned());
    let mut shade_seconds = 0.0_f64;
    for frame in 0..frames {
        let shade_start = Instant::now();
        shade(
            &samples,
            refinements.as_ref(),
            config,
            trace,
            plasma,
            sky_texture,
            initial_time + frame as f32 * frame_dt,
            &mut framebuffer,
        );
        shade_seconds += shade_start.elapsed().as_secs_f64();
        if let Some(path) = snapshot_path(&snapshot, frame) {
            write_ppm(&path, config, &framebuffer);
        }
    }
    println!(
        "kerr: {frames} cached holographic+voxel frame(s) in {:.1} ms ({:.1} fps, {} thread(s))",
        shade_seconds * 1_000.0,
        frames as f64 / shade_seconds.max(f64::EPSILON),
        config.threads.min(config.height),
    );
    if !snapshot.contains("{frame}") {
        write_ppm(&snapshot, config, &framebuffer);
    }
}

fn run_infall_headless<C: Accelerator, A: Accelerator + TosaCapabilityProvider>(
    runtime: &mut AxHetero<C, A>,
    backend_label: &str,
    device_model: &AxDeviceModel,
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
) {
    let start = [
        env_or("AXGEOM_INFALL_START_X", 0.0),
        env_or("AXGEOM_INFALL_START_Y", -2.15),
        env_or("AXGEOM_INFALL_START_Z", 0.35),
    ];
    let speed = env_or("AXGEOM_INFALL_SPEED", 0.34);
    let frames = env_or("AXGEOM_INFALL_FRAMES", 12_u64).max(1);
    let substeps = env_or("AXGEOM_INFALL_SUBSTEPS", 8_u32).max(1);
    let proper_step = env_or("AXGEOM_INFALL_PROPER_STEP", 0.018_f32);
    let interior_cutoff = env_or("AXGEOM_INTERIOR_CUTOFF", 0.08_f32);
    assert!(interior_cutoff.is_finite() && interior_cutoff > 0.0);
    assert!(interior_cutoff < horizon_radius(params));
    let mut observer = zero_angular_momentum_infalling_observer(
        params,
        start,
        speed,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    .expect("construct zero-angular-momentum infalling camera");
    let snapshot =
        std::env::var("AXGEOM_SNAPSHOT").unwrap_or_else(|_| "kerr-infall-{frame}.ppm".to_owned());
    assert!(
        snapshot.contains("{frame}"),
        "infall mode requires {{frame}} snapshots"
    );
    let mut framebuffer = vec![0; config.width * config.height];
    println!(
        "kerr: infall from r={:.3} through r_+={:.3}, cutoff {interior_cutoff:.3}; {frames} camera frame(s)",
        kerr_radius(params, observer.tetrad.position),
        horizon_radius(params),
    );
    for frame in 0..frames {
        let radius = kerr_radius(params, observer.tetrad.position);
        if radius <= interior_cutoff {
            println!("kerr: infall stopped at interior cutoff after {frame} frame(s)");
            break;
        }
        println!("kerr: infall frame {frame}/{frames}, observer r={radius:.4}");
        let samples = trace_scene(
            runtime,
            backend_label,
            device_model,
            config,
            params,
            trace,
            &observer.tetrad,
            interior_cutoff,
            true,
            |_| true,
        )
        .expect("headless infall trace cannot be cancelled");
        print_event_counts(&samples);
        let refinements = refine_projection_boundaries(
            &samples,
            config,
            params,
            trace,
            &observer.tetrad,
            interior_cutoff,
            true,
        );
        shade(
            &samples,
            refinements.as_ref(),
            config,
            trace,
            plasma,
            sky_texture,
            frame as f32 * proper_step * substeps as f32,
            &mut framebuffer,
        );
        let path = snapshot_path(&snapshot, frame).expect("infall snapshot pattern");
        write_ppm(&path, config, &framebuffer);
        for _ in 0..substeps {
            advance_infalling_observer(params, &mut observer, proper_step, trace.gradient_epsilon)
                .expect("advance infalling camera");
        }
    }
}

#[cfg(feature = "window")]
#[allow(clippy::too_many_arguments)]
fn run_windowed<C: Accelerator, A: Accelerator + TosaCapabilityProvider>(
    runtime: &mut AxHetero<C, A>,
    backend_label: &str,
    device_model: &AxDeviceModel,
    config: RenderConfig,
    params: &KerrParams,
    trace: &TraceConfig,
    camera: &CameraTetrad,
    plasma: &PlasmaVolume,
    sky_texture: Option<&SkyTexture>,
) {
    let mut framebuffer = vec![pack_aces([0.0005, 0.001, 0.002]); config.width * config.height];
    let max_frames: Option<u64> = std::env::var("AXGEOM_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok());
    let mut window = Window::new(
        "axiom kerr - initializing boundary transport",
        config.width,
        config.height,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("open Kerr window");
    let uncap = std::env::var("AXGEOM_UNCAP").is_ok_and(|value| value != "0");
    window.set_target_fps(if max_frames.is_some() || uncap { 0 } else { 60 });
    window
        .update_with_buffer(&framebuffer, config.width, config.height)
        .expect("present initial Kerr frame");

    let trace_epoch = Instant::now();
    let samples = trace_scene(
        runtime,
        backend_label,
        device_model,
        config,
        params,
        trace,
        camera,
        axgeom::black_hole::horizon_radius(params),
        false,
        |partial| {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                return false;
            }
            shade_available(
                partial,
                config,
                trace,
                plasma,
                sky_texture,
                0.0,
                &mut framebuffer,
            );
            let percent = partial.len() as f64 / config.sample_count() as f64 * 100.0;
            window.set_title(&format!(
                "axiom kerr - tracing boundary map {percent:.1}% ({:.1}s)",
                trace_epoch.elapsed().as_secs_f64(),
            ));
            window
                .update_with_buffer(&framebuffer, config.width, config.height)
                .is_ok()
        },
    );
    let Some(samples) = samples else {
        return;
    };
    print_event_counts(&samples);
    let refinements = refine_projection_boundaries(
        &samples,
        config,
        params,
        trace,
        camera,
        horizon_radius(params),
        false,
    );

    let epoch = Instant::now();
    let mut frame = 0_u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0_u64;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        shade(
            &samples,
            refinements.as_ref(),
            config,
            trace,
            plasma,
            sky_texture,
            epoch.elapsed().as_secs_f32(),
            &mut framebuffer,
        );
        window
            .update_with_buffer(&framebuffer, config.width, config.height)
            .expect("present Kerr frame");
        frame += 1;
        if last_title.elapsed().as_secs_f32() >= 0.5 {
            let fps = (frame - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axiom kerr - cached geodesics + {} plasma - {fps:.1} fps - {}t",
                plasma.storage.label(),
                config.threads.min(config.height),
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
        "kerr: {frame} animated holographic+voxel frames in {seconds:.2}s ({:.2} fps)",
        frame as f32 / seconds.max(f32::EPSILON),
    );
    if let Ok(path) = std::env::var("AXGEOM_SNAPSHOT") {
        write_ppm(&path, config, &framebuffer);
    }
}

fn main() {
    let config = RenderConfig::from_env();
    let params = KerrParams::REFERENCE;
    let trace = TraceConfig {
        step_size: config.step_size,
        max_steps: config.max_steps,
        escape_radius: 32.0,
        ..TraceConfig::REFERENCE
    };
    let camera_position = [
        env_or("AXGEOM_CAMERA_X", 0.0),
        env_or("AXGEOM_CAMERA_Y", -22.0),
        env_or("AXGEOM_CAMERA_Z", 7.5),
    ];
    let camera_target = [
        env_or("AXGEOM_TARGET_X", 0.0),
        env_or("AXGEOM_TARGET_Y", 0.0),
        env_or("AXGEOM_TARGET_Z", 0.0),
    ];
    let camera = static_camera_tetrad(&params, camera_position, camera_target, [0.0, 0.0, 1.0])
        .expect("stationary Kerr camera");
    println!(
        "kerr: {}x{}, {} spp, {} steps, step {}, tile {}, {}, shade {} thread(s)",
        config.width,
        config.height,
        config.samples_per_pixel,
        config.max_steps,
        config.step_size,
        config.tile_rays,
        config.plasma_mode.label(),
        config.threads.min(config.height),
    );
    let (mut runtime, backend_label, device_model) = runtime();
    #[cfg(all(feature = "xdna", axnn_vaccel_xdna_runtime))]
    run_xdna_kerr_probe(&mut runtime, config, &params, &camera);

    let sky_started = Instant::now();
    let sky_texture = match std::env::var("AXGEOM_SKY_TEXTURE") {
        Ok(path) => load_sky_texture(&path).unwrap_or_else(|error| panic!("{error}")),
        Err(_) => {
            let sky_width = env_or("AXGEOM_SKY_WIDTH", 2_048_usize);
            let sky_height = env_or("AXGEOM_SKY_HEIGHT", 1_024_usize);
            SkyTexture::generate(sky_width, sky_height, config.threads, |direction| {
                sky(direction, 0.0, 0.0, 0.0, None)
            })
        }
    };
    println!(
        "kerr: {}x{} sky atlas generated/loaded in {:.1} ms (escaped-ray lookup)",
        sky_texture.width,
        sky_texture.height,
        sky_started.elapsed().as_secs_f64() * 1_000.0,
    );
    let plasma = PlasmaVolume::build(config, &params, &trace);
    if std::env::var_os("AXGEOM_INFALL").is_some() {
        run_infall_headless(
            &mut runtime,
            &backend_label,
            &device_model,
            config,
            &params,
            &trace,
            &plasma,
            Some(&sky_texture),
        );
        return;
    }
    #[cfg(feature = "window")]
    if std::env::var_os("AXGEOM_HEADLESS").is_none() {
        run_windowed(
            &mut runtime,
            &backend_label,
            &device_model,
            config,
            &params,
            &trace,
            &camera,
            &plasma,
            Some(&sky_texture),
        );
        return;
    }
    run_headless(
        &mut runtime,
        &backend_label,
        &device_model,
        config,
        &params,
        &trace,
        &camera,
        &plasma,
        Some(&sky_texture),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(threads: usize) -> RenderConfig {
        RenderConfig {
            width: 16,
            height: 8,
            tile_rays: 128,
            max_steps: 1,
            samples_per_pixel: 1,
            step_size: 0.18,
            tan_half_fov: 0.58,
            threads,
            plasma_xy: 8,
            plasma_z: 4,
            plasma_half_thickness: 1.8,
            plasma_extinction: 0.65,
            plasma_emission: 0.65,
            plasma_mode: PlasmaMode::Volume,
            plasma_time_scale: 5.0,
            sky_speed: 0.35,
            exposure: 0.45,
            boundary_spp: 4,
            boundary_step_scale: 0.5,
        }
    }

    fn empty_plasma() -> PlasmaVolume {
        PlasmaVolume {
            dims: [1, 1, 1],
            storage: PlasmaStorage::Uf8E4M4,
            extinction: vec![0],
            source_rgb: vec![0; 3],
            temperature_kk: vec![0],
            xy_half_extent: TraceConfig::REFERENCE.disk_outer_radius,
            half_thickness: 1.8,
            kinetic_reference: 1.0,
            blackbody: BlackbodyLut::new(),
            surface_transfer: None,
            surface_field: None,
        }
    }

    #[test]
    fn plasma_storage_selector_is_hermetic_and_defaults_to_uf8e4m4() {
        assert_eq!(PlasmaStorage::parse(None), Ok(PlasmaStorage::Uf8E4M4));
        assert_eq!(
            PlasmaStorage::parse(Some("uf8e4m4")),
            Ok(PlasmaStorage::Uf8E4M4),
        );
        assert_eq!(
            PlasmaStorage::parse(Some("uf8e5m3")),
            Ok(PlasmaStorage::Uf8E5M3),
        );
        assert_eq!(PlasmaStorage::parse(Some("b8")), Ok(PlasmaStorage::B8),);
        assert_eq!(PlasmaStorage::parse(Some("f32")), Err("f32"));
    }

    #[test]
    fn plasma_mode_keeps_reference_volume_and_live_surface_explicit() {
        assert_eq!(PlasmaMode::parse(None), Ok(PlasmaMode::Volume));
        assert_eq!(PlasmaMode::parse(Some("volume")), Ok(PlasmaMode::Volume));
        assert_eq!(PlasmaMode::parse(Some("surface")), Ok(PlasmaMode::Surface));
        assert_eq!(PlasmaMode::parse(Some("fast")), Err("fast"));
    }

    #[test]
    fn cached_retarded_sky_rotation_matches_direct_phase() {
        let direction = normalized([0.31, -0.77, 0.56]);
        let time = 3.25_f32;
        let travel_time = 17.4_f32;
        let angular_speed = 0.035_f32;
        let (retarded_sin, retarded_cos) = (-0.05 * travel_time * angular_speed).sin_cos();
        let (frame_sin, frame_cos) = (time * angular_speed).sin_cos();
        let cached = rotate_z(
            direction,
            frame_sin * retarded_cos + frame_cos * retarded_sin,
            frame_cos * retarded_cos - frame_sin * retarded_sin,
        );
        let direct_phase = (time - 0.05 * travel_time) * angular_speed;
        let (direct_sin, direct_cos) = direct_phase.sin_cos();
        let direct = rotate_z(direction, direct_sin, direct_cos);
        for channel in 0..3 {
            assert!((cached[channel] - direct[channel]).abs() < 1.0e-6);
        }
        let BoundarySample::Sky { retarded_uv, .. } =
            BoundarySample::sky(direction, travel_time, angular_speed)
        else {
            unreachable!()
        };
        let cached_u = retarded_uv[0] + time * angular_speed / core::f32::consts::TAU;
        let direct_u = direct[1].atan2(direct[0]) * (0.5 / core::f32::consts::PI) + 0.5;
        let wrapped_u_error = (cached_u - direct_u)
            .rem_euclid(1.0)
            .min((direct_u - cached_u).rem_euclid(1.0));
        let direct_v = direct[2].clamp(-1.0, 1.0).acos() / core::f32::consts::PI;
        assert!(wrapped_u_error < 1.0e-6);
        assert!((retarded_uv[1] - direct_v).abs() < 1.0e-6);
    }

    #[test]
    fn live_surface_transfer_cache_tracks_reference_spectrum() {
        let blackbody = BlackbodyLut::new();
        let cache = SurfaceTransferLut::new(&blackbody);
        for temperature in [PLASMA_FLOOR_K, 510_000.0, 970_000.0, PLASMA_PEAK_K] {
            for redshift in [0.12, 0.63, 1.0, 1.74, 3.0] {
                let actual = cache.sample(temperature, redshift);
                let shifted = blackbody.sample(display_color_temperature(temperature * redshift));
                let expected = shifted
                    .spectral_rgb
                    .map(|channel| channel * display_radiance_gain(temperature));
                for channel in 0..3 {
                    let relative_error =
                        (actual[channel] - expected[channel]).abs() / expected[channel].max(1.0e-4);
                    assert!(
                        relative_error < 0.03,
                        "relative error {relative_error} at T={temperature}, g={redshift}, channel={channel}",
                    );
                }
            }
        }
    }

    #[test]
    fn every_plasma_storage_builds_finite_non_negative_fields() {
        let mut config = config(1);
        config.plasma_mode = PlasmaMode::Surface;
        let params = KerrParams::REFERENCE;
        let trace = TraceConfig::REFERENCE;
        for storage in [
            PlasmaStorage::B8,
            PlasmaStorage::Uf8E4M4,
            PlasmaStorage::Uf8E5M3,
        ] {
            let plasma = PlasmaVolume::build_with_storage(config, &params, &trace, storage);
            let voxel_count = (plasma.dims[0] * plasma.dims[1] * plasma.dims[2]) as usize;
            assert_eq!(plasma.extinction.len(), voxel_count);
            assert_eq!(plasma.temperature_kk.len(), voxel_count);
            assert_eq!(plasma.source_rgb.len(), voxel_count * 3);
            for point in [[0.5, 0.5], [2.25, 3.75], [7.5, 7.5]] {
                let (temperature, density) = plasma
                    .surface_field
                    .as_ref()
                    .expect("surface test config builds the field")
                    .sample(point);
                let volume_point = [point[0], point[1], plasma.dims[2] as f32 * 0.5];
                let expected_temperature = TEMPERATURE_FP8_SCALE_K
                    * storage.sample_trilinear(&plasma.temperature_kk, plasma.dims, volume_point);
                let expected_density =
                    storage.sample_trilinear(&plasma.extinction, plasma.dims, volume_point);
                assert!((temperature - expected_temperature).abs() < 1.0);
                assert!((density - expected_density).abs() < 1.0e-6);
            }

            let decode = |bits| match storage {
                PlasmaStorage::B8 => fp8::e4m3_to_f32(bits),
                PlasmaStorage::Uf8E4M4 => uf::uf8e4m4_to_f32(bits),
                PlasmaStorage::Uf8E5M3 => uf::uf8e5m3_to_f32(bits),
            };
            for bits in plasma
                .extinction
                .iter()
                .chain(&plasma.source_rgb)
                .chain(&plasma.temperature_kk)
            {
                let value = decode(*bits);
                assert!(
                    value.is_finite() && value >= 0.0,
                    "{} decoded illegal plasma value {value} from {bits:#04x}",
                    storage.label(),
                );
            }
        }
    }

    #[test]
    fn cached_sky_boundary_animates_without_retracing() {
        let config = config(1);
        let samples: Vec<_> = (0..config.sample_count())
            .map(|index| {
                BoundarySample::sky(
                    normalized([
                        -0.9 + 1.8 * (index % config.width) as f32 / config.width as f32,
                        -0.6 + 1.2 * (index / config.width) as f32 / config.height as f32,
                        0.7,
                    ]),
                    12.0,
                    config.sky_speed,
                )
            })
            .collect();
        let mut initial = vec![0; config.width * config.height];
        let mut animated = initial.clone();
        let plasma = empty_plasma();
        shade(
            &samples,
            None,
            config,
            &TraceConfig::REFERENCE,
            &plasma,
            None,
            0.0,
            &mut initial,
        );
        shade(
            &samples,
            None,
            config,
            &TraceConfig::REFERENCE,
            &plasma,
            None,
            3.0,
            &mut animated,
        );
        assert_ne!(initial, animated);
    }

    #[test]
    fn equirectangular_sky_texture_wraps_longitude_and_interpolates_latitude() {
        let texture = SkyTexture::new(
            2,
            2,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        );
        let sample = texture.sample_equirectangular([0.0, 1.0, 0.0]);
        assert!((sample[0] - 1.0).abs() < 1.0e-6);
        assert!((sample[1] - 0.5).abs() < 1.0e-6);
        assert_eq!(sample[2], 0.0);
    }

    #[test]
    fn cached_disk_regions_animate_kinetic_blackbody_radiance() {
        let mut config = config(1);
        config.plasma_xy = 32;
        config.plasma_z = 8;
        let params = KerrParams::REFERENCE;
        let trace = TraceConfig::REFERENCE;
        let plasma =
            PlasmaVolume::build_with_storage(config, &params, &trace, PlasmaStorage::Uf8E4M4);
        let camera =
            static_camera_tetrad(&params, [0.0, -22.0, 7.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();
        let orbital_radius = 6.0;
        let point = [
            (orbital_radius * orbital_radius + params.spin * params.spin).sqrt(),
            0.0,
            0.0,
        ];
        let boundary = disk_boundary_transport(
            &params,
            &trace,
            &camera,
            point,
            orbital_radius,
            [-1.0, 0.0, 0.0, 0.0],
            20.0,
        )
        .unwrap();
        assert!(boundary.specific_kinetic_energy > 0.0);
        let initial = disk(config, &trace, &plasma, boundary, 0.0);
        let animated = disk(config, &trace, &plasma, boundary, 11.0);
        assert!(initial.into_iter().all(f32::is_finite));
        assert!(animated.into_iter().all(f32::is_finite));
        assert_ne!(initial, animated);
    }

    #[test]
    fn effective_temperature_obeys_fourth_power_heating_mix() {
        assert_eq!(temperature_from_heating(0.0), PLASMA_FLOOR_K);
        assert_eq!(temperature_from_heating(1.0), PLASMA_PEAK_K);
        let midpoint = temperature_from_heating(0.5);
        let midpoint_fourth = midpoint * midpoint * midpoint * midpoint;
        let floor_fourth = PLASMA_FLOOR_K.powi(4);
        let peak_fourth = PLASMA_PEAK_K.powi(4);
        let expected = 0.5 * (floor_fourth + peak_fourth);
        assert!((midpoint_fourth / expected - 1.0).abs() < 2.0e-6);
    }

    #[test]
    fn thick_disk_flares_to_a_bounded_geometric_aspect_ratio() {
        let inner_ratio = thick_disk_scale_height(4.0, 0.0, 0.8) / 4.0;
        let outer_ratio = thick_disk_scale_height(15.0, 1.0, 0.8) / 15.0;
        assert!((0.10..0.15).contains(&inner_ratio));
        assert!((0.20..0.27).contains(&outer_ratio));
        assert!(outer_ratio > inner_ratio);
    }

    #[test]
    fn disk_hologram_reprojects_terminal_crossing_in_f64() {
        let params = KerrParams::REFERENCE;
        let trace = TraceConfig::REFERENCE;
        let target_radius = 6.0_f64;
        let target_x = (target_radius * target_radius + f64::from(params.spin).powi(2)).sqrt();
        let from = [10.0, (target_x - 0.2) as f32, 0.0, 1.0e-6];
        let to = [13.0, (target_x + 0.4) as f32, 0.0, -2.0e-6];
        let (point, radius, fraction) =
            reproject_disk_crossing_f64(&params, &trace, from, to).unwrap();
        assert!((fraction - 1.0 / 3.0).abs() < 1.0e-8);
        assert!((f64::from(point[0]) - target_x).abs() < 1.0e-6);
        assert!((radius - target_radius as f32).abs() < 1.0e-6);
    }

    #[test]
    fn adaptive_reprojection_selects_only_event_neighborhoods() {
        let mut config = config(1);
        config.width = 3;
        config.height = 3;
        config.samples_per_pixel = 1;
        let mut samples = vec![
            BoundarySample::Sky {
                direction: [1.0, 0.0, 0.0],
                retarded_sin: 0.0,
                retarded_cos: 1.0,
                retarded_uv: [0.5, 0.5],
            };
            9
        ];
        samples[4] = BoundarySample::Capture;
        assert_eq!(
            boundary_pixels(&samples, config),
            (0..9).collect::<Vec<_>>()
        );
        config.boundary_spp = 1;
        assert!(boundary_pixels(&samples, config).is_empty());
    }

    #[test]
    fn snapshot_sequence_uses_zero_padded_frame_numbers() {
        assert_eq!(
            snapshot_path("/tmp/kerr-{frame}.ppm", 12).as_deref(),
            Some("/tmp/kerr-0012.ppm")
        );
        assert_eq!(snapshot_path("/tmp/kerr.ppm", 12), None);
    }

    #[test]
    fn threaded_boundary_shading_is_pixel_identical() {
        let single = config(1);
        let threaded = config(4);
        let samples: Vec<_> = (0..single.sample_count())
            .map(|index| {
                BoundarySample::sky(
                    normalized([0.2 + index as f32 * 0.013, -0.7 + index as f32 * 0.007, 0.6]),
                    8.0 + index as f32 * 0.01,
                    single.sky_speed,
                )
            })
            .collect();
        let plasma = empty_plasma();
        let mut expected = vec![0; single.width * single.height];
        let mut actual = expected.clone();
        shade(
            &samples,
            None,
            single,
            &TraceConfig::REFERENCE,
            &plasma,
            None,
            1.25,
            &mut expected,
        );
        shade(
            &samples,
            None,
            threaded,
            &TraceConfig::REFERENCE,
            &plasma,
            None,
            1.25,
            &mut actual,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "manual 2560x1440 cached-boundary shading probe"]
    fn two_k_cached_boundary_shading_probe() {
        let mut config = RenderConfig::from_env();
        config.width = 2_560;
        config.height = 1_440;
        config.samples_per_pixel = 1;
        let params = KerrParams::REFERENCE;
        let trace = TraceConfig::REFERENCE;
        let plasma = PlasmaVolume::build(config, &params, &trace);
        let sky_texture = SkyTexture::generate(2_048, 1_024, config.threads, |direction| {
            sky(direction, 0.0, 0.0, 0.0, None)
        });
        let disk_percent = env_or("AXGEOM_PROBE_DISK_PERCENT", 42_usize).min(100);
        let samples: Vec<_> = (0..config.sample_count())
            .map(|index| {
                let x = index % config.width;
                let y = index / config.width;
                let population = index % 100;
                if population < disk_percent {
                    let angle = x as f32 / config.width as f32 * core::f32::consts::TAU;
                    let (sin, cos) = angle.sin_cos();
                    BoundarySample::Disk(KerrDiskBoundary {
                        point: [6.0 * cos, 6.0 * sin, 0.0],
                        radius: 6.0,
                        direction: normalized([0.35 * cos, 0.35 * sin, 0.94]),
                        redshift: 0.65 + 0.7 * x as f32 / config.width as f32,
                        local_flux: 0.5,
                        specific_kinetic_energy: 0.12,
                        angular_velocity: 0.06,
                        travel_time: 12.0 + 4.0 * y as f32 / config.height as f32,
                    })
                } else if population < 95 {
                    BoundarySample::sky(
                        normalized([
                            x as f32 / config.width as f32 * 2.0 - 1.0,
                            y as f32 / config.height as f32 * 2.0 - 1.0,
                            0.75,
                        ]),
                        16.0,
                        config.sky_speed,
                    )
                } else if population < 98 {
                    BoundarySample::Capture
                } else {
                    BoundarySample::Unresolved
                }
            })
            .collect();
        let refinement_percent = env_or("AXGEOM_PROBE_REFINEMENT_PERCENT", 2_usize).min(100);
        let refinements = if refinement_percent == 0 || config.boundary_spp == 1 {
            None
        } else {
            let mut lookup = vec![u32::MAX; config.width * config.height];
            let mut refined = Vec::new();
            for (pixel, &sample) in samples.iter().enumerate() {
                if hash_u32(pixel as u32) as usize % 100 < refinement_percent {
                    lookup[pixel] = (refined.len() / config.boundary_spp) as u32;
                    refined.extend(std::iter::repeat_n(sample, config.boundary_spp));
                }
            }
            Some(BoundaryRefinements {
                lookup,
                samples: refined,
                samples_per_pixel: config.boundary_spp,
            })
        };
        let mut framebuffer = vec![0; config.width * config.height];
        let frames = env_or("AXGEOM_PROBE_FRAMES", 3_u32).max(1);
        let started = Instant::now();
        for frame in 0..frames {
            shade(
                &samples,
                refinements.as_ref(),
                config,
                &trace,
                &plasma,
                Some(&sky_texture),
                frame as f32 / 30.0,
                &mut framebuffer,
            );
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "kerr 2K cached-boundary probe: {frames} frame(s), {:.1} ms/frame, {:.2} fps, {} thread(s), {refinement_percent}% boundary refinement, checksum {:#010x}",
            elapsed * 1_000.0 / f64::from(frames),
            f64::from(frames) / elapsed,
            config.threads.min(config.height),
            framebuffer
                .iter()
                .fold(0_u32, |sum, pixel| sum.wrapping_add(*pixel)),
        );
    }

    #[test]
    fn material_ray_midpoint_is_the_rotated_disk_boundary() {
        let plasma = PlasmaVolume {
            dims: [96, 96, 24],
            storage: PlasmaStorage::Uf8E4M4,
            extinction: Vec::new(),
            source_rgb: Vec::new(),
            temperature_kk: Vec::new(),
            xy_half_extent: 15.0,
            half_thickness: 1.8,
            kinetic_reference: 1.0,
            blackbody: BlackbodyLut::new(),
            surface_transfer: None,
            surface_field: None,
        };
        let point = [5.0, -2.0, 0.0];
        let direction = normalized([0.2, 0.7, -0.4]);
        let phase = 0.73_f32;
        let (sin, cos) = (-phase).sin_cos();
        let rotated = [
            cos * point[0] - sin * point[1],
            sin * point[0] + cos * point[1],
            point[2],
        ];
        let rotated_direction = [
            cos * direction[0] - sin * direction[1],
            sin * direction[0] + cos * direction[1],
            direction[2],
        ];
        let (origin, grid_direction, t_max) = plasma.material_ray(rotated, rotated_direction);
        let midpoint = std::array::from_fn::<_, 3, _>(|axis| {
            origin[axis] + 0.5 * t_max * grid_direction[axis]
        });
        let expected = [
            (rotated[0] + plasma.xy_half_extent) * plasma.dims[0] as f32
                / (2.0 * plasma.xy_half_extent),
            (rotated[1] + plasma.xy_half_extent) * plasma.dims[1] as f32
                / (2.0 * plasma.xy_half_extent),
            (rotated[2] + plasma.half_thickness) * plasma.dims[2] as f32
                / (2.0 * plasma.half_thickness),
        ];
        for axis in 0..3 {
            assert!((midpoint[axis] - expected[axis]).abs() < 2.0e-5);
        }
    }
}
