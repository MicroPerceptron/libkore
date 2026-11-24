#![feature(allocator_api)]

//! WGSL -> `.ax` -> AxHetero procedural UI demo.
//!
//! Run the portable host path with:
//!
//! ```text
//! cargo run --release -p axrd --example wgsl --features "window,graph"
//! ```
//!
//! Add `coreml` on macOS or `openvino` on a host with `libopenvino_c` to use
//! the same optional providers as the boot and axui examples. The environment
//! contract is deliberately parallel to those demos:
//!
//! - `AXRD_WGSL_BACKEND=auto|cpu|accelerator|heterogeneous`
//! - `AXRD_WGSL_OPENVINO_DEVICE=<device>`
//! - `AXRD_WGSL_SIZE=WIDTHxHEIGHT`
//! - `AXRD_WGSL_HEADLESS=1` and optionally `AXRD_WGSL_PPM=<path>`
//! - `AXRD_WGSL_SCENE=shell` with the `font` feature for the Kore shell study
//! - `AXRD_WGSL_THEME=day|night`, `AXRD_WGSL_LAYOUT=tiled|floating`
//! - `AXRD_WGSL_AX=<path>` to export the composed canonical graph
//! - `AXRD_WGSL_TILE=<edge>` (shell defaults to 480; 0 disables tiling)
//!
//! The host only specializes the embedded source, supplies WGSL position
//! planes, and packs completed graph outputs for the window. There is no
//! second CPU implementation of the scene.
//! See `examples/WGSL_SHELL.md` for the shell's approximations, tile ABI, and
//! measured limits; its status labels are mock data, not live telemetry.

use ax::{AdmissionPlanner, DispatchLane, DispatchRequest};
#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
use axnn_vaccel::Accelerator;
use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxDispatchVariant, AxExecutable, AxHetero, AxTosaAccelerator,
    CPU_BACKEND, CPU_KERNEL_IMPLEMENTATION, Session, admit_variants,
};
use axshader::{CompileOptions, compile_wgsl_in};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use std::{
    alloc::{Allocator, Global},
    path::Path,
    time::{Duration, Instant},
};

#[cfg(feature = "font")]
#[path = "support/wgsl_shell.rs"]
mod shell;

#[derive(Clone, Copy)]
enum SceneKind {
    Nocturne,
    #[cfg(feature = "font")]
    Shell(shell::Options),
}

impl SceneKind {
    fn from_env() -> Self {
        match std::env::var("AXRD_WGSL_SCENE").as_deref() {
            Err(_) | Ok("nocturne") => Self::Nocturne,
            #[cfg(feature = "font")]
            Ok("shell") => Self::Shell(shell::Options::from_env()),
            #[cfg(not(feature = "font"))]
            Ok("shell") => panic!("the shell study requires --features window,graph,font"),
            _ => panic!("AXRD_WGSL_SCENE must be nocturne or shell"),
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Nocturne => "axrd · WGSL nocturne",
            #[cfg(feature = "font")]
            Self::Shell(_) => "axrd · Kore shell · T layout / N theme",
        }
    }

    fn tile_edge(self) -> usize {
        let default = match self {
            Self::Nocturne => 0,
            #[cfg(feature = "font")]
            Self::Shell(_) => 480,
        };
        std::env::var("AXRD_WGSL_TILE").map_or(default, |value| {
            let edge: usize = value.parse().expect("AXRD_WGSL_TILE must be an integer");
            assert!(
                edge == 0 || (32..=3840).contains(&edge),
                "tile edge must be 0 or 32..=3840"
            );
            edge
        })
    }

    fn handle_keys(&mut self, window: &Window) -> bool {
        #[cfg(feature = "font")]
        if let Self::Shell(options) = self {
            let theme = window.is_key_pressed(Key::N, minifb::KeyRepeat::No);
            let layout = window.is_key_pressed(Key::T, minifb::KeyRepeat::No);
            options.night ^= theme;
            options.floating ^= layout;
            return theme || layout;
        }
        let _ = window;
        false
    }
}

const SHADER_TEMPLATE: &str = include_str!("../assets/shaders/nocturne.wgsl");
const DEFAULT_WIDTH: usize = 800;
const DEFAULT_HEIGHT: usize = 500;
const MAX_WIDTH: usize = 3_840;
const MAX_HEIGHT: usize = 2_160;
const RESIZE_SETTLE: Duration = Duration::from_millis(140);
const FRAME_DEADLINE_US: u64 = 16_667;
const INITIAL_CPU_FRAME_US: u64 = 9_000;
const INITIAL_ACCELERATOR_FRAME_US: u64 = 6_000;

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
type HostRuntime = AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>;

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
type HostRuntime = AxHetero<AxTosaAccelerator, virtio_accel_coreml::CoreMlAccelerator>;

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
type HostRuntime = AxHetero<AxTosaAccelerator, AxTosaAccelerator>;

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target())
        .expect("failed to create the portable ax CPU provider")
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn host_runtime() -> (HostRuntime, String) {
    let provider = std::env::var("AXRD_WGSL_OPENVINO_DEVICE").map_or_else(
        |_| virtio_accel_openvino::OpenVinoAccelerator::new(),
        |device| virtio_accel_openvino::OpenVinoAccelerator::with_device(&device),
    );
    let (accelerator, label) = match provider {
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
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn host_runtime() -> (HostRuntime, String) {
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
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn host_runtime() -> (HostRuntime, String) {
    (AxHetero::new(cpu_session(), None), "CPU".to_owned())
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn device_model(runtime: &HostRuntime, force_accelerator: bool) -> AxDeviceModel {
    let mut model = AxDeviceModel::default();
    if force_accelerator {
        model.cpu = false;
        model.accelerator_launch_cost = 0;
        model.accelerator_speedup = 64;
        model.transfer_fixed_cost = 0;
        model.transfer_byte_cost = 0;
    }
    runtime.accel.as_ref().map_or(model.clone(), |session| {
        model.with_accelerator_tosa_capability(session.provider(), session.target())
    })
}

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn device_model(runtime: &HostRuntime, force_accelerator: bool) -> AxDeviceModel {
    let mut model = AxDeviceModel::coreml();
    if force_accelerator {
        model.cpu = false;
        model.accelerator_launch_cost = 0;
        model.accelerator_speedup = 64;
        model.transfer_fixed_cost = 0;
        model.transfer_byte_cost = 0;
    }
    runtime.accel.as_ref().map_or(model.clone(), |session| {
        model.with_accelerator_tosa_capability(session.provider(), session.target())
    })
}

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn device_model(_runtime: &HostRuntime, _force_accelerator: bool) -> AxDeviceModel {
    AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendRequest {
    Auto,
    Cpu,
    Accelerator,
    Heterogeneous,
}

fn backend_request() -> BackendRequest {
    match std::env::var("AXRD_WGSL_BACKEND").ok().as_deref() {
        None | Some("auto") => BackendRequest::Auto,
        Some("cpu") => BackendRequest::Cpu,
        Some("accelerator") => BackendRequest::Accelerator,
        Some("heterogeneous") | Some("hetero") => BackendRequest::Heterogeneous,
        Some(value) => panic!(
            "AXRD_WGSL_BACKEND must be auto, cpu, accelerator, or heterogeneous; got {value:?}"
        ),
    }
}

struct RenderedFrame {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
    graph_bytes: usize,
    graph_nodes: usize,
    backend: String,
    compile_elapsed: Duration,
    execute_elapsed: Duration,
    accelerator_nodes: usize,
    tiles: usize,
}

fn valid_size(width: usize, height: usize) -> bool {
    width > 0 && height > 0 && width <= MAX_WIDTH && height <= MAX_HEIGHT
}

fn configured_size() -> (usize, usize) {
    let Some(value) = std::env::var_os("AXRD_WGSL_SIZE") else {
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    };
    let value = value.to_string_lossy();
    let (width, height) = value
        .split_once('x')
        .or_else(|| value.split_once('X'))
        .expect("AXRD_WGSL_SIZE must be WIDTHxHEIGHT");
    let width = width
        .parse::<usize>()
        .expect("invalid AXRD_WGSL_SIZE width");
    let height = height
        .parse::<usize>()
        .expect("invalid AXRD_WGSL_SIZE height");
    assert!(
        valid_size(width, height),
        "AXRD_WGSL_SIZE must be between 1x1 and {MAX_WIDTH}x{MAX_HEIGHT}"
    );
    (width, height)
}

fn specialized_source(width: usize, height: usize) -> String {
    SHADER_TEMPLATE
        .replace("__VIEWPORT_WIDTH__", &format!("{width}.0"))
        .replace("__VIEWPORT_HEIGHT__", &format!("{height}.0"))
}

fn coordinate_axis(length: usize, origin: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(length * size_of::<f32>());
    for index in 0..length {
        bytes.extend_from_slice(&((index + origin) as f32 + 0.5).to_le_bytes());
    }
    bytes
}

fn choose_executable<'a>(
    request: BackendRequest,
    cpu: &'a AxExecutable<Global>,
    alternate: Option<&'a AxExecutable<Global>>,
) -> &'a AxExecutable<Global> {
    match request {
        BackendRequest::Cpu => cpu,
        BackendRequest::Auto | BackendRequest::Accelerator => alternate.unwrap_or(cpu),
        BackendRequest::Heterogeneous => {
            let Some(accelerator) = alternate.and_then(|executable| {
                AxDispatchVariant::new(executable, ACCEL_BACKEND, INITIAL_ACCELERATOR_FRAME_US, 100)
            }) else {
                return alternate.unwrap_or(cpu);
            };
            let cpu = AxDispatchVariant::new(cpu, CPU_BACKEND, INITIAL_CPU_FRAME_US, 100)
                .expect("the portable shader executable occupies the CPU lane");
            let mut lanes = [
                DispatchLane::new(CPU_BACKEND, 0),
                DispatchLane::new(ACCEL_BACKEND, 0),
            ];
            let mut planner =
                AdmissionPlanner::new(&mut lanes).expect("WGSL provider lanes are valid");
            let dispatch = DispatchRequest::new(0, FRAME_DEADLINE_US, 100);
            admit_variants(&mut planner, dispatch, &[cpu, accelerator])
                .expect("WGSL graph has an eligible provider lane")
                .executable()
        }
    }
}

fn placement_label(executable: &AxExecutable<Global>, accelerator_label: &str) -> String {
    match executable.single_backend() {
        Some(CPU_BACKEND)
            if executable
                .schedule()
                .implementations()
                .iter()
                .all(|&implementation| implementation == CPU_KERNEL_IMPLEMENTATION) =>
        {
            "CPU / axnn native kernels".to_owned()
        }
        Some(CPU_BACKEND) => "CPU / TOSA provider".to_owned(),
        Some(ACCEL_BACKEND) => accelerator_label.to_owned(),
        Some(other) => format!("backend {other}"),
        None => format!("CPU + {accelerator_label}"),
    }
}

/// Convert provider-produced RGBA F32 planes to the host window's packed
/// format. This is presentation transfer only; every scene pixel originates
/// in the executed `.ax` graph.
fn graph_channels_to_xrgb<A: Allocator>(
    channels: &[Vec<u8, A>],
    width: usize,
    height: usize,
) -> Vec<u32> {
    assert!(
        matches!(channels.len(), 3 | 4),
        "shader RGBA or composed RGB planes"
    );
    let pixels = width * height;
    assert!(channels.iter().all(|channel| channel.len() == pixels * 4));
    let mut output = Vec::with_capacity(pixels);
    for index in 0..pixels {
        let channel = |plane: usize| {
            let offset = index * 4;
            let value = f32::from_le_bytes(
                channels[plane][offset..offset + 4]
                    .try_into()
                    .expect("F32 shader output plane"),
            );
            assert!(
                value.is_finite(),
                "non-finite shader output at pixel {index}, channel {plane}"
            );
            (value.clamp(0.0, 1.0) * 255.0).round() as u32
        };
        output.push(0xff00_0000 | channel(0) << 16 | channel(1) << 8 | channel(2));
    }
    output
}

fn render_viewport(
    runtime: &mut HostRuntime,
    accelerator_label: &str,
    request: BackendRequest,
    width: usize,
    height: usize,
    scene: SceneKind,
) -> RenderedFrame {
    let edge = scene.tile_edge();
    if edge == 0 || (width <= edge && height <= edge) {
        return render_region(
            runtime,
            accelerator_label,
            request,
            [width, height],
            [0, 0, width, height],
            scene,
        );
    }
    let mut pixels = vec![0; width * height];
    let mut combined = RenderedFrame {
        width,
        height,
        pixels: Vec::new(),
        graph_bytes: 0,
        graph_nodes: 0,
        backend: String::new(),
        compile_elapsed: Duration::ZERO,
        execute_elapsed: Duration::ZERO,
        accelerator_nodes: 0,
        tiles: 0,
    };
    for y in (0..height).step_by(edge) {
        for x in (0..width).step_by(edge) {
            let tw = edge.min(width - x);
            let th = edge.min(height - y);
            let tile = render_region(
                runtime,
                accelerator_label,
                request,
                [width, height],
                [x, y, tw, th],
                scene,
            );
            // Presentation stitching only. All shading and text source-over
            // has already executed in the tile's canonical graph.
            for row in 0..th {
                pixels[(y + row) * width + x..(y + row) * width + x + tw]
                    .copy_from_slice(&tile.pixels[row * tw..(row + 1) * tw]);
            }
            if combined.backend.is_empty() {
                combined.backend = tile.backend;
            } else if combined.backend != tile.backend {
                combined.backend = "mixed tile placements".to_owned();
            }
            combined.graph_bytes += tile.graph_bytes;
            combined.graph_nodes += tile.graph_nodes;
            combined.accelerator_nodes += tile.accelerator_nodes;
            combined.compile_elapsed += tile.compile_elapsed;
            combined.execute_elapsed += tile.execute_elapsed;
            combined.tiles += 1;
        }
    }
    combined.pixels = pixels;
    combined
}

fn render_region(
    runtime: &mut HostRuntime,
    accelerator_label: &str,
    request: BackendRequest,
    viewport: [usize; 2],
    region: [usize; 4],
    scene: SceneKind,
) -> RenderedFrame {
    let [x_origin, y_origin, width, height] = region;
    let compile_started = Instant::now();
    #[cfg(feature = "font")]
    let shell_scene = match scene {
        SceneKind::Shell(options) => Some(shell::Shell::new(options)),
        SceneKind::Nocturne => None,
    };
    let source = match scene {
        SceneKind::Nocturne => specialized_source(viewport[0], viewport[1]),
        #[cfg(feature = "font")]
        SceneKind::Shell(_) => shell_scene
            .as_ref()
            .unwrap()
            .source(viewport[0], viewport[1]),
    };
    let width_u32 = u32::try_from(width).expect("validated WGSL width fits u32");
    let height_u32 = u32::try_from(height).expect("validated WGSL height fits u32");
    // `Global` is selected explicitly here, at the std-only example boundary;
    // axshader and AxHetero retain allocator ownership in their returned data.
    let compiled = compile_wgsl_in(
        &source,
        CompileOptions::for_viewport("main", width_u32, height_u32),
        Global,
    )
    .unwrap_or_else(|error| panic!("failed to compile embedded WGSL scene: {error}"));
    let graph_buf = compiled.into_graph();
    #[cfg(feature = "font")]
    let graph_buf = match shell_scene {
        Some(shell) => shell.compose_text_in(&graph_buf.parse().unwrap(), viewport, region, Global),
        None => graph_buf,
    };
    if let Some(mut path) = std::env::var_os("AXRD_WGSL_AX") {
        if [width, height] != viewport {
            path.push(format!(".tile-{x_origin}-{y_origin}.ax"));
        }
        std::fs::write(path, graph_buf.as_bytes()).expect("failed to export composed .ax graph");
    }
    let graph_bytes = graph_buf.as_bytes().len();
    let graph = graph_buf
        .parse()
        .expect("axshader emitted a validated .ax graph");
    let graph_nodes = graph.node_count();

    let cpu_model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let cpu = runtime
        .compile_in(&graph, &cpu_model, Global)
        .expect("failed to prepare the portable WGSL CPU executable");

    let force_accelerator = matches!(
        request,
        BackendRequest::Accelerator | BackendRequest::Heterogeneous
    );
    let accelerator_model = device_model(runtime, force_accelerator);
    let mut alternate = (request != BackendRequest::Cpu && runtime.accel.is_some())
        .then(|| runtime.compile_in(&graph, &accelerator_model, Global))
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!(
                "wgsl: accelerator graph admission failed ({error:?}); retaining the CPU variant"
            );
            None
        });
    if request == BackendRequest::Auto
        && alternate.as_ref().is_some_and(|executable| {
            !executable
                .schedule()
                .placements()
                .iter()
                .any(|&backend| backend == ACCEL_BACKEND)
        })
    {
        let duplicate = alternate.take().expect("checked above");
        if let Err(error) = runtime.release_executable(duplicate) {
            eprintln!("wgsl: failed to release duplicate CPU provider program: {error:?}");
        }
    }
    let compile_elapsed = compile_started.elapsed();

    let x = coordinate_axis(width, x_origin);
    let y = coordinate_axis(height, y_origin);
    let executable = choose_executable(request, &cpu, alternate.as_ref());
    let backend = placement_label(executable, accelerator_label);
    let accelerator_nodes = executable
        .schedule()
        .placements()
        .iter()
        .filter(|&&backend| backend == ACCEL_BACKEND)
        .count();
    let execute_started = Instant::now();
    let outputs = runtime
        .run_in(executable, &[x.as_slice(), y.as_slice()], Global)
        .expect("failed to execute the compiled WGSL graph through AxHetero");
    let execute_elapsed = execute_started.elapsed();
    let pixels = graph_channels_to_xrgb(&outputs, width, height);

    if let Some(executable) = alternate {
        if let Err(error) = runtime.release_executable(executable) {
            eprintln!("wgsl: failed to release accelerator provider program: {error:?}");
        }
    }
    if let Err(error) = runtime.release_executable(cpu) {
        eprintln!("wgsl: failed to release CPU provider program: {error:?}");
    }

    RenderedFrame {
        width,
        height,
        pixels,
        graph_bytes,
        graph_nodes,
        backend,
        compile_elapsed,
        execute_elapsed,
        accelerator_nodes,
        tiles: 1,
    }
}

fn report(frame: &RenderedFrame) {
    eprintln!(
        "wgsl: {}x{}, {} tile(s), {} scheduled nodes / {} encoded bytes, {} ({} accelerator nodes); compiled in {:.2} ms, executed in {:.2} ms",
        frame.width,
        frame.height,
        frame.tiles,
        frame.graph_nodes,
        frame.graph_bytes,
        frame.backend,
        frame.accelerator_nodes,
        frame.compile_elapsed.as_secs_f64() * 1_000.0,
        frame.execute_elapsed.as_secs_f64() * 1_000.0,
    );
}

fn write_ppm(path: &Path, frame: &RenderedFrame) {
    let header = format!("P6\n{} {}\n255\n", frame.width, frame.height);
    let mut bytes = Vec::with_capacity(header.len() + frame.pixels.len() * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &pixel in &frame.pixels {
        bytes.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8]);
    }
    std::fs::write(path, bytes).expect("failed to write WGSL PPM preview");
}

fn main() {
    let requested_size = configured_size();
    let request = backend_request();
    let mut scene = SceneKind::from_env();
    let (mut runtime, accelerator_label) = host_runtime();

    if std::env::var_os("AXRD_WGSL_HEADLESS").is_some() {
        let frame = render_viewport(
            &mut runtime,
            &accelerator_label,
            request,
            requested_size.0,
            requested_size.1,
            scene,
        );
        report(&frame);
        if let Some(path) = std::env::var_os("AXRD_WGSL_PPM") {
            write_ppm(Path::new(&path), &frame);
        }
        return;
    }

    let mut window = Window::new(
        scene.title(),
        requested_size.0,
        requested_size.1,
        WindowOptions {
            resize: true,
            scale: Scale::X1,
            // Stretch the last completed graph frame through live resize so
            // every component follows the viewport immediately. The shader
            // recompiles at the settled dimensions and applies its own
            // design-space aspect policy without transient window bands.
            scale_mode: ScaleMode::Stretch,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create WGSL demo window");
    window.set_background_color(9, 13, 11);
    window.set_target_fps(60);

    let initial_size = match window.get_size() {
        size if valid_size(size.0, size.1) => size,
        _ => requested_size,
    };
    let mut frame = render_viewport(
        &mut runtime,
        &accelerator_label,
        request,
        initial_size.0,
        initial_size.1,
        scene,
    );
    report(&frame);
    if let Some(path) = std::env::var_os("AXRD_WGSL_PPM") {
        write_ppm(Path::new(&path), &frame);
    }

    let mut observed_size = initial_size;
    let mut resize_started = None;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let scene_changed = scene.handle_keys(&window);
        let size = window.get_size();
        if valid_size(size.0, size.1) && size != observed_size {
            observed_size = size;
            resize_started = Some(Instant::now());
        }
        if scene_changed || resize_started.is_some_and(|started| started.elapsed() >= RESIZE_SETTLE)
        {
            resize_started = None;
            if scene_changed || (frame.width, frame.height) != observed_size {
                frame = render_viewport(
                    &mut runtime,
                    &accelerator_label,
                    request,
                    observed_size.0,
                    observed_size.1,
                    scene,
                );
                report(&frame);
            }
        }
        window
            .update_with_buffer(&frame.pixels, frame.width, frame.height)
            .expect("failed to present WGSL graph output");
    }
}
