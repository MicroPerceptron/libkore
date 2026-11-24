#![feature(allocator_api)]

//! “Nocturne” boot-screen UI study using the bundled scalable-font assets.
//!
//! Run with:
//!   cargo +nightly run -p axrd --example boot --features "window,font,graph"
//! Add `coreml` on macOS or `openvino` on a host with `libopenvino_c` to place
//! the graph through the same `virtio-accel` `Accelerator` contract.
//! Set `AXRD_BOOT_SIZE=WIDTHxHEIGHT` for a specific initial/headless backing.
//! Set `AXRD_BOOT_FRAMES=N` with `AXRD_BOOT_HEADLESS=1` to measure warm
//! retained-executable frames with changing pointer state.
//! Set `AXRD_BOOT_BACKEND=heterogeneous` to compile both the CPU and selected
//! host-accelerator variants and let ax's deadline/capacity admission choose
//! between them in the interactive loop. `cpu` and `accelerator` force one
//! lane; `auto` keeps the calibrated graph placement selected by the model.
//!
//! The scene owns the crisp UI layer. `axrd` lowers fills and text coverage to
//! a canonical mixed-precision `.ax` graph, then the regular scheduler executes
//! it through a provider. The soft depth field is itself ordinary `.ax`
//! arithmetic. Nine scalar inputs move its lights and two normalized pointer
//! inputs translate its shallow foreground layer across warm submissions; no
//! interaction frame rebuilds the graph or adds an axrd raster kernel to any
//! backend.

use ax::{AdmissionPlanner, DispatchDecision, DispatchLane, DispatchRequest};
#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
use axnn_vaccel::Accelerator;
use axnn_vaccel::{
    ACCEL_BACKEND, AxDeviceModel, AxDispatchVariant, AxExecutable, AxHetero, AxInvocation,
    AxInvocationStatus, AxPartitionPhase, AxTosaAccelerator, CPU_BACKEND, Session, admit_variants,
};
use axrd::{
    Anchor, Bind, Corners, FontAtlas, FontFace, LayoutParams, Offset, Parallax, PlacedGlyph, Rect,
    Scene, Viewport,
    graph::{
        AlphaAtlas, GraphBackground, GraphOutputFormat, GraphRadialProfile, GraphRadialRim,
        GraphRadialStyle, compile_scene_graph_with_output_in,
    },
};
use minifb::{Key, MouseMode, Scale, ScaleMode, Window, WindowOptions};
use std::{
    alloc::{Allocator, System},
    time::{Duration, Instant},
};

const DESIGN_WIDTH: u16 = 843;
const DESIGN_HEIGHT: u16 = 530;
const DEFAULT_WIDTH: usize = 1265;
const DEFAULT_HEIGHT: usize = 795;
// Render the larger window at its native resolution. Using minifb's X2 mode
// would enlarge the old pixels and make the font edges look rougher.
const WINDOW_SCALE: Scale = Scale::X1;
const RESIZE_DEBOUNCE: Duration = Duration::from_millis(180);
const FRAME_DEADLINE_US: u64 = 16_667;
const INITIAL_CPU_FRAME_US: u64 = 9_000;
const INITIAL_ACCELERATOR_FRAME_US: u64 = 18_000;
const POINTER_RESPONSE: f32 = 0.18;

// “Nocturne” palette: the retained graph is equally useful as a lighting
// surface, so lean into emissive depth instead of reproducing the reference's
// paper backdrop. Every colour remains ordinary scene/graph data.
const BACKGROUND: u32 = 0x00181E18;
const SURFACE: u32 = 0x0020251F;
const SURFACE_GREEN: u32 = 0x00465638;
const SAGE_GLOW: u32 = 0x003E5038;
const CORAL: u32 = 0x00F08B62;
const CREAM: u32 = 0x00FFE9CC;
const INK: u32 = 0x00F3E9DA;
const SHADOW: u32 = 0x000A0E0B;

const FRAUNCES: &[u8] = include_bytes!("../assets/fonts/Fraunces.ttf");
const DM_SANS: &[u8] = include_bytes!("../assets/fonts/DMSans.ttf");
const DM_MONO: &[u8] = include_bytes!("../assets/fonts/DMMono-Regular.ttf");

fn layout_for_rect(rect: Rect) -> LayoutParams {
    LayoutParams {
        anchor: Anchor::TOP_LEFT,
        offset: Offset {
            x: rect.x,
            y: rect.y,
        },
        w: rect.w,
        h: rect.h,
    }
}

#[derive(Clone, Copy)]
enum InkAlign {
    Start,
    Center,
}

/// Place the visible glyph coverage, rather than the font's invisible line
/// box, inside a design-space slot. Compact controls otherwise expose each
/// face's bearings as apparently inconsistent padding.
#[allow(clippy::too_many_arguments)]
fn append_run_in(
    glyphs: &mut Vec<PlacedGlyph>,
    face: &FontFace<'_>,
    atlas: &mut FontAtlas,
    viewport: Viewport,
    slot: [f32; 4],
    text: &str,
    px: f32,
    horizontal: InkAlign,
    vertical: InkAlign,
) -> (u16, u16) {
    let rect = viewport.snapped_rect(slot[0], slot[1], slot[2], slot[3]);
    // The slots are edge-snapped, so rasterize their text at the same integral
    // phase and keep the alignment invariant at every viewport scale.
    let run = face
        .layout_line(atlas, text, viewport.scale_px(px))
        .expect("boot text layout failed");
    let bounds = run
        .glyphs()
        .iter()
        .filter(|glyph| glyph.aw != 0 && glyph.ah != 0)
        .fold(None::<(i32, i32, i32, i32)>, |bounds, glyph| {
            let glyph_bounds = (
                i32::from(glyph.dx),
                i32::from(glyph.dy),
                i32::from(glyph.dx) + i32::from(glyph.aw),
                i32::from(glyph.dy) + i32::from(glyph.ah),
            );
            Some(bounds.map_or(glyph_bounds, |(left, top, right, bottom)| {
                (
                    left.min(glyph_bounds.0),
                    top.min(glyph_bounds.1),
                    right.max(glyph_bounds.2),
                    bottom.max(glyph_bounds.3),
                )
            }))
        })
        .unwrap_or((0, 0, 0, 0));
    let align = |available: i32, extent: i32, mode: InkAlign| match mode {
        InkAlign::Start => 0,
        InkAlign::Center => (available - extent) / 2,
    };
    let origin_x = rect.x + align(i32::from(rect.w), bounds.2 - bounds.0, horizontal) - bounds.0;
    let origin_y = rect.y + align(i32::from(rect.h), bounds.3 - bounds.1, vertical) - bounds.1;
    let origin_x = i16::try_from(origin_x).expect("boot text x exceeds glyph ABI");
    let origin_y = i16::try_from(origin_y).expect("boot text y exceeds glyph ABI");
    let start = u16::try_from(glyphs.len()).expect("boot demo has fewer than 65K glyphs");
    let len = u16::try_from(run.len()).expect("one text run has fewer than 65K glyphs");
    glyphs.extend(run.glyphs().iter().map(|glyph| {
        let mut glyph = *glyph;
        glyph.dx = glyph
            .dx
            .checked_add(origin_x)
            .expect("boot text x exceeds glyph ABI");
        glyph.dy = glyph
            .dy
            .checked_add(origin_y)
            .expect("boot text y exceeds glyph ABI");
        glyph
    }));
    (start, len)
}

fn graph_channels_to_xrgb<A: Allocator>(
    channels: &[Vec<u8, A>],
    width: usize,
    height: usize,
) -> Vec<u32> {
    assert_eq!(channels.len(), 3);
    let pixels = width * height;
    assert!(channels.iter().all(|channel| channel.len() == pixels * 4));
    let mut output = Vec::with_capacity(pixels);
    for index in 0..pixels {
        let channel = |plane: usize| {
            let offset = index * 4;
            let value = f32::from_le_bytes(
                channels[plane][offset..offset + 4]
                    .try_into()
                    .expect("F32 display plane"),
            );
            (value.clamp(0.0, 1.0) * 255.0).round() as u32
        };
        output.push(0xff00_0000 | (channel(0) << 16) | (channel(1) << 8) | channel(2));
    }
    output
}

fn write_ppm(path: &std::path::Path, pixels: &[u32], width: usize, height: usize) {
    let header = format!("P6\n{width} {height}\n255\n");
    let mut bytes = Vec::with_capacity(header.len() + pixels.len() * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &pixel in pixels {
        bytes.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8]);
    }
    std::fs::write(path, bytes).expect("failed to write boot PPM preview");
}

struct RenderedBoot {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
    encoded_bytes: usize,
    scale: f32,
}

#[derive(Clone, Copy)]
struct BootLight {
    x: f32,
    y: f32,
    strength: f32,
    parallax_x: f32,
    parallax_y: f32,
}

struct PreparedBoot {
    width: usize,
    height: usize,
    cpu_executable: AxExecutable<System>,
    accelerator_executable: Option<AxExecutable<System>>,
    encoded_bytes: usize,
    scale: f32,
    lights: [BootLight; 3],
    graph_inputs: usize,
}

struct BootRenderer<'a> {
    display: FontFace<'a>,
    ui: FontFace<'a>,
    mono: FontFace<'a>,
    runtime: BootRuntime,
    backend_label: String,
}

type BootInvocation<'a> = AxInvocation<'a, 'static, System, System>;

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
type BootRuntime = AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>;

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
type BootRuntime = AxHetero<AxTosaAccelerator, virtio_accel_coreml::CoreMlAccelerator>;

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
type BootRuntime = AxHetero<AxTosaAccelerator, AxTosaAccelerator>;

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target())
        .expect("failed to create ax CPU provider session")
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn boot_runtime() -> (BootRuntime, String) {
    let provider = std::env::var("AXRD_BOOT_OPENVINO_DEVICE").map_or_else(
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
fn boot_runtime() -> (BootRuntime, String) {
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
fn boot_runtime() -> (BootRuntime, String) {
    (AxHetero::new(cpu_session(), None), "CPU".to_owned())
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn device_model(runtime: &BootRuntime, force_accelerator: bool) -> AxDeviceModel {
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
fn device_model(runtime: &BootRuntime, force_accelerator: bool) -> AxDeviceModel {
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
fn device_model(_runtime: &BootRuntime, _force_accelerator: bool) -> AxDeviceModel {
    AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BootBackendRequest {
    Auto,
    Cpu,
    Accelerator,
    Heterogeneous,
}

fn backend_request() -> BootBackendRequest {
    match std::env::var("AXRD_BOOT_BACKEND").ok().as_deref() {
        None | Some("auto") => BootBackendRequest::Auto,
        Some("cpu") => BootBackendRequest::Cpu,
        Some("accelerator") => BootBackendRequest::Accelerator,
        Some("heterogeneous") | Some("hetero") => BootBackendRequest::Heterogeneous,
        Some(value) => panic!(
            "AXRD_BOOT_BACKEND must be auto, cpu, accelerator, or heterogeneous; got {value:?}"
        ),
    }
}

impl BootRenderer<'static> {
    fn new() -> Self {
        let display = FontFace::parse_with_weight(FRAUNCES, 900.0).unwrap();
        let ui = FontFace::parse_with_weight(DM_SANS, 560.0).unwrap();
        let mono = FontFace::parse(DM_MONO).unwrap();
        let (runtime, backend_label) = boot_runtime();
        Self {
            display,
            ui,
            mono,
            runtime,
            backend_label,
        }
    }

    fn prepare(&mut self, width: usize, height: usize) -> PreparedBoot {
        let backing_width = u16::try_from(width).expect("boot backing width exceeds axrd ABI");
        let backing_height = u16::try_from(height).expect("boot backing height exceeds axrd ABI");
        let viewport =
            Viewport::contain(DESIGN_WIDTH, DESIGN_HEIGHT, backing_width, backing_height)
                .expect("boot backing dimensions must be non-zero");
        let atlas_extent = viewport.scale_px(1024.0).ceil().clamp(1024.0, 4096.0) as u16;
        let mut font_atlas = FontAtlas::new(atlas_extent, atlas_extent).unwrap();
        let mut glyphs = Vec::new();

        let brand = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 38.0, 210.0, 9.0],
            "AXIOM  /  NOCTURNE",
            6.2,
            InkAlign::Start,
            InkAlign::Center,
        );
        let brand_state = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 50.0, 210.0, 8.0],
            "LOCAL GRAPH  ·  DEPTH ONLINE",
            5.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let clock = append_run_in(
            &mut glyphs,
            &self.display,
            &mut font_atlas,
            viewport,
            [50.0, 300.0, 250.0, 70.0],
            "21:19",
            70.0,
            InkAlign::Start,
            InkAlign::Start,
        );
        let _greeting = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [50.0, 388.0, 360.0, 14.0],
            "Good evening. Your workspace is warm.",
            12.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let invitation = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [50.0, 406.0, 360.0, 14.0],
            "Nothing moves until you do.",
            12.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let parked = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 429.0, 72.0, 15.0],
            "graph ready",
            5.3,
            InkAlign::Center,
            InkAlign::Center,
        );
        let _held = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [128.0, 429.0, 76.0, 15.0],
            "3 depth planes",
            5.3,
            InkAlign::Center,
            InkAlign::Center,
        );
        let _quiet = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [210.0, 429.0, 68.0, 15.0],
            "quiet boot",
            5.3,
            InkAlign::Center,
            InkAlign::Center,
        );
        let _guide_one = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 456.0, 280.0, 8.0],
            "pointer reveals the distance between layers",
            5.2,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _guide_two = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 467.0, 280.0, 8.0],
            "escape lets the room fall still",
            5.2,
            InkAlign::Start,
            InkAlign::Center,
        );
        let hint = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [50.0, 493.0, 300.0, 9.0],
            "MOVE TO EXPLORE  ·  ESC TO RETURN TO QUIET",
            5.5,
            InkAlign::Start,
            InkAlign::Center,
        );
        let card_start = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [276.0, 145.0, 123.0, 11.0],
            "A small next step",
            8.5,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _weather = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [276.0, 158.0, 123.0, 9.0],
            "Sketch the opening",
            6.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _coat = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [276.0, 170.0, 123.0, 9.0],
            "Everything else can wait",
            6.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _sunlight = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [628.0, 105.0, 92.0, 9.0],
            "FIELD 03 · LIVE",
            6.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _hours = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [628.0, 119.0, 92.0, 14.0],
            "3 planes",
            12.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _through = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [628.0, 135.0, 92.0, 8.0],
            "moving as one",
            6.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _aiden = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [704.0, 450.0, 42.0, 11.0],
            "Cass",
            8.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let _tea = append_run_in(
            &mut glyphs,
            &self.mono,
            &mut font_atlas,
            viewport,
            [704.0, 461.0, 42.0, 8.0],
            "local workspace",
            5.0,
            InkAlign::Start,
            InkAlign::Center,
        );
        let card_end = append_run_in(
            &mut glyphs,
            &self.ui,
            &mut font_atlas,
            viewport,
            [750.0, 449.0, 36.0, 19.0],
            "Enter",
            5.5,
            InkAlign::Center,
            InkAlign::Center,
        );
        let mut scene = Scene::<64>::new();
        let root = scene
            .add_root(layout_for_rect(viewport.root_rect()), Bind::Composite)
            .unwrap();

        let brand_len = brand_state
            .0
            .checked_add(brand_state.1)
            .and_then(|end| end.checked_sub(brand.0))
            .expect("boot brand glyph range overflow");
        scene
            .add_child_with_opacity(
                root,
                layout_for_rect(viewport.root_rect()),
                Bind::TextRun {
                    glyph_start: brand.0,
                    glyph_len: brand_len,
                    color: INK,
                },
                164,
            )
            .unwrap();

        let cards = [
            (260.0, 128.0, 155.0, 58.0),
            (618.0, 91.0, 112.0, 70.0),
            (674.0, 443.0, 120.0, 32.0),
        ];
        let radius = |logical: f32| viewport.scale_px(logical).round() as u16;
        let parallax_q8 = |logical: f32| {
            (viewport.scale_px(logical) * 256.0)
                .round()
                .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
        };
        // The reference frames move the clock/copy plane by roughly a pixel
        // while the depth object travels much farther. Keep this coefficient
        // deliberately subpixel-to-one-pixel so the linearized translation
        // remains crisp rather than turning typography into a soft image warp.
        let shallow_foreground = Parallax::new(parallax_q8(0.7), parallax_q8(0.5));

        // Sparse calibration marks make relative motion legible without
        // outlining the aperture. They are fixed in the glass plane while
        // the luminous field travels behind them.
        for (x, y, width, height) in [
            (479.0, 88.0, 28.0, 1.0),
            (479.0, 88.0, 1.0, 10.0),
            (755.0, 348.0, 28.0, 1.0),
            (782.0, 339.0, 1.0, 10.0),
            (468.0, 322.0, 12.0, 1.0),
        ] {
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(viewport.snapped_rect(x, y, width, height)),
                    Bind::RoundedFill {
                        color: INK,
                        radii: Corners::uniform(radius(0.5)),
                    },
                    58,
                )
                .unwrap();
        }
        // Two restrained, exact-resolution shadow steps establish foreground
        // separation without a host blur kernel. Equal-style rects remain
        // adjacent so the graph compiler can collapse each immutable group
        // before it meets the interactive background.
        for &(x, y, width, height) in &cards {
            let shadow = viewport.snapped_rect(x + 2.0, y + 5.0, width, height);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(shadow),
                    Bind::RoundedFill {
                        color: SHADOW,
                        radii: Corners::uniform(radius(7.0)),
                    },
                    82,
                )
                .unwrap();
        }
        for &(x, y, width, height) in &cards {
            let shadow = viewport.snapped_rect(x + 1.0, y + 2.0, width, height);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(shadow),
                    Bind::RoundedFill {
                        color: SHADOW,
                        radii: Corners::uniform(radius(6.0)),
                    },
                    46,
                )
                .unwrap();
        }
        for &(x, y, width, height) in &cards {
            let edge = viewport.snapped_rect(x - 0.5, y - 0.5, width + 1.0, height + 1.0);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(edge),
                    Bind::RoundedFill {
                        color: INK,
                        radii: Corners::uniform(radius(6.5)),
                    },
                    24,
                )
                .unwrap();
        }
        for &(x, y, width, height) in &cards {
            let card = viewport.snapped_rect(x, y, width, height);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(card),
                    Bind::RoundedFill {
                        color: SURFACE,
                        radii: Corners::uniform(radius(6.0)),
                    },
                    246,
                )
                .unwrap();
        }
        for (x, height) in [
            (628.0, 3.0),
            (638.0, 5.0),
            (648.0, 7.0),
            (658.0, 6.0),
            (668.0, 9.0),
            (678.0, 8.0),
            (688.0, 10.0),
        ] {
            let bar = viewport.snapped_rect(x, 156.0 - height, 7.0, height);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(bar),
                    Bind::RoundedFill {
                        color: CORAL,
                        radii: Corners::uniform(radius(2.0)),
                    },
                    210,
                )
                .unwrap();
        }
        for (rect, color, corner_radius, opacity) in [
            (
                viewport.snapped_rect(681.0, 449.0, 16.0, 20.0),
                SURFACE_GREEN,
                radius(8.0),
                230,
            ),
            (
                viewport.snapped_rect(750.0, 449.0, 36.0, 19.0),
                CORAL,
                radius(10.0),
                230,
            ),
        ] {
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(rect),
                    Bind::RoundedFill {
                        color,
                        radii: Corners::uniform(corner_radius),
                    },
                    opacity,
                )
                .unwrap();
        }

        // Keep the floating-card text in the neutral prepared layer. It is
        // emitted before the lower-left foreground plane so consecutive scene
        // paints retain one coherent parallax coefficient.
        let card_len = card_end
            .0
            .checked_add(card_end.1)
            .and_then(|end| end.checked_sub(card_start.0))
            .expect("boot card glyph range overflow");
        scene
            .add_child_with_opacity(
                root,
                layout_for_rect(viewport.root_rect()),
                Bind::TextRun {
                    glyph_start: card_start.0,
                    glyph_len: card_len,
                    color: INK,
                },
                205,
            )
            .unwrap();

        for (x, width, color) in [
            (50.0, 72.0, SURFACE),
            (128.0, 76.0, SURFACE),
            (210.0, 68.0, SURFACE_GREEN),
        ] {
            let pill = viewport.snapped_rect(x, 429.0, width, 15.0);
            scene
                .add_child_with_opacity(
                    root,
                    layout_for_rect(pill),
                    Bind::RoundedFill {
                        color,
                        radii: Corners::uniform(radius(7.5)),
                    },
                    220,
                )
                .unwrap();
        }

        let muted_len = hint
            .0
            .checked_add(hint.1)
            .and_then(|end| end.checked_sub(parked.0))
            .expect("boot muted glyph range overflow");
        scene
            .add_child_with_opacity(
                root,
                layout_for_rect(viewport.root_rect()),
                Bind::TextRun {
                    glyph_start: parked.0,
                    glyph_len: muted_len,
                    color: INK,
                },
                192,
            )
            .unwrap();
        let ink_len = invitation
            .0
            .checked_add(invitation.1)
            .and_then(|end| end.checked_sub(clock.0))
            .expect("boot ink glyph range overflow");
        let clock_copy = scene
            .add_child(
                root,
                layout_for_rect(viewport.root_rect()),
                Bind::TextRun {
                    glyph_start: clock.0,
                    glyph_len: ink_len,
                    color: INK,
                },
            )
            .unwrap();
        scene.set_parallax(clock_copy, shallow_foreground).unwrap();

        let physical = |x: f32, y: f32| {
            let point = viewport.point(x, y);
            (
                point.x as f32 + f32::from(point.phase_x) / 256.0,
                point.y as f32 + f32::from(point.phase_y) / 256.0,
            )
        };
        let aperture = physical(604.0, 240.0);
        let green = physical(165.0, 315.0);
        let highlight = physical(552.0, 188.0);

        let background = GraphBackground::RadialInputs {
            base: BACKGROUND,
            lights: [
                GraphRadialStyle {
                    radius: viewport.scale_px(355.0),
                    color: SAGE_GLOW,
                    profile: GraphRadialProfile::Rational { sharpness: 1.0 },
                    rim: None,
                },
                GraphRadialStyle {
                    radius: viewport.scale_px(158.0),
                    color: CORAL,
                    // Signed-distance coverage is a display-sampling
                    // concern: hold its transition near one physical pixel
                    // while the object geometry follows the viewport.
                    // Slightly wider than one physical pixel so providers
                    // with narrower internal lanes still resolve a stable
                    // analytical edge instead of quantizing it to a stair.
                    profile: GraphRadialProfile::Disc { softness: 1.25 },
                    rim: Some(GraphRadialRim {
                        strength: 0.18,
                        sharpness: 112.0,
                    }),
                },
                GraphRadialStyle {
                    radius: viewport.scale_px(82.0),
                    color: CREAM,
                    profile: GraphRadialProfile::Rational { sharpness: 2.0 },
                    rim: None,
                },
            ],
        };
        let compiled = compile_scene_graph_with_output_in(
            &mut scene,
            viewport.root_rect(),
            background,
            Some(AlphaAtlas::from_u16(
                font_atlas.coverage(),
                font_atlas.width() as usize,
                font_atlas.height() as usize,
            )),
            &glyphs,
            // Preserve the precision-sensitive compositor through the
            // provider boundary and avoid an unsupported CAST tail on Core
            // ML. The host narrows only after provider completion when it
            // packs the actual display surface.
            GraphOutputFormat::PlanarF32,
            System,
        )
        .expect("failed to compile boot scene to .ax");
        let encoded_bytes = compiled.as_bytes().len();

        let cpu_model = AxDeviceModel {
            accelerator: false,
            ..AxDeviceModel::default()
        };
        let cpu_executable = self
            .runtime
            .compile_in(&compiled.graph(), &cpu_model, System)
            .expect("failed to schedule portable CPU boot compositor graph");
        let request = backend_request();
        let force_accelerator = matches!(
            request,
            BootBackendRequest::Accelerator | BootBackendRequest::Heterogeneous
        );
        let accelerator_executable = (request != BootBackendRequest::Cpu
            && self.runtime.accel.is_some())
        .then(|| {
            self.runtime.compile_in(
                &compiled.graph(),
                &device_model(&self.runtime, force_accelerator),
                System,
            )
        })
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!(
                "boot: accelerator graph admission failed ({error:?}); retaining the CPU variant"
            );
            None
        })
        .filter(|executable| {
            // `auto` asks the placement model for one best executable. Do not
            // retain a duplicate CPU-only plan as an alleged accelerator
            // variant. `heterogeneous` deliberately compiles the second,
            // accelerator-forced plan for runtime lane admission.
            request != BootBackendRequest::Auto
                || executable.single_backend() == Some(ACCEL_BACKEND)
        });
        let (partition_count, accelerator_nodes) = accelerator_executable
            .as_ref()
            .map(|executable| {
                (
                    executable.schedule().plan().partitions.len(),
                    executable
                        .schedule()
                        .placements()
                        .iter()
                        .filter(|&&backend| backend == ACCEL_BACKEND)
                        .count(),
                )
            })
            .unwrap_or_else(|| (cpu_executable.schedule().plan().partitions.len(), 0));
        eprintln!(
            "boot: {} nodes, {} partition(s), {accelerator_nodes} accelerator node(s); runtime {}",
            compiled.graph().node_count(),
            partition_count,
            self.backend_label,
        );
        PreparedBoot {
            width,
            height,
            cpu_executable,
            accelerator_executable,
            encoded_bytes,
            scale: viewport.scale(),
            lights: [
                BootLight {
                    x: green.0,
                    y: green.1,
                    strength: 0.56,
                    parallax_x: viewport.scale_px(-18.0),
                    parallax_y: viewport.scale_px(-13.0),
                },
                BootLight {
                    x: aperture.0,
                    y: aperture.1,
                    strength: 0.90,
                    parallax_x: viewport.scale_px(38.0),
                    parallax_y: viewport.scale_px(26.0),
                },
                BootLight {
                    x: highlight.0,
                    y: highlight.1,
                    strength: 0.78,
                    parallax_x: viewport.scale_px(55.0),
                    parallax_y: viewport.scale_px(40.0),
                },
            ],
            graph_inputs: compiled.input_count() as usize,
        }
    }

    fn begin_render_on<'executable>(
        &self,
        prepared: &PreparedBoot,
        executable: &'executable AxExecutable<System>,
        pointer: (f32, f32),
    ) -> BootInvocation<'executable> {
        let mut values = [0.0_f32; 11];
        for (index, light) in prepared.lights.iter().enumerate() {
            values[index * 3] = light.x + pointer.0 * light.parallax_x;
            values[index * 3 + 1] = light.y + pointer.1 * light.parallax_y;
            values[index * 3 + 2] = light.strength;
        }
        values[9] = pointer.0;
        values[10] = pointer.1;
        let bytes = values.map(f32::to_le_bytes);
        let inputs: Vec<&[u8]> = bytes[..prepared.graph_inputs]
            .iter()
            .map(<[u8; 4]>::as_slice)
            .collect();
        self.runtime
            .begin_owned_in(executable, &inputs, System)
            .expect("failed to prepare boot compositor invocation")
    }

    fn begin_render<'prepared>(
        &self,
        prepared: &'prepared PreparedBoot,
        pointer: (f32, f32),
    ) -> BootInvocation<'prepared> {
        let executable = if backend_request() == BootBackendRequest::Accelerator {
            prepared
                .accelerator_executable
                .as_ref()
                .unwrap_or(&prepared.cpu_executable)
        } else {
            &prepared.cpu_executable
        };
        self.begin_render_on(prepared, executable, pointer)
    }

    fn complete_render(
        prepared: &PreparedBoot,
        outputs: Vec<Vec<u8, System>, System>,
    ) -> RenderedBoot {
        let pixels = graph_channels_to_xrgb(&outputs, prepared.width, prepared.height);
        RenderedBoot {
            width: prepared.width,
            height: prepared.height,
            pixels,
            encoded_bytes: prepared.encoded_bytes,
            scale: prepared.scale,
        }
    }

    fn render(&mut self, prepared: &PreparedBoot, pointer: (f32, f32)) -> RenderedBoot {
        let mut invocation = self.begin_render(prepared, pointer);
        loop {
            match self
                .runtime
                .advance_in(&mut invocation)
                .expect("failed to advance boot compositor graph")
            {
                AxInvocationStatus::Progress(_) => {}
                AxInvocationStatus::Complete(outputs) => {
                    return Self::complete_render(prepared, outputs);
                }
                AxInvocationStatus::Cancelled(progress) => {
                    panic!("boot compositor was cancelled at {progress:?}");
                }
            }
        }
    }
}

fn configured_size() -> (usize, usize) {
    let Some(value) = std::env::var_os("AXRD_BOOT_SIZE") else {
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    };
    let value = value.to_string_lossy();
    let (width, height) = value
        .split_once('x')
        .or_else(|| value.split_once('X'))
        .expect("AXRD_BOOT_SIZE must be WIDTHxHEIGHT");
    let width = width
        .parse::<usize>()
        .expect("invalid AXRD_BOOT_SIZE width");
    let height = height
        .parse::<usize>()
        .expect("invalid AXRD_BOOT_SIZE height");
    assert!(
        width > 0 && height > 0 && width <= u16::MAX as usize && height <= u16::MAX as usize,
        "AXRD_BOOT_SIZE is outside axrd's physical backing ABI"
    );
    (width, height)
}

fn elapsed_micros(epoch: Instant) -> u64 {
    u64::try_from(epoch.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn main() {
    let mut renderer = BootRenderer::new();
    let requested_size = configured_size();
    if std::env::var_os("AXRD_BOOT_HEADLESS").is_some() {
        let frame_count = std::env::var("AXRD_BOOT_FRAMES")
            .ok()
            .map(|value| value.parse::<usize>().expect("invalid AXRD_BOOT_FRAMES"))
            .unwrap_or(1);
        assert!(frame_count > 0, "AXRD_BOOT_FRAMES must be non-zero");
        let prepare_started = Instant::now();
        let prepared = renderer.prepare(requested_size.0, requested_size.1);
        let prepare_elapsed = prepare_started.elapsed();
        let render_started = Instant::now();
        let mut frame = None;
        for index in 0..frame_count {
            let pointer_x = if frame_count == 1 {
                0.0
            } else {
                index as f32 / (frame_count - 1) as f32 * 2.0 - 1.0
            };
            frame = Some(renderer.render(&prepared, (pointer_x, pointer_x * -0.35)));
        }
        let render_elapsed = render_started.elapsed();
        let frame = frame.expect("at least one frame was requested");
        if let Some(path) = std::env::var_os("AXRD_BOOT_PPM") {
            write_ppm(
                std::path::Path::new(&path),
                &frame.pixels,
                frame.width,
                frame.height,
            );
        }
        eprintln!(
            "boot: scheduled {}x{} at {:.4}x ({} encoded bytes); prepared in {:.2} ms, {} warm frame(s) at {:.2} ms/frame",
            frame.width,
            frame.height,
            frame.scale,
            frame.encoded_bytes,
            prepare_elapsed.as_secs_f64() * 1_000.0,
            frame_count,
            render_elapsed.as_secs_f64() * 1_000.0 / frame_count as f64,
        );
        return;
    }

    let mut window = Window::new(
        "axrd · nocturne boot surface",
        requested_size.0,
        requested_size.1,
        WindowOptions {
            resize: true,
            scale: WINDOW_SCALE,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");
    window.set_background_color(
        ((BACKGROUND >> 16) & 0xff) as u8,
        ((BACKGROUND >> 8) & 0xff) as u8,
        (BACKGROUND & 0xff) as u8,
    );
    // The retained native graph now fits inside a 60 Hz CPU frame on the
    // portable provider. Slower or shared backends naturally coalesce pointer
    // targets through the invocation state machine instead of queueing stale
    // frames.
    window.set_target_fps(60);

    let initial_size = match window.get_size() {
        (0, _) | (_, 0) => requested_size,
        size => size,
    };
    let mut prepared = renderer.prepare(initial_size.0, initial_size.1);
    let mut frame = renderer.render(&prepared, (0.0, 0.0));
    if let Some(path) = std::env::var_os("AXRD_BOOT_PPM") {
        write_ppm(
            std::path::Path::new(&path),
            &frame.pixels,
            frame.width,
            frame.height,
        );
    }
    let mut observed_size = initial_size;
    let mut resize_started = None;
    let mut rendered_pointer = (0.0_f32, 0.0_f32);
    let mut target_pointer = rendered_pointer;
    // The host owns the clock, backend-capacity snapshot, and provider-event
    // polling. `ax` owns only admission policy and the reservation timeline.
    let scheduler_epoch = Instant::now();
    let mut lanes = [
        DispatchLane::new(CPU_BACKEND, 0),
        DispatchLane::new(ACCEL_BACKEND, 0),
    ];
    let mut planner = AdmissionPlanner::new(&mut lanes).expect("boot provider lanes are valid");
    let mut estimated_frame_us = [INITIAL_CPU_FRAME_US, INITIAL_ACCELERATOR_FRAME_US];

    'window: loop {
        // One prepared executable is immutable for this scope, allowing its
        // invocation to borrow it across event-loop iterations. Resize exits
        // the scope only after any live provider event has completed.
        let mut submitted_pointer = rendered_pointer;
        let mut invocation: Option<BootInvocation<'_>> = None;
        let mut reservation: Option<DispatchDecision> = None;
        let rebuild = loop {
            let closing = !window.is_open() || window.is_key_down(Key::Escape);
            if closing && invocation.is_none() {
                break None;
            }
            let size = window.get_size();
            if size.0 > 0
                && size.1 > 0
                && size.0 <= u16::MAX as usize
                && size.1 <= u16::MAX as usize
                && size != observed_size
            {
                observed_size = size;
                resize_started = Some(Instant::now());
            }
            if invocation.is_none()
                && resize_started.is_some_and(|started| started.elapsed() >= RESIZE_DEBOUNCE)
            {
                resize_started = None;
                if (frame.width, frame.height) != observed_size {
                    break Some(observed_size);
                }
            }
            if resize_started.is_none()
                && let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Clamp)
            {
                let pointer = (
                    (mouse_x / prepared.width as f32 * 2.0 - 1.0).clamp(-1.0, 1.0),
                    (mouse_y / prepared.height as f32 * 2.0 - 1.0).clamp(-1.0, 1.0),
                );
                if (pointer.0 - rendered_pointer.0).abs() >= 0.0025
                    || (pointer.1 - rendered_pointer.1).abs() >= 0.0025
                {
                    target_pointer = pointer;
                }
            }
            if invocation.is_none()
                && resize_started.is_none()
                && ((target_pointer.0 - rendered_pointer.0).abs() >= 0.0025
                    || (target_pointer.1 - rendered_pointer.1).abs() >= 0.0025
                    || (frame.width, frame.height) != (prepared.width, prepared.height))
            {
                // The pointer is an intent target, not a camera transform.
                // Ease the retained graph inputs so the deep field settles
                // behind the shallow type plane instead of snapping to the
                // cursor at input-device frequency.
                let delta = (
                    target_pointer.0 - rendered_pointer.0,
                    target_pointer.1 - rendered_pointer.1,
                );
                submitted_pointer = (
                    if delta.0.abs() < 0.01 {
                        target_pointer.0
                    } else {
                        rendered_pointer.0 + delta.0 * POINTER_RESPONSE
                    },
                    if delta.1.abs() < 0.01 {
                        target_pointer.1
                    } else {
                        rendered_pointer.1 + delta.1 * POINTER_RESPONSE
                    },
                );
                let ready_at = elapsed_micros(scheduler_epoch);
                let request =
                    DispatchRequest::new(ready_at, ready_at.saturating_add(FRAME_DEADLINE_US), 100);
                let cpu_variant = AxDispatchVariant::new(
                    &prepared.cpu_executable,
                    CPU_BACKEND,
                    estimated_frame_us[CPU_BACKEND as usize],
                    100,
                )
                .expect("portable boot graph needs one CPU capacity reservation");
                let accelerator_variant =
                    prepared
                        .accelerator_executable
                        .as_ref()
                        .and_then(|executable| {
                            AxDispatchVariant::new(
                                executable,
                                ACCEL_BACKEND,
                                estimated_frame_us[ACCEL_BACKEND as usize],
                                100,
                            )
                        });
                let choice = match (backend_request(), accelerator_variant) {
                    (BootBackendRequest::Accelerator, Some(accelerator)) => {
                        admit_variants(&mut planner, request, &[accelerator])
                    }
                    (
                        BootBackendRequest::Auto | BootBackendRequest::Heterogeneous,
                        Some(accelerator),
                    ) => admit_variants(&mut planner, request, &[cpu_variant, accelerator]),
                    _ => admit_variants(&mut planner, request, &[cpu_variant]),
                }
                .expect("boot graph has an eligible provider lane");
                reservation = Some(choice.decision());
                invocation = Some(renderer.begin_render_on(
                    &prepared,
                    choice.executable(),
                    submitted_pointer,
                ));
            }
            let mut completed = None;
            if let Some(active) = invocation.as_mut() {
                // A synchronous provider can expose terminal state on the
                // immediate poll; an asynchronous provider returns
                // Waiting here. Either way, control returns after at most
                // one submission plus one bounded poll.
                for _ in 0..2 {
                    match renderer
                        .runtime
                        .advance_in(active)
                        .expect("failed to advance interactive boot frame")
                    {
                        AxInvocationStatus::Progress(progress) => {
                            if progress.phase != AxPartitionPhase::Submitted {
                                break;
                            }
                        }
                        AxInvocationStatus::Complete(outputs) => {
                            completed = Some(outputs);
                            break;
                        }
                        AxInvocationStatus::Cancelled(progress) => {
                            panic!("interactive boot frame was cancelled at {progress:?}");
                        }
                    }
                }
            }
            if let Some(outputs) = completed {
                let completed_at = elapsed_micros(scheduler_epoch);
                let reservation = reservation
                    .take()
                    .expect("every interactive invocation reserves capacity");
                planner
                    .reconcile(reservation, completed_at)
                    .expect("completed frame is the boot lane tail");
                let observed_us = completed_at.saturating_sub(reservation.start_at).max(1);
                let estimate = &mut estimated_frame_us[reservation.backend as usize];
                *estimate = estimate
                    .saturating_mul(3)
                    .saturating_add(observed_us)
                    .div_ceil(4)
                    .clamp(1, FRAME_DEADLINE_US.saturating_mul(4));
                frame = BootRenderer::complete_render(&prepared, outputs);
                rendered_pointer = submitted_pointer;
                invocation = None;
            }
            if closing {
                continue;
            }
            window
                .update_with_buffer(&frame.pixels, frame.width, frame.height)
                .expect("display failed");
        };
        drop(invocation);
        let Some(size) = rebuild else {
            break 'window;
        };
        prepared = renderer.prepare(size.0, size.1);
    }
}
