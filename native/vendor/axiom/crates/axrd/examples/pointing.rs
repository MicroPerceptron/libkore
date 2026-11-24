#![feature(allocator_api)]

//! Axiom-only pointing lab. See docs/pointing-evaluation.md for controls,
//! measurement limits and reproducible headless checks. Host APIs end here.

use ax::{DType, Graph, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axcomp::{
    ARROW_HEIGHT, ARROW_HOTSPOT, ARROW_RGBA, ARROW_WIDTH, RING_HEIGHT, RING_HOTSPOT, RING_RGBA,
    RING_WIDTH,
};
use axgfx::CursorHotspot;
use axinput::{
    InputTimestamp, PointerSource,
    frame::{MotionContext, MotionPolicy, TimestampOrigin},
    pointing::{AngleGain, PointingTrial},
};
use axnn_vaccel::{AxDeviceModel, AxExecutable, AxHetero, AxTosaAccelerator, Session};
use axrd::{
    Anchor, Bind, Corners, LayoutParams, Offset, PlacedGlyph, Rect, Scene, font8x16,
    graph::{AlphaAtlas, GraphBackground, GraphOutputFormat, compile_scene_graph_with_output_in},
};
use softbuffer::{Context, Surface};
use std::{alloc::System, f64::consts::TAU, io::Write, num::NonZeroU32, rc::Rc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop, OwnedDisplayHandle},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

type Runtime = AxHetero<AxTosaAccelerator, AxTosaAccelerator>;

#[derive(Clone, Copy, Debug)]
enum Shape {
    Arrow,
    Ring,
}

impl Shape {
    fn asset(self) -> (&'static [u8], usize, usize, CursorHotspot) {
        match self {
            Self::Arrow => (&ARROW_RGBA, ARROW_WIDTH, ARROW_HEIGHT, ARROW_HOTSPOT),
            Self::Ring => (&RING_RGBA, RING_WIDTH, RING_HEIGHT, RING_HOTSPOT),
        }
    }
}

struct Graphics {
    runtime: Runtime,
    arrows: [Option<AxExecutable<System>>; 2],
}

impl Graphics {
    fn new() -> Self {
        let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        let mut this = Self {
            runtime: AxHetero::new(cpu, None),
            arrows: [None, None],
        };
        for (i, shape) in [Shape::Arrow, Shape::Ring].iter().enumerate() {
            let (rgba, w, h, _) = shape.asset();
            let dims = [h as u32, w as u32];
            let mut b = GraphBuilder::new_in(System);
            let retained: Vec<_> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| (1.0 - p[3] as f32 / 255.0).to_le_bytes())
                .collect();
            let alpha = b
                .tensor(DType::F32, TensorFlags::CONSTANT, &dims, &retained)
                .unwrap();
            for channel in 0..3 {
                let source: Vec<_> = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| (p[channel] as f32 / 255.0 * (p[3] as f32 / 255.0)).to_le_bytes())
                    .collect();
                let src = b
                    .tensor(DType::F32, TensorFlags::CONSTANT, &dims, &source)
                    .unwrap();
                let dst = b
                    .tensor(DType::F32, TensorFlags::INPUT, &dims, &[])
                    .unwrap();
                let product = b
                    .tensor(DType::F32, TensorFlags::default(), &dims, &[])
                    .unwrap();
                let out = b
                    .tensor(DType::F32, TensorFlags::OUTPUT, &dims, &[])
                    .unwrap();
                b.node(
                    OpCode::MUL,
                    NodeFlags::default(),
                    &[dst, alpha],
                    &[product],
                    &[],
                )
                .unwrap();
                b.node(
                    OpCode::ADD,
                    NodeFlags::default(),
                    &[src, product],
                    &[out],
                    &[],
                )
                .unwrap();
            }
            let graph = b.finish().unwrap();
            this.arrows[i] = Some(this.prepare(graph.parse().unwrap()));
        }
        this
    }
    fn prepare(&mut self, graph: Graph<'_>) -> AxExecutable<System> {
        self.runtime
            .compile_in(
                &graph,
                &AxDeviceModel {
                    accelerator: false,
                    cpu_tosa: false,
                    sparse_bsr: None,
                    ..AxDeviceModel::default()
                },
                System,
            )
            .unwrap()
    }

    fn background(&mut self, lab: &Lab) -> Vec<u32> {
        let (w, h) = lab.size;
        let mut scene = Scene::<64>::new();
        let root = scene.add_root(layout(0, 0, w, h), Bind::Composite).unwrap();
        let mut glyphs = Vec::new();
        let title = if lab.adaptive {
            "AXIOM / POINTING LAB     ANGLE GAIN"
        } else {
            "AXIOM / POINTING LAB     FIXED GAIN"
        };
        let source = if lab.processed {
            "PROCESSED WINDOW MOTION - NOT A RAW-INPUT COMPARISON"
        } else {
            "RAW RELATIVE MOTION - DEVICE UNITS UNCALIBRATED"
        };
        for (text, y, color) in [
            (title, 24, 0xEEE7D8),
            (source, 48, 0xC6BCA8),
            (
                "SPACE capture/release   A gain   C cursor   +/- target size",
                h - 56,
                0xC6BCA8,
            ),
            (
                "Click lit targets. First hit starts timing. ESC releases / quits.",
                h - 32,
                0xC6BCA8,
            ),
        ] {
            let start = glyphs.len();
            let visible = text.len().min((w - 48) / 8);
            for (i, ch) in text.bytes().take(visible).enumerate() {
                glyphs.push(PlacedGlyph {
                    ax: font8x16::atlas_x(ch),
                    ay: 0,
                    aw: 8,
                    ah: 16,
                    dx: (i * 8) as i16,
                    dy: 0,
                    subpixel_x: 0,
                    subpixel_y: 0,
                });
            }
            scene
                .add_child(
                    root,
                    layout(24, y as i32, visible * 8, 16),
                    Bind::TextRun {
                        glyph_start: start as u16,
                        glyph_len: visible as u16,
                        color,
                    },
                )
                .unwrap();
        }
        let radius = lab.radius;
        for i in 0..11 {
            let p = lab.target_at(i);
            scene
                .add_child(
                    root,
                    layout(
                        (p[0] - radius) as i32,
                        (p[1] - radius) as i32,
                        radius as usize * 2,
                        radius as usize * 2,
                    ),
                    Bind::RoundedFill {
                        color: if i == lab.target { 0xEEA374 } else { 0x4E5F56 },
                        radii: Corners::uniform(radius as u16),
                    },
                )
                .unwrap();
            if i == lab.target {
                scene
                    .add_child(
                        root,
                        layout(p[0] as i32 - 2, p[1] as i32 - 2, 4, 4),
                        Bind::RoundedFill {
                            color: 0x24352E,
                            radii: Corners::uniform(2),
                        },
                    )
                    .unwrap();
            }
        }
        let atlas = font8x16::build_atlas_pixels();
        let graph = compile_scene_graph_with_output_in(
            &mut scene,
            Rect {
                x: 0,
                y: 0,
                w: w as u16,
                h: h as u16,
            },
            GraphBackground::Solid(0x24352E),
            Some(AlphaAtlas::new(
                &atlas,
                font8x16::ATLAS_W,
                font8x16::ATLAS_H,
            )),
            &glyphs,
            GraphOutputFormat::PlanarF32,
            System,
        )
        .unwrap();
        let executable = self.prepare(graph.graph());
        let out = self.runtime.run_in(&executable, &[], System).unwrap();
        let pixels = pack(&out);
        self.runtime.release_executable(executable).unwrap();
        pixels
    }

    fn cursor(
        &mut self,
        background: &[u32],
        size: (usize, usize),
        position: [f64; 2],
        shape: Shape,
    ) -> Vec<u32> {
        let (_, w, h, hotspot) = shape.asset();
        let x0 = position[0].round() as i32 - hotspot.x as i32;
        let y0 = position[1].round() as i32 - hotspot.y as i32;
        let mut inputs: [Vec<u8>; 3] = core::array::from_fn(|_| Vec::with_capacity(w * h * 4));
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x0 + x as i32, y0 + y as i32);
                let pixel = if sx >= 0 && sy >= 0 && sx < size.0 as i32 && sy < size.1 as i32 {
                    background[sy as usize * size.0 + sx as usize]
                } else {
                    0
                };
                for (channel, input) in inputs.iter_mut().enumerate() {
                    input.extend_from_slice(
                        &(((pixel >> (16 - channel * 8)) & 255) as f32 / 255.0).to_le_bytes(),
                    );
                }
            }
        }
        let index = match shape {
            Shape::Arrow => 0,
            Shape::Ring => 1,
        };
        let output = self
            .runtime
            .run_in(
                self.arrows[index].as_ref().unwrap(),
                &[&inputs[0], &inputs[1], &inputs[2]],
                System,
            )
            .unwrap();
        let patch = pack(&output);
        let mut pixels = background.to_vec();
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x0 + x as i32, y0 + y as i32);
                if sx >= 0 && sy >= 0 && sx < size.0 as i32 && sy < size.1 as i32 {
                    pixels[sy as usize * size.0 + sx as usize] = patch[y * w + x];
                }
            }
        }
        pixels
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        for executable in &mut self.arrows {
            if let Some(e) = executable.take() {
                let _ = self.runtime.release_executable(e);
            }
        }
    }
}

fn layout(x: i32, y: i32, w: usize, h: usize) -> LayoutParams {
    LayoutParams {
        anchor: Anchor::TOP_LEFT,
        offset: Offset { x, y },
        w: w as u16,
        h: h as u16,
    }
}
fn pack<A: std::alloc::Allocator>(planes: &[Vec<u8, A>]) -> Vec<u32> {
    assert_eq!(planes.len(), 3);
    (0..planes[0].len() / 4)
        .map(|i| {
            let mut pixel = 0;
            for p in planes {
                let f = f32::from_le_bytes(p[i * 4..i * 4 + 4].try_into().unwrap());
                pixel = (pixel << 8) | (f.clamp(0.0, 1.0) * 255.0).round() as u32;
            }
            pixel
        })
        .collect()
}

struct Lab {
    size: (usize, usize),
    position: [f64; 2],
    adaptive: bool,
    shape: Shape,
    processed: bool,
    gain: AngleGain,
    radius: f64,
    target: usize,
    trial: Option<PointingTrial>,
    trial_id: u64,
    epoch: u64,
}
impl Lab {
    fn new(processed: bool) -> Self {
        Self {
            size: (960, 600),
            position: [480.0, 300.0],
            adaptive: false,
            shape: Shape::Arrow,
            processed,
            gain: AngleGain::default(),
            radius: 16.0,
            target: 0,
            trial: None,
            trial_id: 0,
            epoch: 0,
        }
    }
    fn target_at(&self, i: usize) -> [f64; 2] {
        let a = i as f64 * TAU / 11.0;
        [
            (self.size.0 as f64 / 2.0 + a.cos() * (self.size.0 as f64 * 0.33)).round(),
            (self.size.1 as f64 / 2.0 + a.sin() * (self.size.1 as f64 * 0.30)).round(),
        ]
    }
    fn reset(&mut self) {
        self.gain.reset();
        self.trial = None;
        self.epoch += 1;
    }
    fn motion(&mut self, delta: [f64; 2], now: InputTimestamp) {
        if !delta.iter().all(|v| v.is_finite()) {
            self.reset();
            return;
        }
        let d = if self.adaptive {
            let rejected = self.gain.status().rejected_motions;
            // winit's raw deltas are a mouse's, stamped on receipt by this host.
            let context = MotionContext::new(
                PointerSource::Mouse,
                Some(now),
                TimestampOrigin::Received,
            );
            let d = self.gain.apply(delta, context);
            if self.gain.status().rejected_motions != rejected {
                eprintln!("Input segment rejected by bounded gain policy; cancelling trial.");
                self.reset();
                return;
            }
            d
        } else {
            delta
        };
        for (axis, value) in d.iter().enumerate() {
            self.position[axis] = (self.position[axis] + value).clamp(
                0.0,
                (if axis == 0 { self.size.0 } else { self.size.1 } - 1) as f64,
            );
        }
        if let Some(trial) = &mut self.trial {
            trial.observe(self.position, now).unwrap();
        }
    }
    fn click(&mut self, now: InputTimestamp) -> bool {
        if let Some(trial) = &mut self.trial {
            let Some(r) = trial.click(self.position, now).unwrap() else {
                return false;
            };
            self.trial_id += 1;
            println!(
                "{},{},{},{:?},{},{},{},{},{:.6},{:.3},{:.3},{},{},{},{:.6}",
                self.epoch,
                self.trial_id,
                if self.adaptive { "angle" } else { "fixed" },
                self.shape,
                if self.processed {
                    "processed"
                } else {
                    "raw_uncalibrated"
                },
                self.size.0,
                self.size.1,
                self.radius * 2.0,
                r.elapsed_ns as f64 / 1e6,
                r.path_length,
                r.endpoint_error,
                r.target_entries,
                r.axial_overshoots,
                r.missed_clicks,
                r.nominal_difficulty
            );
        } else {
            let t = self.target_at(self.target);
            if (self.position[0] - t[0]).hypot(self.position[1] - t[1]) > self.radius {
                return false;
            }
        }
        self.target = (self.target + 5) % 11;
        // Timing begins after the new target has actually been submitted by
        // redraw(), not while its graph is compiling.
        self.trial = None;
        true
    }
}

struct App {
    context: Context<OwnedDisplayHandle>,
    window: Option<Rc<Window>>,
    surface: Option<Surface<OwnedDisplayHandle, Rc<Window>>>,
    graphics: Graphics,
    lab: Lab,
    background: Vec<u32>,
    captured: bool,
    focused: bool,
    device: Option<DeviceId>,
    last_processed: Option<[f64; 2]>,
    started: Instant,
    input_received: Option<Instant>,
    dirty: bool,
    arm_trial: bool,
}
impl App {
    fn now(&self) -> InputTimestamp {
        InputTimestamp(self.started.elapsed().as_nanos() as u64)
    }
    fn release(&mut self) {
        if let Some(w) = &self.window {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
            w.request_redraw();
        }
        self.captured = false;
        self.device = None;
        self.last_processed = None;
        self.lab.reset();
        self.arm_trial = false;
        self.input_received = None;
    }
    fn invalidate_trial(&mut self) {
        self.lab.reset();
        self.arm_trial = false;
        self.dirty = true;
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
    fn redraw(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let s = window.inner_size();
        if s.width == 0 || s.height == 0 {
            return;
        }
        // Keep allocation bounded; refuse rather than stretch a differently
        // sized hit-test surface. Normal resizes use exact physical pixels.
        if !(640..=1920).contains(&s.width) || !(400..=1200).contains(&s.height) {
            window.set_title("Pointing lab paused: supported size 640x400 through 1920x1200");
            self.lab.reset();
            self.arm_trial = false;
            return;
        }
        let size = (s.width as usize, s.height as usize);
        if size != self.lab.size {
            self.lab.size = size;
            self.lab.position[0] = self.lab.position[0].min((size.0 - 1) as f64);
            self.lab.position[1] = self.lab.position[1].min((size.1 - 1) as f64);
            self.lab.reset();
            self.arm_trial = false;
            self.dirty = true;
        }
        if self.dirty {
            self.background = self.graphics.background(&self.lab);
            self.dirty = false;
        }
        let pixels =
            self.graphics
                .cursor(&self.background, size, self.lab.position, self.lab.shape);
        let surface = self.surface.as_mut().unwrap();
        surface
            .resize(
                NonZeroU32::new(s.width).unwrap(),
                NonZeroU32::new(s.height).unwrap(),
            )
            .unwrap();
        let mut buffer = surface.buffer_mut().unwrap();
        buffer.copy_from_slice(&pixels);
        buffer.present().unwrap();
        if let Some(received) = self.input_received.take() {
            // Host receipt -> present RETURN, not sensor -> photons or a GPU fence.
            window.set_title(&format!(
                "Axiom pointing | {:?} | gain {:.2} | receipt-to-present-return {:.2} ms",
                self.lab.shape,
                if self.lab.adaptive {
                    self.lab.gain.status().gain
                } else {
                    1.0
                },
                received.elapsed().as_secs_f64() * 1000.0
            ));
        } else {
            window.set_title(if self.captured {
                "Axiom pointing | captured; click the lit target"
            } else {
                "Axiom pointing | SPACE to capture; ESC to quit"
            });
        }
        if self.arm_trial {
            self.lab.trial = PointingTrial::new(
                self.lab.position,
                self.lab.target_at(self.lab.target),
                self.lab.radius,
                self.now(),
            )
            .ok();
            self.arm_trial = false;
        }
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Rc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Axiom pointing lab")
                        .with_inner_size(PhysicalSize::new(960, 600))
                        .with_min_inner_size(PhysicalSize::new(640, 400))
                        .with_max_inner_size(PhysicalSize::new(1920, 1200)),
                )
                .unwrap(),
        );
        self.surface = Some(Surface::new(&self.context, Rc::clone(&window)).unwrap());
        window.request_redraw();
        self.window = Some(window);
        event_loop.listen_device_events(DeviceEvents::WhenFocused);
    }
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.release();
    }
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if !self.captured || !self.focused || self.lab.processed {
            return;
        }
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.device.is_some_and(|id| id != device_id) {
                    return;
                }
                self.device = Some(device_id);
                self.input_received = Some(Instant::now());
                self.lab.motion([delta.0, delta.1], self.now());
                self.window.as_ref().unwrap().request_redraw();
            }
            DeviceEvent::Removed if self.device == Some(device_id) => self.release(),
            _ => {}
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                self.release();
                event_loop.exit();
            }
            WindowEvent::Focused(focus) => {
                self.focused = focus;
                if !focus {
                    self.release();
                }
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.invalidate_trial()
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. }
                if self.lab.processed && self.captured && self.focused =>
            {
                let p = [position.x, position.y];
                if let Some(old) = self.last_processed {
                    self.input_received = Some(Instant::now());
                    self.lab.motion([p[0] - old[0], p[1] - old[1]], self.now());
                    self.window.as_ref().unwrap().request_redraw();
                }
                self.last_processed = Some(p);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if self.captured && self.focused && !self.arm_trial => {
                if self.lab.click(self.now()) {
                    self.dirty = true;
                    self.arm_trial = true;
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if self.captured {
                            self.release();
                        } else {
                            event_loop.exit();
                        }
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        if self.captured {
                            self.release();
                        } else {
                            let w = self.window.as_ref().unwrap();
                            let mode = if self.lab.processed {
                                CursorGrabMode::Confined
                            } else {
                                CursorGrabMode::Locked
                            };
                            if self.lab.processed || w.set_cursor_grab(mode).is_ok() {
                                // Processed preview deliberately does not warp/lock.
                                self.captured = true;
                                w.set_cursor_visible(false);
                                self.lab.reset();
                            } else {
                                eprintln!(
                                    "Raw capture unavailable; no automatic processed fallback. Use --processed for a labelled preview."
                                );
                            }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.lab.adaptive = !self.lab.adaptive;
                        self.invalidate_trial();
                    }
                    PhysicalKey::Code(KeyCode::KeyC) => {
                        self.lab.shape = match self.lab.shape {
                            Shape::Arrow => Shape::Ring,
                            Shape::Ring => Shape::Arrow,
                        };
                        self.invalidate_trial();
                    }
                    PhysicalKey::Code(KeyCode::Equal) => {
                        self.lab.radius = (self.lab.radius + 4.0).min(32.0);
                        self.invalidate_trial();
                    }
                    PhysicalKey::Code(KeyCode::Minus) => {
                        self.lab.radius = (self.lab.radius - 4.0).max(8.0);
                        self.invalidate_trial();
                    }
                    _ => {}
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let mut processed = false;
    let mut screenshot = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--processed" => processed = true,
            "--headless" => {
                screenshot = Some(args.next().expect("--headless requires an output PPM path"))
            }
            _ => panic!("Usage: pointing [--processed] [--headless output.ppm]"),
        }
    }
    let mut graphics = Graphics::new();
    let lab = Lab::new(processed);
    if let Some(path) = screenshot {
        let bg = graphics.background(&lab);
        let pixels = graphics.cursor(&bg, lab.size, lab.position, Shape::Arrow);
        let mut file = std::fs::File::create_new(path).expect("use a new output filename");
        write!(file, "P6\n{} {}\n255\n", lab.size.0, lab.size.1).unwrap();
        for p in pixels {
            file.write_all(&[(p >> 16) as u8, (p >> 8) as u8, p as u8])
                .unwrap();
        }
        eprintln!("Headless .ax scene + cursor rendered; no physical input or latency validation.");
        return;
    }
    eprintln!(
        "SPACE captures/releases. A changes policy; C changes cursor; +/- target size. ESC releases then quits."
    );
    eprintln!(
        "Raw device units are uncalibrated. --processed is a preview, not a raw-input experiment."
    );
    println!(
        "epoch,trial,policy,cursor,input,width,height,target_diameter,elapsed_ms,path_px,endpoint_error_px,entries,axial_overshoots,misses,nominal_id"
    );
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let context = Context::new(event_loop.owned_display_handle()).unwrap();
    let mut app = App {
        context,
        window: None,
        surface: None,
        graphics,
        lab,
        background: Vec::new(),
        captured: false,
        focused: false,
        device: None,
        last_processed: None,
        started: Instant::now(),
        input_received: None,
        dirty: true,
        arm_trial: false,
    };
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cursor_damage_is_bounded_and_shape_does_not_move_hotspot() {
        let mut graphics = Graphics::new();
        let size = (80, 60);
        let bg = vec![0x24352E; size.0 * size.1];
        for shape in [Shape::Arrow, Shape::Ring] {
            let (_, w, h, hotspot) = shape.asset();
            for p in [[40.0, 30.0], [0.0, 0.0], [79.0, 59.0]] {
                let output = graphics.cursor(&bg, size, p, shape);
                assert!(output.iter().zip(&bg).any(|(a, b)| a != b));
                for (i, (a, b)) in output.iter().zip(&bg).enumerate() {
                    if a != b {
                        let x = i as i32 % size.0 as i32 - p[0] as i32 + hotspot.x as i32;
                        let y = i as i32 / size.0 as i32 - p[1] as i32 + hotspot.y as i32;
                        assert!(x >= 0 && y >= 0 && x < w as i32 && y < h as i32);
                    }
                }
            }
        }
        // Direct CPU kernel plans are retained in the executables; they need
        // not create TOSA provider programs.
        for executable in graphics.arrows.iter().flatten() {
            assert_eq!(
                executable.schedule().implementations(),
                &[axnn_vaccel::CPU_KERNEL_IMPLEMENTATION; 6]
            );
        }
    }
    #[test]
    fn resize_relayout_and_pointer_gain_are_separate() {
        let mut a = Lab::new(false);
        let mut b = Lab::new(false);
        b.size = (1600, 1000);
        a.adaptive = true;
        b.adaptive = true;
        a.motion([20.0, 10.0], InputTimestamp(1));
        b.motion([20.0, 10.0], InputTimestamp(1));
        assert_eq!(a.position, b.position);
        assert_ne!(a.target_at(0), b.target_at(0));
    }

    #[test]
    fn targets_use_identical_physical_centres_for_drawing_and_hit_testing() {
        let lab = Lab::new(false);
        for i in 0..11 {
            let p = lab.target_at(i);
            for coordinate in p {
                assert_eq!(
                    (coordinate - lab.radius) as i32 as f64 + lab.radius,
                    coordinate
                );
            }
        }
    }

    #[test]
    fn rejected_motion_cancels_a_trial_instead_of_improving_its_score() {
        let mut lab = Lab::new(false);
        lab.adaptive = true;
        lab.trial = Some(
            PointingTrial::new(
                lab.position,
                lab.target_at(0),
                lab.radius,
                InputTimestamp(0),
            )
            .unwrap(),
        );
        let position = lab.position;
        lab.motion([1000.0, 0.0], InputTimestamp(1));
        assert_eq!(lab.position, position);
        assert!(lab.trial.is_none());
        assert_eq!(lab.gain.status().rejected_motions, 1);
    }
}
