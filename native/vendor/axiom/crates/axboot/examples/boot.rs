#![feature(allocator_api)]
//! The exact production asset, executed through ax admission and axnn plans.
use axboot::{Clock, ComposeInputs, Frame, Hole, HoleImage, HoleInputs, TILE_HEIGHT, TILE_WIDTH, Viewport};
use axnn_vaccel::{
    AxDeviceModel, AxHetero, AxTosaAccelerator, NativeResourceBinding, RgbVolumeRef, Session,
};
use std::{
    alloc::Global,
    io::Write,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn clock() -> Clock {
    let seconds = std::env::var("AXBOOT_UNIX_SECONDS")
        .ok()
        .map(|s| s.parse().expect("UNIX seconds"))
        .or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
        });
    Clock::from_unix_seconds(seconds)
}
fn main() {
    let size = std::env::var("AXBOOT_SIZE").unwrap_or_else(|_| "1920x1200".into());
    let (w, h) = size.split_once('x').expect("AXBOOT_SIZE=WIDTHxHEIGHT");
    let mut viewport = Viewport::new(w.parse().unwrap(), h.parse().unwrap()).unwrap();
    let session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::<_, AxTosaAccelerator>::new(session, None);
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let hole_graph = ax::Graph::parse(axboot::HOLE_GRAPH).unwrap();
    let compose_graph = ax::Graph::parse(axboot::COMPOSE_GRAPH).unwrap();
    let hole_exe = runtime.compile_in(&hole_graph, &model, Global).expect("admit the hole pass");
    let compose_exe = runtime.compile_in(&compose_graph, &model, Global).expect("admit the compose pass");
    eprintln!(
        "boot: hole {} nodes, compose {} nodes",
        hole_graph.node_count(),
        compose_graph.node_count()
    );
    // The baked turbulence table and sky atlas bind as the hole pass's RGB
    // volumes.
    let volume = |points, dims, (group, binding): (u32, u32)| {
        let key = ax::sampling::ResourceKey { group, binding };
        NativeResourceBinding::rgb_volume(key, RgbVolumeRef::new_uf16(points, dims).unwrap())
    };
    let resources = [
        volume(axboot::turbulence(), axboot::TURBULENCE_DIMS, axboot::TURBULENCE_BINDING),
        volume(axboot::sky_atlas(), axboot::SKY_DIMS, axboot::SKY_BINDING),
    ];
    let mut hole_inputs = HoleInputs::new_in(Global).unwrap();
    let mut compose_inputs = ComposeInputs::new_in(Global).unwrap();
    let mut hole_pixels = Vec::new();
    let mut pixels = Vec::new();
    let pack = |r: f32, g: f32, b: f32| {
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        byte(r) << 16 | byte(g) << 8 | byte(b)
    };
    let mut render = |viewport: Viewport, clock: Clock, frame: Frame| {
        let started = Instant::now();
        let [width, height] = viewport.size().map(|v| v as usize);
        let hole = viewport.hole_frame();
        let [hole_width, hole_height] = hole.size().map(|v| v as usize);
        pixels.resize(width * height, 0u32);
        hole_pixels.resize(hole_width * hole_height, 0u32);
        let (mut hole_ms, mut compose_ms, mut composed) = (0.0f64, 0.0f64, 0);
        let as_bytes = |p: &[f32]| unsafe {
            // F32 slices are initialized; byte views are read-only and scoped
            // to the synchronous graph invocation.
            std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), std::mem::size_of_val(p))
        };
        // Pass 1: the hole, at the hole frame.
        let pass = Instant::now();
        for y in (0..hole_height).step_by(TILE_HEIGHT) {
            for x in (0..hole_width).step_by(TILE_WIDTH) {
                hole_inputs.prepare(hole, [x as u32, y as u32], frame).unwrap();
                let bytes = hole_inputs.planes().map(as_bytes);
                let output = runtime.run_with_resources_in(&hole_exe, &bytes, &resources, Global).unwrap();
                let channel = |c: usize, i: usize| f32::from_le_bytes(output[c][i * 4..i * 4 + 4].try_into().unwrap());
                for row in 0..TILE_HEIGHT.min(hole_height - y) {
                    for column in 0..TILE_WIDTH.min(hole_width - x) {
                        let i = row * TILE_WIDTH + column;
                        hole_pixels[(y + row) * hole_width + x + column] = pack(channel(0, i), channel(1, i), channel(2, i));
                    }
                }
            }
        }
        hole_ms += pass.elapsed().as_secs_f64() * 1000.0;
        // Pass 2: the overlay over the (upsampled) hole, at the display.
        let pass = Instant::now();
        let image = HoleImage { pixels: &hole_pixels, width: hole_width, height: hole_height };
        for y in (0..height).step_by(TILE_HEIGHT) {
            for x in (0..width).step_by(TILE_WIDTH) {
                let overlaid = compose_inputs.prepare(viewport, [x as u32, y as u32], clock, Hole::Image(image)).unwrap();
                let output;
                let planes: [&[f32]; 3] = if overlaid {
                    composed += 1;
                    let bytes = compose_inputs.planes().map(as_bytes);
                    output = runtime.run_with_resources_in(&compose_exe, &bytes, &[], Global).unwrap();
                    core::array::from_fn(|c| unsafe {
                        std::slice::from_raw_parts(output[c].as_ptr().cast::<f32>(), output[c].len() / 4)
                    })
                } else {
                    core::array::from_fn(|c| compose_inputs.hole[c].as_slice())
                };
                for row in 0..TILE_HEIGHT.min(height - y) {
                    for column in 0..TILE_WIDTH.min(width - x) {
                        let i = row * TILE_WIDTH + column;
                        pixels[(y + row) * width + x + column] = pack(planes[0][i], planes[1][i], planes[2][i]);
                    }
                }
            }
        }
        compose_ms += pass.elapsed().as_secs_f64() * 1000.0;
        let tiles = composed;
        eprintln!(
            "boot: {width}x{height} (hole {hole_width}x{hole_height}), {tiles} composed tile(s), {} UTC, {:.1}ms (hole {hole_ms:.1}ms, compose {compose_ms:.1}ms)",
            std::str::from_utf8(&clock.text()).unwrap(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
        pixels.clone()
    };
    if let Ok(path) = std::env::var("AXBOOT_PPM") {
        // AXBOOT_SECONDS sets the animation time (disk and swing) for a still.
        let seconds: f64 = std::env::var("AXBOOT_SECONDS")
            .ok()
            .map(|s| s.parse().expect("seconds"))
            .unwrap_or(0.0);
        let frame = Frame::at_uptime_ns((seconds * 1e9) as u64);
        let pixels = render(viewport, clock(), frame);
        // AXBOOT_STEP: also time one animation step after the still.
        if std::env::var_os("AXBOOT_STEP").is_some() {
            render(viewport, clock(), Frame(frame.0 + 1));
        }
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        let [w, h] = viewport.size();
        write!(out, "P6\n{w} {h}\n255\n").unwrap();
        for p in pixels {
            out.write_all(&[(p >> 16) as u8, (p >> 8) as u8, p as u8])
                .unwrap();
        }
        return;
    }
    let [w, h] = viewport.size();
    let mut window = minifb::Window::new(
        "Kore · production WGSL welcome",
        w as usize,
        h as usize,
        minifb::WindowOptions {
            resize: true,
            scale_mode: minifb::ScaleMode::Stretch,
            ..Default::default()
        },
    )
    .unwrap();
    window.set_target_fps(60);
    let mut previous = None;
    let mut frame = Vec::new();
    let opened = Instant::now();
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let (w, h) = window.get_size();
        if let Ok(v) = Viewport::new(w as u32, h as u32) {
            viewport = v;
        }
        let now = clock();
        let tick = Frame::at_uptime_ns(opened.elapsed().as_nanos() as u64);
        let state = (viewport.size(), now, tick);
        if previous != Some(state) {
            frame = render(viewport, now, tick);
            previous = Some(state);
        }
        let [fw, fh] = viewport.size();
        window
            .update_with_buffer(&frame, fw as usize, fh as usize)
            .unwrap();
    }
}
