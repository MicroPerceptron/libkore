#![feature(allocator_api)]
use axboot::{
    Clock, ComposeInputs, FRAME_NS, Frame, Hole, HoleImage, HoleInputs, ORBIT_SWAY_DEG, TILE_HEIGHT, TILE_PIXELS, TILE_WIDTH,
    Viewport, orbit_angle,
};
use axnn_vaccel::{
    AxDeviceModel, AxHetero, AxTosaAccelerator, Session,
};
use std::alloc::Global;

#[test]
fn clock_is_calendar_time_with_honest_missing_state_and_minute_updates() {
    assert_eq!(Clock::default().text(), *b"--:--");
    assert_eq!(Clock::from_unix_seconds(Some(76740)).text(), *b"21:19");
    assert_eq!(Clock::from_unix_seconds(Some(86399)).text(), *b"23:59");
    assert_eq!(Clock::from_unix_seconds(Some(86400)).text(), *b"00:00");
    assert_eq!(
        Clock::from_unix_seconds(Some(1)),
        Clock::from_unix_seconds(Some(59))
    );
    assert_eq!(Clock::next_update_ns(Some(59_500_000_000)), 500_000_000);
    assert_eq!(Clock::next_update_ns(Some(60_000_000_000)), 60_000_000_000);
    assert_eq!(Clock::next_update_ns(None), 60_000_000_000);
}

#[test]
fn layout_accepts_real_viewports_and_refuses_degenerate_ones() {
    for [w, h] in [[960, 600], [1500, 600], [800, 1200], [3840, 2160]] {
        assert_eq!(Viewport::new(w, h).unwrap().size(), [w, h]);
    }
    assert!(Viewport::new(0, 600).is_err());
    assert!(Viewport::new(1920, 9000).is_err());
}

#[test]
fn the_camera_swings_slowly_within_the_padded_sway() {
    let sway = ORBIT_SWAY_DEG.to_radians();
    assert_eq!(orbit_angle(Frame(0)), 0.0);
    let frames = (axboot::ORBIT_PERIOD_S * 1e9 / FRAME_NS as f64) as u64;
    let mut widest = 0.0f32;
    let mut previous = orbit_angle(Frame(0));
    for f in 1..=frames + 10 {
        let angle = orbit_angle(Frame(f));
        assert!(angle.abs() <= sway * 1.000_001, "frame {f} swings past the atlas padding: {angle}");
        // The fastest step is a few hundredths of a degree: smooth at 15 fps.
        assert!((angle - previous).abs() < 0.03f32.to_radians(), "frame {f} jumps");
        widest = widest.max(angle.abs());
        previous = angle;
    }
    assert!(widest > sway * 0.99, "the swing reaches its ends");
    // Days of uptime keep the swing smooth: its phase is taken in f64.
    let late = Frame(10_000_000);
    assert!((orbit_angle(late) - orbit_angle(Frame(late.0 + 1))).abs() < 0.03f32.to_radians());
}

fn graph_inputs(bytes: &[u8]) -> Vec<Vec<u32>> {
    let graph = ax::Graph::parse(bytes).unwrap();
    let mut inputs = Vec::new();
    for i in 0..graph.tensor_count() {
        let t = graph.tensor(ax::TensorId(i as u32)).unwrap();
        assert!(matches!(t.dtype, ax::DType::F32 | ax::DType::U8));
        if t.flags.contains(ax::TensorFlags::INPUT) {
            inputs.push(t.dimensions().collect::<Vec<_>>());
        }
    }
    inputs
}

#[test]
fn production_asset_abi_is_f32_and_static_across_viewports() {
    // Hole: position x and y; the sky's atlas point and coverage (3), near
    // and far crossings (4 each) as 11 full-tile fields; the hole frame's
    // size, the animation's time and the camera's swing.
    let mut hole = vec![vec![1, 128], vec![32, 1]];
    hole.extend(std::iter::repeat_n(vec![32, 128], 11));
    hole.extend(std::iter::repeat_n(vec![1, 1], 4));
    assert_eq!(graph_inputs(axboot::HOLE_GRAPH), hole);
    // Compose: the hole (3), ink (2) and shape (3) as 8 full-tile fields.
    let compose = graph_inputs(axboot::COMPOSE_GRAPH);
    assert_eq!(compose.iter().filter(|d| **d == vec![32, 128]).count(), 8);
    assert!(compose.iter().all(|d| *d == vec![32, 128] || *d == vec![1, 128] || *d == vec![32, 1]));
}

#[test]
fn panels_taller_than_the_lens_map_draw_the_hole_at_its_height() {
    for ([w, h], expected) in [
        ([1920, 1200], [1920, 1200]),
        ([1437, 897], [1437, 897]),
        ([2880, 1800], [1920, 1200]),
        ([2560, 1600], [1920, 1200]),
        ([3840, 2160], [2134, 1200]),
        ([1200, 1920], [750, 1200]),
    ] {
        let viewport = Viewport::new(w, h).unwrap();
        assert_eq!(viewport.hole_frame().size(), expected, "{w}x{h}");
        assert_eq!(viewport.upsamples_hole(), h > 1200);
    }
}

#[test]
fn the_animation_clock_counts_whole_frames_and_parks_to_the_next() {
    assert_eq!(Frame::at_uptime_ns(0), Frame(0));
    assert_eq!(Frame::at_uptime_ns(FRAME_NS - 1), Frame(0));
    assert_eq!(Frame::at_uptime_ns(FRAME_NS), Frame(1));
    assert_eq!(Frame::next_update_ns(0), FRAME_NS);
    assert_eq!(Frame::next_update_ns(FRAME_NS + 5), FRAME_NS - 5);
    assert!((Frame(15).seconds() - 15.0 * FRAME_NS as f32 / 1e9).abs() < 1e-6);
}

#[test]
fn the_shadow_is_black_and_the_disk_crosses_in_front_of_it() {
    // At 1920x1200 a pixel is one lens texel. The shadow sits right of
    // centre (the camera yaws left); the disk's near side crosses its lower
    // half.
    let viewport = Viewport::new(1920, 1200).unwrap().hole_frame();
    let mut tile = HoleInputs::new_in(Global).unwrap();
    tile.prepare(viewport, [1024, 448], Frame(0)).unwrap();
    let dark = (0..TILE_WIDTH * TILE_HEIGHT)
        .filter(|&i| tile.sky[2][i] == 0.0 && tile.near[0][i] == 0.0)
        .count();
    assert!(dark > TILE_WIDTH * TILE_HEIGHT / 2, "the shadow is captured: {dark} dark texels");
    tile.prepare(viewport, [1024, 704], Frame(0)).unwrap();
    assert!(tile.near[0].iter().any(|&c| c > 0.99), "the disk's near side crosses the shadow");
    // Crossings carry ln r.
    let (ln_inner, ln_outer) = (axboot::kerr::DISK_INNER.ln() - 1e-5, axboot::DISK_OUTER.ln() + 1e-5);
    assert!(tile.near[1].iter().zip(&tile.near[0]).filter(|(_, c)| **c > 0.0).all(|(r, _)| *r >= ln_inner && *r <= ln_outer));
    // Every prepared value is finite.
    for plane in tile.sky.iter().chain(&tile.near).chain(&tile.far) {
        assert!(plane.iter().all(|v| v.is_finite()));
    }
    assert_eq!(axboot::turbulence().len() as u32, axboot::TURBULENCE_DIMS.iter().product::<u32>());
    assert_eq!(axboot::sky_atlas().len() as u32, axboot::SKY_DIMS.iter().product::<u32>());
}

#[test]
fn every_sky_point_stays_inside_its_tier_through_the_whole_swing() {
    // The shader shifts a sky point along its row by at most the sway in its
    // tier's texels; with the bilinear reach it must stay in its rectangle.
    let sway = ORBIT_SWAY_DEG.to_radians();
    let tier_of = |x: f32, y: f32| {
        axboot::kerr::SKY_TIERS.iter().find(|(r, _)| x >= r[0] as f32 && x < r[2] as f32 && y >= r[1] as f32 && y < r[3] as f32)
    };
    let viewport = Viewport::new(2880, 1800).unwrap().hole_frame();
    let mut tile = HoleInputs::new_in(Global).unwrap();
    let mut tiers = [0usize; 4];
    for y in (0..1200).step_by(TILE_HEIGHT) {
        for x in (0..1920).step_by(TILE_WIDTH) {
            tile.prepare(viewport, [x, y], Frame(0)).unwrap();
            for i in 0..TILE_WIDTH * TILE_HEIGHT {
                let (sx, sy) = (tile.sky[0][i], tile.sky[1][i]);
                let (rect, per_radian) = tier_of(sx, sy).unwrap_or_else(|| panic!("({sx}, {sy}) is in no tier"));
                let reach = sway * per_radian;
                assert!(sx - reach >= rect[0] as f32 && sx + reach + 1.0 < rect[2] as f32, "x {sx} leaves its tier at tile {x},{y}");
                assert!(sy >= rect[1] as f32 && sy + 1.0 < rect[3] as f32, "y {sy} leaves its tier at tile {x},{y}");
                assert!((0.0..=1.0).contains(&tile.sky[2][i]));
                if tile.sky[2][i] > 0.0 {
                    tiers[axboot::kerr::SKY_TIERS.iter().position(|t| t.0 == *rect).unwrap()] += 1;
                }
            }
        }
    }
    // From 30 M the hole's Einstein ring spans some 21° of the 31° view, so
    // much of the frame is demagnified past the fine tier; still, the fine
    // tier carries a large share, and every tier is used.
    let total: usize = tiers.iter().sum();
    assert!(tiers[0] * 10 > total * 3 && tiers.iter().all(|&n| n > 0), "tier census {tiers:?}");
}

/// A hole image for `viewport`: a smooth gradient, every pixel distinct
/// enough to show a misread.
fn gradient(viewport: Viewport) -> Vec<u32> {
    let [w, h] = viewport.hole_frame().size().map(|v| v as usize);
    (0..w * h).map(|i| ((i % w * 255 / w) as u32) << 16 | ((i / w * 255 / h) as u32) << 8 | 0x40).collect()
}

#[test]
fn subpixel_font_coverage_is_tile_invariant_and_bounded() {
    let viewport = Viewport::new(1437, 897).unwrap();
    let pixels = gradient(viewport);
    let image = HoleImage { pixels: &pixels, width: 1437, height: 897 };
    let mut a = ComposeInputs::new_in(Global).unwrap();
    let mut b = ComposeInputs::new_in(Global).unwrap();
    let clock = Clock::from_unix_seconds(Some(76740));
    assert!(a.prepare(viewport, [65, 628], clock, Hole::Image(image)).unwrap());
    assert!(b.prepare(viewport, [66, 629], clock, Hole::Image(image)).unwrap());
    assert!(a.ink[0].iter().any(|&v| v > 0.0 && v < 1.0));
    for y in 0..31 {
        for x in 0..127 {
            assert!((a.ink[0][(y + 1) * 128 + x + 1] - b.ink[0][y * 128 + x]).abs() < 0.0001);
        }
    }
    assert!(a.ink.iter().flatten().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
    assert!(a.prepare(viewport, [1437, 0], Clock::default(), Hole::Image(image)).is_err());
    let short = HoleImage { pixels: &pixels[..10], width: 1437, height: 897 };
    assert!(a.prepare(viewport, [0, 0], Clock::default(), Hole::Image(short)).is_err());
}

#[test]
fn a_display_at_the_lens_height_composes_over_the_tiles_own_hole() {
    // The hole pass's planes for the clock's tile on a 1920x1200 display
    // reach the compose pass untouched: no frame, no quantization.
    let viewport = Viewport::new(1920, 1200).unwrap();
    let planes: [Vec<f32>; 3] = core::array::from_fn(|c| (0..TILE_PIXELS).map(|i| (i * (c + 3) % 1000) as f32 / 999.0).collect());
    let hole = Hole::Tile(planes.each_ref().map(Vec::as_slice));
    let mut inputs = ComposeInputs::new_in(Global).unwrap();
    assert!(inputs.prepare(viewport, [128, 832], Clock::from_unix_seconds(Some(76740)), hole).unwrap());
    for c in 0..3 {
        assert_eq!(inputs.hole[c], planes[c]);
    }
    // A display that upsamples the hole needs the whole frame, and a short
    // plane is no tile.
    assert!(inputs.prepare(Viewport::new(2880, 1800).unwrap(), [0, 0], Clock::default(), hole).is_err());
    let short = Hole::Tile([&planes[0][..10], &planes[1], &planes[2]]);
    assert!(inputs.prepare(viewport, [0, 0], Clock::default(), short).is_err());
}

#[test]
fn the_compose_pass_returns_the_hole_wherever_nothing_covers_it() {
    let session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::<_, AxTosaAccelerator>::new(session, None);
    let graph = ax::Graph::parse(axboot::COMPOSE_GRAPH).unwrap();
    let exe = runtime
        .compile_in(
            &graph,
            &AxDeviceModel { accelerator: false, cpu_tosa: false, sparse_bsr: None, ..AxDeviceModel::default() },
            Global,
        )
        .unwrap();
    // The clock's tile on Biggertron's panel: text over an upsampled hole.
    let viewport = Viewport::new(2880, 1800).unwrap();
    let pixels = gradient(viewport);
    let [w, h] = viewport.hole_frame().size().map(|v| v as usize);
    let image = HoleImage { pixels: &pixels, width: w, height: h };
    let mut inputs = ComposeInputs::new_in(Global).unwrap();
    let origin = [256, 1184];
    assert!(inputs.prepare(viewport, origin, Clock::from_unix_seconds(Some(76740)), Hole::Image(image)).unwrap());
    let planes = inputs.planes();
    let bytes = planes.map(|p| unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), std::mem::size_of_val(p)) });
    let outputs = runtime.run_with_resources_in(&exe, &bytes, &[], Global).unwrap();
    let channel = |c: usize, i: usize| f32::from_le_bytes(outputs[c][i * 4..i * 4 + 4].try_into().unwrap());
    let (mut bare, mut covered) = (0, 0);
    for i in 0..TILE_PIXELS {
        let uncovered = inputs.ink.iter().chain(&inputs.shape).all(|p| p[i] == 0.0);
        for c in 0..3 {
            if uncovered {
                assert_eq!(channel(c, i).to_bits(), inputs.hole[c][i].to_bits(), "pixel {i}");
            }
        }
        if uncovered { bare += 1 } else { covered += 1 }
    }
    assert!(bare > 0 && covered > 0, "the tile holds both: {bare} bare, {covered} covered");
}
