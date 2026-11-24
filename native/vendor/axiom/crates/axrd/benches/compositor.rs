//! Compositor micro-benchmarks — the scene program on the host
//! (`SceneHost`), end to end: full repaints, one-panel damage, a 12-px
//! hover, a flyout opening and a rounded client aperture at 1280×720.
//!
//! Run:  cargo bench -p axrd --features host --bench compositor -- --max-time 3
//!
//! Bytes/s counts framebuffer bytes per full repaint — the ceiling is the
//! raster blit throughput.

#![feature(allocator_api)]

use axnn_cpu::XrgbSurfaceRef;
use axrd::host::SceneHost;
use axrd::{
    Anchor, Bind, Corners, LayoutParams, NodeFlags, NodeId, Offset, Rect, Scene, shadow_halo,
};
use divan::counter::BytesCount;
use std::alloc::Global;

fn main() {
    divan::main();
}

const FB_W: usize = 1280;
const FB_H: usize = 720;
const N: usize = 128;
const TILE: usize = 64;
const ROOT: Rect = Rect {
    x: 0,
    y: 0,
    w: FB_W as u16,
    h: FB_H as u16,
};

type Host = SceneHost<Global, N, TILE>;

fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
    LayoutParams {
        anchor: Anchor::TOP_LEFT,
        offset: Offset { x, y },
        w,
        h,
    }
}

/// A ground, `panels` rounded panels in a grid, and `overlays` translucent
/// fills over them.  Returns the scene, its root and the first panel's id.
fn build_scene(panels: usize, overlays: usize) -> (Scene<N>, NodeId, NodeId) {
    let mut s = Scene::<N>::new();
    let root = s
        .add_root(
            lp(0, 0, FB_W as u16, FB_H as u16),
            Bind::Fill { color: 0x20_2028 },
        )
        .unwrap();
    let mut first = None;
    for i in 0..panels {
        let (col, row) = (i % 12, i / 12);
        let id = s
            .add_child(
                root,
                lp(8 + col as i32 * 104, 8 + row as i32 * 70, 96, 62),
                Bind::RoundedFill {
                    color: 0x40_4048 + (i as u32 * 0x10_0000),
                    radii: Corners::uniform(6),
                },
            )
            .unwrap();
        first.get_or_insert(id);
    }
    for i in 0..overlays {
        s.add_child_with_opacity(
            root,
            lp(200 + i as i32 * 90, 300, 160, 120),
            Bind::Fill { color: 0xFF_FFFF },
            96,
        )
        .unwrap();
    }
    (s, root, first.unwrap())
}

fn no_source(_: u32) -> Option<XrgbSurfaceRef<'static>> {
    None
}

fn frame(host: &mut Host, scene: &mut Scene<N>, fb: &mut [u32]) {
    host.paint(scene, fb, &no_source, None, &[]).unwrap();
}

#[divan::bench]
fn full_repaint_128_nodes(bencher: divan::Bencher) {
    let (mut scene, _, _) = build_scene(110, 16);
    let mut host = Host::new_in(&mut scene, ROOT, Global).unwrap();
    let mut fb = vec![0u32; FB_W * FB_H];
    frame(&mut host, &mut scene, &mut fb);
    bencher
        .counter(BytesCount::new(FB_W * FB_H * 4))
        .bench_local(|| {
            scene.add_damage(ROOT);
            frame(&mut host, &mut scene, &mut fb);
        });
}

#[divan::bench]
fn incremental_one_panel(bencher: divan::Bencher) {
    let (mut scene, _, panel) = build_scene(110, 16);
    let mut host = Host::new_in(&mut scene, ROOT, Global).unwrap();
    let mut fb = vec![0u32; FB_W * FB_H];
    frame(&mut host, &mut scene, &mut fb);
    let mut tick = 0u32;
    bencher.bench_local(|| {
        tick = tick.wrapping_add(1);
        scene
            .set_bind(
                panel,
                Bind::RoundedFill {
                    color: 0x40_4048 + tick,
                    radii: Corners::uniform(6),
                },
            )
            .unwrap();
        frame(&mut host, &mut scene, &mut fb);
    });
}

/// A 12-px pip changes colour: one tile repaints.
#[divan::bench]
fn hover_pip_12px(bencher: divan::Bencher) {
    let (mut scene, root, _) = build_scene(60, 4);
    let pip = scene
        .add_child(
            root,
            lp(20, 20, 12, 12),
            Bind::RoundedFill {
                color: 0xC6_483C,
                radii: Corners::uniform(6),
            },
        )
        .unwrap();
    let mut host = Host::new_in(&mut scene, ROOT, Global).unwrap();
    let mut fb = vec![0u32; FB_W * FB_H];
    frame(&mut host, &mut scene, &mut fb);
    let mut on = false;
    bencher.bench_local(|| {
        on = !on;
        scene
            .set_bind(
                pip,
                Bind::RoundedFill {
                    color: if on { 0xE0_6050 } else { 0xC6_483C },
                    radii: Corners::uniform(6),
                },
            )
            .unwrap();
        frame(&mut host, &mut scene, &mut fb);
    });
}

/// A 158×200 flyout with its shadow appears and disappears.
#[divan::bench]
fn flyout_open_158x200(bencher: divan::Bencher) {
    let (mut scene, root, _) = build_scene(60, 4);
    let halo = shadow_halo(8);
    let shadow = scene
        .add_child_with_opacity(
            root,
            lp(
                40 - i32::from(halo),
                60 - i32::from(halo),
                158 + halo * 2,
                200 + halo * 2,
            ),
            Bind::Shadow {
                color: 0,
                radii: Corners::uniform(8),
                falloff: 8,
            },
            0,
        )
        .unwrap();
    let panel = scene
        .add_child_with_opacity(
            root,
            lp(40, 60, 158, 200),
            Bind::RoundedFill {
                color: 0xED_EEEB,
                radii: Corners::uniform(8),
            },
            0,
        )
        .unwrap();
    let mut host = Host::new_in(&mut scene, ROOT, Global).unwrap();
    let mut fb = vec![0u32; FB_W * FB_H];
    frame(&mut host, &mut scene, &mut fb);
    let mut open = false;
    bencher.bench_local(|| {
        open = !open;
        let opacity = if open { 255 } else { 0 };
        scene.set_opacity(shadow, opacity).unwrap();
        scene.set_opacity(panel, opacity).unwrap();
        frame(&mut host, &mut scene, &mut fb);
    });
}

/// A 640×480 client surface inside a rounded aperture, fully damaged.
#[divan::bench]
fn rounded_aperture_640x480_r9(bencher: divan::Bencher) {
    let (mut scene, root, _) = build_scene(24, 0);
    let client = scene
        .add_node(
            Some(root),
            lp(200, 100, 640, 480),
            Bind::Surface { id: 3 },
            255,
            NodeFlags::CLIP_ROUNDED,
        )
        .unwrap();
    scene.set_radii(client, Corners::uniform(9)).unwrap();
    let words: Vec<u32> = (0..640 * 480)
        .map(|i: u32| 0xFF00_0000 | (i % 640) << 8 | (i / 640) & 0xFF)
        .collect();
    let source =
        |slot: u32| (slot == 3).then(|| XrgbSurfaceRef::new(&words, 640, 480, 640).unwrap());
    let mut host = Host::new_in(&mut scene, ROOT, Global).unwrap();
    let mut fb = vec![0u32; FB_W * FB_H];
    host.paint(&mut scene, &mut fb, &source, None, &[]).unwrap();
    let damage = Rect {
        x: 200,
        y: 100,
        w: 640,
        h: 480,
    };
    bencher
        .counter(BytesCount::new(640usize * 480 * 4))
        .bench_local(|| {
            scene.add_damage(damage);
            host.paint(&mut scene, &mut fb, &source, None, &[]).unwrap();
        });
}
