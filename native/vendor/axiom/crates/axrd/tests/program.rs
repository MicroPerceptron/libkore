//! The scene program's pixel contracts on the host: every bind kind,
//! damage, occlusion and clipping, painted through the `ax` raster
//! program on the CPU lane (`SceneHost`), band by band — and the same
//! pixels from the dynamic path (a per-frame op table, rows painted by
//! their own ops) through every kind of scene change.
#![cfg(feature = "host")]
#![feature(allocator_api)]

use axnn::{CoverageFormat, PlacedGlyph, RASTER_ROW_WORDS};
use axnn_cpu::{CoverageAtlasRef, GlyphTableRef, XrgbSurfaceRef, XrgbTargetRef, paint_rows};
use axrd::host::{BINDINGS, SceneHost};
use axrd::{
    Anchor, BLIT_SOURCE_SLOT, Bind, Corners, LayoutParams, NodeFlags, Offset, Rect, Scene,
    SceneProgram, TileMask, TiledFrame, raster_ops, shadow_halo,
};
use std::alloc::Global;

const W: usize = 96;
const H: usize = 80;
const TILE: usize = 16;
const COLS: usize = W / TILE;
const ROOT: Rect = Rect {
    x: 0,
    y: 0,
    w: W as u16,
    h: H as u16,
};
const BG: u32 = 0xFF10_1014;
const N: usize = 12;
const SENTINEL: u32 = 0xDEAD_BEEF;

fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
    LayoutParams {
        anchor: Anchor::TOP_LEFT,
        offset: Offset { x, y },
        w,
        h,
    }
}

/// A 64×48 gradient surface for slot 0, opaque, plus a 20×20 translucent
/// sprite for slot 5.
fn surface_words(id: u16) -> Option<(Vec<u32>, u32, u32)> {
    match id {
        0 => Some((
            (0..64 * 48)
                .map(|i: u32| 0xFF00_0000 | ((i % 64) * 4) << 16 | ((i / 64) * 5) << 8 | 0x33)
                .collect(),
            64,
            48,
        )),
        5 => Some((
            (0..20 * 20)
                .map(|i: u32| {
                    let a = ((i % 20) * 12).min(255);
                    // Premultiplied red at alpha `a`.
                    (a << 24) | ((a * 200 / 255) << 16)
                })
                .collect(),
            20,
            20,
        )),
        _ => None,
    }
}

/// A 32×16 U8 atlas: a soft box glyph at (0, 0) 8×12 and a solid one at
/// (12, 0) 6×10.
fn atlas_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; 32 * 16];
    for y in 0..12 {
        for x in 0..8 {
            bytes[y * 32 + x] = ((x + y) * 20).min(255) as u8;
        }
    }
    for y in 0..10 {
        for x in 12..18 {
            bytes[y * 32 + x] = 255;
        }
    }
    bytes
}

const GLYPHS: [PlacedGlyph; 3] = [
    PlacedGlyph {
        ax: 0,
        ay: 0,
        aw: 8,
        ah: 12,
        dx: 2,
        dy: 1,
        subpixel_x: 0,
        subpixel_y: 0,
    },
    PlacedGlyph {
        ax: 12,
        ay: 0,
        aw: 6,
        ah: 10,
        dx: 11,
        dy: 3,
        subpixel_x: 0,
        subpixel_y: 0,
    },
    PlacedGlyph {
        ax: 0,
        ay: 0,
        aw: 8,
        ah: 12,
        dx: -3,
        dy: -4,
        subpixel_x: 0,
        subpixel_y: 0,
    },
];

/// Every bind kind, overlapping, straddling 16-pixel tile edges, with a
/// rounded aperture, a translucent overlay and a rounded clip.
fn scene() -> Scene<N> {
    let mut s = Scene::<N>::new();
    let root = s
        .add_root(
            lp(0, 0, W as u16, H as u16),
            Bind::Fill { color: 0x10_1014 },
        )
        .unwrap();
    let halo = shadow_halo(6);
    s.add_child(
        root,
        lp(
            30 - i32::from(halo),
            26 - i32::from(halo),
            50 + halo * 2,
            40 + halo * 2,
        ),
        Bind::Shadow {
            color: 0x00_0000,
            radii: Corners::uniform(8),
            falloff: 6,
        },
    )
    .unwrap();
    s.add_child(root, lp(3, 2, 80, 60), Bind::Surface { id: 0 })
        .unwrap();
    let win = s
        .add_node(
            Some(root),
            lp(40, 30, 48, 40),
            Bind::Surface { id: 0 },
            255,
            NodeFlags::CLIP_ROUNDED,
        )
        .unwrap();
    s.set_radii(win, Corners::uniform(9)).unwrap();
    s.add_child(
        root,
        lp(8, 8, 48, 48),
        Bind::RoundedFill {
            color: 0xC0_5030,
            radii: Corners::uniform(12),
        },
    )
    .unwrap();
    s.add_child(
        root,
        lp(8, 8, 48, 48),
        Bind::Border {
            color: 0xFF_FFFF,
            width: 2,
            radii: Corners::uniform(12),
        },
    )
    .unwrap();
    let clipped = s
        .add_node(
            Some(root),
            lp(20, 50, 40, 26),
            Bind::Fill { color: 0x20_A040 },
            255,
            NodeFlags::CLIP_ROUNDED,
        )
        .unwrap();
    s.set_radii(clipped, Corners::uniform(7)).unwrap();
    s.add_child_with_opacity(root, lp(0, 0, 96, 40), Bind::Fill { color: 0xFF_FFFF }, 96)
        .unwrap();
    s.add_child_with_opacity(root, lp(60, 55, 20, 20), Bind::Surface { id: 5 }, 254)
        .unwrap();
    s.add_child(
        root,
        lp(14, 60, 40, 16),
        Bind::TextRun {
            glyph_start: 0,
            glyph_len: 3,
            color: 0x10_20_F0,
        },
    )
    .unwrap();
    s.add_child(root, lp(0, 70, 10, 10), Bind::Surface { id: 7 })
        .unwrap();
    s
}

fn bindings() -> axrd::ProgramBindings {
    BINDINGS
}

type Host = SceneHost<Global, N, TILE>;

fn host(scene: &mut Scene<N>) -> Host {
    Host::new_in(scene, ROOT, Global).unwrap()
}

/// Paint one frame of `scene` into `fb` through the host, the fixture's
/// surfaces, atlas and glyphs bound.
fn paint(host: &mut Host, scene: &mut Scene<N>, fb: &mut [u32]) -> TileMask {
    let sources: Vec<(u16, Vec<u32>, u32, u32)> = [0u16, 5]
        .iter()
        .map(|&id| {
            let (words, w, h) = surface_words(id).unwrap();
            (id, words, w, h)
        })
        .collect();
    let atlas_bytes = atlas_bytes();
    let source = |slot: u32| {
        sources
            .iter()
            .find(|(id, ..)| u32::from(*id) == slot)
            .map(|(_, words, w, h)| XrgbSurfaceRef::new(words, *w, *h, *w as usize).unwrap())
    };
    let atlas = CoverageAtlasRef::new(&atlas_bytes, 32, 16, CoverageFormat::U8).unwrap();
    host.paint(scene, fb, &source, Some(atlas), &GLYPHS)
        .unwrap()
}

/// Paint one frame of `scene` into `fb` with no program: read this
/// frame's op table, lower every band against it and paint each band's
/// rows by their own ops straight into `fb`, the fixture's surfaces,
/// atlas and glyphs bound.
fn paint_dynamic(scene: &mut Scene<N>, fb: &mut [u32]) -> TileMask {
    let sources: Vec<(u16, Vec<u32>, u32, u32)> = [0u16, 5]
        .iter()
        .map(|&id| {
            let (words, w, h) = surface_words(id).unwrap();
            (id, words, w, h)
        })
        .collect();
    let atlas_bytes = atlas_bytes();
    let source = |slot: u32| {
        sources
            .iter()
            .find(|(id, ..)| u32::from(*id) == slot)
            .map(|(_, words, w, h)| XrgbSurfaceRef::new(words, *w, *h, *w as usize).unwrap())
    };
    let extent = |id: u16| source(u32::from(id)).map(|s| (s.width(), s.height()));
    let atlas = CoverageAtlasRef::new(&atlas_bytes, 32, 16, CoverageFormat::U8).unwrap();
    scene.resolve_layout_host(ROOT);
    let frame = TiledFrame::<N, TILE, TILE>::prepare(scene, (W, H)).unwrap();
    let mut table = [None; N];
    let ops = raster_ops(scene, &mut table).unwrap();
    let mut rows = vec![0i32; ops.len() * RASTER_ROW_WORDS];
    let mut dirty = vec![0u8; frame.columns()];
    for band in 0..frame.bands() {
        let painted = frame
            .lower_band_dynamic(ops, band, &extent, &mut rows, &mut dirty)
            .unwrap();
        if painted == [0; 2] {
            continue;
        }
        let (y, h) = (band * TILE, TILE.min(H - band * TILE));
        let target = XrgbTargetRef::new(
            &mut fb[y * W..(y + h) * W],
            (0, y as i32),
            W as u32,
            h as u32,
            W,
        )
        .unwrap();
        paint_rows(
            ops,
            &rows,
            TILE as u32,
            &dirty,
            target,
            &source,
            Some(atlas),
            Some(GlyphTableRef::new(&GLYPHS)),
        )
        .unwrap();
    }
    let mask = TileMask {
        bits: frame.dirty().bits,
    };
    drop(frame);
    scene.commit_frame();
    mask
}

fn at(fb: &[u32], x: usize, y: usize) -> u32 {
    fb[y * W + x]
}

/// Surface 0's word at surface-local `(x, y)`.
fn gradient(x: u32, y: u32) -> u32 {
    let i = y * 64 + x;
    0xFF00_0000 | ((i % 64) * 4) << 16 | ((i / 64) * 5) << 8 | 0x33
}

#[test]
fn the_first_frame_paints_every_tile_from_the_ground() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    let mask = paint(&mut host, &mut scene, &mut fb);
    assert_eq!(mask.count(), COLS * H.div_ceil(TILE), "every tile dirty");
    assert!(!fb.contains(&SENTINEL), "every pixel painted");
    assert_eq!(at(&fb, 2, 45), BG, "the ground");
    // The rounded window shows surface 0 at its own origin (right of the
    // rounded fill that covers its left half).
    assert_eq!(at(&fb, 70, 45), gradient(30, 15));
    // Beside the window the big surface shows its own pixels.
    assert_eq!(at(&fb, 5, 45), gradient(2, 43));
    // The rounded fill lies over both.
    assert_eq!(at(&fb, 50, 45), 0xFFC0_5030);
    // A clean frame paints nothing.
    fb.fill(SENTINEL);
    let mask = paint(&mut host, &mut scene, &mut fb);
    assert_eq!(mask.count(), 0);
    assert!(fb.iter().all(|&p| p == SENTINEL));
}

#[test]
fn a_client_commit_repaints_only_its_tiles() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    paint(&mut host, &mut scene, &mut fb);
    let expected = fb.clone();
    fb.fill(SENTINEL);
    scene.add_damage(Rect {
        x: 20,
        y: 20,
        w: 4,
        h: 4,
    });
    let mask = paint(&mut host, &mut scene, &mut fb);
    assert_eq!(mask.count(), 1);
    for y in 0..H {
        for x in 0..W {
            let inside = (16..32).contains(&x) && (16..32).contains(&y);
            let want = if inside {
                expected[y * W + x]
            } else {
                SENTINEL
            };
            assert_eq!(at(&fb, x, y), want, "({x}, {y})");
        }
    }
}

#[test]
fn rounded_apertures_spare_their_corners() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    paint(&mut host, &mut scene, &mut fb);
    // The clipped fill (20, 50, 40 × 26, r 7): interior exact, corner spared.
    assert_eq!(at(&fb, 56, 55), 0xFF20_A040);
    assert_ne!(
        at(&fb, 20, 75),
        0xFF20_A040,
        "the corner pixel is not the fill's"
    );
    assert_eq!(at(&fb, 20, 75) >> 24, 0xFF, "…but painted");
    // The rounded window (40, 30, 48 × 40, r 9) does not sample its source
    // into its bottom-right corner.
    assert_ne!(at(&fb, 87, 69), gradient(47, 39));
    // The rounded fill (8, 8, 48 × 48, r 12) under the translucent white
    // overlay: the corner pixel is not the interior's.
    assert_ne!(at(&fb, 8, 8), at(&fb, 30, 30));
}

#[test]
fn translucent_nodes_blend_over_what_they_cover() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    paint(&mut host, &mut scene, &mut fb);
    // White at opacity 96 over the ground: each channel moves 96/255 of
    // the way to 255.
    let p = at(&fb, 2, 5);
    for shift in [16, 8, 0] {
        let ground = ((BG >> shift) & 0xFF) as i32;
        let want = ground + (255 - ground) * 96 / 255;
        let got = ((p >> shift) & 0xFF) as i32;
        assert!((got - want).abs() <= 2, "channel {shift}: {got} vs {want}");
    }
    // The sprite's transparent column leaves the window's pixel; its
    // opaque side does not.
    assert_eq!(at(&fb, 60, 55), gradient(20, 25));
    assert_ne!(at(&fb, 79, 55), gradient(39, 25));
}

#[test]
fn text_tints_atlas_coverage_with_the_run_colour() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    paint(&mut host, &mut scene, &mut fb);
    // Somewhere in the run (blue ink over the green clipped fill and the
    // ground) blue outweighs green.
    let tinted = (60..76)
        .flat_map(|y| (14..54).map(move |x| (x, y)))
        .any(|(x, y)| {
            let p = at(&fb, x, y);
            (p & 0xFF) > ((p >> 8) & 0xFF)
        });
    assert!(tinted, "no glyph pixel took the run colour");
}

/// A frame's blit and sample source is the slot [`BLIT_SOURCE_SLOT`]:
/// a sample scales it, a blit copies it.
#[test]
fn blit_and_sample_nodes_read_the_frame_source() {
    let checker: Vec<u32> = (0..16)
        .map(|i| {
            if ((i % 4) / 2 + (i / 4) / 2) % 2 == 0 {
                0xFFFF_0000
            } else {
                0xFF00_00FF
            }
        })
        .collect();
    let mut s = Scene::<4>::new();
    let root = s
        .add_root(lp(0, 0, 64, 48), Bind::Fill { color: 0x10_1014 })
        .unwrap();
    s.add_child(
        root,
        lp(8, 8, 32, 32),
        Bind::Sample {
            sx: 0,
            sy: 0,
            sw: 4,
            sh: 4,
        },
    )
    .unwrap();
    s.add_child(root, lp(50, 8, 4, 4), Bind::Blit { src_x: 0, src_y: 0 })
        .unwrap();
    let root_rect = Rect {
        x: 0,
        y: 0,
        w: 64,
        h: 48,
    };
    let mut host = SceneHost::<Global, 4, 16>::new_in(&mut s, root_rect, Global).unwrap();
    let mut fb = vec![SENTINEL; 64 * 48];
    let source = |slot: u32| {
        (slot == BLIT_SOURCE_SLOT).then(|| XrgbSurfaceRef::new(&checker, 4, 4, 4).unwrap())
    };
    host.paint(&mut s, &mut fb, &source, None, &[]).unwrap();
    let px = |x: usize, y: usize| fb[y * 64 + x];
    assert_eq!(px(8, 8), 0xFFFF_0000, "sample: first texel");
    assert_eq!(px(39, 39), 0xFFFF_0000, "sample: last texel");
    assert_eq!(px(39, 8), 0xFF00_00FF, "sample: top-right texel");
    assert_eq!(px(50, 8), 0xFFFF_0000, "blit: first texel");
    assert_eq!(px(53, 11), 0xFFFF_0000, "blit: last texel");
    assert_eq!(px(52, 8), 0xFF00_00FF);
    assert_eq!(px(4, 4), 0xFF10_1014, "the ground around them");
}

/// Changing a node's bind KIND drifts the scene from its program; the
/// host rebuilds and the frame lands.
#[test]
fn the_host_rebuilds_a_drifted_program() {
    let mut scene = scene();
    let mut host = host(&mut scene);
    let mut fb = vec![SENTINEL; W * H];
    paint(&mut host, &mut scene, &mut fb);
    let id = scene.iter_alive().last().unwrap();
    scene.set_bind(id, Bind::Fill { color: 0xFF_0000 }).unwrap();
    paint(&mut host, &mut scene, &mut fb);
    assert_eq!(at(&fb, 2, 72), 0xFFFF_0000, "the rebound node painted");
}

#[test]
fn a_drifted_scene_is_refused_before_any_row_is_written() {
    let mut s = scene();
    s.resolve_layout_host(ROOT);
    let program = SceneProgram::build_in(&s, bindings(), TILE as u32, COLS, Global).unwrap();
    // Rebinding a node to another kind is drift.
    let id = s.iter_alive().last().unwrap();
    s.set_bind(id, Bind::Fill { color: 0 }).unwrap();
    s.resolve_layout_host(ROOT);
    let frame = TiledFrame::<N, TILE, TILE>::prepare(&s, (W, H)).unwrap();
    let mut rows = vec![0i32; program.table_words()];
    let mut dirty = vec![0u8; COLS];
    assert_eq!(
        frame
            .lower_band(&program, 4, &|_| None, &mut rows, &mut dirty)
            .err(),
        Some(axrd::ProgramError::Drift)
    );
    assert_eq!(rows.len(), program.rows() * RASTER_ROW_WORDS);
    assert!(rows.iter().all(|&w| w == 0), "no row written");
}

/// Consecutive same-kind rows share one node: the program has one node
/// per run of `op(row)`, and its rows stay one per slot.
#[test]
fn consecutive_same_kind_rows_share_one_node() {
    let mut scene = scene();
    scene.resolve_layout_host(ROOT);
    let program = SceneProgram::build_in(&scene, bindings(), TILE as u32, COLS, Global).unwrap();
    let mut runs = 0;
    let mut previous = None;
    for row in 0..program.rows() {
        let op = program.op(row);
        if op.is_some() && op != previous {
            runs += 1;
        }
        previous = op;
    }
    let graph = program.graph().parse().unwrap();
    assert_eq!(graph.node_count(), runs);
    assert!(runs < program.rows(), "the fixture has runs to merge");
}

/// One change a test applies to a scene, with the group node it tracks.
type Step<'a> = dyn Fn(&mut Scene<N>, &mut Option<axrd::NodeId>) + 'a;

/// The dynamic path paints the program's pixels frame after frame while
/// the scene changes under it: a kind changed, a group made paintable and
/// back, a node added, raised and removed — the program path rebuilding on
/// each drift, the dynamic path only reading a new op table.
#[test]
fn the_dynamic_path_paints_what_the_program_paints_through_every_change() {
    let mut by_program = scene();
    let mut by_ops = scene();
    let mut host = host(&mut by_program);
    let mut fb_program = vec![SENTINEL; W * H];
    let mut fb_ops = vec![SENTINEL; W * H];
    let root = by_program.iter_alive().next().unwrap();
    let last = by_program.iter_alive().last().unwrap();
    let mut group = None;
    let steps: [&Step<'_>; 7] = [
        &|_, _| {},
        &|s, _| s.set_bind(last, Bind::Fill { color: 0xFF_0000 }).unwrap(),
        &|s, group| {
            *group = Some(
                s.add_child(root, lp(70, 4, 20, 12), Bind::Composite)
                    .unwrap(),
            );
        },
        &|s, group| {
            s.set_bind(group.unwrap(), Bind::Fill { color: 0x20_C0E0 })
                .unwrap();
        },
        &|s, _| s.raise_to_top(last).unwrap(),
        &|s, group| s.set_bind(group.unwrap(), Bind::Composite).unwrap(),
        &|s, group| s.remove(group.take().unwrap()).unwrap(),
    ];
    for (step, change) in steps.iter().enumerate() {
        let mut twin = group;
        change(&mut by_program, &mut group);
        change(&mut by_ops, &mut twin);
        assert_eq!(twin, group, "both scenes hand out the same ids");
        let mask_program = paint(&mut host, &mut by_program, &mut fb_program);
        let mask_ops = paint_dynamic(&mut by_ops, &mut fb_ops);
        assert_eq!(
            mask_ops.bits, mask_program.bits,
            "step {step}: the same tiles"
        );
        assert!(mask_ops.count() > 0, "step {step} repaints");
        assert!(fb_ops == fb_program, "step {step}: the same pixels");
        if step == 3 {
            assert_eq!(at(&fb_ops, 80, 10), 0xFF20_C0E0, "the group painted");
        }
    }
    assert!(!fb_ops.contains(&SENTINEL));
    // A clean frame paints nothing either way.
    fb_ops.fill(SENTINEL);
    assert_eq!(paint_dynamic(&mut by_ops, &mut fb_ops).count(), 0);
    assert!(fb_ops.iter().all(|&p| p == SENTINEL));
}
