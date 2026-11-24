//! Live window demo — renders the compositor output into a host window.
//!
//! Run with:
//!   cargo +nightly run -p axrd --example window --features window
//!
//! Controls:
//!   Mouse move/click  — hover/focus nodes
//!   Tab / Shift+Tab   — cycle focus
//!   Escape            — quit
//!
//! The scene animates continuously: button A fades in/out, button B
//! slides back and forth, and the sprite rotates through color tints.

#![feature(allocator_api)]

use axnn::CoverageFormat;
use axnn_cpu::{CoverageAtlasRef, XrgbSurfaceRef};
use axrd::host::SceneHost;
use axrd::{
    Anchor, BLIT_SOURCE_SLOT, Bind, InputEvent, Interaction, LayoutParams, NodeEvent, Offset,
    PlacedGlyph, Rect, Scene, anim,
};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use std::alloc::Global;

const WIDTH: usize = 256;
const HEIGHT: usize = 128;

// ── Synthetic glyph bitmaps (5×5 pixel font) ──────────────────

const GLYPH_H: [u32; 25] = [
    0xFFFFFFFF, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0,
    0xFFFFFFFF,
];

const GLYPH_I: [u32; 25] = [
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0,
    0xFFFFFFFF, 0, 0, 0, 0, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF,
];

/// The two glyphs side by side as a packed-alpha atlas (`5 × 10`), as
/// little-endian bytes.
fn build_atlas() -> Vec<u8> {
    let mut words = vec![0u32; 5 * 10];
    for row in 0..5 {
        for col in 0..5 {
            words[row * 10 + col] = GLYPH_H[row * 5 + col];
            words[row * 10 + col + 5] = GLYPH_I[row * 5 + col];
        }
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A 4 × 4 red/blue checker.
fn build_sprite() -> Vec<u32> {
    (0..16)
        .map(|i| {
            if ((i % 4) / 2 + (i / 4) / 2) % 2 == 0 {
                0xFFFF_0000
            } else {
                0xFF00_00FF
            }
        })
        .collect()
}

fn main() {
    // ── Assets ─────────────────────────────────────────────────
    let atlas_bytes = build_atlas();
    let sprite = build_sprite();
    let glyphs = [
        PlacedGlyph {
            ax: 0,
            ay: 0,
            aw: 5,
            ah: 5,
            dx: 0,
            dy: 0,
            subpixel_x: 0,
            subpixel_y: 0,
        },
        PlacedGlyph {
            ax: 5,
            ay: 0,
            aw: 5,
            ah: 5,
            dx: 6,
            dy: 0,
            subpixel_x: 0,
            subpixel_y: 0,
        },
    ];

    // ── Framebuffer ────────────────────────────────────────────
    let mut pixel_buf = vec![0u32; WIDTH * HEIGHT];

    // ── Scene ──────────────────────────────────────────────────
    let mut scene = Scene::<64>::new();

    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIDTH as u16,
                h: HEIGHT as u16,
            },
            Bind::Composite,
        )
        .unwrap();

    // Background.
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIDTH as u16,
                h: HEIGHT as u16,
            },
            Bind::Fill { color: 0x00303030 },
        )
        .unwrap();

    // Button A: red, 60×30 at (16, 16).
    let btn_a = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 16, y: 16 },
                w: 60,
                h: 30,
            },
            Bind::Fill { color: 0x00CC2222 },
        )
        .unwrap();

    // Button B: green, 60×30 at (96, 16).
    let btn_b = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 96, y: 16 },
                w: 60,
                h: 30,
            },
            Bind::Fill { color: 0x0022CC22 },
        )
        .unwrap();

    // Text label "HI" at (16, 80).
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 16, y: 80 },
                w: 12,
                h: 5,
            },
            Bind::TextRun {
                glyph_start: 0,
                glyph_len: 2,
                color: 0x00FFFFFF,
            },
        )
        .unwrap();

    // Scaled sprite: 4×4 → 32×32 at (190, 16).
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 190, y: 16 },
                w: 32,
                h: 32,
            },
            Bind::Sample {
                sx: 0,
                sy: 0,
                sw: 4,
                sh: 4,
            },
        )
        .unwrap();

    // Focus indicator — a small yellow rect that follows focus.
    let focus_ind = scene
        .add_child_with_opacity(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 0, y: 0 },
                w: 4,
                h: 4,
            },
            Bind::Fill { color: 0x00FFFF00 },
            0, // invisible until something is focused
        )
        .unwrap();

    let root_rect = Rect {
        x: 0,
        y: 0,
        w: WIDTH as u16,
        h: HEIGHT as u16,
    };

    // ── Window ─────────────────────────────────────────────────
    let mut window = Window::new(
        "axrd compositor demo",
        WIDTH,
        HEIGHT,
        WindowOptions {
            scale: minifb::Scale::X4,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");

    // Cap at ~60 fps.
    window.set_target_fps(60);

    // ── Interaction state ──────────────────────────────────────
    let mut inter = Interaction::new();
    let mut node_events = [NodeEvent::NONE; 16];

    // ── Animation state ────────────────────────────────────────
    let mut frame: u32 = 0;
    let fade_period: u32 = 120; // frames per fade cycle
    let slide_period: u32 = 180;
    let btn_b_x0 = 96i32;
    let btn_b_x1 = 160i32;

    // ── Main loop ──────────────────────────────────────────────
    // The scene's program, built once over its node population.
    let mut host = SceneHost::<Global, 64, 64>::new_in(&mut scene, root_rect, Global).unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        frame += 1;

        // ── Input ──────────────────────────────────────────────
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            let x = mx as i32;
            let y = my as i32;

            inter.feed(&scene, InputEvent::PointerMove { x, y }, &mut node_events);

            if window.get_mouse_down(MouseButton::Left) {
                inter.feed(
                    &scene,
                    InputEvent::PointerDown { x, y, button: 0 },
                    &mut node_events,
                );
            }
        }

        if window.is_key_pressed(Key::Tab, minifb::KeyRepeat::No) {
            let evt = if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) {
                InputEvent::FocusPrev
            } else {
                InputEvent::FocusNext
            };
            inter.feed(&scene, evt, &mut node_events);
        }

        // ── Update focus indicator ─────────────────────────────
        if let Some(r) = inter.focused().and_then(|focused| scene.rect_of(focused)) {
            scene
                .set_layout(
                    focus_ind,
                    LayoutParams {
                        anchor: Anchor::TOP_LEFT,
                        offset: Offset {
                            x: r.x - 2,
                            y: r.y - 2,
                        },
                        w: r.w + 4,
                        h: 2,
                    },
                )
                .unwrap();
            scene.set_opacity(focus_ind, 255).unwrap();
        } else {
            scene.set_opacity(focus_ind, 0).unwrap();
        }

        // ── Animate button A opacity ───────────────────────────
        let fade_t = (frame % fade_period) as f32 / fade_period as f32;
        // Ping-pong: 0→1→0 over one cycle.
        let fade_v = if fade_t < 0.5 {
            anim::ease_in_out_quad(fade_t * 2.0)
        } else {
            anim::ease_in_out_quad((1.0 - fade_t) * 2.0)
        };
        scene
            .set_opacity(btn_a, anim::lerp_u8(40, 255, fade_v))
            .unwrap();

        // ── Animate button B position ──────────────────────────
        let slide_t = (frame % slide_period) as f32 / slide_period as f32;
        let slide_v = if slide_t < 0.5 {
            anim::ease_in_out_cubic(slide_t * 2.0)
        } else {
            anim::ease_in_out_cubic((1.0 - slide_t) * 2.0)
        };
        let bx = anim::lerp_i32(btn_b_x0, btn_b_x1, slide_v);
        scene
            .set_layout(
                btn_b,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x: bx, y: 16 },
                    w: 60,
                    h: 30,
                },
            )
            .unwrap();

        // ── Animate button A color ─────────────────────────────
        let color_t = ((frame % 360) as f32 / 360.0 * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        let btn_a_color = anim::lerp_color(0x00CC2222, 0x002222CC, color_t);
        scene
            .set_bind(btn_a, Bind::Fill { color: btn_a_color })
            .unwrap();

        // ── Render ─────────────────────────────────────────────
        let source = |slot: u32| {
            (slot == BLIT_SOURCE_SLOT).then(|| XrgbSurfaceRef::new(&sprite, 4, 4, 4).unwrap())
        };
        let atlas =
            CoverageAtlasRef::new(&atlas_bytes, 10, 5, CoverageFormat::PackedAlpha).unwrap();
        host.paint(&mut scene, &mut pixel_buf, &source, Some(atlas), &glyphs)
            .unwrap();

        // ── Display ────────────────────────────────────────────
        window
            .update_with_buffer(&pixel_buf, WIDTH, HEIGHT)
            .unwrap();
    }
}
