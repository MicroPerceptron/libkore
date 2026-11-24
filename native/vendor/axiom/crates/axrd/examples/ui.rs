//! Interactive neobrutalist UI demo — loads pre-baked bspl widget sprites
//! and wires mouse/keyboard interaction through the compositor.
//!
//! Run with:
//!   cargo +nightly run -p axrd --example ui --features ui
//!
//! Controls:
//!   Mouse click     — toggle widgets (buttons, checkboxes, radio, toggle)
//!   Tab / Shift+Tab — cycle keyboard focus
//!   Space / Enter   — activate focused widget
//!   Escape          — quit
//!
//! Each widget is a pair of bspl assets (normal + active state). They
//! are decoded into tensors at startup and bound via `Bind::Surface`.
//! Interaction swaps the surface ID between states — zero runtime
//! rasterization needed.

#![feature(allocator_api)]

use axnn::Backend;
use axnn_cpu::{CpuBackend, XrgbSurfaceRef};
use axrd::host::SceneHost;
use axrd::{
    Anchor, Bind, InputEvent, Interaction, LayoutParams, NodeEvent, NodeId, Offset, Rect, Scene,
    splash,
};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use std::alloc::Global;

// ── Constants ─────────────────────────────────────────────────────
const WIN_W: usize = 800;
const WIN_H: usize = 480;
const ARENA_SIZE: usize = 16 * 1024 * 1024; // 16 MB — the bspl decoder's arena at load

type Be = CpuBackend<ARENA_SIZE>;

// ── Widget definition ─────────────────────────────────────────────
struct WidgetDef {
    label: &'static str,
    /// Position in the demo window.
    dx: i32,
    dy: i32,
    /// bspl file for "normal" state.
    normal_bspl: &'static str,
    /// bspl file for "active" state.
    active_bspl: &'static str,
    /// Whether this is a toggle (click toggles) or momentary (reverts on release).
    toggle: bool,
}

const WIDGETS: &[WidgetDef] = &[
    WidgetDef {
        label: "Primary",
        dx: 30,
        dy: 20,
        normal_bspl: "primary.bspl",
        active_bspl: "pressed.bspl",
        toggle: false,
    },
    WidgetDef {
        label: "Secondary",
        dx: 220,
        dy: 20,
        normal_bspl: "secondary.bspl",
        active_bspl: "pressed.bspl",
        toggle: false,
    },
    WidgetDef {
        label: "Danger",
        dx: 410,
        dy: 20,
        normal_bspl: "danger.bspl",
        active_bspl: "pressed.bspl",
        toggle: false,
    },
    WidgetDef {
        label: "Checkbox",
        dx: 30,
        dy: 110,
        normal_bspl: "checkbox-unchecked.bspl",
        active_bspl: "checkbox-checked.bspl",
        toggle: true,
    },
    WidgetDef {
        label: "Radio A",
        dx: 200,
        dy: 110,
        normal_bspl: "radio-option-a.bspl",
        active_bspl: "radio-option-b.bspl",
        toggle: true,
    },
    WidgetDef {
        label: "Toggle",
        dx: 360,
        dy: 110,
        normal_bspl: "toggle-off.bspl",
        active_bspl: "toggle-on.bspl",
        toggle: true,
    },
];

struct WidgetSurfaces {
    normal_id: usize,
    active_id: usize,
    normal_w: u16,
    normal_h: u16,
    active_w: u16,
    active_h: u16,
}

// ── Helpers ───────────────────────────────────────────────────────

/// Resolve a bspl file path relative to the workspace assets/neobrutalist/bspl/ dir.
fn bspl_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/neobrutalist/bspl")
        .join(name)
}

/// Load a bspl file and decode its first frame into XRGB words.
/// Returns (words, width, height).
fn load_widget_bspl(path: &std::path::Path, be: &Be) -> (Vec<u32>, u16, u16) {
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let anim = bspl::parse(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {:?}", path.display(), e));
    let header = anim.header();
    let w = header.img_w as u16;
    let h = header.img_h as u16;

    // Decode first frame. For XRGB dense assets, decode_first_frame
    // does a zero-copy upload.
    let tensor = splash::decode_first_frame(&anim, be)
        .unwrap_or_else(|_| panic!("Failed to decode {}", path.display()));
    let mut bytes = vec![0u8; tensor.desc().storage_bytes()];
    be.download(&tensor, &mut bytes).unwrap();
    let words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (words, w, h)
}

fn main() {
    eprintln!("ui: loading widget bspl assets...");

    // ── Backend ───────────────────────────────────────────────────
    let be = Be::new_boxed();

    // ── Load widget surfaces ──────────────────────────────────────
    // Each widget has two surfaces (normal + active). We store all
    // tensors in a flat Vec and track indices per widget.
    let mut all_surfaces: Vec<(Vec<u32>, u16, u16)> = Vec::new();
    let mut widget_info: Vec<WidgetSurfaces> = Vec::new();

    for wd in WIDGETS {
        let (norm_words, nw, nh) = load_widget_bspl(&bspl_path(wd.normal_bspl), &be);
        let normal_id = all_surfaces.len();
        all_surfaces.push((norm_words, nw, nh));

        let (act_words, aw, ah) = load_widget_bspl(&bspl_path(wd.active_bspl), &be);
        let active_id = all_surfaces.len();
        all_surfaces.push((act_words, aw, ah));

        widget_info.push(WidgetSurfaces {
            normal_id,
            active_id,
            normal_w: nw,
            normal_h: nh,
            active_w: aw,
            active_h: ah,
        });
        eprintln!(
            "  {} ({nw}x{nh} / {aw}x{ah}): normal=S{normal_id}, active=S{active_id}",
            wd.label
        );
    }

    // ── The background as a surface ───────────────────────────────
    let bg_color: u32 = 0xFFF5_F0E8; // neobrutalist warm beige
    let bg_id = all_surfaces.len();
    all_surfaces.push((vec![bg_color; WIN_W * WIN_H], WIN_W as u16, WIN_H as u16));
    let source = |slot: u32| {
        all_surfaces.get(slot as usize).map(|(words, w, h)| {
            XrgbSurfaceRef::new(words, u32::from(*w), u32::from(*h), usize::from(*w)).unwrap()
        })
    };

    eprintln!(
        "ui: {} surfaces loaded ({:.1} KB decode arena used)",
        all_surfaces.len(),
        be.used() as f64 / 1024.0
    );

    // ── Framebuffer ───────────────────────────────────────────────
    let mut pixel_buf = vec![0u32; WIN_W * WIN_H];

    // ── Scene ─────────────────────────────────────────────────────
    let mut scene = Scene::<64>::new();

    let root = scene
        .add_root(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIN_W as u16,
                h: WIN_H as u16,
            },
            Bind::Composite,
        )
        .unwrap();

    // Background — pre-allocated surface, no per-frame fill allocation.
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIN_W as u16,
                h: WIN_H as u16,
            },
            Bind::Surface { id: bg_id as u16 },
        )
        .unwrap();

    // ── Widget nodes (Surface binds) ──────────────────────────────
    let mut widget_nodes = Vec::new();
    let mut widget_active = Vec::new();

    for (i, wd) in WIDGETS.iter().enumerate() {
        let info = &widget_info[i];
        // Use max of normal/active dimensions so both states fit
        // without needing set_layout (which invalidates resolve_layout).
        let max_w = info.normal_w.max(info.active_w);
        let max_h = info.normal_h.max(info.active_h);
        let node_id = scene
            .add_child(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x: wd.dx, y: wd.dy },
                    w: max_w,
                    h: max_h,
                },
                Bind::Surface {
                    id: info.normal_id as u16,
                },
            )
            .unwrap();
        widget_nodes.push(node_id);
        widget_active.push(false);
    }

    // ── Decorative separator ──────────────────────────────────────
    // Use Composite (no paint) to avoid per-frame fill allocation.
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 30, y: 190 },
                w: 540,
                h: 2,
            },
            Bind::Composite, // purely structural, no per-frame alloc
        )
        .unwrap();

    let root_rect = Rect {
        x: 0,
        y: 0,
        w: WIN_W as u16,
        h: WIN_H as u16,
    };

    // ── Window ────────────────────────────────────────────────────
    let mut window = Window::new(
        "axrd — neobrutalist UI demo",
        WIN_W,
        WIN_H,
        WindowOptions {
            scale: minifb::Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    // ── Interaction state ─────────────────────────────────────────
    let mut inter = Interaction::new();
    let mut node_events = [NodeEvent::NONE; 16];
    let mut prev_mouse_down = false;

    eprintln!("ui: running. Click widgets, Tab to cycle focus, Space/Enter to activate.");

    // ── Main loop ─────────────────────────────────────────────────
    // The scene's program, built once over its node population.
    let mut host = SceneHost::<Global, 64, 64>::new_in(&mut scene, root_rect, Global).unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ── Input ─────────────────────────────────────────────────
        let mouse_down = window.get_mouse_down(MouseButton::Left);

        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            let x = mx as i32;
            let y = my as i32;

            inter.feed(&scene, InputEvent::PointerMove { x, y }, &mut node_events);

            if mouse_down && !prev_mouse_down {
                let n = inter.feed(
                    &scene,
                    InputEvent::PointerDown { x, y, button: 0 },
                    &mut node_events,
                );
                handle_press(
                    &node_events[..n],
                    &widget_nodes,
                    &widget_info,
                    &mut widget_active,
                    &mut scene,
                );
            }
            if !mouse_down && prev_mouse_down {
                let n = inter.feed(
                    &scene,
                    InputEvent::PointerUp { x, y, button: 0 },
                    &mut node_events,
                );
                handle_release(
                    &node_events[..n],
                    &widget_nodes,
                    &widget_info,
                    &mut widget_active,
                    &mut scene,
                );
            }
        }
        prev_mouse_down = mouse_down;

        // Tab cycling.
        if window.is_key_pressed(Key::Tab, minifb::KeyRepeat::No) {
            let evt = if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) {
                InputEvent::FocusPrev
            } else {
                InputEvent::FocusNext
            };
            inter.feed(&scene, evt, &mut node_events);
        }

        // Space/Enter activates focused widget.
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No)
            || window.is_key_pressed(Key::Enter, minifb::KeyRepeat::No)
        {
            if let Some(focused) = inter.focused() {
                activate_widget(
                    focused,
                    &widget_nodes,
                    &widget_info,
                    &mut widget_active,
                    &mut scene,
                );
            }
        }

        // ── Render ────────────────────────────────────────────────
        host.paint(&mut scene, &mut pixel_buf, &source, None, &[])
            .unwrap();

        // ── Display ───────────────────────────────────────────────
        window.update_with_buffer(&pixel_buf, WIN_W, WIN_H).unwrap();
    }
}

// ── Event handlers ────────────────────────────────────────────────

fn handle_press(
    events: &[NodeEvent],
    widget_nodes: &[NodeId],
    widget_info: &[WidgetSurfaces],
    widget_active: &mut [bool],
    scene: &mut Scene<64>,
) {
    for evt in events {
        if let NodeEvent::PointerDown { node, .. } = evt {
            activate_widget(*node, widget_nodes, widget_info, widget_active, scene);
        }
    }
}

fn handle_release(
    events: &[NodeEvent],
    widget_nodes: &[NodeId],
    widget_info: &[WidgetSurfaces],
    widget_active: &mut [bool],
    scene: &mut Scene<64>,
) {
    for evt in events {
        if let NodeEvent::PointerUp { node, .. } = evt {
            if let Some(idx) = widget_nodes.iter().position(|&n| n == *node) {
                let wd = &WIDGETS[idx];
                if !wd.toggle {
                    widget_active[idx] = false;
                    let info = &widget_info[idx];
                    scene
                        .set_bind(
                            widget_nodes[idx],
                            Bind::Surface {
                                id: info.normal_id as u16,
                            },
                        )
                        .unwrap();
                    eprintln!("  [{}] released", wd.label);
                }
            }
        }
    }
}

fn activate_widget(
    node: NodeId,
    widget_nodes: &[NodeId],
    widget_info: &[WidgetSurfaces],
    widget_active: &mut [bool],
    scene: &mut Scene<64>,
) {
    if let Some(idx) = widget_nodes.iter().position(|&n| n == node) {
        let wd = &WIDGETS[idx];
        let info = &widget_info[idx];

        if wd.toggle {
            widget_active[idx] = !widget_active[idx];
        } else {
            widget_active[idx] = true;
        }

        let active = widget_active[idx];
        let surface_id = if active {
            info.active_id
        } else {
            info.normal_id
        };

        scene
            .set_bind(
                widget_nodes[idx],
                Bind::Surface {
                    id: surface_id as u16,
                },
            )
            .unwrap();
        eprintln!(
            "  [{}] → {}",
            wd.label,
            if active { "active" } else { "normal" }
        );
    }
}
