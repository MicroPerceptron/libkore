//! Interactive terminal emulator widget — 80×25 character grid with an
//! 8×16 VGA bitmap font rendered via the compositor's TextRun pipeline.
//!
//! Run with:
//!   cargo +nightly run -p axrd --example shell --features window
//!
//! Controls:
//!   Type         — characters echo into the terminal grid
//!   Enter        — submit line (echoed back as a response)
//!   Backspace    — delete character behind cursor
//!   Escape       — quit

#![feature(allocator_api)]

use axnn::CoverageFormat;
use axnn_cpu::{CoverageAtlasRef, XrgbSurfaceRef};
use axrd::host::SceneHost;
use axrd::{
    Anchor, Bind, LayoutParams, Offset, PlacedGlyph, Rect, Scene,
    font8x16::{self, ATLAS_H, ATLAS_W, GLYPH_H, GLYPH_W},
    term::TermGrid,
};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::alloc::Global;

// ── Constants ─────────────────────────────────────────────────────
const TERM_COLS: usize = 80;
const TERM_ROWS: usize = 25;
const WIN_W: usize = TERM_COLS * GLYPH_W as usize; // 640
const WIN_H: usize = TERM_ROWS * GLYPH_H as usize; // 400
const BG_COLOR: u32 = 0x001A1A2E; // dark navy
const FG_COLOR: u32 = 0x00FFFFFF; // white text (unused by atlas, but passed to TextRun)
const CURSOR_COLOR: u32 = 0x0000FF88; // green cursor

// ── Helpers ───────────────────────────────────────────────────────

/// The 8×16 font atlas as premultiplied-alpha words, little-endian bytes
/// (its alpha byte is the coverage the program tints).
fn atlas_bytes() -> Vec<u8> {
    font8x16::build_atlas_pixels()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// Build the PlacedGlyph array from the current TermGrid state.
/// Only emits glyphs for non-empty cells. Returns the count of glyphs placed.
fn build_glyphs(term: &TermGrid, glyphs: &mut [PlacedGlyph]) -> usize {
    let mut n = 0;
    for row in 0..term.rows() {
        for col in 0..term.cols() {
            let ch = term.cell(col, row);
            if ch == 0 || ch == b' ' {
                continue;
            }
            if n >= glyphs.len() {
                return n;
            }
            glyphs[n] = PlacedGlyph {
                ax: font8x16::atlas_x(ch),
                ay: 0,
                aw: GLYPH_W,
                ah: GLYPH_H,
                dx: (col as u16 * GLYPH_W) as i16,
                dy: (row as u16 * GLYPH_H) as i16,
                subpixel_x: 0,
                subpixel_y: 0,
            };
            n += 1;
        }
    }
    n
}

/// Map a minifb Key to an ASCII byte, accounting for shift state.
fn key_to_ascii(key: Key, shift: bool) -> Option<u8> {
    match key {
        Key::A => Some(if shift { b'A' } else { b'a' }),
        Key::B => Some(if shift { b'B' } else { b'b' }),
        Key::C => Some(if shift { b'C' } else { b'c' }),
        Key::D => Some(if shift { b'D' } else { b'd' }),
        Key::E => Some(if shift { b'E' } else { b'e' }),
        Key::F => Some(if shift { b'F' } else { b'f' }),
        Key::G => Some(if shift { b'G' } else { b'g' }),
        Key::H => Some(if shift { b'H' } else { b'h' }),
        Key::I => Some(if shift { b'I' } else { b'i' }),
        Key::J => Some(if shift { b'J' } else { b'j' }),
        Key::K => Some(if shift { b'K' } else { b'k' }),
        Key::L => Some(if shift { b'L' } else { b'l' }),
        Key::M => Some(if shift { b'M' } else { b'm' }),
        Key::N => Some(if shift { b'N' } else { b'n' }),
        Key::O => Some(if shift { b'O' } else { b'o' }),
        Key::P => Some(if shift { b'P' } else { b'p' }),
        Key::Q => Some(if shift { b'Q' } else { b'q' }),
        Key::R => Some(if shift { b'R' } else { b'r' }),
        Key::S => Some(if shift { b'S' } else { b's' }),
        Key::T => Some(if shift { b'T' } else { b't' }),
        Key::U => Some(if shift { b'U' } else { b'u' }),
        Key::V => Some(if shift { b'V' } else { b'v' }),
        Key::W => Some(if shift { b'W' } else { b'w' }),
        Key::X => Some(if shift { b'X' } else { b'x' }),
        Key::Y => Some(if shift { b'Y' } else { b'y' }),
        Key::Z => Some(if shift { b'Z' } else { b'z' }),
        Key::Key0 => Some(if shift { b')' } else { b'0' }),
        Key::Key1 => Some(if shift { b'!' } else { b'1' }),
        Key::Key2 => Some(if shift { b'@' } else { b'2' }),
        Key::Key3 => Some(if shift { b'#' } else { b'3' }),
        Key::Key4 => Some(if shift { b'$' } else { b'4' }),
        Key::Key5 => Some(if shift { b'%' } else { b'5' }),
        Key::Key6 => Some(if shift { b'^' } else { b'6' }),
        Key::Key7 => Some(if shift { b'&' } else { b'7' }),
        Key::Key8 => Some(if shift { b'*' } else { b'8' }),
        Key::Key9 => Some(if shift { b'(' } else { b'9' }),
        Key::Space => Some(b' '),
        Key::Period => Some(if shift { b'>' } else { b'.' }),
        Key::Comma => Some(if shift { b'<' } else { b',' }),
        Key::Slash => Some(if shift { b'?' } else { b'/' }),
        Key::Backslash => Some(if shift { b'|' } else { b'\\' }),
        Key::Semicolon => Some(if shift { b':' } else { b';' }),
        Key::Apostrophe => Some(if shift { b'"' } else { b'\'' }),
        Key::LeftBracket => Some(if shift { b'{' } else { b'[' }),
        Key::RightBracket => Some(if shift { b'}' } else { b']' }),
        Key::Minus => Some(if shift { b'_' } else { b'-' }),
        Key::Equal => Some(if shift { b'+' } else { b'=' }),
        Key::Backquote => Some(if shift { b'~' } else { b'`' }),
        _ => None,
    }
}

/// Simple command handler: echoes unknown commands, responds to a few builtins.
fn handle_command(line: &[u8], term: &mut TermGrid) {
    let trimmed = line
        .iter()
        .copied()
        .skip_while(|&b| b == b' ')
        .collect::<Vec<u8>>();

    if trimmed.is_empty() {
        // Empty line, just show a new prompt.
    } else if trimmed == b"help" {
        term.put_str(b"Available commands: help, echo, clear, hello\n");
    } else if trimmed == b"clear" {
        term.clear();
    } else if trimmed == b"hello" {
        term.put_str(b"Hello from axrd compositor!\n");
    } else if trimmed.starts_with(b"echo ") {
        term.put_str(&trimmed[5..]);
        term.newline();
    } else {
        term.put_str(b"unknown: ");
        term.put_str(&trimmed);
        term.newline();
    }
}

fn main() {
    eprintln!(
        "shell: starting terminal emulator ({}x{})...",
        TERM_COLS, TERM_ROWS
    );

    // ── Font atlas ────────────────────────────────────────────────
    let atlas_bytes = atlas_bytes();
    eprintln!("shell: font atlas built ({ATLAS_W}x{ATLAS_H})");

    // ── Background surface (slot 0) ───────────────────────────────
    let bg_color_argb = BG_COLOR | 0xFF00_0000;
    let bg: Vec<u32> = vec![bg_color_argb; WIN_W * WIN_H];

    // ── Framebuffer ───────────────────────────────────────────────
    let mut pixel_buf = vec![0u32; WIN_W * WIN_H];

    // ── Terminal state ────────────────────────────────────────────
    let mut term = TermGrid::new(TERM_COLS, TERM_ROWS);
    term.put_str(b"axrd shell v0.1\n");
    term.put_str(b"Type 'help' for commands.\n");
    term.put_str(b"$ ");

    // Line input buffer (for command parsing on Enter).
    let mut line_buf: Vec<u8> = Vec::new();
    // Cursor position within line_buf (0 = before first char).
    let mut line_pos: usize = 0;

    // Glyph buffer: worst case = all cells filled.
    let max_glyphs = TERM_COLS * TERM_ROWS;
    let mut glyphs = vec![
        PlacedGlyph {
            ax: 0,
            ay: 0,
            aw: 0,
            ah: 0,
            dx: 0,
            dy: 0,
            subpixel_x: 0,
            subpixel_y: 0,
        };
        max_glyphs
    ];

    // ── Cursor surface (slot 1) ───────────────────────────────────
    let cursor_argb = CURSOR_COLOR | 0xFF00_0000;
    let cursor: Vec<u32> = vec![cursor_argb; GLYPH_W as usize * GLYPH_H as usize];
    let source = |slot: u32| match slot {
        0 => XrgbSurfaceRef::new(&bg, WIN_W as u32, WIN_H as u32, WIN_W).ok(),
        1 => XrgbSurfaceRef::new(
            &cursor,
            u32::from(GLYPH_W),
            u32::from(GLYPH_H),
            usize::from(GLYPH_W),
        )
        .ok(),
        _ => None,
    };

    // ── Scene ─────────────────────────────────────────────────────
    let mut scene = Scene::<8>::new();

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

    // Background — pre-allocated surface.
    scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIN_W as u16,
                h: WIN_H as u16,
            },
            Bind::Surface { id: 0 },
        )
        .unwrap();

    // Text layer — covers full terminal area.
    let text_node = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset::ZERO,
                w: WIN_W as u16,
                h: WIN_H as u16,
            },
            Bind::TextRun {
                glyph_start: 0,
                glyph_len: 0,
                color: FG_COLOR,
            },
        )
        .unwrap();

    // Cursor — pre-allocated surface (id=1), not Fill.
    let cursor_node = scene
        .add_child(
            root,
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: 0, y: 0 },
                w: GLYPH_W,
                h: GLYPH_H,
            },
            Bind::Surface { id: 1 },
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
        "axrd — shell",
        WIN_W,
        WIN_H,
        WindowOptions {
            scale: minifb::Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    let mut frame_count: u32 = 0;

    eprintln!("shell: running. Type commands, Enter to submit, Escape to quit.");

    // ── Main loop ─────────────────────────────────────────────────
    // The scene's program, built once over its node population.
    let mut host = SceneHost::<Global, 8, 64>::new_in(&mut scene, root_rect, Global).unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        frame_count = frame_count.wrapping_add(1);

        // ── Keyboard input ────────────────────────────────────────
        let shift = window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift);

        for key in window.get_keys_pressed(KeyRepeat::Yes) {
            match key {
                Key::Enter => {
                    term.newline();
                    handle_command(&line_buf, &mut term);
                    line_buf.clear();
                    line_pos = 0;
                    term.put_str(b"$ ");
                }
                Key::Backspace => {
                    if line_pos > 0 {
                        line_pos -= 1;
                        line_buf.remove(line_pos);
                        term.cursor_left();
                        term.delete_at_cursor();
                    }
                }
                Key::Left => {
                    if line_pos > 0 {
                        line_pos -= 1;
                        term.cursor_left();
                    }
                }
                Key::Right => {
                    if line_pos < line_buf.len() {
                        line_pos += 1;
                        term.cursor_right();
                    }
                }
                _ => {
                    if let Some(ch) = key_to_ascii(key, shift) {
                        if line_pos == line_buf.len() {
                            // Append at end.
                            term.put_char(ch);
                            line_buf.push(ch);
                        } else {
                            // Insert mid-line.
                            line_buf.insert(line_pos, ch);
                            term.insert_char(ch);
                        }
                        line_pos += 1;
                    }
                }
            }
        }

        // ── Update glyphs from terminal grid ──────────────────────
        let glyph_count = build_glyphs(&term, &mut glyphs);

        // Update TextRun bind with new glyph count.
        scene
            .set_bind(
                text_node,
                Bind::TextRun {
                    glyph_start: 0,
                    glyph_len: glyph_count as u16,
                    color: FG_COLOR,
                },
            )
            .unwrap();

        // Update cursor position.
        let cursor_x = term.cx as i32 * GLYPH_W as i32;
        let cursor_y = term.cy as i32 * GLYPH_H as i32;
        scene
            .set_layout(
                cursor_node,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset {
                        x: cursor_x,
                        y: cursor_y,
                    },
                    w: GLYPH_W,
                    h: GLYPH_H,
                },
            )
            .unwrap();

        // Blink cursor: visible 60% of the time (36 frames on, 24 off at 60fps).
        let blink = (frame_count % 60) < 36;
        scene
            .set_opacity(cursor_node, if blink { 180 } else { 0 })
            .unwrap();

        // ── Render ────────────────────────────────────────────────
        let atlas = CoverageAtlasRef::new(
            &atlas_bytes,
            ATLAS_W as u32,
            ATLAS_H as u32,
            CoverageFormat::PackedAlpha,
        )
        .unwrap();
        host.paint(
            &mut scene,
            &mut pixel_buf,
            &source,
            Some(atlas),
            &glyphs[..glyph_count],
        )
        .unwrap();

        // ── Display ───────────────────────────────────────────────
        window.update_with_buffer(&pixel_buf, WIN_W, WIN_H).unwrap();
    }
}
