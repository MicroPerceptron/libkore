//! Boot splash player — loads a `.bspl` animation and plays it in a window.
//!
//! Run with:
//!   cargo +nightly run -p axrd --example splash --features splash
//!
//! Expects `assets/splash.bspl` relative to the workspace root.

use bspl::{AnimationFrameData, PixelFormat, decompress_sparse_lz4_payload, parse};
use minifb::{Key, Window, WindowOptions};

fn main() {
    // ── Load asset ─────────────────────────────────────────────────
    let bspl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/splash.bspl");
    let bspl_data = std::fs::read(&bspl_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", bspl_path.display()));

    let anim = parse(&bspl_data).expect("Failed to parse bspl asset");
    let header = anim.header();
    let img_w = header.img_w as usize;
    let img_h = header.img_h as usize;

    eprintln!(
        "splash: {}x{}, {} frames, {:?}, {:?}",
        img_w, img_h, header.frame_count, header.pixel_format, header.encoding
    );

    // ── Pixel buffer (persistent across frames) ────────────────────
    let mut pixels = vec![0u32; img_w * img_h];

    // ── LZ4 scratch buffer ─────────────────────────────────────────
    // Generous scratch for decompressed sparse payloads.
    let mut scratch = vec![0u8; img_w * img_h * 4];

    // ── Collect frames ─────────────────────────────────────────────
    // Pre-parse all frame metadata so we can loop the animation.
    // Frame data references into bspl_data (zero-copy).
    let frames: Vec<_> = anim
        .frames()
        .map(|r| r.expect("Failed to parse frame"))
        .collect();

    // ── Window ─────────────────────────────────────────────────────
    let scale = pick_scale(img_w, img_h);
    let mut window = Window::new(
        "axrd splash",
        img_w,
        img_h,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    // Target frame rate from bspl duration_us (first frame).
    let target_us = frames.first().map_or(33_333, |f| f.duration_us as u64);
    let target_fps = (1_000_000 + target_us / 2) / target_us; // round to nearest
    window.set_target_fps(target_fps as usize);

    // ── Decode first frame fully (dense path if v2, sparse otherwise) ──
    // For v4 sparse, the first frame typically has all tiles dirty.
    decode_frame(&header, &frames[0], &mut pixels, &mut scratch);

    window
        .update_with_buffer(&pixels, img_w, img_h)
        .expect("display failed");

    // ── Animation loop ─────────────────────────────────────────────
    let mut frame_idx = 1usize;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if frame_idx >= frames.len() {
            frame_idx = 0;
            // Reset pixel buffer for clean loop.
            pixels.fill(0);
        }

        decode_frame(&header, &frames[frame_idx], &mut pixels, &mut scratch);

        window
            .update_with_buffer(&pixels, img_w, img_h)
            .expect("display failed");

        frame_idx += 1;
    }
}

/// Decode one animation frame into the pixel buffer.
fn decode_frame(
    header: &bspl::Header,
    frame: &bspl::AnimationFrame<'_>,
    pixels: &mut [u32],
    scratch: &mut [u8],
) {
    match frame.data {
        AnimationFrameData::Dense { pixels: px } => {
            convert_to_xrgb(header.pixel_format, px, pixels);
        }
        AnimationFrameData::Sparse { payload } => {
            apply_sparse(header, frame.dirty_words_le, payload, pixels);
        }
        AnimationFrameData::SparseLz4 {
            compressed_payload,
            raw_payload_len,
        } => {
            let n = decompress_sparse_lz4_payload(
                *header,
                frame.dirty_words_le,
                compressed_payload,
                raw_payload_len,
                scratch,
            )
            .expect("LZ4 decompression failed");
            apply_sparse(header, frame.dirty_words_le, &scratch[..n], pixels);
        }
    }
}

/// Convert full-frame pixel data to XRGB u32 buffer.
fn convert_to_xrgb(fmt: PixelFormat, src: &[u8], dst: &mut [u32]) {
    let bpp = fmt.bytes_per_pixel();
    for (i, chunk) in src.chunks_exact(bpp).enumerate() {
        if i >= dst.len() {
            break;
        }
        dst[i] = pixel_to_xrgb(fmt, chunk);
    }
}

/// Apply sparse tile payload into pixel buffer.
fn apply_sparse(header: &bspl::Header, dirty_words_le: &[u8], payload: &[u8], dst: &mut [u32]) {
    let tiles_x = header.tiles_x() as usize;
    let tile_w = header.tile_w as usize;
    let tile_h = header.tile_h as usize;
    let img_w = header.img_w as usize;
    let img_h = header.img_h as usize;
    let bpp = header.pixel_format.bytes_per_pixel();

    let word_count = dirty_words_le.len() / 8;
    let tile_count = (header.tiles_x() as usize) * (header.tiles_y() as usize);
    let mut cursor = 0usize;

    for word_idx in 0..word_count {
        let off = word_idx * 8;
        let word = u64::from_le_bytes([
            dirty_words_le[off],
            dirty_words_le[off + 1],
            dirty_words_le[off + 2],
            dirty_words_le[off + 3],
            dirty_words_le[off + 4],
            dirty_words_le[off + 5],
            dirty_words_le[off + 6],
            dirty_words_le[off + 7],
        ]);
        if word == 0 {
            continue;
        }

        let mut bits = word;
        while bits != 0 {
            let lsb = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let idx = word_idx * 64 + lsb;
            if idx >= tile_count {
                continue;
            }

            let tx = idx % tiles_x;
            let ty = idx / tiles_x;
            let dst_x = tx * tile_w;
            let dst_y = ty * tile_h;
            let copy_w = tile_w.min(img_w.saturating_sub(dst_x));
            let copy_h = tile_h.min(img_h.saturating_sub(dst_y));
            let tile_bytes = copy_w * copy_h * bpp;

            if cursor + tile_bytes > payload.len() {
                return;
            }

            let tile_data = &payload[cursor..cursor + tile_bytes];
            cursor += tile_bytes;

            for row in 0..copy_h {
                let py = dst_y + row;
                if py >= img_h {
                    break;
                }
                let dst_row = py * img_w + dst_x;
                let src_row = row * copy_w * bpp;
                for col in 0..copy_w {
                    let src_off = src_row + col * bpp;
                    dst[dst_row + col] = pixel_to_xrgb(header.pixel_format, &tile_data[src_off..]);
                }
            }
        }
    }
}

/// Convert a single pixel from any bspl format to 0x00RRGGBB.
#[inline]
fn pixel_to_xrgb(fmt: PixelFormat, pixel: &[u8]) -> u32 {
    match fmt {
        PixelFormat::Xrgb8888 => u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]),
        PixelFormat::Rgba8 => {
            // [R, G, B, A] → 0x00RRGGBB (ignore alpha for display)
            (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | pixel[2] as u32
        }
        _ => {
            let (r, g, b) = fmt.decode_rgb(pixel);
            (r as u32) << 16 | (g as u32) << 8 | b as u32
        }
    }
}

/// Pick a window scale factor that keeps the window reasonable on screen.
fn pick_scale(w: usize, h: usize) -> minifb::Scale {
    let max_dim = w.max(h);
    if max_dim <= 256 {
        minifb::Scale::X4
    } else if max_dim <= 512 {
        minifb::Scale::X2
    } else {
        minifb::Scale::X1
    }
}
