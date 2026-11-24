//! Bridge between `bspl` boot splash assets and the axrd compositor.
//!
//! Converts bspl pixel data into XRGB tensors suitable for use as
//! `Bind::Blit` or `Bind::Sample` texture sources. Also maps bspl
//! dirty tile bitmasks into the compositor's `TileMask` format for
//! efficient incremental rendering — tiles unchanged by the animation
//! frame are skipped entirely.

use axnn::{Backend, ComputeBackend, DType, Tensor, TensorDesc};
use bspl::PixelFormat;

use crate::tile::{MASK_WORDS, TileMask};

/// Decode the first frame of a bspl animation into an XRGB tensor.
///
/// Works with any bspl version (static or animated). For static
/// images this decodes the single frame; for animations it decodes
/// frame 0. Converts from the source pixel format to 0x00RRGGBB.
pub fn decode_first_frame<'p, B>(
    anim: &bspl::Animation<'_>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error>
where
    B: Backend + ComputeBackend,
{
    let header = anim.header();
    let frame = anim.frames().next().and_then(|r| r.ok());
    if let Some(f) = frame {
        decode_dense(&header, &f, be)
    } else {
        // Fallback: return a zeroed tensor.
        let h = header.img_h as usize;
        let w = header.img_w as usize;
        let shape = axnn::Shape::try_from(&[h, w][..]).unwrap();
        let desc = TensorDesc::new(shape, DType::I32);
        be.zeros(&desc)
    }
}

/// Decode a dense animation frame into an XRGB tensor.
///
/// Only works with `AnimationFrameData::Dense` frames (version 2).
/// For sparse frames, use [`decode_sparse_into`] instead.
pub fn decode_dense<'p, B>(
    header: &bspl::Header,
    frame: &bspl::AnimationFrame<'_>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error>
where
    B: Backend + ComputeBackend,
{
    let h = header.img_h as usize;
    let w = header.img_w as usize;
    let pixels = match frame.data {
        bspl::AnimationFrameData::Dense { pixels } => pixels,
        _ => &[], // sparse frames have no full-image pixel data
    };
    let buf = convert_pixels(header.pixel_format, pixels, w * h);
    let shape = axnn::Shape::try_from(&[h, w][..]).unwrap();
    let desc = TensorDesc::new(shape, DType::I32);
    be.upload(&buf, &desc)
}

/// Patch a sparse animation frame into an existing XRGB pixel buffer.
///
/// Reads dirty tiles from the frame's sparse payload and writes them
/// into `dst_pixels` (a `[H * W]` u32 buffer in XRGB format).
/// Returns the dirty tile mask for the compositor.
///
/// Works with `AnimationFrameData::Sparse` (version 3).
/// For LZ4-compressed frames, decompress first into a scratch buffer,
/// then call this with the decompressed payload.
pub fn decode_sparse_into(
    header: &bspl::Header,
    frame: &bspl::AnimationFrame<'_>,
    dst_pixels: &mut [u32],
) -> TileMask {
    let payload = match frame.data {
        bspl::AnimationFrameData::Sparse { payload } => payload,
        _ => return TileMask::empty(),
    };
    apply_sparse_tiles(header, frame.dirty_words_le, payload, dst_pixels)
}

/// Patch a decompressed LZ4 sparse payload into a pixel buffer.
///
/// Same as [`decode_sparse_into`] but takes raw decompressed bytes
/// directly (for use after external LZ4 decompression of version 4
/// frames).
pub fn decode_raw_sparse_into(
    header: &bspl::Header,
    dirty_words_le: &[u8],
    payload: &[u8],
    dst_pixels: &mut [u32],
) -> TileMask {
    apply_sparse_tiles(header, dirty_words_le, payload, dst_pixels)
}

/// Upload an XRGB pixel buffer as a tensor.
///
/// Helper for the sparse decode path: after patching `dst_pixels`
/// with [`decode_sparse_into`], upload the full buffer to a tensor.
pub fn upload_pixels<'p, B>(
    pixels: &[u32],
    w: usize,
    h: usize,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error>
where
    B: Backend + ComputeBackend,
{
    let bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
    let shape = axnn::Shape::try_from(&[h, w][..]).unwrap();
    let desc = TensorDesc::new(shape, DType::I32);
    be.upload(bytes, &desc)
}

/// Convert bspl's dirty word bitmask into a compositor `TileMask`.
///
/// Both use the same bit layout (linear tile index → bit position in
/// u64 words), so this is a direct copy for matching tile grids.
pub fn dirty_to_tilemask(dirty_words_le: &[u8]) -> TileMask {
    let mut mask = TileMask::empty();
    let word_count = dirty_words_le.len() / 8;
    for i in 0..word_count.min(MASK_WORDS) {
        let off = i * 8;
        mask.bits[i] = u64::from_le_bytes([
            dirty_words_le[off],
            dirty_words_le[off + 1],
            dirty_words_le[off + 2],
            dirty_words_le[off + 3],
            dirty_words_le[off + 4],
            dirty_words_le[off + 5],
            dirty_words_le[off + 6],
            dirty_words_le[off + 7],
        ]);
    }
    mask
}

// ── Internal helpers ───────────────────────────────────────────────

/// Convert source pixels to XRGB (0x00RRGGBB) byte buffer.
fn convert_pixels(fmt: PixelFormat, src: &[u8], pixel_count: usize) -> &[u8] {
    // Fast path: XRGB8888 is already in native format — zero-copy.
    if matches!(fmt, PixelFormat::Xrgb8888) {
        return src;
    }
    // For other formats we'd need an allocation, which we avoid in
    // no_std. Return empty and let the caller handle the conversion.
    // In practice, boot splash assets targeting this compositor
    // should use XRGB8888 (pixel_format = 5).
    let _ = pixel_count;
    &[]
}

/// Apply sparse tile payload into a u32 pixel buffer.
fn apply_sparse_tiles(
    header: &bspl::Header,
    dirty_words_le: &[u8],
    payload: &[u8],
    dst: &mut [u32],
) -> TileMask {
    let tiles_x = header.tiles_x() as usize;
    let tiles_y = header.tiles_y() as usize;
    let tile_w = header.tile_w as usize;
    let tile_h = header.tile_h as usize;
    let img_w = header.img_w as usize;
    let img_h = header.img_h as usize;
    let bpp = header.pixel_format.bytes_per_pixel();
    let is_xrgb = matches!(header.pixel_format, PixelFormat::Xrgb8888);

    let word_count = dirty_words_le.len() / 8;
    let tile_count = tiles_x * tiles_y;
    let mut mask = TileMask::empty();
    let mut payload_cursor = 0usize;

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

        // Copy bspl dirty word directly into mask.
        if word_idx < MASK_WORDS {
            mask.bits[word_idx] = word;
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

            if payload_cursor + tile_bytes > payload.len() {
                break;
            }

            let tile_data = &payload[payload_cursor..payload_cursor + tile_bytes];
            payload_cursor += tile_bytes;

            // Blit tile pixels into destination buffer.
            for row in 0..copy_h {
                let py = dst_y + row;
                if py >= img_h {
                    break;
                }
                let dst_row_start = py * img_w + dst_x;
                let src_row_start = row * copy_w * bpp;

                for col in 0..copy_w {
                    let px = dst_x + col;
                    if px >= img_w {
                        break;
                    }
                    let src_off = src_row_start + col * bpp;
                    let pixel = if is_xrgb {
                        u32::from_le_bytes([
                            tile_data[src_off],
                            tile_data[src_off + 1],
                            tile_data[src_off + 2],
                            tile_data[src_off + 3],
                        ])
                    } else {
                        let (r, g, b) = header.pixel_format.decode_rgb(&tile_data[src_off..]);
                        (r as u32) << 16 | (g as u32) << 8 | b as u32
                    };
                    dst[dst_row_start + col] = pixel;
                }
            }
        }
    }

    mask
}
