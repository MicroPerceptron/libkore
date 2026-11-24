use crate::constants::{HEADER_BYTES, MAGIC};
use crate::error::Error;
use crate::format::PixelFormat;
use crate::model::{Animation, AnimationFrame, AnimationFrameData, Encoding, Header};
use crate::read::{read_u16_le, read_u32_le};

/// Parse any bspl asset as an `Animation`.
pub fn parse(data: &[u8]) -> Result<Animation<'_>, Error> {
    let header = parse_header(data)?;

    // Validate that pixel payload exists for single-frame dense assets.
    if header.frame_count == 1 && matches!(header.encoding, Encoding::Dense) {
        let pixel_len = (header.img_w as usize)
            .checked_mul(header.img_h as usize)
            .and_then(|n| n.checked_mul(header.pixel_format.bytes_per_pixel()))
            .ok_or(Error::InvalidGeometry)?;
        // Per-frame: duration_us (4) + dirty bitmask + pixels.
        let dirty_bytes = header.dirty_word_count() * 8;
        let frame_len = 4 + dirty_bytes + pixel_len;
        let total = HEADER_BYTES
            .checked_add(frame_len)
            .ok_or(Error::InvalidGeometry)?;
        if data.len() < total {
            return Err(Error::Truncated);
        }
    }

    Ok(Animation::new(
        header,
        data.get(HEADER_BYTES..).ok_or(Error::Truncated)?,
    ))
}

/// Number of u64 words needed for a tile grid of `tiles_x × tiles_y`.
pub fn dirty_tile_required_words(tiles_x: u32, tiles_y: u32) -> usize {
    let tile_count = (tiles_x as usize).saturating_mul(tiles_y as usize);
    tile_count.div_ceil(64)
}

fn parse_header(data: &[u8]) -> Result<Header, Error> {
    if data.len() < HEADER_BYTES {
        return Err(Error::AssetTooSmall);
    }
    let magic = <&[u8; 4]>::try_from(&data[0..4]).map_err(|_| Error::BadMagic)?;
    if magic != MAGIC {
        return Err(Error::BadMagic);
    }

    let encoding = Encoding::from_u8(data[4]).ok_or(Error::UnsupportedVersion)?;
    let pixel_format = PixelFormat::from_u8(data[5]).ok_or(Error::UnsupportedVersion)?;
    let frame_count = read_u16_le(data, 6).ok_or(Error::Truncated)?;
    let img_w = read_u32_le(data, 8).ok_or(Error::Truncated)?;
    let img_h = read_u32_le(data, 12).ok_or(Error::Truncated)?;
    let tile_w = read_u16_le(data, 16).ok_or(Error::Truncated)?;
    let tile_h = read_u16_le(data, 18).ok_or(Error::Truncated)?;
    // bytes 20-23: reserved, ignored on read.

    if frame_count == 0 || img_w == 0 || img_h == 0 || tile_w == 0 || tile_h == 0 {
        return Err(Error::InvalidGeometry);
    }

    Ok(Header {
        encoding,
        pixel_format,
        frame_count,
        img_w,
        img_h,
        tile_w,
        tile_h,
    })
}

/// Parse the next frame from the byte stream at `cursor`.
///
/// Frame layout (dense/sparse):
///   duration_us: u32, dirty[]: [u64; word_count], payload
///
/// Frame layout (sparse_lz4):
///   duration_us: u32, compressed_len: u32, raw_len: u32,
///   dirty[]: [u64; word_count], lz4_payload
pub(crate) fn parse_next_frame<'a>(
    header: Header,
    bytes: &'a [u8],
    cursor: &mut usize,
) -> Result<AnimationFrame<'a>, Error> {
    let duration_us = read_u32_le(bytes, *cursor).ok_or(Error::Truncated)?;
    *cursor += 4;

    // LZ4 frames have compressed_len and raw_len before the dirty mask.
    let (compressed_payload_len, raw_payload_len) =
        if matches!(header.encoding, Encoding::SparseLz4) {
            let compressed = read_u32_le(bytes, *cursor).ok_or(Error::Truncated)? as usize;
            *cursor += 4;
            let raw = read_u32_le(bytes, *cursor).ok_or(Error::Truncated)? as usize;
            *cursor += 4;
            (compressed, raw)
        } else {
            (0, 0)
        };

    // Dirty bitmask — word count derived from header geometry.
    let word_count = header.dirty_word_count();
    let dirty_bytes_len = word_count * 8;
    let dirty_end = cursor
        .checked_add(dirty_bytes_len)
        .ok_or(Error::Truncated)?;
    let dirty_words_le = bytes.get(*cursor..dirty_end).ok_or(Error::Truncated)?;
    *cursor = dirty_end;

    let data = match header.encoding {
        Encoding::Dense => {
            let pixel_len = (header.img_w as usize)
                .checked_mul(header.img_h as usize)
                .and_then(|n| n.checked_mul(header.pixel_format.bytes_per_pixel()))
                .ok_or(Error::InvalidGeometry)?;
            let end = cursor.checked_add(pixel_len).ok_or(Error::Truncated)?;
            let pixels = bytes.get(*cursor..end).ok_or(Error::Truncated)?;
            *cursor = end;
            AnimationFrameData::Dense { pixels }
        }
        Encoding::Sparse => {
            let payload_len = expected_sparse_payload_len(header, dirty_words_le)?;
            let end = cursor.checked_add(payload_len).ok_or(Error::Truncated)?;
            let payload = bytes.get(*cursor..end).ok_or(Error::Truncated)?;
            *cursor = end;
            AnimationFrameData::Sparse { payload }
        }
        Encoding::SparseLz4 => {
            let expected_raw = expected_sparse_payload_len(header, dirty_words_le)?;
            if raw_payload_len != expected_raw {
                return Err(Error::InvalidGeometry);
            }
            let end = cursor
                .checked_add(compressed_payload_len)
                .ok_or(Error::Truncated)?;
            let compressed_payload = bytes.get(*cursor..end).ok_or(Error::Truncated)?;
            *cursor = end;
            AnimationFrameData::SparseLz4 {
                compressed_payload,
                raw_payload_len,
            }
        }
    };

    Ok(AnimationFrame {
        duration_us,
        dirty_words_le,
        data,
    })
}

pub(crate) fn expected_sparse_payload_len(
    header: Header,
    dirty_words_le: &[u8],
) -> Result<usize, Error> {
    let expected_words = header.dirty_word_count();
    if dirty_words_le.len() != expected_words * 8 {
        return Err(Error::InvalidGeometry);
    }
    let tile_count = (header.tiles_x() as usize) * (header.tiles_y() as usize);
    let mut total = 0usize;
    for word_idx in 0..expected_words {
        let word = u64::from_le_bytes(
            dirty_words_le[word_idx * 8..word_idx * 8 + 8]
                .try_into()
                .map_err(|_| Error::Truncated)?,
        );
        let mut bits = word;
        while bits != 0 {
            let lsb = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let idx = word_idx * 64 + lsb;
            if idx >= tile_count {
                continue;
            }
            let tx = (idx as u32) % header.tiles_x();
            let ty = (idx as u32) / header.tiles_x();
            let tile_src_x = tx * (header.tile_w as u32);
            let tile_src_y = ty * (header.tile_h as u32);
            let copy_w = (header.tile_w as u32).min(header.img_w.saturating_sub(tile_src_x));
            let copy_h = (header.tile_h as u32).min(header.img_h.saturating_sub(tile_src_y));
            let tile_bytes = (copy_w as usize)
                .checked_mul(copy_h as usize)
                .and_then(|n| n.checked_mul(header.pixel_format.bytes_per_pixel()))
                .ok_or(Error::InvalidGeometry)?;
            total = total
                .checked_add(tile_bytes)
                .ok_or(Error::InvalidGeometry)?;
        }
    }
    Ok(total)
}
