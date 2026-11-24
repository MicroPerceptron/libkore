use crate::constants::{HEADER_BYTES, MAGIC};
use crate::{AnimationFrameData, Encoding, Error, PixelFormat, dirty_tile_required_words, parse};
use alloc::{vec, vec::Vec};

/// Build a minimal header with the new layout.
fn make_header(
    encoding: Encoding,
    pixel_format: PixelFormat,
    frame_count: u16,
    img_w: u32,
    img_h: u32,
    tile_w: u16,
    tile_h: u16,
) -> Vec<u8> {
    let mut data = vec![0u8; HEADER_BYTES];
    data[0..4].copy_from_slice(MAGIC);
    data[4] = encoding as u8;
    data[5] = pixel_format as u8;
    data[6..8].copy_from_slice(&frame_count.to_le_bytes());
    data[8..12].copy_from_slice(&img_w.to_le_bytes());
    data[12..16].copy_from_slice(&img_h.to_le_bytes());
    data[16..18].copy_from_slice(&tile_w.to_le_bytes());
    data[18..20].copy_from_slice(&tile_h.to_le_bytes());
    // 20-23: reserved (zeros)
    data
}

#[test]
fn parse_truncated() {
    let data = make_header(Encoding::Dense, PixelFormat::Rgba8, 1, 16, 9, 8, 8);
    // Header-only, no frame data.
    assert_eq!(parse(&data).unwrap_err(), Error::Truncated);
}

#[test]
fn parse_single_frame_dense() {
    // 2×1 Rgba8, dense, 1 frame.
    let mut data = make_header(Encoding::Dense, PixelFormat::Rgba8, 1, 2, 1, 2, 1);
    // Frame: duration_us + dirty mask (1 word) + 2 pixels.
    data.extend_from_slice(&33_333u32.to_le_bytes()); // duration
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tile dirty
    data.extend_from_slice(&[255, 0, 0, 255]); // pixel 0
    data.extend_from_slice(&[0, 255, 0, 255]); // pixel 1

    let anim = parse(&data).expect("should parse");
    assert_eq!(anim.header().frame_count, 1);
    assert!(matches!(anim.header().encoding, Encoding::Dense));

    let frames: Vec<_> = anim.frames().collect();
    assert_eq!(frames.len(), 1);
    let frame = frames[0].as_ref().expect("frame ok");
    assert_eq!(frame.duration_us, 33_333);
    assert_eq!(frame.dirty_words_le.len(), 8); // 1 word = 8 bytes
    match frame.data {
        AnimationFrameData::Dense { pixels } => {
            assert_eq!(pixels.len(), 8);
            assert_eq!(&pixels[0..4], &[255, 0, 0, 255]);
        }
        _ => panic!("expected dense"),
    }
}

#[test]
fn parse_sparse_lz4_header() {
    let data = make_header(Encoding::SparseLz4, PixelFormat::Rgba8, 2, 16, 9, 8, 8);
    let anim = parse(&data).expect("should parse");
    assert!(matches!(anim.header().encoding, Encoding::SparseLz4));
    assert_eq!(anim.header().frame_count, 2);
}

#[test]
fn pixel_format_decodes_rgb565() {
    let rgb = PixelFormat::Rgb565.decode_rgb(&[0x00, 0xF8]);
    assert_eq!(rgb, (255, 0, 0));
}

#[test]
fn dirty_words_helper_counts_tiles() {
    assert_eq!(dirty_tile_required_words(80, 45), 57);
    assert_eq!(dirty_tile_required_words(1, 1), 1);
    assert_eq!(dirty_tile_required_words(0, 0), 0);
}

#[test]
fn header_dirty_word_count() {
    let data = make_header(Encoding::Dense, PixelFormat::Rgba8, 1, 128, 64, 16, 16);
    let _ = parse(&data).unwrap_err(); // truncated, but header parsed fine
    // Can't test header directly from error, so build manually:
    let h = crate::Header {
        encoding: Encoding::Dense,
        pixel_format: PixelFormat::Rgba8,
        frame_count: 1,
        img_w: 128,
        img_h: 64,
        tile_w: 16,
        tile_h: 16,
    };
    // 128/16 = 8 tiles_x, 64/16 = 4 tiles_y = 32 tiles → 1 word
    assert_eq!(h.dirty_word_count(), 1);
    assert_eq!(h.tiles_x(), 8);
    assert_eq!(h.tiles_y(), 4);
}

#[test]
fn dense_frame_iterator_reads_payload() {
    let mut data = make_header(Encoding::Dense, PixelFormat::Rgba8, 1, 2, 1, 2, 1);
    // 1 tile (2×1), 1 dirty word.
    data.extend_from_slice(&16u32.to_le_bytes()); // duration_us
    data.extend_from_slice(&1u64.to_le_bytes()); // dirty mask (all set)
    data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]); // 2 RGBA pixels

    let anim = parse(&data).expect("should parse");
    let frame = anim.frames().next().expect("one frame").expect("frame ok");
    assert_eq!(frame.duration_us, 16);
    assert_eq!(frame.dirty_words_le.len(), 8);
    match frame.data {
        AnimationFrameData::Dense { pixels } => assert_eq!(pixels.len(), 8),
        _ => panic!("expected dense"),
    }
}
