//! Text micro-benchmarks — atlas lookup, warm layout, cold raster.
//!
//! Run:  cargo bench -p axrd --features font --bench text -- --max-time 3

#![feature(allocator_api)]

use axrd::PlacedGlyph;
use axrd::font::{
    FontAtlas, FontError, FontFace, GlyphAtlas, GlyphKey, GlyphMetrics, GlyphSource, LineSpec,
    Overflow, VerticalMetrics, layout_line_into,
};
use divan::{Bencher, black_box};
use std::alloc::Global;

fn main() {
    divan::main();
}

const DMSANS: &[u8] = include_bytes!("../assets/fonts/DMSans.ttf");
const TEXT_24: &str = "The quick brown fox jump";

const BLANK: PlacedGlyph = PlacedGlyph {
    ax: 0,
    ay: 0,
    aw: 0,
    ah: 0,
    dx: 0,
    dy: 0,
    subpixel_x: 0,
    subpixel_y: 0,
};

/// A source whose every glyph is one solid pixel, so an atlas can be
/// filled with many distinct keys without font work.
struct Dots;

impl GlyphSource for Dots {
    fn font_id(&self) -> u32 {
        0xD075
    }
    fn glyph_id(&self, c: char) -> u16 {
        c as u16
    }
    fn vertical_metrics(&self, _px: f32) -> VerticalMetrics {
        VerticalMetrics {
            ascent: 1.0,
            descent: 0.0,
            line_gap: 0.0,
        }
    }
    fn metrics(&self, _glyph: u16, _px: f32) -> GlyphMetrics {
        GlyphMetrics {
            xmin: 0,
            ymin: 0,
            width: 1,
            height: 1,
            advance: 1.0,
        }
    }
    fn kern(&self, _l: u16, _r: u16, _px: f32) -> f32 {
        0.0
    }
    fn rasterize_into(
        &self,
        glyph: u16,
        px: f32,
        _phase_x: u8,
        _phase_y: u8,
        out: &mut [u16],
        _stride: usize,
    ) -> Result<GlyphMetrics, FontError> {
        out[0] = u16::MAX;
        Ok(self.metrics(glyph, px))
    }
}

/// Binary search over a 512-entry index, cycling through every key.
#[divan::bench]
fn cache_hit_512_entries(bencher: Bencher) {
    let mut atlas = GlyphAtlas::with_capacity_in(64, 64, 0, 1024, Global).unwrap();
    let keys: Vec<GlyphKey> = (0..512u16)
        .map(|glyph| {
            let phase = (glyph % 4) as u8 * 64;
            atlas.entry(&Dots, glyph, 12.0, phase, 0).unwrap().key
        })
        .collect();
    assert_eq!(atlas.glyph_count(), 512);
    let mut i = 0usize;
    bencher.bench_local(|| {
        let key = &keys[i];
        i = (i + 1) % keys.len();
        black_box(atlas.lookup(black_box(key)).unwrap().ax)
    });
}

/// 24 characters of DM Sans at 12.5 px with every raster already cached.
#[divan::bench]
fn layout_line_into_24_chars_warm(bencher: Bencher) {
    let face = FontFace::parse(DMSANS).unwrap();
    let mut atlas = FontAtlas::new(512, 256).unwrap();
    let spec = LineSpec::new(12.5).with_phase(64, 0);
    let mut out = [BLANK; 32];
    layout_line_into(&face, &mut atlas, TEXT_24, spec, &mut out).unwrap();
    bencher.bench_local(|| {
        layout_line_into(
            black_box(&face),
            &mut atlas,
            black_box(TEXT_24),
            spec,
            &mut out,
        )
        .unwrap()
        .advance_px
    });
}

/// The same line under a 100 px ellipsis bound (measure pass + placement).
#[divan::bench]
fn layout_line_into_24_chars_warm_ellipsis(bencher: Bencher) {
    let face = FontFace::parse(DMSANS).unwrap();
    let mut atlas = FontAtlas::new(512, 256).unwrap();
    let spec = LineSpec::new(12.5).with_max_width(100, Overflow::Ellipsis);
    let mut out = [BLANK; 32];
    layout_line_into(&face, &mut atlas, TEXT_24, spec, &mut out).unwrap();
    bencher.bench_local(|| {
        layout_line_into(
            black_box(&face),
            &mut atlas,
            black_box(TEXT_24),
            spec,
            &mut out,
        )
        .unwrap()
        .glyphs
    });
}

/// One supersampled raster of DM Sans 'A' at 12 px into a scratch plane.
#[divan::bench]
fn rasterize_dmsans_12px_cold(bencher: Bencher) {
    let face = FontFace::parse(DMSANS).unwrap();
    let glyph = face.glyph_id('A');
    let mut out = [0u16; 32 * 32];
    bencher.bench_local(|| {
        face.rasterize_into(black_box(glyph), black_box(12.0), 0, 0, &mut out, 32)
            .unwrap()
            .width
    });
}
