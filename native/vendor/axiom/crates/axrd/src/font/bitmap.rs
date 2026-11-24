//! Bitmap glyph source: 1-bit packed rows scaled by an integer factor.

use super::{FontError, GlyphMetrics, GlyphSource, VerticalMetrics, fnv1a, round_f32, valid_px};

/// Largest integer scale a [`BitmapFace`] renders at.
pub const BITMAP_MAX_SCALE: u8 = 4;

/// A fixed-cell bitmap font (a VGA ROM font, Terminus 12×24, …).
///
/// `rows` holds `count` glyphs of `height` rows each; every row is
/// `ceil(width / 8)` bytes, most significant bit leftmost.  Glyph ids are
/// `1..=count` in code-point order from `first`; `0` is "missing" and
/// renders as a blank cell with the cell advance.
///
/// The requested pixel size selects an integer scale: `round(px / height)`
/// clamped to `1..=`[`BITMAP_MAX_SCALE`].  A 12×24 face therefore renders
/// at 1× for `px` in `12..36` and at 2× from `36` up.  Cells are trimmed to
/// their ink so the atlas never stores blank margins and ink-centring
/// works; coverage is exactly `0` or `65535`, and a fractional phase
/// rounds to the nearest whole pixel instead of blurring the bitmap.
#[derive(Clone, Copy, Debug)]
pub struct BitmapFace<'a> {
    /// Cell width in pixels (`1..=255`).
    pub width: u8,
    /// Cell height in pixels (`1..=255`).
    pub height: u8,
    /// Packed 1-bit rows, glyph-major.
    pub rows: &'a [u8],
    /// Code point of glyph id `1`.
    pub first: char,
    /// Number of consecutive glyphs from `first`.
    pub count: u16,
    /// Rows of the cell below the baseline.
    pub descent: u8,
    /// Identity mixed into this face's glyph keys.
    pub font: u32,
}

impl<'a> BitmapFace<'a> {
    /// Wrap packed rows, checking that they cover `count` cells.
    ///
    /// The baseline defaults to the cell bottom ([`Self::with_descent`]
    /// moves it) and the font id to a hash of the row slice.
    pub fn new(
        width: u8,
        height: u8,
        rows: &'a [u8],
        first: char,
        count: u16,
    ) -> Result<Self, FontError> {
        if width == 0 || height == 0 {
            return Err(FontError::InvalidRaster);
        }
        let needed = usize::from(count) * usize::from(height) * bytes_per_row(width);
        if rows.len() < needed {
            return Err(FontError::InvalidRaster);
        }
        let mut seed = [0u8; 16];
        seed[..8].copy_from_slice(&(rows.as_ptr() as usize as u64).to_le_bytes());
        seed[8..].copy_from_slice(&(rows.len() as u64).to_le_bytes());
        let hash = fnv1a(&seed);
        Ok(Self {
            width,
            height,
            rows,
            first,
            count,
            descent: 0,
            font: (hash as u32) ^ ((hash >> 32) as u32),
        })
    }

    /// The crate's 8×16 VGA ROM font as a bitmap source.
    pub fn vga_8x16() -> Self {
        let mut face = Self::new(8, 16, &crate::font8x16::FONT_8X16, ' ', 95)
            .expect("FONT_8X16 covers 95 glyphs");
        face.descent = 4;
        face.font = 0x5647_4116; // "VGA" + 16
        face
    }

    /// Set how many cell rows lie below the baseline.
    pub fn with_descent(mut self, descent: u8) -> Self {
        self.descent = descent.min(self.height);
        self
    }

    /// Assign the id mixed into this face's glyph keys.
    pub fn with_font_id(mut self, font: u32) -> Self {
        self.font = font;
        self
    }

    /// Bytes per packed row.
    #[inline]
    pub const fn bytes_per_row(&self) -> usize {
        bytes_per_row(self.width)
    }

    /// The integer scale `px` selects.
    pub fn scale_for(&self, px: f32) -> Result<u8, FontError> {
        if !valid_px(px) {
            return Err(FontError::InvalidPixelSize);
        }
        let scale = round_f32(px / f32::from(self.height));
        Ok(scale.clamp(1.0, f32::from(BITMAP_MAX_SCALE)) as u8)
    }

    /// Whether cell pixel `(x, y)` of `glyph` is set.  Out-of-range
    /// coordinates and the missing glyph are clear.
    #[inline]
    pub fn bit(&self, glyph: u16, x: u16, y: u16) -> bool {
        if glyph == 0
            || glyph > self.count
            || x >= u16::from(self.width)
            || y >= u16::from(self.height)
        {
            return false;
        }
        let bpr = self.bytes_per_row();
        let row = (usize::from(glyph) - 1) * usize::from(self.height) + usize::from(y);
        let byte = self.rows[row * bpr + usize::from(x) / 8];
        byte & (0x80 >> (x % 8)) != 0
    }

    /// Ink bounds of a cell as `(x0, y0, x1, y1)` (half-open), or `None`
    /// for a blank cell.
    fn ink(&self, glyph: u16) -> Option<(u8, u8, u8, u8)> {
        if glyph == 0 || glyph > self.count {
            return None;
        }
        let bpr = self.bytes_per_row();
        let base = (usize::from(glyph) - 1) * usize::from(self.height) * bpr;
        let mut bounds: Option<(u8, u8, u8, u8)> = None;
        for y in 0..self.height {
            let row = &self.rows[base + usize::from(y) * bpr..base + usize::from(y) * bpr + bpr];
            let mut x0 = None;
            let mut x1 = 0u8;
            for x in 0..self.width {
                if row[usize::from(x) / 8] & (0x80 >> (x % 8)) != 0 {
                    if x0.is_none() {
                        x0 = Some(x);
                    }
                    x1 = x + 1;
                }
            }
            if let Some(x0) = x0 {
                bounds = Some(match bounds {
                    None => (x0, y, x1, y + 1),
                    Some((bx0, by0, bx1, _)) => (bx0.min(x0), by0, bx1.max(x1), y + 1),
                });
            }
        }
        bounds
    }

    fn metrics_at(&self, glyph: u16, scale: u8) -> GlyphMetrics {
        let s = i32::from(scale);
        let advance = (i32::from(self.width) * s) as f32;
        match self.ink(glyph) {
            None => GlyphMetrics {
                advance,
                ..GlyphMetrics::default()
            },
            Some((x0, y0, x1, y1)) => GlyphMetrics {
                xmin: i32::from(x0) * s,
                ymin: (i32::from(self.height) - i32::from(self.descent) - i32::from(y1)) * s,
                width: u16::from(x1 - x0) * u16::from(scale),
                height: u16::from(y1 - y0) * u16::from(scale),
                advance,
            },
        }
    }
}

#[inline]
const fn bytes_per_row(width: u8) -> usize {
    (width as usize).div_ceil(8)
}

impl GlyphSource for BitmapFace<'_> {
    #[inline]
    fn font_id(&self) -> u32 {
        self.font
    }

    #[inline]
    fn glyph_id(&self, c: char) -> u16 {
        let offset = (c as u32).wrapping_sub(self.first as u32);
        if offset < u32::from(self.count) {
            offset as u16 + 1
        } else {
            0
        }
    }

    fn vertical_metrics(&self, px: f32) -> VerticalMetrics {
        let Ok(scale) = self.scale_for(px) else {
            return VerticalMetrics::default();
        };
        let s = f32::from(scale);
        VerticalMetrics {
            ascent: f32::from(self.height - self.descent) * s,
            descent: f32::from(self.descent) * s,
            line_gap: 0.0,
        }
    }

    fn metrics(&self, glyph: u16, px: f32) -> GlyphMetrics {
        match self.scale_for(px) {
            Ok(scale) => self.metrics_at(glyph, scale),
            Err(_) => GlyphMetrics::default(),
        }
    }

    #[inline]
    fn kern(&self, _left: u16, _right: u16, _px: f32) -> f32 {
        0.0
    }

    fn rasterize_into(
        &self,
        glyph: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
        out: &mut [u16],
        stride: usize,
    ) -> Result<GlyphMetrics, FontError> {
        let scale = self.scale_for(px)?;
        let metrics = self.metrics_at(glyph, scale);
        let Some((x0, y0, _, _)) = self.ink(glyph) else {
            return Ok(metrics);
        };
        let (sw, sh) = metrics.stored_extent(phase_x, phase_y);
        let (sw, sh) = (usize::from(sw), usize::from(sh));
        if sw > stride || out.len() < (sh - 1) * stride + sw {
            return Err(FontError::InvalidRaster);
        }
        // A phase rounds to the nearest whole pixel: the bitmap either
        // stays put or shifts one pixel into the expansion column / row.
        let shift_x = usize::from(phase_x >= 128);
        let shift_y = usize::from(phase_y >= 128);
        let s = usize::from(scale);
        let (w, h) = (usize::from(metrics.width), usize::from(metrics.height));
        for y in 0..sh {
            let row = &mut out[y * stride..y * stride + sw];
            for (x, dst) in row.iter_mut().enumerate() {
                let inside =
                    (shift_x..shift_x + w).contains(&x) && (shift_y..shift_y + h).contains(&y);
                *dst = if inside
                    && self.bit(
                        glyph,
                        u16::from(x0) + ((x - shift_x) / s) as u16,
                        u16::from(y0) + ((y - shift_y) / s) as u16,
                    ) {
                    u16::MAX
                } else {
                    0
                };
            }
        }
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font8x16::FONT_8X16;

    #[test]
    fn glyph_ids_cover_the_declared_range() {
        let face = BitmapFace::vga_8x16();
        assert_eq!(face.glyph_id(' '), 1);
        assert_eq!(face.glyph_id('A'), u16::from(b'A' - b' ') + 1);
        assert_eq!(face.glyph_id('~'), 95);
        assert_eq!(face.glyph_id('\u{7f}'), 0);
        assert_eq!(face.glyph_id('\n'), 0);
        assert_eq!(face.glyph_id('…'), 0);
    }

    #[test]
    fn rows_must_cover_the_declared_glyphs() {
        assert!(matches!(
            BitmapFace::new(8, 16, &FONT_8X16[..100], ' ', 95),
            Err(FontError::InvalidRaster)
        ));
        assert!(matches!(
            BitmapFace::new(0, 16, &FONT_8X16, ' ', 95),
            Err(FontError::InvalidRaster)
        ));
        // 12-wide rows are two bytes each.
        let rows = [0u8; 2 * 24 * 3];
        let face = BitmapFace::new(12, 24, &rows, 'a', 3).unwrap();
        assert_eq!(face.bytes_per_row(), 2);
        assert!(BitmapFace::new(12, 24, &rows[..143], 'a', 3).is_err());
    }

    #[test]
    fn pixel_size_selects_an_integer_scale() {
        let face = BitmapFace::vga_8x16();
        assert_eq!(face.scale_for(8.0).unwrap(), 1);
        assert_eq!(face.scale_for(16.0).unwrap(), 1);
        assert_eq!(face.scale_for(23.0).unwrap(), 1);
        assert_eq!(face.scale_for(24.0).unwrap(), 2);
        assert_eq!(face.scale_for(32.0).unwrap(), 2);
        assert_eq!(face.scale_for(64.0).unwrap(), 4);
        assert_eq!(face.scale_for(640.0).unwrap(), 4);
        assert!(matches!(
            face.scale_for(0.0),
            Err(FontError::InvalidPixelSize)
        ));
        let terminus = BitmapFace::new(12, 24, &[0u8; 2 * 24], ' ', 1).unwrap();
        assert_eq!(terminus.scale_for(24.0).unwrap(), 1);
        assert_eq!(terminus.scale_for(35.0).unwrap(), 1);
        assert_eq!(terminus.scale_for(36.0).unwrap(), 2);
    }

    #[test]
    fn vga_a_at_scale_two_is_the_nearest_scaled_bitmap() {
        let face = BitmapFace::vga_8x16();
        let glyph = face.glyph_id('A');
        let cell = &FONT_8X16[usize::from(glyph - 1) * 16..usize::from(glyph) * 16];
        // Trim the cell to its ink by hand.
        let rows: alloc::vec::Vec<usize> = (0..16).filter(|&y| cell[y] != 0).collect();
        let (y0, y1) = (rows[0], rows[rows.len() - 1] + 1);
        let mut x0 = 8;
        let mut x1 = 0;
        for &bits in cell {
            for x in 0..8 {
                if bits & (0x80 >> x) != 0 {
                    x0 = x0.min(x);
                    x1 = x1.max(x + 1);
                }
            }
        }
        let metrics = face.metrics(glyph, 32.0);
        assert_eq!(
            (metrics.xmin, metrics.width, metrics.height),
            (
                x0 as i32 * 2,
                ((x1 - x0) * 2) as u16,
                ((y1 - y0) * 2) as u16
            )
        );
        assert_eq!(metrics.ymin, (16 - 4 - y1 as i32) * 2);
        assert_eq!(metrics.advance, 16.0);

        let stride = 24;
        let mut out = [0xAAAAu16; 24 * 32];
        let raster = face
            .rasterize_into(glyph, 32.0, 0, 0, &mut out, stride)
            .unwrap();
        assert_eq!(raster, metrics);
        for y in 0..usize::from(metrics.height) {
            for x in 0..usize::from(metrics.width) {
                let expected = cell[y0 + y / 2] & (0x80 >> (x0 + x / 2)) != 0;
                assert_eq!(out[y * stride + x] == u16::MAX, expected, "({x},{y})");
                assert!(out[y * stride + x] == 0 || out[y * stride + x] == u16::MAX);
            }
            // Untouched beyond the box.
            assert_eq!(out[y * stride + usize::from(metrics.width)], 0xAAAA);
        }
        assert_eq!(out[usize::from(metrics.height) * stride], 0xAAAA);

        // A half-pixel phase shifts the crisp bitmap by one whole pixel
        // into the expansion column / row.
        let mut shifted = [0u16; 24 * 32];
        face.rasterize_into(glyph, 32.0, 200, 128, &mut shifted, stride)
            .unwrap();
        for y in 0..usize::from(metrics.height) {
            for x in 0..usize::from(metrics.width) {
                assert_eq!(shifted[(y + 1) * stride + x + 1], out[y * stride + x]);
            }
            assert_eq!(shifted[(y + 1) * stride], 0);
        }
        assert!(
            shifted[..usize::from(metrics.width) + 1]
                .iter()
                .all(|&c| c == 0)
        );
        // A small phase leaves the bitmap in place and the expansion clear.
        let mut small = [0u16; 24 * 32];
        face.rasterize_into(glyph, 32.0, 60, 0, &mut small, stride)
            .unwrap();
        for y in 0..usize::from(metrics.height) {
            assert_eq!(
                small[y * stride..y * stride + usize::from(metrics.width)],
                out[y * stride..y * stride + usize::from(metrics.width)]
            );
            assert_eq!(small[y * stride + usize::from(metrics.width)], 0);
        }
    }

    #[test]
    fn blank_and_missing_glyphs_keep_their_advance() {
        let face = BitmapFace::vga_8x16();
        let space = face.metrics(face.glyph_id(' '), 16.0);
        assert!(space.is_empty());
        assert_eq!(space.advance, 8.0);
        let missing = face.metrics(0, 16.0);
        assert!(missing.is_empty());
        assert_eq!(missing.advance, 8.0);
        let mut out = [7u16; 4];
        assert_eq!(
            face.rasterize_into(0, 16.0, 0, 0, &mut out, 2).unwrap(),
            missing
        );
        assert_eq!(out, [7; 4], "nothing written for an empty glyph");
        let vertical = face.vertical_metrics(16.0);
        assert_eq!((vertical.ascent, vertical.descent), (12.0, 4.0));
    }

    #[test]
    fn short_destination_is_rejected() {
        let face = BitmapFace::vga_8x16();
        let glyph = face.glyph_id('A');
        let mut out = [0u16; 8];
        assert_eq!(
            face.rasterize_into(glyph, 16.0, 0, 0, &mut out, 8),
            Err(FontError::InvalidRaster)
        );
    }
}
