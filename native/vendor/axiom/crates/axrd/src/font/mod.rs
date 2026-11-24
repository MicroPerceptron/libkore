//! Text preparation for [`crate::Bind::TextRun`]: glyph sources, an indexed
//! glyph atlas and a bounded single-line layout.
//!
//! # Scope
//!
//! This module lays out **one line of left-to-right text with pair
//! kerning**, bounded by an optional maximum width that is enforced either
//! by dropping trailing glyphs ([`Overflow::Clip`]) or by replacing them
//! with an ellipsis ([`Overflow::Ellipsis`]).  That is the whole contract.
//! It deliberately does **not** implement
//!
//! * OpenType shaping (ligatures, mark positioning, contextual forms),
//! * line wrapping or paragraph layout,
//! * bidirectional or vertical text.
//!
//! Callers that need any of those shape externally and feed glyph ids to
//! [`GlyphAtlas::entry`] / [`GlyphAtlas::cache_glyph_id`] themselves.  The
//! compositor never sees code points or font tables: it consumes
//! [`PlacedGlyph`]s plus the atlas coverage plane.
//!
//! # Glyph sources
//!
//! Anything implementing [`GlyphSource`] can feed the atlas.  Two are
//! provided:
//!
//! * [`FontFace`] — a parsed OpenType font (via `femtofont`).  Glyphs are
//!   rasterised at [`FontFace::supersampling`]× the requested size and
//!   area-reconstructed directly at their final 1/256-pixel phase, so the
//!   atlas holds phase-exact coverage and the compositor never bilinearly
//!   translates a mask.
//! * [`BitmapFace`] — a 1-bit packed bitmap font (a VGA ROM font, Terminus
//!   12×24, …).  The requested pixel size selects an integer scale of one
//!   to four; coverage is crisp (`0` or `65535`) and phases round to the
//!   nearest whole pixel.
//!
//! # Atlas
//!
//! [`GlyphAtlas`] is a shelf-packed coverage plane plus a sorted hash index
//! ([`GlyphKey`] → placement), looked up by binary search.  Storage is
//! generic over [`AtlasStorage`]: [`OwnedStorage`] allocates its planes and
//! index once, at construction, from an explicit
//! [`Allocator`](core::alloc::Allocator); [`BorrowedStorage`] wraps a
//! caller-owned coverage slice (an arena page) with an inline fixed-capacity
//! index and never allocates.  [`FontAtlas`] is the owned, global-allocator
//! configuration that also keeps the packed premultiplied-white `u32` view
//! the raster program binds.
//!
//! # Layout
//!
//! [`layout_line_into`] is allocation-free on the warm path: every glyph is
//! a key hash, a binary search and a pen advance.  On a miss it rasterises
//! straight into the atlas plane (a `FontFace` miss allocates inside
//! `femtofont` for the outline raster; a `BitmapFace` miss allocates
//! nothing).  A full plane answers [`FontError::AtlasFull`], a full index
//! [`FontError::IndexFull`], and an `out` slice too small for the run
//! [`FontError::RunCapacity`].  [`FontFace::layout_line_at`] remains as an
//! allocating, multi-line convenience over it.

mod atlas;
mod bitmap;
mod face;
mod layout;

use core::fmt;

pub use atlas::{
    AtlasGlyph, AtlasStorage, BorrowedStorage, DEFAULT_GLYPH_CAPACITY, FontAtlas, GlyphAtlas,
    IndexSlot, OwnedStorage,
};
pub use bitmap::{BITMAP_MAX_SCALE, BitmapFace};
pub use face::{DEFAULT_OUTLINE_CACHE_ENTRIES, DEFAULT_SUPERSAMPLING, FontFace, TextRun};
pub use layout::{LineMetrics, LineSpec, Overflow, layout_line_into};

/// Font preparation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontError {
    /// The supplied byte slice was not a supported, valid OpenType font.
    InvalidFont(&'static str),
    /// Atlas dimensions must both be non-zero (and a borrowed plane must
    /// hold exactly `width × height` samples).
    InvalidAtlasSize,
    /// The requested atlas dimensions overflow addressable memory.
    AtlasSizeOverflow,
    /// Glyph supersampling must be between one and eight samples per axis.
    InvalidSupersampling,
    /// The requested pixel size is not finite and positive.
    InvalidPixelSize,
    /// The atlas plane does not have space for another glyph.
    AtlasFull,
    /// The atlas index has no slot for another distinct glyph.
    IndexFull,
    /// The output glyph slice is too small for the run.
    RunCapacity,
    /// A glyph bitmap or placement does not fit `PlacedGlyph`'s ABI.
    CoordinateOverflow,
    /// Rasterizer coverage was inconsistent with its reported metrics, or a
    /// bitmap font's row data does not cover its declared glyphs.
    InvalidRaster,
    /// The allocator could not reserve an atlas or cache entry.
    Allocation,
}

impl fmt::Display for FontError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFont(message) => write!(f, "invalid font: {message}"),
            Self::InvalidAtlasSize => f.write_str("font atlas dimensions must be non-zero"),
            Self::AtlasSizeOverflow => f.write_str("font atlas size overflows addressable memory"),
            Self::InvalidSupersampling => f.write_str("font atlas supersampling must be in 1..=8"),
            Self::InvalidPixelSize => f.write_str("font pixel size must be finite and positive"),
            Self::AtlasFull => f.write_str("font atlas is full"),
            Self::IndexFull => f.write_str("font atlas glyph index is full"),
            Self::RunCapacity => f.write_str("glyph output slice is too small for the run"),
            Self::CoordinateOverflow => f.write_str("glyph does not fit the compositor ABI"),
            Self::InvalidRaster => f.write_str("font raster size disagrees with its metrics"),
            Self::Allocation => f.write_str("font atlas allocation failed"),
        }
    }
}

/// Identity of one cached raster: which font, which glyph, at which size
/// and subpixel phase.
///
/// `px_bits` is the `f32` bit pattern of the pixel size, so two sizes that
/// print the same but differ in the last ulp are (correctly) distinct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GlyphKey {
    /// [`GlyphSource::font_id`] of the source that rasterised the glyph.
    pub font: u32,
    /// Glyph index within that source.
    pub glyph: u16,
    /// `px.to_bits()`.
    pub px_bits: u32,
    /// Horizontal phase baked into the raster, in 1/256 pixel.
    pub phase_x: u8,
    /// Vertical phase baked into the raster, in 1/256 pixel.
    pub phase_y: u8,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl GlyphKey {
    /// FNV-1a over the key's twelve bytes.  This is the sort key of the
    /// atlas index; equal hashes are disambiguated by a full-key compare.
    #[inline]
    pub const fn hash(&self) -> u64 {
        let mut h = FNV_OFFSET;
        let font = self.font.to_le_bytes();
        let px = self.px_bits.to_le_bytes();
        let glyph = self.glyph.to_le_bytes();
        let bytes = [
            font[0],
            font[1],
            font[2],
            font[3],
            glyph[0],
            glyph[1],
            px[0],
            px[1],
            px[2],
            px[3],
            self.phase_x,
            self.phase_y,
        ];
        let mut i = 0;
        while i < bytes.len() {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(FNV_PRIME);
            i += 1;
        }
        h
    }
}

/// FNV-1a over an arbitrary byte string; used to derive default font ids.
pub(crate) const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    h
}

/// Placement metrics of one glyph at one pixel size.
///
/// The bitmap is `width × height` whole pixels whose bottom-left corner
/// sits at `(xmin, ymin)` relative to the pen position on the baseline,
/// Y up (so a descender has a negative `ymin`).  `advance` is the pen
/// movement in fractional pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlyphMetrics {
    /// Whole-pixel offset of the bitmap's left edge from the pen.
    pub xmin: i32,
    /// Whole-pixel offset of the bitmap's bottom edge from the baseline.
    pub ymin: i32,
    /// Bitmap width in whole pixels, before phase expansion.
    pub width: u16,
    /// Bitmap height in whole pixels, before phase expansion.
    pub height: u16,
    /// Horizontal advance in pixels.
    pub advance: f32,
}

impl GlyphMetrics {
    /// Whether the glyph has no ink (a space, a control character).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Stored bitmap extent once the fractional phase is baked in: a
    /// non-zero phase spills coverage into one extra column / row.
    #[inline]
    pub const fn stored_extent(&self, phase_x: u8, phase_y: u8) -> (u16, u16) {
        if self.is_empty() {
            (0, 0)
        } else {
            (
                self.width.saturating_add((phase_x != 0) as u16),
                self.height.saturating_add((phase_y != 0) as u16),
            )
        }
    }
}

/// Font-wide vertical metrics at one pixel size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VerticalMetrics {
    /// Distance from the baseline up to the top of the line box, in pixels.
    pub ascent: f32,
    /// Distance from the baseline down to the bottom of the line box, in
    /// pixels (positive, unlike the OpenType convention).
    pub descent: f32,
    /// Extra leading between consecutive lines.
    pub line_gap: f32,
}

/// A supplier of glyph ids, metrics, kerning and coverage rasters.
///
/// Implemented by [`FontFace`] (OpenType outlines) and [`BitmapFace`]
/// (1-bit bitmaps).  The atlas and [`layout_line_into`] are generic over
/// it, so an external shaper only has to supply glyph ids.
pub trait GlyphSource {
    /// Stable identity mixed into every [`GlyphKey`] this source produces.
    /// Two sources sharing an atlas must answer different ids.
    fn font_id(&self) -> u32;

    /// Glyph index for a character; `0` when the font has no glyph for it.
    fn glyph_id(&self, c: char) -> u16;

    /// Line-box metrics at `px`.
    fn vertical_metrics(&self, px: f32) -> VerticalMetrics;

    /// Placement metrics of `glyph` at `px`.
    fn metrics(&self, glyph: u16, px: f32) -> GlyphMetrics;

    /// Horizontal advance of `glyph` at `px`, for measuring without a
    /// raster.  Defaults to [`Self::metrics`]`.advance`; sources whose full
    /// metrics are costly override it.
    #[inline]
    fn advance(&self, glyph: u16, px: f32) -> f32 {
        self.metrics(glyph, px).advance
    }

    /// Horizontal kerning adjustment to add to the pen between `left` and
    /// `right`, in pixels (`0.0` when the pair is unkerned).
    fn kern(&self, left: u16, right: u16, px: f32) -> f32;

    /// Rasterise `glyph` at `px` with the given phase into `out`.
    ///
    /// `out` addresses the top-left destination pixel; consecutive rows are
    /// `stride` samples apart.  Exactly
    /// [`GlyphMetrics::stored_extent`]`(phase_x, phase_y)` samples per row
    /// are written and nothing outside that box is touched.  Coverage is
    /// linear `0..=65535`.  An `out` too small for the box answers
    /// [`FontError::InvalidRaster`]; the returned metrics equal
    /// [`Self::metrics`] for the same glyph and size.
    fn rasterize_into(
        &self,
        glyph: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
        out: &mut [u16],
        stride: usize,
    ) -> Result<GlyphMetrics, FontError>;
}

impl<G: GlyphSource + ?Sized> GlyphSource for &G {
    #[inline]
    fn font_id(&self) -> u32 {
        (**self).font_id()
    }
    #[inline]
    fn glyph_id(&self, c: char) -> u16 {
        (**self).glyph_id(c)
    }
    #[inline]
    fn vertical_metrics(&self, px: f32) -> VerticalMetrics {
        (**self).vertical_metrics(px)
    }
    #[inline]
    fn metrics(&self, glyph: u16, px: f32) -> GlyphMetrics {
        (**self).metrics(glyph, px)
    }
    #[inline]
    fn advance(&self, glyph: u16, px: f32) -> f32 {
        (**self).advance(glyph, px)
    }
    #[inline]
    fn kern(&self, left: u16, right: u16, px: f32) -> f32 {
        (**self).kern(left, right, px)
    }
    #[inline]
    fn rasterize_into(
        &self,
        glyph: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
        out: &mut [u16],
        stride: usize,
    ) -> Result<GlyphMetrics, FontError> {
        (**self).rasterize_into(glyph, px, phase_x, phase_y, out, stride)
    }
}

/// Whether `px` is a usable pixel size.
#[inline]
pub(crate) fn valid_px(px: f32) -> bool {
    px.is_finite() && px > 0.0
}

/// `ceil` without `std`/`libm`.
#[inline]
pub(crate) fn ceil_f32(value: f32) -> f32 {
    let whole = value as i32 as f32;
    if whole < value { whole + 1.0 } else { whole }
}

/// Round-half-up to the nearest integer without `std`/`libm`.
#[inline]
pub(crate) fn round_f32(value: f32) -> f32 {
    let whole = value as i32 as f32;
    let whole = if whole > value { whole - 1.0 } else { whole };
    if value - whole >= 0.5 {
        whole + 1.0
    } else {
        whole
    }
}

/// Split a fractional position into a whole pixel and a 1/256 phase.
pub(crate) fn split_glyph_position(value: f32) -> Result<(i16, u8), FontError> {
    if !value.is_finite() || value < f32::from(i16::MIN) || value > f32::from(i16::MAX) {
        return Err(FontError::CoordinateOverflow);
    }
    let truncated = value as i32;
    let mut whole = if (truncated as f32) > value {
        truncated - 1
    } else {
        truncated
    };
    let mut phase = ((value - whole as f32) * 256.0 + 0.5) as u16;
    if phase == 256 {
        whole += 1;
        phase = 0;
    }
    let whole = i16::try_from(whole).map_err(|_| FontError::CoordinateOverflow)?;
    Ok((whole, phase as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_positions_preserve_fractional_and_negative_phase() {
        assert_eq!(split_glyph_position(12.25).unwrap(), (12, 64));
        assert_eq!(split_glyph_position(-0.25).unwrap(), (-1, 192));
        assert_eq!(split_glyph_position(3.999).unwrap(), (4, 0));
    }

    #[test]
    fn key_hash_is_fnv1a_over_the_key_bytes() {
        let key = GlyphKey {
            font: 1,
            glyph: 2,
            px_bits: 12.5f32.to_bits(),
            phase_x: 3,
            phase_y: 4,
        };
        let mut bytes = [0u8; 12];
        bytes[..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        bytes[6..10].copy_from_slice(&12.5f32.to_bits().to_le_bytes());
        bytes[10] = 3;
        bytes[11] = 4;
        assert_eq!(key.hash(), fnv1a(&bytes));
        assert_ne!(key.hash(), GlyphKey { phase_y: 5, ..key }.hash());
    }

    #[test]
    fn rounding_helpers_ceil_and_round_half_up() {
        for (value, ceil, round) in [
            (0.0, 0.0, 0.0),
            (0.4, 1.0, 0.0),
            (0.5, 1.0, 1.0),
            (1.5, 2.0, 2.0),
            (-0.4, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (-1.6, -1.0, -2.0),
            (12.0, 12.0, 12.0),
        ] {
            assert_eq!(ceil_f32(value), ceil, "ceil({value})");
            assert_eq!(round_f32(value), round, "round({value})");
        }
    }
}
