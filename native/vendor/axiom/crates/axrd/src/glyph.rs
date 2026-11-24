//! Glyph run types for text rendering.
//!
//! These types represent the *output* of a text shaping pipeline (e.g.,
//! HarfBuzz). The compositor never sees codepoints or font tables — only
//! pre-positioned glyphs ready to blit from an atlas.
//!
//! # Pipeline
//!
//! ```text
//! Unicode string + font
//!       ↓
//!   [text shaper]          ← external (HarfBuzz, etc.)
//!       ↓
//!   glyph IDs + positions
//!       ↓
//!   [atlas lookup]         ← maps glyph ID → alpha atlas (ax, ay, aw, ah)
//!       ↓
//!   PlacedGlyph[]          ← this module's type
//!       ↓
//!   [axrd compositor]      ← Bind::TextRun samples the atlas coverage
//! ```

pub use axnn::PlacedGlyph;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translated_phase_carries_into_integer_position() {
        let glyph = PlacedGlyph {
            ax: 0,
            ay: 0,
            aw: 1,
            ah: 1,
            dx: 4,
            dy: -2,
            subpixel_x: 224,
            subpixel_y: 32,
        }
        .translated_subpixel(64, 96)
        .unwrap();
        assert_eq!((glyph.dx, glyph.subpixel_x), (5, 32));
        assert_eq!((glyph.dy, glyph.subpixel_y), (-2, 128));
    }
}
