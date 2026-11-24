//! # embedded-bitmap-fonts
//!
//! A comprehensive collection of bitmap fonts for the [`embedded-graphics`] crate.
//!
//! This crate provides high-quality bitmap fonts from the [Tecate bitmap-fonts collection](https://github.com/Tecate/bitmap-fonts)
//! for use with embedded-graphics. Key features include:
//!
//! - **Pixel-doubling**: Scale fonts by 2x, 3x, or more with no additional flash cost
//! - **Multiple font families**: Tamzen, Cherry, Terminus, Spleen, and many more
//! - **Feature flags**: Include only the fonts you need to minimize binary size
//! - **No-std compatible**: Works on embedded systems without an allocator
//!
//! ## Usage
//!
//! ```rust,no_run
//! use embedded_bitmap_fonts::{BitmapFont, TextStyle};
//! # #[cfg(feature = "tamzen")]
//! use embedded_bitmap_fonts::tamzen::FONT_8x15;
//! use embedded_graphics::{pixelcolor::BinaryColor, prelude::*, text::Text};
//!
//! # struct Display;
//! # impl OriginDimensions for Display { fn size(&self) -> Size { Size::zero() } }
//! # impl DrawTarget for Display {
//! #     type Color = BinaryColor;
//! #     type Error = core::convert::Infallible;
//! #     fn draw_iter<I>(&mut self, _: I) -> Result<(), Self::Error>
//! #     where I: IntoIterator<Item = Pixel<BinaryColor>> { Ok(()) }
//! # }
//! # fn main() -> Result<(), core::convert::Infallible> {
//! # let mut display = Display;
//! # #[cfg(feature = "tamzen")]
//! # {
//! // Draw text with a bitmap font
//! let text = Text::new(
//!     "Hello World!",
//!     Point::zero(),
//!     TextStyle::new(&FONT_8x15, BinaryColor::On)
//! );
//! text.draw(&mut display)?;
//!
//! // Use pixel-doubled font for larger text (same flash cost!)
//! let doubled = FONT_8x15.pixel_double();
//! let large_text = Text::new(
//!     "Big Text!",
//!     Point::new(0, 20),
//!     TextStyle::new(&doubled, BinaryColor::On)
//! );
//! large_text.draw(&mut display)?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ## Available Font Families
//!
//! Enable fonts using feature flags in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! embedded-bitmap-fonts = { version = "0.1", features = ["tamzen", "cherry"] }
//! ```
//!
//! Or enable all fonts:
//!
//! ```toml
//! [dependencies]
//! embedded-bitmap-fonts = { version = "0.1", features = ["all-fonts"] }
//! ```
//!
//! [`embedded-graphics`]: embedded_graphics

#![cfg_attr(not(test), no_std)]
#![warn(rust_2018_idioms)]

use core::num::NonZeroU8;
use embedded_graphics::{
    Pixel,
    draw_target::DrawTarget,
    geometry::{Dimensions, OriginDimensions, Point, Size},
    image::{ImageDrawable, ImageRaw},
    mono_font::mapping::GlyphMapping,
    pixelcolor::BinaryColor,
    primitives::Rectangle,
    text::{
        Baseline,
        renderer::{TextMetrics, TextRenderer},
    },
};

// Font modules - conditionally compiled based on features
#[cfg(feature = "artwiz")]
pub mod artwiz;

#[cfg(feature = "bitocra")]
pub mod bitocra;

#[cfg(feature = "cherry")]
pub mod cherry;

#[cfg(feature = "creep")]
pub mod creep;

#[cfg(feature = "ctrld")]
pub mod ctrld;

#[cfg(feature = "dina")]
pub mod dina;

#[cfg(feature = "dylex")]
pub mod dylex;

#[cfg(feature = "envypn")]
pub mod envypn;

#[cfg(feature = "gohufont")]
pub mod gohufont;

#[cfg(feature = "gomme")]
pub mod gomme;

#[cfg(feature = "haxor")]
pub mod haxor;

#[cfg(feature = "jmk")]
pub mod jmk;

#[cfg(feature = "kakwa")]
pub mod kakwa;

#[cfg(feature = "knxt")]
pub mod knxt;

#[cfg(feature = "lokaltog")]
pub mod lokaltog;

#[cfg(feature = "mplus")]
pub mod mplus;

#[cfg(feature = "orp")]
pub mod orp;

#[cfg(feature = "peep")]
pub mod peep;

#[cfg(feature = "phallus")]
pub mod phallus;

#[cfg(feature = "progsole")]
pub mod progsole;

#[cfg(feature = "scientifica")]
pub mod scientifica;

#[cfg(feature = "spleen")]
pub mod spleen;

#[cfg(feature = "tamzen")]
pub mod tamzen;

#[cfg(feature = "terminus")]
pub mod terminus;

#[cfg(feature = "tewi")]
pub mod tewi;

#[cfg(feature = "trisk")]
pub mod trisk;

#[cfg(feature = "xbmicons")]
pub mod xbmicons;

/// Constant 1 as NonZeroU8 for initialization
const ONE: NonZeroU8 = match NonZeroU8::new(1) {
    Some(one) => one,
    None => unreachable!(),
};

/// Stores the font bitmap and metadata for rendering.
///
/// Each `BitmapFont` contains the raw bitmap data for all glyphs arranged in a
/// sprite sheet, along with information about glyph dimensions and character mapping.
///
/// # Pixel Doubling
///
/// The key feature of this crate is pixel-doubling support. You can call
/// [`pixel_double()`](BitmapFont::pixel_double) to get a font that renders at 2x size
/// without any additional memory cost. This works by drawing each pixel as a 2x2 block.
///
/// ```rust
/// # #[cfg(feature = "tamzen")]
/// # {
/// use embedded_bitmap_fonts::tamzen::FONT_8x15;
///
/// // Original 8x15 font
/// let small = &FONT_8x15;
/// assert_eq!(small.width(), 8);
/// assert_eq!(small.height(), 15);
///
/// // Pixel-doubled 16x30 font (same bitmap data!)
/// let large = FONT_8x15.pixel_double();
/// assert_eq!(large.width(), 16);
/// assert_eq!(large.height(), 30);
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct BitmapFont<'a> {
    /// The raw bitmap data for the font sprite sheet.
    pub bitmap: ImageRaw<'a, BinaryColor>,

    /// Maps characters to glyph indices in the sprite sheet.
    pub glyph_mapping: &'a dyn GlyphMapping,

    /// The size of each glyph in the raw bitmap (before pixel multiplication).
    pub size: Size,

    /// Pixel multiplier for scaling. 1 = normal, 2 = double, etc.
    pub pixels: NonZeroU8,
}

impl<'a> BitmapFont<'a> {
    /// Creates a new BitmapFont from raw bitmap data.
    ///
    /// This is typically called by the generated font modules, not directly by users.
    #[inline]
    pub const fn new(
        bitmap: ImageRaw<'a, BinaryColor>,
        glyph_mapping: &'a dyn GlyphMapping,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            bitmap,
            glyph_mapping,
            size: Size::new(width, height),
            pixels: ONE,
        }
    }

    /// Returns the width of each character in pixels (after pixel multiplication).
    #[inline]
    pub const fn width(&self) -> u32 {
        self.size.width * self.pixels.get() as u32
    }

    /// Returns the height of each character in pixels (after pixel multiplication).
    #[inline]
    pub const fn height(&self) -> u32 {
        self.size.height * self.pixels.get() as u32
    }

    /// Returns the base (unmultiplied) width of each character.
    #[inline]
    pub const fn base_width(&self) -> u32 {
        self.size.width
    }

    /// Returns the base (unmultiplied) height of each character.
    #[inline]
    pub const fn base_height(&self) -> u32 {
        self.size.height
    }

    /// Returns the current pixel multiplier.
    #[inline]
    pub const fn pixel_multiplier(&self) -> u8 {
        self.pixels.get()
    }

    /// Draw a single glyph at the specified position.
    pub fn draw_glyph<D>(
        &self,
        idx: u32,
        target: &mut D,
        color: BinaryColor,
        pos: Point,
    ) -> Result<(), D::Error>
    where
        D: DrawTarget<Color = BinaryColor>,
    {
        let bitmap_size = self.bitmap.size();
        let chars_per_row = bitmap_size.width / self.size.width;
        let row = idx / chars_per_row;

        // Calculate position in the sprite sheet
        let char_x = (idx - (row * chars_per_row)) * self.size.width;
        let char_y = row * self.size.height;
        let area = Rectangle::new(Point::new(char_x as _, char_y as _), self.size);

        // Draw with pixel multiplication
        let mut pixel_target = PixelMultiplyDrawTarget {
            target,
            color,
            offset: pos,
            pixels: self.pixels,
        };
        self.bitmap.draw_sub_image(&mut pixel_target, &area)?;

        Ok(())
    }

    /// Returns a pixel-doubled version of this font.
    ///
    /// The returned font renders at 2x the original size by drawing each pixel
    /// as a 2x2 block. This incurs no additional memory cost since it uses the
    /// same underlying bitmap data.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #[cfg(feature = "tamzen")]
    /// # {
    /// use embedded_bitmap_fonts::tamzen::FONT_8x15;
    ///
    /// let doubled = FONT_8x15.pixel_double();
    /// assert_eq!(doubled.width(), 16);
    /// assert_eq!(doubled.height(), 30);
    /// # }
    /// ```
    #[must_use]
    pub const fn pixel_double(self) -> Self {
        self.pixel_multiply(2)
    }

    /// Returns a pixel-tripled version of this font.
    ///
    /// The returned font renders at 3x the original size.
    #[must_use]
    pub const fn pixel_triple(self) -> Self {
        self.pixel_multiply(3)
    }

    /// Returns a version of this font with custom pixel multiplication.
    ///
    /// # Panics
    ///
    /// Panics if `multiplier` is 0.
    #[must_use]
    pub const fn pixel_multiply(mut self, multiplier: u8) -> Self {
        // Multiply the existing multiplier
        let new_multiplier = self.pixels.get() * multiplier;
        self.pixels = match NonZeroU8::new(new_multiplier) {
            Some(px) => px,
            None => panic!("pixel multiplier cannot be zero"),
        };
        self
    }
}

/// Text style for rendering with a [`BitmapFont`].
///
/// This is the equivalent of [`MonoTextStyle`](embedded_graphics::mono_font::MonoTextStyle)
/// for bitmap fonts.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "tamzen")]
/// # {
/// use embedded_bitmap_fonts::{TextStyle, tamzen::FONT_8x15};
/// use embedded_graphics::{pixelcolor::BinaryColor, prelude::*, text::Text};
///
/// # struct Display;
/// # impl OriginDimensions for Display { fn size(&self) -> Size { Size::zero() } }
/// # impl DrawTarget for Display {
/// #     type Color = BinaryColor;
/// #     type Error = core::convert::Infallible;
/// #     fn draw_iter<I>(&mut self, _: I) -> Result<(), Self::Error>
/// #     where I: IntoIterator<Item = Pixel<BinaryColor>> { Ok(()) }
/// # }
/// # fn main() -> Result<(), core::convert::Infallible> {
/// # let mut display = Display;
/// let style = TextStyle::new(&FONT_8x15, BinaryColor::On);
/// let text = Text::new("Hello!", Point::zero(), style);
/// text.draw(&mut display)?;
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct TextStyle<'a> {
    /// The font to use for rendering.
    pub font: &'a BitmapFont<'a>,
    /// The foreground color for text pixels.
    pub color: BinaryColor,
}

impl<'a> TextStyle<'a> {
    /// Creates a new text style with the given font and color.
    #[inline]
    pub const fn new(font: &'a BitmapFont<'a>, color: BinaryColor) -> Self {
        Self { font, color }
    }
}

impl TextRenderer for TextStyle<'_> {
    type Color = BinaryColor;

    fn draw_string<D>(
        &self,
        text: &str,
        mut pos: Point,
        _baseline: Baseline,
        target: &mut D,
    ) -> Result<Point, D::Error>
    where
        D: DrawTarget<Color = Self::Color>,
    {
        for c in text.chars() {
            let glyph_idx = self.font.glyph_mapping.index(c) as u32;
            self.font.draw_glyph(glyph_idx, target, self.color, pos)?;
            pos += Size::new(self.font.width(), 0);
        }
        Ok(pos)
    }

    fn draw_whitespace<D>(
        &self,
        width: u32,
        pos: Point,
        _baseline: Baseline,
        _target: &mut D,
    ) -> Result<Point, D::Error>
    where
        D: DrawTarget<Color = Self::Color>,
    {
        // No background drawing, just advance position
        Ok(pos + Size::new(width * self.font.width(), 0))
    }

    fn measure_string(&self, text: &str, pos: Point, _baseline: Baseline) -> TextMetrics {
        let width = self.font.width() * text.len() as u32;
        let height = self.font.height();
        TextMetrics {
            bounding_box: Rectangle::new(pos, Size::new(width, height)),
            next_position: pos + Size::new(width, 0),
        }
    }

    fn line_height(&self) -> u32 {
        self.font.height()
    }
}

/// Internal draw target that handles pixel multiplication.
///
/// This wraps another draw target and multiplies each pixel by the specified
/// factor, drawing NxN blocks for each source pixel.
struct PixelMultiplyDrawTarget<'a, D: DrawTarget<Color = BinaryColor>> {
    target: &'a mut D,
    color: BinaryColor,
    offset: Point,
    pixels: NonZeroU8,
}

impl<D> Dimensions for PixelMultiplyDrawTarget<'_, D>
where
    D: DrawTarget<Color = BinaryColor>,
{
    fn bounding_box(&self) -> Rectangle {
        let mut bb = self.target.bounding_box();
        bb.top_left -= self.offset;
        bb.top_left /= self.pixels.get().into();
        bb.size /= self.pixels.get().into();
        bb
    }
}

impl<D> DrawTarget for PixelMultiplyDrawTarget<'_, D>
where
    D: DrawTarget<Color = BinaryColor>,
{
    type Color = BinaryColor;
    type Error = D::Error;

    fn draw_iter<I>(&mut self, pixel_iter: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<BinaryColor>>,
    {
        let color = self.color;
        let offset = self.offset;
        let multiplier = self.pixels.get();

        self.target.draw_iter(
            pixel_iter
                .into_iter()
                .filter(|pixel| pixel.1 == BinaryColor::On)
                .flat_map(|pixel| {
                    PixelMultiplyIterator::new(pixel, multiplier)
                        .map(move |p| Pixel(p.0 + offset, color))
                }),
        )
    }
}

/// Iterator that expands a single pixel into an NxN block.
struct PixelMultiplyIterator {
    color: BinaryColor,
    base_pos: Point,
    x: u8,
    y: u8,
    multiplier: u8,
}

impl PixelMultiplyIterator {
    fn new(pixel: Pixel<BinaryColor>, multiplier: u8) -> Self {
        Self {
            color: pixel.1,
            base_pos: pixel.0 * multiplier as i32,
            x: 0,
            y: 0,
            multiplier,
        }
    }
}

impl Iterator for PixelMultiplyIterator {
    type Item = Pixel<BinaryColor>;

    fn next(&mut self) -> Option<Pixel<BinaryColor>> {
        if self.y >= self.multiplier {
            return None;
        }

        let pixel = Pixel(
            Point::new(
                self.base_pos.x + self.x as i32,
                self.base_pos.y + self.y as i32,
            ),
            self.color,
        );

        self.x += 1;
        if self.x >= self.multiplier {
            self.x = 0;
            self.y += 1;
        }

        Some(pixel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining =
            (self.multiplier - self.y) as usize * self.multiplier as usize - self.x as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PixelMultiplyIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(x: i32, y: i32) -> Pixel<BinaryColor> {
        Pixel(Point::new(x, y), BinaryColor::On)
    }

    #[test]
    fn pixel_multiply_iterator_1x() {
        let pixels: Vec<_> = PixelMultiplyIterator::new(px(0, 0), 1).collect();
        assert_eq!(pixels, vec![px(0, 0)]);

        let pixels: Vec<_> = PixelMultiplyIterator::new(px(5, 3), 1).collect();
        assert_eq!(pixels, vec![px(5, 3)]);
    }

    #[test]
    fn pixel_multiply_iterator_2x() {
        let pixels: Vec<_> = PixelMultiplyIterator::new(px(0, 0), 2).collect();
        assert_eq!(pixels, vec![px(0, 0), px(1, 0), px(0, 1), px(1, 1)]);

        let pixels: Vec<_> = PixelMultiplyIterator::new(px(1, 2), 2).collect();
        assert_eq!(pixels, vec![px(2, 4), px(3, 4), px(2, 5), px(3, 5)]);
    }

    #[test]
    fn pixel_multiply_iterator_3x() {
        let pixels: Vec<_> = PixelMultiplyIterator::new(px(0, 0), 3).collect();
        assert_eq!(
            pixels,
            vec![
                px(0, 0),
                px(1, 0),
                px(2, 0),
                px(0, 1),
                px(1, 1),
                px(2, 1),
                px(0, 2),
                px(1, 2),
                px(2, 2)
            ]
        );
    }

    #[test]
    fn pixel_multiply_iterator_exact_size() {
        let iter = PixelMultiplyIterator::new(px(0, 0), 3);
        assert_eq!(iter.len(), 9);

        let mut iter = PixelMultiplyIterator::new(px(0, 0), 2);
        assert_eq!(iter.len(), 4);
        iter.next();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
    }
}
