//! OpenType glyph source: a parsed `femtofont` face, supersampled and
//! area-reconstructed at the final subpixel phase.

use alloc::vec::Vec;

use femtofont::{Font, FontSettings, Metrics};
use ttf_parser::GlyphId;

use super::{
    AtlasStorage, FontError, GlyphAtlas, GlyphMetrics, GlyphSource, LineSpec, VerticalMetrics,
    ceil_f32, fnv1a, layout_line_into, valid_px,
};
use crate::PlacedGlyph;

/// Default number of compiled outlines cached per parsed face.
pub const DEFAULT_OUTLINE_CACHE_ENTRIES: usize = 128;
/// Default glyph raster scale. Coverage is rendered at this multiple in both
/// axes and box-filtered into the atlas.
pub const DEFAULT_SUPERSAMPLING: u8 = 8;

/// A parsed scalable font backed by caller-owned font bytes.
///
/// Parsing happens once.  Glyphs reach an atlas through [`GlyphSource`]:
/// the outline is rasterised at [`Self::supersampling`]× the requested size
/// and box-filtered straight into the atlas plane at the requested phase.
/// The face's [`GlyphSource::font_id`] defaults to a hash of the byte
/// slice's address, length, collection index and weight; embeddings that
/// want stable ids across reloads set one with [`Self::with_font_id`].
#[derive(Clone, Debug)]
pub struct FontFace<'a> {
    inner: Font<'a>,
    font_id: u32,
    supersampling: u8,
}

impl<'a> FontFace<'a> {
    /// Parse the first face in a TTF, OTF, or font collection.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FontError> {
        Self::parse_with_settings(
            bytes,
            FontSettings {
                cachesize: DEFAULT_OUTLINE_CACHE_ENTRIES,
                ..FontSettings::default()
            },
        )
    }

    /// Parse a font with explicit FemtoFont geometry-cache settings.
    pub fn parse_with_settings(bytes: &'a [u8], settings: FontSettings) -> Result<Self, FontError> {
        let font_id = derive_font_id(bytes, settings.collection_index, 0);
        Font::from_bytes(bytes, settings)
            .map(|inner| Self {
                inner,
                font_id,
                supersampling: DEFAULT_SUPERSAMPLING,
            })
            .map_err(FontError::InvalidFont)
    }

    /// Parse a font with an explicit `wght` variation value.
    ///
    /// Static fonts simply ignore this request. For variable fonts it selects
    /// the weight before any outline reaches the atlas; this makes a large
    /// display treatment and compact UI copy able to share the same source
    /// file without sharing incorrect cached glyphs.
    pub fn parse_with_weight(bytes: &'a [u8], weight: f32) -> Result<Self, FontError> {
        Self::parse_with_weight_and_settings(
            bytes,
            weight,
            FontSettings {
                cachesize: DEFAULT_OUTLINE_CACHE_ENTRIES,
                ..FontSettings::default()
            },
        )
    }

    /// Parse a font with explicit weight and geometry-cache settings.
    pub fn parse_with_weight_and_settings(
        bytes: &'a [u8],
        weight: f32,
        settings: FontSettings,
    ) -> Result<Self, FontError> {
        if !weight.is_finite() {
            return Err(FontError::InvalidFont("font weight must be finite"));
        }
        let font_id = derive_font_id(bytes, settings.collection_index, weight.to_bits());
        Font::from_bytes_and_weight(bytes, weight, settings)
            .map(|inner| Self {
                inner,
                font_id,
                supersampling: DEFAULT_SUPERSAMPLING,
            })
            .map_err(FontError::InvalidFont)
    }

    /// Assign the id mixed into this face's glyph keys.
    pub fn with_font_id(mut self, font_id: u32) -> Self {
        self.font_id = font_id;
        self
    }

    /// Set the raster supersampling scale (`1..=8`).
    ///
    /// One uses FemtoFont's native pixel coverage.  Two through eight render
    /// a denser temporary bitmap and area-reconstruct it directly at the
    /// final subpixel phase; eight is the default.
    pub fn with_supersampling(mut self, supersampling: u8) -> Result<Self, FontError> {
        if !(1..=8).contains(&supersampling) {
            return Err(FontError::InvalidSupersampling);
        }
        self.supersampling = supersampling;
        Ok(self)
    }

    /// Raster samples per output pixel along each axis.
    #[inline]
    pub const fn supersampling(&self) -> u8 {
        self.supersampling
    }

    /// Look up a glyph ID in this face.
    #[inline]
    pub fn glyph_id(&self, character: char) -> u16 {
        self.inner.lookup_glyph_index(character)
    }

    /// The parsed FemtoFont face.
    #[inline]
    pub fn inner(&self) -> &Font<'a> {
        &self.inner
    }

    /// Rasterize and position one ordinary left-to-right text run.
    ///
    /// The returned offsets are relative to a [`crate::Bind::TextRun`] node's
    /// top-left corner.  Newlines are honored, but no bounds or wrapping are
    /// applied; use [`layout_line_into`] for a bounded single line.
    pub fn layout_line<S: AtlasStorage>(
        &self,
        atlas: &mut GlyphAtlas<S>,
        text: &str,
        px: f32,
    ) -> Result<TextRun, FontError> {
        self.layout_line_at(atlas, text, px, 0, 0)
    }

    /// Rasterize and position a text run at its final physical subpixel phase.
    ///
    /// The phase is the fractional part of the run node's physical origin.
    /// Folding it into atlas construction performs exactly one area
    /// reconstruction from the supersampled outline. The returned glyphs
    /// therefore have zero compositor phase and do not need a later bilinear
    /// translation.
    ///
    /// This is the allocating convenience over [`layout_line_into`]: one
    /// `PlacedGlyph` per character is reserved up front and each `\n`
    /// starts a new line one line-height lower.  Rasterisation uses the
    /// atlas's [`GlyphAtlas::supersampling`].
    pub fn layout_line_at<S: AtlasStorage>(
        &self,
        atlas: &mut GlyphAtlas<S>,
        text: &str,
        px: f32,
        origin_phase_x: u8,
        origin_phase_y: u8,
    ) -> Result<TextRun, FontError> {
        if !valid_px(px) {
            return Ok(TextRun { glyphs: Vec::new() });
        }
        let count = text.chars().count();
        let mut glyphs = Vec::new();
        glyphs
            .try_reserve_exact(count)
            .map_err(|_| FontError::Allocation)?;
        glyphs.resize(
            count,
            PlacedGlyph {
                ax: 0,
                ay: 0,
                aw: 0,
                ah: 0,
                dx: 0,
                dy: 0,
                subpixel_x: 0,
                subpixel_y: 0,
            },
        );

        let source = self.with_atlas_supersampling(atlas.supersampling());
        let line = self.inner.horizontal_line_metrics(px);
        let line_height = ceil_f32(line.new_line_size);
        let spec = LineSpec::new(px).with_phase(origin_phase_x, origin_phase_y);
        let mut filled = 0usize;
        let mut baseline_shift = 0.0f32;
        for line_text in text.split('\n') {
            let metrics = layout_line_into(&source, atlas, line_text, spec, &mut glyphs[filled..])?;
            let placed = usize::from(metrics.glyphs);
            if baseline_shift != 0.0 {
                let shift = baseline_shift as i32;
                for glyph in &mut glyphs[filled..filled + placed] {
                    glyph.dy = i16::try_from(i32::from(glyph.dy) + shift)
                        .map_err(|_| FontError::CoordinateOverflow)?;
                }
            }
            filled += placed;
            baseline_shift += line_height;
        }
        glyphs.truncate(filled);
        Ok(TextRun { glyphs })
    }

    /// This face rasterising at the atlas's supersampling instead of its
    /// own — the legacy `FontFace`-specific helpers' contract.
    pub(crate) fn with_atlas_supersampling(&self, supersampling: u8) -> Supersampled<'_, 'a> {
        Supersampled {
            face: self,
            supersampling,
        }
    }

    fn glyph_metrics(&self, glyph: u16, px: f32) -> GlyphMetrics {
        if !valid_px(px) {
            return GlyphMetrics::default();
        }
        from_femto(self.inner.metrics_indexed(glyph, px))
    }

    /// Advance straight from `hmtx` (and `HVAR` for a variable face): the
    /// same value `femtofont` folds into its metrics, without cloning the
    /// cached outline the way `metrics_indexed` does.
    fn glyph_advance(&self, glyph: u16, px: f32) -> f32 {
        if !valid_px(px) {
            return 0.0;
        }
        let units = self
            .inner
            .face
            .glyph_hor_advance(GlyphId(glyph))
            .map_or(0.0, f32::from);
        units * self.inner.scale_factor(px)
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_supersampled_into(
        &self,
        supersampling: u8,
        glyph: u16,
        px: f32,
        phase_x: u8,
        phase_y: u8,
        out: &mut [u16],
        stride: usize,
    ) -> Result<GlyphMetrics, FontError> {
        if !valid_px(px) {
            return Err(FontError::InvalidPixelSize);
        }
        let metrics = self.inner.metrics_indexed(glyph, px);
        let result = from_femto(metrics);
        if metrics.width == 0 || metrics.height == 0 {
            return Ok(result);
        }
        if supersampling == 1 {
            let (raster_metrics, native) = self.inner.rasterize_indexed(glyph, px);
            if raster_metrics != metrics {
                return Err(FontError::InvalidRaster);
            }
            downsample_coverage_phase(metrics, metrics, &native, 1, phase_x, phase_y, out, stride)?;
        } else {
            let (high_metrics, high) = self
                .inner
                .rasterize_indexed(glyph, px * f32::from(supersampling));
            downsample_coverage_phase(
                metrics,
                high_metrics,
                &high,
                supersampling,
                phase_x,
                phase_y,
                out,
                stride,
            )?;
        }
        Ok(result)
    }
}

/// A [`FontFace`] viewed at a different supersampling scale.
pub(crate) struct Supersampled<'f, 'a> {
    face: &'f FontFace<'a>,
    supersampling: u8,
}

impl GlyphSource for Supersampled<'_, '_> {
    #[inline]
    fn font_id(&self) -> u32 {
        self.face.font_id
    }
    #[inline]
    fn glyph_id(&self, c: char) -> u16 {
        self.face.glyph_id(c)
    }
    #[inline]
    fn vertical_metrics(&self, px: f32) -> VerticalMetrics {
        self.face.vertical_metrics(px)
    }
    #[inline]
    fn metrics(&self, glyph: u16, px: f32) -> GlyphMetrics {
        self.face.glyph_metrics(glyph, px)
    }
    #[inline]
    fn advance(&self, glyph: u16, px: f32) -> f32 {
        self.face.glyph_advance(glyph, px)
    }
    #[inline]
    fn kern(&self, left: u16, right: u16, px: f32) -> f32 {
        self.face.kern(left, right, px)
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
        self.face.rasterize_supersampled_into(
            self.supersampling,
            glyph,
            px,
            phase_x,
            phase_y,
            out,
            stride,
        )
    }
}

impl GlyphSource for FontFace<'_> {
    #[inline]
    fn font_id(&self) -> u32 {
        self.font_id
    }

    #[inline]
    fn glyph_id(&self, c: char) -> u16 {
        self.inner.lookup_glyph_index(c)
    }

    fn vertical_metrics(&self, px: f32) -> VerticalMetrics {
        if !valid_px(px) {
            return VerticalMetrics::default();
        }
        let line = self.inner.horizontal_line_metrics(px);
        VerticalMetrics {
            ascent: line.ascent,
            descent: -line.descent,
            line_gap: line.line_gap,
        }
    }

    #[inline]
    fn metrics(&self, glyph: u16, px: f32) -> GlyphMetrics {
        self.glyph_metrics(glyph, px)
    }

    #[inline]
    fn advance(&self, glyph: u16, px: f32) -> f32 {
        self.glyph_advance(glyph, px)
    }

    #[inline]
    fn kern(&self, left: u16, right: u16, px: f32) -> f32 {
        self.inner
            .horizontal_kern_indexed(left, right, px)
            .unwrap_or(0.0)
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
        self.rasterize_supersampled_into(
            self.supersampling,
            glyph,
            px,
            phase_x,
            phase_y,
            out,
            stride,
        )
    }
}

fn derive_font_id(bytes: &[u8], collection_index: u32, variation_key: u32) -> u32 {
    let mut seed = [0u8; 24];
    seed[..8].copy_from_slice(&(bytes.as_ptr() as usize as u64).to_le_bytes());
    seed[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    seed[16..20].copy_from_slice(&collection_index.to_le_bytes());
    seed[20..].copy_from_slice(&variation_key.to_le_bytes());
    let hash = fnv1a(&seed);
    (hash as u32) ^ ((hash >> 32) as u32)
}

fn from_femto(metrics: Metrics) -> GlyphMetrics {
    GlyphMetrics {
        xmin: metrics.xmin,
        ymin: metrics.ymin,
        // The atlas re-checks against `u16` on insert; saturating here
        // keeps `metrics()` total.
        width: u16::try_from(metrics.width).unwrap_or(u16::MAX),
        height: u16::try_from(metrics.height).unwrap_or(u16::MAX),
        advance: metrics.advance_width,
    }
}

/// A reusable run of compositor-ready glyph positions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextRun {
    glyphs: Vec<PlacedGlyph>,
}

impl TextRun {
    /// Glyphs ready to append to the flat glyph slice bound to the raster program.
    #[inline]
    pub fn glyphs(&self) -> &[PlacedGlyph] {
        &self.glyphs
    }

    /// Number of positioned glyphs, including zero-area whitespace glyphs.
    #[inline]
    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    /// Whether the run contains no glyphs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }
}

/// Area-reconstruct a supersampled glyph directly at its final pixel phase
/// into `out` (rows `stride` apart).
///
/// Source sample and destination pixel boundaries use fixed 1/256 high-sample
/// units. This keeps the result deterministic in `no_std`, preserves all
/// partial boundary weights, and avoids translating an already filtered mask.
#[allow(clippy::too_many_arguments)]
fn downsample_coverage_phase(
    target: Metrics,
    high: Metrics,
    high_coverage: &[u8],
    scale: u8,
    phase_x: u8,
    phase_y: u8,
    out: &mut [u16],
    stride: usize,
) -> Result<(), FontError> {
    let high_expected = high
        .width
        .checked_mul(high.height)
        .ok_or(FontError::InvalidRaster)?;
    if high_coverage.len() != high_expected {
        return Err(FontError::InvalidRaster);
    }
    let output_width = target
        .width
        .checked_add(usize::from(phase_x != 0))
        .ok_or(FontError::InvalidRaster)?;
    let output_height = target
        .height
        .checked_add(usize::from(phase_y != 0))
        .ok_or(FontError::InvalidRaster)?;
    if output_width > stride {
        return Err(FontError::InvalidRaster);
    }
    let needed = (output_height - 1)
        .checked_mul(stride)
        .and_then(|rows| rows.checked_add(output_width))
        .ok_or(FontError::InvalidRaster)?;
    if out.len() < needed {
        return Err(FontError::InvalidRaster);
    }

    let scale = i64::from(scale);
    let destination_span = scale * 256;
    let denominator = (scale * scale * 256 * 256) as u64;
    // Font bearings are Y-up and ymin names the bitmap's BOTTOM, whereas
    // coverage rows are top-down. Align their top edges around the common
    // baseline. Bottom-edge alignment shifts/clips glyphs differently when
    // the native and supersampled bounds round by different amounts.
    let target_top = i64::from(target.ymin) + target.height as i64;
    let high_top = i64::from(high.ymin) + high.height as i64;
    for y in 0..output_height {
        let start_y =
            (high_top - target_top * scale + y as i64 * scale) * 256 - i64::from(phase_y) * scale;
        let end_y = start_y + destination_span;
        let first_y = start_y.div_euclid(256);
        let last_y = (end_y - 1).div_euclid(256);
        let row = &mut out[y * stride..y * stride + output_width];
        for (x, dst) in row.iter_mut().enumerate() {
            let start_x = (i64::from(target.xmin) + x as i64) * scale * 256
                - i64::from(high.xmin) * 256
                - i64::from(phase_x) * scale;
            let end_x = start_x + destination_span;
            let first_x = start_x.div_euclid(256);
            let last_x = (end_x - 1).div_euclid(256);
            let mut sum = 0_u64;
            for source_y in first_y..=last_y {
                if source_y < 0 || source_y >= high.height as i64 {
                    continue;
                }
                let sample_y0 = source_y * 256;
                let overlap_y = (end_y.min(sample_y0 + 256) - start_y.max(sample_y0)).max(0);
                for source_x in first_x..=last_x {
                    if source_x < 0 || source_x >= high.width as i64 {
                        continue;
                    }
                    let sample_x0 = source_x * 256;
                    let overlap_x = (end_x.min(sample_x0 + 256) - start_x.max(sample_x0)).max(0);
                    let alpha = u64::from(
                        high_coverage[source_y as usize * high.width + source_x as usize],
                    );
                    sum = sum.saturating_add(
                        alpha
                            .saturating_mul(overlap_x as u64)
                            .saturating_mul(overlap_y as u64),
                    );
                }
            }
            *dst = ((sum.saturating_mul(257) + denominator / 2) / denominator) as u16;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::{FontAtlas, GlyphKey};

    fn downsample(
        target: Metrics,
        high: Metrics,
        high_coverage: &[u8],
        scale: u8,
        phase_x: u8,
        phase_y: u8,
    ) -> Result<Vec<u16>, FontError> {
        let width = target.width + usize::from(phase_x != 0);
        let height = target.height + usize::from(phase_y != 0);
        let mut out = alloc::vec![0u16; width * height];
        downsample_coverage_phase(
            target,
            high,
            high_coverage,
            scale,
            phase_x,
            phase_y,
            &mut out,
            width,
        )?;
        Ok(out)
    }

    #[test]
    fn invalid_font_is_rejected() {
        assert!(matches!(
            FontFace::parse(&[]),
            Err(FontError::InvalidFont(_))
        ));
    }

    #[test]
    fn variable_weight_font_rasterizes_into_the_atlas() {
        let face =
            FontFace::parse_with_weight(include_bytes!("../../assets/fonts/Fraunces.ttf"), 900.0)
                .unwrap();
        let mut atlas = FontAtlas::new(512, 256).unwrap();

        let run = face.layout_line(&mut atlas, "21:19", 96.0).unwrap();

        assert!(!run.is_empty());
        assert!(atlas.glyph_count() >= 4);
        assert!(atlas.pixels().iter().any(|&pixel| pixel != 0));
    }

    #[test]
    fn supersampling_box_filters_coverage() {
        let target = Metrics {
            width: 1,
            height: 1,
            ..Metrics::default()
        };
        let high = Metrics {
            width: 4,
            height: 4,
            ..Metrics::default()
        };
        let high_coverage = [
            255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0,
        ];

        assert_eq!(
            downsample(target, high, &high_coverage, 4, 0, 0).unwrap(),
            [32_768]
        );
        assert_eq!(FontAtlas::new(8, 8).unwrap().supersampling(), 8);
        assert!(matches!(
            FontAtlas::with_supersampling(8, 8, 1, 0),
            Err(FontError::InvalidSupersampling)
        ));
        let face = FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap();
        assert!(matches!(
            face.clone().with_supersampling(9),
            Err(FontError::InvalidSupersampling)
        ));
        assert_eq!(face.with_supersampling(2).unwrap().supersampling(), 2);
    }

    #[test]
    fn phase_aware_reconstruction_filters_only_once() {
        let metrics = Metrics {
            width: 1,
            height: 1,
            ..Metrics::default()
        };
        assert_eq!(
            downsample(metrics, metrics, &[255], 1, 128, 0).unwrap(),
            [32_768, 32_768]
        );
    }

    #[test]
    fn downsample_rejects_a_short_destination() {
        let metrics = Metrics {
            width: 2,
            height: 2,
            ..Metrics::default()
        };
        let mut out = [0u16; 3];
        assert_eq!(
            downsample_coverage_phase(metrics, metrics, &[255; 4], 1, 0, 0, &mut out, 2),
            Err(FontError::InvalidRaster)
        );
        assert_eq!(
            downsample_coverage_phase(metrics, metrics, &[255; 4], 1, 0, 0, &mut out, 1),
            Err(FontError::InvalidRaster)
        );
    }

    #[test]
    fn supersampling_aligns_top_down_rows_using_top_bearings() {
        // A 1.25 px tall rectangle lies half a pixel below the target's top.
        // Top/bottom rounding differ at the two raster sizes; treating ymin
        // (a bottom bearing) as the first bitmap row swaps the coverage here.
        for (target_bottom, high_bottom) in [(0, 1), (-2, -7)] {
            let target = Metrics {
                ymin: target_bottom,
                width: 1,
                height: 2,
                ..Metrics::default()
            };
            let high = Metrics {
                ymin: high_bottom,
                width: 4,
                height: 5,
                ..Metrics::default()
            };
            assert_eq!(
                downsample(target, high, &[255; 20], 4, 0, 0).unwrap(),
                [32_768, 49_151],
                "integer baseline, bottom bearing {target_bottom}",
            );
            assert_eq!(
                downsample(target, high, &[255; 20], 4, 0, 128).unwrap(),
                [0, 65_535, 16_384],
                "half-pixel downward translation, bottom bearing {target_bottom}",
            );
        }
    }

    #[test]
    fn supersampled_glyphs_keep_the_native_raster_baseline() {
        let face =
            FontFace::parse(include_bytes!("../../assets/fonts/DMMono-Regular.ttf")).unwrap();
        let mut atlas = FontAtlas::new(512, 256).unwrap();
        for px in [18.0, 24.0] {
            let baseline = ceil_f32(face.inner.horizontal_line_metrics(px).ascent);
            for scalar in "HIMNXhmnpqgy_.,():0123456789".chars() {
                // Independent reference: the rasterizer's native coverage,
                // not another mask produced by our atlas/downsampler.
                let (metrics, native) = face.inner.rasterize(scalar, px);
                let native_top = baseline - (metrics.ymin as f32 + metrics.height as f32);
                let mut reference_mass = 0.0;
                let mut reference_moment = 0.0;
                for (index, &alpha) in native.iter().enumerate() {
                    reference_mass += f64::from(alpha);
                    reference_moment += f64::from(alpha)
                        * (f64::from(native_top) + (index / metrics.width) as f64 + 0.5);
                }
                let mut utf8 = [0; 4];
                let run = face
                    .layout_line(&mut atlas, scalar.encode_utf8(&mut utf8), px)
                    .unwrap();
                let glyph = run.glyphs()[0];
                let mut mass = 0.0;
                let mut moment = 0.0;
                for y in 0..usize::from(glyph.ah) {
                    for x in 0..usize::from(glyph.aw) {
                        let alpha = atlas.coverage()
                            [(usize::from(glyph.ay) + y) * 512 + usize::from(glyph.ax) + x];
                        mass += f64::from(alpha);
                        moment += f64::from(alpha) * (f64::from(glyph.dy) + y as f64 + 0.5);
                    }
                }
                assert!(mass > 0.0 && reference_mass > 0.0);
                let drift = moment / mass - reference_moment / reference_mass;
                assert!(
                    drift.abs() < 0.25,
                    "{scalar:?} at {px}px moved vertically by {drift}px"
                );
            }
        }
    }

    #[test]
    fn fractional_layout_bakes_phase_into_atlas() {
        let face = FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap();
        let mut atlas = FontAtlas::new(512, 256).unwrap();
        let run = face
            .layout_line_at(&mut atlas, "a quieter line", 17.25, 73, 41)
            .unwrap();

        assert!(
            run.glyphs()
                .iter()
                .all(|glyph| glyph.subpixel_x == 0 && glyph.subpixel_y == 0)
        );
        assert!(
            atlas
                .entries()
                .iter()
                .any(|glyph| glyph.key.phase_x != 0 || glyph.key.phase_y != 0)
        );

        let whitespace = face.layout_line_at(&mut atlas, " ", 17.25, 73, 41).unwrap();
        assert_eq!(
            (whitespace.glyphs()[0].aw, whitespace.glyphs()[0].ah),
            (0, 0)
        );
    }

    #[test]
    fn multi_line_convenience_steps_the_baseline() {
        let face = FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap();
        let mut atlas = FontAtlas::new(256, 256).unwrap();
        let run = face.layout_line(&mut atlas, "A\nA", 16.0).unwrap();
        assert_eq!(run.len(), 2);
        let line_height = ceil_f32(face.inner.horizontal_line_metrics(16.0).new_line_size) as i16;
        assert_eq!(run.glyphs()[1].dy - run.glyphs()[0].dy, line_height);
        assert_eq!(run.glyphs()[1].dx, run.glyphs()[0].dx);
        assert_eq!(atlas.glyph_count(), 1, "both lines share one raster");
    }

    #[test]
    fn dmsans_a_at_12_5_px_has_coverage_and_stable_metrics() {
        let face = FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap();
        let glyph = face.glyph_id('A');
        assert_ne!(glyph, 0);
        let metrics = face.metrics(glyph, 12.5);
        assert!(!metrics.is_empty());
        assert!((6..=12).contains(&metrics.width), "{metrics:?}");
        assert!((7..=12).contains(&metrics.height), "{metrics:?}");
        assert!(
            metrics.advance > 6.0 && metrics.advance < 12.0,
            "{metrics:?}"
        );
        assert_eq!(metrics.ymin, 0, "a capital sits on the baseline");

        let mut out = [0u16; 32 * 32];
        let raster = face
            .rasterize_into(glyph, 12.5, 0, 0, &mut out, 32)
            .unwrap();
        assert_eq!(raster, metrics);
        let mass: u64 = out.iter().map(|&c| u64::from(c)).sum();
        assert!(mass > 0);
        assert!(out.iter().any(|&c| c > 60_000), "a stem is fully covered");
        // Nothing outside the stored box is touched.
        for y in 0..32 {
            for x in 0..32 {
                if x >= usize::from(metrics.width) || y >= usize::from(metrics.height) {
                    assert_eq!(out[y * 32 + x], 0, "({x},{y})");
                }
            }
        }
        // Re-parsing yields the same metrics and raster.
        let again = FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap();
        let mut out2 = [0u16; 32 * 32];
        again
            .rasterize_into(glyph, 12.5, 0, 0, &mut out2, 32)
            .unwrap();
        assert_eq!(again.metrics(glyph, 12.5), metrics);
        assert_eq!(out2, out);
        // Phase expansion is honoured.
        let mut phased = [0u16; 32 * 32];
        let raster = face
            .rasterize_into(glyph, 12.5, 128, 64, &mut phased, 32)
            .unwrap();
        assert_eq!(raster, metrics);
        let (sw, sh) = metrics.stored_extent(128, 64);
        assert_eq!((sw, sh), (metrics.width + 1, metrics.height + 1));
        assert!(phased[usize::from(sw) - 1] > 0 || phased[usize::from(sh - 1) * 32] > 0);
        let phased_mass: u64 = phased.iter().map(|&c| u64::from(c)).sum();
        assert!((phased_mass as i64 - mass as i64).abs() < mass as i64 / 50);
    }

    #[test]
    fn advance_agrees_with_full_metrics() {
        for (bytes, weight) in [
            (&include_bytes!("../../assets/fonts/DMSans.ttf")[..], None),
            (
                &include_bytes!("../../assets/fonts/Fraunces.ttf")[..],
                Some(900.0),
            ),
        ] {
            let face = match weight {
                Some(weight) => FontFace::parse_with_weight(bytes, weight).unwrap(),
                None => FontFace::parse(bytes).unwrap(),
            };
            for c in "AWij. 0…".chars() {
                let glyph = face.glyph_id(c);
                for px in [8.0, 12.5, 96.0] {
                    assert_eq!(
                        face.advance(glyph, px),
                        face.metrics(glyph, px).advance,
                        "{c:?} {px}"
                    );
                }
            }
        }
    }

    #[test]
    fn font_ids_distinguish_weights_and_can_be_pinned() {
        let bytes = include_bytes!("../../assets/fonts/Fraunces.ttf");
        let regular = FontFace::parse(bytes).unwrap();
        let bold = FontFace::parse_with_weight(bytes, 900.0).unwrap();
        assert_ne!(regular.font_id(), bold.font_id());
        assert_eq!(regular.clone().with_font_id(7).font_id(), 7);
        let key = GlyphKey {
            font: regular.font_id(),
            glyph: 1,
            px_bits: 0,
            phase_x: 0,
            phase_y: 0,
        };
        assert_ne!(
            key.hash(),
            GlyphKey {
                font: bold.font_id(),
                ..key
            }
            .hash()
        );
    }
}
