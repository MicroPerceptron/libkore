//! Bounded single-line layout over any [`GlyphSource`] and [`GlyphAtlas`].

use super::{
    AtlasStorage, FontError, GlyphAtlas, GlyphSource, ceil_f32, round_f32, split_glyph_position,
    valid_px,
};
use crate::PlacedGlyph;

/// What a line does with text past its `max_width`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Overflow {
    /// Emit only the glyphs whose advance keeps the run within the bound
    /// and drop the rest.
    #[default]
    Clip,
    /// Drop trailing glyphs until an ellipsis (`…`, or `...` when the font
    /// has no ellipsis glyph) fits, then emit it.  Falls back to
    /// [`Overflow::Clip`] when even the ellipsis is wider than the bound.
    Ellipsis,
}

/// Parameters of one laid-out line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSpec {
    /// Pixel size.
    pub px: f32,
    /// Fractional X of the run origin in 1/256 pixel; baked into the atlas
    /// rasters so the emitted glyphs carry zero compositor phase.
    pub phase_x: u8,
    /// Fractional Y of the run origin in 1/256 pixel.
    pub phase_y: u8,
    /// Bound on the run's advance, in pixels.
    pub max_width: Option<u16>,
    /// How the bound is enforced.
    pub overflow: Overflow,
}

impl LineSpec {
    /// An unbounded line at `px` with integer origin.
    pub const fn new(px: f32) -> Self {
        Self {
            px,
            phase_x: 0,
            phase_y: 0,
            max_width: None,
            overflow: Overflow::Clip,
        }
    }

    /// Set the origin phase.
    pub const fn with_phase(mut self, phase_x: u8, phase_y: u8) -> Self {
        self.phase_x = phase_x;
        self.phase_y = phase_y;
        self
    }

    /// Bound the run.
    pub const fn with_max_width(mut self, max_width: u16, overflow: Overflow) -> Self {
        self.max_width = Some(max_width);
        self.overflow = overflow;
        self
    }
}

/// What [`layout_line_into`] produced.
///
/// Vertical values are relative to the top of the line box: the baseline
/// sits `ascent` pixels down and the box is `ascent + descent` tall.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LineMetrics {
    /// Glyphs written to `out` (one per emitted character, including
    /// zero-area ones, plus the ellipsis glyphs when truncated).
    pub glyphs: u16,
    /// Pen position after the last glyph, in pixels: the run's width.
    pub advance_px: f32,
    /// Baseline offset from the line top (`ceil` of the font ascent).
    pub ascent: u16,
    /// Line box extent below the baseline (`ceil` of the font descent).
    pub descent: u16,
    /// Top of the ink, relative to the line top (`0` when there is none).
    pub ink_top: i16,
    /// Bottom of the ink (exclusive), relative to the line top.
    pub ink_bottom: i16,
    /// Whether any text was dropped (overflow, or a second line).
    pub truncated: bool,
}

impl LineMetrics {
    /// Line-box height.
    #[inline]
    pub const fn line_height(&self) -> u16 {
        self.ascent.saturating_add(self.descent)
    }

    /// Offset that centres the run's advance and line box in a
    /// `box_w × box_h` rectangle, to add to every glyph's `(dx, dy)`.
    pub fn centered_offset(&self, box_w: u16, box_h: u16) -> (i16, i16) {
        (
            to_i16((f32::from(box_w) - self.advance_px) / 2.0),
            to_i16((f32::from(box_h) - f32::from(self.line_height())) / 2.0),
        )
    }

    /// Like [`Self::centered_offset`], but centres the ink vertically
    /// instead of the line box — the right choice for a single capital in
    /// a small pip.  Falls back to the line box when the run has no ink.
    pub fn centered_ink_offset(&self, box_w: u16, box_h: u16) -> (i16, i16) {
        if self.ink_bottom <= self.ink_top {
            return self.centered_offset(box_w, box_h);
        }
        let ink_height = f32::from(self.ink_bottom) - f32::from(self.ink_top);
        (
            to_i16((f32::from(box_w) - self.advance_px) / 2.0),
            to_i16((f32::from(box_h) - ink_height) / 2.0 - f32::from(self.ink_top)),
        )
    }
}

#[inline]
fn to_i16(value: f32) -> i16 {
    round_f32(value).clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

/// The trailing glyphs an overflowing line ends with.
#[derive(Clone, Copy)]
enum Ellipsis {
    None,
    Single(u16),
    Dots(u16),
}

impl Ellipsis {
    fn glyphs(self) -> ([u16; 3], usize) {
        match self {
            Self::None => ([0; 3], 0),
            Self::Single(g) => ([g, 0, 0], 1),
            Self::Dots(g) => ([g; 3], 3),
        }
    }
}

struct Plan {
    keep: usize,
    ellipsis: Ellipsis,
    truncated: bool,
}

/// Lay out one line of `text` from `source` into `out`, caching rasters in
/// `atlas`.
///
/// Every glyph's `(dx, dy)` is relative to the top-left of the line box;
/// the baseline is `LineMetrics::ascent` pixels down.  One `PlacedGlyph`
/// is emitted per character (zero-area ones for spaces), plus the ellipsis
/// when [`Overflow::Ellipsis`] truncates.  Text after the first `\n` is
/// dropped and reported as truncated.
///
/// Warm path: no allocation — a kern lookup, a key hash and a binary
/// search per glyph.  Cold path: the miss is rasterised into the atlas.
/// On error the contents of `out` are unspecified.
///
/// # Errors
///
/// * [`FontError::InvalidPixelSize`] — `spec.px` is not finite and
///   positive.
/// * [`FontError::RunCapacity`] — `out` cannot hold the run.
/// * [`FontError::AtlasFull`] / [`FontError::IndexFull`] — the atlas cannot
///   take another raster.
/// * [`FontError::CoordinateOverflow`] — a position left `i16`.
pub fn layout_line_into<G: GlyphSource + ?Sized, S: AtlasStorage>(
    source: &G,
    atlas: &mut GlyphAtlas<S>,
    text: &str,
    spec: LineSpec,
    out: &mut [PlacedGlyph],
) -> Result<LineMetrics, FontError> {
    let px = spec.px;
    if !valid_px(px) {
        return Err(FontError::InvalidPixelSize);
    }
    let vertical = source.vertical_metrics(px);
    let ascent = to_u16(ceil_f32(vertical.ascent))?;
    let descent = to_u16(ceil_f32(vertical.descent))?;
    let baseline = i32::from(ascent);

    let line = text.split('\n').next().unwrap_or("");
    let plan = plan_line(source, line, px, spec)?;

    let mut pen = Pen {
        x: 0.0,
        previous: None,
        placed: 0,
        ink_top: i16::MAX,
        ink_bottom: i16::MIN,
    };
    for c in line.chars().take(plan.keep) {
        pen.place(source, atlas, spec, baseline, source.glyph_id(c), out)?;
    }
    let (trail, trail_len) = plan.ellipsis.glyphs();
    for &glyph in &trail[..trail_len] {
        pen.place(source, atlas, spec, baseline, glyph, out)?;
    }

    let (ink_top, ink_bottom) = if pen.ink_bottom > pen.ink_top {
        (pen.ink_top, pen.ink_bottom)
    } else {
        (0, 0)
    };
    Ok(LineMetrics {
        glyphs: u16::try_from(pen.placed).map_err(|_| FontError::RunCapacity)?,
        advance_px: pen.x,
        ascent,
        descent,
        ink_top,
        ink_bottom,
        truncated: plan.truncated || line.len() != text.len(),
    })
}

fn to_u16(value: f32) -> Result<u16, FontError> {
    if value < 0.0 || value > f32::from(u16::MAX) {
        return Err(FontError::CoordinateOverflow);
    }
    Ok(value as u16)
}

struct Pen {
    x: f32,
    previous: Option<u16>,
    placed: usize,
    ink_top: i16,
    ink_bottom: i16,
}

impl Pen {
    fn place<G: GlyphSource + ?Sized, S: AtlasStorage>(
        &mut self,
        source: &G,
        atlas: &mut GlyphAtlas<S>,
        spec: LineSpec,
        baseline: i32,
        glyph: u16,
        out: &mut [PlacedGlyph],
    ) -> Result<(), FontError> {
        let Some(slot) = out.get_mut(self.placed) else {
            return Err(FontError::RunCapacity);
        };
        if let Some(left) = self.previous {
            self.x += source.kern(left, glyph, spec.px);
        }
        let (whole, phase_x) = split_glyph_position(self.x + f32::from(spec.phase_x) / 256.0)?;
        let entry = atlas.entry(source, glyph, spec.px, phase_x, spec.phase_y)?;
        let dx = whole
            .checked_add(entry.xmin)
            .ok_or(FontError::CoordinateOverflow)?;
        let dy = i16::try_from(baseline - i32::from(entry.ymin) - i32::from(entry.height))
            .map_err(|_| FontError::CoordinateOverflow)?;
        *slot = PlacedGlyph {
            ax: entry.ax,
            ay: entry.ay,
            aw: entry.aw,
            ah: entry.ah,
            dx,
            dy,
            subpixel_x: 0,
            subpixel_y: 0,
        };
        if !entry.is_empty() {
            self.ink_top = self.ink_top.min(dy);
            let bottom = i16::try_from(i32::from(dy) + i32::from(entry.ah))
                .map_err(|_| FontError::CoordinateOverflow)?;
            self.ink_bottom = self.ink_bottom.max(bottom);
        }
        self.x += entry.advance;
        self.previous = Some(glyph);
        self.placed += 1;
        Ok(())
    }
}

/// Measure `line` against `spec.max_width` and decide how many leading
/// characters to keep and what to append.
///
/// Only advances and kerning are consulted, so the measure pass never
/// rasterises and — for the provided sources — never allocates.
fn plan_line<G: GlyphSource + ?Sized>(
    source: &G,
    line: &str,
    px: f32,
    spec: LineSpec,
) -> Result<Plan, FontError> {
    let Some(max_width) = spec.max_width else {
        return Ok(Plan {
            keep: usize::MAX,
            ellipsis: Ellipsis::None,
            truncated: false,
        });
    };
    let max_width = f32::from(max_width);

    let ellipsis = match spec.overflow {
        Overflow::Clip => Ellipsis::None,
        Overflow::Ellipsis => {
            let single = source.glyph_id('\u{2026}');
            let dot = source.glyph_id('.');
            if single != 0 {
                Ellipsis::Single(single)
            } else if dot != 0 {
                Ellipsis::Dots(dot)
            } else {
                Ellipsis::None
            }
        }
    };
    let (trail, trail_len) = ellipsis.glyphs();
    let mut trail_width = 0.0f32;
    for i in 0..trail_len {
        if i > 0 {
            trail_width += source.kern(trail[i - 1], trail[i], px);
        }
        trail_width += source.advance(trail[i], px);
    }
    let trail_fits = trail_len > 0 && trail_width <= max_width;

    let mut pen = 0.0f32;
    let mut previous = None;
    let mut count = 0usize;
    let mut keep_clip = 0usize;
    let mut keep_ellipsis = 0usize;
    for (i, c) in line.chars().enumerate() {
        let glyph = source.glyph_id(c);
        // Would the prefix before this glyph, followed by the ellipsis,
        // fit?
        if trail_fits {
            let lead = previous.map_or(0.0, |left| source.kern(left, trail[0], px));
            if pen + lead + trail_width <= max_width {
                keep_ellipsis = i;
            }
        }
        if let Some(left) = previous {
            pen += source.kern(left, glyph, px);
        }
        pen += source.advance(glyph, px);
        if pen <= max_width {
            keep_clip = i + 1;
        }
        previous = Some(glyph);
        count = i + 1;
    }
    if pen <= max_width {
        return Ok(Plan {
            keep: count,
            ellipsis: Ellipsis::None,
            truncated: false,
        });
    }
    Ok(if trail_fits {
        Plan {
            keep: keep_ellipsis,
            ellipsis,
            truncated: true,
        }
    } else {
        Plan {
            keep: keep_clip,
            ellipsis: Ellipsis::None,
            truncated: true,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::{BitmapFace, BorrowedStorage, FontAtlas, FontFace};
    use core::cell::Cell;
    use std::alloc::{GlobalAlloc, Layout, System};

    /// Counts this thread's heap allocations so the warm path can prove it
    /// makes none while other tests run alongside.
    struct Counting;

    std::thread_local! {
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    fn allocations() -> usize {
        ALLOCATIONS.with(Cell::get)
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

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

    fn dmsans() -> FontFace<'static> {
        FontFace::parse(include_bytes!("../../assets/fonts/DMSans.ttf")).unwrap()
    }

    fn run_width(source: &impl GlyphSource, out: &[PlacedGlyph]) -> i32 {
        let _ = source;
        out.iter()
            .filter(|g| g.aw != 0)
            .map(|g| i32::from(g.dx) + i32::from(g.aw))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn unbounded_line_places_every_character() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(256, 128).unwrap();
        let text = "Flyout label";
        let mut out = [BLANK; 16];
        let metrics =
            layout_line_into(&face, &mut atlas, text, LineSpec::new(12.5), &mut out).unwrap();
        assert_eq!(usize::from(metrics.glyphs), text.chars().count());
        assert!(!metrics.truncated);
        assert!(
            metrics.advance_px > 40.0 && metrics.advance_px < 90.0,
            "{metrics:?}"
        );
        assert!(metrics.ascent >= 10 && metrics.descent >= 2, "{metrics:?}");
        assert!(
            metrics.ink_top >= 0 && metrics.ink_bottom > metrics.ink_top,
            "{metrics:?}"
        );
        assert!(i32::from(metrics.ink_bottom) <= i32::from(metrics.line_height()) + 1);
        // Glyphs advance monotonically and the space is zero-area.
        let space = text.chars().position(|c| c == ' ').unwrap();
        assert_eq!((out[space].aw, out[space].ah), (0, 0));
        assert!(
            out[..usize::from(metrics.glyphs)]
                .windows(2)
                .all(|w| w[0].dx <= w[1].dx)
        );
        assert!(run_width(&face, &out) as f32 <= metrics.advance_px + 2.0);
    }

    #[test]
    fn ellipsis_truncation_stays_within_max_width() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(256, 128).unwrap();
        let text = "A fairly long flyout label";
        let mut out = [BLANK; 32];
        let spec = LineSpec::new(12.5).with_max_width(60, Overflow::Ellipsis);
        let metrics = layout_line_into(&face, &mut atlas, text, spec, &mut out).unwrap();
        assert!(metrics.truncated);
        assert!(metrics.advance_px <= 60.0, "{metrics:?}");
        assert!(
            metrics.advance_px > 40.0,
            "ellipsis fills the bound: {metrics:?}"
        );
        assert!(usize::from(metrics.glyphs) < text.chars().count());
        // The last glyph is the ellipsis raster.
        let last = out[usize::from(metrics.glyphs) - 1];
        let ellipsis = atlas
            .entries()
            .iter()
            .find(|e| e.key.glyph == face.glyph_id('…'))
            .unwrap();
        assert_eq!((last.ax, last.ay), (ellipsis.ax, ellipsis.ay));
        assert!(run_width(&face, &out[..usize::from(metrics.glyphs)]) <= 61);

        // A wider bound keeps more text; the whole text needs no ellipsis.
        let wide = layout_line_into(
            &face,
            &mut atlas,
            text,
            LineSpec::new(12.5).with_max_width(400, Overflow::Ellipsis),
            &mut out,
        )
        .unwrap();
        assert!(!wide.truncated);
        assert_eq!(usize::from(wide.glyphs), text.chars().count());
        assert!(
            atlas
                .entries()
                .iter()
                .filter(|e| e.key.glyph == face.glyph_id('…'))
                .count()
                >= 1
        );
    }

    #[test]
    fn clip_truncation_drops_whole_glyphs() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(256, 128).unwrap();
        let text = "A fairly long flyout label";
        let mut out = [BLANK; 32];
        let spec = LineSpec::new(12.5).with_max_width(60, Overflow::Clip);
        let metrics = layout_line_into(&face, &mut atlas, text, spec, &mut out).unwrap();
        assert!(metrics.truncated);
        assert!(metrics.advance_px <= 60.0, "{metrics:?}");
        let kept = usize::from(metrics.glyphs);
        assert!(kept < text.chars().count());
        // Clipping keeps the longest fitting prefix: one more glyph would
        // overflow.
        let next = face.glyph_id(text.chars().nth(kept).unwrap());
        assert!(metrics.advance_px + face.metrics(next, 12.5).advance > 60.0);
        assert!(
            atlas
                .entries()
                .iter()
                .all(|e| e.key.glyph != face.glyph_id('…'))
        );
    }

    #[test]
    fn ellipsis_falls_back_to_dots_and_then_to_clip() {
        let face = BitmapFace::vga_8x16();
        let mut plane = [0u16; 128 * 64];
        let mut atlas =
            GlyphAtlas::<BorrowedStorage<'_, 64>>::from_slice(128, 64, &mut plane).unwrap();
        let mut out = [BLANK; 16];
        // 8 px cells: "HELLO WORLD" is 88 px; 48 px keeps three cells plus
        // three dots.
        let spec = LineSpec::new(16.0).with_max_width(48, Overflow::Ellipsis);
        let metrics = layout_line_into(&face, &mut atlas, "HELLO WORLD", spec, &mut out).unwrap();
        assert!(metrics.truncated);
        assert_eq!(metrics.glyphs, 6);
        assert_eq!(metrics.advance_px, 48.0);
        let dot = atlas.entry(&face, face.glyph_id('.'), 16.0, 0, 0).unwrap();
        assert!(out[3..6].iter().all(|g| (g.ax, g.ay) == (dot.ax, dot.ay)));
        // Narrower than the dots: clip.
        let spec = LineSpec::new(16.0).with_max_width(20, Overflow::Ellipsis);
        let metrics = layout_line_into(&face, &mut atlas, "HELLO WORLD", spec, &mut out).unwrap();
        assert!(metrics.truncated);
        assert_eq!(metrics.glyphs, 2);
        assert_eq!(metrics.advance_px, 16.0);
    }

    #[test]
    fn run_capacity_is_reported() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(256, 128).unwrap();
        let mut out = [BLANK; 3];
        assert_eq!(
            layout_line_into(&face, &mut atlas, "hello", LineSpec::new(12.0), &mut out),
            Err(FontError::RunCapacity)
        );
        // Truncation can bring a long text under the capacity.
        let spec = LineSpec::new(12.0).with_max_width(14, Overflow::Ellipsis);
        let metrics = layout_line_into(&face, &mut atlas, "hello", spec, &mut out).unwrap();
        assert!(metrics.glyphs <= 3);
        assert!(metrics.truncated);
        assert_eq!(
            layout_line_into(&face, &mut atlas, "x", LineSpec::new(0.0), &mut out),
            Err(FontError::InvalidPixelSize)
        );
        let mut empty: [PlacedGlyph; 0] = [];
        assert_eq!(
            layout_line_into(&face, &mut atlas, "", LineSpec::new(12.0), &mut empty)
                .unwrap()
                .glyphs,
            0
        );
    }

    #[test]
    fn atlas_full_surfaces_from_layout() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(16, 16).unwrap();
        let mut out = [BLANK; 8];
        assert_eq!(
            layout_line_into(&face, &mut atlas, "MMMM", LineSpec::new(48.0), &mut out),
            Err(FontError::AtlasFull)
        );
    }

    #[test]
    fn second_line_is_dropped_and_reported() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(256, 128).unwrap();
        let mut out = [BLANK; 8];
        let metrics =
            layout_line_into(&face, &mut atlas, "ab\ncd", LineSpec::new(12.0), &mut out).unwrap();
        assert_eq!(metrics.glyphs, 2);
        assert!(metrics.truncated);
    }

    #[test]
    fn bitmap_glyphs_advance_by_the_cell() {
        let face = BitmapFace::vga_8x16();
        let mut atlas = FontAtlas::new(128, 64).unwrap();
        let mut out = [BLANK; 4];
        let metrics =
            layout_line_into(&face, &mut atlas, "AB", LineSpec::new(16.0), &mut out).unwrap();
        assert_eq!(metrics.advance_px, 16.0);
        assert_eq!((metrics.ascent, metrics.descent), (12, 4));
        assert_eq!(out[1].dx, out[0].dx + 8);
        // 'A' sits on the baseline: ink bottom is the baseline row.
        assert_eq!(i32::from(out[0].dy) + i32::from(out[0].ah), 12);
        assert_eq!(metrics.ink_bottom, 12);
        // 12 px pip: the ink-centred offset puts the cap in the middle.
        let (x, y) = metrics.centered_ink_offset(12, 12);
        let ink_h = metrics.ink_bottom - metrics.ink_top;
        assert_eq!(y + metrics.ink_top, (12 - ink_h) / 2);
        assert!(x < 0, "16 px run in a 12 px box overhangs: {x}");
    }

    #[test]
    fn centered_offset_centres_the_line_box() {
        let metrics = LineMetrics {
            glyphs: 1,
            advance_px: 8.0,
            ascent: 9,
            descent: 3,
            ink_top: 2,
            ink_bottom: 9,
            truncated: false,
        };
        assert_eq!(metrics.centered_offset(12, 12), (2, 0));
        assert_eq!(metrics.centered_offset(20, 20), (6, 4));
        assert_eq!(metrics.centered_ink_offset(12, 12), (2, 1));
        let no_ink = LineMetrics {
            ink_top: 0,
            ink_bottom: 0,
            ..metrics
        };
        assert_eq!(
            no_ink.centered_ink_offset(12, 12),
            no_ink.centered_offset(12, 12)
        );
    }

    #[test]
    fn pip_letter_at_8px_centres_in_a_12px_pip() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(64, 64).unwrap();
        let mut out = [BLANK; 1];
        let metrics =
            layout_line_into(&face, &mut atlas, "A", LineSpec::new(8.0), &mut out).unwrap();
        let (x, y) = metrics.centered_ink_offset(12, 12);
        let glyph = out[0];
        let left = i32::from(glyph.dx) + i32::from(x);
        let top = i32::from(glyph.dy) + i32::from(y);
        let right = left + i32::from(glyph.aw);
        let bottom = top + i32::from(glyph.ah);
        assert!(left >= 0 && right <= 12, "{left}..{right}");
        assert!(top >= 0 && bottom <= 12, "{top}..{bottom}");
        assert!((left - (12 - right)).abs() <= 1, "horizontally centred");
        assert!((top - (12 - bottom)).abs() <= 1, "vertically centred");
    }

    #[test]
    fn warm_path_does_not_allocate() {
        let face = dmsans();
        let mut atlas = FontAtlas::new(512, 256).unwrap();
        let text = "The quick brown fox jump";
        let mut out = [BLANK; 32];
        let spec = LineSpec::new(12.5).with_phase(64, 128);
        let cold = layout_line_into(&face, &mut atlas, text, spec, &mut out).unwrap();
        let bounded = LineSpec::new(12.5).with_max_width(100, Overflow::Ellipsis);
        layout_line_into(&face, &mut atlas, text, bounded, &mut out).unwrap();

        let before = allocations();
        let warm = layout_line_into(&face, &mut atlas, text, spec, &mut out).unwrap();
        assert_eq!(allocations(), before, "warm unbounded layout allocated");
        let warm_bounded = layout_line_into(&face, &mut atlas, text, bounded, &mut out).unwrap();
        assert_eq!(allocations(), before, "warm bounded layout allocated");
        // A bitmap face allocates on neither path.
        let vga = BitmapFace::vga_8x16();
        let mut plane = [0u16; 128 * 64];
        let mut borrowed =
            GlyphAtlas::<BorrowedStorage<'_, 64>>::from_slice(128, 64, &mut plane).unwrap();
        let before = allocations();
        layout_line_into(&vga, &mut borrowed, "HELLO WORLD", bounded, &mut out).unwrap();
        layout_line_into(&vga, &mut borrowed, "HELLO WORLD", spec, &mut out).unwrap();
        assert_eq!(allocations(), before, "bitmap layout allocated");
        assert_eq!(warm, cold);
        assert!(warm_bounded.truncated);
    }
}
