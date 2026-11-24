//! Rounded-shape rasterization over packed XRGB surfaces.
//!
//! Row decomposition for `rounded_fill` / `border_ring` / `shadow_fill`:
//! solid spans go through `fill_rect_u32` / `fill_blend_u32`, and only the
//! pixels whose coverage actually varies — the corner boxes, or a shadow's
//! halo — evaluate the rounded-rectangle distance field, in strips of at
//! most [`SDF_CHUNK`] lanes staged on the stack. ISA-neutral: every kernel
//! comes from `arch::raster`, so the tiers decide the lane width.
//!
//! Coordinates are `i64` throughout: an `i32` origin plus a `u16` extent
//! never overflows, and clipping happens before anything is narrowed.

use core::ops::Range;

use axnn::RoundedShape;

use crate::arch::raster as kernels;
use crate::arch::scalar::raster::scale_pixel_u8;

/// Longest distance-field strip evaluated at once (three stack buffers).
const SDF_CHUNK: usize = 64;

/// A writable pitched `[height, width]` view of packed XRGB pixels.
///
/// `words` starts at the view origin; `pitch` is in pixels. The backend
/// validates that `(height − 1) · pitch + width ≤ words.len()`.
pub(crate) struct Surface<'a> {
    pub words: &'a mut [u32],
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
}

impl Surface<'_> {
    /// Pixels `[x0, x1)` of row `y` (view coordinates, in bounds).
    #[inline]
    pub(crate) fn row_mut(&mut self, y: usize, x0: usize, x1: usize) -> &mut [u32] {
        let base = y * self.pitch;
        &mut self.words[base + x0..base + x1]
    }

    /// Solid fill of `[x0, x1) × [y0, y1)` — already clipped to the view;
    /// empty ranges are no-ops. Opaque pixels memset, others source-over.
    fn fill(&mut self, x0: i64, x1: i64, y0: i64, y1: i64, pixel: u32) {
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        let region = &mut self.words[y0 * self.pitch + x0..];
        if pixel >> 24 == 255 {
            kernels::fill_rect_u32(region, self.pitch, pixel, x1 - x0, y1 - y0);
        } else {
            kernels::fill_blend_u32(region, self.pitch, pixel, x1 - x0, y1 - y0);
        }
    }
}

/// Shape-centred geometry the distance field is evaluated against.
#[derive(Clone, Copy)]
struct Geo {
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
}

impl Geo {
    fn inset(self, by: i64) -> Self {
        Self {
            half_w: self.half_w - by as f32,
            half_h: self.half_h - by as f32,
            ..self
        }
    }
}

/// How a distance-field strip becomes coverage.
#[derive(Clone, Copy)]
enum Mode<'a> {
    /// `clamp(0.5 − d, 0, 1)`.
    Fill,
    /// `cov(outer) − cov(outer inset by width)`.
    Ring { inner: Geo, width: i64 },
    /// Table lookup at `d` (see `axnn::gaussian_falloff`).
    Falloff(&'a [u8]),
}

/// A non-empty shape placed on a surface with its clip window resolved.
struct Placed {
    sx: i64,
    sy: i64,
    sw: i64,
    sh: i64,
    /// `[top-left, top-right, bottom-right, bottom-left]`, clamped.
    radii: [i64; 4],
    /// Clip window in view coordinates (exclusive ends), non-empty.
    cx0: i64,
    cx1: i64,
    cy0: i64,
    cy1: i64,
    geo: Geo,
}

impl Placed {
    /// `None` when the shape is empty or `shape` grown by `margin` on every
    /// side misses the view entirely.
    fn new(surface: &Surface<'_>, shape: RoundedShape, margin: i64) -> Option<Self> {
        if shape.is_empty() {
            return None;
        }
        let (sx, sy) = (i64::from(shape.x), i64::from(shape.y));
        let (sw, sh) = (i64::from(shape.w), i64::from(shape.h));
        let cx0 = (sx - margin).max(0);
        let cx1 = (sx + sw + margin).min(surface.width as i64);
        let cy0 = (sy - margin).max(0);
        let cy1 = (sy + sh + margin).min(surface.height as i64);
        if cx0 >= cx1 || cy0 >= cy1 {
            return None;
        }
        let radii = shape.clamped_radii().map(i64::from);
        let geo = Geo {
            cx: sx as f32 + sw as f32 * 0.5,
            cy: sy as f32 + sh as f32 * 0.5,
            half_w: sw as f32 * 0.5,
            half_h: sh as f32 * 0.5,
        };
        Some(Self {
            sx,
            sy,
            sw,
            sh,
            radii,
            cx0,
            cx1,
            cy0,
            cy1,
            geo,
        })
    }

    /// First row below the top corner boxes.
    fn top_end(&self) -> i64 {
        self.sy + self.radii[0].max(self.radii[1])
    }

    /// First row of the bottom corner boxes.
    fn bottom_start(&self) -> i64 {
        self.sy + self.sh - self.radii[2].max(self.radii[3])
    }

    /// Corner-box widths `(left, right)` on row `y`: the radius of the
    /// corner whose rows include `y`, else 0. Clamped radii guarantee the
    /// top and bottom boxes never share a row.
    fn row_radii(&self, y: i64) -> (i64, i64) {
        let j = y - self.sy;
        let [tl, tr, br, bl] = self.radii;
        let left = if j < tl {
            tl
        } else if j >= self.sh - bl {
            bl
        } else {
            0
        };
        let right = if j < tr {
            tr
        } else if j >= self.sh - br {
            br
        } else {
            0
        };
        (left, right)
    }

    /// Visible rows above `top_end` or from `bottom_start` on — the rows
    /// that need per-span treatment (corner boxes, or a ring's caps).
    fn edge_rows(&self, top_end: i64, bottom_start: i64) -> impl Iterator<Item = i64> {
        let top = self.cy0..top_end.min(self.cy1);
        let bottom = bottom_start.max(self.cy0)..self.cy1;
        top.chain(bottom)
    }

    /// Radius of the corner whose quadrant contains pixel `(x, y)`:
    /// `x < centre` is left, `y < centre` is top (integer centre column
    /// `sx + sw / 2` is exactly where `px` turns non-negative).
    fn quadrant_radius(&self, x: i64, y: i64) -> i64 {
        let left = x < self.sx + self.sw / 2;
        let top = y < self.sy + self.sh / 2;
        match (top, left) {
            (true, true) => self.radii[0],
            (true, false) => self.radii[1],
            (false, false) => self.radii[2],
            (false, true) => self.radii[3],
        }
    }

    /// Evaluate the distance field over columns `x` of row `y` with corner
    /// `radius`, clip to the window, and blend `pixel` through the coverage.
    fn sdf_span(
        &self,
        s: &mut Surface<'_>,
        y: i64,
        x: Range<i64>,
        radius: i64,
        mode: Mode<'_>,
        pixel: u32,
    ) {
        let (x0, x1) = (x.start.max(self.cx0), x.end.min(self.cx1));
        if x0 >= x1 || y < self.cy0 || y >= self.cy1 {
            return;
        }
        let geo = self.geo;
        let radius_f = radius as f32;
        let py = (y as f32 + 0.5) - geo.cy;
        let mut d = [0f32; SDF_CHUNK];
        let mut d_inner = [0f32; SDF_CHUNK];
        let mut cov = [0u8; SDF_CHUNK];
        let mut x = x0;
        while x < x1 {
            let n = ((x1 - x) as usize).min(SDF_CHUNK);
            let px0 = (x as f32 + 0.5) - geo.cx;
            kernels::rounded_sdf_row_f32(&mut d[..n], px0, py, geo.half_w, geo.half_h, radius_f);
            match mode {
                Mode::Fill => kernels::sdf_coverage_u8(&d[..n], &mut cov[..n]),
                Mode::Ring { inner, width } => {
                    let inner_radius = (radius - width).max(0) as f32;
                    kernels::rounded_sdf_row_f32(
                        &mut d_inner[..n],
                        (x as f32 + 0.5) - inner.cx,
                        (y as f32 + 0.5) - inner.cy,
                        inner.half_w,
                        inner.half_h,
                        inner_radius,
                    );
                    kernels::sdf_ring_coverage_u8(&d[..n], &d_inner[..n], &mut cov[..n]);
                }
                Mode::Falloff(table) => kernels::sdf_falloff_u8(&d[..n], table, &mut cov[..n]),
            }
            let row = s.row_mut(y as usize, x as usize, x as usize + n);
            kernels::blend_row_alpha(row, pixel, &cov[..n]);
            x += n as i64;
        }
    }

    /// [`sdf_span`](Self::sdf_span) for a span that may cross the centre
    /// column: split it so each part uses its quadrant's radius.
    fn quadrant_span(
        &self,
        s: &mut Surface<'_>,
        y: i64,
        x: Range<i64>,
        mode: Mode<'_>,
        pixel: u32,
    ) {
        let split = (self.sx + self.sw / 2).clamp(x.start, x.end);
        if x.start < split {
            let radius = self.quadrant_radius(x.start, y);
            self.sdf_span(s, y, x.start..split, radius, mode, pixel);
        }
        if split < x.end {
            let radius = self.quadrant_radius(split, y);
            self.sdf_span(s, y, split..x.end, radius, mode, pixel);
        }
    }
}

/// Fill `shape` with one premultiplied pixel; see `RasterBackend::rounded_fill`.
pub(crate) fn rounded_fill(s: &mut Surface<'_>, pixel: u32, shape: RoundedShape) {
    if pixel >> 24 == 0 {
        return;
    }
    let Some(g) = Placed::new(s, shape, 0) else {
        return;
    };
    // Interior rows: one pitched fill.
    s.fill(
        g.cx0,
        g.cx1,
        g.top_end().max(g.cy0),
        g.bottom_start().min(g.cy1),
        pixel,
    );
    // Corner rows: box · span · box.
    for y in g.edge_rows(g.top_end(), g.bottom_start()) {
        let (lr, rr) = g.row_radii(y);
        let (left_end, right_start) = (g.sx + lr, g.sx + g.sw - rr);
        g.sdf_span(s, y, g.sx..left_end, lr, Mode::Fill, pixel);
        s.fill(left_end.max(g.cx0), right_start.min(g.cx1), y, y + 1, pixel);
        g.sdf_span(s, y, right_start..g.sx + g.sw, rr, Mode::Fill, pixel);
    }
}

/// Stroke the inside edge of `shape`; see `RasterBackend::border_ring`.
pub(crate) fn border_ring(s: &mut Surface<'_>, pixel: u32, shape: RoundedShape, width: u16) {
    if width == 0 || pixel >> 24 == 0 {
        return;
    }
    let w = i64::from(width);
    if w * 2 >= i64::from(shape.w) || w * 2 >= i64::from(shape.h) {
        return rounded_fill(s, pixel, shape);
    }
    let Some(g) = Placed::new(s, shape, 0) else {
        return;
    };
    let ring = Mode::Ring {
        inner: g.geo.inset(w),
        width: w,
    };
    let (sx, sy, sw, sh) = (g.sx, g.sy, g.sw, g.sh);

    // Rows past both the corner boxes and the horizontal caps: the two
    // vertical edges are solid spans.
    let top_end = g.top_end().max(sy + w);
    let bottom_start = g.bottom_start().min(sy + sh - w);
    let (iy0, iy1) = (top_end.max(g.cy0), bottom_start.min(g.cy1));
    s.fill(sx.max(g.cx0), (sx + w).min(g.cx1), iy0, iy1, pixel);
    s.fill(
        (sx + sw - w).max(g.cx0),
        (sx + sw).min(g.cx1),
        iy0,
        iy1,
        pixel,
    );

    for y in g.edge_rows(top_end, bottom_start) {
        let j = y - sy;
        let (lr, rr) = g.row_radii(y);
        let (left_end, right_start) = (sx + lr, sx + sw - rr);
        let edge_row = j < w || j >= sh - w;
        if lr > 0 {
            g.sdf_span(s, y, sx..left_end, lr, ring, pixel);
        } else if !edge_row {
            s.fill(sx.max(g.cx0), (sx + w).min(g.cx1), y, y + 1, pixel);
        }
        if edge_row {
            s.fill(left_end.max(g.cx0), right_start.min(g.cx1), y, y + 1, pixel);
        }
        if rr > 0 {
            g.sdf_span(s, y, right_start..sx + sw, rr, ring, pixel);
        } else if !edge_row {
            s.fill(
                (sx + sw - w).max(g.cx0),
                (sx + sw).min(g.cx1),
                y,
                y + 1,
                pixel,
            );
        }
    }
}

/// Soft shadow of `shape`; see `RasterBackend::shadow_fill`. `falloff`
/// holds `1..=axnn::MAX_FALLOFF` entries (validated by the backend, asserted here).
pub(crate) fn shadow_fill(s: &mut Surface<'_>, pixel: u32, shape: RoundedShape, falloff: &[u8]) {
    assert!((1..=axnn::MAX_FALLOFF).contains(&falloff.len()));
    if pixel >> 24 == 0 {
        return;
    }
    let len = falloff.len() as i64;
    let half = len / 2;
    // The table clamps beyond `len − 1 − half` px outside; nothing farther
    // is painted, so the halo is the shape grown by `len − half`.
    let Some(g) = Placed::new(s, shape, len - half) else {
        return;
    };
    // Deep interior: inset by max(radius, half) puts every pixel centre at
    // `d ≤ −half`, where the table is pinned to `falloff[0]`.
    let k = g.radii.iter().copied().max().unwrap_or(0).max(half);
    let interior = (g.sw > 2 * k && g.sh > 2 * k).then(|| {
        (
            (g.sx + k).max(g.cx0),
            (g.sx + g.sw - k).min(g.cx1),
            (g.sy + k).max(g.cy0),
            (g.sy + g.sh - k).min(g.cy1),
        )
    });
    if let Some((ix0, ix1, iy0, iy1)) = interior
        && falloff[0] > 0
    {
        s.fill(
            ix0,
            ix1,
            iy0,
            iy1,
            scale_pixel_u8(pixel, u32::from(falloff[0])),
        );
    }
    let mode = Mode::Falloff(falloff);
    for y in g.cy0..g.cy1 {
        match interior {
            Some((ix0, ix1, iy0, iy1)) if y >= iy0 && y < iy1 && ix0 < ix1 => {
                g.quadrant_span(s, y, g.cx0..ix0, mode, pixel);
                g.quadrant_span(s, y, ix1..g.cx1, mode, pixel);
            }
            _ => g.quadrant_span(s, y, g.cx0..g.cx1, mode, pixel),
        }
    }
}
