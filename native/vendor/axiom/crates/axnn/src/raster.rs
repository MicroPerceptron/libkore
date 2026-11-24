use crate::backend::Backend;
use crate::dtype::DType;
use crate::tensor::Tensor;

/// Largest coverage table [`RasterBackend::shadow_fill`] accepts.
pub const MAX_FALLOFF: usize = 32;

/// A rounded rectangle positioned relative to a destination view's origin.
///
/// `x`/`y` may be negative and the box may extend past the view: backends
/// clip to the view and never touch pixels outside it. `radii` are the
/// corner radii in pixels, ordered top-left, top-right, bottom-right,
/// bottom-left; each is clamped to `min(w, h) / 2` when rasterized (see
/// [`clamped_radii`](Self::clamped_radii)) so adjacent arcs never overlap.
///
/// Coverage is defined by the rounded-rectangle signed-distance field
/// sampled at pixel centres, with the radius of whichever quadrant the
/// pixel centre lies in (`x < centre` is left, `y < centre` is top).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RoundedShape {
    /// Left edge, relative to the view origin.
    pub x: i32,
    /// Top edge, relative to the view origin.
    pub y: i32,
    /// Width in pixels.
    pub w: u16,
    /// Height in pixels.
    pub h: u16,
    /// Corner radii: `[top-left, top-right, bottom-right, bottom-left]`.
    pub radii: [u16; 4],
}

impl RoundedShape {
    /// A plain rectangle (all radii zero).
    pub const fn rect(x: i32, y: i32, w: u16, h: u16) -> Self {
        Self {
            x,
            y,
            w,
            h,
            radii: [0; 4],
        }
    }

    /// A rectangle with the same radius on every corner.
    pub const fn uniform(x: i32, y: i32, w: u16, h: u16, radius: u16) -> Self {
        Self {
            x,
            y,
            w,
            h,
            radii: [radius; 4],
        }
    }

    /// Whether the shape covers no pixels at all.
    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// The radii a backend actually rasterizes: each corner clamped to
    /// `min(w, h) / 2`.
    pub const fn clamped_radii(self) -> [u16; 4] {
        let limit = if self.w < self.h { self.w } else { self.h } / 2;
        let mut radii = self.radii;
        let mut i = 0;
        while i < 4 {
            if radii[i] > limit {
                radii[i] = limit;
            }
            i += 1;
        }
        radii
    }

    /// The shape `inset` pixels inside this one, every radius reduced by
    /// `inset` — the hole a [`border_ring`](RasterBackend::border_ring) of
    /// that width leaves. `None` when nothing remains (or the origin would
    /// overflow).
    pub const fn inset(self, inset: u16) -> Option<Self> {
        let twice = inset as u32 * 2;
        if twice >= self.w as u32 || twice >= self.h as u32 {
            return None;
        }
        let (Some(x), Some(y)) = (
            self.x.checked_add(inset as i32),
            self.y.checked_add(inset as i32),
        ) else {
            return None;
        };
        let outer = self.clamped_radii();
        let mut radii = [0u16; 4];
        let mut i = 0;
        while i < 4 {
            radii[i] = outer[i].saturating_sub(inset);
            i += 1;
        }
        Some(Self {
            x,
            y,
            w: self.w - inset * 2,
            h: self.h - inset * 2,
            radii,
        })
    }
}

/// Coverage table for [`RasterBackend::shadow_fill`].
///
/// Entry `i` is the coverage (0..=255) at signed distance `i - L / 2`
/// pixels from the shape edge — negative is inside — for a step edge
/// blurred by a Gaussian of `sigma = blur / 2` px (the CSS `box-shadow`
/// blur convention), i.e. `255 · Φ(−d / sigma)`. `blur == 0` is a crisp
/// step (255 inside, 128 on the edge, 0 outside). Entries are
/// non-increasing; the table reaches 0 once `L / 2 ≳ 3 · sigma`, which is
/// what bounds the painted halo, so choose `L` (≤ [`MAX_FALLOFF`]) to
/// cover roughly `3 · sigma` on each side.
///
/// `const`-capable: the polynomial `erfc` (Abramowitz–Stegun 7.1.26,
/// error < 1.5e-7) and the reduced-range `exp` are evaluated in `f64`.
pub const fn gaussian_falloff<const L: usize>(blur: u8) -> [u8; L] {
    let mut lut = [0u8; L];
    let sigma = blur as f64 * 0.5;
    let half = (L / 2) as i64;
    let mut i = 0;
    while i < L {
        let d = (i as i64 - half) as f64;
        let coverage = if sigma <= 0.0 {
            if d < 0.0 {
                1.0
            } else if d > 0.0 {
                0.0
            } else {
                0.5
            }
        } else {
            0.5 * erfc_const(d / (sigma * core::f64::consts::SQRT_2))
        };
        lut[i] = (coverage * 255.0 + 0.5) as u8;
        i += 1;
    }
    lut
}

/// Complementary error function, Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7).
const fn erfc_const(x: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let tail = poly * exp_const(-ax * ax);
    if x < 0.0 { 2.0 - tail } else { tail }
}

/// `exp(x)` for `x <= 0`: reduce by `ln 2`, Taylor on `[0, ln 2)`, rescale.
const fn exp_const(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    let ln2 = core::f64::consts::LN_2;
    // n = floor(x / ln 2) for non-positive x, so r = x − n·ln 2 ∈ [0, ln 2).
    let n = (x / ln2) as i64 - 1;
    let r = x - n as f64 * ln2;
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1;
    while k < 24 {
        term = term * r / k as f64;
        sum += term;
        k += 1;
    }
    let mut count = -n;
    while count > 0 {
        sum *= 0.5;
        count -= 1;
    }
    sum
}

/// A writable pitched `[height, width]` view of packed XRGB words, lent by
/// [`RasterBackend::surface_mut`]: `words[y · pitch + x]` is pixel `(x, y)`
/// of the view, `pitch ≥ width`, and `(height − 1) · pitch + width ≤
/// words.len()`.
pub struct SurfaceMut<'a> {
    pub words: &'a mut [u32],
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
}

/// Axis-aligned rectangle in pixel coordinates — the scene, layout and
/// raster vocabulary's one rectangle (`i32` origin, `u16` extent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u16,
    pub h: u16,
}

impl Rect {
    pub const ZERO: Self = Self {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    /// Compute the intersection of two rects. Returns `Rect::ZERO` if
    /// they don't overlap or either is zero-area.
    pub fn intersect(&self, other: &Rect) -> Rect {
        if !self.intersects(other) {
            return Rect::ZERO;
        }
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.w as i32).min(other.x + other.w as i32);
        let y1 = (self.y + self.h as i32).min(other.y + other.h as i32);
        Rect {
            x: x0,
            y: y0,
            w: (x1 - x0) as u16,
            h: (y1 - y0) as u16,
        }
    }

    /// Does this rect overlap `other`? Zero-area rects never intersect.
    #[inline]
    pub fn intersects(&self, other: &Rect) -> bool {
        self.w > 0
            && self.h > 0
            && other.w > 0
            && other.h > 0
            && self.x < other.x + other.w as i32
            && self.x + self.w as i32 > other.x
            && self.y < other.y + other.h as i32
            && self.y + self.h as i32 > other.y
    }

    /// Does this rect contain the pixel `(x, y)`?
    #[inline]
    pub const fn contains_point(&self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.w as i32 && y < self.y + self.h as i32
    }

    /// Does this rect fully contain `other`? Zero-area rects contain
    /// nothing and are contained by nothing.
    #[inline]
    pub const fn contains_rect(&self, other: &Rect) -> bool {
        self.w > 0
            && self.h > 0
            && other.w > 0
            && other.h > 0
            && other.x >= self.x
            && other.y >= self.y
            && other.x + other.w as i32 <= self.x + self.w as i32
            && other.y + other.h as i32 <= self.y + self.h as i32
    }

    /// Smallest rect containing both `self` and `other`.
    ///
    /// If either rect is zero-area, returns the other.
    pub fn union(&self, other: &Rect) -> Rect {
        if self.w == 0 || self.h == 0 {
            return *other;
        }
        if other.w == 0 || other.h == 0 {
            return *self;
        }
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w as i32).max(other.x + other.w as i32);
        let y1 = (self.y + self.h as i32).max(other.y + other.h as i32);
        Rect {
            x: x0,
            y: y0,
            w: (x1 - x0) as u16,
            h: (y1 - y0) as u16,
        }
    }
}

/// How a glyph atlas stores coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoverageFormat {
    /// One byte per pixel, `0..=255`.
    U8,
    /// One `u16` per pixel, `0..=65535`, quantized to 8 bits at blend time.
    U16,
    /// Packed `0xAARRGGBB` words whose top byte is the coverage.
    PackedAlpha,
}

impl CoverageFormat {
    /// The tensor dtype that carries this format.
    pub const fn dtype(self) -> DType {
        match self {
            Self::U8 => DType::U8,
            Self::U16 => DType::I16,
            Self::PackedAlpha => DType::I32,
        }
    }

    /// Bytes per coverage sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::PackedAlpha => 4,
        }
    }
}

/// Words per row of a raster program's parameter table (`I32 [rows, WORDS]`).
pub const RASTER_ROW_WORDS: usize = 20;

/// One raster node's per-frame parameters — a row of the frame program's
/// I32 `[rows, RASTER_ROW_WORDS]` parameter tensor, filled by the scene
/// lowering for one band and read by the node that owns the row.  Every
/// backend sees the same row; the lowering has already resolved clipping,
/// damage and occlusion into `vis` and `tiles`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterRow {
    /// The node's shape rect in frame coordinates: corner arcs, sample
    /// UVs and glyph origins are measured from it.
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    /// The part of the shape this node may touch in the band (shape ∩
    /// ancestor clip ∩ band); empty means the node paints nothing.
    pub vx: i32,
    pub vy: i32,
    pub vw: u32,
    pub vh: u32,
    /// Corner radii `[top-left, top-right, bottom-right, bottom-left]`.
    pub radii: [u16; 4],
    /// Premultiplied `0xAARRGGBB` for fills, rings and shadows; straight
    /// `0x00RRGGBB` ink for glyphs.
    pub argb: u32,
    /// Node opacity `0..=255` for surfaces and glyphs.
    pub opacity: u32,
    /// Op-specific words: ring width; shadow falloff; a surface's source
    /// slot and origin; a sample's source rect; a glyph run's start and
    /// length.
    pub aux: [u32; 3],
    /// Tile columns of the band this node paints (bit `c` = column `c`),
    /// after damage and occlusion.
    pub tiles: [u64; 2],
}

impl RasterRow {
    pub const EMPTY: Self = Self {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
        vx: 0,
        vy: 0,
        vw: 0,
        vh: 0,
        radii: [0; 4],
        argb: 0,
        opacity: 0,
        aux: [0; 3],
        tiles: [0; 2],
    };

    /// Whether the row paints anything at all.
    pub const fn is_empty(&self) -> bool {
        self.vw == 0 || self.vh == 0 || (self.tiles[0] == 0 && self.tiles[1] == 0)
    }

    pub fn write(&self, words: &mut [i32]) {
        assert!(words.len() >= RASTER_ROW_WORDS);
        words[0] = self.x;
        words[1] = self.y;
        words[2] = self.w as i32;
        words[3] = self.h as i32;
        words[4] = self.vx;
        words[5] = self.vy;
        words[6] = self.vw as i32;
        words[7] = self.vh as i32;
        words[8] = (u32::from(self.radii[0]) | (u32::from(self.radii[1]) << 16)) as i32;
        words[9] = (u32::from(self.radii[2]) | (u32::from(self.radii[3]) << 16)) as i32;
        words[10] = self.argb as i32;
        words[11] = self.opacity as i32;
        words[12] = self.aux[0] as i32;
        words[13] = self.aux[1] as i32;
        words[14] = self.aux[2] as i32;
        words[15] = 0;
        words[16] = self.tiles[0] as u32 as i32;
        words[17] = (self.tiles[0] >> 32) as u32 as i32;
        words[18] = self.tiles[1] as u32 as i32;
        words[19] = (self.tiles[1] >> 32) as u32 as i32;
    }

    pub fn read(words: &[i32]) -> Option<Self> {
        if words.len() < RASTER_ROW_WORDS {
            return None;
        }
        let u = |i: usize| words[i] as u32;
        Some(Self {
            x: words[0],
            y: words[1],
            w: u(2),
            h: u(3),
            vx: words[4],
            vy: words[5],
            vw: u(6),
            vh: u(7),
            radii: [
                u(8) as u16,
                (u(8) >> 16) as u16,
                u(9) as u16,
                (u(9) >> 16) as u16,
            ],
            argb: u(10),
            opacity: u(11),
            aux: [u(12), u(13), u(14)],
            tiles: [
                u64::from(u(16)) | (u64::from(u(17)) << 32),
                u64::from(u(18)) | (u64::from(u(19)) << 32),
            ],
        })
    }
}

/// A single glyph positioned within a text run, ready to sample — the
/// record of a raster program's glyph-table resource and the output of an
/// external shaping + atlas-lookup pipeline.
///
/// `ax, ay` is the glyph bitmap's top-left in the atlas, `aw, ah` its
/// size; `dx, dy` the offset from the owning node's origin (negative for a
/// left-bearing overhang); `subpixel_x, subpixel_y` an unsigned fractional
/// phase in `1/256` pixel.  `#[repr(C)]`, 14 bytes: the resource is these
/// records back to back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PlacedGlyph {
    pub ax: u16,
    pub ay: u16,
    pub aw: u16,
    pub ah: u16,
    pub dx: i16,
    pub dy: i16,
    pub subpixel_x: u8,
    pub subpixel_y: u8,
}

impl PlacedGlyph {
    /// Add a fractional origin, carrying into the integer offsets.
    ///
    /// Returns `None` only if the translated integer offset exceeds `i16`.
    pub fn translated_subpixel(mut self, x: u8, y: u8) -> Option<Self> {
        let x = u16::from(self.subpixel_x) + u16::from(x);
        let y = u16::from(self.subpixel_y) + u16::from(y);
        self.dx = self.dx.checked_add((x >> 8) as i16)?;
        self.dy = self.dy.checked_add((y >> 8) as i16)?;
        self.subpixel_x = x as u8;
        self.subpixel_y = y as u8;
        Some(self)
    }
}

/// The glyph-table resource is `PlacedGlyph` records back to back.
pub const PLACED_GLYPH_BYTES: usize = 14;
const _: () = assert!(core::mem::size_of::<PlacedGlyph>() == PLACED_GLYPH_BYTES);

/// Raster pipeline operations on tensors.
///
/// Separate from [`ComputeBackend`](crate::compute::ComputeBackend)
/// because rasterization is an optional capability — not every backend
/// needs software triangle rendering. Backends opt in by implementing
/// this trait.
///
/// All operations dispatch on the input tensor's dtype. Currently
/// only F32 edge functions and U32 framebuffers are implemented;
/// the trait is dtype-agnostic so F16/Q8 kernels can be added
/// without changing the API.
pub trait RasterBackend: Backend {
    /// Fill a `[H, W]` view with one packed pixel — the direct path
    /// for opaque solid fills (no intermediate tile allocation).
    fn fill_rect(&self, dst: &mut Tensor<Self::Storage<'_>>, pixel: u32)
    -> Result<(), Self::Error>;

    /// Source-over blend one premultiplied packed pixel across a
    /// `[H, W]` view — the direct path for translucent solid fills.
    fn fill_blend(
        &self,
        dst: &mut Tensor<Self::Storage<'_>>,
        pixel: u32,
    ) -> Result<(), Self::Error>;

    /// Fused 3-edge coverage test + conditional framebuffer write.
    ///
    /// For each pixel: if all three edge functions ≥ `thresh`,
    /// write `color[i]` into `fb[i]`.
    fn edge_coverage(
        &self,
        e0: &Tensor<Self::Storage<'_>>,
        e1: &Tensor<Self::Storage<'_>>,
        e2: &Tensor<Self::Storage<'_>>,
        thresh: f32,
        color: &Tensor<Self::Storage<'_>>,
        fb: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Depth test: compare fragment Z against Z-buffer, update both.
    ///
    /// For covered pixels where `frag_z ≤ zbuf`: writes new Z and
    /// sets `out_mask` to all-ones. Other pixels get zero mask.
    fn depth_test(
        &self,
        coverage: &Tensor<Self::Storage<'_>>,
        frag_z: &Tensor<Self::Storage<'_>>,
        zbuf: &mut Tensor<Self::Storage<'_>>,
        out_mask: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Barycentric interpolation of a single attribute.
    ///
    /// `dst[i] = w0[i]*a0 + w1[i]*a1 + w2[i]*a2`
    fn bary_interp(
        &self,
        w0: &Tensor<Self::Storage<'_>>,
        w1: &Tensor<Self::Storage<'_>>,
        w2: &Tensor<Self::Storage<'_>>,
        a0: f32,
        a1: f32,
        a2: f32,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Alpha blend: composite premultiplied-alpha src over dst
    /// where mask is non-zero.
    fn alpha_blend(
        &self,
        mask: &Tensor<Self::Storage<'_>>,
        src: &Tensor<Self::Storage<'_>>,
        dst: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Unpack XRGB u32 pixels into separate R, G, B channel tensors.
    fn xrgb_unpack(
        &self,
        packed: &Tensor<Self::Storage<'_>>,
        r: &mut Tensor<Self::Storage<'_>>,
        g: &mut Tensor<Self::Storage<'_>>,
        b: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Pack R, G, B channel tensors into XRGB u32 pixels.
    fn xrgb_pack(
        &self,
        r: &Tensor<Self::Storage<'_>>,
        g: &Tensor<Self::Storage<'_>>,
        b: &Tensor<Self::Storage<'_>>,
        packed: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Opaque rectangular blit.
    fn blit_opaque(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        dst: &mut Tensor<Self::Storage<'_>>,
        width: usize,
        height: usize,
    ) -> Result<(), Self::Error>;

    /// Alpha-blended rectangular blit.
    fn blit_blend(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        dst: &mut Tensor<Self::Storage<'_>>,
        width: usize,
        height: usize,
    ) -> Result<(), Self::Error>;

    /// Nearest-neighbour scaled blit with alpha blending.
    ///
    /// Samples a `src_w × src_h` region from `src` and renders it
    /// into `dst_w × dst_h` pixels of `dst`, using integer fixed-point
    /// UV mapping. Every sampled pixel is alpha-blended (premultiplied
    /// src-over).
    fn blit_nearest(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        dst: &mut Tensor<Self::Storage<'_>>,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<(), Self::Error>;

    // ── System-UI primitives: rounded shapes, rings, shadows, masks ────

    /// Fill a rounded rectangle with one premultiplied ARGB pixel.
    ///
    /// Rows outside the corner boxes are plain [`fill_rect`] /
    /// [`fill_blend`] spans; pixels inside a corner box receive analytic
    /// coverage from the rounded-rectangle signed-distance field
    /// (`clamp(0.5 − d, 0, 1)` at the pixel centre), which scales the
    /// pixel's alpha before a source-over blend. `shape` is relative to
    /// the view origin and is clipped to the `[H, W]` view; empty,
    /// off-view, and fully transparent (`alpha == 0`) fills are no-ops.
    /// A shape whose radii are all zero is exactly `fill_rect`/`fill_blend`.
    ///
    /// [`fill_rect`]: Self::fill_rect
    /// [`fill_blend`]: Self::fill_blend
    fn rounded_fill(
        &self,
        dst: &mut Tensor<Self::Storage<'_>>,
        pixel: u32,
        shape: RoundedShape,
    ) -> Result<(), Self::Error>;

    /// Stroke the inside edge of a rounded rectangle `width` pixels wide.
    ///
    /// Coverage is `coverage(shape) − coverage(shape.inset(width))`, so
    /// the ring lies entirely inside `shape`. Interior rows are two solid
    /// spans; only the corner boxes evaluate the two distance fields.
    /// `width == 0` is a no-op and a width that leaves no hole degenerates
    /// to [`rounded_fill`](Self::rounded_fill).
    fn border_ring(
        &self,
        dst: &mut Tensor<Self::Storage<'_>>,
        pixel: u32,
        shape: RoundedShape,
        width: u16,
    ) -> Result<(), Self::Error>;

    /// Paint a soft shadow of a rounded rectangle.
    ///
    /// Every pixel's coverage is `falloff` sampled at its signed distance
    /// from the shape edge (entry `i` ↔ distance `i − len / 2`, see
    /// [`gaussian_falloff`]), linearly interpolated between entries and
    /// clamped at both ends — `falloff[0]` covers the deep interior. Pixels
    /// farther than `len − len / 2` outside the shape are never touched,
    /// so the last entry should be 0. `falloff` must hold
    /// `1..=MAX_FALLOFF` entries.
    fn shadow_fill(
        &self,
        dst: &mut Tensor<Self::Storage<'_>>,
        pixel: u32,
        shape: RoundedShape,
        falloff: &[u8],
    ) -> Result<(), Self::Error>;

    /// Tint a coverage mask onto the view:
    /// `dst = over(premultiply(color, coverage · opacity), dst)`.
    ///
    /// Serves glyphs and icon masks. `coverage` is a `[H, W]` view of the
    /// same shape as `dst`: `U8` (0..=255), `I16` holding u16 coverage
    /// (0..=65535, quantized to 8 bits), or `I32` packed pixels whose top
    /// byte is the coverage. `color` is straight (non-premultiplied)
    /// `0xXXRRGGBB`; its top byte is ignored.
    fn coverage_blend(
        &self,
        coverage: &Tensor<Self::Storage<'_>>,
        color: u32,
        opacity: u8,
        dst: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Blit premultiplied ARGB `src` over `dst` through a `U8` coverage
    /// mask: each source pixel is scaled by its mask byte and blended
    /// source-over. All three are `[H, W]` views of the same shape.
    fn blit_masked(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        mask: &Tensor<Self::Storage<'_>>,
        dst: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Lend a writable `[H, W]` packed-XRGB view as pitched host words —
    /// the seam through which pixels that exist only outside the backend
    /// (a compositor's volatile snapshot of a client surface another
    /// address space is writing) land on the view directly, with no
    /// staging tensor in between. A backend whose surfaces are not
    /// host-addressable refuses; the accelerator route binds its surfaces
    /// as tensors instead.
    fn surface_mut<'a>(
        &self,
        dst: &'a mut Tensor<Self::Storage<'_>>,
    ) -> Result<SurfaceMut<'a>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radii_clamp_to_half_the_shorter_side_and_inset_shrinks_them() {
        let shape = RoundedShape {
            x: 3,
            y: -4,
            w: 20,
            h: 11,
            radii: [0, 4, 9, 16],
        };
        assert_eq!(shape.clamped_radii(), [0, 4, 5, 5]);
        let inner = shape.inset(2).unwrap();
        assert_eq!(
            inner,
            RoundedShape {
                x: 5,
                y: -2,
                w: 16,
                h: 7,
                radii: [0, 2, 3, 3],
            }
        );
        assert_eq!(
            shape.inset(5),
            Some(RoundedShape {
                x: 8,
                y: 1,
                w: 10,
                h: 1,
                radii: [0; 4]
            })
        );
        assert_eq!(shape.inset(6), None);
        assert!(RoundedShape::rect(0, 0, 0, 5).is_empty());
        assert_eq!(RoundedShape::uniform(1, 2, 8, 8, 6).clamped_radii(), [4; 4]);
    }

    #[test]
    fn crisp_falloff_is_a_step_and_blurred_falloffs_are_monotone_symmetric() {
        const STEP: [u8; 8] = gaussian_falloff::<8>(0);
        assert_eq!(STEP, [255, 255, 255, 255, 128, 0, 0, 0]);

        for blur in [1u8, 2, 4, 8, 12] {
            let lut = gaussian_falloff::<32>(blur);
            // 50 % coverage is 127.5 levels; the polynomial erfc lands on 127.
            assert!(
                (127..=128).contains(&lut[16]),
                "edge sample for blur {blur}"
            );
            assert!(
                lut.windows(2).all(|w| w[0] >= w[1]),
                "monotone for blur {blur}"
            );
            for k in 1..16 {
                let sum = u32::from(lut[16 - k]) + u32::from(lut[16 + k]);
                assert!(
                    (254..=256).contains(&sum),
                    "symmetry at ±{k} for blur {blur}"
                );
            }
        }
        // sigma = 2 px: Φ(−1) ≈ 0.1587 at d = 2, Φ(−2) ≈ 0.0228 at d = 4.
        let lut = gaussian_falloff::<32>(4);
        assert_eq!(lut[18], 40);
        assert_eq!(lut[20], 6);
        assert_eq!(lut[24], 0);
        assert_eq!(lut[0], 255);
    }
}

// ── Rounded-silhouette geometry ───────────────────────────────────────
//
// The partition of a rounded rect into its four corner boxes (masked) and
// the full-width row bands between them (painted unmasked), the radius
// clamp every rasteriser applies, and the shadow falloff tables — shared
// by the scene lowering (which culls and clips by them) and the kernels
// (which paint by them), so the two never disagree about a corner.

/// Side of the stack coverage chunk a masked corner is rasterised in.
///
/// A corner box (or the part of one inside a tile) is painted in pieces of
/// at most `MASK_SIDE × MASK_SIDE` pixels, so the per-piece mask is a fixed
/// 4 KiB stack array whatever the tile size or radius.
pub const MASK_SIDE: usize = 64;

/// Corner radii `[top-left, top-right, bottom-right, bottom-left]`.
pub type Radii = [u16; 4];

/// The radii a rect is rasterised with: each corner clamped to half the
/// shorter edge — [`RoundedShape::clamped_radii`]'s rule, so adjacent arcs
/// never overlap.
#[inline]
pub fn clamped(rect: Rect, radii: Radii) -> Radii {
    let limit = rect.w.min(rect.h) / 2;
    radii.map(|r| r.min(limit))
}

/// One corner box: the `radius × radius` square the arc lives in.
#[derive(Clone, Copy, Debug)]
pub struct CornerBox {
    pub rect: Rect,
    pub radius: u16,
}

/// The four corner boxes of `rect` (`radii` already clamped); a square
/// corner yields an empty box.
pub fn corner_boxes(rect: Rect, radii: Radii) -> [CornerBox; 4] {
    let right = rect.x + i32::from(rect.w);
    let bottom = rect.y + i32::from(rect.h);
    let square = |x: i32, y: i32, r: u16| CornerBox {
        rect: Rect { x, y, w: r, h: r },
        radius: r,
    };
    let [tl, tr, br, bl] = radii;
    [
        square(rect.x, rect.y, tl),
        square(right - i32::from(tr), rect.y, tr),
        square(right - i32::from(br), bottom - i32::from(br), br),
        square(rect.x, bottom - i32::from(bl), bl),
    ]
}

/// The full-coverage bands of `rect`: the rows between the corner boxes,
/// each inset past whichever boxes share its rows. Returns the band rects
/// (screen coordinates) and how many are live; at most five.
pub fn bands(rect: Rect, radii: Radii) -> ([Rect; 5], usize) {
    let [tl, tr, br, bl] = radii;
    let h = rect.h;
    let mut ys = [0, tl, tr, h - bl, h - br, h];
    // Insertion sort: six elements, no allocation.
    for i in 1..ys.len() {
        let mut j = i;
        while j > 0 && ys[j - 1] > ys[j] {
            ys.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut out = [Rect::ZERO; 5];
    let mut n = 0;
    for k in 0..ys.len() - 1 {
        let (a, b) = (ys[k], ys[k + 1]);
        if a >= b {
            continue;
        }
        let left = if a < tl {
            tl
        } else if a >= h - bl {
            bl
        } else {
            0
        };
        let right = if a < tr {
            tr
        } else if a >= h - br {
            br
        } else {
            0
        };
        out[n] = Rect {
            x: rect.x + i32::from(left),
            y: rect.y + i32::from(a),
            w: rect.w - left - right,
            h: b - a,
        };
        n += 1;
    }
    (out, n)
}

/// One full-width Gaussian table per blur extent, built at compile time.
const SHADOW_TABLES: [[u8; MAX_FALLOFF]; MAX_FALLOFF + 1] = build_shadow_tables();

const fn build_shadow_tables() -> [[u8; MAX_FALLOFF]; MAX_FALLOFF + 1] {
    let mut tables = [[0u8; MAX_FALLOFF]; MAX_FALLOFF + 1];
    let mut blur = 0;
    while blur <= MAX_FALLOFF {
        tables[blur] = gaussian_falloff::<MAX_FALLOFF>(blur as u8);
        blur += 1;
    }
    tables
}

/// Length of the falloff window for `blur`: even, covering about `3σ`
/// (`σ = blur / 2`) on each side so the last entry is zero, capped at
/// [`MAX_FALLOFF`] — beyond a blur of ten the tail is truncated.
const fn shadow_table_len(blur: u8) -> usize {
    let blur = if blur as usize > MAX_FALLOFF {
        MAX_FALLOFF
    } else {
        blur as usize
    };
    let len = (3 * blur + 3) & !1;
    if len < 4 {
        4
    } else if len > MAX_FALLOFF {
        MAX_FALLOFF
    } else {
        len
    }
}

/// The falloff table [`RasterBackend::shadow_fill`] consumes for a shadow
/// of `blur` — the centred window of the full table, since entry `i` of
/// an `L`-table is distance `i − L / 2` in both.
pub fn shadow_falloff(blur: u8) -> &'static [u8] {
    let len = shadow_table_len(blur);
    let lo = MAX_FALLOFF / 2 - len / 2;
    &SHADOW_TABLES[(blur as usize).min(MAX_FALLOFF)][lo..lo + len]
}

/// How far a shadow of `falloff` reaches outside its casting shape, in
/// pixels.
///
/// A shadow node's rect is the shadow's whole extent: the shape that casts
/// it is the rect inset by this halo on every side, so the fade never
/// spills past the node's damage or tile spans. Callers size a shadow node
/// as the shaded rect grown by `shadow_halo(falloff)`.
pub const fn shadow_halo(falloff: u8) -> u16 {
    (shadow_table_len(falloff) / 2) as u16
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    fn r(x: i32, y: i32, w: u16, h: u16) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn bands_and_boxes_partition_the_rect() {
        let rect = r(4, 6, 40, 30);
        let radii = clamped(rect, [6, 0, 9, 3]);
        let (bands, n) = bands(rect, radii);
        let boxes = corner_boxes(rect, radii);
        let mut covered = 0usize;
        for y in rect.y..rect.y + i32::from(rect.h) {
            for x in rect.x..rect.x + i32::from(rect.w) {
                let in_band = bands[..n].iter().filter(|b| b.contains_point(x, y)).count();
                let in_box = boxes.iter().filter(|b| b.rect.contains_point(x, y)).count();
                assert_eq!(in_band + in_box, 1, "({x}, {y}) covered {in_band}+{in_box}");
                covered += 1;
            }
        }
        assert_eq!(covered, 40 * 30);
    }

    #[test]
    fn shadow_windows_reach_zero_and_the_halo_matches() {
        for blur in 0..=12u8 {
            let table = shadow_falloff(blur);
            assert_eq!(table.len() % 2, 0);
            assert_eq!(usize::from(shadow_halo(blur)), table.len() / 2);
            // Up to a blur of ten the window reaches zero; past it the tail
            // is truncated at `MAX_FALLOFF`.
            if usize::from(blur) * 3 + 3 <= MAX_FALLOFF {
                assert_eq!(*table.last().unwrap(), 0, "blur {blur}");
            } else {
                assert_eq!(table.len(), MAX_FALLOFF);
            }
        }
    }
}
