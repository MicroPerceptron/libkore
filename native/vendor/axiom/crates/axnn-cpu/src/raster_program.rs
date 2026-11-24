//! The CPU lane's raster nodes (`SemanticOp::Raster`): the borrowed
//! target, surface, atlas and glyph-table views a run binds, and the one
//! kernel that paints a node's row of the parameter table onto the target.
//!
//! A raster node's row ([`RasterRow`]) carries everything that changes per
//! frame — shape, visible rect, radii, colour, opacity, source slot, tile
//! mask — and the node's static parameters ([`RasterNodeParams`]) say which
//! row it reads and where its target, surfaces, atlas and glyph table are
//! bound.  The kernel paints `vis ∩ target ∩ (dirty ∩ row.tiles)` one tile
//! run at a time through the crate's shape and blit kernels, so the CPU
//! lane's pixels are exactly the immediate raster path's.

use core::marker::PhantomData;
use core::simd::Simd;

use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, KernelAccuracy, KernelCandidate, KernelConstraints,
    KernelCost, KernelEffects, KernelParams, KernelRequest, KernelShapePolicy, RasterNodeParams,
    RasterOp, SemanticOp,
};
use axnn::{CoverageFormat, DType, PlacedGlyph, RASTER_ROW_WORDS, RasterRow, Rect};

use crate::arch::raster as kernels;
use crate::planner::{typed_input_matches, typed_output_matches};
use crate::rounded::{self, MASK_SIDE, Radii};
use crate::shapes::{self, Surface};
use crate::{CpuError, CpuImplementation, CpuKernelPlan};

/// A packed-XRGB surface a raster node samples: a client's backing another
/// address space may be writing (rows are then read volatile), or an
/// ordinary slice.
#[derive(Clone, Copy, Debug)]
pub struct XrgbSurfaceRef<'a> {
    base: *const u32,
    width: u32,
    height: u32,
    pitch: usize,
    volatile: bool,
    _borrow: PhantomData<&'a [u32]>,
}

impl<'a> XrgbSurfaceRef<'a> {
    /// A surface over `words`: `height` rows of `width` pixels, `pitch`
    /// words apart.
    pub fn new(words: &'a [u32], width: u32, height: u32, pitch: usize) -> Result<Self, CpuError> {
        if width == 0
            || height == 0
            || pitch < width as usize
            || (height as usize - 1)
                .checked_mul(pitch)
                .and_then(|n| n.checked_add(width as usize))
                .is_none_or(|n| n > words.len())
        {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed XRGB surface",
            });
        }
        Ok(Self {
            base: words.as_ptr(),
            width,
            height,
            pitch,
            volatile: false,
            _borrow: PhantomData,
        })
    }

    /// A surface another party writes concurrently: every row is read
    /// volatile, torn at most, never borrowed as a slice.
    ///
    /// # Safety
    /// `[base + y · pitch, + width)` is readable and 4-aligned for every
    /// `y < height` for the lifetime `'a`.
    pub unsafe fn from_raw_volatile(
        base: *const u32,
        width: u32,
        height: u32,
        pitch: usize,
    ) -> Self {
        Self {
            base,
            width,
            height,
            pitch,
            volatile: true,
            _borrow: PhantomData,
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Copy `out.len()` pixels of row `y` from column `x`; the span is
    /// inside the surface by the caller's clipping.
    pub fn read_row(&self, x: u32, y: u32, out: &mut [u32]) {
        debug_assert!(y < self.height && x as usize + out.len() <= self.width as usize);
        // SAFETY: the span is inside the surface (`new` / `from_raw_volatile`
        // contract) and the caller clipped to it.
        let src = unsafe { self.base.add(y as usize * self.pitch + x as usize) };
        if self.volatile {
            unsafe { read_row_volatile(src, out) };
        } else {
            // SAFETY: a plain surface is an ordinary borrowed slice.
            out.copy_from_slice(unsafe { core::slice::from_raw_parts(src, out.len()) });
        }
    }
}

/// Volatile row snapshot: lane-aligned SIMD loads on the aligned body, one
/// word at a time on the unaligned head and the tail.
///
/// # Safety
/// `[src, src + dst.len())` is readable and 4-aligned.
unsafe fn read_row_volatile(src: *const u32, dst: &mut [u32]) {
    const LANES: usize = 8;
    let align = core::mem::align_of::<Simd<u32, LANES>>();
    let head = ((align - (src as usize % align)) % align) / 4;
    let head = head.min(dst.len());
    let body = (dst.len() - head) / LANES;
    // SAFETY: every read is inside the span the caller vouched for; the
    // body pointer is lane-aligned by construction.
    unsafe {
        for (i, word) in dst[..head].iter_mut().enumerate() {
            *word = src.add(i).read_volatile();
        }
        let mut p = src.add(head) as *const Simd<u32, LANES>;
        for k in 0..body {
            let lanes = p.read_volatile();
            let at = head + k * LANES;
            dst[at..at + LANES].copy_from_slice(&lanes.to_array());
            p = p.add(1);
        }
        let done = head + body * LANES;
        for (i, word) in dst[done..].iter_mut().enumerate() {
            *word = src.add(done + i).read_volatile();
        }
    }
}

/// The packed-XRGB target a raster node paints: a view (a band stage or
/// a whole framebuffer) whose first pixel stands for frame position
/// `origin`.  The executor hands it to one node at a time.
#[derive(Clone, Copy, Debug)]
pub struct XrgbTargetRef<'a> {
    base: *mut u32,
    origin_x: i32,
    origin_y: i32,
    width: u32,
    height: u32,
    pitch: usize,
    _borrow: PhantomData<&'a mut [u32]>,
}

impl<'a> XrgbTargetRef<'a> {
    /// A target over `words`: `height` rows of `width` pixels, `pitch`
    /// words apart, standing for the frame rect at `origin`.
    pub fn new(
        words: &'a mut [u32],
        origin: (i32, i32),
        width: u32,
        height: u32,
        pitch: usize,
    ) -> Result<Self, CpuError> {
        if width == 0
            || height == 0
            || width > u32::from(u16::MAX)
            || height > u32::from(u16::MAX)
            || pitch < width as usize
            || (height as usize - 1)
                .checked_mul(pitch)
                .and_then(|n| n.checked_add(width as usize))
                .is_none_or(|n| n > words.len())
        {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed XRGB target",
            });
        }
        Ok(Self {
            base: words.as_mut_ptr(),
            origin_x: origin.0,
            origin_y: origin.1,
            width,
            height,
            pitch,
            _borrow: PhantomData,
        })
    }

    /// A target over raw memory (a pool carve).
    ///
    /// # Safety
    /// `[base + y · pitch, + width)` is writable and 4-aligned for every
    /// `y < height` for the lifetime `'a`, and nothing else writes it while
    /// a run paints.
    pub unsafe fn from_raw(
        base: *mut u32,
        origin: (i32, i32),
        width: u32,
        height: u32,
        pitch: usize,
    ) -> Result<Self, CpuError> {
        if width == 0
            || height == 0
            || width > u32::from(u16::MAX)
            || height > u32::from(u16::MAX)
            || pitch < width as usize
            || base.is_null()
            || !base.is_aligned()
        {
            return Err(CpuError::ShapeMismatch {
                op: "raw XRGB target",
            });
        }
        Ok(Self {
            base,
            origin_x: origin.0,
            origin_y: origin.1,
            width,
            height,
            pitch,
            _borrow: PhantomData,
        })
    }

    /// The frame rect this target stands for.
    pub const fn frame_rect(&self) -> Rect {
        Rect {
            x: self.origin_x,
            y: self.origin_y,
            w: self.width as u16,
            h: self.height as u16,
        }
    }

    /// The writable view under `rect` (frame coordinates, inside
    /// [`frame_rect`](Self::frame_rect)).
    ///
    /// # Safety
    /// The executor runs one node at a time, so no two views of one
    /// target are live together; `rect` is inside the target.
    unsafe fn view(&self, rect: Rect) -> Surface<'_> {
        let x = (rect.x - self.origin_x) as usize;
        let y = (rect.y - self.origin_y) as usize;
        let (w, h) = (usize::from(rect.w), usize::from(rect.h));
        let words = unsafe {
            core::slice::from_raw_parts_mut(
                self.base.add(y * self.pitch + x),
                (h - 1) * self.pitch + w,
            )
        };
        Surface {
            words,
            pitch: self.pitch,
            width: w,
            height: h,
        }
    }
}

/// A glyph coverage atlas: one `width × height` plane in `format`.
#[derive(Clone, Copy, Debug)]
pub struct CoverageAtlasRef<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    format: CoverageFormat,
}

impl<'a> CoverageAtlasRef<'a> {
    pub fn new(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        format: CoverageFormat,
    ) -> Result<Self, CpuError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(format.bytes_per_sample()));
        if width == 0 || height == 0 || expected != Some(bytes.len()) {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed coverage atlas",
            });
        }
        Ok(Self {
            bytes,
            width,
            height,
            format,
        })
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn format(&self) -> CoverageFormat {
        self.format
    }
}

/// The placed glyphs a frame's text runs index into.
#[derive(Clone, Copy, Debug)]
pub struct GlyphTableRef<'a> {
    glyphs: &'a [PlacedGlyph],
}

impl<'a> GlyphTableRef<'a> {
    pub const fn new(glyphs: &'a [PlacedGlyph]) -> Self {
        Self { glyphs }
    }

    pub const fn glyphs(&self) -> &'a [PlacedGlyph] {
        self.glyphs
    }
}

/// Offer the scalar raster kernel for every raster request.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    if !matches!(request.semantic, SemanticOp::Raster(_)) {
        return false;
    }
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints: KernelConstraints {
            input_dtypes: [
                DTypeSet::one(DType::I32),
                DTypeSet::one(DType::U8),
                DTypeSet::one(DType::I32),
            ],
            output_dtypes: DTypeSet::one(DType::I32),
            max_rank: 2,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::I32)
        },
        accuracy: KernelAccuracy::Strict,
        cost: KernelCost::new(0, 1),
    });
    true
}

/// Paint one raster node: its row of `rows` (the I32 parameter table),
/// restricted to the band's `dirty` tile columns, onto `target`.
/// `source(slot)` resolves a surface bound in the node's surface group;
/// `atlas` and `glyphs` serve a glyph run.  `token[0]` reports whether any
/// pixel was painted.  A missing surface, atlas or glyph table paints
/// nothing — the frame goes on.
#[allow(clippy::too_many_arguments)]
pub fn invoke_raster<'r>(
    plan: &CpuKernelPlan,
    rows: &[i32],
    dirty: &[u8],
    target: XrgbTargetRef<'_>,
    source: &dyn Fn(u32) -> Option<XrgbSurfaceRef<'r>>,
    atlas: Option<CoverageAtlasRef<'r>>,
    glyphs: Option<GlyphTableRef<'r>>,
    token: &mut [i32],
) -> Result<(), CpuError> {
    let (SemanticOp::Raster(op), KernelParams::Raster(node)) = (plan.semantic, plan.params())
    else {
        return Err(CpuError::ShapeMismatch {
            op: "raster node parameters",
        });
    };
    let start = node.row as usize * RASTER_ROW_WORDS;
    let end = node.end_row() as usize * RASTER_ROW_WORDS;
    if !node.is_valid()
        || plan.implementation != CpuImplementation::Scalar
        || plan.effects() != KernelEffects::RASTER
        || plan.temporary_bytes != 0
        || plan.result_count() != 1
        || plan.output_dtype != DType::I32
        || !typed_input_matches(plan, 0, DType::I32, rows.len())
        || !typed_input_matches(plan, 1, DType::U8, dirty.len())
        || plan.input(2).is_some()
        || !typed_output_matches(plan, 0, DType::I32, plan.shape, token.len())
        || end > rows.len()
        || dirty.len() > RasterNodeParams::MAX_TILE_COLUMNS
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned raster node",
        });
    }
    token[0] = 0;
    let mut painted = false;
    for words in rows[start..end].chunks_exact(RASTER_ROW_WORDS) {
        let Some(row) = RasterRow::read(words) else {
            continue;
        };
        if row.is_empty() || (row.tiles[0] == 0 && row.tiles[1] == 0) {
            continue;
        }
        painted |= paint_row(
            op,
            node.tile_width,
            &row,
            dirty,
            &target,
            source,
            atlas,
            glyphs,
        )?;
    }
    token[0] = i32::from(painted);
    Ok(())
}

/// Paint a band's rows, each by its own op, in order: what a program's
/// planned raster nodes do, for a scene whose node kinds and paint order
/// change from frame to frame (a document), for which no program can be
/// built once. `ops[i]` is row `i`'s op, `None` for a row that paints
/// nothing; a row whose tile mask is empty is skipped. The row kernel is
/// the planned nodes', so the pixels are too. `rows` holds at least
/// `ops.len()` rows of the I32 parameter table; `tile_width` is the width
/// of the columns `dirty` and every row's tile mask count. `Ok(true)` when
/// any pixel was painted.
#[allow(clippy::too_many_arguments)]
pub fn paint_rows<'r>(
    ops: &[Option<RasterOp>],
    rows: &[i32],
    tile_width: u32,
    dirty: &[u8],
    target: XrgbTargetRef<'_>,
    source: &dyn Fn(u32) -> Option<XrgbSurfaceRef<'r>>,
    atlas: Option<CoverageAtlasRef<'r>>,
    glyphs: Option<GlyphTableRef<'r>>,
) -> Result<bool, CpuError> {
    if ops
        .len()
        .checked_mul(RASTER_ROW_WORDS)
        .is_none_or(|words| words > rows.len())
        || tile_width == 0
        || tile_width > u32::from(u16::MAX)
        || dirty.len() > RasterNodeParams::MAX_TILE_COLUMNS
    {
        return Err(CpuError::ShapeMismatch { op: "raster rows" });
    }
    let mut painted = false;
    for (&op, words) in ops.iter().zip(rows.as_chunks::<RASTER_ROW_WORDS>().0) {
        let Some(op) = op else {
            continue;
        };
        let Some(row) = RasterRow::read(words) else {
            continue;
        };
        if row.is_empty() || (row.tiles[0] == 0 && row.tiles[1] == 0) {
            continue;
        }
        painted |= paint_row(op, tile_width, &row, dirty, &target, source, atlas, glyphs)?;
    }
    Ok(painted)
}

/// Paint one row of a node over the dirty columns its tile mask names;
/// `true` when a piece was painted.
#[allow(clippy::too_many_arguments)]
fn paint_row<'r>(
    op: RasterOp,
    tile_width: u32,
    row: &RasterRow,
    dirty: &[u8],
    target: &XrgbTargetRef<'_>,
    source: &dyn Fn(u32) -> Option<XrgbSurfaceRef<'r>>,
    atlas: Option<CoverageAtlasRef<'r>>,
    glyphs: Option<GlyphTableRef<'r>>,
) -> Result<bool, CpuError> {
    let shape = rect_of(row.x, row.y, row.w, row.h);
    let vis = rect_of(row.vx, row.vy, row.vw, row.vh).intersect(&target.frame_rect());
    if shape.w == 0 || shape.h == 0 || vis.w == 0 || vis.h == 0 {
        return Ok(false);
    }
    let tile = i64::from(tile_width);
    let on = |c: usize| dirty[c] != 0 && (row.tiles[c / 64] >> (c % 64)) & 1 == 1;
    let mut painted = false;
    let mut c = 0;
    while c < dirty.len() {
        if !on(c) {
            c += 1;
            continue;
        }
        let first = c;
        while c < dirty.len() && on(c) {
            c += 1;
        }
        let x0 = (first as i64 * tile).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let x1 = (c as i64 * tile).clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let run = Rect {
            x: x0,
            y: vis.y,
            w: (x1 - i64::from(x0)).clamp(0, i64::from(u16::MAX)) as u16,
            h: vis.h,
        };
        let piece = vis.intersect(&run);
        if piece.w == 0 || piece.h == 0 {
            continue;
        }
        painted |= paint_piece(op, row, shape, piece, target, source, atlas, glyphs)?;
    }
    Ok(painted)
}

fn rect_of(x: i32, y: i32, w: u32, h: u32) -> Rect {
    Rect {
        x,
        y,
        w: w.min(u32::from(u16::MAX)) as u16,
        h: h.min(u32::from(u16::MAX)) as u16,
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_piece<'r>(
    op: RasterOp,
    row: &RasterRow,
    shape: Rect,
    piece: Rect,
    target: &XrgbTargetRef<'_>,
    source: &dyn Fn(u32) -> Option<XrgbSurfaceRef<'r>>,
    atlas: Option<CoverageAtlasRef<'r>>,
    glyphs: Option<GlyphTableRef<'r>>,
) -> Result<bool, CpuError> {
    let radii = rounded::clamped(shape, row.radii);
    match op {
        RasterOp::Fill => {
            // SAFETY: `piece` is inside the target; one node paints at a time.
            let mut s = unsafe { target.view(piece) };
            shapes::rounded_fill(
                &mut s,
                row.argb,
                rounded::shape_at(shape, radii, piece.x, piece.y),
            );
            Ok(true)
        }
        RasterOp::Ring => {
            let mut s = unsafe { target.view(piece) };
            shapes::border_ring(
                &mut s,
                row.argb,
                rounded::shape_at(shape, radii, piece.x, piece.y),
                row.aux[0].min(u32::from(u16::MAX)) as u16,
            );
            Ok(true)
        }
        RasterOp::Shadow => {
            let falloff = row.aux[0].min(255) as u8;
            let halo = rounded::shadow_halo(falloff);
            if shape.w <= halo * 2 || shape.h <= halo * 2 {
                return Ok(false);
            }
            let cast = Rect {
                x: shape.x + i32::from(halo),
                y: shape.y + i32::from(halo),
                w: shape.w - halo * 2,
                h: shape.h - halo * 2,
            };
            let mut s = unsafe { target.view(piece) };
            shapes::shadow_fill(
                &mut s,
                row.argb,
                rounded::shape_at(cast, radii, piece.x, piece.y),
                rounded::shadow_falloff(falloff),
            );
            Ok(true)
        }
        RasterOp::SurfaceBlit => {
            let Some(surface) = source(row.aux[0]) else {
                return Ok(false);
            };
            let (src_x, src_y) = (row.aux[1] & 0xffff, row.aux[1] >> 16);
            // The source's pixels sit at the node origin minus the source
            // offset; nothing past the source's extent is painted.
            let placed = Rect {
                x: shape.x - src_x as i32,
                y: shape.y - src_y as i32,
                w: surface.width().min(u32::from(u16::MAX)) as u16,
                h: surface.height().min(u32::from(u16::MAX)) as u16,
            };
            let piece = piece.intersect(&placed).intersect(&shape);
            if piece.w == 0 || piece.h == 0 {
                return Ok(false);
            }
            let opacity = row.opacity.min(255) as u8;
            if radii == [0; 4] {
                blit_plain(&surface, placed, piece, opacity, target);
                return Ok(true);
            }
            let (bands, n) = rounded::bands(shape, radii);
            for band in &bands[..n] {
                let part = band.intersect(&piece);
                if part.w != 0 && part.h != 0 {
                    blit_plain(&surface, placed, part, opacity, target);
                }
            }
            for corner in rounded::corner_boxes(shape, radii) {
                let part = corner.rect.intersect(&piece);
                if part.w != 0 && part.h != 0 {
                    blit_masked(&surface, placed, shape, corner.radius, part, target);
                }
            }
            Ok(true)
        }
        RasterOp::SurfaceSample => {
            let Some(surface) = source(row.aux[0]) else {
                return Ok(false);
            };
            let (sx, sy) = (row.aux[1] & 0xffff, row.aux[1] >> 16);
            let (sw, sh) = (row.aux[2] & 0xffff, row.aux[2] >> 16);
            Ok(sample_nearest(
                &surface, shape, piece, sx, sy, sw, sh, target,
            ))
        }
        RasterOp::Glyphs => {
            let (Some(atlas), Some(table)) = (atlas, glyphs) else {
                return Ok(false);
            };
            let start = row.aux[0] as usize;
            let Some(run) = start
                .checked_add(row.aux[1] as usize)
                .and_then(|end| table.glyphs().get(start..end))
            else {
                return Ok(false);
            };
            let opacity = row.opacity.min(255) as u8;
            let mut painted = false;
            for glyph in run {
                painted |= blend_glyph(&atlas, glyph, shape, piece, row.argb, opacity, target);
            }
            Ok(painted)
        }
    }
}

/// Longest row run staged on the stack for a blit.
const ROW_CHUNK: usize = 256;

/// Blit `part` of `surface` (placed at `placed`) onto the target, opaque
/// or through the source's own alpha below full opacity.
fn blit_plain(
    surface: &XrgbSurfaceRef<'_>,
    placed: Rect,
    part: Rect,
    opacity: u8,
    target: &XrgbTargetRef<'_>,
) {
    let mut chunk = [0u32; ROW_CHUNK];
    // SAFETY: `part` is inside the target; one node paints at a time.
    let mut dst = unsafe { target.view(part) };
    for dy in 0..usize::from(part.h) {
        let sy = (part.y - placed.y) as u32 + dy as u32;
        let mut done = 0usize;
        while done < usize::from(part.w) {
            let n = (usize::from(part.w) - done).min(ROW_CHUNK);
            let sx = (part.x - placed.x) as u32 + done as u32;
            surface.read_row(sx, sy, &mut chunk[..n]);
            let row = dst.row_mut(dy, done, done + n);
            if opacity == 255 {
                row.copy_from_slice(&chunk[..n]);
            } else {
                kernels::blit_blend(&chunk[..n], n, row, n, n, 1);
            }
            done += n;
        }
    }
}

/// Blit `part` of `surface` through the coverage of `shape`'s corner arc.
fn blit_masked(
    surface: &XrgbSurfaceRef<'_>,
    placed: Rect,
    shape: Rect,
    radius: u16,
    part: Rect,
    target: &XrgbTargetRef<'_>,
) {
    let mut chunk = [0u32; MASK_SIDE];
    let mut mask = [0u8; MASK_SIDE];
    let mut dst = unsafe { target.view(part) };
    for dy in 0..usize::from(part.h) {
        let y = part.y + dy as i32;
        let sy = (y - placed.y) as u32;
        let mut done = 0usize;
        while done < usize::from(part.w) {
            let n = (usize::from(part.w) - done).min(MASK_SIDE);
            let x0 = part.x + done as i32;
            surface.read_row((x0 - placed.x) as u32, sy, &mut chunk[..n]);
            for (i, m) in mask[..n].iter_mut().enumerate() {
                *m = rounded::corner_coverage(shape, radius, x0 + i as i32, y);
            }
            kernels::blend_row_masked(&chunk[..n], &mask[..n], dst.row_mut(dy, done, done + n));
            done += n;
        }
    }
}

/// Widest source region a scaled sample stages per row.
const SAMPLE_ROW: usize = 4096;

/// Nearest-neighbour sample of `(sx, sy, sw, sh)` of `surface` scaled to
/// the node's `shape`, restricted to `piece`, alpha-blended.
#[allow(clippy::too_many_arguments)]
fn sample_nearest(
    surface: &XrgbSurfaceRef<'_>,
    shape: Rect,
    piece: Rect,
    sx: u32,
    sy: u32,
    sw: u32,
    sh: u32,
    target: &XrgbTargetRef<'_>,
) -> bool {
    let (orig_w, orig_h) = (usize::from(shape.w), usize::from(shape.h));
    if sw == 0 || sh == 0 || orig_w == 0 || orig_h == 0 {
        return false;
    }
    // Keep the transform anchored to the original node. Rounding a clipped
    // source rectangle and rescaling it again changes pixels across damage
    // partitions (and can reduce a one-pixel upscale fragment to zero).
    if piece.w == 0
        || piece.h == 0
        || shape.intersect(&piece) != piece
        || u64::from(sx) + u64::from(sw) > u64::from(surface.width())
        || u64::from(sy) + u64::from(sh) > u64::from(surface.height())
    {
        return false;
    }
    let clip_left = (piece.x - shape.x) as usize;
    let clip_top = (piece.y - shape.y) as usize;
    let (w, h) = (usize::from(piece.w), usize::from(piece.h));
    let first = clip_left * sw as usize / orig_w;
    let last = (clip_left + w - 1) * sw as usize / orig_w;
    let source_width = last - first + 1;
    if source_width > SAMPLE_ROW {
        return false;
    }
    let mut src_row = [0u32; SAMPLE_ROW];
    let mut gathered = [0u32; ROW_CHUNK];
    let mut dst = unsafe { target.view(piece) };
    for dy in 0..h {
        let source_y = sy as usize + (clip_top + dy) * sh as usize / orig_h;
        surface.read_row(
            sx + first as u32,
            source_y as u32,
            &mut src_row[..source_width],
        );
        let mut done = 0;
        while done < w {
            let count = (w - done).min(ROW_CHUNK);
            for (group, chunk) in gathered[..count].chunks_mut(8).enumerate() {
                let x =
                    Simd::from_array(core::array::from_fn(|i| clip_left + done + group * 8 + i));
                let indices =
                    x * Simd::splat(sw as usize) / Simd::splat(orig_w) - Simd::splat(first);
                let pixels =
                    Simd::<u32, 8>::gather_or_default(&src_row[..source_width], indices).to_array();
                chunk.copy_from_slice(&pixels[..chunk.len()]);
            }
            kernels::blit_blend(
                &gathered[..count],
                count,
                dst.row_mut(dy, done, done + count),
                count,
                count,
                1,
            );
            done += count;
        }
    }
    true
}

/// Tint one glyph's atlas coverage with `color` at `opacity` inside `piece`.
fn blend_glyph(
    atlas: &CoverageAtlasRef<'_>,
    glyph: &PlacedGlyph,
    shape: Rect,
    piece: Rect,
    color: u32,
    opacity: u8,
    target: &XrgbTargetRef<'_>,
) -> bool {
    if glyph.aw == 0 || glyph.ah == 0 {
        return false;
    }
    let gx = shape.x + i32::from(glyph.dx);
    let gy = shape.y + i32::from(glyph.dy);
    let rect = Rect {
        x: gx,
        y: gy,
        w: glyph.aw,
        h: glyph.ah,
    }
    .intersect(&piece);
    if rect.w == 0 || rect.h == 0 {
        return false;
    }
    let clip_dx = (rect.x - gx) as usize;
    let clip_dy = (rect.y - gy) as usize;
    let ax = usize::from(glyph.ax) + clip_dx;
    let ay = usize::from(glyph.ay) + clip_dy;
    let (w, h) = (usize::from(rect.w), usize::from(rect.h));
    if ax + w > atlas.width as usize || ay + h > atlas.height as usize {
        return false;
    }
    let pitch = atlas.width as usize;
    let mut dst = unsafe { target.view(rect) };
    for dy in 0..h {
        let row = dst.row_mut(dy, 0, w);
        let start = (ay + dy) * pitch + ax;
        match atlas.format {
            CoverageFormat::U8 => {
                kernels::coverage_blend_u8(&atlas.bytes[start..start + w], color, opacity, row)
            }
            CoverageFormat::U16 => {
                let mut cov = [0u16; MASK_SIDE];
                let mut done = 0;
                while done < w {
                    let n = (w - done).min(MASK_SIDE);
                    for (i, c) in cov[..n].iter_mut().enumerate() {
                        let at = (start + done + i) * 2;
                        *c = u16::from_le_bytes([atlas.bytes[at], atlas.bytes[at + 1]]);
                    }
                    kernels::coverage_blend_u16(
                        &cov[..n],
                        color,
                        opacity,
                        &mut row[done..done + n],
                    );
                    done += n;
                }
            }
            CoverageFormat::PackedAlpha => {
                let mut cov = [0u32; MASK_SIDE];
                let mut done = 0;
                while done < w {
                    let n = (w - done).min(MASK_SIDE);
                    for (i, c) in cov[..n].iter_mut().enumerate() {
                        let at = (start + done + i) * 4;
                        *c = u32::from_le_bytes([
                            atlas.bytes[at],
                            atlas.bytes[at + 1],
                            atlas.bytes[at + 2],
                            atlas.bytes[at + 3],
                        ]);
                    }
                    kernels::coverage_blend_a32(
                        &cov[..n],
                        color,
                        opacity,
                        &mut row[done..done + n],
                    );
                    done += n;
                }
            }
        }
    }
    true
}

/// Clamp the row's `[u16; 4]` radii to the shape (kept as a free helper so
/// the lowering and the kernel agree).
pub fn clamped_radii(shape: Rect, radii: Radii) -> Radii {
    rounded::clamped(shape, radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use axnn::lowering::ResourceKey;
    use axnn::{TensorDesc, shape};

    const W: usize = 96;
    const H: usize = 32;
    const BG: u32 = 0xFF10_1014;

    fn node(row: u32) -> RasterNodeParams {
        span(row, 1)
    }

    fn span(row: u32, rows: u32) -> RasterNodeParams {
        RasterNodeParams::new(
            row,
            rows,
            ResourceKey {
                group: 2,
                binding: 0,
            },
            3,
            ResourceKey {
                group: 4,
                binding: 0,
            },
            ResourceKey {
                group: 4,
                binding: 1,
            },
            16,
        )
        .unwrap()
    }

    fn plan(op: RasterOp, rows: usize, row: u32) -> CpuKernelPlan {
        plan_span(op, rows, span(row, 1))
    }

    fn plan_span(op: RasterOp, rows: usize, node: RasterNodeParams) -> CpuKernelPlan {
        let request = KernelRequest::raster(
            op,
            &TensorDesc::try_new(shape![rows, RASTER_ROW_WORDS], DType::I32).unwrap(),
            &TensorDesc::try_new(shape![W / 16], DType::U8).unwrap(),
            node,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    /// One node painting two rows paints them in order: the second row's
    /// fill lands over the first's where they overlap, an empty row in the
    /// range paints nothing, and the token says whether anything landed.
    #[test]
    fn sampled_pixels_do_not_depend_on_damage_partition() {
        let pixels: [u32; 72 * 70] = core::array::from_fn(|i| {
            if i % 3 == 0 {
                0x8080_0000
            } else {
                0xff00_0000 | (i as u32 * 317 & 0x00ff_ffff)
            }
        });
        let source = XrgbSurfaceRef::new(&pixels, 72, 70, 72).unwrap();
        let shape = Rect {
            x: 13,
            y: 5,
            w: 48,
            h: 47,
        };
        for (sw, sh) in [(64, 64), (3, 3)] {
            let mut full = [BG; 96 * 64];
            let mut parts = full;
            let target = XrgbTargetRef::new(&mut full, (0, 0), 96, 64, 96).unwrap();
            assert!(sample_nearest(&source, shape, shape, 3, 2, sw, sh, &target));
            let target = XrgbTargetRef::new(&mut parts, (0, 0), 96, 64, 96).unwrap();
            for y in (0..47).step_by(7) {
                for x in (0..48).step_by(8) {
                    let piece = Rect {
                        x: 13 + x,
                        y: 5 + y,
                        w: 8,
                        h: (47 - y).min(7) as u16,
                    };
                    assert!(sample_nearest(&source, shape, piece, 3, 2, sw, sh, &target));
                }
            }
            assert_eq!(full, parts, "clipping must not alter sampling phase");
        }
    }

    #[test]
    fn a_node_paints_its_rows_in_order() {
        let mut fb = [BG; W * H];
        let fill = |x, y, w, h, argb| RasterRow {
            x,
            y,
            w,
            h,
            vx: x,
            vy: y,
            vw: w,
            vh: h,
            radii: [0; 4],
            argb,
            opacity: 255,
            aux: [0; 3],
            tiles: [u64::MAX; 2],
        };
        let mut words = [0i32; RASTER_ROW_WORDS * 4];
        fill(0, 0, 32, 8, 0xFFFF_0000).write(&mut words[RASTER_ROW_WORDS..2 * RASTER_ROW_WORDS]);
        // Row 2 stays empty; row 3 lands over row 1's right half.
        fill(16, 0, 32, 8, 0xFF00_00FF).write(&mut words[3 * RASTER_ROW_WORDS..]);
        let dirty = [1u8; W / 16];
        let mut token = [7i32];
        let target = XrgbTargetRef::new(&mut fb, (0, 0), W as u32, H as u32, W).unwrap();
        invoke_raster(
            &plan_span(RasterOp::Fill, 4, span(1, 3)),
            &words,
            &dirty,
            target,
            &|_| None,
            None,
            None,
            &mut token,
        )
        .unwrap();
        assert_eq!(token, [1]);
        assert_eq!(fb[4] & 0x00FF_FFFF, 0x00FF_0000, "row 1 alone on the left");
        assert_eq!(fb[20] & 0x00FF_FFFF, 0x0000_00FF, "row 3 over row 1");
        assert_eq!(
            fb[40] & 0x00FF_FFFF,
            0x0000_00FF,
            "row 3 alone on the right"
        );
        assert_eq!(fb[60], BG, "past both");
        // A range past the table is refused, not read: the request itself
        // refuses a node whose range leaves its table, and the kernel
        // refuses a table shorter than the plan's.
        assert!(
            KernelRequest::raster(
                RasterOp::Fill,
                &TensorDesc::try_new(shape![4, RASTER_ROW_WORDS], DType::I32).unwrap(),
                &TensorDesc::try_new(shape![W / 16], DType::U8).unwrap(),
                span(2, 3),
            )
            .is_none()
        );
        let mut fb2 = [BG; W * H];
        let target = XrgbTargetRef::new(&mut fb2, (0, 0), W as u32, H as u32, W).unwrap();
        assert!(
            invoke_raster(
                &plan_span(RasterOp::Fill, 5, span(2, 3)),
                &words,
                &dirty,
                target,
                &|_| None,
                None,
                None,
                &mut token,
            )
            .is_err()
        );
    }

    /// Rows painted by their own ops land exactly as a program's planned
    /// nodes paint them: a fill, a row no node paints (left alone though its
    /// words hold a shape), a ring over the fill and a fill over both.
    #[test]
    fn rows_paint_by_their_own_op_as_planned_nodes_do() {
        let row = |x: i32, w: u32, argb: u32, ring: u32| RasterRow {
            x,
            y: 2,
            w,
            h: 12,
            vx: x,
            vy: 2,
            vw: w,
            vh: 12,
            radii: [0; 4],
            argb,
            opacity: 255,
            aux: [ring, 0, 0],
            tiles: [u64::MAX; 2],
        };
        let mut words = [0i32; RASTER_ROW_WORDS * 4];
        row(0, 40, 0xFFFF_0000, 0).write(&mut words[..RASTER_ROW_WORDS]);
        row(60, 30, 0xFF00_FF00, 0).write(&mut words[RASTER_ROW_WORDS..2 * RASTER_ROW_WORDS]);
        row(8, 40, 0xFF00_00FF, 3).write(&mut words[2 * RASTER_ROW_WORDS..3 * RASTER_ROW_WORDS]);
        row(30, 20, 0xFFFF_FFFF, 0).write(&mut words[3 * RASTER_ROW_WORDS..]);
        let ops = [
            Some(RasterOp::Fill),
            None,
            Some(RasterOp::Ring),
            Some(RasterOp::Fill),
        ];
        let dirty = [1u8; W / 16];

        let mut by_rows = [BG; W * H];
        let target = XrgbTargetRef::new(&mut by_rows, (0, 0), W as u32, H as u32, W).unwrap();
        assert!(paint_rows(&ops, &words, 16, &dirty, target, &|_| None, None, None).unwrap());

        let mut by_nodes = [BG; W * H];
        let mut token = [0i32];
        for (op, row) in [
            (RasterOp::Fill, 0),
            (RasterOp::Ring, 2),
            (RasterOp::Fill, 3),
        ] {
            let target = XrgbTargetRef::new(&mut by_nodes, (0, 0), W as u32, H as u32, W).unwrap();
            invoke_raster(
                &plan(op, 4, row),
                &words,
                &dirty,
                target,
                &|_| None,
                None,
                None,
                &mut token,
            )
            .unwrap();
        }
        assert_eq!(by_rows, by_nodes);
        assert_ne!(by_rows[4 * W + 4], BG, "the fill landed");
        assert_eq!(
            by_rows[4 * W + 70],
            BG,
            "the row no node paints stays unpainted"
        );

        // A table shorter than its ops, or no tile width, is refused, not read.
        let target = XrgbTargetRef::new(&mut by_rows, (0, 0), W as u32, H as u32, W).unwrap();
        let short = &words[..3 * RASTER_ROW_WORDS];
        assert!(paint_rows(&ops, short, 16, &dirty, target, &|_| None, None, None).is_err());
        let target = XrgbTargetRef::new(&mut by_rows, (0, 0), W as u32, H as u32, W).unwrap();
        assert!(paint_rows(&ops, &words, 0, &dirty, target, &|_| None, None, None).is_err());
    }

    fn table(row: RasterRow) -> [i32; RASTER_ROW_WORDS * 2] {
        let mut words = [0i32; RASTER_ROW_WORDS * 2];
        row.write(&mut words[RASTER_ROW_WORDS..]);
        words
    }

    #[test]
    fn a_rounded_fill_lands_on_its_dirty_tiles_only() {
        let mut fb = [BG; W * H];
        let row = RasterRow {
            x: 4,
            y: 4,
            w: 80,
            h: 24,
            vx: 4,
            vy: 4,
            vw: 80,
            vh: 24,
            radii: [6, 6, 6, 6],
            argb: 0xFF20_A040,
            opacity: 255,
            aux: [0; 3],
            tiles: [0b1111_0, 0],
            ..RasterRow::EMPTY
        };
        let words = table(row);
        let dirty = [1u8, 1, 1, 0, 1, 1];
        let plan = plan(RasterOp::Fill, 2, 1);
        let mut token = [0i32];
        let target = XrgbTargetRef::new(&mut fb, (0, 0), W as u32, H as u32, W).unwrap();
        invoke_raster(
            &plan,
            &words,
            &dirty,
            target,
            &|_| None,
            None,
            None,
            &mut token,
        )
        .unwrap();
        assert_eq!(token, [1]);
        // Column 0 is off the tile mask, column 3 is clean: both untouched.
        assert_eq!(fb[10 * W + 8], BG);
        assert_eq!(fb[10 * W + 50], BG);
        // Columns 1, 2 and 4 painted; the arc spares the corner pixel.
        assert_eq!(fb[10 * W + 20], 0xFF20_A040);
        assert_eq!(fb[10 * W + 70], 0xFF20_A040);
        assert_eq!(fb[4 * W + 4], BG, "the corner is outside the r=6 arc");
        // Nothing outside the shape.
        assert_eq!(fb[0], BG);
        assert_eq!(fb[31 * W + 20], BG);
    }

    #[test]
    fn a_surface_blit_reads_the_bound_slot_and_respects_its_extent() {
        let mut fb = [BG; W * H];
        let src: [u32; 20 * 8] = core::array::from_fn(|i| 0xFF00_0000 | i as u32);
        let surface = XrgbSurfaceRef::new(&src, 20, 8, 20).unwrap();
        let row = RasterRow {
            x: 10,
            y: 3,
            w: 40,
            h: 20,
            vx: 10,
            vy: 3,
            vw: 40,
            vh: 20,
            argb: 0,
            opacity: 255,
            aux: [7, 0, 0],
            tiles: [u64::MAX, u64::MAX],
            ..RasterRow::EMPTY
        };
        let words = table(row);
        let dirty = [1u8; W / 16];
        let plan = plan(RasterOp::SurfaceBlit, 2, 1);
        let mut token = [0i32];
        let target = XrgbTargetRef::new(&mut fb, (0, 0), W as u32, H as u32, W).unwrap();
        let sources = |slot: u32| (slot == 7).then_some(surface);
        invoke_raster(
            &plan, &words, &dirty, target, &sources, None, None, &mut token,
        )
        .unwrap();
        assert_eq!(token, [1]);
        assert_eq!(fb[3 * W + 10], 0xFF00_0000);
        assert_eq!(fb[4 * W + 12], 0xFF00_0000 | 22);
        assert_eq!(fb[3 * W + 30], BG, "beyond the 20-wide source");
        assert_eq!(fb[11 * W + 10], BG, "beyond the 8-tall source");
        // An unbound slot paints nothing and says so.
        let mut fb2 = [BG; W * H];
        let target = XrgbTargetRef::new(&mut fb2, (0, 0), W as u32, H as u32, W).unwrap();
        invoke_raster(
            &plan,
            &words,
            &dirty,
            target,
            &|_| None,
            None,
            None,
            &mut token,
        )
        .unwrap();
        assert_eq!(token, [0]);
        assert!(fb2.iter().all(|&p| p == BG));
    }

    #[test]
    fn a_band_target_translates_frame_rows() {
        // A 16-row band standing for frame rows 16..32 of a 96-wide frame.
        let mut band = [BG; W * 16];
        let row = RasterRow {
            x: 0,
            y: 20,
            w: 96,
            h: 40,
            vx: 0,
            vy: 16,
            vw: 96,
            vh: 16,
            argb: 0xFFFF_FFFF,
            opacity: 255,
            tiles: [u64::MAX, 0],
            ..RasterRow::EMPTY
        };
        let words = table(row);
        let dirty = [1u8; W / 16];
        let plan = plan(RasterOp::Fill, 2, 1);
        let mut token = [0i32];
        let target = XrgbTargetRef::new(&mut band, (0, 16), W as u32, 16, W).unwrap();
        invoke_raster(
            &plan,
            &words,
            &dirty,
            target,
            &|_| None,
            None,
            None,
            &mut token,
        )
        .unwrap();
        assert_eq!(token, [1]);
        // Frame rows 16..20 are above the shape; 20..32 inside it.
        assert_eq!(band[3 * W + 5], BG);
        assert_eq!(band[4 * W + 5], 0xFFFF_FFFF);
        assert_eq!(band[15 * W + 90], 0xFFFF_FFFF);
    }

    #[test]
    fn volatile_rows_snapshot_every_word() {
        let src: [u32; 300] = core::array::from_fn(|i| i as u32 * 3);
        // SAFETY: a plain array read as a volatile surface.
        let surface = unsafe { XrgbSurfaceRef::from_raw_volatile(src.as_ptr(), 300, 1, 300) };
        for (x, len) in [(0, 300), (1, 299), (5, 17), (7, 1), (200, 100)] {
            let mut out = [0u32; 300];
            surface.read_row(x, 0, &mut out[..len]);
            assert_eq!(&out[..len], &src[x as usize..x as usize + len]);
        }
    }
}
