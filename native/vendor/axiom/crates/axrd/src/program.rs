//! The scene as a raster program: one [`ax::raster`] node per run of
//! same-kind scene nodes in the scene's paint order, built once per
//! screen, and the per-frame lowering that fills each slot's parameter
//! row for one band.
//!
//! The program is static — a node's opcode is fixed by its bind's kind at
//! build time, its row of the parameter table by its paint-order position
//! — and everything that changes per frame travels in the rows
//! ([`RasterRow`]): the shape and visible rect, the radii, the colour and
//! opacity, the source slot, and the tile columns the node paints once
//! damage and occlusion are resolved.  So a compositor prepares the program
//! once, forks it per worker, and per band hands a worker the rows the
//! lowering wrote; the executor (a CPU lane's raster kernels, or an
//! accelerator's native blits) paints the frame.  This is the crate's one
//! raster path: [`crate::host::SceneHost`] runs it on the host for
//! examples and tests, a compositor on the lane it founds.
//!
//! A scene whose node kinds and paint order change from frame to frame —
//! a document, whose elements come and go and restyle — has no program
//! to build once: a rebuild per change costs more than the frame.  It
//! takes the same path with its ops as per-frame data: [`raster_ops`]
//! reads the frame's op table, [`TiledFrame::lower_band_dynamic`] lowers
//! a band against it with the same culling into the same rows, and the
//! executor dispatches each row by its own op (`axnn_cpu::paint_rows`)
//! where a program dispatches runs of rows by planned nodes.  The row
//! kernels are shared, so both paint the same pixels.

use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::raster::{RasterNodeParams, RasterOp, ResourceKey};
use ax::{DType, GraphBuf, GraphBuilder, TensorFlags};
use axnn::{RASTER_ROW_WORDS, RasterRow, Rect};

use axnn::raster::{clamped, corner_boxes};

use crate::bind::{Bind, Corners};
use crate::scene::{Node, NodeFlags, Scene};
use crate::tile::{MAX_TILE_ROWS, TiledFrame};

/// The surface slot a [`Bind::Blit`] / [`Bind::Sample`] node samples: the
/// frame's one `src` surface, bound at this slot of the program's surface
/// group.
pub const BLIT_SOURCE_SLOT: u32 = u32::MAX;

/// Where a program's resources are bound at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramBindings {
    /// The packed-XRGB target (a band stage or a whole framebuffer).
    pub target: ResourceKey,
    /// The resource group client surfaces are bound in, by
    /// [`Bind::Surface`] slot.
    pub surfaces: u32,
    /// The glyph coverage atlas.
    pub atlas: ResourceKey,
    /// The placed-glyph table.
    pub glyphs: ResourceKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramError {
    Allocation,
    /// The graph the scene lowered to was refused by the `.ax` format.
    Format(ax::AxError),
    /// The scene has no node a raster op could paint.
    Empty,
    /// A node's bind kind, or the scene's paint order, changed since the
    /// program (or the op table) was built: rebuild it.
    Drift,
    /// The frame divides into more tile columns than a row's mask
    /// addresses, or the tile width is zero.
    Columns,
    /// The caller's row table or dirty vector is too small.
    Capacity,
}

impl From<ax::AllocationError> for ProgramError {
    fn from(error: ax::AllocationError) -> Self {
        match error {
            ax::AllocationError::Format(error) => Self::Format(error),
            _ => Self::Allocation,
        }
    }
}

/// The raster op a bind lowers to; a [`Bind::Composite`] paints nothing.
pub const fn raster_op_of(bind: Bind) -> Option<RasterOp> {
    match bind {
        Bind::Fill { .. } | Bind::RoundedFill { .. } => Some(RasterOp::Fill),
        Bind::Border { .. } => Some(RasterOp::Ring),
        Bind::Shadow { .. } => Some(RasterOp::Shadow),
        Bind::Blit { .. } | Bind::Surface { .. } => Some(RasterOp::SurfaceBlit),
        Bind::Sample { .. } => Some(RasterOp::SurfaceSample),
        Bind::TextRun { .. } => Some(RasterOp::Glyphs),
        Bind::Composite => None,
    }
}

/// Write the op of each of `scene`'s paint-order positions to the front
/// of `ops` and return that prefix: a resolved scene's op table, the one a
/// program is built from and the one a scene with no program lowers
/// against per frame ([`TiledFrame::lower_band_dynamic`]).  `Capacity`
/// when `ops` is shorter than the paint order.
pub fn raster_ops<'o, const N: usize>(
    scene: &Scene<N>,
    ops: &'o mut [Option<RasterOp>],
) -> Result<&'o [Option<RasterOp>], ProgramError> {
    let order = scene.order();
    let ops = ops.get_mut(..order.len()).ok_or(ProgramError::Capacity)?;
    for (op, &idx) in ops.iter_mut().zip(order) {
        *op = raster_op_of(scene.node(idx).bind);
    }
    Ok(ops)
}

/// One screen's scene as a program: the graph, and per paint-order
/// position the op its node was built with.
pub struct SceneProgram<A: Allocator> {
    graph: GraphBuf<A>,
    ops: Vec<Option<RasterOp>, A>,
    bindings: ProgramBindings,
    tile_width: u32,
    columns: usize,
}

impl<A: Allocator + Clone> SceneProgram<A> {
    /// Build the program for a resolved `scene` (its paint order is the
    /// row order) on a frame `columns` tiles of `tile_width` pixels wide.
    /// The graph's inputs are the I32 `[rows, RASTER_ROW_WORDS]` parameter
    /// table and the U8 `[columns]` dirty vector; its output the last
    /// node's token.
    pub fn build_in<const N: usize>(
        scene: &Scene<N>,
        bindings: ProgramBindings,
        tile_width: u32,
        columns: usize,
        allocator: A,
    ) -> Result<Self, ProgramError> {
        if columns == 0
            || columns > RasterNodeParams::MAX_TILE_COLUMNS
            || tile_width == 0
            || tile_width > u32::from(u16::MAX)
        {
            return Err(ProgramError::Columns);
        }
        let mut ops = Vec::new_in(allocator.clone());
        ops.try_reserve_exact(scene.order().len())
            .map_err(|_| ProgramError::Allocation)?;
        ops.resize(scene.order().len(), None);
        raster_ops(scene, &mut ops)?;
        let Some(last) = ops.iter().rposition(Option::is_some) else {
            return Err(ProgramError::Empty);
        };
        let mut graph = GraphBuilder::new_in(allocator);
        let params = graph.tensor(
            DType::I32,
            TensorFlags::INPUT,
            &[ops.len() as u32, RASTER_ROW_WORDS as u32],
            &[],
        )?;
        let dirty = graph.tensor(DType::U8, TensorFlags::INPUT, &[columns as u32], &[])?;
        // One node per run of consecutive same-kind rows: the executor
        // walks nodes, the kernel walks rows, and a scene of hundreds of
        // slots becomes a program of tens of nodes.
        let mut row = 0;
        while row < ops.len() {
            let Some(op) = ops[row] else {
                row += 1;
                continue;
            };
            let mut end = row + 1;
            while end < ops.len() && ops[end] == Some(op) {
                end += 1;
            }
            let node = RasterNodeParams::new(
                row as u32,
                (end - row) as u32,
                bindings.target,
                bindings.surfaces,
                bindings.atlas,
                bindings.glyphs,
                tile_width,
            )
            .ok_or(ProgramError::Columns)?;
            let flags = if end > last {
                TensorFlags::OUTPUT
            } else {
                TensorFlags::default()
            };
            graph.raster(op, params, dirty, node, flags)?;
            row = end;
        }
        Ok(Self {
            graph: graph.finish()?,
            ops,
            bindings,
            tile_width,
            columns,
        })
    }
}

impl<A: Allocator> SceneProgram<A> {
    pub fn graph(&self) -> &GraphBuf<A> {
        &self.graph
    }

    pub fn into_graph(self) -> GraphBuf<A> {
        self.graph
    }

    /// Rows of the parameter table (one per paint-order position).
    pub fn rows(&self) -> usize {
        self.ops.len()
    }

    /// The op painting `row`, or `None` for a row no node paints.
    pub fn op(&self, row: usize) -> Option<RasterOp> {
        self.ops.get(row).copied().flatten()
    }

    /// Words the parameter table holds.
    pub fn table_words(&self) -> usize {
        self.ops.len() * RASTER_ROW_WORDS
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn tile_width(&self) -> u32 {
        self.tile_width
    }

    pub const fn bindings(&self) -> ProgramBindings {
        self.bindings
    }
}

impl<const N: usize, const TW: usize, const TH: usize> TiledFrame<'_, N, TW, TH> {
    /// Lower band `tr` into `program`'s parameter rows and the band's
    /// dirty vector: every node reaching a dirty tile of the band that is
    /// not hidden under a node above it gets its row filled and its tile
    /// bits set; every other row is zero.  `surface_extent(id)` reports a
    /// [`Bind::Surface`]'s `(width, height)` for the occlusion test.
    /// Returns the union of the columns some node paints (bit `c` =
    /// column `c`), all zero when the band has nothing to paint.
    pub fn lower_band<A: Allocator>(
        &self,
        program: &SceneProgram<A>,
        tr: usize,
        surface_extent: &dyn Fn(u16) -> Option<(u32, u32)>,
        rows: &mut [i32],
        dirty: &mut [u8],
    ) -> Result<[u64; 2], ProgramError> {
        if program.tile_width as usize != TW {
            return Err(ProgramError::Drift);
        }
        self.lower_rows(
            &program.ops,
            program.columns,
            tr,
            surface_extent,
            rows,
            dirty,
        )
    }

    /// Lower band `tr` of a scene with no program, against `ops` — this
    /// frame's op table ([`raster_ops`] of the scene the frame read) — into
    /// `ops.len()` parameter rows and the band's dirty vector of the
    /// frame's [`columns`](Self::columns), exactly as
    /// [`lower_band`](Self::lower_band) lowers for a program: the same
    /// culling, the same rows, the same union of painted columns.  An
    /// executor paints row `i` by `ops[i]` over columns `TW` pixels wide.
    pub fn lower_band_dynamic(
        &self,
        ops: &[Option<RasterOp>],
        tr: usize,
        surface_extent: &dyn Fn(u16) -> Option<(u32, u32)>,
        rows: &mut [i32],
        dirty: &mut [u8],
    ) -> Result<[u64; 2], ProgramError> {
        if TW > usize::from(u16::MAX) {
            return Err(ProgramError::Columns);
        }
        self.lower_rows(ops, self.cols, tr, surface_extent, rows, dirty)
    }

    /// The lowering both forms share: `ops[i]` paints paint-order
    /// position `i`, and `dirty` addresses `columns` tile columns.  Every
    /// position the band reaches must still lower to its op — a kind
    /// changed either way, a paintable node become a group or a group
    /// become paintable, is [`ProgramError::Drift`], refused before any
    /// row is written, never a node silently left unpainted.
    fn lower_rows(
        &self,
        ops: &[Option<RasterOp>],
        columns: usize,
        tr: usize,
        surface_extent: &dyn Fn(u16) -> Option<(u32, u32)>,
        rows: &mut [i32],
        dirty: &mut [u8],
    ) -> Result<[u64; 2], ProgramError> {
        let order = self.scene.order();
        if ops.len() != order.len() {
            return Err(ProgramError::Drift);
        }
        if self.cols > columns || self.cols > RasterNodeParams::MAX_TILE_COLUMNS {
            return Err(ProgramError::Columns);
        }
        let words = ops.len() * RASTER_ROW_WORDS;
        if rows.len() < words || dirty.len() < columns {
            return Err(ProgramError::Capacity);
        }
        // `RasterRow::is_empty` makes zero visible width the row's inert
        // discriminator. Clear that one word for every slot; reached rows are
        // then fully overwritten below before any painter observes them.
        for row in rows[..words].chunks_exact_mut(RASTER_ROW_WORDS) {
            row[6] = 0;
        }
        dirty[..columns].fill(0);
        let mut painted = [0u64; 2];
        if tr >= self.rows {
            return Ok(painted);
        }
        let row_bits = self.row_mask.row(tr);
        if row_bits.iter().all(|&w| w == 0) {
            return Ok(painted);
        }
        // Refuse a drifted bind before any row is written: the tile masks
        // below are scratch in `rows`, so this runs over every candidate
        // the band could reach, occluded or not.
        for (word, &bits) in row_bits.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let i = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if raster_op_of(self.scene.node(order[i]).bind) != ops[i] {
                    return Err(ProgramError::Drift);
                }
            }
        }
        // Reuse inert parameter storage for dynamic node populations. No
        // scratch is needed when the band has no candidates at all.
        for row in rows[..words].chunks_exact_mut(RASTER_ROW_WORDS) {
            row[..4].fill(0);
        }
        let band_h = TH.min(self.fb_h - tr * TH);
        let band = Rect {
            x: 0,
            y: (tr * TH) as i32,
            w: self.fb_w.min(usize::from(u16::MAX)) as u16,
            h: band_h as u16,
        };
        let mask = |rows: &[i32], i: usize| {
            let scratch = &rows[i * RASTER_ROW_WORDS..][..4];
            [
                u64::from(scratch[0] as u32) | (u64::from(scratch[1] as u32) << 32),
                u64::from(scratch[2] as u32) | (u64::from(scratch[3] as u32) << 32),
            ]
        };
        for tc in 0..self.cols {
            if !self.dirty.test(tc, tr, self.cols) {
                continue;
            }
            dirty[tc] = 1;
            let tile = Rect {
                x: (tc * TW) as i32,
                y: band.y,
                w: TW.min(self.fb_w - tc * TW) as u16,
                h: band_h as u16,
            };
            let hits = |i: usize| {
                let span = &self.spans[i];
                if tr < MAX_TILE_ROWS {
                    span.contains_col(tc)
                } else {
                    span.contains(tc, tr)
                }
            };
            // The topmost node that lays down every pixel of this tile
            // opaquely hides everything beneath it: rows below it get no
            // bit for this column.
            let mut start = 0;
            'cover: for (word, &bits) in row_bits.iter().enumerate().rev() {
                let mut bits = bits;
                while bits != 0 {
                    let bit = 63 - bits.leading_zeros() as usize;
                    bits &= !(1u64 << bit);
                    let i = word * 64 + bit;
                    if !hits(i) {
                        continue;
                    }
                    let idx = order[i];
                    if covers_opaquely(
                        self.scene.node(idx),
                        self.scene.rect(idx),
                        self.bounds[i],
                        tile,
                        surface_extent,
                    ) {
                        start = i;
                        break 'cover;
                    }
                }
            }
            for (word, &bits) in row_bits.iter().enumerate().skip(start / 64) {
                let mut bits = bits;
                if word == start / 64 {
                    bits &= !((1u64 << (start % 64)) - 1);
                }
                while bits != 0 {
                    let i = word * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if !hits(i) || ops[i].is_none() {
                        continue;
                    }
                    let vis = self.bounds[i].intersect(&tile);
                    if vis.w == 0 || vis.h == 0 {
                        continue;
                    }
                    rows[i * RASTER_ROW_WORDS + tc / 32] |= (1u32 << (tc % 32)) as i32;
                    painted[tc / 64] |= 1u64 << (tc % 64);
                }
            }
        }
        for (i, &idx) in order.iter().enumerate() {
            if mask(rows, i) == [0; 2] || ops[i].is_none() {
                continue;
            }
            let node = self.scene.node(idx);
            let shape = self.scene.rect(idx);
            let vis = self.bounds[i].intersect(&band);
            if vis.w == 0 || vis.h == 0 {
                continue;
            }
            let row = lower_node(node, shape, vis, mask(rows, i));
            row.write(&mut rows[i * RASTER_ROW_WORDS..(i + 1) * RASTER_ROW_WORDS]);
        }
        Ok(painted)
    }
}

fn radii_words(radii: Corners) -> [u16; 4] {
    [radii.tl, radii.tr, radii.br, radii.bl]
}

/// A node's aperture radii: its bind's own, else its rounded clip, else
/// square.
fn aperture(node: &Node) -> [u16; 4] {
    match node.bind {
        Bind::RoundedFill { radii, .. }
        | Bind::Border { radii, .. }
        | Bind::Shadow { radii, .. } => radii_words(radii),
        Bind::Fill { .. } | Bind::Surface { .. } | Bind::Blit { .. }
            if node.flags.contains(NodeFlags::CLIP_ROUNDED) =>
        {
            radii_words(node.radii)
        }
        _ => [0; 4],
    }
}

/// One node's row for a band: the executor sees the shape, the visible
/// rect, the aperture, the ink and the columns; the bind's kind is the
/// node's op.
fn lower_node(node: &Node, shape: Rect, vis: Rect, tiles: [u64; 2]) -> RasterRow {
    let mut row = RasterRow {
        x: shape.x,
        y: shape.y,
        w: u32::from(shape.w),
        h: u32::from(shape.h),
        vx: vis.x,
        vy: vis.y,
        vw: u32::from(vis.w),
        vh: u32::from(vis.h),
        radii: aperture(node),
        opacity: u32::from(node.opacity),
        tiles,
        ..RasterRow::EMPTY
    };
    match node.bind {
        Bind::Fill { color } | Bind::RoundedFill { color, .. } => {
            row.argb = premultiply(color, node.opacity);
        }
        Bind::Border { color, width, .. } => {
            row.argb = premultiply(color, node.opacity);
            row.aux[0] = u32::from(width);
        }
        Bind::Shadow { color, falloff, .. } => {
            row.argb = premultiply(color, node.opacity);
            row.aux[0] = u32::from(falloff);
        }
        Bind::Surface { id } => {
            row.aux[0] = u32::from(id);
        }
        Bind::Blit { src_x, src_y } => {
            row.aux[0] = BLIT_SOURCE_SLOT;
            row.aux[1] = u32::from(src_x) | (u32::from(src_y) << 16);
        }
        Bind::Sample { sx, sy, sw, sh } => {
            row.aux[0] = BLIT_SOURCE_SLOT;
            row.aux[1] = u32::from(sx) | (u32::from(sy) << 16);
            row.aux[2] = u32::from(sw) | (u32::from(sh) << 16);
        }
        Bind::TextRun {
            glyph_start,
            glyph_len,
            color,
        } => {
            row.argb = color;
            row.aux[0] = u32::from(glyph_start);
            row.aux[1] = u32::from(glyph_len);
        }
        Bind::Composite => {}
    }
    row
}

/// Whether `node` lays down every pixel of `tile` opaquely: `tile ⊆
/// bounds`, full opacity, a bind that paints every pixel it covers, and no
/// rounded corner box touching the tile.  A surface covers only inside the
/// extent its source reports; a blit or sample never claims to.
fn covers_opaquely(
    node: &Node,
    rect: Rect,
    bounds: Rect,
    tile: Rect,
    surface_extent: &dyn Fn(u16) -> Option<(u32, u32)>,
) -> bool {
    if node.opacity != 255 || !bounds.contains_rect(&tile) {
        return false;
    }
    let misses_corners = |radii: Corners| {
        let radii = clamped(rect, radii.array());
        radii == [0; 4]
            || corner_boxes(rect, radii)
                .iter()
                .all(|corner| !corner.rect.intersects(&tile))
    };
    let aperture_clear =
        || !node.flags.contains(NodeFlags::CLIP_ROUNDED) || misses_corners(node.shape_radii());
    match node.bind {
        Bind::Fill { .. } => aperture_clear(),
        Bind::RoundedFill { radii, .. } => misses_corners(radii),
        Bind::Surface { id } => {
            surface_extent(id).is_some_and(|(w, h)| {
                Rect {
                    x: rect.x,
                    y: rect.y,
                    w: w.min(u32::from(u16::MAX)) as u16,
                    h: h.min(u32::from(u16::MAX)) as u16,
                }
                .contains_rect(&tile)
            }) && aperture_clear()
        }
        _ => false,
    }
}

/// Premultiply an `0x00RRGGBB` colour by opacity, returning `0xAARRGGBB`
/// in premultiplied-alpha form suitable for src-over blending.
///
/// When `opacity == 255`, returns `0xFFRRGGBB` (no channel modification).
#[inline]
pub(crate) fn premultiply(color: u32, opacity: u8) -> u32 {
    if opacity == 255 {
        return color | 0xFF00_0000;
    }
    let a = opacity as u32;
    let r = ((color >> 16) & 0xFF) * a / 255;
    let g = ((color >> 8) & 0xFF) * a / 255;
    let b = (color & 0xFF) * a / 255;
    (a << 24) | (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Anchor, LayoutParams, Offset};
    use alloc::alloc::Global;

    #[test]
    fn premultiply_opaque_half_and_zero() {
        assert_eq!(premultiply(0x00FF8040, 255), 0xFFFF8040);
        let c = premultiply(0x00FF8040, 128);
        assert_eq!(c >> 24, 128);
        assert_eq!((c >> 16) & 0xFF, 0xFF * 128 / 255);
        assert_eq!((c >> 8) & 0xFF, 0x80 * 128 / 255);
        assert_eq!(c & 0xFF, 0x40 * 128 / 255);
        assert_eq!(premultiply(0x00FFFFFF, 0), 0);
    }

    fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
        LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset { x, y },
            w,
            h,
        }
    }

    fn bindings() -> ProgramBindings {
        ProgramBindings {
            target: ResourceKey {
                group: 1,
                binding: 0,
            },
            surfaces: 2,
            atlas: ResourceKey {
                group: 3,
                binding: 0,
            },
            glyphs: ResourceKey {
                group: 3,
                binding: 1,
            },
        }
    }

    #[test]
    fn a_program_has_one_node_per_paintable_scene_node_in_paint_order() {
        let mut scene = Scene::<8>::new();
        let root = scene
            .add_root(lp(0, 0, 96, 80), Bind::Fill { color: 0x10_1014 })
            .unwrap();
        let group = scene
            .add_child(root, lp(0, 0, 96, 80), Bind::Composite)
            .unwrap();
        scene
            .add_child(group, lp(4, 4, 40, 30), Bind::Surface { id: 3 })
            .unwrap();
        scene
            .add_child(
                root,
                lp(50, 10, 20, 20),
                Bind::RoundedFill {
                    color: 0xC0_5030,
                    radii: Corners::uniform(5),
                },
            )
            .unwrap();
        scene.resolve_layout_host(Rect {
            x: 0,
            y: 0,
            w: 96,
            h: 80,
        });
        let program = SceneProgram::build_in(&scene, bindings(), 16, 6, Global).unwrap();
        assert_eq!(program.rows(), 4);
        let graph = program.graph().parse().unwrap();
        // The composite group is a row without a node.
        assert_eq!(graph.node_count(), 3);
        assert_eq!(
            program.ops.as_slice(),
            &[
                Some(RasterOp::Fill),
                None,
                Some(RasterOp::SurfaceBlit),
                Some(RasterOp::Fill)
            ]
        );
        assert_eq!(
            SceneProgram::build_in(&scene, bindings(), 16, 0, Global).err(),
            Some(ProgramError::Columns)
        );
        let mut empty = Scene::<2>::new();
        empty.add_root(lp(0, 0, 8, 8), Bind::Composite).unwrap();
        empty.resolve_layout_host(Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        });
        assert_eq!(
            SceneProgram::build_in(&empty, bindings(), 16, 1, Global).err(),
            Some(ProgramError::Empty)
        );
    }

    #[test]
    fn lowering_fills_rows_for_reached_dirty_tiles_and_culls_under_covers() {
        let mut scene = Scene::<8>::new();
        let root = scene
            .add_root(lp(0, 0, 96, 32), Bind::Fill { color: 0x10_1014 })
            .unwrap();
        // An opaque surface over the left four tiles.
        scene
            .add_child(root, lp(0, 0, 64, 32), Bind::Surface { id: 0 })
            .unwrap();
        scene.resolve_layout_host(Rect {
            x: 0,
            y: 0,
            w: 96,
            h: 32,
        });
        let program = SceneProgram::build_in(&scene, bindings(), 16, 6, Global).unwrap();
        let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 32)).unwrap();
        let mut rows = [0i32; 8 * RASTER_ROW_WORDS];
        let mut dirty = [0u8; 6];
        let extent = |id: u16| (id == 0).then_some((64u32, 32u32));
        let painted = frame
            .lower_band(&program, 0, &extent, &mut rows, &mut dirty)
            .unwrap();
        assert_eq!(painted, [0b11_1111, 0]);
        assert_eq!(dirty, [1; 6]);
        let ground = RasterRow::read(&rows[..RASTER_ROW_WORDS]).unwrap();
        let surface = RasterRow::read(&rows[RASTER_ROW_WORDS..2 * RASTER_ROW_WORDS]).unwrap();
        // The ground paints only the two tiles the surface does not cover.
        assert_eq!(ground.tiles, [0b11_0000, 0]);
        assert_eq!(ground.argb, 0xFF10_1014);
        assert_eq!((ground.vx, ground.vy, ground.vw, ground.vh), (0, 0, 96, 16));
        assert_eq!(surface.tiles, [0b00_1111, 0]);
        assert_eq!(surface.aux[0], 0);
        assert_eq!(surface.opacity, 255);
        // A source that reports no extent covers nothing: the ground paints
        // every tile again.
        let painted = frame
            .lower_band(&program, 0, &|_| None, &mut rows, &mut dirty)
            .unwrap();
        assert_eq!(painted, [0b11_1111, 0]);
        assert_eq!(
            RasterRow::read(&rows[..RASTER_ROW_WORDS]).unwrap().tiles,
            [0b11_1111, 0]
        );
        // Past the last band every slot is inert. Other words deliberately
        // retain their previous values; zero visible width is the contract.
        let stale = rows;
        assert_eq!(
            frame
                .lower_band(&program, 5, &extent, &mut rows, &mut dirty)
                .unwrap(),
            [0; 2]
        );
        for (row, old) in rows
            .chunks_exact(RASTER_ROW_WORDS)
            .zip(stale.chunks_exact(RASTER_ROW_WORDS))
        {
            assert!(RasterRow::read(row).unwrap().is_empty());
            assert_eq!(row[6], 0);
            assert_eq!(&row[..6], &old[..6]);
            assert_eq!(&row[7..], &old[7..]);
        }
    }

    /// A ground, an empty group, an opaque chip and a translucent rounded
    /// ring: 96×48, six columns and three bands of 16.
    fn mixed() -> (Scene<8>, [crate::scene::NodeId; 4]) {
        let mut scene = Scene::<8>::new();
        let root = scene
            .add_root(lp(0, 0, 96, 48), Bind::Fill { color: 0x10_1014 })
            .unwrap();
        let group = scene
            .add_child(root, lp(16, 0, 32, 16), Bind::Composite)
            .unwrap();
        let chip = scene
            .add_child(root, lp(64, 0, 16, 16), Bind::Fill { color: 0x30_6090 })
            .unwrap();
        let ring = scene
            .add_child_with_opacity(
                root,
                lp(40, 20, 40, 20),
                Bind::Border {
                    color: 0xF0_F0F0,
                    width: 2,
                    radii: Corners::uniform(6),
                },
                180,
            )
            .unwrap();
        scene.resolve_layout_host(Rect {
            x: 0,
            y: 0,
            w: 96,
            h: 48,
        });
        (scene, [root, group, chip, ring])
    }

    /// One band lowered both ways from the same row storage: the rows, the
    /// dirty vector and the painted columns must match word for word.
    fn lower_both(
        frame: &TiledFrame<'_, 8, 16, 16>,
        program: &SceneProgram<Global>,
        ops: &[Option<RasterOp>],
        tr: usize,
    ) -> [u64; 2] {
        let extent = |_: u16| None;
        let mut by_program = [0x5A5A_5A5Ai32; 8 * RASTER_ROW_WORDS];
        let mut by_ops = by_program;
        let (mut dirty_program, mut dirty_ops) = ([7u8; 6], [7u8; 6]);
        let painted = frame
            .lower_band(program, tr, &extent, &mut by_program, &mut dirty_program)
            .unwrap();
        assert_eq!(
            frame
                .lower_band_dynamic(ops, tr, &extent, &mut by_ops, &mut dirty_ops)
                .unwrap(),
            painted
        );
        assert_eq!(by_ops, by_program, "band {tr}");
        assert_eq!(dirty_ops, dirty_program, "band {tr}");
        painted
    }

    #[test]
    fn a_dynamic_lowering_writes_the_rows_a_program_lowering_writes() {
        let (scene, _) = mixed();
        let program = SceneProgram::build_in(&scene, bindings(), 16, 6, Global).unwrap();
        let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 48)).unwrap();
        assert_eq!(frame.columns(), 6);
        let mut table = [Some(RasterOp::Glyphs); 8];
        let ops = raster_ops(&scene, &mut table).unwrap();
        assert_eq!(ops, program.ops.as_slice(), "one op table, read once");
        let mut reached = 0;
        for tr in 0..=frame.bands() {
            reached |= lower_both(&frame, &program, ops, tr)[0];
        }
        assert_eq!(reached, 0b11_1111, "the fixture paints every column");
    }

    #[test]
    fn a_kind_change_or_new_order_drifts_a_program_but_lowers_dynamically() {
        let extent = |_: u16| None;
        let (mut scene, [root, group, chip, _]) = mixed();
        let program = SceneProgram::build_in(&scene, bindings(), 16, 6, Global).unwrap();
        let mut stale = [None; 8];
        let stale = raster_ops(&scene, &mut stale).unwrap().to_vec();
        let mut rows = [0i32; 8 * RASTER_ROW_WORDS];
        let mut dirty = [0u8; 6];
        let root_rect = Rect {
            x: 0,
            y: 0,
            w: 96,
            h: 48,
        };

        // A group become paintable: refused, never left unpainted.
        scene
            .set_bind(group, Bind::Fill { color: 0xC0_2020 })
            .unwrap();
        scene.resolve_layout_host(root_rect);
        {
            let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 48)).unwrap();
            assert_eq!(
                frame
                    .lower_band(&program, 0, &extent, &mut rows, &mut dirty)
                    .err(),
                Some(ProgramError::Drift)
            );
            assert_eq!(
                frame
                    .lower_band_dynamic(&stale, 0, &extent, &mut rows, &mut dirty)
                    .err(),
                Some(ProgramError::Drift),
                "a stale op table drifts too"
            );
            let mut table = [None; 8];
            let ops = raster_ops(&scene, &mut table).unwrap();
            frame
                .lower_band_dynamic(ops, 0, &extent, &mut rows, &mut dirty)
                .unwrap();
            let painted = RasterRow::read(&rows[RASTER_ROW_WORDS..]).unwrap();
            assert_eq!(painted.tiles, [0b00_0110, 0]);
            assert_eq!(painted.argb, 0xFFC0_2020);
            // The new fill covers its two tiles: the ground under it drops them.
            let ground = RasterRow::read(&rows).unwrap();
            assert_eq!(ground.tiles, [0b10_1001, 0]);
        }
        scene.commit_frame();

        // A paintable node become a group: the ground takes its tile back.
        scene.set_bind(chip, Bind::Composite).unwrap();
        scene.resolve_layout_host(root_rect);
        {
            let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 48)).unwrap();
            assert_eq!(
                frame
                    .lower_band(&program, 0, &extent, &mut rows, &mut dirty)
                    .err(),
                Some(ProgramError::Drift)
            );
            let mut table = [None; 8];
            let ops = raster_ops(&scene, &mut table).unwrap();
            let painted = frame
                .lower_band_dynamic(ops, 0, &extent, &mut rows, &mut dirty)
                .unwrap();
            assert_eq!(painted, [0b01_0000, 0], "only the chip's tile was dirty");
            assert_eq!(RasterRow::read(&rows).unwrap().tiles, [0b01_0000, 0]);
            assert!(
                RasterRow::read(&rows[2 * RASTER_ROW_WORDS..])
                    .unwrap()
                    .is_empty()
            );
        }
        scene.commit_frame();

        // A new node and a new paint order: the program's rows no longer
        // line up; a table read from this frame does.
        let late = scene
            .add_child(root, lp(0, 32, 16, 16), Bind::Fill { color: 0x20_A040 })
            .unwrap();
        scene.move_before(late, group).unwrap();
        scene.resolve_layout_host(root_rect);
        let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 48)).unwrap();
        assert_eq!(
            frame
                .lower_band(&program, 2, &extent, &mut rows, &mut dirty)
                .err(),
            Some(ProgramError::Drift)
        );
        assert_eq!(
            frame
                .lower_band_dynamic(&stale, 2, &extent, &mut rows, &mut dirty)
                .err(),
            Some(ProgramError::Drift)
        );
        let mut table = [None; 8];
        let ops = raster_ops(&scene, &mut table).unwrap();
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[1], Some(RasterOp::Fill), "the new node paints second");
        frame
            .lower_band_dynamic(ops, 2, &extent, &mut rows, &mut dirty)
            .unwrap();
        let late = RasterRow::read(&rows[RASTER_ROW_WORDS..]).unwrap();
        assert_eq!(late.tiles, [0b00_0001, 0]);
        assert_eq!(late.argb, 0xFF20_A040);
        // The rebuilt program agrees with the table, row for row.
        let program = SceneProgram::build_in(&scene, bindings(), 16, 6, Global).unwrap();
        assert_eq!(program.ops.as_slice(), ops);
        for tr in 0..=frame.bands() {
            lower_both(&frame, &program, ops, tr);
        }
    }

    #[test]
    fn a_dynamic_lowering_refuses_what_it_cannot_lower() {
        let extent = |_: u16| None;
        let (scene, _) = mixed();
        let frame = TiledFrame::<8, 16, 16>::prepare(&scene, (96, 48)).unwrap();
        let mut table = [None; 8];
        assert_eq!(
            raster_ops(&scene, &mut table[..3]).err(),
            Some(ProgramError::Capacity)
        );
        let ops = raster_ops(&scene, &mut table).unwrap();
        let mut rows = [0i32; 8 * RASTER_ROW_WORDS];
        let mut dirty = [0u8; 6];
        assert_eq!(
            frame
                .lower_band_dynamic(&ops[..3], 0, &extent, &mut rows, &mut dirty)
                .err(),
            Some(ProgramError::Drift),
            "a table of another length was read from another scene"
        );
        assert_eq!(
            frame
                .lower_band_dynamic(
                    ops,
                    0,
                    &extent,
                    &mut rows[..3 * RASTER_ROW_WORDS],
                    &mut dirty
                )
                .err(),
            Some(ProgramError::Capacity)
        );
        assert_eq!(
            frame
                .lower_band_dynamic(ops, 0, &extent, &mut rows, &mut dirty[..5])
                .err(),
            Some(ProgramError::Capacity)
        );
        // A tile wider than a row's executor addresses.
        let wide = TiledFrame::<8, 65_536, 16>::prepare(&scene, (96, 48)).unwrap();
        assert_eq!(
            wide.lower_band_dynamic(ops, 0, &extent, &mut rows, &mut dirty)
                .err(),
            Some(ProgramError::Columns)
        );
    }
}
