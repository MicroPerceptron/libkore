//! Tile bookkeeping for a frame.
//!
//! Divides the framebuffer into a grid of `TW × TH` pixel tiles and reads
//! a resolved scene's dirty state into one [`TiledFrame`]:
//!
//! - **Tight damage clipping** — damage maps to a per-tile bitmask
//!   ([`TileMask`]), only dirty tiles repaint. Scattered damage (two
//!   small nodes in opposite corners) skips clean tiles in the middle.
//! - **Row culling** — a per-tile-row node bitset ([`RowNodeMask`]) means
//!   a dirty tile visits only the nodes whose rect reaches its row, in
//!   paint order.
//! - **Bands as items** — tile rows are independent: a compositor hands
//!   them to its workers one per item.
//!
//! What a tile paints is the scene program's ([`crate::program`]):
//! [`TiledFrame::lower_band`] writes one band's parameter rows for the
//! program, culled from the topmost node covering each tile opaquely —
//! [`TiledFrame::lower_band_dynamic`] the same rows against a per-frame op
//! table, for a scene no program is built for.
//! 64×64 is the recommended tile (16 KB per tile at 4 B/pixel).

use crate::layout::Rect;
use crate::scene::Scene;
use crate::storage::Storage;

// ── Tile bitmask ────────────────────────────────────────────────────

/// Maximum number of u64 words in a tile bitmask.
///
/// 128 × 64 bits = 8192 tiles. Enough for 8K displays at 32×32 tiles
/// (240 × 135 = 32400 — for that you'd need a larger mask, but 64×64
/// tiles at 8K is 120 × 68 = 8160, which fits).
pub const MASK_WORDS: usize = 128;

/// Compact bitmask tracking which tiles need repainting.
///
/// Stack-allocated, `no_std`-friendly. Supports up to `MASK_WORDS × 64`
/// tiles (8192 with the default `MASK_WORDS = 128`).
///
/// Each bit corresponds to one tile in row-major order. The bit layout
/// matches the `bspl` dirty tile bitmask, enabling zero-copy damage
/// propagation from boot splash animation frames.
pub struct TileMask {
    pub bits: [u64; MASK_WORDS],
}

impl TileMask {
    /// The most tiles one mask addresses.  A tiled executor refuses a
    /// framebuffer that divides into more (`prepare` answers `None`) instead
    /// of dropping the tiles past the end.
    pub const CAPACITY: usize = MASK_WORDS * 64;

    /// All-zeros: no tiles dirty.
    #[inline]
    pub const fn empty() -> Self {
        TileMask {
            bits: [0u64; MASK_WORDS],
        }
    }

    /// All-ones up to `count` tiles: full repaint.
    fn full(count: usize) -> Self {
        let mut m = Self::empty();
        let full_words = count / 64;
        let remainder = count % 64;
        for i in 0..full_words.min(MASK_WORDS) {
            m.bits[i] = u64::MAX;
        }
        if remainder > 0 && full_words < MASK_WORDS {
            m.bits[full_words] = (1u64 << remainder) - 1;
        }
        m
    }

    /// Set the bit for tile at `(col, row)` in a grid of `cols` columns.
    #[inline]
    fn set(&mut self, col: usize, row: usize, cols: usize) {
        let idx = row * cols + col;
        let word = idx / 64;
        let bit = idx % 64;
        if word < MASK_WORDS {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Test whether tile at `(col, row)` is dirty.
    #[inline]
    pub(crate) fn test(&self, col: usize, row: usize, cols: usize) -> bool {
        let idx = row * cols + col;
        let word = idx / 64;
        let bit = idx % 64;
        word < MASK_WORDS && (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Visit every dirty tile as a framebuffer rect (clipped to `fb_w × fb_h`)
    /// for a grid of `cols × rows` tiles of `TW × TH` pixels — the rects a
    /// caller copies from a back buffer to a front target after
    /// the frame painted them.
    pub fn for_each_rect<const TW: usize, const TH: usize>(
        &self,
        cols: usize,
        rows: usize,
        fb_w: usize,
        fb_h: usize,
        mut visit: impl FnMut(Rect),
    ) {
        for tr in 0..rows {
            for tc in 0..cols {
                if !self.test(tc, tr, cols) {
                    continue;
                }
                let x = tc * TW;
                let y = tr * TH;
                if x >= fb_w || y >= fb_h {
                    continue;
                }
                let w = TW.min(fb_w - x);
                let h = TH.min(fb_h - y);
                visit(Rect {
                    x: x as i32,
                    y: y as i32,
                    w: w as u16,
                    h: h as u16,
                });
            }
        }
    }

    /// Number of dirty tiles.
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Mark all tiles that overlap `rect` as dirty.
    fn mark_rect<const TW: usize, const TH: usize>(
        &mut self,
        rect: &Rect,
        cols: usize,
        rows: usize,
        fb_w: usize,
        fb_h: usize,
    ) {
        if rect.w == 0 || rect.h == 0 {
            return;
        }
        let (tc0, tr0, tc1, tr1) = tile_range::<TW, TH>(rect, cols, rows, fb_w, fb_h);
        for tr in tr0..tr1 {
            for tc in tc0..tc1 {
                self.set(tc, tr, cols);
            }
        }
    }
}

// ── Precomputed node-to-tile span ───────────────────────────────────

/// Which tile columns/rows a node overlaps. Precomputed after layout
/// resolution to replace per-tile `Rect::intersect` with 4 range checks.
#[derive(Clone, Copy)]
pub(crate) struct TileSpan {
    col0: u16,
    col1: u16,
    row0: u16,
    row1: u16,
}

impl TileSpan {
    const EMPTY: Self = TileSpan {
        col0: 0,
        col1: 0,
        row0: 0,
        row1: 0,
    };

    fn from_rect<const TW: usize, const TH: usize>(
        rect: &Rect,
        cols: usize,
        rows: usize,
        fb_w: usize,
        fb_h: usize,
    ) -> Self {
        if rect.w == 0 || rect.h == 0 {
            return Self::EMPTY;
        }
        let (tc0, tr0, tc1, tr1) = tile_range::<TW, TH>(rect, cols, rows, fb_w, fb_h);
        TileSpan {
            col0: tc0 as u16,
            col1: tc1 as u16,
            row0: tr0 as u16,
            row1: tr1 as u16,
        }
    }

    /// Does this span include tile `(col, row)`?
    #[inline]
    pub(crate) fn contains(&self, col: usize, row: usize) -> bool {
        self.contains_col(col) && (row as u16) >= self.row0 && (row as u16) < self.row1
    }

    /// Does this span include tile column `col`?
    #[inline]
    pub(crate) fn contains_col(&self, col: usize) -> bool {
        (col as u16) >= self.col0 && (col as u16) < self.col1
    }
}

// ── Per-tile-row node bitsets ───────────────────────────────────────

/// Tile rows a [`RowNodeMask`] tracks individually — matches
/// [`MASK_WORDS`] so a 64×64-tiled 8K frame fits; taller grids fall back
/// to the all-nodes set for the rows past the end.
pub const MAX_TILE_ROWS: usize = 128;

/// Words per row: 256 nodes.
const ROW_WORDS: usize = 4;

/// Largest scene a [`TiledFrame`] holds.
pub const MAX_TILED_NODES: usize = ROW_WORDS * 64;

/// For each tile row, the set of nodes (as positions in the scene's paint
/// order) whose visible rect reaches that row. Iterating a row's set bits
/// ascending visits exactly the candidate nodes for any tile in the row,
/// in paint order, so a dirty tile costs `O(nodes in its row)` rather
/// than `O(N)`.
pub(crate) struct RowNodeMask {
    rows: [[u64; ROW_WORDS]; MAX_TILE_ROWS],
    /// Every tracked node — the fallback for rows past `MAX_TILE_ROWS`.
    all: [u64; ROW_WORDS],
    #[cfg(feature = "graph")]
    dynamic: Option<(usize, alloc::vec::Vec<u64, crate::SceneArena>)>,
}

impl RowNodeMask {
    const fn empty() -> Self {
        Self {
            rows: [[0; ROW_WORDS]; MAX_TILE_ROWS],
            all: [0; ROW_WORDS],
            #[cfg(feature = "graph")]
            dynamic: None,
        }
    }

    #[cfg(feature = "graph")]
    fn new_in(nodes: usize, arena: crate::SceneArena) -> Option<Self> {
        let words = nodes.div_ceil(64);
        let len = words.checked_mul(MAX_TILE_ROWS + 1)?;
        let mut values = alloc::vec::Vec::new_in(arena);
        values.try_reserve_exact(len).ok()?;
        values.resize(len, 0);
        let mut mask = Self::empty();
        mask.dynamic = Some((words, values));
        Some(mask)
    }

    /// Record paint-order position `i` on every row of `span`.
    fn mark(&mut self, span: &TileSpan, i: usize) {
        let bit = 1u64 << (i % 64);
        let word = i / 64;
        #[cfg(feature = "graph")]
        if let Some((words, values)) = self.dynamic.as_mut() {
            values[MAX_TILE_ROWS * *words + word] |= bit;
            for row in usize::from(span.row0)..usize::from(span.row1).min(MAX_TILE_ROWS) {
                values[row * *words + word] |= bit;
            }
            return;
        }
        self.all[word] |= bit;
        let end = usize::from(span.row1).min(MAX_TILE_ROWS);
        for row in usize::from(span.row0)..end {
            self.rows[row][word] |= bit;
        }
    }

    /// The candidate set for tile row `row`.
    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[u64] {
        #[cfg(feature = "graph")]
        if let Some((words, values)) = &self.dynamic {
            let start = row.min(MAX_TILE_ROWS) * *words;
            return &values[start..start + *words];
        }
        if row < MAX_TILE_ROWS {
            &self.rows[row]
        } else {
            &self.all
        }
    }
}

// ── The frame ───────────────────────────────────────────────────────

/// One frame's tile state, read once ([`prepare`](Self::prepare)) and
/// lowered band by band ([`lower_band`](Self::lower_band)) by any worker
/// with its own program fork and stage: the dirty mask, every node's
/// visible bounds and the per-tile-row node sets are read-only for the
/// frame's life, so bands are independent items — a compositor hands
/// them to its resident workers one per item.  The scene is borrowed
/// shared; once every band painted, drop the frame and
/// [`Scene::commit_frame`](crate::scene::Scene::commit_frame).
pub struct TiledFrame<'s, const N: usize, const TW: usize, const TH: usize> {
    pub(crate) scene: &'s Scene<N>,
    pub(crate) fb_w: usize,
    pub(crate) fb_h: usize,
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) dirty: TileMask,
    pub(crate) bounds: Storage<Rect, N>,
    pub(crate) spans: Storage<TileSpan, N>,
    pub(crate) row_mask: RowNodeMask,
}

impl<'s, const N: usize, const TW: usize, const TH: usize> TiledFrame<'s, N, TW, TH> {
    /// Read a resolved scene's dirty state for a frame of `extent = (W, H)`
    /// — the caller resolved layout ([`Scene::resolve_layout_host`])
    /// beforehand.  `None` when the frame divides into more tiles than a
    /// [`TileMask`] addresses ([`TileMask::CAPACITY`]).
    pub fn prepare(scene: &'s Scene<N>, extent: (usize, usize)) -> Option<Self> {
        Self::prepare_storage(
            scene,
            extent,
            Storage::inline([Rect::ZERO; N]),
            Storage::inline([TileSpan::EMPTY; N]),
            RowNodeMask::empty(),
        )
    }

    /// Prepare a runtime-sized scene using a caller-owned named arena.
    #[cfg(feature = "graph")]
    pub fn prepare_in(
        scene: &'s Scene<N>,
        extent: (usize, usize),
        arena: crate::SceneArena,
    ) -> Option<Self> {
        let len = scene.order().len();
        let mut bounds = Storage::in_arena([Rect::ZERO; N], arena);
        let mut spans = Storage::in_arena([TileSpan::EMPTY; N], arena);
        bounds.reserve_to(len).ok()?;
        spans.reserve_to(len).ok()?;
        bounds.resize_reserved(len, Rect::ZERO);
        spans.resize_reserved(len, TileSpan::EMPTY);
        let mask = RowNodeMask::new_in(len, arena)?;
        Self::prepare_storage(scene, extent, bounds, spans, mask)
    }

    fn prepare_storage(
        scene: &'s Scene<N>,
        extent: (usize, usize),
        mut bounds: Storage<Rect, N>,
        mut spans: Storage<TileSpan, N>,
        mut row_mask: RowNodeMask,
    ) -> Option<Self> {
        if scene.order().len() > bounds.len() {
            return None;
        }
        #[cfg(feature = "graph")]
        let dynamic = row_mask.dynamic.is_some();
        #[cfg(not(feature = "graph"))]
        let dynamic = false;
        if !dynamic && scene.order().len() > MAX_TILED_NODES {
            return None;
        }
        let (fb_w, fb_h) = extent;
        let cols = fb_w.div_ceil(TW);
        let rows = fb_h.div_ceil(TH);
        if cols * rows > TileMask::CAPACITY {
            return None;
        }

        // 2. Build the dirty tile bitmask.  Damage state is committed only
        //    after every tile painted: a backend error leaves the dirty flags
        //    and damage list intact, so the next frame repaints what this one
        //    did not, instead of treating the scene as clean.
        let dirty = build_dirty_mask::<N, TW, TH>(scene, cols, rows, fb_w, fb_h);

        // 3. Per node (by paint-order position): visible bounds, tile span,
        //    and the row bitsets. Invisible and fully clipped nodes are
        //    left out of every set.
        let order = scene.order();
        for (i, &idx) in order.iter().enumerate() {
            let node = scene.node(idx);
            if node.opacity == 0 {
                continue;
            }
            let mut r = scene.rect(idx);
            if let Some(clip) = scene.clip_rect(idx) {
                r = r.intersect(&clip);
            }
            if r.w == 0 || r.h == 0 {
                continue;
            }
            let span = TileSpan::from_rect::<TW, TH>(&r, cols, rows, fb_w, fb_h);
            if span.col0 == span.col1 || span.row0 == span.row1 {
                continue;
            }
            bounds[i] = r;
            spans[i] = span;
            row_mask.mark(&span, i);
        }

        Some(TiledFrame {
            scene,
            fb_w,
            fb_h,
            cols,
            rows,
            dirty,
            bounds,
            spans,
            row_mask,
        })
    }

    /// Tile rows in the frame — the items a parallel paint claims.
    pub const fn bands(&self) -> usize {
        self.rows
    }

    /// Tile columns in the frame — the width of a band's dirty vector.
    pub const fn columns(&self) -> usize {
        self.cols
    }

    /// The tiles this frame repaints.
    pub const fn dirty(&self) -> &TileMask {
        &self.dirty
    }

    /// Whether band `tr` holds a dirty tile at all (a worker skips the
    /// rest without touching its stage).
    pub fn band_is_dirty(&self, tr: usize) -> bool {
        tr < self.rows && (0..self.cols).any(|tc| self.dirty.test(tc, tr, self.cols))
    }
}

/// Build a TileMask from the scene's dirty state.
///
/// - First frame (no prev): all tiles dirty.
/// - Subsequent frames: union of old rects + new rects of dirty nodes,
///   mapped to tile coordinates.
fn build_dirty_mask<const N: usize, const TW: usize, const TH: usize>(
    scene: &Scene<N>,
    cols: usize,
    rows: usize,
    fb_w: usize,
    fb_h: usize,
) -> TileMask {
    if !scene.has_prev() {
        // First frame or structural change — full repaint.
        return TileMask::full(cols * rows);
    }

    // Build precise bitmask from dirty node rects.
    let mut mask = TileMask::empty();
    for &idx in scene.order() {
        if scene.is_node_dirty(idx) {
            // Mark tiles covered by the old rect (where content was).
            mask.mark_rect::<TW, TH>(&scene.prev_rect(idx), cols, rows, fb_w, fb_h);
            // Mark tiles covered by the new rect (where content will be).
            mask.mark_rect::<TW, TH>(&scene.rect(idx), cols, rows, fb_w, fb_h);
        }
    }
    // Rects freed by `remove` live only in the damage list (their slots
    // are gone); mark them rect by rect so scattered damage stays sparse.
    if let Some(rects) = scene.damage_rects() {
        for rect in rects {
            mask.mark_rect::<TW, TH>(rect, cols, rows, fb_w, fb_h);
        }
    }
    mask
}

// ── Shared utilities ────────────────────────────────────────────────

/// Compute the tile column/row range that covers a rect.
///
/// Returns `(col_start, row_start, col_end, row_end)` — half-open ranges.
fn tile_range<const TW: usize, const TH: usize>(
    rect: &Rect,
    cols: usize,
    rows: usize,
    fb_w: usize,
    fb_h: usize,
) -> (usize, usize, usize, usize) {
    let x0 = (rect.x.max(0) as usize).min(fb_w);
    let y0 = (rect.y.max(0) as usize).min(fb_h);
    let x1 = ((rect.x + rect.w as i32).max(0) as usize).min(fb_w);
    let y1 = ((rect.y + rect.h as i32).max(0) as usize).min(fb_h);

    let tc0 = (x0 / TW).min(cols);
    let tr0 = (y0 / TH).min(rows);
    let tc1 = x1.div_ceil(TW).min(cols);
    let tr1 = y1.div_ceil(TH).min(rows);

    (tc0, tr0, tc1, tr1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::Bind;
    use crate::layout::{Anchor, LayoutParams, Offset};

    fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
        LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset { x, y },
            w,
            h,
        }
    }

    #[test]
    fn tile_range_full_fb() {
        let d = Rect {
            x: 0,
            y: 0,
            w: 128,
            h: 128,
        };
        let (tc0, tr0, tc1, tr1) = tile_range::<64, 64>(&d, 2, 2, 128, 128);
        assert_eq!((tc0, tr0, tc1, tr1), (0, 0, 2, 2));
    }

    #[test]
    fn tile_range_partial() {
        let d = Rect {
            x: 70,
            y: 5,
            w: 30,
            h: 20,
        };
        let (tc0, tr0, tc1, tr1) = tile_range::<64, 64>(&d, 2, 2, 128, 128);
        assert_eq!((tc0, tr0, tc1, tr1), (1, 0, 2, 1));
    }

    #[test]
    fn tile_range_single_pixel() {
        let d = Rect {
            x: 65,
            y: 0,
            w: 1,
            h: 1,
        };
        let (tc0, tr0, tc1, tr1) = tile_range::<64, 64>(&d, 4, 4, 256, 256);
        assert_eq!((tc0, tr0, tc1, tr1), (1, 0, 2, 1));
    }

    #[test]
    fn tile_range_non_square() {
        let d = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let (tc0, tr0, tc1, tr1) = tile_range::<32, 16>(&d, 4, 7, 100, 100);
        assert_eq!((tc0, tr0, tc1, tr1), (0, 0, 4, 7));
    }

    #[test]
    fn tilemask_set_test() {
        let mut m = TileMask::empty();
        assert!(!m.test(2, 3, 10));
        m.set(2, 3, 10);
        assert!(m.test(2, 3, 10));
        assert!(!m.test(3, 3, 10));
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn tilemask_full() {
        let m = TileMask::full(100);
        for i in 0..100 {
            assert!(m.test(i, 0, 100), "bit {i} should be set");
        }
        assert!(!m.test(100, 0, 101));
        assert_eq!(m.count(), 100);
    }

    #[test]
    fn tilemask_mark_rect() {
        let mut m = TileMask::empty();
        // 128×128 fb, 64×64 tiles → 2×2 grid.
        // Rect covering bottom-right tile only.
        let r = Rect {
            x: 64,
            y: 64,
            w: 64,
            h: 64,
        };
        m.mark_rect::<64, 64>(&r, 2, 2, 128, 128);
        assert!(!m.test(0, 0, 2));
        assert!(!m.test(1, 0, 2));
        assert!(!m.test(0, 1, 2));
        assert!(m.test(1, 1, 2)); // only bottom-right dirty
    }

    #[test]
    fn tilespan_contains() {
        let s = TileSpan {
            col0: 1,
            col1: 3,
            row0: 0,
            row1: 2,
        };
        assert!(!s.contains(0, 0));
        assert!(s.contains(1, 0));
        assert!(s.contains(2, 1));
        assert!(!s.contains(3, 0));
        assert!(!s.contains(1, 2));
    }

    #[test]
    fn row_mask_tracks_rows_and_falls_back_past_the_end() {
        let mut m = RowNodeMask::empty();
        m.mark(
            &TileSpan {
                col0: 0,
                col1: 1,
                row0: 2,
                row1: 4,
            },
            5,
        );
        m.mark(
            &TileSpan {
                col0: 0,
                col1: 1,
                row0: 0,
                row1: 200,
            },
            70,
        );
        // Position 70 is word 1, bit 6.
        assert_eq!(m.row(1), &[0, 1 << 6, 0, 0]);
        assert_eq!(m.row(2), &[1 << 5, 1 << 6, 0, 0]);
        assert_eq!(m.row(3), &[1 << 5, 1 << 6, 0, 0]);
        assert_eq!(m.row(4), &[0, 1 << 6, 0, 0]);
        assert_eq!(m.row(MAX_TILE_ROWS - 1), &[0, 1 << 6, 0, 0]);
        // Past the tracked rows: everything ever marked.
        assert_eq!(m.row(MAX_TILE_ROWS), &[1 << 5, 1 << 6, 0, 0]);
        assert_eq!(m.row(150), &[1 << 5, 1 << 6, 0, 0]);
    }

    #[test]
    fn hover_colour_change_dirties_exactly_one_tile() {
        // 256×192 framebuffer, 64×64 tiles → 4×3 grid.
        let mut scene = Scene::<8>::new();
        let root = scene
            .add_root(lp(0, 0, 256, 192), Bind::Fill { color: 0x202020 })
            .unwrap();
        let pip = scene
            .add_child(root, lp(20, 20, 12, 12), Bind::Fill { color: 0x404040 })
            .unwrap();
        let root_rect = Rect {
            x: 0,
            y: 0,
            w: 256,
            h: 192,
        };
        scene.resolve_layout_host(root_rect);
        assert_eq!(
            build_dirty_mask::<8, 64, 64>(&scene, 4, 3, 256, 192).count(),
            12,
            "first frame repaints everything"
        );
        scene.commit_frame();

        scene.set_bind(pip, Bind::Fill { color: 0x80A0FF }).unwrap();
        scene.resolve_layout_host(root_rect);
        let mask = build_dirty_mask::<8, 64, 64>(&scene, 4, 3, 256, 192);
        assert_eq!(mask.count(), 1);
        assert!(mask.test(0, 0, 4));
        scene.commit_frame();

        // Nothing changed: nothing dirty.
        scene.resolve_layout_host(root_rect);
        assert_eq!(
            build_dirty_mask::<8, 64, 64>(&scene, 4, 3, 256, 192).count(),
            0
        );
    }

    #[test]
    fn a_grid_past_the_mask_is_refused() {
        let mut scene = Scene::<4>::new();
        scene
            .add_root(lp(0, 0, 128, 80), Bind::Fill { color: 0 })
            .unwrap();
        scene.resolve_layout_host(Rect {
            x: 0,
            y: 0,
            w: 128,
            h: 80,
        });
        // 128 × 80 at 1 × 1 tiles = 10 240 tiles, past the 8 192 the mask holds.
        assert!(TiledFrame::<4, 1, 1>::prepare(&scene, (128, 80)).is_none());
        // The same scene at 64 × 64 tiles (4 tiles) prepares.
        assert!(TiledFrame::<4, 64, 64>::prepare(&scene, (128, 80)).is_some());
    }
}
