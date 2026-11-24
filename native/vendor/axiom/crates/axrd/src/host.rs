//! The scene program on the host: `ax-exec`'s native executor over the CPU
//! raster kernels, driving one [`SceneProgram`] band by band into a caller
//! framebuffer — what examples, benches and tests paint with, and the
//! sequential twin of the band-per-worker loop a compositor runs on the
//! lane it founds.

use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::GraphBuf;
use ax::raster::{RasterOp, ResourceKey};
use ax_exec::{AxCpuKernelPolicy, NativeProgram, NativeResourceBinding, PreparedNative};
use axnn::{RASTER_ROW_WORDS, RasterRow};
use axnn_cpu::{CoverageAtlasRef, GlyphTableRef, XrgbSurfaceRef, XrgbTargetRef};

use crate::glyph::PlacedGlyph;
use crate::layout::Rect;
use crate::program::{ProgramBindings, ProgramError, SceneProgram};
use crate::scene::Scene;
use crate::tile::{TileMask, TiledFrame};

/// Where the host binds a program's resources.
pub const BINDINGS: ProgramBindings = ProgramBindings {
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
};

/// Surfaces one band may bind.
const MAX_BOUND_SURFACES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostError {
    /// The scene could not be lowered, or drifted twice in one frame.
    Program(ProgramError),
    /// The executor refused the program or a band's run.
    Execute,
    /// The frame's extent is empty, too large for the row mask, or the
    /// framebuffer is smaller than it.
    Extent,
    Allocation,
}

impl From<ProgramError> for HostError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

/// One scene's program, prepared on the host and run per band into a
/// framebuffer of the root rect's size (`pitch == width`).
///
/// The program is rebuilt when a frame finds the scene drifted (a node's
/// bind kind changed) — a convenience for examples; a compositor keeps
/// its node kinds fixed and never pays that.
pub struct SceneHost<A: Allocator + Clone, const N: usize, const TILE: usize> {
    program: SceneProgram<A>,
    prepared: PreparedNative<A>,
    rows: Vec<i32, A>,
    dirty: Vec<u8, A>,
    stage: Vec<u32, A>,
    root: Rect,
    columns: usize,
    allocator: A,
}

impl<A: Allocator + Clone, const N: usize, const TILE: usize> SceneHost<A, N, TILE> {
    /// Prepare `scene`'s program for frames of `root`'s size.
    pub fn new_in(scene: &mut Scene<N>, root: Rect, allocator: A) -> Result<Self, HostError> {
        let (width, height) = (usize::from(root.w), usize::from(root.h));
        if width == 0 || height == 0 || TILE == 0 {
            return Err(HostError::Extent);
        }
        let columns = width.div_ceil(TILE);
        scene.resolve_layout_host(root);
        let (program, prepared) = Self::build(scene, columns, allocator.clone())?;
        let mut rows = Vec::new_in(allocator.clone());
        rows.try_reserve_exact(program.table_words())
            .map_err(|_| HostError::Allocation)?;
        rows.resize(program.table_words(), 0);
        let mut dirty = Vec::new_in(allocator.clone());
        dirty
            .try_reserve_exact(columns)
            .map_err(|_| HostError::Allocation)?;
        dirty.resize(columns, 0);
        let mut stage = Vec::new_in(allocator.clone());
        stage
            .try_reserve_exact(width * TILE)
            .map_err(|_| HostError::Allocation)?;
        stage.resize(width * TILE, 0);
        Ok(Self {
            program,
            prepared,
            rows,
            dirty,
            stage,
            root,
            columns,
            allocator,
        })
    }

    fn build(
        scene: &Scene<N>,
        columns: usize,
        allocator: A,
    ) -> Result<(SceneProgram<A>, PreparedNative<A>), HostError> {
        let program =
            SceneProgram::build_in(scene, BINDINGS, TILE as u32, columns, allocator.clone())?;
        let graph = GraphBuf::copy_from_in(program.graph().as_bytes(), allocator.clone())
            .map_err(|_| HostError::Allocation)?;
        let native = NativeProgram::from_graph_in(
            graph,
            AxCpuKernelPolicy::default(),
            true,
            allocator.clone(),
        )
        .map_err(|_| HostError::Execute)?;
        let prepared = native
            .prepare_native_in(allocator)
            .map_err(|_| HostError::Allocation)?;
        Ok((program, prepared))
    }

    /// The program this host runs.
    pub fn program(&self) -> &SceneProgram<A> {
        &self.program
    }

    /// Paint one frame of `scene` into `fb` (`root.w × root.h`, row-major):
    /// resolve layout, read the dirty tiles, lower and run every dirty
    /// band, store the painted tile runs, commit the frame.  Surfaces come
    /// from `source` by slot ([`crate::BLIT_SOURCE_SLOT`] for the frame's
    /// blit/sample source), glyph coverage from `atlas`, placed glyphs from
    /// `glyphs`.  Returns the tiles the frame repainted.
    pub fn paint<'s, 'g>(
        &mut self,
        scene: &mut Scene<N>,
        fb: &mut [u32],
        source: &dyn Fn(u32) -> Option<XrgbSurfaceRef<'s>>,
        atlas: Option<CoverageAtlasRef<'g>>,
        glyphs: &'g [PlacedGlyph],
    ) -> Result<TileMask, HostError> {
        let (width, height) = (usize::from(self.root.w), usize::from(self.root.h));
        if fb.len() < width * height {
            return Err(HostError::Extent);
        }
        scene.resolve_layout_host(self.root);
        let frame = TiledFrame::<N, TILE, TILE>::prepare(scene, (width, height))
            .ok_or(HostError::Extent)?;
        let extent = |id: u16| source(u32::from(id)).map(|s| (s.width(), s.height()));
        for band in 0..frame.bands() {
            if !frame.band_is_dirty(band) {
                continue;
            }
            let painted = match frame.lower_band(
                &self.program,
                band,
                &extent,
                &mut self.rows,
                &mut self.dirty,
            ) {
                Ok(painted) => painted,
                Err(ProgramError::Drift) => {
                    let (program, prepared) =
                        Self::build(scene, self.columns, self.allocator.clone())?;
                    self.program = program;
                    self.prepared = prepared;
                    self.rows.resize(self.program.table_words(), 0);
                    frame.lower_band(
                        &self.program,
                        band,
                        &extent,
                        &mut self.rows,
                        &mut self.dirty,
                    )?
                }
                Err(error) => return Err(error.into()),
            };
            if painted == [0; 2] {
                continue;
            }
            let band_y = band * TILE;
            let band_h = TILE.min(height - band_y);
            // The stage carries the band's current pixels in, so a node
            // painting part of a tile composes over what was there.
            self.stage[..band_h * width]
                .copy_from_slice(&fb[band_y * width..(band_y + band_h) * width]);
            let target = XrgbTargetRef::new(
                &mut self.stage,
                (0, band_y as i32),
                width as u32,
                band_h as u32,
                width,
            )
            .map_err(|_| HostError::Extent)?;
            let table = GlyphTableRef::new(glyphs);
            let mut bindings = [NativeResourceBinding::glyph_table(BINDINGS.glyphs, table);
                3 + MAX_BOUND_SURFACES];
            bindings[0] = NativeResourceBinding::xrgb_target(BINDINGS.target, target);
            let mut bound = 1;
            if let Some(atlas) = atlas {
                bindings[bound] = NativeResourceBinding::coverage_atlas(BINDINGS.atlas, atlas);
                bound += 1;
            }
            bindings[bound] = NativeResourceBinding::glyph_table(BINDINGS.glyphs, table);
            bound += 1;
            let mut slots = [0u32; MAX_BOUND_SURFACES];
            let mut slot_count = 0;
            for row in 0..self.program.rows() {
                if !matches!(
                    self.program.op(row),
                    Some(RasterOp::SurfaceBlit | RasterOp::SurfaceSample)
                ) {
                    continue;
                }
                let Some(record) = RasterRow::read(&self.rows[row * RASTER_ROW_WORDS..]) else {
                    continue;
                };
                if record.is_empty() || slots[..slot_count].contains(&record.aux[0]) {
                    continue;
                }
                let Some(surface) = source(record.aux[0]) else {
                    continue;
                };
                if slot_count == MAX_BOUND_SURFACES {
                    return Err(HostError::Execute);
                }
                slots[slot_count] = record.aux[0];
                slot_count += 1;
                bindings[bound] = NativeResourceBinding::xrgb_surface(
                    ResourceKey {
                        group: BINDINGS.surfaces,
                        binding: record.aux[0],
                    },
                    surface,
                );
                bound += 1;
            }
            // The row table crosses as bytes; `i32` has no padding.
            let rows_bytes = unsafe {
                core::slice::from_raw_parts(
                    self.rows.as_ptr().cast::<u8>(),
                    self.rows.len() * core::mem::size_of::<i32>(),
                )
            };
            self.prepared
                .run_with_resources(
                    &[rows_bytes, &self.dirty[..self.columns]],
                    &bindings[..bound],
                )
                .map_err(|_| HostError::Execute)?;
            let mut c = 0;
            while c < self.columns {
                if (painted[c / 64] >> (c % 64)) & 1 == 0 {
                    c += 1;
                    continue;
                }
                let first = c;
                while c < self.columns && (painted[c / 64] >> (c % 64)) & 1 == 1 {
                    c += 1;
                }
                let x0 = first * TILE;
                let x1 = (c * TILE).min(width);
                for y in 0..band_h {
                    fb[(band_y + y) * width + x0..(band_y + y) * width + x1]
                        .copy_from_slice(&self.stage[y * width + x0..y * width + x1]);
                }
            }
        }
        let mask = TileMask {
            bits: frame.dirty().bits,
        };
        drop(frame);
        scene.commit_frame();
        Ok(mask)
    }
}
