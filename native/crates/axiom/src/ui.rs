//! Retained, build-compiled UI programs shared by Kore clients.
//!
//! UI state remains with the client or compositor that owns it. This module
//! owns the stable shader ABI, resource validation, prepared execution arena,
//! and packed output so applications do not grow private graph-compilation or
//! planar-packing paths. Immutable byte resources are borrowed for one draw;
//! no cell, glyph, atlas, or framebuffer payload is copied into the graph.

use ax_exec::NativeResourceBinding;
use axnn::lowering::{KernelPolicy, ResourceKey};
use axnn_cpu::U8BufferRef;
use core::alloc::Allocator;

use crate::{Error, program::Program};

pub const TERMINAL_TILE_WIDTH: usize = 128;
pub const TERMINAL_TILE_HEIGHT: usize = 32;
/// A 3840×2160 aperture at the minimum 50% density has 6×12 pixel cells.
/// Keep the shader record stride and build-time resource shape in sync.
pub const TERMINAL_MAX_COLUMNS: usize = 640;
pub const TERMINAL_MAX_ROWS: usize = 180;
pub const TERMINAL_CELL_RECORD_BYTES: usize = 10;
/// A cell record's flags (its third byte): an underline under the glyph,
/// read from the record of the character that owns the cell.
pub const TERMINAL_CELL_UNDERLINE: u8 = 1;
/// The line cursor, drawn in the accent along the bottom of the cell. It is
/// read from the cell's own record, so on a wide character's second half it
/// marks that half.
pub const TERMINAL_CELL_CURSOR: u8 = 2;
pub const TERMINAL_CELL_BYTES: usize =
    TERMINAL_MAX_COLUMNS * TERMINAL_MAX_ROWS * TERMINAL_CELL_RECORD_BYTES;
/// 224 separately rasterized Latin glyphs, 12x24 baseline cells, u16 coverage.
pub const fn terminal_atlas_bytes(density: kore_window::appearance::Density) -> usize {
    224 * density.pixels(12) as usize * density.pixels(24) as usize * 2
}
pub const fn terminal_atlas_offset(density: kore_window::appearance::Density) -> usize {
    // Each quarter q has 3q by 6q cells. Sum squares of preceding q=2..n-1.
    let n = density.quarters() as usize - 1;
    224 * 3 * 6 * 2 * (n * (n + 1) * (2 * n + 1) / 6 - 1)
}
pub const TERMINAL_ATLAS_BYTES: usize =
    terminal_atlas_offset(kore_window::appearance::Density::MAX)
        + terminal_atlas_bytes(kore_window::appearance::Density::MAX);

const TERMINAL_GRAPH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/terminal_tile.ax"));
const CELLS: ResourceKey = ResourceKey {
    group: 1,
    binding: 0,
};
const ATLAS: ResourceKey = ResourceKey {
    group: 1,
    binding: 1,
};

#[derive(Clone, Copy)]
pub struct TerminalTileUniforms {
    pub origin: [usize; 2],
    pub viewport: [usize; 2],
    pub grid: [usize; 2],
    pub paper: u32,
    pub accent: u32,
    pub divider: u32,
    /// Zero selects the terminal title; one selects the editor title.
    pub title: u32,
    /// Native pixels per UI unit; selects separately rasterized glyph coverage.
    pub density: kore_window::appearance::Density,
    /// The scrollbar's thumb — `[x0, y0, x1, y1)` in viewport pixels — and
    /// its colour; `None` draws no scrollbar.
    pub thumb: Option<([usize; 4], u32)>,
}

#[derive(Clone, Copy)]
pub struct TerminalTileResources<'a> {
    pub cells: &'a [u8],
    pub atlas: &'a [u8],
}

/// One allocation-free repeated lane for the canonical terminal UI tile.
/// Forks share immutable instructions and retain independent mutable tensors.
pub struct TerminalTileProgram<A: Allocator> {
    prepared: ax_exec::PreparedNative<A>,
    x: [[u8; 4]; TERMINAL_TILE_WIDTH],
    y: [[u8; 4]; TERMINAL_TILE_HEIGHT],
    uniforms: [[u8; 4]; UNIFORMS],
}

/// Floats the tile program's uniforms carry: seven `vec4`s.
const UNIFORMS: usize = 28;

impl<A: Allocator + Clone> TerminalTileProgram<A> {
    pub fn new_in(policy: KernelPolicy, allocator: A) -> Result<Self, Error> {
        let program = Program::compile_in(TERMINAL_GRAPH, policy, allocator.clone())?;
        let prepared = program.prepare_native_in(allocator)?;
        let x = core::array::from_fn(|i| (i as f32 + 0.5).to_le_bytes());
        let y = core::array::from_fn(|i| (i as f32 + 0.5).to_le_bytes());
        Ok(Self {
            prepared,
            x,
            y,
            uniforms: [[0; 4]; UNIFORMS],
        })
    }

    pub fn fork_in(&self, allocator: A) -> Result<Self, Error> {
        Ok(Self {
            prepared: self.prepared.fork_in(allocator).map_err(map_exec)?,
            x: self.x,
            y: self.y,
            uniforms: [[0; 4]; UNIFORMS],
        })
    }

    pub fn render(
        &mut self,
        state: TerminalTileUniforms,
        resources: TerminalTileResources<'_>,
    ) -> Result<&[i32], Error> {
        if resources.cells.len() != TERMINAL_CELL_BYTES
            || resources.atlas.len() != TERMINAL_ATLAS_BYTES
            || state.grid[0] > TERMINAL_MAX_COLUMNS
            || state.grid[1] > TERMINAL_MAX_ROWS
        {
            return Err(Error::InvalidBinding);
        }
        let colors = |word: u32| {
            [
                ((word >> 16) & 0xff) as f32,
                ((word >> 8) & 0xff) as f32,
                (word & 0xff) as f32,
            ]
        };
        let paper = colors(state.paper);
        let accent = colors(state.accent);
        let divider = colors(state.divider);
        let (thumb, thumb_ink, thumb_shown) = match state.thumb {
            Some((rect, ink)) => (rect.map(|v| v as f32), colors(ink), 1.0),
            None => ([0.0; 4], [0.0; 3], 0.0),
        };
        let values = [
            state.origin[0] as f32,
            state.origin[1] as f32,
            state.viewport[0] as f32,
            state.viewport[1] as f32,
            state.grid[0] as f32,
            state.grid[1] as f32,
            // Free: the cursor is a cell flag (`TERMINAL_CELL_CURSOR`).
            0.0,
            0.0,
            0.0,
            paper[0],
            paper[1],
            paper[2],
            accent[0],
            accent[1],
            accent[2],
            state.density.as_f32(),
            divider[0],
            divider[1],
            divider[2],
            state.title as f32,
            thumb[0],
            thumb[1],
            thumb[2],
            thumb[3],
            thumb_ink[0],
            thumb_ink[1],
            thumb_ink[2],
            thumb_shown,
        ];
        for (dst, value) in self.uniforms.iter_mut().zip(values) {
            *dst = value.to_le_bytes();
        }
        let bindings = [
            NativeResourceBinding::u8_buffer(
                CELLS,
                U8BufferRef::new(resources.cells).map_err(|_| Error::InvalidBinding)?,
            ),
            NativeResourceBinding::u8_buffer(
                ATLAS,
                U8BufferRef::new(resources.atlas).map_err(|_| Error::InvalidBinding)?,
            ),
        ];
        let inputs: [&[u8]; UNIFORMS + 2] = core::array::from_fn(|i| match i {
            0 => self.x.as_flattened(),
            1 => self.y.as_flattened(),
            _ => &self.uniforms[i - 2],
        });
        self.prepared
            .run_with_resources(&inputs, &bindings)
            .map_err(map_exec)?;
        self.prepared.output_i32(0).ok_or(Error::InvalidOutput)
    }

    pub fn resident_bytes(&self) -> usize {
        self.prepared.buffer_bytes() + self.prepared.scratch_bytes()
    }
}

fn map_exec(error: ax_exec::Error) -> Error {
    match error {
        ax_exec::Error::Ax(ax::AllocationError::AllocationFailed) => Error::Allocation,
        ax_exec::Error::MissingTensor(id) => Error::MissingValue(id.0),
        _ => Error::InvalidBinding,
    }
}
