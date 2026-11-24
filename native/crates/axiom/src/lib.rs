//! Kore's runtime adapter for Axiom's cross-platform graph layer.
//!
//! The adapter owns three deliberately separate responsibilities:
//!
//! 1. `axrd` lowers retained scene/layout state to canonical `.ax` bytes;
//! 2. `ax` analyzes, places, and partitions that graph through a Kore cost
//!    model; and
//! 3. `axnn` selects typed CPU kernel plans and the admitted CPU lane executes
//!    those plans; `accelerator` admits trusted offline artifacts for explicit
//!    graph execution through a possessed Accelerator client face.
//!
//! Every allocation is made through the allocator supplied by the final
//! runtime. This crate does not name `Global`, install an allocator, or hide a
//! heap assumption behind a convenience constructor.

#![no_std]
#![feature(allocator_api)]
#![cfg_attr(test, feature(portable_simd))]

extern crate alloc;

pub mod accelerator;
pub mod boot;
pub mod cursor;
pub mod pointer;
pub mod program;
pub mod spmd;
pub mod xe_batch;
pub mod xe_executor;
pub mod xe_graph;
pub mod xe_moe;
/// Architecture-level Xe artifacts callable through the possessed accelerator
/// face, without a caller-visible Xe2/Xe3 package choice.
pub use kore_xe_kernels as xe;
pub mod ui;

#[cfg(test)]
extern crate std;

use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::{CostModel, Graph, NodeView, Schedule, TensorView};
use axrd::graph::AlphaAtlas;
use axrd::{Anchor, Bind, LayoutParams, Offset, Parallax, PlacedGlyph, Rect, Scene};
use embedded_bitmap_fonts::terminus::{FONT_12x24, FONT_12x24_RAW};

/// The first Kore provider domain. Further domains are admitted by extending
/// the cost model and providing a partition executor; scene code is unchanged.
pub const CPU_LANE: u16 = 0;

/// Adapter failure, kept independent of provider-specific error types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Allocation,
    Scene,
    Graph,
    Schedule,
    UnsupportedDType,
    UnsupportedShape,
    UnsupportedOperation(u16),
    MissingValue(u32),
    InvalidBinding,
    InvalidOutput,
}

/// One allocator-owned, analyzed and placed Axiom frame graph.
pub struct Frame<A: Allocator> {
    width: u32,
    height: u32,
    input_count: u8,
    program: program::Program<A>,
    allocator: A,
}

impl<A: Allocator> Frame<A> {
    pub fn graph(&self) -> Graph<'_> {
        self.program.graph()
    }

    pub fn schedule(&self) -> &Schedule<A> {
        self.program.schedule()
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn input_count(&self) -> u8 {
        self.input_count
    }

    pub fn encoded_bytes(&self) -> usize {
        self.program.as_bytes().len()
    }
}

/// Three normalized F32 display planes produced by an admitted frame graph.
pub struct PlanarFrame<A: Allocator> {
    pub width: u32,
    pub height: u32,
    pub planes: [Vec<f32, A>; 3],
}

/// CPU-lane telemetry from Axiom's actual scheduler, not a guessed backend
/// label attached after rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    pub nodes: usize,
    pub partitions: usize,
    pub aggregate_cost: u64,
}

/// The initial Kore cost surface. Admission and costs come from the same
/// typed kernel catalog used by execution, never a second opcode allowlist.
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreCostModel;

fn kernel_policy() -> axnn::lowering::KernelPolicy {
    // The kernel's build target is the CPU contract; no host feature probing.
    axnn::lowering::KernelPolicy::new(axnn_cpu::CpuCapabilities::compiled().mask())
        .with_minimum_accuracy(axnn::lowering::KernelAccuracy::Faithful)
}

impl CostModel for KoreCostModel {
    fn backend_count(&self) -> u16 {
        1
    }

    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        program::CatalogModel {
            policy: kernel_policy(),
        }
        .supports(backend, graph, node)
    }

    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        program::CatalogModel {
            policy: kernel_policy(),
        }
        .execute_cost(backend, graph, node)
    }

    fn transfer_cost(&self, tensor: TensorView<'_>, source: Option<u16>, destination: u16) -> u64 {
        program::CatalogModel {
            policy: kernel_policy(),
        }
        .transfer_cost(tensor, source, destination)
    }

    fn external_output_backend(&self) -> Option<u16> {
        Some(CPU_LANE)
    }
}

/// Build Kore's boot surface through `axrd`, then run the complete generic Ax
/// placement pipeline. The geometry is proportional to the physical mode;
/// its parallax coefficients remain physical-pixel displacements.
pub fn compile_boot_in<A>(
    width: u16,
    height: u16,
    allocator: A,
) -> Result<(Frame<A>, Placement), Error>
where
    A: Allocator + Clone,
{
    compile_boot_region_in(width, height, 0, height, allocator)
}

/// Build one horizontal region of Kore's boot surface while retaining the
/// full physical viewport for layout, lighting, and depth math.
///
/// Region execution is the bounded-memory presentation seam: a runtime can
/// schedule and immediately pack consecutive regions into one surface without
/// ever retaining a mode-sized graph and three mode-sized output planes at
/// once. The region still contains ordinary full-resolution Ax tensors; this
/// is not a reduced-resolution render or a post-render scale.
pub fn compile_boot_region_in<A>(
    width: u16,
    height: u16,
    row: u16,
    rows: u16,
    allocator: A,
) -> Result<(Frame<A>, Placement), Error>
where
    A: Allocator + Clone,
{
    if width == 0
        || height == 0
        || rows == 0
        || row
            .checked_add(rows)
            .is_none_or(|region_end| region_end > height)
    {
        return Err(Error::UnsupportedShape);
    }
    let mut scene = Scene::<32>::new();
    let root = scene
        // Resolve the complete physical scene against the region-local
        // output. Children keep their full-mode layout while paints are
        // clipped to rows `row..row + rows` at graph preparation time.
        .add_root(layout(0, -i32::from(row), width, height), Bind::Composite)
        .map_err(|_| Error::Scene)?;

    // A restrained right-hand field establishes the deep plane.
    let orb = (u32::from(width.min(height)) * 54 / 100).max(24) as u16;
    let orb_y = i32::from(height) / 2 - i32::from(orb) / 2;
    if intersects_region(orb_y, orb, row, rows) {
        let orb_node = scene
            .add_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::CENTER_RIGHT,
                    offset: Offset {
                        x: -(i32::from(orb) + i32::from(width) / 18),
                        y: -(i32::from(orb) / 2),
                    },
                    w: orb,
                    h: orb,
                },
                Bind::RoundedFill {
                    color: 0x00E4865E,
                    radii: axrd::Corners::uniform(orb / 2),
                },
                236,
            )
            .map_err(|_| Error::Scene)?;
        scene
            .set_parallax(orb_node, Parallax::new(-1024, 512))
            .map_err(|_| Error::Scene)?;
    }

    // The foreground participates in depth, but with a deliberately shallower
    // coefficient than the luminous field.
    let card_w = (u32::from(width) * 32 / 100).max(120).min(u32::from(width)) as u16;
    let card_h = (u32::from(height) * 18 / 100)
        .max(64)
        .min(u32::from(height)) as u16;
    let card_y = i32::from(height) / 2 - i32::from(card_h) / 2;
    let card = if intersects_region(card_y, card_h, row, rows) {
        let card = scene
            .add_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::CENTER_LEFT,
                    offset: Offset {
                        x: i32::from(width) / 10,
                        y: -(i32::from(card_h) / 2),
                    },
                    w: card_w,
                    h: card_h,
                },
                Bind::RoundedFill {
                    color: 0x001A211B,
                    radii: axrd::Corners::uniform((height / 50).max(4)),
                },
                224,
            )
            .map_err(|_| Error::Scene)?;
        scene
            .set_parallax(card, Parallax::new(320, -192))
            .map_err(|_| Error::Scene)?;
        Some(card)
    } else {
        None
    };

    let rail_w = (u32::from(width) * 28 / 100).max(96).min(u32::from(width)) as u16;
    let rail_y = i32::from(height) - i32::from(height) / 8;
    if intersects_region(rail_y, 3, row, rows) {
        let rail = scene
            .add_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::BOTTOM_LEFT,
                    offset: Offset {
                        x: i32::from(width) / 10,
                        y: -(i32::from(height) / 8),
                    },
                    w: rail_w,
                    h: 3,
                },
                Bind::RoundedFill {
                    color: 0x00D9D0BF,
                    radii: axrd::Corners::uniform(1),
                },
                144,
            )
            .map_err(|_| Error::Scene)?;
        scene
            .set_parallax(rail, Parallax::new(192, -96))
            .map_err(|_| Error::Scene)?;
    }

    // Kore's freestanding first frame uses the same proven Terminus atlas as
    // its console. It enters axrd as ordinary high-precision text coverage;
    // neither the scheduler nor a future accelerator lane knows that the
    // source happened to be a bitmap face. The temporary atlas and glyph
    // vectors use the runtime's allocator and are copied into graph constants.
    let mut coverage = Vec::new_in(allocator.clone());
    let mut glyphs = Vec::new_in(allocator.clone());
    if width >= 640 && height >= 360 {
        let brand_y = i32::from(height) / 10;
        let prompt_y = i32::from(height) - i32::from(height) / 8 - 34;
        let brand_visible = intersects_region(brand_y, 24, row, rows);
        let title_visible = card.is_some() && intersects_region(card_y + 12, 24, row, rows);
        let state_visible = card.is_some() && intersects_region(card_y + 40, 24, row, rows);
        let prompt_visible = intersects_region(prompt_y, 24, row, rows);

        // Most physical regions contain no text. Build the atlas only for a
        // region that can consume it, and copy only the glyph records that
        // survive clipping into that region's graph constants.
        if brand_visible || title_visible || state_visible || prompt_visible {
            build_boot_atlas(&mut coverage)?;
        }

        if brand_visible {
            let brand = append_text(&mut glyphs, "AXIOM / KORE")?;
            let brand_node = scene
                .add_with_opacity(
                    root,
                    layout(i32::from(width) / 10, brand_y, 144, 24),
                    text_bind(brand, 0x00D9D0BF),
                    144,
                )
                .map_err(|_| Error::Scene)?;
            scene
                .set_parallax(brand_node, Parallax::new(192, -96))
                .map_err(|_| Error::Scene)?;
        }

        if let Some(card) = card {
            if title_visible {
                let title = append_text(&mut glyphs, "SYSTEM COMPOSITOR")?;
                let node = scene
                    .add_with_opacity(
                        card,
                        layout(18, 12, card_w.saturating_sub(28), 24),
                        text_bind(title, 0x00D9D0BF),
                        224,
                    )
                    .map_err(|_| Error::Scene)?;
                scene
                    .set_parallax(node, Parallax::new(192, -96))
                    .map_err(|_| Error::Scene)?;
            }
            if state_visible {
                let state = append_text(&mut glyphs, "AX GRAPH ONLINE")?;
                let node = scene
                    .add_with_opacity(
                        card,
                        layout(18, 40, card_w.saturating_sub(28), 24),
                        text_bind(state, 0x00D9D0BF),
                        120,
                    )
                    .map_err(|_| Error::Scene)?;
                scene
                    .set_parallax(node, Parallax::new(192, -96))
                    .map_err(|_| Error::Scene)?;
            }
        }

        if prompt_visible {
            let prompt = append_text(&mut glyphs, "PRESS ANY KEY")?;
            let prompt_node = scene
                .add_with_opacity(
                    root,
                    layout(i32::from(width) / 10, prompt_y, 156, 24),
                    text_bind(prompt, 0x00D9D0BF),
                    160,
                )
                .map_err(|_| Error::Scene)?;
            scene
                .set_parallax(prompt_node, Parallax::new(192, -96))
                .map_err(|_| Error::Scene)?;
        }
    }

    let radius = f32::from(width.min(height)).max(1.0);
    let background = axrd::graph::GraphBackground::Radial {
        base: 0x00131B16,
        lights: [
            axrd::graph::GraphRadialLight {
                x: f32::from(width) * 0.68,
                y: f32::from(height) * 0.46 - f32::from(row),
                radius: radius * 0.72,
                color: 0x00566B45,
                strength: 0.34,
                profile: axrd::graph::GraphRadialProfile::Rational { sharpness: 1.35 },
                rim: None,
            },
            axrd::graph::GraphRadialLight {
                x: f32::from(width) * 0.78,
                y: f32::from(height) * 0.42 - f32::from(row),
                radius: radius * 0.34,
                color: 0x00F0AC82,
                strength: 0.30,
                profile: axrd::graph::GraphRadialProfile::Rational { sharpness: 2.0 },
                rim: None,
            },
            axrd::graph::GraphRadialLight {
                x: f32::from(width) * 0.28,
                y: f32::from(height) * 0.72 - f32::from(row),
                radius: radius * 0.58,
                color: 0x002D4935,
                strength: 0.22,
                profile: axrd::graph::GraphRadialProfile::Rational { sharpness: 1.0 },
                rim: None,
            },
        ],
    };
    let atlas = (!coverage.is_empty())
        .then(|| AlphaAtlas::from_u16(&coverage, BOOT_ATLAS_WIDTH, BOOT_ATLAS_HEIGHT));
    let frame = axrd::graph::compile_scene_graph_with_output_in(
        &mut scene,
        Rect {
            x: 0,
            y: 0,
            w: width,
            h: rows,
        },
        background,
        atlas,
        &glyphs,
        axrd::graph::GraphOutputFormat::PlanarF32,
        allocator.clone(),
    )
    .map_err(|_| Error::Graph)?;

    let (frame_width, frame_height, input_count) =
        (frame.width(), frame.height(), frame.input_count());
    let program =
        program::Program::from_graph_in(frame.into_graph(), kernel_policy(), allocator.clone())?;
    let placement = program.placement();
    Ok((
        Frame {
            width: frame_width,
            height: frame_height,
            input_count,
            program,
            allocator,
        },
        placement,
    ))
}

const BOOT_GLYPH_WIDTH: usize = 12;
const BOOT_GLYPH_HEIGHT: usize = 24;
const BOOT_GLYPHS_PER_ROW: usize = 16;
const BOOT_ATLAS_WIDTH: usize = BOOT_GLYPH_WIDTH * BOOT_GLYPHS_PER_ROW;
const BOOT_ATLAS_HEIGHT: usize = FONT_12x24_RAW.len() * 8 / BOOT_ATLAS_WIDTH;

fn build_boot_atlas<A: Allocator>(coverage: &mut Vec<u16, A>) -> Result<(), Error> {
    coverage
        .try_reserve_exact(BOOT_ATLAS_WIDTH * BOOT_ATLAS_HEIGHT)
        .map_err(|_| Error::Allocation)?;
    for bit in 0..FONT_12x24_RAW.len() * 8 {
        let sample = FONT_12x24_RAW[bit >> 3] & (0x80 >> (bit & 7));
        coverage.push(if sample == 0 { 0 } else { u16::MAX });
    }
    Ok(())
}

fn append_text<A: Allocator>(
    glyphs: &mut Vec<PlacedGlyph, A>,
    text: &str,
) -> Result<(u16, u16), Error> {
    let start = u16::try_from(glyphs.len()).map_err(|_| Error::InvalidBinding)?;
    glyphs
        .try_reserve_exact(text.chars().count())
        .map_err(|_| Error::Allocation)?;
    for (column, character) in text.chars().enumerate() {
        let index = FONT_12x24.glyph_mapping.index(character);
        glyphs.push(PlacedGlyph {
            ax: u16::try_from(index % BOOT_GLYPHS_PER_ROW * BOOT_GLYPH_WIDTH)
                .map_err(|_| Error::InvalidBinding)?,
            ay: u16::try_from(index / BOOT_GLYPHS_PER_ROW * BOOT_GLYPH_HEIGHT)
                .map_err(|_| Error::InvalidBinding)?,
            aw: BOOT_GLYPH_WIDTH as u16,
            ah: BOOT_GLYPH_HEIGHT as u16,
            dx: i16::try_from(column * BOOT_GLYPH_WIDTH).map_err(|_| Error::InvalidBinding)?,
            dy: 0,
            subpixel_x: 0,
            subpixel_y: 0,
        });
    }
    let len =
        u16::try_from(glyphs.len() - usize::from(start)).map_err(|_| Error::InvalidBinding)?;
    Ok((start, len))
}

const fn text_bind((glyph_start, glyph_len): (u16, u16), color: u32) -> Bind {
    Bind::TextRun {
        glyph_start,
        glyph_len,
        color,
    }
}

const fn layout(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
    LayoutParams {
        anchor: Anchor::TOP_LEFT,
        offset: Offset { x, y },
        w,
        h,
    }
}

/// Whether a scene object in full-viewport coordinates can paint this output
/// region. Region-local graphs omit fully offscreen objects before handing the
/// scene to axrd: besides avoiding useless graph constants, this keeps Kore on
/// released Axiom revisions that predate the generic offscreen-clip fix.
fn intersects_region(y: i32, height: u16, row: u16, rows: u16) -> bool {
    let region_start = i32::from(row);
    let region_end = region_start + i32::from(rows);
    y < region_end && y + i32::from(height) > region_start
}

impl<A: Allocator + Clone> Frame<A> {
    /// Execute the retained, automatically selected kernel plan. Inputs remain
    /// the two normalized shallow-depth scalars declared by axrd.
    pub fn execute(&self, pointer: [f32; 2]) -> Result<PlanarFrame<A>, Error> {
        let count = usize::from(self.input_count);
        if !matches!(count, 0 | 2) {
            return Err(Error::InvalidBinding);
        }
        let inputs = [
            program::Input::F32(&pointer[..1]),
            program::Input::F32(&pointer[1..]),
        ];
        let outputs = self
            .program
            .run_in(&inputs[..count], self.allocator.clone())?;
        if outputs.len() != 3 {
            return Err(Error::InvalidOutput);
        }
        let mut outputs = outputs.into_iter();
        let mut plane = || match outputs.next() {
            Some(program::Output::F32(values)) => Ok(values),
            _ => Err(Error::InvalidOutput),
        };
        Ok(PlanarFrame {
            width: self.width(),
            height: self.height(),
            planes: [plane()?, plane()?, plane()?],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Global;

    #[test]
    fn boot_scene_is_an_actual_scheduled_ax_graph() {
        let (frame, placement) = compile_boot_in(64, 48, Global).expect("compile");
        assert!(placement.nodes > 0);
        assert_eq!(placement.partitions, 1);
        assert!(
            frame
                .schedule()
                .placements()
                .iter()
                .all(|&lane| lane == CPU_LANE)
        );
        assert_eq!(frame.graph().node_count(), placement.nodes);
        assert_eq!(frame.input_count(), 2);
    }

    #[test]
    fn scheduled_graph_executes_to_three_finite_display_planes() {
        let (frame, _) = compile_boot_in(40, 30, Global).expect("compile");
        let rendered = frame.execute([0.125, -0.25]).expect("execute");
        assert_eq!((rendered.width, rendered.height), (40, 30));
        for plane in &rendered.planes {
            assert_eq!(plane.len(), 40 * 30);
            assert!(plane.iter().all(|value| value.is_finite()));
            assert!(plane.iter().all(|&value| (0.0..=1.0).contains(&value)));
        }
    }

    #[test]
    fn horizontal_regions_stitch_to_the_full_neutral_frame() {
        let (full, _) = compile_boot_in(40, 30, Global).expect("full compile");
        let full = full.execute([0.0, 0.0]).expect("full execute");
        for (row, rows) in [(0, 11), (11, 19)] {
            let (region, _) =
                compile_boot_region_in(40, 30, row, rows, Global).expect("region compile");
            let region = region.execute([0.0, 0.0]).expect("region execute");
            for (whole, part) in full.planes.iter().zip(&region.planes) {
                let start = usize::from(row) * 40;
                let end = start + usize::from(rows) * 40;
                assert_eq!(&whole[start..end], part);
            }
        }
    }

    #[test]
    fn background_only_region_executes_without_pointer_inputs() {
        let (frame, placement) =
            compile_boot_region_in(1280, 800, 0, 6, Global).expect("region compile");
        assert_eq!(frame.input_count(), 0);
        assert!(placement.nodes > 0);

        let rendered = frame.execute([0.5, -0.5]).expect("region execute");
        assert_eq!((rendered.width, rendered.height), (1280, 6));
        assert!(rendered.planes.iter().all(|plane| plane.len() == 1280 * 6));
    }

    /// Manual release-mode smoke for the same mode-sized allocation and
    /// execution shape the display service sees. Kept ignored so ordinary
    /// unit suites remain fast; the integration PR runs it explicitly.
    #[test]
    #[ignore = "mode-sized allocator and execution smoke"]
    fn mode_sized_boot_executes_through_the_scheduler() {
        let (frame, placement) = compile_boot_in(960, 600, Global).expect("compile");
        std::eprintln!(
            "axiom boot: {} nodes, {} partition(s), {} encoded bytes",
            placement.nodes,
            placement.partitions,
            frame.encoded_bytes()
        );
        let rendered = frame.execute([0.0, 0.0]).expect("execute");
        assert_eq!((rendered.width, rendered.height), (960, 600));
        assert!(rendered.planes.iter().all(|plane| plane.len() == 960 * 600));
    }
}
