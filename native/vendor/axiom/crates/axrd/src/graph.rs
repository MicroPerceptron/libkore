//! Backend-neutral scene compositing expressed as a canonical `.ax` graph.
//!
//! Layout and glyph rasterization produce immutable device-space coverage.
//! This module prepares consecutive paints as exact-resolution layers, then
//! emits their dynamic translation and source-over stages as mixed-precision
//! graph arithmetic:
//!
//! `out = source_premultiplied + destination * (1 - source_alpha)`
//!
//! Fill SDFs, text coverage, paint order, and opacity are invariant across
//! retained submissions, so preparing them once is ordinary graph constant
//! folding rather than a runtime CPU renderer. A layer with parallax metadata
//! carries a prepared F32 coverage derivative; normalized pointer scalars
//! drive the canonical linearized translation in the graph. Coverage and
//! premultiplication stay F32; only the final display boundary narrows to the
//! selected planar format. Placement, compilation, fusion, and execution
//! therefore belong to `ax` and its provider adapters; adding a backend never
//! requires an `axrd` raster-kernel implementation.

use alloc::vec::Vec;
use core::alloc::Allocator;
#[allow(unused_imports)]
// supplies f32::sqrt for no_std builds; std tests have an inherent method
use core_maths::CoreFloat as _;

use ax::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, LinearizedField2d, NodeFlags, OpCode,
    PremultipliedRgba, TensorFlags, TensorId,
};

use crate::{Bind, Parallax, PlacedGlyph, Rect, Scene};

/// The surface below the UI layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GraphBackground {
    /// Embed a solid `0x00RRGGBB` colour in the graph.
    Solid(u32),
    /// Expose three F32 `[height, width]` inputs in R, G, B order.
    ///
    /// This is the handoff for another `.ax` shader graph or an external
    /// render surface. F32 is also the lossless canonical boundary for UF16,
    /// which does not currently have its own graph dtype. Inputs and outputs
    /// use normalized sRGB channel values.
    Inputs,
    /// A solid base illuminated by three soft device-space radial lights.
    ///
    /// Coordinates and radii are physical pixels. The field is evaluated by
    /// ordinary F32 `.ax` operations and can therefore share placement and
    /// execution policy with the rest of the scene instead of arriving as a
    /// host-rendered bitmap.
    Radial {
        base: u32,
        lights: [GraphRadialLight; 3],
    },
    /// A radial field whose centers and strengths are supplied at execution.
    ///
    /// The input ABI is nine F32 `[1, 1]` tensors in light order, with
    /// `x`, `y`, then `strength` for each light. This keeps interaction state
    /// small and lets a compiled executable be retained while pointer or
    /// animation input changes from frame to frame.
    RadialInputs {
        base: u32,
        lights: [GraphRadialStyle; 3],
    },
}

/// One soft light in a [`GraphBackground::Radial`] field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphRadialLight {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color: u32,
    /// Blend strength at the light center, normally in `0..=1`.
    pub strength: f32,
    /// Spatial response of the field.
    pub profile: GraphRadialProfile,
    /// Optional critical-radius lens/throat emphasis.
    pub rim: Option<GraphRadialRim>,
}

/// The compile-time portion of an interactive radial light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphRadialStyle {
    pub radius: f32,
    pub color: u32,
    /// Spatial response of the field.
    pub profile: GraphRadialProfile,
    /// Optional critical-radius lens/throat emphasis.
    pub rim: Option<GraphRadialRim>,
}

/// Reusable radial geometry lowered to ordinary `.ax` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GraphRadialProfile {
    /// `1 / (1 + (d² / r²)^sharpness)`, suitable for environmental light.
    Rational { sharpness: f32 },
    /// A filled circle with an exact signed-distance transition.
    ///
    /// `softness` is the full physical-pixel width of the antialiased edge.
    /// The interior remains uniform so separate fields can model highlight or
    /// shadow without dissolving the object's silhouette.
    Disc { softness: f32 },
}

/// A soft annular emphasis centered on a radial field's nominal radius.
///
/// This is a deliberately small field primitive rather than a renderer
/// effect: it is expressed with canonical arithmetic, moves with the same
/// runtime center/strength inputs, and can be scheduled beside GR or inference
/// graphs. `strength` is relative to the parent light; `sharpness` controls
/// how tightly the response hugs the critical radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphRadialRim {
    pub strength: f32,
    pub sharpness: f32,
}

/// Storage format of the three graph output planes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GraphOutputFormat {
    /// Normalized sRGB channels retained as IEEE F32.
    ///
    /// This is the lossless provider/display handoff and avoids adding a
    /// precision-conversion tail to providers that do not implement `CAST`.
    PlanarF32,
    /// Normalized sRGB channels stored as IEEE F16.
    #[default]
    PlanarF16,
    /// Display-ready sRGB bytes stored in three TOSA INT8 planes.
    ///
    /// TOSA has no numeric unsigned-byte type (`U8` in `.ax` is its Boolean
    /// storage), so each byte is biased by -128 before the final cast. Reading
    /// the stored byte with wrapping `+ 128` recovers the ordinary `0..=255`
    /// channel. Composition and antialiasing remain F32 before this boundary.
    PlanarU8,
}

/// Borrowed alpha atlas used while compiling text coverage planes.
#[derive(Clone, Copy, Debug)]
pub struct AlphaAtlas<'a> {
    samples: AlphaSamples<'a>,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug)]
enum AlphaSamples<'a> {
    PackedU8(&'a [u32]),
    LinearU16(&'a [u16]),
}

impl<'a> AlphaAtlas<'a> {
    /// Read coverage from the alpha byte of packed compatibility pixels.
    pub const fn new(pixels: &'a [u32], width: usize, height: usize) -> Self {
        Self {
            samples: AlphaSamples::PackedU8(pixels),
            width,
            height,
        }
    }

    /// Read high-precision linear coverage spanning `0..=65535`.
    pub const fn from_u16(coverage: &'a [u16], width: usize, height: usize) -> Self {
        Self {
            samples: AlphaSamples::LinearU16(coverage),
            width,
            height,
        }
    }

    /// The linear coverage plane of a glyph atlas, so the graph path and
    /// the immediate path consume the same samples.
    #[cfg(feature = "font")]
    pub fn from_glyph_atlas<S: crate::font::AtlasStorage>(
        atlas: &'a crate::font::GlyphAtlas<S>,
    ) -> Self {
        Self::from_u16(
            atlas.coverage(),
            usize::from(atlas.width()),
            usize::from(atlas.height()),
        )
    }

    #[inline]
    fn len(self) -> usize {
        match self.samples {
            AlphaSamples::PackedU8(pixels) => pixels.len(),
            AlphaSamples::LinearU16(coverage) => coverage.len(),
        }
    }

    #[inline]
    fn coverage(self, index: usize) -> u16 {
        match self.samples {
            AlphaSamples::PackedU8(pixels) => ((pixels[index] >> 24) as u16) * 257,
            AlphaSamples::LinearU16(coverage) => coverage[index],
        }
    }
}

/// Failure while lowering an `axrd` scene into `.ax`.
#[derive(Debug)]
pub enum GraphError {
    Allocation(AllocationError),
    HostAllocation,
    InvalidFrameSize,
    PixelCountOverflow,
    InvalidAtlas,
    InvalidBackground,
    MissingAtlas,
    InvalidGlyphRange,
    AtlasBounds,
    UnsupportedBind { node: u16 },
}

impl From<AllocationError> for GraphError {
    fn from(value: AllocationError) -> Self {
        Self::Allocation(value)
    }
}

/// An owned canonical graph plus its framebuffer ABI.
pub struct AxFrameGraph<A: Allocator> {
    graph: GraphBuf<A>,
    width: u32,
    height: u32,
    background_inputs: u8,
    parallax_inputs: bool,
    output_format: GraphOutputFormat,
}

impl<A: Allocator> AxFrameGraph<A> {
    #[inline]
    pub fn graph(&self) -> Graph<'_> {
        self.graph
            .parse()
            .expect("AxFrameGraph is validated at build time")
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.graph.as_bytes()
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of F32 background tensors expected as graph inputs.
    ///
    /// [`GraphBackground::Inputs`] uses three framebuffer-sized planes;
    /// [`GraphBackground::RadialInputs`] uses nine scalar tensors.
    #[inline]
    pub const fn background_input_count(&self) -> u8 {
        self.background_inputs
    }

    /// Number of normalized pointer tensors appended after background inputs.
    #[inline]
    pub const fn parallax_input_count(&self) -> u8 {
        if self.parallax_inputs { 2 } else { 0 }
    }

    /// Complete ordered graph input count.
    ///
    /// Background inputs come first, followed by normalized pointer `x` and
    /// `y` scalars when the scene contains a parallax layer.
    #[inline]
    pub const fn input_count(&self) -> u8 {
        self.background_inputs + self.parallax_input_count()
    }

    #[inline]
    pub const fn output_format(&self) -> GraphOutputFormat {
        self.output_format
    }

    #[inline]
    pub fn into_graph(self) -> GraphBuf<A> {
        self.graph
    }
}

/// Compile solid/rounded fills and text runs into a portable mixed-precision
/// `.ax` compositor.
///
/// Phase-aware text coverage and physical-pixel fill SDFs are prepared once
/// into consecutive, ordered F32 layers. The scheduled graph performs any
/// shallow linearized translation and the reusable `ax` source-over regions
/// in F32, then narrows the final RGB display planes to
/// [`GraphOutputFormat::PlanarF16`].
/// `Blit`, `Sample`, and `Surface` nodes are rejected until their portable
/// surface ABI is defined.
pub fn compile_scene_graph_in<A: Allocator + Clone, const N: usize>(
    scene: &mut Scene<N>,
    root_rect: Rect,
    background: GraphBackground,
    atlas: Option<AlphaAtlas<'_>>,
    glyphs: &[PlacedGlyph],
    allocator: A,
) -> Result<AxFrameGraph<A>, GraphError> {
    compile_scene_graph_with_output_in(
        scene,
        root_rect,
        background,
        atlas,
        glyphs,
        GraphOutputFormat::PlanarF16,
        allocator,
    )
}

/// Compile a scene with an explicit portable display-plane boundary.
pub fn compile_scene_graph_with_output_in<A: Allocator + Clone, const N: usize>(
    scene: &mut Scene<N>,
    root_rect: Rect,
    background: GraphBackground,
    atlas: Option<AlphaAtlas<'_>>,
    glyphs: &[PlacedGlyph],
    output_format: GraphOutputFormat,
    allocator: A,
) -> Result<AxFrameGraph<A>, GraphError> {
    let width = u32::from(root_rect.w);
    let height = u32::from(root_rect.h);
    if width == 0 || height == 0 {
        return Err(GraphError::InvalidFrameSize);
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(GraphError::PixelCountOverflow)?;
    if let Some(atlas) = atlas {
        let expected = atlas
            .width
            .checked_mul(atlas.height)
            .ok_or(GraphError::InvalidAtlas)?;
        if atlas.width == 0 || atlas.height == 0 || atlas.len() != expected {
            return Err(GraphError::InvalidAtlas);
        }
    }

    scene.resolve_layout_host(root_rect);

    let dimensions = [height, width];
    let x_dimensions = [1, width];
    let y_dimensions = [height, 1];
    let scalar_dimensions = [1, 1];
    let mut builder = GraphBuilder::new_in(allocator.clone());
    let x_coordinates =
        coordinate_plane(&mut builder, width, true, &x_dimensions, allocator.clone())?;
    let y_coordinates = coordinate_plane(
        &mut builder,
        height,
        false,
        &y_dimensions,
        allocator.clone(),
    )?;
    let zero = scalar(&mut builder, 0.0, &scalar_dimensions)?;
    let x_zero = binary(
        &mut builder,
        OpCode::MUL,
        x_coordinates,
        zero,
        DType::F32,
        &x_dimensions,
    )?;
    let y_zero = binary(
        &mut builder,
        OpCode::MUL,
        y_coordinates,
        zero,
        DType::F32,
        &y_dimensions,
    )?;
    let domain = binary(
        &mut builder,
        OpCode::ADD,
        x_zero,
        y_zero,
        DType::F32,
        &dimensions,
    )?;
    let background_inputs;
    let mut channels = match background {
        GraphBackground::Inputs => {
            background_inputs = 3;
            [
                field_input(&mut builder, &dimensions)?,
                field_input(&mut builder, &dimensions)?,
                field_input(&mut builder, &dimensions)?,
            ]
        }
        GraphBackground::Solid(color) => {
            background_inputs = 0;
            solid_channels(&mut builder, domain, color, &scalar_dimensions, &dimensions)?
        }
        GraphBackground::Radial { base, lights } => {
            background_inputs = 0;
            let mut channels =
                solid_channels(&mut builder, domain, base, &scalar_dimensions, &dimensions)?;
            for light in lights {
                validate_radial_light(light)?;
                let center_x = scalar(&mut builder, light.x, &scalar_dimensions)?;
                let center_y = scalar(&mut builder, light.y, &scalar_dimensions)?;
                let strength = scalar(&mut builder, light.strength, &scalar_dimensions)?;
                apply_radial_light(
                    &mut builder,
                    &mut channels,
                    x_coordinates,
                    y_coordinates,
                    center_x,
                    center_y,
                    strength,
                    GraphRadialStyle {
                        radius: light.radius,
                        color: light.color,
                        profile: light.profile,
                        rim: light.rim,
                    },
                    &x_dimensions,
                    &y_dimensions,
                    &scalar_dimensions,
                    &dimensions,
                )?;
            }
            channels
        }
        GraphBackground::RadialInputs { base, lights } => {
            background_inputs = 9;
            let mut channels =
                solid_channels(&mut builder, domain, base, &scalar_dimensions, &dimensions)?;
            for light in lights {
                validate_radial_style(light)?;
                let center_x = field_input(&mut builder, &scalar_dimensions)?;
                let center_y = field_input(&mut builder, &scalar_dimensions)?;
                let strength = field_input(&mut builder, &scalar_dimensions)?;
                apply_radial_light(
                    &mut builder,
                    &mut channels,
                    x_coordinates,
                    y_coordinates,
                    center_x,
                    center_y,
                    strength,
                    light,
                    &x_dimensions,
                    &y_dimensions,
                    &scalar_dimensions,
                    &dimensions,
                )?;
            }
            channels
        }
    };
    // All current scene paints are immutable between prepared-executable
    // submissions. Collapse consecutive paints with one parallax coefficient
    // into exact-resolution premultiplied layers, then leave their dynamic
    // translation and source-over composition in `.ax`. This is the display
    // analogue of constant folding: it removes hundreds of repeated SDF and
    // blend nodes without introducing a backend-specific raster kernel.
    let overlays = prepare_static_overlays(
        scene,
        pixel_count,
        width as usize,
        height as usize,
        atlas,
        glyphs,
        allocator.clone(),
    )?;
    let parallax_inputs = overlays.iter().any(|overlay| !overlay.parallax.is_zero());
    let pointer = parallax_inputs
        .then(|| {
            Ok::<_, AllocationError>([
                field_input(&mut builder, &scalar_dimensions)?,
                field_input(&mut builder, &scalar_dimensions)?,
            ])
        })
        .transpose()?;
    for overlay in overlays {
        let prepared = emit_static_overlay(
            &mut builder,
            overlay,
            pointer,
            width as usize,
            height as usize,
            &scalar_dimensions,
            &dimensions,
            allocator.clone(),
        )?;
        channels = match prepared {
            EmittedStaticOverlay::Solid { alpha, colour } => {
                builder.solid_source_over_opaque_f32(alpha, colour, channels, &dimensions)?
            }
            EmittedStaticOverlay::Premultiplied(source) => {
                builder.premultiplied_source_over_opaque_f32(source, channels, &dimensions)?
            }
        };
    }

    let display_scale = (output_format == GraphOutputFormat::PlanarU8)
        .then(|| scalar(&mut builder, 255.0, &scalar_dimensions))
        .transpose()?;
    let display_bias = (output_format == GraphOutputFormat::PlanarU8)
        .then(|| scalar(&mut builder, -128.0, &scalar_dimensions))
        .transpose()?;
    for mut channel in channels {
        let (dtype, output_op) = match output_format {
            GraphOutputFormat::PlanarF32 => (DType::F32, OpCode::IDENTITY),
            GraphOutputFormat::PlanarF16 => (DType::F16, OpCode::CAST),
            GraphOutputFormat::PlanarU8 => {
                channel = binary(
                    &mut builder,
                    OpCode::MUL,
                    channel,
                    display_scale.expect("U8 display scale exists"),
                    DType::F32,
                    &dimensions,
                )?;
                channel = binary(
                    &mut builder,
                    OpCode::ADD,
                    channel,
                    display_bias.expect("U8 display bias exists"),
                    DType::F32,
                    &dimensions,
                )?;
                (DType::I8, OpCode::CAST)
            }
        };
        let output = builder.tensor(dtype, TensorFlags::OUTPUT, &dimensions, &[])?;
        builder.node(output_op, NodeFlags::default(), &[channel], &[output], &[])?;
    }

    Ok(AxFrameGraph {
        graph: builder.finish()?,
        width,
        height,
        background_inputs,
        parallax_inputs,
        output_format,
    })
}

struct StaticOverlay<A: Allocator> {
    parallax: Parallax,
    premultiplied: [Vec<f32, A>; 3],
    alpha: Vec<f32, A>,
    solid_color: Option<u32>,
    mixed_color: bool,
    painted: bool,
}

enum EmittedStaticOverlay {
    Solid { alpha: TensorId, colour: [f32; 3] },
    Premultiplied(PremultipliedRgba),
}

fn zero_f32_plane<A: Allocator>(
    pixel_count: usize,
    allocator: A,
) -> Result<Vec<f32, A>, GraphError> {
    let mut plane = Vec::new_in(allocator);
    plane
        .try_reserve_exact(pixel_count)
        .map_err(|_| GraphError::HostAllocation)?;
    plane.resize(pixel_count, 0.0);
    Ok(plane)
}

fn f32_plane_bytes<A: Allocator>(plane: &[f32], allocator: A) -> Result<Vec<u8, A>, GraphError> {
    let bytes = plane
        .len()
        .checked_mul(4)
        .ok_or(GraphError::PixelCountOverflow)?;
    let mut encoded = Vec::new_in(allocator);
    encoded
        .try_reserve_exact(bytes)
        .map_err(|_| GraphError::HostAllocation)?;
    for value in plane {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(encoded)
}

fn blend_static_pixel<A: Allocator>(
    overlay: &mut StaticOverlay<A>,
    index: usize,
    color: [f32; 3],
    alpha: f32,
) {
    if alpha <= 0.0 {
        return;
    }
    let alpha = alpha.min(1.0);
    let retained = 1.0 - alpha;
    for (channel, source) in overlay.premultiplied.iter_mut().zip(color) {
        channel[index] = source * alpha + channel[index] * retained;
    }
    overlay.alpha[index] = alpha + overlay.alpha[index] * retained;
}

fn new_static_overlay<A: Allocator + Clone>(
    parallax: Parallax,
    pixel_count: usize,
    allocator: A,
) -> Result<StaticOverlay<A>, GraphError> {
    let mut overlay = StaticOverlay {
        parallax,
        premultiplied: core::array::from_fn(|_| Vec::new_in(allocator.clone())),
        alpha: zero_f32_plane(pixel_count, allocator.clone())?,
        solid_color: None,
        mixed_color: false,
        painted: false,
    };
    for plane in &mut overlay.premultiplied {
        *plane = zero_f32_plane(pixel_count, allocator.clone())?;
    }
    Ok(overlay)
}

fn motion_derivative<A: Allocator>(
    plane: &[f32],
    width: usize,
    height: usize,
    horizontal: bool,
    allocator: A,
) -> Result<Vec<f32, A>, GraphError> {
    let mut derivative = zero_f32_plane(plane.len(), allocator)?;
    for row in 0..height {
        for column in 0..width {
            let index = row * width + column;
            let (before, after) = if horizontal {
                (
                    column
                        .checked_sub(1)
                        .map_or(0.0, |x| plane[row * width + x]),
                    if column + 1 < width {
                        plane[row * width + column + 1]
                    } else {
                        0.0
                    },
                )
            } else {
                (
                    row.checked_sub(1)
                        .map_or(0.0, |y| plane[y * width + column]),
                    if row + 1 < height {
                        plane[(row + 1) * width + column]
                    } else {
                        0.0
                    },
                )
            };
            // For a positive visible translation, sample from the preceding
            // source coordinate: f(x - shift) ~= f(x) + shift*(left-right)/2.
            derivative[index] = (before - after) * 0.5;
        }
    }
    Ok(derivative)
}

fn constant_plane<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    plane: &[f32],
    dimensions: &[u32; 2],
    allocator: A,
) -> Result<TensorId, GraphError> {
    Ok(builder.tensor(
        DType::F32,
        TensorFlags::CONSTANT,
        dimensions,
        &f32_plane_bytes(plane, allocator)?,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn emit_static_plane<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    plane: &[f32],
    offset: Option<[TensorId; 2]>,
    width: usize,
    height: usize,
    dimensions: &[u32; 2],
    allocator: A,
) -> Result<TensorId, GraphError> {
    let value = constant_plane(builder, plane, dimensions, allocator.clone())?;
    let Some(offset) = offset else {
        return Ok(value);
    };
    let derivative_x = motion_derivative(plane, width, height, true, allocator.clone())?;
    let derivative_y = motion_derivative(plane, width, height, false, allocator.clone())?;
    let derivative_x = constant_plane(builder, &derivative_x, dimensions, allocator.clone())?;
    let derivative_y = constant_plane(builder, &derivative_y, dimensions, allocator)?;
    Ok(builder.linearized_translate_2d_f32(
        LinearizedField2d {
            value,
            derivative_x,
            derivative_y,
        },
        offset,
        dimensions,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn emit_static_overlay<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    overlay: StaticOverlay<A>,
    pointer: Option<[TensorId; 2]>,
    width: usize,
    height: usize,
    scalar_dimensions: &[u32; 2],
    dimensions: &[u32; 2],
    allocator: A,
) -> Result<EmittedStaticOverlay, GraphError> {
    let offset = if overlay.parallax.is_zero() {
        None
    } else {
        let pointer = pointer.expect("parallax layers append pointer inputs");
        let scale_x = scalar(builder, overlay.parallax.x_pixels(), scalar_dimensions)?;
        let scale_y = scalar(builder, overlay.parallax.y_pixels(), scalar_dimensions)?;
        Some([
            binary(
                builder,
                OpCode::MUL,
                pointer[0],
                scale_x,
                DType::F32,
                scalar_dimensions,
            )?,
            binary(
                builder,
                OpCode::MUL,
                pointer[1],
                scale_y,
                DType::F32,
                scalar_dimensions,
            )?,
        ])
    };
    let mut alpha = emit_static_plane(
        builder,
        &overlay.alpha,
        offset,
        width,
        height,
        dimensions,
        allocator.clone(),
    )?;
    if offset.is_some() {
        // Linearization is intentionally limited to shallow displacement.
        // Clamp the small edge extrapolation and restore the premultiplied
        // invariant before source-over composition.
        let zero = scalar(builder, 0.0, scalar_dimensions)?;
        let one = scalar(builder, 1.0, scalar_dimensions)?;
        alpha = binary(
            builder,
            OpCode::MAXIMUM,
            alpha,
            zero,
            DType::F32,
            dimensions,
        )?;
        alpha = binary(builder, OpCode::MINIMUM, alpha, one, DType::F32, dimensions)?;
    }
    if !overlay.mixed_color
        && let Some(color) = overlay.solid_color
    {
        return Ok(EmittedStaticOverlay::Solid {
            alpha,
            colour: color_channels(color),
        });
    }
    let mut rgb = [TensorId(0); 3];
    for (output, plane) in rgb.iter_mut().zip(&overlay.premultiplied) {
        *output = emit_static_plane(
            builder,
            plane,
            offset,
            width,
            height,
            dimensions,
            allocator.clone(),
        )?;
    }
    if offset.is_some() {
        let zero = scalar(builder, 0.0, scalar_dimensions)?;
        for channel in &mut rgb {
            *channel = binary(
                builder,
                OpCode::MAXIMUM,
                *channel,
                zero,
                DType::F32,
                dimensions,
            )?;
            *channel = binary(
                builder,
                OpCode::MINIMUM,
                *channel,
                alpha,
                DType::F32,
                dimensions,
            )?;
        }
    }
    Ok(EmittedStaticOverlay::Premultiplied(PremultipliedRgba {
        rgb,
        alpha,
    }))
}

fn prepare_static_overlays<A: Allocator + Clone, const N: usize>(
    scene: &Scene<N>,
    pixel_count: usize,
    width: usize,
    height: usize,
    atlas: Option<AlphaAtlas<'_>>,
    glyphs: &[PlacedGlyph],
    allocator: A,
) -> Result<Vec<StaticOverlay<A>, A>, GraphError> {
    let mut overlays = Vec::new_in(allocator.clone());
    for &index in scene.order() {
        let node = *scene.node(index);
        if node.opacity == 0 || node.bind == Bind::Composite {
            continue;
        }
        let color = match node.bind {
            Bind::Fill { color }
            | Bind::RoundedFill { color, .. }
            | Bind::TextRun { color, .. } => color,
            Bind::Composite => continue,
            // Border and shadow kernels arrive with the chrome PR.
            Bind::Border { .. }
            | Bind::Shadow { .. }
            | Bind::Blit { .. }
            | Bind::Sample { .. }
            | Bind::Surface { .. } => {
                return Err(GraphError::UnsupportedBind { node: index });
            }
        };
        if overlays
            .last()
            .is_none_or(|overlay: &StaticOverlay<A>| overlay.parallax != node.parallax)
        {
            overlays.push(new_static_overlay(
                node.parallax,
                pixel_count,
                allocator.clone(),
            )?);
        }
        let overlay = overlays.last_mut().expect("a paint has a static overlay");
        match overlay.solid_color {
            None if !overlay.mixed_color => overlay.solid_color = Some(color),
            Some(previous) if previous == color => {}
            Some(_) | None => {
                overlay.solid_color = None;
                overlay.mixed_color = true;
            }
        }
        let color = color_channels(color);
        match node.bind {
            Bind::Fill { .. } | Bind::RoundedFill { .. } => {
                let rect = scene.rect(index);
                let Some((x, y, w, h)) = clipped(rect, width, height) else {
                    continue;
                };
                let opacity = f32::from(node.opacity) / 255.0;
                for row in y..y + h {
                    for column in x..x + w {
                        let alpha = host_fill_coverage(
                            node.bind,
                            rect,
                            column as f32 + 0.5,
                            row as f32 + 0.5,
                        ) * opacity;
                        blend_static_pixel(overlay, row * width + column, color, alpha);
                    }
                }
            }
            Bind::TextRun {
                glyph_start,
                glyph_len,
                ..
            } => {
                let mut coverage = zero_u16_plane(pixel_count, allocator.clone())?;
                composite_text_coverage(
                    &mut coverage,
                    pixel_count,
                    width,
                    height,
                    scene.rect(index),
                    glyph_start,
                    glyph_len,
                    atlas.ok_or(GraphError::MissingAtlas)?,
                    glyphs,
                    node.opacity,
                )?;
                for (pixel, coverage) in coverage.into_iter().enumerate() {
                    blend_static_pixel(overlay, pixel, color, f32::from(coverage) / 65_535.0);
                }
            }
            Bind::Composite
            | Bind::Border { .. }
            | Bind::Shadow { .. }
            | Bind::Blit { .. }
            | Bind::Sample { .. }
            | Bind::Surface { .. } => {
                unreachable!("non-paint and unsupported binds were handled above")
            }
        }
        overlay.painted = true;
    }
    overlays.retain(|overlay| overlay.painted);
    Ok(overlays)
}

fn host_fill_coverage(bind: Bind, rect: Rect, x: f32, y: f32) -> f32 {
    let Bind::RoundedFill { radii, .. } = bind else {
        return if x >= rect.x as f32
            && x <= rect.x as f32 + f32::from(rect.w)
            && y >= rect.y as f32
            && y <= rect.y as f32 + f32::from(rect.h)
        {
            1.0
        } else {
            0.0
        };
    };
    let radius = f32::from(radii.max().min(rect.w / 2).min(rect.h / 2));
    if radius == 0.0 {
        return 1.0;
    }
    let half_width = f32::from(rect.w) * 0.5;
    let half_height = f32::from(rect.h) * 0.5;
    let qx = (x - (rect.x as f32 + half_width)).abs() - (half_width - radius);
    let qy = (y - (rect.y as f32 + half_height)).abs() - (half_height - radius);
    let outside_x = qx.max(0.0);
    let outside_y = qy.max(0.0);
    let outside_distance = (outside_x * outside_x + outside_y * outside_y).sqrt();
    let signed_distance = outside_distance + qx.max(qy).min(0.0) - radius;
    (0.5 - signed_distance).clamp(0.0, 1.0)
}

fn field_input<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    builder.tensor(DType::F32, TensorFlags::INPUT, dimensions, &[])
}

fn scalar<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    value: f32,
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    builder.tensor(
        DType::F32,
        TensorFlags::CONSTANT,
        dimensions,
        &value.to_le_bytes(),
    )
}

fn binary<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    op: OpCode,
    left: TensorId,
    right: TensorId,
    dtype: DType,
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    let output = builder.tensor(dtype, TensorFlags::default(), dimensions, &[])?;
    builder.node(op, NodeFlags::default(), &[left, right], &[output], &[])?;
    Ok(output)
}

fn unary<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    op: OpCode,
    input: TensorId,
    dtype: DType,
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    let output = builder.tensor(dtype, TensorFlags::default(), dimensions, &[])?;
    builder.node(op, NodeFlags::default(), &[input], &[output], &[])?;
    Ok(output)
}

fn field_plus_scalar<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    field: TensorId,
    value: TensorId,
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    binary(builder, OpCode::ADD, field, value, DType::F32, dimensions)
}

fn coordinate_plane<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    extent: u32,
    _horizontal: bool,
    dimensions: &[u32; 2],
    allocator: A,
) -> Result<TensorId, AllocationError> {
    let mut bytes = Vec::new_in(allocator);
    bytes
        .try_reserve_exact(extent as usize * 4)
        .map_err(|_| AllocationError::AllocationFailed)?;
    for coordinate in 0..extent {
        bytes.extend_from_slice(&(coordinate as f32 + 0.5).to_le_bytes());
    }
    builder.tensor(DType::F32, TensorFlags::CONSTANT, dimensions, &bytes)
}

fn solid_channels<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    domain: TensorId,
    color: u32,
    scalar_dimensions: &[u32; 2],
    dimensions: &[u32; 2],
) -> Result<[TensorId; 3], AllocationError> {
    let rgb = color_channels(color);
    let red = scalar(builder, rgb[0], scalar_dimensions)?;
    let green = scalar(builder, rgb[1], scalar_dimensions)?;
    let blue = scalar(builder, rgb[2], scalar_dimensions)?;
    Ok([
        field_plus_scalar(builder, domain, red, dimensions)?,
        field_plus_scalar(builder, domain, green, dimensions)?,
        field_plus_scalar(builder, domain, blue, dimensions)?,
    ])
}

#[allow(clippy::too_many_arguments)]
fn apply_radial_light<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    channels: &mut [TensorId; 3],
    x_coordinates: TensorId,
    y_coordinates: TensorId,
    center_x: TensorId,
    center_y: TensorId,
    strength: TensorId,
    light: GraphRadialStyle,
    x_dimensions: &[u32; 2],
    y_dimensions: &[u32; 2],
    scalar_dimensions: &[u32; 2],
    dimensions: &[u32; 2],
) -> Result<(), GraphError> {
    let dx = binary(
        builder,
        OpCode::SUB,
        x_coordinates,
        center_x,
        DType::F32,
        x_dimensions,
    )?;
    let dy = binary(
        builder,
        OpCode::SUB,
        y_coordinates,
        center_y,
        DType::F32,
        y_dimensions,
    )?;
    // Normalize before squaring. The mathematical field is unchanged, while
    // every intermediate stays close to the unit domain instead of growing to
    // physical-coordinate squares. That matters for providers whose F32 ABI
    // can execute parts of an elementwise graph on narrower internal lanes.
    let inverse_radius = scalar(builder, 1.0 / light.radius, scalar_dimensions)?;
    let normalized_dx = binary(
        builder,
        OpCode::MUL,
        dx,
        inverse_radius,
        DType::F32,
        x_dimensions,
    )?;
    let normalized_dy = binary(
        builder,
        OpCode::MUL,
        dy,
        inverse_radius,
        DType::F32,
        y_dimensions,
    )?;
    let dx2 = binary(
        builder,
        OpCode::MUL,
        normalized_dx,
        normalized_dx,
        DType::F32,
        x_dimensions,
    )?;
    let dy2 = binary(
        builder,
        OpCode::MUL,
        normalized_dy,
        normalized_dy,
        DType::F32,
        y_dimensions,
    )?;
    let normalized = binary(builder, OpCode::ADD, dx2, dy2, DType::F32, dimensions)?;
    let one = scalar(builder, 1.0, scalar_dimensions)?;
    let weight = match light.profile {
        GraphRadialProfile::Rational { sharpness } => {
            let profile = radial_power(
                builder,
                normalized,
                sharpness,
                scalar_dimensions,
                dimensions,
            )?;
            let denominator = binary(builder, OpCode::ADD, one, profile, DType::F32, dimensions)?;
            unary(
                builder,
                OpCode::RECIPROCAL,
                denominator,
                DType::F32,
                dimensions,
            )?
        }
        GraphRadialProfile::Disc { softness } => {
            // Evaluate a normalized signed distance without introducing a
            // renderer opcode: sqrt(n²) = n² * rsqrt(max(n², epsilon)), then
            // restore physical units once. The guarded center remains exactly
            // zero and the clamp retains a one-physical-pixel transition.
            let epsilon = scalar(builder, f32::EPSILON, scalar_dimensions)?;
            let safe_normalized = binary(
                builder,
                OpCode::MAXIMUM,
                normalized,
                epsilon,
                DType::F32,
                dimensions,
            )?;
            let inverse_normalized_distance = unary(
                builder,
                OpCode::RSQRT,
                safe_normalized,
                DType::F32,
                dimensions,
            )?;
            let normalized_distance = binary(
                builder,
                OpCode::MUL,
                normalized,
                inverse_normalized_distance,
                DType::F32,
                dimensions,
            )?;
            let signed_normalized_distance = binary(
                builder,
                OpCode::SUB,
                normalized_distance,
                one,
                DType::F32,
                dimensions,
            )?;
            let radius = scalar(builder, light.radius, scalar_dimensions)?;
            let signed_distance = binary(
                builder,
                OpCode::MUL,
                signed_normalized_distance,
                radius,
                DType::F32,
                dimensions,
            )?;
            let inverse_softness = scalar(builder, 1.0 / softness, scalar_dimensions)?;
            let edge_distance = binary(
                builder,
                OpCode::MUL,
                signed_distance,
                inverse_softness,
                DType::F32,
                dimensions,
            )?;
            let half = scalar(builder, 0.5, scalar_dimensions)?;
            let coverage = binary(
                builder,
                OpCode::SUB,
                half,
                edge_distance,
                DType::F32,
                dimensions,
            )?;
            let zero = scalar(builder, 0.0, scalar_dimensions)?;
            let coverage = binary(
                builder,
                OpCode::MAXIMUM,
                coverage,
                zero,
                DType::F32,
                dimensions,
            )?;
            binary(
                builder,
                OpCode::MINIMUM,
                coverage,
                one,
                DType::F32,
                dimensions,
            )?
        }
    };
    let mut weight = binary(
        builder,
        OpCode::MUL,
        weight,
        strength,
        DType::F32,
        dimensions,
    )?;
    if let Some(rim) = light.rim {
        // A wormhole throat and a lensing critical curve both single out a
        // finite areal radius. Model that restrained annular response around
        // normalized radius one, then source-over union it with the parent
        // field so the total weight stays bounded.
        let rim_distance = binary(
            builder,
            OpCode::SUB,
            normalized,
            one,
            DType::F32,
            dimensions,
        )?;
        let rim_distance_squared = binary(
            builder,
            OpCode::MUL,
            rim_distance,
            rim_distance,
            DType::F32,
            dimensions,
        )?;
        let rim_sharpness = scalar(builder, rim.sharpness, scalar_dimensions)?;
        let rim_profile = binary(
            builder,
            OpCode::MUL,
            rim_distance_squared,
            rim_sharpness,
            DType::F32,
            dimensions,
        )?;
        let rim_denominator = binary(
            builder,
            OpCode::ADD,
            one,
            rim_profile,
            DType::F32,
            dimensions,
        )?;
        let rim_weight = unary(
            builder,
            OpCode::RECIPROCAL,
            rim_denominator,
            DType::F32,
            dimensions,
        )?;
        let rim_weight = binary(
            builder,
            OpCode::MUL,
            rim_weight,
            strength,
            DType::F32,
            dimensions,
        )?;
        let rim_strength = scalar(builder, rim.strength, scalar_dimensions)?;
        let rim_weight = binary(
            builder,
            OpCode::MUL,
            rim_weight,
            rim_strength,
            DType::F32,
            // Apply the two scalar factors in an order that keeps both
            // multiplies on the field. TOSA's broadcast result shape is
            // inferred from its inputs, so a scalar × scalar operation cannot
            // legally claim a full-plane output merely to influence cost.
            dimensions,
        )?;
        let uncovered = binary(builder, OpCode::SUB, one, weight, DType::F32, dimensions)?;
        let rim_contribution = binary(
            builder,
            OpCode::MUL,
            rim_weight,
            uncovered,
            DType::F32,
            dimensions,
        )?;
        weight = binary(
            builder,
            OpCode::ADD,
            weight,
            rim_contribution,
            DType::F32,
            dimensions,
        )?;
    }
    for (channel, light_channel) in channels.iter_mut().zip(color_channels(light.color)) {
        let target = scalar(builder, light_channel, scalar_dimensions)?;
        let delta = binary(
            builder,
            OpCode::SUB,
            target,
            *channel,
            DType::F32,
            dimensions,
        )?;
        let contribution = binary(builder, OpCode::MUL, delta, weight, DType::F32, dimensions)?;
        *channel = binary(
            builder,
            OpCode::ADD,
            *channel,
            contribution,
            DType::F32,
            dimensions,
        )?;
    }
    Ok(())
}

fn radial_power<A: Allocator + Clone>(
    builder: &mut GraphBuilder<A>,
    normalized: TensorId,
    sharpness: f32,
    scalar_dimensions: &[u32; 2],
    dimensions: &[u32; 2],
) -> Result<TensorId, AllocationError> {
    let integer = sharpness as u8;
    if sharpness == f32::from(integer) && (1..=8).contains(&integer) {
        let mut power = normalized;
        for _ in 1..integer {
            power = binary(
                builder,
                OpCode::MUL,
                power,
                normalized,
                DType::F32,
                dimensions,
            )?;
        }
        Ok(power)
    } else {
        let exponent = scalar(builder, sharpness, scalar_dimensions)?;
        binary(
            builder,
            OpCode::POW,
            normalized,
            exponent,
            DType::F32,
            dimensions,
        )
    }
}

fn validate_radial_light(light: GraphRadialLight) -> Result<(), GraphError> {
    if !light.x.is_finite()
        || !light.y.is_finite()
        || !light.strength.is_finite()
        || !(0.0..=1.0).contains(&light.strength)
    {
        return Err(GraphError::InvalidBackground);
    }
    validate_radial_style(GraphRadialStyle {
        radius: light.radius,
        color: light.color,
        profile: light.profile,
        rim: light.rim,
    })
}

fn validate_radial_style(light: GraphRadialStyle) -> Result<(), GraphError> {
    if !light.radius.is_finite() || light.radius <= 0.0 {
        return Err(GraphError::InvalidBackground);
    }
    match light.profile {
        GraphRadialProfile::Rational { sharpness }
            if sharpness.is_finite() && (1.0..=8.0).contains(&sharpness) => {}
        GraphRadialProfile::Disc { softness }
            if softness.is_finite() && softness >= 0.25 && softness <= light.radius * 2.0 => {}
        _ => return Err(GraphError::InvalidBackground),
    }
    if let Some(rim) = light.rim
        && (!rim.strength.is_finite()
            || !(0.0..=1.0).contains(&rim.strength)
            || !rim.sharpness.is_finite()
            || !(1.0..=256.0).contains(&rim.sharpness))
    {
        return Err(GraphError::InvalidBackground);
    }
    Ok(())
}

fn color_channels(color: u32) -> [f32; 3] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
    ]
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn text_coverage(
    pixel_count: usize,
    width: usize,
    height: usize,
    node_rect: Rect,
    glyph_start: u16,
    glyph_len: u16,
    atlas: AlphaAtlas<'_>,
    glyphs: &[PlacedGlyph],
) -> Result<Vec<u16>, GraphError> {
    let mut coverage = zero_u16_plane(pixel_count, alloc::alloc::Global)?;
    composite_text_coverage(
        &mut coverage,
        pixel_count,
        width,
        height,
        node_rect,
        glyph_start,
        glyph_len,
        atlas,
        glyphs,
        u8::MAX,
    )?;
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
fn composite_text_coverage(
    coverage: &mut [u16],
    pixel_count: usize,
    width: usize,
    height: usize,
    node_rect: Rect,
    glyph_start: u16,
    glyph_len: u16,
    atlas: AlphaAtlas<'_>,
    glyphs: &[PlacedGlyph],
    opacity: u8,
) -> Result<(), GraphError> {
    if coverage.len() != pixel_count {
        return Err(GraphError::PixelCountOverflow);
    }
    let start = glyph_start as usize;
    let end = start
        .checked_add(glyph_len as usize)
        .ok_or(GraphError::InvalidGlyphRange)?;
    let run = glyphs
        .get(start..end)
        .ok_or(GraphError::InvalidGlyphRange)?;
    for glyph in run {
        let atlas_x = glyph.ax as usize;
        let atlas_y = glyph.ay as usize;
        let glyph_w = glyph.aw as usize;
        let glyph_h = glyph.ah as usize;
        if atlas_x
            .checked_add(glyph_w)
            .is_none_or(|end| end > atlas.width)
            || atlas_y
                .checked_add(glyph_h)
                .is_none_or(|end| end > atlas.height)
        {
            return Err(GraphError::AtlasBounds);
        }
        if glyph_w == 0 || glyph_h == 0 {
            continue;
        }

        let screen_x = node_rect.x + i32::from(glyph.dx);
        let screen_y = node_rect.y + i32::from(glyph.dy);
        let painted_width = glyph
            .aw
            .checked_add(u16::from(glyph.subpixel_x != 0))
            .ok_or(GraphError::AtlasBounds)?;
        let painted_height = glyph
            .ah
            .checked_add(u16::from(glyph.subpixel_y != 0))
            .ok_or(GraphError::AtlasBounds)?;
        let rect = Rect {
            x: screen_x,
            y: screen_y,
            w: painted_width,
            h: painted_height,
        };
        let Some((x, y, w, h)) = clipped(rect, width, height) else {
            continue;
        };
        let source_x = (x as i32 - screen_x) as usize;
        let source_y = (y as i32 - screen_y) as usize;
        for row in 0..h {
            for column in 0..w {
                let source = translated_glyph_coverage(
                    atlas,
                    atlas_x,
                    atlas_y,
                    glyph_w,
                    glyph_h,
                    (source_x + column) as i32,
                    (source_y + row) as i32,
                    glyph.subpixel_x,
                    glyph.subpixel_y,
                );
                let source = ((u32::from(source) * u32::from(opacity) + 127) / 255) as u16;
                let destination = &mut coverage[(y + row) * width + x + column];
                *destination = source_over_alpha(source, *destination);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn translated_glyph_coverage(
    atlas: AlphaAtlas<'_>,
    atlas_x: usize,
    atlas_y: usize,
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    phase_x: u8,
    phase_y: u8,
) -> u16 {
    let sample = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            0
        } else {
            atlas.coverage((atlas_y + y as usize) * atlas.width + atlas_x + x as usize)
        }
    };
    let fx = u64::from(phase_x);
    let fy = u64::from(phase_y);
    let inverse_x = 256 - fx;
    let inverse_y = 256 - fy;
    let weighted = u64::from(sample(x, y)) * inverse_x * inverse_y
        + u64::from(sample(x - 1, y)) * fx * inverse_y
        + u64::from(sample(x, y - 1)) * inverse_x * fy
        + u64::from(sample(x - 1, y - 1)) * fx * fy;
    ((weighted + 32_768) >> 16) as u16
}

fn zero_u16_plane<A: Allocator>(
    pixel_count: usize,
    allocator: A,
) -> Result<Vec<u16, A>, GraphError> {
    let mut output = Vec::new_in(allocator);
    output
        .try_reserve_exact(pixel_count)
        .map_err(|_| GraphError::HostAllocation)?;
    output.resize(pixel_count, 0);
    Ok(output)
}

fn clipped(rect: Rect, width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
    let x0 = rect.x.max(0) as usize;
    let y0 = rect.y.max(0) as usize;
    let x1 = (rect.x + i32::from(rect.w)).min(width as i32).max(0) as usize;
    let y1 = (rect.y + i32::from(rect.h)).min(height as i32).max(0) as usize;
    if x1 > x0 && y1 > y0 {
        Some((x0, y0, x1 - x0, y1 - y0))
    } else {
        None
    }
}

#[inline]
fn source_over_alpha(source: u16, destination: u16) -> u16 {
    let source = u32::from(source);
    let destination = u32::from(destination);
    (source + (destination * (65_535 - source) + 32_767) / 65_535) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Anchor, LayoutParams, Offset};
    use alloc::alloc::Global;

    #[test]
    fn clipping_rejects_fully_offscreen_rectangles_without_underflow() {
        let viewport = (10, 10);

        assert_eq!(
            clipped(
                Rect {
                    x: 12,
                    y: 2,
                    w: 4,
                    h: 4,
                },
                viewport.0,
                viewport.1,
            ),
            None
        );
        assert_eq!(
            clipped(
                Rect {
                    x: 2,
                    y: 12,
                    w: 4,
                    h: 4,
                },
                viewport.0,
                viewport.1,
            ),
            None
        );
    }

    #[test]
    fn scene_graph_uses_only_portable_elementwise_ops() {
        let mut scene = Scene::<4>::new();
        let root = scene
            .add_root(
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset::ZERO,
                    w: 4,
                    h: 2,
                },
                Bind::Composite,
            )
            .unwrap();
        scene
            .add_child_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x: 1, y: 0 },
                    w: 2,
                    h: 2,
                },
                Bind::Fill { color: 0x00ff_8040 },
                128,
            )
            .unwrap();

        let frame = compile_scene_graph_in(
            &mut scene,
            Rect {
                x: 0,
                y: 0,
                w: 4,
                h: 2,
            },
            GraphBackground::Solid(0x0010_2030),
            None,
            &[],
            Global,
        )
        .unwrap();
        let graph = frame.graph();

        assert_eq!(frame.background_input_count(), 0);
        assert_eq!(frame.output_format(), GraphOutputFormat::PlanarF16);
        assert!(graph.node_count() > 0);
        for index in 0..graph.node_count() {
            let op = graph.node(index).unwrap().op;
            assert!(matches!(
                op,
                OpCode::CAST
                    | OpCode::ADD
                    | OpCode::SUB
                    | OpCode::MUL
                    | OpCode::SELECT
                    | OpCode::GREATER_EQUAL
                    | OpCode::LOGICAL_AND
            ));
        }

        let mut outputs = 0;
        for index in 0..graph.tensor_count() {
            let tensor = graph.tensor(TensorId(index as u32)).unwrap();
            if tensor.flags.contains(TensorFlags::OUTPUT) {
                outputs += 1;
                assert_eq!(tensor.dtype, DType::F16);
            } else {
                assert!(matches!(tensor.dtype, DType::F32 | DType::U8));
            }
        }
        assert_eq!(outputs, 3);
    }

    #[test]
    fn alpha_union_is_source_over() {
        assert_eq!(source_over_alpha(32_768, 0), 32_768);
        assert_eq!(source_over_alpha(65_535, 32_768), 65_535);
        assert_eq!(source_over_alpha(32_768, 32_768), 49_152);
    }

    #[test]
    fn adjacent_same_ink_text_runs_share_one_exact_overlay() {
        let mut scene = Scene::<4>::new();
        let layout = LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset::ZERO,
            w: 4,
            h: 2,
        };
        let root = scene.add_root(layout, Bind::Composite).unwrap();
        scene
            .add_child(
                root,
                layout,
                Bind::TextRun {
                    glyph_start: 0,
                    glyph_len: 1,
                    color: 0x0045_2416,
                },
            )
            .unwrap();
        scene
            .add_child_with_opacity(
                root,
                layout,
                Bind::TextRun {
                    glyph_start: 1,
                    glyph_len: 1,
                    color: 0x0045_2416,
                },
                128,
            )
            .unwrap();
        let glyphs = [
            PlacedGlyph {
                ax: 0,
                ay: 0,
                aw: 1,
                ah: 1,
                dx: 0,
                dy: 0,
                subpixel_x: 0,
                subpixel_y: 0,
            },
            PlacedGlyph {
                ax: 0,
                ay: 0,
                aw: 1,
                ah: 1,
                dx: 1,
                dy: 0,
                subpixel_x: 0,
                subpixel_y: 0,
            },
        ];
        let atlas = [u16::MAX];
        let frame = compile_scene_graph_in(
            &mut scene,
            Rect {
                x: 0,
                y: 0,
                w: 4,
                h: 2,
            },
            GraphBackground::Solid(0x00ff_ffff),
            Some(AlphaAtlas::from_u16(&atlas, 1, 1)),
            &glyphs,
            Global,
        )
        .unwrap();
        let graph = frame.graph();
        let fields = (0..graph.tensor_count())
            .filter_map(|index| {
                let tensor = graph.tensor(TensorId(index as u32)).unwrap();
                (tensor.dtype == DType::F32
                    && tensor.flags.contains(TensorFlags::CONSTANT)
                    && tensor.dimensions().eq([2, 4]))
                .then(|| tensor.constant())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            fields.len(),
            1,
            "same-ink paints need only one exact alpha plane"
        );
        let coverage = fields[0]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(coverage[0], 1.0);
        assert_eq!(coverage[1], 32_896.0 / 65_535.0);
        assert!(coverage[2..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn fractional_glyph_phase_distributes_coverage_across_pixels() {
        let atlas = [u16::MAX];
        let glyphs = [PlacedGlyph {
            ax: 0,
            ay: 0,
            aw: 1,
            ah: 1,
            dx: 0,
            dy: 0,
            subpixel_x: 128,
            subpixel_y: 0,
        }];
        let coverage = text_coverage(
            2,
            2,
            1,
            Rect {
                x: 0,
                y: 0,
                w: 2,
                h: 1,
            },
            0,
            1,
            AlphaAtlas::from_u16(&atlas, 1, 1),
            &glyphs,
        )
        .unwrap();
        assert_eq!(coverage, [32_768, 32_768]);
    }
}
