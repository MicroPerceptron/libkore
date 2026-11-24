//! Lowering from the backend-neutral `axui-core` display list to `.ax`.
//!
//! This module owns the logical-to-physical geometry boundary. Frontends keep
//! layout in logical `f32` coordinates, while the resulting [`AxFrameGraph`]
//! contains physical-pixel coverage and ordinary `.ax` composition nodes.
//! Provider selection remains below this layer.

use core::alloc::Allocator;

use axui_core::{Color, Command, FrameView, Paint, Quad, Rect as UiRect};

use crate::{
    Anchor, Bind, Corners, LayoutParams, Offset, Rect, Scene,
    graph::{
        AxFrameGraph, GraphBackground, GraphError, GraphOutputFormat,
        compile_scene_graph_with_output_in,
    },
};

/// Failure while translating an axui display list into a portable graph.
#[derive(Debug)]
pub enum AxuiGraphError {
    /// The display-list frame or primitive geometry is malformed.
    InvalidGeometry,
    /// Physical coordinates exceed axrd's current scene ABI.
    CoordinateOverflow,
    /// More visible quads were supplied than the selected scene capacity.
    SceneCapacity,
    /// Clips and transforms are retained by axui but not lowered in this slice.
    UnsupportedCommand {
        /// Command-stream index of the unsupported operation.
        index: usize,
    },
    /// Only solid paint is supported by this first lowering slice.
    UnsupportedPaint {
        /// Quad-arena index containing the paint.
        quad: usize,
    },
    /// A visible border needs its own coverage pass and is not yet lowered.
    UnsupportedBorder {
        /// Quad-arena index containing the border.
        quad: usize,
    },
    /// A visible shadow needs its own coverage pass and is not yet lowered.
    UnsupportedShadow {
        /// Quad-arena index containing the shadow.
        quad: usize,
    },
    /// The current axrd scene shape has one uniform corner radius.
    NonUniformCornerRadii {
        /// Quad-arena index containing the radii.
        quad: usize,
    },
    /// Unsnapped subpixel edges need a fractional rectangle coverage field.
    UnsnappedQuad {
        /// Quad-arena index requiring that field.
        quad: usize,
    },
    /// The translated scene could not be compiled into `.ax`.
    Graph(GraphError),
}

impl From<GraphError> for AxuiGraphError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

/// Lower one axui frame using axrd's default F16 display-plane boundary.
pub fn compile_frame_graph_in<A: Allocator + Clone, const N: usize>(
    frame: FrameView<'_, A>,
    background: GraphBackground,
    allocator: A,
) -> Result<AxFrameGraph<A>, AxuiGraphError> {
    compile_frame_graph_with_output_in::<A, N>(
        frame,
        background,
        GraphOutputFormat::PlanarF16,
        allocator,
    )
}

/// Lower one allocator-typed axui frame into an allocator-typed `.ax` graph.
///
/// This first executable slice supports snapped solid quads, alpha, and a
/// uniform rounded fill. The error variants deliberately distinguish retained
/// display-list features that still need lowering, so an `axgfx` backend can
/// reject them before provider admission instead of silently changing their
/// appearance.
pub fn compile_frame_graph_with_output_in<A: Allocator + Clone, const N: usize>(
    frame: FrameView<'_, A>,
    background: GraphBackground,
    output_format: GraphOutputFormat,
    allocator: A,
) -> Result<AxFrameGraph<A>, AxuiGraphError> {
    let bounds = frame.bounds();
    let scale = frame.scale_factor();
    if !valid_rect(bounds)
        || bounds.width <= 0.0
        || bounds.height <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(AxuiGraphError::InvalidGeometry);
    }

    let width = rounded_positive_u16(bounds.width * scale)?;
    let height = rounded_positive_u16(bounds.height * scale)?;
    let root_rect = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    let mut scene = Scene::<N>::new();

    for (command_index, command) in frame.commands().iter().copied().enumerate() {
        let Command::DrawQuad(quad_index) = command else {
            return Err(AxuiGraphError::UnsupportedCommand {
                index: command_index,
            });
        };
        let quad = frame
            .quad(quad_index)
            .ok_or(AxuiGraphError::InvalidGeometry)?;
        lower_quad(&mut scene, bounds, scale, quad_index, *quad)?;
    }

    compile_scene_graph_with_output_in(
        &mut scene,
        root_rect,
        background,
        None,
        &[],
        output_format,
        allocator,
    )
    .map_err(Into::into)
}

fn lower_quad<const N: usize>(
    scene: &mut Scene<N>,
    frame_bounds: UiRect,
    scale: f32,
    quad_index: usize,
    quad: Quad,
) -> Result<(), AxuiGraphError> {
    if !valid_rect(quad.bounds)
        || !valid_quad_metadata(quad)
        || !valid_color(match quad.paint {
            Paint::Solid(color) => color,
            Paint::LinearGradient(_) => {
                return Err(AxuiGraphError::UnsupportedPaint { quad: quad_index });
            }
        })
    {
        return Err(AxuiGraphError::InvalidGeometry);
    }
    if quad.border.width > 0.0 && quad.border.color.a > 0.0 {
        return Err(AxuiGraphError::UnsupportedBorder { quad: quad_index });
    }
    if quad.shadow.color.a > 0.0 {
        return Err(AxuiGraphError::UnsupportedShadow { quad: quad_index });
    }
    if !quad.snap {
        return Err(AxuiGraphError::UnsnappedQuad { quad: quad_index });
    }
    if quad.bounds.width == 0.0 || quad.bounds.height == 0.0 {
        return Ok(());
    }

    let Paint::Solid(color) = quad.paint else {
        unreachable!("gradient paint returned above")
    };
    let opacity = normalized_u8(color.a)?;
    if opacity == 0 {
        return Ok(());
    }

    let left = rounded_i32((quad.bounds.x - frame_bounds.x) * scale)?;
    let top = rounded_i32((quad.bounds.y - frame_bounds.y) * scale)?;
    let right = rounded_i32((quad.bounds.x + quad.bounds.width - frame_bounds.x) * scale)?;
    let bottom = rounded_i32((quad.bounds.y + quad.bounds.height - frame_bounds.y) * scale)?;
    let physical_width = right
        .checked_sub(left)
        .ok_or(AxuiGraphError::CoordinateOverflow)?;
    let physical_height = bottom
        .checked_sub(top)
        .ok_or(AxuiGraphError::CoordinateOverflow)?;
    if physical_width <= 0 || physical_height <= 0 {
        return Ok(());
    }
    let physical_width =
        u16::try_from(physical_width).map_err(|_| AxuiGraphError::CoordinateOverflow)?;
    let physical_height =
        u16::try_from(physical_height).map_err(|_| AxuiGraphError::CoordinateOverflow)?;

    let radii = quad.border.radii;
    let radius = radii.top_left;
    if [radii.top_right, radii.bottom_right, radii.bottom_left]
        .iter()
        .any(|other| (other - radius).abs() > 0.001)
    {
        return Err(AxuiGraphError::NonUniformCornerRadii { quad: quad_index });
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(AxuiGraphError::InvalidGeometry);
    }
    let physical_radius = if radius == 0.0 {
        0
    } else {
        rounded_nonnegative_u16(radius * scale)?
    };
    let color = (u32::from(normalized_u8(color.r)?) << 16)
        | (u32::from(normalized_u8(color.g)?) << 8)
        | u32::from(normalized_u8(color.b)?);
    let bind = if physical_radius == 0 {
        Bind::Fill { color }
    } else {
        Bind::RoundedFill {
            color,
            radii: Corners::uniform(physical_radius),
        }
    };
    scene
        .add_root_with_opacity(
            LayoutParams {
                anchor: Anchor::TOP_LEFT,
                offset: Offset { x: left, y: top },
                w: physical_width,
                h: physical_height,
            },
            bind,
            opacity,
        )
        .map(|_| ())
        .map_err(|_| AxuiGraphError::SceneCapacity)
}

fn valid_rect(rect: UiRect) -> bool {
    [rect.x, rect.y, rect.width, rect.height]
        .iter()
        .all(|value| value.is_finite())
        && rect.width >= 0.0
        && rect.height >= 0.0
}

fn valid_color(color: Color) -> bool {
    [color.r, color.g, color.b, color.a]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn valid_quad_metadata(quad: Quad) -> bool {
    valid_color(quad.border.color)
        && valid_color(quad.shadow.color)
        && quad.border.width.is_finite()
        && quad.border.width >= 0.0
        && quad.shadow.offset.x.is_finite()
        && quad.shadow.offset.y.is_finite()
        && quad.shadow.blur_radius.is_finite()
        && quad.shadow.blur_radius >= 0.0
        && [
            quad.border.radii.top_left,
            quad.border.radii.top_right,
            quad.border.radii.bottom_right,
            quad.border.radii.bottom_left,
        ]
        .iter()
        .all(|radius| radius.is_finite() && *radius >= 0.0)
}

fn normalized_u8(value: f32) -> Result<u8, AxuiGraphError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(AxuiGraphError::InvalidGeometry);
    }
    Ok((value * 255.0 + 0.5) as u8)
}

fn rounded_i32(value: f32) -> Result<i32, AxuiGraphError> {
    if !value.is_finite() || value <= i32::MIN as f32 || value >= i32::MAX as f32 {
        return Err(AxuiGraphError::CoordinateOverflow);
    }
    Ok(if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    })
}

fn rounded_positive_u16(value: f32) -> Result<u16, AxuiGraphError> {
    let value = rounded_nonnegative_u16(value)?;
    (value > 0)
        .then_some(value)
        .ok_or(AxuiGraphError::InvalidGeometry)
}

fn rounded_nonnegative_u16(value: f32) -> Result<u16, AxuiGraphError> {
    if !value.is_finite() || value < 0.0 {
        return Err(AxuiGraphError::CoordinateOverflow);
    }
    u16::try_from((value + 0.5) as u32).map_err(|_| AxuiGraphError::CoordinateOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax::{DType, TensorFlags};
    use axui_core::{Border, CornerRadii, Paint, Point, Recorder, Shadow};

    fn quad(bounds: UiRect, radius: f32) -> Quad {
        Quad {
            bounds,
            paint: Paint::Solid(Color::new(0.9, 0.3, 0.2, 0.75)),
            border: Border {
                color: Color::default(),
                width: 0.0,
                radii: CornerRadii {
                    top_left: radius,
                    top_right: radius,
                    bottom_right: radius,
                    bottom_left: radius,
                },
            },
            shadow: Shadow {
                color: Color::default(),
                offset: Point::default(),
                blur_radius: 0.0,
            },
            snap: true,
        }
    }

    #[test]
    fn logical_frame_scale_defines_physical_graph_extent() {
        let mut recorder = Recorder::new_in(
            UiRect::new(20.0, 30.0, 200.0, 100.0),
            1.5,
            std::alloc::System,
        );
        recorder.draw_quad(quad(UiRect::new(30.0, 40.0, 80.0, 32.0), 6.0));
        let frame = recorder.take_frame().unwrap();
        let compiled = compile_frame_graph_with_output_in::<_, 8>(
            frame.view(),
            GraphBackground::Solid(0x0010_1814),
            GraphOutputFormat::PlanarF32,
            std::alloc::System,
        )
        .unwrap();

        assert_eq!((compiled.width(), compiled.height()), (300, 150));
        assert_eq!(compiled.output_format(), GraphOutputFormat::PlanarF32);
        let graph = compiled.graph();
        let output_count = (0..graph.tensor_count())
            .filter(|index| {
                graph
                    .tensor(ax::TensorId(*index as u32))
                    .is_ok_and(|tensor| tensor.flags.contains(TensorFlags::OUTPUT))
            })
            .count();
        assert_eq!(output_count, 3);
        assert!((0..graph.tensor_count()).any(|index| {
            graph
                .tensor(ax::TensorId(index as u32))
                .is_ok_and(|tensor| tensor.dtype == DType::F32)
        }));
    }

    #[test]
    fn rejects_retained_features_instead_of_approximating_them() {
        let mut recorder =
            Recorder::new_in(UiRect::new(0.0, 0.0, 100.0, 80.0), 1.0, std::alloc::System);
        recorder.push_clip(UiRect::new(0.0, 0.0, 40.0, 40.0));
        recorder.draw_quad(quad(UiRect::new(2.0, 2.0, 20.0, 20.0), 2.0));
        recorder.pop_clip();
        let frame = recorder.take_frame().unwrap();

        assert!(matches!(
            compile_frame_graph_with_output_in::<_, 8>(
                frame.view(),
                GraphBackground::Solid(0),
                GraphOutputFormat::PlanarF32,
                std::alloc::System,
            ),
            Err(AxuiGraphError::UnsupportedCommand { index: 0 })
        ));
    }

    #[test]
    fn physical_extent_accepts_the_complete_scene_abi() {
        assert_eq!(rounded_positive_u16(u16::MAX as f32).unwrap(), u16::MAX);
    }
}
