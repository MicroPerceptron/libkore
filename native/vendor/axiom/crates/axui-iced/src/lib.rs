//! Iced frontend adapter for Axiom's backend-neutral display list.
//!
//! This crate implements Iced's base renderer contract without selecting an
//! Iced graphics backend. It records logical primitives into [`axui_core`],
//! ready for a separate `axrd`/`.ax` lowering stage.

#![feature(allocator_api)]

use axinput::{CursorIcon, CursorRequest};
use axui_core::{
    Border, Color, CornerRadii, Frame, GradientStop, IntegrityError, LinearGradient, Paint, Point,
    Quad, Recorder, Rect, Shadow, Transform,
};
use core::alloc::Allocator;
use iced_core::renderer::{self, Scale};
use iced_core::{Background, Rectangle, Transformation, gradient, image};

/// Resolves Iced's widget interaction into Axiom cursor policy.
///
/// Presentation remains a runtime decision: the request can be mapped to a
/// host cursor, an `axgfx::CursorPlane`, or axcomp's composited fallback.
pub const fn cursor_request(interaction: iced_core::mouse::Interaction) -> CursorRequest {
    use iced_core::mouse::Interaction;

    let (icon, visible) = match interaction {
        Interaction::None | Interaction::Idle => (CursorIcon::Default, true),
        Interaction::Hidden => (CursorIcon::Default, false),
        Interaction::ContextMenu => (CursorIcon::ContextMenu, true),
        Interaction::Help => (CursorIcon::Help, true),
        Interaction::Pointer => (CursorIcon::Pointer, true),
        Interaction::Progress => (CursorIcon::Progress, true),
        Interaction::Wait => (CursorIcon::Wait, true),
        Interaction::Cell => (CursorIcon::Cell, true),
        Interaction::Crosshair => (CursorIcon::Crosshair, true),
        Interaction::Text => (CursorIcon::Text, true),
        Interaction::Alias => (CursorIcon::Alias, true),
        Interaction::Copy => (CursorIcon::Copy, true),
        Interaction::Move => (CursorIcon::Move, true),
        Interaction::NoDrop => (CursorIcon::NoDrop, true),
        Interaction::NotAllowed => (CursorIcon::NotAllowed, true),
        Interaction::Grab => (CursorIcon::Grab, true),
        Interaction::Grabbing => (CursorIcon::Grabbing, true),
        Interaction::ResizingHorizontally => (CursorIcon::ResizeHorizontal, true),
        Interaction::ResizingVertically => (CursorIcon::ResizeVertical, true),
        Interaction::ResizingDiagonallyUp => (CursorIcon::ResizeDiagonalUp, true),
        Interaction::ResizingDiagonallyDown => (CursorIcon::ResizeDiagonalDown, true),
        Interaction::ResizingColumn => (CursorIcon::ResizeColumn, true),
        Interaction::ResizingRow => (CursorIcon::ResizeRow, true),
        Interaction::AllScroll => (CursorIcon::AllScroll, true),
        Interaction::ZoomIn => (CursorIcon::ZoomIn, true),
        Interaction::ZoomOut => (CursorIcon::ZoomOut, true),
    };
    CursorRequest {
        icon,
        visible,
        grab: axinput::CursorGrab::None,
    }
}

/// An Iced renderer that records an allocator-owned Axiom display list.
#[derive(Debug)]
pub struct Renderer<A: Allocator> {
    recorder: Recorder<A>,
    settings: renderer::Settings,
    scale: Option<Scale>,
}

impl<A: Allocator + Clone> Renderer<A> {
    /// Creates a renderer whose display data uses `allocator`.
    pub fn new_in(settings: renderer::Settings, allocator: A) -> Self {
        Self {
            recorder: Recorder::new_in(Rect::default(), 1.0, allocator),
            settings,
            scale: None,
        }
    }

    /// Returns the underlying Axiom recorder.
    pub const fn recorder(&self) -> &Recorder<A> {
        &self.recorder
    }

    /// Returns mutable access for Axiom-specific widget extensions.
    pub fn recorder_mut(&mut self) -> &mut Recorder<A> {
        &mut self.recorder
    }

    /// Takes the completed display list and starts another empty frame.
    pub fn take_frame(&mut self) -> Result<Frame<A>, IntegrityError> {
        self.recorder.take_frame()
    }
}

impl<A: Allocator + Clone> iced_core::Renderer for Renderer<A> {
    fn start_layer(&mut self, bounds: Rectangle) {
        self.recorder.push_clip(rect(bounds));
    }

    fn end_layer(&mut self) {
        self.recorder.pop_clip();
    }

    fn start_transformation(&mut self, transformation: Transformation) {
        self.recorder
            .push_transform(Transform(transformation.into()));
    }

    fn end_transformation(&mut self) {
        self.recorder.pop_transform();
    }

    fn fill_quad(&mut self, quad: renderer::Quad, background: impl Into<Background>) {
        self.recorder.draw_quad(Quad {
            bounds: rect(quad.bounds),
            paint: paint(background.into()),
            border: Border {
                color: color(quad.border.color),
                width: quad.border.width,
                radii: CornerRadii {
                    top_left: quad.border.radius.top_left,
                    top_right: quad.border.radius.top_right,
                    bottom_right: quad.border.radius.bottom_right,
                    bottom_left: quad.border.radius.bottom_left,
                },
            },
            shadow: Shadow {
                color: color(quad.shadow.color),
                offset: Point::new(quad.shadow.offset.x, quad.shadow.offset.y),
                blur_radius: quad.shadow.blur_radius,
            },
            snap: quad.snap,
        });
    }

    fn allocate_image(
        &mut self,
        _handle: &image::Handle,
        callback: impl FnOnce(Result<image::Allocation, image::Error>) + Send + 'static,
    ) {
        callback(Err(image::Error::Unsupported));
    }

    fn hint(&mut self, scale: Scale) {
        self.scale = Some(scale);
        self.recorder.set_scale_factor(scale.total());
    }

    fn scale(&self) -> Option<Scale> {
        self.scale
    }

    fn reset(&mut self, new_bounds: Rectangle) {
        let scale_factor = self.scale.map_or(1.0, Scale::total);
        self.recorder.reset(rect(new_bounds), scale_factor);
    }

    fn settings(&self) -> renderer::Settings {
        self.settings
    }
}

fn rect(value: Rectangle) -> Rect {
    Rect::new(value.x, value.y, value.width, value.height)
}

fn color(value: iced_core::Color) -> Color {
    Color::new(value.r, value.g, value.b, value.a)
}

fn paint(background: Background) -> Paint {
    match background {
        Background::Color(value) => Paint::Solid(color(value)),
        Background::Gradient(gradient::Gradient::Linear(value)) => {
            Paint::LinearGradient(LinearGradient {
                angle: value.angle.0,
                stops: value.stops.map(|stop| {
                    stop.map(|stop| GradientStop {
                        offset: stop.offset,
                        color: color(stop.color),
                    })
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axui_core::Command;
    use iced_core::Renderer as _;
    use iced_core::{Border as IcedBorder, Color as IcedColor, Shadow as IcedShadow, Vector};

    #[test]
    fn records_iced_primitives_without_a_graphics_backend() {
        let mut renderer = Renderer::new_in(renderer::Settings::default(), std::alloc::System);
        renderer.hint(Scale {
            window: 2.0,
            application: 1.25,
        });
        renderer.reset(Rectangle::new(
            iced_core::Point::ORIGIN,
            iced_core::Size::new(800.0, 600.0),
        ));
        renderer.start_layer(Rectangle::new(
            iced_core::Point::new(10.0, 20.0),
            iced_core::Size::new(300.0, 200.0),
        ));
        renderer.start_transformation(Transformation::translate(4.0, 8.0));

        let gradient = gradient::Linear::new(iced_core::Radians(0.5))
            .add_stop(0.0, IcedColor::BLACK)
            .add_stop(1.0, IcedColor::WHITE);
        renderer.fill_quad(
            renderer::Quad {
                bounds: Rectangle::new(
                    iced_core::Point::new(30.0, 40.0),
                    iced_core::Size::new(120.0, 64.0),
                ),
                border: IcedBorder::default()
                    .color(IcedColor::from_rgb(0.2, 0.3, 0.4))
                    .width(2.0)
                    .rounded(12.0),
                shadow: IcedShadow {
                    color: IcedColor::from_rgba(0.0, 0.0, 0.0, 0.4),
                    offset: Vector::new(1.0, 3.0),
                    blur_radius: 8.0,
                },
                snap: true,
            },
            gradient,
        );
        renderer.end_transformation();
        renderer.end_layer();

        let frame = renderer.take_frame().unwrap();

        assert_eq!(frame.bounds(), Rect::new(0.0, 0.0, 800.0, 600.0));
        assert_eq!(frame.scale_factor(), 2.5);
        assert_eq!(frame.commands().len(), 5);
        let Command::DrawQuad(quad_index) = frame.commands()[2] else {
            panic!("expected the Iced quad to remain a typed draw command");
        };
        let quad = *frame.quad(quad_index).unwrap();
        assert_eq!(quad.bounds, Rect::new(30.0, 40.0, 120.0, 64.0));
        assert_eq!(quad.border.radii.top_left, 12.0);
        assert_eq!(quad.shadow.blur_radius, 8.0);
        let Paint::LinearGradient(gradient) = quad.paint else {
            panic!("expected a typed gradient");
        };
        assert_eq!(gradient.angle, 0.5);
        assert_eq!(gradient.stops[1].unwrap().offset, 1.0);
    }

    #[test]
    fn reports_image_allocation_as_unsupported() {
        let mut renderer = Renderer::new_in(renderer::Settings::default(), std::alloc::System);
        let handle = image::Handle::from_rgba(1, 1, vec![255, 255, 255, 255]);
        let (sender, receiver) = std::sync::mpsc::channel();

        renderer.allocate_image(&handle, move |result| sender.send(result).unwrap());

        assert!(matches!(
            receiver.recv().unwrap(),
            Err(image::Error::Unsupported)
        ));
    }

    #[test]
    fn records_with_a_caller_selected_allocator() {
        let mut renderer = Renderer::new_in(renderer::Settings::default(), std::alloc::System);
        renderer.reset(Rectangle::new(
            iced_core::Point::ORIGIN,
            iced_core::Size::new(80.0, 40.0),
        ));

        let frame: Frame<std::alloc::System> = renderer.take_frame().unwrap();

        assert_eq!(frame.bounds(), Rect::new(0.0, 0.0, 80.0, 40.0));
    }

    #[test]
    fn maps_iced_cursor_interactions_without_host_types() {
        assert_eq!(
            cursor_request(iced_core::mouse::Interaction::Pointer).icon,
            CursorIcon::Pointer
        );
        assert!(!cursor_request(iced_core::mouse::Interaction::Hidden).visible);
        assert_eq!(
            cursor_request(iced_core::mouse::Interaction::ResizingDiagonallyDown).icon,
            CursorIcon::ResizeDiagonalDown
        );
    }
}
