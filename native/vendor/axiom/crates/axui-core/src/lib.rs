//! Backend-neutral display-list types for Axiom user interfaces.
//!
//! `axui-core` is the seam between a UI frontend and an Axiom renderer. It
//! deliberately contains no windowing, widget, font, raster, or accelerator
//! implementation. Frontends record logical primitives here; an `axrd`
//! integration can lower the resulting [`Frame`] into `.ax` compute graphs.

#![no_std]
#![feature(allocator_api)]

extern crate alloc;

#[cfg(test)]
extern crate std;

use alloc::vec::Vec;
use core::{alloc::Allocator, fmt, marker::PhantomData};

/// A point in logical UI coordinates.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl Point {
    /// Creates a point.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned rectangle in logical UI coordinates.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Horizontal origin.
    pub x: f32,
    /// Vertical origin.
    pub y: f32,
    /// Logical width.
    pub width: f32,
    /// Logical height.
    pub height: f32,
}

impl Rect {
    /// Creates a rectangle.
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// A normalized sRGB color with a linear alpha channel.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

impl Color {
    /// Creates a color from normalized channels.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// Per-corner radii in top-left, top-right, bottom-right, bottom-left order.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CornerRadii {
    /// Top-left radius.
    pub top_left: f32,
    /// Top-right radius.
    pub top_right: f32,
    /// Bottom-right radius.
    pub bottom_right: f32,
    /// Bottom-left radius.
    pub bottom_left: f32,
}

/// An inner border around a quad.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Border {
    /// Border color.
    pub color: Color,
    /// Logical width.
    pub width: f32,
    /// Per-corner radii.
    pub radii: CornerRadii,
}

/// A drop shadow behind a quad.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Shadow {
    /// Shadow color.
    pub color: Color,
    /// Logical offset.
    pub offset: Point,
    /// Logical blur radius.
    pub blur_radius: f32,
}

/// One stop in a linear gradient.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GradientStop {
    /// Position on the gradient axis, normally in `0.0..=1.0`.
    pub offset: f32,
    /// Color at this position.
    pub color: Color,
}

/// A linear gradient with a fixed-size stop table suitable for graph lowering.
/// A renderable gradient contains at least two ordered stops; `axgfx`
/// validates this before backend admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradient {
    /// Angle in radians.
    pub angle: f32,
    /// Ordered color stops. Empty slots remain `None`.
    pub stops: [Option<GradientStop>; 8],
}

/// Paint applied to a quad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Paint {
    /// A constant color.
    Solid(Color),
    /// A linear color gradient.
    LinearGradient(LinearGradient),
}

/// A filled, optionally bordered and shadowed rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// Logical bounds.
    pub bounds: Rect,
    /// Interior paint.
    pub paint: Paint,
    /// Inner border.
    pub border: Border,
    /// Drop shadow.
    pub shadow: Shadow,
    /// Whether the backend should snap edges to physical pixels.
    pub snap: bool,
}

/// A 4×4 column-major transformation matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform(pub [f32; 16]);

impl Transform {
    /// Identity transform.
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]);
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A backend-independent rendering command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command {
    /// Starts a layer clipped to the given rectangle.
    PushClip(Rect),
    /// Ends the most recently started clip layer.
    PopClip,
    /// Starts a transformed command group.
    PushTransform(Transform),
    /// Ends the most recently started transform group.
    PopTransform,
    /// Draws the quad at this index in [`FrameView::quads`].
    DrawQuad(usize),
}

/// An owned logical frame recorded with a caller-selected allocator.
///
/// Backends consume [`FrameView`] so allocator provenance remains in the type
/// chain without transferring ownership of the display tables.
#[derive(Debug, Clone)]
pub struct Frame<A: Allocator> {
    /// Logical frame bounds.
    bounds: Rect,
    /// Logical-to-physical scale hint.
    scale_factor: f32,
    /// Ordered display-list commands.
    commands: Vec<Command, A>,
    /// Quad arena referenced by [`Command::DrawQuad`].
    ///
    /// Keeping the relatively large paint data out of every command makes the
    /// hot command stream compact without allocating each primitive in a box.
    quads: Vec<Quad, A>,
}

impl<A: Allocator, B: Allocator> PartialEq<Frame<B>> for Frame<A> {
    fn eq(&self, other: &Frame<B>) -> bool {
        self.bounds == other.bounds
            && self.scale_factor == other.scale_factor
            && self.commands == other.commands
            && self.quads == other.quads
    }
}

impl<A: Allocator + Clone> Frame<A> {
    /// Creates an empty frame using `allocator` for all owned display data.
    pub fn new_in(bounds: Rect, scale_factor: f32, allocator: A) -> Self {
        Self {
            bounds,
            scale_factor,
            commands: Vec::new_in(allocator.clone()),
            quads: Vec::new_in(allocator),
        }
    }

    /// Copies a borrowed display list into caller-selected persistent storage.
    ///
    /// Rendering adapters use this to retain an exact structural-cache key
    /// without retaining the frontend's transient frame allocation or falling
    /// back to an implicit global allocator.
    pub fn copy_from_in<B: Allocator>(frame: FrameView<'_, B>, allocator: A) -> Self {
        let mut owned = Self::new_in(frame.bounds(), frame.scale_factor(), allocator);
        owned.commands.extend_from_slice(frame.commands());
        owned.quads.extend_from_slice(frame.quads());
        owned
    }
}

impl<A: Allocator> Frame<A> {
    /// Returns the logical frame bounds.
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Returns the logical-to-physical scale hint.
    pub const fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Returns the ordered command stream.
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// Returns the contiguous quad arena.
    pub fn quads(&self) -> &[Quad] {
        &self.quads
    }

    /// Resolves a quad arena index.
    pub fn quad(&self, index: usize) -> Option<&Quad> {
        self.quads.get(index)
    }

    /// Borrows this frame while retaining its allocator type.
    pub fn view(&self) -> FrameView<'_, A> {
        FrameView::new(self.bounds, self.scale_factor, &self.commands, &self.quads)
    }
}

/// Borrowed display data retaining end-to-end allocator provenance.
pub struct FrameView<'a, A: Allocator> {
    bounds: Rect,
    scale_factor: f32,
    commands: &'a [Command],
    quads: &'a [Quad],
    allocator: PhantomData<&'a A>,
}

impl<A: Allocator> Copy for FrameView<'_, A> {}

impl<A: Allocator> Clone for FrameView<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Allocator, B: Allocator> PartialEq<FrameView<'_, B>> for FrameView<'_, A> {
    fn eq(&self, other: &FrameView<'_, B>) -> bool {
        self.bounds == other.bounds
            && self.scale_factor == other.scale_factor
            && self.commands == other.commands
            && self.quads == other.quads
    }
}

impl<A: Allocator> fmt::Debug for FrameView<'_, A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrameView")
            .field("bounds", &self.bounds)
            .field("scale_factor", &self.scale_factor)
            .field("commands", &self.commands)
            .field("quads", &self.quads)
            .finish()
    }
}

impl<'a, A: Allocator> FrameView<'a, A> {
    const fn new(
        bounds: Rect,
        scale_factor: f32,
        commands: &'a [Command],
        quads: &'a [Quad],
    ) -> Self {
        Self {
            bounds,
            scale_factor,
            commands,
            quads,
            allocator: PhantomData,
        }
    }

    /// Returns the logical frame bounds.
    pub const fn bounds(self) -> Rect {
        self.bounds
    }

    /// Returns the logical-to-physical scale hint.
    pub const fn scale_factor(self) -> f32 {
        self.scale_factor
    }

    /// Returns the ordered command stream.
    pub const fn commands(self) -> &'a [Command] {
        self.commands
    }

    /// Returns the contiguous quad arena.
    pub const fn quads(self) -> &'a [Quad] {
        self.quads
    }

    /// Resolves a quad arena index.
    pub const fn quad(self, index: usize) -> Option<&'a Quad> {
        if index < self.quads.len() {
            Some(&self.quads[index])
        } else {
            None
        }
    }
}

impl<'a, A: Allocator> From<&'a Frame<A>> for FrameView<'a, A> {
    fn from(frame: &'a Frame<A>) -> Self {
        frame.view()
    }
}

/// Structural problem in a recorded display list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityError {
    /// A group was ended without a corresponding start.
    StackUnderflow(GroupKind),
    /// A group was ended out of last-in, first-out order.
    MismatchedEnd {
        /// The group that is currently open.
        expected: GroupKind,
        /// The group the frontend attempted to end.
        found: GroupKind,
    },
    /// One or more groups were left open at the end of the frame.
    UnclosedGroups(usize),
}

/// Kind of nested display-list group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// A clipped layer.
    Clip,
    /// A transformed group.
    Transform,
}

/// Records commands while validating group nesting.
///
/// All owned storage uses the allocator supplied to [`Recorder::new_in`].
#[derive(Debug, Clone)]
pub struct Recorder<A: Allocator> {
    frame: Frame<A>,
    group_stack: Vec<GroupKind, A>,
    first_error: Option<IntegrityError>,
    allocator: A,
}

impl<A: Allocator + Clone> Recorder<A> {
    /// Creates a recorder using `allocator` for the frame and nesting stack.
    pub fn new_in(bounds: Rect, scale_factor: f32, allocator: A) -> Self {
        Self {
            frame: Frame::new_in(bounds, scale_factor, allocator.clone()),
            group_stack: Vec::new_in(allocator.clone()),
            first_error: None,
            allocator,
        }
    }

    /// Starts a new frame, retaining the command allocation.
    pub fn reset(&mut self, bounds: Rect, scale_factor: f32) {
        self.frame.bounds = bounds;
        self.frame.scale_factor = scale_factor;
        self.frame.commands.clear();
        self.frame.quads.clear();
        self.group_stack.clear();
        self.first_error = None;
    }

    /// Updates the physical scale hint without changing logical coordinates.
    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.frame.scale_factor = scale_factor;
    }

    /// Starts a clipped layer.
    pub fn push_clip(&mut self, bounds: Rect) {
        self.group_stack.push(GroupKind::Clip);
        self.frame.commands.push(Command::PushClip(bounds));
    }

    /// Ends the current clipped layer.
    pub fn pop_clip(&mut self) {
        if self.pop_group(GroupKind::Clip) {
            self.frame.commands.push(Command::PopClip);
        }
    }

    /// Starts a transformed command group.
    pub fn push_transform(&mut self, transform: Transform) {
        self.group_stack.push(GroupKind::Transform);
        self.frame.commands.push(Command::PushTransform(transform));
    }

    /// Ends the current transformed command group.
    pub fn pop_transform(&mut self) {
        if self.pop_group(GroupKind::Transform) {
            self.frame.commands.push(Command::PopTransform);
        }
    }

    /// Records a quad.
    pub fn draw_quad(&mut self, quad: Quad) {
        let index = self.frame.quads.len();
        self.frame.quads.push(quad);
        self.frame.commands.push(Command::DrawQuad(index));
    }

    /// Returns the current frame even if recording is incomplete.
    pub const fn frame(&self) -> &Frame<A> {
        &self.frame
    }

    /// Returns the frame if all groups are balanced.
    pub fn validated_frame(&self) -> Result<&Frame<A>, IntegrityError> {
        if let Some(error) = self.first_error {
            return Err(error);
        }
        if !self.group_stack.is_empty() {
            return Err(IntegrityError::UnclosedGroups(self.group_stack.len()));
        }

        Ok(&self.frame)
    }

    /// Takes the current frame and starts a fresh empty one with the same target.
    pub fn take_frame(&mut self) -> Result<Frame<A>, IntegrityError> {
        self.validated_frame()?;

        let next = Frame::new_in(
            self.frame.bounds,
            self.frame.scale_factor,
            self.allocator.clone(),
        );
        Ok(core::mem::replace(&mut self.frame, next))
    }

    fn pop_group(&mut self, found: GroupKind) -> bool {
        let Some(expected) = self.group_stack.last().copied() else {
            self.first_error
                .get_or_insert(IntegrityError::StackUnderflow(found));
            return false;
        };

        if expected != found {
            self.first_error
                .get_or_insert(IntegrityError::MismatchedEnd { expected, found });
            return false;
        }

        self.group_stack.pop();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_balanced_groups() {
        let mut recorder =
            Recorder::new_in(Rect::new(0.0, 0.0, 800.0, 600.0), 2.0, std::alloc::System);
        recorder.push_clip(Rect::new(10.0, 20.0, 100.0, 80.0));
        recorder.push_transform(Transform::IDENTITY);
        recorder.pop_transform();
        recorder.pop_clip();

        assert_eq!(recorder.validated_frame().unwrap().commands().len(), 4);
    }

    #[test]
    fn reports_underflow_without_emitting_a_false_pop() {
        let mut recorder = Recorder::new_in(Rect::default(), 1.0, std::alloc::System);
        recorder.pop_clip();

        assert_eq!(
            recorder.validated_frame(),
            Err(IntegrityError::StackUnderflow(GroupKind::Clip))
        );
        assert!(recorder.frame().commands().is_empty());
    }

    #[test]
    fn rejects_groups_ended_out_of_order() {
        let bounds = Rect::new(0.0, 0.0, 10.0, 10.0);
        let mut recorder = Recorder::new_in(bounds, 1.0, std::alloc::System);
        recorder.push_clip(bounds);
        recorder.push_transform(Transform::IDENTITY);
        recorder.pop_clip();

        assert_eq!(
            recorder.validated_frame(),
            Err(IntegrityError::MismatchedEnd {
                expected: GroupKind::Transform,
                found: GroupKind::Clip,
            })
        );
        assert_eq!(recorder.frame().commands().len(), 2);
    }

    #[test]
    fn take_frame_reuses_the_target_contract() {
        let bounds = Rect::new(1.0, 2.0, 3.0, 4.0);
        let mut recorder = Recorder::new_in(bounds, 1.5, std::alloc::System);
        recorder.push_clip(bounds);
        recorder.pop_clip();

        let frame = recorder.take_frame().unwrap();

        assert_eq!(frame.commands().len(), 2);
        assert_eq!(
            recorder.frame(),
            &Frame::new_in(bounds, 1.5, std::alloc::System)
        );
    }

    #[test]
    fn records_with_a_caller_selected_allocator() {
        let bounds = Rect::new(0.0, 0.0, 64.0, 32.0);
        let mut recorder = Recorder::new_in(bounds, 1.0, std::alloc::System);
        recorder.draw_quad(Quad {
            bounds,
            paint: Paint::Solid(Color::new(1.0, 0.5, 0.25, 1.0)),
            border: Border::default(),
            shadow: Shadow::default(),
            snap: true,
        });

        let frame: Frame<std::alloc::System> = recorder.take_frame().unwrap();

        assert_eq!(frame.view().commands(), &[Command::DrawQuad(0)]);
        assert_eq!(frame.view().quads().len(), 1);
    }

    #[test]
    fn copies_a_borrowed_frame_into_explicit_persistent_storage() {
        let bounds = Rect::new(0.0, 0.0, 64.0, 32.0);
        let mut recorder = Recorder::new_in(bounds, 1.25, std::alloc::System);
        recorder.draw_quad(Quad {
            bounds,
            paint: Paint::Solid(Color::new(0.2, 0.4, 0.6, 1.0)),
            border: Border::default(),
            shadow: Shadow::default(),
            snap: true,
        });

        let copy = Frame::copy_from_in(recorder.frame().view(), std::alloc::System);

        assert_eq!(copy, *recorder.frame());
    }
}
