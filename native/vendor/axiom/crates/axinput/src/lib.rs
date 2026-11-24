//! Backend-neutral input policy for Axiom.
//!
//! Platform adapters decode device records into [`PointerInput`] values. A
//! [`Seat`] then owns viewport mapping, relative-motion scaling, button state,
//! and the logical cursor position. Device discovery, interrupts, capability
//! authority, and hardware cursor registers stay behind the runtime boundary.
//!
//! This crate allocates nothing. That is a property of this small state
//! machine, not a restriction on an Axiom runtime: cursor themes and input
//! queues that need owned storage must remain allocator-typed at their owning
//! layer.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "acceleration")]
pub mod acceleration;
pub mod frame;
#[cfg(feature = "pointing")]
pub mod pointing;

/// Logical identity of an input seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SeatId(pub u32);

/// Runtime-stable identity of one physical or virtual input source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DeviceId(pub u32);

/// Monotonic event timestamp in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct InputTimestamp(pub u64);

/// Physical display extent used by pointer policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    width: u32,
    height: u32,
}

impl Viewport {
    /// Creates a nonempty viewport.
    pub const fn new(width: u32, height: u32) -> Result<Self, InputError> {
        if width == 0 || height == 0 {
            return Err(InputError::ZeroExtent);
        }
        if width > i32::MAX as u32 || height > i32::MAX as u32 {
            return Err(InputError::ExtentTooLarge);
        }
        Ok(Self { width, height })
    }

    /// Returns the width in physical pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height in physical pixels.
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// A target-relative physical cursor position.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PointerPosition {
    /// Horizontal position in physical pixels.
    pub x: i32,
    /// Vertical position in physical pixels.
    pub y: i32,
}

impl PointerPosition {
    /// Creates a position.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Broad source physics used by seat policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerSource {
    /// Relative mouse or mouse-compatible collection.
    Mouse,
    /// Absolute contacts interpreted as trackpad gestures.
    Touchpad,
    /// Absolute direct-touch surface.
    Touchscreen,
    /// Pen or stylus digitizer.
    Pen,
    /// Synthetic source authorized by the runtime.
    Virtual,
}

/// A pointer button identifier. Values are not restricted to three buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PointerButton(pub u16);

impl PointerButton {
    /// Primary selection button.
    pub const PRIMARY: Self = Self(0);
    /// Secondary/context button.
    pub const SECONDARY: Self = Self(1);
    /// Middle/tertiary button.
    pub const MIDDLE: Self = Self(2);
}

/// Button edge carried by a pointer event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState {
    /// The button became held.
    Pressed,
    /// The button became released.
    Released,
}

/// Units attached to a scroll delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollUnit {
    /// Physical-pixel precision scrolling.
    Pixel,
    /// Logical line scrolling.
    Line,
    /// Discrete wheel detents.
    Detent,
}

/// Coordinate range published by an absolute-position device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteRange {
    /// Inclusive maximum horizontal device coordinate.
    pub max_x: u32,
    /// Inclusive maximum vertical device coordinate.
    pub max_y: u32,
}

impl AbsoluteRange {
    /// Creates a nonempty coordinate range.
    pub const fn new(max_x: u32, max_y: u32) -> Result<Self, InputError> {
        if max_x == 0 || max_y == 0 {
            return Err(InputError::ZeroRange);
        }
        Ok(Self { max_x, max_y })
    }
}

/// Contact lifecycle after a platform adapter has reconciled device reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactPhase {
    /// A new contact became active.
    Started,
    /// An active contact changed.
    Moved,
    /// A contact ended normally.
    Ended,
    /// A contact was cancelled by device loss or stream discontinuity.
    Cancelled,
}

/// One decoded pointer input record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerInput {
    /// Seat this record belongs to.
    pub seat: SeatId,
    /// Stable source identity supplied by the runtime.
    pub device: DeviceId,
    /// Source physics.
    pub source: PointerSource,
    /// Monotonic capture time, when the transport provides one.
    pub timestamp: Option<InputTimestamp>,
    /// Record payload.
    pub kind: PointerInputKind,
}

/// Payload of a [`PointerInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerInputKind {
    /// Relative device motion.
    RelativeMotion { dx: i32, dy: i32 },
    /// Absolute device motion in a declared coordinate range.
    AbsoluteMotion {
        x: u32,
        y: u32,
        range: AbsoluteRange,
    },
    /// One button edge.
    Button {
        button: PointerButton,
        state: ButtonState,
    },
    /// Two-axis scrolling.
    Scroll {
        horizontal: i32,
        vertical: i32,
        unit: ScrollUnit,
    },
    /// One reconciled absolute contact.
    Contact {
        id: u16,
        phase: ContactPhase,
        x: u32,
        y: u32,
        range: AbsoluteRange,
        contacts: u16,
        confident: bool,
    },
}

/// Rational gain applied to relative motion.
///
/// Keeping the policy rational makes it deterministic on every backend and
/// preserves fractional movement through the seat's private remainders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionScale {
    numerator: u16,
    denominator: u16,
}

impl MotionScale {
    /// No scaling.
    pub const IDENTITY: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    /// Creates a positive rational motion scale.
    pub const fn new(numerator: u16, denominator: u16) -> Result<Self, InputError> {
        if numerator == 0 || denominator == 0 {
            return Err(InputError::InvalidScale);
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }
}

/// High-level event after seat policy has resolved a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeatEvent {
    /// Seat producing the event.
    pub seat: SeatId,
    /// Source device.
    pub device: DeviceId,
    /// Capture timestamp, when known.
    pub timestamp: Option<InputTimestamp>,
    /// Current target-relative pointer position.
    pub position: PointerPosition,
    /// Resolved event payload.
    pub kind: SeatEventKind,
}

/// Payload of a [`SeatEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatEventKind {
    /// Cursor position changed.
    Motion,
    /// One button changed state.
    Button {
        button: PointerButton,
        state: ButtonState,
    },
    /// Scroll input; position identifies its target.
    Scroll {
        horizontal: i32,
        vertical: i32,
        unit: ScrollUnit,
    },
    /// Absolute contact preserved for gesture/direct-touch policy.
    Contact {
        id: u16,
        phase: ContactPhase,
        /// Original device coordinates, not the seat's cursor position.
        x: u32,
        y: u32,
        range: AbsoluteRange,
        contacts: u16,
        confident: bool,
    },
}

/// Seat-owned pointer policy and state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seat {
    id: SeatId,
    viewport: Viewport,
    position: PointerPosition,
    scale: MotionScale,
    remainder_x: i64,
    remainder_y: i64,
    buttons: u64,
}

impl Seat {
    /// Creates a seat with a clamped initial position.
    pub const fn new(
        id: SeatId,
        viewport: Viewport,
        position: PointerPosition,
        scale: MotionScale,
    ) -> Self {
        Self {
            id,
            viewport,
            position: clamp_position(position, viewport),
            scale,
            remainder_x: 0,
            remainder_y: 0,
            buttons: 0,
        }
    }

    /// Returns the seat identity.
    pub const fn id(&self) -> SeatId {
        self.id
    }

    /// Returns the current cursor position.
    pub const fn position(&self) -> PointerPosition {
        self.position
    }

    /// Returns whether a representable button is held.
    pub const fn button_pressed(&self, button: PointerButton) -> bool {
        button.0 < 64 && self.buttons & (1_u64 << button.0) != 0
    }

    /// Reconfigures the target extent and clamps the cursor immediately.
    pub fn set_viewport(&mut self, viewport: Viewport) -> PointerPosition {
        self.viewport = viewport;
        self.position = clamp_position(self.position, viewport);
        self.position
    }

    /// Changes relative-motion policy without discarding subpixel remainder.
    pub fn set_motion_scale(&mut self, scale: MotionScale) {
        self.scale = scale;
    }

    /// Applies one platform record.
    pub fn feed(&mut self, input: PointerInput) -> Result<SeatEvent, InputError> {
        if input.seat != self.id {
            return Err(InputError::WrongSeat);
        }

        let kind = match input.kind {
            PointerInputKind::RelativeMotion { dx, dy } => {
                let dx = scaled_component(dx, self.scale, &mut self.remainder_x);
                let dy = scaled_component(dy, self.scale, &mut self.remainder_y);
                self.position = clamp_position(
                    PointerPosition::new(
                        self.position.x.saturating_add(dx),
                        self.position.y.saturating_add(dy),
                    ),
                    self.viewport,
                );
                SeatEventKind::Motion
            }
            PointerInputKind::AbsoluteMotion { x, y, range } => {
                AbsoluteRange::new(range.max_x, range.max_y)?;
                self.position = map_absolute(x, y, range, self.viewport);
                SeatEventKind::Motion
            }
            PointerInputKind::Button { button, state } => {
                if button.0 < 64 {
                    let mask = 1_u64 << button.0;
                    match state {
                        ButtonState::Pressed => self.buttons |= mask,
                        ButtonState::Released => self.buttons &= !mask,
                    }
                }
                SeatEventKind::Button { button, state }
            }
            PointerInputKind::Scroll {
                horizontal,
                vertical,
                unit,
            } => SeatEventKind::Scroll {
                horizontal,
                vertical,
                unit,
            },
            PointerInputKind::Contact {
                id,
                phase,
                x,
                y,
                range,
                contacts,
                confident,
            } => {
                AbsoluteRange::new(range.max_x, range.max_y)?;
                // Direct devices own an absolute screen position. Touchpads
                // preserve contacts for the gesture layer instead of silently
                // imposing touchscreen semantics.
                if matches!(
                    input.source,
                    PointerSource::Touchscreen | PointerSource::Pen
                ) {
                    self.position = map_absolute(x, y, range, self.viewport);
                }
                SeatEventKind::Contact {
                    id,
                    phase,
                    x,
                    y,
                    range,
                    contacts,
                    confident,
                }
            }
        };

        Ok(SeatEvent {
            seat: input.seat,
            device: input.device,
            timestamp: input.timestamp,
            position: self.position,
            kind,
        })
    }
}

/// Single-finger touchpad policy, independent of any HID report layout.
/// A full device-axis stroke spans an explicit reference extent (1000 by
/// default), independently of viewport size. Prefer `frame::PointerFallback`
/// for calibrated, framed input. Lifts, palm rejection, contact replacement,
/// and multi-finger reports reset the anchor instead of moving the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TouchpadMotion {
    anchor: Option<(DeviceId, u16, u32, u32, AbsoluteRange)>,
    remainder_x: i64,
    remainder_y: i64,
    reference_extent: u32,
}

impl Default for TouchpadMotion {
    fn default() -> Self {
        Self {
            anchor: None,
            remainder_x: 0,
            remainder_y: 0,
            reference_extent: 1000,
        }
    }
}

impl TouchpadMotion {
    pub fn with_reference_extent(extent: u32) -> Result<Self, InputError> {
        if extent == 0 {
            return Err(InputError::ZeroExtent);
        }
        if extent > i32::MAX as u32 {
            return Err(InputError::ExtentTooLarge);
        }
        Ok(Self {
            reference_extent: extent,
            ..Self::default()
        })
    }
    /// Forget an anchor after device loss or a stream discontinuity.
    pub fn reset(&mut self) {
        self.anchor = None;
        self.remainder_x = 0;
        self.remainder_y = 0;
    }

    /// Convert a reconciled contact into relative motion. This deliberately
    /// does not invent taps, clicks, scrolling, or a capture timestamp.
    pub fn feed(
        &mut self,
        input: PointerInput,
        _viewport: Viewport,
    ) -> Result<Option<PointerInput>, InputError> {
        let PointerInputKind::Contact {
            id,
            phase,
            x,
            y,
            range,
            contacts,
            confident,
        } = input.kind
        else {
            return Ok(None);
        };
        AbsoluteRange::new(range.max_x, range.max_y)?;
        if input.source != PointerSource::Touchpad {
            return Ok(None);
        }
        if !confident
            || contacts != 1
            || matches!(phase, ContactPhase::Ended | ContactPhase::Cancelled)
        {
            self.reset();
            return Ok(None);
        }
        let (x, y) = (x.min(range.max_x), y.min(range.max_y));
        let previous = self.anchor;
        self.anchor = Some((input.device, id, x, y, range));
        let Some((old_device, old_id, old_x, old_y, old_range)) = previous else {
            return Ok(None);
        };
        if old_id != id
            || old_range != range
            || old_device != input.device
            || phase == ContactPhase::Started
        {
            self.remainder_x = 0;
            self.remainder_y = 0;
            return Ok(None);
        }
        let delta = |now: u32, old: u32, extent: u32, maximum: u32, remainder: &mut i64| {
            let numerator = (i64::from(now) - i64::from(old)) * i64::from(extent) + *remainder;
            let denominator = i64::from(maximum);
            *remainder = numerator % denominator;
            (numerator / denominator).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
        };
        let dx = delta(
            x,
            old_x,
            self.reference_extent,
            range.max_x,
            &mut self.remainder_x,
        );
        let dy = delta(
            y,
            old_y,
            self.reference_extent,
            range.max_y,
            &mut self.remainder_y,
        );
        Ok(Some(PointerInput {
            kind: PointerInputKind::RelativeMotion { dx, dy },
            ..input
        }))
    }
}

/// Stable cursor image resource identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CursorImageId(pub u64);

/// Backend-neutral cursor shape vocabulary.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorIcon {
    #[default]
    Default,
    ContextMenu,
    Help,
    Pointer,
    Progress,
    Wait,
    Cell,
    Crosshair,
    Text,
    Alias,
    Copy,
    Move,
    NoDrop,
    NotAllowed,
    Grab,
    Grabbing,
    ResizeHorizontal,
    ResizeVertical,
    ResizeDiagonalUp,
    ResizeDiagonalDown,
    ResizeColumn,
    ResizeRow,
    AllScroll,
    ZoomIn,
    ZoomOut,
    /// Runtime-owned image from an allocator-typed cursor theme.
    Custom(CursorImageId),
}

/// Cursor confinement/relative-motion request made by UI policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CursorGrab {
    #[default]
    None,
    Confined,
    Locked,
}

/// Resolved cursor request for one seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorRequest {
    /// Requested shape.
    pub icon: CursorIcon,
    /// Whether the cursor should be visible.
    pub visible: bool,
    /// Confinement or relative-motion posture.
    pub grab: CursorGrab,
}

impl CursorRequest {
    /// Ordinary visible default cursor.
    pub const DEFAULT: Self = Self {
        icon: CursorIcon::Default,
        visible: true,
        grab: CursorGrab::None,
    };
}

impl Default for CursorRequest {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Invalid input contract or seat operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputError {
    ZeroExtent,
    ExtentTooLarge,
    ZeroRange,
    InvalidScale,
    WrongSeat,
}

const fn clamp_position(position: PointerPosition, viewport: Viewport) -> PointerPosition {
    let max_x = viewport.width.saturating_sub(1) as i32;
    let max_y = viewport.height.saturating_sub(1) as i32;
    PointerPosition {
        x: if position.x < 0 {
            0
        } else if position.x > max_x {
            max_x
        } else {
            position.x
        },
        y: if position.y < 0 {
            0
        } else if position.y > max_y {
            max_y
        } else {
            position.y
        },
    }
}

fn scaled_component(value: i32, scale: MotionScale, remainder: &mut i64) -> i32 {
    let dividend = i64::from(value)
        .saturating_mul(i64::from(scale.numerator))
        .saturating_add(*remainder);
    let denominator = i64::from(scale.denominator);
    let quotient = dividend / denominator;
    *remainder = dividend % denominator;
    quotient.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn map_absolute(x: u32, y: u32, range: AbsoluteRange, viewport: Viewport) -> PointerPosition {
    let map = |value: u32, maximum: u32, pixels: u32| {
        let last = pixels.saturating_sub(1);
        (u64::from(value.min(maximum)) * u64::from(last) / u64::from(maximum)) as i32
    };
    PointerPosition::new(
        map(x, range.max_x, viewport.width),
        map(y, range.max_y, viewport.height),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative(dx: i32, dy: i32) -> PointerInput {
        PointerInput {
            seat: SeatId(1),
            device: DeviceId(7),
            source: PointerSource::Mouse,
            timestamp: Some(InputTimestamp(42)),
            kind: PointerInputKind::RelativeMotion { dx, dy },
        }
    }

    #[test]
    fn relative_motion_preserves_fractional_gain_and_clamps() {
        let viewport = Viewport::new(10, 8).unwrap();
        let scale = MotionScale::new(3, 2).unwrap();
        let mut seat = Seat::new(SeatId(1), viewport, PointerPosition::new(8, 6), scale);

        assert_eq!(
            seat.feed(relative(1, 1)).unwrap().position,
            PointerPosition::new(9, 7)
        );
        assert_eq!(
            seat.feed(relative(-1, -1)).unwrap().position,
            PointerPosition::new(8, 6)
        );
        assert_eq!(
            seat.feed(relative(-99, 99)).unwrap().position,
            PointerPosition::new(0, 7)
        );
    }

    #[test]
    fn absolute_motion_maps_device_endpoints_to_viewport_endpoints() {
        let mut seat = Seat::new(
            SeatId(1),
            Viewport::new(1920, 1080).unwrap(),
            PointerPosition::default(),
            MotionScale::IDENTITY,
        );
        let event = seat
            .feed(PointerInput {
                seat: SeatId(1),
                device: DeviceId(9),
                source: PointerSource::Touchscreen,
                timestamp: None,
                kind: PointerInputKind::AbsoluteMotion {
                    x: 4095,
                    y: 2047,
                    range: AbsoluteRange::new(4095, 2047).unwrap(),
                },
            })
            .unwrap();
        assert_eq!(event.position, PointerPosition::new(1919, 1079));
    }

    #[test]
    fn buttons_are_extensible_but_common_state_is_tracked() {
        let mut seat = Seat::new(
            SeatId(1),
            Viewport::new(2, 2).unwrap(),
            PointerPosition::default(),
            MotionScale::IDENTITY,
        );
        let button = PointerButton(5);
        seat.feed(PointerInput {
            kind: PointerInputKind::Button {
                button,
                state: ButtonState::Pressed,
            },
            ..relative(0, 0)
        })
        .unwrap();
        assert!(seat.button_pressed(button));
    }

    #[test]
    fn wrong_seat_is_rejected() {
        let mut seat = Seat::new(
            SeatId(2),
            Viewport::new(2, 2).unwrap(),
            PointerPosition::default(),
            MotionScale::IDENTITY,
        );
        assert_eq!(seat.feed(relative(1, 1)), Err(InputError::WrongSeat));
    }

    fn contact(id: u16, x: u32, contacts: u16, confident: bool) -> PointerInput {
        PointerInput {
            source: PointerSource::Touchpad,
            kind: PointerInputKind::Contact {
                id,
                phase: ContactPhase::Moved,
                x,
                y: 100,
                range: AbsoluteRange::new(1000, 500).unwrap(),
                contacts,
                confident,
            },
            ..relative(0, 0)
        }
    }

    #[test]
    fn touchpad_preserves_fractional_motion_and_reanchors_after_lift_or_palm() {
        let viewport = Viewport::new(100, 50).unwrap();
        let mut motion = TouchpadMotion::with_reference_extent(100).unwrap();
        assert_eq!(motion.feed(contact(1, 10, 1, true), viewport), Ok(None));
        for x in 11..20 {
            assert_eq!(
                motion
                    .feed(contact(1, x, 1, true), viewport)
                    .unwrap()
                    .unwrap()
                    .kind,
                PointerInputKind::RelativeMotion { dx: 0, dy: 0 }
            );
        }
        assert_eq!(
            motion
                .feed(contact(1, 20, 1, true), viewport)
                .unwrap()
                .unwrap()
                .kind,
            PointerInputKind::RelativeMotion { dx: 1, dy: 0 }
        );
        for discarded in [
            contact(1, 200, 2, true),
            contact(1, 200, 1, false),
            contact(1, 200, 0, true),
        ] {
            assert_eq!(motion.feed(discarded, viewport), Ok(None));
            assert_eq!(motion.feed(contact(1, 900, 1, true), viewport), Ok(None));
        }
        assert_eq!(motion.feed(contact(2, 1, 1, true), viewport), Ok(None));
    }

    #[test]
    fn invalid_absolute_range_is_refused_without_panicking_or_moving() {
        let mut seat = Seat::new(
            SeatId(1),
            Viewport::new(100, 50).unwrap(),
            PointerPosition::new(2, 3),
            MotionScale::IDENTITY,
        );
        let input = PointerInput {
            kind: PointerInputKind::AbsoluteMotion {
                x: 1,
                y: 1,
                range: AbsoluteRange { max_x: 0, max_y: 1 },
            },
            ..relative(0, 0)
        };
        assert_eq!(seat.feed(input), Err(InputError::ZeroRange));
        assert_eq!(seat.position(), PointerPosition::new(2, 3));
    }
}
