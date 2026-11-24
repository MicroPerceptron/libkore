//! Framed device input, before cursor mapping or gesture recognition.
//!
//! Views borrow runtime-owned storage. An implementation of [`FramePolicy`]
//! may own `Allocator`-parameterized storage; this boundary requires neither
//! a global allocator nor an OS clock, event loop, or accelerator.

use crate::{
    AbsoluteRange, ButtonState, DeviceId, InputTimestamp, PointerButton, PointerInput,
    PointerInputKind, PointerSource, ScrollUnit, SeatId,
};

/// Logical device lifetime. A reused device ID must have a newer generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceGeneration(pub u32);

/// Inclusive device coordinates. These are not viewport coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisRange {
    min: i32,
    max: i32,
}

impl AxisRange {
    pub const fn new(min: i32, max: i32) -> Result<Self, FrameError> {
        if min >= max {
            return Err(FrameError::InvalidAxes);
        }
        Ok(Self { min, max })
    }
    pub const fn min(self) -> i32 {
        self.min
    }
    pub const fn max(self) -> i32 {
        self.max
    }
    pub const fn span(self) -> u32 {
        (self.max as i64 - self.min as i64) as u32
    }
    pub const fn contains(self, value: i32) -> bool {
        value >= self.min && value <= self.max
    }
}

/// Optional descriptor calibration. Unknown resolution is not inferred from
/// a display size. A rational preserves the descriptor's unit conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisResolution {
    units: u32,
    millimeters: u32,
}

impl AxisResolution {
    pub const fn new(units: u32, millimeters: u32) -> Result<Self, FrameError> {
        if units == 0 || millimeters == 0 {
            return Err(FrameError::InvalidAxes);
        }
        Ok(Self { units, millimeters })
    }
    pub const fn units(self) -> u32 {
        self.units
    }
    pub const fn millimeters(self) -> u32 {
        self.millimeters
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteAxis {
    pub range: AxisRange,
    pub resolution: Option<AxisResolution>,
}

/// Immutable facts for a device lifetime. Additional capability fields can
/// be introduced without making consumers construct this structure directly.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceDescriptor {
    pub seat: SeatId,
    pub device: DeviceId,
    pub generation: DeviceGeneration,
    pub source: PointerSource,
    pub axes: Option<[AbsoluteAxis; 2]>,
    /// Optional calibration of relative device counts (e.g. mouse DPI).
    pub relative_resolution: [Option<AxisResolution>; 2],
    /// Actual device limit if known, not the size of a transport scratch buffer.
    pub max_contacts: Option<u16>,
}

impl DeviceDescriptor {
    pub const fn new(
        seat: SeatId,
        device: DeviceId,
        generation: DeviceGeneration,
        source: PointerSource,
    ) -> Self {
        Self {
            seat,
            device,
            generation,
            source,
            axes: None,
            relative_resolution: [None; 2],
            max_contacts: None,
        }
    }
}

/// One device sample. Optional values retain HID logical units, not fabricated
/// normalized pressure or force. Width/height are contact dimensions, not
/// finger-count estimates. Unknown firmware confidence is distinct from false.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContactSample {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub touching: bool,
    pub in_range: Option<bool>,
    pub confidence: Option<bool>,
    pub pressure: Option<i32>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub orientation: Option<i32>,
}

impl ContactSample {
    pub const fn new(id: u32, x: i32, y: i32, touching: bool) -> Self {
        Self {
            id,
            x,
            y,
            touching,
            in_range: None,
            confidence: None,
            pressure: None,
            width: None,
            height: None,
            orientation: None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampOrigin {
    /// No timestamp was available. Time-dependent policies must not guess one.
    Unknown,
    /// Host receipt time in the runtime's monotonic nanosecond clock domain.
    Received,
    /// Device capture time converted by the runtime into that same clock domain.
    Captured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCompleteness {
    /// A complete contact snapshot, including an empty snapshot on lift.
    Complete,
    /// A coherent report fragment, NOT the full set of active contacts.
    /// Consumers may assemble fragments using device-specific scan metadata.
    Fragment,
}

/// High-resolution scroll retains fractional units, independently of pointer
/// position. Legacy integer scroll events are only a compatibility projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollSample {
    pub horizontal: f64,
    pub vertical: f64,
    pub unit: ScrollUnit,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerFrame<'a> {
    pub descriptor: &'a DeviceDescriptor,
    pub sequence: u32,
    pub timestamp: Option<InputTimestamp>,
    pub timestamp_origin: TimestampOrigin,
    /// Unconverted device scan counter, if available; never a host timestamp.
    pub scan_time: Option<u32>,
    pub completeness: FrameCompleteness,
    pub contacts: &'a [ContactSample],
    /// The count advertised by this hardware report, with no inferred fingers.
    pub reported_contacts: Option<u16>,
    pub buttons: u64,
    /// Raw signed device counts, before acceleration or display mapping.
    pub relative: [i32; 2],
    pub scroll: Option<ScrollSample>,
}

impl<'a> PointerFrame<'a> {
    pub const fn new(
        descriptor: &'a DeviceDescriptor,
        sequence: u32,
        contacts: &'a [ContactSample],
    ) -> Self {
        Self {
            descriptor,
            sequence,
            timestamp: None,
            timestamp_origin: TimestampOrigin::Unknown,
            scan_time: None,
            completeness: FrameCompleteness::Complete,
            contacts,
            reported_contacts: None,
            buttons: 0,
            relative: [0; 2],
            scroll: None,
        }
    }

    /// Validate before mutating any recognizer state. The caller bounds its
    /// contact storage; duplicate-ID checking is quadratic only in that small
    /// contact count, never in history or the number of input devices.
    pub fn validate(self) -> Result<(), FrameError> {
        if (self.timestamp_origin == TimestampOrigin::Unknown) != self.timestamp.is_none() {
            return Err(FrameError::InvalidTimestamp);
        }
        if self
            .descriptor
            .max_contacts
            .is_some_and(|max| self.contacts.len() > usize::from(max))
            || (self.completeness == FrameCompleteness::Complete
                && self
                    .reported_contacts
                    .is_some_and(|n| usize::from(n) > self.contacts.len()))
            || self
                .reported_contacts
                .zip(self.descriptor.max_contacts)
                .is_some_and(|(n, max)| n > max)
        {
            return Err(FrameError::Capacity);
        }
        if !self.contacts.is_empty() && self.descriptor.axes.is_none() {
            return Err(FrameError::InvalidAxes);
        }
        for (index, contact) in self.contacts.iter().enumerate() {
            if self.contacts[..index].iter().any(|c| c.id == contact.id) {
                return Err(FrameError::DuplicateContact);
            }
            if let Some([x, y]) = self.descriptor.axes {
                if !x.range.contains(contact.x) || !y.range.contains(contact.y) {
                    return Err(FrameError::InvalidAxes);
                }
            }
        }
        if self
            .scroll
            .is_some_and(|s| !s.horizontal.is_finite() || !s.vertical.is_finite())
        {
            return Err(FrameError::InvalidMotion);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    InvalidAxes,
    InvalidTimestamp,
    InvalidMotion,
    DuplicateContact,
    Capacity,
    WrongDevice,
    StaleGeneration,
    StaleSequence,
    TimeReversed,
    DescriptorChanged,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    StreamLost,
    DeviceRemoved,
    DeviceReset,
    FocusLost,
    MalformedFrame,
}

/// Per-device replay/continuity guard. Rejected input never advances history.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameHistory {
    last: Option<(DeviceDescriptor, u32, Option<InputTimestamp>)>,
    interrupted: bool,
    retired: bool,
}

impl FrameHistory {
    pub fn interrupt(&mut self) {
        self.interrupted = true;
    }

    /// Removal/reset retires the last accepted device lifetime. A mere queue
    /// loss uses `interrupt`: it must not require unrelated devices to reconnect.
    pub fn retire(&mut self) {
        self.interrupted = true;
        self.retired = self.last.is_some();
    }

    pub fn descriptor(&self) -> Option<&DeviceDescriptor> {
        self.last.as_ref().map(|(descriptor, _, _)| descriptor)
    }

    /// Returns a cancellation reason when the policy must reset before this
    /// frame. Sequence wrap is allowed; duplicate or backwards replay is not.
    pub fn accept(&mut self, frame: PointerFrame<'_>) -> Result<Option<CancelReason>, FrameError> {
        frame.validate()?;
        let mut cancel = self.interrupted.then_some(CancelReason::StreamLost);
        let mut timestamp = frame.timestamp;
        if let Some((old, sequence, time)) = self.last {
            if old.device != frame.descriptor.device || old.seat != frame.descriptor.seat {
                return Err(FrameError::WrongDevice);
            }
            if frame.descriptor.generation < old.generation
                || (self.retired && frame.descriptor.generation == old.generation)
            {
                return Err(FrameError::StaleGeneration);
            }
            if frame.descriptor.generation != old.generation {
                cancel = Some(CancelReason::DeviceReset);
            } else {
                if old != *frame.descriptor {
                    return Err(FrameError::DescriptorChanged);
                }
                let step = frame.sequence.wrapping_sub(sequence);
                if step == 0 || step >= 1 << 31 {
                    return Err(FrameError::StaleSequence);
                }
                if step != 1 {
                    cancel = Some(CancelReason::StreamLost);
                }
                if let (Some(old), Some(now)) = (time, frame.timestamp) {
                    if now < old {
                        return Err(FrameError::TimeReversed);
                    }
                }
                timestamp = timestamp.or(time);
            }
        }
        self.last = Some((*frame.descriptor, frame.sequence, timestamp));
        self.interrupted = false;
        self.retired = false;
        Ok(cancel)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureKind {
    Hold,
    Tap,
    Drag,
    Scroll,
    Transform,
    Swipe,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GesturePhase {
    Begin,
    Update,
    End,
    Cancel,
}

/// Coordinate space before compositor/window transforms. Unknown physical
/// calibration must stay in device coordinates, not be guessed into millimeters.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureSpace {
    DeviceCoordinates,
    Millimeters,
}

/// Continuous recognizer output, before system/widget ownership arbitration.
/// A transform can translate, scale AND rotate in the same update. Policies
/// should not flatten it into synthetic wheel events or shortcut keys.
/// `Begin` starts at the identity; updates and `End` carry totals since Begin,
/// not per-event deltas. `Cancel` terminates ownership without applying motion.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GestureEvent<'a> {
    pub device: DeviceId,
    pub generation: DeviceGeneration,
    pub interaction: u64,
    pub timestamp: Option<InputTimestamp>,
    pub kind: GestureKind,
    pub phase: GesturePhase,
    pub contacts: &'a [u32],
    pub space: GestureSpace,
    /// Initial gesture centroid in `space`, the origin of scale/rotation.
    pub origin: [f64; 2],
    /// Displacement from the initial centroid, in `space`.
    pub translation: [f64; 2],
    /// Positive size ratio relative to Begin (identity = 1).
    pub scale: f64,
    /// Radians since Begin, positive from the device's +X axis toward +Y.
    pub rotation: f64,
    /// Units of `space` per second; None without trustworthy sample timing.
    pub velocity: Option<[f64; 2]>,
}

impl<'a> GestureEvent<'a> {
    pub const fn new(
        device: DeviceId,
        generation: DeviceGeneration,
        interaction: u64,
        kind: GestureKind,
        phase: GesturePhase,
        contacts: &'a [u32],
    ) -> Self {
        Self {
            device,
            generation,
            interaction,
            timestamp: None,
            kind,
            phase,
            contacts,
            space: GestureSpace::DeviceCoordinates,
            origin: [0.0; 2],
            translation: [0.0; 2],
            scale: 1.0,
            rotation: 0.0,
            velocity: None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolicyEvent<'a> {
    Pointer(PointerInput),
    Gesture(GestureEvent<'a>),
}

/// One device's replaceable policy. Runtime-owned state may use an explicit
/// allocator or retained ax graphs. The caller validates continuity, invokes
/// cancellation before a discontinuous frame, and services `next_deadline`
/// through its existing wait/timer facilities. No recognizer owns a thread.
pub trait FramePolicy {
    fn frame(&mut self, frame: PointerFrame<'_>, emit: &mut impl FnMut(PolicyEvent<'_>));
    fn cancel(&mut self, reason: CancelReason, emit: &mut impl FnMut(PolicyEvent<'_>));
    fn next_deadline(&self) -> Option<InputTimestamp> {
        None
    }
    fn advance(&mut self, _now: InputTimestamp, _emit: &mut impl FnMut(PolicyEvent<'_>)) {}
}

/// Explicit compatibility policy, not a gesture recognizer. Calibrated pads
/// use pixels/mm; uncalibrated pads use a fixed reference stroke, independent
/// of viewport dimensions. Fractional motion is retained until cursor output.
#[derive(Debug, Clone, Copy)]
pub struct PointerFallback<M = IdentityMotion> {
    anchor: Option<(u32, i32, i32)>,
    last_input: Option<PointerInput>,
    buttons: u64,
    blocked: u64,
    resync: bool,
    remainder: [f64; 2],
    wheel_remainder: [f64; 2],
    wheel_unit: Option<ScrollUnit>,
    pixels_per_mm: f64,
    reference_stroke: f64,
    motion: M,
    calibrated_mouse: bool,
}

/// What a [`MotionPolicy`] may know about one relative motion: the physics of
/// the device that made it, and when its frame was taken. The fallback builds
/// it from the frame; a policy only reads it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionContext {
    /// The device's source physics, as its descriptor declares them.
    pub source: PointerSource,
    /// The frame's timestamp, in the runtime's monotonic nanosecond domain.
    pub timestamp: Option<InputTimestamp>,
    /// Where [`Self::timestamp`] came from.
    pub timestamp_origin: TimestampOrigin,
}

impl MotionContext {
    pub const fn new(
        source: PointerSource,
        timestamp: Option<InputTimestamp>,
        timestamp_origin: TimestampOrigin,
    ) -> Self {
        Self {
            source,
            timestamp,
            timestamp_origin,
        }
    }

    /// The frame's time, only when its origin names a clock. A timestamp of
    /// unknown origin is not one: time-dependent policies must not guess.
    pub const fn time(self) -> Option<InputTimestamp> {
        match self.timestamp_origin {
            TimestampOrigin::Unknown => None,
            TimestampOrigin::Received | TimestampOrigin::Captured => self.timestamp,
        }
    }
}

/// Optional transformation of continuous, baseline-pixel relative motion,
/// before integer compatibility output. Direct touch, buttons and scroll do
/// not pass through this hook. A policy must return finite bounded deltas.
/// `context` carries the motion's source and time, so a policy can depend
/// on speed or apply to one kind of device only.
pub trait MotionPolicy {
    fn apply(&mut self, delta: [f64; 2], context: MotionContext) -> [f64; 2];
    fn reset(&mut self);
    /// A new motion begins at `context`'s time, unrelated to the device's
    /// earlier motion: a finger landed on a touchpad. A policy that keeps a
    /// history starts it there, so the touch's first motion is measured
    /// from the landing. The default forgets it, as [`Self::reset`] does.
    fn restart(&mut self, context: MotionContext) {
        let _ = context;
        self.reset();
    }
}

/// Preserve the existing fallback's motion exactly.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityMotion;

impl MotionPolicy for IdentityMotion {
    fn apply(&mut self, delta: [f64; 2], _context: MotionContext) -> [f64; 2] {
        delta
    }
    fn reset(&mut self) {}
}

impl Default for PointerFallback {
    fn default() -> Self {
        Self::new(12.0, 1000.0).expect("constant valid gain")
    }
}

impl PointerFallback {
    pub fn new(pixels_per_mm: f64, reference_stroke: f64) -> Result<Self, FrameError> {
        Self::with_motion_policy(pixels_per_mm, reference_stroke, IdentityMotion)
    }
}

impl<M: MotionPolicy> PointerFallback<M> {
    /// Installs explicit relative-motion policy without changing default users.
    pub fn with_motion_policy(
        pixels_per_mm: f64,
        reference_stroke: f64,
        motion: M,
    ) -> Result<Self, FrameError> {
        if !pixels_per_mm.is_finite()
            || pixels_per_mm <= 0.0
            || pixels_per_mm > i32::MAX as f64
            || !reference_stroke.is_finite()
            || reference_stroke <= 0.0
            || reference_stroke > i32::MAX as f64
        {
            return Err(FrameError::InvalidMotion);
        }
        Ok(Self {
            anchor: None,
            last_input: None,
            buttons: 0,
            blocked: 0,
            resync: false,
            remainder: [0.0; 2],
            wheel_remainder: [0.0; 2],
            wheel_unit: None,
            pixels_per_mm,
            reference_stroke,
            motion,
            calibrated_mouse: false,
        })
    }
    /// Opt into descriptor-calibrated mouse counts using `pixels_per_mm`.
    /// If either axis lacks calibration, BOTH axes remain one pixel/count;
    /// never mix millimeters on one axis with counts on the other.
    pub fn with_calibrated_mouse(mut self) -> Self {
        self.calibrated_mouse = true;
        self
    }
    pub fn motion_policy(&self) -> &M {
        &self.motion
    }
    /// Reconfigure the motion policy in place, e.g. a user's speed setting.
    /// The fallback's fractional remainder is kept: what was not yet output
    /// was moved under the previous configuration.
    pub fn motion_policy_mut(&mut self) -> &mut M {
        &mut self.motion
    }
    fn unanchor(&mut self) {
        self.anchor = None;
        self.remainder = [0.0; 2];
        self.motion.reset();
    }
    fn relative(&mut self, values: [f64; 2], context: MotionContext) -> PointerInputKind {
        let values = self.motion.apply(values, context);
        if values.iter().any(|v| !v.is_finite()) {
            self.remainder = [0.0; 2];
            self.motion.reset();
            return PointerInputKind::RelativeMotion { dx: 0, dy: 0 };
        }
        PointerInputKind::RelativeMotion {
            dx: integral(values[0], &mut self.remainder[0]),
            dy: integral(values[1], &mut self.remainder[1]),
        }
    }
}

impl<M: MotionPolicy> FramePolicy for PointerFallback<M> {
    fn frame(&mut self, frame: PointerFrame<'_>, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if frame.validate().is_err() {
            self.cancel(CancelReason::MalformedFrame, emit);
            return;
        }
        if frame.completeness == FrameCompleteness::Fragment {
            self.cancel(CancelReason::StreamLost, emit);
            return;
        }
        let mut input = PointerInput {
            seat: frame.descriptor.seat,
            device: frame.descriptor.device,
            source: frame.descriptor.source,
            timestamp: frame.timestamp,
            kind: PointerInputKind::RelativeMotion { dx: 0, dy: 0 },
        };
        self.last_input = Some(input);
        let context = MotionContext::new(
            frame.descriptor.source,
            frame.timestamp,
            frame.timestamp_origin,
        );
        let mut buttons = frame.buttons;
        let usable = if frame.completeness == FrameCompleteness::Complete {
            let mut contacts = frame.contacts.iter().filter(|c| c.touching);
            let first = contacts.next();
            if contacts.next().is_none() {
                first.filter(|c| c.confidence != Some(false))
            } else {
                None
            }
        } else {
            None
        };
        match frame.descriptor.source {
            PointerSource::Touchpad => {
                if let (Some(contact), Some(axes)) = (usable, frame.descriptor.axes) {
                    let previous = self.anchor;
                    self.anchor = Some((contact.id, contact.x, contact.y));
                    if let Some((_, x, y)) = previous.filter(|p| p.0 == contact.id) {
                        let values = [
                            i64::from(contact.x) - i64::from(x),
                            i64::from(contact.y) - i64::from(y),
                        ];
                        let mut delta = [0.0; 2];
                        for axis in 0..2 {
                            let gain = axes[axis].resolution.map_or(
                                self.reference_stroke / f64::from(axes[axis].range.span()),
                                |r| {
                                    self.pixels_per_mm * f64::from(r.millimeters)
                                        / f64::from(r.units)
                                },
                            );
                            delta[axis] = values[axis] as f64 * gain;
                        }
                        input.kind = self.relative(delta, context);
                        emit(PolicyEvent::Pointer(input));
                    } else {
                        // A finger landed (or another took over): its motion
                        // starts here, unrelated to the last finger's.
                        self.remainder = [0.0; 2];
                        self.motion.restart(context);
                    }
                } else {
                    self.unanchor();
                }
            }
            PointerSource::Touchscreen | PointerSource::Pen => {
                if let (Some(contact), Some(axes)) = (usable, frame.descriptor.axes) {
                    // Direct-touch click emulation is confined to this fallback;
                    // a real recognizer receives the raw frame before any click.
                    buttons |= 1;
                    input.kind = PointerInputKind::AbsoluteMotion {
                        x: (i64::from(contact.x) - i64::from(axes[0].range.min())) as u32,
                        y: (i64::from(contact.y) - i64::from(axes[1].range.min())) as u32,
                        range: AbsoluteRange {
                            max_x: axes[0].range.span(),
                            max_y: axes[1].range.span(),
                        },
                    };
                    emit(PolicyEvent::Pointer(input));
                }
            }
            _ => {
                let mut delta = frame.relative.map(f64::from);
                if self.calibrated_mouse
                    && let [Some(x), Some(y)] = frame.descriptor.relative_resolution
                {
                    for (axis, resolution) in [x, y].iter().enumerate() {
                        delta[axis] *= self.pixels_per_mm * f64::from(resolution.millimeters)
                            / f64::from(resolution.units);
                    }
                }
                input.kind = self.relative(delta, context);
                emit(PolicyEvent::Pointer(input));
            }
        }
        if self.resync {
            self.blocked = buttons;
            self.resync = false;
        }
        self.blocked &= buttons;
        let visible = buttons & !self.blocked;
        for bit in 0..64 {
            if (visible ^ self.buttons) & (1 << bit) != 0 {
                input.kind = PointerInputKind::Button {
                    button: PointerButton(bit),
                    state: if visible & (1 << bit) != 0 {
                        ButtonState::Pressed
                    } else {
                        ButtonState::Released
                    },
                };
                emit(PolicyEvent::Pointer(input));
            }
        }
        self.buttons = visible;
        if let Some(scroll) = frame.scroll {
            if self.wheel_unit != Some(scroll.unit) {
                self.wheel_remainder = [0.0; 2];
                self.wheel_unit = Some(scroll.unit);
            }
            let horizontal = integral(scroll.horizontal, &mut self.wheel_remainder[0]);
            let vertical = integral(scroll.vertical, &mut self.wheel_remainder[1]);
            if horizontal != 0 || vertical != 0 {
                input.kind = PointerInputKind::Scroll {
                    horizontal,
                    vertical,
                    unit: scroll.unit,
                };
                emit(PolicyEvent::Pointer(input));
            }
        }
    }

    fn cancel(&mut self, _reason: CancelReason, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if let Some(mut input) = self.last_input {
            input.timestamp = None;
            for bit in 0..64 {
                if self.buttons & (1 << bit) != 0 {
                    input.kind = PointerInputKind::Button {
                        button: PointerButton(bit),
                        state: ButtonState::Released,
                    };
                    emit(PolicyEvent::Pointer(input));
                }
            }
        }
        self.buttons = 0;
        self.blocked = 0;
        self.resync = true;
        self.unanchor();
        self.wheel_remainder = [0.0; 2];
        self.wheel_unit = None;
    }
}

fn integral(value: f64, remainder: &mut f64) -> i32 {
    let sum = value + *remainder;
    let result = sum as i32;
    *remainder = if sum >= i32::MAX as f64 || sum <= i32::MIN as f64 {
        0.0
    } else {
        sum - f64::from(result)
    };
    result
}
