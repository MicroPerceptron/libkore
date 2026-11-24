//! Touchpad gestures: kore's frame policy for precision touchpads.
//!
//! axinput's [`PointerFallback`] reads a touchpad as one moving finger: two
//! fingers do nothing and a tap is not a click. This policy reads the whole
//! contact set of each frame instead and recognizes what a precision touchpad
//! is used for:
//!
//! - **One finger** moves the pointer, at the fallback's gain on the
//!   acceleration plateau. A fast finger is accelerated on libinput's
//!   touchpad profile (axinput's [`Acceleration`]), the same per-device
//!   policy and preference a mouse's motion takes in the fallback.
//! - **Two fingers** scroll, on both axes, in pixels that follow the fingers
//!   ([`TouchpadConfig::natural_scroll`]). A scroll is held to its dominant
//!   axis until the other axis clearly moves, and a flick coasts on with
//!   inertia until it decays or a finger lands. Lifting the two fingers one
//!   at a time does not move the pointer, and keeps the flick when the second
//!   follows within [`FLING_GRACE_NS`].
//! - **A tap clicks**: one finger the primary button, two the secondary, three
//!   the middle. A one-finger tap's press is held through [`DRAG_WINDOW_NS`]:
//!   a touch that follows and moves (or holds) drags with that same press
//!   until it lifts.
//! - **A physical click counts the fingers** on the pad: two make it a
//!   secondary click, three a middle one, and the button stays the one it
//!   pressed until the pad releases.
//! - **Palms and thumbs.** A contact the firmware marks as not confident is
//!   ignored from then on, for the rest of its life, and leaves the touch it
//!   had joined, so it never taps. A thumb resting in the pad's bottom band is
//!   ignored while another finger works, so it neither turns a tap into a
//!   two-finger tap nor a click into a secondary click.
//! - **Transforms and swipes.** Two fingers spreading, pinching or turning,
//!   and three or more moving together, are recognized as axinput's
//!   continuous [`GestureKind::Transform`] and [`GestureKind::Swipe`] instead
//!   of being misread as scrolling. They leave this policy through the
//!   gesture contract, not as synthetic wheel events: `Begin` is the moment
//!   the gesture is recognized, and every total (translation, scale,
//!   rotation) is measured from there, so content that follows a pinch never
//!   jumps by the travel it took to recognize it. Updates and `End` carry the
//!   centroid's velocity when the frames are timed.
//!
//! Every other source (mouse, touchscreen, pen) keeps axinput's fallback
//! unchanged.
//!
//! **Units.** Contacts are converted to millimetres when the descriptor
//! carries a physical resolution on both axes, and otherwise to the
//! fallback's reference stroke expressed at the same pixels-per-millimetre
//! gain, so every threshold below means the same finger travel on either kind
//! of pad. Pointer motion and scroll leave in baseline pixels (one display
//! density step); the runtime applies the display's density to both.
//!
//! **Time.** Taps, drags and inertia need the frames' monotonic timestamps;
//! the deadlines they ask for ([`FramePolicy::next_deadline`]) are in the same
//! clock. A frame without a timestamp never fabricates one: on such a device
//! taps, drags and inertia are off, and motion, scrolling, clicks, thumbs and
//! gestures still work. Fingers already down when a discontinuity resets the
//! policy landed at no time it knows, so that touch never taps.
//!
//! **Hybrid reports.** A hybrid-mode touchpad splits one scan across several
//! reports: the first carries the scan's contact count, the rest carry zero.
//! Fragments are assembled by their shared scan time into one complete frame;
//! a scan that cannot be assembled exactly is dropped whole, never completed
//! with invented contacts. One shortfall is not a loss: a continuation's
//! lifting contacts reach this policy as none (the transport cannot tell a
//! zero-count report's lifted slots from an empty report's stale ones), and
//! one carrying nothing else is marked complete. A scan that has had a
//! continuation therefore ends short, its undelivered contacts lifted, when
//! the next scan begins or [`HYBRID_SETTLE_NS`] after its last report.
//!
//! The policy allocates nothing and holds a bounded contact set
//! ([`MAX_TRACKS`]); contacts beyond it are ignored, not truncated mid-gesture.

use axinput::acceleration::{Acceleration, AccelerationConfig, Calibration};
use axinput::frame::{
    CancelReason, ContactSample, DeviceDescriptor, DeviceGeneration, FrameCompleteness, FrameError,
    FramePolicy, GestureEvent, GestureKind, GesturePhase, GestureSpace, MotionContext,
    MotionPolicy, PointerFallback, PointerFrame, PolicyEvent, TimestampOrigin,
};
use axinput::{
    ButtonState, InputTimestamp, PointerButton, PointerInput, PointerInputKind, PointerSource,
    ScrollUnit,
};
// Called by path: under a test harness `std`'s inherent methods would shadow
// the trait's and leave a `use` of it unused.
use core_maths::CoreFloat;

const MS: u64 = 1_000_000;

/// Contacts tracked at once: the most a precision touchpad reports (Windows
/// requires five and admits ten).
pub const MAX_TRACKS: usize = 10;

/// A tap lasts at most this long, first finger down to last finger up.
pub const TAP_TIMEOUT_NS: u64 = 200 * MS;
/// A tapping finger stays within this distance of where it landed.
pub const TAP_SLOP_MM: f64 = 1.5;
/// A touch this soon after a one-finger tap can become a drag.
pub const DRAG_WINDOW_NS: u64 = 250 * MS;
/// Two fingers' centroid travels this far before they scroll.
pub const SCROLL_START_MM: f64 = 2.0;
/// Two fingers' spread changes this much, or each turns this far around
/// their centroid, and further than they travel, before they transform
/// instead of scroll.
pub const TRANSFORM_START_MM: f64 = 4.0;
const TRANSFORM_RATIO: f64 = 1.2;
/// Three or more fingers' centroid travels this far before they swipe.
pub const SWIPE_START_MM: f64 = 4.0;
/// A scroll keeps to its dominant axis until the other one travels this far.
pub const AXIS_UNLOCK_MM: f64 = 8.0;
const AXIS_LOCK_RATIO: f64 = 2.0;
/// The bottom band a resting thumb is recognized in, capped at
/// `THUMB_BAND_FRACTION` of the pad's height.
pub const THUMB_BAND_MM: f64 = 10.0;
const THUMB_BAND_FRACTION: f64 = 0.15;
/// A thumb that travels this far beside a working finger is a finger again.
pub const THUMB_RELEASE_MM: f64 = 8.0;
/// After two scrolling fingers become one, the flick survives this long for
/// the last finger to lift (the remaining finger does not point until it
/// clearly moves, however long it stays).
pub const FLING_GRACE_NS: u64 = 60 * MS;
/// Scroll velocity is measured over this much of the gesture's end.
const FLING_WINDOW_NS: u64 = 80 * MS;
/// A scroll that paused this long before lifting does not coast.
const FLING_PAUSE_NS: u64 = 40 * MS;
/// Inertia starts above this finger speed and stops below the second.
pub const INERTIA_START_MM_S: f64 = 50.0;
pub const INERTIA_STOP_MM_S: f64 = 5.0;
/// Inertia never starts faster than this.
const INERTIA_MAX_MM_S: f64 = 1500.0;
/// Inertia decays as `exp(-t / tau)`.
pub const INERTIA_TAU_NS: u64 = 325 * MS;
/// Inertia emits a step at most this often.
pub const INERTIA_STEP_NS: u64 = 8_333_333;

/// A hybrid scan whose lifted contacts went unreported is complete this
/// long after its last report, when no next scan has settled it sooner.
/// Reports of one scan arrive back to back; this also outlasts the I2C
/// relay's 16 ms polling backstop (`GPIO_BACKSTOP_MS`), so a scan whose
/// reports one poll splits across two passes still assembles whole. It
/// delays only a lift that ends the touch, and only on a hybrid pad.
pub const HYBRID_SETTLE_NS: u64 = 20 * MS;

const VELOCITY_SAMPLES: usize = 8;

/// Relative error below which a pixel sum is float noise around a whole one.
const SNAP: f64 = 1e-9;

/// User-facing touchpad behaviour. The gains match axinput's fallback, so a
/// finger moves the pointer exactly as far under either policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TouchpadConfig {
    /// Baseline pixels per millimetre of finger travel, for the pointer and
    /// for scrolling.
    pub pixels_per_mm: f64,
    /// Baseline pixels a full axis stroke spans on an uncalibrated pad.
    pub reference_stroke: f64,
    /// Content follows the fingers (the platform default on precision
    /// touchpads); `false` scrolls the way a wheel does.
    pub natural_scroll: bool,
    pub tap_to_click: bool,
    /// A touch that follows a one-finger tap drags.
    pub tap_drag: bool,
    /// A flicked scroll coasts on.
    pub inertia: bool,
}

impl Default for TouchpadConfig {
    fn default() -> Self {
        Self {
            pixels_per_mm: 12.0,
            reference_stroke: 1000.0,
            natural_scroll: true,
            tap_to_click: true,
            tap_drag: true,
            inertia: true,
        }
    }
}

/// The frame policy a seat runs for every pointer device: gestures for a
/// touchpad, axinput's fallback for everything else. One [`Acceleration`]
/// per device accelerates its relative motion on libinput's profiles: a
/// mouse's in the fallback, a touchpad finger's where the recognizer points
/// with it. A virtual source's, a host's emulated mouse, is not accelerated.
#[derive(Clone)]
pub struct TouchpadPolicy {
    /// Owns the device's [`Acceleration`]: the fallback's motion policy, lent
    /// to the recognizer for a touchpad.
    fallback: PointerFallback<Acceleration>,
    pad: Pad,
}

impl TouchpadPolicy {
    pub fn new(config: TouchpadConfig) -> Result<Self, FrameError> {
        // A finger's travel reaches the acceleration in baseline pixels at
        // the pad's gain, which is the touchpad profile's plateau.
        let calibration = Calibration::DEFAULT
            .with_pixels_per_mm(config.pixels_per_mm)
            .map_err(|_| FrameError::InvalidMotion)?;
        Ok(Self {
            fallback: PointerFallback::with_motion_policy(
                config.pixels_per_mm,
                config.reference_stroke,
                Acceleration::default().with_calibration(calibration),
            )?,
            pad: Pad::new(config),
        })
    }

    /// The device's acceleration, from the user's pointer preference: a
    /// mouse's and a touchpad finger's alike. Applies from this device's next
    /// report; its calibration and motion history are kept.
    pub fn set_acceleration(&mut self, config: AccelerationConfig) {
        self.fallback.motion_policy_mut().set_config(config);
    }

    pub fn acceleration(&self) -> AccelerationConfig {
        self.fallback.motion_policy().config()
    }

    /// Hybrid scans dropped because they could not be assembled exactly.
    pub fn dropped_scans(&self) -> u64 {
        self.pad.hybrid.dropped
    }
}

impl Default for TouchpadPolicy {
    fn default() -> Self {
        Self::new(TouchpadConfig::default()).expect("the default gains are valid")
    }
}

impl FramePolicy for TouchpadPolicy {
    fn frame(&mut self, frame: PointerFrame<'_>, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if frame.descriptor.source == PointerSource::Touchpad {
            self.pad
                .frame(frame, self.fallback.motion_policy_mut(), emit);
        } else {
            self.fallback.frame(frame, emit);
        }
    }

    fn cancel(&mut self, reason: CancelReason, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        self.fallback.cancel(reason, emit);
        self.pad.cancel(emit);
    }

    fn next_deadline(&self) -> Option<InputTimestamp> {
        self.pad.next_deadline()
    }

    fn advance(&mut self, now: InputTimestamp, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        self.pad
            .advance(now.0, self.fallback.motion_policy_mut(), emit);
    }
}

/// One device lifetime's units: contacts stay in device units, thresholds
/// are judged in millimetres, and output leaves at the fallback's exact
/// pixels-per-unit gain (so a stroke moves the pointer to the pixel the
/// fallback would).
#[derive(Clone, Copy, Debug)]
struct Geometry {
    mm: [f64; 2],
    px: [f64; 2],
    min: [i32; 2],
    /// The thumb band's top edge, in millimetres from the pad's top.
    band_top: f64,
    calibrated: bool,
}

impl Geometry {
    fn of(descriptor: &DeviceDescriptor, config: &TouchpadConfig) -> Option<Self> {
        let axes = descriptor.axes?;
        // Both axes calibrated, or neither: a gesture measured in millimetres
        // on one axis and in reference strokes on the other would misjudge
        // every diagonal.
        let calibrated = axes.iter().all(|axis| axis.resolution.is_some());
        let (mut mm, mut px) = ([0.0; 2], [0.0; 2]);
        for axis in 0..2 {
            match (calibrated, axes[axis].resolution) {
                (true, Some(r)) => {
                    mm[axis] = f64::from(r.millimeters()) / f64::from(r.units());
                    px[axis] =
                        config.pixels_per_mm * f64::from(r.millimeters()) / f64::from(r.units());
                }
                _ => {
                    px[axis] = config.reference_stroke / f64::from(axes[axis].range.span());
                    mm[axis] = px[axis] / config.pixels_per_mm;
                }
            }
        }
        let height = f64::from(axes[1].range.span()) * mm[1];
        Some(Self {
            mm,
            px,
            min: [axes[0].range.min(), axes[1].range.min()],
            band_top: height - THUMB_BAND_MM.min(height * THUMB_BAND_FRACTION),
            calibrated,
        })
    }

    fn units(self, contact: &ContactSample) -> [f64; 2] {
        [
            (i64::from(contact.x) - i64::from(self.min[0])) as f64,
            (i64::from(contact.y) - i64::from(self.min[1])) as f64,
        ]
    }

    fn to_mm(self, v: [f64; 2]) -> [f64; 2] {
        [v[0] * self.mm[0], v[1] * self.mm[1]]
    }

    fn to_px(self, v: [f64; 2]) -> [f64; 2] {
        [v[0] * self.px[0], v[1] * self.px[1]]
    }

    /// Millimetres between two positions in device units.
    fn distance(self, a: [f64; 2], b: [f64; 2]) -> f64 {
        norm(self.to_mm([a[0] - b[0], a[1] - b[1]]))
    }

    fn in_band(self, at: [f64; 2]) -> bool {
        at[1] * self.mm[1] >= self.band_top
    }

    /// Gesture output space: millimetres on a calibrated pad, device units
    /// otherwise (never a guess at physical size).
    fn space(self) -> GestureSpace {
        if self.calibrated {
            GestureSpace::Millimeters
        } else {
            GestureSpace::DeviceCoordinates
        }
    }

    /// A displacement in millimetres, in the output space.
    fn to_space(self, value: [f64; 2]) -> [f64; 2] {
        if self.calibrated {
            value
        } else {
            [value[0] / self.mm[0], value[1] / self.mm[1]]
        }
    }

    /// A position in millimetres from the pad's origin, in the output
    /// space: millimetres from that origin, or the device's own coordinates.
    fn point_to_space(self, at: [f64; 2]) -> [f64; 2] {
        if self.calibrated {
            at
        } else {
            let [x, y] = self.to_space(at);
            [x + f64::from(self.min[0]), y + f64::from(self.min[1])]
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Track {
    id: u32,
    /// Device units, as are `start`, `rest` and `delta`.
    at: [f64; 2],
    start: [f64; 2],
    /// Where it was when it became a thumb, to tell when it moves again.
    rest: [f64; 2],
    delta: [f64; 2],
    landed: Option<u64>,
    palm: bool,
    thumb: bool,
    moved: bool,
    seen: bool,
}

impl Track {
    const EMPTY: Self = Self {
        id: 0,
        at: [0.0; 2],
        start: [0.0; 2],
        rest: [0.0; 2],
        delta: [0.0; 2],
        landed: None,
        palm: false,
        thumb: false,
        moved: false,
        seen: false,
    };

    /// A finger that counts: not a palm, not a resting thumb.
    fn counts(&self) -> bool {
        !self.palm && !self.thumb
    }
}

/// One touch sequence, first counted finger down to last counted finger up.
#[derive(Clone, Copy, Debug)]
struct Session {
    /// The counted fingers that took part and when each landed.
    fingers: [(u32, Option<u64>); 5],
    count: usize,
    moved: bool,
    clicked: bool,
    gestured: bool,
    /// Began inside the drag window after a one-finger tap.
    after_tap: bool,
}

impl Session {
    const IDLE: Self = Self {
        fingers: [(0, None); 5],
        count: 0,
        moved: false,
        clicked: false,
        gestured: false,
        after_tap: false,
    };

    fn active(&self) -> bool {
        self.count != 0
    }

    fn join(&mut self, id: u32, landed: Option<u64>) {
        if self.fingers[..self.count].iter().any(|f| f.0 == id) {
            return;
        }
        if self.count < self.fingers.len() {
            self.fingers[self.count] = (id, landed);
            self.count += 1;
        } else {
            // More fingers than any tap: the touch is not a tap.
            self.moved = true;
        }
    }

    /// A finger reclassified as a resting thumb never took part.
    fn leave(&mut self, id: u32) {
        if let Some(index) = self.fingers[..self.count].iter().position(|f| f.0 == id) {
            self.fingers.copy_within(index + 1..self.count, index);
            self.count -= 1;
        }
    }

    fn began(&self) -> Option<u64> {
        self.fingers[..self.count]
            .iter()
            .map(|f| f.1)
            .try_fold(u64::MAX, |earliest, landed| landed.map(|t| earliest.min(t)))
            .filter(|&t| t != u64::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lock {
    Vertical,
    Horizontal,
    Free,
}

/// What the counted fingers are doing.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Idle,
    /// One finger (or the finger that moves during a physical press) drives
    /// the pointer.
    Pointer,
    /// Two or more fingers that have not yet shown what they are doing.
    Undecided,
    Scroll(Lock),
    /// The finger a multi-finger touch left behind, held where it was
    /// (`anchor`) until it clearly moves; and, when the touch was a scroll,
    /// the flick it carried and when its first finger lifted.
    Lifting {
        velocity: [f64; 2],
        since: Option<u64>,
        anchor: [f64; 2],
    },
    Transform,
    Swipe,
}

/// The finger set a multi-finger gesture is measured against.
#[derive(Clone, Copy, Debug)]
struct Baseline {
    ids: [u32; 5],
    count: usize,
    centroid: [f64; 2],
    spread: f64,
    angle: f64,
    start: [[f64; 2]; 5],
    /// When the set was measured from: a finger that points from `start`
    /// moved over the time since.
    since: Option<u64>,
    /// Transform rotation accumulated across frames (unwrapped) and the
    /// angle it was last measured at.
    rotation: f64,
    last_angle: f64,
    interaction: u64,
}

impl Baseline {
    const EMPTY: Self = Self {
        ids: [0; 5],
        count: 0,
        centroid: [0.0; 2],
        spread: 0.0,
        angle: 0.0,
        start: [[0.0; 2]; 5],
        since: None,
        rotation: 0.0,
        last_angle: 0.0,
        interaction: 0,
    };
}

/// Travel a finger made before it drove the pointer, owed to it when it
/// starts to, and when that travel began.
#[derive(Clone, Copy, Debug)]
struct Owed {
    /// Device units.
    travel: [f64; 2],
    since: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct Inertia {
    /// Output pixels per second, horizontal then vertical.
    velocity: [f64; 2],
    last: u64,
}

/// Hybrid-mode fragments of one scan, assembled into one complete frame.
#[derive(Clone)]
struct Hybrid {
    active: bool,
    scan: u32,
    expected: usize,
    contacts: [ContactSample; MAX_TRACKS],
    count: usize,
    buttons: u64,
    /// A continuation arrived: this scan may end short, what it never
    /// delivered lifted.
    short: bool,
    /// The last report's sequence and receive time, for the frame the scan
    /// becomes.
    sequence: u32,
    timestamp: Option<InputTimestamp>,
    origin: TimestampOrigin,
    dropped: u64,
}

impl Hybrid {
    fn new() -> Self {
        Self {
            active: false,
            scan: 0,
            expected: 0,
            contacts: [ContactSample::new(0, 0, 0, false); MAX_TRACKS],
            count: 0,
            buttons: 0,
            short: false,
            sequence: 0,
            timestamp: None,
            origin: TimestampOrigin::Unknown,
            dropped: 0,
        }
    }

    fn drop_frame(&mut self) {
        self.dropped = self.dropped.saturating_add(1);
    }

    /// Whether `frame` is a later report of the scan being assembled.
    fn continues(&self, frame: &PointerFrame<'_>) -> bool {
        self.active && frame.scan_time == Some(self.scan) && frame.reported_contacts == Some(0)
    }

    fn note(&mut self, frame: &PointerFrame<'_>) {
        self.sequence = frame.sequence;
        self.timestamp = frame.timestamp;
        self.origin = frame.timestamp_origin;
    }

    /// Start a scan from its first report. False when it cannot be one.
    fn begin(&mut self, frame: &PointerFrame<'_>) -> bool {
        let (Some(scan), Some(expected)) = (frame.scan_time, frame.reported_contacts) else {
            // Without a scan time fragments cannot be matched to a scan, and
            // a report counting zero continues no scan this policy saw begin.
            self.drop_frame();
            return false;
        };
        if expected == 0 || usize::from(expected) > MAX_TRACKS {
            self.drop_frame();
            return false;
        }
        self.active = true;
        self.scan = scan;
        self.expected = usize::from(expected);
        self.count = 0;
        self.short = false;
        self.buttons = frame.buttons;
        self.fold(frame)
    }

    /// Fold a report's contacts in. False when they cannot belong to this
    /// scan (a repeated contact, or more than it counted): it is dropped.
    fn fold(&mut self, frame: &PointerFrame<'_>) -> bool {
        self.note(frame);
        for contact in frame.contacts {
            if self.contacts[..self.count]
                .iter()
                .any(|c| c.id == contact.id)
                || self.count == self.expected
            {
                self.active = false;
                self.drop_frame();
                return false;
            }
            self.contacts[self.count] = *contact;
            self.count += 1;
        }
        true
    }

    fn complete(&self) -> bool {
        self.active && self.count == self.expected
    }

    /// Complete, or short only by lifted contacts (checked once inactive).
    fn complete_or_short(&self) -> bool {
        self.count == self.expected || self.short
    }

    /// When a short scan is complete without a next one.
    fn settle_at(&self) -> Option<u64> {
        (self.active && self.short)
            .then(|| self.timestamp.map(|t| t.0.saturating_add(HYBRID_SETTLE_NS)))
            .flatten()
    }
}

#[derive(Clone)]
struct Pad {
    config: TouchpadConfig,
    tracks: [Track; MAX_TRACKS],
    tracked: usize,
    mode: Mode,
    driver: Option<u32>,
    baseline: Baseline,
    lock_travel: [f64; 2],
    session: Session,
    /// When a one-finger tap ended whose press is still held, to become a
    /// drag if a touch follows within the drag window.
    tapped: Option<u64>,
    /// The tap's press is a drag now; lifting releases it.
    dragging: bool,
    /// Counted fingers in the previous frame.
    fingers: usize,
    /// Buttons this policy holds pressed.
    held: u64,
    /// The button the pad's physical click pressed (clickfinger).
    click: Option<u16>,
    physical: u64,
    blocked: u64,
    resync: bool,
    motion_remainder: [f64; 2],
    scroll_remainder: [f64; 2],
    /// Cumulative scroll output and when, for the flick's velocity.
    samples: [(u64, [f64; 2]); VELOCITY_SAMPLES],
    sampled: usize,
    scrolled: [f64; 2],
    inertia: Option<Inertia>,
    template: Option<PointerInput>,
    /// The frame being processed, as pointer motion's acceleration reads it.
    context: MotionContext,
    generation: DeviceGeneration,
    geometry: Option<Geometry>,
    interactions: u64,
    /// The device as its last report described it, for a hybrid scan
    /// completed by time alone.
    descriptor: Option<DeviceDescriptor>,
    /// The last totals a transform or swipe reported, repeated by its End.
    last_translation: [f64; 2],
    last_scale: f64,
    hybrid: Hybrid,
}

impl Pad {
    fn new(config: TouchpadConfig) -> Self {
        Self {
            config,
            tracks: [Track::EMPTY; MAX_TRACKS],
            tracked: 0,
            mode: Mode::Idle,
            driver: None,
            baseline: Baseline::EMPTY,
            lock_travel: [0.0; 2],
            session: Session::IDLE,
            tapped: None,
            dragging: false,
            fingers: 0,
            held: 0,
            click: None,
            physical: 0,
            blocked: 0,
            resync: false,
            motion_remainder: [0.0; 2],
            scroll_remainder: [0.0; 2],
            samples: [(0, [0.0; 2]); VELOCITY_SAMPLES],
            sampled: 0,
            scrolled: [0.0; 2],
            inertia: None,
            template: None,
            context: MotionContext::new(PointerSource::Touchpad, None, TimestampOrigin::Unknown),
            generation: DeviceGeneration(0),
            geometry: None,
            interactions: 0,
            descriptor: None,
            last_translation: [0.0; 2],
            last_scale: 1.0,
            hybrid: Hybrid::new(),
        }
    }

    fn frame(
        &mut self,
        frame: PointerFrame<'_>,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        if frame.validate().is_err() {
            self.cancel(emit);
            return;
        }
        self.descriptor = Some(*frame.descriptor);
        if self.hybrid.continues(&frame) {
            // The transport drops a continuation's lifting contacts, so from
            // here the scan may end short: what it never delivers lifted.
            self.hybrid.short = true;
            if self.hybrid.fold(&frame) && self.hybrid.complete() {
                self.settle(accel, emit);
            }
            return;
        }
        // Anything else is a new scan: one still assembling ends here.
        self.settle(accel, emit);
        match frame.completeness {
            FrameCompleteness::Complete => self.process(frame, accel, emit),
            FrameCompleteness::Fragment => {
                if self.hybrid.begin(&frame) && self.hybrid.complete() {
                    self.settle(accel, emit);
                }
            }
        }
    }

    /// End the scan being assembled: as a frame when it is complete, or
    /// short only by contacts that lifted; dropped otherwise.
    fn settle(&mut self, accel: &mut Acceleration, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        let hybrid = &mut self.hybrid;
        if !hybrid.active {
            return;
        }
        hybrid.active = false;
        let (Some(descriptor), true) = (self.descriptor, hybrid.complete_or_short()) else {
            hybrid.drop_frame();
            return;
        };
        let mut contacts = [ContactSample::new(0, 0, 0, false); MAX_TRACKS];
        let count = hybrid.count;
        contacts[..count].copy_from_slice(&hybrid.contacts[..count]);
        let mut whole = PointerFrame::new(&descriptor, hybrid.sequence, &contacts[..count]);
        whole.timestamp = hybrid.timestamp;
        whole.timestamp_origin = hybrid.origin;
        whole.scan_time = Some(hybrid.scan);
        whole.reported_contacts = u16::try_from(count).ok();
        whole.buttons = hybrid.buttons;
        if whole.validate().is_err() {
            self.hybrid.drop_frame();
            return;
        }
        self.process(whole, accel, emit);
    }

    fn process(
        &mut self,
        frame: PointerFrame<'_>,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let Some(geometry) = Geometry::of(frame.descriptor, &self.config) else {
            self.cancel(emit);
            return;
        };
        self.geometry = Some(geometry);
        self.generation = frame.descriptor.generation;
        let now = match frame.timestamp_origin {
            TimestampOrigin::Unknown => None,
            _ => frame.timestamp.map(|t| t.0),
        };
        self.template = Some(PointerInput {
            seat: frame.descriptor.seat,
            device: frame.descriptor.device,
            source: frame.descriptor.source,
            timestamp: frame.timestamp,
            kind: PointerInputKind::RelativeMotion { dx: 0, dy: 0 },
        });
        self.context = MotionContext::new(
            frame.descriptor.source,
            frame.timestamp,
            frame.timestamp_origin,
        );

        if self.update_tracks(frame.contacts, &geometry, now) {
            // A finger landing catches a coasting scroll (a thumb already
            // resting there does not).
            self.inertia = None;
        }
        // Session bookkeeping over counted fingers.
        let mut counted = [0usize; MAX_TRACKS];
        let n = self.counted(&mut counted);
        if n != 0 && !self.session.active() {
            // A tap whose window passed unserviced is over before this touch.
            if self
                .tapped
                .zip(now)
                .is_none_or(|(end, now)| now.saturating_sub(end) > DRAG_WINDOW_NS)
            {
                self.release_tap(emit);
            }
            self.session = Session::IDLE;
            self.session.after_tap = self.tapped.is_some();
        }
        for &index in &counted[..n] {
            let track = self.tracks[index];
            self.session.join(track.id, track.landed);
            self.session.moved |= track.moved;
        }
        if self.session.after_tap && !self.dragging && self.session.count >= 2 {
            // More fingers than a drag: the tap was only a click.
            self.release_tap(emit);
            self.session.after_tap = false;
        }

        self.buttons(frame.buttons, n, emit);
        self.motion(&counted[..n], &geometry, now, accel, emit);
        self.drag_check(now, emit);

        if n == 0 && self.session.active() {
            self.finish_session(now, emit);
        }
    }

    fn tracks(&self) -> &[Track] {
        &self.tracks[..self.tracked]
    }

    /// Fold this frame's contacts into the tracks. Returns whether a contact
    /// that is not a palm landed.
    fn update_tracks(
        &mut self,
        contacts: &[ContactSample],
        geometry: &Geometry,
        now: Option<u64>,
    ) -> bool {
        let mut landed = false;
        // Contacts the firmware flags as palms only now: they leave the touch
        // they joined, so a palm never taps or clicks however it began.
        let mut palmed = [0u32; MAX_TRACKS];
        let mut palms = 0;
        for track in &mut self.tracks[..self.tracked] {
            track.seen = false;
            track.delta = [0.0; 2];
        }
        for contact in contacts.iter().filter(|c| c.touching) {
            let at = geometry.units(contact);
            let palm = contact.confidence == Some(false);
            if let Some(track) = self.tracks[..self.tracked]
                .iter_mut()
                .find(|t| t.id == contact.id)
            {
                track.delta = [at[0] - track.at[0], at[1] - track.at[1]];
                track.at = at;
                track.seen = true;
                if palm && !track.palm {
                    palmed[palms] = track.id;
                    palms += 1;
                }
                track.palm |= palm;
                if geometry.distance(at, track.start) > TAP_SLOP_MM {
                    track.moved = true;
                }
            } else if self.tracked < MAX_TRACKS {
                landed |= !palm;
                self.tracks[self.tracked] = Track {
                    id: contact.id,
                    at,
                    start: at,
                    rest: at,
                    delta: [0.0; 2],
                    // Fingers already down when a discontinuity reset this
                    // policy landed at no time it knows: they never tap.
                    landed: if self.resync { None } else { now },
                    palm,
                    thumb: false,
                    moved: false,
                    seen: true,
                };
                self.tracked += 1;
            }
        }
        for &id in &palmed[..palms] {
            self.session.leave(id);
            if self.driver == Some(id) {
                self.driver = None;
            }
        }
        // Lifted contacts leave; the rest keep their order.
        let mut kept = 0;
        for index in 0..self.tracked {
            if self.tracks[index].seen {
                self.tracks[kept] = self.tracks[index];
                kept += 1;
            } else if self.driver == Some(self.tracks[index].id) {
                self.driver = None;
            }
        }
        self.tracked = kept;
        // A resting thumb: a contact that has not moved from where it landed
        // in the bottom band while a finger works above it. A lone contact
        // there is a finger like any other. A thumb is a finger again once
        // it clearly travels: far beside a working finger, past the tap slop
        // alone.
        let working = self
            .tracks()
            .iter()
            .any(|t| !t.palm && !t.thumb && !geometry.in_band(t.at));
        for index in 0..self.tracked {
            let track = &mut self.tracks[index];
            if track.palm {
                continue;
            }
            if track.thumb {
                let release = if working {
                    THUMB_RELEASE_MM
                } else {
                    TAP_SLOP_MM
                };
                if geometry.distance(track.at, track.rest) > release {
                    track.thumb = false;
                }
            } else if working && !track.moved && geometry.in_band(track.at) {
                track.thumb = true;
                track.rest = track.at;
                let id = track.id;
                self.session.leave(id);
                if self.driver == Some(id) {
                    self.driver = None;
                }
            }
        }
        landed
    }

    /// The counted fingers' track indices, in landing order.
    fn counted(&self, out: &mut [usize; MAX_TRACKS]) -> usize {
        let mut n = 0;
        for (index, track) in self.tracks().iter().enumerate() {
            if track.counts() {
                out[n] = index;
                n += 1;
            }
        }
        n
    }

    fn buttons(&mut self, raw: u64, fingers: usize, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if self.resync {
            // After a cancel, a button already down must be seen up before it
            // can press again: a lost packet never becomes a second click.
            self.blocked = raw;
            self.resync = false;
        }
        self.blocked &= raw;
        let visible = raw & !self.blocked;
        let changed = visible ^ self.physical;
        self.physical = visible;
        if changed & 1 != 0 {
            if visible & 1 != 0 {
                // The pad's own button: what it means depends on how many
                // fingers are on the pad when it goes down.
                let button = match fingers {
                    0 | 1 => 0,
                    2 => 1,
                    _ => 2,
                };
                self.session.clicked = true;
                self.inertia = None;
                // A real click ends a tap's held press or drag first: it is
                // its own press, never folded into theirs.
                self.release_tap(emit);
                if core::mem::take(&mut self.dragging) {
                    self.release(0, emit);
                }
                self.click = Some(button);
                self.press(button, emit);
            } else if let Some(button) = self.click.take() {
                self.release(button, emit);
            }
        }
        // Discrete buttons beside the pad (a pointing stick's) mean what
        // they say.
        for bit in 1..64u16 {
            let mask = 1u64 << bit;
            if changed & mask != 0 {
                if visible & mask != 0 {
                    self.press(bit, emit);
                } else {
                    self.release(bit, emit);
                }
            }
        }
    }

    fn press(&mut self, button: u16, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        let mask = 1u64 << button;
        if self.held & mask == 0 {
            self.held |= mask;
            self.button(button, ButtonState::Pressed, emit);
        }
    }

    fn release(&mut self, button: u16, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        let mask = 1u64 << button;
        if self.held & mask != 0 {
            self.held &= !mask;
            self.button(button, ButtonState::Released, emit);
        }
    }

    fn button(&self, button: u16, state: ButtonState, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if let Some(mut input) = self.template {
            input.kind = PointerInputKind::Button {
                button: PointerButton(button),
                state,
            };
            emit(PolicyEvent::Pointer(input));
        }
    }

    fn motion(
        &mut self,
        counted: &[usize],
        g: &Geometry,
        now: Option<u64>,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let n = counted.len();
        let was = core::mem::replace(&mut self.fingers, n);
        let ids = |tracks: &[Track]| {
            let mut ids = [0u32; 5];
            for (slot, &index) in ids.iter_mut().zip(counted) {
                *slot = tracks[index].id;
            }
            ids
        };
        let set = ids(&self.tracks);
        let same_set = n == self.baseline.count && set[..n.min(5)] == self.baseline.ids[..n.min(5)];

        // A gesture whose finger set changed is over.
        match self.mode {
            Mode::Scroll(_) if n != 2 || !same_set => self.end_scroll(counted, now),
            Mode::Transform if n != 2 || !same_set => {
                self.end_shape(GestureKind::Transform, GesturePhase::End, emit)
            }
            Mode::Swipe if n < 3 || !same_set => {
                self.end_shape(GestureKind::Swipe, GesturePhase::End, emit)
            }
            Mode::Undecided if n < 2 || !same_set => self.mode = Mode::Idle,
            _ => {}
        }

        if n < 2 {
            // A later pair is measured from where it lands, never against an
            // earlier pair that happened to reuse the same contact IDs.
            self.baseline.count = 0;
        }
        if n == 0 {
            if let Mode::Lifting {
                velocity, since, ..
            } = self.mode
            {
                let within = since
                    .zip(now)
                    .is_some_and(|(since, now)| now.saturating_sub(since) <= FLING_GRACE_NS);
                if within {
                    self.coast(velocity, now);
                }
            }
            self.mode = Mode::Idle;
            self.driver = None;
            return;
        }

        // While a button is held (the pad clicked, or a tap dragging), extra
        // fingers start no gesture: the finger that moves drives the pointer
        // (click with one finger, drag with another).
        if (self.physical & 1 != 0 || self.dragging) && n >= 2 {
            // A transform or swipe the press interrupts ends where it got to.
            match self.mode {
                Mode::Transform => self.end_shape(GestureKind::Transform, GesturePhase::End, emit),
                Mode::Swipe => self.end_shape(GestureKind::Swipe, GesturePhase::End, emit),
                _ => {}
            }
            self.drive_mover(counted, g, accel, emit);
            return;
        }

        if n == 1 {
            let index = counted[0];
            let track = self.tracks[index];
            let driving = self.mode == Mode::Pointer && self.driver == Some(track.id);
            if was >= 2 && !driving && !matches!(self.mode, Mode::Lifting { .. }) {
                // The others lifted: this finger stays put until it clearly
                // moves, so lifting them never nudges the pointer before a
                // two-finger tap's click lands.
                self.mode = Mode::Lifting {
                    velocity: [0.0; 2],
                    since: now,
                    anchor: track.at,
                };
            }
            if let Mode::Lifting { anchor, .. } = self.mode {
                if g.distance(track.at, anchor) <= TAP_SLOP_MM {
                    return;
                }
                // It is pointing now: start it in place.
                self.driver = None;
            }
            self.mode = Mode::Pointer;
            self.drive(index, g, None, accel, emit);
            return;
        }

        if !same_set || matches!(self.mode, Mode::Idle | Mode::Lifting { .. }) {
            self.begin_baseline(counted, g, now);
            self.driver = None;
            self.mode = Mode::Undecided;
        } else if self.mode == Mode::Pointer {
            // Two fingers down, one resting: the other keeps pointing.
            if let Some(&index) = counted
                .iter()
                .find(|&&i| Some(self.tracks[i].id) == self.driver)
            {
                self.drive(index, g, None, accel, emit);
            }
            return;
        }
        let (centroid, spread, angle) = self.shape(counted, g);
        let travel = distance(centroid, self.baseline.centroid);
        match self.mode {
            Mode::Undecided if n == 2 => {
                let change = (spread - self.baseline.spread).abs();
                let [a, b] = [counted[0], counted[1]];
                // Each finger's travel since the pair landed, in millimetres.
                let moved = |index: usize, slot: usize| {
                    let track = self.tracks[index];
                    g.to_mm([
                        track.at[0] - self.baseline.start[slot][0],
                        track.at[1] - self.baseline.start[slot][1],
                    ])
                };
                let (da, db) = (moved(a, 0), moved(b, 1));
                // One finger travels as far as a scroll's centroid would
                // while the other rests: it points, from where it began.
                let pointing = if norm(da) > 2.0 * SCROLL_START_MM && norm(db) <= TAP_SLOP_MM {
                    Some((a, 0))
                } else if norm(db) > 2.0 * SCROLL_START_MM && norm(da) <= TAP_SLOP_MM {
                    Some((b, 1))
                } else {
                    None
                };
                // How far each finger has turned around the centroid.
                let arc = wrap(angle - self.baseline.angle).abs() * spread;
                let shaped = change.max(arc);
                if shaped > TRANSFORM_START_MM && shaped > travel * TRANSFORM_RATIO {
                    self.begin_shape(GestureKind::Transform, centroid, spread, angle, emit);
                } else if let Some((index, slot)) = pointing {
                    let track = self.tracks[index];
                    let travelled = [
                        track.at[0] - self.baseline.start[slot][0],
                        track.at[1] - self.baseline.start[slot][1],
                    ];
                    self.mode = Mode::Pointer;
                    self.driver = None;
                    let owed = Owed {
                        travel: travelled,
                        since: self.baseline.since,
                    };
                    self.drive(index, g, Some(owed), accel, emit);
                } else if travel > SCROLL_START_MM && dot(da, db) > 0.0 {
                    let step = [
                        centroid[0] - self.baseline.centroid[0],
                        centroid[1] - self.baseline.centroid[1],
                    ];
                    let lock = if step[1].abs() >= step[0].abs() * AXIS_LOCK_RATIO {
                        Lock::Vertical
                    } else if step[0].abs() >= step[1].abs() * AXIS_LOCK_RATIO {
                        Lock::Horizontal
                    } else {
                        Lock::Free
                    };
                    self.session.gestured = true;
                    self.lock_travel = [0.0; 2];
                    self.scroll_remainder = [0.0; 2];
                    self.sampled = 0;
                    self.scrolled = [0.0; 2];
                    self.mode = Mode::Scroll(lock);
                }
            }
            Mode::Undecided => {
                if travel > SWIPE_START_MM {
                    self.begin_shape(GestureKind::Swipe, centroid, spread, angle, emit);
                }
            }
            Mode::Scroll(lock) => {
                let a = self.tracks[counted[0]].delta;
                let b = self.tracks[counted[1]].delta;
                let step = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                self.scroll(lock, step, g, now, emit);
            }
            Mode::Transform => {
                // Unwrapped, so a turn past half a revolution keeps counting.
                self.baseline.rotation += wrap(angle - self.baseline.last_angle);
                self.baseline.last_angle = angle;
                self.emit_shape(
                    GestureKind::Transform,
                    GesturePhase::Update,
                    centroid,
                    spread,
                    emit,
                );
            }
            Mode::Swipe => {
                self.emit_shape(
                    GestureKind::Swipe,
                    GesturePhase::Update,
                    centroid,
                    spread,
                    emit,
                );
            }
            _ => {}
        }
    }

    /// Move the pointer by one track's travel this frame, or by the travel
    /// it is `owed` from before it drove. A track that just became the driver
    /// anchors without moving unless it is owed.
    ///
    /// The travel is accelerated on the way to pixels. A new driver's motion
    /// is its own, so the speed estimate restarts where its travel begins:
    /// at this frame for a finger that anchors, and when it began for owed
    /// travel, so a touch's first motion is timed from its start.
    fn drive(
        &mut self,
        index: usize,
        g: &Geometry,
        owed: Option<Owed>,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let track = self.tracks[index];
        let driving = self.driver == Some(track.id);
        if !driving {
            self.driver = Some(track.id);
            self.motion_remainder = [0.0; 2];
            let mut start = self.context;
            if let Some(owed) = owed {
                start.timestamp = owed.since.map(InputTimestamp);
            }
            accel.restart(start);
        }
        let travel = match owed {
            Some(owed) => owed.travel,
            None if driving => track.delta,
            None => return,
        };
        let [x, y] = accel.apply(g.to_px(travel), self.context);
        let dx = integral(x, &mut self.motion_remainder[0]);
        let dy = integral(y, &mut self.motion_remainder[1]);
        if (dx != 0 || dy != 0)
            && let Some(mut input) = self.template
        {
            input.kind = PointerInputKind::RelativeMotion { dx, dy };
            emit(PolicyEvent::Pointer(input));
        }
    }

    /// Of several fingers, the one moving drives the pointer. The driver
    /// keeps driving while it moves at least half as fast as the fastest, so
    /// sensor noise on a resting finger never steals (and re-anchors) it.
    fn drive_mover(
        &mut self,
        counted: &[usize],
        g: &Geometry,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let speed = |index: usize| norm(g.to_mm(self.tracks[index].delta));
        let fastest = counted
            .iter()
            .copied()
            .max_by(|&a, &b| speed(a).total_cmp(&speed(b)))
            .unwrap_or(counted[0]);
        let mover = counted
            .iter()
            .copied()
            .find(|&index| Some(self.tracks[index].id) == self.driver)
            .filter(|&driver| speed(driver) * 2.0 >= speed(fastest))
            .unwrap_or(fastest);
        self.mode = Mode::Pointer;
        self.drive(mover, g, None, accel, emit);
    }

    fn begin_baseline(&mut self, counted: &[usize], g: &Geometry, now: Option<u64>) {
        let (centroid, spread, angle) = self.shape(counted, g);
        let mut baseline = Baseline::EMPTY;
        baseline.since = now;
        baseline.count = counted.len().min(5);
        for (slot, &index) in counted.iter().take(5).enumerate() {
            baseline.ids[slot] = self.tracks[index].id;
            baseline.start[slot] = self.tracks[index].at;
        }
        baseline.centroid = centroid;
        baseline.spread = spread;
        baseline.angle = angle;
        baseline.last_angle = angle;
        self.baseline = baseline;
    }

    /// Centroid, mean distance from it, and (for two fingers) the angle of
    /// the line from the first to the second, all in millimetres: a pad
    /// whose axes differ in resolution still turns and spreads true.
    fn shape(&self, counted: &[usize], g: &Geometry) -> ([f64; 2], f64, f64) {
        let n = counted.len().max(1) as f64;
        let at = |index: usize| g.to_mm(self.tracks[index].at);
        let mut centroid = [0.0; 2];
        for &index in counted {
            let at = at(index);
            centroid[0] += at[0] / n;
            centroid[1] += at[1] / n;
        }
        let spread = counted
            .iter()
            .map(|&index| distance(at(index), centroid))
            .sum::<f64>()
            / n;
        let angle = match counted {
            [a, b, ..] => {
                let (a, b) = (at(*a), at(*b));
                CoreFloat::atan2(b[1] - a[1], b[0] - a[0])
            }
            _ => 0.0,
        };
        (centroid, spread, angle)
    }

    /// Recognize a transform or swipe: the gesture's identity is the finger
    /// set's shape right now.
    fn begin_shape(
        &mut self,
        kind: GestureKind,
        centroid: [f64; 2],
        spread: f64,
        angle: f64,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        self.session.gestured = true;
        self.interactions += 1;
        let baseline = &mut self.baseline;
        baseline.interaction = self.interactions;
        baseline.centroid = centroid;
        baseline.spread = spread;
        baseline.angle = angle;
        baseline.last_angle = angle;
        baseline.rotation = 0.0;
        self.sampled = 0;
        self.mode = match kind {
            GestureKind::Transform => Mode::Transform,
            _ => Mode::Swipe,
        };
        self.emit_shape(kind, GesturePhase::Begin, centroid, spread, emit);
    }

    fn emit_shape(
        &mut self,
        kind: GestureKind,
        phase: GesturePhase,
        centroid: [f64; 2],
        spread: f64,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let (Some(template), Some(geometry)) = (self.template, self.geometry) else {
            return;
        };
        let ids = self.baseline.ids;
        let mut event = GestureEvent::new(
            template.device,
            self.generation,
            self.baseline.interaction,
            kind,
            phase,
            &ids[..self.baseline.count],
        );
        event.timestamp = template.timestamp;
        event.space = geometry.space();
        event.origin = geometry.point_to_space(self.baseline.centroid);
        let translation = [
            centroid[0] - self.baseline.centroid[0],
            centroid[1] - self.baseline.centroid[1],
        ];
        event.translation = match phase {
            GesturePhase::Begin => [0.0; 2],
            _ => geometry.to_space(translation),
        };
        if phase == GesturePhase::Update {
            let now = template.timestamp.map(|t| t.0);
            if let Some(now) = now {
                self.sample(now, translation);
            }
            event.velocity = self.velocity(now).map(|v| geometry.to_space(v));
        }
        if kind == GestureKind::Transform && phase != GesturePhase::Begin {
            if self.baseline.spread > 0.0 && spread > 0.0 {
                event.scale = spread / self.baseline.spread;
            }
            event.rotation = self.baseline.rotation;
        }
        self.last_translation = match phase {
            GesturePhase::Begin => [0.0; 2],
            _ => translation,
        };
        self.last_scale = event.scale;
        emit(PolicyEvent::Gesture(event));
    }

    fn end_shape(
        &mut self,
        kind: GestureKind,
        phase: GesturePhase,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        // End repeats the last totals; Cancel carries none.
        let (Some(template), Some(geometry)) = (self.template, self.geometry) else {
            self.mode = Mode::Idle;
            return;
        };
        let ids = self.baseline.ids;
        let mut event = GestureEvent::new(
            template.device,
            self.generation,
            self.baseline.interaction,
            kind,
            phase,
            &ids[..self.baseline.count],
        );
        event.timestamp = template.timestamp;
        event.space = geometry.space();
        event.origin = geometry.point_to_space(self.baseline.centroid);
        if phase == GesturePhase::End {
            event.translation = geometry.to_space(self.last_translation);
            event.velocity = self
                .velocity(template.timestamp.map(|t| t.0))
                .map(|v| geometry.to_space(v));
            if kind == GestureKind::Transform {
                event.scale = self.last_scale;
                event.rotation = self.baseline.rotation;
            }
        }
        emit(PolicyEvent::Gesture(event));
        self.mode = Mode::Idle;
    }

    /// One frame of a scroll whose fingers moved `step` device units.
    fn scroll(
        &mut self,
        lock: Lock,
        step: [f64; 2],
        g: &Geometry,
        now: Option<u64>,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        let mut lock = lock;
        let travel = g.to_mm(step);
        match lock {
            Lock::Vertical => self.lock_travel[0] += travel[0].abs(),
            Lock::Horizontal => self.lock_travel[1] += travel[1].abs(),
            Lock::Free => {}
        }
        if self.lock_travel[0] > AXIS_UNLOCK_MM || self.lock_travel[1] > AXIS_UNLOCK_MM {
            lock = Lock::Free;
        }
        self.mode = Mode::Scroll(lock);
        let step = match lock {
            Lock::Vertical => [0.0, step[1]],
            Lock::Horizontal => [step[0], 0.0],
            Lock::Free => step,
        };
        let out = self.scroll_output(g.to_px(step));
        self.scrolled[0] += out[0];
        self.scrolled[1] += out[1];
        if let Some(now) = now {
            self.sample(now, self.scrolled);
        }
        self.emit_scroll(out, emit);
    }

    /// Finger travel in baseline pixels to scroll output, in the wheel's
    /// sign convention: positive vertical scrolls toward the top, positive
    /// horizontal toward the right.
    fn scroll_output(&self, travel: [f64; 2]) -> [f64; 2] {
        if self.config.natural_scroll {
            // Content follows the fingers: fingers down shows what is above.
            [-travel[0], travel[1]]
        } else {
            [travel[0], -travel[1]]
        }
    }

    fn emit_scroll(&mut self, out: [f64; 2], emit: &mut impl FnMut(PolicyEvent<'_>)) {
        let horizontal = integral(out[0], &mut self.scroll_remainder[0]);
        let vertical = integral(out[1], &mut self.scroll_remainder[1]);
        if horizontal == 0 && vertical == 0 {
            return;
        }
        if let Some(mut input) = self.template {
            input.kind = PointerInputKind::Scroll {
                horizontal,
                vertical,
                unit: ScrollUnit::Pixel,
            };
            emit(PolicyEvent::Pointer(input));
        }
    }

    /// Record where a scroll (output pixels) or a gesture (millimetres) has
    /// got to, for its velocity.
    fn sample(&mut self, now: u64, total: [f64; 2]) {
        if self.sampled == VELOCITY_SAMPLES {
            self.samples.copy_within(1.., 0);
            self.sampled -= 1;
        }
        self.samples[self.sampled] = (now, total);
        self.sampled += 1;
    }

    /// Velocity over the gesture's last moments, in the samples' units per
    /// second; none if it paused before `now` or there is no timing.
    fn velocity(&self, now: Option<u64>) -> Option<[f64; 2]> {
        let now = now?;
        let samples = &self.samples[..self.sampled];
        let &(last, at) = samples.last()?;
        if now.saturating_sub(last) > FLING_PAUSE_NS {
            return None;
        }
        let &(first, from) = samples
            .iter()
            .find(|(t, _)| last.saturating_sub(*t) <= FLING_WINDOW_NS)?;
        let dt = (last - first) as f64 / 1e9;
        if dt <= 0.0 {
            return None;
        }
        Some([(at[0] - from[0]) / dt, (at[1] - from[1]) / dt])
    }

    fn end_scroll(&mut self, counted: &[usize], now: Option<u64>) {
        let velocity = self.velocity(now).unwrap_or([0.0; 2]);
        match counted {
            [] => {
                self.mode = Mode::Idle;
                self.coast(velocity, now);
            }
            [index] => {
                self.mode = Mode::Lifting {
                    velocity,
                    since: now,
                    anchor: self.tracks[*index].at,
                };
            }
            _ => self.mode = Mode::Idle,
        }
    }

    fn coast(&mut self, velocity: [f64; 2], now: Option<u64>) {
        let (Some(now), true) = (now, self.config.inertia) else {
            return;
        };
        let gain = self.config.pixels_per_mm;
        let speed = norm(velocity);
        if speed < INERTIA_START_MM_S * gain {
            return;
        }
        let cap = INERTIA_MAX_MM_S * gain;
        let velocity = if speed > cap {
            [velocity[0] * cap / speed, velocity[1] * cap / speed]
        } else {
            velocity
        };
        self.inertia = Some(Inertia {
            velocity,
            last: now,
        });
    }

    /// A one-finger touch that follows a tap and moves, or holds past the
    /// tap timeout, turns the tap's held press into a drag.
    fn drag_check(&mut self, now: Option<u64>, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if self.dragging
            || !self.session.after_tap
            || self.session.clicked
            || self.session.gestured
            || self.session.count != 1
        {
            return;
        }
        let held = self
            .session
            .began()
            .zip(now)
            .is_some_and(|(began, now)| now.saturating_sub(began) >= TAP_TIMEOUT_NS);
        if self.session.moved || held {
            self.dragging = true;
            self.tapped = None;
            // Held since the tap; pressed here only if a cancel came between.
            self.press(0, emit);
        }
    }

    /// Release a tap's held press: the drag window passed, or what followed
    /// was not a drag.
    fn release_tap(&mut self, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        if self.tapped.take().is_some() {
            self.release(0, emit);
        }
    }

    /// The last counted finger lifted: end a drag, or click for a tap.
    ///
    /// A one-finger tap presses the primary button and, when taps can drag,
    /// holds it through the drag window: a touch that follows and moves drags
    /// with that same press, so an application sees one press and one
    /// release, never a click followed by a second press it could take for a
    /// double-click. Two and three-finger taps click at once.
    fn finish_session(&mut self, now: Option<u64>, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        let session = core::mem::replace(&mut self.session, Session::IDLE);
        if self.dragging {
            self.dragging = false;
            self.release(0, emit);
            return;
        }
        let quick = session
            .began()
            .zip(now)
            .is_some_and(|(began, now)| now.saturating_sub(began) <= TAP_TIMEOUT_NS);
        let tap = self.config.tap_to_click
            && quick
            && !session.moved
            && !session.clicked
            && !session.gestured
            && (1..=3).contains(&session.count);
        // Whatever this touch was, a tap still held before it is a click.
        self.release_tap(emit);
        if !tap {
            return;
        }
        let button = (session.count - 1) as u16;
        if self.held & (1 << button) != 0 {
            return;
        }
        self.press(button, emit);
        if session.count == 1 && self.config.tap_drag {
            // Held until the drag window passes or a following touch decides.
            self.tapped = now;
        } else {
            self.release(button, emit);
        }
    }

    fn next_deadline(&self) -> Option<InputTimestamp> {
        let settle = self.hybrid.settle_at();
        let inertia = self.inertia.map(|i| i.last.saturating_add(INERTIA_STEP_NS));
        let decide = if self.session.active() {
            // A touch after a tap holding still becomes a drag at the tap
            // timeout.
            (self.session.after_tap
                && !self.dragging
                && !self.session.moved
                && !self.session.clicked
                && !self.session.gestured
                && self.session.count == 1)
                .then(|| {
                    self.session
                        .began()
                        .map(|b| b.saturating_add(TAP_TIMEOUT_NS))
                })
                .flatten()
        } else {
            // A tap nothing followed releases when its drag window closes.
            self.tapped.map(|end| end.saturating_add(DRAG_WINDOW_NS))
        };
        [settle, inertia, decide]
            .into_iter()
            .flatten()
            .min()
            .map(InputTimestamp)
    }

    fn advance(
        &mut self,
        now: u64,
        accel: &mut Acceleration,
        emit: &mut impl FnMut(PolicyEvent<'_>),
    ) {
        if self.hybrid.settle_at().is_some_and(|at| at <= now) {
            self.settle(accel, emit);
        }
        if let Some(mut inertia) = self.inertia
            && now > inertia.last
        {
            let tau = INERTIA_TAU_NS as f64 / 1e9;
            let dt = (now - inertia.last) as f64 / 1e9;
            let decay = CoreFloat::exp(-dt / tau);
            let out = [
                inertia.velocity[0] * tau * (1.0 - decay),
                inertia.velocity[1] * tau * (1.0 - decay),
            ];
            inertia.velocity = [inertia.velocity[0] * decay, inertia.velocity[1] * decay];
            inertia.last = now;
            self.inertia = Some(inertia);
            if let Some(template) = self.template.as_mut() {
                template.timestamp = Some(InputTimestamp(now));
            }
            self.emit_scroll(out, emit);
            if norm(inertia.velocity) < INERTIA_STOP_MM_S * self.config.pixels_per_mm {
                self.inertia = None;
                self.scroll_remainder = [0.0; 2];
            }
        }
        if !self.session.active()
            && self
                .tapped
                .is_some_and(|end| now >= end.saturating_add(DRAG_WINDOW_NS))
        {
            if let Some(template) = self.template.as_mut() {
                template.timestamp = Some(InputTimestamp(now));
            }
            self.release_tap(emit);
        }
        self.drag_check(Some(now), emit);
    }

    /// Discontinuity: release what this policy holds, cancel a gesture in
    /// progress, and forget every contact. A button still down must be seen
    /// up before it presses again.
    fn cancel(&mut self, emit: &mut impl FnMut(PolicyEvent<'_>)) {
        match self.mode {
            Mode::Transform => self.end_shape(GestureKind::Transform, GesturePhase::Cancel, emit),
            Mode::Swipe => self.end_shape(GestureKind::Swipe, GesturePhase::Cancel, emit),
            _ => {}
        }
        for button in 0..64u16 {
            if self.held & (1 << button) != 0 {
                if let Some(template) = self.template.as_mut() {
                    template.timestamp = None;
                }
                self.release(button, emit);
            }
        }
        // The device lifetime goes on: its gesture interactions keep counting
        // up, so a consumer never mistakes a new gesture for an old one.
        let (config, dropped, interactions) = (self.config, self.hybrid.dropped, self.interactions);
        *self = Self::new(config);
        self.hybrid.dropped = dropped;
        self.interactions = interactions;
        self.resync = true;
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    norm([a[0] - b[0], a[1] - b[1]])
}

fn norm(v: [f64; 2]) -> f64 {
    CoreFloat::sqrt(v[0] * v[0] + v[1] * v[1])
}

/// An angle difference folded into (-π, π].
fn wrap(turn: f64) -> f64 {
    use core::f64::consts::{PI, TAU};
    if turn > PI {
        turn - TAU
    } else if turn <= -PI {
        turn + TAU
    } else {
        turn
    }
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Whole units out, the fraction kept for the next step. A sum within
/// floating-point noise of a whole unit is that unit: a stroke the gain maps
/// to exactly 100 pixels arrives as 100, not 99 and a remainder of 0.999….
fn integral(value: f64, remainder: &mut f64) -> i32 {
    let mut sum = value + *remainder;
    if !sum.is_finite() {
        *remainder = 0.0;
        return 0;
    }
    let whole = CoreFloat::round(sum);
    if (sum - whole).abs() <= SNAP * whole.abs().max(1.0) {
        sum = whole;
    }
    let result = sum.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
    *remainder = sum - f64::from(result);
    if !(-1.0..1.0).contains(remainder) {
        *remainder = 0.0;
    }
    result
}

#[cfg(test)]
mod tests;
