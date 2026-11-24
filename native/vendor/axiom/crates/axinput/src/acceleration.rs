//! Speed-dependent gain for relative pointing: pointer acceleration.
//!
//! [`Acceleration`] is a [`MotionPolicy`] for
//! [`PointerFallback`](crate::frame::PointerFallback), or for any recognizer
//! that emits relative motion. A mouse's or a touchpad finger's motion is
//! multiplied by a gain that depends on how fast it moves, so slow motion
//! stays precise and fast motion crosses the screen without lifting the hand.
//! Speed is distance over time, `hypot(dx, dy) / dt`, estimated over a short
//! history rather than from one report.
//!
//! The profiles, their speed setting and the velocity estimate follow
//! libinput 1.32.0, the current release, written from its behaviour rather
//! than copied:
//!
//! - `src/filter-mouse.c`: the adaptive mouse profile
//!   (`pointer_accel_profile_linear`) and how one speed setting moves it
//!   (`accelerator_set_speed`).
//! - `src/filter-touchpad.c`: the adaptive touchpad profile
//!   (`touchpad_accel_profile_linear`).
//! - `src/filter.c`: the velocity trackers (`trackers_velocity`,
//!   `trackers_reset`) and the Simpson's-rule mean of the gain across
//!   consecutive velocities (`calculate_acceleration_simpsons`).
//! - `src/filter-flat.c`: the flat profile, a constant gain.
//! - `src/evdev-mt-touchpad.c`: the history restarts when a finger lands
//!   (`filter_restart` on `TOUCH_BEGIN`).
//! - `src/libinput-private.h`: the direction octants (`xy_get_direction`).
//!
//! <https://gitlab.freedesktop.org/libinput/libinput/-/tree/1.32.0/src>
//!
//! # Units
//!
//! The policy measures travel in libinput's normalized unit, 1/1000 inch,
//! and speed in those units per millisecond:
//!
//! - **A mouse count** is one unit: a mouse that does not declare its
//!   resolution is read as 1000 dpi, as libinput reads it, and the fallback
//!   maps one of its counts to one baseline pixel
//!   ([`Calibration::pixels_per_count`] is 1).
//! - **A touchpad finger's** travel arrives in baseline pixels at the pad's
//!   gain, [`Calibration::pixels_per_mm`] (12 by default, the fallback's), so
//!   a millimetre is `1000 / 25.4` units. libinput's touchpad profile is
//!   written in millimetres per second; it is evaluated there.
//!
//! The gain multiplies baseline pixels, before the caller's fractional
//! accumulator, so a gain below one loses no motion.
//!
//! # Where this departs from libinput, and why
//!
//! - **Velocity averaging is always on.** libinput keeps two trackers, one
//!   velocity per report, unless a device quirk asks for sixteen: its kernel
//!   timestamps are taken in the interrupt handler. A runtime's receipt time
//!   ([`TimestampOrigin::Received`](crate::frame::TimestampOrigin::Received))
//!   can bunch reports that left the device apart. Averaging over sixteen
//!   trackers, which libinput's own rules already bound by direction, time
//!   and velocity change, keeps a bunch from reading as a burst of speed.
//! - **Reports received closer than [`MIN_REPORT_INTERVAL_NS`] are one
//!   sample.** 125 µs is the shortest interval a USB interrupt endpoint can
//!   be polled at (one high-speed microframe, USB 2.0 §9.6.6), so reports
//!   received closer together were bunched on their way, not sent that fast.
//!   Their travel joins the newest tracker's motion, and the bunch's speed
//!   is its whole travel over the time since the sample before it. libinput
//!   measures each report on its own and only adds one microsecond to the
//!   interval so it never divides by zero; a bunch would read there as a
//!   burst of speed, the cap's gain for no motion of the hand's.
//! - **A touchpad's plateau is the pad's own gain.** libinput's touchpad
//!   profile sits on a plateau of `0.9 × TP_MAGIC_SLOWDOWN` (0.2968), about
//!   10.5 normalized units per millimetre, a constant it tuned against its
//!   mouse units. Here the plateau is the gain the pad already moves at,
//!   [`Calibration::pixels_per_mm`]: the curve keeps libinput's shape and
//!   its speed thresholds, relative to that gain. With acceleration on, a
//!   finger moves exactly as it did without it until it is fast.
//! - **The speed setting is an even scale on a touchpad and without
//!   acceleration.** libinput scales its touchpad curve by
//!   `(s + 1)^2.38 · 0.95 + 0.05` (0.05 at the slowest setting, 5 at the
//!   fastest) and its flat profile by `max(1 + s, 0.005)`. Their slow ends
//!   all but stop the pointer, and a setting offered as five steps puts them
//!   one choice away. Here both scale by `2^s` ([`SPEED_SCALE_RANGE`]): half
//!   the default at the slowest, twice it at the fastest, and an equal ratio
//!   (√2) between neighbouring half-steps. A mouse's adaptive curve keeps
//!   libinput's speed setting, which already runs from half speed to a cap
//!   of 3.5.
//! - **The flat profile is the plateau's gain on every device.** libinput's
//!   flat touchpad omits the 0.9 baseline its adaptive one carries (its
//!   source calls that a hack), so turning acceleration off there speeds
//!   slow motion up by a ninth. Here no acceleration moves at the plateau's
//!   gain times the speed setting's scale, on a touchpad as on a mouse.
//! - **Every source's history is in normalized units.** libinput feeds a
//!   touchpad's trackers in device units, so its 1 unit/ms velocity-change
//!   limit is a different finger speed on every pad. Here it is one
//!   normalized unit per millisecond, 25.4 mm/s, whatever the pad's
//!   resolution.
//! - **Only [`PointerSource::Mouse`] and [`PointerSource::Touchpad`] are
//!   accelerated.** A [`PointerSource::Virtual`] source, a host's emulated
//!   mouse, carries a pointer the host has already accelerated;
//!   accelerating it again is the double acceleration virtual machines are
//!   known for. Virtual, touchscreen and pen motion passes through
//!   unchanged, whatever the speed setting.
//! - **A pause or a landing ends the smoothing too.** libinput averages each
//!   motion's gain with the previous motion's velocity by Simpson's rule,
//!   however long ago that motion was, and a touchpad's restart keeps it.
//!   After a fast flick and a pause, or a lift and a new touch, the next
//!   slow motion would ride the flick's gain. Here a motion after a pause
//!   longer than [`MOTION_TIMEOUT_NS`], or the first after a
//!   [`MotionPolicy::restart`], averages with its own velocity only.
//! - **A frame without a clock moves at the profile's plateau**: the gain
//!   the curve gives between its slow and fast ends (1 at the default
//!   speed, the speed setting's scale on a touchpad or without
//!   acceleration). Speed cannot be measured without time, and this policy never
//!   guesses one ([`MotionContext::time`]).
//!
//! All state is bounded and allocator-free. No clock is read: time arrives
//! with each motion.

use crate::{
    InputTimestamp, PointerSource,
    frame::{MotionContext, MotionPolicy},
};
use axmath::{atan2_f64, exp_f64, hypot_f64, log_f64};
use core::f64::consts::{PI, TAU};

/// The adaptive mouse profile's plateau ends here at the default speed: up
/// to this speed motion is 1:1, above it the gain climbs. libinput's
/// `DEFAULT_THRESHOLD`, units per millisecond.
pub const DEFAULT_THRESHOLD: f64 = 0.4;
/// The lowest the mouse plateau's end may move to, however fast the speed
/// setting. libinput's `MINIMUM_THRESHOLD`, units per millisecond.
pub const MINIMUM_THRESHOLD: f64 = 0.2;
/// The mouse gain's cap at the default speed. libinput's
/// `DEFAULT_ACCELERATION`.
pub const DEFAULT_MAX_GAIN: f64 = 2.0;
/// The mouse gain's climb above the plateau, per unit per millisecond, at
/// the default speed. libinput's `DEFAULT_INCLINE`.
pub const DEFAULT_INCLINE: f64 = 1.1;
/// Below this speed mouse motion is slowed, down to [`SLOWEST_GAIN`] at
/// rest, so a hand's tremor and a deliberate one-pixel nudge stay small.
/// libinput's 0.07, which its source calls trial and error; units per
/// millisecond.
pub const DECELERATION_LIMIT: f64 = 0.07;
/// The gain at zero speed, relative to the plateau, on either device.
/// libinput chose 0.3 as "above the Nyquist frequency for subpixel motion
/// within a pixel".
pub const SLOWEST_GAIN: f64 = 0.3;

/// The adaptive touchpad profile's plateau ends here, whatever the speed
/// setting: above it the gain climbs. libinput's touchpad `threshold`,
/// millimetres per second.
pub const TOUCHPAD_THRESHOLD_MM_S: f64 = 130.0;
/// The touchpad gain stops climbing at this finger speed. libinput's
/// `upper_threshold`, four times the threshold "because why not".
pub const TOUCHPAD_UPPER_THRESHOLD_MM_S: f64 = 4.0 * TOUCHPAD_THRESHOLD_MM_S;
/// Below this finger speed touchpad motion is slowed, down to a third of
/// the plateau at rest. libinput's 7 mm/s.
pub const TOUCHPAD_DECELERATION_LIMIT_MM_S: f64 = 7.0;
/// libinput's touchpad curve in its own terms: a plateau of 0.9, slowed by
/// 0.1 per mm/s from 0.3 at rest, climbing above the threshold as
/// `0.0025 · (v / threshold) · (v - threshold)`. Only the curve's shape is
/// kept: it is divided by the plateau (see the module documentation).
const TOUCHPAD_BASELINE: f64 = 0.9;
const TOUCHPAD_SLOW_INCLINE: f64 = 0.1;
const TOUCHPAD_CURVATURE: f64 = 0.0025;
/// How far the speed setting reaches on a touchpad and without
/// acceleration. The setting `s` scales the gain by `SPEED_SCALE_RANGE^s`:
/// half the default at -1, twice it at 1, and evenly between. Each step
/// changes the speed by the same ratio, so no step all but stops the pointer
/// and none leaps. Two is where a 1920x1200 screen's diagonal is one
/// comfortable stroke at the default speed and Fastest still aims a pixel.
pub const SPEED_SCALE_RANGE: f64 = 2.0;
/// Trackers further back than this are too old to describe the current
/// motion. The first motion after such a pause is timed over this whole
/// interval instead. libinput's `MOTION_TIMEOUT`, one second.
pub const MOTION_TIMEOUT_NS: u64 = 1_000_000_000;
/// The shortest interval a velocity is measured over: one USB high-speed
/// microframe. See the module documentation.
pub const MIN_REPORT_INTERVAL_NS: u64 = 125_000;
/// Averaging stops at a tracker whose velocity differs from the newest pair's
/// by more than this. libinput's `MAX_VELOCITY_DIFF`, units per millisecond.
pub const MAX_VELOCITY_DIFF: f64 = 1.0;
/// How many past motions the velocity estimate may span. libinput's count
/// with velocity averaging on.
pub const TRACKERS: usize = 16;
/// Normalized units in a millimetre: the unit is 1/1000 inch.
pub const UNITS_PER_MM: f64 = 1000.0 / 25.4;
/// The baseline pixels a millimetre of finger travel arrives as by default:
/// [`PointerFallback::default`](crate::frame::PointerFallback)'s gain.
pub const DEFAULT_PIXELS_PER_MM: f64 = 12.0;

const NANOS_PER_MILLI: f64 = 1_000_000.0;
const MILLIS_PER_SECOND: f64 = 1000.0;
/// The largest calibration either way: a count or a millimetre that moves a
/// million baseline pixels is no calibration.
const MAX_CALIBRATION: f64 = 1_000_000.0;

/// The gain's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// No acceleration: the plateau's gain times the speed setting's
    /// scale, at any speed.
    Flat,
    /// libinput's adaptive profiles: slowed at a crawl, 1:1 on a plateau,
    /// then climbing to a cap. A mouse and a touchpad each take their own.
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationError {
    /// The speed setting is not a finite value in `-1.0..=1.0`.
    InvalidSpeed,
    /// A calibration is not a finite, positive, bounded value.
    InvalidCalibration,
}

/// A user's pointer preference, as libinput's configuration takes it. One
/// preference serves a mouse and a touchpad alike; each reads it on its own
/// profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccelerationConfig {
    profile: Profile,
    speed: f64,
}

impl AccelerationConfig {
    /// `speed` runs from -1 (slowest) through 0 (the default) to 1
    /// (fastest). On a mouse it moves the whole adaptive curve, as
    /// libinput's does: where the plateau ends, how steeply the gain climbs,
    /// and its cap. On a touchpad, and on the flat profile, it scales the
    /// gain by [`SPEED_SCALE_RANGE`]`^speed`.
    pub fn new(profile: Profile, speed: f64) -> Result<Self, AccelerationError> {
        if !speed.is_finite() || !(-1.0..=1.0).contains(&speed) {
            return Err(AccelerationError::InvalidSpeed);
        }
        Ok(Self { profile, speed })
    }

    pub const fn profile(self) -> Profile {
        self.profile
    }
    pub const fn speed(self) -> f64 {
        self.speed
    }
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Adaptive,
            speed: 0.0,
        }
    }
}

/// How the caller's baseline pixels relate to the hand's travel: a fact of
/// the device and its policy, not a preference, so a new
/// [`AccelerationConfig`] keeps it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pixels_per_count: f64,
    pixels_per_mm: f64,
}

impl Calibration {
    /// An uncalibrated mouse's count is one pixel, and a finger's millimetre
    /// is [`DEFAULT_PIXELS_PER_MM`]: [`PointerFallback::default`]'s gains.
    ///
    /// [`PointerFallback::default`]: crate::frame::PointerFallback
    pub const DEFAULT: Self = Self {
        pixels_per_count: 1.0,
        pixels_per_mm: DEFAULT_PIXELS_PER_MM,
    };

    /// The baseline pixels one mouse count arrives as, for a fallback that
    /// calibrates its mouse
    /// ([`PointerFallback::with_calibrated_mouse`](crate::frame::PointerFallback::with_calibrated_mouse)).
    pub fn with_pixels_per_count(mut self, pixels: f64) -> Result<Self, AccelerationError> {
        self.pixels_per_count = calibration(pixels)?;
        Ok(self)
    }

    /// The baseline pixels one millimetre of finger travel arrives as: the
    /// touchpad gain of the fallback or recognizer that feeds this policy,
    /// and the touchpad profile's plateau.
    pub fn with_pixels_per_mm(mut self, pixels: f64) -> Result<Self, AccelerationError> {
        self.pixels_per_mm = calibration(pixels)?;
        Ok(self)
    }

    pub const fn pixels_per_count(self) -> f64 {
        self.pixels_per_count
    }
    pub const fn pixels_per_mm(self) -> f64 {
        self.pixels_per_mm
    }

    /// Normalized units in one baseline pixel of `kind`'s motion.
    fn units_per_pixel(self, kind: Kind) -> f64 {
        match kind {
            Kind::Mouse => 1.0 / self.pixels_per_count,
            Kind::Touchpad => UNITS_PER_MM / self.pixels_per_mm,
        }
    }
}

impl Default for Calibration {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn calibration(pixels: f64) -> Result<f64, AccelerationError> {
    if pixels.is_finite() && pixels > 0.0 && pixels <= MAX_CALIBRATION {
        Ok(pixels)
    } else {
        Err(AccelerationError::InvalidCalibration)
    }
}

/// The devices this policy accelerates, each on its own adaptive profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Mouse,
    Touchpad,
}

impl Kind {
    const fn of(source: PointerSource) -> Option<Self> {
        match source {
            PointerSource::Mouse => Some(Self::Mouse),
            PointerSource::Touchpad => Some(Self::Touchpad),
            _ => None,
        }
    }
}

/// One speed setting's curves. A mouse's adaptive curve moves as
/// libinput's `accelerator_set_speed` moves it, numbers its source calls
/// "trial-and-error magic" that "mostly worked ok". A touchpad's, and the
/// flat profile's, take the setting's even scale.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Curve {
    profile: Profile,
    threshold: f64,
    max_gain: f64,
    incline: f64,
    /// [`SPEED_SCALE_RANGE`]`^speed`: the flat profile's gain and the
    /// touchpad curve's scale.
    scale: f64,
}

impl Curve {
    fn new(config: AccelerationConfig) -> Self {
        let speed = config.speed;
        Self {
            profile: config.profile,
            threshold: (DEFAULT_THRESHOLD - 0.25 * speed).max(MINIMUM_THRESHOLD),
            max_gain: DEFAULT_MAX_GAIN + 1.5 * speed,
            incline: DEFAULT_INCLINE + 0.75 * speed,
            scale: exp_f64(speed * log_f64(SPEED_SCALE_RANGE)),
        }
    }

    /// The adaptive mouse gain at `speed` units per millisecond.
    fn mouse(&self, speed: f64) -> f64 {
        let gain = if speed < DECELERATION_LIMIT {
            // 1 = a * 0.07 + 0.3, so the slow incline meets the plateau.
            (1.0 - SLOWEST_GAIN) / DECELERATION_LIMIT * speed + SLOWEST_GAIN
        } else if speed < self.threshold {
            1.0
        } else {
            self.incline * (speed - self.threshold) + 1.0
        };
        gain.min(self.max_gain)
    }

    /// The adaptive touchpad gain at `speed` units per millisecond, relative
    /// to the plateau.
    fn touchpad(&self, speed: f64) -> f64 {
        let mm_s = speed / UNITS_PER_MM * MILLIS_PER_SECOND;
        let factor = if mm_s < TOUCHPAD_DECELERATION_LIMIT_MM_S {
            (TOUCHPAD_SLOW_INCLINE * mm_s + SLOWEST_GAIN).min(TOUCHPAD_BASELINE)
        } else if mm_s < TOUCHPAD_THRESHOLD_MM_S {
            TOUCHPAD_BASELINE
        } else {
            let v = mm_s.min(TOUCHPAD_UPPER_THRESHOLD_MM_S);
            TOUCHPAD_CURVATURE * (v / TOUCHPAD_THRESHOLD_MM_S) * (v - TOUCHPAD_THRESHOLD_MM_S)
                + TOUCHPAD_BASELINE
        };
        factor / TOUCHPAD_BASELINE * self.scale
    }

    /// The gain at `speed` units per millisecond, on `kind`'s curve.
    fn gain(&self, kind: Kind, speed: f64) -> f64 {
        match (self.profile, kind) {
            (Profile::Flat, _) => self.scale,
            (Profile::Adaptive, Kind::Mouse) => self.mouse(speed),
            (Profile::Adaptive, Kind::Touchpad) => self.touchpad(speed),
        }
    }

    /// The gain between the curve's slow and fast ends.
    fn plateau(&self, kind: Kind) -> f64 {
        match (self.profile, kind) {
            (Profile::Flat, _) => self.scale,
            (Profile::Adaptive, Kind::Mouse) => self.max_gain.min(1.0),
            (Profile::Adaptive, Kind::Touchpad) => self.scale,
        }
    }
}

/// One past motion: the travel since it, and when and which way it went.
#[derive(Debug, Clone, Copy)]
struct Tracker {
    /// Units travelled since this tracker was made.
    delta: [f64; 2],
    /// `None` until a motion fills it: older than any timeout.
    time: Option<u64>,
    /// [`direction`]'s octants of the motion that made this tracker.
    direction: u8,
}

const EMPTY: Tracker = Tracker {
    delta: [0.0; 2],
    time: None,
    direction: 0,
};

/// What the policy last did, for diagnostics and replay.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AccelerationStatus {
    /// The latest velocity estimate, normalized units (1/1000 inch) per
    /// millisecond.
    pub velocity: f64,
    /// The gain the latest accelerated motion was multiplied by.
    pub gain: f64,
    /// Accelerated motions with a clock, measured.
    pub measured: u64,
    /// Accelerated motions without a clock, moved at the plateau gain.
    pub unclocked: u64,
    /// Times a motion's time preceded the newest tracker's, which restarts
    /// the history rather than measuring a negative interval.
    pub time_reversals: u64,
    /// Times a new motion began unrelated to the last
    /// ([`MotionPolicy::restart`]): a finger landed.
    pub restarts: u64,
}

/// libinput's pointer acceleration as an Axiom [`MotionPolicy`]. One per
/// device: its history describes one device's motion.
#[derive(Debug, Clone, Copy)]
pub struct Acceleration {
    config: AccelerationConfig,
    calibration: Calibration,
    curve: Curve,
    trackers: [Tracker; TRACKERS],
    /// Index of the newest tracker.
    current: usize,
    /// The motion that made the newest tracker, including every report
    /// merged into it: its direction is this motion's.
    newest: [f64; 2],
    /// The previous motion's velocity, units per millisecond.
    last_velocity: f64,
    /// The next motion is the first since a restart: it has no previous
    /// velocity to ease from.
    fresh: bool,
    status: AccelerationStatus,
}

impl Default for Acceleration {
    fn default() -> Self {
        Self::new(AccelerationConfig::default())
    }
}

impl Acceleration {
    /// A policy at the [`Calibration::DEFAULT`] gains.
    pub fn new(config: AccelerationConfig) -> Self {
        let curve = Curve::new(config);
        Self {
            config,
            calibration: Calibration::DEFAULT,
            curve,
            trackers: [EMPTY; TRACKERS],
            current: 0,
            newest: [0.0; 2],
            last_velocity: 0.0,
            fresh: false,
            status: AccelerationStatus {
                gain: 1.0,
                ..AccelerationStatus::default()
            },
        }
    }

    /// The same policy for a caller whose pixels relate to the hand's
    /// travel otherwise.
    pub const fn with_calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    pub const fn config(&self) -> AccelerationConfig {
        self.config
    }

    pub const fn calibration(&self) -> Calibration {
        self.calibration
    }

    /// Take a new preference. The calibration and the motion history are
    /// kept: they describe the device and the hand, not the setting, so the
    /// next motion is measured as before.
    pub fn set_config(&mut self, config: AccelerationConfig) {
        self.config = config;
        self.curve = Curve::new(config);
    }

    pub const fn status(&self) -> AccelerationStatus {
        self.status
    }

    /// The gain this configuration gives `source` moving steadily at `speed`
    /// normalized units per millisecond: 1 for a source it does not
    /// accelerate, and `None` for a speed that is not finite and
    /// non-negative.
    pub fn gain_at(&self, source: PointerSource, speed: f64) -> Option<f64> {
        if !speed.is_finite() || speed < 0.0 {
            return None;
        }
        Some(Kind::of(source).map_or(1.0, |kind| self.curve.gain(kind, speed)))
    }

    fn tracker(&self, offset: usize) -> &Tracker {
        &self.trackers[(self.current + TRACKERS - offset) % TRACKERS]
    }

    /// Forget every past motion, as a newly bound device has none.
    fn forget(&mut self) {
        self.trackers = [EMPTY; TRACKERS];
        self.current = 0;
        self.newest = [0.0; 2];
        self.last_velocity = 0.0;
        self.fresh = false;
    }

    /// libinput's `trackers_feed`: a new sample at `now`.
    fn feed(&mut self, units: [f64; 2], now: u64) {
        for tracker in &mut self.trackers {
            tracker.delta[0] += units[0];
            tracker.delta[1] += units[1];
        }
        self.current = (self.current + 1) % TRACKERS;
        self.newest = units;
        self.trackers[self.current] = Tracker {
            delta: [0.0; 2],
            time: Some(now),
            direction: direction(units[0], units[1]),
        };
    }

    /// A report bunched with the newest sample becomes part of it: every
    /// older tracker travelled it too, and the newest's direction is now the
    /// bunch's. The newest tracker's own travel stays zero, since the
    /// motion happened at its time.
    fn merge(&mut self, units: [f64; 2]) {
        for (index, tracker) in self.trackers.iter_mut().enumerate() {
            if index != self.current {
                tracker.delta[0] += units[0];
                tracker.delta[1] += units[1];
            }
        }
        self.newest[0] += units[0];
        self.newest[1] += units[1];
        self.trackers[self.current].direction = direction(self.newest[0], self.newest[1]);
    }

    /// libinput's `trackers_velocity`: the velocity over the oldest past
    /// motion that still describes this one, in units per millisecond, and
    /// whether this motion follows a pause.
    fn velocity(&self, now: u64) -> (f64, bool) {
        let mut direction = self.tracker(0).direction;
        let mut result = 0.0;
        let mut initial = 0.0;
        for offset in 1..TRACKERS {
            let tracker = self.tracker(offset);
            let age = tracker.time.and_then(|time| now.checked_sub(time));
            let Some(age) = age.filter(|&age| age <= MOTION_TIMEOUT_NS) else {
                // Too old, never filled, or (guarded against by `apply`)
                // from the future. A motion after a pause is timed over the
                // whole timeout: slow, never a leap.
                if offset == 1 {
                    return (speed(tracker.delta, MOTION_TIMEOUT_NS), true);
                }
                break;
            };
            let velocity = speed(tracker.delta, age);
            // A restart's anchor points every way (`UNDEFINED_DIRECTION`):
            // the first motion of a touch is timed from its landing.
            direction &= tracker.direction;
            if direction == 0 {
                // The first motion after a turn is as fast as the last one.
                if offset == 1 {
                    result = velocity;
                }
                break;
            }
            // The newest two are always averaged: it smooths a jumpy first
            // report. Older ones are taken while they stay close to them.
            if initial == 0.0 || offset <= 2 {
                result = velocity;
                initial = velocity;
            } else if (initial - velocity).abs() > MAX_VELOCITY_DIFF {
                break;
            } else {
                result = velocity;
            }
        }
        (result, false)
    }

    /// The adaptive gain across the previous and the current velocity, by
    /// Simpson's rule, so the gain eases between motions rather than jumps.
    fn smoothed(&self, kind: Kind, velocity: f64) -> f64 {
        let last = self.last_velocity;
        let curve = &self.curve;
        (curve.gain(kind, velocity)
            + curve.gain(kind, last)
            + 4.0 * curve.gain(kind, (last + velocity) / 2.0))
            / 6.0
    }
}

impl MotionPolicy for Acceleration {
    fn apply(&mut self, delta: [f64; 2], context: MotionContext) -> [f64; 2] {
        let Some(kind) = Kind::of(context.source) else {
            return delta;
        };
        if !delta[0].is_finite() || !delta[1].is_finite() {
            // The fallback discards a nonfinite result and resets us; say
            // nothing about a speed that was never measured.
            self.forget();
            return delta;
        }
        let gain = match context.time() {
            None => {
                self.status.unclocked = self.status.unclocked.saturating_add(1);
                self.curve.plateau(kind)
            }
            Some(InputTimestamp(now)) => {
                if self.tracker(0).time.is_some_and(|newest| now < newest) {
                    self.status.time_reversals = self.status.time_reversals.saturating_add(1);
                    self.forget();
                }
                let scale = self.calibration.units_per_pixel(kind);
                let units = [delta[0] * scale, delta[1] * scale];
                match self.tracker(0).time {
                    Some(newest) if now.saturating_sub(newest) < MIN_REPORT_INTERVAL_NS => {
                        self.merge(units)
                    }
                    _ => self.feed(units, now),
                }
                let (velocity, paused) = self.velocity(now);
                if paused || self.fresh {
                    // The motion before the pause, or before the finger
                    // landed, is not this hand's now.
                    self.last_velocity = velocity;
                    self.fresh = false;
                }
                let gain = match self.curve.profile {
                    Profile::Flat => self.curve.scale,
                    Profile::Adaptive => self.smoothed(kind, velocity),
                };
                self.last_velocity = velocity;
                self.status.velocity = velocity;
                self.status.measured = self.status.measured.saturating_add(1);
                gain
            }
        };
        self.status.gain = gain;
        [delta[0] * gain, delta[1] * gain]
    }

    fn reset(&mut self) {
        self.forget();
    }

    /// libinput's `trackers_reset`: the history is forgotten, and the next
    /// motion is timed from the restart, so a touch's first motion is
    /// measured from when the finger landed.
    fn restart(&mut self, context: MotionContext) {
        self.forget();
        if let Some(InputTimestamp(at)) = context.time() {
            self.trackers[self.current] = Tracker {
                delta: [0.0; 2],
                time: Some(at),
                direction: UNDEFINED_DIRECTION,
            };
            self.fresh = true;
        }
        self.status.restarts = self.status.restarts.saturating_add(1);
    }
}

/// Units per millisecond for `delta` units travelled over `nanos`. No
/// interval is read as shorter than one microframe: merged reports never
/// leave a shorter one, and none can divide by zero.
fn speed(delta: [f64; 2], nanos: u64) -> f64 {
    hypot_f64(delta[0], delta[1]) / (nanos.max(MIN_REPORT_INTERVAL_NS) as f64 / NANOS_PER_MILLI)
}

const N: u8 = 1 << 0;
const NE: u8 = 1 << 1;
const E: u8 = 1 << 2;
const SE: u8 = 1 << 3;
const S: u8 = 1 << 4;
const SW: u8 = 1 << 5;
const W: u8 = 1 << 6;
const NW: u8 = 1 << 7;
const UNDEFINED_DIRECTION: u8 = 0xff;

/// libinput's `xy_get_direction`: the one or two octants a motion points
/// into, y growing downwards. A short motion's angle is too coarse to trust,
/// so it marks the three octants around its quadrant or axis.
fn direction(x: f64, y: f64) -> u8 {
    if x.abs() < 2.0 && y.abs() < 2.0 {
        return if x > 0.0 && y > 0.0 {
            S | SE | E
        } else if x > 0.0 && y < 0.0 {
            N | NE | E
        } else if x < 0.0 && y > 0.0 {
            S | SW | W
        } else if x < 0.0 && y < 0.0 {
            N | NW | W
        } else if x > 0.0 {
            NE | E | SE
        } else if x < 0.0 {
            NW | W | SW
        } else if y > 0.0 {
            SE | S | SW
        } else if y < 0.0 {
            NE | N | NW
        } else {
            UNDEFINED_DIRECTION
        };
    }
    // atan2 is in (-pi, pi]; shifted by 2.5 pi it is in (1.5 pi, 3.5 pi],
    // which two subtractions at most bring into [0, 2 pi), north at 0.
    let mut turn = atan2_f64(y, x) + 2.5 * PI;
    if turn >= TAU {
        turn -= TAU;
    }
    if turn >= TAU {
        turn -= TAU;
    }
    let octant = turn * (4.0 / PI);
    // Mark one octant, or two when the motion lies near their boundary.
    let first = (octant + 0.9) as u32 % 8;
    let second = (octant + 0.1) as u32 % 8;
    (1 << first) | (1 << second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directions_match_libinputs_octants() {
        // Long motions: one octant, or two at a boundary, y down.
        assert_eq!(direction(0.0, -10.0), N);
        assert_eq!(direction(10.0, 0.0), E);
        assert_eq!(direction(0.0, 10.0), S);
        assert_eq!(direction(-10.0, 0.0), W);
        assert_eq!(direction(10.0, -10.0), NE);
        assert_eq!(direction(10.0, 10.0), SE);
        assert_eq!(direction(-10.0, 10.0), SW);
        assert_eq!(direction(-10.0, -10.0), NW);
        // Between E (90 degrees) and SE (135): 22.5 degrees past E is the
        // boundary, marked on both sides.
        let boundary = direction(10.0, 10.0 * (PI / 8.0).tan());
        assert_eq!(boundary, E | SE);
        // Short motions: three octants around the quadrant or axis.
        assert_eq!(direction(1.0, 1.0), S | SE | E);
        assert_eq!(direction(1.0, 0.0), NE | E | SE);
        assert_eq!(direction(0.0, -1.0), NE | N | NW);
        assert_eq!(direction(0.0, 0.0), UNDEFINED_DIRECTION);
        // Every long direction marks one octant or two adjacent ones.
        for step in 0..360 {
            let angle = f64::from(step) * PI / 180.0;
            let bits = direction(10.0 * axmath::cos_f64(angle), 10.0 * axmath::sin_f64(angle));
            let one = (0..8).any(|i| bits == 1 << i);
            let two = (0..8).any(|i| bits == (1 << i) | (1 << ((i + 1) % 8)));
            assert!(one || two, "{step} degrees: {bits:08b}");
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn policy(profile: Profile, speed: f64) -> Acceleration {
        Acceleration::new(AccelerationConfig::new(profile, speed).unwrap())
    }

    #[test]
    fn the_mouse_curve_is_libinputs_at_every_speed_setting() {
        let at = |speed: f64, velocity: f64| {
            policy(Profile::Adaptive, speed)
                .gain_at(PointerSource::Mouse, velocity)
                .unwrap()
        };
        // Default: 0.3 at rest, the plateau from 0.07 to 0.4, 1.1 per unit
        // per ms above it, capped at 2.
        assert!(close(at(0.0, 0.0), 0.3));
        assert!(close(at(0.0, 0.035), 0.65));
        assert!(close(at(0.0, 0.07), 1.0));
        assert!(close(at(0.0, 0.39), 1.0));
        assert!(close(at(0.0, 0.4), 1.0));
        assert!(close(at(0.0, 1.0), 1.0 + 1.1 * 0.6));
        assert!(close(at(0.0, 10.0), 2.0));
        // Fastest: the plateau ends at the 0.2 floor (0.4 - 0.25 is below
        // it), climbs at 1.85 and caps at 3.5.
        assert!(close(at(1.0, 0.19), 1.0));
        assert!(close(at(1.0, 1.0), 1.0 + 1.85 * 0.8));
        assert!(close(at(1.0, 100.0), 3.5));
        // Slowest: the cap is 0.5, below the plateau, so everything above
        // the slow incline's 0.5 moves at half speed.
        assert!(close(at(-1.0, 0.0), 0.3));
        assert!(close(at(-1.0, 0.02), 0.5));
        assert!(close(at(-1.0, 5.0), 0.5));
    }

    #[test]
    fn the_touchpad_curve_is_libinputs_relative_to_its_plateau() {
        // Finger speeds in mm/s, as libinput writes its touchpad profile.
        let at = |speed: f64, mm_s: f64| {
            policy(Profile::Adaptive, speed)
                .gain_at(PointerSource::Touchpad, mm_s * UNITS_PER_MM / 1000.0)
                .unwrap()
        };
        // Default: a third of the plateau at rest, slowed by 0.1/0.9 per
        // mm/s up to 6 mm/s, the plateau from there to 130 mm/s, then
        // libinput's curve up to its cap at 520 mm/s, 4.8/0.9 of the plateau.
        assert!(close(at(0.0, 0.0), 0.3 / 0.9));
        assert!(close(at(0.0, 3.5), 0.65 / 0.9));
        assert!(close(at(0.0, 6.5), 1.0));
        assert!(close(at(0.0, 7.0), 1.0));
        assert!(close(at(0.0, 129.9), 1.0));
        assert!(close(at(0.0, 260.0), 1.55 / 0.9));
        assert!(close(
            at(0.0, 400.0),
            (0.0025 * (400.0 / 130.0) * 270.0 + 0.9) / 0.9
        ));
        assert!(close(at(0.0, 520.0), 4.8 / 0.9));
        assert!(close(at(0.0, 5000.0), 4.8 / 0.9));
        // The setting scales the whole curve evenly, 2^s: half at the
        // slowest, twice at the fastest, sqrt(2) apart at each half-step.
        for (speed, scale) in [
            (-1.0, 0.5),
            (-0.5, core::f64::consts::FRAC_1_SQRT_2),
            (0.5, core::f64::consts::SQRT_2),
            (1.0, 2.0),
        ] {
            assert!(
                close(at(speed, 50.0), scale),
                "{speed}: {}",
                at(speed, 50.0)
            );
            assert!(close(at(speed, 1000.0), scale * 4.8 / 0.9));
            assert!(close(
                policy(Profile::Adaptive, speed)
                    .curve
                    .plateau(Kind::Touchpad),
                scale
            ));
        }
    }

    #[test]
    fn flat_is_the_plateaus_gain_on_every_device() {
        for source in [PointerSource::Mouse, PointerSource::Touchpad] {
            let flat = |speed: f64| policy(Profile::Flat, speed).gain_at(source, 7.0).unwrap();
            assert!(close(flat(0.0), 1.0));
            assert!(close(flat(0.5), core::f64::consts::SQRT_2));
            assert!(close(flat(-0.5), core::f64::consts::FRAC_1_SQRT_2));
            assert!(close(flat(1.0), 2.0));
            assert!(close(flat(-1.0), 0.5));
        }
    }

    #[test]
    fn every_half_step_of_the_setting_is_the_same_ratio() {
        // The five steps a desktop offers, for the flat profile and the
        // touchpad's scale alike: no step stalls the pointer or leaps.
        let steps = [-1.0, -0.5, 0.0, 0.5, 1.0].map(|s| policy(Profile::Flat, s).curve.scale);
        for pair in steps.windows(2) {
            assert!(
                close(pair[1] / pair[0], core::f64::consts::SQRT_2),
                "{steps:?}"
            );
        }
        for s in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_eq!(
                policy(Profile::Flat, s).curve.scale,
                policy(Profile::Adaptive, s).curve.scale
            );
        }
    }

    #[test]
    fn other_sources_and_bad_speeds_have_no_curve() {
        let fastest = policy(Profile::Adaptive, 1.0);
        for source in [
            PointerSource::Virtual,
            PointerSource::Touchscreen,
            PointerSource::Pen,
        ] {
            assert_eq!(fastest.gain_at(source, 9.0), Some(1.0));
        }
        let default = Acceleration::default();
        assert!(default.gain_at(PointerSource::Mouse, f64::NAN).is_none());
        assert!(default.gain_at(PointerSource::Touchpad, -1.0).is_none());
    }

    #[test]
    fn configurations_refuse_what_is_not_a_setting() {
        assert_eq!(
            AccelerationConfig::new(Profile::Adaptive, 1.5),
            Err(AccelerationError::InvalidSpeed)
        );
        assert_eq!(
            AccelerationConfig::new(Profile::Flat, f64::NAN),
            Err(AccelerationError::InvalidSpeed)
        );
        let calibration = Calibration::DEFAULT;
        for pixels in [0.0, -1.0, f64::INFINITY, f64::NAN, 2.0e6] {
            assert_eq!(
                calibration.with_pixels_per_count(pixels),
                Err(AccelerationError::InvalidCalibration)
            );
            assert_eq!(
                calibration.with_pixels_per_mm(pixels),
                Err(AccelerationError::InvalidCalibration)
            );
        }
        let custom = calibration
            .with_pixels_per_count(0.3)
            .and_then(|c| c.with_pixels_per_mm(9.0))
            .unwrap();
        assert_eq!(
            (custom.pixels_per_count(), custom.pixels_per_mm()),
            (0.3, 9.0)
        );
        // A new preference keeps the calibration.
        let mut accel = Acceleration::default().with_calibration(custom);
        accel.set_config(AccelerationConfig::new(Profile::Flat, 0.5).unwrap());
        assert_eq!(accel.calibration(), custom);
    }
}
