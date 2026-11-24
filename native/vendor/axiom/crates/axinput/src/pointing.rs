//! Opt-in pointing experiments, not a default acceleration curve or a gesture
//! recognizer. See `docs/pointing-evaluation.md` for the research and deviations.
//!
//! All state is bounded and allocator-free. Coordinates are continuous
//! baseline pixels BEFORE adaptive gain, viewport clipping, or integer output.
//! No clock, display, device driver, or accelerator is acquired here.

use crate::{
    InputTimestamp,
    frame::{MotionContext, MotionPolicy},
};
use axmath::{atan2_f64, cos_f64, exp_f64, hypot_f64, log2_f64, sin_f64, sqrt_f64};
use core::f64::consts::{PI, TAU};

const HISTORY: usize = 16;
const MAX_SAMPLES_PER_MOTION: usize = 64;
const MAX_SPREAD: f64 = 2.0 * PI / 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointingError {
    InvalidConfiguration,
    InvalidSample,
    TimeReversed,
}

/// Experimental parameters, NOT Windows slider values or a tuned user profile.
#[derive(Debug, Clone, Copy)]
pub struct AngleConfig {
    distance: f64,
    min_gain: f64,
    max_gain: f64,
}

impl AngleConfig {
    /// Distance is in baseline pixels; gains are dimensionless multipliers.
    pub fn new(distance: f64, min_gain: f64, max_gain: f64) -> Result<Self, PointingError> {
        if !distance.is_finite()
            || !(0.000001..=1_000_000.0).contains(&distance)
            || !min_gain.is_finite()
            || !max_gain.is_finite()
            || min_gain <= 0.0
            || min_gain > max_gain
            || max_gain > 16.0
        {
            return Err(PointingError::InvalidConfiguration);
        }
        Ok(Self {
            distance,
            min_gain,
            max_gain,
        })
    }
}

impl Default for AngleConfig {
    fn default() -> Self {
        Self {
            distance: 8.0,
            min_gain: 0.25,
            max_gain: 1.0,
        }
    }
}

/// Inspectable diagnostics, useful for replay and host instrumentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleStatus {
    pub gain: f64,
    pub spread_radians: f64,
    pub samples: usize,
    /// Rejected nonfinite or excessively long segments (not normal resets).
    pub rejected_motions: u64,
}

/// Angle-Mouse-inspired FP64 reference policy.
///
/// Splits the incoming polyline at fixed ARC-LENGTH intervals. This is a
/// deliberate variant of the paper's screen-space distance threshold. Gain
/// is integrated piecewise, so subdividing a straight segment into packets
/// does not change its output, within floating-point tolerance. Coalescing
/// curved paths into straight chords still destroys information.
#[derive(Debug, Clone, Copy)]
pub struct AngleGain {
    config: AngleConfig,
    angles: [f64; HISTORY],
    next: usize,
    len: usize,
    remaining: f64,
    chord: [f64; 2],
    status: AngleStatus,
}

impl Default for AngleGain {
    fn default() -> Self {
        Self::new(AngleConfig::default())
    }
}

impl AngleGain {
    pub fn new(config: AngleConfig) -> Self {
        Self {
            config,
            angles: [0.0; HISTORY],
            next: 0,
            len: 0,
            remaining: config.distance,
            chord: [0.0; 2],
            status: AngleStatus {
                gain: config.max_gain,
                spread_radians: 0.0,
                samples: 0,
                rejected_motions: 0,
            },
        }
    }
    pub const fn status(&self) -> AngleStatus {
        self.status
    }

    fn sample(&mut self) {
        if hypot_f64(self.chord[0], self.chord[1]) < self.config.distance * 1e-9 {
            // A closed/reversing subpixel path has no direction, not angle 0.
            self.status.spread_radians = MAX_SPREAD;
            self.status.gain = self.config.min_gain;
            return;
        }
        self.angles[self.next] = atan2_f64(self.chord[1], self.chord[0]);
        self.next = (self.next + 1) % HISTORY;
        self.len = (self.len + 1).min(HISTORY);
        self.status.samples = self.len;
        if self.len < 2 {
            return;
        }

        // Newest-first Gaussian weights. Prior spread selects the weighting
        // for this sample; the new spread determines the NEXT weights.
        let sigma = 5.0 + 10.0 * (self.status.spread_radians / MAX_SPREAD).clamp(0.0, 1.0);
        let mut weights = [0.0; HISTORY];
        let (mut sum, mut squares, mut x, mut y) = (0.0, 0.0, 0.0, 0.0);
        for (age, w) in weights.iter_mut().enumerate().take(self.len) {
            let i = (age + 1) as f64;
            *w = exp_f64(-i * i / (2.0 * sigma * sigma));
            let angle = self.angles[(self.next + HISTORY - 1 - age) % HISTORY];
            sum += *w;
            squares += *w * *w;
            x += *w * cos_f64(angle);
            y += *w * sin_f64(angle);
        }
        let spread = if hypot_f64(x, y) <= sum * 1e-9 {
            // An undefined circular mean indicates incoherence, not straight travel.
            MAX_SPREAD
        } else {
            let mean = atan2_f64(y, x);
            let mut variance = 0.0;
            for (age, w) in weights.iter().enumerate().take(self.len) {
                let angle = self.angles[(self.next + HISTORY - 1 - age) % HISTORY];
                let distance = (angle - mean).abs();
                let distance = distance.min(TAU - distance);
                variance += w * distance * distance;
            }
            sqrt_f64((sum * variance / (sum * sum - squares)).max(0.0)).min(MAX_SPREAD)
        };
        self.status.spread_radians = spread;
        self.status.gain = self.config.min_gain
            + (1.0 - spread / MAX_SPREAD) * (self.config.max_gain - self.config.min_gain);
    }
}

impl MotionPolicy for AngleGain {
    /// Angular deviation depends on path shape alone, never on time or source.
    fn apply(&mut self, delta: [f64; 2], _context: MotionContext) -> [f64; 2] {
        let length = hypot_f64(delta[0], delta[1]);
        if !delta.iter().all(|v| v.is_finite())
            || !length.is_finite()
            || length > self.config.distance * (MAX_SAMPLES_PER_MOTION - 1) as f64
        {
            self.reset();
            self.status.rejected_motions = self.status.rejected_motions.saturating_add(1);
            return [0.0; 2];
        }
        if length == 0.0 {
            return [0.0; 2];
        }
        let unit = [delta[0] / length, delta[1] / length];
        let mut left = length;
        let mut out = [0.0; 2];
        // Admission above bounds this loop, even for malicious relative counts.
        for _ in 0..MAX_SAMPLES_PER_MOTION {
            if left <= 0.0 {
                break;
            }
            let step = left.min(self.remaining);
            for axis in 0..2 {
                let d = unit[axis] * step;
                out[axis] += d * self.status.gain;
                self.chord[axis] += d;
            }
            left -= step;
            self.remaining -= step;
            if self.remaining <= self.config.distance * 1e-12 {
                self.sample();
                self.chord = [0.0; 2];
                self.remaining = self.config.distance;
            }
        }
        out
    }
    fn reset(&mut self) {
        let rejected = self.status.rejected_motions;
        *self = Self::new(self.config);
        self.status.rejected_motions = rejected;
    }
}

/// Reusable point-target trial. Times are caller-supplied monotonic values;
/// the caller cancels/restarts trials on focus loss, resize or policy changes.
#[derive(Debug, Clone, Copy)]
pub struct PointingTrial {
    start: [f64; 2],
    target: [f64; 2],
    radius: f64,
    started: InputTimestamp,
    last_time: InputTimestamp,
    last: [f64; 2],
    path_length: f64,
    entries: u32,
    overshoots: u32,
    misses: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialResult {
    pub elapsed_ns: u64,
    pub path_length: f64,
    pub endpoint_error: f64,
    pub target_entries: u32,
    /// Crossings beyond the target's far edge along the start-to-target axis.
    /// Not identical to the paper's full trajectory-analysis definition.
    pub axial_overshoots: u32,
    pub missed_clicks: u32,
    /// Nominal log2(distance/diameter + 1), NOT effective Fitts throughput.
    pub nominal_difficulty: f64,
}

impl PointingTrial {
    pub fn new(
        start: [f64; 2],
        target: [f64; 2],
        radius: f64,
        now: InputTimestamp,
    ) -> Result<Self, PointingError> {
        if !valid_point(start)
            || !valid_point(target)
            || !radius.is_finite()
            || radius < 0.000001
            || radius > 1e9
            || distance(start, target) <= radius
        {
            return Err(PointingError::InvalidConfiguration);
        }
        Ok(Self {
            start,
            target,
            radius,
            started: now,
            last_time: now,
            last: start,
            path_length: 0.0,
            entries: 0,
            overshoots: 0,
            misses: 0,
        })
    }

    pub fn observe(
        &mut self,
        position: [f64; 2],
        now: InputTimestamp,
    ) -> Result<(), PointingError> {
        if !valid_point(position) {
            return Err(PointingError::InvalidSample);
        }
        if now < self.last_time {
            return Err(PointingError::TimeReversed);
        }
        self.path_length += distance(self.last, position);
        if distance(self.last, self.target) > self.radius {
            // Segment-circle intersection counts a pass through a small target
            // even when both event endpoints lie outside it.
            let d = [position[0] - self.last[0], position[1] - self.last[1]];
            let square = d[0] * d[0] + d[1] * d[1];
            let t = if square > 0.0 {
                (((self.target[0] - self.last[0]) * d[0] + (self.target[1] - self.last[1]) * d[1])
                    / square)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            if distance(
                [self.last[0] + t * d[0], self.last[1] + t * d[1]],
                self.target,
            ) <= self.radius
            {
                self.entries = self.entries.saturating_add(1);
            }
        }
        let amplitude = distance(self.start, self.target);
        let projection = |p: [f64; 2]| {
            ((p[0] - self.target[0]) * (self.target[0] - self.start[0])
                + (p[1] - self.target[1]) * (self.target[1] - self.start[1]))
                / amplitude
        };
        if projection(self.last) <= self.radius && projection(position) > self.radius {
            self.overshoots = self.overshoots.saturating_add(1);
        }
        self.last = position;
        self.last_time = now;
        Ok(())
    }

    /// A miss increments the counter and keeps the trial alive. On success
    /// the caller records the result and replaces the trial.
    pub fn click(
        &mut self,
        position: [f64; 2],
        now: InputTimestamp,
    ) -> Result<Option<TrialResult>, PointingError> {
        self.observe(position, now)?;
        let error = distance(position, self.target);
        if error > self.radius {
            self.misses = self.misses.saturating_add(1);
            return Ok(None);
        }
        Ok(Some(TrialResult {
            elapsed_ns: now.0 - self.started.0,
            path_length: self.path_length,
            endpoint_error: error,
            target_entries: self.entries,
            axial_overshoots: self.overshoots,
            missed_clicks: self.misses,
            nominal_difficulty: log2_f64(
                distance(self.start, self.target) / (2.0 * self.radius) + 1.0,
            ),
        }))
    }
}

fn valid_point(point: [f64; 2]) -> bool {
    point.iter().all(|v| v.is_finite() && v.abs() <= 1e9)
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    hypot_f64(a[0] - b[0], a[1] - b[1])
}
