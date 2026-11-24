//! The standard Dart cursor: one rigid pointer whose pose carries motion.
//!
//! The portable compositor owns the state and raster vocabulary, while the
//! final runtime supplies position samples, time and the actual blend route.
//! No allocation, clock, event loop or backend policy lives here.

use crate::glyph::{Glyph, LayerPose, rasterize_layers};
use axgfx::{CursorHotspot, Extent2d};
use axmath::{cos_f32, exp_f32, expm1_f32, sin_cos_f32, sqrt_f32, tanh_f32};

pub const DART_WIDTH: usize = 36;
pub const DART_HEIGHT: usize = 36;
pub const DART_PIXELS: usize = DART_WIDTH * DART_HEIGHT;
pub const DART_RGBA_BYTES: usize = DART_PIXELS * 4;
pub const DART_HOTSPOT: CursorHotspot = CursorHotspot { x: 18, y: 6 };

pub(crate) const DEFAULT_DT: f32 = 1.0 / 60.0;
pub(crate) const MIN_DT: f32 = 1.0 / 240.0;
pub(crate) const MAX_DT: f32 = 1.0 / 20.0;
const VELOCITY_FILTER_RATE: f32 = 18.0;
const ACCELERATION_FILTER_RATE: f32 = 11.0;
const GRAVITY: f32 = 62.0;
const ACCEL_LEAN: f32 = 0.0012;
const VELOCITY_LEAN: f32 = 0.0775;
const LATERAL_SATURATION: f32 = 107.0;
const DAMPING: f32 = 8.5;
const HORIZON_BARRIER: f32 = 900.0;
const HORIZON_DAMPING: f32 = 16.0;
const FACING_STIFFNESS: f32 = 230.0;
const FACING_DAMPING: f32 = 18.0;
const SCALE_STIFFNESS: f32 = 260.0;
const SCALE_DAMPING: f32 = 22.0;
pub(crate) const CENTROID_PIVOT: f32 = 7.3667 * 0.5;
pub(crate) const PATH: [[f32; 2]; 4] = [[0.0, 0.0], [-4.7, 12.8], [0.0, 9.3], [4.7, 12.8]];

/// A caller-provided Dart raster could not represent the requested physical
/// sampling grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DartRasterError {
    /// Physical pixels per cursor unit was zero, negative, or non-finite.
    InvalidScale,
    /// The hotspot did not lie inside the destination image.
    HotspotOutsideExtent,
    /// The RGBA destination was not exactly four bytes per physical pixel.
    BufferLength,
}

/// One resolved standard-Dart pose. The tip returns exactly to the hotspot at
/// rest; leaning rotates around a half-centroid pivot to avoid a wide tail arc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DartPose {
    angle: f32,
    scale: f32,
}

impl DartPose {
    pub const REST: Self = Self {
        angle: 0.0,
        scale: 1.0,
    };

    pub fn angle(self) -> f32 {
        fold_up(self.angle)
    }

    pub fn scale(self) -> f32 {
        self.scale
    }

    /// Rasterize straight-alpha sRGB RGBA into caller-owned fixed storage.
    /// RGB remains unassociated; consumers that blend premultiplied surfaces
    /// must multiply each color channel by the emitted alpha first.
    pub fn rasterize(self, rgba: &mut [u8; DART_RGBA_BYTES]) {
        self.rasterize_pixels(DART_WIDTH, DART_HEIGHT, DART_HOTSPOT, 1.0, rgba);
    }

    /// Rasterize the Dart SDF directly at a physical output resolution.
    ///
    /// `pixels_per_unit` is the presentation scale and antialiasing density,
    /// not a post-render bitmap multiplier. Every destination pixel therefore
    /// receives a fresh signed-distance sample. `extent` and `hotspot` remain
    /// explicit because a theme may reserve asymmetric transparent padding.
    pub fn rasterize_at_scale(
        self,
        extent: Extent2d,
        hotspot: CursorHotspot,
        pixels_per_unit: f32,
        rgba: &mut [u8],
    ) -> Result<(), DartRasterError> {
        if !pixels_per_unit.is_finite() || pixels_per_unit <= 0.0 {
            return Err(DartRasterError::InvalidScale);
        }
        if u32::from(hotspot.x) >= extent.width() || u32::from(hotspot.y) >= extent.height() {
            return Err(DartRasterError::HotspotOutsideExtent);
        }
        let width = extent.width() as usize;
        let height = extent.height() as usize;
        if width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            != Some(rgba.len())
        {
            return Err(DartRasterError::BufferLength);
        }
        self.rasterize_pixels(width, height, hotspot, pixels_per_unit, rgba);
        Ok(())
    }

    fn rasterize_pixels(
        self,
        width: usize,
        height: usize,
        hotspot: CursorHotspot,
        pixels_per_unit: f32,
        rgba: &mut [u8],
    ) {
        rasterize_layers(
            &[self.layer()],
            width,
            height,
            [f32::from(hotspot.x), f32::from(hotspot.y)],
            pixels_per_unit,
            rgba,
        );
    }

    /// The Dart as one layer of the shared cursor raster. Leaning rotates
    /// about the half-centroid pivot; the squash scales about the tip.
    pub(crate) fn layer(self) -> LayerPose<'static> {
        LayerPose {
            glyph: DART_GLYPH,
            angle: self.angle(),
            pivot: [0.0, CENTROID_PIVOT],
            scale: [self.scale, self.scale],
            offset: [0.0, 0.0],
            opacity: 1.0,
        }
    }
}

pub(crate) const DART_GLYPH: Glyph<'static> = Glyph {
    polygons: &[&PATH],
    capsules: &[],
    seams: &[],
    keyline: crate::glyph::OUTLINE_WIDTH,
    arcs: &[],
    body: crate::glyph::BODY,
    shadow: true,
};

/// Exact one-step transition of a unit-mass damped spring about its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DampedOscillatorStep {
    xx: f32,
    xv: f32,
    vx: f32,
    vv: f32,
}

impl DampedOscillatorStep {
    pub(crate) fn new(stiffness: f32, damping: f32, dt: f32) -> Self {
        let sigma = damping * 0.5;
        let discriminant = stiffness - sigma * sigma;
        if discriminant <= stiffness * 1.0e-6 {
            // Critically damped (stiffness = sigma²): the underdamped form
            // would divide by a vanishing frequency.
            let envelope = exp_f32(-sigma * dt);
            return Self {
                xx: envelope * (1.0 + sigma * dt),
                xv: envelope * dt,
                vx: -envelope * stiffness * dt,
                vv: envelope * (1.0 - sigma * dt),
            };
        }
        let damped_frequency = sqrt_f32(discriminant);
        let (sin, cos) = sin_cos_f32(damped_frequency * dt);
        let envelope = exp_f32(-sigma * dt);
        let sin_over_frequency = sin / damped_frequency;
        Self {
            xx: envelope * (cos + sigma * sin_over_frequency),
            xv: envelope * sin_over_frequency,
            vx: -envelope * stiffness * sin_over_frequency,
            vv: envelope * (cos - sigma * sin_over_frequency),
        }
    }

    pub(crate) fn apply(self, displacement: f32, velocity: f32) -> (f32, f32) {
        (
            self.xx * displacement + self.xv * velocity,
            self.vx * displacement + self.vv * velocity,
        )
    }

    #[cfg(test)]
    fn determinant(self) -> f32 {
        self.xx * self.vv - self.xv * self.vx
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DampedVelocityStep {
    velocity_decay: f32,
    velocity_force: f32,
    position_velocity: f32,
    position_force: f32,
}

impl DampedVelocityStep {
    fn new(damping: f32, dt: f32) -> Self {
        let one_minus_decay = -expm1_f32(-damping * dt);
        let integrated_decay = one_minus_decay / damping;
        Self {
            velocity_decay: 1.0 - one_minus_decay,
            velocity_force: integrated_decay,
            position_velocity: integrated_decay,
            position_force: (dt - integrated_decay) / damping,
        }
    }

    fn apply(self, position: f32, velocity: f32, force: f32) -> (f32, f32) {
        (
            position + self.position_velocity * velocity + self.position_force * force,
            self.velocity_decay * velocity + self.velocity_force * force,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DartStep {
    dt: f32,
    inverse_dt: f32,
    velocity_mix: f32,
    acceleration_mix: f32,
    pendulum: DampedVelocityStep,
    horizon_pendulum: DampedVelocityStep,
    facing: DampedOscillatorStep,
    scale: DampedOscillatorStep,
}

impl DartStep {
    const INVALID: Self = Self {
        dt: f32::NAN,
        inverse_dt: 0.0,
        velocity_mix: 0.0,
        acceleration_mix: 0.0,
        pendulum: DampedVelocityStep {
            velocity_decay: 0.0,
            velocity_force: 0.0,
            position_velocity: 0.0,
            position_force: 0.0,
        },
        horizon_pendulum: DampedVelocityStep {
            velocity_decay: 0.0,
            velocity_force: 0.0,
            position_velocity: 0.0,
            position_force: 0.0,
        },
        facing: DampedOscillatorStep {
            xx: 0.0,
            xv: 0.0,
            vx: 0.0,
            vv: 0.0,
        },
        scale: DampedOscillatorStep {
            xx: 0.0,
            xv: 0.0,
            vx: 0.0,
            vv: 0.0,
        },
    };

    fn new(dt: f32) -> Self {
        Self {
            dt,
            inverse_dt: dt.recip(),
            velocity_mix: -expm1_f32(-VELOCITY_FILTER_RATE * dt),
            acceleration_mix: -expm1_f32(-ACCELERATION_FILTER_RATE * dt),
            pendulum: DampedVelocityStep::new(DAMPING, dt),
            horizon_pendulum: DampedVelocityStep::new(HORIZON_DAMPING, dt),
            facing: DampedOscillatorStep::new(FACING_STIFFNESS, FACING_DAMPING, dt),
            scale: DampedOscillatorStep::new(SCALE_STIFFNESS, SCALE_DAMPING, dt),
        }
    }
}

/// Allocation-free Point/Press dynamics. Only lateral motion banks the
/// standard dart, matching the supplied reference implementation exactly;
/// vertical travel does not rotate the pointer into a pinwheel.
#[derive(Clone, Copy, Debug)]
pub struct DartCursor {
    sample_x: f32,
    velocity_x: f32,
    acceleration_x: f32,
    angle: f32,
    angular_velocity: f32,
    facing: f32,
    facing_velocity: f32,
    scale: f32,
    scale_velocity: f32,
    step: DartStep,
    active: bool,
}

impl DartCursor {
    pub const fn new(position_x: f32) -> Self {
        Self {
            sample_x: position_x,
            velocity_x: 0.0,
            acceleration_x: 0.0,
            angle: 0.0,
            angular_velocity: 0.0,
            facing: 0.0,
            facing_velocity: 0.0,
            scale: 1.0,
            scale_velocity: 0.0,
            step: DartStep::INVALID,
            active: false,
        }
    }

    pub fn pose(&self) -> DartPose {
        DartPose {
            angle: self.facing,
            scale: self.scale,
        }
    }

    pub const fn is_active(&self) -> bool {
        self.active
    }

    pub fn press(&mut self) {
        self.kick(7.5);
    }

    pub fn release(&mut self) {
        self.kick(-4.0);
    }

    pub fn reset(&mut self, position_x: f32) {
        *self = Self::new(position_x);
    }

    /// Consume one runtime-supplied position/time sample and advance the
    /// pendulum with bounded nonlinear forcing and exact linear damping.
    pub fn advance(&mut self, position_x: f32, dt: f32) -> bool {
        let dt = if dt.is_finite() {
            dt.clamp(MIN_DT, MAX_DT)
        } else {
            DEFAULT_DT
        };
        if self.step.dt != dt {
            self.step = DartStep::new(dt);
        }
        let step = self.step;
        let delta = position_x - self.sample_x;
        self.sample_x = position_x;
        let velocity_delta = (delta * step.inverse_dt - self.velocity_x) * step.velocity_mix;
        self.velocity_x += velocity_delta;
        self.acceleration_x = self.acceleration_x * (1.0 - step.acceleration_mix)
            + velocity_delta * step.acceleration_mix * step.inverse_dt;

        let raw = ACCEL_LEAN * self.acceleration_x + VELOCITY_LEAN * self.velocity_x;
        let lateral = LATERAL_SATURATION * tanh_f32(raw / LATERAL_SATURATION);
        let (angle_sin, angle_cos) = sin_cos_f32(self.angle);
        let mut acceleration = -GRAVITY * angle_sin + lateral * angle_cos;
        let past_horizon = self.angle.abs() - core::f32::consts::FRAC_PI_2;
        let pendulum = if past_horizon > 0.0 {
            acceleration -= HORIZON_BARRIER * past_horizon * self.angle.signum();
            step.horizon_pendulum
        } else {
            step.pendulum
        };
        (self.angle, self.angular_velocity) =
            pendulum.apply(self.angle, self.angular_velocity, acceleration);

        // The visible dart is its own angular spring. The assembly carries
        // physical momentum; facing follows it with the reference's measured
        // lag, so a click bends and recovers instead of rotating as one rigid
        // pointer-plus-pendulum value.
        let target_facing = fold_up(self.angle);
        let facing_displacement = shortest_angle(self.facing - target_facing);
        let (facing_displacement, facing_velocity) =
            step.facing.apply(facing_displacement, self.facing_velocity);
        self.facing = target_facing + facing_displacement;
        self.facing_velocity = facing_velocity;

        let (scale_displacement, scale_velocity) =
            step.scale.apply(self.scale - 1.0, self.scale_velocity);
        self.scale = 1.0 + scale_displacement;
        self.scale_velocity = scale_velocity;

        self.active = delta != 0.0
            || self.velocity_x.abs() >= 0.02
            || self.acceleration_x.abs() >= 0.2
            || self.angle.abs() >= 0.0005
            || self.angular_velocity.abs() >= 0.005
            || self.facing.abs() >= 0.0005
            || self.facing_velocity.abs() >= 0.005
            || (self.scale - 1.0).abs() >= 0.0005
            || self.scale_velocity.abs() >= 0.005;
        if !self.active {
            self.velocity_x = 0.0;
            self.acceleration_x = 0.0;
            self.angle = 0.0;
            self.angular_velocity = 0.0;
            self.facing = 0.0;
            self.facing_velocity = 0.0;
            self.scale = 1.0;
            self.scale_velocity = 0.0;
        }
        self.active
    }

    fn kick(&mut self, impulse: f32) {
        self.angular_velocity += impulse;
        self.scale_velocity -= impulse.abs() * 0.020;
        self.active = true;
    }
}

pub(crate) fn fold_up(angle: f32) -> f32 {
    let mut wrapped = angle % core::f32::consts::TAU;
    if wrapped > core::f32::consts::PI {
        wrapped -= core::f32::consts::TAU;
    } else if wrapped <= -core::f32::consts::PI {
        wrapped += core::f32::consts::TAU;
    }
    if cos_f32(wrapped) < 0.0 {
        wrapped = core::f32::consts::PI - wrapped;
    }
    let mut reflected = wrapped % core::f32::consts::TAU;
    if reflected > core::f32::consts::PI {
        reflected -= core::f32::consts::TAU;
    } else if reflected <= -core::f32::consts::PI {
        reflected += core::f32::consts::TAU;
    }
    reflected
}

fn shortest_angle(mut angle: f32) -> f32 {
    angle %= core::f32::consts::TAU;
    if angle > core::f32::consts::PI {
        angle -= core::f32::consts::TAU;
    } else if angle < -core::f32::consts::PI {
        angle += core::f32::consts::TAU;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_dart_is_colored_antialiased_and_padded() {
        let mut rgba = [0; DART_RGBA_BYTES];
        DartPose::REST.rasterize(&mut rgba);
        assert!(rgba.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
        assert!(rgba.chunks_exact(4).any(|p| p[1] > p[0] && p[3] > 200));
        assert!(rgba.chunks_exact(4).any(|p| p[0] > 220 && p[3] > 100));
        for x in 0..DART_WIDTH {
            assert_eq!(rgba[x * 4 + 3], 0);
            assert_eq!(rgba[((DART_HEIGHT - 1) * DART_WIDTH + x) * 4 + 3], 0);
        }
        for y in 0..DART_HEIGHT {
            assert_eq!(rgba[(y * DART_WIDTH) * 4 + 3], 0);
            assert_eq!(rgba[(y * DART_WIDTH + DART_WIDTH - 1) * 4 + 3], 0);
        }
    }

    #[test]
    fn physical_scale_samples_the_sdf_at_destination_resolution() {
        const WIDTH: usize = DART_WIDTH * 3 / 2;
        const HEIGHT: usize = DART_HEIGHT * 3 / 2;
        const HOTSPOT: CursorHotspot = CursorHotspot {
            x: DART_HOTSPOT.x * 3 / 2,
            y: DART_HOTSPOT.y * 3 / 2,
        };
        let extent = Extent2d::new(WIDTH as u32, HEIGHT as u32).unwrap();
        let mut rgba = [0; WIDTH * HEIGHT * 4];
        DartPose::REST
            .rasterize_at_scale(extent, HOTSPOT, 1.5, &mut rgba)
            .unwrap();

        let tip = (usize::from(HOTSPOT.y) * WIDTH + usize::from(HOTSPOT.x)) * 4;
        assert_eq!(&rgba[tip..tip + 4], &[0xf9, 0xf4, 0xed, 0xff]);
        assert!(rgba.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
        for x in 0..WIDTH {
            assert_eq!(rgba[x * 4 + 3], 0);
            assert_eq!(rgba[((HEIGHT - 1) * WIDTH + x) * 4 + 3], 0);
        }
        for y in 0..HEIGHT {
            assert_eq!(rgba[(y * WIDTH) * 4 + 3], 0);
            assert_eq!(rgba[(y * WIDTH + WIDTH - 1) * 4 + 3], 0);
        }
    }

    #[test]
    fn physical_raster_contract_refuses_ambiguous_geometry() {
        let extent = Extent2d::new(DART_WIDTH as u32, DART_HEIGHT as u32).unwrap();
        let mut exact = [0; DART_RGBA_BYTES];
        assert_eq!(
            DartPose::REST.rasterize_at_scale(extent, DART_HOTSPOT, 0.0, &mut exact),
            Err(DartRasterError::InvalidScale)
        );
        assert_eq!(
            DartPose::REST.rasterize_at_scale(
                extent,
                CursorHotspot {
                    x: DART_WIDTH as u16,
                    y: DART_HOTSPOT.y,
                },
                1.0,
                &mut exact,
            ),
            Err(DartRasterError::HotspotOutsideExtent)
        );
        assert_eq!(
            DartPose::REST.rasterize_at_scale(
                extent,
                DART_HOTSPOT,
                1.0,
                &mut exact[..DART_RGBA_BYTES - 1],
            ),
            Err(DartRasterError::BufferLength)
        );
    }

    #[test]
    fn one_x_physical_raster_is_the_legacy_fixed_raster() {
        let extent = Extent2d::new(DART_WIDTH as u32, DART_HEIGHT as u32).unwrap();
        let mut fixed = [0; DART_RGBA_BYTES];
        let mut physical = [0; DART_RGBA_BYTES];
        DartPose::REST.rasterize(&mut fixed);
        DartPose::REST
            .rasterize_at_scale(extent, DART_HOTSPOT, 1.0, &mut physical)
            .unwrap();
        assert_eq!(physical, fixed);
    }

    #[test]
    fn rest_tip_is_centered_on_the_declared_hotspot_pixel() {
        let mut rgba = [0; DART_RGBA_BYTES];
        DartPose::REST.rasterize(&mut rgba);
        let at = (usize::from(DART_HOTSPOT.y) * DART_WIDTH + usize::from(DART_HOTSPOT.x)) * 4;

        let p = [
            DART_HOTSPOT.x as f32 - DART_HOTSPOT.x as f32,
            DART_HOTSPOT.y as f32 - DART_HOTSPOT.y as f32,
        ];
        assert_eq!(p, [0.0, 0.0]);
        assert_eq!(crate::glyph::glyph_distance(DART_GLYPH, p), 0.0);
        assert_eq!(&rgba[at..at + 4], &[0xf9, 0xf4, 0xed, 0xff]);
    }

    #[test]
    fn fixed_extent_contains_every_legal_facing_and_shadow() {
        for (step, scale) in (-8..=8).flat_map(|step| [(step, 1.0), (step, 1.25)]) {
            let mut rgba = [0; DART_RGBA_BYTES];
            DartPose {
                angle: step as f32 * core::f32::consts::FRAC_PI_2 / 8.0,
                scale,
            }
            .rasterize(&mut rgba);
            for x in 0..DART_WIDTH {
                assert_eq!(rgba[x * 4 + 3], 0, "top step {step}, scale {scale}");
                assert_eq!(
                    rgba[((DART_HEIGHT - 1) * DART_WIDTH + x) * 4 + 3],
                    0,
                    "bottom step {step}, scale {scale}"
                );
            }
            for y in 0..DART_HEIGHT {
                assert_eq!(
                    rgba[(y * DART_WIDTH) * 4 + 3],
                    0,
                    "left step {step}, scale {scale}"
                );
                assert_eq!(
                    rgba[(y * DART_WIDTH + DART_WIDTH - 1) * 4 + 3],
                    0,
                    "right step {step}, scale {scale}"
                );
            }
        }
    }

    #[test]
    fn horizon_is_reflected_instead_of_clamped() {
        let a = fold_up(2.0 * core::f32::consts::FRAC_PI_2 / 3.0);
        let b = fold_up(4.0 * core::f32::consts::FRAC_PI_2 / 3.0);
        assert!(cos_f32(a) >= 0.0);
        assert!(cos_f32(b) >= 0.0);
        assert!(a.abs() < core::f32::consts::FRAC_PI_2);
        assert!(b.abs() < core::f32::consts::FRAC_PI_2);
        assert_ne!(a, core::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn motion_and_press_settle_to_an_exact_idle_pose() {
        let mut cursor = DartCursor::new(100.0);
        assert!(cursor.advance(140.0, DEFAULT_DT));
        cursor.press();
        cursor.release();
        for _ in 0..900 {
            cursor.advance(140.0, DEFAULT_DT);
            if !cursor.is_active() {
                break;
            }
        }
        assert!(!cursor.is_active());
        assert_eq!(cursor.pose(), DartPose::REST);
    }

    #[test]
    fn facing_spring_lags_the_assembly_and_preserves_motion_direction() {
        let mut pressed = DartCursor::new(0.0);
        pressed.press();
        assert!(pressed.advance(0.0, DEFAULT_DT));
        assert!(pressed.angle > 0.0);
        assert!(pressed.pose().angle() > 0.0);
        assert!(pressed.pose().angle() < pressed.angle);
        assert!(pressed.pose().scale() < 1.0);

        let mut right = DartCursor::new(0.0);
        let mut left = DartCursor::new(0.0);
        right.advance(40.0, DEFAULT_DT);
        left.advance(-40.0, DEFAULT_DT);
        assert!(right.pose().angle() > 0.0);
        assert!(left.pose().angle() < 0.0);
    }

    #[test]
    fn invalid_and_extreme_frame_deltas_remain_finite() {
        let mut a = DartCursor::new(0.0);
        let mut b = a;
        a.advance(20.0, f32::NAN);
        b.advance(20.0, DEFAULT_DT);
        assert_eq!(a.pose(), b.pose());
        for dt in [0.0, f32::INFINITY, 10.0] {
            a.advance(20.0, dt);
            assert!(a.pose().angle.is_finite());
            assert!(a.pose().scale.is_finite());
        }
    }

    #[test]
    fn affine_filter_matches_the_original_difference_form() {
        for (velocity, acceleration, delta, dt) in [
            (0.0, 0.0, 40.0, DEFAULT_DT),
            (120.0, -35.0, -7.5, MIN_DT),
            (-44.0, 18.0, 0.25, MAX_DT),
        ] {
            let step = DartStep::new(dt);
            let next_velocity = velocity + (delta * step.inverse_dt - velocity) * step.velocity_mix;
            let next_acceleration = acceleration * (1.0 - step.acceleration_mix)
                + (next_velocity - velocity) * step.acceleration_mix * step.inverse_dt;

            let reference_velocity = velocity + (delta / dt - velocity) * step.velocity_mix;
            let reference_acceleration = acceleration
                + ((reference_velocity - velocity) / dt - acceleration) * step.acceleration_mix;
            assert!((next_velocity - reference_velocity).abs() <= 2.0e-4);
            assert!((next_acceleration - reference_acceleration).abs() <= 2.0e-3);
        }
    }

    #[test]
    fn exact_spring_transition_stays_contractive_at_the_largest_step() {
        for (stiffness, damping) in [
            (FACING_STIFFNESS, FACING_DAMPING),
            (SCALE_STIFFNESS, SCALE_DAMPING),
        ] {
            let transition = DampedOscillatorStep::new(stiffness, damping, MAX_DT);
            let expected = exp_f32(-damping * MAX_DT);
            assert!((transition.determinant() - expected).abs() <= 2.0e-6);
            assert!(transition.determinant() > 0.0);
            assert!(transition.determinant() < 1.0);
        }
    }

    #[test]
    fn fixed_timestep_reuses_cached_transcendental_coefficients() {
        let mut cursor = DartCursor::new(0.0);
        cursor.advance(10.0, DEFAULT_DT);
        let cached = cursor.step;
        cursor.advance(10.0, DEFAULT_DT);
        assert_eq!(cursor.step, cached);

        cursor.advance(10.0, MAX_DT);
        assert_eq!(cursor.step.dt, MAX_DT);
        assert_ne!(cursor.step, cached);
    }

    #[test]
    fn repeated_maximum_steps_settle_without_scale_instability() {
        let mut cursor = DartCursor::new(0.0);
        cursor.scale = 0.75;
        cursor.scale_velocity = -8.0;
        cursor.active = true;
        for _ in 0..600 {
            cursor.advance(0.0, MAX_DT);
            assert!(cursor.scale.is_finite());
            assert!(cursor.scale_velocity.is_finite());
        }
        assert!((cursor.scale - 1.0).abs() < 0.001);
        assert!(cursor.scale_velocity.abs() < 0.001);
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// Digests captured from the Dart's own raster before it moved onto the
    /// shared cursor-family glyph engine: the shipping Dart is byte-identical.
    #[test]
    fn shared_glyph_raster_preserves_the_shipping_dart_bytes() {
        let mut cursor = DartCursor::new(0.0);
        let mut x = 0.0f32;
        let mut poses = std::vec![DartPose::REST];
        for i in 0..400 {
            if i < 50 {
                x += 7.0;
            } else if (150..190).contains(&i) {
                x -= 11.0;
            }
            if i == 100 || i == 260 {
                cursor.press();
            }
            if i == 300 {
                cursor.release();
            }
            cursor.advance(x, 1.0 / 240.0);
            if i % 20 == 7 {
                poses.push(cursor.pose());
            }
        }
        for (ppu, side, hotspot, digest) in [
            (
                1.0,
                36,
                CursorHotspot { x: 18, y: 6 },
                0x3e2e_8583_8a5b_804a,
            ),
            (
                1.5,
                54,
                CursorHotspot { x: 27, y: 9 },
                0xd42f_eb31_efd1_3b69,
            ),
            (
                3.0,
                108,
                CursorHotspot { x: 54, y: 18 },
                0xad17_9f60_0c3d_89ad,
            ),
        ] {
            let extent = Extent2d::new(side, side).unwrap();
            let mut all = std::vec::Vec::new();
            for pose in &poses {
                let mut rgba = std::vec![0; side as usize * side as usize * 4];
                pose.rasterize_at_scale(extent, hotspot, ppu, &mut rgba)
                    .unwrap();
                all.extend_from_slice(&rgba);
            }
            assert_eq!(poses.len(), 21);
            assert_eq!(fnv1a(&all), digest, "{ppu} px/unit");
        }
    }

    #[test]
    fn critically_damped_transition_is_exact_and_contractive() {
        let critical = DampedOscillatorStep::new(900.0, 60.0, MAX_DT);
        assert!(critical.xx.is_finite() && critical.vx.is_finite());
        assert!(critical.determinant() > 0.0 && critical.determinant() < 1.0);
        let (mut x, mut v) = (1.0f32, 0.0f32);
        let mut previous = x;
        for _ in 0..120 {
            (x, v) = DampedOscillatorStep::new(900.0, 60.0, MIN_DT).apply(x, v);
            assert!(
                x <= previous && x >= 0.0,
                "critical damping never overshoots"
            );
            previous = x;
        }
        assert!(x < 1.0e-3 && v.abs() < 1.0e-2);
    }
}
