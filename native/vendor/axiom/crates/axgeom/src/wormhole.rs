//! Dneg traversable-wormhole geometry and reference integration.
//!
//! The spacetime is ultrastatic and spherically symmetric,
//!
//! ```text
//! ds^2 = -dt^2 + dℓ^2 + r(ℓ)^2 (dθ^2 + sin(θ)^2 dφ^2),
//! ```
//!
//! so every null geodesic lies in a plane.  The scalar reference therefore
//! carries only `[ℓ, φ, p_ℓ]` and the conserved impact parameter.  It
//! performs no allocation and has no host-runtime dependencies.

use axmath::{abs_f32, atan_f32, cos_f32, log1p_f32, sin_f32, sqrt_f32};

use crate::integrate::velocity_verlet;

const TWO_OVER_PI: f32 = 2.0 / core::f32::consts::PI;

/// Geometry parameters for the three-parameter Dneg wormhole.
///
/// `rho` is the throat radius, `2a` is the length of the cylindrical
/// interior, and `m` controls how gently each mouth joins its asymptotically
/// flat exterior.  `m` is a lensing scale, not an ADM mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DnegParams {
    pub rho: f32,
    pub a: f32,
    pub m: f32,
}

impl DnegParams {
    /// Parameters used by the issue #33 reference scene.
    pub const REFERENCE: Self = Self {
        rho: 1.0,
        a: 2.0,
        m: 0.1,
    };

    #[inline]
    pub const fn new(rho: f32, a: f32, m: f32) -> Self {
        Self { rho, a, m }
    }

    /// Whether the parameters define the smooth Dneg geometry used here.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.rho.is_finite()
            && self.a.is_finite()
            && self.m.is_finite()
            && self.rho > 0.0
            && self.a >= 0.0
            && self.m > 0.0
    }
}

impl Default for DnegParams {
    #[inline]
    fn default() -> Self {
        Self::REFERENCE
    }
}

/// Fixed-step integration policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceConfig {
    /// Affine-parameter increment for one velocity-Verlet step.
    pub step_size: f32,
    /// Maximum number of integration steps before reporting `Unresolved`.
    pub max_steps: u32,
    /// Magnitude of `ell` at which an outward-moving ray has escaped.
    pub escape_ell: f32,
}

impl TraceConfig {
    /// Settings used by the compact reference shader.
    pub const REFERENCE: Self = Self {
        step_size: 0.1,
        max_steps: 1_000,
        escape_ell: 10.0,
    };

    #[inline]
    pub const fn new(step_size: f32, max_steps: u32, escape_ell: f32) -> Self {
        Self {
            step_size,
            max_steps,
            escape_ell,
        }
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.step_size.is_finite()
            && self.escape_ell.is_finite()
            && self.step_size > 0.0
            && self.escape_ell > 0.0
    }
}

impl Default for TraceConfig {
    #[inline]
    fn default() -> Self {
        Self::REFERENCE
    }
}

/// Planar null-ray state and its conserved impact parameter.
///
/// `state` is `[ell, phi, p_ell]`.  The sign of `impact_parameter` controls
/// the direction in which `phi` advances; its magnitude controls the radial
/// effective potential.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarRay {
    pub state: [f32; 3],
    pub impact_parameter: f32,
}

/// Orientation of a ray's conserved orbital plane in the camera's local
/// frame. Components are ordered as `[ell, transverse_y, transverse_z]`.
///
/// Spherical symmetry lets the integrator discard this orientation while it
/// advances `[ell, phi, p_ell]`; the renderer uses it only to reconstruct the
/// escaped three-dimensional direction.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarFrame {
    pub transverse_axis: [f32; 2],
}

/// Initialize the reduced planar state from a unit local ray direction.
///
/// `direction[0]` points toward increasing `ell`. The remaining components
/// span the local tangent plane. A central radial ray receives a deterministic
/// tangent axis even though its orbital plane is physically arbitrary.
#[inline]
pub fn initialize_planar_ray(
    params: &DnegParams,
    camera_ell: f32,
    direction: [f32; 3],
) -> (PlanarRay, PlanarFrame) {
    let transverse_norm = sqrt_f32(direction[1] * direction[1] + direction[2] * direction[2]);
    let transverse_axis = if transverse_norm > 0.0 {
        [
            direction[1] / transverse_norm,
            direction[2] / transverse_norm,
        ]
    } else {
        [1.0, 0.0]
    };
    let impact_parameter = radius(params, camera_ell) * transverse_norm;
    (
        PlanarRay::new(camera_ell, 0.0, direction[0], impact_parameter),
        PlanarFrame { transverse_axis },
    )
}

/// Reconstruct the local three-dimensional propagation direction of an
/// escaped reduced ray.
///
/// The returned components use the same local camera basis accepted by
/// [`initialize_planar_ray`]. On the negative side, increasing `ell` points
/// toward the throat, so the physical outward radial component has the
/// opposite sign.
#[inline]
pub fn escaped_direction(params: &DnegParams, ray: &PlanarRay, frame: &PlanarFrame) -> [f32; 3] {
    let (sin_phi, cos_phi) = (sin_f32(ray.phi()), cos_f32(ray.phi()));
    let radial_velocity = if ray.ell() < 0.0 {
        -ray.p_ell()
    } else {
        ray.p_ell()
    };
    let angular_velocity = ray.impact_parameter / radius(params, ray.ell());

    // e_r = cos(phi) e_ell + sin(phi) e_perp
    // e_phi = -sin(phi) e_ell + cos(phi) e_perp
    let axial = radial_velocity * cos_phi - angular_velocity * sin_phi;
    let transverse = radial_velocity * sin_phi + angular_velocity * cos_phi;
    let mut direction = [
        axial,
        transverse * frame.transverse_axis[0],
        transverse * frame.transverse_axis[1],
    ];
    let norm = sqrt_f32(
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2],
    );
    if norm > 0.0 {
        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;
    }
    direction
}

impl PlanarRay {
    #[inline]
    pub const fn new(ell: f32, phi: f32, p_ell: f32, impact_parameter: f32) -> Self {
        Self {
            state: [ell, phi, p_ell],
            impact_parameter,
        }
    }

    #[inline]
    pub const fn ell(self) -> f32 {
        self.state[0]
    }

    #[inline]
    pub const fn phi(self) -> f32 {
        self.state[1]
    }

    #[inline]
    pub const fn p_ell(self) -> f32 {
        self.state[2]
    }
}

/// Terminal event for a fixed-step trace.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceEvent {
    ExitNegative,
    ExitPositive,
    Unresolved,
}

/// Circumferential radius `r(ell)` of the Dneg geometry.
#[inline]
pub fn radius(params: &DnegParams, ell: f32) -> f32 {
    let exterior_distance = abs_f32(ell) - params.a;
    if exterior_distance <= 0.0 {
        return params.rho;
    }

    let x = TWO_OVER_PI * exterior_distance / params.m;
    params.rho + params.m * (x * atan_f32(x) - 0.5 * log1p_f32(x * x))
}

/// Derivative `dr/dell` of [`radius`].
#[inline]
pub fn radius_derivative(params: &DnegParams, ell: f32) -> f32 {
    let exterior_distance = abs_f32(ell) - params.a;
    if exterior_distance <= 0.0 {
        return 0.0;
    }

    let x = TWO_OVER_PI * exterior_distance / params.m;
    let magnitude = TWO_OVER_PI * atan_f32(x);
    if ell < 0.0 { -magnitude } else { magnitude }
}

/// Null-geodesic invariant `p_ell^2 + B^2 / r(ell)^2`.
///
/// With the conventional normalization `p_t = -1`, a physically initialized
/// light ray has invariant one.
#[inline]
pub fn invariant(params: &DnegParams, ray: &PlanarRay) -> f32 {
    let r = radius(params, ray.ell());
    let p_ell = ray.p_ell();
    let impact = ray.impact_parameter;
    p_ell * p_ell + impact * impact / (r * r)
}

#[inline]
fn radial_force(params: &DnegParams, ell: f32, impact_parameter: f32) -> f32 {
    let r = radius(params, ell);
    let impact_squared = impact_parameter * impact_parameter;
    impact_squared * radius_derivative(params, ell) / (r * r * r)
}

/// Advance a ray by one symmetric velocity-Verlet step.
///
/// A negative `step_size` applies the exact reverse map up to floating-point
/// rounding, which is useful for deterministic reversibility checks.
#[inline]
pub fn velocity_verlet_step(params: &DnegParams, ray: &mut PlanarRay, step_size: f32) {
    let ell_old = ray.state[0];
    let r_old = radius(params, ell_old);
    let half_step = 0.5 * step_size;
    let impact_parameter = ray.impact_parameter;
    let mut position = [ell_old];
    let mut velocity = [ray.state[2]];
    velocity_verlet(
        &mut position,
        &mut velocity,
        step_size,
        &mut |position: &[f32; 1]| [radial_force(params, position[0], impact_parameter)],
    );
    let ell_new = position[0];
    let r_new = radius(params, ell_new);
    let p_new = velocity[0];

    // Trapezoidal integration keeps the cyclic angle update symmetric under
    // step reversal while the radial state uses kick-drift-kick Verlet.
    let angular_rate_sum = ray.impact_parameter * (1.0 / (r_old * r_old) + 1.0 / (r_new * r_new));

    ray.state[0] = ell_new;
    ray.state[1] += half_step * angular_rate_sum;
    ray.state[2] = p_new;
}

#[inline]
fn terminal_event(config: &TraceConfig, ray: &PlanarRay) -> Option<TraceEvent> {
    if ray.ell() <= -config.escape_ell && ray.p_ell() < 0.0 {
        Some(TraceEvent::ExitNegative)
    } else if ray.ell() >= config.escape_ell && ray.p_ell() > 0.0 {
        Some(TraceEvent::ExitPositive)
    } else {
        None
    }
}

/// Trace until the ray escapes one asymptotic region or exhausts the budget.
///
/// Once an exit is observed, the state is frozen at the first sampled point
/// outside the escape surface.  Calling this function again with that state
/// therefore returns the same event without advancing it.
pub fn trace_fixed(params: &DnegParams, config: &TraceConfig, ray: &mut PlanarRay) -> TraceEvent {
    if !params.is_valid() || !config.is_valid() || config.escape_ell <= params.a {
        return TraceEvent::Unresolved;
    }

    if let Some(event) = terminal_event(config, ray) {
        return event;
    }

    for _ in 0..config.max_steps {
        velocity_verlet_step(params, ray, config.step_size);
        if let Some(event) = terminal_event(config, ray) {
            return event;
        }
    }

    TraceEvent::Unresolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use axmath::{abs_f32, sqrt_f32};

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let error = abs_f32(actual - expected);
        assert!(
            error <= tolerance,
            "actual={actual:?}, expected={expected:?}, error={error:?}, tolerance={tolerance:?}"
        );
    }

    fn inward_null_ray(params: &DnegParams, ell: f32, impact: f32) -> PlanarRay {
        let r = radius(params, ell);
        let p_ell = -sqrt_f32(1.0 - impact * impact / (r * r));
        PlanarRay::new(ell, 0.0, p_ell, impact)
    }

    #[test]
    fn camera_reduction_preserves_null_normalization_and_plane_orientation() {
        let params = DnegParams::REFERENCE;
        let (ray, frame) = initialize_planar_ray(&params, 5.0, [-0.8, 0.36, 0.48]);

        assert_close(invariant(&params, &ray), 1.0, 2.0e-6);
        assert_close(frame.transverse_axis[0], 0.6, 2.0e-6);
        assert_close(frame.transverse_axis[1], 0.8, 2.0e-6);

        let (central, central_frame) = initialize_planar_ray(&params, 5.0, [-1.0, 0.0, 0.0]);
        assert_eq!(central.impact_parameter, 0.0);
        assert_eq!(central_frame.transverse_axis, [1.0, 0.0]);
    }

    #[test]
    fn escaped_direction_is_unit_length_and_rotates_with_the_orbital_plane() {
        let params = DnegParams::REFERENCE;
        let ray = PlanarRay::new(-8.0, 0.7, -0.9, 0.4);
        let first = escaped_direction(
            &params,
            &ray,
            &PlanarFrame {
                transverse_axis: [1.0, 0.0],
            },
        );
        let second = escaped_direction(
            &params,
            &ray,
            &PlanarFrame {
                transverse_axis: [0.0, 1.0],
            },
        );

        let norm = |value: [f32; 3]| {
            sqrt_f32(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
        };
        assert_close(norm(first), 1.0, 2.0e-6);
        assert_close(norm(second), 1.0, 2.0e-6);
        assert_close(first[0], second[0], 2.0e-6);
        assert_close(first[1], second[2], 2.0e-6);
        assert_close(first[2], 0.0, 2.0e-6);
        assert_close(second[1], 0.0, 2.0e-6);
    }

    #[test]
    fn metric_is_even_smooth_and_flat_through_the_throat() {
        let params = DnegParams::REFERENCE;
        assert!(params.is_valid());

        for ell in [0.0, 0.5, 1.999, 2.0, 2.001, 3.0, 8.0] {
            assert_close(radius(&params, ell), radius(&params, -ell), 2.0e-6);
            assert_close(
                radius_derivative(&params, ell),
                -radius_derivative(&params, -ell),
                2.0e-6,
            );
        }

        assert_eq!(radius(&params, -params.a), params.rho);
        assert_eq!(radius(&params, 0.0), params.rho);
        assert_eq!(radius(&params, params.a), params.rho);
        assert_eq!(radius_derivative(&params, -params.a), 0.0);
        assert_eq!(radius_derivative(&params, params.a), 0.0);

        let epsilon = 1.0e-4;
        assert_close(radius(&params, params.a + epsilon), params.rho, 2.0e-6);
        assert_close(radius_derivative(&params, params.a + epsilon), 0.0, 5.0e-4);
    }

    #[test]
    fn radial_ray_traverses_in_a_straight_line() {
        let params = DnegParams::REFERENCE;
        let config = TraceConfig::new(0.125, 128, 6.0);
        let mut ray = PlanarRay::new(5.0, 0.75, -1.0, 0.0);

        assert_eq!(
            trace_fixed(&params, &config, &mut ray),
            TraceEvent::ExitNegative
        );
        assert_eq!(ray.state, [-6.0, 0.75, -1.0]);
        assert_eq!(invariant(&params, &ray), 1.0);
    }

    #[test]
    fn impact_parameter_separates_traversal_from_reflection() {
        let params = DnegParams::REFERENCE;
        let config = TraceConfig::new(0.02, 4_000, 6.0);

        let mut traversing = inward_null_ray(&params, 5.0, 0.75 * params.rho);
        let mut reflecting = inward_null_ray(&params, 5.0, 1.25 * params.rho);

        assert_eq!(
            trace_fixed(&params, &config, &mut traversing),
            TraceEvent::ExitNegative
        );
        assert_eq!(
            trace_fixed(&params, &config, &mut reflecting),
            TraceEvent::ExitPositive
        );
        assert!(traversing.p_ell() < 0.0);
        assert!(reflecting.p_ell() > 0.0);
        assert_close(invariant(&params, &traversing), 1.0, 2.0e-4);
        assert_close(invariant(&params, &reflecting), 1.0, 2.0e-4);
    }

    #[test]
    fn exited_state_is_frozen() {
        let params = DnegParams::REFERENCE;
        let config = TraceConfig::new(0.02, 4_000, 6.0);
        let mut ray = inward_null_ray(&params, 5.0, 1.25 * params.rho);

        assert_eq!(
            trace_fixed(&params, &config, &mut ray),
            TraceEvent::ExitPositive
        );
        let frozen = ray;
        assert_eq!(
            trace_fixed(&params, &config, &mut ray),
            TraceEvent::ExitPositive
        );
        assert_eq!(ray, frozen);
    }

    #[test]
    fn verlet_step_respects_reflection_and_time_symmetry() {
        let params = DnegParams::REFERENCE;
        let mut ray = PlanarRay::new(3.25, 0.2, -0.65, 0.8);
        let mut reflected = PlanarRay::new(-3.25, 0.2, 0.65, 0.8);

        for _ in 0..128 {
            velocity_verlet_step(&params, &mut ray, 0.01);
            velocity_verlet_step(&params, &mut reflected, 0.01);
        }
        assert_close(reflected.ell(), -ray.ell(), 2.0e-6);
        assert_close(reflected.phi(), ray.phi(), 2.0e-6);
        assert_close(reflected.p_ell(), -ray.p_ell(), 2.0e-6);

        let start = ray;
        for _ in 0..64 {
            velocity_verlet_step(&params, &mut ray, 0.01);
        }
        for _ in 0..64 {
            velocity_verlet_step(&params, &mut ray, -0.01);
        }
        assert_close(ray.ell(), start.ell(), 2.0e-5);
        assert_close(ray.phi(), start.phi(), 2.0e-5);
        assert_close(ray.p_ell(), start.p_ell(), 2.0e-5);
    }
}
