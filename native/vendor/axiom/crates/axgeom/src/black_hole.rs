//! Kerr null-geodesic reference in Cartesian Kerr-Schild coordinates.
//!
//! The fixed-step path is intentionally small and allocation-free. It is the
//! scalar correctness oracle for the branchless `.ax` graph, not a host-side
//! production renderer.

use axmath::{abs_f32, cbrt_f32, cos_f32, sin_f32, sqrt_f32};

use crate::integrate::explicit_midpoint;

const DIM: usize = 4;
const SINGULAR_EPSILON: f32 = 1.0e-12;

/// Kerr black-hole parameters in geometrized units (`G = c = 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrParams {
    pub mass: f32,
    pub spin: f32,
}

impl KerrParams {
    pub const REFERENCE: Self = Self {
        mass: 1.0,
        spin: 0.6,
    };

    #[inline]
    pub const fn new(mass: f32, spin: f32) -> Self {
        Self { mass, spin }
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.mass.is_finite()
            && self.spin.is_finite()
            && self.mass > 0.0
            && abs_f32(self.spin) < self.mass
    }
}

impl Default for KerrParams {
    fn default() -> Self {
        Self::REFERENCE
    }
}

/// Fixed-step tracing and thin-disk event policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceConfig {
    pub step_size: f32,
    pub max_steps: u32,
    pub gradient_epsilon: f32,
    pub escape_radius: f32,
    pub disk_inner_radius: f32,
    pub disk_outer_radius: f32,
}

impl TraceConfig {
    pub const REFERENCE: Self = Self {
        step_size: 0.1,
        max_steps: 500,
        gradient_epsilon: 0.01,
        escape_radius: 40.0,
        disk_inner_radius: 3.829_069,
        disk_outer_radius: 15.0,
    };

    #[inline]
    pub fn is_valid(self, params: &KerrParams) -> bool {
        self.step_size.is_finite()
            && self.gradient_epsilon.is_finite()
            && self.escape_radius.is_finite()
            && self.disk_inner_radius.is_finite()
            && self.disk_outer_radius.is_finite()
            && self.max_steps > 0
            && self.step_size > 0.0
            && self.gradient_epsilon > 0.0
            && self.escape_radius > horizon_radius(params)
            && self.disk_inner_radius > horizon_radius(params)
            && self.disk_outer_radius > self.disk_inner_radius
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self::REFERENCE
    }
}

/// Canonical Hamiltonian state: contravariant position and covariant momentum.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HamiltonianRay {
    pub position: [f32; DIM],
    pub momentum: [f32; DIM],
}

/// Metric-orthonormal frame carried by a stationary camera.
///
/// The frame is only defined where the stationary Killing vector is
/// timelike (outside the ergoregion). `time` is future-directed and the
/// three spatial axes have positive unit norm.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraTetrad {
    pub position: [f32; 3],
    pub time: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
}

/// A freely falling observer and the metric-orthonormal frame used to launch
/// its camera rays. `ray` is timelike (`H = -1/2`), unlike image rays.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfallingObserver {
    pub ray: HamiltonianRay,
    pub tetrad: CameraTetrad,
}

/// Coordinate orientation of a circular equatorial orbit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrbitDirection {
    Positive,
    Negative,
}

impl OrbitDirection {
    #[inline]
    const fn sign(self) -> f32 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

/// Four-velocity and angular velocity of a circular equatorial emitter.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircularEmitter {
    pub radius: f32,
    /// Coordinate angular velocity `d phi / dt`.
    pub angular_velocity: f32,
    pub four_velocity: [f32; 4],
}

impl HamiltonianRay {
    #[inline]
    pub const fn new(position: [f32; DIM], momentum: [f32; DIM]) -> Self {
        Self { position, momentum }
    }

    #[inline]
    pub const fn spatial_position(self) -> [f32; 3] {
        [self.position[1], self.position[2], self.position[3]]
    }
}

/// Interpolated first intersection with the equatorial thin disk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskHit {
    pub position: [f32; 3],
    pub radius: f32,
    /// Fraction of the accepted integrator segment in `[0, 1]`.
    pub segment_fraction: f32,
}

/// Cached disk-side boundary projection for a stationary Kerr spacetime.
///
/// This is the compact "holographic" representation used by renderers: the
/// expensive bulk geodesic has already been reduced to its emitting boundary,
/// relativistic transfer, orbital phase rate, and retarded time. An unchanged
/// camera and metric can reuse this record for every animated frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrDiskBoundary {
    pub point: [f32; 3],
    pub radius: f32,
    /// Local spatial Hamiltonian-flow direction at the disk crossing.
    pub direction: [f32; 3],
    /// Frequency transfer `nu_observer / nu_emitter`.
    pub redshift: f32,
    pub local_flux: f32,
    /// Local orbital kinetic energy per unit rest mass, `gamma - 1`,
    /// measured by the stationary observer at the disk crossing.
    pub specific_kinetic_energy: f32,
    pub angular_velocity: f32,
    pub travel_time: f32,
}

impl KerrDiskBoundary {
    /// Transform the cached crossing into a stationary material-space volume.
    ///
    /// `observer_time * time_scale - travel_time` is the retarded emission
    /// time in geometrized units. Each boundary record rotates at its own
    /// circular-orbit rate, providing a local differential-rotation model
    /// without rebuilding or copying the voxel payload.
    pub fn material_coordinates(self, observer_time: f32, time_scale: f32) -> ([f32; 3], [f32; 3]) {
        let phase = self.angular_velocity * (observer_time * time_scale - self.travel_time);
        let sin = sin_f32(-phase);
        let cos = cos_f32(-phase);
        let rotate = |value: [f32; 3]| {
            [
                cos * value[0] - sin * value[1],
                sin * value[0] + cos * value[1],
                value[2],
            ]
        };
        (rotate(self.point), rotate(self.direction))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraceEvent {
    Capture,
    Disk(DiskHit),
    Escape,
    Unresolved,
}

/// Outer event horizon `r_+`.
#[inline]
pub fn horizon_radius(params: &KerrParams) -> f32 {
    params.mass + sqrt_f32(params.mass * params.mass - params.spin * params.spin)
}

/// Oblate Boyer-Lindquist radial coordinate squared at a Cartesian point.
#[inline]
pub fn radius_squared(params: &KerrParams, point: [f32; 3]) -> f32 {
    let [x, y, z] = point;
    let a2 = params.spin * params.spin;
    let s = x * x + y * y + z * z - a2;
    0.5 * (s + sqrt_f32(s * s + 4.0 * a2 * z * z))
}

#[inline]
pub fn radius(params: &KerrParams, point: [f32; 3]) -> f32 {
    sqrt_f32(radius_squared(params, point))
}

/// Covariant Kerr-Schild null vector and scalar metric perturbation.
#[inline]
pub fn kerr_schild_field(params: &KerrParams, point: [f32; 3]) -> ([f32; 4], f32) {
    let [x, y, z] = point;
    let r2 = radius_squared(params, point);
    let r = sqrt_f32(r2);
    let a = params.spin;
    let denominator = r2 + a * a;
    if r <= SINGULAR_EPSILON || denominator <= SINGULAR_EPSILON {
        return ([f32::NAN; 4], f32::NAN);
    }
    let k = [
        1.0,
        (r * x + a * y) / denominator,
        (r * y - a * x) / denominator,
        z / r,
    ];
    let r4 = r2 * r2;
    let field = 2.0 * params.mass * r2 * r / (r4 + a * a * z * z);
    (k, field)
}

/// Covariant and inverse metric. The inverse is analytic because the
/// Kerr-Schild vector is null with respect to Minkowski space.
pub fn metric_pair(params: &KerrParams, point: [f32; 3]) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    let (k_cov, field) = kerr_schild_field(params, point);
    let k_up = [-k_cov[0], k_cov[1], k_cov[2], k_cov[3]];
    let mut cov = [[0.0; DIM]; DIM];
    let mut inverse = [[0.0; DIM]; DIM];
    for row in 0..DIM {
        for column in 0..DIM {
            let eta = if row == column {
                if row == 0 { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            cov[row][column] = eta + field * k_cov[row] * k_cov[column];
            inverse[row][column] = eta - field * k_up[row] * k_up[column];
        }
    }
    (cov, inverse)
}

#[inline]
fn inner_with_metric(metric: &[[f32; 4]; 4], left: [f32; 4], right: [f32; 4]) -> f32 {
    let mut value = 0.0;
    for row in 0..DIM {
        for column in 0..DIM {
            value += metric[row][column] * left[row] * right[column];
        }
    }
    value
}

#[inline]
fn lower_with_metric(metric: &[[f32; 4]; 4], vector: [f32; 4]) -> [f32; 4] {
    let mut lowered = [0.0; DIM];
    for row in 0..DIM {
        for column in 0..DIM {
            lowered[row] += metric[row][column] * vector[column];
        }
    }
    lowered
}

/// Metric inner product of two contravariant vectors at `point`.
#[inline]
pub fn metric_inner(params: &KerrParams, point: [f32; 3], left: [f32; 4], right: [f32; 4]) -> f32 {
    inner_with_metric(&metric_pair(params, point).0, left, right)
}

/// Lower the index of a contravariant vector at `point`.
#[inline]
pub fn lower_vector(params: &KerrParams, point: [f32; 3], vector: [f32; 4]) -> [f32; 4] {
    lower_with_metric(&metric_pair(params, point).0, vector)
}

#[inline]
fn finite_vector(vector: [f32; 4]) -> bool {
    vector[0].is_finite() && vector[1].is_finite() && vector[2].is_finite() && vector[3].is_finite()
}

#[inline]
fn project_from_time(metric: &[[f32; 4]; 4], vector: &mut [f32; 4], time: [f32; 4]) {
    let projection = inner_with_metric(metric, *vector, time);
    for component in 0..DIM {
        vector[component] += projection * time[component];
    }
}

#[inline]
fn project_from_space(metric: &[[f32; 4]; 4], vector: &mut [f32; 4], axis: [f32; 4]) {
    let projection = inner_with_metric(metric, *vector, axis);
    for component in 0..DIM {
        vector[component] -= projection * axis[component];
    }
}

#[inline]
fn normalize_spacelike(metric: &[[f32; 4]; 4], vector: &mut [f32; 4]) -> bool {
    let norm_squared = inner_with_metric(metric, *vector, *vector);
    if !norm_squared.is_finite() || norm_squared <= SINGULAR_EPSILON {
        return false;
    }
    let inverse_norm = 1.0 / sqrt_f32(norm_squared);
    for component in &mut *vector {
        *component *= inverse_norm;
    }
    finite_vector(*vector)
}

#[inline]
fn normalize_timelike(metric: &[[f32; 4]; 4], vector: &mut [f32; 4]) -> bool {
    let norm_squared = inner_with_metric(metric, *vector, *vector);
    if !norm_squared.is_finite() || norm_squared >= -SINGULAR_EPSILON {
        return false;
    }
    let inverse_norm = 1.0 / sqrt_f32(-norm_squared);
    for component in &mut *vector {
        *component *= inverse_norm;
    }
    finite_vector(*vector)
}

fn tetrad_from_time_and_hints(
    metric: &[[f32; 4]; 4],
    position: [f32; 3],
    time: [f32; 4],
    forward_hint: [f32; 3],
    up_hint: [f32; 3],
) -> Option<CameraTetrad> {
    let right_hint = [
        forward_hint[1] * up_hint[2] - forward_hint[2] * up_hint[1],
        forward_hint[2] * up_hint[0] - forward_hint[0] * up_hint[2],
        forward_hint[0] * up_hint[1] - forward_hint[1] * up_hint[0],
    ];
    let mut forward = [0.0, forward_hint[0], forward_hint[1], forward_hint[2]];
    project_from_time(metric, &mut forward, time);
    normalize_spacelike(metric, &mut forward).then_some(())?;

    let mut right = [0.0, right_hint[0], right_hint[1], right_hint[2]];
    project_from_time(metric, &mut right, time);
    project_from_space(metric, &mut right, forward);
    normalize_spacelike(metric, &mut right).then_some(())?;

    let mut up = [0.0, up_hint[0], up_hint[1], up_hint[2]];
    project_from_time(metric, &mut up, time);
    project_from_space(metric, &mut up, forward);
    project_from_space(metric, &mut up, right);
    normalize_spacelike(metric, &mut up).then_some(())?;

    Some(CameraTetrad {
        position,
        time,
        right,
        up,
        forward,
    })
}

/// Construct a locally zero-angular-momentum observer with an inward velocity.
///
/// Unlike [`static_camera_tetrad`], this frame remains valid in the
/// ergoregion. `inward_speed` is measured in the local zero-angular-momentum
/// frame and must be subluminal. The returned observer follows a geodesic once
/// advanced with [`advance_infalling_observer`].
pub fn zero_angular_momentum_infalling_observer(
    params: &KerrParams,
    position: [f32; 3],
    inward_speed: f32,
    target: [f32; 3],
    up_hint: [f32; 3],
) -> Option<InfallingObserver> {
    if !params.is_valid()
        || !position.iter().all(|value| value.is_finite())
        || !target.iter().all(|value| value.is_finite())
        || !up_hint.iter().all(|value| value.is_finite())
        || !inward_speed.is_finite()
        || !(0.0..1.0).contains(&inward_speed)
    {
        return None;
    }
    let metric = metric_pair(params, position).0;
    // The axial Killing direction in Cartesian Kerr-Schild coordinates. A
    // time direction orthogonal to it has p_phi = 0 and is the local ZAMO.
    let azimuth = [0.0, -position[1], position[0], 0.0];
    let azimuth_norm = inner_with_metric(&metric, azimuth, azimuth);
    if !azimuth_norm.is_finite() || azimuth_norm <= SINGULAR_EPSILON {
        return None;
    }
    let coordinate_time = [1.0, 0.0, 0.0, 0.0];
    let angular_rate = -inner_with_metric(&metric, coordinate_time, azimuth) / azimuth_norm;
    let mut zamo_time =
        core::array::from_fn(|axis| coordinate_time[axis] + angular_rate * azimuth[axis]);
    normalize_timelike(&metric, &mut zamo_time).then_some(())?;

    let mut inward = [
        0.0,
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    project_from_time(&metric, &mut inward, zamo_time);
    let mut azimuth_axis = azimuth;
    project_from_time(&metric, &mut azimuth_axis, zamo_time);
    normalize_spacelike(&metric, &mut azimuth_axis).then_some(())?;
    project_from_space(&metric, &mut inward, azimuth_axis);
    normalize_spacelike(&metric, &mut inward).then_some(())?;

    let gamma = 1.0 / sqrt_f32(1.0 - inward_speed * inward_speed);
    let time = core::array::from_fn(|axis| gamma * (zamo_time[axis] + inward_speed * inward[axis]));
    let tetrad = tetrad_from_time_and_hints(
        &metric,
        position,
        time,
        [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        ],
        up_hint,
    )?;
    let momentum = lower_with_metric(&metric, tetrad.time);
    Some(InfallingObserver {
        ray: HamiltonianRay::new([0.0, position[0], position[1], position[2]], momentum),
        tetrad,
    })
}

fn christoffel_symbols(
    params: &KerrParams,
    point: [f32; 3],
    gradient_epsilon: f32,
) -> Option<[[[f32; DIM]; DIM]; DIM]> {
    let (_, inverse) = metric_pair(params, point);
    let scale = radius(params, point).max(1.0);
    let epsilon = gradient_epsilon * scale;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let mut derivative = [[[0.0; DIM]; DIM]; DIM];
    for axis in 1..DIM {
        let mut plus = point;
        let mut minus = point;
        plus[axis - 1] += epsilon;
        minus[axis - 1] -= epsilon;
        let metric_plus = metric_pair(params, plus).0;
        let metric_minus = metric_pair(params, minus).0;
        for row in 0..DIM {
            for column in 0..DIM {
                derivative[axis][row][column] =
                    (metric_plus[row][column] - metric_minus[row][column]) / (2.0 * epsilon);
            }
        }
    }
    let mut connection = [[[0.0; DIM]; DIM]; DIM];
    for upper in 0..DIM {
        for left in 0..DIM {
            for right in 0..DIM {
                let mut value = 0.0;
                for lower in 0..DIM {
                    value += inverse[upper][lower]
                        * (derivative[left][lower][right] + derivative[right][lower][left]
                            - derivative[lower][left][right]);
                }
                connection[upper][left][right] = 0.5 * value;
            }
        }
    }
    connection
        .iter()
        .flatten()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(connection)
}

#[inline]
fn parallel_transport_rate(
    connection: &[[[f32; DIM]; DIM]; DIM],
    velocity: [f32; DIM],
    axis: [f32; DIM],
) -> [f32; DIM] {
    core::array::from_fn(|upper| {
        let mut rate = 0.0;
        for left in 0..DIM {
            for right in 0..DIM {
                rate -= connection[upper][left][right] * velocity[left] * axis[right];
            }
        }
        rate
    })
}

/// Advance a freely falling observer by proper time while parallel-transporting
/// its spatial camera axes. A final metric Gram-Schmidt pass removes only the
/// accumulated finite-step drift; it does not re-anchor the frame to a
/// stationary observer, so the result remains valid through the ergoregion.
pub fn advance_infalling_observer(
    params: &KerrParams,
    observer: &mut InfallingObserver,
    proper_time_step: f32,
    gradient_epsilon: f32,
) -> Option<()> {
    if !params.is_valid()
        || !proper_time_step.is_finite()
        || proper_time_step <= 0.0
        || !gradient_epsilon.is_finite()
        || gradient_epsilon <= 0.0
    {
        return None;
    }
    let old_ray = observer.ray;
    let old_position = old_ray.spatial_position();
    let old_velocity = observer.tetrad.time;
    let old_connection = christoffel_symbols(params, old_position, gradient_epsilon)?;

    midpoint_step(
        params,
        &mut observer.ray,
        proper_time_step,
        gradient_epsilon,
    );
    let new_position = observer.ray.spatial_position();
    let new_metric = metric_pair(params, new_position).0;
    let constraint = hamiltonian(params, &observer.ray);
    if !constraint.is_finite() || constraint >= -SINGULAR_EPSILON {
        return None;
    }
    // Hamilton's timelike flow should retain H=-1/2. Correcting only the
    // scalar normalization bounds the RK2 drift without changing direction.
    let correction = sqrt_f32(-0.5 / constraint);
    for component in &mut observer.ray.momentum {
        *component *= correction;
    }
    let new_velocity = hamiltonian_rhs(params, &observer.ray, gradient_epsilon).position;
    if !finite_vector(new_velocity)
        || (inner_with_metric(&new_metric, new_velocity, new_velocity) + 1.0).abs() > 2.0e-3
    {
        return None;
    }

    let midpoint_position =
        core::array::from_fn(|axis| 0.5 * (old_position[axis] + new_position[axis]));
    let midpoint_metric = metric_pair(params, midpoint_position).0;
    let mut midpoint_velocity =
        core::array::from_fn(|axis| 0.5 * (old_velocity[axis] + new_velocity[axis]));
    normalize_timelike(&midpoint_metric, &mut midpoint_velocity).then_some(())?;
    let midpoint_connection = christoffel_symbols(params, midpoint_position, gradient_epsilon)?;

    let mut transported = [
        observer.tetrad.forward,
        observer.tetrad.right,
        observer.tetrad.up,
    ];
    for axis in &mut transported {
        let first = parallel_transport_rate(&old_connection, old_velocity, *axis);
        let halfway = core::array::from_fn(|component| {
            axis[component] + 0.5 * proper_time_step * first[component]
        });
        let second = parallel_transport_rate(&midpoint_connection, midpoint_velocity, halfway);
        for component in 0..DIM {
            axis[component] += proper_time_step * second[component];
        }
    }

    let mut forward = transported[0];
    project_from_time(&new_metric, &mut forward, new_velocity);
    normalize_spacelike(&new_metric, &mut forward).then_some(())?;
    let mut right = transported[1];
    project_from_time(&new_metric, &mut right, new_velocity);
    project_from_space(&new_metric, &mut right, forward);
    normalize_spacelike(&new_metric, &mut right).then_some(())?;
    let mut up = transported[2];
    project_from_time(&new_metric, &mut up, new_velocity);
    project_from_space(&new_metric, &mut up, forward);
    project_from_space(&new_metric, &mut up, right);
    normalize_spacelike(&new_metric, &mut up).then_some(())?;
    observer.tetrad = CameraTetrad {
        position: new_position,
        time: new_velocity,
        right,
        up,
        forward,
    };
    Some(())
}

/// Construct a stationary look-at camera frame by metric Gram-Schmidt.
///
/// Returns `None` in the ergoregion, at a degenerate look-at direction, or
/// when `up_hint` is parallel to the viewing direction.
pub fn static_camera_tetrad(
    params: &KerrParams,
    position: [f32; 3],
    target: [f32; 3],
    up_hint: [f32; 3],
) -> Option<CameraTetrad> {
    if !params.is_valid()
        || !position.iter().all(|component| component.is_finite())
        || !target.iter().all(|component| component.is_finite())
        || !up_hint.iter().all(|component| component.is_finite())
    {
        return None;
    }

    let metric = metric_pair(params, position).0;
    let stationary_norm = metric[0][0];
    if !stationary_norm.is_finite() || stationary_norm >= -SINGULAR_EPSILON {
        return None;
    }
    let time = [1.0 / sqrt_f32(-stationary_norm), 0.0, 0.0, 0.0];

    let forward_hint = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ];
    let right_hint = [
        forward_hint[1] * up_hint[2] - forward_hint[2] * up_hint[1],
        forward_hint[2] * up_hint[0] - forward_hint[0] * up_hint[2],
        forward_hint[0] * up_hint[1] - forward_hint[1] * up_hint[0],
    ];

    let mut forward = [0.0, forward_hint[0], forward_hint[1], forward_hint[2]];
    project_from_time(&metric, &mut forward, time);
    if !normalize_spacelike(&metric, &mut forward) {
        return None;
    }

    let mut right = [0.0, right_hint[0], right_hint[1], right_hint[2]];
    project_from_time(&metric, &mut right, time);
    project_from_space(&metric, &mut right, forward);
    if !normalize_spacelike(&metric, &mut right) {
        return None;
    }

    let mut up = [0.0, up_hint[0], up_hint[1], up_hint[2]];
    project_from_time(&metric, &mut up, time);
    project_from_space(&metric, &mut up, forward);
    project_from_space(&metric, &mut up, right);
    if !normalize_spacelike(&metric, &mut up) {
        return None;
    }

    Some(CameraTetrad {
        position,
        time,
        right,
        up,
        forward,
    })
}

/// Launch a future-directed null ray through tangent-plane camera coordinates.
///
/// `(0, 0)` is the optical axis. The screen coordinates are tangent offsets,
/// so a finite field of view can be applied by scaling them before this call.
pub fn camera_null_ray(
    params: &KerrParams,
    camera: &CameraTetrad,
    screen: [f32; 2],
) -> Option<HamiltonianRay> {
    if !screen[0].is_finite() || !screen[1].is_finite() {
        return None;
    }
    let direction_norm = sqrt_f32(1.0 + screen[0] * screen[0] + screen[1] * screen[1]);
    if !direction_norm.is_finite() || direction_norm <= SINGULAR_EPSILON {
        return None;
    }
    let inverse_norm = 1.0 / direction_norm;
    let mut tangent = [0.0; DIM];
    for (component, tangent_component) in tangent.iter_mut().enumerate() {
        *tangent_component = camera.time[component]
            + inverse_norm
                * (camera.forward[component]
                    + screen[0] * camera.right[component]
                    + screen[1] * camera.up[component]);
    }
    if !finite_vector(tangent) {
        return None;
    }
    let momentum = lower_vector(params, camera.position, tangent);
    finite_vector(momentum).then_some(HamiltonianRay::new(
        [
            0.0,
            camera.position[0],
            camera.position[1],
            camera.position[2],
        ],
        momentum,
    ))
}

/// Launch a past-directed null ray from a local camera screen coordinate.
///
/// Exterior beauty renders use [`camera_null_ray`] and exploit stationarity to
/// trace forward. An observer inside an event horizon must instead trace its
/// past light cone: those geodesics can reach the external disk and sky.
pub fn camera_past_null_ray(
    params: &KerrParams,
    camera: &CameraTetrad,
    screen: [f32; 2],
) -> Option<HamiltonianRay> {
    if !screen[0].is_finite() || !screen[1].is_finite() {
        return None;
    }
    let direction_norm = sqrt_f32(1.0 + screen[0] * screen[0] + screen[1] * screen[1]);
    if !direction_norm.is_finite() || direction_norm <= SINGULAR_EPSILON {
        return None;
    }
    let inverse_norm = 1.0 / direction_norm;
    let mut tangent = [0.0; DIM];
    for (component, tangent_component) in tangent.iter_mut().enumerate() {
        *tangent_component = -camera.time[component]
            + inverse_norm
                * (camera.forward[component]
                    + screen[0] * camera.right[component]
                    + screen[1] * camera.up[component]);
    }
    if !finite_vector(tangent) {
        return None;
    }
    let momentum = lower_vector(params, camera.position, tangent);
    finite_vector(momentum).then_some(HamiltonianRay::new(
        [
            0.0,
            camera.position[0],
            camera.position[1],
            camera.position[2],
        ],
        momentum,
    ))
}

#[inline]
fn inverse_metric(params: &KerrParams, point: [f32; 3]) -> [[f32; 4]; 4] {
    metric_pair(params, point).1
}

/// Null Hamiltonian `H = 1/2 g^{mu nu} p_mu p_nu`.
pub fn hamiltonian(params: &KerrParams, ray: &HamiltonianRay) -> f32 {
    let inverse = inverse_metric(params, ray.spatial_position());
    let mut value = 0.0;
    for (metric_row, row_momentum) in inverse.iter().zip(ray.momentum) {
        for (metric_component, column_momentum) in metric_row.iter().zip(ray.momentum) {
            value += metric_component * row_momentum * column_momentum;
        }
    }
    0.5 * value
}

/// Hamiltonian flow `[dX/dlambda, dp/dlambda]`. Stationarity makes `dp_t`
/// exactly zero; spatial derivatives use a centered relative difference.
pub fn hamiltonian_rhs(
    params: &KerrParams,
    ray: &HamiltonianRay,
    gradient_epsilon: f32,
) -> HamiltonianRay {
    let point = ray.spatial_position();
    let inverse = inverse_metric(params, point);
    let mut position_rate = [0.0; DIM];
    for (rate, metric_row) in position_rate.iter_mut().zip(inverse) {
        for (metric_component, momentum) in metric_row.iter().zip(ray.momentum) {
            *rate += metric_component * momentum;
        }
    }

    let scale = radius(params, point).max(1.0);
    let epsilon = gradient_epsilon * scale;
    let mut momentum_rate = [0.0; DIM];
    for axis in 0..3 {
        let mut plus = *ray;
        let mut minus = *ray;
        plus.position[axis + 1] += epsilon;
        minus.position[axis + 1] -= epsilon;
        momentum_rate[axis + 1] =
            -(hamiltonian(params, &plus) - hamiltonian(params, &minus)) / (2.0 * epsilon);
    }
    HamiltonianRay::new(position_rate, momentum_rate)
}

/// Explicit midpoint RK2 step. `p_t` is restored bit-for-bit because the
/// metric is stationary and its derivative is analytically zero.
pub fn midpoint_step(
    params: &KerrParams,
    ray: &mut HamiltonianRay,
    step_size: f32,
    gradient_epsilon: f32,
) {
    let conserved_p_t = ray.momentum[0];
    let mut state = [
        ray.position[0],
        ray.position[1],
        ray.position[2],
        ray.position[3],
        ray.momentum[0],
        ray.momentum[1],
        ray.momentum[2],
        ray.momentum[3],
    ];
    explicit_midpoint(&mut state, step_size, &mut |state: &[f32; 8]| {
        let state = HamiltonianRay::new(
            [state[0], state[1], state[2], state[3]],
            [state[4], state[5], state[6], state[7]],
        );
        let flow = hamiltonian_rhs(params, &state, gradient_epsilon);
        [
            flow.position[0],
            flow.position[1],
            flow.position[2],
            flow.position[3],
            flow.momentum[0],
            flow.momentum[1],
            flow.momentum[2],
            flow.momentum[3],
        ]
    });
    ray.position.copy_from_slice(&state[..DIM]);
    ray.momentum.copy_from_slice(&state[DIM..]);
    ray.momentum[0] = conserved_p_t;
}

/// Equatorial ISCO for a fixed positive orbital orientation. Positive spin
/// is prograde and negative spin is retrograde relative to that orientation.
#[inline]
pub fn equatorial_isco(params: &KerrParams) -> f32 {
    equatorial_isco_for(params, OrbitDirection::Positive)
}

/// Equatorial ISCO for the selected coordinate orientation.
pub fn equatorial_isco_for(params: &KerrParams, direction: OrbitDirection) -> f32 {
    let a = params.spin / params.mass;
    let z1 = 1.0 + cbrt_f32(1.0 - a * a) * (cbrt_f32(1.0 + a) + cbrt_f32(1.0 - a));
    let z2 = sqrt_f32(3.0 * a * a + z1 * z1);
    let radical = sqrt_f32((3.0 - z1) * (3.0 + z1 + 2.0 * z2));
    params.mass * (3.0 + z2 - (a * direction.sign()).signum() * radical)
}

/// Angular velocity and four-velocity of a circular equatorial geodesic.
///
/// `point` is expressed in Cartesian Kerr-Schild coordinates and must lie on
/// the equatorial plane. The returned velocity is future-directed and
/// normalized to `g(u, u) = -1`. Orbits inside the circular timelike domain
/// return `None`; callers can use [`equatorial_isco_for`] as a stable-disk
/// cutoff.
pub fn circular_equatorial_emitter(
    params: &KerrParams,
    point: [f32; 3],
    direction: OrbitDirection,
) -> Option<CircularEmitter> {
    if !params.is_valid() || !point.iter().all(|component| component.is_finite()) {
        return None;
    }
    let orbital_radius = radius(params, point);
    if !orbital_radius.is_finite()
        || orbital_radius <= horizon_radius(params)
        || abs_f32(point[2]) > 1.0e-5 * orbital_radius.max(1.0)
    {
        return None;
    }

    let direction_sign = direction.sign();
    let root_mass = sqrt_f32(params.mass);
    let denominator =
        orbital_radius * sqrt_f32(orbital_radius) + direction_sign * params.spin * root_mass;
    if !denominator.is_finite() || abs_f32(denominator) <= SINGULAR_EPSILON {
        return None;
    }
    let angular_velocity = direction_sign * root_mass / denominator;
    let coordinate_tangent = [
        1.0,
        -angular_velocity * point[1],
        angular_velocity * point[0],
        0.0,
    ];
    let metric = metric_pair(params, point).0;
    let tangent_norm = inner_with_metric(&metric, coordinate_tangent, coordinate_tangent);
    if !tangent_norm.is_finite() || tangent_norm >= -SINGULAR_EPSILON {
        return None;
    }
    let normalization = 1.0 / sqrt_f32(-tangent_norm);
    let mut four_velocity = coordinate_tangent;
    for component in &mut four_velocity {
        *component *= normalization;
    }
    finite_vector(four_velocity).then_some(CircularEmitter {
        radius: orbital_radius,
        angular_velocity,
        four_velocity,
    })
}

#[inline]
fn specific_kinetic_energy_for_emitter(
    params: &KerrParams,
    point: [f32; 3],
    emitter_velocity: [f32; 4],
) -> Option<f32> {
    let metric = metric_pair(params, point).0;
    let stationary_norm = metric[0][0];
    if !stationary_norm.is_finite() || stationary_norm >= -SINGULAR_EPSILON {
        return None;
    }
    let stationary_velocity = [1.0 / sqrt_f32(-stationary_norm), 0.0, 0.0, 0.0];
    let gamma = -inner_with_metric(&metric, stationary_velocity, emitter_velocity);
    if !gamma.is_finite() || gamma < 1.0 - 2.0e-5 {
        return None;
    }
    Some((gamma - 1.0).max(0.0))
}

/// Local orbital kinetic energy per unit rest mass for a circular equatorial
/// geodesic, in geometrized units (`c = 1`).
///
/// The energy is `gamma - 1` relative to the normalized stationary observer
/// at the same event. It is a local heating proxy rather than the conserved
/// Killing energy at infinity, and is only defined outside the ergoregion
/// where that stationary observer is timelike.
pub fn circular_equatorial_specific_kinetic_energy(
    params: &KerrParams,
    orbital_radius: f32,
    direction: OrbitDirection,
) -> Option<f32> {
    if !params.is_valid() || !orbital_radius.is_finite() || orbital_radius <= horizon_radius(params)
    {
        return None;
    }
    // On the equatorial plane, Cartesian Kerr-Schild rho obeys
    // rho^2 = r^2 + a^2.
    let point = [
        sqrt_f32(orbital_radius * orbital_radius + params.spin * params.spin),
        0.0,
        0.0,
    ];
    let emitter = circular_equatorial_emitter(params, point, direction)?;
    specific_kinetic_energy_for_emitter(params, point, emitter.four_velocity)
}

/// Photon frequency measured by an observer with contravariant four-velocity.
#[inline]
pub fn measured_frequency(momentum: [f32; 4], observer_velocity: [f32; 4]) -> f32 {
    -(momentum[0] * observer_velocity[0]
        + momentum[1] * observer_velocity[1]
        + momentum[2] * observer_velocity[2]
        + momentum[3] * observer_velocity[3])
}

/// Frequency transfer `g = nu_observer / nu_emitter` along a photon ray.
///
/// Both four-velocities must be future-directed relative to the covariant
/// photon momentum. The factor captures gravitational redshift and the
/// line-of-sight Doppler asymmetry of the rotating disk.
#[inline]
pub fn redshift_factor(
    momentum: [f32; 4],
    observer_velocity: [f32; 4],
    emitter_velocity: [f32; 4],
) -> Option<f32> {
    let observed = measured_frequency(momentum, observer_velocity);
    let emitted = measured_frequency(momentum, emitter_velocity);
    if !observed.is_finite()
        || !emitted.is_finite()
        || observed <= SINGULAR_EPSILON
        || emitted <= SINGULAR_EPSILON
    {
        return None;
    }
    let factor = observed / emitted;
    factor.is_finite().then_some(factor)
}

/// Normalized zero-torque thin-disk surface flux.
///
/// This is the Newtonian radial factor `r^-3 * (1 - sqrt(r_in / r))`
/// retained by relativistic Novikov-Thorne/Page-Thorne disks in their
/// large-radius limit. It enforces the physically important dark inner edge
/// instead of assigning maximum emissivity at the ISCO. The result is scaled
/// to one at `r = (49 / 36) * r_in`; callers layer relativistic transfer and
/// spectral color on top.
pub fn zero_torque_thin_disk_flux(radius: f32, inner_radius: f32) -> Option<f32> {
    if !radius.is_finite()
        || !inner_radius.is_finite()
        || inner_radius <= 0.0
        || radius < inner_radius
    {
        return None;
    }
    let scaled = radius / inner_radius;
    let inverse = scaled.recip();
    let inverse_cubed = inverse * inverse * inverse;
    let edge = 1.0 - sqrt_f32(inverse);
    let peak_scaled = 49.0 / 36.0;
    let peak_inverse = 1.0 / peak_scaled;
    let peak = peak_inverse * peak_inverse * peak_inverse * (1.0 - sqrt_f32(peak_inverse));
    let flux = inverse_cubed * edge / peak;
    flux.is_finite().then_some(flux.max(0.0))
}

/// Collapse a validated disk crossing into a reusable boundary-transport
/// record. This is an observation boundary, so redshift and the local ray
/// direction are computed once rather than repeated on every shaded frame.
pub fn disk_boundary_transport(
    params: &KerrParams,
    config: &TraceConfig,
    camera: &CameraTetrad,
    point: [f32; 3],
    radius: f32,
    momentum: [f32; 4],
    coordinate_time: f32,
) -> Option<KerrDiskBoundary> {
    if !params.is_valid()
        || !config.is_valid(params)
        || !point.iter().all(|value| value.is_finite())
        || !momentum.iter().all(|value| value.is_finite())
        || !radius.is_finite()
        || !coordinate_time.is_finite()
    {
        return None;
    }
    let ray = HamiltonianRay::new([coordinate_time, point[0], point[1], point[2]], momentum);
    let flow = hamiltonian_rhs(params, &ray, config.gradient_epsilon);
    let spatial = [flow.position[1], flow.position[2], flow.position[3]];
    let length =
        sqrt_f32(spatial[0] * spatial[0] + spatial[1] * spatial[1] + spatial[2] * spatial[2]);
    if !length.is_finite() || length <= SINGULAR_EPSILON {
        return None;
    }
    let direction = spatial.map(|value| value / length);
    let emitter = circular_equatorial_emitter(params, point, OrbitDirection::Positive)?;
    let redshift = redshift_factor(momentum, camera.time, emitter.four_velocity)?;
    let local_flux = zero_torque_thin_disk_flux(radius, config.disk_inner_radius)?;
    let specific_kinetic_energy =
        specific_kinetic_energy_for_emitter(params, point, emitter.four_velocity)?;
    Some(KerrDiskBoundary {
        point,
        radius,
        direction,
        redshift,
        local_flux,
        specific_kinetic_energy,
        angular_velocity: emitter.angular_velocity,
        travel_time: abs_f32(coordinate_time),
    })
}

#[inline]
fn outward(params: &KerrParams, ray: &HamiltonianRay) -> bool {
    let velocity = hamiltonian_rhs(params, ray, 1.0e-4).position;
    ray.position[1] * velocity[1] + ray.position[2] * velocity[2] + ray.position[3] * velocity[3]
        > 0.0
}

fn disk_hit(
    params: &KerrParams,
    config: &TraceConfig,
    from: &HamiltonianRay,
    to: &HamiltonianRay,
) -> Option<DiskHit> {
    let z0 = from.position[3];
    let z1 = to.position[3];
    let crossed = (z0 > 0.0 && z1 <= 0.0) || (z0 < 0.0 && z1 >= 0.0);
    if !crossed {
        return None;
    }
    let fraction = z0 / (z0 - z1);
    let point = [
        from.position[1] + fraction * (to.position[1] - from.position[1]),
        from.position[2] + fraction * (to.position[2] - from.position[2]),
        0.0,
    ];
    let hit_radius = radius(params, point);
    (hit_radius >= config.disk_inner_radius && hit_radius <= config.disk_outer_radius).then_some(
        DiskHit {
            position: point,
            radius: hit_radius,
            segment_fraction: fraction,
        },
    )
}

/// Trace to the first horizon, disk, or escape event.
pub fn trace_fixed(
    params: &KerrParams,
    config: &TraceConfig,
    ray: &mut HamiltonianRay,
) -> TraceEvent {
    if !params.is_valid() || !config.is_valid(params) {
        return TraceEvent::Unresolved;
    }
    let horizon = horizon_radius(params);
    for _ in 0..config.max_steps {
        let current_radius = radius(params, ray.spatial_position());
        if current_radius <= horizon {
            return TraceEvent::Capture;
        }
        if current_radius >= config.escape_radius && outward(params, ray) {
            return TraceEvent::Escape;
        }

        let previous = *ray;
        midpoint_step(params, ray, config.step_size, config.gradient_epsilon);
        let next_radius = radius(params, ray.spatial_position());
        if next_radius <= horizon {
            return TraceEvent::Capture;
        }
        if let Some(hit) = disk_hit(params, config, &previous, ray) {
            return TraceEvent::Disk(hit);
        }
        if next_radius >= config.escape_radius && outward(params, ray) {
            return TraceEvent::Escape;
        }
    }
    TraceEvent::Unresolved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            abs_f32(actual - expected) <= tolerance,
            "{actual} vs {expected} (tol {tolerance})"
        );
    }

    #[test]
    fn reference_horizon_and_isco_match_known_values() {
        let params = KerrParams::REFERENCE;
        assert_close(horizon_radius(&params), 1.8, 2.0e-6);
        assert_close(equatorial_isco(&params), 3.829_069, 2.0e-5);
        assert_close(
            equatorial_isco_for(&params, OrbitDirection::Negative),
            7.850_686,
            2.0e-5,
        );
        assert_close(
            equatorial_isco_for(
                &KerrParams::new(params.mass, -params.spin),
                OrbitDirection::Negative,
            ),
            equatorial_isco(&params),
            2.0e-6,
        );
        assert_close(equatorial_isco(&KerrParams::new(1.0, 0.0)), 6.0, 2.0e-6);
    }

    #[test]
    fn kerr_schild_vector_is_eta_null_and_metric_inverse_is_exact() {
        let params = KerrParams::REFERENCE;
        let point = [4.0, -1.5, 0.75];
        let (k, _) = kerr_schild_field(&params, point);
        let eta_norm = -k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
        assert_close(eta_norm, 0.0, 2.0e-6);

        let (metric, inverse) = metric_pair(&params, point);
        for row in 0..DIM {
            for column in 0..DIM {
                let product: f32 = (0..DIM)
                    .map(|inner| metric[row][inner] * inverse[inner][column])
                    .sum();
                assert_close(product, f32::from(row == column), 3.0e-6);
            }
        }
    }

    #[test]
    fn oblate_radius_has_schwarzschild_rotational_symmetry() {
        let params = KerrParams::new(1.0, 0.0);
        for point in [[3.0, 4.0, 0.0], [0.0, 3.0, 4.0], [4.0, 0.0, 3.0]] {
            assert_close(radius(&params, point), 5.0, 1.0e-6);
        }
    }

    #[test]
    fn midpoint_keeps_stationary_momentum_exact() {
        let params = KerrParams::REFERENCE;
        let mut ray = HamiltonianRay::new([0.0, 12.0, 1.0, 2.0], [-1.0, -0.9, 0.1, -0.05]);
        let before = ray.momentum[0].to_bits();
        for _ in 0..32 {
            midpoint_step(&params, &mut ray, 0.02, 1.0e-3);
        }
        assert_eq!(ray.momentum[0].to_bits(), before);
        assert!(hamiltonian(&params, &ray).is_finite());
    }

    #[test]
    fn disk_boundary_cache_applies_retarded_material_rotation() {
        let params = KerrParams::REFERENCE;
        let config = TraceConfig::REFERENCE;
        let camera =
            static_camera_tetrad(&params, [0.0, -22.0, 7.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();
        let point = [6.0, 0.0, 0.0];
        let boundary = disk_boundary_transport(
            &params,
            &config,
            &camera,
            point,
            radius(&params, point),
            [-1.0, 0.0, 0.0, 0.0],
            20.0,
        )
        .unwrap();
        let time_scale = 5.0;
        let (initial_point, initial_direction) =
            boundary.material_coordinates(boundary.travel_time / time_scale, time_scale);
        assert_eq!(initial_point, boundary.point);
        assert_eq!(initial_direction, boundary.direction);

        let (rotated_point, rotated_direction) = boundary.material_coordinates(6.0, time_scale);
        let squared_xy = |value: [f32; 3]| value[0] * value[0] + value[1] * value[1];
        assert_close(
            squared_xy(rotated_point),
            squared_xy(boundary.point),
            2.0e-5,
        );
        assert_close(
            squared_xy(rotated_direction),
            squared_xy(boundary.direction),
            2.0e-5,
        );
        assert_eq!(rotated_point[2], boundary.point[2]);
        assert_eq!(rotated_direction[2], boundary.direction[2]);
    }

    #[test]
    fn thin_disk_crossing_is_interpolated() {
        let params = KerrParams::REFERENCE;
        let config = TraceConfig::REFERENCE;
        let from = HamiltonianRay::new([0.0, 8.0, 0.0, 1.0], [-1.0, 0.0, 0.0, -1.0]);
        let to = HamiltonianRay::new([1.0, 8.0, 0.0, -3.0], [-1.0, 0.0, 0.0, -1.0]);
        let hit = disk_hit(&params, &config, &from, &to).unwrap();
        assert_close(hit.segment_fraction, 0.25, 1.0e-7);
        assert_close(hit.position[2], 0.0, 0.0);
        assert!(hit.radius >= config.disk_inner_radius && hit.radius <= config.disk_outer_radius);
    }

    #[test]
    fn camera_tetrad_is_metric_orthonormal_and_launches_null_rays() {
        let params = KerrParams::REFERENCE;
        let camera =
            static_camera_tetrad(&params, [30.0, -2.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();
        let axes = [camera.time, camera.right, camera.up, camera.forward];
        for row in 0..DIM {
            for column in 0..DIM {
                let expected = if row == column {
                    if row == 0 { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
                assert_close(
                    metric_inner(&params, camera.position, axes[row], axes[column]),
                    expected,
                    4.0e-6,
                );
            }
        }

        for screen in [[0.0, 0.0], [-0.7, 0.25], [1.2, -0.9]] {
            let ray = camera_null_ray(&params, &camera, screen).unwrap();
            assert_close(hamiltonian(&params, &ray), 0.0, 3.0e-6);
            assert!(ray.momentum[0] < 0.0);
        }
    }

    #[test]
    fn stationary_camera_rejects_ergoregion_and_degenerate_orientation() {
        let params = KerrParams::REFERENCE;
        assert!(
            static_camera_tetrad(&params, [1.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],)
                .is_none()
        );
        assert!(
            static_camera_tetrad(&params, [30.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 0.0, 1.0],)
                .is_none()
        );
        assert!(
            static_camera_tetrad(&params, [30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],)
                .is_none()
        );
    }

    #[test]
    fn zero_angular_momentum_infalling_observer_crosses_the_ergoregion() {
        let params = KerrParams::REFERENCE;
        let observer = zero_angular_momentum_infalling_observer(
            &params,
            [1.9, 0.0, 0.0],
            0.25,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .expect("ZAMO-relative infaller remains well-defined in the ergoregion");
        assert_close(
            metric_inner(
                &params,
                observer.tetrad.position,
                observer.tetrad.time,
                observer.tetrad.time,
            ),
            -1.0,
            3.0e-5,
        );
        let angular_momentum = observer.ray.position[1] * observer.ray.momentum[2]
            - observer.ray.position[2] * observer.ray.momentum[1];
        assert!(angular_momentum.abs() < 3.0e-5, "Lz = {angular_momentum}");
        assert!(hamiltonian(&params, &observer.ray) < -0.49);
    }

    #[test]
    fn infalling_observer_transport_preserves_a_local_camera_frame() {
        let params = KerrParams::REFERENCE;
        let mut observer = zero_angular_momentum_infalling_observer(
            &params,
            [2.4, 0.0, 0.0],
            0.35,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();
        let initial_radius = radius(&params, observer.tetrad.position);
        for _ in 0..24 {
            advance_infalling_observer(&params, &mut observer, 0.01, 0.002).unwrap();
        }
        let metric = metric_pair(&params, observer.tetrad.position).0;
        assert!(radius(&params, observer.tetrad.position) < initial_radius);
        assert_close(
            inner_with_metric(&metric, observer.tetrad.time, observer.tetrad.time),
            -1.0,
            3.0e-4,
        );
        for axis in [
            observer.tetrad.forward,
            observer.tetrad.right,
            observer.tetrad.up,
        ] {
            assert_close(inner_with_metric(&metric, axis, axis), 1.0, 3.0e-4);
            assert!(inner_with_metric(&metric, observer.tetrad.time, axis).abs() < 3.0e-4);
        }
        assert!(hamiltonian(&params, &observer.ray).is_finite());
    }

    #[test]
    fn circular_disk_emitter_is_timelike_and_keplerian() {
        let params = KerrParams::REFERENCE;
        let orbital_radius = 8.0;
        let point = [
            sqrt_f32(orbital_radius * orbital_radius + params.spin * params.spin),
            0.0,
            0.0,
        ];
        let positive =
            circular_equatorial_emitter(&params, point, OrbitDirection::Positive).unwrap();
        let negative =
            circular_equatorial_emitter(&params, point, OrbitDirection::Negative).unwrap();
        let expected_positive = 1.0 / (orbital_radius * sqrt_f32(orbital_radius) + params.spin);
        let expected_negative = -1.0 / (orbital_radius * sqrt_f32(orbital_radius) - params.spin);
        assert_close(positive.radius, orbital_radius, 1.0e-6);
        assert_close(positive.angular_velocity, expected_positive, 1.0e-7);
        assert_close(negative.angular_velocity, expected_negative, 1.0e-7);
        assert_close(
            metric_inner(
                &params,
                point,
                positive.four_velocity,
                positive.four_velocity,
            ),
            -1.0,
            2.0e-6,
        );
        assert!(positive.four_velocity[0] > 0.0);
    }

    #[test]
    fn local_orbital_kinetic_energy_has_newtonian_limit_and_inner_heating() {
        let schwarzschild = KerrParams::new(1.0, 0.0);
        let distant = circular_equatorial_specific_kinetic_energy(
            &schwarzschild,
            100.0,
            OrbitDirection::Positive,
        )
        .unwrap();
        let newtonian = schwarzschild.mass / (2.0 * 100.0);
        assert!((distant / newtonian - 1.0).abs() < 0.04, "{distant}");

        let params = KerrParams::REFERENCE;
        let inner = circular_equatorial_specific_kinetic_energy(
            &params,
            equatorial_isco(&params),
            OrbitDirection::Positive,
        )
        .unwrap();
        let outer = circular_equatorial_specific_kinetic_energy(
            &params,
            TraceConfig::REFERENCE.disk_outer_radius,
            OrbitDirection::Positive,
        )
        .unwrap();
        assert!(inner > outer && outer > 0.0, "inner {inner}, outer {outer}");
    }

    #[test]
    fn rotating_disk_has_opposite_doppler_shifts_on_each_side() {
        let params = KerrParams::REFERENCE;
        let orbital_radius = 8.0;
        let cartesian_radius =
            sqrt_f32(orbital_radius * orbital_radius + params.spin * params.spin);
        let approaching = circular_equatorial_emitter(
            &params,
            [0.0, -cartesian_radius, 0.0],
            OrbitDirection::Positive,
        )
        .unwrap();
        let receding = circular_equatorial_emitter(
            &params,
            [0.0, cartesian_radius, 0.0],
            OrbitDirection::Positive,
        )
        .unwrap();
        let distant_observer = [1.0, 0.0, 0.0, 0.0];
        let photon_toward_positive_x = [-1.0, 1.0, 0.0, 0.0];
        let blue = redshift_factor(
            photon_toward_positive_x,
            distant_observer,
            approaching.four_velocity,
        )
        .unwrap();
        let red = redshift_factor(
            photon_toward_positive_x,
            distant_observer,
            receding.four_velocity,
        )
        .unwrap();
        assert!(blue > red, "approaching {blue}, receding {red}");
    }

    #[test]
    fn zero_torque_disk_flux_is_dark_at_isco_and_peaks_outside_it() {
        let inner = equatorial_isco(&KerrParams::REFERENCE);
        assert_eq!(zero_torque_thin_disk_flux(inner, inner), Some(0.0));
        let peak_radius = inner * 49.0 / 36.0;
        let peak = zero_torque_thin_disk_flux(peak_radius, inner).unwrap();
        assert!((peak - 1.0).abs() < 2.0e-6, "peak {peak}");
        assert!(zero_torque_thin_disk_flux(inner * 4.0, inner).unwrap() < peak);
        assert_eq!(zero_torque_thin_disk_flux(inner * 0.99, inner), None);
    }

    #[test]
    fn central_ray_captures_and_outward_ray_escapes() {
        let params = KerrParams::REFERENCE;
        let mut capture_ray = camera_null_ray(
            &params,
            &static_camera_tetrad(&params, [30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap(),
            [0.0, 0.0],
        )
        .unwrap();
        assert_eq!(
            trace_fixed(&params, &TraceConfig::REFERENCE, &mut capture_ray),
            TraceEvent::Capture
        );

        let mut escape_ray = camera_null_ray(
            &params,
            &static_camera_tetrad(&params, [30.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap(),
            [0.0, 0.0],
        )
        .unwrap();
        assert_eq!(
            trace_fixed(&params, &TraceConfig::REFERENCE, &mut escape_ray),
            TraceEvent::Escape
        );
    }

    #[test]
    fn midpoint_has_bounded_null_constraint_drift() {
        let params = KerrParams::REFERENCE;
        let camera =
            static_camera_tetrad(&params, [30.0, -2.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                .unwrap();
        let mut ray = camera_null_ray(&params, &camera, [0.35, 0.2]).unwrap();
        let energy_bits = ray.momentum[0].to_bits();
        let initial = hamiltonian(&params, &ray);
        let mut maximum_drift = 0.0_f32;
        for _ in 0..128 {
            midpoint_step(&params, &mut ray, 0.02, 1.0e-3);
            maximum_drift = maximum_drift.max(abs_f32(hamiltonian(&params, &ray) - initial));
        }
        assert_eq!(ray.momentum[0].to_bits(), energy_bits);
        assert!(maximum_drift < 2.0e-4, "null drift {maximum_drift}");
    }

    #[test]
    fn spin_reversal_and_y_reflection_preserve_hamiltonian_flow() {
        let positive = KerrParams::REFERENCE;
        let negative = KerrParams::new(positive.mass, -positive.spin);
        let mut ray = HamiltonianRay::new([0.0, 9.0, 2.5, 1.25], [-1.0, -0.8, 0.17, -0.06]);
        let mut mirrored = HamiltonianRay::new([0.0, 9.0, -2.5, 1.25], [-1.0, -0.8, -0.17, -0.06]);
        assert_close(
            hamiltonian(&positive, &ray),
            hamiltonian(&negative, &mirrored),
            2.0e-7,
        );
        for _ in 0..24 {
            midpoint_step(&positive, &mut ray, 0.02, 1.0e-3);
            midpoint_step(&negative, &mut mirrored, 0.02, 1.0e-3);
        }
        for component in [0, 1, 3] {
            assert_close(
                ray.position[component],
                mirrored.position[component],
                2.0e-5,
            );
            assert_close(
                ray.momentum[component],
                mirrored.momentum[component],
                2.0e-5,
            );
        }
        assert_close(ray.position[2], -mirrored.position[2], 2.0e-5);
        assert_close(ray.momentum[2], -mirrored.momentum[2], 2.0e-5);
    }
}
