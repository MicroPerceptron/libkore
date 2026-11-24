//! Allocator-owned `.ax` construction for one Kerr geodesic transition.
//!
//! This module emits a fixed, branchless SSA graph for an explicit-midpoint
//! (RK2) step in Cartesian Kerr-Schild coordinates. The Hamiltonian gradients
//! use the same centered, radius-relative spatial differences as the scalar
//! reference in [`crate::black_hole`]. Repeating and caching this transition is
//! scheduler policy; the geometry kernel itself has no loop, provider, host,
//! or global-allocation dependency.

use core::alloc::Allocator;

use ax::{
    AllocationError, DType, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId,
};

use crate::black_hole::{KerrParams, TraceConfig, horizon_radius};

/// Maximum ray-plane extent accepted by a single TOSA Level 8K specialization.
pub const MAX_RAYS: u32 = 8_192;

/// Exact operation count of the current one-step specialization.
pub const KERR_STEP_NODE_COUNT: usize = 989;

/// Exact tensor count, including the 26 ABI tensors and 16 scalar constants.
pub const KERR_STEP_TENSOR_COUNT: usize = 1_014;

/// Stable tensor identifiers for the Kerr transition ABI.
///
/// Inputs are structure-of-arrays planes: `X[t,x,y,z]`, `p[t,x,y,z]`, and an
/// active boolean mask. Outputs are the final state, the still-active mask,
/// mutually exclusive capture/disk/escape event masks, and disk-hit data.
/// `active`, `capture`, `disk`, and `escape` form the event representation:
/// active means unresolved, one event mask means terminal, and all-zero means
/// the input lane was already inactive.
pub mod abi {
    use ax::TensorId;

    pub const X_T_IN: TensorId = TensorId(0);
    pub const X_X_IN: TensorId = TensorId(1);
    pub const X_Y_IN: TensorId = TensorId(2);
    pub const X_Z_IN: TensorId = TensorId(3);
    pub const P_T_IN: TensorId = TensorId(4);
    pub const P_X_IN: TensorId = TensorId(5);
    pub const P_Y_IN: TensorId = TensorId(6);
    pub const P_Z_IN: TensorId = TensorId(7);
    pub const ACTIVE_IN: TensorId = TensorId(8);

    pub const X_T_OUT: TensorId = TensorId(9);
    pub const X_X_OUT: TensorId = TensorId(10);
    pub const X_Y_OUT: TensorId = TensorId(11);
    pub const X_Z_OUT: TensorId = TensorId(12);
    pub const P_T_OUT: TensorId = TensorId(13);
    pub const P_X_OUT: TensorId = TensorId(14);
    pub const P_Y_OUT: TensorId = TensorId(15);
    pub const P_Z_OUT: TensorId = TensorId(16);
    pub const ACTIVE_OUT: TensorId = TensorId(17);
    pub const CAPTURE_OUT: TensorId = TensorId(18);
    pub const DISK_OUT: TensorId = TensorId(19);
    pub const ESCAPE_OUT: TensorId = TensorId(20);
    pub const DISK_X_OUT: TensorId = TensorId(21);
    pub const DISK_Y_OUT: TensorId = TensorId(22);
    pub const DISK_Z_OUT: TensorId = TensorId(23);
    pub const DISK_RADIUS_OUT: TensorId = TensorId(24);
    pub const DISK_FRACTION_OUT: TensorId = TensorId(25);

    pub const INPUT_COUNT: u32 = 9;
    pub const OUTPUT_COUNT: u32 = 17;
    pub const TENSOR_COUNT: u32 = INPUT_COUNT + OUTPUT_COUNT;
}

/// Static specialization parameters for one branchless transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrStepGraphSpec {
    /// Number of independent ray lanes in every ABI tensor plane.
    pub rays: u32,
    pub params: KerrParams,
    pub trace: TraceConfig,
    /// Inward terminal surface for this specialization. Exterior observers
    /// use `r_+`; interior past-light-cone cameras use a smaller positive
    /// cutoff to stay clear of the ring singularity.
    pub termination_radius: f32,
}

/// Validation or allocator failure while constructing a Kerr step graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidRayCount(u32),
    InvalidParameters,
    InvalidTraceConfig,
    InvalidTerminationRadius,
    Build(AllocationError),
}

impl From<AllocationError> for GraphError {
    #[inline]
    fn from(value: AllocationError) -> Self {
        Self::Build(value)
    }
}

/// Builds one allocator-owned, backend-neutral Kerr Hamiltonian transition.
///
/// A lane that is inactive on entry is copied through unchanged. A lane
/// already beyond a terminal boundary is frozen at its input state. Otherwise
/// its accepted RK2 candidate is returned, including for a newly terminal
/// event, matching the scalar reference's state mutation. Disk payloads are
/// zero on non-disk lanes; `DISK_Z_OUT` is identically zero for the thin
/// equatorial disk.
///
/// The graph intentionally represents one step. A scheduler can cache this
/// specialization and repeatedly dispatch it while retaining control over
/// compaction, convergence, and device placement instead of materializing a
/// 500-step graph.
pub fn build_kerr_step_graph_in<A: Allocator + Clone>(
    spec: KerrStepGraphSpec,
    allocator: A,
) -> Result<GraphBuf<A>, GraphError> {
    validate_spec(spec)?;
    let mut graph = Emitter::new_in(spec.rays, allocator);

    // ABI declarations remain first so implementation changes cannot shift
    // externally visible tensor identifiers.
    let x_input = [
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
    ];
    let p_input = [
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::INPUT)?,
    ];
    let active_input = graph.abi_tensor(DType::U8, TensorFlags::INPUT)?;

    let x_output = [
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
    ];
    let p_output = [
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
        graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?,
    ];
    let active_output = graph.abi_tensor(DType::U8, TensorFlags::OUTPUT)?;
    let capture_output = graph.abi_tensor(DType::U8, TensorFlags::OUTPUT)?;
    let disk_output = graph.abi_tensor(DType::U8, TensorFlags::OUTPUT)?;
    let escape_output = graph.abi_tensor(DType::U8, TensorFlags::OUTPUT)?;
    let disk_x_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let disk_y_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let disk_z_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let disk_radius_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let disk_fraction_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;

    let constants = Constants::new(&mut graph, spec)?;
    let state = State {
        position: x_input,
        momentum: p_input,
    };

    let first = hamiltonian_rhs(&mut graph, &constants, state)?;
    let current_outward = outward(&mut graph, &constants, state.position, first.position)?;
    let current_capture = graph.compare(OpCode::GREATER_EQUAL, constants.horizon, first.radius)?;
    let current_at_escape = graph.compare(OpCode::GREATER_EQUAL, first.radius, constants.escape)?;
    let current_escape = graph.logical_and(current_at_escape, current_outward)?;

    // Capture has precedence over escape at the input boundary.
    let pre_capture = graph.logical_and(active_input, current_capture)?;
    let not_current_capture = graph.logical_not(current_capture)?;
    let pre_escape = graph.logical_and(active_input, not_current_capture)?;
    let pre_escape = graph.logical_and(pre_escape, current_escape)?;
    let pre_terminal = graph.logical_or(pre_capture, pre_escape)?;
    let step_active = graph.logical_not(pre_terminal)?;
    let step_active = graph.logical_and(active_input, step_active)?;

    let midpoint = midpoint_state(&mut graph, &constants, state, first)?;
    let second = hamiltonian_rhs(&mut graph, &constants, midpoint)?;
    let candidate = accepted_state(&mut graph, &constants, state, second)?;

    // Evaluate the accepted endpoint once for both capture and outward escape.
    let endpoint = position_flow(&mut graph, &constants, candidate)?;
    let next_capture = graph.compare(OpCode::GREATER_EQUAL, constants.horizon, endpoint.radius)?;
    let post_capture = graph.logical_and(step_active, next_capture)?;

    let hit = disk_intersection(&mut graph, &constants, state.position, candidate.position)?;
    let not_next_capture = graph.logical_not(next_capture)?;
    let disk_event = graph.logical_and(step_active, not_next_capture)?;
    let disk_event = graph.logical_and(disk_event, hit.valid)?;

    let next_outward = outward(
        &mut graph,
        &constants,
        candidate.position,
        endpoint.position,
    )?;
    let next_at_escape = graph.compare(OpCode::GREATER_EQUAL, endpoint.radius, constants.escape)?;
    let next_escape = graph.logical_and(next_at_escape, next_outward)?;
    let not_disk = graph.logical_not(disk_event)?;
    let post_escape = graph.logical_and(step_active, not_next_capture)?;
    let post_escape = graph.logical_and(post_escape, not_disk)?;
    let post_escape = graph.logical_and(post_escape, next_escape)?;

    let capture_event = graph.logical_or(pre_capture, post_capture)?;
    let escape_event = graph.logical_or(pre_escape, post_escape)?;
    let post_terminal = graph.logical_or(post_capture, disk_event)?;
    let post_terminal = graph.logical_or(post_terminal, post_escape)?;
    let still_active = graph.logical_not(post_terminal)?;
    let still_active = graph.logical_and(step_active, still_active)?;

    // Only lanes eligible for a step accept the candidate. Newly terminal
    // lanes retain that endpoint; pre-terminal and inactive lanes are frozen.
    let mut final_state = state;
    for component in 0..4 {
        final_state.position[component] = graph.select(
            step_active,
            candidate.position[component],
            state.position[component],
        )?;
    }
    // p_t is stationary for both the stepped and frozen cases.
    final_state.momentum[0] = state.momentum[0];
    for component in 1..4 {
        final_state.momentum[component] = graph.select(
            step_active,
            candidate.momentum[component],
            state.momentum[component],
        )?;
    }

    let disk_x = graph.select(disk_event, hit.position[0], constants.zero)?;
    let disk_y = graph.select(disk_event, hit.position[1], constants.zero)?;
    let disk_radius = graph.select(disk_event, hit.radius, constants.zero)?;
    let disk_fraction = graph.select(disk_event, hit.fraction, constants.zero)?;

    for (input, output) in final_state.position.into_iter().zip(x_output) {
        graph.copy_to(input, output)?;
    }
    for (input, output) in final_state.momentum.into_iter().zip(p_output) {
        graph.copy_to(input, output)?;
    }
    graph.logical_copy_to(still_active, active_output)?;
    graph.logical_copy_to(capture_event, capture_output)?;
    graph.logical_copy_to(disk_event, disk_output)?;
    graph.logical_copy_to(escape_event, escape_output)?;
    graph.copy_to(disk_x, disk_x_output)?;
    graph.copy_to(disk_y, disk_y_output)?;
    graph.copy_to(hit.position[2], disk_z_output)?;
    graph.copy_to(disk_radius, disk_radius_output)?;
    graph.copy_to(disk_fraction, disk_fraction_output)?;
    graph.finish()
}

fn validate_spec(spec: KerrStepGraphSpec) -> Result<(), GraphError> {
    if !(1..=MAX_RAYS).contains(&spec.rays) {
        return Err(GraphError::InvalidRayCount(spec.rays));
    }
    if !spec.params.is_valid() {
        return Err(GraphError::InvalidParameters);
    }
    if !spec.trace.is_valid(&spec.params) {
        return Err(GraphError::InvalidTraceConfig);
    }
    if !spec.termination_radius.is_finite()
        || spec.termination_radius <= 0.0
        || spec.termination_radius > horizon_radius(&spec.params)
    {
        return Err(GraphError::InvalidTerminationRadius);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct State {
    position: [TensorId; 4],
    momentum: [TensorId; 4],
}

#[derive(Clone, Copy)]
struct Flow {
    position: [TensorId; 4],
    momentum: [TensorId; 4],
    radius: TensorId,
}

#[derive(Clone, Copy)]
struct Radius {
    squared: TensorId,
    value: TensorId,
    inverse: TensorId,
    z_squared: TensorId,
}

#[derive(Clone, Copy)]
struct MetricField {
    radius: Radius,
    k_up: [TensorId; 4],
    field: TensorId,
}

#[derive(Clone, Copy)]
struct DiskIntersection {
    valid: TensorId,
    position: [TensorId; 3],
    radius: TensorId,
    fraction: TensorId,
}

struct Constants {
    zero: TensorId,
    one: TensorId,
    half: TensorId,
    two: TensorId,
    a: TensorId,
    a_squared: TensorId,
    four_a_squared: TensorId,
    two_mass: TensorId,
    step: TensorId,
    half_step: TensorId,
    gradient_epsilon: TensorId,
    horizon: TensorId,
    escape: TensorId,
    disk_inner: TensorId,
    disk_outer: TensorId,
    singular_epsilon: TensorId,
}

impl Constants {
    fn new<A: Allocator + Clone>(
        graph: &mut Emitter<A>,
        spec: KerrStepGraphSpec,
    ) -> Result<Self, GraphError> {
        let a_squared = spec.params.spin * spec.params.spin;
        Ok(Self {
            zero: graph.f32_constant(0.0)?,
            one: graph.f32_constant(1.0)?,
            half: graph.f32_constant(0.5)?,
            two: graph.f32_constant(2.0)?,
            a: graph.f32_constant(spec.params.spin)?,
            a_squared: graph.f32_constant(a_squared)?,
            four_a_squared: graph.f32_constant(4.0 * a_squared)?,
            two_mass: graph.f32_constant(2.0 * spec.params.mass)?,
            step: graph.f32_constant(spec.trace.step_size)?,
            half_step: graph.f32_constant(0.5 * spec.trace.step_size)?,
            gradient_epsilon: graph.f32_constant(spec.trace.gradient_epsilon)?,
            horizon: graph.f32_constant(spec.termination_radius)?,
            escape: graph.f32_constant(spec.trace.escape_radius)?,
            disk_inner: graph.f32_constant(spec.trace.disk_inner_radius)?,
            disk_outer: graph.f32_constant(spec.trace.disk_outer_radius)?,
            singular_epsilon: graph.f32_constant(1.0e-12)?,
        })
    }
}

fn midpoint_state<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    state: State,
    flow: Flow,
) -> Result<State, GraphError> {
    let mut midpoint = state;
    for component in 0..4 {
        let delta = graph.mul(constants.half_step, flow.position[component])?;
        midpoint.position[component] = graph.add(state.position[component], delta)?;
    }
    // Stationarity makes dp_t exactly zero. Preserve p_t by alias rather than
    // by floating arithmetic so its bits survive the transition unchanged.
    midpoint.momentum[0] = state.momentum[0];
    for component in 1..4 {
        let delta = graph.mul(constants.half_step, flow.momentum[component])?;
        midpoint.momentum[component] = graph.add(state.momentum[component], delta)?;
    }
    Ok(midpoint)
}

fn accepted_state<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    state: State,
    flow: Flow,
) -> Result<State, GraphError> {
    let mut accepted = state;
    for component in 0..4 {
        let delta = graph.mul(constants.step, flow.position[component])?;
        accepted.position[component] = graph.add(state.position[component], delta)?;
    }
    accepted.momentum[0] = state.momentum[0];
    for component in 1..4 {
        let delta = graph.mul(constants.step, flow.momentum[component])?;
        accepted.momentum[component] = graph.add(state.momentum[component], delta)?;
    }
    Ok(accepted)
}

/// Hamiltonian flow with centered spatial momentum derivatives.
fn hamiltonian_rhs<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    state: State,
) -> Result<Flow, GraphError> {
    let position = position_flow(graph, constants, state)?;
    let scale = graph.maximum(position.radius, constants.one)?;
    let epsilon = graph.mul(constants.gradient_epsilon, scale)?;
    let two_epsilon = graph.mul(constants.two, epsilon)?;
    let inverse_two_epsilon = graph.reciprocal(two_epsilon)?;
    let negative_inverse_two_epsilon = graph.negate(inverse_two_epsilon)?;

    let mut momentum = [constants.zero; 4];
    for axis in 0..3 {
        let mut plus = state;
        let mut minus = state;
        plus.position[axis + 1] = graph.add(state.position[axis + 1], epsilon)?;
        minus.position[axis + 1] = graph.sub(state.position[axis + 1], epsilon)?;
        let h_plus = hamiltonian(graph, constants, plus)?;
        let h_minus = hamiltonian(graph, constants, minus)?;
        let difference = graph.sub(h_plus, h_minus)?;
        momentum[axis + 1] = graph.mul(negative_inverse_two_epsilon, difference)?;
    }
    Ok(Flow {
        position: position.position,
        momentum,
        radius: position.radius,
    })
}

/// `dX/dlambda = g^{-1} p`, plus the radius reused by event logic.
fn position_flow<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    state: State,
) -> Result<Flow, GraphError> {
    let metric = metric_field(graph, constants, spatial(state.position))?;
    let q = k_dot_momentum(graph, metric.k_up, state.momentum)?;
    let field_q = graph.mul(metric.field, q)?;
    let negative_p_t = graph.negate(state.momentum[0])?;
    let mut position = [constants.zero; 4];
    position[0] = graph.add(negative_p_t, field_q)?;
    for ((output, momentum), k_component) in position
        .iter_mut()
        .zip(state.momentum)
        .zip(metric.k_up)
        .skip(1)
    {
        let correction = graph.mul(field_q, k_component)?;
        *output = graph.sub(momentum, correction)?;
    }
    Ok(Flow {
        position,
        momentum: [constants.zero; 4],
        radius: metric.radius.value,
    })
}

/// `H = 1/2 (eta^{mu nu} p_mu p_nu - f (k^mu p_mu)^2)`.
fn hamiltonian<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    state: State,
) -> Result<TensorId, GraphError> {
    let metric = metric_field(graph, constants, spatial(state.position))?;
    let q = k_dot_momentum(graph, metric.k_up, state.momentum)?;
    let p_t_squared = graph.mul(state.momentum[0], state.momentum[0])?;
    let p_x_squared = graph.mul(state.momentum[1], state.momentum[1])?;
    let p_y_squared = graph.mul(state.momentum[2], state.momentum[2])?;
    let p_z_squared = graph.mul(state.momentum[3], state.momentum[3])?;
    let spatial_xy = graph.add(p_x_squared, p_y_squared)?;
    let spatial_norm = graph.add(spatial_xy, p_z_squared)?;
    let minkowski = graph.sub(spatial_norm, p_t_squared)?;
    let q_squared = graph.mul(q, q)?;
    let perturbation = graph.mul(metric.field, q_squared)?;
    let doubled = graph.sub(minkowski, perturbation)?;
    graph.mul(constants.half, doubled)
}

fn metric_field<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    point: [TensorId; 3],
) -> Result<MetricField, GraphError> {
    let radius = kerr_radius(graph, constants, point)?;
    let denominator = graph.add(radius.squared, constants.a_squared)?;
    let denominator = graph.maximum(denominator, constants.singular_epsilon)?;
    let inverse_denominator = graph.reciprocal(denominator)?;

    let r_x = graph.mul(radius.value, point[0])?;
    let a_y = graph.mul(constants.a, point[1])?;
    let k_x_numerator = graph.add(r_x, a_y)?;
    let k_x = graph.mul(k_x_numerator, inverse_denominator)?;

    let r_y = graph.mul(radius.value, point[1])?;
    let a_x = graph.mul(constants.a, point[0])?;
    let k_y_numerator = graph.sub(r_y, a_x)?;
    let k_y = graph.mul(k_y_numerator, inverse_denominator)?;
    let k_z = graph.mul(point[2], radius.inverse)?;

    let r_fourth = graph.mul(radius.squared, radius.squared)?;
    let a2_z2 = graph.mul(constants.a_squared, radius.z_squared)?;
    let field_denominator = graph.add(r_fourth, a2_z2)?;
    let field_denominator = graph.maximum(field_denominator, constants.singular_epsilon)?;
    let inverse_field_denominator = graph.reciprocal(field_denominator)?;
    let r_cubed = graph.mul(radius.squared, radius.value)?;
    let field_numerator = graph.mul(constants.two_mass, r_cubed)?;
    let field = graph.mul(field_numerator, inverse_field_denominator)?;

    Ok(MetricField {
        radius,
        k_up: [constants.one, k_x, k_y, k_z],
        field,
    })
}

fn kerr_radius<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    point: [TensorId; 3],
) -> Result<Radius, GraphError> {
    let x_squared = graph.mul(point[0], point[0])?;
    let y_squared = graph.mul(point[1], point[1])?;
    let z_squared = graph.mul(point[2], point[2])?;
    let xy_squared = graph.add(x_squared, y_squared)?;
    let norm_squared = graph.add(xy_squared, z_squared)?;
    let s = graph.sub(norm_squared, constants.a_squared)?;
    let s_squared = graph.mul(s, s)?;
    let z_term = graph.mul(constants.four_a_squared, z_squared)?;
    let discriminant = graph.add(s_squared, z_term)?;
    let safe_discriminant = graph.maximum(discriminant, constants.singular_epsilon)?;
    let inverse_discriminant_root = graph.rsqrt(safe_discriminant)?;
    let discriminant_root = graph.reciprocal(inverse_discriminant_root)?;
    let radial_sum = graph.add(s, discriminant_root)?;
    let squared = graph.mul(constants.half, radial_sum)?;
    let safe_squared = graph.maximum(squared, constants.singular_epsilon)?;
    let inverse = graph.rsqrt(safe_squared)?;
    let value = graph.reciprocal(inverse)?;
    Ok(Radius {
        squared,
        value,
        inverse,
        z_squared,
    })
}

fn k_dot_momentum<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    k_up: [TensorId; 4],
    momentum: [TensorId; 4],
) -> Result<TensorId, GraphError> {
    // k^t = -1 in the (-,+,+,+) convention. MetricField stores +1 in the
    // first slot so it can be broadcast as a constant without another tensor.
    let negative_p_t = graph.negate(momentum[0])?;
    let x_term = graph.mul(k_up[1], momentum[1])?;
    let y_term = graph.mul(k_up[2], momentum[2])?;
    let z_term = graph.mul(k_up[3], momentum[3])?;
    let tx = graph.add(negative_p_t, x_term)?;
    let txy = graph.add(tx, y_term)?;
    graph.add(txy, z_term)
}

fn outward<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    position: [TensorId; 4],
    velocity: [TensorId; 4],
) -> Result<TensorId, GraphError> {
    let x = graph.mul(position[1], velocity[1])?;
    let y = graph.mul(position[2], velocity[2])?;
    let z = graph.mul(position[3], velocity[3])?;
    let xy = graph.add(x, y)?;
    let radial_rate = graph.add(xy, z)?;
    graph.compare(OpCode::GREATER, radial_rate, constants.zero)
}

fn disk_intersection<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    from: [TensorId; 4],
    to: [TensorId; 4],
) -> Result<DiskIntersection, GraphError> {
    let from_positive = graph.compare(OpCode::GREATER, from[3], constants.zero)?;
    let to_nonpositive = graph.compare(OpCode::GREATER_EQUAL, constants.zero, to[3])?;
    let positive_crossing = graph.logical_and(from_positive, to_nonpositive)?;
    let from_negative = graph.compare(OpCode::GREATER, constants.zero, from[3])?;
    let to_nonnegative = graph.compare(OpCode::GREATER_EQUAL, to[3], constants.zero)?;
    let negative_crossing = graph.logical_and(from_negative, to_nonnegative)?;
    let crossed = graph.logical_or(positive_crossing, negative_crossing)?;

    let denominator = graph.sub(from[3], to[3])?;
    // Both SELECT branches are eagerly evaluated by most accelerator IRs.
    // Substitute one on non-crossing lanes before RECIPROCAL so a stationary
    // z=0 lane cannot manufacture infinities in the unused hit payload.
    let safe_denominator = graph.select(crossed, denominator, constants.one)?;
    let inverse_denominator = graph.reciprocal(safe_denominator)?;
    let fraction = graph.mul(from[3], inverse_denominator)?;
    let fraction = graph.maximum(fraction, constants.zero)?;
    let fraction = graph.minimum(fraction, constants.one)?;

    let x_delta = graph.sub(to[1], from[1])?;
    let x_offset = graph.mul(fraction, x_delta)?;
    let hit_x = graph.add(from[1], x_offset)?;
    let y_delta = graph.sub(to[2], from[2])?;
    let y_offset = graph.mul(fraction, y_delta)?;
    let hit_y = graph.add(from[2], y_offset)?;
    // Materialize a plane-shaped zero for the radius expression. Scalar `[1]`
    // constants may broadcast with a ray plane, but `[1] * [1] -> [rays]` is
    // not a valid TOSA elementwise result shape.
    let hit_z = graph.mul(fraction, constants.zero)?;
    let hit_position = [hit_x, hit_y, hit_z];
    let hit_radius = kerr_radius(graph, constants, hit_position)?.value;
    let outside_inner = graph.compare(OpCode::GREATER_EQUAL, hit_radius, constants.disk_inner)?;
    let inside_outer = graph.compare(OpCode::GREATER_EQUAL, constants.disk_outer, hit_radius)?;
    let in_disk = graph.logical_and(outside_inner, inside_outer)?;
    let valid = graph.logical_and(crossed, in_disk)?;
    Ok(DiskIntersection {
        valid,
        position: hit_position,
        radius: hit_radius,
        fraction,
    })
}

#[inline]
fn spatial(position: [TensorId; 4]) -> [TensorId; 3] {
    [position[1], position[2], position[3]]
}

struct Emitter<A: Allocator> {
    builder: GraphBuilder<A>,
    plane: [u32; 1],
}

impl<A: Allocator + Clone> Emitter<A> {
    fn new_in(rays: u32, allocator: A) -> Self {
        Self {
            builder: GraphBuilder::new_in(allocator),
            plane: [rays],
        }
    }

    fn abi_tensor(&mut self, dtype: DType, flags: TensorFlags) -> Result<TensorId, GraphError> {
        Ok(self.builder.tensor(dtype, flags, &self.plane, &[])?)
    }

    fn f32_constant(&mut self, value: f32) -> Result<TensorId, GraphError> {
        Ok(self.builder.tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &value.to_le_bytes(),
        )?)
    }

    fn temporary(&mut self, dtype: DType) -> Result<TensorId, GraphError> {
        Ok(self
            .builder
            .tensor(dtype, TensorFlags::default(), &self.plane, &[])?)
    }

    fn unary(&mut self, op: OpCode, input: TensorId, dtype: DType) -> Result<TensorId, GraphError> {
        let output = self.temporary(dtype)?;
        self.builder
            .node(op, NodeFlags::default(), &[input], &[output], &[])?;
        Ok(output)
    }

    fn binary(
        &mut self,
        op: OpCode,
        left: TensorId,
        right: TensorId,
        dtype: DType,
    ) -> Result<TensorId, GraphError> {
        let output = self.temporary(dtype)?;
        self.builder
            .node(op, NodeFlags::default(), &[left, right], &[output], &[])?;
        Ok(output)
    }

    fn add(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::ADD, left, right, DType::F32)
    }

    fn sub(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::SUB, left, right, DType::F32)
    }

    fn mul(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::MUL, left, right, DType::F32)
    }

    fn maximum(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::MAXIMUM, left, right, DType::F32)
    }

    fn minimum(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::MINIMUM, left, right, DType::F32)
    }

    fn negate(&mut self, input: TensorId) -> Result<TensorId, GraphError> {
        self.unary(OpCode::NEGATE, input, DType::F32)
    }

    fn reciprocal(&mut self, input: TensorId) -> Result<TensorId, GraphError> {
        self.unary(OpCode::RECIPROCAL, input, DType::F32)
    }

    fn rsqrt(&mut self, input: TensorId) -> Result<TensorId, GraphError> {
        self.unary(OpCode::RSQRT, input, DType::F32)
    }

    fn compare(
        &mut self,
        op: OpCode,
        left: TensorId,
        right: TensorId,
    ) -> Result<TensorId, GraphError> {
        self.binary(op, left, right, DType::U8)
    }

    fn logical_and(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::LOGICAL_AND, left, right, DType::U8)
    }

    fn logical_or(&mut self, left: TensorId, right: TensorId) -> Result<TensorId, GraphError> {
        self.binary(OpCode::LOGICAL_OR, left, right, DType::U8)
    }

    fn logical_not(&mut self, input: TensorId) -> Result<TensorId, GraphError> {
        self.unary(OpCode::LOGICAL_NOT, input, DType::U8)
    }

    fn select(
        &mut self,
        condition: TensorId,
        when_true: TensorId,
        when_false: TensorId,
    ) -> Result<TensorId, GraphError> {
        let output = self.temporary(DType::F32)?;
        self.builder.node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[condition, when_true, when_false],
            &[output],
            &[],
        )?;
        Ok(output)
    }

    fn copy_to(&mut self, input: TensorId, output: TensorId) -> Result<(), GraphError> {
        Ok(self
            .builder
            .node(OpCode::COPY, NodeFlags::default(), &[input], &[output], &[])?)
    }

    fn logical_copy_to(&mut self, input: TensorId, output: TensorId) -> Result<(), GraphError> {
        Ok(self.builder.node(
            OpCode::LOGICAL_OR,
            NodeFlags::default(),
            &[input, input],
            &[output],
            &[],
        )?)
    }

    fn finish(self) -> Result<GraphBuf<A>, GraphError> {
        Ok(self.builder.finish()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr::NonNull;
    use std::alloc::{AllocError, Allocator, Layout, System};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct AllocationCounts {
        live: AtomicUsize,
        total: AtomicUsize,
    }

    #[derive(Clone)]
    struct TrackingAllocator(Arc<AllocationCounts>);

    // SAFETY: System owns each allocation. This wrapper only observes
    // successful ownership transitions and delegates the original pair.
    unsafe impl Allocator for TrackingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let allocation = System.allocate(layout)?;
            self.0.live.fetch_add(1, Ordering::Relaxed);
            self.0.total.fetch_add(1, Ordering::Relaxed);
            Ok(allocation)
        }

        unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
            self.0.live.fetch_sub(1, Ordering::Relaxed);
            // SAFETY: this is the pointer/layout pair produced by System.
            unsafe { System.deallocate(pointer, layout) };
        }
    }

    fn reference_spec() -> KerrStepGraphSpec {
        KerrStepGraphSpec {
            rays: 64,
            params: KerrParams::REFERENCE,
            trace: TraceConfig::REFERENCE,
            termination_radius: horizon_radius(&KerrParams::REFERENCE),
        }
    }

    #[test]
    fn graph_has_stable_abi_standard_ops_and_allocator_ownership() {
        let counts = Arc::new(AllocationCounts::default());
        {
            let bytes =
                build_kerr_step_graph_in(reference_spec(), TrackingAllocator(Arc::clone(&counts)))
                    .unwrap();
            assert!(counts.live.load(Ordering::Relaxed) > 0);
            let graph = bytes.parse().unwrap();
            assert_eq!(graph.tensor_count(), KERR_STEP_TENSOR_COUNT);
            assert_eq!(graph.node_count(), KERR_STEP_NODE_COUNT);
            let schedule =
                ax::Schedule::analyze_in(&graph, TrackingAllocator(Arc::clone(&counts))).unwrap();
            assert_eq!(schedule.info().nodes, graph.node_count());

            let stable_abi = [
                abi::X_T_IN,
                abi::X_X_IN,
                abi::X_Y_IN,
                abi::X_Z_IN,
                abi::P_T_IN,
                abi::P_X_IN,
                abi::P_Y_IN,
                abi::P_Z_IN,
                abi::ACTIVE_IN,
                abi::X_T_OUT,
                abi::X_X_OUT,
                abi::X_Y_OUT,
                abi::X_Z_OUT,
                abi::P_T_OUT,
                abi::P_X_OUT,
                abi::P_Y_OUT,
                abi::P_Z_OUT,
                abi::ACTIVE_OUT,
                abi::CAPTURE_OUT,
                abi::DISK_OUT,
                abi::ESCAPE_OUT,
                abi::DISK_X_OUT,
                abi::DISK_Y_OUT,
                abi::DISK_Z_OUT,
                abi::DISK_RADIUS_OUT,
                abi::DISK_FRACTION_OUT,
            ];
            for (expected, actual) in stable_abi.into_iter().enumerate() {
                assert_eq!(actual, TensorId(expected as u32));
            }

            for id in 0..abi::INPUT_COUNT {
                let tensor = graph.tensor(TensorId(id)).unwrap();
                assert_eq!(tensor.flags, TensorFlags::INPUT);
                assert_eq!(tensor.dtype, if id == 8 { DType::U8 } else { DType::F32 });
                assert_eq!(tensor.dimensions().next(), Some(64));
            }
            for id in abi::INPUT_COUNT..abi::TENSOR_COUNT {
                let tensor = graph.tensor(TensorId(id)).unwrap();
                assert_eq!(tensor.flags, TensorFlags::OUTPUT);
                assert_eq!(
                    tensor.dtype,
                    if (17..=20).contains(&id) {
                        DType::U8
                    } else {
                        DType::F32
                    }
                );
                assert_eq!(tensor.dimensions().next(), Some(64));
            }

            let mut op_counts = [0_usize; 35];
            for index in 0..graph.node_count() {
                let node = graph.node(index).unwrap();
                assert_ne!(node.op, OpCode::DIV);
                assert!(node.attributes().is_empty());
                assert!(matches!(
                    node.op,
                    OpCode::ADD
                        | OpCode::SUB
                        | OpCode::MUL
                        | OpCode::MAXIMUM
                        | OpCode::MINIMUM
                        | OpCode::NEGATE
                        | OpCode::RECIPROCAL
                        | OpCode::RSQRT
                        | OpCode::SELECT
                        | OpCode::GREATER
                        | OpCode::GREATER_EQUAL
                        | OpCode::LOGICAL_AND
                        | OpCode::LOGICAL_OR
                        | OpCode::LOGICAL_NOT
                        | OpCode::COPY
                ));
                op_counts[node.op.0 as usize] += 1;
            }
            assert_eq!(
                op_counts,
                [
                    0, 0, 207, 79, 451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 65, 1, 0, 0, 0, 0, 5,
                    20, 65, 32, 0, 12, 0, 4, 8, 17, 10, 0,
                ]
            );

            // Output emission is the final 17 nodes. p_t is the fifth state
            // copy and must reference the input tensor directly, proving that
            // stationarity is preserved without a floating-point round trip.
            let p_t_copy = graph.node(graph.node_count() - 13).unwrap();
            assert_eq!(p_t_copy.op, OpCode::COPY);
            assert_eq!(p_t_copy.inputs().next(), Some(abi::P_T_IN));
            assert_eq!(p_t_copy.outputs().next(), Some(abi::P_T_OUT));
        }
        assert!(counts.total.load(Ordering::Relaxed) > 0);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn invalid_specs_fail_before_allocating() {
        let counts = Arc::new(AllocationCounts::default());
        let allocator = TrackingAllocator(Arc::clone(&counts));

        let mut spec = reference_spec();
        spec.rays = 0;
        assert!(matches!(
            build_kerr_step_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidRayCount(0))
        ));
        spec = reference_spec();
        spec.rays = MAX_RAYS + 1;
        assert!(matches!(
            build_kerr_step_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidRayCount(rays)) if rays == MAX_RAYS + 1
        ));
        spec = reference_spec();
        spec.params.mass = f32::NAN;
        assert!(matches!(
            build_kerr_step_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidParameters)
        ));
        spec = reference_spec();
        spec.trace.step_size = 0.0;
        assert!(matches!(
            build_kerr_step_graph_in(spec, allocator),
            Err(GraphError::InvalidTraceConfig)
        ));
        assert_eq!(counts.total.load(Ordering::Relaxed), 0);
    }
}
