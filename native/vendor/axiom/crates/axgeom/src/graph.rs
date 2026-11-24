//! Allocator-owned `.ax` construction for the Dneg ray integrator.
//!
//! The emitted graph is a fixed, branchless SSA program. Every ray lane is
//! advanced independently, and the `active` mask freezes a lane at the first
//! outward escape sample. Geometry parameters and integration settings are
//! graph constants, so no host callback or backend-specific operation is
//! required at execution time.

use core::alloc::Allocator;

use ax::{
    AllocationError, DType, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId,
};

use crate::wormhole::{DnegParams, TraceConfig};

const MAX_RAYS: u32 = 8_192;
const TWO_OVER_PI: f32 = 2.0 / core::f32::consts::PI;
const RAJAN_A: f32 = 0.2447;
const RAJAN_B: f32 = 0.0663;

// One initial impact square, one initial r/r' evaluation, one initial escape
// test, and four ABI output copies surround the unrolled steps.
const FIXED_NODE_COUNT: u32 = 42;
const NODES_PER_STEP: u32 = 62;

/// Stable tensor IDs for the wormhole graph ABI.
pub mod abi {
    use ax::TensorId;

    pub const ELL_IN: TensorId = TensorId(0);
    pub const PHI_IN: TensorId = TensorId(1);
    pub const P_ELL_IN: TensorId = TensorId(2);
    pub const IMPACT_IN: TensorId = TensorId(3);
    pub const ACTIVE_IN: TensorId = TensorId(4);

    pub const ELL_OUT: TensorId = TensorId(5);
    pub const PHI_OUT: TensorId = TensorId(6);
    pub const P_ELL_OUT: TensorId = TensorId(7);
    pub const ACTIVE_OUT: TensorId = TensorId(8);
}

/// Static specialization parameters for a Dneg tracing graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WormholeGraphSpec {
    /// Number of independent ray lanes in each ABI tensor plane.
    pub rays: u32,
    /// Number of velocity-Verlet steps unrolled into the graph.
    pub steps: u32,
    pub params: DnegParams,
    pub trace: TraceConfig,
}

/// Validation or allocator failure while constructing a wormhole graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidRayCount(u32),
    InvalidStepCount(u32),
    InvalidParameters,
    InvalidTraceConfig,
    GraphTooLarge,
    Build(AllocationError),
}

impl From<AllocationError> for GraphError {
    #[inline]
    fn from(value: AllocationError) -> Self {
        Self::Build(value)
    }
}

/// Builds an allocator-owned, backend-neutral Dneg ray-tracing graph.
///
/// Tensor IDs `0`, `1`, `2`, `3`, and `4` are ABI inputs `ell`, `phi`, `p_ell`,
/// `impact`, and `active`, respectively. Tensor IDs `5`, `6`, `7`, and `8` are
/// ABI outputs `ell`, `phi`, `p_ell`, and `active`, respectively.
/// Floating-point planes are `F32`, the active mask is `U8`, and every plane
/// has shape `[spec.rays]`.
///
/// `spec.steps` is the compiled unroll count. `TraceConfig::max_steps` remains
/// useful to callers as policy metadata, but does not introduce a runtime loop
/// or alter the specialized graph shape.
pub fn build_wormhole_graph_in<A: Allocator + Clone>(
    spec: WormholeGraphSpec,
    allocator: A,
) -> Result<GraphBuf<A>, GraphError> {
    validate_spec(spec)?;

    let mut graph = Emitter::new_in(spec.rays, allocator);

    // ABI tensors are deliberately declared first so their IDs and order are
    // independent of changes to the implementation's constants or algebra.
    let ell_input = graph.abi_tensor(DType::F32, TensorFlags::INPUT)?;
    let phi_input = graph.abi_tensor(DType::F32, TensorFlags::INPUT)?;
    let p_ell_input = graph.abi_tensor(DType::F32, TensorFlags::INPUT)?;
    let impact_input = graph.abi_tensor(DType::F32, TensorFlags::INPUT)?;
    let active_input = graph.abi_tensor(DType::U8, TensorFlags::INPUT)?;

    let ell_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let phi_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let p_ell_output = graph.abi_tensor(DType::F32, TensorFlags::OUTPUT)?;
    let active_output = graph.abi_tensor(DType::U8, TensorFlags::OUTPUT)?;

    let constants = Constants::new(&mut graph, spec)?;
    let impact_squared = graph.binary(OpCode::MUL, impact_input, impact_input, DType::F32)?;

    let mut ell = ell_input;
    let mut phi = phi_input;
    let mut p_ell = p_ell_input;
    let (mut radius, mut radius_derivative) =
        dneg_radius_and_derivative(&mut graph, &constants, ell)?;
    let mut active = deactivate_if_escaped(&mut graph, &constants, active_input, ell, p_ell)?;

    for _ in 0..spec.steps {
        let (force_old, inverse_radius_squared_old) = radial_force_and_inverse_radius_squared(
            &mut graph,
            impact_squared,
            radius,
            radius_derivative,
        )?;
        let first_kick = graph.binary(OpCode::MUL, constants.half_step, force_old, DType::F32)?;
        let p_half = graph.binary(OpCode::ADD, p_ell, first_kick, DType::F32)?;
        let drift = graph.binary(OpCode::MUL, constants.step, p_half, DType::F32)?;
        let ell_candidate = graph.binary(OpCode::ADD, ell, drift, DType::F32)?;

        let (radius_candidate, radius_derivative_candidate) =
            dneg_radius_and_derivative(&mut graph, &constants, ell_candidate)?;
        let (force_new, inverse_radius_squared_new) = radial_force_and_inverse_radius_squared(
            &mut graph,
            impact_squared,
            radius_candidate,
            radius_derivative_candidate,
        )?;
        let second_kick = graph.binary(OpCode::MUL, constants.half_step, force_new, DType::F32)?;
        let p_ell_candidate = graph.binary(OpCode::ADD, p_half, second_kick, DType::F32)?;

        // Trapezoidal angular integration preserves the time symmetry of the
        // kick-drift-kick radial update.
        let inverse_radius_squared_sum = graph.binary(
            OpCode::ADD,
            inverse_radius_squared_old,
            inverse_radius_squared_new,
            DType::F32,
        )?;
        let angular_rate_sum = graph.binary(
            OpCode::MUL,
            impact_input,
            inverse_radius_squared_sum,
            DType::F32,
        )?;
        let angular_delta = graph.binary(
            OpCode::MUL,
            constants.half_step,
            angular_rate_sum,
            DType::F32,
        )?;
        let phi_candidate = graph.binary(OpCode::ADD, phi, angular_delta, DType::F32)?;

        // SELECT implements the active-mask freeze without control flow. Keep
        // r and r' in lockstep with the selected ell so the next step can
        // reuse the expensive Dneg evaluation.
        ell = graph.select(active, ell_candidate, ell, DType::F32)?;
        phi = graph.select(active, phi_candidate, phi, DType::F32)?;
        p_ell = graph.select(active, p_ell_candidate, p_ell, DType::F32)?;
        radius = graph.select(active, radius_candidate, radius, DType::F32)?;
        radius_derivative = graph.select(
            active,
            radius_derivative_candidate,
            radius_derivative,
            DType::F32,
        )?;

        active = deactivate_if_escaped(
            &mut graph,
            &constants,
            active,
            ell_candidate,
            p_ell_candidate,
        )?;
    }

    graph.copy_to(ell, ell_output)?;
    graph.copy_to(phi, phi_output)?;
    graph.copy_to(p_ell, p_ell_output)?;
    graph.logical_copy_to(active, active_output)?;
    graph.finish()
}

fn validate_spec(spec: WormholeGraphSpec) -> Result<(), GraphError> {
    if !(1..=MAX_RAYS).contains(&spec.rays) {
        return Err(GraphError::InvalidRayCount(spec.rays));
    }
    if spec.steps == 0 {
        return Err(GraphError::InvalidStepCount(spec.steps));
    }
    if !spec.params.is_valid() || !(TWO_OVER_PI / spec.params.m).is_finite() {
        return Err(GraphError::InvalidParameters);
    }
    if !spec.trace.is_valid() || spec.trace.escape_ell <= spec.params.a {
        return Err(GraphError::InvalidTraceConfig);
    }
    if NODES_PER_STEP
        .checked_mul(spec.steps)
        .and_then(|nodes| nodes.checked_add(FIXED_NODE_COUNT))
        .is_none()
    {
        return Err(GraphError::GraphTooLarge);
    }
    Ok(())
}

struct Constants {
    zero: TensorId,
    one: TensorId,
    half: TensorId,
    two_over_pi: TensorId,
    pi_over_four: TensorId,
    pi_over_two: TensorId,
    rajan_a: TensorId,
    rajan_b: TensorId,
    rho: TensorId,
    a: TensorId,
    m: TensorId,
    x_scale: TensorId,
    step: TensorId,
    half_step: TensorId,
    escape: TensorId,
    negative_escape: TensorId,
}

impl Constants {
    fn new<A: Allocator + Clone>(
        graph: &mut Emitter<A>,
        spec: WormholeGraphSpec,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            zero: graph.f32_constant(0.0)?,
            one: graph.f32_constant(1.0)?,
            half: graph.f32_constant(0.5)?,
            two_over_pi: graph.f32_constant(TWO_OVER_PI)?,
            pi_over_four: graph.f32_constant(core::f32::consts::FRAC_PI_4)?,
            pi_over_two: graph.f32_constant(core::f32::consts::FRAC_PI_2)?,
            rajan_a: graph.f32_constant(RAJAN_A)?,
            rajan_b: graph.f32_constant(RAJAN_B)?,
            rho: graph.f32_constant(spec.params.rho)?,
            a: graph.f32_constant(spec.params.a)?,
            m: graph.f32_constant(spec.params.m)?,
            x_scale: graph.f32_constant(TWO_OVER_PI / spec.params.m)?,
            step: graph.f32_constant(spec.trace.step_size)?,
            half_step: graph.f32_constant(0.5 * spec.trace.step_size)?,
            escape: graph.f32_constant(spec.trace.escape_ell)?,
            negative_escape: graph.f32_constant(-spec.trace.escape_ell)?,
        })
    }
}

fn dneg_radius_and_derivative<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    ell: TensorId,
) -> Result<(TensorId, TensorId), GraphError> {
    let absolute_ell = graph.unary(OpCode::ABS, ell, DType::F32)?;
    let exterior = graph.binary(OpCode::SUB, absolute_ell, constants.a, DType::F32)?;
    let exterior = graph.binary(OpCode::MAXIMUM, exterior, constants.zero, DType::F32)?;
    let x = graph.binary(OpCode::MUL, exterior, constants.x_scale, DType::F32)?;
    let atan_x = nonnegative_atan(graph, constants, x)?;

    // r = rho + m * (x atan(x) - 0.5 log(1 + x^2)).
    let x_squared = graph.binary(OpCode::MUL, x, x, DType::F32)?;
    let log_argument = graph.binary(OpCode::ADD, constants.one, x_squared, DType::F32)?;
    let logarithm = graph.unary(OpCode::LOG, log_argument, DType::F32)?;
    let x_atan_x = graph.binary(OpCode::MUL, x, atan_x, DType::F32)?;
    let half_logarithm = graph.binary(OpCode::MUL, constants.half, logarithm, DType::F32)?;
    let radial_shape = graph.binary(OpCode::SUB, x_atan_x, half_logarithm, DType::F32)?;
    let scaled_shape = graph.binary(OpCode::MUL, constants.m, radial_shape, DType::F32)?;
    let radius = graph.binary(OpCode::ADD, constants.rho, scaled_shape, DType::F32)?;

    // r' = sign(ell) * (2/pi) atan(x). The magnitude is already zero in
    // the cylindrical interior because x was clamped to zero.
    let derivative_magnitude =
        graph.binary(OpCode::MUL, constants.two_over_pi, atan_x, DType::F32)?;
    let negative_derivative = graph.unary(OpCode::NEGATE, derivative_magnitude, DType::F32)?;
    let negative_side = graph.binary(OpCode::GREATER, constants.zero, ell, DType::U8)?;
    let radius_derivative = graph.select(
        negative_side,
        negative_derivative,
        derivative_magnitude,
        DType::F32,
    )?;
    Ok((radius, radius_derivative))
}

/// Rajan's atan approximation on `[0, 1]`, with `atan(x) = pi/2 - atan(1/x)`
/// for `x > 1`. Clamping the reciprocal denominator to one keeps every lane
/// finite even though SELECT is evaluated branchlessly.
fn nonnegative_atan<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    x: TensorId,
) -> Result<TensorId, GraphError> {
    let use_reciprocal = graph.binary(OpCode::GREATER, x, constants.one, DType::U8)?;
    let denominator = graph.binary(OpCode::MAXIMUM, x, constants.one, DType::F32)?;
    let reciprocal = graph.unary(OpCode::RECIPROCAL, denominator, DType::F32)?;
    let reduced = graph.select(use_reciprocal, reciprocal, x, DType::F32)?;

    // atan(z) ~= z * (pi/4 - (z - 1) * (0.2447 + 0.0663 z)).
    let z_minus_one = graph.binary(OpCode::SUB, reduced, constants.one, DType::F32)?;
    let linear_term = graph.binary(OpCode::MUL, constants.rajan_b, reduced, DType::F32)?;
    let coefficient = graph.binary(OpCode::ADD, constants.rajan_a, linear_term, DType::F32)?;
    let correction = graph.binary(OpCode::MUL, z_minus_one, coefficient, DType::F32)?;
    let slope = graph.binary(OpCode::SUB, constants.pi_over_four, correction, DType::F32)?;
    let reduced_atan = graph.binary(OpCode::MUL, reduced, slope, DType::F32)?;
    let complementary_atan =
        graph.binary(OpCode::SUB, constants.pi_over_two, reduced_atan, DType::F32)?;
    graph.select(use_reciprocal, complementary_atan, reduced_atan, DType::F32)
}

fn radial_force_and_inverse_radius_squared<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    impact_squared: TensorId,
    radius: TensorId,
    radius_derivative: TensorId,
) -> Result<(TensorId, TensorId), GraphError> {
    // `DnegParams::is_valid` guarantees a strictly positive throat radius, so
    // every emitted radius is positive. Pay for one reciprocal and derive the
    // two powers needed by the force and angular rate from it.
    let inverse_radius = graph.unary(OpCode::RECIPROCAL, radius, DType::F32)?;
    let inverse_radius_squared =
        graph.binary(OpCode::MUL, inverse_radius, inverse_radius, DType::F32)?;
    let inverse_radius_cubed = graph.binary(
        OpCode::MUL,
        inverse_radius_squared,
        inverse_radius,
        DType::F32,
    )?;
    let numerator = graph.binary(OpCode::MUL, impact_squared, radius_derivative, DType::F32)?;
    let force = graph.binary(OpCode::MUL, numerator, inverse_radius_cubed, DType::F32)?;
    Ok((force, inverse_radius_squared))
}

fn deactivate_if_escaped<A: Allocator + Clone>(
    graph: &mut Emitter<A>,
    constants: &Constants,
    active: TensorId,
    ell: TensorId,
    p_ell: TensorId,
) -> Result<TensorId, GraphError> {
    let positive_position =
        graph.binary(OpCode::GREATER_EQUAL, ell, constants.escape, DType::U8)?;
    let positive_direction = graph.binary(OpCode::GREATER, p_ell, constants.zero, DType::U8)?;
    let positive_escape = graph.binary(
        OpCode::LOGICAL_AND,
        positive_position,
        positive_direction,
        DType::U8,
    )?;

    let negative_position = graph.binary(
        OpCode::GREATER_EQUAL,
        constants.negative_escape,
        ell,
        DType::U8,
    )?;
    let negative_direction = graph.binary(OpCode::GREATER, constants.zero, p_ell, DType::U8)?;
    let negative_escape = graph.binary(
        OpCode::LOGICAL_AND,
        negative_position,
        negative_direction,
        DType::U8,
    )?;
    let escaped = graph.binary(
        OpCode::LOGICAL_OR,
        positive_escape,
        negative_escape,
        DType::U8,
    )?;
    let not_escaped = graph.unary(OpCode::LOGICAL_NOT, escaped, DType::U8)?;
    graph.binary(OpCode::LOGICAL_AND, active, not_escaped, DType::U8)
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

    fn select(
        &mut self,
        condition: TensorId,
        when_true: TensorId,
        when_false: TensorId,
        dtype: DType,
    ) -> Result<TensorId, GraphError> {
        let output = self.temporary(dtype)?;
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
    use std::alloc::{AllocError, Layout, System};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::vec::Vec;

    #[derive(Default)]
    struct AllocationCounts {
        live: AtomicUsize,
        total: AtomicUsize,
    }

    #[derive(Clone)]
    struct TrackingAllocator(Arc<AllocationCounts>);

    // SAFETY: System performs the allocation. The wrapper only observes
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

    fn reference_spec(steps: u32) -> WormholeGraphSpec {
        WormholeGraphSpec {
            rays: 64,
            steps,
            params: DnegParams::REFERENCE,
            trace: TraceConfig::new(0.1, steps, 10.0),
        }
    }

    #[test]
    fn graph_has_stable_abi_standard_ops_and_linear_growth() {
        let counts = Arc::new(AllocationCounts::default());
        {
            let graph =
                build_wormhole_graph_in(reference_spec(2), TrackingAllocator(Arc::clone(&counts)))
                    .unwrap();
            assert!(counts.live.load(Ordering::Relaxed) > 0);

            let graph = graph.parse().unwrap();
            assert_eq!(
                graph.node_count(),
                (FIXED_NODE_COUNT + 2 * NODES_PER_STEP) as usize
            );
            let schedule =
                ax::Schedule::analyze_in(&graph, TrackingAllocator(Arc::clone(&counts))).unwrap();
            assert_eq!(schedule.info().nodes, graph.node_count());
            assert_eq!(schedule.info().tensors, graph.tensor_count());

            let expected_abi = [
                (DType::F32, TensorFlags::INPUT),
                (DType::F32, TensorFlags::INPUT),
                (DType::F32, TensorFlags::INPUT),
                (DType::F32, TensorFlags::INPUT),
                (DType::U8, TensorFlags::INPUT),
                (DType::F32, TensorFlags::OUTPUT),
                (DType::F32, TensorFlags::OUTPUT),
                (DType::F32, TensorFlags::OUTPUT),
                (DType::U8, TensorFlags::OUTPUT),
            ];
            for (id, (dtype, flags)) in expected_abi.into_iter().enumerate() {
                let tensor = graph.tensor(TensorId(id as u32)).unwrap();
                assert_eq!(tensor.dtype, dtype);
                assert_eq!(tensor.flags, flags);
                let mut dimensions = tensor.dimensions();
                assert_eq!(dimensions.next(), Some(64));
                assert_eq!(dimensions.next(), None);
            }

            let mut present = [false; 35];
            for index in 0..graph.node_count() {
                let node = graph.node(index).unwrap();
                assert!(node.attributes().is_empty());
                assert!(matches!(
                    node.op,
                    OpCode::COPY
                        | OpCode::ADD
                        | OpCode::SUB
                        | OpCode::MUL
                        | OpCode::ABS
                        | OpCode::LOG
                        | OpCode::MAXIMUM
                        | OpCode::NEGATE
                        | OpCode::RECIPROCAL
                        | OpCode::SELECT
                        | OpCode::GREATER
                        | OpCode::GREATER_EQUAL
                        | OpCode::LOGICAL_AND
                        | OpCode::LOGICAL_OR
                        | OpCode::LOGICAL_NOT
                ));
                present[node.op.0 as usize] = true;
            }
            for required in [
                OpCode::ABS,
                OpCode::LOG,
                OpCode::MAXIMUM,
                OpCode::NEGATE,
                OpCode::RECIPROCAL,
                OpCode::SELECT,
                OpCode::GREATER,
                OpCode::GREATER_EQUAL,
                OpCode::LOGICAL_AND,
                OpCode::LOGICAL_OR,
                OpCode::LOGICAL_NOT,
            ] {
                assert!(present[required.0 as usize]);
            }

            for (offset, output) in [TensorId(5), TensorId(6), TensorId(7)]
                .into_iter()
                .enumerate()
            {
                let node = graph.node(graph.node_count() - 4 + offset).unwrap();
                assert_eq!(node.op, OpCode::COPY);
                assert_eq!(node.outputs().next(), Some(output));
            }
            let active_copy = graph.node(graph.node_count() - 1).unwrap();
            assert_eq!(active_copy.op, OpCode::LOGICAL_OR);
            assert_eq!(active_copy.outputs().next(), Some(TensorId(8)));
            let active_inputs: Vec<_> = active_copy.inputs().collect();
            assert_eq!(active_inputs.len(), 2);
            assert_eq!(active_inputs[0], active_inputs[1]);
        }
        assert!(counts.total.load(Ordering::Relaxed) > 0);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn invalid_specializations_fail_before_allocating() {
        let counts = Arc::new(AllocationCounts::default());
        let allocator = TrackingAllocator(Arc::clone(&counts));

        let mut spec = reference_spec(1);
        spec.rays = 0;
        assert!(matches!(
            build_wormhole_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidRayCount(0))
        ));

        spec = reference_spec(1);
        spec.rays = MAX_RAYS + 1;
        assert!(matches!(
            build_wormhole_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidRayCount(rays)) if rays == MAX_RAYS + 1
        ));

        spec = reference_spec(0);
        assert!(matches!(
            build_wormhole_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidStepCount(0))
        ));

        spec = reference_spec(1);
        spec.params.rho = f32::NAN;
        assert!(matches!(
            build_wormhole_graph_in(spec, allocator.clone()),
            Err(GraphError::InvalidParameters)
        ));

        spec = reference_spec(1);
        spec.trace.escape_ell = spec.params.a;
        assert!(matches!(
            build_wormhole_graph_in(spec, allocator),
            Err(GraphError::InvalidTraceConfig)
        ));
        assert_eq!(counts.total.load(Ordering::Relaxed), 0);
    }
}
