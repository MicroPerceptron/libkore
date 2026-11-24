//! Allocation-free integration stencils over fixed-size state arrays.
//!
//! These routines own numerical schemes, not clocks, convergence policy, or
//! storage. Callers decide how many steps to execute and where state lives.

use core::convert::Infallible;

use crate::algebra::{self, F32Ops};

#[inline(always)]
fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// Advance an autonomous first-order system with explicit midpoint RK2.
///
/// The derivative is evaluated exactly twice. A negative `step` follows the
/// same algebra in reverse time; unlike velocity-Verlet, explicit midpoint is
/// second-order accurate but not exactly self-adjoint.
#[inline]
pub fn explicit_midpoint<const N: usize>(
    state: &mut [f32; N],
    step: f32,
    derivative: &mut impl FnMut(&[f32; N]) -> [f32; N],
) {
    let mut ops = F32Ops;
    let mut rhs = |_: &mut F32Ops, rhs_state| Ok(derivative(&rhs_state));
    *state = infallible(algebra::explicit_midpoint(&mut ops, *state, step, &mut rhs));
}

/// Apply multiple explicit-midpoint steps without allocating trajectory data.
#[inline]
pub fn explicit_midpoint_steps<const N: usize>(
    state: &mut [f32; N],
    step: f32,
    steps: u32,
    derivative: &mut impl FnMut(&[f32; N]) -> [f32; N],
) {
    let mut ops = F32Ops;
    let mut rhs = |_: &mut F32Ops, rhs_state| Ok(derivative(&rhs_state));
    *state = infallible(algebra::explicit_midpoint_steps(
        &mut ops, *state, step, steps, &mut rhs,
    ));
}

/// Symmetric kick-drift-kick velocity-Verlet update.
///
/// Acceleration may depend on position but not velocity. It is evaluated
/// exactly twice. Supplying a negative step applies the inverse map up to
/// floating-point rounding.
#[inline]
pub fn velocity_verlet<const N: usize>(
    position: &mut [f32; N],
    velocity: &mut [f32; N],
    step: f32,
    acceleration: &mut impl FnMut(&[f32; N]) -> [f32; N],
) {
    let mut ops = F32Ops;
    let mut force = |_: &mut F32Ops, force_position| Ok(acceleration(&force_position));
    let (new_position, new_velocity) = infallible(algebra::velocity_verlet(
        &mut ops, *position, *velocity, step, &mut force,
    ));
    *position = new_position;
    *velocity = new_velocity;
}

/// Apply multiple velocity-Verlet steps without allocating trajectory data.
#[inline]
pub fn velocity_verlet_steps<const N: usize>(
    position: &mut [f32; N],
    velocity: &mut [f32; N],
    step: f32,
    steps: u32,
    acceleration: &mut impl FnMut(&[f32; N]) -> [f32; N],
) {
    let mut ops = F32Ops;
    let mut force = |_: &mut F32Ops, force_position| Ok(acceleration(&force_position));
    let (new_position, new_velocity) = infallible(algebra::velocity_verlet_steps(
        &mut ops, *position, *velocity, step, steps, &mut force,
    ));
    *position = new_position;
    *velocity = new_velocity;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axmath::{abs_f32, cos_f32, sin_f32};

    fn oscillator(state: &[f32; 2]) -> [f32; 2] {
        [state[1], -state[0]]
    }

    fn endpoint_error(step: f32, steps: u32) -> f32 {
        let mut state = [1.0, 0.0];
        explicit_midpoint_steps(&mut state, step, steps, &mut oscillator);
        abs_f32(state[0] - cos_f32(1.0)) + abs_f32(state[1] + sin_f32(1.0))
    }

    #[test]
    fn midpoint_has_second_order_global_convergence() {
        let coarse = endpoint_error(0.1, 10);
        let fine = endpoint_error(0.05, 20);
        assert!(fine < coarse / 3.5, "coarse {coarse}, fine {fine}");
        assert!(fine > coarse / 4.6, "coarse {coarse}, fine {fine}");
    }

    #[test]
    fn midpoint_uses_two_derivative_evaluations_per_step() {
        let mut evaluations = 0;
        let mut derivative = |state: &[f32; 3]| {
            evaluations += 1;
            [state[1], state[2], -state[0]]
        };
        let mut state = [1.0, 2.0, 3.0];
        explicit_midpoint_steps(&mut state, 0.01, 7, &mut derivative);
        assert_eq!(evaluations, 14);
    }

    #[test]
    fn verlet_is_time_reversible_for_position_dependent_force() {
        let start_position = [1.0, -0.25];
        let start_velocity = [0.1, 0.4];
        let mut position = start_position;
        let mut velocity = start_velocity;
        let mut acceleration = |position: &[f32; 2]| [-position[0], -4.0 * position[1]];
        velocity_verlet_steps(&mut position, &mut velocity, 0.01, 256, &mut acceleration);
        velocity_verlet_steps(&mut position, &mut velocity, -0.01, 256, &mut acceleration);
        for component in 0..2 {
            assert!(abs_f32(position[component] - start_position[component]) < 2.0e-5);
            assert!(abs_f32(velocity[component] - start_velocity[component]) < 2.0e-5);
        }
    }

    #[test]
    fn verlet_uses_two_acceleration_evaluations_per_step() {
        let mut evaluations = 0;
        let mut acceleration = |position: &[f32; 1]| {
            evaluations += 1;
            [-position[0]]
        };
        let mut position = [1.0];
        let mut velocity = [0.0];
        velocity_verlet_steps(&mut position, &mut velocity, 0.1, 9, &mut acceleration);
        assert_eq!(evaluations, 18);
    }

    #[test]
    fn zero_steps_leave_state_untouched_without_evaluating_callbacks() {
        let mut midpoint_evaluations = 0;
        let mut derivative = |state: &[f32; 2]| {
            midpoint_evaluations += 1;
            [state[1], -state[0]]
        };
        let mut state = [1.25, -0.75];
        explicit_midpoint_steps(&mut state, 0.1, 0, &mut derivative);
        assert_eq!(state, [1.25, -0.75]);
        assert_eq!(midpoint_evaluations, 0);

        let mut verlet_evaluations = 0;
        let mut acceleration = |position: &[f32; 2]| {
            verlet_evaluations += 1;
            [-position[0], -position[1]]
        };
        let mut position = [2.0, -3.0];
        let mut velocity = [0.5, 0.25];
        velocity_verlet_steps(&mut position, &mut velocity, 0.1, 0, &mut acceleration);
        assert_eq!(position, [2.0, -3.0]);
        assert_eq!(velocity, [0.5, 0.25]);
        assert_eq!(verlet_evaluations, 0);
    }
}
