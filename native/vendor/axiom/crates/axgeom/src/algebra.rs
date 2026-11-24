//! Arithmetic-generic field algebra and integration stencils.
//!
//! The algorithms in this module describe arithmetic once.  A scalar
//! implementation can use them as a numerical oracle, while a graph emitter
//! can implement [`RealOps`] to turn the same source into a tensor graph.
//! Fixed-size arrays keep the kernels allocation-free and make their sample
//! counts independent of the arithmetic backend.

use core::convert::Infallible;

use axmath::sqrt_f32;

/// The small real-arithmetic surface needed by geometric field kernels.
///
/// Methods are fallible because graph implementations may need to allocate a
/// tensor or node for every operation.  `Value: Copy` deliberately fits both
/// machine scalars and cheap graph-local tensor handles.
pub trait RealOps {
    type Value: Copy;
    type Error;

    fn constant(&mut self, value: f32) -> Result<Self::Value, Self::Error>;
    fn add(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn sub(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn mul(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn negate(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn reciprocal(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn rsqrt(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn minimum(
        &mut self,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn maximum(
        &mut self,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

/// Position/velocity pair returned by second-order phase-space integrators.
pub type PhaseState<V, const N: usize> = ([V; N], [V; N]);

/// Infallible `f32` arithmetic for scalar references and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct F32Ops;

impl RealOps for F32Ops {
    type Value = f32;
    type Error = Infallible;

    #[inline(always)]
    fn constant(&mut self, value: f32) -> Result<f32, Infallible> {
        Ok(value)
    }

    #[inline(always)]
    fn add(&mut self, left: f32, right: f32) -> Result<f32, Infallible> {
        Ok(left + right)
    }

    #[inline(always)]
    fn sub(&mut self, left: f32, right: f32) -> Result<f32, Infallible> {
        Ok(left - right)
    }

    #[inline(always)]
    fn mul(&mut self, left: f32, right: f32) -> Result<f32, Infallible> {
        Ok(left * right)
    }

    #[inline(always)]
    fn negate(&mut self, value: f32) -> Result<f32, Infallible> {
        Ok(-value)
    }

    #[inline(always)]
    fn reciprocal(&mut self, value: f32) -> Result<f32, Infallible> {
        Ok(value.recip())
    }

    #[inline(always)]
    fn rsqrt(&mut self, value: f32) -> Result<f32, Infallible> {
        Ok(sqrt_f32(value).recip())
    }

    #[inline(always)]
    fn minimum(&mut self, left: f32, right: f32) -> Result<f32, Infallible> {
        Ok(propagating_minimum(left, right))
    }

    #[inline(always)]
    fn maximum(&mut self, left: f32, right: f32) -> Result<f32, Infallible> {
        Ok(propagating_maximum(left, right))
    }
}

#[inline(always)]
fn propagating_minimum(left: f32, right: f32) -> f32 {
    if left.is_nan() {
        left
    } else if right.is_nan() {
        right
    } else {
        left.min(right)
    }
}

#[inline(always)]
fn propagating_maximum(left: f32, right: f32) -> f32 {
    if left.is_nan() {
        left
    } else if right.is_nan() {
        right
    } else {
        left.max(right)
    }
}

/// Dot product of two fixed-size vectors.
#[inline]
pub fn dot<R: RealOps, const N: usize>(
    ops: &mut R,
    left: [R::Value; N],
    right: [R::Value; N],
) -> Result<R::Value, R::Error> {
    let mut result = ops.constant(0.0)?;
    for component in 0..N {
        let product = ops.mul(left[component], right[component])?;
        result = ops.add(result, product)?;
    }
    Ok(result)
}

/// Squared Euclidean norm of a fixed-size vector.
#[inline]
pub fn norm_squared<R: RealOps, const N: usize>(
    ops: &mut R,
    value: [R::Value; N],
) -> Result<R::Value, R::Error> {
    dot(ops, value, value)
}

/// Euclidean norm of a fixed-size vector.
///
/// Expressing `sqrt(x)` as `1 / rsqrt(x)` keeps the primitive arithmetic
/// surface compact and still returns zero for a zero-length scalar vector.
#[inline]
pub fn norm<R: RealOps, const N: usize>(
    ops: &mut R,
    value: [R::Value; N],
) -> Result<R::Value, R::Error> {
    let squared = norm_squared(ops, value)?;
    let inverse = ops.rsqrt(squared)?;
    ops.reciprocal(inverse)
}

/// Normalize a vector without branching on a zero or non-finite norm.
///
/// Scalar facades that promise checked normalization should validate the norm
/// first.  Symbolic graph clients can instead clamp the squared norm or select
/// a fallback according to their own precision policy.
#[inline]
pub fn normalize<R: RealOps, const N: usize>(
    ops: &mut R,
    value: [R::Value; N],
) -> Result<[R::Value; N], R::Error> {
    let squared = norm_squared(ops, value)?;
    let inverse = ops.rsqrt(squared)?;
    let mut result = value;
    for component in 0..N {
        result[component] = ops.mul(value[component], inverse)?;
    }
    Ok(result)
}

/// Three-dimensional cross product.
#[inline]
pub fn cross3<R: RealOps>(
    ops: &mut R,
    left: [R::Value; 3],
    right: [R::Value; 3],
) -> Result<[R::Value; 3], R::Error> {
    let left_y_right_z = ops.mul(left[1], right[2])?;
    let left_z_right_y = ops.mul(left[2], right[1])?;
    let x = ops.sub(left_y_right_z, left_z_right_y)?;
    let left_z_right_x = ops.mul(left[2], right[0])?;
    let left_x_right_z = ops.mul(left[0], right[2])?;
    let y = ops.sub(left_z_right_x, left_x_right_z)?;
    let left_x_right_y = ops.mul(left[0], right[1])?;
    let left_y_right_x = ops.mul(left[1], right[0])?;
    let z = ops.sub(left_x_right_y, left_y_right_x)?;
    Ok([x, y, z])
}

/// Outer product, indexed `[left][right]`.
#[inline]
pub fn outer_product<R: RealOps, const M: usize, const N: usize>(
    ops: &mut R,
    left: [R::Value; M],
    right: [R::Value; N],
) -> Result<[[R::Value; N]; M], R::Error> {
    let zero = ops.constant(0.0)?;
    let mut result = [[zero; N]; M];
    for row in 0..M {
        for column in 0..N {
            result[row][column] = ops.mul(left[row], right[column])?;
        }
    }
    Ok(result)
}

/// Row-major matrix-vector product.
#[inline]
pub fn matrix_vector<R: RealOps, const M: usize, const N: usize>(
    ops: &mut R,
    matrix: [[R::Value; N]; M],
    vector: [R::Value; N],
) -> Result<[R::Value; M], R::Error> {
    let zero = ops.constant(0.0)?;
    let mut result = [zero; M];
    for row in 0..M {
        result[row] = dot(ops, matrix[row], vector)?;
    }
    Ok(result)
}

/// Row-major matrix product `(M x K) * (K x N)`.
#[inline]
pub fn matrix_multiply<R: RealOps, const M: usize, const K: usize, const N: usize>(
    ops: &mut R,
    left: [[R::Value; K]; M],
    right: [[R::Value; N]; K],
) -> Result<[[R::Value; N]; M], R::Error> {
    let zero = ops.constant(0.0)?;
    let mut result = [[zero; N]; M];
    for row in 0..M {
        for column in 0..N {
            let mut entry = ops.constant(0.0)?;
            for inner in 0..K {
                let product = ops.mul(left[row][inner], right[inner][column])?;
                entry = ops.add(entry, product)?;
            }
            result[row][column] = entry;
        }
    }
    Ok(result)
}

/// Central finite-difference gradient of a scalar field.
///
/// `field` is evaluated exactly `2 * D` times.  Each `steps[axis]` must
/// represent a nonzero displacement; validation is left to the caller so the
/// same kernel remains meaningful for symbolic graph values.
#[inline]
pub fn central_gradient<R, F, const D: usize>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; D],
    steps: [R::Value; D],
) -> Result<[R::Value; D], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; D]) -> Result<R::Value, R::Error>,
{
    let two = ops.constant(2.0)?;
    let mut gradient = point;
    for axis in 0..D {
        let mut plus = point;
        let mut minus = point;
        plus[axis] = ops.add(point[axis], steps[axis])?;
        minus[axis] = ops.sub(point[axis], steps[axis])?;
        let plus_value = field(ops, plus)?;
        let minus_value = field(ops, minus)?;
        let numerator = ops.sub(plus_value, minus_value)?;
        let denominator = ops.mul(two, steps[axis])?;
        let scale = ops.reciprocal(denominator)?;
        gradient[axis] = ops.mul(numerator, scale)?;
    }
    Ok(gradient)
}

/// Central finite-difference Jacobian in output-major orientation.
///
/// The return value is indexed `[output][input]`.  Neighbor samples are
/// shared across all output components, so `field` is evaluated exactly
/// `2 * I` times rather than `2 * I * O` times.
#[inline]
pub fn central_jacobian<R, F, const I: usize, const O: usize>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; I],
    steps: [R::Value; I],
) -> Result<[[R::Value; I]; O], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; I]) -> Result<[R::Value; O], R::Error>,
{
    let zero = ops.constant(0.0)?;
    let two = ops.constant(2.0)?;
    let mut jacobian = [[zero; I]; O];
    for input in 0..I {
        let mut plus = point;
        let mut minus = point;
        plus[input] = ops.add(point[input], steps[input])?;
        minus[input] = ops.sub(point[input], steps[input])?;
        let plus_value = field(ops, plus)?;
        let minus_value = field(ops, minus)?;
        let denominator = ops.mul(two, steps[input])?;
        let scale = ops.reciprocal(denominator)?;
        for output in 0..O {
            let difference = ops.sub(plus_value[output], minus_value[output])?;
            jacobian[output][input] = ops.mul(difference, scale)?;
        }
    }
    Ok(jacobian)
}

/// Central finite-difference divergence of a square vector field.
///
/// The sampler is evaluated exactly `2 * D` times.
#[inline]
pub fn central_divergence<R, F, const D: usize>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; D],
    steps: [R::Value; D],
) -> Result<R::Value, R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; D]) -> Result<[R::Value; D], R::Error>,
{
    let jacobian = central_jacobian(ops, field, point, steps)?;
    let mut divergence = ops.constant(0.0)?;
    for (axis, row) in jacobian.into_iter().enumerate() {
        divergence = ops.add(divergence, row[axis])?;
    }
    Ok(divergence)
}

/// Central finite-difference curl of a three-dimensional vector field.
///
/// The sampler is evaluated exactly six times.
#[inline]
pub fn central_curl3<R, F>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; 3],
    steps: [R::Value; 3],
) -> Result<[R::Value; 3], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; 3]) -> Result<[R::Value; 3], R::Error>,
{
    let jacobian = central_jacobian(ops, field, point, steps)?;
    Ok([
        ops.sub(jacobian[2][1], jacobian[1][2])?,
        ops.sub(jacobian[0][2], jacobian[2][0])?,
        ops.sub(jacobian[1][0], jacobian[0][1])?,
    ])
}

/// Central finite-difference Laplacian of a scalar field.
///
/// The center is shared by all axes, for exactly `1 + 2 * D` evaluations.
#[inline]
pub fn central_laplacian<R, F, const D: usize>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; D],
    steps: [R::Value; D],
) -> Result<R::Value, R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; D]) -> Result<R::Value, R::Error>,
{
    let center = field(ops, point)?;
    let two = ops.constant(2.0)?;
    let twice_center = ops.mul(two, center)?;
    let mut laplacian = ops.constant(0.0)?;
    for axis in 0..D {
        let mut plus = point;
        let mut minus = point;
        plus[axis] = ops.add(point[axis], steps[axis])?;
        minus[axis] = ops.sub(point[axis], steps[axis])?;
        let plus_value = field(ops, plus)?;
        let minus_value = field(ops, minus)?;
        let neighbors = ops.add(plus_value, minus_value)?;
        let numerator = ops.sub(neighbors, twice_center)?;
        let step_squared = ops.mul(steps[axis], steps[axis])?;
        let scale = ops.reciprocal(step_squared)?;
        let contribution = ops.mul(numerator, scale)?;
        laplacian = ops.add(laplacian, contribution)?;
    }
    Ok(laplacian)
}

/// Central finite-difference Hessian of a scalar field.
///
/// The symmetric result is indexed `[row][column]`.  Diagonal entries use a
/// three-point stencil and mixed entries use the four-corner stencil.  The
/// exact sampler count is `1 + 2 * D * D`.
#[inline]
pub fn central_hessian<R, F, const D: usize>(
    ops: &mut R,
    field: &mut F,
    point: [R::Value; D],
    steps: [R::Value; D],
) -> Result<[[R::Value; D]; D], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; D]) -> Result<R::Value, R::Error>,
{
    let center = field(ops, point)?;
    let zero = ops.constant(0.0)?;
    let two = ops.constant(2.0)?;
    let four = ops.constant(4.0)?;
    let twice_center = ops.mul(two, center)?;
    let mut hessian = [[zero; D]; D];

    for row in 0..D {
        let mut plus = point;
        let mut minus = point;
        plus[row] = ops.add(point[row], steps[row])?;
        minus[row] = ops.sub(point[row], steps[row])?;
        let plus_value = field(ops, plus)?;
        let minus_value = field(ops, minus)?;
        let neighbors = ops.add(plus_value, minus_value)?;
        let numerator = ops.sub(neighbors, twice_center)?;
        let step_squared = ops.mul(steps[row], steps[row])?;
        let scale = ops.reciprocal(step_squared)?;
        hessian[row][row] = ops.mul(numerator, scale)?;

        for column in row + 1..D {
            let mut plus_plus = point;
            let mut plus_minus = point;
            let mut minus_plus = point;
            let mut minus_minus = point;
            plus_plus[row] = ops.add(point[row], steps[row])?;
            plus_plus[column] = ops.add(point[column], steps[column])?;
            plus_minus[row] = ops.add(point[row], steps[row])?;
            plus_minus[column] = ops.sub(point[column], steps[column])?;
            minus_plus[row] = ops.sub(point[row], steps[row])?;
            minus_plus[column] = ops.add(point[column], steps[column])?;
            minus_minus[row] = ops.sub(point[row], steps[row])?;
            minus_minus[column] = ops.sub(point[column], steps[column])?;

            let pp = field(ops, plus_plus)?;
            let pm = field(ops, plus_minus)?;
            let mp = field(ops, minus_plus)?;
            let mm = field(ops, minus_minus)?;
            let upper_difference = ops.sub(pp, pm)?;
            let lower_difference = ops.sub(mm, mp)?;
            let numerator = ops.add(upper_difference, lower_difference)?;
            let step_product = ops.mul(steps[row], steps[column])?;
            let denominator = ops.mul(four, step_product)?;
            let scale = ops.reciprocal(denominator)?;
            let mixed = ops.mul(numerator, scale)?;
            hessian[row][column] = mixed;
            hessian[column][row] = mixed;
        }
    }
    Ok(hessian)
}

/// Advance an autonomous first-order system with explicit midpoint RK2.
///
/// `rhs` is evaluated exactly twice.  Returning the new state instead of
/// mutating in place also makes the routine natural for immutable graph
/// handles.
#[inline]
pub fn explicit_midpoint<R, F, const N: usize>(
    ops: &mut R,
    state: [R::Value; N],
    step: R::Value,
    rhs: &mut F,
) -> Result<[R::Value; N], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; N]) -> Result<[R::Value; N], R::Error>,
{
    let first = rhs(ops, state)?;
    let half = ops.constant(0.5)?;
    let half_step = ops.mul(half, step)?;
    let mut midpoint = state;
    for component in 0..N {
        let delta = ops.mul(half_step, first[component])?;
        midpoint[component] = ops.add(state[component], delta)?;
    }
    let second = rhs(ops, midpoint)?;
    let mut result = state;
    for component in 0..N {
        let delta = ops.mul(step, second[component])?;
        result[component] = ops.add(state[component], delta)?;
    }
    Ok(result)
}

/// Apply a fixed number of explicit-midpoint steps without trajectory
/// allocation.  `rhs` is evaluated exactly `2 * steps` times.
#[inline]
pub fn explicit_midpoint_steps<R, F, const N: usize>(
    ops: &mut R,
    mut state: [R::Value; N],
    step: R::Value,
    steps: u32,
    rhs: &mut F,
) -> Result<[R::Value; N], R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; N]) -> Result<[R::Value; N], R::Error>,
{
    for _ in 0..steps {
        state = explicit_midpoint(ops, state, step, rhs)?;
    }
    Ok(state)
}

/// Symmetric kick-drift-kick velocity-Verlet update.
///
/// Acceleration may depend on position but not velocity.  It is evaluated
/// exactly twice, and a negative step applies the inverse map up to arithmetic
/// rounding.
#[inline]
pub fn velocity_verlet<R, F, const N: usize>(
    ops: &mut R,
    position: [R::Value; N],
    velocity: [R::Value; N],
    step: R::Value,
    acceleration: &mut F,
) -> Result<PhaseState<R::Value, N>, R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; N]) -> Result<[R::Value; N], R::Error>,
{
    let old_acceleration = acceleration(ops, position)?;
    let half = ops.constant(0.5)?;
    let half_step = ops.mul(half, step)?;
    let mut half_velocity = velocity;
    for component in 0..N {
        let kick = ops.mul(half_step, old_acceleration[component])?;
        half_velocity[component] = ops.add(velocity[component], kick)?;
    }
    let mut new_position = position;
    for component in 0..N {
        let drift = ops.mul(step, half_velocity[component])?;
        new_position[component] = ops.add(position[component], drift)?;
    }
    let new_acceleration = acceleration(ops, new_position)?;
    let mut new_velocity = half_velocity;
    for component in 0..N {
        let kick = ops.mul(half_step, new_acceleration[component])?;
        new_velocity[component] = ops.add(half_velocity[component], kick)?;
    }
    Ok((new_position, new_velocity))
}

/// Apply a fixed number of velocity-Verlet steps without trajectory
/// allocation.  `acceleration` is evaluated exactly `2 * steps` times.
#[inline]
pub fn velocity_verlet_steps<R, F, const N: usize>(
    ops: &mut R,
    mut position: [R::Value; N],
    mut velocity: [R::Value; N],
    step: R::Value,
    steps: u32,
    acceleration: &mut F,
) -> Result<PhaseState<R::Value, N>, R::Error>
where
    R: RealOps,
    F: FnMut(&mut R, [R::Value; N]) -> Result<[R::Value; N], R::Error>,
{
    for _ in 0..steps {
        (position, velocity) = velocity_verlet(ops, position, velocity, step, acceleration)?;
    }
    Ok((position, velocity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} vs {expected}"
        );
    }

    #[test]
    fn scalar_stencils_match_quadratic_field() {
        let mut ops = F32Ops;
        // 3x^2 + 2xy - yz + 4z^2 + 5x - 7z.
        let mut field =
            |_: &mut F32Ops, p: [f32; 3]| {
                Ok(3.0 * p[0] * p[0] + 2.0 * p[0] * p[1] - p[1] * p[2]
                    + 4.0 * p[2] * p[2]
                    + 5.0 * p[0]
                    - 7.0 * p[2])
            };
        let point = [1.25, -0.75, 0.5];
        let steps = [0.02, 0.03125, 0.015625];
        let gradient = central_gradient(&mut ops, &mut field, point, steps).unwrap();
        let expected_gradient = [
            6.0 * point[0] + 2.0 * point[1] + 5.0,
            2.0 * point[0] - point[2],
            -point[1] + 8.0 * point[2] - 7.0,
        ];
        for (actual, expected) in gradient.into_iter().zip(expected_gradient) {
            close(actual, expected, 3.0e-4);
        }

        let hessian = central_hessian(&mut ops, &mut field, point, steps).unwrap();
        let expected = [[6.0, 2.0, 0.0], [2.0, 0.0, -1.0], [0.0, -1.0, 8.0]];
        for row in 0..3 {
            for column in 0..3 {
                close(hessian[row][column], expected[row][column], 3.0e-3);
            }
        }
        close(
            central_laplacian(&mut ops, &mut field, point, steps).unwrap(),
            14.0,
            3.0e-3,
        );
    }

    #[test]
    fn vector_and_matrix_algebra_is_dimension_generic() {
        let mut ops = F32Ops;
        assert_eq!(
            dot(&mut ops, [1.0, 2.0, 3.0], [4.0, -2.0, 0.5]).unwrap(),
            1.5
        );
        assert_eq!(norm_squared(&mut ops, [3.0, 4.0]).unwrap(), 25.0);
        close(norm(&mut ops, [3.0, 4.0]).unwrap(), 5.0, 1.0e-6);
        let unit = normalize(&mut ops, [2.0, -1.0, 2.0]).unwrap();
        close(norm(&mut ops, unit).unwrap(), 1.0, 2.0e-6);
        assert_eq!(
            cross3(&mut ops, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap(),
            [0.0, 0.0, 1.0]
        );
        assert_eq!(
            outer_product(&mut ops, [2.0, -1.0], [3.0, 4.0, 5.0]).unwrap(),
            [[6.0, 8.0, 10.0], [-3.0, -4.0, -5.0]]
        );
        assert_eq!(
            matrix_vector(&mut ops, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]).unwrap(),
            [17.0, 39.0]
        );
        assert_eq!(
            matrix_multiply(
                &mut ops,
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            )
            .unwrap(),
            [[58.0, 64.0], [139.0, 154.0]]
        );
    }

    #[test]
    fn vector_stencils_share_axis_samples() {
        let evaluations = Cell::new(0);
        let mut ops = F32Ops;
        let mut field = |_: &mut F32Ops, p: [f32; 3]| {
            evaluations.set(evaluations.get() + 1);
            Ok([p[0] * p[1], p[1] * p[2], p[2] * p[0]])
        };
        let point = [2.0, 3.0, 5.0];
        let steps = [0.25, 0.125, 0.5];
        let jacobian = central_jacobian(&mut ops, &mut field, point, steps).unwrap();
        assert_eq!(evaluations.get(), 6);
        assert_eq!(
            jacobian,
            [[3.0, 2.0, 0.0], [0.0, 5.0, 3.0], [5.0, 0.0, 2.0]]
        );

        evaluations.set(0);
        close(
            central_divergence(&mut ops, &mut field, point, steps).unwrap(),
            10.0,
            1.0e-6,
        );
        assert_eq!(evaluations.get(), 6);
        evaluations.set(0);
        assert_eq!(
            central_curl3(&mut ops, &mut field, point, steps).unwrap(),
            [-3.0, -5.0, -2.0]
        );
        assert_eq!(evaluations.get(), 6);
    }

    #[test]
    fn scalar_evaluation_counts_are_locked() {
        let evaluations = Cell::new(0);
        let mut ops = F32Ops;
        let mut field = |_: &mut F32Ops, p: [f32; 4]| {
            evaluations.set(evaluations.get() + 1);
            Ok(p.into_iter().sum())
        };
        let point = [0.25, -0.5, 0.75, 1.0];
        let steps = [0.1; 4];
        let _ = central_gradient(&mut ops, &mut field, point, steps).unwrap();
        assert_eq!(evaluations.get(), 8);
        evaluations.set(0);
        let _ = central_laplacian(&mut ops, &mut field, point, steps).unwrap();
        assert_eq!(evaluations.get(), 9);
        evaluations.set(0);
        let _ = central_hessian(&mut ops, &mut field, point, steps).unwrap();
        assert_eq!(evaluations.get(), 33);
    }

    #[test]
    fn integrator_evaluation_counts_and_reversibility_are_locked() {
        let midpoint_evaluations = Cell::new(0);
        let mut ops = F32Ops;
        let mut rhs = |_: &mut F32Ops, state: [f32; 2]| {
            midpoint_evaluations.set(midpoint_evaluations.get() + 1);
            Ok([state[1], -state[0]])
        };
        let _ = explicit_midpoint(&mut ops, [1.0, 0.0], 0.1, &mut rhs).unwrap();
        assert_eq!(midpoint_evaluations.get(), 2);
        let _ = explicit_midpoint_steps(&mut ops, [1.0, 0.0], 0.1, 3, &mut rhs).unwrap();
        assert_eq!(midpoint_evaluations.get(), 8);

        let verlet_evaluations = Cell::new(0);
        let mut acceleration = |_: &mut F32Ops, position: [f32; 2]| {
            verlet_evaluations.set(verlet_evaluations.get() + 1);
            Ok([-position[0], -4.0 * position[1]])
        };
        let position = [1.0, -0.25];
        let velocity = [0.1, 0.4];
        let (forward_position, forward_velocity) =
            velocity_verlet(&mut ops, position, velocity, 0.01, &mut acceleration).unwrap();
        let (back_position, back_velocity) = velocity_verlet(
            &mut ops,
            forward_position,
            forward_velocity,
            -0.01,
            &mut acceleration,
        )
        .unwrap();
        assert_eq!(verlet_evaluations.get(), 4);
        for component in 0..2 {
            close(back_position[component], position[component], 2.0e-6);
            close(back_velocity[component], velocity[component], 2.0e-6);
        }
        let _ = velocity_verlet_steps(&mut ops, position, velocity, 0.01, 3, &mut acceleration)
            .unwrap();
        assert_eq!(verlet_evaluations.get(), 10);
    }

    #[test]
    fn min_and_max_propagate_nan_like_tosa_geometry_ops() {
        let mut ops = F32Ops;
        assert!(ops.minimum(1.0, f32::NAN).unwrap().is_nan());
        assert!(ops.maximum(f32::NAN, 1.0).unwrap().is_nan());
        assert_eq!(ops.minimum(-2.0, 3.0).unwrap(), -2.0);
        assert_eq!(ops.maximum(-2.0, 3.0).unwrap(), 3.0);
    }
}
