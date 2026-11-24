//! Dimension-generic field sampling and differential operators.
//!
//! Kernels operate on fixed arrays and caller-supplied samplers. They perform
//! no allocation and carry no storage/layout policy, so the same equations can
//! serve scalar oracles, SIMD tile kernels, and `.ax` graph frontends.

use core::convert::Infallible;

use crate::algebra::{self, F32Ops};

#[inline(always)]
fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// A scalar field sampled in `D` dimensions.
pub trait ScalarField<const D: usize> {
    fn sample(&mut self, point: [f32; D]) -> f32;
}

impl<const D: usize, F> ScalarField<D> for F
where
    F: FnMut([f32; D]) -> f32,
{
    #[inline(always)]
    fn sample(&mut self, point: [f32; D]) -> f32 {
        self(point)
    }
}

/// A vector field with `I` input dimensions and `O` output components.
pub trait VectorField<const I: usize, const O: usize> {
    fn sample(&mut self, point: [f32; I]) -> [f32; O];
}

impl<const I: usize, const O: usize, F> VectorField<I, O> for F
where
    F: FnMut([f32; I]) -> [f32; O],
{
    #[inline(always)]
    fn sample(&mut self, point: [f32; I]) -> [f32; O] {
        self(point)
    }
}

/// Whether every per-axis finite-difference step is finite and nonzero.
#[inline]
pub fn valid_steps<const D: usize>(steps: [f32; D]) -> bool {
    steps
        .into_iter()
        .all(|step| step.is_finite() && step != 0.0)
}

/// Dot product of two fixed-size vectors.
#[inline(always)]
pub fn dot<const D: usize>(left: [f32; D], right: [f32; D]) -> f32 {
    infallible(algebra::dot(&mut F32Ops, left, right))
}

/// Squared Euclidean norm.
#[inline(always)]
pub fn norm_squared<const D: usize>(value: [f32; D]) -> f32 {
    infallible(algebra::norm_squared(&mut F32Ops, value))
}

/// Euclidean norm.
#[inline(always)]
pub fn norm<const D: usize>(value: [f32; D]) -> f32 {
    infallible(algebra::norm(&mut F32Ops, value))
}

/// Normalize a vector, returning `None` for zero or non-finite magnitude.
#[inline]
pub fn normalized<const D: usize>(value: [f32; D]) -> Option<[f32; D]> {
    let magnitude = norm(value);
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return None;
    }
    Some(infallible(algebra::normalize(&mut F32Ops, value)))
}

/// Three-dimensional cross product.
#[inline(always)]
pub fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    infallible(algebra::cross3(&mut F32Ops, left, right))
}

/// Central finite-difference gradient of a scalar field.
///
/// `steps[axis]` is the signed sample displacement for that axis. Each step
/// must be finite and nonzero. The sampler is evaluated exactly `2 * D` times.
#[inline]
pub fn central_gradient<const D: usize>(
    field: &mut impl ScalarField<D>,
    point: [f32; D],
    steps: [f32; D],
) -> [f32; D] {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_gradient(
        &mut ops,
        &mut sample,
        point,
        steps,
    ))
}

/// Central finite-difference Jacobian in output-major orientation.
///
/// The returned value is indexed as `jacobian[output][input]`. The sampler is
/// evaluated exactly `2 * I` times, regardless of the output width.
#[inline]
pub fn central_jacobian<const I: usize, const O: usize>(
    field: &mut impl VectorField<I, O>,
    point: [f32; I],
    steps: [f32; I],
) -> [[f32; I]; O] {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_jacobian(
        &mut ops,
        &mut sample,
        point,
        steps,
    ))
}

/// Divergence of a square vector field.
#[inline]
pub fn central_divergence<const D: usize>(
    field: &mut impl VectorField<D, D>,
    point: [f32; D],
    steps: [f32; D],
) -> f32 {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_divergence(
        &mut ops,
        &mut sample,
        point,
        steps,
    ))
}

/// Curl of a three-dimensional vector field.
#[inline]
pub fn central_curl3(
    field: &mut impl VectorField<3, 3>,
    point: [f32; 3],
    steps: [f32; 3],
) -> [f32; 3] {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_curl3(&mut ops, &mut sample, point, steps))
}

/// Central finite-difference Laplacian of a scalar field.
///
/// The center is sampled once and each axis contributes two neighboring
/// samples, for exactly `1 + 2 * D` evaluations.
#[inline]
pub fn central_laplacian<const D: usize>(
    field: &mut impl ScalarField<D>,
    point: [f32; D],
    steps: [f32; D],
) -> f32 {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_laplacian(
        &mut ops,
        &mut sample,
        point,
        steps,
    ))
}

/// Central finite-difference Hessian of a scalar field.
///
/// The matrix is symmetric and indexed `[row][column]`. Diagonal entries use
/// the standard three-point stencil; mixed entries use the four-corner
/// stencil. The sampler is evaluated exactly `1 + 2 * D * D` times.
#[inline]
pub fn central_hessian<const D: usize>(
    field: &mut impl ScalarField<D>,
    point: [f32; D],
    steps: [f32; D],
) -> [[f32; D]; D] {
    let mut ops = F32Ops;
    let mut sample = |_: &mut F32Ops, sample_point| Ok(field.sample(sample_point));
    infallible(algebra::central_hessian(
        &mut ops,
        &mut sample,
        point,
        steps,
    ))
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
    fn vector_algebra_is_dimension_generic() {
        assert_eq!(dot([1.0, 2.0, 3.0], [4.0, -2.0, 0.5]), 1.5);
        close(norm([3.0, 4.0]), 5.0, 1.0e-6);
        assert_eq!(normalized([0.0, 0.0]), None);
        assert_eq!(normalized([f32::INFINITY, 0.0]), None);
        assert_eq!(normalized([f32::NAN, 0.0]), None);
        let unit = normalized([2.0, -1.0, 2.0]).unwrap();
        close(norm(unit), 1.0, 2.0e-6);
        assert_eq!(cross3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn scalar_stencils_match_quadratic_gradient_hessian_and_laplacian() {
        // 3x² + 2xy - yz + 4z² + 5x - 7z
        let mut field = |p: [f32; 3]| {
            3.0 * p[0] * p[0] + 2.0 * p[0] * p[1] - p[1] * p[2] + 4.0 * p[2] * p[2] + 5.0 * p[0]
                - 7.0 * p[2]
        };
        let point = [1.25, -0.75, 0.5];
        let steps = [0.02, 0.03125, 0.015625];
        let gradient = central_gradient(&mut field, point, steps);
        let expected_gradient = [
            6.0 * point[0] + 2.0 * point[1] + 5.0,
            2.0 * point[0] - point[2],
            -point[1] + 8.0 * point[2] - 7.0,
        ];
        for (actual, expected) in gradient.into_iter().zip(expected_gradient) {
            close(actual, expected, 2.0e-4);
        }

        let hessian = central_hessian(&mut field, point, steps);
        let expected = [[6.0, 2.0, 0.0], [2.0, 0.0, -1.0], [0.0, -1.0, 8.0]];
        for row in 0..3 {
            for column in 0..3 {
                close(hessian[row][column], expected[row][column], 2.0e-3);
            }
        }
        close(central_laplacian(&mut field, point, steps), 14.0, 2.0e-3);
    }

    #[test]
    fn vector_stencils_share_axis_samples() {
        let evaluations = Cell::new(0);
        let mut field = |p: [f32; 3]| {
            evaluations.set(evaluations.get() + 1);
            [p[0] * p[1], p[1] * p[2], p[2] * p[0]]
        };
        let point = [2.0, 3.0, 5.0];
        let steps = [0.25, 0.125, 0.5];
        let jacobian = central_jacobian(&mut field, point, steps);
        assert_eq!(evaluations.get(), 6);
        let expected = [[3.0, 2.0, 0.0], [0.0, 5.0, 3.0], [5.0, 0.0, 2.0]];
        assert_eq!(jacobian, expected);

        let divergence = central_divergence(&mut field, point, steps);
        close(divergence, 10.0, 1.0e-6);
        let curl = central_curl3(&mut field, point, steps);
        assert_eq!(curl, [-3.0, -5.0, -2.0]);
    }

    #[test]
    fn scalar_evaluation_counts_are_locked() {
        let point = [0.25, -0.5, 0.75, 1.0];
        let steps = [0.1; 4];
        let count = Cell::new(0);
        let mut field = |p: [f32; 4]| {
            count.set(count.get() + 1);
            p.into_iter().sum()
        };
        let _ = central_gradient(&mut field, point, steps);
        assert_eq!(count.get(), 8);
        count.set(0);
        let _ = central_laplacian(&mut field, point, steps);
        assert_eq!(count.get(), 9);
        count.set(0);
        let _ = central_hessian(&mut field, point, steps);
        assert_eq!(count.get(), 33);
    }

    #[test]
    fn step_validation_rejects_degenerate_axes() {
        assert!(valid_steps([0.1, -0.25, 1.0]));
        assert!(!valid_steps([0.1, 0.0, 1.0]));
        assert!(!valid_steps([0.1, f32::NAN, 1.0]));
    }
}
