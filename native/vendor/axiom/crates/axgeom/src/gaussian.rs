//! Allocation-free geometry for projecting three-dimensional Gaussians.
//!
//! This module owns the geometry upstream of a 3D Gaussian splat rasterizer:
//! world-to-camera transformation, perspective covariance projection,
//! covariance stabilization, conservative footprints, culling, and tile
//! bounds.  Spherical-harmonic shading and pixel compositing remain optimized
//! backend kernels; in particular, [`Symmetric2::packed`] emits the `[a,b,c]`
//! conic convention consumed by `axnn-cpu`'s splat compositor.
//!
//! The projection equations are expressed through [`RealOps`].  The same
//! source therefore works as an `f32` reference and as a tensor-graph emitter.
//! Decisions which change control flow (visibility, finite-value rejection,
//! and precision classification) intentionally remain scalar batch/tile
//! policy, rather than pretending that a graph can change dtype per lane.

use axmath::{floor_f32, sqrt_f32, sqrt_f64};

use crate::algebra::{F32Ops, RealOps, dot, matrix_vector};
use crate::precision::PrecisionEstimate;

/// A packed symmetric 2x2 matrix.
///
/// When this is an inverse screen-space covariance, [`Self::packed`] returns
/// the compositor convention `[a, b, c] = [xx, xy, yy]`.  Its Gaussian power
/// is `-0.5 * (a*dx^2 + c*dy^2) - b*dx*dy`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Symmetric2<T> {
    pub xx: T,
    pub xy: T,
    pub yy: T,
}

impl<T: Copy> Symmetric2<T> {
    #[inline]
    pub const fn packed(self) -> [T; 3] {
        [self.xx, self.xy, self.yy]
    }
}

/// A packed symmetric 3x3 matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Symmetric3<T> {
    pub xx: T,
    pub xy: T,
    pub xz: T,
    pub yy: T,
    pub yz: T,
    pub zz: T,
}

impl<T: Copy> Symmetric3<T> {
    /// Row-major expansion of the symmetric matrix.
    #[inline]
    pub const fn rows(self) -> [[T; 3]; 3] {
        [
            [self.xx, self.xy, self.xz],
            [self.xy, self.yy, self.yz],
            [self.xz, self.yz, self.zz],
        ]
    }
}

/// Mean and covariance of a three-dimensional Gaussian.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaussian3<T> {
    pub mean: [T; 3],
    pub covariance: Symmetric3<T>,
}

/// Pinhole camera parameters in the arithmetic backend's value type.
///
/// `world_to_view` stores camera basis vectors as rows.  The third row points
/// forward, so points in front of the camera have positive view-space `z`.
/// A negative vertical focal length gives the usual image convention in which
/// pixel `y` grows downward; this matches the existing `axrd` gsplat example.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeCamera<T> {
    pub origin: [T; 3],
    pub world_to_view: [[T; 3]; 3],
    pub focal_px: [T; 2],
    pub principal_px: [T; 2],
}

/// Perspective projection result before scalar visibility policy.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedGaussian<T> {
    pub mean_view: [T; 3],
    pub center_px: [T; 2],
    pub covariance_px: Symmetric2<T>,
}

#[inline]
fn symmetric3_product<R: RealOps>(
    ops: &mut R,
    matrix: Symmetric3<R::Value>,
    vector: [R::Value; 3],
) -> Result<[R::Value; 3], R::Error> {
    let xx = ops.mul(matrix.xx, vector[0])?;
    let xy = ops.mul(matrix.xy, vector[1])?;
    let xz = ops.mul(matrix.xz, vector[2])?;
    let x_xy = ops.add(xx, xy)?;
    let x = ops.add(x_xy, xz)?;

    let yx = ops.mul(matrix.xy, vector[0])?;
    let yy = ops.mul(matrix.yy, vector[1])?;
    let yz = ops.mul(matrix.yz, vector[2])?;
    let y_xy = ops.add(yx, yy)?;
    let y = ops.add(y_xy, yz)?;

    let zx = ops.mul(matrix.xz, vector[0])?;
    let zy = ops.mul(matrix.yz, vector[1])?;
    let zz = ops.mul(matrix.zz, vector[2])?;
    let z_xy = ops.add(zx, zy)?;
    let z = ops.add(z_xy, zz)?;
    Ok([x, y, z])
}

fn sparse_projection_covariance<R: RealOps>(
    ops: &mut R,
    matrix: Symmetric3<R::Value>,
    j_xx: R::Value,
    j_xz: R::Value,
    j_yy: R::Value,
    j_yz: R::Value,
) -> Result<Symmetric2<R::Value>, R::Error> {
    // The two Jacobian rows are [j_xx, 0, j_xz] and
    // [0, j_yy, j_yz]. Form only the nonzero products needed by
    // J * covariance * J^T instead of evaluating three dense bilinear forms.
    let x_x = ops.mul(matrix.xx, j_xx)?;
    let x_z = ops.mul(matrix.xz, j_xz)?;
    let product_x_x = ops.add(x_x, x_z)?;
    let y_x = ops.mul(matrix.xy, j_xx)?;
    let y_z = ops.mul(matrix.yz, j_xz)?;
    let product_x_y = ops.add(y_x, y_z)?;
    let z_x = ops.mul(matrix.xz, j_xx)?;
    let z_z = ops.mul(matrix.zz, j_xz)?;
    let product_x_z = ops.add(z_x, z_z)?;

    let y_y = ops.mul(matrix.yy, j_yy)?;
    let y_z = ops.mul(matrix.yz, j_yz)?;
    let product_y_y = ops.add(y_y, y_z)?;
    let z_y = ops.mul(matrix.yz, j_yy)?;
    let z_z = ops.mul(matrix.zz, j_yz)?;
    let product_y_z = ops.add(z_y, z_z)?;

    let xx_x = ops.mul(j_xx, product_x_x)?;
    let xx_z = ops.mul(j_xz, product_x_z)?;
    let xx = ops.add(xx_x, xx_z)?;
    let xy_y = ops.mul(j_yy, product_x_y)?;
    let xy_z = ops.mul(j_yz, product_x_z)?;
    let xy = ops.add(xy_y, xy_z)?;
    let yy_y = ops.mul(j_yy, product_y_y)?;
    let yy_z = ops.mul(j_yz, product_y_z)?;
    let yy = ops.add(yy_y, yy_z)?;

    Ok(Symmetric2 { xx, xy, yy })
}

/// Transform a Gaussian from world coordinates to camera coordinates.
///
/// The covariance transform is `R * covariance * R^T`; translation affects
/// only the mean.  The camera rows need not be hard-coded constants, so a
/// graph backend may make pose values graph inputs if desired.
pub fn camera_relative_gaussian<R: RealOps>(
    ops: &mut R,
    gaussian: Gaussian3<R::Value>,
    camera: PinholeCamera<R::Value>,
) -> Result<Gaussian3<R::Value>, R::Error> {
    let delta = [
        ops.sub(gaussian.mean[0], camera.origin[0])?,
        ops.sub(gaussian.mean[1], camera.origin[1])?,
        ops.sub(gaussian.mean[2], camera.origin[2])?,
    ];
    let mean = matrix_vector(ops, camera.world_to_view, delta)?;

    let row_x = camera.world_to_view[0];
    let row_y = camera.world_to_view[1];
    let row_z = camera.world_to_view[2];
    // Share each covariance-vector product across the three output entries in
    // its column. The previous dense bilinear helper recomputed it twice.
    let product_x = symmetric3_product(ops, gaussian.covariance, row_x)?;
    let product_y = symmetric3_product(ops, gaussian.covariance, row_y)?;
    let product_z = symmetric3_product(ops, gaussian.covariance, row_z)?;
    let covariance = Symmetric3 {
        xx: dot(ops, row_x, product_x)?,
        xy: dot(ops, row_x, product_y)?,
        xz: dot(ops, row_x, product_z)?,
        yy: dot(ops, row_y, product_y)?,
        yz: dot(ops, row_y, product_z)?,
        zz: dot(ops, row_z, product_z)?,
    };
    Ok(Gaussian3 { mean, covariance })
}

/// Project a camera-space Gaussian through a pinhole camera.
///
/// The screen covariance is `J * covariance * J^T`, with
/// `J = d([fx*x/z + cx, fy*y/z + cy]) / d([x,y,z])`.  The caller must
/// enforce a positive near plane before relying on the result.
pub fn project_camera_gaussian<R: RealOps>(
    ops: &mut R,
    gaussian: Gaussian3<R::Value>,
    focal_px: [R::Value; 2],
    principal_px: [R::Value; 2],
) -> Result<ProjectedGaussian<R::Value>, R::Error> {
    let [x, y, z] = gaussian.mean;
    let reciprocal_z = ops.reciprocal(z)?;
    let reciprocal_z_squared = ops.mul(reciprocal_z, reciprocal_z)?;

    let normalized_x = ops.mul(x, reciprocal_z)?;
    let normalized_y = ops.mul(y, reciprocal_z)?;
    let pixel_x_delta = ops.mul(focal_px[0], normalized_x)?;
    let pixel_y_delta = ops.mul(focal_px[1], normalized_y)?;
    let center_px = [
        ops.add(principal_px[0], pixel_x_delta)?,
        ops.add(principal_px[1], pixel_y_delta)?,
    ];

    let j_xx = ops.mul(focal_px[0], reciprocal_z)?;
    let fx_x = ops.mul(focal_px[0], x)?;
    let j_xz_unsigned = ops.mul(fx_x, reciprocal_z_squared)?;
    let j_xz = ops.negate(j_xz_unsigned)?;
    let j_yy = ops.mul(focal_px[1], reciprocal_z)?;
    let fy_y = ops.mul(focal_px[1], y)?;
    let j_yz_unsigned = ops.mul(fy_y, reciprocal_z_squared)?;
    let j_yz = ops.negate(j_yz_unsigned)?;
    let covariance_px =
        sparse_projection_covariance(ops, gaussian.covariance, j_xx, j_xz, j_yy, j_yz)?;
    Ok(ProjectedGaussian {
        mean_view: gaussian.mean,
        center_px,
        covariance_px,
    })
}

/// Transform and project a world-space Gaussian with shared equations.
pub fn project_world_gaussian<R: RealOps>(
    ops: &mut R,
    gaussian: Gaussian3<R::Value>,
    camera: PinholeCamera<R::Value>,
) -> Result<ProjectedGaussian<R::Value>, R::Error> {
    let view = camera_relative_gaussian(ops, gaussian, camera)?;
    project_camera_gaussian(ops, view, camera.focal_px, camera.principal_px)
}

/// Add a screen-space diagonal variance before inversion.
///
/// This is the graph-friendly form of 3DGS low-pass regularization.  Scalar
/// callers which also need finite checks and numerical repair should use
/// [`analyze_covariance2`].
pub fn add_diagonal_regularization<R: RealOps>(
    ops: &mut R,
    covariance: Symmetric2<R::Value>,
    variance: R::Value,
) -> Result<Symmetric2<R::Value>, R::Error> {
    Ok(Symmetric2 {
        xx: ops.add(covariance.xx, variance)?,
        xy: covariance.xy,
        yy: ops.add(covariance.yy, variance)?,
    })
}

/// Determinant of a packed symmetric 2x2 matrix.
pub fn symmetric2_determinant<R: RealOps>(
    ops: &mut R,
    matrix: Symmetric2<R::Value>,
) -> Result<R::Value, R::Error> {
    let diagonal = ops.mul(matrix.xx, matrix.yy)?;
    let off_diagonal = ops.mul(matrix.xy, matrix.xy)?;
    ops.sub(diagonal, off_diagonal)
}

/// Invert a packed symmetric 2x2 matrix.
///
/// This arithmetic-only routine cannot reject a nonpositive determinant.
/// Graph callers should add diagonal regularization first; scalar callers can
/// use [`analyze_covariance2`] for checked stabilization and inversion.
pub fn invert_symmetric2<R: RealOps>(
    ops: &mut R,
    matrix: Symmetric2<R::Value>,
) -> Result<Symmetric2<R::Value>, R::Error> {
    let determinant = symmetric2_determinant(ops, matrix)?;
    let reciprocal = ops.reciprocal(determinant)?;
    let negated_xy = ops.negate(matrix.xy)?;
    Ok(Symmetric2 {
        xx: ops.mul(matrix.yy, reciprocal)?,
        xy: ops.mul(negated_xy, reciprocal)?,
        yy: ops.mul(matrix.xx, reciprocal)?,
    })
}

/// Ordered eigenvalues of a symmetric 2x2 covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eigenvalues2 {
    pub min: f32,
    pub max: f32,
}

/// Checked, regularized screen covariance and its inverse conic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Covariance2Analysis {
    pub covariance: Symmetric2<f32>,
    pub conic: Symmetric2<f32>,
    /// Determinant is retained in `f64` so large, valid `f32` covariances do
    /// not overflow merely while being classified.
    pub determinant: f64,
    pub eigenvalues: Eigenvalues2,
    /// Total diagonal variance added, including any tiny numerical repair.
    pub diagonal_shift: f32,
}

impl Covariance2Analysis {
    /// Ratio of major to minor variance, saturated to a finite scalar so it is
    /// a valid [`PrecisionEstimate::condition`].
    #[inline]
    pub fn condition_number(self) -> f32 {
        let ratio = f64::from(self.eigenvalues.max) / f64::from(self.eigenvalues.min);
        if ratio >= f64::from(f32::MAX) {
            f32::MAX
        } else {
            ratio as f32
        }
    }
}

#[inline]
fn eigenvalues_f64(matrix: Symmetric2<f32>) -> (f64, f64) {
    let xx = f64::from(matrix.xx);
    let xy = f64::from(matrix.xy);
    let yy = f64::from(matrix.yy);
    let trace = xx + yy;
    let difference = xx - yy;
    let discriminant = sqrt_f64(difference * difference + 4.0 * xy * xy);
    let maximum = 0.5 * (trace + discriminant);
    let minimum = 0.5 * (trace - discriminant);
    (minimum, maximum)
}

/// Add conventional diagonal low-pass variance, repair round-off-level
/// indefiniteness, and calculate a stable inverse.
///
/// `regularization_variance` is added unconditionally to both diagonal terms,
/// matching the regularization used by 3DGS projection pipelines.  A further
/// scale-relative shift is applied only when the result is positive
/// semidefinite up to `f32` roundoff. Materially indefinite covariance is
/// rejected rather than changed into unrelated geometry. `None` also rejects
/// non-finite input, a negative regularizer, or a result whose inverse cannot
/// be represented.
pub fn analyze_covariance2(
    covariance: Symmetric2<f32>,
    regularization_variance: f32,
) -> Option<Covariance2Analysis> {
    if !covariance.xx.is_finite()
        || !covariance.xy.is_finite()
        || !covariance.yy.is_finite()
        || !regularization_variance.is_finite()
        || regularization_variance < 0.0
    {
        return None;
    }

    let mut xx = f64::from(covariance.xx) + f64::from(regularization_variance);
    let xy = f64::from(covariance.xy);
    let mut yy = f64::from(covariance.yy) + f64::from(regularization_variance);
    if !xx.is_finite() || !yy.is_finite() {
        return None;
    }

    let initial = Symmetric2 {
        xx: xx as f32,
        xy: covariance.xy,
        yy: yy as f32,
    };
    let (minimum, _) = eigenvalues_f64(initial);
    let scale = xx.abs().max(xy.abs()).max(yy.abs()).max(1.0);
    let numerical_floor = 4.0 * f64::from(f32::EPSILON) * scale;
    if !minimum.is_finite() || minimum < -numerical_floor {
        return None;
    }
    let repair = (numerical_floor - minimum).max(0.0);
    xx += repair;
    yy += repair;

    let stabilized = Symmetric2 {
        xx: xx as f32,
        xy: covariance.xy,
        yy: yy as f32,
    };
    if !stabilized.xx.is_finite() || !stabilized.yy.is_finite() {
        return None;
    }

    // Re-evaluate after the f64 -> f32 boundary. A second bounded,
    // scale-relative repair handles rounding that swallowed the first shift.
    let (minimum, _) = eigenvalues_f64(stabilized);
    if !minimum.is_finite() || minimum < -numerical_floor {
        return None;
    }
    let mut final_covariance = stabilized;
    let mut cast_repair = 0.0f64;
    if minimum < numerical_floor {
        cast_repair = numerical_floor - minimum;
        final_covariance.xx = (f64::from(final_covariance.xx) + cast_repair) as f32;
        final_covariance.yy = (f64::from(final_covariance.yy) + cast_repair) as f32;
    }

    let xx = f64::from(final_covariance.xx);
    let xy = f64::from(final_covariance.xy);
    let yy = f64::from(final_covariance.yy);
    let determinant = xx * yy - xy * xy;
    let (minimum, maximum) = eigenvalues_f64(final_covariance);
    if !determinant.is_finite()
        || determinant <= 0.0
        || !minimum.is_finite()
        || minimum <= 0.0
        || !maximum.is_finite()
        || maximum < minimum
    {
        return None;
    }

    let conic = Symmetric2 {
        xx: (yy / determinant) as f32,
        xy: (-xy / determinant) as f32,
        yy: (xx / determinant) as f32,
    };
    if !conic.xx.is_finite() || !conic.xy.is_finite() || !conic.yy.is_finite() {
        return None;
    }

    Some(Covariance2Analysis {
        covariance: final_covariance,
        conic,
        determinant,
        eigenvalues: Eigenvalues2 {
            min: minimum as f32,
            max: maximum as f32,
        },
        diagonal_shift: (f64::from(regularization_variance) + repair + cast_repair) as f32,
    })
}

/// Conservative screen-space support of a truncated Gaussian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianFootprint {
    /// Number of standard deviations retained by the rasterizer.
    pub sigma_extent: f32,
    /// Major-axis radius, useful for sorting/coarse culling.
    pub radius_px: f32,
    /// Exact axis-aligned half extents implied by the covariance diagonal.
    pub half_extent_px: [f32; 2],
}

impl GaussianFootprint {
    pub fn from_covariance(analysis: Covariance2Analysis, sigma_extent: f32) -> Option<Self> {
        if !sigma_extent.is_finite() || sigma_extent < 0.0 {
            return None;
        }
        let radius_px = sigma_extent * sqrt_f32(analysis.eigenvalues.max);
        let half_extent_px = [
            sigma_extent * sqrt_f32(analysis.covariance.xx),
            sigma_extent * sqrt_f32(analysis.covariance.yy),
        ];
        if !radius_px.is_finite()
            || !half_extent_px[0].is_finite()
            || !half_extent_px[1].is_finite()
        {
            return None;
        }
        Some(Self {
            sigma_extent,
            radius_px,
            half_extent_px,
        })
    }

    #[inline]
    pub fn bounds(self, center_px: [f32; 2]) -> Bounds2 {
        Bounds2 {
            min: [
                center_px[0] - self.half_extent_px[0],
                center_px[1] - self.half_extent_px[1],
            ],
            max: [
                center_px[0] + self.half_extent_px[0],
                center_px[1] + self.half_extent_px[1],
            ],
        }
    }
}

/// Continuous screen-space bounds in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds2 {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

/// Half-open tile rectangle `[min, max)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileBounds {
    pub min: [u32; 2],
    pub max: [u32; 2],
}

impl TileBounds {
    #[inline]
    pub fn tile_count(self) -> u64 {
        u64::from(self.max[0].saturating_sub(self.min[0]))
            * u64::from(self.max[1].saturating_sub(self.min[1]))
    }
}

/// Conservatively map a Gaussian footprint to an image's tile grid.
///
/// The footprint is an axis-aligned bound of the ellipse, so returned tiles
/// may contain false positives but never omit a potentially covered tile.
pub fn footprint_tile_bounds(
    center_px: [f32; 2],
    footprint: GaussianFootprint,
    image_extent: [u32; 2],
    tile_extent: [u32; 2],
) -> Option<TileBounds> {
    if image_extent[0] == 0
        || image_extent[1] == 0
        || tile_extent[0] == 0
        || tile_extent[1] == 0
        || !center_px[0].is_finite()
        || !center_px[1].is_finite()
    {
        return None;
    }
    let bounds = footprint.bounds(center_px);
    if !bounds.min[0].is_finite()
        || !bounds.min[1].is_finite()
        || !bounds.max[0].is_finite()
        || !bounds.max[1].is_finite()
        || bounds.max[0] < 0.0
        || bounds.max[1] < 0.0
        || bounds.min[0] >= image_extent[0] as f32
        || bounds.min[1] >= image_extent[1] as f32
    {
        return None;
    }

    let tile_counts = [
        image_extent[0].div_ceil(tile_extent[0]),
        image_extent[1].div_ceil(tile_extent[1]),
    ];
    let mut minimum = [0u32; 2];
    let mut maximum = [0u32; 2];
    for axis in 0..2 {
        let low = bounds.min[axis].max(0.0);
        let high = bounds.max[axis].min(image_extent[axis] as f32);
        minimum[axis] =
            (floor_f32(low / tile_extent[axis] as f32) as u32).min(tile_counts[axis] - 1);
        let inclusive_max =
            (floor_f32(high / tile_extent[axis] as f32) as u32).min(tile_counts[axis] - 1);
        maximum[axis] = inclusive_max + 1;
    }
    Some(TileBounds {
        min: minimum,
        max: maximum,
    })
}

/// Scalar raster policy applied after backend-neutral projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianRasterConfig {
    pub near_z: f32,
    pub regularization_variance_px2: f32,
    pub sigma_extent: f32,
    pub image_extent: [u32; 2],
    pub tile_extent: [u32; 2],
}

impl GaussianRasterConfig {
    pub fn is_valid(self) -> bool {
        self.near_z.is_finite()
            && self.near_z > 0.0
            && self.regularization_variance_px2.is_finite()
            && self.regularization_variance_px2 >= 0.0
            && self.sigma_extent.is_finite()
            && self.sigma_extent >= 0.0
            && self.image_extent[0] > 0
            && self.image_extent[1] > 0
            && self.tile_extent[0] > 0
            && self.tile_extent[1] > 0
    }
}

/// Why a Gaussian did not produce raster work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GaussianCullReason {
    InvalidInput,
    BehindNearPlane,
    DegenerateCovariance,
    OutsideImage,
}

/// Geometry ready for depth sorting, tile binning, and compositing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedGaussian {
    pub mean_view: [f32; 3],
    pub center_px: [f32; 2],
    pub covariance: Covariance2Analysis,
    pub footprint: GaussianFootprint,
    pub tiles: TileBounds,
}

/// Scalar visibility result.  Batch schedulers can compact the visible arm by
/// precision class before dispatching cached graph specializations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussianProjection {
    Visible(PreparedGaussian),
    Culled(GaussianCullReason),
}

#[inline]
fn finite_gaussian(gaussian: Gaussian3<f32>) -> bool {
    gaussian.mean.into_iter().all(f32::is_finite)
        && gaussian
            .covariance
            .rows()
            .into_iter()
            .flatten()
            .all(f32::is_finite)
}

#[inline]
fn finite_camera(camera: PinholeCamera<f32>) -> bool {
    camera.origin.into_iter().all(f32::is_finite)
        && camera
            .world_to_view
            .into_iter()
            .flatten()
            .all(f32::is_finite)
        && camera
            .focal_px
            .into_iter()
            .all(|focal| focal.is_finite() && focal != 0.0)
        && camera.principal_px.into_iter().all(f32::is_finite)
}

#[inline]
fn infallible<T>(result: Result<T, core::convert::Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

/// Project, stabilize, cull, and bin one scalar Gaussian.
///
/// This is an allocation-free reference/orchestration path.  Large batches
/// should use the generic projection equations in a graph, then compact the
/// scalar visibility/precision results into backend-specialized work lists.
pub fn prepare_gaussian(
    gaussian: Gaussian3<f32>,
    camera: PinholeCamera<f32>,
    config: GaussianRasterConfig,
) -> GaussianProjection {
    if !config.is_valid() || !finite_gaussian(gaussian) || !finite_camera(camera) {
        return GaussianProjection::Culled(GaussianCullReason::InvalidInput);
    }

    let mut ops = F32Ops;
    let view = infallible(camera_relative_gaussian(&mut ops, gaussian, camera));
    if !finite_gaussian(view) {
        return GaussianProjection::Culled(GaussianCullReason::InvalidInput);
    }
    if view.mean[2] < config.near_z {
        return GaussianProjection::Culled(GaussianCullReason::BehindNearPlane);
    }
    let projected = infallible(project_camera_gaussian(
        &mut ops,
        view,
        camera.focal_px,
        camera.principal_px,
    ));
    let Some(covariance) =
        analyze_covariance2(projected.covariance_px, config.regularization_variance_px2)
    else {
        return GaussianProjection::Culled(GaussianCullReason::DegenerateCovariance);
    };
    let Some(footprint) = GaussianFootprint::from_covariance(covariance, config.sigma_extent)
    else {
        return GaussianProjection::Culled(GaussianCullReason::DegenerateCovariance);
    };
    let Some(tiles) = footprint_tile_bounds(
        projected.center_px,
        footprint,
        config.image_extent,
        config.tile_extent,
    ) else {
        return GaussianProjection::Culled(GaussianCullReason::OutsideImage);
    };
    GaussianProjection::Visible(PreparedGaussian {
        mean_view: projected.mean_view,
        center_px: projected.center_px,
        covariance,
        footprint,
        tiles,
    })
}

/// Error terms for dynamic-precision projection classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianProjectionError {
    /// Absolute quantization/approximation error of the projected center.
    pub center_error_px: f32,
    /// Relative error of the projected covariance or conic.
    pub covariance_relative_error: f32,
    /// Distance to the nearest culling, tile, or application event boundary.
    pub boundary_margin: f32,
}

impl PreparedGaussian {
    /// Convert concrete projection error bounds into the backend-neutral
    /// precision estimator used for batch compaction.
    ///
    /// Covariance error is scaled by the visible major-axis radius; the
    /// covariance condition number remains a separate sensitivity multiplier.
    pub fn precision_estimate(self, error: GaussianProjectionError) -> Option<PrecisionEstimate> {
        if !error.center_error_px.is_finite()
            || error.center_error_px < 0.0
            || !error.covariance_relative_error.is_finite()
            || error.covariance_relative_error < 0.0
            || !error.boundary_margin.is_finite()
        {
            return None;
        }
        let screen_error_px =
            error.center_error_px + error.covariance_relative_error * self.footprint.radius_px;
        let estimate = PrecisionEstimate {
            screen_error_px,
            condition: self.covariance.condition_number(),
            boundary_margin: error.boundary_margin,
        };
        estimate.is_valid().then_some(estimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::{PrecisionClass, ProjectionPrecisionPolicy};
    use core::convert::Infallible;

    #[derive(Default)]
    struct CountingOps {
        operations: usize,
    }

    impl CountingOps {
        fn operation(&mut self) {
            self.operations += 1;
        }
    }

    impl RealOps for CountingOps {
        type Value = ();
        type Error = Infallible;

        fn constant(&mut self, _: f32) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn add(&mut self, _: (), _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn sub(&mut self, _: (), _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn mul(&mut self, _: (), _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn negate(&mut self, _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn reciprocal(&mut self, _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn rsqrt(&mut self, _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn minimum(&mut self, _: (), _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }

        fn maximum(&mut self, _: (), _: ()) -> Result<(), Infallible> {
            self.operation();
            Ok(())
        }
    }

    const IDENTITY_CAMERA: PinholeCamera<f32> = PinholeCamera {
        origin: [0.0, 0.0, 0.0],
        world_to_view: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        focal_px: [100.0, -100.0],
        principal_px: [320.0, 240.0],
    };

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} vs {expected}"
        );
    }

    fn diagonal_gaussian(mean: [f32; 3], diagonal: [f32; 3]) -> Gaussian3<f32> {
        Gaussian3 {
            mean,
            covariance: Symmetric3 {
                xx: diagonal[0],
                xy: 0.0,
                xz: 0.0,
                yy: diagonal[1],
                yz: 0.0,
                zz: diagonal[2],
            },
        }
    }

    #[test]
    fn covariance_projection_operation_budgets_are_locked() {
        let gaussian = Gaussian3 {
            mean: [(); 3],
            covariance: Symmetric3 {
                xx: (),
                xy: (),
                xz: (),
                yy: (),
                yz: (),
                zz: (),
            },
        };
        let camera = PinholeCamera {
            origin: [(); 3],
            world_to_view: [[(); 3]; 3],
            focal_px: [(); 2],
            principal_px: [(); 2],
        };
        let mut ops = CountingOps::default();
        camera_relative_gaussian(&mut ops, gaussian, camera).unwrap();
        assert_eq!(
            ops.operations, 112,
            "camera covariance transform must share its three matrix products"
        );

        ops.operations = 0;
        project_camera_gaussian(&mut ops, gaussian, [(); 2], [(); 2]).unwrap();
        assert_eq!(
            ops.operations, 40,
            "perspective covariance must exploit the sparse two-row Jacobian"
        );
    }

    #[test]
    fn camera_transform_rotates_mean_and_covariance() {
        let camera = PinholeCamera {
            origin: [1.0, 2.0, 3.0],
            world_to_view: [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ..IDENTITY_CAMERA
        };
        let mut ops = F32Ops;
        let view = camera_relative_gaussian(
            &mut ops,
            diagonal_gaussian([3.0, 5.0, 7.0], [1.0, 4.0, 9.0]),
            camera,
        )
        .unwrap();
        assert_eq!(view.mean, [3.0, -2.0, 4.0]);
        assert_eq!(
            view.covariance,
            Symmetric3 {
                xx: 4.0,
                xy: 0.0,
                xz: 0.0,
                yy: 1.0,
                yz: 0.0,
                zz: 9.0,
            }
        );
    }

    #[test]
    fn perspective_projection_matches_analytic_jacobian() {
        let mut ops = F32Ops;
        let projected = project_world_gaussian(
            &mut ops,
            diagonal_gaussian([2.0, -1.0, 10.0], [4.0, 4.0, 4.0]),
            IDENTITY_CAMERA,
        )
        .unwrap();
        assert_eq!(projected.mean_view, [2.0, -1.0, 10.0]);
        assert_eq!(projected.center_px, [340.0, 250.0]);
        close(projected.covariance_px.xx, 416.0, 1.0e-5);
        close(projected.covariance_px.xy, 8.0, 1.0e-5);
        close(projected.covariance_px.yy, 404.0, 1.0e-5);
    }

    #[test]
    fn optical_axis_matches_existing_isotropic_gsplat_projection() {
        let mut ops = F32Ops;
        let projected = project_world_gaussian(
            &mut ops,
            diagonal_gaussian([0.0, 0.0, 10.0], [4.0, 4.0, 4.0]),
            IDENTITY_CAMERA,
        )
        .unwrap();
        // axrd's isotropic shortcut is sigma_px = focal_px * sigma / z:
        // 100 * 2 / 10 = 20 px, hence variance 400 px^2.
        assert_eq!(projected.center_px, [320.0, 240.0]);
        assert_eq!(
            projected.covariance_px,
            Symmetric2 {
                xx: 400.0,
                xy: 0.0,
                yy: 400.0,
            }
        );
        let analysis = analyze_covariance2(projected.covariance_px, 0.0).unwrap();
        assert_eq!(analysis.conic.packed(), [1.0 / 400.0, 0.0, 1.0 / 400.0]);
    }

    #[test]
    fn regularization_repairs_rank_one_covariance_and_inverts_stably() {
        let analysis = analyze_covariance2(
            Symmetric2 {
                xx: 4.0,
                xy: 2.0,
                yy: 1.0,
            },
            0.25,
        )
        .unwrap();
        assert_eq!(
            analysis.covariance,
            Symmetric2 {
                xx: 4.25,
                xy: 2.0,
                yy: 1.25,
            }
        );
        close(analysis.determinant as f32, 1.3125, 1.0e-6);
        close(analysis.eigenvalues.min, 0.25, 1.0e-6);
        close(analysis.eigenvalues.max, 5.25, 1.0e-6);
        close(analysis.condition_number(), 21.0, 1.0e-5);
        close(analysis.conic.xx, 1.25 / 1.3125, 1.0e-6);
        close(analysis.conic.xy, -2.0 / 1.3125, 1.0e-6);
        close(analysis.conic.yy, 4.25 / 1.3125, 1.0e-6);
    }

    #[test]
    fn covariance_repair_is_limited_to_roundoff_scale() {
        let roundoff_indefinite = Symmetric2 {
            xx: 1.0,
            xy: 1.0 + f32::EPSILON,
            yy: 1.0,
        };
        let repaired = analyze_covariance2(roundoff_indefinite, 0.0)
            .expect("one-ULP indefiniteness should be repairable");
        assert!(repaired.diagonal_shift > 0.0);
        assert!(repaired.eigenvalues.min > 0.0);

        let materially_indefinite = Symmetric2 {
            xx: 1.0,
            xy: 2.0,
            yy: 1.0,
        };
        assert_eq!(analyze_covariance2(materially_indefinite, 0.0), None);
        assert_eq!(
            analyze_covariance2(
                Symmetric2 {
                    xx: -1.0,
                    xy: 0.0,
                    yy: -1.0,
                },
                0.0,
            ),
            None
        );
    }

    #[test]
    fn generic_regularization_and_inverse_share_graph_equations() {
        let mut ops = F32Ops;
        let covariance = Symmetric2 {
            xx: 4.0,
            xy: 2.0,
            yy: 1.0,
        };
        let regularized = add_diagonal_regularization(&mut ops, covariance, 0.25).unwrap();
        let determinant = symmetric2_determinant(&mut ops, regularized).unwrap();
        let conic = invert_symmetric2(&mut ops, regularized).unwrap();
        close(determinant, 1.3125, 1.0e-6);
        close(conic.xx, 1.25 / 1.3125, 1.0e-6);
        close(conic.xy, -2.0 / 1.3125, 1.0e-6);
        close(conic.yy, 4.25 / 1.3125, 1.0e-6);
    }

    #[test]
    fn conic_packing_matches_axnn_cpu_power_convention() {
        let analysis = analyze_covariance2(
            Symmetric2 {
                xx: 4.0,
                xy: 0.0,
                yy: 9.0,
            },
            0.0,
        )
        .unwrap();
        let [a, b, c] = analysis.conic.packed();
        let (dx, dy) = (2.0, 3.0);
        let compositor_power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy;
        close(compositor_power, -1.0, 1.0e-6);
    }

    #[test]
    fn anisotropic_footprint_produces_conservative_half_open_tiles() {
        let analysis = analyze_covariance2(
            Symmetric2 {
                xx: 4.0,
                xy: 0.0,
                yy: 9.0,
            },
            0.0,
        )
        .unwrap();
        let footprint = GaussianFootprint::from_covariance(analysis, 3.0).unwrap();
        assert_eq!(footprint.half_extent_px, [6.0, 9.0]);
        assert_eq!(footprint.radius_px, 9.0);
        assert_eq!(
            footprint_tile_bounds([15.0, 18.0], footprint, [64, 64], [16, 16]),
            Some(TileBounds {
                min: [0, 0],
                max: [2, 2],
            })
        );
        assert_eq!(
            footprint_tile_bounds([-20.0, 18.0], footprint, [64, 64], [16, 16]),
            None
        );
    }

    #[test]
    fn prepare_culls_near_and_offscreen_work() {
        let config = GaussianRasterConfig {
            near_z: 0.8,
            regularization_variance_px2: 0.3,
            sigma_extent: 3.0,
            image_extent: [640, 480],
            tile_extent: [16, 16],
        };
        assert_eq!(
            prepare_gaussian(
                diagonal_gaussian([0.0, 0.0, 0.5], [1.0; 3]),
                IDENTITY_CAMERA,
                config,
            ),
            GaussianProjection::Culled(GaussianCullReason::BehindNearPlane)
        );
        assert_eq!(
            prepare_gaussian(
                diagonal_gaussian([100.0, 0.0, 2.0], [0.01; 3]),
                IDENTITY_CAMERA,
                config,
            ),
            GaussianProjection::Culled(GaussianCullReason::OutsideImage)
        );
    }

    #[test]
    fn projected_footprint_feeds_semantic_precision_policy() {
        let config = GaussianRasterConfig {
            near_z: 0.8,
            regularization_variance_px2: 0.0,
            sigma_extent: 3.0,
            image_extent: [640, 480],
            tile_extent: [16, 16],
        };
        let GaussianProjection::Visible(prepared) = prepare_gaussian(
            diagonal_gaussian([0.0, 0.0, 100.0], [4.0, 9.0, 1.0]),
            IDENTITY_CAMERA,
            config,
        ) else {
            panic!("far-field Gaussian should remain visible");
        };
        close(prepared.footprint.radius_px, 9.0, 1.0e-6);
        let estimate = prepared
            .precision_estimate(GaussianProjectionError {
                center_error_px: 0.05,
                covariance_relative_error: 0.1,
                boundary_margin: 2.0,
            })
            .unwrap();
        close(estimate.screen_error_px, 0.95, 1.0e-6);
        close(estimate.condition, 2.25, 1.0e-6);
        assert_eq!(
            ProjectionPrecisionPolicy::DEFAULT.classify(estimate),
            Some(PrecisionClass::Near)
        );

        let boundary_sensitive = prepared
            .precision_estimate(GaussianProjectionError {
                center_error_px: 0.0,
                covariance_relative_error: 0.0,
                boundary_margin: 0.0,
            })
            .unwrap();
        assert_eq!(
            ProjectionPrecisionPolicy::DEFAULT.classify(boundary_sensitive),
            Some(PrecisionClass::Sensitive)
        );
    }
}
