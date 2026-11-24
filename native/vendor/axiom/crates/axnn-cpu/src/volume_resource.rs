//! Checked borrowed RGB volumes and exact planned trilinear sampling.

use core::simd::Simd;
use core::simd::prelude::*;

use axnn::lowering::{
    KernelEffects, KernelParams, KernelResult, ResourceKey, SemanticOp, VolumeRgbParams,
    VolumeRgbStorage,
};
use axnn::{DType, Shape};

use crate::CpuError;
use crate::planner::{
    CpuImplementation, CpuKernelPlan, broadcast_index, typed_input_matches, typed_output_matches,
};

#[derive(Clone, Copy, Debug)]
enum RgbVolumeBacking<'a> {
    F32(&'a [[f32; 3]]),
    Uf16(&'a [[u16; 3]]),
    Uf8(&'a [[u8; 3]]),
}

/// Immutable caller-owned RGB triples, ordered x + nx * (y + ny * z).
///
/// Construction validates exact dimensions/length and finite nonnegative
/// contents in release builds. F32 negative zero is allowed; unsigned E5M11
/// words must be at most 0xf7ff and E5M3 bytes at most 0xf7. Dimensions, component counts, and byte sizes
/// use the checked VolumeRgbParams contract. No payload copy or allocation
/// occurs, and a kernel plan never retains this view or its storage pointer.
#[derive(Clone, Copy, Debug)]
pub struct RgbVolumeRef<'a> {
    dims: [u32; 3],
    backing: RgbVolumeBacking<'a>,
}

impl<'a> RgbVolumeRef<'a> {
    pub fn new_f32(rgb: &'a [[f32; 3]], dims: [u32; 3]) -> Result<Self, CpuError> {
        validate_descriptor(
            dims,
            VolumeRgbStorage::F32,
            rgb.len(),
            core::mem::size_of_val(rgb),
        )?;
        if rgb
            .iter()
            .flatten()
            .any(|&value| !value.is_finite() || value < 0.0)
        {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed RGB volume values",
            });
        }
        Ok(Self {
            dims,
            backing: RgbVolumeBacking::F32(rgb),
        })
    }

    pub fn new_uf16(rgb: &'a [[u16; 3]], dims: [u32; 3]) -> Result<Self, CpuError> {
        validate_descriptor(
            dims,
            VolumeRgbStorage::Uf16E5M11,
            rgb.len(),
            core::mem::size_of_val(rgb),
        )?;
        if rgb.iter().flatten().any(|&value| value > 0xf7ff) {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed RGB volume values",
            });
        }
        Ok(Self {
            dims,
            backing: RgbVolumeBacking::Uf16(rgb),
        })
    }

    /// Borrow finite nonnegative E5M3 storage without copying or widening it.
    pub fn new_uf8e5m3(rgb: &'a [[u8; 3]], dims: [u32; 3]) -> Result<Self, CpuError> {
        validate_descriptor(
            dims,
            VolumeRgbStorage::Uf8E5M3,
            rgb.len(),
            core::mem::size_of_val(rgb),
        )?;
        if rgb.iter().flatten().any(|&value| value > 0xf7) {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed RGB volume values",
            });
        }
        Ok(Self {
            dims,
            backing: RgbVolumeBacking::Uf8(rgb),
        })
    }

    pub const fn as_uf8e5m3(self) -> Option<&'a [[u8; 3]]> {
        match self.backing {
            RgbVolumeBacking::Uf8(rgb) => Some(rgb),
            RgbVolumeBacking::F32(_) | RgbVolumeBacking::Uf16(_) => None,
        }
    }

    pub const fn dims(self) -> [u32; 3] {
        self.dims
    }

    pub const fn storage(self) -> VolumeRgbStorage {
        match self.backing {
            RgbVolumeBacking::F32(_) => VolumeRgbStorage::F32,
            RgbVolumeBacking::Uf16(_) => VolumeRgbStorage::Uf16E5M11,
            RgbVolumeBacking::Uf8(_) => VolumeRgbStorage::Uf8E5M3,
        }
    }

    /// Borrow the validated F32 backing, without copying or changing storage.
    pub const fn as_f32(self) -> Option<&'a [[f32; 3]]> {
        match self.backing {
            RgbVolumeBacking::F32(rgb) => Some(rgb),
            RgbVolumeBacking::Uf16(_) | RgbVolumeBacking::Uf8(_) => None,
        }
    }

    /// Borrow validated unsigned E5M11 words; these are not IEEE F16 values.
    pub const fn as_uf16(self) -> Option<&'a [[u16; 3]]> {
        match self.backing {
            RgbVolumeBacking::Uf16(rgb) => Some(rgb),
            RgbVolumeBacking::F32(_) | RgbVolumeBacking::Uf8(_) => None,
        }
    }
}

fn validate_descriptor(
    dims: [u32; 3],
    storage: VolumeRgbStorage,
    voxels: usize,
    bytes: usize,
) -> Result<(), CpuError> {
    let params = VolumeRgbParams::new(
        ResourceKey {
            group: 0,
            binding: 0,
        },
        dims,
        storage,
    )
    .ok_or(CpuError::ShapeMismatch {
        op: "borrowed RGB volume dimensions",
    })?;
    if params.voxel_count() != Some(voxels) {
        return Err(CpuError::SizeMismatch {
            expected: params.storage_bytes().expect("validated volume size"),
            actual: bytes,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SampleGeometry {
    dims: [usize; 3],
    last: [f32; 3],
    /// Corners accumulated, a prefix of the z/y/x order: all eight, or the
    /// four of the z = 0 slice when the volume is one slice deep. The other
    /// four would repeat that slice with weight `(wx * wy) * 0 = +0`; the
    /// volume is validated finite and nonnegative and the sum starts at +0,
    /// so their products are +0 and adding them changes no bit.
    corners: usize,
}

impl SampleGeometry {
    fn new(dims: [u32; 3]) -> Self {
        Self {
            dims: dims.map(|v| v as usize),
            last: dims.map(|v| (v - 1) as f32),
            corners: if dims[2] == 1 { 4 } else { 8 },
        }
    }

    #[inline]
    fn stencil(self, point: [f32; 3]) -> Option<([usize; 8], [f32; 8])> {
        if point.iter().any(|value| value.is_nan()) {
            return None;
        }
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        let mut fraction = [0.0; 3];
        for axis in 0..3 {
            if self.dims[axis] != 1 {
                let coordinate = point[axis].clamp(0.0, self.last[axis]);
                lo[axis] = coordinate as usize;
                hi[axis] = (lo[axis] + 1).min(self.dims[axis] - 1);
                fraction[axis] = coordinate - lo[axis] as f32;
            }
        }
        let mut indices = [0; 8];
        let mut weights = [0.0; 8];
        for z in 0..self.corners / 4 {
            for y in 0..2 {
                for x in 0..2 {
                    let ix = if x == 0 { lo[0] } else { hi[0] };
                    let iy = if y == 0 { lo[1] } else { hi[1] };
                    let iz = if z == 0 { lo[2] } else { hi[2] };
                    let corner = x + 2 * y + 4 * z;
                    let wx = if x == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    };
                    let wy = if y == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    };
                    let wz = if z == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    };
                    weights[corner] = (wx * wy) * wz;
                    indices[corner] = ix + self.dims[0] * (iy + self.dims[1] * iz);
                }
            }
        }
        Some((indices, weights))
    }
}

#[inline]
fn accumulate_rgb<const VECTOR: bool>(
    corners: usize,
    weights: [f32; 8],
    mut colour_at: impl FnMut(usize) -> [f32; 3],
) -> [f32; 3] {
    let mut scalar = [0.0_f32; 3];
    let mut vector = Simd::<f32, 4>::splat(0.0);
    for (corner, weight) in weights.into_iter().enumerate().take(corners) {
        let colour = colour_at(corner);
        if VECTOR {
            // SIMD is across RGB channels, not across the ordered corners.
            // Keep separate multiply/add operations; no fused accumulation.
            let product =
                Simd::from_array([colour[0], colour[1], colour[2], 0.0]) * Simd::splat(weight);
            vector += product;
        } else {
            for channel in 0..3 {
                let product = colour[channel] * weight;
                scalar[channel] += product;
            }
        }
    }
    if VECTOR {
        [vector[0], vector[1], vector[2]]
    } else {
        scalar
    }
}

trait SampleRgb {
    fn sample<const VECTOR: bool>(&self, geometry: SampleGeometry, point: [f32; 3]) -> [f32; 3];
}

impl SampleRgb for [[f32; 3]] {
    #[inline]
    fn sample<const VECTOR: bool>(&self, geometry: SampleGeometry, point: [f32; 3]) -> [f32; 3] {
        let Some((indices, weights)) = geometry.stencil(point) else {
            return [f32::NAN; 3];
        };
        accumulate_rgb::<VECTOR>(geometry.corners, weights, |corner| self[indices[corner]])
    }
}

impl SampleRgb for [[u16; 3]] {
    #[inline]
    fn sample<const VECTOR: bool>(&self, geometry: SampleGeometry, point: [f32; 3]) -> [f32; 3] {
        sample_uf16_direct::<VECTOR>(self, geometry, point)
    }
}

impl SampleRgb for [[u8; 3]] {
    #[inline]
    fn sample<const VECTOR: bool>(&self, geometry: SampleGeometry, point: [f32; 3]) -> [f32; 3] {
        sample_uf8::<VECTOR>(
            self,
            geometry,
            point,
            crate::arch::portable::f8::finite_uf8e5m3_rgb,
        )
    }
}

// The catalog admits this decoder separately from portable SIMD. Reuse the
// same interpolation/broadcast engine; only the corner expansion differs.
struct TierUf8<'a>(&'a [[u8; 3]]);

impl SampleRgb for TierUf8<'_> {
    #[inline]
    fn sample<const VECTOR: bool>(&self, geometry: SampleGeometry, point: [f32; 3]) -> [f32; 3] {
        sample_uf8::<VECTOR>(self.0, geometry, point, crate::arch::uf::finite_uf8e5m3_rgb)
    }
}

#[inline]
fn sample_uf8<const VECTOR: bool>(
    rgb: &[[u8; 3]],
    geometry: SampleGeometry,
    point: [f32; 3],
    decode: impl Fn([u8; 3]) -> [f32; 4],
) -> [f32; 3] {
    let Some((indices, weights)) = geometry.stencil(point) else {
        return [f32::NAN; 3];
    };
    if !VECTOR {
        return accumulate_rgb::<false>(geometry.corners, weights, |corner| {
            rgb[indices[corner]].map(crate::cast::uf8e5m3_to_f32)
        });
    }
    let mut sum = Simd::<f32, 4>::splat(0.0);
    for (index, weight) in indices.into_iter().zip(weights).take(geometry.corners) {
        // Validated finite E5M3 triples; the selected architecture expands
        // only this corner into registers, never a persistent F32 mirror.
        let color = Simd::from_array(decode(rgb[index]));
        sum += color * Simd::splat(weight);
    }
    [sum[0], sum[1], sum[2]]
}

// Previous implementation retained only as a matched benchmark control.
#[cfg(test)]
#[inline]
fn sample_uf16_gathered<const VECTOR: bool>(
    rgb: &[[u16; 3]],
    geometry: SampleGeometry,
    point: [f32; 3],
) -> [f32; 3] {
    let Some((indices, weights)) = geometry.stencil(point) else {
        return [f32::NAN; 3];
    };
    let mut gathered = [0_u16; 24];
    for (colour, index) in gathered.as_chunks_mut::<3>().0.iter_mut().zip(indices) {
        *colour = rgb[index];
    }
    let mut expanded = [0.0_f32; 24];
    // Expand only these eight RGB corners; no persistent F32 mirror.
    crate::uf::promote_uf16e5m11_to_f32(&gathered, &mut expanded, 24);
    let colours = expanded.as_chunks::<3>().0;
    accumulate_rgb::<VECTOR>(geometry.corners, weights, |corner| colours[corner])
}

// Decode directly into the ordered RGB accumulator: no 24-component gathered
// array, promotion call, or expanded scratch. Selection remains in SampleRgb;
// caller resource validation and scalar/SIMD policy dispatch are unchanged.
#[inline]
fn sample_uf16_direct<const VECTOR: bool>(
    rgb: &[[u16; 3]],
    geometry: SampleGeometry,
    point: [f32; 3],
) -> [f32; 3] {
    let Some((indices, weights)) = geometry.stencil(point) else {
        return [f32::NAN; 3];
    };
    if !VECTOR {
        return accumulate_rgb::<false>(geometry.corners, weights, |corner| {
            rgb[indices[corner]].map(crate::uf::uf16e5m11_to_f32)
        });
    }
    let mut sum = Simd::<f32, 4>::splat(0.0);
    for (index, weight) in indices.into_iter().zip(weights).take(geometry.corners) {
        let [r, g, b] = rgb[index];
        // RgbVolumeRef excludes infinity/NaN encodings before invocation.
        // Handle zero/subnormal words without subnormal F32 intermediates.
        let bits = Simd::from_array([r as u32, g as u32, b as u32, 0]);
        let normal = Simd::<f32, 4>::from_bits((bits << Simd::splat(12)) + Simd::splat(112 << 23));
        let subnormal = bits.cast::<f32>() * Simd::splat(0.000_000_029_802_322);
        let color = bits.simd_lt(Simd::splat(0x800)).select(subnormal, normal);
        sum += color * Simd::splat(weight);
    }
    [sum[0], sum[1], sum[2]]
}

/// Whether a sample can take [`sample_uf16_planar_lanes`]: a one-slice
/// volume whose corner indices fit its u32 lanes, dense x and y
/// coordinates, and z dense or uniform.
fn planar_lanes(geometry: SampleGeometry, coordinates: [&[f32]; 3], elements: usize) -> bool {
    geometry.corners == 4
        && geometry.dims[0].checked_mul(geometry.dims[1]).is_some_and(|texels| texels <= u32::MAX as usize)
        && coordinates[0].len() == elements
        && coordinates[1].len() == elements
        && (coordinates[2].len() == 1 || coordinates[2].len() == elements)
}

/// Four samples per vector for a one-slice UF16 volume at dense x/y
/// coordinates — every table a shader reads as an image. The stencil, the
/// UF16 decode and the ordered multiply/adds run four lanes wide, one vector
/// per colour channel, where [`sample_uf16_direct`] spends a vector on one
/// sample's RGB. Each lane does exactly what that path does for its sample —
/// the same clamp, the same four corners in z/y/x order, separate F32
/// multiply then add from positive zero, NaN coordinates giving NaN — so
/// every output bit matches. A tail of fewer than four samples takes the
/// per-sample path.
fn sample_uf16_planar_lanes(
    rgb: &[[u16; 3]],
    geometry: SampleGeometry,
    coordinates: [&[f32]; 3],
    outputs: [&mut [f32]; 3],
) {
    const LANES: usize = 4;
    type F = Simd<f32, LANES>;
    type U = Simd<u32, LANES>;
    let [red, green, blue] = outputs;
    let elements = red.len();
    let full = elements / LANES * LANES;
    let at = |coordinate: &[f32], base: usize| {
        if coordinate.len() == 1 {
            F::splat(coordinate[0])
        } else {
            F::from_slice(&coordinate[base..base + LANES])
        }
    };
    // One axis of the stencil, as `SampleGeometry::stencil` computes it.
    let axis = |point: F, axis: usize| -> (U, U, F) {
        if geometry.dims[axis] == 1 {
            return (U::splat(0), U::splat(0), F::splat(0.0));
        }
        let last = F::splat(geometry.last[axis]);
        // f32::clamp for every non-NaN value; NaN lanes are replaced below.
        let coordinate = point
            .simd_lt(F::splat(0.0))
            .select(F::splat(0.0), point.simd_gt(last).select(last, point));
        let lo = coordinate.cast::<u32>();
        let hi = (lo + U::splat(1)).simd_min(U::splat(geometry.dims[axis] as u32 - 1));
        (lo, hi, coordinate - lo.cast::<f32>())
    };
    let decode = |bits: U| -> F {
        // As sample_uf16_direct: finite UF16 E5M11 without subnormal F32
        // intermediates.
        let normal = F::from_bits((bits << U::splat(12)) + U::splat(112 << 23));
        let subnormal = bits.cast::<f32>() * F::splat(0.000_000_029_802_322);
        bits.simd_lt(U::splat(0x800)).select(subnormal, normal)
    };
    let width = geometry.dims[0] as u32;
    for base in (0..full).step_by(LANES) {
        let point = [at(coordinates[0], base), at(coordinates[1], base), at(coordinates[2], base)];
        let nan = point[0].is_nan() | point[1].is_nan() | point[2].is_nan();
        let (x_lo, x_hi, x_fraction) = axis(point[0], 0);
        let (y_lo, y_hi, y_fraction) = axis(point[1], 1);
        let one = F::splat(1.0);
        // Corner order x + 2y; z = 0's weight (1 - 0) multiplies exactly.
        let corners = [
            (x_lo + U::splat(width) * y_lo, (one - x_fraction) * (one - y_fraction)),
            (x_hi + U::splat(width) * y_lo, x_fraction * (one - y_fraction)),
            (x_lo + U::splat(width) * y_hi, (one - x_fraction) * y_fraction),
            (x_hi + U::splat(width) * y_hi, x_fraction * y_fraction),
        ];
        let mut sum = [F::splat(0.0); 3];
        for (index, weight) in corners {
            let index = index.to_array();
            for channel in 0..3 {
                let bits = U::from_array(core::array::from_fn(|lane| u32::from(rgb[index[lane] as usize][channel])));
                let product = decode(bits) * weight;
                sum[channel] += product;
            }
        }
        for (channel, output) in [&mut *red, &mut *green, &mut *blue].into_iter().enumerate() {
            nan.select(F::splat(f32::NAN), sum[channel]).copy_to_slice(&mut output[base..base + LANES]);
        }
    }
    for index in full..elements {
        let point = core::array::from_fn(|axis| {
            let coordinate = coordinates[axis];
            coordinate[if coordinate.len() == 1 { 0 } else { index }]
        });
        let value = sample_uf16_direct::<true>(rgb, geometry, point);
        red[index] = value[0];
        green[index] = value[1];
        blue[index] = value[2];
    }
}

#[cfg(test)]
#[path = "volume_probe.rs"]
mod probe;

/// Sample a borrowed RGB volume into three planned F32 field planes.
///
/// Coordinates are in probe-index space and clamp to the volume's edges;
/// infinities clamp and any NaN coordinate returns NaN in all channels.
/// Eight z/y/x corners accumulate from positive zero using (wx * wy) * wz
/// and separate F32 multiply/add operations. Singleton axes select index zero
/// with fraction zero. No corner is omitted except the z = 1 slice of a
/// one-slice volume, whose four products are exactly +0 (finite nonnegative
/// colour times `(wx * wy) * 0`) and leave every bit of the sum unchanged:
/// a 2-D table costs four corners, not eight.
///
/// Inputs must have their exact planned lengths and broadcast to the output
/// field. All descriptors must be contiguous with offset zero; ordinary Rust
/// subslices are supported. Every shape/resource/output check precedes writes.
/// Storage and scalar/dense coordinate choices are outside the sample loop.
/// There is no allocation, scratch argument, or retained resource pointer.
pub fn invoke_volume_trilinear_rgb(
    plan: &CpuKernelPlan,
    volume: RgbVolumeRef<'_>,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    out_r: &mut [f32],
    out_g: &mut [f32],
    out_b: &mut [f32],
) -> Result<(), CpuError> {
    let KernelParams::VolumeTrilinearRgb(params) = plan.params() else {
        return Err(CpuError::UnsupportedOp {
            op: "planned RGB volume sample",
            dtype: plan.output_dtype,
        });
    };
    let coordinates = [x, y, z];
    let outputs = [out_r, out_g, out_b];
    if plan.semantic != SemanticOp::VolumeTrilinearRgb
        || !params.is_valid()
        || params.dims != volume.dims()
        || params.storage != volume.storage()
        || plan.effects() != KernelEffects::READS_RESOURCE
        || !matches!(plan.result(), KernelResult::Tensors(_))
        || plan.result_count() != 3
        || plan.output_dtype != DType::F32
        || plan.shape.rank() != 2
        || (0..3).any(|index| {
            !typed_input_matches(plan, index, DType::F32, coordinates[index].len())
                || !typed_output_matches(plan, index, DType::F32, plan.shape, outputs[index].len())
                || !plan
                    .input(index)
                    .is_some_and(|desc| desc.layout.offset() == 0)
                || !plan
                    .result_tensor(index)
                    .is_some_and(|desc| desc.layout.offset() == 0)
        })
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned RGB volume sample",
        });
    }
    let shapes = plan.input_shapes.map(Option::unwrap);
    if shapes
        .iter()
        .any(|shape| shape.broadcast_with(&plan.shape) != Some(plan.shape))
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned RGB volume broadcast",
        });
    }
    let vector = match plan.implementation {
        CpuImplementation::Qdot => {
            return Err(CpuError::UnsupportedOp {
                op: "volume",
                dtype: plan.output_dtype,
            });
        }
        CpuImplementation::Scalar => false,
        CpuImplementation::PortableSimd => true,
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        CpuImplementation::Avx2 => {
            if let RgbVolumeBacking::Uf8(rgb) = volume.backing {
                sample_field::<true, _>(
                    &TierUf8(rgb),
                    SampleGeometry::new(volume.dims()),
                    coordinates,
                    shapes,
                    plan.shape,
                    outputs,
                );
                return Ok(());
            }
            return Err(CpuError::UnsupportedOp {
                op: "planned RGB volume implementation",
                dtype: DType::F32,
            });
        }
    };
    let geometry = SampleGeometry::new(volume.dims());
    match (volume.backing, vector) {
        (RgbVolumeBacking::F32(rgb), false) => {
            sample_field::<false, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
        (RgbVolumeBacking::F32(rgb), true) => {
            sample_field::<true, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
        (RgbVolumeBacking::Uf16(rgb), false) => {
            sample_field::<false, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
        (RgbVolumeBacking::Uf16(rgb), true) if planar_lanes(geometry, coordinates, outputs[0].len()) => {
            sample_uf16_planar_lanes(rgb, geometry, coordinates, outputs)
        }
        (RgbVolumeBacking::Uf16(rgb), true) => {
            sample_field::<true, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
        (RgbVolumeBacking::Uf8(rgb), false) => {
            sample_field::<false, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
        (RgbVolumeBacking::Uf8(rgb), true) => {
            sample_field::<true, _>(rgb, geometry, coordinates, shapes, plan.shape, outputs)
        }
    }
    Ok(())
}

fn sample_field<const VECTOR: bool, S: SampleRgb + ?Sized>(
    rgb: &S,
    geometry: SampleGeometry,
    coordinates: [&[f32]; 3],
    shapes: [Shape; 3],
    field: Shape,
    mut outputs: [&mut [f32]; 3],
) {
    let elements = outputs[0].len();
    if elements == 0 {
        return;
    }
    if coordinates.iter().all(|coordinate| coordinate.len() == 1) {
        let value = rgb.sample::<VECTOR>(geometry, coordinates.map(|coordinate| coordinate[0]));
        for (output, channel) in outputs.iter_mut().zip(value) {
            output.fill(channel);
        }
        return;
    }
    if coordinates
        .iter()
        .all(|coordinate| coordinate.len() == 1 || coordinate.len() == elements)
    {
        macro_rules! dense {
            ($x:literal, $y:literal, $z:literal) => {
                sample_scalar_dense::<VECTOR, $x, $y, $z, S>(rgb, geometry, coordinates, outputs)
            };
        }
        match (
            coordinates[0].len() == 1,
            coordinates[1].len() == 1,
            coordinates[2].len() == 1,
        ) {
            (false, false, false) => dense!(false, false, false),
            (false, false, true) => dense!(false, false, true),
            (false, true, false) => dense!(false, true, false),
            (false, true, true) => dense!(false, true, true),
            (true, false, false) => dense!(true, false, false),
            (true, false, true) => dense!(true, false, true),
            (true, true, false) => dense!(true, true, false),
            (true, true, true) => unreachable!("uniform coordinates handled above"),
        }
        return;
    }
    for index in 0..elements {
        let point = core::array::from_fn(|axis| {
            let source = broadcast_index(shapes[axis], field, index)
                .expect("volume coordinate broadcast was validated");
            *coordinates[axis]
                .get(source)
                .expect("volume coordinate length was validated")
        });
        let value = rgb.sample::<VECTOR>(geometry, point);
        for channel in 0..3 {
            outputs[channel][index] = value[channel];
        }
    }
}

fn sample_scalar_dense<
    const VECTOR: bool,
    const X: bool,
    const Y: bool,
    const Z: bool,
    S: SampleRgb + ?Sized,
>(
    rgb: &S,
    geometry: SampleGeometry,
    coordinates: [&[f32]; 3],
    outputs: [&mut [f32]; 3],
) {
    for index in 0..outputs[0].len() {
        let point = [
            coordinates[0][if X { 0 } else { index }],
            coordinates[1][if Y { 0 } else { index }],
            coordinates[2][if Z { 0 } else { index }],
        ];
        let value = rgb.sample::<VECTOR>(geometry, point);
        for channel in 0..3 {
            outputs[channel][index] = value[channel];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, select_cpu_kernel};
    use alloc::vec;
    use alloc::vec::Vec;
    use axnn::lowering::{KernelAccuracy, KernelPolicy, KernelRequest};
    use axnn::{Layout, TensorDesc, shape};

    fn request(
        storage: VolumeRgbStorage,
        dims: [u32; 3],
        shapes: [Shape; 3],
        field: Shape,
    ) -> KernelRequest {
        let inputs = shapes.map(|shape| TensorDesc::new(shape, DType::F32));
        let params = VolumeRgbParams::new(
            ResourceKey {
                group: 3,
                binding: 5,
            },
            dims,
            storage,
        )
        .unwrap();
        KernelRequest::volume_trilinear_rgb(&inputs[0], &inputs[1], &inputs[2], params, field)
            .unwrap()
    }

    fn plan(
        storage: VolumeRgbStorage,
        dims: [u32; 3],
        shapes: [Shape; 3],
        field: Shape,
        capabilities: CpuCapabilities,
    ) -> CpuKernelPlan {
        let mut policy =
            KernelPolicy::new(capabilities.mask()).with_minimum_accuracy(KernelAccuracy::Strict);
        policy.max_temporary_bytes = 0;
        select_cpu_kernel(&request(storage, dims, shapes, field), policy).unwrap()
    }

    // Independent fixed-order F32 oracle: no production stencil, sampler,
    // accumulator, or broadcast helper is reused here.
    fn oracle(rgb: &[[f32; 3]], dims: [u32; 3], point: [f32; 3]) -> [f32; 3] {
        if point.iter().any(|coordinate| coordinate.is_nan()) {
            return [f32::NAN; 3];
        }
        let mut lower = [0; 3];
        let mut upper = [0; 3];
        let mut fraction = [0.0_f32; 3];
        for axis in 0..3 {
            if dims[axis] > 1 {
                let end = (dims[axis] - 1) as f32;
                let q = if point[axis] < 0.0 {
                    0.0
                } else if point[axis] > end {
                    end
                } else {
                    point[axis]
                };
                lower[axis] = q as usize;
                upper[axis] = (lower[axis] + 1).min(dims[axis] as usize - 1);
                fraction[axis] = q - lower[axis] as f32;
            }
        }
        let mut output = [0.0_f32; 3];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let px = if x == 0 { lower[0] } else { upper[0] };
                    let py = if y == 0 { lower[1] } else { upper[1] };
                    let pz = if z == 0 { lower[2] } else { upper[2] };
                    let wx = if x == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    };
                    let wy = if y == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    };
                    let wz = if z == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    };
                    let weight = (wx * wy) * wz;
                    let value = rgb[px + dims[0] as usize * (py + dims[1] as usize * pz)];
                    for channel in 0..3 {
                        let product = value[channel] * weight;
                        output[channel] += product;
                    }
                }
            }
        }
        output
    }

    fn decode_uf16(bits: u16) -> f32 {
        let exponent = u32::from(bits >> 11);
        let fraction = u32::from(bits & 0x7ff);
        if exponent == 0 {
            fraction as f32 * f32::from_bits(102 << 23) // fraction * 2^-25
        } else {
            f32::from_bits(((exponent + 112) << 23) | (fraction << 12))
        }
    }

    fn run(
        plan: &CpuKernelPlan,
        volume: RgbVolumeRef<'_>,
        coordinates: [&[f32]; 3],
    ) -> [Vec<f32>; 3] {
        let mut outputs = core::array::from_fn(|_| vec![-123.0; plan.shape.num_elements()]);
        let [red, green, blue] = &mut outputs;
        invoke_volume_trilinear_rgb(
            plan,
            volume,
            coordinates[0],
            coordinates[1],
            coordinates[2],
            red,
            green,
            blue,
        )
        .unwrap();
        outputs
    }

    fn assert_sample(actual: [f32; 3], expected: [f32; 3]) {
        for channel in 0..3 {
            if expected[channel].is_nan() {
                assert!(actual[channel].is_nan(), "channel {channel}: {actual:?}");
            } else {
                assert_eq!(
                    actual[channel].to_bits(),
                    expected[channel].to_bits(),
                    "channel {channel}: {actual:?} vs {expected:?}"
                );
            }
        }
    }

    fn random(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    #[test]
    fn borrowed_volumes_validate_storage_values_and_lengths_without_copying() {
        let rgb = [[-0.0, 1.0, f32::MAX]; 6];
        let view = RgbVolumeRef::new_f32(&rgb, [3, 1, 2]).unwrap();
        assert_eq!(view.dims(), [3, 1, 2]);
        assert_eq!(view.storage(), VolumeRgbStorage::F32);
        assert_eq!(view.as_f32().unwrap().as_ptr(), rgb.as_ptr());
        assert!(view.as_uf16().is_none());
        let RgbVolumeBacking::F32(borrowed) = view.backing else {
            panic!()
        };
        assert_eq!(borrowed.as_ptr(), rgb.as_ptr());
        let words = [[0, 1, 0xf7ff]; 6];
        let view = RgbVolumeRef::new_uf16(&words, [3, 1, 2]).unwrap();
        assert_eq!(view.storage(), VolumeRgbStorage::Uf16E5M11);
        assert_eq!(view.as_uf16().unwrap().as_ptr(), words.as_ptr());
        assert!(view.as_f32().is_none());
        let RgbVolumeBacking::Uf16(borrowed) = view.backing else {
            panic!()
        };
        assert_eq!(borrowed.as_ptr(), words.as_ptr());
        assert!(matches!(
            RgbVolumeRef::new_f32(&rgb[..5], [3, 1, 2]),
            Err(CpuError::SizeMismatch {
                expected: 72,
                actual: 60
            })
        ));
        assert!(matches!(
            RgbVolumeRef::new_f32(&rgb, [1; 3]),
            Err(CpuError::SizeMismatch {
                expected: 12,
                actual: 72
            })
        ));
        assert!(matches!(
            RgbVolumeRef::new_uf16(&words[..5], [3, 1, 2]),
            Err(CpuError::SizeMismatch {
                expected: 36,
                actual: 30
            })
        ));
        for dims in [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [16_777_218, 1, 1],
            [u32::MAX; 3],
            [16_777_217; 3],
        ] {
            assert!(matches!(
                RgbVolumeRef::new_f32(&[], dims),
                Err(CpuError::ShapeMismatch { .. })
            ));
            assert!(matches!(
                RgbVolumeRef::new_uf16(&[], dims),
                Err(CpuError::ShapeMismatch { .. })
            ));
        }
        assert!(matches!(
            RgbVolumeRef::new_f32(&[], [16_777_217, 1, 1]),
            Err(CpuError::SizeMismatch { .. })
        ));
        for bad in [
            -f32::from_bits(1),
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            for channel in 0..3 {
                let mut invalid = rgb;
                invalid[5][channel] = bad;
                assert!(matches!(
                    RgbVolumeRef::new_f32(&invalid, [3, 1, 2]),
                    Err(CpuError::ShapeMismatch { .. })
                ));
            }
        }
        for bits in 0xf800..=0xffff_u32 {
            assert!(RgbVolumeRef::new_uf16(&[[0, 1, bits as u16]], [1; 3]).is_err());
        }
    }

    #[test]
    fn f32_sampling_matches_independent_bits_across_random_fields_and_edges() {
        let mut state = 0x3217_b459;
        for dims in [[1, 1, 1], [2, 3, 4], [5, 1, 3], [1, 4, 2], [3, 2, 1]] {
            let count = dims.iter().product::<u32>() as usize;
            for iteration in 0..8 {
                let rgb: Vec<[f32; 3]> = (0..count)
                    .map(|index| {
                        core::array::from_fn(|channel| match (index + channel + iteration) % 4 {
                            0 => f32::from_bits(random(&mut state) % 0x0080_0000),
                            1 => f32::from_bits((random(&mut state) & 0x7fff_ffff) % 0x7f80_0000),
                            2 => (random(&mut state) % 4096) as f32 / 32.0,
                            _ => -0.0,
                        })
                    })
                    .collect();
                let mut points = vec![
                    [0.0; 3],
                    [-0.0; 3],
                    [f32::INFINITY; 3],
                    [f32::NEG_INFINITY; 3],
                    [f32::MAX, -f32::MAX, 0.5],
                    [f32::NAN, 0.0, 0.0],
                    [0.0, f32::from_bits(0x7f80_0001), 0.0],
                    [0.0, 0.0, f32::from_bits(0xffc1_2345)],
                    dims.map(|dimension| (dimension - 1) as f32),
                    dims.map(|dimension| {
                        f32::from_bits(((dimension - 1) as f32).to_bits().saturating_sub(1))
                    }),
                    [0.5, 0.25, 0.75],
                ];
                for _ in 0..129 {
                    points.push(core::array::from_fn(|axis| {
                        (random(&mut state) & 0xffff) as f32 / 65536.0 * (dims[axis] + 4) as f32
                            - 2.0
                    }));
                }
                let coordinates: [Vec<f32>; 3] =
                    core::array::from_fn(|axis| points.iter().map(|point| point[axis]).collect());
                let field = shape![1, points.len()];
                let volume = RgbVolumeRef::new_f32(&rgb, dims).unwrap();
                for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
                    let plan = plan(VolumeRgbStorage::F32, dims, [field; 3], field, capabilities);
                    let actual = run(&plan, volume, coordinates.each_ref().map(Vec::as_slice));
                    for (index, point) in points.iter().enumerate() {
                        assert_sample(
                            core::array::from_fn(|channel| actual[channel][index]),
                            oracle(&rgb, dims, *point),
                        );
                    }
                }
            }
        }
        // All integer sampling endpoints remain exact at the admitted limit.
        // A one-slice volume accumulates the z = 0 prefix only.
        let geometry = SampleGeometry::new([16_777_217, 1, 1]);
        assert_eq!(geometry.corners, 4);
        let (indices, weights) = geometry
            .stencil([f32::INFINITY, -f32::INFINITY, 0.5])
            .unwrap();
        assert_eq!(indices[..4], [16_777_216; 4]);
        assert_eq!(weights[..4], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(SampleGeometry::new([2, 2, 2]).corners, 8);
    }

    #[test]
    fn volume_sampling_keeps_separate_rounding_and_subnormal_products() {
        let tiny = f32::from_bits(1);
        let mut rgb = [[0.0, tiny, f32::MAX]; 8];
        rgb[0][0] = 8_388_611.0;
        rgb[1][0] = 1.0 + f32::EPSILON;
        let points = [
            [1.0 - f32::EPSILON, 0.0, 0.0],
            [0.5; 3],
            [0.0; 3],
            [f32::INFINITY; 3],
        ];
        let expected = points.map(|point| oracle(&rgb, [2; 3], point));
        // The first contribution is 1+3*EPSILON. The next product rounds to
        // 1 before addition, giving 0x40000002; a fused multiply/add would
        // instead give 0x40000001. This also checks the oracle independently
        // of any compiler decision shared with the production sampler.
        assert_eq!(expected[0][0].to_bits(), 0x4000_0002);
        // Eight individually rounded tiny/8 products underflow to zero.
        // Replacing the corner sum with a constant-volume shortcut is wrong.
        assert_eq!(expected[1][1].to_bits(), 0);
        assert_eq!(expected[2][1].to_bits(), 1);
        let coordinates: [[f32; 4]; 3] =
            core::array::from_fn(|axis| points.map(|point| point[axis]));
        let field = shape![1, 4];
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            let plan = plan(
                VolumeRgbStorage::F32,
                [2; 3],
                [field; 3],
                field,
                capabilities,
            );
            let actual = run(
                &plan,
                RgbVolumeRef::new_f32(&rgb, [2; 3]).unwrap(),
                coordinates.each_ref().map(|values| &values[..]),
            );
            for index in 0..4 {
                assert_sample(
                    core::array::from_fn(|channel| actual[channel][index]),
                    expected[index],
                );
            }
        }
    }

    #[test]
    fn uf16_sampling_matches_independent_decode_for_all_finite_words() {
        let field = shape![1, 4];
        let scalar = plan(
            VolumeRgbStorage::Uf16E5M11,
            [1; 3],
            [Shape::SCALAR; 3],
            field,
            CpuCapabilities::SCALAR,
        );
        let vector = plan(
            VolumeRgbStorage::Uf16E5M11,
            [1; 3],
            [Shape::SCALAR; 3],
            field,
            CpuCapabilities::PORTABLE_SIMD,
        );
        for bits in 0..=0xf7ff_u16 {
            let words = [[bits, 0xf7ff - bits, bits ^ 1]];
            let volume = RgbVolumeRef::new_uf16(&words, [1; 3]).unwrap();
            let expected = words[0].map(decode_uf16);
            for plan in [&scalar, &vector] {
                let mut red = [0.0; 4];
                let mut green = [0.0; 4];
                let mut blue = [0.0; 4];
                invoke_volume_trilinear_rgb(
                    plan,
                    volume,
                    &[f32::NEG_INFINITY],
                    &[f32::INFINITY],
                    &[-0.0],
                    &mut red,
                    &mut green,
                    &mut blue,
                )
                .unwrap();
                for index in 0..4 {
                    assert_sample([red[index], green[index], blue[index]], expected);
                }
            }
        }
        let mut state = 0x6172_8cb9;
        for dims in [[2, 3, 4], [3, 1, 5], [1, 4, 2]] {
            let words: Vec<[u16; 3]> = (0..dims.iter().product::<u32>())
                .map(|_| core::array::from_fn(|_| (random(&mut state) % 0xf800) as u16))
                .collect();
            let decoded: Vec<_> = words.iter().map(|value| value.map(decode_uf16)).collect();
            let mut points: Vec<[f32; 3]> = (0..257)
                .map(|_| {
                    core::array::from_fn(|axis| {
                        (random(&mut state) & 0xffff) as f32 / 65536.0 * (dims[axis] + 2) as f32
                            - 1.0
                    })
                })
                .collect();
            points.extend_from_slice(&[
                [f32::NAN, 0.0, 0.0],
                [0.0, f32::NAN, 0.0],
                [0.0, 0.0, f32::NAN],
                [f32::INFINITY; 3],
                [f32::NEG_INFINITY; 3],
            ]);
            let coordinates: [Vec<f32>; 3] =
                core::array::from_fn(|axis| points.iter().map(|point| point[axis]).collect());
            let field = shape![1, points.len()];
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
                let plan = plan(
                    VolumeRgbStorage::Uf16E5M11,
                    dims,
                    [field; 3],
                    field,
                    capabilities,
                );
                let actual = run(
                    &plan,
                    RgbVolumeRef::new_uf16(&words, dims).unwrap(),
                    coordinates.each_ref().map(Vec::as_slice),
                );
                for (index, point) in points.iter().enumerate() {
                    assert_sample(
                        core::array::from_fn(|channel| actual[channel][index]),
                        oracle(&decoded, dims, *point),
                    );
                }
            }
        }
    }

    #[test]
    fn planar_uf16_lanes_match_per_sample_bits_and_the_oracle() {
        let mut state = 0x51a7_e0d3;
        for dims in [[1, 1, 1], [7, 5, 1], [33, 1, 1], [1, 9, 1], [64, 48, 1]] {
            let count = (dims[0] * dims[1]) as usize;
            // Every finite UF16 word class: zero, subnormal, normal, top.
            let rgb: Vec<[u16; 3]> = (0..count)
                .map(|_| core::array::from_fn(|channel| match random(&mut state) % 5 {
                    0 => 0,
                    1 => (random(&mut state) % 0x800) as u16,
                    2 => (random(&mut state) % 0xf800) as u16,
                    3 => 0xf7ff,
                    _ => (random(&mut state) % 0x7c00) as u16 + channel as u16,
                }))
                .collect();
            let volume = RgbVolumeRef::new_uf16(&rgb, dims).unwrap();
            let geometry = SampleGeometry::new(dims);
            // Lengths around the four-lane boundary, edges, NaN and infinities.
            for length in [1, 3, 4, 5, 8, 11, 37] {
                let point = |i: usize, state: &mut u32| -> [f32; 3] {
                    match i % 9 {
                        0 => [f32::NAN, 1.0, 0.0],
                        1 => [0.5, 0.25, f32::NAN],
                        2 => [f32::INFINITY, f32::NEG_INFINITY, 0.0],
                        3 => [-0.0, -0.0, 0.0],
                        4 => [(dims[0] - 1) as f32, (dims[1] - 1) as f32, 0.0],
                        _ => core::array::from_fn(|axis| {
                            (random(state) & 0xffff) as f32 / 65536.0 * (dims[axis] + 4) as f32 - 2.0
                        }),
                    }
                };
                let points: Vec<[f32; 3]> = (0..length).map(|i| point(i, &mut state)).collect();
                let coordinates: [Vec<f32>; 3] = core::array::from_fn(|axis| points.iter().map(|p| p[axis]).collect());
                let mut planes = [vec![0.0; length], vec![0.0; length], vec![0.0; length]];
                let [red, green, blue] = &mut planes;
                let coordinate_refs = coordinates.each_ref().map(Vec::as_slice);
                assert!(planar_lanes(geometry, coordinate_refs, length));
                sample_uf16_planar_lanes(volume.as_uf16().unwrap(), geometry, coordinate_refs, [red, green, blue]);
                let expanded: Vec<[f32; 3]> = rgb.iter().map(|c| c.map(decode_uf16)).collect();
                for (index, point) in points.iter().enumerate() {
                    let actual = core::array::from_fn(|channel| planes[channel][index]);
                    let per_sample = sample_uf16_direct::<true>(volume.as_uf16().unwrap(), geometry, *point);
                    for channel in 0..3 {
                        assert_eq!(actual[channel].to_bits(), per_sample[channel].to_bits(), "{dims:?} {point:?}");
                    }
                    assert_sample(actual, oracle(&expanded, dims, *point));
                }
            }
        }
        // A slice past u32 indices keeps the per-sample path.
        let huge = SampleGeometry::new([16_777_217, 300, 1]);
        assert!(!planar_lanes(huge, [&[0.0; 4], &[0.0; 4], &[0.0]], 4));
    }

    #[test]
    fn uf8_volume_validates_storage_and_preserves_all_finite_encodings() {
        let field = shape![1, 4];
        let points = [
            [0.13, 0.79, 0.37],
            [0.0; 3],
            [f32::INFINITY; 3],
            [f32::NAN; 3],
        ];
        let coordinates: [[f32; 4]; 3] = core::array::from_fn(|a| points.map(|p| p[a]));
        for first in 0..=0xf7_u8 {
            let bytes: [[u8; 3]; 8] = core::array::from_fn(|i| {
                core::array::from_fn(|c| ((first as usize + i * 7 + c * 11) % 248) as u8)
            });
            let rgb = bytes.map(|p| p.map(|b| unsigned_float::Uf8E5M3::from_bits(b).to_f32()));
            let volume = RgbVolumeRef::new_uf8e5m3(&bytes, [2; 3]).unwrap();
            assert_eq!(volume.as_uf8e5m3(), Some(bytes.as_slice()));
            assert!(volume.as_f32().is_none() && volume.as_uf16().is_none());
            for capabilities in [
                CpuCapabilities::SCALAR,
                CpuCapabilities::PORTABLE_SIMD,
                CpuCapabilities::compiled(),
            ] {
                let plan = plan(
                    VolumeRgbStorage::Uf8E5M3,
                    [2; 3],
                    [field; 3],
                    field,
                    capabilities,
                );
                if capabilities == CpuCapabilities::SCALAR {
                    assert_eq!(plan.implementation, CpuImplementation::Scalar);
                } else if capabilities == CpuCapabilities::PORTABLE_SIMD {
                    assert_eq!(plan.implementation, CpuImplementation::PortableSimd);
                }
                #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
                if capabilities == CpuCapabilities::compiled() {
                    assert_eq!(plan.implementation, CpuImplementation::Avx2);
                }
                let actual = run(&plan, volume, coordinates.each_ref().map(|p| &p[..]));
                for i in 0..4 {
                    assert_sample(
                        core::array::from_fn(|c| actual[c][i]),
                        oracle(&rgb, [2; 3], points[i]),
                    );
                }
            }
        }
        for invalid in 0xf8..=0xff_u8 {
            assert!(RgbVolumeRef::new_uf8e5m3(&[[invalid; 3]], [1; 3]).is_err());
        }
        assert!(RgbVolumeRef::new_uf8e5m3(&[[0; 3]], [2; 3]).is_err());
        assert!(RgbVolumeRef::new_uf8e5m3(&[], [0; 3]).is_err());
        let mismatched = plan(
            VolumeRgbStorage::Uf16E5M11,
            [1; 3],
            [field; 3],
            field,
            CpuCapabilities::PORTABLE_SIMD,
        );
        let mut out = [[-9.0; 4]; 3];
        let [r, g, b] = &mut out;
        assert!(
            invoke_volume_trilinear_rgb(
                &mismatched,
                RgbVolumeRef::new_uf8e5m3(&[[0; 3]], [1; 3]).unwrap(),
                &[0.0; 4],
                &[0.0; 4],
                &[0.0; 4],
                r,
                g,
                b
            )
            .is_err()
        );
        assert_eq!(out, [[-9.0; 4]; 3]);
    }

    #[test]
    fn volume_sampling_preserves_scalar_dense_row_and_column_broadcasts() {
        let dims = [3, 2, 4];
        let words: Vec<[u16; 3]> = (0..24)
            .map(|index| [0x3800 + index * 7, 0x4000 + index * 11, 0x4400 + index * 13])
            .collect();
        let decoded: Vec<_> = words.iter().map(|value| value.map(decode_uf16)).collect();
        let volumes = [
            RgbVolumeRef::new_f32(&decoded, dims).unwrap(),
            RgbVolumeRef::new_uf16(&words, dims).unwrap(),
        ];
        let field = shape![3, 5];
        for volume in volumes {
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
                for mask in 0..8 {
                    let shapes = core::array::from_fn(|axis| {
                        if mask & (1 << axis) != 0 {
                            Shape::SCALAR
                        } else {
                            field
                        }
                    });
                    let coordinates: [Vec<f32>; 3] = core::array::from_fn(|axis| {
                        (0..shapes[axis].num_elements())
                            .map(|index| axis as f32 * 0.25 + index as f32 * 0.125)
                            .collect()
                    });
                    let plan = plan(volume.storage(), dims, shapes, field, capabilities);
                    let actual = run(&plan, volume, coordinates.each_ref().map(Vec::as_slice));
                    for index in 0..15 {
                        let point = core::array::from_fn(|axis| {
                            coordinates[axis][if mask & (1 << axis) != 0 { 0 } else { index }]
                        });
                        assert_sample(
                            core::array::from_fn(|channel| actual[channel][index]),
                            oracle(&decoded, dims, point),
                        );
                    }
                }
                let row = [-1.0, 0.25, 0.75, 1.25, f32::INFINITY];
                let column = [0.125, 0.75, f32::NAN];
                let dense: Vec<_> = (0..15).map(|index| index as f32 * 0.125).collect();
                for row_shape in [shape![5], shape![1, 5]] {
                    let shapes = [row_shape, shape![3, 1], field];
                    let plan = plan(volume.storage(), dims, shapes, field, capabilities);
                    let actual = run(&plan, volume, [&row, &column, &dense]);
                    for index in 0..15 {
                        assert_sample(
                            core::array::from_fn(|channel| actual[channel][index]),
                            oracle(
                                &decoded,
                                dims,
                                [row[index % 5], column[index / 5], dense[index]],
                            ),
                        );
                    }
                }
                let empty = shape![0, 5];
                let plan = plan(volume.storage(), dims, [empty; 3], empty, capabilities);
                let actual = run(&plan, volume, [&[], &[], &[]]);
                assert!(actual.iter().all(Vec::is_empty));
            }
        }
    }

    fn rejects_without_writing(
        plan: &CpuKernelPlan,
        volume: RgbVolumeRef<'_>,
        coordinates: [&[f32]; 3],
        lengths: [usize; 3],
    ) {
        let mut outputs: [Vec<f32>; 3] = core::array::from_fn(|channel| {
            vec![f32::from_bits(0x7fc1_2340 + channel as u32); lengths[channel]]
        });
        let before = outputs.each_ref().map(|values| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        });
        let [red, green, blue] = &mut outputs;
        assert!(
            invoke_volume_trilinear_rgb(
                plan,
                volume,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                red,
                green,
                blue
            )
            .is_err()
        );
        assert_eq!(
            outputs.each_ref().map(|values| values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()),
            before
        );
    }

    #[test]
    fn volume_preflight_rejects_mismatches_without_touching_any_output() {
        let rgb = [[1.0, 2.0, 3.0]; 8];
        let words = [[0x3c00, 0x4000, 0x4400]; 8];
        let volume = RgbVolumeRef::new_f32(&rgb, [2; 3]).unwrap();
        let field = shape![1, 7];
        let plan = plan(
            VolumeRgbStorage::F32,
            [2; 3],
            [field; 3],
            field,
            CpuCapabilities::PORTABLE_SIMD,
        );
        let coordinates = [0.5; 7];
        for axis in 0..3 {
            for length in [0, 6, 8] {
                let mut lengths = [7; 3];
                lengths[axis] = length;
                rejects_without_writing(&plan, volume, [&coordinates; 3], lengths);
            }
            let mut inputs: [&[f32]; 3] = [&coordinates; 3];
            inputs[axis] = &coordinates[..6];
            rejects_without_writing(&plan, volume, inputs, [7; 3]);
            inputs[axis] = &[0.5; 8];
            rejects_without_writing(&plan, volume, inputs, [7; 3]);
        }
        rejects_without_writing(
            &plan,
            RgbVolumeRef::new_f32(&rgb, [4, 2, 1]).unwrap(),
            [&coordinates; 3],
            [7; 3],
        );
        rejects_without_writing(
            &plan,
            RgbVolumeRef::new_uf16(&words, [2; 3]).unwrap(),
            [&coordinates; 3],
            [7; 3],
        );
        for variant in 0..5 {
            let mut invalid = plan;
            match variant {
                0 => invalid.semantic = SemanticOp::Identity,
                1 => invalid.output_dtype = DType::I32,
                2 => invalid.input_dtypes[1] = Some(DType::U8),
                3 => invalid.input_shapes[2] = Some(Shape::SCALAR),
                _ => invalid.shape = shape![7],
            }
            rejects_without_writing(&invalid, volume, [&coordinates; 3], [7; 3]);
        }
        let identity_request = KernelRequest::identity(&TensorDesc::new(field, DType::F32));
        let identity = select_cpu_kernel(
            &identity_request,
            KernelPolicy::new(CpuCapabilities::SCALAR.mask()),
        )
        .unwrap();
        rejects_without_writing(&identity, volume, [&coordinates; 3], [7; 3]);
    }

    #[test]
    fn volume_catalog_rejects_strided_and_offset_coordinate_descriptors() {
        let field = shape![3, 5];
        let plain = TensorDesc::new(field, DType::F32);
        let params = VolumeRgbParams::new(
            ResourceKey {
                group: 0,
                binding: 0,
            },
            [2; 3],
            VolumeRgbStorage::F32,
        )
        .unwrap();
        for layout in [Layout::from_raw(&[40, 8], 0), Layout::from_raw(&[20, 4], 4)] {
            for axis in 0..3 {
                let mut inputs = [plain; 3];
                inputs[axis].layout = layout;
                let request = KernelRequest::volume_trilinear_rgb(
                    &inputs[0], &inputs[1], &inputs[2], params, field,
                )
                .unwrap();
                assert!(
                    select_cpu_kernel(
                        &request,
                        KernelPolicy::new(CpuCapabilities::compiled().mask())
                    )
                    .is_none()
                );
            }
        }
    }

    #[test]
    fn volume_subslices_and_fresh_borrows_preserve_guards_and_observe_edits() {
        let dims = [2; 3];
        let field = shape![2, 5];
        let shapes = [field, Shape::SCALAR, field];
        let mut rgb = [[0.25, 0.5, 0.75]; 10];
        let x = [
            -9.0,
            -1.0,
            0.0,
            0.125,
            0.25,
            0.5,
            0.75,
            0.875,
            1.0,
            2.0,
            f32::INFINITY,
            -9.0,
        ];
        let y = [-9.0, 0.75, -9.0];
        let z = [
            -9.0, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0, -0.0, -1.0, 1.0, 0.5, -9.0,
        ];
        let coordinates = [&x[1..11], &y[1..2], &z[1..11]];
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            let plan = plan(VolumeRgbStorage::F32, dims, shapes, field, capabilities);
            for edit in 0..3 {
                rgb[8] = [edit as f32 + 1.0, 3.0, 7.0];
                let volume = RgbVolumeRef::new_f32(&rgb[1..9], dims).unwrap();
                let mut outputs = [[-123.0_f32; 12]; 3];
                let [red, green, blue] = &mut outputs;
                invoke_volume_trilinear_rgb(
                    &plan,
                    volume,
                    coordinates[0],
                    coordinates[1],
                    coordinates[2],
                    &mut red[1..11],
                    &mut green[1..11],
                    &mut blue[1..11],
                )
                .unwrap();
                for channel in &outputs {
                    assert_eq!(channel[0], -123.0);
                    assert_eq!(channel[11], -123.0);
                }
                for index in 0..10 {
                    assert_sample(
                        core::array::from_fn(|channel| outputs[channel][index + 1]),
                        oracle(
                            &rgb[1..9],
                            dims,
                            [
                                coordinates[0][index],
                                coordinates[1][0],
                                coordinates[2][index],
                            ],
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn volume_catalog_admits_strict_typed_outputs_without_scratch() {
        for storage in [VolumeRgbStorage::F32, VolumeRgbStorage::Uf16E5M11] {
            let field = shape![3, 5];
            let shapes = [Shape::SCALAR, shape![3, 1], shape![1, 5]];
            for (capabilities, implementation) in [
                (CpuCapabilities::SCALAR, CpuImplementation::Scalar),
                (
                    CpuCapabilities::PORTABLE_SIMD,
                    CpuImplementation::PortableSimd,
                ),
            ] {
                let plan = plan(storage, [2, 3, 4], shapes, field, capabilities);
                assert_eq!(plan.implementation, implementation);
                assert_eq!(plan.accuracy, KernelAccuracy::Strict);
                assert_eq!(plan.temporary_bytes, 0);
                assert_eq!(plan.effects(), KernelEffects::READS_RESOURCE);
                assert_eq!(plan.result_count(), 3);
                for index in 0..3 {
                    assert_eq!(
                        *plan.result_tensor(index).unwrap(),
                        TensorDesc::new(field, DType::F32)
                    );
                }
                assert!(plan.result_tensor(3).is_none());
            }
        }
    }
}
