//! General F32 Xe vector operations over caller-owned accelerator buffers.
//!
//! The package carries only immutable program bytes. A caller retains its
//! imported buffer handles and binds offsets/ranges through the existing
//! accelerator submission API; no tensor allocation or staging occurs here.

use core::num::NonZeroU32;

use kore_driver::accel::{LaunchShape, Workgroups};
use virtio_accel_core::BackendError;

use crate::{XeKernelArtifact, artifact};

// The captured shader's fixed local_size_x. Changing this requires recapturing
// every native package and updating the checked manifests before launch.
const F32_MAP_LOCAL_SIZE: u32 = 256;

/// One precompiled F32 map operation. The Xe provider selects its generation
/// from the checked family at program load; the caller never selects Xe2/Xe3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32Map {
    /// `out[i] = lhs[i] + rhs[i]`.
    Add,
    /// `out[i] = lhs[i] * rhs[i]`.
    Multiply,
    /// `out[i] = input[i] / (1 + exp(-input[i]))`.
    Silu,
    /// `out[i] = fma(input[i], scale[i], bias[i])`.
    /// Useful as a fused per-element affine epilogue.
    Affine,
    /// PyTorch's `softplus` (beta 1, threshold 20): the input above 20, else
    /// `log1p(exp(input[i]))` (the delta rule's step size). Xe3 only.
    Softplus,
    /// `out[i] = 1 / (1 + exp(-input[i]))` (the gated attention's output
    /// gate). Xe3 only.
    Sigmoid,
}

impl F32Map {
    pub const ALL: [Self; 6] = [
        Self::Add,
        Self::Multiply,
        Self::Silu,
        Self::Affine,
        Self::Softplus,
        Self::Sigmoid,
    ];
}

/// A nonempty F32 vector. The same value supplies the four-byte parameter
/// mapping and the explicit workgroup count, so a ragged final group is masked
/// by the kernel without padding or copying the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32Elements(NonZeroU32);

impl F32Elements {
    /// Zero-length launches are invalid in the accelerator contract.
    pub const fn new(count: u32) -> Option<Self> {
        match NonZeroU32::new(count) {
            Some(count) => Some(Self(count)),
            None => None,
        }
    }

    /// Number of F32 elements read and written by this submission.
    pub const fn count(self) -> u32 {
        self.0.get()
    }

    /// Minimum byte length of each F32 data binding.
    pub const fn data_bytes(self) -> u64 {
        self.count() as u64 * 4
    }

    /// The caller writes these four bytes into a held, read-only parameter
    /// buffer before submitting. This is control data, never tensor staging.
    pub const fn parameter_bytes(self) -> [u8; 4] {
        self.count().to_le_bytes()
    }

    /// The artifact's explicit launch geometry, including the final masked
    /// workgroup. The Xe backend still validates its per-device dispatch limits.
    pub fn launch(self) -> Result<LaunchShape, BackendError> {
        let groups = 1 + (self.count() - 1) / F32_MAP_LOCAL_SIZE;
        Ok(LaunchShape::Workgroups(Workgroups::new(
            [groups, 1, 1],
            [F32_MAP_LOCAL_SIZE, 1, 1],
        )?))
    }
}

/// Select a precompiled XVE artifact for a general F32 map.
///
/// Bind caller-owned F32 buffers with at least `F32Elements::data_bytes()`:
///
/// - Add/Multiply: slot 0 lhs (read), 1 rhs (read), 2 count parameter (read,
///   four bytes), 3 output (write).
/// - SiLU, Softplus, Sigmoid: slot 0 input (read), 1 count parameter (read),
///   2 output (write).
/// - Affine: slot 0 input (read), 1 scale (read), 2 bias (read), 3 count
///   parameter (read), 4 output (write).
///
/// Use `F32Elements::launch()` for submission. Exact tensor ranges can be
/// reused from imported shared DMA buffers; the parameter is a four-byte
/// control mapping. The family contains independently captured and checked
/// Xe2 and Xe3 native ISA (Softplus and Sigmoid Xe3 only, refused at load on
/// Xe2); the provider selects the matching generation.
pub fn f32_map(op: F32Map) -> XeKernelArtifact {
    match op {
        F32Map::Add => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-add.kxefamily"
        )),
        F32Map::Multiply => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-mul.kxefamily"
        )),
        F32Map::Silu => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-silu.kxefamily"
        )),
        F32Map::Affine => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-affine.kxefamily"
        )),
        F32Map::Softplus => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-softplus.kxefamily"
        )),
        F32Map::Sigmoid => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-sigmoid.kxefamily"
        )),
    }
}

/// Epilogue of an F32 `[M,K] * [N,K]^T` projection, without weight or
/// activation quantization. Reduction order is not a bit-exact CPU contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32Projection {
    Linear,
    Silu,
    Sigmoid,
}

impl F32Projection {
    pub const ALL: [Self; 3] = [Self::Linear, Self::Silu, Self::Sigmoid];
}

/// Checked runtime dimensions for `xe-xve-f32-projection.comp`. The shader
/// owns one output per workgroup and uses 32-bit indexing. Bounding every
/// matrix's byte extent also prevents the final 32-lane K increment wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32ProjectionShape {
    rows: u32,
    columns: u32,
    inner: u32,
}

impl F32ProjectionShape {
    pub fn new(rows: u32, columns: u32, inner: u32) -> Option<Self> {
        if rows == 0 || columns == 0 || inner == 0 {
            return None;
        }
        for (a, b) in [(rows, inner), (columns, inner), (rows, columns)] {
            a.checked_mul(b)?
                .checked_mul(core::mem::size_of::<f32>() as u32)?;
        }
        Some(Self {
            rows,
            columns,
            inner,
        })
    }

    /// Input, transposed weight, and output byte extents, respectively.
    pub fn data_bytes(self) -> [u64; 3] {
        let [m, n, k] = [self.rows, self.columns, self.inner].map(u64::from);
        [m * k * 4, n * k * 4, m * n * 4]
    }

    /// Read-only `{F32 scale, u32 K}` control. M and N come from the exact
    /// workgroup counts, so the control and dispatch describe one shape.
    pub fn parameter_bytes(self, scale: f32) -> Option<[u8; 8]> {
        if !scale.is_finite() {
            return None;
        }
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&scale.to_le_bytes());
        bytes[4..].copy_from_slice(&self.inner.to_le_bytes());
        Some(bytes)
    }

    pub fn launch(self) -> Result<LaunchShape, BackendError> {
        // Captured shader local_size_x; changing it requires recapture.
        const LANES: u32 = 32;
        Ok(LaunchShape::Workgroups(Workgroups::new(
            [self.columns, self.rows, 1],
            [LANES, 1, 1],
        )?))
    }
}

/// General F32 projection, currently captured on Xe2. Other generations
/// refuse the family at admission and retain the ordinary CPU fallback.
/// Bind slots 0 activations (read), 1 transposed weights (read), 2 the eight
/// control bytes (read), and 3 output (write), using `F32ProjectionShape`.
pub fn f32_projection(op: F32Projection) -> XeKernelArtifact {
    match op {
        F32Projection::Linear => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-projection-linear.kxefamily"
        )),
        F32Projection::Silu => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-projection-silu.kxefamily"
        )),
        F32Projection::Sigmoid => artifact(include_bytes!(
            "../../../drivers/xe/kernels/anv-xve-f32-projection-sigmoid.kxefamily"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_extents_control_and_grid_agree_for_ragged_shapes() {
        for (m, n, k) in [(1, 1, 1), (3, 17, 33), (8, 128, 2688), (311, 128, 2688)] {
            let shape = F32ProjectionShape::new(m, n, k).unwrap();
            assert_eq!(
                shape.data_bytes(),
                [
                    m as u64 * k as u64 * 4,
                    n as u64 * k as u64 * 4,
                    m as u64 * n as u64 * 4
                ]
            );
            let control = shape.parameter_bytes(0.5).unwrap();
            assert_eq!(&control[..4], &0.5f32.to_le_bytes());
            assert_eq!(&control[4..], &k.to_le_bytes());
            assert_eq!(
                shape.launch().unwrap(),
                LaunchShape::Workgroups(Workgroups::new([n, m, 1], [32, 1, 1]).unwrap())
            );
            assert!(shape.parameter_bytes(f32::NAN).is_none());
            assert!(shape.parameter_bytes(f32::INFINITY).is_none());
        }
    }

    #[test]
    fn projection_refuses_empty_and_overflowing_bindings() {
        for shape in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (u32::MAX, 1, 1),
            (1, u32::MAX, 1),
            (1, 1, u32::MAX),
            (65536, 65536, 1),
            (65536, 1, 65536),
            (1, 65536, 65536),
        ] {
            assert!(F32ProjectionShape::new(shape.0, shape.1, shape.2).is_none());
        }
    }

    #[test]
    fn vector_extent_keeps_count_and_launch_in_sync_at_tile_edges() {
        assert_eq!(F32Elements::new(0), None);
        for (count, groups) in [(1, 1), (256, 1), (257, 2), (513, 3), (u32::MAX, 1 << 24)] {
            let extent = F32Elements::new(count).unwrap();
            assert_eq!(extent.data_bytes(), count as u64 * 4);
            assert_eq!(extent.parameter_bytes(), count.to_le_bytes());
            assert_eq!(
                extent.launch().unwrap(),
                LaunchShape::Workgroups(Workgroups::new([groups, 1, 1], [256, 1, 1]).unwrap())
            );
        }
    }
}
