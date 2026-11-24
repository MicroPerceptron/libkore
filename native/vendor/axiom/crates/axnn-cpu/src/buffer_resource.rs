//! Checked, borrowed byte resources with explicit zero-outside loads.

use crate::planner::{broadcast_index, typed_input_matches, typed_output_matches};
use crate::{CpuError, CpuImplementation, CpuKernelPlan};
use axnn::lowering::{KernelEffects, KernelParams, SemanticOp};
use axnn::{DType, Shape};

#[derive(Clone, Copy, Debug)]
pub struct U8BufferRef<'a> {
    bytes: &'a [u8],
}

impl<'a> U8BufferRef<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, CpuError> {
        if bytes.len() > i32::MAX as usize {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed U8 buffer length",
            });
        }
        Ok(Self { bytes })
    }

    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub const fn elements(self) -> u32 {
        self.bytes.len() as u32
    }
}

/// Every input and output descriptor is validated before any output write.
/// Negative/out-of-bounds indices produce zero; valid bytes widen to I32.
/// No pointers are retained, allocations performed, or resource copies made.
pub fn invoke_buffer_load_u8(
    plan: &CpuKernelPlan,
    buffer: U8BufferRef<'_>,
    index: &[i32],
    dst: &mut [i32],
) -> Result<(), CpuError> {
    let KernelParams::BufferLoadU8(params) = plan.params() else {
        return Err(CpuError::ShapeMismatch {
            op: "U8 resource parameters",
        });
    };
    if plan.semantic != SemanticOp::BufferLoadU8
        || !params.is_valid()
        || params.elements != buffer.elements()
        || plan.implementation != CpuImplementation::Scalar
        || plan.effects() != KernelEffects::READS_RESOURCE
        || plan.temporary_bytes != 0
        || plan.result_count() != 1
        || plan.output_dtype != DType::I32
        || plan.shape.rank() != 2
        || plan.input_dtypes != [Some(DType::I32), None, None]
        || !typed_input_matches(plan, 0, DType::I32, index.len())
        || plan.input(0).is_none_or(|d| d.layout.offset() != 0)
        || plan.input(1).is_some()
        || plan.input(2).is_some()
        || !typed_output_matches(plan, 0, DType::I32, plan.shape, dst.len())
        || plan.result_tensor(0).is_none_or(|d| d.layout.offset() != 0)
        || plan.input_shapes[0].and_then(|s| s.broadcast_with(&plan.shape)) != Some(plan.shape)
    {
        return Err(CpuError::ShapeMismatch {
            op: "planned U8 resource load",
        });
    }
    let shape = plan.input_shapes[0].unwrap();
    for (pixel, out) in dst.iter_mut().enumerate() {
        let i = index[load_index(shape, plan.shape, pixel)];
        *out = usize::try_from(i)
            .ok()
            .and_then(|i| buffer.bytes.get(i))
            .copied()
            .unwrap_or(0) as i32;
    }
    Ok(())
}

#[inline]
fn load_index(input: Shape, output: Shape, index: usize) -> usize {
    broadcast_index(input, output, index).expect("broadcast validated before output writes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use axnn::lowering::{KernelRequest, ResourceKey, U8BufferParams};
    use axnn::{TensorDesc, shape};

    #[test]
    fn byte_loads_are_exact_bounded_and_borrow_current_storage() {
        let mut bytes = [0, 127, 128, 255];
        let params = U8BufferParams::new(
            ResourceKey {
                group: 1,
                binding: 2,
            },
            4,
        )
        .unwrap();
        let request = KernelRequest::buffer_load_u8(
            &TensorDesc::new(shape![1, 6], DType::I32),
            params,
            shape![2, 6],
        )
        .unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
        let indices = [-1, 0, 1, 2, 3, i32::MAX];
        let mut out = [-9; 12];
        invoke_buffer_load_u8(&plan, U8BufferRef::new(&bytes).unwrap(), &indices, &mut out)
            .unwrap();
        assert_eq!(out, [0, 0, 127, 128, 255, 0, 0, 0, 127, 128, 255, 0]);
        bytes[2] = 200;
        invoke_buffer_load_u8(&plan, U8BufferRef::new(&bytes).unwrap(), &indices, &mut out)
            .unwrap();
        assert_eq!(out[3], 200);
        assert_eq!(out[9], 200);
        out.fill(-9);
        assert!(
            invoke_buffer_load_u8(
                &plan,
                U8BufferRef::new(&bytes[..3]).unwrap(),
                &indices,
                &mut out
            )
            .is_err()
        );
        assert_eq!(out, [-9; 12]);
    }

    #[test]
    fn empty_byte_buffer_and_scalar_broadcast_are_defined() {
        let params = U8BufferParams::new(
            ResourceKey {
                group: 0,
                binding: 0,
            },
            0,
        )
        .unwrap();
        let request = KernelRequest::buffer_load_u8(
            &TensorDesc::new(Shape::SCALAR, DType::I32),
            params,
            shape![1, 7],
        )
        .unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let mut out = [-9; 7];
        invoke_buffer_load_u8(&plan, U8BufferRef::new(&[]).unwrap(), &[0], &mut out).unwrap();
        assert_eq!(out, [0; 7]);
    }
}
