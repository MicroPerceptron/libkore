use axnn::{Scalar, Tensor, TensorElementAccess};

use crate::backend::{get_cpu_tensor_element, set_cpu_tensor_element};
use crate::error::CpuError;
use crate::storage::CpuStorage;

impl<'p> TensorElementAccess<CpuStorage<'p>> for Tensor<CpuStorage<'p>> {
    type Error = CpuError;

    #[inline]
    fn set(&mut self, indices: &[usize], value: Scalar) -> Result<(), CpuError> {
        set_cpu_tensor_element(self, indices, value)
    }

    #[inline]
    fn get(&self, indices: &[usize]) -> Result<Scalar, CpuError> {
        get_cpu_tensor_element(self, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axnn::{Q1, TensorElementAccess, shape, tensor};

    use crate::CpuBackend;

    #[test]
    fn q1_set_get_without_backend_argument() {
        let be = CpuBackend::<4096>::new();
        let mut adj = tensor!(&be, Q1, shape![4, 4]).unwrap();

        adj.set(&[0, 1], true.into()).unwrap();
        assert_eq!(adj.get(&[0, 1]).unwrap(), Scalar::Bool(true));

        adj.set(&[0, 1], false.into()).unwrap();
        assert_eq!(adj.get(&[0, 1]).unwrap(), Scalar::Bool(false));
    }
}
