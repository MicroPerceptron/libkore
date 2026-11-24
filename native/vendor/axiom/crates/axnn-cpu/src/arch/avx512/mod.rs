//! AVX-512 integer dot products, packed FP4, BF16, and attention kernels.
pub(crate) mod attention;
pub(crate) mod f4;
pub(crate) mod gemv;
#[allow(dead_code)]
pub(crate) mod qdot;
