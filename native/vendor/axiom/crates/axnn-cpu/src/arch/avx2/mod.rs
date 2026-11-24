//! AVX2 + FMA + F16C kernel tier — stub.
//!
//! Only compiled when `feature = "avx2"` is enabled.
//! Assumes FMA and F16C are present (true for all real AVX2 CPUs).

pub(crate) mod cast;
pub(crate) mod conv;
pub(crate) mod f4;
pub(crate) mod f8;
pub(crate) mod gemm;
pub(crate) mod gemv;
pub(crate) mod mem;
pub(crate) mod qdot;
pub(crate) mod raster;
pub(crate) mod reduce;
pub(crate) mod rope;
pub(crate) mod sh;
pub(crate) mod splat;
pub(crate) mod vec;
