//! `axnn` — tensor descriptors, backend traits, and compute operations.
//!
//! This crate is `no_std` and `no_alloc`. It defines the common
//! vocabulary (shapes, dtypes, tensor metadata) and the trait
//! interfaces that backend crates (e.g., `axnn-cpu`) implement.
//!
//! # Quick start
//!
//! ```ignore
//! use axnn::{shape, tensor, Backend, F32, Q4};
//! use axnn_cpu::CpuBackend;
//!
//! let be = CpuBackend::new();
//!
//! // Create tensors
//! let x = tensor!(&be, [1.0, 2.0, 3.0]).unwrap();     // 1D f32 from values
//! let w = tensor!(&be, F32, shape![3]).unwrap();        // zeros
//! let q = tensor!(&be, Q4, shape![256], &bytes).unwrap(); // from raw bytes
//!
//! // Compute — torch.nn.functional style
//! let y = axnn::relu(&x, &be).unwrap();
//! let s = axnn::softmax(&y, -1, &be).unwrap();
//! let n = axnn::rms_norm(&x, &w, 1e-5, &be).unwrap();
//!
//! // Download result
//! let mut buf = [0u8; 12];
//! be.download(&s, &mut buf).unwrap();
//! ```
//!
//! # Backend setup
//!
//! `axnn` only defines the traits, tensor metadata, dtype vocabulary, and
//! free-function API. To execute anything, create a backend from an
//! implementation crate such as `axnn-cpu`, then pass `&be` to `tensor!`
//! and the operation functions.
//!
//! # Automatic kernel lowering
//!
//! Semantic graph operations must not name tuned kernels directly. The
//! [`lowering`] module is the common optimization boundary: a typed
//! [`lowering::KernelRequest`] describes complete input and result tensor
//! descriptors, operation parameters, aliasing effects, and logical work;
//! provider-owned [`lowering::KernelCatalog`] implementations enumerate legal
//! scalar, SIMD, ISA-specific, packed, or accelerator candidates; and
//! [`lowering::select_kernel`] admits and prices them using runtime policy.
//!
//! This separation keeps `.ax` compact while ensuring that tuned kernels are
//! actually reachable. Adding an implementation without a catalog candidate
//! does not make it part of automatic graph execution. Catalog enumeration is
//! allocation-free; allocation and executable caching remain choices of the
//! final runtime and its explicit allocator.
//!
//! The CPU backend is arena-backed and `no_std` by default, so tensor
//! allocation is explicit and bounded:
//!
//! ```ignore
//! use axnn::{shape, tensor, F32};
//! use axnn_cpu::CpuBackend;
//!
//! // 64 KiB inline arena. Pick a size that fits your workload.
//! let be = CpuBackend::<{ 64 * 1024 }>::new();
//! let x = tensor!(&be, F32, shape![128]).unwrap();
//! ```
//!
//! With `axnn-cpu/std`, `CpuBackend::new_boxed()` is also available when
//! the default 16 MiB arena would be too large for the stack.
//!
//! # Q1 Graphs
//!
//! `Q1` tensors are bit-packed binary matrices. For graph scheduling,
//! use a square `Q1` adjacency matrix where bit `(i, j) = true` means
//! "node `i` must run before node `j`". The public `set`, `get`,
//! `indegree`, and `topo_sort` functions route through the backend's
//! optimized bit-matrix kernels.
//!
//! ```ignore
//! use axnn::{shape, tensor, Q1, TensorElementAccess};
//! use axnn_cpu::CpuBackend;
//!
//! let be = CpuBackend::<4096>::new();
//! let adj = tensor!(&be, Q1, shape![4, 4]).unwrap();
//!
//! // Diamond DAG: 0 -> {1, 2} -> 3.
//! adj.set(&[0, 1], true.into()).unwrap();
//! adj.set(&[0, 2], true.into()).unwrap();
//! adj.set(&[1, 3], true.into()).unwrap();
//! adj.set(&[2, 3], true.into()).unwrap();
//!
//! let mut indegree = [0u32; 4];
//! let n = axnn::indegree(&adj, &mut indegree, &be).unwrap();
//! assert_eq!(n, 4);
//! assert_eq!(indegree, [0, 1, 1, 2]);
//!
//! let mut order = [0u16; 4];
//! let n = axnn::topo_sort(&adj, &mut order, &be).unwrap();
//! assert_eq!(n, 4);
//!
//! let pos = |node| order.iter().position(|&x| x == node).unwrap();
//! assert!(pos(0) < pos(1));
//! assert!(pos(0) < pos(2));
//! assert!(pos(1) < pos(3));
//! assert!(pos(2) < pos(3));
//! ```
//!
//! Q1 rows are padded to 256-bit blocks internally; you should still
//! describe the graph with its logical shape, e.g. `shape![4, 4]` or
//! `shape![256, 256]`. The descriptor computes the physical row stride.

#![no_std]
#![feature(f16)]

#[cfg(test)]
extern crate alloc;

/// Maximum tensor rank supported without heap allocation.
/// Rank 8 covers all practical NN and rendering workloads.
pub const MAX_RANK: usize = 8;

#[macro_use]
mod macros;

// Modules are pub(crate) unless macros or downstream crates need
// to path into them. Only re-exported symbols form the public API.
pub(crate) mod backend;
pub(crate) mod compute;
pub(crate) mod dtype;
pub(crate) mod layout;
pub mod lowering;
pub(crate) mod ops;
pub mod raster;
pub(crate) mod scalar;
pub(crate) mod tensor;

// `shape` must be pub because the `shape![]` and `tensor![]` macros
// expand to `$crate::shape::Shape::from(...)` in downstream crates.
#[doc(hidden)]
pub mod shape;

// ── Public API ──────────────────────────────────────────────────────

// Traits — users import these to call backend methods directly.
pub use backend::Backend;
pub use compute::ComputeBackend;
pub use raster::{
    CoverageFormat, MAX_FALLOFF, PLACED_GLYPH_BYTES, PlacedGlyph, RASTER_ROW_WORDS, RasterBackend,
    RasterRow, Rect, RoundedShape, SurfaceMut, gaussian_falloff,
};

// Types — the core vocabulary.
pub use dtype::DType;
pub use layout::Layout;
pub use scalar::Scalar;
pub use shape::Shape;
pub use tensor::{Tensor, TensorDesc, TensorElementAccess};

// DType constants — shorthand for `DType::F32` etc.
// Enables `tensor!(&be, F32, shape![2, 3])` without `DType::` prefix.
pub const F64: DType = DType::F64;
pub const F32: DType = DType::F32;
pub const F16: DType = DType::F16;
pub const B16: DType = DType::B16;
pub const I64: DType = DType::I64;
pub const I32: DType = DType::I32;
pub const I16: DType = DType::I16;
pub const I8: DType = DType::I8;
pub const U8: DType = DType::U8;
pub const Q8: DType = DType::Q8;
pub const Q6: DType = DType::Q6;
pub const Q4: DType = DType::Q4;
pub const Q1: DType = DType::Q1;

// Enums — user-facing (passed as arguments to ops).
pub use compute::RopeMode;

// Enums — backend implementors need these to match on dispatch.
// Not part of the user-facing API (users call axnn::relu, not
// be.unary(UnaryOp::Relu, ...)).
pub use compute::{ArgReduceOp, BinaryOp, CompareOp, ReduceOp, TernaryOp, UnaryOp};

// Ops — torch.nn.functional-style free functions.
// `pub use ops::*` re-exports: relu, gelu, silu, sigmoid, tanh,
// exp, sqrt, neg, add, sub, mul, div, scale, sum, max, argmin, argmax, softmax,
// rms_norm, layer_norm, matmul, vec_dot, rope, conv_1d, ssm_conv.
pub use ops::*;
