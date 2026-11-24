use crate::DType;
use crate::tensor::{Tensor, TensorDesc};

/// A compute backend that can allocate and manipulate tensor storage.
///
/// Backends are **synchronous** — each call records or executes an
/// operation and returns immediately. Async dispatch lives one layer
/// up, at the session/graph level: a batch of backend calls is
/// submitted as a unit and the completion is awaited via an event
/// ticket from the runtime.
///
/// # Associated types
///
/// - **`Storage<'a>`**: the handle to device memory, borrowing from
///   the backend's pool with lifetime `'a`. `Clone` is required so
///   that view operations (transpose, slice, etc.) can share one
///   underlying buffer across multiple tensor descriptors. The
///   lifetime ensures all tensors are dropped before the pool.
///
/// - **`Error`**: backend-specific failure (OOM, unsupported dtype
///   pair, device lost, etc.).
///
/// # Scope
///
/// This trait covers the **data-management** surface that every
/// backend must support regardless of which compute ops it offers:
///
/// - Allocation and initialization
/// - Host ↔ device data transfer
/// - View materialization (non-contiguous → contiguous)
/// - Type casting / quantization
///
/// Compute operations (matmul, elementwise, reductions) are dispatched
/// through the graph layer, not through this trait.
pub trait Backend {
    /// Handle to allocated device memory.
    ///
    /// The lifetime `'a` ties the storage to the backend's pool —
    /// tensors cannot outlive the backing memory. `Clone` enables
    /// zero-copy views: multiple `Tensor<S>` values can reference
    /// the same storage with different descriptors.
    type Storage<'a>: Clone
    where
        Self: 'a;

    /// Backend-specific error type.
    type Error;

    // ── Allocation & initialization ────────────────────────────────

    /// Allocate storage for a tensor descriptor.
    ///
    /// Returns a tensor with the given shape and dtype, backed by
    /// fresh contiguous storage. **Contents are uninitialized.**
    fn alloc(&self, desc: &TensorDesc) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Allocate and zero-fill storage for a tensor descriptor.
    ///
    /// Equivalent to `alloc` + memset-zero but backends may fuse this
    /// into a single operation (e.g., `calloc`, zeroed device memory).
    fn zeros(&self, desc: &TensorDesc) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Allocate and fill storage with a byte pattern repeated per block.
    ///
    /// `pattern` must be exactly `desc.dtype.block_byte_size()` bytes.
    /// Each block in the allocated tensor is filled with this pattern.
    /// For scalar types, this fills every element with the same value.
    ///
    /// Returns `Err` if `pattern.len() != dtype.block_byte_size()`.
    fn fill(
        &self,
        desc: &TensorDesc,
        pattern: &[u8],
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    // ── Host ↔ device transfer ──────────────────────────────────────

    /// Create a device tensor from host bytes.
    ///
    /// `src` must contain exactly `desc.storage_bytes()` bytes in
    /// contiguous row-major block layout. The backend allocates
    /// device storage and copies the data in.
    fn upload(
        &self,
        src: &[u8],
        desc: &TensorDesc,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Wrap caller-owned host bytes as a zero-copy tensor for the
    /// duration of the borrow — the path for a bounded stack chunk a
    /// renderer just filled (a snapshot of a remotely writable surface)
    /// that must not be copied into device storage first.  `bytes` must
    /// hold exactly `desc.storage_bytes()` bytes past `desc.layout.offset()`.
    fn wrap<'a>(
        &self,
        bytes: &'a [u8],
        desc: &TensorDesc,
    ) -> Result<Tensor<Self::Storage<'a>>, Self::Error>
    where
        Self: 'a;

    /// Copy device tensor data into a host byte buffer.
    ///
    /// Writes into `dst` in contiguous row-major block layout.
    /// If the tensor is non-contiguous, the backend implicitly
    /// materializes it first. `dst` must be at least
    /// `tensor.desc().storage_bytes()` bytes.
    fn download(
        &self,
        tensor: &Tensor<Self::Storage<'_>>,
        dst: &mut [u8],
    ) -> Result<(), Self::Error>;

    // ── View materialization & type conversion ──────────────────────

    /// Materialize a non-contiguous tensor into contiguous storage.
    ///
    /// If the input is already contiguous, implementations may
    /// return it unchanged (avoiding a copy). This is the operation
    /// that turns a transposed/sliced view into a layout that
    /// compute kernels can consume directly.
    fn contiguous<'a>(
        &'a self,
        tensor: &Tensor<Self::Storage<'a>>,
    ) -> Result<Tensor<Self::Storage<'a>>, Self::Error>;

    /// Convert a tensor to a different data type.
    ///
    /// Returns a new contiguous tensor with the target dtype.
    /// Conversion semantics depend on the dtype pair:
    ///
    /// | Direction            | Operation                              |
    /// |----------------------|----------------------------------------|
    /// | Scalar → Scalar      | Element-wise truncation or widening    |
    /// | Quantized → Scalar   | Dequantization (unpack × scale)        |
    /// | Scalar → Quantized   | Quantization (find scales, pack)       |
    /// | Quantized → Quantized| Requantization (dequant then quant)    |
    ///
    /// Returns `Err` if the backend doesn't support the requested
    /// conversion or the tensor's innermost dimension violates
    /// block-alignment constraints for the target dtype.
    fn cast<'a>(
        &'a self,
        tensor: &Tensor<Self::Storage<'a>>,
        target: DType,
    ) -> Result<Tensor<Self::Storage<'a>>, Self::Error>;
}
