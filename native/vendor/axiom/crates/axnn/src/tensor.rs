use crate::dtype::DType;
use crate::layout::Layout;
use crate::scalar::Scalar;
use crate::shape::Shape;

/// Complete metadata description of a tensor — no backing storage.
///
/// This is the common language that all layers speak:
/// - The graph IR references tensors by `TensorDesc`
/// - Operations define signatures over `TensorDesc`
/// - Device drivers receive `TensorDesc` to know what memory to bind
/// - Serialization encodes `TensorDesc` for IPC
///
/// `shape` is always in **logical elements** (what the user thinks in).
/// The physical block layout is derived on demand via `dtype.block_shape()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDesc {
    /// Logical element shape (e.g., [batch, seq_len, hidden_dim]).
    pub shape: Shape,
    /// Block format (unified — scalar types are degenerate blocks).
    pub dtype: DType,
    /// Physical byte layout. Strides operate over the block shape,
    /// not the logical element shape.
    pub layout: Layout,
}

impl TensorDesc {
    /// Create a new tensor descriptor with contiguous (row-major) layout.
    ///
    /// The layout is computed from the block shape: the innermost
    /// dimension is in units of blocks, with stride = `block_byte_size`.
    /// For scalar types this is identical to a per-element layout.
    pub fn new(shape: Shape, dtype: DType) -> Self {
        Self::try_new(shape, dtype).expect("tensor descriptor size overflow")
    }

    /// Checked form of [`TensorDesc::new`].
    ///
    /// Returns `None` when the logical element count, quantized block count,
    /// or any contiguous byte stride would overflow `usize`.
    pub fn try_new(shape: Shape, dtype: DType) -> Option<Self> {
        shape.checked_num_elements()?;
        let block_dims = Self::checked_block_dims_for(shape, dtype)?;
        let layout = Layout::try_contiguous(&block_dims, dtype.block_byte_size())?;
        layout.checked_span_bytes(&block_dims, dtype.block_byte_size())?;
        Some(TensorDesc {
            shape,
            dtype,
            layout,
        })
    }

    /// Total number of logical elements.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    fn checked_block_dims_for(shape: Shape, dtype: DType) -> Option<Shape> {
        // `DType::block_shape` uses an infallible ceil-division helper. Do the
        // same conversion without `last + block_size - 1`, which can overflow
        // for descriptors arriving from an external graph.
        if dtype.block_size() == 1 || shape.rank() == 0 {
            return Some(shape);
        }
        let rank = shape.rank();
        let mut dims = [0usize; crate::MAX_RANK];
        dims[..rank].copy_from_slice(shape.dims());
        let last = dims[rank - 1];
        let block_size = dtype.block_size();
        dims[rank - 1] = last
            .checked_div(block_size)?
            .checked_add(usize::from(last % block_size != 0))?;
        Some(Shape::from_parts(dims, rank))
    }

    fn checked_block_dims(&self) -> Option<Shape> {
        Self::checked_block_dims_for(self.shape, self.dtype)
    }

    /// Whether the underlying memory is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.checked_block_dims().is_some_and(|block_dims| {
            self.layout
                .is_contiguous(&block_dims, self.dtype.block_byte_size())
        })
    }

    /// Total bytes needed for storage backing this tensor.
    pub fn storage_bytes(&self) -> usize {
        self.checked_storage_bytes()
            .expect("tensor storage byte span overflow")
    }

    /// Checked byte span of the logical view, excluding its starting offset.
    pub fn checked_storage_bytes(&self) -> Option<usize> {
        let block_dims = self.checked_block_dims()?;
        self.layout
            .checked_span_bytes(&block_dims, self.dtype.block_byte_size())
    }

    /// Exact byte range occupied by a contiguous view inside its backing
    /// storage. Returns `None` for strided views or an overflowing end offset.
    pub fn contiguous_byte_range(&self) -> Option<core::ops::Range<usize>> {
        if !self.is_contiguous() {
            return None;
        }
        let start = self.layout.offset();
        let bytes = self.checked_storage_bytes()?;
        let end = start.checked_add(bytes)?;
        Some(start..end)
    }

    /// Swap two dimensions. Pure metadata — no memory copy.
    ///
    /// Returns `None` if either dimension is out of range.
    ///
    /// For block-quantized types, the innermost dimension is the physical
    /// block axis and cannot be moved by this descriptor representation.
    /// Such a transpose returns `None` and requires explicit materialization.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Option<Self> {
        let rank = self.shape.rank();
        self.checked_storage_bytes()?;
        if dim0 >= rank || dim1 >= rank {
            return None;
        }
        // The descriptor records block structure only on the innermost axis.
        // Moving that axis would require an explicit block-axis field rather
        // than merely swapping byte strides.
        if self.dtype.block_size() > 1 && dim0 != dim1 && (dim0 == rank - 1 || dim1 == rank - 1) {
            return None;
        }
        let new_layout = self.layout.transpose(dim0, dim1);
        let mut new_dims = [0usize; crate::MAX_RANK];
        new_dims[..rank].copy_from_slice(self.shape.dims());
        new_dims.swap(dim0, dim1);
        let new_shape = Shape::from_parts(new_dims, rank);

        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout: new_layout,
        })
    }

    /// Reorder all dimensions. Pure metadata — no memory copy.
    ///
    /// `axes` must be a permutation of `0..rank`.
    /// Returns `None` if the permutation is invalid, or if the
    /// innermost (quantization) axis of a block type would move.
    pub fn permute(&self, axes: &[usize]) -> Option<Self> {
        let rank = self.shape.rank();
        self.checked_storage_bytes()?;
        // block_shape always applies the block divisor to the innermost dim,
        // so moving it out of position would corrupt block_dims / storage_bytes.
        if self.dtype.block_size() > 1 && rank > 0 && axes.last() != Some(&(rank - 1)) {
            return None;
        }
        let new_shape = self.shape.permute(axes)?;
        let new_layout = self.layout.permute(axes);
        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout: new_layout,
        })
    }

    /// Slice along a single dimension. Pure metadata — no memory copy.
    ///
    /// For block-quantized types, slicing the innermost dimension
    /// requires `start` and `len` to be multiples of `block_size`.
    /// Returns `None` if this constraint is violated.
    ///
    /// For scalar types (block_size=1) or non-innermost dimensions,
    /// any range is valid.
    pub fn slice(&self, dim: usize, start: usize, len: usize) -> Option<Self> {
        let rank = self.shape.rank();
        self.checked_storage_bytes()?;
        if dim >= rank {
            return None;
        }
        let end = start.checked_add(len)?;
        if end > self.shape.dim(dim as isize) {
            return None;
        }
        let is_innermost = dim == rank - 1;
        let bs = self.dtype.block_size();

        // Block alignment check on the quantization axis.
        if is_innermost && bs > 1 {
            if start % bs != 0 || len % bs != 0 {
                return None;
            }
        }

        // Compute the stride unit for this dimension.
        // For the innermost dim of a block type, we stride in blocks.
        let stride_start = if is_innermost && bs > 1 {
            start / bs
        } else {
            start
        };

        let new_layout = self.layout.try_slice(dim, stride_start)?;

        let mut new_dims = [0usize; crate::MAX_RANK];
        new_dims[..rank].copy_from_slice(self.shape.dims());
        new_dims[dim] = len;
        let new_shape = Shape::from_parts(new_dims, rank);

        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout: new_layout,
        })
    }

    /// Insert a size-1 dimension at position `dim`. Pure metadata.
    ///
    /// Returns `None` if the resulting rank would exceed `MAX_RANK`.
    pub fn unsqueeze(&self, dim: usize) -> Option<Self> {
        self.checked_storage_bytes()?;
        let new_shape = self.shape.unsqueeze(dim)?;
        // Stride hint for the new size-1 dim. Must equal
        // strides[dim] × block_dims.dim(dim) so that is_contiguous
        // still holds after inserting a leading dimension.
        let rank = self.shape.rank();
        let block_dims = self.checked_block_dims()?;
        let hint = if dim < rank {
            self.layout.strides()[dim].checked_mul(block_dims.dim(dim as isize))?
        } else {
            self.dtype.block_byte_size()
        };
        let new_layout = self.layout.unsqueeze(dim, hint);
        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout: new_layout,
        })
    }

    /// Remove a size-1 dimension at position `dim`. Pure metadata.
    ///
    /// Returns `None` if `dim` is out of range, not size 1, or would
    /// remove the innermost dimension of a block-quantized type
    /// (which would leave a non-block-aligned innermost dim).
    pub fn squeeze(&self, dim: usize) -> Option<Self> {
        let rank = self.shape.rank();
        self.checked_storage_bytes()?;
        // Squeezing the innermost dim of a quantized tensor would
        // expose a non-block-aligned dim as the new innermost.
        if self.dtype.block_size() > 1 && dim == rank - 1 && rank > 1 {
            return None;
        }
        let new_shape = self.shape.squeeze(dim)?;
        let new_layout = self.layout.squeeze(dim);
        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout: new_layout,
        })
    }

    /// Broadcast to a larger shape by setting stride=0 on expanded dims.
    ///
    /// Only size-1 dimensions can be expanded. The target shape must
    /// be broadcast-compatible with the current shape (same rank,
    /// each dim either matches or is 1→N).
    ///
    /// This is **zero-memory broadcast**: the expanded dimension
    /// reads the same data repeatedly via stride=0. No storage change.
    ///
    /// Returns `None` if:
    /// - Ranks don't match (use `unsqueeze` first to align ranks)
    /// - A non-size-1 dimension differs from the target
    /// - Quantized innermost dim would be expanded (breaks block structure)
    pub fn expand(&self, target: &Shape) -> Option<Self> {
        let rank = self.shape.rank();
        self.checked_storage_bytes()?;
        target.checked_num_elements()?;
        if target.rank() != rank {
            return None;
        }

        let bs = self.dtype.block_size();
        let mut new_dims = [0usize; crate::MAX_RANK];
        let mut new_strides = [0usize; crate::MAX_RANK];
        let strides = self.layout.strides();

        for i in 0..rank {
            let src = self.shape.dim(i as isize);
            let dst = target.dim(i as isize);

            if src == dst {
                // Dimension unchanged — keep stride.
                new_dims[i] = src;
                new_strides[i] = strides[i];
            } else if src == 1 {
                // Expanding size-1 → dst via stride=0.
                // Block the innermost dim of quantized types.
                if bs > 1 && i == rank - 1 {
                    return None;
                }
                new_dims[i] = dst;
                new_strides[i] = 0;
            } else {
                // Non-size-1 dim doesn't match target — invalid.
                return None;
            }
        }

        Some(TensorDesc {
            shape: Shape::from_parts(new_dims, rank),
            dtype: self.dtype,
            layout: Layout::from_raw(&new_strides[..rank], self.layout.offset()),
        })
    }

    /// Select a single index along dimension 0, reducing rank by 1.
    ///
    /// `tensor.at(i)` is equivalent to `tensor.slice(0, i, 1).squeeze(0)`.
    /// Chainable: `tensor.at(2).at(0)` selects index 2 of dim 0, then
    /// index 0 of what was dim 1.
    ///
    /// Returns `None` if the index is out of bounds, or if the tensor
    /// is scalar (rank 0). For block-quantized types on a rank-1 tensor,
    /// returns `None` because selecting a single element from the
    /// innermost (quantization) axis breaks block alignment.
    pub fn at(&self, index: usize) -> Option<Self> {
        let rank = self.shape.rank();
        if rank == 0 || index >= self.shape.dim(0) {
            return None;
        }
        // slice(dim=0, start=index, len=1) then squeeze(dim=0)
        self.slice(0, index, 1)?.squeeze(0)
    }

    /// Reinterpret as a different shape without copying.
    ///
    /// Returns `None` if:
    /// - The layout is non-contiguous (would require a copy)
    /// - The new shape has a different number of elements
    /// - For quantized types: the new innermost dim isn't
    ///   a multiple of `block_size`
    pub fn view(&self, new_shape: Shape) -> Option<Self> {
        self.checked_storage_bytes()?;
        if new_shape.checked_num_elements()? != self.shape.checked_num_elements()? {
            return None;
        }
        if !self.is_contiguous() {
            return None;
        }
        let bs = self.dtype.block_size();
        if bs > 1 && new_shape.rank() > 0 {
            if new_shape.dim(-1) % bs != 0 {
                return None;
            }
        }
        let block_dims = Self::checked_block_dims_for(new_shape, self.dtype)?;
        let mut layout = Layout::try_contiguous(&block_dims, self.dtype.block_byte_size())?;
        // Preserve the byte offset — view of a sliced tensor must still
        // point into the same position within the backing buffer.
        layout = Layout::from_raw(layout.strides(), self.layout.offset());
        Some(TensorDesc {
            shape: new_shape,
            dtype: self.dtype,
            layout,
        })
    }
}

// ── Tensor<S>: descriptor + storage ─────────────────────────────

/// A tensor with backing storage of type `S`.
///
/// `S` encodes where the data lives at the type level:
///
/// - `Tensor<()>` — unbound. Same as a bare `TensorDesc`. Used in
///   graph IR, op signatures, and anywhere storage hasn't been
///   assigned yet. Zero-cost: `()` is a ZST.
///
/// - `Tensor<HostBuf>` — CPU heap memory (future).
/// - `Tensor<LeaseRef>` — kraph lease handle (future).
/// - `Tensor<DeviceBuf>` — device-local memory (future).
///
/// View operations (transpose, slice, reshape) produce a new
/// `Tensor<S>` sharing the same storage (`S: Clone`). The descriptor
/// changes; the storage is aliased. This mirrors how views work in
/// every major tensor library — multiple descriptors over one buffer.
#[derive(Debug, Clone)]
pub struct Tensor<S> {
    desc: TensorDesc,
    storage: S,
}

impl<S: Copy> Copy for Tensor<S> {}

/// Backend-provided ergonomic element access for placed tensors.
///
/// Backends implement this for `Tensor<BackendStorage>` so callers can use
/// `tensor.set(&[...], value)` / `tensor.get(&[...])` without passing the
/// backend object at every access site.
///
/// The `Storage` parameter is intentionally part of the trait. It gives
/// backend crates a local type in the implementation, keeping the API
/// compatible with Rust's coherence rules.
pub trait TensorElementAccess<Storage> {
    type Error;

    /// Set a single tensor element by logical multi-dimensional index.
    fn set(&mut self, indices: &[usize], value: Scalar) -> Result<(), Self::Error>;

    /// Get a single tensor element by logical multi-dimensional index.
    fn get(&self, indices: &[usize]) -> Result<Scalar, Self::Error>;
}

/// Unbound tensor — pure metadata, identical footprint to `TensorDesc`.
// pub type TensorUnbound = Tensor<()>;

// ── Methods available on all tensors regardless of storage ──────

impl<S> Tensor<S> {
    /// Wrap a descriptor with a storage backend.
    pub fn new(desc: TensorDesc, storage: S) -> Self {
        Tensor { desc, storage }
    }

    /// The metadata descriptor.
    #[inline]
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// Borrow the storage.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Logical element shape.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.desc.shape
    }

    /// Data type.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.desc.dtype
    }

    /// Total logical elements.
    #[inline]
    pub fn count(&self) -> usize {
        self.desc.num_elements()
    }

    /// Whether the layout is contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.desc.is_contiguous()
    }

    /// Total bytes the storage must provide.
    #[inline]
    pub fn size(&self) -> usize {
        self.desc.storage_bytes()
    }

    /// Discard storage, returning just the descriptor.
    pub fn into_desc(self) -> TensorDesc {
        self.desc
    }
}

// ── View operations (require shareable storage) ─────────────────

impl<S: Clone> Tensor<S> {
    /// Transpose two dimensions. Shares storage.
    /// Returns `None` if either dimension is out of range.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Option<Self> {
        self.desc.transpose(dim0, dim1).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Reorder all dimensions. Shares storage.
    /// Returns `None` if the permutation is invalid.
    pub fn permute(&self, axes: &[usize]) -> Option<Self> {
        self.desc.permute(axes).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Select a single index along dimension 0, reducing rank by 1.
    /// Shares storage. Chainable: `tensor.at(2).at(0)`.
    pub fn at(&self, index: usize) -> Option<Self> {
        self.desc.at(index).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Slice along a dimension. Shares storage.
    /// Returns `None` if block alignment is violated.
    pub fn slice(&self, dim: usize, start: usize, len: usize) -> Option<Self> {
        self.desc.slice(dim, start, len).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Insert a size-1 dimension. Shares storage.
    pub fn unsqueeze(&self, dim: usize) -> Option<Self> {
        self.desc.unsqueeze(dim).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Remove a size-1 dimension. Shares storage.
    pub fn squeeze(&self, dim: usize) -> Option<Self> {
        self.desc.squeeze(dim).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// Broadcast to a larger shape. Shares storage.
    /// Returns `None` if not broadcast-compatible.
    pub fn expand(&self, target: &Shape) -> Option<Self> {
        self.desc.expand(target).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }

    /// View as a different shape. Shares storage.
    /// Returns `None` if non-contiguous or element count mismatch.
    pub fn view(&self, new_shape: Shape) -> Option<Self> {
        self.desc.view(new_shape).map(|desc| Tensor {
            desc,
            storage: self.storage.clone(),
        })
    }
}

// ── Unbound tensors ─────────────────────────────────────────────

impl Tensor<()> {
    /// Create an unbound tensor from shape and dtype.
    pub fn unbound(shape: Shape, dtype: DType) -> Self {
        Tensor {
            desc: TensorDesc::new(shape, dtype),
            storage: (),
        }
    }

    /// Bind storage, promoting from unbound to placed.
    pub fn bind<S>(self, storage: S) -> Tensor<S> {
        Tensor {
            desc: self.desc,
            storage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, shape};

    fn desc(dims: &[usize], dt: DType) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), dt)
    }

    // ── TensorDesc basics ───────────────────────────────────────

    #[test]
    fn contiguous_f32() {
        let d = desc(&[2, 3, 4], DType::F32);
        assert!(d.is_contiguous());
        assert_eq!(d.num_elements(), 24);
        assert_eq!(d.storage_bytes(), 96); // 24 × 4
    }

    #[test]
    fn contiguous_q4() {
        let d = desc(&[2, 256], DType::Q4);
        assert!(d.is_contiguous());
        assert_eq!(d.num_elements(), 512);
        // 2 rows × 1 block/row × 144 B/block = 288
        assert_eq!(d.storage_bytes(), 288);
    }

    // ── Transpose ───────────────────────────────────────────────

    #[test]
    fn transpose_swaps_shape_and_strides() {
        let d = desc(&[2, 3], DType::F32);
        let t = d.transpose(0, 1).unwrap();
        assert_eq!(t.shape.dims(), &[3, 2]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn transpose_is_involutory() {
        let d = desc(&[2, 3, 4], DType::F32);
        assert_eq!(d.transpose(0, 2).unwrap().transpose(0, 2).unwrap(), d,);
    }

    #[test]
    fn transpose_out_of_range_returns_none() {
        let d = desc(&[2, 3], DType::F32);
        assert!(d.transpose(0, 5).is_none());
        assert!(d.transpose(2, 0).is_none());
    }

    #[test]
    fn transpose_rejects_moving_the_quantized_block_axis() {
        let d = desc(&[2, 256], DType::Q4);
        assert!(d.transpose(0, 1).is_none());
        assert_eq!(d.transpose(1, 1), Some(d));
    }

    // ── Slice ───────────────────────────────────────────────────

    #[test]
    fn slice_outer_dim() {
        let d = desc(&[4, 3], DType::F32);
        let s = d.slice(0, 1, 2).unwrap();
        assert_eq!(s.shape.dims(), &[2, 3]);
        assert_eq!(s.layout.offset(), 12); // 1 × stride[0] = 1 × 12
    }

    #[test]
    fn slice_q4_aligned() {
        let d = desc(&[2, 512], DType::Q4);
        // Slice innermost dim: must be block-aligned (multiple of 256)
        let s = d.slice(1, 256, 256).unwrap();
        assert_eq!(s.shape.dims(), &[2, 256]);
    }

    #[test]
    fn slice_q4_unaligned_fails() {
        let d = desc(&[2, 512], DType::Q4);
        assert!(d.slice(1, 100, 256).is_none()); // 100 not multiple of 256
    }

    #[test]
    fn slice_rejects_invalid_dimension_range_and_offset_overflow() {
        let d = desc(&[4, 3], DType::F32);
        assert!(d.slice(2, 0, 1).is_none());
        assert!(d.slice(0, 3, 2).is_none());
        assert!(d.slice(0, usize::MAX, 2).is_none());
        assert!(
            TensorDesc::new(Shape::SCALAR, DType::F32)
                .slice(0, 0, 1)
                .is_none()
        );

        let overflowing_offset = TensorDesc {
            shape: shape![4],
            dtype: DType::F32,
            layout: Layout::from_raw(&[4], usize::MAX - 1),
        };
        assert!(overflowing_offset.slice(0, 1, 1).is_none());
    }

    // ── At (indexing) ───────────────────────────────────────────

    #[test]
    fn at_reduces_rank() {
        let d = desc(&[2, 3, 4], DType::F32);
        let row = d.at(1).unwrap();
        assert_eq!(row.shape.dims(), &[3, 4]);
    }

    #[test]
    fn at_chained() {
        let d = desc(&[2, 3, 4], DType::F32);
        let cell = d.at(1).unwrap().at(2).unwrap();
        assert_eq!(cell.shape.dims(), &[4]); // rank 1 — one row
    }

    #[test]
    fn at_out_of_bounds() {
        let d = desc(&[2, 3], DType::F32);
        assert!(d.at(5).is_none());
    }

    #[test]
    fn at_scalar_fails() {
        let d = TensorDesc::new(Shape::SCALAR, DType::F32);
        assert!(d.at(0).is_none());
    }

    // ── Unsqueeze / squeeze ─────────────────────────────────────

    #[test]
    fn unsqueeze_squeeze_preserves_storage_bytes() {
        let d = desc(&[2, 3], DType::F32);
        let expanded = d.unsqueeze(0).unwrap();
        assert_eq!(expanded.shape.dims(), &[1, 2, 3]);
        assert_eq!(expanded.storage_bytes(), d.storage_bytes());
        assert_eq!(expanded.squeeze(0).unwrap(), d);
    }

    // ── Expand (broadcast) ──────────────────────────────────────

    #[test]
    fn expand_size_one_to_n() {
        let d = desc(&[1, 3], DType::F32);
        let target = shape![4, 3];
        let e = d.expand(&target).unwrap();
        assert_eq!(e.shape, target);
        // Expanded dim has stride=0
        assert_eq!(e.layout.strides()[0], 0);
    }

    #[test]
    fn expand_q4_innermost_blocked() {
        let d = desc(&[1, 256], DType::Q4);
        // Can't expand the innermost dim of a quantized type
        let target = shape![1, 512];
        assert!(d.expand(&target).is_none());
    }

    // ── View (reshape) ──────────────────────────────────────────

    #[test]
    fn view_flatten() {
        let d = desc(&[2, 3, 4], DType::F32);
        let flat = d.view(shape![24]).unwrap();
        assert_eq!(flat.num_elements(), 24);
        assert!(flat.is_contiguous());
    }

    #[test]
    fn view_wrong_element_count() {
        let d = desc(&[2, 3], DType::F32);
        assert!(d.view(shape![7]).is_none());
    }

    #[test]
    fn view_non_contiguous_fails() {
        let d = desc(&[2, 3], DType::F32).transpose(0, 1).unwrap();
        assert!(d.view(shape![6]).is_none());
    }

    #[test]
    fn view_preserves_slice_offset() {
        let d = desc(&[4, 3], DType::F32);
        let sliced = d.slice(0, 2, 2).unwrap();
        assert_eq!(sliced.layout.offset(), 24); // 2 × stride[0] = 2 × 12
        let viewed = sliced.view(shape![6]).unwrap();
        assert_eq!(viewed.layout.offset(), 24); // must preserve offset
    }

    #[test]
    fn checked_descriptor_rejects_element_and_byte_overflow() {
        assert!(TensorDesc::try_new(shape![usize::MAX, 2], DType::U8).is_none());
        assert!(TensorDesc::try_new(shape![usize::MAX], DType::F32).is_none());
        assert!(TensorDesc::try_new(shape![usize::MAX], DType::Q8).is_none());
        assert_eq!(
            TensorDesc::try_new(shape![2, 3], DType::F32),
            Some(TensorDesc::new(shape![2, 3], DType::F32))
        );
    }

    #[test]
    fn checked_descriptor_rejects_shape_layout_rank_mismatch() {
        let malformed = TensorDesc {
            shape: shape![2, 2],
            dtype: DType::F32,
            layout: Layout::from_raw(&[], 0),
        };
        assert_eq!(malformed.checked_storage_bytes(), None);
        assert!(!malformed.is_contiguous());
        assert_eq!(malformed.contiguous_byte_range(), None);
        assert!(malformed.slice(0, 0, 1).is_none());
        assert!(malformed.view(shape![4]).is_none());
    }

    #[test]
    fn contiguous_byte_range_includes_view_offset() {
        let sliced = desc(&[4, 3], DType::F32).slice(0, 2, 1).unwrap();
        assert!(sliced.is_contiguous());
        assert_eq!(sliced.contiguous_byte_range(), Some(24..36));
        assert!(
            desc(&[2, 3], DType::F32)
                .transpose(0, 1)
                .unwrap()
                .contiguous_byte_range()
                .is_none()
        );
    }

    // ── Quantized guards ────────────────────────────────────────

    #[test]
    fn squeeze_q4_innermost_blocked() {
        let d = desc(&[4, 1], DType::Q4);
        // Can't squeeze the innermost dim of a quantized type.
        assert!(d.squeeze(1).is_none());
    }

    #[test]
    fn squeeze_q4_outer_dim_ok() {
        let d = desc(&[1, 256], DType::Q4);
        // Squeezing an outer dim is fine.
        let s = d.squeeze(0).unwrap();
        assert_eq!(s.shape.dims(), &[256]);
    }

    #[test]
    fn permute_q4_innermost_must_stay() {
        let d = desc(&[2, 3, 256], DType::Q4);
        // Can't move the innermost (quantization) axis.
        assert!(d.permute(&[2, 0, 1]).is_none());
        // Identity permute is fine.
        assert!(d.permute(&[0, 1, 2]).is_some());
        // Swapping outer dims only is fine.
        assert!(d.permute(&[1, 0, 2]).is_some());
    }

    #[test]
    fn unsqueeze_preserves_contiguity() {
        let d = desc(&[2, 3], DType::F32);
        let u = d.unsqueeze(0).unwrap();
        assert_eq!(u.shape.dims(), &[1, 2, 3]);
        assert!(u.is_contiguous());
        // view should work on a contiguous unsqueezed tensor.
        assert!(u.view(shape![6]).is_some());
    }

    #[test]
    fn unsqueeze_q4_preserves_contiguity() {
        let d = desc(&[2, 256], DType::Q4);
        let u = d.unsqueeze(0).unwrap();
        assert_eq!(u.shape.dims(), &[1, 2, 256]);
        assert!(u.is_contiguous());
    }

    // ── Tensor<S> ───────────────────────────────────────────────

    #[test]
    fn unbound_is_copy() {
        let t = Tensor::unbound(shape![2, 3], DType::F32);
        let t2 = t; // Copy
        assert_eq!(t.shape(), t2.shape());
    }

    #[test]
    fn bind_and_access() {
        let t = Tensor::unbound(shape![2, 3], DType::F32);
        let placed = t.bind(42u64); // bind with a dummy storage
        assert_eq!(*placed.storage(), 42u64);
        assert_eq!(placed.dtype(), DType::F32);
    }

    #[test]
    fn tensor_view_ops_share_storage() {
        let t = Tensor::unbound(shape![4, 3], DType::F32).bind(99u32);
        let sliced = t.slice(0, 1, 2).unwrap();
        assert_eq!(*sliced.storage(), 99u32); // same storage
        assert_eq!(sliced.shape().dims(), &[2, 3]);
    }

    #[test]
    fn tensor_at_chaining() {
        let t = Tensor::unbound(shape![2, 3, 4], DType::F32).bind(());
        let row = t.at(0).unwrap().at(1).unwrap();
        assert_eq!(row.shape().dims(), &[4]);
    }
}
