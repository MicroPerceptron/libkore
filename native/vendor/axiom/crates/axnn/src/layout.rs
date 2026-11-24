use crate::MAX_RANK;
use crate::shape::Shape;

/// Memory layout: how a tensor's logical shape maps to physical byte storage.
///
/// Strides are in **bytes**. For block-quantized types, strides operate at
/// block granularity on the innermost (quantization) axis: you stride by
/// `block_byte_size` per block, not per element. Outer dimensions stride
/// by `num_blocks_inner * block_byte_size`.
///
/// For scalar types (block_size=1), this degenerates to the familiar
/// per-element byte strides.
///
/// # Block-awareness
///
/// The layout itself is a "dumb" byte-stride container. Block structure
/// is accounted for during construction (`contiguous()`) and validated
/// during view operations (`transpose`, `slice`). The layout doesn't
/// store the DType — that's the TensorDesc's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    strides: [usize; MAX_RANK],
    rank: u8,
    /// Byte offset from the start of the backing storage.
    /// Non-zero when this layout describes a view into a larger buffer.
    offset: usize,
}

impl Layout {
    /// Construct a layout from explicit byte strides and offset.
    pub fn from_raw(strides: &[usize], offset: usize) -> Self {
        let rank = strides.len();
        let mut arr = [0usize; MAX_RANK];
        arr[..rank].copy_from_slice(strides);
        Layout {
            strides: arr,
            rank: rank as u8,
            offset,
        }
    }

    /// Compute a contiguous (row-major, C-order) layout.
    ///
    /// For block-quantized types, the innermost dimension is measured
    /// in blocks (logical_dim / block_size), and the innermost stride
    /// is `block_byte_size`. For scalar types (block_size=1), this
    /// reduces to the standard per-element layout.
    ///
    /// `block_dims` should be the shape with the innermost dimension
    /// already converted to block count. Use `DType::block_shape()`
    /// to obtain this.
    pub fn contiguous(block_dims: &Shape, block_byte_size: usize) -> Self {
        Self::try_contiguous(block_dims, block_byte_size)
            .expect("contiguous tensor layout overflow")
    }

    /// Checked form of [`Layout::contiguous`].
    ///
    /// Returns `None` when any byte stride would overflow `usize`.
    pub fn try_contiguous(block_dims: &Shape, block_byte_size: usize) -> Option<Self> {
        let rank = block_dims.rank();
        let mut strides = [0usize; MAX_RANK];

        if rank > 0 {
            strides[rank - 1] = block_byte_size;
            for i in (0..rank - 1).rev() {
                strides[i] = strides[i + 1].checked_mul(block_dims.dim(i as isize + 1))?;
            }
        }

        Some(Layout {
            strides,
            rank: rank as u8,
            offset: 0,
        })
    }

    /// Byte strides as a slice.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank as usize]
    }

    /// Byte offset from the start of storage.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of physical dimensions described by this layout.
    #[inline]
    pub const fn rank(&self) -> usize {
        self.rank as usize
    }

    /// Whether this layout is contiguous in memory (row-major, no gaps).
    pub fn is_contiguous(&self, block_dims: &Shape, block_byte_size: usize) -> bool {
        let rank = self.rank as usize;
        if rank != block_dims.rank() {
            return false;
        }
        if rank == 0 {
            return true;
        }
        if self.strides[rank - 1] != block_byte_size {
            return false;
        }
        for i in (0..rank - 1).rev() {
            let Some(expected) = self.strides[i + 1].checked_mul(block_dims.dim(i as isize + 1))
            else {
                return false;
            };
            if self.strides[i] != expected {
                return false;
            }
        }
        true
    }

    /// Swap two dimensions' strides.
    /// Pure metadata — no memory movement.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Layout {
        let mut new_strides = self.strides;
        new_strides.swap(dim0, dim1);
        Layout {
            strides: new_strides,
            rank: self.rank,
            offset: self.offset,
        }
    }

    /// Reorder strides according to the given axis permutation.
    /// `axes` must be a valid permutation of `0..rank`.
    pub fn permute(&self, axes: &[usize]) -> Layout {
        let mut new_strides = [0usize; MAX_RANK];
        for (i, &a) in axes.iter().enumerate() {
            new_strides[i] = self.strides[a];
        }
        Layout {
            strides: new_strides,
            rank: self.rank,
            offset: self.offset,
        }
    }

    /// Insert a stride for a new size-1 dimension at position `dim`.
    ///
    /// The new stride is set to the same as the stride at `dim`
    /// (or `block_byte_size` if inserted at the end). Since the dim
    /// is size 1, the stride value doesn't affect addressing — it
    /// only matters if the dim is later expanded via broadcast.
    pub fn unsqueeze(&self, dim: usize, stride_hint: usize) -> Layout {
        let rank = self.rank as usize;
        let mut new_strides = [0usize; MAX_RANK];
        new_strides[..dim].copy_from_slice(&self.strides[..dim]);
        new_strides[dim] = stride_hint;
        new_strides[dim + 1..=rank].copy_from_slice(&self.strides[dim..rank]);
        Layout {
            strides: new_strides,
            rank: (rank + 1) as u8,
            offset: self.offset,
        }
    }

    /// Remove the stride at position `dim` (for a size-1 dimension).
    pub fn squeeze(&self, dim: usize) -> Layout {
        let rank = self.rank as usize;
        let mut new_strides = [0usize; MAX_RANK];
        new_strides[..dim].copy_from_slice(&self.strides[..dim]);
        new_strides[dim..rank - 1].copy_from_slice(&self.strides[dim + 1..rank]);
        Layout {
            strides: new_strides,
            rank: (rank - 1) as u8,
            offset: self.offset,
        }
    }

    /// Advance the offset by `start * stride[dim]` and record a new
    /// extent for that dimension. Does not validate block alignment —
    /// that's the caller's (TensorDesc's) responsibility.
    pub fn slice(&self, dim: usize, start: usize) -> Layout {
        self.try_slice(dim, start)
            .expect("tensor slice layout offset overflow")
    }

    /// Checked form of [`Layout::slice`].
    ///
    /// Returns `None` when `dim` is out of range or advancing the byte offset
    /// would overflow `usize`.
    pub fn try_slice(&self, dim: usize, start: usize) -> Option<Layout> {
        if dim >= self.rank as usize {
            return None;
        }
        Some(Layout {
            strides: self.strides,
            rank: self.rank,
            offset: self
                .offset
                .checked_add(start.checked_mul(self.strides[dim])?)?,
        })
    }

    /// Total bytes spanned from `offset` to the last addressable byte.
    pub fn span_bytes(&self, block_dims: &Shape, block_byte_size: usize) -> usize {
        self.checked_span_bytes(block_dims, block_byte_size)
            .expect("tensor layout byte span overflow")
    }

    /// Checked form of [`Layout::span_bytes`].
    pub fn checked_span_bytes(&self, block_dims: &Shape, block_byte_size: usize) -> Option<usize> {
        let rank = self.rank as usize;
        if rank != block_dims.rank() {
            return None;
        }
        if rank == 0 {
            return Some(block_byte_size); // scalar: one block
        }
        if block_dims.dims().contains(&0) {
            return Some(0);
        }
        let mut max_byte = 0usize;
        for i in 0..rank {
            let extent = block_dims.dim(i as isize);
            let axis_span = (extent - 1).checked_mul(self.strides[i])?;
            max_byte = max_byte.checked_add(axis_span)?;
        }
        max_byte.checked_add(block_byte_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    // ── Contiguous construction ─────────────────────────────────

    #[test]
    fn contiguous_f32_2x3x4() {
        // F32: block_size=1, block_byte_size=4
        let dims = shape![2, 3, 4]; // block_dims = logical dims for scalar
        let layout = Layout::contiguous(&dims, 4);
        // Innermost stride = 4 (one f32)
        // Middle stride = 4 * 4 = 16
        // Outer stride = 16 * 3 = 48
        assert_eq!(layout.strides(), &[48, 16, 4]);
        assert_eq!(layout.offset(), 0);
    }

    #[test]
    fn contiguous_q4_block_dims() {
        // Q4: block_size=256, block_byte_size=144
        // Logical shape [2, 512] → block_dims [2, 2] (512/256=2 blocks)
        let block_dims = shape![2, 2];
        let layout = Layout::contiguous(&block_dims, 144);
        assert_eq!(layout.strides(), &[288, 144]); // 2*144, 144
    }

    #[test]
    fn checked_contiguous_rejects_stride_overflow() {
        assert!(Layout::try_contiguous(&shape![2, usize::MAX], 2).is_none());
        assert_eq!(
            Layout::try_contiguous(&shape![2, 3], 4).unwrap().strides(),
            &[12, 4]
        );
    }

    // ── Contiguity check ────────────────────────────────────────

    #[test]
    fn contiguous_layout_is_contiguous() {
        let dims = shape![2, 3, 4];
        let layout = Layout::contiguous(&dims, 4);
        assert!(layout.is_contiguous(&dims, 4));
    }

    #[test]
    fn transposed_layout_is_not_contiguous() {
        let dims = shape![2, 3, 4];
        let layout = Layout::contiguous(&dims, 4).transpose(0, 2);
        // After transpose, dims become [4, 3, 2] but strides are swapped
        let transposed_dims = shape![4, 3, 2];
        assert!(!layout.is_contiguous(&transposed_dims, 4));
    }

    // ── Transpose ───────────────────────────────────────────────

    #[test]
    fn transpose_swaps_strides() {
        let layout = Layout::contiguous(&shape![2, 3, 4], 4);
        let t = layout.transpose(0, 2);
        assert_eq!(t.strides(), &[4, 16, 48]); // swapped [0] and [2]
    }

    // ── Permute ─────────────────────────────────────────────────

    #[test]
    fn permute_reorders_strides() {
        let layout = Layout::contiguous(&shape![2, 3, 4], 4);
        let p = layout.permute(&[2, 0, 1]);
        assert_eq!(p.strides(), &[4, 48, 16]);
    }

    // ── Slice ───────────────────────────────────────────────────

    #[test]
    fn slice_advances_offset() {
        let layout = Layout::contiguous(&shape![4, 3], 4);
        // Slice dim 0 at start=2: offset += 2 * stride[0]
        let sliced = layout.slice(0, 2);
        assert_eq!(sliced.offset(), 2 * 12); // stride[0]=3*4=12
        assert_eq!(sliced.strides(), layout.strides()); // strides unchanged
    }

    // ── Unsqueeze / squeeze ─────────────────────────────────────

    #[test]
    fn unsqueeze_inserts_stride() {
        let layout = Layout::contiguous(&shape![3, 4], 4);
        let u = layout.unsqueeze(0, 48); // hint = total inner bytes
        assert_eq!(u.strides(), &[48, 16, 4]);
    }

    #[test]
    fn squeeze_removes_stride() {
        let layout = Layout::from_raw(&[48, 16, 4], 0);
        let s = layout.squeeze(0);
        assert_eq!(s.strides(), &[16, 4]);
    }

    // ── Span bytes ──────────────────────────────────────────────

    #[test]
    fn span_bytes_contiguous() {
        let dims = shape![2, 3, 4];
        let layout = Layout::contiguous(&dims, 4);
        // 2*3*4 = 24 elements × 4 bytes = 96
        assert_eq!(layout.span_bytes(&dims, 4), 96);
    }

    #[test]
    fn span_bytes_scalar() {
        let layout = Layout::contiguous(&Shape::SCALAR, 4);
        assert_eq!(layout.span_bytes(&Shape::SCALAR, 4), 4);
    }

    #[test]
    fn checked_span_handles_empty_and_overflow() {
        let empty = Layout::contiguous(&shape![0, 4], 4);
        assert_eq!(empty.checked_span_bytes(&shape![0, 4], 4), Some(0));

        let huge = Layout::from_raw(&[usize::MAX], 0);
        assert_eq!(huge.checked_span_bytes(&shape![2], 4), None);
    }

    #[test]
    fn checked_span_rejects_rank_mismatch_and_slice_offset_overflow() {
        let scalar_layout = Layout::from_raw(&[], 0);
        assert_eq!(scalar_layout.checked_span_bytes(&shape![2], 4), None);
        assert!(!scalar_layout.is_contiguous(&shape![2], 4));

        let layout = Layout::from_raw(&[4], usize::MAX - 1);
        assert_eq!(layout.try_slice(0, 1), None);
        assert_eq!(layout.try_slice(1, 0), None);
    }
}
