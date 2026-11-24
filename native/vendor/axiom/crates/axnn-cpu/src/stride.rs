use axnn::{Layout, MAX_RANK, Shape};

/// Iterator that yields byte offsets for each block in row-major logical order.
///
/// For a tensor with `block_dims` [D0, D1, …, Dn-1] and byte strides
/// [S0, S1, …, Sn-1] plus a base offset, this produces:
///
/// ```text
/// offset + i0*S0 + i1*S1 + … + in*Sn
/// ```
///
/// for all `(i0, i1, …, in)` swept in row-major (C) order.
///
/// Handles rank-0 (scalar) as a single block at the base offset.
pub(crate) struct StrideIter {
    block_dims: [usize; MAX_RANK],
    strides: [usize; MAX_RANK],
    rank: usize,
    base_offset: usize,
    /// Current multi-index (odometer).
    index: [usize; MAX_RANK],
    total: usize,
    count: usize,
}

impl StrideIter {
    /// Create from a block-level shape and the tensor's layout.
    ///
    /// `block_dims` is the physical block shape (innermost dim divided
    /// by block_size). Obtain it via `dtype.block_shape(&shape)`.
    pub fn new(block_dims: &Shape, layout: &Layout) -> Self {
        let rank = block_dims.rank();
        let mut dims = [0usize; MAX_RANK];
        dims[..rank].copy_from_slice(block_dims.dims());
        let mut strides = [0usize; MAX_RANK];
        strides[..rank].copy_from_slice(layout.strides());

        let total = if rank == 0 {
            1 // scalar: one block
        } else {
            block_dims.num_elements()
        };

        StrideIter {
            block_dims: dims,
            strides,
            rank,
            base_offset: layout.offset(),
            index: [0; MAX_RANK],
            total,
            count: 0,
        }
    }
}

impl Iterator for StrideIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.count >= self.total {
            return None;
        }

        // Compute byte offset for the current multi-index.
        let mut offset = self.base_offset;
        for d in 0..self.rank {
            offset += self.index[d] * self.strides[d];
        }

        // Advance the odometer (innermost dimension first).
        self.count += 1;
        for d in (0..self.rank).rev() {
            self.index[d] += 1;
            if self.index[d] < self.block_dims[d] {
                break;
            }
            self.index[d] = 0;
        }

        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.count;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for StrideIter {}

#[cfg(test)]
mod tests {
    use super::*;
    use axnn::{DType, Layout, shape};

    #[test]
    fn contiguous_offsets_sequential() {
        // 2×3 F32 (block_byte_size=4), contiguous layout.
        // Offsets should be 0, 4, 8, 12, 16, 20.
        let shape = shape![2, 3];
        let block_dims = DType::F32.block_shape(&shape);
        let layout = Layout::contiguous(&block_dims, DType::F32.block_byte_size());

        let offsets: alloc::vec::Vec<_> = StrideIter::new(&block_dims, &layout).collect();
        assert_eq!(offsets, alloc::vec![0, 4, 8, 12, 16, 20]);
    }

    #[test]
    fn transposed_offsets() {
        // 2×3 F32 contiguous, then transposed to 3×2.
        // Original strides: [12, 4]. After transpose(0,1): [4, 12].
        // Row-major sweep of 3×2 with strides [4, 12]:
        //   (0,0)=0, (0,1)=12, (1,0)=4, (1,1)=16, (2,0)=8, (2,1)=20
        let shape = shape![2, 3];
        let block_dims = DType::F32.block_shape(&shape);
        let layout = Layout::contiguous(&block_dims, DType::F32.block_byte_size());
        let t_layout = layout.transpose(0, 1);
        let t_block_dims = block_dims.permute(&[1, 0]).unwrap();

        let offsets: alloc::vec::Vec<_> = StrideIter::new(&t_block_dims, &t_layout).collect();
        assert_eq!(offsets, alloc::vec![0, 12, 4, 16, 8, 20]);
    }

    #[test]
    fn scalar_single_offset() {
        // Rank-0 tensor: one block at offset 0.
        let block_dims = DType::F32.block_shape(&shape![]);
        let layout = Layout::contiguous(&block_dims, DType::F32.block_byte_size());

        let offsets: alloc::vec::Vec<_> = StrideIter::new(&block_dims, &layout).collect();
        assert_eq!(offsets, alloc::vec![0]);
    }

    #[test]
    fn exact_size() {
        let shape = shape![4, 8];
        let block_dims = DType::F32.block_shape(&shape);
        let layout = Layout::contiguous(&block_dims, DType::F32.block_byte_size());
        let iter = StrideIter::new(&block_dims, &layout);
        assert_eq!(iter.len(), 32);
    }
}
