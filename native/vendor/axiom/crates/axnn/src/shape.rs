use crate::MAX_RANK;

/// A tensor shape: an ordered sequence of dimension sizes.
///
/// Stored inline — no heap allocation. Supports up to rank 8.
/// This is a value type: `Copy`, `Clone`, trivially comparable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// Scalar (rank-0 tensor).
    pub const SCALAR: Shape = Shape {
        dims: [0; MAX_RANK],
        rank: 0,
    };

    /// Construct from a pre-filled dims array and rank.
    ///
    /// Crate-internal: callers already hold a `[usize; MAX_RANK]` buffer
    /// and a known-valid rank, so no bounds check or slice copy needed.
    pub(crate) fn from_parts(dims: [usize; MAX_RANK], rank: usize) -> Self {
        Shape {
            dims,
            rank: rank as u8,
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    /// Dimension sizes as a slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank()]
    }

    /// Size of a single dimension. Supports negative indexing from the end.
    ///
    /// `shape.dim(0)` is the outermost dimension.
    /// `shape.dim(-1)` is the innermost dimension.
    pub fn dim(&self, index: isize) -> usize {
        let rank = self.rank();
        let i = if index < 0 {
            (rank as isize + index) as usize
        } else {
            index as usize
        };
        self.dims[i]
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.checked_num_elements()
            .expect("tensor shape element count overflow")
    }

    /// Total number of elements, or `None` when the product does not fit in
    /// `usize`.
    ///
    /// Graph and wire-format boundaries should use this form before deriving
    /// layouts or allocation sizes from untrusted dimensions.
    pub fn checked_num_elements(&self) -> Option<usize> {
        if self.rank == 0 {
            return Some(1); // scalar
        }
        self.dims()
            .iter()
            .try_fold(1usize, |elements, &dim| elements.checked_mul(dim))
    }

    /// Reorder dimensions according to the given axis permutation.
    ///
    /// `axes` must be a permutation of `0..rank`.
    /// Returns `None` if `axes` length doesn't match rank or contains duplicates.
    pub fn permute(&self, axes: &[usize]) -> Option<Shape> {
        let rank = self.rank();
        if axes.len() != rank {
            return None;
        }
        // Validate: each axis appears exactly once.
        let mut seen = [false; MAX_RANK];
        for &a in axes {
            if a >= rank || seen[a] {
                return None;
            }
            seen[a] = true;
        }
        let mut new_dims = [0usize; MAX_RANK];
        for (i, &a) in axes.iter().enumerate() {
            new_dims[i] = self.dims[a];
        }
        Some(Shape {
            dims: new_dims,
            rank: rank as u8,
        })
    }

    /// Insert a size-1 dimension at position `dim`.
    ///
    /// `dim` can range from `0` (before all dims) to `rank` (after all dims).
    /// Returns `None` if the resulting rank would exceed `MAX_RANK`.
    pub fn unsqueeze(&self, dim: usize) -> Option<Shape> {
        let rank = self.rank();
        if dim > rank || rank + 1 > MAX_RANK {
            return None;
        }
        let mut new_dims = [0usize; MAX_RANK];
        new_dims[..dim].copy_from_slice(&self.dims[..dim]);
        new_dims[dim] = 1;
        new_dims[dim + 1..=rank].copy_from_slice(&self.dims[dim..rank]);
        Some(Shape {
            dims: new_dims,
            rank: (rank + 1) as u8,
        })
    }

    /// Remove a size-1 dimension at position `dim`.
    ///
    /// Returns `None` if `dim` is out of range or the dimension isn't size 1.
    pub fn squeeze(&self, dim: usize) -> Option<Shape> {
        let rank = self.rank();
        if dim >= rank || self.dims[dim] != 1 {
            return None;
        }
        let mut new_dims = [0usize; MAX_RANK];
        new_dims[..dim].copy_from_slice(&self.dims[..dim]);
        new_dims[dim..rank - 1].copy_from_slice(&self.dims[dim + 1..rank]);
        Some(Shape {
            dims: new_dims,
            rank: (rank - 1) as u8,
        })
    }

    /// Whether this shape can be broadcast with `other` (NumPy-style rules).
    /// Returns the resulting broadcast shape, or `None` if incompatible.
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let out_rank = self.rank().max(other.rank());
        let mut out_dims = [0usize; MAX_RANK];

        for i in 0..out_rank {
            let a = if i < self.rank() {
                self.dims[self.rank() - 1 - i]
            } else {
                1
            };
            let b = if i < other.rank() {
                other.dims[other.rank() - 1 - i]
            } else {
                1
            };
            let d = match (a, b) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => return None,
            };
            out_dims[out_rank - 1 - i] = d;
        }

        Some(Shape {
            dims: out_dims,
            rank: out_rank as u8,
        })
    }
}

/// Convenience macro for constructing shapes inline.
///
/// ```ignore
/// let s = shape![2, 3, 4]; // rank-3, dims [2, 3, 4]
/// let s = shape![];         // scalar
/// ```
#[macro_export]
macro_rules! shape {
    () => { $crate::shape::Shape::SCALAR };
    ($($d:expr),+ $(,)?) => {
        $crate::shape::Shape::from([$($d),+])
    };
}

// ── From fixed-size arrays (infallible) ──────────────────────────

macro_rules! impl_from_array {
    ($($n:literal),*) => {
        $(
            impl From<[usize; $n]> for Shape {
                fn from(dims: [usize; $n]) -> Self {
                    let mut arr = [0usize; MAX_RANK];
                    arr[..$n].copy_from_slice(&dims);
                    Shape { dims: arr, rank: $n }
                }
            }
        )*
    };
}

impl_from_array!(0, 1, 2, 3, 4, 5, 6, 7, 8);

// ── TryFrom slice (fallible) ────────────────────────────────────

impl<'a> TryFrom<&'a [usize]> for Shape {
    type Error = ();

    /// Returns `Err(())` if the slice length exceeds `MAX_RANK`.
    fn try_from(dims: &'a [usize]) -> Result<Self, ()> {
        if dims.len() > MAX_RANK {
            return Err(());
        }
        let mut arr = [0usize; MAX_RANK];
        arr[..dims.len()].copy_from_slice(dims);
        Ok(Shape {
            dims: arr,
            rank: dims.len() as u8,
        })
    }
}

impl core::fmt::Debug for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Shape{:?}", self.dims())
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ────────────────────────────────────────────

    #[test]
    fn scalar() {
        let s = shape![];
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
        assert_eq!(s, Shape::SCALAR);
    }

    #[test]
    fn from_array() {
        let s: Shape = [2, 3, 4].into();
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn macro_construction() {
        let s = shape![8, 768, 32];
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dims(), &[8, 768, 32]);
    }

    #[test]
    fn try_from_slice_ok() {
        let v = [1usize, 2, 3];
        let s = Shape::try_from(v.as_slice()).unwrap();
        assert_eq!(s.dims(), &[1, 2, 3]);
    }

    #[test]
    fn try_from_slice_too_long() {
        let v = [0usize; 9];
        assert!(Shape::try_from(v.as_slice()).is_err());
    }

    // ── Accessors ───────────────────────────────────────────────

    #[test]
    fn negative_indexing() {
        let s = shape![2, 3, 4];
        assert_eq!(s.dim(-1), 4); // innermost
        assert_eq!(s.dim(-2), 3);
        assert_eq!(s.dim(0), 2); // outermost
    }

    #[test]
    fn num_elements() {
        assert_eq!(shape![2, 3, 4].num_elements(), 24);
        assert_eq!(shape![1].num_elements(), 1);
        assert_eq!(shape![0, 5].num_elements(), 0);
    }

    // ── Permute ─────────────────────────────────────────────────

    #[test]
    fn permute_valid() {
        let s = shape![2, 3, 4];
        let p = s.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.dims(), &[4, 2, 3]);
    }

    #[test]
    fn permute_identity() {
        let s = shape![2, 3, 4];
        assert_eq!(s.permute(&[0, 1, 2]).unwrap(), s);
    }

    #[test]
    fn permute_wrong_length() {
        assert!(shape![2, 3].permute(&[0]).is_none());
    }

    #[test]
    fn permute_duplicate_axis() {
        assert!(shape![2, 3].permute(&[0, 0]).is_none());
    }

    // ── Unsqueeze / squeeze ─────────────────────────────────────

    #[test]
    fn unsqueeze_front() {
        let s = shape![3, 4].unsqueeze(0).unwrap();
        assert_eq!(s.dims(), &[1, 3, 4]);
    }

    #[test]
    fn unsqueeze_back() {
        let s = shape![3, 4].unsqueeze(2).unwrap();
        assert_eq!(s.dims(), &[3, 4, 1]);
    }

    #[test]
    fn unsqueeze_at_max_rank_fails() {
        let s: Shape = [1; MAX_RANK].into();
        assert!(s.unsqueeze(0).is_none());
    }

    #[test]
    fn squeeze_middle() {
        let s = shape![2, 1, 4].squeeze(1).unwrap();
        assert_eq!(s.dims(), &[2, 4]);
    }

    #[test]
    fn squeeze_non_one_fails() {
        assert!(shape![2, 3].squeeze(0).is_none());
    }

    #[test]
    fn unsqueeze_squeeze_roundtrip() {
        let s = shape![2, 3];
        assert_eq!(s.unsqueeze(1).unwrap().squeeze(1).unwrap(), s);
    }

    // ── Broadcast ───────────────────────────────────────────────

    #[test]
    fn broadcast_same_shape() {
        let s = shape![2, 3];
        assert_eq!(s.broadcast_with(&s).unwrap(), s);
    }

    #[test]
    fn broadcast_expand_ones() {
        let a = shape![1, 3];
        let b = shape![2, 1];
        assert_eq!(a.broadcast_with(&b).unwrap(), shape![2, 3]);
    }

    #[test]
    fn broadcast_different_rank() {
        let a = shape![3, 4];
        let b = shape![4];
        assert_eq!(a.broadcast_with(&b).unwrap(), shape![3, 4]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(shape![2, 3].broadcast_with(&shape![4, 3]).is_none());
    }

    #[test]
    fn checked_element_count_rejects_overflow() {
        assert_eq!(shape![2, 3, 4].checked_num_elements(), Some(24));
        assert_eq!(Shape::SCALAR.checked_num_elements(), Some(1));
        assert_eq!(shape![0, usize::MAX].checked_num_elements(), Some(0));
        assert_eq!(shape![usize::MAX, 2].checked_num_elements(), None);
    }

    // ── Display ─────────────────────────────────────────────────

    #[test]
    fn display_format() {
        use alloc::format;
        assert_eq!(format!("{}", shape![2, 3, 4]), "[2, 3, 4]");
        assert_eq!(format!("{}", shape![]), "[]");
    }
}
