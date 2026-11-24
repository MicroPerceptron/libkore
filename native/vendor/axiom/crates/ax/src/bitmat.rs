//! Allocation-free bit vectors and square bit matrices used by the scheduler.
//!
//! Storage is supplied by the caller as `u64` words. This keeps the scheduler
//! usable in kernels and embedded/userland environments while allowing graph
//! size to be selected at runtime.

use core::simd::Simd;

// One native register under the maximum-performance builds used by axiom:
// four u64 lanes on AVX2, two on NEON, and a portable 128-bit fallback.
#[cfg(target_arch = "x86_64")]
const WORD_LANES: usize = 4;
#[cfg(not(target_arch = "x86_64"))]
const WORD_LANES: usize = 2;

/// Words required to store `bits` bits.
pub const fn words_for(bits: usize) -> usize {
    bits.div_ceil(64)
}

/// Words required for an `nodes × nodes` matrix, or `None` on overflow.
pub const fn matrix_words(nodes: usize) -> Option<usize> {
    nodes.checked_mul(words_for(nodes))
}

/// Words required for a `rows × columns` bit table, or `None` on overflow.
pub const fn table_words(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(words_for(columns))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitMatrixError {
    SizeOverflow,
    StorageTooSmall { needed: usize, actual: usize },
    OutOfBounds,
}

/// Borrowed immutable square bit matrix.
#[derive(Clone, Copy)]
pub struct BitMatrixRef<'a> {
    words: &'a [u64],
    nodes: usize,
    row_words: usize,
}

/// Borrowed immutable rectangular bit table.
#[derive(Clone, Copy)]
pub struct BitTableRef<'a> {
    words: &'a [u64],
    rows: usize,
    columns: usize,
    row_words: usize,
}

impl<'a> BitTableRef<'a> {
    pub fn new(words: &'a [u64], rows: usize, columns: usize) -> Result<Self, BitMatrixError> {
        let needed = table_words(rows, columns).ok_or(BitMatrixError::SizeOverflow)?;
        if words.len() < needed {
            return Err(BitMatrixError::StorageTooSmall {
                needed,
                actual: words.len(),
            });
        }
        Ok(Self {
            words: &words[..needed],
            rows,
            columns,
            row_words: words_for(columns),
        })
    }

    pub fn get(&self, row: usize, column: usize) -> Result<bool, BitMatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(BitMatrixError::OutOfBounds);
        }
        let index = row * self.row_words + column / 64;
        Ok(self.words[index] & (1_u64 << (column % 64)) != 0)
    }

    pub fn row(&self, row: usize) -> Result<&'a [u64], BitMatrixError> {
        if row >= self.rows {
            return Err(BitMatrixError::OutOfBounds);
        }
        let start = row * self.row_words;
        Ok(&self.words[start..start + self.row_words])
    }
}

/// Mutable bit set over caller-provided words.
pub struct BitSetMut<'a> {
    words: &'a mut [u64],
    bits: usize,
}

impl<'a> BitSetMut<'a> {
    pub fn new(words: &'a mut [u64], bits: usize) -> Result<Self, BitMatrixError> {
        let needed = words_for(bits);
        if words.len() < needed {
            return Err(BitMatrixError::StorageTooSmall {
                needed,
                actual: words.len(),
            });
        }
        let words = &mut words[..needed];
        words.fill(0);
        Ok(Self { words, bits })
    }

    pub fn set(&mut self, bit: usize) -> Result<bool, BitMatrixError> {
        if bit >= self.bits {
            return Err(BitMatrixError::OutOfBounds);
        }
        let mask = 1_u64 << (bit % 64);
        let word = &mut self.words[bit / 64];
        let was_set = *word & mask != 0;
        *word |= mask;
        Ok(!was_set)
    }

    pub fn clear(&mut self, bit: usize) -> Result<bool, BitMatrixError> {
        if bit >= self.bits {
            return Err(BitMatrixError::OutOfBounds);
        }
        let mask = 1_u64 << (bit % 64);
        let word = &mut self.words[bit / 64];
        let was_set = *word & mask != 0;
        *word &= !mask;
        Ok(was_set)
    }

    pub fn first(&self) -> Option<usize> {
        self.first_where(|_| true)
    }

    /// Finds the lowest set bit accepted by `predicate`.
    ///
    /// Sparse word iteration avoids the scheduler's former full-order scan
    /// while retaining stable topological tie-breaking.
    pub fn first_where(&self, mut predicate: impl FnMut(usize) -> bool) -> Option<usize> {
        for (word_index, &stored) in self.words.iter().enumerate() {
            let mut word = stored;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if index >= self.bits {
                    return None;
                }
                if predicate(index) {
                    return Some(index);
                }
                word &= word - 1;
            }
        }
        None
    }
}

impl<'a> BitMatrixRef<'a> {
    pub fn new(words: &'a [u64], nodes: usize) -> Result<Self, BitMatrixError> {
        let needed = matrix_words(nodes).ok_or(BitMatrixError::SizeOverflow)?;
        if words.len() < needed {
            return Err(BitMatrixError::StorageTooSmall {
                needed,
                actual: words.len(),
            });
        }
        Ok(Self {
            words: &words[..needed],
            nodes,
            row_words: words_for(nodes),
        })
    }

    pub const fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn row(&self, row: usize) -> Result<&'a [u64], BitMatrixError> {
        if row >= self.nodes {
            return Err(BitMatrixError::OutOfBounds);
        }
        let start = row * self.row_words;
        Ok(&self.words[start..start + self.row_words])
    }

    pub fn get(&self, row: usize, column: usize) -> Result<bool, BitMatrixError> {
        if column >= self.nodes {
            return Err(BitMatrixError::OutOfBounds);
        }
        let words = self.row(row)?;
        Ok(words[column / 64] & (1_u64 << (column % 64)) != 0)
    }

    pub fn for_each_set_bit(
        &self,
        row: usize,
        mut visit: impl FnMut(usize),
    ) -> Result<(), BitMatrixError> {
        for (word_index, &stored) in self.row(row)?.iter().enumerate() {
            let mut word = stored;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let node = word_index * 64 + bit;
                if node >= self.nodes {
                    break;
                }
                visit(node);
                word &= word - 1;
            }
        }
        Ok(())
    }
}

/// Mutable square bit matrix over caller-provided storage.
pub struct BitMatrixMut<'a> {
    words: &'a mut [u64],
    nodes: usize,
    row_words: usize,
}

/// Mutable rectangular bit table over caller-provided storage.
pub struct BitTableMut<'a> {
    words: &'a mut [u64],
    rows: usize,
    columns: usize,
    row_words: usize,
}

impl<'a> BitTableMut<'a> {
    pub fn new(words: &'a mut [u64], rows: usize, columns: usize) -> Result<Self, BitMatrixError> {
        let needed = table_words(rows, columns).ok_or(BitMatrixError::SizeOverflow)?;
        if words.len() < needed {
            return Err(BitMatrixError::StorageTooSmall {
                needed,
                actual: words.len(),
            });
        }
        let words = &mut words[..needed];
        words.fill(0);
        Ok(Self {
            words,
            rows,
            columns,
            row_words: words_for(columns),
        })
    }

    pub fn get(&self, row: usize, column: usize) -> Result<bool, BitMatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(BitMatrixError::OutOfBounds);
        }
        let index = row * self.row_words + column / 64;
        Ok(self.words[index] & (1_u64 << (column % 64)) != 0)
    }

    pub fn as_ref(&self) -> BitTableRef<'_> {
        BitTableRef {
            words: self.words,
            rows: self.rows,
            columns: self.columns,
            row_words: self.row_words,
        }
    }

    pub fn set(&mut self, row: usize, column: usize) -> Result<bool, BitMatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(BitMatrixError::OutOfBounds);
        }
        let index = row * self.row_words + column / 64;
        let mask = 1_u64 << (column % 64);
        let was_set = self.words[index] & mask != 0;
        self.words[index] |= mask;
        Ok(!was_set)
    }
}

impl<'a> BitMatrixMut<'a> {
    pub fn new(words: &'a mut [u64], nodes: usize) -> Result<Self, BitMatrixError> {
        let needed = matrix_words(nodes).ok_or(BitMatrixError::SizeOverflow)?;
        if words.len() < needed {
            return Err(BitMatrixError::StorageTooSmall {
                needed,
                actual: words.len(),
            });
        }
        let words = &mut words[..needed];
        words.fill(0);
        Ok(Self {
            words,
            nodes,
            row_words: words_for(nodes),
        })
    }

    pub fn as_ref(&self) -> BitMatrixRef<'_> {
        BitMatrixRef {
            words: self.words,
            nodes: self.nodes,
            row_words: self.row_words,
        }
    }

    pub fn set(&mut self, row: usize, column: usize) -> Result<bool, BitMatrixError> {
        if row >= self.nodes || column >= self.nodes {
            return Err(BitMatrixError::OutOfBounds);
        }
        let index = row * self.row_words + column / 64;
        let mask = 1_u64 << (column % 64);
        let was_set = self.words[index] & mask != 0;
        self.words[index] |= mask;
        Ok(!was_set)
    }

    pub fn row_mut(&mut self, row: usize) -> Result<&mut [u64], BitMatrixError> {
        if row >= self.nodes {
            return Err(BitMatrixError::OutOfBounds);
        }
        let start = row * self.row_words;
        Ok(&mut self.words[start..start + self.row_words])
    }
}

pub fn or_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    let chunks = dst.len() / WORD_LANES;
    for chunk in 0..chunks {
        let start = chunk * WORD_LANES;
        let value = Simd::<u64, WORD_LANES>::from_slice(&dst[start..])
            | Simd::<u64, WORD_LANES>::from_slice(&src[start..]);
        value.copy_to_slice(&mut dst[start..start + WORD_LANES]);
    }
    for (dst, src) in dst[chunks * WORD_LANES..]
        .iter_mut()
        .zip(&src[chunks * WORD_LANES..])
    {
        *dst |= *src;
    }
}

pub fn and_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    let chunks = dst.len() / WORD_LANES;
    for chunk in 0..chunks {
        let start = chunk * WORD_LANES;
        let value = Simd::<u64, WORD_LANES>::from_slice(&dst[start..])
            & Simd::<u64, WORD_LANES>::from_slice(&src[start..]);
        value.copy_to_slice(&mut dst[start..start + WORD_LANES]);
    }
    for (dst, src) in dst[chunks * WORD_LANES..]
        .iter_mut()
        .zip(&src[chunks * WORD_LANES..])
    {
        *dst &= *src;
    }
}

pub fn and_not_into(dst: &mut [u64], mask: &[u64]) {
    debug_assert_eq!(dst.len(), mask.len());
    let chunks = dst.len() / WORD_LANES;
    for chunk in 0..chunks {
        let start = chunk * WORD_LANES;
        let value = Simd::<u64, WORD_LANES>::from_slice(&dst[start..])
            & !Simd::<u64, WORD_LANES>::from_slice(&mask[start..]);
        value.copy_to_slice(&mut dst[start..start + WORD_LANES]);
    }
    for (dst, mask) in dst[chunks * WORD_LANES..]
        .iter_mut()
        .zip(&mask[chunks * WORD_LANES..])
    {
        *dst &= !*mask;
    }
}

pub fn is_zero(words: &[u64]) -> bool {
    let zero = Simd::<u64, WORD_LANES>::splat(0);
    let chunks = words.len() / WORD_LANES;
    for chunk in 0..chunks {
        let start = chunk * WORD_LANES;
        if Simd::<u64, WORD_LANES>::from_slice(&words[start..]) != zero {
            return false;
        }
    }
    words[chunks * WORD_LANES..].iter().all(|word| *word == 0)
}

pub fn popcount(words: &[u64]) -> u32 {
    words.iter().map(|word| word.count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_matrix_crosses_word_boundaries() {
        let mut words = [0_u64; 3 * 3];
        assert!(matches!(
            BitMatrixMut::new(&mut words, 130),
            Err(BitMatrixError::StorageTooSmall { .. })
        ));

        let mut words = [0_u64; 390];
        let mut matrix = BitMatrixMut::new(&mut words, 130).unwrap();
        assert!(matrix.set(2, 129).unwrap());
        assert!(!matrix.set(2, 129).unwrap());
        assert!(matrix.as_ref().get(2, 129).unwrap());

        let nodes = 1_025;
        let mut words = std::vec![0_u64; matrix_words(nodes).unwrap()];
        let mut matrix = BitMatrixMut::new(&mut words, nodes).unwrap();
        matrix.set(1_024, 1_024).unwrap();
        assert!(matrix.as_ref().get(1_024, 1_024).unwrap());
    }

    #[test]
    fn row_algebra() {
        let mut a = [0b1010_u64, 0b0101];
        let b = [0b1100_u64, 0b0011];
        and_into(&mut a, &b);
        assert_eq!(a, [0b1000, 0b0001]);
        or_into(&mut a, &b);
        assert_eq!(a, b);
        and_not_into(&mut a, &[0b0100, 0b0001]);
        assert_eq!(a, [0b1000, 0b0010]);
        assert_eq!(popcount(&a), 2);
    }

    #[test]
    fn rectangular_table_ref_preserves_backend_rows() {
        let mut words = [0_u64; 4];
        {
            let mut table = BitTableMut::new(&mut words, 2, 70).unwrap();
            table.set(0, 69).unwrap();
            table.set(1, 3).unwrap();
        }
        let table = BitTableRef::new(&words, 2, 70).unwrap();
        assert!(table.get(0, 69).unwrap());
        assert!(table.get(1, 3).unwrap());
        assert!(!table.get(0, 3).unwrap());
        assert_eq!(table.row(0).unwrap().len(), 2);
    }

    #[test]
    fn sparse_frontier_finds_stable_matching_position() {
        let mut words = [0_u64; 3];
        let mut ready = BitSetMut::new(&mut words, 130).unwrap();
        ready.set(97).unwrap();
        ready.set(4).unwrap();
        ready.set(65).unwrap();
        assert_eq!(ready.first(), Some(4));
        assert_eq!(ready.first_where(|position| position > 64), Some(65));
        assert!(ready.clear(4).unwrap());
        assert_eq!(ready.first(), Some(65));
        assert!(!ready.clear(4).unwrap());
    }
}
