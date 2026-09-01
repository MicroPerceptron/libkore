//! Bit-matrix machinery: word-parallel row kernels + a flat row-major
//! bit matrix.
//!
//! The kernels operate on `u128` word slices (the house word size,
//! matching [`BitVec`](super::BitVec)'s storage) and are the building
//! blocks for set-algebra over densely-indexed id spaces: union /
//! intersection / subtraction / disjointness in `O(words)` per row.
//! Layout ideas are borrowed from the axnn portable bit-matrix; the
//! kernels here are deliberately plain-word loops rather than explicit
//! SIMD — they must run in *any* kernel context, including early boot
//! before extended state (and therefore vector registers) is enabled,
//! and LLVM auto-vectorizes them under the hard-float kernel target.
//!
//! [`BitMat`] is the matrix: `rows × cols` bits, row-major, flat
//! allocator-backed storage, no summary tiers (row operations are linear
//! sweeps; summaries only pay for sparse point queries — that is
//! [`BitVec`](super::BitVec)'s niche).  Rows interoperate with `BitVec`
//! through its read-only [`as_data_words`](super::BitVec::as_data_words)
//! view; the mutating side of any cross-op must be a `BitMat` row, since
//! `BitVec`'s summaries cannot survive external word writes.
//!
//! # Tail invariant
//!
//! When `cols` is not a multiple of 128 the last word of each row has
//! trailing slack bits.  Every mutator here masks to `cols`, so slack
//! bits are always zero — kernels and [`popcount`] rely on that and
//! preserve it (`or`/`and`/`andn` of zero-tailed rows are zero-tailed).

use core::alloc::Allocator;

use alloc::vec::Vec;

/// Bits per storage word.
const WORD_BITS: usize = 128;

/// Words needed for `bits` bits.
#[inline]
const fn words_for(bits: usize) -> usize {
    bits.div_ceil(WORD_BITS)
}

// =============================================================================
// Row kernels — free functions over u128 word slices
// =============================================================================

/// `dst |= src`, word-wise.  Panics if lengths differ (debug).
#[inline]
pub fn or_into(dst: &mut [u128], src: &[u128]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

/// `dst &= src`, word-wise.
#[inline]
pub fn and_into(dst: &mut [u128], src: &[u128]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= *s;
    }
}

/// `dst &= !mask`, word-wise (subtract `mask`'s set bits from `dst`).
#[inline]
pub fn andn_into(dst: &mut [u128], mask: &[u128]) {
    debug_assert_eq!(dst.len(), mask.len());
    for (d, m) in dst.iter_mut().zip(mask) {
        *d &= !*m;
    }
}

/// Whether every bit is clear.
#[inline]
pub fn is_zero(row: &[u128]) -> bool {
    row.iter().all(|w| *w == 0)
}

/// Number of set bits.
#[inline]
pub fn popcount(row: &[u128]) -> usize {
    row.iter().map(|w| w.count_ones() as usize).sum()
}

/// Whether `a ∩ b = ∅`, without materialising the intersection — the
/// KRAPH §10.6 conflict predicate is `!is_disjoint(write, read ∪ write)`.
#[inline]
pub fn is_disjoint(a: &[u128], b: &[u128]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).all(|(x, y)| x & y == 0)
}

/// Invoke `f(index)` for every set bit below `limit`, ascending.  Skips
/// zero words; per set bit cost is one `trailing_zeros` + clear-lowest.
pub fn for_each_set(row: &[u128], limit: usize, mut f: impl FnMut(usize)) {
    for (wi, &w) in row.iter().enumerate() {
        let mut word = w;
        let base = wi * WORD_BITS;
        if base >= limit {
            return;
        }
        while word != 0 {
            let bit = base + word.trailing_zeros() as usize;
            if bit >= limit {
                return;
            }
            f(bit);
            word &= word - 1;
        }
    }
}

// =============================================================================
// BitMat
// =============================================================================

/// A `rows × cols` bit matrix: flat row-major `u128` storage, each row
/// `cols` bits wide.  Single-threaded — share behind a lock
/// ([`SpinLock`](crate::sync::SpinLock)) like any other plain collection.
#[derive(Debug)]
pub struct BitMat<'a, A: Allocator + 'a> {
    storage: Vec<u128, &'a A>,
    rows: usize,
    cols: usize,
    /// Words per row (`cols` rounded up to whole words).
    row_words: usize,
}

impl<'a, A: Allocator + 'a> BitMat<'a, A> {
    /// Build a zeroed `rows × cols` matrix backed by `alloc`.
    ///
    /// # Panics
    ///
    /// Panics if `alloc` cannot satisfy the storage allocation.
    pub fn new_in(rows: usize, cols: usize, alloc: &'a A) -> Self {
        let row_words = words_for(cols);
        let total = rows * row_words;
        let mut storage = Vec::with_capacity_in(total, alloc);
        storage.resize(total, 0u128);
        Self {
            storage,
            rows,
            cols,
            row_words,
        }
    }

    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn offsets(&self, row: usize, col: usize) -> (usize, u128) {
        debug_assert!(row < self.rows && col < self.cols);
        (
            row * self.row_words + col / WORD_BITS,
            1u128 << (col % WORD_BITS),
        )
    }

    /// Set bit `(row, col)`; returns whether it was previously set.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize) -> bool {
        let (w, mask) = self.offsets(row, col);
        let prior = self.storage[w] & mask != 0;
        self.storage[w] |= mask;
        prior
    }

    /// Clear bit `(row, col)`; returns whether it was previously set.
    #[inline]
    pub fn clear(&mut self, row: usize, col: usize) -> bool {
        let (w, mask) = self.offsets(row, col);
        let prior = self.storage[w] & mask != 0;
        self.storage[w] &= !mask;
        prior
    }

    #[inline]
    pub fn contains(&self, row: usize, col: usize) -> bool {
        let (w, mask) = self.offsets(row, col);
        self.storage[w] & mask != 0
    }

    /// The words of `row`, for the read side of kernel ops.
    #[inline]
    pub fn row(&self, row: usize) -> &[u128] {
        debug_assert!(row < self.rows);
        &self.storage[row * self.row_words..(row + 1) * self.row_words]
    }

    /// The words of `row`, mutably — the destination side of kernel ops.
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [u128] {
        debug_assert!(row < self.rows);
        &mut self.storage[row * self.row_words..(row + 1) * self.row_words]
    }

    /// Disjoint `(dst, src)` row borrow for intra-matrix kernel ops,
    /// e.g. `or_into(dst, src)` between two rows of the same matrix.
    ///
    /// # Panics
    ///
    /// Panics if `dst == src`.
    pub fn rows_mut(&mut self, dst: usize, src: usize) -> (&mut [u128], &[u128]) {
        assert_ne!(dst, src, "rows_mut requires distinct rows");
        debug_assert!(dst < self.rows && src < self.rows);
        let rw = self.row_words;
        if dst < src {
            let (lo, hi) = self.storage.split_at_mut(src * rw);
            (&mut lo[dst * rw..(dst + 1) * rw], &hi[..rw])
        } else {
            let (lo, hi) = self.storage.split_at_mut(dst * rw);
            (&mut hi[..rw], &lo[src * rw..(src + 1) * rw])
        }
    }

    /// Clear every bit of `row`.
    #[inline]
    pub fn clear_row(&mut self, row: usize) {
        self.row_mut(row).fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::TestArena;

    static TEST_ARENA: TestArena<{ 1 << 16 }> = TestArena::new();

    fn arena() -> &'static TestArena<{ 1 << 16 }> {
        &TEST_ARENA
    }

    #[test]
    fn set_contains_clear_round_trip() {
        let mut m = BitMat::new_in(4, 200, arena());
        assert!(!m.set(2, 150), "first set reports previously-clear");
        assert!(m.contains(2, 150));
        assert!(m.set(2, 150), "second set reports previously-set");
        assert!(m.clear(2, 150));
        assert!(!m.contains(2, 150));
        assert!(!m.clear(2, 150), "clearing a clear bit reports clear");
        // Neighbours untouched.
        assert!(!m.contains(2, 149) && !m.contains(2, 151) && !m.contains(1, 150));
    }

    #[test]
    fn kernels_compute_set_algebra() {
        let mut m = BitMat::new_in(3, 300, arena());
        for c in [0, 127, 128, 200, 299] {
            m.set(0, c);
        }
        for c in [127, 200, 250] {
            m.set(1, c);
        }

        // Union: row2 = row0 | row1.
        let (dst, src) = m.rows_mut(2, 0);
        or_into(dst, src);
        let (dst, src) = m.rows_mut(2, 1);
        or_into(dst, src);
        assert_eq!(popcount(m.row(2)), 6, "union of {{5}} and {{3}} sharing 2");

        // Intersection: row2 &= row1 → exactly row1's overlap with row0∪row1.
        let (dst, src) = m.rows_mut(2, 1);
        and_into(dst, src);
        assert_eq!(popcount(m.row(2)), 3);

        // Subtraction: row2 &= !row0 → row1-only bits.
        let (dst, src) = m.rows_mut(2, 0);
        andn_into(dst, src);
        assert_eq!(popcount(m.row(2)), 1, "only 250 is row1-exclusive");
        assert!(m.contains(2, 250));

        // Disjointness — the OCC conflict shape.
        assert!(!is_disjoint(m.row(0), m.row(1)), "127 and 200 are shared");
        assert!(
            is_disjoint(m.row(2), m.row(0)),
            "the subtraction removed all overlap"
        );
        assert!(!is_zero(m.row(2)));
        m.clear_row(2);
        assert!(is_zero(m.row(2)));
    }

    #[test]
    fn for_each_set_ascends_and_respects_limit() {
        let mut m = BitMat::new_in(1, 400, arena());
        for c in [3, 129, 256, 390] {
            m.set(0, c);
        }
        let mut seen = alloc::vec::Vec::new();
        for_each_set(m.row(0), 400, |b| seen.push(b));
        assert_eq!(seen, [3, 129, 256, 390]);

        seen.clear();
        for_each_set(m.row(0), 257, |b| seen.push(b));
        assert_eq!(
            seen,
            [3, 129, 256],
            "limit is exclusive of 390, inclusive of 256"
        );
    }

    #[test]
    fn rows_mut_splits_in_both_directions() {
        let mut m = BitMat::new_in(2, 130, arena());
        m.set(0, 129);
        let (dst, src) = m.rows_mut(1, 0); // dst above src
        or_into(dst, src);
        assert!(m.contains(1, 129));
        m.set(1, 5);
        let (dst, src) = m.rows_mut(0, 1); // dst below src
        or_into(dst, src);
        assert!(m.contains(0, 5));
    }

    #[test]
    #[should_panic(expected = "distinct rows")]
    fn rows_mut_rejects_aliasing() {
        let mut m = BitMat::new_in(2, 64, arena());
        let _ = m.rows_mut(1, 1);
    }

    #[test]
    fn interops_with_bitvec_read_view() {
        use crate::BitVec;
        // A BitVec mask ANDs into a BitMat row through the read-only view.
        let a = arena();
        let mut live = BitVec::new_in(200, a);
        live.set(10);
        live.set(150);
        let mut m = BitMat::new_in(1, 200, a);
        for c in [10, 20, 150] {
            m.set(0, c);
        }
        and_into(m.row_mut(0), live.as_data_words());
        assert_eq!(popcount(m.row(0)), 2, "20 was not live");
        assert!(m.contains(0, 10) && m.contains(0, 150) && !m.contains(0, 20));
    }
}
