//! The reachability walk's visited set: one bit per directory-arena slot.
//!
//! The cycle proof before a directory edge is published walks the
//! directory subgraph and must know, per node, whether it has been seen.
//! A visited *list* answers that in O(n) per query and O(n²) per walk; a
//! bit row answers it in O(1) and iterates in O(popcount). The row is keyed
//! by the directory's arena slot index, which is exactly the thing that is
//! bounded — [`MAX_DIRECTORIES`] — so the set's capacity is the arena's and
//! an index at or past it is refused, never truncated.
//!
//! Walk-local by construction: a `ReachSet` lives on the stack of the one
//! walk that made it and is never a static. Scalar words on purpose — the
//! house ruling in `kore_collections::bitmat` is scalar-first with
//! auto-vectorisation, and a row of one `u128` has nothing for a SIMD lane
//! to do; if the arena ever grows to where a kernel is measured to matter,
//! it goes into `bitmat`, not here.

use kore_collections::BitSet;

use crate::{
    bounds::MAX_DIRECTORIES,
    error::{FsError, FsResult},
};

/// Storage words for the directory arena: `BitSet` words are 128 bits.
pub(crate) const REACH_WORDS: usize = MAX_DIRECTORIES.div_ceil(128);

/// A walk-local set of directory-arena slot indices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReachSet(BitSet<REACH_WORDS>);

impl ReachSet {
    /// The logical bound: the directory arena, not the word row's slack.
    pub(crate) const CAPACITY: usize = MAX_DIRECTORIES;
    /// The row's footprint in bytes — `bounds` pins it.
    pub(crate) const BYTES: usize = REACH_WORDS * 16;

    pub(crate) const fn new() -> Self {
        Self(BitSet::new())
    }

    /// O(1). An index beyond the arena is simply not a member.
    pub(crate) const fn contains(&self, index: usize) -> bool {
        index < Self::CAPACITY && self.0.contains(index)
    }

    /// Mark `index`; `Ok(true)` when it was not yet a member. An index at or
    /// past the arena refuses [`FsError::Range`]: the walk cannot describe a
    /// directory the arena cannot hold, and silently dropping it would let a
    /// cycle through that directory go unproven.
    pub(crate) fn insert(&mut self, index: usize) -> FsResult<bool> {
        if index >= Self::CAPACITY {
            return Err(FsError::Range);
        }
        Ok(self.0.set(index))
    }

    /// Members marked so far.
    pub(crate) fn len(&self) -> usize {
        self.0.count_ones()
    }

    /// Members in ascending index order, one step per set bit.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_is_exact_and_insert_reports_novelty() {
        let mut set = ReachSet::new();
        assert!(!set.contains(3));
        assert_eq!(set.insert(3), Ok(true));
        assert!(set.contains(3));
        assert_eq!(set.insert(3), Ok(false));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn iteration_is_ascending_and_costs_one_step_per_set_bit() {
        let mut set = ReachSet::new();
        for index in [9, 0, MAX_DIRECTORIES - 1, 4] {
            set.insert(index).unwrap();
        }
        let seen: std::vec::Vec<usize> = set.iter().collect();
        assert_eq!(seen, [0, 4, 9, MAX_DIRECTORIES - 1]);
        assert_eq!(seen.len(), set.len());
    }

    #[test]
    fn an_index_beyond_the_directory_arena_is_refused_not_truncated() {
        let mut set = ReachSet::new();
        assert_eq!(set.insert(MAX_DIRECTORIES), Err(FsError::Range));
        assert!(!set.contains(MAX_DIRECTORIES));
        // The row has slack past the arena (128 bits per word); the bound is
        // the arena's, not the row's.
        assert_eq!(set.insert(ReachSet::CAPACITY + 1), Err(FsError::Range));
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn footprint_is_one_word_per_128_directories() {
        assert_eq!(ReachSet::BYTES, REACH_WORDS * 16);
        assert_eq!(ReachSet::CAPACITY, MAX_DIRECTORIES);
        assert!(BitSet::<REACH_WORDS>::BITS >= MAX_DIRECTORIES);
        assert_eq!(core::mem::size_of::<ReachSet>(), ReachSet::BYTES);
    }
}
