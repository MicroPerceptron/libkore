//! **Grants** — the authority a ring's submissions are checked against
//! (`res/STORAGE_TRANSPORT.md` §5).
//!
//! A grant names up to [`MAX_RUNS`] extents of the device, a set of rights,
//! and the classes a submission against it may ride. Its byte offsets are
//! *logical*: offset 0 is the first byte of the first run, and a range that
//! crosses a run boundary continues at the next run. The filesystem installs
//! one per open file (the file's placement), the pack partition is one READ
//! grant, and the filesystem's own metadata region is one more.
//!
//! [`Grant::segment`] is the pack lane's region clamp and rebase, generalized
//! from one extent to a list: it answers where the next piece of a range
//! lives on the device, never past a run's end or the transfer ceiling.

use bitflags::bitflags;
use kore_abi::stor::Class;

/// Extents one grant may name.
pub const MAX_RUNS: usize = 8;

bitflags! {
    /// What a submission against a grant may do.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Rights: u8 {
        /// Read into a target.
        const READ = 1 << 0;
        /// Write from a target.
        const WRITE = 1 << 1;
        /// Flush the device's volatile cache.
        const FLUSH = 1 << 2;
        /// Discard a range.
        const DISCARD = 1 << 3;
    }
}

/// A set of [`Class`]es.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassSet(u8);

impl ClassSet {
    /// No class.
    pub const NONE: Self = Self(0);
    /// Every class.
    pub const ALL: Self = Self(0b111);
    /// Demand and Stream — what an ordinary client may ride.
    pub const DATA: Self = Self(0b011);

    /// The set holding exactly `class`.
    pub const fn only(class: Class) -> Self {
        Self(1 << class as u8)
    }

    /// This set plus `class`.
    #[must_use]
    pub const fn with(self, class: Class) -> Self {
        Self(self.0 | (1 << class as u8))
    }

    /// Whether `class` is in the set.
    pub const fn contains(self, class: Class) -> bool {
        self.0 & (1 << class as u8) != 0
    }
}

/// One device extent: `blocks` logical blocks starting at `lba`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtentRun {
    /// First logical block.
    pub lba: u64,
    /// Blocks it spans.
    pub blocks: u32,
}

/// Why a grant could not be built or a range could not resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantError {
    /// No runs, more than [`MAX_RUNS`], or an empty run.
    Shape,
    /// A run's end overflows the block address space.
    Overflow,
}

/// A grant as its authority installs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grant {
    runs: [ExtentRun; MAX_RUNS],
    count: u8,
    rights: Rights,
    classes: ClassSet,
}

/// Where the next piece of a range lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// First device block.
    pub lba: u64,
    /// Blocks in this piece.
    pub blocks: u32,
}

impl Grant {
    /// A grant over `runs` with `rights`, admitting `classes`.
    pub fn new(runs: &[ExtentRun], rights: Rights, classes: ClassSet) -> Result<Self, GrantError> {
        if runs.is_empty() || runs.len() > MAX_RUNS || runs.iter().any(|run| run.blocks == 0) {
            return Err(GrantError::Shape);
        }
        if runs
            .iter()
            .any(|run| run.lba.checked_add(u64::from(run.blocks)).is_none())
        {
            return Err(GrantError::Overflow);
        }
        let mut stored = [ExtentRun::default(); MAX_RUNS];
        stored[..runs.len()].copy_from_slice(runs);
        Ok(Self {
            runs: stored,
            count: runs.len() as u8,
            rights,
            classes,
        })
    }

    /// The grant's runs.
    pub fn runs(&self) -> &[ExtentRun] {
        &self.runs[..usize::from(self.count)]
    }

    /// What the grant permits.
    pub const fn rights(&self) -> Rights {
        self.rights
    }

    /// The classes it admits.
    pub const fn classes(&self) -> ClassSet {
        self.classes
    }

    /// Blocks the grant spans in total.
    pub fn blocks(&self) -> u64 {
        self.runs().iter().map(|run| u64::from(run.blocks)).sum()
    }

    /// The device piece holding logical block `block` of the grant, at most
    /// `max_blocks` long and never past its run's end. `None` past the
    /// grant's end or for a zero ceiling.
    pub fn segment(&self, block: u64, max_blocks: u32) -> Option<Segment> {
        if max_blocks == 0 {
            return None;
        }
        let mut skip = block;
        for run in self.runs() {
            let span = u64::from(run.blocks);
            if skip < span {
                let left = span - skip;
                let blocks = left.min(u64::from(max_blocks)) as u32;
                return Some(Segment {
                    lba: run.lba + skip,
                    blocks,
                });
            }
            skip -= span;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_runs() -> Grant {
        Grant::new(
            &[
                ExtentRun { lba: 100, blocks: 4 },
                ExtentRun { lba: 900, blocks: 6 },
            ],
            Rights::READ,
            ClassSet::DATA,
        )
        .unwrap()
    }

    #[test]
    fn a_grant_refuses_empty_oversized_and_overflowing_shapes() {
        assert_eq!(Grant::new(&[], Rights::READ, ClassSet::ALL), Err(GrantError::Shape));
        let empty_run = [ExtentRun { lba: 1, blocks: 0 }];
        assert_eq!(Grant::new(&empty_run, Rights::READ, ClassSet::ALL), Err(GrantError::Shape));
        let many = [ExtentRun { lba: 0, blocks: 1 }; MAX_RUNS + 1];
        assert_eq!(Grant::new(&many, Rights::READ, ClassSet::ALL), Err(GrantError::Shape));
        let wrap = [ExtentRun { lba: u64::MAX, blocks: 1 }];
        assert_eq!(Grant::new(&wrap, Rights::READ, ClassSet::ALL), Err(GrantError::Overflow));
    }

    #[test]
    fn a_logical_block_resolves_across_runs_and_stops_at_run_ends() {
        let grant = two_runs();
        assert_eq!(grant.blocks(), 10);
        assert_eq!(grant.segment(0, 64), Some(Segment { lba: 100, blocks: 4 }));
        assert_eq!(grant.segment(2, 64), Some(Segment { lba: 102, blocks: 2 }));
        assert_eq!(grant.segment(4, 64), Some(Segment { lba: 900, blocks: 6 }));
        assert_eq!(grant.segment(5, 2), Some(Segment { lba: 901, blocks: 2 }));
        assert_eq!(grant.segment(10, 64), None, "past the end");
        assert_eq!(grant.segment(0, 0), None, "no ceiling, no piece");
    }

    #[test]
    fn class_sets_hold_what_they_name() {
        assert!(ClassSet::DATA.contains(Class::Demand));
        assert!(ClassSet::DATA.contains(Class::Stream));
        assert!(!ClassSet::DATA.contains(Class::Commit));
        assert!(ClassSet::only(Class::Commit).contains(Class::Commit));
        assert!(ClassSet::NONE.with(Class::Stream).contains(Class::Stream));
    }
}
