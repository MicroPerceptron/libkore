//! Unsigned integer traits.

pub trait Unsigned: Copy + Ord + Sized {
    const MAX: usize;

    fn checked_add(self, offset: usize) -> Option<Self>;
    fn checked_sub(self, offset: usize) -> Option<Self>;
}
