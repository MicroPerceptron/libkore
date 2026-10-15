//! [`BoundedVec`] — a fixed-capacity vector whose storage is in an arena
//! rather than inline.
//!
//! This is the heap-backed sibling of a `heapless::Vec`, and it exists for
//! one reason: a fixed-capacity inline vector puts its **whole worst case**
//! wherever it is declared, and "wherever it is declared" is often a stack
//! frame. A userland service's entire state is polled inside one future on a
//! delegated stack, so an inline `Vec<T, N>` there makes `N` a stack budget:
//! raising the bound is then a stack question rather than a capacity one, and
//! the two are not related by anything but an accident of storage.
//!
//! What it deliberately keeps from the inline form is the **bound as a
//! refusal**: capacity is fixed at construction, [`push`](BoundedVec::push)
//! returns the value back when full, and the storage is never reallocated. A
//! bare [`Vec`] is not a substitute — its push is infallible and grows, so a
//! bound expressed through it is a bound nothing enforces. Callers whose
//! fullness is an admission decision ("does this mutation have somewhere to
//! retire the extent it displaces?") need the refusal, not the growth.
//!
//! Single-owner and `&mut self`, like [`BitVec`](crate::BitVec): it imposes
//! no locking policy. Reach for [`AtomicSlotVec`](crate::AtomicSlotVec)
//! instead when entries need stable identities, per-entry generations, or
//! claims that outlive a borrow; this is for an ordered working set — a
//! worklist, a batch, a retirement stack — where position is the only
//! identity and the whole thing is owned by one caller.

use alloc::vec::Vec;
use core::alloc::{AllocError, Allocator};
use core::ops::{Deref, DerefMut};

/// A vector with a capacity fixed at construction and storage in `alloc`.
///
/// `capacity` is recorded separately from the backing [`Vec`]'s own capacity
/// because a reservation may legitimately hand back more than was asked for.
/// The bound this type enforces is the one the caller named, so a `Vec` that
/// over-allocated must not quietly raise it.
pub struct BoundedVec<T, A: Allocator + ?Sized + 'static> {
    items: Vec<T, &'static A>,
    capacity: usize,
}

impl<T, A: Allocator + ?Sized + 'static> BoundedVec<T, A> {
    /// An empty vector that will hold at most `capacity` items, with its
    /// storage reserved up front from `alloc` so no later push allocates.
    ///
    /// Fallible by design: the allocation happens once, where the caller can
    /// still report it, rather than at an unpredictable push deep inside an
    /// operation.
    pub fn try_new_in(capacity: usize, alloc: &'static A) -> Result<Self, AllocError> {
        let mut items = Vec::new_in(alloc);
        items.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        Ok(Self { items, capacity })
    }

    /// Append `value`, or hand it back when the vector is already at its
    /// bound.
    ///
    /// Returning the value rather than dropping it keeps the refusal
    /// non-destructive: a caller that cannot place an item can still account
    /// for it, or put it somewhere else.
    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.items.len() == self.capacity {
            return Err(value);
        }
        self.items.push(value);
        Ok(())
    }

    /// Insert `value` at `index`, shifting the items after it right, or hand
    /// it back when the vector is already at its bound.
    ///
    /// Index 0 is the case that matters: it is how a caller re-queues an item
    /// at the front of a worklist it is otherwise draining from the back.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`, matching [`Vec::insert`]. A full vector is a
    /// refusal rather than a panic, so the two failure kinds stay distinct: a
    /// bad index is a caller bug, a full buffer is a bound.
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), T> {
        // The index is judged before fullness so a caller bug is a panic even
        // when the buffer happens to be full: an out-of-range index handed
        // back as `Err(value)` would read as "no room", and the bug would
        // survive exactly as long as the buffer stayed full.
        assert!(
            index <= self.items.len(),
            "insertion index (is {index}) should be <= len (is {})",
            self.items.len()
        );
        if self.items.len() == self.capacity {
            return Err(value);
        }
        self.items.insert(index, value);
        Ok(())
    }

    /// Remove and return the last item.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    /// Drop every item, keeping the storage.
    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// The items, as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// The items, as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items
    }

    /// The bound named at construction, not the backing allocation's size.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether a further [`push`](Self::push) would be refused.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.items.len() == self.capacity
    }
}

impl<T, A: Allocator + ?Sized + 'static> Deref for BoundedVec<T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.items
    }
}

impl<T, A: Allocator + ?Sized + 'static> DerefMut for BoundedVec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items
    }
}

/// Consuming iteration, so a filled buffer can be drained by value the way a
/// [`Vec`] is.
impl<T, A: Allocator + ?Sized + 'static> IntoIterator for BoundedVec<T, A> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T, &'static A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T, A: Allocator + ?Sized + 'static> IntoIterator for &'a BoundedVec<T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T, A: Allocator + ?Sized + 'static> IntoIterator for &'a mut BoundedVec<T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

/// Equality is over the **contents**, not the capacity or the allocator: two
/// buffers holding the same items are equal even when their bounds differ.
/// This is what lets a caller assert on a returned buffer without naming how
/// it was reserved.
impl<T: PartialEq, A: Allocator + ?Sized + 'static, B: Allocator + ?Sized + 'static>
    PartialEq<BoundedVec<T, B>> for BoundedVec<T, A>
{
    #[inline]
    fn eq(&self, other: &BoundedVec<T, B>) -> bool {
        self.items.as_slice() == other.items.as_slice()
    }
}

impl<T: Eq, A: Allocator + ?Sized + 'static> Eq for BoundedVec<T, A> {}

impl<T: PartialEq, A: Allocator + ?Sized + 'static> PartialEq<[T]> for BoundedVec<T, A> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        self.items.as_slice() == other
    }
}

impl<T: core::fmt::Debug, A: Allocator + ?Sized + 'static> core::fmt::Debug for BoundedVec<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.items.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;

    static TEST_ARENA: TestArena<{ 1 << 16 }> = TestArena::new();

    fn vec_of(capacity: usize) -> BoundedVec<u32, TestArena<{ 1 << 16 }>> {
        BoundedVec::try_new_in(capacity, &TEST_ARENA).expect("test arena is large enough")
    }

    #[test]
    fn a_full_vector_hands_the_value_back_instead_of_growing() {
        let mut items = vec_of(3);
        for value in 0..3 {
            assert_eq!(items.push(value), Ok(()));
        }
        assert!(items.is_full());
        // The refusal is non-destructive: the caller still holds the item.
        assert_eq!(items.push(99), Err(99));
        assert_eq!(items.len(), 3);
        assert_eq!(&*items, &[0, 1, 2]);
    }

    #[test]
    fn the_bound_is_the_one_named_not_the_allocation_that_backs_it() {
        let items = vec_of(5);
        assert_eq!(items.capacity(), 5);
        assert!(items.is_empty());
        assert!(!items.is_full());
    }

    #[test]
    fn clearing_keeps_the_capacity_so_a_reused_buffer_never_reallocates() {
        let mut items = vec_of(2);
        assert_eq!(items.push(1), Ok(()));
        assert_eq!(items.push(2), Ok(()));
        items.clear();
        assert!(items.is_empty());
        assert_eq!(items.capacity(), 2);
        assert_eq!(items.push(3), Ok(()));
        assert_eq!(&*items, &[3]);
    }

    #[test]
    fn it_behaves_as_a_stack_and_as_a_slice() {
        let mut items = vec_of(4);
        for value in 1..=3 {
            assert_eq!(items.push(value), Ok(()));
        }
        assert_eq!(items.pop(), Some(3));
        assert_eq!(items.iter().copied().sum::<u32>(), 3);
        assert!(items.contains(&2));
        items[0] = 7;
        assert_eq!(&*items, &[7, 2]);
        assert_eq!(items.pop(), Some(2));
        assert_eq!(items.pop(), Some(7));
        assert_eq!(items.pop(), None);
    }

    #[test]
    fn it_iterates_by_value_by_reference_and_mutably() {
        let mut items = vec_of(3);
        for value in 1..=3 {
            assert_eq!(items.push(value), Ok(()));
        }
        for value in &mut items {
            *value *= 2;
        }
        assert_eq!(
            (&items).into_iter().copied().collect::<std::vec::Vec<_>>(),
            [2, 4, 6]
        );
        assert_eq!(items.into_iter().sum::<u32>(), 12);
    }

    #[test]
    fn equality_is_over_contents_not_over_the_bound() {
        let mut four = vec_of(4);
        let mut nine = vec_of(9);
        for value in 1..=2 {
            assert_eq!(four.push(value), Ok(()));
            assert_eq!(nine.push(value), Ok(()));
        }
        assert_eq!(four.capacity(), 4);
        assert_eq!(nine.capacity(), 9);
        assert_eq!(four, nine);
        assert_eq!(nine.push(3), Ok(()));
        assert_ne!(four, nine);
    }

    #[test]
    fn inserting_at_the_front_requeues_without_growing() {
        let mut items = vec_of(3);
        assert_eq!(items.push(1), Ok(()));
        assert_eq!(items.push(2), Ok(()));
        assert_eq!(items.insert(0, 9), Ok(()));
        assert_eq!(&*items, &[9, 1, 2]);
        assert!(items.is_full());
        assert_eq!(items.insert(0, 4), Err(4));
        assert_eq!(&*items, &[9, 1, 2]);
    }

    #[test]
    fn a_zero_capacity_vector_refuses_every_push() {
        let mut items = vec_of(0);
        assert!(items.is_full());
        assert_eq!(items.push(1), Err(1));
        assert!(items.is_empty());
    }

    /// The two failure kinds stay distinct at the bound: a full vector
    /// refuses, but a bad index is a caller bug and panics like
    /// `Vec::insert` even when the vector is full, so fullness can never mask
    /// the bug.
    #[test]
    #[should_panic(expected = "insertion index")]
    fn a_bad_index_panics_even_when_the_vector_is_full() {
        let mut items = vec_of(1);
        assert_eq!(items.push(1), Ok(()));
        assert!(items.is_full());
        let _ = items.insert(2, 9);
    }
}
