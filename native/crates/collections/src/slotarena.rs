//! Single-owner generational slot arena for intrusively linked nodes.
//!
//! [`SlotArena`] is the **plain** (`&mut self`) member of the slot-storage
//! family — beside the lock-free [`AtomicSlotMap`](crate::AtomicSlotMap) and
//! [`AtomicSlotVec`](crate::AtomicSlotVec) — and it exists for the shape those
//! two deliberately cannot express: **node graphs whose edges are raw slot
//! indices stored inside the nodes themselves** (a free-space index's
//! physical-neighbour links, a heap back-pointer, a hash chain).  An intrusive
//! graph needs raw-index reads and writes across several nodes in one
//! operation, and it needs *occupancy* and *handle-worthiness* to be separate
//! events: a slot is claimed, wired into the graph, and only when it reaches
//! the state a caller may hold is a generational handle **minted** for that
//! incarnation.
//!
//! The protocol is therefore three-phase where the atomic siblings are
//! one-phase (`insert`):
//!
//! 1. [`claim`](SlotArena::claim) pops a slot off the vacancy freelist
//!    (LIFO), resets its payload to `T::default()`, and hands back the raw
//!    index for the caller to wire up;
//! 2. [`mint`](SlotArena::mint) bumps the slot's generation and issues a
//!    [`SlotId`] for the new incarnation — every previously minted handle for
//!    the slot is *detectably* stale from this moment, whatever recycling
//!    path brought the slot here;
//! 3. [`release`](SlotArena::release) returns the slot to the freelist,
//!    resetting the payload but **preserving the generation**, so a handle
//!    that outlives the release still refuses.
//!
//! [`resolve`](SlotArena::resolve) is the only path from a handle back to an
//! index: bounds, occupancy, and exact generation must all agree, so a stale
//! or fabricated handle fails closed instead of silently aliasing whatever
//! incarnation recycled the slot ([`GenId`](crate::GenId)'s posture, applied
//! to owned storage).  Raw indices, by contrast, are for the *owner's own
//! intrusive links* — they are maintained under the owner's invariants and
//! must never outlive an incarnation.
//!
//! ## Posture
//!
//! Single-owner and lock-policy-free: wrap it in the caller's exclusivity if
//! shared.  Storage is two parallel arrays (payloads, then per-slot
//! bookkeeping) reserved once at construction from a **named allocator**
//! ([`try_new_in`](SlotArena::try_new_in), `try_reserve_exact`, never a
//! global, never growth); payloads stay contiguous so an intrusive engine —
//! [`PriorityIndex`](crate::PriorityIndex) is the intended pairing — can
//! operate over [`values_mut`](SlotArena::values_mut) without seeing arena
//! bookkeeping.  Handle-facing operations refuse typed (`Option`); raw-index
//! operations index like a slice and share its panic-on-misuse contract,
//! because a bad raw index is the *owner's* broken invariant, not input.

use core::alloc::{AllocError, Allocator};
use core::ops::{Index, IndexMut};

use alloc::vec::Vec;

use crate::slotmap::SlotId;

/// Freelist terminator.
const NONE: u32 = u32::MAX;
/// `next_vacant` marker for a claimed slot — distinct from [`NONE`] so a
/// double release is detectable and occupancy is a field read.
const OCCUPIED: u32 = u32::MAX - 1;

#[derive(Clone, Copy)]
struct SlotMeta {
    /// Incarnation counter: bumped by [`SlotArena::mint`], preserved by every
    /// claim/release/recycle path, `0` until first minted (so [`SlotId`]'s
    /// never-issued generation `0` can never resolve).
    generation: u32,
    /// Next slot in the vacancy freelist, [`NONE`] at the tail, or
    /// [`OCCUPIED`] while claimed.
    next_vacant: u32,
}

/// Single-owner generational slot arena; see the module docs for the
/// claim → mint → release protocol and the raw-index vs handle split.
pub struct SlotArena<T, A: Allocator + ?Sized + 'static> {
    values: Vec<T, &'static A>,
    meta: Vec<SlotMeta, &'static A>,
    vacant: u32,
}

impl<T: Default, A: Allocator + ?Sized + 'static> SlotArena<T, A> {
    /// A fully vacant arena of exactly `capacity` slots backed by `alloc`,
    /// its freelist chained in ascending index order.
    ///
    /// Fallible by construction (`try_reserve_exact`), and these two
    /// reservations are the only allocations the arena ever performs.
    /// Refuses a capacity that could not distinguish a real index from the
    /// freelist sentinels.
    pub fn try_new_in(capacity: usize, alloc: &'static A) -> Result<Self, AllocError> {
        if capacity >= OCCUPIED as usize {
            return Err(AllocError);
        }
        let mut values = Vec::new_in(alloc);
        values.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        let mut meta = Vec::new_in(alloc);
        meta.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        for index in 0..capacity {
            values.push(T::default());
            meta.push(SlotMeta {
                generation: 0,
                next_vacant: Self::chained(index, capacity),
            });
        }
        Ok(Self {
            values,
            meta,
            vacant: if capacity == 0 { NONE } else { 0 },
        })
    }

    /// The ascending-order freelist link for `index` in a fully vacant arena.
    #[inline]
    const fn chained(index: usize, capacity: usize) -> u32 {
        if index + 1 < capacity {
            (index + 1) as u32
        } else {
            NONE
        }
    }

    /// Claim the freelist head: the slot leaves the vacancy list, its payload
    /// resets to `T::default()` (its generation survives), and the raw index
    /// is the caller's to wire into its graph.  `None` when every slot is
    /// claimed.
    pub fn claim(&mut self) -> Option<usize> {
        if self.vacant == NONE {
            return None;
        }
        let index = self.vacant as usize;
        self.vacant = self.meta[index].next_vacant;
        self.meta[index].next_vacant = OCCUPIED;
        self.values[index] = T::default();
        Some(index)
    }

    /// Return a claimed slot to the freelist (LIFO), resetting its payload
    /// but preserving its generation so outstanding handles stay detectably
    /// stale.  Releasing a vacant slot is an owner bug (it would cycle the
    /// freelist) and is refused loudly under debug.
    pub fn release(&mut self, index: usize) {
        debug_assert_eq!(
            self.meta[index].next_vacant, OCCUPIED,
            "release of a slot that is not claimed"
        );
        self.values[index] = T::default();
        self.meta[index].next_vacant = self.vacant;
        self.vacant = index as u32;
    }

    /// Return **every** slot to the freelist — claimed or not — rebuilding it
    /// in ascending index order and resetting every payload to
    /// `T::default()`.  Generations are preserved: the arena cannot see the
    /// caller's argument for why no handle survives a wholesale reset, so
    /// even a handle that wrongly does stays detectably stale rather than
    /// resolving against a post-reset incarnation.
    pub fn release_all(&mut self) {
        let capacity = self.values.len();
        for index in 0..capacity {
            self.values[index] = T::default();
            self.meta[index].next_vacant = Self::chained(index, capacity);
        }
        self.vacant = if capacity == 0 { NONE } else { 0 };
    }
}

impl<T, A: Allocator + ?Sized + 'static> SlotArena<T, A> {
    /// Mark `index`'s current occupancy as a fresh incarnation and issue its
    /// handle.  The generation bump is what retires every previously minted
    /// [`SlotId`] for this slot; generations skip `0` on wrap so an issued
    /// handle is always distinguishable from [`SlotId::INVALID`].
    pub fn mint(&mut self, index: usize) -> SlotId {
        debug_assert_eq!(
            self.meta[index].next_vacant, OCCUPIED,
            "mint of a slot that is not claimed"
        );
        let meta = &mut self.meta[index];
        meta.generation = meta.generation.wrapping_add(1);
        if meta.generation == 0 {
            meta.generation = 1;
        }
        SlotId::new(index as u32, meta.generation)
    }

    /// Resolve a caller-held handle to its raw index: bounds, occupancy, and
    /// exact generation must all agree, so a handle from a superseded
    /// incarnation — or a fabricated one — fails closed instead of aliasing
    /// the slot's current occupant.
    pub fn resolve(&self, id: SlotId) -> Option<usize> {
        let index = id.index() as usize;
        if index >= self.meta.len() || id.generation() == 0 {
            return None;
        }
        let meta = self.meta[index];
        if meta.next_vacant != OCCUPIED || meta.generation != id.generation() {
            return None;
        }
        Some(index)
    }

    /// Bounds-checked payload read by raw index, occupancy-blind: a vacant
    /// slot reads as `T::default()`.  Intrusive links are the owner's
    /// invariant, so distinguishing vacant from claimed is the owner's job
    /// ([`resolve`](Self::resolve) is the handle-facing path that does it).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Bounds-checked payload write access by raw index; see
    /// [`get`](Self::get) for the occupancy caveat.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.values.get_mut(index)
    }

    /// Every payload, vacant slots included, as one contiguous slice.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Every payload as one contiguous mutable slice — the face an intrusive
    /// engine ([`PriorityIndex`](crate::PriorityIndex)) operates over.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Total slots, claimed or vacant — fixed since construction.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.values.len()
    }
}

impl<T, A: Allocator + ?Sized + 'static> Index<usize> for SlotArena<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.values[index]
    }
}

impl<T, A: Allocator + ?Sized + 'static> IndexMut<usize> for SlotArena<T, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.values[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;

    static ARENA: TestArena<{ 1 << 16 }> = TestArena::new();

    #[test]
    fn claims_ascend_and_releases_recycle_lifo() {
        let mut arena = SlotArena::<u64, _>::try_new_in(4, &ARENA).unwrap();
        assert_eq!(arena.claim(), Some(0));
        assert_eq!(arena.claim(), Some(1));
        assert_eq!(arena.claim(), Some(2));
        arena.release(0);
        arena.release(2);
        // Most recently released first.
        assert_eq!(arena.claim(), Some(2));
        assert_eq!(arena.claim(), Some(0));
        assert_eq!(arena.claim(), Some(3));
        assert_eq!(arena.claim(), None);
    }

    #[test]
    fn claim_resets_the_payload_but_not_the_generation() {
        let mut arena = SlotArena::<u64, _>::try_new_in(1, &ARENA).unwrap();
        let slot = arena.claim().unwrap();
        arena[slot] = 99;
        let first = arena.mint(slot);
        arena.release(slot);
        let again = arena.claim().unwrap();
        assert_eq!(again, slot);
        assert_eq!(arena[again], 0);
        let second = arena.mint(again);
        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
    }

    #[test]
    fn stale_and_fabricated_handles_fail_closed() {
        let mut arena = SlotArena::<u64, _>::try_new_in(2, &ARENA).unwrap();
        let slot = arena.claim().unwrap();
        let live = arena.mint(slot);
        assert_eq!(arena.resolve(live), Some(slot));
        // A superseded incarnation: same slot, older generation.
        let fresh = arena.mint(slot);
        assert_eq!(arena.resolve(live), None);
        assert_eq!(arena.resolve(fresh), Some(slot));
        // Released occupancy: the handle's generation still matches, but the
        // slot is vacant.
        arena.release(slot);
        assert_eq!(arena.resolve(fresh), None);
        // Fabrications: never-issued generation zero, out-of-range index.
        assert_eq!(arena.resolve(SlotId::new(1, 0)), None);
        assert_eq!(arena.resolve(SlotId::new(7, 1)), None);
        assert_eq!(arena.resolve(SlotId::INVALID), None);
    }

    #[test]
    fn release_all_rebuilds_ascending_and_preserves_generations() {
        let mut arena = SlotArena::<u64, _>::try_new_in(3, &ARENA).unwrap();
        let a = arena.claim().unwrap();
        let _ = arena.claim().unwrap();
        let minted = arena.mint(a);
        arena.release_all();
        // Freelist is ascending again, not the scrambled claim/release order.
        assert_eq!(arena.claim(), Some(0));
        assert_eq!(arena.claim(), Some(1));
        assert_eq!(arena.claim(), Some(2));
        // The pre-reset handle stays stale even against its own slot's next
        // incarnation: generations continued, they did not restart.
        let reminted = arena.mint(0);
        assert_eq!(arena.resolve(minted), None);
        assert!(reminted.generation() > minted.generation());
    }

    #[test]
    fn raw_access_is_bounds_checked_and_slice_shaped() {
        let mut arena = SlotArena::<u64, _>::try_new_in(2, &ARENA).unwrap();
        let slot = arena.claim().unwrap();
        *arena.get_mut(slot).unwrap() = 7;
        assert_eq!(arena.get(slot), Some(&7));
        assert_eq!(arena.get(5), None);
        assert_eq!(arena.values().len(), 2);
        arena.values_mut()[1] = 3;
        assert_eq!(arena[1], 3);
        assert_eq!(arena.capacity(), 2);
    }

    #[test]
    fn zero_capacity_never_claims() {
        let mut arena = SlotArena::<u64, _>::try_new_in(0, &ARENA).unwrap();
        assert_eq!(arena.claim(), None);
        assert_eq!(arena.capacity(), 0);
        arena.release_all();
        assert_eq!(arena.claim(), None);
    }
}
