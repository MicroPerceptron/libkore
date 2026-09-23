//! Runtime-sized, generational, independently-claimable slot storage.
//!
//! [`AtomicSlotVec`] is the allocator-backed sibling of
//! [`AtomicSlotMap`](super::AtomicSlotMap), exactly as [`AtomicBitVec`] is the
//! sibling of [`AtomicBitSet`]: same per-slot lock-free protocol, same affine
//! [`SlotVecWrite`] guard that may cross an `await` while an unrelated slot
//! stays usable — but the capacity is chosen at construction from an
//! [`Allocator`] instead of baked into the type as `const N`.
//!
//! Use it where the slot count is a *runtime* property of the instance (a
//! filesystem's derived-index arena, sized at mount to the namespace bound)
//! rather than a compile-time constant. The fixed [`AtomicSlotMap`] remains
//! the right choice when the bound is known at compile time and belongs in
//! `.bss`.
//!
//! The lock-free protocol — the `LIVE`/`WRITING` state word, the announced
//! reader count, the CAS-to-claim insertion — is shared verbatim with
//! [`AtomicSlotMap`] (the primitives are `pub(crate)` in that module), so the
//! two siblings cannot drift in their concurrency semantics. Only the storage
//! backing and the capacity source differ.

use core::{
    alloc::{AllocError, Allocator},
    cell::UnsafeCell,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

use alloc::vec::Vec;

use crate::slotmap::{GEN_MASK, LIVE, ReaderGuard, SlotAccess, SlotId, WRITING, classify};

/// A runtime-sized generational arena whose slots are claimed independently.
///
/// Backed by `Vec<_, &'static A>`; the capacity is fixed at construction and
/// never grows (a growable variant would surrender the stable-address
/// guarantee the affine guards rely on). Every slot address is stable for the
/// arena's life, so a [`SlotVecWrite`]/[`SlotVecRead`] guard may be held across
/// suspension while provider I/O is pending.
pub struct AtomicSlotVec<T, A: Allocator + ?Sized + 'static> {
    states: Vec<AtomicU64, &'static A>,
    readers: Vec<AtomicU32, &'static A>,
    slots: Vec<UnsafeCell<MaybeUninit<T>>, &'static A>,
    live: AtomicUsize,
}

// SAFETY: identical argument to `AtomicSlotMap` — publication and movement are
// serialized by each slot's state word; mutation proceeds only after observing
// that no guarded reader exists. `A: Sync` is required because the arena shares
// `&'static A` across threads.
unsafe impl<T: Send, A: Allocator + Sync + ?Sized + 'static> Send for AtomicSlotVec<T, A> {}
// SAFETY: shared reads require `T: Sync`; mutation is exposed only after the
// per-slot CAS exclusion and reader drain. Different slots are independent.
unsafe impl<T: Send + Sync, A: Allocator + Sync + ?Sized + 'static> Sync for AtomicSlotVec<T, A> {}

impl<T, A: Allocator + ?Sized + 'static> AtomicSlotVec<T, A> {
    /// Build an empty arena of exactly `capacity` slots, backed by `alloc`.
    ///
    /// Fallible by construction: a founded service never aborts on allocator
    /// exhaustion, so the reservation is `try_reserve_exact` and the caller
    /// receives [`AllocError`] rather than an `handle_alloc_error` abort. Every
    /// slot begins at generation one, matching [`AtomicSlotMap::new`].
    ///
    /// [`AtomicSlotMap::new`]: super::AtomicSlotMap::new
    pub fn try_new_in(capacity: usize, alloc: &'static A) -> Result<Self, AllocError> {
        let mut states = Vec::new_in(alloc);
        states.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        for _ in 0..capacity {
            states.push(AtomicU64::new(1));
        }
        let mut readers = Vec::new_in(alloc);
        readers
            .try_reserve_exact(capacity)
            .map_err(|_| AllocError)?;
        for _ in 0..capacity {
            readers.push(AtomicU32::new(0));
        }
        let mut slots = Vec::new_in(alloc);
        slots.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        Ok(Self {
            states,
            readers,
            slots,
            live: AtomicUsize::new(0),
        })
    }

    /// Insert into the first vacant slot. A full arena returns ownership of
    /// `value`; it never drops a caller's affine state on exhaustion.
    pub fn insert(&self, value: T) -> Result<SlotId, T> {
        self.reserve(value).map(SlotVecReservation::commit)
    }

    /// Reserve initialized storage without exposing its payload. The exact
    /// generation reports `Busy` until commit; dropping the reservation
    /// retires it and drops its value synchronously. Capacity includes held
    /// reservations, so suspended transactions cannot overbook the arena.
    pub fn reserve(&self, value: T) -> Result<SlotVecReservation<'_, T, A>, T> {
        for index in 0..self.states.len() {
            let observed = self.states[index].load(Ordering::Acquire);
            if observed & (LIVE | WRITING) != 0 {
                continue;
            }
            let generation = (observed & GEN_MASK) as u32;
            if self.states[index]
                .compare_exchange(
                    observed,
                    generation as u64 | WRITING,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                continue;
            }
            // SAFETY: this CAS uniquely owns a vacant slot; no reader can
            // validate it without LIVE and no other insertion can win it.
            unsafe { (*self.slots[index].get()).write(value) };
            self.live.fetch_add(1, Ordering::Relaxed);
            self.states[index].store(generation as u64 | LIVE | WRITING, Ordering::Release);
            return Ok(SlotVecReservation {
                claim: Some(SlotVecWrite {
                    map: self,
                    id: SlotId::new(index as u32, generation),
                    armed: true,
                }),
            });
        }
        Err(value)
    }

    /// Visit one live value under an exact per-slot temporal guard. `f` must be
    /// bounded and must not suspend.
    pub fn read<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Result<R, SlotAccess> {
        let guard = self.try_read(id)?;
        Ok(f(&guard))
    }

    /// Borrow one exact generation under its per-slot reader announcement.
    pub fn try_read(&self, id: SlotId) -> Result<SlotVecRead<'_, T, A>, SlotAccess> {
        let index = self.index(id)?;
        let reader = &self.readers[index];
        reader.fetch_add(1, Ordering::SeqCst);
        let reader_guard = ReaderGuard(reader);
        let live = id.generation() as u64 | LIVE;
        let observed = self.states[index].load(Ordering::SeqCst);
        if observed != live {
            return Err(classify(observed, id.generation()));
        }
        Ok(SlotVecRead {
            map: self,
            id,
            _reader: reader_guard,
        })
    }

    /// Claim one exact live generation for mutation. The returned affine guard
    /// may cross an `await`; conflicting access receives `Busy`.
    pub fn try_write(&self, id: SlotId) -> Result<SlotVecWrite<'_, T, A>, SlotAccess> {
        let index = self.index(id)?;
        let live = id.generation() as u64 | LIVE;
        match self.states[index].compare_exchange(
            live,
            live | WRITING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {}
            Err(observed) => return Err(classify(observed, id.generation())),
        }
        if self.readers[index].load(Ordering::SeqCst) != 0 {
            self.states[index].store(live, Ordering::Release);
            return Err(SlotAccess::Busy);
        }
        Ok(SlotVecWrite {
            map: self,
            id,
            armed: true,
        })
    }

    /// Remove one exact generation. Stale and contended slots fail closed.
    pub fn try_remove(&self, id: SlotId) -> Result<T, SlotAccess> {
        self.try_write(id)?.remove()
    }

    /// Single-owner compatibility form; prefer [`try_read`](Self::try_read).
    #[inline]
    pub fn get(&self, id: SlotId) -> Option<SlotVecRead<'_, T, A>> {
        self.try_read(id).ok()
    }

    /// Single-owner compatibility form; prefer [`try_write`](Self::try_write).
    #[inline]
    pub fn get_mut(&self, id: SlotId) -> Option<SlotVecWrite<'_, T, A>> {
        self.try_write(id).ok()
    }

    /// Single-owner compatibility form; prefer [`try_remove`](Self::try_remove).
    #[inline]
    pub fn remove(&self, id: SlotId) -> Option<T> {
        self.try_remove(id).ok()
    }

    /// Whether the exact generation is currently live.
    pub fn contains(&self, id: SlotId) -> bool {
        let Ok(index) = self.index(id) else {
            return false;
        };
        let observed = self.states[index].load(Ordering::Acquire);
        observed & LIVE != 0 && (observed & GEN_MASK) as u32 == id.generation()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.states.len()
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// Visit a best-effort stable pass over every live slot in ascending index
    /// order. A contended slot is skipped for this pass.
    pub fn for_each(&self, mut f: impl FnMut(SlotId, &T)) {
        for index in 0..self.states.len() {
            let observed = self.states[index].load(Ordering::Acquire);
            if observed & LIVE == 0 || observed & WRITING != 0 {
                continue;
            }
            let id = SlotId::new(index as u32, (observed & GEN_MASK) as u32);
            let _ = self.read(id, |value| f(id, value));
        }
    }

    /// Iterate stable live slots in ascending index order.
    pub fn iter(&self) -> SlotVecIter<'_, T, A> {
        SlotVecIter { map: self, next: 0 }
    }

    /// Iterate independently claimed live slots for mutation.
    pub fn iter_mut(&self) -> SlotVecWriteIter<'_, T, A> {
        SlotVecWriteIter { map: self, next: 0 }
    }

    /// Resume a read iteration at one slot index.
    pub fn iter_from(&self, from: u32) -> SlotVecIter<'_, T, A> {
        SlotVecIter {
            map: self,
            next: from as usize,
        }
    }

    /// The live id at an index, if it is currently stable. Routing hint only.
    pub fn id_at(&self, index: usize) -> Option<SlotId> {
        if index >= self.states.len() {
            return None;
        }
        let observed = self.states[index].load(Ordering::Acquire);
        if observed & LIVE == 0 || observed & WRITING != 0 {
            return None;
        }
        Some(SlotId::new(index as u32, (observed & GEN_MASK) as u32))
    }

    /// The exact live generation at an index, including a writer-claimed one.
    /// A routing hint for bounded retirement audits; conveys no payload access.
    pub fn live_id_at(&self, index: usize) -> Option<SlotId> {
        if index >= self.states.len() {
            return None;
        }
        let observed = self.states[index].load(Ordering::Acquire);
        if observed & LIVE == 0 {
            return None;
        }
        Some(SlotId::new(index as u32, (observed & GEN_MASK) as u32))
    }

    #[inline]
    fn index(&self, id: SlotId) -> Result<usize, SlotAccess> {
        let index = id.index() as usize;
        if index >= self.states.len() || id.generation() == 0 {
            Err(SlotAccess::Stale)
        } else {
            Ok(index)
        }
    }
}

impl<T, A: Allocator + ?Sized + 'static> Drop for AtomicSlotVec<T, A> {
    fn drop(&mut self) {
        for (state, slot) in self.states.iter_mut().zip(&mut self.slots) {
            if *state.get_mut() & LIVE != 0 {
                // SAFETY: `&mut self` excludes every reader/writer; LIVE means
                // this slot contains one initialized value.
                unsafe { slot.get_mut().assume_init_drop() };
            }
        }
    }
}

/// One affine mutation claim over an exact live slot generation.
pub struct SlotVecWrite<'a, T, A: Allocator + ?Sized + 'static> {
    map: &'a AtomicSlotVec<T, A>,
    id: SlotId,
    armed: bool,
}

/// One guarded read of an exact live slot generation.
pub struct SlotVecRead<'a, T, A: Allocator + ?Sized + 'static> {
    map: &'a AtomicSlotVec<T, A>,
    id: SlotId,
    _reader: ReaderGuard<'a>,
}

/// An unpublished value owned by one bounded transaction. Unlike a write
/// claim, drop aborts insertion; only explicit commit makes the value readable.
pub struct SlotVecReservation<'a, T, A: Allocator + ?Sized + 'static> {
    claim: Option<SlotVecWrite<'a, T, A>>,
}

impl<T, A: Allocator + ?Sized + 'static> SlotVecReservation<'_, T, A> {
    pub fn id(&self) -> SlotId {
        self.claim.as_ref().expect("live reservation").id()
    }

    pub fn commit(mut self) -> SlotId {
        let claim = self.claim.take().expect("live reservation");
        let id = claim.id();
        drop(claim);
        id
    }
}

impl<T, A: Allocator + ?Sized + 'static> Deref for SlotVecReservation<'_, T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        self.claim.as_ref().expect("live reservation")
    }
}

impl<T, A: Allocator + ?Sized + 'static> DerefMut for SlotVecReservation<'_, T, A> {
    fn deref_mut(&mut self) -> &mut T {
        self.claim.as_mut().expect("live reservation")
    }
}

impl<T, A: Allocator + ?Sized + 'static> Drop for SlotVecReservation<'_, T, A> {
    fn drop(&mut self) {
        if let Some(claim) = self.claim.take() {
            drop(
                claim
                    .remove()
                    .expect("reservation owns its exact generation"),
            );
        }
    }
}

impl<T, A: Allocator + ?Sized + 'static> SlotVecRead<'_, T, A> {
    #[inline]
    pub const fn id(&self) -> SlotId {
        self.id
    }
}

impl<T, A: Allocator + ?Sized + 'static> Deref for SlotVecRead<'_, T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let index = self.id.index() as usize;
        // SAFETY: `_reader` announced itself before the exact generation was
        // revalidated, and a writer cannot touch the payload until it drops.
        unsafe { (*self.map.slots[index].get()).assume_init_ref() }
    }
}

impl<T: core::fmt::Debug, A: Allocator + ?Sized + 'static> core::fmt::Debug
    for SlotVecRead<'_, T, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlotVecRead")
            .field("id", &self.id)
            .field("value", &**self)
            .finish()
    }
}

impl<T, A: Allocator + ?Sized + 'static> SlotVecWrite<'_, T, A> {
    #[inline]
    pub const fn id(&self) -> SlotId {
        self.id
    }

    /// Consume the claim and retire this generation, returning its value.
    pub fn remove(mut self) -> Result<T, SlotAccess> {
        let index = self.id.index() as usize;
        // SAFETY: this guard uniquely owns the writer state and drained every
        // reader before construction.
        let value = unsafe { (*self.map.slots[index].get()).assume_init_read() };
        let mut generation = self.id.generation().wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        self.map.states[index].store(generation as u64, Ordering::Release);
        self.map.live.fetch_sub(1, Ordering::Relaxed);
        self.armed = false;
        Ok(value)
    }
}

impl<T, A: Allocator + ?Sized + 'static> Deref for SlotVecWrite<'_, T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let index = self.id.index() as usize;
        // SAFETY: the live writer exclusion is held for this guard's lifetime.
        unsafe { (*self.map.slots[index].get()).assume_init_ref() }
    }
}

impl<T, A: Allocator + ?Sized + 'static> DerefMut for SlotVecWrite<'_, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let index = self.id.index() as usize;
        // SAFETY: the live writer exclusion is held uniquely by this guard.
        unsafe { (*self.map.slots[index].get()).assume_init_mut() }
    }
}

impl<T: core::fmt::Debug, A: Allocator + ?Sized + 'static> core::fmt::Debug
    for SlotVecWrite<'_, T, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlotVecWrite")
            .field("id", &self.id)
            .field("value", &**self)
            .finish()
    }
}

impl<T, A: Allocator + ?Sized + 'static> Drop for SlotVecWrite<'_, T, A> {
    fn drop(&mut self) {
        if self.armed {
            self.map.states[self.id.index() as usize]
                .store(self.id.generation() as u64 | LIVE, Ordering::Release);
        }
    }
}

/// Stable read iterator over a runtime-sized slot arena.
pub struct SlotVecIter<'a, T, A: Allocator + ?Sized + 'static> {
    map: &'a AtomicSlotVec<T, A>,
    next: usize,
}

impl<'a, T, A: Allocator + ?Sized + 'static> Iterator for SlotVecIter<'a, T, A> {
    type Item = (SlotId, SlotVecRead<'a, T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.map.capacity() {
            let index = self.next;
            self.next += 1;
            let Some(id) = self.map.id_at(index) else {
                continue;
            };
            if let Ok(value) = self.map.try_read(id) {
                return Some((id, value));
            }
        }
        None
    }
}

/// Stable write-claim iterator over a runtime-sized slot arena.
pub struct SlotVecWriteIter<'a, T, A: Allocator + ?Sized + 'static> {
    map: &'a AtomicSlotVec<T, A>,
    next: usize,
}

impl<'a, T, A: Allocator + ?Sized + 'static> Iterator for SlotVecWriteIter<'a, T, A> {
    type Item = (SlotId, SlotVecWrite<'a, T, A>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.map.capacity() {
            let index = self.next;
            self.next += 1;
            let Some(id) = self.map.id_at(index) else {
                continue;
            };
            if let Ok(value) = self.map.try_write(id) {
                return Some((id, value));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;
    use std::sync::{Arc, Barrier};

    static ARENA: TestArena<{ 1 << 20 }> = TestArena::new();

    #[test]
    fn reservation_is_unreadable_until_commit_and_blocks_only_its_slot() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(2, &ARENA).unwrap();
        let mut reserved = slots.reserve(7).unwrap();
        let id = reserved.id();
        assert_eq!(slots.read(id, |value| *value), Err(SlotAccess::Busy));
        assert_eq!(slots.try_remove(id), Err(SlotAccess::Busy));
        let sibling = slots.insert(9).unwrap();
        assert_eq!(slots.read(sibling, |value| *value), Ok(9));
        assert!(matches!(slots.reserve(11), Err(11)));
        *reserved = 8;
        assert_eq!(reserved.commit(), id);
        assert_eq!(slots.read(id, |value| *value), Ok(8));
    }

    #[test]
    fn cancelled_reservation_drops_affine_value_once_and_retires_identity() {
        struct Counted<'a>(&'a AtomicUsize);
        impl Drop for Counted<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = AtomicUsize::new(0);
        let slots = AtomicSlotVec::<Counted<'_>, _>::try_new_in(1, &ARENA).unwrap();
        let reserved = slots.reserve(Counted(&drops)).ok().unwrap();
        let old = reserved.id();
        drop(reserved);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(slots.len(), 0);
        let next = slots.reserve(Counted(&drops)).ok().unwrap().commit();
        assert_eq!(old.index(), next.index());
        assert_ne!(old.generation(), next.generation());
        assert_eq!(slots.read(old, |_| ()), Err(SlotAccess::Stale));
        drop(slots);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn generations_fail_closed_after_reuse() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(1, &ARENA).unwrap();
        let old = slots.insert(7).unwrap();
        assert_eq!(slots.read(old, |v| *v), Ok(7));
        assert_eq!(slots.try_remove(old), Ok(7));
        let new = slots.insert(9).unwrap();
        assert_eq!(old.index(), new.index());
        assert_ne!(old.generation(), new.generation());
        assert_eq!(slots.read(old, |v| *v), Err(SlotAccess::Stale));
        assert_eq!(slots.read(new, |v| *v), Ok(9));
    }

    #[test]
    fn full_arena_returns_the_uninserted_value() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(1, &ARENA).unwrap();
        slots.insert(1).unwrap();
        assert_eq!(slots.insert(2), Err(2));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots.capacity(), 1);
    }

    #[test]
    fn capacity_is_the_runtime_argument() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(300, &ARENA).unwrap();
        // Past the old fixed-64 ceiling: a slot index well beyond 64 is live
        // and independently claimable — the property the fixed u64 mask denied.
        let mut ids = std::vec::Vec::new();
        for value in 0..300u32 {
            ids.push(slots.insert(value).unwrap());
        }
        assert_eq!(slots.len(), 300);
        assert!(slots.insert(9999).is_err());
        assert_eq!(slots.read(ids[200], |v| *v), Ok(200));
        assert!(ids[200].index() >= 64);
    }

    #[test]
    fn one_claim_blocks_only_its_exact_slot() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(2, &ARENA).unwrap();
        let first = slots.insert(1).unwrap();
        let second = slots.insert(2).unwrap();
        let mut held = slots.try_write(first).unwrap();
        *held = 11;
        assert!(matches!(slots.try_write(first), Err(SlotAccess::Busy)));
        let mut other = slots.try_write(second).unwrap();
        *other = 22;
        drop(other);
        drop(held);
        assert_eq!(slots.read(first, |v| *v), Ok(11));
        assert_eq!(slots.read(second, |v| *v), Ok(22));
    }

    #[test]
    fn writer_contention_on_a_reader_is_nonblocking_and_retryable() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(1, &ARENA).unwrap();
        let id = slots.insert(1).unwrap();
        let read = slots.try_read(id).unwrap();
        assert!(matches!(slots.try_write(id), Err(SlotAccess::Busy)));
        assert_eq!(slots.read(id, |value| *value), Ok(1));
        drop(read);
        let mut write = slots.try_write(id).unwrap();
        *write = 2;
        drop(write);
        assert_eq!(slots.read(id, |value| *value), Ok(2));
    }

    #[test]
    fn disjoint_claims_are_live_on_different_threads() {
        let slots = Arc::new(AtomicSlotVec::<u32, _>::try_new_in(2, &ARENA).unwrap());
        let first = slots.insert(1).unwrap();
        let second = slots.insert(2).unwrap();
        let entered = Arc::new(Barrier::new(3));
        std::thread::scope(|scope| {
            for id in [first, second] {
                let slots = Arc::clone(&slots);
                let entered = Arc::clone(&entered);
                scope.spawn(move || {
                    let mut claim = slots.try_write(id).unwrap();
                    entered.wait();
                    *claim += 10;
                    entered.wait();
                });
            }
            entered.wait();
            assert!(matches!(slots.try_write(first), Err(SlotAccess::Busy)));
            assert!(matches!(slots.try_write(second), Err(SlotAccess::Busy)));
            entered.wait();
        });
        assert_eq!(slots.read(first, |v| *v), Ok(11));
        assert_eq!(slots.read(second, |v| *v), Ok(12));
    }

    #[test]
    fn type_erased_dyn_allocator_backs_the_arena() {
        // The fs service holds one arena type regardless of whether the
        // allocator is a test arena or the service heap: `dyn Allocator`.
        let alloc: &'static dyn core::alloc::Allocator = &ARENA;
        let slots =
            AtomicSlotVec::<u32, dyn core::alloc::Allocator>::try_new_in(128, alloc).unwrap();
        let id = slots.insert(42).unwrap();
        assert_eq!(slots.capacity(), 128);
        assert_eq!(slots.read(id, |v| *v), Ok(42));
    }

    #[test]
    fn unwind_republishes_a_claimed_slot() {
        let slots = AtomicSlotVec::<u32, _>::try_new_in(1, &ARENA).unwrap();
        let id = slots.insert(1).unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut claim = slots.try_write(id).unwrap();
            *claim = 5;
            panic!("exercise the guard");
        }));
        assert_eq!(slots.read(id, |v| *v), Ok(5));
    }
}
