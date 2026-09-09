//! Bounded, generational, independently-claimable slot storage.
//!
//! [`AtomicSlotMap`] is the fixed-capacity userland sibling of the kernel's
//! fixed slab: values live at stable addresses, each reuse advances only that
//! slot's generation, and mutation excludes one slot rather than the whole
//! arena.  The important addition over a plain slab is an affine
//! [`SlotWrite`] guard.  A filesystem transaction may retain that guard while
//! provider I/O is pending, so an unrelated object can continue while a
//! conflicting operation receives explicit [`SlotAccess::Busy`] backpressure.
//!
//! There is no allocator and no global lock.  Insertion probes the inline
//! slots with CAS; reads announce themselves on the exact slot; a writer makes
//! that slot unstable and succeeds only when its reader count is already zero.
//! Contention republishes immediately as [`SlotAccess::Busy`] rather than
//! spinning behind a suspended cooperative task.

use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

pub(crate) const LIVE: u64 = 1 << 63;
pub(crate) const WRITING: u64 = 1 << 62;
pub(crate) const GEN_MASK: u64 = u32::MAX as u64;

/// A stable slot index paired with that slot's exact reuse generation.
///
/// The all-zero value is never issued: generations start at one and skip zero
/// on wrap.  Packing is suitable for wire/storage identity, but the bits are
/// not authority; every use revalidates the live slot state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId {
    index: u32,
    generation: u32,
}

impl SlotId {
    /// A never-live sentinel.
    pub const INVALID: Self = Self {
        index: 0,
        generation: 0,
    };

    /// Rebuild an identifier.  Deliberately validates nothing; fabricated or
    /// stale values fail closed when presented to the map.
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.generation != 0
    }

    #[inline]
    pub const fn to_bits(self) -> u64 {
        ((self.index as u64) << 32) | self.generation as u64
    }

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            index: (bits >> 32) as u32,
            generation: bits as u32,
        }
    }
}

/// Why one exact slot could not be visited or claimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotAccess {
    /// The index is outside the map, the generation is stale, or the slot is
    /// vacant.
    Stale,
    /// The exact live generation is already held by another writer.
    Busy,
}

pub(crate) struct ReaderGuard<'a>(pub(crate) &'a AtomicU32);

impl Drop for ReaderGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A fixed-capacity generational arena whose slots are claimed independently.
pub struct AtomicSlotMap<T, const N: usize> {
    states: [AtomicU64; N],
    readers: [AtomicU32; N],
    slots: [UnsafeCell<MaybeUninit<T>>; N],
    live: AtomicUsize,
}

// SAFETY: publication and movement are serialized by each slot's state word;
// mutation proceeds only after observing that no guarded reader exists.
unsafe impl<T: Send, const N: usize> Send for AtomicSlotMap<T, N> {}
// SAFETY: shared reads require `T: Sync`; mutation is exposed only after the
// per-slot CAS exclusion and reader drain.  Different slots are independent.
unsafe impl<T: Send + Sync, const N: usize> Sync for AtomicSlotMap<T, N> {}

impl<T, const N: usize> AtomicSlotMap<T, N> {
    /// An empty inline arena.  Every slot begins at generation one.
    pub const fn new() -> Self {
        Self {
            states: [const { AtomicU64::new(1) }; N],
            readers: [const { AtomicU32::new(0) }; N],
            slots: [const { UnsafeCell::new(MaybeUninit::uninit()) }; N],
            live: AtomicUsize::new(0),
        }
    }

    /// Insert into the first vacant slot.  A full arena returns ownership of
    /// `value`; it never drops a caller's affine state on exhaustion.
    pub fn insert(&self, value: T) -> Result<SlotId, T> {
        for index in 0..N {
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
            self.states[index].store(generation as u64 | LIVE, Ordering::Release);
            self.live.fetch_add(1, Ordering::Relaxed);
            return Ok(SlotId::new(index as u32, generation));
        }
        Err(value)
    }

    /// Visit one live value under an exact per-slot temporal guard.
    ///
    /// `f` must be bounded and must not suspend: a writer that has made the
    /// slot unstable waits only for this announced reader, never for readers
    /// of any other slot.
    pub fn read<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Result<R, SlotAccess> {
        let guard = self.try_read(id)?;
        Ok(f(&guard))
    }

    /// Borrow one exact generation under its per-slot reader announcement.
    /// Prefer [`read`](Self::read) when a closure is sufficient: the named
    /// guard exists for bounded iterators and adapters migrating from a
    /// single-owner arena, and retaining it across suspension delays a writer.
    pub fn try_read(&self, id: SlotId) -> Result<SlotRead<'_, T, N>, SlotAccess> {
        let index = self.index(id)?;
        let reader = &self.readers[index];
        reader.fetch_add(1, Ordering::SeqCst);
        let reader_guard = ReaderGuard(reader);
        let live = id.generation as u64 | LIVE;
        let observed = self.states[index].load(Ordering::SeqCst);
        if observed != live {
            return Err(classify(observed, id.generation));
        }
        Ok(SlotRead {
            map: self,
            id,
            _reader: reader_guard,
        })
    }

    /// Claim one exact live generation for mutation.
    ///
    /// The returned affine guard may cross an `await`: all conflicting access
    /// receives `Busy`, while unrelated slots remain independently usable.
    pub fn try_write(&self, id: SlotId) -> Result<SlotWrite<'_, T, N>, SlotAccess> {
        let index = self.index(id)?;
        let live = id.generation as u64 | LIVE;
        match self.states[index].compare_exchange(
            live,
            live | WRITING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {}
            Err(observed) => return Err(classify(observed, id.generation)),
        }
        if self.readers[index].load(Ordering::SeqCst) != 0 {
            self.states[index].store(live, Ordering::Release);
            return Err(SlotAccess::Busy);
        }
        Ok(SlotWrite {
            map: self,
            id,
            armed: true,
        })
    }

    /// Remove one exact generation.  Stale and contended slots fail closed.
    pub fn try_remove(&self, id: SlotId) -> Result<T, SlotAccess> {
        self.try_write(id)?.remove()
    }

    /// Single-owner compatibility form.  New concurrent code should retain
    /// the [`SlotAccess`] distinction from [`try_read`](Self::try_read).
    #[inline]
    pub fn get(&self, id: SlotId) -> Option<SlotRead<'_, T, N>> {
        self.try_read(id).ok()
    }

    /// Single-owner compatibility form.  New concurrent code should retain
    /// the [`SlotAccess`] distinction from [`try_write`](Self::try_write).
    #[inline]
    pub fn get_mut(&self, id: SlotId) -> Option<SlotWrite<'_, T, N>> {
        self.try_write(id).ok()
    }

    /// Single-owner compatibility form.  New concurrent code should retain
    /// the [`SlotAccess`] distinction from [`try_remove`](Self::try_remove).
    #[inline]
    pub fn remove(&self, id: SlotId) -> Option<T> {
        self.try_remove(id).ok()
    }

    /// Whether the exact generation is currently live.  A writer-held slot is
    /// still live, even though a new access to it would receive `Busy`.
    pub fn contains(&self, id: SlotId) -> bool {
        let Ok(index) = self.index(id) else {
            return false;
        };
        let observed = self.states[index].load(Ordering::Acquire);
        observed & LIVE != 0 && (observed & GEN_MASK) as u32 == id.generation
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
    pub const fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == N
    }

    /// Visit a best-effort stable pass over every live slot in ascending
    /// index order.  A contended slot is skipped for this pass; callers that
    /// require a complete transaction acquire the relevant write claims
    /// explicitly instead of treating a scan as a snapshot.
    pub fn for_each(&self, mut f: impl FnMut(SlotId, &T)) {
        for index in 0..N {
            let observed = self.states[index].load(Ordering::Acquire);
            if observed & LIVE == 0 || observed & WRITING != 0 {
                continue;
            }
            let id = SlotId::new(index as u32, (observed & GEN_MASK) as u32);
            let _ = self.read(id, |value| f(id, value));
        }
    }

    /// Iterate stable live slots in ascending index order.  A writer-held
    /// slot is skipped for this pass; transaction snapshots must explicitly
    /// claim their footprint rather than treating an iterator as atomic.
    pub fn iter(&self) -> SlotIter<'_, T, N> {
        SlotIter { map: self, next: 0 }
    }

    /// Iterate independently claimed live slots for mutation.
    pub fn iter_mut(&self) -> SlotWriteIter<'_, T, N> {
        SlotWriteIter { map: self, next: 0 }
    }

    /// Resume a read iteration at one slot index.
    pub fn iter_from(&self, from: u32) -> SlotIter<'_, T, N> {
        SlotIter {
            map: self,
            next: from as usize,
        }
    }

    /// The live id at an index, if it is currently stable.  The result is a
    /// routing hint only; `read`/`try_write` revalidates it before use.
    pub fn id_at(&self, index: usize) -> Option<SlotId> {
        if index >= N {
            return None;
        }
        let observed = self.states[index].load(Ordering::Acquire);
        if observed & LIVE == 0 || observed & WRITING != 0 {
            return None;
        }
        Some(SlotId::new(index as u32, (observed & GEN_MASK) as u32))
    }

    /// The exact live generation at an index, including a writer-claimed one.
    ///
    /// This is a routing hint for bounded retirement audits. It conveys no
    /// payload access; callers still use `try_read`/`try_write`, which return
    /// `Busy` while that generation is claimed.
    pub fn live_id_at(&self, index: usize) -> Option<SlotId> {
        if index >= N {
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
        let index = id.index as usize;
        if index >= N || id.generation == 0 {
            Err(SlotAccess::Stale)
        } else {
            Ok(index)
        }
    }
}

impl<T, const N: usize> Default for AtomicSlotMap<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for AtomicSlotMap<T, N> {
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
pub struct SlotWrite<'a, T, const N: usize> {
    map: &'a AtomicSlotMap<T, N>,
    id: SlotId,
    armed: bool,
}

/// One guarded read of an exact live slot generation.
pub struct SlotRead<'a, T, const N: usize> {
    map: &'a AtomicSlotMap<T, N>,
    id: SlotId,
    _reader: ReaderGuard<'a>,
}

impl<T, const N: usize> SlotRead<'_, T, N> {
    #[inline]
    pub const fn id(&self) -> SlotId {
        self.id
    }
}

impl<T, const N: usize> Deref for SlotRead<'_, T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let index = self.id.index as usize;
        // SAFETY: `_reader` announced itself before the exact generation was
        // revalidated, and a writer cannot touch the payload until it drops.
        unsafe { (*self.map.slots[index].get()).assume_init_ref() }
    }
}

impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for SlotRead<'_, T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlotRead")
            .field("id", &self.id)
            .field("value", &**self)
            .finish()
    }
}

/// Stable read iterator over a bounded slot map.
pub struct SlotIter<'a, T, const N: usize> {
    map: &'a AtomicSlotMap<T, N>,
    next: usize,
}

impl<'a, T, const N: usize> Iterator for SlotIter<'a, T, N> {
    type Item = (SlotId, SlotRead<'a, T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < N {
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

/// Stable write-claim iterator over a bounded slot map.
pub struct SlotWriteIter<'a, T, const N: usize> {
    map: &'a AtomicSlotMap<T, N>,
    next: usize,
}

impl<'a, T, const N: usize> Iterator for SlotWriteIter<'a, T, N> {
    type Item = (SlotId, SlotWrite<'a, T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < N {
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

impl<T, const N: usize> SlotWrite<'_, T, N> {
    #[inline]
    pub const fn id(&self) -> SlotId {
        self.id
    }

    /// Consume the claim and retire this generation, returning its value.
    pub fn remove(mut self) -> Result<T, SlotAccess> {
        let index = self.id.index as usize;
        // SAFETY: this guard uniquely owns the writer state and drained every
        // reader before construction.
        let value = unsafe { (*self.map.slots[index].get()).assume_init_read() };
        let mut generation = self.id.generation.wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        self.map.states[index].store(generation as u64, Ordering::Release);
        self.map.live.fetch_sub(1, Ordering::Relaxed);
        self.armed = false;
        Ok(value)
    }
}

impl<T, const N: usize> Deref for SlotWrite<'_, T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let index = self.id.index as usize;
        // SAFETY: the live writer exclusion is held for this guard's lifetime.
        unsafe { (*self.map.slots[index].get()).assume_init_ref() }
    }
}

impl<T, const N: usize> DerefMut for SlotWrite<'_, T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let index = self.id.index as usize;
        // SAFETY: the live writer exclusion is held uniquely by this guard.
        unsafe { (*self.map.slots[index].get()).assume_init_mut() }
    }
}

impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for SlotWrite<'_, T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlotWrite")
            .field("id", &self.id)
            .field("value", &**self)
            .finish()
    }
}

impl<T, const N: usize> Drop for SlotWrite<'_, T, N> {
    fn drop(&mut self) {
        if self.armed {
            self.map.states[self.id.index as usize]
                .store(self.id.generation as u64 | LIVE, Ordering::Release);
        }
    }
}

pub(crate) fn classify(observed: u64, wanted_generation: u32) -> SlotAccess {
    if observed & LIVE != 0
        && observed & WRITING != 0
        && (observed & GEN_MASK) as u32 == wanted_generation
    {
        SlotAccess::Busy
    } else {
        SlotAccess::Stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn generations_fail_closed_after_reuse() {
        let slots = AtomicSlotMap::<u32, 1>::new();
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
    fn full_map_returns_the_uninserted_value() {
        let slots = AtomicSlotMap::<u32, 1>::new();
        slots.insert(1).unwrap();
        assert_eq!(slots.insert(2), Err(2));
        assert_eq!(slots.len(), 1);
    }

    #[test]
    fn one_claim_blocks_only_its_exact_slot() {
        let slots = AtomicSlotMap::<u32, 2>::new();
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
        let slots = AtomicSlotMap::<u32, 1>::new();
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
        let slots = Arc::new(AtomicSlotMap::<u32, 2>::new());
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
    fn unwind_republishes_a_claimed_slot() {
        let slots = AtomicSlotMap::<u32, 1>::new();
        let id = slots.insert(1).unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut claim = slots.try_write(id).unwrap();
            *claim = 5;
            panic!("exercise the guard");
        }));
        assert_eq!(slots.read(id, |v| *v), Ok(5));
    }

    #[test]
    fn scan_skips_a_contended_slot_without_hiding_others() {
        let slots = AtomicSlotMap::<u32, 3>::new();
        let first = slots.insert(1).unwrap();
        let second = slots.insert(2).unwrap();
        let _held = slots.try_write(first).unwrap();
        let mut seen = [0u32; 3];
        slots.for_each(|id, value| seen[id.index() as usize] = *value);
        assert_eq!(seen[first.index() as usize], 0);
        assert_eq!(seen[second.index() as usize], 2);
    }
}
