//! Sparse, concurrently published pages over the full `u32` key space.
//!
//! A page is a caller-defined `T`: arrays of atomics, an immutable object, or
//! another collection. Slot allocation and capability policy stay with that
//! caller. Keys have a monotone lifecycle: absent -> live -> retiring -> retired.
//! Retirement is permanent. Backing allocations can be reused by the allocator,
//! but an old key never names a new object. There are no wrapping generations,
//! recycled pointer headers, or allocator calls in lookup or logical retirement.
//!
//! Three lazy radix branches and a leaf select the four bytes of a key. The
//! inline root costs 2 KiB on 64-bit targets. Published directories never move or
//! disappear before store drop. Only payloads are reclaimed. A leaf's pin
//! counters are transposed: one cache-aligned row per shard, one column per key.
//! Eight shards cost 32 bytes per leaf entry, not eight padded cache lines.
//!
//! # Concurrency proof
//!
//! Readers announce in permanent directory storage BEFORE loading a payload
//! pointer. Retirement replaces that pointer with a tombstone, seals every
//! counter, and queues the old pointer in the same entry. SeqCst orders the
//! announcement, pointer observation, and sealing: a reader that observed the
//! old pointer must either still be counted or have finished before reclamation.
//! No payload address is dereferenced before admission. Counters never reopen.
//! A delayed reader therefore cannot race a header reset or touch freed memory.
//!
//! Guards remember their counter, so migration between workers is supported.
//! A guard preserves memory, not current authorization: existing guards may
//! continue reading after retirement. Callers must separately validate policy.
//!
//! Publication and admission are lock-free apart from the caller's allocator
//! and initializer. Retirement uses a bounded number of atomic operations.
//! Reclamation is explicit and never waits for a guard. Destructors and allocator
//! deallocation have their own progress guarantees. There is no global refcount.

use alloc::boxed::Box;
use core::{
    alloc::{AllocError, Allocator},
    marker::PhantomData,
    ops::Deref,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicU32, Ordering},
};

const FANOUT: usize = 256;
const SEALED: u32 = 1 << 31;
const COUNT: u32 = SEALED - 1;

// Address-only sentinels are never dereferenced. Live payloads are wrapped in an
// over-aligned allocation, including zero-sized or byte-aligned user types.
fn closing<T>() -> *mut Payload<T> {
    ptr::without_provenance_mut(1)
}
fn retired<T>() -> *mut Payload<T> {
    ptr::without_provenance_mut(2)
}
fn live<T>(p: *mut Payload<T>) -> bool {
    p.addr() > 2
}

#[repr(C, align(8))]
struct Payload<T>(T);

struct Branch<T> {
    entries: [AtomicPtr<T>; FANOUT],
}
impl<T> Branch<T> {
    const fn new() -> Self {
        Self {
            entries: [const { AtomicPtr::new(ptr::null_mut()) }; FANOUT],
        }
    }
}

#[repr(align(128))]
struct PinRow([AtomicU32; FANOUT]);
struct Leaf<T, const S: usize> {
    entries: [AtomicPtr<Payload<T>>; FANOUT],
    pending: [AtomicPtr<Payload<T>>; FANOUT],
    pins: [PinRow; S],
}

/// Outcome of a page lookup, including permanent retirement after reclamation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagePresence {
    Absent,
    Live,
    Retiring,
    Retired,
}

/// Fallible publication. A retired key can never be republished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageError {
    Alloc,
    Retired,
}
impl From<AllocError> for PageError {
    fn from(_: AllocError) -> Self {
        Self::Alloc
    }
}

/// Allocator-owned sparse page directory. `S` is a fixed, nonzero shard count.
/// Select the shard at lookup (e.g. worker id); it is retained by the guard.
///
/// Directory allocations persist as tombstones until drop. Payload allocations
/// are returned by `reclaim`; allocator exhaustion leaves existing pages intact.
pub struct PageStore<T, A: Allocator, const S: usize = 1> {
    root: Branch<Branch<Branch<Leaf<T, S>>>>,
    alloc: A,
    owns: PhantomData<T>,
}

// SAFETY: published directories are immutable except for atomics. Payload
// access requires a pin. Moving the store transfers ownership of T and A.
unsafe impl<T: Send, A: Allocator + Send, const S: usize> Send for PageStore<T, A, S> {}
// SAFETY: shared readers see only &T; reclamation can drop T on another thread.
unsafe impl<T: Send + Sync, A: Allocator + Sync, const S: usize> Sync for PageStore<T, A, S> {}

impl<T, A: Allocator, const S: usize> PageStore<T, A, S> {
    pub const fn new_in(alloc: A) -> Self {
        assert!(S > 0, "page store needs at least one pin shard");
        Self {
            root: Branch::new(),
            alloc,
            owns: PhantomData,
        }
    }

    fn branch<U>(
        &self,
        entry: &AtomicPtr<Branch<U>>,
        prepare: bool,
    ) -> Result<Option<&Branch<U>>, PageError> {
        let mut p = entry.load(Ordering::Acquire);
        if p.is_null() && prepare {
            let fresh =
                Box::into_raw_with_allocator(Box::try_new_in(Branch::new(), &self.alloc)?).0;
            match entry.compare_exchange(p, fresh, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => p = fresh,
                Err(winner) => {
                    // SAFETY: the losing allocation was never published.
                    unsafe {
                        drop(Box::from_raw_in(fresh, &self.alloc));
                    }
                    p = winner;
                }
            }
        }
        // SAFETY: published directories live for the store's entire lifetime.
        Ok(if p.is_null() {
            None
        } else {
            Some(unsafe { &*p })
        })
    }

    fn leaf(&self, key: u32, prepare: bool) -> Result<Option<&Leaf<T, S>>, PageError> {
        let Some(a) = self.branch(&self.root.entries[(key >> 24) as usize], prepare)? else {
            return Ok(None);
        };
        let Some(b) = self.branch(&a.entries[((key >> 16) & 255) as usize], prepare)? else {
            return Ok(None);
        };
        let entry = &b.entries[((key >> 8) & 255) as usize];
        let mut p = entry.load(Ordering::Acquire);
        if p.is_null() && prepare {
            // Initialize in the allocation, never putting a multi-shard leaf
            // (12 KiB for S=8) on a kernel/syscall stack.
            let mut boxed = Box::<Leaf<T, S>, _>::try_new_uninit_in(&self.alloc)?;
            let raw = boxed.as_mut_ptr();
            // SAFETY: this is a fresh, exclusive, correctly aligned allocation.
            // Every field is initialized before assume_init or publication.
            unsafe {
                for i in 0..FANOUT {
                    ptr::addr_of_mut!((*raw).entries[i]).write(AtomicPtr::new(ptr::null_mut()));
                    ptr::addr_of_mut!((*raw).pending[i]).write(AtomicPtr::new(ptr::null_mut()));
                    for shard in 0..S {
                        ptr::addr_of_mut!((*raw).pins[shard].0[i]).write(AtomicU32::new(0));
                    }
                }
            }
            // SAFETY: every field was initialized above.
            let fresh = Box::into_raw_with_allocator(unsafe { boxed.assume_init() }).0;
            match entry.compare_exchange(p, fresh, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => p = fresh,
                Err(winner) => {
                    // SAFETY: the losing leaf has never been published.
                    unsafe {
                        drop(Box::from_raw_in(fresh, &self.alloc));
                    }
                    p = winner;
                }
            }
        }
        // SAFETY: published directories outlive all guards borrowing the store.
        Ok(if p.is_null() {
            None
        } else {
            Some(unsafe { &*p })
        })
    }

    /// Prepare one page. The initializer is called only if no page was seen;
    /// concurrent preparers may each initialize, but exactly one value wins.
    /// Losing values are dropped. No outstanding reservation can publish after
    /// retirement: publication CASes the same word that retirement tombstones.
    pub fn prepare(&self, key: u32, init: impl FnOnce() -> T) -> Result<(), PageError> {
        let leaf = self.leaf(key, true)?.expect("prepared leaf");
        let entry = &leaf.entries[(key & 255) as usize];
        let old = entry.load(Ordering::Acquire);
        if live(old) {
            return Ok(());
        }
        if !old.is_null() {
            return Err(PageError::Retired);
        }
        let fresh = Box::into_raw_with_allocator(Box::try_new_in(Payload(init()), &self.alloc)?).0;
        match entry.compare_exchange(ptr::null_mut(), fresh, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Ok(()),
            Err(winner) => {
                // SAFETY: this payload was never reachable from the directory.
                unsafe {
                    drop(Box::from_raw_in(fresh, &self.alloc));
                }
                if live(winner) {
                    Ok(())
                } else {
                    Err(PageError::Retired)
                }
            }
        }
    }

    pub fn presence(&self, key: u32) -> PagePresence {
        let Some(leaf) = self.leaf(key, false).ok().flatten() else {
            return PagePresence::Absent;
        };
        let p = leaf.entries[(key & 255) as usize].load(Ordering::Acquire);
        if p.is_null() {
            PagePresence::Absent
        } else if p == closing() {
            PagePresence::Retiring
        } else if p == retired() {
            PagePresence::Retired
        } else {
            PagePresence::Live
        }
    }

    /// Pin one page. No allocation or payload access precedes admission.
    /// Counter exhaustion refuses admission; it can never wrap into zero.
    pub fn pin(&self, key: u32, shard: usize) -> Option<PageGuard<'_, T>> {
        let leaf = self.leaf(key, false).ok().flatten()?;
        let slot = (key & 255) as usize;
        let counter = &leaf.pins[shard % S].0[slot];
        counter
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < COUNT).then(|| n + 1)
            })
            .ok()?;
        let ptr = leaf.entries[slot].load(Ordering::SeqCst);
        if !live(ptr) {
            counter.fetch_sub(1, Ordering::Release);
            return None;
        }
        Some(PageGuard {
            ptr: NonNull::new(ptr).expect("live payload"),
            counter,
        })
    }

    /// Permanently close a key, including a prepared but empty page. Returns
    /// true only for the thread that retired a live payload. No allocation,
    /// destructor, waiting, or retry loop. An absent key is unchanged.
    pub fn retire(&self, key: u32) -> bool {
        let Some(leaf) = self.leaf(key, false).ok().flatten() else {
            return false;
        };
        let slot = (key & 255) as usize;
        let old = leaf.entries[slot].load(Ordering::SeqCst);
        if !live(old) {
            return false;
        }
        if leaf.entries[slot]
            .compare_exchange(old, closing(), Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        for row in &leaf.pins {
            row.0[slot].fetch_or(SEALED, Ordering::SeqCst);
        }
        leaf.pending[slot].store(old, Ordering::Release);
        true
    }

    /// Try to reclaim one retired payload. Returns immediately if a guard is
    /// outstanding. Exactly one caller drops/deallocates the payload. A leaked
    /// guard leaks only this payload; the directory and other pages progress.
    pub fn reclaim(&self, key: u32) -> bool {
        let Some(leaf) = self.leaf(key, false).ok().flatten() else {
            return false;
        };
        let slot = (key & 255) as usize;
        let p = leaf.pending[slot].load(Ordering::Acquire);
        if p.is_null() {
            return false;
        }
        if leaf
            .pins
            .iter()
            .any(|row| row.0[slot].load(Ordering::Acquire) != SEALED)
        {
            return false;
        }
        if leaf.pending[slot]
            .compare_exchange(p, ptr::null_mut(), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        // SAFETY: all counters are permanently sealed and empty; the successful
        // CAS transfers sole ownership of the disconnected payload to us.
        unsafe {
            drop(Box::from_raw_in(p, &self.alloc));
        }
        leaf.entries[slot].store(retired(), Ordering::Release);
        true
    }

    /// Find the next published key (including tombstones). Empty subtrees are
    /// skipped. `start` is u64 so advancing past u32::MAX does not wrap.
    pub fn next_key(&self, start: u64) -> Option<u32> {
        if start > u32::MAX as u64 {
            return None;
        }
        for a in (start >> 24) as usize..FANOUT {
            let ap = self.root.entries[a].load(Ordering::Acquire);
            if ap.is_null() {
                continue;
            }
            // SAFETY: these directories remain allocated until store drop.
            let branch = unsafe { &*ap };
            for b in 0..FANOUT {
                let prefix = ((a as u64) << 24) | ((b as u64) << 16);
                if prefix + 0xffff < start {
                    continue;
                }
                let bp = branch.entries[b].load(Ordering::Acquire);
                if bp.is_null() {
                    continue;
                }
                // SAFETY: immutable published directory.
                let branch = unsafe { &*bp };
                for c in 0..FANOUT {
                    let prefix = prefix | ((c as u64) << 8);
                    if prefix + 255 < start {
                        continue;
                    }
                    let lp = branch.entries[c].load(Ordering::Acquire);
                    if lp.is_null() {
                        continue;
                    }
                    // SAFETY: immutable published leaf.
                    let leaf = unsafe { &*lp };
                    for d in start.saturating_sub(prefix).min(256) as usize..FANOUT {
                        if !leaf.entries[d].load(Ordering::Acquire).is_null() {
                            return Some((prefix | d as u64) as u32);
                        }
                    }
                }
            }
        }
        None
    }

    /// Incremental sweep; budget counts visited published keys, including
    /// pinned/previously reclaimed ones. Returns reclaimed payload count.
    /// The caller owns its cursor, so independent maintenance passes coexist.
    pub fn reclaim_step(&self, cursor: &mut u64, budget: usize) -> usize {
        let mut reclaimed = 0;
        for _ in 0..budget {
            let Some(key) = self.next_key(*cursor) else {
                *cursor = 0;
                break;
            };
            reclaimed += usize::from(self.reclaim(key));
            *cursor = key as u64 + 1;
        }
        reclaimed
    }
}

/// An affine page pin. Dropping on another worker releases the original shard.
/// It permits shared payload access; mutation belongs to T's own safe API.
pub struct PageGuard<'a, T> {
    ptr: NonNull<Payload<T>>,
    counter: &'a AtomicU32,
}
// SAFETY: a guard only shares T, and its permanent atomic counter is not
// thread-affine. Ownership of the payload remains with the borrowed store.
unsafe impl<T: Sync> Send for PageGuard<'_, T> {}
// SAFETY: concurrent dereferences yield shared references to a Sync T.
unsafe impl<T: Sync> Sync for PageGuard<'_, T> {}
impl<T> Deref for PageGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: admission preceded pointer loading; this counted guard keeps
        // the payload alive until every borrow through it has ended.
        unsafe { &self.ptr.as_ref().0 }
    }
}
impl<T> Drop for PageGuard<'_, T> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

impl<T, A: Allocator, const S: usize> Drop for PageStore<T, A, S> {
    fn drop(&mut self) {
        // SAFETY: exclusive store access excludes every operation and guard.
        // Each branch/leaf was published once; pending and live payloads are
        // disjoint. Tombstones are never reconstructed as pointers to objects.
        unsafe {
            for a in &mut self.root.entries {
                let ap = *a.get_mut();
                if ap.is_null() {
                    continue;
                }
                let mut a = Box::from_raw_in(ap, &self.alloc);
                for b in &mut a.entries {
                    let bp = *b.get_mut();
                    if bp.is_null() {
                        continue;
                    }
                    let mut b = Box::from_raw_in(bp, &self.alloc);
                    for c in &mut b.entries {
                        let lp = *c.get_mut();
                        if lp.is_null() {
                            continue;
                        }
                        let mut leaf = Box::from_raw_in(lp, &self.alloc);
                        for i in 0..FANOUT {
                            let p = *leaf.entries[i].get_mut();
                            if live(p) {
                                drop(Box::from_raw_in(p, &self.alloc));
                            }
                            let p = *leaf.pending[i].get_mut();
                            if !p.is_null() {
                                drop(Box::from_raw_in(p, &self.alloc));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    struct CountDrop<'a>(&'a AtomicUsize);
    impl Drop for CountDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn full_width_keys_and_sparse_iteration_do_not_wrap() {
        let arena = TestArena::<{ 256 << 10 }>::new();
        let store = PageStore::<_, _, 8>::new_in(&arena);
        let keys = [0, 255, 256, 65_536, 0xff00_0100, u32::MAX];
        for &key in &keys {
            store.prepare(key, || key).unwrap();
        }
        let mut cursor = 0;
        for key in keys {
            assert_eq!(store.next_key(cursor), Some(key));
            assert_eq!(*store.pin(key, 7).unwrap(), key);
            cursor = key as u64 + 1;
        }
        assert_eq!(store.next_key(cursor), None);
        assert_eq!(store.presence(100), PagePresence::Absent);
    }

    #[test]
    fn retired_identity_survives_reclamation_and_empty_payload() {
        let arena = TestArena::<{ 64 << 10 }>::new();
        let store = PageStore::<_, _, 1>::new_in(&arena);
        store.prepare(1, || ()).unwrap();
        let guard = store.pin(1, 0).unwrap();
        assert!(store.retire(1));
        assert_eq!(store.presence(1), PagePresence::Retiring);
        assert!(store.pin(1, 0).is_none());
        assert!(!store.reclaim(1));
        drop(guard);
        assert!(store.reclaim(1));
        assert!(!store.reclaim(1));
        assert_eq!(store.presence(1), PagePresence::Retired);
        assert_eq!(store.prepare(1, || ()), Err(PageError::Retired));
    }

    #[test]
    fn old_guard_can_migrate_and_only_delays_its_page() {
        let arena = TestArena::<{ 64 << 10 }>::new();
        let drops = AtomicUsize::new(0);
        let store = PageStore::<_, _, 8>::new_in(&arena);
        store.prepare(1, || CountDrop(&drops)).unwrap();
        store.prepare(2, || CountDrop(&drops)).unwrap();
        let held = store.pin(1, 3).unwrap();
        assert!(store.retire(1));
        assert!(store.retire(2));
        let mut cursor = 0;
        assert_eq!(store.reclaim_step(&mut cursor, 2), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        std::thread::scope(|s| {
            s.spawn(move || drop(held)).join().unwrap();
        });
        assert!(store.reclaim(1));
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        drop(store);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn delayed_initializer_cannot_publish_over_retirement() {
        let arena = TestArena::<{ 64 << 10 }>::new();
        let drops = AtomicUsize::new(0);
        let store = PageStore::<_, _, 8>::new_in(&arena);
        let entered = Barrier::new(2);
        let resume = Barrier::new(2);
        std::thread::scope(|s| {
            let delayed = s.spawn(|| {
                store.prepare(7, || {
                    entered.wait();
                    resume.wait();
                    CountDrop(&drops)
                })
            });
            entered.wait();
            store.prepare(7, || CountDrop(&drops)).unwrap();
            assert!(store.retire(7));
            assert!(store.reclaim(7));
            resume.wait();
            assert_eq!(delayed.join().unwrap(), Err(PageError::Retired));
        });
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        assert_eq!(store.presence(7), PagePresence::Retired);
    }

    #[test]
    fn all_shards_prevent_reclamation_until_every_guard_drains() {
        let arena = TestArena::<{ 64 << 10 }>::new();
        let drops = AtomicUsize::new(0);
        let store = PageStore::<_, _, 8>::new_in(&arena);
        store.prepare(0, || CountDrop(&drops)).unwrap();
        let entered = Barrier::new(9);
        let release = Barrier::new(9);
        std::thread::scope(|s| {
            for shard in 0..8 {
                let store = &store;
                let entered = &entered;
                let release = &release;
                s.spawn(move || {
                    let guard = store.pin(0, shard).unwrap();
                    entered.wait();
                    release.wait();
                    assert_eq!(guard.0.load(Ordering::Relaxed), 0);
                    drop(guard);
                });
            }
            entered.wait();
            assert!(store.retire(0));
            assert!(!store.reclaim(0));
            for shard in 0..8 {
                assert!(store.pin(0, shard).is_none());
            }
            release.wait();
        });
        assert!(store.reclaim(0));
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn pin_retire_race_never_observes_dropped_payload() {
        let keys = if cfg!(miri) { 8 } else { 128 };
        let rounds = if cfg!(miri) { 2 } else { 50 };
        let arena = TestArena::<{ 256 << 10 }>::new();
        let drops = AtomicUsize::new(0);
        let store = PageStore::<_, _, 8>::new_in(&arena);
        for key in 0..keys {
            store.prepare(key, || CountDrop(&drops)).unwrap();
        }
        std::thread::scope(|s| {
            for shard in 0..8 {
                let store = &store;
                s.spawn(move || {
                    for _ in 0..rounds {
                        for key in 0..keys {
                            if let Some(guard) = store.pin(key, shard) {
                                core::hint::black_box(&*guard);
                            }
                        }
                    }
                });
            }
            for key in 0..keys {
                store.retire(key);
                store.reclaim(key);
            }
        });
        for key in 0..keys {
            store.reclaim(key);
        }
        assert_eq!(drops.load(Ordering::Relaxed), keys as usize);
    }

    #[test]
    fn counter_saturation_refuses_without_wrapping() {
        let arena = TestArena::<{ 32 << 10 }>::new();
        let store = PageStore::<_, _, 1>::new_in(&arena);
        store.prepare(0, || 42).unwrap();
        let leaf = store.leaf(0, false).unwrap().unwrap();
        leaf.pins[0].0[0].store(COUNT, Ordering::SeqCst);
        assert!(store.pin(0, 0).is_none());
        assert_eq!(leaf.pins[0].0[0].load(Ordering::SeqCst), COUNT);
        leaf.pins[0].0[0].store(0, Ordering::SeqCst);
        assert_eq!(*store.pin(0, 0).unwrap(), 42);
    }

    #[test]
    fn allocation_failure_does_not_publish_or_run_initializer() {
        let arena = TestArena::<1>::new();
        let store = PageStore::<u32, _, 8>::new_in(&arena);
        assert_eq!(
            store.prepare(1, || panic!("no directory backing")),
            Err(PageError::Alloc)
        );
        assert_eq!(store.presence(1), PagePresence::Absent);
    }

    #[test]
    fn transposed_shards_have_separate_lines_without_per_counter_padding() {
        assert_eq!(core::mem::size_of::<PinRow>(), 256 * 4);
        let arena = TestArena::<{ 32 << 10 }>::new();
        let store = PageStore::<_, _, 8>::new_in(&arena);
        store.prepare(0, || ()).unwrap();
        let leaf = store.leaf(0, false).unwrap().unwrap();
        for shard in 1..8 {
            let a = ptr::addr_of!(leaf.pins[shard - 1].0[0]).addr();
            let b = ptr::addr_of!(leaf.pins[shard].0[0]).addr();
            assert_eq!(b - a, 1024);
            assert_eq!(b % 128, 0);
        }
    }
}
