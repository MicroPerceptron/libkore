//! Shards — the per-task slice of the allocator, and how a task finds its own.
//!
//! Each task allocates from **its own shard**, so the hot path (pop a block off
//! a size-class free-list, push one back) touches only that task's memory and
//! executes **zero atomic operations** — the mimalloc property that makes a
//! sharded allocator scale: N cores never serialize on one cache line.
//!
//! Two lists per size class live in a shard:
//!
//! - a **local** free-list ([`Shard::local_pop`]/[`Shard::local_push`]),
//!   touched only by the owning task, so it needs no synchronization at all;
//! - a **remote** free-list ([`Shard::remote_push`]), a lock-free MPSC Treiber
//!   stack onto which *other* tasks push blocks they freed but this shard owns.
//!   The owner drains it ([`Shard::reclaim_remote`]) on its own allocation
//!   miss, folding the cross-thread frees back into the local list.
//!
//! The single-owner discipline of the local lists is what lets them be
//! `UnsafeCell` rather than atomics; it rests on [`current_shard`] being an
//! **injective** task→shard map (no two live tasks share a shard).  The real
//! wiring guarantees this per task; the host-test wiring claims a distinct
//! shard per OS thread (see [`current_shard`]).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicPtr};

use super::Extent;
use super::pool::{self, NUM_CLASSES};

/// The number of shards.  A design constant, not a discovery: the ceiling on
/// concurrently-allocating tasks that each get a lock-free lane before two must
/// share one.  32 tracks `MAX_THREADS` (the G4c widening) — a shard is a
/// const-empty bundle of list heads, so the wider `static` costs pointers,
/// not pages.
pub(crate) const NUM_SHARDS: usize = 32;

/// One task's allocator lane: a local (owner-only) and a remote (MPSC)
/// free-list per size class, plus the pool extent currently being carved.
///
/// Not `Sync` on its own (the local lists and carve cursor are `UnsafeCell`s);
/// the enclosing [`Heap`](super::Heap) carries the `unsafe impl Sync` together
/// with the argument that discharges it (the single-owner invariant above).
pub(crate) struct Shard {
    /// Owner-only intrusive free-list heads, one per class.  A freed block owned
    /// by this shard, freed *by this task*, lands here — no atomics.
    local: [UnsafeCell<*mut u8>; NUM_CLASSES],
    /// MPSC remote-free stacks: other tasks push here (lock-free), the owner
    /// drains.  One per class so a drain is targeted and the class of a remote
    /// block is implicit in which stack holds it (no per-block class tag).
    remote: [AtomicPtr<u8>; NUM_CLASSES],
    /// The pool [`Extent`](super::Extent) this shard bump-carves fresh blocks
    /// from, or null before its first grow.  Owner-only.
    current: UnsafeCell<*mut Extent>,
    /// Debug-only single-owner detector: set while a thread is inside this
    /// shard's owner-only region ([`Shard::owner_guard`]).  A second concurrent
    /// entry — the exact precondition for the `UnsafeCell` data race a
    /// non-injective [`current_shard`] would cause — trips an assert, turning
    /// silent release-build UB into a loud panic in debug/test.  Free and
    /// untouched in release (where it is never read — hence the allow).
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    owner_active: AtomicBool,
}

// SAFETY: a `&Shard` is shared across threads (it lives inside the `Heap`), but
// the `UnsafeCell` state — the local free-lists and the current-extent pointer —
// is reached only through the `unsafe` owner-only methods, whose contract is
// that the caller is the shard's single owning task (`current_shard()`).  The
// remote free-lists are atomics, safe for the many-producer/one-consumer split.
// This is the same discipline the enclosing `Heap`'s `Sync` impl rests on, and
// `owner_guard` enforces it at runtime in debug/test (a loud panic if two
// threads ever enter one shard's owner-only region).
unsafe impl Sync for Shard {}

impl Shard {
    /// An empty shard — every list empty, no extent yet.  `const`, so a
    /// `[Shard; NUM_SHARDS]` lives in a `Heap` placed in a `static`.
    pub(crate) const fn new() -> Self {
        Self {
            local: [const { UnsafeCell::new(core::ptr::null_mut()) }; NUM_CLASSES],
            remote: [const { AtomicPtr::new(core::ptr::null_mut()) }; NUM_CLASSES],
            current: UnsafeCell::new(core::ptr::null_mut()),
            owner_active: AtomicBool::new(false),
        }
    }

    /// Pop a block off the local free-list for `class`, or null if empty.
    ///
    /// # Safety
    /// The caller must be this shard's owning task (`current_shard()` names this
    /// shard) — the local lists admit exactly one accessor.
    #[inline]
    pub(crate) unsafe fn local_pop(&self, class: usize) -> *mut u8 {
        debug_assert!(class < NUM_CLASSES);
        // SAFETY: owner-only access (caller contract); the cell holds the list
        // head, and a non-null head is a free block whose first word is its
        // successor.
        unsafe {
            let head = self.local[class].get();
            let block = *head;
            if !block.is_null() {
                *head = pool::block_next(block);
            }
            block
        }
    }

    /// Push `block` onto the local free-list for `class`.
    ///
    /// # Safety
    /// Caller must be this shard's owning task, and `block` must be an unaliased
    /// block of `class` (`>= class_size` bytes, correctly aligned).
    #[inline]
    pub(crate) unsafe fn local_push(&self, class: usize, block: *mut u8) {
        debug_assert!(class < NUM_CLASSES);
        // SAFETY: owner-only access (caller contract); `block` is unaliased, so
        // writing its intrusive link and swinging the head is a private update.
        unsafe {
            let head = self.local[class].get();
            pool::set_block_next(block, *head);
            *head = block;
        }
    }

    /// Push `block` onto the remote free-list for `class` — the cross-thread
    /// free path.  Lock-free and safe from **any** thread (this is the "someone
    /// else owns it" branch of a free).
    ///
    /// # Safety
    /// `block` must be an unaliased block of `class` owned by this shard.
    #[inline]
    pub(crate) unsafe fn remote_push(&self, class: usize, block: *mut u8) {
        debug_assert!(class < NUM_CLASSES);
        // SAFETY: `block` is unaliased (caller contract); `push_stack` links and
        // CAS-publishes it — many producers safe.
        unsafe { pool::push_stack(&self.remote[class], block) }
    }

    /// Drain the remote free-list for `class` into the local list, returning
    /// whether anything was reclaimed.  The single-consumer half of the MPSC
    /// stack: it swaps the whole chain out atomically, then threads each block
    /// onto the local list.
    ///
    /// # Safety
    /// Caller must be this shard's owning task (single consumer).
    #[inline]
    pub(crate) unsafe fn reclaim_remote(&self, class: usize) -> bool {
        debug_assert!(class < NUM_CLASSES);
        let mut node = pool::drain_stack(&self.remote[class]);
        let mut reclaimed = false;
        while !node.is_null() {
            // SAFETY: `node` is a detached (now private) free block; read its
            // successor before `local_push` overwrites the same word.
            unsafe {
                let next = pool::block_next(node);
                self.local_push(class, node);
                node = next;
            }
            reclaimed = true;
        }
        reclaimed
    }

    /// The pool extent this shard is carving from (null before the first grow).
    ///
    /// # Safety
    /// Owner-only read.
    #[inline]
    pub(crate) unsafe fn current(&self) -> *mut Extent {
        // SAFETY: owner-only access (caller contract).
        unsafe { *self.current.get() }
    }

    /// Set the pool extent this shard carves from (after a grow).
    ///
    /// # Safety
    /// Owner-only write.
    #[inline]
    pub(crate) unsafe fn set_current(&self, extent: *mut Extent) {
        // SAFETY: owner-only access (caller contract).
        unsafe { *self.current.get() = extent }
    }

    /// Enter this shard's owner-only region, returning a guard that marks it busy
    /// until dropped.  A **detector, not a lock**: in debug/test it panics if
    /// another thread is already inside the same shard's region (the precise
    /// precondition for the local-free-list data race); in release it compiles to
    /// nothing.  Acquire it *once* at the top of an owner-only operation (alloc's
    /// local-pop / carve / grow, or the owner branch of a free) — never per inner
    /// method, which would false-positive on the `reclaim_remote` → `local_push`
    /// nesting.
    ///
    /// Wiring [`current_shard`] to an injective per-task index is the real fix;
    /// this guard only guarantees a violation surfaces as a panic, not corruption.
    #[inline]
    pub(crate) fn owner_guard(&self) -> OwnerGuard<'_> {
        #[cfg(debug_assertions)]
        {
            let already = self
                .owner_active
                .swap(true, core::sync::atomic::Ordering::AcqRel);
            assert!(
                !already,
                "kore-rt heap: concurrent owner-only access to one shard — \
                 current_shard() is not injective (UB in release). Two live \
                 threads reported one index: either their runtime blocks are \
                 not distinct thread_block band slots, or a block was never \
                 anchored and fell back to shard 0."
            );
        }
        OwnerGuard { shard: self }
    }
}

/// The witness returned by [`Shard::owner_guard`]: clears the shard's owner-busy
/// flag on drop.  In release the flag is never touched, so this is a zero-cost
/// no-op; the `&Shard` borrow ties it to the shard it guards.
#[must_use]
pub(crate) struct OwnerGuard<'a> {
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    shard: &'a Shard,
}

impl Drop for OwnerGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.shard
            .owner_active
            .store(false, core::sync::atomic::Ordering::Release);
    }
}

/// The index of the shard the **current thread** allocates from — the one seam
/// between the engine and the per-thread runtime.
///
/// Keeping shard selection behind this single function is the point: the engine
/// is written against "my shard" and the per-thread wiring lives only here.
///
/// This reads the thread's `thread_block` band slot, stamped into its runtime
/// block at anchor ([`shard_index`](crate::entry::geometry::shard_index)) and
/// reached through the thread pointer — the same one-register read every other
/// per-thread accessor makes, and no atomics on the hot path.
///
/// **Why this is injective**, which is the invariant the owner-only local
/// free-lists rest on: a `thread_block` band slot is address space held
/// exclusively by one live thread ([`BANDS`](crate::thread::BANDS)), and
/// [`Band::slot_of`] is exact, so two live threads cannot report the same
/// index.  [`MAX_THREADS`](crate::thread::MAX_THREADS) slots map onto
/// [`NUM_SHARDS`] lanes one-for-one, asserted below, so the map needs no
/// modulo and no two threads are folded onto one lane.
///
/// An unstamped block — pre-anchor, or a block outside the declared band —
/// answers shard 0.  Pre-anchor that is exact (the anchoring thread is the
/// only one running, and the heap already cannot grow before the anchor,
/// since its supply slot is read from the same block).  Out-of-band it is a
/// fallback, not a proof: `Shard::owner_guard` turns the collision it would
/// take into a debug panic rather than silent corruption.  No plan in the
/// tree produces one — the founder's task plan aliases slot 0 and
/// `thread::spawn` draws every other window from the band itself.
#[cfg(all(not(any(test, feature = "std")), target_os = "none"))]
#[inline]
pub(crate) fn current_shard() -> usize {
    crate::entry::geometry::shard_index().unwrap_or(0)
}

/// Off-metal, non-test: there is no thread pointer to read, and no second
/// thread to be wrong about — `thread::spawn` cannot run at all, because
/// every commit refuses [`TrapUnavailable`](crate::transport::TrapUnavailable)
/// off the metal target.  One live thread makes shard 0 exact here, the same
/// way it is exact pre-anchor.
#[cfg(all(not(any(test, feature = "std")), not(target_os = "none")))]
#[inline]
pub(crate) fn current_shard() -> usize {
    0
}

/// One band slot, one shard: the band is what proves two live threads differ,
/// so a band wider than the shard array would fold two provably-distinct
/// threads onto one set of owner-only free-lists.
#[cfg(not(any(test, feature = "std")))]
const _: () = assert!(
    crate::thread::MAX_THREADS as usize <= NUM_SHARDS,
    "every thread band slot needs its own shard: MAX_THREADS exceeds NUM_SHARDS"
);

/// Host-test shard selection: hand **each OS thread a distinct shard**, claimed
/// once on first touch.  libtest runs allocation across real threads, and the
/// local free-lists demand an injective task→shard map, so a stable *and*
/// unique per-thread index is required — a hash of the thread id would be stable
/// but could collide two threads onto one shard and race its local list.  The
/// claim uses one atomic *per thread* (not per allocation), so the hot path
/// stays atomic-free.  Tests keep concurrently-allocating threads `<= NUM_SHARDS`
/// so the round-robin claim never wraps two live threads onto one shard.
#[cfg(any(test, feature = "std"))]
#[inline]
pub(crate) fn current_shard() -> usize {
    use core::sync::atomic::{AtomicUsize, Ordering};
    static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);
    std::thread_local! {
        static MY_SHARD: core::cell::Cell<usize> = const { core::cell::Cell::new(usize::MAX) };
    }
    MY_SHARD.with(|cell| {
        let mut v = cell.get();
        if v == usize::MAX {
            v = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % NUM_SHARDS;
            cell.set(v);
        }
        v
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_list_is_lifo() {
        let shard = Shard::new();
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        unsafe {
            assert!(shard.local_pop(0).is_null(), "empty list pops null");
            shard.local_push(0, pa);
            shard.local_push(0, pb);
            assert_eq!(shard.local_pop(0), pb, "LIFO: last pushed pops first");
            assert_eq!(shard.local_pop(0), pa);
            assert!(shard.local_pop(0).is_null());
        }
    }

    #[test]
    fn remote_push_then_owner_reclaims() {
        let shard = Shard::new();
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        unsafe {
            shard.remote_push(0, pa);
            shard.remote_push(0, pb);
            // Nothing is on the local list until the owner drains the remote.
            assert!(shard.local_pop(0).is_null());
            assert!(shard.reclaim_remote(0), "drain reports work done");
            // Both blocks are now local and poppable.
            let x = shard.local_pop(0);
            let y = shard.local_pop(0);
            assert!(!x.is_null() && !y.is_null());
            assert!((x == pa && y == pb) || (x == pb && y == pa));
            assert!(shard.local_pop(0).is_null());
            assert!(!shard.reclaim_remote(0), "empty remote reports no work");
        }
    }

    #[test]
    fn remote_stack_is_mpsc_safe_under_contention() {
        // Many producers push distinct blocks onto one shard's remote list; the
        // single consumer drains. Every block must be accounted for exactly once
        // — the MPSC ABA-safety claim, exercised on real parallelism.
        use alloc::vec::Vec;
        use std::collections::HashSet;
        const PRODUCERS: usize = 6;
        const PER: usize = 2000;
        // Backing storage for the blocks (16 B each, distinct addresses).
        let backing: Vec<[u8; 16]> = alloc::vec![[0u8; 16]; PRODUCERS * PER];
        let base = backing.as_ptr() as usize;

        let shard = Shard::new();
        let drained: std::sync::Mutex<HashSet<usize>> = std::sync::Mutex::new(HashSet::new());

        std::thread::scope(|s| {
            for p in 0..PRODUCERS {
                let shard = &shard;
                s.spawn(move || {
                    for i in 0..PER {
                        let addr = base + (p * PER + i) * 16;
                        // SAFETY: distinct, unaliased 16-byte block in `backing`.
                        unsafe { shard.remote_push(0, addr as *mut u8) };
                    }
                });
            }
            // Consumer: drain repeatedly until all blocks seen.
            let shard = &shard;
            let drained = &drained;
            s.spawn(move || {
                let mut seen = 0;
                while seen < PRODUCERS * PER {
                    // SAFETY: this closure is the sole consumer of the remote list.
                    unsafe {
                        let mut node = pool::drain_stack(&shard.remote[0]);
                        while !node.is_null() {
                            let next = pool::block_next(node);
                            assert!(
                                drained.lock().unwrap().insert(node as usize),
                                "block drained twice — MPSC corruption"
                            );
                            seen += 1;
                            node = next;
                        }
                    }
                    std::thread::yield_now();
                }
            });
        });

        assert_eq!(
            drained.lock().unwrap().len(),
            PRODUCERS * PER,
            "no block lost"
        );
    }

    #[test]
    fn shard_selection_is_stable_and_spreads() {
        // Stable within a thread — the invariant the owner-only local lists rely
        // on (a task must keep the same shard for the life of its allocations).
        let (x, y) = std::thread::spawn(|| (current_shard(), current_shard()))
            .join()
            .unwrap();
        assert_eq!(x, y, "one thread keeps one shard");
        assert!(x < NUM_SHARDS);

        // Distinct threads spread across shards, so the cross-shard remote-free
        // path is reachable. At least two shards appear over several threads —
        // robust to the process-global round-robin being advanced concurrently by
        // other tests (which asserting two *specific* threads differ would not be).
        use std::collections::HashSet;
        let mut shards = HashSet::new();
        for _ in 0..NUM_SHARDS {
            shards.insert(std::thread::spawn(current_shard).join().unwrap());
        }
        assert!(shards.len() >= 2, "distinct threads reach distinct shards");
    }

    /// The debug single-owner guard fires when two owner entries overlap on one
    /// shard — the precise precondition a non-injective `current_shard()` would
    /// create.  Debug-only: the guard compiles to nothing in release, so does the
    /// test (else `#[should_panic]` would fail on a release `cargo test`).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "not injective")]
    fn owner_guard_detects_concurrent_owner_access() {
        let shard = Shard::new();
        let _g1 = shard.owner_guard();
        let _g2 = shard.owner_guard(); // overlapping entry on the same shard → panic
    }
}
