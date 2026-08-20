//! Test-local allocation for the arena-parameterized structures.
//!
//! The allocator-parameterized collections here ([`BitVec`](crate::BitVec),
//! [`BitMat`](crate::BitMat)) need *an* allocator to be exercised, and
//! `res/ALLOC.md`'s rule holds in tests too: a named, tightly-scoped arena,
//! never `alloc::alloc::Global`.  The kernel's `BumpArena` cannot serve —
//! it stays kernel-side because its failure path attributes OOM to a named
//! *kernel* allocator, which is policy this crate has no business carrying.
//!
//! So this is a deliberately minimal bump arena: enough to hand out aligned
//! blocks that live for the test, and nothing else.  It is test-only
//! scaffolding, not a second production arena.

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

/// `N` bytes of storage, over-aligned so a caller asking for `u64` /
/// `AtomicU64` / `u128` granularity can actually be satisfied.
///
/// Load-bearing: a bare `[u8; N]` is 1-aligned, so aligning the *offset*
/// into it yields an address that is only aligned if the block itself
/// happened to be — which is how the first draft of this arena handed a
/// `BitVec` a misaligned word array and segfaulted on the first store.
#[repr(align(128))]
struct Storage<const N: usize>([u8; N]);

/// A fixed `N`-byte bump arena with an **atomic** cursor.
///
/// Atomic because the tests hold their arena in a `static` and the harness
/// runs them on parallel threads — a plain `UnsafeCell<usize>` cursor here
/// handed two tests the same block and failed *intermittently*, which is
/// worse than failing.  The kernel's `BumpArena` makes the same choice for
/// the same reason.
pub struct TestArena<const N: usize> {
    bytes: UnsafeCell<Storage<N>>,
    head: AtomicUsize,
}

// SAFETY: the only mutable state is `head`, which is atomic; every handed-out
// block is a disjoint sub-range carved by a successful CAS, and no block is
// ever handed out twice or reclaimed.
unsafe impl<const N: usize> Sync for TestArena<N> {}

impl<const N: usize> Default for TestArena<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> TestArena<N> {
    /// A fresh, empty arena.
    pub const fn new() -> Self {
        Self {
            bytes: UnsafeCell::new(Storage([0u8; N])),
            head: AtomicUsize::new(0),
        }
    }
}

// SAFETY: `allocate` hands out disjoint sub-blocks of `bytes` and never
// reuses one; `deallocate` is a no-op, so no block is ever invalidated while
// a caller holds it (a bump arena's whole contract).
unsafe impl<const N: usize> Allocator for TestArena<N> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let base = self.bytes.get() as *mut u8;
        let mut cur = self.head.load(Ordering::Relaxed);
        loop {
            // Align the real ADDRESS, not the offset: the two agree only
            // when the block itself is already aligned, and relying on that
            // is the bug this comment exists to keep fixed.  `Storage`'s
            // over-alignment is what makes the request satisfiable at all.
            let addr = (base as usize).checked_add(cur).ok_or(AllocError)?;
            let aligned =
                addr.checked_add(layout.align() - 1).ok_or(AllocError)? & !(layout.align() - 1);
            let start = aligned - base as usize;
            let end = start.checked_add(layout.size()).ok_or(AllocError)?;
            if end > N {
                return Err(AllocError);
            }
            // The CAS is what makes the carve exclusive: the winner owns
            // `start..end` and nobody else can be handed it.
            match self
                .head
                .compare_exchange_weak(cur, end, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    // SAFETY: `start..end` is in bounds of the `N`-byte
                    // block and was claimed exclusively by this CAS.
                    let ptr = unsafe { base.add(start) };
                    return NonNull::new(ptr)
                        .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
                        .ok_or(AllocError);
                }
                Err(seen) => cur = seen,
            }
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // A bump arena frees only by going out of scope.
    }
}
