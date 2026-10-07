//! `store` — the link engine's **caller-provided capacity** (Phase A,
//! `res/USERLAND.md` §7): the fixed `[T; MAX_*]` tables become
//! [`Allocator`]-parameterized slots carved from an arena the *caller*
//! owns — the abi cap.rs TODO's direction, realized here first.
//!
//! Two pieces:
//!
//! - [`LinkArena`] — a bump allocator over a **borrowed buffer**
//!   (`core::alloc::Allocator`, no `alloc` crate anywhere: the linker
//!   stub must keep linking with **no global allocator at all**).  The
//!   stub sizes one buffer for its whole link; a test leaks one; either
//!   way the capacity decision moves to the caller, and a small program
//!   (a static exe, two modules) pays for two modules' tables, not
//!   sixteen.
//! - `Slots` (crate-internal) — a fixed-capacity, fully-initialized
//!   slice allocated from any `A: Allocator` (arena-backed in
//!   practice), the engine's table primitive.  Never grows; capacity is
//!   decided at construction ([`LinkCaps`](crate::link::LinkCaps) sizes
//!   it), so the bounded-everything regime is unchanged — only *who
//!   chooses the bound* moved.
//!
//! Dropping a `Slots` drops its elements and returns the block to its
//! allocator (a no-op for the bump arena — the buffer frees wholesale
//! when the caller's borrow ends).

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut};
use core::ptr::NonNull;

/// A bump allocator over a caller-borrowed byte buffer — the link
/// engine's arena.  `allocate` bumps (aligned, bounds-checked,
/// fail-closed [`AllocError`] on exhaustion); `deallocate` is a no-op
/// (arena discipline: the buffer is reclaimed wholesale when the borrow
/// ends).  Single-threaded (`Cell` cursor) — the linker stub is one
/// task, once, at startup.
pub struct LinkArena<'a> {
    base: *mut u8,
    len: usize,
    cursor: Cell<usize>,
    _buf: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

impl<'a> LinkArena<'a> {
    /// An arena over `buf` — the caller's capacity decision, in bytes.
    pub fn new(buf: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            base: buf.as_mut_ptr() as *mut u8,
            len: buf.len(),
            cursor: Cell::new(0),
            _buf: PhantomData,
        }
    }

    /// Bytes handed out so far (telemetry — what a stub's buffer must
    /// at least hold for a given link).
    pub fn used(&self) -> usize {
        self.cursor.get()
    }
}

// SAFETY: `allocate` returns blocks that stay valid and disjoint until
// the arena's borrow ends (the buffer is exclusively borrowed for 'a,
// the cursor only grows, and deallocate never recycles); clones of the
// handle (`&LinkArena`) share the one cursor.
unsafe impl Allocator for LinkArena<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            // A zero-size allocation is a dangling, aligned, valid-empty
            // block (the Allocator contract's degenerate case).
            let dangling = NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }
        let cur = self.cursor.get();
        // Align the next block relative to the buffer's real address.
        let at = (self.base as usize).checked_add(cur).ok_or(AllocError)?;
        let aligned = at.checked_add(layout.align() - 1).ok_or(AllocError)? & !(layout.align() - 1);
        let pad = aligned - at;
        let start = cur.checked_add(pad).ok_or(AllocError)?;
        let end = start.checked_add(layout.size()).ok_or(AllocError)?;
        if end > self.len {
            return Err(AllocError); // exhausted — fail closed, never wrap
        }
        self.cursor.set(end);
        // SAFETY: `start..end` is in bounds of the borrowed buffer.
        let ptr = unsafe { self.base.add(start) };
        Ok(NonNull::slice_from_raw_parts(
            NonNull::new(ptr).ok_or(AllocError)?,
            layout.size(),
        ))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Bump arena: individual frees are no-ops; the buffer is
        // reclaimed wholesale when the caller's borrow ends.
    }
}

/// A fixed-capacity, always-fully-initialized slice allocated from `A`
/// — the engine's table primitive (the `[T; MAX]` replacement).  The
/// capacity is chosen at construction and never changes; element access
/// is the plain slice API.
pub(crate) struct Slots<T, A: Allocator> {
    ptr: NonNull<T>,
    cap: usize,
    alloc: A,
}

impl<T, A: Allocator> Slots<T, A> {
    /// Allocate `cap` slots from `alloc`, initializing slot `i` with
    /// `f(i)`.  Zero-capacity allocates nothing (a dangling, never-read
    /// pointer).
    pub(crate) fn new_from(
        cap: usize,
        alloc: A,
        mut f: impl FnMut(usize) -> T,
    ) -> Result<Self, AllocError> {
        if cap == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                cap: 0,
                alloc,
            });
        }
        let layout = Layout::array::<T>(cap).map_err(|_| AllocError)?;
        let raw = alloc.allocate(layout)?;
        let ptr = raw.cast::<T>();
        for i in 0..cap {
            // SAFETY: `i < cap`, inside the just-allocated block.
            unsafe { ptr.add(i).write(f(i)) };
        }
        Ok(Self { ptr, cap, alloc })
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `cap` elements were initialized at construction and
        // stay owned by this value.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, with `&mut self` exclusivity.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }
}

impl<T, A: Allocator> Drop for Slots<T, A> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // SAFETY: fully initialized; dropped exactly once here.
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap));
            // Layout::array succeeded at construction for this cap.
            let layout = Layout::array::<T>(self.cap).unwrap_unchecked();
            self.alloc.deallocate(self.ptr.cast(), layout);
        }
    }
}

impl<T, A: Allocator> Index<usize> for Slots<T, A> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.as_slice()[i]
    }
}

impl<T, A: Allocator> IndexMut<usize> for Slots<T, A> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.as_mut_slice()[i]
    }
}

impl<T: core::fmt::Debug, A: Allocator> core::fmt::Debug for Slots<T, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_slice().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_bumps_aligned_and_fails_closed() {
        let mut buf = [MaybeUninit::<u8>::uninit(); 256];
        let arena = LinkArena::new(&mut buf);

        let a = arena
            .allocate(Layout::from_size_align(10, 1).unwrap())
            .expect("fits");
        let b = arena
            .allocate(Layout::from_size_align(16, 16).unwrap())
            .expect("fits aligned");
        assert_eq!(b.cast::<u8>().as_ptr() as usize % 16, 0, "aligned bump");
        assert_ne!(a.cast::<u8>().as_ptr(), b.cast::<u8>().as_ptr());
        // Exhaustion refuses, never wraps.
        assert!(
            arena
                .allocate(Layout::from_size_align(1024, 1).unwrap())
                .is_err(),
            "over-ask fails closed"
        );
        // Zero-size succeeds without consuming the buffer.
        let used = arena.used();
        arena
            .allocate(Layout::from_size_align(0, 8).unwrap())
            .expect("zero-size is the degenerate success");
        assert_eq!(arena.used(), used);
    }

    #[test]
    fn slots_init_index_and_drop() {
        let mut buf = [MaybeUninit::<u8>::uninit(); 1024];
        let arena = LinkArena::new(&mut buf);
        let mut s = Slots::<u64, _>::new_from(8, &arena, |i| i as u64 * 3).expect("slots");
        assert_eq!(s.as_slice().len(), 8);
        assert_eq!(s[5], 15);
        s[5] = 99;
        assert_eq!(s.as_slice()[5], 99);

        // Zero-capacity is a valid empty table.
        let z = Slots::<u64, _>::new_from(0, &arena, |_| 0).expect("empty");
        assert!(z.as_slice().is_empty());

        // Exhaustion propagates as AllocError.
        assert!(Slots::<u64, _>::new_from(4096, &arena, |_| 0).is_err());
    }
}
