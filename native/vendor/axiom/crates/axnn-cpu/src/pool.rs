//! Bump arena for tensor storage allocation.
//!
//! Uses interior mutability (`Cell` cursor) so that allocations
//! go through `&self`, allowing allocated slices and new allocations
//! to coexist without borrow conflicts.

use core::cell::{Cell, UnsafeCell};

/// Alignment for all arena allocations (32 bytes = AVX2 YMM).
const ALIGN: usize = 32;

/// Backing storage, aligned to `ALIGN` in absolute address. A bare
/// `[u8; CAP]` field has alignment 1, so the cursor math below would
/// only align *offsets* — the absolute address of an allocation would
/// be left to field-layout luck. The typed reinterprets taken over
/// tensor storage (`&[f32]` / `&[u32]` / `&[f64]` views in
/// `backend.rs`) require the real thing.
#[repr(align(32))]
struct AlignedBuf<const CAP: usize>([u8; CAP]);

/// A fixed-size bump allocator for tensor byte buffers.
///
/// Allocations are served from a flat `[u8; CAP]` by advancing a
/// cursor. No per-allocation free — the entire arena is reclaimed
/// at once via `reset()` (which requires `&mut self`, enforcing
/// that no live slices exist).
pub struct BumpArena<const CAP: usize> {
    // `alloc_zeroed` intentionally initializes disjoint regions through
    // `&self`; the backing allocation therefore requires Rust's explicit
    // interior-mutability marker rather than casting an ordinary shared field.
    buf: UnsafeCell<AlignedBuf<CAP>>,
    cursor: Cell<usize>,
}

impl<const CAP: usize> BumpArena<CAP> {
    pub const fn new() -> Self {
        Self {
            buf: UnsafeCell::new(AlignedBuf([0u8; CAP])),
            cursor: Cell::new(0),
        }
    }

    #[allow(dead_code)]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    pub fn used(&self) -> usize {
        self.cursor.get()
    }

    pub fn remaining(&self) -> usize {
        CAP - self.cursor.get()
    }

    /// Allocate `size` bytes, zero-initialized, returning a mutable slice.
    ///
    /// Takes `&self` (not `&mut self`) thanks to the `Cell` cursor.
    /// The returned `&mut [u8]` is safe because the bump only moves
    /// forward — each allocation occupies a unique, non-overlapping
    /// region that is never handed out again until `reset()`.
    ///
    /// # Safety justification
    /// The `&self` → `&mut [u8]` conversion is safe because:
    /// 1. Each allocation covers a unique byte range (cursor only increases).
    /// 2. No two calls to `alloc_zeroed` ever return overlapping ranges.
    /// 3. `reset()` requires `&mut self`, which the borrow checker
    ///    enforces is exclusive (no outstanding `&self` borrows).
    pub fn alloc_zeroed(&self, size: usize) -> Option<&mut [u8]> {
        let cur = self.cursor.get();
        let aligned = cur.checked_add(ALIGN - 1)? & !(ALIGN - 1);
        let end = aligned.checked_add(size)?;
        if end > CAP {
            return None;
        }
        self.cursor.set(end);
        // SAFETY: see justification above. `buf` is an `UnsafeCell`, and the
        // monotonic cursor guarantees this region has not been handed out.
        let ptr = unsafe { (*self.buf.get()).0.as_mut_ptr().add(aligned) };
        let slice = unsafe { core::slice::from_raw_parts_mut(ptr, size) };
        slice.fill(0);
        Some(slice)
    }

    /// Reset the arena, reclaiming all allocations.
    ///
    /// Requires `&mut self` — the borrow checker ensures no
    /// outstanding slices exist.
    pub fn reset(&mut self) {
        self.cursor.set(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_and_reset() {
        let arena = BumpArena::<1024>::new();
        let a = arena.alloc_zeroed(64).unwrap();
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&b| b == 0));
        a[0] = 42;

        let b = arena.alloc_zeroed(128).unwrap();
        assert_eq!(b.len(), 128);
        assert!(arena.used() >= 64 + 128);
    }

    #[test]
    fn alignment_spacing() {
        let arena = BumpArena::<1024>::new();
        let a = arena.alloc_zeroed(7).unwrap();
        let b = arena.alloc_zeroed(16).unwrap();
        let gap = b.as_ptr() as usize - a.as_ptr() as usize;
        assert!(gap >= 7);
        assert_eq!(gap % ALIGN, 0);
        // Absolute alignment, not just offset spacing — `AlignedBuf`
        // anchors the whole arena, so typed reinterprets of tensor
        // storage (f32/u32/f64 views) are sound.
        assert_eq!(a.as_ptr() as usize % ALIGN, 0);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
    }

    #[test]
    fn oom_returns_none() {
        let arena = BumpArena::<64>::new();
        assert!(arena.alloc_zeroed(32).is_some());
        assert!(arena.alloc_zeroed(64).is_none());
    }

    #[test]
    fn cursor_alignment_overflow_returns_none() {
        let arena = BumpArena::<64>::new();
        arena.cursor.set(usize::MAX - (ALIGN - 2));
        assert!(arena.alloc_zeroed(1).is_none());
    }

    #[test]
    fn reset_requires_exclusive() {
        let mut arena = BumpArena::<256>::new();
        let _ = arena.alloc_zeroed(64).unwrap();
        assert!(arena.used() > 0);
        arena.reset();
        assert_eq!(arena.used(), 0);
    }
}
