//! Size-class geometry for the segregated free-lists (mimalloc-shape).
//!
//! A sharded allocator serves small/medium requests from per-shard
//! free-lists, one list **per size class**.  This module is the pure
//! arithmetic that maps a request to its class and back — no state, no
//! allocation, so both the hot path ([`class_index`]) and the carve path
//! ([`class_size`]) share one definition and can never disagree.
//!
//! # Why power-of-two classes
//!
//! Classes are the powers of two from [`MIN_BLOCK`] (16 B) up to
//! [`LARGE_THRESHOLD`] (64 KiB).  The choice is deliberate: it makes the
//! **alignment contract fall out for free**.  A request folds its size and
//! its alignment into one figure — `need = max(size, align)` — and the class
//! is `ceil_pow2(need)`.  In the shared lane ([`carve_lane`]) blocks of a
//! class are laid out contiguously from a class-aligned offset, so every block
//! is `class_size`-aligned; since `class_size >= need >= align` and both are
//! powers of two, the block is `align`-aligned by construction (see the `Heap`
//! carve path).  A large class's blocks are page-aligned instead (whole pages
//! after a header page), and the `Heap` sends a large request aligned past a
//! page to the huge route, so the contract holds there too.  Coarser than
//! mimalloc's ~3-bit-mantissa classes — up to 2x internal waste — but the
//! correctness argument is a one-liner, which is the right trade for a
//! from-scratch allocator whose first duty is *never hand back a bad pointer*.
//!
//! Requests above [`LARGE_THRESHOLD`] leave the segregated path entirely and
//! take the `Heap`'s whole-extent "huge" route, which aligns the extent to the
//! request directly.

use core::sync::atomic::{AtomicPtr, Ordering};

/// The smallest block a class serves, and the allocator's minimum alignment
/// (16 B — `max_align_t` on the targets, and enough to hold the intrusive
/// free-list link a freed block stores in its own first word).
pub(crate) const MIN_BLOCK: usize = 16;

/// The largest segregated class.  A `need` above this is a "huge" request the
/// `Heap` serves from a dedicated whole-extent allocation, not a free-list.
///
/// 64 KiB, up from 8 KiB: a huge extent is a mapping's grain (64 KiB) at
/// the least, so every request between the two cost a whole one — and a
/// compiler's working set lives there.  Compiling CPython on the host
/// harness asked 59,024 times for 8–16 KiB, 37,116 for 16–32 KiB and
/// 16,035 for 32–64 KiB (137 MiB live at their peak), and mapped 364 MiB
/// for them as huge extents.  As classes they recycle exactly and share
/// class-sized extents ([`carve_lane`]).
pub(crate) const LARGE_THRESHOLD: usize = 64 * 1024;

/// Number of segregated size classes — the powers of two in
/// `[MIN_BLOCK, LARGE_THRESHOLD]`, i.e. `16, 32, …, 65536` (indices `0..=12`).
pub(crate) const NUM_CLASSES: usize = 13;

/// The last class that carves from the shard's shared bump extent: 8 KiB.
/// Classes above it carve from extents of their own ([`carve_lane`]).
pub(crate) const SHARED_LANE_TOP_CLASS: usize = 9;

/// Carve lanes per shard: the shared one, then one per class above
/// [`SHARED_LANE_TOP_CLASS`] (16, 32 and 64 KiB).
pub(crate) const NUM_LANES: usize = NUM_CLASSES - SHARED_LANE_TOP_CLASS;

/// Which of a shard's current extents class `class` carves fresh blocks
/// from.
///
/// Small classes share one bump extent: a 16 B block after a 4 KiB one
/// costs at most that much alignment padding, and a 64 KiB extent holds
/// many of either.  A block of 16 KiB and up carved from a shared extent
/// would pad the cursor to its own size on every alternation with a small
/// class — up to the block's size lost per carve — and a 64 KiB block does
/// not fit a 64 KiB extent at all once the header is at its base.  So each
/// large class carves from extents of its own (`Heap::pool_extent_bytes`):
/// a header page, then its blocks back to back — one in a shard's first,
/// twice the last's in each later one, up to eight.
#[inline]
pub(crate) const fn carve_lane(class: usize) -> usize {
    if class <= SHARED_LANE_TOP_CLASS {
        0
    } else {
        class - SHARED_LANE_TOP_CLASS
    }
}

// The class table and the huge threshold are one definition, pinned so a later
// edit to one constant that desyncs the other fails the build, not a test.
const _: () = {
    assert!(MIN_BLOCK << (NUM_CLASSES - 1) == LARGE_THRESHOLD);
    assert!(MIN_BLOCK.is_power_of_two());
    assert!(LARGE_THRESHOLD.is_power_of_two());
    assert!(MIN_BLOCK << SHARED_LANE_TOP_CLASS == 8192);
    assert!(NUM_LANES == 4);
};

/// Round `x` up to a multiple of `align` (a power of two).  The workhorse for
/// carving aligned sub-blocks and aligning extent payloads.
#[inline]
pub(crate) const fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + (align - 1)) & !(align - 1)
}

/// Checked sibling of [`align_up`] for caller- or address-derived geometry.
///
/// The ordinary helper remains const and intentionally terse for bounded class
/// sizes.  Mapping sizes and virtual addresses are not similarly bounded: an
/// overflow there must refuse an allocation, never wrap into an overlapping
/// extent.
#[inline]
pub(crate) const fn checked_align_up(x: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    match x.checked_add(align - 1) {
        Some(sum) => Some(sum & !(align - 1)),
        None => None,
    }
}

/// Does this request escape the segregated classes onto the huge path?
/// `need` is the alignment-folded requirement (`max(size, align)`).
#[inline]
pub(crate) const fn is_huge(need: usize) -> bool {
    need > LARGE_THRESHOLD
}

/// The class index serving `need` (`max(size, align)` of the request).
///
/// Caller guarantees `!is_huge(need)`; the result is in `0..NUM_CLASSES` and
/// [`class_size`] of it is `>= need`.  `need = 1..16 → 0`, `17..32 → 1`, …,
/// `32769..65536 → 12`.
#[inline]
pub(crate) fn class_index(need: usize) -> usize {
    debug_assert!(!is_huge(need));
    let block = need.max(MIN_BLOCK).next_power_of_two();
    // `block` is a power of two in `[16, 65536]`, so `trailing_zeros` is its
    // log2; subtract log2(MIN_BLOCK)=4 to index from the first class.
    (block.trailing_zeros() - MIN_BLOCK.trailing_zeros()) as usize
}

/// The exact byte size of every block in class `idx` (a power of two).
#[inline]
pub(crate) const fn class_size(idx: usize) -> usize {
    MIN_BLOCK << idx
}

// ── Intrusive free-list links ────────────────────────────────────────────────
//
// A free block *is* its own list node: its first word holds the successor
// pointer.  No external metadata table means no metadata to allocate — the
// allocator never has to allocate in order to allocate (the self-hosting rule).
// A block is `>= MIN_BLOCK` (16 B) and `class_size`-aligned (`>= 8`), so the
// word write is in-bounds and aligned.

/// Read the intrusive successor stored in a free block's first word.
///
/// # Safety
/// `block` must point at a currently-free block of a class list (`>= 8` bytes,
/// 8-aligned) — i.e. one this module linked, not a live allocation.
#[inline]
pub(crate) unsafe fn block_next(block: *mut u8) -> *mut u8 {
    // SAFETY: `block` is a free block ≥ 8 bytes and ≥ 8-aligned (its class size
    // is a power of two ≥ 16), so its first word is a valid `*mut u8` slot.
    unsafe { block.cast::<*mut u8>().read() }
}

/// Write the intrusive successor into a free block's first word.
///
/// # Safety
/// `block` must point at a block being placed on a free list (`>= 8` bytes,
/// 8-aligned) that no live reference aliases.
#[inline]
pub(crate) unsafe fn set_block_next(block: *mut u8, next: *mut u8) {
    // SAFETY: see [`block_next`] — first word is a valid, unaliased `*mut u8`.
    unsafe { block.cast::<*mut u8>().write(next) }
}

/// Push `block` onto a lock-free Treiber stack whose head is `head`.
///
/// Used for the cross-thread remote-free lists.  Correct for **many concurrent
/// producers**: each links its node to the observed head and CAS-swings the
/// head, retrying on loss.  ABA cannot corrupt a push — a failed CAS simply
/// re-reads (the node address a producer installs is always fresh), and the
/// single consumer drains by *swapping the whole stack out* ([`drain_stack`])
/// rather than popping, so it never races a producer onto a reused node.
///
/// # Safety
/// `block` is an unaliased free block (`>= 8` B, 8-aligned); `head` is the
/// stack this block is being handed to.
#[inline]
pub(crate) unsafe fn push_stack(head: &AtomicPtr<u8>, block: *mut u8) {
    let mut cur = head.load(Ordering::Relaxed);
    loop {
        // SAFETY: caller's contract — `block` is an unaliased free block.
        unsafe { set_block_next(block, cur) };
        // Release so the consumer's Acquire swap observes the linked `next`.
        match head.compare_exchange_weak(cur, block, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// Atomically detach the entire Treiber stack, returning its head (or null).
///
/// The **single consumer** calls this; swapping the whole chain out in one
/// atomic step (rather than a CAS-pop loop) is what makes the MPSC stack
/// ABA-safe without generation tags — the consumer never observes, then
/// re-CASes, an individual node a producer could have recycled underneath it.
/// After the swap the returned chain is private to the consumer.
#[inline]
pub(crate) fn drain_stack(head: &AtomicPtr<u8>) -> *mut u8 {
    // Acquire pairs with the producers' Release pushes: every linked `next` in
    // the detached chain is visible.
    head.swap(core::ptr::null_mut(), Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_cover_and_round_up() {
        // Every class maps to itself; sizes just under a class round up to it.
        assert_eq!(class_index(1), 0);
        assert_eq!(class_index(16), 0);
        assert_eq!(class_size(0), 16);
        assert_eq!(class_index(17), 1);
        assert_eq!(class_size(1), 32);
        assert_eq!(class_index(4097), 9);
        assert_eq!(class_index(LARGE_THRESHOLD), NUM_CLASSES - 1);
        assert_eq!(class_size(NUM_CLASSES - 1), LARGE_THRESHOLD);
    }

    #[test]
    fn class_size_always_covers_the_request() {
        // The core allocator guarantee: the served block is ≥ the ask.
        for need in [
            1usize, 8, 15, 16, 31, 33, 100, 1000, 4096, 8191, 8192, 8193, 16384, 40_000, 65536,
        ] {
            let sz = class_size(class_index(need));
            assert!(sz >= need, "class {sz} must cover need {need}");
            assert!(sz.is_power_of_two());
        }
    }

    #[test]
    fn class_size_covers_alignment_because_need_folds_it() {
        // The alignment contract: with need = max(size, align), the class is a
        // power of two ≥ align, so a class-aligned block is align-aligned.
        for (size, align) in [(8usize, 64usize), (16, 256), (1, 4096), (100, 8)] {
            let need = size.max(align);
            if is_huge(need) {
                continue;
            }
            let sz = class_size(class_index(need));
            assert!(sz >= align, "class {sz} must cover align {align}");
            assert!(sz >= size);
            assert_eq!(sz % align, 0, "a class-aligned block satisfies align");
        }
    }

    #[test]
    fn huge_threshold_is_exclusive() {
        assert!(!is_huge(LARGE_THRESHOLD));
        assert!(is_huge(LARGE_THRESHOLD + 1));
    }

    #[test]
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(checked_align_up(4095, 4096), Some(4096));
        assert_eq!(checked_align_up(usize::MAX, 4096), None);
    }

    #[test]
    fn intrusive_links_round_trip() {
        // A block's first word holds its successor — the whole free-list state.
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        unsafe {
            set_block_next(pa, pb);
            assert_eq!(block_next(pa), pb);
            set_block_next(pa, core::ptr::null_mut());
            assert!(block_next(pa).is_null());
        }
    }
}
