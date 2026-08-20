//! Runtime-sized bitsets, backed by an `Allocator`.
//!
//! Two siblings, mirroring [`super::bitset`]:
//!
//! - [`BitVec<A>`] — single-threaded, `u128` words, strict point-in-time
//!   semantics, `&mut self` mutation.
//! - [`AtomicBitVec<A>`] — SMP-safe, `AtomicU64` words, `&self` methods,
//!   wait-free single-bit ops, lock-free multi-bit claims with
//!   bounded-retry rollback.
//!
//! # When to use this versus `BitSet`
//!
//! `BitSet<const N: usize>` bakes the bit count into the type — perfect
//! for things sized at compile time (per-CPU vector mask, fixed-width
//! IRQ bitmap, the static ASID pool).  `BitVec<A>` runs on a runtime bit
//! count — perfect for per-instance data where each instance has its
//! own size (per-region frame bitmap, per-address-space bookkeeping).
//!
//! Both share the same flat single-storage layout (data words ‖ nonzero
//! summary [‖ full summary]) and the same find-acceleration tricks.
//!
//! # Allocator integration
//!
//! Backed by `alloc::Vec<_, &'a A>`.  Construction takes an `&'a A`
//! reference to your allocator (a [`super::TestArena`], in practice);
//! the bitvec lives as long as the allocator does.  Drop is the Vec's
//! drop, which on `TestArena` is a no-op (the arena never reclaims).
//!
//! Construction panics on allocator exhaustion — same convention as
//! [`super::pool::ResourcePool`] and the rest of the catalog.  Boot-time
//! bit-budget violations are programmer errors, not runtime conditions.

use core::{
    alloc::Allocator,
    sync::atomic::{AtomicU64, Ordering},
};

use alloc::vec::Vec;

// =============================================================================
// Geometry helpers
//
// `BitVec` retains its data + nonzero summary + full summary layout, and
// `AtomicBitVec` retains its data + nonzero summary layout — the
// summary-tier acceleration matters for the runtime-sized bitvec, which
// can scale to hundreds of millions of bits (per-region frame bitmaps).
// The fixed-size `BitSet`/`AtomicBitSet` siblings dropped their summaries
// because their consumers stay in the hundreds-of-bits range; that
// tradeoff doesn't carry here.
// =============================================================================

#[inline]
const fn words_128(bits: usize) -> usize {
    (bits + 127) / 128
}

#[inline]
const fn words_64(bits: usize) -> usize {
    (bits + 63) / 64
}

/// Total `u128` words backing a [`BitVec`]: data + `nonzero` summary +
/// `full` summary.
#[inline]
const fn bitvec_storage(bits: usize) -> usize {
    let w = words_128(bits);
    let s = words_128(w);
    w + s + s
}

/// Total `AtomicU64` words backing an [`AtomicBitVec`]: data + `nonzero`
/// summary.  No `full` summary — keeping it consistent under contention
/// would require races that violate find-first-clear's contract.
#[inline]
const fn atomic_bitvec_storage(bits: usize) -> usize {
    let w = words_64(bits);
    let s = words_64(w);
    w + s
}

// =============================================================================
// BitVec (single-threaded)
// =============================================================================

/// Runtime-sized single-threaded bitset.
///
/// Storage layout matches [`super::BitSet`]: one flat
/// `Vec<u128, &'a A>` holding `data ‖ nonzero ‖ full`.
#[derive(Debug)]
pub struct BitVec<'a, A: Allocator + 'a> {
    storage: Vec<u128, &'a A>,
    bits: usize,
    /// Cached offsets into `storage`.  Computed once at construction;
    /// every access path uses these instead of recomputing from `bits`.
    words: usize,
    nonzero_at: usize,
    full_at: usize,
}

impl<'a, A: Allocator + 'a> BitVec<'a, A> {
    /// Build a `BitVec` holding `bits` bits, backed by `alloc`.  Every
    /// bit starts cleared.
    ///
    /// # Panics
    ///
    /// Panics if `alloc` cannot satisfy the storage allocation.
    pub fn new_in(bits: usize, alloc: &'a A) -> Self {
        let words = words_128(bits);
        let summary = words_128(words);
        let total = bitvec_storage(bits);
        debug_assert_eq!(total, words + summary + summary);

        let mut storage = Vec::with_capacity_in(total, alloc);
        storage.resize(total, 0u128);

        Self {
            storage,
            bits,
            words,
            nonzero_at: words,
            full_at: words + summary,
        }
    }

    /// Logical bit count.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.bits
    }

    /// Internal: number of `u128` data words in `storage`.
    #[inline]
    fn data_words(&self) -> usize {
        self.words
    }

    /// Read-only view of the data words (summary tiers excluded), for
    /// word-parallel set algebra against [`BitMat`](super::BitMat) rows
    /// via the [`bitmat`](super::bitmat) kernels.  Read-only by design:
    /// external word writes would desynchronise the `nonzero`/`full`
    /// summaries, so the mutating side of a cross-op must be the
    /// `BitMat` row.
    #[inline]
    pub fn as_data_words(&self) -> &[u128] {
        &self.storage[..self.words]
    }

    /// Internal: number of summary words per tier (nonzero / full).
    #[inline]
    fn summary_words(&self) -> usize {
        self.full_at - self.nonzero_at
    }

    pub fn clear_all(&mut self) {
        self.storage.fill(0);
    }

    #[inline(always)]
    pub fn contains(&self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        (self.storage[w] & mask) != 0
    }

    #[inline(always)]
    pub fn set(&mut self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        let before = self.storage[w];
        let after = before | mask;
        self.storage[w] = after;
        // Unconditional summary update is cheaper than a branch and
        // idempotent.
        let summary_idx = self.nonzero_at + w / 128;
        self.storage[summary_idx] |= 1u128 << (w % 128);
        let full_idx = self.full_at + w / 128;
        self.storage[full_idx] |= ((after == u128::MAX) as u128) << (w % 128);
        (before & mask) == 0
    }

    #[inline(always)]
    pub fn clear(&mut self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        let before = self.storage[w];
        let after = before & !mask;
        self.storage[w] = after;
        let summary_idx = self.nonzero_at + w / 128;
        self.storage[summary_idx] &= !(((after == 0) as u128) << (w % 128));
        let full_idx = self.full_at + w / 128;
        self.storage[full_idx] &= !(1u128 << (w % 128));
        (before & mask) != 0
    }

    pub fn count_ones(&self) -> usize {
        self.storage[..self.data_words()]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn pop_first_set(&mut self) -> Option<usize> {
        let summary = self.summary_words();
        for si in 0..summary {
            let s = self.storage[self.nonzero_at + si];
            if s == 0 {
                continue;
            }
            let w = si * 128 + s.trailing_zeros() as usize;
            // Defensive symmetry with `find_first_set_from` / `find_first_clear_from`,
            // which both bound the data-word index before indexing `storage[w]`.
            // In correct operation the nonzero summary only sets bits for valid
            // data words (`w < words`), so this never trips — but a summary that
            // ever claimed a padding word would otherwise index past the data
            // region into the summary tiers.
            if w >= self.words {
                return None;
            }
            let bit = self.storage[w].trailing_zeros() as usize;
            self.storage[w] &= self.storage[w] - 1;
            if self.storage[w] == 0 {
                self.storage[self.nonzero_at + si] = s & (s - 1);
            }
            self.storage[self.full_at + w / 128] &= !(1u128 << (w % 128));
            return Some(w * 128 + bit);
        }
        None
    }

    pub fn find_first_set_from(&self, start: usize) -> Option<usize> {
        if start >= self.bits {
            return None;
        }
        let sw = start / 128;
        let sb = start % 128;

        let first = self.storage[sw] & (!0u128 << sb);
        if first != 0 {
            let bit = sw * 128 + first.trailing_zeros() as usize;
            return (bit < self.bits).then_some(bit);
        }

        let summary = self.summary_words();
        let mut si = sw / 128;
        let sub = (sw % 128) as u32 + 1;
        let mut s = self.storage[self.nonzero_at + si] & (!0u128).checked_shl(sub).unwrap_or(0);

        while si < summary {
            if s != 0 {
                let w = si * 128 + s.trailing_zeros() as usize;
                if w >= self.words {
                    return None;
                }
                let bit = w * 128 + self.storage[w].trailing_zeros() as usize;
                return (bit < self.bits).then_some(bit);
            }
            si += 1;
            if si < summary {
                s = self.storage[self.nonzero_at + si];
            }
        }
        None
    }

    pub fn find_first_clear_from(&self, start: usize) -> Option<usize> {
        if start >= self.bits {
            return None;
        }
        let sw = start / 128;
        let sb = start % 128;

        let first = !self.storage[sw] & (!0u128 << sb);
        if first != 0 {
            let bit = sw * 128 + first.trailing_zeros() as usize;
            return (bit < self.bits).then_some(bit);
        }

        let summary = self.summary_words();
        let mut si = sw / 128;
        let sub = (sw % 128) as u32 + 1;
        let mut s = !self.storage[self.full_at + si] & (!0u128).checked_shl(sub).unwrap_or(0);

        while si < summary {
            if s != 0 {
                let w = si * 128 + s.trailing_zeros() as usize;
                if w >= self.words {
                    return None;
                }
                let bit = w * 128 + (!self.storage[w]).trailing_zeros() as usize;
                return (bit < self.bits).then_some(bit);
            }
            si += 1;
            if si < summary {
                s = !self.storage[self.full_at + si];
            }
        }
        None
    }

    #[inline(always)]
    fn loc(&self, bit: usize) -> Option<(usize, u128)> {
        if bit >= self.bits {
            return None;
        }
        let word = bit / 128;
        let offset = bit % 128;
        Some((word, 1u128 << offset))
    }
}

// =============================================================================
// AtomicBitVec (SMP-safe)
// =============================================================================

/// Runtime-sized SMP-safe bitset.
///
/// Storage layout matches [`super::AtomicBitSet`]: one flat
/// `Vec<AtomicU64, &'a A>` holding `data ‖ nonzero`.  No `full`
/// summary — the racy ordering required to keep it consistent breaks
/// `find_first_clear_from`'s strict-first contract under concurrent
/// writers.  `find_first_clear_from` falls back to a linear data-word
/// scan.
///
/// Summary semantics, ordering rules, and racing-writer hints are
/// identical to [`super::AtomicBitSet`]; see its docs for the contract.
#[derive(Debug)]
pub struct AtomicBitVec<'a, A: Allocator + 'a> {
    storage: Vec<AtomicU64, &'a A>,
    bits: usize,
    words: usize,
    nonzero_at: usize,
}

impl<'a, A: Allocator + 'a> AtomicBitVec<'a, A> {
    /// Build an `AtomicBitVec` holding `bits` bits, backed by `alloc`.
    /// Every bit starts cleared.
    ///
    /// # Panics
    ///
    /// Panics if `alloc` cannot satisfy the storage allocation.
    pub fn new_in(bits: usize, alloc: &'a A) -> Self {
        let words = words_64(bits);
        let total = atomic_bitvec_storage(bits);
        debug_assert_eq!(total, words + words_64(words));

        let mut storage = Vec::with_capacity_in(total, alloc);
        // `AtomicU64` isn't `Clone`, so `resize` doesn't work — push
        // fresh zero atomics one at a time.
        for _ in 0..total {
            storage.push(AtomicU64::new(0));
        }

        Self {
            storage,
            bits,
            words,
            nonzero_at: words,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.bits
    }

    #[inline]
    fn data_words(&self) -> usize {
        self.words
    }

    #[inline]
    fn summary_words(&self) -> usize {
        self.storage.len() - self.nonzero_at
    }

    /// Wait-free.  Returns `true` if the bit was newly set.
    #[inline]
    pub fn set(&self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        // Order matters: stamp the summary first so any reader that
        // later sees the word bit set has already seen `nonzero` claim
        // it.  See `AtomicBitSet::set` for the full reasoning.
        let summary_idx = self.nonzero_at + w / 64;
        self.storage[summary_idx].fetch_or(1u64 << (w % 64), Ordering::AcqRel);
        let prev = self.storage[w].fetch_or(mask, Ordering::AcqRel);
        (prev & mask) == 0
    }

    /// Wait-free.  Returns `true` if the bit was previously set.
    #[inline]
    pub fn clear(&self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        let prev = self.storage[w].fetch_and(!mask, Ordering::AcqRel);
        let was_set = (prev & mask) != 0;
        if was_set && (prev & !mask) == 0 {
            let summary_idx = self.nonzero_at + w / 64;
            let summary_mask = 1u64 << (w % 64);
            self.storage[summary_idx].fetch_and(!summary_mask, Ordering::AcqRel);
            if self.storage[w].load(Ordering::Acquire) != 0 {
                self.storage[summary_idx].fetch_or(summary_mask, Ordering::AcqRel);
            }
        }
        was_set
    }

    /// Wait-free snapshot read of a single bit.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        let Some((w, mask)) = self.loc(bit) else {
            return false;
        };
        (self.storage[w].load(Ordering::Acquire) & mask) != 0
    }

    /// Approximate count of set bits.  Same snapshot caveats as
    /// [`super::AtomicBitSet::approx_count_ones`].
    pub fn approx_count_ones(&self) -> usize {
        self.storage[..self.data_words()]
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Bulk reset.  Not atomic across the whole bitmap — see
    /// [`super::AtomicBitSet::clear_all`] for the discipline.
    pub fn clear_all(&self) {
        for w in self.storage.iter() {
            w.store(0, Ordering::Release);
        }
    }

    pub fn find_first_set_from(&self, start: usize) -> Option<usize> {
        if start >= self.bits {
            return None;
        }
        let sw = start / 64;
        let sb = start % 64;

        let first = self.storage[sw].load(Ordering::Acquire) & (!0u64 << sb);
        if first != 0 {
            let bit = sw * 64 + first.trailing_zeros() as usize;
            return (bit < self.bits).then_some(bit);
        }

        let summary = self.summary_words();
        let mut si = sw / 64;
        let sub = (sw % 64) as u32 + 1;
        let mut s = self.storage[self.nonzero_at + si].load(Ordering::Acquire)
            & (!0u64).checked_shl(sub).unwrap_or(0);

        while si < summary {
            while s != 0 {
                let w = si * 64 + s.trailing_zeros() as usize;
                s &= s - 1;
                if w >= self.words {
                    return None;
                }
                let word = self.storage[w].load(Ordering::Acquire);
                if word != 0 {
                    let bit = w * 64 + word.trailing_zeros() as usize;
                    return (bit < self.bits).then_some(bit);
                }
            }
            si += 1;
            if si < summary {
                s = self.storage[self.nonzero_at + si].load(Ordering::Acquire);
            }
        }
        None
    }

    pub fn find_first_clear_from(&self, start: usize) -> Option<usize> {
        if start >= self.bits {
            return None;
        }
        let sw = start / 64;
        let sb = start % 64;

        let first = !self.storage[sw].load(Ordering::Acquire) & (!0u64 << sb);
        if first != 0 {
            let bit = sw * 64 + first.trailing_zeros() as usize;
            return (bit < self.bits).then_some(bit);
        }

        for w in (sw + 1)..self.words {
            let word = self.storage[w].load(Ordering::Acquire);
            if word != u64::MAX {
                let bit = w * 64 + (!word).trailing_zeros() as usize;
                return (bit < self.bits).then_some(bit);
            }
        }
        None
    }

    /// Atomically claim a contiguous run of `length` bits starting at
    /// `start`.  See [`super::AtomicBitSet::try_claim_run`] for the
    /// contract.
    pub fn try_claim_run(&self, start: usize, length: usize) -> Result<(), ()> {
        if length == 0 {
            return Ok(());
        }
        let end = start.checked_add(length).ok_or(())?;
        if end > self.bits {
            return Err(());
        }

        let first_word = start / 64;
        let last_word = (end - 1) / 64;

        let mut cur = start;
        let mut cleared_through: Option<usize> = None;

        for w in first_word..=last_word {
            let word_lo = w * 64;
            let word_hi = word_lo + 64;
            let lo_bit = cur.max(word_lo) - word_lo;
            let hi_bit = end.min(word_hi) - word_lo;
            let claim_mask = if hi_bit == 64 {
                !0u64 << lo_bit
            } else {
                ((1u64 << hi_bit) - 1) & (!0u64 << lo_bit)
            };

            let prev = self.storage[w].fetch_and(!claim_mask, Ordering::AcqRel);
            if (prev & claim_mask) != claim_mask {
                let acquired_here = prev & claim_mask;
                if acquired_here != 0 {
                    self.storage[w].fetch_or(acquired_here, Ordering::AcqRel);
                }
                if let Some(through) = cleared_through {
                    self.rollback_run(start, through);
                }
                return Err(());
            }

            if (prev & !claim_mask) == 0 {
                let summary_idx = self.nonzero_at + w / 64;
                let summary_mask = 1u64 << (w % 64);
                self.storage[summary_idx].fetch_and(!summary_mask, Ordering::AcqRel);
                if self.storage[w].load(Ordering::Acquire) != 0 {
                    self.storage[summary_idx].fetch_or(summary_mask, Ordering::AcqRel);
                }
            }

            cleared_through = Some(end.min(word_hi));
            cur = word_hi;
        }
        Ok(())
    }

    /// Restore a previously-cleared run.
    pub fn release_run(&self, start: usize, length: usize) {
        if length == 0 {
            return;
        }
        let end = start.saturating_add(length).min(self.bits);
        self.rollback_run(start, end);
    }

    fn rollback_run(&self, start: usize, end: usize) {
        let first_word = start / 64;
        let last_word = (end - 1) / 64;
        let mut cur = start;
        for w in first_word..=last_word {
            let word_lo = w * 64;
            let word_hi = word_lo + 64;
            let lo_bit = cur.max(word_lo) - word_lo;
            let hi_bit = end.min(word_hi) - word_lo;
            let restore_mask = if hi_bit == 64 {
                !0u64 << lo_bit
            } else {
                ((1u64 << hi_bit) - 1) & (!0u64 << lo_bit)
            };
            self.storage[self.nonzero_at + w / 64].fetch_or(1u64 << (w % 64), Ordering::AcqRel);
            self.storage[w].fetch_or(restore_mask, Ordering::AcqRel);
            cur = word_hi;
        }
    }

    #[inline(always)]
    fn loc(&self, bit: usize) -> Option<(usize, u64)> {
        if bit >= self.bits {
            return None;
        }
        let word = bit / 64;
        let offset = bit % 64;
        Some((word, 1u64 << offset))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;

    // 1 MiB test arena — generous for tests that build a few bitvecs.
    static TEST_ARENA: TestArena<{ 1 << 20 }> = TestArena::new();

    // ---- BitVec ----

    #[test]
    fn bitvec_basic_set_clear() {
        let mut bv = BitVec::new_in(256, &TEST_ARENA);
        assert!(bv.set(3));
        assert!(bv.set(131));
        assert_eq!(bv.count_ones(), 2);
        assert!(bv.contains(3));
        assert_eq!(bv.find_first_set_from(0), Some(3));
        assert_eq!(bv.find_first_set_from(4), Some(131));
        assert_eq!(bv.find_first_clear_from(0), Some(0));
        assert_eq!(bv.pop_first_set(), Some(3));
        assert!(bv.clear(131));
        assert_eq!(bv.count_ones(), 0);
    }

    #[test]
    fn bitvec_odd_size_only_advertises_logical_bits() {
        // 1000 bits, not a 128-aligned multiple — the trailing
        // 24 storage bits are inaccessible.
        let mut bv = BitVec::new_in(1000, &TEST_ARENA);
        assert_eq!(bv.capacity(), 1000);
        assert!(!bv.set(1000));
        assert!(!bv.contains(1000));
        assert!(bv.set(999));
        assert_eq!(bv.find_first_set_from(0), Some(999));
    }

    #[test]
    fn bitvec_find_first_clear_crosses_summary_words() {
        // Same shape as the `BitSet` test, runtime sized.
        let mut bv = BitVec::new_in(8320, &TEST_ARENA);
        for bit in 0..8193 {
            assert!(bv.set(bit));
        }
        assert_eq!(bv.find_first_clear_from(0), Some(8193));
        assert_eq!(bv.find_first_clear_from(8192), Some(8193));
        for bit in 8193..bv.capacity() {
            assert!(bv.set(bit));
        }
        assert_eq!(bv.find_first_clear_from(0), None);
    }

    // ---- AtomicBitVec ----

    #[test]
    fn atomic_bitvec_basic_set_clear() {
        let bv = AtomicBitVec::new_in(256, &TEST_ARENA);
        assert!(bv.set(3));
        assert!(!bv.set(3)); // already set
        assert!(bv.set(131));
        assert!(bv.contains(3));
        assert!(bv.contains(131));
        assert!(!bv.contains(7));
        assert_eq!(bv.find_first_set_from(0), Some(3));
        assert_eq!(bv.find_first_set_from(4), Some(131));
        assert!(bv.clear(3));
        assert!(!bv.clear(3));
        assert_eq!(bv.find_first_set_from(0), Some(131));
    }

    #[test]
    fn atomic_bitvec_claim_run() {
        let bv = AtomicBitVec::new_in(256, &TEST_ARENA);
        for bit in 60..70 {
            bv.set(bit);
        }
        assert!(bv.try_claim_run(62, 6).is_ok());
        for bit in 62..68 {
            assert!(!bv.contains(bit));
        }
        assert!(bv.contains(60));
        assert!(bv.contains(68));
    }

    #[test]
    fn atomic_bitvec_claim_fails_and_rolls_back() {
        let bv = AtomicBitVec::new_in(128, &TEST_ARENA);
        for bit in (0..3).chain(5..8) {
            bv.set(bit);
        }
        assert!(bv.try_claim_run(0, 8).is_err());
        for bit in (0..3).chain(5..8) {
            assert!(
                bv.contains(bit),
                "bit {bit} should still be set after rollback"
            );
        }
        assert!(!bv.contains(4));
    }

    #[test]
    fn atomic_bitvec_release_run() {
        let bv = AtomicBitVec::new_in(256, &TEST_ARENA);
        bv.release_run(70, 10);
        for bit in 70..80 {
            assert!(bv.contains(bit));
        }
        assert!(!bv.contains(69));
        assert!(!bv.contains(80));
    }

    #[test]
    fn atomic_bitvec_clear_all() {
        let bv = AtomicBitVec::new_in(256, &TEST_ARENA);
        bv.set(3);
        bv.set(131);
        bv.clear_all();
        assert_eq!(bv.find_first_set_from(0), None);
        assert_eq!(bv.approx_count_ones(), 0);
    }
}
