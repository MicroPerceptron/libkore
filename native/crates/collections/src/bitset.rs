//! Fixed-capacity bitsets, parameterised by `WORDS` = storage word count.
//!
//! Two siblings:
//!
//! - [`BitSet<WORDS>`] — single-threaded, `u128` words, strict point-in-time
//!   semantics, `&mut self` mutation.  Bit capacity is `WORDS * 128`.
//! - [`AtomicBitSet<WORDS>`] — SMP-safe, `AtomicU64` words, `&self` methods,
//!   wait-free single-bit ops, lock-free multi-bit claims with
//!   bounded-retry rollback.  Bit capacity is `WORDS * 64`.
//!
//! # Storage width asymmetry
//!
//! `BitSet` uses `u128` to halve scan iteration count.  `AtomicBitSet`
//! uses `AtomicU64` because `AtomicU128` is still nightly-only and
//! lock-freeness isn't guaranteed on every supported target.  The
//! cognitive cost of the asymmetry is small — they're different types
//! with different semantics anyway — and the per-op cost difference is
//! dwarfed by cross-CPU coordination overhead.
//!
//! # `WORDS` is storage word count, not bit count
//!
//! `BitSet<2>` holds `2 * 128 = 256` bits.  `AtomicBitSet<4>` holds
//! `4 * 64 = 256` bits.  This parameterisation lets the storage field
//! be a plain `[T; WORDS]` array — no const expression in any type
//! position, so rust-analyzer evaluates the type without invoking its
//! incomplete `generic_const_exprs` machinery.
//!
//! # No summary tiers
//!
//! Earlier revisions kept a `nonzero` (and a `full`, for `BitSet`)
//! summary tier packed into the same flat array, to accelerate
//! `find_first_*`.  The tiers were dropped along with the `generic_const_exprs`
//! patterns they required; `find_first_*` now scans every word linearly.
//! For the kernel's current consumers (CPU masks, interrupt vector pools,
//! at most a few thousand bits) the scan is bounded and dominated by
//! cache cost, not by inner-loop work.

use core::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// BitSet (single-threaded, u128 words)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BitSet<const WORDS: usize> {
    storage: [u128; WORDS],
}

impl<const WORDS: usize> BitSet<WORDS> {
    /// Logical bit count = `WORDS * 128`.
    pub const BITS: usize = WORDS * 128;

    pub const fn new() -> Self {
        Self {
            storage: [0; WORDS],
        }
    }

    /// Logical bit count (= `Self::BITS`).
    #[inline]
    pub const fn capacity(&self) -> usize {
        Self::BITS
    }

    pub fn clear_all(&mut self) {
        self.storage.fill(0);
    }

    #[inline(always)]
    pub const fn contains(&self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        (self.storage[w] & mask) != 0
    }

    /// Builder form of [`set`](Self::set) for const tables: the set with
    /// `bit` added.  Out-of-range bits are ignored (the `loc` contract).
    #[must_use]
    #[inline(always)]
    pub const fn with(mut self, bit: usize) -> Self {
        let _ = self.set(bit);
        self
    }

    /// Builder form for a consecutive run: the set with
    /// `base..base + len` added.
    #[must_use]
    pub const fn with_run(mut self, base: usize, len: usize) -> Self {
        let mut i = 0;
        while i < len {
            let _ = self.set(base + i);
            i += 1;
        }
        self
    }

    /// The union of two sets — const, for building lifetime-windowed
    /// masks out of per-phase const tables.
    #[must_use]
    pub const fn union(mut self, other: &Self) -> Self {
        let mut w = 0;
        while w < WORDS {
            self.storage[w] |= other.storage[w];
            w += 1;
        }
        self
    }

    #[inline(always)]
    pub const fn set(&mut self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        let before = self.storage[w];
        self.storage[w] = before | mask;
        (before & mask) == 0
    }

    #[inline(always)]
    pub fn clear(&mut self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        let before = self.storage[w];
        self.storage[w] = before & !mask;
        (before & mask) != 0
    }

    pub fn count_ones(&self) -> usize {
        self.storage
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Iterate the indices of all set bits in ascending order.  Does
    /// not mutate the set; for the consume-while-popping idiom use
    /// [`pop_first_set`](Self::pop_first_set) instead.
    pub fn iter(&self) -> Iter<'_, WORDS> {
        Iter { set: self, next: 0 }
    }

    pub fn pop_first_set(&mut self) -> Option<usize> {
        for w in 0..WORDS {
            let word = self.storage[w];
            if word == 0 {
                continue;
            }
            let bit = word.trailing_zeros() as usize;
            self.storage[w] = word & (word - 1);
            return Some(w * 128 + bit);
        }
        None
    }

    pub fn find_first_set_from(&self, start: usize) -> Option<usize> {
        if start >= Self::BITS {
            return None;
        }
        let sw = start / 128;
        let sb = start % 128;

        // The starting word: mask off bits below `start`.
        let first = self.storage[sw] & (!0u128 << sb);
        if first != 0 {
            return Some(sw * 128 + first.trailing_zeros() as usize);
        }

        // Walk subsequent words linearly.
        for w in (sw + 1)..WORDS {
            let word = self.storage[w];
            if word != 0 {
                return Some(w * 128 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    pub fn find_first_clear_from(&self, start: usize) -> Option<usize> {
        if start >= Self::BITS {
            return None;
        }
        let sw = start / 128;
        let sb = start % 128;

        let first = !self.storage[sw] & (!0u128 << sb);
        if first != 0 {
            return Some(sw * 128 + first.trailing_zeros() as usize);
        }

        for w in (sw + 1)..WORDS {
            let word = self.storage[w];
            if word != u128::MAX {
                return Some(w * 128 + (!word).trailing_zeros() as usize);
            }
        }
        None
    }

    #[inline(always)]
    const fn loc(bit: usize) -> Option<(usize, u128)> {
        if bit >= Self::BITS {
            return None;
        }
        let word = bit / 128;
        let offset = bit % 128;
        Some((word, 1u128 << offset))
    }
}

impl<const WORDS: usize> Default for BitSet<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable iterator over the set indices of a [`BitSet`], in
/// ascending order.
pub struct Iter<'a, const WORDS: usize> {
    set: &'a BitSet<WORDS>,
    next: usize,
}

impl<'a, const WORDS: usize> Iterator for Iter<'a, WORDS> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let bit = self.set.find_first_set_from(self.next)?;
        self.next = bit + 1;
        Some(bit)
    }
}

// =============================================================================
// AtomicBitSet (SMP-safe, u64 words)
// =============================================================================

/// SMP-safe fixed-capacity bitset over `WORDS * 64` bits.
///
/// Every method takes `&self`; primitive bit ops are wait-free, multi-bit
/// allocation claims compose into bounded-retry lock-free sequences via
/// [`Self::try_claim_run`].  `find_first_*` walks every word linearly —
/// no summary tier is maintained, since the static budget for the
/// kernel's actual bitset sizes keeps the scan bounded.
#[derive(Debug)]
pub struct AtomicBitSet<const WORDS: usize> {
    storage: [AtomicU64; WORDS],
}

impl<const WORDS: usize> AtomicBitSet<WORDS> {
    /// Logical bit count = `WORDS * 64`.
    pub const BITS: usize = WORDS * 64;

    pub const fn new() -> Self {
        Self {
            storage: [const { AtomicU64::new(0) }; WORDS],
        }
    }

    /// A set whose first `bits` bits start **set**, the rest clear —
    /// `const`, so an all-free-at-birth claim ledger (a strip plan, a
    /// frame map) can be a `static` without an init pass.  Under the
    /// [`try_claim_run`](Self::try_claim_run) discipline (set = free,
    /// clear = claimed) the tail beyond `bits` is born claimed and, with
    /// no releaser holding it, stays so forever — which is exactly how a
    /// capacity that is not a word multiple stays unclaimable.
    ///
    /// # Panics
    /// `bits > Self::BITS` (compile-time when const-evaluated).
    pub const fn with_first_set(bits: usize) -> Self {
        assert!(bits <= Self::BITS, "more initial bits than capacity");
        let mut storage = [const { AtomicU64::new(0) }; WORDS];
        let mut w = 0;
        while w * 64 < bits {
            let in_word = bits - w * 64;
            let word = if in_word >= 64 {
                !0u64
            } else {
                (1u64 << in_word) - 1
            };
            // Plain stores would race nothing here — `self` is not yet
            // shared — but `AtomicU64::new` keeps this fully const.
            storage[w] = AtomicU64::new(word);
            w += 1;
        }
        Self { storage }
    }

    /// Logical bit count (= `Self::BITS`).
    #[inline]
    pub const fn capacity(&self) -> usize {
        Self::BITS
    }

    /// Wait-free.  Returns `true` if the bit was newly set.
    #[inline]
    pub fn set(&self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        let prev = self.storage[w].fetch_or(mask, Ordering::AcqRel);
        (prev & mask) == 0
    }

    /// Wait-free.  Returns `true` if the bit was previously set.
    #[inline]
    pub fn clear(&self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        let prev = self.storage[w].fetch_and(!mask, Ordering::AcqRel);
        (prev & mask) != 0
    }

    /// Wait-free.  Snapshot read of a single bit.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        let Some((w, mask)) = Self::loc(bit) else {
            return false;
        };
        (self.storage[w].load(Ordering::Acquire) & mask) != 0
    }

    /// Approximate count of set bits.  Sums per-word `count_ones` from
    /// independent loads — the result reflects no consistent global
    /// snapshot under concurrent writers.
    pub fn approx_count_ones(&self) -> usize {
        self.storage
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Find the first set bit at or after `start`.  Snapshot semantics —
    /// a returned bit was set at *some* point during the scan; concurrent
    /// clears may invalidate it before the caller can act.
    pub fn find_first_set_from(&self, start: usize) -> Option<usize> {
        if start >= Self::BITS {
            return None;
        }
        let sw = start / 64;
        let sb = start % 64;

        let first = self.storage[sw].load(Ordering::Acquire) & (!0u64 << sb);
        if first != 0 {
            return Some(sw * 64 + first.trailing_zeros() as usize);
        }

        for w in (sw + 1)..WORDS {
            let word = self.storage[w].load(Ordering::Acquire);
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Find the first clear bit at or after `start`.  Walks every word.
    /// Snapshot semantics, same as [`Self::find_first_set_from`].
    pub fn find_first_clear_from(&self, start: usize) -> Option<usize> {
        if start >= Self::BITS {
            return None;
        }
        let sw = start / 64;
        let sb = start % 64;

        let first = !self.storage[sw].load(Ordering::Acquire) & (!0u64 << sb);
        if first != 0 {
            return Some(sw * 64 + first.trailing_zeros() as usize);
        }

        for w in (sw + 1)..WORDS {
            let word = self.storage[w].load(Ordering::Acquire);
            if word != u64::MAX {
                return Some(w * 64 + (!word).trailing_zeros() as usize);
            }
        }
        None
    }

    /// Bulk reset every bit to 0.  **Not atomic across the whole bitmap** —
    /// individual word writes are atomic, but a concurrent reader can
    /// observe a partially-cleared state.  Callers that need a strict
    /// reset must serialize this with all other accessors (e.g., bump an
    /// epoch counter to invalidate observations).
    pub fn clear_all(&self) {
        for w in self.storage.iter() {
            w.store(0, Ordering::Release);
        }
    }

    /// Atomically claim a contiguous run of `length` bits starting at
    /// `start`.  All bits in `[start, start + length)` must currently
    /// be set; on success, every bit in the range is cleared.
    ///
    /// Returns:
    /// - `Ok(())` if every bit was set and we cleared them.
    /// - `Err(())` if at least one bit was already clear (lost the race
    ///   to another claimer); any bits we did claim are restored
    ///   (rolled back) before returning.
    ///
    /// Each attempt is wait-free; the surrounding scan + retry loop is
    /// lock-free.
    pub fn try_claim_run(&self, start: usize, length: usize) -> Result<(), ()> {
        if length == 0 {
            return Ok(());
        }
        let end = start.checked_add(length).ok_or(())?;
        if end > Self::BITS {
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
                // Lost the race — restore only the bits we actually
                // cleared, then unwind earlier whole words.
                let acquired_here = prev & claim_mask;
                if acquired_here != 0 {
                    self.storage[w].fetch_or(acquired_here, Ordering::AcqRel);
                }
                if let Some(through) = cleared_through {
                    self.rollback_run(start, through);
                }
                return Err(());
            }

            cleared_through = Some(end.min(word_hi));
            cur = word_hi;
        }
        Ok(())
    }

    /// Restore a previously-cleared run.  Used by [`Self::try_claim_run`]
    /// for partial-failure rollback, and externally by frame-allocator
    /// `release` paths.
    pub fn release_run(&self, start: usize, length: usize) {
        if length == 0 {
            return;
        }
        let end = start.saturating_add(length).min(Self::BITS);
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
            self.storage[w].fetch_or(restore_mask, Ordering::AcqRel);
            cur = word_hi;
        }
    }

    #[inline(always)]
    const fn loc(bit: usize) -> Option<(usize, u64)> {
        if bit >= Self::BITS {
            return None;
        }
        let word = bit / 64;
        let offset = bit % 64;
        Some((word, 1u64 << offset))
    }
}

impl<const WORDS: usize> Default for AtomicBitSet<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BitSet ----

    #[test]
    fn bitset_set_clear_and_scan() {
        // 2 u128 words = 256 bits.
        let mut set = BitSet::<2>::new();
        assert!(set.set(3));
        assert!(set.set(131));
        assert_eq!(set.count_ones(), 2);
        assert!(set.contains(3));
        assert_eq!(set.find_first_set_from(0), Some(3));
        assert_eq!(set.find_first_set_from(4), Some(131));
        assert_eq!(set.find_first_clear_from(0), Some(0));
        assert_eq!(set.pop_first_set(), Some(3));
        assert!(set.clear(131));
        assert_eq!(set.count_ones(), 0);
    }

    #[test]
    fn bitset_find_first_clear_crosses_word_boundaries() {
        // 65 u128 words = 8320 bits — covers many word boundaries.
        let mut set = BitSet::<65>::new();
        for bit in 0..8193 {
            assert!(set.set(bit));
        }
        assert_eq!(set.find_first_clear_from(0), Some(8193));
        assert_eq!(set.find_first_clear_from(8192), Some(8193));

        for bit in 8193..set.capacity() {
            assert!(set.set(bit));
        }
        assert_eq!(set.find_first_clear_from(0), None);
    }

    #[test]
    fn bitset_rejects_out_of_range() {
        // 1 u128 word = 128 bits.
        let mut set = BitSet::<1>::new();
        assert!(!set.set(128));
        assert!(!set.contains(128));
        assert!(!set.clear(128));
        assert_eq!(set.find_first_set_from(128), None);
    }

    // ---- AtomicBitSet ----

    #[test]
    fn atomic_with_first_set_is_a_ready_claim_ledger() {
        // 100 free bits over two words: word 0 full, word 1 a 36-bit tail.
        static S: AtomicBitSet<2> = AtomicBitSet::with_first_set(100);
        assert!(S.contains(0) && S.contains(99));
        assert!(!S.contains(100), "the tail is born claimed");
        // The whole free span claims as one run; the tail never does.
        assert!(S.try_claim_run(0, 100).is_ok());
        assert!(S.try_claim_run(100, 1).is_err());
        S.release_run(0, 100);
        assert!(S.contains(0) && S.contains(99) && !S.contains(100));
    }

    #[test]
    fn atomic_bitset_basic_set_clear() {
        // 4 u64 words = 256 bits.
        let set = AtomicBitSet::<4>::new();
        assert!(set.set(3));
        assert!(!set.set(3)); // already set
        assert!(set.set(131));
        assert!(set.contains(3));
        assert!(set.contains(131));
        assert!(!set.contains(7));
        assert_eq!(set.find_first_set_from(0), Some(3));
        assert_eq!(set.find_first_set_from(4), Some(131));
        assert!(set.clear(3));
        assert!(!set.clear(3));
        assert_eq!(set.find_first_set_from(0), Some(131));
    }

    #[test]
    fn atomic_bitset_clear_all_resets_state() {
        let set = AtomicBitSet::<4>::new();
        set.set(3);
        set.set(131);
        set.clear_all();
        assert_eq!(set.find_first_set_from(0), None);
        assert_eq!(set.approx_count_ones(), 0);
    }

    #[test]
    fn atomic_bitset_claim_within_word() {
        let set = AtomicBitSet::<2>::new(); // 128 bits
        for bit in 0..32 {
            set.set(bit);
        }
        assert!(set.try_claim_run(8, 4).is_ok());
        assert!(!set.contains(8));
        assert!(!set.contains(11));
        assert!(set.contains(12));
        assert!(set.contains(7));
    }

    #[test]
    fn atomic_bitset_claim_across_words() {
        let set = AtomicBitSet::<4>::new(); // 256 bits
        for bit in 60..70 {
            set.set(bit);
        }
        assert!(set.try_claim_run(62, 6).is_ok());
        for bit in 62..68 {
            assert!(!set.contains(bit));
        }
        assert!(set.contains(60));
        assert!(set.contains(68));
    }

    #[test]
    fn atomic_bitset_claim_fails_and_rolls_back() {
        let set = AtomicBitSet::<2>::new(); // 128 bits
        for bit in (0..3).chain(5..8) {
            set.set(bit);
        }
        assert!(set.try_claim_run(0, 8).is_err());
        for bit in (0..3).chain(5..8) {
            assert!(
                set.contains(bit),
                "bit {bit} should still be set after rollback"
            );
        }
        assert!(!set.contains(4));
    }

    #[test]
    fn atomic_bitset_release_run() {
        let set = AtomicBitSet::<4>::new(); // 256 bits
        set.release_run(70, 10);
        for bit in 70..80 {
            assert!(set.contains(bit));
        }
        assert!(!set.contains(69));
        assert!(!set.contains(80));
    }

    #[test]
    fn atomic_bitset_rejects_out_of_range() {
        let set = AtomicBitSet::<1>::new(); // 64 bits
        assert!(!set.set(64));
        assert!(!set.contains(64));
        assert_eq!(set.find_first_set_from(64), None);
        assert!(set.try_claim_run(60, 10).is_err());
    }

    #[test]
    fn atomic_bitset_constructs_at_16_bit_id_width() {
        // Sanity: a 65 536-bit pool (1024 u64 words) constructs as const
        // cleanly even at the larger sizes.
        const C: AtomicBitSet<1024> = AtomicBitSet::new();
        let _ = C; // Touch it so the const isn't optimised away.
    }
}
