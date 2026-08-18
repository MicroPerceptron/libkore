//! Generational IDs and their slot-array allocator.
//!
//! [`GenId<MAX>`] packs `(id, gen)` into a `u32` — id in the low
//! `id_bits(MAX)` bits, generation in the rest.  [`GenIdAllocator<MAX, MIN>`]
//! hands IDs out of an `[AtomicU8; MAX]` slot array and pairs each one
//! with the allocator's current generation, so post-rollover staleness is
//! detectable with a single equality check on [`GenIdAllocator::is_current`].
//!
//! # Parameter shape
//!
//! `MAX` is the bitmap capacity — the (exclusive) ceiling on issuable IDs.
//! Pick it to match the architectural identifier's value space: `256` for
//! 8-bit ASIDs, `4096` for x86 PCID, `65536` for 16-bit ASIDs, `1 << 20`
//! for IOMMU PASIDs.  All derived quantities (encoding bit-width, id mask,
//! generation mask) are computed by reverse-math on `MAX` inside associated
//! consts — `id_bits = MAX.next_power_of_two().trailing_zeros()`.
//!
//! `MIN` is the issuance floor (default `0`).  Set it to `1` to reserve
//! slot 0 as a kernel-context / "no tagging" identity (the convention for
//! ASIDs and PASIDs on every arch this crate targets).  `MIN` is part of
//! the type, so the floor is constant across the allocator's lifetime and
//! preserved across [`GenIdAllocator::rollover`].
//!
//! # Storage choice
//!
//! The slot array is `[AtomicU8; MAX]` — one byte per slot, no bit-packing.
//! That trades 64× more static memory than a packed `AtomicBitSet` for two
//! things: rust-analyzer compatibility (no `generic_const_exprs` patterns
//! in any type position — `MAX` is a plain const-generic substituted
//! directly into the array length), and a simpler single-CAS allocation
//! path with no summary-bitmap coherency to maintain.  For the kernel's
//! typical usage — one allocator per arch, sized to the hardware ASID/PCID/
//! PASID width — the memory cost is bounded by the static budget for those
//! widths.
//!
//! # Sentinels
//!
//! [`GenId::INVALID`] is `(id = 0, gen = 0)`.  The allocator starts at
//! `gen == 1` and skips `0` on rollover, so any allocator-issued ID has
//! `gen > 0` and is always distinguishable from `INVALID`.

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

/// Bits the id field occupies for capacity `max`: `ceil(log2(max))`.
/// `id_bits(0) = 0`, `id_bits(1) = 0`, `id_bits(4) = 2`, `id_bits(65536) = 16`.
/// Saturates at `32` for capacities beyond `u32`'s range so the packing
/// math stays well-defined.
#[inline]
const fn id_bits(max: usize) -> u32 {
    if max <= 1 {
        return 0;
    }
    let bits = max.next_power_of_two().trailing_zeros();
    if bits > 32 { 32 } else { bits }
}

/// `(id, gen)` packed into a `u32`.  Low `id_bits(MAX)` bits hold the id;
/// the remaining high bits hold the generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GenId<const MAX: usize>(u32);

impl<const MAX: usize> GenId<MAX> {
    /// Bit-width of the id field — derived from `MAX` by reverse-math.
    pub const ID_BITS: u32 = id_bits(MAX);

    /// Mask covering the id field.  Edges (`ID_BITS == 0` or `== 32`)
    /// dodge the `1u32 << 32` UB.
    const ID_MASK: u32 = if Self::ID_BITS >= 32 {
        u32::MAX
    } else if Self::ID_BITS == 0 {
        0
    } else {
        (1u32 << Self::ID_BITS) - 1
    };

    /// "Not a real ID" sentinel.  `(id = 0, gen = 0)` — distinguishable
    /// from any allocator-issued ID since generations start at `1`.
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn new(id: u32, generation: u32) -> Self {
        let gen_bits = if Self::ID_BITS >= 32 {
            0
        } else {
            generation << Self::ID_BITS
        };
        Self(gen_bits | (id & Self::ID_MASK))
    }

    #[inline]
    pub const fn id(&self) -> u32 {
        self.0 & Self::ID_MASK
    }

    #[inline]
    pub const fn generation(&self) -> u32 {
        if Self::ID_BITS >= 32 {
            0
        } else {
            self.0 >> Self::ID_BITS
        }
    }

    /// `true` if this isn't [`Self::INVALID`].  Allocator-issued IDs all
    /// satisfy this — they carry `gen >= 1`.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.0 != 0
    }

    /// Pack into a `u32` — low `ID_BITS` bits id, high bits gen.  Useful
    /// for atomic load/store of an ID payload.
    #[inline]
    pub const fn pack(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn unpack(value: u32) -> Self {
        Self(value)
    }
}

impl<const MAX: usize> Default for GenId<MAX> {
    #[inline]
    fn default() -> Self {
        Self::INVALID
    }
}

/// Slot-array allocator that pairs every issued ID with the current
/// generation.
///
/// `MAX` is the bitmap capacity; the slot array is sized to exactly `MAX`
/// bytes.  `MIN` is the issuance floor — IDs below `MIN` are never
/// returned (use `MIN = 1` to reserve slot 0).  Both are type-level so the
/// allocator's shape is fully known at compile time; runtime tuning is via
/// [`Self::set_effective_cap`].
///
/// Concurrency: every method takes `&self`.  Allocation is lock-free under
/// peers on the same generation; rollover bumps the generation and clears
/// the slot array, after which previously-issued IDs are no longer current.
pub struct GenIdAllocator<const MAX: usize, const MIN: usize = 0> {
    used: [AtomicU8; MAX],
    current_gen: AtomicU32,
    /// Runtime ceiling on issuance.  Defaults to `MAX`; boot code lowers
    /// it to the hardware width if narrower.  `Relaxed` is fine —
    /// `set_effective_cap` is expected once at boot before any `try_alloc`.
    effective_cap: AtomicUsize,
}

impl<const MAX: usize, const MIN: usize> GenIdAllocator<MAX, MIN> {
    /// Bit-width of the id field of [`GenId<MAX>`].
    pub const ID_BITS: u32 = id_bits(MAX);

    /// Wrap mask for the generation field on rollover.  `0` when
    /// `ID_BITS == 32` (no gen bits — rollover is a no-op semantically,
    /// but kept legal).
    const GEN_MASK: u32 = if Self::ID_BITS >= 32 {
        0
    } else {
        u32::MAX >> Self::ID_BITS
    };

    pub const fn new() -> Self {
        Self {
            used: [const { AtomicU8::new(0) }; MAX],
            current_gen: AtomicU32::new(1),
            effective_cap: AtomicUsize::new(MAX),
        }
    }

    /// Compile-time bitmap capacity (= `MAX`, the exclusive id ceiling).
    #[inline]
    pub const fn capacity(&self) -> usize {
        MAX
    }

    /// Compile-time issuance floor (= `MIN`).
    #[inline]
    pub const fn min_id(&self) -> usize {
        MIN
    }

    /// Current runtime issuance ceiling.
    #[inline]
    pub fn effective_cap(&self) -> usize {
        self.effective_cap.load(Ordering::Relaxed)
    }

    /// Set the runtime issuance ceiling.  **Must be called before any
    /// [`try_alloc`](Self::try_alloc) on a given generation** — IDs
    /// already issued above the new ceiling are not retroactively
    /// invalidated.
    ///
    /// `cap` is clamped to `[0, MAX]`.  A `cap` at or below `MIN` makes
    /// every subsequent [`try_alloc`] return `None`.
    pub fn set_effective_cap(&self, cap: usize) {
        self.effective_cap.store(cap.min(MAX), Ordering::Relaxed);
    }

    #[inline]
    pub fn current_generation(&self) -> u32 {
        self.current_gen.load(Ordering::Acquire)
    }

    /// `true` iff `id` was issued in the current generation and still
    /// represents a live allocation (not stale across rollover, not below
    /// `MIN`).
    #[inline]
    pub fn is_current(&self, id: GenId<MAX>) -> bool {
        id.is_valid() && id.generation() == self.current_generation() && (id.id() as usize) >= MIN
    }

    /// Try to allocate a fresh ID.  Returns `None` if every slot in
    /// `[MIN, effective_cap)` is taken.
    ///
    /// Lock-free under concurrent peers on the same generation: each
    /// candidate slot is claimed via `compare_exchange` from free to a
    /// nonzero generation marker, and the scan advances past contested
    /// slots monotonically.
    pub fn try_alloc(&self) -> Option<GenId<MAX>> {
        let cap = self.effective_cap();
        loop {
            let generation = self.current_generation();
            let marker = Self::slot_marker(generation);
            let mut idx = MIN;
            while idx < cap {
                if self.used[idx]
                    .compare_exchange(0, marker, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    if self.current_generation() == generation {
                        return Some(GenId::new(idx as u32, generation));
                    }
                    let _ = self.used[idx].compare_exchange(
                        marker,
                        0,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    break;
                }
                idx += 1;
            }
            if idx == cap && self.current_generation() == generation {
                return None;
            }
        }
    }

    #[inline]
    const fn slot_marker(generation: u32) -> u8 {
        ((generation % u8::MAX as u32) + 1) as u8
    }

    /// Bump the generation and clear every slot.  **Caller must invalidate
    /// every downstream state that could reference a previously-issued ID
    /// before any post-rollover ID is observed** — typically a TLB flush
    /// for ASIDs, a context-cache invalidation for IOMMU domains.
    ///
    /// Ordering: generation is bumped first, then `used` is cleared.
    pub fn rollover(&self) {
        let prev = self.current_gen.load(Ordering::Acquire);
        let next = if Self::GEN_MASK == 0 {
            // No generation bits → rollover has nothing to bump.
            0
        } else {
            let candidate = prev.wrapping_add(1) & Self::GEN_MASK;
            // Skip 0 — it's the INVALID sentinel.
            if candidate == 0 { 1 } else { candidate }
        };
        self.current_gen.store(next, Ordering::Release);
        let mut idx = 0;
        while idx < MAX {
            self.used[idx].store(0, Ordering::Release);
            idx += 1;
        }
    }

    /// Allocate, rolling over via `rollover_fn` on exhaustion.  Idiomatic
    /// install-time call site.
    pub fn alloc_or_rollover(&self, rollover_fn: impl FnOnce()) -> GenId<MAX> {
        if let Some(id) = self.try_alloc() {
            return id;
        }
        rollover_fn();
        self.rollover();
        self.try_alloc()
            .expect("post-rollover allocation cannot fail with effective_cap > MIN")
    }

    /// Return `id`'s slot to the pool, but only if it's still current.
    /// Stale frees from before a rollover are no-ops — the rollover already
    /// reclaimed the slot.
    pub fn free(&self, id: GenId<MAX>) {
        if self.is_current(id) {
            self.used[id.id() as usize].store(0, Ordering::Release);
        }
    }
}

impl<const MAX: usize, const MIN: usize> Default for GenIdAllocator<MAX, MIN> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_sentinel_is_not_current() {
        let alloc = GenIdAllocator::<256, 1>::new();
        assert!(!alloc.is_current(GenId::<256>::INVALID));
        assert!(!GenId::<256>::INVALID.is_valid());
    }

    #[test]
    fn alloc_respects_min_floor() {
        let alloc = GenIdAllocator::<256, 1>::new();
        let id = alloc.try_alloc().unwrap();
        assert_eq!(id.id(), 1);
        assert_eq!(id.generation(), 1);
    }

    #[test]
    fn alloc_from_zero_when_min_is_zero() {
        let alloc = GenIdAllocator::<256>::new();
        let id = alloc.try_alloc().unwrap();
        assert_eq!(id.id(), 0);
    }

    #[test]
    fn free_returns_slot_to_pool() {
        let alloc = GenIdAllocator::<8, 1>::new();
        let a = alloc.try_alloc().unwrap();
        let b = alloc.try_alloc().unwrap();
        assert_eq!(a.id(), 1);
        assert_eq!(b.id(), 2);
        alloc.free(a);
        let c = alloc.try_alloc().unwrap();
        assert_eq!(c.id(), 1);
    }

    #[test]
    fn exhausted_alloc_returns_none() {
        // MAX = 4 slots; MIN = 1 → 3 issuable IDs.
        let alloc = GenIdAllocator::<4, 1>::new();
        let _a = alloc.try_alloc().unwrap();
        let _b = alloc.try_alloc().unwrap();
        let _c = alloc.try_alloc().unwrap();
        assert!(alloc.try_alloc().is_none());
    }

    #[test]
    fn effective_cap_clamps_issuance() {
        let alloc = GenIdAllocator::<256, 1>::new();
        alloc.set_effective_cap(3);
        let _a = alloc.try_alloc().unwrap(); // id 1
        let _b = alloc.try_alloc().unwrap(); // id 2
        assert!(alloc.try_alloc().is_none()); // id 3 would be >= cap
    }

    #[test]
    fn rollover_invalidates_old_ids() {
        let alloc = GenIdAllocator::<8, 1>::new();
        let stale = alloc.try_alloc().unwrap();
        assert!(alloc.is_current(stale));
        assert_eq!(stale.generation(), 1);

        alloc.rollover();
        assert_eq!(alloc.current_generation(), 2);
        assert!(!alloc.is_current(stale));

        let fresh = alloc.try_alloc().unwrap();
        assert_eq!(fresh.id(), 1); // slots were cleared
        assert_eq!(fresh.generation(), 2);
    }

    #[test]
    fn free_of_stale_id_is_noop() {
        let alloc = GenIdAllocator::<8, 1>::new();
        let stale = alloc.try_alloc().unwrap();
        alloc.rollover();
        let fresh = alloc.try_alloc().unwrap(); // gets id=1 in gen=2
        // Freeing the stale (gen=1, id=1) must NOT clear gen=2's slot.
        alloc.free(stale);
        assert!(alloc.is_current(fresh));
    }

    #[test]
    fn alloc_or_rollover_drives_full_cycle() {
        let alloc = GenIdAllocator::<4, 1>::new();
        let mut rollover_count = 0;
        // Fill to exhaustion (3 issuable IDs with MIN=1, MAX=4).
        for _ in 0..3 {
            alloc.alloc_or_rollover(|| rollover_count += 1);
        }
        assert_eq!(rollover_count, 0);
        let after = alloc.alloc_or_rollover(|| rollover_count += 1);
        assert_eq!(rollover_count, 1);
        assert_eq!(after.generation(), 2);
        assert_eq!(after.id(), 1);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let id = GenId::<65536>::new(0x1234, 0x5678);
        assert_eq!(GenId::<65536>::unpack(id.pack()), id);
        assert_eq!(GenId::<65536>::INVALID.pack(), 0);
    }

    #[test]
    fn min_id_reports_const_floor() {
        let alloc = GenIdAllocator::<256, 1>::new();
        assert_eq!(alloc.min_id(), 1);
        let alloc0 = GenIdAllocator::<256>::new();
        assert_eq!(alloc0.min_id(), 0);
    }

    #[test]
    fn wider_max_keeps_gen_in_high_bits() {
        // MAX = 1 << 20 (PASID-shaped): id in low 20 bits, gen in high 12.
        let id = GenId::<{ 1 << 20 }>::new(0xABCDE, 0xF12);
        assert_eq!(id.id(), 0xABCDE);
        assert_eq!(id.generation(), 0xF12);
        assert_eq!(id.pack(), (0xF12 << 20) | 0xABCDE);
    }

    #[test]
    fn id_bits_reverse_math_matches_widths() {
        assert_eq!(id_bits(1), 0);
        assert_eq!(id_bits(2), 1);
        assert_eq!(id_bits(4), 2);
        assert_eq!(id_bits(256), 8);
        assert_eq!(id_bits(4096), 12);
        assert_eq!(id_bits(65536), 16);
        assert_eq!(id_bits(1 << 20), 20);
    }
}
