//! **The stack arena** — strips of stack, carved lock-free from a plan
//! (`res/SPMD.md` §8, R4).
//!
//! Worker stacks leave the per-thread band plan for arena **strips**: a
//! region owns one virtual window, and stacks are *allocated* from it
//! rather than tiled by a boot-time constant.  This module is the
//! engine — pure geometry over an atomic ledger, no memory of its own —
//! in two halves:
//!
//! - [`StripPlan`] — a granule ledger over a caller-named window.  The
//!   ledger is [`AtomicBitSet`] run-claims (`try_claim_run`'s
//!   bounded-rollback discipline, set = free), not a bespoke word
//!   scheme: the lock-free multi-granule carve this needed already
//!   lives in `kore-collections`, proven.  Plans are `const`-built
//!   (`with_first_set`), so a region's plan can be a `static` beside
//!   its [`SpmdCtx`](crate::spmd::SpmdCtx).
//! - `BackingClaim` + [`provision`] — the memory half's seam: an
//!   initially-unbacked plan reserves its next contiguous tail, maps it
//!   through a [`MemSource`], and publishes those granules to carvers
//!   only after the mapping exists.  The affine claim rolls the
//!   reservation back if the source refuses, so an interrupted grow can
//!   never strand the plan in a half-provisioned state.
//!
//! # The two strip classes
//!
//! §8's classes are two **carve flavors**, not two engines:
//!
//! - [`carve`](StripPlan::carve) — an **audited** strip: every claimed
//!   granule is usable.  The lane tier's fibers (§13.3) take these;
//!   their safety story is the yield-time audit plus the bounded-frame
//!   contract, never a guard.
//! - [`carve_guarded`](StripPlan::carve_guarded) — a **guarded** strip:
//!   one extra granule is claimed *below* the usable span and never
//!   handed out.  The founding session maps the usable span with its
//!   guard page below (the map verb's guard lands exactly in the
//!   reserved granule), so overflow takes a fault, not a neighbour.
//!   Kernel-thread workers take these.
//!
//! A window hosting both classes would fragment one against the other,
//! so the intended composition is one plan per class over disjoint
//! windows — policy the integration states, not the engine.
//!
//! # Carve policy: first-fit from the bottom
//!
//! [`carve`](StripPlan::carve) scans from granule 0 and takes the first
//! free run.  Low addresses cluster deliberately: the high-water mark
//! stays low, so a grow-window source maps tail pages late or never —
//! the same reason the heap bumps from its window's base.  A rotating
//! cursor would spread contention but pin the watermark high; stacks
//! churn too slowly for that trade.  One forward pass per carve: `None`
//! means "no run observed free this pass" — a racing release may admit
//! a retry, and the caller owns that loop.

use core::sync::atomic::{AtomicU32, Ordering};

use kore_collections::AtomicBitSet;

/// The high bit serialises one plan's rare backing-growth path.  Live
/// capacities are bounded by the ledger (`WORDS * 64`) and therefore
/// never approach it.
const BACKING_IN_PROGRESS: u32 = 1 << 31;
const BACKED_MASK: u32 = !BACKING_IN_PROGRESS;

/// One granule: the page, the quantum both strip classes share.  A
/// guarded strip's guard is one granule; the finest audited strip
/// (§13.3's 8-bit-field fibers) is one granule.
pub const GRANULE: usize = 4096;

/// An affine strip — the carve's receipt, and the only way to release.
///
/// Move-only (no `Copy`/`Clone`): a strip freed twice would hand one
/// stack to two workers, so the ticket is consumed by
/// [`release`](StripPlan::release), the same discipline every kore
/// capability follows even though this is bookkeeping, not authority.
#[must_use = "an unreleased strip keeps its granules claimed for the plan's life"]
#[derive(Debug, PartialEq, Eq)]
pub struct Strip {
    /// First byte of the **usable** span (past the guard, if any).
    base: u64,
    /// Usable bytes.
    bytes: u32,
    /// First claimed granule (the guard's, on a guarded strip).
    first: u32,
    /// Claimed granules, guard included.
    claimed: u32,
}

impl Strip {
    /// First byte of the usable span.
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// Usable bytes — what a founding maps, what a fiber owns.
    pub const fn bytes(&self) -> u32 {
        self.bytes
    }

    /// One past the last usable byte — where a stack's top begins.
    pub const fn end(&self) -> u64 {
        self.base + self.bytes as u64
    }

    /// Whether a guard granule rides below the usable span.
    pub const fn guarded(&self) -> bool {
        (self.claimed as u64) * GRANULE as u64 != self.bytes as u64
    }
}

/// A granule ledger over one virtual window — the arena's engine.
///
/// `GRANULES` is the window's capacity; the ledger word count follows
/// (`AtomicBitSet` packs 64 granules per word, rounded up).  The plan
/// holds **numbers, never memory**: whether the window is backed by a
/// mapped region, a [`provision`]ed extent, or nothing yet is the
/// caller's affair — which is what makes every law below provable on
/// the host.
pub struct StripPlan<const WORDS: usize> {
    /// The claim ledger: set = free, clear = claimed, tail born claimed.
    ledger: AtomicBitSet<WORDS>,
    /// The window's first byte; granule-aligned.
    base: u64,
    /// Live capacity in granules (≤ `WORDS * 64`).
    granules: u32,
    /// Strips currently claimed — telemetry for a status row, never a
    /// synchronization fact.
    live: AtomicU32,
    /// Contiguous backed prefix in granules, with the high bit set while
    /// one affine [`BackingClaim`] owns the next tail.  This is a
    /// per-plan slow-path gate, not a global lock: ordinary carves touch
    /// only `ledger`.
    backing: AtomicU32,
}

impl<const WORDS: usize> StripPlan<WORDS> {
    /// A plan over `[base, base + granules · GRANULE)`.
    ///
    /// `const`, so a region's plan is a `static` beside its ctx — no
    /// init pass, the ledger born all-free ([`AtomicBitSet::with_first_set`]).
    ///
    /// # Panics
    /// A misaligned base, zero granules, or more granules than the
    /// ledger's `WORDS` hold — all compile-time when const-evaluated.
    pub const fn new(base: u64, granules: u32) -> Self {
        assert!(base % GRANULE as u64 == 0, "window base off the granule");
        assert!(granules > 0, "a plan needs a window");
        assert!(granules as usize <= WORDS * 64, "ledger too small");
        assert!(granules < BACKING_IN_PROGRESS, "backing state overflow");
        Self {
            ledger: AtomicBitSet::with_first_set(granules as usize),
            base,
            granules,
            live: AtomicU32::new(0),
            backing: AtomicU32::new(granules),
        }
    }

    /// A plan whose virtual window is reserved but has no mapped pages
    /// yet.  Carves refuse until a [`BackingClaim`] successfully
    /// [`back_with`](BackingClaim::back_with)s a prefix through a
    /// [`MemSource`](crate::mem_source::MemSource).
    ///
    /// This is the paged-arena posture: capacity names the maximum VA
    /// window, while the free ledger contains only pages already made
    /// live.  Tail publication is monotone; strip release recycles only
    /// within the published prefix.
    pub(crate) const fn unbacked(base: u64, granules: u32) -> Self {
        assert!(base % GRANULE as u64 == 0, "window base off the granule");
        assert!(granules > 0, "a plan needs a window");
        assert!(granules as usize <= WORDS * 64, "ledger too small");
        assert!(granules < BACKING_IN_PROGRESS, "backing state overflow");
        Self {
            ledger: AtomicBitSet::new(),
            base,
            granules,
            live: AtomicU32::new(0),
            backing: AtomicU32::new(0),
        }
    }

    /// The window's capacity, in granules.
    pub const fn granules(&self) -> u32 {
        self.granules
    }

    /// Strips currently out — telemetry, racy by nature.
    pub fn live(&self) -> u32 {
        self.live.load(Ordering::Relaxed)
    }

    /// Mapped prefix, in granules.  Telemetry and growth policy only;
    /// `ledger` remains the allocation truth.
    pub fn backed(&self) -> u32 {
        self.backing.load(Ordering::Acquire) & BACKED_MASK
    }

    /// Reserve the next unbacked tail for one mapping attempt.
    ///
    /// `needed` is the run the caller must be able to carve after a
    /// successful grow; `quantum` amortises the mapping trap.  The claim
    /// takes `max(needed, quantum)`, capped by the plan tail, and refuses
    /// rather than publishing a tail shorter than `needed`.
    ///
    /// The returned ticket is affine.  Dropping it before
    /// [`back_with`](BackingClaim::back_with) succeeds clears the
    /// in-progress bit and wakes contenders; no explicit rollback path
    /// can be forgotten.
    pub(crate) fn reserve_backing(
        &self,
        needed: u32,
        quantum: u32,
    ) -> Result<BackingClaim<'_, WORDS>, BackingRefused> {
        if needed == 0 || quantum == 0 || needed > self.granules {
            return Err(BackingRefused::Invalid);
        }
        loop {
            let observed = self.backing.load(Ordering::Acquire);
            if observed & BACKING_IN_PROGRESS != 0 {
                return Err(BackingRefused::Busy { observed });
            }
            let first = observed & BACKED_MASK;
            let remaining = self.granules - first;
            if remaining < needed {
                return Err(BackingRefused::Full);
            }
            let granules = needed.max(quantum).min(remaining);
            match self.backing.compare_exchange_weak(
                observed,
                observed | BACKING_IN_PROGRESS,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(BackingClaim {
                        plan: self,
                        first,
                        granules,
                        published: false,
                    });
                }
                Err(_) => continue,
            }
        }
    }

    /// The provision-state word contenders wait on.  Kept crate-private:
    /// the word is a wake hint, while [`reserve_backing`](Self::reserve_backing)
    /// remains the only authority to change it.
    pub(crate) const fn backing_word(&self) -> &AtomicU32 {
        &self.backing
    }

    /// Exact wait value while another tail claim owns the mapping slow
    /// path.  `None` means no mapper is live; callers without backing
    /// authority must refuse then rather than trying to become one.
    pub(crate) fn backing_in_progress(&self) -> Option<u32> {
        let observed = self.backing.load(Ordering::Acquire);
        (observed & BACKING_IN_PROGRESS != 0).then_some(observed)
    }

    /// Carve an **audited** strip of `granules` granules: every claimed
    /// byte usable, no guard — the fiber class (§13.3), whose safety is
    /// the yield audit, not a fault.
    pub fn carve(&self, granules: u32) -> Option<Strip> {
        self.carve_run(granules, 0)
    }

    /// Carve a **guarded** strip: `granules` usable granules plus one
    /// claimed guard granule below them, never handed out.  The
    /// founding maps the usable span with its guard page below — the
    /// verb's guard lands exactly in the reserved granule.
    pub fn carve_guarded(&self, granules: u32) -> Option<Strip> {
        self.carve_run(granules, 1)
    }

    fn carve_run(&self, usable: u32, guard: u32) -> Option<Strip> {
        if usable == 0 {
            return None;
        }
        let want = (usable + guard) as usize;
        // First-fit from the bottom (module docs).  `find_first_set_from`
        // locates a free granule; the run claim then settles the race —
        // an `Err` means some granule inside was taken between the look
        // and the claim (or was never free), and the scan resumes one
        // past the disappointment.  One forward pass, monotone cursor,
        // so exhaustion terminates.
        let mut cursor = 0usize;
        while let Some(first) = self.ledger.find_first_set_from(cursor) {
            if first + want > self.granules as usize {
                return None;
            }
            if self.ledger.try_claim_run(first, want).is_ok() {
                self.live.fetch_add(1, Ordering::Relaxed);
                return Some(Strip {
                    base: self.base + ((first + guard as usize) * GRANULE) as u64,
                    bytes: (usable as usize * GRANULE) as u32,
                    first: first as u32,
                    claimed: (usable + guard),
                });
            }
            cursor = first + 1;
        }
        None
    }

    /// Give a strip back; its granules are immediately carveable.
    ///
    /// Consumes the affine ticket — release is once, by construction.
    ///
    /// # Panics
    /// Debug builds refuse a strip whose coordinates lie outside this
    /// plan (a ticket from another plan): release would free granules
    /// someone else owns.
    pub fn release(&self, strip: Strip) {
        debug_assert!(
            strip.first + strip.claimed <= self.granules
                && strip.base >= self.base
                && strip.end() <= self.base + (self.granules as u64) * GRANULE as u64,
            "a strip from another plan"
        );
        self.ledger
            .release_run(strip.first as usize, strip.claimed as usize);
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Why a plan could not reserve another mapped tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackingRefused {
    /// Another affine claim owns the slow path.  `observed` is the exact
    /// futex value a bounded waiter may monitor.
    Busy { observed: u32 },
    /// No tail large enough for the requested run remains.
    Full,
    /// Zero geometry or a request wider than the plan.
    Invalid,
}

/// An affine reservation of an unbacked plan tail.
///
/// The ticket owns only the right to publish this exact tail.  Memory is
/// still absent until [`back_with`](Self::back_with) obtains an extent at
/// the claim's fixed address.  Drop is rollback; successful publication
/// consumes the ticket.
#[must_use = "dropping a backing claim rolls its plan reservation back"]
pub(crate) struct BackingClaim<'a, const WORDS: usize> {
    plan: &'a StripPlan<WORDS>,
    first: u32,
    granules: u32,
    published: bool,
}

impl<const WORDS: usize> BackingClaim<'_, WORDS> {
    /// Fixed virtual base this extent must map at.
    pub(crate) const fn base(&self) -> u64 {
        self.plan.base + self.first as u64 * GRANULE as u64
    }

    /// Granules in this tail.
    pub(crate) const fn granules(&self) -> u32 {
        self.granules
    }

    /// Map this exact tail through `source`, then make it carveable.
    ///
    /// A source-selected address is not good enough: the ledger already
    /// names a fixed VA.  A mismatching or refused extent leaves the
    /// claim unpublished and Drop restores the prior backing state.
    pub(crate) fn back_with<M: crate::mem_source::MemSource>(mut self, source: &M) -> Option<()> {
        let mapped = provision(source, self.granules)?;
        if mapped != self.base() {
            return None;
        }

        // Mapping commit precedes this release in the source.  AcqRel on
        // the bitset publication makes a carved strip observe only live
        // pages; clearing the gate afterwards wakes growth contenders.
        self.plan
            .ledger
            .release_run(self.first as usize, self.granules as usize);
        self.plan
            .backing
            .store(self.first + self.granules, Ordering::Release);
        self.published = true;
        #[cfg(target_os = "none")]
        {
            let _ = crate::futex::wake_all(self.plan.backing_word());
        }
        Some(())
    }
}

impl<const WORDS: usize> Drop for BackingClaim<'_, WORDS> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        debug_assert_eq!(
            self.plan.backing.load(Ordering::Relaxed),
            self.first | BACKING_IN_PROGRESS,
            "only this affine claim may own the backing gate"
        );
        self.plan.backing.store(self.first, Ordering::Release);
        #[cfg(target_os = "none")]
        {
            let _ = crate::futex::wake_all(self.plan.backing_word());
        }
    }
}

/// Carve a granule-aligned window for a plan out of a
/// [`MemSource`](crate::mem_source::MemSource) — the arena's memory half,
/// over the same seam the heap grows through.
///
/// Asks the source for `granules · GRANULE` and returns its base, or
/// `None` when the source is exhausted, short, or violates the
/// [`MemSource`](crate::mem_source::MemSource) alignment contract.
/// The caller's backing owner keeps the extent live; the plan owns only
/// the geometry — exactly the split that lets the engine above stay
/// memoryless.
///
/// The seam and this engine are allocation-free and unconditional; the
/// optional heap is merely another consumer.  A heapless program can
/// therefore provision a retained arena without linking `alloc`.
pub fn provision<M: crate::mem_source::MemSource>(source: &M, granules: u32) -> Option<u64> {
    let bytes = (granules as usize).checked_mul(GRANULE)?;
    if bytes == 0 {
        return None;
    }
    let extent = source.map_extent(bytes)?;
    let raw = extent.as_ptr().cast::<u8>() as u64;
    if raw % GRANULE as u64 != 0 || extent.len() < bytes {
        return None;
    }
    Some(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    const BASE: u64 = 0x5000_0000;

    #[test]
    fn a_static_plan_carves_and_releases_at_its_edges() {
        static PLAN: StripPlan<1> = StripPlan::new(BASE, 8);
        let a = PLAN.carve(8).expect("the whole window is one strip");
        assert_eq!((a.base(), a.bytes()), (BASE, 8 * GRANULE as u32));
        assert!(!a.guarded());
        assert!(PLAN.carve(1).is_none(), "nothing left");
        PLAN.release(a);
        assert_eq!(PLAN.live(), 0);
        assert!(PLAN.carve(8).is_some(), "release restored everything");
    }

    #[test]
    fn a_guarded_strip_reserves_its_guard_below() {
        static PLAN: StripPlan<1> = StripPlan::new(BASE, 8);
        let s = PLAN.carve_guarded(3).expect("3 + guard fit in 8");
        // The guard granule is claimed but below the usable span.
        assert_eq!(s.base(), BASE + GRANULE as u64);
        assert_eq!(s.bytes(), 3 * GRANULE as u32);
        assert!(s.guarded());
        // 4 of 8 granules are gone (guard included).
        let rest = PLAN.carve(4).expect("the other 4 remain");
        assert!(PLAN.carve(1).is_none());
        PLAN.release(rest);
        PLAN.release(s);
        assert!(PLAN.carve(8).is_some(), "guard granules came back too");
    }

    #[test]
    fn a_freed_hole_is_found_below_the_watermark() {
        static PLAN: StripPlan<1> = StripPlan::new(BASE, 8);
        let low = PLAN.carve(2).expect("low");
        let _hi = PLAN.carve(2).expect("high");
        PLAN.release(low);
        // First-fit returns to the bottom hole, not past the watermark.
        let again = PLAN.carve(2).expect("the hole");
        assert_eq!(again.base(), BASE);
        PLAN.release(again);
    }

    #[test]
    fn oversized_and_zero_carves_refuse() {
        static PLAN: StripPlan<1> = StripPlan::new(BASE, 8);
        assert!(PLAN.carve(9).is_none());
        assert!(PLAN.carve(0).is_none());
        assert!(PLAN.carve_guarded(8).is_none(), "guard needs its granule");
    }

    #[test]
    fn an_unbacked_plan_publishes_nothing_and_an_abandoned_claim_rolls_back() {
        let plan = StripPlan::<1>::unbacked(BASE, 16);
        assert_eq!(plan.backed(), 0);
        assert!(plan.carve(1).is_none(), "unmapped bytes are never free");

        let claim = plan.reserve_backing(4, 8).expect("first tail claim");
        assert_eq!((claim.base(), claim.granules()), (BASE, 8));
        assert!(plan.backing_in_progress().is_some());
        assert!(matches!(
            plan.reserve_backing(1, 1),
            Err(BackingRefused::Busy { .. })
        ));
        drop(claim);

        assert_eq!(plan.backed(), 0, "drop restored the prior prefix");
        assert_eq!(plan.backing_in_progress(), None);
        assert!(
            plan.reserve_backing(16, 1).is_ok(),
            "the whole tail reopened"
        );
    }

    #[cfg(feature = "heap")]
    #[test]
    fn a_mem_source_extent_becomes_carveable_only_after_back_with() {
        let mut arena = std::vec![0u8; GRANULE * 16 + crate::heap::EXTENT_ALIGN];
        let raw = arena.as_mut_ptr() as usize;
        let aligned = raw.next_multiple_of(crate::heap::EXTENT_ALIGN);
        // SAFETY: `arena` remains live and otherwise untouched for the
        // source and plan's complete lifetime in this test.
        let source = unsafe { crate::heap::StaticMemSource::over(arena.as_mut_ptr(), arena.len()) };
        let plan = StripPlan::<1>::unbacked(aligned as u64, 16);
        let claim = plan.reserve_backing(16, 16).expect("whole mapped tail");
        claim
            .back_with(&source)
            .expect("source mapped the exact plan base");

        assert_eq!(plan.backed(), 16);
        let strip = plan.carve(16).expect("publication made pages live");
        assert_eq!(strip.base(), aligned as u64);
        plan.release(strip);

        let recycled = plan.carve(16).expect("released live pages are reusable");
        assert_eq!(
            recycled.base(),
            aligned as u64,
            "recycling geometry does not require a second mapping"
        );
        plan.release(recycled);
        assert!(matches!(
            plan.reserve_backing(1, 1),
            Err(BackingRefused::Full)
        ));
    }

    /// The concurrency law: strips carved by racing threads are
    /// pairwise disjoint, and a full drain restores full capacity.
    #[test]
    fn racing_carves_never_overlap_and_all_come_back() {
        static PLAN: StripPlan<4> = StripPlan::new(BASE, 256);
        let strips: Vec<Strip> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        let mut got = Vec::new();
                        for round in 0..64u32 {
                            let n = 1 + (round % 3);
                            if let Some(strip) = if round % 2 == 0 {
                                PLAN.carve(n)
                            } else {
                                PLAN.carve_guarded(n)
                            } {
                                got.push(strip);
                            }
                        }
                        got
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().expect("no carver panics"))
                .collect()
        });
        // Pairwise disjoint over CLAIMED coordinates (guards included).
        let mut owned = [false; 256];
        for s in &strips {
            for g in s.first..s.first + s.claimed {
                assert!(!owned[g as usize], "granule {g} carved twice");
                owned[g as usize] = true;
            }
        }
        for s in strips {
            PLAN.release(s);
        }
        assert_eq!(PLAN.live(), 0);
        let all = PLAN.carve(256).expect("every granule came back");
        PLAN.release(all);
    }
}
