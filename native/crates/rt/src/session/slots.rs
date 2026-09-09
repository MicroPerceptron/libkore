//! `SlotAlloc` — the **arena slot allocator**, one of `SlotRef`'s two
//! lawful mints (the other is [`SlotRef::from_wire`], the wire-ingress
//! audit point; see the type's docs in `kore-abi`).
//!
//! A task's cap arena is a bounded array of slots; every staged result,
//! seeded record, and scratch run needs a slot nobody else is using.
//! Hand-picking indices proved doubly broken: literals collided silently
//! (the founder's slot-35/40/45 class — three metal/gate rounds), and a
//! `pub u32` constructor let any code path — including ones handling
//! peer-influenced bytes — steer which of the task's own capabilities a
//! session names (the in-task confused deputy).  The allocator replaces
//! *picking* with *claiming*: a claimed slot is yours until released, a
//! double claim is impossible by construction, and code that never
//! claimed a slot has no way to name one.
//!
//! Plain (non-atomic) by design: an arena belongs to one task, and after
//! the unified task model (#407) a *thread* is a task with its own arena
//! — so allocator instances are never shared across threads of control.
//! (The executor's cross-wake `OWNED_SLOTS` bitmap in `entry::geometry`
//! keeps its atomics for its own reasons.)

use kore_abi::session::SlotRef;
use kore_collections::BitSet;

const SLOT_BITMAP_WORDS: usize =
    (crate::entry::ARENA_SLOT_CEILING as usize + BitSet::<1>::BITS - 1) / BitSet::<1>::BITS;
type SlotBitmap = BitSet<SLOT_BITMAP_WORDS>;

/// Bitmap allocator over an arena's slot space, bounded by
/// [`ARENA_SLOT_CEILING`](crate::entry::ARENA_SLOT_CEILING).
///
/// The bitmap is **derived from that ceiling, never spelled**: it read
/// `u64` while the ceiling said 64, and when the ceiling moved to 128 the
/// saturation silently capped every allocator at half its arena — the
/// founder asks for `ARENA_SLOTS` and would have got 64, with no refusal
/// to notice.  A literal width here is a trap, because the failure is a
/// quiet ceiling rather than a compile error.
#[derive(Clone, Debug)]
pub struct SlotAlloc {
    /// Bit `i` set ⇒ slot `i` is claimed.
    used: SlotBitmap,
    /// One past the highest allocatable slot.
    slots: u32,
}

/// The allocation phase after every ABI-pinned position has been claimed.
///
/// This view deliberately exposes no [`SlotAlloc::claim_at`]: once variable-
/// sized manifest data starts consuming holes, introducing another fixed slot
/// would make correctness depend on source ordering again.  Consuming the
/// fixed planner into this type makes that phase boundary structural while
/// preserving the full arena's remaining capacity.
#[derive(Clone, Debug)]
pub struct DynamicSlotAlloc {
    inner: SlotAlloc,
}

const _: () = assert!(
    (crate::entry::ARENA_SLOT_CEILING as usize) <= SlotBitmap::BITS,
    "SlotAlloc's bitmap must cover a full arena — widen its shared BitSet with the ceiling"
);

impl SlotAlloc {
    /// An allocator over `slots` slots, all free.  `slots` saturates at
    /// [`ARENA_SLOT_CEILING`](crate::entry::ARENA_SLOT_CEILING).
    pub const fn new(slots: u32) -> Self {
        let ceiling = crate::entry::ARENA_SLOT_CEILING;
        Self {
            used: SlotBitmap::new(),
            slots: if slots > ceiling { ceiling } else { slots },
        }
    }

    /// An allocator with `reserved` low slots pre-claimed — the common
    /// shape when a range is owned by a frozen convention (a stamped
    /// boot block's named run, a wire contract) and everything above it
    /// is this allocator's to hand out.
    pub const fn with_reserved(slots: u32, reserved: u32) -> Self {
        let mut a = Self::new(slots);
        let r = if reserved > a.slots {
            a.slots
        } else {
            reserved
        };
        let mut i = 0;
        while i < r {
            let _ = a.used.set(i as usize);
            i += 1;
        }
        a
    }

    /// Claim the lowest free slot.  `None` when the arena is exhausted —
    /// fail-closed: the caller refuses its operation rather than
    /// guessing an index.
    pub fn claim(&mut self) -> Option<SlotRef> {
        let i = self.used.find_first_clear_from(0)?;
        if i >= self.slots as usize {
            return None;
        }
        let _ = self.used.set(i);
        Some(SlotRef::from_wire(i as u32))
    }

    /// Claim the lowest free slot the predicate admits — the
    /// **lifetime-classed** shape (#636): a value that must outlive this
    /// allocation round passes an `admit` that rejects every position a
    /// later round's fixed map may pin, so cross-round homes are chosen
    /// by allocation instead of by hand-census.  Deterministic
    /// (lowest-admitted-first), fail-closed on exhaustion.
    pub fn claim_where(&mut self, admit: impl Fn(u32) -> bool) -> Option<SlotRef> {
        let mut from = 0usize;
        loop {
            let i = self.used.find_first_clear_from(from)?;
            if i >= self.slots as usize {
                return None;
            }
            if admit(i as u32) {
                let _ = self.used.set(i);
                return Some(SlotRef::from_wire(i as u32));
            }
            from = i + 1;
        }
    }

    /// Claim a **specific** slot — for positions pinned by a frozen
    /// contract (an ABI-named landing slot, a fixture mirroring a
    /// stamped block).  `None` if out of range or already claimed: a
    /// collision is a loud refusal here, never a silent overwrite.
    pub fn claim_at(&mut self, index: u32) -> Option<SlotRef> {
        if index >= self.slots || self.used.contains(index as usize) {
            return None;
        }
        let _ = self.used.set(index as usize);
        Some(SlotRef::from_wire(index))
    }

    /// Claim `count` **consecutive** slots, returning the first — the
    /// scratch-run shape (a grant-staging run, a per-index result band).
    pub fn claim_run(&mut self, count: u32) -> Option<SlotRef> {
        if count == 0 || count > self.slots {
            return None;
        }
        let mut base = 0u32;
        while base + count <= self.slots {
            if (base..base + count).all(|i| !self.used.contains(i as usize)) {
                for i in base..base + count {
                    let _ = self.used.set(i as usize);
                }
                return Some(SlotRef::from_wire(base));
            }
            base += 1;
        }
        None
    }

    /// Release a claimed slot.  Releasing a free slot is a no-op (the
    /// teardown convention: best-effort, never double-faults).
    pub fn release(&mut self, slot: SlotRef) {
        if slot.index() < self.slots {
            let _ = self.used.clear(slot.index() as usize);
        }
    }

    /// Whether `slot` is currently claimed.
    pub fn is_claimed(&self, slot: SlotRef) -> bool {
        slot.index() < self.slots && self.used.contains(slot.index() as usize)
    }

    /// The number of free slots remaining.
    pub fn free(&self) -> u32 {
        self.slots - self.used.count_ones() as u32
    }

    /// Finish the ABI-pinned phase and permit only dynamic allocation.
    ///
    /// Consumes `self` so the caller cannot retain an alias capable of adding
    /// another fixed claim after the first dynamic result has been allocated.
    pub fn finish_fixed(self) -> DynamicSlotAlloc {
        DynamicSlotAlloc { inner: self }
    }
}

impl DynamicSlotAlloc {
    /// Claim the lowest free slot from the completed fixed plan.
    pub fn claim(&mut self) -> Option<SlotRef> {
        self.inner.claim()
    }

    /// Claim the lowest free slot the predicate admits — the
    /// lifetime-classed shape ([`SlotAlloc::claim_where`]), lawful after
    /// the fixed phase closed: the predicate CONSTRAINS a dynamic claim,
    /// it never names a specific position, so the phase boundary's
    /// no-new-pins guarantee holds.
    pub fn claim_where(&mut self, admit: impl Fn(u32) -> bool) -> Option<SlotRef> {
        self.inner.claim_where(admit)
    }

    /// Claim a consecutive run from the completed fixed plan.
    pub fn claim_run(&mut self, count: u32) -> Option<SlotRef> {
        self.inner.claim_run(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allocator's whole contract: lowest-free claiming, loud
    /// `claim_at` collisions, run claiming, release/reuse, exhaustion.
    #[test]
    fn claim_release_collide_exhaust() {
        let mut a = SlotAlloc::new(4);
        let s0 = a.claim().unwrap();
        let s1 = a.claim().unwrap();
        assert_eq!(s0.index(), 0);
        assert_eq!(s1.index(), 1);
        // A pinned claim on a taken slot refuses loudly.
        assert!(a.claim_at(1).is_none(), "collision is a refusal");
        assert!(a.claim_at(3).is_some(), "a free pinned slot claims");
        assert_eq!(a.claim().unwrap().index(), 2, "lowest free");
        assert!(a.claim().is_none(), "exhaustion refuses");
        a.release(s1);
        assert_eq!(a.claim().unwrap().index(), 1, "released slots reuse");
    }

    /// `claim_where` skips rejected positions, stays deterministic
    /// (lowest admitted first), and refuses when nothing is admitted.
    #[test]
    fn claim_where_is_lowest_admitted() {
        let mut a = SlotAlloc::new(8);
        assert_eq!(a.claim().unwrap().index(), 0);
        // 1 and 2 are free but rejected; 3 is the lowest admitted.
        let s = a.claim_where(|i| i >= 3).unwrap();
        assert_eq!(s.index(), 3);
        // The rejected holes stay free for an unconstrained claim.
        assert_eq!(a.claim().unwrap().index(), 1);
        // An unsatisfiable predicate refuses rather than mis-allocating.
        assert!(a.claim_where(|i| i >= 8).is_none());
    }

    /// `with_reserved` pre-claims the frozen low run; `claim_run` finds
    /// a consecutive band above it.
    #[test]
    fn reserved_and_runs() {
        let mut a = SlotAlloc::with_reserved(16, 10);
        assert!(a.claim_at(9).is_none(), "reserved slots are claimed");
        let run = a.claim_run(3).unwrap();
        assert_eq!(run.index(), 10, "the run starts past the reservation");
        assert!(a.is_claimed(SlotRef::from_wire(12)));
        assert_eq!(a.free(), 3);
        assert!(a.claim_run(4).is_none(), "an oversized run refuses");
    }

    /// The dynamic phase reuses holes in the completed fixed plan rather than
    /// imposing a high-band capacity ceiling.
    #[test]
    fn dynamic_phase_reuses_fixed_holes() {
        let mut a = SlotAlloc::new(16);
        assert!(a.claim_at(3).is_some());
        assert!(a.claim_at(10).is_some());
        let mut dynamic = a.finish_fixed();
        assert_eq!(dynamic.claim().unwrap().index(), 0);
        assert_eq!(dynamic.claim_run(2).unwrap().index(), 1);
        assert_eq!(dynamic.claim().unwrap().index(), 4);
    }

    /// The full-arena boundary: mask arithmetic must not shift out of
    /// range at the widest allocator the ceiling admits.
    ///
    /// Derived from [`ARENA_SLOT_CEILING`], never spelled — a literal 64
    /// here kept passing after the ceiling moved to 128, while every
    /// allocator was silently capped at half its arena.  The number is
    /// the thing under test, so it cannot also be the assumption.
    ///
    /// [`ARENA_SLOT_CEILING`]: crate::entry::ARENA_SLOT_CEILING
    #[test]
    fn full_width_arena() {
        const N: u32 = crate::entry::ARENA_SLOT_CEILING;
        let mut a = SlotAlloc::new(N);
        for i in 0..N {
            assert_eq!(a.claim().unwrap().index(), i);
        }
        assert!(a.claim().is_none());
        assert_eq!(a.free(), 0);
        a.release(SlotRef::from_wire(N - 1));
        assert_eq!(a.claim().unwrap().index(), N - 1);
    }

    /// **An allocator asked for a full arena gets one.**  The regression
    /// this file's saturation caused: `SlotAlloc::new(ARENA_SLOTS)` --
    /// which is exactly what the founder calls (`found_service`) -- must
    /// hand out every slot, not the first 64.  A cap here is invisible at
    /// the callsite because exhaustion is a legitimate `None`.
    #[test]
    fn a_full_arena_request_is_not_silently_capped() {
        let a = SlotAlloc::new(crate::entry::ARENA_SLOT_CEILING);
        assert_eq!(
            a.free(),
            crate::entry::ARENA_SLOT_CEILING,
            "the allocator must span the whole ceiling"
        );
    }
}
