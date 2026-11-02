//! `mapped` — **which held slot a mapped window came from**, so a memory
//! word can be named to the kernel by *authority* (`res/EVENT.md` §6).
//!
//! The kernel address-wait parks on a `u32` named as `(subject slot, byte
//! offset)` — a held lease or mapped view plus an offset within it — and
//! never as an address: authority is the mapping, and a raw address
//! crosses no boundary.  Userland code, though, holds the word as a
//! `&AtomicU32` inside a window some map verb returned.  This registry is
//! the bridge: every verb that installs a held slot into this task's
//! address space [`note`]s `(base, len, slot)` here, every unmap
//! [`forget`]s it, and [`locate`] turns a word's address back into the
//! `(slot, offset)` pair the kernel wants.  That is what makes the rung
//! **automatic** for existing callers — `event::wait_past` and the futex
//! consult it and fall back to the doorbell park when the word is not in
//! a noted window (an ordinary heap or static word, or a window mapped by
//! a path that does not note itself).
//!
//! Heapless and lock-free: a static bounded table of atomics claimed by
//! CAS on the base word (`0` = free, the null page is never mapped).  The
//! table is a **hint**, never an authority: the kernel re-validates the
//! slot it is handed against the caller's holdings, so a stale entry costs
//! a refused park (and the doorbell fallback), never a wrong wait.  A full
//! table drops the note — the window is still mapped, only the word rung
//! is declined for it.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use kore_abi::session::SlotRef;

/// Concurrently noted windows per task — a design bound sized to the
/// windows a service holds at once (rings, event channels, surfaces,
/// stashed views), with room to spare.  Refusal on overflow is a declined
/// rung, not a lost wake.
#[cfg(feature = "heap")]
const MAX_MAPPED: usize = 64;

/// The vacant entry.  Zero, and safely so: no mapped window starts at the
/// null page in any kore task.
#[cfg(feature = "heap")]
const FREE: usize = 0;

/// An entry mid-claim: base reserved, `len`/`slot` not yet published.  A
/// scan skips it exactly as it skips [`FREE`].  Not a mappable base either
/// (the top of the address space is never a window).
#[cfg(feature = "heap")]
const CLAIMING: usize = usize::MAX;

#[cfg(feature = "heap")]
struct Entry {
    base: AtomicUsize,
    len: AtomicUsize,
    slot: AtomicU32,
    /// The thread whose arena holds `slot` — capabilities are per-thread
    /// arenas, so a slot is meaningful only to the thread that mapped the
    /// window; a sibling handing that slot to the kernel is `Forged`/`Denied`
    /// (the storm the first `USER_SIMD=v3` boot hit in the terminal's parallel
    /// renderer).  `locate` answers only the noting thread; every other
    /// thread keeps the doorbell rung for that window.
    owner: AtomicUsize,
}

#[cfg(feature = "heap")]
impl Entry {
    const fn new() -> Self {
        Self {
            base: AtomicUsize::new(FREE),
            len: AtomicUsize::new(0),
            slot: AtomicU32::new(0),
            owner: AtomicUsize::new(0),
        }
    }
}

/// The bounded registry — generic over its capacity so the host suite can
/// exercise a private instance (including exhaustion) without touching
/// the task-global one.
#[cfg(feature = "heap")]
struct Registry<const N: usize> {
    entries: [Entry; N],
}

#[cfg(feature = "heap")]
impl<const N: usize> Registry<N> {
    const fn new() -> Self {
        Self {
            entries: [const { Entry::new() }; N],
        }
    }

    /// Claim a free entry for `(base, len, slot)`.  `false` when the table
    /// is full or the window is empty / degenerate (nothing to name).
    fn note(&self, base: usize, len: usize, slot: SlotRef, owner: usize) -> bool {
        if base == FREE || base == CLAIMING || len == 0 || base.checked_add(len).is_none() {
            return false;
        }
        // A remap supersedes: every entry of this thread overlapping the new
        // window is stale (its view died kernel-side — a lease retired by
        // its owner unmaps without a verb here — or it was replaced at the
        // same address).  Left in place it would be located first, and once
        // its slot index is reissued to another capability the wait keys
        // the wrong memory with no error at either end.
        self.forget_overlapping(base, len, owner);
        for entry in &self.entries {
            if entry
                .base
                .compare_exchange(FREE, CLAIMING, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                entry.len.store(len, Ordering::Relaxed);
                entry.slot.store(slot.index(), Ordering::Relaxed);
                entry.owner.store(owner, Ordering::Relaxed);
                // Publish: a reader that sees `base` sees `len`/`slot`/`owner`.
                entry.base.store(base, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Release the entry noted at `base`, if any.  Idempotent.
    fn forget(&self, base: usize) {
        if base == FREE || base == CLAIMING {
            return;
        }
        for entry in &self.entries {
            let _ = entry
                .base
                .compare_exchange(base, FREE, Ordering::AcqRel, Ordering::Relaxed);
        }
    }

    /// Release every entry of `owner` whose window overlaps `[base, base+len)`.
    fn forget_overlapping(&self, base: usize, len: usize, owner: usize) {
        let end = base.saturating_add(len);
        for entry in &self.entries {
            let old = entry.base.load(Ordering::Acquire);
            if old == FREE || old == CLAIMING || entry.owner.load(Ordering::Relaxed) != owner {
                continue;
            }
            let old_end = old.saturating_add(entry.len.load(Ordering::Relaxed));
            if old < end && base < old_end {
                let _ = entry
                    .base
                    .compare_exchange(old, FREE, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
    }

    /// Release every entry noted under `slot` — the unmap face that knows
    /// the capability name but not the window it installed.
    fn forget_owner(&self, owner: usize) {
        for entry in &self.entries {
            let base = entry.base.load(Ordering::Acquire);
            if base != FREE && base != CLAIMING && entry.owner.load(Ordering::Relaxed) == owner {
                let _ = entry
                    .base
                    .compare_exchange(base, FREE, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
    }

    fn forget_slot(&self, slot: SlotRef) {
        for entry in &self.entries {
            let base = entry.base.load(Ordering::Acquire);
            if base != FREE && base != CLAIMING && entry.slot.load(Ordering::Relaxed) == slot.index()
            {
                let _ = entry
                    .base
                    .compare_exchange(base, FREE, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
    }

    /// The `(slot, offset)` naming `addr`, when it falls inside a noted
    /// window.  Bounded scan; `None` when no window holds it or the offset
    /// does not fit the wire's `u32`.
    fn locate(&self, addr: usize, owner: usize) -> Option<(SlotRef, u32)> {
        for entry in &self.entries {
            let base = entry.base.load(Ordering::Acquire);
            if base == FREE || base == CLAIMING || addr < base {
                continue;
            }
            if entry.owner.load(Ordering::Relaxed) != owner {
                continue;
            }
            let len = entry.len.load(Ordering::Relaxed);
            if addr - base < len {
                let offset = u32::try_from(addr - base).ok()?;
                return Some((SlotRef::from_wire(entry.slot.load(Ordering::Relaxed)), offset));
            }
        }
        None
    }
}

/// The task-global registry — one per address space, shared by every
/// thread exactly as the futex registry is.
///
/// **Program builds only** (the `heap` feature — "everything a userland
/// program wants").  The one consumer built without it is the linker stub,
/// whose image the founder **shares read-only into every child**
/// (`SharedSegment` carries `execute`, never write): a static the stub
/// writes is a `#PF` (error 7) on its first store.  The stub runs the map
/// verbs that note windows here, so in that build the registry must not
/// merely be unused — it must not exist, or the stub's `.bss` grows by a
/// page that nothing may touch.  Without it the stub's own mappings are
/// simply not located, and every wait it could have taken keeps the
/// doorbell rung, which is what a task with no runtime wants anyway.
/// The calling thread's index — the arena the noted slot lives in.  The
/// futex registry's numbering (`entry::geometry::shard_index`); `None` when
/// the thread cannot name itself, which notes nothing and locates nothing.
#[cfg(feature = "heap")]
fn current_thread() -> Option<usize> {
    #[cfg(target_os = "none")]
    {
        crate::entry::geometry::shard_index()
    }
    #[cfg(not(target_os = "none"))]
    {
        Some(0)
    }
}

#[cfg(feature = "heap")]
static MAPPED: Registry<MAX_MAPPED> = Registry::new();

/// Note that `slot` (a held lease or mapped view) is installed at
/// `base..base+len` in this address space.  Called by every map verb that
/// returns a window; a full table declines quietly (the word rung falls
/// back to the doorbell for that window).  A no-op in the stub build (see
/// [`MAPPED`]).
#[cfg_attr(all(not(target_os = "none"), not(test)), expect(dead_code, reason = "noted only by the metal map verbs"))]
pub(crate) fn note(base: *const u8, len: usize, slot: SlotRef) {
    #[cfg(feature = "heap")]
    {
        if let Some(owner) = current_thread() {
            let _ = MAPPED.note(base as usize, len, slot, owner);
        }
    }
    #[cfg(not(feature = "heap"))]
    {
        let _ = (base, len, slot);
    }
}

/// Forget the window installed at `base` — the unmap face.
pub(crate) fn forget(base: *const u8) {
    #[cfg(feature = "heap")]
    MAPPED.forget(base as usize);
    #[cfg(not(feature = "heap"))]
    let _ = base;
}

/// Forget every window installed under `slot` — the unmap face that names
/// the capability rather than the address (`retire_mapped_view`).
pub(crate) fn forget_slot(slot: SlotRef) {
    #[cfg(feature = "heap")]
    MAPPED.forget_slot(slot);
    #[cfg(not(feature = "heap"))]
    let _ = slot;
}

/// Forget every window noted by thread `owner` — the thread-index reuse
/// face.  A thread's entries outlive it (its windows die with it, but the
/// registry cannot see that), and `claim_thread_index` hands the same index
/// to a later thread, which would then locate a dead sibling's slots as its
/// own — the confused deputy the per-thread key exists to prevent.  Called
/// at spawn, before the new thread runs.
pub(crate) fn forget_thread(owner: usize) {
    #[cfg(feature = "heap")]
    MAPPED.forget_owner(owner);
    #[cfg(not(feature = "heap"))]
    let _ = owner;
}

/// Name the word at `addr` by authority: the held slot whose window holds
/// it and the byte offset within that window — exactly what a
/// `WordWaitReq` / `WakeWordReq` carries.  `None` when the word is not in
/// a noted window (always, in the stub build); the caller keeps its
/// doorbell rung.
pub fn locate(addr: *const u8) -> Option<(SlotRef, u32)> {
    #[cfg(feature = "heap")]
    {
        MAPPED.locate(addr as usize, current_thread()?)
    }
    #[cfg(not(feature = "heap"))]
    {
        let _ = addr;
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_reused_thread_index_inherits_nothing() {
        let table = super::Registry::<2>::new();
        let slot = super::SlotRef::from_wire(3);
        assert!(table.note(0x1000, 0x100, slot, 7));
        table.forget_owner(7);
        assert_eq!(table.locate(0x1010, 7), None, "the dead thread's window is gone");
        assert!(table.note(0x2000, 0x100, slot, 7), "the entry is free again");
    }

    #[test]
    fn a_sibling_thread_does_not_locate_a_window_another_thread_noted() {
        let table = super::Registry::<2>::new();
        let slot = super::SlotRef::from_wire(3);
        assert!(table.note(0x1000, 0x100, slot, 7));
        assert_eq!(table.locate(0x1010, 7), Some((slot, 0x10)));
        assert_eq!(table.locate(0x1010, 8), None, "the slot lives in thread 7's arena");
    }

    use super::*;

    #[test]
    fn a_noted_window_locates_its_words_by_slot_and_offset() {
        let table = Registry::<4>::new();
        let slot = SlotRef::from_wire(17);
        assert!(table.note(0x1000, 0x200, slot, 0));
        assert_eq!(table.locate(0x1000, 0), Some((slot, 0)));
        assert_eq!(table.locate(0x1000 + 0x1fc, 0), Some((slot, 0x1fc)));
        assert_eq!(table.locate(0x1000 + 0x200, 0), None, "one past the end");
        assert_eq!(table.locate(0xfff, 0), None, "one before the base");
    }

    #[test]
    fn forgetting_by_base_or_slot_releases_the_entry() {
        let table = Registry::<4>::new();
        let a = SlotRef::from_wire(3);
        let b = SlotRef::from_wire(5);
        assert!(table.note(0x1000, 0x100, a, 0));
        assert!(table.note(0x2000, 0x100, b, 0));
        table.forget(0x1000);
        assert_eq!(table.locate(0x1010, 0), None);
        assert_eq!(table.locate(0x2010, 0), Some((b, 0x10)), "siblings survive");
        table.forget(0x1000);
        table.forget_slot(b);
        assert_eq!(table.locate(0x2010, 0), None);
        assert!(table.note(0x3000, 0x100, a, 0), "released entries are reusable");
    }

    /// The window lifecycle keytest's "editor did not receive keyboard
    /// input after re-layout": the client's view died kernel-side (the
    /// window's lease retired by its owner — no unmap verb ran here), the
    /// re-layout mapped the replacement at the same address, and the stale
    /// entry stood in front of the new one.  Once its slot index was
    /// reissued, every word wait keyed another capability's memory.
    #[test]
    fn a_remap_over_a_noted_window_supersedes_the_stale_entry() {
        let table = Registry::<4>::new();
        let stale = SlotRef::from_wire(3);
        let fresh = SlotRef::from_wire(11);
        assert!(table.note(0x1000, 0x2000, stale, 0));
        // The same address, a shorter window: the stale entry overlaps.
        assert!(table.note(0x1000, 0x1000, fresh, 0));
        assert_eq!(table.locate(0x1010, 0), Some((fresh, 0x10)), "the remap wins");
        assert_eq!(table.locate(0x1800, 0), Some((fresh, 0x800)));
        assert_eq!(table.locate(0x1fff, 0), Some((fresh, 0xfff)));
        assert_eq!(table.locate(0x2800, 0), None, "the stale tail is gone too");
        // A partial overlap from below evicts as well; a sibling thread's
        // entry at the same address is not this thread's to evict.
        assert!(table.note(0x1800, 0x100, stale, 5));
        assert!(table.note(0x0800, 0x1000, fresh, 0));
        assert_eq!(table.locate(0x1010, 0), Some((fresh, 0x810)), "the overlapping remap owns the address now");
        assert_eq!(table.locate(0x1900, 0), None, "the evicted tail past the remap is gone");
        assert_eq!(table.locate(0x1810, 5), Some((stale, 0x10)), "thread 5 keeps its own");
    }

    #[test]
    fn forgetting_by_base_releases_every_entry_at_that_base() {
        let table = Registry::<4>::new();
        let a = SlotRef::from_wire(3);
        let b = SlotRef::from_wire(5);
        assert!(table.note(0x1000, 0x100, a, 1));
        assert!(table.note(0x1000, 0x100, b, 2));
        table.forget(0x1000);
        assert_eq!(table.locate(0x1010, 1), None);
        assert_eq!(table.locate(0x1010, 2), None, "the second entry at the base is released too");
    }

    #[test]
    fn a_full_table_declines_the_note_and_degenerate_windows_are_refused() {
        let table = Registry::<2>::new();
        let slot = SlotRef::from_wire(1);
        assert!(table.note(0x1000, 0x100, slot, 0));
        assert!(table.note(0x2000, 0x100, slot, 0));
        assert!(!table.note(0x3000, 0x100, slot, 0), "bounded: the third is declined");
        assert_eq!(table.locate(0x3000, 0), None);
        assert!(!table.note(FREE, 0x100, slot, 0));
        assert!(!table.note(0x4000, 0, slot, 0));
        assert!(!table.note(usize::MAX - 1, 0x100, slot, 0), "wrapping span");
    }

    #[test]
    fn an_offset_past_u32_is_not_nameable() {
        let table = Registry::<1>::new();
        let slot = SlotRef::from_wire(9);
        let len = 1usize << 33;
        assert!(table.note(0x1000, len, slot, 0));
        assert_eq!(table.locate(0x1000 + 4, 0), Some((slot, 4)));
        assert_eq!(table.locate(0x1000 + (1usize << 32), 0), None);
    }

    #[test]
    fn the_task_global_face_notes_and_forgets_through_pointers() {
        // Distinct fake bases so parallel tests never alias one another.
        let base = 0x7f00_0000_1000usize as *const u8;
        let slot = SlotRef::from_wire(42);
        note(base, 64, slot);
        assert_eq!(locate(base.wrapping_add(8)), Some((slot, 8)));
        forget(base);
        assert_eq!(locate(base.wrapping_add(8)), None);
    }
}
