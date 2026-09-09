//! `MemSource` — where the allocator gets fresh extents to carve, behind a
//! trait so the backing can change without touching the engine.
//!
//! On a true miss (no local block, no remote frees to reclaim) the `Heap` grows
//! **synchronously**: it maps one fresh extent and carves it.  Userland can do
//! this on the miss path itself — there is no ISR / saved-FP-context / lock-held
//! hazard that forces the kernel heap's *background* grower — so growth is just
//! a call through this seam.
//!
//! The engine ships one backing; the default lives with its service:
//!
//! - [`StaticMemSource`] (here) — hands out sub-ranges of a fixed byte
//!   arena by bump.  Fully functional and host-testable **now**, and the
//!   shape a freestanding binary uses before the kernel can map memory.
//! - [`HeapSupplySource`] (here) — the **metal** source every founded
//!   binary links: a miss raises the founder-window grow trap and the
//!   kernel maps the next slice of the task's reserved backing.  An
//!   mm-*served* source (extent growth as memory-manager grants, for
//!   mm's peers) is the vertical-slice rung ahead — it will be built
//!   against the resident mm's real serve loop.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::mem_source::MemSource;

use super::EXTENT_ALIGN;
use super::pool::checked_align_up;

/// Bump `want` bytes off `[base, base+len)`, returning an [`EXTENT_ALIGN`]-aligned
/// start (so the heap's self-hosted header lands aligned), or `None` when the
/// remaining tail can't fit.  Lock-free: concurrent growers race on `cursor`.
fn bump(base: usize, len: usize, cursor: &AtomicUsize, want: usize) -> Option<usize> {
    let limit = base.checked_add(len)?;
    let mut cur = cursor.load(Ordering::Relaxed);
    loop {
        let start = checked_align_up(base.checked_add(cur)?, EXTENT_ALIGN)?;
        let end = start.checked_add(want)?;
        if end > limit {
            return None;
        }
        match cursor.compare_exchange_weak(cur, end - base, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return Some(start),
            Err(actual) => cur = actual,
        }
    }
}

/// A [`MemSource`] over a fixed byte arena, handed out by bump.  The functional,
/// host-testable backing (and the shape a freestanding binary uses before the
/// kernel can map memory).  Never reclaims a mapped extent — matching the kernel
/// heap, extents are retained for the arena's life; the `Heap` recycles freed
/// pool blocks and huge extents internally instead.
pub struct StaticMemSource {
    base: *mut u8,
    len: usize,
    cursor: AtomicUsize,
}

// SAFETY: the arena is addressed only through atomics (`cursor`) and raw reads
// of disjoint sub-ranges the heap then owns exclusively; no aliasing of a live
// sub-range crosses threads, so `&StaticMemSource` is safe to share.
unsafe impl Send for StaticMemSource {}
unsafe impl Sync for StaticMemSource {}

impl StaticMemSource {
    /// A source over the caller-owned arena `[base, base+len)`.  `const`, so a
    /// binary can back a `static` `Heap` with a `static` arena.
    ///
    /// # Safety
    /// `base` must point at `len` bytes valid for the source's whole life, owned
    /// by no one else (the heap carves them).
    pub const unsafe fn over(base: *mut u8, len: usize) -> Self {
        Self {
            base,
            len,
            cursor: AtomicUsize::new(0),
        }
    }
}

// SAFETY: `over` requires one exclusively-owned, process-lived arena.  The
// atomic bump hands every successful caller a disjoint aligned sub-range and
// never rewinds or otherwise touches returned bytes.
unsafe impl MemSource for StaticMemSource {
    fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
        let want = checked_align_up(min_bytes.max(1), EXTENT_ALIGN)?;
        let start = bump(self.base as usize, self.len, &self.cursor, want)?;
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(start as *mut u8)?,
            want,
        ))
    }
}

/// A [`MemSource`] that grows from the task's **founder-seeded
/// `role::HEAP` supply** (#336 rung 6 — the retired `HEAP_GROW` sentinel's
/// session form).  The founder endows every service with a real `MemLease`
/// over the manifest's `heap_bytes` of contiguous backing; each heap miss
/// runs **one ordinary session of two invocations** through the trap
/// transport:
///
/// 1. a **detached window carve** off the supply (`Derive × kind =
///    LeasedRegion`, the mm-grant wire shape) at this source's own monotone
///    offset cursor, staged into the carve scratch;
/// 2. the **staged map flavor** on that carve — the kernel installs it at
///    the task's map-window cursor and answers with the
///    [`GeomReport`](kore_abi::session::GeomReport) naming the fresh extent.
///
/// Extents land at cursor-chosen (disjoint) VAs, which [`MemSource`]'s
/// contract already permits — the heap never required adjacency.  Exhaustion
/// is a typed refusal at the carve (offset past the supply span ⇒ the
/// session refuses ⇒ `AllocError`), never a wrap; the client cursor uses a
/// checked atomic reservation whose over-reserve on a refused carve is
/// terminal anyway.
///
/// A **zero-sized** handle — the supply is the endowment, not this value —
/// so a binary's `#[global_allocator] static A: Heap<HeapSupplySource>`
/// stays `const`.  It routes a miss straight to the kernel's session engine,
/// so it works for **mm itself** and breaks the bootstrap circularity — a
/// service grows its own heap before mm serves, and mm never re-enters its
/// own client grow path.  (An mm-served peer source arrives with the
/// client→mm vertical slice.)
#[derive(Clone, Copy, Debug, Default)]
pub struct HeapSupplySource;

// SAFETY: a ZST with no interior state; the grow cursor is a module static
// with atomic bumps, and the session engine serialises the actual carve+map
// kernel-side — `&HeapSupplySource` is trivially shareable.
unsafe impl Send for HeapSupplySource {}
unsafe impl Sync for HeapSupplySource {}

impl HeapSupplySource {
    /// A handle to the task's `role::HEAP` supply.  `const`, so a binary can
    /// back a `static` `Heap` with it.
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: `grow_from_supply` atomically reserves a disjoint supply offset and
// the two-invocation session maps that detached carve once.  The task retains
// the resulting mapping for its lifetime; later calls cannot reuse an offset.
unsafe impl MemSource for HeapSupplySource {
    fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
        grow_from_supply(min_bytes)
    }
}

/// This task's monotone offset into its heap supply — the client half of the
/// grow (which window of the supply the next carve names).  A refused carve
/// leaves the over-reserved span dead (terminal exhaustion — every later
/// grow refuses too).  Checked `try_update` preserves that monotone rule
/// without admitting arithmetic wrap back into a live offset.
#[cfg(target_os = "none")]
static HEAP_GROWN: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Bytes reserved off the **top** of the heap supply for code and data
/// windows a task carves itself (`reserve_from_top`) — the late loader's
/// arenas.  One supply, two monotone cursors: the allocator grows upward
/// from 0 behind [`HEAP_GROWN`], reservations grow downward from the
/// supply's end behind this one, and each side refuses the moment it
/// would cross the other's current mark.  A refused reservation, like a
/// refused grow, leaves its interval spent: the two cursors only ever
/// advance, so "spent" is what keeps them apart without a lock.
#[cfg(target_os = "none")]
static HEAP_RESERVED: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// The supply's total length, learned once from the kernel
/// (`lease_geometry`) the first time either cursor needs the far end.
/// Zero = not yet known.
#[cfg(target_os = "none")]
static HEAP_SUPPLY_LEN: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Reserve one disjoint supply interval without admitting wraparound.
///
/// A refused carve deliberately leaves its interval consumed (terminal supply
/// exhaustion), but arithmetic overflow is a refusal with no cursor mutation:
/// wrapping to an old offset would alias a live heap extent.
#[cfg(any(target_os = "none", test))]
fn reserve_supply(cursor: &core::sync::atomic::AtomicU64, want: u64) -> Option<u64> {
    cursor
        .try_update(Ordering::AcqRel, Ordering::Acquire, |offset| {
            offset.checked_add(want)
        })
        .ok()
}

/// The bottom-up grow's check against the top-down reservations: the
/// interval `[offset, offset + want)` may not reach into the reserved
/// tail.  `total == 0` means the far end is unknown, which can only be
/// the case while nothing has been reserved (a reservation needs the
/// total first), so an unknown total refuses nothing here — the kernel's
/// own supply bound still applies.
#[cfg(any(target_os = "none", test))]
fn grow_clears_reserved(offset: u64, want: u64, total: u64, reserved: u64) -> bool {
    if total == 0 {
        return true;
    }
    match (offset.checked_add(want), total.checked_sub(reserved)) {
        (Some(end), Some(limit)) => end <= limit,
        _ => false,
    }
}

/// The top-down reservation: claim `want` more bytes off the end of a
/// `total`-byte supply, refusing when the claim would cross the grow
/// cursor's current mark (`grown`).  The claim is made first and checked
/// second — a refused claim stays spent, exactly like a refused grow — so
/// the two cursors can never both succeed into the same bytes.  Returns
/// the offset of the reserved window.
#[cfg(any(target_os = "none", test))]
fn reserve_top(
    reserved: &core::sync::atomic::AtomicU64,
    grown: &core::sync::atomic::AtomicU64,
    total: u64,
    want: u64,
) -> Option<u64> {
    let end = reserve_supply(reserved, want)?.checked_add(want)?;
    if end > total {
        return None;
    }
    let offset = total - end;
    (offset >= grown.load(Ordering::Acquire)).then_some(offset)
}

/// Carve a page-granular sub-lease of `len` bytes off the **top** of this
/// task's `role::HEAP@0` supply — memory the task owns for a window it
/// maps itself (a late-linked module's code arena and data span,
/// `res/IMAGE.md` §8), kept apart from the allocator's extents by the
/// two-cursor rule on [`HEAP_RESERVED`].  The lease carries the ordinary
/// carve rights, so it can become a `JitArena` or an exact writable
/// mapping.  `None`: no supply, no geometry, the supply is spent, or the
/// reservation would meet the allocator's cursor.
#[cfg(target_os = "none")]
pub fn reserve_from_top(len: u64) -> Option<crate::cap::StashedLease> {
    let align = EXTENT_ALIGN as u64;
    let want = len.max(1).checked_add(align - 1)? & !(align - 1);
    let supply = crate::entry::geometry::heap_supply_slot()?;
    let total = supply_len(supply)?;
    let offset = reserve_top(&HEAP_RESERVED, &HEAP_GROWN, total, want)?;
    crate::cap::verbs::carve_owned_lease_window(supply, offset, want)
}

/// Off metal there is no supply to carve.
#[cfg(not(target_os = "none"))]
pub fn reserve_from_top(_len: u64) -> Option<crate::cap::StashedLease> {
    None
}

/// The supply's length, cached after one kernel query.
#[cfg(target_os = "none")]
fn supply_len(supply: kore_abi::session::SlotRef) -> Option<u64> {
    let known = HEAP_SUPPLY_LEN.load(Ordering::Acquire);
    if known != 0 {
        return Some(known);
    }
    let (_, len) = crate::cap::verbs::lease_geometry(supply)?;
    if len == 0 {
        return None;
    }
    // Two first-time queriers learn the same number; the store is idempotent.
    HEAP_SUPPLY_LEN.store(len, Ordering::Release);
    Some(len)
}

/// The metal grow (#336 rung 6): one session — carve `[grown, grown+want)`
/// off the `role::HEAP` supply, map the staged carve at the task's
/// map-window cursor, return the reported extent.  Every refusal (no supply
/// endowed, no report slot, exhausted span, a non-`Ok` commit) is `None` —
/// the heap surfaces `AllocError`, never an abort.
#[cfg(target_os = "none")]
fn grow_from_supply(min_bytes: usize) -> Option<NonNull<[u8]>> {
    let want = checked_align_up(min_bytes.max(1), EXTENT_ALIGN)?;
    let want = u64::try_from(want).ok()?;
    let supply = crate::entry::geometry::heap_supply_slot()?;
    let offset = reserve_supply(&HEAP_GROWN, want)?;
    // The reserved tail is the other cursor's; a grow that would enter it
    // refuses with its interval spent (the allocator surfaces AllocError).
    if !grow_clears_reserved(
        offset,
        want,
        HEAP_SUPPLY_LEN.load(Ordering::Acquire),
        HEAP_RESERVED.load(Ordering::Acquire),
    ) {
        return None;
    }
    carve_and_map(supply, offset, want)
}

/// **Carve `[offset, offset+len)` off a held supply and map the staged
/// carve** at the task's map-window cursor — one ordinary session of two
/// invocations (the mm-grant wire shape), the heap grow's primitive
/// generalized so any service can back an allocation from its **own**
/// endowed supply (swap's store arena is the first; #336 rung 6's
/// discipline: the carve is a detached `Derive × LeasedRegion`, the map
/// the staged flavor, the answer a bind-inert `GeomReport`).
///
/// `offset` is the caller-side monotone cursor into the supply; an
/// offset past the supply span refuses at the carve.  Every refusal (no
/// carve/report scratch seeded, a non-`Ok` commit, a short report) is
/// `None`, never an abort.
#[cfg(target_os = "none")]
pub fn carve_and_map(
    supply: kore_abi::session::SlotRef,
    offset: u64,
    len: u64,
) -> Option<NonNull<[u8]>> {
    use kore_abi::cap::{CapRecord, ResourceKind, Rights};
    use kore_abi::session::{GeomReport, Status};

    use crate::session::SessionBuilder;
    use crate::transport::{Transport, TrapTransport};

    let align = EXTENT_ALIGN as u64;
    let want = len.max(1).checked_add(align - 1)? & !(align - 1);
    let want_usize = usize::try_from(want).ok()?;
    if want_usize > isize::MAX as usize {
        return None;
    }
    let carve = crate::entry::geometry::carve_slot()?;
    let report = crate::entry::geometry::report_slot()?;

    // ι0: the detached window carve off the supply (the mm-grant wire shape —
    // `WindowCarveReq { offset, len, class: 0 }`).  DERIVE rides the request
    // so the staged child clears the map flavor's Σ5 gate.
    let mut req = [0u8; 24];
    req[0..8].copy_from_slice(&offset.to_le_bytes());
    req[8..16].copy_from_slice(&want.to_le_bytes());
    let mut b = SessionBuilder::new();
    b.derive_raw(
        supply,
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
        ResourceKind::LeasedRegion as u8,
        carve,
        Some(&req),
    )
    .ok()?;
    // ι1: the staged map — subject is ι0's result slot (Move D dataflow).
    b.derive_raw(carve, Rights::empty(), 0, report, None).ok()?;
    let committed = TrapTransport::new().commit_now(&b).ok()?;
    if committed.status() != Status::Ok {
        return None;
    }
    // Decode the report off the R/O slots alias.  SAFETY: the founder mapped
    // the alias R/O over this task's own records; the kernel wrote the report
    // before the trap returned.  Read volatile.
    let ptr = crate::entry::geometry::slot_record_va(report)? as *const CapRecord;
    let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
    if g.len < want || g.base == 0 || g.base % align != 0 || g.base.checked_add(want).is_none() {
        return None;
    }
    Some(NonNull::slice_from_raw_parts(
        NonNull::new(g.base as *mut u8)?,
        want_usize,
    ))
}

/// Off the metal userland targets — host builds: the honest refusal (the
/// heap surfaces `AllocError`), never a panic.
#[cfg(not(target_os = "none"))]
fn grow_from_supply(_min_bytes: usize) -> Option<NonNull<[u8]>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_cursors_refuse_each_other_and_stay_spent() {
        use core::sync::atomic::AtomicU64;
        let reserved = AtomicU64::new(0);
        let grown = AtomicU64::new(0);
        let total = 0x10_0000;
        // Top-down: two pages off the end.
        assert_eq!(reserve_top(&reserved, &grown, total, 0x2000), Some(total - 0x2000));
        // Bottom-up growth may use everything below the tail...
        assert!(grow_clears_reserved(0, total - 0x2000, total, 0x2000));
        // ...but not one byte into it.
        assert!(!grow_clears_reserved(0, total - 0x1fff, total, 0x2000));
        // An unknown total cannot have a tail, so it refuses nothing.
        assert!(grow_clears_reserved(0, total, 0, 0));
        // A reservation that would cross the grow mark refuses, and the
        // claim stays spent: the next reservation lands below it.
        grown.store(total - 0x3000, core::sync::atomic::Ordering::Release);
        assert_eq!(reserve_top(&reserved, &grown, total, 0x2000), None);
        assert_eq!(reserved.load(core::sync::atomic::Ordering::Acquire), 0x4000);
        // Overflow past the supply refuses without wrapping.
        assert_eq!(reserve_top(&reserved, &grown, total, u64::MAX), None);
    }

    /// A leaked host arena for a `StaticMemSource` — bytes valid for the whole
    /// test process, so the raw-pointer source never dangles.
    fn host_arena(bytes: usize) -> StaticMemSource {
        let buf: &'static mut [u8] = alloc::vec![0u8; bytes].leak();
        // SAFETY: `buf` is leaked, so it lives for the process; the source owns it.
        unsafe { StaticMemSource::over(buf.as_mut_ptr(), bytes) }
    }

    #[test]
    fn static_source_bumps_aligned_then_exhausts() {
        // Room for ~3 extents of one page after front alignment.
        let src = host_arena(EXTENT_ALIGN * 4);
        let a = src.map_extent(1).expect("first extent");
        let b = src.map_extent(EXTENT_ALIGN).expect("second extent");
        // Distinct, page-aligned, and at least the requested size.
        assert_ne!(a.as_ptr() as *mut u8, b.as_ptr() as *mut u8);
        assert_eq!(a.as_ptr() as *mut u8 as usize % EXTENT_ALIGN, 0);
        // map_extent rounds the request up to a whole page.
        assert!(a.len() >= EXTENT_ALIGN && b.len() >= EXTENT_ALIGN);
        // Exhaust the rest; eventually the tail can't fit a whole extent.
        let mut n = 0;
        while src.map_extent(EXTENT_ALIGN).is_some() {
            n += 1;
            assert!(n < 100, "a bounded arena must exhaust");
        }
    }

    #[test]
    fn supply_reservation_refuses_wrap_without_moving_the_cursor() {
        let cursor = core::sync::atomic::AtomicU64::new(u64::MAX - 1023);
        assert_eq!(reserve_supply(&cursor, 4096), None);
        assert_eq!(cursor.load(Ordering::Acquire), u64::MAX - 1023);
    }

    #[test]
    fn static_source_refuses_an_unrepresentable_round_up() {
        let src = host_arena(EXTENT_ALIGN * 2);
        assert!(src.map_extent(usize::MAX).is_none());
    }
}
