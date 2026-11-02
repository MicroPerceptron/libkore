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
//!   binary links: a miss carves the next slice of the task's own
//!   `role::HEAP` supply.  The task's own thread maps it in granules in the
//!   task's heap window (`geom::HEAP`), off the per-thread receiver-map
//!   cursor ([`window`]); a spawned thread, or a task founded without the
//!   window, maps each extent at its map cursor in one session, as before.

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

/// **One supply, two monotone cursors.**  The allocator claims extents and
/// granules upward from offset 0 behind `grown`; [`reserve_from_top`] claims
/// code and data windows a task maps itself (the late loader's arenas, a
/// Wasmtime engine's compiled images and guest memories) downward from the
/// supply's end behind `reserved`.  Every supply a task draws on — the launch
/// supply and each adopted one — is held apart by one of these, so the rule is
/// written once.
///
/// Each claim is made first and checked second: it advances its own cursor,
/// then refuses if its interval reaches past the other cursor's mark.  A claim
/// that loses that race stays spent — the cursors only ever advance, so
/// "spent" is what keeps them apart without a lock — and a refused carve after
/// a successful claim is terminal the same way.  A claim the unclaimed middle
/// visibly cannot hold is refused before it is made, and spends nothing: a
/// task tries a window against each of its supplies in turn (a guest memory
/// relocating to a larger one, `reserve_from_top`), and a spent misfit would
/// take the rest of that supply from both claimants.
///
/// **Why `SeqCst`.**  Claim-then-check across two words is the
/// store-buffering shape: each side writes its own cursor, then reads the
/// other's.  Acquire and release alone let both reads miss both writes, and
/// then both claims pass into the same bytes.  Every cursor access here is
/// `SeqCst`, whose single total order puts one claim first, so the later
/// claim's check sees it.  On x86-64 and AArch64 that costs nothing over
/// acquire-release (a locked RMW and a plain load; `LDAXR`/`STLXR` and
/// `LDAR`), and a claim is made once per granule, extent or window, never per
/// allocation.
#[cfg(any(target_os = "none", test))]
struct Cursors {
    /// Bytes claimed off the bottom so far: the allocator's.
    grown: core::sync::atomic::AtomicU64,
    /// Bytes claimed off the top so far: [`reserve_from_top`]'s.
    reserved: core::sync::atomic::AtomicU64,
}

#[cfg(any(target_os = "none", test))]
impl Cursors {
    const fn new() -> Self {
        Self {
            grown: core::sync::atomic::AtomicU64::new(0),
            reserved: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// What neither cursor has claimed yet of a `total`-byte supply — a
    /// snapshot for sizing a claim, which the claim itself then checks.
    fn room(&self, total: u64) -> Option<u64> {
        total
            .checked_sub(self.reserved.load(Ordering::SeqCst))?
            .checked_sub(self.grown.load(Ordering::SeqCst))
    }

    /// Claim `want` bytes off the bottom of a `total`-byte supply: the offset
    /// of `[offset, offset + want)`, refused when the interval reaches into
    /// the reserved top.
    fn claim_bottom(&self, total: u64, want: u64) -> Option<u64> {
        if self.room(total)? < want {
            return None;
        }
        let offset = claim(&self.grown, want)?;
        let limit = total.checked_sub(self.reserved.load(Ordering::SeqCst))?;
        (offset.checked_add(want)? <= limit).then_some(offset)
    }

    /// Claim `want` bytes off the top of a `total`-byte supply: the offset of
    /// the window, refused when it reaches below the grow mark.
    fn claim_top(&self, total: u64, want: u64) -> Option<u64> {
        if self.room(total)? < want {
            return None;
        }
        let end = claim(&self.reserved, want)?.checked_add(want)?;
        let offset = total.checked_sub(end)?;
        (offset >= self.grown.load(Ordering::SeqCst)).then_some(offset)
    }
}

/// Advance a monotone cursor by `want`: where the claim starts.  A refused
/// carve deliberately leaves its interval consumed (terminal supply
/// exhaustion), but arithmetic overflow is a refusal with no cursor
/// mutation: wrapping to an old offset would alias a live heap extent.
#[cfg(any(target_os = "none", test))]
fn claim(cursor: &core::sync::atomic::AtomicU64, want: u64) -> Option<u64> {
    cursor
        .try_update(Ordering::SeqCst, Ordering::SeqCst, |offset| {
            offset.checked_add(want)
        })
        .ok()
}

/// The launch supply's cursors (`role::HEAP@0`): the per-extent grow and the
/// heap window's launch granules claim its bottom, [`reserve_from_top`] its
/// top once every adopted supply has refused.
#[cfg(target_os = "none")]
static LAUNCH: Cursors = Cursors::new();

/// The launch supply's total length, learned once from the kernel
/// (`lease_geometry`) the first time a claim needs the far end.  Zero = not
/// yet known.
#[cfg(target_os = "none")]
static HEAP_SUPPLY_LEN: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Carve a page-granular sub-lease of `len` bytes off the **top** of a
/// supply this task holds — memory the task owns for a window it maps
/// itself (a late-linked module's code arena and data span, `res/IMAGE.md`
/// §8; a Wasmtime engine's every compiled image and guest memory) — kept
/// apart from the allocator's extents by the two-cursor rule ([`Cursors`]).
/// Supplies adopted for windows alone ([`adopt_window_supply`]) are tried
/// first, then those adopted for the heap ([`adopt_supply`]), each in
/// adoption order; the launch supply `role::HEAP@0` last.  A window thus
/// lands where its program reserved room for windows, and only then in a
/// heap reservation — which a program sized for its job's windows too, so
/// they land there rather than in the launch supply the job's heap growth
/// would otherwise have left alone.  A supply too short for the window is
/// passed over intact ([`Cursors`]).  The lease carries the ordinary carve rights, so it can
/// become a `JitArena` or an exact writable mapping.  `None`: no supply,
/// no geometry, or every supply is spent or would meet its allocator
/// cursor.
#[cfg(target_os = "none")]
pub fn reserve_from_top(len: u64) -> Option<crate::cap::StashedLease> {
    let align = EXTENT_ALIGN as u64;
    let want = len.max(1).checked_add(align - 1)? & !(align - 1);
    window_supplies().find_map(|(_, supply, total, cursors)| {
        let offset = cursors.claim_top(total, want)?;
        crate::cap::verbs::carve_owned_lease_window(supply, offset, want)
    })
}

/// The supplies a window this task maps itself draws on, in
/// [`reserve_from_top`]'s order — those adopted for windows alone, then those
/// adopted for the heap, each in adoption order, then the launch supply —
/// each with its slot, its length and its cursors.  Lazy: the launch
/// supply's length is learned only once every adopted supply is passed over.
#[cfg(target_os = "none")]
fn window_supplies() -> impl Iterator<
    Item = (
        CommitSupply,
        kore_abi::session::SlotRef,
        u64,
        &'static Cursors,
    ),
> {
    let adopted = |heap: bool| {
        ADOPTED.iter().enumerate().filter_map(move |(i, entry)| {
            let slot = entry.slot.load(Ordering::Acquire).checked_sub(1)?;
            (entry.heap.load(Ordering::Relaxed) == heap).then(|| {
                (
                    CommitSupply(i as u8),
                    kore_abi::session::SlotRef::from_wire(slot),
                    entry.len.load(Ordering::Relaxed),
                    &entry.cursors,
                )
            })
        })
    };
    let launch = core::iter::once_with(|| {
        let supply = crate::entry::geometry::heap_supply_slot()?;
        Some((CommitSupply::LAUNCH, supply, supply_len(supply)?, &LAUNCH))
    })
    .flatten();
    adopted(false).chain(adopted(true)).chain(launch)
}

/// **Where a reservation commit drew its pages** ([`commit_reserved`]): one
/// of this task's supplies — an adoption entry, or the launch supply — and
/// what [`release_reserved`] names to unmap them, since a release removes
/// only the named lessor's commits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CommitSupply(u8);

impl CommitSupply {
    /// How many supplies a commit may draw on: every adoption entry and the
    /// launch supply — the width of a set of them, which [`index`](Self::index)
    /// numbers densely.
    pub(crate) const COUNT: usize = MAX_ADOPTED + 1;

    /// The launch supply (`role::HEAP@0`), numbered after every adoption
    /// entry.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    const LAUNCH: Self = Self(MAX_ADOPTED as u8);

    /// This supply's number, `< COUNT`.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    /// The supply numbered `index`, for `index < COUNT`.
    pub(crate) const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self(index as u8))
        } else {
            None
        }
    }
}

// Every supply's number fits the byte it is carried in.
const _: () = assert!(CommitSupply::COUNT <= u8::MAX as usize + 1);

/// **Commit `len` bytes at page `page` of this task's reservation window**
/// (`geom::RESERVE`, `res/WASM_PLATFORM.md` §12) — the body of
/// [`Reservation::commit`](crate::reserve::Reservation::commit), which owns
/// the range and has proven the span uncommitted.  The bytes are carved off
/// the top of a supply this task holds, [`reserve_from_top`]'s supplies in
/// its order under the same two-cursor rule, and the carve is mapped at
/// exactly that page ([`map_view_reserved`]) in one session that holds
/// nothing afterwards: [`carve_and_map_in`]'s shape, so a memory grown in
/// hundreds of steps costs hundreds of sessions and no capability.  `page`
/// counts [`RESERVE_PAGE_BYTES`] from the window's base; `len` is a whole
/// number of them.
///
/// The answer names the supply the pages came from: what
/// [`release_reserved`] must name to unmap them.  A map refused partway
/// leaves the prefix it published, so a refused commit releases its own
/// range from the supply it tried before it answers — exact, because the
/// caller proved no earlier commit lies there — and leaves nothing mapped;
/// the carved bytes stay spent, as every refused carve's do.
///
/// **The task's own thread only**, the heap window's refill rule: the
/// supplies are named by slots of its arena, which on a spawned thread name
/// something else or nothing.  Elsewhere this refuses before any claim.
///
/// [`map_view_reserved`]: crate::session::SessionBuilder::map_view_reserved
/// [`RESERVE_PAGE_BYTES`]: kore_abi::session::RESERVE_PAGE_BYTES
#[cfg(target_os = "none")]
pub(crate) fn commit_reserved(
    page: u32,
    len: u64,
) -> Result<CommitSupply, crate::reserve::CommitError> {
    use crate::reserve::CommitError;
    use kore_abi::session::RESERVE_PAGE_BYTES;

    if len == 0 || len % RESERVE_PAGE_BYTES != 0 {
        return Err(CommitError::OutOfRange);
    }
    if crate::entry::geometry::shard_index() != Some(0) {
        return Err(CommitError::Refused);
    }
    let pages = u32::try_from(len / RESERVE_PAGE_BYTES).map_err(|_| CommitError::OutOfRange)?;
    let (id, supply, offset) = window_supplies()
        .find_map(|(id, supply, total, cursors)| Some((id, supply, cursors.claim_top(total, len)?)))
        .ok_or(CommitError::NoRoom)?;
    if carve_and_map_at(supply, offset, len, Placement::ReservePage(page)).is_some() {
        return Ok(id);
    }
    let _ = release_reserved(id, page, pages);
    Err(CommitError::Refused)
}

/// Off metal there is no window to commit into.
#[cfg(not(target_os = "none"))]
pub(crate) fn commit_reserved(
    _page: u32,
    _len: u64,
) -> Result<CommitSupply, crate::reserve::CommitError> {
    Err(crate::reserve::CommitError::Refused)
}

/// **Release `[page, page + pages)` of this task's reservation window**: every
/// commit [`commit_reserved`] drew from `supply` inside the range is unmapped
/// (`DERIVE_KIND_RESERVE_RELEASE`), in one session.  Commits from another
/// supply stay — a reservation grown from several releases once per supply
/// it drew on.  `false` off the task's own thread ([`commit_reserved`]'s
/// rule), when the supply is gone, the range leaves the window, or a commit
/// straddles the range's edge (and then nothing was unmapped).
#[cfg(target_os = "none")]
pub(crate) fn release_reserved(supply: CommitSupply, page: u32, pages: u32) -> bool {
    use crate::session::SessionBuilder;
    use crate::transport::{Transport, TrapTransport};

    if crate::entry::geometry::shard_index() != Some(0) {
        return false;
    }
    let slot = if supply == CommitSupply::LAUNCH {
        crate::entry::geometry::heap_supply_slot()
    } else {
        ADOPTED
            .get(supply.index())
            .and_then(|entry| entry.slot.load(Ordering::Acquire).checked_sub(1))
            .map(kore_abi::session::SlotRef::from_wire)
    };
    let Some(slot) = slot else {
        return false;
    };
    let mut b = SessionBuilder::new();
    b.release_reserved(slot, page, pages).is_ok()
        && TrapTransport::new()
            .commit_now(&b)
            .is_ok_and(|done| done.status() == kore_abi::session::Status::Ok)
}

/// Off metal there is nothing to release.
#[cfg(not(target_os = "none"))]
pub(crate) fn release_reserved(_supply: CommitSupply, _page: u32, _pages: u32) -> bool {
    false
}

/// Off metal there is no supply to carve.
#[cfg(not(target_os = "none"))]
pub fn reserve_from_top(_len: u64) -> Option<crate::cap::StashedLease> {
    None
}

/// **The longest window a supply held for windows could take now**: the
/// most room any one supply adopted for windows alone
/// ([`adopt_window_supply`]) has left between its two cursors, in whole
/// extents.  A window is one carve, so it must fit in one supply — the sum
/// of the rooms is no answer.  [`reserve_from_top`] and a reservation's
/// commit try those supplies before any other, so a window no longer than
/// this lands in one, and the supplies the heap draws on keep the room
/// their program reserved for work that cannot wait on mm (a frame's
/// document mutations).  A snapshot for deciding whether to reserve a
/// window supply before a window is needed (a guest memory about to grow);
/// the carve itself still checks.
#[cfg(target_os = "none")]
pub fn window_supply_room() -> u64 {
    let rooms = ADOPTED.iter().filter_map(|entry| {
        entry.slot.load(Ordering::Acquire).checked_sub(1)?;
        // Published with the slot, as the length is.
        if entry.heap.load(Ordering::Relaxed) {
            return None;
        }
        entry.cursors.room(entry.len.load(Ordering::Relaxed))
    });
    whole_extents(rooms.max().unwrap_or(0))
}

/// Off metal there is no supply.
#[cfg(not(target_os = "none"))]
pub fn window_supply_room() -> u64 {
    0
}

/// **The room heap granules can still draw on**: what the heap can still
/// grow by off the supplies this task adopted for it ([`adopt_supply`]),
/// without a new one — their room in whole 2 MiB granules, and the heap
/// window's free bytes a granule placed next extends. Any allocation, or run
/// of allocations, the heap must map fresh up to this size finds room: one
/// longer than any one supply's room is assembled from several in the heap
/// window. A snapshot for deciding whether to reserve more before a burst of
/// synchronous allocation needs it (a frame's document mutations, a photo's
/// decode — `kore-svc-mm`'s `ProcessMm::reserve_heap`), which cannot wait on
/// mm; the carve itself still checks. A lower bound, never an over-promise:
/// the launch supply is not counted — [`reserve_from_top`]'s windows share
/// it, and the heap never assembles a run from it — nor are supplies held
/// for one window ([`adopt_window_supply`]), nor free blocks the heap
/// already holds, which the burst may reuse.
#[cfg(target_os = "none")]
pub fn heap_room() -> u64 {
    window::room()
}

/// Off metal there is no supply.
#[cfg(not(target_os = "none"))]
pub fn heap_room() -> u64 {
    0
}

/// `bytes` down to whole extents: room a granule can use.
#[cfg(target_os = "none")]
const fn whole_extents(bytes: u64) -> u64 {
    bytes & !(EXTENT_ALIGN as u64 - 1)
}

/// A supply a program **adopted** ([`adopt_supply`], [`adopt_window_supply`]):
/// a root lease it obtained itself — an mm supply minted on its own
/// per-launch face — drawn on before the launch supply: heap granules from
/// its bottom (unless it was adopted for windows alone), [`reserve_from_top`]'s
/// windows from its top.
#[cfg(target_os = "none")]
struct Adopted {
    /// The lease's slot index + 1; zero marks a vacant entry.
    slot: core::sync::atomic::AtomicU32,
    /// The lease's length.
    len: core::sync::atomic::AtomicU64,
    /// Whether heap granules may draw on it: false for a supply obtained for
    /// one window, which a granule mapped first would leave too short.
    heap: core::sync::atomic::AtomicBool,
    /// Its two-cursor rule, the launch supply's own ([`Cursors`]).
    cursors: Cursors,
}

/// Adopted supplies one task may hold — design-bounded: a reservation is
/// adopted as leases of `kore-svc-mm`'s `HEAP_RESERVATION_LEASE_BYTES`
/// (128 MiB) down to its `HEAP_RESERVATION_LEASE_MIN_BYTES` (16 MiB) when
/// the class is fragmented, so sixty-four hold the `wasm` command's 1 GiB
/// ceiling at the shortest lease; a program adopts one reservation per
/// large synchronous job it prepares for, and one window per guest memory
/// relocation that finds no supply with room (CPython's 104 MiB workload
/// relocates five times), never one per allocation.  Two KiB of
/// every image's `.bss`.
pub const MAX_ADOPTED: usize = 64;

#[cfg(target_os = "none")]
static ADOPTED: [Adopted; MAX_ADOPTED] = [const {
    Adopted {
        slot: core::sync::atomic::AtomicU32::new(0),
        len: core::sync::atomic::AtomicU64::new(0),
        heap: core::sync::atomic::AtomicBool::new(false),
        cursors: Cursors::new(),
    }
}; MAX_ADOPTED];

/// **Give this task's heap more room**: adopt `lease` — a root supply the
/// program obtained itself, typically from mm on its per-launch face — as
/// backing for the heap window, drawn on **before** the launch supply, which
/// stays for the code and data windows `reserve_from_top` carves.
///
/// This is how a program prepares for a large *synchronous* job whose
/// allocations cannot wait on a service: the global allocator never talks to
/// mm (`res/USERLAND.md` §2.2), so the program asks mm on its executor first,
/// then adopts the grant. The lease is held for the task's life — its
/// granules back live heap memory, which is never returned.
///
/// Refused (the lease back) off the task's own thread (only it refills the
/// heap window), when the task has no heap window, when the lease's geometry
/// is unreadable, or when every adoption entry is taken.
#[cfg(target_os = "none")]
pub fn adopt_supply(lease: crate::cap::StashedLease) -> Result<(), crate::cap::StashedLease> {
    adopt(lease, true)
}

/// **Hold `lease` for one window the task will map itself** — a code image,
/// a guest memory ([`reserve_from_top`]) — that no supply it holds has room
/// for. Adopted as [`adopt_supply`] adopts, refused for the same reasons,
/// but heap granules never draw on it: a granule mapped into its bottom
/// before the window is carved would leave it too short for the window it
/// was obtained for. What the window leaves stays for later windows.
#[cfg(target_os = "none")]
pub fn adopt_window_supply(
    lease: crate::cap::StashedLease,
) -> Result<(), crate::cap::StashedLease> {
    adopt(lease, false)
}

/// [`adopt_supply`] and [`adopt_window_supply`]: `heap` says whether heap
/// granules may draw on the lease.
#[cfg(target_os = "none")]
fn adopt(lease: crate::cap::StashedLease, heap: bool) -> Result<(), crate::cap::StashedLease> {
    let has_window = crate::entry::thread_context()
        .and_then(|rt| rt.geom())
        .and_then(|g| g.window(kore_abi::task::geom::HEAP))
        .is_some();
    let len = lease
        .slot()
        .and_then(crate::cap::verbs::lease_geometry)
        .map(|(_, len)| len)
        .filter(|&len| len != 0);
    let (Some(slot), Some(len), Some(0), true) = (
        lease.slot(),
        len,
        crate::entry::geometry::shard_index(),
        has_window,
    ) else {
        return Err(lease);
    };
    let Some(entry) = ADOPTED
        .iter()
        .find(|entry| entry.slot.load(Ordering::Acquire) == 0)
    else {
        return Err(lease);
    };
    // Its cursors are still at zero: an entry is never vacated, and both
    // claimants pass over a vacant one.  The slot's release publishes the
    // length and the heap flag with it.
    entry.len.store(len, Ordering::Relaxed);
    entry.heap.store(heap, Ordering::Relaxed);
    entry.slot.store(slot.index() + 1, Ordering::Release);
    // Held for the task's life: its granules back live heap memory.
    core::mem::forget(lease);
    Ok(())
}

/// Off metal there is no heap window to back.
#[cfg(not(target_os = "none"))]
pub fn adopt_supply(lease: crate::cap::StashedLease) -> Result<(), crate::cap::StashedLease> {
    Err(lease)
}

/// Off metal there is no window to map.
#[cfg(not(target_os = "none"))]
pub fn adopt_window_supply(
    lease: crate::cap::StashedLease,
) -> Result<(), crate::cap::StashedLease> {
    Err(lease)
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
    match window::grow(want) {
        window::Grow::Extent(extent) => return Some(extent),
        window::Grow::Refused => return None,
        window::Grow::NoWindow => {}
    }
    let supply = crate::entry::geometry::heap_supply_slot()?;
    // The far end is learned before the claim, not read from the cache: a
    // grow that took an unlearned length as "nothing reserved" could pass
    // its check against a reservation another thread had just claimed.
    let total = supply_len(supply)?;
    // The reserved top is the other cursor's; a grow that would enter it
    // refuses with its interval spent (the allocator surfaces AllocError).
    let offset = LAUNCH.claim_bottom(total, want)?;
    carve_and_map(supply, offset, want)
}

/// **The process heap window** (`geom::HEAP`, GEOM_ABI v7): the task's heap
/// maps what it carves off its own `role::HEAP` supply in a founder-stamped
/// reservation of its own, instead of at the thread's receiver-map cursor.
///
/// That cursor's 16 MiB stride is shared by every view on the thread and never
/// reused, so a heap could not map more than 16 MiB however large its supply
/// (a compiler maps 234 extents, 17.7 MiB, compiling one stock Rust program).
/// Here the supply is taken in **granules**, each one session of
/// [`carve_and_map_in`] staged into [`ViewMapWindow::Heap`]: the same scratch
/// slots and bind-inert report as the per-extent grow, so growth still holds
/// no capability and reserves no arena slot — a grow may run at any moment,
/// including between a slot index being chosen for a delivery and the
/// delivery landing.
///
/// **One free interval, every thread.** `[at, end)` is packed into one atomic
/// word as page offsets from the window base (a 1 TiB window is 2^28 pages).
/// Any thread bumps extents off its front by compare-and-swap and holds
/// nothing — the mapping is the task's, so a worker shares the window's memory
/// without carving. A refill extends the interval when the kernel places the
/// granule at its end, and replaces it otherwise.
///
/// **Only the task's own thread refills** (shard 0): the window is stamped
/// for the founded task, not its threads, and the kernel's heap cursor is the
/// task's. A worker finding the interval short takes the per-extent path on
/// its own stride, as does a task founded without the window.
///
/// **Never greedy.** The same supply feeds worker growth and
/// [`reserve_from_top`]'s code windows, so a refill takes what is missing,
/// rounded up to at most `min(doubling, remaining / 4)` ([`GRANULE_MIN`] up to
/// [`GRANULE_MAX`]): each refill leaves three quarters of what remains to the
/// supply's other consumers, and a large heap still costs a handful of
/// sessions.
///
/// **One extent, several supplies.** A reservation arrives as leases of at
/// most 128 MiB (`kore-svc-mm`'s `HEAP_RESERVATION_LEASE_BYTES`), each
/// claimed from the bottom as the heap grows, so its room ends up spread
/// over the tails of several. An extent no one supply holds — a 100 MP
/// photo's 400 MB of pixels, or a 65 MB one once the leases are part spent
/// — is assembled in the window instead: each **part** is the whole room of
/// the adopted supply that has the most, in whole granules, which the
/// kernel places end to end ([`Window::room`] says when), until one supply
/// holds what is still missing. A run the free interval and the adopted
/// supplies cannot hold together is refused before any part is mapped.
///
/// The policy is [`Window`]'s, written once over the [`Granules`] seam: the
/// task's supplies and the kernel on metal, a simulated kernel in the host
/// tests.
///
/// [`ViewMapWindow::Heap`]: kore_abi::session::ViewMapWindow::Heap
#[cfg(any(target_os = "none", test))]
mod window {
    use core::ptr::NonNull;
    use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    use super::Cursors;

    /// The doubling's first step: 2 MiB — the frame class the physical
    /// allocator reserves for huge leaves (`res/ACCEL_MEMORY_SCALE.md` §1),
    /// and the boundary the kernel places a view of at least that on.
    const GRANULE_MIN: u64 = 2 << 20;
    /// Where doubling stops: four doublings reach it.
    const GRANULE_MAX: u64 = 32 << 20;
    /// Doublings from [`GRANULE_MIN`] to [`GRANULE_MAX`].
    const DOUBLINGS: u32 = GRANULE_MAX.ilog2() - GRANULE_MIN.ilog2();
    /// A refill may take at most this fraction of the supply that remains.
    const REFILL_SHARE: u64 = 4;
    /// Interval offsets count pages of this size.
    const PAGE: u64 = 4096;
    /// The most refills one grow makes: a part off each adopted supply, the
    /// granule that completes the run, and one more for a worker's bump that
    /// raced it.
    const REFILLS: usize = super::MAX_ADOPTED + 2;

    const fn pack(at: u64, end: u64) -> u64 {
        (end << 32) | at
    }
    const fn unpack(word: u64) -> (u64, u64) {
        (word & 0xffff_ffff, word >> 32)
    }

    /// What the window takes granules from, and how one is placed: the
    /// task's supplies and the kernel on metal ([`Supplies`]), a simulated
    /// kernel in the host tests. Which supply, how much and when the free
    /// interval extends are [`Window`]'s alone.
    pub(super) trait Granules {
        /// A supply, as [`map`](Self::map) names it.
        type Supply: Copy;
        /// The adopted supplies heap granules may draw on, in adoption
        /// order: each one, its length and its cursors.
        fn adopted(&self) -> impl Iterator<Item = (Self::Supply, u64, &Cursors)>;
        /// The launch supply (`role::HEAP@0`), when the task has one.
        fn launch(&self) -> Option<(Self::Supply, u64, &Cursors)>;
        /// Map the claimed `[offset, offset + size)` of `supply` in the
        /// window: the address it landed at.
        fn map(&self, supply: Self::Supply, offset: u64, size: u64) -> Option<u64>;
    }

    /// The window's state: its base and the one free interval.
    pub(super) struct Window {
        /// The window base, published before any interval is non-empty (zero
        /// = not learned yet).
        base: AtomicU64,
        /// The free interval: bits 0..32 its start, bits 32..64 its end, both
        /// in pages from `base`.
        free: AtomicU64,
        /// Granules mapped so far: the doubling's exponent.
        granules: AtomicU32,
    }

    impl Window {
        pub(super) const fn new() -> Self {
            Self {
                base: AtomicU64::new(0),
                free: AtomicU64::new(0),
                granules: AtomicU32::new(0),
            }
        }

        /// The window base, zero until learned.
        pub(super) fn base(&self) -> u64 {
            self.base.load(Ordering::Acquire)
        }

        /// Publish the window base, before the first refill.
        pub(super) fn learn_base(&self, base: u64) {
            self.base.store(base, Ordering::Release);
        }

        /// Take `want` bytes off the front of the free interval, lock-free:
        /// the compare-and-swap retries only when another thread changed the
        /// interval first.
        pub(super) fn bump(&self, want: u64) -> Option<NonNull<[u8]>> {
            let pages = want / PAGE;
            let mut word = self.free.load(Ordering::Acquire);
            loop {
                let (at, end) = unpack(word);
                let next = at.checked_add(pages).filter(|&next| next <= end)?;
                match self.free.compare_exchange_weak(
                    word,
                    pack(next, end),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        let va = self.base().checked_add(at * PAGE)?;
                        return Some(NonNull::slice_from_raw_parts(
                            NonNull::new(va as *mut u8)?,
                            usize::try_from(want).ok()?,
                        ));
                    }
                    Err(actual) => word = actual,
                }
            }
        }

        /// Take `want` bytes, refilling off `supplies` until the free
        /// interval holds them. `None` when the supplies cannot: a run is
        /// assembled from parts only when they hold it together, and a
        /// refill the kernel refuses ends the grow.
        pub(super) fn fill<G: Granules>(&self, supplies: &G, want: u64) -> Option<NonNull<[u8]>> {
            // A worker may bump the fresh bytes away between a refill and
            // this thread's own bump; one more refill answers that race, and
            // a second loss is reported rather than chased.
            for _ in 0..REFILLS {
                if let Some(extent) = self.bump(want) {
                    return Some(extent);
                }
                self.refill(supplies, want)?;
            }
            self.bump(want)
        }

        /// **What the heap can still map** off the adopted supplies without
        /// a new one: every adopted supply's room in whole granules, and the
        /// free interval when a granule placed next extends it. Any extent,
        /// or run of extents, up to this maps — one longer than any one
        /// supply's room in parts ([`fill`](Self::fill)).
        ///
        /// The kernel places a granule of [`GRANULE_MIN`] or more on a
        /// [`GRANULE_MIN`] boundary, so one lands at the interval's end only
        /// when that end is on one; a free interval ending elsewhere (a
        /// page-grain granule took a supply's last pages) serves the extents
        /// it holds alone, and a longer one replaces it. The launch supply is
        /// left out: [`reserve_from_top`](super::reserve_from_top)'s windows
        /// share it, and a refill takes at most its share.
        pub(super) fn room<G: Granules>(&self, supplies: &G) -> u64 {
            supplies
                .adopted()
                .filter_map(|(_, total, cursors)| cursors.room(total))
                .fold(self.extendable(), |room, part| {
                    room.saturating_add(part & !(GRANULE_MIN - 1))
                })
        }

        /// The free interval's bytes a granule placed next extends: all of
        /// them when its end lies on a [`GRANULE_MIN`] boundary, else none.
        fn extendable(&self) -> u64 {
            let (at, end) = unpack(self.free.load(Ordering::Acquire));
            let on_boundary = self
                .base()
                .checked_add(end * PAGE)
                .is_some_and(|va| va % GRANULE_MIN == 0);
            if on_boundary { (end - at) * PAGE } else { 0 }
        }

        /// Carve the next granule off a supply and have it mapped in the
        /// window, then publish it as free: the whole of what is missing from
        /// the first supply that holds it, else a part of the run.
        fn refill<G: Granules>(&self, supplies: &G, want: u64) -> Option<()> {
            // What the granule must bring: what the free interval lacks. The
            // kernel places a granule under `GRANULE_MIN` at page grain, at
            // the interval's very end, so the whole free tail counts; one it
            // places on a boundary past a gap replaces the interval, and the
            // next refill counts from it.
            let (at, end) = unpack(self.free.load(Ordering::Acquire));
            let missing = want.saturating_sub((end - at) * PAGE);
            let doubling = GRANULE_MIN << self.granules.load(Ordering::Relaxed).min(DOUBLINGS);
            // Adopted supplies first: they were obtained for the heap alone,
            // while the launch supply is shared with `reserve_from_top`'s code
            // windows (a Wasmtime engine maps every compiled module there).
            let (supply, offset, size) = adopted_granule(supplies, missing, doubling)
                .or_else(|| launch_granule(supplies, missing, doubling))
                .or_else(|| self.part(supplies, want))?;
            let va = supplies.map(supply, offset, size)?;
            self.granules.fetch_add(1, Ordering::Relaxed);
            let start = va.checked_sub(self.base())? / PAGE;
            let stop = start
                .checked_add(size / PAGE)
                .filter(|&stop| stop <= u32::MAX as u64)?;
            // Extend the interval when the granule lands at its end; otherwise
            // the new granule replaces it and the old tail is left unused.
            let mut word = self.free.load(Ordering::Acquire);
            loop {
                let (at, end) = unpack(word);
                let next = if start == end {
                    pack(at, stop)
                } else {
                    pack(start, stop)
                };
                match self
                    .free
                    .compare_exchange_weak(word, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => return Some(()),
                    Err(actual) => word = actual,
                }
            }
        }

        /// One part of a run no one supply holds whole: all the room of the
        /// adopted supply that has the most (the first, of equals), in whole
        /// granules — reserved, and only when the free interval and the
        /// adopted supplies hold `want` together ([`room`](Self::room)), so a
        /// run that cannot be assembled spends nothing. The launch supply is
        /// never taken whole for a part.
        fn part<G: Granules>(&self, supplies: &G, want: u64) -> Option<(G::Supply, u64, u64)> {
            if self.room(supplies) < want {
                return None;
            }
            let (supply, total, cursors, size) = supplies
                .adopted()
                .filter_map(|(supply, total, cursors)| {
                    let size = cursors.room(total)? & !(GRANULE_MIN - 1);
                    (size != 0).then_some((supply, total, cursors, size))
                })
                .reduce(|best, next| if next.3 > best.3 { next } else { best })?;
            let offset = cursors.claim_bottom(total, size)?;
            Some((supply, offset, size))
        }
    }

    /// How big the next granule is: what is missing, rounded up to
    /// `min(doubling, share)`, in 2 MiB multiples when that stays inside the
    /// share (see [`launch_granule`]), else in pages; never past `room`.
    fn granule_size(missing: u64, doubling: u64, room: u64, share: u64) -> Option<u64> {
        let want = missing.max(doubling.min(share));
        [GRANULE_MIN, PAGE]
            .into_iter()
            .filter_map(|grain| Some(want.checked_add(grain - 1)? & !(grain - 1)))
            .find(|&size| size <= room && (size <= want.max(share) || size - want < PAGE))
    }

    /// The next granule off the launch supply (`role::HEAP@0`), reserved:
    /// `(supply, offset, size)`.
    fn launch_granule<G: Granules>(
        supplies: &G,
        missing: u64,
        doubling: u64,
    ) -> Option<(G::Supply, u64, u64)> {
        let (supply, total, cursors) = supplies.launch()?;
        let room = cursors.room(total)?;
        // The kernel places a view of at least 2 MiB on a 2 MiB boundary (the
        // huge-leaf alignment), so whole multiples of it land end to end and
        // the interval extends. That rounding is taken only inside the share:
        // on the ordinary 2 MiB supply it would otherwise swallow the whole
        // supply on the first refill and starve `reserve_from_top` (Lua's
        // native `require` then has no JIT arena). Page grain is the rest.
        let size = granule_size(missing, doubling, room, room / REFILL_SHARE)?;
        let offset = cursors.claim_bottom(total, size)?;
        Some((supply, offset, size))
    }

    /// The next granule off an adopted supply ([`super::adopt_supply`]),
    /// reserved, from the first whose unclaimed middle holds it — a supply
    /// adopted for one window ([`super::adopt_window_supply`]) is passed
    /// over. Its top serves [`super::reserve_from_top`]'s windows, but both
    /// claimants were sized into the lease by the program that obtained it
    /// for this job, so a granule may take what neither has claimed — the
    /// doubling still bounds it — rather than the launch supply's quarter.
    fn adopted_granule<G: Granules>(
        supplies: &G,
        missing: u64,
        doubling: u64,
    ) -> Option<(G::Supply, u64, u64)> {
        supplies.adopted().find_map(|(supply, total, cursors)| {
            let room = cursors.room(total)?;
            let size = granule_size(missing, doubling, room, room)?;
            let offset = cursors.claim_bottom(total, size)?;
            Some((supply, offset, size))
        })
    }

    /// The task's own window.
    #[cfg(target_os = "none")]
    static WINDOW: Window = Window::new();

    /// The task's supplies and the kernel's heap window: the [`Granules`]
    /// [`WINDOW`] grows over on metal.
    #[cfg(target_os = "none")]
    struct Supplies;

    #[cfg(target_os = "none")]
    impl Granules for Supplies {
        type Supply = kore_abi::session::SlotRef;

        fn adopted(&self) -> impl Iterator<Item = (Self::Supply, u64, &Cursors)> {
            super::ADOPTED.iter().filter_map(|entry| {
                let slot = entry.slot.load(Ordering::Acquire).checked_sub(1)?;
                entry.heap.load(Ordering::Relaxed).then(|| {
                    (
                        kore_abi::session::SlotRef::from_wire(slot),
                        entry.len.load(Ordering::Relaxed),
                        &entry.cursors,
                    )
                })
            })
        }

        fn launch(&self) -> Option<(Self::Supply, u64, &Cursors)> {
            let supply = crate::entry::geometry::heap_supply_slot()?;
            Some((supply, super::supply_len(supply)?, &super::LAUNCH))
        }

        fn map(&self, supply: Self::Supply, offset: u64, size: u64) -> Option<u64> {
            let granule = super::carve_and_map_in(
                supply,
                offset,
                size,
                Some(kore_abi::session::ViewMapWindow::Heap),
            )?;
            Some(granule.cast::<u8>().as_ptr() as u64)
        }
    }

    #[cfg(target_os = "none")]
    pub(super) enum Grow {
        Extent(NonNull<[u8]>),
        /// The window exists and could not supply `want`: the heap surfaces
        /// `AllocError`, exactly as the per-extent path would.
        Refused,
        /// Not from the window: this thread cannot refill it, or the task has
        /// none. Use the per-extent path.
        NoWindow,
    }

    #[cfg(target_os = "none")]
    pub(super) fn grow(want: u64) -> Grow {
        if let Some(extent) = WINDOW.bump(want) {
            return Grow::Extent(extent);
        }
        if crate::entry::geometry::shard_index() != Some(0) {
            return Grow::NoWindow;
        }
        if WINDOW.base() == 0 {
            let Some((base, _)) = crate::entry::thread_context()
                .and_then(|rt| rt.geom())
                .and_then(|g| g.window(kore_abi::task::geom::HEAP))
            else {
                return Grow::NoWindow;
            };
            WINDOW.learn_base(base);
        }
        match WINDOW.fill(&Supplies, want) {
            Some(extent) => Grow::Extent(extent),
            None => Grow::Refused,
        }
    }

    /// [`Window::room`] of the task's own window.
    #[cfg(target_os = "none")]
    pub(super) fn room() -> u64 {
        WINDOW.room(&Supplies)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use core::cell::RefCell;
        use std::vec::Vec;

        const KIB: u64 = 1 << 10;
        const MIB: u64 = 1 << 20;
        /// Where the simulated window starts: on a 2 MiB boundary, as the
        /// founder stamps it.
        const BASE: u64 = 1 << 40;
        /// The launch supply's handle in the simulation.
        const LAUNCH: usize = usize::MAX;

        /// Leases and a kernel heap cursor, placing as the session engine
        /// does: a view of at least 2 MiB over a 2 MiB-aligned backing on a
        /// 2 MiB boundary, anything else at the next page. A lease's backing
        /// starts on a 2 MiB frame (its class), so the backing is aligned
        /// when the carve's offset is.
        struct Sim {
            adopted: Vec<(u64, Cursors)>,
            launch: Option<(u64, Cursors)>,
            cursor: RefCell<u64>,
            /// Every granule mapped: `(supply, offset, size, va)`.
            maps: RefCell<Vec<(usize, u64, u64, u64)>>,
        }

        impl Sim {
            fn leases(lengths: &[u64]) -> Self {
                Self {
                    adopted: lengths.iter().map(|&len| (len, Cursors::new())).collect(),
                    launch: None,
                    cursor: RefCell::new(BASE),
                    maps: RefCell::new(Vec::new()),
                }
            }

            fn rooms(&self) -> Vec<u64> {
                self.adopted
                    .iter()
                    .map(|(total, cursors)| cursors.room(*total).unwrap())
                    .collect()
            }

            fn sizes(&self) -> Vec<u64> {
                self.maps.borrow().iter().map(|&(_, _, size, _)| size).collect()
            }
        }

        impl Granules for Sim {
            type Supply = usize;

            fn adopted(&self) -> impl Iterator<Item = (usize, u64, &Cursors)> {
                self.adopted
                    .iter()
                    .enumerate()
                    .map(|(index, (total, cursors))| (index, *total, cursors))
            }

            fn launch(&self) -> Option<(usize, u64, &Cursors)> {
                self.launch.as_ref().map(|(total, cursors)| (LAUNCH, *total, cursors))
            }

            fn map(&self, supply: usize, offset: u64, size: u64) -> Option<u64> {
                let align = if offset % GRANULE_MIN == 0 && size >= GRANULE_MIN {
                    GRANULE_MIN
                } else {
                    PAGE
                };
                let mut cursor = self.cursor.borrow_mut();
                let va = cursor.next_multiple_of(align);
                *cursor = va + size;
                self.maps.borrow_mut().push((supply, offset, size, va));
                Some(va)
            }
        }

        fn window() -> Window {
            let window = Window::new();
            window.learn_base(BASE);
            window
        }

        fn at(extent: NonNull<[u8]>) -> (u64, u64) {
            (extent.cast::<u8>().as_ptr() as u64, extent.len() as u64)
        }

        /// The photo that could not be saved: 64 MiB of pixels against two
        /// leases with 28 and 56 MiB left. Neither holds it, together they
        /// do, and the run is placed whole: the longer lease's room, then
        /// what is still missing off the first lease that holds it, end to
        /// end.
        #[test]
        fn a_run_longer_than_any_one_supply_is_assembled_from_several() {
            let sim = Sim::leases(&[128 * MIB, 56 * MIB]);
            assert_eq!(sim.adopted[0].1.claim_bottom(128 * MIB, 100 * MIB), Some(0));
            let window = window();
            assert_eq!(window.room(&sim), 84 * MIB);

            let extent = window.fill(&sim, 64 * MIB).expect("assembled");
            assert_eq!(at(extent), (BASE, 64 * MIB));
            assert_eq!(
                *sim.maps.borrow(),
                [(1, 0, 56 * MIB, BASE), (0, 100 * MIB, 8 * MIB, BASE + 56 * MIB)],
                "the longest room whole, then the rest from the first lease holding it"
            );
            // Eight of the first lease's 28 MiB are the run's tail; the doubling
            // took nothing more, and none of it went astray.
            assert_eq!(sim.rooms(), [20 * MIB, 0]);
            assert_eq!(window.room(&sim), 20 * MIB);
        }

        /// A run the supplies cannot hold together is refused before a
        /// granule is mapped: their room stays whole for what does fit.
        #[test]
        fn a_run_the_supplies_cannot_hold_spends_nothing() {
            let sim = Sim::leases(&[128 * MIB, 56 * MIB]);
            assert_eq!(sim.adopted[0].1.claim_bottom(128 * MIB, 100 * MIB), Some(0));
            let window = window();
            assert!(window.fill(&sim, 84 * MIB + PAGE).is_none());
            assert!(sim.maps.borrow().is_empty());
            assert_eq!(sim.rooms(), [28 * MIB, 56 * MIB]);
            // What the room says fits does.
            assert_eq!(at(window.fill(&sim, 84 * MIB).unwrap()), (BASE, 84 * MIB));
            assert_eq!(window.room(&sim), 0);
        }

        /// A free interval that ends off a 2 MiB boundary is not counted:
        /// the next granule of 2 MiB or more lands past a gap and replaces
        /// it. What the room says still maps, the stranded tail aside.
        #[test]
        fn the_room_counts_the_interval_only_where_a_granule_extends_it() {
            // Three of four MiB heap's, one taken off its top for a window.
            let sim = Sim::leases(&[4 * MIB, 8 * MIB]);
            assert_eq!(sim.adopted[0].1.claim_top(4 * MIB, MIB), Some(3 * MIB));
            // A busy second lease, so the first is drawn on alone.
            assert_eq!(sim.adopted[1].1.claim_bottom(8 * MIB, 8 * MIB), Some(0));
            let window = window();
            assert_eq!(at(window.fill(&sim, MIB).unwrap()), (BASE, MIB));
            // Its last MiB takes page grain: the interval ends at 3 MiB.
            let tail = 3 * MIB / 2;
            assert_eq!(at(window.fill(&sim, tail).unwrap()), (BASE + MIB, tail));
            assert_eq!(sim.sizes(), [2 * MIB, MIB]);
            assert_eq!(window.room(&sim), 0, "half a MiB free, but off the boundary");

            // A fresh lease of 8 MiB: the room is its 8, not 8½.
            let mut sim = sim;
            sim.adopted.push((8 * MIB, Cursors::new()));
            assert_eq!(window.room(&sim), 8 * MIB);
            assert!(window.fill(&sim, 8 * MIB + PAGE).is_none());
            let extent = window.fill(&sim, 8 * MIB).expect("the room maps");
            assert_eq!(at(extent), (BASE + 4 * MIB, 8 * MIB), "past the gap");
        }

        /// A founded service's heap: a 2 MiB launch supply alone. Its quarter
        /// is under a granule, so every granule is of page grain, which the
        /// kernel places at the interval's very end, and the free interval
        /// ends off a 2 MiB boundary from the first refill on. The free tail
        /// still counts toward what the next granule must bring, and the
        /// supply is spent to its last page. Counting only what a 2 MiB
        /// granule would extend asked each refill for the whole extent: the
        /// hypervisor's MicroPython package, some 1.8 MiB of extents, no
        /// longer fit once anything was allocated before it, and every
        /// boot-userland-gate lane with hardware virtualization went red on
        /// 2570cd4.
        #[test]
        fn a_service_supply_is_spent_to_its_last_page() {
            let mut sim = Sim::leases(&[]);
            sim.launch = Some((2 * MIB, Cursors::new()));
            let window = window();
            assert_eq!(at(window.fill(&sim, 64 * KIB).unwrap()), (BASE, 64 * KIB));
            assert_eq!(sim.sizes(), [512 * KIB], "the supply's quarter, in pages");
            // 448 KiB free, ending off the boundary; 1½ MiB left to claim.
            let rest = 2 * MIB - 64 * KIB;
            let extent = window.fill(&sim, rest).expect("the tail and the supply's rest hold it");
            assert_eq!(at(extent), (BASE + 64 * KIB, rest), "one run, the tail first");
            assert_eq!(sim.sizes(), [512 * KIB, 3 * MIB / 2], "only what was missing");
            assert!(window.fill(&sim, PAGE).is_none(), "spent");
        }

        /// The doubling still sizes whole granules — 2, 4, 8, 16, then 32 MiB
        /// — and stays at 32 MiB however many granules a long-lived heap has
        /// mapped (a shift once wrapped it to zero after forty-three).
        #[test]
        fn whole_granules_double_to_their_ceiling_and_stay_there() {
            let sim = Sim::leases(&[512 * MIB]);
            let window = window();
            for _ in 0..31 {
                window.fill(&sim, 2 * MIB).unwrap();
            }
            assert_eq!(
                sim.sizes(),
                [2 * MIB, 4 * MIB, 8 * MIB, 16 * MIB, 32 * MIB],
                "62 MiB in five sessions"
            );
            window.granules.store(60, Ordering::Relaxed);
            window.fill(&sim, 2 * MIB).unwrap();
            assert_eq!(sim.sizes().last(), Some(&(32 * MIB)));
        }

        /// The launch supply serves whole granules as it did, and is never
        /// taken whole for a part: what the adopted supplies cannot hold is
        /// refused, and the launch supply is left to its windows.
        #[test]
        fn the_launch_supply_is_never_split_into_a_run() {
            let mut sim = Sim::leases(&[10 * MIB]);
            sim.launch = Some((16 * MIB, Cursors::new()));
            let window = window();
            assert_eq!(window.room(&sim), 10 * MIB);
            assert!(window.fill(&sim, 20 * MIB).is_none());
            assert!(sim.maps.borrow().is_empty());
            // A granule the launch supply holds whole still comes from it once
            // the adopted supplies cannot.
            assert_eq!(sim.adopted[0].1.claim_bottom(10 * MIB, 10 * MIB), Some(0));
            window.fill(&sim, 3 * MIB).unwrap();
            assert_eq!(sim.maps.borrow()[0].0, LAUNCH);
        }
    }
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
    carve_and_map_in(supply, offset, len, None)
}

/// [`carve_and_map`] with the map staged into a named receiver window
/// (`ViewMapWindow`) instead of the thread's receiver-map cursor — the same
/// single session, the same scratch slots, the same bind-inert report: the
/// window selects placement only, and nothing is held afterwards.
#[cfg(target_os = "none")]
pub fn carve_and_map_in(
    supply: kore_abi::session::SlotRef,
    offset: u64,
    len: u64,
    window: Option<kore_abi::session::ViewMapWindow>,
) -> Option<NonNull<[u8]>> {
    carve_and_map_at(
        supply,
        offset,
        len,
        window.map_or(Placement::Cursor, Placement::Window),
    )
}

/// Where [`carve_and_map_at`] maps its carve.
#[cfg(target_os = "none")]
#[derive(Clone, Copy)]
enum Placement {
    /// The thread's receiver-map cursor.
    Cursor,
    /// A named receiver window, at its own placement.
    Window(kore_abi::session::ViewMapWindow),
    /// An exact page of the `RESERVE` window ([`commit_reserved`]).
    ReservePage(u32),
}

/// [`carve_and_map_in`]'s session, with the carve's [`Placement`].
#[cfg(target_os = "none")]
fn carve_and_map_at(
    supply: kore_abi::session::SlotRef,
    offset: u64,
    len: u64,
    placement: Placement,
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
    match placement {
        Placement::Cursor => b.derive_raw(carve, Rights::empty(), 0, report, None).ok()?,
        Placement::Window(window) => b.map_view_in(carve, window, report).ok()?,
        Placement::ReservePage(page) => b.map_view_reserved(carve, page, report).ok()?,
    };
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
    fn the_two_cursors_refuse_each_other() {
        let total = 0x10_0000;
        // Top-down: two pages off the end.
        let cursors = Cursors::new();
        assert_eq!(cursors.claim_top(total, 0x2000), Some(total - 0x2000));
        assert_eq!(cursors.room(total), Some(total - 0x2000));
        // Bottom-up growth may use everything below the top...
        assert_eq!(cursors.claim_bottom(total, total - 0x2000), Some(0));
        assert_eq!(cursors.room(total), Some(0));
        // ...but not one page into it.  The misfit is seen before it is
        // claimed, so it spends nothing.
        let cursors = Cursors::new();
        assert_eq!(cursors.claim_top(total, 0x2000), Some(total - 0x2000));
        assert_eq!(cursors.claim_bottom(total, total - 0x1000), None);
        assert_eq!(cursors.grown.load(Ordering::SeqCst), 0);
        assert_eq!(cursors.room(total), Some(total - 0x2000));
        // A reservation that would cross the grow mark refuses the same way.
        let cursors = Cursors::new();
        assert_eq!(cursors.claim_bottom(total, total - 0x3000), Some(0));
        assert_eq!(cursors.claim_top(total, 0x2000), Some(total - 0x2000));
        assert_eq!(cursors.claim_top(total, 0x2000), None);
        assert_eq!(cursors.reserved.load(Ordering::SeqCst), 0x2000);
        // Overflow past the supply refuses, on either side, without moving
        // the cursor it would have wrapped.
        assert_eq!(cursors.claim_top(total, u64::MAX), None);
        assert_eq!(cursors.reserved.load(Ordering::SeqCst), 0x2000);
        assert_eq!(cursors.claim_bottom(total, u64::MAX), None);
        assert_eq!(cursors.grown.load(Ordering::SeqCst), total - 0x3000);
        // The page left between them is still there for either side.
        assert_eq!(cursors.room(total), Some(0x1000));
        assert_eq!(cursors.claim_top(total, 0x1000), Some(total - 0x3000));
        assert_eq!(cursors.room(total), Some(0));
    }

    /// A claim that loses the race to the other cursor is refused and stays
    /// spent: the pre-check reads a snapshot, and only claim-then-check keeps
    /// the two apart.  Replayed by hand: the bottom's snapshot saw the whole
    /// supply free, the top then claimed two pages, and the bottom's claim
    /// landed after it — the half of `claim_bottom` its check then refuses.
    #[test]
    fn a_claim_that_loses_the_race_stays_spent() {
        let total = 0x10_0000;
        let cursors = Cursors::new();
        assert_eq!(cursors.room(total), Some(total));
        assert_eq!(cursors.claim_top(total, 0x2000), Some(total - 0x2000));
        assert_eq!(claim(&cursors.grown, total), Some(0));
        // Spent: the supply refuses both sides rather than let them overlap.
        assert_eq!(cursors.room(total), None);
        assert_eq!(cursors.claim_top(total, 0x1000), None);
        assert_eq!(cursors.claim_bottom(total, 0x1000), None);
    }

    /// An adopted supply serves both claimants, so a heap granule sized off
    /// it must leave its reserved top alone: with 96 of 128 MiB grown and a
    /// 16 MiB window reserved, 16 MiB remain, and a 32 MiB granule refuses
    /// rather than mapping the heap over the live window.
    #[test]
    fn a_granule_never_reaches_a_reserved_window() {
        const MIB: u64 = 1 << 20;
        let total = 128 * MIB;
        let cursors = Cursors::new();
        assert_eq!(cursors.claim_bottom(total, 96 * MIB), Some(0));
        assert_eq!(cursors.claim_top(total, 16 * MIB), Some(112 * MIB));
        assert_eq!(cursors.room(total), Some(16 * MIB));
        assert_eq!(cursors.claim_bottom(total, 32 * MIB), None);
        // The refused granule spent nothing: the 16 MiB between them is
        // whole, and a granule it holds is placed below the window, ending
        // exactly at it.
        assert_eq!(cursors.room(total), Some(16 * MIB));
        assert_eq!(cursors.claim_bottom(total, 16 * MIB), Some(96 * MIB));
        assert_eq!(cursors.room(total), Some(0));
        assert_eq!(cursors.claim_top(total, 4096), None);
    }

    /// A guest memory relocating tries its window against each supply in
    /// turn: the ones too short for it are passed over intact, and the
    /// first that holds it takes it.
    #[test]
    fn a_window_too_long_for_a_supply_leaves_it_whole() {
        const MIB: u64 = 1 << 20;
        let short = Cursors::new();
        let long = Cursors::new();
        assert_eq!(short.claim_bottom(128 * MIB, 100 * MIB), Some(0));
        let window = 60 * MIB;
        let placed = [(&short, 128 * MIB), (&long, 64 * MIB)]
            .iter()
            .position(|(cursors, total)| cursors.claim_top(*total, window).is_some());
        assert_eq!(placed, Some(1));
        // The short supply's 28 MiB are still the heap's and later windows'.
        assert_eq!(short.room(128 * MIB), Some(28 * MIB));
        assert_eq!(short.claim_top(128 * MIB, 16 * MIB), Some(112 * MIB));
        assert_eq!(long.room(64 * MIB), Some(4 * MIB));
    }

    /// Claims racing from both ends never hand out one byte twice: every
    /// interval either claimant was granted is disjoint from every other.
    #[test]
    fn racing_claims_from_both_ends_are_disjoint() {
        use std::sync::Barrier;
        use std::vec::Vec;

        const PAGE: u64 = 4096;
        const CLAIMANTS: usize = 4;
        let total = 512 * PAGE;
        for round in 0..64u64 {
            let cursors = Cursors::new();
            let start = Barrier::new(CLAIMANTS);
            let mut granted: Vec<(u64, u64)> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..CLAIMANTS)
                    .map(|claimant| {
                        let (cursors, start) = (&cursors, &start);
                        scope.spawn(move || {
                            let mut mine = Vec::new();
                            start.wait();
                            for step in 0..64u64 {
                                let want = PAGE * (1 + (round + step + claimant as u64) % 3);
                                let offset = if claimant % 2 == 0 {
                                    cursors.claim_bottom(total, want)
                                } else {
                                    cursors.claim_top(total, want)
                                };
                                mine.extend(offset.map(|offset| (offset, offset + want)));
                            }
                            mine
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("a claimant panicked"))
                    .collect()
            });
            granted.sort_unstable();
            for pair in granted.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "round {round}: {pair:?} overlap");
            }
            assert!(granted.last().is_none_or(|&(_, end)| end <= total));
        }
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
        assert_eq!(claim(&cursor, 4096), None);
        assert_eq!(cursor.load(Ordering::Acquire), u64::MAX - 1023);
    }

    #[test]
    fn static_source_refuses_an_unrepresentable_round_up() {
        let src = host_arena(EXTENT_ALIGN * 2);
        assert!(src.map_extent(usize::MAX).is_none());
    }
}
