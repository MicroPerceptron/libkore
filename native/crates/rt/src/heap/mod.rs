//! # `heap` — the userland global allocator
//!
//! A **sharded, lock-free** allocator in the mimalloc shape, the userland
//! analog of the kernel's growable heap (`src/runtime/mm/heap.rs`, #205/#208):
//! a lock-free directory of extent-backed pools, self-hosted metadata in each
//! extent's head, and deallocation routed by which extent contains the pointer.
//! Where the kernel heap lays one TLSF engine over each extent, this lays
//! **per-shard segregated free-lists** over them — the userland workload is
//! many tasks allocating in parallel, so the design optimizes for *no shared
//! hot-path state* over *low fragmentation*.
//!
//! ## Two faces, one engine
//!
//! [`Heap`] answers both allocator traits over a single fallible core
//! ([`Heap::try_alloc`] / [`Heap::dealloc_ptr`]), exactly as libstd's `System`
//! does:
//!
//! - [`GlobalAlloc`] (for `#[global_allocator]`): a null return is the sole
//!   failure signal, and the caller's `handle_alloc_error` aborts — intended.
//! - [`Allocator`] (allocator_api, for `Box::new_in`/`Vec::new_in`): the
//!   `AllocError` is **propagated**, never aborted.
//!
//! kore-rt *ships the type*; it does **not** install it as the crate's global
//! allocator (that would bind every consumer and the test harness).  A binary
//! installs it over its chosen [`MemSource`] — a founded binary links the
//! metal source:
//!
//! ```ignore
//! use kore_rt::heap::{Heap, HeapSupplySource};
//! #[global_allocator]
//! static ALLOC: Heap<HeapSupplySource> = Heap::new_in(HeapSupplySource);
//! ```
//!
//! ## The shape
//!
//! - **Small/medium** (`<= 64 KiB` after folding alignment): per-shard, per-size
//!   free-lists.  A task pops from *its own* shard — zero atomics on the hot
//!   path.  A free block holds its successor pointer intrusively, so the
//!   allocator never allocates to allocate.
//! - **Cross-thread free**: a block owned by another shard is pushed onto that
//!   shard's lock-free MPSC remote list (the owner drains it on its own miss).
//! - **Huge** (`> 64 KiB`): a dedicated whole-extent allocation, aligned to
//!   the request, recycled whole through generation-tagged free stacks kept
//!   by size ([`HUGE_BUCKETS`]).
//! - **Grow on miss**: map a fresh extent through the [`MemSource`] seam and
//!   carve it — synchronously, since userland has no ISR/FP/lock-held hazard.
//! - **Routing is O(1)**, never a directory walk.  Every extent starts on a
//!   page and no two overlap, so each page of the address space belongs to
//!   at most one extent, and a four-level **page map** (a radix over the
//!   user half of the address space, keyed by page number) names that
//!   extent: a free is four dependent loads.  The map asks nothing of how
//!   or where a source places an extent, so a 2 MiB service supply pays no
//!   alignment skip and a worker's per-extent carve costs what it asks.
//! - **The extent directory grows.**  Live extents are registered in pages of
//!   [`PAGE_SLOTS`] pointers: the first page is inline in the [`Heap`] (so a
//!   `static` heap needs no init pass), later pages are added as grows need
//!   them.  Compiling a 7.8 MB WebAssembly component on kore needs some
//!   3,900 extents; a fixed 1,024-slot directory failed that compile at
//!   90 MiB mapped.
//! - **A grow is one mapping.**  The metadata a grow needs — a directory
//!   page, the page-map nodes its span lacks — is carved from the tail of
//!   the mapping it asked for, or taken from a reserve of pages earlier
//!   grows left over (metadata self-hosted in source memory, never
//!   allocated from the heap itself).  A source that answers a miss with
//!   one extent of what was asked, once (`kore-svc-mm`'s refill protocol),
//!   is therefore always enough.
//!
//! See [`source`] for the [`MemSource`] seam — `kore-svc-mm`'s default
//! backing routes growth through the memory-manager service, so a heap
//! extent is a lessor *grant*, never a self-derive (what leaves task heaps
//! swappable, the lessor inversion).

mod pool;
mod shard;
mod source;

pub use crate::mem_source::MemSource;
pub use source::MAX_ADOPTED;
#[cfg(target_os = "none")]
pub use source::carve_and_map;
pub(crate) use source::{CommitSupply, commit_reserved, release_reserved};
pub use source::{HeapSupplySource, StaticMemSource};
pub use source::{
    adopt_supply, adopt_window_supply, heap_room, reserve_from_top, window_supply_room,
};

use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use kore_collections::AtomicBitSet;

use pool::{MIN_BLOCK, align_up, carve_lane, checked_align_up, class_index, class_size, is_huge};
use shard::{NUM_SHARDS, Shard, current_shard};

/// Alignment of every mapped extent (and its self-hosted header) — a page.  The
/// [`MemSource`] returns extents aligned to this.  Public because it is the
/// seam's granularity contract: an out-of-crate source (and `kore-svc-mm`'s
/// request artifact, `client::alloc_props`) sizes and aligns to it.
pub const EXTENT_ALIGN: usize = crate::mem_source::ALIGN;

/// Bytes mapped per pool-extent grow.  A dial: large enough to amortize the map
/// over many small allocations, small enough that growth tracks live demand.
pub const POOL_EXTENT_BYTES: usize = 64 * 1024;

/// The most blocks a large class's extent holds.  A large class's first
/// extent in a shard holds one block and each later one twice what the
/// last held, up to this ([`Heap::pool_extent_bytes`]): what a lane maps
/// ahead of its demand is never more than it has carved already, so a
/// heap serving a few large requests — a bounded arena of a megabyte —
/// pays about a block and a page for each large class it uses, while one
/// serving thousands (a Cranelift compile) maps them eight to an extent
/// after its first fifteen.  Eight keeps the mapping a few times the
/// block: a miss costs no more than the shared lane's, relative to what
/// it serves.
const LARGE_CLASS_MAX_BLOCKS: usize = 8;

/// The alignment `class`'s blocks are carved at.  In the shared lane, the
/// class size: that is how a small class honours any `align` up to its
/// size.  In a large class's lane, a page: its blocks follow the header
/// page back to back, so an extent of one block wastes nothing to
/// alignment, and a large request aligned past a page is served on the
/// huge path instead ([`Heap::try_alloc`]).
#[inline]
const fn block_align(class: usize) -> usize {
    if carve_lane(class) == 0 {
        class_size(class)
    } else {
        EXTENT_ALIGN
    }
}

// A large class's header fits the page its blocks follow, and its blocks
// are whole pages, so every block in its lane is page-aligned.
const _: () = {
    assert!(extent_header_bytes() <= EXTENT_ALIGN);
    assert!(class_size(pool::SHARED_LANE_TOP_CLASS + 1) % EXTENT_ALIGN == 0);
};
// The shared lane carves a grow's metadata off its own tail, down to half.
const _: () = assert!(meta_pages_bound(POOL_EXTENT_BYTES) * EXTENT_ALIGN <= POOL_EXTENT_BYTES / 2);

/// **The page map's shape.**  Keyed by page number ([`EXTENT_ALIGN`] is a
/// page), four levels of [`PAGE_MAP_FANOUT`]-way nodes cover
/// `PAGE_MAP_BITS` bits of page number — every user address below
/// 2^47, which is where kore places a task's windows on all three ISAs.
/// The first level is inline in the [`Heap`] (`PAGE_MAP_FANOUT` pointers,
/// 2 KiB — the inline directory page shrank by the same amount so a
/// `static` heap's footprint did not grow); every other node is one 4 KiB
/// source extent, mapped when the first extent in its span is registered:
/// a heap's first extent costs three nodes, 12 KiB, and a 16 MiB span of
/// extents costs one leaf node after that.
///
/// **What breaks if it moves.**  A leaf covering less than a pool extent
/// would cost a node per extent; a wider inline root is `.bss` every image
/// pays from its link supply (64 KiB on the ordinary launch plan).  An
/// address past the covered bits cannot be registered, so an extent there
/// refuses at the grow rather than routing wrong.
const PAGE_MAP_FANOUT: usize = 512;
const PAGE_MAP_LEVELS: usize = 4;
const PAGE_MAP_ROOT_FANOUT: usize = 256;
const PAGE_MAP_LEVEL_BITS: usize = PAGE_MAP_FANOUT.trailing_zeros() as usize;
const PAGE_MAP_ROOT_BITS: usize = PAGE_MAP_ROOT_FANOUT.trailing_zeros() as usize;
const PAGE_BITS: usize = EXTENT_ALIGN.trailing_zeros() as usize;
/// Bits of page number the map covers: `8 + 9 + 9 + 9 = 35`, so addresses
/// below `2^47`.
const PAGE_MAP_BITS: usize = PAGE_MAP_ROOT_BITS + (PAGE_MAP_LEVELS - 1) * PAGE_MAP_LEVEL_BITS;
const _: () = assert!(PAGE_MAP_FANOUT.is_power_of_two() && PAGE_MAP_ROOT_FANOUT.is_power_of_two());
const _: () = assert!(EXTENT_ALIGN.is_power_of_two());
/// A page-map node is exactly one page; a directory page fits one.
const _: () = assert!(core::mem::size_of::<PageMapNode>() == EXTENT_ALIGN);
const DIR_PAGE_BYTES: usize = align_up(core::mem::size_of::<DirPage>(), EXTENT_ALIGN);
const _: () = assert!(DIR_PAGE_BYTES == EXTENT_ALIGN);

/// Extent-directory slots per page: what fits one page with the page's
/// own ledger, so a directory page is one unit of grow metadata like a
/// map node.  The first page is inline in the [`Heap`] (its only up-front
/// cost: `PAGE_SLOTS` pointers and a bitset in the `static`); every later
/// page is carved from a grow's own mapping.
pub const PAGE_SLOTS: usize = 256;
const PAGE_WORDS: usize = PAGE_SLOTS.div_ceil(u64::BITS as usize);
/// How many directory pages a heap may hold: an architectural ceiling on
/// live extents of `DIR_PAGES × PAGE_SLOTS` (131,072 — 8 GiB of pool
/// extents; the CPython compile of `res/WASM_PLATFORM.md` §11 holds some
/// 3,900), with `DIR_PAGES - 1` page pointers inline (4 KiB of `.bss`).
/// Memory tracks demand, not this ceiling (the #205 shape).
pub const DIR_PAGES: usize = 512;
const _: () = assert!(DIR_PAGES * PAGE_SLOTS < FREE_EMPTY as usize);

/// How many free huge extents one cold allocation examines before mapping.
/// A bounded scan defeats head-of-line blocking (one undersized head hiding a
/// suitable extent) without turning a huge allocation into an unbounded walk.
const HUGE_RECYCLE_SCAN: usize = 16;

/// **Recycled huge extents are kept by size.**  Bucket `b` holds the free
/// huge extents whose length is in `[EXTENT_ALIGN << b, EXTENT_ALIGN << (b + 1))`,
/// each bucket a generation-tagged Treiber stack.  A request looks in the
/// bucket its length falls in — where a fit is likely but not certain, so
/// it scans [`HUGE_RECYCLE_SCAN`] entries — and then takes the head of the
/// first non-empty larger bucket, which fits by construction, so a free
/// extent of any size is found in O(buckets) with no scan.  Thirty-two
/// buckets reach lengths past the page map's 2^47 bytes of address space.
///
/// A recycled extent is handed out **whole**, however long: splitting a
/// longer one down to the request was measured and rejected.  Without
/// coalescing on free, split tails fragment the long extents a compiler's
/// code buffers double into, and every doubling then maps afresh — the
/// large requests alone (above 64 KiB, 88 MiB live at their peak) mapped
/// 378 MiB split against 122 MiB whole, compiling CPython on the host
/// harness.  Whole, the buckets' best fit keeps that population within
/// 1.4× of its peak; the small-but-huge population (8 KiB to 64 KiB) is
/// the pool's ([`pool::LARGE_THRESHOLD`]).  Coalescing adjacent free
/// extents, which would let a split pay, needs the recycle links and a
/// per-extent state out of the extent's own memory and is the next step.
const HUGE_BUCKETS: usize = 32;
const EXTENT_ALIGN_BITS: usize = EXTENT_ALIGN.trailing_zeros() as usize;

/// The recycle bucket a huge extent of `len` bytes (`>= EXTENT_ALIGN`) goes in.
#[inline]
const fn huge_bucket(len: usize) -> usize {
    let bits = if len == 0 { 0 } else { len.ilog2() as usize };
    let bucket = bits.saturating_sub(EXTENT_ALIGN_BITS);
    if bucket >= HUGE_BUCKETS {
        HUGE_BUCKETS - 1
    } else {
        bucket
    }
}

// ── Huge-free stack packing (the board.rs generation-tagged Treiber scheme) ───
const FREE_EMPTY: u32 = u32::MAX;

#[inline]
const fn free_pack(generation: u32, id: u32) -> u64 {
    ((generation as u64) << 32) | id as u64
}
#[inline]
const fn free_gen(word: u64) -> u32 {
    (word >> 32) as u32
}
#[inline]
const fn free_id(word: u64) -> u32 {
    word as u32
}

/// Which allocation path an extent backs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtentKind {
    /// Carved into segregated size-class blocks for one shard.
    Pool,
    /// A single huge allocation (the whole payload).
    Huge,
}

/// What a fresh extent's payload is once its metadata is carved
/// ([`Heap::install_extent`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Payload {
    /// All the mapping the metadata leaves, which may carve it down to
    /// half the request: the shared lane, whose small blocks use any tail.
    Rest,
    /// The request, page-rounded.  The metadata comes from the reserve or
    /// the over-ask past the request, and what it leaves goes to the
    /// reserve: a huge extent, and a large class's, whose whole blocks a
    /// stray page could not hold.
    Exact,
}

/// One mapped extent, its bookkeeping **self-hosted in its own head** (the
/// #205/§9.4 rule: the allocator cannot allocate its own metadata *from
/// itself*).  Its address is stable for the heap's life — extents are held,
/// never moved — so a lock-free reader may hold an `&Extent` safely.
///
/// The geometry fields (`base`/`len`/`owner`/`kind`/`dir_idx`) are immutable
/// after publication, so cross-thread routing reads them without
/// synchronization; the mutable state is atomic (`cursor` for owner-only pool
/// carving, `next_free` for the huge-free stack link).
struct Extent {
    /// Start of the mapped region — the header lives here, and `base` is the
    /// deallocate-routing key with `len`.
    base: usize,
    /// Payload bytes: the header, then what is carved or served.
    len: usize,
    /// Bytes of the mapping this extent came in — `len`, the metadata
    /// carved from its tail and what it left to the reserve — for the
    /// occupancy snapshot.
    mapped: usize,
    /// Owning shard index (Pool), or `usize::MAX` (Huge — owned by no shard).
    owner: usize,
    /// Which path this extent backs.
    kind: ExtentKind,
    /// This extent's directory slot — its id on the huge-free stack.
    dir_idx: u32,
    /// Pool carve cursor: the next un-carved absolute address.  Owner-only
    /// writer, so `Relaxed` suffices; other threads never read it.
    cursor: AtomicUsize,
    /// Huge-free Treiber link: the directory id of the next free huge extent.
    next_free: AtomicU32,
}

impl Extent {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    #[inline]
    fn end(&self) -> usize {
        self.base + self.len()
    }
    #[inline]
    fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr - self.base < self.len()
    }
}

/// Bytes reserved at an extent's head for its self-hosted [`Extent`], rounded so
/// the carve area that follows keeps [`MIN_BLOCK`] alignment.
#[inline]
const fn extent_header_bytes() -> usize {
    align_up(core::mem::size_of::<Extent>(), MIN_BLOCK)
}

/// One interior or leaf node of the page map: [`PAGE_MAP_FANOUT`] pointers,
/// `null` until published — to a child node (interior levels) or to the
/// [`Extent`] owning that page (the leaf level).  One 4 KiB source extent
/// each, written in place and published by a compare-and-swap on the parent
/// slot, so two threads registering into one empty span race on the slot
/// and the loser keeps its node for the next empty slot it meets.
struct PageMapNode {
    slots: [AtomicPtr<PageMapNode>; PAGE_MAP_FANOUT],
}

impl PageMapNode {
    const fn new() -> Self {
        Self {
            slots: [const { AtomicPtr::new(core::ptr::null_mut()) }; PAGE_MAP_FANOUT],
        }
    }
}

/// The slot index at each level of the page map for the page holding
/// `addr`: the root's first, then the three node levels, the leaf last.
/// `None` for an address past the covered bits.
#[inline]
fn page_map_path(addr: usize) -> Option<[usize; PAGE_MAP_LEVELS]> {
    let page = addr >> PAGE_BITS;
    if page >> PAGE_MAP_BITS != 0 {
        return None;
    }
    let mut path = [0; PAGE_MAP_LEVELS];
    let mut rest = page;
    let mut level = PAGE_MAP_LEVELS - 1;
    while level > 0 {
        path[level] = rest & (PAGE_MAP_FANOUT - 1);
        rest >>= PAGE_MAP_LEVEL_BITS;
        level -= 1;
    }
    path[0] = rest & (PAGE_MAP_ROOT_FANOUT - 1);
    Some(path)
}

/// Occupancy snapshot (diagnostics / tests).  Live-byte accounting is
/// deliberately *not* maintained — it would put an atomic on the hot path — so
/// this reports the mapped footprint, computed by a cold directory scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapStats {
    /// Total bytes mapped across all live extents.
    pub mapped_bytes: usize,
    /// Number of live extents (pool + huge).
    pub extent_count: usize,
}

/// The outcome of a grow attempt (see [`Heap::grow_pool`]).
enum GrowOutcome {
    /// A fresh extent was mapped and installed; retry the fast path.
    Grew,
    /// The [`MemSource`] is exhausted (or the directory is full) — surface
    /// `AllocError`.
    Exhausted,
}

/// One page of the extent directory: [`PAGE_SLOTS`] slots published
/// `Release` by [`Heap::install_extent`], read `Acquire`, `null` until
/// published (#205), with the page's own free-slot ledger and cold-scan
/// bound.  Page 0 lives inline in the [`Heap`]; every later page is one
/// source extent, written in place and published by a compare-and-swap on
/// its [`Heap::pages`] pointer.
struct DirPage {
    slots: [AtomicPtr<Extent>; PAGE_SLOTS],
    /// Free slots.  Claims happen before mapping; rollback on source refusal
    /// makes the capacity failure-atomic.
    free: AtomicBitSet<PAGE_WORDS>,
    /// One past the highest slot ever published in this page — the cold scan
    /// bound.
    high_water: AtomicUsize,
}

impl DirPage {
    const fn new() -> Self {
        Self {
            slots: [const { AtomicPtr::new(core::ptr::null_mut()) }; PAGE_SLOTS],
            free: AtomicBitSet::with_first_set(PAGE_SLOTS),
            high_water: AtomicUsize::new(0),
        }
    }
}

/// Affine reservation of one extent-directory slot.
///
/// A miss claims before it calls the source, so directory exhaustion cannot
/// strand a freshly mapped extent.  Refusal or malformed source geometry drops
/// the claim and restores the bit; publication consumes it permanently because
/// live extents remain mapped for the process lifetime.
struct DirectoryClaim<'a> {
    page: &'a DirPage,
    /// The slot within `page`.
    slot: usize,
    /// The directory-wide id (`page index × PAGE_SLOTS + slot`): what the
    /// extent records and the huge-free stack links by.
    id: u32,
    published: bool,
}

impl DirectoryClaim<'_> {
    #[inline]
    const fn id(&self) -> u32 {
        self.id
    }

    #[inline]
    fn publish(mut self) {
        self.published = true;
    }
}

impl Drop for DirectoryClaim<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.page.free.release_run(self.slot, 1);
        }
    }
}

/// **What a grow may need beyond its payload.**  Every grow completes with
/// the one mapping it asked for: its directory page (when every installed
/// page is full) and the page-map nodes its span lacks are carved from
/// that mapping's tail, or taken from the metadata reserve — never mapped
/// on their own.  A source that answers one miss with one extent and no
/// more (`kore-svc-mm`'s `MmSource`: a miss records the size, the service
/// refills once, the allocation retries once) is therefore served exactly.
///
/// The bound for a span of `len` bytes: one directory page, and at each
/// node level the spans the extent can touch — one more than its length
/// divides into, since it need not start on one.  64 KiB: 1 + 2 + 2 + 2 =
/// 7 pages; 16 MiB: 1 + 2 + 2 + 9 = 14.  What the grow does not use goes
/// to the reserve (a huge grow) or stays carve space (a pool grow).
const fn meta_pages_bound(len: usize) -> usize {
    let mut pages = DIR_PAGE_BYTES / EXTENT_ALIGN;
    let mut level = 1;
    while level < PAGE_MAP_LEVELS {
        pages += (len >> level_shift(level)) + 2;
        level += 1;
    }
    pages
}

/// The address bits below one node `level` steps under the root: the span
/// one of its slots covers is `1 << level_shift(level)` bytes (a leaf's,
/// at `PAGE_MAP_LEVELS - 1`, is a page).
const fn level_shift(level: usize) -> usize {
    PAGE_BITS + (PAGE_MAP_LEVELS - level) * PAGE_MAP_LEVEL_BITS
}

/// How many runs the metadata reserve holds.  A run is one over-ask's
/// leftover; a grow over-asks only when the reserve is short of its bound,
/// which drains the reserve first, so runs seldom accumulate — sixteen
/// is room for concurrent growers.  A seventeenth is that race's lost
/// pages.
const META_RESERVE_RUNS: usize = 16;
const META_RUN_COUNT_BITS: u32 = 16;
const _: () = assert!(PAGE_MAP_BITS + META_RUN_COUNT_BITS as usize <= 64);

/// A run of unregistered, page-aligned pages held for metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetaRun {
    base: usize,
    pages: usize,
}

impl MetaRun {
    const fn pack(self) -> u64 {
        (((self.base >> PAGE_BITS) as u64) << META_RUN_COUNT_BITS) | self.pages as u64
    }
    const fn unpack(word: u64) -> Self {
        Self {
            base: ((word >> META_RUN_COUNT_BITS) as usize) << PAGE_BITS,
            pages: (word & ((1 << META_RUN_COUNT_BITS) - 1)) as usize,
        }
    }
}

/// **The metadata reserve**: runs of pages a huge grow over-asked and did
/// not need, held for the nodes and directory pages of later grows so
/// that those ask their source for their payload alone.  Lock-free: a run
/// is claimed from, or offered into, its slot by compare-and-swap.
struct MetaReserve {
    runs: [AtomicU64; META_RESERVE_RUNS],
}

impl MetaReserve {
    const fn new() -> Self {
        Self {
            runs: [const { AtomicU64::new(0) }; META_RESERVE_RUNS],
        }
    }

    /// Move up to `pages` pages into `budget`, draining the reserve's runs
    /// in slot order; what it holds short of that, the grow over-asks.
    fn claim(&self, mut pages: usize, budget: &mut MetaBudget) {
        for slot in &self.runs {
            let mut word = slot.load(Ordering::Acquire);
            while pages > 0 {
                let run = MetaRun::unpack(word);
                if run.pages == 0 {
                    break;
                }
                let take = run.pages.min(pages);
                let rest = MetaRun {
                    base: run.base + (take << PAGE_BITS),
                    pages: run.pages - take,
                };
                let next = if rest.pages == 0 { 0 } else { rest.pack() };
                match slot.compare_exchange_weak(word, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        if budget.add(MetaRun {
                            base: run.base,
                            pages: take,
                        }) {
                            pages -= take;
                        } else {
                            // No room in the budget: hand the run back.
                            let _ = self.offer(MetaRun {
                                base: run.base,
                                pages: take,
                            });
                            return;
                        }
                        break;
                    }
                    Err(actual) => word = actual,
                }
            }
            if pages == 0 {
                return;
            }
        }
    }

    /// Hold `run` for later grows; `false` when every slot is taken, and
    /// the run is this heap's lost pages.
    fn offer(&self, run: MetaRun) -> bool {
        if run.pages == 0 {
            return true;
        }
        if run.pages >= 1 << META_RUN_COUNT_BITS {
            return false;
        }
        self.runs.iter().any(|slot| {
            slot.compare_exchange(0, run.pack(), Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        })
    }

    /// Pages held: what the suite checks a grow left behind.
    #[cfg(test)]
    fn pages(&self) -> usize {
        self.runs
            .iter()
            .map(|slot| MetaRun::unpack(slot.load(Ordering::Acquire)).pages)
            .sum()
    }
}

/// The pages one grow may spend on metadata: runs claimed from the
/// reserve, the pages a lost publication race hands back, and — last —
/// the tail of the grow's own mapping down to a floor that keeps its
/// payload whole.  Local to the grow, so no atomics; what is left when
/// the grow is done goes back to the reserve.
struct MetaBudget {
    runs: [MetaRun; META_BUDGET_RUNS],
    len: usize,
    /// The grow's own mapping: pages are carved downward from `end`,
    /// never below `floor`.
    tail_end: usize,
    tail_floor: usize,
}

/// Reserve runs, the tail, and a page per lost race at each level.
const META_BUDGET_RUNS: usize = META_RESERVE_RUNS + PAGE_MAP_LEVELS + 1;

impl MetaBudget {
    const fn new() -> Self {
        Self {
            runs: [MetaRun { base: 0, pages: 0 }; META_BUDGET_RUNS],
            len: 0,
            tail_end: 0,
            tail_floor: 0,
        }
    }

    /// Let the budget carve `[floor, end)` of the grow's own mapping.
    fn with_tail(&mut self, floor: usize, end: usize) {
        self.tail_floor = floor;
        self.tail_end = end;
    }

    /// Pages in hand, the tail excluded.
    fn pages(&self) -> usize {
        self.runs[..self.len].iter().map(|run| run.pages).sum()
    }

    fn add(&mut self, run: MetaRun) -> bool {
        if run.pages == 0 {
            return true;
        }
        if self.len == META_BUDGET_RUNS {
            return false;
        }
        self.runs[self.len] = run;
        self.len += 1;
        true
    }

    /// One page: from a run in hand first, else off the tail.
    fn take(&mut self) -> Option<usize> {
        while self.len > 0 {
            let run = &mut self.runs[self.len - 1];
            if run.pages == 0 {
                self.len -= 1;
                continue;
            }
            let page = run.base;
            run.base += EXTENT_ALIGN;
            run.pages -= 1;
            if run.pages == 0 {
                self.len -= 1;
            }
            return Some(page);
        }
        let end = self.tail_end.checked_sub(EXTENT_ALIGN)?;
        if end < self.tail_floor {
            return None;
        }
        self.tail_end = end;
        Some(end)
    }

    /// A page back — from a lost publication race, or carved for a node
    /// another grower published first — merged into the run it adjoins
    /// when it adjoins one.
    fn put(&mut self, page: usize) {
        for run in &mut self.runs[..self.len] {
            if page == run.base + (run.pages << PAGE_BITS) {
                run.pages += 1;
                return;
            }
            if page + EXTENT_ALIGN == run.base {
                run.base = page;
                run.pages += 1;
                return;
            }
        }
        let _ = self.add(MetaRun {
            base: page,
            pages: 1,
        });
    }

    /// Where the grow's mapping now ends: its payload and whatever tail
    /// the metadata did not take.
    fn tail_end(&self) -> usize {
        self.tail_end
    }

    /// Hand every run still in hand to the reserve.
    fn release(&mut self, reserve: &MetaReserve) {
        for run in &self.runs[..self.len] {
            let _ = reserve.offer(*run);
        }
        self.len = 0;
    }
}

/// The sharded, lock-free global allocator.  Generic over its [`MemSource`]
/// backing — the engine names no concrete source; founded binaries link
/// [`Heap<HeapSupplySource>`](HeapSupplySource).
pub struct Heap<S: MemSource> {
    /// Per-task allocator lanes.
    shards: [Shard; NUM_SHARDS],
    /// The extent directory's first page, inline: a `static` heap registers
    /// its first [`PAGE_SLOTS`] extents without mapping anything.
    page0: DirPage,
    /// The directory's later pages, each mapped from the source when every
    /// earlier page is full; `null` until published (`Release` by the
    /// installing compare-and-swap, read `Acquire`).  Pages are installed in
    /// order, so the live pages are always a prefix.
    pages: [AtomicPtr<DirPage>; DIR_PAGES - 1],
    /// The page map's root: the deallocate route.  Each slot covers
    /// 2^39 bytes of address space through three node levels below it.
    page_map: [AtomicPtr<PageMapNode>; PAGE_MAP_ROOT_FANOUT],
    /// Pages held for later grows' metadata ([`MetaReserve`]).
    reserve: MetaReserve,
    /// Free huge extents for recycling, by size bucket ([`HUGE_BUCKETS`]):
    /// each a generation-tagged Treiber stack of directory ids (the
    /// board.rs scheme — the generation defeats ABA on the multi-consumer
    /// pop).
    huge_free: [AtomicU64; HUGE_BUCKETS],
    /// Where fresh extents come from.
    source: S,
}

// SAFETY: cross-thread state is all atomic — the directory pages (their slot
// pointers, free ledgers and high-water marks, and the page pointers that
// publish them), the `huge_free` buckets, each shard's remote free-lists, and the pool
// carve cursors.  The only non-atomic state — each shard's local free-lists and its
// current-extent pointer — is touched solely by that shard's owning task, per
// `current_shard()`'s injective task→shard map (see `shard.rs`).  No global lock
// sits on the hot path, so a `#[global_allocator] static` is sound.  In
// debug/test that injectivity is enforced at runtime by `Shard::owner_guard`
// (a panic if two threads ever enter one shard's owner-only region).
unsafe impl<S: MemSource + Sync> Sync for Heap<S> {}

impl<S: MemSource> Heap<S> {
    /// A heap over an explicit [`MemSource`] — the host-test entry
    /// (`Heap::new_in(StaticMemSource::over(..))`) and the const path for a
    /// binary that supplies its own arena.
    pub const fn new_in(source: S) -> Self {
        Self {
            shards: [const { Shard::new() }; NUM_SHARDS],
            page0: DirPage::new(),
            pages: [const { AtomicPtr::new(core::ptr::null_mut()) }; DIR_PAGES - 1],
            page_map: [const { AtomicPtr::new(core::ptr::null_mut()) }; PAGE_MAP_ROOT_FANOUT],
            reserve: MetaReserve::new(),
            huge_free: [const { AtomicU64::new(free_pack(0, FREE_EMPTY)) }; HUGE_BUCKETS],
            source,
        }
    }

    /// The backing source, for sources provisioned outside synchronous
    /// allocation (for example an async memory-service refill after a miss).
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Directory page `index`, or `None` past the published prefix.
    #[inline]
    fn page(&self, index: usize) -> Option<&DirPage> {
        if index == 0 {
            return Some(&self.page0);
        }
        let p = self.pages.get(index - 1)?.load(Ordering::Acquire);
        // SAFETY: a published page pointer names a `DirPage` written in place
        // before the `Release` publication, held for the heap's life.
        (!p.is_null()).then(|| unsafe { &*p })
    }

    /// The directory slot of extent id `id` (a published id — every page an
    /// id names is installed before any extent in it is).
    #[inline]
    fn slot(&self, id: u32) -> &AtomicPtr<Extent> {
        let id = id as usize;
        let page = self
            .page(id / PAGE_SLOTS)
            .expect("a published extent id names an installed directory page");
        &page.slots[id % PAGE_SLOTS]
    }

    /// Occupancy snapshot: a cold scan of the directory, page by page.
    /// `mapped_bytes` is every byte taken from the source — each extent's
    /// whole mapping, the metadata carved from its tail and the pages it
    /// left to the reserve included — so it bounds what the source gave
    /// out; `extent_count` is the live extents.
    pub fn stats(&self) -> HeapStats {
        let mut mapped_bytes = 0;
        let mut extent_count = 0;
        for index in 0..DIR_PAGES {
            let Some(page) = self.page(index) else {
                break;
            };
            let n = page.high_water.load(Ordering::Acquire).min(PAGE_SLOTS);
            for slot in &page.slots[..n] {
                let p = slot.load(Ordering::Acquire);
                if p.is_null() {
                    continue;
                }
                // SAFETY: a published pointer names a live `Extent` held for
                // the heap's life.
                let ext = unsafe { &*p };
                mapped_bytes += ext.mapped;
                extent_count += 1;
            }
        }
        HeapStats {
            mapped_bytes,
            extent_count,
        }
    }

    // ── The fallible core (both faces route through here) ─────────────────────

    /// Allocate `layout`, or `Err(AllocError)` on exhaustion — **never aborts**.
    /// [`GlobalAlloc::alloc`] maps the `Err` to null (its only failure signal);
    /// [`Allocator::allocate`] propagates it.
    fn try_alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (size, align) = (layout.size(), layout.align());
        if size == 0 {
            // Allocator contract: a unique, `align`-aligned, zero-length slice,
            // with no backing allocation.  `align` is a non-zero power of two, so
            // the address is a valid dangling aligned pointer (never dereferenced).
            let dangling = align as *mut u8;
            // SAFETY: `align >= 1`, so `dangling` is non-null.
            return Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(dangling) },
                0,
            ));
        }
        let need = size.max(align);
        if is_huge(need) {
            return self.huge_alloc(size, align);
        }
        let class = class_index(need);
        if align > block_align(class) {
            // A large class's blocks are page-aligned; the huge path aligns
            // any payload.  The free routes by the extent, not the layout.
            return self.huge_alloc(size, align);
        }
        let shard = &self.shards[current_shard()];
        // Hold the shard's owner guard across the whole owner-only region (local
        // pop, then carve/grow in `alloc_slow`) — acquired once here, not per
        // inner call, so the `reclaim_remote` → `local_push` nesting doesn't
        // self-trip.  Debug/test-only detector (a loud panic if `current_shard()`
        // ever collocates two threads); zero-cost in release.
        let _owner = shard.owner_guard();
        // Hot path: pop the owner's local free-list (no atomics).
        // SAFETY: `current_shard()` names this task's shard — its sole accessor
        // (enforced by `_owner` in debug builds).
        let block = unsafe { shard.local_pop(class) };
        if !block.is_null() {
            return Ok(block_slice(block, class));
        }
        self.alloc_slow(shard, class)
    }

    /// The miss path: reclaim cross-thread frees, then carve, then grow.
    #[cold]
    fn alloc_slow(&self, shard: &Shard, class: usize) -> Result<NonNull<[u8]>, AllocError> {
        loop {
            // 1. Fold this shard's cross-thread frees back into the local list.
            // SAFETY: the owner (current task) is the remote list's sole consumer.
            if unsafe { shard.reclaim_remote(class) } {
                // SAFETY: owner-only local access.
                let block = unsafe { shard.local_pop(class) };
                if !block.is_null() {
                    return Ok(block_slice(block, class));
                }
            }
            // 2. Carve a fresh block from the shard's current pool extent.
            if let Some(block) = self.carve(shard, class) {
                return Ok(block_slice(block, class));
            }
            // 3. Grow through this thread's independent miss lane and retry,
            // or surface exhaustion.
            match self.grow_pool(shard, class) {
                GrowOutcome::Grew => continue,
                GrowOutcome::Exhausted => return Err(AllocError),
            }
        }
    }

    /// Bump one `class`-sized block off the shard's current pool extent, aligning
    /// the block's **absolute address** to [`block_align`] — the class size in
    /// the shared lane, a page in a large class's — which `try_alloc` has
    /// checked is `>= align`, so the returned pointer honors the request's
    /// alignment.
    fn carve(&self, shard: &Shard, class: usize) -> Option<*mut u8> {
        // SAFETY: owner reads its own current-extent pointer.
        let ext_ptr = unsafe { shard.current(carve_lane(class)) };
        if ext_ptr.is_null() {
            return None;
        }
        // SAFETY: a published extent pointer names a live `Extent`.
        let ext = unsafe { &*ext_ptr };
        let sz = class_size(class);
        // Owner-only cursor: a plain load/store pair, no CAS (no rival writer).
        let cursor = ext.cursor.load(Ordering::Relaxed);
        let block = checked_align_up(cursor, block_align(class))?;
        let end = block.checked_add(sz)?;
        if end > ext.end() {
            return None; // this extent is spent for `class`
        }
        ext.cursor.store(end, Ordering::Relaxed);
        Some(block as *mut u8)
    }

    // ── The grow path ─────────────────────────────────────────────────────

    /// The extent a grow maps for `class` in `shard`: the shared lane's
    /// [`POOL_EXTENT_BYTES`]; for a large class, its header page and its
    /// blocks — one when the lane has no extent yet, else twice what the
    /// lane's current extent holds, up to [`LARGE_CLASS_MAX_BLOCKS`].
    /// Derived from the lane's own extent, so a refused grow asks the same
    /// size again on its retry.
    fn pool_extent_bytes(shard: &Shard, class: usize) -> usize {
        let lane = carve_lane(class);
        if lane == 0 {
            return POOL_EXTENT_BYTES;
        }
        let size = class_size(class);
        // SAFETY: the owner reads its own current-extent pointer.
        let current = unsafe { shard.current(lane) };
        let blocks = if current.is_null() {
            1
        } else {
            // SAFETY: a published extent pointer names a live `Extent`,
            // whose geometry is immutable.
            let held = unsafe { &*current }.len().saturating_sub(EXTENT_ALIGN) / size;
            held.saturating_mul(2).clamp(1, LARGE_CLASS_MAX_BLOCKS)
        };
        EXTENT_ALIGN + blocks * size
    }

    /// Map and install one fresh pool extent for `shard`'s lane of `class`:
    /// one mapping of [`pool_extent_bytes`](Self::pool_extent_bytes), so a
    /// source that answers a miss with one extent of what was asked serves
    /// the grow exactly.  The shared lane carves the grow's metadata off
    /// its own tail (down to half the extent, which the bound never
    /// reaches) and keeps whatever tail is left as carve space.  A large
    /// class's extent is its header page and whole blocks: a page carved
    /// off its tail would cost it a block, and pages left over would never
    /// hold one, so it over-asks by what the reserve is short, as a huge
    /// grow does, keeps exactly its blocks, and leaves the reserve the rest.
    ///
    /// Every live thread owns a distinct shard and distinct session scratch, so
    /// concurrent misses may map concurrently.  Directory/source reservation is
    /// already atomic; a process-wide grow flag would only create preemption
    /// inversion around a synchronous kernel session.
    #[cold]
    fn grow_pool(&self, shard: &Shard, class: usize) -> GrowOutcome {
        let bytes = Self::pool_extent_bytes(shard, class);
        let mut meta = MetaBudget::new();
        let bound = meta_pages_bound(bytes);
        self.reserve.claim(bound, &mut meta);
        let (short, payload) = if carve_lane(class) == 0 {
            (0, Payload::Rest)
        } else {
            (bound.saturating_sub(meta.pages()), Payload::Exact)
        };
        let want = match short
            .checked_mul(EXTENT_ALIGN)
            .and_then(|extra| bytes.checked_add(extra))
        {
            Some(want) => want,
            None => return GrowOutcome::Exhausted,
        };
        let outcome = match self.source.map_extent(want) {
            None => GrowOutcome::Exhausted,
            Some(mem) => {
                let base = mem.cast::<u8>().as_ptr() as usize;
                let total = mem.len();
                match self.install_extent(
                    base,
                    total,
                    bytes,
                    payload,
                    ExtentKind::Pool,
                    current_shard(),
                    &mut meta,
                ) {
                    Ok(ext_ptr) => {
                        // SAFETY: owner sets its own current-extent pointer.
                        unsafe { shard.set_current(carve_lane(class), ext_ptr) };
                        GrowOutcome::Grew
                    }
                    // An unsafe source violated its geometry contract, or the
                    // directory is at its ceiling.  Nothing is published; the
                    // mapping cannot be returned through this one-way seam.
                    Err(()) => GrowOutcome::Exhausted,
                }
            }
        };
        meta.release(&self.reserve);
        outcome
    }

    /// Claim one slot in `page` (directory page `index`) without mapping.  The
    /// bitset's failed-claim rollback makes concurrent scans lock-free; a
    /// bounded retry gives a later allocation another chance if this snapshot
    /// lost every observed bit.  `None`: the page is full.
    fn claim_in<'a>(page: &'a DirPage, index: usize) -> Option<DirectoryClaim<'a>> {
        for _ in 0..PAGE_SLOTS {
            let slot = page.free.find_first_set_from(0)?;
            if page.free.try_claim_run(slot, 1).is_ok() {
                return Some(DirectoryClaim {
                    page,
                    slot,
                    id: (index * PAGE_SLOTS + slot) as u32,
                    published: false,
                });
            }
        }
        None
    }

    /// Claim one directory slot: the first free slot in the first page
    /// that has one, installing the next page — written in place on a
    /// page from `meta` (#205/§9.4: metadata self-hosted in source
    /// memory, never allocated from the heap) and published by
    /// compare-and-swap — when every installed page is full.  Two threads
    /// growing at once race on the page pointer; the loser's page goes
    /// back into its budget.  `None`: the [`DIR_PAGES`] ceiling, or no
    /// page to install one with.
    fn claim_directory(&self, meta: &mut MetaBudget) -> Option<DirectoryClaim<'_>> {
        for index in 0..DIR_PAGES {
            let page = match self.page(index) {
                Some(page) => page,
                None => {
                    let fresh = meta.take()? as *mut DirPage;
                    // SAFETY: a fresh, exclusively owned, page-aligned page
                    // (`DirPage` fits one); written before any publication.
                    unsafe { fresh.write(DirPage::new()) };
                    match self.pages[index - 1].compare_exchange(
                        core::ptr::null_mut(),
                        fresh,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        // SAFETY: just published, held for the heap's life.
                        Ok(_) => unsafe { &*fresh },
                        Err(installed) => {
                            meta.put(fresh as usize);
                            // SAFETY: the winner's page, published `Release`.
                            unsafe { &*installed }
                        }
                    }
                }
            };
            if let Some(claim) = Self::claim_in(page, index) {
                return Some(claim);
            }
        }
        None
    }

    /// The published node `level` steps below the root on `addr`'s path
    /// (`1`: a root slot's node; `PAGE_MAP_LEVELS - 1`: the leaf).
    fn node_at(&self, addr: usize, level: usize) -> Option<&PageMapNode> {
        let path = page_map_path(addr)?;
        let mut node = self.page_map[path[0]].load(Ordering::Acquire);
        for &index in &path[1..level] {
            if node.is_null() {
                return None;
            }
            // SAFETY: a published node pointer names a live `PageMapNode`.
            node = unsafe { &*node }.slots[index].load(Ordering::Acquire);
        }
        // SAFETY: as above.
        (!node.is_null()).then(|| unsafe { &*node })
    }

    /// The page-map nodes `[base, base + len)` lacks: at each level, those
    /// of its spans whose node is not published.  A concurrent grower can
    /// only publish more of them, so the count is an upper bound by the
    /// time the pages are carved.
    fn missing_nodes(&self, base: usize, len: usize) -> usize {
        let last = base + len - 1;
        let mut missing = 0;
        for level in 1..PAGE_MAP_LEVELS {
            let shift = level_shift(level);
            for span in (base >> shift)..=(last >> shift) {
                if self.node_at(span << shift, level).is_none() {
                    missing += 1;
                }
            }
        }
        missing
    }

    /// The node below `slot`, installing one there if it is still empty on
    /// a page from `meta`, written in place and published by
    /// compare-and-swap; a lost race puts the page back.  `None`: the slot
    /// is empty and the budget is spent, which the bound rules out.
    fn descend<'a>(
        &self,
        slot: &'a AtomicPtr<PageMapNode>,
        meta: &mut MetaBudget,
    ) -> Option<&'a PageMapNode> {
        let mut node = slot.load(Ordering::Acquire);
        if node.is_null() {
            let fresh = meta.take()? as *mut PageMapNode;
            // SAFETY: a fresh, exclusively owned, page-aligned page (a node
            // fits one); written before any publication.
            unsafe { fresh.write(PageMapNode::new()) };
            node = match slot.compare_exchange(
                core::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => fresh,
                Err(installed) => {
                    meta.put(fresh as usize);
                    installed
                }
            };
        }
        // SAFETY: a published node pointer names a `PageMapNode` written in
        // place before its `Release` publication, held for the heap's life.
        Some(unsafe { &*node })
    }

    /// Enter every page of `[base, base + len)` into the page map as owned
    /// by `ext`, installing the nodes the span lacks from `meta`.  Pages of
    /// one extent are claimed by nobody else (the source's disjointness),
    /// so the leaf stores race with nothing; a reader that sees the
    /// extent's slot `null` is routing a pointer that was never handed
    /// out.  The caller has checked the span lies within the map's reach
    /// and budgeted its nodes; `Err`: the budget ran out regardless.
    fn register_pages(
        &self,
        base: usize,
        len: usize,
        ext: *mut Extent,
        meta: &mut MetaBudget,
    ) -> Result<(), ()> {
        let mut page = base;
        let end = align_up(base + len, EXTENT_ALIGN);
        while page < end {
            let path = page_map_path(page).ok_or(())?;
            let mut slot = &self.page_map[path[0]];
            for &index in &path[1..PAGE_MAP_LEVELS - 1] {
                let node = self.descend(slot, meta).ok_or(())?;
                slot = &node.slots[index];
            }
            let leaf = self.descend(slot, meta).ok_or(())?;
            // Consecutive pages share a leaf until its span ends: fill the
            // leaf's run in one pass rather than walking per page.
            let first = path[PAGE_MAP_LEVELS - 1];
            let run = ((end - page) >> PAGE_BITS).min(PAGE_MAP_FANOUT - first);
            for index in first..first + run {
                leaf.slots[index].store(ext.cast::<PageMapNode>(), Ordering::Release);
            }
            page += run << PAGE_BITS;
        }
        Ok(())
    }

    /// The extent owning the page `addr` lies in — the deallocate route, four
    /// loads — or `None` for an address no registered extent covers.
    fn route(&self, addr: usize) -> Option<&Extent> {
        let path = page_map_path(addr)?;
        let mut node = self.page_map[path[0]].load(Ordering::Acquire);
        for &index in &path[1..] {
            if node.is_null() {
                return None;
            }
            // SAFETY: a published node pointer names a live `PageMapNode`.
            node = unsafe { &*node }.slots[index].load(Ordering::Acquire);
        }
        // SAFETY: a leaf slot holds a published `Extent` pointer or null.
        (!node.is_null()).then(|| unsafe { &*node.cast::<Extent>() })
    }

    /// Consume the fresh mapping `[base, base + total)` into a published
    /// [`Extent`] of at least `min_len` payload bytes: validate the
    /// geometry, claim a directory slot, carve the metadata the span needs
    /// (from `meta`, then the mapping's tail, down to where `payload`
    /// says), self-host the header, enter the pages into the page map, and
    /// `Release`-publish.  A concurrent reader sees the slot `null` until
    /// the complete header is visible.  What the metadata leaves of the
    /// tail is carve space or goes to the reserve, as `payload` says.
    /// `Err`: malformed geometry (refused before any write), an address
    /// past the map's reach, the directory ceiling, or a budget the bound
    /// did not cover — the mapping is lost, nothing reachable names it.
    #[allow(clippy::too_many_arguments)]
    fn install_extent(
        &self,
        base: usize,
        total: usize,
        min_len: usize,
        payload: Payload,
        kind: ExtentKind,
        owner: usize,
        meta: &mut MetaBudget,
    ) -> Result<*mut Extent, ()> {
        let floor = match payload {
            Payload::Rest => min_len / 2,
            Payload::Exact => min_len,
        };
        if base % EXTENT_ALIGN != 0
            || total % EXTENT_ALIGN != 0
            || total < min_len.max(extent_header_bytes())
            || total > isize::MAX as usize
            || base
                .checked_add(total - 1)
                .and_then(page_map_path)
                .is_none()
        {
            return Err(());
        }
        meta.with_tail(base + floor, base + total);
        let claim = self.claim_directory(meta).ok_or(())?;
        // What the extent will cover — all the metadata leaves it, or its
        // page-rounded payload — and the nodes that span lacks, counted on
        // its widest form: carving them off the tail only shrinks it, and
        // the bound covers the payload's page rounding.
        let span = match payload {
            Payload::Rest => meta.tail_end() - base,
            Payload::Exact => align_up(min_len, EXTENT_ALIGN),
        };
        let nodes = self.missing_nodes(base, span);
        let mut stock = MetaBudget::new();
        for _ in 0..nodes {
            let page = meta.take().ok_or(())?;
            stock.put(page);
        }
        let end = meta.tail_end();
        let len = match payload {
            Payload::Rest => end - base,
            Payload::Exact => {
                // The over-ask the metadata did not take goes to the
                // reserve.  The span was budgeted for the payload alone, so
                // when the reserve has no room (sixteen runs held already)
                // those pages are this heap's lost ones, never its extent's.
                let payload_end = base + span;
                if end > payload_end {
                    let _ = self.reserve.offer(MetaRun {
                        base: payload_end,
                        pages: (end - payload_end) >> PAGE_BITS,
                    });
                }
                span
            }
        };
        let id = claim.id();
        let ext_ptr = base as *mut Extent;
        // SAFETY: `base` is a fresh page-aligned mapping of `len >= header`
        // bytes, owned exclusively here (not yet published), so the header
        // write is sound and lands before the `Release` store below publishes
        // the pointer.
        unsafe {
            ext_ptr.write(Extent {
                base,
                len,
                mapped: total,
                owner,
                kind,
                dir_idx: id,
                cursor: AtomicUsize::new(base + extent_header_bytes()),
                next_free: AtomicU32::new(0),
            });
        }
        let registered = self.register_pages(base, len, ext_ptr, &mut stock);
        // Node pages a concurrent grower made unnecessary go back to the
        // grow's budget, and from there to the reserve.
        while let Some(page) = stock.take() {
            meta.put(page);
        }
        registered?;
        claim.page.slots[claim.slot].store(ext_ptr, Ordering::Release);
        claim
            .page
            .high_water
            .fetch_max(claim.slot + 1, Ordering::Release);
        claim.publish();
        Ok(ext_ptr)
    }

    /// Return `layout`'s block to its owner — **never leaks**.  Routes the
    /// pointer to its extent through the page map (O(1)), then: a huge extent
    /// goes onto the recycle stack; a pool block goes to the owning shard's
    /// local list (if we are the owner) or its MPSC remote list (if not).
    ///
    /// # Safety
    /// `ptr`/`layout` must name a live allocation this heap produced.
    unsafe fn dealloc_ptr(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return; // zero-size: a dangling pointer, nothing was mapped
        }
        let addr = ptr.as_ptr() as usize;
        let Some(ext) = self.route(addr) else {
            debug_assert!(
                false,
                "kore-rt heap: free of a pointer this heap never handed out"
            );
            return;
        };
        debug_assert!(ext.contains(addr));
        match ext.kind {
            ExtentKind::Huge => self.huge_push(huge_bucket(ext.len()), ext.dir_idx),
            ExtentKind::Pool => {
                let class = class_index(layout.size().max(layout.align()));
                let owner = ext.owner;
                if owner == current_shard() {
                    // Owner-only local push — same debug single-owner guard as
                    // the alloc path.
                    let _owner = self.shards[owner].owner_guard();
                    // SAFETY: we are the owner; `ptr` is an unaliased block of `class`.
                    unsafe { self.shards[owner].local_push(class, ptr.as_ptr()) };
                } else {
                    // SAFETY: cross-thread free; MPSC remote push of an unaliased block.
                    unsafe { self.shards[owner].remote_push(class, ptr.as_ptr()) };
                }
            }
        }
    }

    // ── The huge path (whole-extent allocations, recycled) ────────────────────

    /// Serve a `> LARGE_THRESHOLD` request: the best recycled extent that
    /// fits, whole ([`HUGE_BUCKETS`]), else map fresh `header + align + size`
    /// backing.
    #[cold]
    fn huge_alloc(&self, size: usize, align: usize) -> Result<NonNull<[u8]>, AllocError> {
        let bytes = extent_header_bytes()
            .checked_add(align)
            .and_then(|head| head.checked_add(size))
            .ok_or(AllocError)?;
        let needed = checked_align_up(bytes, EXTENT_ALIGN).ok_or(AllocError)?;
        let first = huge_bucket(needed);
        let mut recycled = None;
        for bucket in first..HUGE_BUCKETS {
            if bucket == first {
                // The request's own bucket: a fit is likely, not certain.
                // Search a bounded prefix, restoring the skipped entries in
                // reverse pop order so the stack keeps its relative order
                // and every off-stack extent stays exclusively held.
                let mut deferred = [FREE_EMPTY; HUGE_RECYCLE_SCAN];
                let mut deferred_len = 0;
                while deferred_len < HUGE_RECYCLE_SCAN {
                    let Some(id) = self.huge_pop(bucket) else {
                        break;
                    };
                    // SAFETY: an id off a huge-free stack names a published Huge extent.
                    let ext = unsafe { &*self.slot(id).load(Ordering::Acquire) };
                    if ext.len() >= needed {
                        recycled = Some(id);
                        break;
                    }
                    deferred[deferred_len] = id;
                    deferred_len += 1;
                }
                for &id in deferred[..deferred_len].iter().rev() {
                    self.huge_push(bucket, id);
                }
            } else if let Some(id) = self.huge_pop(bucket) {
                // Every extent of a larger bucket is at least twice the
                // request's bucket floor, so it fits.
                recycled = Some(id);
            }
            if recycled.is_some() {
                break;
            }
        }
        if let Some(id) = recycled {
            // SAFETY: a popped id names a published Huge extent, held
            // exclusively by this thread until it is handed out.
            let ext = unsafe { &*self.slot(id).load(Ordering::Acquire) };
            return huge_slice(ext, size, align).ok_or(AllocError);
        }

        // The metadata the span may need comes from the reserve, else is
        // over-asked alongside the payload: the grow completes with the one
        // mapping it asks for, and leaves the reserve what it did not use.
        let bound = meta_pages_bound(bytes);
        let mut meta = MetaBudget::new();
        self.reserve.claim(bound, &mut meta);
        let short = bound.saturating_sub(meta.pages());
        let served = short
            .checked_mul(EXTENT_ALIGN)
            .and_then(|extra| checked_align_up(bytes, EXTENT_ALIGN)?.checked_add(extra))
            .and_then(|want| self.source.map_extent(want))
            .ok_or(())
            .and_then(|mem| {
                let base = mem.cast::<u8>().as_ptr() as usize;
                self.install_extent(
                    base,
                    mem.len(),
                    bytes,
                    Payload::Exact,
                    ExtentKind::Huge,
                    usize::MAX,
                    &mut meta,
                )
            });
        meta.release(&self.reserve);
        let ext_ptr = served.map_err(|()| AllocError)?;
        // SAFETY: just published.
        let ext = unsafe { &*ext_ptr };
        huge_slice(ext, size, align).ok_or(AllocError)
    }

    /// Push a huge extent (by directory id) onto recycle `bucket`.  The
    /// generation bump defeats the ABA a racing [`huge_pop`](Self::huge_pop)
    /// would otherwise admit.
    fn huge_push(&self, bucket: usize, id: u32) {
        // SAFETY: `id` is a published extent's directory slot.
        let ext = unsafe { &*self.slot(id).load(Ordering::Acquire) };
        let stack = &self.huge_free[bucket];
        let mut head = stack.load(Ordering::Acquire);
        loop {
            ext.next_free.store(free_id(head), Ordering::Relaxed);
            let new = free_pack(free_gen(head).wrapping_add(1), id);
            match stack.compare_exchange_weak(head, new, Ordering::Release, Ordering::Acquire) {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
        }
    }

    /// Pop a recycled huge extent (by directory id) off `bucket`, or `None`
    /// when it is empty.  Multi-consumer safe: the generation in the CAS'd
    /// word rejects a stale head.
    fn huge_pop(&self, bucket: usize) -> Option<u32> {
        let stack = &self.huge_free[bucket];
        let mut head = stack.load(Ordering::Acquire);
        loop {
            let id = free_id(head);
            if id == FREE_EMPTY {
                return None;
            }
            // SAFETY: an id on the stack was pushed for a published extent.
            let ext = unsafe { &*self.slot(id).load(Ordering::Acquire) };
            let next = ext.next_free.load(Ordering::Relaxed);
            let new = free_pack(free_gen(head).wrapping_add(1), next);
            match stack.compare_exchange_weak(head, new, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(id),
                Err(actual) => head = actual,
            }
        }
    }
}

/// Wrap a live pool block as a returned slice — usable length is the whole class
/// size (`>= layout.size()`), per the [`Allocator`] contract.
#[inline]
fn block_slice(block: *mut u8, class: usize) -> NonNull<[u8]> {
    // SAFETY: `block` is a live, non-null block of `class`.
    NonNull::slice_from_raw_parts(unsafe { NonNull::new_unchecked(block) }, class_size(class))
}

/// The aligned payload slice inside a huge `ext` for a `(size, align)` request,
/// or `None` if this extent can't fit it (a recycled extent too small/mis-aligned
/// for the new request).  Usable length runs to the end of the extent.
#[inline]
fn huge_slice(ext: &Extent, size: usize, align: usize) -> Option<NonNull<[u8]>> {
    let after_header = ext.base.checked_add(extent_header_bytes())?;
    let payload = checked_align_up(after_header, align)?;
    match payload.checked_add(size) {
        Some(end) if end <= ext.end() => Some(NonNull::slice_from_raw_parts(
            // SAFETY: `payload >= base + header >= 1`, so non-null.
            unsafe { NonNull::new_unchecked(payload as *mut u8) },
            ext.end() - payload,
        )),
        _ => None,
    }
}

// ── The two trait faces ──────────────────────────────────────────────────────

// SAFETY: `try_alloc` returns blocks of `>= layout.size()` bytes aligned to
// `layout.align()` (a block's [`block_align`], which `try_alloc` checks covers
// it, else the huge payload's alignment), and
// `dealloc_ptr` returns exactly such blocks to their owner; a null return is the
// sole failure signal, which the caller's `handle_alloc_error` turns into an
// abort — the intended `GlobalAlloc` semantics.
unsafe impl<S: MemSource> GlobalAlloc for Heap<S> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.try_alloc(layout)
            .map_or(core::ptr::null_mut(), |p| p.as_ptr().cast())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(p) = NonNull::new(ptr) {
            // SAFETY: `GlobalAlloc` contract — `ptr`/`layout` name a live block
            // from this allocator.
            unsafe { self.dealloc_ptr(p, layout) };
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Pool blocks are recycled dirt — zero explicitly.
        match self.try_alloc(layout) {
            Ok(p) => {
                let ptr = p.as_ptr() as *mut u8;
                // SAFETY: `p` is a fresh allocation of `>= layout.size()` bytes.
                unsafe { core::ptr::write_bytes(ptr, 0, layout.size()) };
                ptr
            }
            Err(_) => core::ptr::null_mut(),
        }
    }
}

// SAFETY: same block guarantees as the `GlobalAlloc` face; here the `AllocError`
// is *propagated* (never an abort), and the returned slice length is the actual
// usable size.  Allocated blocks keep their contents/alignment until deallocated.
unsafe impl<S: MemSource> Allocator for Heap<S> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.try_alloc(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: `Allocator` contract — `ptr`/`layout` name a live block from
        // this allocator.
        unsafe { self.dealloc_ptr(ptr, layout) };
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let p = self.try_alloc(layout)?;
        // SAFETY: `p` is a fresh allocation of `p.len() >= layout.size()` bytes.
        unsafe { core::ptr::write_bytes(p.as_ptr() as *mut u8, 0, p.len()) };
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::sync::atomic::AtomicPtr;

    /// A heap over a leaked host arena — bytes live for the whole test process,
    /// so the raw-pointer source never dangles.
    fn test_heap(arena_bytes: usize) -> Heap<StaticMemSource> {
        let buf: &'static mut [u8] = alloc::vec![0u8; arena_bytes].leak();
        // SAFETY: `buf` is leaked (process-lived) and owned solely by the source.
        let src = unsafe { StaticMemSource::over(buf.as_mut_ptr(), arena_bytes) };
        Heap::new_in(src)
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).expect("valid layout")
    }

    #[test]
    fn alloc_dealloc_round_trip_recycles_blocks() {
        let heap = test_heap(EXTENT_ALIGN * 40);
        let l = layout(48, 8);
        let a = heap.allocate(l).expect("first");
        let a_ptr = a.as_ptr() as *mut u8;
        assert!(a.len() >= 48, "usable size covers the ask");
        // The block is real, writable memory.
        unsafe { core::ptr::write_bytes(a_ptr, 0xAB, 48) };
        assert_eq!(unsafe { a_ptr.read() }, 0xAB);
        // Free then re-allocate the same class: the freed block comes back (proof
        // the free returned it to the list, not a leak).
        unsafe { heap.deallocate(NonNull::new_unchecked(a_ptr), l) };
        let b = heap.allocate(l).expect("second");
        assert_eq!(b.as_ptr() as *mut u8, a_ptr, "freed block recycled");
        unsafe { heap.deallocate(b.cast::<u8>(), l) };
    }

    #[test]
    fn alignment_is_honored_across_classes() {
        let heap = test_heap(EXTENT_ALIGN * 64);
        // Spans small classes, a page alignment a small class honours, and an
        // alignment past a page in a large class (16384 → the huge path aligns
        // the extent).
        for &(size, align) in &[
            (1usize, 1usize),
            (8, 8),
            (16, 16),
            (10, 64),
            (200, 256),
            (17, 1024),
            (64, 4096),
            (32, 16384),
        ] {
            let l = layout(size, align);
            let p = heap
                .allocate(l)
                .unwrap_or_else(|_| panic!("alloc {size}/{align}"));
            let addr = p.as_ptr() as *mut u8 as usize;
            assert_eq!(addr % align, 0, "size {size} align {align}: misaligned");
            assert!(p.len() >= size, "size {size}: short block");
            // Touch both ends — the mapping is real.
            unsafe {
                let ptr = p.as_ptr() as *mut u8;
                ptr.write(1);
                ptr.add(size - 1).write(2);
                heap.deallocate(NonNull::new_unchecked(ptr), l);
            }
        }
    }

    #[test]
    fn huge_path_allocates_and_recycles() {
        let heap = test_heap(EXTENT_ALIGN * 64);
        let l = layout(100_000, 64); // > 64 KiB → huge
        assert_eq!(heap.stats().extent_count, 0);
        let a = heap.allocate(l).expect("huge alloc");
        let a_ptr = a.as_ptr() as *mut u8;
        assert_eq!(a_ptr as usize % 64, 0);
        assert!(a.len() >= 20_000);
        unsafe { core::ptr::write_bytes(a_ptr, 0x5A, 20_000) };
        assert_eq!(heap.stats().extent_count, 1, "one huge extent mapped");
        // Free → recycle; the next same-size huge reuses the extent (no new map).
        unsafe { heap.deallocate(NonNull::new_unchecked(a_ptr), l) };
        let b = heap.allocate(l).expect("huge realloc");
        assert_eq!(b.as_ptr() as *mut u8, a_ptr, "huge extent recycled");
        assert_eq!(heap.stats().extent_count, 1, "no new extent — reuse");
        unsafe { heap.deallocate(b.cast::<u8>(), l) };
    }

    #[test]
    fn huge_recycle_search_looks_past_an_undersized_head() {
        let heap = test_heap(EXTENT_ALIGN * 1024);
        let large = layout(480_000, 64);
        let small = layout(120_000, 64);
        let large_block = heap.allocate(large).expect("large huge extent");
        let large_ptr = large_block.as_ptr() as *mut u8;
        let small_block = heap.allocate(small).expect("small huge extent");

        // Push large first, then small: the undersized small extent is the
        // Treiber head and the suitable large extent sits immediately behind.
        unsafe {
            heap.deallocate(large_block.cast(), large);
            heap.deallocate(small_block.cast(), small);
        }
        assert_eq!(heap.stats().extent_count, 2);

        let reused = heap.allocate(large).expect("fit behind small head");
        assert_eq!(
            reused.as_ptr() as *mut u8,
            large_ptr,
            "bounded fit search must recover the suitable recycled extent"
        );
        assert_eq!(
            heap.stats().extent_count,
            2,
            "head-of-line mismatch must not force another mapping"
        );
        unsafe { heap.deallocate(reused.cast(), large) };
    }

    #[test]
    fn huge_geometry_overflow_refuses_before_touching_the_source() {
        let heap = test_heap(EXTENT_ALIGN * 4);
        assert_eq!(heap.huge_alloc(usize::MAX, 4096), Err(AllocError));
        assert_eq!(heap.stats().extent_count, 0);
    }

    /// A source that admits a chosen number of maps and refuses the rest, so
    /// the heap's growth can be driven into a refusal at a chosen moment.
    struct GatedSource {
        inner: StaticMemSource,
        /// Maps still admitted; `usize::MAX` admits every one.
        allow: AtomicUsize,
    }

    impl GatedSource {
        fn admit(&self, maps: usize) {
            self.allow.store(maps, Ordering::Release);
        }
    }

    // SAFETY: `inner` is an atomic bump over one leaked exclusive arena; the
    // gate only refuses, never changing the geometry it returns.
    unsafe impl MemSource for GatedSource {
        fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
            self.allow
                .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    (n == usize::MAX).then_some(n).or_else(|| n.checked_sub(1))
                })
                .ok()?;
            self.inner.map_extent(min_bytes)
        }
    }

    fn gated_heap(arena_bytes: usize) -> Heap<GatedSource> {
        let buf: &'static mut [u8] = alloc::vec![0u8; arena_bytes].leak();
        // SAFETY: `buf` is leaked (process-lived) and owned solely by the source.
        let inner = unsafe { StaticMemSource::over(buf.as_mut_ptr(), arena_bytes) };
        Heap::new_in(GatedSource {
            inner,
            allow: AtomicUsize::new(usize::MAX),
        })
    }

    /// `kore-svc-mm`'s `MmSource` protocol: a miss records the size asked,
    /// the owner refills with exactly that, and the allocation retries
    /// once.  This source refuses every map but one it was told to admit,
    /// of exactly the size last refused.
    struct OneShotSource {
        inner: StaticMemSource,
        /// The size of the last refused map.
        asked: AtomicUsize,
        /// The one size admitted next; zero admits nothing.
        admit: AtomicUsize,
    }

    impl OneShotSource {
        fn asked(&self) -> usize {
            self.asked.load(Ordering::Acquire)
        }
        /// Admit one map of the size last refused.
        fn refill(&self) {
            self.admit.store(self.asked(), Ordering::Release);
        }
    }

    // SAFETY: `inner` is an atomic bump over one leaked exclusive arena; the
    // gate only refuses, never changing the geometry it returns.
    unsafe impl MemSource for OneShotSource {
        fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
            if self
                .admit
                .compare_exchange(min_bytes, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                self.asked.store(min_bytes, Ordering::Release);
                return None;
            }
            self.inner.map_extent(min_bytes)
        }
    }

    /// The arena starts on a leaf's span (`1 << level_shift(PAGE_MAP_LEVELS - 1)`,
    /// 2 MiB), so the extents this test sizes in nodes straddle no leaf
    /// boundary and the node counts it asserts do not depend on where the
    /// allocator put the arena.
    fn one_shot_heap(arena_bytes: usize) -> Heap<OneShotSource> {
        let span = 1usize << level_shift(PAGE_MAP_LEVELS - 1);
        let buf: &'static mut [u8] = alloc::vec![0u8; arena_bytes + span].leak();
        let raw = buf.as_mut_ptr() as usize;
        let base = align_up(raw, span);
        // SAFETY: `buf` is leaked (process-lived), owned solely by the source,
        // and `[base, base + arena_bytes)` lies inside it.
        let inner = unsafe { StaticMemSource::over(base as *mut u8, arena_bytes) };
        Heap::new_in(OneShotSource {
            inner,
            asked: AtomicUsize::new(0),
            admit: AtomicUsize::new(0),
        })
    }

    /// Every pool block routes through the page map to the pool extent
    /// holding it — one whose tail may have carried the map's own nodes.
    #[test]
    fn pool_blocks_route_through_the_page_map() {
        let heap = test_heap(EXTENT_ALIGN * 64);
        let l = layout(48, 8);
        let mut held = Vec::new();
        for _ in 0..2000 {
            let p = heap.allocate(l).expect("alloc");
            let addr = p.as_ptr() as *mut u8 as usize;
            let ext = heap.route(addr).expect("a live block routes");
            assert_eq!(ext.kind, ExtentKind::Pool);
            assert!(ext.contains(addr));
            assert_eq!(ext.mapped, POOL_EXTENT_BYTES);
            assert!(ext.len() <= POOL_EXTENT_BYTES && ext.len() >= POOL_EXTENT_BYTES / 2);
            held.push(p);
        }
        assert!(
            heap.stats().extent_count >= 2,
            "the working set spans extents"
        );
        for p in held {
            unsafe { heap.deallocate(p.cast::<u8>(), l) };
        }
    }

    /// Every huge payload routes through the page map to its own extent —
    /// from its first page and its last, however far the alignment pushed
    /// it — and back onto the recycle stack when freed.
    #[test]
    fn huge_payloads_route_through_the_page_map() {
        let heap = test_heap(EXTENT_ALIGN * 1024);
        for &(size, align) in &[(200_000usize, 16usize), (200_000, 4096), (90_000, 65_536)] {
            let l = layout(size, align);
            let p = heap.allocate(l).expect("huge alloc");
            let addr = p.as_ptr() as *mut u8 as usize;
            let ext = heap.route(addr).expect("a live payload routes");
            assert_eq!(ext.kind, ExtentKind::Huge, "size {size} align {align}");
            assert!(ext.contains(addr));
            let tail = heap.route(addr + size - 1).expect("its last page routes");
            assert!(core::ptr::eq(tail, ext), "one extent, first page to last");
            if align <= MIN_BLOCK {
                assert_eq!(addr, ext.base + extent_header_bytes(), "unpadded payload");
            }
            unsafe { heap.deallocate(p.cast::<u8>(), l) };
            // Routed to the recycle stack: the next fit comes back from it.
            let again = heap.allocate(l).expect("recycled");
            assert_eq!(again.as_ptr() as *mut u8 as usize, addr);
            unsafe { heap.deallocate(again.cast::<u8>(), l) };
        }
    }

    /// The map answers only for pages a live extent covers, and an extent
    /// past its reach refuses before the header write — a write there would
    /// fault, which is the test — leaving the directory untouched.
    #[test]
    fn the_page_map_ends_where_user_addresses_do() {
        let heap = test_heap(EXTENT_ALIGN * 32);
        let beyond = 1usize << (PAGE_BITS + PAGE_MAP_BITS);
        assert_eq!(beyond, 1 << 47);
        assert!(page_map_path(beyond).is_none());
        assert!(page_map_path(beyond - 1).is_some());
        assert!(
            heap.route(0).is_none(),
            "the null page is never an extent's"
        );
        assert!(heap.route(beyond).is_none());

        let mut meta = MetaBudget::new();
        assert!(
            heap.install_extent(
                beyond,
                EXTENT_ALIGN,
                EXTENT_ALIGN,
                Payload::Exact,
                ExtentKind::Pool,
                0,
                &mut meta
            )
            .is_err(),
            "an extent starting past the map refuses"
        );
        assert!(
            heap.install_extent(
                beyond - EXTENT_ALIGN,
                2 * EXTENT_ALIGN,
                EXTENT_ALIGN,
                Payload::Exact,
                ExtentKind::Pool,
                0,
                &mut meta
            )
            .is_err(),
            "an extent whose last page lies past the map refuses"
        );
        assert_eq!(heap.stats().extent_count, 0);
        let l = layout(48, 8);
        let p = heap.allocate(l).expect("alloc");
        let ext = heap.route(p.as_ptr() as *mut u8 as usize).expect("routes");
        assert_eq!(ext.dir_idx, 0, "no claim was spent on the refusals");
        unsafe { heap.deallocate(p.cast::<u8>(), l) };
    }

    /// A grow completes with the one mapping it asked for: a source that
    /// answers a miss with exactly that extent, once, serves a pool grow
    /// (its nodes carved from the extent's own tail) and a huge grow (over-
    /// asked by the metadata bound while the reserve is short, the unused
    /// part left to the reserve, so the next huge grow asks its payload
    /// alone).
    #[test]
    fn a_grow_completes_with_the_one_extent_it_asked_for() {
        let heap = one_shot_heap(EXTENT_ALIGN * 1024);
        let small = layout(48, 8);
        assert_eq!(heap.allocate(small), Err(AllocError), "the first miss");
        assert_eq!(heap.source().asked(), POOL_EXTENT_BYTES);
        heap.source().refill();
        let p = heap.allocate(small).expect("the refill serves the retry");
        let ext = heap.route(p.as_ptr() as *mut u8 as usize).expect("routes");
        assert_eq!(ext.mapped, POOL_EXTENT_BYTES);
        assert_eq!(
            ext.len(),
            POOL_EXTENT_BYTES - (PAGE_MAP_LEVELS - 1) * EXTENT_ALIGN,
            "the map's three nodes came off the pool extent's tail"
        );
        assert_eq!(heap.reserve.pages(), 0, "a pool grow leaves nothing over");
        unsafe { heap.deallocate(p.cast::<u8>(), small) };

        let big = layout(200_000, 64);
        let bytes = extent_header_bytes() + 64 + 200_000;
        assert_eq!(heap.allocate(big), Err(AllocError), "a huge miss");
        let asked = heap.source().asked();
        assert_eq!(
            asked,
            align_up(bytes, EXTENT_ALIGN) + meta_pages_bound(bytes) * EXTENT_ALIGN,
            "over-asked by the bound with the reserve empty"
        );
        heap.source().refill();
        let p = heap.allocate(big).expect("the refill serves the retry");
        let ext = heap.route(p.as_ptr() as *mut u8 as usize).expect("routes");
        assert_eq!(ext.kind, ExtentKind::Huge);
        assert_eq!(ext.mapped, asked);
        assert_eq!(ext.len(), align_up(bytes, EXTENT_ALIGN), "payload alone");
        let reserved = heap.reserve.pages();
        assert_eq!(
            reserved,
            meta_pages_bound(bytes),
            "the span shared the pool extent's nodes: the whole over-ask is reserved"
        );

        let more = layout(200_000, 64);
        assert_eq!(heap.allocate(more), Err(AllocError), "another huge miss");
        assert_eq!(
            heap.source().asked(),
            align_up(bytes, EXTENT_ALIGN),
            "the reserve covers the bound: the payload alone is asked"
        );
        heap.source().refill();
        let q = heap.allocate(more).expect("served");
        assert_eq!(
            heap.reserve.pages(),
            reserved,
            "nothing taken, nothing lost"
        );
        unsafe {
            heap.deallocate(p.cast::<u8>(), big);
            heap.deallocate(q.cast::<u8>(), more);
        }
    }

    /// The reserve hands runs to a budget and takes them back, page by
    /// page, merging what comes back adjacent.
    #[test]
    fn the_reserve_claims_and_offers_runs() {
        let reserve = MetaReserve::new();
        let base = 0x7000_0000usize;
        assert!(reserve.offer(MetaRun { base, pages: 3 }));
        assert!(reserve.offer(MetaRun {
            base: base + 16 * EXTENT_ALIGN,
            pages: 2
        }));
        assert_eq!(reserve.pages(), 5);

        let mut budget = MetaBudget::new();
        reserve.claim(4, &mut budget);
        assert_eq!(budget.pages(), 4);
        assert_eq!(reserve.pages(), 1, "the second run kept its last page");
        assert_eq!(
            budget.take(),
            Some(base + 16 * EXTENT_ALIGN),
            "last run first"
        );
        assert_eq!(
            budget.take(),
            Some(base),
            "then the first run, from its base"
        );
        assert_eq!(budget.take(), Some(base + EXTENT_ALIGN));
        budget.put(base + EXTENT_ALIGN);
        assert_eq!(budget.pages(), 2, "the page back merged with its run");
        assert_eq!(budget.len, 1);
        budget.release(&reserve);
        assert_eq!(reserve.pages(), 3);

        // The tail serves after the runs, down to its floor.
        let mut budget = MetaBudget::new();
        budget.with_tail(base + 2 * EXTENT_ALIGN, base + 4 * EXTENT_ALIGN);
        assert_eq!(budget.take(), Some(base + 3 * EXTENT_ALIGN));
        assert_eq!(budget.take(), Some(base + 2 * EXTENT_ALIGN));
        assert_eq!(budget.take(), None, "the floor holds");
        assert_eq!(budget.tail_end(), base + 2 * EXTENT_ALIGN);
    }

    /// Past the inline page, the directory installs further pages on
    /// metadata the grows carry: more live extents than one page holds,
    /// every one of them routable, and recycled rather than remapped once
    /// freed.
    #[test]
    fn the_directory_grows_past_its_inline_page() {
        // 320 huge extents of 17 pages (header + align + 65,537 bytes, page
        // rounded) plus the metadata bound a few of them over-ask.
        let count = PAGE_SLOTS + PAGE_SLOTS / 4;
        let heap = test_heap(count * 17 * EXTENT_ALIGN + 128 * EXTENT_ALIGN);
        let l = layout(pool::LARGE_THRESHOLD + 1, 16);
        let mut held: Vec<NonNull<[u8]>> = Vec::with_capacity(count);
        for i in 0..count {
            let p = heap
                .allocate(l)
                .unwrap_or_else(|_| panic!("extent {i} of {count}"));
            held.push(p);
        }
        let grown = heap.stats();
        assert_eq!(
            grown.extent_count, count,
            "every allocation is its own extent"
        );
        assert!(count > PAGE_SLOTS, "the working set needs a second page");
        assert!(
            heap.page(1).is_some() && heap.page(2).is_none(),
            "one mapped page"
        );
        // Each routes to its extent, the last with a directory id past page 0.
        let last = held[count - 1].as_ptr() as *mut u8 as usize;
        let ext = heap.route(last).expect("a live payload routes");
        assert!(
            ext.dir_idx as usize >= PAGE_SLOTS,
            "registered in a mapped page"
        );
        assert!(ext.contains(last));
        assert!(core::ptr::eq(
            heap.slot(ext.dir_idx).load(Ordering::Acquire),
            ext
        ));
        for p in held.drain(..) {
            unsafe { heap.deallocate(p.cast::<u8>(), l) };
        }
        // Freed extents recycle through the stack; nothing new is mapped.
        for _ in 0..count {
            held.push(heap.allocate(l).expect("recycled extent"));
        }
        assert_eq!(heap.stats(), grown, "all recycled");
        for p in held {
            unsafe { heap.deallocate(p.cast::<u8>(), l) };
        }
    }

    /// A large class (16, 32, 64 KiB) carves from extents of its own: a
    /// header page, then whole blocks back to back, page-aligned.  A
    /// shard's first holds one block and each later one twice the last's,
    /// up to eight, so twenty-three requests take extents of 1, 2, 4, 8
    /// and 8 blocks; the class recycles exactly — a free block serves the
    /// next request without a mapping — and a small class keeps its own
    /// lane.
    #[test]
    fn a_large_class_ramps_its_extents_from_one_block() {
        let heap = test_heap(16 << 20);
        let size = 64 * 1024;
        let l = layout(size, 16);
        assert!(!is_huge(size));
        let mut held: Vec<NonNull<[u8]>> = Vec::new();
        let mut bases: Vec<usize> = Vec::new();
        for _ in 0..23 {
            let block = heap.allocate(l).expect("a 64 KiB block");
            let addr = block.cast::<u8>().as_ptr() as usize;
            assert_eq!(addr % EXTENT_ALIGN, 0, "page-aligned");
            let ext = heap.route(addr).expect("routes");
            assert_eq!(ext.kind, ExtentKind::Pool);
            if bases.last() != Some(&ext.base) {
                assert_eq!(
                    addr,
                    ext.base + EXTENT_ALIGN,
                    "the first block follows the header page"
                );
                bases.push(ext.base);
            } else {
                let last = held.last().expect("an earlier block").cast::<u8>().as_ptr() as usize;
                assert_eq!(addr, last + size, "blocks lie back to back");
            }
            held.push(block);
        }
        let blocks: Vec<usize> = bases
            .iter()
            .map(|&base| {
                let len = heap.route(base).expect("routes").len();
                assert_eq!(
                    (len - EXTENT_ALIGN) % size,
                    0,
                    "whole blocks, nothing left over"
                );
                (len - EXTENT_ALIGN) / size
            })
            .collect();
        assert_eq!(blocks, [1, 2, 4, 8, 8]);
        let grown = heap.stats();
        // SAFETY: a live allocation of `l`.
        unsafe { heap.deallocate(held.pop().expect("held").cast(), l) };
        let again = heap.allocate(l).expect("a recycled 64 KiB block");
        assert_eq!(
            heap.stats().mapped_bytes,
            grown.mapped_bytes,
            "recycled, not mapped"
        );
        let small = heap.allocate(layout(32, 16)).expect("a small block");
        assert_eq!(
            heap.stats().extent_count,
            grown.extent_count + 1,
            "the shared lane grows"
        );
        // SAFETY: live allocations of their layouts.
        unsafe {
            heap.deallocate(small.cast(), layout(32, 16));
            heap.deallocate(again.cast(), l);
            for block in held {
                heap.deallocate(block.cast(), l);
            }
        }
    }

    /// A bounded arena pays about a block and a page for each large class
    /// it uses: one request in each of the 16, 32 and 64 KiB classes fits
    /// a 256 KiB arena, where extents of eight blocks would have asked for
    /// 896 KiB — what refused a megabyte arena's first few large requests.
    /// What the grows mapped is their blocks, header pages and the metadata
    /// they over-asked for, the unused part of which waits in the reserve.
    #[test]
    fn a_bounded_arena_pays_a_block_and_a_page_per_large_class() {
        let heap = test_heap(256 * 1024 + EXTENT_ALIGN);
        let sizes = [16 * 1024, 32 * 1024, 64 * 1024];
        let held: Vec<(NonNull<[u8]>, Layout)> = sizes
            .iter()
            .map(|&size| {
                let l = layout(size, 16);
                (
                    heap.allocate(l)
                        .unwrap_or_else(|_| panic!("a {size}-byte block")),
                    l,
                )
            })
            .collect();
        let stats = heap.stats();
        assert_eq!(stats.extent_count, sizes.len());
        let payload: usize = sizes.iter().map(|size| EXTENT_ALIGN + size).sum();
        let over_ask: usize = sizes
            .iter()
            .map(|size| meta_pages_bound(EXTENT_ALIGN + size) * EXTENT_ALIGN)
            .sum();
        assert!(
            stats.mapped_bytes >= payload && stats.mapped_bytes <= payload + over_ask,
            "a block and a page each, and their metadata: {stats:?}"
        );
        for (p, l) in held {
            // SAFETY: live allocations of their layouts.
            unsafe { heap.deallocate(p.cast(), l) };
        }
    }

    /// A large class's blocks are page-aligned, so a large request aligned
    /// past a page is served on the huge path, which aligns any payload;
    /// its free routes by the extent, back to the huge buckets.
    #[test]
    fn a_large_request_aligned_past_a_page_is_served_whole() {
        let heap = test_heap(4 << 20);
        let paged = layout(20_000, EXTENT_ALIGN);
        let p = heap.allocate(paged).expect("page-aligned");
        let ext = heap
            .route(p.cast::<u8>().as_ptr() as usize)
            .expect("routes");
        assert_eq!(
            ext.kind,
            ExtentKind::Pool,
            "a page-aligned large request is a class block"
        );
        let wide = layout(20_000, 4 * EXTENT_ALIGN);
        let q = heap.allocate(wide).expect("aligned past a page");
        let addr = q.cast::<u8>().as_ptr() as usize;
        assert_eq!(addr % (4 * EXTENT_ALIGN), 0, "aligned as asked");
        assert_eq!(heap.route(addr).expect("routes").kind, ExtentKind::Huge);
        let before = heap.stats();
        // SAFETY: a live allocation of `wide`.
        unsafe { heap.deallocate(q.cast(), wide) };
        let again = heap.allocate(wide).expect("recycled");
        assert_eq!(
            again.cast::<u8>(),
            q.cast::<u8>(),
            "the huge extent came back"
        );
        assert_eq!(
            heap.stats().mapped_bytes,
            before.mapped_bytes,
            "recycled, not mapped"
        );
        // SAFETY: live allocations of their layouts.
        unsafe {
            heap.deallocate(again.cast(), wide);
            heap.deallocate(p.cast(), paged);
        }
    }

    /// On `kore-svc-mm`'s protocol a large class's grow is one miss, one
    /// refill of exactly what was asked, one retry: its size is derived
    /// from the lane's own extent, which a refused grow leaves as it was,
    /// so the retry asks the same again — one block, then two.
    #[test]
    fn a_large_class_grow_is_served_by_one_refill() {
        let heap = one_shot_heap(EXTENT_ALIGN * 1024);
        let size = 64 * 1024;
        let l = layout(size, 16);
        let mut held = Vec::new();
        for blocks in [1, 2] {
            assert_eq!(heap.allocate(l), Err(AllocError), "a miss");
            heap.source().refill();
            let p = heap.allocate(l).expect("the refill serves the retry");
            let ext = heap
                .route(p.cast::<u8>().as_ptr() as usize)
                .expect("routes");
            assert_eq!(ext.len(), EXTENT_ALIGN + blocks * size);
            held.push(p);
        }
        for p in held {
            // SAFETY: live allocations of `l`.
            unsafe { heap.deallocate(p.cast(), l) };
        }
    }

    /// Free huge extents are found by size: a request takes the smallest
    /// free extent that fits, not the most recently freed one.
    #[test]
    fn a_huge_request_takes_the_smallest_free_extent_that_fits() {
        let heap = test_heap(64 << 20);
        let big = layout(1 << 20, 16);
        let fit = layout(128 << 10, 16);
        let p_fit = heap.allocate(fit).expect("128 KiB");
        let p_big = heap.allocate(big).expect("1 MiB");
        let mapped = heap.stats();
        // Free the fit first, then the big one: a size-blind stack would hand
        // the big one (most recent) to the next 128 KiB request.
        // SAFETY: live allocations of their layouts.
        unsafe {
            heap.deallocate(p_fit.cast(), fit);
            heap.deallocate(p_big.cast(), big);
        }
        let again = heap.allocate(fit).expect("128 KiB again");
        let after = heap.stats();
        assert_eq!(
            again.cast::<u8>(),
            p_fit.cast::<u8>(),
            "the 128 KiB extent, not the 1 MiB one"
        );
        assert_eq!(
            after.mapped_bytes, mapped.mapped_bytes,
            "recycled, not mapped"
        );
        assert_eq!(after.extent_count, mapped.extent_count, "and not split");
        // A request past every free extent maps fresh.
        let bigger = layout(2 << 20, 16);
        let fresh = heap.allocate(bigger).expect("2 MiB");
        assert!(heap.stats().mapped_bytes > mapped.mapped_bytes);
        // SAFETY: live allocations of their layouts.
        unsafe {
            heap.deallocate(again.cast(), fit);
            heap.deallocate(fresh.cast(), bigger);
        }
    }

    /// The bucket ladder: an extent's bucket is the power of two of its
    /// length above the alignment, and the last bucket holds the rest.
    #[test]
    fn huge_buckets_follow_the_length() {
        assert_eq!(huge_bucket(EXTENT_ALIGN), 0);
        assert_eq!(huge_bucket(2 * EXTENT_ALIGN - 1), 0);
        assert_eq!(huge_bucket(2 * EXTENT_ALIGN), 1);
        assert_eq!(huge_bucket(1 << 20), 20 - EXTENT_ALIGN_BITS);
        assert_eq!(huge_bucket(usize::MAX), HUGE_BUCKETS - 1);
    }

    /// With the inline page full and the source spent, the next grow
    /// refuses with no page published, and the first grow after the source
    /// returns brings the page with it and lands in its first slot.
    #[test]
    fn a_spent_source_refuses_a_directory_page() {
        // One 17-page huge extent each (header + align + 65,537 bytes, page
        // rounded), and the metadata bound a few of them over-ask.
        let heap = gated_heap(PAGE_SLOTS * 17 * EXTENT_ALIGN + 128 * EXTENT_ALIGN);
        let l = layout(pool::LARGE_THRESHOLD + 1, 16);
        let mut held: Vec<NonNull<[u8]>> = Vec::with_capacity(PAGE_SLOTS + 1);
        for i in 0..PAGE_SLOTS {
            held.push(
                heap.allocate(l)
                    .unwrap_or_else(|_| panic!("extent {i} fills the inline page")),
            );
        }
        let full = heap.stats();
        assert_eq!(full.extent_count, PAGE_SLOTS);
        assert!(heap.page(1).is_none());

        heap.source().admit(0);
        assert_eq!(heap.allocate(l), Err(AllocError), "no extent, no page");
        assert!(heap.page(1).is_none(), "nothing published");
        assert_eq!(heap.stats(), full);

        heap.source().admit(usize::MAX);
        let p = heap.allocate(l).expect("the source is back");
        let ext = heap.route(p.as_ptr() as *mut u8 as usize).expect("routes");
        assert_eq!(
            ext.dir_idx as usize, PAGE_SLOTS,
            "first slot of the mapped page"
        );
        assert!(heap.page(1).is_some());
        held.push(p);
        for p in held {
            unsafe { heap.deallocate(p.cast::<u8>(), l) };
        }
    }

    #[test]
    fn a_short_aligned_extent_refuses_before_the_header_write() {
        let heap = test_heap(EXTENT_ALIGN * 4);
        let buf: &'static mut [u8] = alloc::vec![0u8; EXTENT_ALIGN * 2].leak();
        let raw = buf.as_mut_ptr() as usize;
        let aligned = checked_align_up(raw, EXTENT_ALIGN).expect("host address aligns");
        let mut meta = MetaBudget::new();
        for (total, note) in [
            (extent_header_bytes(), "shorter than the request"),
            (EXTENT_ALIGN + 1, "not a whole number of pages"),
        ] {
            assert!(
                heap.install_extent(
                    aligned,
                    total,
                    EXTENT_ALIGN,
                    Payload::Exact,
                    ExtentKind::Pool,
                    0,
                    &mut meta,
                )
                .is_err(),
                "source geometry {note} must be rejected"
            );
        }
        assert!(
            buf.iter().all(|byte| *byte == 0),
            "validation must happen before writing self-hosted metadata"
        );
        assert_eq!(heap.stats().extent_count, 0);
    }

    #[test]
    fn grows_a_second_extent_on_miss() {
        // Arena for a few pool extents; allocate more than one extent holds.
        let heap = test_heap(EXTENT_ALIGN + 3 * POOL_EXTENT_BYTES);
        let l = layout(8192, 8); // one block per (8192) class
        let mut held: Vec<NonNull<[u8]>> = Vec::new();
        // ~7 blocks per 64 KiB extent, so 10 forces a grow to a 2nd extent.
        for _ in 0..10 {
            held.push(heap.allocate(l).expect("alloc across grow"));
        }
        assert!(
            heap.stats().extent_count >= 2,
            "the working set forced a grow: {:?}",
            heap.stats()
        );
        for b in held {
            unsafe { heap.deallocate(b.cast::<u8>(), l) };
        }
    }

    #[test]
    fn allocator_face_errs_when_source_exhausted() {
        // Room for one pool extent on its own boundary (whatever the arena's
        // front alignment costs) and no second; the second grow must fail
        // with AllocError, NOT abort (the Allocator face never aborts).
        let heap = test_heap(EXTENT_ALIGN + 2 * POOL_EXTENT_BYTES);
        let l = layout(8192, 8);
        let mut ok = 0;
        let err = loop {
            match heap.allocate(l) {
                Ok(_) => {
                    ok += 1;
                    assert!(ok < 10_000, "a bounded arena must exhaust");
                }
                Err(e) => break e,
            }
        };
        assert!(ok > 0, "some allocations succeeded before exhaustion");
        assert_eq!(err, AllocError, "exhaustion surfaces AllocError, no abort");
    }

    #[test]
    fn box_and_vec_new_in_ride_the_allocator_face() {
        let heap = test_heap(EXTENT_ALIGN * 40);
        // Box::new_in / Vec::new_in go through `&Heap: Allocator`.
        let boxed = Box::new_in(0xC0FFEEu64, &heap);
        assert_eq!(*boxed, 0xC0FFEE);
        let mut v: Vec<u32, &Heap<StaticMemSource>> = Vec::new_in(&heap);
        for i in 0..500 {
            v.push(i);
        }
        assert_eq!(v.iter().copied().sum::<u32>(), (0..500).sum());
        drop(boxed);
        drop(v);
    }

    #[test]
    fn global_alloc_face_direct_and_zeroed() {
        // Exercise the GlobalAlloc methods directly — NOT installed as the real
        // #[global_allocator] (that would hijack the test harness).
        let heap = test_heap(EXTENT_ALIGN * 40);
        let l = layout(128, 16);
        unsafe {
            let p = GlobalAlloc::alloc(&heap, l);
            assert!(!p.is_null());
            assert_eq!(p as usize % 16, 0);
            p.write_bytes(0xEE, 128);
            GlobalAlloc::dealloc(&heap, p, l);

            // alloc_zeroed hands back zeroed memory even though pool blocks are
            // recycled dirty (this reuses the block just freed).
            let z = GlobalAlloc::alloc_zeroed(&heap, l);
            assert!(!z.is_null());
            for i in 0..128 {
                assert_eq!(z.add(i).read(), 0, "alloc_zeroed must zero");
            }
            GlobalAlloc::dealloc(&heap, z, l);
        }
    }

    #[test]
    fn cross_thread_free_drains_into_the_owner() {
        // A block allocated on one thread and freed on another must route to the
        // owner's shard (its MPSC remote list in the common case where the two
        // threads hold different shards), and the owner reclaims it on its next
        // allocation.  Sequenced so no two threads touch one shard's local list
        // concurrently, so it stays correct even if the round-robin collocates.
        use std::sync::Barrier;

        let heap = test_heap(EXTENT_ALIGN * 40);
        let l = layout(64, 8);
        let shared: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
        let published = Barrier::new(2);
        let freed = Barrier::new(2);

        let reused = std::thread::scope(|s| {
            // Owner: allocate, publish, wait for the remote free, then re-allocate.
            let owner = s.spawn(|| {
                let b1 = heap.allocate(l).expect("owner alloc").as_ptr() as *mut u8;
                shared.store(b1, Ordering::SeqCst);
                published.wait();
                freed.wait();
                let b2 = heap.allocate(l).expect("owner realloc").as_ptr() as *mut u8;
                b1 == b2
            });
            // Freer: a different thread (different shard) frees the owner's block.
            s.spawn(|| {
                published.wait();
                let p = shared.load(Ordering::SeqCst);
                // SAFETY: `p` is the owner's live block of `l`.
                unsafe { heap.deallocate(NonNull::new_unchecked(p), l) };
                freed.wait();
            });
            owner.join().unwrap()
        });

        assert!(reused, "the owner reclaimed the cross-thread-freed block");
    }

    #[test]
    fn independent_shard_misses_enter_the_source_concurrently() {
        use std::sync::{Condvar, Mutex};
        use std::time::Duration;

        struct ConcurrentSource {
            inner: StaticMemSource,
            entered: Mutex<usize>,
            wake: Condvar,
        }

        // SAFETY: `inner` is an atomic bump over one leaked exclusive arena.
        // The host-only condition variable observes concurrency but never
        // changes the disjoint/aligned extent geometry it returns.
        unsafe impl MemSource for ConcurrentSource {
            fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
                let mut entered = self.entered.lock().unwrap();
                *entered += 1;
                self.wake.notify_all();
                if *entered < 2 {
                    let (observed, timed) = self
                        .wake
                        .wait_timeout_while(entered, Duration::from_secs(1), |n| *n < 2)
                        .unwrap();
                    entered = observed;
                    if timed.timed_out() && *entered < 2 {
                        return None;
                    }
                }
                drop(entered);
                self.inner.map_extent(min_bytes)
            }
        }

        // Room for two pool extents after the arena's front alignment; the
        // map's nodes come off the extents themselves.
        let buf: &'static mut [u8] = alloc::vec![0u8; 3 * POOL_EXTENT_BYTES].leak();
        let source = ConcurrentSource {
            // SAFETY: leaked and handed exclusively to this source.
            inner: unsafe { StaticMemSource::over(buf.as_mut_ptr(), buf.len()) },
            entered: Mutex::new(0),
            wake: Condvar::new(),
        };
        let heap = Heap::new_in(source);
        let l = layout(64, 16);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| heap.allocate(l).is_ok());
            let b = scope.spawn(|| heap.allocate(l).is_ok());
            assert!(a.join().unwrap() && b.join().unwrap());
        });
        // One map per miss: each pool extent carries the metadata its span
        // needs, so the source saw exactly the two.
        assert_eq!(*heap.source.entered.lock().unwrap(), 2);
        assert_eq!(heap.stats().extent_count, 2);
    }
}
