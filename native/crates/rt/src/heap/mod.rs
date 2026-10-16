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
//! - **Small/medium** (`<= 8 KiB` after folding alignment): per-shard, per-size
//!   free-lists.  A task pops from *its own* shard — zero atomics on the hot
//!   path.  A free block holds its successor pointer intrusively, so the
//!   allocator never allocates to allocate.
//! - **Cross-thread free**: a block owned by another shard is pushed onto that
//!   shard's lock-free MPSC remote list (the owner drains it on its own miss).
//! - **Huge** (`> 8 KiB`): a dedicated whole-extent allocation, aligned to the
//!   request, recycled through a generation-tagged free stack.
//! - **Grow on miss**: map a fresh extent through the [`MemSource`] seam and
//!   carve it — synchronously, since userland has no ISR/FP/lock-held hazard.
//!
//! See [`source`] for the [`MemSource`] seam — `kore-svc-mm`'s default
//! backing routes growth through the memory-manager service, so a heap
//! extent is a lessor *grant*, never a self-derive (what leaves task heaps
//! swappable, the lessor inversion).

mod pool;
mod shard;
mod source;

pub use crate::mem_source::MemSource;
#[cfg(target_os = "none")]
pub use source::carve_and_map;
pub use source::{HeapSupplySource, StaticMemSource};
pub use source::{adopt_supply, reserve_from_top};

use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use kore_collections::AtomicBitSet;

use pool::{MIN_BLOCK, align_up, checked_align_up, class_index, class_size, is_huge};
use shard::{NUM_SHARDS, Shard, current_shard};

/// Alignment of every mapped extent (and its self-hosted header) — a page.  The
/// [`MemSource`] returns extents aligned to this.  Public because it is the
/// seam's granularity contract: an out-of-crate source (and `kore-svc-mm`'s
/// request artifact, `client::alloc_props`) sizes and aligns to it.
pub const EXTENT_ALIGN: usize = crate::mem_source::ALIGN;

/// Bytes mapped per pool-extent grow.  A dial: large enough to amortize the map
/// over many small allocations, small enough that growth tracks live demand.
pub const POOL_EXTENT_BYTES: usize = 64 * 1024;

/// Chunk-directory width — the structural ceiling on live extents.  The
/// directory is `DIR_N` pointers in the `Heap`; the extents it points at are
/// mapped on demand, so memory tracks demand, not this ceiling (the #205 shape).
const DIR_N: usize = 1024;
const DIR_WORDS: usize = DIR_N.div_ceil(u64::BITS as usize);

/// How many free huge extents one cold allocation examines before mapping.
/// A bounded scan defeats head-of-line blocking (one undersized head hiding a
/// suitable extent) without turning a huge allocation into an unbounded walk.
const HUGE_RECYCLE_SCAN: usize = 16;

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
    /// Total mapped bytes.
    len: usize,
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
    fn end(&self) -> usize {
        self.base + self.len
    }
    #[inline]
    fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr - self.base < self.len
    }
}

/// Bytes reserved at an extent's head for its self-hosted [`Extent`], rounded so
/// the carve area that follows keeps [`MIN_BLOCK`] alignment.
#[inline]
const fn extent_header_bytes() -> usize {
    align_up(core::mem::size_of::<Extent>(), MIN_BLOCK)
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

/// Affine reservation of one extent-directory slot.
///
/// A miss claims before it calls the source, so directory exhaustion cannot
/// strand a freshly mapped extent.  Refusal or malformed source geometry drops
/// the claim and restores the bit; publication consumes it permanently because
/// live extents remain mapped for the process lifetime.
struct DirectoryClaim<'a> {
    free: &'a AtomicBitSet<DIR_WORDS>,
    index: usize,
    published: bool,
}

impl DirectoryClaim<'_> {
    #[inline]
    const fn index(&self) -> usize {
        self.index
    }

    #[inline]
    fn publish(mut self) {
        self.published = true;
    }
}

impl Drop for DirectoryClaim<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.free.release_run(self.index, 1);
        }
    }
}

/// The sharded, lock-free global allocator.  Generic over its [`MemSource`]
/// backing — the engine names no concrete source; founded binaries link
/// [`Heap<HeapSupplySource>`](HeapSupplySource).
pub struct Heap<S: MemSource> {
    /// Per-task allocator lanes.
    shards: [Shard; NUM_SHARDS],
    /// Lock-free extent directory: slots published `Release` by
    /// [`install_extent`](Self::install_extent), read `Acquire`, `null` until
    /// published (#205).
    directory: [AtomicPtr<Extent>; DIR_N],
    /// Free directory slots.  Claims happen before mapping; rollback on source
    /// refusal makes the capacity failure-atomic.
    directory_free: AtomicBitSet<DIR_WORDS>,
    /// One past the highest slot ever published — the cold scan bound.
    high_water: AtomicUsize,
    /// Generation-tagged Treiber stack of free huge extents (by directory id),
    /// for recycling (the board.rs scheme — the generation defeats ABA on the
    /// multi-consumer pop).
    huge_free: AtomicU64,
    /// Where fresh extents come from.
    source: S,
}

// SAFETY: cross-thread state is all atomic — the directory and slot ledger,
// `high_water`, `huge_free`, each shard's remote free-lists, and the pool carve
// cursors.  The only non-atomic state — each shard's local free-lists and its
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
            directory: [const { AtomicPtr::new(core::ptr::null_mut()) }; DIR_N],
            directory_free: AtomicBitSet::with_first_set(DIR_N),
            high_water: AtomicUsize::new(0),
            huge_free: AtomicU64::new(free_pack(0, FREE_EMPTY)),
            source,
        }
    }

    /// The backing source, for sources provisioned outside synchronous
    /// allocation (for example an async memory-service refill after a miss).
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Occupancy snapshot (cold directory scan).
    pub fn stats(&self) -> HeapStats {
        let n = self.high_water.load(Ordering::Acquire).min(DIR_N);
        let mut mapped_bytes = 0;
        let mut extent_count = 0;
        for slot in &self.directory[..n] {
            let p = slot.load(Ordering::Acquire);
            if p.is_null() {
                continue;
            }
            // SAFETY: a published pointer names a live `Extent` held for the
            // heap's life.
            let ext = unsafe { &*p };
            mapped_bytes += ext.len;
            extent_count += 1;
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
            match self.grow_pool(shard) {
                GrowOutcome::Grew => continue,
                GrowOutcome::Exhausted => return Err(AllocError),
            }
        }
    }

    /// Bump one `class`-sized block off the shard's current pool extent, aligning
    /// the block's **absolute address** to the class size (a power of two), which
    /// is `>= align` — so the returned pointer honors the request's alignment.
    fn carve(&self, shard: &Shard, class: usize) -> Option<*mut u8> {
        // SAFETY: owner reads its own current-extent pointer.
        let ext_ptr = unsafe { shard.current() };
        if ext_ptr.is_null() {
            return None;
        }
        // SAFETY: a published extent pointer names a live `Extent`.
        let ext = unsafe { &*ext_ptr };
        let sz = class_size(class);
        // Owner-only cursor: a plain load/store pair, no CAS (no rival writer).
        let cursor = ext.cursor.load(Ordering::Relaxed);
        let block = checked_align_up(cursor, sz)?;
        let end = block.checked_add(sz)?;
        if end > ext.end() {
            return None; // this extent is spent for `class`
        }
        ext.cursor.store(end, Ordering::Relaxed);
        Some(block as *mut u8)
    }

    /// Map and install one fresh pool extent for `shard`.
    ///
    /// Every live thread owns a distinct shard and distinct session scratch, so
    /// concurrent misses may map concurrently.  Directory/source reservation is
    /// already atomic; a process-wide grow flag would only create preemption
    /// inversion around a synchronous kernel session.
    #[cold]
    fn grow_pool(&self, shard: &Shard) -> GrowOutcome {
        let Some(claim) = self.claim_directory() else {
            return GrowOutcome::Exhausted;
        };
        match self.source.map_extent(POOL_EXTENT_BYTES) {
            None => GrowOutcome::Exhausted,
            Some(mem) => {
                let base = mem.cast::<u8>().as_ptr() as usize;
                let len = mem.len();
                match self.install_extent(
                    claim,
                    base,
                    len,
                    POOL_EXTENT_BYTES,
                    ExtentKind::Pool,
                    current_shard(),
                ) {
                    Ok(ext_ptr) => {
                        // SAFETY: owner sets its own current-extent pointer.
                        unsafe { shard.set_current(ext_ptr) };
                        GrowOutcome::Grew
                    }
                    // An unsafe source violated its geometry contract.  Refuse
                    // before writing the self-hosted header; the mapping cannot
                    // be returned through this one-way source seam, but no
                    // allocator state or directory capacity is published.
                    Err(()) => GrowOutcome::Exhausted,
                }
            }
        }
    }

    /// Claim one directory slot without mapping.  The bitset's failed-claim
    /// rollback makes concurrent scans lock-free; a bounded retry gives a later
    /// allocation another chance if this snapshot lost every observed bit.
    fn claim_directory(&self) -> Option<DirectoryClaim<'_>> {
        for _ in 0..DIR_N {
            let index = self.directory_free.find_first_set_from(0)?;
            if self.directory_free.try_claim_run(index, 1).is_ok() {
                return Some(DirectoryClaim {
                    free: &self.directory_free,
                    index,
                    published: false,
                });
            }
        }
        None
    }

    /// Consume `[base, base+len)` and its affine slot claim into a fresh
    /// [`Extent`]: validate geometry, self-host the header, then
    /// `Release`-publish the pointer.  A concurrent reader sees the slot as
    /// `null` until the complete header is visible.
    fn install_extent(
        &self,
        claim: DirectoryClaim<'_>,
        base: usize,
        len: usize,
        min_len: usize,
        kind: ExtentKind,
        owner: usize,
    ) -> Result<*mut Extent, ()> {
        if base % EXTENT_ALIGN != 0
            || len < min_len.max(extent_header_bytes())
            || len > isize::MAX as usize
            || base.checked_add(len).is_none()
        {
            return Err(());
        }
        let idx = claim.index();
        let ext_ptr = base as *mut Extent;
        // SAFETY: `base` is a fresh page-aligned mapping of `len >= header` bytes,
        // owned exclusively here (not yet published), so the header write is sound
        // and lands before the `Release` store below publishes the pointer.
        unsafe {
            ext_ptr.write(Extent {
                base,
                len,
                owner,
                kind,
                dir_idx: idx as u32,
                cursor: AtomicUsize::new(base + extent_header_bytes()),
                next_free: AtomicU32::new(0),
            });
        }
        self.directory[idx].store(ext_ptr, Ordering::Release);
        self.high_water.fetch_max(idx + 1, Ordering::Release);
        claim.publish();
        Ok(ext_ptr)
    }

    /// The extent whose `[base, base+len)` contains `addr` (the deallocate route),
    /// or `None` for a pointer this heap never handed out.
    fn route(&self, addr: usize) -> Option<&Extent> {
        let n = self.high_water.load(Ordering::Acquire).min(DIR_N);
        for slot in &self.directory[..n] {
            let p = slot.load(Ordering::Acquire);
            if p.is_null() {
                continue;
            }
            // SAFETY: published pointer → live `Extent`.
            let ext = unsafe { &*p };
            if ext.contains(addr) {
                return Some(ext);
            }
        }
        None
    }

    /// Return `layout`'s block to its owner — **never leaks**.  Routes the
    /// pointer to its extent, then: a huge extent goes onto the recycle stack; a
    /// pool block goes to the owning shard's local list (if we are the owner) or
    /// its MPSC remote list (if not).
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
        match ext.kind {
            ExtentKind::Huge => self.huge_push(ext.dir_idx),
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

    /// Serve a `> LARGE_THRESHOLD` request: search a bounded prefix of recycled
    /// extents for a fit, else map fresh `header + align + size` backing.
    #[cold]
    fn huge_alloc(&self, size: usize, align: usize) -> Result<NonNull<[u8]>, AllocError> {
        let mut deferred = [FREE_EMPTY; HUGE_RECYCLE_SCAN];
        let mut deferred_len = 0;
        let mut recycled = None;
        while deferred_len < HUGE_RECYCLE_SCAN {
            let Some(id) = self.huge_pop() else {
                break;
            };
            // SAFETY: an id off the huge-free stack names a published Huge extent.
            let ext = unsafe { &*self.directory[id as usize].load(Ordering::Acquire) };
            if let Some(slice) = huge_slice(ext, size, align) {
                recycled = Some(slice);
                break;
            }
            deferred[deferred_len] = id;
            deferred_len += 1;
        }
        // Restore the skipped nodes in reverse pop order, preserving the stack's
        // relative order and keeping each off-stack extent exclusively held.
        for &id in deferred[..deferred_len].iter().rev() {
            self.huge_push(id);
        }
        if let Some(slice) = recycled {
            return Ok(slice);
        }

        let bytes = extent_header_bytes()
            .checked_add(align)
            .and_then(|head| head.checked_add(size))
            .ok_or(AllocError)?;
        let claim = self.claim_directory().ok_or(AllocError)?;
        let mem = self.source.map_extent(bytes).ok_or(AllocError)?;
        let base = mem.cast::<u8>().as_ptr() as usize;
        let len = mem.len();
        let ext_ptr = self
            .install_extent(claim, base, len, bytes, ExtentKind::Huge, usize::MAX)
            .map_err(|()| AllocError)?;
        // SAFETY: just published.
        let ext = unsafe { &*ext_ptr };
        huge_slice(ext, size, align).ok_or(AllocError)
    }

    /// Push a huge extent (by directory id) onto the recycle stack.  The
    /// generation bump defeats the ABA a racing [`huge_pop`](Self::huge_pop)
    /// would otherwise admit.
    fn huge_push(&self, id: u32) {
        // SAFETY: `id` is a published extent's directory slot.
        let ext = unsafe { &*self.directory[id as usize].load(Ordering::Acquire) };
        let mut head = self.huge_free.load(Ordering::Acquire);
        loop {
            ext.next_free.store(free_id(head), Ordering::Relaxed);
            let new = free_pack(free_gen(head).wrapping_add(1), id);
            match self.huge_free.compare_exchange_weak(
                head,
                new,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
        }
    }

    /// Pop a recycled huge extent (by directory id), or `None` when the stack is
    /// empty.  Multi-consumer safe: the generation in the CAS'd word rejects a
    /// stale head.
    fn huge_pop(&self) -> Option<u32> {
        let mut head = self.huge_free.load(Ordering::Acquire);
        loop {
            let id = free_id(head);
            if id == FREE_EMPTY {
                return None;
            }
            // SAFETY: an id on the stack was pushed for a published extent.
            let ext = unsafe { &*self.directory[id as usize].load(Ordering::Acquire) };
            let next = ext.next_free.load(Ordering::Relaxed);
            let new = free_pack(free_gen(head).wrapping_add(1), next);
            match self.huge_free.compare_exchange_weak(
                head,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
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
// `layout.align()` (the class-address alignment / huge-payload alignment), and
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
        // Spans small classes, a large-but-pool alignment (4096 <= threshold),
        // and a huge alignment (16384 > threshold → huge path aligns the extent).
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
        let l = layout(20_000, 64); // > 8 KiB → huge
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
        let heap = test_heap(EXTENT_ALIGN * 128);
        let large = layout(48_000, 64);
        let small = layout(12_000, 64);
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

    #[test]
    fn a_rejected_extent_restores_its_directory_claim() {
        let heap = test_heap(EXTENT_ALIGN * 4);
        let first = heap.claim_directory().expect("slot zero");
        assert_eq!(first.index(), 0);
        assert!(
            heap.install_extent(first, 1, EXTENT_ALIGN, EXTENT_ALIGN, ExtentKind::Pool, 0,)
                .is_err(),
            "misaligned geometry refuses before the header write"
        );
        let again = heap.claim_directory().expect("rolled-back slot zero");
        assert_eq!(again.index(), 0, "affine claim rolled back on refusal");
    }

    #[test]
    fn a_short_aligned_extent_refuses_before_the_header_write() {
        let heap = test_heap(EXTENT_ALIGN * 4);
        let buf: &'static mut [u8] = alloc::vec![0u8; EXTENT_ALIGN * 2].leak();
        let raw = buf.as_mut_ptr() as usize;
        let aligned = checked_align_up(raw, EXTENT_ALIGN).expect("host address aligns");
        let first = heap.claim_directory().expect("slot zero");

        assert!(
            heap.install_extent(
                first,
                aligned,
                extent_header_bytes(),
                POOL_EXTENT_BYTES,
                ExtentKind::Pool,
                0,
            )
            .is_err(),
            "source geometry shorter than the request must be rejected"
        );
        assert!(
            buf.iter().all(|byte| *byte == 0),
            "validation must happen before writing self-hosted metadata"
        );
        let again = heap.claim_directory().expect("rolled-back slot zero");
        assert_eq!(again.index(), 0, "short geometry restores its claim");
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
        // Room for ~one pool extent; the second grow must fail with AllocError,
        // NOT abort (the Allocator face never aborts).
        let heap = test_heap(EXTENT_ALIGN * 2 + POOL_EXTENT_BYTES);
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

        let buf: &'static mut [u8] = alloc::vec![0u8; 4 * POOL_EXTENT_BYTES].leak();
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
        assert_eq!(*heap.source.entered.lock().unwrap(), 2);
        assert_eq!(heap.stats().extent_count, 2);
    }
}
