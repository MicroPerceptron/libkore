//! # `service` — the swap-service protocol + the default in-RAM (zram) backend
//!
//! **The `SwapBackend` seam, rewritten as a service**: [`SwapRequest`] is
//! the vocabulary, and *implementations are services* — [`ZramSwap`] (this
//! module: compress into RAM) now, a disk driver implementing the same
//! `SystemService<Request = SwapRequest>` later.  The refactor is what
//! makes the disk case natural: `SWAP_OUT`/`SWAP_IN` are already kore-band
//! uring opcodes ([`kore_abi::uring`]), the ring's CQ completion is the await
//! point, and the #206 tower veneer wraps the service for free.  kore-ring's
//! `SwapBackend` trait retires when the ring adapter lands.
//!
//! **How page bytes move** (#148: no capability rides request data): the
//! client attaches a **staging window** once ([`SwapRequest::Attach`], a
//! capability transferred over the return channel and mapped for the
//! service's life), and every later request names a page *index* into it.
//! Page-out reads that page; page-in decompresses into it.  The reply's
//! token is **data** — the server's private map key (the SwapBackend
//! contract: the kernel never interprets it) — never a capability.
//!
//! The shape it replaces delivered a fresh view per operation, which made
//! the backend map a view per page — and a receiver's map is one-way (no
//! unmap verb exists), so at 4 KiB against a 16 MiB per-thread stride the
//! store could serve ~4096 pages and then never map again.  One window,
//! attached once, also makes the whole datapath ordinary memory: the
//! round-trip tests below could not exist under the old shape, because a
//! host build can possess no delivered view at all.
//!
//! **The store budget is possession-shaped in the host model**: the
//! endowment is a [`SupplyToken`] whose [`len`](SupplyToken::len) is the
//! ceiling — no config strings. A page-out that would exceed it refuses
//! [`StoreFull`](SwapRefusal::StoreFull); an incompressible page stores
//! **raw** (the zram huge-page class), so one stored page costs at most
//! [`PAGE_BYTES`] and the books never lie about the win
//! ([`ZramStats::saved_bytes`]).
//!
//! **Every capacity is relative to that endowment** — the second ceiling,
//! the page-count table (Linux zram's *disksize*), is sized at bind as
//! [`DISK_FACTOR`]` × budget` worth of pages: endow a bigger store, get a
//! proportionally bigger swap, no absolute anywhere.
//!
//! **Zero grants, ever**: a swap service consumes delivered views and
//! returns data — it can mint no authority (contrast mm: the lessor
//! grants; the swapper only holds).  Asserted by every test below.
//!
//! The codec is [`lz4_flex`] block mode — pure Rust, `no_std`, the
//! wide-word fast paths.  A compressing swapper is also the first real
//! userland customer for the FPU capability: vector-heavy
//! by nature, it runs under an `FpuTicket` when the ring-3 tier lands.
//!
//! **Metal posture (the 2026-07-27 completion):** compressed bytes live
//! in the **endowed store itself** — `bind` carves `[0, budget)` off the
//! possessed supply and maps it (one ordinary session, no heap), and the
//! packed arena is that mapped span.  No compressed byte ever touches an
//! unrelated heap; the token table and scratch are ordinary service
//! working memory (every service heap-grows through its `role::HEAP`).

use alloc::vec::Vec;

use kore_abi::session::SlotRef;
use kore_abi::task::role;

use kore_rt::cap::{SupplyToken, verbs};
use kore_rt::service::{Serve, SystemService};

use kore_svc_mm::PAGE_BYTES;

/// Advertised-capacity factor: a bound service's page-count ceiling
/// ([`ZramStats::disk_pages`]) is `DISK_FACTOR × budget` worth of pages —
/// the Linux zram shape, where *disksize* (the uncompressed ceiling) is
/// conventionally ~2× the RAM the store may consume.  Filling it needs
/// average `DISK_FACTOR×` compression — the LZ4 norm on swappable pages,
/// and same-fill pages cost nothing at all.
pub const DISK_FACTOR: u64 = 2;

/// The swap service's role bundle: its compressed-store **budget** — a
/// possessed supply whose length is the ceiling.
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "zram")]
pub struct ZramEndowment {
    /// The claimed store-budget token.
    #[bundle(role = role::MEM_SUPPLY)]
    pub store: SupplyToken,
}

/// The swap vocabulary — what *any* swap backend serves (zram here, a
/// disk driver later).  One enum, one intake; the ring muxes.
#[derive(Clone, Copy, Debug)]
pub enum SwapRequest {
    /// **Adopt the staging window** transferred over the return channel —
    /// one shared mapping, attached once, that every later `Out`/`In`
    /// addresses by page index.
    ///
    /// `announced` is the founder-published return-landing correlation
    /// value (the [`Serve::returned_input`](kore_rt::service::Serve::returned_input)
    /// idiom the image service's `Register` uses): it names *which*
    /// transfer this request adopts, and carries no authority itself.
    ///
    /// Re-attaching replaces the window — the previous mapping stays
    /// installed (a receiver cannot unmap), so a client that re-attaches
    /// repeatedly is spending the backend's map cursor.  Attach once.
    Attach {
        /// The return-landing correlation value naming the transfer.
        announced: SlotRef,
    },
    /// Persist the staging window's page `page`; reply with the restore
    /// token.  The client wrote the bytes there before sending.
    Out {
        /// Staging page index.
        page: u32,
    },
    /// Restore `token`'s bytes into the staging window's page `page`,
    /// which the client then hands to the faulting tenant.  Restoring
    /// consumes the stored copy.
    In {
        /// The token [`SwapResponse::Stored`] returned.
        token: u32,
        /// Staging page index to restore into.
        page: u32,
    },
    /// Drop a stored page without restoring it — the tenant died while
    /// swapped out; its lessor releases the store.
    Discard {
        /// The token to retire.
        token: u32,
    },
    /// Read the tier's books ([`SwapStatus`]) — the control-plane verb a
    /// status client polls.  Carries nothing, moves nothing, and is
    /// answerable in every posture (a disk row with no controller
    /// answers with its `device` bit clear rather than refusing).
    Status,
}

/// The swap replies — data only (the token is the server's map key,
/// never a capability).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapResponse {
    /// Persisted; restore with this token.
    Stored {
        /// The server's private key for this page.
        token: u32,
    },
    /// The staging window is live; `pages` is how many pages it holds
    /// (the client's own bound, echoed back kernel-vouched).
    Attached {
        /// Staging capacity, in pages.
        pages: u32,
    },
    /// The staging page now holds the restored bytes; the stored copy is
    /// gone.
    Restored,
    /// The stored copy is dropped and its budget freed.
    Discarded,
    /// The tier's books — [`SwapRequest::Status`]'s answer.
    Status(SwapStatus),
    /// Refused — nothing stored, nothing lost, no authority moved.
    Refused(SwapRefusal),
}

/// Why the swapper refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapRefusal {
    /// The staging source page could not be read (a malformed window).
    BadPage,
    /// The staging destination page could not be written.
    BadDst,
    /// No page stored under that token.
    BadToken,
    /// Budget or table exhausted — evict elsewhere or grow the endowment.
    StoreFull,
    /// The backing store failed the transfer (a disk backend's I/O error
    /// — the page is not stored / not restored; retry or evict elsewhere).
    Backend,
    /// No staging window is attached, or the page index lies outside the
    /// attached one — the client must `Attach` before moving bytes, and
    /// must address only pages the window holds.
    NoStaging,
}

/// **One swap tier's books** — [`SwapRequest::Status`]'s typed answer,
/// shared by every backend the way the request vocabulary is (one wire
/// form, N services).  The union of both tiers' summaries: the zram tier
/// reports no queue and no route, so its io/msix words stay zero with
/// [`device`](Self::device)/[`route`](Self::route) saying so; a disk row
/// with no controller answers all-zero with `device` clear — the honest
/// posture, never a refusal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwapStatus {
    /// A backing device is bound (the disk tier's controller; always
    /// `true` for the zram tier, whose store is RAM).
    pub device: bool,
    /// The completion path holds its delivered interrupt route —
    /// [`fires`](Self::fires)/[`parks`](Self::parks) are live.  `false`
    /// = the polled fallback (or a route-less tier).
    pub route: bool,
    /// Store capacity, in pages (zram: its `disk_pages` budget).
    pub capacity_pages: u32,
    /// Pages currently stored.
    pub stored_pages: u32,
    /// Attached staging-window pages (`0` = unattached).
    pub staging_pages: u32,
    /// `Out` ops answered `Stored`.
    pub outs: u64,
    /// `In` ops answered `Restored`.
    pub ins: u64,
    /// `Discard` ops answered `Discarded`.
    pub discards: u64,
    /// Device commands submitted (disk tier; zram: zero).
    pub io_submitted: u64,
    /// Device commands completed.
    pub io_completed: u64,
    /// Device commands that completed in error.
    pub io_errors: u64,
    /// Completion parks resolved as delivered route fires — the MSI-X
    /// delivery witness (`0` beside growing `parks` = delivery-dead).
    pub fires: u64,
    /// Completion parks issued on the held route.
    pub parks: u64,
    /// Route-LESS operations paced by a clock deadline — the posture on a
    /// machine with no delivered vector but a readable clock. Bounded by
    /// elapsed time, so an expiry is real evidence about the device.
    pub deadline_waits: u64,
    /// Route-less operations paced only by a count of spin hints, because no
    /// clock could supply a deadline at all.
    ///
    /// Reported apart from [`deadline_waits`](Self::deadline_waits) because a
    /// spin count bounds retired instructions rather than time: an expiry
    /// there says the device was slower than N hints on this core under this
    /// contention, which is not a statement about the device. Non-zero means
    /// disk waits are being judged by something that is not a duration —
    /// which on a serial-less box appears nowhere else. Zero for the zram
    /// tier, whose store is RAM and waits on nothing.
    pub spin_waits: u64,
}

/// Occupancy snapshot — the compression win, visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZramStats {
    /// Pages currently stored (all classes).
    pub stored_pages: u32,
    /// Of those, pages in the **same-fill** class — stored at ~zero cost.
    pub same_fill_pages: u32,
    /// Page-count ceiling (the zram *disksize* analogue): the advertised
    /// uncompressed capacity, [`DISK_FACTOR`]` × budget` worth of pages —
    /// relative to the endowment by construction.
    pub disk_pages: u32,
    /// Arena bytes packed by the compressed / raw pages (same-fill costs 0).
    pub used_bytes: u64,
    /// The endowed ceiling.
    pub budget_bytes: u64,
    /// Bytes saved vs. storing every page raw + unpacked — compression +
    /// same-fill + packing all fold into one visible win: `stored·PAGE − used`.
    pub saved_bytes: u64,
}

/// A stored page's residence.
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// A free token.
    Vacant,
    /// A **same-filled** page (zram's zero / same-element class): the whole
    /// page is one repeated 8-byte element, stored as just the element —
    /// **no arena bytes**.  Zero pages are the common case.
    SameFill { element: u64 },
    /// Compressed (or, at `len == PAGE_BYTES`, raw) bytes **packed** into the
    /// store arena at `[off, off+len)`.
    Packed { off: u32, len: u32 },
}

/// **Adopt the staging window a request transferred** — the metal half of
/// [`SwapRequest::Attach`], shared by every backend so the capability
/// handling has one implementation.
///
/// The client transferred a writable view over the return channel
/// (`ReturnMode::Retained`); `announced` must name that landing, or the
/// caller gets nothing.  The view is **mapped once and kept forever**:
/// the mapping is what makes byte moves ordinary memory accesses, and a
/// receiver cannot unmap, so the returned owner is deliberately leaked
/// into the service's lifetime rather than dropped (dropping it would
/// retire the record and leave the mapping pointing at reclaimed frames).
///
/// `None` on any refusal — wrong correlation, nothing delivered, a
/// non-view record, or a mapping the kernel declined.
pub fn adopt_staging(
    sv: &mut Serve<'_, SwapResponse>,
    announced: SlotRef,
) -> Option<&'static mut [u8]> {
    let mut input = sv.returned_input(announced)?;
    let owner = input.take()?;
    let slot = owner.slot()?;
    let view = verbs::take_view(slot)?;
    let window = verbs::map_view(view)?;
    // The mapping must outlive every request, so the owner must never be
    // dropped: its Drop retires the delivered record, and the window's
    // frames would be free while this service still addressed them.
    core::mem::forget(owner);
    // SAFETY: `map_view` installed `[base, +len)` R/W-USER in THIS task's
    // own space over the client's staging backing, page-aligned and live
    // for the task's whole life (nothing unmaps it — a receiver has no
    // unmap verb, and the owner above is retained).  The client writes
    // page bytes there and this service reads/writes them under the
    // request/reply protocol, which is the same single-owner-at-a-time
    // discipline the pair rings rely on.
    Some(unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) })
}

/// How many whole pages a staging window holds — the client's own bound,
/// read off the mapping rather than trusted from the wire.
pub fn staging_pages(window: &[u8]) -> u32 {
    (window.len() as u64 / PAGE_BYTES) as u32
}

/// Page `index` of a staging window — `None` when the index lies outside
/// it.  The **bounds check every byte-moving verb goes through**: a page
/// index is client data, and the window's length is the only authority on
/// what it may name.
pub fn staging_page(window: &[u8], index: u32) -> Option<&[u8]> {
    let off = (index as u64).checked_mul(PAGE_BYTES)?;
    let end = off.checked_add(PAGE_BYTES)?;
    if end > window.len() as u64 {
        return None;
    }
    window.get(off as usize..end as usize)
}

/// [`staging_page`]'s mutable twin (the restore destination).
pub fn staging_page_mut(window: &mut [u8], index: u32) -> Option<&mut [u8]> {
    let off = (index as u64).checked_mul(PAGE_BYTES)?;
    let end = off.checked_add(PAGE_BYTES)?;
    if end > window.len() as u64 {
        return None;
    }
    window.get_mut(off as usize..end as usize)
}

/// SIMD-scan a page frame for the same-fill class: `Some(element)` iff every
/// 8-byte word equals the first (a zram same-filled page — zeros, or any
/// repeated element).  512 words / 8-lane = 64 vector compares; the common
/// zero page exits on the first mismatch.
fn same_fill(page: &[u8]) -> Option<u64> {
    use core::simd::{Simd, cmp::SimdPartialEq};
    debug_assert_eq!(page.len(), PAGE_BYTES as usize);
    debug_assert_eq!(page.as_ptr() as usize % 8, 0, "page frame is u64-aligned");
    // SAFETY: a page frame is 4096-aligned ⟹ 8-aligned, and 4096 = 512·8.
    let words = unsafe { core::slice::from_raw_parts(page.as_ptr() as *const u64, 512) };
    let first = Simd::<u64, 8>::splat(words[0]);
    for chunk in words.chunks_exact(8) {
        if Simd::<u64, 8>::from_slice(chunk).simd_ne(first).any() {
            return None;
        }
    }
    Some(words[0])
}

/// SIMD-fill a page frame with a repeated 8-byte `element` — the same-fill
/// restore (the inverse of [`same_fill`]).
fn simd_fill(out: &mut [u8], element: u64) {
    use core::simd::Simd;
    debug_assert_eq!(out.len(), PAGE_BYTES as usize);
    debug_assert_eq!(out.as_ptr() as usize % 8, 0);
    // SAFETY: as in `same_fill`.
    let words = unsafe { core::slice::from_raw_parts_mut(out.as_ptr() as *mut u64, 512) };
    let splat = Simd::<u64, 8>::splat(element);
    for chunk in words.chunks_exact_mut(8) {
        splat.copy_to_slice(chunk);
    }
}

/// The packed-byte arena's backing — where the compressed bytes lawfully
/// live.  Off metal there is no trap to carve through, so the host keeps
/// an ordinary Vec (tests of the codec + accounting); on metal the arena
/// is the endowed store supply itself, carved + mapped once in `bind`.
struct Arena {
    /// Host: an ordinary heap Vec, grown on demand to the budget ceiling.
    #[cfg(not(target_os = "none"))]
    host: Vec<u8>,
    /// Metal: the store supply's `[0, budget)` window, carved + mapped in
    /// `bind` (one session, no heap).  `None` when the carve/map refused —
    /// [`start_on_metal`](ZramSwap::start_on_metal) fails closed on it.
    #[cfg(target_os = "none")]
    mapped: Option<core::ptr::NonNull<[u8]>>,
}

impl Arena {
    /// The host backing: an empty grow-on-demand Vec.
    #[cfg(not(target_os = "none"))]
    fn bind(_store: &SupplyToken) -> Self {
        Self { host: Vec::new() }
    }

    /// The metal backing: carve `[0, budget)` off the endowed store
    /// supply and map it — one ordinary session, no heap (kore-rt's
    /// [`carve_and_map`](kore_rt::heap::carve_and_map), the heap
    /// grow's primitive off this service's OWN supply).  `None` on any
    /// refusal: `start_on_metal` fails closed.
    #[cfg(target_os = "none")]
    fn bind(store: &SupplyToken) -> Self {
        Self {
            mapped: kore_rt::heap::carve_and_map(store.slot(), 0, store.len()),
        }
    }

    /// Whether the backing is live — `start_on_metal`'s fail-closed gate.
    fn live(&self) -> bool {
        #[cfg(not(target_os = "none"))]
        {
            true
        }
        #[cfg(target_os = "none")]
        {
            self.mapped.is_some()
        }
    }

    /// Ensure the span covers `[0, n)` — the host grow; the metal span is
    /// the full budget from `bind`, so this is a no-op there.
    fn cover(&mut self, n: usize) {
        #[cfg(not(target_os = "none"))]
        {
            if self.host.len() < n {
                self.host.resize(n, 0);
            }
        }
        #[cfg(target_os = "none")]
        {
            let _ = n;
        }
    }

    /// The arena bytes, mutably.  Callers stay within the budget ceiling
    /// ([`ZramSwap::arena_alloc`] enforces it).
    fn bytes_mut(&mut self) -> &mut [u8] {
        #[cfg(not(target_os = "none"))]
        {
            &mut self.host
        }
        #[cfg(target_os = "none")]
        {
            let mut p = self.mapped.expect("arena live (gated at start_on_metal)");
            // SAFETY: the mapped span is this task's own R/W carve of its
            // endowed store, live for the task's life; offsets are
            // budget-bounded by `arena_alloc`.
            unsafe { p.as_mut() }
        }
    }

    /// The arena bytes, shared.
    fn bytes(&self) -> &[u8] {
        #[cfg(not(target_os = "none"))]
        {
            &self.host
        }
        #[cfg(target_os = "none")]
        {
            let p = self.mapped.expect("arena live (gated at start_on_metal)");
            // SAFETY: as `bytes_mut` — a read-only view of the same span.
            unsafe { p.as_ref() }
        }
    }
}

/// The default in-RAM swap backend: LZ4-compress evicted pages into a
/// **slab-packed**, budget-accounted store, with a **same-fill fast path**
/// (SIMD-scanned) that stores zero / same-element pages for free.
pub struct ZramSwap {
    /// The possessed budget — `len()` is the arena ceiling.
    store: SupplyToken,
    /// Token table — each stored page's residence.  Sized at bind to
    /// [`DISK_FACTOR`] × the budget's page count (the *disksize*), so
    /// capacity scales with the endowment — never a magic absolute.
    slots: Vec<Slot>,
    /// The packed compressed-byte arena (a zsmalloc-analogue): compressed
    /// and raw pages share it contiguously, so a 500-byte page costs 500
    /// bytes, not a whole frame.  On metal it IS the endowed store,
    /// carved + mapped at bind, capped by the budget.
    arena: Arena,
    /// The **service-owned** compression scratch — `PAGE_BYTES + 512`
    /// (LZ4's worst-case block output for one frame, well over the
    /// `n + n/255 + 16` bound).  Allocated once at bind and reused across
    /// every page-out, so the compress path never puts a ~4.6 KiB buffer on
    /// the stack per call (§7 named finding — the former per-call
    /// `[0u8; PAGE_BYTES + 512]`).
    scratch: Vec<u8>,
    /// Bump high-water into `arena`; reuse comes off `free` first.
    cursor: usize,
    /// Freed `(off, len)` arena regions, first-fit-reused (split on reuse;
    /// no coalescing — the host codec model, not capability-backed zsmalloc).
    free: Vec<(u32, u32)>,
    /// Live packed bytes (arena occupancy; same-fill contributes nothing).
    used: u64,
    /// Pages stored (all classes).
    stored: u32,
    /// Pages in the same-fill class (stored at ~zero cost).
    same_fill: u32,
    /// **The attached staging window** — the client's shared page buffer,
    /// mapped once at [`SwapRequest::Attach`] and addressed by page index
    /// thereafter.  `None` until the client attaches; every byte-moving
    /// verb refuses [`SwapRefusal::NoStaging`] until then.
    ///
    /// `'static` because the mapping outlives every request: a receiver
    /// cannot unmap, so the window is installed for the service's life
    /// (which is exactly why it is attached once and not per operation).
    staging: Option<&'static mut [u8]>,
    /// `Out` ops answered `Stored` — the [`SwapStatus`] books (occupancy
    /// alone cannot recover op counts: an out+discard pair leaves the
    /// store as it found it).
    outs: u64,
    /// `In` ops answered `Restored`.
    ins: u64,
    /// `Discard` ops answered `Discarded`.
    discards: u64,
}

impl ZramSwap {
    /// Occupancy snapshot.
    pub fn stats(&self) -> ZramStats {
        ZramStats {
            stored_pages: self.stored,
            same_fill_pages: self.same_fill,
            disk_pages: self.slots.len() as u32,
            used_bytes: self.used,
            budget_bytes: self.store.len(),
            saved_bytes: self.stored as u64 * PAGE_BYTES - self.used,
        }
    }

    /// The tier's books, in the shared [`SwapStatus`] shape: RAM-backed,
    /// so the device bit is always set and the io/route words stay zero.
    pub fn status(&self) -> SwapStatus {
        let s = self.stats();
        SwapStatus {
            device: true,
            route: false,
            capacity_pages: s.disk_pages,
            stored_pages: s.stored_pages,
            staging_pages: self.staging_pages(),
            outs: self.outs,
            ins: self.ins,
            discards: self.discards,
            ..SwapStatus::default()
        }
    }

    /// Reserve `len` packed bytes: first-fit a freed region (splitting the
    /// remainder back), else bump the arena within the budget ceiling.
    fn arena_alloc(&mut self, len: usize) -> Option<u32> {
        let Self {
            free,
            cursor,
            arena,
            store,
            ..
        } = self;
        Self::arena_alloc_in(free, cursor, arena, store, len)
    }

    /// [`arena_alloc`](Self::arena_alloc) over borrowed fields — the form
    /// the byte-moving paths call, which hold a disjoint borrow of
    /// `staging` at the same time.
    fn arena_alloc_in(
        free: &mut Vec<(u32, u32)>,
        cursor: &mut usize,
        arena: &mut Arena,
        store: &SupplyToken,
        len: usize,
    ) -> Option<u32> {
        for i in 0..free.len() {
            let (off, flen) = free[i];
            if flen as usize >= len {
                if flen as usize == len {
                    free.swap_remove(i);
                } else {
                    free[i] = (off + len as u32, flen - len as u32);
                }
                return Some(off);
            }
        }
        let off = *cursor;
        if off + len > store.len() as usize {
            return None; // budget ceiling
        }
        arena.cover(off + len);
        *cursor = off + len;
        Some(off as u32)
    }

    /// Return a packed region to the free list.
    fn arena_free(&mut self, off: u32, len: u32) {
        self.free.push((off, len));
    }

    /// Page-out: read the staging page; store same-fill pages for free
    /// (SIMD-scanned), else compress (or keep raw) and pack into the arena.
    fn page_out(&mut self, index: u32) -> Result<u32, SwapRefusal> {
        // Disjoint field borrows: the staging read, the compression
        // scratch, and the arena are three separate fields, so destructure
        // rather than borrow `self` three ways (the `scratch`/`arena`
        // discipline this function already followed, one field wider).
        let Self {
            staging,
            slots,
            scratch,
            arena,
            free,
            cursor,
            store,
            used,
            stored,
            same_fill: same_fill_count,
            ..
        } = self;
        let page = staging
            .as_deref()
            .and_then(|s| staging_page(s, index))
            .ok_or(SwapRefusal::NoStaging)?;

        // Claim a token first — fail before touching the store.
        let token = slots
            .iter()
            .position(|s| matches!(s, Slot::Vacant))
            .ok_or(SwapRefusal::StoreFull)?;

        // Same-fill fast path — no arena bytes.
        if let Some(element) = same_fill(page) {
            slots[token] = Slot::SameFill { element };
            *stored += 1;
            *same_fill_count += 1;
            return Ok(token as u32);
        }

        // Compress into the service-owned scratch, or keep raw when LZ4
        // cannot beat the page.  `Some(n)` = compressed to `n` bytes in
        // `scratch[..n]`; `None` = the raw huge-page class (never store
        // an expansion).
        let compressed = match lz4_flex::block::compress_into(page, scratch) {
            Ok(n) if (n as u64) < PAGE_BYTES => Some(n),
            _ => None,
        };
        let len = compressed.unwrap_or(PAGE_BYTES as usize);

        // Pack into the arena — the budget check *is* the allocation.
        let off =
            Self::arena_alloc_in(free, cursor, arena, store, len).ok_or(SwapRefusal::StoreFull)?;
        let dst = &mut arena.bytes_mut()[off as usize..off as usize + len];
        match compressed {
            Some(n) => dst.copy_from_slice(&scratch[..n]),
            None => dst.copy_from_slice(page),
        }
        slots[token] = Slot::Packed {
            off,
            len: len as u32,
        };
        *used += len as u64;
        *stored += 1;
        Ok(token as u32)
    }

    /// Page-in: restore the stored page into the staging page; success
    /// consumes the copy and frees its arena region.
    fn page_in(&mut self, token: u32, index: u32) -> Result<(), SwapRefusal> {
        // Validate the token first — a bad token must not touch staging.
        let residence = match self.slots.get(token as usize) {
            Some(s) if !matches!(*s, Slot::Vacant) => *s,
            _ => return Err(SwapRefusal::BadToken),
        };
        // Disjoint borrows again: the destination is inside `staging`, the
        // source inside `arena`.
        let Self {
            staging,
            arena,
            free,
            used,
            same_fill: same_fill_count,
            slots,
            stored,
            ..
        } = self;
        let out = staging
            .as_deref_mut()
            .and_then(|s| staging_page_mut(s, index))
            .ok_or(SwapRefusal::NoStaging)?;

        match residence {
            Slot::SameFill { element } => {
                simd_fill(out, element);
                *same_fill_count -= 1;
            }
            Slot::Packed { off, len } => {
                let bytes = &arena.bytes()[off as usize..off as usize + len as usize];
                if len as u64 == PAGE_BYTES {
                    out.copy_from_slice(bytes); // raw
                } else if !matches!(
                    lz4_flex::block::decompress_into(bytes, out),
                    Ok(n) if n as u64 == PAGE_BYTES
                ) {
                    // A stored copy that no longer restores is a service bug;
                    // fail closed, keep nothing.
                    return Err(SwapRefusal::BadToken);
                }
                free.push((off, len));
                *used -= len as u64;
            }
            Slot::Vacant => unreachable!("validated above"),
        }
        slots[token as usize] = Slot::Vacant;
        *stored -= 1;
        Ok(())
    }

    /// **Adopt a staging window** — the mapped bytes and the page count
    /// they hold.  Shared by the metal `Attach` path (which maps the
    /// transferred view first) and by host tests (which hand over an
    /// ordinary leaked buffer), so the datapath has exactly one shape and
    /// no `cfg(test)` divergence.
    ///
    /// A window shorter than one page attaches nothing: the client asked
    /// for a buffer that cannot carry a page.
    pub fn attach_staging(&mut self, window: &'static mut [u8]) -> Result<u32, SwapRefusal> {
        let pages = staging_pages(window);
        if pages == 0 {
            return Err(SwapRefusal::NoStaging);
        }
        self.staging = Some(window);
        Ok(pages)
    }

    /// The attached staging capacity in pages (`0` = unattached).
    pub fn staging_pages(&self) -> u32 {
        self.staging.as_deref().map_or(0, staging_pages)
    }

    fn discard(&mut self, token: u32) -> Result<(), SwapRefusal> {
        let residence = match self.slots.get(token as usize) {
            Some(s) if !matches!(*s, Slot::Vacant) => *s,
            _ => return Err(SwapRefusal::BadToken),
        };
        match residence {
            Slot::SameFill { .. } => self.same_fill -= 1,
            Slot::Packed { off, len } => {
                self.arena_free(off, len);
                self.used -= len as u64;
            }
            Slot::Vacant => unreachable!(),
        }
        self.slots[token as usize] = Slot::Vacant;
        self.stored -= 1;
        Ok(())
    }
}

impl SystemService for ZramSwap {
    const NAME: &'static str = "zram";

    /// The staging window is **retained** service authority: it arrives
    /// once over the return channel ([`SwapRequest::Attach`]) and lives
    /// as long as the service does, so the host must donate a
    /// runtime-owned landing per transfer rather than the reusable fixed
    /// one (the image service's registrar shape).
    const RETURN_MODE: kore_rt::service::ReturnMode = kore_rt::service::ReturnMode::Retained {
        kind: kore_abi::cap::ResourceKind::LeasedRegion,
        // A staging window is written from both sides: the client stages
        // page bytes for `Out`, the service restores into it for `In`.
        minimum_rights: kore_abi::cap::Rights::ACCESS.union(kore_abi::cap::Rights::MODIFY),
    };
    type Endowment = ZramEndowment;

    fn bind(e: ZramEndowment) -> Self {
        let disk = (e.store.len() * DISK_FACTOR / PAGE_BYTES) as usize;
        Self {
            arena: Arena::bind(&e.store),
            store: e.store,
            slots: alloc::vec![Slot::Vacant; disk],
            scratch: alloc::vec![0u8; PAGE_BYTES as usize + 512],
            cursor: 0,
            free: Vec::new(),
            used: 0,
            stored: 0,
            same_fill: 0,
            staging: None,
            outs: 0,
            ins: 0,
            discards: 0,
        }
    }

    /// The store backing must have materialized — a service whose
    /// endowed store never carved/mapped must fail closed, never beacon
    /// (the same discipline as the output backend's `online()`).
    /// The #407 thread witness rides swap (the simplest founded row):
    /// a second thread of this service paces itself on pure deadline
    /// parks while thread 0 serves doorbell-parked — the unified task
    /// model's independent-parks proof, observed by the kernel's
    /// `thread-founding:` / `thread-parks:` gate markers.
    ///
    /// **Declared, not spawned**: the runtime turns this into a thread
    /// (`run_service` → `spawn_workers`), so this service — like every
    /// other — carries no spawn code, no ordering decision, and no
    /// refusal handling of its own.
    const WORKERS: &'static [kore_rt::service::Worker] = &[kore_rt::service::Worker {
        name: "swap-witness",
        entry: thread_witness,
    }];

    fn start_on_metal(
        &mut self,
        _rt: &kore_rt::entry::Rt,
    ) -> Result<(), kore_rt::service::StartFailure> {
        kore_rt::service::StartFailure::require(self.arena.live())
    }

    /// Serve one swap op.  **Never calls [`Serve::grant`]** — a swap
    /// backend holds pages, it mints nothing.
    fn serve(&mut self, req: SwapRequest, sv: &mut Serve<'_, SwapResponse>) {
        let resp = match req {
            SwapRequest::Attach { announced } => match adopt_staging(sv, announced) {
                Some(window) => match self.attach_staging(window) {
                    Ok(pages) => SwapResponse::Attached { pages },
                    Err(r) => SwapResponse::Refused(r),
                },
                None => SwapResponse::Refused(SwapRefusal::NoStaging),
            },
            SwapRequest::Out { page } => match self.page_out(page) {
                Ok(token) => {
                    self.outs += 1;
                    SwapResponse::Stored { token }
                }
                Err(r) => SwapResponse::Refused(r),
            },
            SwapRequest::In { token, page } => match self.page_in(token, page) {
                Ok(()) => {
                    self.ins += 1;
                    SwapResponse::Restored
                }
                Err(r) => SwapResponse::Refused(r),
            },
            SwapRequest::Discard { token } => match self.discard(token) {
                Ok(()) => {
                    self.discards += 1;
                    SwapResponse::Discarded
                }
                Err(r) => SwapResponse::Refused(r),
            },
            SwapRequest::Status => SwapResponse::Status(self.status()),
        };
        sv.reply(resp);
    }
}

/// The #407 witness worker: learn "now" from an already-past deadline
/// park's clock stamp, take three real 5 ms deadline parks (each a full
/// ring-3 park/resume round trip through the timer wheel), and **return**.
///
/// The return is now part of what it witnesses.  This worker used to park
/// in day-long strides afterwards — the honest "forever" when threads had
/// no exit verb — and the trampoline turns the return into
/// `thread::exit(ExitCode::OK)` instead: a thread that has finished its
/// work says so, and its status is there for a joiner to read.
fn thread_witness() {
    use kore_rt::cap::verbs;

    let mut now_ms = match verbs::sleep_until(1) {
        Some(word) => (word >> 32) as u64,
        None => 0,
    };
    for _ in 0..3 {
        if let Some(word) = verbs::sleep_until((now_ms + 5) * 1_000_000) {
            now_ms = (word >> 32) as u64;
        }
    }
    // The G7 join storm runs FIRST, so a wedged join blocks the gang
    // witness below and its `spmd-gang:` gate markers go missing — the
    // storm's completion witness costs no marker of its own.
    if !join_storm() {
        return;
    }
    // The G5 gang witness rides the same simplest founded row: a
    // one-shot width-2 SPMD region — one gang session, two barrier
    // phases, joined — so the boot gate's `spmd-gang: founded 2/2`
    // marker proves N-founds-in-one-trap end to end on every CI boot
    // (SPMD I6: a frozen-contract change is metal-proven, never
    // mocked).  Width 2 is boot-invariant on the gate's -smp range; the
    // star degrades honestly here (this row holds no CHANNEL_SPACE) and
    // the pins ride the std class under QEMU.  A refusal is loud on
    // serial via the missing marker, and this witness thread simply
    // exits — the service row it rides is already serving.
    //
    // The joins race the workers' exits DELIBERATELY (no pacing): the
    // park-vs-exit window is kernel-proven now — the storm above and the
    // gate's `join-park: PASS` line — so nothing here may rely on the
    // probe-first fast path winning.
    if let Some(region) = gang_witness() {
        let mut region = region;
        for _ in 0..16 {
            match region.join_all() {
                Ok(_) => break,
                Err(refused) => region = refused.spmd,
            }
        }
    }
}

/// **The G7 join storm** — the WAIT_TASK park-vs-exit race, sampled on
/// every gated boot: short-lived siblings plus one explicitly readied lingerer
/// joined by this doorbell-less thread, so at least one join reaches the kernel
/// while its joinee is live (the shape that wedged under TCG timing) instead of
/// every round riding the probe's already-exited fast path.
///
/// Two child shapes:
/// - **sprinters** exit the instant they run — the join commits its park
///   in the same window the exit publishes, sampling the Dekker pairing
///   (`install_joiner` vs `mark_exited`/drain) under real SMP timing;
/// - the first child publishes READY and stays in short deadline parks until
///   the parent publishes GO immediately before joining it. On one standard
///   CPU, the child cannot observe GO until the running parent parks or rotates;
///   on SMP, its following one-second delay is a deliberately large bounded
///   margin, not a scheduler-latency proof. The join probe therefore exercises
///   a live child instead of relying on a 3 ms timing guess. Its exit drain must
///   deliver the installed waiter's wake, which is what the kernel's one-shot
///   `join-park: PASS` marker attests.
///
/// A join that never resolves wedges this witness right here, and the
/// gang markers downstream (`spmd-gang: founded 2/2`, `retired epoch`)
/// go missing — loud on serial, the witness idiom.
static STORM_LINGERER_READY: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
static STORM_LINGERER_GO: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

fn join_storm() -> bool {
    use core::sync::atomic::Ordering;
    use kore_rt::cap::verbs;

    for round in 0u32..16 {
        let linger = round == 0;
        let entry: fn() = if linger {
            STORM_LINGERER_READY.store(false, Ordering::Release);
            STORM_LINGERER_GO.store(false, Ordering::Release);
            storm_lingerer
        } else {
            storm_sprinter
        };
        let Some(mut child) = kore_rt::thread::spawn(entry) else {
            // A refused founding is the gang witness's failure to report
            // (its founding markers go missing); the storm judges joins.
            return false;
        };
        if linger {
            // The child publishes before waiting for GO. Pace this readiness
            // wait on the deadline path available to every thread; a missing
            // publish fails after a bounded interval and suppresses the
            // downstream completion markers instead of becoming a userspace
            // spin.
            let Some(word) = verbs::sleep_until(1) else {
                return false;
            };
            let mut ms = (word >> 32) as u64;
            let ready_deadline_ms = ms.saturating_add(5_000);
            while !STORM_LINGERER_READY.load(Ordering::Acquire) {
                if ms >= ready_deadline_ms {
                    return false;
                }
                let deadline_ms = ms.saturating_add(2).min(ready_deadline_ms);
                let Some(word) = verbs::sleep_until(deadline_ms * 1_000_000) else {
                    return false;
                };
                ms = (word >> 32) as u64;
            }
            // This is deliberately adjacent to join: READY keeps the child in
            // deadline parks until this publication, and the child's long exit
            // delay starts only after it observes GO.
            STORM_LINGERER_GO.store(true, Ordering::Release);
        }
        loop {
            match kore_rt::thread::join(child) {
                Ok(_) => break,
                Err(refused) => {
                    // The affine ticket rides back on a typed refusal
                    // (slots, transport).  Pace the retry on the deadline
                    // park every thread can take; a persistent refusal
                    // wedges here, and the missing gang markers say so.
                    child = refused.into_thread();
                    if let Some(word) = verbs::sleep_until(1) {
                        let ms = (word >> 32) as u64;
                        let _ = verbs::sleep_until((ms + 2) * 1_000_000);
                    }
                }
            }
        }
    }
    true
}

/// A storm child that exits the instant it runs: its trampoline return
/// becomes `thread::exit(ExitCode::OK)` while the parent's join is still
/// committing.
fn storm_sprinter() {}

/// A storm child that proves readiness, waits for its parent's immediate-join
/// handoff, then stays live for a large bounded margin while the parent installs
/// its task waiter.
fn storm_lingerer() {
    use core::sync::atomic::Ordering;
    use kore_rt::cap::verbs;

    let Some(word) = verbs::sleep_until(1) else {
        return;
    };
    let mut ms = (word >> 32) as u64;
    let go_deadline_ms = ms.saturating_add(5_000);
    STORM_LINGERER_READY.store(true, Ordering::Release);
    while !STORM_LINGERER_GO.load(Ordering::Acquire) {
        if ms >= go_deadline_ms {
            return;
        }
        let deadline_ms = ms.saturating_add(2).min(go_deadline_ms);
        let Some(word) = verbs::sleep_until(deadline_ms * 1_000_000) else {
            return;
        };
        ms = (word >> 32) as u64;
    }
    let _ = verbs::sleep_until((ms + 1_000) * 1_000_000);
}

/// Two workers, two barrier phases, exit — the least region that
/// exercises the gang founding, the launch gate, and the barrier.
#[kore_rt::spmd(1, 1, 2)]
fn gang_witness(ctx: &'static kore_rt::spmd::SpmdCtx, _worker: u32) {
    ctx.barrier().arrive_and_wait();
    ctx.barrier().arrive_and_wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leaked staging window of `pages` pages — the host stand-in for
    /// the mapping `Attach` installs on metal.  Leaked because the real
    /// one is `'static` by construction (a receiver cannot unmap).
    fn staging(pages: usize) -> &'static mut [u8] {
        alloc::vec![0u8; pages * PAGE_BYTES as usize].leak()
    }

    /// A bound zram over a geometry-only host supply.
    fn bound(budget_pages: u64) -> ZramSwap {
        ZramSwap::bind(ZramEndowment {
            store: SupplyToken::for_test(0, budget_pages * PAGE_BYTES),
        })
    }

    /// The datapath, end to end: stage a compressible page, `Out` it,
    /// clobber staging, `In` it back, and the bytes are identical.  This
    /// is what the per-op-delivered-view shape could never test on the
    /// host — the whole point of the staging window.
    #[test]
    fn a_page_round_trips_through_the_staging_window() {
        let mut z = bound(64);
        let pages = z.attach_staging(staging(4)).expect("staging attaches");
        assert_eq!(pages, 4);

        // A compressible but not same-fill pattern.
        let page: Vec<u8> = (0..PAGE_BYTES as usize).map(|i| (i / 64) as u8).collect();
        let out = staging_page_mut(z.staging.as_deref_mut().unwrap(), 1).unwrap();
        out.copy_from_slice(&page);

        let token = z.page_out(1).expect("page-out stores");
        // Clobber staging so a restore that silently no-ops cannot pass.
        staging_page_mut(z.staging.as_deref_mut().unwrap(), 1)
            .unwrap()
            .fill(0xAA);
        z.page_in(token, 1).expect("page-in restores");
        assert_eq!(
            staging_page(z.staging.as_deref().unwrap(), 1).unwrap(),
            &page[..],
            "the restored bytes are the stored bytes"
        );
        assert_eq!(z.stats().stored_pages, 0, "restore consumed the copy");
    }

    /// The same-fill class round-trips too, and costs no arena bytes.
    #[test]
    fn a_same_fill_page_costs_nothing_and_restores() {
        let mut z = bound(64);
        z.attach_staging(staging(2)).unwrap();
        staging_page_mut(z.staging.as_deref_mut().unwrap(), 0)
            .unwrap()
            .fill(0);

        let token = z.page_out(0).expect("same-fill stores");
        assert_eq!(z.stats().same_fill_pages, 1);
        assert_eq!(z.stats().used_bytes, 0, "same-fill uses no arena bytes");

        staging_page_mut(z.staging.as_deref_mut().unwrap(), 0)
            .unwrap()
            .fill(0xFF);
        z.page_in(token, 0).expect("same-fill restores");
        assert!(
            staging_page(z.staging.as_deref().unwrap(), 0)
                .unwrap()
                .iter()
                .all(|&b| b == 0),
            "the zero page came back zero"
        );
    }

    /// Every byte-moving verb refuses before an `Attach`, and refuses a
    /// page index outside the attached window — the client's index is
    /// data, and the window's length is the only authority on it.
    #[test]
    fn staging_bounds_and_absence_are_refusals_not_panics() {
        let mut z = bound(64);
        assert_eq!(z.page_out(0), Err(SwapRefusal::NoStaging), "no window yet");
        assert_eq!(z.staging_pages(), 0);

        z.attach_staging(staging(2)).unwrap();
        assert_eq!(z.staging_pages(), 2);
        assert_eq!(
            z.page_out(2),
            Err(SwapRefusal::NoStaging),
            "one past the last page"
        );
        assert_eq!(
            z.page_out(u32::MAX),
            Err(SwapRefusal::NoStaging),
            "an index whose byte offset overflows"
        );

        let token = z.page_out(0).expect("in-bounds page stores");
        assert_eq!(
            z.page_in(token, 2),
            Err(SwapRefusal::NoStaging),
            "restore is bounds-checked too"
        );
        // The refused restore consumed nothing.
        assert_eq!(z.stats().stored_pages, 1);
        z.page_in(token, 0)
            .expect("the in-bounds restore still works");
    }

    /// A window too short for one page attaches nothing.
    #[test]
    fn a_sub_page_window_is_refused() {
        let mut z = bound(64);
        let short: &'static mut [u8] = alloc::vec![0u8; 64].leak();
        assert_eq!(z.attach_staging(short), Err(SwapRefusal::NoStaging));
    }
}
