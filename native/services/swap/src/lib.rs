//! # kore-svc-swap — the swap service
//!
//! **The `SwapBackend` seam, rewritten as a service**:
//! [`SwapRequest`] is the vocabulary, and *implementations are services* —
//! [`ZramSwap`] (compress into RAM) now, a disk driver implementing the
//! same `SystemService<Request = SwapRequest>` later (`kore-nvme`'s
//! `service` feature is exactly that face).
//!
//! **The lessor inversion.** Swap is not a kernel subsystem: residency —
//! *where* an evicted page went (a compressed-buffer id, a block LBA) —
//! is the lessor's private bookkeeping, and the swap token space is the
//! *server's*; the kernel never interprets a token and holds no swap
//! table (its whole footprint is the paged-out PTE marker, the fault
//! routing in `src/runtime/mm/pager.rs`, and the pager session verbs).
//!
//! **The seal + rollback resistance.** An evicted page transits
//! untrusted userland, so the kernel seals it in place
//! (ChaCha20-Poly1305 AEAD, `src/schema/seal.rs`; the key derives from
//! the entropy-seeded token master, domain-separated from every
//! capability epoch key, never exposed): this service stores only
//! ciphertext and can neither read a tenant's plaintext nor feed back
//! tampered bytes.  The `(nonce, tag)` authenticator is *authentication*
//! state, not residency — and the kernel holding the **latest** tag per
//! frame is what buys **rollback resistance**: a backend replaying a
//! stale but validly-sealed ciphertext fails the check.  (The recorded
//! fork — zero kernel state, authenticators stored beside the
//! ciphertext — costs exactly the rollback property.)
//!
//! One concern per module (`res/USERLAND.md` §1.2 rule 4; `main.rs` —
//! the process entry — arrives with the founder's spawn, #96/#298):
//!
//! - [`service`] — the default in-RAM (zram) backend: LZ4 + same-fill
//!   SIMD fast path, budget-accounted slab packing.
//! - [`client`] — the client seam: the async [`SwapClient`] (§4 pinned),
//!   the [`RingSwap`] ring + channel proxy, and the pager doorbell.
//! - [`wire`] — the fail-closed enum ⇄ wire codecs over
//!   [`kore_abi::svc`]'s frozen forms (§5.3).
//!
//! The former `SwapRegion` lessor façade is intentionally absent: it
//! depended on never-wired `view_of_window`/`deliver_view` verbs and a
//! hard-coded ambient holder.  The next demand-paging client will attach one
//! staging window ([`SwapRequest::Attach`]) and move bounded immutable extents
//! through it: store is all-or-nothing, load is non-destructive, and release
//! is explicit.  No pager or residency consumer is founded yet; the service
//! contract is deliberately model-neutral and does not interpret KV geometry.
//!
//! The zram engine's codec, accounting, and **datapath** are host-tested
//! end to end (the staging window is ordinary memory, so a round-trip is
//! an ordinary test).  On metal the store bytes come from the endowed
//! supply itself, carved and mapped at bind.

#![no_std]
// Typed portable SIMD for the zram same-fill scan/fill (service.rs) — the
// doctrine's "typed over hand-rolled asm" for the vector fast path.
#![feature(portable_simd)]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod client;
pub mod extent;
pub mod service;
pub mod wire;

pub use client::{PagerDoorbell, PagerEvent, RingSwap, SwapClient};
pub use extent::{ExtentTable, MAX_EXTENT_PAGES, SwapExtent};
pub use service::{
    DISK_FACTOR, SwapRefusal, SwapRequest, SwapResponse, SwapStatus, ZramEndowment, ZramStats,
    ZramSwap, adopt_staging, staging_extent, staging_extent_mut, staging_page, staging_page_mut,
    staging_pages,
};
