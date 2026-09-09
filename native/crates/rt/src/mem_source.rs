//! Allocation-free backing-source contract shared by retained arenas and
//! the optional userland heap engine.
//!
//! The seam is smaller than either consumer: it promises one exclusively
//! owned, page-aligned mapped extent and nothing about how that extent is
//! carved afterwards.  Keeping the trait outside the `heap` feature lets a
//! heapless program provision an audited stack arena without linking
//! `alloc` or the global-allocator machinery.

use core::ptr::NonNull;

/// Alignment every mapped extent supplied through this seam must satisfy.
pub const ALIGN: usize = 4096;

/// A source of fresh, exclusively owned mapped extents.
///
/// [`map_extent`](MemSource::map_extent) returns an [`ALIGN`]-aligned region
/// of at least `min_bytes`, or `None` when the source is exhausted.  The
/// consumer owns the returned bytes exclusively and decides whether to
/// self-host allocator metadata or publish a page subset through an arena
/// ledger.  Refusal stays typed; this contract never aborts.
///
/// # Safety
///
/// Every successful call must return a region that:
///
/// - is readable and writable for the lifetime of the consuming allocator or
///   retained arena;
/// - starts on an [`ALIGN`] boundary and contains at least `min_bytes`;
/// - does not overlap any region previously returned by this source; and
/// - is not read, written, remapped, or revoked by the source after return.
///
/// Those are ownership and lifetime proofs, not geometry that a consumer can
/// recover from a raw slice.  Keeping this an `unsafe trait` prevents a safe
/// implementation from manufacturing aliased allocator backing and moving the
/// resulting undefined behaviour into otherwise-safe `Heap::allocate` calls.
pub unsafe trait MemSource {
    /// Map a fresh [`ALIGN`]-aligned extent of at least `min_bytes`.
    /// The returned length is the actual usable size and may be larger.
    fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>>;
}
