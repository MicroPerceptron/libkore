//! Capacity and shape constants — the whole of the engine's memory budget.
//!
//! Every bound is named rather than discovered at runtime, and exhaustion is
//! a typed refusal ([`crate::FsError`]'s `OutOf*` family), never a
//! panic or a hidden allocation. Every arena in the engine is sized from a
//! constant here, so the service's footprint is a compile-time sum.
//!
//! # Sizing rationale
//!
//! These are measured bootstrap bounds, not a durable storage promise. They
//! cover the founded-service and host-test workloads while keeping a footprint
//! that a founded service can hold without a large `.bss`:
//!
//! | Constant | Value | Dominant cost |
//! |---|---|---|
//! | [`MAX_OBJECTS`] | 128 | 128 × ~48 B ≈ 6 KiB |
//! | [`MAX_DIRECTORIES`] × [`MAX_DIR_ENTRIES`] | 16 × 128 | ≈ 96 KiB |
//! | [`MAX_NAME_LEN`] | 32 | (per entry, inline) |
//! | [`RAM_CHUNKS`] × [`RAM_CHUNK_BYTES`] | 64 × 256 | 16 KiB pool |
//! | `MAX_SESSIONS` (service-side) × [`MAX_HANDLES_PER_SESSION`] | 8 × 16 | ≈ 22 KiB |
//! | [`MAX_PLACEMENTS`] | 72 | ≈ 5 KiB RAM placement table |
//! | [`MAX_BLOCK_PLACEMENTS`] | 256 | file (≤128) + durable-derived (≤128); block provider only |
//! | [`MAX_BLOCK_EXTENT_NODES`] | 1539 | ≈ 68 KiB heap-backed nodes + free heap; block provider only |
//! | [`MAX_AUTHORITY_EDGES`] | 64 | ≈ 4 KiB |
//!
//! Directory child lists live in their own arena rather than inside
//! [`MAX_OBJECTS`] precisely so a *file* object does not reserve directory
//! space: an inline `Body::Directory` would have made every object as large
//! as the largest directory.
//!
//! The whole engine comes to roughly **140 KiB**.  That number is load-bearing,
//! not incidental: an engine is moved by value on construction, and an
//! unoptimized build does not elide those moves, so an engine of a few hundred
//! KiB overflows a test thread's stack before a single assertion runs.  A
//! workload that needs a bigger pool should raise [`RAM_CHUNKS`] *and* have
//! its callers hold the engine behind a box.
//!
//! ## Deliberately compact presentation bounds
//!
//! [`MAX_NAME_LEN`] is **32**, not POSIX's common 255-byte ceiling. Names are
//! stored inline in a directory's entry slots, so the
//! component maximum multiplies by [`MAX_DIR_ENTRIES`] × [`MAX_DIRECTORIES`]:
//! at 255 bytes the directory arena alone would be ~140 KiB, more than
//! everything else combined. Thirty-two bytes covers every current native
//! service, shell, and gate name.
//!
//! [`MAX_PATH_DEPTH`] is **8** for the same reason from the other direction:
//! it is the length of the edge proof stored in *every* open handle, so it
//! multiplies by the handle arena.  Eight levels of nesting below a client's
//! own root covers the current bounded profile.
//!
//! Neither value is assumed anywhere in the engine — both are bounds, not
//! layouts, and raising either is a one-line change plus a footprint
//! recalculation.

/// Maximum live objects (files + directories) in the namespace.  Exhaustion
/// refuses with [`FsError::OutOfObjects`](crate::FsError::OutOfObjects).
pub const MAX_OBJECTS: usize = 128;

/// Maximum live directory *child lists*.  Every directory object owns one,
/// so this is the ceiling on live directories — necessarily
/// `<= MAX_OBJECTS`.  Exhaustion refuses with
/// [`FsError::OutOfObjects`](crate::FsError::OutOfObjects), since from
/// a client's view the thing that could not be created is the directory.
pub const MAX_DIRECTORIES: usize = 16;

/// Maximum containment edges beneath one directory.  Exhaustion refuses with
/// [`FsError::OutOfEdges`](crate::FsError::OutOfEdges).
pub const MAX_DIR_ENTRIES: usize = 128;

/// Maximum bytes in one name component.  See the deviation note in the
/// module docs.  Names are non-empty, so the legal range is `1..=MAX_NAME_LEN`.
pub const MAX_NAME_LEN: usize = 32;

/// Maximum containment edges a single traversal may follow, and therefore
/// the depth of the bounded edge proof an open handle carries
/// (the service's `EdgeProof`).  A path deeper than this refuses with
/// [`FsError::Range`](crate::FsError::Range) rather than truncating.
pub const MAX_PATH_DEPTH: usize = 8;

/// Stable task views retained independently of short-lived command pairs.
/// Each view owns one private root and is selected only by an authenticated
/// endpoint lineage. Four covers the two interactive terminals plus bounded
/// headroom without turning the table into a task roster.
pub const MAX_TASK_VIEWS: usize = 4;

/// Maximum open handles held by one session.  Exhaustion refuses with
/// [`FsError::OutOfHandles`](crate::FsError::OutOfHandles).
pub const MAX_HANDLES_PER_SESSION: usize = 16;

/// Maximum live authority edges — private roots plus direct grants — across
/// all sessions.  Exhaustion refuses with
/// [`FsError::OutOfEdges`](crate::FsError::OutOfEdges).
pub const MAX_AUTHORITY_EDGES: usize = 64;

/// Maximum directory entries returned in one `list`
/// page.  A larger directory is read across several calls via the returned
/// cursor; the engine never returns a partial entry or a short page with
/// more entries silently dropped.
pub const MAX_LIST_PAGE: usize = 8;

// ── RAM provider ────────────────────────────────────────────────────────────

/// Bytes per RAM-provider chunk — the pool's allocation granularity.
pub const RAM_CHUNK_BYTES: usize = 256;

/// Chunks in the RAM provider pool.  `RAM_CHUNKS * RAM_CHUNK_BYTES` (16 KiB)
/// is the RAM provider's total byte bound.
pub const RAM_CHUNKS: usize = 64;

/// Maximum live placements in the RAM provider — one per file that has ever
/// needed bytes.
///
/// Below [`MAX_OBJECTS`] on purpose: directories never take a placement, and
/// neither does a file that has not been written, so placements are strictly
/// scarcer than objects.  Running out is
/// [`FsError::OutOfStorage`](crate::FsError::OutOfStorage) — the file
/// exists, but its bytes cannot be placed.
/// Includes headroom for provider-backed derived batches in addition to file
/// payloads. RAM never duplicates derived bytes; the durable provider uses at
/// most [`MAX_DERIVED_RESULTS`] extra placements.
pub const MAX_PLACEMENTS: usize = 72;

/// Durable block providers retain file extents AND provider-backed derived
/// (search-index) batches in ONE placement table, so it must hold the
/// worst-case SUM of the two, not `MAX_OBJECTS` alone.
///
/// It was `MAX_OBJECTS`, which coupled the two classes: a volume whose file
/// objects filled the object table left no placement for a derived batch, so
/// `find --reindex` on an accumulated durable volume refused `OutOfStorage`
/// mid-index — and, before the witness tolerated it, wedged the first shell's
/// founding. A volume may hold up to `MAX_OBJECTS` file objects at the same
/// time as up to `MAX_DERIVED_RESULTS` durable batches, so the table must
/// cover their sum for neither class to starve the other.
///
/// Lives in the block provider only. The RAM engine — whose by-value move is
/// the stack concern in the module docs above — uses [`MAX_PLACEMENTS`] and is
/// unaffected by this figure.
pub const MAX_BLOCK_PLACEMENTS: usize = MAX_OBJECTS + MAX_DERIVED_RESULTS;

/// Maximum simultaneous physical allocations the block provider may have to
/// protect: live placements plus one retired generation plus one fallback
/// generation. Keeping this derivation next to [`MAX_BLOCK_PLACEMENTS`] makes
/// the allocator footprint change explicit when the object bound changes.
/// Live, retired, and fallback placements can each fill their bounded table.
/// One additional reservation may coexist while an update or copy-on-write is
/// in flight, before the replaced live extent moves to the retired set.
pub const MAX_TRACKED_BLOCK_EXTENTS: usize = MAX_BLOCK_PLACEMENTS * 3 + 1;

/// Fixed node arena for the block free-space index. With K occupied runs,
/// there can be at most K+1 intervening free runs; 2K+1 nodes therefore cover
/// every legal fragmentation pattern without growing after provider startup.
pub const MAX_BLOCK_EXTENT_NODES: usize = MAX_TRACKED_BLOCK_EXTENTS * 2 + 1;

/// Maximum chunks one object may hold, and therefore the largest file the
/// RAM provider can store: `MAX_CHUNKS_PER_OBJECT * RAM_CHUNK_BYTES` = 4 KiB.
/// A `resize` past it refuses with
/// [`FsError::OutOfStorage`](crate::FsError::OutOfStorage).
pub const MAX_CHUNKS_PER_OBJECT: usize = 16;

/// Largest byte count one read or write request may transfer.  Bounds the
/// engine's per-request work independently of the file size.
pub const MAX_TRANSFER: usize = 4096;

// ── Extraction snapshots and derived data ──────────────────────────────────

/// Immutable file versions concurrently lent to parser workers. A lend may
/// temporarily duplicate one file's placement, so this is intentionally
/// much smaller than the provider placement arena.
pub const MAX_LENT_VERSIONS: usize = 4;

/// Parser-neutral results retained by the filesystem. The hot representation
/// is the bounded 4 KiB #476 wire batch rather than the much larger decoded
/// schema object, so the arena can cover every file that can consume one of
/// the RAM profile's 64 chunks. Reprocessing replaces one parser lineage in
/// place; exhaustion remains a typed extraction refusal.
pub const MAX_DERIVED_RESULTS: usize = MAX_OBJECTS;

/// Executor turns one reclamation may spend waiting on contended slots before
/// it abandons the remaining subtree and counts the loss
/// (`Namespace::note_abandoned_reclaim`).
///
/// `reclaim_one` and `purge_object` are **synchronous**: every slot claim they
/// take is released before they return, so a `Busy` answer names a claim held
/// by a *different* in-flight operation, and one executor turn is what lets
/// that operation reach its next await. The budget is one turn per object the
/// namespace can hold — the worst case where every live object is contended
/// exactly once — so an expiry means a claim outlived the operation that took
/// it. That is a defect to count, not a hang to sit in: reclamation runs after
/// its client's namespace effect has already committed, so no one is waiting
/// on it, and an unbounded retry here wedges the whole filesystem where nothing
/// can report it.
pub const RECLAIM_CONTENTION_TURNS: u32 = MAX_OBJECTS as u32;

/// Shared storage for hot #476 wire batches. Results consume only the chunks
/// their encoded form needs instead of reserving 4 KiB in every result slot.
/// The 128 KiB arena holds 64 ordinary small-text batches or at least 32
/// maximum-sized worker reports, with typed exhaustion for larger mixes.
pub const DERIVED_WIRE_CHUNK_BYTES: usize = 1024;
pub const MAX_DERIVED_WIRE_CHUNKS: usize = MAX_DERIVED_RESULTS * 2;

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory object needs a child list, so the child-list arena must
    /// not claim to hold more directories than there are objects.
    #[test]
    fn directory_arena_fits_inside_object_arena() {
        assert!(MAX_DIRECTORIES <= MAX_OBJECTS);
    }

    /// The edge proof is a `heapless::Vec<_, MAX_PATH_DEPTH>`; a traversal
    /// that could outrun it would truncate the proof and silently weaken
    /// revalidation, so depth must bound the walk, not the storage.
    #[test]
    fn proof_depth_is_positive() {
        assert!(MAX_PATH_DEPTH > 0);
    }

    /// A one-chunk object must be representable, and the per-object chunk
    /// ceiling cannot exceed the pool it draws from.
    #[test]
    fn per_object_chunks_fit_the_pool() {
        assert!(MAX_CHUNKS_PER_OBJECT >= 1);
        assert!(MAX_CHUNKS_PER_OBJECT <= RAM_CHUNKS);
    }

    /// The RAM placement table is scarcer than objects — only a *written* file
    /// takes one, and directories never do. The DURABLE table is deliberately
    /// NOT: it holds file extents and derived search batches at once, so it
    /// must cover their worst-case sum, or one class starves the other (the
    /// `find --reindex` `OutOfStorage` wedge this decoupling fixed).
    #[test]
    fn placements_cover_their_providers_worst_case() {
        assert!(MAX_PLACEMENTS <= MAX_OBJECTS);
        assert!(MAX_BLOCK_PLACEMENTS >= MAX_OBJECTS + MAX_DERIVED_RESULTS);
    }

    /// A single transfer must be able to fill the largest object the provider
    /// can hold, or the largest file would be unwritable in one request and
    /// `MAX_TRANSFER` would be a silent second file-size limit.
    #[test]
    fn one_transfer_can_span_the_largest_object() {
        assert!(MAX_TRANSFER >= MAX_CHUNKS_PER_OBJECT * RAM_CHUNK_BYTES);
    }

    /// Names are non-empty and inline; a zero maximum would make every
    /// create refuse.
    #[test]
    fn name_bound_admits_a_name() {
        assert!(MAX_NAME_LEN >= 1);
    }

    /// A list page must be able to carry at least one entry, or listing
    /// could never advance its cursor.
    #[test]
    fn list_page_admits_an_entry() {
        assert!(MAX_LIST_PAGE >= 1);
    }

    #[test]
    fn derived_result_shard_mask_covers_the_result_arena() {
        assert!(
            MAX_DERIVED_WIRE_CHUNKS * DERIVED_WIRE_CHUNK_BYTES
                >= kore_svc_extract::WORKER_REPORT_BYTES
        );
    }
}
