//! Frozen MVP bounds, chosen to fit the filesystem and HostLink foundations.

/// Snapshot format version.
pub const FORMAT_VERSION: u16 = 1;
/// Import policy version.
pub const POLICY_VERSION: u16 = 1;
/// Filesystem-native component ceiling.
pub const MAX_COMPONENT_BYTES: usize = 32;
/// Filesystem-native traversal ceiling.
pub const MAX_PATH_DEPTH: usize = 8;
/// Canonical path storage, including seven separators.
pub const MAX_PATH_BYTES: usize = MAX_PATH_DEPTH * MAX_COMPONENT_BYTES + (MAX_PATH_DEPTH - 1);
/// Leaves headroom beneath the filesystem's 128-object arena for system roots.
pub const MAX_ENTRIES: usize = 48;
/// Leaves headroom beneath the filesystem's 16-directory arena.
pub const MAX_DIRECTORIES: usize = 8;
/// Native filesystem edge capacity for any one directory.
pub const MAX_CHILDREN_PER_DIRECTORY: usize = 32;
/// The founded HostLink profile carries at most 256 4-KiB wire chunks.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;
/// Aggregate staging reservation for one import.
pub const MAX_SNAPSHOT_BYTES: u64 = 32 * 1024 * 1024;
/// Bounded transfer fragment accepted by the transaction engine.
pub const MAX_CHUNK_BYTES: usize = 4096;
/// Export operations retained at once.
pub const MAX_CHANGES: usize = 64;
/// One independently verified host-backed range. This matches the filesystem
/// provider granule and stays comfortably below HostLink's bulk-object bound.
pub const RANGE_BYTES: usize = 4096;
/// A 1 MiB file has at most 256 range leaves and therefore an eight-sibling
/// proof in the fixed binary Merkle tree.
pub const MAX_RANGE_PROOF_DEPTH: usize = 8;
/// Verified host-backed ranges retained by the guest at once.
pub const MAX_CACHE_ENTRIES: usize = 16;
/// Total bytes retained by the verified range cache.
pub const MAX_CACHE_BYTES: usize = MAX_CACHE_ENTRIES * RANGE_BYTES;
/// Concurrent cache misses admitted by one live workspace.
pub const MAX_CONCURRENT_READS: usize = 4;
/// Files with kore-owned copy-on-write state at once.
pub const MAX_OVERLAY_FILES: usize = 4;
/// Aggregate copy-on-write bytes reserved before writeback.
pub const MAX_OVERLAY_BYTES: u64 = 4 * MAX_FILE_BYTES;
/// Host topology transactions that may be retained for replay/idempotence.
pub const MAX_DELTA_HISTORY: usize = 8;
/// Excluded subtrees in one standing automatic-writeback grant.
pub const MAX_GRANT_EXCLUSIONS: usize = 8;
/// Simultaneous host apply transactions.
pub const MAX_TRANSACTIONS: usize = 4;
/// Bounded committed/conflict/partial results retained across reconnects.
pub const MAX_TRANSACTION_RESULTS: usize = 8;
/// Retry ceiling for one unavailable or stale live operation.
pub const MAX_RETRIES: u8 = 3;

const _: () = {
    assert!(MAX_COMPONENT_BYTES > 0);
    assert!(MAX_PATH_DEPTH > 0);
    assert!(MAX_PATH_BYTES >= MAX_COMPONENT_BYTES);
    assert!(MAX_DIRECTORIES <= MAX_ENTRIES);
    assert!(MAX_FILE_BYTES <= u32::MAX as u64);
    assert!(MAX_SNAPSHOT_BYTES >= MAX_FILE_BYTES);
    assert!(MAX_CHUNK_BYTES > 0);
    assert!(RANGE_BYTES == MAX_CHUNK_BYTES);
    assert!(MAX_FILE_BYTES.div_ceil(RANGE_BYTES as u64) <= 1 << MAX_RANGE_PROOF_DEPTH);
    assert!(MAX_CACHE_BYTES == MAX_CACHE_ENTRIES * RANGE_BYTES);
    assert!(MAX_CONCURRENT_READS <= MAX_CACHE_ENTRIES);
    assert!(MAX_OVERLAY_BYTES >= MAX_FILE_BYTES);
    assert!(MAX_DELTA_HISTORY > 0);
    assert!(MAX_TRANSACTIONS > 0);
};
