//! Bounded live-workspace state owned only by the trusted guest importer.
//!
//! Agent-visible filesystem objects are keyed by kore identity. Host source
//! correlations stay in this table and are emitted only by `begin_read`, the
//! seam used by the importer to construct a HostLink request.

use heapless::Vec;

use crate::{
    Manifest, Policy, RangeProof,
    bounds::{
        MAX_CACHE_BYTES, MAX_CACHE_ENTRIES, MAX_CONCURRENT_READS, MAX_ENTRIES, MAX_OVERLAY_BYTES,
        MAX_OVERLAY_FILES, MAX_TRANSACTION_RESULTS, RANGE_BYTES,
    },
    merkle::verify_range,
    wire::{ContentDigest, SnapshotId, SourceCorrelation},
};

pub type KoreObjectIdentity = [u8; 32];
pub type TransactionId = [u8; 32];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheKey {
    pub snapshot: SnapshotId,
    pub source: SourceCorrelation,
    pub version: ContentDigest,
    pub range_root: ContentDigest,
    pub file_len: u32,
    pub range_index: u16,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    key: CacheKey,
    bytes: [u8; RANGE_BYTES],
    len: u16,
    used: u64,
    valid: bool,
    pinned: bool,
}

impl CacheEntry {
    const EMPTY: Self = Self {
        key: CacheKey {
            snapshot: [0; 32],
            source: [0; 16],
            version: [0; 32],
            range_root: [0; 32],
            file_len: 0,
            range_index: 0,
        },
        bytes: [0; RANGE_BYTES],
        len: 0,
        used: 0,
        valid: false,
        pinned: false,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheError {
    InvalidGeometry,
    Integrity,
    Full,
    ClockExhausted,
}

pub struct VerifiedRangeCache {
    entries: [CacheEntry; MAX_CACHE_ENTRIES],
    clock: u64,
    pinned_bytes: usize,
}

impl Default for VerifiedRangeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifiedRangeCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [CacheEntry::EMPTY; MAX_CACHE_ENTRIES],
            clock: 1,
            pinned_bytes: 0,
        }
    }

    pub fn insert(
        &mut self,
        key: CacheKey,
        proof: &RangeProof,
        bytes: &[u8],
        pin: bool,
    ) -> Result<(), CacheError> {
        if proof.file_len != key.file_len
            || proof.leaf_index != key.range_index
            || bytes.len() > RANGE_BYTES
        {
            return Err(CacheError::InvalidGeometry);
        }
        if !verify_range(key.range_root, proof, bytes) {
            return Err(CacheError::Integrity);
        }
        let now = self.tick()?;
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.valid && entry.key == key)
        {
            if entry.bytes[..usize::from(entry.len)] != *bytes {
                return Err(CacheError::Integrity);
            }
            if pin && !entry.pinned {
                self.pinned_bytes = self
                    .pinned_bytes
                    .checked_add(RANGE_BYTES)
                    .ok_or(CacheError::Full)?;
                if self.pinned_bytes > MAX_CACHE_BYTES {
                    self.pinned_bytes -= RANGE_BYTES;
                    return Err(CacheError::Full);
                }
                entry.pinned = true;
            }
            entry.used = now;
            return Ok(());
        }
        let slot = self
            .entries
            .iter()
            .position(|entry| !entry.valid)
            .or_else(|| {
                self.entries
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| !entry.pinned)
                    .min_by_key(|(_, entry)| entry.used)
                    .map(|(index, _)| index)
            })
            .ok_or(CacheError::Full)?;
        let entry = &mut self.entries[slot];
        entry.bytes.fill(0);
        entry.bytes[..bytes.len()].copy_from_slice(bytes);
        entry.key = key;
        entry.len = bytes.len() as u16;
        entry.used = now;
        entry.valid = true;
        entry.pinned = pin;
        if pin {
            self.pinned_bytes = self
                .pinned_bytes
                .checked_add(RANGE_BYTES)
                .filter(|bytes| *bytes <= MAX_CACHE_BYTES)
                .ok_or(CacheError::Full)?;
        }
        Ok(())
    }

    pub fn copy(&mut self, key: CacheKey, range_offset: usize, out: &mut [u8]) -> Option<usize> {
        let now = self.tick().ok()?;
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.valid && entry.key == key)?;
        let tail = entry.bytes.get(range_offset..usize::from(entry.len))?;
        let count = tail.len().min(out.len());
        out[..count].copy_from_slice(&tail[..count]);
        entry.used = now;
        Some(count)
    }

    pub fn revoke_snapshot(&mut self, snapshot: SnapshotId) {
        for entry in &mut self.entries {
            if entry.valid && entry.key.snapshot == snapshot {
                if entry.pinned {
                    self.pinned_bytes = self.pinned_bytes.saturating_sub(RANGE_BYTES);
                }
                entry.bytes.fill(0);
                *entry = CacheEntry::EMPTY;
            }
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.iter().filter(|entry| entry.valid).count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|entry| !entry.valid)
    }

    fn tick(&mut self) -> Result<u64, CacheError> {
        let now = self.clock;
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(CacheError::ClockExhausted)?;
        Ok(now)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LiveFile {
    pub kore_object: KoreObjectIdentity,
    snapshot: SnapshotId,
    source: SourceCorrelation,
    version: ContentDigest,
    range_root: ContentDigest,
    len: u32,
}

impl LiveFile {
    pub fn new(
        kore_object: KoreObjectIdentity,
        snapshot: SnapshotId,
        source: SourceCorrelation,
        version: ContentDigest,
        range_root: ContentDigest,
        len: u32,
    ) -> Option<Self> {
        (kore_object != [0; 32]
            && snapshot != [0; 32]
            && source != [0; 16]
            && version != [0; 32]
            && range_root != [0; 32]
            && u64::from(len) <= crate::bounds::MAX_FILE_BYTES)
            .then_some(Self {
                kore_object,
                snapshot,
                source,
                version,
                range_root,
                len,
            })
    }

    #[must_use]
    pub const fn len(&self) -> u32 {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn key(self, range_index: u16) -> CacheKey {
        CacheKey {
            snapshot: self.snapshot,
            source: self.source,
            version: self.version,
            range_root: self.range_root,
            file_len: self.len,
            range_index,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingRead {
    request: u64,
    key: CacheKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OverlayReservation {
    object: KoreObjectIdentity,
    bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionOutcome {
    Applied { snapshot: SnapshotId },
    Conflict,
    Partial { applied: u16 },
    Refused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransactionRecord {
    pub id: TransactionId,
    pub outcome: TransactionOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveError {
    InvalidManifest,
    InvalidBinding,
    DuplicateObject,
    UnknownObject,
    Revoked,
    Range,
    Backpressure,
    DuplicateRequest,
    RequestUnknown,
    Cache(CacheError),
    OverlayFull,
    ArithmeticOverflow,
    TransactionConflict,
    ResultFull,
}

pub struct LiveWorkspace {
    snapshot: SnapshotId,
    files: Vec<LiveFile, MAX_ENTRIES>,
    cache: VerifiedRangeCache,
    pending: Vec<PendingRead, MAX_CONCURRENT_READS>,
    overlays: Vec<OverlayReservation, MAX_OVERLAY_FILES>,
    overlay_bytes: u64,
    results: Vec<TransactionRecord, MAX_TRANSACTION_RESULTS>,
    revoked: bool,
}

impl LiveWorkspace {
    pub fn attach(
        manifest: &Manifest,
        policy: Policy,
        bindings: &[LiveFile],
    ) -> Result<Self, LiveError> {
        manifest
            .verify(policy)
            .map_err(|_| LiveError::InvalidManifest)?;
        let file_entries = manifest
            .entries()
            .iter()
            .filter(|entry| entry.kind == crate::EntryKind::File)
            .count();
        if bindings.len() != file_entries {
            return Err(LiveError::InvalidBinding);
        }
        let mut files = Vec::new();
        for binding in bindings {
            if binding.snapshot != manifest.snapshot_id()
                || binding.kore_object == [0; 32]
                || files
                    .iter()
                    .any(|file: &LiveFile| file.kore_object == binding.kore_object)
            {
                return Err(LiveError::DuplicateObject);
            }
            let matching = manifest.entries().iter().any(|entry| {
                entry.kind == crate::EntryKind::File
                    && entry.source == binding.source
                    && entry.digest == binding.version
                    && entry.range_root == binding.range_root
                    && entry.content_len == binding.len
            });
            if !matching {
                return Err(LiveError::InvalidBinding);
            }
            files
                .push(*binding)
                .map_err(|_| LiveError::InvalidBinding)?;
        }
        Ok(Self {
            snapshot: manifest.snapshot_id(),
            files,
            cache: VerifiedRangeCache::new(),
            pending: Vec::new(),
            overlays: Vec::new(),
            overlay_bytes: 0,
            results: Vec::new(),
            revoked: false,
        })
    }

    #[must_use]
    pub const fn snapshot(&self) -> SnapshotId {
        self.snapshot
    }

    pub fn begin_read(
        &mut self,
        request: u64,
        object: KoreObjectIdentity,
        offset: u32,
        len: u16,
    ) -> Result<CacheKey, LiveError> {
        if self.revoked {
            return Err(LiveError::Revoked);
        }
        if request == 0
            || self
                .pending
                .iter()
                .any(|pending| pending.request == request)
        {
            return Err(LiveError::DuplicateRequest);
        }
        let file = *self
            .files
            .iter()
            .find(|file| file.kore_object == object)
            .ok_or(LiveError::UnknownObject)?;
        let end = offset.checked_add(u32::from(len)).ok_or(LiveError::Range)?;
        if len == 0 || usize::from(len) > RANGE_BYTES || end > file.len {
            return Err(LiveError::Range);
        }
        let range_index = offset / RANGE_BYTES as u32;
        let range_start = range_index * RANGE_BYTES as u32;
        if end > range_start.saturating_add(RANGE_BYTES as u32) {
            return Err(LiveError::Range);
        }
        let key = file.key(range_index as u16);
        self.pending
            .push(PendingRead { request, key })
            .map_err(|_| LiveError::Backpressure)?;
        Ok(key)
    }

    pub fn complete_read(
        &mut self,
        request: u64,
        proof: &RangeProof,
        bytes: &[u8],
    ) -> Result<CacheKey, LiveError> {
        let index = self
            .pending
            .iter()
            .position(|pending| pending.request == request)
            .ok_or(LiveError::RequestUnknown)?;
        let pending = self.pending[index];
        self.cache
            .insert(pending.key, proof, bytes, false)
            .map_err(LiveError::Cache)?;
        self.pending.swap_remove(index);
        Ok(pending.key)
    }

    pub fn cached_read(
        &mut self,
        object: KoreObjectIdentity,
        offset: u32,
        out: &mut [u8],
    ) -> Result<Option<usize>, LiveError> {
        if self.revoked {
            return Err(LiveError::Revoked);
        }
        let file = *self
            .files
            .iter()
            .find(|file| file.kore_object == object)
            .ok_or(LiveError::UnknownObject)?;
        if offset >= file.len || out.is_empty() {
            return Ok(Some(0));
        }
        let range_index = offset / RANGE_BYTES as u32;
        let range_offset = (offset % RANGE_BYTES as u32) as usize;
        Ok(self
            .cache
            .copy(file.key(range_index as u16), range_offset, out))
    }

    pub fn reserve_overlay(
        &mut self,
        object: KoreObjectIdentity,
        bytes: u32,
    ) -> Result<(), LiveError> {
        if self.revoked {
            return Err(LiveError::Revoked);
        }
        if self.files.iter().all(|file| file.kore_object != object) {
            return Err(LiveError::UnknownObject);
        }
        if let Some(existing) = self
            .overlays
            .iter_mut()
            .find(|overlay| overlay.object == object)
        {
            let next = self
                .overlay_bytes
                .checked_sub(u64::from(existing.bytes))
                .and_then(|value| value.checked_add(u64::from(bytes)))
                .ok_or(LiveError::ArithmeticOverflow)?;
            if next > MAX_OVERLAY_BYTES {
                return Err(LiveError::OverlayFull);
            }
            existing.bytes = bytes;
            self.overlay_bytes = next;
            return Ok(());
        }
        let next = self
            .overlay_bytes
            .checked_add(u64::from(bytes))
            .ok_or(LiveError::ArithmeticOverflow)?;
        if next > MAX_OVERLAY_BYTES {
            return Err(LiveError::OverlayFull);
        }
        self.overlays
            .push(OverlayReservation { object, bytes })
            .map_err(|_| LiveError::OverlayFull)?;
        self.overlay_bytes = next;
        Ok(())
    }

    pub fn record_transaction(
        &mut self,
        record: TransactionRecord,
    ) -> Result<TransactionOutcome, LiveError> {
        if record.id == [0; 32] {
            return Err(LiveError::TransactionConflict);
        }
        if let Some(existing) = self.results.iter().find(|entry| entry.id == record.id) {
            return if existing.outcome == record.outcome {
                Ok(existing.outcome)
            } else {
                Err(LiveError::TransactionConflict)
            };
        }
        self.results
            .push(record)
            .map_err(|_| LiveError::ResultFull)?;
        Ok(record.outcome)
    }

    pub fn revoke(&mut self) {
        if self.revoked {
            return;
        }
        self.revoked = true;
        self.cache.revoke_snapshot(self.snapshot);
        self.pending.clear();
        self.overlays.clear();
        self.overlay_bytes = 0;
    }

    #[must_use]
    pub fn cache_entries(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalPath, Entry, ManifestBuilder, build_range_proof, range_root};

    fn fixture() -> (Manifest, LiveFile, std::vec::Vec<u8>) {
        let bytes = (0..(RANGE_BYTES + 17))
            .map(|index| index as u8)
            .collect::<std::vec::Vec<_>>();
        let digest = *blake3::hash(&bytes).as_bytes();
        let range_root = range_root(&bytes).unwrap();
        let source = [7; 16];
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(
                Entry::file_with_range_root(
                    CanonicalPath::new(b"file").unwrap(),
                    false,
                    bytes.len() as u32,
                    digest,
                    range_root,
                    source,
                )
                .unwrap(),
            )
            .unwrap();
        let manifest = builder.finish().unwrap();
        let file = LiveFile {
            kore_object: [9; 32],
            snapshot: manifest.snapshot_id(),
            source,
            version: digest,
            range_root,
            len: bytes.len() as u32,
        };
        (manifest, file, bytes)
    }

    #[test]
    fn first_read_misses_then_verified_range_hits_without_source_exposure() {
        let (manifest, file, bytes) = fixture();
        let mut live = LiveWorkspace::attach(&manifest, Policy::default(), &[file]).unwrap();
        let mut out = [0; 16];
        assert_eq!(live.cached_read(file.kore_object, 8, &mut out), Ok(None));
        live.begin_read(1, file.kore_object, 8, 16).unwrap();
        let proof = build_range_proof(&bytes, 0).unwrap();
        live.complete_read(1, &proof, &bytes[..RANGE_BYTES])
            .unwrap();
        assert_eq!(
            live.cached_read(file.kore_object, 8, &mut out),
            Ok(Some(16))
        );
        assert_eq!(&out, &bytes[8..24]);
        assert_eq!(live.cache_entries(), 1);
    }

    #[test]
    fn bad_proof_publishes_nothing_and_revocation_blocks_cache_and_fetch() {
        let (manifest, file, bytes) = fixture();
        let mut live = LiveWorkspace::attach(&manifest, Policy::default(), &[file]).unwrap();
        live.begin_read(1, file.kore_object, 0, 1).unwrap();
        let proof = build_range_proof(&bytes, 0).unwrap();
        let mut corrupt = bytes[..RANGE_BYTES].to_vec();
        corrupt[0] ^= 1;
        assert_eq!(
            live.complete_read(1, &proof, &corrupt),
            Err(LiveError::Cache(CacheError::Integrity))
        );
        assert_eq!(live.cache_entries(), 0);
        live.revoke();
        assert_eq!(
            live.begin_read(2, file.kore_object, 0, 1),
            Err(LiveError::Revoked)
        );
        assert_eq!(
            live.cached_read(file.kore_object, 0, &mut [0; 1]),
            Err(LiveError::Revoked)
        );
    }

    #[test]
    fn read_and_overlay_arithmetic_is_bounded() {
        let (manifest, file, _) = fixture();
        let mut live = LiveWorkspace::attach(&manifest, Policy::default(), &[file]).unwrap();
        assert_eq!(
            live.begin_read(1, file.kore_object, u32::MAX, 2),
            Err(LiveError::Range)
        );
        assert_eq!(live.reserve_overlay(file.kore_object, 1024), Ok(()));
        assert_eq!(live.reserve_overlay(file.kore_object, 2048), Ok(()));
        assert_eq!(live.overlay_bytes, 2048);
    }

    #[test]
    fn duplicate_transaction_is_idempotent_but_equivocation_is_refused() {
        let (manifest, file, _) = fixture();
        let mut live = LiveWorkspace::attach(&manifest, Policy::default(), &[file]).unwrap();
        let record = TransactionRecord {
            id: [3; 32],
            outcome: TransactionOutcome::Conflict,
        };
        assert_eq!(
            live.record_transaction(record),
            Ok(TransactionOutcome::Conflict)
        );
        assert_eq!(
            live.record_transaction(record),
            Ok(TransactionOutcome::Conflict)
        );
        assert_eq!(
            live.record_transaction(TransactionRecord {
                outcome: TransactionOutcome::Refused,
                ..record
            }),
            Err(LiveError::TransactionConflict)
        );
    }
}
