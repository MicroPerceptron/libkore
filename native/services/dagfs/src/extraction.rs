//! Immutable file-version lending and parser-result association.
//!
//! This is the filesystem-specific half of extraction. The generic broker
//! and workers never receive namespace handles, paths, or containment-graph
//! authority. They receive one owner-bound [`VersionLease`] whose only
//! operation is reading a frozen byte range. A committed write therefore
//! cannot change the bytes beneath an in-flight parser.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use heapless::Vec;
use kore_svc_extract::{
    WORKER_REPORT_BYTES, decode_result, encode_result,
    registry::{ParserId, Registry, ValidationError},
    schema::{FactValue, InputIdentity, NormalizedResult},
};

use crate::{
    authority::{Handle, SessionId},
    bounds::{
        DERIVED_WIRE_CHUNK_BYTES, MAX_DERIVED_RESULTS, MAX_DERIVED_WIRE_CHUNKS, MAX_LENT_VERSIONS,
        MAX_OBJECTS,
    },
    error::{FsError, FsResult, ProviderError},
    object::{ObjectId, ObjectIdExt, Revision},
    search::{IndexMatch, MAX_SEARCH_SNIPPET, QueryIndex, SearchContent, SearchPlane, SearchScore},
    slot::{
        AtomicBitVec, AtomicSlotVec, Keyed, SlotAccess, SlotKey, SlotRead, SlotTable, SlotVecWrite,
        SlotWrite,
    },
};
use core::alloc::Allocator;

/// Identity stamped by the FS serving host for one extraction connection.
///
/// It is deliberately crate-private to construct: an untrusted request can
/// carry a lease number, but cannot choose the owner against which the
/// service validates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExtractionOwner(u64);

impl ExtractionOwner {
    pub(crate) const fn new(stamped_peer: u64) -> Self {
        Self(stamped_peer)
    }
}
pub use kore_dagfs::VersionLease;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionError {
    Filesystem(FsError),
    Provider(ProviderError),
    NoSnapshotSlots,
    NoResultSlots,
    InvalidLease,
    ObsoleteVersion,
    InvalidResult(ValidationError),
    Range,
}

impl From<FsError> for ExtractionError {
    fn from(value: FsError) -> Self {
        Self::Filesystem(value)
    }
}

impl From<ProviderError> for ExtractionError {
    fn from(value: ProviderError) -> Self {
        Self::Provider(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivedResultInfo {
    pub input: InputIdentity,
    pub parser: ParserId,
    pub parser_version: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct Snapshot<K> {
    pub(crate) owner: ExtractionOwner,
    pub(crate) session: SessionId,
    pub(crate) handle: Handle,
    pub(crate) object: ObjectId,
    pub(crate) identity: InputIdentity,
    /// The placement the lease reads: the object's own live placement,
    /// **borrowed**, never a copy.  The engine keeps it frozen by moving
    /// the object to a fresh placement before any mutation while a lease
    /// borrows it ([`Engine::ensure_capacity`](crate::engine::Engine)),
    /// and a lease that outlives its object becomes the placement's
    /// owner; the last borrower releases it.
    pub(crate) storage: Option<K>,
}

struct DerivedResult<K> {
    object: ObjectId,
    input: InputIdentity,
    parser: ParserId,
    parser_version: u32,
    /// Exact bounded #476 wire batch. Keeping the normalized object decoded in
    /// every arena cell multiplies its largest enum/collection variants and
    /// exhausts the founded stack after only a handful of files.
    encoded: Vec<SlotKey, 4>,
    hot_len: u16,
    /// Provider-backed copy of the validated wire batch. Present only for a
    /// provider whose placements survive restart; the in-memory result is the
    /// rebuildable hot representation used by synchronous queries.
    storage: Option<K>,
    encoded_len: u32,
    encoded_digest: [u8; 32],
}

struct WireChunk {
    len: u16,
    bytes: [u8; DERIVED_WIRE_CHUNK_BYTES],
}

/// Durable placement metadata exported to the persistence layer. `K` remains
/// the provider's non-authority placement identifier.
#[derive(Clone, Copy)]
pub(crate) struct DurableDerived<K> {
    pub(crate) object: ObjectId,
    pub(crate) input: InputIdentity,
    pub(crate) key: K,
    pub(crate) len: u32,
    pub(crate) digest: [u8; 32],
}

pub(crate) struct ExtractionState<K> {
    snapshots: SlotTable<Snapshot<K>, MAX_LENT_VERSIONS>,
    /// Derived-index results, sized at construction to the namespace bound
    /// ([`MAX_DERIVED_RESULTS`] = [`MAX_OBJECTS`]) and drawn from the service
    /// heap, so a fully-indexed namespace fits — the fixed 64-slot arena sat
    /// below the object ceiling and refused mid-reindex on a full volume.
    results: AtomicSlotVec<DerivedResult<K>, dyn Allocator>,
    wire_chunks: AtomicSlotVec<WireChunk, dyn Allocator>,
    /// Result-arena cells owned by each object slot, as a
    /// `MAX_OBJECTS` × [`MAX_DERIVED_RESULTS`] occupancy bitmap (row = object
    /// slot). Query lookup touches only the visible candidate's row; hidden
    /// result occupancy and contention cannot change another view's work or
    /// refusal behavior. Replaces a per-object `AtomicU64` whose 64-bit width
    /// capped the result arena below the object count.
    result_slots: AtomicBitVec<dyn Allocator>,
    /// Rebuildable per-object index generations. A hidden object's publication
    /// cannot perturb another authority partition's observation identity.
    result_generations: [AtomicU64; MAX_OBJECTS],
    /// Result mutation is scoped to one object-arena cell. A replacement
    /// generation may briefly contend with reclamation of its predecessor,
    /// but every record comparison still uses the full generational
    /// [`ObjectId`], so neither generation can select or purge the other.
    result_claims: [AtomicBool; MAX_OBJECTS],
}

struct ResultClaim<'a>(&'a AtomicBool);

impl Drop for ResultClaim<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<K: Copy> ExtractionState<K> {
    pub(crate) fn new() -> Self {
        // The derived index scales with the namespace, so its arenas are
        // heap-backed rather than inline `.bss` — this both lifts the old
        // 64-slot ceiling to the object count and keeps the by-value engine
        // small enough to move on the founding stack. The allocations are
        // bootstrap-scale and fixed (not volume-scaled), so an allocator that
        // cannot satisfy them at mount is an undersized service heap — a build
        // configuration error, the same class as the compile-time size guard —
        // not a runtime volume condition. The genuinely volume-scaled durable
        // *data* path remains fallible via the provider.
        let alloc = crate::service_alloc();
        Self {
            snapshots: SlotTable::new(),
            results: AtomicSlotVec::try_new_in(MAX_DERIVED_RESULTS, alloc).expect(
                "fs derived-result index arena (bootstrap-scale) — service heap undersized",
            ),
            wire_chunks: AtomicSlotVec::try_new_in(MAX_DERIVED_WIRE_CHUNKS, alloc)
                .expect("fs derived wire-chunk arena (bootstrap-scale) — service heap undersized"),
            result_slots: AtomicBitVec::try_new_in(MAX_OBJECTS * MAX_DERIVED_RESULTS, alloc)
                .expect("fs result-occupancy bitmap (bootstrap-scale) — service heap undersized"),
            result_generations: [const { AtomicU64::new(1) }; MAX_OBJECTS],
            result_claims: [const { AtomicBool::new(false) }; MAX_OBJECTS],
        }
    }

    pub(crate) fn can_lend(&self) -> bool {
        !self.snapshots.is_full()
    }

    /// How many live leases borrow placement `key`, not counting `except`.
    /// A key with a borrower must not be mutated in place or released.
    pub(crate) fn storage_borrowers(
        &self,
        key: K,
        except: Option<VersionLease>,
    ) -> Result<usize, FsError>
    where
        K: PartialEq + Copy,
    {
        let mut count = 0;
        for index in 0..self.snapshots.capacity() {
            let Some(id) = self.snapshots.live_id_at(index) else {
                continue;
            };
            if except == Some(VersionLease::from_key(id)) {
                continue;
            }
            let snapshot = self.snapshots.try_read(id).map_err(|_| FsError::Busy)?;
            if snapshot.storage == Some(key) {
                count += 1;
            }
        }
        Ok(count)
    }

    pub(crate) fn install_snapshot(
        &self,
        snapshot: Snapshot<K>,
    ) -> Result<VersionLease, Snapshot<K>> {
        self.snapshots.insert(snapshot).map(VersionLease::from_key)
    }

    pub(crate) fn snapshot(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
    ) -> Result<SlotRead<'_, Snapshot<K>, MAX_LENT_VERSIONS>, ExtractionError> {
        let snapshot = self
            .snapshots
            .try_read(lease.key())
            .map_err(|error| match error {
                SlotAccess::Stale => ExtractionError::InvalidLease,
                SlotAccess::Busy => ExtractionError::Filesystem(FsError::Busy),
            })?;
        if snapshot.owner != owner {
            return Err(ExtractionError::InvalidLease);
        }
        Ok(snapshot)
    }

    pub(crate) fn claim_snapshot(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
    ) -> Result<SlotWrite<'_, Snapshot<K>, MAX_LENT_VERSIONS>, ExtractionError> {
        let snapshot = self
            .snapshots
            .try_write(lease.key())
            .map_err(|error| match error {
                SlotAccess::Stale => ExtractionError::InvalidLease,
                SlotAccess::Busy => ExtractionError::Filesystem(FsError::Busy),
            })?;
        if snapshot.owner != owner {
            return Err(ExtractionError::InvalidLease);
        }
        Ok(snapshot)
    }

    pub(crate) fn remove_claimed_snapshot(
        &self,
        snapshot: SlotWrite<'_, Snapshot<K>, MAX_LENT_VERSIONS>,
    ) -> Result<Snapshot<K>, ExtractionError> {
        snapshot.remove().map_err(|error| match error {
            SlotAccess::Stale => ExtractionError::InvalidLease,
            SlotAccess::Busy => ExtractionError::Filesystem(FsError::Busy),
        })
    }

    fn claim_results(&self, object: ObjectId) -> Result<ResultClaim<'_>, FsError> {
        let index = object.key().index() as usize;
        let claim = self.result_claims.get(index).ok_or(FsError::AccessLost)?;
        claim
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| FsError::Busy)?;
        Ok(ResultClaim(claim))
    }

    /// Base bit index of one object's occupancy row in `result_slots`. The map
    /// is a `MAX_OBJECTS` × [`MAX_DERIVED_RESULTS`] bit matrix; a result at
    /// arena slot `s` owned by this object sets bit `base + s`. Replaces the
    /// per-object `AtomicU64` word — the width is now the result arena, not 64.
    fn result_row_base(&self, object: ObjectId) -> Result<usize, FsError> {
        let object = object.key().index() as usize;
        if object >= MAX_OBJECTS {
            return Err(FsError::AccessLost);
        }
        Ok(object * MAX_DERIVED_RESULTS)
    }

    fn result_generation(&self, object: ObjectId) -> Result<&AtomicU64, FsError> {
        self.result_generations
            .get(object.key().index() as usize)
            .ok_or(FsError::AccessLost)
    }

    fn object_result_keys(
        &self,
        object: ObjectId,
    ) -> Result<Vec<SlotKey, MAX_DERIVED_RESULTS>, FsError> {
        let base = self.result_row_base(object)?;
        let mut keys = Vec::new();
        for slot in 0..self.results.capacity() {
            if !self.result_slots.contains(base + slot) {
                continue;
            }
            keys.push(self.results.live_id_at(slot).ok_or(FsError::Busy)?)
                .map_err(|_| FsError::OutOfEdges)?;
        }
        Ok(keys)
    }

    fn matching_result(
        &self,
        object: ObjectId,
        parser: ParserId,
    ) -> Result<Option<SlotKey>, FsError> {
        let mut found = None;
        for key in self.object_result_keys(object)? {
            let stored = self.results.try_read(key).map_err(|error| match error {
                SlotAccess::Stale | SlotAccess::Busy => FsError::Busy,
            })?;
            if stored.object == object && stored.parser == parser {
                if found.replace(key).is_some() {
                    // The per-object claim makes a duplicate unreachable in
                    // the current implementation. Treat a legacy/corrupt
                    // duplicate as contention rather than selecting one.
                    return Err(FsError::Busy);
                }
            }
        }
        Ok(found)
    }

    fn pack_result(
        &self,
        result: &NormalizedResult,
    ) -> Result<(Vec<SlotKey, 4>, u16), ExtractionError> {
        let mut wire = [0u8; WORKER_REPORT_BYTES];
        let used = encode_result(result, &mut wire).map_err(|_| ExtractionError::Range)?;
        let hot_len = u16::try_from(used).map_err(|_| ExtractionError::Range)?;
        let mut encoded = Vec::new();
        for part in wire[..used].chunks(DERIVED_WIRE_CHUNK_BYTES) {
            let mut bytes = [0u8; DERIVED_WIRE_CHUNK_BYTES];
            bytes[..part.len()].copy_from_slice(part);
            let chunk = WireChunk {
                len: part.len() as u16,
                bytes,
            };
            let key = match self.wire_chunks.insert(chunk) {
                Ok(key) => key,
                Err(_) => {
                    let _ = self.release_packed(&encoded);
                    return Err(ExtractionError::NoResultSlots);
                }
            };
            if encoded.push(key).is_err() {
                let _ = self.wire_chunks.remove(key);
                let _ = self.release_packed(&encoded);
                return Err(ExtractionError::Range);
            }
        }
        Ok((encoded, hot_len))
    }

    fn release_packed(&self, encoded: &[SlotKey]) -> FsResult<()> {
        for key in encoded {
            self.wire_chunks.remove(*key).ok_or(FsError::Busy)?;
        }
        Ok(())
    }

    fn decode_packed(&self, encoded: &[SlotKey], hot_len: u16) -> FsResult<NormalizedResult> {
        let wanted = usize::from(hot_len);
        if wanted == 0 || wanted > WORKER_REPORT_BYTES {
            return Err(FsError::Conflict);
        }
        let mut wire = [0u8; WORKER_REPORT_BYTES];
        let mut at = 0usize;
        for key in encoded {
            let chunk = self.wire_chunks.try_read(*key).map_err(|_| FsError::Busy)?;
            let len = usize::from(chunk.len);
            let end = at.checked_add(len).ok_or(FsError::Conflict)?;
            if len == 0 || len > DERIVED_WIRE_CHUNK_BYTES || end > wanted {
                return Err(FsError::Conflict);
            }
            wire[at..end].copy_from_slice(&chunk.bytes[..len]);
            at = end;
        }
        if at != wanted {
            return Err(FsError::Conflict);
        }
        decode_result(&wire[..wanted]).map_err(|_| FsError::Conflict)
    }

    pub(crate) fn commit_result(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
        current_revision: Option<Revision>,
        result: NormalizedResult,
        output_window_bytes: u64,
    ) -> Result<DerivedResultInfo, ExtractionError> {
        let snapshot = self.snapshot(owner, lease)?;
        if current_revision != Some(snapshot.identity.version) {
            return Err(ExtractionError::ObsoleteVersion);
        }
        Registry::validate(&result, snapshot.identity, output_window_bytes)
            .map_err(ExtractionError::InvalidResult)?;
        let (encoded, hot_len) = self.pack_result(&result)?;

        let info = DerivedResultInfo {
            input: snapshot.identity,
            parser: result.provenance.parser,
            parser_version: result.provenance.parser_version,
        };

        let _claim = match self.claim_results(snapshot.object) {
            Ok(claim) => claim,
            Err(error) => {
                let _ = self.release_packed(&encoded);
                return Err(ExtractionError::Filesystem(error));
            }
        };

        // One selected result per object and parser lineage. A new parser
        // version replaces that lineage, while results from other parsers
        // remain independently selectable.
        let mut record = DerivedResult {
            object: snapshot.object,
            input: result.provenance.input,
            parser: result.provenance.parser,
            parser_version: result.provenance.parser_version,
            encoded,
            hot_len,
            storage: None,
            encoded_len: 0,
            encoded_digest: [0; 32],
        };
        let replaced = match self.matching_result(snapshot.object, record.parser) {
            Ok(replaced) => replaced,
            Err(error) => {
                let _ = self.release_packed(&record.encoded);
                return Err(ExtractionError::Filesystem(error));
            }
        };
        if let Some(key) = replaced {
            let mut stored = match self.results.try_write(key) {
                Ok(stored) => stored,
                Err(SlotAccess::Stale | SlotAccess::Busy) => {
                    let _ = self.release_packed(&record.encoded);
                    return Err(ExtractionError::Filesystem(FsError::Busy));
                }
            };
            // Keep custody of the previous durable placement until the new
            // encoded batch is attached. The engine releases it only after
            // that publication succeeds.
            record.storage = stored.storage;
            record.encoded_len = stored.encoded_len;
            let predecessor = core::mem::replace(&mut *stored, record);
            drop(stored);
            self.release_packed(&predecessor.encoded)
                .map_err(ExtractionError::Filesystem)?;
        } else {
            let key = self.results.insert(record).map_err(|record| {
                let _ = self.release_packed(&record.encoded);
                ExtractionError::NoResultSlots
            })?;
            let base = self
                .result_row_base(snapshot.object)
                .map_err(ExtractionError::Filesystem)?;
            self.result_slots.set(base + key.index() as usize);
        }
        self.result_generation(snapshot.object)
            .map_err(ExtractionError::Filesystem)?
            .fetch_add(1, Ordering::AcqRel);
        Ok(info)
    }

    /// Attach the durable wire placement after the ordinary validated commit.
    /// This is a separate, object-scoped publication step so RAM users retain
    /// the exact same path and a failed durable write cannot publish metadata
    /// that names missing bytes.
    pub(crate) fn attach_storage(
        &self,
        object: ObjectId,
        parser: ParserId,
        storage: K,
        len: u32,
        digest: [u8; 32],
    ) -> Result<Option<K>, FsError> {
        let _claim = self.claim_results(object)?;
        let key = self
            .matching_result(object, parser)?
            .ok_or(FsError::IndexIncomplete)?;
        let mut stored = self.results.try_write(key).map_err(|_| FsError::Busy)?;
        let previous = stored.storage.replace(storage);
        stored.encoded_len = len;
        stored.encoded_digest = digest;
        Ok(previous)
    }

    /// Restore a checksummed durable batch after its provider extent has been
    /// adopted and decoded. Recovery never fabricates a lease; it revalidates
    /// provenance before calling this method and installs only derived state.
    pub(crate) fn restore_result(
        &self,
        object: ObjectId,
        result: NormalizedResult,
        storage: K,
        len: u32,
        digest: [u8; 32],
    ) -> FsResult<()> {
        let _claim = self.claim_results(object)?;
        if self
            .matching_result(object, result.provenance.parser)?
            .is_some()
        {
            return Err(FsError::Conflict);
        }
        let (encoded, hot_len) = self
            .pack_result(&result)
            .map_err(|_| FsError::OutOfStorage)?;
        let key = self
            .results
            .insert(DerivedResult {
                object,
                input: result.provenance.input,
                parser: result.provenance.parser,
                parser_version: result.provenance.parser_version,
                encoded,
                hot_len,
                storage: Some(storage),
                encoded_len: len,
                encoded_digest: digest,
            })
            .map_err(|record| {
                let _ = self.release_packed(&record.encoded);
                FsError::OutOfStorage
            })?;
        let base = self.result_row_base(object)?;
        self.result_slots.set(base + key.index() as usize);
        self.result_generation(object)?
            .fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    pub(crate) fn durable_results<const N: usize>(&self) -> FsResult<Vec<DurableDerived<K>, N>> {
        let mut out = Vec::new();
        for index in 0..self.results.capacity() {
            let Some(key) = self.results.live_id_at(index) else {
                continue;
            };
            let stored = self.results.try_read(key).map_err(|_| FsError::Busy)?;
            let Some(storage) = stored.storage else {
                continue;
            };
            out.push(DurableDerived {
                object: stored.object,
                input: stored.input,
                key: storage,
                len: stored.encoded_len,
                digest: stored.encoded_digest,
            })
            .map_err(|_| FsError::OutOfStorage)?;
        }
        Ok(out)
    }

    pub(crate) fn result(
        &self,
        object: ObjectId,
        current_revision: Revision,
        parser: ParserId,
    ) -> Result<Option<NormalizedResult>, FsError> {
        let Some(key) = self.matching_result(object, parser)? else {
            return Ok(None);
        };
        let stored = self.results.try_read(key).map_err(|error| match error {
            SlotAccess::Stale | SlotAccess::Busy => FsError::Busy,
        })?;
        if stored.input.version != current_revision {
            return Ok(None);
        }
        self.decode_packed(&stored.encoded, stored.hot_len)
            .map(Some)
    }

    /// Visit parser-neutral results bound to exactly the candidate's current
    /// revision. Concurrent replacement is reported as bounded backpressure;
    /// search never observes a torn derived record.
    pub(crate) fn visit_current(
        &self,
        object: ObjectId,
        current_revision: Revision,
        mut visit: impl FnMut(&NormalizedResult),
    ) -> Result<(), FsError> {
        for key in self.object_result_keys(object)? {
            let stored = self.results.try_read(key).map_err(|_| FsError::Busy)?;
            if stored.object != object {
                return Err(FsError::Busy);
            }
            if stored.input.version == current_revision {
                let result = self.decode_packed(&stored.encoded, stored.hot_len)?;
                if result.provenance.input != stored.input
                    || result.provenance.parser != stored.parser
                    || result.provenance.parser_version != stored.parser_version
                {
                    return Err(FsError::Conflict);
                }
                visit(&result);
            }
        }
        Ok(())
    }

    pub(crate) fn purge_object(
        &self,
        object: ObjectId,
    ) -> Result<Vec<K, MAX_DERIVED_RESULTS>, FsError> {
        let _claim = self.claim_results(object)?;
        let matches = self.object_result_keys(object)?;
        for key in &matches {
            let stored = self.results.try_read(*key).map_err(|error| match error {
                SlotAccess::Stale | SlotAccess::Busy => FsError::Busy,
            })?;
            if stored.object != object {
                return Err(FsError::Busy);
            }
        }

        // Claim the complete footprint before removing the first record.
        let mut claimed =
            Vec::<SlotVecWrite<'_, DerivedResult<K>, dyn Allocator>, MAX_DERIVED_RESULTS>::new();
        for key in matches {
            claimed
                .push(self.results.try_write(key).map_err(|error| match error {
                    SlotAccess::Stale | SlotAccess::Busy => FsError::Busy,
                })?)
                .map_err(|_| FsError::OutOfEdges)?;
        }
        let mut storage = Vec::new();
        for result in claimed {
            if let Some(placement) = result.storage {
                storage.push(placement).map_err(|_| FsError::OutOfStorage)?;
            }
            let removed = result.remove().map_err(|_| FsError::Busy)?;
            self.release_packed(&removed.encoded)?;
        }
        let base = self.result_row_base(object)?;
        for slot in 0..self.results.capacity() {
            self.result_slots.clear(base + slot);
        }
        // Reclamation is the only caller. Reset rather than carry a retired
        // object's update history into a future occupant of the same slot.
        self.result_generation(object)?.store(1, Ordering::Release);
        Ok(storage)
    }

    #[cfg(test)]
    pub(crate) fn result_count(&self) -> usize {
        self.results.len()
    }
}

impl<K: Copy> Default for ExtractionState<K> {
    fn default() -> Self {
        Self::new()
    }
}

fn contains_folded(haystack: &[u8], needle: &[u8]) -> bool {
    needle.is_empty()
        || haystack.windows(needle.len()).any(|candidate| {
            candidate
                .iter()
                .zip(needle)
                .all(|(left, right)| left.eq_ignore_ascii_case(right))
        })
}

fn snippet(bytes: &[u8]) -> heapless::String<MAX_SEARCH_SNIPPET> {
    let mut out = heapless::String::new();
    let Ok(text) = core::str::from_utf8(bytes) else {
        return out;
    };
    for character in text.chars() {
        if out.push(character).is_err() {
            break;
        }
    }
    out
}

const fn token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || !byte.is_ascii()
}

fn for_each_token(bytes: &[u8], mut visit: impl FnMut(&[u8])) {
    let mut at = 0;
    while at < bytes.len() {
        while at < bytes.len() && !token_byte(bytes[at]) {
            at += 1;
        }
        let start = at;
        while at < bytes.len() && token_byte(bytes[at]) {
            at += 1;
        }
        if start != at {
            visit(&bytes[start..at]);
        }
    }
}

fn contains_folded_token(bytes: &[u8], wanted: &[u8]) -> bool {
    let mut found = false;
    for_each_token(bytes, |candidate| {
        if candidate.len() == wanted.len()
            && candidate
                .iter()
                .zip(wanted)
                .all(|(left, right)| left.eq_ignore_ascii_case(right))
        {
            found = true;
        }
    });
    found
}

impl<K: Copy> QueryIndex for ExtractionState<K> {
    fn evaluate(
        &self,
        object: ObjectId,
        revision: Revision,
        query: SearchContent,
    ) -> Result<IndexMatch, FsError> {
        let generation = self.result_generation(object)?.load(Ordering::Acquire);
        let needle = query.term.as_bytes();
        let mut matched = false;
        let mut score = SearchScore::NONE;
        let mut visible_snippet = heapless::String::new();
        let mut work = 0u32;
        let mut current_batches = 0u32;
        let mut semantic_terms = 0u32;
        let mut semantic_found = 0u32;

        self.visit_current(object, revision, |result| {
            current_batches = current_batches.saturating_add(1);
            match query.plane {
                SearchPlane::FactKey => {
                    for fact in &result.facts {
                        work = work.saturating_add(1);
                        if contains_folded(fact.key.as_bytes(), needle) {
                            matched = true;
                            score = SearchScore::EXACT;
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(fact.key.as_bytes());
                            }
                        }
                    }
                }
                SearchPlane::FullText => {
                    for span in &result.text {
                        work = work.saturating_add(1);
                        if contains_folded(span.text.as_bytes(), needle) {
                            matched = true;
                            score = SearchScore::EXACT;
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(span.text.as_bytes());
                            }
                        }
                    }
                    for fact in &result.facts {
                        let FactValue::Utf8(value) = &fact.value else {
                            continue;
                        };
                        work = work.saturating_add(1);
                        if contains_folded(value.as_bytes(), needle) {
                            matched = true;
                            score = SearchScore::EXACT;
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(value.as_bytes());
                            }
                        }
                    }
                    for symbol in &result.symbols {
                        work = work.saturating_add(1);
                        if contains_folded(symbol.name.as_bytes(), needle) {
                            matched = true;
                            score = SearchScore::EXACT;
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(symbol.name.as_bytes());
                            }
                        }
                    }
                }
                SearchPlane::Relationship => {
                    for reference in &result.references {
                        work = work.saturating_add(1);
                        if contains_folded(reference.target.as_bytes(), needle) {
                            matched = true;
                            score = SearchScore::EXACT;
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(reference.target.as_bytes());
                            }
                        }
                    }
                }
                SearchPlane::Semantic => {
                    // This plane is deliberately lexical-semantic until a
                    // model-backed QueryIndex implementation is installed.
                    // Compare tokens exactly instead of a 64-bit bloom sketch:
                    // collisions must never manufacture a filesystem hit.
                    for_each_token(needle, |wanted| {
                        semantic_terms = semantic_terms.saturating_add(1);
                        let mut term_found = false;
                        for span in &result.text {
                            work = work.saturating_add(1);
                            term_found |= contains_folded_token(span.text.as_bytes(), wanted);
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(span.text.as_bytes());
                            }
                        }
                        for fact in &result.facts {
                            let FactValue::Utf8(value) = &fact.value else {
                                continue;
                            };
                            work = work.saturating_add(1);
                            term_found |= contains_folded_token(value.as_bytes(), wanted);
                            if visible_snippet.is_empty() {
                                visible_snippet = snippet(value.as_bytes());
                            }
                        }
                        for reference in &result.references {
                            work = work.saturating_add(1);
                            term_found |=
                                contains_folded_token(reference.target.as_bytes(), wanted);
                        }
                        for symbol in &result.symbols {
                            work = work.saturating_add(1);
                            term_found |= contains_folded_token(symbol.name.as_bytes(), wanted);
                        }
                        if term_found {
                            semantic_found = semantic_found.saturating_add(1);
                        }
                    });
                }
            }
        })?;

        if current_batches == 0 {
            return Err(FsError::IndexIncomplete);
        }

        if query.plane == SearchPlane::Semantic {
            matched = semantic_found != 0;
            score = if semantic_terms == 0 {
                SearchScore::NONE
            } else {
                SearchScore::new(((semantic_found * 10_000) / semantic_terms) as u16)
            };
            if !matched {
                visible_snippet.clear();
            }
        }

        if self.result_generation(object)?.load(Ordering::Acquire) != generation {
            return Err(FsError::Conflict);
        }
        Ok(IndexMatch {
            matched,
            score,
            snippet: visible_snippet,
            work,
            generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_svc_extract::schema::{NormalizedResult, ParserProvenance};

    fn object(index: u32, generation: u32) -> ObjectId {
        ObjectId::from_bits(SlotKey::new(index, generation).to_bits())
    }

    fn snapshot(owner: ExtractionOwner, object: ObjectId) -> Snapshot<u32> {
        Snapshot {
            owner,
            session: SessionId::new(0),
            handle: Handle::from_bits(SlotKey::new(0, 1).to_bits()),
            object,
            identity: InputIdentity::of(object.to_bits(), 1, b"immutable"),
            storage: Some(7),
        }
    }

    fn derived(
        state: &ExtractionState<u32>,
        object: ObjectId,
        parser: ParserId,
        bytes: &[u8],
    ) -> DerivedResult<u32> {
        let input = InputIdentity::of(object.to_bits(), 1, bytes);
        let result = NormalizedResult::empty(ParserProvenance::deterministic(input, parser, 1));
        let (encoded, hot_len) = state.pack_result(&result).unwrap();
        DerivedResult {
            object,
            input,
            parser,
            parser_version: 1,
            encoded,
            hot_len,
            storage: None,
            encoded_len: 0,
            encoded_digest: [0; 32],
        }
    }

    #[test]
    fn snapshot_release_waits_for_every_guarded_reader() {
        let state = ExtractionState::<u32>::new();
        let owner = ExtractionOwner::new(9);
        let lease = state
            .install_snapshot(snapshot(owner, object(1, 1)))
            .ok()
            .expect("snapshot slot");

        let reader = state.snapshot(owner, lease).unwrap();
        assert!(matches!(
            state.claim_snapshot(owner, lease),
            Err(ExtractionError::Filesystem(FsError::Busy))
        ));
        drop(reader);

        let writer = state.claim_snapshot(owner, lease).unwrap();
        assert!(matches!(
            state.snapshot(owner, lease),
            Err(ExtractionError::Filesystem(FsError::Busy))
        ));
        let removed = state.remove_claimed_snapshot(writer).unwrap();
        assert_eq!(removed.storage, Some(7));
        assert!(matches!(
            state.snapshot(owner, lease),
            Err(ExtractionError::InvalidLease)
        ));
    }

    #[test]
    fn contended_result_purge_is_atomic_and_retryable() {
        let state = ExtractionState::<u32>::new();
        let object = object(2, 1);
        let parser = ParserId(4);
        let key = state
            .results
            .insert(derived(&state, object, parser, b"source"))
            .ok()
            .expect("result slot");
        state
            .result_slots
            .set(object.key().index() as usize * MAX_DERIVED_RESULTS + key.index() as usize);

        let reader = state.results.try_read(key).unwrap();
        assert_eq!(state.purge_object(object), Err(FsError::Busy));
        assert_eq!(state.result_count(), 1);
        drop(reader);

        assert_eq!(state.purge_object(object), Ok(Vec::new()));
        assert_eq!(state.result_count(), 0);
    }

    #[test]
    fn hidden_result_contention_cannot_change_a_visible_query() {
        let state = ExtractionState::<u32>::new();
        let visible = object(3, 1);
        let hidden = object(4, 1);
        let parser = ParserId(7);

        let visible_key = state
            .results
            .insert(derived(&state, visible, parser, b"visible"))
            .ok()
            .expect("visible result slot");
        state.result_slots.set(
            visible.key().index() as usize * MAX_DERIVED_RESULTS + visible_key.index() as usize,
        );

        let hidden_key = state
            .results
            .insert(derived(&state, hidden, parser, b"hidden"))
            .ok()
            .expect("hidden result slot");
        state
            .result_slots
            .set(hidden.key().index() as usize * MAX_DERIVED_RESULTS + hidden_key.index() as usize);

        let hidden_writer = state.results.try_write(hidden_key).unwrap();
        let mut visited = 0;
        assert_eq!(
            state.visit_current(visible, 1, |_| visited += 1),
            Ok(()),
            "a hidden shard's writer must not backpressure visible search"
        );
        assert_eq!(visited, 1);
        drop(hidden_writer);
    }

    #[test]
    fn hot_arena_covers_one_result_per_namespace_object() {
        let state = ExtractionState::<u32>::new();
        for index in 0..MAX_DERIVED_RESULTS as u32 {
            let object = object(index, 1);
            let key = state
                .results
                .insert(derived(&state, object, ParserId(7), b"small text"))
                .ok()
                .expect("one result slot per writable file");
            state
                .result_slots
                .set(index as usize * MAX_DERIVED_RESULTS + key.index() as usize);
        }
        assert_eq!(state.result_count(), MAX_DERIVED_RESULTS);
        assert!(state.results.is_full());
    }
}
