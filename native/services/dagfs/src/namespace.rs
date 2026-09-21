//! The containment DAG: names, edges, traversal, and the link/move/unlink
//! rules.
//!
//! # Where an edge lives
//!
//! A containment edge is a **slot in its parent directory's entry table**,
//! not a row in a global edge arena.  That is a deliberate choice with two
//! consequences the authority model depends on:
//!
//! - **One source of truth.** An edge's name, its child, and its generation
//!   are one record in one place.  A global edge table plus per-directory
//!   child indexes would be two structures that could disagree.
//! - **Per-edge staleness for free.** [`SlotTable`] already bumps a slot's
//!   generation on release, so [`EdgeId`] — parent plus entry key — goes
//!   stale exactly when that one edge is removed, while every sibling and
//!   every other route stays live.  That is the bounded edge proof's central
//!   route-identity guarantee.
//!
//! # No paths here
//!
//! The engine's vocabulary is single-component: `lookup(parent, name)`.
//! Paths are client-side traversal input, so this module never
//! parses a `/`-joined string, has no notion of a current directory, and has
//! no ambient root.  A client walks components itself, and each step extends
//! its handle's proof by one edge.
//!
//! # Directory shape
//!
//! Files and directories may both have multiple incoming containment edges.
//! Every directory-edge insertion performs a bounded reachability proof before
//! publication, so diamonds are lawful while self-links and back-edges fail
//! closed. No parent pointer is stored: incoming counts govern reclamation,
//! while exact edge generations govern route validity.

use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::{
    ServiceAlloc,
    bounds::{MAX_DIR_ENTRIES, MAX_DIRECTORIES, MAX_LIST_PAGE, MAX_OBJECTS},
    error::{FsError, FsResult, ProviderError},
    object::{
        DirId, Object, ObjectId, ObjectIdExt, ObjectKind, ObjectSubtype, Revision, StableObjectId,
    },
    reach::ReachSet,
    slot::{
        AtomicSlotVec, BoundedVec, SlotKey, SlotTable, SlotVecRead, SlotVecReservation,
        SlotVecWrite, fs_access,
    },
};

/// One object copied from a fully pinned durable graph snapshot.
pub(crate) struct DurableGraphObject<K> {
    pub(crate) stable: StableObjectId,
    pub(crate) kind: ObjectKind,
    pub(crate) subtype: ObjectSubtype,
    pub(crate) len: u64,
    pub(crate) revision: Revision,
    pub(crate) storage: Option<K>,
}

/// One containment edge copied from a fully pinned durable graph snapshot.
pub(crate) struct DurableGraphEdge {
    pub(crate) parent: StableObjectId,
    pub(crate) name: Name,
    pub(crate) child: StableObjectId,
}

/// A complete, internally consistent durable namespace image.
///
/// Both tables are arena-backed. Inline this type was **119,832 bytes**, its
/// edge table alone 114,688, and it is built on **every commit** rather than
/// only at mount — so it was the largest single consumer of the `serve`
/// future's stack, on the hottest path that has one. The edge bound is a
/// runtime capacity now rather than a const generic, because the caller
/// already knows it from the volume's profile.
pub(crate) struct DurableGraph<K> {
    pub(crate) root: StableObjectId,
    pub(crate) objects: BoundedVec<DurableGraphObject<K>, ServiceAlloc>,
    pub(crate) edges: BoundedVec<DurableGraphEdge, ServiceAlloc>,
}

pub use kore_dagfs::Name;

/// A containment edge's identity: which directory holds it, and which entry
/// slot (with generation) it is.
///
/// Goes stale the moment that one edge is removed — including when a
/// [`Namespace::relocate`] removes the source edge, which is what makes a
/// handle opened through the source fail while a handle opened through a
/// parallel link keeps working.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EdgeId {
    parent: ObjectId,
    entry: SlotKey,
}

impl EdgeId {
    /// The directory this edge hangs beneath.
    #[inline]
    pub const fn parent(self) -> ObjectId {
        self.parent
    }
}

/// One directory entry — the edge's payload.
#[derive(Clone, Copy, Debug)]
struct DirEntry {
    name: Name,
    child: ObjectId,
}

/// A directory's child list.
type DirEntries = SlotTable<DirEntry, MAX_DIR_ENTRIES>;

/// Fresh storage and one parent edge, still hidden behind their native claims.
/// There is no provider placement to release: cancellation is synchronous.
pub(crate) struct PreparedCreation<'a, K> {
    entries: DirWrite<'a>,
    edge: Option<EdgeId>,
    object: Option<SlotVecReservation<'a, Object<K>, ServiceAlloc>>,
    directory: Option<SlotVecReservation<'a, DirEntries, ServiceAlloc>>,
    snapshot: &'a ObjectSnapshot,
}

impl<K: Copy + Eq> PreparedCreation<'_, K> {
    pub(crate) fn child(&self) -> ObjectId {
        ObjectId::from_bits(
            self.object
                .as_ref()
                .expect("prepared object")
                .id()
                .to_bits(),
        )
    }

    pub(crate) fn edge(&self) -> EdgeId {
        self.edge.expect("prepared edge")
    }

    pub(crate) fn stable(&self) -> StableObjectId {
        self.object.as_ref().expect("prepared object").stable()
    }

    pub(crate) fn revision(&self) -> Revision {
        self.object.as_ref().expect("prepared object").revision()
    }

    pub(crate) fn commit(mut self) {
        let child = self.child();
        let dir = self.object.as_ref().expect("prepared object").dir();
        if let Some(directory) = self.directory.take() {
            directory.commit();
        }
        self.object.take().expect("prepared object").commit();
        self.snapshot.publish(child, 0, dir);
        self.edge.take();
        // Releasing the retained parent claim is the namespace publication.
    }
}

impl<K> Drop for PreparedCreation<'_, K> {
    fn drop(&mut self) {
        if let Some(edge) = self.edge.take() {
            self.entries
                .try_remove(edge.entry)
                .expect("unpublished edge is exclusively owned");
        }
        // Reserved object/directory drop retires their generations. Neither
        // their payload nor an object snapshot has been published on abort.
    }
}
// The object arena and the directory arena are the namespace's two
// **volume-scaled** stores, so they are heap-backed rather than inline. A
// service's whole state is polled inside one `serve` future on the founder's
// shared 4 MiB stack, which made the object bound and the stack budget the
// same number: PR #958 overflowed that stack by 744 bytes by growing one
// placement record 8 bytes. Inline arenas are therefore a capacity ceiling
// disguised as a footprint, and these aliases are where that coupling is cut.
// Matches the reasoning already recorded on `ExtractionState`'s arenas.
// Session-scaled state (handles, authority edges) stays inline on purpose.

/// Heap-backed object arena.
type ObjectArena<K> = AtomicSlotVec<Object<K>, ServiceAlloc>;
/// A read claim on one object slot.
type ObjectRead<'a, K> = SlotVecRead<'a, Object<K>, ServiceAlloc>;
/// An exclusive claim on one object slot.
type ObjectWrite<'a, K> = SlotVecWrite<'a, Object<K>, ServiceAlloc>;
/// Heap-backed arena of directory child lists.
type DirArena = AtomicSlotVec<DirEntries, ServiceAlloc>;
/// A read claim on one directory's child list.
type DirRead<'a> = SlotVecRead<'a, DirEntries, ServiceAlloc>;
/// An exclusive claim on one directory's child list.
type DirWrite<'a> = SlotVecWrite<'a, DirEntries, ServiceAlloc>;

/// A canonically ordered set of exact directory mutation claims.
///
/// Multi-directory commits acquire every member before final validation and
/// publication, so refusal drops the whole bounded footprint without a
/// partial move, stale emptiness decision, or graph cycle.
struct DirectoryClaims<'a> {
    entries: heapless::Vec<DirectoryClaim<'a>, MAX_DIRECTORIES>,
    /// Directory-arena slot index → position in `entries`, or [`UNCLAIMED`].
    /// The walks look a claim up once per popped node and once per edge;
    /// an index makes each of those O(1) instead of a scan of the claim
    /// list. It is filled in claim order, after every generation is pinned.
    by_dir: [u8; MAX_DIRECTORIES],
}

/// The `by_dir` sentinel: no claim stands for that directory slot.
const UNCLAIMED: u8 = u8::MAX;
const _: () = assert!(MAX_DIRECTORIES < UNCLAIMED as usize);

struct DirectoryClaim<'a> {
    object: ObjectId,
    dir: DirId,
    entries: DirWrite<'a>,
}

/// A checked rename holds its exact directory until publication or drop.
/// Dropping a prepared mutation has no namespace effect.
pub(crate) struct PreparedRename<'a> {
    entries: DirWrite<'a>,
    edge: EdgeId,
    child: ObjectId,
    new: Name,
}

impl PreparedRename<'_> {
    pub(crate) fn child(&self) -> ObjectId {
        self.child
    }

    pub(crate) fn commit(self) -> FsResult<EdgeId> {
        let mut entry = self
            .entries
            .try_write(self.edge.entry)
            .map_err(|error| fs_access(error, FsError::AccessLost))?;
        entry.name = self.new;
        Ok(self.edge)
    }
}

/// A checked removal retains both the namespace footprint and the child's
/// incoming counter, so post-receipt publication needs no new object claim.
pub(crate) struct PreparedRemoval<'a, K> {
    claims: DirectoryClaims<'a>,
    edge: EdgeId,
    child: ObjectId,
    object: ObjectWrite<'a, K>,
}

pub(crate) struct PreparedRelocate<'a> {
    claims: DirectoryClaims<'a>,
    source: EdgeId,
    child: ObjectId,
    destination: ObjectId,
    name: Name,
    unchanged: bool,
}

impl PreparedRelocate<'_> {
    pub(crate) fn child(&self) -> ObjectId {
        self.child
    }

    /// Revalidate a route through a directory already claimed by this move.
    /// Unclaimed ancestors still use the namespace's ordinary proof walk.
    pub(crate) fn edge_child(&self, edge: EdgeId) -> Option<FsResult<ObjectId>> {
        if !self.claims.contains(edge.parent) {
            return None;
        }
        Some(self.claims.get(edge.parent).and_then(|entries| {
            entries
                .try_read(edge.entry)
                .map(|entry| entry.child)
                .map_err(|error| fs_access(error, FsError::AccessLost))
        }))
    }

    pub(crate) fn commit(mut self) -> FsResult<(EdgeId, EdgeId)> {
        if self.unchanged {
            return Ok((self.source, self.source));
        }
        self.claims
            .get_mut(self.source.parent)?
            .try_remove(self.source.entry)
            .map_err(|error| fs_access(error, FsError::AccessLost))?;
        let entry = self
            .claims
            .get_mut(self.destination)?
            .insert(DirEntry {
                name: self.name,
                child: self.child,
            })
            .map_err(|_| FsError::OutOfEdges)?;
        Ok((
            self.source,
            EdgeId {
                parent: self.destination,
                entry,
            },
        ))
    }
}

impl<K: Copy + Eq> PreparedRemoval<'_, K> {
    pub(crate) fn stable(&self) -> StableObjectId {
        self.object.stable()
    }

    pub(crate) fn commit(mut self) -> FsResult<(EdgeId, ObjectId)> {
        self.claims
            .get_mut(self.edge.parent)?
            .try_remove(self.edge.entry)
            .map_err(|error| fs_access(error, FsError::AccessLost))?;
        self.object.incoming_dec();
        Ok((self.edge, self.child))
    }
}

impl DirectoryClaims<'_> {
    /// The claim standing for `dir`, when it is the claim of exactly `object`
    /// — the object↔directory correspondence `claim_directories` pinned is
    /// re-checked, so a reclaimed slot can never answer for its successor.
    fn position_of_dir(&self, dir: DirId, object: ObjectId) -> Option<usize> {
        let at = *self.by_dir.get(dir.0.index() as usize)?;
        if at == UNCLAIMED {
            return None;
        }
        let claim = self.entries.get(at as usize)?;
        (claim.dir == dir && claim.object == object).then_some(at as usize)
    }

    fn entries_at(&self, at: usize) -> &DirEntries {
        &self.entries[at].entries
    }

    fn contains(&self, object: ObjectId) -> bool {
        self.entries.iter().any(|claim| claim.object == object)
    }

    fn get(&self, object: ObjectId) -> FsResult<&DirEntries> {
        self.entries
            .iter()
            .find(|claim| claim.object == object)
            .map(|claim| &*claim.entries)
            .ok_or(FsError::Busy)
    }

    fn get_mut(&mut self, object: ObjectId) -> FsResult<&mut DirEntries> {
        self.entries
            .iter_mut()
            .find(|claim| claim.object == object)
            .map(|claim| &mut *claim.entries)
            .ok_or(FsError::Busy)
    }
}

/// One entry as reported by [`Namespace::list`].
#[derive(Clone, Copy, Debug)]
pub struct ListEntry {
    pub name: Name,
    pub object: ObjectId,
    pub kind: ObjectKind,
    /// Data, never authority ([`ObjectSubtype`]).
    pub subtype: ObjectSubtype,
}

/// The object arena plus the containment DAG over it.
pub(crate) struct Namespace<K> {
    objects: ObjectArena<K>,
    /// Lock-free quota/traversal projection of each object slot.
    ///
    /// File provider transactions retain the canonical object write claim
    /// across suspension. Readers that only need immutable graph shape and
    /// the last committed length use this generation-validated atomic
    /// snapshot instead of contending on the file payload.
    snapshots: Box<[ObjectSnapshot]>,
    dirs: DirArena,
    /// Placements the provider refused to release.  A reclamation's
    /// namespace effect has already committed by the time the provider is
    /// asked, so a failed release cannot un-remove the edge; it is counted
    /// here and surfaced in engine statistics rather than reported as a
    /// failure of a request that genuinely succeeded.
    leaked_placements: AtomicU32,
    /// Reclamations that abandoned part of a subtree — a structural refusal
    /// (a concurrently stale slot, an exhausted arena) or contention that
    /// outlasted [`RECLAIM_CONTENTION_TURNS`](crate::bounds::RECLAIM_CONTENTION_TURNS).
    ///
    /// Distinct from [`leaked_placements`](Self::leaked_placements), which is
    /// the *provider* refusing to release bytes the namespace already
    /// disowned. This one says the namespace walk itself stopped early, so the
    /// objects below the stopping point stay resident and unreachable. Counted
    /// rather than raised for the same reason: the effect that triggered the
    /// reclamation has already committed.
    abandoned_reclaims: AtomicU32,
    next_stable: AtomicU64,
}

/// Exact live graph usage below one authority root.
///
/// Objects are counted once even when a directory diamond reaches them by
/// several routes. Containment edges are counted individually because each is
/// independently named, retired, and consumes one bounded entry slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GraphUsage {
    pub(crate) objects: u32,
    pub(crate) edges: u32,
    pub(crate) bytes: u64,
}

/// One exact object's atomically published graph/accounting projection.
#[derive(Clone, Copy, Debug)]
struct ObjectSnapshotView {
    len: u64,
    dir: Option<DirId>,
}

/// Keep independently hot object snapshots on distinct cache lines.
///
/// `sequence` is even while stable and odd only for the handful of stores in
/// a metadata commit. A reader never spins behind a preempted writer: it
/// returns typed [`FsError::Busy`] and lets the bounded request retry policy
/// decide when to revisit the snapshot.
#[repr(align(64))]
struct ObjectSnapshot {
    sequence: AtomicU64,
    generation: AtomicU32,
    _pad: AtomicU32,
    len: AtomicU64,
    dir: AtomicU64,
}

impl ObjectSnapshot {
    const fn vacant() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            _pad: AtomicU32::new(0),
            len: AtomicU64::new(0),
            dir: AtomicU64::new(0),
        }
    }

    fn publish(&self, id: ObjectId, len: u64, dir: Option<DirId>) {
        let stable = self.sequence.load(Ordering::Acquire);
        assert_eq!(stable & 1, 0, "one object snapshot writer per slot");
        self.sequence
            .store(stable.wrapping_add(1), Ordering::Release);
        self.generation
            .store(id.key().generation(), Ordering::Relaxed);
        self.len.store(len, Ordering::Relaxed);
        self.dir
            .store(dir.map_or(0, |dir| dir.0.to_bits()), Ordering::Relaxed);
        self.sequence
            .store(stable.wrapping_add(2), Ordering::Release);
    }

    fn clear(&self, id: ObjectId) {
        let stable = self.sequence.load(Ordering::Acquire);
        assert_eq!(stable & 1, 0, "one object snapshot writer per slot");
        self.sequence
            .store(stable.wrapping_add(1), Ordering::Release);
        if self.generation.load(Ordering::Relaxed) == id.key().generation() {
            self.generation.store(0, Ordering::Relaxed);
            self.len.store(0, Ordering::Relaxed);
            self.dir.store(0, Ordering::Relaxed);
        }
        self.sequence
            .store(stable.wrapping_add(2), Ordering::Release);
    }

    fn read(&self, id: ObjectId) -> FsResult<ObjectSnapshotView> {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 != 0 {
            return Err(FsError::Busy);
        }
        let generation = self.generation.load(Ordering::Relaxed);
        let len = self.len.load(Ordering::Relaxed);
        let dir = self.dir.load(Ordering::Relaxed);
        let after = self.sequence.load(Ordering::Acquire);
        if before != after {
            return Err(FsError::Busy);
        }
        if generation != id.key().generation() || generation == 0 {
            return Err(FsError::AccessLost);
        }
        Ok(ObjectSnapshotView {
            len,
            dir: (dir != 0).then(|| DirId(SlotKey::from_bits(dir))),
        })
    }
}

/// One vacant snapshot per object slot, heap-backed for the reason recorded on
/// [`ObjectArena`]. `ObjectSnapshot` holds atomics and is deliberately not
/// `Clone`, so the slice is filled by construction rather than by `resize`.
fn vacant_snapshots() -> Box<[ObjectSnapshot]> {
    let mut snapshots = Vec::new();
    snapshots
        .try_reserve_exact(MAX_OBJECTS)
        .expect("fs object-snapshot projection (bootstrap-scale) — service heap undersized");
    for _ in 0..MAX_OBJECTS {
        snapshots.push(ObjectSnapshot::vacant());
    }
    snapshots.into_boxed_slice()
}

impl<K: Copy + Eq> Namespace<K> {
    pub(crate) fn new() -> Self {
        // Bootstrap-scale and fixed after startup, exactly like
        // `ExtractionState`'s arenas: an allocator that cannot satisfy these
        // at mount is an undersized service heap, which is a build
        // configuration error of the same class as the compile-time size
        // guard — not a runtime volume condition. Keeping construction
        // infallible is therefore deliberate, and it keeps every caller
        // (including `Engine::new`) unchanged.
        let alloc = crate::service_alloc();
        Self {
            objects: AtomicSlotVec::try_new_in(MAX_OBJECTS, alloc)
                .expect("fs object arena (bootstrap-scale) — service heap undersized"),
            snapshots: vacant_snapshots(),
            dirs: AtomicSlotVec::try_new_in(MAX_DIRECTORIES, alloc)
                .expect("fs directory arena (bootstrap-scale) — service heap undersized"),
            leaked_placements: AtomicU32::new(0),
            abandoned_reclaims: AtomicU32::new(0),
            next_stable: AtomicU64::new(1),
        }
    }

    // ── object access ───────────────────────────────────────────────────────

    pub(crate) fn object(&self, id: ObjectId) -> FsResult<ObjectRead<'_, K>> {
        self.objects
            .try_read(id.key())
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    pub(crate) fn object_mut(&self, id: ObjectId) -> FsResult<ObjectWrite<'_, K>> {
        self.objects
            .try_write(id.key())
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    fn snapshot(&self, id: ObjectId) -> FsResult<ObjectSnapshotView> {
        let index = usize::try_from(id.key().index()).map_err(|_| FsError::AccessLost)?;
        self.snapshots
            .get(index)
            .ok_or(FsError::AccessLost)?
            .read(id)
    }

    /// Publish one file's visible length and revision as a single file
    /// transaction. The quota snapshot is updated while the caller still
    /// holds the exact object claim and its pending growth reservation.
    pub(crate) fn commit_file_len(
        &self,
        id: ObjectId,
        object: &mut ObjectWrite<'_, K>,
        len: u64,
    ) -> Revision {
        debug_assert_eq!(object.id(), id.key());
        object.commit_len(len);
        self.snapshots[id.key().index() as usize].publish(id, len, object.dir());
        object.revision()
    }

    /// Verified host hydration changes placement, not source revision. Publish
    /// its logical length into the same quota snapshot used by native writes.
    pub(crate) fn hydrate_file_len(
        &self,
        id: ObjectId,
        object: &mut ObjectWrite<'_, K>,
        len: u64,
    ) -> Revision {
        debug_assert_eq!(object.id(), id.key());
        let revision = object.revision();
        object.restore_len_revision(len, revision);
        self.snapshots[id.key().index() as usize].publish(id, len, object.dir());
        revision
    }

    pub(crate) fn exists(&self, id: ObjectId) -> bool {
        self.objects.contains(id.key())
    }

    pub(crate) fn kind(&self, id: ObjectId) -> FsResult<ObjectKind> {
        Ok(self.object(id)?.kind())
    }

    pub(crate) fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub(crate) fn directory_count(&self) -> usize {
        self.dirs.len()
    }

    pub(crate) fn leaked_placements(&self) -> u32 {
        self.leaked_placements.load(Ordering::Relaxed)
    }

    pub(crate) fn abandoned_reclaims(&self) -> u32 {
        self.abandoned_reclaims.load(Ordering::Relaxed)
    }

    pub(crate) fn objects(&self) -> impl Iterator<Item = (ObjectId, ObjectRead<'_, K>)> {
        self.objects
            .iter()
            .map(|(key, object)| (ObjectId::from_bits(key.to_bits()), object))
    }

    pub(crate) fn find_stable(&self, stable: StableObjectId) -> Option<ObjectId> {
        self.objects()
            .find_map(|(id, object)| (object.stable() == stable).then_some(id))
    }

    #[cfg(test)]
    pub(crate) fn collect_reachable(
        &self,
        root: ObjectId,
    ) -> FsResult<heapless::Vec<ObjectId, MAX_OBJECTS>> {
        let mut found = heapless::Vec::new();
        let mut pending = heapless::Vec::<ObjectId, MAX_OBJECTS>::new();
        pending.push(root).map_err(|_| FsError::OutOfObjects)?;
        while let Some(id) = pending.pop() {
            if found.contains(&id) {
                continue;
            }
            let snapshot = self.snapshot(id)?;
            found.push(id).map_err(|_| FsError::OutOfObjects)?;
            if let Some(dir) = snapshot.dir {
                let entries = self
                    .dirs
                    .try_read(dir.0)
                    .map_err(|error| fs_access(error, FsError::AccessLost))?;
                for (_, entry) in entries.iter() {
                    if !found.contains(&entry.child) && !pending.contains(&entry.child) {
                        pending
                            .push(entry.child)
                            .map_err(|_| FsError::OutOfObjects)?;
                    }
                }
            }
        }
        Ok(found)
    }

    /// Copy one complete reachable graph while retaining exact per-slot read
    /// guards over every object and directory in the footprint.
    ///
    /// Writers never wait behind this snapshot: a conflicting request gets
    /// typed [`FsError::Busy`]. The guards are released before the caller
    /// performs provider or device I/O, so an fsync cannot pin hot namespace
    /// cache lines for the duration of an NVMe flush.
    /// `max_edges` is the volume profile's containment-edge ceiling, which
    /// the caller already holds; it bounds the two edge buffers exactly as the
    /// old const generic did.
    ///
    /// An allocation that cannot be satisfied reports the same error a full
    /// buffer reports, because it is the same condition from the caller's
    /// side: the buffer this snapshot needs cannot be provided. Commit is a
    /// live path in a founded service, so it may not panic — see the contract
    /// on `Engine::reclaim`.
    pub(crate) fn durable_graph(
        &self,
        root: ObjectId,
        max_edges: usize,
    ) -> FsResult<DurableGraph<K>> {
        let alloc = crate::service_alloc();
        let mut pending = BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_OBJECTS, alloc)
            .map_err(|_| FsError::OutOfObjects)?;
        let mut objects = BoundedVec::<(ObjectId, ObjectRead<'_, K>), ServiceAlloc>::try_new_in(
            MAX_OBJECTS,
            alloc,
        )
        .map_err(|_| FsError::OutOfObjects)?;
        let mut dirs =
            BoundedVec::<(DirId, DirRead<'_>), ServiceAlloc>::try_new_in(MAX_DIRECTORIES, alloc)
                .map_err(|_| FsError::OutOfObjects)?;
        let mut raw_edges =
            BoundedVec::<(ObjectId, Name, ObjectId), ServiceAlloc>::try_new_in(max_edges, alloc)
                .map_err(|_| FsError::OutOfEdges)?;
        pending.push(root).map_err(|_| FsError::OutOfObjects)?;

        while let Some(id) = pending.pop() {
            if objects.iter().any(|(seen, _)| *seen == id) {
                continue;
            }
            let object = self.objects.try_read(id.key()).map_err(|_| FsError::Busy)?;
            let dir = object.dir();
            objects
                .push((id, object))
                .map_err(|_| FsError::OutOfObjects)?;
            let Some(dir) = dir else { continue };
            if dirs.iter().any(|(seen, _)| *seen == dir) {
                return Err(FsError::AccessLost);
            }
            let entries = self.dirs.try_read(dir.0).map_err(|_| FsError::Busy)?;
            for (_, entry) in entries.iter() {
                raw_edges
                    .push((id, entry.name, entry.child))
                    .map_err(|_| FsError::OutOfEdges)?;
                if !objects.iter().any(|(seen, _)| *seen == entry.child)
                    && !pending.contains(&entry.child)
                {
                    pending
                        .push(entry.child)
                        .map_err(|_| FsError::OutOfObjects)?;
                }
            }
            dirs.push((dir, entries))
                .map_err(|_| FsError::OutOfObjects)?;
        }

        let stable = |wanted: ObjectId| {
            objects
                .iter()
                .find_map(|(id, object)| (*id == wanted).then(|| object.stable()))
                .ok_or(FsError::AccessLost)
        };
        let root_stable = stable(root)?;
        let mut copied_objects =
            BoundedVec::<DurableGraphObject<K>, ServiceAlloc>::try_new_in(MAX_OBJECTS, alloc)
                .map_err(|_| FsError::OutOfObjects)?;
        for (id, object) in objects.iter() {
            copied_objects
                .push(DurableGraphObject {
                    stable: object.stable(),
                    kind: object.kind(),
                    subtype: object.subtype(),
                    len: object.len(),
                    revision: object.revision(),
                    storage: object.storage(),
                })
                .map_err(|_| FsError::OutOfObjects)?;
        }
        let mut copied_edges =
            BoundedVec::<DurableGraphEdge, ServiceAlloc>::try_new_in(max_edges, alloc)
                .map_err(|_| FsError::OutOfEdges)?;
        for (parent, name, child) in raw_edges {
            copied_edges
                .push(DurableGraphEdge {
                    parent: stable(parent)?,
                    name,
                    child: stable(child)?,
                })
                .map_err(|_| FsError::OutOfEdges)?;
        }
        Ok(DurableGraph {
            root: root_stable,
            objects: copied_objects,
            edges: copied_edges,
        })
    }

    /// Derive quota usage from the live namespace rather than maintaining a
    /// second counter ledger beside it.
    pub(crate) fn graph_usage(&self, root: ObjectId) -> FsResult<GraphUsage> {
        self.graph_usage_inner(root)
    }

    /// Derive graph usage while the caller owns one exact object write claim.
    ///
    /// The commit snapshot deliberately remains readable while that canonical
    /// record is claimed, so quota observes the last visible length plus all
    /// pending reservations. The arguments document and type-check the
    /// claimed object/route correspondence at the engine boundary; no payload
    /// alias is created here.
    pub(crate) fn graph_usage_with_claimed(
        &self,
        root: ObjectId,
        _claimed_id: ObjectId,
        _claimed: &Object<K>,
    ) -> FsResult<GraphUsage> {
        self.graph_usage_inner(root)
    }

    fn graph_usage_inner(&self, root: ObjectId) -> FsResult<GraphUsage> {
        let alloc = crate::service_alloc();
        let mut found = BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_OBJECTS, alloc)
            .map_err(|_| FsError::OutOfObjects)?;
        let mut pending = BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_OBJECTS, alloc)
            .map_err(|_| FsError::OutOfObjects)?;
        pending.push(root).map_err(|_| FsError::OutOfObjects)?;
        let mut usage = GraphUsage::default();
        while let Some(id) = pending.pop() {
            if found.contains(&id) {
                continue;
            }
            let snapshot = self.snapshot(id)?;
            found.push(id).map_err(|_| FsError::OutOfObjects)?;
            usage.objects = usage.objects.checked_add(1).ok_or(FsError::OutOfObjects)?;
            usage.bytes = usage
                .bytes
                .checked_add(snapshot.len)
                .ok_or(FsError::OutOfStorage)?;
            if let Some(dir) = snapshot.dir {
                let entries = self
                    .dirs
                    .try_read(dir.0)
                    .map_err(|error| fs_access(error, FsError::AccessLost))?;
                usage.edges = usage
                    .edges
                    .checked_add(u32::try_from(entries.len()).map_err(|_| FsError::OutOfEdges)?)
                    .ok_or(FsError::OutOfEdges)?;
                for (_, entry) in entries.iter() {
                    if !found.contains(&entry.child) && !pending.contains(&entry.child) {
                        pending
                            .push(entry.child)
                            .map_err(|_| FsError::OutOfObjects)?;
                    }
                }
            }
        }
        Ok(usage)
    }

    pub(crate) fn restore_object(
        &self,
        stable: StableObjectId,
        kind: ObjectKind,
        subtype: ObjectSubtype,
        len: u64,
        revision: u64,
        storage: Option<K>,
    ) -> FsResult<ObjectId> {
        if stable == StableObjectId::INVALID || self.find_stable(stable).is_some() {
            return Err(FsError::ProviderFailure(ProviderError::Integrity));
        }
        // A directory carries no subtype; the durable decoder refuses one, so
        // reaching here with one is the caller disagreeing with itself.
        if kind.is_dir() && !subtype.is_plain() {
            return Err(FsError::ProviderFailure(ProviderError::Integrity));
        }
        let id = match kind {
            ObjectKind::File => self
                .objects
                .insert(Object::new_file_with(subtype).with_stable(stable))
                .map(|key| ObjectId::from_bits(key.to_bits()))
                .map_err(|_| FsError::OutOfObjects)?,
            ObjectKind::Directory => {
                let dir = self
                    .dirs
                    .insert(DirEntries::new())
                    .map(DirId)
                    .map_err(|_| FsError::OutOfObjects)?;
                match self
                    .objects
                    .insert(Object::new_dir(dir).with_stable(stable))
                {
                    Ok(key) => ObjectId::from_bits(key.to_bits()),
                    Err(_) => {
                        self.dirs.remove(dir.0);
                        return Err(FsError::OutOfObjects);
                    }
                }
            }
        };
        let mut object = self.object_mut(id)?;
        if let Some(key) = storage {
            object.set_storage(key);
        }
        object.restore_len_revision(len, revision);
        let dir = object.dir();
        drop(object);
        self.snapshots[id.key().index() as usize].publish(id, len, dir);
        self.next_stable
            .fetch_max(stable.get().saturating_add(1), Ordering::Relaxed);
        Ok(id)
    }

    pub(crate) fn restore_edge(
        &self,
        parent: StableObjectId,
        name: Name,
        child: StableObjectId,
    ) -> FsResult<()> {
        let parent = self
            .find_stable(parent)
            .ok_or(FsError::ProviderFailure(ProviderError::Integrity))?;
        let child = self
            .find_stable(child)
            .ok_or(FsError::ProviderFailure(ProviderError::Integrity))?;
        self.link(parent, &name, child)?;
        Ok(())
    }

    fn allocate_stable(&self) -> FsResult<StableObjectId> {
        let mut current = self.next_stable.load(Ordering::Relaxed);
        loop {
            let id = StableObjectId::new(current).ok_or(FsError::OutOfObjects)?;
            let next = current.checked_add(1).ok_or(FsError::OutOfObjects)?;
            match self.next_stable.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(id),
                Err(observed) => current = observed,
            }
        }
    }

    // ── object creation ─────────────────────────────────────────────────────

    pub(crate) fn prepare_creation(
        &self,
        parent: ObjectId,
        name: &Name,
        kind: ObjectKind,
        subtype: ObjectSubtype,
    ) -> FsResult<PreparedCreation<'_, K>> {
        let entries = self.dir_entries_mut(parent)?;
        if entries.iter().any(|(_, entry)| entry.name == *name) {
            return Err(FsError::AlreadyExists);
        }
        let stable = self.allocate_stable()?;
        let directory = if kind.is_dir() {
            Some(
                self.dirs
                    .reserve(DirEntries::new())
                    .map_err(|_| FsError::OutOfObjects)?,
            )
        } else {
            None
        };
        let value = match directory.as_ref() {
            Some(directory) => Object::new_dir(DirId(directory.id())),
            None => Object::new_file_with(subtype),
        }
        .with_stable(stable);
        let mut object = self
            .objects
            .reserve(value)
            .map_err(|_| FsError::OutOfObjects)?;
        let child = ObjectId::from_bits(object.id().to_bits());
        let entry = entries
            .insert(DirEntry { name: *name, child })
            .map_err(|_| FsError::OutOfEdges)?;
        object.incoming_inc();
        Ok(PreparedCreation {
            entries,
            edge: Some(EdgeId { parent, entry }),
            object: Some(object),
            directory,
            snapshot: &self.snapshots[child.key().index() as usize],
        })
    }

    /// Create an unlinked file object.  The caller links it in the same
    /// request; until then it is reclaimable, which is what makes a failed
    /// `create` leave nothing behind.
    pub(crate) fn create_file_object(&self) -> FsResult<ObjectId> {
        self.create_file_object_with(ObjectSubtype::PLAIN)
    }

    /// [`Self::create_file_object`], stamped `subtype`.
    pub(crate) fn create_file_object_with(&self, subtype: ObjectSubtype) -> FsResult<ObjectId> {
        let stable = self.allocate_stable()?;
        let id = self
            .objects
            .insert(Object::new_file_with(subtype).with_stable(stable))
            .map(|key| ObjectId::from_bits(key.to_bits()))
            .map_err(|_| FsError::OutOfObjects)?;
        self.snapshots[id.key().index() as usize].publish(id, 0, None);
        Ok(id)
    }

    /// Create an unlinked directory object, allocating its child list.
    pub(crate) fn create_dir_object(&self) -> FsResult<ObjectId> {
        let stable = self.allocate_stable()?;
        // Reserve the child list first: if the object arena is full we must
        // not leak a directory slot, and if the directory arena is full we
        // must not leak an object slot.
        let dir = self
            .dirs
            .insert(DirEntries::new())
            .map(DirId)
            .map_err(|_| FsError::OutOfObjects)?;
        match self
            .objects
            .insert(Object::new_dir(dir).with_stable(stable))
        {
            Ok(key) => {
                let id = ObjectId::from_bits(key.to_bits());
                self.snapshots[id.key().index() as usize].publish(id, 0, Some(dir));
                Ok(id)
            }
            Err(_) => {
                self.dirs.remove(dir.0);
                Err(FsError::OutOfObjects)
            }
        }
    }

    // ── edges ───────────────────────────────────────────────────────────────

    fn dir_entries(&self, parent: ObjectId) -> FsResult<DirRead<'_>> {
        let dir = self.object(parent)?.dir().ok_or(FsError::NotDirectory)?;
        // A live directory object always owns a live child list; a missing
        // one would mean the two arenas had desynchronized.
        self.dirs
            .try_read(dir.0)
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    fn dir_entries_mut(&self, parent: ObjectId) -> FsResult<DirWrite<'_>> {
        let dir = self.object(parent)?.dir().ok_or(FsError::NotDirectory)?;
        self.dirs
            .try_write(dir.0)
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    /// Claim an explicit directory set in stable slot order.
    fn claim_directories<'a>(&'a self, objects: &[ObjectId]) -> FsResult<DirectoryClaims<'a>> {
        let mut wanted = BoundedVec::<(ObjectId, DirId), ServiceAlloc>::try_new_in(
            MAX_DIRECTORIES,
            crate::service_alloc(),
        )
        .map_err(|_| FsError::OutOfObjects)?;
        for object in objects {
            let dir = self.snapshot(*object)?.dir.ok_or(FsError::NotDirectory)?;
            if let Some((_, existing)) = wanted.iter().find(|(id, _)| id == object) {
                if *existing != dir {
                    return Err(FsError::AccessLost);
                }
                continue;
            }
            if wanted.iter().any(|(_, existing)| *existing == dir) {
                return Err(FsError::AccessLost);
            }
            wanted
                .push((*object, dir))
                .map_err(|_| FsError::OutOfObjects)?;
        }
        wanted.as_mut_slice().sort_unstable_by_key(|(_, dir)| dir.0);

        let mut claimed = DirectoryClaims {
            entries: heapless::Vec::new(),
            by_dir: [UNCLAIMED; MAX_DIRECTORIES],
        };
        for (object, dir) in wanted {
            let entries = self
                .dirs
                .try_write(dir.0)
                .map_err(|error| fs_access(error, FsError::AccessLost))?;
            claimed
                .entries
                .push(DirectoryClaim {
                    object,
                    dir,
                    entries,
                })
                .map_err(|_| FsError::OutOfObjects)?;
        }

        // Close the optimistic object→directory lookup after every exact
        // directory generation is pinned. A reclaimed/reused object cannot
        // make this set accidentally name its successor.
        for (at, claim) in claimed.entries.iter().enumerate() {
            if self.snapshot(claim.object)?.dir != Some(claim.dir) {
                return Err(FsError::AccessLost);
            }
            let slot = claimed
                .by_dir
                .get_mut(claim.dir.0.index() as usize)
                .ok_or(FsError::Range)?;
            *slot = at as u8;
        }
        Ok(claimed)
    }

    /// Snapshot the directory subgraph below `root` for later canonical
    /// claiming. The claimed traversal revalidates this set before mutation;
    /// an edge added in the observation→claim interval yields `Busy` rather
    /// than escaping the cycle footprint.
    fn directory_subtree(&self, root: ObjectId) -> FsResult<BoundedVec<ObjectId, ServiceAlloc>> {
        let alloc = crate::service_alloc();
        let mut found = BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_DIRECTORIES, alloc)
            .map_err(|_| FsError::OutOfObjects)?;
        let mut pending = BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_DIRECTORIES, alloc)
            .map_err(|_| FsError::OutOfObjects)?;
        // Marked on enqueue, keyed by directory slot: each directory is
        // queued exactly once, so the worklist is bounded by the arena and
        // membership is O(1) rather than two list scans per edge.
        let mut seen = ReachSet::new();
        let Some(dir) = self.snapshot(root)?.dir else {
            return Ok(found);
        };
        seen.insert(dir.0.index() as usize)?;
        pending.push(root).map_err(|_| FsError::OutOfObjects)?;
        while let Some(object) = pending.pop() {
            let Some(dir) = self.snapshot(object)?.dir else {
                continue;
            };
            found.push(object).map_err(|_| FsError::OutOfObjects)?;
            let entries = self
                .dirs
                .try_read(dir.0)
                .map_err(|error| fs_access(error, FsError::AccessLost))?;
            for (_, entry) in entries.iter() {
                let Some(child) = self.snapshot(entry.child)?.dir else {
                    continue;
                };
                if seen.insert(child.0.index() as usize)? {
                    pending
                        .push(entry.child)
                        .map_err(|_| FsError::OutOfObjects)?;
                }
            }
        }
        Ok(found)
    }

    fn claim_subtree_with<'a>(
        &'a self,
        root: ObjectId,
        extras: &[ObjectId],
    ) -> FsResult<DirectoryClaims<'a>> {
        let mut objects = self.directory_subtree(root)?;
        for extra in extras {
            if !objects.contains(extra) {
                objects.push(*extra).map_err(|_| FsError::OutOfObjects)?;
            }
        }
        self.claim_directories(&objects)
    }

    fn is_reachable_claimed(
        &self,
        claims: &DirectoryClaims<'_>,
        ancestor: ObjectId,
        candidate: ObjectId,
    ) -> FsResult<bool> {
        if ancestor == candidate {
            return Ok(true);
        }
        // The walk speaks claim positions, keyed by directory slot: one bit
        // row for "seen", marked on enqueue so every directory is queued at
        // most once, and the claim index answers each edge in O(1). The
        // protocol is unchanged — a child directory outside the pinned
        // footprint is `Busy`, and the caller decides `Cycle`.
        let mut seen = ReachSet::new();
        let mut pending =
            BoundedVec::<u8, ServiceAlloc>::try_new_in(MAX_DIRECTORIES, crate::service_alloc())
                .map_err(|_| FsError::OutOfObjects)?;
        let start_dir = self.snapshot(ancestor)?.dir.ok_or(FsError::NotDirectory)?;
        let start = claims
            .position_of_dir(start_dir, ancestor)
            .ok_or(FsError::Busy)?;
        seen.insert(start_dir.0.index() as usize)?;
        pending.push(start as u8).map_err(|_| FsError::Range)?;
        while let Some(at) = pending.pop() {
            for (_, entry) in claims.entries_at(at as usize).iter() {
                if entry.child == candidate {
                    return Ok(true);
                }
                let Some(dir) = self.snapshot(entry.child)?.dir else {
                    continue;
                };
                // The optimistic footprint changed before it was pinned.
                // Retrying takes a complete new snapshot.
                let Some(next) = claims.position_of_dir(dir, entry.child) else {
                    return Err(FsError::Busy);
                };
                if seen.insert(dir.0.index() as usize)? {
                    pending.push(next as u8).map_err(|_| FsError::Range)?;
                }
            }
        }
        Ok(false)
    }

    /// Resolve one name component beneath `parent`.
    pub(crate) fn lookup(&self, parent: ObjectId, name: &Name) -> FsResult<(EdgeId, ObjectId)> {
        let entries = self.dir_entries(parent)?;
        for (key, entry) in entries.iter() {
            if entry.name == *name {
                return Ok((EdgeId { parent, entry: key }, entry.child));
            }
        }
        Err(FsError::NotFound)
    }

    /// The child an edge currently names — `AccessLost` if the edge has been
    /// removed or its generation retired.  This is the primitive the bounded
    /// edge proof revalidates against.
    pub(crate) fn edge_child(&self, edge: EdgeId) -> FsResult<ObjectId> {
        let entries = self.dir_entries(edge.parent)?;
        entries
            .try_read(edge.entry)
            .map(|entry| entry.child)
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    /// Add a containment edge `parent/name -> child`, incrementing the
    /// child's incoming count.
    ///
    /// Directory children are legal only when adding this edge cannot make
    /// `parent` reachable from `child`. The bounded visited set distinguishes
    /// a lawful diamond from a back-edge; all checks complete before the slot
    /// and incoming count are mutated.
    pub(crate) fn link(&self, parent: ObjectId, name: &Name, child: ObjectId) -> FsResult<EdgeId> {
        if !self.kind(parent)?.is_dir() {
            return Err(FsError::NotDirectory);
        }
        if !self.exists(child) {
            return Err(FsError::AccessLost);
        }
        let child_is_dir = self.kind(child)?.is_dir();
        let mut claims = if child_is_dir {
            self.claim_subtree_with(child, &[parent])?
        } else {
            self.claim_directories(&[parent])?
        };
        if child_is_dir && self.is_reachable_claimed(&claims, child, parent)? {
            return Err(FsError::Cycle);
        }
        // The final name check happens under the exact destination claim. Two
        // concurrent creators cannot both publish the same label.
        let entries = claims.get_mut(parent)?;
        if entries.iter().any(|(_, entry)| entry.name == *name) {
            return Err(FsError::AlreadyExists);
        }
        // Claim the child's incoming counter before publishing the edge, so
        // contention cannot leave a visible name without its matching count.
        let mut child_object = self.object_mut(child)?;
        let entry = entries
            .insert(DirEntry { name: *name, child })
            .map_err(|_| FsError::OutOfEdges)?;
        // The edge exists now, so the child has one more name.
        child_object.incoming_inc();
        Ok(EdgeId { parent, entry })
    }

    /// Remove exactly one containment edge and decrement the child's incoming
    /// count, regardless of whether a directory child is empty.
    ///
    /// This is the native graph operation. The caller invokes reclamation only
    /// if that exact edge made the child unreachable; another containment or
    /// authority edge keeps the full subgraph live.
    pub(crate) fn detach(&self, parent: ObjectId, name: &Name) -> FsResult<(EdgeId, ObjectId)> {
        let entries = self.dir_entries_mut(parent)?;
        let (entry, child) = entries
            .iter()
            .find(|(_, candidate)| candidate.name == *name)
            .map(|(key, candidate)| (key, candidate.child))
            .ok_or(FsError::NotFound)?;
        let edge = EdgeId { parent, entry };
        let mut child_object = self.object_mut(child)?;
        entries
            .try_remove(edge.entry)
            .map_err(|error| fs_access(error, FsError::AccessLost))?;
        child_object.incoming_dec();
        Ok((edge, child))
    }

    /// POSIX-style `unlink`/`rmdir` compatibility: directory children must be
    /// empty. Native callers that mean “remove this graph edge” use
    /// [`Self::detach`].
    pub(crate) fn unlink_empty(
        &self,
        parent: ObjectId,
        name: &Name,
    ) -> FsResult<(EdgeId, ObjectId)> {
        self.prepare_removal(parent, name, true)?.commit()
    }

    pub(crate) fn prepare_removal(
        &self,
        parent: ObjectId,
        name: &Name,
        require_empty: bool,
    ) -> FsResult<PreparedRemoval<'_, K>> {
        let (_, observed_child) = self.lookup(parent, name)?;
        let child_is_dir = self.kind(observed_child)?.is_dir();
        let claims = if child_is_dir {
            self.claim_directories(&[parent, observed_child])?
        } else {
            self.claim_directories(&[parent])?
        };
        let (entry, child) = claims
            .get(parent)?
            .iter()
            .find(|(_, candidate)| candidate.name == *name)
            .map(|(key, candidate)| (key, candidate.child))
            .ok_or(FsError::NotFound)?;
        if child != observed_child {
            return Err(FsError::Busy);
        }
        if require_empty && child_is_dir && !claims.get(child)?.is_empty() {
            return Err(FsError::NotEmpty);
        }
        let object = self.object_mut(child)?;
        Ok(PreparedRemoval {
            claims,
            edge: EdgeId { parent, entry },
            child,
            object,
        })
    }

    /// Change one edge's label, keeping the edge — and therefore every
    /// handle whose proof runs through it.
    ///
    /// Rename changes the edge label without changing the child object or
    /// retiring the route, so the entry slot and its generation are preserved.
    /// That is the visible difference
    /// from [`Self::relocate`], which does retire the source edge.
    ///
    /// Refuses when the destination name exists — replacement
    /// semantics wait until atomic reclamation is proven.
    pub(crate) fn rename(&self, parent: ObjectId, old: &Name, new: &Name) -> FsResult<EdgeId> {
        self.prepare_rename(parent, old, new)?.commit()
    }

    pub(crate) fn prepare_rename(
        &self,
        parent: ObjectId,
        old: &Name,
        new: &Name,
    ) -> FsResult<PreparedRename<'_>> {
        let entries = self.dir_entries_mut(parent)?;
        let (entry, child) = entries
            .iter()
            .find(|(_, candidate)| candidate.name == *old)
            .map(|(key, candidate)| (key, candidate.child))
            .ok_or(FsError::NotFound)?;
        let edge = EdgeId { parent, entry };
        if old != new && entries.iter().any(|(_, candidate)| candidate.name == *new) {
            return Err(FsError::AlreadyExists);
        }
        Ok(PreparedRename {
            entries,
            edge,
            child,
            new: *new,
        })
    }

    /// Atomically remove `src/name` and create `dst/new_name` for the same
    /// object.
    ///
    /// The source edge is retired, so a handle opened through it fails closed
    /// on its next operation; a handle opened through a *different* link to
    /// the same file keeps working. Everything that can refuse
    /// is checked before anything is mutated, so a refused move leaves both
    /// directories exactly as they were.
    pub(crate) fn relocate(
        &self,
        src: ObjectId,
        name: &Name,
        dst: ObjectId,
        new_name: &Name,
    ) -> FsResult<(EdgeId, EdgeId)> {
        self.prepare_relocate(src, name, dst, new_name)?.commit()
    }

    pub(crate) fn prepare_relocate(
        &self,
        src: ObjectId,
        name: &Name,
        dst: ObjectId,
        new_name: &Name,
    ) -> FsResult<PreparedRelocate<'_>> {
        if !self.kind(dst)?.is_dir() {
            return Err(FsError::NotDirectory);
        }
        let (observed_src_edge, observed_child) = self.lookup(src, name)?;
        let child_is_dir = self.kind(observed_child)?.is_dir();
        let claims = if child_is_dir {
            self.claim_subtree_with(observed_child, &[src, dst])?
        } else {
            self.claim_directories(&[src, dst])?
        };
        let (src_entry, child) = claims
            .get(src)?
            .iter()
            .find(|(_, candidate)| candidate.name == *name)
            .map(|(key, candidate)| (key, candidate.child))
            .ok_or(FsError::NotFound)?;
        let src_edge = EdgeId {
            parent: src,
            entry: src_entry,
        };
        if src_edge != observed_src_edge || child != observed_child {
            return Err(FsError::Busy);
        }

        // Same directory, same name: nothing would change, and doing the
        // remove/insert anyway would needlessly retire live handles.
        if src == dst && name == new_name {
            return Ok(PreparedRelocate {
                claims,
                source: src_edge,
                child,
                destination: dst,
                name: *new_name,
                unchanged: true,
            });
        }

        // The destination name must be free.  When moving within one
        // directory the source edge is about to vacate, so only a *different*
        // name can collide.
        if claims
            .get(dst)?
            .iter()
            .any(|(_, candidate)| candidate.name == *new_name)
        {
            return Err(FsError::AlreadyExists);
        }

        // Moving a directory into its own subtree would orphan a cycle.
        if child_is_dir && self.is_reachable_claimed(&claims, child, dst)? {
            return Err(FsError::Cycle);
        }

        // Reserve destination room before removing the source edge, so the
        // insert below cannot fail and leave the object nameless.  Moving
        // within one directory reuses the vacated slot, so no room is needed.
        if src != dst && claims.get(dst)?.is_full() {
            return Err(FsError::OutOfEdges);
        }

        Ok(PreparedRelocate {
            claims,
            source: src_edge,
            child,
            destination: dst,
            name: *new_name,
            unchanged: false,
        })
    }

    /// One page of `parent`'s entries starting at slot `cursor`, plus the
    /// cursor to resume at (or `None` when the listing is complete).
    pub(crate) fn list(
        &self,
        parent: ObjectId,
        cursor: u32,
    ) -> FsResult<(heapless::Vec<ListEntry, MAX_LIST_PAGE>, Option<u32>)> {
        let entries = self.dir_entries(parent)?;
        let mut page: heapless::Vec<ListEntry, MAX_LIST_PAGE> = heapless::Vec::new();
        let mut next = None;
        for (key, entry) in entries.iter_from(cursor) {
            if page.is_full() {
                // Resume *at* the entry that did not fit — never past it.
                next = Some(key.index());
                break;
            }
            let (kind, subtype) = {
                let object = self.object(entry.child)?;
                (object.kind(), object.subtype())
            };
            let pushed = page.push(ListEntry {
                name: entry.name,
                object: entry.child,
                kind,
                subtype,
            });
            debug_assert!(pushed.is_ok(), "checked is_full above");
            if pushed.is_err() {
                next = Some(key.index());
                break;
            }
        }
        Ok((page, next))
    }

    // ── reclamation ─────────────────────────────────────────────────────────

    /// Try to release one exact object, returning children that became
    /// reclaimable when its containment edges disappeared.
    ///
    /// Called after an operation drops the last route to an object.  A
    /// directory becoming unreachable takes its whole subtree with it —
    /// `unlink` refuses non-empty directories, but retiring a *grant* or a
    /// private root can legitimately strand a populated subtree, and leaving
    /// it in the arenas would leak every object under it.
    ///
    /// Every payload needed for this one commit is claimed before the first
    /// incoming count changes. Contention therefore returns [`FsError::Busy`]
    /// with the object intact; the engine retains the work item and retries it
    /// cooperatively instead of silently leaking a skipped slot.
    pub(crate) fn reclaim_one(
        &self,
        id: ObjectId,
        placements: &mut BoundedVec<K, ServiceAlloc>,
        objects: &mut BoundedVec<ObjectId, ServiceAlloc>,
    ) -> FsResult<heapless::Vec<ObjectId, MAX_DIR_ENTRIES>> {
        let object = match self.objects.try_write(id.key()) {
            Ok(object) => object,
            Err(crate::slot::SlotAccess::Stale) => return Ok(heapless::Vec::new()),
            Err(crate::slot::SlotAccess::Busy) => return Err(FsError::Busy),
        };
        if !object.is_reclaimable() {
            return Ok(heapless::Vec::new());
        }
        let needed = usize::from(object.storage().is_some())
            + usize::from(object.staged_storage().is_some());
        if objects.is_full() || placements.capacity() - placements.len() < needed {
            return Err(FsError::OutOfObjects);
        }

        let storage = object.storage();
        let staged = object.staged_storage();
        let mut directory = match object.dir() {
            Some(dir) => Some(
                self.dirs
                    .try_write(dir.0)
                    .map_err(|error| fs_access(error, FsError::AccessLost))?,
            ),
            None => None,
        };

        let mut child_ids = heapless::Vec::<ObjectId, MAX_DIR_ENTRIES>::new();
        if let Some(entries) = directory.as_ref() {
            for (_, entry) in entries.iter() {
                if !child_ids.contains(&entry.child) {
                    child_ids
                        .push(entry.child)
                        .map_err(|_| FsError::OutOfObjects)?;
                }
            }
        }
        child_ids.as_mut_slice().sort_unstable_by_key(|id| id.key());

        let mut children = heapless::Vec::<(ObjectId, ObjectWrite<'_, K>), MAX_DIR_ENTRIES>::new();
        for child in child_ids {
            let claimed = self
                .objects
                .try_write(child.key())
                .map_err(|error| fs_access(error, FsError::AccessLost))?;
            children
                .push((child, claimed))
                .map_err(|_| FsError::OutOfObjects)?;
        }

        // All-or-nothing commit: decrement once per removed edge, then record
        // only the child generations that crossed to zero reachability.
        if let Some(entries) = directory.as_ref() {
            for (_, entry) in entries.iter() {
                let (_, child) = children
                    .iter_mut()
                    .find(|(id, _)| *id == entry.child)
                    .ok_or(FsError::AccessLost)?;
                child.incoming_dec();
            }
        }
        let mut became_reclaimable = heapless::Vec::new();
        for (id, child) in children.iter() {
            if child.is_reclaimable() {
                became_reclaimable
                    .push(*id)
                    .map_err(|_| FsError::OutOfObjects)?;
            }
        }

        if let Some(entries) = directory.take() {
            entries.remove().map_err(|_| FsError::AccessLost)?;
        }
        if let Some(key) = storage {
            placements.push(key).map_err(|_| FsError::OutOfObjects)?;
        }
        if let Some(key) = staged {
            placements.push(key).map_err(|_| FsError::OutOfObjects)?;
        }
        object.remove().map_err(|_| FsError::AccessLost)?;
        self.snapshots[id.key().index() as usize].clear(id);
        objects.push(id).map_err(|_| FsError::OutOfObjects)?;
        Ok(became_reclaimable)
    }

    #[cfg(test)]
    pub(crate) fn reclaim(
        &self,
        start: ObjectId,
        placements: &mut BoundedVec<K, ServiceAlloc>,
        objects: &mut BoundedVec<ObjectId, ServiceAlloc>,
    ) -> usize {
        let before = objects.len();
        // The caller's batches are already arena-backed; so is this, for the
        // same reason. An allocation failure here takes the route a full
        // worklist takes below: a counted, reportable abandonment, never a
        // panic (see the contract on `Engine::reclaim`).
        let Ok(mut worklist) =
            BoundedVec::<ObjectId, ServiceAlloc>::try_new_in(MAX_OBJECTS, crate::service_alloc())
        else {
            self.note_abandoned_reclaim();
            return 0;
        };
        worklist
            .push(start)
            .expect("one root fits the object bound");
        while let Some(id) = worklist.pop() {
            let children = self
                .reclaim_one(id, placements, objects)
                .expect("single-threaded reclaim fixture cannot contend");
            for child in children {
                worklist
                    .push(child)
                    .expect("one queue entry per reclaimed object");
            }
        }
        objects.len() - before
    }

    /// Record that the provider refused to release a placement.
    pub(crate) fn note_leaked_placement(&self) {
        let _ =
            self.leaked_placements
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_add(1))
                });
    }

    /// Hold a write claim on one object, so a test can create the exact
    /// contention a concurrent operation creates.
    ///
    /// Ordinary `pub(crate)` API for the same reason
    /// [`RamProvider::inject`](crate::provider::ram::RamProvider::inject) is
    /// ordinary API: the alternative is a racy test that only sometimes
    /// exercises the path it names, and contention is precisely the axis this
    /// service was nondeterministic along.
    pub(crate) fn claim_object(&self, id: ObjectId) -> Option<ObjectWrite<'_, K>> {
        self.objects.try_write(id.key()).ok()
    }

    /// Record that a reclamation walk stopped before it reached the whole
    /// unreachable subtree.
    pub(crate) fn note_abandoned_reclaim(&self) {
        let _ =
            self.abandoned_reclaims
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_add(1))
                });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The namespace under test uses a `u32` placement key: these tests are
    /// about the DAG, and the key's only job here is to be carried.
    type Ns = Namespace<u32>;

    fn name(s: &str) -> Name {
        Name::new(s).expect("test names are valid")
    }

    /// The namespace must not spend the `serve` future's stack on state that
    /// scales with the volume.
    ///
    /// Before the arenas moved to the heap this type was **130,944 bytes**,
    /// which is why the object bound and the 4 MiB shared stack were one
    /// number and why an 8-byte growth in an unrelated record overflowed that
    /// stack on metal (PR #958). The ceiling here is deliberately loose: it
    /// pins the *property* (handles and counters only, no arena) rather than
    /// an exact figure, so adding a counter is free and re-inlining an arena
    /// fails.
    #[test]
    fn the_namespace_holds_no_volume_scaled_arena_inline() {
        let bytes = core::mem::size_of::<Ns>();
        assert!(
            bytes <= 1024,
            "namespace grew to {bytes} B: a volume-scaled arena is inline again"
        );
    }

    /// The commit-path snapshot must not put the volume on the stack either.
    ///
    /// Inline this was **119,832 bytes**, its edge table alone 114,688, and
    /// unlike the decoded image it is built on *every commit* rather than at
    /// mount. Pins the property, not a figure.
    #[test]
    fn the_durable_snapshot_holds_no_table_inline() {
        let bytes = core::mem::size_of::<DurableGraph<u32>>();
        assert!(
            bytes <= 512,
            "DurableGraph grew to {bytes} B: a volume-scaled table is inline again"
        );
    }

    /// A root directory plus the namespace holding it.
    fn ns_with_root() -> (Ns, ObjectId) {
        let mut ns = Ns::new();
        let root = ns.create_dir_object().unwrap();
        // A root is reached by an authority edge, not a name; the engine
        // marks that with a grant, which these tests stand in for.
        ns.object_mut(root).unwrap().grants_inc();
        (ns, root)
    }

    // ── names ───────────────────────────────────────────────────────────────

    #[test]
    fn valid_names_round_trip() {
        for s in ["a", "main.rs", "hello-again", "Ünïcødé"] {
            let n = name(s);
            assert_eq!(n.as_str(), s);
            assert_eq!(n.as_bytes(), s.as_bytes());
            assert_eq!(n.len(), s.len());
            assert!(!n.is_empty());
        }
    }

    #[test]
    fn empty_and_oversized_names_are_refused() {
        assert_eq!(Name::new("").unwrap_err(), FsError::InvalidName);
        let too_long = "x".repeat(crate::bounds::MAX_NAME_LEN + 1);
        assert_eq!(Name::new(&too_long).unwrap_err(), FsError::InvalidName);
        // Exactly at the bound is legal — the refusal must be off-by-none.
        assert!(Name::new(&"x".repeat(crate::bounds::MAX_NAME_LEN)).is_ok());
    }

    #[test]
    fn separator_nul_and_dot_names_are_refused() {
        for bad in ["a/b", "/", "a\0b", ".", ".."] {
            assert_eq!(
                Name::new(bad).unwrap_err(),
                FsError::InvalidName,
                "{bad:?} must be refused"
            );
        }
        // A name merely *containing* dots is fine; only the two reserved
        // components are refused.
        assert!(Name::new("...").is_ok());
        assert!(Name::new(".hidden").is_ok());
    }

    #[test]
    fn invalid_utf8_names_are_refused() {
        assert_eq!(
            Name::from_bytes(&[0xff, 0xfe]).unwrap_err(),
            FsError::InvalidName
        );
    }

    #[test]
    fn names_compare_by_content_not_by_padding() {
        // Two names built from different-length inputs must not compare
        // equal just because their tails are both zero-padded.
        assert_eq!(name("ab"), name("ab"));
        assert_ne!(name("ab"), name("abc"));
    }

    // ── creation and lookup ─────────────────────────────────────────────────

    #[test]
    fn create_and_look_up_a_file() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &name("hello"), file).unwrap();
        let (found_edge, found) = ns.lookup(root, &name("hello")).unwrap();
        assert_eq!(found, file);
        assert_eq!(found_edge, edge);
        assert_eq!(ns.edge_child(edge).unwrap(), file);
        assert_eq!(ns.object(file).unwrap().incoming(), 1);
    }

    #[test]
    fn lookup_of_an_absent_name_is_not_found() {
        let (ns, root) = ns_with_root();
        assert_eq!(
            ns.lookup(root, &name("nope")).unwrap_err(),
            FsError::NotFound
        );
    }

    #[test]
    fn lookup_beneath_a_file_is_not_a_directory() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        assert_eq!(
            ns.lookup(file, &name("x")).unwrap_err(),
            FsError::NotDirectory
        );
    }

    #[test]
    fn duplicate_names_are_refused() {
        let (mut ns, root) = ns_with_root();
        let a = ns.create_file_object().unwrap();
        let b = ns.create_file_object().unwrap();
        ns.link(root, &name("dup"), a).unwrap();
        assert_eq!(
            ns.link(root, &name("dup"), b).unwrap_err(),
            FsError::AlreadyExists
        );
        // The refused insert left nothing behind.
        assert_eq!(ns.object(b).unwrap().incoming(), 0);
    }

    // ── links ───────────────────────────────────────────────────────────────

    #[test]
    fn one_file_can_carry_two_names() {
        let (mut ns, root) = ns_with_root();
        let projects = ns.create_dir_object().unwrap();
        ns.link(root, &name("projects"), projects).unwrap();
        let file = ns.create_file_object().unwrap();
        ns.link(projects, &name("main.rs"), file).unwrap();
        ns.link(root, &name("notes.rs"), file).unwrap();

        assert_eq!(ns.object(file).unwrap().incoming(), 2);
        assert_eq!(ns.lookup(projects, &name("main.rs")).unwrap().1, file);
        assert_eq!(ns.lookup(root, &name("notes.rs")).unwrap().1, file);
    }

    #[test]
    fn unlinking_one_name_leaves_the_other() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("one"), file).unwrap();
        ns.link(root, &name("two"), file).unwrap();

        let (_, child) = ns.detach(root, &name("one")).unwrap();
        assert_eq!(child, file);
        assert_eq!(ns.object(file).unwrap().incoming(), 1);
        assert_eq!(
            ns.lookup(root, &name("one")).unwrap_err(),
            FsError::NotFound
        );
        assert_eq!(ns.lookup(root, &name("two")).unwrap().1, file);
    }

    #[test]
    fn directory_diamonds_are_lawful_and_count_each_edge() {
        let (mut ns, root) = ns_with_root();
        let left = ns.create_dir_object().unwrap();
        let right = ns.create_dir_object().unwrap();
        let dir = ns.create_dir_object().unwrap();
        ns.link(root, &name("left"), left).unwrap();
        ns.link(root, &name("right"), right).unwrap();
        let first = ns.link(left, &name("shared"), dir).unwrap();
        let second = ns.link(right, &name("shared"), dir).unwrap();

        assert_eq!(ns.object(dir).unwrap().incoming(), 2);
        assert_eq!(ns.edge_child(first).unwrap(), dir);
        assert_eq!(ns.edge_child(second).unwrap(), dir);
        let reachable = ns.collect_reachable(root).unwrap();
        assert_eq!(reachable.iter().filter(|id| **id == dir).count(), 1);
    }

    #[test]
    fn dense_convergence_spends_worklist_capacity_once_per_object() {
        let (mut ns, root) = ns_with_root();
        let mut dirs = [ObjectId::INVALID; MAX_DIRECTORIES - 1];
        for (index, dir) in dirs.iter_mut().enumerate() {
            *dir = ns.create_dir_object().unwrap();
            ns.link(root, &name(&std::format!("d{index}")), *dir)
                .unwrap();
        }
        // Every higher-numbered directory also reaches every lower-numbered
        // one. This is acyclic but highly convergent: an implementation that
        // budgets queue entries rather than unique objects exhausts despite
        // the graph fitting exactly within the directory arena.
        for high in (1..dirs.len()).rev() {
            for low in 0..high {
                ns.link(dirs[high], &name(&std::format!("to-{low}")), dirs[low])
                    .unwrap();
            }
        }

        let reachable = ns.collect_reachable(root).unwrap();
        assert_eq!(reachable.len(), MAX_DIRECTORIES);
        for dir in dirs {
            assert_eq!(reachable.iter().filter(|id| **id == dir).count(), 1);
        }
    }

    /// A diamond is lawful, a back edge across it is not, and the proof
    /// visits the shared directory once whichever route reaches it first.
    #[test]
    fn back_edge_across_a_diamond_is_refused_and_a_forward_link_is_lawful() {
        let (mut ns, root) = ns_with_root();
        let left = ns.create_dir_object().unwrap();
        let right = ns.create_dir_object().unwrap();
        let shared = ns.create_dir_object().unwrap();
        let deeper = ns.create_dir_object().unwrap();
        ns.link(root, &name("left"), left).unwrap();
        ns.link(root, &name("right"), right).unwrap();
        ns.link(left, &name("shared"), shared).unwrap();
        ns.link(right, &name("shared"), shared).unwrap();
        ns.link(shared, &name("deeper"), deeper).unwrap();

        // `left` is reachable from `deeper` only through the diamond's
        // second route — the proof must follow it, not stop at the first
        // visit of `shared`.
        assert_eq!(
            ns.link(deeper, &name("up-left"), left).unwrap_err(),
            FsError::Cycle
        );
        assert_eq!(
            ns.link(deeper, &name("up-root"), root).unwrap_err(),
            FsError::Cycle
        );
        // A second name for the shared base from the root is a lawful
        // forward projection.
        ns.link(root, &name("shared2"), shared).unwrap();
        assert_eq!(ns.object(shared).unwrap().incoming(), 3);
        assert_eq!(ns.object(left).unwrap().incoming(), 1);
        assert_eq!(ns.lookup(deeper, &name("up-left")), Err(FsError::NotFound));
    }

    /// Moving a directory beneath its own descendant is a cycle even when
    /// the descendant is reached through two routes.
    #[test]
    fn subtree_move_into_its_own_descendant_through_a_diamond_is_refused() {
        let (mut ns, root) = ns_with_root();
        let outer = ns.create_dir_object().unwrap();
        let a = ns.create_dir_object().unwrap();
        let b = ns.create_dir_object().unwrap();
        let deeper = ns.create_dir_object().unwrap();
        ns.link(root, &name("outer"), outer).unwrap();
        ns.link(outer, &name("a"), a).unwrap();
        ns.link(outer, &name("b"), b).unwrap();
        ns.link(a, &name("deeper"), deeper).unwrap();
        ns.link(b, &name("deeper"), deeper).unwrap();

        assert_eq!(
            ns.relocate(root, &name("outer"), deeper, &name("moved"))
                .unwrap_err(),
            FsError::Cycle
        );
        assert_eq!(ns.lookup(root, &name("outer")).unwrap().1, outer);
        assert_eq!(ns.lookup(deeper, &name("moved")), Err(FsError::NotFound));
        assert_eq!(ns.object(outer).unwrap().incoming(), 1);
    }

    /// The full arena is a lawful footprint: a walk keyed by directory
    /// slot never runs out of room before the arena does.
    #[test]
    fn dense_convergent_footprint_fits_the_directory_bound_exactly() {
        let (mut ns, root) = ns_with_root();
        let mut dirs = [ObjectId::INVALID; MAX_DIRECTORIES - 1];
        for (index, dir) in dirs.iter_mut().enumerate() {
            *dir = ns.create_dir_object().unwrap();
            ns.link(root, &name(&std::format!("d{index}")), *dir)
                .unwrap();
        }
        for high in (1..dirs.len()).rev() {
            for low in 0..high {
                ns.link(dirs[high], &name(&std::format!("to-{low}")), dirs[low])
                    .unwrap();
            }
        }
        // One more forward edge across the whole footprint is lawful; the
        // matching back edge is a cycle. Neither may report `Range`.
        let last = dirs[dirs.len() - 1];
        let first = dirs[0];
        ns.link(last, &name("forward"), first).unwrap();
        assert_eq!(
            ns.link(first, &name("back"), last).unwrap_err(),
            FsError::Cycle
        );
    }

    /// A child directory that entered the footprint after it was observed
    /// and before it was claimed is reported as contention, never walked.
    #[test]
    fn reachability_reports_busy_when_a_child_directory_escapes_the_claimed_footprint() {
        let (ns, root) = ns_with_root();
        let a = ns.create_dir_object().unwrap();
        let stray = ns.create_dir_object().unwrap();
        ns.link(root, &name("a"), a).unwrap();
        ns.link(a, &name("stray"), stray).unwrap();

        // Claim only `a`: `stray` is a directory child outside the claim.
        let claims = ns.claim_directories(&[a]).unwrap();
        assert_eq!(
            ns.is_reachable_claimed(&claims, a, root),
            Err(FsError::Busy)
        );
    }

    #[test]
    fn directory_self_links_and_back_edges_are_refused_without_mutation() {
        let (mut ns, root) = ns_with_root();
        let outer = ns.create_dir_object().unwrap();
        let inner = ns.create_dir_object().unwrap();
        ns.link(root, &name("outer"), outer).unwrap();
        ns.link(outer, &name("inner"), inner).unwrap();

        assert_eq!(
            ns.link(inner, &name("back"), outer).unwrap_err(),
            FsError::Cycle
        );
        assert_eq!(
            ns.link(inner, &name("self"), inner).unwrap_err(),
            FsError::Cycle
        );
        assert_eq!(ns.object(outer).unwrap().incoming(), 1);
        assert_eq!(ns.object(inner).unwrap().incoming(), 1);
        assert_eq!(ns.lookup(inner, &name("back")), Err(FsError::NotFound));
    }

    #[test]
    fn removing_an_edge_retires_only_that_edge() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let first = ns.link(root, &name("one"), file).unwrap();
        let second = ns.link(root, &name("two"), file).unwrap();

        ns.detach(root, &name("one")).unwrap();
        // The removed edge fails closed; its sibling is untouched.
        assert_eq!(ns.edge_child(first).unwrap_err(), FsError::AccessLost);
        assert_eq!(ns.edge_child(second).unwrap(), file);
    }

    // ── rename ──────────────────────────────────────────────────────────────

    #[test]
    fn rename_changes_the_label_and_keeps_the_edge() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &name("before"), file).unwrap();

        let renamed = ns.rename(root, &name("before"), &name("after")).unwrap();
        // Same edge identity: a handle proving through it stays valid.
        assert_eq!(renamed, edge);
        assert_eq!(ns.edge_child(edge).unwrap(), file);
        assert_eq!(
            ns.lookup(root, &name("before")).unwrap_err(),
            FsError::NotFound
        );
        assert_eq!(ns.lookup(root, &name("after")).unwrap().1, file);
        assert_eq!(ns.object(file).unwrap().incoming(), 1);
    }

    /// One claimed directory is not a namespace-wide mutex. A mutation in a
    /// disjoint directory completes while the exact conflicting directory
    /// reports retryable backpressure, with neither operation changing the
    /// other's entries.
    #[test]
    fn disjoint_directory_mutations_do_not_serialize_globally() {
        let (mut ns, root) = ns_with_root();
        let left = ns.create_dir_object().unwrap();
        let right = ns.create_dir_object().unwrap();
        ns.link(root, &name("left"), left).unwrap();
        ns.link(root, &name("right"), right).unwrap();
        let left_file = ns.create_file_object().unwrap();
        let right_file = ns.create_file_object().unwrap();
        ns.link(left, &name("old"), left_file).unwrap();
        ns.link(right, &name("old"), right_file).unwrap();

        let left_claim = ns.dir_entries_mut(left).unwrap();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                assert_eq!(
                    ns.rename(left, &name("old"), &name("blocked")),
                    Err(FsError::Busy)
                );
                ns.rename(right, &name("old"), &name("new")).unwrap();
            });
            worker.join().unwrap();
        });
        drop(left_claim);

        assert_eq!(ns.lookup(left, &name("old")).unwrap().1, left_file);
        assert_eq!(ns.lookup(left, &name("blocked")), Err(FsError::NotFound));
        assert_eq!(ns.lookup(right, &name("new")).unwrap().1, right_file);
    }

    #[test]
    fn durable_graph_refuses_contention_then_copies_one_complete_footprint() {
        let (ns, root) = ns_with_root();
        let dir = ns.create_dir_object().unwrap();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("left"), dir).unwrap();
        ns.link(root, &name("right"), dir).unwrap();
        ns.link(dir, &name("file"), file).unwrap();

        let held = ns.object_mut(file).unwrap();
        assert!(matches!(ns.durable_graph(root, 16), Err(FsError::Busy)));
        drop(held);

        let graph = ns.durable_graph(root, 16).unwrap();
        assert_eq!(
            graph.objects.len(),
            3,
            "the diamond copies each object once"
        );
        assert_eq!(graph.edges.len(), 3, "each named edge remains distinct");
        assert_eq!(graph.root, ns.object(root).unwrap().stable());
    }

    #[test]
    fn rename_onto_an_existing_name_is_refused() {
        let (mut ns, root) = ns_with_root();
        let a = ns.create_file_object().unwrap();
        let b = ns.create_file_object().unwrap();
        ns.link(root, &name("a"), a).unwrap();
        ns.link(root, &name("b"), b).unwrap();
        assert_eq!(
            ns.rename(root, &name("a"), &name("b")).unwrap_err(),
            FsError::AlreadyExists
        );
        // Neither name moved.
        assert_eq!(ns.lookup(root, &name("a")).unwrap().1, a);
        assert_eq!(ns.lookup(root, &name("b")).unwrap().1, b);
    }

    #[test]
    fn rename_to_the_same_name_is_permitted() {
        let (mut ns, root) = ns_with_root();
        let a = ns.create_file_object().unwrap();
        ns.link(root, &name("a"), a).unwrap();
        // Self-rename must not trip the AlreadyExists check against itself.
        assert!(ns.rename(root, &name("a"), &name("a")).is_ok());
        assert_eq!(ns.lookup(root, &name("a")).unwrap().1, a);
    }

    // ── move ────────────────────────────────────────────────────────────────

    #[test]
    fn move_retires_the_source_edge_and_keeps_the_object() {
        let (mut ns, root) = ns_with_root();
        let dst = ns.create_dir_object().unwrap();
        ns.link(root, &name("dst"), dst).unwrap();
        let file = ns.create_file_object().unwrap();
        let src_edge = ns.link(root, &name("f"), file).unwrap();

        let (retired, created) = ns.relocate(root, &name("f"), dst, &name("moved")).unwrap();
        assert_eq!(retired, src_edge);
        // Source edge fails closed; the object itself is untouched.
        assert_eq!(ns.edge_child(src_edge).unwrap_err(), FsError::AccessLost);
        assert_eq!(ns.edge_child(created).unwrap(), file);
        assert_eq!(ns.object(file).unwrap().incoming(), 1);
        assert_eq!(ns.lookup(root, &name("f")).unwrap_err(), FsError::NotFound);
        assert_eq!(ns.lookup(dst, &name("moved")).unwrap().1, file);
    }

    #[test]
    fn contended_destination_cannot_partially_remove_a_move_source() {
        let (ns, root) = ns_with_root();
        let dst = ns.create_dir_object().unwrap();
        ns.link(root, &name("dst"), dst).unwrap();
        let file = ns.create_file_object().unwrap();
        let source = ns.link(root, &name("f"), file).unwrap();

        let held_destination = ns.dir_entries_mut(dst).unwrap();
        assert_eq!(
            ns.relocate(root, &name("f"), dst, &name("moved")),
            Err(FsError::Busy)
        );
        assert_eq!(ns.edge_child(source).unwrap(), file);
        assert_eq!(ns.lookup(root, &name("f")).unwrap().1, file);
        drop(held_destination);
        assert_eq!(ns.lookup(dst, &name("moved")), Err(FsError::NotFound));
    }

    #[test]
    fn opposing_directory_links_cannot_commit_a_cycle() {
        let (ns, _root) = ns_with_root();
        let a = ns.create_dir_object().unwrap();
        let b = ns.create_dir_object().unwrap();
        let start = std::sync::Barrier::new(3);
        let (left, right) = std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                start.wait();
                ns.link(a, &name("b"), b)
            });
            let right = scope.spawn(|| {
                start.wait();
                ns.link(b, &name("a"), a)
            });
            start.wait();
            (left.join().unwrap(), right.join().unwrap())
        });

        // What must never happen is both directions committing. Both racers
        // *losing* is lawful: a claim the other side holds answers `Busy`
        // (bounded backpressure, the engine's contract), and a reader in the
        // footprint snapshot can refuse a writer just as a writer refuses a
        // reader, so the race may leave neither edge published.
        assert!(
            !(left.is_ok() && right.is_ok()),
            "both directions committed"
        );
        for outcome in [&left, &right] {
            if let Err(error) = outcome {
                assert!(
                    matches!(error, FsError::Busy | FsError::AccessLost | FsError::Cycle),
                    "a racer may only lose to contention or to the other's edge: {error:?}"
                );
            }
        }
        // Driven to a fixed point, exactly one direction stands and the other
        // is a cycle.
        let mut forward = left.map(|_| ());
        if forward.is_err() && right.is_err() {
            forward = ns.link(a, &name("b"), b).map(|_| ());
        }
        let refused = if forward.is_ok() {
            ns.link(b, &name("a"), a)
        } else {
            ns.link(a, &name("b"), b)
        };
        assert_eq!(refused, Err(FsError::Cycle));
    }

    #[test]
    fn move_leaves_a_parallel_link_working() {
        let (mut ns, root) = ns_with_root();
        let dst = ns.create_dir_object().unwrap();
        ns.link(root, &name("dst"), dst).unwrap();
        let file = ns.create_file_object().unwrap();
        let src_edge = ns.link(root, &name("f"), file).unwrap();
        let parallel = ns.link(root, &name("also-f"), file).unwrap();

        ns.relocate(root, &name("f"), dst, &name("moved")).unwrap();
        assert_eq!(ns.edge_child(src_edge).unwrap_err(), FsError::AccessLost);
        // The other route to the same object is unaffected.
        assert_eq!(ns.edge_child(parallel).unwrap(), file);
        assert_eq!(ns.object(file).unwrap().incoming(), 2);
    }

    #[test]
    fn move_within_one_directory_renames_by_retiring_the_edge() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &name("a"), file).unwrap();

        let (retired, created) = ns.relocate(root, &name("a"), root, &name("b")).unwrap();
        assert_eq!(retired, edge);
        assert_ne!(created, edge);
        assert_eq!(ns.edge_child(edge).unwrap_err(), FsError::AccessLost);
        assert_eq!(ns.lookup(root, &name("b")).unwrap().1, file);
        assert_eq!(ns.object(file).unwrap().incoming(), 1);
    }

    #[test]
    fn move_to_the_identical_place_is_a_no_op() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &name("a"), file).unwrap();
        let (retired, created) = ns.relocate(root, &name("a"), root, &name("a")).unwrap();
        // Nothing changed, so nothing was retired.
        assert_eq!(retired, edge);
        assert_eq!(created, edge);
        assert_eq!(ns.edge_child(edge).unwrap(), file);
    }

    #[test]
    fn move_onto_an_existing_name_is_refused_and_changes_nothing() {
        let (mut ns, root) = ns_with_root();
        let dst = ns.create_dir_object().unwrap();
        ns.link(root, &name("dst"), dst).unwrap();
        let a = ns.create_file_object().unwrap();
        let b = ns.create_file_object().unwrap();
        let a_edge = ns.link(root, &name("a"), a).unwrap();
        ns.link(dst, &name("taken"), b).unwrap();

        assert_eq!(
            ns.relocate(root, &name("a"), dst, &name("taken"))
                .unwrap_err(),
            FsError::AlreadyExists
        );
        // The source edge survived the refusal.
        assert_eq!(ns.edge_child(a_edge).unwrap(), a);
        assert_eq!(ns.lookup(root, &name("a")).unwrap().1, a);
        assert_eq!(ns.lookup(dst, &name("taken")).unwrap().1, b);
    }

    #[test]
    fn moving_a_directory_into_itself_is_a_cycle() {
        let (mut ns, root) = ns_with_root();
        let dir = ns.create_dir_object().unwrap();
        ns.link(root, &name("d"), dir).unwrap();
        assert_eq!(
            ns.relocate(root, &name("d"), dir, &name("self"))
                .unwrap_err(),
            FsError::Cycle
        );
    }

    #[test]
    fn moving_a_directory_beneath_its_own_descendant_is_a_cycle() {
        let (mut ns, root) = ns_with_root();
        let outer = ns.create_dir_object().unwrap();
        let inner = ns.create_dir_object().unwrap();
        let deeper = ns.create_dir_object().unwrap();
        ns.link(root, &name("outer"), outer).unwrap();
        ns.link(outer, &name("inner"), inner).unwrap();
        ns.link(inner, &name("deeper"), deeper).unwrap();

        assert_eq!(
            ns.relocate(root, &name("outer"), deeper, &name("loop"))
                .unwrap_err(),
            FsError::Cycle
        );
        // The tree is intact after the refusal.
        assert_eq!(ns.lookup(root, &name("outer")).unwrap().1, outer);
        assert_eq!(ns.lookup(inner, &name("deeper")).unwrap().1, deeper);
    }

    #[test]
    fn moving_a_directory_to_an_unrelated_place_is_permitted() {
        let (mut ns, root) = ns_with_root();
        let a = ns.create_dir_object().unwrap();
        let b = ns.create_dir_object().unwrap();
        let child = ns.create_dir_object().unwrap();
        ns.link(root, &name("a"), a).unwrap();
        ns.link(root, &name("b"), b).unwrap();
        ns.link(a, &name("child"), child).unwrap();

        // `b` is not under `child`, so this is a legal move.
        assert!(ns.relocate(a, &name("child"), b, &name("child")).is_ok());
        assert_eq!(ns.lookup(b, &name("child")).unwrap().1, child);
        assert_eq!(ns.object(child).unwrap().incoming(), 1);
    }

    #[test]
    fn move_into_a_file_is_refused() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let victim = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        ns.link(root, &name("v"), victim).unwrap();
        assert_eq!(
            ns.relocate(root, &name("v"), file, &name("x")).unwrap_err(),
            FsError::NotDirectory
        );
    }

    // ── unlink and directories ──────────────────────────────────────────────

    #[test]
    fn unlinking_a_non_empty_directory_is_refused() {
        let (mut ns, root) = ns_with_root();
        let dir = ns.create_dir_object().unwrap();
        ns.link(root, &name("d"), dir).unwrap();
        let file = ns.create_file_object().unwrap();
        ns.link(dir, &name("f"), file).unwrap();

        assert_eq!(
            ns.unlink_empty(root, &name("d")).unwrap_err(),
            FsError::NotEmpty
        );
        assert_eq!(ns.lookup(root, &name("d")).unwrap().1, dir);

        // Emptying it first makes the unlink succeed.
        ns.detach(dir, &name("f")).unwrap();
        assert!(ns.unlink_empty(root, &name("d")).is_ok());
    }

    #[test]
    fn unlink_rechecks_directory_emptiness_under_the_child_claim() {
        let (ns, root) = ns_with_root();
        let dir = ns.create_dir_object().unwrap();
        let edge = ns.link(root, &name("d"), dir).unwrap();

        let held_child = ns.dir_entries_mut(dir).unwrap();
        assert_eq!(ns.unlink_empty(root, &name("d")), Err(FsError::Busy));
        assert_eq!(ns.edge_child(edge).unwrap(), dir);
        drop(held_child);
        assert!(ns.unlink_empty(root, &name("d")).is_ok());
    }

    // ── listing ─────────────────────────────────────────────────────────────

    #[test]
    fn listing_pages_without_losing_or_repeating_entries() {
        let (mut ns, root) = ns_with_root();
        let total = MAX_LIST_PAGE + 3;
        let mut created = std::vec::Vec::new();
        for i in 0..total {
            let file = ns.create_file_object().unwrap();
            let mut buf = [0u8; 8];
            let n = write_index(&mut buf, i);
            ns.link(root, &Name::from_bytes(&buf[..n]).unwrap(), file)
                .unwrap();
            created.push(file);
        }

        let mut seen = std::vec::Vec::new();
        let mut cursor = 0u32;
        loop {
            let (page, next) = ns.list(root, cursor).unwrap();
            assert!(page.len() <= MAX_LIST_PAGE);
            for entry in page.iter() {
                seen.push(entry.object);
            }
            match next {
                Some(c) => {
                    // The cursor must advance, or paging would not terminate.
                    assert!(c > cursor || cursor == 0);
                    cursor = c;
                }
                None => break,
            }
        }
        assert_eq!(seen.len(), total);
        for object in &created {
            assert!(seen.contains(object));
        }
    }

    #[test]
    fn listing_an_empty_directory_yields_one_empty_page() {
        let (ns, root) = ns_with_root();
        let (page, next) = ns.list(root, 0).unwrap();
        assert!(page.is_empty());
        assert_eq!(next, None);
    }

    #[test]
    fn listing_a_file_is_refused() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        assert_eq!(ns.list(file, 0).unwrap_err(), FsError::NotDirectory);
    }

    #[test]
    fn listing_reports_each_entry_kind() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        let dir = ns.create_dir_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        ns.link(root, &name("d"), dir).unwrap();
        let (page, _) = ns.list(root, 0).unwrap();
        let kinds: std::vec::Vec<_> = page.iter().map(|e| (e.name.as_str(), e.kind)).collect();
        assert!(kinds.contains(&("f", ObjectKind::File)));
        assert!(kinds.contains(&("d", ObjectKind::Directory)));
    }

    // ── exhaustion ──────────────────────────────────────────────────────────

    #[test]
    fn a_full_directory_refuses_more_edges() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        for i in 0..MAX_DIR_ENTRIES {
            let mut buf = [0u8; 8];
            let n = write_index(&mut buf, i);
            ns.link(root, &Name::from_bytes(&buf[..n]).unwrap(), file)
                .unwrap();
        }
        let extra = ns.create_file_object().unwrap();
        assert_eq!(
            ns.link(root, &name("overflow"), extra).unwrap_err(),
            FsError::OutOfEdges
        );
    }

    #[test]
    fn a_full_directory_arena_refuses_more_directories() {
        let mut ns = Ns::new();
        let mut made = 0usize;
        while ns.create_dir_object().is_ok() {
            made += 1;
            assert!(made <= MAX_DIRECTORIES, "arena must stop at its bound");
        }
        assert_eq!(made, MAX_DIRECTORIES);
        // The object arena is larger, so files still succeed — the refusal
        // was the directory arena's, and it did not consume object slots
        // on the way out.
        assert!(ns.create_file_object().is_ok());
    }

    // ── reclamation ─────────────────────────────────────────────────────────

    #[test]
    fn reclaim_releases_an_unreachable_file_and_its_placement() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        ns.object_mut(file).unwrap().set_storage(7);

        let (_, child) = ns.detach(root, &name("f")).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 1);
        assert_eq!(placements.as_slice(), &[7]);
        assert_eq!(objects.as_slice(), &[file]);
        assert!(!ns.exists(file));
    }

    #[test]
    fn contended_reclaim_keeps_exact_custody_for_retry() {
        let (ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        ns.object_mut(file).unwrap().set_storage(7);
        let (_, child) = ns.detach(root, &name("f")).unwrap();
        let held = ns.object_mut(file).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();

        assert!(matches!(
            ns.reclaim_one(child, &mut placements, &mut objects),
            Err(FsError::Busy)
        ));
        assert!(placements.is_empty() && objects.is_empty());
        assert!(ns.exists(file), "a busy object remains in its exact slot");

        drop(held);
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 1);
        assert_eq!(placements.as_slice(), &[7]);
        assert!(!ns.exists(file));
    }

    #[test]
    fn reclaim_spares_an_object_that_still_has_a_name() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("one"), file).unwrap();
        ns.link(root, &name("two"), file).unwrap();
        ns.object_mut(file).unwrap().set_storage(9);

        let (_, child) = ns.detach(root, &name("one")).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 0);
        assert!(placements.is_empty());
        assert!(ns.exists(file));
    }

    #[test]
    fn reclaim_spares_an_object_that_still_has_a_grant() {
        let (mut ns, root) = ns_with_root();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &name("f"), file).unwrap();
        ns.object_mut(file).unwrap().grants_inc();

        let (_, child) = ns.detach(root, &name("f")).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 0);
        assert!(ns.exists(file));

        // Dropping the grant too makes it collectable.
        ns.object_mut(file).unwrap().grants_dec();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 1);
        assert!(!ns.exists(file));
    }

    #[test]
    fn reclaiming_a_stranded_root_takes_its_whole_subtree() {
        // Retiring a private root can strand a populated subtree; leaving it
        // in the arenas would leak every object beneath it.
        let (mut ns, root) = ns_with_root();
        let dir = ns.create_dir_object().unwrap();
        ns.link(root, &name("d"), dir).unwrap();
        let a = ns.create_file_object().unwrap();
        let b = ns.create_file_object().unwrap();
        ns.link(dir, &name("a"), a).unwrap();
        ns.link(dir, &name("b"), b).unwrap();
        ns.object_mut(a).unwrap().set_storage(1);
        ns.object_mut(b).unwrap().set_storage(2);

        let objects_before = ns.object_count();
        let dirs_before = ns.directory_count();

        // Retire the root's only route.
        ns.object_mut(root).unwrap().grants_dec();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut reclaimed: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let released = ns.reclaim(root, &mut placements, &mut reclaimed);

        assert_eq!(released, 4, "root, dir, and both files");
        assert_eq!(objects_before - ns.object_count(), 4);
        assert_eq!(dirs_before - ns.directory_count(), 2);
        assert!(placements.contains(&1) && placements.contains(&2));
        assert_eq!(placements.len(), 2, "each placement released exactly once");
        assert!(!ns.exists(root) && !ns.exists(dir) && !ns.exists(a) && !ns.exists(b));
    }

    #[test]
    fn reclaiming_a_subtree_spares_a_file_linked_from_outside() {
        let (mut ns, root) = ns_with_root();
        let doomed = ns.create_dir_object().unwrap();
        ns.link(root, &name("doomed"), doomed).unwrap();
        let shared = ns.create_file_object().unwrap();
        ns.link(doomed, &name("shared"), shared).unwrap();
        // A second name from outside the doomed subtree.
        ns.link(root, &name("kept"), shared).unwrap();

        let (_, child) = ns.detach(root, &name("doomed")).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        ns.reclaim(child, &mut placements, &mut objects);

        assert!(!ns.exists(doomed));
        // The outside name kept the file alive.
        assert!(ns.exists(shared));
        assert_eq!(ns.object(shared).unwrap().incoming(), 1);
        assert!(placements.is_empty());
        assert_eq!(ns.lookup(root, &name("kept")).unwrap().1, shared);
    }

    #[test]
    fn detaching_one_directory_route_preserves_a_diamond_until_the_last_route() {
        let (mut ns, root) = ns_with_root();
        let left = ns.create_dir_object().unwrap();
        let right = ns.create_dir_object().unwrap();
        let shared = ns.create_dir_object().unwrap();
        let leaf = ns.create_file_object().unwrap();
        ns.link(root, &name("left"), left).unwrap();
        ns.link(root, &name("right"), right).unwrap();
        let left_edge = ns.link(left, &name("shared"), shared).unwrap();
        let right_edge = ns.link(right, &name("shared"), shared).unwrap();
        ns.link(shared, &name("leaf"), leaf).unwrap();

        let (_, child) = ns.detach(left, &name("shared")).unwrap();
        let mut placements: BoundedVec<u32, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        let mut objects: BoundedVec<ObjectId, ServiceAlloc> =
            BoundedVec::try_new_in(MAX_OBJECTS, crate::service_alloc()).unwrap();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 0);
        assert_eq!(ns.edge_child(left_edge), Err(FsError::AccessLost));
        assert_eq!(ns.edge_child(right_edge), Ok(shared));
        assert_eq!(ns.lookup(shared, &name("leaf")).unwrap().1, leaf);

        let (_, child) = ns.detach(right, &name("shared")).unwrap();
        assert_eq!(ns.reclaim(child, &mut placements, &mut objects), 2);
        assert!(!ns.exists(shared));
        assert!(!ns.exists(leaf));
        assert!(ns.exists(left) && ns.exists(right));
    }

    #[test]
    fn leaked_placement_counter_records_provider_refusals() {
        let (mut ns, _root) = ns_with_root();
        assert_eq!(ns.leaked_placements(), 0);
        ns.note_leaked_placement();
        ns.note_leaked_placement();
        assert_eq!(ns.leaked_placements(), 2);
    }

    /// Write `value` as decimal ASCII into `buf`, returning the byte count.
    /// Test-local so name generation needs no formatting machinery.
    fn write_index(buf: &mut [u8; 8], value: usize) -> usize {
        if value == 0 {
            buf[0] = b'0';
            return 1;
        }
        let mut digits = [0u8; 8];
        let mut n = 0;
        let mut v = value;
        while v > 0 && n < digits.len() {
            digits[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
        for i in 0..n {
            buf[i] = digits[n - 1 - i];
        }
        n
    }
}
