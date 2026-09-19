//! Stable object identity, kinds, revisions, and reclamation accounting.
//!
//! An object's identity is a [`SlotKey`] in the engine's object arena — not a
//! name and not a storage placement.  That is what lets `rename` change a
//! label, `move` swap one containment edge for another, and `link` add a
//! second name, all without the object becoming a different object.  Paths
//! are client-side traversal input; they are neither identity nor authority.
//!
//! # Reclamation
//!
//! Every object counts the two things that can keep it reachable: incoming
//! containment edges and live authority edges (grants).  When both reach
//! zero the object is unreachable by construction — no route exists to open
//! it — and [`Object::is_reclaimable`] says so, at which point the engine
//! releases its storage and its arena slot.
//!
//! Outstanding *handles* deliberately do not keep an object alive. "No live
//! route means no read" is immediate: a
//! handle is a bounded proof over edges, so once the edges are gone the
//! handle fails closed on its next use.  This is an intentional difference
//! from Unix, where an open file descriptor commonly keeps an unlinked
//! object alive.

use crate::slot::SlotKey;

pub use kore_dagfs::{ANY_REVISION, ObjectId, ObjectKind, ObjectSubtype, Revision, StableObjectId};

/// Service-private access to the generational naming key.
///
/// The public contract exposes only stable wire bits. Reconstructing the slot
/// key here keeps the arena implementation out of the application API while
/// preserving the exact stale-generation checks used throughout the engine.
pub(crate) trait ObjectIdExt {
    fn key(self) -> SlotKey;
}

impl ObjectIdExt for ObjectId {
    #[inline]
    fn key(self) -> SlotKey {
        SlotKey::from_bits(self.to_bits())
    }
}

/// A handle into the directory child-list arena.
///
/// Directory child lists live in their own arena rather than inline in
/// [`Object`], so a file object does not reserve directory-sized space (see
/// [`crate::bounds`]).  Purely internal: a client never sees one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DirId(pub(crate) SlotKey);

/// The revision a freshly created object carries.  Above [`ANY_REVISION`] so
/// the sentinel is unambiguous.
pub const FIRST_REVISION: Revision = 1;

/// What an object's kind-specific state is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Body<K> {
    /// A file's placement in the provider.  `None` until the file first
    /// needs bytes, so an empty file costs no storage.
    File {
        storage: Option<K>,
        staged: Option<K>,
    },
    /// A directory's child list.
    Directory { dir: DirId },
}

/// One object record.
///
/// Generic over the provider's placement key `K`
/// ([`StorageProvider::Key`](crate::provider::StorageProvider::Key)) so the
/// engine stays generic over its provider rather than erasing the key —
/// the seam's associated type reaches all the way into the object model, and
/// a key that grants no authority stays that way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Object<K> {
    stable: StableObjectId,
    kind: ObjectKind,
    /// What the object contains — data set at creation, never authority
    /// ([`ObjectSubtype`]). Immutable for the object's life: `commit_len`
    /// and every other mutation leave it alone.
    subtype: ObjectSubtype,
    body: Body<K>,
    len: u64,
    revision: Revision,
    /// Incoming containment edges — how many names place this object
    /// somewhere.
    incoming: u32,
    /// Live authority edges rooted at this object — private roots and
    /// direct grants.
    grants: u32,
}

impl<K: Copy> Object<K> {
    /// A new plain file with no placement and no bytes.
    pub(crate) const fn new_file() -> Self {
        Self::new_file_with(ObjectSubtype::PLAIN)
    }

    /// A new file stamped `subtype`, with no placement and no bytes.
    pub(crate) const fn new_file_with(subtype: ObjectSubtype) -> Self {
        Self {
            stable: StableObjectId::INVALID,
            kind: ObjectKind::File,
            subtype,
            body: Body::File {
                storage: None,
                staged: None,
            },
            len: 0,
            revision: FIRST_REVISION,
            incoming: 0,
            grants: 0,
        }
    }

    /// A new directory owning the child list `dir`.
    pub(crate) const fn new_dir(dir: DirId) -> Self {
        Self {
            stable: StableObjectId::INVALID,
            kind: ObjectKind::Directory,
            subtype: ObjectSubtype::PLAIN,
            body: Body::Directory { dir },
            len: 0,
            revision: FIRST_REVISION,
            incoming: 0,
            grants: 0,
        }
    }

    pub(crate) const fn with_stable(mut self, stable: StableObjectId) -> Self {
        self.stable = stable;
        self
    }

    pub(crate) const fn stable(&self) -> StableObjectId {
        self.stable
    }

    pub(crate) fn restore_len_revision(&mut self, len: u64, revision: Revision) {
        self.len = len;
        self.revision = revision;
    }

    #[inline]
    pub(crate) const fn kind(&self) -> ObjectKind {
        self.kind
    }

    #[inline]
    pub(crate) const fn subtype(&self) -> ObjectSubtype {
        self.subtype
    }

    #[inline]
    pub(crate) const fn len(&self) -> u64 {
        self.len
    }

    #[inline]
    pub(crate) const fn revision(&self) -> Revision {
        self.revision
    }

    #[inline]
    pub(crate) const fn incoming(&self) -> u32 {
        self.incoming
    }

    #[inline]
    pub(crate) const fn grants(&self) -> u32 {
        self.grants
    }

    /// This object's child-list handle, or `None` if it is a file.
    #[inline]
    pub(crate) const fn dir(&self) -> Option<DirId> {
        match self.body {
            Body::Directory { dir } => Some(dir),
            Body::File { .. } => None,
        }
    }

    /// This object's provider placement, or `None` if it is a directory or a
    /// file that has never needed bytes.
    #[inline]
    pub(crate) const fn storage(&self) -> Option<K> {
        match self.body {
            Body::File { storage, .. } => storage,
            Body::Directory { .. } => None,
        }
    }

    /// Record the placement the provider handed back.  A file only.
    pub(crate) fn set_storage(&mut self, key: K) {
        if let Body::File { storage, .. } = &mut self.body {
            *storage = Some(key);
        }
    }

    /// A bounded, unpublished placement retained by this exact object across
    /// cancelled or failed hydration. It is never serialized or lent. Keeping
    /// custody here permits async provider I/O without an async Drop escape.
    pub(crate) const fn staged_storage(&self) -> Option<K> {
        match self.body {
            Body::File { staged, .. } => staged,
            Body::Directory { .. } => None,
        }
    }

    pub(crate) fn set_staged_storage(&mut self, key: Option<K>) {
        if let Body::File { staged, .. } = &mut self.body {
            *staged = key;
        }
    }

    /// Commit a new length and advance the revision.  Called once, at the
    /// end of a mutation that has already succeeded, so a refused or failed
    /// request leaves the revision untouched and is invisible to readers.
    pub(crate) fn commit_len(&mut self, len: u64) {
        self.len = len;
        self.revision = self.revision.saturating_add(1);
    }

    /// `true` if `expected` permits a write against the current revision:
    /// either it is [`ANY_REVISION`] (unconditional) or it matches exactly.
    #[inline]
    pub(crate) const fn revision_admits(&self, expected: Revision) -> bool {
        expected == ANY_REVISION || expected == self.revision
    }

    pub(crate) fn incoming_inc(&mut self) {
        self.incoming = self.incoming.saturating_add(1);
    }

    pub(crate) fn incoming_dec(&mut self) {
        self.incoming = self.incoming.saturating_sub(1);
    }

    pub(crate) fn grants_inc(&mut self) {
        self.grants = self.grants.saturating_add(1);
    }

    pub(crate) fn grants_dec(&mut self) {
        self.grants = self.grants.saturating_sub(1);
    }

    /// No name places this object and no grant reaches it — so no route to
    /// it can exist, and its storage and slot may be released.
    ///
    /// Outstanding handles are intentionally not consulted; see the module
    /// docs.
    #[inline]
    pub(crate) const fn is_reclaimable(&self) -> bool {
        self.incoming == 0 && self.grants == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Obj = Object<u32>;

    #[test]
    fn a_fresh_file_has_no_storage_and_no_bytes() {
        let file = Obj::new_file();
        assert_eq!(file.kind(), ObjectKind::File);
        assert!(file.kind().is_file() && !file.kind().is_dir());
        assert_eq!(file.len(), 0);
        assert_eq!(file.storage(), None);
        assert_eq!(file.dir(), None);
    }

    #[test]
    fn a_fresh_directory_owns_a_child_list_and_no_storage() {
        let dir = Obj::new_dir(DirId(SlotKey::new(3, 1)));
        assert_eq!(dir.kind(), ObjectKind::Directory);
        assert!(dir.kind().is_dir());
        assert_eq!(dir.dir(), Some(DirId(SlotKey::new(3, 1))));
        assert_eq!(dir.storage(), None);
    }

    #[test]
    fn revisions_start_above_the_any_sentinel() {
        // The sentinel shares the wire field with real revisions, so it must
        // never collide with one.
        assert!(FIRST_REVISION > ANY_REVISION);
        assert_eq!(Obj::new_file().revision(), FIRST_REVISION);
    }

    #[test]
    fn commit_len_advances_the_revision_monotonically() {
        let mut file = Obj::new_file();
        let before = file.revision();
        file.commit_len(10);
        assert_eq!(file.len(), 10);
        assert!(file.revision() > before);
        let mid = file.revision();
        file.commit_len(4);
        assert_eq!(file.len(), 4);
        assert!(file.revision() > mid);
    }

    #[test]
    fn any_revision_is_unconditional_and_exact_revisions_are_not() {
        let mut file = Obj::new_file();
        file.commit_len(1);
        let live = file.revision();
        assert!(file.revision_admits(ANY_REVISION));
        assert!(file.revision_admits(live));
        assert!(!file.revision_admits(live - 1));
        assert!(!file.revision_admits(live + 1));
    }

    #[test]
    fn storage_placement_is_recorded_once_set() {
        let mut file = Obj::new_file();
        assert_eq!(file.storage(), None);
        file.set_storage(42);
        assert_eq!(file.storage(), Some(42));
    }

    #[test]
    fn a_directory_never_takes_a_storage_placement() {
        let mut dir = Obj::new_dir(DirId(SlotKey::new(0, 1)));
        dir.set_storage(7);
        assert_eq!(dir.storage(), None);
    }

    #[test]
    fn reclaimable_only_when_no_name_and_no_grant_remain() {
        let mut file = Obj::new_file();
        // A brand-new object is unreachable until an edge names it — the
        // engine links it in the same request that creates it.
        assert!(file.is_reclaimable());

        file.incoming_inc();
        assert!(!file.is_reclaimable());
        file.grants_inc();
        assert!(!file.is_reclaimable());

        // Dropping the only name still leaves the grant as a live route.
        file.incoming_dec();
        assert!(!file.is_reclaimable());
        assert_eq!(file.incoming(), 0);
        assert_eq!(file.grants(), 1);

        file.grants_dec();
        assert!(file.is_reclaimable());
    }

    #[test]
    fn two_names_survive_losing_one() {
        let mut file = Obj::new_file();
        file.incoming_inc();
        file.incoming_inc();
        assert_eq!(file.incoming(), 2);
        file.incoming_dec();
        assert_eq!(file.incoming(), 1);
        assert!(!file.is_reclaimable());
    }

    #[test]
    fn counts_saturate_rather_than_wrap() {
        let mut file = Obj::new_file();
        // An extra decrement must not wrap to u32::MAX and resurrect the
        // object forever.
        file.incoming_dec();
        file.grants_dec();
        assert_eq!(file.incoming(), 0);
        assert_eq!(file.grants(), 0);
        assert!(file.is_reclaimable());
    }

    #[test]
    fn object_id_bits_round_trip() {
        let id = ObjectId::from_bits(SlotKey::new(5, 9).to_bits());
        assert_eq!(ObjectId::from_bits(id.to_bits()), id);
        assert_eq!(ObjectId::INVALID.to_bits(), 0);
    }
}

#[cfg(test)]
mod subtype_tests {
    use super::*;

    #[test]
    fn a_fresh_file_is_plain_and_a_stamped_one_keeps_its_stamp() {
        let plain = Object::<u32>::new_file();
        assert_eq!(plain.subtype(), ObjectSubtype::PLAIN);
        let stamped = Object::<u32>::new_file_with(ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(stamped.subtype(), ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(
            Object::<u32>::new_dir(DirId(crate::slot::SlotKey::INVALID)).subtype(),
            ObjectSubtype::PLAIN
        );
    }

    /// There is no setter: every mutation the object admits leaves the
    /// stamp alone.
    #[test]
    fn a_subtype_is_fixed_at_creation() {
        let mut stamped = Object::<u32>::new_file_with(ObjectSubtype::EXECUTABLE_IMAGE);
        stamped.set_storage(9);
        stamped.commit_len(77);
        stamped.restore_len_revision(3, 5);
        stamped.incoming_inc();
        stamped.grants_inc();
        assert_eq!(stamped.subtype(), ObjectSubtype::EXECUTABLE_IMAGE);
    }

    /// Subtype still fits beside kind. The row now also retains one unpublished
    /// storage key while hydration is pending; bound that additional custody.
    #[test]
    fn object_size_includes_pending_storage_custody() {
        assert!(core::mem::size_of::<Object<u32>>() <= 56);
    }
}
