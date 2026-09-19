//! The request engine — generic over its [`StorageProvider`].
//!
//! This is the whole of the filesystem's behaviour, and it is deliberately
//! the only place the three lower layers meet: it revalidates a route through
//! [`Authority`], mutates the DAG through [`Namespace`], and moves bytes
//! through the provider.  Nothing below it knows about the others.
//!
//! # Every operation revalidates
//!
//! There is no fast path.  Each method begins by re-walking the handle's
//! bounded edge proof ([`Authority::revalidate`]), and only then checks
//! rights.  A client that holds a numeric handle whose route has been
//! dismantled gets [`FsError::AccessLost`] on its next call, not on some
//! later cache expiry — and because the engine keeps no decision from a
//! previous request, there is nothing that could bypass the check.
//!
//! # What makes a request atomic
//!
//! A mutation either commits completely or has no visible effect.
//! Two properties deliver that:
//!
//! 1. **Capacity is reserved before any byte lands.** The engine calls
//!    [`StorageProvider::resize`] to cover the whole target range first, so
//!    the failure that can run out of room happens while nothing is visible.
//!    The subsequent write goes into already-reserved space.
//! 2. **Length and revision commit last, together.** A file's visible length
//!    and its revision advance in one step *after* the bytes are in place
//!    ([`Object::commit_len`](crate::object)).  A refused or failed write
//!    leaves both untouched, so a reader can never observe a half-applied
//!    request — it sees the previous complete revision.
//!
//! `append` gets its atomicity from the same structure: it claims the exact
//! file generation before choosing the current end and retains that affine
//! claim through provider I/O and the final metadata commit. A conflicting
//! operation receives bounded [`FsError::Busy`] backpressure while unrelated
//! objects and directories continue independently; there is no global
//! namespace mutation gate.

mod hydration;
mod namespace_commit;

pub(crate) use hydration::Hydration;

use crate::{
    authority::{
        Authority, AuthorityKind, AuthorityQuota, EdgeProof, GrantId, Handle, QuotaClaim, Rights,
        Route, SessionId,
    },
    bounds::{
        MAX_AUTHORITY_EDGES, MAX_DERIVED_RESULTS, MAX_LIST_PAGE, MAX_OBJECTS, MAX_PATH_DEPTH,
        MAX_TRANSFER, RECLAIM_CONTENTION_TURNS,
    },
    error::{FsError, FsResult, ProviderError},
    extraction::{
        DerivedResultInfo, ExtractionError, ExtractionOwner, ExtractionState, Snapshot,
        VersionLease,
    },
    namespace::{ListEntry, Name, Namespace},
    object::{Object, ObjectId, ObjectKind, ObjectSubtype, Revision, StableObjectId},
    provider::{RamProvider, StorageProvider},
    search::{
        FreshnessPolicy, MAX_SEARCH_PAGE, MAX_SEARCH_SCAN, QueryIndex, SearchCursor, SearchHit,
        SearchPage, SearchQuery, SearchScore,
    },
};
use kore_svc_extract::{
    registry::ParserId,
    schema::{InputIdentity, NormalizedResult},
};

struct SearchWalk {
    hash: u64,
    snapshot: u64,
    scanned: u32,
    work: u32,
    work_limit: u32,
    index_generation: u64,
    start: u32,
    next: Option<u32>,
    hits: heapless::Vec<SearchHit, MAX_SEARCH_PAGE>,
}

impl SearchWalk {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new(start: u32, work_limit: u32) -> Self {
        Self {
            hash: Self::FNV_OFFSET,
            snapshot: Self::FNV_OFFSET,
            scanned: 0,
            work: 0,
            work_limit,
            index_generation: Self::FNV_OFFSET,
            start,
            next: None,
            hits: heapless::Vec::new(),
        }
    }

    fn hash_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(Self::FNV_PRIME);
        }
    }

    fn hash_u64(&mut self, value: u64) {
        self.hash_bytes(&value.to_le_bytes());
    }

    fn hash_structural_bytes(&mut self, bytes: &[u8]) {
        self.hash_bytes(bytes);
        for byte in bytes {
            self.snapshot ^= u64::from(*byte);
            self.snapshot = self.snapshot.wrapping_mul(Self::FNV_PRIME);
        }
    }

    fn hash_structural_u64(&mut self, value: u64) {
        self.hash_structural_bytes(&value.to_le_bytes());
    }

    fn hash_index_generation(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.index_generation ^= u64::from(byte);
            self.index_generation = self.index_generation.wrapping_mul(Self::FNV_PRIME);
        }
    }

    fn charge(&mut self, units: u32) -> FsResult<()> {
        self.work = self
            .work
            .checked_add(units)
            .ok_or(FsError::BudgetExceeded)?;
        if self.work > self.work_limit {
            return Err(FsError::BudgetExceeded);
        }
        Ok(())
    }
}

/// Yield one executor turn without a timer or allocation.
async fn yield_once() {
    let mut yielded = false;
    core::future::poll_fn(|cx| {
        if core::mem::replace(&mut yielded, true) {
            core::task::Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    })
    .await
}

/// The bounded in-memory composition of the provider-neutral engine.
///
/// `Engine<BlockProvider<_>>` uses this same engine with no provider branch
/// in namespace or authority code.
pub type RamEngine = Engine<RamProvider>;

/// What a client learns when it opens or creates an object.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OpenedObject {
    pub handle: Handle,
    /// Durable identity for cache correlation. It grants no authority; the
    /// live session-bound `handle` remains required for every operation.
    pub stable_object: StableObjectId,
    pub kind: ObjectKind,
    /// Data, never authority ([`ObjectSubtype`]).
    pub subtype: ObjectSubtype,
    pub len: u64,
    pub revision: Revision,
}

/// An object's metadata.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StatInfo {
    /// Which object this is.
    ///
    /// Safe to hand out: an object id conveys no authority, and a
    /// client that has one still needs a live route to do anything with it.
    /// Exposing it is what lets a caller tell whether two names it holds are
    /// two names for the *same* object rather than for copies.
    pub object: ObjectId,
    /// Stable naming identity stored in durable metadata. It grants no
    /// authority; handles and [`ObjectId`] remain boot-local.
    pub stable_object: StableObjectId,
    pub kind: ObjectKind,
    /// Data, never authority ([`ObjectSubtype`]).
    pub subtype: ObjectSubtype,
    pub len: u64,
    pub revision: Revision,
    /// Containment edges naming this object — how many names it has.
    pub links: u32,
    /// Live authority edges rooted at it.
    pub grants: u32,
}

/// One page of a directory listing.
#[derive(Clone, Debug)]
pub struct ListPage {
    pub entries: heapless::Vec<ListEntry, MAX_LIST_PAGE>,
    /// Cursor to resume at, or `None` when the listing is complete.
    pub next: Option<u32>,
}

/// One of a session's authority edges, as the session sees it.
#[derive(Clone, Copy, Debug)]
pub struct GrantInfo {
    pub grant: GrantId,
    pub kind: AuthorityKind,
    pub rights: Rights,
    pub quota: AuthorityQuota,
    pub display_name: Option<Name>,
}

/// The filesystem engine.
pub struct Engine<P: StorageProvider> {
    ns: Namespace<P::Key>,
    auth: Authority,
    provider: P,
    extraction: ExtractionState<P::Key>,
    /// Whether the volume behind this engine can record an object subtype.
    /// A fresh or RAM engine always can; a recovered volume can only from
    /// the durable format that added the byte, and it keeps committing at
    /// its own version — so on an older volume a non-plain create refuses
    /// rather than holding a fact the disk would silently drop.
    durable_subtypes: bool,
}

impl<P: StorageProvider> Engine<P> {
    pub fn new(provider: P) -> Self {
        Self {
            ns: Namespace::new(),
            auth: Authority::new(),
            provider,
            extraction: ExtractionState::new(),
            durable_subtypes: true,
        }
    }

    pub(crate) fn namespace(&self) -> &Namespace<P::Key> {
        &self.ns
    }

    /// Live sessions (attached or mid-transition), for the diagnostic
    /// snapshot. A count, never an enumeration.
    pub(crate) fn live_sessions(&self) -> u16 {
        self.auth.live_sessions()
    }

    /// The persisted tree's aggregate usage — objects, containment edges,
    /// file bytes reachable from session zero's root.
    pub(crate) fn system_tree_usage(&self) -> FsResult<(u32, u32, u64)> {
        let root = self.auth.root_object(SessionId::new(0))?;
        let usage = self.ns.graph_usage(root)?;
        Ok((usage.objects, usage.edges, usage.bytes))
    }

    pub(crate) fn root_object(&self, session: SessionId) -> FsResult<ObjectId> {
        self.auth.root_object(session)
    }

    pub(crate) fn from_recovered(
        provider: P,
        ns: Namespace<P::Key>,
        session: SessionId,
        root: ObjectId,
        durable_subtypes: bool,
    ) -> FsResult<Self> {
        if !ns.kind(root)?.is_dir() {
            return Err(FsError::NotDirectory);
        }
        let auth = Authority::new();
        auth.attach_session(session)?;
        auth.install_root(session, root)?;
        ns.object_mut(root)?.grants_inc();
        Ok(Self {
            ns,
            auth,
            provider,
            extraction: ExtractionState::new(),
            durable_subtypes,
        })
    }

    /// The provider, for limits and (in tests) fault injection.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub(crate) fn into_provider(self) -> P {
        self.provider
    }

    /// Live objects across every session's namespace.
    pub fn object_count(&self) -> usize {
        self.ns.object_count()
    }

    /// Live directory child lists — the arena bounded separately from
    /// objects, so it is worth being able to see it drain.
    pub fn directory_count(&self) -> usize {
        self.ns.directory_count()
    }

    /// Live authority edges.
    pub fn authority_edge_count(&self) -> usize {
        self.auth.edge_count()
    }

    /// Placements the provider refused to release.  See
    /// [`Namespace::reclaim`] on why a release failure is counted rather than
    /// reported as a failure of the request that caused it.
    pub fn leaked_placements(&self) -> u32 {
        self.ns.leaked_placements()
    }

    /// Reclamations that stopped before reaching the whole unreachable
    /// subtree — a structural refusal or exhausted contention budget. Non-zero
    /// means storage is occupied by objects nothing can reach.
    pub fn abandoned_reclaims(&self) -> u32 {
        self.ns.abandoned_reclaims()
    }

    /// Open handles a session holds.
    pub fn live_handles(&self, session: SessionId) -> usize {
        self.auth.live_handles(session)
    }

    // ── sessions ────────────────────────────────────────────────────────────

    /// Attach a session and give it a fresh, empty private root.
    ///
    /// This is the only way a session acquires filesystem access: there is no
    /// machine-wide root to inherit, so a task that is never attached has no
    /// route to anything: filesystem access is never ambient.
    pub async fn attach_session(&self, session: SessionId) -> FsResult<()> {
        self.attach_namespace(session).await.map(|_| ())
    }

    /// Return the freshly installed root as part of the attach transaction,
    /// so a caller retaining namespace custody needs no second fallible lookup.
    pub(crate) async fn attach_namespace(&self, session: SessionId) -> FsResult<GrantId> {
        // Validate the session slot before allocating anything, so a bad
        // session id cannot leave a stray root behind.
        self.auth.attach_session(session)?;
        let root = match self.ns.create_dir_object() {
            Ok(root) => root,
            Err(error) => {
                self.auth.abort_attach(session);
                return Err(error);
            }
        };
        match self.auth.install_root(session, root) {
            Ok(grant) => {
                self.ns.object_mut(root)?.grants_inc();
                Ok(grant)
            }
            Err(err) => {
                self.auth.abort_attach(session);
                // Undo the root: nothing names it and nothing grants it, so
                // reclamation removes it cleanly.
                self.reclaim(root).await;
                Err(err)
            }
        }
    }

    /// Claim a task session without manufacturing an ambient root. A later
    /// endpoint-authenticated invocation view completes the root transition.
    pub(crate) fn attach_empty_session(&self, session: SessionId) -> FsResult<()> {
        self.auth.attach_empty_session(session)
    }

    /// Pin one caller-authorized route as a service-private delegation
    /// ancestor. The returned id never crosses the public filesystem wire.
    pub(crate) fn pin_view(
        &self,
        session: SessionId,
        handle: Handle,
        rights: Rights,
    ) -> FsResult<(GrantId, SessionId)> {
        let route = self.route(session, handle)?;
        // Session zero is the persistent bootstrap owner and therefore has
        // no earlier invocation ancestor. Every recyclable process must pin
        // beneath the stable ancestor from which its root was installed.
        if session == SessionId::new(0) {
            return self
                .grant(session, handle, session, rights, None)
                .map(|grant| (grant, session));
        }
        self.auth.pin_route(session, route, rights)
    }

    /// Complete an empty dynamic session with an attenuated descendant of a
    /// pinned structural view.
    pub(crate) fn install_view_root(
        &self,
        session: SessionId,
        view: GrantId,
        rights: Rights,
    ) -> FsResult<()> {
        let grant = self.auth.install_delegated_root(session, view, rights)?;
        let object = self.auth.validated_edge(grant)?.root();
        self.ns.object_mut(object)?.grants_inc();
        Ok(())
    }

    /// The private-root edge owned by a stable task-view session. The id is
    /// service-private and is used only as a derivation ancestor for transient
    /// command sessions.
    pub(crate) fn task_view_root(&self, session: SessionId) -> FsResult<GrantId> {
        self.auth.root_grant(session)
    }

    /// Attach a session whose structural root is an attenuated delegation of
    /// `source` — the invocation-view posture ported from the
    /// capability-selected execution branch: the session enters the
    /// filesystem through an existing view (typically the persisted system
    /// root pinned at bootstrap) instead of receiving a fresh ambient
    /// private namespace.  Everything it creates therefore lands inside the
    /// source's tree — which is what lets ordinary task writes become
    /// durable facts — while rights can only narrow and the source edge
    /// remains the revocation ancestor.
    ///
    /// The delegated root edge participates in the ordinary lifecycle: it
    /// pins the source's root object like any grant, and
    /// [`detach_session`](Self::detach_session) retires it through the same
    /// stranded-root accounting, where reachability keeps the shared tree
    /// alive.
    pub fn attach_delegated_session(
        &self,
        session: SessionId,
        source: GrantId,
        rights: Rights,
    ) -> FsResult<()> {
        self.auth.attach_session(session)?;
        match self.auth.install_delegated_root(session, source, rights) {
            Ok(_) => {
                let root = self.auth.root_object(session)?;
                self.ns.object_mut(root)?.grants_inc();
                Ok(())
            }
            Err(error) => {
                self.auth.abort_attach(session);
                Err(error)
            }
        }
    }

    /// Detach a session: drop its handles, retire the authority edges it
    /// holds, and reclaim whatever that left unreachable.
    pub async fn detach_session(&self, session: SessionId) -> FsResult<()> {
        let stranded = self.auth.detach_session(session)?;
        for root in stranded.iter() {
            if let Ok(mut object) = self.ns.object_mut(*root) {
                object.grants_dec();
            }
        }
        for root in stranded.iter() {
            self.reclaim(*root).await;
        }
        Ok(())
    }

    /// Open the session's private root.
    pub fn open_root(&self, session: SessionId, requested: Rights) -> FsResult<OpenedObject> {
        let grant = self.auth.root_grant(session)?;
        self.open_authority_edge(session, grant, requested)
    }

    /// Open a granted file or subtree root.
    ///
    /// Only the grant's holder may do this — a [`GrantId`] handed to a third
    /// session is inert.
    pub fn open_grant(
        &self,
        session: SessionId,
        grant: GrantId,
        requested: Rights,
    ) -> FsResult<OpenedObject> {
        self.open_authority_edge(session, grant, requested)
    }

    /// Open an attachment held by `view` into one transient command session.
    /// The command receives only a handle; the stable grant remains owned by
    /// the task view and therefore survives process teardown.
    pub(crate) fn open_task_grant(
        &self,
        session: SessionId,
        view: SessionId,
        grant: GrantId,
        requested: Rights,
    ) -> FsResult<OpenedObject> {
        let edge = self.auth.validated_edge(grant)?;
        if edge.holder() != view || edge.kind() == AuthorityKind::PrivateRoot {
            return Err(FsError::InvalidHandle);
        }
        let root = edge.root();
        let rights = edge.rights().attenuate(requested)?;
        let quota = edge.quota();
        drop(edge);
        let projected = self
            .auth
            .insert_grant(view, session, grant, root, rights, quota, None)?;
        self.ns.object_mut(root)?.grants_inc();
        match self.open_authority_edge(session, projected, rights) {
            Ok(opened) => Ok(opened),
            Err(error) => {
                if let Ok(roots) = self.auth.retire_edge(session, projected) {
                    for object in roots {
                        if let Ok(mut object) = self.ns.object_mut(object) {
                            object.grants_dec();
                        }
                    }
                }
                Err(error)
            }
        }
    }

    fn open_authority_edge(
        &self,
        session: SessionId,
        grant: GrantId,
        requested: Rights,
    ) -> FsResult<OpenedObject> {
        let edge = self.auth.validated_edge(grant)?;
        if edge.holder() != session {
            // Not this session's route; from here it does not exist.
            return Err(FsError::InvalidHandle);
        }
        let root = edge.root();
        let rights = edge.rights().attenuate(requested)?;
        let (stable_object, kind, subtype, len, revision) = {
            let object = self.ns.object(root)?;
            (
                object.stable(),
                object.kind(),
                object.subtype(),
                object.len(),
                object.revision(),
            )
        };
        let handle = self
            .auth
            .insert_handle(session, root, grant, EdgeProof::root(), rights)?;
        Ok(OpenedObject {
            handle,
            stable_object,
            kind,
            subtype,
            len,
            revision,
        })
    }

    /// The authority edges this session holds — its private root plus every
    /// grant made to it. This is how a client discovers its task-local
    /// attachments.
    pub fn grants(
        &self,
        session: SessionId,
    ) -> FsResult<heapless::Vec<GrantInfo, MAX_AUTHORITY_EDGES>> {
        Ok(self
            .auth
            .grants_of(session)?
            .iter()
            .map(|(grant, kind, rights, quota, display_name)| GrantInfo {
                grant: *grant,
                kind: *kind,
                rights: *rights,
                quota: *quota,
                display_name: *display_name,
            })
            .collect())
    }

    // ── traversal ───────────────────────────────────────────────────────────

    /// Resolve one name component beneath an open directory handle.
    ///
    /// Single-component by design: paths are client-side traversal input, and
    /// each step extends the new handle's proof by exactly one edge.
    pub fn open_at(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        requested: Rights,
    ) -> FsResult<OpenedObject> {
        let route = self.route(session, parent)?;
        // Resolving a name is reading the directory.
        route.rights.require(Rights::READ)?;
        if !self.ns.kind(route.object)?.is_dir() {
            return Err(FsError::NotDirectory);
        }
        let rights = route.rights.attenuate(requested)?;
        let (edge, child) = self.ns.lookup(route.object, name)?;

        let (grant, proof) = {
            let record = self.auth.handle(session, parent)?;
            (record.grant(), record.proof().extend(edge)?)
        };
        let (stable_object, kind, subtype, len, revision) = {
            let object = self.ns.object(child)?;
            (
                object.stable(),
                object.kind(),
                object.subtype(),
                object.len(),
                object.revision(),
            )
        };
        let handle = self
            .auth
            .insert_handle(session, child, grant, proof, rights)?;
        Ok(OpenedObject {
            handle,
            stable_object,
            kind,
            subtype,
            len,
            revision,
        })
    }

    /// Close a handle.  Does not affect the object or any other route to it.
    pub fn close(&self, session: SessionId, handle: Handle) -> FsResult<()> {
        self.auth.remove_handle(session, handle).map(|_| ())
    }

    /// Only the trusted overlay owner uses this to discard tombstones after
    /// native reclamation. Presence conveys no route or operation rights.
    pub(crate) fn contains_stable_object(&self, stable: StableObjectId) -> bool {
        self.ns.find_stable(stable).is_some()
    }

    /// Service-private classification after native route revalidation. This
    /// does not authorize a read or mutation: the eventual operation must
    /// still check its own rights. In particular a host overlay must not add
    /// a READ requirement to an unrelated native write-only handle.
    pub(crate) fn stable_object_for_handle(
        &self,
        session: SessionId,
        handle: Handle,
    ) -> FsResult<StableObjectId> {
        let route = self.route(session, handle)?;
        Ok(self.ns.object(route.object)?.stable())
    }

    /// An object's metadata.
    pub fn stat(&self, session: SessionId, handle: Handle) -> FsResult<StatInfo> {
        self.stat_with_rights(session, handle, Rights::READ)
    }

    /// Trusted operation preparation; this metadata is not returned through a
    /// READ-less Stat reply. Writeback must also work on write-only handles.
    pub(crate) fn stat_with_rights(
        &self,
        session: SessionId,
        handle: Handle,
        required: Rights,
    ) -> FsResult<StatInfo> {
        let route = self.route(session, handle)?;
        route.rights.require(required)?;
        let object = self.ns.object(route.object)?;
        Ok(StatInfo {
            object: route.object,
            stable_object: object.stable(),
            kind: object.kind(),
            subtype: object.subtype(),
            len: object.len(),
            revision: object.revision(),
            links: object.incoming(),
            grants: object.grants(),
        })
    }

    /// One page of a directory's entries.
    pub fn list(&self, session: SessionId, handle: Handle, cursor: u32) -> FsResult<ListPage> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::READ)?;
        let (entries, next) = self.ns.list(route.object, cursor)?;
        Ok(ListPage { entries, next })
    }

    /// Search begins at the caller's revalidated route. Candidate generation
    /// is authority-filtered before predicates run; object ids in results are
    /// observation data and never handles.
    pub fn search(
        &self,
        session: SessionId,
        handle: Handle,
        query: SearchQuery,
        cursor: Option<SearchCursor>,
    ) -> FsResult<SearchPage> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::READ)?;
        if !self.ns.kind(route.object)?.is_dir() || query.min_len > query.max_len {
            return Err(if query.min_len > query.max_len {
                FsError::Range
            } else {
                FsError::NotDirectory
            });
        }
        match query.freshness {
            FreshnessPolicy::StructuralOnly
                if query.content.is_some() || query.max_index_lag != 0 =>
            {
                return Err(FsError::Range);
            }
            FreshnessPolicy::CurrentOnly if query.max_index_lag != 0 => {
                return Err(FsError::Range);
            }
            _ => {}
        }

        let start = cursor.map_or(0, |cursor| cursor.position);
        let mut walk = SearchWalk::new(start, query.work_limit());
        walk.hash_u64(handle.to_bits());
        walk.hash_u64(route.grant.to_bits());
        walk.hash_u64(route.object.to_bits());
        walk.hash_bytes(&[route.rights.bits()]);
        walk.hash_u64(query.min_len);
        walk.hash_u64(query.max_len);
        walk.hash_bytes(&[query.freshness as u8]);
        walk.hash_u64(u64::from(query.max_index_lag));
        walk.hash_u64(u64::from(query.max_work));
        walk.hash_bytes(&[match query.kind {
            None => 0,
            Some(ObjectKind::File) => 1,
            Some(ObjectKind::Directory) => 2,
        }]);
        if let Some(name) = query.name_contains {
            walk.hash_bytes(&[name.len() as u8]);
            walk.hash_bytes(name.as_bytes());
        } else {
            walk.hash_bytes(&[0]);
        }
        if let Some(content) = query.content {
            walk.hash_bytes(&[content.plane as u8, content.term.len() as u8]);
            walk.hash_bytes(content.term.as_bytes());
        } else {
            walk.hash_bytes(&[0]);
        }

        let mut path = heapless::Vec::new();
        self.search_dir(route.object, &query, &mut path, &mut walk)?;
        if walk.hash == 0 {
            walk.hash = SearchWalk::FNV_OFFSET;
        }
        if walk.snapshot == 0 {
            walk.snapshot = SearchWalk::FNV_OFFSET;
        }
        let index_generation = if query.freshness == FreshnessPolicy::StructuralOnly {
            0
        } else {
            walk.index_generation
        };
        walk.hash_u64(index_generation);
        if usize::try_from(start).map_or(true, |start| start > walk.scanned as usize) {
            return Err(FsError::Range);
        }
        if cursor.is_some_and(|cursor| cursor.observation != walk.hash) {
            return Err(FsError::Conflict);
        }
        // Ranking is scoped to this authorized result page. Keep traversal as
        // the deterministic tie-breaker while presenting higher-confidence
        // content matches first; hidden candidates never enter this vector.
        for index in 1..walk.hits.len() {
            let mut at = index;
            while at > 0 && walk.hits[at].score > walk.hits[at - 1].score {
                walk.hits.swap(at, at - 1);
                at -= 1;
            }
        }
        Ok(SearchPage {
            hits: walk.hits,
            observation: walk.hash,
            snapshot: walk.snapshot,
            index_generation,
            freshness: query.freshness,
            next: walk.next.map(|position| SearchCursor {
                observation: walk.hash,
                position,
            }),
            scanned: walk.scanned,
            work: walk.work,
        })
    }

    fn search_dir(
        &self,
        directory: ObjectId,
        query: &SearchQuery,
        path: &mut heapless::Vec<Name, MAX_PATH_DEPTH>,
        walk: &mut SearchWalk,
    ) -> FsResult<()> {
        let mut cursor = 0;
        loop {
            let (entries, next) = self.ns.list(directory, cursor)?;
            for entry in entries {
                if walk.scanned as usize >= MAX_SEARCH_SCAN {
                    return Err(FsError::BudgetExceeded);
                }
                walk.charge(1)?;
                let position = walk.scanned;
                walk.scanned += 1;
                let object = self.ns.object(entry.object)?;
                let stable = object.stable();
                let len = object.len();
                let revision = object.revision();

                walk.hash_structural_bytes(&[path.len() as u8]);
                for component in path.iter() {
                    walk.hash_structural_bytes(&[component.len() as u8]);
                    walk.hash_structural_bytes(component.as_bytes());
                }
                walk.hash_structural_bytes(&[entry.name.len() as u8]);
                walk.hash_structural_bytes(entry.name.as_bytes());
                walk.hash_structural_u64(entry.object.to_bits());
                walk.hash_structural_u64(stable.get());
                walk.hash_structural_u64(len);
                walk.hash_structural_u64(revision);
                walk.hash_structural_bytes(&[if entry.kind.is_dir() { 2 } else { 1 }]);

                let structural_matches = query.matches_structural(&entry.name, entry.kind, len);
                let indexed = if structural_matches && entry.kind.is_file() {
                    if let Some(content) = query.content {
                        let indexed = self.extraction.evaluate(entry.object, revision, content)?;
                        walk.charge(indexed.work)?;
                        indexed
                    } else {
                        crate::search::IndexMatch {
                            matched: true,
                            score: SearchScore::NONE,
                            snippet: heapless::String::new(),
                            work: 0,
                            generation: 0,
                        }
                    }
                } else if structural_matches && query.content.is_none() {
                    crate::search::IndexMatch {
                        matched: true,
                        score: SearchScore::NONE,
                        snippet: heapless::String::new(),
                        work: 0,
                        generation: 0,
                    }
                } else {
                    crate::search::IndexMatch::miss(0)
                };
                if query.content.is_some() && structural_matches {
                    walk.hash_index_generation(indexed.generation);
                }

                path.push(entry.name).map_err(|_| FsError::Range)?;
                if structural_matches && indexed.matched && position >= walk.start {
                    if walk.hits.is_full() {
                        walk.next.get_or_insert(position);
                    } else {
                        walk.hits
                            .push(SearchHit {
                                path: path.clone(),
                                object: entry.object,
                                stable_object: stable,
                                kind: entry.kind,
                                len,
                                revision,
                                score: indexed.score,
                                snippet: indexed.snippet,
                            })
                            .map_err(|_| FsError::Range)?;
                    }
                }
                if entry.kind.is_dir() {
                    self.search_dir(entry.object, query, path, walk)?;
                }
                let _ = path.pop();
            }
            let Some(resume) = next else {
                break;
            };
            cursor = resume;
        }
        Ok(())
    }

    // ── namespace mutation ──────────────────────────────────────────────────

    /// Create one object and one containment edge naming it, and hand the
    /// creator a handle to it.
    ///
    /// The creator immediately receives the rights its parent authority
    /// allowed, so a task can use what it just made without a
    /// second lookup.  All-or-nothing: if the edge or the handle cannot be
    /// made, the object is reclaimed and nothing is left behind.
    pub async fn create(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
    ) -> FsResult<OpenedObject> {
        self.create_typed(session, parent, name, kind, ObjectSubtype::PLAIN)
            .await
    }

    /// [`Self::create`], stamping `subtype` — data the object carries for its
    /// life, never authority.
    pub async fn create_typed(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
        subtype: ObjectSubtype,
    ) -> FsResult<OpenedObject> {
        self.create_with_commit(session, parent, name, kind, subtype, async |_| Ok(()))
            .await
    }

    /// Add another containment route to an existing file or directory.
    ///
    /// Directory links run a bounded graph reachability proof before the edge
    /// is published, so diamonds are accepted and cycles are refused.
    ///
    /// Both handles must have been entered through the *same* authority edge
    /// — see [`Self::require_same_authority`], which explains the escalation
    /// that rule prevents.
    pub fn link(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        child: Handle,
    ) -> FsResult<()> {
        let parent_route = self.route(session, parent)?;
        parent_route.rights.require(Rights::MUTATE_NAMESPACE)?;
        // You must be able to see the object to give it a name.
        let child_route = self.route(session, child)?;
        child_route.rights.require(Rights::READ)?;
        self.require_same_authority(session, parent, child)?;
        let _view_claim = self.claim_view_if_root(session, parent)?;
        self.require_view_name_free(session, parent, name)?;
        let _quota = self.require_growth(parent_route, 0, 1, 0)?;
        self.ns
            .link(parent_route.object, name, child_route.object)
            .map(|_| ())
    }

    /// Change one edge's label.  The edge — and therefore every handle
    /// proving through it — survives.
    pub fn rename(
        &self,
        session: SessionId,
        parent: Handle,
        old: &Name,
        new: &Name,
    ) -> FsResult<()> {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        let _view_claim = self.claim_view_if_root(session, parent)?;
        self.require_view_name_free(session, parent, new)?;
        self.ns.rename(route.object, old, new).map(|_| ())
    }

    /// Atomically remove one edge and create one edge.
    ///
    /// Requires namespace-mutation rights at *both* parents.  A handle opened
    /// through the source edge becomes invalid; one opened through a separate
    /// link to the same object stays valid.
    ///
    /// Both parents must be reached through the same authority edge — a move
    /// across two attachments is refused explicitly so a weaker grant cannot
    /// be laundered into the destination root's stronger authority. See
    /// [`Self::require_same_authority`].
    pub fn relocate(
        &self,
        session: SessionId,
        src_parent: Handle,
        name: &Name,
        dst_parent: Handle,
        new_name: &Name,
    ) -> FsResult<()> {
        let src = self.route(session, src_parent)?;
        src.rights.require(Rights::MUTATE_NAMESPACE)?;
        let dst = self.route(session, dst_parent)?;
        dst.rights.require(Rights::MUTATE_NAMESPACE)?;
        self.require_same_authority(session, src_parent, dst_parent)?;
        let _view_claim = self.claim_view_if_root(session, dst_parent)?;
        self.require_view_name_free(session, dst_parent, new_name)?;
        self.ns
            .relocate(src.object, name, dst.object, new_name)
            .map(|_| ())
    }

    /// Remove exactly one native graph edge, including an edge to a populated
    /// directory. Other containment routes and grants are untouched; graph
    /// reclamation runs only if this was the final route.
    pub async fn detach_edge(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
    ) -> FsResult<()> {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        let (_, child) = self.ns.detach(route.object, name)?;
        self.reclaim(child).await;
        Ok(())
    }

    /// Compatibility unlink/rmdir. Files and empty directories are removed;
    /// a populated directory is refused. Use [`Self::detach_edge`] for the
    /// native edge-removal operation.
    pub async fn unlink(&self, session: SessionId, parent: Handle, name: &Name) -> FsResult<()> {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        let (_, child) = self.ns.unlink_empty(route.object, name)?;
        self.reclaim(child).await;
        Ok(())
    }

    // ── sharing ─────────────────────────────────────────────────────────────

    /// Give another session an attenuated authority edge to this object.
    ///
    /// Requires [`Rights::SHARE`] on the route, and the delegated rights are
    /// intersected with what the caller holds — sharing can only narrow.  The
    /// recipient's `display_name`, if given, must be free in *its* view, so a
    /// sender cannot overwrite a name in someone else's namespace of
    /// attachments.
    pub fn grant(
        &self,
        session: SessionId,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        let quota = self.route(session, handle)?.quota;
        self.grant_scoped(session, handle, destination, rights, quota, display_name)
    }

    /// Give another session an attenuated route with an explicit budget.
    ///
    /// Quota attenuation follows rights attenuation: a child edge may ask for
    /// less, never more. No capacity is reserved eagerly. Every later growth
    /// derives exact usage from this grant's live graph before mutation.
    pub fn grant_scoped(
        &self,
        session: SessionId,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        quota: AuthorityQuota,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::SHARE)?;
        let delegated = route.rights.attenuate(rights)?;
        let delegated_quota = route.quota.attenuate(quota)?;
        let _destination_view = self.auth.claim_view(destination)?;
        if let Some(name) = display_name.as_ref() {
            if self.view_name_taken(destination, name)? {
                return Err(FsError::AlreadyExists);
            }
        }
        let grant = self.auth.insert_grant(
            session,
            destination,
            route.grant,
            route.object,
            delegated,
            delegated_quota,
            display_name,
        )?;
        // The grant is now a live route to the object, so it keeps it alive.
        self.ns.object_mut(route.object)?.grants_inc();
        Ok(grant)
    }

    /// Publish a route resolved by a transient command into another stable
    /// task view. `source_view` must be an actual ancestor of the command's
    /// route; the authority layer proves that relation before reparenting the
    /// grant, so command teardown cannot revoke a successful share.
    pub(crate) fn grant_between_task_views(
        &self,
        session: SessionId,
        handle: Handle,
        source_view: SessionId,
        destination_view: SessionId,
        rights: Rights,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::SHARE)?;
        let delegated = route.rights.attenuate(rights)?;
        let _destination = self.auth.claim_view(destination_view)?;
        if let Some(name) = display_name.as_ref()
            && self.view_name_taken(destination_view, name)?
        {
            return Err(FsError::AlreadyExists);
        }
        let grant = self.auth.grant_route_from_ancestor(
            session,
            source_view,
            destination_view,
            route,
            delegated,
            display_name,
        )?;
        self.ns.object_mut(route.object)?.grants_inc();
        Ok(grant)
    }

    /// Retire a grant.  Either side of it may do so; nobody else may.
    ///
    /// Every handle that entered through the grant fails its next
    /// revalidation with [`FsError::AccessLost`].
    pub async fn revoke(&self, session: SessionId, grant: GrantId) -> FsResult<()> {
        let roots = self.auth.retire_edge(session, grant)?;
        for root in roots.iter() {
            if let Ok(mut object) = self.ns.object_mut(*root) {
                object.grants_dec();
            }
        }
        for root in roots {
            self.reclaim(root).await;
        }
        Ok(())
    }

    /// Close a service-retained view grant whose exact generation was minted
    /// here. An earlier ancestor revocation may already have retired it. Only
    /// an absent exact edge generation proves that case; contention and other
    /// failures keep the view's cleanup owner. Public Revoke remains unchanged.
    pub(crate) async fn retire_view_grant(&self, session: SessionId, grant: GrantId) -> FsResult<()> {
        match self.revoke(session, grant).await {
            Err(FsError::AccessLost) if matches!(self.auth.edge(grant), Err(FsError::AccessLost)) => Ok(()),
            result => result,
        }
    }

    // ── bytes ───────────────────────────────────────────────────────────────

    /// Read from a file into `dst`, returning the bytes transferred.
    ///
    /// Short at end-of-file rather than an error; a read starting at or past
    /// the end transfers zero.
    pub async fn read(
        &self,
        session: SessionId,
        handle: Handle,
        offset: u64,
        dst: &mut [u8],
    ) -> FsResult<usize> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::READ)?;
        let (key, len) = {
            let object = self.ns.object(route.object)?;
            if !object.kind().is_file() {
                return Err(FsError::IsDirectory);
            }
            (object.storage(), object.len())
        };
        // Overflow is a refusal, never a wrap into a valid-looking range.
        offset.checked_add(dst.len() as u64).ok_or(FsError::Range)?;
        if offset >= len {
            return Ok(0);
        }
        let want = (len - offset).min(dst.len() as u64);
        let want = usize::try_from(want).map_err(|_| FsError::Range)?;
        let want = want.min(self.provider.limits().max_transfer);
        let Some(key) = key else {
            // No placement: the file has never held bytes, so its length is
            // zero and the early return above already covered it.
            return Ok(0);
        };
        Ok(self.provider.read(key, offset, &mut dst[..want]).await?)
    }

    // ── immutable extraction versions ────────────────────────────────────────

    /// Freeze the exact file revision visible through `handle` and lend only
    /// that byte range to an extraction connection.
    ///
    /// The snapshot is copied before the lease becomes visible. Subsequent
    /// writes continue against the file's ordinary placement, so a parser
    /// can never observe a mixture of two revisions.
    pub(crate) async fn lend_version(
        &self,
        owner: ExtractionOwner,
        session: SessionId,
        handle: Handle,
    ) -> Result<(VersionLease, InputIdentity), ExtractionError> {
        self.lend_version_with_rights(owner, session, handle, Rights::READ)
            .await
    }

    /// Freeze one executable source revision.
    ///
    /// Reading a file is intentionally insufficient: the route must also
    /// carry the separately attenuable [`Rights::BIND_EXECUTABLE`] edge.  The
    /// returned version is still only immutable bytes; ImageService performs
    /// independent ELF validation and is the only component that can mint X.
    pub(crate) async fn lend_executable_version(
        &self,
        owner: ExtractionOwner,
        session: SessionId,
        handle: Handle,
    ) -> Result<(VersionLease, InputIdentity), ExtractionError> {
        self.lend_version_with_rights(
            owner,
            session,
            handle,
            Rights::READ.union(Rights::BIND_EXECUTABLE),
        )
        .await
    }

    async fn lend_version_with_rights(
        &self,
        owner: ExtractionOwner,
        session: SessionId,
        handle: Handle,
        required: Rights,
    ) -> Result<(VersionLease, InputIdentity), ExtractionError> {
        let route = self.route(session, handle)?;
        route.rights.require(required)?;
        let (source, len, revision) = {
            let object = self.ns.object(route.object)?;
            if !object.kind().is_file() {
                return Err(FsError::IsDirectory.into());
            }
            (object.storage(), object.len(), object.revision())
        };
        if !self.extraction.can_lend() {
            return Err(ExtractionError::NoSnapshotSlots);
        }

        // The lease borrows the object's live placement: no copy.  It stays
        // frozen because every mutation path runs through `ensure_capacity`,
        // which moves the object to a fresh placement first while a lease
        // borrows this one, and reclaim hands a borrowed placement to its
        // borrowers instead of releasing it.
        let snapshot_key = if len == 0 {
            None
        } else {
            Some(source.ok_or(ExtractionError::Provider(ProviderError::Media))?)
        };

        // Hash the bytes the lease will read: the parser/cache identity is
        // exact, and a short read is the medium's refusal, not a copy's.
        let mut hasher = blake3::Hasher::new();
        if let Some(key) = snapshot_key {
            let mut buffer = [0u8; MAX_TRANSFER];
            let chunk = self.provider.limits().max_transfer.min(buffer.len());
            let mut offset = 0u64;
            while offset < len {
                let want = usize::try_from((len - offset).min(chunk as u64))
                    .map_err(|_| ExtractionError::Range)?;
                let read = self.provider.read(key, offset, &mut buffer[..want]).await?;
                if read != want {
                    return Err(ProviderError::Media.into());
                }
                hasher.update(&buffer[..read]);
                offset += read as u64;
            }
        }
        let identity = InputIdentity {
            source: route.object.to_bits(),
            version: revision,
            len,
            digest: *hasher.finalize().as_bytes(),
        };
        let snapshot = Snapshot {
            owner,
            session,
            handle,
            object: route.object,
            identity,
            storage: snapshot_key,
        };
        match self.extraction.install_snapshot(snapshot) {
            Ok(lease) => Ok((lease, identity)),
            // The placement is the object's own; nothing to release.
            Err(_) => Err(ExtractionError::NoSnapshotSlots),
        }
    }

    /// Read through a frozen lease. No namespace handle is accepted here.
    pub(crate) async fn read_lent_version(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, ExtractionError> {
        let snapshot = self.extraction.snapshot(owner, lease)?;
        let route = self.route(snapshot.session, snapshot.handle)?;
        route.rights.require(Rights::READ)?;
        if route.object != snapshot.object {
            return Err(FsError::AccessLost.into());
        }
        offset
            .checked_add(dst.len() as u64)
            .ok_or(ExtractionError::Range)?;
        if dst.len() > self.provider.limits().max_transfer {
            return Err(ExtractionError::Range);
        }
        if offset >= snapshot.identity.len {
            return Ok(0);
        }
        let want = usize::try_from((snapshot.identity.len - offset).min(dst.len() as u64))
            .map_err(|_| ExtractionError::Range)?;
        let Some(key) = snapshot.storage else {
            return Ok(0);
        };
        let read = self.provider.read(key, offset, &mut dst[..want]).await;
        drop(snapshot);
        Ok(read?)
    }

    /// Publish a validated parser-neutral result only if the lent revision is
    /// still the object's selected version.
    pub(crate) fn commit_derived_result(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
        result: NormalizedResult,
        output_window_bytes: u64,
    ) -> Result<DerivedResultInfo, ExtractionError> {
        let snapshot = self.extraction.snapshot(owner, lease)?;
        let route = self.route(snapshot.session, snapshot.handle)?;
        route.rights.require(Rights::READ)?;
        if route.object != snapshot.object {
            return Err(FsError::AccessLost.into());
        }
        let object = self
            .ns
            .object(snapshot.object)
            .map_err(ExtractionError::from)?;
        let current_revision = Some(object.revision());
        let committed = self.extraction.commit_result(
            owner,
            lease,
            current_revision,
            result,
            output_window_bytes,
        );
        drop(object);
        drop(snapshot);
        committed
    }

    /// Validate and publish one normalized batch together with its durable
    /// provider-backed wire representation. The new placement is complete and
    /// synced before it becomes reachable; the replaced placement is released
    /// only after the in-memory generation points at the successor.
    pub(crate) async fn commit_encoded_derived_result(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
        result: NormalizedResult,
        encoded: &[u8],
        output_window_bytes: u64,
    ) -> Result<DerivedResultInfo, ExtractionError> {
        let snapshot = self.extraction.snapshot(owner, lease)?;
        let route = self.route(snapshot.session, snapshot.handle)?;
        route.rights.require(Rights::READ)?;
        if route.object != snapshot.object {
            return Err(FsError::AccessLost.into());
        }
        let object = snapshot.object;
        let identity = snapshot.identity;
        drop(snapshot);
        kore_svc_extract::registry::Registry::validate(&result, identity, output_window_bytes)
            .map_err(ExtractionError::InvalidResult)?;
        let encoded_len = u32::try_from(encoded.len()).map_err(|_| ExtractionError::Range)?;
        if encoded.is_empty()
            || encoded.len() > self.provider.limits().max_transfer
            || encoded.len() as u64 > self.provider.limits().max_object_bytes
        {
            return Err(ExtractionError::Range);
        }

        let storage = if P::PERSISTS_ACROSS_RESTART {
            let key = self
                .provider
                .allocate(ObjectKind::File, encoded.len() as u64)
                .await?;
            if let Err(error) = self.provider.write(key, 0, encoded).await {
                let _ = self.provider.release(key).await;
                return Err(error.into());
            }
            if let Err(error) = self.provider.sync(key).await {
                let _ = self.provider.release(key).await;
                return Err(error.into());
            }
            Some(key)
        } else {
            None
        };

        let info = match self.commit_derived_result(owner, lease, result, output_window_bytes) {
            Ok(info) => info,
            Err(error) => {
                if let Some(key) = storage {
                    let _ = self.provider.release(key).await;
                }
                return Err(error);
            }
        };
        if let Some(key) = storage {
            let old = match self.extraction.attach_storage(
                object,
                info.parser,
                key,
                encoded_len,
                *blake3::hash(encoded).as_bytes(),
            ) {
                Ok(old) => old,
                Err(error) => {
                    let _ = self.provider.release(key).await;
                    return Err(error.into());
                }
            };
            if let Some(old) = old {
                if self.provider.release(old).await.is_err() {
                    self.ns.note_leaked_placement();
                }
            }
        }
        Ok(info)
    }

    pub(crate) fn durable_derived_results<const N: usize>(
        &self,
    ) -> FsResult<heapless::Vec<crate::extraction::DurableDerived<P::Key>, N>> {
        let records = self.extraction.durable_results::<MAX_DERIVED_RESULTS>()?;
        let mut current = heapless::Vec::new();
        for record in records {
            // A content mutation advances the authoritative object revision
            // before its application-owned extractor can publish the
            // successor batch. That interval is IndexIncomplete, not durable
            // corruption: omit the obsolete derived extent from the new root.
            // CommitDerived will replace the same parser row and reclaim its
            // old placement after the successor is fully synced.
            let Ok(object) = self.ns.object(record.object) else {
                continue;
            };
            let selected = object.kind().is_file()
                && object.revision() == record.input.version
                && object.len() == record.input.len;
            drop(object);
            if selected {
                current.push(record).map_err(|_| FsError::OutOfStorage)?;
            }
        }
        Ok(current)
    }

    pub(crate) fn restore_derived_result(
        &self,
        object: ObjectId,
        result: NormalizedResult,
        storage: P::Key,
        len: u32,
        digest: [u8; 32],
    ) -> FsResult<()> {
        self.extraction
            .restore_result(object, result, storage, len, digest)
    }

    /// Release an immutable lease. A failed provider release retains the
    /// lease so teardown can be retried without losing ownership.
    pub(crate) async fn release_lent_version(
        &self,
        owner: ExtractionOwner,
        lease: VersionLease,
    ) -> Result<(), ExtractionError> {
        // The write claim is the storage-lifetime owner: reads retain a
        // shared snapshot guard across provider I/O, so release receives
        // `Busy` instead of freeing a placement beneath an in-flight read.
        let snapshot = self.extraction.claim_snapshot(owner, lease)?;
        if let Some(key) = snapshot.storage {
            // Release only a placement nobody else holds: not the object
            // (still on it, never mutated in place) and no other lease.
            let owned = self
                .ns
                .object(snapshot.object)
                .map(|object| object.storage() == Some(key))
                .unwrap_or(false);
            if !owned && self.extraction.storage_borrowers(key, Some(lease))? == 0 {
                self.provider.release(key).await?;
            }
        }
        let _ = self.extraction.remove_claimed_snapshot(snapshot)?;
        Ok(())
    }

    /// Selected derived data for an authorized live file route.
    pub fn derived_result(
        &self,
        session: SessionId,
        handle: Handle,
        parser: ParserId,
    ) -> FsResult<Option<NormalizedResult>> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::READ)?;
        let object = self.ns.object(route.object)?;
        let result = self
            .extraction
            .result(route.object, object.revision(), parser);
        drop(object);
        result
    }

    /// Write `src` at `offset`, returning the new revision.
    ///
    /// `expected` is either [`ANY_REVISION`](crate::object::ANY_REVISION) —
    /// unconditional — or an exact revision, which refuses with
    /// [`FsError::Conflict`] if stale.
    pub async fn write_at(
        &self,
        session: SessionId,
        handle: Handle,
        offset: u64,
        src: &[u8],
        expected: Revision,
    ) -> FsResult<Revision> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::WRITE)?;
        self.commit_write(route, Some(offset), src, expected)
            .await
            .map(|(_, revision)| revision)
    }

    /// Append `src` to a file, returning the offset it landed at and the new
    /// revision.
    ///
    /// Choosing the end and writing are one indivisible step: both happen
    /// under the file's exact write claim, so a concurrent append receives
    /// [`FsError::Busy`] rather than landing on the same offset.
    pub async fn append(
        &self,
        session: SessionId,
        handle: Handle,
        src: &[u8],
        expected: Revision,
    ) -> FsResult<(u64, Revision)> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::WRITE)?;
        self.commit_write(route, None, src, expected).await
    }

    /// Set a file's length, returning the new revision.  Growth reads as
    /// zeros; shrinking discards the tail.
    pub async fn resize(
        &self,
        session: SessionId,
        handle: Handle,
        len: u64,
        expected: Revision,
    ) -> FsResult<Revision> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::WRITE)?;
        if len > self.provider.limits().max_object_bytes {
            return Err(FsError::OutOfStorage);
        }
        let mut object = self.ns.object_mut(route.object)?;
        if !object.kind().is_file() {
            return Err(FsError::IsDirectory);
        }
        if !object.revision_admits(expected) {
            return Err(FsError::Conflict);
        }
        let _quota = self.require_growth_with_claimed(
            route,
            &object,
            0,
            0,
            len.saturating_sub(object.len()),
        )?;
        let key = self.ensure_capacity(&mut object, len).await?;
        // Capacity is in place; commit the visible length and revision.
        let _ = key;
        Ok(self.ns.commit_file_len(route.object, &mut object, len))
    }

    /// Cross the provider's durability contract for this file.
    ///
    /// For the RAM provider this reports success without doing anything,
    /// because "visible in this boot" is all RAM promises and a successful
    /// write already achieved it.  It is never reported as more than that.
    pub async fn sync(&self, session: SessionId, handle: Handle) -> FsResult<()> {
        let route = self.route(session, handle)?;
        route.rights.require(Rights::WRITE)?;
        let key = {
            let object = self.ns.object(route.object)?;
            if !object.kind().is_file() {
                return Err(FsError::IsDirectory);
            }
            object.storage()
        };
        match key {
            Some(key) => Ok(self.provider.sync(key).await?),
            // Nothing has ever been placed, so there is nothing to flush.
            None => Ok(()),
        }
    }

    // ── internals ───────────────────────────────────────────────────────────

    /// Revalidate a handle and report what its route permits.  The first act
    /// of every operation.
    fn route(&self, session: SessionId, handle: Handle) -> FsResult<Route> {
        self.auth.revalidate(session, handle, &self.ns)
    }

    /// Refuse graph growth that would outrun this authority edge's budget.
    ///
    /// Usage starts at the edge root, not at the object a child handle names,
    /// so walking deeper cannot reset quota. Diamonds count each object once
    /// and each independently named containment edge once.
    fn require_growth(
        &self,
        route: Route,
        objects: u32,
        edges: u32,
        bytes: u64,
    ) -> FsResult<QuotaClaim<'_>> {
        let epoch = self.auth.quota_epoch(route.grant)?;
        let root = self.auth.validated_edge(route.grant)?.root();
        let usage = self.ns.graph_usage(root)?;
        self.auth.claim_growth(
            route.grant,
            epoch,
            usage.objects,
            usage.edges,
            usage.bytes,
            objects,
            edges,
            bytes,
        )
    }

    /// Reserve growth while one exact object is already write-claimed.
    ///
    /// This is load-bearing for write/append/resize: calculating growth before
    /// the claim lets a concurrent shrink make the later transaction consume
    /// more quota than it reserved. The namespace substitutes `claimed` into
    /// its graph walk, so the exact current length and the reservation remain
    /// one transaction.
    fn require_growth_with_claimed(
        &self,
        route: Route,
        claimed: &Object<P::Key>,
        objects: u32,
        edges: u32,
        bytes: u64,
    ) -> FsResult<QuotaClaim<'_>> {
        let epoch = self.auth.quota_epoch(route.grant)?;
        let root = self.auth.validated_edge(route.grant)?.root();
        let usage = self
            .ns
            .graph_usage_with_claimed(root, route.object, claimed)?;
        self.auth.claim_growth(
            route.grant,
            epoch,
            usage.objects,
            usage.edges,
            usage.bytes,
            objects,
            edges,
            bytes,
        )
    }

    /// Refuse an operation that would span two authority edges.
    ///
    /// This closes a real privilege escalation.  A containment edge carries no
    /// rights of its own — the rights an object is used with come from the
    /// authority edge the *caller entered through*.  So if a session holding a
    /// read-only grant to one of A's files could `link` that object into its
    /// own private root, it would afterwards reach the same object through its
    /// own root's full-rights authority edge: read-only access laundered into
    /// read-write, and access that survives A revoking the grant.  `relocate`
    /// offers the same trick across an attachment boundary.
    ///
    /// Requiring one authority edge for every handle in a
    /// multiple-handle namespace operation removes the laundering step: the
    /// route in and the route out are the same edge, so the rights cannot
    /// change. Cross-attachment moves are therefore refused explicitly rather
    /// than pretending two authority lineages form one commit domain.
    ///
    /// Note this is *not* a restriction on sharing: [`Self::grant`] is how an
    /// object crosses between sessions, and it attenuates instead of
    /// laundering.
    fn require_same_authority(
        &self,
        session: SessionId,
        first: Handle,
        second: Handle,
    ) -> FsResult<()> {
        let a = self.auth.handle(session, first)?.grant();
        let b = self.auth.handle(session, second)?.grant();
        if a != b {
            return Err(FsError::AccessDenied);
        }
        Ok(())
    }

    /// Keep the task-visible root unambiguous.
    ///
    /// A session's view root is the union of its private root's containment
    /// entries and the named authority edges it holds.  The wire protocol
    /// keeps those two edge kinds separate, but ordinary pathname resolution
    /// must not have a precedence rule that silently hides one behind the
    /// other.  Mutations at the private root and incoming grants therefore
    /// share one name set.
    fn require_view_name_free(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
    ) -> FsResult<()> {
        let record = self.auth.handle(session, parent)?;
        if record.grant() == self.auth.root_grant(session)?
            && record.proof().depth() == 0
            && self.auth.display_name_taken(session, name)?
        {
            return Err(FsError::AlreadyExists);
        }
        Ok(())
    }

    /// Whether `name` is already occupied in a session's task-visible root.
    fn view_name_taken(&self, session: SessionId, name: &Name) -> FsResult<bool> {
        if self.auth.display_name_taken(session, name)? {
            return Ok(true);
        }
        let root = self.auth.edge(self.auth.root_grant(session)?)?.root();
        match self.ns.lookup(root, name) {
            Ok(_) => Ok(true),
            Err(FsError::NotFound) => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// Claim the combined top-level name set only when `parent` is the
    /// session's private-root route. Nested directories are already isolated
    /// by their exact directory claims and do not contend on this cache line.
    fn claim_view_if_root(
        &self,
        session: SessionId,
        parent: Handle,
    ) -> FsResult<Option<crate::authority::ViewClaim<'_>>> {
        let record = self.auth.handle(session, parent)?;
        if record.grant() == self.auth.root_grant(session)? && record.proof().depth() == 0 {
            Ok(Some(self.auth.claim_view(session)?))
        } else {
            Ok(None)
        }
    }

    /// The write commit path shared by `write_at` and `append`.
    ///
    /// Ordering is the atomicity argument: validate, reserve capacity, write
    /// bytes, then commit length and revision together.  A failure at any
    /// step before the last leaves nothing visible.
    async fn commit_write(
        &self,
        route: Route,
        requested_offset: Option<u64>,
        src: &[u8],
        expected: Revision,
    ) -> FsResult<(u64, Revision)> {
        if src.len() > self.provider.limits().max_transfer {
            return Err(FsError::Range);
        }
        let mut object = self.ns.object_mut(route.object)?;
        if !object.kind().is_file() {
            return Err(FsError::IsDirectory);
        }
        if !object.revision_admits(expected) {
            return Err(FsError::Conflict);
        }
        let current_len = object.len();
        let offset = requested_offset.unwrap_or(current_len);
        let end = offset.checked_add(src.len() as u64).ok_or(FsError::Range)?;
        if end > self.provider.limits().max_object_bytes {
            return Err(FsError::OutOfStorage);
        }
        let new_len = current_len.max(end);
        let _quota = self.require_growth_with_claimed(
            route,
            &object,
            0,
            0,
            new_len.saturating_sub(current_len),
        )?;

        // Reserve first: this is the step that can run out of room, and it
        // happens while nothing is visible.
        let key = self.ensure_capacity(&mut object, new_len).await?;
        // Then the bytes, into space that is already reserved.
        self.provider.write(key, offset, src).await?;
        // Then, and only then, the visible length and revision.
        let revision = self.ns.commit_file_len(route.object, &mut object, new_len);
        Ok((offset, revision))
    }

    /// Make the object's placement hold at least `len` bytes, allocating one
    /// if it has never had bytes.
    async fn ensure_capacity(
        &self,
        object: &mut crate::object::Object<P::Key>,
        len: u64,
    ) -> FsResult<P::Key> {
        let existing = object.storage();
        match existing {
            Some(key) => {
                let key = self.detach_borrowed(object, key).await?;
                self.provider.resize(key, len).await?;
                Ok(key)
            }
            None => {
                let key = self.provider.allocate(ObjectKind::File, len).await?;
                object.set_storage(key);
                Ok(key)
            }
        }
    }

    /// Before mutating `key` in place, move the object off it if a lease
    /// borrows it: the object continues on a fresh placement holding the
    /// same bytes (the provider's own copy), and the lease keeps the old
    /// one frozen.  The one moment a lend costs a copy — and only when a
    /// write races the lease, which the loader's short lends never do.
    async fn detach_borrowed(
        &self,
        object: &mut crate::object::Object<P::Key>,
        key: P::Key,
    ) -> FsResult<P::Key> {
        if self.extraction.storage_borrowers(key, None)? == 0 {
            return Ok(key);
        }
        let len = object.len();
        let fresh = self.provider.allocate(ObjectKind::File, len).await?;
        if let Err(error) = self.provider.copy_object(key, fresh, len).await {
            if self.provider.release(fresh).await.is_err() {
                self.ns.note_leaked_placement();
            }
            return Err(error.into());
        }
        object.set_storage(fresh);
        Ok(fresh)
    }

    /// Release `start` if nothing reaches it any more, along with anything it
    /// made unreachable, and return their placements to the provider.
    ///
    /// **Every failure here is counted, never raised and never fatal.**
    ///
    /// A provider that refuses a release is counted through
    /// [`Self::leaked_placements`]; a structural refusal or an exhausted
    /// contention budget is counted through [`Self::abandoned_reclaims`] and
    /// abandons only the rest of the walk. Neither is reported to a client,
    /// because the namespace effect that caused the reclamation has already
    /// committed — failing the request would describe an operation that did in
    /// fact happen.
    ///
    /// Neither may panic or spin, either. This runs inside the founded
    /// filesystem service, which has no `role::SELF` to exit through and whose
    /// `COOKIE_REFUSED` reaches no drained channel after readiness: a panic
    /// here parks the whole filesystem permanently with no witness, and every
    /// client then waits on it forever (`SvcClient::call` has no reply
    /// deadline). A resident unreachable subtree is a bounded, reportable
    /// loss; a wedged filesystem is neither.
    async fn reclaim(&self, start: ObjectId) {
        // Each file can retain one selected and one unpublished placement.
        let mut placements: heapless::Vec<P::Key, { 2 * MAX_OBJECTS }> = heapless::Vec::new();
        let mut objects: heapless::Vec<ObjectId, MAX_OBJECTS> = heapless::Vec::new();
        let mut worklist = heapless::Vec::<ObjectId, MAX_OBJECTS>::new();
        if worklist.push(start).is_err() {
            self.ns.note_abandoned_reclaim();
            return;
        }
        // One budget for the whole reclamation, spanning both phases.
        let mut turns: u32 = 0;
        while let Some(object) = worklist.pop() {
            match self.ns.reclaim_one(object, &mut placements, &mut objects) {
                Ok(children) => {
                    for child in children {
                        // The worklist holds one entry per object that crossed
                        // to zero reachability, and an object crosses once, so
                        // MAX_OBJECTS is sufficient by construction. A full
                        // queue would mean that invariant broke: lose the
                        // child, keep draining what is already queued.
                        if worklist.push(child).is_err() {
                            self.ns.note_abandoned_reclaim();
                            break;
                        }
                    }
                }
                // A claim held by another in-flight operation. Keep the exact
                // generation in the bounded worklist and let that operation
                // make progress — for a bounded number of turns.
                Err(FsError::Busy) if turns < RECLAIM_CONTENTION_TURNS => {
                    turns += 1;
                    if worklist.insert(0, object).is_err() {
                        self.ns.note_abandoned_reclaim();
                        break;
                    }
                    yield_once().await;
                }
                // Contention that outlasted the budget, or a structural
                // refusal — a concurrently stale slot (`AccessLost`) or an
                // exhausted arena (`OutOfObjects`), both reachable. The
                // objects below this point stay resident and unreachable.
                Err(_) => {
                    self.ns.note_abandoned_reclaim();
                    break;
                }
            }
        }
        let mut derived_placements = heapless::Vec::<P::Key, MAX_DERIVED_RESULTS>::new();
        for object in objects {
            loop {
                match self.extraction.purge_object(object) {
                    Ok(keys) => {
                        for key in keys {
                            if derived_placements.push(key).is_err() {
                                self.ns.note_abandoned_reclaim();
                                break;
                            }
                        }
                        break;
                    }
                    Err(FsError::Busy) if turns < RECLAIM_CONTENTION_TURNS => {
                        turns += 1;
                        yield_once().await;
                    }
                    Err(_) => {
                        self.ns.note_abandoned_reclaim();
                        break;
                    }
                }
            }
        }
        for key in placements.iter() {
            // A placement a lease still borrows passes to that lease: the
            // last borrower releases it.
            match self.extraction.storage_borrowers(*key, None) {
                Ok(0) => {
                    if self.provider.release(*key).await.is_err() {
                        self.ns.note_leaked_placement();
                    }
                }
                Ok(_) => {}
                Err(_) => self.ns.note_leaked_placement(),
            }
        }
        for key in derived_placements {
            if self.provider.release(key).await.is_err() {
                self.ns.note_leaked_placement();
            }
        }
    }
}

impl Default for RamEngine {
    fn default() -> Self {
        Self::new(RamProvider::new())
    }
}

#[cfg(test)]
mod tests {
    //! Engine-level unit tests.  The plan's full §10.1 matrix lives in the
    //! integration suites under `tests/`; these cover the wiring each method
    //! does — rights gates, kind gates, and the commit ordering — so a
    //! regression in one method is localized here rather than only showing up
    //! as a failed end-to-end scenario.

    use super::*;
    use crate::{block_on, error::ProviderError, object::ANY_REVISION, provider::Fault};
    use kore_svc_extract::{
        registry::{C_JSON, RUST_JSON},
        schema::{FactValue, Reference, SourceRange},
    };
    use kore_svc_extract_worker_c::parse_json_c;
    use kore_svc_extract_worker_rust::{parse_json_rust, parse_plain_text};

    fn name(s: &str) -> Name {
        Name::new(s).unwrap()
    }

    /// An engine with one attached session and an open root handle.
    fn engine() -> (RamEngine, SessionId, Handle) {
        let mut engine = RamEngine::default();
        let session = SessionId::new(0);
        block_on(engine.attach_session(session)).unwrap();
        let root = engine.open_root(session, Rights::ALL).unwrap();
        (engine, session, root.handle)
    }

    #[test]
    fn a_fresh_session_sees_an_empty_root() {
        let (engine, session, root) = engine();
        let info = engine.stat(session, root).unwrap();
        assert_eq!(info.kind, ObjectKind::Directory);
        assert_eq!(info.len, 0);
        let page = engine.list(session, root, 0).unwrap();
        assert!(page.entries.is_empty());
        assert_eq!(page.next, None);
        assert_eq!(engine.live_handles(session), 1);
    }

    #[test]
    fn retained_view_grant_retirement_distinguishes_contention_from_stale_generation() {
        let (engine, session, root) = engine();
        let grant = engine.grant(session, root, session, Rights::READ, None).unwrap();
        let held = engine.auth.edge(grant).unwrap();
        assert_eq!(block_on(engine.retire_view_grant(session, grant)), Err(FsError::Busy));
        assert_eq!(block_on(engine.retire_view_grant(SessionId::new(1), grant)), Err(FsError::AccessDenied));
        drop(held);
        assert_eq!(block_on(engine.retire_view_grant(session, grant)), Ok(()));
        assert_eq!(block_on(engine.retire_view_grant(session, grant)), Ok(()));
        assert_eq!(block_on(engine.revoke(session, grant)), Err(FsError::AccessLost));
        assert!(engine.open_root(session, Rights::ALL).is_ok());
    }

    #[test]
    fn engine_inline_footprint_stays_below_the_founded_stack() {
        let bytes = core::mem::size_of::<RamEngine>();
        std::println!("RamEngine bytes: {bytes}");
        assert!(bytes < 512 * 1024, "inline engine grew to {bytes} bytes");
    }

    #[test]
    fn attaching_twice_is_refused() {
        let (mut engine, session, _root) = engine();
        assert_eq!(
            block_on(engine.attach_session(session)).unwrap_err(),
            FsError::AlreadyExists
        );
    }

    #[test]
    fn an_unattached_session_has_no_root_to_open() {
        let (mut engine, _session, _root) = engine();
        assert_eq!(
            engine
                .open_root(SessionId::new(1), Rights::ALL)
                .unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn create_hands_the_creator_a_usable_handle() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        assert_eq!(made.kind, ObjectKind::File);
        assert_eq!(made.len, 0);
        // Immediately usable, with no second lookup.
        let revision =
            block_on(engine.write_at(session, made.handle, 0, b"kore", ANY_REVISION)).unwrap();
        assert!(revision > made.revision);
        let mut buf = [0u8; 4];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(4)
        );
        assert_eq!(&buf, b"kore");
    }

    #[test]
    fn create_needs_namespace_rights() {
        let (mut engine, session, _root) = engine();
        let read_only = engine.open_root(session, Rights::READ).unwrap();
        assert_eq!(
            block_on(engine.create(session, read_only.handle, &name("f"), ObjectKind::File))
                .unwrap_err(),
            FsError::AccessDenied
        );
        // And nothing was created on the way to the refusal.
        let page = engine.list(session, read_only.handle, 0).unwrap();
        assert!(page.entries.is_empty());
    }

    #[test]
    fn create_beneath_a_file_is_refused() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        assert_eq!(
            block_on(engine.create(session, file.handle, &name("x"), ObjectKind::File))
                .unwrap_err(),
            FsError::NotDirectory
        );
    }

    #[test]
    fn create_with_a_duplicate_name_leaves_nothing_behind() {
        let (mut engine, session, root) = engine();
        block_on(engine.create(session, root, &name("dup"), ObjectKind::File)).unwrap();
        let before = engine.object_count();
        assert_eq!(
            block_on(engine.create(session, root, &name("dup"), ObjectKind::File)).unwrap_err(),
            FsError::AlreadyExists
        );
        assert_eq!(
            engine.object_count(),
            before,
            "the refused create must not leak an object"
        );
    }

    #[test]
    fn writes_are_gated_on_write_rights() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let reader = engine
            .open_at(session, root, &name("f"), Rights::READ)
            .unwrap();
        assert_eq!(
            block_on(engine.write_at(session, reader.handle, 0, b"no", ANY_REVISION)).unwrap_err(),
            FsError::AccessDenied
        );
        // The writable handle still works.
        assert!(block_on(engine.write_at(session, made.handle, 0, b"ok", ANY_REVISION)).is_ok());
    }

    #[test]
    fn reads_and_writes_refuse_a_directory() {
        let (mut engine, session, root) = engine();
        let mut buf = [0u8; 4];
        assert_eq!(
            block_on(engine.read(session, root, 0, &mut buf)).unwrap_err(),
            FsError::IsDirectory
        );
        assert_eq!(
            block_on(engine.write_at(session, root, 0, b"x", ANY_REVISION)).unwrap_err(),
            FsError::IsDirectory
        );
        assert_eq!(
            block_on(engine.resize(session, root, 4, ANY_REVISION)).unwrap_err(),
            FsError::IsDirectory
        );
        assert_eq!(
            block_on(engine.append(session, root, b"x", ANY_REVISION)).unwrap_err(),
            FsError::IsDirectory
        );
    }

    #[test]
    fn listing_refuses_a_file_and_reading_refuses_a_directory() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        assert_eq!(
            engine.list(session, file.handle, 0).unwrap_err(),
            FsError::NotDirectory
        );
    }

    #[test]
    fn a_hole_left_by_a_sparse_write_reads_as_zeros() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 100, b"tail", ANY_REVISION)).unwrap();

        let mut buf = [0xffu8; 104];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(104)
        );
        assert!(
            buf[..100].iter().all(|b| *b == 0),
            "the hole reads as zeros"
        );
        assert_eq!(&buf[100..], b"tail");
    }

    #[test]
    fn reads_past_the_end_are_short_then_empty() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 0, b"abcd", ANY_REVISION)).unwrap();

        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(engine.read(session, made.handle, 2, &mut buf)),
            Ok(2)
        );
        assert_eq!(&buf[..2], b"cd");
        assert_eq!(
            block_on(engine.read(session, made.handle, 4, &mut buf)),
            Ok(0)
        );
        assert_eq!(
            block_on(engine.read(session, made.handle, 99, &mut buf)),
            Ok(0)
        );
    }

    #[test]
    fn a_never_written_file_reads_empty() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let mut buf = [0u8; 4];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(0)
        );
        // And it cost the provider nothing.
        assert_eq!(engine.provider().chunks_in_use(), 0);
    }

    #[test]
    fn offset_plus_length_overflow_is_refused() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(engine.read(session, made.handle, u64::MAX, &mut buf)).unwrap_err(),
            FsError::Range
        );
        assert_eq!(
            block_on(engine.write_at(session, made.handle, u64::MAX, b"x", ANY_REVISION))
                .unwrap_err(),
            FsError::Range
        );
    }

    #[test]
    fn an_oversized_transfer_is_refused() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let big = [0u8; crate::bounds::MAX_TRANSFER + 1];
        assert_eq!(
            block_on(engine.write_at(session, made.handle, 0, &big, ANY_REVISION)).unwrap_err(),
            FsError::Range
        );
    }

    #[test]
    fn a_file_larger_than_the_provider_allows_is_refused_as_storage() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let ceiling = engine.provider().limits().max_object_bytes;
        assert_eq!(
            block_on(engine.resize(session, made.handle, ceiling + 1, ANY_REVISION)).unwrap_err(),
            FsError::OutOfStorage
        );
        // The refusal left the file as it was.
        assert_eq!(engine.stat(session, made.handle).unwrap().len, 0);
    }

    #[test]
    fn resize_grows_with_zeros_and_shrinks_by_discarding() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 0, b"abcdefgh", ANY_REVISION)).unwrap();

        block_on(engine.resize(session, made.handle, 4, ANY_REVISION)).unwrap();
        assert_eq!(engine.stat(session, made.handle).unwrap().len, 4);
        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(4)
        );
        assert_eq!(&buf[..4], b"abcd");

        block_on(engine.resize(session, made.handle, 8, ANY_REVISION)).unwrap();
        let mut buf = [0xffu8; 8];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(8)
        );
        assert_eq!(&buf[..4], b"abcd");
        assert!(buf[4..].iter().all(|b| *b == 0), "regrowth reads as zeros");
    }

    #[test]
    fn sync_reports_success_without_promising_more() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        // Before any bytes there is no placement, and that is still a success.
        assert!(block_on(engine.sync(session, made.handle)).is_ok());
        block_on(engine.write_at(session, made.handle, 0, b"x", ANY_REVISION)).unwrap();
        assert!(block_on(engine.sync(session, made.handle)).is_ok());
    }

    #[test]
    fn sync_needs_write_rights() {
        let (mut engine, session, root) = engine();
        block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let reader = engine
            .open_at(session, root, &name("f"), Rights::READ)
            .unwrap();
        assert_eq!(
            block_on(engine.sync(session, reader.handle)).unwrap_err(),
            FsError::AccessDenied
        );
    }

    #[test]
    fn a_failed_provider_write_leaves_no_visible_effect() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let committed =
            block_on(engine.write_at(session, made.handle, 0, b"original", ANY_REVISION)).unwrap();

        engine.provider().inject(Fault::Write(ProviderError::Media));
        assert_eq!(
            block_on(engine.write_at(session, made.handle, 0, b"replaced", ANY_REVISION))
                .unwrap_err(),
            FsError::ProviderFailure(ProviderError::Media)
        );

        // Length, revision, and bytes are all as they were.
        let info = engine.stat(session, made.handle).unwrap();
        assert_eq!(info.len, 8);
        assert_eq!(info.revision, committed);
        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(engine.read(session, made.handle, 0, &mut buf)),
            Ok(8)
        );
        assert_eq!(&buf, b"original");
    }

    #[test]
    fn close_drops_the_handle_but_not_the_object() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 0, b"keep", ANY_REVISION)).unwrap();
        engine.close(session, made.handle).unwrap();

        assert_eq!(
            engine.stat(session, made.handle).unwrap_err(),
            FsError::InvalidHandle
        );
        // Reopening through the still-live name finds the same bytes.
        let again = engine
            .open_at(session, root, &name("f"), Rights::ALL)
            .unwrap();
        assert_eq!(again.len, 4);
        let mut buf = [0u8; 4];
        assert_eq!(
            block_on(engine.read(session, again.handle, 0, &mut buf)),
            Ok(4)
        );
        assert_eq!(&buf, b"keep");
    }

    #[test]
    fn a_closed_handle_cannot_be_closed_twice() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        engine.close(session, made.handle).unwrap();
        assert_eq!(
            engine.close(session, made.handle).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn opening_more_rights_than_the_route_holds_is_refused() {
        let (mut engine, session, root) = engine();
        block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        let read_only_root = engine.open_root(session, Rights::READ).unwrap();
        assert_eq!(
            engine
                .open_at(session, read_only_root.handle, &name("f"), Rights::WRITE)
                .unwrap_err(),
            FsError::AccessDenied
        );
    }

    #[test]
    fn a_traversal_deeper_than_the_proof_bound_is_refused() {
        let (mut engine, session, root) = engine();
        // Build a chain as deep as the proof allows, then one more.
        let mut parent = root;
        for depth in 0..MAX_PATH_DEPTH {
            let made = block_on(engine.create(session, parent, &name("d"), ObjectKind::Directory));
            match made {
                Ok(made) => parent = made.handle,
                Err(err) => {
                    // The directory arena may bound this before the proof
                    // does; either way it must be a typed refusal.
                    assert!(
                        matches!(err, FsError::Range | FsError::OutOfObjects),
                        "unexpected refusal at depth {depth}: {err:?}"
                    );
                    return;
                }
            }
        }
        // At the bound, a further create must refuse rather than record a
        // truncated proof.
        assert_eq!(
            block_on(engine.create(session, parent, &name("d"), ObjectKind::Directory))
                .unwrap_err(),
            FsError::Range
        );
    }

    #[test]
    fn detaching_a_session_reclaims_its_whole_root() {
        let (mut engine, session, root) = engine();
        let dir =
            block_on(engine.create(session, root, &name("d"), ObjectKind::Directory)).unwrap();
        let file =
            block_on(engine.create(session, dir.handle, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"bytes", ANY_REVISION)).unwrap();
        assert!(engine.provider().chunks_in_use() > 0);

        block_on(engine.detach_session(session)).unwrap();
        assert_eq!(engine.object_count(), 0, "root, dir, and file all released");
        assert_eq!(engine.provider().chunks_in_use(), 0);
        assert_eq!(engine.provider().placements(), 0);
        assert_eq!(engine.leaked_placements(), 0);
        // Every handle it held is gone.
        assert_eq!(
            engine.stat(session, root).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    /// Contention must not be able to hang the filesystem.
    ///
    /// Before this was bounded, a claim that outlived the turn it was taken in
    /// made `reclaim` retry forever: the founded service livelocked, and
    /// because it can neither exit nor report after readiness, the only
    /// symptom was a machine that stopped answering. The harness reproduces
    /// exactly that — `block_on` panics after 1024 polls — so an unbounded
    /// retry fails this test by hanging rather than by asserting.
    #[test]
    fn contention_that_outlives_its_budget_is_abandoned_not_awaited_forever() {
        let (engine, session, _root) = engine();
        let root_id = engine.root_object(session).unwrap();

        // Exactly the contention a concurrent operation creates: every
        // `reclaim_one` on this object now answers `Busy`.
        let held = engine
            .ns
            .claim_object(root_id)
            .expect("a fresh root object is claimable");

        block_on(engine.reclaim(root_id));

        // It gave up rather than spinning, and said so.
        assert_eq!(engine.abandoned_reclaims(), 1);
        // The object is still there — abandoning leaves it resident and
        // unreachable, which is the whole bounded loss being traded for.
        drop(held);
        assert!(engine.object_count() > 0);
        // And the engine still serves: the wedge is what we were avoiding.
        assert!(engine.stat(session, _root).is_ok());
    }

    /// The budget is spent, not per-object: a reclaim that already burned it
    /// on one contended object does not then spend it again on the next.
    #[test]
    fn an_uncontended_reclaim_still_completes_and_counts_nothing() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 0, b"bytes", ANY_REVISION)).unwrap();

        assert!(block_on(engine.unlink(session, root, &name("f"))).is_ok());
        assert_eq!(engine.abandoned_reclaims(), 0);
        assert_eq!(engine.leaked_placements(), 0);
        assert_eq!(engine.provider().chunks_in_use(), 0);
    }

    #[test]
    fn a_refused_release_is_counted_not_reported() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create(session, root, &name("f"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, made.handle, 0, b"bytes", ANY_REVISION)).unwrap();

        engine
            .provider()
            .inject(Fault::Release(ProviderError::Media));
        // The unlink genuinely happened, so it reports success…
        assert!(block_on(engine.unlink(session, root, &name("f"))).is_ok());
        // …and the placement the provider would not take back is accounted
        // for rather than silently forgotten.
        assert_eq!(engine.leaked_placements(), 1);
        assert_eq!(
            engine
                .open_at(session, root, &name("f"), Rights::ALL)
                .unwrap_err(),
            FsError::NotFound
        );
    }

    #[test]
    fn a_grant_cannot_hide_a_private_root_name() {
        let (mut engine, source, source_root) = engine();
        let destination = SessionId::new(1);
        block_on(engine.attach_session(destination)).unwrap();
        let destination_root = engine.open_root(destination, Rights::ALL).unwrap().handle;
        block_on(engine.create(
            destination,
            destination_root,
            &name("shared"),
            ObjectKind::File,
        ))
        .unwrap();
        let source_file =
            block_on(engine.create(source, source_root, &name("source"), ObjectKind::File))
                .unwrap();

        assert_eq!(
            engine
                .grant(
                    source,
                    source_file.handle,
                    destination,
                    Rights::READ,
                    Some(name("shared")),
                )
                .unwrap_err(),
            FsError::AlreadyExists
        );
    }

    #[test]
    fn scoped_grant_enforces_graph_byte_and_handle_budgets() {
        let (mut engine, source, source_root) = engine();
        let destination = SessionId::new(1);
        block_on(engine.attach_session(destination)).unwrap();
        let quota = AuthorityQuota::new(2, 1, 3, 2);
        let grant = engine
            .grant_scoped(
                source,
                source_root,
                destination,
                Rights::ALL,
                quota,
                Some(name("data")),
            )
            .unwrap();
        let root = engine.open_grant(destination, grant, Rights::ALL).unwrap();
        let file =
            block_on(engine.create(destination, root.handle, &name("one"), ObjectKind::File))
                .unwrap();
        block_on(engine.write_at(destination, file.handle, 0, b"abc", ANY_REVISION)).unwrap();
        assert_eq!(
            block_on(engine.write_at(destination, file.handle, 3, b"d", ANY_REVISION)).unwrap_err(),
            FsError::OutOfStorage
        );
        assert_eq!(
            block_on(engine.create(destination, root.handle, &name("two"), ObjectKind::File,))
                .unwrap_err(),
            FsError::OutOfObjects
        );
        assert_eq!(
            engine
                .list(destination, root.handle, 0)
                .unwrap()
                .entries
                .len(),
            1
        );
    }

    #[test]
    fn a_file_claim_scopes_contention_and_quota_to_that_exact_object() {
        let (engine, source, source_root) = engine();
        let destination = SessionId::new(1);
        block_on(engine.attach_session(destination)).unwrap();
        let grant = engine
            .grant_scoped(
                source,
                source_root,
                destination,
                Rights::ALL,
                AuthorityQuota::new(3, 2, 3, 3),
                Some(name("data")),
            )
            .unwrap();
        let root = engine.open_grant(destination, grant, Rights::ALL).unwrap();
        let first =
            block_on(engine.create(destination, root.handle, &name("first"), ObjectKind::File))
                .unwrap();
        let second =
            block_on(engine.create(destination, root.handle, &name("second"), ObjectKind::File))
                .unwrap();

        let route = engine.route(destination, first.handle).unwrap();
        let claimed = engine.ns.object_mut(route.object).unwrap();
        let quota = engine
            .require_growth_with_claimed(route, &claimed, 0, 0, 3)
            .unwrap();

        assert_eq!(
            block_on(engine.append(destination, first.handle, b"x", ANY_REVISION)).unwrap_err(),
            FsError::Busy,
            "the same file reports typed retryable contention"
        );
        assert!(
            block_on(engine.append(destination, second.handle, b"", ANY_REVISION)).is_ok(),
            "an unrelated file remains independently usable"
        );
        assert!(matches!(
            engine.require_growth_with_claimed(route, &claimed, 0, 0, 1),
            Err(FsError::OutOfStorage)
        ));

        drop(quota);
        drop(claimed);
        assert!(block_on(engine.append(destination, first.handle, b"abc", ANY_REVISION)).is_ok());
    }

    #[test]
    fn scoped_quota_cannot_widen_while_delegating() {
        let (mut engine, source, source_root) = engine();
        let middle = SessionId::new(1);
        let destination = SessionId::new(2);
        block_on(engine.attach_session(middle)).unwrap();
        block_on(engine.attach_session(destination)).unwrap();
        let quota = AuthorityQuota::new(4, 4, 64, 4);
        let first = engine
            .grant_scoped(source, source_root, middle, Rights::ALL, quota, None)
            .unwrap();
        let middle_root = engine.open_grant(middle, first, Rights::ALL).unwrap();
        assert_eq!(
            engine
                .grant_scoped(
                    middle,
                    middle_root.handle,
                    destination,
                    Rights::READ,
                    AuthorityQuota::new(5, 4, 64, 4),
                    None,
                )
                .unwrap_err(),
            FsError::AccessDenied
        );
    }

    #[test]
    fn detached_temporary_root_lives_exactly_as_long_as_its_grant() {
        let (mut engine, launcher, launcher_root) = engine();
        let task = SessionId::new(1);
        block_on(engine.attach_session(task)).unwrap();
        let tmp = block_on(engine.create(
            launcher,
            launcher_root,
            &name("staging"),
            ObjectKind::Directory,
        ))
        .unwrap();
        let grant = engine
            .grant_scoped(
                launcher,
                tmp.handle,
                task,
                Rights::ALL,
                AuthorityQuota::new(4, 4, 64, 4),
                Some(name("tmp")),
            )
            .unwrap();

        block_on(engine.detach_edge(launcher, launcher_root, &name("staging"))).unwrap();
        engine.close(launcher, tmp.handle).unwrap();
        assert_eq!(
            engine
                .open_at(launcher, launcher_root, &name("staging"), Rights::READ)
                .unwrap_err(),
            FsError::NotFound,
            "the launcher-only containment label is gone"
        );

        let task_tmp = engine.open_grant(task, grant, Rights::ALL).unwrap();
        block_on(engine.create(task, task_tmp.handle, &name("scratch"), ObjectKind::File)).unwrap();
        assert_eq!(engine.object_count(), 4, "two private roots plus tmp graph");

        block_on(engine.detach_session(task)).unwrap();
        assert_eq!(
            engine.object_count(),
            1,
            "retiring the task grant reclaims the detached temporary graph"
        );
    }

    #[test]
    fn a_private_root_mutation_cannot_hide_a_grant_name() {
        let (mut engine, source, source_root) = engine();
        let destination = SessionId::new(1);
        block_on(engine.attach_session(destination)).unwrap();
        let destination_root = engine.open_root(destination, Rights::ALL).unwrap().handle;
        let source_file =
            block_on(engine.create(source, source_root, &name("source"), ObjectKind::File))
                .unwrap();
        engine
            .grant(
                source,
                source_file.handle,
                destination,
                Rights::READ,
                Some(name("shared")),
            )
            .unwrap();

        assert_eq!(
            block_on(engine.create(
                destination,
                destination_root,
                &name("shared"),
                ObjectKind::File,
            ))
            .unwrap_err(),
            FsError::AlreadyExists
        );

        let local = block_on(engine.create(
            destination,
            destination_root,
            &name("local"),
            ObjectKind::File,
        ))
        .unwrap();
        assert_eq!(
            engine
                .link(destination, destination_root, &name("shared"), local.handle)
                .unwrap_err(),
            FsError::AlreadyExists
        );
        assert_eq!(
            engine
                .rename(
                    destination,
                    destination_root,
                    &name("local"),
                    &name("shared"),
                )
                .unwrap_err(),
            FsError::AlreadyExists
        );
    }

    /// A lend borrows the object's live placement — no copy — and stays
    /// frozen: a write during the lease moves the object to a fresh
    /// placement, an unlink hands the placement to the lease, and the last
    /// release returns it.  Placement counts prove which path ran.
    #[test]
    fn a_lend_borrows_the_placement_and_only_a_racing_write_copies() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("img"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"abcdef", ANY_REVISION)).unwrap();
        let owner = ExtractionOwner::new(3);
        let baseline = engine.provider().placements();

        // Lend and release without a write: no placement ever appears.
        let (lease, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        assert_eq!(
            engine.provider().placements(),
            baseline,
            "a lend allocates nothing"
        );
        let mut back = [0u8; 6];
        assert_eq!(
            block_on(engine.read_lent_version(owner, lease, 0, &mut back)),
            Ok(6)
        );
        assert_eq!(&back, b"abcdef");
        block_on(engine.release_lent_version(owner, lease)).unwrap();
        assert_eq!(
            engine.provider().placements(),
            baseline,
            "the object keeps its placement"
        );
        let mut live = [0u8; 6];
        assert_eq!(
            block_on(engine.read(session, file.handle, 0, &mut live)),
            Ok(6)
        );
        assert_eq!(&live, b"abcdef");

        // A write during the lease detaches: one fresh placement for the
        // object, the lease frozen on the old one, released with the lease.
        let (lease, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"ABCDEF", ANY_REVISION)).unwrap();
        assert_eq!(engine.provider().placements(), baseline + 1);
        assert_eq!(
            block_on(engine.read_lent_version(owner, lease, 0, &mut back)),
            Ok(6)
        );
        assert_eq!(&back, b"abcdef");
        assert_eq!(
            block_on(engine.read(session, file.handle, 0, &mut live)),
            Ok(6)
        );
        assert_eq!(&live, b"ABCDEF");
        // A second write during the same lease costs nothing more.
        block_on(engine.write_at(session, file.handle, 0, b"abCDEF", ANY_REVISION)).unwrap();
        assert_eq!(engine.provider().placements(), baseline + 1);
        block_on(engine.release_lent_version(owner, lease)).unwrap();
        assert_eq!(engine.provider().placements(), baseline);
        assert_eq!(engine.leaked_placements(), 0);

        // Two leases on one placement: a write detaches once; the first
        // release keeps the placement for the second, the last frees it.
        let (first, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        let (second, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"xxxxxx", ANY_REVISION)).unwrap();
        assert_eq!(engine.provider().placements(), baseline + 1);
        block_on(engine.release_lent_version(owner, first)).unwrap();
        assert_eq!(engine.provider().placements(), baseline + 1);
        assert_eq!(
            block_on(engine.read_lent_version(owner, second, 0, &mut back)),
            Ok(6)
        );
        assert_eq!(&back, b"abCDEF");
        block_on(engine.release_lent_version(owner, second)).unwrap();
        assert_eq!(engine.provider().placements(), baseline);

        // Reclaimed during the lease: unlinked and closed, the object is
        // gone, but its placement passes to the lease rather than being
        // released beneath it, and the lease's release returns it.
        let (lease, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        assert!(block_on(engine.unlink(session, root, &name("img"))).is_ok());
        engine.close(session, file.handle).unwrap();
        assert_eq!(
            engine.provider().placements(),
            baseline,
            "not released beneath the lease"
        );
        block_on(engine.release_lent_version(owner, lease)).unwrap();
        assert_eq!(engine.provider().placements(), baseline - 1);
        assert_eq!(engine.leaked_placements(), 0);
    }

    #[test]
    fn lent_version_is_frozen_across_writes_and_bound_to_its_owner() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("doc"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, br#"{"old":1}"#, ANY_REVISION)).unwrap();
        let owner = ExtractionOwner::new(7);
        let stranger = ExtractionOwner::new(8);
        let (lease, identity) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();

        block_on(engine.write_at(session, file.handle, 0, br#"{"new":2}"#, ANY_REVISION)).unwrap();
        let mut frozen = [0u8; 9];
        assert_eq!(
            block_on(engine.read_lent_version(owner, lease, 0, &mut frozen)),
            Ok(9)
        );
        assert_eq!(&frozen, br#"{"old":1}"#);
        assert_eq!(identity.digest, *blake3::hash(&frozen).as_bytes());
        assert_eq!(
            block_on(engine.read_lent_version(stranger, lease, 0, &mut frozen)),
            Err(ExtractionError::InvalidLease)
        );

        block_on(engine.release_lent_version(owner, lease)).unwrap();
        assert_eq!(
            block_on(engine.read_lent_version(owner, lease, 0, &mut frozen)),
            Err(ExtractionError::InvalidLease)
        );
        // Cancelling extraction changed neither the current bytes nor its
        // namespace identity.
        let mut current = [0u8; 9];
        block_on(engine.read(session, file.handle, 0, &mut current)).unwrap();
        assert_eq!(&current, br#"{"new":2}"#);
    }

    #[test]
    fn create_stamps_a_subtype_that_open_stat_and_list_report() {
        let (mut engine, session, root) = engine();
        let made = block_on(engine.create_typed(
            session,
            root,
            &name("tool"),
            ObjectKind::File,
            ObjectSubtype::EXECUTABLE_IMAGE,
        ))
        .unwrap();
        assert_eq!(made.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
        let plain =
            block_on(engine.create(session, root, &name("note"), ObjectKind::File)).unwrap();
        assert_eq!(plain.subtype, ObjectSubtype::PLAIN);

        let reopened = engine
            .open_at(session, root, &name("tool"), Rights::READ)
            .unwrap();
        assert_eq!(reopened.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(
            engine.stat(session, reopened.handle).unwrap().subtype,
            ObjectSubtype::EXECUTABLE_IMAGE
        );
        let page = engine.list(session, root, 0).unwrap();
        let tool = page
            .entries
            .iter()
            .find(|entry| entry.name == name("tool"))
            .unwrap();
        assert_eq!(tool.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
        let note = page
            .entries
            .iter()
            .find(|entry| entry.name == name("note"))
            .unwrap();
        assert_eq!(note.subtype, ObjectSubtype::PLAIN);
    }

    /// A directory carries no subtype, and the refusal leaves nothing behind.
    #[test]
    fn a_directory_refuses_a_subtype() {
        let (mut engine, session, root) = engine();
        assert_eq!(
            block_on(engine.create_typed(
                session,
                root,
                &name("dir"),
                ObjectKind::Directory,
                ObjectSubtype::EXECUTABLE_IMAGE,
            ))
            .unwrap_err(),
            FsError::Unsupported
        );
        assert!(engine.list(session, root, 0).unwrap().entries.is_empty());
        assert_eq!(engine.live_handles(session), 1);
    }

    /// The stamp is data: an executable-image file is opened, read and
    /// written like any other, and it is *not* executable-source authority —
    /// that stays with `BIND_EXECUTABLE` on the route.
    #[test]
    fn a_subtype_grants_no_rights() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create_typed(
            session,
            root,
            &name("tool"),
            ObjectKind::File,
            ObjectSubtype::EXECUTABLE_IMAGE,
        ))
        .unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"ELF", ANY_REVISION)).unwrap();
        let readable = engine
            .open_at(session, root, &name("tool"), Rights::READ)
            .unwrap();
        assert_eq!(readable.subtype, ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(
            block_on(engine.lend_executable_version(
                ExtractionOwner::new(21),
                session,
                readable.handle
            )),
            Err(ExtractionError::Filesystem(FsError::AccessDenied)),
            "a subtype is data; it never stands in for BIND_EXECUTABLE"
        );
    }

    /// An engine whose volume predates subtypes refuses to hold a fact the
    /// disk would drop; plain creates are unaffected.
    #[test]
    fn a_legacy_version_engine_refuses_subtyped_creates() {
        let (mut engine, session, root) = engine();
        engine.durable_subtypes = false;
        assert_eq!(
            block_on(engine.create_typed(
                session,
                root,
                &name("tool"),
                ObjectKind::File,
                ObjectSubtype::EXECUTABLE_IMAGE,
            ))
            .unwrap_err(),
            FsError::Unsupported
        );
        let plain =
            block_on(engine.create(session, root, &name("note"), ObjectKind::File)).unwrap();
        assert_eq!(plain.subtype, ObjectSubtype::PLAIN);
    }

    #[test]
    fn executable_lending_requires_its_separate_route_authority() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("tool"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"ELF", ANY_REVISION)).unwrap();

        let visible = engine
            .open_at(session, root, &name("tool"), Rights::READ)
            .expect("ordinary readable view");
        let owner = ExtractionOwner::new(17);
        assert_eq!(
            block_on(engine.lend_executable_version(owner, session, visible.handle)),
            Err(ExtractionError::Filesystem(FsError::AccessDenied)),
            "readable bytes alone are not executable-source authority"
        );

        let bindable = engine
            .open_at(
                session,
                root,
                &name("tool"),
                Rights::READ.union(Rights::BIND_EXECUTABLE),
            )
            .expect("explicit executable source view");
        let (lease, identity) =
            block_on(engine.lend_executable_version(owner, session, bindable.handle))
                .expect("binding authority freezes the exact revision");
        assert_eq!(identity.len, 3);
        block_on(engine.release_lent_version(owner, lease)).unwrap();
    }

    #[test]
    fn a_home_face_cannot_nominate_its_own_bytes_as_executable_source() {
        // The self-bless hole, closed.  A task's writable home is `HOME`,
        // which withholds `BIND_EXECUTABLE`.  It may create and write files —
        // but it can never open ANY of them with the executable-source right,
        // so it cannot freeze its own output as launch authority; that stays
        // the install authority's (`ALL`) privilege.
        let mut engine = RamEngine::default();
        let session = SessionId::new(0);
        block_on(engine.attach_session(session)).unwrap();
        let home = engine.open_root(session, Rights::HOME).unwrap().handle;

        // Writing works — HOME carries WRITE and MUTATE_NAMESPACE.
        let file = block_on(engine.create(session, home, &name("tool"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"ELF", ANY_REVISION)).unwrap();

        // Nominating does not — the route lacks BIND_EXECUTABLE, so the
        // executable-source open is refused before any binding can begin.
        assert!(
            engine
                .open_at(
                    session,
                    home,
                    &name("tool"),
                    Rights::READ.union(Rights::BIND_EXECUTABLE),
                )
                .is_err(),
            "a writable home must not open its own bytes as executable source"
        );
    }

    #[test]
    fn task_local_executable_routes_freeze_distinct_revisions() {
        let (mut engine, first, first_root) = engine();
        let second = SessionId::new(1);
        block_on(engine.attach_session(second)).unwrap();
        let second_root = engine.open_root(second, Rights::ALL).unwrap().handle;

        // Local display names are not executable authority. Each task opens
        // the object through its own native App grant and the exact immutable
        // revision is what later becomes launch authority.
        let tool = name("tool");

        let first_file =
            block_on(engine.create(first, first_root, &tool, ObjectKind::File)).unwrap();
        let second_file =
            block_on(engine.create(second, second_root, &tool, ObjectKind::File)).unwrap();
        block_on(engine.write_at(first, first_file.handle, 0, b"task-a", ANY_REVISION)).unwrap();
        block_on(engine.write_at(second, second_file.handle, 0, b"task-b", ANY_REVISION)).unwrap();

        let source_rights = Rights::READ.union(Rights::BIND_EXECUTABLE);
        let first_source = engine
            .open_at(first, first_root, &tool, source_rights)
            .unwrap();
        let second_source = engine
            .open_at(second, second_root, &tool, source_rights)
            .unwrap();
        let first_owner = ExtractionOwner::new(21);
        let second_owner = ExtractionOwner::new(22);
        let (first_lease, first_identity) =
            block_on(engine.lend_executable_version(first_owner, first, first_source.handle))
                .unwrap();
        let (second_lease, second_identity) =
            block_on(engine.lend_executable_version(second_owner, second, second_source.handle))
                .unwrap();

        assert_ne!(first_identity.source, second_identity.source);
        assert_ne!(first_identity.digest, second_identity.digest);

        // Replacing the first task's selected file creates a new source
        // identity, but the old lease remains an immutable launch source.
        block_on(engine.write_at(first, first_file.handle, 0, b"task-c", ANY_REVISION)).unwrap();
        let (replacement_lease, replacement_identity) =
            block_on(engine.lend_executable_version(first_owner, first, first_source.handle))
                .unwrap();
        assert_eq!(replacement_identity.source, first_identity.source);
        assert!(replacement_identity.version > first_identity.version);
        assert_ne!(replacement_identity.digest, first_identity.digest);

        let mut old = [0u8; 6];
        let mut replacement = [0u8; 6];
        let mut other = [0u8; 6];
        assert_eq!(
            block_on(engine.read_lent_version(first_owner, first_lease, 0, &mut old)),
            Ok(old.len())
        );
        assert_eq!(
            block_on(
                engine.read_lent_version(first_owner, replacement_lease, 0, &mut replacement,)
            ),
            Ok(replacement.len())
        );
        assert_eq!(
            block_on(engine.read_lent_version(second_owner, second_lease, 0, &mut other)),
            Ok(other.len())
        );
        assert_eq!(&old, b"task-a");
        assert_eq!(&replacement, b"task-c");
        assert_eq!(&other, b"task-b");

        block_on(engine.release_lent_version(first_owner, first_lease)).unwrap();
        block_on(engine.release_lent_version(first_owner, replacement_lease)).unwrap();
        block_on(engine.release_lent_version(second_owner, second_lease)).unwrap();
    }

    #[test]
    fn revoked_source_authority_blocks_lent_reads_and_result_commit_but_not_release() {
        let (mut engine, source, source_root) = engine();
        let destination = SessionId::new(1);
        block_on(engine.attach_session(destination)).unwrap();
        let file =
            block_on(engine.create(source, source_root, &name("shared.json"), ObjectKind::File))
                .unwrap();
        let bytes = br#"{"shared":true}"#;
        block_on(engine.write_at(source, file.handle, 0, bytes, ANY_REVISION)).unwrap();
        let grant = engine
            .grant(source, file.handle, destination, Rights::READ, None)
            .unwrap();
        let shared = engine.open_grant(destination, grant, Rights::READ).unwrap();
        let owner = ExtractionOwner::new(u64::from(destination.index()));
        let (lease, input) =
            block_on(engine.lend_version(owner, destination, shared.handle)).unwrap();
        let result = parse_json_rust(input, bytes).unwrap();

        block_on(engine.revoke(source, grant)).unwrap();
        let mut frozen = [0u8; 15];
        assert_eq!(
            block_on(engine.read_lent_version(owner, lease, 0, &mut frozen)),
            Err(ExtractionError::Filesystem(FsError::AccessLost))
        );
        assert_eq!(
            engine.commit_derived_result(owner, lease, result, 0),
            Err(ExtractionError::Filesystem(FsError::AccessLost))
        );

        // Revocation cancels authority to use the frozen version, but never
        // strands its storage: the stamped owner can still release it.
        block_on(engine.release_lent_version(owner, lease)).unwrap();
        let mut current = [0u8; 15];
        assert_eq!(
            block_on(engine.read(source, file.handle, 0, &mut current)),
            Ok(bytes.len())
        );
        assert_eq!(&current, bytes);
        assert_eq!(engine.extraction.result_count(), 0);
    }

    #[test]
    fn result_publication_is_exact_validated_and_obsolete_versions_fail_closed() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("doc"), ObjectKind::File)).unwrap();
        let bytes = br#"{"answer":42}"#;
        block_on(engine.write_at(session, file.handle, 0, bytes, ANY_REVISION)).unwrap();
        let owner = ExtractionOwner::new(11);
        let (lease, input) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();

        let valid = parse_json_rust(input, bytes).unwrap();
        let info = engine
            .commit_derived_result(owner, lease, valid, 0)
            .unwrap();
        assert_eq!(info.input, input);
        assert_eq!(info.parser, RUST_JSON);
        assert!(
            engine
                .derived_result(session, file.handle, RUST_JSON)
                .unwrap()
                .is_some()
        );

        let mut forged = parse_json_rust(input, bytes).unwrap();
        forged.provenance.input.digest[0] ^= 1;
        assert!(matches!(
            engine.commit_derived_result(owner, lease, forged, 0),
            Err(ExtractionError::InvalidResult(_))
        ));
        assert_eq!(engine.extraction.result_count(), 1);

        block_on(engine.append(session, file.handle, b" ", ANY_REVISION)).unwrap();
        assert!(
            engine
                .derived_result(session, file.handle, RUST_JSON)
                .unwrap()
                .is_none()
        );
        let stale = parse_json_rust(input, bytes).unwrap();
        assert_eq!(
            engine.commit_derived_result(owner, lease, stale, 0),
            Err(ExtractionError::ObsoleteVersion)
        );
        assert_eq!(engine.extraction.result_count(), 1);
        block_on(engine.release_lent_version(owner, lease)).unwrap();
    }

    #[test]
    fn reprocessing_replaces_only_the_selected_parser_lineage() {
        let (mut engine, session, root) = engine();
        let file = block_on(engine.create(session, root, &name("doc"), ObjectKind::File)).unwrap();
        let bytes = br#"{"a":1,"b":2}"#;
        block_on(engine.write_at(session, file.handle, 0, bytes, ANY_REVISION)).unwrap();
        let owner = ExtractionOwner::new(12);
        let (lease, input) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();

        engine
            .commit_derived_result(owner, lease, parse_json_rust(input, bytes).unwrap(), 0)
            .unwrap();
        engine
            .commit_derived_result(owner, lease, parse_json_c(input, bytes).unwrap(), 0)
            .unwrap();
        assert_eq!(engine.extraction.result_count(), 2);

        let mut rerun = parse_json_rust(input, bytes).unwrap();
        rerun.facts[1].value = FactValue::Unsigned(99);
        engine
            .commit_derived_result(owner, lease, rerun, 0)
            .unwrap();
        assert_eq!(engine.extraction.result_count(), 2);
        let selected = engine
            .derived_result(session, file.handle, RUST_JSON)
            .unwrap()
            .unwrap();
        assert_eq!(selected.facts[1].value, FactValue::Unsigned(99));
        assert!(
            engine
                .derived_result(session, file.handle, C_JSON)
                .unwrap()
                .is_some()
        );

        block_on(engine.release_lent_version(owner, lease)).unwrap();
        block_on(engine.unlink(session, root, &name("doc"))).unwrap();
        assert_eq!(engine.extraction.result_count(), 0);
    }

    fn query(plane: crate::search::SearchPlane, term: &str) -> SearchQuery {
        SearchQuery {
            content: Some(crate::search::SearchContent {
                plane,
                term: crate::search::SearchTerm::new(term).unwrap(),
            }),
            freshness: FreshnessPolicy::CurrentOnly,
            ..SearchQuery::ALL
        }
    }

    #[test]
    fn unlinking_an_unindexed_witness_restores_search_completeness() {
        let (mut engine, session, root) = engine();
        let witness = name("kore-fs-search-witness");
        let file = block_on(engine.create(session, root, &witness, ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, &[0xff], ANY_REVISION)).unwrap();

        let full_text = query(crate::search::SearchPlane::FullText, "answer");
        assert_eq!(
            engine.search(session, root, full_text, None),
            Err(FsError::IndexIncomplete)
        );

        block_on(engine.unlink(session, root, &witness)).unwrap();
        let complete = engine.search(session, root, full_text, None).unwrap();
        assert!(complete.hits.is_empty());
    }

    #[test]
    fn all_content_planes_use_current_parser_results_and_typed_budgets() {
        let (mut engine, session, root) = engine();
        let file =
            block_on(engine.create(session, root, &name("manual"), ObjectKind::File)).unwrap();
        let bytes = b"capability graph search";
        block_on(engine.write_at(session, file.handle, 0, bytes, ANY_REVISION)).unwrap();
        let owner = ExtractionOwner::new(31);
        let (lease, input) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        let mut result = parse_plain_text(input, bytes).unwrap();
        result
            .references
            .push(Reference {
                source: SourceRange::new(0, 10),
                target: heapless::String::try_from("object://manual").unwrap(),
            })
            .unwrap();
        engine
            .commit_derived_result(owner, lease, result, 0)
            .unwrap();

        for (plane, term) in [
            (crate::search::SearchPlane::FactKey, "lines"),
            (crate::search::SearchPlane::FullText, "graph"),
            (crate::search::SearchPlane::Relationship, "object://manual"),
            (crate::search::SearchPlane::Semantic, "capability search"),
        ] {
            let page = engine
                .search(session, root, query(plane, term), None)
                .unwrap();
            assert_eq!(page.hits.len(), 1, "{plane:?}");
            assert!(page.hits[0].score.get() > 0, "{plane:?}");
            assert!(!page.hits[0].snippet.is_empty(), "{plane:?}");
            assert_ne!(page.index_generation, 0);
        }

        let mut bounded = query(crate::search::SearchPlane::FullText, "graph");
        bounded.max_work = 1;
        assert_eq!(
            engine.search(session, root, bounded, None),
            Err(FsError::BudgetExceeded)
        );
        block_on(engine.release_lent_version(owner, lease)).unwrap();
    }

    #[test]
    fn semantic_page_orders_higher_token_coverage_before_traversal_order() {
        let (mut engine, session, root) = engine();
        for (ordinal, file_name, bytes) in [
            (0, "partial", &b"capability only"[..]),
            (1, "complete", &b"capability graph"[..]),
        ] {
            let file =
                block_on(engine.create(session, root, &name(file_name), ObjectKind::File)).unwrap();
            block_on(engine.write_at(session, file.handle, 0, bytes, ANY_REVISION)).unwrap();
            let owner = ExtractionOwner::new(80 + ordinal);
            let (lease, input) =
                block_on(engine.lend_version(owner, session, file.handle)).unwrap();
            engine
                .commit_derived_result(owner, lease, parse_plain_text(input, bytes).unwrap(), 0)
                .unwrap();
            block_on(engine.release_lent_version(owner, lease)).unwrap();
        }

        let page = engine
            .search(
                session,
                root,
                query(crate::search::SearchPlane::Semantic, "capability graph"),
                None,
            )
            .unwrap();
        assert_eq!(page.hits.len(), 2);
        assert_eq!(page.hits[0].path[0], name("complete"));
        assert_eq!(page.hits[0].score.get(), 10_000);
        assert_eq!(page.hits[1].score.get(), 5_000);
    }

    #[test]
    fn current_only_search_never_presents_an_unindexed_file_as_no_match() {
        let (engine, session, root) = engine();
        let file =
            block_on(engine.create(session, root, &name("legacy"), ObjectKind::File)).unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"needle", ANY_REVISION)).unwrap();
        assert_eq!(
            engine.search(
                session,
                root,
                query(crate::search::SearchPlane::FullText, "needle"),
                None,
            ),
            Err(FsError::IndexIncomplete)
        );
        let structural = engine
            .search(
                session,
                root,
                SearchQuery {
                    name_contains: Some(name("legacy")),
                    ..SearchQuery::ALL
                },
                None,
            )
            .unwrap();
        assert_eq!(structural.hits.len(), 1);
    }

    #[test]
    fn relationship_rebuild_preserves_durable_identity() {
        let (mut engine, session, root) = engine();
        let file =
            block_on(engine.create(session, root, &name("source"), ObjectKind::File)).unwrap();
        let bytes = b"source text";
        block_on(engine.write_at(session, file.handle, 0, bytes, ANY_REVISION)).unwrap();
        let identity = engine.stat(session, file.handle).unwrap();
        let owner = ExtractionOwner::new(32);
        let (lease, input) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();

        let mut first = parse_plain_text(input, bytes).unwrap();
        first
            .references
            .push(Reference {
                source: SourceRange::new(0, 6),
                target: heapless::String::try_from("object://old").unwrap(),
            })
            .unwrap();
        engine
            .commit_derived_result(owner, lease, first, 0)
            .unwrap();

        let mut rebuilt = parse_plain_text(input, bytes).unwrap();
        rebuilt
            .references
            .push(Reference {
                source: SourceRange::new(0, 6),
                target: heapless::String::try_from("object://new").unwrap(),
            })
            .unwrap();
        engine
            .commit_derived_result(owner, lease, rebuilt, 0)
            .unwrap();

        assert!(
            engine
                .search(
                    session,
                    root,
                    query(crate::search::SearchPlane::Relationship, "object://old"),
                    None,
                )
                .unwrap()
                .hits
                .is_empty()
        );
        assert_eq!(
            engine
                .search(
                    session,
                    root,
                    query(crate::search::SearchPlane::Relationship, "object://new"),
                    None,
                )
                .unwrap()
                .hits
                .len(),
            1
        );
        let after = engine.stat(session, file.handle).unwrap();
        assert_eq!(after.stable_object, identity.stable_object);
        assert_eq!(after.revision, identity.revision);
        assert_eq!(after.len, identity.len);
        block_on(engine.release_lent_version(owner, lease)).unwrap();
    }

    #[test]
    fn hidden_index_publication_does_not_change_another_views_reply() {
        let (mut engine, visible_session, visible_root) = engine();
        let visible = block_on(engine.create(
            visible_session,
            visible_root,
            &name("visible"),
            ObjectKind::File,
        ))
        .unwrap();
        let visible_bytes = b"shared graph text";
        block_on(engine.write_at(
            visible_session,
            visible.handle,
            0,
            visible_bytes,
            ANY_REVISION,
        ))
        .unwrap();
        let visible_owner = ExtractionOwner::new(33);
        let (visible_lease, visible_input) =
            block_on(engine.lend_version(visible_owner, visible_session, visible.handle)).unwrap();
        engine
            .commit_derived_result(
                visible_owner,
                visible_lease,
                parse_plain_text(visible_input, visible_bytes).unwrap(),
                0,
            )
            .unwrap();
        let wanted = query(crate::search::SearchPlane::FullText, "graph");
        let before = engine
            .search(visible_session, visible_root, wanted, None)
            .unwrap();

        let hidden_session = SessionId::new(1);
        block_on(engine.attach_session(hidden_session)).unwrap();
        let hidden_root = engine
            .open_root(hidden_session, Rights::ALL)
            .unwrap()
            .handle;
        let hidden = block_on(engine.create(
            hidden_session,
            hidden_root,
            &name("hidden"),
            ObjectKind::File,
        ))
        .unwrap();
        let hidden_bytes = b"hidden graph match";
        block_on(engine.write_at(hidden_session, hidden.handle, 0, hidden_bytes, ANY_REVISION))
            .unwrap();
        let hidden_owner = ExtractionOwner::new(34);
        let (hidden_lease, hidden_input) =
            block_on(engine.lend_version(hidden_owner, hidden_session, hidden.handle)).unwrap();
        engine
            .commit_derived_result(
                hidden_owner,
                hidden_lease,
                parse_plain_text(hidden_input, hidden_bytes).unwrap(),
                0,
            )
            .unwrap();

        let after = engine
            .search(visible_session, visible_root, wanted, None)
            .unwrap();
        assert_eq!(
            after, before,
            "hidden facts must not perturb any reply field"
        );
        block_on(engine.release_lent_version(visible_owner, visible_lease)).unwrap();
        block_on(engine.release_lent_version(hidden_owner, hidden_lease)).unwrap();
    }
}
