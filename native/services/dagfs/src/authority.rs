//! Views, grants, rights attenuation, and the bounded handle proof.
//!
//! This module is where "a handle is not authority" is made true.
//!
//! # The two authority edges
//!
//! A session reaches objects through two **authority edge** kinds
//! ([`AuthorityKind`]):
//!
//! - its **private root** — every session that receives filesystem access
//!   starts with its own, so there is no machine-wide `/` and a displayed
//!   `/` means "this client's root"; and
//! - zero or more **direct grants** to a file or a directory subtree, each
//!   bound to the *destination* session.
//!
//! A [`GrantId`] is therefore not bearer authority: the edge records both the
//! session allowed to open it and the session that issued it, and both sides
//! are checked.  Handing the bits to a third session buys nothing.
//!
//! # The bounded proof
//!
//! An [`OpenHandle`] records the session that opened it, the authority edge
//! it entered through, the containment chain it followed ([`EdgeProof`]), the
//! rights attenuated along that route, and — through the edges' own slot
//! generations — the generation of every edge involved.
//! [`Authority::revalidate`] re-walks that proof on **every** operation:
//!
//! ```text
//! grant.root ──edge₀──▶ o₁ ──edge₁──▶ o₂ … ──edgeₙ──▶ handle.object
//! ```
//!
//! Each step must still exist, still hang beneath the previous object, and
//! still carry its recorded generation.  If any link is gone the operation
//! returns [`FsError::AccessLost`] even though the client still holds the old
//! numeric handle.  There is no cached decision to bypass — the walk is the
//! check, and it is O(depth) with depth bounded by
//! [`MAX_PATH_DEPTH`](crate::bounds::MAX_PATH_DEPTH). This gives immediate
//! loss of access without a global reachability scan.
//!
//! # Rights only narrow
//!
//! [`Rights`] is a small bit set with no "administrator" bypass. Every
//! transition — delegating a grant, opening a child, opening at reduced
//! rights — is an intersection with what the caller already holds, so
//! authority can only shrink.  In particular [`Rights::SHARE`] is never
//! inherited implicitly: a grant carries it only if the issuer both held it
//! and asked for it.

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};

use alloc::{boxed::Box, vec::Vec};

use crate::{
    ServiceAlloc,
    bounds::{
        MAX_EDGES_PER_SESSION, MAX_HANDLES_PER_SESSION, MAX_LINEAGE_DEPTH, MAX_PATH_DEPTH,
        MAX_SESSIONS,
    },
    error::{FsError, FsResult},
    namespace::{EdgeId, Name, Namespace},
    object::ObjectId,
    service_alloc,
    slot::{AtomicSlotVec, Keyed, SlotAccess, SlotKey, SlotVecRead, SlotVecWrite, fs_access},
};

/// The widest handle index: every session's handle slots, end to end.
///
/// A session's handles live in an arena of its own (`SessionArena`), and a
/// handle's index names the session slot first and the arena slot second
/// (`slot × MAX_HANDLES_PER_SESSION + local`). That keeps one fact exact
/// which per-session tables would otherwise lose: session B presenting
/// session A's handle bits names A's slot, not a live slot of B's, so the
/// refusal is exact rather than a lookup that happens to miss. The record
/// still carries its owner, and the check reads both.
pub const MAX_HANDLE_INDEX: usize = MAX_SESSIONS * MAX_HANDLES_PER_SESSION;

/// The widest grant index, laid out like [`MAX_HANDLE_INDEX`]: an edge is
/// stored with the session that holds it.
pub const MAX_GRANT_INDEX: usize = MAX_SESSIONS * MAX_EDGES_PER_SESSION;

/// Ancestors a validated edge can have: a lineage is the edge itself and at
/// most this many parents ([`MAX_LINEAGE_DEPTH`]).
const MAX_ANCESTORS: usize = MAX_LINEAGE_DEPTH - 1;

/// The session slot an arena-wide id names, and the id within that slot's
/// arena. `None` for an index past every session (`per_session` slots each).
const fn split(id: SlotKey, per_session: usize) -> Option<(usize, SlotKey)> {
    let index = id.index() as usize;
    let slot = index / per_session;
    if slot >= MAX_SESSIONS {
        return None;
    }
    Some((
        slot,
        SlotKey::new((index % per_session) as u32, id.generation()),
    ))
}

/// The arena-wide id of `local` in session slot `slot` ([`split`]'s inverse).
const fn join(slot: usize, local: SlotKey, per_session: usize) -> SlotKey {
    SlotKey::new(
        (slot * per_session + local.index() as usize) as u32,
        local.generation(),
    )
}

pub use kore_dagfs::{AuthorityQuota, GrantId, Handle, Rights, SessionId};

/// Why an authority edge exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuthorityKind {
    /// The session's own starting view.
    PrivateRoot,
    /// A direct grant from another session.
    Grant,
}

/// One authority edge.
#[derive(Debug)]
pub(crate) struct AuthorityEdge {
    /// The session allowed to enter through this edge.
    holder: SessionId,
    /// The session that created it.  A private root's issuer is its holder.
    issuer: SessionId,
    /// The authority edge this edge was attenuated from. A private root has
    /// no parent; every delegated edge has exactly one. The generational key
    /// is the revocation lineage: removing an ancestor can never be confused
    /// with a later occupant of the same arena slot.
    parent: Option<GrantId>,
    /// Where entering lands.
    root: ObjectId,
    rights: Rights,
    quota: AuthorityQuota,
    kind: AuthorityKind,
    /// The name the holder sees this attachment under, if it chose one.
    display_name: Option<Name>,
    /// Handles rooted at this exact authority generation. The atomic claim is
    /// what makes a per-grant ceiling exact under concurrent opens; an arena
    /// scan alone would let two callers observe the same free budget.
    open_handles: AtomicU32,
    /// In-flight quota reservations. They bridge the interval between a live
    /// graph snapshot and publication without maintaining a duplicate usage
    /// ledger: completed usage remains derived from the graph.
    pending_objects: AtomicU32,
    pending_edges: AtomicU32,
    pending_bytes: AtomicU64,
    /// Advances before one reservation leaves the pending set. A claimant
    /// whose graph snapshot straddled another completed mutation retries
    /// instead of admitting growth against stale usage.
    mutation_epoch: AtomicU64,
}

impl AuthorityEdge {
    #[inline]
    pub(crate) const fn holder(&self) -> SessionId {
        self.holder
    }

    #[inline]
    pub(crate) const fn issuer(&self) -> SessionId {
        self.issuer
    }

    #[inline]
    pub(crate) const fn parent(&self) -> Option<GrantId> {
        self.parent
    }

    #[inline]
    pub(crate) const fn root(&self) -> ObjectId {
        self.root
    }

    #[inline]
    pub(crate) const fn rights(&self) -> Rights {
        self.rights
    }

    #[inline]
    pub(crate) const fn quota(&self) -> AuthorityQuota {
        self.quota
    }

    #[inline]
    pub(crate) const fn kind(&self) -> AuthorityKind {
        self.kind
    }

    #[inline]
    pub(crate) const fn display_name(&self) -> Option<Name> {
        self.display_name
    }
}

/// The containment chain an open handle followed, from its authority edge's
/// root to the object itself.
///
/// Empty when the handle *is* the authority edge's root — opening a private
/// root or a granted object directly.  Bounded by
/// [`MAX_PATH_DEPTH`](crate::bounds::MAX_PATH_DEPTH), so a traversal that
/// would outrun the proof is refused rather than recorded incompletely: a
/// truncated proof would silently stop revalidating its upper edges.
///
/// Stored at its own length on the service heap, not at the bound: a handle
/// three directories down holds three edges. Reserving the deepest legal
/// path in every handle made the proof nearly all of a session's storage
/// (`SessionArena`) — 512 of a handle record's 544 bytes — while almost
/// every path a client walks is shallow (`res/FILESYSTEM_SCALE.md`).
#[derive(Debug, Default)]
pub struct EdgeProof(Box<[EdgeId], ServiceAlloc>);

impl EdgeProof {
    /// The empty proof — the authority edge's own root. Allocates nothing.
    pub fn root() -> Self {
        Self::default()
    }

    /// This proof extended by one edge: [`FsError::Range`] past the depth
    /// bound, [`FsError::OutOfHandles`] when the heap cannot hold it. The
    /// extended proof is a new allocation, so a refusal leaves this one
    /// whole.
    pub fn extend(&self, edge: EdgeId) -> FsResult<Self> {
        let depth = self.0.len();
        if depth >= MAX_PATH_DEPTH {
            return Err(FsError::Range);
        }
        let mut next = Box::try_new_uninit_slice_in(depth + 1, *service_alloc())
            .map_err(|_| FsError::OutOfHandles)?;
        let (prefix, last) = next.split_at_mut(depth);
        prefix.write_copy_of_slice(&self.0);
        last[0].write(edge);
        // SAFETY: the prefix and the one slot after it are every element.
        Ok(Self(unsafe { next.assume_init() }))
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn edges(&self) -> &[EdgeId] {
        &self.0
    }
}

/// One open handle.
///
/// Deliberately does **not** cache the revision observed at open time. The
/// plan's data model sketches one, but the engine already returns the
/// revision from every open and every write, so the client holds it, and a
/// conditional write names the revision it expects in the request itself. A
/// second copy inside the service would be state that could only go stale
/// (`kore-design-doctrine` §4 — derive, don't store).
#[derive(Debug)]
pub(crate) struct OpenHandle {
    session: SessionId,
    object: ObjectId,
    grant: GrantId,
    proof: EdgeProof,
    rights: Rights,
}

impl OpenHandle {
    #[inline]
    pub(crate) const fn grant(&self) -> GrantId {
        self.grant
    }

    #[inline]
    pub(crate) const fn proof(&self) -> &EdgeProof {
        &self.proof
    }
}

/// Per-session bookkeeping.
struct Session {
    /// Zero is vacant, `SESSION_TRANSITION` is unpublished attach ownership,
    /// and every other value is the exact private-root grant generation.
    root: AtomicU64,
    /// Peer/session retirement leaves the exact root key published while
    /// refusing new work. That lets a contended retirement drop its claims
    /// and retry without losing the identity it still owns.
    closing: AtomicBool,
    /// Serializes only this session's task-visible top-level name set: direct
    /// children of the private root plus named authority attachments.  It is
    /// deliberately per-session rather than a namespace lock; mutations in
    /// unrelated views remain independent.
    view_claimed: AtomicBool,
    /// This slot's storage ([`SessionArena`]): null until the slot first
    /// attaches, then the same arena for every later occupant.
    arena: AtomicPtr<SessionArena>,
}

const SESSION_VACANT: u64 = 0;
const SESSION_TRANSITION: u64 = u64::MAX;

impl Session {
    const fn vacant() -> Self {
        Self {
            root: AtomicU64::new(SESSION_VACANT),
            closing: AtomicBool::new(false),
            view_claimed: AtomicBool::new(false),
            arena: AtomicPtr::new(core::ptr::null_mut()),
        }
    }
}

/// One session slot's authority storage: the handles its occupant opens and
/// the authority edges it holds, each in an arena of its own.
///
/// Allocated from the service heap the first time the slot attaches, and kept
/// until the [`Authority`] drops: a read or write guard borrows a slot inside
/// it and may cross an `await`, so the storage never moves and is never freed
/// under one. The slot's next occupant reuses it, and each arena slot's
/// generation keeps counting, so a handle or grant left over from an earlier
/// occupant fails closed. The serve host reuses the lowest free slot, so the
/// arenas follow the most sessions ever attached at once, not
/// [`MAX_SESSIONS`].
///
/// The arenas are the per-session ceilings themselves: a session that has
/// filled its [`MAX_HANDLES_PER_SESSION`] handles or its
/// [`MAX_EDGES_PER_SESSION`] edges is refused, and no other session's
/// storage is touched.
struct SessionArena {
    handles: AtomicSlotVec<OpenHandle, ServiceAlloc>,
    edges: AtomicSlotVec<AuthorityEdge, ServiceAlloc>,
}

impl SessionArena {
    /// The heap bytes one attached slot costs: both arenas' slots and the
    /// arena record. What a service's heap must hold for every session it
    /// can serve at once. The proofs its handles carry are allocated per
    /// open, at their own depth: at most [`MAX_HANDLES_PER_SESSION`] ×
    /// [`MAX_PATH_DEPTH`] edges a session, apart from this.
    pub(crate) const BYTES: usize = MAX_HANDLES_PER_SESSION
        * AtomicSlotVec::<OpenHandle, ServiceAlloc>::SLOT_BYTES
        + MAX_EDGES_PER_SESSION * AtomicSlotVec::<AuthorityEdge, ServiceAlloc>::SLOT_BYTES
        + core::mem::size_of::<Self>();

    fn try_new() -> Option<Box<Self, &'static ServiceAlloc>> {
        let handles = AtomicSlotVec::try_new_in(MAX_HANDLES_PER_SESSION, service_alloc()).ok()?;
        let edges = AtomicSlotVec::try_new_in(MAX_EDGES_PER_SESSION, service_alloc()).ok()?;
        Box::try_new_in(Self { handles, edges }, service_alloc()).ok()
    }
}

/// The heap bytes one engine's authority can come to with every session
/// attached at once (`SessionArena::BYTES`, about 5.7 KiB, per slot).
pub const SESSION_HEAP_BYTES: usize = MAX_SESSIONS * SessionArena::BYTES;

/// One affine claim over a session's combined native-view root names.
pub(crate) struct ViewClaim<'a> {
    session: &'a Session,
}

impl Drop for ViewClaim<'_> {
    fn drop(&mut self) {
        self.session.view_claimed.store(false, Ordering::Release);
    }
}

/// A validated route: what an operation may do, and to what.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Route {
    pub(crate) object: ObjectId,
    pub(crate) rights: Rights,
    pub(crate) grant: GrantId,
    pub(crate) quota: AuthorityQuota,
}

/// One validated edge plus every exact ancestor generation that authorizes it.
///
/// Retaining the whole bounded lineage closes the validate→publish gap: an
/// ancestor retirement cannot claim its root while a descendant grant, handle,
/// quota reservation, or request is still being established.
pub(crate) struct ValidatedEdge<'a> {
    leaf: SlotVecRead<'a, AuthorityEdge, ServiceAlloc>,
    ancestors: heapless::Vec<SlotVecRead<'a, AuthorityEdge, ServiceAlloc>, MAX_ANCESTORS>,
}

impl ValidatedEdge<'_> {
    /// Edges in this lineage: the edge itself and every parent above it.
    fn lineage_len(&self) -> usize {
        self.ancestors.len() + 1
    }
}

impl core::ops::Deref for ValidatedEdge<'_> {
    type Target = AuthorityEdge;

    fn deref(&self) -> &Self::Target {
        &self.leaf
    }
}

/// The authority arena: one record per session, each owning the storage of
/// the handles it opens and the edges it holds ([`SessionArena`]).
///
/// There is no machine-wide handle or edge table, so nothing here is sized
/// by a guess at how many sessions run at once: [`MAX_SESSIONS`] is the
/// serve host's slot space, and storage is allocated per slot as slots are
/// first used. The session records themselves are a few words each, inline.
pub(crate) struct Authority {
    sessions: [Session; MAX_SESSIONS],
}

/// The object roots a retirement released, one per retired edge, for the
/// engine's grant-count and reclamation fold. Heap-backed: one retirement can
/// reach every edge descending from the retired one, across sessions.
pub(crate) type Retired = Vec<ObjectId, &'static ServiceAlloc>;

/// A heap list could not be grown for a closure being retired.
///
/// The closure is the retiring session's own edges and every edge derived
/// from them, wherever it is held, so it has no per-session bound; staging it
/// on the heap is what lets any number of sessions descend from one task
/// view. The refusal is retryable: nothing has been removed yet, and every
/// claim taken so far is republished on the way out.
fn unstaged<E>(_: E) -> FsError {
    FsError::OutOfEdges
}

struct ClaimedEdge<'a> {
    grant: GrantId,
    edge: SlotVecWrite<'a, AuthorityEdge, ServiceAlloc>,
    /// Claimed ancestors above this edge within the closure, set when the
    /// closure is ordered for retirement.
    depth: usize,
}

/// A claimed derivation closure, kept sorted by grant id so membership is a
/// binary search: the closure can span every session descending from a task
/// view, so the rescans that collect and order it must not be quadratic.
struct EdgeClaims<'a>(Vec<ClaimedEdge<'a>, &'static ServiceAlloc>);

impl<'a> EdgeClaims<'a> {
    fn new() -> Self {
        Self(Vec::new_in(service_alloc()))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn contains(&self, grant: GrantId) -> bool {
        self.position_within(grant, self.0.len()).is_some()
    }

    /// Where `grant` sits among the first `sorted` claims, which must be in
    /// grant id order.
    fn position_within(&self, grant: GrantId, sorted: usize) -> Option<usize> {
        self.0[..sorted]
            .binary_search_by_key(&grant.key(), |claim| claim.grant.key())
            .ok()
    }

    fn sort(&mut self) {
        self.0.sort_unstable_by_key(|claim| claim.grant.key());
    }

    /// Order the closure for retirement, children first, before anything is
    /// removed.
    ///
    /// An edge's depth in the closure is the number of claimed edges above
    /// it on its parent chain; a child is one deeper than its parent, so
    /// removing the deepest first never leaves a live edge whose parent is
    /// gone. One sort, rather than a rescan for a leaf before every removal.
    /// A chain longer than any lineage ([`MAX_LINEAGE_DEPTH`]) could only be
    /// a cycle, and fails closed here, while every claim is still held and
    /// nothing has been retired.
    fn into_retirement(mut self) -> FsResult<Retirement<'a>> {
        for index in 0..self.0.len() {
            let mut depth = 0;
            let mut cursor = self.0[index].edge.parent();
            while let Some(parent) = cursor
                && let Some(at) = self.position_within(parent, self.0.len())
            {
                depth += 1;
                if depth > MAX_ANCESTORS {
                    return Err(FsError::AccessLost);
                }
                cursor = self.0[at].edge.parent();
            }
            self.0[index].depth = depth;
        }
        self.0
            .sort_unstable_by(|left, right| right.depth.cmp(&left.depth));
        Ok(Retirement(self.0))
    }
}

/// A claimed closure in retirement order: deepest first.
struct Retirement<'a>(Vec<ClaimedEdge<'a>, &'static ServiceAlloc>);

impl Retirement<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Remove every claimed edge in order, pushing each one's root.
    /// `retired` must already have room for [`len`](Self::len) more roots,
    /// so a heap that cannot grow cannot stop a retirement halfway.
    fn retire_into(self, retired: &mut Retired) -> FsResult<()> {
        if retired.capacity() - retired.len() < self.0.len() {
            return Err(FsError::OutOfEdges);
        }
        for claim in self.0 {
            let edge = claim.edge.remove().map_err(|_| FsError::AccessLost)?;
            retired.push(edge.root());
        }
        Ok(())
    }
}

/// One affine reservation against an authority edge's growth budget.
///
/// The edge read guard pins this exact grant generation (and, through
/// child-first retirement, its ancestry) for the operation. Dropping the
/// claim advances the mutation epoch before releasing pending usage, so a
/// concurrent claimant can never combine an old graph snapshot with a newly
/// freed reservation and over-admit.
pub(crate) struct QuotaClaim<'a> {
    edge: ValidatedEdge<'a>,
    objects: u32,
    edges: u32,
    bytes: u64,
}

impl Drop for QuotaClaim<'_> {
    fn drop(&mut self) {
        self.edge.mutation_epoch.fetch_add(1, Ordering::AcqRel);
        if self.objects != 0 {
            self.edge
                .pending_objects
                .fetch_sub(self.objects, Ordering::AcqRel);
        }
        if self.edges != 0 {
            self.edge
                .pending_edges
                .fetch_sub(self.edges, Ordering::AcqRel);
        }
        if self.bytes != 0 {
            self.edge
                .pending_bytes
                .fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

fn reserve_u32(counter: &AtomicU32, live: u32, delta: u32, limit: u32) -> bool {
    counter
        .try_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
            live.checked_add(pending)?
                .checked_add(delta)
                .filter(|total| *total <= limit)
                .map(|_| pending + delta)
        })
        .is_ok()
}

fn reserve_u64(counter: &AtomicU64, live: u64, delta: u64, limit: u64) -> bool {
    counter
        .try_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
            live.checked_add(pending)?
                .checked_add(delta)
                .filter(|total| *total <= limit)
                .map(|_| pending + delta)
        })
        .is_ok()
}

impl Authority {
    pub(crate) fn new() -> Self {
        Self {
            sessions: [const { Session::vacant() }; MAX_SESSIONS],
        }
    }

    // ── session storage ─────────────────────────────────────────────────────

    /// A session slot's storage, once the slot has attached.
    fn arena(&self, slot: usize) -> Option<&SessionArena> {
        let arena = self.sessions.get(slot)?.arena.load(Ordering::Acquire);
        // SAFETY: a non-null pointer was published once, with release
        // ordering, from a live allocation (`attach_session`), and is freed
        // only by `Drop`, whose `&mut self` excludes every borrow of `self`.
        unsafe { arena.as_ref() }
    }

    /// The arena holding `grant`'s edge, and the edge's id inside it. A
    /// grant naming no session, or a slot that never attached, is stale.
    fn edge_slot(
        &self,
        grant: GrantId,
    ) -> Result<(&AtomicSlotVec<AuthorityEdge, ServiceAlloc>, SlotKey), SlotAccess> {
        let (slot, local) = split(grant.key(), MAX_EDGES_PER_SESSION).ok_or(SlotAccess::Stale)?;
        let arena = self.arena(slot).ok_or(SlotAccess::Stale)?;
        Ok((&arena.edges, local))
    }

    /// Every attached slot's edge arena with its slot: the closure walks.
    fn edge_arenas(
        &self,
    ) -> impl Iterator<Item = (usize, &AtomicSlotVec<AuthorityEdge, ServiceAlloc>)> {
        (0..MAX_SESSIONS).filter_map(|slot| Some((slot, &self.arena(slot)?.edges)))
    }

    /// Store an edge with its holder, which must hold its slot's attach
    /// ownership or its published root. A holder whose edges are all live is
    /// refused [`FsError::OutOfEdges`].
    fn insert_edge(&self, edge: AuthorityEdge) -> FsResult<GrantId> {
        let slot = self.session_slot(edge.holder)?;
        let arena = self.arena(slot).ok_or(FsError::InvalidHandle)?;
        let local = arena.edges.insert(edge).map_err(|_| FsError::OutOfEdges)?;
        Ok(GrantId::from_key(join(slot, local, MAX_EDGES_PER_SESSION)))
    }

    // ── sessions ────────────────────────────────────────────────────────────

    /// Attach a session, giving it a private root at `root` with full rights
    /// over it.
    ///
    /// Refuses [`FsError::AlreadyExists`] for a session that is already
    /// attached: silently replacing its root would strand every handle and
    /// grant it holds. The slot's first attach allocates its storage
    /// ([`SessionArena`]); a heap that cannot hold it refuses
    /// [`FsError::OutOfHandles`] and leaves the slot vacant, so the launch
    /// that asked reads as a full table.
    pub(crate) fn attach_session(&self, session: SessionId) -> FsResult<usize> {
        let slot = self.session_slot(session)?;
        if self.sessions[slot].closing.load(Ordering::Acquire) {
            return Err(FsError::Busy);
        }
        match self.sessions[slot].root.compare_exchange(
            SESSION_VACANT,
            SESSION_TRANSITION,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(SESSION_TRANSITION) => return Err(FsError::Busy),
            Err(_) => return Err(FsError::AlreadyExists),
        }
        // The transition is this caller's alone, so it is the only one that
        // can find the slot without storage and publish it.
        if self.sessions[slot].arena.load(Ordering::Acquire).is_null() {
            let Some(arena) = SessionArena::try_new() else {
                self.sessions[slot]
                    .root
                    .store(SESSION_VACANT, Ordering::Release);
                return Err(FsError::OutOfHandles);
            };
            let (arena, _) = Box::into_raw_with_allocator(arena);
            self.sessions[slot].arena.store(arena, Ordering::Release);
        }
        Ok(slot)
    }

    /// Record a freshly attached session's private-root edge.
    pub(crate) fn install_root(&self, session: SessionId, root: ObjectId) -> FsResult<GrantId> {
        let slot = self.session_slot(session)?;
        if self.sessions[slot].root.load(Ordering::Acquire) != SESSION_TRANSITION {
            return Err(FsError::InvalidHandle);
        }
        let grant = self.insert_edge(AuthorityEdge {
            holder: session,
            issuer: session,
            parent: None,
            root,
            rights: Rights::ALL,
            quota: AuthorityQuota::MAX,
            kind: AuthorityKind::PrivateRoot,
            display_name: None,
            open_handles: AtomicU32::new(0),
            pending_objects: AtomicU32::new(0),
            pending_edges: AtomicU32::new(0),
            pending_bytes: AtomicU64::new(0),
            mutation_epoch: AtomicU64::new(0),
        })?;
        self.sessions[slot].closing.store(false, Ordering::Release);
        self.sessions[slot]
            .root
            .store(grant.to_bits(), Ordering::Release);
        Ok(grant)
    }

    /// Install an attenuated descendant of `source` as an already-claimed
    /// session's structural root — the invocation-view posture: the session
    /// enters the filesystem *through a delegation* of an existing view
    /// instead of receiving a fresh ambient private namespace.  The source
    /// generation remains the revocation ancestor, rights can only narrow,
    /// and no object is created.
    ///
    /// Like [`install_root`](Self::install_root), this publishes the root of
    /// a session parked in its attach transition; the two are the only exits
    /// from that state besides [`abort_attach`](Self::abort_attach).
    pub(crate) fn install_delegated_root(
        &self,
        session: SessionId,
        source: GrantId,
        rights: Rights,
    ) -> FsResult<GrantId> {
        let slot = self.session_slot(session)?;
        if self.sessions[slot].root.load(Ordering::Acquire) != SESSION_TRANSITION {
            return Err(FsError::AlreadyExists);
        }
        let (delegated, quota, root, issuer) = {
            let source_edge = self.validate_lineage(source)?;
            if source_edge.lineage_len() >= MAX_LINEAGE_DEPTH {
                return Err(FsError::Range);
            }
            (
                source_edge.rights().attenuate(rights)?,
                source_edge.quota(),
                source_edge.root(),
                source_edge.holder(),
            )
        };
        // A closing issuer cannot extend a lineage while its peer teardown
        // is claiming that same bounded derivation closure.
        self.root_grant(issuer)?;
        let grant = self.insert_edge(AuthorityEdge {
            holder: session,
            issuer,
            parent: Some(source),
            root,
            rights: delegated,
            quota,
            kind: AuthorityKind::PrivateRoot,
            display_name: None,
            open_handles: AtomicU32::new(0),
            pending_objects: AtomicU32::new(0),
            pending_edges: AtomicU32::new(0),
            pending_bytes: AtomicU64::new(0),
            mutation_epoch: AtomicU64::new(0),
        })?;
        self.sessions[slot].closing.store(false, Ordering::Release);
        self.sessions[slot]
            .root
            .store(grant.to_bits(), Ordering::Release);
        Ok(grant)
    }

    /// Give back an unpublished attach claim. No handle or root edge exists
    /// while a session is in this state.
    pub(crate) fn abort_attach(&self, session: SessionId) {
        let Ok(slot) = self.session_slot(session) else {
            return;
        };
        let _ = self.sessions[slot].root.compare_exchange(
            SESSION_TRANSITION,
            SESSION_VACANT,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.sessions[slot].closing.store(false, Ordering::Release);
    }

    fn session_slot(&self, session: SessionId) -> FsResult<usize> {
        let slot = session.index() as usize;
        if slot >= MAX_SESSIONS {
            return Err(FsError::Range);
        }
        Ok(slot)
    }

    pub(crate) fn is_attached(&self, session: SessionId) -> bool {
        self.root_grant(session).is_ok()
    }

    /// Sessions whose root slot is occupied (rooted or mid-transition) — the
    /// diagnostic aggregate, deliberately a bare count.
    pub(crate) fn live_sessions(&self) -> u16 {
        self.sessions
            .iter()
            .filter(|session| session.root.load(Ordering::Acquire) != SESSION_VACANT)
            .count() as u16
    }

    /// The session's private-root authority edge.
    pub(crate) fn root_grant(&self, session: SessionId) -> FsResult<GrantId> {
        let slot = self.session_slot(session)?;
        if self.sessions[slot].closing.load(Ordering::Acquire) {
            return Err(FsError::Busy);
        }
        let bits = self.sessions[slot].root.load(Ordering::Acquire);
        match bits {
            SESSION_VACANT => Err(FsError::InvalidHandle),
            SESSION_TRANSITION => Err(FsError::Busy),
            _ => Ok(GrantId::from_bits(bits)),
        }
    }

    pub(crate) fn root_object(&self, session: SessionId) -> FsResult<ObjectId> {
        let grant = self.root_grant(session)?;
        Ok(self.validated_edge(grant)?.root())
    }

    /// Claim one task's combined private-root/attachment name set.
    pub(crate) fn claim_view(&self, session: SessionId) -> FsResult<ViewClaim<'_>> {
        self.root_grant(session)?;
        let slot = self.session_slot(session)?;
        self.claim_view_slot(slot)
    }

    fn claim_view_slot(&self, slot: usize) -> FsResult<ViewClaim<'_>> {
        self.sessions[slot]
            .view_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| FsError::Busy)?;
        Ok(ViewClaim {
            session: &self.sessions[slot],
        })
    }

    /// Snapshot the per-grant mutation generation before graph usage is
    /// derived. [`Self::claim_growth`] closes the observation by requiring
    /// this exact value after reserving against all in-flight growth.
    pub(crate) fn quota_epoch(&self, grant: GrantId) -> FsResult<u64> {
        Ok(self
            .validated_edge(grant)?
            .mutation_epoch
            .load(Ordering::Acquire))
    }

    /// Reserve one operation's growth against live graph usage plus every
    /// other in-flight reservation on this exact authority edge.
    pub(crate) fn claim_growth(
        &self,
        grant: GrantId,
        expected_epoch: u64,
        live_objects: u32,
        live_edges: u32,
        live_bytes: u64,
        objects: u32,
        edges: u32,
        bytes: u64,
    ) -> FsResult<QuotaClaim<'_>> {
        let edge = self.validated_edge(grant)?;
        if edge.mutation_epoch.load(Ordering::Acquire) != expected_epoch {
            return Err(FsError::Busy);
        }
        if !reserve_u32(
            &edge.pending_objects,
            live_objects,
            objects,
            edge.quota.max_objects(),
        ) {
            return Err(FsError::OutOfObjects);
        }
        if !reserve_u32(
            &edge.pending_edges,
            live_edges,
            edges,
            edge.quota.max_edges(),
        ) {
            edge.pending_objects.fetch_sub(objects, Ordering::AcqRel);
            return Err(FsError::OutOfEdges);
        }
        if !reserve_u64(
            &edge.pending_bytes,
            live_bytes,
            bytes,
            edge.quota.max_bytes(),
        ) {
            edge.pending_edges.fetch_sub(edges, Ordering::AcqRel);
            edge.pending_objects.fetch_sub(objects, Ordering::AcqRel);
            return Err(FsError::OutOfStorage);
        }
        if edge.mutation_epoch.load(Ordering::Acquire) != expected_epoch {
            edge.pending_bytes.fetch_sub(bytes, Ordering::AcqRel);
            edge.pending_edges.fetch_sub(edges, Ordering::AcqRel);
            edge.pending_objects.fetch_sub(objects, Ordering::AcqRel);
            return Err(FsError::Busy);
        }
        Ok(QuotaClaim {
            edge,
            objects,
            edges,
            bytes,
        })
    }

    /// A session's live handles: its own arena's, which no other session's
    /// handles can occupy.
    fn session_handle_ids(
        &self,
        session: SessionId,
    ) -> FsResult<heapless::Vec<Handle, MAX_HANDLES_PER_SESSION>> {
        let slot = self.session_slot(session)?;
        let mut ids = heapless::Vec::new();
        let Some(arena) = self.arena(slot) else {
            return Ok(ids);
        };
        for index in 0..arena.handles.capacity() {
            let Some(id) = arena.handles.live_id_at(index) else {
                continue;
            };
            let record = arena.handles.try_read(id).map_err(|error| match error {
                SlotAccess::Busy => FsError::Busy,
                SlotAccess::Stale => FsError::Busy,
            })?;
            if record.session == session {
                ids.push(Handle::from_key(join(slot, id, MAX_HANDLES_PER_SESSION)))
                    .map_err(|_| FsError::OutOfHandles)?;
            }
        }
        Ok(ids)
    }

    fn claim_handles<'a>(
        &'a self,
        session: SessionId,
        ids: &[Handle],
    ) -> FsResult<
        heapless::Vec<
            (Handle, SlotVecWrite<'a, OpenHandle, ServiceAlloc>),
            MAX_HANDLES_PER_SESSION,
        >,
    > {
        let mut claims = heapless::Vec::new();
        for handle in ids {
            let (handles, local) = self.handle_slot(session, *handle)?;
            let record = handles.try_write(local).map_err(|_| FsError::Busy)?;
            if record.session != session {
                return Err(FsError::Busy);
            }
            claims
                .push((*handle, record))
                .map_err(|_| FsError::OutOfHandles)?;
        }
        Ok(claims)
    }

    /// The edges a session holds: its own arena's.
    fn session_edge_roots(
        &self,
        session: SessionId,
    ) -> FsResult<heapless::Vec<GrantId, MAX_EDGES_PER_SESSION>> {
        let slot = self.session_slot(session)?;
        let mut roots = heapless::Vec::new();
        let Some(arena) = self.arena(slot) else {
            return Ok(roots);
        };
        for index in 0..arena.edges.capacity() {
            let Some(id) = arena.edges.live_id_at(index) else {
                continue;
            };
            let edge = arena.edges.try_read(id).map_err(|_| FsError::Busy)?;
            if edge.holder == session {
                roots
                    .push(GrantId::from_key(join(slot, id, MAX_EDGES_PER_SESSION)))
                    .map_err(|_| FsError::OutOfEdges)?;
            }
        }
        Ok(roots)
    }

    /// Claim a complete derivation closure before removing its first edge.
    ///
    /// Claimed roots make any new descendant's lineage validation return
    /// `Busy`. Existing descendants are discovered from immutable parent ids,
    /// one generation per pass over every attached session's edges; a
    /// concurrently claimed edge causes the whole attempt to back off with
    /// every already-acquired guard republished. The closure is staged on the
    /// heap ([`unstaged`]): a task view's reaches every command descending
    /// from it, wherever their edges are held.
    fn claim_edge_closure<'a>(&'a self, roots: &[GrantId]) -> FsResult<EdgeClaims<'a>> {
        // Claims are taken in canonical (grant id) order, here and below, so
        // two overlapping retirements back off rather than chase each other.
        let mut unique = heapless::Vec::<GrantId, MAX_EDGES_PER_SESSION>::new();
        for root in roots {
            if !unique.contains(root) {
                unique.push(*root).map_err(|_| FsError::OutOfEdges)?;
            }
        }
        unique
            .as_mut_slice()
            .sort_unstable_by_key(|grant| grant.key());

        let mut claimed = EdgeClaims::new();
        claimed.0.try_reserve(unique.len()).map_err(unstaged)?;
        for grant in unique {
            let (edges, local) = self.edge_slot(grant).map_err(|_| FsError::Busy)?;
            let edge = edges.try_write(local).map_err(|_| FsError::Busy)?;
            claimed.0.push(ClaimedEdge {
                grant,
                edge,
                depth: 0,
            });
        }

        let mut children = Vec::new_in(service_alloc());
        loop {
            children.clear();
            for (slot, edges) in self.edge_arenas() {
                for index in 0..edges.capacity() {
                    let Some(id) = edges.live_id_at(index) else {
                        continue;
                    };
                    let grant = GrantId::from_key(join(slot, id, MAX_EDGES_PER_SESSION));
                    if claimed.contains(grant) {
                        continue;
                    }
                    let edge = edges.try_read(id).map_err(|_| FsError::Busy)?;
                    if edge.parent().is_some_and(|parent| claimed.contains(parent)) {
                        children.try_reserve(1).map_err(unstaged)?;
                        children.push(grant);
                    }
                }
            }
            if children.is_empty() {
                break;
            }
            // Collected slot by slot and index by index, so already in
            // ascending grant id order: the canonical claim order. Every
            // child's parent was claimed in an earlier generation — one
            // claimed only now was not yet a parent when this one was
            // collected — so the parent check searches that sorted prefix,
            // not this generation's unsorted tail.
            claimed.0.try_reserve(children.len()).map_err(unstaged)?;
            let parents = claimed.len();
            for grant in children.iter().copied() {
                let (edges, local) = self.edge_slot(grant).map_err(|_| FsError::Busy)?;
                let edge = edges.try_write(local).map_err(|_| FsError::Busy)?;
                if !edge
                    .parent()
                    .is_some_and(|parent| claimed.position_within(parent, parents).is_some())
                {
                    return Err(FsError::Busy);
                }
                claimed.0.push(ClaimedEdge {
                    grant,
                    edge,
                    depth: 0,
                });
            }
            // One sort per generation keeps the membership tests above
            // binary searches; a generation never claims an edge twice,
            // because every candidate was unclaimed when it was collected.
            claimed.sort();
        }
        Ok(claimed)
    }

    /// Detach a session: retire every handle it holds and every authority
    /// edge it holds, and report the roots whose last route just went away so
    /// the caller can reclaim them.
    pub(crate) fn detach_session(&self, session: SessionId) -> FsResult<Retired> {
        let slot = self.session_slot(session)?;
        self.sessions[slot]
            .closing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| FsError::Busy)?;

        let attempt = (|| {
            let root_bits = self.sessions[slot].root.load(Ordering::Acquire);
            match root_bits {
                SESSION_VACANT => return Err(FsError::InvalidHandle),
                // Admission parks a peer's session here until a view installs
                // its root, and a peer can close before it is ever viewed (a
                // launch that failed after admission). Such a session holds no
                // handle — opening one needs a published root — and no edge,
                // because a root's edge is published with the root; giving the
                // slot back is the whole of its retirement. Refusing it instead
                // pinned the slot for good, its close retried on every wake.
                // An edge already held is a root install in flight, which
                // publishes or aborts on its own: retry after it.
                SESSION_TRANSITION => {
                    return if self.session_edge_roots(session)?.is_empty() {
                        Ok(Retired::new_in(service_alloc()))
                    } else {
                        Err(FsError::Busy)
                    };
                }
                _ => {}
            }
            let private_root = GrantId::from_bits(root_bits);
            let _view = self.claim_view_slot(slot)?;

            // Closing blocks every new handle/grant insertion. Claim all
            // existing records before removing the first one, so an in-flight
            // request makes peer cleanup retry rather than disappear from a
            // best-effort iterator.
            let handle_ids = self.session_handle_ids(session)?;
            let handles = self.claim_handles(session, &handle_ids)?;
            let roots = self.session_edge_roots(session)?;
            if !roots.contains(&private_root) {
                return Err(FsError::AccessLost);
            }
            let edges = self.claim_edge_closure(&roots)?;
            if handles
                .iter()
                .any(|(_, handle)| !edges.contains(handle.grant()))
            {
                return Err(FsError::AccessLost);
            }
            // Everything that can refuse — the order and the room for its
            // roots — is settled before the first record goes.
            let edges = edges.into_retirement()?;
            let mut stranded = Retired::new_in(service_alloc());
            stranded.try_reserve_exact(edges.len()).map_err(unstaged)?;

            for (_, handle) in handles {
                handle.remove().map_err(|_| FsError::AccessLost)?;
            }
            edges.retire_into(&mut stranded)?;
            Ok(stranded)
        })();

        match attempt {
            Ok(stranded) => {
                self.sessions[slot]
                    .root
                    .store(SESSION_VACANT, Ordering::Release);
                self.sessions[slot].closing.store(false, Ordering::Release);
                Ok(stranded)
            }
            Err(error) => {
                self.sessions[slot].closing.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    // ── authority edges ─────────────────────────────────────────────────────

    pub(crate) fn edge(
        &self,
        grant: GrantId,
    ) -> FsResult<SlotVecRead<'_, AuthorityEdge, ServiceAlloc>> {
        self.edge_slot(grant)
            .and_then(|(edges, local)| edges.try_read(local))
            .map_err(|error| fs_access(error, FsError::AccessLost))
    }

    /// One edge after validating its complete delegation ancestry.
    pub(crate) fn validated_edge(&self, grant: GrantId) -> FsResult<ValidatedEdge<'_>> {
        self.validate_lineage(grant)
    }

    /// Create a grant for `holder` rooted at `root` with `rights`.
    pub(crate) fn insert_grant(
        &self,
        issuer: SessionId,
        holder: SessionId,
        parent: GrantId,
        root: ObjectId,
        rights: Rights,
        quota: AuthorityQuota,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        // A closing issuer cannot extend a lineage while its peer teardown is
        // claiming that same bounded derivation closure.
        self.root_grant(issuer)?;
        // A grant to a session that has never attached could never be opened
        // and would keep its root alive forever.
        if !self.is_attached(holder) {
            return Err(FsError::InvalidHandle);
        }
        let source = self.validate_lineage(parent)?;
        if source.holder() != issuer {
            return Err(FsError::AccessDenied);
        }
        if source.lineage_len() >= MAX_LINEAGE_DEPTH {
            return Err(FsError::Range);
        }
        source.rights().attenuate(rights)?;
        source.quota().attenuate(quota)?;
        self.insert_edge(AuthorityEdge {
            holder,
            issuer,
            parent: Some(parent),
            root,
            rights,
            quota,
            kind: AuthorityKind::Grant,
            display_name,
            open_handles: AtomicU32::new(0),
            pending_objects: AtomicU32::new(0),
            pending_edges: AtomicU32::new(0),
            pending_bytes: AtomicU64::new(0),
            mutation_epoch: AtomicU64::new(0),
        })
    }

    /// Pin an authorized route beneath its nearest non-process ancestor.
    ///
    /// A recyclable process root is a short-lived descendant installed from
    /// an invocation view. Pinning a `cd` result beneath that process root
    /// would make peer teardown revoke the new cwd even though the caller
    /// retained its independent channel-root revocation pivot. Reparent the
    /// narrowed object beneath the exact stable invocation ancestor already
    /// present in the validated route; no unrelated root or pathname is
    /// consulted. The current route still needs SHARE and bounds both rights
    /// and quota, so this cannot amplify authority.
    pub(crate) fn pin_route(
        &self,
        session: SessionId,
        route: Route,
        rights: Rights,
    ) -> FsResult<(GrantId, SessionId)> {
        route.rights.require(Rights::SHARE)?;
        let delegated = route.rights.attenuate(rights)?;
        let (grant, holder) = self.stable_ancestor(session, route.grant)?;
        let edge = self.validated_edge(grant)?;
        edge.rights().attenuate(delegated)?;
        edge.quota().attenuate(route.quota)?;
        drop(edge);
        let pinned = self.insert_grant(
            holder,
            holder,
            grant,
            route.object,
            delegated,
            route.quota,
            None,
        )?;
        Ok((pinned, holder))
    }

    /// The nearest edge on `grant`'s validated lineage held by a session
    /// other than `session`, and its holder: the stable invocation view a
    /// recyclable process descends from. [`Self::pin_route`] pins beneath
    /// it, and a cwd anchor is matched against its task by it.
    pub(crate) fn stable_ancestor(
        &self,
        session: SessionId,
        grant: GrantId,
    ) -> FsResult<(GrantId, SessionId)> {
        let mut cursor = Some(grant);
        for _ in 0..MAX_LINEAGE_DEPTH {
            let grant = cursor.ok_or(FsError::AccessDenied)?;
            let edge = self.validated_edge(grant)?;
            let holder = edge.holder();
            if holder != session {
                return Ok((grant, holder));
            }
            cursor = edge.parent();
        }
        Err(FsError::AccessLost)
    }

    /// Grant a transient process route from its exact stable task ancestor.
    /// This is the sharing analogue of [`Self::pin_route`]: it walks the
    /// already-validated generational lineage and refuses unless
    /// `source_view` is on that lineage. The new edge is issued by the stable
    /// task view, not by the process that happens to perform the request.
    pub(crate) fn grant_route_from_ancestor(
        &self,
        session: SessionId,
        source_view: SessionId,
        destination_view: SessionId,
        route: Route,
        rights: Rights,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        route.rights.require(Rights::SHARE)?;
        route.rights.attenuate(rights)?;
        let mut cursor = Some(route.grant);
        for _ in 0..MAX_LINEAGE_DEPTH {
            let grant = cursor.ok_or(FsError::AccessDenied)?;
            let edge = self.validated_edge(grant)?;
            let holder = edge.holder();
            cursor = edge.parent();
            if holder == session {
                continue;
            }
            if holder != source_view {
                return Err(FsError::AccessDenied);
            }
            edge.rights().attenuate(rights)?;
            edge.quota().attenuate(route.quota)?;
            drop(edge);
            return self.insert_grant(
                source_view,
                destination_view,
                grant,
                route.object,
                rights,
                route.quota,
                display_name,
            );
        }
        Err(FsError::AccessLost)
    }

    /// Retire an authority edge, returning the object it rooted.
    ///
    /// Retiring bumps the edge slot's generation, so every handle that
    /// entered through it fails its next revalidation — which is the whole of
    /// "losing a direct-grant route makes existing reads fail immediately".
    ///
    /// A private root may not be revoked this way: a session's starting view
    /// goes away with the session ([`Self::detach_session`]), and letting a
    /// caller retire it would leave an attached session with no route at all.
    pub(crate) fn retire_edge(&self, caller: SessionId, grant: GrantId) -> FsResult<Retired> {
        let edge = self.edge(grant)?;
        if edge.kind() == AuthorityKind::PrivateRoot {
            return Err(FsError::Unsupported);
        }
        // Either side of the grant may end it; nobody else may.
        if edge.holder() != caller && edge.issuer() != caller {
            return Err(FsError::AccessDenied);
        }
        drop(edge);
        let mut retired = Retired::new_in(service_alloc());
        self.collect_and_retire(grant, &mut retired)?;
        Ok(retired)
    }

    /// Validate one authority edge and every generational parent it derives
    /// from. Every edge is created within [`MAX_LINEAGE_DEPTH`] of its root,
    /// so a longer walk would be a cycle or corrupt state and fails closed.
    fn validate_lineage(&self, grant: GrantId) -> FsResult<ValidatedEdge<'_>> {
        let leaf = self.edge(grant)?;
        let mut cursor = leaf.parent();
        let mut ancestors = heapless::Vec::new();
        while let Some(parent) = cursor {
            let edge = self.edge(parent)?;
            cursor = edge.parent();
            ancestors.push(edge).map_err(|_| FsError::AccessLost)?;
        }
        Ok(ValidatedEdge { leaf, ancestors })
    }

    /// Retire `root` and the bounded derivation closure below it. Each edge is
    /// removed once and contributes one object root for the engine's exact
    /// grant-count/reclamation fold.
    fn collect_and_retire(&self, root: GrantId, retired: &mut Retired) -> FsResult<()> {
        let closure = self.claim_edge_closure(&[root])?.into_retirement()?;
        retired.try_reserve_exact(closure.len()).map_err(unstaged)?;
        closure.retire_into(retired)
    }

    /// The grants a session holds, as `(id, kind, rights, display name)`.
    pub(crate) fn grants_of(
        &self,
        session: SessionId,
    ) -> FsResult<
        heapless::Vec<
            (GrantId, AuthorityKind, Rights, AuthorityQuota, Option<Name>),
            MAX_EDGES_PER_SESSION,
        >,
    > {
        let _view = self.claim_view(session)?;
        let slot = self.session_slot(session)?;
        let mut grants = heapless::Vec::new();
        let Some(arena) = self.arena(slot) else {
            return Ok(grants);
        };
        for index in 0..arena.edges.capacity() {
            let Some(id) = arena.edges.live_id_at(index) else {
                continue;
            };
            let edge = arena.edges.try_read(id).map_err(|_| FsError::Busy)?;
            if edge.holder() == session {
                grants
                    .push((
                        GrantId::from_key(join(slot, id, MAX_EDGES_PER_SESSION)),
                        edge.kind(),
                        edge.rights(),
                        edge.quota(),
                        edge.display_name(),
                    ))
                    .map_err(|_| FsError::OutOfEdges)?;
            }
        }
        Ok(grants)
    }

    /// `true` if `session` already sees an attachment named `name`.
    ///
    /// Used to keep a sender from overwriting a name in the recipient's view
    /// because the recipient's namespace of attachment names is its own.
    pub(crate) fn display_name_taken(&self, session: SessionId, name: &Name) -> FsResult<bool> {
        let slot = self.session_slot(session)?;
        let Some(arena) = self.arena(slot) else {
            return Ok(false);
        };
        for index in 0..arena.edges.capacity() {
            let Some(id) = arena.edges.live_id_at(index) else {
                continue;
            };
            let edge = arena.edges.try_read(id).map_err(|_| FsError::Busy)?;
            if edge.holder() == session && edge.display_name().as_ref() == Some(name) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ── handles ─────────────────────────────────────────────────────────────

    /// Mint a handle in `session`'s own arena.
    ///
    /// The arena's capacity is the per-session ceiling
    /// ([`MAX_HANDLES_PER_SESSION`]): a session that holds that many is
    /// refused, and no other session's handles share its storage, so one
    /// client cannot starve another.
    pub(crate) fn insert_handle(
        &self,
        session: SessionId,
        object: ObjectId,
        grant: GrantId,
        proof: EdgeProof,
        rights: Rights,
    ) -> FsResult<Handle> {
        let slot = self.session_slot(session)?;
        // A transition-owned session accepts no new work. This check also
        // rejects a forged/unattached session before consuming either quota.
        self.root_grant(session)?;
        let arena = self.arena(slot).ok_or(FsError::InvalidHandle)?;
        if arena.handles.is_full() {
            return Err(FsError::OutOfHandles);
        }
        let edge = self.validate_lineage(grant)?;
        if edge
            .open_handles
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                (live < edge.quota().max_handles()).then_some(live + 1)
            })
            .is_err()
        {
            return Err(FsError::OutOfHandles);
        }
        let inserted = arena.handles.insert(OpenHandle {
            session,
            object,
            grant,
            proof,
            rights,
        });
        match inserted {
            Ok(local) => Ok(Handle::from_key(join(slot, local, MAX_HANDLES_PER_SESSION))),
            Err(_) => {
                edge.open_handles.fetch_sub(1, Ordering::AcqRel);
                Err(FsError::OutOfHandles)
            }
        }
    }

    /// The arena holding `session`'s handle `handle`, and the handle's id
    /// inside it. A handle whose index names another session's slot is not
    /// this session's at all: [`FsError::InvalidHandle`], exactly.
    fn handle_slot(
        &self,
        session: SessionId,
        handle: Handle,
    ) -> FsResult<(&AtomicSlotVec<OpenHandle, ServiceAlloc>, SlotKey)> {
        let (slot, local) =
            split(handle.key(), MAX_HANDLES_PER_SESSION).ok_or(FsError::InvalidHandle)?;
        if slot != self.session_slot(session)? {
            return Err(FsError::InvalidHandle);
        }
        let arena = self.arena(slot).ok_or(FsError::InvalidHandle)?;
        Ok((&arena.handles, local))
    }

    /// The record behind a handle, if `session` owns it.
    ///
    /// A handle presented by the wrong session is [`FsError::InvalidHandle`],
    /// not `AccessDenied`: from the presenting session's point of view the
    /// handle does not exist at all.
    pub(crate) fn handle(
        &self,
        session: SessionId,
        handle: Handle,
    ) -> FsResult<SlotVecRead<'_, OpenHandle, ServiceAlloc>> {
        self.root_grant(session)?;
        let (handles, local) = self.handle_slot(session, handle)?;
        let record = handles
            .try_read(local)
            .map_err(|error| fs_access(error, FsError::InvalidHandle))?;
        if record.session != session {
            return Err(FsError::InvalidHandle);
        }
        Ok(record)
    }

    /// Close a handle.
    pub(crate) fn remove_handle(&self, session: SessionId, handle: Handle) -> FsResult<OpenHandle> {
        // Check ownership before removing, so one session cannot close
        // another's handle.
        self.handle(session, handle)?;
        let (handles, local) = self.handle_slot(session, handle)?;
        let record = handles
            .try_remove(local)
            .map_err(|error| fs_access(error, FsError::InvalidHandle))?;
        if let Ok(edge) = self.edge(record.grant) {
            let _ = edge
                .open_handles
                .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                    Some(live.saturating_sub(1))
                });
        }
        Ok(record)
    }

    /// Live handles held by a session.
    pub(crate) fn live_handles(&self, session: SessionId) -> usize {
        self.root_grant(session)
            .ok()
            .and_then(|_| self.session_slot(session).ok())
            .and_then(|slot| self.arena(slot))
            .map_or(0, |arena| arena.handles.len())
    }

    /// Live authority edges, every session's together.
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_arenas().map(|(_, edges)| edges.len()).sum()
    }

    /// Session slots whose storage has been allocated: the most sessions
    /// ever attached at once, which is what the authority's heap footprint
    /// follows ([`SessionArena::BYTES`] each).
    #[cfg(test)]
    fn session_arenas(&self) -> usize {
        (0..MAX_SESSIONS)
            .filter(|&slot| self.arena(slot).is_some())
            .count()
    }

    // ── the proof walk ──────────────────────────────────────────────────────

    /// Re-establish that a handle still names a reachable object, and report
    /// what the route permits.
    ///
    /// Walked in full on every operation.  The steps, in order, are the
    /// design: the authority edge must still be live; its root must still
    /// exist; every containment edge in the proof must still hang beneath the
    /// previous object at its recorded generation; and the chain must still
    /// arrive at the handle's object.
    pub(crate) fn revalidate<K: Copy + Eq>(
        &self,
        session: SessionId,
        handle: Handle,
        ns: &Namespace<K>,
    ) -> FsResult<Route> {
        let record = self.handle(session, handle)?;

        // 1. The authority edge itself — retired grants land here.
        let edge = self.validate_lineage(record.grant)?;
        // A grant that changed hands cannot be ridden by the old holder.
        if edge.holder != session {
            return Err(FsError::AccessLost);
        }

        // 2. Its root must still be a live object.
        let mut current = edge.root();
        if !ns.exists(current) {
            return Err(FsError::AccessLost);
        }

        // 3. Every containment edge, in order.
        for step in record.proof.edges() {
            // The chain must be connected: this edge has to hang beneath the
            // object the previous step produced, or the proof is describing
            // some other route.
            if step.parent() != current {
                return Err(FsError::AccessLost);
            }
            // `edge_child` fails closed on a removed edge or a retired entry
            // generation — the per-slot generation doing its job.
            current = ns.edge_child(*step)?;
        }

        // 4. And it must arrive where the handle says.
        if current != record.object {
            return Err(FsError::AccessLost);
        }

        Ok(Route {
            object: record.object,
            rights: record.rights,
            grant: record.grant,
            quota: edge.quota(),
        })
    }
}

impl Drop for Authority {
    fn drop(&mut self) {
        for session in &mut self.sessions {
            let arena = *session.arena.get_mut();
            if !arena.is_null() {
                // SAFETY: published once by `attach_session` from
                // `Box::into_raw_with_allocator` over `service_alloc()`, and
                // `&mut self` excludes every guard that could still borrow it.
                drop(unsafe { Box::from_raw_in(arena, service_alloc()) });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rights ──────────────────────────────────────────────────────────────

    #[test]
    fn rights_bits_are_distinct() {
        let all = [
            Rights::READ,
            Rights::WRITE,
            Rights::MUTATE_NAMESPACE,
            Rights::SHARE,
            Rights::BIND_EXECUTABLE,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert!(a.intersection(*b).is_empty(), "rights must not share a bit");
                }
            }
        }
        assert!(Rights::ALL.contains(Rights::READ));
        assert!(Rights::ALL.contains(Rights::SHARE));
        assert!(Rights::ALL.contains(Rights::BIND_EXECUTABLE));
        assert!(Rights::NONE.is_empty());
    }

    #[test]
    fn contains_and_subset_agree() {
        let rw = Rights::READ.union(Rights::WRITE);
        assert!(rw.contains(Rights::READ));
        assert!(rw.contains(Rights::WRITE));
        assert!(!rw.contains(Rights::SHARE));
        assert!(Rights::READ.is_subset_of(rw));
        assert!(!rw.is_subset_of(Rights::READ));
        assert!(rw.is_subset_of(Rights::ALL));
    }

    #[test]
    fn attenuation_cannot_add_a_right() {
        let read_only = Rights::READ;
        // Asking for more than you hold is refused, not clamped.
        assert_eq!(
            read_only.attenuate(Rights::WRITE).unwrap_err(),
            FsError::AccessDenied
        );
        assert_eq!(
            read_only.attenuate(Rights::ALL).unwrap_err(),
            FsError::AccessDenied
        );
        // Asking for a subset yields exactly that subset.
        assert_eq!(Rights::ALL.attenuate(Rights::READ).unwrap(), Rights::READ);
        assert_eq!(read_only.attenuate(Rights::READ).unwrap(), Rights::READ);
    }

    #[test]
    fn share_is_never_implied() {
        // Holding everything else must not confer SHARE.
        let no_share = Rights::READ
            .union(Rights::WRITE)
            .union(Rights::MUTATE_NAMESPACE)
            .union(Rights::BIND_EXECUTABLE);
        assert!(!no_share.contains(Rights::SHARE));
        assert_eq!(
            no_share.attenuate(Rights::SHARE).unwrap_err(),
            FsError::AccessDenied
        );
    }

    #[test]
    fn unknown_right_bits_are_discarded() {
        // An unrecognized wire flag must not become authority.
        let forged = Rights::from_bits_truncating(0xff);
        assert_eq!(forged, Rights::ALL);
        assert_eq!(forged.bits(), Rights::ALL.bits());
        assert_eq!(Rights::from_bits_truncating(1 << 7), Rights::NONE);
    }

    #[test]
    fn require_reports_the_missing_right() {
        assert!(Rights::ALL.require(Rights::WRITE).is_ok());
        assert_eq!(
            Rights::READ.require(Rights::WRITE).unwrap_err(),
            FsError::AccessDenied
        );
        assert!(Rights::NONE.require(Rights::NONE).is_ok());
    }

    // ── proofs ──────────────────────────────────────────────────────────────

    #[test]
    fn a_root_proof_is_empty() {
        let proof = EdgeProof::root();
        assert_eq!(proof.depth(), 0);
        assert!(proof.edges().is_empty());
    }

    #[test]
    fn proofs_extend_up_to_the_depth_bound_then_refuse() {
        let mut ns = Namespace::<u32>::new();
        let root = ns.create_dir_object().unwrap();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &Name::new("f").unwrap(), file).unwrap();

        let mut proof = EdgeProof::root();
        for expected in 1..=MAX_PATH_DEPTH {
            proof = proof.extend(edge).unwrap();
            assert_eq!(proof.depth(), expected);
        }
        // Refuses rather than truncating: a truncated proof would stop
        // revalidating its upper edges.
        assert_eq!(proof.extend(edge).unwrap_err(), FsError::Range);
        assert_eq!(proof.depth(), MAX_PATH_DEPTH);
    }

    #[test]
    fn extend_does_not_mutate_the_original() {
        let mut ns = Namespace::<u32>::new();
        let root = ns.create_dir_object().unwrap();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &Name::new("f").unwrap(), file).unwrap();

        let base = EdgeProof::root();
        let extended = base.extend(edge).unwrap();
        assert_eq!(base.depth(), 0);
        assert_eq!(extended.depth(), 1);
    }

    // ── sessions, grants, handles ───────────────────────────────────────────

    /// An authority arena with one attached session over a small namespace.
    fn fixture() -> (Authority, Namespace<u32>, SessionId, ObjectId, GrantId) {
        let mut ns = Namespace::<u32>::new();
        let mut auth = Authority::new();
        let session = SessionId::new(0);
        let root = ns.create_dir_object().unwrap();
        auth.attach_session(session).unwrap();
        let grant = auth.install_root(session, root).unwrap();
        ns.object_mut(root).unwrap().grants_inc();
        (auth, ns, session, root, grant)
    }

    #[test]
    fn a_private_root_carries_full_rights_and_is_its_own_issuer() {
        let (auth, _ns, session, root, grant) = fixture();
        let edge = auth.edge(grant).unwrap();
        assert_eq!(edge.root(), root);
        assert_eq!(edge.rights(), Rights::ALL);
        assert_eq!(edge.kind(), AuthorityKind::PrivateRoot);
        assert_eq!(edge.holder(), session);
        assert_eq!(edge.issuer(), session);
        assert_eq!(auth.root_grant(session).unwrap(), grant);
        assert!(auth.is_attached(session));
    }

    #[test]
    fn attaching_a_session_twice_is_refused() {
        let (mut auth, _ns, session, _root, _grant) = fixture();
        assert_eq!(
            auth.attach_session(session).unwrap_err(),
            FsError::AlreadyExists
        );
    }

    #[test]
    fn an_out_of_range_session_is_refused() {
        let mut auth = Authority::new();
        let bogus = SessionId::new(MAX_SESSIONS as u16);
        assert_eq!(auth.attach_session(bogus).unwrap_err(), FsError::Range);
        assert!(!auth.is_attached(bogus));
    }

    #[test]
    fn an_unattached_session_has_no_root() {
        let auth = Authority::new();
        assert_eq!(
            auth.root_grant(SessionId::new(1)).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn a_handle_is_invalid_when_presented_by_another_session() {
        let (mut auth, mut ns, session_a, root, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let handle_a = auth
            .insert_handle(session_a, root, grant_a, EdgeProof::root(), Rights::ALL)
            .unwrap();
        // Session B presenting A's handle bits must be refused outright.
        assert_eq!(
            auth.handle(session_b, handle_a).unwrap_err(),
            FsError::InvalidHandle
        );
        assert!(auth.handle(session_a, handle_a).is_ok());
    }

    #[test]
    fn forged_handles_fail_closed() {
        let (auth, _ns, session, _root, _grant) = fixture();
        assert_eq!(
            auth.handle(session, Handle::from_bits(0)).unwrap_err(),
            FsError::InvalidHandle
        );
        assert_eq!(
            auth.handle(session, Handle::from_bits(u64::MAX))
                .unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn one_session_cannot_close_anothers_handle() {
        let (mut auth, mut ns, session_a, root, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let handle_a = auth
            .insert_handle(session_a, root, grant_a, EdgeProof::root(), Rights::ALL)
            .unwrap();
        assert_eq!(
            auth.remove_handle(session_b, handle_a).unwrap_err(),
            FsError::InvalidHandle
        );
        // A's handle survived the attempt.
        assert!(auth.handle(session_a, handle_a).is_ok());
        assert_eq!(auth.live_handles(session_a), 1);
    }

    #[test]
    fn the_per_session_handle_ceiling_is_enforced_per_session() {
        let (mut auth, mut ns, session_a, root, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        let grant_b = auth.install_root(session_b, root_b).unwrap();

        for _ in 0..MAX_HANDLES_PER_SESSION {
            auth.insert_handle(session_a, root, grant_a, EdgeProof::root(), Rights::ALL)
                .unwrap();
        }
        assert_eq!(
            auth.insert_handle(session_a, root, grant_a, EdgeProof::root(), Rights::ALL)
                .unwrap_err(),
            FsError::OutOfHandles
        );
        // One client exhausting its own ceiling must not starve another.
        assert!(
            auth.insert_handle(session_b, root_b, grant_b, EdgeProof::root(), Rights::ALL)
                .is_ok()
        );
        assert_eq!(auth.live_handles(session_a), MAX_HANDLES_PER_SESSION);
        assert_eq!(auth.live_handles(session_b), 1);
    }

    #[test]
    fn closing_a_handle_frees_its_quota() {
        let (mut auth, _ns, session, root, grant) = fixture();
        let handle = auth
            .insert_handle(session, root, grant, EdgeProof::root(), Rights::ALL)
            .unwrap();
        assert_eq!(auth.live_handles(session), 1);
        auth.remove_handle(session, handle).unwrap();
        assert_eq!(auth.live_handles(session), 0);
        // And the closed handle is gone for good.
        assert_eq!(
            auth.remove_handle(session, handle).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    // ── revalidation ────────────────────────────────────────────────────────

    #[test]
    fn a_root_handle_revalidates() {
        let (mut auth, ns, session, root, grant) = fixture();
        let handle = auth
            .insert_handle(session, root, grant, EdgeProof::root(), Rights::ALL)
            .unwrap();
        let route = auth.revalidate(session, handle, &ns).unwrap();
        assert_eq!(route.object, root);
        assert_eq!(route.rights, Rights::ALL);
    }

    #[test]
    fn a_handle_through_one_edge_revalidates_then_fails_when_the_edge_goes() {
        let (mut auth, mut ns, session, root, grant) = fixture();
        let file = ns.create_file_object().unwrap();
        let name = Name::new("f").unwrap();
        let edge = ns.link(root, &name, file).unwrap();

        let handle = auth
            .insert_handle(
                session,
                file,
                grant,
                EdgeProof::root().extend(edge).unwrap(),
                Rights::ALL,
            )
            .unwrap();
        assert_eq!(auth.revalidate(session, handle, &ns).unwrap().object, file);

        // Remove the one edge the proof depends on.
        ns.detach(root, &name).unwrap();
        assert_eq!(
            auth.revalidate(session, handle, &ns).unwrap_err(),
            FsError::AccessLost
        );
    }

    #[test]
    fn removing_an_intermediate_edge_invalidates_a_deeper_handle() {
        let (mut auth, mut ns, session, root, grant) = fixture();
        let mid = ns.create_dir_object().unwrap();
        let leaf = ns.create_file_object().unwrap();
        let mid_name = Name::new("mid").unwrap();
        let mid_edge = ns.link(root, &mid_name, mid).unwrap();
        let leaf_edge = ns.link(mid, &Name::new("leaf").unwrap(), leaf).unwrap();

        let proof = EdgeProof::root()
            .extend(mid_edge)
            .unwrap()
            .extend(leaf_edge)
            .unwrap();
        let handle = auth
            .insert_handle(session, leaf, grant, proof, Rights::ALL)
            .unwrap();
        assert!(auth.revalidate(session, handle, &ns).is_ok());

        // Emptying and removing the intermediate directory cuts the chain at
        // its first link even though the leaf edge itself is what vanishes.
        ns.detach(mid, &Name::new("leaf").unwrap()).unwrap();
        ns.detach(root, &mid_name).unwrap();
        assert_eq!(
            auth.revalidate(session, handle, &ns).unwrap_err(),
            FsError::AccessLost
        );
    }

    #[test]
    fn retiring_a_grant_invalidates_handles_that_entered_through_it() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        // A shares one of its files with B, read-only.
        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("shared").unwrap(), file)
            .unwrap();
        let grant = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::READ_ONLY,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();
        ns.object_mut(file).unwrap().grants_inc();

        let handle = auth
            .insert_handle(session_b, file, grant, EdgeProof::root(), Rights::READ)
            .unwrap();
        let route = auth.revalidate(session_b, handle, &ns).unwrap();
        assert_eq!(route.rights, Rights::READ);
        assert!(route.rights.require(Rights::WRITE).is_err());

        // A revokes. B's existing handle must fail on its very next use.
        assert_eq!(
            auth.retire_edge(session_a, grant).unwrap().as_slice(),
            &[file]
        );
        assert_eq!(
            auth.revalidate(session_b, handle, &ns).unwrap_err(),
            FsError::AccessLost
        );
        // The file object itself is untouched — A still has its own name for it.
        assert!(ns.exists(file));
    }

    #[test]
    fn a_third_session_cannot_revoke_someone_elses_grant() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let session_c = SessionId::new(2);
        for session in [session_b, session_c] {
            let r = ns.create_dir_object().unwrap();
            auth.attach_session(session).unwrap();
            auth.install_root(session, r).unwrap();
        }
        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let grant = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();

        assert_eq!(
            auth.retire_edge(session_c, grant).unwrap_err(),
            FsError::AccessDenied
        );
        // Still live for its holder.
        assert!(auth.edge(grant).is_ok());
        // Both the issuer and the holder may end it.
        assert!(auth.retire_edge(session_b, grant).is_ok());
    }

    #[test]
    fn validated_descendant_pins_its_whole_lineage_during_retirement() {
        let (auth, ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let session_c = SessionId::new(2);
        for session in [session_b, session_c] {
            let root = ns.create_dir_object().unwrap();
            auth.attach_session(session).unwrap();
            auth.install_root(session, root).unwrap();
        }
        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let parent = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::ALL,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();
        let child = auth
            .insert_grant(
                session_b,
                session_c,
                parent,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();

        let in_flight = auth.validated_edge(child).unwrap();
        assert_eq!(auth.retire_edge(session_a, parent), Err(FsError::Busy));
        assert!(auth.edge(parent).is_ok() && auth.edge(child).is_ok());
        drop(in_flight);

        assert_eq!(auth.retire_edge(session_a, parent).unwrap().len(), 2);
        assert_eq!(auth.edge(parent).unwrap_err(), FsError::AccessLost);
        assert_eq!(auth.edge(child).unwrap_err(), FsError::AccessLost);
    }

    #[test]
    fn a_private_root_cannot_be_revoked() {
        let (mut auth, _ns, session, _root, grant) = fixture();
        assert_eq!(
            auth.retire_edge(session, grant).unwrap_err(),
            FsError::Unsupported
        );
        assert!(auth.edge(grant).is_ok());
    }

    #[test]
    fn a_grant_to_an_unattached_session_is_refused() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        // Nobody is attached at slot 3; a grant there could never be opened
        // and would pin the file forever.
        assert_eq!(
            auth.insert_grant(
                session_a,
                SessionId::new(3),
                grant_a,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn a_proof_naming_a_disconnected_edge_is_refused() {
        // A handcrafted proof whose edge does not hang beneath the grant root
        // must not validate just because the edge itself is live.
        let (mut auth, mut ns, session, root, grant) = fixture();
        let elsewhere = ns.create_dir_object().unwrap();
        ns.link(root, &Name::new("elsewhere").unwrap(), elsewhere)
            .unwrap();
        let stray = ns.create_file_object().unwrap();
        let stray_edge = ns
            .link(elsewhere, &Name::new("stray").unwrap(), stray)
            .unwrap();

        // Claim a route from the root straight to `stray` via an edge whose
        // parent is `elsewhere`, not `root`.
        let handle = auth
            .insert_handle(
                session,
                stray,
                grant,
                EdgeProof::root().extend(stray_edge).unwrap(),
                Rights::ALL,
            )
            .unwrap();
        assert_eq!(
            auth.revalidate(session, handle, &ns).unwrap_err(),
            FsError::AccessLost
        );
    }

    #[test]
    fn a_proof_arriving_at_the_wrong_object_is_refused() {
        let (mut auth, mut ns, session, root, grant) = fixture();
        let real = ns.create_file_object().unwrap();
        let other = ns.create_file_object().unwrap();
        let edge = ns.link(root, &Name::new("real").unwrap(), real).unwrap();
        ns.link(root, &Name::new("other").unwrap(), other).unwrap();

        // The chain is intact but lands on `real`, not the claimed `other`.
        let handle = auth
            .insert_handle(
                session,
                other,
                grant,
                EdgeProof::root().extend(edge).unwrap(),
                Rights::ALL,
            )
            .unwrap();
        assert_eq!(
            auth.revalidate(session, handle, &ns).unwrap_err(),
            FsError::AccessLost
        );
    }

    #[test]
    fn a_handle_survives_a_rename_of_the_edge_it_proves_through() {
        let (mut auth, mut ns, session, root, grant) = fixture();
        let file = ns.create_file_object().unwrap();
        let edge = ns.link(root, &Name::new("before").unwrap(), file).unwrap();
        let handle = auth
            .insert_handle(
                session,
                file,
                grant,
                EdgeProof::root().extend(edge).unwrap(),
                Rights::ALL,
            )
            .unwrap();

        ns.rename(
            root,
            &Name::new("before").unwrap(),
            &Name::new("after").unwrap(),
        )
        .unwrap();
        // Rename keeps the edge, so the route — and the handle — survive.
        assert_eq!(auth.revalidate(session, handle, &ns).unwrap().object, file);
    }

    #[test]
    fn a_handle_through_a_moved_edge_fails_while_a_parallel_link_survives() {
        let (mut auth, mut ns, session, root, grant) = fixture();
        let dst = ns.create_dir_object().unwrap();
        ns.link(root, &Name::new("dst").unwrap(), dst).unwrap();
        let file = ns.create_file_object().unwrap();
        let src_edge = ns.link(root, &Name::new("f").unwrap(), file).unwrap();
        let parallel = ns.link(root, &Name::new("also").unwrap(), file).unwrap();

        let via_src = auth
            .insert_handle(
                session,
                file,
                grant,
                EdgeProof::root().extend(src_edge).unwrap(),
                Rights::ALL,
            )
            .unwrap();
        let via_parallel = auth
            .insert_handle(
                session,
                file,
                grant,
                EdgeProof::root().extend(parallel).unwrap(),
                Rights::ALL,
            )
            .unwrap();

        ns.relocate(
            root,
            &Name::new("f").unwrap(),
            dst,
            &Name::new("moved").unwrap(),
        )
        .unwrap();

        assert_eq!(
            auth.revalidate(session, via_src, &ns).unwrap_err(),
            FsError::AccessLost
        );
        assert_eq!(
            auth.revalidate(session, via_parallel, &ns).unwrap().object,
            file
        );
    }

    // ── attachment names and teardown ───────────────────────────────────────

    #[test]
    fn attachment_names_are_per_recipient() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let shared = Name::new("shared").unwrap();
        auth.insert_grant(
            session_a,
            session_b,
            grant_a,
            file,
            Rights::READ,
            AuthorityQuota::MAX,
            Some(shared),
        )
        .unwrap();

        assert!(auth.display_name_taken(session_b, &shared).unwrap());
        // A's own view is unaffected by what B calls it.
        assert!(!auth.display_name_taken(session_a, &shared).unwrap());
    }

    #[test]
    fn grants_of_lists_a_sessions_routes() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        auth.insert_grant(
            session_a,
            session_b,
            grant_a,
            file,
            Rights::READ,
            AuthorityQuota::MAX,
            Some(Name::new("gift").unwrap()),
        )
        .unwrap();

        let listed = auth.grants_of(session_b).unwrap();
        assert_eq!(listed.len(), 2, "its private root plus one grant");
        assert!(
            listed
                .iter()
                .any(|(_, kind, _, _, _)| *kind == AuthorityKind::PrivateRoot)
        );
        let gift = listed
            .iter()
            .find(|(_, kind, _, _, _)| *kind == AuthorityKind::Grant)
            .unwrap();
        assert_eq!(gift.2, Rights::READ);
        assert_eq!(gift.3, AuthorityQuota::MAX);
        assert_eq!(gift.4.unwrap().as_str(), "gift");
    }

    #[test]
    fn grant_snapshots_refuse_contention_instead_of_hiding_a_live_edge() {
        let (auth, ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();
        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let shared = Name::new("shared").unwrap();
        let gift = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                Some(shared),
            )
            .unwrap();
        let (edges, local) = auth.edge_slot(gift).unwrap();
        let _writer = edges.try_write(local).unwrap();

        assert_eq!(auth.grants_of(session_b).unwrap_err(), FsError::Busy);
        assert_eq!(
            auth.display_name_taken(session_b, &shared),
            Err(FsError::Busy)
        );
    }

    #[test]
    fn task_view_name_claims_are_scoped_per_session() {
        let (auth, ns, session_a, _root_a, _grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let _a = auth.claim_view(session_a).unwrap();
        let b = auth.claim_view(session_b).unwrap();
        assert!(matches!(auth.claim_view(session_b), Err(FsError::Busy)));
        drop(b);
        assert!(auth.claim_view(session_b).is_ok());
    }

    #[test]
    fn detaching_a_session_drops_its_handles_and_edges() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let to_b = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();
        let a_handle = auth
            .insert_handle(session_a, root_a, grant_a, EdgeProof::root(), Rights::ALL)
            .unwrap();
        let b_handle = auth
            .insert_handle(session_b, file, to_b, EdgeProof::root(), Rights::READ)
            .unwrap();

        let stranded = auth.detach_session(session_b).unwrap();
        // B's own root and its grant were the edges it held.
        assert_eq!(stranded.len(), 2);
        assert!(stranded.contains(&root_b) && stranded.contains(&file));
        assert!(!auth.is_attached(session_b));
        assert_eq!(
            auth.handle(session_b, b_handle).unwrap_err(),
            FsError::InvalidHandle
        );
        // A is entirely unaffected.
        assert!(auth.handle(session_a, a_handle).is_ok());
        assert!(auth.revalidate(session_a, a_handle, &ns).is_ok());
    }

    #[test]
    fn contended_session_detach_republishes_every_exact_record_for_retry() {
        let (auth, _ns, session, root, grant) = fixture();
        let handle = auth
            .insert_handle(session, root, grant, EdgeProof::root(), Rights::ALL)
            .unwrap();
        let in_flight = auth.handle(session, handle).unwrap();

        assert_eq!(auth.detach_session(session), Err(FsError::Busy));
        assert!(
            auth.handle(session, handle).is_ok(),
            "a refused teardown reopens the same session generation"
        );
        drop(in_flight);

        let stranded = auth.detach_session(session).unwrap();
        assert_eq!(stranded.as_slice(), &[root]);
        assert_eq!(
            auth.handle(session, handle).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn detaching_invalidates_grants_derived_from_the_sessions_authority() {
        let (mut auth, mut ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let file = ns.create_file_object().unwrap();
        ns.link(root_a, &Name::new("f").unwrap(), file).unwrap();
        let to_b = auth
            .insert_grant(
                session_a,
                session_b,
                grant_a,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap();

        // A's private root was the grant's parent. Once A goes away the
        // derived route cannot survive as ambient authority in B.
        auth.detach_session(session_a).unwrap();
        assert_eq!(auth.edge(to_b).unwrap_err(), FsError::AccessLost);
    }

    #[test]
    fn detaching_an_unattached_session_is_refused() {
        let mut auth = Authority::new();
        assert_eq!(
            auth.detach_session(SessionId::new(0)).unwrap_err(),
            FsError::InvalidHandle
        );
    }

    #[test]
    fn a_full_session_is_refused_without_touching_another() {
        let (auth, ns, session_a, root, root_grant) = fixture();
        let session_b = SessionId::new(1);
        let root_b = ns.create_dir_object().unwrap();
        auth.attach_session(session_b).unwrap();
        auth.install_root(session_b, root_b).unwrap();

        let file = ns.create_file_object().unwrap();
        ns.link(root, &Name::new("f").unwrap(), file).unwrap();
        let self_grant = || {
            auth.insert_grant(
                session_a,
                session_a,
                root_grant,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
        };
        // The root is the first of the session's edges.
        for _ in 1..MAX_EDGES_PER_SESSION {
            self_grant().unwrap();
        }
        assert_eq!(self_grant().unwrap_err(), FsError::OutOfEdges);
        assert_eq!(auth.edge_count(), MAX_EDGES_PER_SESSION + 1);

        // The other session's storage is its own: it still takes a grant,
        // even one issued by the full session.
        auth.insert_grant(
            session_a,
            session_b,
            root_grant,
            file,
            Rights::READ,
            AuthorityQuota::MAX,
            None,
        )
        .unwrap();
        assert_eq!(auth.edge_count(), MAX_EDGES_PER_SESSION + 2);
    }

    /// Attach `session` with a fresh private root.
    fn attach(auth: &Authority, ns: &Namespace<u32>, session: SessionId) -> (ObjectId, GrantId) {
        let root = ns.create_dir_object().unwrap();
        auth.attach_session(session).unwrap();
        let grant = auth.install_root(session, root).unwrap();
        (root, grant)
    }

    #[test]
    fn every_slot_the_host_can_name_attaches_a_session() {
        let ns = Namespace::<u32>::new();
        let auth = Authority::new();
        let mut opened = std::vec::Vec::new();
        for slot in 0..MAX_SESSIONS {
            let session = SessionId::new(slot as u16);
            let (root, grant) = attach(&auth, &ns, session);
            let handle = auth
                .insert_handle(session, root, grant, EdgeProof::root(), Rights::READ)
                .unwrap();
            opened.push((session, handle));
        }
        assert_eq!(auth.live_sessions() as usize, MAX_SESSIONS);
        assert_eq!(auth.session_arenas(), MAX_SESSIONS);
        assert_eq!(auth.edge_count(), MAX_SESSIONS);
        for (session, handle) in &opened {
            assert_eq!(auth.live_handles(*session), 1);
            assert_eq!(
                auth.revalidate(*session, *handle, &ns).unwrap().rights,
                Rights::READ
            );
        }
        // Past the slot space there is nothing to attach.
        assert_eq!(
            auth.attach_session(SessionId::new(MAX_SESSIONS as u16))
                .unwrap_err(),
            FsError::Range
        );
    }

    #[test]
    fn storage_follows_the_sessions_attached_not_the_slot_space() {
        let (auth, ns, _session, _root, _grant) = fixture();
        assert_eq!(auth.session_arenas(), 1);
        // A slot far up the space costs one arena, not every slot below it.
        let far = SessionId::new(crate::bounds::FIRST_TASK_VIEW_SESSION as u16 + 1);
        attach(&auth, &ns, far);
        assert_eq!(auth.session_arenas(), 2);
        // Detaching keeps the slot's arena for its next occupant.
        auth.detach_session(far).unwrap();
        assert_eq!(auth.session_arenas(), 2);
        attach(&auth, &ns, far);
        assert_eq!(auth.session_arenas(), 2);
    }

    #[test]
    fn a_handle_names_its_sessions_slot_and_no_other() {
        let (auth, ns, session_a, root_a, grant_a) = fixture();
        let session_b = SessionId::new(9);
        let (root_b, grant_b) = attach(&auth, &ns, session_b);
        let handle_a = auth
            .insert_handle(session_a, root_a, grant_a, EdgeProof::root(), Rights::READ)
            .unwrap();
        let handle_b = auth
            .insert_handle(session_b, root_b, grant_b, EdgeProof::root(), Rights::READ)
            .unwrap();
        // The same arena position in two sessions is two different handles.
        assert_ne!(handle_a, handle_b);
        assert_eq!(
            auth.handle(session_b, handle_a).unwrap_err(),
            FsError::InvalidHandle
        );
        assert_eq!(
            auth.handle(session_a, handle_b).unwrap_err(),
            FsError::InvalidHandle
        );
        // An index past every session's slots names nothing.
        let forged = Handle::from_key(SlotKey::new(MAX_HANDLE_INDEX as u32, 1));
        assert_eq!(
            auth.handle(session_a, forged).unwrap_err(),
            FsError::InvalidHandle
        );
        assert!(auth.handle(session_a, handle_a).is_ok());
        assert!(auth.handle(session_b, handle_b).is_ok());
    }

    #[test]
    fn a_session_parked_in_its_attach_transition_detaches_empty() {
        let (auth, ns, _session, _root, _grant) = fixture();
        let parked = SessionId::new(7);
        auth.attach_session(parked).unwrap();
        assert!(!auth.is_attached(parked));
        let retired = auth.detach_session(parked).unwrap();
        assert!(retired.is_empty());
        assert_eq!(auth.live_sessions(), 1);
        // The slot takes its next occupant, rooted as usual.
        let (root, grant) = attach(&auth, &ns, parked);
        assert_eq!(auth.root_grant(parked).unwrap(), grant);
        assert_eq!(auth.root_object(parked).unwrap(), root);
    }

    #[test]
    fn a_reused_slot_refuses_its_previous_occupants_ids() {
        let (auth, ns, _session, _root, _grant) = fixture();
        let slot = SessionId::new(5);
        let (root, grant) = attach(&auth, &ns, slot);
        let handle = auth
            .insert_handle(slot, root, grant, EdgeProof::root(), Rights::READ)
            .unwrap();
        auth.detach_session(slot).unwrap();

        let (next_root, next_grant) = attach(&auth, &ns, slot);
        // The arena slots are reused; their generations moved on.
        assert_ne!(next_grant, grant);
        assert_eq!(
            auth.handle(slot, handle).unwrap_err(),
            FsError::InvalidHandle
        );
        assert_eq!(auth.edge(grant).unwrap_err(), FsError::AccessLost);
        let next = auth
            .insert_handle(slot, next_root, next_grant, EdgeProof::root(), Rights::READ)
            .unwrap();
        assert_ne!(next, handle);
        assert!(auth.revalidate(slot, next, &ns).is_ok());
    }

    #[test]
    fn a_lineage_deeper_than_the_bound_is_refused() {
        let (auth, ns, session_a, root, root_grant) = fixture();
        let session_b = SessionId::new(1);
        attach(&auth, &ns, session_b);
        let file = ns.create_file_object().unwrap();
        ns.link(root, &Name::new("f").unwrap(), file).unwrap();

        // Alternate holders so the chain, not a session's edge count, is
        // what runs out: each grant is issued by its parent's holder.
        let mut parent = root_grant;
        let mut issuer = session_a;
        for depth in 2..=MAX_LINEAGE_DEPTH {
            let holder = if issuer == session_a {
                session_b
            } else {
                session_a
            };
            parent = auth
                .insert_grant(
                    issuer,
                    holder,
                    parent,
                    file,
                    Rights::READ,
                    AuthorityQuota::MAX,
                    None,
                )
                .unwrap();
            assert_eq!(auth.validated_edge(parent).unwrap().lineage_len(), depth);
            issuer = holder;
        }
        let holder = if issuer == session_a {
            session_b
        } else {
            session_a
        };
        assert_eq!(
            auth.insert_grant(
                issuer,
                holder,
                parent,
                file,
                Rights::READ,
                AuthorityQuota::MAX,
                None,
            )
            .unwrap_err(),
            FsError::Range
        );
        // The deepest edge still validates its whole lineage.
        assert!(auth.validated_edge(parent).is_ok());
    }

    #[test]
    fn detaching_retires_a_closure_spread_over_many_sessions_children_first() {
        let (auth, ns, session_a, root, root_grant) = fixture();
        let file = ns.create_file_object().unwrap();
        ns.link(root, &Name::new("f").unwrap(), file).unwrap();

        // A fan of holders, each re-granting to the next: every edge below
        // the root descends from session A's authority.
        let holders = 24;
        let mut parent = root_grant;
        let mut issuer = session_a;
        let mut descendants = 0;
        for index in 1..=holders {
            let holder = SessionId::new(index as u16);
            attach(&auth, &ns, holder);
            let grant = |parent| {
                auth.insert_grant(
                    issuer,
                    holder,
                    parent,
                    file,
                    Rights::READ,
                    AuthorityQuota::MAX,
                    None,
                )
                .unwrap()
            };
            // A sibling beside the chain, from the same parent, then the
            // chain's next link.
            grant(parent);
            parent = grant(parent);
            descendants += 2;
            issuer = holder;
        }
        let deepest = parent;
        // Every session's root, plus the chain and its siblings.
        assert_eq!(auth.edge_count(), 1 + holders + descendants);

        let retired = auth.detach_session(session_a).unwrap();
        // A's root and every edge descending from it, once each.
        assert_eq!(retired.len(), 1 + descendants);
        assert_eq!(retired.iter().filter(|object| **object == root).count(), 1);
        assert_eq!(auth.edge_count(), holders);
        assert_eq!(auth.edge(deepest).unwrap_err(), FsError::AccessLost);
        for index in 1..=holders {
            assert!(auth.is_attached(SessionId::new(index as u16)));
        }
    }

    #[test]
    fn a_handle_record_is_small_and_its_proof_holds_its_own_depth() {
        let (_auth, ns, _session, root, _grant) = fixture();
        let dir = ns.create_dir_object().unwrap();
        let edge = ns.link(root, &Name::new("d").unwrap(), dir).unwrap();
        // The proof is a pointer and a length; the edges live beside it at
        // the depth actually walked.
        assert!(core::mem::size_of::<OpenHandle>() <= 64);
        let proof = EdgeProof::root().extend(edge).unwrap();
        assert_eq!(proof.edges(), &[edge]);
        assert!(SessionArena::BYTES < 8 * 1024);
        assert_eq!(SESSION_HEAP_BYTES, MAX_SESSIONS * SessionArena::BYTES);
    }
}
