//! The filesystem service: its request/reply vocabulary, and the dispatch that
//! drives [`Engine`](crate::engine::Engine) from it.
//!
//! One enum in, one enum out — the shape every `kore-svc-*` uses.
//! [`crate::wire`] carries the codecs to and from the frozen records in
//! `kore_abi::svc`, and the engine below knows about neither.
//!
//! # Where bulk data goes
//!
//! File bytes and directory-listing entries ride the pair's separately shared
//! **data window**, addressed by an offset and a length *into that window*
//! (the retained-return contract). Nothing here carries payload: a `Read` says where in the window
//! to land the bytes, and its reply says how many arrived. That keeps the reply
//! record 48 bytes wide regardless of transfer size.
//!
//! The client carves that window out of its own memory and delivers it through
//! the return channel, then names the landing slot in
//! [`FsRequest::AttachWindow`] — `OutReqWire::OP_ATTACH_TX`'s precedent. Until
//! it does, every bulk verb refuses [`FsError::NoWindow`]: a read with nowhere
//! to land is not a read.
//!
//! Production clients deliver the view through the serving pair's retained
//! return leg. `AttachWindow` adopts only the capability returned for the
//! request's announced slot; the request's integer fields cannot select a
//! mapping by themselves. Exec uses a fixed administration-pair window for
//! immutable image-source reads, while recyclable process faces receive
//! independent pair-bound windows.
//!
//! # Sessions
//!
//! [`FsService::dispatch`] takes the session as a **parameter**, never from the
//! request, because the request crossed a process boundary and the session did
//! not. The serving host derives it from the pair that delivered the bytes
//! (the pair-derived identity contract). `ServeHost` supplies it per pair: `Serve::via` names the
//! arrival, and it is 0 for the ordinary single-pair row, so one client and
//! several are the same line. `ensure_session` is what creates it, on first
//! contact — every piece of per-client state in this file is keyed the same
//! way, so a pair that can be served is a pair whose state exists.
//!
//! # Why dispatch is synchronous
//!
//! `SystemService::serve` is deliberately synchronous and allocation-free,
//! while the engine is async so a hardware-backed provider cannot block a
//! worker. Over the RAM provider those futures are `Ready` on first poll, so
//! driving them here with [`block_on`](crate::block_on) is exact rather than a
//! shortcut. **It stops being exact the moment a provider genuinely parks**;
//! a blocking provider must instead be driven by the executor without nesting
//! an async wait inside this synchronous serve step.

pub mod contract;

pub use contract::FsEndowment;

use alloc::vec::Vec;

use kore_abi::svc::{FS_SEARCH_PATH_BYTES, FsDirEntryWire, FsReqWire, FsSearchHitWire};
use kore_rt::{
    cap::{ChanSendToken, ReturnedCapability, verbs},
    service::{PeerClose, Serve},
    svc::{
        PeerAdmissionRefusal, PeerAssemblyRefusal, PeerBundleAssembler, PeerBundlePart, PeerId,
        PeerMapPlan,
    },
};
use kore_svc_extract::schema::InputIdentity;

use crate::{
    ServiceAlloc,
    authority::{GrantId, Handle, Retired, Rights, SessionId},
    block_on,
    bounds::{FIRST_TASK_VIEW_SESSION, MAX_SESSIONS},
    engine::Engine,
    error::{FsError, FsResult},
    namespace::Name,
    object::ObjectKind,
    provider::{RamProvider, StorageProvider},
    window::DataWindow,
};

/// The bounded RAM-backed service composition.
pub type RamService = FsService<RamProvider>;

/// Stable display name of the read-only system namespace grant installed for
/// exec's fixed administration session. The name is discovery metadata only:
/// opening it still requires the pair-bound [`GrantId`] written by
/// [`FsRequest::ListGrants`].
pub const SYSTEM_SOURCE_GRANT: &str = "sys";

/// Conventional names in one task's native filesystem view.
///
/// These are presentation roles, never global objects: the launcher selects
/// the exact object and quota behind each pair-bound grant. Two tasks may see
/// unrelated objects under the same role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeView {
    App,
    Data,
    Config,
    Tmp,
    Shared,
}

impl NativeView {
    pub const ALL: [Self; 5] = [Self::App, Self::Data, Self::Config, Self::Tmp, Self::Shared];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::App => "app",
            Self::Data => "data",
            Self::Config => "config",
            Self::Tmp => "tmp",
            Self::Shared => "shared",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::App => 0,
            Self::Data => 1,
            Self::Config => 2,
            Self::Tmp => 3,
            Self::Shared => 4,
        }
    }

    pub fn from_name(name: &Name) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|view| view.as_str().as_bytes() == name.as_bytes())
    }

    pub fn name(self) -> Name {
        Name::new(self.as_str()).expect("native view names are statically valid")
    }
}

/// Name of the directory beneath session zero's root that holds the durable
/// workspace. On-disk state: it is an ordinary namespace edge, serialized and
/// recovered like any other, and renaming it would orphan every existing
/// volume's user data.
pub const SYSTEM_WORKSPACE_DIR: &str = "work";

/// Bytes of an init-minted face's home label: `face-` and one digit.
pub const FACE_HOME_LABEL_BYTES: usize = 6;

/// The durable home of the init-minted face `face`: `face-N`, beneath
/// [`SYSTEM_WORKSPACE_DIR`]. Exec names a face's home by its index when it
/// mints the face, and init mints its terminals in one deterministic order,
/// so `face-N` is terminal N's home on every boot. A host tool writing into
/// terminal N's home (`tools/kore-vol`) finds it by this same name. `None`
/// past the one digit the label holds.
pub const fn face_home_label(face: u8) -> Option<[u8; FACE_HOME_LABEL_BYTES]> {
    if face > 9 {
        return None;
    }
    let mut label = *b"face-0";
    label[5] = b'0' + face;
    Some(label)
}

/// The shared executable directory beneath the persisted root: where the boot
/// archive's members land once, and where a later install writes.
pub const SYSTEM_APPS_DIR: &str = "app";

/// Canonical display name of the read-only application view in a user task.
///
/// The durable backing directory stays `app` for recovery compatibility.  A
/// task sees that one object as `apps`, whose children carry the structured
/// native and compatibility projections.  This is a view name, not a second
/// directory or byte copy.
///
/// This is the ONLY application view a terminal is granted. A second, flat
/// `app` grant onto the same object rode beside it while callers migrated;
/// nothing spells the flat form any more, and carrying it showed one store
/// object under two names — which on a live machine reads as a stale
/// projection that survived a boot and invites an attempt to delete it.
/// There was never anything to delete: the object behind the flat name is
/// this one, so the view ends by not being granted, not by any unlink.
pub const SYSTEM_APPS_VIEW: &str = "apps";

/// Display name of the administration session's writable grant over
/// [`SYSTEM_APPS_DIR`].
///
/// Deliberately *not* the directory's own name. The administration session's
/// structural root is the persisted root, so a grant displayed `app` would
/// collide with the `app` directory hanging beneath it — `view_name_taken`
/// claims the combined display-name/child-name set. The grant is what exec
/// discovers through `ListGrants`; the directory is what a terminal's view links
/// into.
pub const SYSTEM_APPS_GRANT: &str = "sys-app";
/// The shared **library** directory beneath the persisted root — the
/// `libs/` half of the library convention: archive libraries (and, later,
/// a package's `lib/` members) live here, stamped `EXECUTABLE_IMAGE`, and
/// a task links them into itself at runtime through its own arena
/// (`res/IMAGE.md` §8).  Deliberately not UNIX `lib`: a system view, not a
/// search path.
pub const SYSTEM_LIBS_DIR: &str = "lib";
/// The READ-only display name every task view carries for it, beside
/// `apps` — the leaf's spelling, because a late loader opens it by name.
pub const SYSTEM_LIBS_VIEW: &str = kore_dagfs::LIBS_VIEW;
/// The administration session's writable grant over it (`sys-lib`, for the
/// same collision reason as `sys-app`).
pub const SYSTEM_LIBS_GRANT: &str = "sys-lib";
/// The shared **data** directory beneath the persisted root: the boot
/// archive's DATA members (`ARCHIVE_MEMBER_DATA` — stock `.wasm` modules,
/// for one) land here once, under a per-kind subdirectory
/// (`shared/wasm/hello.wasm` as a task sees it), and a later install
/// writes here.  Stamped
/// `PLAIN`: bytes to read, never an image to run or link.
/// The directory itself, in the short spelling its two siblings use:
/// `app` carries the `apps` view and `lib` the `libs` view, and this one
/// carries `shared`.  Keeping the two apart is not cosmetic —
/// `view_name_taken` counts a real child of the destination root as a
/// collision, so a view whose name IS its directory's cannot install into
/// any session rooted where that directory lives, silently, because every
/// call site takes the bool with `let _ =`.  On the two-store durable
/// layout that is every session the system store serves, which is why
/// `share` was absent from `list` and `shares` on metal while `apps` and
/// `libs` were there, and why no single-store QEMU gate ever saw it.
pub const SYSTEM_SHARE_DIR: &str = "share";
/// The READ-only display name every task view carries for it, beside
/// `apps` and `libs`, and the name a task types:
/// `shared/wasm/hello.wasm`.  The longer form, as those two are.
pub const SYSTEM_SHARE_VIEW: &str = "shared";
/// The administration session's writable grant over it (`sys-share`, for
/// the same collision reason as `sys-app`): what exec's data seeder
/// discovers and writes through.
pub const SYSTEM_SHARE_GRANT: &str = "sys-share";
/// The **package** directory: one subtree per installed `.pkg`
/// (`res/PKG.md`) — a typed manifest plus `app/` and `lib/` members that
/// `apps/` and `libs/` project by Link edge, never by copy.  No task view;
/// installers reach it through the administration grant only.
pub const SYSTEM_PKGS_DIR: &str = "pkgs";
/// The administration session's writable grant over it.
pub const SYSTEM_PKGS_GRANT: &str = "sys-pkg";

/// Whether two names differ — `const`, because the invariant below is a
/// compile-time one.
const fn name_differs(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return true;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return true;
        }
        i += 1;
    }
    false
}

// A view's display name must not be any system directory's own name.
// `view_name_taken` treats a real child of the destination root as taken,
// so such a view refuses to install for every session rooted there, and
// the three call sites discard that refusal (`let _ =`) — the failure is
// silent, survives every gate whose store has one region, and shows up as
// a missing name on a metal install.  `share`/`share` shipped exactly that.
const _: () = {
    assert!(name_differs(SYSTEM_APPS_VIEW, SYSTEM_APPS_DIR));
    assert!(name_differs(SYSTEM_APPS_VIEW, SYSTEM_LIBS_DIR));
    assert!(name_differs(SYSTEM_APPS_VIEW, SYSTEM_SHARE_DIR));
    assert!(name_differs(SYSTEM_APPS_VIEW, SYSTEM_PKGS_DIR));
    assert!(name_differs(SYSTEM_LIBS_VIEW, SYSTEM_APPS_DIR));
    assert!(name_differs(SYSTEM_LIBS_VIEW, SYSTEM_LIBS_DIR));
    assert!(name_differs(SYSTEM_LIBS_VIEW, SYSTEM_SHARE_DIR));
    assert!(name_differs(SYSTEM_LIBS_VIEW, SYSTEM_PKGS_DIR));
    assert!(name_differs(SYSTEM_SHARE_VIEW, SYSTEM_APPS_DIR));
    assert!(name_differs(SYSTEM_SHARE_VIEW, SYSTEM_LIBS_DIR));
    assert!(name_differs(SYSTEM_SHARE_VIEW, SYSTEM_SHARE_DIR));
    assert!(name_differs(SYSTEM_SHARE_VIEW, SYSTEM_PKGS_DIR));
};

pub use kore_dagfs::{FsRequest, FsResponse};

/// What a client told us about its delivered data window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAttachment {
    /// The service-arena slot the client's view landed in.
    pub slot: u32,
    /// The view's length, as the client declared it.
    pub bytes: u32,
}

fn write_identity(dst: &mut [u8], identity: InputIdentity) {
    debug_assert!(dst.len() >= 56);
    dst[0..8].copy_from_slice(&identity.source.to_le_bytes());
    dst[8..16].copy_from_slice(&identity.version.to_le_bytes());
    dst[16..24].copy_from_slice(&identity.len.to_le_bytes());
    dst[24..56].copy_from_slice(&identity.digest);
}

/// The filesystem service: an engine plus **one record per session slot**
/// (`SessionSlot`): the client's window, the peer generation that owns the
/// slot, its task view, its commit hold and its staged replacement.
///
/// Per slot, not one shared record, is the whole point: `ServeHost` fans in
/// over several pairs on one `FsService`, so a shared attachment record
/// would let client B's `AttachWindow` answer client A's presence check — A's
/// bulk verbs would then be validated against B's declared slot and length.
/// Keying it the same way the session itself is keyed (pair *n* → session *n*)
/// makes "this client has a window" and "this client's window" the same fact.
pub struct FsService<P: StorageProvider> {
    engine: Engine<P>,
    /// Every session slot's service-side state, grown to the highest slot a
    /// session has used ([`SessionSlots`]).
    slots: SessionSlots<P::Key>,
    /// The single client pair's session (the one-pair [`contract::bind`]
    /// posture — see [`contract`]). The served path derives it per pair from
    /// `Serve::via` instead.
    pub(crate) session: Option<SessionId>,
    /// Administration-pair assembler for one bounded recyclable face.
    peer_assembler: PeerBundleAssembler,
    /// Founder-declared exact-mapping band for recyclable pair rings.
    peer_map: Option<PeerMapPlan>,
    /// Completion of the last admission sequence, consumed by the fixed
    /// administration pair before another sequence may begin.
    last_admission: Option<Result<PeerId, PeerAdmissionRefusal>>,
    /// Endpoint-bound structural views. The scope comes only from a returned
    /// ChanSend capability; no request field selects a row.
    invocation_views: [Option<InvocationView>; MAX_INVOCATION_VIEWS],
    /// The pinned system invocation view — a session-zero self-grant of the
    /// persisted root, installed at bootstrap beside the administration
    /// grant.  The administration session attaches *through a delegation of
    /// this view* rather than receiving an ambient private root, so what it
    /// creates lands in the tree the durable entry serializes; dynamic peers
    /// receive their roots only through endpoint-authenticated `ApplyView`.
    system_view: Option<crate::GrantId>,
    /// The boot-local workspace anchor: the administration session's grant
    /// of the durable `work` directory ([`SYSTEM_WORKSPACE_DIR`]), held for
    /// the service's whole life as the delegation parent every private task
    /// view's structural root descends from.  Grants are boot-local; the
    /// directory beneath the persisted root is what survives.
    workspace_view: Option<crate::GrantId>,
    /// The administration session's writable grant over the shared executable
    /// directory (`app/`), captured when it is published. Each terminal's task view
    /// is minted ONE READ-only, BIND_EXECUTABLE-free `apps/` child of it at
    /// registration — the #605 Stage 2 store view. `None` on a store-less
    /// machine, where terminals simply see no projection.
    apps_view: Option<crate::GrantId>,
    /// The same for the shared library directory (`lib/`): the anchor of
    /// each task's READ-only `libs/` view.
    libs_view: Option<crate::GrantId>,
    /// The same for the shared data directory (`share/`): the anchor of each
    /// task's READ-only `share/` view.
    share_view: Option<crate::GrantId>,
    /// The administration session's writable grant over `pkgs/`; no task
    /// view descends from it.
    pkgs_store: Option<crate::GrantId>,
    /// Private roots stranded by peers that died, awaiting reclamation on
    /// the next async dispatch. The peer-death hook is synchronous, and
    /// reclaiming inside it meant a nested `block_on` in the serve loop —
    /// see `Engine::detach_session_authority`. On the heap, because one
    /// death strands its whole derivation closure, which descendants held by
    /// other sessions can make as large as the sessions themselves; the hook
    /// falls back to draining in place only if the heap cannot hold the
    /// queue.
    pending_reclaims: Retired,
    /// Staged replacements whose sessions died, awaiting release on the next
    /// async dispatch beside `pending_reclaims` — the same rule: nothing
    /// awaits inside the synchronous peer-death hook. Taken out of the slot
    /// at death so a reused slot never inherits a dead peer's stage; a
    /// release the provider refuses stays here and is retried, retaining
    /// custody. One stage per session bounds it.
    pending_aborts: Vec<crate::engine::Replacement<P::Key>, &'static ServiceAlloc>,
    /// The storage backend's last posture snapshot, folded into
    /// [`FsRequest::SystemStatus`] answers. The durable entry refreshes it
    /// immediately before delegating a status request; the default zeros
    /// honestly describe the bare volatile posture every other harness runs.
    backend: FsBackendStatus,
}

/// What the mounted backend reports about itself for `SystemStatus` — set
/// by the layer that owns the mount (the durable entry), consumed by the
/// serving frontend, never derived from wire input.
#[derive(Clone, Copy, Debug, Default)]
pub struct FsBackendStatus {
    /// A durable volume is mounted (false: the RAM posture).
    pub durable: bool,
    /// The durable mount recovered degraded and is read-only.
    pub degraded_read_only: bool,
    /// The realized interrupt route is held (false: polled).
    pub route_held: bool,
    /// The mounted block channel crossed its one-way lost boundary.
    pub channel_lost: bool,
    /// The controller holds a host memory buffer for its mapping tables.
    pub host_memory_buffer: bool,
    /// The readiness note ([`crate::durable::note`] vocabulary; `0` is the
    /// fully healthy durable posture).
    pub note: u64,
    /// Conservative user-region lifecycle (`FsSystemStatusWire::LIFECYCLE_*`).
    pub user_lifecycle: u16,
    /// Rebuildable system-region lifecycle (`FsSystemStatusWire::LIFECYCLE_*`).
    pub system_lifecycle: u16,
    /// Whole-tree serializations completed since bring-up.
    pub commits: u64,
    /// The block provider's operation counters.
    pub provider: crate::provider::BlockProviderStats,
    /// The block channel's submission/completion counters.
    pub channel: kore_driver::block::ChannelStats,
    /// The whole namespace, authorized mount region, and allocator pressure.
    /// Empty in the volatile posture.
    pub storage: kore_abi::monitor::FilesystemStorageSnapshot,
    /// Backend waits that genuinely parked on the realized interrupt route.
    pub route_parks: u64,
    /// Route parks the interrupt actually woke. A gap against `route_parks`
    /// is delivery the route did not make.
    pub route_fires: u64,
    /// Waits bounded by a clock deadline with no route to park on — the
    /// route-less posture on a machine whose clock is readable.
    pub deadline_waits: u64,
    /// Waits bounded only by a count of spin hints, because no clock could
    /// supply a deadline at all.
    pub spin_waits: u64,
    /// Route-less turns that slept this core on the completion queue's own
    /// cache line — the monitored fast tier's numerator. Always zero under an
    /// emulator, which implements no user-mode monitor.
    pub monitor_sleeps: u64,
    /// Route-less turns that reached the monitored tier at all — its
    /// denominator, counted whether or not the machine could sleep.
    pub monitor_turns: u64,
}

const MAX_INVOCATION_VIEWS: usize = 16;

// The most edges this service's own views make one session hold: a task
// view's root, its three store views (`apps`, `libs`, `shared`), and a
// directory pinned beneath it for every invocation view at once. Grants other
// views send it share the rest of its edges.
const _: () = assert!(
    1 + 3 + MAX_INVOCATION_VIEWS <= crate::bounds::MAX_EDGES_PER_SESSION,
    "a task view must hold its root, its store views and every pinned directory"
);

#[derive(Debug)]
struct InvocationView {
    scope: u64,
    grant: GrantId,
    rights: Rights,
    task_session: SessionId,
    owner: InvocationViewOwner,
    canary: Option<ChanSendToken>,
    /// The view's durable anchoring beneath the administration workspace,
    /// when ambient policy established one: the home directory's name (for
    /// same-boot duplicate refusal) and the administration session's anchor
    /// grant of it (the delegation parent of this view's root, revoked when
    /// the view is reaped — the directory itself is a fact of the persisted
    /// tree and survives).
    home: Option<(Name, GrantId)>,
}

#[derive(Debug)]
enum InvocationViewOwner {
    /// A narrowed cwd pinned below the stable task view (or the one explicit
    /// system bootstrap view).
    Pinned { issuer: SessionId },
    /// The stable private root that defines one task's filesystem namespace.
    Private { session: SessionId },
}

/// What the service keeps for one session slot, beside the engine's
/// authority for it.
struct SessionSlot<K> {
    /// What the client declared about its window.
    attached: Option<WindowAttachment>,
    /// The metal mapping, paired with the affine returned capability that
    /// keeps its backing live. Host tests pass an ordinary `DataWindow`
    /// directly and therefore leave it empty.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    mapped: Option<MappedWindow>,
    /// The exact host-authored `Serve::via` of the recyclable peer that owns
    /// the slot. A fixed pair's slot keeps `None`; generations stop a late
    /// request or close from an old occupant naming its replacement.
    via: Option<u16>,
    /// The stable task view the slot's command pair was applied to. Set only
    /// by `ApplyView` after endpoint authentication; cleared with the exact
    /// peer generation.
    task_view: Option<SessionId>,
    /// Holding a [`FsRequest::DeferCommits`] window open. Per session by
    /// construction: a bulk writer defers only its own commits, never
    /// another client's durability.
    deferring: bool,
    /// The session's one private provider placement. The service already
    /// serializes dispatch; no additional lock or namespace object.
    replacement: Option<crate::engine::Replacement<K>>,
}

impl<K> SessionSlot<K> {
    const fn vacant() -> Self {
        Self {
            attached: None,
            mapped: None,
            via: None,
            task_view: None,
            deferring: false,
            replacement: None,
        }
    }
}

/// The heap one session slot's [`SessionSlot`] costs in the largest
/// composition's frontend: the durable engines' block keys, or the RAM
/// engines' where those are larger.
pub(crate) const SESSION_SLOT_BYTES: usize = {
    let block = core::mem::size_of::<SessionSlot<crate::provider::BlockKey>>();
    let ram = core::mem::size_of::<SessionSlot<crate::provider::RamKey>>();
    if block > ram { block } else { ram }
};

/// [`SessionSlot`] for every session slot used so far, indexed by slot.
///
/// Grown on the heap to the highest slot a session has used, never sized by
/// [`MAX_SESSIONS`]: the serve host reuses its lowest free slot, so the
/// table follows the most sessions attached at once, as the engine's
/// per-slot authority arenas do. A slot past the table holds a vacant
/// record. Growth is the one fallible step, and it happens where a session
/// first needs a record — a peer's admission above all, which a heap that
/// cannot grow refuses whole.
struct SessionSlots<K>(Vec<SessionSlot<K>, &'static ServiceAlloc>);

impl<K> SessionSlots<K> {
    fn new() -> Self {
        Self(Vec::new_in(crate::service_alloc()))
    }

    fn get(&self, slot: usize) -> Option<&SessionSlot<K>> {
        self.0.get(slot)
    }

    fn get_mut(&mut self, slot: usize) -> Option<&mut SessionSlot<K>> {
        self.0.get_mut(slot)
    }

    /// `slot`'s record, growing the table to reach it: [`FsError::Range`]
    /// past every session, [`FsError::OutOfHandles`] when the heap cannot
    /// hold the table.
    fn claim(&mut self, slot: usize) -> FsResult<&mut SessionSlot<K>> {
        if slot >= MAX_SESSIONS {
            return Err(FsError::Range);
        }
        if slot >= self.0.len() {
            self.0
                .try_reserve(slot + 1 - self.0.len())
                .map_err(|_| FsError::OutOfHandles)?;
            self.0.resize_with(slot + 1, SessionSlot::vacant);
        }
        Ok(&mut self.0[slot])
    }

    /// Records whose state passes `filter`, for the diagnostic snapshot.
    fn count(&self, filter: impl Fn(&SessionSlot<K>) -> bool) -> u16 {
        self.0.iter().filter(|state| filter(state)).count() as u16
    }
}

/// One retained, mapped client data window.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
struct MappedWindow {
    /// Owns the delivered `LeasedRegion` record and its runtime slot
    /// reservation for exactly as long as the service may touch `base`.
    _owner: ReturnedCapability,
    base: core::ptr::NonNull<u8>,
    len: usize,
}

/// Borrow-free description of one retained data-window mapping.
///
/// The affine [`ReturnedCapability`] that keeps this span live remains in the
/// owning [`FsService`]. The dual-volume router may use the span sequentially
/// for either engine, but never stores it or lets both engines borrow it at
/// once.
#[derive(Clone, Copy)]
pub(crate) struct MappedWindowSpan {
    pub(crate) base: core::ptr::NonNull<u8>,
    pub(crate) len: usize,
}

impl<P: StorageProvider> FsService<P> {
    /// The per-request transfer limit this service advertises at window
    /// attach: the provider's own, narrowed to the wire.
    pub(crate) fn transfer_limit(&self) -> u32 {
        u32::try_from(self.engine.provider().limits().max_transfer).unwrap_or(u32::MAX)
    }

    pub fn new(engine: Engine<P>) -> Self {
        Self {
            engine,
            slots: SessionSlots::new(),
            session: None,
            peer_assembler: PeerBundleAssembler::new(0, true),
            peer_map: None,
            last_admission: None,
            invocation_views: core::array::from_fn(|_| None),
            system_view: None,
            workspace_view: None,
            apps_view: None,
            libs_view: None,
            share_view: None,
            pkgs_store: None,
            pending_reclaims: Vec::new_in(crate::service_alloc()),
            pending_aborts: Vec::new_in(crate::service_alloc()),
            backend: FsBackendStatus::default(),
        }
    }

    /// Replace the backend posture snapshot folded into `SystemStatus`
    /// answers. The mount owner calls this immediately before delegating a
    /// status request, so the record reads as one moment.
    pub fn set_backend_status(&mut self, backend: FsBackendStatus) {
        self.backend = backend;
    }

    fn task_view_for_session(&self, session: SessionId) -> Option<SessionId> {
        self.slots
            .get(session.index() as usize)
            .and_then(|state| state.task_view)
    }

    fn task_view_for_scope(&self, scope: u64) -> Option<SessionId> {
        self.invocation_views.iter().flatten().find_map(|view| {
            (view.scope == scope && matches!(view.owner, InvocationViewOwner::Private { .. }))
                .then_some(view.task_session)
        })
    }

    fn register_invocation_view_inner(
        &mut self,
        session: SessionId,
        handle: Handle,
        rights: Rights,
        scope: u64,
        canary: Option<ChanSendToken>,
    ) -> FsResult<()> {
        self.reap_dead_invocation_views();
        if self
            .invocation_views
            .iter()
            .flatten()
            .any(|view| view.scope == scope)
        {
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
            return Err(FsError::AlreadyExists);
        }
        let Some(slot) = self.invocation_views.iter().position(Option::is_none) else {
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
            return Err(FsError::OutOfEdges);
        };
        let (grant, issuer) = match self.engine.pin_view(session, handle, rights) {
            Ok(pinned) => pinned,
            Err(error) => {
                if let Some(canary) = canary {
                    let _ = canary.try_retire();
                }
                return Err(error);
            }
        };
        self.invocation_views[slot] = Some(InvocationView {
            scope,
            grant,
            rights,
            task_session: issuer,
            owner: InvocationViewOwner::Pinned { issuer },
            canary,
            home: None,
        });
        Ok(())
    }

    fn register_invocation_view(
        &mut self,
        session: SessionId,
        handle: Handle,
        rights: Rights,
        scope: u64,
    ) -> FsResult<()> {
        self.register_invocation_view_inner(session, handle, rights, scope, None)
    }

    fn register_invocation_endpoint(
        &mut self,
        session: SessionId,
        handle: Handle,
        rights: Rights,
        returned: ReturnedCapability,
    ) -> FsResult<()> {
        let canary = returned
            .try_into_chan_send(crate::FsViewCapability::INVOCATION_RIGHTS)
            .map_err(|_| FsError::AccessDenied)?;
        let scope = verbs::capability_scope_rid(
            canary.slot(),
            kore_abi::cap::ResourceKind::ChanSend,
            kore_abi::cap::Rights::ACCESS,
        );
        let Some(scope) = scope else {
            let _ = canary.try_retire();
            return Err(FsError::AccessDenied);
        };
        self.register_invocation_view_inner(session, handle, rights, scope, Some(canary))
    }

    fn reap_dead_invocation_views(&mut self) {
        for slot in 0..self.invocation_views.len() {
            let dead = self.invocation_views[slot]
                .as_ref()
                .and_then(|view| view.canary.as_ref())
                .is_some_and(|canary| {
                    verbs::capability_scope_rid(
                        canary.slot(),
                        kore_abi::cap::ResourceKind::ChanSend,
                        kore_abi::cap::Rights::ACCESS,
                    )
                    .is_none()
                });
            if dead && let Some(view) = self.invocation_views[slot].take() {
                match view.owner {
                    InvocationViewOwner::Pinned { issuer } => {
                        let _ = block_on(self.engine.revoke(issuer, view.grant));
                    }
                    InvocationViewOwner::Private { session } => {
                        let _ = block_on(self.engine.detach_session(session));
                    }
                }
                // The anchor grant outlives the view's own root (it is that
                // root's delegation parent), so it is released only after
                // the session teardown above retired the descendant edges.
                // The home directory stays: it is a fact of the persisted
                // tree, and the next registration of this name reopens it.
                if let Some((_, anchor)) = view.home {
                    let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
                    let _ = block_on(self.engine.revoke(admin, anchor));
                }
            }
        }
    }

    /// The standing durable-workspace anchor, when bootstrap has published
    /// it — the delegation parent for every private task view's root.
    fn workspace_grant(&self) -> Option<GrantId> {
        self.workspace_view
    }

    /// Open — or, on a blank tree, create — the view's home directory
    /// `work/<name>` and hand back the administration session's anchor grant
    /// of it: the delegation parent the private view's root will descend
    /// from. Reuse-before-create is the persistence half of the contract
    /// (a recovered volume already holds the directory and the task's
    /// files under it); the distinct directory per name is the isolation
    /// half (two tasks never share a home unless someone grants across).
    fn anchor_view_home(&mut self, workspace: GrantId, name: &Name) -> FsResult<GrantId> {
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let work = self
            .engine
            .open_grant(admin, workspace, Rights::ALL)?
            .handle;
        let home = match self.engine.open_at(admin, work, name, Rights::ALL) {
            Ok(opened) if opened.kind == ObjectKind::Directory => Ok(opened),
            Ok(opened) => {
                // A file squatting on the home name: refuse rather than
                // root a task's namespace inside something that cannot
                // contain names.
                let _ = self.engine.close(admin, opened.handle);
                Err(FsError::NotDirectory)
            }
            Err(FsError::NotFound) => {
                block_on(self.engine.create(admin, work, name, ObjectKind::Directory))
            }
            Err(error) => Err(error),
        };
        let anchored = home.and_then(|home| {
            let anchored = self
                .engine
                .grant(admin, home.handle, admin, Rights::ALL, None);
            let _ = self.engine.close(admin, home.handle);
            anchored
        });
        let _ = self.engine.close(admin, work);
        anchored
    }

    fn register_private_invocation_view(
        &mut self,
        scope: u64,
        name: &Name,
        rights: Rights,
        canary: Option<ChanSendToken>,
    ) -> FsResult<()> {
        self.reap_dead_invocation_views();
        if rights.is_empty()
            || self.invocation_views.iter().flatten().any(|view| {
                view.scope == scope || view.home.as_ref().is_some_and(|(home, _)| home == name)
            })
        {
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
            return Err(FsError::AlreadyExists);
        }
        let Some(slot) = self.invocation_views.iter().position(Option::is_none) else {
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
            return Err(FsError::OutOfEdges);
        };
        let session = (FIRST_TASK_VIEW_SESSION..MAX_SESSIONS)
            .map(|index| SessionId::new(index as u16))
            .find(|candidate| {
                !self.invocation_views.iter().flatten().any(|view| {
                    matches!(
                        view.owner,
                        InvocationViewOwner::Private { session } if session == *candidate
                    )
                })
            });
        let Some(session) = session else {
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
            return Err(FsError::OutOfEdges);
        };
        // The #575 invariant in the invocation-view model's own terms: a
        // task view's namespace must be a fact of the persisted tree, so
        // the fresh session roots in a delegation of its own durable home
        // directory (`work/<name>` under session zero) rather than an
        // ambient private root — `commit_engine` serializes exactly what
        // session zero reaches, and an ambient root's objects reached
        // their data blocks and lost their names at every remount. One
        // home per name keeps distinct tasks' namespaces distinct (the
        // two-terminal isolation contract); reopening the same name across
        // boots is precisely what makes a task's files survive. Host
        // fixtures that never establish ambient policy keep the isolated
        // ambient posture, so isolation tests stay honest either way.
        let home = match self.workspace_grant() {
            Some(workspace) => match self.anchor_view_home(workspace, name) {
                Ok(anchor) => Some((*name, anchor)),
                Err(error) => {
                    if let Some(canary) = canary {
                        let _ = canary.try_retire();
                    }
                    return Err(error);
                }
            },
            None => None,
        };
        let attached = match home {
            Some((_, anchor)) => self
                .engine
                .attach_delegated_session(session, anchor, rights),
            None => block_on(self.engine.attach_session(session)),
        };
        let unwound = |service: &mut Self, canary: Option<ChanSendToken>| {
            if let Some((_, anchor)) = home {
                let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
                let _ = block_on(service.engine.revoke(admin, anchor));
            }
            if let Some(canary) = canary {
                let _ = canary.try_retire();
            }
        };
        if let Err(error) = attached {
            unwound(self, canary);
            return Err(error);
        }
        let grant = match self.engine.task_view_root(session) {
            Ok(grant) => grant,
            Err(error) => {
                let _ = block_on(self.engine.detach_session(session));
                unwound(self, canary);
                return Err(error);
            }
        };
        // #605 Stage 2: mint this terminal the canonical READ-only,
        // BIND_EXECUTABLE-free `apps` view onto the shared store object (no new
        // object, no byte copy).  ONE name only: the flat `app` spelling rode
        // beside it while callers migrated, and showing one object under two
        // names reads on a live machine as a stale projection that survived a
        // boot.  BEST-EFFORT and strictly NON-FATAL: a collision in a private
        // home or edge exhaustion must never fail terminal founding — a terminal
        // without the view still boots, it just cannot see the store.
        // READ_ONLY strips WRITE, MUTATE_NAMESPACE, SHARE, and
        // BIND_EXECUTABLE, so a terminal can read and run store executables but
        // never write, relink, or bless them.
        let _ = self.install_apps_view(session);
        let _ = self.install_libs_view(session);
        let _ = self.install_share_view(session);
        self.invocation_views[slot] = Some(InvocationView {
            scope,
            grant,
            rights,
            task_session: session,
            owner: InvocationViewOwner::Private { session },
            canary,
            home,
        });
        Ok(())
    }

    pub(crate) fn register_system_invocation_view(
        &mut self,
        scope: u64,
        rights: Rights,
    ) -> FsResult<()> {
        if rights.is_empty() {
            return Err(FsError::AccessDenied);
        }
        let system = SessionId::new(0);
        // Pinning is itself a delegation, so the short-lived source handle
        // must carry SHARE. The pinned view receives exactly `rights`; source
        // authority is never copied into the descendant.
        let source_rights = rights.union(Rights::SHARE);
        let root = self.engine.open_root(system, source_rights)?;
        let result = self.register_invocation_view(system, root.handle, rights, scope);
        let _ = self.engine.close(system, root.handle);
        // A peer applied onto this view resolves names through ITS task
        // session — the system session — so the two system views every
        // task carries must stand there too, exactly as a private view
        // installs them on its own session.  Init's post-ready witnesses
        // are the consumers: `require "lua_demo"` walks `libs/` from a
        // session anchored here.  Best-effort like the private path.
        if result.is_ok() {
            let _ = self.install_apps_view(system);
            let _ = self.install_libs_view(system);
            let _ = self.install_share_view(system);
        }
        result
    }

    pub(crate) fn apply_invocation_view(
        &mut self,
        scope: u64,
        peer: PeerId,
        profile_rights: Rights,
    ) -> FsResult<()> {
        self.reap_dead_invocation_views();
        if self.peer_is_retired(peer) {
            return Err(FsError::AccessDenied);
        }
        let (grant, view_rights, task_session) = self
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == scope)
            .map(|view| (view.grant, view.rights, view.task_session))
            .ok_or(FsError::AccessDenied)?;
        let rights = view_rights.intersection(profile_rights);
        if rights.is_empty() {
            return Err(FsError::AccessDenied);
        }
        // A live peer's slot has its record: admission claimed it.
        let state = self
            .slots
            .get_mut(peer.slot())
            .ok_or(FsError::AccessDenied)?;
        self.engine
            .install_view_root(SessionId::new(peer.slot() as u16), grant, rights)?;
        state.task_view = Some(task_session);
        Ok(())
    }

    /// Enable the generic recyclable-pair host over one founder-stamped
    /// exact-mapping band.  The map is service-local VA geometry, never wire
    /// data, and therefore cannot be selected by a client.
    pub(crate) fn configure_peer_map(&mut self, map: PeerMapPlan) {
        self.peer_map = Some(map);
    }

    /// Resolve a host-authored arrival identity to the bounded filesystem
    /// session it owns. Static founder pairs map one-to-one to the equally
    /// numbered sessions; recyclable peers must match the exact current
    /// generation. Pair one is therefore exec's administration session, not
    /// an alias for pair zero's persisted handles.
    pub(crate) fn session_for_via(&self, via: u16) -> Option<SessionId> {
        if via <= crate::service::contract::ADMIN_VIA {
            return Some(SessionId::new(via));
        }
        // The arrival names its slot; only the slot's current occupant
        // matches it exactly.
        let peer = PeerId::from_arrival(via)?;
        (!self.peer_is_retired(peer)).then(|| SessionId::new(peer.slot() as u16))
    }

    /// Complete service-local policy after the generic host installed a
    /// peer. Returning `false` asks the host to tear the just-installed pair
    /// back down before it can issue a request.
    pub(crate) fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        let Ok(peer) = result else {
            self.last_admission = Some(result);
            return true;
        };
        let installed = self.install_peer(peer);
        self.last_admission = Some(installed.map(|()| peer));
        installed.is_ok()
    }

    /// Mirror a host-admitted pair into a secondary engine without claiming
    /// ownership of the admission protocol or its one completion slot. The
    /// primary frontend consumed the returned endpoint bundle; this engine
    /// receives only the host-authored peer identity and creates an empty
    /// pair-bound session into which explicit system grants may later land.
    pub(crate) fn mirror_peer_admitted(
        &mut self,
        peer: PeerId,
    ) -> Result<(), PeerAdmissionRefusal> {
        self.install_peer(peer)
    }

    fn install_peer(&mut self, peer: PeerId) -> Result<(), PeerAdmissionRefusal> {
        let slot = peer.slot();
        if slot == 0
            || self
                .slots
                .get(slot)
                .is_some_and(|state| state.via.is_some())
        {
            // An occupied slot is a registration a previous peer never
            // retired, not a full table.  Reporting it as capacity sent the
            // caller to retry a table with room, and surfaced on the console
            // as the rung that means exhaustion.
            return Err(PeerAdmissionRefusal::PolicyCollision);
        }
        // The slot's record and the session's storage are the admission's
        // whole cost; a heap that cannot hold either refuses it as the full
        // table it is, before anything is published.
        let state = self
            .slots
            .claim(slot)
            .map_err(|_| PeerAdmissionRefusal::Capacity)?;
        match self
            .engine
            .attach_empty_session(SessionId::new(slot as u16))
        {
            Ok(()) => {}
            Err(FsError::OutOfHandles) => return Err(PeerAdmissionRefusal::Capacity),
            Err(_) => return Err(PeerAdmissionRefusal::PolicyUnavailable),
        }
        state.via = Some(peer.arrival());
        Ok(())
    }

    /// Give a fixed session its root — the invocation-view posture: a
    /// delegation of the pinned system view, so what the session creates
    /// lands in the persisted tree.  Before bootstrap pins the view (host
    /// fixtures that never establish ambient policy) every session keeps the
    /// ambient private-root posture so isolation tests stay honest.
    ///
    /// Dynamic peers never root here: admission parks them in the attach
    /// transition and only endpoint-authenticated `ApplyView` installs their
    /// structural root.  A request arriving on a parked pair maps that
    /// transition to `Busy` below and is refused, so the pair-timing race
    /// can never mint an ambient root for an un-viewed client.
    pub(crate) fn ensure_dynamic_session(&mut self, session: SessionId) -> bool {
        let fixed = session == SessionId::new(0);
        match self.system_view {
            Some(view) if !fixed => self
                .engine
                .attach_delegated_session(session, view, Rights::ALL)
                .or_else(|error| match error {
                    FsError::AlreadyExists => Ok(()),
                    other => Err(other),
                })
                .is_ok(),
            _ => self.ensure_session(session),
        }
    }

    /// Drop exactly the state belonging to the retired generation. A late
    /// close for an earlier occupant is inert.
    pub(crate) fn peer_closed(&mut self, via: u16, _reason: PeerClose) -> bool {
        let Some(peer) = PeerId::from_arrival(via).filter(|peer| !self.peer_is_retired(*peer))
        else {
            return true;
        };
        let slot = peer.slot();
        // A staged replacement dies with its session. Its placement is
        // released on the next dispatch, never here: the same rule as the
        // stranded roots below. Taking it out of the slot now keeps a
        // reused slot from inheriting a dead peer's stage.
        if let Some(stage) = self
            .slots
            .get_mut(slot)
            .and_then(|state| state.replacement.take())
        {
            if self.pending_aborts.try_reserve(1).is_ok() {
                self.pending_aborts.push(stage);
            } else if block_on(self.engine.abort_replace(&stage)).is_err() {
                // A heap that cannot hold the list releases in place, as
                // the reclaim fallback below does, and keeps the stage if
                // the provider refuses.
                if let Some(state) = self.slots.get_mut(slot) {
                    state.replacement = Some(stage);
                }
                return false;
            }
        }
        let Ok(stranded) = self
            .engine
            .detach_session_authority(SessionId::new(slot as u16))
        else {
            return false;
        };
        self.queue_reclaims(stranded);
        // Everything else the slot held goes with its occupant: the window
        // and its mapping, the task view, and the commit hold — a client
        // that died inside a bulk-write window must not leave the slot
        // deferring, or the next occupant would inherit held commits it never
        // asked for. That drops only the durability hold; it cannot roll back
        // mutations already visible in the live engine.
        if let Some(state) = self.slots.get_mut(slot) {
            *state = SessionSlot::vacant();
        }
        true
    }

    /// Queue the roots a dead peer stranded for the next async dispatch,
    /// where reclamation's contention budget means something. A heap that
    /// cannot hold the queue drains it here instead, so no root is ever
    /// dropped on the floor, at the old nested cost.
    fn queue_reclaims(&mut self, stranded: Retired) {
        if self.pending_reclaims.is_empty() {
            self.pending_reclaims = stranded;
            return;
        }
        if self.pending_reclaims.try_reserve(stranded.len()).is_ok() {
            self.pending_reclaims.extend_from_slice(&stranded);
            return;
        }
        block_on(self.engine.reclaim_stranded(&self.pending_reclaims));
        self.pending_reclaims.clear();
        block_on(self.engine.reclaim_stranded(&stranded));
    }

    /// Whether one exact host-authored peer generation is no longer live.
    /// A replacement in the same bounded cell is intentionally different:
    /// confirming the old ticket must never wait on or disturb the new peer.
    pub(crate) fn peer_is_retired(&self, peer: PeerId) -> bool {
        self.slots.get(peer.slot()).and_then(|state| state.via) != Some(peer.arrival())
    }

    /// Administration control plane for recyclable clients. Ordinary
    /// filesystem requests never enter this assembler.
    pub(crate) fn admit_peer_component(
        &mut self,
        part: PeerBundlePart,
        announced: kore_abi::session::SlotRef,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        if sv.via() != crate::service::contract::ADMIN_VIA {
            return FsResponse::Refused(FsError::AccessDenied);
        }
        // A `Ring` is the boundary between bundles, so one arriving while the
        // assembler still owns components proves the previous assembly was
        // abandoned — its asker spent its bounded re-ask and gave up.  Retire
        // what it left, and refuse the replacement while any of those owners
        // is still closing (`PeerBundleAssembler::retire_incomplete`).
        //
        // Without this the leftovers are permanent: the next launch's matching
        // part answers `Duplicate`, the bundle never completes, no verdict is
        // ever published, and every later launch that needs this face refuses
        // for the rest of the boot.
        //
        // Admin only, and checked before anything else touches the assembler:
        // a foreign via must never retire the components the administration
        // pair is still assembling.
        if matches!(part, PeerBundlePart::Ring)
            && !self.peer_assembler.is_empty()
            && !self.peer_assembler.retire_incomplete()
        {
            return FsResponse::Refused(FsError::Conflict);
        }
        if self.last_admission.is_some() {
            return FsResponse::Refused(FsError::Conflict);
        }
        let Some(map) = self.peer_map else {
            return FsResponse::Refused(FsError::Unsupported);
        };
        let Some(mut input) = sv.returned_input(announced) else {
            return FsResponse::Refused(FsError::NoWindow);
        };
        let Some(capability) = input.capability() else {
            return FsResponse::Refused(FsError::NoWindow);
        };
        if let Err(reason) = self.peer_assembler.accepts(part, capability) {
            return FsResponse::Refused(peer_assembly_error(reason));
        }
        let capability = input
            .take()
            .expect("validated returned peer component remains present");
        let bundle = self
            .peer_assembler
            .push(part, capability)
            .expect("same synchronous peer validation cannot change");
        if let Some(bundle) = bundle {
            sv.admit_peer(bundle, map);
        }
        FsResponse::Done
    }

    /// Consume the completion published by the outer serving host after it
    /// mapped (or refused) the preceding peer bundle.
    pub(crate) fn take_peer_admission(&mut self) -> FsResponse {
        match self.last_admission.take() {
            Some(Ok(peer)) => FsResponse::PeerAdmitted { peer },
            Some(Err(reason)) => FsResponse::Refused(peer_admission_error(reason)),
            // No verdict, and the assembler still owns components: this
            // bundle never completed, so `admit_peer` was never called and
            // no verdict can EVER be published.  `NotFound` here would be a
            // lie the asker waits on forever — and exec fences new
            // provisioning while an id-less face exists, so that wait
            // refuses every later launch that needs this face.  Retire what
            // the abandoned assembly still owns and answer terminally, which
            // is exec's signal to let the peer go.
            None if !self.peer_assembler.is_empty() => {
                let _ = self.peer_assembler.retire_incomplete();
                FsResponse::Refused(FsError::Conflict)
            }
            // No verdict and an EMPTY assembler: the bundle completed and
            // the host has not drained its admission action yet.  One IS
            // coming, so this is an ordinary "not yet" — settling here would
            // orphan the face the host is about to admit.
            None => FsResponse::Refused(FsError::NotFound),
        }
    }

    /// Whether `session` is holding its commits for a bulk write.
    pub fn is_deferring(&self, session: SessionId) -> bool {
        self.slots
            .get(session.index() as usize)
            .is_some_and(|state| state.deferring)
    }

    /// The engine, for startup wiring (attaching sessions) and diagnostics.
    pub fn engine(&mut self) -> &mut Engine<P> {
        &mut self.engine
    }

    /// The engine, read-only.
    pub fn engine_ref(&self) -> &Engine<P> {
        &self.engine
    }

    /// The device blocks of `session`'s lent version `lease`, for a read
    /// grant (`Engine::lent_blocks`).
    pub(crate) fn lent_blocks(
        &self,
        session: SessionId,
        lease: crate::extraction::VersionLease,
    ) -> FsResult<Option<crate::provider::DeviceBlocks>> {
        let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
        self.engine
            .lent_blocks(owner, lease)
            .map_err(extraction_error)
    }

    /// Release the engine — restart simulations and provider transitions.
    pub(crate) fn into_engine(self) -> Engine<P> {
        self.engine
    }

    /// Give `session` a private root if it has none.
    ///
    /// The postcondition both callers need is "this session has a root", not
    /// "a root was created here" — the one-pair [`contract::bind`] path
    /// establishes session 0 before `ServeHost` ever attaches pair 0, so an
    /// already-attached session is success, not a double-attach bug. Every
    /// other refusal (an out-of-range session, an exhausted edge arena) is a
    /// refusal, because it leaves the client rootless.
    ///
    /// [`contract::bind`]: crate::service::contract
    pub(crate) fn ensure_session(&mut self, session: SessionId) -> bool {
        match block_on(self.engine.attach_session(session)) {
            Ok(()) | Err(FsError::AlreadyExists) => true,
            Err(_) => false,
        }
    }

    /// Install the sole boot-policy bridge from the persisted system
    /// namespace (session zero) to exec's fixed administration session.
    ///
    /// The named source grant carries `READ|SHARE|BIND_EXECUTABLE`: exec may
    /// inspect immutable source bytes, nominate an exact revision to
    /// ImageService, and delegate a read-only `app` view — the
    /// executable-source right is deliberately attenuated out of that child
    /// view, and exec must discover its pair-bound [`GrantId`] through
    /// `ListGrants` before it can open anything.  The administration
    /// session's *structural root*, though, is a delegation of the persisted
    /// system root (the invocation-view posture): the launcher-owned layout
    /// and every view it grants a child are facts of the durable tree, so
    /// task files survive a reboot rather than dying with a private root.
    /// This keeps the pair-is-session rule intact without giving pair one a
    /// second ambient root that aliases session zero's handles.
    pub(crate) fn establish_admin_namespace(&mut self) -> bool {
        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        if !self.ensure_session(system) {
            return false;
        }
        if !self.establish_system_view() {
            return false;
        }
        // The administration session enters through the system view like
        // every dynamic session: exec's launcher-owned layout objects
        // (`data`/`config`/`tmp`/`shared`) and the views it grants children
        // are thereby facts of the persisted tree, which is what lets a
        // task's files survive a reboot.  The named source grant below stays
        // the executable-acquisition bridge.
        if !self.ensure_dynamic_session(admin) {
            return false;
        }
        true
    }

    pub(crate) fn establish_admin_source(&mut self) -> bool {
        if !self.establish_admin_namespace() {
            return false;
        }
        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let Ok(display) = Name::new(SYSTEM_SOURCE_GRANT) else {
            return false;
        };
        let Ok(grants) = self.engine.grants(admin) else {
            return false;
        };
        if grants.iter().any(|grant| {
            grant.display_name == Some(display)
                && grant.kind == crate::authority::AuthorityKind::Grant
                && grant.rights
                    == Rights::READ
                        .union(Rights::SHARE)
                        .union(Rights::BIND_EXECUTABLE)
        }) {
            return true;
        }
        let Ok(root) = self.engine.open_root(system, Rights::ALL) else {
            return false;
        };
        let granted = self
            .engine
            .grant(
                system,
                root.handle,
                admin,
                Rights::READ
                    .union(Rights::SHARE)
                    .union(Rights::BIND_EXECUTABLE),
                Some(display),
            )
            .is_ok();
        let _ = self.engine.close(system, root.handle);
        granted
    }

    /// Pin the system invocation view: a session-zero self-grant of the
    /// persisted root, held for the service's whole life as the delegation
    /// parent every dynamic peer's structural root descends from (the
    /// capability-selected execution branch's `register_system_invocation_
    /// view`, folded into this bootstrap hook because ambient policy is
    /// installed exactly once, here).  Idempotent like the administration
    /// grant above it.
    fn establish_system_view(&mut self) -> bool {
        if self.system_view.is_some() {
            return true;
        }
        let system = SessionId::new(0);
        let Ok(root) = self.engine.open_root(system, Rights::ALL) else {
            return false;
        };
        let pinned = self
            .engine
            .grant(system, root.handle, system, Rights::ALL, None);
        let _ = self.engine.close(system, root.handle);
        match pinned {
            Ok(view) => {
                self.system_view = Some(view);
                true
            }
            Err(_) => false,
        }
    }

    /// Publish the shared executable directory (`app` beneath the persisted
    /// root) and hand the administration session a writable grant over it.
    ///
    /// This is the store half of the archive/filesystem unification: the boot
    /// archive is a seed, and the durable tree is where executables actually
    /// live. Exec discovers the grant by its display name and copies each
    /// registered member in once; a later boot finds the directory already
    /// populated and writes nothing.
    ///
    /// Idempotent in both directions, for the reason the workspace is: on a
    /// recovered volume the directory and its contents are already there and
    /// are reused — never re-created, which would strand the executables
    /// beneath an orphaned object — and a second call finds the standing
    /// grant.
    pub(crate) fn establish_admin_apps(&mut self) -> bool {
        match self.establish_admin_directory(SYSTEM_APPS_DIR, SYSTEM_APPS_GRANT) {
            Some(grant) => {
                // The terminal app-view anchor: each terminal's READ-only
                // `apps` descends from this admin-held grant.
                self.apps_view = Some(grant);
                true
            }
            None => false,
        }
    }

    /// The shared library directory, `lib/` — the same bring-up as `app/`
    /// (create-or-adopt, one writable admin grant), anchoring every task's
    /// READ-only `libs` view.
    pub(crate) fn establish_admin_libs(&mut self) -> bool {
        match self.establish_admin_directory(SYSTEM_LIBS_DIR, SYSTEM_LIBS_GRANT) {
            Some(grant) => {
                self.libs_view = Some(grant);
                true
            }
            None => false,
        }
    }

    /// The shared data directory, `share/` — the same bring-up as `lib/`
    /// (create-or-adopt, one writable admin grant), anchoring every task's
    /// READ-only `share` view.
    pub(crate) fn establish_admin_share(&mut self) -> bool {
        match self.establish_admin_directory(SYSTEM_SHARE_DIR, SYSTEM_SHARE_GRANT) {
            Some(grant) => {
                self.share_view = Some(grant);
                true
            }
            None => false,
        }
    }

    /// The package directory, `pkgs/` — admin-writable, never viewed.
    pub(crate) fn establish_admin_pkgs(&mut self) -> bool {
        match self.establish_admin_directory(SYSTEM_PKGS_DIR, SYSTEM_PKGS_GRANT) {
            Some(grant) => {
                self.pkgs_store = Some(grant);
                true
            }
            None => false,
        }
    }

    /// Open-or-create `directory` beneath the persisted root and grant it
    /// to the administration session under `display` with every right —
    /// the one bring-up the three system directories share.  Idempotent in
    /// both directions: a standing grant is adopted (a recovered volume
    /// already carries the directory and its members), and a file squatting
    /// on the name refuses rather than scattering members somewhere
    /// unexpected.
    fn establish_admin_directory(
        &mut self,
        directory: &str,
        display: &str,
    ) -> Option<crate::GrantId> {
        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        if !self.ensure_session(system) || !self.ensure_session(admin) {
            return None;
        }
        let (Ok(directory), Ok(display)) = (Name::new(directory), Name::new(display)) else {
            return None;
        };
        let grants = self.engine.grants(admin).ok()?;
        if let Some(found) = grants
            .iter()
            .find(|grant| grant.display_name == Some(display))
        {
            // Reuse across a reboot: the standing grant is the anchor.
            return Some(found.grant);
        }
        let root = self.engine.open_root(system, Rights::ALL).ok()?;
        let opened = match self
            .engine
            .open_at(system, root.handle, &directory, Rights::ALL)
        {
            Ok(opened) if opened.kind == ObjectKind::Directory => Ok(opened),
            Ok(opened) => {
                let _ = self.engine.close(system, opened.handle);
                Err(FsError::NotDirectory)
            }
            Err(FsError::NotFound) => {
                block_on(
                    self.engine
                        .create(system, root.handle, &directory, ObjectKind::Directory),
                )
            }
            Err(error) => Err(error),
        };
        let granted = match opened {
            Ok(opened) => {
                let granted = self
                    .engine
                    .grant(system, opened.handle, admin, Rights::ALL, Some(display))
                    .ok();
                let _ = self.engine.close(system, opened.handle);
                granted
            }
            Err(_) => None,
        };
        let _ = self.engine.close(system, root.handle);
        granted
    }

    /// Publish the canonical read-only application view into one already
    /// attached session. In the legacy composition the destination is the
    /// stable task-view session; in the split composition it is the secondary
    /// engine's empty pair-bound session. The latter must first leave its
    /// attach transition through an `apps`-rooted READ-only delegation: an
    /// empty session cannot receive a named grant, and leaving it transitional
    /// would make orderly peer teardown refuse forever. Either way only the
    /// application directory becomes reachable—never the ambient system root.
    pub(crate) fn install_apps_view(&mut self, session: SessionId) -> bool {
        let Some(apps) = self.apps_view else {
            return false;
        };
        match self
            .engine
            .install_view_root(session, apps, Rights::READ_ONLY)
        {
            // Split composition: complete the empty pair-bound session with
            // the narrowest structural root before publishing its display
            // name. Legacy composition already installed a workspace root.
            Ok(()) | Err(FsError::AlreadyExists) => {}
            Err(_) => return false,
        }
        self.install_named_view(session, apps, SYSTEM_APPS_VIEW)
    }

    /// The READ-only `libs` view beside `apps` — a named grant onto the
    /// shared library directory (the session's root is `apps`'s to
    /// install).  Best-effort like `apps`: a task without it still boots,
    /// it just cannot `require` a native module.
    pub(crate) fn install_libs_view(&mut self, session: SessionId) -> bool {
        let Some(libs) = self.libs_view else {
            return false;
        };
        self.install_named_view(session, libs, SYSTEM_LIBS_VIEW)
    }

    /// The READ-only `share` view beside `libs` — a named grant onto the
    /// shared data directory.  Best-effort like `libs`: a task without it
    /// still boots, it just cannot read the shipped data (`shared/wasm/…`).
    pub(crate) fn install_share_view(&mut self, session: SessionId) -> bool {
        let Some(share) = self.share_view else {
            return false;
        };
        self.install_named_view(session, share, SYSTEM_SHARE_VIEW)
    }

    /// Grant `anchor` (an admin-held directory grant) into `session` as a
    /// READ-only view displayed `display`, idempotently.  READ_ONLY strips
    /// WRITE, MUTATE_NAMESPACE, SHARE, and BIND_EXECUTABLE: a task can read
    /// and run what the directory holds but never write, relink, or bless it.
    fn install_named_view(
        &mut self,
        session: SessionId,
        anchor: crate::GrantId,
        display: &str,
    ) -> bool {
        let Ok(name) = Name::new(display) else {
            return false;
        };
        let Ok(grants) = self.engine.grants(session) else {
            return false;
        };
        if grants
            .iter()
            .any(|grant| grant.display_name == Some(name) && grant.rights == Rights::READ_ONLY)
        {
            return true;
        }
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let Ok(opened) = self
            .engine
            .open_grant(admin, anchor, Rights::READ.union(Rights::SHARE))
        else {
            return false;
        };
        let installed = self
            .engine
            .grant(admin, opened.handle, session, Rights::READ_ONLY, Some(name))
            .is_ok();
        let _ = self.engine.close(admin, opened.handle);
        installed
    }

    /// Install the writable workspace bridge from the persisted system
    /// namespace (session zero) to exec's fixed administration session.
    ///
    /// Every private task view roots its structural namespace in a
    /// delegation of this edge, so it is the route by which *all* ordinary
    /// user data is created. It must therefore be rooted inside session zero's
    /// tree: [`commit_engine`] serializes exactly what is reachable from that
    /// root, and an object reachable from anywhere else is written to its data
    /// blocks and then silently forgotten at the next mount — the whole of
    /// #575's data loss.
    ///
    /// The workspace is a **dedicated directory** beneath the system root, not
    /// the system root itself: [`SYSTEM_SOURCE_GRANT`] hands exec the same tree
    /// read-only to load executable images from, and the two authority edges
    /// must not alias one object, or a fault in ordinary file handling could
    /// reach the images exec is about to run.
    ///
    /// Idempotent in both directions, which is what makes it correct across a
    /// reboot: on a recovered volume the directory is already there and is
    /// reused (never re-created, which would strand the user's data under an
    /// orphaned object), and a second call finds the standing grant.
    ///
    /// [`commit_engine`]: crate::persistence::commit_engine
    pub(crate) fn establish_admin_workspace(&mut self) -> bool {
        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        if !self.ensure_session(system) || !self.ensure_session(admin) {
            return false;
        }
        let Ok(directory) = Name::new(SYSTEM_WORKSPACE_DIR) else {
            return false;
        };
        // Boot-local idempotence: the anchor grant lives exactly as long as
        // this service value.  The admin session's root IS the persisted
        // system root, so the directory name itself already occupies the
        // task-visible namespace — the grant therefore carries no display
        // name (the name-collision rule would refuse it), and the anchor is
        // held by id here instead of rediscovered by name.
        if self.workspace_view.is_some() {
            return true;
        }
        let Ok(root) = self.engine.open_root(system, Rights::ALL) else {
            return false;
        };
        // Reuse before create: `VolumeProbe::Kore` has already restored the
        // directory and everything under it, and creating a second edge of the
        // same name would either refuse or bury the recovered tree.
        let workspace = match self
            .engine
            .open_at(system, root.handle, &directory, Rights::ALL)
        {
            Ok(opened) if opened.kind == ObjectKind::Directory => Ok(opened),
            Ok(opened) => {
                // A non-directory squatting on the name: refuse rather than
                // silently relocate the workspace somewhere unexpected.
                let _ = self.engine.close(system, opened.handle);
                Err(FsError::NotDirectory)
            }
            // Blank volume. The happy path of `create` for a directory touches
            // only the namespace and authority arenas — its `.await`s are all
            // on unwind — so it completes on the first poll and `block_on`
            // cannot spin here.
            Err(FsError::NotFound) => {
                block_on(
                    self.engine
                        .create(system, root.handle, &directory, ObjectKind::Directory),
                )
            }
            Err(error) => Err(error),
        };
        let granted = match workspace {
            Ok(workspace) => {
                let granted = self
                    .engine
                    .grant(system, workspace.handle, admin, Rights::ALL, None)
                    .map(|grant| self.workspace_view = Some(grant))
                    .is_ok();
                let _ = self.engine.close(system, workspace.handle);
                granted
            }
            Err(_) => false,
        };
        let _ = self.engine.close(system, root.handle);
        granted
    }

    /// What `session` declared about its window, if it has attached one.
    pub fn attachment(&self, session: SessionId) -> Option<WindowAttachment> {
        self.slots
            .get(session.index() as usize)
            .and_then(|state| state.attached)
    }

    /// Locate the retained mapping owned by this frontend. The router borrows
    /// the returned span only while dispatching one serialized request into a
    /// sibling engine; the owner and mapping never move.
    #[cfg(target_os = "none")]
    pub(crate) fn mapped_window_span(&self, session: SessionId) -> Option<MappedWindowSpan> {
        self.slots
            .get(session.index() as usize)
            .and_then(|state| state.mapped.as_ref())
            .map(|window| MappedWindowSpan {
                base: window.base,
                len: window.len,
            })
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn mapped_window_span(&self, _session: SessionId) -> Option<MappedWindowSpan> {
        None
    }

    /// Adopt one returned `LeasedRegion` as `session`'s sole data window.
    /// The mapped length, not the peer's declaration, is authoritative.
    #[cfg(target_os = "none")]
    pub(crate) fn adopt_window(
        &mut self,
        session: SessionId,
        announced: u32,
        owner: ReturnedCapability,
    ) -> FsResult<()> {
        use kore_rt::cap::verbs;

        // The record first: a heap that cannot hold it refuses before the
        // view is taken.
        let state = self.slots.claim(session.index() as usize)?;
        if state.mapped.is_some() {
            return Err(FsError::AlreadyExists);
        }
        let slot = owner.slot().ok_or(FsError::NoWindow)?;
        let view = verbs::take_view(slot).ok_or(FsError::NoWindow)?;
        let mapped = verbs::map_view(view).ok_or(FsError::NoWindow)?;
        let len = mapped.len();
        if len == 0 || len > u32::MAX as usize {
            return Err(FsError::Range);
        }
        let base = mapped.cast::<u8>();
        state.mapped = Some(MappedWindow {
            _owner: owner,
            base,
            len,
        });
        state.attached = Some(WindowAttachment {
            slot: announced,
            bytes: len as u32,
        });
        Ok(())
    }

    /// Host builds have no capability arena or receiver mapping operation.
    #[cfg(not(target_os = "none"))]
    pub(crate) fn adopt_window(
        &mut self,
        _session: SessionId,
        _announced: u32,
        _owner: ReturnedCapability,
    ) -> FsResult<()> {
        Err(FsError::NoWindow)
    }

    /// Dispatch through this session's retained metal mapping.
    #[cfg(target_os = "none")]
    pub(crate) fn dispatch_mapped(
        &mut self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResponse {
        let mut mapped = self.mapped_window_span(session).map(|span| {
            // SAFETY: `adopt_window` mapped this exact live returned view and
            // retained its affine owner in the session's slot record.
            // `ServeHost` drives one request at a time, and the client
            // touches its sibling view only before publication or after the
            // correlated reply.
            DataWindow::new(unsafe {
                core::slice::from_raw_parts_mut(span.base.as_ptr(), span.len)
            })
        });
        self.dispatch(session, request, mapped.as_mut())
    }

    #[cfg(target_os = "none")]
    pub(crate) async fn dispatch_mapped_async(
        &mut self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResponse {
        let mut mapped = self.mapped_window_span(session).map(|span| {
            // SAFETY: the retained returned-capability owner keeps this exact
            // mapping live; the service executor serializes access to it.
            DataWindow::new(unsafe {
                core::slice::from_raw_parts_mut(span.base.as_ptr(), span.len)
            })
        });
        self.dispatch_async(session, request, mapped.as_mut()).await
    }

    /// Host tests inject their window directly through [`Self::dispatch`].
    #[cfg(not(target_os = "none"))]
    pub(crate) fn dispatch_mapped(
        &mut self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResponse {
        self.dispatch(session, request, None)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) async fn dispatch_mapped_async(
        &mut self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResponse {
        self.dispatch_async(session, request, None).await
    }

    /// Serve one request.
    ///
    /// `session` comes from the serving host, never from `request` — see the
    /// module docs. `window` is the host's view of whatever the client
    /// delivered; a bulk verb with no attachment, or with no window mapped,
    /// refuses [`FsError::NoWindow`] rather than reading memory it was not
    /// given.
    pub fn dispatch(
        &mut self,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResponse {
        block_on(self.dispatch_async(session, request, window))
    }

    pub(crate) async fn dispatch_async(
        &mut self,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResponse {
        self.drain_pending_reclaims().await;
        match self.serve_async(session, request, window).await {
            Ok(response) => response,
            Err(err) => FsResponse::Refused(err),
        }
    }

    /// Release the placements and reclaim the roots dead peers left behind,
    /// before the request that woke this dispatch is served. Ordinary await
    /// points: a `Busy` claim held by an in-flight request resolves on that
    /// request's next turn, and a placement the provider refuses to release
    /// stays pending for the next dispatch, retaining custody.
    async fn drain_pending_reclaims(&mut self) {
        // Released in place: a refused stage keeps its position, and its
        // custody, without the list ever growing.
        let mut index = 0;
        while index < self.pending_aborts.len() {
            if self
                .engine
                .abort_replace(&self.pending_aborts[index])
                .await
                .is_ok()
            {
                self.pending_aborts.swap_remove(index);
            } else {
                index += 1;
            }
        }
        if self.pending_reclaims.is_empty() {
            return;
        }
        self.engine.reclaim_stranded(&self.pending_reclaims).await;
        self.pending_reclaims.clear();
    }

    /// The dispatch body, written against `?` so every refusal path is one
    /// line. [`Self::dispatch`] folds the error into a `Refused`.
    async fn serve_async(
        &mut self,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResult<FsResponse> {
        // A bulk verb needs both halves: **this session's** declared
        // attachment, and a window the host actually mapped. Either missing is
        // `NoWindow` — the service will not touch memory it was not handed,
        // and it will not answer one client's presence check with another's
        // record.
        let window = if request.needs_window() {
            if self.attachment(session).is_none() {
                return Err(FsError::NoWindow);
            }
            Some(window.ok_or(FsError::NoWindow)?)
        } else {
            None
        };

        Ok(match *request {
            FsRequest::AttachWindow { slot, bytes } => {
                // An out-of-range session has no attachment slot, and refusing
                // is the only honest answer: writing it anywhere else would
                // hand this client somebody's record.
                self.slots.claim(session.index() as usize)?.attached =
                    Some(WindowAttachment { slot, bytes });
                FsResponse::WindowAttached {
                    max_transfer: self.transfer_limit(),
                }
            }

            FsRequest::OpenRoot { rights } => opened(self.engine.open_root(session, rights)?),
            FsRequest::OpenAt {
                parent,
                name,
                rights,
            } => opened(self.engine.open_at(session, parent, &name, rights)?),
            FsRequest::OpenGrant { grant, rights } => {
                opened(match self.task_view_for_session(session) {
                    Some(view) => self.engine.open_task_grant(session, view, grant, rights)?,
                    None => self.engine.open_grant(session, grant, rights)?,
                })
            }
            FsRequest::Create {
                parent,
                name,
                kind,
                subtype,
            } => opened(
                self.engine
                    .create_typed(session, parent, &name, kind, subtype)
                    .await?,
            ),

            FsRequest::Close { handle } => {
                if let Some(state) = self.slots.get_mut(session.index() as usize)
                    && let Some(stage) = state
                        .replacement
                        .as_ref()
                        .filter(|stage| stage.handle == handle)
                {
                    self.engine.abort_replace(stage).await?;
                    state.replacement = None;
                }
                self.engine.close(session, handle)?;
                FsResponse::Done
            }
            FsRequest::Stat { handle } => {
                let info = self.engine.stat(session, handle)?;
                FsResponse::Stat {
                    object: info.object,
                    stable_object: info.stable_object,
                    kind: info.kind,
                    subtype: info.subtype,
                    len: info.len,
                    revision: info.revision,
                    links: info.links,
                    grants: info.grants,
                }
            }

            FsRequest::List {
                handle,
                cursor,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let page = self.engine.list(session, handle, cursor)?;
                // Refuse rather than truncate: a client that asked for a page
                // its window cannot hold must hear so, not silently receive
                // fewer entries than the service listed.
                if page.entries.len() > window.entry_capacity(window_offset) {
                    return Err(FsError::Range);
                }
                for (index, entry) in page.entries.iter().enumerate() {
                    let mut wire = FsDirEntryWire {
                        name_len: entry.name.len() as u8,
                        kind: match entry.kind {
                            ObjectKind::File => FsReqWire::KIND_FILE,
                            ObjectKind::Directory => FsReqWire::KIND_DIRECTORY,
                        },
                        // The low byte carries the entry's subtype (data, never
                        // authority); the high byte stays reserved.
                        reserved0: u16::from(entry.subtype.to_bits()).into(),
                        reserved1: 0.into(),
                        object: entry.object.to_bits().into(),
                        name: [0; kore_abi::svc::FS_NAME_BYTES],
                    };
                    let bytes = entry.name.as_bytes();
                    wire.name[..bytes.len()].copy_from_slice(bytes);
                    window.put_entry(window_offset, index, &wire)?;
                }
                FsResponse::Listed {
                    entries: page.entries.len() as u32,
                    next: page.next,
                }
            }

            FsRequest::Locate {
                handle,
                stable,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                if window.search_capacity(window_offset) == 0 {
                    return Err(FsError::Range);
                }
                let found = self.engine.locate(session, handle, stable)?;
                let (path, path_len) = spell_path(&found.path)?;
                let kind = if found.kind.is_dir() {
                    FsReqWire::KIND_DIRECTORY
                } else {
                    FsReqWire::KIND_FILE
                };
                window.put_search_hit(
                    window_offset,
                    0,
                    &FsSearchHitWire {
                        path_len: u16::try_from(path_len).map_err(|_| FsError::Range)?.into(),
                        kind,
                        reserved0: 0,
                        ranking: 0.into(),
                        object: found.object.to_bits().into(),
                        stable_object: found.stable_object.get().into(),
                        revision: found.revision.into(),
                        len: found.len.into(),
                        path,
                        snippet: [0; crate::search::MAX_SEARCH_SNIPPET],
                        reserved2: [0; 1],
                    },
                )?;
                FsResponse::Listed {
                    entries: 1,
                    next: None,
                }
            }

            FsRequest::Parents {
                handle,
                cursor,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                // The page is sized to the window, so a small window pages
                // instead of refusing; one that holds no record is `Range`.
                let capacity = window.search_capacity(window_offset);
                let page = self.engine.parents(session, handle, cursor, capacity)?;
                for (index, parent) in page.parents.iter().enumerate() {
                    let (path, path_len) = spell_path(&parent.path)?;
                    window.put_search_hit(
                        window_offset,
                        index,
                        &FsSearchHitWire {
                            path_len: u16::try_from(path_len).map_err(|_| FsError::Range)?.into(),
                            kind: FsReqWire::KIND_DIRECTORY,
                            reserved0: 0,
                            ranking: 0.into(),
                            object: parent.object.to_bits().into(),
                            stable_object: parent.stable_object.get().into(),
                            revision: parent.revision.into(),
                            len: 0.into(),
                            path,
                            snippet: [0; crate::search::MAX_SEARCH_SNIPPET],
                            reserved2: [0; 1],
                        },
                    )?;
                }
                FsResponse::Listed {
                    entries: page.parents.len() as u32,
                    next: page.next,
                }
            }

            FsRequest::ListGrants { window_offset } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let grants = self
                    .engine
                    .grants(self.task_view_for_session(session).unwrap_or(session))?;
                if grants.len() > window.entry_capacity(window_offset) {
                    return Err(FsError::Range);
                }
                for (index, info) in grants.iter().enumerate() {
                    let mut wire = kore_abi::svc::FsGrantEntryWire {
                        name_len: 0,
                        rights: info.rights.bits(),
                        kind: match info.kind {
                            crate::authority::AuthorityKind::PrivateRoot => {
                                kore_abi::svc::FsGrantEntryWire::KIND_PRIVATE_ROOT
                            }
                            crate::authority::AuthorityKind::Grant => {
                                kore_abi::svc::FsGrantEntryWire::KIND_GRANT
                            }
                        },
                        reserved0: 0,
                        reserved1: 0.into(),
                        grant: info.grant.to_bits().into(),
                        name: [0; kore_abi::svc::FS_NAME_BYTES],
                    };
                    if let Some(display) = info.display_name.as_ref() {
                        let bytes = display.as_bytes();
                        wire.name[..bytes.len()].copy_from_slice(bytes);
                        wire.name_len = bytes.len() as u8;
                    }
                    window.put_grant_entry(window_offset, index, &wire)?;
                }
                FsResponse::GrantsListed {
                    entries: grants.len() as u32,
                }
            }

            FsRequest::SystemStatus { window_offset } => {
                use kore_abi::svc::FsSystemStatusWire;
                let window = window.ok_or(FsError::NoWindow)?;
                let (tree_objects, tree_edges, tree_bytes) = self.engine.system_tree_usage()?;
                let mut views_private: u16 = 0;
                let mut views_pinned: u16 = 0;
                for view in self.invocation_views.iter().flatten() {
                    match view.owner {
                        InvocationViewOwner::Private { .. } => views_private += 1,
                        InvocationViewOwner::Pinned { .. } => views_pinned += 1,
                    }
                }
                let mut flags: u16 = 0;
                if self.backend.durable {
                    flags |= FsSystemStatusWire::FLAG_DURABLE;
                }
                if self.backend.degraded_read_only {
                    flags |= FsSystemStatusWire::FLAG_DEGRADED_READ_ONLY;
                }
                if self.backend.route_held {
                    flags |= FsSystemStatusWire::FLAG_ROUTE_HELD;
                }
                if self.workspace_view.is_some() {
                    flags |= FsSystemStatusWire::FLAG_WORKSPACE;
                }
                if self.system_view.is_some() {
                    flags |= FsSystemStatusWire::FLAG_SYSTEM_VIEW;
                }
                if self.backend.channel_lost {
                    flags |= FsSystemStatusWire::FLAG_CHANNEL_LOST;
                }
                if self.backend.host_memory_buffer {
                    flags |= FsSystemStatusWire::FLAG_HOST_MEMORY_BUFFER;
                }
                let wire = FsSystemStatusWire {
                    sessions: self.engine.live_sessions().into(),
                    peers: self.slots.count(|state| state.via.is_some()).into(),
                    views_private: views_private.into(),
                    views_pinned: views_pinned.into(),
                    windows: self.slots.count(|state| state.attached.is_some()).into(),
                    flags: flags.into(),
                    tree_objects: tree_objects.into(),
                    tree_edges: tree_edges.into(),
                    user_lifecycle: self.backend.user_lifecycle.into(),
                    system_lifecycle: self.backend.system_lifecycle.into(),
                    tree_bytes: tree_bytes.into(),
                    storage: self.backend.storage,
                    note: self.backend.note.into(),
                    commits: self.backend.commits.into(),
                    reads: self.backend.provider.reads.into(),
                    writes: self.backend.provider.writes.into(),
                    flushes: self.backend.provider.flushes.into(),
                    discards: self.backend.provider.discards.into(),
                    copied_bytes: self.backend.provider.copied_bytes.into(),
                    rollbacks: self.backend.provider.rollbacks.into(),
                    late_completions: self.backend.provider.late_completions.into(),
                    orphan_reclaims: self.backend.provider.orphan_reclaims.into(),
                    reclaim_failures: self.backend.provider.reclaim_failures.into(),
                    submissions: self.backend.channel.submissions.into(),
                    completions: self.backend.channel.completions.into(),
                    backpressure: self.backend.channel.backpressure.into(),
                    poll_calls: self.backend.channel.poll_calls.into(),
                    device_waits: self.backend.channel.device_waits.into(),
                    route_parks: self.backend.route_parks.into(),
                    route_fires: self.backend.route_fires.into(),
                    deadline_waits: self.backend.deadline_waits.into(),
                    spin_waits: self.backend.spin_waits.into(),
                    monitor_sleeps: self.backend.monitor_sleeps.into(),
                    monitor_turns: self.backend.monitor_turns.into(),
                    expired_waits: self.backend.provider.expired_waits.into(),
                    slow_waits: self.backend.provider.slow_waits.into(),
                    leaked_placements: u64::from(self.engine.leaked_placements()).into(),
                    abandoned_reclaims: u64::from(self.engine.abandoned_reclaims()).into(),
                    readahead_hits: self.backend.provider.readahead_hits.into(),
                    readahead_reads: self.backend.provider.readahead_reads.into(),
                    latency: self
                        .backend
                        .channel
                        .latency
                        .snapshot(self.backend.channel.max_outstanding),
                };
                window.put_system_status(window_offset, &wire)?;
                FsResponse::SystemStatusReported {
                    len: FsSystemStatusWire::BYTES as u32,
                }
            }

            FsRequest::Search {
                handle,
                query,
                cursor,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let page = self.engine.search(session, handle, query, cursor)?;
                if page.hits.len() > window.search_capacity(window_offset) {
                    return Err(FsError::Range);
                }
                for (index, hit) in page.hits.iter().enumerate() {
                    let (path, path_len) = spell_path(&hit.path)?;
                    let mut snippet = [0u8; crate::search::MAX_SEARCH_SNIPPET];
                    let snippet_len = hit.snippet.len();
                    snippet[..snippet_len].copy_from_slice(hit.snippet.as_bytes());
                    let ranking = u32::from(hit.score.get())
                        | (u32::try_from(snippet_len).map_err(|_| FsError::Range)? << 16);
                    window.put_search_hit(
                        window_offset,
                        index,
                        &FsSearchHitWire {
                            path_len: u16::try_from(path_len).map_err(|_| FsError::Range)?.into(),
                            kind: match hit.kind {
                                ObjectKind::File => FsReqWire::KIND_FILE,
                                ObjectKind::Directory => FsReqWire::KIND_DIRECTORY,
                            },
                            reserved0: 0,
                            ranking: ranking.into(),
                            object: hit.object.to_bits().into(),
                            stable_object: hit.stable_object.get().into(),
                            revision: hit.revision.into(),
                            len: hit.len.into(),
                            path,
                            snippet,
                            reserved2: [0; 1],
                        },
                    )?;
                }
                FsResponse::Searched {
                    hits: page.hits.len() as u32,
                    observation: page.observation,
                    snapshot: page.snapshot,
                    index_generation: page.index_generation,
                    freshness: page.freshness,
                    next: page.next,
                    scanned: page.scanned,
                    work: page.work,
                }
            }

            FsRequest::Read {
                handle,
                offset,
                window_offset,
                length,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let dst = window.destination(window_offset, length)?;
                let count = self.engine.read(session, handle, offset, dst).await?;
                FsResponse::Transferred {
                    count: count as u32,
                }
            }
            FsRequest::BeginReplace {
                handle,
                length,
                expected,
            } => {
                let state = self.slots.claim(session.index() as usize)?;
                if state.replacement.is_some() {
                    return Err(FsError::Busy);
                }
                state.replacement = Some(
                    self.engine
                        .begin_replace(session, handle, length, expected)
                        .await?,
                );
                FsResponse::Done
            }
            FsRequest::WriteReplacement {
                handle,
                offset,
                window_offset,
                length,
            } => {
                let stage = self
                    .slots
                    .get_mut(session.index() as usize)
                    .and_then(|state| state.replacement.as_mut())
                    .filter(|stage| stage.handle == handle)
                    .ok_or(FsError::InvalidHandle)?;
                let src = window
                    .ok_or(FsError::NoWindow)?
                    .source(window_offset, length)?;
                self.engine
                    .write_replacement(session, stage, offset, src)
                    .await?;
                FsResponse::Done
            }
            FsRequest::CommitReplace { handle } => {
                // A slot past the table has staged nothing.
                let state = self
                    .slots
                    .get_mut(session.index() as usize)
                    .ok_or(FsError::InvalidHandle)?;
                let stage = state
                    .replacement
                    .as_ref()
                    .filter(|stage| stage.handle == handle)
                    .ok_or(FsError::InvalidHandle)?;
                let revision = self.engine.commit_replace(session, stage).await?;
                // The object owns the placement from this point onward.
                state.replacement = None;
                FsResponse::Committed { revision }
            }
            FsRequest::AbortReplace { handle } => {
                if let Some(state) = self.slots.get_mut(session.index() as usize)
                    && let Some(stage) = state.replacement.as_ref()
                {
                    if stage.handle != handle {
                        return Err(FsError::InvalidHandle);
                    }
                    self.engine.abort_replace(stage).await?;
                    state.replacement = None;
                }
                FsResponse::Done
            }
            FsRequest::WriteAt {
                handle,
                offset,
                window_offset,
                length,
                expected,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                // The borrow must end before the engine call: `write_at` takes
                // `&mut self` on the engine, and the source is the window, not
                // the engine — but the compiler cannot see that through the
                // service, so the bytes are copied out of the window's borrow
                // by taking a shared one instead.
                let src = window.source(window_offset, length)?;
                let revision = self
                    .engine
                    .write_at(session, handle, offset, src, expected)
                    .await?;
                FsResponse::Committed { revision }
            }
            FsRequest::Append {
                handle,
                window_offset,
                length,
                expected,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let src = window.source(window_offset, length)?;
                let (offset, revision) = self.engine.append(session, handle, src, expected).await?;
                FsResponse::Appended { offset, revision }
            }
            FsRequest::Resize {
                handle,
                length,
                expected,
            } => {
                let revision = self
                    .engine
                    .resize(session, handle, length, expected)
                    .await?;
                FsResponse::Committed { revision }
            }

            FsRequest::Rename { parent, old, new } => {
                self.engine.rename(session, parent, &old, &new)?;
                FsResponse::Done
            }
            FsRequest::Relocate {
                src_parent,
                name,
                dst_parent,
                new_name,
            } => {
                self.engine
                    .relocate(session, src_parent, &name, dst_parent, &new_name)?;
                FsResponse::Done
            }
            FsRequest::Link {
                parent,
                name,
                child,
            } => {
                self.engine.link(session, parent, &name, child)?;
                FsResponse::Done
            }
            FsRequest::Unlink { parent, name } => {
                self.engine.unlink(session, parent, &name).await?;
                FsResponse::Done
            }
            FsRequest::DetachEdge { parent, name } => {
                self.engine.detach_edge(session, parent, &name).await?;
                FsResponse::Done
            }

            FsRequest::Grant {
                handle,
                destination,
                rights,
                display_name,
            } => {
                let grant =
                    self.engine
                        .grant(session, handle, destination, rights, display_name)?;
                FsResponse::Granted { grant }
            }
            FsRequest::GrantScoped {
                handle,
                destination,
                rights,
                quota,
                display_name,
            } => {
                let grant = self.engine.grant_scoped(
                    session,
                    handle,
                    destination,
                    rights,
                    quota,
                    display_name,
                )?;
                FsResponse::Granted { grant }
            }
            FsRequest::Revoke { grant } => {
                self.engine
                    .revoke(
                        self.task_view_for_session(session).unwrap_or(session),
                        grant,
                    )
                    .await?;
                FsResponse::Done
            }
            FsRequest::DeferCommits { hold } => {
                // Releasing a hold never needs a record a slot does not have.
                if hold {
                    self.slots.claim(session.index() as usize)?.deferring = true;
                } else if let Some(state) = self.slots.get_mut(session.index() as usize) {
                    state.deferring = false;
                }
                FsResponse::Done
            }

            FsRequest::SetSystemSeed { .. } => {
                return Err(FsError::Unsupported);
            }

            FsRequest::Sync { handle } => {
                self.engine.sync(session, handle).await?;
                FsResponse::Done
            }
            FsRequest::LendVersion {
                handle,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                // Validate the destination before installing an affine
                // snapshot.  Once `lend_version` succeeds the response is the
                // caller's only route to the generational lease; discovering
                // a bad window afterward would strand that lease inside the
                // service with no owner able to release it.
                let dst = window.destination(window_offset, 56)?;
                let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
                let (lease, identity) = self
                    .engine
                    .lend_version(owner, session, handle)
                    .await
                    .map_err(extraction_error)?;
                write_identity(dst, identity);
                FsResponse::VersionLent { lease }
            }
            FsRequest::LendExecutable {
                handle,
                window_offset,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let dst = window.destination(window_offset, 56)?;
                let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
                let (lease, identity) = self
                    .engine
                    .lend_executable_version(owner, session, handle)
                    .await
                    .map_err(extraction_error)?;
                write_identity(dst, identity);
                FsResponse::VersionLent { lease }
            }
            FsRequest::ReadVersion {
                lease,
                offset,
                window_offset,
                length,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let dst = window.destination(window_offset, length)?;
                let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
                let count = self
                    .engine
                    .read_lent_version(owner, lease, offset, dst)
                    .await
                    .map_err(extraction_error)?;
                FsResponse::Transferred {
                    count: count as u32,
                }
            }
            FsRequest::CommitDerived {
                lease,
                window_offset,
                length,
                output_window_bytes,
            } => {
                let window = window.ok_or(FsError::NoWindow)?;
                let bytes = window.source(window_offset, length)?;
                let result =
                    kore_svc_extract::decode_result(bytes).map_err(|_| FsError::Unsupported)?;
                let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
                let info = self
                    .engine
                    .commit_encoded_derived_result(
                        owner,
                        lease,
                        result,
                        bytes,
                        u64::from(output_window_bytes),
                    )
                    .await
                    .map_err(extraction_error)?;
                FsResponse::DerivedCommitted {
                    parser: info.parser,
                    parser_version: info.parser_version,
                }
            }
            FsRequest::ReleaseVersion { lease } => {
                let owner = crate::extraction::ExtractionOwner::new(u64::from(session.index()));
                self.engine
                    .release_lent_version(owner, lease)
                    .await
                    .map_err(extraction_error)?;
                FsResponse::Done
            }
            FsRequest::DerivedInfo { handle, parser } => {
                let Some(result) = self.engine.derived_result(session, handle, parser)? else {
                    return Err(FsError::NotFound);
                };
                FsResponse::DerivedInfo {
                    input_version: result.provenance.input.version,
                    parser_version: result.provenance.parser_version,
                }
            }
            // Host-control verbs are intercepted by `serve_request`, where a
            // live `Serve` proves the arrival pair and carries the returned
            // capability.  The pure engine dispatch has neither authority and
            // must never emulate them for a host-test caller.
            FsRequest::RegisterSystemView { .. }
            | FsRequest::RegisterPrivateView { .. }
            | FsRequest::GrantView { .. }
            | FsRequest::ApplyView { .. }
            | FsRequest::BindView { .. }
            | FsRequest::AdmitPeer { .. }
            | FsRequest::TakeAdmission
            | FsRequest::RetirePeer { .. }
            | FsRequest::ConfirmRetired { .. }
            | FsRequest::ClosePeer => {
                return Err(FsError::AccessDenied);
            }
            // Storage rings ride the transport engine beside the engines,
            // served by the durable row that owns the controller. An engine
            // has no ring to attach.
            FsRequest::StorAttach { .. }
            | FsRequest::StorGrantPack
            | FsRequest::StorGrantVersion { .. }
            | FsRequest::StorTarget { .. }
            | FsRequest::StorRelease { .. }
            | FsRequest::StorKick
            | FsRequest::StorInspect
            | FsRequest::StorMaintain { .. }
            | FsRequest::StorDetach => {
                return Err(FsError::Unsupported);
            }
        })
    }
}

fn extraction_error(error: crate::ExtractionError) -> FsError {
    use crate::ExtractionError;
    match error {
        ExtractionError::Filesystem(error) => error,
        ExtractionError::Provider(error) => error.into(),
        // Two different bounded RAM tables, neither of them storage. They
        // used to share `OutOfStorage`, which sent a reader looking at the
        // disk for a table that had simply filled — and put `out of storage`
        // on screen above an `ls` listing the file that had just been made.
        ExtractionError::NoSnapshotSlots => FsError::OutOfVersionLeases,
        ExtractionError::NoResultSlots => FsError::OutOfIndexSlots,
        ExtractionError::InvalidLease => FsError::InvalidHandle,
        ExtractionError::ObsoleteVersion => FsError::Conflict,
        ExtractionError::InvalidResult(_) => FsError::Unsupported,
        // A batch the index cannot hold leaves its version unindexed, which
        // every producer already reports as such; as `Range` it failed the
        // write the batch describes, after that write had committed.
        ExtractionError::Oversize => FsError::Unsupported,
        ExtractionError::Range => FsError::Range,
    }
}

/// Domain-level refusal for a structurally invalid recyclable-peer component.
///
/// The generic assembler keeps the detailed lifecycle vocabulary private to
/// the serving host.  On the filesystem wire, duplicates are ordinary
/// conflicts, an impossible grant position is a range error, and every
/// authority/species mismatch is denied.  A return leg on a protocol that did
/// not declare one is a protocol-shape refusal rather than missing authority.
fn peer_assembly_error(error: PeerAssemblyRefusal) -> FsError {
    match error {
        PeerAssemblyRefusal::Duplicate => FsError::AlreadyExists,
        PeerAssemblyRefusal::GrantOutOfRange => FsError::Range,
        PeerAssemblyRefusal::UnexpectedReturn => FsError::Unsupported,
        PeerAssemblyRefusal::WrongAuthority => FsError::AccessDenied,
    }
}

fn peer_admission_error(error: PeerAdmissionRefusal) -> FsError {
    match error {
        PeerAdmissionRefusal::Capacity | PeerAdmissionRefusal::NoScratchSlots => {
            FsError::OutOfHandles
        }
        PeerAdmissionRefusal::WrongRingAuthority
        | PeerAdmissionRefusal::WrongEndpointAuthority
        | PeerAdmissionRefusal::WrongGrantCount
        | PeerAdmissionRefusal::WrongReturnShape => FsError::AccessDenied,
        PeerAdmissionRefusal::MapFailed => FsError::NoWindow,
        // A stale registration standing in the arrival's slot. `AlreadyExists`
        // is the reading that is true, and exec routes it to the refusal that
        // means "fs refused for a non-capacity reason" rather than to the one
        // that means its object table is full.
        PeerAdmissionRefusal::PolicyCollision => FsError::AlreadyExists,
        PeerAdmissionRefusal::PolicyUnavailable => FsError::OutOfObjects,
        PeerAdmissionRefusal::InvalidMapPlan
        | PeerAdmissionRefusal::RingGeometry
        | PeerAdmissionRefusal::RingMalformed => FsError::Unsupported,
    }
}

/// The four open-shaped verbs share one reply.
fn opened(object: crate::engine::OpenedObject) -> FsResponse {
    FsResponse::Opened {
        handle: object.handle,
        stable_object: object.stable_object,
        kind: object.kind,
        subtype: object.subtype,
        len: object.len,
        revision: object.revision,
    }
}

#[cfg(test)]
mod recyclable_session_tests {
    use super::*;

    fn install_private_task_view(
        service: &mut RamService,
        scope: u64,
        session: SessionId,
        rights: Rights,
    ) {
        block_on(service.engine.attach_session(session)).unwrap();
        let grant = service.engine.task_view_root(session).unwrap();
        let slot = service
            .invocation_views
            .iter()
            .position(Option::is_none)
            .unwrap();
        service.invocation_views[slot] = Some(InvocationView {
            scope,
            grant,
            rights,
            task_session: session,
            owner: InvocationViewOwner::Private { session },
            canary: None,
            home: None,
        });
    }

    fn admit_view(service: &mut RamService, arrival: u16, scope: u64, rights: Rights) -> SessionId {
        let peer = PeerId::from_arrival(arrival).unwrap();
        assert!(service.peer_admitted(Ok(peer)));
        service.apply_invocation_view(scope, peer, rights).unwrap();
        SessionId::new(peer.slot() as u16)
    }

    #[test]
    fn recyclable_arrivals_require_the_exact_current_generation() {
        let mut service = RamService::default();
        service.slots.claim(2).unwrap().via = Some(0x102);
        service.slots.claim(3).unwrap().via = Some(0x203);

        assert_eq!(service.session_for_via(0), Some(SessionId::new(0)));
        assert_eq!(service.session_for_via(1), Some(SessionId::new(1)));
        assert_eq!(service.session_for_via(0x102), Some(SessionId::new(2)));
        assert_eq!(service.session_for_via(0x203), Some(SessionId::new(3)));
        assert_eq!(service.session_for_via(0x202), None);
    }

    #[test]
    fn peer_close_releases_only_the_exact_arrival_state() {
        let mut service = RamService::default();
        assert!(service.ensure_session(SessionId::new(2)));
        service.slots.claim(2).unwrap().via = Some(0x102);
        service.slots.claim(2).unwrap().attached = Some(WindowAttachment {
            slot: 9,
            bytes: 4096,
        });

        assert!(service.peer_closed(0x202, PeerClose::Died));
        assert_eq!(service.session_for_via(0x102), Some(SessionId::new(2)));
        assert!(service.attachment(SessionId::new(2)).is_some());

        assert!(service.peer_closed(0x102, PeerClose::Orderly));
        assert_eq!(service.session_for_via(0x102), None);
        assert!(service.attachment(SessionId::new(2)).is_none());
        assert_eq!(
            service.engine.open_root(SessionId::new(2), Rights::READ),
            Err(FsError::InvalidHandle)
        );
    }

    #[test]
    fn admin_pair_gets_a_named_executable_source_grant() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert_eq!(service.session_for_via(1), Some(SessionId::new(1)));

        let display = Name::new(SYSTEM_SOURCE_GRANT).unwrap();
        let grant = service
            .engine
            .grants(SessionId::new(1))
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(display))
            .expect("admin source grant");
        assert_eq!(
            grant.rights,
            Rights::READ
                .union(Rights::SHARE)
                .union(Rights::BIND_EXECUTABLE)
        );
        let opened = service
            .engine
            .open_grant(SessionId::new(1), grant.grant, Rights::READ)
            .expect("admin can open the granted system root");
        assert_eq!(opened.kind, ObjectKind::Directory);
    }

    /// The invariant every user file's durability rests on: what exec's
    /// workspace creates must hang beneath session zero's root, because that
    /// is the only tree [`commit_engine`] walks. #575 shipped a workspace
    /// rooted in exec's own private root instead — writes reached their data
    /// blocks and the names that found them were never serialized, so the
    /// filesystem came back empty after a reboot.
    ///
    /// [`commit_engine`]: crate::persistence::commit_engine
    /// The two bounded index tables must never share one refusal again.
    ///
    /// They both used to answer `OutOfStorage`, which is what put
    /// `touch: out of storage` on screen above an `ls` listing the file that
    /// had just been created and committed. Neither table is storage: a full
    /// lease pool means a snapshot is still held, a full result table means
    /// the index cannot grow, and the volume is fine in both cases.
    #[test]
    fn the_two_index_tables_do_not_share_a_refusal() {
        use crate::ExtractionError;
        assert_eq!(
            extraction_error(ExtractionError::NoSnapshotSlots),
            FsError::OutOfVersionLeases
        );
        assert_eq!(
            extraction_error(ExtractionError::NoResultSlots),
            FsError::OutOfIndexSlots
        );
        assert_ne!(
            extraction_error(ExtractionError::NoSnapshotSlots),
            extraction_error(ExtractionError::NoResultSlots)
        );
        // And neither is reported as the medium being full.
        for error in [
            ExtractionError::NoSnapshotSlots,
            ExtractionError::NoResultSlots,
        ] {
            assert_ne!(extraction_error(error), FsError::OutOfStorage);
        }
    }

    #[test]
    fn the_workspace_creates_inside_the_tree_commit_engine_serializes() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let grant = service.workspace_view.expect("workspace anchor stands");

        // Create through the delegated edge, exactly as task views do.
        let workspace = service
            .engine
            .open_grant(admin, grant, Rights::ALL)
            .expect("admin can open the granted workspace");
        assert_eq!(workspace.kind, ObjectKind::Directory);
        let probe = Name::new("probe").unwrap();
        block_on(
            service
                .engine
                .create(admin, workspace.handle, &probe, ObjectKind::File),
        )
        .expect("the workspace is writable");

        // Now walk to it from the persisted root. A workspace that is not
        // reachable from here is a workspace whose contents are dropped at
        // the next mount, which is exactly the bug this asserts against.
        let root = service
            .engine
            .open_root(system, Rights::ALL)
            .expect("system root");
        let directory = Name::new(SYSTEM_WORKSPACE_DIR).unwrap();
        let reached = service
            .engine
            .open_at(system, root.handle, &directory, Rights::ALL)
            .expect("the workspace hangs beneath the persisted root");
        service
            .engine
            .open_at(system, reached.handle, &probe, Rights::READ)
            .expect("a file created through the workspace is reachable from the persisted root");
    }

    /// The rerouting this stack exists for: a private task view's session
    /// roots in a delegation of the workspace, so what a task creates is
    /// reachable from the persisted root — the #575 invariant expressed in
    /// the invocation-view model.
    #[test]
    fn a_private_task_views_files_are_facts_of_the_persisted_tree() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        // The real registration path: the view roots in its own durable
        // home, `work/alpha`.
        let alpha = Name::new("alpha").unwrap();
        service
            .register_private_invocation_view(0xA1, &alpha, Rights::ALL, None)
            .expect("private view registers over its durable home");
        let task = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xA1)
            .map(|view| view.task_session)
            .expect("registered view");

        let root = service
            .engine
            .open_root(task, Rights::ALL)
            .expect("task view root");
        let name = Name::new("journal").unwrap();
        block_on(
            service
                .engine
                .create(task, root.handle, &name, ObjectKind::File),
        )
        .expect("task creates in its view");

        // Walk to it from the persisted root: work/alpha/journal.
        let system = SessionId::new(0);
        let system_root = service
            .engine
            .open_root(system, Rights::ALL)
            .expect("system root");
        let directory = Name::new(SYSTEM_WORKSPACE_DIR).unwrap();
        let work = service
            .engine
            .open_at(system, system_root.handle, &directory, Rights::ALL)
            .expect("workspace under the persisted root");
        let home = service
            .engine
            .open_at(system, work.handle, &alpha, Rights::ALL)
            .expect("the view's home directory under the workspace");
        service
            .engine
            .open_at(system, home.handle, &name, Rights::READ)
            .expect("a task-created file is a fact of the persisted tree");
    }

    /// The two-terminal contract the shared-workspace draft of this fix broke:
    /// distinct views must never alias one namespace just because both are
    /// durable. Each roots in its own `work/<name>` home, so a creation in
    /// one is invisible from the other while both remain facts of the one
    /// tree `commit_engine` serializes.
    #[test]
    fn named_views_are_isolated_yet_both_persist_under_the_workspace() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        let alpha = Name::new("alpha").unwrap();
        let beta = Name::new("beta").unwrap();
        service
            .register_private_invocation_view(0xA1, &alpha, Rights::ALL, None)
            .expect("first view registers");
        service
            .register_private_invocation_view(0xB1, &beta, Rights::ALL, None)
            .expect("second view registers");
        let session_of = |service: &RamService, scope: u64| {
            service
                .invocation_views
                .iter()
                .flatten()
                .find(|view| view.scope == scope)
                .map(|view| view.task_session)
                .expect("registered view")
        };
        let first = session_of(&service, 0xA1);
        let second = session_of(&service, 0xB1);

        let root = service.engine.open_root(first, Rights::ALL).unwrap();
        let secret = Name::new("a-only").unwrap();
        block_on(
            service
                .engine
                .create(first, root.handle, &secret, ObjectKind::File),
        )
        .expect("first view creates");

        // Isolation: the second view's root simply has no such name.
        let other_root = service.engine.open_root(second, Rights::ALL).unwrap();
        assert_eq!(
            service
                .engine
                .open_at(second, other_root.handle, &secret, Rights::READ)
                .map(|_| ())
                .unwrap_err(),
            FsError::NotFound
        );

        // Persistence: both homes hang beneath the one persisted root.
        let system = SessionId::new(0);
        let system_root = service.engine.open_root(system, Rights::ALL).unwrap();
        let directory = Name::new(SYSTEM_WORKSPACE_DIR).unwrap();
        let work = service
            .engine
            .open_at(system, system_root.handle, &directory, Rights::ALL)
            .unwrap();
        for home in [&alpha, &beta] {
            service
                .engine
                .open_at(system, work.handle, home, Rights::READ)
                .expect("each view's home is a fact of the persisted tree");
        }
    }

    /// The reboot story at view granularity: a reaped view's home directory
    /// stays in the tree, and the next registration of the same name roots
    /// over it — the files are still there. This is what carries a terminal's
    /// data across a power cycle.
    #[test]
    fn re_registering_a_home_name_reopens_the_task_files() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);

        let terminal = Name::new("terminal-0").unwrap();
        service
            .register_private_invocation_view(0xA1, &terminal, Rights::ALL, None)
            .expect("first registration");
        let (slot, task) = service
            .invocation_views
            .iter()
            .enumerate()
            .find_map(|(slot, view)| {
                view.as_ref()
                    .filter(|view| view.scope == 0xA1)
                    .map(|view| (slot, view.task_session))
            })
            .expect("registered view");
        let root = service.engine.open_root(task, Rights::ALL).unwrap();
        let name = Name::new("journal").unwrap();
        block_on(
            service
                .engine
                .create(task, root.handle, &name, ObjectKind::File),
        )
        .expect("task creates");

        // Retire the view exactly as the canary reap does: session teardown,
        // then the anchor grant. The home directory is not touched.
        let view = service.invocation_views[slot].take().expect("live view");
        block_on(service.engine.detach_session(task)).expect("detach view session");
        let (_, anchor) = view.home.expect("workspace-rooted view has a home");
        block_on(service.engine.revoke(admin, anchor)).expect("revoke anchor");

        // Next boot's registration of the same terminal name.
        service
            .register_private_invocation_view(0xA2, &terminal, Rights::ALL, None)
            .expect("re-registration over the standing home");
        let revived = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xA2)
            .map(|view| view.task_session)
            .expect("revived view");
        let root = service.engine.open_root(revived, Rights::ALL).unwrap();
        service
            .engine
            .open_at(revived, root.handle, &name, Rights::READ)
            .expect("the earlier boot's file is still reachable in the revived view");
    }

    /// The diagnostic snapshot: frontend load counted from the live serving
    /// state, backend posture folded in exactly as the mount owner injected
    /// it, landed as one frozen record in the caller's window.
    #[test]
    fn the_status_snapshot_reports_frontend_load_and_backend_posture() {
        use kore_abi::svc::FsSystemStatusWire;

        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());
        let terminal = Name::new("terminal-0").unwrap();
        service
            .register_private_invocation_view(0xA1, &terminal, Rights::ALL, None)
            .expect("view registers");

        // The posture the mount owner would fold in before delegating.
        service.set_backend_status(FsBackendStatus {
            durable: true,
            route_held: true,
            channel_lost: true,
            host_memory_buffer: true,
            user_lifecycle: FsSystemStatusWire::LIFECYCLE_CURRENT,
            system_lifecycle: FsSystemStatusWire::LIFECYCLE_UNDESIGNATED,
            commits: 7,
            storage: kore_abi::monitor::FilesystemStorageSnapshot::new(
                4096,
                1 << 30,
                512 << 20,
                16 << 20,
                256 << 10,
                false,
                true,
                true,
            )
            .with_health(kore_abi::monitor::FilesystemHealthSnapshot::new(
                0, 36, 98, 10, 3, 7, 123, 2, 0,
            )),
            ..Default::default()
        });

        let session = SessionId::new(1);
        assert!(matches!(
            service.dispatch(
                session,
                &FsRequest::AttachWindow {
                    slot: 0,
                    bytes: 512
                },
                None
            ),
            FsResponse::WindowAttached { .. }
        ));
        let mut backing = [0u8; 512];
        let response = {
            let mut window = DataWindow::new(&mut backing);
            service.dispatch(
                session,
                &FsRequest::SystemStatus { window_offset: 0 },
                Some(&mut window),
            )
        };
        let FsResponse::SystemStatusReported { len } = response else {
            panic!("status refused: {response:?}");
        };
        assert_eq!(len as usize, FsSystemStatusWire::BYTES);

        let wire = crate::client::read_system_status(&backing, 0).expect("valid record");
        let flags = wire.flags.get();
        assert_ne!(flags & FsSystemStatusWire::FLAG_DURABLE, 0);
        assert_ne!(flags & FsSystemStatusWire::FLAG_ROUTE_HELD, 0);
        assert_ne!(flags & FsSystemStatusWire::FLAG_HOST_MEMORY_BUFFER, 0);
        assert_eq!(flags & FsSystemStatusWire::FLAG_DEGRADED_READ_ONLY, 0);
        assert_ne!(flags & FsSystemStatusWire::FLAG_WORKSPACE, 0);
        assert_ne!(flags & FsSystemStatusWire::FLAG_SYSTEM_VIEW, 0);
        assert_ne!(flags & FsSystemStatusWire::FLAG_CHANNEL_LOST, 0);
        assert_eq!(
            wire.user_lifecycle.get(),
            FsSystemStatusWire::LIFECYCLE_CURRENT
        );
        assert_eq!(
            wire.system_lifecycle.get(),
            FsSystemStatusWire::LIFECYCLE_UNDESIGNATED
        );
        assert_eq!(wire.commits.get(), 7);
        assert_eq!(wire.storage.physical_bytes(), 1 << 30);
        assert_eq!(wire.storage.volume_bytes(), 512 << 20);
        assert_eq!(wire.storage.occupied_bytes(), 16 << 20);
        assert_eq!(wire.storage.health().temperature_celsius(), 36);
        assert_eq!(wire.storage.health().percentage_used(), 3);
        assert_eq!(wire.storage.health().media_errors(), 0);
        // System zero, the administration session, and the registered view.
        assert!(wire.sessions.get() >= 3);
        assert_eq!(wire.views_private.get(), 1);
        assert_eq!(wire.windows.get(), 1);
        // The persisted root, `work`, and the view's home beneath it.
        assert!(wire.tree_objects.get() >= 3);
    }

    /// The store half of the archive/filesystem unification: `app` hangs
    /// beneath the persisted root (so its contents are serialized like every
    /// other fact of that tree) and the administration session holds a
    /// writable grant over it — under a display name that does not collide
    /// with the directory itself.
    #[test]
    fn the_shared_app_directory_is_writable_and_hangs_off_the_persisted_root() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_apps());

        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let display = Name::new(SYSTEM_APPS_GRANT).unwrap();
        let apps = service
            .engine
            .grants(admin)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(display))
            .expect("administration app grant stands");
        assert_eq!(apps.rights, Rights::ALL);

        // Writable through the grant, exactly as the seeding copy will be.
        let opened = service
            .engine
            .open_grant(admin, apps.grant, Rights::ALL)
            .expect("admin opens the app directory");
        assert_eq!(opened.kind, ObjectKind::Directory);
        let member = Name::new("hello").unwrap();
        block_on(
            service
                .engine
                .create(admin, opened.handle, &member, ObjectKind::File),
        )
        .expect("an executable can be written into the shared directory");

        // …and what lands there is a fact of the tree `commit_engine` walks.
        let root = service.engine.open_root(system, Rights::ALL).unwrap();
        let directory = Name::new(SYSTEM_APPS_DIR).unwrap();
        let reached = service
            .engine
            .open_at(system, root.handle, &directory, Rights::ALL)
            .expect("app hangs beneath the persisted root");
        service
            .engine
            .open_at(system, reached.handle, &member, Rights::READ)
            .expect("a seeded executable is reachable from the persisted root");
    }

    /// Recovery re-runs bring-up against a volume that already carries the
    /// directory and its executables, so a second establish must adopt the
    /// standing grant rather than re-create the directory and strand them.
    #[test]
    fn establishing_the_app_directory_twice_adopts_the_standing_grant() {
        let mut service = RamService::default();
        assert!(service.establish_admin_apps());
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let display = Name::new(SYSTEM_APPS_GRANT).unwrap();
        let first = service
            .engine
            .grants(admin)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(display))
            .expect("first establish grants")
            .grant;
        assert!(service.establish_admin_apps());
        let grants = service.engine.grants(admin).unwrap();
        let named: heapless::Vec<_, 8> = grants
            .iter()
            .filter(|grant| grant.display_name == Some(display))
            .collect();
        assert_eq!(named.len(), 1, "no rival grant is minted");
        assert_eq!(named[0].grant, first, "the standing grant is adopted");
    }

    /// The library convention's bring-up is the app directory's: `lib/`
    /// hangs off the persisted root, the admin grant over it is writable
    /// (the library seeder copies through it), and a second establish adopts
    /// the standing grant instead of minting a rival.  `pkgs/` follows suit
    /// with no view of its own.
    #[test]
    fn the_library_and_package_directories_hang_off_the_persisted_root() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_libs());
        assert!(service.establish_admin_pkgs());
        assert!(service.establish_admin_share());

        let system = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        for (grant_display, directory, member) in [
            (SYSTEM_LIBS_GRANT, SYSTEM_LIBS_DIR, "lua_demo.so"),
            (SYSTEM_PKGS_GRANT, SYSTEM_PKGS_DIR, "kore"),
            (SYSTEM_SHARE_GRANT, SYSTEM_SHARE_DIR, "wasm"),
        ] {
            let display = Name::new(grant_display).unwrap();
            let held = service
                .engine
                .grants(admin)
                .unwrap()
                .into_iter()
                .find(|grant| grant.display_name == Some(display))
                .expect("administration grant stands");
            assert_eq!(held.rights, Rights::ALL);
            let opened = service
                .engine
                .open_grant(admin, held.grant, Rights::ALL)
                .expect("admin opens the directory");
            assert_eq!(opened.kind, ObjectKind::Directory);
            let member = Name::new(member).unwrap();
            block_on(
                service
                    .engine
                    .create(admin, opened.handle, &member, ObjectKind::File),
            )
            .expect("a member can be written through the grant");

            let root = service.engine.open_root(system, Rights::ALL).unwrap();
            let directory = Name::new(directory).unwrap();
            let reached = service
                .engine
                .open_at(system, root.handle, &directory, Rights::ALL)
                .expect("the directory hangs beneath the persisted root");
            service
                .engine
                .open_at(system, reached.handle, &member, Rights::READ)
                .expect("the member is reachable from the persisted root");
        }

        // Recovery: adopt, never re-create.
        let display = Name::new(SYSTEM_LIBS_GRANT).unwrap();
        let first = service.libs_view.expect("the libs anchor is cached");
        assert!(service.establish_admin_libs());
        let named: heapless::Vec<_, 8> = service
            .engine
            .grants(admin)
            .unwrap()
            .iter()
            .filter(|grant| grant.display_name == Some(display))
            .cloned()
            .collect();
        assert_eq!(named.len(), 1, "no rival grant is minted");
        assert_eq!(named[0].grant, first, "the standing grant is adopted");
        assert_eq!(service.libs_view, Some(first));

        let first_share = service.share_view.expect("the share anchor is cached");
        assert!(service.establish_admin_share());
        assert_eq!(service.share_view, Some(first_share), "share adopts too");
    }

    /// A task view carries `share` beside `libs`: READ-only, so a task can
    /// read a shipped data member (`shared/wasm/hello.wasm`) but never write,
    /// relink, or bless one.  On session zero the name is the directory's own
    /// child, so the view install declines and the directory is reached
    /// directly — one object either way.
    #[test]
    fn a_task_view_carries_a_read_only_share_window() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_apps());
        assert!(service.establish_admin_libs());
        assert!(service.establish_admin_share());
        assert!(service.establish_admin_workspace());

        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let share = service.share_view.unwrap();
        let opened = service
            .engine
            .open_grant(admin, share, Rights::ALL)
            .unwrap();
        let wasm = Name::new("wasm").unwrap();
        let directory = block_on(service.engine.create(
            admin,
            opened.handle,
            &wasm,
            ObjectKind::Directory,
        ))
        .unwrap();
        let member = Name::new("hello.wasm").unwrap();
        block_on(
            service
                .engine
                .create(admin, directory.handle, &member, ObjectKind::File),
        )
        .unwrap();
        let _ = service.engine.close(admin, directory.handle);
        let _ = service.engine.close(admin, opened.handle);

        let session = SessionId::new(7);
        assert!(service.ensure_session(session));
        assert!(service.install_share_view(session));
        assert!(service.install_share_view(session), "idempotent");
        let grants = service.engine.grants(session).unwrap();
        let share_display = Name::new(SYSTEM_SHARE_VIEW).unwrap();
        let view: heapless::Vec<_, 8> = grants
            .iter()
            .filter(|grant| grant.display_name == Some(share_display))
            .collect();
        assert_eq!(view.len(), 1, "one `share` name, installed once");
        assert_eq!(view[0].rights, Rights::READ_ONLY);
        assert!(!view[0].rights.contains(Rights::WRITE));
        assert!(!view[0].rights.contains(Rights::BIND_EXECUTABLE));

        let through = service
            .engine
            .open_grant(session, view[0].grant, Rights::READ)
            .expect("the task opens its share view");
        let nested = service
            .engine
            .open_at(session, through.handle, &wasm, Rights::READ)
            .expect("the per-kind directory is reachable through the view");
        service
            .engine
            .open_at(session, nested.handle, &member, Rights::READ)
            .expect("the data member is reachable through the view");
        assert!(
            service
                .engine
                .open_grant(session, view[0].grant, Rights::WRITE)
                .is_err(),
            "the view refuses WRITE"
        );

        // The session rooted where the system directories themselves live
        // carries all three views like any other.  It could not while the
        // view's name was its directory's: `view_name_taken` counts a real
        // child of the destination root as taken, and the three call sites
        // discard that refusal.  On the two-store durable layout every
        // session the system store serves is rooted there, so `shared` was
        // missing from `list` and `shares` on a metal install while `apps`
        // and `libs` were present — and no single-store gate could see it.
        let rooted = SessionId::new(0);
        assert!(service.install_share_view(rooted), "shared installs");
        assert!(service.install_apps_view(rooted), "apps installs");
        assert!(service.install_libs_view(rooted), "libs installs");
    }

    /// A task view carries `libs` beside `apps`: READ-only, with neither
    /// WRITE nor BIND_EXECUTABLE, so a task can read and link a library but
    /// never write, relink, or bless one -- and `pkgs` never appears in a
    /// task view at all.
    #[test]
    fn a_task_view_carries_a_read_only_libs_window_and_no_pkgs_window() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_apps());
        assert!(service.establish_admin_libs());
        assert!(service.establish_admin_pkgs());
        assert!(service.establish_admin_workspace());

        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let libs = service.libs_view.unwrap();
        let opened = service.engine.open_grant(admin, libs, Rights::ALL).unwrap();
        let member = Name::new("lua_demo.so").unwrap();
        block_on(
            service
                .engine
                .create(admin, opened.handle, &member, ObjectKind::File),
        )
        .unwrap();
        let _ = service.engine.close(admin, opened.handle);

        let session = SessionId::new(7);
        assert!(service.ensure_session(session));
        assert!(service.install_libs_view(session));
        assert!(service.install_libs_view(session), "idempotent");
        let grants = service.engine.grants(session).unwrap();
        let libs_display = Name::new(SYSTEM_LIBS_VIEW).unwrap();
        let view: heapless::Vec<_, 8> = grants
            .iter()
            .filter(|grant| grant.display_name == Some(libs_display))
            .collect();
        assert_eq!(view.len(), 1, "one `libs` name, installed once");
        assert_eq!(view[0].rights, Rights::READ_ONLY);
        assert!(!view[0].rights.contains(Rights::WRITE));
        assert!(!view[0].rights.contains(Rights::BIND_EXECUTABLE));
        assert!(
            grants
                .iter()
                .all(|grant| grant.display_name != Some(Name::new(SYSTEM_PKGS_DIR).unwrap())),
            "pkgs is admin-only"
        );

        // Readable through the view, never writable.
        let through = service
            .engine
            .open_grant(session, view[0].grant, Rights::READ)
            .expect("the task opens its libs view");
        service
            .engine
            .open_at(session, through.handle, &member, Rights::READ)
            .expect("the library is reachable through the view");
        assert!(
            service
                .engine
                .open_grant(session, view[0].grant, Rights::WRITE)
                .is_err(),
            "the view refuses WRITE"
        );
    }

    /// #605 Stage 2: a terminal's task view carries the canonical `apps`
    /// READ-only, BIND_EXECUTABLE-free window onto the shared store object --
    /// and ONLY that name, the flat `app` view being retired -- which the terminal
    /// can read and traverse but never write, relink, or bless.
    #[test]
    fn a_terminal_view_carries_a_read_only_app_window_onto_the_shared_store() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_apps());
        assert!(service.establish_admin_workspace());

        // Seed one executable into the shared store through the admin grant.
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let apps_display = Name::new(SYSTEM_APPS_GRANT).unwrap();
        let apps = service
            .engine
            .grants(admin)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(apps_display))
            .expect("admin app grant stands")
            .grant;
        let store = service
            .engine
            .open_grant(admin, apps, Rights::ALL)
            .expect("admin opens the store");
        let member = Name::new("ls").unwrap();
        block_on(
            service
                .engine
                .create(admin, store.handle, &member, ObjectKind::File),
        )
        .expect("seed a store member");
        let _ = service.engine.close(admin, store.handle);

        // Found a terminal.
        let terminal = Name::new("alpha").unwrap();
        service
            .register_private_invocation_view(0xA1, &terminal, Rights::ALL, None)
            .expect("terminal view registers");
        let task = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xA1)
            .map(|view| view.task_session)
            .expect("registered terminal view");

        // The terminal carries canonical `apps` and NOTHING ELSE naming the store:
        // the flat `app` view is retired, so one object is shown under one
        // name. `apps` is exactly READ-only — no WRITE, MUTATE_NAMESPACE,
        // SHARE, or BIND_EXECUTABLE.
        let apps_display = Name::new(SYSTEM_APPS_VIEW).unwrap();
        let legacy_display = Name::new(SYSTEM_APPS_DIR).unwrap();
        let apps = service
            .engine
            .grants(task)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(apps_display))
            .expect("the terminal carries the canonical apps view");
        assert!(
            service
                .engine
                .grants(task)
                .unwrap()
                .into_iter()
                .all(|grant| grant.display_name != Some(legacy_display)),
            "the retired flat `app` view must not be granted to a terminal — one \
             store object under two names reads as a stale projection"
        );
        assert_eq!(
            apps.rights,
            Rights::READ_ONLY,
            "the app view is exactly READ"
        );
        assert!(!apps.rights.contains(Rights::WRITE));
        assert!(!apps.rights.contains(Rights::MUTATE_NAMESPACE));
        assert!(!apps.rights.contains(Rights::SHARE));
        assert!(!apps.rights.contains(Rights::BIND_EXECUTABLE));

        // The seeded member is reachable read-only through the one view.
        let opened = service
            .engine
            .open_grant(task, apps.grant, Rights::READ)
            .expect("terminal opens its canonical apps view");
        service
            .engine
            .open_at(task, opened.handle, &member, Rights::READ)
            .expect("the seeded executable is visible read-only through apps/");

        // But the terminal cannot mutate the store: creating a name refuses.
        let intruder = Name::new("intruder").unwrap();
        assert!(
            block_on(
                service
                    .engine
                    .create(task, opened.handle, &intruder, ObjectKind::File)
            )
            .is_err(),
            "a terminal cannot write into the shared store through its apps view"
        );
    }

    #[test]
    fn dagfs_and_posix_app_projections_are_dag_edges_to_one_file() {
        let mut service = RamService::default();
        assert!(service.establish_admin_apps());
        assert!(service.establish_admin_workspace());

        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let apps_grant = service
            .engine
            .grants(admin)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(Name::new(SYSTEM_APPS_GRANT).unwrap()))
            .expect("admin owns the writable app store")
            .grant;
        let apps = service
            .engine
            .open_grant(admin, apps_grant, Rights::ALL)
            .expect("admin opens app store");
        let ls = block_on(service.engine.create(
            admin,
            apps.handle,
            &Name::new("ls").unwrap(),
            ObjectKind::File,
        ))
        .expect("seed one image object");
        let dagfs = block_on(service.engine.create(
            admin,
            apps.handle,
            &Name::new("dagfs").unwrap(),
            ObjectKind::Directory,
        ))
        .expect("create native app family");
        let compat = block_on(service.engine.create(
            admin,
            apps.handle,
            &Name::new("compat").unwrap(),
            ObjectKind::Directory,
        ))
        .expect("create compatibility family");
        let posix = block_on(service.engine.create(
            admin,
            compat.handle,
            &Name::new("posix").unwrap(),
            ObjectKind::Directory,
        ))
        .expect("create POSIX projection family");
        service
            .engine
            .link(admin, dagfs.handle, &Name::new("list").unwrap(), ls.handle)
            .expect("native projection links the source object");
        service
            .engine
            .link(admin, posix.handle, &Name::new("ls").unwrap(), ls.handle)
            .expect("POSIX projection links the same source object");

        let terminal = Name::new("alpha").unwrap();
        service
            .register_private_invocation_view(0xB1, &terminal, Rights::ALL, None)
            .expect("terminal view registers");
        let task = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xB1)
            .expect("terminal view stands")
            .task_session;
        let canonical = service
            .engine
            .grants(task)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(Name::new(SYSTEM_APPS_VIEW).unwrap()))
            .expect("canonical apps view is present");
        let apps = service
            .engine
            .open_grant(task, canonical.grant, Rights::READ)
            .expect("terminal opens canonical apps view");
        let dagfs = service
            .engine
            .open_at(
                task,
                apps.handle,
                &Name::new("dagfs").unwrap(),
                Rights::READ,
            )
            .expect("native directory is visible");
        let compat = service
            .engine
            .open_at(
                task,
                apps.handle,
                &Name::new("compat").unwrap(),
                Rights::READ,
            )
            .expect("compatibility directory is visible");
        let posix = service
            .engine
            .open_at(
                task,
                compat.handle,
                &Name::new("posix").unwrap(),
                Rights::READ,
            )
            .expect("POSIX directory is visible");
        let native = service
            .engine
            .open_at(
                task,
                dagfs.handle,
                &Name::new("list").unwrap(),
                Rights::READ,
            )
            .expect("native projection is visible");
        let posix = service
            .engine
            .open_at(task, posix.handle, &Name::new("ls").unwrap(), Rights::READ)
            .expect("POSIX projection is visible");
        assert_eq!(native.stable_object, posix.stable_object);
        assert_eq!(native.stable_object, ls.stable_object);
    }

    /// A dead writer's staged replacement is released on the service's next
    /// dispatch, never inside the synchronous peer-death hook. A release the
    /// provider refuses stays pending — custody retained — while the dead
    /// peer's slot is already free for the next admission, and the old bytes
    /// are what that next peer reads either way.
    #[test]
    fn a_dead_writers_stage_is_released_on_the_next_dispatch_not_in_the_hook() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        service
            .register_private_invocation_view(
                0xD0,
                &Name::new("terminal-0").unwrap(),
                Rights::ALL,
                None,
            )
            .unwrap();
        let peer = admit_view(&mut service, 0x106, 0xD0, Rights::ALL);
        let root = service.engine.open_root(peer, Rights::ALL).unwrap();
        let file = crate::block_on(service.engine.create(
            peer,
            root.handle,
            &Name::new("saved").unwrap(),
            crate::ObjectKind::File,
        ))
        .unwrap();
        crate::block_on(
            service
                .engine
                .write_at(peer, file.handle, 0, b"old", crate::ANY_REVISION),
        )
        .unwrap();
        assert!(matches!(
            service.dispatch(
                peer,
                &FsRequest::BeginReplace {
                    handle: file.handle,
                    length: 6,
                    expected: crate::ANY_REVISION
                },
                None
            ),
            FsResponse::Done
        ));
        service.dispatch(peer, &FsRequest::AttachWindow { slot: 0, bytes: 3 }, None);
        let mut backing = *b"new";
        let mut window = crate::window::DataWindow::new(&mut backing);
        assert!(matches!(
            service.dispatch(
                peer,
                &FsRequest::WriteReplacement {
                    handle: file.handle,
                    offset: 0,
                    window_offset: 0,
                    length: 3
                },
                Some(&mut window)
            ),
            FsResponse::Done
        ));
        service
            .engine
            .provider()
            .inject(crate::provider::Fault::Release(
                crate::error::ProviderError::Media,
            ));
        assert!(
            service.peer_closed(0x106, PeerClose::Died),
            "the hook never awaits the release; the stage leaves with its peer"
        );
        assert!(
            service
                .slots
                .get(peer.index() as usize)
                .is_none_or(|state| state.replacement.is_none())
        );
        assert_eq!(service.pending_aborts.len(), 1);
        // The dead peer's slot is free at once — the stage no longer lives in
        // it — and the next dispatch tries the release. The armed fault
        // refuses that one: custody is retained for the dispatch after.
        let next = admit_view(&mut service, 0x206, 0xD0, Rights::ALL);
        let _ = service.dispatch(next, &FsRequest::DeferCommits { hold: false }, None);
        assert_eq!(
            service.pending_aborts.len(),
            1,
            "a refused release stays pending"
        );
        let _ = service.dispatch(next, &FsRequest::DeferCommits { hold: false }, None);
        assert!(
            service.pending_aborts.is_empty(),
            "released once the provider accepts"
        );
        let root = service.engine.open_root(next, Rights::ALL).unwrap();
        let file = service
            .engine
            .open_at(next, root.handle, &Name::new("saved").unwrap(), Rights::ALL)
            .unwrap();
        let mut bytes = [0; 3];
        crate::block_on(service.engine.read(next, file.handle, 0, &mut bytes)).unwrap();
        assert_eq!(&bytes, b"old");
    }

    /// The editor's open by identity, through the wire: a file's stated
    /// identity, located from the view root, lands one record the client's
    /// locate reader lifts — the file itself, with its length. The parents
    /// reader refuses that same record (a parent is a directory with no
    /// length), which is what once made every located file `Range`.
    #[test]
    fn a_located_file_reads_back_through_the_window() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        service
            .register_private_invocation_view(
                0xD0,
                &Name::new("terminal-0").unwrap(),
                Rights::ALL,
                None,
            )
            .unwrap();
        let peer = admit_view(&mut service, 0x106, 0xD0, Rights::ALL);
        let root = service.engine.open_root(peer, Rights::ALL).unwrap();
        let file = crate::block_on(service.engine.create(
            peer,
            root.handle,
            &Name::new("editor-test").unwrap(),
            crate::ObjectKind::File,
        ))
        .unwrap();
        crate::block_on(
            service
                .engine
                .write_at(peer, file.handle, 0, b"before", crate::ANY_REVISION),
        )
        .unwrap();
        let stable = service.engine.stat(peer, file.handle).unwrap().stable_object;

        const BYTES: usize = FsSearchHitWire::BYTES;
        service.dispatch(peer, &FsRequest::AttachWindow { slot: 0, bytes: BYTES as u32 }, None);
        let mut backing = [0u8; BYTES];
        let response = {
            let mut window = crate::window::DataWindow::new(&mut backing);
            service.dispatch(
                peer,
                &FsRequest::Locate {
                    handle: root.handle,
                    stable,
                    window_offset: 0,
                },
                Some(&mut window),
            )
        };
        assert_eq!(response, FsResponse::Listed { entries: 1, next: None });
        let located = crate::client::read_located(&backing, 0).expect("the record lifts");
        assert_eq!(located.stable_object, stable);
        assert_eq!(located.kind, crate::ObjectKind::File);
        assert_eq!(located.len, 6);
        assert_eq!(located.path.len(), 1);
        assert_eq!(located.path[0].as_bytes(), b"editor-test");
        assert_eq!(crate::client::read_parent(&backing, 0, 0), Err(FsError::Range));
    }

    /// The deferral is per session: a bulk writer holds only its own
    /// commits, and a second client's durability is untouched by it.
    #[test]
    fn deferring_commits_binds_one_session_and_ends_with_its_peer() {
        let mut service = RamService::default();
        let writer = SessionId::new(1);
        // Not slot 2: `PeerId`'s slot is the arrival's low bits, so 0x106
        // below lands there and the admission must find it free.
        let other = SessionId::new(3);
        assert!(service.ensure_session(writer));
        assert!(service.ensure_session(other));

        assert!(matches!(
            service.dispatch(writer, &FsRequest::DeferCommits { hold: true }, None),
            FsResponse::Done
        ));
        assert!(service.is_deferring(writer));
        assert!(
            !service.is_deferring(other),
            "one client's bulk write must not defer another's durability"
        );

        assert!(matches!(
            service.dispatch(writer, &FsRequest::DeferCommits { hold: false }, None),
            FsResponse::Done
        ));
        assert!(!service.is_deferring(writer));

        // A client that dies inside the window leaves no held state for the
        // next occupant of its slot to inherit.
        assert!(service.establish_admin_source());
        let terminal = Name::new("terminal-0").unwrap();
        service
            .register_private_invocation_view(0xD0, &terminal, Rights::ALL, None)
            .expect("task view registers");
        let peer = admit_view(&mut service, 0x106, 0xD0, Rights::ALL);
        assert!(matches!(
            service.dispatch(peer, &FsRequest::DeferCommits { hold: true }, None),
            FsResponse::Done
        ));
        assert!(service.is_deferring(peer));
        assert!(service.peer_closed(0x106, PeerClose::Orderly));
        assert!(
            !service.is_deferring(peer),
            "a dead writer's window must not be inherited by its slot's next occupant"
        );
    }

    /// A dead peer's authority goes in the hook; its subtree goes on the next
    /// async dispatch, never inside the hook. The hook is synchronous, and a
    /// reclamation run inside it was a nested executor that could not let
    /// the in-flight request holding a claim release it.
    #[test]
    fn a_dead_peers_subtree_is_reclaimed_on_the_next_dispatch_not_in_the_hook() {
        let mut service = RamService::default();
        let session = SessionId::new(2);
        assert!(service.ensure_session(session));
        service.slots.claim(2).unwrap().via = Some(0x102);
        let root = service.engine.open_root(session, Rights::ALL).unwrap();
        let before = service.engine.object_count();
        let file = block_on(service.engine.create(
            session,
            root.handle,
            &Name::new("stranded").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        assert_eq!(service.engine.object_count(), before + 1);

        assert!(service.peer_closed(0x102, PeerClose::Orderly));
        // Authority is gone at once...
        assert!(service.engine.open_root(session, Rights::READ).is_err());
        // ...the bytes are not, until a dispatch runs on the service's own
        // executor.
        assert_eq!(service.engine.object_count(), before + 1);
        assert_eq!(service.pending_reclaims.len(), 1);
        let _ = service.dispatch(
            session,
            &FsRequest::Close {
                handle: file.handle,
            },
            None,
        );
        assert!(service.pending_reclaims.is_empty());
        assert!(service.engine.object_count() <= before);
    }

    /// One home, one live owner: a second live registration naming the same
    /// home is refused whole, so two tasks cannot be silently fused into a
    /// shared namespace by a name collision.
    #[test]
    fn a_live_home_name_cannot_be_registered_twice() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        let terminal = Name::new("terminal-0").unwrap();
        service
            .register_private_invocation_view(0xA1, &terminal, Rights::ALL, None)
            .expect("first registration");
        assert_eq!(
            service
                .register_private_invocation_view(0xB1, &terminal, Rights::ALL, None)
                .unwrap_err(),
            FsError::AlreadyExists
        );
    }

    /// The command lifecycle over the workspace-rooted view: one command
    /// creates, exits (peer teardown), and the next command of the same
    /// task must still see the creation — the exact shell sequence
    /// (`mkdir` then `touch`) the boot gate drives.
    #[test]
    fn a_later_command_sees_an_earlier_commands_workspace_creations() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        // The real registration path: the stable task view roots in its own
        // durable home beneath the workspace.
        let terminal = Name::new("terminal-0").unwrap();
        service
            .register_private_invocation_view(0xC0, &terminal, Rights::ALL, None)
            .expect("task view registers over its durable home");

        // Command one: mkdir durable; then the process exits.
        let first = admit_view(&mut service, 0x106, 0xC0, Rights::ALL);
        let root = service.engine.open_root(first, Rights::ALL).unwrap();
        let durable = Name::new("durable").unwrap();
        block_on(
            service
                .engine
                .create(first, root.handle, &durable, ObjectKind::Directory),
        )
        .expect("mkdir durable");
        assert!(service.peer_closed(0x106, PeerClose::Orderly));

        // Command two, same task: touch durable/data.
        let second = admit_view(&mut service, 0x206, 0xC0, Rights::ALL);
        let root = service.engine.open_root(second, Rights::ALL).unwrap();
        let reached = service
            .engine
            .open_at(second, root.handle, &durable, Rights::ALL)
            .expect("the second command still sees durable/");
        let data = Name::new("data").unwrap();
        block_on(
            service
                .engine
                .create(second, reached.handle, &data, ObjectKind::File),
        )
        .expect("touch durable/data");
    }

    /// The Python worker's HOME face can edit its seat's durable files, but
    /// neither climb into another seat's home nor write a system preopen.
    /// The service enforces this on the routes; Python source does not need a
    /// path blacklist to make a read-only grant stay read-only.
    #[test]
    fn writable_script_home_keeps_system_grants_and_other_homes_outside() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_share());
        assert!(service.establish_admin_workspace());
        let home = Name::new("terminal-0").unwrap();
        let other = Name::new("terminal-1").unwrap();
        service
            .register_private_invocation_view(0xA1, &home, Rights::HOME, None)
            .unwrap();
        service
            .register_private_invocation_view(0xB1, &other, Rights::HOME, None)
            .unwrap();
        let peer = admit_view(&mut service, 0x106, 0xA1, Rights::HOME);
        let root = service.engine.open_root(peer, Rights::HOME).unwrap();
        assert!(matches!(
            service.engine.open_root(peer, Rights::ALL),
            Err(FsError::AccessDenied)
        ));
        assert!(matches!(
            service.engine.open_at(peer, root.handle, &other, Rights::READ),
            Err(FsError::NotFound)
        ));
        assert!(Name::new("..").is_err(), "a containment edge cannot climb out");

        let note = Name::new("note.txt").unwrap();
        let file = block_on(service.engine.create(peer, root.handle, &note, ObjectKind::File))
            .expect("create in the writable home");
        block_on(service.engine.write_at(peer, file.handle, 0, b"old", crate::ANY_REVISION))
            .expect("write in the home");
        block_on(service.engine.write_at(peer, file.handle, 0, b"new", crate::ANY_REVISION))
            .expect("edit in the home");
        service.engine.close(peer, file.handle).unwrap();
        assert!(service.peer_closed(0x106, PeerClose::Orderly));

        // Each Python call starts a fresh worker. A later call must see the
        // prior call's edits through the same delegated home.
        let later = admit_view(&mut service, 0x206, 0xA1, Rights::HOME);
        let later_root = service.engine.open_root(later, Rights::HOME).unwrap();
        let file = service
            .engine
            .open_at(later, later_root.handle, &note, Rights::READ)
            .expect("later worker reopens the durable home file");
        let mut bytes = [0; 3];
        assert_eq!(block_on(service.engine.read(later, file.handle, 0, &mut bytes)), Ok(3));
        assert_eq!(&bytes, b"new");
        service.engine.close(later, file.handle).unwrap();
        block_on(service.engine.unlink(later, later_root.handle, &note))
            .expect("delete in the home");
        assert!(matches!(
            service.engine.open_at(later, later_root.handle, &note, Rights::READ),
            Err(FsError::NotFound)
        ));

        let owner = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xA1)
            .unwrap()
            .task_session;
        let share = service
            .engine
            .grants(owner)
            .unwrap()
            .into_iter()
            .find(|grant| grant.display_name == Some(Name::new(SYSTEM_SHARE_VIEW).unwrap()))
            .expect("the system share remains a named grant");
        assert_eq!(share.rights, Rights::READ_ONLY);
        assert!(matches!(
            service.engine.open_task_grant(later, owner, share.grant, Rights::WRITE),
            Err(FsError::AccessDenied)
        ));
        let read_only = service
            .engine
            .open_task_grant(later, owner, share.grant, Rights::READ)
            .unwrap();
        assert!(matches!(
            block_on(service.engine.create(later, read_only.handle, &note, ObjectKind::File)),
            Err(FsError::AccessDenied)
        ));
    }

    /// Recovery calls this on a volume that already carries the directory, so
    /// a second call must adopt the standing grant rather than mint a rival
    /// (or refuse on the taken display name) — otherwise the first reboot
    /// after any successful boot would strand the user's data.
    #[test]
    fn establishing_the_workspace_twice_adopts_the_standing_grant() {
        let mut service = RamService::default();
        assert!(service.establish_admin_workspace());
        let first = service
            .workspace_view
            .expect("anchor after first establish");
        assert!(service.establish_admin_workspace());
        assert_eq!(
            service.workspace_view,
            Some(first),
            "the second establish adopts the standing anchor"
        );
    }

    /// The read-only program source and the writable workspace must stay
    /// distinct objects: exec loads executable images through `sys`, and a
    /// fault in ordinary user-file handling must not be able to reach them.
    #[test]
    fn the_workspace_is_not_the_system_source_object() {
        let mut service = RamService::default();
        assert!(service.establish_admin_source());
        assert!(service.establish_admin_workspace());

        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let source = Name::new(SYSTEM_SOURCE_GRANT).unwrap();
        let grants = service.engine.grants(admin).expect("admin grants listing");
        let source = grants
            .iter()
            .find(|grant| grant.display_name == Some(source))
            .expect("system source grant");
        let workspace = service.workspace_view.expect("workspace anchor stands");
        assert_eq!(
            source.rights,
            Rights::READ
                .union(Rights::SHARE)
                .union(Rights::BIND_EXECUTABLE)
        );

        let workspace_opened = service
            .engine
            .open_grant(admin, workspace, Rights::ALL)
            .expect("admin opens the workspace");
        // The source root cannot name itself through the workspace's edge:
        // the workspace is a child of it, never an alias for it.
        let opened = service
            .engine
            .open_grant(admin, source.grant, Rights::READ)
            .expect("admin opens the read-only source");
        assert_ne!(
            opened.stable_object, workspace_opened.stable_object,
            "the workspace and the source are distinct objects"
        );
        let directory = Name::new(SYSTEM_WORKSPACE_DIR).unwrap();
        let nested = service
            .engine
            .open_at(admin, opened.handle, &directory, Rights::READ)
            .expect("the workspace is reachable beneath the source root");
        assert_eq!(nested.kind, ObjectKind::Directory);
    }

    #[test]
    fn sibling_invocation_lineages_keep_independent_native_rights_ceilings() {
        let mut service = RamService::default();
        assert!(service.ensure_session(SessionId::new(0)));
        service
            .register_system_invocation_view(0xA, Rights::ALL)
            .expect("full-rights sibling view");
        service
            .register_system_invocation_view(0xB, Rights::READ_ONLY.union(Rights::SHARE))
            .expect("delegable read-only sibling view");

        let full = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xA)
            .expect("full view registered");
        let read_only = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xB)
            .expect("read-only view registered");
        assert_eq!(full.rights, Rights::ALL);
        assert_eq!(read_only.rights, Rights::READ_ONLY.union(Rights::SHARE));
        assert!(!read_only.rights.contains(Rights::WRITE));
        assert!(!read_only.rights.contains(Rights::MUTATE_NAMESPACE));
    }

    #[test]
    fn private_task_views_start_isolated_and_share_only_by_stable_grant() {
        let mut service = RamService::default();
        let task_a = SessionId::new(FIRST_TASK_VIEW_SESSION as u16);
        let task_b = SessionId::new(FIRST_TASK_VIEW_SESSION as u16 + 1);
        install_private_task_view(&mut service, 0xA0, task_a, Rights::ALL);
        install_private_task_view(&mut service, 0xB0, task_b, Rights::ALL);

        let command_a = admit_view(&mut service, 0x102, 0xA0, Rights::ALL);
        let root_a = service.engine.open_root(command_a, Rights::ALL).unwrap();
        let shared = block_on(service.engine.create(
            command_a,
            root_a.handle,
            &Name::new("shared").unwrap(),
            ObjectKind::Directory,
        ))
        .unwrap();
        let file = block_on(service.engine.create(
            command_a,
            shared.handle,
            &Name::new("value").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        block_on(
            service
                .engine
                .write_at(command_a, file.handle, 0, b"one", 0),
        )
        .unwrap();
        assert!(service.peer_closed(0x102, PeerClose::Orderly));

        let command_b = admit_view(&mut service, 0x103, 0xB0, Rights::ALL);
        let root_b = service.engine.open_root(command_b, Rights::ALL).unwrap();
        assert_eq!(
            service.engine.open_at(
                command_b,
                root_b.handle,
                &Name::new("shared").unwrap(),
                Rights::READ,
            ),
            Err(FsError::NotFound),
            "a sibling private root must not reveal another task's names"
        );
        assert!(service.peer_closed(0x103, PeerClose::Orderly));
        let authority_baseline = service.engine.authority_edge_count();

        let command_a = admit_view(&mut service, 0x202, 0xA0, Rights::ALL);
        let root_a = service.engine.open_root(command_a, Rights::ALL).unwrap();
        let shared = service
            .engine
            .open_at(
                command_a,
                root_a.handle,
                &Name::new("shared").unwrap(),
                Rights::ALL,
            )
            .unwrap();
        let grant = service
            .engine
            .grant_between_task_views(
                command_a,
                shared.handle,
                task_a,
                task_b,
                Rights::READ,
                Some(Name::new("from-a").unwrap()),
            )
            .unwrap();
        assert!(service.peer_closed(0x202, PeerClose::Orderly));

        let command_b = admit_view(&mut service, 0x203, 0xB0, Rights::ALL);
        let opened = service
            .engine
            .open_task_grant(command_b, task_b, grant, Rights::READ)
            .unwrap();
        let value = service
            .engine
            .open_at(
                command_b,
                opened.handle,
                &Name::new("value").unwrap(),
                Rights::READ,
            )
            .unwrap();
        let mut bytes = [0u8; 8];
        assert_eq!(
            block_on(service.engine.read(command_b, value.handle, 0, &mut bytes)),
            Ok(3)
        );
        assert_eq!(&bytes[..3], b"one");

        // The attachment is a live route to the same object, not a copied
        // snapshot. A later command in A's independent task view changes the
        // object and B's already-open attenuated route observes that revision.
        let command_a = admit_view(&mut service, 0x302, 0xA0, Rights::ALL);
        let root_a = service.engine.open_root(command_a, Rights::ALL).unwrap();
        let shared_a = service
            .engine
            .open_at(
                command_a,
                root_a.handle,
                &Name::new("shared").unwrap(),
                Rights::ALL,
            )
            .unwrap();
        let value_a = service
            .engine
            .open_at(
                command_a,
                shared_a.handle,
                &Name::new("value").unwrap(),
                Rights::ALL,
            )
            .unwrap();
        block_on(
            service
                .engine
                .write_at(command_a, value_a.handle, 0, b"two", 0),
        )
        .unwrap();
        assert!(service.peer_closed(0x302, PeerClose::Orderly));
        bytes.fill(0);
        assert_eq!(
            block_on(service.engine.read(command_b, value.handle, 0, &mut bytes)),
            Ok(3)
        );
        assert_eq!(&bytes[..3], b"two");
        assert_eq!(
            service
                .engine
                .open_task_grant(command_b, task_b, grant, Rights::WRITE),
            Err(FsError::AccessDenied)
        );
        assert!(service.peer_closed(0x203, PeerClose::Orderly));

        block_on(service.engine.revoke(task_b, grant)).unwrap();
        let command_b = admit_view(&mut service, 0x303, 0xB0, Rights::ALL);
        assert_eq!(
            service
                .engine
                .open_task_grant(command_b, task_b, grant, Rights::READ),
            Err(FsError::AccessLost)
        );
        assert!(service.peer_closed(0x303, PeerClose::Orderly));
        assert_eq!(
            service.engine.authority_edge_count(),
            authority_baseline,
            "revoking the stable attachment and closing its transient projections returns the authority edge to baseline"
        );
    }

    /// The faces programs read files through are bounded by the slots the
    /// serve host can name, not by its bootstrap table: a face in every
    /// slot roots under one view at once, each in storage of its own, and
    /// every slot takes its next occupant after the last one closes.
    #[test]
    fn a_face_in_every_slot_the_host_can_grow_reads_files_at_once() {
        let mut service = RamService::default();
        let system = SessionId::new(0);
        assert!(service.ensure_session(system));
        let system_root = service.engine.open_root(system, Rights::ALL).unwrap();
        block_on(service.engine.create(
            system,
            system_root.handle,
            &Name::new("visible").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        service
            .register_system_invocation_view(0xA, Rights::READ)
            .unwrap();
        let edges = service.engine.authority_edge_count();

        let first = usize::from(crate::service::contract::ADMIN_VIA) + 1;
        let slots = first..crate::bounds::PAIR_SESSIONS;
        assert!(slots.len() > kore_rt::svc::MAX_SERVE_PAIRS);
        for generation in 1..=2u16 {
            for slot in slots.clone() {
                let peer = PeerId::from_arrival((generation << 8) | slot as u16).unwrap();
                assert!(service.peer_admitted(Ok(peer)), "slot {slot}");
                assert_eq!(
                    service.take_peer_admission(),
                    FsResponse::PeerAdmitted { peer }
                );
                service
                    .apply_invocation_view(0xA, peer, Rights::READ)
                    .unwrap();
            }
            for slot in slots.clone() {
                let via = (generation << 8) | slot as u16;
                let session = service.session_for_via(via).unwrap();
                assert_eq!(session, SessionId::new(slot as u16));
                let root = service.engine.open_root(session, Rights::READ).unwrap();
                service
                    .engine
                    .open_at(
                        session,
                        root.handle,
                        &Name::new("visible").unwrap(),
                        Rights::READ,
                    )
                    .unwrap();
            }
            assert_eq!(
                service.engine.authority_edge_count(),
                edges + slots.len(),
                "one delegated root per face"
            );
            for slot in slots.clone() {
                assert!(service.peer_closed((generation << 8) | slot as u16, PeerClose::Orderly));
            }
            assert_eq!(service.engine.authority_edge_count(), edges);
            // The previous generation names nothing.
            assert_eq!(
                service.session_for_via((generation << 8) | first as u16),
                None
            );
        }
    }

    /// A launch can fail after its face was admitted and before its view
    /// was applied. The peer's session is still parked in its attach
    /// transition then; closing it gives the slot back, and the slot's next
    /// occupant is admitted and rooted as usual.
    #[test]
    fn a_peer_closed_before_its_view_gives_its_slot_back() {
        let mut service = RamService::default();
        let system = SessionId::new(0);
        assert!(service.ensure_session(system));
        service
            .register_system_invocation_view(0xA, Rights::READ)
            .unwrap();
        let parked = PeerId::from_arrival(0x105).unwrap();
        assert!(service.peer_admitted(Ok(parked)));
        assert!(service.peer_closed(parked.arrival(), PeerClose::Died));
        assert!(service.peer_is_retired(parked));

        let next = PeerId::from_arrival(0x205).unwrap();
        assert!(service.peer_admitted(Ok(next)));
        service
            .apply_invocation_view(0xA, next, Rights::READ)
            .unwrap();
        let session = SessionId::new(next.slot() as u16);
        assert!(service.engine.open_root(session, Rights::READ).is_ok());
        assert!(service.peer_closed(next.arrival(), PeerClose::Orderly));
    }

    #[test]
    fn narrowed_invocation_view_outlives_the_process_that_bound_it() {
        let mut service = RamService::default();
        let system = SessionId::new(0);
        assert!(service.ensure_session(system));
        let system_root = service.engine.open_root(system, Rights::ALL).unwrap();
        let directory = block_on(service.engine.create(
            system,
            system_root.handle,
            &Name::new("shared").unwrap(),
            ObjectKind::Directory,
        ))
        .unwrap();
        block_on(service.engine.create(
            system,
            directory.handle,
            &Name::new("visible").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();

        let delegated = Rights::READ.union(Rights::SHARE);
        service
            .register_system_invocation_view(0xA, delegated)
            .unwrap();
        let first = PeerId::from_arrival(0x102).unwrap();
        assert!(service.peer_admitted(Ok(first)));
        service
            .apply_invocation_view(0xA, first, delegated)
            .unwrap();
        let first_session = SessionId::new(first.slot() as u16);
        let first_root = service.engine.open_root(first_session, delegated).unwrap();
        let narrowed = service
            .engine
            .open_at(
                first_session,
                first_root.handle,
                &Name::new("shared").unwrap(),
                delegated,
            )
            .unwrap();
        service
            .register_invocation_view(first_session, narrowed.handle, delegated, 0xB)
            .unwrap();

        assert!(service.peer_closed(0x102, PeerClose::Orderly));
        let second = PeerId::from_arrival(0x103).unwrap();
        assert!(service.peer_admitted(Ok(second)));
        service
            .apply_invocation_view(0xB, second, delegated)
            .expect("the caller-rooted cwd survives its binder task");
        let second_session = SessionId::new(second.slot() as u16);
        let second_root = service
            .engine
            .open_root(second_session, Rights::READ)
            .unwrap();
        assert!(
            service
                .engine
                .open_at(
                    second_session,
                    second_root.handle,
                    &Name::new("visible").unwrap(),
                    Rights::READ,
                )
                .is_ok()
        );
        assert!(service.peer_closed(0x103, PeerClose::Orderly));
        let pinned = service
            .invocation_views
            .iter()
            .flatten()
            .find(|view| view.scope == 0xB)
            .map(|view| (view.task_session, view.grant))
            .unwrap();
        block_on(service.engine.revoke(pinned.0, pinned.1))
            .expect("the stable pin owner can retire the narrowed view exactly once");
    }

    #[test]
    fn recycled_process_sessions_share_only_explicit_native_grants() {
        let mut service = RamService::default();
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let process = SessionId::new(2);
        assert!(service.ensure_session(admin));
        assert!(service.ensure_session(process));

        let workspace = service
            .engine
            .open_root(admin, Rights::ALL)
            .expect("admin workspace root");
        let display = NativeView::Data.name();
        let first = service
            .engine
            .grant(admin, workspace.handle, process, Rights::ALL, Some(display))
            .expect("first process native grant");
        let first_root = service
            .engine
            .open_grant(process, first, Rights::ALL)
            .expect("first process opens workspace");
        block_on(service.engine.create(
            process,
            first_root.handle,
            &Name::new("kept").unwrap(),
            ObjectKind::File,
        ))
        .expect("first process mutates workspace");

        block_on(service.engine.detach_session(process)).expect("first process exits");
        assert!(service.ensure_session(process));
        let replacement_private = service
            .engine
            .open_root(process, Rights::READ)
            .expect("replacement private root");
        assert_eq!(
            service.engine.open_at(
                process,
                replacement_private.handle,
                &Name::new("kept").unwrap(),
                Rights::READ,
            ),
            Err(FsError::NotFound),
            "recycling the pair must still create an isolated private root"
        );
        let second = service
            .engine
            .grant(admin, workspace.handle, process, Rights::ALL, Some(display))
            .expect("replacement process native grant");
        let second_root = service
            .engine
            .open_grant(process, second, Rights::READ)
            .expect("replacement process opens workspace");
        assert!(
            service
                .engine
                .open_at(
                    process,
                    second_root.handle,
                    &Name::new("kept").unwrap(),
                    Rights::READ,
                )
                .is_ok(),
            "retiring the ephemeral recipient must not retire the issuer's route"
        );
    }

    #[test]
    fn native_view_vocabulary_is_complete_and_collision_free() {
        let expected = ["app", "data", "config", "tmp", "shared"];
        for (index, view) in NativeView::ALL.into_iter().enumerate() {
            assert_eq!(view.as_str(), expected[index]);
            assert_eq!(view.name().as_str(), expected[index]);
            assert!(
                NativeView::ALL[..index]
                    .iter()
                    .all(|prior| prior.as_str() != view.as_str()),
                "native attachment names must be unique"
            );
        }
    }

    #[test]
    fn retirement_fence_is_exact_across_slot_reuse() {
        let mut service = RamService::default();
        let first = PeerId::from_arrival(0x101).expect("valid first generation");
        let replacement = PeerId::from_arrival(0x201).expect("valid replacement generation");

        service.slots.claim(1).unwrap().via = Some(first.arrival());
        assert!(!service.peer_is_retired(first));
        assert!(service.peer_is_retired(replacement));

        service.slots.claim(1).unwrap().via = Some(replacement.arrival());
        assert!(service.peer_is_retired(first));
        assert!(!service.peer_is_retired(replacement));
    }
}

impl Default for RamService {
    fn default() -> Self {
        Self::new(Engine::new(RamProvider::new()))
    }
}

/// A route as the wire spells it: names joined by `/`, no leading slash,
/// into the search-hit record's fixed path field.  The field is proven wide
/// enough for the deepest legal route at compile time (`bounds`); this keeps
/// a drift there a refusal, never an index past the array.
fn spell_path(names: &[Name]) -> FsResult<([u8; FS_SEARCH_PATH_BYTES], usize)> {
    let mut path = [0u8; FS_SEARCH_PATH_BYTES];
    let mut len = 0usize;
    for (index, name) in names.iter().enumerate() {
        if index != 0 {
            if len >= path.len() {
                return Err(FsError::Range);
            }
            path[len] = b'/';
            len += 1;
        }
        let bytes = name.as_bytes();
        let end = len.checked_add(bytes.len()).ok_or(FsError::Range)?;
        if end > path.len() {
            return Err(FsError::Range);
        }
        path[len..end].copy_from_slice(bytes);
        len = end;
    }
    Ok((path, len))
}
