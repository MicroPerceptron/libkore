//! # kore-svc-dagfs — the filesystem service
//!
//! A **capability-confined userland filesystem over a named containment
//! graph**. The kernel learns
//! no filenames, directories, paths, or filesystem policy; it supplies
//! isolation, capability transfer, shared memory, and scheduling, and this
//! service owns everything else.
//!
//! ## The two edge kinds
//!
//! The graph has exactly two kinds of edge, and keeping them distinct is the
//! design:
//!
//! - a **containment edge** places a child object beneath a parent directory
//!   under a name ([`namespace`]) — these and only these form the namespace
//!   DAG; and
//! - an **authority edge** lets a session enter a file or subtree with
//!   bounded rights ([`authority`]) — a private root, or a direct grant.
//!
//! More than one containment edge may name the same file or directory object.
//! Diamonds are lawful; removing one route leaves every other route alone,
//! while bounded reachability checks refuse self-links and back-edges.
//!
//! ## Handles are not authority
//!
//! An object ID or open handle conveys nothing by itself.  Every open handle
//! is bound to the session that opened it, the authority edge it entered
//! through, the containment chain it followed, the rights attenuated along
//! that route, and the *generations* of every edge involved — an
//! [`authority::EdgeProof`].  Each operation revalidates that proof, so
//! retiring any edge on the route makes the next operation fail with
//! [`FsError::AccessLost`] even though the client still holds the old
//! numeric handle.  A client holding a second live route may reopen the
//! object through it and proceed.  There is deliberately no cache anywhere
//! that could bypass this check.
//!
//! ## Layering
//!
//! ```text
//! engine        the request engine — generic over the provider
//!  ├ authority  sessions, private roots, grants, rights, handle proofs
//!  ├ namespace  containment edges, names, traversal, link/move/unlink
//!  ├ object     stable object identity, kinds, revisions, reclamation
//!  └ provider   the async storage seam; `provider::ram` is the first impl
//! ```
//!
//! [`slot`] supplies the one arena shape all of the above use, and
//! [`bounds`] is the entire memory budget as compile-time constants.
//!
//! Above the service, [`client`] is the typed caller and [`path`] is the
//! `/`-separated path syntax clients walk with — neither is reachable from
//! the engine, and the service never sees a path.
//!
//! ## Phase
//!
//! This crate now includes the Phase-1 engine, compact IPC protocol, founded
//! multi-client service, typed async client, and capability-backed bulk data
//! windows used by ordinary founded processes. The RAM and direct-block
//! providers implement one conformance contract; [`persistence`] adds
//! copy-on-write payloads, atomic root publication, and recovery without
//! serializing runtime authority. Exec provisions each filesystem-capable
//! image a private recyclable pair rather than routing commands through a
//! shell-specific protocol.

#![no_std]
// The volume-scaled derived and block-extent indexes are heap-backed via the
// `Allocator` seam (`service_alloc` → the service's own global heap), which is
// the unstable `allocator_api`. Their capacities remain fixed after startup.
#![feature(allocator_api)]

extern crate alloc;

/// The one place filesystem-owned, volume-scaled metadata names its allocator.
/// The service binary installs a private global heap; host tests receive the
/// host allocator through the same seam.
///
/// Named concretely rather than as `dyn Allocator` because an arena's
/// `Sync`ness is inherited from its allocator, and `Namespace` **must** stay
/// `Sync`: the per-object and per-directory claim maps are shared across
/// threads by the tests that witness "no filesystem-global lock"
/// (`DAGFS.md` §2), and `dyn Allocator` is not `Sync`. The seam is still one
/// name in one place; it is simply a type alias instead of an erased trait
/// object, which is strictly more information and costs no generality — both
/// builds resolve to the same allocator, differing only in which global heap
/// `#[global_allocator]` installs behind it.
pub(crate) type ServiceAlloc = alloc::alloc::Global;

/// The allocator every volume-scaled filesystem arena is built in.
pub(crate) fn service_alloc() -> &'static ServiceAlloc {
    static GLOBAL: ServiceAlloc = alloc::alloc::Global;
    &GLOBAL
}

/// Engines one filesystem row mounts at most: a provisioned disk's user and
/// system stores, each with its own authority and service frontend.
pub const MOUNTED_ENGINES: usize = 2;

/// What the volume-scaled arenas need of the service heap: two mounted
/// engines' fixed reservations — about 1.75 MiB each at
/// [`bounds::MAX_OBJECTS`] = 512, the derived wire-chunk arena alone 1 MiB —
/// with better than 2× headroom for recovery transients and the heap's
/// growth granularity. Measured rather than derived: 4 MiB fit the one engine
/// a blank disk mounts and ran out on the second (#1003), and the fs-dual
/// gate is the witness that keeps it honest. It was the row's whole supply
/// until sessions were allocated per slot.
pub const VOLUME_HEAP_BYTES: usize = 8 << 20;

/// **What the filesystem's heap must hold** — the floor its manifest row's
/// `heap_supply_bytes` is held to (exec's founder test
/// `dagfs_heap_supply_holds_every_session`): the volume arenas
/// ([`VOLUME_HEAP_BYTES`]); every session slot attached at once in both
/// engines ([`authority::SESSION_HEAP_BYTES`]), with each frontend's slot
/// table grown to all of them; and the serve host's peer table grown to
/// every slot a face can occupy. The two tables count twice: a doubling table
/// holds its next block before it frees the one it outgrew, and every block
/// it ever held sums to under twice its last. Derived from the per-slot costs
/// themselves, so a heavier session or serving pair raises the floor instead
/// of overrunning it.
///
/// It does not reserve the proofs open handles carry — 16 KiB at most per
/// session and engine, every handle at the deepest path
/// ([`bounds::MAX_HANDLES_PER_SESSION`] × [`bounds::MAX_PATH_DEPTH`] edges):
/// those are allocated per open at their own depth from what the heap has
/// left, and a heap that cannot hold one refuses that open
/// ([`FsError::OutOfHandles`]) without touching any session's admission.
pub const HEAP_DEMAND_BYTES: u64 = {
    let serve_slot = {
        let durable = kore_rt::svc::peer_slot_bytes::<durable::DurableService>();
        let ram = kore_rt::svc::peer_slot_bytes::<RamService>();
        if durable > ram { durable } else { ram }
    };
    (VOLUME_HEAP_BYTES
        + MOUNTED_ENGINES
            * (authority::SESSION_HEAP_BYTES
                + 2 * bounds::MAX_SESSIONS * service::SESSION_SLOT_BYTES)
        + 2 * kore_rt::svc::PeerId::SLOT_CAPACITY * serve_slot) as u64
};

#[cfg(test)]
extern crate std;

pub mod authority;
pub mod bounds;
pub mod client;
pub mod digests;
pub mod durable;
pub mod engine;
pub mod error;
pub mod extraction;
pub mod namespace;
pub mod object;
mod parent_index;
pub mod path;
pub mod persistence;
pub mod provider;
mod reach;
mod router;
pub mod search;
pub mod service;
pub mod slot;
mod watches;
pub mod window;
pub mod wire;

pub use authority::{AuthorityKind, AuthorityQuota, EdgeProof, GrantId, Handle, Rights, SessionId};
pub use client::process_face;
pub use client::{
    DirEntry, FS_GRANT_TARGET_DELEGATE_RIGHTS, FS_GRANT_TARGET_RIGHTS, FS_VIEW_ANCHOR_RIGHTS,
    FilesystemFace, FsClient,
    FsViewCapability, GrantEntry, IdentifyVersionFailure, LentVersion, Opened, PendingVersion,
    ReleasePendingVersionFailure, ReleaseVersionFailure, RingFs, RingFsError,
    RingReleasePendingVersionFailure, RingReleaseVersionFailure, SearchResultPage, Stat, Transport,
    VERSION_IDENTITY_BYTES, read_dir_entry, read_grant_entry, read_parent, read_search_hit,
    read_system_status,
    read_version_identity,
};
pub use durable::DurableService;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub use durable::run;
pub use engine::{
    Engine, GrantInfo, ListPage, OpenedObject, ParentInfo, ParentsPage, RamEngine, StatInfo,
};
pub use error::{FsError, FsResult, ProviderError};
pub use extraction::{DerivedResultInfo, ExtractionError, VersionLease};
pub use kore_dagfs::DagfsProtocol;
/// The storage-ring client (`res/STORAGE_TRANSPORT.md` §4), for rows that
/// read through the filesystem's transport — exec's image loads.
pub use kore_dagfs::stor;
/// The watch plane every filesystem pair carries after its rings
/// (`kore_dagfs::watch`), for the founders that format pair windows.
pub use kore_dagfs::watch;
pub use namespace::{EdgeId, ListEntry, Name};
pub use object::{ANY_REVISION, ObjectId, ObjectKind, ObjectSubtype, Revision, StableObjectId};
pub use provider::{
    BlockKey, BlockProvider, BlockProviderStats, DurableExtent, RamProvider, StorageLimits,
    StorageProvider,
};
pub use search::{
    FreshnessPolicy, IndexMatch, QueryIndex, SearchContent, SearchCursor, SearchHit, SearchPage,
    SearchPlane, SearchQuery, SearchScore, SearchTerm,
};
pub use service::{
    FACE_HOME_LABEL_BYTES, FsEndowment, FsRequest, FsResponse, FsService, NativeView, RamService,
    SYSTEM_APPS_DIR, SYSTEM_APPS_GRANT, SYSTEM_APPS_VIEW, SYSTEM_LIBS_DIR, SYSTEM_LIBS_GRANT,
    SYSTEM_LIBS_VIEW, SYSTEM_PKGS_DIR, SYSTEM_PKGS_GRANT, SYSTEM_SHARE_DIR, SYSTEM_SHARE_GRANT,
    SYSTEM_SHARE_VIEW, SYSTEM_SOURCE_GRANT, SYSTEM_WORKSPACE_DIR, WindowAttachment,
    face_home_label,
};

/// Architecture-neutral service entry selected at this crate boundary.
///
/// x86 composes the persistent block-backed provider. Targets without that
/// binding serve the identical filesystem authority and wire contract from
/// the bounded RAM provider; callers and crt0 never branch on storage.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    kore_rt::entry::serve::<RamService>(rt).await
}

/// The provider-neutral filesystem wire contract cited by the boot manifest.
///
/// RAM and durable entries implement this exact protocol; naming the alias in
/// the manifest keeps provider selection at the entry seam rather than making
/// the founder pretend that x86 always runs the RAM composition.
pub type FsProtocol = RamService;

/// **Who stops the filesystem's device** (#809 §L.5).  The row selects no
/// device at founding (`DEVICES` is empty): its NVMe function arrives at
/// runtime from the block row, which `Delegated` it here.  So the binding is
/// cited as a runtime-claimed class — the one citation that puts this row in
/// the terminal sweep.  Declared once and cited by both `SystemService`
/// impls: [`FsProtocol`] is what the manifest reads, `DurableService` is
/// what holds the binding on metal.
pub const FS_DEVICE_CLASSES: &[kore_rt::service::DeviceClass] =
    &[kore_driver::host::device_class::<durable::NvmeFsBinding>()];
pub use slot::{SlotKey, SlotTable};
pub use window::DataWindow;

/// The client-side future driver, shared with every host suite (moved to the
/// leaf with the client it drives; see [`kore_dagfs::block_on`]).
pub use kore_dagfs::block_on;
