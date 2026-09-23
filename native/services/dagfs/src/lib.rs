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

#[cfg(test)]
extern crate std;

pub mod authority;
pub mod bounds;
pub mod client;
pub mod durable;
pub mod engine;
pub mod error;
pub mod extraction;
pub mod host_workspace;
mod live_workspace;
pub mod namespace;
pub mod object;
pub mod path;
pub mod persistence;
pub mod provider;
mod reach;
mod router;
pub mod search;
pub mod service;
pub mod slot;
pub mod window;
pub mod wire;
pub mod workspace;

pub use authority::{AuthorityKind, AuthorityQuota, EdgeProof, GrantId, Handle, Rights, SessionId};
pub use client::process_face;
pub use client::{
    DirEntry, FS_GRANT_TARGET_DELEGATE_RIGHTS, FS_GRANT_TARGET_RIGHTS, FilesystemFace, FsClient,
    FsViewCapability, GrantEntry, IdentifyVersionFailure, LentVersion, Opened, PendingVersion,
    ReleasePendingVersionFailure, ReleaseVersionFailure, RingFs, RingFsError,
    RingReleasePendingVersionFailure, RingReleaseVersionFailure, SearchResultPage, Stat, Transport,
    VERSION_IDENTITY_BYTES, read_dir_entry, read_grant_entry, read_search_hit, read_system_status,
    read_version_identity,
};
pub use durable::DurableService;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub use durable::run;
pub use engine::{Engine, GrantInfo, ListPage, OpenedObject, RamEngine, StatInfo};
pub use error::{FsError, FsResult, ProviderError};
pub use extraction::{DerivedResultInfo, ExtractionError, VersionLease};
pub use kore_dagfs::DagfsProtocol;
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
    FsEndowment, FsRequest, FsResponse, FsService, NativeView, RamService, SYSTEM_APPS_DIR,
    SYSTEM_APPS_GRANT, SYSTEM_APPS_VIEW, SYSTEM_LIBS_DIR, SYSTEM_LIBS_GRANT, SYSTEM_LIBS_VIEW,
    SYSTEM_PKGS_DIR, SYSTEM_PKGS_GRANT, SYSTEM_SHARE_DIR, SYSTEM_SHARE_GRANT, SYSTEM_SHARE_VIEW,
    SYSTEM_SOURCE_GRANT, SYSTEM_WORKSPACE_DIR, WindowAttachment,
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
