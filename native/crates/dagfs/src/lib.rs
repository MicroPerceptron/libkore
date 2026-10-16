//! # kore-dagfs — the DagFS process-side contract
//!
//! Everything an ordinary application needs in order to *speak* to the
//! filesystem, and nothing it needs in order to *be* one. The storage
//! service (`kore-svc-dagfs`) depends on this crate for its vocabulary; no
//! application depends on the service.
//!
//! Two tiers, selected by the `runtime` feature:
//!
//! - **Pure types** (always): [`Name`], the object identities, the
//!   wire-visible ids ([`Rights`], [`Handle`], [`GrantId`], [`SessionId`],
//!   [`VersionLease`], [`AuthorityQuota`]), the refusal vocabulary
//!   ([`FsError`]), the capacity [`bounds`], and the `/`-separated
//!   [`path`] syntax every client walks with. A linker stub or a
//!   no-allocator image can hold these.
//! - **The protocol** (`runtime`): [`FsRequest`]/[`FsResponse`], the
//!   fail-closed [`wire`] codec over `kore_abi::svc`'s frozen records, the
//!   [`search`] query types, the ring client ([`RingFs`]) and the host
//!   client ([`FsClient`]), and the view capability leaves.
//!
//! None of it is cached authorization: every operation still crosses the
//! service and revalidates its bounded route. A handle here is a word the
//! service minted for one session; on any other pair it is inert.

#![no_std]

#[cfg(test)]
extern crate std;

/// The display name of the **library view** every task view carries: the
/// shared `lib/` directory the store installer seeds, granted `READ` alone —
/// no `WRITE`, `MUTATE_NAMESPACE`, `SHARE`, or `BIND_EXECUTABLE`. The
/// service mints the edge under this name and a late loader enters it by
/// this name, so the two halves share one spelling.
pub const LIBS_VIEW: &str = "libs";

pub mod bounds;
#[cfg(feature = "runtime")]
mod capability;
#[cfg(feature = "runtime")]
pub mod client;
mod entry;
mod error;
mod face;
#[cfg(feature = "runtime")]
mod fs;
#[cfg(feature = "runtime")]
mod graph;
mod ids;
#[cfg(feature = "index")]
pub mod index;
#[cfg(feature = "runtime")]
mod io;
mod name;
mod object;
pub mod path;
#[cfg(feature = "runtime")]
pub mod protocol;
mod refusal;
#[cfg(feature = "runtime")]
pub mod search;
#[cfg(feature = "runtime")]
pub mod stor;
pub mod text;
#[cfg(feature = "runtime")]
mod view;
#[cfg(feature = "runtime")]
mod walk;
#[cfg(feature = "runtime")]
pub mod wire;

#[cfg(feature = "runtime")]
pub use capability::{FS_GRANT_TARGET_DELEGATE_RIGHTS, FS_GRANT_TARGET_RIGHTS, FsViewCapability};
#[cfg(feature = "runtime")]
pub use client::{
    FilesystemFace as ClientFace, FsClient, GrantEntry, IdentifyVersionFailure, LentVersion,
    Opened, RING_TARGET_RIGHTS, PendingVersion, ReleasePendingVersionFailure,
    ReleaseVersionFailure, RingFs, RingFsError, RingReleasePendingVersionFailure,
    RingReleaseVersionFailure, SearchResultPage, Stat, Transport, VERSION_IDENTITY_BYTES,
    process_face, read_dir_entry, read_grant_entry, read_search_hit, read_system_status,
    read_version_identity,
};
pub use entry::DirEntry;
pub use error::{FsError, FsResult, ProviderError};
pub use face::FilesystemFace;
#[cfg(feature = "runtime")]
pub use fs::{Fs, PEAK_HANDLES};
pub use ids::{AuthorityQuota, GrantId, Handle, Rights, SessionId, VersionLease};
#[cfg(feature = "runtime")]
pub use io::SaveOutcome;
pub use name::Name;
pub use object::{ANY_REVISION, ObjectId, ObjectKind, ObjectSubtype, Revision, StableObjectId};
#[cfg(feature = "runtime")]
pub use protocol::{DagfsProtocol, FsRequest, FsResponse};
pub use refusal::{Refusal, Stop, describe};
#[cfg(feature = "runtime")]
pub use search::{
    FreshnessPolicy, MAX_SEARCH_PAGE, MAX_SEARCH_SCAN, MAX_SEARCH_SNIPPET, MAX_SEARCH_WORK,
    SearchContent, SearchCursor, SearchHit, SearchPlane, SearchQuery, SearchScore, SearchTerm,
};
#[cfg(feature = "runtime")]
pub use walk::{
    CloseFailure, View, WALK_MUTATE, WALK_READ, WALK_WRITE, rights_label, rights_parse,
};

#[cfg(feature = "runtime")]
/// Drive a future to completion by polling it on the current thread.
///
/// The engine and the provider seam are async so that a hardware- or
/// service-backed provider cannot block an executor worker. The
/// RAM provider, though, never yields: every one of its futures is `Ready`
/// on first poll.  This driver exists for exactly that case — host tests,
/// and any future host tool that runs the engine over RAM — and it is not
/// the service's executor.  On metal, `kore_rt`'s `Executor` drives these
/// same futures, and a provider that genuinely parks is why the seam is
/// async in the first place.
///
/// # Panics
///
/// Panics if the future is still `Pending` after [`SPIN_LIMIT`] polls.  With
/// no waker there is nothing to make progress, so spinning forever would
/// hang a test run rather than fail it; a panic names the real bug — a
/// future that parks being driven by the wrong executor.
///
/// [`SPIN_LIMIT`]: block_on#panics
pub fn block_on<F: core::future::Future>(future: F) -> F::Output {
    /// Polls allowed before we call it a hang.  Any non-parking future
    /// completes on the first poll; the slack is for a future that yields a
    /// bounded number of times.
    const SPIN_LIMIT: usize = 1024;

    kore_rt::try_block_on(future, SPIN_LIMIT).unwrap_or_else(|| {
        panic!(
            "block_on: future still pending after {SPIN_LIMIT} polls — it parks, so it needs a real executor"
        )
    })
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    #[test]
    fn block_on_returns_a_ready_value() {
        assert_eq!(block_on(async { 7u32 }), 7);
    }

    #[test]
    fn block_on_drives_a_chain_of_awaits() {
        async fn inner(x: u32) -> u32 {
            x + 1
        }
        async fn outer() -> u32 {
            inner(inner(1).await).await
        }
        assert_eq!(block_on(outer()), 3);
    }
}
