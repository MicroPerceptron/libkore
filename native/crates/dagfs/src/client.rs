//! The typed filesystem client.
//!
//! What a program calls instead of hand-assembling records. It owns the two
//! things a caller should not have to think about — encoding a request into the
//! frozen form, and reading the reply *against the request it answers* — and
//! nothing else.
//!
//! # No authority cache, deliberately
//!
//! The authority contract permits no cache that bypasses route revalidation.
//! This client therefore remembers **no** rights, no resolved
//! routes, and no liveness. Every call goes to the service, and a handle whose
//! route was dismantled gets [`FsError::AccessLost`] on its next use rather
//! than a stale local "yes". The only state here is the data-window
//! bookkeeping, which is about where bytes go, never about who may touch them.
//!
//! # The transport seam
//!
//! [`Transport`] is one method: hand over a request record, get a reply record.
//! On metal that is the pair's SPSC rings; in tests it is a direct call into a
//! the service's `FsService`. Keeping it a trait is what lets
//! the whole client be exercised on the host over *real encoded bytes* rather
//! than mocked at the enum level — the codec is in the loop either way, which
//! is where protocol bugs actually live.

use kore_abi::svc::{FsDirEntryWire, FsRepWire, FsReqWire, FsSearchHitWire};
use kore_abi::{
    cap::Rights as CapRights,
    session::{SlotRef, Status},
};
use kore_rt::{
    svc::{
        CallError, ReturnCallError, ReturnCapability, ReturnDeriveError, ReturnDeriver,
        ReturnLeaseSource, StagedReturn, StagedReturnError, SvcClient,
    },
    transport as rt_transport,
};
use kore_svc_extract::schema::InputIdentity;

use crate::{
    AuthorityQuota, FsError, FsRequest, FsResponse, FsResult, GrantId, Handle, Name, ObjectId,
    ObjectKind, Revision, Rights, SearchCursor, SearchHit, SearchQuery, SessionId, VersionLease,
    wire::{decode_rep_for, encode_req, reply_matches},
};

pub use crate::{
    DirEntry, FS_GRANT_TARGET_DELEGATE_RIGHTS, FS_GRANT_TARGET_RIGHTS, FilesystemFace,
    FsViewCapability,
};

/// Bytes written by [`FsRequest::LendVersion`] or
/// [`FsRequest::LendExecutable`] for the immutable source identity. This is
/// deliberately the service contract's exact width rather than
/// `size_of::<InputIdentity>()`: the data window is a byte protocol and must
/// not inherit Rust padding.
pub const VERSION_IDENTITY_BYTES: usize = 56;

/// One owner-bound immutable filesystem revision awaiting its data-window
/// identity readback.
///
/// This wrapper is intentionally neither `Copy` nor `Clone`.  The underlying
/// [`VersionLease`] is an opaque wire name whose bits convey no authority
/// without this client's service pair, but making the typed client owner
/// affine still prevents ordinary adapter code from losing track of which
/// snapshot must be released after registration.
#[derive(Debug, PartialEq, Eq)]
pub struct PendingVersion {
    lease: VersionLease,
}

impl PendingVersion {
    /// Pair this lease with the identity written by the same request.
    pub fn identify(
        self,
        window: &[u8],
        window_offset: u32,
    ) -> Result<LentVersion, IdentifyVersionFailure> {
        match read_version_identity(window, window_offset) {
            Ok(identity) => Ok(LentVersion {
                lease: self.lease,
                identity,
            }),
            Err(error) => Err(IdentifyVersionFailure {
                error,
                version: self,
            }),
        }
    }
}

/// A pinned version paired with its immutable byte identity.
#[derive(Debug, PartialEq, Eq)]
pub struct LentVersion {
    lease: VersionLease,
    identity: InputIdentity,
}

impl LentVersion {
    /// The immutable byte identity written by the filesystem when this lease
    /// was installed.
    pub const fn identity(&self) -> InputIdentity {
        self.identity
    }
}

/// Identity readback failed after the filesystem had already installed the
/// lease.  The pending owner is returned so it can still be released.
#[derive(Debug, PartialEq, Eq)]
pub struct IdentifyVersionFailure {
    pub error: FsError,
    pub version: PendingVersion,
}

/// A version release that did not reach a terminal filesystem answer.
///
/// The affine owner comes back beside the refusal so the caller can retry or
/// deliberately retain it; a transport error must never silently discard the
/// only typed cleanup witness.
#[derive(Debug, PartialEq, Eq)]
pub struct ReleaseVersionFailure {
    pub error: FsError,
    pub version: LentVersion,
}

/// One round trip: a request record in, a reply record out.
///
/// Implementations do not interpret either — the codec above and the dispatch
/// below own meaning. A transport that cannot complete the exchange reports
/// [`FsError::ProviderFailure`] or [`FsError::Unsupported`] as fits; it must
/// never fabricate a reply.
pub trait Transport {
    /// Exchange one request for its reply.
    fn exchange(&mut self, request: &FsReqWire) -> FsResult<FsRepWire>;
}

/// One authority edge this session holds, lifted out of the data window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrantEntry {
    pub grant: GrantId,
    pub rights: Rights,
    /// `true` for this session's own private root, `false` for a grant made to
    /// it by someone else.
    pub is_private_root: bool,
    /// The displayed attachment name the holder chose, if any.
    pub display_name: Option<Name>,
}

/// A page of a listing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListPage {
    /// Entries written into the data window by this call.
    pub entries: u32,
    /// Cursor to resume at, or `None` when the listing is complete.
    pub next: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchResultPage {
    pub hits: u32,
    pub observation: u64,
    pub snapshot: u64,
    pub index_generation: u64,
    pub freshness: crate::FreshnessPolicy,
    pub next: Option<SearchCursor>,
    pub scanned: u32,
    pub work: u32,
}

/// The typed client.
pub struct FsClient<T: Transport> {
    transport: T,
}

/// The production asynchronous filesystem client over one generic service
/// pair. Unlike [`FsClient`]'s byte-exchange seam, this owns the SPSC client
/// halves and participates in the retained-return capability protocol.
pub struct RingFs<'w, 't, T: rt_transport::Transport> {
    pair: SvcClient<'w, 't, T, crate::DagfsProtocol>,
    return_tx: SlotRef,
    return_land: SlotRef,
}

/// Failure of the production ring client, preserving the distinction between
/// a filesystem refusal and a torn or malformed transport path.
#[derive(Debug)]
pub enum RingFsError {
    Infrastructure(CallError),
    Refused(FsError),
    Wire,
}

impl RingFsError {
    /// Whether retrying the exact outstanding operation can make progress.
    ///
    /// A freshly admitted recyclable peer can become visible to its client
    /// before the serving task has completed the next receive/post turn on
    /// another CPU.  These outcomes prove either that the request did not
    /// move or that the already-published request remains the sole operation
    /// on the pair.  Filesystem refusals, malformed replies, revoked
    /// authority, and poisoned pairs are terminal and must never be replayed.
    pub const fn retryable_transport(&self) -> bool {
        matches!(
            self,
            Self::Infrastructure(
                CallError::RingFull
                    | CallError::Transport
                    | CallError::Backpressure
                    | CallError::Refused(Status::WouldBlock | Status::Conflict)
            )
        )
    }
}

/// The exact request identity of one bulk-window attachment.
///
/// Both the founder-authored landing and the byte span participate in
/// correlation. A lifecycle reply outstanding on the same administration
/// pair therefore cannot be resumed as an attachment merely because both
/// successful operations answer [`FsResponse::Done`].
fn attach_window_request(return_land: SlotRef, bytes: u32) -> FsRequest {
    FsRequest::AttachWindow {
        slot: return_land.index(),
        bytes,
    }
}

/// A version release that did not reach a terminal service answer. The affine
/// version owner is returned so a caller can retry instead of leaking a pinned
/// immutable revision behind an uncorrelated request.
#[derive(Debug)]
pub struct RingReleaseVersionFailure {
    pub error: RingFsError,
    pub version: LentVersion,
}

/// [`RingReleaseVersionFailure`]'s pre-identity counterpart.
#[derive(Debug)]
pub struct RingReleasePendingVersionFailure {
    pub error: RingFsError,
    pub version: PendingVersion,
}

/// Rights of the detached data-window view received by the filesystem.
pub const DATA_WINDOW_RIGHTS: CapRights = CapRights::ACCESS.union(CapRights::MODIFY);
/// The launcher-local staged view must additionally carry DERIVE so the
/// rendezvous operation can attenuate it to [`DATA_WINDOW_RIGHTS`].
const DATA_WINDOW_STAGED_RIGHTS: CapRights = DATA_WINDOW_RIGHTS.union(CapRights::DERIVE);

/// Standard boot-role indices of one exec-synthesized process face.
///
/// Compatibility names for [`FilesystemFace::CANONICAL`], the
/// cross-boundary layout shared by exec, application presets, and clients.
/// Existing clients retain their source path while the filesystem leaf
/// contract becomes the single coordinate owner.
pub mod process_face {
    /// Client request doorbell (`role::SVC_SEND`).
    pub const REQUEST_KICK: u16 = super::FilesystemFace::CANONICAL.request_kick();
    /// Control-pair backing (`role::SVC_RING`).
    pub const CONTROL_RING: u16 = super::FilesystemFace::CANONICAL.control_ring();
    /// Bulk data-window backing (`role::SVC_RING`).
    pub const DATA_WINDOW: u16 = super::FilesystemFace::CANONICAL.data_window();
    /// Retained-return send endpoint (`role::SVC_RETURN`).
    pub const RETURN_SEND: u16 = super::FilesystemFace::CANONICAL.return_send();
    /// Service-authored retained landing coordinate (`role::SVC_RETURN`).
    pub const RETURN_LAND: u16 = super::FilesystemFace::CANONICAL.return_land();
}

impl<'w, 't, T: rt_transport::Transport> RingFs<'w, 't, T> {
    /// Assemble a filesystem client from its exclusive pair and the fixed
    /// return-leg coordinates founder topology delivered to this client.
    pub fn new(
        pair: SvcClient<'w, 't, T, crate::DagfsProtocol>,
        return_tx: SlotRef,
        return_land: SlotRef,
    ) -> Self {
        Self {
            pair,
            return_tx,
            return_land,
        }
    }

    fn finish(
        &mut self,
        reply: kore_rt::svc::CallReply<FsResponse>,
    ) -> Result<FsResponse, CallError> {
        if reply.grants != 0
            || reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().any(Option::is_some)
        {
            self.pair.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        Ok(reply.response)
    }

    /// Issue one ordinary data-only filesystem operation.
    pub async fn call(&mut self, request: &FsRequest) -> Result<FsResponse, CallError> {
        let reply = self.pair.call(request).await?;
        self.finish(reply)
    }

    /// Whether the exclusive administration pair can publish a new request.
    pub fn is_idle(&self) -> bool {
        self.pair.is_idle()
    }

    /// Resume the one already-published administration request.
    ///
    /// Callers retain the operation phase in their affine lifecycle owner;
    /// the uncorrelated pair therefore never guesses which request a late
    /// reply answers.
    pub async fn resume(&mut self) -> Result<FsResponse, CallError> {
        let reply = self.pair.resume().await?;
        self.finish(reply)
    }

    /// Publish `request` when idle, or continue it only when the exact same
    /// encoded request already owns this pair.
    ///
    /// This is the lifecycle-safe counterpart of [`resume`](Self::resume): a
    /// cleanup row visited out of order gets [`CallError::Outstanding`]
    /// without kicking the service or consuming another row's late reply.
    pub(crate) async fn call_exact(&mut self, request: &FsRequest) -> Result<FsResponse, CallError> {
        let reply = if self.pair.is_idle() {
            self.pair.call(request).await?
        } else {
            self.pair.resume_exact(request).await?
        };
        self.finish(reply)
    }

    async fn checked(&mut self, request: &FsRequest) -> Result<FsResponse, RingFsError> {
        let response = self
            .call_exact(request)
            .await
            .map_err(RingFsError::Infrastructure)?;
        match response {
            FsResponse::Refused(error) => Err(RingFsError::Refused(error)),
            response => Ok(response),
        }
    }

    /// Derive and convey this pair's writable bulk-data view, then attach the
    /// service to the exact founder-published landing. The parent lease remains
    /// with the client and is the lifetime/revocation pivot for both mappings.
    pub async fn attach_window<S: ReturnLeaseSource>(
        &mut self,
        source: &S,
        bytes: u32,
    ) -> Result<(), RingFsError> {
        if bytes == 0 {
            return Err(RingFsError::Refused(FsError::Range));
        }
        let request = attach_window_request(self.return_land, bytes);
        let reply = self
            .pair
            .call_with_staged_return(&request, self.return_tx, DATA_WINDOW_RIGHTS, |derive| {
                derive.lease_window(source, DATA_WINDOW_STAGED_RIGHTS, 0, u64::from(bytes))
            })
            .await
            .map_err(|error| match error {
                StagedReturnError::NotTransferred(error)
                | StagedReturnError::AfterTransfer(error) => RingFsError::Infrastructure(error),
            })?;
        self.finish_attach_window(reply)
    }

    /// Continue only the exact bulk-window attachment already published by
    /// this pair.
    ///
    /// This never publishes a replacement request. In particular, an
    /// outstanding `RetirePeer`/`ConfirmRetired` returns
    /// [`CallError::Outstanding`] without a kick or reply-ring pop, leaving
    /// that lifecycle operation with its affine cleanup owner.
    pub async fn resume_attach_window(&mut self, bytes: u32) -> Result<(), RingFsError> {
        if bytes == 0 {
            return Err(RingFsError::Refused(FsError::Range));
        }
        let request = attach_window_request(self.return_land, bytes);
        let reply = self
            .pair
            .resume_exact(&request)
            .await
            .map_err(RingFsError::Infrastructure)?;
        self.finish_attach_window(reply)
    }

    fn finish_attach_window(
        &mut self,
        reply: kore_rt::svc::CallReply<FsResponse>,
    ) -> Result<(), RingFsError> {
        match self.finish(reply).map_err(RingFsError::Infrastructure)? {
            FsResponse::Done => Ok(()),
            FsResponse::Refused(error) => Err(RingFsError::Refused(error)),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn open_root(&mut self, rights: Rights) -> Result<Opened, RingFsError> {
        self.opened(&FsRequest::OpenRoot { rights }).await
    }

    pub async fn open_at(
        &mut self,
        parent: Handle,
        name: &Name,
        rights: Rights,
    ) -> Result<Opened, RingFsError> {
        self.opened(&FsRequest::OpenAt {
            parent,
            name: *name,
            rights,
        })
        .await
    }

    pub async fn open_grant(
        &mut self,
        grant: GrantId,
        rights: Rights,
    ) -> Result<Opened, RingFsError> {
        self.opened(&FsRequest::OpenGrant { grant, rights }).await
    }

    /// Hold this session's whole-tree commits until the window closes.
    ///
    /// Open one around a bulk write and close it when the write is complete:
    /// the copy then pays one commit instead of one per chunk. The caller
    /// owns cleanup of anything left half-written if it never closes.
    pub async fn defer_commits(&mut self, hold: bool) -> Result<(), RingFsError> {
        match self.checked(&FsRequest::DeferCommits { hold }).await? {
            FsResponse::Done => Ok(()),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Publish the measured system catalogue's bounded seed lifecycle.
    /// Only the permanent administration pair is authorized by the durable
    /// router; ordinary filesystem faces receive a refusal.
    pub async fn set_system_seed_active(&mut self, active: bool) -> Result<(), RingFsError> {
        match self.checked(&FsRequest::SetSystemSeed { active }).await? {
            FsResponse::Done => Ok(()),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Write from this pair's data window, returning the new revision.
    pub async fn write_at(
        &mut self,
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
        expected: Revision,
    ) -> Result<Revision, RingFsError> {
        match self
            .checked(&FsRequest::WriteAt {
                handle,
                offset,
                window_offset,
                length,
                expected,
            })
            .await?
        {
            FsResponse::Committed { revision } => Ok(revision),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Set one file's complete visible length, returning its new revision.
    pub async fn resize(
        &mut self,
        handle: Handle,
        length: u64,
        expected: Revision,
    ) -> Result<Revision, RingFsError> {
        match self
            .checked(&FsRequest::Resize {
                handle,
                length,
                expected,
            })
            .await?
        {
            FsResponse::Committed { revision } => Ok(revision),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn create(
        &mut self,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
    ) -> Result<Opened, RingFsError> {
        self.create_typed(parent, name, kind, crate::ObjectSubtype::PLAIN)
            .await
    }

    /// Create an object stamped with `subtype` — data the object carries for
    /// its life, never authority. A directory refuses any but `PLAIN`.
    pub async fn create_typed(
        &mut self,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
        subtype: crate::ObjectSubtype,
    ) -> Result<Opened, RingFsError> {
        self.opened(&FsRequest::Create {
            parent,
            name: *name,
            kind,
            subtype,
        })
        .await
    }

    /// List this pair-bound session's authority edges into its attached data
    /// window. Callers decode entries with [`read_grant_entry`]; the returned
    /// numeric grant remains inert on every other pair.
    pub async fn list_grants(&mut self, window_offset: u32) -> Result<u32, RingFsError> {
        match self
            .checked(&FsRequest::ListGrants { window_offset })
            .await?
        {
            FsResponse::GrantsListed { entries } => Ok(entries),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Land the aggregate diagnostic snapshot in the caller's window; the
    /// caller lifts it with [`read_system_status`].
    pub async fn system_status(&mut self, window_offset: u32) -> Result<u32, RingFsError> {
        match self
            .checked(&FsRequest::SystemStatus { window_offset })
            .await?
        {
            FsResponse::SystemStatusReported { len } => Ok(len),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn search(
        &mut self,
        handle: Handle,
        query: SearchQuery,
        cursor: Option<SearchCursor>,
        window_offset: u32,
    ) -> Result<SearchResultPage, RingFsError> {
        match self
            .checked(&FsRequest::Search {
                handle,
                query,
                cursor,
                window_offset,
            })
            .await?
        {
            FsResponse::Searched {
                hits,
                observation,
                snapshot,
                index_generation,
                freshness,
                next,
                scanned,
                work,
            } => Ok(SearchResultPage {
                hits,
                observation,
                snapshot,
                index_generation,
                freshness,
                next,
                scanned,
                work,
            }),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    async fn opened(&mut self, request: &FsRequest) -> Result<Opened, RingFsError> {
        match self.checked(request).await? {
            FsResponse::Opened {
                handle,
                stable_object,
                kind,
                subtype,
                len,
                revision,
            } => Ok(Opened {
                handle,
                stable_object,
                kind,
                subtype,
                len,
                revision,
            }),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn close(&mut self, handle: Handle) -> Result<(), RingFsError> {
        match self.checked(&FsRequest::Close { handle }).await? {
            FsResponse::Done => Ok(()),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Add one native containment edge to an already-held object.
    ///
    /// `Link` is the DAG primitive used by application projections: callers
    /// can expose one immutable app image through multiple namespace shapes
    /// without copying bytes or minting a second executable identity.
    pub async fn link(
        &mut self,
        parent: Handle,
        name: &Name,
        child: Handle,
    ) -> Result<(), RingFsError> {
        match self
            .checked(&FsRequest::Link {
                parent,
                name: *name,
                child,
            })
            .await?
        {
            FsResponse::Done => Ok(()),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Remove one exact native containment edge. Unlike compatibility
    /// `Unlink`, a populated directory is legal: the graph survives through
    /// any other containment or authority route.
    pub async fn detach_edge(&mut self, parent: Handle, name: &Name) -> Result<(), RingFsError> {
        match self
            .checked(&FsRequest::DetachEdge {
                parent,
                name: *name,
            })
            .await?
        {
            FsResponse::Done => Ok(()),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Give one already-attached session an attenuated route to `handle`.
    ///
    /// The destination is validated by the service's own session table; the
    /// numeric id is routing data, never authority. The returned grant remains
    /// inert on every pair except that exact destination.
    pub async fn grant(
        &mut self,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        display_name: Option<Name>,
    ) -> Result<GrantId, RingFsError> {
        match self
            .checked(&FsRequest::Grant {
                handle,
                destination,
                rights,
                display_name,
            })
            .await?
        {
            FsResponse::Granted { grant } => Ok(grant),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Give one attached session an attenuated, explicitly budgeted route.
    pub async fn grant_scoped(
        &mut self,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        quota: AuthorityQuota,
        display_name: Option<Name>,
    ) -> Result<GrantId, RingFsError> {
        match self
            .checked(&FsRequest::GrantScoped {
                handle,
                destination,
                rights,
                quota,
                display_name,
            })
            .await?
        {
            FsResponse::Granted { grant } => Ok(grant),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn lend_version(
        &mut self,
        handle: Handle,
        window_offset: u32,
    ) -> Result<PendingVersion, RingFsError> {
        match self
            .checked(&FsRequest::LendVersion {
                handle,
                window_offset,
            })
            .await?
        {
            FsResponse::VersionLent { lease } => Ok(PendingVersion { lease }),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    /// Pin a revision for executable registration. The filesystem installs
    /// no lease unless this route carries its separate binding authority.
    pub async fn lend_executable(
        &mut self,
        handle: Handle,
        window_offset: u32,
    ) -> Result<PendingVersion, RingFsError> {
        match self
            .checked(&FsRequest::LendExecutable {
                handle,
                window_offset,
            })
            .await?
        {
            FsResponse::VersionLent { lease } => Ok(PendingVersion { lease }),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn read_version(
        &mut self,
        version: &LentVersion,
        offset: u64,
        window_offset: u32,
        length: u32,
    ) -> Result<u32, RingFsError> {
        match self
            .checked(&FsRequest::ReadVersion {
                lease: version.lease,
                offset,
                window_offset,
                length,
            })
            .await?
        {
            FsResponse::Transferred { count } => Ok(count),
            _ => {
                self.pair.poison();
                Err(RingFsError::Wire)
            }
        }
    }

    pub async fn release_version(
        &mut self,
        version: LentVersion,
    ) -> Result<(), RingReleaseVersionFailure> {
        match self
            .checked(&FsRequest::ReleaseVersion {
                lease: version.lease,
            })
            .await
        {
            Ok(FsResponse::Done) => Ok(()),
            Ok(_) => {
                self.pair.poison();
                Err(RingReleaseVersionFailure {
                    error: RingFsError::Wire,
                    version,
                })
            }
            Err(error) => Err(RingReleaseVersionFailure { error, version }),
        }
    }

    pub async fn release_pending_version(
        &mut self,
        version: PendingVersion,
    ) -> Result<(), RingReleasePendingVersionFailure> {
        match self
            .checked(&FsRequest::ReleaseVersion {
                lease: version.lease,
            })
            .await
        {
            Ok(FsResponse::Done) => Ok(()),
            Ok(_) => {
                self.pair.poison();
                Err(RingReleasePendingVersionFailure {
                    error: RingFsError::Wire,
                    version,
                })
            }
            Err(error) => Err(RingReleasePendingVersionFailure { error, version }),
        }
    }

    /// Convey one affine component of a recyclable peer bundle through this
    /// pair's retained return leg.
    pub async fn admit_component<C: ReturnCapability>(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        capability: C,
        rights: CapRights,
    ) -> Result<FsResponse, ReturnCallError<C>> {
        let request = FsRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_return(&request, self.return_tx, capability, rights)
            .await?;
        self.finish(reply).map_err(ReturnCallError::AfterTransfer)
    }

    /// Convey one freshly derived peer component while retaining the source
    /// lineage locally. The derive and return send commit atomically, so a
    /// late transport result can never erase exec's only cleanup pivot.
    pub async fn admit_component_staged<F>(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        rights: CapRights,
        stage: F,
    ) -> Result<FsResponse, StagedReturnError>
    where
        F: FnOnce(ReturnDeriver<'_>) -> Result<StagedReturn, ReturnDeriveError>,
    {
        let request = FsRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_staged_return(&request, self.return_tx, rights, stage)
            .await?;
        self.finish(reply).map_err(StagedReturnError::AfterTransfer)
    }

    pub async fn register_system_view<C: ReturnCapability>(
        &mut self,
        capability: C,
        fs_rights: Rights,
        rights: CapRights,
    ) -> Result<FsResponse, ReturnCallError<C>> {
        let request = FsRequest::RegisterSystemView {
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_return(&request, self.return_tx, capability, rights)
            .await?;
        self.finish(reply).map_err(ReturnCallError::AfterTransfer)
    }

    pub async fn register_system_view_staged(
        &mut self,
        fs_rights: Rights,
        source: SlotRef,
    ) -> Result<FsResponse, StagedReturnError> {
        let request = FsRequest::RegisterSystemView {
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_staged_return(
                &request,
                self.return_tx,
                FsViewCapability::INVOCATION_RIGHTS,
                |stage| stage.send_from(source, FsViewCapability::INVOCATION_DELEGATE_RIGHTS),
            )
            .await?;
        self.finish(reply).map_err(StagedReturnError::AfterTransfer)
    }

    /// Register one independently rooted task view. The source remains with
    /// exec as the revocation pivot; FS retains only the staged descendant
    /// that authenticates and bounds this view's lifetime. `name` selects
    /// the view's durable home directory beneath the administration
    /// workspace — the identity that carries this task's files across a
    /// reboot — and must be unique among live views.
    pub async fn register_private_view_staged(
        &mut self,
        fs_rights: Rights,
        name: Name,
        source: SlotRef,
    ) -> Result<FsResponse, StagedReturnError> {
        let request = FsRequest::RegisterPrivateView {
            rights: fs_rights,
            name,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_staged_return(
                &request,
                self.return_tx,
                FsViewCapability::INVOCATION_RIGHTS,
                |stage| stage.send_from(source, FsViewCapability::INVOCATION_DELEGATE_RIGHTS),
            )
            .await?;
        self.finish(reply).map_err(StagedReturnError::AfterTransfer)
    }

    pub async fn bind_view<C: ReturnCapability>(
        &mut self,
        handle: Handle,
        fs_rights: Rights,
        capability: C,
    ) -> Result<FsResponse, ReturnCallError<C>> {
        let request = FsRequest::BindView {
            handle,
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_exact_return(&request, self.return_tx, capability)
            .await?;
        self.finish(reply).map_err(ReturnCallError::AfterTransfer)
    }

    pub async fn grant_view<C: ReturnCapability>(
        &mut self,
        handle: Handle,
        fs_rights: Rights,
        display_name: Name,
        target: C,
    ) -> Result<FsResponse, ReturnCallError<C>> {
        let request = FsRequest::GrantView {
            handle,
            rights: fs_rights,
            display_name,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_exact_return(&request, self.return_tx, target)
            .await?;
        self.finish(reply).map_err(ReturnCallError::AfterTransfer)
    }

    pub async fn bind_view_staged(
        &mut self,
        handle: Handle,
        fs_rights: Rights,
        source: SlotRef,
    ) -> Result<FsResponse, StagedReturnError> {
        let request = FsRequest::BindView {
            handle,
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_staged_return(
                &request,
                self.return_tx,
                FsViewCapability::INVOCATION_RIGHTS,
                |stage| stage.send_from(source, FsViewCapability::INVOCATION_DELEGATE_RIGHTS),
            )
            .await?;
        self.finish(reply).map_err(StagedReturnError::AfterTransfer)
    }

    pub async fn apply_view<C: ReturnCapability>(
        &mut self,
        peer: kore_rt::svc::PeerId,
        fs_rights: Rights,
        capability: C,
        rights: CapRights,
    ) -> Result<FsResponse, ReturnCallError<C>> {
        let request = FsRequest::ApplyView {
            peer,
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_return(&request, self.return_tx, capability, rights)
            .await?;
        self.finish(reply).map_err(ReturnCallError::AfterTransfer)
    }

    pub async fn apply_view_staged(
        &mut self,
        peer: kore_rt::svc::PeerId,
        fs_rights: Rights,
        source: SlotRef,
    ) -> Result<FsResponse, StagedReturnError> {
        let request = FsRequest::ApplyView {
            peer,
            rights: fs_rights,
            announced: self.return_land,
        };
        let reply = self
            .pair
            .call_with_exact_held_return(&request, self.return_tx, source)
            .await?;
        self.finish(reply).map_err(StagedReturnError::AfterTransfer)
    }

    /// Consume the host completion published after the final component was
    /// accepted and mapped.
    pub async fn take_admission(&mut self) -> Result<FsResponse, CallError> {
        self.call(&FsRequest::TakeAdmission).await
    }

    /// Ask the service host to retire one exact dynamic generation.
    pub async fn retire_peer(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<FsResponse, CallError> {
        self.call_exact(&FsRequest::RetirePeer { peer }).await
    }

    /// Fence one targeted retirement. Success means the host no longer owns
    /// this exact generation, so the caller may reclaim the local pair roots
    /// and backing lease without racing the service's final access.
    pub async fn confirm_retired(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<FsResponse, CallError> {
        self.call_exact(&FsRequest::ConfirmRetired { peer }).await
    }
}

impl<T: Transport> FsClient<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// The transport, for host tests that want to inspect the far side.
    pub fn transport(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Issue one request and lift its reply.
    ///
    /// The reply is decoded *against* `request` — one fixed record serves every
    /// verb, so the request is what says whether `count` is a byte count or a
    /// link count. A reply that does not answer this verb is refused here
    /// rather than handed up as a plausible wrong answer.
    pub fn call(&mut self, request: &FsRequest) -> FsResult<FsResponse> {
        let wire = encode_req(request);
        let reply = self.transport.exchange(&wire)?;
        let (response, _grants) = decode_rep_for(request, &reply)
            .ok_or(FsError::ProviderFailure(crate::error::ProviderError::Media))?;
        if !reply_matches(request, &response) {
            // The rings are uncorrelated SPSC pairs: a reply for another verb
            // means the stream desynchronized, which is not a refusal to hand
            // the caller — it is a transport fault.
            return Err(FsError::ProviderFailure(crate::error::ProviderError::Media));
        }
        Ok(response)
    }

    /// `call`, folding a `Refused` into the error channel.
    ///
    /// Most callers want a refusal to be an `Err`; the few that want to inspect
    /// the response shape use [`Self::call`].
    fn checked(&mut self, request: &FsRequest) -> FsResult<FsResponse> {
        match self.call(request)? {
            FsResponse::Refused(err) => Err(err),
            other => Ok(other),
        }
    }

    // ── window ──────────────────────────────────────────────────────────────

    /// Tell the service where the caller's delivered data window landed.
    ///
    /// The view itself crosses through the return channel; this names only the
    /// arena slot it went to and how long it is. Until this succeeds, every
    /// bulk verb refuses [`FsError::NoWindow`].
    pub fn attach_window(&mut self, slot: u32, bytes: u32) -> FsResult<()> {
        self.checked(&FsRequest::AttachWindow { slot, bytes })?;
        Ok(())
    }

    // ── opening ─────────────────────────────────────────────────────────────

    pub fn open_root(&mut self, rights: Rights) -> FsResult<Opened> {
        self.opened(&FsRequest::OpenRoot { rights })
    }

    pub fn open_at(&mut self, parent: Handle, name: &Name, rights: Rights) -> FsResult<Opened> {
        self.opened(&FsRequest::OpenAt {
            parent,
            name: *name,
            rights,
        })
    }

    pub fn open_grant(&mut self, grant: GrantId, rights: Rights) -> FsResult<Opened> {
        self.opened(&FsRequest::OpenGrant { grant, rights })
    }

    pub fn create(&mut self, parent: Handle, name: &Name, kind: ObjectKind) -> FsResult<Opened> {
        self.create_typed(parent, name, kind, crate::ObjectSubtype::PLAIN)
    }

    /// Synchronous host form of [`RingFs::create_typed`].
    pub fn create_typed(
        &mut self,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
        subtype: crate::ObjectSubtype,
    ) -> FsResult<Opened> {
        self.opened(&FsRequest::Create {
            parent,
            name: *name,
            kind,
            subtype,
        })
    }

    fn opened(&mut self, request: &FsRequest) -> FsResult<Opened> {
        match self.checked(request)? {
            FsResponse::Opened {
                handle,
                stable_object,
                kind,
                subtype,
                len,
                revision,
            } => Ok(Opened {
                handle,
                stable_object,
                kind,
                subtype,
                len,
                revision,
            }),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    pub fn close(&mut self, handle: Handle) -> FsResult<()> {
        self.checked(&FsRequest::Close { handle })?;
        Ok(())
    }

    // ── metadata ────────────────────────────────────────────────────────────

    pub fn stat(&mut self, handle: Handle) -> FsResult<Stat> {
        match self.checked(&FsRequest::Stat { handle })? {
            FsResponse::Stat {
                object,
                stable_object,
                kind,
                subtype,
                len,
                revision,
                links,
                grants,
            } => Ok(Stat {
                object,
                stable_object,
                kind,
                subtype,
                len,
                revision,
                links,
                grants,
            }),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// One page of a directory, landed in the data window at `window_offset`.
    ///
    /// The entries themselves are read back with [`Self::read_entry`] against
    /// the caller's own view of the window — the client does not hold that
    /// view, because on metal the window is the caller's memory, not the
    /// library's.
    pub fn list(&mut self, handle: Handle, cursor: u32, window_offset: u32) -> FsResult<ListPage> {
        match self.checked(&FsRequest::List {
            handle,
            cursor,
            window_offset,
        })? {
            FsResponse::Listed { entries, next } => Ok(ListPage { entries, next }),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// Search only the namespace reachable through this client's native view.
    /// Results are written into the attached data window at `window_offset`.
    pub fn search(
        &mut self,
        handle: Handle,
        query: SearchQuery,
        cursor: Option<SearchCursor>,
        window_offset: u32,
    ) -> FsResult<SearchResultPage> {
        match self.checked(&FsRequest::Search {
            handle,
            query,
            cursor,
            window_offset,
        })? {
            FsResponse::Searched {
                hits,
                observation,
                snapshot,
                index_generation,
                freshness,
                next,
                scanned,
                work,
            } => Ok(SearchResultPage {
                hits,
                observation,
                snapshot,
                index_generation,
                freshness,
                next,
                scanned,
                work,
            }),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// List the authority edges this session holds into the data window,
    /// returning how many landed.
    ///
    /// This is how a recipient discovers a grant made to it: without it, a
    /// founded client would have no way to learn a [`GrantId`] and delegated
    /// authority would be unreachable from a real client.
    pub fn list_grants(&mut self, window_offset: u32) -> FsResult<u32> {
        match self.checked(&FsRequest::ListGrants { window_offset })? {
            FsResponse::GrantsListed { entries } => Ok(entries),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// Land the aggregate diagnostic snapshot in the window; the caller lifts
    /// it with [`read_system_status`].
    pub fn system_status(&mut self, window_offset: u32) -> FsResult<u32> {
        match self.checked(&FsRequest::SystemStatus { window_offset })? {
            FsResponse::SystemStatusReported { len } => Ok(len),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// Lift grant entry `index` out of the caller's window bytes.
    pub fn read_grant_entry(
        window: &[u8],
        window_offset: u32,
        index: usize,
    ) -> FsResult<GrantEntry> {
        use kore_abi::svc::FsGrantEntryWire;
        use zerocopy::FromBytes;
        let at = (window_offset as usize)
            .checked_add(
                index
                    .checked_mul(FsGrantEntryWire::BYTES)
                    .ok_or(FsError::Range)?,
            )
            .ok_or(FsError::Range)?;
        let end = at
            .checked_add(FsGrantEntryWire::BYTES)
            .ok_or(FsError::Range)?;
        let bytes = window.get(at..end).ok_or(FsError::Range)?;
        let wire = FsGrantEntryWire::read_from_bytes(bytes).map_err(|_| FsError::Range)?;
        if wire.reserved0 != 0 || wire.reserved1.get() != 0 {
            return Err(FsError::Range);
        }
        let len = wire.name_len as usize;
        if len > kore_abi::svc::FS_NAME_BYTES || wire.name[len..].iter().any(|b| *b != 0) {
            return Err(FsError::InvalidName);
        }
        Ok(GrantEntry {
            grant: GrantId::from_bits(wire.grant.get()),
            // Exact, not truncating: an undefined right in a listing means the
            // service is speaking a protocol this client does not implement.
            rights: Rights::from_bits_exact(wire.rights).ok_or(FsError::Range)?,
            is_private_root: match wire.kind {
                FsGrantEntryWire::KIND_PRIVATE_ROOT => true,
                FsGrantEntryWire::KIND_GRANT => false,
                _ => return Err(FsError::Range),
            },
            display_name: if len == 0 {
                None
            } else {
                Some(Name::from_bytes(&wire.name[..len])?)
            },
        })
    }

    /// Lift entry `index` of a listing out of the caller's window bytes.
    ///
    /// A free function in spirit — it touches no service state — but kept here
    /// so the window's entry layout has exactly one reader.
    pub fn read_entry(window: &[u8], window_offset: u32, index: usize) -> FsResult<DirEntry> {
        read_dir_entry(window, window_offset, index)
    }

    // ── bytes ───────────────────────────────────────────────────────────────

    /// Read into the data window at `window_offset`, returning the byte count.
    pub fn read(
        &mut self,
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
    ) -> FsResult<u32> {
        match self.checked(&FsRequest::Read {
            handle,
            offset,
            window_offset,
            length,
        })? {
            FsResponse::Transferred { count } => Ok(count),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    // ── immutable source versions ──────────────────────────────────────────

    /// Pin the exact file revision visible through `handle`.
    ///
    /// The service writes the immutable identity into the caller's attached
    /// data window.  The returned [`PendingVersion`] must be identified from
    /// those bytes before it can be read through the typed API.
    pub fn lend_version(&mut self, handle: Handle, window_offset: u32) -> FsResult<PendingVersion> {
        let lease = match self.checked(&FsRequest::LendVersion {
            handle,
            window_offset,
        })? {
            FsResponse::VersionLent { lease } => lease,
            _ => return Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        };
        Ok(PendingVersion { lease })
    }

    /// Synchronous host/client form of [`RingFs::lend_executable`].
    pub fn lend_executable(
        &mut self,
        handle: Handle,
        window_offset: u32,
    ) -> FsResult<PendingVersion> {
        let lease = match self.checked(&FsRequest::LendExecutable {
            handle,
            window_offset,
        })? {
            FsResponse::VersionLent { lease } => lease,
            _ => return Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        };
        Ok(PendingVersion { lease })
    }

    /// Read through an immutable version into the attached data window.
    pub fn read_version(
        &mut self,
        version: &LentVersion,
        offset: u64,
        window_offset: u32,
        length: u32,
    ) -> FsResult<u32> {
        match self.checked(&FsRequest::ReadVersion {
            lease: version.lease,
            offset,
            window_offset,
            length,
        })? {
            FsResponse::Transferred { count } => Ok(count),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    /// Release a pinned version, returning its affine owner on refusal.
    pub fn release_version(&mut self, version: LentVersion) -> Result<(), ReleaseVersionFailure> {
        match self.checked(&FsRequest::ReleaseVersion {
            lease: version.lease,
        }) {
            Ok(FsResponse::Done) => Ok(()),
            Ok(_) => Err(ReleaseVersionFailure {
                error: FsError::ProviderFailure(crate::error::ProviderError::Media),
                version,
            }),
            Err(error) => Err(ReleaseVersionFailure { error, version }),
        }
    }

    /// Release a pinned version whose identity could not be read back.
    pub fn release_pending_version(
        &mut self,
        version: PendingVersion,
    ) -> Result<(), ReleasePendingVersionFailure> {
        match self.checked(&FsRequest::ReleaseVersion {
            lease: version.lease,
        }) {
            Ok(FsResponse::Done) => Ok(()),
            Ok(_) => Err(ReleasePendingVersionFailure {
                error: FsError::ProviderFailure(crate::error::ProviderError::Media),
                version,
            }),
            Err(error) => Err(ReleasePendingVersionFailure { error, version }),
        }
    }

    /// Write from the data window, returning the new revision.
    pub fn write_at(
        &mut self,
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
        expected: Revision,
    ) -> FsResult<Revision> {
        self.committed(&FsRequest::WriteAt {
            handle,
            offset,
            window_offset,
            length,
            expected,
        })
    }

    /// Append from the data window, returning the offset it landed at and the
    /// new revision.
    pub fn append(
        &mut self,
        handle: Handle,
        window_offset: u32,
        length: u32,
        expected: Revision,
    ) -> FsResult<(u64, Revision)> {
        match self.checked(&FsRequest::Append {
            handle,
            window_offset,
            length,
            expected,
        })? {
            FsResponse::Appended { offset, revision } => Ok((offset, revision)),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    pub fn resize(
        &mut self,
        handle: Handle,
        length: u64,
        expected: Revision,
    ) -> FsResult<Revision> {
        self.committed(&FsRequest::Resize {
            handle,
            length,
            expected,
        })
    }

    fn committed(&mut self, request: &FsRequest) -> FsResult<Revision> {
        match self.checked(request)? {
            FsResponse::Committed { revision } => Ok(revision),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    pub fn sync(&mut self, handle: Handle) -> FsResult<()> {
        self.checked(&FsRequest::Sync { handle })?;
        Ok(())
    }

    // ── namespace ───────────────────────────────────────────────────────────

    pub fn rename(&mut self, parent: Handle, old: &Name, new: &Name) -> FsResult<()> {
        self.checked(&FsRequest::Rename {
            parent,
            old: *old,
            new: *new,
        })?;
        Ok(())
    }

    pub fn relocate(
        &mut self,
        src_parent: Handle,
        name: &Name,
        dst_parent: Handle,
        new_name: &Name,
    ) -> FsResult<()> {
        self.checked(&FsRequest::Relocate {
            src_parent,
            name: *name,
            dst_parent,
            new_name: *new_name,
        })?;
        Ok(())
    }

    pub fn link(&mut self, parent: Handle, name: &Name, child: Handle) -> FsResult<()> {
        self.checked(&FsRequest::Link {
            parent,
            name: *name,
            child,
        })?;
        Ok(())
    }

    pub fn unlink(&mut self, parent: Handle, name: &Name) -> FsResult<()> {
        self.checked(&FsRequest::Unlink {
            parent,
            name: *name,
        })?;
        Ok(())
    }

    pub fn detach_edge(&mut self, parent: Handle, name: &Name) -> FsResult<()> {
        self.checked(&FsRequest::DetachEdge {
            parent,
            name: *name,
        })?;
        Ok(())
    }

    // ── sharing ─────────────────────────────────────────────────────────────

    pub fn grant(
        &mut self,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        match self.checked(&FsRequest::Grant {
            handle,
            destination,
            rights,
            display_name,
        })? {
            FsResponse::Granted { grant } => Ok(grant),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    pub fn grant_scoped(
        &mut self,
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        quota: AuthorityQuota,
        display_name: Option<Name>,
    ) -> FsResult<GrantId> {
        match self.checked(&FsRequest::GrantScoped {
            handle,
            destination,
            rights,
            quota,
            display_name,
        })? {
            FsResponse::Granted { grant } => Ok(grant),
            _ => Err(FsError::ProviderFailure(crate::error::ProviderError::Media)),
        }
    }

    pub fn revoke(&mut self, grant: GrantId) -> FsResult<()> {
        self.checked(&FsRequest::Revoke { grant })?;
        Ok(())
    }
}

/// Lift one authority-filtered search hit from service-written window bytes.
pub fn read_search_hit(window: &[u8], window_offset: u32, index: usize) -> FsResult<SearchHit> {
    use zerocopy::FromBytes;
    let at = usize::try_from(window_offset)
        .map_err(|_| FsError::Range)?
        .checked_add(
            index
                .checked_mul(FsSearchHitWire::BYTES)
                .ok_or(FsError::Range)?,
        )
        .ok_or(FsError::Range)?;
    let end = at
        .checked_add(FsSearchHitWire::BYTES)
        .ok_or(FsError::Range)?;
    let wire = FsSearchHitWire::read_from_bytes(window.get(at..end).ok_or(FsError::Range)?)
        .map_err(|_| FsError::Range)?;
    if wire.reserved0 != 0 || wire.reserved2.iter().any(|byte| *byte != 0) {
        return Err(FsError::Range);
    }
    let len = usize::from(wire.path_len.get());
    if len == 0 || len > wire.path.len() || wire.path[len..].iter().any(|byte| *byte != 0) {
        return Err(FsError::InvalidName);
    }
    let path = core::str::from_utf8(&wire.path[..len]).map_err(|_| FsError::InvalidName)?;
    let mut components = heapless::Vec::new();
    for component in path.split('/') {
        components
            .push(Name::new(component)?)
            .map_err(|_| FsError::Range)?;
    }
    let ranking = wire.ranking.get();
    let score = u16::try_from(ranking & 0xffff).map_err(|_| FsError::Range)?;
    let snippet_len = usize::try_from(ranking >> 16).map_err(|_| FsError::Range)?;
    if score > 10_000
        || snippet_len > wire.snippet.len()
        || wire.snippet[snippet_len..].iter().any(|byte| *byte != 0)
    {
        return Err(FsError::Range);
    }
    let snippet_text =
        core::str::from_utf8(&wire.snippet[..snippet_len]).map_err(|_| FsError::InvalidName)?;
    let mut snippet = heapless::String::new();
    snippet.push_str(snippet_text).map_err(|_| FsError::Range)?;
    Ok(SearchHit {
        path: components,
        object: ObjectId::from_bits(wire.object.get()),
        stable_object: crate::StableObjectId::new(wire.stable_object.get())
            .ok_or(FsError::InvalidHandle)?,
        kind: match wire.kind {
            FsReqWire::KIND_FILE => ObjectKind::File,
            FsReqWire::KIND_DIRECTORY => ObjectKind::Directory,
            _ => return Err(FsError::Range),
        },
        len: wire.len.get(),
        revision: wire.revision.get(),
        score: crate::SearchScore::new(score),
        snippet,
    })
}

/// Lift and validate one directory entry from service-written window bytes.
///
/// This is free of a transport type so small founded clients can decode a
/// listing without fabricating an implementation of the synchronous host-test
/// [`Transport`] seam.
pub fn read_dir_entry(window: &[u8], window_offset: u32, index: usize) -> FsResult<DirEntry> {
    use zerocopy::FromBytes;
    let at = (window_offset as usize)
        .checked_add(
            index
                .checked_mul(FsDirEntryWire::BYTES)
                .ok_or(FsError::Range)?,
        )
        .ok_or(FsError::Range)?;
    let end = at
        .checked_add(FsDirEntryWire::BYTES)
        .ok_or(FsError::Range)?;
    let bytes = window.get(at..end).ok_or(FsError::Range)?;
    let wire = FsDirEntryWire::read_from_bytes(bytes).map_err(|_| FsError::Range)?;
    // The window is the caller's own memory, but the service wrote these
    // bytes, so every reserved and tail byte remains untrusted input.
    // The low byte of `reserved0` is the entry's subtype (data, never
    // authority); its high byte and `reserved1` remain reserved.
    let subtype = u8::try_from(wire.reserved0.get()).map_err(|_| FsError::Range)?;
    if wire.reserved1.get() != 0 {
        return Err(FsError::Range);
    }
    let len = wire.name_len as usize;
    if len > kore_abi::svc::FS_NAME_BYTES || wire.name[len..].iter().any(|b| *b != 0) {
        return Err(FsError::InvalidName);
    }
    Ok(DirEntry {
        name: Name::from_bytes(&wire.name[..len])?,
        object: ObjectId::from_bits(wire.object.get()),
        kind: match wire.kind {
            FsReqWire::KIND_FILE => ObjectKind::File,
            FsReqWire::KIND_DIRECTORY => ObjectKind::Directory,
            _ => return Err(FsError::NotDirectory),
        },
        subtype: crate::ObjectSubtype::from_bits(subtype),
    })
}

/// Lift and validate one authority entry from service-written window bytes.
///
/// This is transport-free for founded clients using the async metal pair.
pub fn read_grant_entry(window: &[u8], window_offset: u32, index: usize) -> FsResult<GrantEntry> {
    use kore_abi::svc::FsGrantEntryWire;
    use zerocopy::FromBytes;
    let at = (window_offset as usize)
        .checked_add(
            index
                .checked_mul(FsGrantEntryWire::BYTES)
                .ok_or(FsError::Range)?,
        )
        .ok_or(FsError::Range)?;
    let end = at
        .checked_add(FsGrantEntryWire::BYTES)
        .ok_or(FsError::Range)?;
    let bytes = window.get(at..end).ok_or(FsError::Range)?;
    let wire = FsGrantEntryWire::read_from_bytes(bytes).map_err(|_| FsError::Range)?;
    if wire.reserved0 != 0 || wire.reserved1.get() != 0 {
        return Err(FsError::Range);
    }
    let len = wire.name_len as usize;
    if len > kore_abi::svc::FS_NAME_BYTES || wire.name[len..].iter().any(|b| *b != 0) {
        return Err(FsError::InvalidName);
    }
    Ok(GrantEntry {
        grant: GrantId::from_bits(wire.grant.get()),
        rights: Rights::from_bits_exact(wire.rights).ok_or(FsError::Range)?,
        is_private_root: match wire.kind {
            FsGrantEntryWire::KIND_PRIVATE_ROOT => true,
            FsGrantEntryWire::KIND_GRANT => false,
            _ => return Err(FsError::Range),
        },
        display_name: if len == 0 {
            None
        } else {
            Some(Name::from_bytes(&wire.name[..len])?)
        },
    })
}

/// Lift and validate the one diagnostic snapshot record from
/// service-written window bytes.
pub fn read_system_status(
    window: &[u8],
    window_offset: u32,
) -> FsResult<kore_abi::svc::FsSystemStatusWire> {
    use kore_abi::svc::FsSystemStatusWire;
    use zerocopy::FromBytes;
    let at = usize::try_from(window_offset).map_err(|_| FsError::Range)?;
    let end = at
        .checked_add(FsSystemStatusWire::BYTES)
        .ok_or(FsError::Range)?;
    let bytes = window.get(at..end).ok_or(FsError::Range)?;
    let wire = FsSystemStatusWire::read_from_bytes(bytes).map_err(|_| FsError::Range)?;
    let durable = wire.flags.get() & FsSystemStatusWire::FLAG_DURABLE != 0;
    if wire.user_lifecycle.get() > FsSystemStatusWire::LIFECYCLE_CURRENT
        || wire.system_lifecycle.get() > FsSystemStatusWire::LIFECYCLE_CURRENT
        || !wire.storage.validate()
        || wire.storage.available() != durable
    {
        return Err(FsError::Range);
    }
    Ok(wire)
}

/// Lift the source identity emitted by [`FsRequest::LendVersion`] or
/// [`FsRequest::LendExecutable`] from an
/// untrusted service-written data window.
pub fn read_version_identity(window: &[u8], window_offset: u32) -> FsResult<InputIdentity> {
    let at = usize::try_from(window_offset).map_err(|_| FsError::Range)?;
    let end = at
        .checked_add(VERSION_IDENTITY_BYTES)
        .ok_or(FsError::Range)?;
    let bytes = window.get(at..end).ok_or(FsError::Range)?;
    let word = |start: usize| -> FsResult<u64> {
        let raw: [u8; 8] = bytes
            .get(start..start + 8)
            .ok_or(FsError::Range)?
            .try_into()
            .map_err(|_| FsError::Range)?;
        Ok(u64::from_le_bytes(raw))
    };
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes[24..VERSION_IDENTITY_BYTES]);
    Ok(InputIdentity {
        source: word(0)?,
        version: word(8)?,
        len: word(16)?,
        digest,
    })
}

/// What an open or create returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opened {
    pub handle: Handle,
    pub stable_object: crate::StableObjectId,
    pub kind: ObjectKind,
    pub subtype: crate::ObjectSubtype,
    pub len: u64,
    pub revision: Revision,
}

/// What a stat returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stat {
    pub object: ObjectId,
    pub stable_object: crate::StableObjectId,
    pub kind: ObjectKind,
    pub subtype: crate::ObjectSubtype,
    pub len: u64,
    pub revision: Revision,
    pub links: u32,
    pub grants: u32,
}

/// A pending-version release that did not reach a terminal answer.
#[derive(Debug, PartialEq, Eq)]
pub struct ReleasePendingVersionFailure {
    pub error: FsError,
    pub version: PendingVersion,
}

#[cfg(test)]
mod exact_attach_tests {
    use core::cell::Cell;

    use super::*;
    use kore_abi::session::{Committed, Status};
    use kore_rt::{
        service::MAX_GRANTS,
        session::SessionBuilder,
        svc::{
            ClientEndpoints, REP_CAP, REQ_CAP, ServeRings, SvcProtocol, split_window, window_bytes,
        },
    };

    struct RecordingTransport {
        commits: Cell<usize>,
        succeed: Cell<bool>,
    }

    impl RecordingTransport {
        const fn refusing() -> Self {
            Self {
                commits: Cell::new(0),
                succeed: Cell::new(false),
            }
        }

        fn complete(&self) -> Result<Committed, ()> {
            self.commits.set(self.commits.get() + 1);
            if self.succeed.get() {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            } else {
                Err(())
            }
        }
    }

    impl rt_transport::Transport for RecordingTransport {
        type Error = ();

        async fn commit(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            self.complete()
        }

        fn commit_now(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            self.complete()
        }
    }

    fn ring_fs(
        transport: &RecordingTransport,
        return_land: SlotRef,
    ) -> (
        RingFs<'static, '_, RecordingTransport>,
        ServeRings<'static, crate::DagfsProtocol>,
    ) {
        let bytes = window_bytes::<crate::DagfsProtocol>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: the test intentionally leaks one correctly aligned window;
        // `split_window` gives its two aliases only the SPSC producer/consumer
        // halves that may lawfully share it.
        let window = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        };
        let (client, serve) =
            split_window::<crate::DagfsProtocol>(window, REQ_CAP, REP_CAP).expect("fs rings");
        let pair = SvcClient::new(
            transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(0x30),
                grants: [None; MAX_GRANTS],
            },
        );
        (
            RingFs::new(pair, SlotRef::from_wire(0x31), return_land),
            serve,
        )
    }

    fn take_request(serve: &mut ServeRings<'_, crate::DagfsProtocol>) -> FsRequest {
        let wire = serve.req.pop().expect("published request");
        <crate::DagfsProtocol as SvcProtocol>::decode_req(&wire).expect("valid filesystem request")
    }

    fn publish_done(serve: &mut ServeRings<'_, crate::DagfsProtocol>) {
        let wire = <crate::DagfsProtocol as SvcProtocol>::encode_rep(&FsResponse::Done, 0);
        assert!(serve.rep.push(wire).is_ok());
    }

    #[test]
    fn close_retry_cannot_consume_another_handles_late_reply() {
        let transport = RecordingTransport::refusing();
        let (mut fs, mut serve) = ring_fs(&transport, SlotRef::from_wire(0x41));
        let close = FsRequest::Close { handle: crate::Handle::from_bits(7) };
        assert_eq!(crate::block_on(fs.call_exact(&close)), Err(CallError::Transport));
        assert_eq!(take_request(&mut serve), close);
        publish_done(&mut serve);
        let other = FsRequest::Close { handle: crate::Handle::from_bits(8) };
        assert_eq!(crate::block_on(fs.call_exact(&other)), Err(CallError::Outstanding));
        assert_eq!(transport.commits.get(), 1);
        transport.succeed.set(true);
        assert_eq!(crate::block_on(fs.call_exact(&close)), Ok(FsResponse::Done));
        assert!(fs.is_idle());
        assert!(serve.req.pop().is_none(), "retry must not publish another Close");
    }

    #[test]
    fn attach_retries_only_transport_progress_outcomes() {
        for error in [
            CallError::RingFull,
            CallError::Transport,
            CallError::Backpressure,
            CallError::Refused(Status::WouldBlock),
            CallError::Refused(Status::Conflict),
        ] {
            assert!(RingFsError::Infrastructure(error).retryable_transport());
        }

        for error in [
            CallError::Outstanding,
            CallError::NoOutstanding,
            CallError::Poisoned,
            CallError::Wire,
            CallError::Wiring,
            CallError::Refused(Status::Denied),
            CallError::Refused(Status::Revoked),
        ] {
            assert!(!RingFsError::Infrastructure(error).retryable_transport());
        }
        assert!(!RingFsError::Refused(FsError::Busy).retryable_transport());
        assert!(!RingFsError::Wire.retryable_transport());
    }

    #[test]
    fn attach_resume_identity_cannot_alias_cleanup_or_different_geometry() {
        let landing = SlotRef::from_wire(0x41);
        let attach = attach_window_request(landing, 0x4000);
        assert_eq!(
            attach,
            FsRequest::AttachWindow {
                slot: 0x41,
                bytes: 0x4000,
            }
        );

        let wire = encode_req(&attach);
        let peer = kore_rt::svc::PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16)
            .expect("generation-one peer");
        for lifecycle in [
            FsRequest::RetirePeer { peer },
            FsRequest::ConfirmRetired { peer },
        ] {
            assert_ne!(
                wire,
                encode_req(&lifecycle),
                "a lifecycle Done reply must not satisfy AttachWindow"
            );
        }
        for different in [
            attach_window_request(SlotRef::from_wire(0x42), 0x4000),
            attach_window_request(landing, 0x3000),
        ] {
            assert_ne!(
                wire,
                encode_req(&different),
                "both the founder landing and complete span belong to the identity"
            );
        }
    }

    #[test]
    fn resume_attach_refuses_outstanding_confirm_without_transport_or_reply_consumption() {
        let transport = RecordingTransport::refusing();
        let (mut fs, mut serve) = ring_fs(&transport, SlotRef::from_wire(0x41));
        let peer = kore_rt::svc::PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16)
            .expect("generation-one peer");

        assert_eq!(
            crate::block_on(fs.confirm_retired(peer)),
            Err(CallError::Transport)
        );
        assert_eq!(transport.commits.get(), 1);
        assert_eq!(take_request(&mut serve), FsRequest::ConfirmRetired { peer });
        publish_done(&mut serve);

        assert!(matches!(
            crate::block_on(fs.resume_attach_window(0x4000)),
            Err(RingFsError::Infrastructure(CallError::Outstanding))
        ));
        assert_eq!(
            transport.commits.get(),
            1,
            "a mismatched exact resume must not cross the transport"
        );
        assert!(!fs.is_idle(), "the confirmation must remain outstanding");

        transport.succeed.set(true);
        assert_eq!(
            crate::block_on(fs.confirm_retired(peer)),
            Ok(FsResponse::Done),
            "the matching owner must still consume its queued reply"
        );
        assert_eq!(transport.commits.get(), 2);
        assert!(fs.is_idle());
    }

    #[test]
    fn resume_attach_continues_exact_outstanding_attach() {
        let transport = RecordingTransport::refusing();
        let landing = SlotRef::from_wire(0x41);
        let bytes = 0x4000;
        let (mut fs, mut serve) = ring_fs(&transport, landing);
        let attach = attach_window_request(landing, bytes);

        assert!(matches!(
            crate::block_on(fs.pair.call(&attach)),
            Err(CallError::Transport)
        ));
        assert_eq!(transport.commits.get(), 1);
        assert_eq!(take_request(&mut serve), attach);
        publish_done(&mut serve);

        transport.succeed.set(true);
        assert!(crate::block_on(fs.resume_attach_window(bytes)).is_ok());
        assert_eq!(transport.commits.get(), 2);
        assert!(fs.is_idle());
    }
}

#[cfg(test)]
mod dir_entry_tests {
    use super::*;
    use kore_abi::svc::FsDirEntryWire;
    use zerocopy::IntoBytes as _;

    fn entry(subtype: u16, reserved1: u32) -> FsDirEntryWire {
        let mut name = [0u8; kore_abi::svc::FS_NAME_BYTES];
        name[..4].copy_from_slice(b"tool");
        FsDirEntryWire {
            name_len: 4,
            kind: FsReqWire::KIND_FILE,
            reserved0: subtype.into(),
            reserved1: reserved1.into(),
            object: 9u64.into(),
            name,
        }
    }

    /// The low byte of the entry's reserved word is its subtype; the high
    /// byte and the second reserved word stay reserved.
    #[test]
    fn dir_entry_round_trips_the_subtype_and_refuses_a_high_byte() {
        let stamped = read_dir_entry(entry(1, 0).as_bytes(), 0, 0).unwrap();
        assert_eq!(stamped.subtype, crate::ObjectSubtype::EXECUTABLE_IMAGE);
        assert_eq!(stamped.kind, ObjectKind::File);
        let plain = read_dir_entry(entry(0, 0).as_bytes(), 0, 0).unwrap();
        assert_eq!(plain.subtype, crate::ObjectSubtype::PLAIN);
        assert_eq!(
            read_dir_entry(entry(0x100, 0).as_bytes(), 0, 0).unwrap_err(),
            FsError::Range
        );
        assert_eq!(
            read_dir_entry(entry(0, 1).as_bytes(), 0, 0).unwrap_err(),
            FsError::Range
        );
    }
}
