//! The filesystem request/reply vocabulary — one enum in, one enum out, the
//! shape every `kore-svc-*` protocol takes.
//!
//! These are the values a client builds and the service dispatches. The
//! frozen records they cross the ring as live in `kore_abi::svc`; the codec
//! between the two is [`crate::wire`]. Nothing here carries payload — bulk
//! bytes ride the pair's data window, addressed by offset and length into
//! that window — and no variant carries authority: a handle or grant id is an
//! opaque service-local word that means nothing without the session that
//! owns it, and no kernel capability ever rides a reply.

use kore_abi::svc::{FsRepWire, FsReqWire};
use kore_rt::svc::{PeerBundlePart, PeerId, SvcProtocol};

use crate::{
    AuthorityQuota, FsError, GrantId, Handle, Name, ObjectId, ObjectKind, ObjectSubtype, Revision,
    Rights, SessionId, VersionLease,
};

/// What a client asks the filesystem to do.
///
/// Deliberately carries **no session field**: the engine keys every view on the
/// session derived from the pair that delivered the request, never on wire
/// data. `Grant`'s `destination` is the exception that proves the rule —
/// it is a routing argument the service validates against its own
/// attached-session table, not a claim about who the caller is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsRequest {
    /// Attach the caller's bulk data window: the arena slot its delivered view
    /// landed in, and the view's length.
    AttachWindow {
        slot: u32,
        bytes: u32,
    },
    /// List the authority edges this session holds, into the data window.
    ListGrants {
        window_offset: u32,
    },
    /// Open the calling session's own private root.
    OpenRoot {
        rights: Rights,
    },
    /// Resolve one name component beneath an open directory.
    OpenAt {
        parent: Handle,
        name: Name,
        rights: Rights,
    },
    /// Open a granted file or subtree root.
    OpenGrant {
        grant: GrantId,
        rights: Rights,
    },
    Close {
        handle: Handle,
    },
    Stat {
        handle: Handle,
    },
    /// One page of a directory's entries, landed in the data window.
    List {
        handle: Handle,
        cursor: u32,
        window_offset: u32,
    },
    Create {
        parent: Handle,
        name: Name,
        kind: ObjectKind,
        /// Data stamped at creation; [`ObjectSubtype::PLAIN`] for a directory
        /// and for every ordinary file. An attesting stamp is refused
        /// unless `parent`'s route carries its
        /// [`stamp_rights`](ObjectSubtype::stamp_rights).
        subtype: ObjectSubtype,
    },
    Read {
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
    },
    WriteAt {
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
        expected: Revision,
    },
    /// Reserve a private replacement. One outstanding replacement per session
    /// and volume; begin captures the current revision, even for ANY_REVISION.
    BeginReplace {
        handle: Handle,
        length: u64,
        expected: Revision,
    },
    /// Append the next contiguous chunk to the private replacement.
    WriteReplacement {
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
    },
    /// Publish only a complete replacement with unchanged authority/revision.
    CommitReplace {
        handle: Handle,
    },
    /// Discard private bytes; client death also discards them.
    AbortReplace {
        handle: Handle,
    },
    Append {
        handle: Handle,
        window_offset: u32,
        length: u32,
        expected: Revision,
    },
    Resize {
        handle: Handle,
        length: u64,
        expected: Revision,
    },
    Rename {
        parent: Handle,
        old: Name,
        new: Name,
    },
    /// The plan's `Move`. Named `Relocate` because `move` is a keyword and the
    /// engine's method carries the same name.
    Relocate {
        src_parent: Handle,
        name: Name,
        dst_parent: Handle,
        new_name: Name,
    },
    Link {
        parent: Handle,
        name: Name,
        child: Handle,
    },
    /// Compatibility unlink/rmdir: refuses a populated directory.
    Unlink {
        parent: Handle,
        name: Name,
    },
    /// Native graph operation: remove exactly one containment edge.
    DetachEdge {
        parent: Handle,
        name: Name,
    },
    Grant {
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        display_name: Option<Name>,
    },
    /// Grant with an explicitly attenuated authority-scoped quota.
    GrantScoped {
        handle: Handle,
        destination: SessionId,
        rights: Rights,
        quota: AuthorityQuota,
        display_name: Option<Name>,
    },
    /// Grant a route to the stable task view authenticated by the returned
    /// target endpoint. `announced` is return-post correlation only.
    GrantView {
        handle: Handle,
        rights: Rights,
        display_name: Name,
        announced: kore_abi::session::SlotRef,
    },
    Revoke {
        grant: GrantId,
    },
    Sync {
        handle: Handle,
    },
    /// Hold this session's whole-tree commits until the window closes.
    ///
    /// The durable contract serializes session zero's entire tree after every
    /// mutating request, which is right for interactive work and pathological
    /// for a bulk write: copying one 105 KiB executable through a 16 KiB
    /// window costs seven whole-tree commits, and a boot-time catalogue copy
    /// costs enough of them to outrun init's readiness budget.
    ///
    /// Inside the window, mutations stay in the live engine and no durable
    /// commit runs; closing it commits once. This is a persistence batching
    /// mechanism, not a transaction or visibility fence: another live route
    /// can observe every successful mutation. Bulk replacement must therefore
    /// use a separate session-bound staging operation whose bytes are not
    /// reachable through engine routes; one conditional commit publishes the
    /// complete replacement only after every chunk succeeds. A later
    /// crash/restart recovers the last durable generation, but mere client
    /// death cannot roll back mutations already visible in the live service.
    ///
    /// Deliberately per session, so one bulk writer cannot defer another
    /// client's durability.
    DeferCommits {
        /// `true` opens the window, `false` closes it and commits once.
        hold: bool,
    },
    /// Admin-only testimony that exec is verifying/copying (`active`) or has
    /// completed the whole measured system catalogue (`!active`). The dual
    /// durable router owns the lifecycle; a bare engine refuses this verb.
    SetSystemSeed {
        active: bool,
    },
    /// Freeze the exact revision visible through `handle` and write its
    /// source-neutral identity to the data window.
    LendVersion {
        handle: Handle,
        window_offset: u32,
    },
    /// Freeze an exact executable source revision. This has the same affine
    /// snapshot result as [`LendVersion`](Self::LendVersion), but additionally
    /// requires [`Rights::BIND_EXECUTABLE`] on the route.
    LendExecutable {
        handle: Handle,
        window_offset: u32,
    },
    /// Read bytes through an immutable version lease without granting the
    /// parser any namespace handle.
    ReadVersion {
        lease: VersionLease,
        offset: u64,
        window_offset: u32,
        length: u32,
    },
    /// Validate and commit normalized parser output from the data window.
    CommitDerived {
        lease: VersionLease,
        window_offset: u32,
        length: u32,
        output_window_bytes: u32,
    },
    ReleaseVersion {
        lease: VersionLease,
    },
    DerivedInfo {
        handle: Handle,
        parser: kore_svc_extract::ParserId,
    },
    /// Search authoritative metadata beneath one live authority anchor.
    Search {
        handle: Handle,
        query: crate::SearchQuery,
        cursor: Option<crate::SearchCursor>,
        window_offset: u32,
    },
    /// Bind the authenticated bootstrap endpoint lineage to the durable
    /// system view at an explicitly attenuated native-rights ceiling. The
    /// wire carries return-post correlation and attenuation only; the
    /// administration pair is still the sole caller that may name this root.
    RegisterSystemView {
        rights: Rights,
        announced: kore_abi::session::SlotRef,
    },
    /// Bind an authenticated endpoint lineage to a fresh stable task-private
    /// root. The endpoint is retained as the root's revocation canary.
    ///
    /// `name` is the view's durable home: when the administration workspace
    /// is established, the root delegates `work/<name>` (opened if the
    /// directory already exists, created on a blank tree), which is what
    /// carries one task's files across a reboot without collapsing distinct
    /// tasks into one shared namespace. The registrar names the home; a
    /// second live registration of the same name is refused.
    RegisterPrivateView {
        rights: Rights,
        name: Name,
        announced: kore_abi::session::SlotRef,
    },
    /// Install a structurally presented view as one exact recyclable peer's
    /// root. The peer ticket is lifecycle routing, never authority.
    ApplyView {
        peer: PeerId,
        rights: Rights,
        announced: kore_abi::session::SlotRef,
    },
    /// Bind a caller-minted endpoint lineage to a directory resolved through
    /// this exact task view.
    BindView {
        handle: Handle,
        rights: Rights,
        announced: kore_abi::session::SlotRef,
    },
    /// Land one aggregate diagnostic snapshot ([`FsSystemStatusWire`]) in
    /// the caller's data window: the serving frontend's load beside the
    /// storage backend's posture. Any attached session may ask — the record
    /// carries no name, scope, or per-session detail.
    ///
    /// [`FsSystemStatusWire`]: kore_abi::svc::FsSystemStatusWire
    SystemStatus {
        window_offset: u32,
    },
    /// Administration-pair control: convey one authenticated component of a
    /// new recyclable client face.  `announced` is only the return-post
    /// correlation; the capability itself rides the retained return leg and
    /// is independently species/rights checked by the generic assembler.
    AdmitPeer {
        part: PeerBundlePart,
        announced: kore_abi::session::SlotRef,
    },
    /// Consume the host's completion for the preceding admission sequence.
    /// Valid only on exec's fixed administration pair.
    TakeAdmission,
    /// Retire one exact recyclable generation through the fixed admin pair.
    RetirePeer {
        peer: PeerId,
    },
    /// Confirm that the serving host has completed retirement of one exact
    /// generation.  The initial retire reply precedes the host action; this
    /// second round trip is the lifecycle fence after which the owner may
    /// revoke the pair's local endpoints and backing lease.
    ConfirmRetired {
        peer: PeerId,
    },
    /// Retire the exact recyclable pair on which this request arrived.
    ClosePeer,
    /// Attach this session's storage ring (`res/STORAGE_TRANSPORT.md` §4):
    /// a window of `kore_ring::stor::WINDOW_BYTES` the caller funded,
    /// formatted (`kore_ring::stor::format`) and lent as a view. `announced`
    /// is return-post correlation only; the view rides the retained return
    /// leg. A session holds at most one ring. From here the caller submits
    /// by writing its own window: the pair carries only control.
    StorAttach {
        announced: kore_abi::session::SlotRef,
    },
    /// Install a read grant over the designated pack region into this
    /// session's ring: Demand for reads a thread is blocked on, Stream for
    /// read-ahead. Answers the grant's id and length.
    StorGrantPack,
    /// Adopt one device-mapping fabric route over caller-funded memory as a
    /// target of this session's ring, with [`RING_TARGET_RIGHTS`]'s custody.
    /// Answers the target slot a submission names in `buf_index`.
    ///
    /// [`RING_TARGET_RIGHTS`]: crate::RING_TARGET_RIGHTS
    StorTarget {
        announced: kore_abi::session::SlotRef,
    },
    /// Hand ring target `slot` back. Refused `Busy` while a command uses it;
    /// its device mapping retires once the controller is idle.
    StorRelease {
        slot: u16,
    },
    /// The notice a parked row is owed: the caller published submissions
    /// and its publish found the row's park intent. Answered once the row
    /// has advanced the ring.
    StorKick,
    /// Install a read grant over lent version `lease` into this session's
    /// ring: the version's whole blocks, pinned by the lease for as long as
    /// the lease is held. Answers the grant's id and the bytes it covers; a
    /// partial last block is read through the lease. Releasing the lease is
    /// refused `Busy` while a read under the grant is in flight.
    StorGrantVersion {
        lease: crate::VersionLease,
    },
    /// Read the transport engine's counters: what a stalled wait reports.
    StorInspect,
    /// Detach this session's ring, release its targets and retire it.
    /// Refused `Busy` while commands are still in flight — the ring stops
    /// taking submissions at the first ask; ask again.
    StorDetach,
}

impl FsRequest {
    /// Whether this verb moves bulk data and therefore needs an attached
    /// window.
    pub const fn needs_window(&self) -> bool {
        matches!(
            self,
            FsRequest::Read { .. }
                | FsRequest::WriteAt { .. }
                | FsRequest::WriteReplacement { .. }
                | FsRequest::Append { .. }
                | FsRequest::List { .. }
                | FsRequest::ListGrants { .. }
                | FsRequest::LendVersion { .. }
                | FsRequest::LendExecutable { .. }
                | FsRequest::ReadVersion { .. }
                | FsRequest::CommitDerived { .. }
                | FsRequest::Search { .. }
                | FsRequest::SystemStatus { .. }
        )
    }
}

/// What the filesystem answers.
///
/// **Data only.** No variant carries authority: a handle and a grant id are
/// opaque service-local words that mean nothing without the session that owns
/// them, and no kernel capability ever rides a reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsResponse {
    /// An object was opened or created.
    Opened {
        handle: Handle,
        /// Durable correlation identity. This grants no authority.
        stable_object: crate::StableObjectId,
        kind: ObjectKind,
        subtype: ObjectSubtype,
        len: u64,
        revision: Revision,
    },
    /// Metadata. `object` is safe to expose — an object id conveys no
    /// authority — and is what lets a client tell two names for one object
    /// from two separate objects.
    Stat {
        object: ObjectId,
        stable_object: crate::StableObjectId,
        kind: ObjectKind,
        subtype: ObjectSubtype,
        len: u64,
        revision: Revision,
        links: u32,
        grants: u32,
    },
    /// `entries` directory entries were written to the data window; `next` is
    /// the cursor to resume at, or `None` when the listing is complete.
    Listed {
        entries: u32,
        next: Option<u32>,
    },
    /// `entries` authority edges were written to the data window.
    GrantsListed {
        entries: u32,
    },
    /// One [`FsSystemStatusWire`] record of `len` bytes landed in the data
    /// window.
    ///
    /// [`FsSystemStatusWire`]: kore_abi::svc::FsSystemStatusWire
    SystemStatusReported {
        len: u32,
    },
    /// `count` bytes crossed the data window.
    Transferred {
        count: u32,
    },
    /// A mutation committed, advancing the revision.
    Committed {
        revision: Revision,
    },
    /// An append committed at `offset`.
    Appended {
        offset: u64,
        revision: Revision,
    },
    /// A grant was minted for another session.
    Granted {
        grant: GrantId,
    },
    /// The data window is attached, and this is the largest byte count one
    /// bulk verb may move through it: the backend's own per-request limit
    /// (64 KiB on the durable block provider, the protocol floor on RAM),
    /// folded to the smaller across regions.  A client sizes its chunks from
    /// this rather than from [`bounds::MAX_TRANSFER`](crate::bounds::MAX_TRANSFER),
    /// which stays the floor every backend admits.  Zero means the service
    /// did not say; the floor applies.
    WindowAttached {
        max_transfer: u32,
    },
    /// The operation succeeded and has nothing to report.
    Done,
    VersionLent {
        lease: VersionLease,
    },
    DerivedCommitted {
        parser: kore_svc_extract::ParserId,
        parser_version: u32,
    },
    DerivedInfo {
        input_version: Revision,
        parser_version: u32,
    },
    Searched {
        hits: u32,
        observation: u64,
        snapshot: u64,
        index_generation: u64,
        freshness: crate::FreshnessPolicy,
        next: Option<crate::SearchCursor>,
        scanned: u32,
        work: u32,
    },
    /// The host installed one recyclable pair and published its exact
    /// generation. This ticket is lifecycle routing data, not authority.
    PeerAdmitted {
        peer: PeerId,
    },
    /// The session's ring is attached: the geometry every submission obeys.
    StorAttached {
        /// Logical block size — every offset and length is a multiple.
        block_size: u32,
        /// Largest single device command, in bytes; the row splits longer
        /// submissions itself.
        max_transfer: u32,
    },
    /// A grant was installed into the session's ring. `grant` is the
    /// `kore_abi::stor::GrantId` bit pattern a submission carries in `fd`;
    /// it names the grant only inside that ring.
    StorGranted {
        grant: u32,
        bytes: u64,
    },
    /// A target was adopted into the session's ring as `slot`.
    StorTargetAdopted {
        slot: u16,
        bytes: u64,
    },
    /// The transport engine's counters, every ring together.
    StorInspected {
        /// Submissions popped.
        submissions: u64,
        /// Completions posted.
        completions: u64,
        /// Device commands in flight now.
        in_flight: u32,
        /// Ops reported past their deadline.
        expired: u32,
        /// Device completions naming no live command.
        stale: u32,
        /// Completions a timeline refused (each stops a fence for good).
        timeline_refusals: u32,
        /// Device commands the queue set refused to stage.
        stage_refusals: u32,
        /// Abort commands sent for expired device commands (saturating at
        /// 2^24 − 1 on the wire).
        aborts: u32,
        /// Aborts the controller refused (saturating at 15 on the wire).
        aborts_refused: u8,
        /// The controller arbitrates by weighted round robin.
        weighted: bool,
        /// Rings attached: the row's own lanes and every client's (at most 7
        /// on the wire).
        rings: u8,
        /// The last stage refusal, as `kore_driver::block::BlockError`'s
        /// discriminant plus one; zero for none.
        last_stage_error: u8,
    },
    /// The operation was refused. The engine's vocabulary crosses intact.
    Refused(FsError),
}

/// The DagFS wire protocol, as the runtime's ring client and serving host
/// see it: the request/reply vocabulary, the frozen record forms, the grant
/// budget, the pair count, and the codec that ties a reply to its request.
///
/// A marker rather than the service type, so a client is typed by the
/// *protocol* and never by the engine behind it. The service's own
/// [`SvcProtocol`] impl delegates every item here, which is what keeps the
/// two sides from drifting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DagfsProtocol;

impl SvcProtocol for DagfsProtocol {
    type Request = FsRequest;
    type Response = FsResponse;
    type ReqWire = FsReqWire;
    type RepWire = FsRepWire;

    /// **Zero grants, ever** — structural, not a convention. The filesystem
    /// mints service-local authority only; no kernel capability rides a reply,
    /// so a grant-bearing filesystem reply is a contradiction the decoder also
    /// refuses.
    const REPLY_GRANTS: usize = 0;

    /// Pair zero is the boot/legacy client and pair one is exec's sole
    /// administration face. Additional clients arrive through the runtime's
    /// recyclable-session seam; fixed per-command pairs were a parallel
    /// execution pipeline and made boot topology scale with the command set.
    const SERVE_PAIRS: usize = 2;

    fn reply_matches(request: &FsReqWire, response: &FsResponse, grants: u8) -> bool {
        if grants != 0 {
            return false;
        }
        // The rings are uncorrelated SPSC pairs, so this is what ties a reply
        // to the call in flight. A request record that does not decode cannot
        // have produced any reply.
        match crate::wire::decode_req(request) {
            Some(decoded) => crate::wire::reply_matches(&decoded, response),
            None => false,
        }
    }

    fn encode_req(req: &FsRequest) -> FsReqWire {
        crate::wire::encode_req(req)
    }

    fn decode_req(w: &FsReqWire) -> Option<FsRequest> {
        crate::wire::decode_req(w)
    }

    fn encode_rep(rep: &FsResponse, grants: u8) -> FsRepWire {
        crate::wire::encode_rep(rep, grants)
    }

    fn decode_rep(w: &FsRepWire) -> Option<(FsResponse, u8)> {
        // The uncorrelated decode: without the request there is no way to know
        // whether `count` is a byte count or a link count, so this reports only
        // what is unambiguous and refuses the rest. Callers that hold the
        // request use [`crate::wire::decode_rep_for`], which is what
        // [`FsClient`](crate::client::FsClient) does.
        crate::wire::decode_rep_unambiguous(w)
    }

    fn decode_rep_for(request: &FsReqWire, reply: &FsRepWire) -> Option<(FsResponse, u8)> {
        let request = crate::wire::decode_req(request)?;
        crate::wire::decode_rep_for(&request, reply)
    }
}
