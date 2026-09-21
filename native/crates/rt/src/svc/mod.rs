//! # `svc` — the service request path (`res/USERLAND.md` §5.3)
//!
//! The transport arrangement every client↔service pair shares, made one
//! seam instead of four per-crate copies:
//!
//! - **one mm-leased shared window** holding a request
//!   [`ring::spsc`](kore_ring::spsc) of the service's request wire form
//!   and a reply ring of its reply form (the wire structs live in
//!   [`kore_abi::svc`]; the enum ⇄ wire codecs in each service's
//!   `wire.rs` implement [`SvcProtocol`]);
//! - **a channel pair carrying only notices and capability transfers**:
//!   the client kicks the service's doorbell after a request-ring push,
//!   the service kicks the client's after a reply-ring push, and
//!   grant-bearing replies ride capability transfers into the client's
//!   pre-pinned reply-recv posts (§5.5) — one **grant channel** per
//!   expected grant-carrying send, because the board law is one standing
//!   post per channel (`runtime/ipc/board.rs`), so a multi-grant reply
//!   crosses a *bank of channels*, not a bank of posts.
//!
//! **Payloads ride rings; authority rides rendezvous; nothing else
//! exists.**
//!
//! ## Window layout (frozen within the spsc ABI)
//!
//! ```text
//! offset 0                      : request ring  (spsc<ReqWire>, req_cap)
//! offset region_bytes(req ring) : reply ring    (spsc<RepWire>, rep_cap)
//! ```
//!
//! Both regions start 64-byte aligned ([`kore_ring::spsc::region_bytes`]
//! rounds to the layout unit), so one page-aligned lease window carries
//! the pair.  Each side attaches its half over **its own mapping** of
//! the shared lease ([`client_rings`] / [`serve_rings`]); the host-test
//! form binds both halves over one exclusively-borrowed buffer
//! ([`split_window`], the `spsc::split` idiom).
//!
//! The machines over these rings: [`SvcClient`] (the born-async §5.6
//! call protocol) and [`ServeHost`] (the §5.5 serve loop with its
//! grant-backpressure split) — this module is the geometry + codec
//! seam they (and the founder) share.

use kore_ring::spsc::{self, Consumer, Producer, RingError, RingRecord};

pub use kore_ring::spsc::region_bytes;

mod client;
mod recyclable;
mod serve;

pub use client::{
    AffineBundleCallError, CallError, CallReply, ClientEndpoints, GrantPin, InvocationAuxiliary,
    InvocationAuxiliaryCapability, MAX_INVOCATION_AUXILIARIES, RetainedReturnCallError,
    ReturnCallError, ReturnCapability, ReturnDeriveError, ReturnDeriver, ReturnLeaseSource,
    StagedReturn, StagedReturnError, SvcClient,
};
pub use recyclable::{RecyclableRequest, RecyclableResponse, RecyclableSessionAcceptor};
pub use serve::{
    MAX_MINTED_PAIRS, MAX_SERVE_PAIRS, MintedPair, PeerAdmissionError, PeerAdmissionRefusal,
    PeerAssemblyError, PeerAssemblyRefusal, PeerBundleAssembler, PeerBundlePart, PeerId,
    PeerMapPlan, RecyclablePeerBundle, ServeEndpoints, ServeHost, ServePair, ServeStats,
};
pub(crate) use serve::{ServiceOwner, own_service};

/// The **service-protocol codec seam**: what a service crate's `wire.rs`
/// declares so the generic request path can carry its vocabulary — the
/// request/reply vocabularies, the wire forms ([`kore_abi::svc`]), and
/// the fail-closed enum ⇄ wire codecs.  Implemented on the service type
/// itself (the two sides of one protocol stay one declaration, the
/// `Bundle::PROFILE` discipline).
///
/// **Fail-closed decode is the contract**: an out-of-vocabulary
/// discriminant, refusal byte, or flag pattern decodes to `None` — the
/// peer wrote garbage (or a future revision), and a wrong guess would
/// be a protocol break, not a convenience.
///
/// ## Why this trait stands alone — a wire is not a host
///
/// A protocol is a **wire**: a request/reply vocabulary, the frozen
/// byte forms it lowers to, the grant policy its replies may ride.  A
/// [`SystemService`](crate::service::SystemService) is a **host** that
/// speaks one: it has a name, a typed endowment, a `bind`, worker
/// threads, a serve step.  Those are different facts about different
/// things, and the dependency runs one way only —
/// `SystemService: SvcProtocol`, never the reverse.
///
/// This trait therefore has **no supertrait**, and that is load-bearing
/// rather than tidy.  A [`SvcClient`] needs the wire and nothing else:
/// it encodes requests, decodes replies, and pins the grant bank.  It
/// never binds an endowment, so requiring the host trait of it would
/// demand a [`Bundle`](crate::service::Bundle) — and a bundle is
/// resolved from a [`Bootstrap`](kore_abi::task::Bootstrap) against
/// role slots that a *consumer* of the protocol may not have and may
/// not be able to represent.
///
/// The concrete consumer that forced the split is the **startup
/// linker**: it runs inside a freshly spawned child before any
/// allocator exists there, and it must speak the image protocol to
/// resolve its executable and every `DT_NEEDED` library
/// (`res/IMAGE.md` §4).  It is a client of the image wire and is not,
/// and cannot be, an image *service* — the image host model (its
/// fallible table, its lease bookkeeping) lives behind `heap`, so on
/// that side of the feature the host trait is not implemented at all.
/// With the vocabularies here, the client seam links; with them on the
/// host trait, linking a client would drag in an allocator the child
/// does not have.
///
/// The rule that follows: bound on `SvcProtocol` wherever only wire
/// items are used (the client, the window geometry, the founder's
/// ring formatting), and on `SystemService` wherever the *host* is
/// touched (`bind`, the serve loop, worker spawning).
pub trait SvcProtocol {
    /// The request vocabulary clients — and, later, kernel notifications —
    /// speak to this protocol.  One enum, one intake: the ring muxes
    /// sources, the vocabulary stays closed.
    type Request;

    /// The reply vocabulary.  **Data only** — authority is a delivered
    /// token, never reply payload.
    type Response;

    /// The request wire form riding the request ring.
    type ReqWire: RingRecord;
    /// The reply wire form riding the reply ring.
    type RepWire: RingRecord;

    /// Grant-carrying transfers a conforming reply may ride — the number
    /// of grant channels a client of this service pre-pins (§5.5),
    /// `<=` [`MAX_GRANTS`](crate::service::MAX_GRANTS).  Zero for a
    /// service that can mint no authority (swap).
    const REPLY_GRANTS: usize;

    /// Grant lanes on founder-stamped serving faces. Most protocols expose
    /// their full reply vocabulary there. A service may instead bootstrap a
    /// narrower control face and admit grant-capable data faces afterwards;
    /// its dispatcher must refuse grant-producing verbs on those controls.
    /// Dynamic peer admission and reply validation still use `REPLY_GRANTS`.
    /// Must not exceed `REPLY_GRANTS` (checked by the boot manifest).
    const BOOT_REPLY_GRANTS: usize = Self::REPLY_GRANTS;

    /// Per-protocol bounded ring depth. Large-record data planes may choose a
    /// single-flight pair without forcing every small control protocol to pay
    /// for the same window geometry.
    const REQUEST_CAPACITY: u32 = REQ_CAP;
    const RESPONSE_CAPACITY: u32 = REP_CAP;

    /// Grant channels this exact encoded request can consume.
    ///
    /// The default preserves the fixed-width protocol shape. Protocols whose
    /// verbs have materially different authority results may return a tighter
    /// bound, allowing one generic client to expose the protocol maximum while
    /// a restricted face wires only the channels its reachable verbs use.
    /// The result must be `<= REPLY_GRANTS`; clients validate that bound before
    /// publication and reject replies that exceed it.
    fn request_grants(_request: &Self::ReqWire) -> usize {
        Self::REPLY_GRANTS
    }

    /// **Serving pairs this row advertises** — 1 for the ordinary
    /// one-client shape; a multi-client service (the device broker, whose
    /// every claimant is founded with its own client face) declares how
    /// many pairs its founding stamps, `<=`
    /// [`MAX_SERVE_PAIRS`](crate::svc::MAX_SERVE_PAIRS).
    ///
    /// Pair `k`'s serving roles sit at fixed indices derived from `k`:
    /// ring `SVC_RING@k`, client kick `SVC_SEND@k`, grant channels
    /// `SVC_GRANT@(k·BOOT_REPLY_GRANTS ..)`, return legs `SVC_RETURN@2k/2k+1`.
    /// Pair 0 is exactly the historical single-pair layout.  Each request
    /// is stamped with its arrival pair
    /// ([`Serve::via`](crate::service::Serve::via)) — the §5.8 structural
    /// identity, generalized (`res/DEVICE_SERVICE.md` §10).
    ///
    /// Retained-return services receive an independent landing rotation per
    /// pair, including peers attached after founding. A
    /// [`ReturnMode::Walking`](crate::service::ReturnMode) service must keep
    /// the default because its service-selected landing rotation is pair 0's
    /// reply-and-wait state.
    const SERVE_PAIRS: usize = 1;

    /// Format this protocol's complete shared pair window before either side
    /// maps it.  The default is the canonical request/reply pair.  A protocol
    /// with a byte-only auxiliary plane (the time service's expiration SPSC)
    /// may append it here; authority still moves only through rendezvous.
    ///
    /// This is a protocol hook, not a service-name branch in the founder: the
    /// manifest stores one monomorphized formatter per row.
    fn init_pair_window(window: &mut [u8], req_cap: u32, rep_cap: u32) -> Result<(), RingError>
    where
        Self: Sized,
    {
        init_window::<Self>(window, req_cap, rep_cap)
    }

    /// Optional receiver-authored admission policy for grant channel `index`.
    /// A typed policy is enforced by the rendezvous board before sender
    /// authority moves; `None` preserves the original accept-any post for
    /// protocols whose grant bank carries heterogeneous kinds.
    fn grant_policy(_index: usize) -> Option<(kore_abi::cap::ResourceKind, kore_abi::cap::Rights)> {
        None
    }

    /// Whether grant channel `index` delivers authority that outlives this
    /// call's reusable fixed landing.
    ///
    /// Retained grants land in runtime-reserved affine owners and rotate to a
    /// fresh slot after each consumed delivery. The default keeps the existing
    /// fixed-landing protocol for ephemeral grants whose wrapper possesses or
    /// consumes the record before the next call.
    fn retain_grant(_index: usize) -> bool {
        false
    }

    /// Check that a decoded reply is an answer to the encoded request that is
    /// still outstanding on this uncorrelated SPSC pair.
    ///
    /// Protocols with operation-specific response vocabularies override this
    /// hook. The permissive default preserves existing fixed protocols whose
    /// response enum itself already has only one legal shape.
    fn reply_matches(_request: &Self::ReqWire, _response: &Self::Response, _grants: u8) -> bool {
        true
    }

    /// The request represented solely by a capability delivery on a walking
    /// return post. `None` keeps the ordinary ring-record + doorbell protocol.
    ///
    /// This hook is deliberately closed and static: the generic host invokes
    /// exactly this request only after observing a capability on the current
    /// walking landing; arbitrary return traffic cannot choose an operation.
    fn implicit_return_request() -> Option<Self::Request> {
        None
    }

    /// Whether `request` may use [`implicit_return_request`](Self::implicit_return_request)
    /// instead of publishing a request-ring record and kick. Protocols that
    /// opt in implement both halves so client and host cannot drift.
    fn is_implicit_return_request(_request: &Self::Request) -> bool {
        false
    }

    /// Client side: lower a request to its wire form.
    fn encode_req(req: &Self::Request) -> Self::ReqWire;
    /// Serve side: lift a wire request — `None` refuses (fail-closed).
    fn decode_req(wire: &Self::ReqWire) -> Option<Self::Request>;
    /// Serve side: lower a response + the effected grant count.
    fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire;
    /// Client side: lift a wire reply + its grant count — `None` refuses.
    fn decode_rep(wire: &Self::RepWire) -> Option<(Self::Response, u8)>;

    /// Client side: lift a reply against the request still outstanding on
    /// this uncorrelated SPSC pair.
    ///
    /// Most protocols have self-describing replies and inherit the ordinary
    /// decoder. Compact protocols may reuse fields across verbs; they must
    /// override this hook rather than guess a response shape without the
    /// request that gives those fields meaning.
    fn decode_rep_for(
        _request: &Self::ReqWire,
        wire: &Self::RepWire,
    ) -> Option<(Self::Response, u8)> {
        Self::decode_rep(wire)
    }
}

/// **The rights a founder-carved `ChanSend` carries** — every send end
/// `stage_pair` mints for a pair: the client's kick, the service's
/// kick-back, each grant channel's, the return leg's, and the service's
/// own **self-kick** (`role::DOORBELL@2`).
///
/// `ACCESS` to post on it; `DERIVE` so its holder can attenuate it
/// further. The second one is not decoration and is why this is a named
/// constant rather than a literal in the founder: a service endows a
/// **worker thread** with a send end into its own doorbell by deriving
/// off the self-kick, and Σ5 clamps that child to a subset of *this*
/// set. So the number is a ceiling read by code the founder does not
/// know about, and a literal at one end could drift past the other's
/// silently — the derive would answer `Denied` and the worker would
/// simply never spawn.
///
/// The sibling `recv` / `mint` rights stay literals inside `stage_pair`:
/// nothing outside the founder derives off those.
pub const SEND_RIGHTS: kore_abi::cap::Rights =
    kore_abi::cap::Rights::ACCESS.union(kore_abi::cap::Rights::DERIVE);

/// Doorbell-notice cookie: "your **request** ring has news" (client →
/// service kick).  A hint — the §3.1 drain is total.
pub const COOKIE_REQUEST: u64 = kore_abi::session::NOTICE_COOKIE_SERVICE_REQUEST;

/// Doorbell-notice cookie: "your **reply** ring has news" (service →
/// client kick).
pub const COOKIE_REPLY: u64 = 0x6B72_5F73_7663_5250; // "kr_svcRP"

/// Doorbell-notice cookie: "your **data-plane** queue has news" (producer
/// service → consumer kick).  Advisory and droppable — the queue is the
/// truth, the notice only wakes.  Sent at most **once per drained batch**,
/// never per record (`res/USERLAND.md` §7's bare-notice exception): the HID
/// router's report fan-out is its first sender.
pub const COOKIE_REPORT: u64 = 0x6B72_5F73_7663_5254; // "kr_svcRT"

/// Doorbell-notice cookie: **"my endowment is bound and I am parked"** — a
/// founded, non-serving service's ready edge.  Like [`COOKIE_SERVING`], this
/// travels over the founder-delivered `role::PARENT_SEND` endpoint and is
/// observed only after the service's executor has established its standing
/// doorbell post.  It replaces the former raw `SERVICE_READY` supervisor-call
/// discriminant: readiness now requires possessed channel authority and an
/// ordinary session commit.
pub const COOKIE_READY: u64 = 0x6B72_5F73_7663_5244; // "kr_svcRD"

/// Founded accelerator client completed three owned 16 MiB attach/detach cycles.
pub const COOKIE_ACCEL_STAGING: u64 = 0x6B72_5F61_7374_6721; // "kr_astg!"
pub const ACCEL_STAGING_VALUE: u64 = 3;
/// Owned import disposition: 0=unavailable/refusal cleanup, 1=admitted,
/// 2=disabled, 3=unavailable with pre-transfer cancellation recovery,
/// 4=unavailable with post-transfer cancellation and exact resumption,
/// 5=the same plus authenticated lease hand-back, contents and reuse.
pub const COOKIE_ACCEL_IMPORT: u64 = 0x6B72_5F61_696D_7021;
/// Founded native peer close followed by two same-cell generation replacements.
pub const COOKIE_ACCEL_PEERS: u64 = 0x6B72_5F61_7065_7221; // "kr_aper!"

/// Founded executable fabric route: two real CPU children, retryable close,
/// exact sibling preservation, and released exclusion. No device DMA claim.
pub const COOKIE_ACCEL_FABRIC: u64 = 0x6B72_5F61_6661_6221; // "kr_afab!"

/// Bounded numerical-capture stream on init's existing parent channel. Word
/// V2 ordinals encode graph, artifact and input hashes (32 bytes each), 32
/// output bytes and a 32-byte output hash. All hashes are BLAKE3; words are LE.
pub const COOKIE_ACCEL_CAPTURE_WORD_BASE: u64 = 0x6B72_6361_7000_0000;
pub const ACCEL_CAPTURE_WORDS: u64 = (32 + 32 + 32 + 32 + 32) / 8;
/// Overlap V1: five BLAKE3 identities followed by three rounds, each with
/// pending/cancel/terminal flags, two event IDs and two canonical output hashes.
pub const ACCEL_OVERLAP_WORDS: u64 = 20 + 3 * 11;
pub const ACCEL_CAPTURE_MAX_WORDS: u64 = ACCEL_OVERLAP_WORDS;

/// Versioned capture framing on the existing founded parent channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelCaptureKind {
    Regions,
    Overlap,
}
impl AccelCaptureKind {
    pub const fn words(self) -> u64 {
        match self {
            Self::Regions => ACCEL_CAPTURE_WORDS,
            Self::Overlap => ACCEL_OVERLAP_WORDS,
        }
    }
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Regions => "ivpu-capture: V2",
            Self::Overlap => "ivpu-overlap: V1",
        }
    }
    /// Upper 16 bits select framing; lower 48 preserve PCI vendor/device.
    pub const fn begin(self, vendor: u16, device: u32) -> u64 {
        let tag = match self {
            Self::Regions => 0,
            Self::Overlap => 1,
        };
        (tag << 48) | ((vendor as u64) << 32) | device as u64
    }
    pub const fn decode_begin(payload: u64) -> Option<(Self, u64)> {
        let kind = match payload >> 48 {
            0 => Self::Regions,
            1 => Self::Overlap,
            _ => return None,
        };
        Some((kind, payload & 0x0000_ffff_ffff_ffff))
    }
}

pub const COOKIE_ACCEL_CAPTURE_BEGIN: u64 = COOKIE_ACCEL_CAPTURE_WORD_BASE + 0x1000;
pub const COOKIE_ACCEL_CAPTURE_END: u64 = COOKIE_ACCEL_CAPTURE_WORD_BASE + 0x1001;
pub const COOKIE_ACCEL_CAPTURE_SKIPPED: u64 = COOKIE_ACCEL_CAPTURE_WORD_BASE + 0x1002;
/// Init admitted the real ax/compiler fixture and refused strict F32 policy.
pub const COOKIE_ACCEL_GRAPH_ADMITTED: u64 = COOKIE_ACCEL_CAPTURE_WORD_BASE + 0x1003;

/// Founded ring-3 witness that ordinary MM memory completed the consuming
/// writable-to-executable lease transition and executed through a fresh RX
/// mapping. The payload is [`EXEC_FINALIZE_VALUE`].
pub const COOKIE_EXEC_FINALIZED: u64 = 0x6B72_5F78_6669_6E21; // "kr_xfin!"

/// Kernel-founder → init notice carrying the immutable terminal-seat plan.
/// Init consumes this notice before changing the same private configuration
/// channel from notice reception to affine capability reception.
pub const COOKIE_INIT_CONFIG_PLAN: u64 = 0x4b4f_5245_4346_4701; // "KORECFG\x01"

/// Init → kernel-founder acknowledgement that the capability-only receiver's
/// next landing is standing. The payload is the zero-based resource sequence;
/// the founder transfers exactly that member only after observing this edge.
pub const COOKIE_INIT_CONFIG_POSTED: u64 = 0x4b4f_5245_4346_4702; // "KORECFG\x02"

/// Shell → init acknowledgement that the shell's next capability-only
/// configuration landing is standing. The payload is the zero-based
/// handshake sequence. Init transfers the corresponding member only after
/// observing this edge, so progress never depends on scheduler timing.
pub const COOKIE_SHELL_CONFIG_POSTED: u64 = 0x4b4f_5245_5343_4601; // "KORESCF\x01"

/// Number of affine configuration members in Seat A's package. Seat A owns
/// the #488 hypervisor console ring and kick in addition to the common shell
/// resources (including its self-kick, plus vm 2). System power is owned by exec and never enters a
/// shell package.
pub const INIT_SHELL_CONFIG_PRIMARY_RESOURCES: u64 = INIT_SHELL_CONFIG_RESOURCES + 2;

/// Number of affine configuration members in a sibling seat's package.
/// Sibling seats omit the primary seat's hypervisor console.
pub const INIT_SHELL_CONFIG_RESOURCES: u64 = 14;

/// A shell observed its separately founded Axiom presenter through compositor
/// status. Low 32 bits: presented epoch; bits 32..47: terminal; bit 63: the
/// two-terminal acceptance profile also proved close/reopen with fresh IDs.
pub const COOKIE_TERMINAL_APP: u64 = 0x6b72_5f61_7874_6572; // "kr_axter"

/// Return value of the position-independent function used by the executable
/// finalization boot witness.
pub const EXEC_FINALIZE_VALUE: u64 = 0x4b4f;

/// Doorbell-notice cookie: **"I am serving"** — the founded service's ready
/// edge, sent once on its [`role::PARENT_SEND`](kore_abi::task::role::PARENT_SEND)
/// beacon just before it enters [`ServeHost::run`] (Phase-A checkpoint G2a-3a).
/// Serving readiness is a distinct cookie because it promises the ring and
/// `ServeHost` are live; [`COOKIE_READY`] promises only a bound endowment and
/// lawful executor park.  Both are real capability deliveries.
pub const COOKIE_SERVING: u64 = 0x6B72_5F73_7663_5259; // "kr_svcRY"

/// A service **refused** during its bring-up ladder — the NACK twin of
/// [`COOKIE_SERVING`], sent best-effort on the same `role::PARENT_SEND`
/// endpoint with the failed rung's code as the payload.  Before this, a
/// refusal was an infinite silent spin: the founder burned its whole
/// readiness budget (120 s) against a service that already knew it was
/// dead, and on a serial-less machine the *reason* never existed anywhere.
/// A founder that sees this cookie fails the row immediately and prints
/// the rung.
pub const COOKIE_REFUSED: u64 = 0x6B72_5F73_7663_5246; // "kr_svcRF"

/// Completion cookie for the first shell (`"kr_shell"`).
///
/// The founded shell sends this to its founder twice: once with
/// [`SHELL_READY`] after both service pairs are wired and both queue
/// attaches acked (the banner is on its way), and once with
/// [`SHELL_FIRST_LINE`] after the first canonical line crossed the
/// input queue.  Partial progress is deliberately observable: a shell
/// that never beacons refused somewhere in its fail-closed ladder.
pub const COOKIE_SHELL: u64 = 0x6B72_5F73_6865_6C6C; // "kr_shell"

// The service-protocol cookie run must not collide with a cookie the
// KERNEL reserves: the session engine refuses a notice-mode send naming
// `NOTICE_COOKIE_PAGER_FAULT` or `NOTICE_COOKIE_POWER_QUIESCE`, so a service
// cookie equal to either would make that service's kicks unsendable — a
// puzzling runtime refusal where this is a build error.  (The two namespaces are deliberately distinct —
// `kr_svc*` vs `kore*` — and this is what keeps them so.)
/// Is `cookie` one of the words the session engine refuses on a userland
/// notice-mode send?  The pager's fault notice and the terminal-quiesce ask
/// (#809 §L) — both `kore*`, neither ever a `kr_svc*` service cookie.
const fn kernel_reserved(cookie: u64) -> bool {
    cookie == kore_abi::session::NOTICE_COOKIE_PAGER_FAULT
        || cookie == kore_abi::session::NOTICE_COOKIE_POWER_QUIESCE
}

const _: () = {
    assert!(!kernel_reserved(COOKIE_REQUEST));
    assert!(!kernel_reserved(COOKIE_REPLY));
    assert!(!kernel_reserved(COOKIE_REPORT));
    assert!(!kernel_reserved(COOKIE_READY));
    assert!(!kernel_reserved(COOKIE_EXEC_FINALIZED));
    assert!(!kernel_reserved(COOKIE_ACCEL_STAGING));
    assert!(!kernel_reserved(COOKIE_ACCEL_PEERS));
    assert!(!kernel_reserved(COOKIE_ACCEL_FABRIC));
    assert!(!kernel_reserved(COOKIE_ACCEL_IMPORT));
    assert!(ACCEL_CAPTURE_MAX_WORDS < 0x1000);
    assert!(!kernel_reserved(COOKIE_ACCEL_CAPTURE_BEGIN));
    assert!(!kernel_reserved(COOKIE_ACCEL_CAPTURE_END));
    assert!(!kernel_reserved(COOKIE_ACCEL_CAPTURE_SKIPPED));
    assert!(!kernel_reserved(COOKIE_ACCEL_GRAPH_ADMITTED));
    assert!(!kernel_reserved(COOKIE_INIT_CONFIG_PLAN));
    assert!(!kernel_reserved(COOKIE_INIT_CONFIG_POSTED));
    assert!(!kernel_reserved(COOKIE_SHELL_CONFIG_POSTED));
    assert!(COOKIE_INIT_CONFIG_PLAN != COOKIE_INIT_CONFIG_POSTED);
    assert!(COOKIE_INIT_CONFIG_PLAN != COOKIE_SHELL_CONFIG_POSTED);
    assert!(COOKIE_INIT_CONFIG_POSTED != COOKIE_SHELL_CONFIG_POSTED);
    assert!(!kernel_reserved(COOKIE_SERVING));
    assert!(!kernel_reserved(COOKIE_REFUSED));
    assert!(!kernel_reserved(COOKIE_SHELL));
    assert!(!kernel_reserved(COOKIE_STORE));
    assert!(!kernel_reserved(COOKIE_SPMD));
    assert!(!kernel_reserved(COOKIE_LUA));
    assert!(!kernel_reserved(COOKIE_LUA_REPL));
    assert!(!kernel_reserved(COOKIE_MPY));
    assert!(COOKIE_LUA_REPL != COOKIE_LUA);
    assert!(!kernel_reserved(COOKIE_TERMINAL_APP));
    assert!(COOKIE_TERMINAL_APP != COOKIE_SHELL);
    assert!(COOKIE_MPY != COOKIE_LUA);
    assert!(COOKIE_MPY != COOKIE_LUA_REPL);
};

/// Shell witness payload: wired, attached, and serving the prompt.
pub const SHELL_READY: u64 = 0x1;

/// INIT readiness payload bit: a separately founded `find --text answer`
/// process completed through the ordinary exec + filesystem capability path.
/// Completion means either a successful reindex of an empty/supported tree or
/// the exact `IndexIncomplete` result for a tree containing an unsupported
/// version; those are the two content-dependent healthy outcomes. The founder
/// consumes this bit when it drains INIT's existing terminal [`COOKIE_READY`]
/// notice; it is not a second rendezvous lifecycle.
pub const INIT_FS_SEARCH_READY: u64 = 1 << 63;

/// The serving beacon's **testimony** vocabulary, re-exported from the boot
/// page's own decoder ring ([`kore_abi::founding`]).
///
/// A row may OR these into its readiness payload for facts only it can
/// establish — a driver that proved its own datapath before it beaconed.  They
/// are testimony, not a note: the founder prints each one and masks it out of
/// the *console* line (so the readiness marker stays byte-identical for every
/// row that has nothing to say), but keeps it in the row's durable founding
/// note, which is where a machine with no serial cable reads the proof —
/// `status devices`. Same shape as [`INIT_FS_SEARCH_READY`],
/// one row over; that one is consumed by the founder and never reaches the
/// ledger.
///
/// They live in `kore-abi` because the founding block stores them; they are
/// re-exported here because the services that SET them speak `kore-rt`.
pub use kore_abi::founding::{
    NET_FRAME_RT_READY, NET_LOAD_READY, NET_PCI_FRAME_READY, NET_VIF_ISOLATION_READY,
    NET_WIFI_FW_ALIVE, SERVING_WITNESS_BITS,
};

/// INIT's control-plane bit and the durable testimony carve must not overlap:
/// the founder strips [`INIT_FS_SEARCH_READY`] on the way in and keeps
/// [`SERVING_WITNESS_BITS`] all the way to the boot page, so a shared bit would
/// make one silently erase the other on the one row that carries both.
const _: () = assert!(
    INIT_FS_SEARCH_READY & SERVING_WITNESS_BITS == 0,
    "INIT's readiness bit collides with the durable testimony carve"
);

/// Shell witness payload bit: the first completed line arrived — its
/// stripped byte length rides `payload >> 8`.
pub const SHELL_FIRST_LINE: u64 = 0x2;

/// Shell bootstrap witness: the complete task-local executable namespace was
/// consumed and the runtime-owned landing for the following configuration
/// capability is already standing. Init consumes this control-plane bit
/// locally; it is the ordering edge between namespace publication and the
/// remaining configuration publication, not shell readiness testimony for the
/// kernel founder.
pub const SHELL_NAMESPACE_READY: u64 = 0x80;

/// Shell witness payload bit (posted by INIT, not the shell): the
/// shell's FOUNDING failed — the rung of init's fail-closed ladder that
/// refused rides `payload >> 8` (1 faces, 2 plan absent, 3 acquire
/// refused, 4 convey refused, 5 spawn refused, 6 start refused,
/// 7 shell-READY timeout, 8 NACK/early exit).  The kernel's drain prints
/// the rung loudly; init itself still beacons `COOKIE_READY` after
/// posting it — the init ROW came up, it is the shell that did not.
pub const SHELL_FOUNDING_FAILED: u64 = 0x20;

/// Marker in the 56-bit founding-failure rung (`COOKIE_SHELL.payload >> 8`)
/// indicating that the rung also carries an init phase and the exact exec
/// refusal code. Legacy rungs leave this bit clear and retain their existing
/// spelling.
pub const SHELL_FAILURE_DETAIL: u64 = 1 << 55;
/// Marker in the 56-bit founding-failure rung indicating that the rung also
/// carries the exact shell-configuration handshake sequence, stage, and
/// transfer status. This is disjoint from [`SHELL_FAILURE_DETAIL`], which is
/// reserved for exec refusals.
pub const SHELL_CONFIG_FAILURE_DETAIL: u64 = 1 << 54;
/// A terminal presentation failure. Unlike the historical `8000 + error` /
/// `8100 + error` spelling, the tagged form records which launch failed: an
/// initial exit 116 and a recycled exit 16 otherwise produce the same rung.
pub const SHELL_TERMINAL_FAILURE_DETAIL: u64 = 1 << 53;
const SHELL_TERMINAL_RECYCLED: u64 = 1 << 52;
const SHELL_TERMINAL_LEGACY_MASK: u64 = SHELL_TERMINAL_RECYCLED - 1;
/// Low field of a structured founding-failure rung: the historical rung that
/// keeps old logs and acceptance tooling comparable.
pub const SHELL_FAILURE_LEGACY_MASK: u64 = u32::MAX as u64;

/// Init timed out or rejected testimony while waiting for a posted landing.
pub const SHELL_CONFIG_STAGE_POST: u8 = 1;
/// An exact affine move failed while source custody was still proven local.
pub const SHELL_CONFIG_STAGE_EXACT_NOT_MOVED: u8 = 2;
/// An exact affine move failed after custody could no longer be reconstructed.
pub const SHELL_CONFIG_STAGE_EXACT_AFTER_TRANSFER: u8 = 3;
/// A fixed delegated package member failed after its first posted landing.
pub const SHELL_CONFIG_STAGE_PACKAGE: u8 = 4;
/// A derived channel-space move failed.
pub const SHELL_CONFIG_STAGE_DERIVED: u8 = 5;
/// No transport status exists because the userland trap could not commit.
pub const SHELL_CONFIG_ERROR_TRANSPORT: u8 = 0xfe;

/// The terminal renderer's closed failure phases. `exit_code()` assigns each
/// phase an application-reserved slot on the ordinary task-exit path; the
/// enum discriminants are phase identities, not the wire exit numbers. The
/// shared decoder keeps photographed failures attributable to their phase.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalRendererFailure {
    Bootstrap = 16,
    SurfaceRequest = 17,
    SurfaceMap = 18,
    Application = 19,
    Scratch = 20,
    WorkerAdmission = 21,
    Render = 22,
    Commit = 23,
    TargetGeometry = 24,
}

impl TerminalRendererFailure {
    pub const fn exit_code(self) -> crate::exit::ExitCode {
        match self {
            Self::Bootstrap => crate::ExitCode::APPLICATION_0,
            Self::SurfaceRequest => crate::ExitCode::APPLICATION_1,
            Self::SurfaceMap => crate::ExitCode::APPLICATION_2,
            Self::Application => crate::ExitCode::APPLICATION_3,
            Self::Scratch => crate::ExitCode::APPLICATION_4,
            Self::WorkerAdmission => crate::ExitCode::APPLICATION_5,
            Self::Render => crate::ExitCode::APPLICATION_6,
            Self::Commit => crate::ExitCode::APPLICATION_7,
            Self::TargetGeometry => crate::ExitCode::APPLICATION_8,
        }
    }

    pub const fn from_exit_code(code: i32) -> Option<Self> {
        match code {
            x if x == Self::Bootstrap.exit_code().get() => Some(Self::Bootstrap),
            x if x == Self::SurfaceRequest.exit_code().get() => Some(Self::SurfaceRequest),
            x if x == Self::SurfaceMap.exit_code().get() => Some(Self::SurfaceMap),
            x if x == Self::Application.exit_code().get() => Some(Self::Application),
            x if x == Self::Scratch.exit_code().get() => Some(Self::Scratch),
            x if x == Self::WorkerAdmission.exit_code().get() => Some(Self::WorkerAdmission),
            x if x == Self::Render.exit_code().get() => Some(Self::Render),
            x if x == Self::Commit.exit_code().get() => Some(Self::Commit),
            x if x == Self::TargetGeometry.exit_code().get() => Some(Self::TargetGeometry),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::SurfaceRequest => "surface-request",
            Self::SurfaceMap => "surface-map",
            Self::Application => "application",
            Self::Scratch => "scratch",
            Self::WorkerAdmission => "worker-admission",
            Self::Render => "render",
            Self::Commit => "commit",
            Self::TargetGeometry => "target-geometry",
        }
    }
}

/// Tag the shell's historical presentation rung without changing its value
/// in the low field. `legacy` is `8000 + SessionError.0` for the initial
/// renderer or `8100 + SessionError.0` for its acceptance-profile replacement.
/// Bit 52 records that distinction independently of the error, including
/// signed/unknown task exit statuses. All fields fit the 56-bit rung carried
/// by `COOKIE_SHELL.payload >> 8`.
pub const fn shell_terminal_failure_rung(legacy: u64, recycled: bool) -> u64 {
    SHELL_TERMINAL_FAILURE_DETAIL
        | if recycled { SHELL_TERMINAL_RECYCLED } else { 0 }
        | (legacy & SHELL_TERMINAL_LEGACY_MASK)
}

/// Return `(legacy rung, recycled renderer, SessionError.0)` without treating
/// an unrelated numeric or structured failure as a terminal exit. Historical
/// renderers used only `ExitCode::IO`, so those two exact legacy spellings
/// remain readable; other untagged numbers are deliberately not guessed.
pub const fn shell_terminal_failure_detail(rung: u64) -> Option<(u64, bool, u64)> {
    const LEGACY_IO: u64 = (1 << 32) | 2;
    if rung & (SHELL_FAILURE_DETAIL | SHELL_CONFIG_FAILURE_DETAIL) != 0 {
        return None;
    }
    if rung & SHELL_TERMINAL_FAILURE_DETAIL == 0 {
        return match rung {
            x if x == 8000 + LEGACY_IO => Some((rung, false, LEGACY_IO)),
            x if x == 8100 + LEGACY_IO => Some((rung, true, LEGACY_IO)),
            _ => None,
        };
    }
    let legacy = rung & SHELL_TERMINAL_LEGACY_MASK;
    let recycled = rung & SHELL_TERMINAL_RECYCLED != 0;
    let base = if recycled { 8100 } else { 8000 };
    match legacy.checked_sub(base) {
        Some(error) if error >> 32 <= 1 => Some((legacy, recycled, error)),
        _ => None,
    }
}

/// Pack an actionable exec refusal into init's existing failure witness.
///
/// Layout within `payload >> 8`: bit 55 detail marker, bits 40..47 exec's
/// stable `TaskRepWire::REFUSE_*` code, bits 32..39 init phase, and bits
/// 0..31 the legacy rung. The remaining bits stay zero for future detail.
pub const fn shell_exec_failure_rung(legacy: u64, phase: u8, refusal: u8) -> u64 {
    SHELL_FAILURE_DETAIL
        | ((refusal as u64) << 40)
        | ((phase as u64) << 32)
        | (legacy & SHELL_FAILURE_LEGACY_MASK)
}

/// Decode the structured exec-refusal extension, retaining a clean fallback
/// for every historical image that posts only a numeric rung.
pub const fn shell_exec_failure_detail(rung: u64) -> Option<(u64, u8, u8)> {
    if rung & SHELL_FAILURE_DETAIL == 0 {
        None
    } else {
        Some((
            rung & SHELL_FAILURE_LEGACY_MASK,
            ((rung >> 32) & 0xff) as u8,
            ((rung >> 40) & 0xff) as u8,
        ))
    }
}

/// Pack an actionable shell configuration failure into init's existing
/// founding witness.
///
/// Layout within `payload >> 8`: bit 54 detail marker, bits 48..53 stage,
/// bits 40..47 status/error, bits 32..39 handshake sequence, and bits 0..31
/// the historical rung. Stage values are kept below 64 so their high two bits
/// cannot overlap either structured-detail marker.
pub const fn shell_config_failure_rung(legacy: u64, sequence: u8, stage: u8, error: u8) -> u64 {
    SHELL_CONFIG_FAILURE_DETAIL
        | (((stage & 0x3f) as u64) << 48)
        | ((error as u64) << 40)
        | ((sequence as u64) << 32)
        | (legacy & SHELL_FAILURE_LEGACY_MASK)
}

/// Decode the structured configuration extension while retaining a clean
/// fallback for historical numeric and structured exec rungs.
pub const fn shell_config_failure_detail(rung: u64) -> Option<(u64, u8, u8, u8)> {
    if rung & SHELL_CONFIG_FAILURE_DETAIL == 0 || rung & SHELL_FAILURE_DETAIL != 0 {
        None
    } else {
        Some((
            rung & SHELL_FAILURE_LEGACY_MASK,
            ((rung >> 32) & 0xff) as u8,
            ((rung >> 48) & 0x3f) as u8,
            ((rung >> 40) & 0xff) as u8,
        ))
    }
}

#[cfg(test)]
mod shell_failure_tests {
    use super::{
        SHELL_CONFIG_ERROR_TRANSPORT, SHELL_CONFIG_STAGE_EXACT_AFTER_TRANSFER,
        SHELL_FOUNDING_FAILED, TerminalRendererFailure, shell_config_failure_detail,
        shell_config_failure_rung, shell_exec_failure_detail, shell_exec_failure_rung,
        shell_terminal_failure_detail, shell_terminal_failure_rung,
    };

    #[test]
    fn structured_exec_failure_retains_legacy_phase_and_refusal() {
        let rung = shell_exec_failure_rung(577, 2, 17);
        assert_eq!(shell_exec_failure_detail(rung), Some((577, 2, 17)));
        assert_eq!(shell_exec_failure_detail(565), None);
    }

    #[test]
    fn structured_config_failure_retains_legacy_sequence_stage_and_error() {
        let rung = shell_config_failure_rung(
            10_102,
            2,
            SHELL_CONFIG_STAGE_EXACT_AFTER_TRANSFER,
            SHELL_CONFIG_ERROR_TRANSPORT,
        );
        assert_eq!(
            shell_config_failure_detail(rung),
            Some((
                10_102,
                2,
                SHELL_CONFIG_STAGE_EXACT_AFTER_TRANSFER,
                SHELL_CONFIG_ERROR_TRANSPORT,
            ))
        );
        assert_eq!(shell_exec_failure_detail(rung), None);
        assert_eq!(shell_config_failure_detail(565), None);
        assert_eq!(
            shell_config_failure_detail(shell_exec_failure_rung(577, 2, 17)),
            None
        );
    }

    #[test]
    fn renderer_failure_codes_are_stable_program_exits() {
        for (phase, code, name) in [
            (TerminalRendererFailure::Bootstrap, 256, "bootstrap"),
            (
                TerminalRendererFailure::SurfaceRequest,
                257,
                "surface-request",
            ),
            (TerminalRendererFailure::SurfaceMap, 258, "surface-map"),
            (TerminalRendererFailure::Application, 259, "application"),
            (TerminalRendererFailure::Scratch, 260, "scratch"),
            (
                TerminalRendererFailure::WorkerAdmission,
                261,
                "worker-admission",
            ),
            (TerminalRendererFailure::Render, 262, "render"),
            (TerminalRendererFailure::Commit, 263, "commit"),
            (
                TerminalRendererFailure::TargetGeometry,
                264,
                "target-geometry",
            ),
        ] {
            assert_eq!(phase.exit_code().get(), code);
            assert_eq!(TerminalRendererFailure::from_exit_code(code), Some(phase));
            assert_eq!(phase.name(), name);
        }
        for code in [i32::MIN, -1, 0, 2, 15, 25, 116, 125, 134] {
            assert_eq!(TerminalRendererFailure::from_exit_code(code), None);
        }
    }

    #[test]
    fn terminal_failures_preserve_signed_unknown_exits_and_launch_stage() {
        for recycled in [false, true] {
            let base = if recycled { 8100 } else { 8000 };
            for status in [i32::MIN, -1, 0, 2, 16, 20, 116, 125, 134, i32::MAX] {
                let error = (1u64 << 32) | u64::from(status as u32);
                let legacy = base + error;
                let rung = shell_terminal_failure_rung(legacy, recycled);
                let payload = SHELL_FOUNDING_FAILED | (rung << 8);
                assert_eq!(payload >> 8, rung, "rung must fit its wire field");
                assert_eq!(
                    shell_terminal_failure_detail(payload >> 8),
                    Some((legacy, recycled, error))
                );
                assert_eq!(shell_exec_failure_detail(rung), None);
                assert_eq!(shell_config_failure_detail(rung), None);
                assert_eq!(error as u32 as i32, status);
            }
            let rung = shell_terminal_failure_rung(base + 27, recycled);
            assert_eq!(
                shell_terminal_failure_detail(rung),
                Some((base + 27, recycled, 27))
            );
        }
        let initial = shell_terminal_failure_rung(8000 + (1 << 32) + 116, false);
        let recycled = shell_terminal_failure_rung(8100 + (1 << 32) + 16, true);
        assert_ne!(
            initial, recycled,
            "launch stage must not alias an exit code"
        );
        assert_eq!(
            shell_terminal_failure_detail(initial).unwrap().2 as u32,
            116
        );
        assert_eq!(
            shell_terminal_failure_detail(recycled).unwrap().2 as u32,
            16
        );
    }

    #[test]
    fn terminal_failure_decoder_does_not_guess_other_rung_vocabularies() {
        let io = (1 << 32) | 2;
        assert_eq!(
            shell_terminal_failure_detail(4_294_975_298),
            Some((8000 + io, false, io))
        );
        assert_eq!(
            shell_terminal_failure_detail(8100 + io),
            Some((8100 + io, true, io))
        );
        for rung in [
            0,
            7,
            8027,
            8000 + (1 << 32) + 20,
            shell_exec_failure_rung(577, 2, 17),
            shell_config_failure_rung(10_102, 2, 3, 0xfe),
            shell_terminal_failure_rung(1, false),
        ] {
            assert_eq!(shell_terminal_failure_detail(rung), None);
        }
    }
}

/// Completion cookie for the durable-store seed pass (`"kr_store"`),
/// posted by INIT after readiness. The payload carries the verdict:
/// bits 0..7 the catalogue size, bits 8..15 how many members answered
/// seeded. Every per-member failure beneath it is swallowed by design
/// (provisioning must not cost a machine that already booted), so this
/// summary is the ONLY place a dead seed path and a working one differ
/// — the boot gates key on it. A distinct cookie rather than a
/// `COOKIE_SHELL` flag bit: the shell's flag byte is fully claimed
/// (its image-local witness bits 0x4..0x40 ride READY and launch
/// testimony), and a seed verdict is not shell testimony.
pub const COOKIE_STORE: u64 = 0x6B72_5F73_746F_7265; // "kr_store"

/// First executable seed failure, separate from the full store census payload.
pub const COOKIE_STORE_APP_FAILURE: u64 = 0x6B72_5F73_746F_7246; // "kr_storF"

/// One init phase boundary (`"kr_initP"`): `[7:0]` the phase ordinal,
/// `[63:8]` init's committed session clock in milliseconds. Posted at
/// every step of the bring-up ladder so a boot log attributes the
/// welcome-to-dock interval to the phase that spent it — the kernel
/// stamps only founding/readiness as a whole, and a serial-less machine
/// shows nothing at all (#1003).
pub const COOKIE_INIT_PHASE: u64 = 0x6B72_5F69_6E69_7450; // "kr_initP"

/// One step of init's bring-up that is finer than a phase (`"kr_initS"`):
/// `[7:0]` the step's index, `[15:8]` its catalogue size (kind 0 only),
/// `[23:16]` the kind, `[63:24]` init's committed session clock in
/// milliseconds. Kind 0 is a bootstrap store member about to be installed
/// (index into `BOOTSTRAP_EXECUTABLES`); kind 1 is one of the named calls
/// between the `installed` and `extraction` phase stamps (the kernel's
/// `init_step_name` ladder). Posted before each such call because every one
/// of them is a deadline-less exec or mm call, and a boot that stalls in one
/// died with nothing on the glass but "installed 19779 | extraction -": the
/// kernel keeps the latest per kind and the terminal verdict names the row
/// or the step, so a stall is one call rather than forty.
pub const COOKIE_INIT_STEP: u64 = 0x6B72_5F69_6E69_7453; // "kr_initS"

/// Completion cookie for the boot SPMD validate run (`"kr_spmdW"`),
/// posted by INIT after the store seed: it launches `spmd-witness`
/// (the one `kore-rt/validate` build) through the ordinary
/// generation-bound spawn path and forwards the verdict.  Payload 0 is
/// the clean run; 900 no row/capability; 1000+s is a nonzero witness
/// exit s. Application slots 256..261 identify spmd / room / lane / rotate /
/// dispatch / resident respectively; the terminal output retains each typed
/// fault. Anything else is the launch's spawn rung. Like [`COOKIE_STORE`], the
/// summary is deliberately non-fatal: the validate ladder must never
/// cost a machine that already booted, so this cookie is the ONLY
/// place a dead SPMD tier and a working one differ at boot — the
/// gates key on it (production metal has no serial; a terminal rerun of
/// `spmd-witness` prints the detailed line instead).
pub const COOKIE_SPMD: u64 = 0x6B72_5F73_706D_6457; // "kr_spmdW"

/// Completion cookie for INIT's production Lua VM validation (`"kr_luaOK"`).
/// The same archive generation exposed to each shell is launched through the
/// ordinary profile-driven path with `--selftest`; payload zero is success and
/// every other value is the common-launch refusal/exit rung.  As with the SPMD
/// witness, failure is reported but never costs an already serving machine.
pub const COOKIE_LUA: u64 = 0x6B72_5F6C_7561_4F4B; // "kr_luaOK"

/// Completion cookie for the first shell's production Lua REPL witness
/// (`"kr_luaRP"`).
///
/// Unlike [`COOKIE_LUA`]'s later batch/library selftest, this verdict comes
/// from the shell itself after it launched the authenticated `lua` generation
/// with empty arguments, relayed a fixed multiline session through the live
/// process-stream page, observed the child's explicit close, reaped it, and
/// returned to its own published prompt.  Payload byte zero is the verdict
/// (`0` success); higher bytes carry verdict-specific detail.  Init relays this
/// cookie before its own READY, making the interactive mechanism part of the
/// production founding witness rather than an external keyboard test.
pub const COOKIE_LUA_REPL: u64 = 0x6B72_5F6C_7561_5250; // "kr_luaRP"

/// Completion cookie for the first shell's post-ready MicroPython guest
/// witness (`"kr_mpyOK"`). The shell launches the real interpreter through
/// its ordinary hypervisor client face, scans the guest console for the
/// computed `mpy-validate: 42` line, sends Ctrl-D, and waits for the guest to
/// halt. Payload 0 is success, 1 means virtualization was honestly unavailable,
/// and higher values identify a broken transport, protocol, marker, or teardown
/// step. Unlike [`COOKIE_LUA_REPL`], this testimony is deliberately post-ready:
/// the hypervisor is the final manifest row, so the request must not hold INIT's
/// readiness (and therefore prevent that row from being founded).
pub const COOKIE_MPY: u64 = 0x6B72_5F6D_7079_4F4B; // "kr_mpyOK"

/// Request-ring capacity of a founded pair's shared window — the
/// **cross-boundary geometry contract** (Phase-A checkpoint G2a-3a): the
/// founder formats the window with [`init_window`]`(REQ_CAP, REP_CAP)` before
/// the founding, and the service attaches its halves with
/// [`serve_rings`]`(window, REQ_CAP)`.  The split offset is a pure function of
/// this number ([`rep_offset`]), so a drift between the two sides silently
/// points the reply ring at the wrong bytes — hence one constant, cited by both.
///
/// Eight records is a comfortable depth for the boot pairs and keeps the whole
/// pair inside one page for every landed protocol (mm: `8·32 + 8·8` plus two
/// 192-byte headers = 704 B of the founder's 4 KiB ring lease).
pub const REQ_CAP: u32 = 8;

/// Reply-ring capacity of a founded pair's shared window — [`REQ_CAP`]'s twin
/// (same contract, same reasoning).
pub const REP_CAP: u32 = 8;

/// Byte offset of the reply ring inside the pair's window.
pub const fn rep_offset<S: SvcProtocol>(req_cap: u32) -> usize {
    region_bytes::<S::ReqWire>(req_cap)
}

/// Bytes the pair's shared window must hold for the given capacities —
/// what the founder sizes the mm lease to (page-rounded by the lease).
pub const fn window_bytes<S: SvcProtocol>(req_cap: u32, rep_cap: u32) -> usize {
    region_bytes::<S::ReqWire>(req_cap) + region_bytes::<S::RepWire>(rep_cap)
}

/// The client's two ring halves over its mapping of the pair window.
pub struct ClientRings<'w, S: SvcProtocol> {
    /// The request ring's producer half (push, then kick).
    pub req: Producer<'w, S::ReqWire>,
    /// The reply ring's consumer half.
    pub rep: Consumer<'w, S::RepWire>,
}

/// The service's two ring halves — [`ClientRings`]' mirror.
pub struct ServeRings<'w, S: SvcProtocol> {
    /// The request ring's consumer half (the serve loop's intake).
    pub req: Consumer<'w, S::ReqWire>,
    /// The reply ring's producer half (push, then kick).
    pub rep: Producer<'w, S::RepWire>,
}

/// Format `window` as an empty request/reply ring pair — the **founder's
/// verb** (exclusive access: the window is not yet shared).
pub fn init_window<S: SvcProtocol>(
    window: &mut [u8],
    req_cap: u32,
    rep_cap: u32,
) -> Result<(), RingError> {
    let at = rep_offset::<S>(req_cap);
    if window.len() < window_bytes::<S>(req_cap, rep_cap) {
        return Err(RingError::TooSmall);
    }
    let (req, rep) = window.split_at_mut(at);
    spsc::init::<S::ReqWire>(req, req_cap)?;
    spsc::init::<S::RepWire>(rep, rep_cap)
}

/// Attach the **client** halves over this side's own mapping of an
/// already-initialized pair window (the cross-space form).  Fail-closed:
/// a window that does not carry well-formed rings of the protocol's
/// exact wire forms refuses.
pub fn client_rings<S: SvcProtocol>(
    window: &mut [u8],
    req_cap: u32,
    rep_cap: u32,
) -> Result<ClientRings<'_, S>, RingError> {
    let at = rep_offset::<S>(req_cap);
    let rep_len = region_bytes::<S::RepWire>(rep_cap);
    let Some(end) = at.checked_add(rep_len) else {
        return Err(RingError::TooSmall);
    };
    if window.len() < end {
        return Err(RingError::TooSmall);
    }
    let (req, tail) = window.split_at_mut(at);
    let (rep, _) = tail.split_at_mut(rep_len);
    Ok(ClientRings {
        req: spsc::producer::<S::ReqWire>(req)?,
        rep: spsc::consumer::<S::RepWire>(rep)?,
    })
}

/// Attach the **service** halves — [`client_rings`]' mirror.
pub fn serve_rings<S: SvcProtocol>(
    window: &mut [u8],
    req_cap: u32,
    rep_cap: u32,
) -> Result<ServeRings<'_, S>, RingError> {
    let at = rep_offset::<S>(req_cap);
    let rep_len = region_bytes::<S::RepWire>(rep_cap);
    let Some(end) = at.checked_add(rep_len) else {
        return Err(RingError::TooSmall);
    };
    if window.len() < end {
        return Err(RingError::TooSmall);
    }
    let (req, tail) = window.split_at_mut(at);
    let (rep, _) = tail.split_at_mut(rep_len);
    Ok(ServeRings {
        req: spsc::consumer::<S::ReqWire>(req)?,
        rep: spsc::producer::<S::RepWire>(rep)?,
    })
}

/// [`init_window`] + both sides over one exclusively-borrowed window —
/// the same-address-space / host-test form (the `spsc::split` idiom:
/// the sides alias the window only through the ring protocol).
pub fn split_window<S: SvcProtocol>(
    window: &mut [u8],
    req_cap: u32,
    rep_cap: u32,
) -> Result<(ClientRings<'_, S>, ServeRings<'_, S>), RingError> {
    init_window::<S>(window, req_cap, rep_cap)?;
    let at = rep_offset::<S>(req_cap);
    let (req_region, rep_region) = window.split_at_mut(at);
    // Two independent `split`s would double-init; attach each half pair
    // over raw region borrows instead.  SAFETY of the aliasing: exactly
    // `spsc::split`'s — disjoint cursors, Release/Acquire slot hand-off.
    let (req_len, rep_len) = (req_region.len(), rep_region.len());
    let (req_base, rep_base) = (req_region.as_mut_ptr(), rep_region.as_mut_ptr());
    // SAFETY: the `&mut window` borrow pins both regions for '_; each
    // half touches its region only through the SPSC protocol.
    let (req_a, req_b) = unsafe {
        (
            core::slice::from_raw_parts_mut(req_base, req_len),
            core::slice::from_raw_parts_mut(req_base, req_len),
        )
    };
    // SAFETY: as above.
    let (rep_a, rep_b) = unsafe {
        (
            core::slice::from_raw_parts_mut(rep_base, rep_len),
            core::slice::from_raw_parts_mut(rep_base, rep_len),
        )
    };
    Ok((
        ClientRings {
            req: spsc::producer::<S::ReqWire>(req_a)?,
            rep: spsc::consumer::<S::RepWire>(rep_a)?,
        },
        ServeRings {
            req: spsc::consumer::<S::ReqWire>(req_b)?,
            rep: spsc::producer::<S::RepWire>(rep_b)?,
        },
    ))
}

#[cfg(test)]
#[test]
fn accelerator_capture_kind_preserves_identity_and_rejects_unknown_versions() {
    for kind in [AccelCaptureKind::Regions, AccelCaptureKind::Overlap] {
        let payload = kind.begin(0x8086, 0x643e);
        assert_eq!(
            AccelCaptureKind::decode_begin(payload),
            Some((kind, 0x8086_0000_643e))
        );
        assert!(kind.words() <= ACCEL_CAPTURE_MAX_WORDS);
    }
    assert_eq!(
        AccelCaptureKind::Regions.begin(0x8086, 0x643e),
        0x8086_0000_643e
    );
    assert_eq!(AccelCaptureKind::decode_begin(2 << 48), None);
    assert_eq!(AccelCaptureKind::decode_begin(u64::MAX), None);
}
