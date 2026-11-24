//! **The frame row's verb table** — one row per request verb.
//!
//! Each verb is a zero-sized type carrying everything that verb decides:
//! its wire code, how it lowers and lifts, which reply answers it, how many
//! grant lanes that reply may ride, and the arm that serves it. The codecs
//! in [`crate::wire`] and the serve step in [`crate::service`] read the
//! table below and name no verb at all.
//!
//! **Adding a verb is an `impl` block here and a name in the row list** —
//! not an arm in the encoder, another in the decoder, a third in the
//! correlation table and a fourth in the serve match. `res/INET_MIGRATION.md`
//! §2.3 measured what those four edits cost when two workstreams make them
//! at once, and §2.1 the one collision that compiled anyway; the seam is
//! [`kore_rt::svc::Verb`] and the reasoning is there.

use kore_abi::session::SlotRef;
use kore_abi::svc::NetReqWire;
use kore_driver::net::VifId;
use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_rt::service::Serve;
use kore_rt::svc::{PeerBundlePart, ServeVerb, Verb};
use kore_rt::verb_table;
use zerocopy::little_endian::{U16, U32};

use crate::service::{IdentityReport, NetHost, NetRefusal, NetRequest, NetResponse, NetService};
use crate::wire::{NetWire, decode_filter, encode_filter};

verb_table! {
    /// The frame row's request vocabulary.
    ///
    /// The list is the registration: a verb absent here is unreachable, and
    /// two verbs sharing a code is a build error (the assertion the macro
    /// emits beside this alias). Order is free — these replies are
    /// one-to-one with their verbs, so no row claims another's answer.
    pub type NetVerbs over NetWire = [
        // the frame program
        Query,
        OpenVif,
        SetFilter,
        CloseVif,
        Signal,
        Stats,
        Radio,
        // the flow program — `InetQuery` before `Configure`, because both
        // are answered with an identity and the first row that claims a
        // reply is the one that echoes it.
        InetQuery,
        Configure,
        OpenFlow,
        OpenListener,
        CloseFlow,
        // the peer program — the row's own mint (`res/PEER_LINK.md` P1)
        AdmitPeer,
        TakePeerAdmission,
        ClosePeer,
        RetirePeer,
        ConfirmRetired,
    ];
}

/// The vif identity every verb that names one reads from the same two
/// fields — stated once so three rows cannot drift on which fields those
/// are.
fn vif_of(wire: &NetReqWire) -> VifId {
    VifId {
        index: wire.vif_index.get(),
        generation: wire.vif_generation.get(),
    }
}

/// Stamp a vif identity into the two fields [`vif_of`] reads back.
fn set_vif(wire: &mut NetReqWire, vif: VifId) {
    wire.vif_index = U16::new(vif.index);
    wire.vif_generation = U16::new(vif.generation);
}

// ── the rows ───────────────────────────────────────────────────────────

/// Report link, address, geometry, and this pair's policy.
pub struct Query;

impl Verb<NetWire> for Query {
    const CODE: u8 = NetReqWire::OP_QUERY;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::Query)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::Query)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Info(_))
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for Query {
    fn serve(
        svc: &mut NetService<H>,
        _sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::Query = req else { return None };
        Some(match svc.info() {
            Some(info) => NetResponse::Info(info),
            None => NetResponse::Refused(NetRefusal::NotReady),
        })
    }
}

/// Open a vif over the window riding the return leg.
pub struct OpenVif;

impl Verb<NetWire> for OpenVif {
    const CODE: u8 = NetReqWire::OP_OPEN_VIF;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::OpenVif { filter, announced } = req else {
            return false;
        };
        wire.announced = U32::new(announced.index());
        encode_filter(filter, wire);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::OpenVif {
            filter: decode_filter(wire)?,
            announced: SlotRef::from_wire(wire.announced.get()),
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Opened { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for OpenVif {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::OpenVif { filter, announced } = req else {
            return None;
        };
        let via = sv.via();
        Some(svc.open_arm(via, *filter, *announced, sv))
    }
}

/// Replace the receive filter on a vif this pair already holds.
pub struct SetFilter;

impl Verb<NetWire> for SetFilter {
    const CODE: u8 = NetReqWire::OP_SET_FILTER;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::SetFilter { vif, filter } = req else {
            return false;
        };
        set_vif(wire, *vif);
        encode_filter(filter, wire);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::SetFilter {
            vif: vif_of(wire),
            filter: decode_filter(wire)?,
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Filtered { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for SetFilter {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::SetFilter { vif, filter } = req else {
            return None;
        };
        Some(match svc.set_filter(sv.via(), *vif, *filter) {
            Ok(vif) => NetResponse::Filtered { vif },
            Err(refusal) => NetResponse::Refused(refusal),
        })
    }
}

/// Close a vif and invalidate every identifier issued for it.
pub struct CloseVif;

impl Verb<NetWire> for CloseVif {
    const CODE: u8 = NetReqWire::OP_CLOSE_VIF;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::CloseVif { vif } = req else {
            return false;
        };
        set_vif(wire, *vif);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::CloseVif { vif: vif_of(wire) })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Closed { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for CloseVif {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::CloseVif { vif } = req else {
            return None;
        };
        Some(match svc.close_vif(sv.via(), *vif) {
            Ok(vif) => NetResponse::Closed { vif },
            Err(refusal) => NetResponse::Refused(refusal),
        })
    }
}

/// Mint this pair's receive doorbell.
pub struct Signal;

impl Verb<NetWire> for Signal {
    const CODE: u8 = NetReqWire::OP_SIGNAL;
    /// The one verb whose reply carries authority: the minted recv end.
    /// Stated here so a face that cannot reach this verb wires no grant
    /// lane — least authority and a working face being the same answer
    /// (#936).
    const GRANTS: usize = 1;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::Signal)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::Signal)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Signaled)
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for Signal {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::Signal = req else { return None };
        let via = sv.via();
        Some(svc.signal_arm(via, sv))
    }
}

/// Report one vif's counters.
pub struct Stats;

impl Verb<NetWire> for Stats {
    const CODE: u8 = NetReqWire::OP_STATS;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::Stats { vif } = req else {
            return false;
        };
        set_vif(wire, *vif);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::Stats { vif: vif_of(wire) })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Stats { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for Stats {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::Stats { vif } = req else {
            return None;
        };
        Some(match svc.stats_arm(sv.via(), *vif) {
            Ok(stats) => NetResponse::Stats { vif: *vif, stats },
            Err(refusal) => NetResponse::Refused(refusal),
        })
    }
}

/// Report the bound radio's facts.
pub struct Radio;

impl Verb<NetWire> for Radio {
    const CODE: u8 = NetReqWire::OP_RADIO;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::Radio)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::Radio)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Radio(_))
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for Radio {
    // Not gated on a backend: a radio at D1 is bound and answering while
    // lending no frame path, and its facts are exactly what a serial-less
    // machine needs to read in that posture.
    fn serve(
        svc: &mut NetService<H>,
        _sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::Radio = req else { return None };
        Some(match svc.radio() {
            Some(report) => NetResponse::Radio(report),
            None => NetResponse::Refused(NetRefusal::Unsupported),
        })
    }
}

// ── the flow program ───────────────────────────────────────────────────
//
// The arbitrating verbs, on this row's vocabulary because the arbitration
// lives on this row. `res/INET_MIGRATION.md` §5.7a is why it moved: a flow
// is authority only once a `FrameFilter` enforces it, the table those go on
// is here, and reaching it from another row wants a founder face onto a
// chained provider that cannot be built.
//
// Each of these drives `NetService`'s own methods, which hold the
// `InetRow` and the vif table together under a split borrow. None of them
// can answer `NoFrameFace`: there is no face.

/// Report the machine's address, prefix, gateway and resolver.
pub struct InetQuery;

impl Verb<NetWire> for InetQuery {
    const CODE: u8 = NetReqWire::OP_INET_QUERY;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::InetQuery)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::InetQuery)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Identity { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for InetQuery {
    fn serve(
        svc: &mut NetService<H>,
        _sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        if !matches!(req, NetRequest::InetQuery) {
            return None;
        }
        Some(NetResponse::Identity {
            identity: svc.identity_report(),
        })
    }
}

/// Give the machine its address.
pub struct Configure;

impl Verb<NetWire> for Configure {
    const CODE: u8 = NetReqWire::OP_CONFIGURE;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::Configure {
            addr,
            prefix_len,
            gateway,
        } = req
        else {
            return false;
        };
        wire.local_addr = addr.0;
        wire.prefix_len = *prefix_len;
        wire.gateway = gateway.map(|g| g.0).unwrap_or_default();
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        // An all-zero gateway is "none", not 0.0.0.0: a station may not
        // hold the unspecified address, so the encoding is unambiguous.
        let gateway = (wire.gateway != [0; 4]).then(|| Ipv4Addr(wire.gateway));
        Some(NetRequest::Configure {
            addr: Ipv4Addr(wire.local_addr),
            prefix_len: wire.prefix_len,
            gateway,
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Identity { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for Configure {
    fn serve(
        svc: &mut NetService<H>,
        _sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::Configure {
            addr,
            prefix_len,
            gateway,
        } = req
        else {
            return None;
        };
        Some(match svc.configure_arm(*addr, *prefix_len, *gateway) {
            Ok(()) => NetResponse::Identity {
                identity: svc.identity_report(),
            },
            Err(refusal) => NetResponse::Refused(refusal),
        })
    }
}

/// Mint a flow to one peer, binding a local port.
pub struct OpenFlow;

impl Verb<NetWire> for OpenFlow {
    const CODE: u8 = NetReqWire::OP_OPEN_FLOW;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::OpenFlow {
            vif,
            proto,
            local_port,
            peer,
        } = req
        else {
            return false;
        };
        set_vif(wire, *vif);
        wire.proto = proto.0;
        // Port zero is never a binding, so it is free to mean "draw me an
        // ephemeral one" — the same reading `PortTable` already takes.
        wire.local_port = U16::new(local_port.unwrap_or(0));
        wire.peer_addr = peer.0.0;
        wire.peer_port = U16::new(peer.1);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        let local_port = wire.local_port.get();
        Some(NetRequest::OpenFlow {
            vif: vif_of(wire),
            proto: IpProto(wire.proto),
            local_port: (local_port != 0).then_some(local_port),
            peer: (Ipv4Addr(wire.peer_addr), wire.peer_port.get()),
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Flow { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for OpenFlow {
    async fn serve_async(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::OpenFlow {
            vif,
            proto,
            local_port,
            peer,
        } = req
        else {
            return None;
        };
        let grant = svc.flow_grant(sv.via());
        Some(
            match svc
                .open_flow(sv.via(), *vif, grant, *proto, *local_port, *peer)
                .await
            {
                Ok(cap) => flow_reply(&cap),
                Err(refusal) => NetResponse::Refused(refusal.into()),
            },
        )
    }
}

/// Mint a listening flow: a local endpoint, any peer.
pub struct OpenListener;

impl Verb<NetWire> for OpenListener {
    const CODE: u8 = NetReqWire::OP_OPEN_LISTENER;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::OpenListener {
            vif,
            proto,
            local_port,
        } = req
        else {
            return false;
        };
        set_vif(wire, *vif);
        wire.proto = proto.0;
        wire.local_port = U16::new(*local_port);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::OpenListener {
            vif: vif_of(wire),
            proto: IpProto(wire.proto),
            local_port: wire.local_port.get(),
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::Flow { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for OpenListener {
    async fn serve_async(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::OpenListener {
            vif,
            proto,
            local_port,
        } = req
        else {
            return None;
        };
        let grant = svc.flow_grant(sv.via());
        Some(
            match svc
                .open_listener(sv.via(), *vif, grant, *proto, *local_port)
                .await
            {
                Ok(cap) => flow_reply(&cap),
                Err(refusal) => NetResponse::Refused(refusal.into()),
            },
        )
    }
}

/// Release a vif's flows and the ports they hold.
pub struct CloseFlow;

impl Verb<NetWire> for CloseFlow {
    const CODE: u8 = NetReqWire::OP_CLOSE_FLOW;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::CloseFlow { vif } = req else {
            return false;
        };
        set_vif(wire, *vif);
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::CloseFlow { vif: vif_of(wire) })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::FlowClosed { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for CloseFlow {
    async fn serve_async(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        let NetRequest::CloseFlow { vif } = req else {
            return None;
        };
        Some(match svc.close_flows(sv.via(), *vif).await {
            (released, Ok(())) => NetResponse::FlowClosed {
                released: released as u16,
            },
            // A vif that is not this caller's frees nothing, and the count
            // alone could not say so.
            (_, Err(refusal)) => NetResponse::Refused(refusal.into()),
        })
    }
}

/// The reply a mint earns: what the capability says, read back off it.
///
/// Off the capability rather than off the request, so the record cannot
/// disagree with the authority it is reporting — the one way a caller could
/// be told it holds something other than what it holds.
fn flow_reply(cap: &kore_inet::FlowCap) -> NetResponse {
    use kore_driver::net::flow::{EndpointMatch, ProtoMatch};
    use kore_inet::flow::Scope;

    let Scope::Flow(m) = cap.scope() else {
        // Only S1 is mintable through these verbs; anything else is a bug
        // in this row rather than a refusal the caller can act on.
        return NetResponse::Refused(NetRefusal::Internal);
    };
    let local = match m.local {
        EndpointMatch::AddrPort { addr, port } => (addr, port),
        _ => return NetResponse::Refused(NetRefusal::Internal),
    };
    let peer = match m.peer {
        EndpointMatch::AddrPort { addr, port } => Some((addr, port)),
        // A listener pins no peer. Said here as `None` rather than as a
        // zeroed address, because the capability itself draws that line.
        _ => None,
    };
    // The mint always pins one protocol; `Any` is a widening power no
    // flow verb can ask for, so reaching it here is this row's bug.
    let ProtoMatch::Only(proto) = m.proto else {
        return NetResponse::Refused(NetRefusal::Internal);
    };
    NetResponse::Flow { proto, local, peer }
}

// ── the peer program ───────────────────────────────────────────────────

/// Contribute one component of a peer bundle.
pub struct AdmitPeer;

impl Verb<NetWire> for AdmitPeer {
    const CODE: u8 = NetReqWire::OP_ADMIT_PEER;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::AdmitPeer { part, announced } = req else {
            return false;
        };
        let Some(code) = part_code(*part) else {
            return false;
        };
        wire.peer_part = code;
        wire.announced = U32::new(announced.index());
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::AdmitPeer {
            part: part_of(wire.peer_part)?,
            announced: SlotRef::from_wire(wire.announced.get()),
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::PeerAccepted)
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for AdmitPeer {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        svc.peer_arm(req, sv)
    }
}

/// Collect the verdict on the bundle most recently completed.
pub struct TakePeerAdmission;

impl Verb<NetWire> for TakePeerAdmission {
    const CODE: u8 = NetReqWire::OP_TAKE_PEER_ADMISSION;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::TakePeerAdmission)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::TakePeerAdmission)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::PeerAdmitted { .. })
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>>
    for TakePeerAdmission
{
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        svc.peer_arm(req, sv)
    }
}

/// Close the minted pair this request arrives on.
pub struct ClosePeer;

impl Verb<NetWire> for ClosePeer {
    const CODE: u8 = NetReqWire::OP_CLOSE_PEER;

    fn encode(req: &NetRequest, _wire: &mut NetReqWire) -> bool {
        matches!(req, NetRequest::ClosePeer)
    }
    fn decode(_wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::ClosePeer)
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::PeerClosing)
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for ClosePeer {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        svc.peer_arm(req, sv)
    }
}

/// A bundle component's wire code.
///
/// The named parts are small and distinct; a grant lane is based well past
/// them so a lane index can never be read as a named part. A lane that does
/// not fit the byte is refused rather than truncated — a truncated lane would
/// name a *different*, valid lane.
fn part_code(part: PeerBundlePart) -> Option<u8> {
    Some(match part {
        PeerBundlePart::Ring => NetReqWire::PART_RING,
        PeerBundlePart::Kick => NetReqWire::PART_KICK,
        PeerBundlePart::ReturnRecv => NetReqWire::PART_RETURN_RECV,
        PeerBundlePart::Grant(lane) => NetReqWire::PART_GRANT_BASE.checked_add(lane)?,
    })
}

/// The inverse of [`part_code`], fail-closed.
fn part_of(code: u8) -> Option<PeerBundlePart> {
    Some(match code {
        NetReqWire::PART_RING => PeerBundlePart::Ring,
        NetReqWire::PART_KICK => PeerBundlePart::Kick,
        NetReqWire::PART_RETURN_RECV => PeerBundlePart::ReturnRecv,
        _ if code >= NetReqWire::PART_GRANT_BASE => {
            PeerBundlePart::Grant(code - NetReqWire::PART_GRANT_BASE)
        }
        _ => return None,
    })
}

/// Retire one exact minted generation, from the administration side.
pub struct RetirePeer;

impl Verb<NetWire> for RetirePeer {
    const CODE: u8 = NetReqWire::OP_RETIRE_PEER;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::RetirePeer { peer } = req else {
            return false;
        };
        wire.announced = U32::new(u32::from(peer.arrival()));
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::RetirePeer {
            peer: peer_of(wire)?,
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::PeerClosing)
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for RetirePeer {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        svc.peer_arm(req, sv)
    }
}

/// Ask whether that generation's cell is free again.
pub struct ConfirmRetired;

impl Verb<NetWire> for ConfirmRetired {
    const CODE: u8 = NetReqWire::OP_CONFIRM_RETIRED;

    fn encode(req: &NetRequest, wire: &mut NetReqWire) -> bool {
        let NetRequest::ConfirmRetired { peer } = req else {
            return false;
        };
        wire.announced = U32::new(u32::from(peer.arrival()));
        true
    }
    fn decode(wire: &NetReqWire) -> Option<NetRequest> {
        Some(NetRequest::ConfirmRetired {
            peer: peer_of(wire)?,
        })
    }
    fn answers(rep: &NetResponse) -> bool {
        matches!(rep, NetResponse::PeerRetired)
    }
}

impl<'o, H: NetHost> ServeVerb<NetWire, NetService<H>, Serve<'o, NetResponse>> for ConfirmRetired {
    fn serve(
        svc: &mut NetService<H>,
        sv: &mut Serve<'o, NetResponse>,
        req: &NetRequest,
    ) -> Option<NetResponse> {
        svc.peer_arm(req, sv)
    }
}

/// The generation a retirement verb names, read back from the one field both
/// put it in. A zero generation names no live pair and is refused.
fn peer_of(wire: &NetReqWire) -> Option<kore_rt::svc::PeerId> {
    let arrival = u16::try_from(wire.announced.get()).ok()?;
    kore_rt::svc::PeerId::from_arrival(arrival)
}
