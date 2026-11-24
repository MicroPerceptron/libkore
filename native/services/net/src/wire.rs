//! **The frame service's wire vocabulary** — the request/response enums, and
//! the codecs between them and the frozen `kore_abi::svc` records.
//!
//! The codecs are free functions so the backend-free [`NetWire`] marker and
//! the host both delegate to them: byte agreement between a consumer that
//! never imports a hardware type and a service that does holds by
//! construction rather than by review.

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::svc::{
    MAX_NET_ETHERTYPES, NetCountersWire, NetInetWire, NetRadioWire, NetRepWire, NetReqWire,
};
use kore_driver::net::filter::{EtherTypeMatch, MulticastMatch};
use kore_driver::net::flow::{EndpointMatch, FlowMatch, IpProto, Ipv4Addr, ProtoMatch};
use kore_driver::net::{EtherType, FrameFilter, MacAddr, NicCounters, VifId};
use kore_rt::svc::{PeerId, SvcProtocol, VerbTable, VerbWire};
use zerocopy::little_endian::{U16, U32, U64};

use crate::metal::NetRow;
use crate::service::{
    IdentityReport, NetHost, NetInfo, NetRefusal, NetRequest, NetResponse, NetService, NetStats,
    RadioFirmware, RadioPhy, RadioReport,
};
use crate::verbs::NetVerbs;

/// Grants riding beside a reply — one, the receive doorbell's recv end.
pub const NET_REPLY_GRANTS: usize = 1;

/// Serving pairs.
///
/// **Two: the row's own, and exec's.** The kernel founder stages extra pairs
/// by class rather than by looping over this constant, and the one class a
/// chained driver row may declare is the exec-provider pair (`exec_client =
/// true` in the row's manifest): the row's FINAL pair belongs to the resident
/// exec manager, and the founder refuses the whole plan — before any row
/// founds — for an exec provider with fewer than two (`services/exec/src/
/// founder/mod.rs` asserts the rule at build time). Pair 0 stays the row's
/// own: the founding self-test opens its vifs through it in-process, and a
/// frame client arrives as a *minted* face on top of it, which needs no
/// founder edit and no change to this number (`services/accel/boot.toml`
/// names that path).
pub const NET_SERVE_PAIRS: usize = 2;

/// The pair the resident exec manager holds — the row's final one, by the
/// founder's rule above. Its authority is [`VifPolicy::DENY`]: exec issues
/// the two read-only verbs at a launch and can open nothing.
pub const NET_EXEC_PAIR: usize = NET_SERVE_PAIRS - 1;

/// **May this arrival pair assemble peer bundles?**
///
/// Only a FIXED pair — the row's own and exec's. A minted face must never
/// mint another: the row hands out client faces, and a client that could
/// hand out more would make the roster unbounded and the authority
/// untraceable to the launcher that asked for it.
pub const fn is_admin_pair(via: u16) -> bool {
    (via as usize) < NET_SERVE_PAIRS
}

/// The backend-free protocol marker.
///
/// A consumer of this vocabulary must not have to import a hardware backend
/// type — or prove it could host one — merely to place a call. `NetWire`
/// carries the codecs; [`NetService`](crate::service::NetService) delegates to
/// the same free functions.
pub struct NetWire;

// ── requests ─────────────────────────────────────────────────────────

/// The record this protocol's verbs register into.
///
/// The blank record is the whole of the encoder's discipline: a row sets
/// only its own fields, so the re-encode comparison in
/// [`VerbTable::decode`](kore_rt::svc::VerbTable::decode) is what enforces
/// reserved-zero and unused-word-zero — no per-field guard to write, and
/// none to forget.
impl VerbWire for NetWire {
    type Request = NetRequest;
    type Response = NetResponse;
    type ReqWire = NetReqWire;

    fn blank() -> NetReqWire {
        NetReqWire {
            op: 0,
            ethertype_kind: 0,
            ethertype_count: 0,
            multicast_kind: 0,
            flags: 0,
            peer_part: 0,
            reserved: [0; 2],
            vif_index: U16::ZERO,
            vif_generation: U16::ZERO,
            announced: U32::ZERO,
            ethertypes: [U16::ZERO; MAX_NET_ETHERTYPES],
            multicast_group: [0; 6],
            reserved2: [0; 2],
            flow_kind: 0,
            proto_kind: 0,
            local_kind: 0,
            peer_kind: 0,
            proto: 0,
            family: 0,
            reserved3: [0; 1],
            local_port: U16::ZERO,
            peer_port: U16::ZERO,
            local_addr: [0; 4],
            peer_addr: [0; 4],
            prefix_len: 0,
            gateway: [0; 4],
            reserved4: [0; 8],
        }
    }

    fn code_of(wire: &NetReqWire) -> u8 {
        wire.op
    }

    fn set_code(wire: &mut NetReqWire, code: u8) {
        wire.op = code;
    }
}

/// Encode a request — the table's row for it does the work.
pub fn encode_req(req: &NetRequest) -> NetReqWire {
    NetVerbs::encode(req)
}

/// Decode a request, **fail-closed**.
///
/// The structural lift is the matching row's; the re-encode comparison the
/// table applies then rejects any record whose reserved bytes, unused words,
/// or dead ethertype slots are non-zero. One comparison replaces a guard per
/// field, and cannot drift from the encoder the way hand-written guards do.
pub fn decode_req(w: &NetReqWire) -> Option<NetRequest> {
    NetVerbs::decode(w)
}

pub(crate) fn encode_filter(filter: &FrameFilter, w: &mut NetReqWire) {
    match filter.ethertype {
        EtherTypeMatch::Any => w.ethertype_kind = NetReqWire::ET_ANY,
        EtherTypeMatch::Only { .. } => {
            w.ethertype_kind = NetReqWire::ET_ONLY;
            let listed = filter.ethertype.listed();
            w.ethertype_count = listed.len() as u8;
            for (slot, t) in w.ethertypes.iter_mut().zip(listed) {
                *slot = U16::new(t.0);
            }
        }
    }
    match filter.multicast {
        MulticastMatch::None => w.multicast_kind = NetReqWire::MC_NONE,
        MulticastMatch::All => w.multicast_kind = NetReqWire::MC_ALL,
        MulticastMatch::Group(mac) => {
            w.multicast_kind = NetReqWire::MC_GROUP;
            w.multicast_group = mac.0;
        }
    }
    if filter.unicast {
        w.flags |= NetReqWire::FLAG_UNICAST;
    }
    if filter.broadcast {
        w.flags |= NetReqWire::FLAG_BROADCAST;
    }
    if filter.promiscuous {
        w.flags |= NetReqWire::FLAG_PROMISCUOUS;
    }
    // The flow block stays all-zero unless a flow is bound, which is what
    // makes the record's growth additive in meaning: every filter that
    // predates flows encodes exactly as it did.
    if let Some(flow) = filter.flow.as_ref() {
        w.flow_kind = NetReqWire::FLOW_SOME;
        w.family = NetReqWire::FAMILY_IPV4;
        match flow.proto {
            ProtoMatch::Only(p) => {
                w.proto_kind = NetReqWire::PROTO_ONLY;
                w.proto = p.0;
            }
            ProtoMatch::Any => w.proto_kind = NetReqWire::PROTO_ANY,
        }
        encode_endpoint(
            &flow.local,
            &mut w.local_kind,
            &mut w.local_addr,
            &mut w.local_port,
        );
        encode_endpoint(
            &flow.peer,
            &mut w.peer_kind,
            &mut w.peer_addr,
            &mut w.peer_port,
        );
    }
}

fn encode_endpoint(ep: &EndpointMatch, kind: &mut u8, addr: &mut [u8; 4], port: &mut U16) {
    match ep {
        EndpointMatch::AddrPort { addr: a, port: p } => {
            *kind = NetReqWire::EP_ADDRPORT;
            *addr = a.0;
            *port = U16::new(*p);
        }
        EndpointMatch::Addr(a) => {
            *kind = NetReqWire::EP_ADDR;
            *addr = a.0;
        }
        EndpointMatch::Any => *kind = NetReqWire::EP_ANY,
    }
}

fn decode_endpoint(kind: u8, addr: [u8; 4], port: U16) -> Option<EndpointMatch> {
    Some(match kind {
        NetReqWire::EP_ADDRPORT => EndpointMatch::AddrPort {
            addr: Ipv4Addr(addr),
            port: port.get(),
        },
        NetReqWire::EP_ADDR => EndpointMatch::Addr(Ipv4Addr(addr)),
        NetReqWire::EP_ANY => EndpointMatch::Any,
        _ => return None,
    })
}

pub(crate) fn decode_filter(w: &NetReqWire) -> Option<FrameFilter> {
    let ethertype = match w.ethertype_kind {
        NetReqWire::ET_ANY => EtherTypeMatch::Any,
        NetReqWire::ET_ONLY => {
            let count = w.ethertype_count as usize;
            if count > MAX_NET_ETHERTYPES {
                return None;
            }
            let mut types = [EtherType(0); MAX_NET_ETHERTYPES];
            for (slot, raw) in types.iter_mut().zip(&w.ethertypes[..count]) {
                *slot = EtherType(raw.get());
            }
            EtherTypeMatch::Only {
                types,
                count: count as u8,
            }
        }
        _ => return None,
    };
    let multicast = match w.multicast_kind {
        NetReqWire::MC_NONE => MulticastMatch::None,
        NetReqWire::MC_ALL => MulticastMatch::All,
        NetReqWire::MC_GROUP => MulticastMatch::Group(MacAddr(w.multicast_group)),
        _ => return None,
    };
    // Any bit outside the three defined ones is a record this version does
    // not understand: refused, not masked off.
    const KNOWN: u8 =
        NetReqWire::FLAG_UNICAST | NetReqWire::FLAG_BROADCAST | NetReqWire::FLAG_PROMISCUOUS;
    if w.flags & !KNOWN != 0 {
        return None;
    }
    // An unknown family is refused rather than reinterpreted: the byte exists
    // so an IPv6 sibling is a new value here, and a version that does not
    // know it must not read these addresses as v4.
    let flow = match w.flow_kind {
        NetReqWire::FLOW_NONE => None,
        NetReqWire::FLOW_SOME => {
            if w.family != NetReqWire::FAMILY_IPV4 {
                return None;
            }
            let proto = match w.proto_kind {
                NetReqWire::PROTO_ONLY => ProtoMatch::Only(IpProto(w.proto)),
                NetReqWire::PROTO_ANY => ProtoMatch::Any,
                _ => return None,
            };
            Some(FlowMatch {
                proto,
                local: decode_endpoint(w.local_kind, w.local_addr, w.local_port)?,
                peer: decode_endpoint(w.peer_kind, w.peer_addr, w.peer_port)?,
            })
        }
        _ => return None,
    };
    Some(FrameFilter {
        ethertype,
        unicast: w.flags & NetReqWire::FLAG_UNICAST != 0,
        multicast,
        broadcast: w.flags & NetReqWire::FLAG_BROADCAST != 0,
        promiscuous: w.flags & NetReqWire::FLAG_PROMISCUOUS != 0,
        flow,
    })
}

// ── responses ──────────────────────────────────────────────────────────

/// Encode a response.
pub fn encode_rep(rep: &NetResponse, grants: u8) -> NetRepWire {
    let mut w = NetRepWire {
        tag: 0,
        grants,
        refusal: 0,
        link: 0,
        vif_index: U16::ZERO,
        vif_generation: U16::ZERO,
        mac: [0; 6],
        mtu: U16::ZERO,
        link_mbps: U32::ZERO,
        window_bytes: U32::ZERO,
        rx_frames: U64::ZERO,
        tx_frames: U64::ZERO,
        dropped: U64::ZERO,
        peer_arrival: U16::ZERO,
        reserved: [0; 6],
        counters: NetCountersWire::EMPTY,
        radio: NetRadioWire::EMPTY,
        inet: NetInetWire::default(),
    };
    match rep {
        NetResponse::Radio(report) => {
            w.tag = NetRepWire::TAG_RADIO;
            w.radio = encode_radio(report);
        }
        NetResponse::Identity { identity } => {
            w.tag = NetRepWire::TAG_IDENTITY;
            if let Some(id) = identity {
                w.inet.addr = id.addr.0;
                w.inet.prefix_len = id.prefix_len;
                w.inet.gateway = id.gateway.map(|g| g.0).unwrap_or_default();
                w.inet.dns = id.dns.map(|d| d.0).unwrap_or_default();
            }
            // No identity leaves the block zero, which is exactly how a
            // reader tells "nothing configured" from "configured to
            // 0.0.0.0" — the latter is refused at configure time, so the
            // all-zero address is unambiguous.
        }
        NetResponse::Flow { proto, local, peer } => {
            w.tag = NetRepWire::TAG_FLOW;
            w.inet.proto = proto.0;
            w.inet.addr = local.0.0;
            w.inet.local_port = U16::new(local.1);
            if let Some((paddr, pport)) = peer {
                w.inet.peer_addr = paddr.0;
                w.inet.peer_port = U16::new(*pport);
            }
            // A listener pins no peer, and the zeroed peer is how it says
            // so — the same distinction the capability itself draws.
        }
        NetResponse::FlowClosed { released } => {
            w.tag = NetRepWire::TAG_FLOW_CLOSED;
            w.inet.released = U16::new(*released);
        }
        NetResponse::PeerAccepted => w.tag = NetRepWire::TAG_PEER_ACCEPTED,
        NetResponse::PeerAdmitted { peer } => {
            w.tag = NetRepWire::TAG_PEER_ADMITTED;
            w.peer_arrival = U16::new(peer.arrival());
        }
        NetResponse::PeerClosing => w.tag = NetRepWire::TAG_PEER_CLOSING,
        NetResponse::PeerRetired => w.tag = NetRepWire::TAG_PEER_RETIRED,
        NetResponse::Info(info) => {
            w.tag = NetRepWire::TAG_INFO;
            w.link = u8::from(info.link_up);
            w.mac = info.mac.0;
            w.mtu = U16::new(info.mtu);
            w.link_mbps = U32::new(info.link_mbps);
            w.window_bytes = U32::new(info.window_bytes);
            w.counters = NetCountersWire {
                irqs: U64::new(info.counters.irqs),
                link_changes: U32::new(info.counters.link_changes),
                overruns: U32::new(info.counters.overruns),
                rx_errors: U32::new(info.counters.rx_errors),
                dropped_notices: U32::new(info.dropped_notices),
            };
        }
        NetResponse::Opened { vif, mac } => {
            w.tag = NetRepWire::TAG_OPENED;
            w.vif_index = U16::new(vif.index);
            w.vif_generation = U16::new(vif.generation);
            w.mac = mac.0;
        }
        NetResponse::Filtered { vif } => {
            w.tag = NetRepWire::TAG_FILTERED;
            w.vif_index = U16::new(vif.index);
            w.vif_generation = U16::new(vif.generation);
        }
        NetResponse::Closed { vif } => {
            w.tag = NetRepWire::TAG_CLOSED;
            w.vif_index = U16::new(vif.index);
            w.vif_generation = U16::new(vif.generation);
        }
        NetResponse::Signaled => w.tag = NetRepWire::TAG_SIGNALED,
        NetResponse::Stats { vif, stats } => {
            w.tag = NetRepWire::TAG_STATS;
            w.vif_index = U16::new(vif.index);
            w.vif_generation = U16::new(vif.generation);
            w.rx_frames = U64::new(stats.rx_frames);
            w.tx_frames = U64::new(stats.tx_frames);
            w.dropped = U64::new(stats.dropped);
        }
        NetResponse::Refused(refusal) => {
            w.tag = NetRepWire::TAG_REFUSED;
            w.refusal = refusal.code();
        }
    }
    w
}

/// Decode a response, fail-closed, and pin the grant count per tag.
pub fn decode_rep(w: &NetRepWire) -> Option<(NetResponse, u8)> {
    let rep = raw_decode_rep(w)?;
    // Exactly one tag carries a grant. Checking it here means a reply that
    // claims authority it did not bring — or brings authority it did not
    // announce — never reaches the client's possession step.
    if w.grants != u8::from(matches!(rep, NetResponse::Signaled)) {
        return None;
    }
    (encode_rep(&rep, w.grants) == *w).then_some((rep, w.grants))
}

fn raw_decode_rep(w: &NetRepWire) -> Option<NetResponse> {
    let vif = VifId {
        index: w.vif_index.get(),
        generation: w.vif_generation.get(),
    };
    match w.tag {
        NetRepWire::TAG_INFO => Some(NetResponse::Info(NetInfo {
            link_up: w.link != 0,
            link_mbps: w.link_mbps.get(),
            mac: MacAddr(w.mac),
            mtu: w.mtu.get(),
            window_bytes: w.window_bytes.get(),
            counters: NicCounters {
                irqs: w.counters.irqs.get(),
                link_changes: w.counters.link_changes.get(),
                overruns: w.counters.overruns.get(),
                rx_errors: w.counters.rx_errors.get(),
            },
            dropped_notices: w.counters.dropped_notices.get(),
        })),
        NetRepWire::TAG_OPENED => Some(NetResponse::Opened {
            vif,
            mac: MacAddr(w.mac),
        }),
        NetRepWire::TAG_FILTERED => Some(NetResponse::Filtered { vif }),
        NetRepWire::TAG_CLOSED => Some(NetResponse::Closed { vif }),
        NetRepWire::TAG_SIGNALED => Some(NetResponse::Signaled),
        NetRepWire::TAG_STATS => Some(NetResponse::Stats {
            vif,
            stats: NetStats {
                rx_frames: w.rx_frames.get(),
                tx_frames: w.tx_frames.get(),
                dropped: w.dropped.get(),
            },
        }),
        NetRepWire::TAG_RADIO => Some(NetResponse::Radio(decode_radio(&w.radio))),
        NetRepWire::TAG_IDENTITY => Some(NetResponse::Identity {
            // An all-zero address is "nothing configured". A station may
            // not hold the unspecified address — `Identity::new` refuses
            // it — so the encoding cannot collide with a real one.
            identity: (w.inet.addr != [0; 4]).then(|| IdentityReport {
                addr: Ipv4Addr(w.inet.addr),
                prefix_len: w.inet.prefix_len,
                gateway: (w.inet.gateway != [0; 4]).then(|| Ipv4Addr(w.inet.gateway)),
                dns: (w.inet.dns != [0; 4]).then(|| Ipv4Addr(w.inet.dns)),
            }),
        }),
        NetRepWire::TAG_FLOW => Some(NetResponse::Flow {
            proto: IpProto(w.inet.proto),
            local: (Ipv4Addr(w.inet.addr), w.inet.local_port.get()),
            // A listener pins no peer and encodes it zeroed, which is the
            // same line the capability draws.
            peer: (w.inet.peer_addr != [0; 4])
                .then(|| (Ipv4Addr(w.inet.peer_addr), w.inet.peer_port.get())),
        }),
        NetRepWire::TAG_FLOW_CLOSED => Some(NetResponse::FlowClosed {
            released: w.inet.released.get(),
        }),
        NetRepWire::TAG_PEER_ACCEPTED => Some(NetResponse::PeerAccepted),
        // A zero arrival is not a pair this row ever minted, so an
        // identity-less admission is refused rather than decoded into a
        // `PeerId` nothing answers to.
        NetRepWire::TAG_PEER_ADMITTED => Some(NetResponse::PeerAdmitted {
            peer: PeerId::from_arrival(w.peer_arrival.get())?,
        }),
        NetRepWire::TAG_PEER_CLOSING => Some(NetResponse::PeerClosing),
        NetRepWire::TAG_PEER_RETIRED => Some(NetResponse::PeerRetired),
        NetRepWire::TAG_REFUSED => Some(NetResponse::Refused(NetRefusal::from_code(w.refusal)?)),
        _ => None,
    }
}

/// The radio report as its wire block — also what the launcher's
/// [`NetSnapshot`](kore_abi::monitor::NetSnapshot) stores, so one codec
/// serves both and the snapshot cannot drift from the reply.
pub fn encode_radio(r: &RadioReport) -> NetRadioWire {
    let mut flags = 0;
    if r.secure_boot {
        flags |= NetRadioWire::RADIO_SECURE_BOOT;
    }
    if r.efuse_autoload {
        flags |= NetRadioWire::RADIO_EFUSE_AUTOLOAD;
    }
    if r.normal_mac {
        flags |= NetRadioWire::RADIO_MAC_READY;
    }
    if r.phy_configured {
        flags |= NetRadioWire::RADIO_PHY_CONFIGURED;
    }
    if r.mac_features {
        flags |= NetRadioWire::RADIO_MAC_FEATURES;
    }
    NetRadioWire {
        cut: r.cut,
        fw_header: r.firmware.header,
        fw_major: r.firmware.major,
        fw_minor: r.firmware.minor,
        fw_sub: r.firmware.sub,
        fw_idx: r.firmware.idx,
        country: r.country,
        xtal_cap: r.xtal_cap,
        rfe_type: r.rfe_type,
        flags,
        mss_dev_type: r.mss_dev_type,
        mss_cust_idx: r.mss_cust_idx,
        mss_key_num: r.mss_key_num,
        rx_nss: r.phy.rx_nss,
        bw: r.phy.bw,
        tx_nss: r.phy.tx_nss,
        prot: r.phy.prot,
        nic: r.phy.nic,
        wl_func: r.phy.wl_func,
        hw_type: r.phy.hw_type,
        ant_tx: r.phy.ant_tx,
        ant_rx: r.phy.ant_rx,
        band_sel: r.phy.band_sel,
        fw_commit: U32::new(r.firmware.commit),
        sb_sel_mgn: U32::new(r.sb_sel_mgn),
        download_packets: U32::new(r.download_packets),
    }
}

/// Lift the radio block. Unknown flag bits are not a decode failure here —
/// [`decode_rep`]'s re-encode comparison rejects them, the same way it
/// rejects every other stray bit.
/// The wire block back to a report.
pub fn decode_radio(w: &NetRadioWire) -> RadioReport {
    RadioReport {
        cut: w.cut,
        firmware: RadioFirmware {
            header: w.fw_header,
            major: w.fw_major,
            minor: w.fw_minor,
            sub: w.fw_sub,
            idx: w.fw_idx,
            commit: w.fw_commit.get(),
        },
        country: w.country,
        xtal_cap: w.xtal_cap,
        rfe_type: w.rfe_type,
        efuse_autoload: w.flags & NetRadioWire::RADIO_EFUSE_AUTOLOAD != 0,
        normal_mac: w.flags & NetRadioWire::RADIO_MAC_READY != 0,
        phy_configured: w.flags & NetRadioWire::RADIO_PHY_CONFIGURED != 0,
        mac_features: w.flags & NetRadioWire::RADIO_MAC_FEATURES != 0,
        secure_boot: w.flags & NetRadioWire::RADIO_SECURE_BOOT != 0,
        sb_sel_mgn: w.sb_sel_mgn.get(),
        mss_dev_type: w.mss_dev_type,
        mss_cust_idx: w.mss_cust_idx,
        mss_key_num: w.mss_key_num,
        phy: RadioPhy {
            rx_nss: w.rx_nss,
            bw: w.bw,
            tx_nss: w.tx_nss,
            prot: w.prot,
            nic: w.nic,
            wl_func: w.wl_func,
            hw_type: w.hw_type,
            ant_tx: w.ant_tx,
            ant_rx: w.ant_rx,
            band_sel: w.band_sel,
        },
        download_packets: w.download_packets.get(),
    }
}

/// Pin `(request, response)` pairs and the grant count.
///
/// The uncorrelated-pair gate: a reply that does not answer the verb that was
/// asked is refused before a client can act on it.
pub fn reply_matches(request: &NetReqWire, response: &NetResponse, grants: u8) -> bool {
    // A refusal answers whichever verb asked, so it is the one correlation
    // no row owns — and it carries nothing, so it may ride no grant.
    if matches!(response, NetResponse::Refused(_)) {
        return grants == 0;
    }
    usize::from(grants) == NetVerbs::grants(request) && NetVerbs::answers(request, response)
}

/// The receiver-authored admission post for the reply's one grant lane.
pub fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
    (index == 0).then_some((ResourceKind::ChanRecv, Rights::ACCESS))
}

/// How many grant lanes a request's reply can carry: one for `Signal` (the
/// minted receive doorbell), none for every other verb.  Per verb, not the
/// protocol maximum, so a face that never signals — exec's, which issues
/// only `Query` and `Radio` at a launch — wires no grant lane at all: the
/// client validates its pins against THIS count before publishing, and a
/// pin-less face on a protocol-wide count of one refused every call as
/// `Wiring` before a byte left (#936, `status net: provider unavailable`
/// on a row that was up).  Least authority and a working face are the same
/// answer here.
pub fn request_grants(request: &NetReqWire) -> usize {
    NetVerbs::grants(request)
}

impl SvcProtocol for NetWire {
    type Request = NetRequest;
    type Response = NetResponse;
    type ReqWire = NetReqWire;
    type RepWire = NetRepWire;

    const REPLY_GRANTS: usize = NET_REPLY_GRANTS;
    const SERVE_PAIRS: usize = NET_SERVE_PAIRS;

    fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
        grant_policy(index)
    }
    fn request_grants(request: &NetReqWire) -> usize {
        request_grants(request)
    }
    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn reply_matches(request: &NetReqWire, response: &NetResponse, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
    fn encode_req(req: &NetRequest) -> NetReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &NetReqWire) -> Option<NetRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &NetResponse, grants: u8) -> NetRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &NetRepWire) -> Option<(NetResponse, u8)> {
        decode_rep(wire)
    }
}

/// The **host** side of the same wire.
///
/// Every codec and constant delegates to the free functions above, so byte
/// agreement between a client that named [`NetWire`] and a host that named a
/// hardware backend holds by construction rather than by test.
impl<H: NetHost> SvcProtocol for NetService<H> {
    type Request = NetRequest;
    type Response = NetResponse;
    type ReqWire = NetReqWire;
    type RepWire = NetRepWire;

    const REPLY_GRANTS: usize = NET_REPLY_GRANTS;
    const SERVE_PAIRS: usize = NET_SERVE_PAIRS;

    fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
        grant_policy(index)
    }
    fn request_grants(request: &NetReqWire) -> usize {
        request_grants(request)
    }
    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn reply_matches(request: &NetReqWire, response: &NetResponse, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
    fn encode_req(req: &NetRequest) -> NetReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &NetReqWire) -> Option<NetRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &NetResponse, grants: u8) -> NetRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &NetRepWire) -> Option<(NetResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use kore_abi::session::SlotRef;
    use kore_driver::net::EtherType;

    use super::*;
    use crate::service::{NetInfo, NetStats};

    fn filter() -> FrameFilter {
        let mut f = FrameFilter::protocol(EtherType::KORE_PEER);
        f.multicast = MulticastMatch::Group(MacAddr([0x01, 0x80, 0xc2, 0, 0, 3]));
        f
    }

    /// Every field non-zero and distinct, so a swapped pair of bytes in
    /// either codec fails the round trip.
    fn flow_filter() -> FrameFilter {
        let mut f = FrameFilter::protocol(EtherType::IPV4);
        f.flow = Some(FlowMatch::connection(
            IpProto::TCP,
            (Ipv4Addr([10, 0, 2, 15]), 443),
            (Ipv4Addr([10, 0, 2, 2]), 44000),
        ));
        f
    }

    /// A bound flow survives the wire intact — every dimension, both
    /// endpoints, in both directions.
    #[test]
    fn a_flow_round_trips() {
        let req = NetRequest::OpenVif {
            filter: flow_filter(),
            announced: SlotRef::from_wire(0),
        };
        let w = encode_req(&req);
        assert_eq!(w.flow_kind, NetReqWire::FLOW_SOME);
        assert_eq!(w.family, NetReqWire::FAMILY_IPV4);
        assert_eq!(decode_req(&w), Some(req));
    }

    /// The compatibility property, on the wire: a filter binding no flow
    /// leaves the whole block zero, so the record a pre-flow client would
    /// have sent is byte-identical to the one sent now.
    #[test]
    fn no_flow_leaves_the_block_zero() {
        let w = encode_req(&NetRequest::OpenVif {
            filter: filter(),
            announced: SlotRef::from_wire(0),
        });
        assert_eq!(w.flow_kind, NetReqWire::FLOW_NONE);
        assert_eq!(w.proto_kind, 0);
        assert_eq!(w.local_kind, 0);
        assert_eq!(w.peer_kind, 0);
        assert_eq!(w.proto, 0);
        assert_eq!(w.family, 0);
        assert_eq!(w.local_addr, [0; 4]);
        assert_eq!(w.peer_addr, [0; 4]);
        assert_eq!(w.local_port.get(), 0);
        assert_eq!(w.peer_port.get(), 0);
        assert_eq!(w.reserved3, [0; 1]);
        assert_eq!(w.reserved4, [0; 8]);
        // The two the flow verbs took out of that reserve. A frame verb
        // leaves them zero, which is what keeps this record readable by a
        // peer that predates the arbitrating half.
        assert_eq!(w.prefix_len, 0);
        assert_eq!(w.gateway, [0; 4]);
    }

    /// Fail-closed on the new block: a record claiming no flow but carrying
    /// flow bytes is refused, not masked. The re-encode comparison is what
    /// enforces it, so there is no per-field guard to forget.
    #[test]
    fn a_flow_block_that_should_be_zero_is_refused() {
        let mut w = encode_req(&NetRequest::OpenVif {
            filter: filter(),
            announced: SlotRef::from_wire(0),
        });
        assert!(decode_req(&w).is_some());

        let mut dirty = w;
        dirty.local_addr = [10, 0, 2, 15];
        assert_eq!(decode_req(&dirty), None, "addresses without a bound flow");

        let mut dirty = w;
        dirty.family = NetReqWire::FAMILY_IPV4;
        assert_eq!(decode_req(&dirty), None, "a family without a bound flow");

        w.reserved4[0] = 1;
        assert_eq!(decode_req(&w), None, "reserved bytes are not spare room");
    }

    /// An address family this version does not know is refused rather than
    /// read as IPv4 — which is what makes an IPv6 sibling additive instead of
    /// a silent reinterpretation of these bytes.
    #[test]
    fn an_unknown_address_family_is_refused() {
        let mut w = encode_req(&NetRequest::OpenVif {
            filter: flow_filter(),
            announced: SlotRef::from_wire(0),
        });
        w.family = 6;
        assert_eq!(decode_req(&w), None);
    }

    /// Every endpoint and protocol tag round trips, including the two
    /// widening ones.
    #[test]
    fn every_flow_tag_round_trips() {
        let shapes = [
            FlowMatch::listener(IpProto::UDP, (Ipv4Addr([10, 0, 2, 15]), 68)),
            FlowMatch {
                proto: ProtoMatch::Any,
                local: EndpointMatch::Addr(Ipv4Addr([10, 0, 2, 15])),
                peer: EndpointMatch::Any,
            },
            FlowMatch {
                proto: ProtoMatch::Only(IpProto::ICMP),
                local: EndpointMatch::Any,
                peer: EndpointMatch::Addr(Ipv4Addr([8, 8, 8, 8])),
            },
        ];
        for shape in shapes {
            let mut f = FrameFilter::protocol(EtherType::IPV4);
            f.flow = Some(shape);
            let req = NetRequest::OpenVif {
                filter: f,
                announced: SlotRef::from_wire(0),
            };
            assert_eq!(decode_req(&encode_req(&req)), Some(req), "{shape:?}");
        }
    }

    /// `SetFilter` carries a vif identity *and* a whole filter through the
    /// unchanged record, and both come back intact.
    ///
    /// The verb is additive in the strongest sense available: it reuses the
    /// vif words `CloseVif`/`Stats` already use and the filter block
    /// `OpenVif` already uses, so no byte of `NetReqWire` moved and every
    /// record a pre-`SetFilter` peer can produce still decodes identically.
    #[test]
    fn a_set_filter_round_trips_its_vif_and_its_filter() {
        let vif = VifId {
            index: 3,
            generation: 7,
        };
        for f in [filter(), flow_filter()] {
            let req = NetRequest::SetFilter { vif, filter: f };
            let w = encode_req(&req);
            assert_eq!(w.op, NetReqWire::OP_SET_FILTER);
            assert_eq!(w.vif_index.get(), 3);
            assert_eq!(w.vif_generation.get(), 7);
            assert_eq!(
                w.announced.get(),
                0,
                "no window rides this verb, so the return-landing word stays zero"
            );
            assert_eq!(decode_req(&w), Some(req));
        }
    }

    /// The reply names the vif the new filter took effect on, and survives
    /// the wire.
    ///
    /// A client that re-filters several vifs must be able to tell which
    /// answer belongs to which; a bare acknowledgement could be credited to
    /// the wrong one.
    #[test]
    fn a_filtered_reply_names_its_vif() {
        let rep = NetResponse::Filtered {
            vif: VifId {
                index: 2,
                generation: 9,
            },
        };
        let w = encode_rep(&rep, 0);
        assert_eq!(w.tag, NetRepWire::TAG_FILTERED);
        assert_eq!(decode_rep(&w), Some((rep, 0)));
    }

    /// `SetFilter` is answered by `Filtered` and nothing else, so a reply
    /// that does not answer the verb asked never reaches a client.
    #[test]
    fn a_set_filter_is_correlated_to_its_own_answer() {
        let vif = VifId {
            index: 1,
            generation: 1,
        };
        let w = encode_req(&NetRequest::SetFilter {
            vif,
            filter: filter(),
        });
        assert!(reply_matches(&w, &NetResponse::Filtered { vif }, 0));
        assert!(
            !reply_matches(&w, &NetResponse::Closed { vif }, 0),
            "a close must never be credited to a re-filter"
        );
        assert!(!reply_matches(&w, &NetResponse::Signaled, 1));
        assert!(reply_matches(
            &w,
            &NetResponse::Refused(NetRefusal::NotOwner),
            0
        ));

        let open = encode_req(&NetRequest::OpenVif {
            filter: filter(),
            announced: SlotRef::from_wire(0),
        });
        assert!(
            !reply_matches(&open, &NetResponse::Filtered { vif }, 0),
            "nor a re-filter to an open"
        );
    }

    fn radio() -> RadioReport {
        RadioReport {
            cut: 1,
            firmware: RadioFirmware {
                header: 2,
                major: 3,
                minor: 4,
                sub: 5,
                idx: 6,
                commit: 0x0708_090a,
            },
            country: *b"US",
            xtal_cap: 0x0b,
            rfe_type: 0x0c,
            efuse_autoload: true,
            normal_mac: true,
            phy_configured: true,
            mac_features: true,
            secure_boot: true,
            sb_sel_mgn: 0x0d0e_0f10,
            mss_dev_type: 0x11,
            mss_cust_idx: 0x12,
            mss_key_num: 0x13,
            phy: RadioPhy {
                rx_nss: 0x14,
                bw: 0x15,
                tx_nss: 0x16,
                prot: 0x17,
                nic: 0x18,
                wl_func: 0x19,
                hw_type: 0x1a,
                ant_tx: 0x1b,
                ant_rx: 0x1c,
                band_sel: 0x1d,
            },
            download_packets: 0x1e1f_2021,
        }
    }

    /// The radio block is zero on every tag but `TAG_RADIO`, and a stray
    /// radio flag bit — the one place a flag byte has spare bits — is
    /// refused like any other set reserved bit.
    #[test]
    fn the_radio_block_is_reserved_zero_off_its_own_tag() {
        let info = NetResponse::Info(NetInfo {
            link_up: true,
            link_mbps: 100,
            mac: MacAddr([1, 2, 3, 4, 5, 6]),
            mtu: 1514,
            window_bytes: 4096,
            counters: NicCounters::default(),
            dropped_notices: 0,
        });
        let mut w = encode_rep(&info, 0);
        assert_eq!(w.radio, NetRadioWire::EMPTY);
        w.radio.cut = 1;
        assert!(decode_rep(&w).is_none(), "a radio byte on an Info reply");

        // And the counters block is `TAG_INFO`'s alone.
        let mut w = encode_rep(&NetResponse::Radio(radio()), 0);
        assert_eq!(w.counters, NetCountersWire::EMPTY);
        w.counters.irqs = U64::new(1);
        assert!(decode_rep(&w).is_none(), "a counter on a Radio reply");

        let mut w = encode_rep(&NetResponse::Radio(radio()), 0);
        w.radio.flags |= 1 << 7;
        assert!(decode_rep(&w).is_none(), "an unknown radio flag");
        assert!(reply_matches(
            &encode_req(&NetRequest::Radio),
            &NetResponse::Radio(radio()),
            0
        ));
        assert!(
            !reply_matches(
                &encode_req(&NetRequest::Query),
                &NetResponse::Radio(radio()),
                0
            ),
            "Query is answered by Info, never by a radio report"
        );
    }

    #[test]
    fn mac_and_phy_witnesses_round_trip_without_promoting_legacy_reports() {
        for (normal_mac, phy_configured, mac_features) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, true, true),
        ] {
            let report = RadioReport {
                normal_mac,
                phy_configured,
                mac_features,
                ..radio()
            };
            let wire = encode_rep(&NetResponse::Radio(report), 0);
            assert_eq!(
                wire.radio.flags,
                3 | if normal_mac { 4 } else { 0 }
                    | if phy_configured { 8 } else { 0 }
                    | if mac_features { 16 } else { 0 }
            );
            assert_eq!(decode_rep(&wire), Some((NetResponse::Radio(report), 0)));
        }
    }

    /// A bundle part's wire code is never confusable with another's, and a
    /// code outside the vocabulary is refused rather than decoded into a
    /// lane that happens to be in range.
    #[test]
    fn a_bundle_part_code_is_unambiguous() {
        use kore_rt::svc::PeerBundlePart as P;
        let named = [
            (P::Ring, NetReqWire::PART_RING),
            (P::Kick, NetReqWire::PART_KICK),
            (P::ReturnRecv, NetReqWire::PART_RETURN_RECV),
        ];
        for (part, code) in named {
            let w = encode_req(&NetRequest::AdmitPeer {
                part,
                announced: SlotRef::from_wire(1),
            });
            assert_eq!(w.peer_part, code);
            assert!(
                code < NetReqWire::PART_GRANT_BASE,
                "a named part must never land in the grant band"
            );
        }
        // The byte between the named parts and the lanes names nothing.
        let mut stray = encode_req(&NetRequest::AdmitPeer {
            part: P::Ring,
            announced: SlotRef::from_wire(1),
        });
        stray.peer_part = NetReqWire::PART_GRANT_BASE - 1;
        assert_eq!(
            decode_req(&stray),
            None,
            "an unassigned part code is refused"
        );
    }

    /// `peer_part` is zero on every verb that is not assembling a bundle, so
    /// a stray value in it is caught by the re-encode comparison rather than
    /// silently ignored.
    #[test]
    fn a_stray_peer_part_is_refused_on_another_verb() {
        let mut w = encode_req(&NetRequest::Query);
        assert_eq!(w.peer_part, 0, "Query sets no part");
        w.peer_part = NetReqWire::PART_KICK;
        assert_eq!(
            decode_req(&w),
            None,
            "the re-encode comparison rejects a field this verb never sets"
        );
    }

    /// An admission reply carries the minted pair's identity, and an
    /// identity-less one is refused rather than decoded into a `PeerId`
    /// nothing answers to.
    #[test]
    fn an_admission_reply_carries_its_peer_and_refuses_a_zero() {
        // An arrival is `generation * SLOT_CAPACITY + slot`, and a zero
        // generation names no live pair — which is what makes the zeroed
        // record below a refusal rather than a decode.
        let arrival = kore_rt::svc::PeerId::SLOT_CAPACITY as u16 + 5;
        let peer = kore_rt::svc::PeerId::from_arrival(arrival).expect("a live arrival");
        let wire = encode_rep(&NetResponse::PeerAdmitted { peer }, 0);
        assert_eq!(wire.tag, NetRepWire::TAG_PEER_ADMITTED);
        assert_eq!(
            decode_rep(&wire),
            Some((NetResponse::PeerAdmitted { peer }, 0))
        );

        let mut zeroed = wire;
        zeroed.peer_arrival = U16::ZERO;
        assert_eq!(decode_rep(&zeroed), None, "arrival zero names no pair");
    }

    /// A retirement verb naming a zero generation names no live pair, and is
    /// refused rather than decoded into one.
    #[test]
    fn a_retirement_of_no_generation_is_refused() {
        let peer = kore_rt::svc::PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16)
            .expect("a live arrival");
        let mut wire = encode_req(&NetRequest::RetirePeer { peer });
        assert!(decode_req(&wire).is_some(), "a live generation decodes");
        wire.announced = U32::ZERO;
        assert_eq!(decode_req(&wire), None, "generation zero names no pair");
    }

    /// Every verb survives the round trip, filter and all.
    #[test]
    fn every_request_round_trips() {
        let vif = VifId {
            index: 3,
            generation: 7,
        };
        for req in [
            NetRequest::Query,
            NetRequest::Signal,
            NetRequest::Radio,
            NetRequest::TakePeerAdmission,
            NetRequest::ClosePeer,
            NetRequest::RetirePeer {
                peer: kore_rt::svc::PeerId::from_arrival(
                    kore_rt::svc::PeerId::SLOT_CAPACITY as u16 + 2,
                )
                .expect("a live arrival"),
            },
            NetRequest::ConfirmRetired {
                peer: kore_rt::svc::PeerId::from_arrival(
                    kore_rt::svc::PeerId::SLOT_CAPACITY as u16 + 3,
                )
                .expect("a live arrival"),
            },
            NetRequest::AdmitPeer {
                part: kore_rt::svc::PeerBundlePart::Ring,
                announced: SlotRef::from_wire(9),
            },
            NetRequest::AdmitPeer {
                part: kore_rt::svc::PeerBundlePart::Kick,
                announced: SlotRef::from_wire(10),
            },
            NetRequest::AdmitPeer {
                part: kore_rt::svc::PeerBundlePart::ReturnRecv,
                announced: SlotRef::from_wire(11),
            },
            NetRequest::AdmitPeer {
                part: kore_rt::svc::PeerBundlePart::Grant(0),
                announced: SlotRef::from_wire(12),
            },
            NetRequest::CloseVif { vif },
            NetRequest::Stats { vif },
            NetRequest::SetFilter {
                vif,
                filter: filter(),
            },
            NetRequest::OpenVif {
                filter: filter(),
                announced: SlotRef::from_wire(42),
            },
            NetRequest::OpenVif {
                filter: FrameFilter {
                    flow: None,
                    ethertype: EtherTypeMatch::Any,
                    unicast: true,
                    multicast: MulticastMatch::All,
                    broadcast: true,
                    promiscuous: true,
                },
                announced: SlotRef::from_wire(1),
            },
        ] {
            assert_eq!(decode_req(&encode_req(&req)), Some(req), "{req:?}");
        }
    }

    /// The decode is byte-exact fail-closed: the re-encode comparison rejects
    /// any record whose reserved bytes, unused words, or dead EtherType slots
    /// carry anything — one comparison instead of a guard per field, and it
    /// cannot drift from the encoder the way hand-written guards do.
    #[test]
    fn a_stray_bit_anywhere_refuses() {
        let req = NetRequest::Query;
        let clean = encode_req(&req);
        assert!(decode_req(&clean).is_some());

        let mut dirty = clean;
        dirty.reserved[1] = 1;
        assert!(decode_req(&dirty).is_none(), "a set reserved byte");

        let mut dirty = clean;
        dirty.reserved2[0] = 0x80;
        assert!(decode_req(&dirty).is_none(), "a set trailing reserved byte");

        let mut dirty = clean;
        dirty.vif_index = U16::new(4);
        assert!(
            decode_req(&dirty).is_none(),
            "a verb that names no vif may not carry one"
        );

        let mut dirty = clean;
        dirty.announced = U32::new(9);
        assert!(decode_req(&dirty).is_none(), "an unused word");

        // A live EtherType past the declared count is the dead-slot case.
        let mut dirty = encode_req(&NetRequest::OpenVif {
            filter: FrameFilter::protocol(EtherType::ARP),
            announced: SlotRef::from_wire(1),
        });
        assert!(decode_req(&dirty).is_some());
        dirty.ethertypes[2] = U16::new(0x0800);
        assert!(decode_req(&dirty).is_none(), "a live dead slot");

        let mut dirty = clean;
        dirty.op = 0;
        assert!(decode_req(&dirty).is_none(), "0 is not a verb");
        dirty.op = 99;
        assert!(decode_req(&dirty).is_none(), "and neither is 99");
    }

    /// A filter flag this version does not define is refused, not masked off:
    /// a future revision's bit must not be silently reinterpreted as a
    /// narrower filter than the client asked for.
    #[test]
    fn an_undefined_filter_bit_refuses() {
        let mut w = encode_req(&NetRequest::OpenVif {
            filter: FrameFilter::protocol(EtherType::ARP),
            announced: SlotRef::from_wire(1),
        });
        w.flags |= 1 << 5;
        assert!(decode_req(&w).is_none());
    }

    /// An EtherType count past what the record can hold refuses rather than
    /// reading a truncated list.
    #[test]
    fn an_overlong_ethertype_count_refuses() {
        let mut w = encode_req(&NetRequest::OpenVif {
            filter: FrameFilter::protocol(EtherType::ARP),
            announced: SlotRef::from_wire(1),
        });
        w.ethertype_count = (MAX_NET_ETHERTYPES + 1) as u8;
        assert!(decode_req(&w).is_none());
    }

    /// Replies round-trip, and the grant count is pinned per tag: exactly one
    /// tag carries authority, so a reply that claims a grant it did not bring
    /// — or brings one it did not announce — never reaches possession.
    #[test]
    fn replies_round_trip_and_pin_their_grant_count() {
        let vif = VifId {
            index: 2,
            generation: 5,
        };
        let cases = [
            (
                NetResponse::Info(NetInfo {
                    link_up: true,
                    link_mbps: 1000,
                    mac: MacAddr([1, 2, 3, 4, 5, 6]),
                    mtu: 1514,
                    window_bytes: 131_968,
                    counters: NicCounters {
                        irqs: 0x0102_0304_0506_0708,
                        link_changes: 3,
                        overruns: 1,
                        rx_errors: 2,
                    },
                    dropped_notices: 4,
                }),
                0u8,
            ),
            (
                NetResponse::Opened {
                    vif,
                    mac: MacAddr([2, 0, 0, 0, 0, 1]),
                },
                0,
            ),
            (NetResponse::Closed { vif }, 0),
            (NetResponse::Filtered { vif }, 0),
            (NetResponse::Signaled, 1),
            (
                NetResponse::Stats {
                    vif,
                    stats: NetStats {
                        rx_frames: 9,
                        tx_frames: 4,
                        dropped: 1,
                    },
                },
                0,
            ),
            (NetResponse::Refused(NetRefusal::StaleVif), 0),
            (NetResponse::Radio(radio()), 0),
        ];
        for (rep, grants) in cases {
            let w = encode_rep(&rep, grants);
            assert_eq!(decode_rep(&w), Some((rep, grants)), "{rep:?}");
            let mut wrong = w;
            wrong.grants = if grants == 0 { 1 } else { 0 };
            assert!(
                decode_rep(&wrong).is_none(),
                "{rep:?} with the wrong grant count"
            );
        }
    }

    /// The uncorrelated-pair gate: a reply that answers a verb nobody asked
    /// is refused before a client can act on it.
    #[test]
    fn a_reply_must_answer_the_verb_that_was_asked() {
        let query = encode_req(&NetRequest::Query);
        let signal = encode_req(&NetRequest::Signal);
        let info = NetResponse::Info(NetInfo {
            link_up: false,
            link_mbps: 0,
            mac: MacAddr::ZERO,
            mtu: 1514,
            window_bytes: 0,
            counters: NicCounters::default(),
            dropped_notices: 0,
        });
        assert!(reply_matches(&query, &info, 0));
        assert!(!reply_matches(&signal, &info, 0), "Signal answers Signaled");
        assert!(reply_matches(&signal, &NetResponse::Signaled, 1));
        assert!(
            !reply_matches(&query, &NetResponse::Signaled, 1),
            "Query may not come back carrying authority"
        );
        assert!(
            reply_matches(&query, &NetResponse::Refused(NetRefusal::NotReady), 0),
            "any verb may be refused"
        );
        assert!(!reply_matches(
            &query,
            &NetResponse::Refused(NetRefusal::NotReady),
            1
        ));
    }

    /// The one grant lane is admission-posted for a channel receive end and
    /// nothing else; a second lane does not exist.
    #[test]
    fn the_grant_lane_is_pinned_to_a_channel_receive() {
        assert_eq!(
            grant_policy(0),
            Some((ResourceKind::ChanRecv, Rights::ACCESS))
        );
        assert_eq!(grant_policy(1), None);
        assert_eq!(NET_REPLY_GRANTS, 1);
        assert!(NET_SERVE_PAIRS <= kore_rt::svc::MAX_SERVE_PAIRS);
    }

    /// Only `Signal` asks for the lane; every other verb's reply carries no
    /// grant, so a face that never signals needs no pin — and the answer is
    /// the same through every spelling of the protocol.
    #[test]
    fn only_signal_requests_the_grant_lane() {
        // The real codec's wire for the two verbs exec issues and the one
        // that mints; the remaining ops are set on that wire directly.
        assert_eq!(request_grants(&encode_req(&NetRequest::Query)), 0);
        assert_eq!(request_grants(&encode_req(&NetRequest::Radio)), 0);
        assert_eq!(request_grants(&encode_req(&NetRequest::Signal)), 1);
        let mut w = encode_req(&NetRequest::Query);
        for (op, expect) in [
            (NetReqWire::OP_QUERY, 0),
            (NetReqWire::OP_OPEN_VIF, 0),
            (NetReqWire::OP_CLOSE_VIF, 0),
            (NetReqWire::OP_SIGNAL, 1),
            (NetReqWire::OP_STATS, 0),
            (NetReqWire::OP_RADIO, 0),
        ] {
            w.op = op;
            assert_eq!(request_grants(&w), expect, "op {op}");
            assert_eq!(<NetWire as SvcProtocol>::request_grants(&w), expect);
            assert_eq!(<NetRow as SvcProtocol>::request_grants(&w), expect);
            assert!(expect <= <NetWire as SvcProtocol>::REPLY_GRANTS);
        }
    }

    /// The founder's exec-provider rule, pinned where the row declares its
    /// pairs: an `exec_client` row serves at least two, the final one is
    /// exec's, and pair 0 stays the row's own.
    #[test]
    fn the_final_pair_is_execs_and_pair_zero_is_the_rows_own() {
        assert_eq!(NET_SERVE_PAIRS, 2);
        assert!(NET_SERVE_PAIRS >= 2);
        assert_eq!(NET_EXEC_PAIR, 1);
        assert_ne!(NET_EXEC_PAIR, 0);
    }
}

/// The **row**'s side of the same wire, delegating identically.
///
/// Two hosts and one client marker all speak these bytes; every one of them
/// routes through the free functions above, so byte agreement between the
/// founded row, the generic engine, and a consumer that named [`NetWire`]
/// holds by construction rather than by review.
impl SvcProtocol for NetRow {
    type Request = NetRequest;
    type Response = NetResponse;
    type ReqWire = NetReqWire;
    type RepWire = NetRepWire;

    const REPLY_GRANTS: usize = NET_REPLY_GRANTS;
    const SERVE_PAIRS: usize = NET_SERVE_PAIRS;

    fn grant_policy(index: usize) -> Option<(ResourceKind, Rights)> {
        grant_policy(index)
    }
    fn request_grants(request: &NetReqWire) -> usize {
        request_grants(request)
    }
    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn reply_matches(request: &NetReqWire, response: &NetResponse, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
    fn encode_req(req: &NetRequest) -> NetReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &NetReqWire) -> Option<NetRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &NetResponse, grants: u8) -> NetRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &NetRepWire) -> Option<(NetResponse, u8)> {
        decode_rep(wire)
    }
}
