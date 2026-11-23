//! **Vif policy** — what a client's authority permits it to ask for.
//!
//! A client does not get the filter it requests; it gets the filter its
//! authority allows, or a refusal naming what exceeded.  This is the
//! difference between a capability system and a configuration file: the
//! request is data, the permission is authority, and they are checked against
//! each other at one place.
//!
//! Every widening power is **separate and default-off**, because they widen
//! different things:
//!
//! - [`EtherTypeMatch::Any`] exposes protocols other clients own.
//! - [`promiscuous`](VifPolicy::promiscuous) exposes traffic addressed to
//!   other stations.
//! - [`source_mac`](VifPolicy::source_mac) lets a client *speak as* an
//!   address the service did not assign it.
//! - [`flow`](VifPolicy::flow) bounds which *connections* a client may bind,
//!   and each of its three dimensions is refused by name.
//!
//! Bundling them into one "privileged" bit would mean a client that needs to
//! answer broadcast also gets to impersonate its neighbour.

use super::addr::MacAddr;
use super::filter::{EtherTypeMatch, FrameFilter, MulticastMatch};
use super::flow::FlowMatch;

/// What a client's authority permits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VifPolicy {
    /// EtherTypes this authority may bind.
    pub ethertype: EtherTypeMatch,
    /// Multicast groups this authority may join.
    pub multicast: MulticastMatch,
    /// May bind broadcast receive.
    pub broadcast: bool,
    /// May receive frames addressed to other stations.
    pub promiscuous: bool,
    /// May transmit from a source address the service did not assign.
    pub source_mac: bool,
    /// Vifs this authority may hold open at once.
    pub max_vifs: u8,
    /// The widest flow this authority may bind, or `None` to impose no L3
    /// restriction at all.
    ///
    /// The containment here runs **opposite** to the matches above, because
    /// `None` is the *widest* value rather than the narrowest: a grant of
    /// `None` admits any request, and a request that binds no flow under a
    /// grant that names one is a widening, refused as
    /// [`PolicyRefusal::FlowRequired`].
    ///
    /// That inversion is what keeps every policy written before flows existed
    /// meaning what it meant — `None` is "this authority says nothing about
    /// L3", which was the only thing it could say.
    pub flow: Option<FlowMatch>,
}

impl VifPolicy {
    /// The policy an ordinary protocol client gets: one named EtherType, its
    /// own assigned address, broadcast, and nothing else.
    pub fn protocol(ethertype: super::addr::EtherType) -> Self {
        Self {
            ethertype: EtherTypeMatch::one(ethertype),
            multicast: MulticastMatch::None,
            broadcast: true,
            promiscuous: false,
            source_mac: false,
            max_vifs: 1,
            flow: None,
        }
    }

    /// The policy #387's peer link needs: the kore peer EtherType, its
    /// discovery multicast group, and broadcast — still no promiscuity and no
    /// caller-chosen source address.
    pub fn peer(group: MacAddr) -> Self {
        Self {
            ethertype: EtherTypeMatch::one(super::addr::EtherType::KORE_PEER),
            multicast: MulticastMatch::Group(group),
            broadcast: true,
            promiscuous: false,
            source_mac: false,
            max_vifs: 1,
            flow: None,
        }
    }

    /// Deny everything.  The policy an authority carries until something
    /// grants it more — so a missing grant fails closed.
    pub const DENY: Self = Self {
        ethertype: EtherTypeMatch::Only {
            types: [super::addr::EtherType(0); super::filter::MAX_FILTER_ETHERTYPES],
            count: 0,
        },
        multicast: MulticastMatch::None,
        broadcast: false,
        promiscuous: false,
        source_mac: false,
        max_vifs: 0,
        // `max_vifs: 0` is what denies here; an L3 restriction on an
        // authority that may open nothing would be decoration.
        flow: None,
    };
}

/// Why a vif request was refused.
///
/// Each variant names the *specific* power that was exceeded.  A single
/// "denied" would leave a caller unable to tell a misconfiguration from an
/// attempted escalation, and would leave an operator unable to tell which
/// grant to widen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyRefusal {
    /// The request named an EtherType the authority may not bind, or claimed
    /// [`EtherTypeMatch::Any`] without the grant.
    EtherType,
    /// The request joined a multicast group the authority may not.
    Multicast,
    /// The request asked for broadcast receive without the grant.
    Broadcast,
    /// The request asked to receive other stations' traffic without the grant.
    Promiscuous,
    /// The request asked to transmit from an unassigned source address.
    SourceMac,
    /// The authority may only bind a restricted flow, and the request bound
    /// none — which would have accepted every datagram reaching its address.
    FlowRequired,
    /// The request named an IP protocol the authority may not bind, or
    /// claimed [`ProtoMatch::Any`](super::flow::ProtoMatch::Any) without it.
    FlowProto,
    /// The request named a local endpoint outside the authority's grant —
    /// binding an address or port that is not its to bind.
    FlowLocal,
    /// The request named a peer endpoint outside the authority's grant.
    FlowPeer,
    /// The authority already holds [`VifPolicy::max_vifs`] vifs open.
    TooManyVifs,
}

impl VifPolicy {
    /// Authorize `request`, returning the filter to install.
    ///
    /// **Refuses rather than narrows.** Silently handing back a narrower
    /// filter than was asked for produces a client that believes it is
    /// listening for something it will never receive — a failure that
    /// surfaces much later, somewhere else, as "the protocol doesn't work".
    /// A refusal surfaces it here, naming the power that was missing.
    pub fn authorize(
        &self,
        request: &FrameFilter,
        open_vifs: usize,
    ) -> Result<FrameFilter, PolicyRefusal> {
        if open_vifs >= self.max_vifs as usize {
            return Err(PolicyRefusal::TooManyVifs);
        }
        self.authorize_filter(request)
    }

    /// Authorize `request` **for a vif this authority already holds**.
    ///
    /// Every dimension of [`authorize`](Self::authorize) except the open-vif
    /// census: re-filtering an open vif creates no new vif, so counting the
    /// one being re-filtered would refuse a client at exactly its granted
    /// ceiling for doing nothing new. Split out rather than duplicated, so a
    /// dimension added above is enforced on both paths or on neither — the
    /// drift that a second hand-written check would eventually produce.
    pub fn authorize_filter(&self, request: &FrameFilter) -> Result<FrameFilter, PolicyRefusal> {
        if !request.ethertype.within(&self.ethertype) {
            return Err(PolicyRefusal::EtherType);
        }
        if !request.multicast.within(&self.multicast) {
            return Err(PolicyRefusal::Multicast);
        }
        if request.broadcast && !self.broadcast {
            return Err(PolicyRefusal::Broadcast);
        }
        if request.promiscuous && !self.promiscuous {
            return Err(PolicyRefusal::Promiscuous);
        }
        // Checked per dimension rather than wholesale, so the refusal names
        // which half of the flow exceeded — the rule the rest of this
        // vocabulary already follows.
        match (&request.flow, &self.flow) {
            (_, None) => {}
            // Releasing an authority installs the empty predicate
            // ([`FrameFilter::CLOSED`]), which binds no flow because it binds
            // nothing at all. `FlowRequired` exists to stop a filter that
            // would accept every datagram reaching its address; one that
            // accepts no frame whatsoever is the opposite of that, and
            // refusing it here would leave the authority being released still
            // enforced.
            (None, Some(_)) if request.admits_nothing() => {}
            (None, Some(_)) => return Err(PolicyRefusal::FlowRequired),
            (Some(r), Some(g)) => {
                if !r.proto.within(&g.proto) {
                    return Err(PolicyRefusal::FlowProto);
                }
                if !r.local.within(&g.local) {
                    return Err(PolicyRefusal::FlowLocal);
                }
                if !r.peer.within(&g.peer) {
                    return Err(PolicyRefusal::FlowPeer);
                }
            }
        }
        Ok(*request)
    }

    /// Authorize a transmit's source address against `assigned`, the address
    /// the service gave this vif.
    ///
    /// A client that may not choose its own source may only send as itself.
    /// This is checked on **every** frame, not once at open: the source
    /// address lives in the frame body, which the client writes into a shared
    /// window and can change after the fact.
    pub fn authorize_source(
        &self,
        source: MacAddr,
        assigned: MacAddr,
    ) -> Result<(), PolicyRefusal> {
        if source == assigned || self.source_mac {
            Ok(())
        } else {
            Err(PolicyRefusal::SourceMac)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::addr::EtherType;
    use crate::net::filter::MAX_FILTER_ETHERTYPES;

    const GROUP: MacAddr = MacAddr([0x01, 0x80, 0xc2, 0, 0, 0x03]);
    const ASSIGNED: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const NEIGHBOUR: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

    /// The headline hostile case: a client whose authority names one protocol
    /// asks for `Any`.  It must be **refused by name**, not quietly narrowed —
    /// a narrowed filter produces a client that believes it is listening for
    /// something it will never receive.
    #[test]
    fn a_forged_any_is_refused_by_name() {
        let policy = VifPolicy::protocol(EtherType::KORE_PEER);
        let mut request = FrameFilter::protocol(EtherType::KORE_PEER);
        request.ethertype = EtherTypeMatch::Any;

        assert_eq!(policy.authorize(&request, 0), Err(PolicyRefusal::EtherType));
    }

    /// Naming a protocol the authority does not carry is the same refusal.
    #[test]
    fn an_unauthorized_ethertype_is_refused() {
        let policy = VifPolicy::protocol(EtherType::KORE_PEER);
        let request = FrameFilter::protocol(EtherType::IPV4);
        assert_eq!(policy.authorize(&request, 0), Err(PolicyRefusal::EtherType));

        // And the authorized one passes through unchanged.
        let ok = FrameFilter::protocol(EtherType::KORE_PEER);
        assert_eq!(policy.authorize(&ok, 0), Ok(ok));
    }

    /// Each widening power is separate, so needing one never grants another.
    #[test]
    fn each_widening_power_is_refused_independently() {
        let policy = VifPolicy::peer(GROUP);
        let base = || {
            let mut f = FrameFilter::protocol(EtherType::KORE_PEER);
            f.multicast = MulticastMatch::Group(GROUP);
            f
        };

        assert!(policy.authorize(&base(), 0).is_ok());

        let mut promisc = base();
        promisc.promiscuous = true;
        assert_eq!(
            policy.authorize(&promisc, 0),
            Err(PolicyRefusal::Promiscuous)
        );

        let mut all_groups = base();
        all_groups.multicast = MulticastMatch::All;
        assert_eq!(
            policy.authorize(&all_groups, 0),
            Err(PolicyRefusal::Multicast)
        );

        let mut elsewhere = base();
        elsewhere.multicast = MulticastMatch::Group(MacAddr([0x01, 0, 0, 0, 0, 0x09]));
        assert_eq!(
            policy.authorize(&elsewhere, 0),
            Err(PolicyRefusal::Multicast)
        );

        let mut no_broadcast_policy = policy;
        no_broadcast_policy.broadcast = false;
        assert_eq!(
            no_broadcast_policy.authorize(&base(), 0),
            Err(PolicyRefusal::Broadcast)
        );
    }

    /// The vif budget is authority too, and is checked before anything else
    /// is validated.
    #[test]
    fn the_vif_budget_is_enforced() {
        let policy = VifPolicy::protocol(EtherType::KORE_PEER);
        let request = FrameFilter::protocol(EtherType::KORE_PEER);
        assert!(policy.authorize(&request, 0).is_ok());
        assert_eq!(
            policy.authorize(&request, 1),
            Err(PolicyRefusal::TooManyVifs),
            "max_vifs is 1 for an ordinary protocol client"
        );
    }

    /// Source-address spoofing is refused per frame, because the source lives
    /// in the frame body — in a window the client keeps writing to after the
    /// vif was opened.
    #[test]
    fn transmitting_as_another_station_is_refused() {
        let policy = VifPolicy::protocol(EtherType::KORE_PEER);
        assert!(policy.authorize_source(ASSIGNED, ASSIGNED).is_ok());
        assert_eq!(
            policy.authorize_source(NEIGHBOUR, ASSIGNED),
            Err(PolicyRefusal::SourceMac)
        );

        let mut privileged = policy;
        privileged.source_mac = true;
        assert!(privileged.authorize_source(NEIGHBOUR, ASSIGNED).is_ok());
    }

    /// The default-deny policy admits nothing at all — a missing grant must
    /// fail closed, not open.
    #[test]
    fn deny_admits_nothing() {
        let request = FrameFilter::protocol(EtherType::KORE_PEER);
        assert_eq!(
            VifPolicy::DENY.authorize(&request, 0),
            Err(PolicyRefusal::TooManyVifs),
            "zero vifs is the first wall"
        );
        let mut one = VifPolicy::DENY;
        one.max_vifs = 1;
        assert_eq!(one.authorize(&request, 0), Err(PolicyRefusal::EtherType));
        assert_eq!(
            one.ethertype.listed().len(),
            0,
            "DENY names no EtherType at all"
        );
        assert!(MAX_FILTER_ETHERTYPES > 0);
    }

    // ── the flow dimension ──────────────────────────────────────────────

    const LOCAL: crate::net::flow::Ipv4Addr = crate::net::flow::Ipv4Addr([10, 0, 2, 15]);
    const REMOTE: crate::net::flow::Ipv4Addr = crate::net::flow::Ipv4Addr([10, 0, 2, 2]);

    fn ip_policy(flow: Option<crate::net::flow::FlowMatch>) -> VifPolicy {
        let mut p = VifPolicy::protocol(EtherType::IPV4);
        p.flow = flow;
        p
    }

    fn ip_request(flow: Option<crate::net::flow::FlowMatch>) -> FrameFilter {
        let mut f = FrameFilter::protocol(EtherType::IPV4);
        f.flow = flow;
        f
    }

    /// The compatibility property the additive design rests on: an authority
    /// that says nothing about L3 behaves exactly as it did before flows
    /// existed, for both kinds of request.
    #[test]
    fn a_grant_with_no_flow_is_unchanged() {
        let policy = ip_policy(None);

        assert!(policy.authorize(&ip_request(None), 0).is_ok());
        assert!(
            policy
                .authorize(
                    &ip_request(Some(crate::net::flow::FlowMatch::connection(
                        crate::net::flow::IpProto::TCP,
                        (LOCAL, 443),
                        (REMOTE, 44000),
                    ))),
                    0
                )
                .is_ok(),
            "narrowing yourself is never an escalation"
        );
    }

    /// The inversion that makes `None` dangerous to get wrong: an authority
    /// restricted to one flow must not be satisfied by a request that binds
    /// none, because that request accepts every datagram reaching the address.
    #[test]
    fn an_unrestricted_request_under_a_restricting_grant_is_refused() {
        let policy = ip_policy(Some(crate::net::flow::FlowMatch::connection(
            crate::net::flow::IpProto::TCP,
            (LOCAL, 443),
            (REMOTE, 44000),
        )));

        assert_eq!(
            policy.authorize(&ip_request(None), 0),
            Err(PolicyRefusal::FlowRequired)
        );
    }

    /// Each widening dimension is refused **by name**, separately — the rule
    /// the rest of this vocabulary already follows, so an operator can tell
    /// which grant to widen and a caller can tell escalation from typo.
    #[test]
    fn each_flow_dimension_is_refused_by_name() {
        use crate::net::flow::{EndpointMatch, FlowMatch, IpProto, ProtoMatch};

        let grant = FlowMatch::connection(IpProto::TCP, (LOCAL, 443), (REMOTE, 44000));
        let policy = ip_policy(Some(grant));

        let mut wide_proto = grant;
        wide_proto.proto = ProtoMatch::Any;
        assert_eq!(
            policy.authorize(&ip_request(Some(wide_proto)), 0),
            Err(PolicyRefusal::FlowProto),
            "any-protocol is not within one named protocol"
        );

        let mut wide_local = grant;
        wide_local.local = EndpointMatch::Any;
        assert_eq!(
            policy.authorize(&ip_request(Some(wide_local)), 0),
            Err(PolicyRefusal::FlowLocal),
            "any local endpoint reaches other vifs' addresses"
        );

        let mut wide_peer = grant;
        wide_peer.peer = EndpointMatch::Any;
        assert_eq!(
            policy.authorize(&ip_request(Some(wide_peer)), 0),
            Err(PolicyRefusal::FlowPeer)
        );

        let mut other_port = grant;
        other_port.local = EndpointMatch::AddrPort {
            addr: LOCAL,
            port: 80,
        };
        assert_eq!(
            policy.authorize(&ip_request(Some(other_port)), 0),
            Err(PolicyRefusal::FlowLocal),
            "binding a port that is not yours to bind"
        );
    }

    /// A grant by address admits a request that pins a port under it, and
    /// never the reverse — the asymmetry `EndpointMatch::within` encodes.
    #[test]
    fn a_port_pin_is_within_an_address_grant() {
        use crate::net::flow::{EndpointMatch, FlowMatch, IpProto, ProtoMatch};

        let policy = ip_policy(Some(FlowMatch {
            proto: ProtoMatch::Only(IpProto::UDP),
            local: EndpointMatch::Addr(LOCAL),
            peer: EndpointMatch::Any,
        }));

        let pinned = FlowMatch {
            proto: ProtoMatch::Only(IpProto::UDP),
            local: EndpointMatch::AddrPort {
                addr: LOCAL,
                port: 68,
            },
            peer: EndpointMatch::Any,
        };
        assert!(policy.authorize(&ip_request(Some(pinned)), 0).is_ok());

        let elsewhere = FlowMatch {
            proto: ProtoMatch::Only(IpProto::UDP),
            local: EndpointMatch::Addr(REMOTE),
            peer: EndpointMatch::Any,
        };
        assert_eq!(
            policy.authorize(&ip_request(Some(elsewhere)), 0),
            Err(PolicyRefusal::FlowLocal),
            "another station's address is not within this grant"
        );
    }

    // ── re-filtering an open vif ────────────────────────────────────────

    /// The census is the *only* difference between the two entry points, so a
    /// dimension enforced at open is enforced on a re-filter too.
    ///
    /// Split rather than duplicated precisely so this holds: a second
    /// hand-written check would eventually drift and leave a dimension
    /// enforced on one path only — which is a policy bypass reachable by
    /// asking twice.
    #[test]
    fn a_re_filter_is_checked_on_every_dimension_an_open_is() {
        let policy = VifPolicy::protocol(EtherType::KORE_PEER);
        let mut capture = FrameFilter::protocol(EtherType::KORE_PEER);
        capture.promiscuous = true;
        assert_eq!(
            policy.authorize_filter(&capture),
            Err(PolicyRefusal::Promiscuous)
        );
        assert_eq!(
            policy.authorize_filter(&capture),
            policy.authorize(&capture, 0),
            "the two answers agree wherever the census does not decide"
        );

        let mut any = FrameFilter::protocol(EtherType::KORE_PEER);
        any.ethertype = EtherTypeMatch::Any;
        assert_eq!(policy.authorize_filter(&any), Err(PolicyRefusal::EtherType));

        let narrow = ip_policy(Some(crate::net::flow::FlowMatch::connection(
            crate::net::flow::IpProto::UDP,
            (LOCAL, 68),
            (REMOTE, 67),
        )));
        assert_eq!(
            narrow.authorize_filter(&ip_request(None)),
            Err(PolicyRefusal::FlowRequired),
            "and the flow dimension too"
        );
    }

    /// A vif at the authority's ceiling may still be re-filtered: no new vif
    /// is created, so the open-vif census must not be consulted.
    ///
    /// Counting it would refuse exactly the client that has spent its whole
    /// grant — the one with the most reason to narrow what it already holds.
    #[test]
    fn a_re_filter_does_not_consult_the_vif_census() {
        let policy = VifPolicy::protocol(EtherType::IPV4);
        assert_eq!(policy.max_vifs, 1);
        let request = FrameFilter::protocol(EtherType::IPV4);
        assert_eq!(
            policy.authorize(&request, 1),
            Err(PolicyRefusal::TooManyVifs),
            "a second vif is refused"
        );
        assert!(
            policy.authorize_filter(&request).is_ok(),
            "but re-filtering the one it holds is not"
        );
    }

    /// **Releasing an authority is never refused for binding no flow.** The
    /// empty predicate accepts no frame at all, so the hazard
    /// `FlowRequired` exists to stop — a filter accepting every datagram
    /// reaching its address — provably does not apply to it.
    ///
    /// Without this, a grant that names a flow would refuse the release of
    /// its own authority, leaving the old filter installed and enforced after
    /// the capability behind it was given up. Fail-open, and silently.
    #[test]
    fn the_empty_predicate_is_installable_under_a_flow_bearing_grant() {
        let narrow = ip_policy(Some(crate::net::flow::FlowMatch::connection(
            crate::net::flow::IpProto::UDP,
            (LOCAL, 68),
            (REMOTE, 67),
        )));
        assert_eq!(
            narrow.authorize_filter(&ip_request(None)),
            Err(PolicyRefusal::FlowRequired),
            "a request that would admit everything at this address is refused"
        );
        assert_eq!(
            narrow.authorize_filter(&FrameFilter::CLOSED),
            Ok(FrameFilter::CLOSED),
            "but one that admits nothing is not the hazard the rule guards"
        );
        assert_eq!(
            VifPolicy::DENY.authorize_filter(&FrameFilter::CLOSED),
            Ok(FrameFilter::CLOSED),
            "and the deny-everything authority can still release"
        );
    }
}
