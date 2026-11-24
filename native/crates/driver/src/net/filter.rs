//! **Frame filters** — what a vif receives, and the matching that decides it.
//!
//! A filter is the vif's whole receive specification.  It replaces the socket:
//! a client does not bind a port, it binds a predicate over L2 frames, and the
//! classifier delivers a frame to exactly the vifs whose predicate accepts it.

use super::addr::{EtherType, MacAddr};
use super::flow::{FiveTuple, FlowMatch};
use super::frame::FrameHead;

/// Distinct EtherTypes one filter may name.
///
/// A design bound: a vif is a protocol endpoint, not a packet-capture tap.
/// Wanting more than this is wanting [`EtherTypeMatch::Any`], which is a
/// separately-granted power (see [`VifPolicy`](super::policy::VifPolicy)).
pub const MAX_FILTER_ETHERTYPES: usize = 4;

/// Which EtherTypes a vif accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtherTypeMatch {
    /// Exactly the listed types.  `count` names how many entries of `types`
    /// are live; the rest are ignored.
    Only {
        /// The accepted types.
        types: [EtherType; MAX_FILTER_ETHERTYPES],
        /// How many entries of `types` are live.
        count: u8,
    },
    /// Every EtherType.
    ///
    /// This is a **capture** power, not a convenience: it lets the holder read
    /// every protocol crossing the vif's address, including protocols owned by
    /// other clients.  It is granted separately and refused by default.
    Any,
}

impl EtherTypeMatch {
    /// A filter accepting exactly `t`.
    pub fn one(t: EtherType) -> Self {
        let mut types = [EtherType(0); MAX_FILTER_ETHERTYPES];
        types[0] = t;
        Self::Only { types, count: 1 }
    }

    /// A filter accepting each of `list`, or `None` if it is longer than
    /// [`MAX_FILTER_ETHERTYPES`] — refused, never truncated, because a
    /// silently dropped type is a protocol that mysteriously stops arriving.
    pub fn only(list: &[EtherType]) -> Option<Self> {
        if list.len() > MAX_FILTER_ETHERTYPES {
            return None;
        }
        let mut types = [EtherType(0); MAX_FILTER_ETHERTYPES];
        types[..list.len()].copy_from_slice(list);
        Some(Self::Only {
            types,
            count: list.len() as u8,
        })
    }

    /// The live types, empty for [`Any`](Self::Any).
    pub fn listed(&self) -> &[EtherType] {
        match self {
            Self::Only { types, count } => &types[..(*count as usize).min(MAX_FILTER_ETHERTYPES)],
            Self::Any => &[],
        }
    }

    /// Whether `t` is accepted.
    pub fn accepts(&self, t: EtherType) -> bool {
        match self {
            Self::Any => true,
            Self::Only { .. } => self.listed().contains(&t),
        }
    }

    /// Whether this match is within `grant` — i.e. every type it names is one
    /// the grant allows, and it only claims [`Any`](Self::Any) if the grant
    /// does.
    pub fn within(&self, grant: &EtherTypeMatch) -> bool {
        match (self, grant) {
            (_, EtherTypeMatch::Any) => true,
            (EtherTypeMatch::Any, _) => false,
            (EtherTypeMatch::Only { .. }, g) => self.listed().iter().all(|t| g.accepts(*t)),
        }
    }
}

/// Which multicast groups a vif accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MulticastMatch {
    /// No multicast (broadcast is separate).
    #[default]
    None,
    /// One specific group — what a discovery protocol needs (#387 §3.2's
    /// well-known announcement address), and all it needs.
    Group(MacAddr),
    /// Every multicast group.  A capture-adjacent power, granted separately.
    All,
}

impl MulticastMatch {
    /// Whether `dst` (already known to be multicast, non-broadcast) is
    /// accepted.
    pub fn accepts(&self, dst: MacAddr) -> bool {
        match self {
            Self::None => false,
            Self::Group(g) => *g == dst,
            Self::All => true,
        }
    }

    /// Whether this is within `grant`.
    pub fn within(&self, grant: &MulticastMatch) -> bool {
        match (self, grant) {
            (_, MulticastMatch::All) => true,
            (MulticastMatch::None, _) => true,
            (MulticastMatch::Group(a), MulticastMatch::Group(b)) => a == b,
            _ => false,
        }
    }
}

/// A vif's complete receive specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameFilter {
    /// Accepted protocols.
    pub ethertype: EtherTypeMatch,
    /// Accept frames addressed to this vif's own unicast address.
    pub unicast: bool,
    /// Accepted multicast groups.
    pub multicast: MulticastMatch,
    /// Accept broadcast frames.
    pub broadcast: bool,
    /// Accept frames addressed to **any** station, not just this vif's.
    ///
    /// Separate from the multicast and broadcast bits because it is a
    /// different kind of authority: it exposes traffic addressed to other
    /// vifs on the same device, so it is what a capture tool asks for and
    /// what an ordinary client must never receive by accident.  It is also
    /// only *effective* if the backend's [`PacketFilter`](super::backend::PacketFilter)
    /// puts the device in a promiscuous receive mode.
    pub promiscuous: bool,
    /// What this vif accepts **above** L2, or `None` for no L3 restriction.
    ///
    /// `None` is the contract exactly as it stood before flows existed, and
    /// is what every filter built before this field carried implicitly. A
    /// `Some` is strictly *narrowing*: it can only reject frames the L2 half
    /// already accepted, so granting one never widens a vif's reach.
    ///
    /// It is also the only way two clients can both speak IP under
    /// `AddressMode::Shared`, where vifs separate by protocol alone and an
    /// L2 filter therefore cannot tell their traffic apart.
    pub flow: Option<FlowMatch>,
}

impl FrameFilter {
    /// **The filter that accepts nothing** — what a vif carries once the
    /// authority it was enforcing has been released.
    ///
    /// Revoking an installed filter needs a value to install in its place,
    /// and the only safe one is the empty predicate: a vif whose authority is
    /// gone must receive nothing until a fresh authority says otherwise, not
    /// fall back to whatever it accepted before. Named here rather than
    /// rebuilt at each release site, so "released" means one thing.
    pub const CLOSED: Self = Self {
        ethertype: EtherTypeMatch::Only {
            types: [EtherType(0); MAX_FILTER_ETHERTYPES],
            count: 0,
        },
        unicast: false,
        multicast: MulticastMatch::None,
        broadcast: false,
        promiscuous: false,
        flow: None,
    };

    /// Whether this filter can accept **no frame at all**, whatever arrives.
    ///
    /// True when the L2 half is empty — no EtherType listed, or no address
    /// class admitted — because either alone makes [`accepts`](Self::accepts)
    /// false for every frame, and the flow half can only narrow further.
    ///
    /// Asked by [`VifPolicy::authorize_filter`](super::policy::VifPolicy::authorize_filter):
    /// a grant that names a flow refuses a request binding none, because such
    /// a request would accept *every* datagram reaching its address — a thing
    /// a filter accepting nothing provably cannot do. Without this, releasing
    /// an authority would be refused under exactly the policies that most
    /// need the release to happen, leaving the old filter enforced.
    pub fn admits_nothing(&self) -> bool {
        let no_protocol = matches!(self.ethertype, EtherTypeMatch::Only { count: 0, .. });
        let no_address = !self.unicast
            && !self.broadcast
            && !self.promiscuous
            && matches!(self.multicast, MulticastMatch::None);
        no_protocol || no_address
    }

    /// The filter an ordinary protocol client wants: one EtherType, its own
    /// address, and broadcast (so ARP-shaped discovery works).
    pub fn protocol(t: EtherType) -> Self {
        Self {
            ethertype: EtherTypeMatch::one(t),
            unicast: true,
            multicast: MulticastMatch::None,
            broadcast: true,
            promiscuous: false,
            flow: None,
        }
    }

    /// Whether a frame with head `head` is accepted by a vif whose own
    /// address is `mine` — **the L2 half only**.
    ///
    /// Address and protocol are **both** required: a filter that matched on
    /// protocol alone would deliver another station's traffic to whoever
    /// happened to name the same EtherType.
    ///
    /// A filter carrying a [`flow`](Self::flow) is not fully evaluated by
    /// this method. A classifier must call [`accepts_frame`](Self::accepts_frame),
    /// which is this check *and* the flow half; this one remains public
    /// because it is the whole answer for the `flow: None` filters that
    /// predate flows.
    pub fn accepts(&self, head: &FrameHead, mine: MacAddr) -> bool {
        if !self.ethertype.accepts(head.ethertype) {
            return false;
        }
        if self.promiscuous {
            return true;
        }
        if head.dst.is_broadcast() {
            return self.broadcast;
        }
        if head.dst.is_multicast() {
            return self.multicast.accepts(head.dst);
        }
        self.unicast && head.dst == mine
    }

    /// Whether the frame body satisfies this filter's [`flow`](Self::flow).
    ///
    /// Vacuously true when no flow is bound. When one is, a body that does
    /// not parse as the flow's address family is **rejected** rather than
    /// admitted: a vif that asked for one connection must not receive
    /// something merely because it could not be understood.
    pub fn accepts_flow(&self, ethertype: EtherType, payload: &[u8]) -> bool {
        if self.flow.is_none() {
            return true;
        }
        self.accepts_flow_parsed(FiveTuple::parse(ethertype, payload).as_ref())
    }

    /// The flow half, against a tuple the caller already parsed.
    ///
    /// Same decision as [`accepts_flow`](Self::accepts_flow) — it is written
    /// in terms of this one — for a classifier that is asking several vifs
    /// about one frame. The parse is a property of the frame, not of the
    /// filter, so a caller that asks eight times should pay for it once.
    ///
    /// `None` means the body did not parse, and is **rejected** rather than
    /// admitted, for the same reason as above: a vif that asked for one
    /// connection must not receive something merely because it could not be
    /// understood. A caller must therefore pass the result of a parse, not
    /// `None` for "I did not try".
    pub fn accepts_flow_parsed(&self, tuple: Option<&FiveTuple>) -> bool {
        let Some(flow) = self.flow.as_ref() else {
            return true;
        };
        match tuple {
            Some(t) => flow.accepts(t),
            None => false,
        }
    }

    /// Whether this filter's flow covers a frame **being sent**.
    ///
    /// The transmit counterpart of [`accepts_flow`](Self::accepts_flow), and
    /// separate for the reason [`FlowMatch::accepts_outbound`] gives: the
    /// grant names one conversation, and which end of it is ours depends on
    /// which way the frame is going.
    ///
    /// Vacuously true when no flow is bound, so an L2-only vif transmits
    /// exactly as it did. A body that does not parse is **rejected**, same
    /// as inbound: a vif that asked for one connection must not be able to
    /// send something merely because it could not be understood.
    pub fn accepts_flow_outbound(&self, ethertype: EtherType, payload: &[u8]) -> bool {
        let Some(flow) = self.flow.as_ref() else {
            return true;
        };
        match FiveTuple::parse(ethertype, payload) {
            Some(tuple) => flow.accepts_outbound(&tuple),
            None => false,
        }
    }

    /// The complete receive decision: the L2 half and the flow half.
    ///
    /// `payload` is the frame **after** its 14-byte Ethernet header. This is
    /// what a classifier calls.
    pub fn accepts_frame(&self, head: &FrameHead, payload: &[u8], mine: MacAddr) -> bool {
        self.accepts(head, mine) && self.accepts_flow(head.ethertype, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINE: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const OTHER: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
    const GROUP: MacAddr = MacAddr([0x01, 0x80, 0xc2, 0, 0, 0x03]);

    fn head(dst: MacAddr, t: EtherType) -> FrameHead {
        FrameHead {
            dst,
            src: OTHER,
            ethertype: t,
        }
    }

    /// The property that makes a vif an endpoint rather than a tap: matching
    /// requires **both** the protocol and the address.  A filter that keyed on
    /// protocol alone would hand one client another station's traffic.
    #[test]
    fn matching_requires_both_protocol_and_address() {
        let f = FrameFilter::protocol(EtherType::KORE_PEER);

        assert!(f.accepts(&head(MINE, EtherType::KORE_PEER), MINE));
        assert!(
            !f.accepts(&head(MINE, EtherType::IPV4), MINE),
            "right address, wrong protocol"
        );
        assert!(
            !f.accepts(&head(OTHER, EtherType::KORE_PEER), MINE),
            "right protocol, another station's address"
        );
    }

    /// Broadcast and multicast are separately gated, and a multicast group
    /// filter admits exactly its own group.
    #[test]
    fn group_addressing_is_gated_per_kind() {
        let mut f = FrameFilter::protocol(EtherType::KORE_PEER);
        assert!(f.accepts(&head(MacAddr::BROADCAST, EtherType::KORE_PEER), MINE));

        f.broadcast = false;
        assert!(!f.accepts(&head(MacAddr::BROADCAST, EtherType::KORE_PEER), MINE));

        assert!(
            !f.accepts(&head(GROUP, EtherType::KORE_PEER), MINE),
            "multicast is off by default"
        );
        f.multicast = MulticastMatch::Group(GROUP);
        assert!(f.accepts(&head(GROUP, EtherType::KORE_PEER), MINE));

        let elsewhere = MacAddr([0x01, 0x80, 0xc2, 0, 0, 0x09]);
        assert!(
            !f.accepts(&head(elsewhere, EtherType::KORE_PEER), MINE),
            "a joined group is one group, not all of them"
        );
        f.multicast = MulticastMatch::All;
        assert!(f.accepts(&head(elsewhere, EtherType::KORE_PEER), MINE));
    }

    /// Promiscuity overrides addressing but **not** protocol: a capture vif
    /// still only sees the EtherTypes its filter names.
    #[test]
    fn promiscuous_widens_addressing_not_protocol() {
        let mut f = FrameFilter::protocol(EtherType::KORE_PEER);
        f.promiscuous = true;

        assert!(f.accepts(&head(OTHER, EtherType::KORE_PEER), MINE));
        assert!(
            !f.accepts(&head(OTHER, EtherType::IPV4), MINE),
            "promiscuity is not a protocol grant"
        );
    }

    /// `within` is what policy authorization is built on, so its asymmetry is
    /// load-bearing: `Any` is inside `Any` but nothing else is.
    #[test]
    fn ethertype_containment_is_asymmetric() {
        let peer = EtherTypeMatch::one(EtherType::KORE_PEER);
        let both = EtherTypeMatch::only(&[EtherType::KORE_PEER, EtherType::ARP]).unwrap();

        assert!(peer.within(&both));
        assert!(
            !both.within(&peer),
            "a wider request is not within a narrower grant"
        );
        assert!(peer.within(&EtherTypeMatch::Any));
        assert!(
            !EtherTypeMatch::Any.within(&peer),
            "Any is never within a listed grant — this is the forged-Any check"
        );
        assert!(EtherTypeMatch::Any.within(&EtherTypeMatch::Any));
    }

    /// The trap this split creates, pinned: a flow-carrying filter admits a
    /// frame at L2 that the complete decision rejects.  A classifier calling
    /// only `accepts` would deliver another connection's segments.
    #[test]
    fn the_l2_half_is_not_the_whole_decision() {
        use crate::net::flow::{IpProto, Ipv4Addr};

        let local = Ipv4Addr([10, 0, 2, 15]);
        let peer = Ipv4Addr([10, 0, 2, 2]);
        let stranger = Ipv4Addr([10, 0, 2, 99]);

        let mut f = FrameFilter::protocol(EtherType::IPV4);
        f.flow = Some(crate::net::flow::FlowMatch::connection(
            IpProto::UDP,
            (local, 68),
            (peer, 67),
        ));

        // An IPv4 datagram from a stranger, addressed to this station.
        let mut body = [0u8; 28];
        body[0] = 0x45;
        body[9] = IpProto::UDP.0;
        body[12..16].copy_from_slice(&stranger.0);
        body[16..20].copy_from_slice(&local.0);
        body[20..22].copy_from_slice(&67u16.to_be_bytes());
        body[22..24].copy_from_slice(&68u16.to_be_bytes());

        let h = head(MINE, EtherType::IPV4);
        assert!(f.accepts(&h, MINE), "the L2 half admits it");
        assert!(
            !f.accepts_frame(&h, &body, MINE),
            "the complete decision rejects another peer's datagram"
        );
    }

    /// A filter with no flow is unchanged by the new half — the compatibility
    /// property the whole additive design rests on.
    #[test]
    fn no_flow_means_the_old_contract_exactly() {
        let f = FrameFilter::protocol(EtherType::IPV4);
        assert_eq!(f.flow, None);

        let h = head(MINE, EtherType::IPV4);
        assert!(f.accepts_flow(EtherType::IPV4, &[]), "nothing to satisfy");
        assert_eq!(
            f.accepts(&h, MINE),
            f.accepts_frame(&h, &[], MINE),
            "the two decisions agree when no flow is bound"
        );
    }

    /// A list longer than the bound is refused, not truncated: a silently
    /// dropped EtherType is a protocol that mysteriously stops arriving.
    #[test]
    fn an_overlong_ethertype_list_refuses() {
        let five = [
            EtherType(1),
            EtherType(2),
            EtherType(3),
            EtherType(4),
            EtherType(5),
        ];
        assert!(EtherTypeMatch::only(&five).is_none());
        assert!(EtherTypeMatch::only(&five[..MAX_FILTER_ETHERTYPES]).is_some());
    }

    // ── the empty predicate ─────────────────────────────────────────────

    /// **[`FrameFilter::CLOSED`] accepts nothing**, whatever arrives and
    /// whatever address the vif answers to.
    ///
    /// It is what a vif carries once the authority it was enforcing has been
    /// released, so a hole in it would be traffic still being delivered for a
    /// capability nobody holds.
    #[test]
    fn the_closed_filter_accepts_no_frame() {
        let f = FrameFilter::CLOSED;
        for dst in [MINE, OTHER, GROUP, MacAddr::BROADCAST] {
            for t in [EtherType::IPV4, EtherType::ARP, EtherType::KORE_PEER] {
                assert!(!f.accepts(&head(dst, t), MINE), "{dst:?} / {t:?}");
            }
        }
        assert!(f.admits_nothing());
    }

    /// `admits_nothing` is true exactly when the L2 half can match no frame —
    /// no protocol listed, or no address class admitted — and false as soon
    /// as either half can.
    ///
    /// It gates a policy carve-out, so a `true` where a frame could still
    /// arrive would let a filter past a check it should have met.
    #[test]
    fn admits_nothing_is_true_only_for_an_empty_l2_half() {
        assert!(FrameFilter::CLOSED.admits_nothing());

        let mut no_protocol = FrameFilter::protocol(EtherType::IPV4);
        no_protocol.ethertype = EtherTypeMatch::only(&[]).expect("empty fits");
        assert!(no_protocol.admits_nothing(), "nothing to match on");

        let mut no_address = FrameFilter::protocol(EtherType::IPV4);
        no_address.unicast = false;
        no_address.broadcast = false;
        assert!(no_address.admits_nothing(), "nobody to match for");

        assert!(
            !FrameFilter::protocol(EtherType::IPV4).admits_nothing(),
            "an ordinary protocol filter admits its own station's frames"
        );
        let mut only_broadcast = FrameFilter::CLOSED;
        only_broadcast.ethertype = EtherTypeMatch::one(EtherType::ARP);
        only_broadcast.broadcast = true;
        assert!(
            !only_broadcast.admits_nothing(),
            "one address class is enough to admit something"
        );
        assert!(only_broadcast.accepts(&head(MacAddr::BROADCAST, EtherType::ARP), MINE));
    }
}
