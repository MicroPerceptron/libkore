//! **The flow vocabulary** — the L3/L4 scalars, the five-tuple a received
//! frame carries, and what a vif accepts above the Ethernet header.
//!
//! The sibling of [`filter`](super::filter), which stays L2: that module
//! decides whether a frame is addressed to this vif at all, and this one
//! decides whether it belongs to the *flow* the vif was granted.
//!
//! # Why this exists one layer down
//!
//! A vif's receive specification used to be purely L2, and
//! [`AddressMode::Shared`] — the shipping default — gives every vif the
//! device's own address, so vifs "separate by protocol alone". Two clients
//! both wanting IPv4 therefore both match every inbound IPv4 frame, and the
//! classifier copies it into both. That is sound while the only consumers are
//! distinct EtherTypes; it stops being sound the moment a second IP consumer
//! exists.
//!
//! Narrowing the predicate is the fix. A [`FlowMatch`] is strictly *less*
//! than the L2 filter it accompanies — it can only reject frames the L2 half
//! already accepted — so adding it to a filter never widens a vif's reach.
//!
//! [`AddressMode::Shared`]: https://docs.rs/kore-svc-net

use super::addr::EtherType;

/// An IPv4 address, network order.
///
/// IPv4 only, deliberately: no consumer carries IPv6 yet, and the wire
/// encoding reserves a family byte so the sibling is additive rather than a
/// break (`res/INET_STACK.md` §9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ipv4Addr(pub [u8; 4]);

impl Ipv4Addr {
    /// `0.0.0.0` — the unspecified address.
    pub const UNSPECIFIED: Self = Self([0; 4]);
    /// `255.255.255.255` — the limited broadcast address.
    pub const BROADCAST: Self = Self([0xff; 4]);

    /// Whether this is a multicast address (`224.0.0.0/4`).
    pub const fn is_multicast(self) -> bool {
        self.0[0] & 0xf0 == 0xe0
    }

    /// Whether this is the limited broadcast address.
    pub const fn is_broadcast(self) -> bool {
        matches!(self.0, [0xff, 0xff, 0xff, 0xff])
    }

    /// Whether this is the unspecified address.
    pub const fn is_unspecified(self) -> bool {
        matches!(self.0, [0, 0, 0, 0])
    }
}

/// An IP protocol number — the IPv4 header's `protocol` field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IpProto(pub u8);

impl IpProto {
    /// ICMP for IPv4.
    pub const ICMP: Self = Self(1);
    /// TCP.
    pub const TCP: Self = Self(6);
    /// UDP.
    pub const UDP: Self = Self(17);

    /// Whether this protocol carries a port pair in its first four bytes.
    ///
    /// The two the stack binds flows for. A protocol that is not one of these
    /// has no ports to match, which is why [`EndpointMatch::AddrPort`] cannot
    /// accept it.
    pub const fn has_ports(self) -> bool {
        self.0 == Self::TCP.0 || self.0 == Self::UDP.0
    }
}

/// The smallest IPv4 header, with no options.
const IPV4_MIN_HEADER: usize = 20;

/// What a received frame carries above L2, as far as matching needs it.
///
/// `ports` is `None` for a protocol that has none, and — load-bearing — for a
/// **non-initial fragment**, which carries no L4 header at all. A parser that
/// read bytes at the header offset regardless would match a fragment's
/// payload against a port and admit it into the wrong vif.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiveTuple {
    /// The IP protocol.
    pub proto: IpProto,
    /// Source address — the peer, on receive.
    pub src: Ipv4Addr,
    /// Destination address — the local end, on receive.
    pub dst: Ipv4Addr,
    /// `(source port, destination port)`, when this datagram has them.
    pub ports: Option<(u16, u16)>,
}

impl FiveTuple {
    /// Lift a five-tuple out of a frame body, or `None` when there is not a
    /// well-formed IPv4 datagram there.
    ///
    /// `payload` is the frame **after** the 14-byte Ethernet header.
    /// Every read is bounds-checked against the declared header length rather
    /// than assumed, because this runs on bytes a peer chose.
    pub fn parse(ethertype: EtherType, payload: &[u8]) -> Option<Self> {
        if ethertype != EtherType::IPV4 || payload.len() < IPV4_MIN_HEADER {
            return None;
        }
        // Version must be 4; IHL counts 32-bit words and may not claim fewer
        // than the minimum header or more than the frame holds.
        if payload[0] >> 4 != 4 {
            return None;
        }
        let header_len = ((payload[0] & 0x0f) as usize) * 4;
        if header_len < IPV4_MIN_HEADER || payload.len() < header_len {
            return None;
        }

        let proto = IpProto(payload[9]);
        let src = Ipv4Addr([payload[12], payload[13], payload[14], payload[15]]);
        let dst = Ipv4Addr([payload[16], payload[17], payload[18], payload[19]]);

        // The low 13 bits of the flags/offset word. Non-zero means this is a
        // continuation fragment, whose first bytes are payload and not a
        // transport header.
        let fragment_offset = u16::from_be_bytes([payload[6], payload[7]]) & 0x1fff;

        let ports = if fragment_offset == 0 && proto.has_ports() {
            let l4 = &payload[header_len..];
            // A truncated transport header yields no ports rather than a
            // guess: `AddrPort` then fails to match, which is the direction
            // that fails closed.
            (l4.len() >= 4).then(|| {
                (
                    u16::from_be_bytes([l4[0], l4[1]]),
                    u16::from_be_bytes([l4[2], l4[3]]),
                )
            })
        } else {
            None
        };

        Some(Self {
            proto,
            src,
            dst,
            ports,
        })
    }
}

/// Which IP protocols a flow accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProtoMatch {
    /// Exactly this protocol.
    Only(IpProto),
    /// Every protocol carried over IP.
    ///
    /// A widening power in the same family as [`EtherTypeMatch::Any`]: it
    /// exposes protocols other clients own at the same address.
    ///
    /// [`EtherTypeMatch::Any`]: super::filter::EtherTypeMatch::Any
    #[default]
    Any,
}

impl ProtoMatch {
    /// Whether `p` is accepted.
    pub fn accepts(&self, p: IpProto) -> bool {
        match self {
            Self::Any => true,
            Self::Only(q) => *q == p,
        }
    }

    /// Whether this match is within `grant`.
    pub fn within(&self, grant: &ProtoMatch) -> bool {
        match (self, grant) {
            (_, Self::Any) => true,
            (Self::Any, _) => false,
            (Self::Only(a), Self::Only(b)) => a == b,
        }
    }
}

/// Which end of a flow an address and port name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EndpointMatch {
    /// Exactly this address and port — one end of one connection.
    AddrPort {
        /// The address.
        addr: Ipv4Addr,
        /// The port.
        port: u16,
    },
    /// This address, any port.
    Addr(Ipv4Addr),
    /// Any address, any port.
    ///
    /// On the **local** end this is a capture-adjacent power — it admits
    /// traffic addressed to other vifs' addresses. On the **peer** end it is
    /// ordinary: a listening endpoint accepts from whoever connects.
    #[default]
    Any,
}

impl EndpointMatch {
    /// Whether `(addr, port)` is accepted. `port` is `None` when the datagram
    /// carried none, which only [`Self::AddrPort`] rejects.
    pub fn accepts(&self, addr: Ipv4Addr, port: Option<u16>) -> bool {
        match self {
            Self::Any => true,
            Self::Addr(a) => *a == addr,
            Self::AddrPort { addr: a, port: p } => *a == addr && port == Some(*p),
        }
    }

    /// Whether this match is within `grant`.
    ///
    /// [`AddrPort`](Self::AddrPort) is *narrower* than [`Addr`](Self::Addr),
    /// so it is within it; the reverse is a widening and is refused.
    pub fn within(&self, grant: &EndpointMatch) -> bool {
        match (self, grant) {
            (_, Self::Any) => true,
            (Self::Any, _) => false,
            (Self::Addr(a), Self::Addr(g)) => a == g,
            (Self::AddrPort { addr, .. }, Self::Addr(g)) => addr == g,
            (Self::Addr(_), Self::AddrPort { .. }) => false,
            (
                Self::AddrPort { addr: a, port: p },
                Self::AddrPort {
                    addr: g,
                    port: q,
                },
            ) => a == g && p == q,
        }
    }
}

/// What a vif accepts above L2.
///
/// Every field must accept for the flow to accept. Absent from a filter
/// ([`FrameFilter::flow`](super::filter::FrameFilter::flow) is `None`) this
/// imposes nothing, which is the L2-only contract as it stood before flows
/// existed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlowMatch {
    /// The accepted IP protocol.
    pub proto: ProtoMatch,
    /// The local end — the datagram's destination, on receive.
    pub local: EndpointMatch,
    /// The remote end — the datagram's source, on receive.
    pub peer: EndpointMatch,
}

impl FlowMatch {
    /// The flow one connection needs: a protocol, both ends pinned.
    pub fn connection(
        proto: IpProto,
        local: (Ipv4Addr, u16),
        peer: (Ipv4Addr, u16),
    ) -> Self {
        Self {
            proto: ProtoMatch::Only(proto),
            local: EndpointMatch::AddrPort {
                addr: local.0,
                port: local.1,
            },
            peer: EndpointMatch::AddrPort {
                addr: peer.0,
                port: peer.1,
            },
        }
    }

    /// The flow a listener needs: a protocol, a local endpoint, any peer.
    pub fn listener(proto: IpProto, local: (Ipv4Addr, u16)) -> Self {
        Self {
            proto: ProtoMatch::Only(proto),
            local: EndpointMatch::AddrPort {
                addr: local.0,
                port: local.1,
            },
            peer: EndpointMatch::Any,
        }
    }

    /// Whether a received datagram belongs to this flow.
    pub fn accepts(&self, t: &FiveTuple) -> bool {
        let (src_port, dst_port) = match t.ports {
            Some((s, d)) => (Some(s), Some(d)),
            None => (None, None),
        };
        self.proto.accepts(t.proto)
            && self.local.accepts(t.dst, dst_port)
            && self.peer.accepts(t.src, src_port)
    }

    /// Whether this flow covers a datagram **being sent**.
    ///
    /// The same grant read in the other direction, and it has to be a
    /// separate question rather than the same one: [`accepts`](Self::accepts)
    /// is written for a frame arriving, where our address is the
    /// destination and the peer's is the source. On a frame leaving, those
    /// are the other way round. Asking the inbound question of an outbound
    /// frame does not merely fail to catch a forgery — it refuses every
    /// lawful frame and admits only one addressed *to* us that we are
    /// somehow sending.
    ///
    /// A flow is one conversation, so the pair that may leave is exactly
    /// the pair that may arrive, reversed. Nothing else differs; `peer` as
    /// `Any` still means a listener accepts whoever connects, and on the
    /// way out it means it may answer whoever did.
    pub fn accepts_outbound(&self, t: &FiveTuple) -> bool {
        let (src_port, dst_port) = match t.ports {
            Some((s, d)) => (Some(s), Some(d)),
            None => (None, None),
        };
        self.proto.accepts(t.proto)
            && self.local.accepts(t.src, src_port)
            && self.peer.accepts(t.dst, dst_port)
    }

    /// Whether this flow is within `grant` — every dimension no wider.
    pub fn within(&self, grant: &FlowMatch) -> bool {
        self.proto.within(&grant.proto)
            && self.local.within(&grant.local)
            && self.peer.within(&grant.peer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const PEER: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const ELSEWHERE: Ipv4Addr = Ipv4Addr([10, 0, 2, 99]);

    /// Build an IPv4 datagram with an optional transport header.
    fn ipv4(proto: IpProto, src: Ipv4Addr, dst: Ipv4Addr, l4: &[u8], frag: u16) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0] = 0x45; // version 4, IHL 5
        let total = (IPV4_MIN_HEADER + l4.len()) as u16;
        b[2..4].copy_from_slice(&total.to_be_bytes());
        b[6..8].copy_from_slice(&frag.to_be_bytes());
        b[8] = 64; // TTL
        b[9] = proto.0;
        b[12..16].copy_from_slice(&src.0);
        b[16..20].copy_from_slice(&dst.0);
        b[IPV4_MIN_HEADER..IPV4_MIN_HEADER + l4.len()].copy_from_slice(l4);
        b
    }

    fn ports(src: u16, dst: u16) -> [u8; 4] {
        let mut b = [0u8; 4];
        b[0..2].copy_from_slice(&src.to_be_bytes());
        b[2..4].copy_from_slice(&dst.to_be_bytes());
        b
    }

    #[test]
    fn a_well_formed_datagram_yields_its_tuple() {
        let d = ipv4(IpProto::TCP, PEER, LOCAL, &ports(44000, 443), 0);
        let t = FiveTuple::parse(EtherType::IPV4, &d).expect("parses");
        assert_eq!(t.proto, IpProto::TCP);
        assert_eq!(t.src, PEER);
        assert_eq!(t.dst, LOCAL);
        assert_eq!(t.ports, Some((44000, 443)));
    }

    /// A non-IPv4 EtherType has no tuple — the flow half must not run on
    /// bytes it cannot interpret.
    #[test]
    fn a_non_ip_frame_has_no_tuple() {
        let d = ipv4(IpProto::TCP, PEER, LOCAL, &ports(1, 2), 0);
        assert!(FiveTuple::parse(EtherType::ARP, &d).is_none());
        assert!(FiveTuple::parse(EtherType::KORE_PEER, &d).is_none());
    }

    /// The headline parser hazard: a continuation fragment carries payload
    /// where a transport header would be. Reading ports from it would admit
    /// an attacker-chosen body into a port-pinned vif.
    #[test]
    fn a_continuation_fragment_yields_no_ports() {
        let d = ipv4(IpProto::TCP, PEER, LOCAL, &ports(44000, 443), 1);
        let t = FiveTuple::parse(EtherType::IPV4, &d).expect("parses");
        assert_eq!(t.ports, None, "offset 1 is not the first fragment");

        let flow = FlowMatch::connection(IpProto::TCP, (LOCAL, 443), (PEER, 44000));
        assert!(
            !flow.accepts(&t),
            "a port-pinned flow must reject a datagram with no ports"
        );
    }

    /// Malformed length fields are refused, never trusted.
    #[test]
    fn malformed_headers_are_refused() {
        let good = ipv4(IpProto::UDP, PEER, LOCAL, &ports(1, 2), 0);

        assert!(
            FiveTuple::parse(EtherType::IPV4, &good[..19]).is_none(),
            "shorter than the minimum header"
        );

        let mut wrong_version = good;
        wrong_version[0] = 0x65;
        assert!(FiveTuple::parse(EtherType::IPV4, &wrong_version).is_none());

        let mut short_ihl = good;
        short_ihl[0] = 0x44; // IHL 4 = 16 bytes, below the minimum
        assert!(FiveTuple::parse(EtherType::IPV4, &short_ihl).is_none());

        let mut long_ihl = good;
        long_ihl[0] = 0x4f; // IHL 15 = 60 bytes
        assert!(
            FiveTuple::parse(EtherType::IPV4, &long_ihl[..40]).is_none(),
            "IHL may not claim more than the frame holds"
        );
    }

    /// A truncated transport header yields no ports rather than reading past
    /// what arrived.
    #[test]
    fn a_truncated_transport_header_yields_no_ports() {
        let d = ipv4(IpProto::UDP, PEER, LOCAL, &[0u8; 2], 0);
        let t = FiveTuple::parse(EtherType::IPV4, &d[..IPV4_MIN_HEADER + 2]).expect("parses");
        assert_eq!(t.ports, None);
    }

    /// A protocol with no ports still yields its addresses, and a flow pinned
    /// only by address still matches it — which is what ICMP needs.
    #[test]
    fn a_portless_protocol_matches_by_address() {
        let d = ipv4(IpProto::ICMP, PEER, LOCAL, &[8, 0, 0, 0], 0);
        let t = FiveTuple::parse(EtherType::IPV4, &d).expect("parses");
        assert_eq!(t.ports, None, "ICMP carries no ports");

        let flow = FlowMatch {
            proto: ProtoMatch::Only(IpProto::ICMP),
            local: EndpointMatch::Addr(LOCAL),
            peer: EndpointMatch::Any,
        };
        assert!(flow.accepts(&t));
    }

    /// The isolation property this whole module exists for: two connections
    /// to the same local port from different peers do not match each other's
    /// flow.
    #[test]
    fn two_connections_do_not_match_each_other() {
        let mine = FlowMatch::connection(IpProto::TCP, (LOCAL, 443), (PEER, 44000));

        let theirs = ipv4(IpProto::TCP, ELSEWHERE, LOCAL, &ports(44000, 443), 0);
        let t = FiveTuple::parse(EtherType::IPV4, &theirs).expect("parses");
        assert!(
            !mine.accepts(&t),
            "same local endpoint, another peer — not my flow"
        );

        let other_port = ipv4(IpProto::TCP, PEER, LOCAL, &ports(44001, 443), 0);
        let t = FiveTuple::parse(EtherType::IPV4, &other_port).expect("parses");
        assert!(!mine.accepts(&t), "same peer address, another peer port");
    }

    /// Containment is asymmetric in the same way the L2 matches are, and
    /// `AddrPort` is inside `Addr` rather than the other way round.
    #[test]
    fn containment_is_asymmetric() {
        let pinned = EndpointMatch::AddrPort {
            addr: LOCAL,
            port: 443,
        };
        let by_addr = EndpointMatch::Addr(LOCAL);

        assert!(pinned.within(&by_addr), "a port pin is inside an address");
        assert!(!by_addr.within(&pinned), "an address is wider than a port pin");
        assert!(pinned.within(&EndpointMatch::Any));
        assert!(
            !EndpointMatch::Any.within(&pinned),
            "Any is never within a named grant — the forged-Any check"
        );
        assert!(!by_addr.within(&EndpointMatch::Addr(PEER)));

        assert!(ProtoMatch::Only(IpProto::TCP).within(&ProtoMatch::Any));
        assert!(
            !ProtoMatch::Any.within(&ProtoMatch::Only(IpProto::TCP)),
            "any-protocol is not within one named protocol"
        );
    }
}
