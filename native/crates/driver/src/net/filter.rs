//! **Frame filters** — what a vif receives, and the matching that decides it.
//!
//! A filter is the vif's whole receive specification.  It replaces the socket:
//! a client does not bind a port, it binds a predicate over L2 frames, and the
//! classifier delivers a frame to exactly the vifs whose predicate accepts it.

use super::addr::{EtherType, MacAddr};
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
}

impl FrameFilter {
    /// The filter an ordinary protocol client wants: one EtherType, its own
    /// address, and broadcast (so ARP-shaped discovery works).
    pub fn protocol(t: EtherType) -> Self {
        Self {
            ethertype: EtherTypeMatch::one(t),
            unicast: true,
            multicast: MulticastMatch::None,
            broadcast: true,
            promiscuous: false,
        }
    }

    /// Whether a frame with head `head` is accepted by a vif whose own
    /// address is `mine`.
    ///
    /// Address and protocol are **both** required: a filter that matched on
    /// protocol alone would deliver another station's traffic to whoever
    /// happened to name the same EtherType.
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
}
