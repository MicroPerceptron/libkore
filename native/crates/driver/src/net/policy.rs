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
//!
//! Bundling them into one "privileged" bit would mean a client that needs to
//! answer broadcast also gets to impersonate its neighbour.

use super::addr::MacAddr;
use super::filter::{EtherTypeMatch, FrameFilter, MulticastMatch};

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
}
