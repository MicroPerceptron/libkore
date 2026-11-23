//! **The strata** — how much traffic one authority covers, and the two powers
//! that deliberately are not rungs.
//!
//! # The ladder
//!
//! Four rungs, each strictly containing the one below, so authority can be
//! compared and attenuated downward:
//!
//! | rung | covers | who holds it |
//! |---|---|---|
//! | S0 [`Mediated`](Scope::Mediated) | nothing | everything, by default |
//! | S1 [`Flow`](Scope::Flow) | one 4-tuple | a client opening a connection |
//! | S2 [`Range`](Scope::Range) | a port range at one address | a server |
//! | S3 [`Interface`](Scope::Interface) | raw L2 at one station address | a guest |
//!
//! S2 is **modelled and not yet mintable** (`res/INET_STACK.md` §4.4): it has
//! no consumer, and the rule of two says the abstraction lands with the
//! second one. Having it in the ladder anyway is what makes `within`
//! total — a containment check that silently lacked a rung would answer
//! "not within" for a case it simply could not express.
//!
//! # Why a dedicated queue is not a fifth rung
//!
//! A hardware queue pair — the NIC steering matching traffic into a ring
//! the holder owns — was drafted as S4 and is **not on this ladder**, for
//! the reason that decides what a rung is: a queue names no set of
//! traffic. S3 covers every protocol at one station, S1 one four-tuple;
//! "the frames the steering table happened to put here" is not a set an
//! authority can be compared against.
//!
//! So it cannot contain S3, and on address steering it would sit *below*
//! it while remaining incomparable with S1 — which would make
//! [`Scope::within`] non-total and stop `derive` meaning "narrows". The
//! one property this ladder exists to have is the one a queue variant
//! would take away.
//!
//! It is a second axis, exactly like [`Powers`] below and off the ladder
//! for the same kind of reason: a `delivery` field on the capability, when
//! a consumer appears. None has. And on the part in tree the mechanism is
//! not there either — `drivers/net/e1000e/tools/i82574.h` transcribes two
//! rings with per-queue MSI-X but no n-tuple filter, no RSS, and nothing
//! binding a receive-address entry to a queue. The security argument is
//! the harder half regardless: the driver holds one IOMMU domain with
//! client windows shared into it, so queues are not isolation and the
//! steering table would become the boundary. Not buildable, not needed,
//! and not this shape.
//!
//! # S3 names its station, and that is the whole of its boundary
//!
//! "Raw L2 on a vif" is **not** raw L2 on the device, and the difference is
//! one address. The receive decision one layer down
//! ([`FrameFilter::accepts`](kore_driver::net::FrameFilter::accepts)) is
//! evaluated against the vif's own station address, so an S3 holder sees the
//! frames addressed to *that* station, plus broadcast — every protocol
//! crossing it, and no other station's unicast.
//!
//! The address is therefore carried in the grant rather than left to the
//! context the filter happens to be evaluated in. A rung that cannot state
//! its own boundary cannot be audited against one, and S1 and S2 both name
//! their local end already — S1 a four-tuple, S2 an address and a port span.
//! S3 names the local end at *its* layer, which is a MAC.
//!
//! **The precondition this rung rests on, stated once.** The isolation above
//! is real only where two vifs have two station addresses.
//! `AddressMode::Shared` — the frame row's shipping default — gives every vif
//! the device's own address, and under it "addressed to this station" and
//! "addressed to every vif" are the same sentence: an S3 holder would receive
//! every other client's traffic without ever asking for
//! [`promiscuous`](Powers::promiscuous), because under that mode the traffic
//! genuinely *is* addressed to it. Minting an S3 is therefore gated on a
//! grant whose composer has established a per-vif station address
//! (`res/INET_STACK.md` §4.4), and the row refuses a second S3 at a station
//! one of its own already holds.
//!
//! # S0 is a grant, not an absence
//!
//! [`Mediated`](Scope::Mediated) is the *narrowest* rung: it permits nothing,
//! so the row stays on the datapath because the authority covers nothing —
//! not because a branch was taken. One engine, strata as data.
//!
//! Note the deliberate asymmetry with
//! [`FrameFilter::flow`](kore_driver::net::FrameFilter::flow), where `None`
//! means *no restriction* and is therefore the **widest** value. A filter says
//! what a vif will accept; a scope says what an authority may ask for. They
//! point in opposite directions, and naming this rung `Mediated` rather than
//! `None` is what keeps the two from reading alike at a glance.

use kore_driver::net::MacAddr;
use kore_driver::net::flow::{EndpointMatch, FlowMatch, Ipv4Addr};

/// An inclusive range of local ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortRange {
    first: u16,
    last: u16,
}

impl PortRange {
    /// A range, or `None` when it is inverted.
    ///
    /// Refused rather than normalised: a caller that passed its bounds the
    /// wrong way round asked for something it did not mean, and silently
    /// swapping them would grant a range nobody checked.
    pub fn new(first: u16, last: u16) -> Option<Self> {
        (first <= last).then_some(Self { first, last })
    }

    /// Exactly one port.
    pub fn one(port: u16) -> Self {
        Self {
            first: port,
            last: port,
        }
    }

    /// The lowest port in the range.
    pub fn first(&self) -> u16 {
        self.first
    }

    /// The highest port in the range.
    pub fn last(&self) -> u16 {
        self.last
    }

    /// Whether `port` is in the range.
    pub fn contains(&self, port: u16) -> bool {
        self.first <= port && port <= self.last
    }

    /// Whether every port of this range is in `grant`.
    pub fn within(&self, grant: &PortRange) -> bool {
        grant.first <= self.first && self.last <= grant.last
    }
}

/// The two powers that cut across every rung.
///
/// Kept off the ladder on purpose. A flow-scoped NAT legitimately needs
/// [`spoof`](Self::spoof) while an interface holder may be denied it — a
/// sentence a single ladder cannot form. Both are refused by default, and a
/// derivation may drop them but never add one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Powers {
    /// May send from an address that was not assigned.
    pub spoof: bool,
    /// May receive traffic that was not addressed to the holder.
    pub promiscuous: bool,
}

impl Powers {
    /// Neither power — what every authority starts with.
    pub const NONE: Self = Self {
        spoof: false,
        promiscuous: false,
    };

    /// Whether every power held here is also held by `grant`.
    pub fn within(&self, grant: &Powers) -> bool {
        (!self.spoof || grant.spoof) && (!self.promiscuous || grant.promiscuous)
    }
}

/// How much traffic one authority covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// **S0** — nothing. The row stays on the datapath.
    Mediated,
    /// **S1** — one flow, pinned at both ends.
    Flow(FlowMatch),
    /// **S2** — a port range at one local address. *Modelled, not mintable.*
    Range {
        /// The local address.
        addr: Ipv4Addr,
        /// The local ports.
        ports: PortRange,
    },
    /// **S3** — raw L2 at one station address.
    ///
    /// Every protocol crossing `station`, with no L3 predicate at all — so
    /// it is the only rung that carries traffic which has no five-tuple to
    /// match on. Naming the station is what keeps it a grant over *a vif*
    /// rather than over the device; see the module header for the
    /// addressing precondition that makes the distinction real.
    Interface {
        /// The link-layer address this authority covers, and the only one it
        /// may speak as without [`Powers::spoof`].
        station: MacAddr,
    },
}

impl Scope {
    /// The rung, `0..=3`, for comparing breadth at a glance.
    pub fn rung(&self) -> u8 {
        match self {
            Self::Mediated => 0,
            Self::Flow(_) => 1,
            Self::Range { .. } => 2,
            Self::Interface { .. } => 3,
        }
    }

    /// Whether this scope is mintable today.
    ///
    /// S2 is in the ladder so containment is total, and refused at the mint
    /// so nothing depends on an untested rung. When a consumer arrives, this
    /// is the one place that changes.
    ///
    /// S3 is mintable, and being mintable here is not the same as being
    /// reachable: the row gates it behind an explicit grant
    /// ([`OpenGrant::raw_l2`](crate::row::OpenGrant::raw_l2)) that nothing
    /// in this tree sets. The ladder's job is to say the transition is *in
    /// the model*; who may make it is the row's.
    pub fn is_mintable(&self) -> bool {
        matches!(self, Self::Mediated | Self::Flow(_) | Self::Interface { .. })
    }

    /// The station address this scope covers at L2, for the one rung that
    /// names one.
    ///
    /// `None` below S3 is not "any station" — it is "this rung says nothing
    /// about L2", which leaves the vif's assigned address the only answer.
    /// The two readings differ, and
    /// [`FlowCap::may_source`](crate::FlowCap::may_source) is where the
    /// difference is spent.
    pub fn station(&self) -> Option<MacAddr> {
        match self {
            Self::Interface { station } => Some(*station),
            Self::Mediated | Self::Flow(_) | Self::Range { .. } => None,
        }
    }

    /// Whether this scope covers no traffic at all.
    pub fn is_mediated(&self) -> bool {
        matches!(self, Self::Mediated)
    }

    /// Whether every flow this scope covers is also covered by `grant`.
    ///
    /// The ladder's containment, and the whole of what makes attenuation
    /// safe: a derivation is admitted only when its scope is within its
    /// parent's.
    pub fn within(&self, grant: &Scope) -> bool {
        match (self, grant) {
            // Nothing is within anything.
            (Self::Mediated, _) => true,
            // Only nothing is within nothing.
            (_, Self::Mediated) => false,
            // A derivation may narrow S3 but never re-point it: a child
            // naming another station would be claiming an address that was
            // never its parent's, which is the escape this rung exists to
            // not have.
            (Self::Interface { station: a }, Self::Interface { station: g }) => a == g,
            // The whole interface contains every rung below it. No station
            // comparison is possible here — the rungs below name an IPv4
            // address, not a MAC — and none is needed: containment is only
            // ever asked between a child and its parent, `derive` pins the
            // child to the parent's vif, and one vif answers to one station.
            (_, Self::Interface { .. }) => true,
            (Self::Interface { .. }, _) => false,

            (Self::Flow(a), Self::Flow(b)) => a.within(b),
            (Self::Range { addr: a, ports: p }, Self::Range { addr: g, ports: q }) => {
                a == g && p.within(q)
            }
            // A flow is within a range when its local end is inside it. The
            // peer end is unconstrained by a range, which names only the
            // local side.
            (Self::Flow(f), Self::Range { addr, ports }) => match f.local {
                EndpointMatch::AddrPort { addr: a, port } => a == *addr && ports.contains(port),
                // An address-wide or any-address flow is wider than a port
                // range, whichever ports the range names.
                EndpointMatch::Addr(_) | EndpointMatch::Any => false,
            },
            // A range is wider than one flow.
            (Self::Range { .. }, Self::Flow(_)) => false,
        }
    }
}
