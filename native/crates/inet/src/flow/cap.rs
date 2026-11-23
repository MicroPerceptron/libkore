//! **[`FlowCap`]** — the minted authority to move traffic without asking
//! again.
//!
//! # It is not a bypass
//!
//! "Bypass" would mean skipping a check, and a skipped check is a back door.
//! Nothing here skips anything. The row's job is to **mint**, and minting is
//! where the checking happens: is this 4-tuple free, may this task bind it,
//! does the policy permit this peer. Once minted, exercising that authority
//! needs no further consultation — because the consultation already happened,
//! and its answer *is* the capability.
//!
//! So the fast lane is not unmediated. It is mediated once, earlier, over a
//! longer span. There is no setting at which the security argument gets
//! weaker; there is a setting at which it gets cheaper.
//!
//! # Affine, by construction
//!
//! [`FlowCap`] is neither `Copy` nor `Clone`. A copyable authority is a
//! backdoor in a capability system: one holder could fan a grant out to many
//! and no revocation would find them. Here a cap is moved to transfer it,
//! borrowed to derive from it, and consumed to revoke it — the three-tier
//! discipline, which also happens to be exactly the three durations a holder
//! needs (borrow for one request, hold for a flow, store for a process).

use kore_driver::net::flow::{EndpointMatch, FiveTuple, FlowMatch, ProtoMatch};
use kore_driver::net::{EtherType, EtherTypeMatch, FrameFilter, MacAddr, MulticastMatch, VifId};

use super::scope::{Powers, Scope};

/// Why an authority could not be minted or derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowRefusal {
    /// The requested scope is wider than the authority it was derived from.
    ///
    /// Names the direction explicitly: a derivation may only narrow, so this
    /// is an attempted escalation rather than a mismatch.
    WiderScope,
    /// The derivation asked for a power its parent does not hold.
    WiderPowers,
    /// The rung is modelled but not yet mintable — S2 awaits a consumer
    /// (`res/INET_STACK.md` §4.4).
    ///
    /// Reachable from a derivation as well as a mint, and that is the point:
    /// S3 may be narrowed to S1 while S2 between them stays modelled, so a
    /// child landing on the unbuilt rung is refused by name rather than
    /// silently admitted because its parent was wider.
    RungNotMintable,
    // No `Stale` variant: revocation is not a refusal this type produces.
    // It surfaces as `FlowCap::resolves_against` answering `false`, and a
    // variant nothing constructs is dead API that a public enum will never
    // warn about.
}

/// The authority to move one scope's worth of traffic on one vif.
///
/// Deliberately **not** `Copy` and **not** `Clone`, and the two doctests
/// below are what hold that rather than the absence of a derive — which a
/// later edit could restore without anyone noticing.
///
/// Moving it is lawful, and the mover keeps nothing:
///
/// ```
/// use kore_inet::{FlowCap, Scope};
/// use kore_driver::net::VifId;
///
/// let cap = FlowCap::mediated(VifId { index: 0, generation: 1 });
/// let moved = cap;
/// assert!(moved.scope().is_mediated());
/// ```
///
/// Using it after the move is not, so one holder cannot fan a grant out to
/// many and leave revocation with nothing to find:
///
/// ```compile_fail
/// use kore_inet::FlowCap;
/// use kore_driver::net::VifId;
///
/// let cap = FlowCap::mediated(VifId { index: 0, generation: 1 });
/// let first = cap;
/// let second = cap; // `cap` has already moved — a capability is affine
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct FlowCap {
    vif: VifId,
    scope: Scope,
    powers: Powers,
}

impl FlowCap {
    /// Mint the S0 authority: a grant that covers nothing.
    ///
    /// Not a degenerate case to be special-cased away — it is the ordinary
    /// result of a mint that granted no scope, and it keeps the row on the
    /// datapath without a second code path existing anywhere.
    pub fn mediated(vif: VifId) -> Self {
        Self {
            vif,
            scope: Scope::Mediated,
            powers: Powers::NONE,
        }
    }

    /// Mint an authority at `scope`.
    ///
    /// The row calls this **after** it has done the checking the scope
    /// asserts. A rung with no consumer is refused here rather than left to
    /// an untested path (`FlowRefusal::RungNotMintable`).
    pub fn mint(vif: VifId, scope: Scope, powers: Powers) -> Result<Self, FlowRefusal> {
        if !scope.is_mintable() {
            return Err(FlowRefusal::RungNotMintable);
        }
        Ok(Self { vif, scope, powers })
    }

    /// Build an authority at a rung no mint will grant.
    ///
    /// Only for proving what an unreachable arm of
    /// [`frame_filter`](Self::frame_filter) would install. That arm is
    /// exactly the dangerous kind — nothing exercises it, so nothing
    /// notices when it is wrong, and it has been wrong once already — so
    /// the test reaches it directly rather than the arm going unwitnessed.
    #[cfg(test)]
    pub(crate) fn at_unmintable_rung(vif: VifId, scope: Scope, powers: Powers) -> Self {
        debug_assert!(
            !scope.is_mintable(),
            "use `mint` for a rung a holder can actually be granted"
        );
        Self { vif, scope, powers }
    }

    /// The vif this authority is bound to, generation included.
    pub fn vif(&self) -> VifId {
        self.vif
    }

    /// What this authority covers.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The orthogonal powers this authority holds.
    pub fn powers(&self) -> Powers {
        self.powers
    }

    /// Derive a narrower authority.
    ///
    /// Borrows the parent, which survives — a capability's children are a
    /// derivation DAG, not a hand-off. Both dimensions are checked and the
    /// refusal names which one exceeded, because an operator who cannot tell
    /// a too-wide scope from a too-strong power cannot tell what to grant.
    pub fn derive(&self, scope: Scope, powers: Powers) -> Result<FlowCap, FlowRefusal> {
        if !scope.within(&self.scope) {
            return Err(FlowRefusal::WiderScope);
        }
        if !powers.within(&self.powers) {
            return Err(FlowRefusal::WiderPowers);
        }
        // A derivation can only narrow, so a parent at a mintable rung cannot
        // produce a child at an unmintable one — but the check is kept rather
        // than argued, so the rule holds if the ladder ever grows.
        if !scope.is_mintable() {
            return Err(FlowRefusal::RungNotMintable);
        }
        Ok(FlowCap {
            vif: self.vif,
            scope,
            powers,
        })
    }

    /// Whether this authority still resolves against the vif table's live
    /// identifier for its slot.
    ///
    /// Revocation is the generation tag already in [`VifId`]: closing a vif
    /// bumps it, so every outstanding authority for that slot stops resolving
    /// at its next use. O(1), and no holder has to be found.
    pub fn resolves_against(&self, live: VifId) -> bool {
        self.vif.index == live.index && self.vif.generation == live.generation
    }

    /// Whether a received datagram falls inside this authority.
    ///
    /// S0 admits nothing, which is what keeps the row on the datapath for a
    /// holder that was granted no scope.
    pub fn admits(&self, tuple: &FiveTuple) -> bool {
        match &self.scope {
            Scope::Mediated => false,
            Scope::Flow(f) => f.accepts(tuple),
            Scope::Range { addr, ports } => {
                tuple.dst == *addr && tuple.ports.is_some_and(|(_, d)| ports.contains(d))
            }
            // S3 imposes no L3 predicate, so every datagram that reached the
            // vif at all is inside it. The bound on an S3 holder is at L2
            // and is spent in [`frame_filter`](Self::frame_filter) — asking
            // this question of an interface grant is asking the wrong layer,
            // and answering `true` is what says so.
            Scope::Interface { .. } => true,
        }
    }

    /// The receive filter that **enforces** this authority.
    ///
    /// The bridge between the mint and the classifier, and the reason a
    /// capability here is not merely a description: the row installs this on
    /// the holder's vif, and from then on the predicate one layer down is what
    /// keeps another client's datagrams out of this one's window. The holder's
    /// `FlowCap` is its local proof; this is the enforcement.
    ///
    /// `None` for [`Scope::Mediated`] — an authority that covers nothing has
    /// no filter to install, because nothing is being delivered directly. That
    /// is S0 working as designed rather than a gap: the row stays on the
    /// datapath for that holder.
    pub fn frame_filter(&self) -> Option<FrameFilter> {
        let flow = match &self.scope {
            Scope::Mediated => return None,
            Scope::Flow(m) => Some(*m),
            // S2 is kept shut rather than approximated. `EndpointMatch` is
            // `AddrPort | Addr | Any`, so a span of ports has no
            // representation, and the nearest filter that *is* expressible —
            // the whole address, any protocol, any peer — is **wider than the
            // grant**, which is the one thing an enforcement may never be.
            // An earlier spelling emitted exactly that, under a comment
            // saying the classifier's flow half would narrow it further: this
            // is that flow half, and nothing narrows it.
            //
            // Closed costs nothing today because S2 cannot be minted
            // (`Scope::is_mintable`), so no holder reaches this arm. Giving
            // S2 an enforcement means giving `EndpointMatch` a range first.
            Scope::Range { .. } => return Some(FrameFilter::CLOSED),
            // S3's grant *is* the L2 half, so it is the one rung that does
            // not narrow the IPv4 filter below — it replaces it. Falling
            // through to that filter would have produced the two wrong
            // answers at once: IPv4-only, which is not raw L2, and
            // `flow: None` under an IPv4 EtherType, which is every other IP
            // client's datagram at a shared station address.
            Scope::Interface { .. } => return Some(self.raw_l2_filter()),
        };
        let mut filter = FrameFilter::protocol(EtherType::IPV4);
        filter.flow = flow;
        filter.promiscuous = self.powers.promiscuous;
        Some(filter)
    }

    /// **The filter S3 installs**: raw L2 at this authority's station, and
    /// nothing wider.
    ///
    /// Each field is a separate decision because each is a separate
    /// authority one layer down, and bundling them is how "raw L2" would
    /// come to imply the two powers the ladder deliberately keeps off it.
    fn raw_l2_filter(&self) -> FrameFilter {
        FrameFilter {
            // Every protocol — the one dimension in which S3 really is
            // wider than every rung below it, and the reason the frame row
            // is a second wall: `EtherTypeMatch::Any` is a capture power
            // there, so this filter does not install under an ordinary
            // client's policy (`PolicyRefusal::EtherType`).
            ethertype: EtherTypeMatch::Any,
            // ...and the dimension in which it is *not* wider: addressing.
            // The classifier evaluates this against the vif's own station
            // address, so it admits frames addressed to this station and no
            // other — which is the whole of the difference between raw L2
            // on a vif and raw L2 on the device.
            unicast: true,
            // A group this holder never joined carries traffic that was not
            // addressed to it, which is precisely what `promiscuous` names.
            // Tying the two here is what keeps the power orthogonal instead
            // of letting the rung imply it.
            multicast: if self.powers.promiscuous {
                MulticastMatch::All
            } else {
                MulticastMatch::None
            },
            // Broadcast is addressed to every station, this one included, so
            // it costs no power — and an L2 holder that could not hear ARP
            // would not have a working link.
            broadcast: true,
            promiscuous: self.powers.promiscuous,
            // The one place `flow: None` is the honest value rather than a
            // widening: S3 genuinely covers every protocol, including the
            // ones that carry no five-tuple to match on. A `Some` here would
            // be an L3 predicate silently narrowing an L2 grant, so the
            // holder would stop receiving ARP and never learn why.
            flow: None,
        }
    }

    /// Whether this authority may put `src` in the source address of a frame
    /// it transmits, given the `assigned` address its vif was issued.
    ///
    /// S3 is the rung that makes the question live, because it is the only
    /// one whose holder writes its own Ethernet header; below it the stack
    /// writes the header from [`Ifnet::hwaddr`](crate::Ifnet::hwaddr) and
    /// the assigned address is the only one on offer. Either way
    /// [`spoof`](Powers::spoof) is what lifts the restriction and nothing
    /// else does — an interface grant without it is bound to one source
    /// address, which is the orthogonality the ladder exists to keep.
    ///
    /// # This is the authority's answer, not the wire's
    ///
    /// **Advisory, deliberately.** The predicate that actually decides is
    /// the frame row's `validate_tx`, which refuses any frame whose source
    /// is not the vif's own address — unconditionally, consulting no grant
    /// at all. So today the wire is *stricter* than every authority this
    /// method can describe: an S3 holder with [`spoof`](Powers::spoof) still
    /// cannot forge, because nothing routes a power into that check.
    ///
    /// Fail-closed, and therefore safe to leave standing — but it means the
    /// spoof half of [`Powers`] is modelled and not yet wired, exactly as S2
    /// is modelled and not yet mintable. The sentence this method lets a
    /// holder form before it builds a frame is the same sentence the row
    /// will enforce once
    /// [`VifPolicy::source_mac`](kore_driver::net::VifPolicy::source_mac)
    /// has a caller; until then it agrees with the wire in the direction
    /// that matters and is tighter nowhere.
    pub fn may_source(&self, src: MacAddr, assigned: MacAddr) -> bool {
        self.powers.spoof || src == self.scope.station().unwrap_or(assigned)
    }

    /// Consume this authority.
    ///
    /// Takes `self` by value, so the holder provably no longer has it. The
    /// vif's generation is what makes revocation *effective* for copies that
    /// never existed; this is what makes it explicit for the one that did.
    pub fn revoke(self) {}
}
