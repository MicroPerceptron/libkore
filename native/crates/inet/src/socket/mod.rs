//! **Tier 2** — the socket-shaped face, written in terms of Tier 1 and
//! nothing else.
//!
//! `res/INET_STACK.md` §6.2 names three tiers. Tier 0 is the window ABI, Tier
//! 1 is the borrow-based datapath, and this is the third: the ergonomic face
//! for everyone who is not writing a datapath.
//!
//! ```ignore
//! let mut s = TcpStream::connect(flow, storage, (WAN, 80)).await?;
//! s.write_all(b"GET / HTTP/1.1\r\nHost: google.com\r\n\r\n").await?;
//! ```
//!
//! # The constraint that is the whole point
//!
//! This layer has **no privileged access**. It never calls
//! [`InetRow`](crate::row::InetRow), never mints a [`FlowCap`], never binds a
//! port, never installs a filter, and never learns a vif identifier the
//! authority it holds did not already name. Everything it needs arrives
//! through [`FlowFace`] — the Tier 1 seam — and if something it needs is not
//! there, that is a finding about Tier 1 rather than a reason to reach past
//! it.
//!
//! That is not a stylistic preference. It is the proof obligation: a Tier 2
//! that cannot be written without privilege is a Tier 1 that is not
//! sufficient, and a POSIX shim written later by someone who is not us would
//! hit the same wall. Writing this layer *is* the test.
//!
//! # A socket is a view of a capability, not an ambient name
//!
//! Every socket here is scoped by exactly one [`FlowCap`], and it takes its
//! own endpoints *from* that capability rather than being told them:
//!
//! - the local address and port are the ones the authority pins, so a socket
//!   cannot bind a port the row did not hand it;
//! - a peer a socket is asked to address is checked against the authority
//!   before a byte is written, and refused by name when it is outside;
//! - what the socket *receives* is already narrowed by the
//!   [`FrameFilter`](kore_driver::net::FrameFilter) the row installed when it
//!   minted the capability, one layer below this one.
//!
//! So the two directions are closed by different mechanisms, and deliberately:
//! receive by the classifier's predicate (which a socket cannot influence),
//! transmit by the refusal in this module (which is the socket declining to
//! use authority it does not hold). Neither is load-bearing alone, and the
//! transmit check is **not** the security boundary — `VifPolicy` one layer
//! down is. It is here so a misuse is named at its call site instead of
//! surfacing as a frame the row silently drops.
//!
//! # Refusals, not panics
//!
//! [`SocketRefusal`] names the dimension that failed, the way
//! [`OpenRefusal`](crate::row::OpenRefusal) and
//! [`InstallRefusal`](crate::flow::InstallRefusal) do one layer down. A
//! caller that cannot tell "outside my flow" from "the ring was full" cannot
//! tell a bug from back-pressure.
//!
//! # Why the futures drive the interface themselves
//!
//! Tier 1 offers no readiness signal: [`Ifnet::poll`] is a *drain*, not a
//! wake, and nothing in the seam says "a frame arrived". So every future here
//! does its own work on each poll — drain the interface, then try the
//! operation — and yields with [`Waker::wake_by_ref`] when it could not
//! finish, which is the level-triggered shape
//! [`kore_rt::block_on`](https://docs.rs/kore-rt) documents and the one
//! `services/dagfs`'s `yield_once` already uses.
//!
//! The cost is honest and recorded: a socket awaiting a peer occupies an
//! executor turn per sweep rather than parking. Fixing it means a doorbell
//! Tier 1 does not yet expose, which is a finding and not something this
//! module should invent.
//!
//! # What Tier 1 could not give, recorded rather than worked around
//!
//! Writing this layer is the test of Tier 1's sufficiency, so what it had to
//! reach for is the result. Five findings, each of which is a question for
//! the Tier 1 handle rather than something this module should decide:
//!
//! 1. **No readiness signal.** [`Ifnet::poll`] is a drain, not a wake, and
//!    nothing in the seam says "a frame arrived". Every future here therefore
//!    yields and re-polls instead of parking. The fix is a doorbell on the
//!    Tier 1 handle; until there is one, a socket awaiting a peer costs an
//!    executor turn per sweep.
//! 2. **No clock.** TCP cannot retransmit, and ARP cannot expire, without
//!    monotonic time — and this crate names no runtime, so [`FlowFace`]
//!    carries [`now_ms`](FlowFace::now_ms). If the production handle sources
//!    time differently, that method is where the two have to agree.
//! 3. **A capability carries endpoints but not routes.** [`FlowCap`] pins a
//!    5-tuple; reaching the peer also needs the prefix and the gateway, which
//!    are the row's [`Identity`] and are not in the authority. So the seam
//!    carries that too, and a socket reads it without being able to set it.
//! 4. **Revocation is not observable from here.** A capability stops
//!    resolving when its vif's generation moves
//!    ([`FlowCap::resolves_against`]), but that comparison needs the vif
//!    table's *live* identifier, which Tier 2 has no way to obtain and must
//!    not be handed. A socket on a revoked flow therefore looks exactly like
//!    a socket on a quiet link. Surfacing it belongs to Tier 1 — the handle
//!    knows when its vif died, and this layer never can.
//! 5. **One vif carries one filter, so a listener yields one connection.**
//!    See [`TcpListener`]. `accept()` cannot hand an accepted connection its
//!    own enforced flow, because there is no second vif to enforce it on and
//!    `InetRow::open_flow` refuses a second flow on a bound one. A server
//!    wanting concurrency opens a second vif.
//!
//! None of the five was worked around by reaching past the seam, which is the
//! claim this module exists to make good on.
//!
//! [`Waker::wake_by_ref`]: core::task::Waker::wake_by_ref
//! [`FlowCap::resolves_against`]: crate::flow::FlowCap::resolves_against

pub mod tcp;
pub mod udp;

pub use tcp::{TcpListener, TcpStream};
pub use udp::UdpSocket;

use kore_driver::net::flow::{EndpointMatch, FiveTuple, IpProto, Ipv4Addr};
use kore_driver::net::{EtherType, LinkState, MacAddr, VifId};

use crate::flow::{FaceRefusal, Flow, FlowCap, Scope};
use crate::ifnet::Ifnet;
use crate::netinet::{OutputError, TcpError};
use crate::row::Identity;

/// **The Tier 1 seam** — everything a socket may know, and nothing else.
///
/// A Tier 1 flow handle is three things at once, and this trait says so:
///
/// 1. **A frame face.** It *is* an [`Ifnet`], and the frames it delivers have
///    already been narrowed by the filter the row installed when it minted
///    the authority. That is why a socket needs no filtering of its own on
///    receive, and why handing a socket this face grants nothing beyond the
///    flow.
/// 2. **The authority the face is a view of** — [`cap`](Self::cap). Affine,
///    borrowed here and never cloned.
/// 3. **The facts a socket cannot derive.** The station's
///    [`Identity`](Self::identity) (a peer's next hop is a routing question,
///    and a capability says nothing about routes) and the
///    [clock](Self::now_ms) (TCP cannot retransmit without one, and
///    `crates/inet` names no runtime, so the time arrives from whoever built
///    the flow).
///
/// # Why a trait here rather than a concrete handle
///
/// The production Tier 1 handle is being built beside this
/// (`res/INET_STACK.md` §6.2); naming it from here would couple the two and
/// make whichever lands second rewrite the other. A seam is the tree's answer
/// to that everywhere else — [`FlowInstall`](crate::flow::FlowInstall) for the
/// mint, `StorageProvider` for `services/dagfs` — so it is the answer here.
/// [`LocalFlow`] is the impl for a composition holding both halves, and it is
/// what the host tests run against.
pub trait FlowFace: Ifnet {
    /// The authority this face is a view of.
    fn cap(&self) -> &FlowCap;

    /// The station's address, prefix and gateway.
    ///
    /// Not privilege: a socket can read what the row decided, and cannot set
    /// it. [`Identity`] has no constructor a holder of one can use to widen
    /// anything — changing it means minting a new one through the row.
    fn identity(&self) -> &Identity;

    /// Milliseconds since an arbitrary fixed origin.
    ///
    /// Monotonic, and the only time this crate ever sees. Every protocol
    /// below takes `now_ms` as a parameter for exactly this reason: the
    /// engine owns no clock, so the composition supplies one.
    fn now_ms(&self) -> u64;
}

/// Why a socket refused.
///
/// One variant per dimension that can fail, because a caller that cannot tell
/// them apart cannot tell a misuse from back-pressure. In particular
/// [`OutsideFlow`](Self::OutsideFlow) is a *programming* error — the caller
/// asked to use authority it does not hold — while [`Output`](Self::Output)
/// carrying [`OutputError::NoSlot`] is an ordinary transient the socket
/// itself retries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketRefusal {
    /// The authority grants no scope at all (S0), so a socket over it could
    /// address nothing and receive nothing.
    ///
    /// Named separately from [`OutsideFlow`](Self::OutsideFlow) because the
    /// fix is different: an S0 holder stays on the row's datapath by design
    /// and wants a *mint*, not a narrower peer.
    Mediated,
    /// The authority does not pin a local address *and* port, so a socket
    /// over it has no endpoint to be.
    ///
    /// What an S2/S3 authority looks like from here. Those rungs are modelled
    /// and not mintable (`res/INET_STACK.md` §4.4); when one gains a consumer
    /// it will need a socket that draws an endpoint from inside the range,
    /// and that is a slice of its own rather than a guess made here.
    Unpinned,
    /// The protocol or peer named is outside what the authority covers.
    OutsideFlow,
    /// The borrowed storage handed in was too small for the row that needs it.
    Storage,
    /// The peer refused the connection, or reset an established one.
    Reset,
    /// The connection is closed; no further byte will move.
    Closed,
    /// The interface refused the datagram, naming why.
    Output(OutputError),
    /// The borrowed TCP implementation refused, naming why.
    Tcp(TcpError),
}

impl From<OutputError> for SocketRefusal {
    fn from(e: OutputError) -> Self {
        Self::Output(e)
    }
}

impl From<TcpError> for SocketRefusal {
    fn from(e: TcpError) -> Self {
        Self::Tcp(e)
    }
}

/// The local endpoint an authority pins.
///
/// A socket *is* this endpoint — it does not choose one and it cannot be told
/// one. An authority that pins no port has no socket shape, which is
/// [`SocketRefusal::Unpinned`] rather than a default nobody checked.
pub(crate) fn local_of(cap: &FlowCap) -> Result<(Ipv4Addr, u16), SocketRefusal> {
    match cap.scope() {
        Scope::Mediated => Err(SocketRefusal::Mediated),
        Scope::Flow(m) => match m.local {
            EndpointMatch::AddrPort { addr, port } => Ok((addr, port)),
            EndpointMatch::Addr(_) | EndpointMatch::Any => Err(SocketRefusal::Unpinned),
        },
        Scope::Range { .. } | Scope::Interface { .. } => Err(SocketRefusal::Unpinned),
    }
}

/// The peer an authority pins, when it pins one.
///
/// `None` for a listening authority, whose peer end is deliberately
/// [`EndpointMatch::Any`] — that is the difference between a connection and a
/// listener, stated in the capability rather than in a flag beside it.
pub(crate) fn peer_of(cap: &FlowCap) -> Option<(Ipv4Addr, u16)> {
    match cap.scope() {
        Scope::Flow(m) => match m.peer {
            EndpointMatch::AddrPort { addr, port } => Some((addr, port)),
            EndpointMatch::Addr(_) | EndpointMatch::Any => None,
        },
        _ => None,
    }
}

/// Whether `cap` admits the conversation `(proto, local, peer)`.
///
/// Asked in the **receive** direction, which is the whole trick: an authority
/// is a statement about what may be delivered, so "may I send here" is
/// answered by "would I be allowed to hear the reply". A socket that could
/// send where it cannot receive would be a one-way spoofing primitive, and
/// the authority already says it is not one.
pub(crate) fn admits(
    cap: &FlowCap,
    proto: IpProto,
    local: (Ipv4Addr, u16),
    peer: (Ipv4Addr, u16),
) -> bool {
    cap.admits(&FiveTuple {
        proto,
        src: peer.0,
        dst: local.0,
        ports: Some((peer.1, local.1)),
    })
}

/// **The Tier 1 impl for a composition that holds both halves** — the cap, the
/// facts, and the interface, in one address space.
///
/// The same shape [`RecordingInstall`](crate::flow::RecordingInstall) has one
/// layer down and [`MockIfnet`](crate::ifnet::MockIfnet) has one below that: a
/// real impl of the shipping contract, so a host test exercises the engine
/// rather than a parallel one. It is also what a founded row uses when it
/// mints a flow for itself, which is why it is not `cfg(test)`.
///
/// The clock is set rather than read from anywhere: this crate names no
/// runtime, so a composition that has one calls
/// [`set_now_ms`](Self::set_now_ms) on its sweep, and a test advances it by
/// hand.
pub struct LocalFlow<I: Ifnet> {
    net: I,
    cap: FlowCap,
    identity: Identity,
    now_ms: u64,
}

impl<I: Ifnet> LocalFlow<I> {
    /// Bind an authority, an identity and an interface into one flow handle.
    ///
    /// Takes the [`FlowCap`] by value because it is affine — the composition
    /// that minted it keeps nothing it could hand to a second socket. Note
    /// what that does *not* mean: the cap is the recipe the filter was
    /// computed from, not the authority itself. What enforces this flow is
    /// the [`FrameFilter`] the row installed on the vif, which is why
    /// [`Self::tier1`] reads the filter back rather than rebuilding it.
    ///
    /// [`FrameFilter`]: kore_driver::net::FrameFilter
    pub fn new(net: I, cap: FlowCap, identity: Identity) -> Self {
        Self {
            net,
            cap,
            identity,
            now_ms: 0,
        }
    }

    /// Lend this composition's Tier 1 face.
    ///
    /// The two layers meet here. A [`Flow`] is the authority and the
    /// interface — zero-copy [`reserve`](Flow::reserve), `commit` and
    /// [`drain`](Flow::drain), and nothing else. This composition is that
    /// plus the two facts a protocol needs and an authority does not carry:
    /// the station's [`Identity`] and a clock. Tier 2 is written against the
    /// composition; anything wanting the frames themselves takes a face.
    ///
    /// `live` is the vif table's current identifier for the slot, and it is
    /// supplied by the caller rather than remembered here on purpose: the
    /// live generation is the row's fact, not the holder's, and a holder that
    /// cached it could not be revoked. A stale authority refuses with
    /// [`FaceRefusal::Stale`].
    pub fn tier1(&mut self, live: VifId) -> Result<Flow<'_, I>, FaceRefusal> {
        Flow::bind(&self.cap, &mut self.net, live)
    }

    /// Set the clock this flow reports.
    pub fn set_now_ms(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
    }

    /// Move the clock forward, saturating.
    pub fn advance_ms(&mut self, ms: u64) {
        self.now_ms = self.now_ms.saturating_add(ms);
    }

    /// The interface, for a composition that must reach it directly.
    pub fn ifnet(&self) -> &I {
        &self.net
    }

    /// The interface, mutably.
    pub fn ifnet_mut(&mut self) -> &mut I {
        &mut self.net
    }

    /// Give the authority back, consuming the flow.
    ///
    /// By value in both directions, so a flow that has been dissolved cannot
    /// still be driving a socket.
    pub fn into_parts(self) -> (I, FlowCap) {
        (self.net, self.cap)
    }
}

impl<I: Ifnet> Ifnet for LocalFlow<I> {
    type TxSlot<'a>
        = I::TxSlot<'a>
    where
        Self: 'a;

    fn hwaddr(&self) -> MacAddr {
        self.net.hwaddr()
    }

    fn mtu(&self) -> usize {
        self.net.mtu()
    }

    fn link(&self) -> LinkState {
        self.net.link()
    }

    fn reserve(&mut self) -> Option<Self::TxSlot<'_>> {
        self.net.reserve()
    }

    fn poll<F>(&mut self, f: F) -> bool
    where
        F: FnMut(&[u8], EtherType),
    {
        self.net.poll(f)
    }
}

impl<I: Ifnet> FlowFace for LocalFlow<I> {
    fn cap(&self) -> &FlowCap {
        &self.cap
    }

    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn now_ms(&self) -> u64 {
        self.now_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Powers, Scope};
    use crate::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
    use kore_driver::net::VifId;
    use kore_driver::net::flow::FlowMatch;

    const MY_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const PEER: Ipv4Addr = Ipv4Addr([10, 0, 2, 50]);
    const OTHER: Ipv4Addr = Ipv4Addr([10, 0, 2, 99]);

    fn vif() -> VifId {
        VifId {
            index: 0,
            generation: 1,
        }
    }

    fn connected() -> FlowCap {
        FlowCap::mint(
            vif(),
            Scope::Flow(FlowMatch::connection(IpProto::UDP, (ME, 5000), (PEER, 7))),
            Powers::NONE,
        )
        .expect("S1 mints")
    }

    /// A socket takes its endpoint *from* the authority, so the two cannot
    /// disagree.
    #[test]
    fn the_local_endpoint_comes_from_the_authority() {
        assert_eq!(local_of(&connected()), Ok((ME, 5000)));
        assert_eq!(peer_of(&connected()), Some((PEER, 7)));
    }

    /// A listening authority pins no peer — that is what makes it a listener,
    /// and it is said in the capability rather than in a flag beside it.
    #[test]
    fn a_listening_authority_pins_no_peer() {
        let cap = FlowCap::mint(
            vif(),
            Scope::Flow(FlowMatch::listener(IpProto::TCP, (ME, 8080))),
            Powers::NONE,
        )
        .expect("S1 mints");
        assert_eq!(local_of(&cap), Ok((ME, 8080)));
        assert_eq!(peer_of(&cap), None, "any peer may connect");
    }

    /// The composition lends the Tier 1 face, so the two layers are one
    /// object and not two spellings of one.
    ///
    /// What this pins is the direction of the dependency: Tier 2 holds the
    /// composition, and reaches frames by asking it for a face. Were the
    /// face a separate handle a caller assembled itself, the clock and the
    /// identity would have to be threaded past it, and the two could be
    /// bound to different interfaces.
    #[test]
    fn the_composition_lends_its_tier_1_face() {
        let mut scratch = [0u8; MOCK_SCRATCH_BYTES];
        let net = MockIfnet::new(&mut scratch, MY_MAC).expect("scratch");
        let ident = Identity::new(ME, 24, Some(GW)).expect("/24");
        let mut flow = LocalFlow::new(net, connected(), ident);
        let face = flow.tier1(vif()).expect("the live generation matches");
        assert_eq!(face.vif(), vif());
        assert_eq!(
            face.ethertype(),
            EtherType::IPV4,
            "read back out of the installed filter, never chosen here"
        );
    }

    /// A revoked authority refuses at the seam, and says so by name.
    ///
    /// The live generation is the row's fact and arrives as an argument, so
    /// a holder cannot cache its way out of a revocation. This is the one
    /// refusal that is not a misuse: the holder did nothing wrong, it
    /// outlived its grant.
    #[test]
    fn a_stale_generation_is_lent_no_face() {
        let mut scratch = [0u8; MOCK_SCRATCH_BYTES];
        let net = MockIfnet::new(&mut scratch, MY_MAC).expect("scratch");
        let ident = Identity::new(ME, 24, Some(GW)).expect("/24");
        let mut flow = LocalFlow::new(net, connected(), ident);
        let moved_on = VifId {
            index: 0,
            generation: 2,
        };
        match flow.tier1(moved_on) {
            Err(FaceRefusal::Stale) => {}
            Err(other) => panic!("refused, but not as a revocation: {other:?}"),
            Ok(_) => panic!("a stale generation was lent a face"),
        }
    }

    /// S0 grants nothing, so there is no socket to be — and it is refused by
    /// its own name, because the fix is a mint and not a narrower peer.
    #[test]
    fn the_mediated_grant_is_no_socket() {
        let cap = FlowCap::mediated(vif());
        assert_eq!(local_of(&cap), Err(SocketRefusal::Mediated));
    }

    /// Admission is asked in the receive direction, so a socket may send
    /// exactly where it would be allowed to hear the answer.
    #[test]
    fn admission_follows_the_authority_in_both_directions() {
        let cap = connected();
        assert!(admits(&cap, IpProto::UDP, (ME, 5000), (PEER, 7)));
        assert!(
            !admits(&cap, IpProto::UDP, (ME, 5000), (OTHER, 7)),
            "a different peer is outside the flow"
        );
        assert!(
            !admits(&cap, IpProto::UDP, (ME, 5000), (PEER, 8)),
            "a different peer port is outside the flow"
        );
        assert!(
            !admits(&cap, IpProto::TCP, (ME, 5000), (PEER, 7)),
            "a different protocol is outside the flow"
        );
    }

    /// The handle delegates the whole frame contract, so a socket over it is
    /// a socket over the interface the row narrowed — not over a wrapper that
    /// could have widened something.
    #[test]
    fn the_handle_is_the_interface_it_wraps() {
        #[repr(align(64))]
        struct Aligned<const N: usize>([u8; N]);
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut scratch.0, MY_MAC).expect("scratch");
        let identity = Identity::new(ME, 24, Some(GW)).expect("valid");
        let mut flow = LocalFlow::new(net, connected(), identity);

        assert_eq!(flow.hwaddr(), MY_MAC);
        assert_eq!(flow.identity().addr(), ME);
        assert_eq!(flow.now_ms(), 0);
        flow.advance_ms(17);
        assert_eq!(flow.now_ms(), 17);
        assert!(flow.reserve().is_some(), "the double always has a slot");
    }
}
