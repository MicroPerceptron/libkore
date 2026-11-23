//! **The AF_INET domain** — the protocol switch, and the engine that drives
//! it over an [`Ifnet`].
//!
//! # The switch, and what it is not yet
//!
//! `res/INET_STACK.md` §6.1 describes dispatch as a table keyed by IP
//! protocol number — contract trait, generic engine, selection by row, the
//! construction every other seam in this tree uses and which BSD reached
//! first and called `protosw`. **That is the design, and it is not what is
//! written here.**
//!
//! [`Protosw`] is a `matches!` over the two protocols that exist, and
//! [`IpStack::input`] dispatches with a `match` on the protocol number.
//! Adding a protocol is an arm there and an implementation beside
//! [`icmp`] — more than a row, less than a rewrite. The row is **not yet
//! seamed**, which is a weaker position than "not yet replaced", and the
//! difference matters to anyone planning to swap one: there is no trait a
//! replacement could satisfy today.
//!
//! Drawing that trait is its own slice (`res/INET_STACK.md` §6.1 names the
//! shape it would take). Until it lands, this header says what the code is
//! rather than what the design intends, because the gap between those two
//! is where every correction in this subsystem has come from.
//!
//! # What is borrowed and what is not
//!
//! The wire *encodings* come from the vendored smoltcp (`vendor/smoltcp`).
//! The ARP cache policy, the routing decision, the reserve-forward output
//! discipline and this switch are written here, because each carries part of
//! the capability model and none of them is a byte layout.

pub mod arp;
pub mod dhcp;
pub mod dns;
pub mod icmp;
pub mod ip;
pub mod ping;
pub mod tcp;
pub mod udp;

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_driver::net::{EtherType, MacAddr};
use smoltcp::wire::{EthernetAddress, Ipv4Address};

use crate::ifnet::Ifnet;
use crate::row::Identity;

pub use arp::{ArpCache, Resolution};
pub use dhcp::{DhcpClient, DhcpState, Lease};
pub use dns::{DnsState, Resolver};
pub use icmp::IcmpEvent;
pub use ip::{Datagram, OutputError};
pub use tcp::{TcpError, TcpRow, TcpState, TcpStorage};
pub use udp::{UdpBindings, UdpDatagram};

// ── the one place kore's scalars and the borrowed wire's meet ───────────

/// kore's station address as the wire crate's.
pub fn to_eth(a: MacAddr) -> EthernetAddress {
    EthernetAddress(a.0)
}

/// The wire crate's station address as kore's.
pub fn from_eth(a: EthernetAddress) -> MacAddr {
    MacAddr(a.0)
}

/// kore's IPv4 address as the wire crate's.
pub fn to_ip(a: Ipv4Addr) -> Ipv4Address {
    Ipv4Address::from_octets(a.0)
}

/// The wire crate's IPv4 address as kore's.
pub fn from_ip(a: Ipv4Address) -> Ipv4Addr {
    Ipv4Addr(a.octets())
}

/// Something a received frame caused.
///
/// Reported rather than swallowed, so a caller can witness a round trip
/// without the engine keeping state about who was waiting for what.
///
/// Borrowed for the duration of the dispatch, because [`Event::Udp`] carries
/// the datagram's bytes. An earlier shape reported a *length* instead, which
/// is enough to witness that something arrived and not enough to be a
/// datagram's only delivery — the socket layer over this engine
/// ([`crate::socket`]) is the consumer that made the difference load-bearing.
/// The borrow is the staging buffer [`IpStack::input`] owns, so a handler
/// reads the frame in place and nothing is copied twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event<'a> {
    /// An ARP exchange moved — a translation was learned, or a request for
    /// this station was answered.
    Arp {
        /// Whether this station answered a request.
        answered: bool,
    },
    /// An ICMP datagram was understood.
    Icmp(IcmpEvent),
    /// A UDP datagram reached a bound port the stack does not itself own.
    Udp {
        /// Who sent it.
        from: (Ipv4Addr, u16),
        /// The local port it reached.
        local_port: u16,
        /// The payload, borrowed from the staging buffer.
        payload: &'a [u8],
    },
    /// DHCP completed: this station now has an address.
    Leased(Lease),
    /// A name resolved.
    Resolved {
        /// The address it resolved to.
        addr: Ipv4Addr,
    },
    /// A datagram arrived for a protocol with no row in the switch.
    ///
    /// Named rather than dropped silently: "nothing is listening" and "the
    /// packet was malformed" are different facts, and a caller debugging a
    /// protocol wants to tell them apart.
    NoProtocol {
        /// The protocol nobody claimed.
        proto: IpProto,
        /// Who sent it.
        from: Ipv4Addr,
    },
}

/// Which protocol rows this build carries.
///
/// The switch itself. A row is `(number, handler)`; the engine matches the
/// number and calls the handler, and knows nothing else about either.
/// B2 adds UDP here, B3 adds TCP, and neither touches [`IpStack::input`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Protosw;

impl Protosw {
    /// Whether a row claims `proto`.
    pub fn claims(&self, proto: IpProto) -> bool {
        matches!(proto, IpProto::ICMP | IpProto::UDP)
    }
}

/// The AF_INET engine over one interface.
///
/// Generic over [`Ifnet`], so the production instantiation rides a vif and the
/// test instantiation rides the host double — the same engine, selected by
/// type.
pub struct IpStack<I: Ifnet> {
    net: I,
    identity: Identity,
    arp: ArpCache,
    protosw: Protosw,
    udp: UdpBindings,
    dhcp: DhcpClient,
    dns: Resolver,
}

impl<I: Ifnet> IpStack<I> {
    /// Bind the engine to an interface and an identity.
    pub fn new(net: I, identity: Identity) -> Self {
        let mut udp = UdpBindings::new();
        // The stack's own two clients are bound from the start: a DHCP reply
        // and a DNS answer have to be deliverable before anything above has
        // had a chance to ask for them.
        udp.bind(smoltcp::wire::DHCP_CLIENT_PORT);
        Self {
            net,
            identity,
            arp: ArpCache::new(),
            protosw: Protosw,
            udp,
            dhcp: DhcpClient::new(),
            dns: Resolver::new(),
        }
    }

    /// The DHCP client.
    pub fn dhcp(&self) -> &DhcpClient {
        &self.dhcp
    }

    /// The resolver.
    pub fn dns(&self) -> &Resolver {
        &self.dns
    }

    /// The UDP bindings.
    pub fn udp(&self) -> &UdpBindings {
        &self.udp
    }

    /// Deliver `port` to the caller as [`Event::Udp`].
    pub fn bind_udp(&mut self, port: u16) -> bool {
        self.udp.bind(port)
    }

    /// Stop delivering `port`.
    pub fn unbind_udp(&mut self, port: u16) {
        self.udp.unbind(port);
    }

    /// Send a UDP datagram.
    pub fn send_udp(
        &mut self,
        src_port: u16,
        dst: (Ipv4Addr, u16),
        payload: &[u8],
        now_ms: u64,
    ) -> Result<(), OutputError> {
        udp::output(
            &mut self.net,
            &mut self.arp,
            &self.identity,
            src_port,
            dst,
            payload,
            now_ms,
        )
    }

    /// Drive the DHCP client one step.
    ///
    /// Returns whether a message went out. A caller runs this on its sweep
    /// until [`DhcpClient::lease`] answers, then adopts the lease as the
    /// station's identity.
    pub fn drive_dhcp(&mut self, now_ms: u64, xid_seed: u32) -> bool {
        self.dhcp.poll(&mut self.net, now_ms, xid_seed)
    }

    /// Ask a resolver for `name`, if a query is due.
    ///
    /// Returns whether a query went out. The resolver's address comes from
    /// the identity, so this refuses rather than guessing when DHCP has not
    /// supplied one.
    pub fn resolve_name(
        &mut self,
        name: &str,
        xid: u16,
        local_port: u16,
        now_ms: u64,
    ) -> Result<bool, OutputError> {
        if !self.dns.due(now_ms) {
            return Ok(false);
        }
        let server = self.identity.dns().ok_or(OutputError::Unreachable)?;
        let mut query = [0u8; 512];
        let Some(len) = dns::encode_query(name, xid, &mut query) else {
            return Ok(false);
        };
        self.udp.bind(local_port);
        self.send_udp(local_port, (server, dns::DNS_PORT), &query[..len], now_ms)?;
        self.dns.sent(xid, server, local_port, name, now_ms);
        Ok(true)
    }

    /// The interface.
    pub fn ifnet(&self) -> &I {
        &self.net
    }

    /// The interface, mutably.
    pub fn ifnet_mut(&mut self) -> &mut I {
        &mut self.net
    }

    /// This station's identity.
    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    /// Adopt a new identity — what DHCP's lease does.
    ///
    /// The ARP cache is cleared with it: translations learned under one
    /// address say nothing about the segment the next one is on, and keeping
    /// them would route the first packets of a new lease at a stale gateway.
    pub fn set_identity(&mut self, identity: Identity) {
        self.identity = identity;
        self.arp.clear();
    }

    /// The translation table.
    pub fn arp(&self) -> &ArpCache {
        &self.arp
    }

    /// The translation table, mutably — for a composition that seeds it from
    /// a lease, and for tests that start past the ARP exchange.
    pub fn arp_mut(&mut self) -> &mut ArpCache {
        &mut self.arp
    }

    /// Resolve a station's hardware address, asking if needed.
    pub fn resolve(&mut self, target: Ipv4Addr, now_ms: u64) -> Resolution {
        arp::resolve(
            &mut self.arp,
            &mut self.net,
            self.identity.addr(),
            target,
            now_ms,
        )
    }

    /// Send an ICMP echo request.
    pub fn echo_request(
        &mut self,
        dst: Ipv4Addr,
        ident: u16,
        seq: u16,
        payload: &[u8],
        now_ms: u64,
    ) -> Result<(), OutputError> {
        icmp::echo_request(
            &mut self.net,
            &mut self.arp,
            &self.identity,
            dst,
            ident,
            seq,
            payload,
            now_ms,
        )
    }

    /// Drain the interface, dispatching each frame, and report what happened.
    ///
    /// `f` is called once per understood frame. Returns whether more may be
    /// waiting — the drive-cycle posture, so one sweep is bounded and the
    /// caller comes back rather than looping on an unknown arrival rate.
    pub fn input<F>(&mut self, now_ms: u64, mut f: F) -> bool
    where
        F: FnMut(Event<'_>),
    {
        // One frame is staged out of the borrow before anything is dispatched:
        // the handlers need `&mut self.net` to answer, and the frame bytes
        // borrow it. The staging buffer is this engine's own, so the copy is
        // bounded by one MTU and never allocates.
        let mut staged = [0u8; kore_driver::net::FRAME_MAX];
        let mut len = 0usize;
        let mut ethertype = EtherType(0);
        let more = self.net.poll(|bytes, et| {
            let n = bytes.len().min(staged.len());
            staged[..n].copy_from_slice(&bytes[..n]);
            len = n;
            ethertype = et;
        });
        if len == 0 {
            return more;
        }
        let body = &staged[kore_driver::net::frame::HEADER_LEN..len];

        match ethertype {
            EtherType::ARP => {
                let answered = arp::input(
                    &mut self.arp,
                    &mut self.net,
                    self.identity.addr(),
                    body,
                    now_ms,
                );
                f(Event::Arp { answered });
            }
            EtherType::IPV4 => {
                let Some(datagram) = ip::input(&self.identity, body) else {
                    return more;
                };
                if !self.protosw.claims(datagram.proto) {
                    f(Event::NoProtocol {
                        proto: datagram.proto,
                        from: datagram.src,
                    });
                    return more;
                }
                // The switch's one row today. Each further row is a `match`
                // arm here and an implementation beside `icmp`; the engine
                // above this point does not change.
                match datagram.proto {
                    IpProto::ICMP => {
                        if let Some(event) = icmp::input(
                            &mut self.net,
                            &mut self.arp,
                            &self.identity,
                            datagram.src,
                            datagram.payload,
                            now_ms,
                        ) {
                            f(Event::Icmp(event));
                        }
                    }
                    IpProto::UDP => {
                        let Some(d) =
                            udp::input(&self.udp, datagram.src, datagram.dst, datagram.payload)
                        else {
                            return more;
                        };
                        // The stack's own two clients are served here rather
                        // than reported: a DHCP reply and a DNS answer are
                        // this engine's business, not its caller's.
                        if d.local_port == smoltcp::wire::DHCP_CLIENT_PORT {
                            let hw = self.net.hwaddr();
                            if let Some(lease) = self.dhcp.input(d.payload, hw) {
                                f(Event::Leased(lease));
                            }
                        } else if self.dns.state() != dns::DnsState::Idle
                            && let Some(addr) = self.dns.input(d.src, d.local_port, d.payload)
                        {
                            f(Event::Resolved { addr });
                        } else {
                            f(Event::Udp {
                                from: d.src,
                                local_port: d.local_port,
                                payload: d.payload,
                            });
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        more
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
    use kore_driver::net::frame::HEADER_LEN;
    use kore_driver::net::{FrameHead, MacAddr};

    const MY_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const GW_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
    const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const WAN: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    fn identity() -> Identity {
        Identity::new(ME, 24, Some(GW)).expect("valid")
    }

    /// Build an Ethernet frame around `body`.
    fn frame(dst: MacAddr, src: MacAddr, et: EtherType, body: &[u8], out: &mut [u8]) -> usize {
        FrameHead {
            dst,
            src,
            ethertype: et,
        }
        .write(out)
        .expect("header fits");
        out[HEADER_LEN..HEADER_LEN + body.len()].copy_from_slice(body);
        HEADER_LEN + body.len()
    }

    /// An ARP packet, as a peer would put it on the wire.
    fn arp_packet(
        op: ArpOp,
        sender_mac: MacAddr,
        sender_ip: Ipv4Addr,
        target_mac: MacAddr,
        target_ip: Ipv4Addr,
        out: &mut [u8],
    ) -> usize {
        use smoltcp::wire::{ArpOperation, ArpPacket, ArpRepr};
        let repr = ArpRepr::EthernetIpv4 {
            operation: match op {
                ArpOp::Request => ArpOperation::Request,
                ArpOp::Reply => ArpOperation::Reply,
            },
            source_hardware_addr: to_eth(sender_mac),
            source_protocol_addr: to_ip(sender_ip),
            target_hardware_addr: to_eth(target_mac),
            target_protocol_addr: to_ip(target_ip),
        };
        let len = repr.buffer_len();
        let mut p = ArpPacket::new_unchecked(&mut out[..len]);
        repr.emit(&mut p);
        len
    }

    enum ArpOp {
        Request,
        Reply,
    }

    /// What a [`Event::Udp`] said, as values that outlive the dispatch.
    ///
    /// The event borrows the engine's staging buffer, so an assertion about
    /// it has to name the facts rather than keep the event — which is the
    /// honest shape anyway: a test asserts what arrived, not which enum
    /// carried it.
    fn delivered(e: Event<'_>) -> Option<((Ipv4Addr, u16), u16, usize)> {
        match e {
            Event::Udp {
                from,
                local_port,
                payload,
            } => Some((from, local_port, payload.len())),
            _ => None,
        }
    }

    fn stack(scratch: &mut [u8]) -> IpStack<MockIfnet<'_>> {
        let net = MockIfnet::new(scratch, MY_MAC).expect("scratch");
        IpStack::new(net, identity())
    }

    // ── ARP ─────────────────────────────────────────────────────────────

    /// Resolving an unknown station puts a broadcast request on the wire and
    /// says so, rather than failing.
    #[test]
    fn resolving_an_unknown_station_asks_on_the_wire() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        assert_eq!(s.resolve(GW, 0), Resolution::Asked);
        let (sent, et) = s.ifnet().last_sent().expect("a request went out");
        assert_eq!(et, EtherType::ARP);
        let head = FrameHead::parse(sent).expect("well formed");
        assert_eq!(head.dst, MacAddr::BROADCAST, "a request is broadcast");
        assert_eq!(head.src, MY_MAC);
    }

    /// The rate limit: a second ask for the same target inside the interval
    /// does not put another broadcast on the wire.
    #[test]
    fn repeated_asks_are_rate_limited() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        assert_eq!(s.resolve(GW, 0), Resolution::Asked);
        assert_eq!(s.ifnet().sent_count(), 1);
        assert_eq!(s.resolve(GW, 10), Resolution::Pending, "too soon");
        assert_eq!(s.ifnet().sent_count(), 1, "no second broadcast");
        assert_eq!(
            s.resolve(GW, arp::ARP_RETRY_INTERVAL_MS),
            Resolution::Asked,
            "the interval elapsed"
        );
        assert_eq!(s.ifnet().sent_count(), 2);
    }

    /// A reply addressed to this station is learned, and the next resolve is
    /// answered from cache without touching the wire.
    #[test]
    fn a_reply_is_learned_and_then_served_from_cache() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let mut body = [0u8; 64];
        let n = arp_packet(ArpOp::Reply, GW_MAC, GW, MY_MAC, ME, &mut body);
        let mut f = [0u8; 128];
        let len = frame(MY_MAC, GW_MAC, EtherType::ARP, &body[..n], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::ARP)
            .expect("injects");

        let mut events = 0;
        s.input(0, |_| events += 1);
        assert_eq!(events, 1);
        assert_eq!(s.arp().len(), 1);

        let before = s.ifnet().sent_count();
        assert_eq!(s.resolve(GW, 0), Resolution::Known(GW_MAC));
        assert_eq!(s.ifnet().sent_count(), before, "cache hit sends nothing");
    }

    /// **The spoofing closure.** A request for a *neighbour's* address is not
    /// answered and teaches nothing — answering for someone else is exactly
    /// the attack ARP is known for.
    #[test]
    fn a_request_for_a_neighbour_is_neither_answered_nor_learned() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let neighbour = Ipv4Addr([10, 0, 2, 99]);
        let stranger_mac = MacAddr([0x02, 0, 0, 0, 0, 0x77]);
        let mut body = [0u8; 64];
        let n = arp_packet(
            ArpOp::Request,
            stranger_mac,
            Ipv4Addr([10, 0, 2, 50]),
            MacAddr::ZERO,
            neighbour,
            &mut body,
        );
        let mut f = [0u8; 128];
        let len = frame(
            MacAddr::BROADCAST,
            stranger_mac,
            EtherType::ARP,
            &body[..n],
            &mut f,
        );
        s.ifnet_mut()
            .inject(&f[..len], EtherType::ARP)
            .expect("injects");

        let before = s.ifnet().sent_count();
        let mut answered = None;
        s.input(0, |e| {
            if let Event::Arp { answered: a } = e {
                answered = Some(a);
            }
        });
        assert_eq!(answered, Some(false), "not ours to answer");
        assert_eq!(s.ifnet().sent_count(), before, "nothing was sent");
        assert!(
            s.arp().is_empty(),
            "a request for a neighbour teaches this cache nothing"
        );
    }

    /// A request for *this* station is answered, to the asker, and the asker
    /// is learned.
    #[test]
    fn a_request_for_this_station_is_answered() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let mut body = [0u8; 64];
        let n = arp_packet(ArpOp::Request, GW_MAC, GW, MacAddr::ZERO, ME, &mut body);
        let mut f = [0u8; 128];
        let len = frame(
            MacAddr::BROADCAST,
            GW_MAC,
            EtherType::ARP,
            &body[..n],
            &mut f,
        );
        s.ifnet_mut()
            .inject(&f[..len], EtherType::ARP)
            .expect("injects");

        let mut answered = None;
        s.input(0, |e| {
            if let Event::Arp { answered: a } = e {
                answered = Some(a);
            }
        });
        assert_eq!(answered, Some(true));

        let (sent, et) = s.ifnet().last_sent().expect("a reply went out");
        assert_eq!(et, EtherType::ARP);
        let head = FrameHead::parse(sent).expect("well formed");
        assert_eq!(head.dst, GW_MAC, "answered to the asker, not broadcast");
        assert_eq!(
            s.arp().get(GW, 0),
            Some(GW_MAC),
            "and the asker was learned"
        );
    }

    /// A learned translation expires, and the stack asks again rather than
    /// using a stale one.
    #[test]
    fn a_translation_expires() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let mut body = [0u8; 64];
        let n = arp_packet(ArpOp::Reply, GW_MAC, GW, MY_MAC, ME, &mut body);
        let mut f = [0u8; 128];
        let len = frame(MY_MAC, GW_MAC, EtherType::ARP, &body[..n], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::ARP)
            .expect("injects");
        s.input(0, |_| {});

        assert_eq!(
            s.arp().get(GW, arp::ARP_ENTRY_LIFETIME_MS - 1),
            Some(GW_MAC)
        );
        assert_eq!(
            s.arp().get(GW, arp::ARP_ENTRY_LIFETIME_MS),
            None,
            "the lifetime is not open-ended"
        );
    }

    // ── IPv4 + ICMP ─────────────────────────────────────────────────────

    /// An echo to an unresolved peer is `Pending`, not a failure: the ARP
    /// request goes out and the caller retries.
    #[test]
    fn an_echo_to_an_unresolved_peer_is_pending() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        assert_eq!(
            s.echo_request(GW, 1, 1, b"probe", 0),
            Err(OutputError::Pending)
        );
        assert_eq!(s.ifnet().sent_count(), 1, "an ARP request, not the echo");
    }

    /// An echo to an off-link peer with no gateway is unreachable, and says
    /// so where the caller asked rather than as a timeout.
    #[test]
    fn an_off_link_echo_without_a_gateway_is_unreachable() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut scratch.0, MY_MAC).expect("scratch");
        let mut s = IpStack::new(net, Identity::new(ME, 24, None).expect("valid"));
        assert_eq!(
            s.echo_request(WAN, 1, 1, b"probe", 0),
            Err(OutputError::Unreachable)
        );
    }

    /// **The B1 witness.** With the gateway resolved, an echo request reaches
    /// the wire as a well-formed IPv4 datagram addressed to the gateway's
    /// hardware address — the whole path from "a peer's address" to "bytes on
    /// the link".
    #[test]
    fn an_echo_request_reaches_the_wire() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        s.arp_mut().insert(GW, GW_MAC, 0);

        s.echo_request(WAN, 0xbeef, 7, b"kore", 0).expect("sends");
        let (sent, et) = s.ifnet().last_sent().expect("on the wire");
        assert_eq!(et, EtherType::IPV4);

        let head = FrameHead::parse(sent).expect("well formed");
        assert_eq!(
            head.dst, GW_MAC,
            "off-link traffic is addressed to the gateway's station address"
        );

        // The datagram validates as one addressed to the gateway's IP peer.
        let body = &sent[HEADER_LEN..];
        let packet = smoltcp::wire::Ipv4Packet::new_checked(body).expect("parses");
        assert!(packet.verify_checksum(), "the checksum we wrote is correct");
        assert_eq!(from_ip(packet.src_addr()), ME);
        assert_eq!(
            from_ip(packet.dst_addr()),
            WAN,
            "addressed to the peer, not the gateway"
        );
    }

    /// **A needs-frag report reaches the caller with its MTU.**
    ///
    /// This is PLPMTUD's whole input, and `res/INET_STACK.md` §9.2 names it
    /// as the one absent item that *crosses* the protosw boundary: the
    /// report arrives at this row and is needed by a transport above, so the
    /// event has to carry it or the hook becomes a seam redraw later.
    #[test]
    fn a_fragmentation_needed_report_carries_the_next_hop_mtu() {
        use smoltcp::phy::ChecksumCapabilities;
        use smoltcp::wire::{
            Icmpv4DstUnreachable as DstUnreachable, Icmpv4Packet, Icmpv4Repr, Ipv4Repr,
        };

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        s.arp_mut().insert(GW, GW_MAC, 0);

        // The offending datagram the router quotes back at us.
        let quoted = Ipv4Repr {
            src_addr: to_ip(ME),
            dst_addr: to_ip(WAN),
            next_header: smoltcp::wire::IpProtocol::Udp,
            payload_len: 8,
            hop_limit: 64,
        };
        let icmp = Icmpv4Repr::DstUnreachable {
            reason: DstUnreachable::FragRequired,
            header: quoted,
            data: &[0u8; 8],
        };
        let mut payload = [0u8; 128];
        let ilen = icmp.buffer_len();
        {
            let mut p = Icmpv4Packet::new_unchecked(&mut payload[..ilen]);
            icmp.emit(&mut p, &ChecksumCapabilities::default());
        }
        // RFC 1191's next-hop MTU lives in bytes 6..8, which the repr above
        // does not model — so it is written here and the checksum redone,
        // exactly as a real router's packet would have both.
        payload[6..8].copy_from_slice(&1400u16.to_be_bytes());
        {
            let mut p = Icmpv4Packet::new_unchecked(&mut payload[..ilen]);
            p.fill_checksum();
        }

        let ip_repr = Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(ME),
            next_header: smoltcp::wire::IpProtocol::Icmp,
            payload_len: ilen,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 192];
        let dlen = ip_repr.buffer_len() + ilen;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
            p.payload_mut()[..ilen].copy_from_slice(&payload[..ilen]);
        }
        let mut f = [0u8; 256];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut seen = None;
        s.input(0, |e| {
            if let Event::Icmp(ev) = e {
                seen = Some(ev);
            }
        });
        assert_eq!(
            seen,
            Some(IcmpEvent::Unreachable {
                from: GW,
                code: crate::netinet::icmp::Unreachable::FragmentationNeeded,
                next_hop_mtu: Some(1400),
            }),
            "the code and the MTU both reach the caller"
        );
    }

    /// Any other unreachable code carries no MTU — the field is read only
    /// where the class defines it, never off whatever happened to be there.
    #[test]
    fn a_port_unreachable_carries_no_mtu() {
        use smoltcp::phy::ChecksumCapabilities;
        use smoltcp::wire::{
            Icmpv4DstUnreachable as DstUnreachable, Icmpv4Packet, Icmpv4Repr, Ipv4Repr,
        };

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        s.arp_mut().insert(GW, GW_MAC, 0);

        let quoted = Ipv4Repr {
            src_addr: to_ip(ME),
            dst_addr: to_ip(WAN),
            next_header: smoltcp::wire::IpProtocol::Udp,
            payload_len: 8,
            hop_limit: 64,
        };
        let icmp = Icmpv4Repr::DstUnreachable {
            reason: DstUnreachable::PortUnreachable,
            header: quoted,
            data: &[0u8; 8],
        };
        let mut payload = [0u8; 128];
        let ilen = icmp.buffer_len();
        {
            let mut p = Icmpv4Packet::new_unchecked(&mut payload[..ilen]);
            icmp.emit(&mut p, &ChecksumCapabilities::default());
        }
        let ip_repr = Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(ME),
            next_header: smoltcp::wire::IpProtocol::Icmp,
            payload_len: ilen,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 192];
        let dlen = ip_repr.buffer_len() + ilen;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
            p.payload_mut()[..ilen].copy_from_slice(&payload[..ilen]);
        }
        let mut f = [0u8; 256];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut seen = None;
        s.input(0, |e| {
            if let Event::Icmp(ev) = e {
                seen = Some(ev);
            }
        });
        assert_eq!(
            seen,
            Some(IcmpEvent::Unreachable {
                from: GW,
                code: crate::netinet::icmp::Unreachable::Port,
                next_hop_mtu: None,
            })
        );
    }

    /// An echo request from a peer is answered, which is what makes this
    /// station pingable.
    #[test]
    fn an_echo_request_from_a_peer_is_answered() {
        use smoltcp::phy::ChecksumCapabilities;
        use smoltcp::wire::{Icmpv4Packet, Icmpv4Repr};

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        s.arp_mut().insert(GW, GW_MAC, 0);

        // Build ICMP echo request, wrapped in IPv4, wrapped in Ethernet.
        let icmp = Icmpv4Repr::EchoRequest {
            ident: 0x1234,
            seq_no: 9,
            data: b"hello",
        };
        let mut payload = [0u8; 64];
        let ilen = icmp.buffer_len();
        {
            let mut p = Icmpv4Packet::new_unchecked(&mut payload[..ilen]);
            icmp.emit(&mut p, &ChecksumCapabilities::default());
        }

        let ip_repr = smoltcp::wire::Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(ME),
            next_header: smoltcp::wire::IpProtocol::Icmp,
            payload_len: ilen,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 128];
        let dlen = ip_repr.buffer_len() + ilen;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
            p.payload_mut()[..ilen].copy_from_slice(&payload[..ilen]);
        }

        let mut f = [0u8; 256];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let before = s.ifnet().sent_count();
        let mut icmp = None;
        s.input(0, |e| {
            if let Event::Icmp(ev) = e {
                icmp = Some(ev);
            }
        });
        assert_eq!(icmp, Some(IcmpEvent::EchoAnswered { from: GW }));
        assert_eq!(s.ifnet().sent_count(), before + 1, "a reply went out");
    }

    /// A datagram for a protocol with no row is named, not silently dropped —
    /// "nothing is listening" and "malformed" are different facts.
    ///
    /// TCP stands in for the unclaimed protocol now that B2 has given UDP a
    /// row; B3 will claim this one and the test moves to whatever is next.
    #[test]
    fn a_datagram_for_an_unclaimed_protocol_is_named() {
        use smoltcp::phy::ChecksumCapabilities;

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let ip_repr = smoltcp::wire::Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(ME),
            next_header: smoltcp::wire::IpProtocol::Tcp,
            payload_len: 8,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 64];
        let dlen = ip_repr.buffer_len() + 8;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
        }
        let mut f = [0u8; 128];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut unclaimed = None;
        s.input(0, |e| {
            if let Event::NoProtocol { proto, from } = e {
                unclaimed = Some((proto, from));
            }
        });
        assert_eq!(
            unclaimed,
            Some((IpProto::TCP, GW)),
            "TCP has no row until B3"
        );
    }

    /// A datagram addressed to someone else is dropped: it reached us because
    /// the link is shared, not because it was ours.
    #[test]
    fn a_datagram_for_another_station_is_dropped() {
        use smoltcp::phy::ChecksumCapabilities;

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let ip_repr = smoltcp::wire::Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(Ipv4Addr([10, 0, 2, 99])),
            next_header: smoltcp::wire::IpProtocol::Icmp,
            payload_len: 8,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 64];
        let dlen = ip_repr.buffer_len() + 8;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
        }
        let mut f = [0u8; 128];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut events = 0;
        s.input(0, |_| events += 1);
        assert_eq!(events, 0, "not addressed here");
    }

    /// A corrupt header is refused rather than acted on.
    #[test]
    fn a_corrupt_datagram_is_refused() {
        use smoltcp::phy::ChecksumCapabilities;

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let ip_repr = smoltcp::wire::Ipv4Repr {
            src_addr: to_ip(GW),
            dst_addr: to_ip(ME),
            next_header: smoltcp::wire::IpProtocol::Icmp,
            payload_len: 8,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 64];
        let dlen = ip_repr.buffer_len() + 8;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip_repr.emit(&mut p, &ChecksumCapabilities::default());
        }
        datagram[12] ^= 0xff; // corrupt the source address, breaking the checksum

        let mut f = [0u8; 128];
        let len = frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut events = 0;
        s.input(0, |_| events += 1);
        assert_eq!(events, 0, "the checksum is verified, not assumed");
    }

    // ── B2: UDP, DHCP, DNS ──────────────────────────────────────────────

    /// Wrap a UDP payload in IPv4 + Ethernet, as a server would put it on the
    /// wire.
    fn udp_frame(
        src: (Ipv4Addr, u16),
        dst: (Ipv4Addr, u16),
        payload: &[u8],
        out: &mut [u8],
    ) -> usize {
        use smoltcp::phy::ChecksumCapabilities;
        use smoltcp::wire::{IpAddress, UdpPacket, UdpRepr};

        let udp = UdpRepr {
            src_port: src.1,
            dst_port: dst.1,
        };
        let udp_len = udp.header_len() + payload.len();
        let ip = smoltcp::wire::Ipv4Repr {
            src_addr: to_ip(src.0),
            dst_addr: to_ip(dst.0),
            next_header: smoltcp::wire::IpProtocol::Udp,
            payload_len: udp_len,
            hop_limit: 64,
        };
        let mut datagram = [0u8; 1024];
        let dlen = ip.buffer_len() + udp_len;
        {
            let mut p = smoltcp::wire::Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
            ip.emit(&mut p, &ChecksumCapabilities::default());
            let mut u = UdpPacket::new_unchecked(&mut p.payload_mut()[..udp_len]);
            udp.emit(
                &mut u,
                &IpAddress::Ipv4(ip.src_addr),
                &IpAddress::Ipv4(ip.dst_addr),
                payload.len(),
                |buf| buf.copy_from_slice(payload),
                &ChecksumCapabilities::default(),
            );
        }
        frame(MY_MAC, GW_MAC, EtherType::IPV4, &datagram[..dlen], out)
    }

    /// A UDP datagram to a bound port is delivered; one to an unbound port is
    /// not — the demultiplex is the whole of what this layer adds.
    #[test]
    fn udp_delivers_only_to_a_bound_port() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);

        let mut f = [0u8; 256];
        let len = udp_frame((GW, 5353), (ME, 9000), b"hello", &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");
        let mut seen = None;
        s.input(0, |e| seen = delivered(e));
        assert_eq!(seen, None, "nothing is bound to 9000");

        assert!(s.bind_udp(9000));
        let len = udp_frame((GW, 5353), (ME, 9000), b"hello", &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");
        let mut seen = None;
        s.input(0, |e| seen = delivered(e));
        assert_eq!(
            seen,
            Some(((GW, 5353), 9000, 5)),
            "the bound port is delivered, with its bytes"
        );
    }

    /// **The DHCP exchange, end to end.** DISCOVER goes out, an OFFER comes
    /// back, a REQUEST goes out, an ACK binds the lease — and the lease
    /// carries the address, the prefix, the gateway and the resolver.
    #[test]
    fn dhcp_discovers_offers_requests_and_binds() {
        use smoltcp::wire::{
            DHCP_CLIENT_PORT, DHCP_SERVER_PORT, DhcpMessageType, DhcpPacket, DhcpRepr,
        };

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        // A station with no identity yet is the honest starting state.
        let net = MockIfnet::new(&mut scratch.0, MY_MAC).expect("scratch");
        let mut s = IpStack::new(
            net,
            Identity::new(Ipv4Addr([0, 0, 0, 1]), 0, None).expect("placeholder"),
        );

        // 1. DISCOVER
        assert!(s.drive_dhcp(0, 0xabcd1234));
        assert_eq!(s.dhcp().state(), DhcpState::Discovering);
        let (sent, _) = s.ifnet().last_sent().expect("a DISCOVER went out");
        let head = FrameHead::parse(sent).expect("well formed");
        assert_eq!(
            head.dst,
            MacAddr::BROADCAST,
            "sent before we have an address"
        );

        // The server's reply, built the way a server builds one.
        let server = Ipv4Addr([10, 0, 2, 2]);
        let offered = Ipv4Addr([10, 0, 2, 15]);
        let mut reply = |kind: DhcpMessageType, xid: u32, out: &mut [u8]| -> usize {
            let repr = DhcpRepr {
                message_type: kind,
                transaction_id: xid,
                secs: 0,
                client_hardware_address: smoltcp::wire::EthernetAddress(MY_MAC.0),
                client_ip: to_ip(Ipv4Addr::UNSPECIFIED),
                your_ip: to_ip(offered),
                server_ip: to_ip(server),
                router: Some(to_ip(server)),
                subnet_mask: Some(to_ip(Ipv4Addr([255, 255, 255, 0]))),
                relay_agent_ip: to_ip(Ipv4Addr::UNSPECIFIED),
                broadcast: false,
                requested_ip: None,
                client_identifier: None,
                server_identifier: Some(to_ip(server)),
                parameter_request_list: None,
                dns_servers: Some(
                    heapless::Vec::from_slice(&[to_ip(Ipv4Addr([1, 1, 1, 1]))]).unwrap(),
                ),
                max_size: None,
                lease_duration: Some(3600),
                renew_duration: None,
                rebind_duration: None,
                additional_options: &[],
            };
            let n = repr.buffer_len();
            let mut buf = [0u8; 512];
            {
                let mut p = DhcpPacket::new_unchecked(&mut buf[..n]);
                repr.emit(&mut p).expect("emits");
            }
            udp_frame(
                (server, DHCP_SERVER_PORT),
                (Ipv4Addr::BROADCAST, DHCP_CLIENT_PORT),
                &buf[..n],
                out,
            )
        };

        // 2. OFFER → the client moves to Requesting
        let mut f = [0u8; 1024];
        let len = reply(DhcpMessageType::Offer, 0xabcd1234, &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");
        s.input(0, |_| {});
        assert!(matches!(s.dhcp().state(), DhcpState::Requesting { .. }));

        // 3. REQUEST goes out on the next sweep
        assert!(s.drive_dhcp(1, 0));

        // 4. ACK → bound
        let len = reply(DhcpMessageType::Ack, 0xabcd1234, &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");
        let mut leased = None;
        s.input(1, |e| {
            if let Event::Leased(l) = e {
                leased = Some(l);
            }
        });

        let lease = leased.expect("the ACK bound a lease");
        assert_eq!(lease.addr, offered);
        assert_eq!(lease.prefix_len, 24, "derived from the offered mask");
        assert_eq!(lease.gateway, Some(server));
        assert_eq!(lease.dns, Some(Ipv4Addr([1, 1, 1, 1])));
        assert_eq!(s.dhcp().lease(), Some(lease));
    }

    /// **A neighbour's exchange is not ours.** On a shared segment every
    /// client sees every broadcast; acting on another station's ACK would
    /// take its address.
    #[test]
    fn a_dhcp_reply_for_another_transaction_is_ignored() {
        use smoltcp::wire::{
            DHCP_CLIENT_PORT, DHCP_SERVER_PORT, DhcpMessageType, DhcpPacket, DhcpRepr,
        };

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut scratch.0, MY_MAC).expect("scratch");
        let mut s = IpStack::new(
            net,
            Identity::new(Ipv4Addr([0, 0, 0, 1]), 0, None).expect("placeholder"),
        );
        s.drive_dhcp(0, 0xabcd1234);

        let repr = DhcpRepr {
            message_type: DhcpMessageType::Offer,
            transaction_id: 0x99999999, // somebody else's exchange
            secs: 0,
            client_hardware_address: smoltcp::wire::EthernetAddress(MY_MAC.0),
            client_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            your_ip: to_ip(Ipv4Addr([10, 0, 2, 77])),
            server_ip: to_ip(GW),
            router: None,
            subnet_mask: None,
            relay_agent_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            broadcast: false,
            requested_ip: None,
            client_identifier: None,
            server_identifier: Some(to_ip(GW)),
            parameter_request_list: None,
            dns_servers: None,
            max_size: None,
            lease_duration: Some(3600),
            renew_duration: None,
            rebind_duration: None,
            additional_options: &[],
        };
        let n = repr.buffer_len();
        let mut buf = [0u8; 512];
        {
            let mut p = DhcpPacket::new_unchecked(&mut buf[..n]);
            repr.emit(&mut p).expect("emits");
        }
        let mut f = [0u8; 1024];
        let len = udp_frame(
            (GW, DHCP_SERVER_PORT),
            (Ipv4Addr::BROADCAST, DHCP_CLIENT_PORT),
            &buf[..n],
            &mut f,
        );
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");
        s.input(0, |_| {});

        assert_eq!(
            s.dhcp().state(),
            DhcpState::Discovering,
            "still discovering — a neighbour's offer moved nothing"
        );
    }

    /// **The DNS round trip.** A query goes out for a name, the answer comes
    /// back, and the address is reported — the last step before `ping`.
    #[test]
    fn a_name_resolves() {
        use smoltcp::wire::{DnsFlags, DnsPacket, DnsQueryType, DnsQuestion, DnsRepr};

        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let resolver = Ipv4Addr([1, 1, 1, 1]);
        let net = MockIfnet::new(&mut scratch.0, MY_MAC).expect("scratch");
        let mut id = Identity::new(ME, 24, Some(GW)).expect("valid");
        id.set_dns(Some(resolver));
        let mut s = IpStack::new(net, id);
        s.arp_mut().insert(GW, GW_MAC, 0);

        let xid = 0x4242;
        assert!(
            s.resolve_name("google.com", xid, 50000, 0).expect("sends"),
            "a query went out"
        );
        assert_eq!(s.dns().state(), dns::DnsState::Asking { xid });

        // The resolver's answer. smoltcp's DnsRepr carries one question and
        // the answers follow it in the payload, so the response is assembled
        // by hand the way a server would.
        let name = b"\x06google\x03com\x00";
        let question = DnsQuestion {
            name,
            type_: DnsQueryType::A,
        };
        let repr = DnsRepr {
            transaction_id: xid,
            flags: DnsFlags::RESPONSE | DnsFlags::RECURSION_DESIRED,
            opcode: smoltcp::wire::DnsOpcode::Query,
            question,
        };
        let hdr = repr.buffer_len();
        let mut buf = [0u8; 512];
        {
            let mut p = DnsPacket::new_unchecked(&mut buf[..hdr]);
            repr.emit(&mut p);
            p.set_answer_record_count(1);
        }
        // One A record appended after the echoed question.
        let answer: [u8; 16] = [
            0xc0, 0x0c, // a pointer to the question's name
            0x00, 0x01, // type A
            0x00, 0x01, // class IN
            0x00, 0x00, 0x00, 0x3c, // TTL
            0x00, 0x04, // rdlength
            142, 250, 72, 46, // the address
        ];
        buf[hdr..hdr + answer.len()].copy_from_slice(&answer);
        let total = hdr + answer.len();

        let mut f = [0u8; 1024];
        let len = udp_frame((resolver, 53), (ME, 50000), &buf[..total], &mut f);
        s.ifnet_mut()
            .inject(&f[..len], EtherType::IPV4)
            .expect("injects");

        let mut resolved = None;
        s.input(0, |e| {
            if let Event::Resolved { addr } = e {
                resolved = Some(addr);
            }
        });
        assert_eq!(
            resolved,
            Some(Ipv4Addr([142, 250, 72, 46])),
            "the name resolved"
        );
        assert_eq!(s.dns().answer(), Some(Ipv4Addr([142, 250, 72, 46])));
    }

    // ── the resolver's four checks ───────────────────────────────────────

    /// The resolver this test's queries go to, and the port they go from.
    const DNS_SERVER: Ipv4Addr = Ipv4Addr([10, 0, 2, 3]);
    const DNS_FROM: u16 = 40_000;
    const WANT: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

    /// Build a DNS response: header, one echoed question, one A answer.
    ///
    /// Written out by hand rather than round-tripped through the encoder,
    /// because the point of these tests is what an *attacker* can put on the
    /// wire, and an attacker is not restricted to what our encoder emits.
    /// A response buffer and how much of it is the message.
    ///
    /// Fixed-size because this crate is heapless: there is no `alloc` here,
    /// and a test that needed one would be testing a different crate.
    struct Response {
        bytes: [u8; 128],
        len: usize,
    }

    impl Response {
        fn as_slice(&self) -> &[u8] {
            &self.bytes[..self.len]
        }
        fn put(&mut self, src: &[u8]) {
            self.bytes[self.len..self.len + src.len()].copy_from_slice(src);
            self.len += src.len();
        }
    }

    /// Build a DNS response: header, one echoed question, one A answer.
    ///
    /// Written out by hand rather than round-tripped through the encoder,
    /// because the point of these tests is what an *attacker* can put on the
    /// wire, and an attacker is not restricted to what our encoder emits.
    fn dns_response(xid: u16, name: &str, addr: Ipv4Addr) -> Response {
        let mut wire = [0u8; 64];
        let mut at = 0usize;
        for label in name.split('.') {
            wire[at] = label.len() as u8;
            wire[at + 1..at + 1 + label.len()].copy_from_slice(label.as_bytes());
            at += 1 + label.len();
        }
        wire[at] = 0;
        at += 1;
        let wire = &wire[..at];

        let mut r = Response {
            bytes: [0u8; 128],
            len: 0,
        };
        r.put(&xid.to_be_bytes());
        r.put(&0x8180u16.to_be_bytes()); // RESPONSE | RD | RA
        r.put(&1u16.to_be_bytes()); // qdcount
        r.put(&1u16.to_be_bytes()); // ancount
        r.put(&0u16.to_be_bytes()); // nscount
        r.put(&0u16.to_be_bytes()); // arcount
        r.put(wire); // question name
        r.put(&1u16.to_be_bytes()); // A
        r.put(&1u16.to_be_bytes()); // IN
        r.put(wire); // answer name, spelled out rather than a pointer
        r.put(&1u16.to_be_bytes()); // A
        r.put(&1u16.to_be_bytes()); // IN
        r.put(&60u32.to_be_bytes()); // ttl
        r.put(&4u16.to_be_bytes()); // rdlength
        r.put(&addr.0);
        r
    }

    fn asking() -> dns::Resolver {
        let mut r = dns::Resolver::new();
        r.sent(0x1111, DNS_SERVER, DNS_FROM, "google.com", 0);
        r
    }

    /// The happy path, so the three refusals below are refusing something
    /// that would otherwise have been accepted.
    #[test]
    fn the_resolver_accepts_its_own_answer() {
        let mut r = asking();
        let msg = dns_response(0x1111, "google.com", WANT);
        assert_eq!(
            r.input((DNS_SERVER, dns::DNS_PORT), DNS_FROM, msg.as_slice()),
            Some(WANT)
        );
    }

    /// An answer to a question we did not ask is ignored — otherwise anything
    /// that can reach this port decides what a name means.
    #[test]
    fn a_dns_answer_with_the_wrong_transaction_is_ignored() {
        let mut r = asking();
        let msg = dns_response(0x2222, "google.com", WANT);
        assert_eq!(r.input((DNS_SERVER, dns::DNS_PORT), DNS_FROM, msg.as_slice()), None);
        assert_eq!(r.state(), dns::DnsState::Asking { xid: 0x1111 });
    }

    /// **The identifier alone is 16 bits.** An off-path station that guesses
    /// it still has to be the resolver we asked — and is not.
    #[test]
    fn an_answer_from_another_station_is_ignored() {
        let mut r = asking();
        let msg = dns_response(0x1111, "google.com", Ipv4Addr([6, 6, 6, 6]));
        let forger = Ipv4Addr([203, 0, 113, 9]);
        assert_eq!(r.input((forger, dns::DNS_PORT), DNS_FROM, msg.as_slice()), None);
        assert_eq!(r.answer(), None, "nothing cached from a stranger");
    }

    /// It also has to have come from the DNS port, and landed on the port the
    /// query went out from.
    #[test]
    fn an_answer_on_the_wrong_port_is_ignored() {
        let mut r = asking();
        let msg = dns_response(0x1111, "google.com", WANT);
        assert_eq!(
            r.input((DNS_SERVER, 1234), DNS_FROM, msg.as_slice()),
            None,
            "not from the resolver's port"
        );
        assert_eq!(
            r.input((DNS_SERVER, dns::DNS_PORT), DNS_FROM + 1, msg.as_slice()),
            None,
            "not to the port we asked from"
        );
    }

    /// And it has to answer the question that was actually asked: a response
    /// carrying somebody else's question under our identifier is not ours.
    #[test]
    fn an_answer_to_a_different_question_is_ignored() {
        let mut r = asking();
        let msg = dns_response(0x1111, "evil.example", WANT);
        assert_eq!(r.input((DNS_SERVER, dns::DNS_PORT), DNS_FROM, msg.as_slice()), None);
        assert_eq!(r.answer(), None);
    }

    /// A malformed name is refused rather than truncated — a truncated name
    /// is a different name, and would resolve to somebody else.
    #[test]
    fn a_malformed_name_is_refused() {
        let mut out = [0u8; 512];
        assert_eq!(dns::encode_query("", 1, &mut out), None);
        assert_eq!(
            dns::encode_query("a..b", 1, &mut out),
            None,
            "an empty label"
        );
        let long = "a".repeat(64);
        assert_eq!(
            dns::encode_query(&long, 1, &mut out),
            None,
            "a label over 63"
        );
        assert!(dns::encode_query("google.com", 1, &mut out).is_some());
    }

    /// Adopting a new identity forgets the old segment's translations — they
    /// say nothing about the segment the new address is on.
    #[test]
    fn a_new_identity_forgets_the_old_translations() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut s = stack(&mut scratch.0);
        s.arp_mut().insert(GW, GW_MAC, 0);
        assert_eq!(s.arp().len(), 1);

        s.set_identity(Identity::new(Ipv4Addr([192, 168, 1, 5]), 24, None).expect("valid"));
        assert!(s.arp().is_empty(), "a new lease starts with no neighbours");
    }
}
