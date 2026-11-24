//! **Tier 2 over Tier 1** — the socket face, proved over `MockIfnet`.
//!
//! `res/INET_STACK.md` §6.2 says Tier 2 is "written in terms of Tier 1 **with
//! no privileged access**, which is what proves Tier 1 sufficient and lets a
//! POSIX shim be written by someone who is not us". That is a claim about an
//! *absence*, and an absence is only provable by writing the layer and seeing
//! what it had to reach for. This file is that proof, in three witnesses:
//!
//! 1. **A UDP round trip** between two stations on one shuttled link — ARP
//!    included, because the first datagram to anyone new always waits on a
//!    translation.
//! 2. **A TCP connect, echo and close**, FIN both ways, over the borrowed
//!    state machine driven through kore's own seam.
//! 3. **A socket derived from flow A cannot touch flow B's traffic** — in
//!    both directions, and through the *shipping* predicates rather than
//!    through test logic that agrees with them.
//!
//! # What drives the wire
//!
//! Nothing in `crates/inet` owns an executor, and Tier 1 offers no readiness
//! signal, so these tests are the composition that supplies both: a clock
//! that advances by hand, a [`Link`] that carries each committed frame to the
//! other station, and [`step`] — one poll of one future. Polling once and
//! carrying once means no frame is ever overwritten in the double's single
//! slot, so a failure here is a protocol failure and never a lost packet.

use core::future::Future;

use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};
use kore_driver::net::{EtherType, FRAME_MAX, FrameFilter, FrameHead, MacAddr, VifId};
use kore_inet::flow::{Powers, Scope};
use kore_inet::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
use kore_inet::netinet::{TcpState, TcpStorage};
use kore_inet::row::{Identity, InetRow, OpenGrant};
use kore_inet::smol::SMOL_SCRATCH_BYTES;
use kore_inet::{
    FlowCap, LocalFlow, RecordingInstall, SocketRefusal, TcpListener, TcpStream, UdpSocket,
};
use smoltcp::iface::SocketStorage;

const A_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
const B_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
const GW_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x0f]);
/// Station A — the client.
const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
/// Station B — the server.
const SRV: Ipv4Addr = Ipv4Addr([10, 0, 2, 50]);
const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
const PEER_A: Ipv4Addr = Ipv4Addr([93, 184, 216, 34]);
const PEER_B: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

/// Turns one phase may take before it is a failure rather than slow progress.
///
/// Bounded on purpose: a test that spun until a protocol worked would hang
/// rather than fail, and a hang reports nothing.
const TURNS: usize = 400;

/// Milliseconds one turn advances the clock.
///
/// Larger than one so the timers smoltcp actually uses — delayed ACK, the
/// retransmit backoff — are reachable within the turn budget instead of being
/// quietly excluded from what these witnesses cover.
const TICK_MS: u64 = 5;

/// A window must be 64-byte aligned, which a bare stack array is not.
#[repr(align(64))]
struct Aligned<const N: usize>([u8; N]);

/// One poll of one future.
///
/// `None` means it was not ready, and the caller comes back next turn after
/// the wire has moved. Every future in `kore_inet::socket` but `write_all`
/// re-reads the socket on each poll and so survives being dropped here; that
/// one completes in a single poll for a write that fits the window, which
/// every write below does.
fn step<T>(f: impl Future<Output = T>) -> Option<T> {
    kore_rt::try_block_on(f, 1)
}

/// The shared segment: whatever one station commits, the other receives.
///
/// Deliberately not a queue. One frame each way per turn is exactly what the
/// double holds, and carrying at most one keeps the test honest about never
/// having silently dropped anything.
struct Link {
    from_a: usize,
    from_b: usize,
}

impl Link {
    fn new() -> Self {
        Self {
            from_a: 0,
            from_b: 0,
        }
    }

    fn carry(from: &MockIfnet<'_>, to: &mut MockIfnet<'_>, seen: &mut usize) {
        if from.sent_count() == *seen {
            return;
        }
        *seen = from.sent_count();
        let Some((bytes, ethertype)) = from.last_sent() else {
            return;
        };
        let mut staged = [0u8; FRAME_MAX];
        let n = bytes.len().min(staged.len());
        staged[..n].copy_from_slice(&bytes[..n]);
        to.inject(&staged[..n], ethertype).expect("the double injects");
    }

    fn pump(&mut self, a: &mut MockIfnet<'_>, b: &mut MockIfnet<'_>) {
        Self::carry(a, b, &mut self.from_a);
        Self::carry(b, a, &mut self.from_b);
    }
}

fn identity(addr: Ipv4Addr, gateway: Option<Ipv4Addr>) -> Identity {
    Identity::new(addr, 24, gateway).expect("valid")
}

/// Mint an S1 authority directly, for the two witnesses whose subject is the
/// socket rather than the mint.
///
/// The isolation witness below does *not* use this: it goes through
/// [`InetRow`], because what it proves is that a socket sits on an authority
/// the row minted and installed, not on one a test wrote down.
fn cap(index: u16, scope: Scope) -> FlowCap {
    FlowCap::mint(
        VifId {
            index,
            generation: 1,
        },
        scope,
        Powers::NONE,
    )
    .expect("S1 mints")
}

// ── 1. a UDP round trip ─────────────────────────────────────────────────

/// **The datagram witness.** Two stations, two capability-scoped sockets, one
/// link — and a payload goes out and an answer comes back.
///
/// The ARP exchange is part of what is proved rather than set up around:
/// station A has never spoken to station B, so the first `send` cannot
/// succeed until the translation arrives, and the socket awaiting it is the
/// thing that drains the reply. A socket whose send path did not also drain
/// input would deadlock here, which is why the ordering is asserted by
/// construction instead of described in a comment.
#[test]
fn a_udp_datagram_makes_a_round_trip_between_two_stations() {
    let mut scratch_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut scratch_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);

    let mut a = UdpSocket::open(LocalFlow::new(
        MockIfnet::new(&mut scratch_a.0, A_MAC).expect("scratch"),
        cap(
            0,
            Scope::Flow(FlowMatch::connection(IpProto::UDP, (ME, 5000), (SRV, 5001))),
        ),
        identity(ME, None),
    ))
    .expect("A's socket");
    let mut b = UdpSocket::open(LocalFlow::new(
        MockIfnet::new(&mut scratch_b.0, B_MAC).expect("scratch"),
        cap(
            1,
            Scope::Flow(FlowMatch::connection(IpProto::UDP, (SRV, 5001), (ME, 5000))),
        ),
        identity(SRV, None),
    ))
    .expect("B's socket");

    assert_eq!(a.local_addr(), (ME, 5000), "the endpoint is the authority's");
    assert_eq!(a.peer_addr(), Some((SRV, 5001)));

    let mut link = Link::new();
    let mut heard = [0u8; 32];
    let mut sent = false;
    let mut received = None;

    for turn in 0..TURNS {
        let now = turn as u64 * TICK_MS;
        a.flow_mut().set_now_ms(now);
        b.flow_mut().set_now_ms(now);

        if !sent
            && let Some(result) = step(a.send(b"ping"))
        {
            assert_eq!(result.expect("A sent"), 4);
            sent = true;
        }
        if received.is_none()
            && let Some(result) = step(b.recv_from(&mut heard))
        {
            received = Some(result.expect("B received"));
        }
        link.pump(a.flow_mut().ifnet_mut(), b.flow_mut().ifnet_mut());

        if received.is_some() {
            break;
        }
    }

    let (n, from) = received.expect("the datagram reached B within the turn budget");
    assert_eq!(&heard[..n], b"ping");
    assert_eq!(from, (ME, 5000), "and it says who sent it");

    // ── the answer, so neither socket is merely inert ───────────────────
    let mut answered = false;
    let mut back = None;
    for turn in 0..TURNS {
        let now = (TURNS + turn) as u64 * TICK_MS;
        a.flow_mut().set_now_ms(now);
        b.flow_mut().set_now_ms(now);

        if !answered
            && let Some(result) = step(b.send(b"pong"))
        {
            result.expect("B answered");
            answered = true;
        }
        if back.is_none()
            && let Some(result) = step(a.recv_from(&mut heard))
        {
            back = Some(result.expect("A received"));
        }
        link.pump(a.flow_mut().ifnet_mut(), b.flow_mut().ifnet_mut());

        if back.is_some() {
            break;
        }
    }

    let (n, from) = back.expect("the answer reached A within the turn budget");
    assert_eq!(&heard[..n], b"pong");
    assert_eq!(from, (SRV, 5001));
}

// ── 2. a TCP connection ─────────────────────────────────────────────────

/// Declare every buffer one stream borrows as a local, then build the socket
/// over them.
///
/// They have to be locals rather than fields of a fixture: `TcpStorage<'a>`
/// ties the socket slice and its contents to one lifetime, so a struct
/// holding them would force `'static` and nothing in a test is static. The
/// same reason `netinet::tcp`'s own tests use a macro here.
macro_rules! storage {
    ($rx:ident, $tx:ident, $sockets:ident, $scratch:ident) => {
        let mut $scratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let mut $rx = [0u8; 2048];
        let mut $tx = [0u8; 2048];
        let mut $sockets = [SocketStorage::EMPTY; 1];
    };
}

/// **The stream witness.** A connection is established, a payload is echoed,
/// and both ends close — over the borrowed TCP state machine, driven entirely
/// through Tier 2.
///
/// Both stations are on one prefix, so there is no gateway and the ARP each
/// smoltcp interface does is for the other station directly. That is the
/// shape a LAN connection actually has, and it keeps the witness about TCP
/// rather than about routing.
#[test]
fn a_tcp_connection_handshakes_echoes_and_closes_both_ways() {
    let mut if_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut if_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    storage!(rx_a, tx_a, sock_a, smol_a);
    storage!(rx_b, tx_b, sock_b, smol_b);

    let mut client = TcpStream::open(
        LocalFlow::new(
            MockIfnet::new(&mut if_a.0, A_MAC).expect("scratch"),
            cap(
                0,
                Scope::Flow(FlowMatch::connection(IpProto::TCP, (ME, 49152), (SRV, 80))),
            ),
            identity(ME, None),
        ),
        TcpStorage {
            rx: &mut rx_a,
            tx: &mut tx_a[..256],
            sockets: &mut sock_a,
            scratch: &mut smol_a.0,
        },
        (SRV, 80),
    )
    .expect("the client opens");

    let mut server = TcpListener::bind(
        LocalFlow::new(
            MockIfnet::new(&mut if_b.0, B_MAC).expect("scratch"),
            cap(
                1,
                Scope::Flow(FlowMatch::listener(IpProto::TCP, (SRV, 80))),
            ),
            identity(SRV, None),
        ),
        TcpStorage {
            rx: &mut rx_b,
            tx: &mut tx_b,
            sockets: &mut sock_b,
            scratch: &mut smol_b.0,
        },
    )
    .expect("the server listens");

    assert_eq!(server.local_addr(), (SRV, 80));
    let mut link = Link::new();
    let mut clock = 0u64;

    // ── the handshake ───────────────────────────────────────────────────
    let mut established = false;
    let mut accepted = None;
    for _ in 0..TURNS {
        clock += TICK_MS;
        client.flow_mut().set_now_ms(clock);
        server.flow_mut().set_now_ms(clock);

        if !established
            && let Some(result) = step(client.ready())
        {
            result.expect("the handshake completed");
            established = true;
        }
        if accepted.is_none()
            && let Some(result) = step(server.accept())
        {
            accepted = Some(result.expect("the server accepted"));
        }
        link.pump(client.flow_mut().ifnet_mut(), server.flow_mut().ifnet_mut());

        if established && accepted.is_some() {
            break;
        }
    }
    assert!(established, "the client never established");
    assert_eq!(
        accepted,
        Some((ME, 49152)),
        "the server learned who connected"
    );
    assert_eq!(client.state(), TcpState::Open);

    // ── the echo ────────────────────────────────────────────────────────
    const MESSAGE: &[u8] = b"kore";
    let mut at_server = [0u8; 32];
    let mut at_client = [0u8; 32];
    let mut wrote = false;
    let mut server_read = None;
    let mut echoed = false;
    let mut client_read = None;

    for _ in 0..TURNS {
        clock += TICK_MS;
        client.flow_mut().set_now_ms(clock);
        server.flow_mut().set_now_ms(clock);

        if !wrote
            && let Some(result) = step(client.write_all(MESSAGE))
        {
            result.expect("the client wrote");
            wrote = true;
        }
        if wrote
            && server_read.is_none()
            && let Some(result) = step(server.stream().read(&mut at_server))
        {
            server_read = Some(result.expect("the server read"));
        }
        if server_read.is_some()
            && !echoed
            && let Some(result) = step(server.stream().write_all(&at_server[..MESSAGE.len()]))
        {
            result.expect("the server echoed");
            echoed = true;
        }
        if echoed
            && client_read.is_none()
            && let Some(result) = step(client.read(&mut at_client))
        {
            client_read = Some(result.expect("the client read"));
        }
        link.pump(client.flow_mut().ifnet_mut(), server.flow_mut().ifnet_mut());

        if client_read.is_some() {
            break;
        }
    }

    assert_eq!(server_read, Some(MESSAGE.len()), "the server saw the bytes");
    assert_eq!(&at_server[..MESSAGE.len()], MESSAGE);
    assert_eq!(client_read, Some(MESSAGE.len()), "the echo came back");
    assert_eq!(&at_client[..MESSAGE.len()], MESSAGE);

    // A send queues one prefix, and a FIN follows the accepted bytes.
    assert_eq!(step(client.read(&mut [])), Some(Ok(0)));
    let bulk = [0x5a; 4096];
    assert_eq!(step(client.write(&bulk)), Some(Ok(256)));
    link.pump(client.flow_mut().ifnet_mut(), server.flow_mut().ifnet_mut());
    let mut received = [0u8; 256];
    let mut read = 0;
    let mut client_finished = false;
    let mut server_eof = false;
    for _ in 0..TURNS {
        clock += TICK_MS;
        client.flow_mut().set_now_ms(clock);
        server.flow_mut().set_now_ms(clock);

        if !client_finished && let Some(result) = step(client.finish_write()) {
            result.expect("the client FIN was acknowledged");
            client_finished = true;
        }
        let mut chunk = [0u8; 31];
        if !server_eof && let Some(result) = step(server.stream().read(&mut chunk)) {
            let n = result.expect("the server drained data before EOF");
            received[read..read + n].copy_from_slice(&chunk[..n]);
            read += n;
            server_eof = n == 0;
        } else {
            server.sweep();
        }
        link.pump(client.flow_mut().ifnet_mut(), server.flow_mut().ifnet_mut());
        if client_finished && server_eof {
            break;
        }
    }
    assert!(client_finished && server_eof);
    assert_eq!(read, received.len());
    assert_eq!(received, bulk[..read]);

    // ── the close, FIN both ways ────────────────────────────────────────
    let mut client_closed = false;
    let mut server_closed = false;
    for _ in 0..TURNS {
        clock += TICK_MS;
        client.flow_mut().set_now_ms(clock);
        server.flow_mut().set_now_ms(clock);

        if !client_closed
            && let Some(result) = step(client.close())
        {
            result.expect("the client closed");
            client_closed = true;
        }
        if !server_closed
            && let Some(result) = step(server.stream().close())
        {
            result.expect("the server closed");
            server_closed = true;
        }
        link.pump(client.flow_mut().ifnet_mut(), server.flow_mut().ifnet_mut());

        if client_closed && server_closed {
            break;
        }
    }
    assert!(client_closed, "the client never saw the peer's FIN");
    assert!(server_closed, "the server never saw the peer's FIN");
    assert!(
        matches!(
            client.state(),
            TcpState::Closing | TcpState::Closed
        ),
        "a closed stream is past Open"
    );
}

// ── 3. the isolation witness ────────────────────────────────────────────

/// Wrap a UDP payload in IPv4 + Ethernet, as a peer would put it on the wire.
fn udp_frame(src: (Ipv4Addr, u16), dst: (Ipv4Addr, u16), payload: &[u8], out: &mut [u8]) -> usize {
    use smoltcp::phy::ChecksumCapabilities;
    use smoltcp::wire::{IpAddress, Ipv4Address, Ipv4Packet, Ipv4Repr, UdpPacket, UdpRepr};

    let to_ip = |a: Ipv4Addr| Ipv4Address::from_octets(a.0);
    let udp = UdpRepr {
        src_port: src.1,
        dst_port: dst.1,
    };
    let udp_len = udp.header_len() + payload.len();
    let ip = Ipv4Repr {
        src_addr: to_ip(src.0),
        dst_addr: to_ip(dst.0),
        next_header: smoltcp::wire::IpProtocol::Udp,
        payload_len: udp_len,
        hop_limit: 64,
    };
    let mut datagram = [0u8; 512];
    let dlen = ip.buffer_len() + udp_len;
    {
        let mut p = Ipv4Packet::new_unchecked(&mut datagram[..dlen]);
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
    FrameHead {
        dst: A_MAC,
        src: GW_MAC,
        ethertype: EtherType::IPV4,
    }
    .write(out)
    .expect("header fits");
    out[14..14 + dlen].copy_from_slice(&datagram[..dlen]);
    14 + dlen
}

/// Offer a frame to a vif through the **shipping** classifier predicate.
///
/// `FrameFilter::accepts_frame` rather than a reimplementation of the
/// decision, so what passes here is what passes in `services/net`'s
/// classifier. Returns whether it was delivered.
fn offer(net: &mut MockIfnet<'_>, filter: &FrameFilter, frame: &[u8], station: MacAddr) -> bool {
    let head = FrameHead::parse(frame).expect("well formed");
    if !filter.accepts_frame(&head, &frame[14..], station) {
        return false;
    }
    net.inject(frame, head.ethertype).expect("injects");
    true
}

/// Drive one of the row's futures to its answer.
///
/// Bounded rather than spun: the install double never parks, so one poll is
/// always enough, and an exhaustion here would mean a mint had grown a
/// dependency a host test cannot satisfy.
fn run<T>(f: impl Future<Output = T>) -> T {
    kore_rt::try_block_on(f, 4).expect("the row's futures settle without an executor")
}

/// **The authority witness, at Tier 2.** A socket is a view of one
/// capability, and it can reach neither way outside it.
///
/// This is `inet-flow-isolation` asked again one tier up, and the answer has
/// to come from the same two mechanisms rather than from anything the socket
/// invented:
///
/// - **Receive** is closed by the [`FrameFilter`] the row installed when it
///   minted the authority. The socket does no filtering of its own and must
///   not: a check in Tier 2 that the classifier did not also make would be a
///   boundary a buggy client could step over.
/// - **Transmit** is closed by the socket refusing, by name, to address a
///   peer the authority does not cover. That is not the security boundary —
///   `VifPolicy` one layer down is — it is the socket declining to use
///   authority it does not hold, at the call site where a caller can read it.
///
/// Both capabilities are minted by one [`InetRow`] through the same seam a
/// founded row uses, so what is proved is what ships.
#[test]
fn a_socket_on_flow_a_cannot_touch_flow_bs_traffic() {
    let mut row = InetRow::new();
    row.set_identity(identity(ME, Some(GW)));

    let vif_a = VifId {
        index: 0,
        generation: 1,
    };
    let vif_b = VifId {
        index: 1,
        generation: 1,
    };
    let mut install = RecordingInstall::new();
    let cap_a = run(row.open_flow(
        vif_a,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5000),
        (PEER_A, 53),
        &mut install,
    ))
    .expect("A's flow");
    let cap_b = run(row.open_flow(
        vif_b,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5001),
        (PEER_B, 53),
        &mut install,
    ))
    .expect("B's flow");

    // Read back from **what the row installed**, not from the capabilities
    // the test happens to hold: if the mint had handed back an authority it
    // never installed, these lookups would be `None` and nothing below would
    // be enforcing anything.
    let filter_a = install.filter(vif_a).expect("A's enforcement is installed");
    let filter_b = install.filter(vif_b).expect("B's enforcement is installed");

    let mut scratch_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut a = UdpSocket::open(LocalFlow::new(
        MockIfnet::new(&mut scratch_a.0, A_MAC).expect("scratch"),
        cap_a,
        identity(ME, Some(GW)),
    ))
    .expect("A's socket");

    // ── receive: B's datagram never reaches A's interface ───────────────
    let mut frame = [0u8; 512];
    let len = udp_frame((PEER_B, 53), (ME, 5001), b"for-b", &mut frame);
    assert!(
        !offer(a.flow_mut().ifnet_mut(), &filter_a, &frame[..len], A_MAC),
        "A's enforcement rejected B's datagram — same station address, different flow"
    );
    let mut scratch_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut net_b = MockIfnet::new(&mut scratch_b.0, A_MAC).expect("scratch");
    assert!(
        offer(&mut net_b, &filter_b, &frame[..len], A_MAC),
        "and it is B's, not merely unroutable — B's own enforcement takes it"
    );

    let mut heard = [0u8; 32];
    assert!(
        step(a.recv_from(&mut heard)).is_none(),
        "A's socket has nothing to read"
    );

    // ── and A's own datagram does reach it, so A is not merely deaf ─────
    let len = udp_frame((PEER_A, 53), (ME, 5000), b"for-a", &mut frame);
    assert!(offer(
        a.flow_mut().ifnet_mut(),
        &filter_a,
        &frame[..len],
        A_MAC
    ));
    let (n, from) = step(a.recv_from(&mut heard))
        .expect("A's own flow is delivered")
        .expect("and it is a datagram");
    assert_eq!(&heard[..n], b"for-a");
    assert_eq!(from, (PEER_A, 53));

    // ── transmit: A's socket will not address B's peer ──────────────────
    assert_eq!(
        step(a.send_to(b"stray", (PEER_B, 53))),
        Some(Err(SocketRefusal::OutsideFlow)),
        "refused at the call site, by name, without a frame being built"
    );
    assert_eq!(
        step(a.send_to(b"stray", (PEER_A, 54))),
        Some(Err(SocketRefusal::OutsideFlow)),
        "a different peer *port* is outside the flow too"
    );

    // The capability B holds is untouched by any of this — affine, and never
    // shared with A's socket.
    assert!(cap_b.frame_filter().is_some(), "B still holds its own");
}

/// S0 grants nothing, so there is no socket over it — refused by its own
/// name, because the fix is a mint and not a narrower peer.
#[test]
fn a_mediated_authority_opens_no_socket() {
    let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let flow = LocalFlow::new(
        MockIfnet::new(&mut scratch.0, A_MAC).expect("scratch"),
        FlowCap::mediated(VifId {
            index: 0,
            generation: 1,
        }),
        identity(ME, Some(GW)),
    );
    assert_eq!(
        UdpSocket::open(flow).err(),
        Some(SocketRefusal::Mediated),
        "an authority that covers nothing is not a socket"
    );
}

/// A connected authority is not a listening one, and binding a listener over
/// it is refused rather than silently accepting from exactly one peer.
#[test]
fn a_connected_authority_is_not_a_listener() {
    let mut if_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    storage!(rx, tx, sockets, smol);
    let flow = LocalFlow::new(
        MockIfnet::new(&mut if_b.0, B_MAC).expect("scratch"),
        cap(
            1,
            Scope::Flow(FlowMatch::connection(IpProto::TCP, (SRV, 80), (ME, 49152))),
        ),
        identity(SRV, None),
    );
    let built = TcpListener::bind(
        flow,
        TcpStorage {
            rx: &mut rx,
            tx: &mut tx,
            sockets: &mut sockets,
            scratch: &mut smol.0,
        },
    );
    assert!(matches!(built.err(), Some(SocketRefusal::OutsideFlow)));
}
