//! **`inet-flow-isolation`** — the authority half of `net-vif-isolation`.
//!
//! `res/NETWORK_SERVICE.md` §7 records what the frame program could *not*
//! prove: "it does not prove mutual distrust between two founded *tasks*,
//! because both vifs belong to one client." That gap has stood since it was
//! written, for want of a second mutually-distrusting consumer. The internet
//! stack is the first thing that produces one.
//!
//! What this file proves, end to end and without hardware:
//!
//! 1. **One arbiter, two clients.** A single [`InetRow`] mints both flows. It
//!    is the only party that can, because it is the only one holding the port
//!    space and the station's address.
//! 2. **The separation is the flow, not the address.** Both clients answer to
//!    the *same* station address — the shape `AddressMode::Shared` forces, and
//!    the one where an L2 filter provably cannot tell them apart.
//! 3. **The enforcement is the real predicate.** The segment below delivers
//!    through [`FrameFilter::accepts_frame`] — the same function the row's
//!    classifier calls — rather than through test logic. A pass here is a pass
//!    of the shipping contract, not of a double that agrees with it.

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_driver::net::{EtherType, FrameFilter, FrameHead, MacAddr, VifId};
use kore_inet::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
use kore_inet::row::{Identity, InetRow, OpenGrant};
use kore_inet::{Event, IpStack, RecordingInstall};

const STATION_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
const GW_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
const PEER_A: Ipv4Addr = Ipv4Addr([93, 184, 216, 34]);
const PEER_B: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

#[repr(align(64))]
struct Aligned<const N: usize>([u8; N]);

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
    let mut datagram = [0u8; 1024];
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
        dst: STATION_MAC,
        src: GW_MAC,
        ethertype: EtherType::IPV4,
    }
    .write(out)
    .expect("header fits");
    out[14..14 + dlen].copy_from_slice(&datagram[..dlen]);
    14 + dlen
}

/// One shared link, delivering through the **shipping** classifier predicate.
///
/// This stands in for `services/net`'s classifier. Crucially it calls
/// `FrameFilter::accepts_frame` rather than reimplementing the decision, so
/// what passes here is what passes in the row.
fn deliver_to(
    net: &mut MockIfnet<'_>,
    filter: &FrameFilter,
    frame: &[u8],
    station: MacAddr,
) -> bool {
    let head = FrameHead::parse(frame).expect("well formed");
    let body = &frame[14..];
    if !filter.accepts_frame(&head, body, station) {
        return false;
    }
    net.inject(frame, head.ethertype).expect("injects");
    true
}

fn identity() -> Identity {
    Identity::new(ME, 24, Some(GW)).expect("valid")
}

/// What a [`Event::Udp`] said, as values that outlive the dispatch.
///
/// The event borrows the engine's staging buffer, so an assertion has to name
/// the facts rather than keep the event. The five payload bytes are among
/// them on purpose: "B received nothing of A's" is a claim about *content*,
/// and a test that only counted arrivals would not have made it.
fn delivered(e: Event<'_>) -> Option<((Ipv4Addr, u16), u16, [u8; 5])> {
    match e {
        Event::Udp {
            from,
            local_port,
            payload,
        } => {
            let mut body = [0u8; 5];
            body[..payload.len().min(5)].copy_from_slice(&payload[..payload.len().min(5)]);
            Some((from, local_port, body))
        }
        _ => None,
    }
}

/// Drive one of the row's futures to its answer.
///
/// Bounded rather than spun: the install double never parks, so one poll is
/// always enough, and an exhaustion here would mean a mint had grown a
/// dependency a host test cannot satisfy.
fn run<T>(f: impl core::future::Future<Output = T>) -> T {
    kore_rt::try_block_on(f, 4).expect("the row's futures settle without an executor")
}

/// **The witness.** Two mutually-distrusting clients, one station address, one
/// arbiter — and a datagram reaches exactly the one whose flow it belongs to.
#[test]
fn two_clients_on_one_station_receive_only_their_own_flow() {
    // One arbiter. It holds the port space and the address, and it is the
    // only party that can mint either client's authority.
    let mut row = InetRow::new();
    row.set_identity(identity());

    let vif_a = VifId {
        index: 0,
        generation: 1,
    };
    let vif_b = VifId {
        index: 1,
        generation: 1,
    };

    // The row installs each client's enforcement as part of its mint, through
    // the same seam a founded row uses onto `services/net`.
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

    // The filters are read back from **what the row installed**, not from the
    // capabilities the test happens to hold. That is the whole difference
    // between a capability that is enforced and one that is merely described:
    // if the mint had handed back an authority without installing it, these
    // two lookups would be `None` and the segment below would never run.
    let filter_a = install.filter(vif_a).expect("A's enforcement is installed");
    let filter_b = install.filter(vif_b).expect("B's enforcement is installed");
    assert_eq!(filter_a, cap_a.frame_filter().expect("S1 has a filter"));
    assert_eq!(filter_b, cap_b.frame_filter().expect("S1 has a filter"));

    // Two independent stacks, each over its own vif — and note both answer to
    // the *same* station address.
    let mut scratch_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut scratch_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut stack_a = IpStack::new(
        MockIfnet::new(&mut scratch_a.0, STATION_MAC).expect("scratch"),
        identity(),
    );
    let mut stack_b = IpStack::new(
        MockIfnet::new(&mut scratch_b.0, STATION_MAC).expect("scratch"),
        identity(),
    );
    assert!(stack_a.bind_udp(5000));
    assert!(stack_b.bind_udp(5001));

    // ── A's datagram ────────────────────────────────────────────────────
    let mut f = [0u8; 512];
    let len = udp_frame((PEER_A, 53), (ME, 5000), b"for-a", &mut f);

    let to_a = deliver_to(stack_a.ifnet_mut(), &filter_a, &f[..len], STATION_MAC);
    let to_b = deliver_to(stack_b.ifnet_mut(), &filter_b, &f[..len], STATION_MAC);
    assert!(to_a, "A's flow accepted A's datagram");
    assert!(
        !to_b,
        "B's filter rejected it — same station address, different flow"
    );

    let mut seen_a = None;
    stack_a.input(0, |e| seen_a = delivered(e));
    assert_eq!(seen_a, Some(((PEER_A, 53), 5000, *b"for-a")));
    let mut seen_b = None;
    stack_b.input(0, |e| seen_b = delivered(e));
    assert_eq!(seen_b, None, "B saw nothing of A's conversation");

    // ── B's datagram — the mirror image, so neither client is merely inert ─
    let len = udp_frame((PEER_B, 53), (ME, 5001), b"for-b", &mut f);

    assert!(
        !deliver_to(stack_a.ifnet_mut(), &filter_a, &f[..len], STATION_MAC),
        "A's filter rejected B's datagram"
    );
    assert!(deliver_to(
        stack_b.ifnet_mut(),
        &filter_b,
        &f[..len],
        STATION_MAC
    ));

    let mut seen_a = None;
    stack_a.input(1, |e| seen_a = delivered(e));
    assert_eq!(seen_a, None, "A saw nothing of B's conversation");
    let mut seen_b = None;
    stack_b.input(1, |e| seen_b = delivered(e));
    assert_eq!(seen_b, Some(((PEER_B, 53), 5001, *b"for-b")));

    // ── A datagram belonging to neither reaches neither ──────────────────
    let len = udp_frame((PEER_A, 53), (ME, 9999), b"stray", &mut f);
    assert!(!deliver_to(stack_a.ifnet_mut(), &filter_a, &f[..len], STATION_MAC));
    assert!(!deliver_to(stack_b.ifnet_mut(), &filter_b, &f[..len], STATION_MAC));
}

/// Without the flow half this is what would happen, stated as a test so the
/// claim is measured rather than asserted in prose.
///
/// Both clients bind IPv4 at the same station address. With an L2-only filter
/// — `flow: None`, which is every filter that predates A0 — the *same*
/// datagram is accepted by both. That is the confidentiality hole, and it is
/// why the filter grew a flow dimension.
#[test]
fn an_l2_only_filter_cannot_separate_two_ip_clients() {
    let l2_only = FrameFilter::protocol(EtherType::IPV4);
    assert_eq!(l2_only.flow, None, "the pre-A0 contract");

    let mut f = [0u8; 512];
    let len = udp_frame((PEER_A, 53), (ME, 5000), b"for-a", &mut f);
    let head = FrameHead::parse(&f[..len]).expect("well formed");

    // One filter, one frame — and it is accepted. Two clients holding this
    // same filter would both receive it.
    assert!(
        l2_only.accepts_frame(&head, &f[14..len], STATION_MAC),
        "an L2 filter admits any IPv4 frame to this station"
    );

    // The flow-bearing filter for a *different* conversation rejects it.
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();
    let other = run(row.open_flow(
        VifId {
            index: 1,
            generation: 1,
        },
        OpenGrant::default(),
        IpProto::UDP,
        Some(5001),
        (PEER_B, 53),
        &mut install,
    ))
    .expect("mints");
    let narrowed = other.frame_filter().expect("installs");
    assert!(
        !narrowed.accepts_frame(&head, &f[14..len], STATION_MAC),
        "the flow dimension is what separates them"
    );
}

/// S0 installs no filter, because nothing is delivered directly — the row
/// stays on the datapath for that holder. A gap in the ladder would show up
/// here as an authority that silently received everything.
#[test]
fn the_mediated_grant_installs_no_filter() {
    let row = InetRow::new();
    let cap = row.open_mediated(VifId {
        index: 0,
        generation: 1,
    });
    assert!(
        cap.frame_filter().is_none(),
        "S0 delivers nothing directly, so there is nothing to install"
    );
}

/// A reused vif slot does not inherit the previous tenant's *enforcement*
/// either: the generation is part of the identifier the install is recorded
/// under, so the authority and the filter backing it die together.
#[test]
fn a_reused_vif_slot_inherits_no_installed_filter() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();
    let first = VifId {
        index: 2,
        generation: 1,
    };
    run(row.open_flow(
        first,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5000),
        (PEER_A, 53),
        &mut install,
    ))
    .expect("mints");
    assert!(install.filter(first).is_some());

    let next_tenant = VifId {
        index: 2,
        generation: 2,
    };
    assert!(
        install.filter(next_tenant).is_none(),
        "the slot's next tenant starts with no enforcement of this row's"
    );
}

/// A revoked vif's authority stops resolving, so the filter it would install
/// belongs to an identifier the table no longer honours.
#[test]
fn a_reused_vif_slot_does_not_inherit_the_previous_flow() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let first = VifId {
        index: 2,
        generation: 1,
    };
    let mut install = RecordingInstall::new();
    let cap = run(row.open_flow(
        first,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5000),
        (PEER_A, 53),
        &mut install,
    ))
    .expect("mints");

    let next_tenant = VifId {
        index: 2,
        generation: 2,
    };
    assert!(
        !cap.resolves_against(next_tenant),
        "the slot's next tenant does not inherit this authority"
    );
    // And the port is still the first holder's until it closes.
    assert_eq!(row.ports().holder(IpProto::UDP, 5000), Some(first));
}
