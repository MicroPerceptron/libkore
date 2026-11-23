//! **S3 — raw L2 at one station**, and the exact edge of what it grants.
//!
//! The sibling of [`flow_isolation`](../flow_isolation.rs), one rung up. That
//! file proves two S1 holders at *one* station address cannot see each other,
//! because the flow dimension separates them. This one is about the rung
//! where that dimension is deliberately absent: an S3 holder has no L3
//! predicate at all, so the only thing standing between it and another
//! client's traffic is the station address its filter is evaluated against.
//!
//! What this file proves, end to end and without hardware:
//!
//! 1. **The grant is real.** An S3 holder receives every protocol crossing
//!    its station — including ARP, which carries no five-tuple and which no
//!    rung below S3 can be granted at all.
//! 2. **The grant is bounded.** An S3 holder on vif A receives nothing
//!    addressed to vif B's station. Raw L2 on a vif is not raw L2 on the
//!    device, and one address is the whole of the difference.
//! 3. **The precondition is measured, not asserted.** That bound exists only
//!    where the two vifs have two station addresses. Under the frame row's
//!    shipping `AddressMode::Shared` they do not, and the last test here
//!    pins what that costs rather than leaving it in a doc comment.
//! 4. **The powers stay orthogonal.** Without `promiscuous` the holder hears
//!    nothing it was not addressed; without `spoof` it may speak only as its
//!    own station.
//!
//! Every receive decision below goes through [`FrameFilter::accepts_frame`] —
//! the function the row's classifier calls — so a pass here is a pass of the
//! shipping contract rather than of a double that agrees with it.

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_driver::net::{
    EtherType, EtherTypeMatch, FrameFilter, FrameHead, MacAddr, MulticastMatch, VifId,
};
use kore_inet::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
use kore_inet::row::{Identity, InetRow, OpenGrant, OpenRefusal};
use kore_inet::{FlowCap, Ifnet, Powers, RecordingInstall, Scope};

/// Two vifs, two stations — the shape `AddressMode::PerVif` produces and the
/// one S3 is an isolated grant under.
const STATION_A: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x0a]);
const STATION_B: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x0b]);
const GW_MAC: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
const PEER: Ipv4Addr = Ipv4Addr([93, 184, 216, 34]);

#[repr(align(64))]
struct Aligned<const N: usize>([u8; N]);

fn identity() -> Identity {
    Identity::new(ME, 24, Some(GW)).expect("valid")
}

/// Drive one of the row's futures to its answer.
///
/// Bounded rather than spun: the install double never parks, so one poll is
/// always enough, and an exhaustion here would mean a mint had grown a
/// dependency a host test cannot satisfy.
fn run<T>(f: impl core::future::Future<Output = T>) -> T {
    kore_rt::try_block_on(f, 4).expect("the row's futures settle without an executor")
}

/// The grant a composer that has established per-vif station addresses hands
/// in. Nothing in the tree builds one; see `the_rung_has_no_holder_in_tree`.
fn raw_grant() -> OpenGrant {
    OpenGrant {
        raw_l2: true,
        ..OpenGrant::default()
    }
}

/// A frame of `ethertype` addressed to `dst`, with a body that is only bytes.
///
/// Deliberately *not* a well-formed datagram for the non-IP cases: the point
/// of S3 is that it carries protocols the flow vocabulary cannot parse, and a
/// body `FiveTuple::parse` can make nothing of is the sharpest form of that.
fn frame(dst: MacAddr, ethertype: EtherType, body: &[u8], out: &mut [u8]) -> usize {
    FrameHead {
        dst,
        src: GW_MAC,
        ethertype,
    }
    .write(out)
    .expect("header fits");
    out[14..14 + body.len()].copy_from_slice(body);
    14 + body.len()
}

/// One UDP datagram, as a peer would put it on the wire — the traffic an S1
/// holder owns, used here to ask whether an S3 holder elsewhere can see it.
fn udp_frame(dst_mac: MacAddr, dst: (Ipv4Addr, u16), payload: &[u8], out: &mut [u8]) -> usize {
    use smoltcp::phy::ChecksumCapabilities;
    use smoltcp::wire::{IpAddress, Ipv4Address, Ipv4Packet, Ipv4Repr, UdpPacket, UdpRepr};

    let to_ip = |a: Ipv4Addr| Ipv4Address::from_octets(a.0);
    let udp = UdpRepr {
        src_port: 53,
        dst_port: dst.1,
    };
    let udp_len = udp.header_len() + payload.len();
    let ip = Ipv4Repr {
        src_addr: to_ip(PEER),
        dst_addr: to_ip(dst.0),
        next_header: smoltcp::wire::IpProtocol::Udp,
        payload_len: udp_len,
        hop_limit: 64,
    };
    let mut datagram = [0u8; 256];
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
    frame(dst_mac, EtherType::IPV4, &datagram[..dlen], out)
}

/// One shared link, delivering through the **shipping** classifier predicate.
///
/// This stands in for `services/net`'s classifier. Crucially it calls
/// `FrameFilter::accepts_frame` rather than reimplementing the decision, so
/// what passes here is what passes in the row — and `station` is the vif's
/// own address, which is the parameter the whole of S3's boundary rests on.
fn deliver_to(
    net: &mut MockIfnet<'_>,
    filter: &FrameFilter,
    frame: &[u8],
    station: MacAddr,
) -> bool {
    let head = FrameHead::parse(frame).expect("well formed");
    if !filter.accepts_frame(&head, &frame[14..], station) {
        return false;
    }
    net.inject(frame, head.ethertype).expect("injects");
    true
}

/// Everything one holder's vif receives in a sweep, as EtherTypes.
///
/// [`MockIfnet`] holds one frame in flight, so a sweep follows every
/// delivery rather than a batch of them — which is also the drive-cycle
/// posture the real seam has.
fn drain(net: &mut MockIfnet<'_>) -> std::vec::Vec<EtherType> {
    let mut seen = std::vec::Vec::new();
    net.poll(|_, t| seen.push(t));
    seen
}

// ── 1. what the grant actually carries ──────────────────────────────────

/// **An S3 holder sees raw frames on its own vif.**
///
/// Every protocol, including the ones that have no five-tuple to match on.
/// That is the whole of what S3 adds over S1, and it is asserted against
/// frames rather than against the filter's fields, because a filter that
/// looks right and admits the wrong frame is the failure that matters.
#[test]
fn a_raw_holder_sees_every_protocol_on_its_own_station() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();
    let vif_a = VifId {
        index: 0,
        generation: 1,
    };

    let cap = run(row.open_interface(vif_a, raw_grant(), STATION_A, &mut install))
        .expect("the composition granted raw L2");

    // Read back from what the row *installed*, not from the capability the
    // test holds: an authority whose enforcement was never installed would
    // show up here as `None`, and every delivery below would be vacuous.
    let filter = install
        .filter(vif_a)
        .expect("the mint installed S3's enforcement");
    assert_eq!(filter, cap.frame_filter().expect("S3 has a filter"));

    let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut net = MockIfnet::new(&mut scratch.0, STATION_A).expect("scratch");

    let mut f = [0u8; 512];
    // ARP: no IPv4 header, so `FiveTuple::parse` yields nothing at all. A
    // flow-bearing filter rejects a body it cannot parse — by design — so
    // this frame is reachable at no rung below S3.
    //
    // KORE_PEER: a protocol another client might own.
    // IPv4: the one every rung below can carry, here with no flow
    // restriction narrowing it.
    let cases = [
        (frame(STATION_A, EtherType::ARP, &[0xde; 28], &mut f), f),
        (frame(STATION_A, EtherType::KORE_PEER, &[0x11; 40], &mut f), f),
        (udp_frame(STATION_A, (ME, 5000), b"raw", &mut f), f),
    ];
    for (len, bytes) in cases {
        let t = FrameHead::parse(&bytes[..len]).expect("well formed").ethertype;
        assert!(
            deliver_to(&mut net, &filter, &bytes[..len], STATION_A),
            "raw L2 carries {t:?}"
        );
        assert_eq!(drain(&mut net), std::vec![t], "and it reached the ring");
    }

    // The counter-half, so "S3 adds the protocol dimension" is measured: the
    // same ARP frame against the station-wide S1 grant the row can also mint.
    let mut other = InetRow::new();
    other.set_identity(identity());
    let s1 = run(other.open_station(
        VifId {
            index: 1,
            generation: 1,
        },
        OpenGrant::default(),
        &mut install,
    ))
    .expect("S1 at this station");
    let s1_filter = s1.frame_filter().expect("installs");
    let len = frame(STATION_A, EtherType::ARP, &[0xde; 28], &mut f);
    let head = FrameHead::parse(&f[..len]).expect("well formed");
    assert!(
        !s1_filter.accepts_frame(&head, &f[14..len], STATION_A),
        "below S3 an ARP frame is not grantable at all"
    );
}

// ── 2. where the grant stops ────────────────────────────────────────────

/// **An S3 holder on vif A sees nothing of vif B.**
///
/// The headline boundary. A's authority is the widest this ladder mints, and
/// it still does not reach a frame addressed to B's station — not its IPv4
/// datagrams, not its ARP, not anything. Raw L2 on a vif is not raw L2 on the
/// device.
#[test]
fn a_raw_holder_on_one_vif_sees_nothing_of_another() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();

    let vif_a = VifId {
        index: 0,
        generation: 1,
    };
    let vif_b = VifId {
        index: 1,
        generation: 1,
    };

    // A holds raw L2. B holds an ordinary flow — the asymmetric case, which
    // is the one that matters: the strongest authority against the weakest.
    run(row.open_interface(vif_a, raw_grant(), STATION_A, &mut install)).expect("A's raw L2");
    run(row.open_flow(
        vif_b,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5001),
        (PEER, 53),
        &mut install,
    ))
    .expect("B's flow");

    let filter_a = install.filter(vif_a).expect("A's enforcement");
    let filter_b = install.filter(vif_b).expect("B's enforcement");

    let mut scratch_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut scratch_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut net_a = MockIfnet::new(&mut scratch_a.0, STATION_A).expect("scratch");
    let mut net_b = MockIfnet::new(&mut scratch_b.0, STATION_B).expect("scratch");

    // B's datagram, addressed to B's station.
    let mut f = [0u8; 512];
    let len = udp_frame(STATION_B, (ME, 5001), b"for-b", &mut f);

    assert!(
        !deliver_to(&mut net_a, &filter_a, &f[..len], STATION_A),
        "the widest grant in the ladder does not reach another station"
    );
    assert!(
        deliver_to(&mut net_b, &filter_b, &f[..len], STATION_B),
        "and B still receives its own"
    );

    // Non-IP traffic at B's station is equally out of A's reach — so the
    // separation is the address, not an accident of the flow vocabulary.
    let len = frame(STATION_B, EtherType::KORE_PEER, &[0x22; 40], &mut f);
    assert!(!deliver_to(&mut net_a, &filter_a, &f[..len], STATION_A));

    assert!(drain(&mut net_a).is_empty(), "A's ring stayed empty");
    assert_eq!(drain(&mut net_b), std::vec![EtherType::IPV4]);
}

/// **The precondition, measured.** The isolation above is the station
/// address and nothing else, so under an addressing mode that gives two vifs
/// one address it is gone — and gone *silently*, without the holder having
/// asked for a power.
///
/// `AddressMode::Shared` is the frame row's `#[default]`. This test exists so
/// that fact is a failing-closed gate rather than a sentence in a design
/// document: it pins what a raw grant would capture there, which is why
/// minting one is gated on a composition that has established per-vif
/// addressing (`OpenGrant::raw_l2`).
#[test]
fn under_one_shared_station_a_raw_grant_captures_everything() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();

    let vif_a = VifId {
        index: 0,
        generation: 1,
    };
    let vif_b = VifId {
        index: 1,
        generation: 1,
    };

    run(row.open_interface(vif_a, raw_grant(), STATION_A, &mut install)).expect("A's raw L2");
    let cap_b = run(row.open_flow(
        vif_b,
        OpenGrant::default(),
        IpProto::UDP,
        Some(5001),
        (PEER, 53),
        &mut install,
    ))
    .expect("B's flow");

    let filter_a = install.filter(vif_a).expect("A's enforcement");
    let filter_b = cap_b.frame_filter().expect("B's enforcement");

    // The one change from the test above: B answers to A's address, which is
    // exactly what `AddressMode::Shared` does to every vif on a device.
    let shared = STATION_A;
    let mut f = [0u8; 512];
    let len = udp_frame(shared, (ME, 5001), b"for-b", &mut f);
    let head = FrameHead::parse(&f[..len]).expect("well formed");

    assert!(
        filter_b.accepts_frame(&head, &f[14..len], shared),
        "B receives its own datagram"
    );
    assert!(
        filter_a.accepts_frame(&head, &f[14..len], shared),
        "AND SO DOES A — one address, so B's traffic is addressed to A too. \
         This is why `OpenGrant::raw_l2` is a statement about the composition \
         rather than about the caller."
    );
    assert!(
        !filter_a.promiscuous,
        "and A never asked for promiscuity — the power cannot defend a \
         boundary that the addressing mode already removed"
    );
}

// ── 3. the two orthogonal powers ────────────────────────────────────────

/// **S3 without `promiscuous` cannot overhear.** The rung is the strongest
/// below a hardware queue and it still does not imply the power; holding it
/// is what moves the promiscuous bit and the multicast match, together.
#[test]
fn raw_l2_without_promiscuity_hears_only_its_own_station() {
    let vif = VifId {
        index: 0,
        generation: 1,
    };
    let scope = Scope::Interface {
        station: STATION_A,
    };

    let deaf = FlowCap::mint(vif, scope, Powers::NONE).expect("S3 mints");
    let tap = FlowCap::mint(
        vif,
        scope,
        Powers {
            spoof: false,
            promiscuous: true,
        },
    )
    .expect("S3 mints");

    let deaf_f = deaf.frame_filter().expect("installs");
    let tap_f = tap.frame_filter().expect("installs");
    assert_eq!(deaf_f.multicast, MulticastMatch::None);
    assert_eq!(tap_f.multicast, MulticastMatch::All);

    let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
    let mut net = MockIfnet::new(&mut scratch.0, STATION_A).expect("scratch");
    let mut f = [0u8; 512];

    // A neighbour's unicast, and a multicast group this holder never joined.
    let group = MacAddr([0x01, 0x80, 0xc2, 0, 0, 0x03]);
    for dst in [STATION_B, group] {
        let len = frame(dst, EtherType::IPV4, &[0x45; 40], &mut f);
        assert!(
            !deliver_to(&mut net, &deaf_f, &f[..len], STATION_A),
            "{dst:?} was not addressed to this holder"
        );
        let head = FrameHead::parse(&f[..len]).expect("well formed");
        assert!(
            tap_f.accepts_frame(&head, &f[14..len], STATION_A),
            "{dst:?} reaches a holder that was granted the power"
        );
    }
    assert!(drain(&mut net).is_empty(), "nothing overheard");

    // Broadcast needs no power, because it is addressed to this station too.
    let len = frame(MacAddr::BROADCAST, EtherType::ARP, &[0xde; 28], &mut f);
    assert!(deliver_to(&mut net, &deaf_f, &f[..len], STATION_A));
}

/// **S3 without `spoof` cannot forge.** The authority permits one source
/// address — the station its grant names — and the power, not the rung, is
/// what lifts that.
///
/// # What this test does and does not witness
///
/// The authority's own answer, [`FlowCap::may_source`]. The *enforcement* is
/// the frame row's `validate_tx`, which refuses any frame whose source is
/// not the vif's own address, consulting no grant at all — so the wire is
/// today stricter than any authority here can describe, and a holder with
/// `spoof` still could not forge. That predicate is private to
/// `services/net` and takes a live `Vif`, so it is not reachable from a host
/// test; it is named here rather than approximated by a local copy that
/// could drift away from it.
#[test]
fn raw_l2_without_spoof_may_speak_only_as_its_own_station() {
    let vif = VifId {
        index: 0,
        generation: 1,
    };
    let scope = Scope::Interface {
        station: STATION_A,
    };

    let honest = FlowCap::mint(vif, scope, Powers::NONE).expect("S3 mints");
    assert!(honest.may_source(STATION_A, STATION_A));
    assert!(
        !honest.may_source(STATION_B, STATION_A),
        "raw L2 is not a licence to be another station"
    );
    assert!(!honest.may_source(GW_MAC, STATION_A));

    let forger = FlowCap::mint(
        vif,
        scope,
        Powers {
            spoof: true,
            promiscuous: false,
        },
    )
    .expect("S3 mints");
    assert!(forger.may_source(STATION_B, STATION_A));

    // A derived child that dropped the power cannot pick it back up, so a
    // forging authority cannot be laundered through an attenuation.
    let child = forger
        .derive(scope, Powers::NONE)
        .expect("dropping a power narrows");
    assert!(!child.may_source(STATION_B, STATION_A));
}

// ── 4. attenuation, and revocation ──────────────────────────────────────

/// **Derivation from S3 narrows and never widens, in any dimension** — and
/// the narrowing reaches the *enforcement*, not only the record.
///
/// S2 sits between S3 and S1 and is still modelled, so the ladder is crossed
/// rather than walked: the child lands on S1 and its installed filter is
/// strictly tighter than its parent's on every field that moved.
#[test]
fn a_child_of_a_raw_grant_installs_a_narrower_filter() {
    let vif = VifId {
        index: 0,
        generation: 1,
    };
    let scope = Scope::Interface {
        station: STATION_A,
    };
    let parent = FlowCap::mint(
        vif,
        scope,
        Powers {
            spoof: true,
            promiscuous: true,
        },
    )
    .expect("S3 mints");

    let flow = kore_driver::net::flow::FlowMatch::connection(IpProto::UDP, (ME, 5000), (PEER, 53));
    let child = parent
        .derive(Scope::Flow(flow), Powers::NONE)
        .expect("S3 narrows to S1 across the modelled rung");

    let wide = parent.frame_filter().expect("installs");
    let narrow = child.frame_filter().expect("installs");

    assert_eq!(wide.ethertype, EtherTypeMatch::Any);
    assert!(
        narrow.ethertype.within(&wide.ethertype) && !wide.ethertype.within(&narrow.ethertype),
        "the protocol dimension strictly narrowed"
    );
    assert_eq!(wide.flow, None, "the parent imposes no L3 predicate");
    assert_eq!(narrow.flow, Some(flow), "the child imposes one");
    assert!(wide.promiscuous && !narrow.promiscuous);
    assert!(narrow.multicast.within(&wide.multicast));

    // And the child cannot climb back, at any rung or any power.
    assert!(child.derive(scope, Powers::NONE).is_err());
    assert!(
        child
            .derive(
                Scope::Flow(flow),
                Powers {
                    spoof: true,
                    promiscuous: false
                }
            )
            .is_err()
    );
}

/// A revoked generation stops resolving, for a raw grant exactly as for a
/// flow: the authority outlives the vif's close and is refused at its next
/// use rather than needing to be hunted down.
#[test]
fn a_revoked_generation_stops_resolving_a_raw_grant() {
    let mut row = InetRow::new();
    let mut install = RecordingInstall::new();
    let first = VifId {
        index: 2,
        generation: 1,
    };
    let cap = run(row.open_interface(first, raw_grant(), STATION_A, &mut install))
        .expect("raw L2, with no identity needed");

    assert!(cap.resolves_against(first));

    let next_tenant = VifId {
        index: 2,
        generation: 2,
    };
    assert!(
        !cap.resolves_against(next_tenant),
        "the slot's next tenant does not inherit raw L2"
    );
    assert!(
        install.filter(next_tenant).is_none(),
        "nor the enforcement behind it"
    );

    // A child inherits the binding, so one close stops the whole subtree.
    let child = cap
        .derive(
            Scope::Interface {
                station: STATION_A,
            },
            Powers::NONE,
        )
        .expect("narrows");
    assert!(!child.resolves_against(next_tenant));
}

// ── 5. who can hold one ─────────────────────────────────────────────────

/// **Nothing in this tree can hold an S3 today, and that is stated here
/// rather than disguised.**
///
/// `res/INET_STACK.md` §9 names the claimant as a hypervisor guest wanting
/// raw L2. `services/hypervisor` has no frame path of its own, the inet
/// service's wire protocol has no verb that reaches
/// [`InetRow::open_interface`], and the grant that would admit one is
/// default-off — so the only caller is this file.
///
/// The test asserts the half that is mechanically checkable: the default
/// grant, which is the one `services/inet` hands to every arriving client,
/// refuses the rung. A holder appearing later changes that composition, not
/// this mechanism.
#[test]
fn the_rung_has_no_holder_in_tree() {
    let mut row = InetRow::new();
    row.set_identity(identity());
    let mut install = RecordingInstall::new();

    assert!(
        !OpenGrant::default().raw_l2,
        "the grant every in-tree caller builds"
    );
    assert_eq!(
        run(row.open_interface(
            VifId {
                index: 0,
                generation: 1
            },
            OpenGrant::default(),
            STATION_A,
            &mut install,
        )),
        Err(OpenRefusal::RawL2NotGranted),
        "so the rung is built and unreachable, which is the honest state"
    );
    assert_eq!(install.attempts(), 0, "and nothing was installed for it");
}
