//! **IPv4** — input validation and dispatch, and the output path that turns a
//! destination into a frame on the wire.
//!
//! Written here because this is where the routing decision, the ARP wait and
//! the reserve-forward discipline meet, and all three are kore's. The header
//! *encoding* is borrowed (`res/INET_STACK.md` §5.1).

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_driver::net::frame::HEADER_LEN;
use kore_driver::net::{EtherType, FrameHead};
use smoltcp::wire::{Ipv4Packet, Ipv4Repr};

use crate::ifnet::{Ifnet, TxSlot};
use crate::netinet::arp::{self, ArpCache, Resolution};
use crate::netinet::{from_ip, to_ip};
use crate::row::Identity;

/// The IPv4 header this stack emits — no options, so a fixed 20 bytes.
pub const IPV4_HEADER_LEN: usize = 20;

/// The default hop limit.
pub const DEFAULT_TTL: u8 = 64;

/// Why a datagram could not be sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputError {
    /// No route: the destination is off-link and no gateway is configured.
    Unreachable,
    /// The next hop's hardware address is not known yet. A request has been
    /// sent or is already outstanding; retry.
    ///
    /// Not a failure — the ordinary first outcome of talking to anyone new.
    /// A caller retries rather than giving up, and the ARP layer's own rate
    /// limit is what stops a retry loop becoming a broadcast storm.
    Pending,
    /// The transmit ring has no slot.
    NoSlot,
    /// The payload will not fit the link's MTU.
    TooLong,
}

/// Send one datagram to `dst`.
///
/// `fill` writes the transport payload into the region after the reserved
/// headers; the headers are then written in place. That ordering is the whole
/// of the reserve-forward discipline — the caller never computes an offset,
/// so a header length and a frame length cannot disagree.
pub fn output<I: Ifnet, F>(
    net: &mut I,
    cache: &mut ArpCache,
    identity: &Identity,
    dst: Ipv4Addr,
    proto: IpProto,
    payload_len: usize,
    now_ms: u64,
    fill: F,
) -> Result<(), OutputError>
where
    F: FnOnce(&mut [u8]),
{
    let next_hop = identity.next_hop(dst).ok_or(OutputError::Unreachable)?;
    let mac = match arp::resolve(cache, net, identity.addr(), next_hop, now_ms) {
        Resolution::Known(mac) => mac,
        Resolution::Asked | Resolution::Pending => return Err(OutputError::Pending),
    };
    if payload_len + IPV4_HEADER_LEN > net.mtu() {
        return Err(OutputError::TooLong);
    }

    // Read before reserving, for the same reason: the slot holds the
    // interface for its whole life.
    let src_mac = net.hwaddr();
    let mut slot = net.reserve().ok_or(OutputError::NoSlot)?;
    let reserve = HEADER_LEN + IPV4_HEADER_LEN;
    let body = slot.reserve_headers(reserve).ok_or(OutputError::TooLong)?;
    if body.len() < payload_len {
        return Err(OutputError::TooLong);
    }
    fill(&mut body[..payload_len]);

    let repr = Ipv4Repr {
        src_addr: to_ip(identity.addr()),
        dst_addr: to_ip(dst),
        next_header: smoltcp::wire::IpProtocol::from(proto.0),
        payload_len,
        hop_limit: DEFAULT_TTL,
    };

    let headers = slot.headers_mut();
    FrameHead {
        dst: mac,
        src: src_mac,
        ethertype: EtherType::IPV4,
    }
    .write(&mut headers[..HEADER_LEN])
    .ok_or(OutputError::TooLong)?;
    {
        // `new_unchecked`, not `new_checked`: the total-length field this
        // buffer will carry is written by `emit` below, so checking it first
        // validates a zero that nothing has filled in yet.
        let mut packet = Ipv4Packet::new_unchecked(&mut headers[HEADER_LEN..reserve]);
        repr.emit(&mut packet, &smoltcp::phy::ChecksumCapabilities::default());
    }
    // The reservation already covers the Ethernet *and* IP headers, so what
    // is committed past it is the transport payload alone. Adding the IP
    // header again here would declare a frame 20 bytes longer than the one
    // actually written.
    slot.commit(payload_len, EtherType::IPV4)
        .map_err(|_| OutputError::TooLong)?;
    Ok(())
}

/// A validated datagram addressed to this station.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Datagram<'a> {
    /// Who sent it.
    pub src: Ipv4Addr,
    /// Who it was addressed to — this station, or a broadcast it accepts.
    pub dst: Ipv4Addr,
    /// The protocol carried.
    pub proto: IpProto,
    /// The transport payload.
    pub payload: &'a [u8],
}

/// Validate and classify one received IPv4 frame body.
///
/// `None` for anything malformed, not addressed here, or fragmented. The
/// fragment case is deliberate and not a silent drop of convenience: this
/// stack does not reassemble, and handing a protocol a fragment's tail as if
/// it were a datagram is how a port-pinned receiver gets fed attacker-chosen
/// bytes.
pub fn input<'a>(identity: &Identity, body: &'a [u8]) -> Option<Datagram<'a>> {
    let packet = Ipv4Packet::new_checked(body).ok()?;
    // The checksum is verified rather than assumed — this runs on bytes a
    // peer chose.
    if !packet.verify_checksum() {
        return None;
    }
    let repr = Ipv4Repr::parse(&packet, &smoltcp::phy::ChecksumCapabilities::default()).ok()?;
    if packet.more_frags() || packet.frag_offset() != 0 {
        return None;
    }

    let dst = from_ip(repr.dst_addr);
    let mine = identity.addr();
    // Addressed to this station, or to a broadcast it answers for. Anything
    // else reached us because the link is shared, not because it was ours.
    if dst != mine && !dst.is_broadcast() && !is_directed_broadcast(identity, dst) {
        return None;
    }

    let payload = packet.payload();
    let len = repr.payload_len.min(payload.len());
    Some(Datagram {
        src: from_ip(repr.src_addr),
        dst,
        proto: IpProto(repr.next_header.into()),
        payload: &payload[..len],
    })
}

/// Whether `dst` is this prefix's all-ones broadcast.
fn is_directed_broadcast(identity: &Identity, dst: Ipv4Addr) -> bool {
    if !identity.on_link(dst) {
        return false;
    }
    let host_bits = 32u32.saturating_sub(u32::from(identity.prefix_len()));
    if host_bits == 0 || host_bits >= 32 {
        return false;
    }
    let mask = (1u32 << host_bits) - 1;
    u32::from_be_bytes(dst.0) & mask == mask
}
