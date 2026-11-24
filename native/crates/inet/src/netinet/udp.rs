//! **UDP** — the second row of the switch, and the one DHCP and DNS ride.
//!
//! The datagram itself is almost nothing: eight bytes of header and a
//! checksum. What matters here is the **demultiplex**, and it is deliberately
//! the same shape as the one a layer down — a bounded table of bindings, each
//! owned, matched by the tuple rather than by a scan of every listener's
//! guesses.

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use smoltcp::phy::ChecksumCapabilities;
use smoltcp::wire::{IpAddress, UdpPacket, UdpRepr};

use crate::ifnet::Ifnet;
use crate::netinet::arp::ArpCache;
use crate::netinet::ip::{self, OutputError};
use crate::netinet::to_ip;
use crate::row::Identity;

/// Local ports this stack will deliver to at once.
///
/// A design bound sizing an inline array. Four is enough for the stack's own
/// business — a DHCP client and a resolver — plus a caller's own listener;
/// a composition that needs more raises this deliberately.
pub const MAX_UDP_BINDINGS: usize = 4;

/// A received datagram, delivered to whoever bound its port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UdpDatagram<'a> {
    /// Who sent it.
    pub src: (Ipv4Addr, u16),
    /// The local port it was addressed to.
    pub local_port: u16,
    /// The payload.
    pub payload: &'a [u8],
}

/// Who is listening on which local port.
///
/// Deliberately *not* a general socket table: this is the stack's own
/// demultiplex, and the authority question — may this caller bind this
/// port — belongs to [`InetRow`](crate::row::InetRow), which answered it
/// before anything got here.
#[derive(Debug, Default)]
pub struct UdpBindings {
    ports: [Option<u16>; MAX_UDP_BINDINGS],
}

impl UdpBindings {
    /// Nothing bound.
    pub const fn new() -> Self {
        Self {
            ports: [None; MAX_UDP_BINDINGS],
        }
    }

    /// Start delivering `port`.
    ///
    /// Idempotent, and `false` only when the table is full — the *ownership*
    /// question was settled upstream, so a repeat here is a composition
    /// re-registering rather than a collision.
    pub fn bind(&mut self, port: u16) -> bool {
        if self.is_bound(port) {
            return true;
        }
        if let Some(slot) = self.ports.iter_mut().find(|s| s.is_none()) {
            *slot = Some(port);
            return true;
        }
        false
    }

    /// Stop delivering `port`.
    pub fn unbind(&mut self, port: u16) {
        for slot in self.ports.iter_mut() {
            if *slot == Some(port) {
                *slot = None;
            }
        }
    }

    /// Whether `port` is delivered.
    pub fn is_bound(&self, port: u16) -> bool {
        self.ports.iter().flatten().any(|p| *p == port)
    }

    /// How many ports are bound.
    pub fn len(&self) -> usize {
        self.ports.iter().flatten().count()
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Send a datagram.
pub fn output<I: Ifnet>(
    net: &mut I,
    cache: &mut ArpCache,
    identity: &Identity,
    src_port: u16,
    dst: (Ipv4Addr, u16),
    payload: &[u8],
    now_ms: u64,
) -> Result<(), OutputError> {
    let repr = UdpRepr {
        src_port,
        dst_port: dst.1,
    };
    let len = repr.header_len() + payload.len();
    let src_addr = IpAddress::Ipv4(to_ip(identity.addr()));
    let dst_addr = IpAddress::Ipv4(to_ip(dst.0));
    ip::output(
        net,
        cache,
        identity,
        dst.0,
        IpProto::UDP,
        len,
        now_ms,
        |body| {
            let mut packet = UdpPacket::new_unchecked(&mut body[..len]);
            repr.emit(
                &mut packet,
                &src_addr,
                &dst_addr,
                payload.len(),
                |buf| buf.copy_from_slice(payload),
                &ChecksumCapabilities::default(),
            );
        },
    )
}

/// Classify one received UDP datagram.
///
/// `None` when it is malformed, or when nothing is bound to its port. The
/// second case is a real answer rather than a silent drop: a datagram to an
/// unbound port is the ordinary state of a shared link, and the caller above
/// decides whether that deserves an ICMP unreachable.
pub fn input<'a>(
    bindings: &UdpBindings,
    src: Ipv4Addr,
    dst: Ipv4Addr,
    payload: &'a [u8],
) -> Option<UdpDatagram<'a>> {
    let packet = UdpPacket::new_checked(payload).ok()?;
    let src_addr = IpAddress::Ipv4(to_ip(src));
    let dst_addr = IpAddress::Ipv4(to_ip(dst));
    // Verified against the pseudo-header, so a datagram that was corrupted in
    // transit — or addressed to a different station and reflected here — does
    // not reach a binding.
    let repr = UdpRepr::parse(
        &packet,
        &src_addr,
        &dst_addr,
        &ChecksumCapabilities::default(),
    )
    .ok()?;
    if !bindings.is_bound(repr.dst_port) {
        return None;
    }
    Some(UdpDatagram {
        src: (src, repr.src_port),
        local_port: repr.dst_port,
        payload: packet.payload(),
    })
}
