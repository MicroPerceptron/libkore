//! **ARP** — the one thing that must happen before any IPv4 packet reaches
//! anyone.
//!
//! Ethernet addresses frames by a 48-bit station address; IP addresses packets
//! by a 32-bit one. Nothing crosses that gap by itself, so every datagram —
//! including the first one to the gateway — waits on a translation this module
//! either has cached or must ask for.
//!
//! Written here rather than taken from smoltcp's interface: the cache is a
//! capability-shaped, design-bounded table with an injected clock, and its
//! eviction and refresh policy is ours. Only the *wire format* is borrowed
//! (`res/INET_STACK.md` §5.1).

use kore_driver::net::{EtherType, MacAddr};
use kore_driver::net::flow::Ipv4Addr;
use smoltcp::wire::{ArpOperation, ArpPacket, ArpRepr, EthernetAddress};

use crate::ifnet::{Ifnet, TxSlot};
use crate::netinet::{to_eth, to_ip, from_eth, from_ip};

/// Translations held at once.
///
/// A design bound, not a tuning knob: it sizes an inline array. Sixteen is
/// comfortably more than the handful of stations a client actually talks to
/// (its gateway, its resolver, a peer or two), and small enough that the
/// linear scan beats anything with a hash.
pub const ARP_CACHE_ENTRIES: usize = 16;

/// How long a learned translation stays good, in milliseconds.
///
/// One minute, the same figure smoltcp and most hosts use. Long enough that a
/// conversation does not re-ask, short enough that a station which moved is
/// forgotten before its stale entry outlives anyone's patience.
pub const ARP_ENTRY_LIFETIME_MS: u64 = 60_000;

/// How often an unanswered request may be repeated.
///
/// One per second per target. The bound matters: without it a stream of
/// packets to an unreachable address becomes a stream of broadcasts, which is
/// a denial of service a client can trigger by accident.
pub const ARP_RETRY_INTERVAL_MS: u64 = 1_000;

#[derive(Clone, Copy, Debug)]
struct Entry {
    ip: Ipv4Addr,
    mac: MacAddr,
    /// When this translation stops being trusted.
    expires_ms: u64,
}

/// What a lookup produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The translation is known.
    Known(MacAddr),
    /// Not known; a request was just sent and the caller should retry.
    Asked,
    /// Not known, and a request went out too recently to repeat.
    Pending,
}

/// The translation table.
#[derive(Debug)]
pub struct ArpCache {
    entries: [Option<Entry>; ARP_CACHE_ENTRIES],
    /// The last target asked for, and when — the rate limit's whole state.
    last_ask: Option<(Ipv4Addr, u64)>,
}

impl Default for ArpCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ArpCache {
    /// An empty cache.
    pub const fn new() -> Self {
        Self {
            entries: [None; ARP_CACHE_ENTRIES],
            last_ask: None,
        }
    }

    /// Translations currently held.
    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up `ip`, honouring expiry.
    pub fn get(&self, ip: Ipv4Addr, now_ms: u64) -> Option<MacAddr> {
        self.entries
            .iter()
            .flatten()
            .find(|e| e.ip == ip && e.expires_ms > now_ms)
            .map(|e| e.mac)
    }

    /// Learn a translation.
    ///
    /// Learning is **not** the same as trusting a stranger: this is called
    /// from the receive path only for a packet addressed to this station, so
    /// an unsolicited reply from a third party never lands here. Overwriting
    /// an existing entry is deliberate — a station that legitimately changed
    /// address must be followed, and the expiry is what bounds the damage if
    /// the new mapping was a lie.
    pub fn insert(&mut self, ip: Ipv4Addr, mac: MacAddr, now_ms: u64) {
        let expires_ms = now_ms.saturating_add(ARP_ENTRY_LIFETIME_MS);
        if let Some(slot) = self.entries.iter_mut().flatten().find(|e| e.ip == ip) {
            slot.mac = mac;
            slot.expires_ms = expires_ms;
            return;
        }
        let fresh = Entry {
            ip,
            mac,
            expires_ms,
        };
        // A free slot, else the one closest to expiry — never a live entry
        // chosen at random, which would evict the gateway as readily as a
        // stranger.
        if let Some(slot) = self.entries.iter_mut().find(|s| {
            s.is_none_or(|e| e.expires_ms <= now_ms)
        }) {
            *slot = Some(fresh);
            return;
        }
        if let Some(slot) = self
            .entries
            .iter_mut()
            .flatten()
            .min_by_key(|e| e.expires_ms)
        {
            *slot = fresh;
        }
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.entries = [None; ARP_CACHE_ENTRIES];
        self.last_ask = None;
    }

    /// Whether a request for `target` may go out now.
    fn may_ask(&self, target: Ipv4Addr, now_ms: u64) -> bool {
        match self.last_ask {
            Some((ip, at)) if ip == target => now_ms.saturating_sub(at) >= ARP_RETRY_INTERVAL_MS,
            _ => true,
        }
    }
}

/// Build an ARP frame into a transmit slot and publish it.
fn emit<I: Ifnet>(
    net: &mut I,
    dst: MacAddr,
    repr: ArpRepr,
) -> bool {
    // Read before reserving: the slot borrows the interface mutably for its
    // whole life, so the station address has to be in hand first.
    let src = net.hwaddr();
    let Some(mut slot) = net.reserve() else {
        return false;
    };
    let len = repr.buffer_len();
    let Some(body) = slot.reserve_headers(kore_driver::net::frame::HEADER_LEN) else {
        return false;
    };
    if body.len() < len {
        return false;
    }
    {
        let Ok(mut packet) = ArpPacket::new_checked(&mut body[..len]) else {
            return false;
        };
        repr.emit(&mut packet);
    }
    kore_driver::net::FrameHead {
        dst,
        src,
        ethertype: EtherType::ARP,
    }
    .write(slot.headers_mut())
    .is_some()
        && slot.commit(len, EtherType::ARP).is_ok()
}

/// Resolve `target`, asking on the wire when the answer is not cached.
///
/// `mine` is this station's address, which an ARP request must carry: a
/// request with an unspecified sender is a probe, and answering one is a
/// different conversation than this.
pub fn resolve<I: Ifnet>(
    cache: &mut ArpCache,
    net: &mut I,
    mine: Ipv4Addr,
    target: Ipv4Addr,
    now_ms: u64,
) -> Resolution {
    if let Some(mac) = cache.get(target, now_ms) {
        return Resolution::Known(mac);
    }
    if !cache.may_ask(target, now_ms) {
        return Resolution::Pending;
    }
    let repr = ArpRepr::EthernetIpv4 {
        operation: ArpOperation::Request,
        source_hardware_addr: to_eth(net.hwaddr()),
        source_protocol_addr: to_ip(mine),
        // The field a request leaves empty: it is what is being asked for.
        target_hardware_addr: EthernetAddress::from_bytes(&[0; 6]),
        target_protocol_addr: to_ip(target),
    };
    if emit(net, MacAddr::BROADCAST, repr) {
        cache.last_ask = Some((target, now_ms));
        Resolution::Asked
    } else {
        Resolution::Pending
    }
}

/// Handle one received ARP frame.
///
/// Returns whether a reply was sent. `mine` is this station's address; a
/// request for any other address is **not** answered, because answering for a
/// neighbour is exactly the attack ARP is famous for.
pub fn input<I: Ifnet>(
    cache: &mut ArpCache,
    net: &mut I,
    mine: Ipv4Addr,
    frame_body: &[u8],
    now_ms: u64,
) -> bool {
    let Ok(packet) = ArpPacket::new_checked(frame_body) else {
        return false;
    };
    let Ok(ArpRepr::EthernetIpv4 {
        operation,
        source_hardware_addr,
        source_protocol_addr,
        target_protocol_addr,
        ..
    }) = ArpRepr::parse(&packet)
    else {
        return false;
    };

    let sender_ip = from_ip(source_protocol_addr);
    let sender_mac = from_eth(source_hardware_addr);

    // Learn only from traffic that concerns this station. A request broadcast
    // to the segment for somebody else teaches nothing: taking it would let
    // any station populate every cache on the wire at will.
    let concerns_me = from_ip(target_protocol_addr) == mine;
    if concerns_me && sender_mac.is_unicast() && !sender_ip.is_unspecified() {
        cache.insert(sender_ip, sender_mac, now_ms);
    }

    if operation != ArpOperation::Request || !concerns_me {
        return false;
    }
    let reply = ArpRepr::EthernetIpv4 {
        operation: ArpOperation::Reply,
        source_hardware_addr: to_eth(net.hwaddr()),
        source_protocol_addr: to_ip(mine),
        target_hardware_addr: source_hardware_addr,
        target_protocol_addr: source_protocol_addr,
    };
    emit(net, sender_mac, reply)
}
