//! **The DHCPv4 client** — how this station learns where it is.
//!
//! Four messages and one honest problem: every one of them has to travel
//! *before* the station has an address. DISCOVER goes out from `0.0.0.0` to
//! the all-ones broadcast, and the OFFER comes back to a station the server
//! cannot yet ARP for. So this module is the one place in the stack that
//! transmits without an identity, and the one place that accepts a datagram
//! addressed to the broadcast rather than to us.
//!
//! The state machine is written here rather than taken from smoltcp's
//! `socket-dhcpv4`, for the reason `res/INET_STACK.md` §5.1 gives: the lease
//! it produces becomes the row's [`Identity`], which is the row's to own. The
//! message *encoding* is borrowed (`vendor/smoltcp`).

use kore_driver::net::flow::Ipv4Addr;
use kore_driver::net::frame::HEADER_LEN;
use kore_driver::net::{EtherType, FrameHead, MacAddr};
use smoltcp::phy::ChecksumCapabilities;
use smoltcp::wire::{
    DHCP_CLIENT_PORT, DHCP_SERVER_PORT, DhcpMessageType, DhcpPacket, DhcpRepr, IpAddress,
    Ipv4Packet, Ipv4Repr, UdpPacket, UdpRepr,
};

use crate::ifnet::{Ifnet, TxSlot};
use crate::netinet::{from_ip, to_ip};

/// How long to wait for an answer before asking again, in milliseconds.
pub const DHCP_RETRY_MS: u64 = 2_000;

/// What the client is doing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DhcpState {
    /// Nothing asked yet.
    #[default]
    Idle,
    /// A DISCOVER is outstanding.
    Discovering,
    /// An offer was taken and a REQUEST is outstanding.
    Requesting {
        /// The address being asked for.
        offered: Ipv4Addr,
        /// The server that offered it.
        server: Ipv4Addr,
    },
    /// A lease is held.
    Bound(Lease),
}

/// What a server granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lease {
    /// This station's address.
    pub addr: Ipv4Addr,
    /// The prefix length, derived from the offered mask.
    pub prefix_len: u8,
    /// The default gateway, when one was offered.
    pub gateway: Option<Ipv4Addr>,
    /// The resolver, when one was offered.
    pub dns: Option<Ipv4Addr>,
    /// The server that granted it.
    pub server: Ipv4Addr,
}

/// The client.
#[derive(Debug, Default)]
pub struct DhcpClient {
    state: DhcpState,
    /// The transaction this client is in the middle of. Carried so a reply to
    /// a *previous* attempt — or to another station entirely — is refused
    /// rather than acted on.
    xid: u32,
    last_sent_ms: u64,
}

impl DhcpClient {
    /// A client that has asked for nothing.
    pub const fn new() -> Self {
        Self {
            state: DhcpState::Idle,
            xid: 0,
            last_sent_ms: 0,
        }
    }

    /// What the client is doing.
    pub fn state(&self) -> DhcpState {
        self.state
    }

    /// The lease, once one is held.
    pub fn lease(&self) -> Option<Lease> {
        match self.state {
            DhcpState::Bound(l) => Some(l),
            _ => None,
        }
    }

    /// Start over — what a link coming back up runs.
    pub fn reset(&mut self) {
        self.state = DhcpState::Idle;
        self.xid = 0;
        self.last_sent_ms = 0;
    }

    /// Drive the client: send whatever this state owes, if it is due.
    ///
    /// `xid_seed` supplies transaction-identifier entropy; a caller passes
    /// something that varies (a counter, a clock reading). The identifier only
    /// has to be unlikely to collide with a concurrent client on the segment,
    /// not unguessable.
    pub fn poll<I: Ifnet>(&mut self, net: &mut I, now_ms: u64, xid_seed: u32) -> bool {
        let due = self.last_sent_ms == 0 || now_ms.saturating_sub(self.last_sent_ms) >= DHCP_RETRY_MS;
        match self.state {
            DhcpState::Bound(_) => false,
            DhcpState::Idle => {
                self.xid = xid_seed;
                self.state = DhcpState::Discovering;
                self.last_sent_ms = now_ms;
                self.emit(net, DhcpMessageType::Discover, None, None)
            }
            DhcpState::Discovering if due => {
                self.last_sent_ms = now_ms;
                self.emit(net, DhcpMessageType::Discover, None, None)
            }
            DhcpState::Requesting { offered, server } if due => {
                self.last_sent_ms = now_ms;
                self.emit(net, DhcpMessageType::Request, Some(offered), Some(server))
            }
            _ => false,
        }
    }

    /// Build and send one message.
    ///
    /// Everything goes out broadcast, from the unspecified address: until the
    /// ACK lands this station has no address to send from and no translation
    /// for the server.
    fn emit<I: Ifnet>(
        &self,
        net: &mut I,
        kind: DhcpMessageType,
        requested_ip: Option<Ipv4Addr>,
        server_ip: Option<Ipv4Addr>,
    ) -> bool {
        let hw = net.hwaddr();
        let mut repr = DhcpRepr {
            message_type: kind,
            transaction_id: self.xid,
            secs: 0,
            client_hardware_address: smoltcp::wire::EthernetAddress(hw.0),
            client_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            your_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            server_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            router: None,
            subnet_mask: None,
            relay_agent_ip: to_ip(Ipv4Addr::UNSPECIFIED),
            broadcast: true,
            requested_ip: requested_ip.map(to_ip),
            client_identifier: Some(smoltcp::wire::EthernetAddress(hw.0)),
            server_identifier: server_ip.map(to_ip),
            parameter_request_list: None,
            dns_servers: None,
            max_size: None,
            lease_duration: None,
            renew_duration: None,
            rebind_duration: None,
            additional_options: &[],
        };
        repr.parameter_request_list = Some(&[1, 3, 6]); // mask, router, DNS

        let dhcp_len = repr.buffer_len();
        let udp = UdpRepr {
            src_port: DHCP_CLIENT_PORT,
            dst_port: DHCP_SERVER_PORT,
        };
        let udp_len = udp.header_len() + dhcp_len;
        let ip = Ipv4Repr {
            src_addr: to_ip(Ipv4Addr::UNSPECIFIED),
            dst_addr: to_ip(Ipv4Addr::BROADCAST),
            next_header: smoltcp::wire::IpProtocol::Udp,
            payload_len: udp_len,
            hop_limit: 64,
        };

        let reserve = HEADER_LEN + ip.buffer_len();
        let Some(mut slot) = net.reserve() else {
            return false;
        };
        let Some(body) = slot.reserve_headers(reserve) else {
            return false;
        };
        if body.len() < udp_len {
            return false;
        }
        {
            let mut packet = UdpPacket::new_unchecked(&mut body[..udp_len]);
            udp.emit(
                &mut packet,
                &IpAddress::Ipv4(ip.src_addr),
                &IpAddress::Ipv4(ip.dst_addr),
                dhcp_len,
                |buf| {
                    let mut d = DhcpPacket::new_unchecked(buf);
                    // The encoder can refuse a representation that will not
                    // fit; a half-written message must not reach the wire.
                    let _ = repr.emit(&mut d);
                },
                &ChecksumCapabilities::default(),
            );
        }
        let headers = slot.headers_mut();
        let head = FrameHead {
            dst: MacAddr::BROADCAST,
            src: hw,
            ethertype: EtherType::IPV4,
        };
        if head.write(&mut headers[..HEADER_LEN]).is_none() {
            return false;
        }
        {
            let mut p = Ipv4Packet::new_unchecked(&mut headers[HEADER_LEN..reserve]);
            ip.emit(&mut p, &ChecksumCapabilities::default());
        }
        slot.commit(udp_len, EtherType::IPV4).is_ok()
    }

    /// Feed one received DHCP payload to the client.
    ///
    /// Returns the lease when this message completed one. A message whose
    /// transaction identifier is not ours is ignored — on a shared segment
    /// every client sees every broadcast, and acting on a neighbour's ACK
    /// would take its address.
    pub fn input(&mut self, payload: &[u8], hw: MacAddr) -> Option<Lease> {
        let packet = DhcpPacket::new_checked(payload).ok()?;
        let repr = DhcpRepr::parse(&packet).ok()?;

        if repr.transaction_id != self.xid {
            return None;
        }
        // And the message must name *this* station's hardware address: the
        // transaction identifier alone is a 32-bit number a neighbour could
        // collide with by accident.
        if repr.client_hardware_address.0 != hw.0 {
            return None;
        }

        match (self.state, repr.message_type) {
            (DhcpState::Discovering, DhcpMessageType::Offer) => {
                let server = repr
                    .server_identifier
                    .map(from_ip)
                    .unwrap_or_else(|| from_ip(repr.server_ip));
                self.state = DhcpState::Requesting {
                    offered: from_ip(repr.your_ip),
                    server,
                };
                // Send the REQUEST on the next poll rather than from here, so
                // every transmit in this module goes through one path.
                self.last_sent_ms = 0;
                None
            }
            (DhcpState::Requesting { offered, .. }, DhcpMessageType::Ack) => {
                let addr = from_ip(repr.your_ip);
                // The ACK must grant what was asked for. A server that
                // answers with a different address has not completed this
                // exchange, and taking it would bind an address nobody
                // offered.
                if addr != offered {
                    return None;
                }
                let lease = Lease {
                    addr,
                    prefix_len: repr
                        .subnet_mask
                        .map(|m| u32::from_be_bytes(m.octets()).count_ones() as u8)
                        .unwrap_or(24),
                    gateway: repr.router.map(from_ip),
                    dns: repr
                        .dns_servers
                        .as_ref()
                        .and_then(|s| s.first().copied())
                        .map(from_ip),
                    server: repr
                        .server_identifier
                        .map(from_ip)
                        .unwrap_or_else(|| from_ip(repr.server_ip)),
                };
                self.state = DhcpState::Bound(lease);
                Some(lease)
            }
            (DhcpState::Requesting { .. }, DhcpMessageType::Nak) => {
                // Refused: start over rather than retrying a request the
                // server has already declined.
                self.reset();
                None
            }
            _ => None,
        }
    }
}
