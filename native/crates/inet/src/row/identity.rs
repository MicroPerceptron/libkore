//! **The machine's L3 identity** — the one address the row owns, and what it
//! takes to reach anything off-link.
//!
//! Scarce and singular, which is why it belongs to the row rather than to a
//! client: there is one lease, one address, one default route. A client that
//! could set its own would be able to answer for a neighbour.

use kore_driver::net::flow::Ipv4Addr;

/// Why an identity was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityRefusal {
    /// The address is unusable as a station address — unspecified, broadcast,
    /// multicast, or loopback.
    NotAStation,
    /// A prefix length an IPv4 address cannot carry.
    BadPrefix,
    /// The gateway is not reachable on the configured prefix, so a default
    /// route through it could never resolve.
    GatewayOffLink,
}

/// What the row knows about where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    addr: Ipv4Addr,
    prefix_len: u8,
    gateway: Option<Ipv4Addr>,
    dns: Option<Ipv4Addr>,
}

impl Identity {
    /// Adopt an address and prefix, with an optional default gateway.
    ///
    /// The gateway is checked against the prefix here rather than at first
    /// use: a gateway off-link is a configuration that can never work, and
    /// finding that out when the first packet fails to route is finding out
    /// in the wrong place.
    pub fn new(
        addr: Ipv4Addr,
        prefix_len: u8,
        gateway: Option<Ipv4Addr>,
    ) -> Result<Self, IdentityRefusal> {
        if addr.is_unspecified() || addr.is_broadcast() || addr.is_multicast() || addr.0[0] == 127 {
            return Err(IdentityRefusal::NotAStation);
        }
        if prefix_len > 32 {
            return Err(IdentityRefusal::BadPrefix);
        }
        let me = Self {
            addr,
            prefix_len,
            gateway: None,
            dns: None,
        };
        if let Some(g) = gateway
            && !me.on_link(g)
        {
            return Err(IdentityRefusal::GatewayOffLink);
        }
        Ok(Self { gateway, ..me })
    }

    /// This station's address.
    pub fn addr(&self) -> Ipv4Addr {
        self.addr
    }

    /// The prefix length, in bits.
    pub fn prefix_len(&self) -> u8 {
        self.prefix_len
    }

    /// The default gateway, when one is configured.
    pub fn gateway(&self) -> Option<Ipv4Addr> {
        self.gateway
    }

    /// The resolver, when one is configured.
    pub fn dns(&self) -> Option<Ipv4Addr> {
        self.dns
    }

    /// Record a resolver.
    pub fn set_dns(&mut self, dns: Option<Ipv4Addr>) {
        self.dns = dns;
    }

    /// Whether `peer` is on this station's own link.
    pub fn on_link(&self, peer: Ipv4Addr) -> bool {
        let mask = match self.prefix_len {
            0 => 0u32,
            n => u32::MAX << (32 - n.min(32)),
        };
        u32::from_be_bytes(self.addr.0) & mask == u32::from_be_bytes(peer.0) & mask
    }

    /// The next hop for `dst`: the peer itself when on-link, otherwise the
    /// gateway.
    ///
    /// `None` means unreachable, which is a fact rather than a failure — a
    /// machine with no gateway genuinely cannot reach off-link, and saying so
    /// here beats discovering it as a timeout.
    pub fn next_hop(&self, dst: Ipv4Addr) -> Option<Ipv4Addr> {
        if self.on_link(dst) {
            Some(dst)
        } else {
            self.gateway
        }
    }
}
