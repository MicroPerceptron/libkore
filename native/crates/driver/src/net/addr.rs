//! **Ethernet addressing** — the two wire scalars every L2 frame carries.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// A 48-bit Ethernet address, in wire order.
///
/// Wire order matters and is not decoration: an Ethernet address is
/// transmitted most-significant octet first, so the array *is* the wire, and
/// there is deliberately no integer conversion that would invite a caller to
/// think about endianness.
#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct MacAddr(pub [u8; 6]);

impl MacAddr {
    /// The all-ones broadcast address.
    pub const BROADCAST: Self = Self([0xff; 6]);
    /// The all-zero address — never valid as a source or destination, and the
    /// value an unconfigured device reports.
    pub const ZERO: Self = Self([0; 6]);

    /// Whether the group bit (the low bit of the first octet) is set — true
    /// for every multicast address, and for broadcast.
    pub const fn is_multicast(self) -> bool {
        self.0[0] & 0x01 != 0
    }

    /// Whether this is the broadcast address specifically.
    pub const fn is_broadcast(self) -> bool {
        matches!(self.0, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff])
    }

    /// Whether this addresses exactly one station.
    pub const fn is_unicast(self) -> bool {
        !self.is_multicast()
    }

    /// Whether the locally-administered bit (bit 1 of the first octet) is set.
    pub const fn is_local(self) -> bool {
        self.0[0] & 0x02 != 0
    }

    /// Whether this address is usable as a frame's source.
    pub const fn is_valid_source(self) -> bool {
        self.is_unicast() && !matches!(self.0, [0, 0, 0, 0, 0, 0])
    }

    /// Derive a **locally-administered** unicast address for vif `index` from
    /// this one.
    ///
    /// Per-vif MAC identity is what lets one physical device present several
    /// independent stations (#388's peer vif) and what makes address
    /// randomization a vif property rather than a device-wide mode (#387
    /// §3.2).  The derived address sets the locally-administered bit and
    /// clears the group bit, so it can never collide with a manufacturer's
    /// globally-unique assignment nor be mistaken for a multicast group.
    ///
    /// Whether frames sent from a derived address actually *reach* the network
    /// is a property of the upstream port, not of this function — some
    /// switches and access points refuse a source address they did not learn.
    pub const fn derive_local(self, index: u8) -> Self {
        let mut octets = self.0;
        octets[0] = (octets[0] | 0x02) & !0x01;
        // Vary the last octet so sibling vifs are distinct stations; the top
        // five octets keep the device's identity, which is what makes a
        // derived address recognizable in a capture.
        octets[5] ^= index;
        Self(octets)
    }
}

/// An EtherType — the 16-bit protocol selector in an Ethernet II header.
///
/// Stored host-native; [`from_wire`](Self::from_wire) and
/// [`to_wire`](Self::to_wire) are the only places byte order is decided, so a
/// caller never has to.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EtherType(pub u16);

impl EtherType {
    /// IPv4.
    pub const IPV4: Self = Self(0x0800);
    /// Address Resolution Protocol.
    pub const ARP: Self = Self(0x0806);
    /// IPv6.
    pub const IPV6: Self = Self(0x86dd);
    /// 802.1Q VLAN-tagged frame.
    pub const VLAN: Self = Self(0x8100);

    /// kore's peer link (#387).
    ///
    /// `0x88B5` is IEEE 802's **Local Experimental EtherType 1**, reserved by
    /// the standard for exactly this — a private protocol on a local network
    /// — so it is not squatting on an assignment that could later collide.
    pub const KORE_PEER: Self = Self(0x88b5);

    /// The smallest value that is an EtherType rather than an 802.3 length
    /// field.  A header whose type field is below this is a legacy
    /// length-prefixed frame, not Ethernet II.
    pub const MIN: Self = Self(0x0600);

    /// Decode from the two wire octets (big-endian, as on the wire).
    pub const fn from_wire(b: [u8; 2]) -> Self {
        Self(u16::from_be_bytes(b))
    }

    /// Encode to the two wire octets.
    pub const fn to_wire(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }

    /// Whether this is an Ethernet II type rather than an 802.3 length.
    pub const fn is_ethertype(self) -> bool {
        self.0 >= Self::MIN.0
    }
}
