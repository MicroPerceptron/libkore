//! **CDC Ethernet spec facts** — the constants, request encodings, and
//! notification layout of the USB Communications Device Class's Ethernet
//! Control Model.
//!
//! Transcribed from the specification, not recalled. Every value here is a
//! wire encoding: a plausible-looking wrong one produces a device that
//! enumerates cleanly and then behaves subtly wrongly, which is the most
//! expensive kind of bug to find on real hardware.

use kore_driver::net::{MacAddr, PacketFilter};

/// `bInterfaceClass` 0x02 — Communications, the ECM *control* interface.
pub const CLASS_COMM: u8 = 0x02;
/// `bInterfaceSubClass` 0x06 — Ethernet Control Model.
pub const SUBCLASS_ECM: u8 = 0x06;
/// `bInterfaceClass` 0x0A — CDC Data, the interface carrying the bulk pair.
pub const CLASS_CDC_DATA: u8 = 0x0a;

/// `bDescriptorType` 0x24 — a class-specific interface (functional)
/// descriptor.
pub const CS_INTERFACE: u8 = 0x24;

/// Functional-descriptor subtypes this driver reads.
pub mod subtype {
    /// Header functional descriptor.
    pub const HEADER: u8 = 0x00;
    /// Union functional descriptor — names the control and data interfaces
    /// that belong to one function.
    pub const UNION: u8 = 0x06;
    /// Ethernet Networking functional descriptor — carries the MAC address
    /// string index and the maximum segment size.
    pub const ETHERNET: u8 = 0x0f;
}

/// Field offsets **within a functional descriptor's payload**, i.e. after the
/// three-byte `bLength ‖ bDescriptorType ‖ bDescriptorSubtype` header.
pub mod ethernet_fd {
    /// `iMACAddress` — the string index of the twelve-hex-digit MAC.
    pub const I_MAC_ADDRESS: usize = 0;
    /// `wMaxSegmentSize`, little-endian.
    pub const W_MAX_SEGMENT_SIZE: usize = 5;
    /// `bLength` of a well-formed Ethernet Networking functional descriptor.
    pub const LEN: u8 = 13;
}

/// Field offsets within a Union functional descriptor's payload.
pub mod union_fd {
    /// `bControlInterface`.
    pub const CONTROL: usize = 0;
    /// `bSubordinateInterface0` — the data interface.
    pub const SUBORDINATE: usize = 1;
}

// ── SetEthernetPacketFilter ────────────────────────────────────────────

/// `bmRequestType` for a host-to-device class request to an interface.
pub const REQ_CLASS_INTERFACE_OUT: u8 = 0x21;
/// `bRequest` — `SetEthernetPacketFilter`.
pub const SET_ETHERNET_PACKET_FILTER: u8 = 0x43;

/// The `wValue` bitmap of `SetEthernetPacketFilter`.
pub mod packet_type {
    /// D0 — every frame, regardless of destination.
    pub const PROMISCUOUS: u16 = 1 << 0;
    /// D1 — every multicast group.
    pub const ALL_MULTICAST: u16 = 1 << 1;
    /// D2 — frames addressed to the device's own unicast address.
    pub const DIRECTED: u16 = 1 << 2;
    /// D3 — broadcast.
    pub const BROADCAST: u16 = 1 << 3;
    /// D4 — the specifically-joined multicast groups.
    pub const MULTICAST: u16 = 1 << 4;
}

/// Encode a neutral [`PacketFilter`] into the class's `wValue` bitmap.
///
/// The class deliberately publishes **no descriptor** saying which bits a
/// device honours, and the specification's own guidance is that the host asks
/// for what it needs and does any filtering the device ignores itself. So a
/// bit set here is a request, never a guarantee — which is exactly why the
/// frame service's classifier is authoritative and not a fallback.
pub fn encode_packet_filter(filter: PacketFilter) -> u16 {
    let mut bits = 0;
    if filter.promiscuous {
        bits |= packet_type::PROMISCUOUS;
    }
    if filter.all_multicast {
        bits |= packet_type::ALL_MULTICAST;
    }
    if filter.directed {
        bits |= packet_type::DIRECTED;
    }
    if filter.broadcast {
        bits |= packet_type::BROADCAST;
    }
    if filter.multicast {
        bits |= packet_type::MULTICAST;
    }
    bits
}

// ── Notifications (the interrupt IN endpoint) ──────────────────────────

/// `bmRequestType` every class notification carries.
pub const NOTIFY_REQUEST_TYPE: u8 = 0xa1;
/// Bytes of notification header before any payload.
pub const NOTIFY_HEADER_LEN: usize = 8;

/// `bNotificationCode` values this driver acts on.
pub mod notification {
    /// Link up / down; `wValue` is 1 for connected, 0 for disconnected.
    pub const NETWORK_CONNECTION: u8 = 0x00;
    /// Rate change; the 8-byte payload is `DLBitRate ‖ ULBitRate`, u32 LE,
    /// in bits per second.
    pub const CONNECTION_SPEED_CHANGE: u8 = 0x2a;
}

/// A decoded class notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notification {
    /// The link came up or went down.
    NetworkConnection {
        /// Whether a carrier is present.
        connected: bool,
    },
    /// The link rate changed.
    SpeedChange {
        /// Downlink rate, bits per second.
        downlink: u32,
        /// Uplink rate, bits per second.
        uplink: u32,
    },
}

impl Notification {
    /// Decode one notification, or `None` when the bytes are not a
    /// well-formed one this driver acts on.
    ///
    /// Refused rather than guessed at: a device is free to send
    /// notifications this driver does not model, and inventing a link state
    /// from an unrecognized code would flap the interface for no reason.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < NOTIFY_HEADER_LEN || bytes[0] != NOTIFY_REQUEST_TYPE {
            return None;
        }
        let value = u16::from_le_bytes([bytes[2], bytes[3]]);
        let length = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        match bytes[1] {
            notification::NETWORK_CONNECTION => Some(Self::NetworkConnection {
                connected: value != 0,
            }),
            notification::CONNECTION_SPEED_CHANGE => {
                // The payload is mandatory for this code; a short one is a
                // malformed notification, not a zero-rate link.
                if length < 8 || bytes.len() < NOTIFY_HEADER_LEN + 8 {
                    return None;
                }
                let p = &bytes[NOTIFY_HEADER_LEN..];
                Some(Self::SpeedChange {
                    downlink: u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                    uplink: u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
                })
            }
            _ => None,
        }
    }
}

// ── The MAC address string ─────────────────────────────────────────────

/// ASCII hex digits a CDC MAC address string carries.
pub const MAC_HEX_DIGITS: usize = 12;

/// Decode a MAC address from the twelve ASCII hex digits the class's
/// `iMACAddress` string holds.
///
/// Refuses anything that is not exactly twelve hex digits. A device whose
/// MAC string is malformed has no usable address, and substituting a
/// plausible one would put a station on the wire under an identity nobody
/// chose.
pub fn mac_from_hex(ascii: &[u8]) -> Option<MacAddr> {
    if ascii.len() != MAC_HEX_DIGITS {
        return None;
    }
    let mut mac = [0u8; 6];
    for (i, byte) in mac.iter_mut().enumerate() {
        let hi = hex_nibble(ascii[i * 2])?;
        let lo = hex_nibble(ascii[i * 2 + 1])?;
        *byte = (hi << 4) | lo;
    }
    Some(MacAddr(mac))
}

const fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bitmap is a wire encoding: each neutral flag maps to exactly one
    /// specified bit, and nothing else is set.
    #[test]
    fn the_packet_filter_bitmap_matches_the_specification() {
        assert_eq!(encode_packet_filter(PacketFilter::default()), 0);
        assert_eq!(
            encode_packet_filter(PacketFilter::DIRECTED),
            packet_type::DIRECTED | packet_type::BROADCAST
        );
        assert_eq!(
            encode_packet_filter(PacketFilter::PROMISCUOUS),
            packet_type::PROMISCUOUS
                | packet_type::ALL_MULTICAST
                | packet_type::DIRECTED
                | packet_type::BROADCAST
                | packet_type::MULTICAST
        );

        // Each flag independently, so a transposed bit cannot hide behind a
        // combination that happens to add up.
        let one = |f: PacketFilter| encode_packet_filter(f);
        assert_eq!(
            one(PacketFilter {
                promiscuous: true,
                ..Default::default()
            }),
            1 << 0
        );
        assert_eq!(
            one(PacketFilter {
                all_multicast: true,
                ..Default::default()
            }),
            1 << 1
        );
        assert_eq!(
            one(PacketFilter {
                directed: true,
                ..Default::default()
            }),
            1 << 2
        );
        assert_eq!(
            one(PacketFilter {
                broadcast: true,
                ..Default::default()
            }),
            1 << 3
        );
        assert_eq!(
            one(PacketFilter {
                multicast: true,
                ..Default::default()
            }),
            1 << 4
        );
    }

    #[test]
    fn network_connection_notifications_decode() {
        let up = [NOTIFY_REQUEST_TYPE, 0x00, 1, 0, 0, 0, 0, 0];
        assert_eq!(
            Notification::parse(&up),
            Some(Notification::NetworkConnection { connected: true })
        );
        let down = [NOTIFY_REQUEST_TYPE, 0x00, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            Notification::parse(&down),
            Some(Notification::NetworkConnection { connected: false })
        );
    }

    #[test]
    fn speed_change_notifications_decode_both_rates() {
        // 1 Gb/s down, 100 Mb/s up, little-endian.
        let mut n = [0u8; 16];
        n[0] = NOTIFY_REQUEST_TYPE;
        n[1] = notification::CONNECTION_SPEED_CHANGE;
        n[6] = 8; // wLength
        n[8..12].copy_from_slice(&1_000_000_000u32.to_le_bytes());
        n[12..16].copy_from_slice(&100_000_000u32.to_le_bytes());
        assert_eq!(
            Notification::parse(&n),
            Some(Notification::SpeedChange {
                downlink: 1_000_000_000,
                uplink: 100_000_000,
            })
        );
    }

    /// Anything malformed, or any code this driver does not model, refuses.
    /// Inventing a link state from an unrecognized notification would flap
    /// the interface for no reason.
    #[test]
    fn malformed_and_unmodelled_notifications_refuse() {
        assert_eq!(Notification::parse(&[]), None);
        assert_eq!(Notification::parse(&[NOTIFY_REQUEST_TYPE, 0, 1, 0]), None);
        // Wrong bmRequestType.
        assert_eq!(Notification::parse(&[0x21, 0x00, 1, 0, 0, 0, 0, 0]), None);
        // A code this driver does not act on.
        assert_eq!(
            Notification::parse(&[NOTIFY_REQUEST_TYPE, 0x99, 1, 0, 0, 0, 0, 0]),
            None
        );
        // Speed change whose payload is absent — a malformed notification,
        // not a zero-rate link.
        let mut short = [0u8; 8];
        short[0] = NOTIFY_REQUEST_TYPE;
        short[1] = notification::CONNECTION_SPEED_CHANGE;
        short[6] = 8;
        assert_eq!(Notification::parse(&short), None);
    }

    /// The MAC arrives as twelve ASCII hex digits; anything else is refused,
    /// because a substituted address puts a station on the wire under an
    /// identity nobody chose.
    #[test]
    fn mac_strings_decode_or_refuse() {
        assert_eq!(
            mac_from_hex(b"400102030405"),
            Some(MacAddr([0x40, 0x01, 0x02, 0x03, 0x04, 0x05]))
        );
        // Case-insensitive, as hex is.
        assert_eq!(mac_from_hex(b"0A1B2C3D4E5F"), mac_from_hex(b"0a1b2c3d4e5f"));

        assert_eq!(mac_from_hex(b""), None);
        assert_eq!(mac_from_hex(b"40010203040"), None, "eleven digits");
        assert_eq!(mac_from_hex(b"4001020304050"), None, "thirteen");
        assert_eq!(mac_from_hex(b"40010203040g"), None, "not hex");
        assert_eq!(mac_from_hex(b"4001020304 5"), None, "a space is not zero");
    }
}
