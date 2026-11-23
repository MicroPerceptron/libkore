//! ARP vocabulary for the founding self-test and for connected flows that
//! need to resolve their authorized next hop.
//!
//! Two things: building a request, and reading what came back. Reading is
//! a classification, not a parser — the self-test needs to know whether a
//! frame is a reply (its round trip), a request (a station it can ask for
//! its own address, on a network where nobody answers for QEMU's gateway),
//! or neither.

use kore_driver::net::{EtherType, FrameHead, MacAddr};

/// Bytes of one ARP frame for IPv4 over Ethernet: the 14-byte header and
/// the 28-byte ARP body (hardware and protocol types, the two lengths, the
/// operation, sender and target hardware and protocol addresses). RFC 826.
pub const FRAME_LEN: usize = 14 + 28;

/// Where the fields sit in the frame: the header, then hardware type (2),
/// protocol type (2), hardware length (1), protocol length (1).
const OP_AT: usize = 14 + 2 + 2 + 1 + 1;
const SENDER_MAC_AT: usize = OP_AT + 2;
const SENDER_IP_AT: usize = SENDER_MAC_AT + 6;
const TARGET_MAC_AT: usize = SENDER_IP_AT + 4;
const TARGET_IP_AT: usize = TARGET_MAC_AT + 6;
const _: () = assert!(TARGET_IP_AT + 4 == FRAME_LEN);

/// The operation codes, big-endian on the wire.
const OP_REQUEST: [u8; 2] = 1u16.to_be_bytes();
const OP_REPLY: [u8; 2] = 2u16.to_be_bytes();

/// ARP traffic a connected flow may exchange with its next hop. The service
/// derives both addresses from the station identity and the admitted peer.
pub(crate) struct Neighbor {
    pub(crate) local: [u8; 4],
    pub(crate) next_hop: [u8; 4],
}

impl Neighbor {
    pub(crate) fn accepts_request(&self, head: &FrameHead, frame: &[u8], mac: MacAddr) -> bool {
        head.src == mac
            && head.dst.is_broadcast()
            && valid_header(frame, OP_REQUEST)
            && frame[SENDER_MAC_AT..SENDER_IP_AT] == mac.0
            && frame[SENDER_IP_AT..TARGET_MAC_AT] == self.local
            && frame[TARGET_MAC_AT..TARGET_IP_AT] == [0; 6]
            && frame[TARGET_IP_AT..FRAME_LEN] == self.next_hop
    }

    pub(crate) fn accepts_reply(&self, head: &FrameHead, frame: &[u8], mac: MacAddr) -> bool {
        head.dst == mac
            && head.src.is_valid_source()
            && valid_header(frame, OP_REPLY)
            && frame[SENDER_MAC_AT..SENDER_IP_AT] == head.src.0
            && frame[SENDER_IP_AT..TARGET_MAC_AT] == self.next_hop
            && frame[TARGET_MAC_AT..TARGET_IP_AT] == mac.0
            && frame[TARGET_IP_AT..FRAME_LEN] == self.local
    }
}

fn valid_header(frame: &[u8], operation: [u8; 2]) -> bool {
    frame.len() >= FRAME_LEN
        && frame[14..OP_AT] == [0, 1, 0x08, 0, 6, 4]
        && frame[OP_AT..SENDER_MAC_AT] == operation
}

/// One ARP request from `mac` at `sender_ip`, addressed to `dst` on the
/// wire (broadcast, or one station), asking who holds `target_ip`.
pub fn request(
    mac: MacAddr,
    sender_ip: [u8; 4],
    dst: MacAddr,
    target_ip: [u8; 4],
) -> [u8; FRAME_LEN] {
    let mut frame = [0u8; FRAME_LEN];
    FrameHead {
        dst,
        src: mac,
        ethertype: EtherType::ARP,
    }
    .write(&mut frame)
    .expect("the header fits");
    let arp = &mut frame[14..];
    arp[0..2].copy_from_slice(&1u16.to_be_bytes()); // hardware: Ethernet
    arp[2..4].copy_from_slice(&EtherType::IPV4.to_wire()); // protocol: IPv4
    arp[4] = 6; // hardware address length
    arp[5] = 4; // protocol address length
    frame[OP_AT..OP_AT + 2].copy_from_slice(&OP_REQUEST);
    frame[SENDER_MAC_AT..SENDER_MAC_AT + 6].copy_from_slice(&mac.0);
    frame[SENDER_IP_AT..SENDER_IP_AT + 4].copy_from_slice(&sender_ip);
    // The target hardware address stays zero — that is what is being asked.
    frame[TARGET_IP_AT..TARGET_IP_AT + 4].copy_from_slice(&target_ip);
    frame
}

/// What one frame the ARP vif delivered turned out to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heard {
    /// An ARP reply: `station` says it holds `ip`. The self-test's round
    /// trip, and the burst's responder.
    Reply {
        /// The replying station's hardware address.
        station: MacAddr,
        /// The protocol address it holds.
        ip: [u8; 4],
    },
    /// An ARP request from `station`, which named `ip` as its own. On a
    /// network where nobody answers for QEMU's gateway, a station heard
    /// asking is a station that can be asked for its own address — every
    /// host answers that.
    Request {
        /// The asking station's hardware address.
        station: MacAddr,
        /// The protocol address it named as its own.
        ip: [u8; 4],
    },
    /// Something the filter admitted that is neither: a malformed or
    /// non-IPv4 ARP, a gratuitous announcement with no usable sender.
    Other,
}

/// Classify one delivered frame.
///
/// A request whose sender protocol address is zero (an RFC 5227 probe) is
/// `Other`: the sender is asking whether an address is free, not claiming
/// one, so there is nothing to ask it for.
pub fn heard(frame: &[u8]) -> Heard {
    let Some(head) = FrameHead::parse(frame) else {
        return Heard::Other;
    };
    if head.ethertype != EtherType::ARP || frame.len() < FRAME_LEN {
        return Heard::Other;
    }
    let ipv4 = frame[16..18] == EtherType::IPV4.to_wire() && frame[18] == 6 && frame[19] == 4;
    if !ipv4 {
        return Heard::Other;
    }
    let mut station = [0u8; 6];
    station.copy_from_slice(&frame[SENDER_MAC_AT..SENDER_MAC_AT + 6]);
    let station = MacAddr(station);
    let mut ip = [0u8; 4];
    ip.copy_from_slice(&frame[SENDER_IP_AT..SENDER_IP_AT + 4]);
    if !station.is_valid_source() {
        return Heard::Other;
    }
    let op = &frame[OP_AT..OP_AT + 2];
    if op == OP_REPLY {
        Heard::Reply { station, ip }
    } else if op == OP_REQUEST && ip != [0; 4] {
        Heard::Request { station, ip }
    } else {
        Heard::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: MacAddr = MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]);
    const GATEWAY: MacAddr = MacAddr([0x52, 0x55, 0x0a, 0, 2, 2]);

    /// The request is laid out as RFC 826 says, field by field, so a
    /// responder that parses it strictly (SLIRP does) answers.
    #[test]
    fn a_request_is_well_formed() {
        let f = request(ME, [10, 0, 2, 15], MacAddr::BROADCAST, [10, 0, 2, 2]);
        assert_eq!(f.len(), 42);
        assert_eq!(&f[0..6], &[0xff; 6], "broadcast");
        assert_eq!(&f[6..12], &ME.0);
        assert_eq!(&f[12..14], &[0x08, 0x06], "EtherType ARP");
        assert_eq!(
            &f[14..22],
            &[0, 1, 0x08, 0, 6, 4, 0, 1],
            "Ethernet/IPv4/6/4/request"
        );
        assert_eq!(&f[22..28], &ME.0, "sender hardware address");
        assert_eq!(&f[28..32], &[10, 0, 2, 15], "sender protocol address");
        assert_eq!(&f[32..38], &[0; 6], "target hardware address unknown");
        assert_eq!(&f[38..42], &[10, 0, 2, 2], "target protocol address");
        let unicast = request(ME, [10, 0, 2, 15], GATEWAY, [10, 0, 2, 2]);
        assert_eq!(&unicast[0..6], &GATEWAY.0, "addressed to the station");
        assert_eq!(&unicast[14..], &f[14..], "the ARP body does not change");
    }

    #[test]
    fn a_flow_can_resolve_only_its_own_next_hop() {
        let neighbor = Neighbor {
            local: [10, 0, 2, 15],
            next_hop: [10, 0, 2, 2],
        };
        let mut query = request(ME, neighbor.local, MacAddr::BROADCAST, neighbor.next_hop);
        let header = FrameHead::parse(&query).unwrap();
        assert!(neighbor.accepts_request(&header, &query, ME));
        query[SENDER_IP_AT] ^= 1;
        assert!(!neighbor.accepts_request(&header, &query, ME));
        query[SENDER_IP_AT] ^= 1;
        query[TARGET_IP_AT] ^= 1;
        assert!(!neighbor.accepts_request(&header, &query, ME));

        let mut reply = request(GATEWAY, neighbor.next_hop, ME, neighbor.local);
        reply[OP_AT..SENDER_MAC_AT].copy_from_slice(&OP_REPLY);
        reply[TARGET_MAC_AT..TARGET_IP_AT].copy_from_slice(&ME.0);
        let header = FrameHead::parse(&reply).unwrap();
        assert!(neighbor.accepts_reply(&header, &reply, ME));
        assert!(!neighbor.accepts_reply(&header, &reply, GATEWAY));
        assert!(!neighbor.accepts_reply(&header, &reply[..FRAME_LEN - 1], ME));
        reply[SENDER_MAC_AT] ^= 1;
        assert!(!neighbor.accepts_reply(&header, &reply, ME));
    }

    /// A reply names the responder and the address it holds; a request
    /// names a station that can be asked; everything else is nothing.
    #[test]
    fn heard_classifies_replies_requests_and_noise() {
        let mut reply = request(GATEWAY, [10, 0, 2, 2], ME, [10, 0, 2, 15]);
        reply[OP_AT..OP_AT + 2].copy_from_slice(&OP_REPLY);
        assert_eq!(
            heard(&reply),
            Heard::Reply {
                station: GATEWAY,
                ip: [10, 0, 2, 2]
            }
        );
        let asking = request(
            GATEWAY,
            [192, 168, 2, 1],
            MacAddr::BROADCAST,
            [192, 168, 2, 7],
        );
        assert_eq!(
            heard(&asking),
            Heard::Request {
                station: GATEWAY,
                ip: [192, 168, 2, 1]
            }
        );
        // An RFC 5227 probe claims no address: nothing to ask it for.
        let probe = request(GATEWAY, [0; 4], MacAddr::BROADCAST, [192, 168, 2, 1]);
        assert_eq!(heard(&probe), Heard::Other);
        // Too short, wrong protocol, a multicast sender, a non-ARP frame.
        assert_eq!(heard(&reply[..40]), Heard::Other);
        let mut ipv6 = asking;
        ipv6[16..18].copy_from_slice(&[0x86, 0xdd]);
        assert_eq!(heard(&ipv6), Heard::Other);
        let mut bad_src = asking;
        bad_src[SENDER_MAC_AT] = 0x01;
        assert_eq!(heard(&bad_src), Heard::Other);
        let mut ip = asking;
        ip[12..14].copy_from_slice(&[0x08, 0x00]);
        assert_eq!(heard(&ip), Heard::Other);
    }
}
