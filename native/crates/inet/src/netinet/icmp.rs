//! **ICMPv4** — echo, which is the whole of what `ping` is.
//!
//! The first protocol row of the switch, and the one that proves the path:
//! an echo reply from a real host means ARP resolved, the route held, the
//! checksum was right at both ends, and the frame went out and came back
//! through the vif.

use kore_driver::net::flow::{IpProto, Ipv4Addr};
use smoltcp::phy::ChecksumCapabilities;
use smoltcp::wire::{Icmpv4Message, Icmpv4Packet, Icmpv4Repr};

use crate::ifnet::Ifnet;
use crate::netinet::arp::ArpCache;
use crate::netinet::ip::{self, OutputError};
use crate::row::Identity;

/// What a received ICMP datagram meant.
/// Why a peer said a datagram could not be delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unreachable {
    /// Code 0 — no route to the destination network.
    Network,
    /// Code 1 — the network is reachable, the host is not.
    Host,
    /// Code 2 — the host does not speak that IP protocol.
    Protocol,
    /// Code 3 — nothing is listening on that port. The one a live peer
    /// sends, and therefore evidence the path works.
    Port,
    /// Code 4 — the datagram was too big for a hop and carried Don't
    /// Fragment.
    ///
    /// **The one code that is an input rather than a diagnosis.** It is what
    /// path-MTU discovery runs on: the reporter names the next hop's MTU and
    /// a sender that heeds it stops blackholing. Carried through this seam
    /// so a transport above can consume it — `res/INET_STACK.md` §9.2 notes
    /// PLPMTUD is the one absent item that crosses the protosw boundary,
    /// because the report arrives at this row and is needed in another.
    FragmentationNeeded,
    /// Code 9, 10 or 13 — a filter said no.
    Prohibited,
    /// Any other code, kept as its number rather than discarded.
    Other(u8),
}

impl Unreachable {
    /// Classify an ICMP destination-unreachable code.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Network,
            1 => Self::Host,
            2 => Self::Protocol,
            3 => Self::Port,
            4 => Self::FragmentationNeeded,
            9 | 10 | 13 => Self::Prohibited,
            other => Self::Other(other),
        }
    }

    /// A short phrase naming this code, for a report a person reads.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Network => "destination network unreachable",
            Self::Host => "destination host unreachable",
            Self::Protocol => "destination protocol unreachable",
            Self::Port => "destination port unreachable",
            Self::FragmentationNeeded => "fragmentation needed and DF set",
            Self::Prohibited => "administratively prohibited",
            Self::Other(_) => "destination unreachable",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcmpEvent {
    /// A peer answered our echo.
    EchoReply {
        /// Who answered.
        from: Ipv4Addr,
        /// The identifier we sent.
        ident: u16,
        /// The sequence number we sent.
        seq: u16,
        /// Payload bytes echoed back.
        payload_len: usize,
    },
    /// A peer asked us to echo, and we did.
    EchoAnswered {
        /// Who asked.
        from: Ipv4Addr,
    },
    /// A peer reported our datagram undeliverable.
    Unreachable {
        /// Who reported it.
        from: Ipv4Addr,
        /// Why, as the reporter said it.
        ///
        /// Carried rather than flattened to "unreachable" because the codes
        /// mean operationally different things — a port refused by a live
        /// host and a network nobody routes to are not the same finding —
        /// and because one of them is load-bearing (see
        /// [`Unreachable::FragmentationNeeded`]).
        code: Unreachable,
        /// The next hop's MTU, for [`Unreachable::FragmentationNeeded`] only.
        ///
        /// `None` on every other code, and on a reporter that sent the
        /// pre-RFC-1191 form with the field zeroed — which is why it is an
        /// `Option` and not a `u16` a caller would read as "MTU zero".
        next_hop_mtu: Option<u16>,
    },
}

/// Send an echo request to `dst`.
///
/// The payload is copied into the datagram; a caller that wants the classic
/// timing probe puts a timestamp in it and reads it back off the reply.
pub fn echo_request<I: Ifnet>(
    net: &mut I,
    cache: &mut ArpCache,
    identity: &Identity,
    dst: Ipv4Addr,
    ident: u16,
    seq: u16,
    payload: &[u8],
    now_ms: u64,
) -> Result<(), OutputError> {
    let repr = Icmpv4Repr::EchoRequest {
        ident,
        seq_no: seq,
        data: payload,
    };
    let len = repr.buffer_len();
    ip::output(
        net,
        cache,
        identity,
        dst,
        IpProto::ICMP,
        len,
        now_ms,
        |body| {
            let mut packet = Icmpv4Packet::new_unchecked(&mut body[..len]);
            repr.emit(&mut packet, &ChecksumCapabilities::default());
        },
    )
}

/// Handle one received ICMP datagram.
///
/// An echo request addressed to this station is answered here and reported as
/// [`IcmpEvent::EchoAnswered`]; a reply is reported for whoever is waiting on
/// it. Returns `None` for anything malformed or not understood — never a
/// guess.
pub fn input<I: Ifnet>(
    net: &mut I,
    cache: &mut ArpCache,
    identity: &Identity,
    src: Ipv4Addr,
    payload: &[u8],
    now_ms: u64,
) -> Option<IcmpEvent> {
    let packet = Icmpv4Packet::new_checked(payload).ok()?;
    // Verified, not assumed: a corrupt echo answered as if it were sound
    // would put this station's name to bytes it never checked.
    let repr = Icmpv4Repr::parse(&packet, &ChecksumCapabilities::default()).ok()?;

    match repr {
        Icmpv4Repr::EchoRequest {
            ident,
            seq_no,
            data,
        } => {
            // Answer from this station's own address, to the asker. A reply
            // is only sent for a request that actually reached us, which
            // `ip::input` has already established.
            let reply = Icmpv4Repr::EchoReply {
                ident,
                seq_no,
                data,
            };
            let len = reply.buffer_len();
            let _ = ip::output(
                net,
                cache,
                identity,
                src,
                IpProto::ICMP,
                len,
                now_ms,
                |body| {
                    let mut p = Icmpv4Packet::new_unchecked(&mut body[..len]);
                    reply.emit(&mut p, &ChecksumCapabilities::default());
                },
            );
            Some(IcmpEvent::EchoAnswered { from: src })
        }
        Icmpv4Repr::EchoReply {
            ident,
            seq_no,
            data,
        } => Some(IcmpEvent::EchoReply {
            from: src,
            ident,
            seq: seq_no,
            payload_len: data.len(),
        }),
        Icmpv4Repr::DstUnreachable { reason, .. } => {
            let code = Unreachable::from_code(u8::from(reason));
            // RFC 1191 puts the next hop's MTU in the low half of the
            // header's second word — bytes 6..8 — which the `Icmpv4Repr`
            // above does not carry. Read only for the code that defines it,
            // and only when the bytes are actually there; a zero is the
            // older form that supplied none.
            let next_hop_mtu = (code == Unreachable::FragmentationNeeded)
                .then(|| {
                    payload
                        .get(6..8)
                        .map(|b| u16::from_be_bytes([b[0], b[1]]))
                        .filter(|mtu| *mtu != 0)
                })
                .flatten();
            Some(IcmpEvent::Unreachable {
                from: src,
                code,
                next_hop_mtu,
            })
        }
        _ => {
            let _ = Icmpv4Message::EchoRequest;
            None
        }
    }
}
