//! **The DNS resolver** — A records, which is all `ping google.com` needs.
//!
//! One question in flight, answered or timed out. That is not a simplification
//! to be apologised for: a stack whose only resolver client is its own
//! `ping` has exactly one question at a time, and a query table sized for
//! concurrency nobody has would be infrastructure ahead of its consumer.
//!
//! The wire format is borrowed; the query lifecycle, the transaction matching
//! and the refusal to believe an answer to a question we did not ask are
//! written here.

use kore_driver::net::flow::Ipv4Addr;
use smoltcp::wire::{DnsFlags, DnsPacket, DnsQueryType, DnsQuestion, DnsRecordData, DnsRepr};

/// The port a resolver listens on.
pub const DNS_PORT: u16 = 53;

/// How long to wait before asking again, in milliseconds.
pub const DNS_RETRY_MS: u64 = 2_000;

/// The longest name this resolver will encode.
///
/// 255 octets is the protocol's own ceiling for a wire-format name, so this
/// is the spec's bound rather than a chosen one.
pub const MAX_NAME: usize = 255;

/// What the resolver is doing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DnsState {
    /// Nothing asked.
    #[default]
    Idle,
    /// A query is outstanding.
    Asking {
        /// The transaction identifier it carries.
        xid: u16,
    },
    /// An answer arrived.
    Answered {
        /// The address resolved.
        addr: Ipv4Addr,
    },
    /// The server answered that the name does not exist.
    NoSuchName,
}

/// What a query committed to, and what an answer must match.
///
/// A transaction identifier alone is 16 bits of guessing. These four fields
/// together are what make an answer *this* session's: the server it was asked
/// of, the port it was asked from, and the question it asked.
#[derive(Debug)]
struct Asked {
    /// The resolver the query went to.
    server: Ipv4Addr,
    /// The local port it went from.
    local_port: u16,
    /// The wire-form question, as [`encode_name`] built it.
    qname: [u8; MAX_NAME + 2],
    /// How much of `qname` is the name.
    qname_len: u16,
}

impl Asked {
    /// The wire-form question this query asked.
    fn question(&self) -> &[u8] {
        &self.qname[..self.qname_len as usize]
    }
}

/// One outstanding question.
#[derive(Debug, Default)]
pub struct Resolver {
    state: DnsState,
    last_sent_ms: u64,
    /// What the outstanding query committed to. `None` whenever `state` is
    /// not `Asking`.
    asked: Option<Asked>,
}

impl Resolver {
    /// A resolver with nothing outstanding.
    pub const fn new() -> Self {
        Self {
            state: DnsState::Idle,
            last_sent_ms: 0,
            asked: None,
        }
    }

    /// What the resolver is doing.
    pub fn state(&self) -> DnsState {
        self.state
    }

    /// The answer, once there is one.
    pub fn answer(&self) -> Option<Ipv4Addr> {
        match self.state {
            DnsState::Answered { addr } => Some(addr),
            _ => None,
        }
    }

    /// Forget the current question.
    pub fn reset(&mut self) {
        self.state = DnsState::Idle;
        self.last_sent_ms = 0;
        self.asked = None;
    }

    /// Whether a query should be (re)sent now.
    pub fn due(&self, now_ms: u64) -> bool {
        match self.state {
            DnsState::Idle => true,
            DnsState::Asking { .. } => now_ms.saturating_sub(self.last_sent_ms) >= DNS_RETRY_MS,
            _ => false,
        }
    }

    /// Record that a query carrying `xid` just went out.
    ///
    /// Takes the server it went to, the port it went from and the name it
    /// asked for, because [`input`](Self::input) checks all three. A name
    /// that will not encode leaves the resolver idle rather than outstanding
    /// — there is no question to match an answer against.
    pub fn sent(&mut self, xid: u16, server: Ipv4Addr, local_port: u16, name: &str, now_ms: u64) {
        let mut qname = [0u8; MAX_NAME + 2];
        let Some(len) = encode_name(name, &mut qname) else {
            self.reset();
            return;
        };
        self.asked = Some(Asked {
            server,
            local_port,
            qname,
            qname_len: len as u16,
        });
        self.state = DnsState::Asking { xid };
        self.last_sent_ms = now_ms;
    }

    /// Feed one received DNS payload.
    ///
    /// Returns the resolved address when this message answered our question.
    ///
    /// **Four things must match, not one.** The transaction identifier is 16
    /// bits; on its own it lets anything that can reach this port decide what
    /// a name means, which is cache-poisoning shape. So the datagram must
    /// also come *from the resolver the query went to*, from the DNS port,
    /// *to the port the query went from*, and it must echo back the question
    /// that was asked. `from` and `to_port` are the datagram's, which the
    /// caller already has.
    pub fn input(
        &mut self,
        from: (Ipv4Addr, u16),
        to_port: u16,
        payload: &[u8],
    ) -> Option<Ipv4Addr> {
        let DnsState::Asking { xid } = self.state else {
            return None;
        };
        let asked = self.asked.as_ref()?;
        // Who it came from, and where it landed.
        if from.0 != asked.server || from.1 != DNS_PORT || to_port != asked.local_port {
            return None;
        }
        let packet = DnsPacket::new_checked(payload).ok()?;
        if packet.transaction_id() != xid {
            return None;
        }
        if !packet.flags().contains(DnsFlags::RESPONSE) {
            return None;
        }
        // And that it answers the question that was actually asked. A
        // response carrying somebody else's question with our identifier is
        // not an answer to us.
        if !echoes_question(&packet, asked.question()) {
            return None;
        }

        if packet.rcode() == smoltcp::wire::DnsRcode::NXDomain {
            self.state = DnsState::NoSuchName;
            return None;
        }

        // Walk past the echoed questions to reach the answers.
        let mut rest = packet.payload();
        for _ in 0..packet.question_count() {
            let (next, _) = DnsQuestion::parse(rest).ok()?;
            rest = next;
        }
        for _ in 0..packet.answer_record_count() {
            let (next, record) = smoltcp::wire::DnsRecord::parse(rest).ok()?;
            rest = next;
            // CNAMEs and anything else are skipped rather than followed: the
            // one consumer wants an address, and chasing a chain is a
            // different protocol feature with its own loop bound.
            if let DnsRecordData::A(addr) = record.data {
                let addr = Ipv4Addr(addr.octets());
                self.state = DnsState::Answered { addr };
                return Some(addr);
            }
        }
        None
    }
}

/// Encode a query for `name` into `out`, returning its length.
///
/// `name` is a dotted host name; its labels are written in the wire's
/// length-prefixed form. `None` when the name is malformed or will not fit —
/// refused rather than truncated, because a truncated name is a *different*
/// name and would resolve to somebody else.
/// Encode `name` into its wire form — a sequence of length-prefixed labels
/// closed by the root — returning how many bytes it took.
///
/// Its own function because the resolver remembers the question it asked, and
/// an answer is only this session's if it echoes that exact question back.
fn encode_name(name: &str, wire: &mut [u8; MAX_NAME + 2]) -> Option<usize> {
    if name.is_empty() || name.len() > MAX_NAME {
        return None;
    }
    let mut at = 0usize;
    for label in name.split('.') {
        // An empty label is only lawful as the trailing root, which this
        // appends itself; one in the middle is a malformed name.
        if label.is_empty() || label.len() > 63 {
            return None;
        }
        if at + 1 + label.len() >= wire.len() {
            return None;
        }
        wire[at] = label.len() as u8;
        wire[at + 1..at + 1 + label.len()].copy_from_slice(label.as_bytes());
        at += 1 + label.len();
    }
    wire[at] = 0; // the root label
    at += 1;
    Some(at)
}

/// Whether `packet`'s first question is the one we asked.
///
/// Compares the wire-form name and requires the A type. A response with no
/// question section at all is not an echo of ours.
fn echoes_question(packet: &DnsPacket<&[u8]>, want: &[u8]) -> bool {
    if packet.question_count() == 0 {
        return false;
    }
    let Ok((_, question)) = DnsQuestion::parse(packet.payload()) else {
        return false;
    };
    question.type_ == DnsQueryType::A && question.name == want
}

pub fn encode_query(name: &str, xid: u16, out: &mut [u8]) -> Option<usize> {
    if name.is_empty() || name.len() > MAX_NAME {
        return None;
    }
    let mut wire = [0u8; MAX_NAME + 2];
    let at = encode_name(name, &mut wire)?;

    let question = DnsQuestion {
        name: &wire[..at],
        type_: DnsQueryType::A,
    };
    let repr = DnsRepr {
        transaction_id: xid,
        flags: DnsFlags::RECURSION_DESIRED,
        opcode: smoltcp::wire::DnsOpcode::Query,
        question,
    };
    let len = repr.buffer_len();
    if len > out.len() {
        return None;
    }
    let mut packet = DnsPacket::new_unchecked(&mut out[..len]);
    repr.emit(&mut packet);
    Some(len)
}
