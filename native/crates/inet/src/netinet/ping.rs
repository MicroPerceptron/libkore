//! The echo session — what a `ping` is, once the datagram is someone else's
//! problem.
//!
//! [`IpStack::echo_request`](super::IpStack::echo_request) puts one probe on
//! the wire and [`IpStack::input`](super::IpStack::input) reports one reply.
//! Neither remembers that the two are related, and that relation — a sequence
//! number, the instant it departed, and whether anything came back before a
//! deadline — is the whole of the measurement. This module owns it, and
//! nothing else: it never sends, never parses, and never sees a frame.
//!
//! That split is deliberate. The session is pure arithmetic over a clock, so
//! it is exhaustively testable with no interface at all, and the two callers
//! that drive it — the row's founding self-test and the shell's `ping` — share
//! one definition of what "lost" and "duplicate" mean instead of each
//! inventing one.
//!
//! # Composition
//!
//! The caller keeps the loop, because the caller is the one that knows what
//! *else* its stack is doing (resolving a name, driving DHCP). Three calls per
//! sweep:
//!
//! ```ignore
//! session.send_due(&mut stack, now_us);
//! stack.input(now_us / 1_000, |event| match event {
//!     Event::Icmp(icmp) => {
//!         if let Some(reply) = session.absorb(icmp, now_us) {
//!             // one line of output
//!         }
//!     }
//!     other => { /* still the caller's, never swallowed here */ }
//! });
//! session.expire(now_us);
//! ```
//!
//! # Why the departure time is kept here and not in the payload
//!
//! `icmp.rs` notes that a caller wanting the classic timing probe "puts a
//! timestamp in it and reads it back off the reply". That is not reachable
//! through this engine: [`IcmpEvent::EchoReply`] reports `payload_len` and not
//! the payload bytes, so there is nothing to read the timestamp back *out of*.
//!
//! Keeping it here is the better answer regardless, and would have been worth
//! choosing even if the bytes were available: a timestamp that round-trips
//! through a peer is a number the peer chose. A host that echoes a doctored
//! payload would be reporting its own round-trip time, and on a loopback or a
//! hostile link that is a measurement of nothing. The table below cannot be
//! influenced from the wire — the only thing a reply contributes is *which*
//! slot it matches.
//!
//! [`IcmpEvent::EchoReply`]: super::IcmpEvent::EchoReply

use kore_driver::net::flow::Ipv4Addr;

use super::icmp::Unreachable;
use super::{IcmpEvent, IpStack, OutputError};
use crate::ifnet::Ifnet;

/// Outstanding probes tracked at once.
///
/// A probe occupies a slot from departure until it is answered or expires, so
/// the table must hold one sweep's worth of unanswered probes: at the default
/// one-per-second cadence against the default four-second timeout, four. The
/// bound is sixteen — four times that headroom, so a caller may shorten the
/// interval or lengthen the timeout by a factor of four before the table is
/// the binding constraint, and the whole table is 16 × 16 bytes = one quarter
/// of a 1 KiB page.
///
/// A power of two because the slot index is `seq & (WINDOW - 1)`: a mask, not
/// a division, on a path that runs once per probe.
const WINDOW: usize = 16;

/// Payload bytes a probe carries.
///
/// Fifty-six, which with the 8-byte ICMP header is the 64-byte datagram every
/// other `ping` sends by default — so a reply line here is comparable with one
/// read off any other machine, and a path that silently drops on size behaves
/// the same way here as it does there.
pub const PAYLOAD_BYTES: usize = 56;

/// The classic data pattern: `0x10` ascending.
///
/// The bytes are never interpreted — not on send, and not on receipt, since
/// the reply's payload does not reach this engine at all. They exist to give
/// the datagram its conventional size and to make a corrupted path visible to
/// a packet capture sitting beside it.
fn pattern() -> [u8; PAYLOAD_BYTES] {
    let mut p = [0u8; PAYLOAD_BYTES];
    let mut i = 0;
    while i < PAYLOAD_BYTES {
        p[i] = (0x10 + i) as u8;
        i += 1;
    }
    p
}

/// One tracked probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// Nothing has used this slot, or what did has been accounted for.
    Free,
    /// Departed at this instant, not yet answered.
    Outstanding {
        /// The sequence number, kept because a slot is shared by every
        /// sequence congruent to it and a late reply must not be credited to
        /// its successor.
        seq: u16,
        /// Microseconds, on the caller's clock.
        sent_us: u64,
    },
    /// Answered. Retained rather than freed so a second copy of the same reply
    /// is a *duplicate* and not a reply to whatever next occupies the slot.
    Answered {
        /// The sequence number that was answered.
        seq: u16,
    },
}

/// What one accepted reply says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reply {
    /// Who answered.
    pub from: Ipv4Addr,
    /// The sequence number it answers.
    pub seq: u16,
    /// Payload bytes echoed back, as the peer reported them.
    pub bytes: usize,
    /// Round trip, in microseconds, measured from this station's own record of
    /// the departure — never from anything the peer sent back.
    pub rtt_us: u64,
}

/// A reply that matched a slot already answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Duplicate {
    /// Who answered.
    pub from: Ipv4Addr,
    /// The sequence number answered twice.
    pub seq: u16,
}

/// What [`Ping::absorb`] made of one ICMP event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heard {
    /// A reply to an outstanding probe of this session.
    Reply(Reply),
    /// A second reply to a probe already answered.
    Duplicate(Duplicate),
    /// A peer reported one of our datagrams undeliverable.
    Unreachable {
        /// Who reported it.
        from: Ipv4Addr,
        /// Why — the code, not a flattened "unreachable".
        code: Unreachable,
        /// The next hop's MTU, when the code was
        /// [`Unreachable::FragmentationNeeded`].
        next_hop_mtu: Option<u16>,
    },
}

/// The session's running totals.
///
/// Derived quantities — loss and mean round trip — are computed on demand
/// rather than stored, so there is no second copy of a number to drift.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Probes that reached the wire.
    pub transmitted: u32,
    /// Probes answered exactly once.
    pub received: u32,
    /// Replies matching a probe already answered.
    pub duplicates: u32,
    /// Probes that passed their deadline unanswered.
    pub lost: u32,
    /// Smallest round trip observed, microseconds.
    pub rtt_min_us: u64,
    /// Largest round trip observed, microseconds.
    pub rtt_max_us: u64,
    /// Sum of every round trip, microseconds — the mean's numerator.
    pub rtt_total_us: u64,
}

impl Stats {
    /// Mean round trip in microseconds, or `None` before anything was
    /// answered.
    pub fn rtt_mean_us(&self) -> Option<u64> {
        (self.received > 0).then(|| self.rtt_total_us / self.received as u64)
    }

    /// Loss as tenths of a percent, so a caller with no float prints `12.3%`
    /// from one integer.
    ///
    /// The denominator is what has been *accounted for* — answered plus
    /// expired — and not what was transmitted, because a probe still inside
    /// its timeout is neither lost nor received and counting it as lost would
    /// report a figure that improves as the session runs.
    pub fn loss_tenths(&self) -> u32 {
        let accounted = self.received + self.lost;
        if accounted == 0 {
            return 0;
        }
        (self.lost as u64 * 1000 / accounted as u64) as u32
    }
}

/// One echo session against one peer.
pub struct Ping {
    target: Ipv4Addr,
    ident: u16,
    next_seq: u16,
    interval_us: u64,
    timeout_us: u64,
    /// `None` runs until the caller stops.
    count: Option<u16>,
    /// `None` until the first probe departs, so the first one is due at once.
    last_send_us: Option<u64>,
    slots: [Slot; WINDOW],
    payload: [u8; PAYLOAD_BYTES],
    stats: Stats,
}

impl Ping {
    /// Open a session.
    ///
    /// `ident` separates this session's replies from those of any other on the
    /// same station; a caller with one session may pass anything.
    pub fn new(
        target: Ipv4Addr,
        ident: u16,
        interval_us: u64,
        timeout_us: u64,
        count: Option<u16>,
    ) -> Self {
        Self {
            target,
            ident,
            next_seq: 1,
            interval_us,
            timeout_us,
            count,
            last_send_us: None,
            slots: [Slot::Free; WINDOW],
            payload: pattern(),
            stats: Stats::default(),
        }
    }

    /// Who this session is addressed to.
    pub fn target(&self) -> Ipv4Addr {
        self.target
    }

    /// The running totals.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Probes this session will still send, if it is bounded.
    fn remaining(&self) -> Option<u16> {
        self.count
            .map(|n| n.saturating_sub(self.stats.transmitted as u16))
    }

    /// Whether every probe this session will send has been sent.
    pub fn sending_done(&self) -> bool {
        self.remaining() == Some(0)
    }

    /// Whether the session is finished: nothing left to send, and nothing left
    /// outstanding that could still arrive.
    ///
    /// A caller stops on this rather than on a probe count, so a reply that
    /// lands inside its deadline is still counted.
    pub fn complete(&self, now_us: u64) -> bool {
        self.sending_done() && !self.any_outstanding(now_us)
    }

    /// Whether any probe is still inside its deadline.
    fn any_outstanding(&self, now_us: u64) -> bool {
        self.slots.iter().any(|slot| match slot {
            Slot::Outstanding { sent_us, .. } => now_us.saturating_sub(*sent_us) < self.timeout_us,
            _ => false,
        })
    }

    /// Whether a probe is due now.
    fn due(&self, now_us: u64) -> bool {
        if self.sending_done() {
            return false;
        }
        match self.last_send_us {
            None => true,
            Some(last) => now_us.saturating_sub(last) >= self.interval_us,
        }
    }

    /// Send the probe that is due, if one is.
    ///
    /// Returns the sequence number that reached the wire. [`OutputError::Pending`]
    /// is not a failure and not a transmission — the next hop's hardware
    /// address is still being learned, the cadence clock is left alone, and the
    /// same probe is due again on the next sweep. Every other refusal is
    /// reported to the caller and likewise costs no sequence number, because a
    /// probe that never departed cannot be lost.
    pub fn send_due<I: Ifnet>(
        &mut self,
        stack: &mut IpStack<I>,
        now_us: u64,
    ) -> Result<Option<u16>, OutputError> {
        if !self.due(now_us) {
            return Ok(None);
        }
        let seq = self.next_seq;
        match stack.echo_request(self.target, self.ident, seq, &self.payload, now_us / 1_000) {
            Ok(()) => {
                self.depart(seq, now_us);
                Ok(Some(seq))
            }
            // The ordinary first outcome of addressing anyone new: retry.
            Err(OutputError::Pending) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// Record that `seq` departed at `now_us`.
    ///
    /// Taking the slot evicts whatever held it. An eviction can only be a probe
    /// still marked outstanding a full window later, which is `WINDOW`
    /// intervals past its own departure — far beyond any timeout the table is
    /// sized for — so it is counted lost here rather than silently dropped.
    fn depart(&mut self, seq: u16, now_us: u64) {
        let at = Self::slot_of(seq);
        if let Slot::Outstanding { .. } = self.slots[at] {
            self.stats.lost += 1;
        }
        self.slots[at] = Slot::Outstanding {
            seq,
            sent_us: now_us,
        };
        self.stats.transmitted += 1;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.last_send_us = Some(now_us);
    }

    /// Take one ICMP event and say what this session made of it.
    ///
    /// `None` for anything that is not this session's business — another
    /// station's echo, a reply under a different identifier, a reply to a
    /// sequence number this session never sent or has already accounted for.
    /// Call it from inside the [`IpStack::input`](super::IpStack::input)
    /// closure; every other event stays the caller's.
    pub fn absorb(&mut self, event: IcmpEvent, now_us: u64) -> Option<Heard> {
        match event {
            IcmpEvent::EchoReply {
                from,
                ident,
                seq,
                payload_len,
            } => {
                // Not ours: a reply to another session on this station.
                if ident != self.ident {
                    return None;
                }
                // Not ours either: a reply from a station this session never
                // probed. The identifier is a 16-bit field anyone on the path
                // can echo, so without this an off-path station that guesses
                // it reports reachability and a round trip for a host that
                // may be down — the measurement the whole command exists to
                // make, taken from the wrong peer.
                if from != self.target {
                    return None;
                }
                let at = Self::slot_of(seq);
                match self.slots[at] {
                    Slot::Outstanding {
                        seq: want, sent_us, ..
                    } if want == seq => {
                        let rtt_us = now_us.saturating_sub(sent_us);
                        self.slots[at] = Slot::Answered { seq };
                        self.credit(rtt_us);
                        Some(Heard::Reply(Reply {
                            from,
                            seq,
                            bytes: payload_len,
                            rtt_us,
                        }))
                    }
                    Slot::Answered { seq: want } if want == seq => {
                        self.stats.duplicates += 1;
                        Some(Heard::Duplicate(Duplicate { from, seq }))
                    }
                    // A sequence number this session never sent, or one whose
                    // slot a later probe has taken: too late to be credited.
                    _ => None,
                }
            }
            IcmpEvent::Unreachable {
                from,
                code,
                next_hop_mtu,
            } => Some(Heard::Unreachable {
                from,
                code,
                next_hop_mtu,
            }),
            // This station answering someone else's probe is not this
            // session's business.
            IcmpEvent::EchoAnswered { .. } => None,
        }
    }

    /// Fold one round trip into the totals.
    fn credit(&mut self, rtt_us: u64) {
        if self.stats.received == 0 || rtt_us < self.stats.rtt_min_us {
            self.stats.rtt_min_us = rtt_us;
        }
        if rtt_us > self.stats.rtt_max_us {
            self.stats.rtt_max_us = rtt_us;
        }
        self.stats.rtt_total_us = self.stats.rtt_total_us.saturating_add(rtt_us);
        self.stats.received += 1;
    }

    /// Account every probe whose deadline has passed as lost.
    ///
    /// Returns how many were newly lost on this sweep. Idempotent: a slot moves
    /// out of `Outstanding` as it is counted, so calling this twice on one
    /// instant counts nothing twice.
    pub fn expire(&mut self, now_us: u64) -> u32 {
        let mut newly = 0;
        for slot in &mut self.slots {
            if let Slot::Outstanding { sent_us, .. } = *slot
                && now_us.saturating_sub(sent_us) >= self.timeout_us
            {
                *slot = Slot::Free;
                newly += 1;
            }
        }
        self.stats.lost += newly;
        newly
    }

    /// The slot a sequence number occupies.
    const fn slot_of(seq: u16) -> usize {
        (seq as usize) & (WINDOW - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PEER: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const OTHER: Ipv4Addr = Ipv4Addr([10, 0, 2, 9]);
    const IDENT: u16 = 0xbeef;

    const SEC: u64 = 1_000_000;

    fn session(count: Option<u16>) -> Ping {
        Ping::new(PEER, IDENT, SEC, 4 * SEC, count)
    }

    fn reply(seq: u16) -> IcmpEvent {
        IcmpEvent::EchoReply {
            from: PEER,
            ident: IDENT,
            seq,
            payload_len: PAYLOAD_BYTES,
        }
    }

    /// Departure, reply, and the round trip measured between them — the whole
    /// session in one probe.
    #[test]
    fn a_probe_answered_is_a_round_trip() {
        let mut p = session(None);
        p.depart(1, 0);
        let heard = p.absorb(reply(1), 12_300);
        assert_eq!(
            heard,
            Some(Heard::Reply(Reply {
                from: PEER,
                seq: 1,
                bytes: PAYLOAD_BYTES,
                rtt_us: 12_300,
            }))
        );
        let s = p.stats();
        assert_eq!((s.transmitted, s.received, s.lost), (1, 1, 0));
        assert_eq!(s.rtt_mean_us(), Some(12_300));
    }

    /// A reply carrying another session's identifier is not this session's,
    /// and must not be credited to the probe that happens to share its slot.
    #[test]
    fn a_reply_under_another_ident_is_not_ours() {
        let mut p = session(None);
        p.depart(1, 0);
        let foreign = IcmpEvent::EchoReply {
            from: PEER,
            ident: IDENT ^ 1,
            seq: 1,
            payload_len: PAYLOAD_BYTES,
        };
        assert_eq!(p.absorb(foreign, 1_000), None);
        assert_eq!(p.stats().received, 0, "nothing credited");
    }

    /// A second copy of a reply is a duplicate, not a second success — the
    /// distinction a path that fans out datagrams is diagnosed by.
    /// A reply carrying this session's identifier but sent by a station this
    /// session never probed is not an answer.
    ///
    /// The identifier is a 16-bit field any station on the path can echo, so
    /// without the source check an off-path answer credits reachability and a
    /// round trip for a host that may be down — the one measurement the
    /// command exists to make, taken from the wrong peer.
    #[test]
    fn a_reply_from_another_station_is_not_ours() {
        let mut p = session(None);
        p.depart(1, 0);
        let forged = IcmpEvent::EchoReply {
            from: OTHER,
            ident: IDENT,
            seq: 1,
            payload_len: PAYLOAD_BYTES,
        };
        assert_eq!(p.absorb(forged, 12_300), None, "not from the target");
        assert_eq!(p.stats().received, 0, "nothing credited");
        // And the real target can still answer the same probe afterwards.
        assert!(matches!(
            p.absorb(reply(1), 12_400),
            Some(Heard::Reply(_))
        ));
        assert_eq!(p.stats().received, 1);
    }

    #[test]
    fn a_second_copy_is_a_duplicate() {
        let mut p = session(None);
        p.depart(1, 0);
        assert!(matches!(p.absorb(reply(1), 1_000), Some(Heard::Reply(_))));
        assert_eq!(
            p.absorb(reply(1), 1_100),
            Some(Heard::Duplicate(Duplicate { from: PEER, seq: 1 }))
        );
        let s = p.stats();
        assert_eq!((s.received, s.duplicates), (1, 1), "one reply, one dup");
        assert_eq!(s.rtt_mean_us(), Some(1_000), "a dup does not move the mean");
    }

    /// A probe nobody answers is lost once its deadline passes, and counted
    /// exactly once however often the caller sweeps.
    #[test]
    fn an_unanswered_probe_expires_once() {
        let mut p = session(None);
        p.depart(1, 0);
        assert_eq!(p.expire(SEC), 0, "still inside its deadline");
        assert_eq!(p.expire(4 * SEC), 1, "deadline reached");
        assert_eq!(p.expire(9 * SEC), 0, "already accounted for");
        assert_eq!(p.stats().lost, 1);
    }

    /// A reply that arrives after its probe expired is not credited: the slot
    /// was freed, and crediting it would report a round trip longer than the
    /// timeout that already counted it lost.
    #[test]
    fn a_reply_after_the_deadline_is_not_credited() {
        let mut p = session(None);
        p.depart(1, 0);
        p.expire(4 * SEC);
        assert_eq!(p.absorb(reply(1), 5 * SEC), None);
        let s = p.stats();
        assert_eq!((s.received, s.lost), (0, 1), "lost stays lost");
    }

    /// Loss is over what has been accounted for, not over what was sent — so a
    /// probe still inside its deadline does not read as loss that then
    /// improves.
    #[test]
    fn loss_counts_only_what_is_settled() {
        let mut p = session(None);
        p.depart(1, 0);
        p.absorb(reply(1), 1_000);
        p.depart(2, SEC);
        assert_eq!(
            p.stats().loss_tenths(),
            0,
            "the second is outstanding, not lost"
        );
        p.expire(SEC + 4 * SEC);
        assert_eq!(p.stats().loss_tenths(), 500, "one of two settled: 50.0%");
    }

    /// Sequence numbers congruent modulo the window share a slot; the later
    /// probe takes it, the earlier one is lost, and the stale reply is refused
    /// rather than credited to its successor.
    #[test]
    fn a_wrapped_sequence_does_not_credit_its_predecessor() {
        let mut p = session(None);
        p.depart(1, 0);
        p.depart(1 + WINDOW as u16, SEC);
        assert_eq!(p.stats().lost, 1, "the evicted probe is lost");
        assert_eq!(p.absorb(reply(1), 2 * SEC), None, "stale, refused");
        let heard = p.absorb(reply(1 + WINDOW as u16), SEC + 500);
        assert!(matches!(heard, Some(Heard::Reply(r)) if r.rtt_us == 500));
    }

    /// Min, mean and max over several round trips.
    #[test]
    fn the_round_trip_summary_spans_every_reply() {
        let mut p = session(None);
        for (i, rtt) in [300u64, 100, 500].iter().enumerate() {
            let seq = i as u16 + 1;
            let at = i as u64 * SEC;
            p.depart(seq, at);
            p.absorb(reply(seq), at + rtt);
        }
        let s = p.stats();
        assert_eq!((s.rtt_min_us, s.rtt_max_us), (100, 500));
        assert_eq!(s.rtt_mean_us(), Some(300), "(300+100+500)/3");
    }

    /// A bounded session stops sending at its count, and is complete only once
    /// every probe is settled — not the moment the last one departs, and not
    /// while an earlier one is still inside its deadline.
    #[test]
    fn a_counted_session_completes_after_its_last_probe_settles() {
        let mut p = session(Some(2));
        p.depart(1, 0);
        p.depart(2, SEC);
        assert!(p.sending_done(), "both sent");
        assert!(!p.complete(SEC), "neither has settled");

        p.absorb(reply(2), SEC + 400);
        assert!(
            !p.complete(SEC + 400),
            "the first is unanswered but still inside its deadline"
        );

        p.absorb(reply(1), SEC + 500);
        assert!(p.complete(SEC + 500), "both settled, nothing outstanding");
        let s = p.stats();
        assert_eq!((s.transmitted, s.received, s.lost), (2, 2, 0));
    }

    /// The same session, with the first probe never answered: it completes
    /// when that probe's deadline passes, and the loss is counted.
    #[test]
    fn a_counted_session_completes_when_the_last_outstanding_expires() {
        let mut p = session(Some(2));
        p.depart(1, 0);
        p.depart(2, SEC);
        p.absorb(reply(2), SEC + 400);

        assert!(
            !p.complete(3 * SEC),
            "the first is still inside its deadline"
        );
        p.expire(4 * SEC);
        assert!(p.complete(4 * SEC), "its deadline passed");
        let s = p.stats();
        assert_eq!((s.transmitted, s.received, s.lost), (2, 1, 1));
        assert_eq!(s.loss_tenths(), 500, "one of two settled probes lost");
    }

    /// The cadence: due immediately, then not again until the interval has
    /// passed.
    #[test]
    fn probes_keep_the_interval() {
        let mut p = session(None);
        assert!(p.due(0), "the first is due at once");
        p.depart(1, 0);
        assert!(!p.due(SEC / 2), "inside the interval");
        assert!(p.due(SEC), "the interval has passed");
    }

    /// An unreachable report names its reporter and is not confused with a
    /// reply from the target.
    #[test]
    fn an_unreachable_report_is_surfaced() {
        let mut p = session(None);
        p.depart(1, 0);
        assert_eq!(
            p.absorb(
                IcmpEvent::Unreachable {
                    from: OTHER,
                    code: Unreachable::Host,
                    next_hop_mtu: None,
                },
                SEC
            ),
            Some(Heard::Unreachable {
                from: OTHER,
                code: Unreachable::Host,
                next_hop_mtu: None,
            })
        );
        assert_eq!(p.stats().received, 0, "not an answer");
    }
}
