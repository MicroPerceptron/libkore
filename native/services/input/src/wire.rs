//! `wire` — the input enum ⇄ wire codecs (`res/USERLAND.md` §5.3).
//!
//! The zerocopy forms live in [`kore_abi::svc`]; decode is fail-closed.
//! **The caller identity is not on the wire** (§5.8): [`encode_req`] drops
//! [`InputRequest::via`] and [`decode_req`] lifts it as
//! [`Endpoint::Anonymous`] (least privilege) — the router restamps the real
//! endpoint, so a subscriber cannot express `SetFocus` privilege in bytes.
//! No capability rides any reply ([`REPLY_GRANTS`](SvcProtocol::REPLY_GRANTS)
//! is zero, the swap posture).

use kore_abi::keyboard::Locks;
use kore_abi::session::SlotRef;
use kore_abi::svc::{InputRepWire, InputReqWire};

use kore_rt::svc::SvcProtocol;

use crate::service::{
    Endpoint, I2cRelayBooks, InputRefusal, InputRequest, InputResponse, InputService, InputVerb,
    RelayBooks, SubscriptionId,
};

/// Lower a request to its wire form (total) — `via` is off-wire (§5.8).
pub fn encode_req(req: &InputRequest) -> InputReqWire {
    match req.verb {
        InputVerb::Diagnostic { row } => InputReqWire {
            op: InputReqWire::OP_DIAGNOSTIC,
            focus: 0,
            reserved: [0; 2],
            slot: u32::from(row).into(),
            subscription: 0.into(),
        },
        InputVerb::Subscribe { queue } => InputReqWire {
            op: InputReqWire::OP_SUBSCRIBE,
            focus: 0,
            reserved: [0; 2],
            slot: queue.index().into(),
            subscription: 0.into(),
        },
        InputVerb::Unsubscribe { subscription } => InputReqWire {
            op: InputReqWire::OP_UNSUBSCRIBE,
            focus: 0,
            reserved: [0; 2],
            slot: 0.into(),
            subscription: subscription.0.into(),
        },
        InputVerb::SetFocus { target } => {
            let (focus, sub) = match target {
                Some(id) => (InputReqWire::FOCUS_ONE, id.0),
                None => (InputReqWire::FOCUS_NONE, 0),
            };
            InputReqWire {
                op: InputReqWire::OP_SET_FOCUS,
                focus,
                reserved: [0; 2],
                slot: 0.into(),
                subscription: sub.into(),
            }
        }
        // Status carries no operand: it steers nothing and names no
        // subscription, so every other field rides zero.
        InputVerb::Status => InputReqWire {
            op: InputReqWire::OP_STATUS,
            focus: 0,
            reserved: [0; 2],
            slot: 0.into(),
            subscription: 0.into(),
        },
        InputVerb::SetIndicators { source, locks } => InputReqWire {
            op: InputReqWire::OP_SET_INDICATORS,
            focus: 0,
            reserved: [0; 2],
            slot: u32::from(source).into(),
            subscription: u32::from(locks.bits()).into(),
        },
        InputVerb::Transmit { bytes, len } => {
            // Inline, in order: reserved[2], then slot, then subscription.
            let mut slot = [0u8; 4];
            let mut sub = [0u8; 4];
            slot.copy_from_slice(&bytes[2..6]);
            sub.copy_from_slice(&bytes[6..10]);
            InputReqWire {
                op: InputReqWire::OP_TRANSMIT,
                focus: len,
                reserved: [bytes[0], bytes[1]],
                slot: u32::from_le_bytes(slot).into(),
                subscription: u32::from_le_bytes(sub).into(),
            }
        }
    }
}

/// Lift a wire request — fail-closed on the op and the focus flag; `via`
/// lifts to [`Anonymous`](Endpoint::Anonymous) (the router restamps it).
pub fn decode_req(w: &InputReqWire) -> Option<InputRequest> {
    let verb = match w.op {
        InputReqWire::OP_DIAGNOSTIC
            if w.focus == 0
                && w.reserved == [0; 2]
                && w.subscription.get() == 0
                && w.slot.get() < u32::from(kore_abi::input_status::INPUT_STATUS_ROWS) =>
        {
            InputVerb::Diagnostic {
                row: w.slot.get() as u8,
            }
        }
        InputReqWire::OP_SUBSCRIBE => InputVerb::Subscribe {
            queue: SlotRef::from_wire(w.slot.get()),
        },
        InputReqWire::OP_UNSUBSCRIBE => InputVerb::Unsubscribe {
            subscription: SubscriptionId(w.subscription.get()),
        },
        InputReqWire::OP_SET_FOCUS => {
            let target = match w.focus {
                InputReqWire::FOCUS_NONE => None,
                InputReqWire::FOCUS_ONE => Some(SubscriptionId(w.subscription.get())),
                _ => return None, // an out-of-vocabulary focus flag refuses
            };
            InputVerb::SetFocus { target }
        }
        InputReqWire::OP_STATUS => InputVerb::Status,
        // Every operand bit is accounted for: a source past the byte, a
        // lock bit no lock owns, or a stray flag refuses rather than being
        // truncated into a request that was never sent.
        InputReqWire::OP_TRANSMIT
            if (1..=InputReqWire::TRANSMIT_BYTES).contains(&usize::from(w.focus)) =>
        {
            let mut bytes = [0u8; InputReqWire::TRANSMIT_BYTES];
            bytes[..2].copy_from_slice(&w.reserved);
            bytes[2..6].copy_from_slice(&w.slot.get().to_le_bytes());
            bytes[6..].copy_from_slice(&w.subscription.get().to_le_bytes());
            // Bytes past the count must be zero: nothing rides unread.
            if bytes[usize::from(w.focus)..].iter().any(|&b| b != 0) {
                return None;
            }
            InputVerb::Transmit {
                bytes,
                len: w.focus,
            }
        }
        InputReqWire::OP_SET_INDICATORS if w.focus == 0 && w.reserved == [0; 2] => {
            InputVerb::SetIndicators {
                source: u8::try_from(w.slot.get()).ok()?,
                locks: u8::try_from(w.subscription.get())
                    .ok()
                    .and_then(Locks::from_bits)?,
            }
        }
        _ => return None,
    };
    Some(InputRequest {
        via: Endpoint::Anonymous,
        verb,
    })
}

/// Lower a response + the effected grant count (total).  An input service
/// grants nothing, so `grants` is always zero — carried for the layout
/// parity its siblings share.
pub fn encode_rep(rep: &InputResponse, grants: u8) -> InputRepWire {
    if let InputResponse::Diagnostic(row) = rep {
        return encode_diagnostic(*row);
    }
    let (tag, refusal, sub) = match *rep {
        InputResponse::Diagnostic(_) => unreachable!(),
        InputResponse::Subscribed(id) => (InputRepWire::TAG_SUBSCRIBED, 0, id.0),
        InputResponse::Unsubscribed => (InputRepWire::TAG_UNSUBSCRIBED, 0, 0),
        InputResponse::Focused => (InputRepWire::TAG_FOCUSED, 0, 0),
        InputResponse::Indicated => (InputRepWire::TAG_INDICATED, 0, 0),
        InputResponse::Transmitted { accepted } => {
            (InputRepWire::TAG_TRANSMITTED, 0, u32::from(accepted))
        }
        InputResponse::Status(_) => (InputRepWire::TAG_STATUS, 0, 0),
        // The refusal's own 1-based discriminant rides verbatim.
        InputResponse::Refused(r) => (InputRepWire::TAG_REFUSED, r as u8, 0),
    };
    // The books ride only on their own tag; every other reply zeroes them
    // rather than leaving stale bytes for a peer to misread.
    let b = match *rep {
        InputResponse::Status(b) => b,
        _ => RelayBooks::default(),
    };
    InputRepWire {
        tag,
        refusal,
        grants,
        rung: b.rung,
        subscription: sub.into(),
        wakes: b.wakes.into(),
        parks: b.parks.into(),
        events: b.events.into(),
        dropped: b.dropped.into(),
        unnamed: b.unnamed.into(),
        xhci_bound: u32::from(b.xhci_bound).into(),
        xhci_bind: b.xhci_bind.into(),
        xhci_sources: b.xhci_sources.into(),
        xhci_discovery_errors: b.xhci_discovery_errors.into(),
        xhci_irq_errors: b.xhci_irq_errors.into(),
        i2c_state: b.i2c.map(|worker| worker.state.into()),
        i2c_transfers: b.i2c.map(|worker| worker.transfers.into()),
        i2c_reports: b.i2c.map(|worker| worker.reports.into()),
    }
}

/// Lift a wire reply + its grant count — fail-closed (unknown tags and
/// refusal bytes refuse; a refusal byte on a success tag refuses too).
pub fn decode_rep(w: &InputRepWire) -> Option<(InputResponse, u8)> {
    if w.tag == InputRepWire::TAG_DIAGNOSTIC {
        return decode_diagnostic(w).map(|row| (InputResponse::Diagnostic(row), 0));
    }
    let rep = match (w.tag, w.refusal) {
        (InputRepWire::TAG_SUBSCRIBED, 0) => {
            InputResponse::Subscribed(SubscriptionId(w.subscription.get()))
        }
        (InputRepWire::TAG_UNSUBSCRIBED, 0) => InputResponse::Unsubscribed,
        (InputRepWire::TAG_FOCUSED, 0) => InputResponse::Focused,
        (InputRepWire::TAG_INDICATED, 0) if w.subscription.get() == 0 => InputResponse::Indicated,
        (InputRepWire::TAG_TRANSMITTED, 0) => InputResponse::Transmitted {
            accepted: u8::try_from(w.subscription.get())
                .ok()
                .filter(|&n| usize::from(n) <= InputReqWire::TRANSMIT_BYTES)?,
        },
        (InputRepWire::TAG_STATUS, 0) => InputResponse::Status(RelayBooks {
            rung: w.rung,
            wakes: w.wakes.get(),
            parks: w.parks.get(),
            events: w.events.get(),
            dropped: w.dropped.get(),
            unnamed: w.unnamed.get(),
            xhci_bound: match w.xhci_bound.get() {
                0 => false,
                1 => true,
                _ => return None,
            },
            xhci_bind: w.xhci_bind.get(),
            xhci_sources: w.xhci_sources.get(),
            xhci_discovery_errors: w.xhci_discovery_errors.get(),
            xhci_irq_errors: w.xhci_irq_errors.get(),
            i2c: core::array::from_fn(|index| I2cRelayBooks {
                state: w.i2c_state[index].get(),
                transfers: w.i2c_transfers[index].get(),
                reports: w.i2c_reports[index].get(),
            }),
        }),
        (InputRepWire::TAG_REFUSED, r) => InputResponse::Refused(match r {
            x if x == InputRefusal::BadQueue as u8 => InputRefusal::BadQueue,
            x if x == InputRefusal::TableFull as u8 => InputRefusal::TableFull,
            x if x == InputRefusal::Unknown as u8 => InputRefusal::Unknown,
            x if x == InputRefusal::Denied as u8 => InputRefusal::Denied,
            x if x == InputRefusal::Unsupported as u8 => InputRefusal::Unsupported,
            _ => return None,
        }),
        _ => return None,
    };
    Some((rep, w.grants))
}

pub fn encode_diagnostic(row: kore_abi::input_status::InputDiagnostic) -> InputRepWire {
    InputRepWire {
        tag: InputRepWire::TAG_DIAGNOSTIC,
        refusal: 0,
        grants: 0,
        rung: row.state,
        subscription: row.packed().into(),
        wakes: row.counters[0].into(),
        parks: row.counters[1].into(),
        events: row.counters[2].into(),
        dropped: row.counters[3].into(),
        unnamed: row.counters[4].into(),
        xhci_bound: 0.into(),
        xhci_bind: 0.into(),
        xhci_sources: 0.into(),
        xhci_discovery_errors: 0.into(),
        xhci_irq_errors: 0.into(),
        i2c_state: [0.into(); 2],
        i2c_transfers: [0.into(); 2],
        i2c_reports: [0.into(); 2],
    }
}
pub fn decode_diagnostic(w: &InputRepWire) -> Option<kore_abi::input_status::InputDiagnostic> {
    if w.tag != InputRepWire::TAG_DIAGNOSTIC
        || w.refusal != 0
        || w.grants != 0
        || w.xhci_bound.get() != 0
        || w.xhci_bind.get() != 0
        || w.xhci_sources.get() != 0
        || w.xhci_discovery_errors.get() != 0
        || w.xhci_irq_errors.get() != 0
        || w.i2c_state.iter().any(|value| value.get() != 0)
        || w.i2c_transfers.iter().any(|value| value.get() != 0)
        || w.i2c_reports.iter().any(|value| value.get() != 0)
    {
        return None;
    }
    kore_abi::input_status::InputDiagnostic::from_parts(
        w.rung,
        w.subscription.get(),
        [
            w.wakes.get(),
            w.parks.get(),
            w.events.get(),
            w.dropped.get(),
            w.unnamed.get(),
        ],
    )
}

impl SvcProtocol for InputService {
    type Request = InputRequest;
    type Response = InputResponse;
    type ReqWire = InputReqWire;
    type RepWire = InputRepWire;

    /// An input reply carries no capability — the data-only doctrine (§3.2),
    /// structural in the zero here (the swap posture).
    const REPLY_GRANTS: usize = 0;

    fn encode_req(req: &InputRequest) -> InputReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &InputReqWire) -> Option<InputRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &InputResponse, grants: u8) -> InputRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &InputRepWire) -> Option<(InputResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    #[test]
    fn diagnostic_rows_roundtrip_without_changing_existing_wire_sizes() {
        use kore_abi::input_status::InputDiagnostic;
        assert_eq!(core::mem::size_of::<InputRepWire>(), 84);
        for row in 0..kore_abi::input_status::I2C_STATUS_ROWS {
            let req = InputRequest {
                via: Endpoint::Anonymous,
                verb: InputVerb::Diagnostic { row },
            };
            assert_eq!(decode_req(&encode_req(&req)).unwrap().verb, req.verb);
            let data = InputDiagnostic {
                row,
                state: 3,
                address_or_count: 4,
                error: 2,
                flags: 1,
                counters: [
                    1,
                    2,
                    3,
                    if row as usize % (kore_abi::input_status::MAX_I2C_CHILDREN + 1) == 0 {
                        4
                    } else {
                        kore_abi::input_status::I2cProtocolPhase::FeatureGet.pack(255, 0x8002)
                    },
                    if row as usize % (kore_abi::input_status::MAX_I2C_CHILDREN + 1) == 0 {
                        5
                    } else {
                        kore_abi::input_status::I2cInputOutcome::UnknownReport.pack(255, u16::MAX)
                    },
                ],
            };
            let mut wire = encode_diagnostic(data);
            assert_eq!(decode_diagnostic(&wire), Some(data));
            wire.grants = 1;
            assert!(decode_diagnostic(&wire).is_none());
        }
        let mut wire = encode_req(&InputRequest {
            via: Endpoint::Anonymous,
            verb: InputVerb::Diagnostic { row: 0 },
        });
        wire.slot.set(256);
        assert!(
            decode_req(&wire).is_none(),
            "do not truncate an out-of-range index"
        );
    }

    /// Every request/response round-trips through its wire form.  Requests
    /// compare by re-encoded *bytes* (the `via` identity is off-wire, so
    /// two requests differing only in `via` share one wire form — exactly
    /// the property that makes identity unforgeable, the image precedent).
    /// The transmit side (`res/IMAGE_PROFILES.md` §4.3): bytes ride inline,
    /// so the count and every byte must cross the wire losslessly, and a byte
    /// past the count must refuse rather than ride unread.
    #[test]
    fn transmit_round_trips_inline_and_refuses_trailing_bytes() {
        for text in [&b"x"[..], b"hello", b"0123456789"] {
            let mut bytes = [0u8; InputReqWire::TRANSMIT_BYTES];
            bytes[..text.len()].copy_from_slice(text);
            let verb = InputVerb::Transmit {
                bytes,
                len: text.len() as u8,
            };
            let req = InputRequest {
                via: Endpoint::Anonymous,
                verb,
            };
            assert_eq!(decode_req(&encode_req(&req)).unwrap().verb, verb);
        }
        for rep in [
            InputResponse::Transmitted { accepted: 0 },
            InputResponse::Transmitted { accepted: 10 },
            InputResponse::Refused(InputRefusal::Unsupported),
        ] {
            assert_eq!(decode_rep(&encode_rep(&rep, 0)), Some((rep, 0)));
        }
        let mut w = encode_req(&InputRequest {
            via: Endpoint::Anonymous,
            verb: InputVerb::Transmit {
                bytes: *b"ab\0\0\0\0\0\0\0\0",
                len: 2,
            },
        });
        w.subscription = 1.into();
        assert!(decode_req(&w).is_none(), "a byte past the count refuses");
        w.subscription = 0.into();
        w.focus = 0;
        assert!(decode_req(&w).is_none(), "an empty transmit refuses");
        w.focus = 11;
        assert!(decode_req(&w).is_none(), "an oversized count refuses");
        let mut r = encode_rep(&InputResponse::Transmitted { accepted: 3 }, 0);
        r.subscription = 11.into();
        assert!(decode_rep(&r).is_none(), "an accepted count past the record refuses");
    }

    #[test]
    fn round_trips() {
        for req in [
            InputRequest::subscribe(SlotRef::from_wire(7)),
            InputRequest::unsubscribe(SubscriptionId(3)),
            InputRequest::set_focus(Some(SubscriptionId(9))),
            InputRequest::set_focus(None),
            InputRequest::set_indicators(0x40, Locks::CAPS),
            InputRequest::set_indicators(255, Locks::CAPS.toggled(Locks::NUM).toggled(Locks::SCROLL)),
            InputRequest::set_indicators(1, Locks::NONE),
        ] {
            let back = decode_req(&encode_req(&req)).expect("decodes");
            assert_eq!(back.verb, req.verb);
            assert_eq!(
                back.via,
                Endpoint::Anonymous,
                "the wire carries no identity"
            );
            assert_eq!(encode_req(&back).as_bytes(), encode_req(&req).as_bytes());
        }
        // A privileged-looking request encodes to the SAME bytes as the
        // anonymous one — the wire cannot convey focus authority.
        let anon = InputRequest::set_focus(Some(SubscriptionId(1)));
        let claims_focus = anon.via(Endpoint::Focus);
        assert_eq!(
            encode_req(&anon).as_bytes(),
            encode_req(&claims_focus).as_bytes(),
            "identity is structural, never wire — a subscriber cannot forge it",
        );

        for (rep, grants) in [
            (InputResponse::Subscribed(SubscriptionId(5)), 0u8),
            (InputResponse::Unsubscribed, 0),
            (InputResponse::Focused, 0),
            (InputResponse::Indicated, 0),
            (InputResponse::Refused(InputRefusal::BadQueue), 0),
            (InputResponse::Refused(InputRefusal::TableFull), 0),
            (InputResponse::Refused(InputRefusal::Unknown), 0),
            (InputResponse::Refused(InputRefusal::Denied), 0),
        ] {
            let (back, g) = decode_rep(&encode_rep(&rep, grants)).expect("decodes");
            assert_eq!(back, rep);
            assert_eq!(g, grants);
        }

        let mut books = RelayBooks::default();
        books.xhci_bind = crate::service::xhci_bind::pack(
            crate::service::xhci_bind::PLATFORM_CLAIM,
            crate::service::xhci_bind::REFUSED_NO_MATCH,
        );
        books.i2c[0] = I2cRelayBooks {
            state: 0x0300_0000_0008_0001,
            transfers: 6,
            reports: 0,
        };
        let status = InputResponse::Status(books);
        let (back, grants) = decode_rep(&encode_rep(&status, 0)).expect("status decodes");
        assert_eq!(back, status, "the exact intake and worker states survive");
        assert_eq!(grants, 0);
    }

    /// Decode is fail-closed: unknown ops/tags/refusals and out-of-range
    /// focus flags refuse; a refusal byte on a success tag refuses.
    #[test]
    fn decode_fails_closed() {
        let mut w = encode_req(&InputRequest::subscribe(SlotRef::from_wire(1)));
        w.op = 0;
        assert!(decode_req(&w).is_none());
        w.op = 200;
        assert!(decode_req(&w).is_none());

        // An out-of-vocabulary focus flag on SetFocus refuses.
        let mut w = encode_req(&InputRequest::set_focus(None));
        w.focus = 9;
        assert!(decode_req(&w).is_none(), "a bad focus flag refuses");

        // SetIndicators accounts for every operand bit.
        let good = encode_req(&InputRequest::set_indicators(2, Locks::CAPS));
        let mut w = good;
        w.slot.set(256);
        assert!(decode_req(&w).is_none(), "a source past the byte refuses");
        let mut w = good;
        w.subscription.set(0x08);
        assert!(decode_req(&w).is_none(), "a lock bit no lock owns refuses");
        let mut w = good;
        w.subscription.set(0x102);
        assert!(decode_req(&w).is_none(), "a lock word past the byte refuses");
        let mut w = good;
        w.focus = 1;
        assert!(decode_req(&w).is_none(), "a stray focus flag refuses");
        let mut w = good;
        w.reserved[1] = 1;
        assert!(decode_req(&w).is_none(), "reserved bytes stay zero");

        let mut r = encode_rep(&InputResponse::Indicated, 0);
        r.subscription.set(1);
        assert!(decode_rep(&r).is_none(), "an indicated reply carries no key");

        let mut r = encode_rep(&InputResponse::Focused, 0);
        r.tag = 0;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&InputResponse::Refused(InputRefusal::Denied), 0);
        r.refusal = 77;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&InputResponse::Unsubscribed, 0);
        r.refusal = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a refusal byte on success refuses"
        );
    }
}
