//! `wire` — the output enum ⇄ wire codecs (`res/USERLAND.md` §5.3).
//!
//! The zerocopy wire forms live in [`kore_abi::svc`] (the frozen
//! contract); this module is the convenience bridge: encode is total,
//! decode is **fail-closed** — an out-of-vocabulary op, tag, or refusal
//! decodes to `None`, never a guess.  The [`SvcProtocol`] impl is what
//! plugs [`OutputService`] into the generic request path (rings +
//! doorbells; no grant channels — output replies never carry authority).

use kore_abi::session::SlotRef;
use kore_abi::svc::{OutRepWire, OutReqWire};

use kore_rt::svc::SvcProtocol;

use crate::service::{OutputRefusal, OutputRequest, OutputResponse, OutputService, UartBackend};

/// Lower a request to its wire form (total).
pub fn encode_req(req: &OutputRequest) -> OutReqWire {
    match *req {
        OutputRequest::AttachTx { slot } => OutReqWire {
            op: OutReqWire::OP_ATTACH_TX,
            reserved0: 0,
            reserved1: 0.into(),
            slot: slot.index().into(),
            through: 0.into(),
            reserved2: 0.into(),
            reserved3: 0.into(),
        },
        OutputRequest::Flush { through } => OutReqWire {
            op: OutReqWire::OP_FLUSH,
            reserved0: 0,
            reserved1: 0.into(),
            slot: 0.into(),
            through: through.into(),
            reserved2: 0.into(),
            reserved3: 0.into(),
        },
        OutputRequest::DetachTx => OutReqWire {
            op: OutReqWire::OP_DETACH_TX,
            reserved0: 0,
            reserved1: 0.into(),
            slot: 0.into(),
            through: 0.into(),
            reserved2: 0.into(),
            reserved3: 0.into(),
        },
        OutputRequest::Status => OutReqWire {
            op: OutReqWire::OP_STATUS,
            reserved0: 0,
            reserved1: 0.into(),
            slot: 0.into(),
            through: 0.into(),
            reserved2: 0.into(),
            reserved3: 0.into(),
        },
    }
}

/// Lift a wire request — fail-closed.
pub fn decode_req(w: &OutReqWire) -> Option<OutputRequest> {
    match w.op {
        OutReqWire::OP_ATTACH_TX => Some(OutputRequest::AttachTx {
            slot: SlotRef::from_wire(w.slot.get()),
        }),
        OutReqWire::OP_FLUSH => Some(OutputRequest::Flush {
            through: w.through.get(),
        }),
        OutReqWire::OP_DETACH_TX => Some(OutputRequest::DetachTx),
        OutReqWire::OP_STATUS => Some(OutputRequest::Status),
        _ => None,
    }
}

/// Lower a response to its wire form (total).
pub fn encode_rep(rep: &OutputResponse, grants: u8) -> OutRepWire {
    let (tag, refusal, through) = match *rep {
        OutputResponse::Attached => (OutRepWire::TAG_ATTACHED, 0, 0),
        OutputResponse::Flushed { through } => (OutRepWire::TAG_FLUSHED, 0, through),
        OutputResponse::Detached => (OutRepWire::TAG_DETACHED, 0, 0),
        OutputResponse::Status(s) => (OutRepWire::TAG_STATUS, 0, s.consumed),
        OutputResponse::Refused(r) => (
            OutRepWire::TAG_REFUSED,
            match r {
                OutputRefusal::Busy => OutRepWire::REFUSE_BUSY,
                OutputRefusal::NotAttached => OutRepWire::REFUSE_NOT_ATTACHED,
                OutputRefusal::BadQueue => OutRepWire::REFUSE_BAD_QUEUE,
            },
            0,
        ),
    };
    let (attached, dropped) = match *rep {
        OutputResponse::Status(s) => (s.attached as u8, s.dropped),
        _ => (0, 0),
    };
    OutRepWire {
        tag,
        refusal,
        grants,
        attached,
        reserved: [0; 4],
        through: through.into(),
        dropped: dropped.into(),
    }
}

/// Lift a wire response — fail-closed.
pub fn decode_rep(w: &OutRepWire) -> Option<OutputResponse> {
    match w.tag {
        OutRepWire::TAG_ATTACHED => Some(OutputResponse::Attached),
        OutRepWire::TAG_FLUSHED => Some(OutputResponse::Flushed {
            through: w.through.get(),
        }),
        OutRepWire::TAG_DETACHED => Some(OutputResponse::Detached),
        OutRepWire::TAG_STATUS => Some(OutputResponse::Status(crate::service::OutputStatus {
            attached: match w.attached {
                0 => false,
                1 => true,
                _ => return None,
            },
            consumed: w.through.get(),
            dropped: w.dropped.get(),
        })),
        OutRepWire::TAG_REFUSED => Some(OutputResponse::Refused(match w.refusal {
            OutRepWire::REFUSE_BUSY => OutputRefusal::Busy,
            OutRepWire::REFUSE_NOT_ATTACHED => OutputRefusal::NotAttached,
            OutRepWire::REFUSE_BAD_QUEUE => OutputRefusal::BadQueue,
            _ => return None,
        })),
        _ => None,
    }
}

/// The output service's [`SvcProtocol`] — the wire shapes the generic
/// request path drives.
impl SvcProtocol for OutputService<UartBackend> {
    type Request = OutputRequest;
    type Response = OutputResponse;
    type ReqWire = OutReqWire;
    type RepWire = OutRepWire;
    const REPLY_GRANTS: usize = 0;

    fn encode_req(req: &Self::Request) -> Self::ReqWire {
        encode_req(req)
    }
    fn decode_req(w: &Self::ReqWire) -> Option<Self::Request> {
        decode_req(w)
    }
    fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(w: &Self::RepWire) -> Option<(Self::Response, u8)> {
        decode_rep(w).map(|r| (r, w.grants))
    }
}

/// The host-test backend speaks the SAME wire protocol — one vocabulary,
/// N backends (`res/USERLAND.md` §5.9 criterion 5: a backend swap is an
/// endowment change, never a protocol change).  The codecs live once
/// above; this impl only re-cites them for the loopback instantiation,
/// which the `service` module's tests drive through the framework step.
#[cfg(test)]
impl SvcProtocol for OutputService<crate::service::LoopBackend> {
    type Request = OutputRequest;
    type Response = OutputResponse;
    type ReqWire = OutReqWire;
    type RepWire = OutRepWire;
    const REPLY_GRANTS: usize = 0;

    fn encode_req(req: &Self::Request) -> Self::ReqWire {
        encode_req(req)
    }
    fn decode_req(w: &Self::ReqWire) -> Option<Self::Request> {
        decode_req(w)
    }
    fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(w: &Self::RepWire) -> Option<(Self::Response, u8)> {
        decode_rep(w).map(|r| (r, w.grants))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_every_verb() {
        for req in [
            OutputRequest::AttachTx {
                slot: SlotRef::from_wire(9),
            },
            OutputRequest::Flush { through: 0xABCD },
            OutputRequest::DetachTx,
            OutputRequest::Status,
        ] {
            assert_eq!(decode_req(&encode_req(&req)), Some(req));
        }
    }

    #[test]
    fn response_round_trips_every_outcome() {
        for rep in [
            OutputResponse::Attached,
            OutputResponse::Flushed { through: 42 },
            OutputResponse::Detached,
            OutputResponse::Status(crate::service::OutputStatus {
                attached: true,
                consumed: 7,
                dropped: 0,
            }),
            OutputResponse::Refused(OutputRefusal::Busy),
            OutputResponse::Refused(OutputRefusal::NotAttached),
            OutputResponse::Refused(OutputRefusal::BadQueue),
        ] {
            assert_eq!(decode_rep(&encode_rep(&rep, 0)), Some(rep));
        }
    }

    #[test]
    fn decode_is_fail_closed() {
        assert_eq!(
            decode_req(&OutReqWire {
                op: 0,
                ..encode_req(&OutputRequest::Status)
            }),
            None
        );
        assert_eq!(
            decode_req(&OutReqWire {
                op: 0xEE,
                ..encode_req(&OutputRequest::Status)
            }),
            None
        );
        assert_eq!(
            decode_rep(&OutRepWire {
                tag: OutRepWire::TAG_REFUSED,
                refusal: 0xEE,
                ..encode_rep(&OutputResponse::Refused(OutputRefusal::Busy), 0)
            }),
            None
        );
    }
}
