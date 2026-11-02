//! `wire` — the swap enum ⇄ wire codecs (`res/USERLAND.md` §5.3).
//!
//! The zerocopy forms live in [`kore_abi::svc`]; decode is fail-closed.
//! A swap service can mint no authority, so its [`SvcProtocol`] declares
//! **zero reply grants** — a client of this protocol pre-pins nothing,
//! and the reply's `grants` byte is pinned zero on encode and refused
//! non-zero on decode (a "grant-bearing" swap reply is a protocol break,
//! not a surprise to absorb).

use kore_abi::session::SlotRef;
use kore_abi::svc::{SwapRepWire, SwapReqWire};

use kore_rt::svc::SvcProtocol;

use crate::extent::{MAX_EXTENT_PAGES, SwapExtent};
use crate::service::{SwapRefusal, SwapRequest, SwapResponse, SwapStatus, ZramSwap};

/// Lower a request to its wire form (total).
pub fn encode_req(req: &SwapRequest) -> SwapReqWire {
    let (op, arg, token, pages) = match *req {
        SwapRequest::Attach { announced } => (SwapReqWire::OP_ATTACH, announced.index(), 0, 0),
        SwapRequest::Out { page } => (SwapReqWire::OP_OUT, page, 0, 0),
        SwapRequest::In { extent, page } => (SwapReqWire::OP_IN, page, extent.token(), 0),
        SwapRequest::Discard { extent } => (SwapReqWire::OP_DISCARD, 0, extent.token(), 0),
        SwapRequest::Store { page, pages } => (SwapReqWire::OP_STORE_EXTENT, page, 0, pages),
        SwapRequest::Load {
            extent,
            page,
            pages,
        } => (SwapReqWire::OP_LOAD_EXTENT, page, extent.token(), pages),
        SwapRequest::Release { extent, pages } => {
            (SwapReqWire::OP_RELEASE_EXTENT, 0, extent.token(), pages)
        }
        SwapRequest::Status => (SwapReqWire::OP_STATUS, 0, 0, 0),
    };
    SwapReqWire {
        op,
        reserved: [0; 3],
        arg: arg.into(),
        token: token.into(),
        pages: pages.into(),
    }
}

/// Lift a wire request — fail-closed.
pub fn decode_req(w: &SwapReqWire) -> Option<SwapRequest> {
    if w.reserved != [0; 3] {
        return None;
    }
    let pages = w.pages.get();
    Some(match w.op {
        SwapReqWire::OP_ATTACH if w.token.get() == 0 && pages == 0 => SwapRequest::Attach {
            announced: SlotRef::from_wire(w.arg.get()),
        },
        SwapReqWire::OP_OUT if w.token.get() == 0 && pages == 0 => {
            SwapRequest::Out { page: w.arg.get() }
        }
        SwapReqWire::OP_IN if pages == 0 => SwapRequest::In {
            extent: SwapExtent::from_raw(w.token.get())?,
            page: w.arg.get(),
        },
        SwapReqWire::OP_DISCARD if w.arg.get() == 0 && pages == 0 => SwapRequest::Discard {
            extent: SwapExtent::from_raw(w.token.get())?,
        },
        SwapReqWire::OP_STORE_EXTENT
            if w.token.get() == 0 && (1..=MAX_EXTENT_PAGES).contains(&pages) =>
        {
            SwapRequest::Store {
                page: w.arg.get(),
                pages,
            }
        }
        SwapReqWire::OP_LOAD_EXTENT if (1..=MAX_EXTENT_PAGES).contains(&pages) => {
            SwapRequest::Load {
                extent: SwapExtent::from_raw(w.token.get())?,
                page: w.arg.get(),
                pages,
            }
        }
        SwapReqWire::OP_RELEASE_EXTENT
            if w.arg.get() == 0 && (1..=MAX_EXTENT_PAGES).contains(&pages) =>
        {
            SwapRequest::Release {
                extent: SwapExtent::from_raw(w.token.get())?,
                pages,
            }
        }
        SwapReqWire::OP_STATUS if w.arg.get() == 0 && w.token.get() == 0 && pages == 0 => {
            SwapRequest::Status
        }
        _ => return None,
    })
}

/// Lower a response (total).  `grants` is pinned zero — see the module
/// docs; the parameter exists so the codec face matches its siblings.
pub fn encode_rep(rep: &SwapResponse, grants: u8) -> SwapRepWire {
    debug_assert_eq!(grants, 0, "a swap service grants nothing, ever");
    let (tag, refusal, token) = match *rep {
        SwapResponse::Stored { extent } => (SwapRepWire::TAG_STORED, 0, extent.token()),
        SwapResponse::Attached { pages } => (SwapRepWire::TAG_ATTACHED, 0, pages),
        SwapResponse::Restored => (SwapRepWire::TAG_RESTORED, 0, 0),
        SwapResponse::Discarded => (SwapRepWire::TAG_DISCARDED, 0, 0),
        SwapResponse::Loaded => (SwapRepWire::TAG_LOADED, 0, 0),
        SwapResponse::Released => (SwapRepWire::TAG_RELEASED, 0, 0),
        SwapResponse::Status(_) => (SwapRepWire::TAG_STATUS, 0, 0),
        SwapResponse::Refused(r) => (
            SwapRepWire::TAG_REFUSED,
            match r {
                SwapRefusal::BadPage => SwapRepWire::REFUSE_BAD_PAGE,
                SwapRefusal::BadDst => SwapRepWire::REFUSE_BAD_DST,
                SwapRefusal::BadToken => SwapRepWire::REFUSE_BAD_TOKEN,
                SwapRefusal::StoreFull => SwapRepWire::REFUSE_STORE_FULL,
                SwapRefusal::Backend => SwapRepWire::REFUSE_BACKEND,
                SwapRefusal::NoStaging => SwapRepWire::REFUSE_NO_STAGING,
            },
            0,
        ),
    };
    // The books ride only the status tag; every other reply zeroes them
    // rather than leaving stale bytes to misread (the input-stream
    // reply's rule).
    let s = match *rep {
        SwapResponse::Status(s) => s,
        _ => SwapStatus::default(),
    };
    SwapRepWire {
        tag,
        refusal,
        grants: 0,
        posture: (if s.device {
            SwapRepWire::STATUS_DEVICE
        } else {
            0
        }) | (if s.route {
            SwapRepWire::STATUS_ROUTE
        } else {
            0
        }),
        token: token.into(),
        capacity_pages: s.capacity_pages.into(),
        stored_pages: s.stored_pages.into(),
        staging_pages: s.staging_pages.into(),
        stored_extents: s.stored_extents.into(),
        outs: s.outs.into(),
        ins: s.ins.into(),
        discards: s.discards.into(),
        io_submitted: s.io_submitted.into(),
        io_completed: s.io_completed.into(),
        io_errors: s.io_errors.into(),
        fires: s.fires.into(),
        parks: s.parks.into(),
        deadline_waits: s.deadline_waits.into(),
        spin_waits: s.spin_waits.into(),
    }
}

fn non_status_books_are_zero(w: &SwapRepWire) -> bool {
    w.posture == 0
        && w.capacity_pages.get() == 0
        && w.stored_pages.get() == 0
        && w.staging_pages.get() == 0
        && w.stored_extents.get() == 0
        && w.outs.get() == 0
        && w.ins.get() == 0
        && w.discards.get() == 0
        && w.io_submitted.get() == 0
        && w.io_completed.get() == 0
        && w.io_errors.get() == 0
        && w.fires.get() == 0
        && w.parks.get() == 0
        && w.deadline_waits.get() == 0
        && w.spin_waits.get() == 0
}

/// Lift a wire reply — fail-closed (including a non-zero `grants`).
pub fn decode_rep(w: &SwapRepWire) -> Option<(SwapResponse, u8)> {
    if w.grants != 0 {
        return None; // a grant-bearing swap reply is a protocol break
    }
    let rep = match (w.tag, w.refusal) {
        (SwapRepWire::TAG_STORED, 0) if non_status_books_are_zero(w) => SwapResponse::Stored {
            extent: SwapExtent::from_raw(w.token.get())?,
        },
        (SwapRepWire::TAG_ATTACHED, 0) if non_status_books_are_zero(w) && w.token.get() != 0 => {
            SwapResponse::Attached {
                pages: w.token.get(),
            }
        }
        (SwapRepWire::TAG_RESTORED, 0) if w.token.get() == 0 && non_status_books_are_zero(w) => {
            SwapResponse::Restored
        }
        (SwapRepWire::TAG_DISCARDED, 0) if w.token.get() == 0 && non_status_books_are_zero(w) => {
            SwapResponse::Discarded
        }
        (SwapRepWire::TAG_LOADED, 0) if w.token.get() == 0 && non_status_books_are_zero(w) => {
            SwapResponse::Loaded
        }
        (SwapRepWire::TAG_RELEASED, 0) if w.token.get() == 0 && non_status_books_are_zero(w) => {
            SwapResponse::Released
        }
        (SwapRepWire::TAG_STATUS, 0) if w.token.get() == 0 => {
            // Fail-closed on posture bits this decoder does not know —
            // a newer tier's semantics must refuse, not silently drop.
            if w.posture & !(SwapRepWire::STATUS_DEVICE | SwapRepWire::STATUS_ROUTE) != 0 {
                return None;
            }
            SwapResponse::Status(SwapStatus {
                device: w.posture & SwapRepWire::STATUS_DEVICE != 0,
                route: w.posture & SwapRepWire::STATUS_ROUTE != 0,
                capacity_pages: w.capacity_pages.get(),
                stored_pages: w.stored_pages.get(),
                stored_extents: w.stored_extents.get(),
                staging_pages: w.staging_pages.get(),
                outs: w.outs.get(),
                ins: w.ins.get(),
                discards: w.discards.get(),
                io_submitted: w.io_submitted.get(),
                io_completed: w.io_completed.get(),
                io_errors: w.io_errors.get(),
                fires: w.fires.get(),
                parks: w.parks.get(),
                deadline_waits: w.deadline_waits.get(),
                spin_waits: w.spin_waits.get(),
            })
        }
        (SwapRepWire::TAG_REFUSED, r) if w.token.get() == 0 && non_status_books_are_zero(w) => {
            SwapResponse::Refused(match r {
                SwapRepWire::REFUSE_BAD_PAGE => SwapRefusal::BadPage,
                SwapRepWire::REFUSE_BAD_DST => SwapRefusal::BadDst,
                SwapRepWire::REFUSE_BAD_TOKEN => SwapRefusal::BadToken,
                SwapRepWire::REFUSE_STORE_FULL => SwapRefusal::StoreFull,
                SwapRepWire::REFUSE_BACKEND => SwapRefusal::Backend,
                SwapRepWire::REFUSE_NO_STAGING => SwapRefusal::NoStaging,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((rep, 0))
}

impl SvcProtocol for ZramSwap {
    type Request = SwapRequest;
    type Response = SwapResponse;
    type ReqWire = SwapReqWire;
    type RepWire = SwapRepWire;

    /// Zero grants, ever — the swap doctrine, now structural: a client
    /// of this protocol pre-pins no reply posts.
    const REPLY_GRANTS: usize = 0;

    fn encode_req(req: &SwapRequest) -> SwapReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &SwapReqWire) -> Option<SwapRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &SwapResponse, grants: u8) -> SwapRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &SwapRepWire) -> Option<(SwapResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(token: u32) -> SwapExtent {
        SwapExtent::from_raw((1 << 16) | token).unwrap()
    }

    /// Every request/response round-trips through its wire form.
    #[test]
    fn round_trips() {
        for req in [
            SwapRequest::Attach {
                announced: SlotRef::from_wire(3),
            },
            SwapRequest::Out { page: 4 },
            SwapRequest::In {
                extent: extent(7),
                page: 5,
            },
            SwapRequest::Discard { extent: extent(9) },
            SwapRequest::Store {
                page: 4,
                pages: 387,
            },
            SwapRequest::Load {
                extent: extent(11),
                page: 32,
                pages: 387,
            },
            SwapRequest::Release {
                extent: extent(11),
                pages: 387,
            },
            SwapRequest::Status,
        ] {
            let back = decode_req(&encode_req(&req)).expect("decodes");
            assert_eq!(encode_req(&back), encode_req(&req));
        }
        for rep in [
            SwapResponse::Stored { extent: extent(3) },
            SwapResponse::Attached { pages: 8 },
            SwapResponse::Restored,
            SwapResponse::Discarded,
            SwapResponse::Loaded,
            SwapResponse::Released,
            SwapResponse::Status(SwapStatus::default()),
            SwapResponse::Status(SwapStatus {
                device: true,
                route: true,
                capacity_pages: 262144,
                stored_pages: 12,
                stored_extents: 2,
                staging_pages: 8,
                outs: 40,
                ins: 30,
                discards: 9,
                io_submitted: 83,
                io_completed: 83,
                io_errors: 1,
                fires: 81,
                parks: 82,
                deadline_waits: 17,
                spin_waits: 5,
            }),
            SwapResponse::Refused(SwapRefusal::BadPage),
            SwapResponse::Refused(SwapRefusal::BadDst),
            SwapResponse::Refused(SwapRefusal::BadToken),
            SwapResponse::Refused(SwapRefusal::StoreFull),
            SwapResponse::Refused(SwapRefusal::Backend),
            SwapResponse::Refused(SwapRefusal::NoStaging),
        ] {
            let (back, grants) = decode_rep(&encode_rep(&rep, 0)).expect("decodes");
            assert_eq!(back, rep);
            assert_eq!(grants, 0);
        }
    }

    /// Decode is fail-closed: unknown discriminants and any claimed
    /// grant refuse.
    #[test]
    fn decode_fails_closed() {
        let mut w = encode_req(&SwapRequest::Discard { extent: extent(1) });
        w.op = 0;
        assert!(decode_req(&w).is_none());
        w.op = 200;
        assert!(decode_req(&w).is_none());

        let mut r = encode_rep(&SwapResponse::Restored, 0);
        r.tag = 0;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&SwapResponse::Refused(SwapRefusal::Backend), 0);
        r.refusal = 42;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&SwapResponse::Stored { extent: extent(1) }, 0);
        r.grants = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a grant-bearing swap reply refuses"
        );
        let mut r = encode_rep(&SwapResponse::Restored, 0);
        r.refusal = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a refusal byte on success refuses"
        );
        let mut r = encode_rep(&SwapResponse::Status(SwapStatus::default()), 0);
        r.posture = 0x80;
        assert!(
            decode_rep(&r).is_none(),
            "an unknown posture bit on a status reply refuses"
        );
        let mut r = encode_rep(&SwapResponse::Loaded, 0);
        r.stored_extents = 1.into();
        assert!(
            decode_rep(&r).is_none(),
            "stale status books on a data reply refuse"
        );

        let mut w = encode_req(&SwapRequest::Store { page: 0, pages: 2 });
        w.pages = 0.into();
        assert!(decode_req(&w).is_none(), "zero-length extent refuses");
        let mut w = encode_req(&SwapRequest::Store { page: 0, pages: 2 });
        w.reserved[0] = 1;
        assert!(decode_req(&w).is_none(), "dirty reserve refuses");
    }

    /// The status books ride ONLY the status tag — every other reply
    /// zeroes them, so a stale record can never be misread as books.
    #[test]
    fn non_status_replies_zero_the_books() {
        let w = encode_rep(&SwapResponse::Stored { extent: extent(3) }, 0);
        assert_eq!(w.posture, 0);
        assert_eq!(w.capacity_pages.get(), 0);
        assert_eq!(w.stored_extents.get(), 0);
        assert_eq!(w.outs.get(), 0);
        assert_eq!(w.parks.get(), 0);
    }
}
