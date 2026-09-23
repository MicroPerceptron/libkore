//! `client` — the swap client seam and the pager doorbell.
//!
//! What a lessor issues [`SwapRequest`]s through ([`SwapClient`] — the
//! `res/USERLAND.md` §4 pinned async shape), how fault notices reach it
//! ([`PagerDoorbell`]), the **born-async ring client** ([`RingSwap`] —
//! the §5.6 ring + channel proxy over the generic
//! [`SvcClient`](kore_rt::svc::SvcClient)).
//! **No sync client is ever written** (§5.6): the lessor awaits.

use kore_rt::svc::SvcClient;
use kore_rt::transport::Transport;

use crate::service::{SwapRefusal, SwapRequest, SwapResponse, ZramSwap};

/// The **client** side of the swap protocol (`res/USERLAND.md` §4,
/// pinned): what a lessor issues a [`SwapRequest`] through and awaits a
/// [`SwapResponse`] back.  The lessor holds `impl SwapClient` and never
/// names the concrete backend — so the disk case is a swap, not a
/// rewrite, and the ring transport is an impl, not a redesign.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait SwapClient {
    /// Issue one request and await the reply.
    async fn request(&mut self, req: SwapRequest) -> SwapResponse;
}

/// One pager-doorbell wake, decoded (the pager fault wiring): the kernel consumed
/// the lessor's standing recv post and — for a fault — wrote a
/// [`Notice`](kore_abi::session::Notice) under
/// [`NOTICE_COOKIE_PAGER_FAULT`](kore_abi::session::NOTICE_COOKIE_PAGER_FAULT)
/// into the posted slot, the faulting VA as its payload.  A plain
/// `Notice`, so the ordinary delivery classifier (`take_delivery` /
/// `take_notice`) yields it — the private-kind record it replaced was
/// invisible to that classifier and every fault wake was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagerEvent {
    /// A tenant faulted on the paged-out page containing `va` — restore it.
    Fault { va: u64 },
    /// The recv post was reaped (`Status::Revoked` from the blocking
    /// recv) — the channel or registration died; the loop exits clean.
    Revoked,
}

/// The lessor's blocking doorbell transport seam.
/// On real kore this is the `WAIT_DELIVERY` blocking recv on the
/// registered pager channel (post → park → decode the delivered fault
/// [`Notice`](kore_abi::session::Notice) by its reserved cookie →
/// re-post).
pub trait PagerDoorbell {
    /// Block until the next doorbell wake; yield the decoded event.
    fn recv(&mut self) -> PagerEvent;
}

/// The **ring + channel proxy** (`res/USERLAND.md` §5.6): the swap
/// client over the pair's shared window — request pushed, doorbell
/// kicked, reply awaited on the reply ring.  A swap reply carries no
/// grants (its [`REPLY_GRANTS`](kore_rt::svc::SvcProtocol::REPLY_GRANTS)
/// is zero), so no reply post is ever pinned.
///
/// An **infrastructure** failure of the request path (torn ring,
/// unavailable transport) surfaces as [`SwapRefusal::Backend`] —
/// exactly the vocabulary's "the backing store failed the transfer;
/// the page is not stored / not restored" arm, so the lessor's error
/// handling stays one path.
pub struct RingSwap<'w, 't, T: Transport>(pub SvcClient<'w, 't, T, ZramSwap>);

impl<T: Transport> SwapClient for RingSwap<'_, '_, T> {
    async fn request(&mut self, req: SwapRequest) -> SwapResponse {
        match self.0.call(&req).await {
            Ok(reply) => reply.response,
            Err(_) => SwapResponse::Refused(SwapRefusal::Backend),
        }
    }
}
