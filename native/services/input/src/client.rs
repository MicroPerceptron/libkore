//! `client` — the input client seam (`res/USERLAND.md` §4/§5.6).
//!
//! What a subscribing app (or the compositor) issues [`InputRequest`]s
//! through and awaits an [`InputResponse`] back.  [`InputClient`] is the §4
//! pinned async trait; [`RingInput`] the born-async ring + channel proxy
//! (§5.6) over the generic [`SvcClient`]. **No sync client is ever written**
//! (§5.6).
//!
//! **Identity is the endpoint's, not the request's** (§5.8): a client holds
//! a *specific* endpoint, and that endpoint's identity is what the service
//! sees — the ring proxy's is stamped service-side by the [`InputHost`]
//! router. The caller cannot forge privilege by setting
//! [`InputRequest::via`] — the endpoint wins.
//!
//! [`InputHost`]: crate::service::InputHost
//! [`SvcClient`]: kore_rt::svc::SvcClient

use kore_rt::svc::SvcClient;
use kore_rt::transport::Transport;

use crate::service::{InputRequest, InputResponse, InputService};

/// The **client** side of the input protocol (`res/USERLAND.md` §4,
/// pinned): issue one request, await the data-only reply.  The app holds
/// `impl InputClient` and never names the transport — the ring proxy is an
/// impl, not a redesign.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait InputClient {
    /// Issue one control request and await its data-only reply.
    async fn request(&mut self, req: InputRequest) -> InputResponse;
}

/// The **ring + channel proxy** (`res/USERLAND.md` §5.6): the input client
/// over the pair's shared window — request pushed, doorbell kicked, reply
/// awaited on the reply ring.  An input reply carries no grants (its
/// [`REPLY_GRANTS`](kore_rt::svc::SvcProtocol::REPLY_GRANTS) is zero), so no
/// reply post is ever pinned.
///
/// An **infrastructure** failure of the request path (torn ring,
/// unavailable transport) **panics** (the executor's transport-failure
/// posture, the image precedent): the closed [`InputRefusal`] vocabulary
/// has no honest infra arm, and a torn pair masquerading as a refusal would
/// corrupt the app's trust in routing.
///
/// [`InputRefusal`]: crate::service::InputRefusal
pub struct RingInput<'w, 't, T: Transport>(pub SvcClient<'w, 't, T, InputService>);

impl<T: Transport> InputClient for RingInput<'_, '_, T> {
    async fn request(&mut self, req: InputRequest) -> InputResponse {
        self.0
            .call(&req)
            .await
            .expect("input request path failed (torn rings/transport — a broken pair)")
            .response
    }
}
