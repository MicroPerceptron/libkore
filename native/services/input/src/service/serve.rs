//! `service::serve` — the input serve loop (`res/USERLAND.md` §5.5/§5.8):
//! the **multiplexing router**, the fan-in generalization of the
//! single-pair [`ServeHost`](kore_rt::svc::ServeHost).
//!
//! A multiplexing service (`res/DEVICE_SERVICE.md` §4) serves *many* client
//! pairs — one per subscribing app, plus the distinguished focus endpoint.
//! Where [`ServeHost`](kore_rt::svc::ServeHost) drains one pair and cannot
//! tell the service *who* asked (the #299 gap), [`InputHost`] drains a bank
//! of endpoints and **stamps the arrival endpoint's identity**
//! ([`Endpoint`]) onto each request before the sync `serve()` step — so
//! `SetFocus` privilege is structural (which endpoint), never a wire field.
//!
//! One request = pop a wire request off some endpoint's request ring →
//! fail-closed [`decode`](crate::wire::decode_req) → **stamp `via` from that
//! endpoint** → the sync `serve()` (#206, untouched) → push the wire reply
//! into that endpoint's reply ring → notice-kick that client's doorbell.
//! An input reply never carries a grant
//! ([`REPLY_GRANTS`](kore_rt::svc::SvcProtocol::REPLY_GRANTS) `= 0`), so the
//! §5.5 grant-backpressure split is trivially satisfied — no transfer
//! session is ever staged.
//!
//! The **report data plane** does not pass through here: the device-drain
//! half pushes decoded reports straight into the focused subscriber's ring
//! via [`InputService::route`], reached through [`svc_mut`](InputHost::svc_mut).

use core::future::poll_fn;
use core::task::Poll;

use kore_abi::session::{Notice, SlotRef};
use kore_abi::svc::InputReqWire;

use kore_rt::chan::ChanTx;
use kore_rt::service::step;
use kore_rt::svc::{COOKIE_REPLY, COOKIE_REPORT, ServeRings};
use kore_rt::transport::Transport;

use super::{Endpoint, InputService, MAX_INPUT_ENDPOINTS};

/// What one router wake carried: a request on endpoint `k`, or the kernel's
/// terminal quiesce ask (#809 §L) with the remaining budget in nanoseconds.
enum Across {
    Request(usize, InputReqWire),
    Quiesce(u64),
}

/// One client pair the router fans in over: its ring halves (request
/// consumer + reply producer), the send end that kicks *its* doorbell, and
/// the **structural identity** every request off it is stamped with.
pub struct HostEndpoint<'w> {
    rings: ServeRings<'w, InputService>,
    kick: SlotRef,
    identity: Endpoint,
}

/// The router's visible books (the serve-host `leaked_bytes` doctrine:
/// every anomaly is counted, never silent).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputHostStats {
    /// Requests served (decoded + stepped).
    pub served: u64,
    /// Ring records that failed the fail-closed decode — dropped.
    pub wire_refused: u64,
    /// Replies lost to a full reply ring (the client's problem, §5.5).
    pub replies_dropped: u64,
}

/// The input router: owns the bound [`InputService`], a fixed bank of
/// endpoints, and the transport their kicks ride.  [`run`](Self::run) is the
/// §5.5 executor task; [`serve_next`](Self::serve_next) the bounded drive
/// (host tests / a caller-owned drain).
pub struct InputHost<'w, 't, T: Transport> {
    svc: InputService,
    transport: &'t T,
    endpoints: [Option<HostEndpoint<'w>>; MAX_INPUT_ENDPOINTS],
    stats: InputHostStats,
}

impl<'w, 't, T: Transport> InputHost<'w, 't, T> {
    /// A router over a bound service and its transport, with no endpoints
    /// wired yet (the founder / a test [`attach`](Self::attach)es them).
    pub fn new(svc: InputService, transport: &'t T) -> Self {
        Self {
            svc,
            transport,
            endpoints: [const { None }; MAX_INPUT_ENDPOINTS],
            stats: InputHostStats::default(),
        }
    }

    /// Wire one client pair at `idx` with its identity — `false` if `idx` is
    /// out of range or already occupied (the bounded-endpoint refusal).
    pub fn attach(
        &mut self,
        idx: usize,
        rings: ServeRings<'w, InputService>,
        kick: SlotRef,
        identity: Endpoint,
    ) -> bool {
        match self.endpoints.get_mut(idx) {
            Some(slot @ None) => {
                *slot = Some(HostEndpoint {
                    rings,
                    kick,
                    identity,
                });
                true
            }
            _ => false,
        }
    }

    /// The bound service (read access — `summarize`, `stats`, `focus`).
    pub fn svc(&self) -> &InputService {
        &self.svc
    }

    /// The bound service (mutable — the device-drain half's
    /// [`route`](InputService::route) reaches the focused queue through this).
    pub fn svc_mut(&mut self) -> &mut InputService {
        &mut self.svc
    }

    /// The router's books so far.
    pub fn stats(&self) -> InputHostStats {
        self.stats
    }

    /// Serve exactly one pending request from the first ready endpoint, if
    /// any — the **bounded drive**: pop → the §5.5 pipeline.  `false` when
    /// every endpoint's request ring was empty.  The production loop is
    /// [`run`](Self::run).
    pub async fn serve_next(&mut self) -> bool {
        for k in 0..MAX_INPUT_ENDPOINTS {
            let wire = self.endpoints[k].as_mut().and_then(|ep| ep.rings.req.pop());
            if let Some(wire) = wire {
                self.serve_from(k, &wire).await;
                return true;
            }
        }
        false
    }

    /// Serve forever — the §5.5 executor task: level-triggered over every
    /// endpoint's request ring (a client's doorbell kick wakes the executor;
    /// the §3.1 total drain re-polls the fan-in).
    pub async fn run(mut self) -> ! {
        loop {
            // Total drain (§3.1), then await the next arrival on any
            // endpoint — level-triggered, the executor re-polls on a kick.
            while self.serve_next().await {}
            match self.next_across().await {
                Across::Request(k, wire) => self.serve_from(k, &wire).await,
                // The kernel's terminal ask (#809 §L): halt and tear down the
                // claimed xHC; the router keeps serving its endpoints.
                Across::Quiesce(budget_ns) => {
                    let _ = kore_rt::service::SystemService::quiesce_devices(
                        &mut self.svc,
                        kore_rt::executor::quiesce_deadline(budget_ns),
                    )
                    .await;
                }
            }
        }
    }

    /// Await the next request on any endpoint (level-triggered — the
    /// [`Consumer::next`](kore_ring::spsc::Consumer::next) idiom, fanned in).
    async fn next_across(&mut self) -> Across {
        poll_fn(|cx| {
            // Observed inside the poll: the ask is a doorbell wake with no
            // ring work, which the park would otherwise swallow.
            if let Some(budget_ns) = kore_rt::executor::quiesce_asked(cx) {
                return Poll::Ready(Across::Quiesce(budget_ns));
            }
            for k in 0..MAX_INPUT_ENDPOINTS {
                if let Some(w) = self.endpoints[k].as_mut().and_then(|ep| ep.rings.req.pop()) {
                    return Poll::Ready(Across::Request(k, w));
                }
            }
            Poll::Pending
        })
        .await
    }

    /// One request off endpoint `k` through the pinned §5.5 pipeline, with
    /// the **identity stamp** that is this router's whole reason to exist.
    async fn serve_from(&mut self, k: usize, wire: &InputReqWire) {
        // Fail-closed decode: garbage on the ring is dropped + counted.
        let Some(mut req) = crate::wire::decode_req(wire) else {
            self.stats.wire_refused += 1;
            return;
        };
        // THE stamp: the structural caller identity is which endpoint this
        // arrived on — never a wire field (§5.8).
        req.via = self.endpoints[k]
            .as_ref()
            .map_or(Endpoint::Anonymous, |ep| ep.identity);

        // The sync authority join-point (#206) — unchanged, §2.2.  Input
        // grants nothing, so there is no grant session, ever.
        let out = step(&mut self.svc, req);
        let Some(reply) = out.into_reply() else {
            // Every kore service replies; a reply-less step is a service bug.
            self.stats.wire_refused += 1;
            return;
        };

        let ep = match self.endpoints[k].as_mut() {
            Some(ep) => ep,
            None => return,
        };
        if ep
            .rings
            .rep
            .push(crate::wire::encode_rep(&reply, 0))
            .is_err()
        {
            self.stats.replies_dropped += 1; // the client's problem (§5.5)
        } else {
            // `served` counts replies that LANDED — see `ServeStats::served`;
            // every host in the tree agrees on that meaning.
            self.stats.served += 1;
        }

        // The notice-kick: its own session, droppable (§3.1).
        let kick = ep.kick;
        let tx = ChanTx::new(self.transport, kick);
        let _ = tx
            .send(Notice {
                cookie: COOKIE_REPLY,
                payload: 0,
            })
            .await;
    }

    /// The **data-plane wake**: one advisory notice on the focused
    /// subscription's owner endpoint — the entry loop sends it after a
    /// drained batch that delivered (never per report; the queue is the
    /// truth, `route_report` stays lossless-drain and kick-free).  Without
    /// it a consumer parked on its doorbell would never learn a report
    /// landed — control-plane replies were the only kicks (`res/USERLAND.md`
    /// §5.8's known gap).  `false` when nothing holds focus, the owner has
    /// no wired endpoint, or the droppable send did not land — all
    /// non-events by design.
    pub async fn kick_focused(&mut self) -> bool {
        let Some(owner) = self.svc.focused_owner() else {
            return false;
        };
        let Some(kick) = self
            .endpoints
            .iter()
            .flatten()
            .find(|ep| ep.identity == owner)
            .map(|ep| ep.kick)
        else {
            return false;
        };
        let tx = ChanTx::new(self.transport, kick);
        tx.send(Notice {
            cookie: COOKIE_REPORT,
            payload: 0,
        })
        .await
        .is_ok()
    }
}
