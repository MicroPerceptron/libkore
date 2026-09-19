//! One frontend exchange over an already-held HostLink face. Request custody
//! outlives a borrowed send/poll future, so dropping a wait cannot resend a
//! claim/result or turn a later reply into a different frontend operation.

use kore_hostlink::{
    CANCELLED, FRONTEND_TEXT_MAX, FrontendIntentId, FrontendPoll, FrontendPollPhase,
    FrontendResult, INFER_REPLY, INFERENCE_PART_MAX, InferenceEventKind, InferenceRequestKind,
    InferenceRequestView, REFUSED, RequestId, decode_frontend_intent, decode_inference_event,
    decode_refusal,
};
use kore_rt::transport::Transport;

use crate::{CLIENT_CHUNK_MAX, ClientError, ClientEvent, RingHostLink, StatusReason};

#[derive(Debug, PartialEq, Eq)]
pub struct ClaimedIntent {
    pub id: FrontendIntentId,
    pub text: heapless::String<FRONTEND_TEXT_MAX>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ClaimedUpdate {
    pub id: FrontendIntentId,
    pub revision: u32,
    pub text: heapless::String<FRONTEND_TEXT_MAX>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FrontendProgress {
    Pending,
    Intent(ClaimedIntent),
    Control(kore_hostlink::FrontendControl),
    Update(ClaimedUpdate),
    /// The host published this intent's terminal outcome. An owning broker's
    /// failure fallback can receive the winning worker result's receipt when
    /// publication races the deadline; it does not publish another response.
    /// For BrokerRetired, acknowledges recording that intent's retirement.
    Receipt,
    /// Preserve events belonging to another operation for the resident owner.
    Other(ClientEvent),
}

#[derive(Debug)]
pub enum FrontendError {
    InvalidRequest,
    AlreadySent,
    NotPending,
    InvalidReply,
    Status(StatusReason),
    Transport(ClientError),
}

#[derive(Clone, Copy)]
enum Expected {
    Intent(Option<FrontendIntentId>),
    Receipt,
    Control(FrontendIntentId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Fresh,
    Sending,
    Awaiting,
    Finished,
}

/// This owner is intentionally neither Clone nor resettable. Construct it
/// with a fresh per-route request correlation for each exchange. Only feed
/// replies obtained from the same held face used by `send`: IDs correlate
/// replies; the service's structural pair supplies provenance/authority.
pub struct FrontendCall {
    request: RequestId,
    expected: Expected,
    state: State,
    wire: [u8; CLIENT_CHUNK_MAX],
    len: usize,
}

/// Select the unique owner of a reply from one already-authenticated held
/// broker face. Callers derive candidates from their live exchanges; this
/// stores no parallel request table and grants no authority from an ID.
pub fn frontend_reply_owner(
    event: &ClientEvent,
    calls: impl IntoIterator<Item = (usize, RequestId)>,
) -> Option<usize> {
    let request = match event {
        ClientEvent::Control { request, .. }
        | ClientEvent::Cancel { request }
        | ClientEvent::BulkBegin { request, .. }
        | ClientEvent::BulkChunk { request, .. }
        | ClientEvent::BulkReady { request, .. }
        | ClientEvent::BulkAbort { request, .. } => *request,
        ClientEvent::Status(status) => status.request,
    };
    let mut owner = None;
    for (lane, pending) in calls {
        if pending == request {
            if owner.is_some() {
                return None;
            }
            owner = Some(lane);
        }
    }
    owner
}

impl FrontendCall {
    pub fn poll_intent(
        request: RequestId,
        poll: FrontendPoll,
        timeout_ticks: u32,
    ) -> Result<Self, FrontendError> {
        let mut inner = [0; INFERENCE_PART_MAX];
        let len = poll
            .encode(&mut inner)
            .ok_or(FrontendError::InvalidRequest)?;
        let expected = match poll.phase {
            FrontendPollPhase::BrokerNext | FrontendPollPhase::BrokerReadyNext => {
                Expected::Intent(None)
            }
            FrontendPollPhase::AgentClaim => Expected::Intent(Some(poll.intent_id)),
            FrontendPollPhase::BrokerRetired => Expected::Receipt,
            FrontendPollPhase::BrokerControl => Expected::Control(poll.intent_id),
        };
        Self::new(
            request,
            expected,
            InferenceRequestKind::FrontendPoll,
            &inner[..len],
            timeout_ticks,
        )
    }

    pub fn result(
        request: RequestId,
        result: FrontendResult<'_>,
        timeout_ticks: u32,
    ) -> Result<Self, FrontendError> {
        let mut inner = [0; INFERENCE_PART_MAX];
        let len = result
            .encode(&mut inner)
            .ok_or(FrontendError::InvalidRequest)?;
        Self::new(
            request,
            Expected::Receipt,
            InferenceRequestKind::FrontendResult,
            &inner[..len],
            timeout_ticks,
        )
    }

    pub fn update_result(
        request: RequestId,
        result: kore_hostlink::FrontendUpdateResult<'_>,
        timeout_ticks: u32,
    ) -> Result<Self, FrontendError> {
        let mut inner = [0; INFERENCE_PART_MAX];
        let len = result
            .encode(&mut inner)
            .ok_or(FrontendError::InvalidRequest)?;
        Self::new(
            request,
            Expected::Receipt,
            InferenceRequestKind::FrontendUpdateResult,
            &inner[..len],
            timeout_ticks,
        )
    }

    fn new(
        request: RequestId,
        expected: Expected,
        kind: InferenceRequestKind,
        input: &[u8],
        timeout_ticks: u32,
    ) -> Result<Self, FrontendError> {
        let mut call = Self {
            request,
            expected,
            state: State::Fresh,
            wire: [0; CLIENT_CHUNK_MAX],
            len: 0,
        };
        call.len = InferenceRequestView {
            kind,
            turn: 1,
            max_output_tokens: 1,
            timeout_ticks,
            input,
        }
        .encode(&mut call.wire)
        .ok_or(FrontendError::InvalidRequest)?;
        Ok(call)
    }

    pub const fn request(&self) -> RequestId {
        self.request
    }

    pub async fn send<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
    ) -> Result<(), FrontendError> {
        if self.state != State::Fresh {
            return Err(FrontendError::AlreadySent);
        }
        self.advance_send_with_budget(link, None).await
    }

    pub fn pending_send(&self) -> bool {
        matches!(self.state, State::Fresh | State::Sending)
    }

    /// Continue one owned send in a bounded observation slice. A timeout
    /// preserves its request, payload and expected frontend receipt. Only an
    /// explicit non-publication/refusal permits another publication attempt.
    pub async fn advance_send<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<(), FrontendError> {
        self.advance_send_with_budget(link, Some(budget_ns)).await
    }

    async fn advance_send_with_budget<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<(), FrontendError> {
        if !self.pending_send() {
            return Err(FrontendError::AlreadySent);
        }
        // Suspending before publication retains this call while the pair may
        // still be idle. Only a published operation needs an exact resume.
        let resume = self.state == State::Sending && !link.is_idle();
        // Set before awaiting: an uncertain service publication retains the
        // operation, even if the caller abandons this borrowed future.
        self.state = State::Sending;
        match link
            .observe_frontend_send(self.request, &self.wire[..self.len], resume, budget_ns)
            .await
        {
            Ok(()) => {
                self.state = State::Awaiting;
                Ok(())
            }
            Err(error) => {
                if matches!(
                    error,
                    ClientError::Refused(kore_hostlink::RefusalCode::Backpressure)
                ) || (!resume
                    && matches!(
                        error,
                        ClientError::Call(
                            kore_rt::svc::CallError::RingFull
                                | kore_rt::svc::CallError::Outstanding
                        )
                    ))
                {
                    self.state = State::Fresh;
                }
                Err(FrontendError::Transport(error))
            }
        }
    }

    pub async fn poll<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
    ) -> Result<FrontendProgress, FrontendError> {
        if self.state != State::Awaiting {
            return Err(FrontendError::NotPending);
        }
        let event = link.poll().await.map_err(FrontendError::Transport)?;
        match event {
            Some(event) => self.accept(event),
            None => Ok(FrontendProgress::Pending),
        }
    }

    /// Drive one bounded send or receipt observation on a private worker face.
    /// Local acceptance does not complete a claim/result; only `accept` can
    /// validate the separately polled frontend event. Keep this call across
    /// observation timeouts, and retire the face if its terminal wait expires.
    pub async fn observe<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<FrontendProgress, FrontendError> {
        self.observe_with_budget(link, Some(budget_ns)).await
    }

    async fn observe_with_budget<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<FrontendProgress, FrontendError> {
        if self.pending_send() {
            return match self.advance_send_with_budget(link, budget_ns).await {
                Ok(()) => Ok(FrontendProgress::Pending),
                Err(FrontendError::Transport(error))
                    if crate::client::retry_observation(&error) =>
                {
                    Ok(FrontendProgress::Pending)
                }
                Err(error) => Err(error),
            };
        }
        if self.state != State::Awaiting {
            return Err(FrontendError::NotPending);
        }
        let resume = link.poll_request_pending();
        match link.poll_observation(resume, budget_ns).await {
            Ok(Some(event)) => self.accept(event),
            Ok(None) => Ok(FrontendProgress::Pending),
            Err(error) if crate::client::retry_observation(&error) => Ok(FrontendProgress::Pending),
            Err(error) => Err(FrontendError::Transport(error)),
        }
    }

    /// Dispatch a mailbox event read by the resident owner of this call's
    /// held face. A matching malformed/terminal event consumes the exchange;
    /// duplicate results cannot satisfy a subsequent wait on this owner.
    pub fn accept(&mut self, event: ClientEvent) -> Result<FrontendProgress, FrontendError> {
        if self.state != State::Awaiting {
            return Err(FrontendError::NotPending);
        }
        match event {
            ClientEvent::Control {
                kind,
                request,
                payload,
            } if request == self.request => {
                self.state = State::Finished;
                if kind == REFUSED {
                    let refusal =
                        decode_refusal(payload.as_slice()).ok_or(FrontendError::InvalidReply)?;
                    return Err(FrontendError::Status(StatusReason::Refused(refusal.code)));
                }
                if kind == CANCELLED {
                    return Err(FrontendError::Status(StatusReason::Cancelled));
                }
                if kind != INFER_REPLY {
                    return Err(FrontendError::InvalidReply);
                }
                let reply = decode_inference_event(payload.as_slice())
                    .ok_or(FrontendError::InvalidReply)?;
                if reply.kind != InferenceEventKind::TurnFinished
                    || !reply.terminal
                    || reply.turn != 1
                    || reply.part != 0
                {
                    return Err(FrontendError::InvalidReply);
                }
                match self.expected {
                    Expected::Control(expected) => {
                        if let Some(control) = kore_hostlink::FrontendControl::decode(reply.payload)
                        {
                            return Ok(FrontendProgress::Control(control));
                        }
                        let update = kore_hostlink::decode_frontend_update(reply.payload)
                            .filter(|update| update.intent_id == expected)
                            .ok_or(FrontendError::InvalidReply)?;
                        Ok(FrontendProgress::Update(ClaimedUpdate {
                            id: update.intent_id,
                            revision: update.revision,
                            text: heapless::String::try_from(update.text)
                                .map_err(|_| FrontendError::InvalidReply)?,
                        }))
                    }
                    Expected::Intent(expected) => {
                        let intent = decode_frontend_intent(reply.payload)
                            .ok_or(FrontendError::InvalidReply)?;
                        if expected.is_some_and(|id| id != intent.intent_id) {
                            return Err(FrontendError::InvalidReply);
                        }
                        Ok(FrontendProgress::Intent(ClaimedIntent {
                            id: intent.intent_id,
                            text: heapless::String::try_from(intent.text)
                                .map_err(|_| FrontendError::InvalidReply)?,
                        }))
                    }
                    Expected::Receipt if reply.payload.is_empty() => Ok(FrontendProgress::Receipt),
                    Expected::Receipt => Err(FrontendError::InvalidReply),
                }
            }
            ClientEvent::Status(status) if status.request == self.request => {
                self.state = State::Finished;
                Err(FrontendError::Status(status.reason))
            }
            ClientEvent::Cancel { request } if request == self.request => {
                self.state = State::Finished;
                Err(FrontendError::Status(StatusReason::Cancelled))
            }
            other => Ok(FrontendProgress::Other(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Payload, StatusNotice};
    use kore_hostlink::{FrontendIntent, FrontendResultKind, InferenceEventView};

    fn request(n: u8) -> RequestId {
        RequestId([n; 16])
    }

    #[test]
    fn frontend_update_delivery_matches_exact_control_intent_and_uses_separate_result_kind() {
        for target in [intent(7), intent(8)] {
            let mut call = FrontendCall::poll_intent(
                request(1),
                FrontendPoll {
                    phase: FrontendPollPhase::BrokerControl,
                    intent_id: intent(7),
                },
                20_000,
            )
            .unwrap();
            call.state = State::Awaiting;
            let mut bytes = [0; INFERENCE_PART_MAX];
            let len = kore_hostlink::FrontendUpdate {
                intent_id: target,
                revision: 9,
                text: "Revoke input.",
            }
            .encode(&mut bytes)
            .unwrap();
            let progress = call.accept(event(request(1), &bytes[..len], 1, 0, true));
            if target == intent(7) {
                assert!(
                    matches!(progress, Ok(FrontendProgress::Update(ClaimedUpdate { id, revision: 9, .. })) if id == intent(7))
                );
            } else {
                assert!(matches!(progress, Err(FrontendError::InvalidReply)));
            }
        }
        let mut call = FrontendCall::update_result(
            request(2),
            kore_hostlink::FrontendUpdateResult {
                intent_id: intent(7),
                revision: 9,
                outcome: kore_hostlink::FrontendUpdateOutcome::Applied,
                text: "",
            },
            20_000,
        )
        .unwrap();
        let wire = kore_hostlink::decode_inference_request(&call.wire[..call.len]).unwrap();
        assert_eq!(wire.kind, InferenceRequestKind::FrontendUpdateResult);
        assert!(kore_hostlink::decode_frontend_result(wire.input).is_none());
        call.state = State::Awaiting;
        assert_eq!(
            call.accept(event(request(2), &[], 1, 0, true)).unwrap(),
            FrontendProgress::Receipt
        );
    }
    fn intent(n: u128) -> FrontendIntentId {
        FrontendIntentId::from_sequence(n).unwrap()
    }

    #[test]
    fn concurrent_frontend_reply_ownership_is_exact_and_never_guessed() {
        let calls = [
            (0, request(1)),
            (0, request(2)),
            (1, request(3)),
            (1, request(4)),
        ];
        for (owner, request) in calls {
            for event in [
                ClientEvent::Control {
                    kind: kore_hostlink::INFER_REPLY,
                    request,
                    payload: Payload::copy_from(&[]).unwrap(),
                },
                ClientEvent::Cancel { request },
                ClientEvent::Status(StatusNotice {
                    request,
                    reason: StatusReason::Cancelled,
                }),
            ] {
                assert_eq!(frontend_reply_owner(&event, calls), Some(owner));
                let remaining = calls.into_iter().filter(|(_, pending)| *pending != request);
                assert_eq!(frontend_reply_owner(&event, remaining), None);
                assert_eq!(
                    frontend_reply_owner(&event, [(0, request), (1, request)]),
                    None
                );
            }
        }
        assert_eq!(
            frontend_reply_owner(
                &ClientEvent::Cancel {
                    request: request(99)
                },
                calls
            ),
            None
        );
    }

    // Pure decoder tests start after service acceptance. The native borrowed
    // send interruption is separately exercised below against real SPSC rings.
    fn awaiting(expected: Option<FrontendIntentId>) -> FrontendCall {
        let mut call = FrontendCall::poll_intent(
            request(1),
            FrontendPoll {
                phase: if expected.is_some() {
                    FrontendPollPhase::AgentClaim
                } else {
                    FrontendPollPhase::BrokerNext
                },
                intent_id: expected.unwrap_or(FrontendIntentId::ZERO),
            },
            20_000,
        )
        .unwrap();
        call.state = State::Awaiting;
        call
    }

    fn event(
        request: RequestId,
        payload: &[u8],
        turn: u64,
        part: u16,
        terminal: bool,
    ) -> ClientEvent {
        let mut bytes = [0; CLIENT_CHUNK_MAX];
        let len = InferenceEventView {
            kind: if terminal {
                InferenceEventKind::TurnFinished
            } else {
                InferenceEventKind::AssistantPart
            },
            terminal,
            part,
            prompt_tokens: 0,
            completion_tokens: 0,
            turn,
            payload,
        }
        .encode(&mut bytes)
        .unwrap();
        ClientEvent::Control {
            kind: INFER_REPLY,
            request,
            payload: Payload::copy_from(&bytes[..len]).unwrap(),
        }
    }

    #[test]
    fn concurrent_claim_replies_preserve_each_lane_and_ready_next_claims_a_new_intent() {
        let mut calls = [
            FrontendCall::poll_intent(
                request(1),
                FrontendPoll {
                    phase: FrontendPollPhase::BrokerNext,
                    intent_id: FrontendIntentId::ZERO,
                },
                20_000,
            )
            .unwrap(),
            FrontendCall::poll_intent(
                request(2),
                FrontendPoll {
                    phase: FrontendPollPhase::BrokerReadyNext,
                    intent_id: intent(7),
                },
                20_000,
            )
            .unwrap(),
        ];
        for call in &mut calls {
            call.state = State::Awaiting;
        }
        for (lane, id, text) in [
            (1, intent(41), "second lane"),
            (0, intent(42), "first lane"),
        ] {
            let mut bytes = [0; INFERENCE_PART_MAX];
            let len = FrontendIntent {
                intent_id: id,
                text,
            }
            .encode(&mut bytes)
            .unwrap();
            let reply = event(calls[lane].request(), &bytes[..len], 1, 0, true);
            let owner = frontend_reply_owner(
                &reply,
                calls
                    .iter()
                    .enumerate()
                    .filter(|(_, call)| call.state == State::Awaiting)
                    .map(|(index, call)| (index, call.request())),
            )
            .unwrap();
            assert_eq!(owner, lane);
            assert_eq!(
                calls[owner].accept(reply.clone()).unwrap(),
                FrontendProgress::Intent(ClaimedIntent {
                    id,
                    text: heapless::String::try_from(text).unwrap()
                })
            );
            assert_eq!(
                frontend_reply_owner(
                    &reply,
                    calls
                        .iter()
                        .enumerate()
                        .filter(|(_, call)| call.state == State::Awaiting)
                        .map(|(index, call)| (index, call.request()))
                ),
                None
            );
        }
        assert!(calls.iter().all(|call| call.state == State::Finished));
    }

    #[test]
    fn claims_preserve_authenticated_text_and_require_exact_worker_intent() {
        let mut bytes = [0; INFERENCE_PART_MAX];
        let text = "User text: <tool> is data; δ stays unchanged.";
        let len = FrontendIntent {
            intent_id: intent(17),
            text,
        }
        .encode(&mut bytes)
        .unwrap();
        for expected in [None, Some(intent(17))] {
            let mut call = awaiting(expected);
            let reply = event(request(1), &bytes[..len], 1, 0, true);
            assert_eq!(
                call.accept(reply.clone()).unwrap(),
                FrontendProgress::Intent(ClaimedIntent {
                    id: intent(17),
                    text: heapless::String::try_from(text).unwrap(),
                })
            );
            assert!(matches!(call.accept(reply), Err(FrontendError::NotPending)));
        }
        assert!(matches!(
            awaiting(Some(intent(18))).accept(event(request(1), &bytes[..len], 1, 0, true)),
            Err(FrontendError::InvalidReply)
        ));
    }

    #[test]
    fn unrelated_requests_are_returned_and_cannot_satisfy_a_pending_claim() {
        let mut call = awaiting(Some(intent(1)));
        let other = event(request(2), b"forged claim", 1, 0, true);
        assert_eq!(
            call.accept(other.clone()).unwrap(),
            FrontendProgress::Other(other)
        );
        assert!(call.state == State::Awaiting);
        let status = ClientEvent::Status(StatusNotice {
            request: request(3),
            reason: StatusReason::HostDisconnected,
        });
        assert_eq!(
            call.accept(status.clone()).unwrap(),
            FrontendProgress::Other(status)
        );
        assert!(call.state == State::Awaiting);
    }

    #[test]
    fn malformed_or_streamed_claim_is_terminal_failure_not_frontend_provenance() {
        let mut bytes = [0; INFERENCE_PART_MAX];
        let len = FrontendIntent {
            intent_id: intent(1),
            text: "claimed",
        }
        .encode(&mut bytes)
        .unwrap();
        for (turn, part, terminal) in [(2, 0, true), (1, 1, true), (1, 0, false)] {
            let mut call = awaiting(Some(intent(1)));
            assert!(matches!(
                call.accept(event(request(1), &bytes[..len], turn, part, terminal)),
                Err(FrontendError::InvalidReply)
            ));
            assert!(call.state == State::Finished);
        }
        let mut call = awaiting(None);
        assert!(matches!(
            call.accept(event(request(1), b"unframed model text", 1, 0, true)),
            Err(FrontendError::InvalidReply)
        ));
        assert!(call.state == State::Finished);
    }

    #[test]
    fn retirement_call_waits_for_its_own_empty_terminal_receipt() {
        let mut call = FrontendCall::poll_intent(
            request(1),
            FrontendPoll {
                phase: FrontendPollPhase::BrokerRetired,
                intent_id: intent(7),
            },
            20_000,
        )
        .unwrap();
        let wire = kore_hostlink::decode_inference_request(&call.wire[..call.len]).unwrap();
        assert_eq!(
            kore_hostlink::decode_frontend_poll(wire.input)
                .unwrap()
                .phase,
            FrontendPollPhase::BrokerRetired
        );
        call.state = State::Awaiting;
        assert!(matches!(
            call.accept(event(request(2), &[], 1, 0, true)),
            Ok(FrontendProgress::Other(_))
        ));
        assert_eq!(
            call.accept(event(request(1), &[], 1, 0, true)).unwrap(),
            FrontendProgress::Receipt
        );
        assert!(matches!(
            call.accept(event(request(1), &[], 1, 0, true)),
            Err(FrontendError::NotPending)
        ));
    }

    #[test]
    fn control_call_preserves_planning_replies_and_rejects_untyped_cancellation() {
        for payload in [&[1][..], b"cancel", &[][..], &[1, 0][..]] {
            let mut call = FrontendCall::poll_intent(
                request(1),
                FrontendPoll {
                    phase: FrontendPollPhase::BrokerControl,
                    intent_id: intent(2),
                },
                20_000,
            )
            .unwrap();
            call.state = State::Awaiting;
            let planning = event(request(9), b"planning response", 1, 0, true);
            assert_eq!(
                call.accept(planning.clone()).unwrap(),
                FrontendProgress::Other(planning)
            );
            let result = call.accept(event(request(1), payload, 1, 0, true));
            if payload == [1] {
                assert_eq!(
                    result.unwrap(),
                    FrontendProgress::Control(kore_hostlink::FrontendControl::CancelRequested)
                );
            } else {
                assert!(matches!(result, Err(FrontendError::InvalidReply)));
            }
        }
    }

    #[test]
    fn frontend_receipt_requires_an_empty_exact_terminal_reply() {
        for payload in [&b""[..], &b"unexpected text"[..]] {
            let mut call = FrontendCall::result(
                request(1),
                FrontendResult {
                    intent_id: intent(2),
                    kind: FrontendResultKind::Completed,
                    text: "Done: δ",
                },
                20_000,
            )
            .unwrap();
            let wire = kore_hostlink::decode_inference_request(&call.wire[..call.len]).unwrap();
            let result = kore_hostlink::decode_frontend_result(wire.input).unwrap();
            assert_eq!(result.text, "Done: δ");
            assert_eq!(result.intent_id, intent(2));
            call.state = State::Awaiting;
            let progress = call.accept(event(request(1), payload, 1, 0, true));
            if payload.is_empty() {
                assert_eq!(progress.unwrap(), FrontendProgress::Receipt);
            } else {
                assert!(matches!(progress, Err(FrontendError::InvalidReply)));
            }
            assert!(call.state == State::Finished);
        }
    }

    #[test]
    fn matching_cancellation_and_disconnect_finish_only_the_matching_exchange() {
        for event in [
            ClientEvent::Cancel {
                request: request(1),
            },
            ClientEvent::Status(StatusNotice {
                request: request(1),
                reason: StatusReason::HostDisconnected,
            }),
            ClientEvent::Status(StatusNotice {
                request: request(1),
                reason: StatusReason::Refused(kore_hostlink::RefusalCode::PolicyDenied),
            }),
        ] {
            let mut call = awaiting(None);
            assert!(matches!(call.accept(event), Err(FrontendError::Status(_))));
            assert!(call.state == State::Finished);
        }
    }

    #[test]
    fn abandoned_native_send_cannot_republish_the_frontend_request() {
        use crate::HostLinkWire;
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::{Committed, SlotRef};
        use kore_rt::{
            session::SessionBuilder,
            svc::{ClientEndpoints, SvcClient, split_window},
        };
        struct PendingTransport;
        impl Transport for PendingTransport {
            type Error = ();
            async fn commit(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                core::future::pending().await
            }
            fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                Err(())
            }
        }
        #[repr(align(64))]
        struct Window([u8; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let mut window = Window([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let (rings, mut serve) = split_window::<HostLinkWire>(
            &mut window.0,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        let transport = PendingTransport;
        let mut link = RingHostLink::from_pair(
            SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(30),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling(),
        );
        let mut call = FrontendCall::poll_intent(
            request(7),
            FrontendPoll {
                phase: FrontendPollPhase::BrokerNext,
                intent_id: FrontendIntentId::ZERO,
            },
            20_000,
        )
        .unwrap();
        let mut context = Context::from_waker(Waker::noop());
        {
            let mut send = core::pin::pin!(call.send(&mut link));
            assert!(send.as_mut().poll(&mut context).is_pending());
            assert!(
                serve.req.pop().is_some(),
                "first service request was published"
            );
        }
        assert!(call.state == State::Sending);
        {
            let mut retry = core::pin::pin!(call.send(&mut link));
            assert!(matches!(
                retry.as_mut().poll(&mut context),
                Poll::Ready(Err(FrontendError::AlreadySent))
            ));
        }
        assert!(
            serve.req.pop().is_none(),
            "no duplicate claim after an uncertain kick"
        );
        let mut poll = core::pin::pin!(call.poll(&mut link));
        assert!(matches!(
            poll.as_mut().poll(&mut context),
            Poll::Ready(Err(FrontendError::NotPending))
        ));
    }

    #[test]
    fn worker_frontend_observation_resumes_claim_and_result_without_duplicate_publication() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::{Committed, SlotRef, Status};
        use kore_rt::{
            session::SessionBuilder,
            svc::{ClientEndpoints, SvcClient, SvcProtocol, split_window},
        };
        struct AcceptKick;
        impl Transport for AcceptKick {
            type Error = ();
            async fn commit(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
            fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
        }
        #[repr(align(64))]
        struct Window([u8; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        for claiming in [false, true] {
            let mut window = Window([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (rings, mut serve) = split_window::<crate::HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    rings,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut call = if claiming {
                FrontendCall::poll_intent(
                    request(51),
                    FrontendPoll {
                        phase: FrontendPollPhase::AgentClaim,
                        intent_id: intent(52),
                    },
                    20_000,
                )
            } else {
                FrontendCall::result(
                    request(51),
                    FrontendResult {
                        intent_id: intent(52),
                        kind: FrontendResultKind::Completed,
                        text: "Done: δ",
                    },
                    20_000,
                )
            }
            .unwrap();
            // The borrowed send was suspended before its request entered the
            // SPSC ring. Attempt metadata is not publication evidence.
            call.state = State::Sending;
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut observation = core::pin::pin!(call.observe_with_budget(&mut link, None));
                assert!(observation.as_mut().poll(&mut context).is_pending());
            }
            let original = serve.req.pop().unwrap();
            assert!(
                matches!(crate::HostLinkWire::decode_req(&original), Some(crate::ClientRequest::Send { request: got, .. }) if got == request(51))
            );
            {
                let mut observation = core::pin::pin!(call.observe_with_budget(&mut link, None));
                assert!(observation.as_mut().poll(&mut context).is_pending());
            }
            assert!(serve.req.pop().is_none());
            serve
                .rep
                .push(crate::HostLinkWire::encode_rep(
                    &crate::ClientResponse::Accepted {
                        request: request(51),
                    },
                    0,
                ))
                .unwrap();
            {
                let mut observation = core::pin::pin!(call.observe_with_budget(&mut link, None));
                assert!(matches!(
                    observation.as_mut().poll(&mut context),
                    Poll::Ready(Ok(FrontendProgress::Pending))
                ));
            }
            assert!(call.state == State::Awaiting);
            assert!(
                serve.req.pop().is_none(),
                "send acceptance does not dequeue the receipt"
            );
            let mut intent_bytes = [0; INFERENCE_PART_MAX];
            let intent_len = FrontendIntent {
                intent_id: intent(52),
                text: "Claim: δ",
            }
            .encode(&mut intent_bytes)
            .unwrap();
            let receipt = event(
                request(51),
                if claiming {
                    &intent_bytes[..intent_len]
                } else {
                    &[]
                },
                1,
                0,
                true,
            );
            for reply in [event(request(53), &[], 1, 0, true), receipt] {
                {
                    let mut observation =
                        core::pin::pin!(call.observe_with_budget(&mut link, None));
                    assert!(observation.as_mut().poll(&mut context).is_pending());
                }
                assert_eq!(
                    crate::HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                    Some(crate::ClientRequest::Poll)
                );
                {
                    let mut observation =
                        core::pin::pin!(call.observe_with_budget(&mut link, None));
                    assert!(observation.as_mut().poll(&mut context).is_pending());
                }
                assert!(
                    serve.req.pop().is_none(),
                    "dropped Poll retains the original dequeue"
                );
                let foreign = matches!(&reply, ClientEvent::Control { request: got, .. } if *got == request(53));
                serve
                    .rep
                    .push(crate::HostLinkWire::encode_rep(
                        &crate::ClientResponse::Event(reply),
                        0,
                    ))
                    .unwrap();
                {
                    let mut observation =
                        core::pin::pin!(call.observe_with_budget(&mut link, None));
                    match observation.as_mut().poll(&mut context) {
                        Poll::Ready(Ok(FrontendProgress::Other(_))) => assert!(foreign),
                        Poll::Ready(Ok(FrontendProgress::Intent(got))) => {
                            assert!(claiming && !foreign);
                            assert_eq!(got.id, intent(52));
                            assert_eq!(got.text, "Claim: δ");
                        }
                        Poll::Ready(Ok(FrontendProgress::Receipt)) => {
                            assert!(!claiming && !foreign)
                        }
                        other => panic!("unexpected frontend progress: {other:?}"),
                    }
                }
                assert!(serve.req.pop().is_none());
            }
            assert!(call.state == State::Finished);
            let mut observation = core::pin::pin!(call.observe_with_budget(&mut link, None));
            assert!(matches!(
                observation.as_mut().poll(&mut context),
                Poll::Ready(Err(FrontendError::NotPending))
            ));
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn frontend_send_resume_preserves_envelope_and_requires_separate_receipt() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::{Committed, SlotRef, Status};
        use kore_rt::{
            session::SessionBuilder,
            svc::{ClientEndpoints, SvcClient, SvcProtocol, split_window},
        };
        struct AcceptKick;
        impl Transport for AcceptKick {
            type Error = ();
            async fn commit(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
            fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, ()> {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
        }
        #[repr(align(64))]
        struct Window([u8; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        for refused in [false, true] {
            let mut window = Window([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (rings, mut serve) = split_window::<crate::HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    rings,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut call = FrontendCall::result(
                request(31),
                FrontendResult {
                    intent_id: intent(41),
                    kind: FrontendResultKind::Failed,
                    text: "Failed: δ",
                },
                20_000,
            )
            .unwrap();
            let original = call.wire;
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut send = core::pin::pin!(call.send(&mut link));
                assert!(send.as_mut().poll(&mut context).is_pending());
            }
            let first = serve.req.pop().unwrap();
            assert!(
                !link.route_request_pending(),
                "a frontend send cannot resume an old uncertain route"
            );
            let decoded = crate::HostLinkWire::decode_req(&first).unwrap();
            assert!(
                matches!(decoded, crate::ClientRequest::Send { request: id, .. } if id == request(31))
            );
            for _ in 0..2 {
                let mut resume = core::pin::pin!(call.advance_send_with_budget(&mut link, None));
                assert!(resume.as_mut().poll(&mut context).is_pending());
                assert!(
                    serve.req.pop().is_none(),
                    "observation never replays the response"
                );
            }
            let response = if refused {
                crate::ClientResponse::Refused {
                    request: request(31),
                    code: kore_hostlink::RefusalCode::Backpressure,
                }
            } else {
                crate::ClientResponse::Accepted {
                    request: request(31),
                }
            };
            serve
                .rep
                .push(crate::HostLinkWire::encode_rep(&response, 0))
                .unwrap();
            // A sibling that has not published stays fresh; its attempt
            // cannot steal the service acknowledgement waiting for this call.
            let mut sibling = FrontendCall::result(
                request(32),
                FrontendResult {
                    intent_id: intent(42),
                    kind: FrontendResultKind::Failed,
                    text: "Sibling",
                },
                20_000,
            )
            .unwrap();
            {
                let mut blocked =
                    core::pin::pin!(sibling.advance_send_with_budget(&mut link, None));
                assert!(matches!(
                    blocked.as_mut().poll(&mut context),
                    Poll::Ready(Err(FrontendError::Transport(ClientError::Call(
                        kore_rt::svc::CallError::Outstanding
                    ))))
                ));
            }
            assert!(sibling.state == State::Fresh);
            // Even reusing the correct request ID does not match a different
            // frontend payload. The exact envelope owns the reply.
            let forged = FrontendCall::result(
                request(31),
                FrontendResult {
                    intent_id: intent(41),
                    kind: FrontendResultKind::Completed,
                    text: "Changed",
                },
                20_000,
            )
            .unwrap();
            {
                let mut wrong = core::pin::pin!(link.observe_frontend_send(
                    request(31),
                    &forged.wire[..forged.len],
                    true,
                    None
                ));
                assert!(matches!(
                    wrong.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(kore_rt::svc::CallError::Outstanding)))
                ));
            }
            assert!(serve.req.pop().is_none());
            {
                let mut wrong_poll = core::pin::pin!(link.observe_poll(true, 1));
                assert!(matches!(
                    wrong_poll.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(kore_rt::svc::CallError::Outstanding)))
                ));
            }
            {
                let mut resume = core::pin::pin!(call.advance_send_with_budget(&mut link, None));
                match resume.as_mut().poll(&mut context) {
                    Poll::Ready(Ok(())) => assert!(!refused),
                    Poll::Ready(Err(FrontendError::Transport(ClientError::Refused(
                        kore_hostlink::RefusalCode::Backpressure,
                    )))) => assert!(refused),
                    other => panic!("unexpected send continuation: {other:?}"),
                }
            }
            assert_eq!(call.wire, original);
            if refused {
                assert!(call.state == State::Fresh);
                {
                    let mut retry = core::pin::pin!(call.advance_send_with_budget(&mut link, None));
                    assert!(retry.as_mut().poll(&mut context).is_pending());
                }
                assert_eq!(
                    serve.req.pop().unwrap(),
                    first,
                    "explicit refusal retries the exact bytes"
                );
                serve
                    .rep
                    .push(crate::HostLinkWire::encode_rep(
                        &crate::ClientResponse::Accepted {
                            request: request(31),
                        },
                        0,
                    ))
                    .unwrap();
                let mut accepted = core::pin::pin!(call.advance_send_with_budget(&mut link, None));
                assert!(matches!(
                    accepted.as_mut().poll(&mut context),
                    Poll::Ready(Ok(()))
                ));
            }
            assert!(
                call.state == State::Awaiting,
                "service acceptance is not frontend publication receipt"
            );
            assert!(!call.pending_send());
            assert!(matches!(
                call.accept(event(request(32), &[], 1, 0, true)),
                Ok(FrontendProgress::Other(_))
            ));
            assert!(call.state == State::Awaiting);
            assert_eq!(
                call.accept(event(request(31), &[], 1, 0, true)).unwrap(),
                FrontendProgress::Receipt
            );
            assert!(call.state == State::Finished);
            assert!(matches!(
                call.accept(event(request(31), &[], 1, 0, true)),
                Err(FrontendError::NotPending)
            ));
            assert!(serve.req.pop().is_none());
        }
    }
}
