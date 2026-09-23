//! Affine, caller-owned construction of an exec client face.
use super::*;
use crate::wire::ExecProtocol;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Mint,
    Bank,
    Context,
    Kick,
    Ready,
    Retire,
    Confirm,
    Local,
    Closed,
    Failed,
}

/// One logical mint, including every partial delivery and its cleanup.
/// Keep this value in the requesting owner's state across cancelled awaits.
/// Equal filesystem rights do not identify the same mint: each owner tracks
/// whether it actually published the pending operation before resuming it.
pub struct ExecClientMint {
    rights: kore_svc_dagfs::Rights,
    phase: Phase,
    in_flight: bool,
    aborting: bool,
    face: Option<u8>,
    retirement: Option<u64>,
    ring: Option<StashedLease>,
    kick: Option<ChanSendToken>,
    grants: [Option<ChanRecvToken>; crate::wire::EXEC_REPLY_GRANTS],
    context: [Option<ChanSendToken>; 3 + MAX_INVOCATION_AUXILIARIES],
}

impl ExecClientMint {
    pub fn new(rights: kore_svc_dagfs::Rights) -> Self {
        Self {
            rights,
            phase: Phase::Mint,
            in_flight: false,
            aborting: false,
            face: None,
            retirement: None,
            ring: None,
            kick: None,
            grants: [const { None }; crate::wire::EXEC_REPLY_GRANTS],
            context: [const { None }; 3 + MAX_INVOCATION_AUXILIARIES],
        }
    }

    pub fn rights(&self) -> kore_svc_dagfs::Rights {
        self.rights
    }

    pub fn aborting(&self) -> bool {
        self.aborting
    }

    pub fn closed(&self) -> bool {
        self.phase == Phase::Closed
    }

    /// Stops construction. Any already-published grant request must first be
    /// consumed on its original pair; no later bank/context is requested.
    pub fn abort(&mut self) {
        self.aborting = true;
        self.select_cleanup();
    }

    fn select_cleanup(&mut self) {
        if !self.aborting || self.in_flight {
            return;
        }
        if matches!(
            self.phase,
            Phase::Mint | Phase::Bank | Phase::Context | Phase::Kick | Phase::Ready
        ) {
            self.phase = if self.face.is_some() {
                Phase::Retire
            } else {
                Phase::Local
            };
        }
    }

    /// Advance at most one remote operation. A successful observation need
    /// not finish construction/cleanup. Only `take_ready` transfers authority;
    /// For an aborted construction, `closed` proves cleanup is complete.
    /// After `take_ready`, all ownership instead belongs to its returned face.
    pub async fn observe<T: Transport>(
        &mut self,
        exec: &mut RingExec<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<(), InstallError> {
        self.select_cleanup();
        if self.phase == Phase::Failed {
            return Err(InstallError::Call(CallError::Poisoned));
        }
        if self.phase == Phase::Closed || self.phase == Phase::Ready {
            return Ok(());
        }
        if self.phase == Phase::Local {
            return if self.close_local() {
                self.phase = Phase::Closed;
                Ok(())
            } else {
                Err(InstallError::Call(CallError::Backpressure))
            };
        }
        if self.phase == Phase::Kick {
            self.kick = verbs::derive_owned_chan_send_with_rights(
                exec.pair.request_kick(),
                kore_rt::cap::CHANNEL_ENDPOINT_RIGHTS,
            );
            if self.kick.is_some() {
                self.phase = Phase::Ready;
                return Ok(());
            }
            self.aborting = true;
            self.select_cleanup();
            return Err(InstallError::Call(CallError::Refused(Status::Denied)));
        }
        exec.ensure_no_discarded_wake()?;
        let request = match self.phase {
            Phase::Mint => TaskRequest::MintClient {
                fs_rights: self.rights,
            },
            Phase::Bank => TaskRequest::ClientBank {
                face: self.face.unwrap(),
            },
            Phase::Context => TaskRequest::ClientContext {
                face: self.face.unwrap(),
            },
            Phase::Retire => TaskRequest::RetireClient {
                face: self.face.unwrap(),
            },
            Phase::Confirm => TaskRequest::ConfirmClientRetired {
                face: self.face.unwrap(),
                retirement: self.retirement.unwrap(),
            },
            _ => unreachable!(),
        };
        let resume = self.in_flight && !exec.pair.is_idle();
        self.in_flight = true;
        let reply = match (resume, budget_ns) {
            (false, None) => exec.pair.call(&request).await,
            (true, None) => exec.pair.resume_exact(&request).await,
            (false, Some(budget)) => exec.pair.call_bounded(&request, budget).await,
            (true, Some(budget)) => exec.pair.resume_exact_bounded(&request, budget).await,
        };
        let reply = match reply {
            Ok(reply) => reply,
            Err(error) => {
                match error {
                    CallError::Outstanding if !resume => self.in_flight = false,
                    CallError::Outstanding => {}
                    CallError::Timeout
                    | CallError::RingFull
                    | CallError::Backpressure
                    | CallError::Transport
                    | CallError::Refused(Status::Conflict | Status::WouldBlock) => {
                        if exec.pair.is_idle() {
                            self.in_flight = false;
                        }
                    }
                    // A terminal preflight refusal with an idle pair proves
                    // that no request was published. Clean up any prior prefix.
                    CallError::Wiring | CallError::Refused(_) if exec.pair.is_idle() => {
                        self.in_flight = false;
                        self.aborting = true;
                    }
                    _ => self.phase = Phase::Failed,
                }
                self.select_cleanup();
                return Err(InstallError::Call(error));
            }
        };
        self.in_flight = false;
        let result = self.accept_reply(&mut exec.pair, reply);
        self.select_cleanup();
        result
    }

    fn accept_reply<T: Transport>(
        &mut self,
        pair: &mut SvcClient<'_, '_, T, ExecProtocol>,
        mut reply: CallReply<TaskResponse>,
    ) -> Result<(), InstallError> {
        if reply.delivered.iter().any(Option::is_some) {
            return self.malformed(pair, reply);
        }
        if let TaskResponse::Refused(refusal) = reply.response {
            if reply.grants != 0 || reply.retained.iter().any(Option::is_some) {
                return self.malformed(pair, reply);
            }
            match self.phase {
                Phase::Mint => {
                    self.aborting = true;
                    self.phase = Phase::Local;
                }
                Phase::Bank | Phase::Context => {
                    self.aborting = true;
                    self.phase = Phase::Retire;
                }
                Phase::Retire if refusal == TaskRefusal::Contended => {}
                _ => self.phase = Phase::Failed,
            }
            return Err(InstallError::Refused(refusal));
        }
        match (self.phase, &reply.response) {
            (Phase::Mint, TaskResponse::ClientMinted { face })
                if reply.grants == 1 && reply.retained[1..].iter().all(Option::is_none) =>
            {
                // Preserve the service-authored identity before interpreting
                // its affine delivery. A malformed delivery retains both.
                self.face = Some(*face);
                let Some(returned) = reply.retained[0].take() else {
                    return self.malformed(pair, reply);
                };
                match returned.try_into_stashed_lease(
                    Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
                ) {
                    Ok(ring) => self.ring = Some(ring),
                    Err(returned) => {
                        reply.retained[0] = Some(returned);
                        return self.malformed(pair, reply);
                    }
                }
                self.phase = Phase::Bank;
            }
            (Phase::Bank, TaskResponse::ClientBanked { face })
                if Some(*face) == self.face && usize::from(reply.grants) == self.grants.len() =>
            {
                for index in 0..self.grants.len() {
                    let Some(returned) = reply.retained[index].take() else {
                        return self.malformed(pair, reply);
                    };
                    match returned.try_into_chan_recv(Rights::ACCESS | Rights::REVOKE) {
                        Ok(endpoint) => self.grants[index] = Some(endpoint),
                        Err(returned) => {
                            reply.retained[index] = Some(returned);
                            return self.malformed(pair, reply);
                        }
                    }
                }
                self.phase = Phase::Context;
            }
            (Phase::Context, TaskResponse::ClientContext { face })
                if Some(*face) == self.face
                    && usize::from(reply.grants) == self.context.len()
                    && reply.retained[self.context.len()..]
                        .iter()
                        .all(Option::is_none) =>
            {
                for index in 0..self.context.len() {
                    let Some(returned) = reply.retained[index].take() else {
                        return self.malformed(pair, reply);
                    };
                    let required = if index == 2 {
                        kore_svc_dagfs::FsViewCapability::DELEGABLE_RIGHTS
                    } else {
                        Rights::ACCESS | Rights::DERIVE | Rights::REVOKE
                    };
                    match returned.try_into_chan_send(required) {
                        Ok(endpoint) => self.context[index] = Some(endpoint),
                        Err(returned) => {
                            reply.retained[index] = Some(returned);
                            return self.malformed(pair, reply);
                        }
                    }
                }
                self.phase = Phase::Kick;
            }
            (Phase::Retire | Phase::Confirm, TaskResponse::ClientRetiring { face, retirement })
                if Some(*face) == self.face
                    && *retirement != 0
                    && reply.grants == 0
                    && reply.retained.iter().all(Option::is_none)
                    && (self.phase == Phase::Retire || self.retirement == Some(*retirement)) =>
            {
                self.retirement = Some(*retirement);
                self.phase = Phase::Confirm;
            }
            (Phase::Confirm, TaskResponse::ClientRetirementConfirmed { face, retirement })
                if Some(*face) == self.face
                    && self.retirement == Some(*retirement)
                    && *retirement != 0
                    && reply.grants == 0
                    && reply.retained.iter().all(Option::is_none) =>
            {
                self.phase = Phase::Local;
            }
            _ => return self.malformed(pair, reply),
        }
        Ok(())
    }

    fn malformed<T: Transport>(
        &mut self,
        pair: &mut SvcClient<'_, '_, T, ExecProtocol>,
        reply: CallReply<TaskResponse>,
    ) -> Result<(), InstallError> {
        // Already retyped owners remain in this construction. The held pair
        // retains all untyped returns and its uncertain posts for reset.
        pair.poison_retained(reply.retained);
        self.phase = Phase::Failed;
        Err(InstallError::Call(CallError::Wire))
    }

    fn close_local(&mut self) -> bool {
        debug_assert_eq!(self.phase, Phase::Local);
        if let Some(owner) = self.kick.take() {
            self.kick = owner.try_retire().err();
        }
        for owner in &mut self.context {
            if let Some(endpoint) = owner.take() {
                *owner = endpoint.try_retire().err();
            }
        }
        for owner in &mut self.grants {
            if let Some(endpoint) = owner.take() {
                *owner = endpoint.try_retire().err();
            }
        }
        if let Some(ring) = self.ring.as_mut() {
            if ring.try_retire() {
                self.ring = None;
            }
        }
        self.kick.is_none()
            && self.ring.is_none()
            && self.context.iter().all(Option::is_none)
            && self.grants.iter().all(Option::is_none)
    }

    pub fn take_ready(&mut self) -> Option<MintedExecClientFace> {
        if self.phase != Phase::Ready || self.aborting {
            return None;
        }
        let face = self.face?;
        // Validate every field before taking any owner.
        if self.ring.is_none()
            || self.kick.is_none()
            || self.grants.iter().any(Option::is_none)
            || self.context.iter().any(Option::is_none)
        {
            return None;
        }
        let grants = core::array::from_fn(|i| self.grants[i].take().unwrap());
        let [proof_tx, process_tx, fs_view_tx, auxiliary @ ..] = core::mem::replace(
            &mut self.context,
            [const { None }; 3 + MAX_INVOCATION_AUXILIARIES],
        );
        let result = MintedExecClientFace {
            face,
            ring: self.ring.take().unwrap(),
            kick: self.kick.take().unwrap(),
            grants,
            proof_tx: proof_tx.unwrap(),
            process_tx: process_tx.unwrap(),
            fs_view_tx: fs_view_tx.unwrap(),
            auxiliary_tx: auxiliary.map(Option::unwrap),
        };
        self.phase = Phase::Closed;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    use kore_abi::session::Committed;
    use kore_rt::{
        SessionBuilder,
        svc::{REP_CAP, REQ_CAP, SvcProtocol, split_window, window_bytes},
    };

    struct ReadyTransport;
    impl Transport for ReadyTransport {
        type Error = core::convert::Infallible;
        fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }
        async fn commit(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }
    }
    const BYTES: usize = window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP);
    #[repr(align(64))]
    struct Window([u8; BYTES]);

    fn reply(response: TaskResponse) -> CallReply<TaskResponse> {
        CallReply {
            response,
            grants: 0,
            delivered: [None; kore_rt::service::MAX_GRANTS],
            retained: [const { None }; kore_rt::service::MAX_GRANTS],
        }
    }

    #[test]
    fn abort_waits_for_its_published_stage_and_preserves_the_retirement_fence() {
        let transport = ReadyTransport;
        let mut window = Window([0; BYTES]);
        let (rings, mut service) =
            split_window::<ExecProtocol>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut exec = RingExec::new(
            SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(100),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling(),
        );
        let mut cx = Context::from_waker(Waker::noop());
        // Control-state fixture for a bank request already in flight. Native
        // grants are absent; this does not claim a successful native mint.
        let mut mint = ExecClientMint::new(kore_svc_dagfs::Rights::NONE);
        mint.face = Some(4);
        mint.phase = Phase::Bank;
        mint.in_flight = true;
        mint.abort();
        assert_eq!(mint.phase, Phase::Bank);
        mint.in_flight = false;
        assert_eq!(
            mint.accept_reply(
                &mut exec.pair,
                reply(TaskResponse::Refused(TaskRefusal::Denied))
            ),
            Err(InstallError::Refused(TaskRefusal::Denied))
        );
        assert_eq!(mint.phase, Phase::Retire);
        let mut sibling = ExecClientMint::new(kore_svc_dagfs::Rights::NONE);
        for round in 0..3 {
            for attempt in 0..3 {
                {
                    let mut observation = core::pin::pin!(mint.observe(&mut exec, None));
                    assert!(observation.as_mut().poll(&mut cx).is_pending());
                }
                if attempt == 0 {
                    let request = ExecProtocol::decode_req(&service.req.pop().unwrap()).unwrap();
                    if round == 0 {
                        assert!(matches!(request, TaskRequest::RetireClient { face: 4 }));
                    } else {
                        assert!(matches!(
                            request,
                            TaskRequest::ConfirmClientRetired {
                                face: 4,
                                retirement: 19
                            }
                        ));
                    }
                } else {
                    assert!(service.req.pop().is_none());
                }
            }
            {
                let mut competing = core::pin::pin!(sibling.observe(&mut exec, None));
                assert!(matches!(
                    competing.as_mut().poll(&mut cx),
                    Poll::Ready(Err(InstallError::Call(CallError::Outstanding)))
                ));
            }
            assert!(
                !sibling.in_flight,
                "a fresh caller cannot claim another owner's operation"
            );
            let response = if round == 2 {
                TaskResponse::ClientRetirementConfirmed {
                    face: 4,
                    retirement: 19,
                }
            } else {
                TaskResponse::ClientRetiring {
                    face: 4,
                    retirement: 19,
                }
            };
            service
                .rep
                .push(ExecProtocol::encode_rep(&response, 0))
                .unwrap();
            {
                let mut observation = core::pin::pin!(mint.observe(&mut exec, None));
                assert!(matches!(
                    observation.as_mut().poll(&mut cx),
                    Poll::Ready(Ok(()))
                ));
            }
            assert!(!mint.closed());
        }
        assert_eq!(mint.phase, Phase::Local);
        {
            let mut observation = core::pin::pin!(mint.observe(&mut exec, None));
            assert!(matches!(
                observation.as_mut().poll(&mut cx),
                Poll::Ready(Ok(()))
            ));
        }
        assert!(mint.closed());
        assert!(service.req.pop().is_none());
    }

    #[test]
    fn abort_before_publication_does_not_create_or_retire_a_face() {
        let transport = ReadyTransport;
        let mut window = Window([0; BYTES]);
        let (rings, mut service) =
            split_window::<ExecProtocol>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut exec = RingExec::new(SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        ));
        let mut mint = ExecClientMint::new(kore_svc_dagfs::Rights::NONE);
        mint.abort();
        let mut cx = Context::from_waker(Waker::noop());
        {
            let mut observation = core::pin::pin!(mint.observe(&mut exec, None));
            assert!(matches!(
                observation.as_mut().poll(&mut cx),
                Poll::Ready(Ok(()))
            ));
        }
        assert!(mint.closed());
        assert!(mint.take_ready().is_none());
        assert!(service.req.pop().is_none());
    }

    #[test]
    fn malformed_mint_keeps_the_known_identity_and_never_publishes_cleanup_on_a_poisoned_pair() {
        let transport = ReadyTransport;
        let mut window = Window([0; BYTES]);
        let (rings, mut service) =
            split_window::<ExecProtocol>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut exec = RingExec::new(SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        ));
        let mut mint = ExecClientMint::new(kore_svc_dagfs::Rights::NONE);
        let mut invalid = reply(TaskResponse::ClientMinted { face: 7 });
        invalid.grants = 1; // Missing native delivery, not a fabricated owner.
        assert_eq!(
            mint.accept_reply(&mut exec.pair, invalid),
            Err(InstallError::Call(CallError::Wire))
        );
        assert_eq!(mint.face, Some(7));
        mint.abort();
        assert_eq!(mint.phase, Phase::Failed);
        let mut cx = Context::from_waker(Waker::noop());
        {
            let mut observation = core::pin::pin!(mint.observe(&mut exec, None));
            assert!(matches!(
                observation.as_mut().poll(&mut cx),
                Poll::Ready(Err(InstallError::Call(CallError::Poisoned)))
            ));
        }
        assert!(!mint.closed());
        assert!(service.req.pop().is_none());
    }
}
