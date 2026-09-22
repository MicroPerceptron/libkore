//! Safe born-async client over one structurally assigned HostLink pair.

use kore_abi::task::layout::Window;
use kore_hostlink::{
    AUTHORITY_REPORT, AuthorityReportView, BulkId, BulkRefV1, INFER_REQUEST,
    INFERENCE_REQUEST_CONTENT, RefusalCode, RequestId, TaskGeneration, WORKSPACE_REQUEST,
};
use kore_rt::cap::{
    ChanSendToken, HeldChanSendToken, LeaseToken, MappedLeaseView, StashedLease, verbs,
};
use kore_rt::svc::{CallError, ClientEndpoints, ClientRings, SvcClient};
use kore_rt::transport::Transport;
use zerocopy::byteorder::little_endian::{U16, U32};

use crate::{
    CLIENT_CHUNK_MAX, ClientEvent, ClientRequest, ClientResponse, ClientStatus,
    FOUNDED_BULK_WIRE_CHUNK, HostLinkWire, Payload,
};

const BULK_CALL_RETRY_LIMIT: usize = 4096;
const ROUTE_CALL_RETRY_LIMIT: usize = 4096;

#[derive(Debug)]
pub enum ClientError {
    Call(CallError),
    Refused(RefusalCode),
    UnexpectedReply,
    InvalidPayload,
    InvalidBulk,
}

/// State for one streamed guest-to-host bulk object. The client advances it
/// only after the service accepts the exact fragment, so backpressure is safe
/// to retry without skipping bytes.
pub struct BulkUpload {
    request: RequestId,
    reference: BulkRefV1,
    offset: u32,
    wire_bytes: u16,
    wire_chunks: u16,
    fragments: u32,
    hasher: blake3::Hasher,
}

impl BulkUpload {
    pub const fn request(&self) -> RequestId {
        self.request
    }

    pub const fn bulk_id(&self) -> BulkId {
        self.reference.bulk_id
    }

    pub const fn offset(&self) -> u32 {
        self.offset
    }

    pub const fn chunks(&self) -> u16 {
        self.wire_chunks
    }

    pub const fn fragments(&self) -> u32 {
        self.fragments
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InferenceStage {
    Inline,
    Begin,
    Chunk,
    Finish,
    Complete,
    Aborting,
    AwaitAbort,
    Refused,
    Failed,
}

/// One immutable inference request retained by its caller across borrowed
/// observation futures. Every fragment advances only on its exact service ack.
/// This contains bounded bytes and correlations, never native authority.
pub struct InferenceSend {
    request: RequestId,
    bytes: heapless::Vec<u8, { kore_hostlink::INLINE_MAX }>,
    reference: BulkRefV1,
    offset: usize,
    stage: InferenceStage,
    in_flight: bool,
    refusal: Option<RefusalCode>,
}

fn inference_bulk_id(request: RequestId) -> BulkId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"kore.inference-upload.v1\0");
    hasher.update(&request.0);
    let digest = hasher.finalize();
    let mut id = [0; 16];
    id.copy_from_slice(&digest.as_bytes()[..16]);
    BulkId(id)
}

impl InferenceSend {
    pub fn new(request: RequestId, bytes: &[u8]) -> Result<Self, ClientError> {
        if kore_hostlink::decode_inference_request(bytes).is_none() {
            return Err(ClientError::InvalidPayload);
        }
        Ok(Self {
            request,
            bytes: heapless::Vec::from_slice(bytes).map_err(|_| ClientError::InvalidPayload)?,
            reference: BulkRefV1 {
                bulk_id: inference_bulk_id(request),
                total_len: U32::new(bytes.len() as u32),
                chunk_count: U16::new(bytes.len().div_ceil(FOUNDED_BULK_WIRE_CHUNK) as u16),
                content_kind: U16::new(INFERENCE_REQUEST_CONTENT),
                digest_blake3: *blake3::hash(bytes).as_bytes(),
            },
            offset: 0,
            stage: if bytes.len() <= CLIENT_CHUNK_MAX {
                InferenceStage::Inline
            } else {
                InferenceStage::Begin
            },
            in_flight: false,
            refusal: None,
        })
    }

    pub fn request(&self) -> RequestId {
        self.request
    }
    pub fn in_flight(&self) -> bool {
        self.in_flight
    }
    pub fn failed(&self) -> bool {
        self.stage == InferenceStage::Failed
    }
    pub fn pending_send(&self) -> bool {
        matches!(
            self.stage,
            InferenceStage::Inline
                | InferenceStage::Begin
                | InferenceStage::Chunk
                | InferenceStage::Finish
                | InferenceStage::Aborting
        )
    }
    /// A refusal is terminal only when no upload was opened or its exact
    /// remote abort receipt has proved retirement.
    pub fn retired_refusal(&self) -> Option<RefusalCode> {
        (self.stage == InferenceStage::Refused)
            .then_some(self.refusal)
            .flatten()
    }
    /// The upload has either been submitted in full or refused with proven
    /// retirement. This does not attest model completion or a frontend receipt.
    pub fn upload_settled(&self) -> bool {
        matches!(
            self.stage,
            InferenceStage::Complete | InferenceStage::Refused
        )
    }

    /// Drive one upload verb or mailbox observation on a private worker pair.
    /// The caller keeps this owner while handling parent controls between
    /// observations. A borrowed future may be dropped without replaying a send
    /// or consuming a second mailbox event. Partial refusal is surfaced only
    /// after its exact remote abort receipt; foreign events stay caller-owned.
    pub async fn observe_event<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<Option<ClientEvent>, ClientError> {
        self.event_observation(link, Some(budget_ns)).await
    }

    async fn event_observation<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<Option<ClientEvent>, ClientError> {
        if self.failed() {
            return Err(ClientError::UnexpectedReply);
        }
        if let Some(code) = self.retired_refusal() {
            return Err(ClientError::Refused(code));
        }
        // A previously published Poll owns its reply even if a new inference
        // owner was prepared before the caller resumed that observation.
        if self.pending_send() && !link.poll_request_pending() {
            let result = self.observe(link, budget_ns).await;
            if let Some(code) = self.retired_refusal() {
                return Err(ClientError::Refused(code));
            }
            return match result {
                Ok(_) => Ok(None),
                // Keep both the rejected prefix and the abort obligation.
                Err(ClientError::Refused(_)) if !self.failed() => Ok(None),
                Err(error) if retry_observation(&error) && !self.failed() => Ok(None),
                Err(error) => Err(error),
            };
        }
        let resume = link.poll_request_pending();
        let event = match link.poll_observation(resume, budget_ns).await {
            Ok(event) => event,
            Err(error) if retry_observation(&error) => return Ok(None),
            Err(error) => {
                self.stage = InferenceStage::Failed;
                return Err(error);
            }
        };
        if event.as_ref().is_some_and(|event| self.accept_abort(event)) {
            return match self.retired_refusal() {
                Some(code) => Err(ClientError::Refused(code)),
                None => Err(ClientError::UnexpectedReply),
            };
        }
        if self.pending_send()
            && event.as_ref().is_some_and(|event| match event {
                ClientEvent::Control { request, .. }
                | ClientEvent::Cancel { request }
                | ClientEvent::BulkBegin { request, .. }
                | ClientEvent::BulkChunk { request, .. }
                | ClientEvent::BulkReady { request, .. }
                | ClientEvent::BulkAbort { request, .. } => *request == self.request,
                ClientEvent::Status(status) => status.request == self.request,
            })
        {
            // A stale/forged reply cannot skip publication of this owner.
            self.stage = InferenceStage::Failed;
            return Err(ClientError::UnexpectedReply);
        }
        Ok(event)
    }

    pub fn accept_abort(&mut self, event: &ClientEvent) -> bool {
        if self.stage != InferenceStage::AwaitAbort {
            return false;
        }
        match event {
            ClientEvent::Control {
                kind,
                request,
                payload,
            } if *request == self.request => {
                if *kind == kore_hostlink::BULK_ABORTED
                    && payload.as_slice() == self.reference.bulk_id.0
                {
                    self.stage = InferenceStage::Refused;
                } else {
                    // A different terminal outcome cannot prove this upload's
                    // retirement; preserve ownership for route recovery.
                    self.stage = InferenceStage::Failed;
                }
                true
            }
            ClientEvent::Cancel { request } if *request == self.request => {
                self.stage = InferenceStage::Failed;
                true
            }
            ClientEvent::Status(status) if status.request == self.request => {
                self.stage = InferenceStage::Failed;
                true
            }
            _ => false,
        }
    }
    /// Only this exact wire request can take precedence over another caller.
    /// A suspended pre-publication wait may retain an attempt while the pair
    /// is still idle, or while another owner subsequently occupies it.
    pub fn resume_pending<T: Transport>(&self, link: &RingHostLink<'_, '_, T>) -> bool {
        self.in_flight
            && self.operation().is_ok_and(|operation| {
                link.pair
                    .outstanding_request()
                    .and_then(<HostLinkWire as kore_rt::svc::SvcProtocol>::decode_req)
                    .as_ref()
                    == Some(&operation)
            })
    }

    fn fragment_end(&self) -> usize {
        // A founded fragment cannot cross a credited wire-frame boundary.
        let remaining = FOUNDED_BULK_WIRE_CHUNK - self.offset % FOUNDED_BULK_WIRE_CHUNK;
        self.bytes
            .len()
            .min(self.offset + CLIENT_CHUNK_MAX.min(remaining))
    }

    fn operation(&self) -> Result<ClientRequest, ClientError> {
        Ok(match self.stage {
            InferenceStage::Inline => ClientRequest::Send {
                kind: INFER_REQUEST,
                request: self.request,
                payload: Payload::copy_from(&self.bytes).ok_or(ClientError::InvalidPayload)?,
            },
            InferenceStage::Begin => ClientRequest::BulkBegin {
                request: self.request,
                reference: self.reference,
            },
            InferenceStage::Chunk => ClientRequest::BulkChunk {
                request: self.request,
                bulk_id: self.reference.bulk_id,
                bytes: Payload::copy_from(&self.bytes[self.offset..self.fragment_end()])
                    .ok_or(ClientError::InvalidPayload)?,
            },
            InferenceStage::Finish => ClientRequest::BulkFinish {
                request: self.request,
                bulk_id: self.reference.bulk_id,
                total_len: self.reference.total_len.get(),
                digest_blake3: self.reference.digest_blake3,
            },
            InferenceStage::Aborting => ClientRequest::BulkAbort {
                request: self.request,
                bulk_id: self.reference.bulk_id,
                reason: self.refusal.ok_or(ClientError::InvalidBulk)?,
            },
            _ => return Err(ClientError::UnexpectedReply),
        })
    }

    /// Observe at most one service verb. Ok(true) means the complete request
    /// was accepted, not that inference or its eventual frontend result finished.
    pub async fn advance<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<bool, ClientError> {
        self.observe(link, Some(budget_ns)).await
    }

    async fn observe<T: Transport>(
        &mut self,
        link: &mut RingHostLink<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<bool, ClientError> {
        if self.stage == InferenceStage::Complete {
            return Ok(true);
        }
        if matches!(
            self.stage,
            InferenceStage::AwaitAbort | InferenceStage::Refused
        ) {
            return Ok(false);
        }
        let operation = self.operation()?;
        let resume = self.resume_pending(link);
        if !resume && !link.is_idle() {
            return Err(ClientError::Call(CallError::Outstanding));
        }
        self.in_flight = true;
        match link.observe_request(operation, resume, budget_ns).await {
            Ok(()) => {
                self.in_flight = false;
                self.stage = match self.stage {
                    InferenceStage::Inline | InferenceStage::Finish => InferenceStage::Complete,
                    InferenceStage::Begin => InferenceStage::Chunk,
                    InferenceStage::Aborting => InferenceStage::AwaitAbort,
                    InferenceStage::Chunk => {
                        self.offset = self.fragment_end();
                        if self.offset == self.bytes.len() {
                            InferenceStage::Finish
                        } else {
                            InferenceStage::Chunk
                        }
                    }
                    _ => return Err(ClientError::UnexpectedReply),
                };
                Ok(self.stage == InferenceStage::Complete)
            }
            Err(error) => {
                if let ClientError::Refused(code) = error {
                    if code != RefusalCode::Backpressure {
                        self.in_flight = false;
                        self.refusal.get_or_insert(code);
                        self.stage = match self.stage {
                            InferenceStage::Inline | InferenceStage::Begin => {
                                InferenceStage::Refused
                            }
                            InferenceStage::Chunk | InferenceStage::Finish => {
                                InferenceStage::Aborting
                            }
                            _ => InferenceStage::Failed,
                        };
                        return Err(ClientError::Refused(code));
                    }
                }
                if matches!(error, ClientError::Refused(RefusalCode::Backpressure))
                    || (matches!(error, ClientError::Call(_)) && link.is_idle())
                    || (!resume
                        && matches!(
                            error,
                            ClientError::Call(CallError::RingFull | CallError::Outstanding)
                        ))
                {
                    self.in_flight = false;
                }
                if !matches!(error, ClientError::Refused(RefusalCode::Backpressure))
                    && matches!(
                        error,
                        ClientError::Refused(_)
                            | ClientError::UnexpectedReply
                            | ClientError::InvalidPayload
                            | ClientError::InvalidBulk
                            | ClientError::Call(
                                CallError::Poisoned | CallError::Wire | CallError::Wiring
                            )
                    )
                {
                    // Preserve the payload and any accepted bulk prefix. A
                    // terminal refusal never authorizes replay or drops custody.
                    self.stage = InferenceStage::Failed;
                }
                Err(error)
            }
        }
    }
}

pub fn retry_observation(error: &ClientError) -> bool {
    matches!(
        error,
        ClientError::Refused(RefusalCode::Backpressure)
            | ClientError::Call(
                CallError::Timeout
                    | CallError::RingFull
                    | CallError::Backpressure
                    | CallError::Transport
                    | CallError::Refused(
                        kore_abi::session::Status::Conflict | kore_abi::session::Status::WouldBlock
                    )
            )
    )
}

/// A client owns exactly one founded pair. There is deliberately no endpoint
/// argument on this type or any method: broker/dynamic-agent/workspace is the
/// pair the founder handed to the caller.
pub struct RingHostLink<'w, 't, T: Transport> {
    pair: SvcClient<'w, 't, T, HostLinkWire>,
    _owned: Option<OwnedClientFace>,
}

// Drop the client before its exact-unshare mapping owner, and the mapping
// before its lessor. No method exposes an extractable static SvcClient.
/// Sole client custody, either received at founding or over a capability
/// channel. Both forms retain their real affine owners; neither mints a face.
pub enum ClientFace {
    Founded {
        ring: LeaseToken,
        kick: HeldChanSendToken,
    },
    Delegated {
        ring: StashedLease,
        kick: ChanSendToken,
    },
}

impl ClientFace {
    pub fn ring_slot(&self) -> Option<kore_abi::session::SlotRef> {
        match self {
            Self::Founded { ring, .. } => Some(ring.slot()),
            Self::Delegated { ring, .. } => ring.slot(),
        }
    }
    pub fn kick_slot(&self) -> kore_abi::session::SlotRef {
        match self {
            Self::Founded { kick, .. } => kick.slot(),
            Self::Delegated { kick, .. } => kick.slot(),
        }
    }
}

struct OwnedClientFace {
    _mapping: MappedLeaseView,
    _face: ClientFace,
}

impl<'w, 't, T: Transport> RingHostLink<'w, 't, T> {
    pub fn new(
        transport: &'t T,
        rings: ClientRings<'w, HostLinkWire>,
        endpoints: ClientEndpoints,
    ) -> Self {
        Self {
            pair: SvcClient::new(transport, rings, endpoints),
            _owned: None,
        }
    }

    pub fn from_pair(pair: SvcClient<'w, 't, T, HostLinkWire>) -> Self {
        Self { pair, _owned: None }
    }

    /// Attach a founded client sublease, retaining both its mapping and
    /// endpoint owners inside the client for its whole lifetime.
    ///
    /// # Safety
    /// The caller owns the sole client side of this SPSC pair. No previous
    /// producer or consumer may remain attached through another alias, and
    /// the supplied VA window must be exclusively reserved for this mapping.
    pub unsafe fn from_owned(
        transport: &'t T,
        ring: LeaseToken,
        kick: HeldChanSendToken,
        window: Window,
    ) -> Option<RingHostLink<'static, 't, T>> {
        // SAFETY: the caller transfers the same exclusive pair/window contract.
        unsafe { Self::from_face(transport, ClientFace::Founded { ring, kick }, window) }
    }

    /// Attach one explicitly delegated or founded client face.
    ///
    /// # Safety
    /// The caller must own the sole SPSC producer/consumer and exclusively
    /// reserve the supplied VA window, as for `from_owned`.
    pub unsafe fn from_face(
        transport: &'t T,
        face: ClientFace,
        window: Window,
    ) -> Option<RingHostLink<'static, 't, T>> {
        if window.base() & 0xfff != 0
            || window.len() & 0xfff != 0
            || window.len() < crate::HOSTLINK_PAIR_WINDOW_BYTES as u64
        {
            return None;
        }
        let mut mapping =
            verbs::map_owned_lease_at_rw(face.ring_slot()?, window.base(), window.len())?;
        let (base, len) = {
            // SAFETY: the caller established exclusive client ownership.
            let bytes = unsafe { mapping.as_mut() };
            (bytes.as_mut_ptr(), bytes.len())
        };
        // The private owned face below keeps this window live. It cannot be
        // extracted or replaced through a public accessor while rings borrow it.
        let bytes = unsafe { core::slice::from_raw_parts_mut(base, len) };
        let rings = kore_rt::svc::client_rings::<HostLinkWire>(
            bytes,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .ok()?;
        Some(RingHostLink {
            pair: SvcClient::new(
                transport,
                rings,
                ClientEndpoints {
                    kick: face.kick_slot(),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            _owned: Some(OwnedClientFace {
                _mapping: mapping,
                _face: face,
            }),
        })
    }

    async fn call(&mut self, request: &ClientRequest) -> Result<ClientResponse, ClientError> {
        self.pair
            .call(request)
            .await
            .map(|reply| reply.response)
            .map_err(ClientError::Call)
    }

    async fn accepted(&mut self, request: ClientRequest) -> Result<(), ClientError> {
        let expected = request.request_id();
        match self.call(&request).await? {
            ClientResponse::Accepted { request } if request == expected => Ok(()),
            ClientResponse::Refused { request, code } if request == expected => {
                Err(ClientError::Refused(code))
            }
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    /// The service accepts a founded bulk verb before the transport peer has
    /// necessarily acknowledged the preceding wire frame. Its Backpressure
    /// refusal is transactional, so retry the exact record across bounded IPC
    /// turns while the HostLink worker pumps the host ACK. Other refusals are
    /// terminal and ordinary control verbs never use this retry policy.
    async fn accepted_bulk(&mut self, request: &ClientRequest) -> Result<(), ClientError> {
        let expected = request.request_id();
        for _ in 0..BULK_CALL_RETRY_LIMIT {
            match self.call(request).await? {
                ClientResponse::Accepted { request } if request == expected => return Ok(()),
                ClientResponse::Refused {
                    request,
                    code: RefusalCode::Backpressure,
                } if request == expected => {}
                ClientResponse::Refused { request, code } if request == expected => {
                    return Err(ClientError::Refused(code));
                }
                _ => return Err(ClientError::UnexpectedReply),
            }
        }
        Err(ClientError::Refused(RefusalCode::Backpressure))
    }

    pub async fn send(&mut self, request: RequestId, payload: &[u8]) -> Result<(), ClientError> {
        let payload = Payload::copy_from(payload).ok_or(ClientError::InvalidPayload)?;
        self.accepted(ClientRequest::Send {
            kind: INFER_REQUEST,
            request,
            payload,
        })
        .await
    }

    /// Send one typed KIR1 request through this held client face. The 4 KiB
    /// inference bound is independent of the founded ring's 512-byte record:
    /// larger requests use the existing credited bulk lane, without trimming
    /// the tool/resource catalogue or changing the inference request bytes.
    ///
    /// `request` must be fresh for this route generation. Its bulk correlation
    /// is domain-separated below; neither correlation selects authority.
    pub async fn send_inference(
        &mut self,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), ClientError> {
        if kore_hostlink::decode_inference_request(payload).is_none() {
            return Err(ClientError::InvalidPayload);
        }
        if payload.len() <= CLIENT_CHUNK_MAX {
            self.send(request, payload).await
        } else {
            self.send_inference_bulk(request, inference_bulk_id(request), payload)
                .await
        }
    }

    /// Send one complete KIR1 request through the already negotiated bulk
    /// lane. The payload remains the ordinary inference protocol; this only
    /// fragments it across the founded client's bounded 512-byte records.
    pub async fn send_inference_bulk(
        &mut self,
        request: RequestId,
        bulk_id: BulkId,
        payload: &[u8],
    ) -> Result<(), ClientError> {
        if payload.is_empty() || payload.len() > kore_hostlink::INLINE_MAX {
            return Err(ClientError::InvalidPayload);
        }
        let chunk_count = payload.len().div_ceil(FOUNDED_BULK_WIRE_CHUNK);
        let reference = BulkRefV1 {
            bulk_id,
            total_len: U32::new(
                u32::try_from(payload.len()).map_err(|_| ClientError::InvalidBulk)?,
            ),
            chunk_count: U16::new(
                u16::try_from(chunk_count).map_err(|_| ClientError::InvalidBulk)?,
            ),
            content_kind: U16::new(INFERENCE_REQUEST_CONTENT),
            digest_blake3: *blake3::hash(payload).as_bytes(),
        };
        let mut upload = self.begin_bulk(request, reference).await?;
        for fragment in payload.chunks(CLIENT_CHUNK_MAX) {
            self.send_bulk_chunk(&mut upload, fragment).await?;
        }
        self.finish_bulk(&mut upload).await
    }

    pub async fn send_workspace(
        &mut self,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), ClientError> {
        let payload = Payload::copy_from(payload).ok_or(ClientError::InvalidPayload)?;
        self.accepted(ClientRequest::Send {
            kind: WORKSPACE_REQUEST,
            request,
            payload,
        })
        .await
    }

    /// Report one post-commit inert authority observation. The HostLink
    /// service accepts this verb only from the structurally assigned broker
    /// pair; an agent pair cannot forge user-intent or authority provenance.
    pub async fn report_authority(
        &mut self,
        request: RequestId,
        report: AuthorityReportView<'_>,
    ) -> Result<(), ClientError> {
        self.authority_observation(request, report, None).await
    }

    /// The caller retains its queue front and correlation until Accepted.
    /// An occupied pair resumes only the byte-identical report, never a
    /// replacement with the same request ID or a sibling operation's reply.
    pub async fn observe_authority(
        &mut self,
        request: RequestId,
        report: AuthorityReportView<'_>,
        budget_ns: u64,
    ) -> Result<(), ClientError> {
        self.authority_observation(request, report, Some(budget_ns))
            .await
    }

    async fn authority_observation(
        &mut self,
        request: RequestId,
        report: AuthorityReportView<'_>,
        budget_ns: Option<u64>,
    ) -> Result<(), ClientError> {
        let mut wire =
            [0; kore_hostlink::AUTHORITY_REPORT_HEADER_LEN + kore_hostlink::AUTHORITY_LABEL_MAX];
        let length = report
            .encode(&mut wire)
            .ok_or(ClientError::InvalidPayload)?;
        let payload = Payload::copy_from(&wire[..length]).ok_or(ClientError::InvalidPayload)?;
        self.observe_request(
            ClientRequest::Send {
                kind: AUTHORITY_REPORT,
                request,
                payload,
            },
            !self.is_idle(),
            budget_ns,
        )
        .await
    }

    pub fn authority_request_pending(&self) -> Option<RequestId> {
        match self
            .pair
            .outstanding_request()
            .and_then(crate::wire::decode_req)
        {
            Some(ClientRequest::Send {
                kind: AUTHORITY_REPORT,
                request,
                ..
            }) => Some(request),
            _ => None,
        }
    }

    pub async fn cancel(&mut self, request: RequestId) -> Result<(), ClientError> {
        self.accepted(ClientRequest::Cancel { request }).await
    }

    pub async fn observe_cancel(
        &mut self,
        request: RequestId,
        budget_ns: u64,
    ) -> Result<(), ClientError> {
        self.cancel_observation(request, Some(budget_ns)).await
    }

    async fn cancel_observation(
        &mut self,
        request: RequestId,
        budget_ns: Option<u64>,
    ) -> Result<(), ClientError> {
        self.observe_request(
            ClientRequest::Cancel { request },
            !self.is_idle(),
            budget_ns,
        )
        .await
    }

    pub fn cancel_request_pending(&self) -> Option<RequestId> {
        match self
            .pair
            .outstanding_request()
            .and_then(crate::wire::decode_req)
        {
            Some(ClientRequest::Cancel { request }) => Some(request),
            _ => None,
        }
    }

    pub fn status_request_pending(&self) -> bool {
        matches!(
            self.pair
                .outstanding_request()
                .and_then(crate::wire::decode_req),
            Some(ClientRequest::Status)
        )
    }

    pub async fn poll(&mut self) -> Result<Option<ClientEvent>, ClientError> {
        match self.call(&ClientRequest::Poll).await? {
            ClientResponse::Event(event) => Ok(Some(event)),
            ClientResponse::Empty => Ok(None),
            ClientResponse::Refused { code, .. } => Err(ClientError::Refused(code)),
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    pub fn poll_request_pending(&self) -> bool {
        matches!(
            self.pair
                .outstanding_request()
                .and_then(crate::wire::decode_req),
            Some(ClientRequest::Poll)
        )
    }

    /// Continue the exact shared mailbox poll after an observation expires.
    /// A replacement Poll could dequeue another event and lose the first one.
    pub async fn observe_poll(
        &mut self,
        resume: bool,
        budget_ns: u64,
    ) -> Result<Option<ClientEvent>, ClientError> {
        self.poll_observation(resume, Some(budget_ns)).await
    }

    pub(crate) async fn poll_observation(
        &mut self,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<Option<ClientEvent>, ClientError> {
        match self
            .observe_call(&ClientRequest::Poll, resume, budget_ns)
            .await?
        {
            ClientResponse::Event(event) => Ok(Some(event)),
            ClientResponse::Empty => Ok(None),
            ClientResponse::Refused { code, .. } => Err(ClientError::Refused(code)),
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    pub async fn status(&mut self) -> Result<ClientStatus, ClientError> {
        match self.call(&ClientRequest::Status).await? {
            ClientResponse::Status(status) => Ok(status),
            ClientResponse::Refused { code, .. } => Err(ClientError::Refused(code)),
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    /// Observe startup readiness without abandoning a published Status query.
    /// A different outstanding operation remains with its original owner.
    pub async fn observe_status(&mut self, budget_ns: u64) -> Result<ClientStatus, ClientError> {
        self.status_observation(Some(budget_ns)).await
    }

    async fn status_observation(
        &mut self,
        budget_ns: Option<u64>,
    ) -> Result<ClientStatus, ClientError> {
        let resume = matches!(
            self.pair
                .outstanding_request()
                .and_then(crate::wire::decode_req),
            Some(ClientRequest::Status)
        );
        match self
            .observe_call(&ClientRequest::Status, resume, budget_ns)
            .await?
        {
            ClientResponse::Status(status) => Ok(status),
            ClientResponse::Refused { code, .. } => Err(ClientError::Refused(code)),
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    /// Bounded startup probe. A timeout leaves this exact request outstanding;
    /// the SPSC owner refuses subsequent publication rather than aliasing a
    /// late status reply with another operation.
    pub async fn status_bounded(&mut self, budget_ns: u64) -> Result<ClientStatus, ClientError> {
        match self
            .pair
            .call_bounded(&ClientRequest::Status, budget_ns)
            .await
            .map_err(ClientError::Call)?
            .response
        {
            ClientResponse::Status(status) => Ok(status),
            ClientResponse::Refused { code, .. } => Err(ClientError::Refused(code)),
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    async fn route_lifecycle(&mut self, request: ClientRequest) -> Result<(), ClientError> {
        let expected = request.request_id();
        for _ in 0..ROUTE_CALL_RETRY_LIMIT {
            match self.call(&request).await? {
                ClientResponse::Accepted { request } if request == expected => return Ok(()),
                ClientResponse::Refused {
                    request,
                    code: RefusalCode::Backpressure,
                } if request == expected => {}
                ClientResponse::Refused { request, code } if request == expected => {
                    return Err(ClientError::Refused(code));
                }
                _ => return Err(ClientError::UnexpectedReply),
            }
        }
        Err(ClientError::Refused(RefusalCode::Backpressure))
    }

    /// Rotate a dynamic agent row through the structurally trusted Broker
    /// pair after close_agent acknowledged the exact predecessor and the old
    /// client was reaped. Agent/workspace pairs receive PolicyDenied.
    pub async fn rotate_agent(
        &mut self,
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
    ) -> Result<(), ClientError> {
        self.route_lifecycle(ClientRequest::RotateAgent {
            request,
            slot,
            generation,
        })
        .await
    }

    pub async fn close_agent(
        &mut self,
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
    ) -> Result<(), ClientError> {
        self.route_lifecycle(ClientRequest::CloseAgent {
            request,
            slot,
            generation,
        })
        .await
    }

    /// A shared native owner must settle its outstanding route operation
    /// before another lane can publish on this uncorrelated service pair.
    pub fn is_idle(&self) -> bool {
        self.pair.is_idle()
    }

    pub fn route_request_pending(&self) -> bool {
        matches!(
            self.pair
                .outstanding_request()
                .and_then(crate::wire::decode_req),
            Some(ClientRequest::CloseAgent { .. } | ClientRequest::RotateAgent { .. })
        )
    }

    /// FrontendCall already owns a complete bounded inline envelope. Resume
    /// compares its complete bytes on the same held face before consuming any
    /// acknowledgement, without replaying a claim or response publication.
    pub(crate) async fn observe_frontend_send(
        &mut self,
        request: RequestId,
        bytes: &[u8],
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<(), ClientError> {
        if kore_hostlink::decode_inference_request(bytes).is_none() {
            return Err(ClientError::InvalidPayload);
        }
        let payload = Payload::copy_from(bytes).ok_or(ClientError::InvalidPayload)?;
        self.observe_request(
            ClientRequest::Send {
                kind: INFER_REQUEST,
                request,
                payload,
            },
            resume,
            budget_ns,
        )
        .await
    }

    async fn observe_request(
        &mut self,
        request: ClientRequest,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<(), ClientError> {
        match self.observe_call(&request, resume, budget_ns).await? {
            ClientResponse::Accepted { request: got } if got == request.request_id() => Ok(()),
            ClientResponse::Refused { request: got, code } if got == request.request_id() => {
                Err(ClientError::Refused(code))
            }
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    async fn observe_call(
        &mut self,
        request: &ClientRequest,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<ClientResponse, ClientError> {
        let reply = match (resume, budget_ns) {
            (true, Some(budget)) => self.pair.resume_exact_bounded(request, budget).await,
            (true, None) => self.pair.resume_exact(request).await,
            (false, Some(budget)) => self.pair.call_bounded(request, budget).await,
            (false, None) => self.pair.call(request).await,
        }
        .map_err(ClientError::Call)?;
        Ok(reply.response)
    }

    /// Observe the surviving owner's exact operation without replaying an
    /// uncertain publication. Supervisor/worker pairs are refused by the
    /// provider; only the separately endowed owner pair can use this API.
    pub async fn observe_supervisor(
        &mut self,
        request: RequestId,
        operation: crate::SupervisorOperation,
        generation: u64,
        resume: bool,
        budget_ns: u64,
    ) -> Result<crate::SupervisorState, ClientError> {
        let command = ClientRequest::Supervisor {
            request,
            operation,
            generation,
        };
        match self.observe_call(&command, resume, Some(budget_ns)).await? {
            ClientResponse::Supervisor {
                request: got,
                state,
            } if got == request => Ok(state),
            ClientResponse::Refused { request: got, code } if got == request => {
                Err(ClientError::Refused(code))
            }
            _ => Err(ClientError::UnexpectedReply),
        }
    }

    /// One observation slice of a retained route-close operation. `resume`
    /// compares the entire already-published request before touching the pair;
    /// it never publishes a replacement, even with the same correlation.
    pub async fn observe_close_agent(
        &mut self,
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
        resume: bool,
        budget_ns: u64,
    ) -> Result<(), ClientError> {
        self.observe_request(
            ClientRequest::CloseAgent {
                request,
                slot,
                generation,
            },
            resume,
            Some(budget_ns),
        )
        .await
    }

    /// Bounded exact-request continuation of the successor route installation.
    pub async fn observe_rotate_agent(
        &mut self,
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
        resume: bool,
        budget_ns: u64,
    ) -> Result<(), ClientError> {
        self.observe_request(
            ClientRequest::RotateAgent {
                request,
                slot,
                generation,
            },
            resume,
            Some(budget_ns),
        )
        .await
    }

    pub async fn begin_bulk(
        &mut self,
        request: RequestId,
        reference: BulkRefV1,
    ) -> Result<BulkUpload, ClientError> {
        let total = reference.total_len.get();
        let chunks = reference.chunk_count.get();
        let wire_capacity = u32::from(chunks).saturating_mul(FOUNDED_BULK_WIRE_CHUNK as u32);
        let expected_chunks = total.div_ceil(FOUNDED_BULK_WIRE_CHUNK as u32);
        if !reference.shape_is_valid()
            || chunks != u16::try_from(expected_chunks).map_err(|_| ClientError::InvalidBulk)?
            || total > wire_capacity
        {
            return Err(ClientError::InvalidBulk);
        }
        self.accepted_bulk(&ClientRequest::BulkBegin { request, reference })
            .await?;
        Ok(BulkUpload {
            request,
            reference,
            offset: 0,
            wire_bytes: 0,
            wire_chunks: 0,
            fragments: 0,
            hasher: blake3::Hasher::new(),
        })
    }

    pub async fn send_bulk_chunk(
        &mut self,
        upload: &mut BulkUpload,
        bytes: &[u8],
    ) -> Result<(), ClientError> {
        if bytes.is_empty() || bytes.len() > CLIENT_CHUNK_MAX {
            return Err(ClientError::InvalidPayload);
        }
        let next = upload
            .offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| ClientError::InvalidBulk)?)
            .ok_or(ClientError::InvalidBulk)?;
        let remaining_wire = FOUNDED_BULK_WIRE_CHUNK - usize::from(upload.wire_bytes);
        if next > upload.reference.total_len.get() || bytes.len() > remaining_wire {
            return Err(ClientError::InvalidBulk);
        }
        let payload = Payload::copy_from(bytes).ok_or(ClientError::InvalidPayload)?;
        self.accepted_bulk(&ClientRequest::BulkChunk {
            request: upload.request,
            bulk_id: upload.reference.bulk_id,
            bytes: payload,
        })
        .await?;
        upload.hasher.update(bytes);
        upload.offset = next;
        upload.fragments = upload.fragments.saturating_add(1);
        upload.wire_bytes += bytes.len() as u16;
        if usize::from(upload.wire_bytes) == FOUNDED_BULK_WIRE_CHUNK
            || upload.offset == upload.reference.total_len.get()
        {
            upload.wire_bytes = 0;
            upload.wire_chunks += 1;
        }
        Ok(())
    }

    pub async fn finish_bulk(&mut self, upload: &mut BulkUpload) -> Result<(), ClientError> {
        let digest = *upload.hasher.clone().finalize().as_bytes();
        if upload.offset != upload.reference.total_len.get()
            || upload.wire_bytes != 0
            || upload.wire_chunks != upload.reference.chunk_count.get()
            || digest != upload.reference.digest_blake3
        {
            return Err(ClientError::InvalidBulk);
        }
        self.accepted_bulk(&ClientRequest::BulkFinish {
            request: upload.request,
            bulk_id: upload.reference.bulk_id,
            total_len: upload.offset,
            digest_blake3: digest,
        })
        .await
    }

    pub async fn abort_bulk(
        &mut self,
        upload: BulkUpload,
        reason: RefusalCode,
    ) -> Result<(), ClientError> {
        self.accepted_bulk(&ClientRequest::BulkAbort {
            request: upload.request,
            bulk_id: upload.reference.bulk_id,
            reason,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    use kore_abi::session::{Committed, SlotRef, Status};
    use kore_rt::{
        session::SessionBuilder,
        svc::{SvcProtocol, split_window},
    };
    use zerocopy::byteorder::little_endian::{U16, U32};

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
    struct WindowBytes([u8; crate::HOSTLINK_PAIR_WINDOW_BYTES]);

    fn worker_observe(
        send: &mut InferenceSend,
        link: &mut RingHostLink<'_, '_, AcceptKick>,
    ) -> Poll<Result<Option<ClientEvent>, ClientError>> {
        let mut context = Context::from_waker(Waker::noop());
        // Drop every borrowed observation, including each published wait.
        let mut future = core::pin::pin!(send.event_observation(link, None));
        future.as_mut().poll(&mut context)
    }

    fn broker_observe(
        link: &mut RingHostLink<'_, '_, AcceptKick>,
        request: RequestId,
        report: Option<AuthorityReportView<'_>>,
    ) -> Poll<Result<(), ClientError>> {
        let mut context = Context::from_waker(Waker::noop());
        let mut future = core::pin::pin!(async {
            match report {
                Some(report) => link.authority_observation(request, report, None).await,
                None => link.cancel_observation(request, None).await,
            }
        });
        future.as_mut().poll(&mut context)
    }

    fn broker_report() -> AuthorityReportView<'static> {
        AuthorityReportView {
            state: kore_hostlink::AuthorityState::Installed,
            kind: kore_hostlink::AuthorityKind::View,
            agent_slot: 1,
            generation: u64::from(u32::MAX) + 9,
            rights: 1,
            label: "selected source",
        }
    }

    #[test]
    fn broker_commands_resume_exact_bytes_and_never_consume_a_sibling_acknowledgement() {
        let request = RequestId(id(71));
        for report in [Some(broker_report()), None] {
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            for attempt in 0..3 {
                assert!(broker_observe(&mut link, request, report).is_pending());
                if attempt == 0 {
                    match HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap() {
                        ClientRequest::Send {
                            kind: AUTHORITY_REPORT,
                            request: got,
                            payload,
                        } => {
                            assert_eq!(got, request);
                            assert_eq!(
                                kore_hostlink::decode_authority_report(payload.as_slice()),
                                report
                            );
                        }
                        ClientRequest::Cancel { request: got } => {
                            assert_eq!(got, request);
                            assert!(report.is_none());
                        }
                        other => panic!("unexpected operation: {other:?}"),
                    }
                } else {
                    assert!(serve.req.pop().is_none());
                }
            }
            assert_eq!(link.authority_request_pending(), report.map(|_| request));
            assert_eq!(
                link.cancel_request_pending(),
                report.is_none().then_some(request)
            );
            assert!(!link.status_request_pending());
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Accepted { request },
                    0,
                ))
                .unwrap();
            assert!(matches!(
                broker_observe(&mut link, RequestId(id(72)), report),
                Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
            ));
            // A Cancel cannot consume a report's ack, or vice versa, even
            // when the untrusted correlation is deliberately identical.
            let opposite = if report.is_some() {
                None
            } else {
                Some(broker_report())
            };
            assert!(matches!(
                broker_observe(&mut link, request, opposite),
                Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
            ));
            if let Some(report) = report {
                for changed in [
                    AuthorityReportView {
                        rights: 3,
                        ..report
                    },
                    AuthorityReportView {
                        generation: 9,
                        ..report
                    },
                    AuthorityReportView {
                        label: "another source",
                        ..report
                    },
                ] {
                    assert!(matches!(
                        broker_observe(&mut link, request, Some(changed)),
                        Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                    ));
                }
            }
            assert!(!link.is_idle());
            assert!(matches!(
                broker_observe(&mut link, request, report),
                Poll::Ready(Ok(()))
            ));
            assert!(link.is_idle());
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn broker_command_backpressure_retains_identity_and_wrong_acceptance_never_confirms() {
        let request = RequestId(id(73));
        for report in [Some(broker_report()), None] {
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut original = None;
            for attempt in 0..3 {
                assert!(broker_observe(&mut link, request, report).is_pending());
                let operation = HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap();
                if let Some(original) = &original {
                    assert_eq!(&operation, original);
                } else {
                    original = Some(operation);
                }
                let response = if attempt < 2 {
                    ClientResponse::Refused {
                        request,
                        code: RefusalCode::Backpressure,
                    }
                } else {
                    ClientResponse::Accepted {
                        request: RequestId(id(74)),
                    }
                };
                serve
                    .rep
                    .push(HostLinkWire::encode_rep(&response, 0))
                    .unwrap();
                let result = broker_observe(&mut link, request, report);
                if attempt < 2 {
                    assert!(matches!(
                        result,
                        Poll::Ready(Err(ClientError::Refused(RefusalCode::Backpressure)))
                    ));
                } else {
                    assert!(matches!(
                        result,
                        Poll::Ready(Err(ClientError::Call(CallError::Wire)))
                    ));
                    assert!(matches!(
                        broker_observe(&mut link, request, report),
                        Poll::Ready(Err(ClientError::Call(CallError::Poisoned)))
                    ));
                }
                assert_eq!(link.is_idle(), attempt < 2);
                assert!(serve.req.pop().is_none());
            }
        }
    }

    #[test]
    fn invalid_authority_observation_cannot_publish_or_consume_a_pending_cancel() {
        let request = RequestId(id(75));
        let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let (client, mut serve) = split_window::<HostLinkWire>(
            &mut window.0,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        let transport = AcceptKick;
        let mut link = RingHostLink::from_pair(
            SvcClient::new(
                &transport,
                client,
                ClientEndpoints {
                    kick: SlotRef::from_wire(30),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling(),
        );
        let invalid = Some(AuthorityReportView {
            generation: 0,
            ..broker_report()
        });
        assert!(matches!(
            broker_observe(&mut link, request, invalid),
            Poll::Ready(Err(ClientError::InvalidPayload))
        ));
        assert!(link.is_idle());
        assert!(serve.req.pop().is_none());
        assert!(broker_observe(&mut link, request, None).is_pending());
        serve.req.pop().unwrap();
        serve
            .rep
            .push(HostLinkWire::encode_rep(
                &ClientResponse::Accepted { request },
                0,
            ))
            .unwrap();
        assert!(matches!(
            broker_observe(&mut link, request, invalid),
            Poll::Ready(Err(ClientError::InvalidPayload))
        ));
        assert_eq!(link.cancel_request_pending(), Some(request));
        assert!(matches!(
            broker_observe(&mut link, request, None),
            Poll::Ready(Ok(()))
        ));
    }

    #[test]
    fn worker_status_observation_resumes_exact_query_and_leaves_other_operations_owned() {
        let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let (client, mut serve) = split_window::<HostLinkWire>(
            &mut window.0,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        let transport = AcceptKick;
        let mut link = RingHostLink::from_pair(
            SvcClient::new(
                &transport,
                client,
                ClientEndpoints {
                    kick: SlotRef::from_wire(30),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling(),
        );
        let mut context = Context::from_waker(Waker::noop());
        for repeat in 0..3 {
            {
                let mut observation = core::pin::pin!(link.status_observation(None));
                assert!(observation.as_mut().poll(&mut context).is_pending());
            }
            if repeat == 0 {
                assert_eq!(
                    HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                    Some(ClientRequest::Status)
                );
            } else {
                assert!(serve.req.pop().is_none());
            }
        }
        let status = ClientStatus {
            phase: crate::ClientLinkPhase::Active,
            bulk_transport: crate::BulkTransport::Serial,
            queued_status: 0,
            queued_control: 0,
            queued_bulk: 0,
        };
        serve
            .rep
            .push(HostLinkWire::encode_rep(&ClientResponse::Status(status), 0))
            .unwrap();
        {
            let mut observation = core::pin::pin!(link.status_observation(None));
            assert!(
                matches!(observation.as_mut().poll(&mut context), Poll::Ready(Ok(got)) if got == status)
            );
        }
        assert!(link.is_idle());
        assert!(serve.req.pop().is_none());
        {
            let mut poll = core::pin::pin!(link.poll_observation(false, None));
            assert!(poll.as_mut().poll(&mut context).is_pending());
        }
        assert_eq!(
            HostLinkWire::decode_req(&serve.req.pop().unwrap()),
            Some(ClientRequest::Poll)
        );
        serve
            .rep
            .push(HostLinkWire::encode_rep(&ClientResponse::Empty, 0))
            .unwrap();
        {
            let mut observation = core::pin::pin!(link.status_observation(None));
            assert!(matches!(
                observation.as_mut().poll(&mut context),
                Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
            ));
        }
        assert!(link.poll_request_pending());
        {
            let mut poll = core::pin::pin!(link.poll_observation(true, None));
            assert!(matches!(
                poll.as_mut().poll(&mut context),
                Poll::Ready(Ok(None))
            ));
        }
        assert!(link.is_idle());
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn worker_inference_observation_cannot_skip_upload_with_a_preexisting_poll_reply() {
        for same_request in [false, true] {
            let request = RequestId(id(85));
            let bytes = inference_bytes(CLIENT_CHUNK_MAX + 1);
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut poll = core::pin::pin!(link.poll_observation(false, None));
                assert!(poll.as_mut().poll(&mut context).is_pending());
            }
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                Some(ClientRequest::Poll)
            );
            let mut send = InferenceSend::new(request, &bytes).unwrap();
            assert!(worker_observe(&mut send, &mut link).is_pending());
            assert!(serve.req.pop().is_none());
            let event = ClientEvent::Control {
                kind: kore_hostlink::INFER_REPLY,
                request: if same_request {
                    request
                } else {
                    RequestId(id(86))
                },
                payload: Payload::copy_from(b"not a reply to an unpublished inference").unwrap(),
            };
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Event(event.clone()),
                    0,
                ))
                .unwrap();
            let result = worker_observe(&mut send, &mut link);
            assert!(
                serve.req.pop().is_none(),
                "the previous Poll owns this observation"
            );
            assert!(!send.upload_settled());
            assert_eq!(send.bytes.as_slice(), bytes.as_slice());
            assert_eq!(send.offset, 0);
            if same_request {
                assert!(matches!(
                    result,
                    Poll::Ready(Err(ClientError::UnexpectedReply))
                ));
                assert!(send.failed());
            } else {
                assert!(matches!(result, Poll::Ready(Ok(Some(got))) if got == event));
                assert!(worker_observe(&mut send, &mut link).is_pending());
                assert!(
                    matches!(HostLinkWire::decode_req(&serve.req.pop().unwrap()), Some(ClientRequest::BulkBegin { request: got, .. }) if got == request)
                );
            }
        }
    }

    #[test]
    fn worker_inference_observation_retains_upload_and_original_mailbox_poll() {
        for length in [CLIENT_CHUNK_MAX, 2 * CLIENT_CHUNK_MAX + 3] {
            let request = RequestId(id(81));
            let bytes = inference_bytes(length);
            let mut send = InferenceSend::new(request, &bytes).unwrap();
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut received = std::vec::Vec::new();
            let mut operation_count = 0;
            while !send.upload_settled() {
                assert!(worker_observe(&mut send, &mut link).is_pending());
                let operation = HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap();
                assert!(worker_observe(&mut send, &mut link).is_pending());
                assert!(serve.req.pop().is_none(), "resume publishes no duplicate");
                match operation {
                    ClientRequest::Send { payload, .. } => {
                        received.extend_from_slice(payload.as_slice())
                    }
                    ClientRequest::BulkChunk { bytes, .. } => {
                        received.extend_from_slice(bytes.as_slice())
                    }
                    ClientRequest::BulkBegin { .. } | ClientRequest::BulkFinish { .. } => {}
                    _ => panic!("an unfinished upload cannot issue a fresh Poll"),
                }
                serve
                    .rep
                    .push(HostLinkWire::encode_rep(
                        &ClientResponse::Accepted { request },
                        0,
                    ))
                    .unwrap();
                assert!(matches!(
                    worker_observe(&mut send, &mut link),
                    Poll::Ready(Ok(None))
                ));
                assert!(
                    serve.req.pop().is_none(),
                    "one service operation per observation"
                );
                operation_count += 1;
                assert!(operation_count < 32);
            }
            assert_eq!(received, bytes);
            assert!(worker_observe(&mut send, &mut link).is_pending());
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                Some(ClientRequest::Poll)
            );
            assert!(worker_observe(&mut send, &mut link).is_pending());
            assert!(serve.req.pop().is_none());
            let event = ClientEvent::Control {
                kind: kore_hostlink::INFER_REPLY,
                request,
                payload: Payload::copy_from(b"one original model event").unwrap(),
            };
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Event(event.clone()),
                    0,
                ))
                .unwrap();
            assert!(
                matches!(worker_observe(&mut send, &mut link), Poll::Ready(Ok(Some(got))) if got == event)
            );
            assert!(link.is_idle());
            assert!(
                serve.req.pop().is_none(),
                "no second dequeue after the late event"
            );
        }
    }

    #[test]
    fn worker_inference_observation_retires_refused_prefix_before_reporting_failure() {
        for malformed in [false, true] {
            let request = RequestId(id(82));
            let bytes = inference_bytes(2 * CLIENT_CHUNK_MAX + 3);
            let mut send = InferenceSend::new(request, &bytes).unwrap();
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            // Begin and the first fragment commit; the second fragment refuses.
            for stage in 0..3 {
                assert!(worker_observe(&mut send, &mut link).is_pending());
                let operation = HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap();
                assert!(matches!(
                    operation,
                    ClientRequest::BulkBegin { .. } | ClientRequest::BulkChunk { .. }
                ));
                let response = if stage == 2 {
                    ClientResponse::Refused {
                        request,
                        code: RefusalCode::RouteClosed,
                    }
                } else {
                    ClientResponse::Accepted { request }
                };
                serve
                    .rep
                    .push(HostLinkWire::encode_rep(&response, 0))
                    .unwrap();
                assert!(matches!(
                    worker_observe(&mut send, &mut link),
                    Poll::Ready(Ok(None))
                ));
                assert!(!send.upload_settled());
            }
            for backpressure in [true, false] {
                assert!(worker_observe(&mut send, &mut link).is_pending());
                assert_eq!(
                    HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                    Some(ClientRequest::BulkAbort {
                        request,
                        bulk_id: inference_bulk_id(request),
                        reason: RefusalCode::RouteClosed,
                    })
                );
                assert!(worker_observe(&mut send, &mut link).is_pending());
                assert!(serve.req.pop().is_none());
                let response = if backpressure {
                    ClientResponse::Refused {
                        request,
                        code: RefusalCode::Backpressure,
                    }
                } else {
                    ClientResponse::Accepted { request }
                };
                serve
                    .rep
                    .push(HostLinkWire::encode_rep(&response, 0))
                    .unwrap();
                assert!(matches!(
                    worker_observe(&mut send, &mut link),
                    Poll::Ready(Ok(None))
                ));
                assert!(
                    !send.upload_settled(),
                    "local acceptance is not abort retirement"
                );
            }
            let foreign = ClientEvent::Cancel {
                request: RequestId(id(83)),
            };
            assert!(worker_observe(&mut send, &mut link).is_pending());
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                Some(ClientRequest::Poll)
            );
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Event(foreign.clone()),
                    0,
                ))
                .unwrap();
            assert!(
                matches!(worker_observe(&mut send, &mut link), Poll::Ready(Ok(Some(got))) if got == foreign)
            );
            assert!(!send.upload_settled());
            assert!(worker_observe(&mut send, &mut link).is_pending());
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                Some(ClientRequest::Poll)
            );
            let abort_id = if malformed {
                id(84)
            } else {
                inference_bulk_id(request).0
            };
            let event = ClientEvent::Control {
                kind: kore_hostlink::BULK_ABORTED,
                request,
                payload: Payload::copy_from(&abort_id).unwrap(),
            };
            serve
                .rep
                .push(HostLinkWire::encode_rep(&ClientResponse::Event(event), 0))
                .unwrap();
            if malformed {
                assert!(matches!(
                    worker_observe(&mut send, &mut link),
                    Poll::Ready(Err(ClientError::UnexpectedReply))
                ));
                assert!(send.failed());
                assert!(!send.upload_settled());
                assert!(worker_observe(&mut send, &mut link).is_ready());
            } else {
                assert!(matches!(
                    worker_observe(&mut send, &mut link),
                    Poll::Ready(Err(ClientError::Refused(RefusalCode::RouteClosed)))
                ));
                assert!(send.upload_settled());
                assert_eq!(send.retired_refusal(), Some(RefusalCode::RouteClosed));
            }
            assert_eq!(send.bytes.as_slice(), bytes.as_slice());
            assert_eq!(send.offset, CLIENT_CHUNK_MAX);
            assert!(
                serve.req.pop().is_none(),
                "failure cannot replay or finish the rejected upload"
            );
        }
    }

    // Drive the real SvcClient and HostLink encoders against the opposite
    // SPSC halves. Only kernel notice delivery and peer responses are mocked.
    fn inference_exchange(
        request: RequestId,
        bytes: &[u8],
        mut reply: impl FnMut(&ClientRequest) -> ClientResponse,
    ) -> Result<(), ClientError> {
        let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let (client, mut serve) = split_window::<HostLinkWire>(
            &mut window.0,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        let transport = AcceptKick;
        let pair = SvcClient::new(
            &transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(30),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        )
        .with_direct_polling();
        let mut link = RingHostLink::from_pair(pair);
        let mut future = core::pin::pin!(link.send_inference(request, bytes));
        let mut context = Context::from_waker(Waker::noop());
        for _ in 0..128 {
            if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
                assert!(
                    serve.req.pop().is_none(),
                    "no unacknowledged trailing request"
                );
                return result;
            }
            while let Some(wire) = serve.req.pop() {
                let request = HostLinkWire::decode_req(&wire).expect("founded request");
                let response = reply(&request);
                serve
                    .rep
                    .push(HostLinkWire::encode_rep(&response, 0))
                    .unwrap();
            }
        }
        panic!("bounded inference exchange did not finish");
    }

    fn inference_bytes(length: usize) -> std::vec::Vec<u8> {
        let input = std::vec![b'x'; length - kore_hostlink::INFERENCE_REQUEST_HEADER_LEN];
        let mut bytes = std::vec![0; length];
        let len = kore_hostlink::InferenceRequestView {
            kind: kore_hostlink::InferenceRequestKind::AgentTurn,
            turn: 1,
            max_output_tokens: 512,
            timeout_ticks: 20_000,
            input: &input,
        }
        .encode(&mut bytes)
        .unwrap();
        assert_eq!(len, length);
        bytes
    }

    #[test]
    fn inference_transport_preserves_bytes_across_the_founded_record_boundary() {
        let mut previous_bulk = None;
        for length in [
            CLIENT_CHUNK_MAX,
            CLIENT_CHUNK_MAX + 1,
            kore_hostlink::INLINE_MAX,
        ] {
            let bytes = inference_bytes(length);
            let request = RequestId((length as u128).to_le_bytes());
            let mut received = std::vec::Vec::new();
            let mut reference = None;
            let mut finishes = 0;
            inference_exchange(request, &bytes, |record| {
                assert_eq!(record.request_id(), request);
                match record {
                    ClientRequest::Send { kind, payload, .. } => {
                        assert_eq!(*kind, INFER_REQUEST);
                        assert!(length <= CLIENT_CHUNK_MAX);
                        received.extend_from_slice(payload.as_slice());
                    }
                    ClientRequest::BulkBegin {
                        reference: begin, ..
                    } => {
                        assert!(length > CLIENT_CHUNK_MAX);
                        assert_eq!(begin.total_len.get() as usize, length);
                        assert_eq!(begin.content_kind.get(), INFERENCE_REQUEST_CONTENT);
                        assert_eq!(begin.chunk_count.get(), 1);
                        assert_eq!(begin.digest_blake3, *blake3::hash(&bytes).as_bytes());
                        assert_ne!(Some(begin.bulk_id), previous_bulk);
                        assert!(reference.replace(*begin).is_none());
                    }
                    ClientRequest::BulkChunk { bulk_id, bytes, .. } => {
                        assert_eq!(*bulk_id, reference.unwrap().bulk_id);
                        assert!(bytes.len() <= CLIENT_CHUNK_MAX);
                        received.extend_from_slice(bytes.as_slice());
                    }
                    ClientRequest::BulkFinish {
                        bulk_id,
                        total_len,
                        digest_blake3,
                        ..
                    } => {
                        assert_eq!(*bulk_id, reference.unwrap().bulk_id);
                        assert_eq!(*total_len as usize, length);
                        assert_eq!(*digest_blake3, *blake3::hash(&received).as_bytes());
                        finishes += 1;
                    }
                    _ => panic!("unexpected inference operation"),
                }
                ClientResponse::Accepted { request }
            })
            .unwrap();
            assert_eq!(received, bytes);
            assert_eq!(finishes, usize::from(length > CLIENT_CHUNK_MAX));
            previous_bulk = reference.map(|reference| reference.bulk_id);
        }
    }

    #[test]
    fn inference_fragment_backpressure_retries_exact_bytes_and_terminal_refusal_stops() {
        let request = RequestId(id(5));
        let bytes = inference_bytes(CLIENT_CHUNK_MAX + 73);
        let mut retry = None;
        let mut received = std::vec::Vec::new();
        let mut finish = false;
        let result = inference_exchange(request, &bytes, |record| {
            match record {
                ClientRequest::BulkChunk { bytes, .. } if retry.is_none() => {
                    retry = Some(record.clone());
                    assert!(!bytes.is_empty());
                    return ClientResponse::Refused {
                        request,
                        code: RefusalCode::Backpressure,
                    };
                }
                ClientRequest::BulkChunk { bytes, .. } if received.is_empty() => {
                    assert_eq!(Some(record), retry.as_ref());
                    received.extend_from_slice(bytes.as_slice());
                }
                ClientRequest::BulkChunk { .. } => {
                    return ClientResponse::Refused {
                        request,
                        code: RefusalCode::RouteClosed,
                    };
                }
                ClientRequest::BulkFinish { .. } => finish = true,
                ClientRequest::BulkBegin { .. } => {}
                _ => panic!("unexpected operation"),
            }
            ClientResponse::Accepted { request }
        });
        assert!(matches!(
            result,
            Err(ClientError::Refused(RefusalCode::RouteClosed))
        ));
        assert_eq!(received, bytes[..CLIENT_CHUNK_MAX]);
        assert!(
            !finish,
            "partial payload must never be submitted as inference"
        );
    }

    #[test]
    fn retained_inference_resumes_every_verb_without_replaying_acknowledged_bytes() {
        for length in [
            CLIENT_CHUNK_MAX,
            CLIENT_CHUNK_MAX + 1,
            kore_hostlink::INLINE_MAX,
        ] {
            let bytes = inference_bytes(length);
            let request = RequestId(id(61));
            let mut send = InferenceSend::new(request, &bytes).unwrap();
            // Model suspension after retaining the attempt but before the
            // SvcClient publishes it. The pair, not this flag, proves resume.
            send.in_flight = true;
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let pair = SvcClient::new(
                &transport,
                client,
                ClientEndpoints {
                    kick: SlotRef::from_wire(30),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling();
            let mut link = RingHostLink::from_pair(pair);
            let mut context = Context::from_waker(Waker::noop());
            let mut received = std::vec::Vec::new();
            let mut verbs = 0;
            loop {
                let operation = send.operation().unwrap();
                let offset = send.offset;
                // Every stage sees transactional backpressure, then a dropped
                // publication wait and another dropped exact-resume wait.
                for refused in [true, false] {
                    {
                        let mut future = core::pin::pin!(send.observe(&mut link, None));
                        assert!(future.as_mut().poll(&mut context).is_pending());
                    }
                    assert!(send.in_flight());
                    let wire = serve.req.pop().unwrap();
                    assert_eq!(HostLinkWire::decode_req(&wire).unwrap(), operation);
                    {
                        let mut future = core::pin::pin!(send.observe(&mut link, None));
                        assert!(future.as_mut().poll(&mut context).is_pending());
                    }
                    assert!(
                        serve.req.pop().is_none(),
                        "resuming cannot publish another verb"
                    );
                    assert_eq!(send.offset, offset);
                    let response = if refused {
                        ClientResponse::Refused {
                            request,
                            code: RefusalCode::Backpressure,
                        }
                    } else {
                        ClientResponse::Accepted { request }
                    };
                    serve
                        .rep
                        .push(HostLinkWire::encode_rep(&response, 0))
                        .unwrap();
                    {
                        // Same correlation and offset are insufficient: the
                        // complete payload/reference must match the owner.
                        let mut altered = bytes.clone();
                        altered[kore_hostlink::INFERENCE_REQUEST_HEADER_LEN..].fill(b'y');
                        let mut other = InferenceSend::new(request, &altered).unwrap();
                        other.stage = send.stage;
                        other.offset = send.offset;
                        other.in_flight = true;
                        assert!(send.resume_pending(&link));
                        assert!(!other.resume_pending(&link));
                        let mut wrong = core::pin::pin!(other.observe(&mut link, None));
                        assert!(matches!(
                            wrong.as_mut().poll(&mut context),
                            Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                        ));
                    }
                    {
                        let mut wrong = core::pin::pin!(link.observe_request(
                            ClientRequest::Cancel { request },
                            true,
                            None
                        ));
                        assert!(matches!(
                            wrong.as_mut().poll(&mut context),
                            Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                        ));
                    }
                    let result = {
                        let mut future = core::pin::pin!(send.observe(&mut link, None));
                        match future.as_mut().poll(&mut context) {
                            Poll::Ready(result) => result,
                            Poll::Pending => panic!("ack available"),
                        }
                    };
                    assert!(!send.in_flight());
                    if refused {
                        assert!(matches!(
                            result,
                            Err(ClientError::Refused(RefusalCode::Backpressure))
                        ));
                        assert_eq!(send.offset, offset);
                        assert_eq!(send.operation().unwrap(), operation);
                    } else {
                        assert!(result.is_ok());
                    }
                }
                verbs += 1;
                match operation {
                    ClientRequest::Send { payload, .. } => {
                        received.extend_from_slice(payload.as_slice())
                    }
                    ClientRequest::BulkChunk { bytes, .. } => {
                        received.extend_from_slice(bytes.as_slice())
                    }
                    ClientRequest::BulkBegin { reference, .. } => {
                        assert_eq!(reference.bulk_id, inference_bulk_id(request));
                        assert_eq!(reference.digest_blake3, *blake3::hash(&bytes).as_bytes());
                    }
                    ClientRequest::BulkFinish {
                        total_len,
                        digest_blake3,
                        ..
                    } => {
                        assert_eq!(total_len as usize, received.len());
                        assert_eq!(digest_blake3, *blake3::hash(&received).as_bytes());
                    }
                    _ => panic!("unexpected verb"),
                }
                assert!(!send.failed());
                if send.stage == InferenceStage::Complete {
                    break;
                }
                assert!(verbs < 32);
            }
            assert_eq!(received, bytes);
            assert_eq!(
                verbs,
                if length <= CLIENT_CHUNK_MAX {
                    1
                } else {
                    2 + length.div_ceil(CLIENT_CHUNK_MAX)
                }
            );
            let mut completed = core::pin::pin!(send.observe(&mut link, None));
            assert!(matches!(
                completed.as_mut().poll(&mut context),
                Poll::Ready(Ok(true))
            ));
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn retained_inference_refusal_waits_for_exact_remote_abort_receipt() {
        let request = RequestId(id(62));
        let bytes = inference_bytes(2 * CLIENT_CHUNK_MAX + 3);
        let mut send = InferenceSend::new(request, &bytes).unwrap();
        let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
        let (client, mut serve) = split_window::<HostLinkWire>(
            &mut window.0,
            crate::HOSTLINK_REQUEST_CAPACITY,
            crate::HOSTLINK_RESPONSE_CAPACITY,
        )
        .unwrap();
        let transport = AcceptKick;
        let pair = SvcClient::new(
            &transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(30),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        )
        .with_direct_polling();
        let mut link = RingHostLink::from_pair(pair);
        let mut context = Context::from_waker(Waker::noop());
        for stage in 0..3 {
            {
                let mut future = core::pin::pin!(send.observe(&mut link, None));
                assert!(future.as_mut().poll(&mut context).is_pending());
            }
            let operation = HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap();
            assert!(!matches!(operation, ClientRequest::BulkFinish { .. }));
            let response = if stage == 2 {
                ClientResponse::Refused {
                    request,
                    code: RefusalCode::RouteClosed,
                }
            } else {
                ClientResponse::Accepted { request }
            };
            serve
                .rep
                .push(HostLinkWire::encode_rep(&response, 0))
                .unwrap();
            let result = {
                let mut future = core::pin::pin!(send.observe(&mut link, None));
                match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => result,
                    _ => panic!("ack available"),
                }
            };
            if stage == 2 {
                assert!(matches!(
                    result,
                    Err(ClientError::Refused(RefusalCode::RouteClosed))
                ));
            } else {
                assert!(matches!(result, Ok(false)));
            }
        }
        assert!(!send.failed());
        assert!(send.retired_refusal().is_none());
        assert_eq!(send.request(), request);
        assert_eq!(send.offset, CLIENT_CHUNK_MAX);
        assert_eq!(send.bytes.as_slice(), bytes.as_slice());
        assert_eq!(send.reference.bulk_id, inference_bulk_id(request));
        for backpressure in [true, false] {
            {
                let mut abort = core::pin::pin!(send.observe(&mut link, None));
                assert!(abort.as_mut().poll(&mut context).is_pending());
            }
            let wire = serve.req.pop().unwrap();
            assert_eq!(
                HostLinkWire::decode_req(&wire).unwrap(),
                ClientRequest::BulkAbort {
                    request,
                    bulk_id: inference_bulk_id(request),
                    reason: RefusalCode::RouteClosed,
                }
            );
            {
                let mut abort = core::pin::pin!(send.observe(&mut link, None));
                assert!(abort.as_mut().poll(&mut context).is_pending());
            }
            assert!(serve.req.pop().is_none());
            let response = if backpressure {
                ClientResponse::Refused {
                    request,
                    code: RefusalCode::Backpressure,
                }
            } else {
                ClientResponse::Accepted { request }
            };
            serve
                .rep
                .push(HostLinkWire::encode_rep(&response, 0))
                .unwrap();
            let result = {
                let mut abort = core::pin::pin!(send.observe(&mut link, None));
                match abort.as_mut().poll(&mut context) {
                    Poll::Ready(result) => result,
                    _ => panic!("ack available"),
                }
            };
            assert_eq!(result.is_ok(), !backpressure);
            assert!(
                send.retired_refusal().is_none(),
                "service acceptance is not remote retirement"
            );
        }
        assert!(!send.pending_send());
        let unrelated = ClientEvent::Control {
            kind: kore_hostlink::BULK_ABORTED,
            request: RequestId(id(63)),
            payload: Payload::copy_from(&inference_bulk_id(request).0).unwrap(),
        };
        assert!(!send.accept_abort(&unrelated));
        let receipt = ClientEvent::Control {
            kind: kore_hostlink::BULK_ABORTED,
            request,
            payload: Payload::copy_from(&inference_bulk_id(request).0).unwrap(),
        };
        assert!(send.accept_abort(&receipt));
        assert_eq!(send.retired_refusal(), Some(RefusalCode::RouteClosed));
        assert!(!send.accept_abort(&receipt));
        let mut retry = core::pin::pin!(send.observe(&mut link, None));
        assert!(matches!(
            retry.as_mut().poll(&mut context),
            Poll::Ready(Ok(false))
        ));
        assert!(
            serve.req.pop().is_none(),
            "terminal failure cannot restart or finish an incomplete upload"
        );
    }

    #[test]
    fn retained_inference_unopened_refusal_needs_no_abort_and_allows_next_request() {
        for length in [CLIENT_CHUNK_MAX, CLIENT_CHUNK_MAX + 1] {
            let request = RequestId(id(65));
            let mut send = InferenceSend::new(request, &inference_bytes(length)).unwrap();
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let pair = SvcClient::new(
                &transport,
                client,
                ClientEndpoints {
                    kick: SlotRef::from_wire(30),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            )
            .with_direct_polling();
            let mut link = RingHostLink::from_pair(pair);
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut future = core::pin::pin!(send.observe(&mut link, None));
                assert!(future.as_mut().poll(&mut context).is_pending());
            }
            let operation = HostLinkWire::decode_req(&serve.req.pop().unwrap()).unwrap();
            assert!(matches!(
                operation,
                ClientRequest::Send { .. } | ClientRequest::BulkBegin { .. }
            ));
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Refused {
                        request,
                        code: RefusalCode::PolicyDenied,
                    },
                    0,
                ))
                .unwrap();
            {
                let mut future = core::pin::pin!(send.observe(&mut link, None));
                assert!(matches!(
                    future.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Refused(RefusalCode::PolicyDenied)))
                ));
            }
            assert_eq!(send.retired_refusal(), Some(RefusalCode::PolicyDenied));
            assert!(!send.pending_send());
            assert!(!send.failed());
            assert!(link.is_idle());
            {
                let mut future = core::pin::pin!(send.observe(&mut link, None));
                assert!(matches!(
                    future.as_mut().poll(&mut context),
                    Poll::Ready(Ok(false))
                ));
            }
            assert!(serve.req.pop().is_none(), "no phantom upload to abort");
            let next_request = RequestId(id(66));
            let mut next =
                InferenceSend::new(next_request, &inference_bytes(CLIENT_CHUNK_MAX)).unwrap();
            {
                let mut future = core::pin::pin!(next.observe(&mut link, None));
                assert!(future.as_mut().poll(&mut context).is_pending());
            }
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap())
                    .unwrap()
                    .request_id(),
                next_request
            );
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Accepted {
                        request: next_request,
                    },
                    0,
                ))
                .unwrap();
            let mut future = core::pin::pin!(next.observe(&mut link, None));
            assert!(matches!(
                future.as_mut().poll(&mut context),
                Poll::Ready(Ok(true))
            ));
        }
    }

    #[test]
    fn retained_inference_rejects_foreign_or_malformed_abort_testimony() {
        let request = RequestId(id(64));
        for (kind, payload) in [
            (kore_hostlink::BULK_ABORTED, &[7u8; 16][..]),
            (kore_hostlink::BULK_ABORTED, &[][..]),
            (kore_hostlink::BULK_READY, &inference_bulk_id(request).0[..]),
        ] {
            let mut send =
                InferenceSend::new(request, &inference_bytes(CLIENT_CHUNK_MAX + 1)).unwrap();
            send.stage = InferenceStage::AwaitAbort;
            send.refusal = Some(RefusalCode::RouteClosed);
            assert!(send.accept_abort(&ClientEvent::Control {
                kind,
                request,
                payload: Payload::copy_from(payload).unwrap()
            }));
            assert!(send.failed());
            assert!(send.retired_refusal().is_none());
        }
    }

    #[test]
    fn malformed_inference_is_refused_before_any_publication() {
        let mut malformed = inference_bytes(CLIENT_CHUNK_MAX + 1);
        malformed[0] ^= 1;
        for bytes in [
            &[][..],
            malformed.as_slice(),
            &[0; kore_hostlink::INLINE_MAX + 1][..],
        ] {
            let result = inference_exchange(RequestId(id(6)), bytes, |_| {
                panic!("malformed input touched the service")
            });
            assert!(matches!(result, Err(ClientError::InvalidPayload)));
        }
    }

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    #[test]
    fn poll_resume_recovers_one_original_event_without_another_dequeue() {
        let first = RequestId(id(31));
        let second = RequestId(id(32));
        for response in [
            ClientResponse::Event(ClientEvent::Control {
                kind: kore_hostlink::INFER_REPLY,
                request: first,
                payload: Payload::copy_from(b"original event").unwrap(),
            }),
            ClientResponse::Event(ClientEvent::Status(crate::StatusNotice {
                request: second,
                reason: crate::StatusReason::HostDisconnected,
            })),
            ClientResponse::Empty,
        ] {
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut poll = core::pin::pin!(link.poll_observation(false, None));
                assert!(poll.as_mut().poll(&mut context).is_pending());
            }
            assert_eq!(
                HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                Some(ClientRequest::Poll)
            );
            assert!(link.poll_request_pending());
            assert!(!link.route_request_pending());
            for _ in 0..2 {
                let mut resume = core::pin::pin!(link.poll_observation(true, None));
                assert!(resume.as_mut().poll(&mut context).is_pending());
                assert!(serve.req.pop().is_none());
            }
            serve
                .rep
                .push(HostLinkWire::encode_rep(&response, 0))
                .unwrap();
            {
                let mut duplicate = core::pin::pin!(link.poll_observation(false, None));
                assert!(matches!(
                    duplicate.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                ));
            }
            {
                let mut wrong = core::pin::pin!(link.observe_request(
                    ClientRequest::CloseAgent {
                        request: first,
                        slot: 0,
                        generation: TaskGeneration::new(1).unwrap(),
                    },
                    true,
                    None
                ));
                assert!(matches!(
                    wrong.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                ));
            }
            let recovered = {
                let mut resume = core::pin::pin!(link.poll_observation(true, None));
                match resume.as_mut().poll(&mut context) {
                    Poll::Ready(Ok(event)) => event,
                    other => panic!("unexpected poll continuation: {other:?}"),
                }
            };
            match response {
                ClientResponse::Event(event) => {
                    assert_eq!(recovered, Some(event.clone()));
                    let owner = crate::frontend_reply_owner(&event, [(0, first), (1, second)]);
                    assert_eq!(
                        owner,
                        Some(if matches!(event, ClientEvent::Status(_)) {
                            1
                        } else {
                            0
                        })
                    );
                }
                ClientResponse::Empty => assert!(recovered.is_none()),
                _ => unreachable!(),
            }
            assert!(!link.poll_request_pending());
            assert!(link.is_idle());
            let mut again = core::pin::pin!(link.poll_observation(true, None));
            assert!(matches!(
                again.as_mut().poll(&mut context),
                Poll::Ready(Err(ClientError::Call(CallError::NoOutstanding)))
            ));
            assert!(
                serve.req.pop().is_none(),
                "only one mailbox event was requested"
            );
        }
    }

    #[test]
    fn route_resume_matches_the_whole_operation_without_republishing() {
        let generation = TaskGeneration::new(u64::from(u32::MAX) + 7).unwrap();
        for rotate in [false, true] {
            let request = RequestId(id(17));
            let operation = if rotate {
                ClientRequest::RotateAgent {
                    request,
                    slot: 1,
                    generation,
                }
            } else {
                ClientRequest::CloseAgent {
                    request,
                    slot: 1,
                    generation,
                }
            };
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut first =
                    core::pin::pin!(link.observe_request(operation.clone(), false, None));
                assert!(first.as_mut().poll(&mut context).is_pending());
                assert_eq!(
                    HostLinkWire::decode_req(&serve.req.pop().unwrap()),
                    Some(operation.clone())
                );
            }
            assert!(!link.is_idle());
            assert!(link.route_request_pending());
            // Repeated observation can be abandoned too. It never sends the
            // closure/rotation again, even before a reply has arrived.
            for _ in 0..2 {
                let mut resume =
                    core::pin::pin!(link.observe_request(operation.clone(), true, None));
                assert!(resume.as_mut().poll(&mut context).is_pending());
                assert!(serve.req.pop().is_none());
            }
            serve
                .rep
                .push(HostLinkWire::encode_rep(
                    &ClientResponse::Accepted { request },
                    0,
                ))
                .unwrap();
            for mismatch in [
                ClientRequest::CloseAgent {
                    request: RequestId(id(18)),
                    slot: 1,
                    generation,
                },
                ClientRequest::CloseAgent {
                    request,
                    slot: 0,
                    generation,
                },
                ClientRequest::CloseAgent {
                    request,
                    slot: 1,
                    generation: TaskGeneration::new(7).unwrap(),
                },
                if rotate {
                    ClientRequest::CloseAgent {
                        request,
                        slot: 1,
                        generation,
                    }
                } else {
                    ClientRequest::RotateAgent {
                        request,
                        slot: 1,
                        generation,
                    }
                },
            ] {
                let mut wrong = core::pin::pin!(link.observe_request(mismatch, true, None));
                assert!(matches!(
                    wrong.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(CallError::Outstanding)))
                ));
                assert!(serve.req.pop().is_none());
            }
            {
                let mut resume =
                    core::pin::pin!(link.observe_request(operation.clone(), true, None));
                assert!(matches!(
                    resume.as_mut().poll(&mut context),
                    Poll::Ready(Ok(()))
                ));
            }
            assert!(link.is_idle());
            assert!(!link.route_request_pending());
            let mut duplicate = core::pin::pin!(link.observe_request(operation, true, None));
            assert!(matches!(
                duplicate.as_mut().poll(&mut context),
                Poll::Ready(Err(ClientError::Call(CallError::NoOutstanding)))
            ));
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn route_resume_requires_exact_acceptance_and_preserves_backpressure() {
        let request = RequestId(id(21));
        for response in [
            ClientResponse::Refused {
                request,
                code: RefusalCode::Backpressure,
            },
            ClientResponse::Accepted {
                request: RequestId(id(22)),
            },
            ClientResponse::Empty,
        ] {
            let operation = ClientRequest::CloseAgent {
                request,
                slot: 0,
                generation: TaskGeneration::new(9).unwrap(),
            };
            let mut window = WindowBytes([0; crate::HOSTLINK_PAIR_WINDOW_BYTES]);
            let (client, mut serve) = split_window::<HostLinkWire>(
                &mut window.0,
                crate::HOSTLINK_REQUEST_CAPACITY,
                crate::HOSTLINK_RESPONSE_CAPACITY,
            )
            .unwrap();
            let transport = AcceptKick;
            let mut link = RingHostLink::from_pair(
                SvcClient::new(
                    &transport,
                    client,
                    ClientEndpoints {
                        kick: SlotRef::from_wire(30),
                        grants: [None; kore_rt::service::MAX_GRANTS],
                    },
                )
                .with_direct_polling(),
            );
            let mut context = Context::from_waker(Waker::noop());
            {
                let mut start =
                    core::pin::pin!(link.observe_request(operation.clone(), false, None));
                assert!(start.as_mut().poll(&mut context).is_pending());
                serve.req.pop().unwrap();
            }
            serve
                .rep
                .push(HostLinkWire::encode_rep(&response, 0))
                .unwrap();
            {
                let mut resume =
                    core::pin::pin!(link.observe_request(operation.clone(), true, None));
                match resume.as_mut().poll(&mut context) {
                    Poll::Ready(Err(ClientError::Refused(RefusalCode::Backpressure))) => {
                        assert!(matches!(response, ClientResponse::Refused { .. }))
                    }
                    Poll::Ready(Err(ClientError::Call(CallError::Wire))) => {
                        assert!(!matches!(response, ClientResponse::Refused { .. }))
                    }
                    other => panic!("unexpected native route outcome: {other:?}"),
                }
            }
            if matches!(response, ClientResponse::Refused { .. }) {
                assert!(
                    link.is_idle(),
                    "transactional backpressure leaves the pair available"
                );
            } else {
                assert!(!link.is_idle(), "a malformed receipt poisons the pair");
                let mut retry = core::pin::pin!(link.observe_request(operation, true, None));
                assert!(matches!(
                    retry.as_mut().poll(&mut context),
                    Poll::Ready(Err(ClientError::Call(CallError::Poisoned)))
                ));
            }
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn upload_state_is_bounded_before_any_transport_call() {
        let bytes = [0u8; CLIENT_CHUNK_MAX + 1];
        assert!(Payload::copy_from(&bytes).is_none());

        let content = [7u8; 4];
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(2)),
            total_len: U32::new(content.len() as u32),
            chunk_count: U16::new(1),
            content_kind: U16::new(9),
            digest_blake3: *blake3::hash(&content).as_bytes(),
        };
        let upload = BulkUpload {
            request: RequestId(id(1)),
            reference,
            offset: 0,
            wire_bytes: 0,
            wire_chunks: 0,
            fragments: 0,
            hasher: blake3::Hasher::new(),
        };
        assert_eq!(upload.offset(), 0);
        assert_eq!(upload.chunks(), 0);
        assert_eq!(upload.fragments(), 0);
        assert_eq!(upload.bulk_id(), BulkId(id(2)));
    }

    #[test]
    fn status_bulk_transport_is_an_explicit_topology_attestation() {
        let serial = ClientStatus {
            phase: crate::ClientLinkPhase::Active,
            bulk_transport: crate::BulkTransport::Serial,
            queued_status: 0,
            queued_control: 0,
            queued_bulk: 0,
        };
        let shared = ClientStatus {
            bulk_transport: crate::BulkTransport::SharedMemory,
            ..serial
        };

        assert_eq!(serial.bulk_transport, crate::BulkTransport::Serial);
        assert_eq!(shared.bulk_transport, crate::BulkTransport::SharedMemory);
        assert_ne!(serial, shared);
    }
}
