//! Transport-neutral HostLink v1 message kinds and bounded payload codecs.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{BulkId, EnvelopeFields, EnvelopeV1, FAMILY_LINK, LinkEpoch, RequestId, RouteId};

pub const HELLO: u16 = 1;
pub const HELLO_ACK: u16 = 2;
pub const ROUTE_OPEN: u16 = 3;
pub const ROUTE_OPENED: u16 = 4;
pub const INFER_REQUEST: u16 = 5;
pub const INFER_REPLY: u16 = 6;
pub const CANCEL: u16 = 7;
pub const CANCELLED: u16 = 8;
pub const ROUTE_CLOSE: u16 = 9;
pub const ROUTE_CLOSED: u16 = 10;
/// A coalescible serial-control edge announcing shared-memory publication.
/// The bulk bytes remain exclusively in the shared window.
pub const SHMEM_WAKE: u16 = 11;
/// Typed request on the structurally dedicated workspace-importer route.
pub const WORKSPACE_REQUEST: u16 = 12;
/// Typed reply on the structurally dedicated workspace-importer route.
pub const WORKSPACE_REPLY: u16 = 13;
/// Broker-route-only post-commit authority observation for the host UI.
pub const AUTHORITY_REPORT: u16 = 14;
/// A complete KIR1 request carried through the existing credited bulk lane
/// when it exceeds the founded client's 512-byte inline record.
pub const INFERENCE_REQUEST_CONTENT: u16 = 0x4b49;
pub const BULK_BEGIN: u16 = 32;
pub const BULK_CHUNK: u16 = 33;
pub const BULK_FINISH: u16 = 34;
pub const BULK_ABORT: u16 = 35;
pub const BULK_READY: u16 = 36;
pub const BULK_ABORTED: u16 = 37;
pub const BULK_ACCEPTED: u16 = 38;
pub const REFUSED: u16 = 255;

pub const INFERENCE_REQUEST_MAGIC: u32 = u32::from_le_bytes(*b"KIR1");
pub const INFERENCE_EVENT_MAGIC: u32 = u32::from_le_bytes(*b"KIE1");
pub const INFERENCE_SCHEMA_V1: u16 = 1;
pub const INFERENCE_REQUEST_HEADER_LEN: usize = 32;
pub const INFERENCE_EVENT_HEADER_LEN: usize = 32;
/// A request rides the bounded HostLink control plane and may use the rest of
/// one inline frame. Broker catalogs are materially larger than streamed
/// model output, so request and event limits are deliberately distinct.
pub const INFERENCE_INPUT_MAX: usize = crate::INLINE_MAX - INFERENCE_REQUEST_HEADER_LEN;
/// Founded HostLink mailboxes carry 512-byte control payloads. Reserving the
/// event header here makes every streamed part deliverable without a hidden
/// service-specific truncation rule.
pub const INFERENCE_PART_MAX: usize = 512 - INFERENCE_EVENT_HEADER_LEN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum InferenceRequestKind {
    BrokerPlan = 1,
    AgentTurn = 2,
    /// Guest-initiated long poll for authenticated frontend intent.
    FrontendPoll = 3,
    /// Terminal result emitted by the capability-scoped founded agent.
    FrontendResult = 4,
    /// Outcome of a live update, separate from the prompt's terminal answer.
    FrontendUpdateResult = 5,
}

impl InferenceRequestKind {
    #[must_use]
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::BrokerPlan),
            2 => Some(Self::AgentTurn),
            3 => Some(Self::FrontendPoll),
            4 => Some(Self::FrontendResult),
            5 => Some(Self::FrontendUpdateResult),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum InferenceEventKind {
    AssistantPart = 1,
    TurnFinished = 2,
    /// The admitted inference request failed; the route and epoch remain live.
    /// Payload is a bounded RefusalView for FAMILY_LINK / INFER_REQUEST.
    TurnFailed = 3,
}

impl InferenceEventKind {
    #[must_use]
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::AssistantPart),
            2 => Some(Self::TurnFinished),
            3 => Some(Self::TurnFailed),
            _ => None,
        }
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct InferenceRequestHeaderV1 {
    magic: U32,
    version: U16,
    kind: u8,
    flags: u8,
    turn: U64,
    max_output_tokens: U32,
    timeout_ticks: U32,
    input_len: U32,
    reserved: U32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceRequestView<'a> {
    pub kind: InferenceRequestKind,
    pub turn: u64,
    pub max_output_tokens: u32,
    pub timeout_ticks: u32,
    pub input: &'a [u8],
}

impl InferenceRequestView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.turn == 0
            || self.max_output_tokens == 0
            || self.timeout_ticks == 0
            || self.input.is_empty()
            || self.input.len() > INFERENCE_INPUT_MAX
        {
            return None;
        }
        let length = INFERENCE_REQUEST_HEADER_LEN.checked_add(self.input.len())?;
        let output = output.get_mut(..length)?;
        let header = InferenceRequestHeaderV1 {
            magic: U32::new(INFERENCE_REQUEST_MAGIC),
            version: U16::new(INFERENCE_SCHEMA_V1),
            kind: self.kind as u8,
            flags: 0,
            turn: U64::new(self.turn),
            max_output_tokens: U32::new(self.max_output_tokens),
            timeout_ticks: U32::new(self.timeout_ticks),
            input_len: U32::new(self.input.len() as u32),
            reserved: U32::ZERO,
        };
        output[..INFERENCE_REQUEST_HEADER_LEN].copy_from_slice(header.as_bytes());
        output[INFERENCE_REQUEST_HEADER_LEN..].copy_from_slice(self.input);
        Some(length)
    }
}

#[must_use]
pub fn decode_inference_request(input: &[u8]) -> Option<InferenceRequestView<'_>> {
    let header =
        InferenceRequestHeaderV1::read_from_bytes(input.get(..INFERENCE_REQUEST_HEADER_LEN)?)
            .ok()?;
    let input_len = header.input_len.get() as usize;
    if header.magic.get() != INFERENCE_REQUEST_MAGIC
        || header.version.get() != INFERENCE_SCHEMA_V1
        || header.flags != 0
        || header.reserved.get() != 0
        || header.turn.get() == 0
        || header.max_output_tokens.get() == 0
        || header.timeout_ticks.get() == 0
        || input_len == 0
        || input_len > INFERENCE_INPUT_MAX
        || input.len() != INFERENCE_REQUEST_HEADER_LEN.checked_add(input_len)?
    {
        return None;
    }
    Some(InferenceRequestView {
        kind: InferenceRequestKind::from_wire(header.kind)?,
        turn: header.turn.get(),
        max_output_tokens: header.max_output_tokens.get(),
        timeout_ticks: header.timeout_ticks.get(),
        input: &input[INFERENCE_REQUEST_HEADER_LEN..],
    })
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct InferenceEventHeaderV1 {
    magic: U32,
    version: U16,
    kind: u8,
    flags: u8,
    part: U16,
    reserved0: U16,
    payload_len: U32,
    prompt_tokens: U32,
    completion_tokens: U32,
    turn: U64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceEventView<'a> {
    pub kind: InferenceEventKind,
    pub terminal: bool,
    pub part: u16,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub turn: u64,
    pub payload: &'a [u8],
}

impl InferenceEventView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.payload.len() > INFERENCE_PART_MAX
            || self.terminal
                != matches!(
                    self.kind,
                    InferenceEventKind::TurnFinished | InferenceEventKind::TurnFailed
                )
            || (self.kind == InferenceEventKind::TurnFailed
                && decode_inference_failure(self.payload).is_none())
        {
            return None;
        }
        let length = INFERENCE_EVENT_HEADER_LEN.checked_add(self.payload.len())?;
        let output = output.get_mut(..length)?;
        let header = InferenceEventHeaderV1 {
            magic: U32::new(INFERENCE_EVENT_MAGIC),
            version: U16::new(INFERENCE_SCHEMA_V1),
            kind: self.kind as u8,
            flags: u8::from(self.terminal),
            part: U16::new(self.part),
            reserved0: U16::ZERO,
            payload_len: U32::new(self.payload.len() as u32),
            prompt_tokens: U32::new(self.prompt_tokens),
            completion_tokens: U32::new(self.completion_tokens),
            turn: U64::new(self.turn),
        };
        output[..INFERENCE_EVENT_HEADER_LEN].copy_from_slice(header.as_bytes());
        output[INFERENCE_EVENT_HEADER_LEN..].copy_from_slice(self.payload);
        Some(length)
    }
}

#[must_use]
pub fn decode_inference_event(input: &[u8]) -> Option<InferenceEventView<'_>> {
    let header =
        InferenceEventHeaderV1::read_from_bytes(input.get(..INFERENCE_EVENT_HEADER_LEN)?).ok()?;
    let payload_len = header.payload_len.get() as usize;
    if header.magic.get() != INFERENCE_EVENT_MAGIC
        || header.version.get() != INFERENCE_SCHEMA_V1
        || header.flags & !1 != 0
        || header.reserved0.get() != 0
        || payload_len > INFERENCE_PART_MAX
        || input.len() != INFERENCE_EVENT_HEADER_LEN.checked_add(payload_len)?
    {
        return None;
    }
    let kind = InferenceEventKind::from_wire(header.kind)?;
    let terminal = header.flags & 1 != 0;
    if terminal
        != matches!(
            kind,
            InferenceEventKind::TurnFinished | InferenceEventKind::TurnFailed
        )
        || (kind == InferenceEventKind::TurnFailed
            && decode_inference_failure(&input[INFERENCE_EVENT_HEADER_LEN..]).is_none())
    {
        return None;
    }
    Some(InferenceEventView {
        kind,
        terminal,
        part: header.part.get(),
        prompt_tokens: header.prompt_tokens.get(),
        completion_tokens: header.completion_tokens.get(),
        turn: header.turn.get(),
        payload: &input[INFERENCE_EVENT_HEADER_LEN..],
    })
}

/// Decode the reason of an admitted inference failure, without treating a
/// protocol refusal for another operation as an inference terminal event.
pub fn decode_inference_failure(input: &[u8]) -> Option<RefusalView<'_>> {
    let refusal = decode_refusal(input)?;
    (refusal.for_family == FAMILY_LINK && refusal.for_kind == INFER_REQUEST).then_some(refusal)
}

#[cfg(test)]
mod inference_failure_tests {
    use super::*;
    #[test]
    fn terminal_inference_failure_requires_a_typed_reason_for_inference() {
        let reason = RefusalView {
            for_family: FAMILY_LINK,
            for_kind: INFER_REQUEST,
            code: RefusalCode::ModelFailed,
            retry: RetryClass::NewRequest,
            retry_after_ms: 0,
            safe_detail: &[],
        };
        let mut body = [0; REFUSAL_HEADER_LEN];
        let len = reason.encode(&mut body).unwrap();
        let event = InferenceEventView {
            kind: InferenceEventKind::TurnFailed,
            terminal: true,
            part: 2,
            prompt_tokens: 0,
            completion_tokens: 0,
            turn: 3,
            payload: &body[..len],
        };
        let mut wire = [0; INFERENCE_EVENT_HEADER_LEN + REFUSAL_HEADER_LEN];
        let len = event.encode(&mut wire).unwrap();
        let decoded = decode_inference_event(&wire[..len]).unwrap();
        assert_eq!(decoded.kind, InferenceEventKind::TurnFailed);
        assert_eq!(
            decode_inference_failure(decoded.payload).unwrap().code,
            RefusalCode::ModelFailed
        );
        assert!(
            InferenceEventView {
                terminal: false,
                ..event
            }
            .encode(&mut wire)
            .is_none()
        );
        assert!(
            InferenceEventView {
                payload: b"model text",
                ..event
            }
            .encode(&mut wire)
            .is_none()
        );
        let other = RefusalView {
            for_kind: WORKSPACE_REQUEST,
            ..reason
        };
        let len = other.encode(&mut body).unwrap();
        assert!(decode_inference_failure(&body[..len]).is_none());
    }
}

pub const SAFE_DETAIL_MAX: usize = 256;
pub const REFUSAL_HEADER_LEN: usize = 14;
pub const BULK_CHUNK_HEADER_LEN: usize = 52;
pub const BULK_FINISH_LEN: usize = 52;
pub const BULK_ABORT_LEN: usize = 18;
pub const SHMEM_WAKE_LEN: usize = 16;
pub const SHMEM_WAKE_VERSION: u16 = 1;
/// Frozen SPSC depth used by both endpoints of the HostLink bulk window.
pub const SHMEM_QUEUE_CAPACITY: u32 = 8;
/// One shared slot carries one complete frozen HostLink bulk frame.
pub const SHMEM_SLOT_PAYLOAD_BYTES: u32 = crate::wire::BULK_FRAME_MAX as u32;

const NEGOTIATED_TRANSCRIPT: &[u8] =
    b"hostlinkd:v1;control=65536;bulk_chunk=1048576;bulk_object=8388608;routes=33;workspace=1;bulk_per_route=4";

#[must_use]
pub const fn negotiated_transcript() -> &'static [u8] {
    NEGOTIATED_TRANSCRIPT
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct HelloV1 {
    pub host_nonce: [u8; 32],
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct HelloAckV1 {
    pub guest_boot: [u8; 32],
    pub epoch: LinkEpoch,
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct RouteOpenV1 {
    pub route_id: RouteId,
    pub task_generation: U64,
    pub role: u8,
    pub ordinal: u8,
}

/// Versioned, bounded serial wake after one or more shared publications.
///
/// `publication` is a per-link-epoch monotonic count. A producer may replace
/// an unsent wake with a later count; a receiver accepts an equal count as an
/// idempotent retry. The counter carries no routing authority: the enclosing
/// envelope must remain link-control (zero route/generation/sequence/request).
#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct ShmemWakeV1 {
    version: U16,
    flags: U16,
    reserved: U32,
    publication: U64,
}

impl ShmemWakeV1 {
    /// Construct one valid publication edge. Zero is reserved for reset.
    #[must_use]
    pub const fn new(publication: u64) -> Option<Self> {
        if publication == 0 {
            return None;
        }
        Some(Self {
            version: U16::new(SHMEM_WAKE_VERSION),
            flags: U16::ZERO,
            reserved: U32::ZERO,
            publication: U64::new(publication),
        })
    }

    /// Validate every frozen field and return the monotonic publication.
    #[must_use]
    pub fn publication(&self) -> Option<u64> {
        (self.version.get() == SHMEM_WAKE_VERSION
            && self.flags.get() == 0
            && self.reserved.get() == 0
            && self.publication.get() != 0)
            .then(|| self.publication.get())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct BulkChunkHeaderV1 {
    pub bulk_id: BulkId,
    pub offset: U32,
    pub digest_blake3: [u8; 32],
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct BulkFinishV1 {
    pub bulk_id: BulkId,
    pub final_total: U32,
    pub digest_blake3: [u8; 32],
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct BulkAbortV1 {
    pub bulk_id: BulkId,
    pub reason: U16,
}

const _: () = assert!(core::mem::size_of::<HelloV1>() == 32);
const _: () = assert!(core::mem::size_of::<HelloAckV1>() == 48);
const _: () = assert!(core::mem::size_of::<RouteOpenV1>() == 26);
const _: () =
    assert!(core::mem::size_of::<InferenceRequestHeaderV1>() == INFERENCE_REQUEST_HEADER_LEN);
const _: () = assert!(core::mem::size_of::<InferenceEventHeaderV1>() == INFERENCE_EVENT_HEADER_LEN);
const _: () = assert!(core::mem::size_of::<ShmemWakeV1>() == SHMEM_WAKE_LEN);
const _: () = assert!(core::mem::size_of::<BulkChunkHeaderV1>() == BULK_CHUNK_HEADER_LEN);
const _: () = assert!(core::mem::size_of::<BulkFinishV1>() == BULK_FINISH_LEN);
const _: () = assert!(core::mem::size_of::<BulkAbortV1>() == BULK_ABORT_LEN);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum RefusalCode {
    UnsupportedVersion = 0,
    UnsupportedFeature = 1,
    UnknownKind = 2,
    Malformed = 3,
    InvalidEncoding = 4,
    InvalidLength = 5,
    TooLarge = 6,
    TooMany = 7,
    ArithmeticOverflow = 8,
    EpochStale = 9,
    RouteUnknown = 10,
    RouteClosed = 11,
    GenerationStale = 12,
    Duplicate = 13,
    OutOfOrder = 14,
    Backpressure = 15,
    CapacityExhausted = 16,
    SelectorUnknown = 17,
    SelectorExpired = 18,
    NotVisible = 19,
    RightsNotSubset = 20,
    BudgetExceeded = 21,
    ToolUnavailable = 22,
    ToolSchemaMismatch = 23,
    VersionConflict = 24,
    Cancelled = 25,
    TimedOut = 26,
    HostDisconnected = 27,
    ModelFailed = 28,
    WorkerFailed = 29,
    StorageFailed = 30,
    PolicyDenied = 31,
    Internal = 32,
}

impl RefusalCode {
    #[must_use]
    pub fn from_wire(value: u16) -> Option<Self> {
        macro_rules! codes {
            ($($name:ident),+ $(,)?) => { match value { $(x if x == Self::$name as u16 => Some(Self::$name),)+ _ => None } };
        }
        codes!(
            UnsupportedVersion,
            UnsupportedFeature,
            UnknownKind,
            Malformed,
            InvalidEncoding,
            InvalidLength,
            TooLarge,
            TooMany,
            ArithmeticOverflow,
            EpochStale,
            RouteUnknown,
            RouteClosed,
            GenerationStale,
            Duplicate,
            OutOfOrder,
            Backpressure,
            CapacityExhausted,
            SelectorUnknown,
            SelectorExpired,
            NotVisible,
            RightsNotSubset,
            BudgetExceeded,
            ToolUnavailable,
            ToolSchemaMismatch,
            VersionConflict,
            Cancelled,
            TimedOut,
            HostDisconnected,
            ModelFailed,
            WorkerFailed,
            StorageFailed,
            PolicyDenied,
            Internal,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RetryClass {
    Never = 0,
    SameEpoch = 1,
    NewRequest = 2,
    AfterReconnect = 3,
}

const _: () = assert!(RefusalCode::UnsupportedVersion as u16 == 0);
const _: () = assert!(RefusalCode::Internal as u16 == 32);
const _: () = assert!(RetryClass::Never as u8 == 0);
const _: () = assert!(RetryClass::AfterReconnect as u8 == 3);

/// Validate the link-control envelope surrounding [`ShmemWakeV1`].
#[must_use]
pub fn shmem_wake_envelope_is_valid(envelope: &EnvelopeV1, epoch: LinkEpoch) -> bool {
    envelope.family.get() == FAMILY_LINK
        && envelope.schema_minor.get() == 0
        && envelope.kind.get() == SHMEM_WAKE
        && !epoch.is_zero()
        && envelope.link_epoch == epoch
        && envelope.route_id.is_zero()
        && envelope.task_generation.get() == 0
        && envelope.sequence.get() == 0
        && envelope.request_id.is_zero()
        && envelope.payload_len.get() as usize == SHMEM_WAKE_LEN
}

impl RetryClass {
    #[must_use]
    pub fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Never),
            1 => Some(Self::SameEpoch),
            2 => Some(Self::NewRequest),
            3 => Some(Self::AfterReconnect),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefusalView<'a> {
    pub for_family: u16,
    pub for_kind: u16,
    pub code: RefusalCode,
    pub retry: RetryClass,
    pub retry_after_ms: u32,
    pub safe_detail: &'a [u8],
}

impl RefusalView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.safe_detail.len() > SAFE_DETAIL_MAX {
            return None;
        }
        let len = REFUSAL_HEADER_LEN.checked_add(self.safe_detail.len())?;
        let out = output.get_mut(..len)?;
        out[0..2].copy_from_slice(&self.for_family.to_le_bytes());
        out[2..4].copy_from_slice(&self.for_kind.to_le_bytes());
        out[4..6].copy_from_slice(&(self.code as u16).to_le_bytes());
        out[6] = self.retry as u8;
        out[7] = 0;
        out[8..12].copy_from_slice(&self.retry_after_ms.to_le_bytes());
        out[12..14].copy_from_slice(&(self.safe_detail.len() as u16).to_le_bytes());
        out[14..].copy_from_slice(self.safe_detail);
        Some(len)
    }
}

pub fn decode_refusal(input: &[u8]) -> Option<RefusalView<'_>> {
    let header = input.get(..REFUSAL_HEADER_LEN)?;
    if header[7] != 0 {
        return None;
    }
    let detail_len = u16::from_le_bytes(header[12..14].try_into().ok()?) as usize;
    if detail_len > SAFE_DETAIL_MAX || input.len() != REFUSAL_HEADER_LEN + detail_len {
        return None;
    }
    Some(RefusalView {
        for_family: u16::from_le_bytes(header[0..2].try_into().ok()?),
        for_kind: u16::from_le_bytes(header[2..4].try_into().ok()?),
        code: RefusalCode::from_wire(u16::from_le_bytes(header[4..6].try_into().ok()?))?,
        retry: RetryClass::from_wire(header[6])?,
        retry_after_ms: u32::from_le_bytes(header[8..12].try_into().ok()?),
        safe_detail: &input[REFUSAL_HEADER_LEN..],
    })
}

#[must_use]
pub fn link_envelope(
    kind: u16,
    epoch: LinkEpoch,
    route: RouteId,
    generation: u64,
    sequence: u64,
    request: RequestId,
    payload_len: usize,
) -> EnvelopeV1 {
    EnvelopeV1::new(EnvelopeFields {
        family: FAMILY_LINK,
        schema_minor: 0,
        kind,
        epoch,
        route,
        generation,
        sequence,
        request,
        payload_len: payload_len as u32,
    })
}
