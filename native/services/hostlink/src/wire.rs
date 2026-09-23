//! Frozen founded-service wire for the four structurally assigned HostLink
//! client pairs.
//!
//! Endpoint identity is intentionally absent. The serving host stamps the
//! arrival pair as `Serve::via`; no record can select the broker, a dynamic
//! agent generation, or the filesystem/importer workspace face by payload.

use heapless::Vec;
use kore_hostlink::{BulkId, BulkRefV1, RefusalCode, RequestId, TaskGeneration};
use kore_rt::svc::{SvcProtocol, window_bytes};
use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::StatusReason;

pub const CLIENT_CHUNK_MAX: usize = 512;
/// The founded ring fragments bulk payloads at 512 bytes, while one HostLink
/// wire chunk aggregates exactly 4 KiB. This keeps a 1 MiB benchmark object at
/// the frozen protocol maximum of 256 chunks without granting an allocator.
pub use kore_hostlink::FOUNDED_BULK_WIRE_CHUNK;
pub const HOSTLINK_REQUEST_CAPACITY: u32 = 2;
pub const HOSTLINK_RESPONSE_CAPACITY: u32 = 2;
/// Trusted image-build selection, never controlled by a host/model message.
/// Keeping it in the native owner makes the founder and every user image use
/// the same environment-tracked dependency. The default is ordinary boot.
pub const SUPERVISOR_AUTOSTART: bool = match option_env!("KORE_AGENT_SUPERVISOR") {
    None => false,
    Some(value) => match value.as_bytes() {
        [b'0'] => false,
        [b'1'] => true,
        _ => panic!("KORE_AGENT_SUPERVISOR must be 0 or 1"),
    },
};

pub const HOSTLINK_OWNER_PAIR: usize = 4;
pub const HOSTLINK_SUPERVISOR_WORKERS: usize = 2;
pub const HOSTLINK_SERVE_PAIRS: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SupervisorOperation {
    Observe = 1,
    Close = 2,
    Open = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SupervisorPhase {
    Available = 1,
    Closing = 2,
    Closed = 3,
    Opening = 4,
    RestartRequested = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SupervisorState {
    pub phase: SupervisorPhase,
    pub incarnation: TaskGeneration,
    pub workers: [TaskGeneration; HOSTLINK_SUPERVISOR_WORKERS],
}

/// One bounded inline fragment. Larger values must use repeated bulk chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Payload(Vec<u8, CLIENT_CHUNK_MAX>);

impl Payload {
    pub fn copy_from(bytes: &[u8]) -> Option<Self> {
        Vec::from_slice(bytes).ok().map(Self)
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientRequest {
    /// Available only on the separately endowed surviving owner's pair.
    /// Observe uses generation zero; Close/Open name an exact incarnation.
    Supervisor {
        request: RequestId,
        operation: SupervisorOperation,
        generation: u64,
    },
    Send {
        kind: u16,
        request: RequestId,
        payload: Payload,
    },
    Cancel {
        request: RequestId,
    },
    Poll,
    Status,
    BulkBegin {
        request: RequestId,
        reference: BulkRefV1,
    },
    BulkChunk {
        request: RequestId,
        bulk_id: BulkId,
        bytes: Payload,
    },
    BulkFinish {
        request: RequestId,
        bulk_id: BulkId,
        total_len: u32,
        digest_blake3: [u8; 32],
    },
    BulkAbort {
        request: RequestId,
        bulk_id: BulkId,
        reason: RefusalCode,
    },
    /// Broker-pair-only control: close the currently bound agent generation,
    /// drain its mailbox, and install the requested successor on the same
    /// structurally assigned physical pair.
    RotateAgent {
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
    },
    /// Broker-pair-only terminal close. Capacity remains unavailable until
    /// the host has acknowledged the old generation and its mailbox drains.
    CloseAgent {
        request: RequestId,
        slot: u8,
        generation: TaskGeneration,
    },
}

impl ClientRequest {
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Supervisor { request, .. }
            | Self::Send { request, .. }
            | Self::Cancel { request }
            | Self::BulkBegin { request, .. }
            | Self::BulkChunk { request, .. }
            | Self::BulkFinish { request, .. }
            | Self::BulkAbort { request, .. }
            | Self::RotateAgent { request, .. }
            | Self::CloseAgent { request, .. } => *request,
            Self::Poll | Self::Status => RequestId::ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ClientLinkPhase {
    Disconnected = 0,
    Negotiating = 1,
    Opening = 2,
    Active = 3,
}

impl ClientLinkPhase {
    fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Disconnected),
            1 => Some(Self::Negotiating),
            2 => Some(Self::Opening),
            3 => Some(Self::Active),
            _ => None,
        }
    }
}

/// Authority-neutral attestation of the bulk data plane selected by the
/// single runtime-selecting HostLink service image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BulkTransport {
    Serial = 1,
    SharedMemory = 2,
}

impl BulkTransport {
    fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Serial),
            2 => Some(Self::SharedMemory),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClientStatus {
    pub phase: ClientLinkPhase,
    pub bulk_transport: BulkTransport,
    pub queued_status: u16,
    pub queued_control: u16,
    pub queued_bulk: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatusNotice {
    pub request: RequestId,
    pub reason: StatusReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientEvent {
    Control {
        kind: u16,
        request: RequestId,
        payload: Payload,
    },
    Cancel {
        request: RequestId,
    },
    BulkBegin {
        request: RequestId,
        reference: BulkRefV1,
    },
    BulkChunk {
        request: RequestId,
        bulk_id: BulkId,
        offset: u32,
        bytes: Payload,
    },
    BulkReady {
        request: RequestId,
        bulk_id: BulkId,
    },
    BulkAbort {
        request: RequestId,
        bulk_id: BulkId,
        reason: RefusalCode,
    },
    Status(StatusNotice),
}

#[derive(Clone, Debug, Eq, PartialEq)]
// Fixed inline storage is the no-allocator transport contract; boxing would
// turn a bounded wire event into hidden global-allocation authority.
#[allow(clippy::large_enum_variant)]
pub enum ClientResponse {
    Supervisor {
        request: RequestId,
        state: SupervisorState,
    },
    Accepted {
        request: RequestId,
    },
    Empty,
    Event(ClientEvent),
    Status(ClientStatus),
    Refused {
        request: RequestId,
        code: RefusalCode,
    },
}

/// Fixed request record. All discriminants are explicit and append-only.
#[derive(Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq)]
#[repr(C)]
pub struct ReqWire {
    pub op: u8,
    pub flags: u8,
    pub reserved0: U16,
    pub kind: U16,
    pub aux: U16,
    pub request: RequestId,
    pub bulk_id: BulkId,
    pub offset: U32,
    pub total_len: U32,
    pub len: U16,
    pub reserved1: U16,
    pub digest_blake3: [u8; 32],
    pub data: [u8; CLIENT_CHUNK_MAX],
}

impl ReqWire {
    pub const OP_SEND: u8 = 1;
    pub const OP_CANCEL: u8 = 2;
    pub const OP_POLL: u8 = 3;
    pub const OP_STATUS: u8 = 4;
    pub const OP_BULK_BEGIN: u8 = 5;
    pub const OP_BULK_CHUNK: u8 = 6;
    pub const OP_BULK_FINISH: u8 = 7;
    pub const OP_BULK_ABORT: u8 = 8;
    pub const OP_ROTATE_AGENT: u8 = 9;
    pub const OP_CLOSE_AGENT: u8 = 10;
    pub const OP_SUPERVISOR: u8 = 11;
}

/// Fixed reply record. The arrival pair remains the sole endpoint identity.
#[derive(Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq)]
#[repr(C)]
pub struct RepWire {
    pub tag: u8,
    pub variant: u8,
    pub code: U16,
    pub kind: U16,
    pub aux: U16,
    pub request: RequestId,
    pub bulk_id: BulkId,
    pub offset: U32,
    pub total_len: U32,
    pub len: U16,
    pub reserved: U16,
    pub digest_blake3: [u8; 32],
    pub data: [u8; CLIENT_CHUNK_MAX],
}

impl RepWire {
    pub const TAG_ACCEPTED: u8 = 1;
    pub const TAG_EMPTY: u8 = 2;
    pub const TAG_EVENT: u8 = 3;
    pub const TAG_STATUS: u8 = 4;
    pub const TAG_REFUSED: u8 = 5;
    pub const TAG_SUPERVISOR: u8 = 6;

    pub const EVENT_CONTROL: u8 = 1;
    pub const EVENT_CANCEL: u8 = 2;
    pub const EVENT_BULK_BEGIN: u8 = 3;
    pub const EVENT_BULK_CHUNK: u8 = 4;
    pub const EVENT_BULK_READY: u8 = 5;
    pub const EVENT_BULK_ABORT: u8 = 6;
    pub const EVENT_STATUS: u8 = 7;

    pub const STATUS_DISCONNECTED: u16 = 1;
    pub const STATUS_CANCELLED: u16 = 2;
    pub const STATUS_REFUSED: u16 = 3;

    /// Version of the status-only metadata stored in `reserved`.
    pub const STATUS_META_VERSION: u8 = 1;
}

pub struct HostLinkWire;

pub const HOSTLINK_PAIR_WINDOW_BYTES: usize =
    window_bytes::<HostLinkWire>(HOSTLINK_REQUEST_CAPACITY, HOSTLINK_RESPONSE_CAPACITY);

const _: () = {
    assert!(core::mem::size_of::<ReqWire>() == 596);
    assert!(core::mem::size_of::<RepWire>() == 596);
    assert!(ReqWire::OP_SEND == 1);
    assert!(ReqWire::OP_BULK_ABORT == 8);
    assert!(RepWire::TAG_ACCEPTED == 1);
    assert!(RepWire::TAG_REFUSED == 5);
    assert!(RepWire::EVENT_CONTROL == 1);
    assert!(RepWire::EVENT_STATUS == 7);
    assert!(ClientLinkPhase::Disconnected as u8 == 0);
    assert!(ClientLinkPhase::Active as u8 == 3);
    assert!(BulkTransport::Serial as u8 == 1);
    assert!(BulkTransport::SharedMemory as u8 == 2);
    assert!(RepWire::STATUS_META_VERSION == 1);
    assert!(HOSTLINK_REQUEST_CAPACITY.is_power_of_two());
    assert!(HOSTLINK_RESPONSE_CAPACITY.is_power_of_two());
    assert!(HOSTLINK_PAIR_WINDOW_BYTES <= 4096);
    assert!(HOSTLINK_SERVE_PAIRS == 5);
};

fn empty_req(op: u8) -> ReqWire {
    ReqWire {
        op,
        flags: 0,
        reserved0: U16::ZERO,
        kind: U16::ZERO,
        aux: U16::ZERO,
        request: RequestId::ZERO,
        bulk_id: BulkId::ZERO,
        offset: U32::ZERO,
        total_len: U32::ZERO,
        len: U16::ZERO,
        reserved1: U16::ZERO,
        digest_blake3: [0; 32],
        data: [0; CLIENT_CHUNK_MAX],
    }
}

fn empty_rep(tag: u8) -> RepWire {
    RepWire {
        tag,
        variant: 0,
        code: U16::ZERO,
        kind: U16::ZERO,
        aux: U16::ZERO,
        request: RequestId::ZERO,
        bulk_id: BulkId::ZERO,
        offset: U32::ZERO,
        total_len: U32::ZERO,
        len: U16::ZERO,
        reserved: U16::ZERO,
        digest_blake3: [0; 32],
        data: [0; CLIENT_CHUNK_MAX],
    }
}

fn put_payload(dst: &mut [u8; CLIENT_CHUNK_MAX], len: &mut U16, payload: &Payload) {
    dst[..payload.len()].copy_from_slice(payload.as_slice());
    *len = U16::new(payload.len() as u16);
}

pub fn encode_req(request: &ClientRequest) -> ReqWire {
    let mut wire = match request {
        ClientRequest::Supervisor { .. } => empty_req(ReqWire::OP_SUPERVISOR),
        ClientRequest::Send { .. } => empty_req(ReqWire::OP_SEND),
        ClientRequest::Cancel { .. } => empty_req(ReqWire::OP_CANCEL),
        ClientRequest::Poll => empty_req(ReqWire::OP_POLL),
        ClientRequest::Status => empty_req(ReqWire::OP_STATUS),
        ClientRequest::BulkBegin { .. } => empty_req(ReqWire::OP_BULK_BEGIN),
        ClientRequest::BulkChunk { .. } => empty_req(ReqWire::OP_BULK_CHUNK),
        ClientRequest::BulkFinish { .. } => empty_req(ReqWire::OP_BULK_FINISH),
        ClientRequest::BulkAbort { .. } => empty_req(ReqWire::OP_BULK_ABORT),
        ClientRequest::RotateAgent { .. } => empty_req(ReqWire::OP_ROTATE_AGENT),
        ClientRequest::CloseAgent { .. } => empty_req(ReqWire::OP_CLOSE_AGENT),
    };
    match request {
        ClientRequest::Supervisor {
            request,
            operation,
            generation,
        } => {
            wire.request = *request;
            wire.kind = U16::new(*operation as u16);
            wire.offset = U32::new(*generation as u32);
            wire.total_len = U32::new((*generation >> 32) as u32);
        }
        ClientRequest::Send {
            kind,
            request,
            payload,
        } => {
            wire.request = *request;
            wire.kind = U16::new(*kind);
            put_payload(&mut wire.data, &mut wire.len, payload);
        }
        ClientRequest::Cancel { request } => wire.request = *request,
        ClientRequest::Poll | ClientRequest::Status => {}
        ClientRequest::BulkBegin { request, reference } => {
            wire.request = *request;
            wire.bulk_id = reference.bulk_id;
            wire.total_len = reference.total_len;
            wire.aux = reference.chunk_count;
            wire.kind = reference.content_kind;
            wire.digest_blake3 = reference.digest_blake3;
        }
        ClientRequest::BulkChunk {
            request,
            bulk_id,
            bytes,
        } => {
            wire.request = *request;
            wire.bulk_id = *bulk_id;
            put_payload(&mut wire.data, &mut wire.len, bytes);
        }
        ClientRequest::BulkFinish {
            request,
            bulk_id,
            total_len,
            digest_blake3,
        } => {
            wire.request = *request;
            wire.bulk_id = *bulk_id;
            wire.total_len = U32::new(*total_len);
            wire.digest_blake3 = *digest_blake3;
        }
        ClientRequest::BulkAbort {
            request,
            bulk_id,
            reason,
        } => {
            wire.request = *request;
            wire.bulk_id = *bulk_id;
            wire.aux = U16::new(*reason as u16);
        }
        ClientRequest::RotateAgent {
            request,
            slot,
            generation,
        }
        | ClientRequest::CloseAgent {
            request,
            slot,
            generation,
        } => {
            wire.request = *request;
            wire.aux = U16::new(u16::from(*slot));
            let [lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3] = generation.get().to_le_bytes();
            wire.offset = U32::new(u32::from_le_bytes([lo0, lo1, lo2, lo3]));
            wire.total_len = U32::new(u32::from_le_bytes([hi0, hi1, hi2, hi3]));
        }
    }
    wire
}

pub fn decode_req(wire: &ReqWire) -> Option<ClientRequest> {
    let len = usize::from(wire.len.get());
    if len > CLIENT_CHUNK_MAX {
        return None;
    }
    let payload = || Payload::copy_from(&wire.data[..len]);
    let request = match wire.op {
        ReqWire::OP_SUPERVISOR => {
            let operation = match wire.kind.get() {
                1 => SupervisorOperation::Observe,
                2 => SupervisorOperation::Close,
                3 => SupervisorOperation::Open,
                _ => return None,
            };
            let generation = u64::from(wire.offset.get()) | (u64::from(wire.total_len.get()) << 32);
            if (operation == SupervisorOperation::Observe) != (generation == 0) {
                return None;
            }
            ClientRequest::Supervisor {
                request: wire.request,
                operation,
                generation,
            }
        }
        ReqWire::OP_SEND => ClientRequest::Send {
            kind: wire.kind.get(),
            request: wire.request,
            payload: payload()?,
        },
        ReqWire::OP_CANCEL => ClientRequest::Cancel {
            request: wire.request,
        },
        ReqWire::OP_POLL => ClientRequest::Poll,
        ReqWire::OP_STATUS => ClientRequest::Status,
        ReqWire::OP_BULK_BEGIN => ClientRequest::BulkBegin {
            request: wire.request,
            reference: BulkRefV1 {
                bulk_id: wire.bulk_id,
                total_len: wire.total_len,
                chunk_count: wire.aux,
                content_kind: wire.kind,
                digest_blake3: wire.digest_blake3,
            },
        },
        ReqWire::OP_BULK_CHUNK => ClientRequest::BulkChunk {
            request: wire.request,
            bulk_id: wire.bulk_id,
            bytes: payload()?,
        },
        ReqWire::OP_BULK_FINISH => ClientRequest::BulkFinish {
            request: wire.request,
            bulk_id: wire.bulk_id,
            total_len: wire.total_len.get(),
            digest_blake3: wire.digest_blake3,
        },
        ReqWire::OP_BULK_ABORT => ClientRequest::BulkAbort {
            request: wire.request,
            bulk_id: wire.bulk_id,
            reason: RefusalCode::from_wire(wire.aux.get())?,
        },
        ReqWire::OP_ROTATE_AGENT | ReqWire::OP_CLOSE_AGENT => {
            let slot = u8::try_from(wire.aux.get()).ok()?;
            let mut generation = [0_u8; 8];
            generation[..4].copy_from_slice(&wire.offset.get().to_le_bytes());
            generation[4..].copy_from_slice(&wire.total_len.get().to_le_bytes());
            let generation = TaskGeneration::new(u64::from_le_bytes(generation))?;
            if wire.op == ReqWire::OP_ROTATE_AGENT {
                ClientRequest::RotateAgent {
                    request: wire.request,
                    slot,
                    generation,
                }
            } else {
                ClientRequest::CloseAgent {
                    request: wire.request,
                    slot,
                    generation,
                }
            }
        }
        _ => return None,
    };
    (encode_req(&request) == *wire).then_some(request)
}

fn status_reason_wire(reason: StatusReason) -> (u16, u16) {
    match reason {
        StatusReason::HostDisconnected => (RepWire::STATUS_DISCONNECTED, 0),
        StatusReason::Cancelled => (RepWire::STATUS_CANCELLED, 0),
        StatusReason::Refused(code) => (RepWire::STATUS_REFUSED, code as u16),
    }
}

fn decode_status_reason(kind: u16, code: u16) -> Option<StatusReason> {
    match kind {
        RepWire::STATUS_DISCONNECTED if code == 0 => Some(StatusReason::HostDisconnected),
        RepWire::STATUS_CANCELLED if code == 0 => Some(StatusReason::Cancelled),
        RepWire::STATUS_REFUSED => Some(StatusReason::Refused(RefusalCode::from_wire(code)?)),
        _ => None,
    }
}

const fn status_meta(transport: BulkTransport) -> u16 {
    u16::from_le_bytes([RepWire::STATUS_META_VERSION, transport as u8])
}

fn decode_status_meta(meta: u16) -> Option<BulkTransport> {
    let [version, transport] = meta.to_le_bytes();
    if version != RepWire::STATUS_META_VERSION {
        return None;
    }
    BulkTransport::from_wire(transport)
}

pub fn encode_rep(response: &ClientResponse) -> RepWire {
    let mut wire = match response {
        ClientResponse::Supervisor { .. } => empty_rep(RepWire::TAG_SUPERVISOR),
        ClientResponse::Accepted { .. } => empty_rep(RepWire::TAG_ACCEPTED),
        ClientResponse::Empty => empty_rep(RepWire::TAG_EMPTY),
        ClientResponse::Event(_) => empty_rep(RepWire::TAG_EVENT),
        ClientResponse::Status(_) => empty_rep(RepWire::TAG_STATUS),
        ClientResponse::Refused { .. } => empty_rep(RepWire::TAG_REFUSED),
    };
    match response {
        ClientResponse::Supervisor { request, state } => {
            wire.request = *request;
            wire.variant = state.phase as u8;
            wire.kind = U16::new(1); // supervisor protocol version
            wire.len = U16::new((8 * (1 + HOSTLINK_SUPERVISOR_WORKERS)) as u16);
            wire.data[..8].copy_from_slice(&state.incarnation.get().to_le_bytes());
            for (row, generation) in wire.data[8..usize::from(wire.len.get())]
                .chunks_exact_mut(8)
                .zip(state.workers)
            {
                row.copy_from_slice(&generation.get().to_le_bytes());
            }
        }
        ClientResponse::Accepted { request } => wire.request = *request,
        ClientResponse::Empty => {}
        ClientResponse::Status(status) => {
            wire.variant = status.phase as u8;
            wire.kind = U16::new(status.queued_status);
            wire.aux = U16::new(status.queued_control);
            wire.offset = U32::new(u32::from(status.queued_bulk));
            wire.reserved = U16::new(status_meta(status.bulk_transport));
        }
        ClientResponse::Refused { request, code } => {
            wire.request = *request;
            wire.code = U16::new(*code as u16);
        }
        ClientResponse::Event(event) => match event {
            ClientEvent::Control {
                kind,
                request,
                payload,
            } => {
                wire.variant = RepWire::EVENT_CONTROL;
                wire.kind = U16::new(*kind);
                wire.request = *request;
                put_payload(&mut wire.data, &mut wire.len, payload);
            }
            ClientEvent::Cancel { request } => {
                wire.variant = RepWire::EVENT_CANCEL;
                wire.request = *request;
            }
            ClientEvent::BulkBegin { request, reference } => {
                wire.variant = RepWire::EVENT_BULK_BEGIN;
                wire.request = *request;
                wire.bulk_id = reference.bulk_id;
                wire.total_len = reference.total_len;
                wire.aux = reference.chunk_count;
                wire.kind = reference.content_kind;
                wire.digest_blake3 = reference.digest_blake3;
            }
            ClientEvent::BulkChunk {
                request,
                bulk_id,
                offset,
                bytes,
            } => {
                wire.variant = RepWire::EVENT_BULK_CHUNK;
                wire.request = *request;
                wire.bulk_id = *bulk_id;
                wire.offset = U32::new(*offset);
                put_payload(&mut wire.data, &mut wire.len, bytes);
            }
            ClientEvent::BulkReady { request, bulk_id } => {
                wire.variant = RepWire::EVENT_BULK_READY;
                wire.request = *request;
                wire.bulk_id = *bulk_id;
            }
            ClientEvent::BulkAbort {
                request,
                bulk_id,
                reason,
            } => {
                wire.variant = RepWire::EVENT_BULK_ABORT;
                wire.request = *request;
                wire.bulk_id = *bulk_id;
                wire.code = U16::new(*reason as u16);
            }
            ClientEvent::Status(status) => {
                wire.variant = RepWire::EVENT_STATUS;
                wire.request = status.request;
                let (kind, code) = status_reason_wire(status.reason);
                wire.kind = U16::new(kind);
                wire.code = U16::new(code);
            }
        },
    }
    wire
}

pub fn decode_rep(wire: &RepWire) -> Option<ClientResponse> {
    let len = usize::from(wire.len.get());
    if len > CLIENT_CHUNK_MAX {
        return None;
    }
    let payload = || Payload::copy_from(&wire.data[..len]);
    let response = match wire.tag {
        RepWire::TAG_SUPERVISOR => {
            if wire.kind.get() != 1 || len != 8 * (1 + HOSTLINK_SUPERVISOR_WORKERS) {
                return None;
            }
            let phase = match wire.variant {
                1 => SupervisorPhase::Available,
                2 => SupervisorPhase::Closing,
                3 => SupervisorPhase::Closed,
                4 => SupervisorPhase::Opening,
                5 => SupervisorPhase::RestartRequested,
                _ => return None,
            };
            let incarnation =
                TaskGeneration::new(u64::from_le_bytes(wire.data[..8].try_into().ok()?))?;
            let mut workers = [incarnation; HOSTLINK_SUPERVISOR_WORKERS];
            for (generation, row) in workers.iter_mut().zip(wire.data[8..len].chunks_exact(8)) {
                *generation = TaskGeneration::new(u64::from_le_bytes(row.try_into().ok()?))?;
            }
            ClientResponse::Supervisor {
                request: wire.request,
                state: SupervisorState {
                    phase,
                    incarnation,
                    workers,
                },
            }
        }
        RepWire::TAG_ACCEPTED => ClientResponse::Accepted {
            request: wire.request,
        },
        RepWire::TAG_EMPTY => ClientResponse::Empty,
        RepWire::TAG_STATUS => ClientResponse::Status(ClientStatus {
            phase: ClientLinkPhase::from_wire(wire.variant)?,
            bulk_transport: decode_status_meta(wire.reserved.get())?,
            queued_status: wire.kind.get(),
            queued_control: wire.aux.get(),
            queued_bulk: u16::try_from(wire.offset.get()).ok()?,
        }),
        RepWire::TAG_REFUSED => ClientResponse::Refused {
            request: wire.request,
            code: RefusalCode::from_wire(wire.code.get())?,
        },
        RepWire::TAG_EVENT => ClientResponse::Event(match wire.variant {
            RepWire::EVENT_CONTROL => ClientEvent::Control {
                kind: wire.kind.get(),
                request: wire.request,
                payload: payload()?,
            },
            RepWire::EVENT_CANCEL => ClientEvent::Cancel {
                request: wire.request,
            },
            RepWire::EVENT_BULK_BEGIN => ClientEvent::BulkBegin {
                request: wire.request,
                reference: BulkRefV1 {
                    bulk_id: wire.bulk_id,
                    total_len: wire.total_len,
                    chunk_count: wire.aux,
                    content_kind: wire.kind,
                    digest_blake3: wire.digest_blake3,
                },
            },
            RepWire::EVENT_BULK_CHUNK => ClientEvent::BulkChunk {
                request: wire.request,
                bulk_id: wire.bulk_id,
                offset: wire.offset.get(),
                bytes: payload()?,
            },
            RepWire::EVENT_BULK_READY => ClientEvent::BulkReady {
                request: wire.request,
                bulk_id: wire.bulk_id,
            },
            RepWire::EVENT_BULK_ABORT => ClientEvent::BulkAbort {
                request: wire.request,
                bulk_id: wire.bulk_id,
                reason: RefusalCode::from_wire(wire.code.get())?,
            },
            RepWire::EVENT_STATUS => ClientEvent::Status(StatusNotice {
                request: wire.request,
                reason: decode_status_reason(wire.kind.get(), wire.code.get())?,
            }),
            _ => return None,
        }),
        _ => return None,
    };
    (encode_rep(&response) == *wire).then_some(response)
}

impl SvcProtocol for HostLinkWire {
    type Request = ClientRequest;
    type Response = ClientResponse;
    type ReqWire = ReqWire;
    type RepWire = RepWire;

    const REPLY_GRANTS: usize = 0;
    const REQUEST_CAPACITY: u32 = HOSTLINK_REQUEST_CAPACITY;
    const RESPONSE_CAPACITY: u32 = HOSTLINK_RESPONSE_CAPACITY;
    const SERVE_PAIRS: usize = HOSTLINK_SERVE_PAIRS;

    fn encode_req(request: &Self::Request) -> Self::ReqWire {
        encode_req(request)
    }

    fn decode_req(wire: &Self::ReqWire) -> Option<Self::Request> {
        decode_req(wire)
    }

    fn encode_rep(response: &Self::Response, grants: u8) -> Self::RepWire {
        debug_assert_eq!(grants, 0);
        encode_rep(response)
    }

    fn decode_rep(wire: &Self::RepWire) -> Option<(Self::Response, u8)> {
        decode_rep(wire).map(|response| (response, 0))
    }

    fn reply_matches(request: &Self::ReqWire, response: &Self::Response, grants: u8) -> bool {
        if grants != 0 {
            return false;
        }
        match (decode_req(request), response) {
            (Some(ClientRequest::Poll), ClientResponse::Event(_) | ClientResponse::Empty) => true,
            (Some(ClientRequest::Status), ClientResponse::Status(_)) => true,
            (
                Some(ClientRequest::Supervisor {
                    request: expected, ..
                }),
                ClientResponse::Supervisor { request, .. },
            ) => expected == *request,
            (Some(ClientRequest::Supervisor { .. }), ClientResponse::Accepted { .. }) => false,
            (Some(req), ClientResponse::Accepted { request }) => req.request_id() == *request,
            (Some(req), ClientResponse::Refused { request, .. }) => req.request_id() == *request,
            _ => false,
        }
    }
}

// The transport-independent core speaks the identical wire when a concrete
// entry later supplies its transport and mailbox types. Keeping this impl
// beside the marker's codec prevents the host and client from drifting.
impl<T, M> SvcProtocol for crate::HostLinkService<T, M> {
    type Request = ClientRequest;
    type Response = ClientResponse;
    type ReqWire = ReqWire;
    type RepWire = RepWire;

    const REPLY_GRANTS: usize = 0;
    const REQUEST_CAPACITY: u32 = HOSTLINK_REQUEST_CAPACITY;
    const RESPONSE_CAPACITY: u32 = HOSTLINK_RESPONSE_CAPACITY;
    const SERVE_PAIRS: usize = HOSTLINK_SERVE_PAIRS;

    fn encode_req(request: &Self::Request) -> Self::ReqWire {
        encode_req(request)
    }

    fn decode_req(wire: &Self::ReqWire) -> Option<Self::Request> {
        decode_req(wire)
    }

    fn encode_rep(response: &Self::Response, grants: u8) -> Self::RepWire {
        debug_assert_eq!(grants, 0);
        encode_rep(response)
    }

    fn decode_rep(wire: &Self::RepWire) -> Option<(Self::Response, u8)> {
        decode_rep(wire).map(|response| (response, 0))
    }

    fn reply_matches(request: &Self::ReqWire, response: &Self::Response, grants: u8) -> bool {
        <HostLinkWire as SvcProtocol>::reply_matches(request, response, grants)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    #[test]
    fn supervisor_control_has_exact_version_identity_and_reply_correlation() {
        for (operation, generation) in [
            (SupervisorOperation::Observe, 0),
            (SupervisorOperation::Close, u64::MAX),
            (SupervisorOperation::Open, 9),
        ] {
            let request = ClientRequest::Supervisor {
                request: RequestId(id(7)),
                operation,
                generation,
            };
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            let state = SupervisorState {
                phase: SupervisorPhase::Closed,
                incarnation: TaskGeneration::new(8).unwrap(),
                workers: [TaskGeneration::new(u64::MAX).unwrap(); HOSTLINK_SUPERVISOR_WORKERS],
            };
            let reply = ClientResponse::Supervisor {
                request: RequestId(id(7)),
                state,
            };
            let mut encoded = encode_rep(&reply);
            assert_eq!(decode_rep(&encoded), Some(reply.clone()));
            assert!(HostLinkWire::reply_matches(&wire, &reply, 0));
            assert!(!HostLinkWire::reply_matches(
                &wire,
                &ClientResponse::Supervisor {
                    request: RequestId(id(8)),
                    state
                },
                0
            ));
            encoded.kind = U16::new(2);
            assert!(decode_rep(&encoded).is_none());
            let mut bad = wire;
            bad.flags = 1;
            assert!(decode_req(&bad).is_none());
        }
        let request = ClientRequest::Supervisor {
            request: RequestId(id(7)),
            operation: SupervisorOperation::Close,
            generation: 0,
        };
        assert!(decode_req(&encode_req(&request)).is_none());
    }

    #[test]
    fn wire_round_trips_and_rejects_reserved_bits() {
        let request = ClientRequest::BulkChunk {
            request: RequestId(id(1)),
            bulk_id: BulkId(id(2)),
            bytes: Payload::copy_from(b"bounded").unwrap(),
        };
        let mut wire = encode_req(&request);
        assert_eq!(decode_req(&wire), Some(request));
        wire.offset = U32::new(17);
        assert_eq!(decode_req(&wire), None);
        wire.offset = U32::ZERO;
        wire.flags = 1;
        assert_eq!(decode_req(&wire), None);

        for lifecycle in [
            ClientRequest::RotateAgent {
                request: RequestId(id(9)),
                slot: 3,
                generation: TaskGeneration::new(0x1_0000_0002).unwrap(),
            },
            ClientRequest::CloseAgent {
                request: RequestId(id(10)),
                slot: 3,
                generation: TaskGeneration::new(0x1_0000_0002).unwrap(),
            },
        ] {
            let mut wire = encode_req(&lifecycle);
            assert_eq!(decode_req(&wire), Some(lifecycle));
            wire.kind = U16::new(1);
            assert_eq!(decode_req(&wire), None);
        }

        let response = ClientResponse::Event(ClientEvent::Status(StatusNotice {
            request: RequestId(id(3)),
            reason: StatusReason::Refused(RefusalCode::PolicyDenied),
        }));
        let mut wire = encode_rep(&response);
        assert_eq!(decode_rep(&wire), Some(response));
        wire.reserved = U16::new(1);
        assert_eq!(decode_rep(&wire), None);
    }

    #[test]
    fn status_attests_frozen_bulk_transport_and_rejects_unknown_metadata() {
        for bulk_transport in [BulkTransport::Serial, BulkTransport::SharedMemory] {
            let response = ClientResponse::Status(ClientStatus {
                phase: ClientLinkPhase::Active,
                bulk_transport,
                queued_status: 3,
                queued_control: 5,
                queued_bulk: 8,
            });
            let mut wire = encode_rep(&response);
            assert_eq!(
                wire.reserved.get().to_le_bytes(),
                [RepWire::STATUS_META_VERSION, bulk_transport as u8]
            );
            assert_eq!(decode_rep(&wire), Some(response));

            wire.reserved = U16::new(u16::from_le_bytes([2, bulk_transport as u8]));
            assert_eq!(decode_rep(&wire), None);
            wire.reserved = U16::new(u16::from_le_bytes([RepWire::STATUS_META_VERSION, 0]));
            assert_eq!(decode_rep(&wire), None);
        }
    }

    #[test]
    fn discriminants_and_geometry_are_frozen() {
        assert_eq!(ReqWire::OP_SEND, 1);
        assert_eq!(ReqWire::OP_BULK_ABORT, 8);
        assert_eq!(ReqWire::OP_ROTATE_AGENT, 9);
        assert_eq!(ReqWire::OP_CLOSE_AGENT, 10);
        assert_eq!(RepWire::TAG_ACCEPTED, 1);
        assert_eq!(RepWire::TAG_REFUSED, 5);
        assert_eq!(RepWire::STATUS_META_VERSION, 1);
        assert_eq!(BulkTransport::Serial as u8, 1);
        assert_eq!(BulkTransport::SharedMemory as u8, 2);
        assert_eq!(core::mem::size_of::<ReqWire>(), 596);
        assert_eq!(core::mem::size_of::<RepWire>(), 596);
        assert_eq!(HOSTLINK_PAIR_WINDOW_BYTES, 2816);
        assert_eq!(HostLinkWire::SERVE_PAIRS, 5);
        assert_eq!(encode_req(&ClientRequest::Poll).as_bytes().len(), 596);
    }

    #[test]
    fn request_has_no_endpoint_selector() {
        let poll = encode_req(&ClientRequest::Poll);
        assert_eq!(decode_req(&poll), Some(ClientRequest::Poll));
        // Any attempt to smuggle a selector through dead fields is rejected
        // by exact re-encoding.
        let mut forged = poll;
        forged.kind = U16::new(1);
        assert_eq!(decode_req(&forged), None);
    }
}
