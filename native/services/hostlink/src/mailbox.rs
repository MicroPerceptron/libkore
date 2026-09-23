//! Allocation-free per-pair mailboxes for the founded HostLink service.

use heapless::Deque;
use kore_hostlink::{BulkId, BulkRefV1, RefusalCode, RequestId, TaskGeneration};

use crate::{
    BulkTransport, CLIENT_CHUNK_MAX, ClientEvent, ClientLinkPhase, ClientStatus, DeliveryError,
    Endpoint, EndpointStatus, LinkPhase, MAX_DYNAMIC_AGENT_ROUTES, MailboxDelivery, Payload,
    StatusNotice, StatusReason,
};

pub const MAILBOX_STATUS_CAPACITY: usize = 17;
pub const MAILBOX_CONTROL_CAPACITY: usize = 8;
/// One 4 KiB workspace range becomes eight founded 512-byte events, plus its
/// Begin and Ready records. Two spare slots preserve bounded control progress.
pub const MAILBOX_BULK_CAPACITY: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MailboxError {
    UnknownPair,
    Backpressure,
}

struct EndpointMailbox {
    status: Deque<StatusNotice, MAILBOX_STATUS_CAPACITY>,
    control: Deque<ClientEvent, MAILBOX_CONTROL_CAPACITY>,
    bulk: Deque<ClientEvent, MAILBOX_BULK_CAPACITY>,
}

impl EndpointMailbox {
    const fn new() -> Self {
        Self {
            status: Deque::new(),
            control: Deque::new(),
            bulk: Deque::new(),
        }
    }
}

const MAILBOX_ROWS: usize = MAX_DYNAMIC_AGENT_ROUTES + 2;

/// Structurally owned broker/workspace mailboxes plus a bounded dynamic agent
/// pool. There is no route/name lookup and no global state: the serving pair's
/// `via` chooses an installed generation-bound row.
pub struct MailboxBank {
    rows: [EndpointMailbox; MAILBOX_ROWS],
    broker_active: bool,
    agents: [Option<TaskGeneration>; MAX_DYNAMIC_AGENT_ROUTES],
}

impl Default for MailboxBank {
    fn default() -> Self {
        Self::new()
    }
}

impl MailboxBank {
    pub const fn new() -> Self {
        Self {
            rows: [const { EndpointMailbox::new() }; MAILBOX_ROWS],
            broker_active: true,
            agents: [None; MAX_DYNAMIC_AGENT_ROUTES],
        }
    }

    /// Authenticate a request by its serving pair. Payload bytes never call
    /// this function and cannot select its result.
    pub fn endpoint_for_via(&self, via: u16) -> Option<Endpoint> {
        let endpoint = match via {
            0 => Endpoint::Broker,
            1 => Endpoint::Agent(0),
            2 => Endpoint::Agent(1),
            3 => Endpoint::Workspace,
            _ if via >= 5 && usize::from(via - 3) < MAX_DYNAMIC_AGENT_ROUTES => {
                Endpoint::Agent((via - 3) as u8)
            }
            _ => return None,
        };
        match endpoint {
            Endpoint::Broker if !self.broker_active => None,
            Endpoint::Agent(slot) if self.agents[slot as usize].is_none() => None,
            _ => Some(endpoint),
        }
    }

    pub fn poll_via(&mut self, via: u16) -> Result<Option<ClientEvent>, MailboxError> {
        let endpoint = self
            .endpoint_for_via(via)
            .ok_or(MailboxError::UnknownPair)?;
        let row = &mut self.rows[index(endpoint)];
        if let Some(status) = row.status.pop_front() {
            return Ok(Some(ClientEvent::Status(status)));
        }
        if let Some(control) = row.control.pop_front() {
            return Ok(Some(control));
        }
        Ok(row.bulk.pop_front())
    }

    pub fn status_via(
        &self,
        via: u16,
        phase: LinkPhase,
        bulk_transport: BulkTransport,
    ) -> Result<ClientStatus, MailboxError> {
        let endpoint = self
            .endpoint_for_via(via)
            .ok_or(MailboxError::UnknownPair)?;
        let row = &self.rows[index(endpoint)];
        Ok(ClientStatus {
            phase: match phase {
                LinkPhase::Disconnected => ClientLinkPhase::Disconnected,
                LinkPhase::Negotiating => ClientLinkPhase::Negotiating,
                LinkPhase::Opening => ClientLinkPhase::Opening,
                LinkPhase::Active => ClientLinkPhase::Active,
            },
            bulk_transport,
            queued_status: row.status.len() as u16,
            queued_control: row.control.len() as u16,
            queued_bulk: row.bulk.len() as u16,
        })
    }

    /// Copy one core-authored cancellation/refusal/disconnect status into the
    /// endpoint's reserved status lane. The entry router calls this while
    /// draining `HostLinkService::pop_status`.
    pub fn push_status(&mut self, status: EndpointStatus) -> Result<(), MailboxError> {
        self.rows[index(status.endpoint)]
            .status
            .push_back(StatusNotice {
                request: status.request,
                reason: status.reason,
            })
            .map_err(|_| MailboxError::Backpressure)
    }

    fn push_control(
        &mut self,
        endpoint: Endpoint,
        event: ClientEvent,
    ) -> Result<(), DeliveryError> {
        self.rows[index(endpoint)]
            .control
            .push_back(event)
            .map_err(|_| DeliveryError::Backpressure)
    }

    fn push_bulk(&mut self, endpoint: Endpoint, event: ClientEvent) -> Result<(), DeliveryError> {
        self.rows[index(endpoint)]
            .bulk
            .push_back(event)
            .map_err(|_| DeliveryError::Backpressure)
    }
}

impl MailboxDelivery for MailboxBank {
    fn install_broker(&mut self) -> Result<(), DeliveryError> {
        if self.broker_active {
            return Err(DeliveryError::Refused);
        }
        self.broker_active = true;
        Ok(())
    }
    fn close_broker(&mut self) -> bool {
        self.rows[index(Endpoint::Broker)] = EndpointMailbox::new();
        self.broker_active = false;
        true
    }

    fn install_agent(&mut self, slot: u8, generation: TaskGeneration) -> Result<(), DeliveryError> {
        let row = self
            .agents
            .get_mut(slot as usize)
            .ok_or(DeliveryError::Malformed)?;
        if row.is_some() {
            return Err(DeliveryError::Refused);
        }
        *row = Some(generation);
        Ok(())
    }

    fn close_agent(&mut self, slot: u8, generation: TaskGeneration) -> bool {
        let Some(row) = self.agents.get_mut(slot as usize) else {
            return false;
        };
        if *row != Some(generation) {
            return false;
        }
        self.rows[agent_index(slot)] = EndpointMailbox::new();
        *row = None;
        true
    }

    fn control(
        &mut self,
        endpoint: Endpoint,
        kind: u16,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), DeliveryError> {
        let payload = Payload::copy_from(payload).ok_or(DeliveryError::Refused)?;
        self.push_control(
            endpoint,
            ClientEvent::Control {
                kind,
                request,
                payload,
            },
        )
    }

    fn bulk_begin(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        reference: BulkRefV1,
    ) -> Result<(), DeliveryError> {
        if !reference.shape_is_valid() {
            return Err(DeliveryError::Malformed);
        }
        self.push_bulk(endpoint, ClientEvent::BulkBegin { request, reference })
    }

    fn bulk_chunk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), DeliveryError> {
        if bytes.is_empty() {
            return Err(DeliveryError::Malformed);
        }
        let chunks = bytes.len().div_ceil(CLIENT_CHUNK_MAX);
        let row = &mut self.rows[index(endpoint)];
        if chunks > MAILBOX_BULK_CAPACITY - row.bulk.len() {
            return Err(DeliveryError::Backpressure);
        }
        let _final_offset = u32::try_from(bytes.len())
            .ok()
            .and_then(|len| offset.checked_add(len))
            .ok_or(DeliveryError::Malformed)?;
        for (fragment_index, fragment) in bytes.chunks(CLIENT_CHUNK_MAX).enumerate() {
            let delta = u32::try_from(fragment_index * CLIENT_CHUNK_MAX)
                .map_err(|_| DeliveryError::Malformed)?;
            let fragment_offset = offset.checked_add(delta).ok_or(DeliveryError::Malformed)?;
            let payload = Payload::copy_from(fragment).ok_or(DeliveryError::Malformed)?;
            // Capacity was checked for the complete batch, so failure here is
            // unreachable without a queue implementation defect.
            row.bulk
                .push_back(ClientEvent::BulkChunk {
                    request,
                    bulk_id: id,
                    offset: fragment_offset,
                    bytes: payload,
                })
                .map_err(|_| DeliveryError::Backpressure)?;
        }
        Ok(())
    }

    fn bulk_ready(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
    ) -> Result<(), DeliveryError> {
        self.push_bulk(
            endpoint,
            ClientEvent::BulkReady {
                request,
                bulk_id: id,
            },
        )
    }

    fn bulk_abort(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        reason: RefusalCode,
    ) {
        if self
            .push_control(
                endpoint,
                ClientEvent::BulkAbort {
                    request,
                    bulk_id: id,
                    reason,
                },
            )
            .is_err()
        {
            let _ = self.rows[index(endpoint)].status.push_back(StatusNotice {
                request,
                reason: StatusReason::Refused(RefusalCode::Backpressure),
            });
        }
    }

    fn cancel(&mut self, endpoint: Endpoint, request: RequestId) -> Result<(), DeliveryError> {
        self.push_control(endpoint, ClientEvent::Cancel { request })
    }
}

const fn index(endpoint: Endpoint) -> usize {
    match endpoint {
        Endpoint::Broker => 0,
        Endpoint::Agent(slot) => agent_index(slot),
        Endpoint::Workspace => 1,
    }
}

const fn agent_index(slot: u8) -> usize {
    slot as usize + 2
}

const _: () = {
    assert!(MAILBOX_STATUS_CAPACITY >= 17);
    assert!(MAILBOX_CONTROL_CAPACITY > 0);
    assert!(MAILBOX_BULK_CAPACITY * CLIENT_CHUNK_MAX >= 4096);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    #[test]
    fn pair_identity_is_the_only_mailbox_selector() {
        let mut bank = MailboxBank::new();
        let generation = TaskGeneration::new(1).unwrap();
        bank.install_agent(0, generation).unwrap();
        bank.install_agent(1, generation).unwrap();
        bank.control(Endpoint::Agent(0), 9, RequestId(id(1)), b"agent-zero")
            .unwrap();
        bank.control(Endpoint::Agent(1), 10, RequestId(id(2)), b"agent-one")
            .unwrap();

        assert!(matches!(
            bank.poll_via(1).unwrap(),
            Some(ClientEvent::Control { kind: 9, .. })
        ));
        assert!(matches!(
            bank.poll_via(2).unwrap(),
            Some(ClientEvent::Control { kind: 10, .. })
        ));
        assert_eq!(bank.poll_via(3), Ok(None));
        assert_eq!(bank.poll_via(4), Err(MailboxError::UnknownPair));
    }

    #[test]
    fn broker_close_clears_only_broker_delivery_and_requires_explicit_reopen() {
        let mut bank = MailboxBank::new();
        assert_eq!(bank.endpoint_for_via(0), Some(Endpoint::Broker));
        assert_eq!(bank.endpoint_for_via(3), Some(Endpoint::Workspace));
        assert!(bank.close_broker());
        assert_eq!(bank.endpoint_for_via(0), None);
        assert_eq!(bank.endpoint_for_via(3), Some(Endpoint::Workspace));
        assert_eq!(bank.install_broker(), Ok(()));
        assert_eq!(bank.endpoint_for_via(0), Some(Endpoint::Broker));
        assert_eq!(bank.install_broker(), Err(DeliveryError::Refused));
    }

    #[test]
    fn dynamic_row_closes_generation_before_reuse_and_stale_close_fails() {
        let mut bank = MailboxBank::new();
        let first = TaskGeneration::new(7).unwrap();
        let replacement = TaskGeneration::new(8).unwrap();
        bank.install_agent(2, first).unwrap();
        assert_eq!(bank.endpoint_for_via(5), Some(Endpoint::Agent(2)));
        bank.control(Endpoint::Agent(2), 9, RequestId(id(1)), b"old")
            .unwrap();
        assert!(bank.close_agent(2, first));
        assert_eq!(bank.endpoint_for_via(5), None);
        bank.install_agent(2, replacement).unwrap();
        assert!(!bank.close_agent(2, first));
        assert_eq!(bank.poll_via(5), Ok(None));
        assert!(bank.close_agent(2, replacement));
    }

    #[test]
    fn every_declared_agent_route_has_one_distinct_mailbox_pair() {
        let mut bank = MailboxBank::new();
        let generation = TaskGeneration::new(1).unwrap();
        for slot in 0..MAX_DYNAMIC_AGENT_ROUTES {
            bank.install_agent(u8::try_from(slot).unwrap(), generation)
                .unwrap();
            let via = if slot < 2 { slot + 1 } else { slot + 3 } as u16;
            assert_eq!(
                bank.endpoint_for_via(via),
                Some(Endpoint::Agent(slot as u8))
            );
            bank.control(
                Endpoint::Agent(slot as u8),
                9,
                RequestId(id(slot as u8)),
                b"private",
            )
            .unwrap();
        }
        assert_eq!(bank.poll_via(0), Ok(None));
        assert_eq!(bank.poll_via(3), Ok(None));
        for slot in 0..MAX_DYNAMIC_AGENT_ROUTES {
            let via = if slot < 2 { slot + 1 } else { slot + 3 } as u16;
            assert!(
                matches!(bank.poll_via(via), Ok(Some(ClientEvent::Control { request, .. })) if request == RequestId(id(slot as u8)))
            );
        }
        assert_eq!(
            bank.endpoint_for_via((MAX_DYNAMIC_AGENT_ROUTES + 3) as u16),
            None
        );
        assert_eq!(bank.endpoint_for_via(u16::MAX), None);
    }

    #[test]
    fn bulk_delivery_is_bounded_and_atomic() {
        let mut bank = MailboxBank::new();
        bank.install_agent(0, TaskGeneration::new(1).unwrap())
            .unwrap();
        let too_many = [0xA5; MAILBOX_BULK_CAPACITY * CLIENT_CHUNK_MAX + 1];
        assert_eq!(
            bank.bulk_chunk(
                Endpoint::Agent(0),
                RequestId(id(1)),
                BulkId(id(2)),
                0,
                &too_many,
            ),
            Err(DeliveryError::Backpressure)
        );
        assert_eq!(bank.poll_via(1).unwrap(), None);

        let bytes = [0x5A; CLIENT_CHUNK_MAX + 7];
        bank.bulk_chunk(
            Endpoint::Agent(0),
            RequestId(id(1)),
            BulkId(id(2)),
            33,
            &bytes,
        )
        .unwrap();
        let first = bank.poll_via(1).unwrap().unwrap();
        let second = bank.poll_via(1).unwrap().unwrap();
        assert!(matches!(
            first,
            ClientEvent::BulkChunk { offset: 33, bytes, .. }
                if bytes.len() == CLIENT_CHUNK_MAX
        ));
        assert!(matches!(
            second,
            ClientEvent::BulkChunk { offset, bytes, .. }
                if offset == 33 + CLIENT_CHUNK_MAX as u32 && bytes.len() == 7
        ));
    }

    #[test]
    fn complete_workspace_range_lifecycle_preserves_wire_order() {
        let mut bank = MailboxBank::new();
        let request = RequestId(id(1));
        let bulk_id = BulkId(id(2));
        let bytes = [0x5A; 4096];
        let reference = BulkRefV1 {
            bulk_id,
            total_len: zerocopy::byteorder::little_endian::U32::new(bytes.len() as u32),
            chunk_count: zerocopy::byteorder::little_endian::U16::new(1),
            content_kind: zerocopy::byteorder::little_endian::U16::new(1),
            digest_blake3: *blake3::hash(&bytes).as_bytes(),
        };

        bank.bulk_begin(Endpoint::Workspace, request, reference)
            .unwrap();
        bank.bulk_chunk(Endpoint::Workspace, request, bulk_id, 0, &bytes)
            .unwrap();
        bank.bulk_ready(Endpoint::Workspace, request, bulk_id)
            .unwrap();

        assert!(matches!(
            bank.poll_via(3).unwrap(),
            Some(ClientEvent::BulkBegin { .. })
        ));
        for expected in 0..8u32 {
            assert!(matches!(
                bank.poll_via(3).unwrap(),
                Some(ClientEvent::BulkChunk { offset, bytes, .. })
                    if offset == expected * CLIENT_CHUNK_MAX as u32
                        && bytes.len() == CLIENT_CHUNK_MAX
            ));
        }
        assert!(matches!(
            bank.poll_via(3).unwrap(),
            Some(ClientEvent::BulkReady { .. })
        ));
    }

    #[test]
    fn status_lane_precedes_saturated_data_lanes() {
        let mut bank = MailboxBank::new();
        bank.install_agent(1, TaskGeneration::new(1).unwrap())
            .unwrap();
        for n in 0..MAILBOX_CONTROL_CAPACITY {
            bank.control(
                Endpoint::Agent(1),
                n as u16,
                RequestId(id(n as u8)),
                b"full",
            )
            .unwrap();
        }
        bank.push_status(EndpointStatus {
            endpoint: Endpoint::Agent(1),
            request: RequestId(id(9)),
            reason: StatusReason::HostDisconnected,
        })
        .unwrap();
        assert!(matches!(
            bank.poll_via(2).unwrap(),
            Some(ClientEvent::Status(StatusNotice {
                reason: StatusReason::HostDisconnected,
                ..
            }))
        ));
    }

    #[test]
    fn status_reports_the_composed_bulk_transport_without_affecting_authority() {
        let bank = MailboxBank::new();
        for bulk_transport in [BulkTransport::Serial, BulkTransport::SharedMemory] {
            let status = bank
                .status_via(0, LinkPhase::Active, bulk_transport)
                .unwrap();
            assert_eq!(status.phase, ClientLinkPhase::Active);
            assert_eq!(status.bulk_transport, bulk_transport);
            assert_eq!(status.queued_status, 0);
            assert_eq!(status.queued_control, 0);
            assert_eq!(status.queued_bulk, 0);
        }
    }
}
