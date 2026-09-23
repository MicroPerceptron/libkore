use heapless::{Deque, Vec};
use kore_hostlink::*;
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use crate::FOUNDED_BULK_WIRE_CHUNK;

pub const MAX_DYNAMIC_AGENT_ROUTES: usize = kore_hostlink::AGENT_ROUTE_CAPACITY;
const ROUTES: usize = MAX_DYNAMIC_AGENT_ROUTES + 2;
const BULKS: usize = 16;
const STATUSES: usize = 64;
const READ_CHUNK: usize = 4096;
const CONTROL_DRAIN_BUDGET: usize = 16;
pub const BULK_DRAIN_BUDGET: usize = 2;
const SMALL_FRAME: usize = ENVELOPE_LEN + SAFE_DETAIL_MAX + REFUSAL_HEADER_LEN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Endpoint {
    Broker,
    Agent(u8),
    Workspace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalRoute {
    pub route: RouteId,
    pub generation: TaskGeneration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bindings {
    pub broker: LocalRoute,
    broker_active: bool,
    pub workspace: LocalRoute,
    agents: [Option<LocalRoute>; MAX_DYNAMIC_AGENT_ROUTES],
}

impl Bindings {
    pub fn new(broker: LocalRoute, workspace: LocalRoute) -> Option<Self> {
        if broker.route.is_zero() || workspace.route.is_zero() || workspace.route == broker.route {
            None
        } else {
            Some(Self {
                broker,
                broker_active: true,
                workspace,
                agents: [None; MAX_DYNAMIC_AGENT_ROUTES],
            })
        }
    }

    pub fn without_broker(mut self) -> Self {
        self.broker_active = false;
        self
    }

    pub fn install_agent(&mut self, slot: usize, local: LocalRoute) -> bool {
        if slot >= MAX_DYNAMIC_AGENT_ROUTES
            || local.route.is_zero()
            || self.agents[slot].is_some()
            || self
                .rows()
                .iter()
                .any(|(_, row, _, _, _)| row.route == local.route)
        {
            return false;
        }
        self.agents[slot] = Some(local);
        true
    }

    pub fn remove_agent(&mut self, slot: usize, generation: TaskGeneration) -> Option<LocalRoute> {
        let row = self.agents.get_mut(slot)?;
        let local = row.as_ref()?;
        if local.generation != generation {
            return None;
        }
        row.take()
    }

    fn rows(&self) -> Vec<(Endpoint, LocalRoute, RouteRole, u8, u8), ROUTES> {
        let mut rows = Vec::new();
        if self.broker_active {
            rows.push((Endpoint::Broker, self.broker, RouteRole::Broker, 0, 0))
                .expect("structural broker route fits");
        }
        for (slot, local) in self.agents.iter().copied().enumerate() {
            if let Some(local) = local {
                rows.push((
                    Endpoint::Agent(slot as u8),
                    local,
                    RouteRole::Agent,
                    1,
                    slot as u8,
                ))
                .expect("bounded agent route fits");
            }
        }
        rows.push((
            Endpoint::Workspace,
            self.workspace,
            RouteRole::Importer,
            2,
            0,
        ))
        .expect("structural workspace route fits");
        rows
    }

    fn for_endpoint(&self, endpoint: Endpoint) -> Option<LocalRoute> {
        match endpoint {
            Endpoint::Broker => self.broker_active.then_some(self.broker),
            Endpoint::Agent(slot) => self.agents.get(slot as usize).copied().flatten(),
            Endpoint::Workspace => Some(self.workspace),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryError {
    Backpressure,
    Malformed,
    Refused,
}

pub trait MailboxDelivery {
    fn install_broker(&mut self) -> Result<(), DeliveryError> {
        Ok(())
    }
    fn close_broker(&mut self) -> bool {
        true
    }

    fn install_agent(
        &mut self,
        _slot: u8,
        _generation: TaskGeneration,
    ) -> Result<(), DeliveryError> {
        Ok(())
    }
    fn close_agent(&mut self, _slot: u8, _generation: TaskGeneration) -> bool {
        true
    }
    fn control(
        &mut self,
        endpoint: Endpoint,
        kind: u16,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), DeliveryError>;
    fn bulk_begin(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        reference: BulkRefV1,
    ) -> Result<(), DeliveryError>;
    fn bulk_chunk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), DeliveryError>;
    fn bulk_ready(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
    ) -> Result<(), DeliveryError>;
    fn bulk_abort(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        reason: RefusalCode,
    );
    fn cancel(&mut self, endpoint: Endpoint, request: RequestId) -> Result<(), DeliveryError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatusReason {
    HostDisconnected,
    Cancelled,
    Refused(RefusalCode),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndpointStatus {
    pub endpoint: Endpoint,
    pub request: RequestId,
    pub reason: StatusReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkPhase {
    Disconnected,
    Negotiating,
    Opening,
    Active,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrainReport {
    pub control_frames: u8,
    pub bulk_frames: u8,
    /// Bulk transport reads that made progress, including partial frames.
    pub bulk_reads: u8,
    /// Latest validated coalesced shared-memory publication edge.
    pub shmem_publication: u64,
    pub disconnected: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Transport(PortError),
    Decode(DecodeError),
    Link,
    Route,
    Backpressure,
    Malformed,
    Bulk,
}

struct InboundBulk {
    route: RouteId,
    request: RequestId,
    reference: BulkRefV1,
    offset: u32,
    chunks: u16,
    hasher: blake3::Hasher,
}

struct OutboundBulk {
    endpoint: Endpoint,
    request: RequestId,
    reference: BulkRefV1,
    received: u32,
    offset: u32,
    chunks: u16,
    staged: u16,
    staging: [u8; FOUNDED_BULK_WIRE_CHUNK],
    hasher: blake3::Hasher,
    phase: OutboundPhase,
    credit_bytes: u64,
    credit_chunks: u64,
    in_flight_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutboundPhase {
    AwaitBegin,
    Ready,
    AwaitChunk,
    AwaitFinish,
    AwaitAbort,
}

struct PlaneInput {
    bytes: [u8; READ_CHUNK],
    start: usize,
    end: usize,
}

#[derive(Clone, Copy)]
struct ClosingRoute {
    local: LocalRoute,
    acknowledged: bool,
}

impl PlaneInput {
    const fn new() -> Self {
        Self {
            bytes: [0; READ_CHUNK],
            start: 0,
            end: 0,
        }
    }
    fn available(&self) -> &[u8] {
        &self.bytes[self.start..self.end]
    }
    fn consume(&mut self, count: usize) {
        self.start += count;
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        }
    }
    fn clear(&mut self) {
        self.start = 0;
        self.end = 0;
    }
}

pub struct HostLinkService<T, M> {
    transport: T,
    delivery: M,
    boot: [u8; 32],
    bindings: Bindings,
    routes: RouteTable<Endpoint, ROUTES>,
    opened: [bool; ROUTES],
    closing_broker: Option<ClosingRoute>,
    closed_broker: Option<TaskGeneration>,
    restart_broker: Option<TaskGeneration>,
    closing_agents: [Option<ClosingRoute>; MAX_DYNAMIC_AGENT_ROUTES],
    closed_agents: [Option<TaskGeneration>; MAX_DYNAMIC_AGENT_ROUTES],
    phase: LinkPhase,
    epoch: Option<LinkEpoch>,
    shmem_publication: u64,
    ack_pending: bool,
    open_cursor: u8,
    control_decoder: ControlFrameDecoder,
    bulk_decoder: BulkFrameDecoder,
    control_input: PlaneInput,
    bulk_input: PlaneInput,
    inbound: Vec<InboundBulk, BULKS>,
    outbound: Vec<OutboundBulk, BULKS>,
    statuses: Deque<EndpointStatus, STATUSES>,
    /// A routed refusal means the peer deliberately did not consume the
    /// corresponding guest sequence. No later frame in that epoch can be
    /// interpreted consistently, so bytes are discarded until the transport
    /// reports the physical close that precedes a fresh Hello.
    poisoned: bool,
}

impl<T: HostLinkTransport, M: MailboxDelivery> HostLinkService<T, M> {
    pub fn new(transport: T, delivery: M, boot: [u8; 32], bindings: Bindings) -> Self {
        Self {
            transport,
            delivery,
            boot,
            bindings,
            routes: RouteTable::new(),
            opened: [false; ROUTES],
            closing_broker: None,
            closed_broker: None,
            restart_broker: None,
            closing_agents: [None; MAX_DYNAMIC_AGENT_ROUTES],
            closed_agents: [None; MAX_DYNAMIC_AGENT_ROUTES],
            phase: LinkPhase::Disconnected,
            epoch: None,
            shmem_publication: 0,
            ack_pending: false,
            open_cursor: 0,
            control_decoder: ControlFrameDecoder::new(),
            bulk_decoder: BulkFrameDecoder::new(),
            control_input: PlaneInput::new(),
            bulk_input: PlaneInput::new(),
            inbound: Vec::new(),
            outbound: Vec::new(),
            statuses: Deque::new(),
            poisoned: false,
        }
    }

    pub const fn phase(&self) -> LinkPhase {
        self.phase
    }
    pub const fn epoch(&self) -> Option<LinkEpoch> {
        self.epoch
    }
    pub const fn shmem_publication(&self) -> u64 {
        self.shmem_publication
    }
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }
    /// End routing and move the sole transport owner to terminal teardown.
    pub fn into_transport(self) -> T {
        self.transport
    }
    pub fn delivery(&self) -> &M {
        &self.delivery
    }
    pub fn delivery_mut(&mut self) -> &mut M {
        &mut self.delivery
    }
    pub fn pop_status(&mut self) -> Option<EndpointStatus> {
        self.statuses.pop_front()
    }

    pub fn supervisor_restart_requested(&self, generation: TaskGeneration) -> bool {
        self.restart_broker == Some(generation)
    }

    /// Inert identity of a live or most recently closed local route.
    pub fn generation(&self, endpoint: Endpoint) -> Option<TaskGeneration> {
        self.bindings
            .for_endpoint(endpoint)
            .map(|local| local.generation)
            .or_else(|| match endpoint {
                Endpoint::Broker => self.closed_broker,
                Endpoint::Agent(slot) => self.closed_agents.get(slot as usize).copied().flatten(),
                Endpoint::Workspace => None,
            })
    }

    /// Install one generation-bound dynamic agent route. The service owns the
    /// route state and mailbox row; the founded runner receives only its
    /// already-wired pair. A failure unwinds both reservations before return.
    pub fn install_agent_route(&mut self, slot: u8, local: LocalRoute) -> Result<(), ServiceError> {
        if slot as usize >= MAX_DYNAMIC_AGENT_ROUTES {
            return Err(ServiceError::Route);
        }
        self.delivery
            .install_agent(slot, local.generation)
            .map_err(|_| ServiceError::Backpressure)?;
        if !self.bindings.install_agent(slot as usize, local) {
            let _ = self.delivery.close_agent(slot, local.generation);
            return Err(ServiceError::Route);
        }
        self.closed_agents[slot as usize] = None;
        if let Some(epoch) = self.epoch {
            let state = RouteState::new(
                epoch,
                local.route,
                local.generation,
                RouteRole::Agent,
                16,
                16 * 1024 * 1024,
                64,
            )
            .ok_or(ServiceError::Route)?;
            if self.routes.bind(state, Endpoint::Agent(slot)).is_err() {
                let _ = self.bindings.remove_agent(slot as usize, local.generation);
                let _ = self.delivery.close_agent(slot, local.generation);
                return Err(ServiceError::Route);
            }
            self.opened[endpoint_index(Endpoint::Agent(slot))] = false;
            self.open_cursor = 0;
            self.phase = LinkPhase::Opening;
        }
        Ok(())
    }

    fn derived_agent_route(&self, slot: u8, generation: TaskGeneration) -> RouteId {
        let mut input = [0_u8; 24];
        input[..15].copy_from_slice(b"agent-route-v1\0");
        input[15] = slot;
        input[16..].copy_from_slice(&generation.get().to_le_bytes());
        RouteId(
            blake3::keyed_hash(&self.boot, &input).as_bytes()[..16]
                .try_into()
                .unwrap(),
        )
    }

    /// Close the current row completely before deriving and publishing its
    /// successor. The caller can retry Backpressure with the same generation;
    /// no identity wraps or becomes reusable while the host/mailbox still
    /// holds the previous generation.
    pub fn rotate_agent_route(
        &mut self,
        slot: u8,
        generation: TaskGeneration,
    ) -> Result<(), ServiceError> {
        let index = usize::from(slot);
        if index >= MAX_DYNAMIC_AGENT_ROUTES {
            return Err(ServiceError::Route);
        }
        if let Some(current) = self.bindings.for_endpoint(Endpoint::Agent(slot)) {
            let expected = current
                .generation
                .get()
                .checked_add(1)
                .and_then(TaskGeneration::new)
                .ok_or(ServiceError::Route)?;
            if generation != expected {
                return Err(ServiceError::Route);
            }
            self.close_agent_route(slot, current.generation)?;
        }
        let closed = self.closed_agents[index].ok_or(ServiceError::Route)?;
        if closed.get().checked_add(1).and_then(TaskGeneration::new) != Some(generation) {
            return Err(ServiceError::Route);
        }
        let local = LocalRoute {
            route: self.derived_agent_route(slot, generation),
            generation,
        };
        self.install_agent_route(slot, local)
    }

    /// Close the old generation before returning its bounded row. Transport
    /// backpressure leaves every owner installed so teardown can retry.
    pub fn close_agent_route(
        &mut self,
        slot: u8,
        generation: TaskGeneration,
    ) -> Result<(), ServiceError> {
        self.close_endpoint(Endpoint::Agent(slot), generation)
    }

    /// The surviving local connection owner invokes this only after reaping
    /// the broker process. The workspace route is independent and stays live.
    pub fn close_broker_route(&mut self, generation: TaskGeneration) -> Result<(), ServiceError> {
        self.close_endpoint(Endpoint::Broker, generation)
    }

    fn closing(&self, endpoint: Endpoint) -> Option<ClosingRoute> {
        match endpoint {
            Endpoint::Broker => self.closing_broker,
            Endpoint::Agent(slot) => self.closing_agents.get(slot as usize).copied().flatten(),
            Endpoint::Workspace => None,
        }
    }

    fn set_closing(&mut self, endpoint: Endpoint, closing: Option<ClosingRoute>) {
        match endpoint {
            Endpoint::Broker => self.closing_broker = closing,
            Endpoint::Agent(slot) => self.closing_agents[slot as usize] = closing,
            Endpoint::Workspace => unreachable!("workspace is not supervisor owned"),
        }
    }

    fn close_endpoint(
        &mut self,
        endpoint: Endpoint,
        generation: TaskGeneration,
    ) -> Result<(), ServiceError> {
        let closed = match endpoint {
            Endpoint::Broker => self.closed_broker,
            Endpoint::Agent(slot) => self.closed_agents.get(slot as usize).copied().flatten(),
            Endpoint::Workspace => return Err(ServiceError::Route),
        };
        if closed == Some(generation) {
            return Ok(());
        }
        let local = self
            .bindings
            .for_endpoint(endpoint)
            .filter(|local| local.generation == generation)
            .ok_or(ServiceError::Route)?;
        if let Some(closing) = self.closing(endpoint) {
            if closing.local != local {
                return Err(ServiceError::Route);
            }
            return if self.finish_route_close(endpoint) {
                Ok(())
            } else {
                Err(ServiceError::Backpressure)
            };
        }
        if self.epoch.is_some() {
            self.send_small_routed(endpoint, ROUTE_CLOSE, RequestId::ZERO, &[])?;
            self.set_closing(
                endpoint,
                Some(ClosingRoute {
                    local,
                    acknowledged: false,
                }),
            );
            return Err(ServiceError::Backpressure);
        }
        self.set_closing(
            endpoint,
            Some(ClosingRoute {
                local,
                acknowledged: true,
            }),
        );
        if self.finish_route_close(endpoint) {
            Ok(())
        } else {
            Err(ServiceError::Backpressure)
        }
    }

    fn finish_agent_close(&mut self, slot: u8) -> bool {
        self.finish_route_close(Endpoint::Agent(slot))
    }

    /// Acknowledgment precedes mailbox disposal and route reuse. Missing ACKs
    /// retain all old-generation work and cannot authorize a replacement.
    fn finish_route_close(&mut self, endpoint: Endpoint) -> bool {
        let Some(closing) = self.closing(endpoint) else {
            return false;
        };
        if !closing.acknowledged {
            return false;
        }
        match endpoint {
            Endpoint::Broker => {
                if !self.delivery.close_broker() {
                    return false;
                }
                self.bindings.broker_active = false;
                self.closed_broker = Some(closing.local.generation);
            }
            Endpoint::Agent(slot) => {
                if !self.delivery.close_agent(slot, closing.local.generation) {
                    return false;
                }
                if self
                    .bindings
                    .remove_agent(slot as usize, closing.local.generation)
                    .is_none()
                {
                    return false;
                }
                self.closed_agents[slot as usize] = Some(closing.local.generation);
            }
            Endpoint::Workspace => return false,
        }
        let _ = self.routes.unbind(closing.local.route);
        self.inbound
            .retain(|bulk| bulk.route != closing.local.route);
        self.outbound.retain(|bulk| bulk.endpoint != endpoint);
        for _ in 0..self.statuses.len() {
            if let Some(status) = self.statuses.pop_front() {
                if status.endpoint != endpoint {
                    let _ = self.statuses.push_back(status);
                }
            }
        }
        self.opened[endpoint_index(endpoint)] = false;
        self.set_closing(endpoint, None);
        true
    }

    /// Install only the exact successor after acknowledged closure. The route
    /// identity is derived from the service's boot secret, never caller bytes.
    pub fn rotate_broker_route(&mut self, generation: TaskGeneration) -> Result<(), ServiceError> {
        if self.bindings.broker_active
            || self.closed_broker.and_then(|g| g.get().checked_add(1)) != Some(generation.get())
        {
            return Err(ServiceError::Route);
        }
        let mut hash = blake3::Hasher::new_keyed(&self.boot);
        hash.update(b"broker-route-v1");
        hash.update(&generation.get().to_le_bytes());
        let route = RouteId(hash.finalize().as_bytes()[..16].try_into().unwrap());
        if route.is_zero()
            || self
                .bindings
                .rows()
                .iter()
                .any(|(_, local, _, _, _)| local.route == route)
        {
            return Err(ServiceError::Route);
        }
        self.delivery
            .install_broker()
            .map_err(|_| ServiceError::Backpressure)?;
        if let Some(epoch) = self.epoch {
            let state = RouteState::new(
                epoch,
                route,
                generation,
                RouteRole::Broker,
                16,
                16 * 1024 * 1024,
                64,
            )
            .ok_or(ServiceError::Route)?;
            if self.routes.bind(state, Endpoint::Broker).is_err() {
                let _ = self.delivery.close_broker();
                return Err(ServiceError::Route);
            }
            self.open_cursor = 0;
            self.phase = LinkPhase::Opening;
        }
        self.bindings.broker = LocalRoute { route, generation };
        self.bindings.broker_active = true;
        self.closed_broker = None;
        self.restart_broker = None;
        Ok(())
    }

    fn retry_agent_closes(&mut self) {
        if self
            .closing_broker
            .is_some_and(|closing| closing.acknowledged)
        {
            let _ = self.finish_route_close(Endpoint::Broker);
        }
        for slot in 0..MAX_DYNAMIC_AGENT_ROUTES {
            if self.closing_agents[slot].is_some_and(|closing| closing.acknowledged) {
                let _ = self.finish_agent_close(slot as u8);
            }
        }
    }

    /// Publish one client-authored control record on the route selected by
    /// the arrival pair. The caller supplies the complete bounded frame
    /// scratch so transport backpressure cannot consume a route sequence.
    pub fn send_request(
        &mut self,
        endpoint: Endpoint,
        kind: u16,
        request: RequestId,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        if self.phase != LinkPhase::Active {
            return Err(ServiceError::Link);
        }
        let permitted = match endpoint {
            Endpoint::Workspace => kind == WORKSPACE_REQUEST,
            Endpoint::Broker | Endpoint::Agent(_) => kind == INFER_REQUEST,
        };
        if !permitted {
            return Err(ServiceError::Malformed);
        }
        let route = self
            .bindings
            .for_endpoint(endpoint)
            .ok_or(ServiceError::Route)?
            .route;
        self.routes
            .bound_mut(route)
            .ok_or(ServiceError::Route)?
            .state_mut()
            .open_request(request)
            .map_err(|_| ServiceError::Backpressure)?;
        let result = self.send_routed(endpoint, Plane::Control, kind, request, payload, output);
        if result.is_err()
            && let Some(bound) = self.routes.bound_mut(route)
        {
            let _ = bound.state_mut().close_request(request);
        }
        result
    }

    pub fn send_infer_request(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        self.send_request(endpoint, INFER_REQUEST, request, payload, output)
    }

    /// Broker-only one-way post-commit observation. Unlike inference, this
    /// does not open a request row because no authority-bearing reply exists.
    pub fn send_authority_report(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        if self.phase != LinkPhase::Active
            || endpoint != Endpoint::Broker
            || request.is_zero()
            || decode_authority_report(payload).is_none()
        {
            return Err(ServiceError::Malformed);
        }
        self.send_routed(
            endpoint,
            Plane::Control,
            AUTHORITY_REPORT,
            request,
            payload,
            output,
        )
    }

    /// Cancel one client request and retire every locally staged bulk object
    /// bound to it only after the cancellation frame has been accepted by the
    /// transport. A backpressured send leaves both sequence and staging
    /// untouched, making retry exact.
    pub fn cancel_outbound_request(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
    ) -> Result<(), ServiceError> {
        self.send_small_routed(endpoint, CANCEL, request, &[])?;
        self.outbound
            .retain(|bulk| bulk.endpoint != endpoint || bulk.request != request);
        Ok(())
    }

    pub fn drain(&mut self, control_payload: &mut [u8], bulk_payload: &mut [u8]) -> DrainReport {
        self.retry_agent_closes();
        let mut report = DrainReport::default();
        if self.poisoned {
            // The host closes after flushing REFUSED. Any bytes preceding that
            // close belong to the divergent epoch and are intentionally
            // discarded. Only an observed transport close permits a new Hello.
            match self
                .transport
                .try_recv_control(&mut self.control_input.bytes)
            {
                Err(PortError::Disconnected | PortError::Fault) => {
                    self.poisoned = false;
                    self.disconnect();
                }
                Ok(_) | Err(PortError::Backpressure) => {}
            }
            report.disconnected = true;
            return report;
        }
        if self.pump_handshake().is_err() {
            return report;
        }
        for _ in 0..CONTROL_DRAIN_BUDGET {
            match self.recv_one(Plane::Control, control_payload) {
                Ok(RecvProgress::Complete) => {
                    report.control_frames = report.control_frames.saturating_add(1);
                    if self.poisoned {
                        report.disconnected = true;
                        return report;
                    }
                }
                Ok(RecvProgress::Partial) => continue,
                Ok(RecvProgress::Idle) => break,
                Err(ServiceError::Transport(PortError::Disconnected | PortError::Fault)) => {
                    self.disconnect();
                    report.disconnected = true;
                    return report;
                }
                Err(_) => break,
            }
        }
        // Control is always drained before the fixed bulk quota. In
        // particular, do not inspect an unformatted/resetting shared window
        // between HelloAck and ROUTE_OPENED: the host initializes the new
        // memory epoch before it opens routes.
        if self.phase == LinkPhase::Active {
            for _ in 0..BULK_DRAIN_BUDGET {
                match self.recv_one(Plane::Bulk, bulk_payload) {
                    Ok(RecvProgress::Complete) => {
                        report.bulk_reads = report.bulk_reads.saturating_add(1);
                        report.bulk_frames = report.bulk_frames.saturating_add(1);
                    }
                    Ok(RecvProgress::Partial) => {
                        report.bulk_reads = report.bulk_reads.saturating_add(1);
                    }
                    Ok(RecvProgress::Idle) => break,
                    Err(ServiceError::Transport(PortError::Disconnected | PortError::Fault)) => {
                        self.disconnect();
                        report.disconnected = true;
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        let _ = self.pump_handshake();
        report.shmem_publication = self.shmem_publication;
        report
    }

    fn recv_one(
        &mut self,
        plane: Plane,
        payload_storage: &mut [u8],
    ) -> Result<RecvProgress, ServiceError> {
        let (mut decoder, input) = match plane {
            Plane::Control => {
                if self.control_input.available().is_empty() {
                    let count = self
                        .transport
                        .try_recv_control(&mut self.control_input.bytes)
                        .map_err(ServiceError::Transport)?;
                    self.control_input.end = count;
                }
                (
                    DecoderRef::Control(&mut self.control_decoder),
                    &mut self.control_input,
                )
            }
            Plane::Bulk => {
                if self.bulk_input.available().is_empty() {
                    let count = self
                        .transport
                        .try_recv_bulk(&mut self.bulk_input.bytes)
                        .map_err(ServiceError::Transport)?;
                    self.bulk_input.end = count;
                }
                (
                    DecoderRef::Bulk(&mut self.bulk_decoder),
                    &mut self.bulk_input,
                )
            }
        };
        if input.available().is_empty() {
            return Ok(RecvProgress::Idle);
        }
        let (used, complete) = decoder
            .push(input.available(), payload_storage)
            .map_err(ServiceError::Decode)?;
        input.consume(used);
        if !complete {
            return Ok(RecvProgress::Partial);
        }
        let envelope = decoder.envelope().ok_or(ServiceError::Malformed)?;
        let length = envelope.payload_len.get() as usize;
        decoder.reset();
        match plane {
            Plane::Control => self.handle_control(envelope, &payload_storage[..length])?,
            Plane::Bulk => self.handle_bulk(envelope, &payload_storage[..length])?,
        }
        Ok(RecvProgress::Complete)
    }

    fn handle_control(&mut self, envelope: EnvelopeV1, payload: &[u8]) -> Result<(), ServiceError> {
        if envelope.family.get() != FAMILY_LINK {
            return self.refuse(&envelope, RefusalCode::Malformed);
        }
        if envelope.kind.get() == HELLO {
            return self.accept_hello(envelope, payload);
        }
        if envelope.kind.get() == SHMEM_WAKE {
            return self.accept_shmem_wake(envelope, payload);
        }
        if envelope.kind.get() == ROUTE_OPENED {
            return self.accept_route_opened(envelope, payload);
        }
        if envelope.kind.get() == ROUTE_CLOSED {
            return self.accept_route_closed(envelope, payload);
        }
        if envelope.kind.get() == REFUSED {
            let refusal = decode_refusal(payload).ok_or(ServiceError::Malformed)?;
            let endpoint = self
                .endpoint_for_route(envelope.route_id)
                .ok_or(ServiceError::Route)?;
            // A routed refusal is itself an ordered host message.  Admit and
            // commit it before exposing the status; otherwise the next valid
            // host reply is rejected as out-of-order.  A bulk refusal is
            // transaction-terminal.  The refusal vocabulary intentionally
            // carries only the request (not an authority-bearing BulkId), so
            // retire every outbound transfer owned by that request.
            self.routes
                .process_received(&envelope, |_| Ok::<(), DeliveryError>(()))
                .map_err(|_| ServiceError::Route)?;
            if matches!(
                refusal.for_kind,
                BULK_BEGIN | BULK_CHUNK | BULK_FINISH | BULK_ABORT
            ) {
                self.outbound.retain(|bulk| {
                    bulk.endpoint != endpoint || bulk.request != envelope.request_id
                });
            }
            let _ = self.statuses.push_back(EndpointStatus {
                endpoint,
                request: envelope.request_id,
                reason: StatusReason::Refused(refusal.code),
            });
            if let Some(bound) = self.routes.bound_mut(envelope.route_id) {
                let _ = bound.state_mut().close_request(envelope.request_id);
            }
            // Socket acceptance advanced the guest send cursor, while the
            // host intentionally leaves its receive cursor unchanged for a
            // semantic refusal. Preserve the typed refusal above, then poison
            // and retire this epoch before any later same-epoch transmission.
            self.disconnect();
            self.poisoned = true;
            return Ok(());
        }
        if self.phase != LinkPhase::Active {
            return self.refuse(&envelope, RefusalCode::RouteClosed);
        }
        if envelope.kind.get() == SUPERVISOR_STOP {
            if !payload.is_empty() || !envelope.request_id.is_zero() {
                return self.refuse(&envelope, RefusalCode::Malformed);
            }
            // Route admission authenticates epoch, generation and sequence.
            // A worker or importer cannot request another process's teardown.
            match self.routes.process_received(&envelope, |bound| {
                (*bound.mailbox() == Endpoint::Broker)
                    .then_some(())
                    .ok_or(DeliveryError::Refused)
            }) {
                Ok(()) => self.restart_broker = Some(self.bindings.broker.generation),
                Err(ProcessError::Admit(error)) => {
                    return self.refuse(&envelope, admit_code(error));
                }
                Err(_) => return self.refuse(&envelope, RefusalCode::PolicyDenied),
            }
            return Ok(());
        }
        if matches!(
            envelope.kind.get(),
            BULK_ACCEPTED | BULK_READY | BULK_ABORTED
        ) {
            let endpoint = self
                .endpoint_for_route(envelope.route_id)
                .ok_or(ServiceError::Route)?;
            return self.outbound_ack(endpoint, envelope, payload);
        }
        let delivery = &mut self.delivery;
        let result = self.routes.process_received(&envelope, |bound| {
            let endpoint = *bound.mailbox();
            let kind = envelope.kind.get();
            let permitted = match endpoint {
                Endpoint::Workspace => matches!(kind, WORKSPACE_REPLY | CANCEL),
                Endpoint::Broker | Endpoint::Agent(_) => {
                    matches!(kind, INFER_REPLY | CANCEL)
                }
            } || kind == CANCELLED;
            if !permitted {
                return Err(DeliveryError::Refused);
            }
            if matches!(kind, INFER_REPLY | WORKSPACE_REPLY | CANCELLED)
                && !bound.state().has_request(envelope.request_id)
            {
                return Err(DeliveryError::Refused);
            }
            let terminal = match kind {
                INFER_REPLY => match decode_inference_event(payload) {
                    Some(event) => event.terminal,
                    None if payload
                        .get(..4)
                        .is_some_and(|magic| magic == INFERENCE_EVENT_MAGIC.to_le_bytes()) =>
                    {
                        return Err(DeliveryError::Refused);
                    }
                    // Compatibility for the deterministic witness used by
                    // older boot images. Production hostlinkd emits KIE1.
                    None => true,
                },
                WORKSPACE_REPLY => true,
                CANCELLED => true,
                _ => false,
            };
            if terminal {
                // Check/retire before delivery so an unsolicited terminal
                // reply cannot mutate a founded mailbox. Restore the bounded
                // registry if that mailbox rejects the event transactionally.
                bound
                    .state_mut()
                    .close_request(envelope.request_id)
                    .map_err(|_| DeliveryError::Refused)?;
            }
            let delivered = if envelope.kind.get() == CANCEL {
                delivery.cancel(endpoint, envelope.request_id)
            } else {
                delivery.control(endpoint, envelope.kind.get(), envelope.request_id, payload)
            };
            if delivered.is_err() && terminal {
                let _ = bound.state_mut().open_request(envelope.request_id);
            }
            delivered
        });
        match result {
            Ok(()) if envelope.kind.get() == CANCEL => {
                let endpoint = self
                    .endpoint_for_route(envelope.route_id)
                    .ok_or(ServiceError::Route)?;
                let _ = self.statuses.push_back(EndpointStatus {
                    endpoint,
                    request: envelope.request_id,
                    reason: StatusReason::Cancelled,
                });
                self.send_small_routed(endpoint, CANCELLED, envelope.request_id, &[])
            }
            Ok(()) => Ok(()),
            Err(ProcessError::Admit(error)) => self.refuse(&envelope, admit_code(error)),
            Err(ProcessError::Rejected(DeliveryError::Backpressure)) => {
                self.refuse(&envelope, RefusalCode::Backpressure)
            }
            Err(ProcessError::Rejected(_)) => self.refuse(&envelope, RefusalCode::Malformed),
        }
    }

    fn accept_hello(&mut self, envelope: EnvelopeV1, payload: &[u8]) -> Result<(), ServiceError> {
        if !envelope.link_epoch.is_zero()
            || !envelope.route_id.is_zero()
            || envelope.task_generation.get() != 0
            || envelope.sequence.get() != 0
            || !envelope.request_id.is_zero()
        {
            return self.refuse(&envelope, RefusalCode::Malformed);
        }
        let hello = HelloV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        let epoch = derive_link_epoch(&hello.host_nonce, &self.boot, negotiated_transcript());
        if self.epoch == Some(epoch) {
            // The host retries the same zero-epoch Hello until it observes our
            // acknowledgement because QEMU may connect its chardev before the
            // guest port is ready. A delayed duplicate is idempotent: while
            // opening, re-offer the Ack; once active, it cannot reopen or
            // perturb any route sequence.
            if self.phase != LinkPhase::Active {
                self.ack_pending = true;
                return self.pump_handshake();
            }
            return Ok(());
        }
        if self.phase != LinkPhase::Disconnected {
            self.disconnect();
        }
        self.routes = RouteTable::new();
        for (endpoint, local, role, _, _) in self.bindings.rows() {
            let state = RouteState::new(
                epoch,
                local.route,
                local.generation,
                role,
                16,
                16 * 1024 * 1024,
                64,
            )
            .ok_or(ServiceError::Route)?;
            self.routes
                .bind(state, endpoint)
                .map_err(|_| ServiceError::Route)?;
        }
        self.transport
            .synchronize_link_epoch(Some(epoch))
            .map_err(ServiceError::Transport)?;
        self.epoch = Some(epoch);
        self.shmem_publication = 0;
        self.phase = LinkPhase::Negotiating;
        self.ack_pending = true;
        self.open_cursor = 0;
        self.opened = [false; ROUTES];
        self.inbound.clear();
        self.outbound.clear();
        self.pump_handshake()
    }

    fn accept_shmem_wake(
        &mut self,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let epoch = self.epoch.ok_or(ServiceError::Link)?;
        if !shmem_wake_envelope_is_valid(&envelope, epoch) || payload.len() != SHMEM_WAKE_LEN {
            return Err(ServiceError::Malformed);
        }
        let wake = ShmemWakeV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        let publication = wake.publication().ok_or(ServiceError::Malformed)?;
        if publication < self.shmem_publication {
            return Err(ServiceError::Malformed);
        }
        // Equality is an idempotent retry; a later publication coalesces all
        // earlier edges. Queue data, not this hint, remains authoritative.
        self.shmem_publication = publication;
        Ok(())
    }

    fn pump_handshake(&mut self) -> Result<(), ServiceError> {
        let Some(epoch) = self.epoch else {
            return Ok(());
        };
        if self.ack_pending {
            let payload = HelloAckV1 {
                guest_boot: self.boot,
                epoch,
            };
            match self.send_small_link(HELLO_ACK, LinkEpoch::ZERO, payload.as_bytes()) {
                Ok(()) => self.ack_pending = false,
                Err(ServiceError::Transport(PortError::Backpressure)) => return Ok(()),
                Err(error) => return Err(error),
            }
        }
        let rows = self.bindings.rows();
        while (self.open_cursor as usize) < rows.len() {
            let (endpoint, local, _, role, ordinal) = rows[self.open_cursor as usize];
            if self.opened[endpoint_index(endpoint)] {
                self.open_cursor += 1;
                continue;
            }
            let payload = RouteOpenV1 {
                route_id: local.route,
                task_generation: U64::new(local.generation.get()),
                role,
                ordinal,
            };
            match self.send_small_link(ROUTE_OPEN, epoch, payload.as_bytes()) {
                Ok(()) => self.open_cursor += 1,
                Err(ServiceError::Transport(PortError::Backpressure)) => return Ok(()),
                Err(error) => return Err(error),
            }
        }
        if self.phase == LinkPhase::Negotiating {
            self.phase = LinkPhase::Opening;
        }
        Ok(())
    }

    fn accept_route_opened(
        &mut self,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        if envelope.link_epoch != self.epoch.ok_or(ServiceError::Link)?
            || !envelope.route_id.is_zero()
            || envelope.task_generation.get() != 0
            || envelope.sequence.get() != 0
            || !envelope.request_id.is_zero()
        {
            return self.refuse(&envelope, RefusalCode::Malformed);
        }
        let opened = RouteOpenV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        let endpoint = self
            .bindings
            .rows()
            .iter()
            .find(|(_, local, _, role, ordinal)| {
                local.route == opened.route_id
                    && local.generation.get() == opened.task_generation.get()
                    && *role == opened.role
                    && *ordinal == opened.ordinal
            })
            .map(|row| row.0)
            .ok_or(ServiceError::Route)?;
        let index = endpoint_index(endpoint);
        if self.opened[index] {
            return self.refuse(&envelope, RefusalCode::Duplicate);
        }
        self.opened[index] = true;
        if self
            .bindings
            .rows()
            .iter()
            .all(|(endpoint, _, _, _, _)| self.opened[endpoint_index(*endpoint)])
        {
            self.phase = LinkPhase::Active;
        }
        Ok(())
    }

    fn accept_route_closed(
        &mut self,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        if !payload.is_empty() || !envelope.request_id.is_zero() {
            return self.refuse(&envelope, RefusalCode::Malformed);
        }
        let endpoint = self
            .endpoint_for_route(envelope.route_id)
            .ok_or(ServiceError::Route)?;
        let Some(closing) = self.closing(endpoint) else {
            return self.refuse(&envelope, RefusalCode::RouteClosed);
        };
        if closing.local.route != envelope.route_id
            || closing.local.generation.get() != envelope.task_generation.get()
        {
            return self.refuse(&envelope, RefusalCode::RouteClosed);
        }
        self.routes
            .process_received(&envelope, |_| Ok::<(), DeliveryError>(()))
            .map_err(|_| ServiceError::Route)?;
        self.set_closing(
            endpoint,
            Some(ClosingRoute {
                acknowledged: true,
                ..closing
            }),
        );
        let _ = self.finish_route_close(endpoint);
        Ok(())
    }

    fn handle_bulk(&mut self, envelope: EnvelopeV1, payload: &[u8]) -> Result<(), ServiceError> {
        if self.phase != LinkPhase::Active {
            return self.refuse(&envelope, RefusalCode::RouteClosed);
        }
        let endpoint = self
            .endpoint_for_route(envelope.route_id)
            .ok_or(ServiceError::Route)?;
        let index = self
            .bindings
            .rows()
            .iter()
            .find(|(_, local, _, _, _)| local.route == envelope.route_id)
            .map(|(endpoint, _, _, _, _)| endpoint_index(*endpoint))
            .ok_or(ServiceError::Route)?;
        if !self.opened[index] {
            return self.refuse(&envelope, RefusalCode::RouteClosed);
        }
        self.routes
            .bound_mut(envelope.route_id)
            .ok_or(ServiceError::Route)?
            .state()
            .check_received(&envelope)
            .map_err(|e| {
                let _ = self.refuse(&envelope, admit_code(e));
                ServiceError::Route
            })?;
        match envelope.kind.get() {
            BULK_BEGIN => self.bulk_begin(endpoint, envelope, payload),
            BULK_CHUNK => self.bulk_chunk(endpoint, envelope, payload),
            BULK_FINISH => self.bulk_finish(endpoint, envelope, payload),
            BULK_ABORT => self.bulk_abort(endpoint, envelope, payload),
            BULK_ACCEPTED | BULK_READY | BULK_ABORTED => {
                self.outbound_ack(endpoint, envelope, payload)
            }
            _ => self.refuse(&envelope, RefusalCode::UnknownKind),
        }
    }

    fn bulk_begin(
        &mut self,
        endpoint: Endpoint,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let reference = BulkRefV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        if !reference.shape_is_valid()
            || self.inbound.len() == BULKS
            || self
                .inbound
                .iter()
                .any(|b| b.route == envelope.route_id && b.reference.bulk_id == reference.bulk_id)
        {
            return self.refuse(&envelope, RefusalCode::Backpressure);
        }
        self.delivery
            .bulk_begin(endpoint, envelope.request_id, reference)
            .map_err(|_| ServiceError::Backpressure)?;
        self.inbound
            .push(InboundBulk {
                route: envelope.route_id,
                request: envelope.request_id,
                reference,
                offset: 0,
                chunks: 0,
                hasher: blake3::Hasher::new(),
            })
            .map_err(|_| ServiceError::Backpressure)?;
        self.commit_received(&envelope)?;
        self.send_small_routed(
            endpoint,
            BULK_ACCEPTED,
            envelope.request_id,
            &reference.bulk_id.0,
        )
    }

    fn bulk_chunk(
        &mut self,
        endpoint: Endpoint,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        if payload.len() <= BULK_CHUNK_HEADER_LEN {
            return self.refuse(&envelope, RefusalCode::InvalidLength);
        }
        let header = BulkChunkHeaderV1::read_from_bytes(&payload[..BULK_CHUNK_HEADER_LEN])
            .map_err(|_| ServiceError::Malformed)?;
        let bytes = &payload[BULK_CHUNK_HEADER_LEN..];
        if bytes.len() > FOUNDED_BULK_WIRE_CHUNK {
            return self.refuse(&envelope, RefusalCode::TooLarge);
        }
        let Some(transfer) = self.inbound.iter_mut().find(|b| {
            b.route == envelope.route_id
                && b.request == envelope.request_id
                && b.reference.bulk_id == header.bulk_id
        }) else {
            return self.refuse(&envelope, RefusalCode::SelectorUnknown);
        };
        if header.offset.get() != transfer.offset
            || *blake3::hash(bytes).as_bytes() != header.digest_blake3
        {
            return self.refuse(&envelope, RefusalCode::OutOfOrder);
        }
        let next = transfer
            .offset
            .checked_add(bytes.len() as u32)
            .ok_or(ServiceError::Bulk)?;
        if next > transfer.reference.total_len.get()
            || transfer.chunks == transfer.reference.chunk_count.get()
        {
            return self.refuse(&envelope, RefusalCode::TooLarge);
        }
        self.delivery
            .bulk_chunk(
                endpoint,
                envelope.request_id,
                header.bulk_id,
                transfer.offset,
                bytes,
            )
            .map_err(|_| ServiceError::Backpressure)?;
        transfer.hasher.update(bytes);
        transfer.offset = next;
        transfer.chunks += 1;
        self.commit_received(&envelope)?;
        self.send_small_routed(
            endpoint,
            BULK_ACCEPTED,
            envelope.request_id,
            &header.bulk_id.0,
        )
    }

    fn bulk_finish(
        &mut self,
        endpoint: Endpoint,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let finish = BulkFinishV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        let Some(index) = self.inbound.iter().position(|b| {
            b.route == envelope.route_id
                && b.request == envelope.request_id
                && b.reference.bulk_id == finish.bulk_id
        }) else {
            return self.refuse(&envelope, RefusalCode::SelectorUnknown);
        };
        let transfer = &self.inbound[index];
        if finish.final_total.get() != transfer.offset
            || transfer.offset != transfer.reference.total_len.get()
            || transfer.chunks != transfer.reference.chunk_count.get()
            || finish.digest_blake3 != transfer.reference.digest_blake3
            || finish.digest_blake3 != *transfer.hasher.clone().finalize().as_bytes()
        {
            return self.refuse(&envelope, RefusalCode::InvalidEncoding);
        }
        self.delivery
            .bulk_ready(endpoint, envelope.request_id, finish.bulk_id)
            .map_err(|_| ServiceError::Backpressure)?;
        let workspace_terminal = transfer.reference.content_kind.get() == WORKSPACE_REPLY;
        self.inbound.swap_remove(index);
        self.commit_received(&envelope)?;
        if workspace_terminal && let Some(bound) = self.routes.bound_mut(envelope.route_id) {
            let _ = bound.state_mut().close_request(envelope.request_id);
        }
        self.send_small_routed(endpoint, BULK_READY, envelope.request_id, &finish.bulk_id.0)
    }

    fn bulk_abort(
        &mut self,
        endpoint: Endpoint,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let abort = BulkAbortV1::read_from_bytes(payload).map_err(|_| ServiceError::Malformed)?;
        let reason = RefusalCode::from_wire(abort.reason.get()).ok_or(ServiceError::Malformed)?;
        let Some(index) = self.inbound.iter().position(|b| {
            b.route == envelope.route_id
                && b.request == envelope.request_id
                && b.reference.bulk_id == abort.bulk_id
        }) else {
            return self.refuse(&envelope, RefusalCode::SelectorUnknown);
        };
        self.delivery
            .bulk_abort(endpoint, envelope.request_id, abort.bulk_id, reason);
        self.inbound.swap_remove(index);
        self.commit_received(&envelope)?;
        self.send_small_routed(
            endpoint,
            BULK_ABORTED,
            envelope.request_id,
            &abort.bulk_id.0,
        )
    }

    fn commit_received(&mut self, envelope: &EnvelopeV1) -> Result<(), ServiceError> {
        self.routes
            .bound_mut(envelope.route_id)
            .ok_or(ServiceError::Route)?
            .state_mut()
            .commit_received(envelope)
            .map_err(|_| ServiceError::Route)
    }

    pub fn begin_outbound_bulk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        reference: BulkRefV1,
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        let total = reference.total_len.get();
        let expected_chunks = total.div_ceil(FOUNDED_BULK_WIRE_CHUNK as u32);
        if !reference.shape_is_valid()
            || u32::from(reference.chunk_count.get()) != expected_chunks
            || self.outbound.len() == BULKS
        {
            return Err(ServiceError::Bulk);
        }
        let route = self
            .bindings
            .for_endpoint(endpoint)
            .ok_or(ServiceError::Route)?
            .route;
        let state = self
            .routes
            .bound_mut(route)
            .ok_or(ServiceError::Route)?
            .state_mut();
        let already_open = endpoint == Endpoint::Workspace && state.has_request(request);
        if !already_open {
            state
                .open_request(request)
                .map_err(|_| ServiceError::Backpressure)?;
        }
        if let Err(error) = self.send_routed(
            endpoint,
            Plane::Bulk,
            BULK_BEGIN,
            request,
            reference.as_bytes(),
            output,
        ) {
            if !already_open && let Some(bound) = self.routes.bound_mut(route) {
                let _ = bound.state_mut().close_request(request);
            }
            return Err(error);
        }
        self.outbound
            .push(OutboundBulk {
                endpoint,
                request,
                reference,
                received: 0,
                offset: 0,
                chunks: 0,
                staged: 0,
                staging: [0; FOUNDED_BULK_WIRE_CHUNK],
                hasher: blake3::Hasher::new(),
                phase: OutboundPhase::AwaitBegin,
                credit_bytes: 16 * 1024 * 1024,
                credit_chunks: 64,
                in_flight_bytes: 0,
            })
            .map_err(|_| ServiceError::Backpressure)
    }

    pub fn send_outbound_chunk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        bytes: &[u8],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        let transfer = self
            .outbound
            .iter_mut()
            .find(|b| b.endpoint == endpoint && b.request == request && b.reference.bulk_id == id)
            .ok_or(ServiceError::Bulk)?;
        if transfer.phase != OutboundPhase::Ready || transfer.credit_bytes < bytes.len() as u64 {
            return Err(ServiceError::Backpressure);
        }
        let next = transfer
            .received
            .checked_add(bytes.len() as u32)
            .ok_or(ServiceError::Bulk)?;
        let wire_len = usize::from(transfer.staged)
            .checked_add(bytes.len())
            .ok_or(ServiceError::Bulk)?;
        if bytes.is_empty()
            || bytes.len() > crate::CLIENT_CHUNK_MAX
            || next > transfer.reference.total_len.get()
            || wire_len > FOUNDED_BULK_WIRE_CHUNK
        {
            return Err(ServiceError::Bulk);
        }
        let flush =
            wire_len == FOUNDED_BULK_WIRE_CHUNK || next == transfer.reference.total_len.get();
        if !flush {
            let at = usize::from(transfer.staged);
            transfer.staging[at..wire_len].copy_from_slice(bytes);
            transfer.staged = wire_len as u16;
            transfer.received = next;
            transfer.hasher.update(bytes);
            transfer.credit_bytes -= bytes.len() as u64;
            return Ok(());
        }
        if transfer.credit_chunks == 0 || transfer.chunks == transfer.reference.chunk_count.get() {
            return Err(ServiceError::Backpressure);
        }
        let mut chunk_hasher = blake3::Hasher::new();
        chunk_hasher.update(&transfer.staging[..usize::from(transfer.staged)]);
        chunk_hasher.update(bytes);
        let header = BulkChunkHeaderV1 {
            bulk_id: id,
            offset: U32::new(transfer.offset),
            digest_blake3: *chunk_hasher.finalize().as_bytes(),
        };
        let payload_len = BULK_CHUNK_HEADER_LEN + wire_len;
        let local = self
            .bindings
            .for_endpoint(endpoint)
            .ok_or(ServiceError::Route)?;
        let bound = self
            .routes
            .bound_mut(local.route)
            .ok_or(ServiceError::Route)?;
        let sequence = bound
            .state()
            .send_sequence()
            .map_err(|_| ServiceError::Route)?;
        encode_frame(
            link_envelope(
                BULK_CHUNK,
                self.epoch.ok_or(ServiceError::Link)?,
                local.route,
                local.generation.get(),
                sequence,
                request,
                payload_len,
            ),
            &[
                header.as_bytes(),
                &transfer.staging[..usize::from(transfer.staged)],
                bytes,
            ],
            output,
        )?;
        self.transport
            .try_send_bulk(&output[..ENVELOPE_LEN + payload_len])
            .map_err(ServiceError::Transport)?;
        bound
            .state_mut()
            .commit_sent(sequence)
            .map_err(|_| ServiceError::Route)?;
        let sent_next = transfer
            .offset
            .checked_add(wire_len as u32)
            .ok_or(ServiceError::Bulk)?;
        transfer.hasher.update(bytes);
        transfer.received = next;
        transfer.offset = sent_next;
        transfer.chunks += 1;
        transfer.staged = 0;
        transfer.credit_bytes -= bytes.len() as u64;
        transfer.credit_chunks -= 1;
        transfer.in_flight_bytes = wire_len as u32;
        transfer.phase = OutboundPhase::AwaitChunk;
        Ok(())
    }

    pub fn finish_outbound_bulk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        let index = self
            .outbound
            .iter()
            .position(|b| {
                b.endpoint == endpoint && b.request == request && b.reference.bulk_id == id
            })
            .ok_or(ServiceError::Bulk)?;
        let transfer = &self.outbound[index];
        if transfer.phase != OutboundPhase::Ready {
            return Err(ServiceError::Backpressure);
        }
        let digest = *transfer.hasher.clone().finalize().as_bytes();
        if transfer.offset != transfer.reference.total_len.get()
            || transfer.received != transfer.reference.total_len.get()
            || transfer.staged != 0
            || transfer.chunks != transfer.reference.chunk_count.get()
            || digest != transfer.reference.digest_blake3
        {
            return Err(ServiceError::Bulk);
        }
        let finish = BulkFinishV1 {
            bulk_id: id,
            final_total: U32::new(transfer.offset),
            digest_blake3: digest,
        };
        self.send_routed(
            endpoint,
            Plane::Bulk,
            BULK_FINISH,
            request,
            finish.as_bytes(),
            output,
        )?;
        self.outbound[index].phase = OutboundPhase::AwaitFinish;
        Ok(())
    }

    pub fn finish_outbound_bulk_claimed(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        claimed_total: u32,
        claimed_digest: [u8; 32],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        let transfer = self
            .outbound
            .iter()
            .find(|bulk| {
                bulk.endpoint == endpoint && bulk.request == request && bulk.reference.bulk_id == id
            })
            .ok_or(ServiceError::Bulk)?;
        if claimed_total != transfer.offset
            || claimed_digest != *transfer.hasher.clone().finalize().as_bytes()
        {
            return Err(ServiceError::Malformed);
        }
        self.finish_outbound_bulk(endpoint, request, id, output)
    }

    pub fn abort_outbound_bulk(
        &mut self,
        endpoint: Endpoint,
        request: RequestId,
        id: BulkId,
        reason: RefusalCode,
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        let index = self
            .outbound
            .iter()
            .position(|b| {
                b.endpoint == endpoint && b.request == request && b.reference.bulk_id == id
            })
            .ok_or(ServiceError::Bulk)?;
        // Settle begin/chunk credit acknowledgements before changing the
        // expected reply kind. Otherwise their earlier sequence could be
        // rejected while AwaitAbort, stranding the later abort receipt.
        if self.outbound[index].phase != OutboundPhase::Ready {
            return Err(ServiceError::Backpressure);
        }
        let abort = BulkAbortV1 {
            bulk_id: id,
            reason: U16::new(reason as u16),
        };
        self.send_routed(
            endpoint,
            Plane::Bulk,
            BULK_ABORT,
            request,
            abort.as_bytes(),
            output,
        )?;
        self.outbound[index].phase = OutboundPhase::AwaitAbort;
        Ok(())
    }

    fn outbound_ack(
        &mut self,
        endpoint: Endpoint,
        envelope: EnvelopeV1,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        if payload.len() != 16 {
            return self.refuse(&envelope, RefusalCode::InvalidLength);
        }
        let id = BulkId(payload.try_into().map_err(|_| ServiceError::Malformed)?);
        let Some(index) = self.outbound.iter().position(|bulk| {
            bulk.endpoint == endpoint
                && bulk.request == envelope.request_id
                && bulk.reference.bulk_id == id
        }) else {
            return self.refuse(&envelope, RefusalCode::SelectorUnknown);
        };
        let next = match (envelope.kind.get(), self.outbound[index].phase) {
            (BULK_ACCEPTED, OutboundPhase::AwaitBegin) => Some(OutboundPhase::Ready),
            (BULK_ACCEPTED, OutboundPhase::AwaitChunk) => {
                let transfer = &mut self.outbound[index];
                transfer.credit_bytes = transfer
                    .credit_bytes
                    .saturating_add(u64::from(transfer.in_flight_bytes))
                    .min(16 * 1024 * 1024);
                transfer.credit_chunks = transfer.credit_chunks.saturating_add(1).min(64);
                transfer.in_flight_bytes = 0;
                Some(OutboundPhase::Ready)
            }
            (BULK_READY, OutboundPhase::AwaitFinish)
            | (BULK_ABORTED, OutboundPhase::AwaitAbort) => None,
            _ => return self.refuse(&envelope, RefusalCode::OutOfOrder),
        };
        // Per-chunk ACCEPTED records are internal sender-window credits. They
        // must not flood the founded client's eight-entry control mailbox;
        // only terminal READY/ABORTED is client-visible.
        if envelope.kind.get() != BULK_ACCEPTED {
            self.delivery
                .control(endpoint, envelope.kind.get(), envelope.request_id, payload)
                .map_err(|_| ServiceError::Backpressure)?;
        }
        self.commit_received(&envelope)?;
        if let Some(phase) = next {
            self.outbound[index].phase = phase;
        } else {
            // READY is transport-terminal, but workspace and inference
            // uploads still await their typed application reply on the same
            // request. Retire aborts and opaque bulk users here.
            let awaits_application_reply = envelope.kind.get() == BULK_READY
                && (endpoint == Endpoint::Workspace
                    || (matches!(endpoint, Endpoint::Broker | Endpoint::Agent(_))
                        && self.outbound[index].reference.content_kind.get()
                            == INFERENCE_REQUEST_CONTENT));
            if !awaits_application_reply {
                let route = self
                    .bindings
                    .for_endpoint(endpoint)
                    .ok_or(ServiceError::Route)?
                    .route;
                if let Some(bound) = self.routes.bound_mut(route) {
                    let _ = bound.state_mut().close_request(envelope.request_id);
                }
            }
            self.outbound.swap_remove(index);
        }
        Ok(())
    }

    fn send_routed(
        &mut self,
        endpoint: Endpoint,
        plane: Plane,
        kind: u16,
        request: RequestId,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<(), ServiceError> {
        if self.closing(endpoint).is_some() {
            return Err(ServiceError::Route);
        }
        let local = self
            .bindings
            .for_endpoint(endpoint)
            .ok_or(ServiceError::Route)?;
        let bound = self
            .routes
            .bound_mut(local.route)
            .ok_or(ServiceError::Route)?;
        let sequence = bound
            .state()
            .send_sequence()
            .map_err(|_| ServiceError::Route)?;
        let env = link_envelope(
            kind,
            self.epoch.ok_or(ServiceError::Link)?,
            local.route,
            local.generation.get(),
            sequence,
            request,
            payload.len(),
        );
        encode_frame(env, &[payload], output)?;
        match plane {
            Plane::Control => self
                .transport
                .try_send_control(&output[..ENVELOPE_LEN + payload.len()]),
            Plane::Bulk => self
                .transport
                .try_send_bulk(&output[..ENVELOPE_LEN + payload.len()]),
        }
        .map_err(ServiceError::Transport)?;
        bound
            .state_mut()
            .commit_sent(sequence)
            .map_err(|_| ServiceError::Route)
    }

    fn send_small_routed(
        &mut self,
        endpoint: Endpoint,
        kind: u16,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let mut output = [0; SMALL_FRAME];
        self.send_routed(
            endpoint,
            Plane::Control,
            kind,
            request,
            payload,
            &mut output,
        )
    }

    fn send_small_link(
        &mut self,
        kind: u16,
        epoch: LinkEpoch,
        payload: &[u8],
    ) -> Result<(), ServiceError> {
        let env = link_envelope(
            kind,
            epoch,
            RouteId::ZERO,
            0,
            0,
            RequestId::ZERO,
            payload.len(),
        );
        let mut output = [0; SMALL_FRAME];
        let len = encode_frame(env, &[payload], &mut output)?;
        self.transport
            .try_send_control(&output[..len])
            .map_err(ServiceError::Transport)
    }

    fn refuse(&mut self, source: &EnvelopeV1, code: RefusalCode) -> Result<(), ServiceError> {
        let refusal = RefusalView {
            for_family: FAMILY_LINK,
            for_kind: source.kind.get(),
            code,
            retry: RetryClass::Never,
            retry_after_ms: 0,
            safe_detail: &[],
        };
        let mut payload = [0; REFUSAL_HEADER_LEN];
        let len = refusal
            .encode(&mut payload)
            .ok_or(ServiceError::Malformed)?;
        if let Some(endpoint) = self.endpoint_for_route(source.route_id) {
            self.send_small_routed(endpoint, REFUSED, source.request_id, &payload[..len])
        } else {
            self.send_small_link(
                REFUSED,
                self.epoch.unwrap_or(LinkEpoch::ZERO),
                &payload[..len],
            )
        }
    }

    fn endpoint_for_route(&self, route: RouteId) -> Option<Endpoint> {
        self.bindings
            .rows()
            .iter()
            .find(|(_, local, _, _, _)| local.route == route)
            .map(|row| row.0)
    }

    fn disconnect(&mut self) {
        if self.epoch.is_some() && self.bindings.broker_active {
            // Loss of the host does not kill the desktop. The surviving owner
            // contains the old supervisor before a new incarnation is issued.
            self.restart_broker = Some(self.bindings.broker.generation);
        }
        for (endpoint, local, _, _, _) in self.bindings.rows() {
            if let Some(bound) = self.routes.unbind(local.route) {
                let (mut state, _) = bound.into_parts();
                let requests = state.disconnect();
                if requests.is_empty() {
                    let _ = self.statuses.push_back(EndpointStatus {
                        endpoint,
                        request: RequestId::ZERO,
                        reason: StatusReason::HostDisconnected,
                    });
                }
                for request in requests {
                    let _ = self.statuses.push_back(EndpointStatus {
                        endpoint,
                        request,
                        reason: StatusReason::HostDisconnected,
                    });
                }
            }
        }
        if let Some(closing) = self.closing_broker {
            self.closing_broker = Some(ClosingRoute {
                acknowledged: true,
                ..closing
            });
        }
        for slot in 0..MAX_DYNAMIC_AGENT_ROUTES {
            if let Some(closing) = self.closing_agents[slot] {
                self.closing_agents[slot] = Some(ClosingRoute {
                    acknowledged: true,
                    ..closing
                });
            }
        }
        self.retry_agent_closes();
        self.phase = LinkPhase::Disconnected;
        let _ = self.transport.synchronize_link_epoch(None);
        self.epoch = None;
        self.shmem_publication = 0;
        self.ack_pending = false;
        self.open_cursor = 0;
        self.opened = [false; ROUTES];
        self.inbound.clear();
        self.outbound.clear();
        self.control_decoder.reset();
        self.bulk_decoder.reset();
        self.control_input.clear();
        self.bulk_input.clear();
    }
}

const fn endpoint_index(endpoint: Endpoint) -> usize {
    match endpoint {
        Endpoint::Broker => 0,
        Endpoint::Agent(slot) => slot as usize + 1,
        Endpoint::Workspace => ROUTES - 1,
    }
}

fn encode_frame(
    envelope: EnvelopeV1,
    parts: &[&[u8]],
    output: &mut [u8],
) -> Result<usize, ServiceError> {
    let len = envelope.frame_len().ok_or(ServiceError::Malformed)?;
    let out = output.get_mut(..len).ok_or(ServiceError::Backpressure)?;
    out[..ENVELOPE_LEN].copy_from_slice(envelope.as_bytes());
    let mut at = ENVELOPE_LEN;
    for part in parts {
        out[at..at + part.len()].copy_from_slice(part);
        at += part.len();
    }
    if at != len {
        return Err(ServiceError::Malformed);
    }
    Ok(len)
}

fn admit_code(error: AdmitError) -> RefusalCode {
    match error {
        AdmitError::Route(RouteError::EpochStale) => RefusalCode::EpochStale,
        AdmitError::Route(RouteError::RouteUnknown) => RefusalCode::RouteUnknown,
        AdmitError::Route(RouteError::GenerationStale) => RefusalCode::GenerationStale,
        AdmitError::Route(RouteError::RouteClosed) => RefusalCode::RouteClosed,
        AdmitError::Route(RouteError::LinkControlOnRoute) => RefusalCode::Malformed,
        AdmitError::Sequence(SequenceDisposition::Duplicate) => RefusalCode::Duplicate,
        AdmitError::Sequence(SequenceDisposition::OutOfOrder) => RefusalCode::OutOfOrder,
        AdmitError::Sequence(SequenceDisposition::Exhausted) => RefusalCode::CapacityExhausted,
        AdmitError::Sequence(SequenceDisposition::Next) => RefusalCode::Internal,
    }
}

enum DecoderRef<'a> {
    Control(&'a mut ControlFrameDecoder),
    Bulk(&'a mut BulkFrameDecoder),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecvProgress {
    Idle,
    Partial,
    Complete,
}
impl DecoderRef<'_> {
    fn push(&mut self, input: &[u8], storage: &mut [u8]) -> Result<(usize, bool), DecodeError> {
        let progress = match self {
            Self::Control(decoder) => decoder.push(input, storage)?,
            Self::Bulk(decoder) => decoder.push(input, storage)?,
        };
        Ok((progress.consumed, progress.complete))
    }
    fn envelope(&self) -> Option<EnvelopeV1> {
        match self {
            Self::Control(decoder) => decoder.envelope().copied(),
            Self::Bulk(decoder) => decoder.envelope().copied(),
        }
    }
    fn reset(self) {
        match self {
            Self::Control(decoder) => decoder.reset(),
            Self::Bulk(decoder) => decoder.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::vec;
    use std::vec::Vec as StdVec;

    #[derive(Default)]
    struct MockTransport {
        control_in: VecDeque<StdVec<u8>>,
        bulk_in: VecDeque<StdVec<u8>>,
        control_out: StdVec<StdVec<u8>>,
        bulk_out: StdVec<StdVec<u8>>,
        disconnect_control: bool,
        backpressure_control: bool,
        backpressure_bulk: bool,
        bulk_reads: usize,
    }

    impl HostLinkTransport for MockTransport {
        fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError> {
            if self.backpressure_control {
                self.backpressure_control = false;
                return Err(PortError::Backpressure);
            }
            self.control_out.push(frame.to_vec());
            Ok(())
        }
        fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
            if self.disconnect_control {
                self.disconnect_control = false;
                return Err(PortError::Disconnected);
            }
            pop(&mut self.control_in, output)
        }
        fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError> {
            if self.backpressure_bulk {
                self.backpressure_bulk = false;
                return Err(PortError::Backpressure);
            }
            self.bulk_out.push(frame.to_vec());
            Ok(())
        }
        fn try_recv_bulk(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
            self.bulk_reads += 1;
            pop(&mut self.bulk_in, output)
        }
    }

    fn pop(queue: &mut VecDeque<StdVec<u8>>, output: &mut [u8]) -> Result<usize, PortError> {
        let Some(frame) = queue.pop_front() else {
            return Err(PortError::Backpressure);
        };
        output[..frame.len()].copy_from_slice(&frame);
        Ok(frame.len())
    }

    #[derive(Default)]
    struct Mailboxes {
        controls: StdVec<(Endpoint, u16, RequestId, StdVec<u8>)>,
        bulk: StdVec<(Endpoint, u16, BulkId)>,
        cancelled: StdVec<(Endpoint, RequestId)>,
    }

    impl MailboxDelivery for Mailboxes {
        fn control(
            &mut self,
            endpoint: Endpoint,
            kind: u16,
            request: RequestId,
            payload: &[u8],
        ) -> Result<(), DeliveryError> {
            self.controls
                .push((endpoint, kind, request, payload.to_vec()));
            Ok(())
        }
        fn bulk_begin(
            &mut self,
            endpoint: Endpoint,
            _: RequestId,
            reference: BulkRefV1,
        ) -> Result<(), DeliveryError> {
            self.bulk.push((endpoint, BULK_BEGIN, reference.bulk_id));
            Ok(())
        }
        fn bulk_chunk(
            &mut self,
            endpoint: Endpoint,
            _: RequestId,
            id: BulkId,
            _: u32,
            _: &[u8],
        ) -> Result<(), DeliveryError> {
            self.bulk.push((endpoint, BULK_CHUNK, id));
            Ok(())
        }
        fn bulk_ready(
            &mut self,
            endpoint: Endpoint,
            _: RequestId,
            id: BulkId,
        ) -> Result<(), DeliveryError> {
            self.bulk.push((endpoint, BULK_READY, id));
            Ok(())
        }
        fn bulk_abort(&mut self, endpoint: Endpoint, _: RequestId, id: BulkId, _: RefusalCode) {
            self.bulk.push((endpoint, BULK_ABORTED, id));
        }
        fn cancel(&mut self, endpoint: Endpoint, request: RequestId) -> Result<(), DeliveryError> {
            self.cancelled.push((endpoint, request));
            Ok(())
        }
    }

    type Service = HostLinkService<MockTransport, Mailboxes>;

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }
    fn bindings() -> Bindings {
        let mut bindings = Bindings::new(
            LocalRoute {
                route: RouteId(id(2)),
                generation: TaskGeneration::new(7).unwrap(),
            },
            LocalRoute {
                route: RouteId(id(5)),
                generation: TaskGeneration::new(10).unwrap(),
            },
        )
        .unwrap();
        assert!(bindings.install_agent(
            0,
            LocalRoute {
                route: RouteId(id(3)),
                generation: TaskGeneration::new(8).unwrap(),
            }
        ));
        assert!(bindings.install_agent(
            1,
            LocalRoute {
                route: RouteId(id(4)),
                generation: TaskGeneration::new(9).unwrap(),
            }
        ));
        assert!(bindings.install_agent(
            2,
            LocalRoute {
                route: RouteId(id(6)),
                generation: TaskGeneration::new(11).unwrap(),
            }
        ));
        assert!(bindings.install_agent(
            3,
            LocalRoute {
                route: RouteId(id(7)),
                generation: TaskGeneration::new(12).unwrap(),
            }
        ));
        bindings
    }

    fn bytes(envelope: EnvelopeV1, payload: &[u8]) -> StdVec<u8> {
        let mut out = vec![0; ENVELOPE_LEN + payload.len()];
        encode_frame(envelope, &[payload], &mut out).unwrap();
        out
    }

    fn activate(nonce: u8) -> Service {
        let mut service = Service::new(
            MockTransport::default(),
            Mailboxes::default(),
            [99; 32],
            bindings(),
        );
        let hello = HelloV1 {
            host_nonce: [nonce; 32],
        };
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                HELLO,
                LinkEpoch::ZERO,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                hello.as_bytes().len(),
            ),
            hello.as_bytes(),
        ));
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        let epoch = service.epoch().unwrap();
        for (_, local, _, role, ordinal) in bindings().rows() {
            let opened = RouteOpenV1 {
                route_id: local.route,
                task_generation: U64::new(local.generation.get()),
                role,
                ordinal,
            };
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    ROUTE_OPENED,
                    epoch,
                    RouteId::ZERO,
                    0,
                    0,
                    RequestId::ZERO,
                    opened.as_bytes().len(),
                ),
                opened.as_bytes(),
            ));
        }
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.phase(), LinkPhase::Active);
        service.transport_mut().control_out.clear();
        service
    }

    fn wake_frame(epoch: LinkEpoch, publication: u64) -> StdVec<u8> {
        let wake = ShmemWakeV1::new(publication).unwrap();
        bytes(
            link_envelope(
                SHMEM_WAKE,
                epoch,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                SHMEM_WAKE_LEN,
            ),
            wake.as_bytes(),
        )
    }

    #[test]
    fn shared_wakes_coalesce_idempotently_without_entering_route_state() {
        let mut service = activate(42);
        let epoch = service.epoch().unwrap();
        for publication in [1, 7, 7] {
            service
                .transport_mut()
                .control_in
                .push_back(wake_frame(epoch, publication));
        }
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        let report = service.drain(&mut control, &mut bulk);
        assert_eq!(report.control_frames, 3);
        assert_eq!(report.shmem_publication, 7);
        assert_eq!(service.shmem_publication(), 7);
        assert!(service.delivery().controls.is_empty());
    }

    #[test]
    fn stale_or_malformed_shared_wake_never_moves_the_observed_edge() {
        let mut service = activate(43);
        let epoch = service.epoch().unwrap();
        let valid = ShmemWakeV1::new(9).unwrap();
        let envelope = link_envelope(
            SHMEM_WAKE,
            epoch,
            RouteId::ZERO,
            0,
            0,
            RequestId::ZERO,
            SHMEM_WAKE_LEN,
        );
        assert_eq!(
            service.accept_shmem_wake(envelope, valid.as_bytes()),
            Ok(())
        );
        assert_eq!(service.shmem_publication(), 9);

        let stale = ShmemWakeV1::new(8).unwrap();
        assert_eq!(
            service.accept_shmem_wake(envelope, stale.as_bytes()),
            Err(ServiceError::Malformed)
        );
        let mut routed = envelope;
        routed.route_id = RouteId(id(3));
        assert_eq!(
            service.accept_shmem_wake(routed, valid.as_bytes()),
            Err(ServiceError::Malformed)
        );
        assert_eq!(service.shmem_publication(), 9);
    }

    #[test]
    fn four_agents_only_receive_through_their_structural_endpoint() {
        let mut service = activate(1);
        let epoch = service.epoch().unwrap();
        let mut output = [0; 4096];
        for (endpoint, route, generation, payload) in [
            (
                Endpoint::Agent(0),
                RouteId(id(3)),
                8,
                b"agent-zero".as_slice(),
            ),
            (
                Endpoint::Agent(1),
                RouteId(id(4)),
                9,
                b"agent-one".as_slice(),
            ),
            (
                Endpoint::Agent(2),
                RouteId(id(6)),
                11,
                b"agent-two".as_slice(),
            ),
            (
                Endpoint::Agent(3),
                RouteId(id(7)),
                12,
                b"agent-three".as_slice(),
            ),
        ] {
            let request = RequestId(id(generation as u8));
            service
                .send_infer_request(endpoint, request, b"pending", &mut output)
                .unwrap();
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    INFER_REPLY,
                    epoch,
                    route,
                    generation,
                    0,
                    request,
                    payload.len(),
                ),
                payload,
            ));
        }
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service.drain(&mut c, &mut b);
        assert_eq!(service.delivery().controls[0].0, Endpoint::Agent(0));
        assert_eq!(service.delivery().controls[1].0, Endpoint::Agent(1));
        assert_eq!(service.delivery().controls[0].3, b"agent-zero");
        assert_eq!(service.delivery().controls[1].3, b"agent-one");
        assert_eq!(service.delivery().controls[2].0, Endpoint::Agent(2));
        assert_eq!(service.delivery().controls[3].0, Endpoint::Agent(3));
        assert_eq!(service.delivery().controls[2].3, b"agent-two");
        assert_eq!(service.delivery().controls[3].3, b"agent-three");
    }

    #[test]
    fn inference_failure_keeps_the_epoch_and_other_request_live() {
        let mut service = activate(1);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(44));
        let other = RequestId(id(45));
        let mut output = [0; 4096];
        service
            .send_infer_request(Endpoint::Broker, request, b"plan", &mut output)
            .unwrap();
        service
            .send_infer_request(Endpoint::Agent(0), other, b"other", &mut output)
            .unwrap();
        let mut reason = [0; REFUSAL_HEADER_LEN];
        let len = RefusalView {
            for_family: FAMILY_LINK,
            for_kind: INFER_REQUEST,
            code: RefusalCode::ModelFailed,
            retry: RetryClass::NewRequest,
            retry_after_ms: 0,
            safe_detail: &[],
        }
        .encode(&mut reason)
        .unwrap();
        let mut payload = [0; INFERENCE_EVENT_HEADER_LEN + REFUSAL_HEADER_LEN];
        let length = InferenceEventView {
            kind: InferenceEventKind::TurnFailed,
            terminal: true,
            part: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            turn: 1,
            payload: &reason[..len],
        }
        .encode(&mut payload)
        .unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(INFER_REPLY, epoch, RouteId(id(2)), 7, 0, request, length),
            &payload[..length],
        ));
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        assert_eq!(service.drain(&mut control, &mut bulk).control_frames, 1);
        assert_eq!(service.phase(), LinkPhase::Active);
        assert_eq!(service.epoch(), Some(epoch));
        assert!(!service.poisoned);
        assert_eq!(
            decode_inference_event(&service.delivery().controls[0].3)
                .unwrap()
                .kind,
            InferenceEventKind::TurnFailed
        );
        assert!(
            !service
                .routes
                .bound_mut(RouteId(id(2)))
                .unwrap()
                .state()
                .has_request(request)
        );
        assert!(
            service
                .routes
                .bound_mut(RouteId(id(3)))
                .unwrap()
                .state()
                .has_request(other)
        );
        // The broker still has a usable face on which to publish its frontend
        // failure and perform cleanup; no reconnect/founding replacement needed.
        service
            .send_infer_request(
                Endpoint::Broker,
                RequestId(id(46)),
                b"failure result",
                &mut output,
            )
            .unwrap();
    }

    #[test]
    fn streamed_parts_do_not_retire_a_request_before_terminal_completion() {
        let mut service = activate(1);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(44));
        let mut output = [0; 4096];
        service
            .send_infer_request(Endpoint::Agent(0), request, b"pending", &mut output)
            .unwrap();

        for (sequence, kind, terminal, part, body) in [
            (
                0,
                InferenceEventKind::AssistantPart,
                false,
                0,
                b"part".as_slice(),
            ),
            (
                1,
                InferenceEventKind::TurnFinished,
                true,
                1,
                b"done".as_slice(),
            ),
        ] {
            let mut payload = [0; INFERENCE_EVENT_HEADER_LEN + INFERENCE_PART_MAX];
            let length = InferenceEventView {
                kind,
                terminal,
                part,
                prompt_tokens: 2,
                completion_tokens: u32::from(terminal),
                turn: 7,
                payload: body,
            }
            .encode(&mut payload)
            .unwrap();
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    INFER_REPLY,
                    epoch,
                    RouteId(id(3)),
                    8,
                    sequence,
                    request,
                    length,
                ),
                &payload[..length],
            ));
        }
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        assert_eq!(service.drain(&mut control, &mut bulk).control_frames, 2);
        assert_eq!(service.delivery().controls.len(), 2);
        assert!(
            !decode_inference_event(&service.delivery().controls[0].3)
                .unwrap()
                .terminal
        );
        assert!(
            decode_inference_event(&service.delivery().controls[1].3)
                .unwrap()
                .terminal
        );
        assert_eq!(service.phase(), LinkPhase::Active);
    }

    #[test]
    fn duplicate_pending_hello_is_idempotent() {
        let mut service = Service::new(
            MockTransport::default(),
            Mailboxes::default(),
            [99; 32],
            bindings(),
        );
        let hello = HelloV1 {
            host_nonce: [17; 32],
        };
        let frame = bytes(
            link_envelope(
                HELLO,
                LinkEpoch::ZERO,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                hello.as_bytes().len(),
            ),
            hello.as_bytes(),
        );
        service.transport_mut().control_in.push_back(frame.clone());
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        let epoch = service.epoch().unwrap();
        assert_eq!(service.phase(), LinkPhase::Opening);

        service.transport_mut().control_out.clear();
        service.transport_mut().control_in.push_back(frame);
        service.drain(&mut control, &mut bulk);

        assert_eq!(service.epoch(), Some(epoch));
        assert_eq!(service.phase(), LinkPhase::Opening);
        assert_eq!(service.transport_mut().control_out.len(), 1);
        let ack =
            EnvelopeV1::read_from_bytes(&service.transport_mut().control_out[0][..ENVELOPE_LEN])
                .unwrap();
        assert_eq!(ack.kind.get(), HELLO_ACK);
    }

    #[test]
    fn opening_routes_never_reads_a_not_yet_reset_shared_plane() {
        let mut service = Service::new(
            MockTransport::default(),
            Mailboxes::default(),
            [99; 32],
            bindings(),
        );
        let hello = HelloV1 {
            host_nonce: [18; 32],
        };
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                HELLO,
                LinkEpoch::ZERO,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                hello.as_bytes().len(),
            ),
            hello.as_bytes(),
        ));
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.phase(), LinkPhase::Opening);
        assert_eq!(service.transport_mut().bulk_reads, 0);
    }

    #[test]
    fn forged_stale_duplicate_and_out_of_order_frames_fail_closed() {
        let mut service = activate(2);
        let epoch = service.epoch().unwrap();
        let mut output = [0; 4096];
        service
            .send_infer_request(
                Endpoint::Agent(0),
                RequestId(id(4)),
                b"pending",
                &mut output,
            )
            .unwrap();
        let frames = [
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(8)),
                8,
                0,
                RequestId(id(1)),
                0,
            ),
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                99,
                0,
                RequestId(id(2)),
                0,
            ),
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                1,
                RequestId(id(3)),
                0,
            ),
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                0,
                RequestId(id(4)),
                0,
            ),
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                0,
                RequestId(id(5)),
                0,
            ),
        ];
        for frame in frames {
            service
                .transport_mut()
                .control_in
                .push_back(bytes(frame, &[]));
        }
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service.drain(&mut c, &mut b);
        assert_eq!(service.delivery().controls.len(), 1);
        let codes: StdVec<_> = service
            .transport_mut()
            .control_out
            .iter()
            .filter_map(|frame| {
                let env = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).ok()?;
                (env.kind.get() == REFUSED)
                    .then(|| decode_refusal(&frame[ENVELOPE_LEN..]).unwrap().code)
            })
            .collect();
        assert_eq!(
            codes,
            [
                RefusalCode::RouteUnknown,
                RefusalCode::GenerationStale,
                RefusalCode::OutOfOrder,
                RefusalCode::Duplicate
            ]
        );
    }

    #[test]
    fn fragmented_and_coalesced_stream_reads_preserve_frame_boundaries() {
        let mut service = activate(7);
        let epoch = service.epoch().unwrap();
        let mut output = [0; 4096];
        for request in [RequestId(id(30)), RequestId(id(31)), RequestId(id(32))] {
            service
                .send_infer_request(Endpoint::Agent(0), request, b"pending", &mut output)
                .unwrap();
        }
        let first = bytes(
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                0,
                RequestId(id(30)),
                1,
            ),
            b"a",
        );
        let second = bytes(
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                1,
                RequestId(id(31)),
                1,
            ),
            b"b",
        );
        let mut coalesced = first;
        coalesced.extend_from_slice(&second);
        service.transport_mut().control_in.push_back(coalesced);
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        assert_eq!(service.drain(&mut c, &mut b).control_frames, 2);

        let fragmented = bytes(
            link_envelope(
                INFER_REPLY,
                epoch,
                RouteId(id(3)),
                8,
                2,
                RequestId(id(32)),
                3,
            ),
            b"xyz",
        );
        service
            .transport_mut()
            .control_in
            .push_back(fragmented[..17].to_vec());
        service
            .transport_mut()
            .control_in
            .push_back(fragmented[17..].to_vec());
        // One bounded drain continues across partial transport reads, so a
        // consumed wake cannot strand the remainder before park.
        assert_eq!(service.drain(&mut c, &mut b).control_frames, 1);
        assert_eq!(service.drain(&mut c, &mut b).control_frames, 0);
        assert_eq!(service.delivery().controls.len(), 3);
    }

    #[test]
    fn unknown_bulk_identity_gets_typed_refusal() {
        let mut service = activate(8);
        let epoch = service.epoch().unwrap();
        let header = BulkChunkHeaderV1 {
            bulk_id: BulkId(id(77)),
            offset: U32::ZERO,
            digest_blake3: *blake3::hash(b"x").as_bytes(),
        };
        let mut payload = header.as_bytes().to_vec();
        payload.push(b'x');
        service.transport_mut().bulk_in.push_back(bytes(
            link_envelope(
                BULK_CHUNK,
                epoch,
                RouteId(id(3)),
                8,
                0,
                RequestId(id(76)),
                payload.len(),
            ),
            &payload,
        ));
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service.drain(&mut c, &mut b);
        let refusal = service
            .transport_mut()
            .control_out
            .iter()
            .find_map(|frame| {
                let env = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).ok()?;
                (env.kind.get() == REFUSED)
                    .then(|| decode_refusal(&frame[ENVELOPE_LEN..]).unwrap().code)
            });
        assert_eq!(refusal, Some(RefusalCode::SelectorUnknown));
    }

    #[test]
    fn partial_bulk_reads_report_saturation_until_the_final_empty_recheck() {
        let mut service = activate(81);
        let epoch = service.epoch().unwrap();
        let header = BulkChunkHeaderV1 {
            bulk_id: BulkId(id(82)),
            offset: U32::ZERO,
            digest_blake3: *blake3::hash(b"chunk").as_bytes(),
        };
        let mut payload = header.as_bytes().to_vec();
        payload.extend_from_slice(b"chunk");
        let frame = bytes(
            link_envelope(
                BULK_CHUNK,
                epoch,
                RouteId(id(3)),
                8,
                0,
                RequestId(id(83)),
                payload.len(),
            ),
            &payload,
        );
        service
            .transport_mut()
            .bulk_in
            .push_back(frame[..17].to_vec());
        service
            .transport_mut()
            .bulk_in
            .push_back(frame[17..41].to_vec());
        service
            .transport_mut()
            .bulk_in
            .push_back(frame[41..].to_vec());
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        let saturated = service.drain(&mut control, &mut bulk);
        assert_eq!(saturated.bulk_reads, BULK_DRAIN_BUDGET as u8);
        assert_eq!(saturated.bulk_frames, 0);
        let completed = service.drain(&mut control, &mut bulk);
        assert_eq!(completed.bulk_reads, 1);
        assert_eq!(completed.bulk_frames, 1);
        let empty = service.drain(&mut control, &mut bulk);
        assert_eq!(empty.bulk_reads, 0);
    }

    #[test]
    fn duplicate_route_opened_is_refused() {
        let mut service = activate(9);
        let epoch = service.epoch().unwrap();
        let local = bindings().for_endpoint(Endpoint::Agent(0)).unwrap();
        let opened = RouteOpenV1 {
            route_id: local.route,
            task_generation: U64::new(local.generation.get()),
            role: 1,
            ordinal: 0,
        };
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_OPENED,
                epoch,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                opened.as_bytes().len(),
            ),
            opened.as_bytes(),
        ));
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service.drain(&mut c, &mut b);
        let refusal = service
            .transport_mut()
            .control_out
            .iter()
            .find_map(|frame| {
                let env = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).ok()?;
                (env.kind.get() == REFUSED)
                    .then(|| decode_refusal(&frame[ENVELOPE_LEN..]).unwrap().code)
            });
        assert_eq!(refusal, Some(RefusalCode::Duplicate));
    }

    #[test]
    fn dynamic_route_reuse_closes_old_generation_before_returning_capacity() {
        let mut service = activate(51);
        let first = LocalRoute {
            route: RouteId(id(61)),
            generation: TaskGeneration::new(11).unwrap(),
        };
        service.install_agent_route(4, first).unwrap();
        assert_eq!(service.phase(), LinkPhase::Opening);
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        let opened = RouteOpenV1 {
            route_id: first.route,
            task_generation: U64::new(first.generation.get()),
            role: 1,
            ordinal: 4,
        };
        let epoch = service.epoch().unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_OPENED,
                epoch,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                opened.as_bytes().len(),
            ),
            opened.as_bytes(),
        ));
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.phase(), LinkPhase::Active);
        assert_eq!(
            service.close_agent_route(4, first.generation),
            Err(ServiceError::Backpressure)
        );
        assert_eq!(
            service.install_agent_route(
                4,
                LocalRoute {
                    route: RouteId(id(62)),
                    generation: TaskGeneration::new(12).unwrap(),
                }
            ),
            Err(ServiceError::Route)
        );
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_CLOSED,
                epoch,
                first.route,
                first.generation.get(),
                0,
                RequestId::ZERO,
                0,
            ),
            &[],
        ));
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.close_agent_route(4, first.generation), Ok(()));

        let replacement = LocalRoute {
            route: RouteId(id(62)),
            generation: TaskGeneration::new(12).unwrap(),
        };
        service.install_agent_route(4, replacement).unwrap();
        assert_eq!(
            service.close_agent_route(4, first.generation),
            Err(ServiceError::Route)
        );
        assert_eq!(
            service.bindings.for_endpoint(Endpoint::Agent(4)).unwrap(),
            replacement
        );
    }

    #[test]
    fn acknowledged_close_drops_old_generation_work_but_preserves_other_routes() {
        let mut service = activate(54);
        let old = service.bindings.for_endpoint(Endpoint::Agent(0)).unwrap();
        let epoch = service.epoch().unwrap();
        let mut output = [0; 4096];
        for (endpoint, byte) in [(Endpoint::Agent(0), 71), (Endpoint::Broker, 72)] {
            let reference = BulkRefV1 {
                bulk_id: BulkId(id(byte)),
                total_len: U32::new(3),
                chunk_count: U16::new(1),
                content_kind: U16::new(1),
                digest_blake3: *blake3::hash(b"abc").as_bytes(),
            };
            service
                .begin_outbound_bulk(endpoint, RequestId(id(byte)), reference, &mut output)
                .unwrap();
            service
                .statuses
                .push_back(EndpointStatus {
                    endpoint,
                    request: RequestId(id(byte)),
                    reason: StatusReason::Cancelled,
                })
                .unwrap();
        }
        assert_eq!(
            service.close_agent_route(0, old.generation),
            Err(ServiceError::Backpressure)
        );
        let sent = service.transport_mut().control_out.len();
        assert_eq!(
            service.send_infer_request(Endpoint::Agent(0), RequestId(id(73)), b"late", &mut output),
            Err(ServiceError::Route)
        );
        assert_eq!(service.transport_mut().control_out.len(), sent);
        assert_eq!(
            service.outbound.len(),
            2,
            "unacknowledged close retains ownership"
        );
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_CLOSED,
                epoch,
                old.route,
                old.generation.get(),
                0,
                RequestId::ZERO,
                0,
            ),
            &[],
        ));
        let mut bulk = [0; 4096];
        service.drain(&mut output, &mut bulk);
        assert_eq!(service.close_agent_route(0, old.generation), Ok(()));
        assert_eq!(service.outbound.len(), 1);
        assert_eq!(service.outbound[0].endpoint, Endpoint::Broker);
        assert_eq!(service.pop_status().unwrap().endpoint, Endpoint::Broker);
        assert_eq!(service.pop_status(), None);
    }

    #[test]
    fn supervisor_stop_is_exact_broker_control_and_disconnect_requests_containment() {
        for endpoint in [Endpoint::Broker, Endpoint::Agent(0), Endpoint::Workspace] {
            let mut service = activate(58);
            let local = service.bindings.for_endpoint(endpoint).unwrap();
            let broker = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
            let envelope = link_envelope(
                SUPERVISOR_STOP,
                service.epoch().unwrap(),
                local.route,
                local.generation.get(),
                0,
                RequestId::ZERO,
                0,
            );
            let _ = service.handle_control(envelope, &[]);
            assert_eq!(
                service.supervisor_restart_requested(broker.generation),
                endpoint == Endpoint::Broker
            );
            assert!(service.bindings.for_endpoint(Endpoint::Workspace).is_some());
        }
        let mut service = activate(58);
        let local = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
        let stale = link_envelope(
            SUPERVISOR_STOP,
            service.epoch().unwrap(),
            local.route,
            local.generation.get() + 1,
            0,
            RequestId::ZERO,
            0,
        );
        let _ = service.handle_control(stale, &[]);
        assert!(!service.supervisor_restart_requested(local.generation));
        service.disconnect();
        assert!(service.supervisor_restart_requested(local.generation));
    }

    #[test]
    fn supervisor_owner_requires_request_drain_and_reuses_only_successor_generations() {
        use crate::{SupervisorOperation as Op, SupervisorOwner, SupervisorPhase as Phase};
        let first = TaskGeneration::new(1).unwrap();
        let bindings = Bindings::new(
            LocalRoute {
                route: RouteId(id(1)),
                generation: first,
            },
            LocalRoute {
                route: RouteId(id(2)),
                generation: first,
            },
        )
        .unwrap();
        let mut service = HostLinkService::new(
            MockTransport::default(),
            crate::MailboxBank::new(),
            [59; 32],
            bindings,
        );
        for slot in 0..2 {
            service
                .install_agent_route(
                    slot,
                    LocalRoute {
                        route: RouteId(id(60 + slot)),
                        generation: first,
                    },
                )
                .unwrap();
        }
        let initial = service.generation(Endpoint::Broker).unwrap();
        let mut owner = SupervisorOwner::new(initial);
        let workspace = service.generation(Endpoint::Workspace);
        for _ in 0..3 {
            let old = owner.state();
            assert_eq!(
                owner.operate(&mut service, Op::Open, old.incarnation.get() + 1),
                Err(ServiceError::Route)
            );
            assert_eq!(
                owner
                    .operate(&mut service, Op::Close, old.incarnation.get())
                    .unwrap()
                    .phase,
                Phase::Closing
            );
            assert!(!owner.client_active());
            assert!(owner.needs_request_drain());
            assert_eq!(
                owner.operate(&mut service, Op::Open, old.incarnation.get() + 1),
                Err(ServiceError::Route)
            );
            assert!(owner.confirm_request_drain());
            assert!(!owner.confirm_request_drain());
            let next = owner
                .operate(&mut service, Op::Open, old.incarnation.get() + 1)
                .unwrap();
            assert_eq!(next.phase, Phase::Available);
            for (next, old) in next.workers.iter().zip(old.workers) {
                assert_eq!(next.get(), old.get() + 1);
            }
            assert_eq!(
                owner.operate(&mut service, Op::Open, next.incarnation.get()),
                Ok(next)
            );
            assert_eq!(
                owner.operate(&mut service, Op::Close, old.incarnation.get()),
                Err(ServiceError::Route)
            );
            assert_eq!(service.generation(Endpoint::Workspace), workspace);
        }
    }

    #[test]
    fn supervisor_broker_replacement_waits_for_close_ack_and_preserves_workspace() {
        let mut service = activate(59);
        let old = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
        let workspace = service.bindings.for_endpoint(Endpoint::Workspace).unwrap();
        let next = TaskGeneration::new(old.generation.get() + 1).unwrap();
        let epoch = service.epoch().unwrap();
        assert_eq!(service.rotate_broker_route(next), Err(ServiceError::Route));
        assert_eq!(
            service.close_broker_route(old.generation),
            Err(ServiceError::Backpressure)
        );
        assert_eq!(service.rotate_broker_route(next), Err(ServiceError::Route));
        let mut output = [0; 4096];
        let mut bulk = [0; 4096];
        assert_eq!(
            service.send_infer_request(Endpoint::Broker, RequestId(id(91)), b"late", &mut output),
            Err(ServiceError::Route)
        );
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_CLOSED,
                epoch,
                old.route,
                old.generation.get(),
                0,
                RequestId::ZERO,
                0,
            ),
            &[],
        ));
        service.drain(&mut output, &mut bulk);
        assert_eq!(service.close_broker_route(old.generation), Ok(()));
        assert_eq!(service.bindings.for_endpoint(Endpoint::Broker), None);
        assert_eq!(
            service.bindings.for_endpoint(Endpoint::Workspace),
            Some(workspace)
        );
        assert_eq!(
            service.rotate_broker_route(old.generation),
            Err(ServiceError::Route)
        );
        assert_eq!(service.rotate_broker_route(next), Ok(()));
        let replacement = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
        assert_ne!(replacement.route, old.route);
        assert_eq!(replacement.generation, next);
        assert_eq!(
            service.close_broker_route(old.generation),
            Err(ServiceError::Route)
        );
        assert_eq!(
            service.bindings.for_endpoint(Endpoint::Workspace),
            Some(workspace)
        );
    }

    #[test]
    fn agent_rotation_closes_generation_before_deriving_a_fresh_route() {
        let mut service = activate(52);
        let old = service.bindings.for_endpoint(Endpoint::Agent(0)).unwrap();
        let next = TaskGeneration::new(old.generation.get() + 1).unwrap();
        assert_eq!(
            service.rotate_agent_route(0, next),
            Err(ServiceError::Backpressure)
        );
        assert_eq!(
            service.rotate_agent_route(0, old.generation),
            Err(ServiceError::Route)
        );
        let epoch = service.epoch().unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_CLOSED,
                epoch,
                old.route,
                old.generation.get(),
                0,
                RequestId::ZERO,
                0,
            ),
            &[],
        ));
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        service.rotate_agent_route(0, next).unwrap();
        let replacement = service.bindings.for_endpoint(Endpoint::Agent(0)).unwrap();
        assert_eq!(replacement.generation, next);
        assert_ne!(replacement.route, old.route);
        assert_eq!(
            service.close_agent_route(0, old.generation),
            Err(ServiceError::Route)
        );
        assert_eq!(
            service.rotate_agent_route(0, next),
            Err(ServiceError::Route)
        );
    }

    #[test]
    fn broker_rotation_requires_close_ack_and_preserves_workspace() {
        let mut service = activate(54);
        let old = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
        let workspace = service.bindings.for_endpoint(Endpoint::Workspace).unwrap();
        let next = TaskGeneration::new(old.generation.get() + 1).unwrap();
        assert_eq!(service.rotate_broker_route(next), Err(ServiceError::Route));
        assert_eq!(
            service.close_broker_route(old.generation),
            Err(ServiceError::Backpressure)
        );
        assert_eq!(service.rotate_broker_route(next), Err(ServiceError::Route));
        let epoch = service.epoch().unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                ROUTE_CLOSED,
                epoch,
                old.route,
                old.generation.get(),
                0,
                RequestId::ZERO,
                0,
            ),
            &[],
        ));
        service.drain(&mut [0; 4096], &mut [0; 4096]);
        service.rotate_broker_route(next).unwrap();
        let replacement = service.bindings.for_endpoint(Endpoint::Broker).unwrap();
        assert_eq!(replacement.generation, next);
        assert_ne!(replacement.route, old.route);
        let retained = service.bindings.for_endpoint(Endpoint::Workspace).unwrap();
        assert_eq!(retained.route, workspace.route);
        assert_eq!(retained.generation, workspace.generation);
        assert_eq!(
            service.close_broker_route(old.generation),
            Err(ServiceError::Route)
        );
        assert_eq!(service.rotate_broker_route(next), Err(ServiceError::Route));
    }

    #[test]
    fn authority_report_is_broker_only_validated_and_does_not_open_a_request_row() {
        let mut service = activate(53);
        let request = RequestId(id(63));
        let report = kore_hostlink::AuthorityReportView {
            state: kore_hostlink::AuthorityState::Installed,
            kind: kore_hostlink::AuthorityKind::Object,
            agent_slot: 1,
            generation: 7,
            rights: 1,
            label: "workspace/docs-architecture.md",
        };
        let mut payload =
            [0; kore_hostlink::AUTHORITY_REPORT_HEADER_LEN + kore_hostlink::AUTHORITY_LABEL_MAX];
        let payload_len = report.encode(&mut payload).unwrap();
        let mut output = [0; 4096];
        assert_eq!(
            service.send_request(
                Endpoint::Broker,
                AUTHORITY_REPORT,
                request,
                &payload[..payload_len],
                &mut output,
            ),
            Err(ServiceError::Malformed),
            "the generic request surface cannot bypass typed report validation"
        );
        service
            .send_authority_report(
                Endpoint::Broker,
                request,
                &payload[..payload_len],
                &mut output,
            )
            .unwrap();
        let frame = service.transport_mut().control_out.last().unwrap();
        let envelope = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).unwrap();
        assert_eq!(envelope.kind.get(), AUTHORITY_REPORT);
        assert_eq!(envelope.request_id, request);

        // The report is one-way telemetry: its correlation value does not
        // reserve an inference request row.
        service
            .send_infer_request(Endpoint::Broker, request, b"next", &mut output)
            .unwrap();
        assert_eq!(
            service.send_authority_report(
                Endpoint::Agent(0),
                RequestId(id(64)),
                &payload[..payload_len],
                &mut output,
            ),
            Err(ServiceError::Malformed)
        );
        payload[10] = 1;
        assert_eq!(
            service.send_authority_report(
                Endpoint::Broker,
                RequestId(id(65)),
                &payload[..payload_len],
                &mut output,
            ),
            Err(ServiceError::Malformed)
        );
    }

    #[test]
    fn authority_report_backpressure_preserves_sequence_and_exact_retry() {
        let mut service = activate(54);
        let request = RequestId(id(66));
        let report = kore_hostlink::AuthorityReportView {
            state: kore_hostlink::AuthorityState::Installed,
            kind: kore_hostlink::AuthorityKind::View,
            agent_slot: 1,
            generation: 7,
            rights: 1,
            label: "output",
        };
        let mut payload =
            [0; kore_hostlink::AUTHORITY_REPORT_HEADER_LEN + kore_hostlink::AUTHORITY_LABEL_MAX];
        let length = report.encode(&mut payload).unwrap();
        let mut output = [0; 4096];
        let count = service.transport_mut().control_out.len();
        service.transport_mut().backpressure_control = true;
        assert_eq!(
            service.send_authority_report(
                Endpoint::Broker,
                request,
                &payload[..length],
                &mut output
            ),
            Err(ServiceError::Transport(PortError::Backpressure))
        );
        let attempted = output[..ENVELOPE_LEN + length].to_vec();
        assert_eq!(service.transport_mut().control_out.len(), count);
        service
            .send_authority_report(Endpoint::Broker, request, &payload[..length], &mut output)
            .unwrap();
        assert_eq!(
            service.transport_mut().control_out.last().unwrap(),
            &attempted
        );
        let first = EnvelopeV1::read_from_bytes(&attempted[..ENVELOPE_LEN]).unwrap();
        let terminal = kore_hostlink::AuthorityReportView {
            state: kore_hostlink::AuthorityState::Teardown,
            kind: kore_hostlink::AuthorityKind::Agent,
            rights: 0,
            label: "agent",
            ..report
        };
        let length = terminal.encode(&mut payload).unwrap();
        service
            .send_authority_report(
                Endpoint::Broker,
                RequestId(id(67)),
                &payload[..length],
                &mut output,
            )
            .unwrap();
        let frame = service.transport_mut().control_out.last().unwrap();
        let last = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).unwrap();
        assert_eq!(last.sequence.get(), first.sequence.get() + 1);
        assert_eq!(
            kore_hostlink::decode_authority_report(&frame[ENVELOPE_LEN..]),
            Some(terminal)
        );
    }

    #[test]
    fn control_cancel_precedes_fixed_bulk_budget() {
        let mut service = activate(3);
        let epoch = service.epoch().unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(CANCEL, epoch, RouteId(id(4)), 9, 0, RequestId(id(7)), 0),
            &[],
        ));
        for sequence in 1..=5 {
            service.transport_mut().bulk_in.push_back(bytes(
                link_envelope(
                    BULK_CHUNK,
                    epoch,
                    RouteId(id(4)),
                    9,
                    sequence,
                    RequestId(id(8)),
                    0,
                ),
                &[],
            ));
        }
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        let report = service.drain(&mut c, &mut b);
        assert_eq!(
            service.delivery().cancelled,
            [(Endpoint::Agent(1), RequestId(id(7)))]
        );
        assert_eq!(report.control_frames, 1);
        assert!(report.bulk_frames <= BULK_DRAIN_BUDGET as u8);
        assert_eq!(service.transport_mut().bulk_in.len(), 3);
    }

    #[test]
    fn disconnect_wakes_endpoints_deterministically_and_reconnect_changes_epoch() {
        let mut service = activate(4);
        let first = service.epoch().unwrap();
        service.transport_mut().disconnect_control = true;
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        assert!(service.drain(&mut c, &mut b).disconnected);
        let statuses = [
            service.pop_status().unwrap(),
            service.pop_status().unwrap(),
            service.pop_status().unwrap(),
            service.pop_status().unwrap(),
            service.pop_status().unwrap(),
            service.pop_status().unwrap(),
        ];
        assert_eq!(
            statuses.map(|s| s.endpoint),
            [
                Endpoint::Broker,
                Endpoint::Agent(0),
                Endpoint::Agent(1),
                Endpoint::Agent(2),
                Endpoint::Agent(3),
                Endpoint::Workspace,
            ]
        );
        let hello = HelloV1 {
            host_nonce: [5; 32],
        };
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                HELLO,
                LinkEpoch::ZERO,
                RouteId::ZERO,
                0,
                0,
                RequestId::ZERO,
                32,
            ),
            hello.as_bytes(),
        ));
        service.drain(&mut c, &mut b);
        assert_ne!(service.epoch().unwrap(), first);
    }

    #[test]
    fn outbound_bulk_abort_waits_for_credits_and_retires_only_on_remote_receipt() {
        let mut service = activate(61);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(62));
        let data = [b'x'; 512];
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(63)),
            total_len: U32::new(data.len() as u32),
            chunk_count: U16::new(1),
            content_kind: U16::new(INFERENCE_REQUEST_CONTENT),
            digest_blake3: *blake3::hash(&data).as_bytes(),
        };
        let mut output = [0; 4096];
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service
            .begin_outbound_bulk(Endpoint::Broker, request, reference, &mut output)
            .unwrap();
        for sequence in 0..2 {
            let frames = service.transport_mut().bulk_out.len();
            assert_eq!(
                service.abort_outbound_bulk(
                    Endpoint::Broker,
                    request,
                    reference.bulk_id,
                    RefusalCode::PolicyDenied,
                    &mut output
                ),
                Err(ServiceError::Backpressure)
            );
            assert_eq!(service.transport_mut().bulk_out.len(), frames);
            assert!(
                service
                    .routes
                    .bound_mut(RouteId(id(2)))
                    .unwrap()
                    .state()
                    .has_request(request)
            );
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    BULK_ACCEPTED,
                    epoch,
                    RouteId(id(2)),
                    7,
                    sequence,
                    request,
                    16,
                ),
                &reference.bulk_id.0,
            ));
            service.drain(&mut c, &mut b);
            assert_eq!(service.outbound[0].phase, OutboundPhase::Ready);
            if sequence == 0 {
                service
                    .send_outbound_chunk(
                        Endpoint::Broker,
                        request,
                        reference.bulk_id,
                        &data,
                        &mut output,
                    )
                    .unwrap();
            }
        }
        service
            .abort_outbound_bulk(
                Endpoint::Broker,
                request,
                reference.bulk_id,
                RefusalCode::PolicyDenied,
                &mut output,
            )
            .unwrap();
        assert_eq!(service.outbound.len(), 1);
        assert!(
            service
                .routes
                .bound_mut(RouteId(id(2)))
                .unwrap()
                .state()
                .has_request(request)
        );
        assert!(
            !service
                .delivery()
                .controls
                .iter()
                .any(|(_, kind, _, _)| *kind == BULK_ABORTED)
        );
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(BULK_ABORTED, epoch, RouteId(id(2)), 7, 2, request, 16),
            &reference.bulk_id.0,
        ));
        service.drain(&mut c, &mut b);
        assert!(service.outbound.is_empty());
        assert!(
            !service
                .routes
                .bound_mut(RouteId(id(2)))
                .unwrap()
                .state()
                .has_request(request)
        );
        assert!(
            service
                .delivery()
                .controls
                .iter()
                .any(
                    |(endpoint, kind, got, payload)| *endpoint == Endpoint::Broker
                        && *kind == BULK_ABORTED
                        && *got == request
                        && payload.as_slice() == reference.bulk_id.0
                )
        );
        let next = RequestId(id(64));
        service
            .begin_outbound_bulk(
                Endpoint::Broker,
                next,
                BulkRefV1 {
                    bulk_id: BulkId(id(65)),
                    ..reference
                },
                &mut output,
            )
            .unwrap();
        assert_eq!(service.outbound[0].request, next);
    }

    #[test]
    fn inbound_and_outbound_bulk_use_the_shared_lifecycle() {
        let mut service = activate(6);
        let epoch = service.epoch().unwrap();
        let data = b"abc";
        let digest = *blake3::hash(data).as_bytes();
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(22)),
            total_len: U32::new(3),
            chunk_count: U16::new(1),
            content_kind: U16::new(1),
            digest_blake3: digest,
        };
        let request = RequestId(id(23));
        let begin = bytes(
            link_envelope(
                BULK_BEGIN,
                epoch,
                RouteId(id(3)),
                8,
                0,
                request,
                reference.as_bytes().len(),
            ),
            reference.as_bytes(),
        );
        let header = BulkChunkHeaderV1 {
            bulk_id: reference.bulk_id,
            offset: U32::ZERO,
            digest_blake3: digest,
        };
        let mut chunk = header.as_bytes().to_vec();
        chunk.extend_from_slice(data);
        let finish = BulkFinishV1 {
            bulk_id: reference.bulk_id,
            final_total: U32::new(3),
            digest_blake3: digest,
        };
        for frame in [
            begin,
            bytes(
                link_envelope(
                    BULK_CHUNK,
                    epoch,
                    RouteId(id(3)),
                    8,
                    1,
                    request,
                    chunk.len(),
                ),
                &chunk,
            ),
            bytes(
                link_envelope(
                    BULK_FINISH,
                    epoch,
                    RouteId(id(3)),
                    8,
                    2,
                    request,
                    finish.as_bytes().len(),
                ),
                finish.as_bytes(),
            ),
        ] {
            service.transport_mut().bulk_in.push_back(frame);
        }
        let mut c = [0; 4096];
        let mut b = [0; 4096];
        service.drain(&mut c, &mut b);
        service.drain(&mut c, &mut b);
        assert_eq!(
            service
                .delivery()
                .bulk
                .iter()
                .map(|e| e.1)
                .collect::<StdVec<_>>(),
            [BULK_BEGIN, BULK_CHUNK, BULK_READY]
        );
        let mut output = [0; 4096];
        let outbound = BulkRefV1 {
            bulk_id: BulkId(id(24)),
            ..reference
        };
        service
            .begin_outbound_bulk(Endpoint::Agent(1), RequestId(id(25)), outbound, &mut output)
            .unwrap();
        assert_eq!(
            service.send_outbound_chunk(
                Endpoint::Agent(1),
                RequestId(id(25)),
                outbound.bulk_id,
                data,
                &mut output,
            ),
            Err(ServiceError::Backpressure)
        );
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                BULK_ACCEPTED,
                epoch,
                RouteId(id(4)),
                9,
                0,
                RequestId(id(25)),
                16,
            ),
            &outbound.bulk_id.0,
        ));
        service.drain(&mut c, &mut b);
        service
            .send_outbound_chunk(
                Endpoint::Agent(1),
                RequestId(id(25)),
                outbound.bulk_id,
                data,
                &mut output,
            )
            .unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                BULK_ACCEPTED,
                epoch,
                RouteId(id(4)),
                9,
                1,
                RequestId(id(25)),
                16,
            ),
            &outbound.bulk_id.0,
        ));
        service.drain(&mut c, &mut b);
        service
            .finish_outbound_bulk(
                Endpoint::Agent(1),
                RequestId(id(25)),
                outbound.bulk_id,
                &mut output,
            )
            .unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(
                BULK_READY,
                epoch,
                RouteId(id(4)),
                9,
                2,
                RequestId(id(25)),
                16,
            ),
            &outbound.bulk_id.0,
        ));
        service.drain(&mut c, &mut b);
        let kinds: StdVec<_> = service
            .transport_mut()
            .bulk_out
            .iter()
            .map(|f| {
                EnvelopeV1::read_from_bytes(&f[..ENVELOPE_LEN])
                    .unwrap()
                    .kind
                    .get()
            })
            .collect();
        assert_eq!(kinds, [BULK_BEGIN, BULK_CHUNK, BULK_FINISH]);
    }

    #[test]
    fn workspace_upload_retains_request_until_application_reply() {
        let mut service = activate(7);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(26));
        let data = b"workspace object";
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(27)),
            total_len: U32::new(data.len() as u32),
            chunk_count: U16::new(1),
            content_kind: U16::new(1),
            digest_blake3: *blake3::hash(data).as_bytes(),
        };
        let mut output = [0; 4096];
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];

        service
            .begin_outbound_bulk(Endpoint::Workspace, request, reference, &mut output)
            .unwrap();
        for (sequence, kind) in [(0, BULK_ACCEPTED), (1, BULK_ACCEPTED)] {
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(kind, epoch, RouteId(id(5)), 10, sequence, request, 16),
                &reference.bulk_id.0,
            ));
            service.drain(&mut control, &mut bulk);
            if sequence == 0 {
                service
                    .send_outbound_chunk(
                        Endpoint::Workspace,
                        request,
                        reference.bulk_id,
                        data,
                        &mut output,
                    )
                    .unwrap();
            }
        }
        service
            .finish_outbound_bulk(Endpoint::Workspace, request, reference.bulk_id, &mut output)
            .unwrap();
        for (sequence, kind, payload) in [
            (2, BULK_READY, reference.bulk_id.0.as_slice()),
            (3, WORKSPACE_REPLY, b"applied".as_slice()),
        ] {
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    kind,
                    epoch,
                    RouteId(id(5)),
                    10,
                    sequence,
                    request,
                    payload.len(),
                ),
                payload,
            ));
        }
        service.drain(&mut control, &mut bulk);

        assert_eq!(service.phase(), LinkPhase::Active);
        assert_eq!(
            service
                .delivery()
                .controls
                .iter()
                .map(|event| event.1)
                .collect::<StdVec<_>>(),
            [BULK_READY, WORKSPACE_REPLY]
        );
        assert!(service.outbound.is_empty());
        assert!(
            !service
                .routes
                .bound_mut(RouteId(id(5)))
                .unwrap()
                .state()
                .has_request(request)
        );
    }

    #[test]
    fn inference_upload_retains_request_until_terminal_inference_reply() {
        let mut service = activate(7);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(80));
        let data = b"broker plan";
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(81)),
            total_len: U32::new(data.len() as u32),
            chunk_count: U16::new(1),
            content_kind: U16::new(INFERENCE_REQUEST_CONTENT),
            digest_blake3: *blake3::hash(data).as_bytes(),
        };
        let mut output = [0; 4096];
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];

        service
            .begin_outbound_bulk(Endpoint::Broker, request, reference, &mut output)
            .unwrap();
        for sequence in 0..=1 {
            service.transport_mut().control_in.push_back(bytes(
                link_envelope(
                    BULK_ACCEPTED,
                    epoch,
                    RouteId(id(2)),
                    7,
                    sequence,
                    request,
                    16,
                ),
                &reference.bulk_id.0,
            ));
            service.drain(&mut control, &mut bulk);
            if sequence == 0 {
                service
                    .send_outbound_chunk(
                        Endpoint::Broker,
                        request,
                        reference.bulk_id,
                        data,
                        &mut output,
                    )
                    .unwrap();
            }
        }
        service
            .finish_outbound_bulk(Endpoint::Broker, request, reference.bulk_id, &mut output)
            .unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(BULK_READY, epoch, RouteId(id(2)), 7, 2, request, 16),
            &reference.bulk_id.0,
        ));
        service.drain(&mut control, &mut bulk);
        assert!(
            service
                .routes
                .bound_mut(RouteId(id(2)))
                .unwrap()
                .state()
                .has_request(request)
        );

        let mut event = [0; INFERENCE_EVENT_HEADER_LEN + 16];
        let event_len = InferenceEventView {
            kind: InferenceEventKind::TurnFinished,
            terminal: true,
            part: 0,
            prompt_tokens: 1,
            completion_tokens: 1,
            turn: 1,
            payload: b"accepted",
        }
        .encode(&mut event)
        .unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(INFER_REPLY, epoch, RouteId(id(2)), 7, 3, request, event_len),
            &event[..event_len],
        ));
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.phase(), LinkPhase::Active);
        assert!(
            !service
                .routes
                .bound_mut(RouteId(id(2)))
                .unwrap()
                .state()
                .has_request(request)
        );
    }

    #[test]
    fn outbound_coalescer_streams_one_mib_as_256_wire_chunks_without_ack_flood() {
        let mut service = activate(60);
        let epoch = service.epoch().unwrap();
        let content = vec![0x5a; 1 << 20];
        let request = RequestId(id(70));
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(71)),
            total_len: U32::new(content.len() as u32),
            chunk_count: U16::new(256),
            content_kind: U16::new(0x538),
            digest_blake3: *blake3::hash(&content).as_bytes(),
        };
        let mut output = vec![0; BULK_FRAME_MAX];
        service
            .begin_outbound_bulk(Endpoint::Broker, request, reference, &mut output)
            .unwrap();
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        let ack = |sequence, kind| {
            bytes(
                link_envelope(kind, epoch, RouteId(id(2)), 7, sequence, request, 16),
                &reference.bulk_id.0,
            )
        };
        service
            .transport_mut()
            .control_in
            .push_back(ack(0, BULK_ACCEPTED));
        service.drain(&mut control, &mut bulk);
        let mut sequence = 1;
        for fragment in content.chunks(crate::CLIENT_CHUNK_MAX) {
            service
                .send_outbound_chunk(
                    Endpoint::Broker,
                    request,
                    reference.bulk_id,
                    fragment,
                    &mut output,
                )
                .unwrap();
            if service.outbound[0].phase == OutboundPhase::AwaitChunk {
                service
                    .transport_mut()
                    .control_in
                    .push_back(ack(sequence, BULK_ACCEPTED));
                sequence += 1;
                service.drain(&mut control, &mut bulk);
            }
        }
        let transfer = &service.outbound[0];
        assert_eq!(
            (
                transfer.received,
                transfer.offset,
                transfer.chunks,
                transfer.staged
            ),
            (1 << 20, 1 << 20, 256, 0)
        );
        assert_eq!(transfer.credit_chunks, 64);
        assert_eq!(transfer.credit_bytes, 16 * 1024 * 1024);
        service
            .finish_outbound_bulk(Endpoint::Broker, request, reference.bulk_id, &mut output)
            .unwrap();
        let frames = &service.transport.bulk_out;
        assert_eq!(frames.len(), 258);
        for (index, frame) in frames[1..257].iter().enumerate() {
            let envelope = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).unwrap();
            assert_eq!(envelope.kind.get(), BULK_CHUNK);
            assert_eq!(
                envelope.payload_len.get(),
                (BULK_CHUNK_HEADER_LEN + 4096) as u32
            );
            let header = BulkChunkHeaderV1::read_from_bytes(
                &frame[ENVELOPE_LEN..ENVELOPE_LEN + BULK_CHUNK_HEADER_LEN],
            )
            .unwrap();
            assert_eq!(header.offset.get(), (index * 4096) as u32);
        }
        service
            .transport_mut()
            .control_in
            .push_back(ack(sequence, BULK_READY));
        service.drain(&mut control, &mut bulk);
        assert_eq!(service.delivery().controls.len(), 1);
        assert_eq!(service.delivery().controls[0].1, BULK_READY);
    }

    #[test]
    fn outbound_coalescer_is_transactional_on_crossing_and_transport_backpressure() {
        let mut service = activate(61);
        let epoch = service.epoch().unwrap();
        let data = [0x33; FOUNDED_BULK_WIRE_CHUNK];
        let request = RequestId(id(72));
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(73)),
            total_len: U32::new(data.len() as u32),
            chunk_count: U16::new(1),
            content_kind: U16::new(0x538),
            digest_blake3: *blake3::hash(&data).as_bytes(),
        };
        let mut output = vec![0; BULK_FRAME_MAX];
        service
            .begin_outbound_bulk(Endpoint::Broker, request, reference, &mut output)
            .unwrap();
        service.outbound[0].phase = OutboundPhase::Ready;
        for fragment in data[..4088].chunks(511) {
            service
                .send_outbound_chunk(
                    Endpoint::Broker,
                    request,
                    reference.bulk_id,
                    fragment,
                    &mut output,
                )
                .unwrap();
        }
        let before = (
            service.outbound[0].received,
            service.outbound[0].staged,
            service.outbound[0].offset,
            service.outbound[0].chunks,
        );
        assert_eq!(
            service.send_outbound_chunk(
                Endpoint::Broker,
                request,
                reference.bulk_id,
                &[0x33; 9],
                &mut output,
            ),
            Err(ServiceError::Bulk)
        );
        assert_eq!(
            (
                service.outbound[0].received,
                service.outbound[0].staged,
                service.outbound[0].offset,
                service.outbound[0].chunks,
            ),
            before
        );
        service.transport_mut().backpressure_bulk = true;
        assert_eq!(
            service.send_outbound_chunk(
                Endpoint::Broker,
                request,
                reference.bulk_id,
                &data[4088..],
                &mut output,
            ),
            Err(ServiceError::Transport(PortError::Backpressure))
        );
        assert_eq!(
            (
                service.outbound[0].received,
                service.outbound[0].staged,
                service.outbound[0].offset,
                service.outbound[0].chunks,
            ),
            before
        );
        service
            .send_outbound_chunk(
                Endpoint::Broker,
                request,
                reference.bulk_id,
                &data[4088..],
                &mut output,
            )
            .unwrap();
        assert_eq!(
            (
                service.outbound[0].received,
                service.outbound[0].staged,
                service.outbound[0].offset,
                service.outbound[0].chunks,
            ),
            (4096, 0, 4096, 1)
        );
        let frame = service.transport.bulk_out.last().unwrap();
        let envelope = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).unwrap();
        assert_eq!(
            envelope.payload_len.get(),
            (BULK_CHUNK_HEADER_LEN + 4096) as u32
        );
        assert_eq!(service.epoch(), Some(epoch));
    }

    #[test]
    fn routed_refusal_poison_retires_epoch_and_wakes_other_requests() {
        let mut service = activate(10);
        let epoch = service.epoch().unwrap();
        let request = RequestId(id(31));
        let reference = BulkRefV1 {
            bulk_id: BulkId(id(32)),
            total_len: U32::new(3),
            chunk_count: U16::new(1),
            content_kind: U16::new(1),
            digest_blake3: *blake3::hash(b"abc").as_bytes(),
        };
        let mut output = [0; 4096];
        service
            .begin_outbound_bulk(Endpoint::Broker, request, reference, &mut output)
            .unwrap();
        let other = RequestId(id(33));
        service
            .send_infer_request(Endpoint::Broker, other, b"pending", &mut output)
            .unwrap();
        assert_eq!(service.outbound.len(), 1);

        let refusal = RefusalView {
            for_family: FAMILY_LINK,
            for_kind: BULK_BEGIN,
            code: RefusalCode::Backpressure,
            retry: RetryClass::SameEpoch,
            retry_after_ms: 0,
            safe_detail: &[],
        };
        let mut refusal_payload = [0; REFUSAL_HEADER_LEN];
        let refusal_len = refusal.encode(&mut refusal_payload).unwrap();
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(REFUSED, epoch, RouteId(id(2)), 7, 0, request, refusal_len),
            &refusal_payload[..refusal_len],
        ));
        // This later same-epoch frame must never reach a founded mailbox after
        // REFUSED poisons the guest send/host receive sequence relationship.
        service.transport_mut().control_in.push_back(bytes(
            link_envelope(INFER_REPLY, epoch, RouteId(id(2)), 7, 1, other, 2),
            b"ok",
        ));
        let mut control = [0; 4096];
        let mut bulk = [0; 4096];
        service.drain(&mut control, &mut bulk);
        service.drain(&mut control, &mut bulk);

        assert!(service.outbound.is_empty());
        assert_eq!(service.phase(), LinkPhase::Disconnected);
        assert!(service.poisoned);
        assert_eq!(
            service.pop_status(),
            Some(EndpointStatus {
                endpoint: Endpoint::Broker,
                request,
                reason: StatusReason::Refused(RefusalCode::Backpressure),
            })
        );
        assert_eq!(
            service.pop_status(),
            Some(EndpointStatus {
                endpoint: Endpoint::Broker,
                request: other,
                reason: StatusReason::HostDisconnected,
            })
        );
        assert!(service.delivery().controls.is_empty());
        assert_eq!(
            service.send_infer_request(
                Endpoint::Broker,
                RequestId(id(34)),
                b"same epoch",
                &mut output,
            ),
            Err(ServiceError::Link)
        );

        // The adapter's explicit fault close is idempotent with the host's
        // refusal-after-flush close. Only the observed physical edge clears
        // poison so a subsequent fresh Hello may negotiate.
        service.transport_mut().disconnect_control = true;
        assert!(service.drain(&mut control, &mut bulk).disconnected);
        assert!(!service.poisoned);
    }
}
