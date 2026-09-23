//! Production HostLink entry: exact device claim, the four bootstrap client
//! pairs, and one control-first VirtIO/HostLink drive loop. Agent route state
//! itself is installed through the bounded dynamic pool in
//! [`crate::HostLinkService`].

#![cfg(target_os = "none")]

extern crate alloc;

use alloc::{boxed::Box, vec};

use kore_abi::session::{Notice, SlotRef};
use kore_abi::task::{Bootstrap, role};
use kore_driver::host::{ClassStage, IntakeError};
use kore_hostlink::{
    BULK_FRAME_MAX, CONTROL_FRAME_MAX, PortError, RefusalCode, RouteId, TaskGeneration,
};
use kore_rt::chan::ChanTx;
use kore_rt::entry::refuse_with;
use kore_rt::service::{Bundle, SystemService};
use kore_rt::svc::{COOKIE_REPLY, COOKIE_SERVING, ServeRings, SvcProtocol, serve_rings};
use kore_svc_device::client::{ClaimedDevice, FaceLayout, MultiBroker};

use crate::shmem::{
    MappedSharedRam, RuntimeBulk, SHMEM_DRAIN_RECHECK_PASSES, ShmemClaimError, needs_bulk_recheck,
};
use crate::virtio::{BoundConsolePorts, SerialPumpBudget, VirtioHostLink, bind_console};
use crate::{
    BULK_DRAIN_BUDGET, Bindings, BulkTransport, ClientRequest, ClientResponse, Endpoint,
    HOSTLINK_SERVE_PAIRS, HostLinkEndowment, HostLinkFront, HostLinkService, HostLinkWire,
    LocalRoute, MailboxBank, MailboxError, ServiceError,
};

pub const ENTRY_ATTEST_RUNG: u64 = kore_rt::entry::RUNG_SERVICE_ATTEST;

mod rung {
    pub const ENDOWMENT: u64 = 2;
    pub const DEVICE_FACE: u64 = 17;
    pub const DEVICE_CLAIM: u64 = 18;
    pub const DEVICE_BIND: u64 = 19;
    pub const ENTROPY: u64 = 20;
    pub const PORTS: u64 = 21;
    pub const PAIRS: u64 = 22;
    pub const BEACON: u64 = 23;
    pub const SHMEM: u64 = 24;
}

fn bind_refusal(error: IntakeError) -> u64 {
    let detail = match error {
        IntakeError::MissingRole { role, index } => {
            1 | (u64::from(role) << 8) | (u64::from(index) << 24)
        }
        IntakeError::BadHandle { role, index } => {
            2 | (u64::from(role) << 8) | (u64::from(index) << 24)
        }
        IntakeError::InvalidBar(bar) => 3 | (u64::from(bar) << 8),
        IntakeError::DuplicateBar(bar) => 4 | (u64::from(bar) << 8),
        IntakeError::BarUnavailable(bar) => 5 | (u64::from(bar) << 8),
        IntakeError::BarMap { bar, code } => 10 | (u64::from(bar) << 8) | (u64::from(code) << 16),
        IntakeError::InvalidInterruptBlock => 6,
        IntakeError::InterruptUnavailable => 7,
        IntakeError::DmaUnavailable => 8,
        IntakeError::DmaMap { code } => 11 | (u64::from(code) << 8),
        IntakeError::ClassBindRefused { at, code } => {
            let stage = match at {
                ClassStage::Prepare => 1,
                ClassStage::Start => 2,
            };
            9 | (stage << 8) | (u64::from(code) << 16)
        }
    };
    rung::DEVICE_BIND | (detail << 8)
}

fn port_refusal(error: PortError) -> u64 {
    let detail = match error {
        PortError::Backpressure => 1,
        PortError::Disconnected => 2,
        PortError::Fault => 3,
    };
    rung::PORTS | (detail << 8)
}

type SerialAdapter = VirtioHostLink<BoundConsolePorts, Box<[u8]>, Box<[u8]>>;
type LinkAdapter = RuntimeBulk<SerialAdapter, MappedSharedRam>;
type Core = HostLinkService<LinkAdapter, MailboxBank>;

struct Pair<'w> {
    rings: ServeRings<'w, HostLinkWire>,
    kick: SlotRef,
}

struct Router<'w, 't> {
    core: Core,
    supervisor: crate::SupervisorOwner,
    transport: &'t kore_rt::transport::TrapTransport,
    pairs: [Option<Pair<'w>>; HOSTLINK_SERVE_PAIRS],
    retired_pairs: [Option<kore_hostlink::TaskGeneration>; crate::MAX_DYNAMIC_AGENT_ROUTES],
    control_work: Box<[u8]>,
    bulk_work: Box<[u8]>,
}

impl Router<'_, '_> {
    async fn drain_requests(&mut self) {
        // Two records per pair are the frozen ring capacity. One extra round
        // catches a producer that published while another pair was served,
        // while keeping a hard upper bound per executor turn.
        for _ in 0..(3 * HOSTLINK_SERVE_PAIRS) {
            let mut found = None;
            for via in 0..self.pairs.len() {
                if !self.supervisor.client_active() && matches!(via, 0 | 1 | 2) {
                    continue;
                }
                if let Some(wire) = self.pairs[via]
                    .as_mut()
                    .and_then(|pair| pair.rings.req.pop())
                {
                    found = Some((via, wire));
                    break;
                }
            }
            let Some((via, wire)) = found else {
                break;
            };
            let response = HostLinkWire::decode_req(&wire)
                .map(|request| self.handle(via as u16, request))
                .unwrap_or(ClientResponse::Refused {
                    request: kore_hostlink::RequestId::ZERO,
                    code: RefusalCode::Malformed,
                });
            let pair = self.pairs[via]
                .as_mut()
                .expect("request arrived on an attached pair");
            if pair
                .rings
                .rep
                .push(HostLinkWire::encode_rep(&response, 0))
                .is_ok()
            {
                let _ = ChanTx::new(self.transport, pair.kick)
                    .send(Notice {
                        cookie: COOKIE_REPLY,
                        payload: 0,
                    })
                    .await;
            }
        }
    }

    fn handle(&mut self, via: u16, request: ClientRequest) -> ClientResponse {
        if let ClientRequest::Supervisor {
            request: request_id,
            operation,
            generation,
        } = request
        {
            if usize::from(via) != crate::HOSTLINK_OWNER_PAIR {
                return refused(&request, RefusalCode::PolicyDenied);
            }
            return match self
                .supervisor
                .operate(&mut self.core, operation, generation)
            {
                Ok(_) => {
                    if self.supervisor.needs_request_drain() {
                        for index in [0, 1, 2] {
                            let Some(pair) = self.pairs[index].as_mut() else {
                                return refused(&request, RefusalCode::PolicyDenied);
                            };
                            if !crate::drain_retired_requests(&mut pair.rings) {
                                return refused(&request, RefusalCode::Malformed);
                            }
                        }
                        self.supervisor.confirm_request_drain();
                        self.retired_pairs[..crate::HOSTLINK_SUPERVISOR_WORKERS].fill(None);
                    }
                    ClientResponse::Supervisor {
                        request: request_id,
                        state: self.supervisor.state(),
                    }
                }
                Err(error) => refused(&request, service_code(error)),
            };
        }
        if usize::from(via) == crate::HOSTLINK_OWNER_PAIR {
            return refused(&request, RefusalCode::PolicyDenied);
        }
        let Some(endpoint) = self.core.delivery().endpoint_for_via(via) else {
            return refused(&request, RefusalCode::PolicyDenied);
        };
        let request_id = request.request_id();
        let result = match request {
            ClientRequest::Supervisor { .. } => unreachable!(),
            ClientRequest::Send {
                kind,
                request,
                payload,
            } => {
                if kind == kore_hostlink::AUTHORITY_REPORT {
                    self.core.send_authority_report(
                        endpoint,
                        request,
                        payload.as_slice(),
                        &mut self.control_work,
                    )
                } else {
                    self.core.send_request(
                        endpoint,
                        kind,
                        request,
                        payload.as_slice(),
                        &mut self.control_work,
                    )
                }
            }
            ClientRequest::Cancel { request } => {
                self.core.cancel_outbound_request(endpoint, request)
            }
            ClientRequest::Poll => {
                return match self.core.delivery_mut().poll_via(via) {
                    Ok(Some(event)) => ClientResponse::Event(event),
                    Ok(None) => ClientResponse::Empty,
                    Err(error) => ClientResponse::Refused {
                        request: request_id,
                        code: mailbox_code(error),
                    },
                };
            }
            ClientRequest::Status => {
                let phase = self.core.phase();
                let bulk_transport = if self.core.transport_mut().is_shared() {
                    BulkTransport::SharedMemory
                } else {
                    BulkTransport::Serial
                };
                return match self.core.delivery().status_via(via, phase, bulk_transport) {
                    Ok(status) => ClientResponse::Status(status),
                    Err(error) => ClientResponse::Refused {
                        request: request_id,
                        code: mailbox_code(error),
                    },
                };
            }
            ClientRequest::BulkBegin { request, reference } => {
                self.core
                    .begin_outbound_bulk(endpoint, request, reference, &mut self.bulk_work)
            }
            ClientRequest::BulkChunk {
                request,
                bulk_id,
                bytes,
            } => self.core.send_outbound_chunk(
                endpoint,
                request,
                bulk_id,
                bytes.as_slice(),
                &mut self.bulk_work,
            ),
            ClientRequest::BulkFinish {
                request,
                bulk_id,
                total_len,
                digest_blake3,
            } => self.core.finish_outbound_bulk_claimed(
                endpoint,
                request,
                bulk_id,
                total_len,
                digest_blake3,
                &mut self.bulk_work,
            ),
            ClientRequest::BulkAbort {
                request,
                bulk_id,
                reason,
            } => self.core.abort_outbound_bulk(
                endpoint,
                request,
                bulk_id,
                reason,
                &mut self.bulk_work,
            ),
            ClientRequest::RotateAgent {
                slot, generation, ..
            } => {
                if endpoint != Endpoint::Broker {
                    return refused(&request, RefusalCode::PolicyDenied);
                }
                if !crate::retirement::successor_after_close(
                    self.retired_pairs.get(slot as usize).copied().flatten(),
                    generation,
                ) {
                    return refused(&request, RefusalCode::RouteClosed);
                }
                self.core.rotate_agent_route(slot, generation).map(|()| {
                    self.retired_pairs[slot as usize] = None;
                })
            }
            ClientRequest::CloseAgent {
                slot, generation, ..
            } => {
                if endpoint != Endpoint::Broker {
                    return refused(&request, RefusalCode::PolicyDenied);
                }
                let Some(pair) = crate::agent_pair_index(slot)
                    .and_then(|index| self.pairs.get_mut(index))
                    .and_then(Option::as_mut)
                else {
                    return refused(&request, RefusalCode::RouteClosed);
                };
                self.core
                    .close_agent_route(slot, generation)
                    .and_then(|()| {
                        if crate::drain_retired_requests(&mut pair.rings) {
                            self.retired_pairs[slot as usize] = Some(generation);
                            Ok(())
                        } else {
                            Err(ServiceError::Malformed)
                        }
                    })
            }
        };
        match result {
            Ok(()) => ClientResponse::Accepted {
                request: request_id,
            },
            Err(error) => ClientResponse::Refused {
                request: request_id,
                code: service_code(error),
            },
        }
    }

    /// Drain the coalesced shared wake completely enough to make parking safe.
    /// `true` means the bounded pass remained saturated, so the caller loops
    /// productively instead of parking or polling an empty queue.
    fn drive_link(&mut self, irq_fired: bool) -> bool {
        let mut irq = OneShotIrq(irq_fired);
        let _ = self.core.transport_mut().serial_mut().pump(
            &mut irq,
            SerialPumpBudget {
                device_control_events: 16,
                control_write_calls: 16,
                bulk_write_calls: 4,
            },
        );
        let _ = self.core.transport_mut().flush_wake();
        let shared = self.core.transport_mut().is_shared();
        let mut saturated = false;
        for _ in 0..SHMEM_DRAIN_RECHECK_PASSES {
            let report = self.core.drain(&mut self.control_work, &mut self.bulk_work);
            saturated = needs_bulk_recheck(shared, report.bulk_reads, BULK_DRAIN_BUDGET);
            if !saturated || report.disconnected {
                break;
            }
        }
        while let Some(status) = self.core.pop_status() {
            if self.core.delivery_mut().push_status(status).is_err() {
                break;
            }
        }
        saturated
    }
}

struct OneShotIrq(bool);

impl kore_virtio::pump::IrqSource for OneShotIrq {
    fn observe(
        &mut self,
    ) -> Result<kore_virtio::pump::IrqNotice, kore_virtio::pump::BytePortError> {
        Ok(if core::mem::take(&mut self.0) {
            kore_virtio::pump::IrqNotice::Fired
        } else {
            kore_virtio::pump::IrqNotice::Quiet
        })
    }
}

pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    let boot = rt.bootstrap();
    let Ok(endowment) = HostLinkEndowment::resolve(boot) else {
        refuse_with(&rt, rung::ENDOWMENT);
    };
    let mut front = HostLinkFront::bind(endowment);
    let Some(pool) = front.take_pool() else {
        refuse_with(&rt, rung::ENDOWMENT);
    };
    let Some(mut broker) = broker_client(kore_rt::entry::transport(), boot) else {
        refuse_with(&rt, rung::DEVICE_FACE);
    };
    let claimed = match broker
        .claim_first(crate::front::HOSTLINK_VIRTIO_CONSOLE.selector)
        .await
    {
        Ok(claimed) => claimed,
        Err(kore_svc_device::service::DeviceRefusal::NoMatch) => serve_disconnected(&rt).await,
        Err(_) => refuse_with(&rt, rung::DEVICE_CLAIM),
    };
    let ClaimedDevice::Pci(device) = claimed.device else {
        refuse_with(&rt, rung::DEVICE_CLAIM);
    };
    let bound = match bind_console(device, pool) {
        Ok(bound) => bound,
        Err(error) => refuse_with(&rt, bind_refusal(error)),
    };
    let ports = BoundConsolePorts::new(bound);
    if !ports.has_route() {
        refuse_with(&rt, rung::DEVICE_BIND);
    }
    kore_rt::executor::enable_irq_parks().await;

    let control_stage = vec![0; crate::virtio::CONTROL_TX_STAGING_BYTES].into_boxed_slice();
    let bulk_stage = vec![0; crate::virtio::BULK_TX_STAGING_BYTES].into_boxed_slice();
    let Ok(mut adapter) = VirtioHostLink::new(ports, control_stage, bulk_stage) else {
        refuse_with(&rt, rung::DEVICE_BIND);
    };

    // The controller sends PORT_ADD/NAME/OPEN on q2. Readiness requires both
    // exact names to have reached their fixed plane bindings.
    let startup_deadline = 2_000_000_000;
    loop {
        let mut irq = OneShotIrq(false);
        match adapter.pump(
            &mut irq,
            SerialPumpBudget {
                device_control_events: 16,
                control_write_calls: 16,
                bulk_write_calls: 4,
            },
        ) {
            Ok(_) | Err(PortError::Backpressure) => {}
            Err(error) => refuse_with(&rt, port_refusal(error)),
        }
        if adapter.ports_mut().planes_bound() {
            break;
        }
        if kore_rt::executor::next_wake_for(startup_deadline).await
            == kore_rt::executor::WakeReason::Deadline
        {
            refuse_with(&rt, rung::PORTS);
        }
    }

    let shared = match claim_optional_bulk(&mut broker).await {
        Ok(shared) => shared,
        Err(error) => refuse_with(&rt, shmem_refusal(error)),
    };
    let adapter = match shared {
        Some(window) => RuntimeBulk::shared(adapter, window),
        None => RuntimeBulk::serial(adapter),
    };

    let mut seed = [0u8; 96];
    if kore_rt::random::fill(&mut seed).is_err() {
        refuse_with(&rt, rung::ENTROPY);
    }
    let generation =
        TaskGeneration::new(crate::BOOTSTRAP_AGENT_GENERATION).expect("nonzero generation");
    let bindings = Bindings::new(
        LocalRoute {
            route: RouteId(seed[32..48].try_into().expect("fixed route slice")),
            generation,
        },
        LocalRoute {
            route: RouteId(seed[80..96].try_into().expect("fixed route slice")),
            generation,
        },
    );
    let Some(bindings) = bindings else {
        refuse_with(&rt, rung::ENTROPY);
    };
    let Some(pairs) = attach_pairs(boot) else {
        refuse_with(&rt, rung::PAIRS);
    };
    let mut core = HostLinkService::new(
        adapter,
        MailboxBank::new(),
        seed[..32].try_into().unwrap(),
        if crate::SUPERVISOR_AUTOSTART {
            bindings
        } else {
            bindings.without_broker()
        },
    );
    if crate::SUPERVISOR_AUTOSTART {
        for (slot, range) in [(0, 48..64), (1, 64..80)] {
            let local = LocalRoute {
                route: RouteId(seed[range].try_into().expect("fixed route slice")),
                generation,
            };
            if core.install_agent_route(slot, local).is_err() {
                refuse_with(&rt, rung::ENTROPY);
            }
        }
    }
    let mut router = Router {
        core,
        supervisor: crate::SupervisorOwner::new(generation),
        transport: kore_rt::entry::transport(),
        pairs,
        retired_pairs: [None; crate::MAX_DYNAMIC_AGENT_ROUTES],
        control_work: vec![0; CONTROL_FRAME_MAX].into_boxed_slice(),
        bulk_work: vec![0; BULK_FRAME_MAX].into_boxed_slice(),
    };

    let Some(beacon) = boot.handle(role::PARENT_SEND, 0) else {
        refuse_with(&rt, rung::BEACON);
    };
    if ChanTx::new(kore_rt::entry::transport(), beacon)
        .send(Notice {
            cookie: COOKIE_SERVING,
            payload: 0,
        })
        .await
        .is_err()
    {
        refuse_with(&rt, rung::BEACON);
    }

    let mut irq_fired = false;
    loop {
        if kore_rt::executor::take_quiesce_ask().await.is_some() {
            // Moving the core makes further byte access impossible. BoundDevice
            // performs the actual reset and fabric/containment proof before park.
            let _ = router.core.into_transport().into_serial().teardown();
            loop {
                kore_rt::executor::next_wake().await;
            }
        }
        let mut busy = router.drive_link(irq_fired);
        router.drain_requests().await;
        // Notification-recheck: drain once after all local work and before
        // arming the park. A publication after this empty observation carries
        // a serial IRQ; a saturated result stays runnable.
        busy |= router.drive_link(false);
        if busy {
            irq_fired = false;
            continue;
        }
        irq_fired = kore_rt::executor::next_wake().await == kore_rt::executor::WakeReason::Irq;
    }
}

/// Ordinary machines need no host transport. Keep their local service faces
/// responsive with an explicit disconnected result so the rest of boot can
/// proceed; malformed or already-claimed hardware still refuses above.
async fn serve_disconnected(rt: &kore_rt::entry::Rt) -> ! {
    let Some(mut pairs) = attach_pairs(rt.bootstrap()) else {
        refuse_with(rt, rung::PAIRS);
    };
    let Some(beacon) = rt.bootstrap().handle(role::PARENT_SEND, 0) else {
        refuse_with(rt, rung::BEACON);
    };
    let transport = kore_rt::entry::transport();
    if ChanTx::new(transport, beacon)
        .send(Notice {
            cookie: COOKIE_SERVING,
            payload: rung::DEVICE_CLAIM,
        })
        .await
        .is_err()
    {
        refuse_with(rt, rung::BEACON);
    }
    loop {
        if kore_rt::executor::take_quiesce_ask().await.is_some() {
            loop {
                kore_rt::executor::next_wake().await;
            }
        }
        for pair in pairs.iter_mut().flatten() {
            for _ in 0..HostLinkWire::REQUEST_CAPACITY {
                let Some(wire) = pair.rings.req.pop() else {
                    break;
                };
                let response = match HostLinkWire::decode_req(&wire) {
                    Some(ClientRequest::Status) => ClientResponse::Status(crate::ClientStatus {
                        phase: crate::ClientLinkPhase::Disconnected,
                        bulk_transport: BulkTransport::Serial,
                        queued_status: 0,
                        queued_control: 0,
                        queued_bulk: 0,
                    }),
                    Some(ClientRequest::Poll) => ClientResponse::Empty,
                    Some(request) => refused(&request, RefusalCode::HostDisconnected),
                    None => ClientResponse::Refused {
                        request: kore_hostlink::RequestId::ZERO,
                        code: RefusalCode::Malformed,
                    },
                };
                if pair
                    .rings
                    .rep
                    .push(HostLinkWire::encode_rep(&response, 0))
                    .is_ok()
                {
                    let _ = ChanTx::new(transport, pair.kick)
                        .send(Notice {
                            cookie: COOKIE_REPLY,
                            payload: 0,
                        })
                        .await;
                }
            }
        }
        kore_rt::executor::next_wake().await;
    }
}

fn shmem_refusal(error: ShmemClaimError) -> u64 {
    let detail = match error {
        ShmemClaimError::Broker(kore_svc_device::service::DeviceRefusal::NoMatch) => 1,
        ShmemClaimError::Broker(kore_svc_device::service::DeviceRefusal::Taken) => 2,
        ShmemClaimError::Broker(_) => 3,
        ShmemClaimError::WrongSpecies => 4,
        ShmemClaimError::MapRefused => 5,
    };
    rung::SHMEM | (detail << 8)
}

fn broker_client<'t>(
    transport: &'t kore_rt::transport::TrapTransport,
    boot: &Bootstrap<'_>,
) -> Option<MultiBroker<'static, 't, kore_rt::transport::TrapTransport>> {
    MultiBroker::open(
        transport,
        boot,
        FaceLayout::for_row(HostLinkFront::SERVE_PAIRS, HostLinkFront::REPLY_GRANTS),
    )
}

async fn claim_optional_bulk(
    brokers: &mut MultiBroker<'static, '_, kore_rt::transport::TrapTransport>,
) -> Result<Option<MappedSharedRam>, ShmemClaimError> {
    let claimed = match brokers
        .claim_first(crate::front::HOSTLINK_IVSHMEM_PLAIN.selector)
        .await
    {
        Ok(claimed) => claimed,
        Err(kore_svc_device::service::DeviceRefusal::NoMatch) => return Ok(None),
        Err(error) => return Err(ShmemClaimError::Broker(error)),
    };
    let ClaimedDevice::Pci(device) = claimed.device else {
        return Err(ShmemClaimError::WrongSpecies);
    };
    MappedSharedRam::new(device)
        .map(Some)
        .map_err(|_| ShmemClaimError::MapRefused)
}

fn attach_pairs(boot: &Bootstrap<'_>) -> Option<[Option<Pair<'static>>; HOSTLINK_SERVE_PAIRS]> {
    let mut pairs = [const { None }; HOSTLINK_SERVE_PAIRS];
    for (index, slot) in pairs.iter_mut().enumerate() {
        let Some(view) = boot.handle(role::SVC_RING, index as u16) else {
            if crate::SUPERVISOR_AUTOSTART || index == 0 || index == 3 {
                return None;
            }
            continue;
        };
        let leased = kore_rt::cap::verbs::take_view(view)?;
        let window = kore_rt::cap::verbs::map_view(leased)?;
        // SAFETY: each index is a distinct founder-minted pair extent; this
        // task is its sole serving-side attacher and keeps it mapped forever.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        let rings = serve_rings::<HostLinkWire>(
            window,
            HostLinkWire::REQUEST_CAPACITY,
            HostLinkWire::RESPONSE_CAPACITY,
        )
        .ok()?;
        *slot = Some(Pair {
            rings,
            kick: boot.handle(role::SVC_SEND, index as u16)?,
        });
    }
    Some(pairs)
}

fn refused(request: &ClientRequest, code: RefusalCode) -> ClientResponse {
    ClientResponse::Refused {
        request: request.request_id(),
        code,
    }
}

fn mailbox_code(error: MailboxError) -> RefusalCode {
    match error {
        MailboxError::UnknownPair => RefusalCode::PolicyDenied,
        MailboxError::Backpressure => RefusalCode::Backpressure,
    }
}

fn service_code(error: ServiceError) -> RefusalCode {
    match error {
        ServiceError::Transport(kore_hostlink::PortError::Backpressure)
        | ServiceError::Backpressure => RefusalCode::Backpressure,
        ServiceError::Transport(kore_hostlink::PortError::Disconnected) => {
            RefusalCode::HostDisconnected
        }
        ServiceError::Transport(kore_hostlink::PortError::Fault) => RefusalCode::Internal,
        ServiceError::Decode(_) | ServiceError::Malformed => RefusalCode::Malformed,
        ServiceError::Link | ServiceError::Route => RefusalCode::RouteClosed,
        ServiceError::Bulk => RefusalCode::OutOfOrder,
    }
}

const _: () = {
    assert!(CONTROL_FRAME_MAX <= 64 * 1024);
    assert!(BULK_FRAME_MAX <= 1024 * 1024 + 128);
    assert!(<HostLinkFront as SvcProtocol>::SERVE_PAIRS == HOSTLINK_SERVE_PAIRS);
    assert!(<HostLinkFront as SystemService>::DEVICES.len() == 2);
};
