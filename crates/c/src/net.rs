//! Thread zero holds the two process faces and serves the C program's network
//! requests. Connected descriptors retain the vif and its window on this thread.

extern crate alloc;

use alloc::boxed::Box;
use core::ffi::c_int;
use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::pin;
use core::ptr::NonNull;
use core::task::{Context, Poll, Waker};

use kore_inet::flow::{FlowCap, Powers, Scope};
use kore_inet::netinet::ping::{Heard, Ping};
use kore_inet::netinet::tcp::{SocketStorage, TcpError, TcpStorage};
use kore_inet::smol::SMOL_SCRATCH_BYTES;
use kore_inet::{
    Event, FlowFace, Identity, Ifnet, IpStack, OutputError, SocketRefusal, TcpStream, UdpSocket,
    VifIfnet,
};
use kore_std::abi::clock::Uptime;
use kore_std::driver::net::{FlowMatch, IpProto, LinkState, MacAddr, VifId};
use kore_std::runtime::cap::{MappedLeaseView, StashedLease, verbs};
use kore_std::runtime::entry::Rt;
use kore_std::runtime::svc::StagedReturnError;
use kore_std::runtime::transport::TrapTransport;
use kore_std::runtime::{clock, executor::sleep_for};
use kore_svc_mm::{MmClient, client::ProcessMm};
use kore_svc_net::client::{ProcessNet, WINDOW_RIGHTS, attach_process_face, format_window};
use kore_svc_net::window::{WINDOW_BYTES, WINDOW_LEASE_BYTES};
use kore_svc_net::{EtherType, EtherTypeMatch, FrameFilter, NetRefusal, NetRequest, NetResponse};

use crate::api::KoreNetStatus as Status;
use crate::fd::{CAPACITY, FIRST, TABLE, errno};
use crate::socket::{Endpoint, Request};

pub(crate) struct Host<'t> {
    network: Option<ProcessNet<'t>>,
    memory: Option<ProcessMm>,
    sockets: [Option<Socket>; CAPACITY],
}

impl<'t> Host<'t> {
    pub(crate) fn new(rt: &Rt, transport: &'t TrapTransport) -> Self {
        let network = attach_process_face(rt, transport);
        let memory = network
            .as_ref()
            .and_then(|_| kore_svc_mm::attach_process_face(rt));
        Self {
            network,
            memory,
            sockets: [const { None }; CAPACITY],
        }
    }

    /// # Safety
    /// The parked C caller owns all buffers until the session is finished.
    pub(crate) async unsafe fn socket(&mut self, request: Request) -> Result<usize, c_int> {
        let Self {
            network,
            memory,
            sockets,
        } = self;
        let network = network.as_mut().ok_or(errno::EACCES)?;
        let memory = memory.as_mut().ok_or(errno::EACCES)?;
        let fd = match request {
            Request::Open(protocol) => {
                let fd = TABLE.claim_socket()?;
                sockets[(fd - FIRST) as usize] = Some(Socket::Unconnected(protocol));
                return Ok(fd as usize);
            }
            Request::Connect(fd, _)
            | Request::Close(fd)
            | Request::Send(fd, ..)
            | Request::Recv(fd, ..) => fd,
        };
        TABLE.require_socket(fd)?;
        let entry = &mut sockets[(fd - FIRST) as usize];
        let socket = entry.as_mut().ok_or(errno::EBADF)?;
        match request {
            Request::Open(_) => unreachable!(),
            Request::Connect(_, peer) => {
                let Socket::Unconnected(protocol) = *socket else {
                    return Err(errno::EISCONN);
                };
                *socket =
                    Socket::Connected(Connected::open(network, memory, protocol, peer).await?);
                Ok(0)
            }
            Request::Close(_) => {
                let mut finished = Ok(());
                if let Socket::Connected(connected) = socket {
                    if let Transport::Tcp(tcp) = &mut connected.transport {
                        finished = match deadline() {
                            Ok(deadline) => {
                                drive(network, tcp.stream.finish_write(), Some(deadline), true)
                                    .await
                            }
                            Err(code) => Err(code),
                        };
                    }
                    retire_vif(network, connected.vif).await?;
                }
                if let Some(Socket::Connected(Connected {
                    transport, window, ..
                })) = entry.take()
                {
                    drop(transport);
                    window.release();
                }
                TABLE.release_socket(fd)?;
                finished.map(|()| 0)
            }
            Request::Send(_, buf, len) => {
                let Socket::Connected(conn) = socket else {
                    return Err(errno::ENOTCONN);
                };
                let bytes = if len == 0 {
                    &[]
                } else {
                    // SAFETY: held by the caller while this request runs.
                    unsafe { core::slice::from_raw_parts(buf, len) }
                };
                let deadline = Some(deadline()?);
                match &mut conn.transport {
                    Transport::Udp(udp) => drive(network, udp.send(bytes), deadline, true).await,
                    Transport::Tcp(tcp) => {
                        drive(network, tcp.stream.write(bytes), deadline, true).await
                    }
                }
            }
            Request::Recv(_, buf, len) => {
                let Socket::Connected(conn) = socket else {
                    return Err(errno::ENOTCONN);
                };
                let bytes = if len == 0 {
                    &mut []
                } else {
                    // SAFETY: held exclusively by the caller while this request runs.
                    unsafe { core::slice::from_raw_parts_mut(buf, len) }
                };
                match &mut conn.transport {
                    Transport::Udp(udp) => drive(network, udp.recv(bytes), None, false).await,
                    Transport::Tcp(tcp) => drive(network, tcp.stream.read(bytes), None, true).await,
                }
            }
        }
    }

    pub(crate) fn has_tcp(&self) -> bool {
        self.sockets.iter().any(|entry| {
            matches!(
                entry,
                Some(Socket::Connected(Connected {
                    transport: Transport::Tcp(_),
                    ..
                }))
            )
        })
    }

    pub(crate) async fn sweep_tcp(&mut self) {
        let mut kick = false;
        for entry in &mut self.sockets {
            if let Some(Socket::Connected(Connected {
                transport: Transport::Tcp(tcp),
                ..
            })) = entry
            {
                tcp.stream.sweep();
                kick |= tcp.stream.flow_mut().net.take_tx_kick();
            }
        }
        if kick && let Some(network) = &self.network {
            network.client.kick().await;
        }
    }

    pub(crate) async fn ping(&mut self, address: [u8; 4], timeout_ms: u32) -> Result<u64, Status> {
        let network = self.network.as_mut().ok_or(Status::Unavailable)?;
        let memory = self.memory.as_mut().ok_or(Status::Unavailable)?;
        let client = &mut network.client;
        let Some(NetResponse::Identity {
            identity: Some(report),
        }) = client.try_request(NetRequest::InetQuery).await
        else {
            return Err(Status::NoAddress);
        };
        let station = Identity::new(report.addr, report.prefix_len, report.gateway)
            .map_err(|_| Status::NoAddress)?;
        let mut filter = FrameFilter::protocol(EtherType::IPV4);
        filter.ethertype =
            EtherTypeMatch::only(&[EtherType::IPV4, EtherType::ARP]).ok_or(Status::Resource)?;
        let (window, vif, mac) = Window::open(network, memory, filter)
            .await
            .map_err(|error| match error {
                WindowOpenError::Indeterminate => Status::Indeterminate,
                _ => Status::Resource,
            })?;
        let client = &mut network.client;
        // SAFETY: the stack drops before the window is retired.
        let result = match unsafe { window.attach(mac) } {
            Ok(ifnet) => {
                let mut stack = IpStack::new(ifnet, station);
                let timeout_us = u64::from(timeout_ms) * 1_000;
                let start = now_us();
                let mut ping = Ping::new(
                    kore_std::driver::net::flow::Ipv4Addr(address),
                    0x6b43,
                    timeout_us,
                    timeout_us,
                    Some(1),
                );
                loop {
                    let (Some(start), Some(now)) = (start, now_us()) else {
                        break Err(Status::Unavailable);
                    };
                    if now.saturating_sub(start) >= timeout_us {
                        break Err(Status::Timeout);
                    }
                    if ping.send_due(&mut stack, now).is_err() {
                        break Err(Status::Unreachable);
                    }
                    let mut reply = None;
                    stack.input(now / 1_000, |event| {
                        if let Event::Icmp(icmp) = event {
                            match ping.absorb(icmp, now) {
                                Some(Heard::Reply(echo)) => reply = Some(Ok(echo.rtt_us)),
                                Some(Heard::Unreachable { .. }) if reply.is_none() => {
                                    reply = Some(Err(Status::Unreachable));
                                }
                                _ => {}
                            }
                        }
                    });
                    if stack.ifnet_mut().take_tx_kick() {
                        client.kick().await;
                    }
                    if let Some(reply) = reply {
                        break reply;
                    }
                    sleep_for(200_000).await;
                }
            }
            Err(_) => Err(Status::Resource),
        };
        retire_vif(network, vif)
            .await
            .map_err(|_| Status::Indeterminate)?;
        window.release();
        result
    }
}

enum Socket {
    Unconnected(IpProto),
    Connected(Connected),
}

enum Transport {
    Udp(Box<UdpSocket<RuntimeFlow>>),
    Tcp(Box<Tcp>),
}

struct Connected {
    // Drop the interface's ring references before the window owner.
    transport: Transport,
    window: Window,
    vif: VifId,
}

impl Connected {
    async fn open(
        network: &mut ProcessNet<'_>,
        memory: &mut ProcessMm,
        protocol: IpProto,
        peer: Endpoint,
    ) -> Result<Self, c_int> {
        let client = &mut network.client;
        let Some(NetResponse::Identity {
            identity: Some(report),
        }) = client.try_request(NetRequest::InetQuery).await
        else {
            return Err(errno::ENETUNREACH);
        };
        let identity = Identity::new(report.addr, report.prefix_len, report.gateway)
            .map_err(|_| errno::ENETUNREACH)?;
        let deadline = (protocol == IpProto::TCP).then(deadline).transpose()?;
        let storage = if protocol == IpProto::TCP {
            PendingTransport::Tcp(
                Box::<Tcp>::try_new_uninit().map_err(|_| errno::ENOMEM)?,
                TcpBufferOwner::new()?,
                kore_std::random::u64().map_err(|_| errno::EIO)?,
            )
        } else {
            PendingTransport::Udp(
                Box::<UdpSocket<RuntimeFlow>>::try_new_uninit().map_err(|_| errno::ENOMEM)?,
            )
        };
        let (window, vif, mac) = Window::open(network, memory, FrameFilter::CLOSED)
            .await
            .map_err(|error| match error {
                WindowOpenError::Resource => errno::ENOMEM,
                WindowOpenError::Refused(reason) => net_errno(reason),
                WindowOpenError::Indeterminate => errno::EIO,
            })?;
        let client = &mut network.client;
        let flow = client
            .try_request(NetRequest::OpenFlow {
                vif,
                proto: protocol,
                local_port: None,
                peer,
            })
            .await;
        let attached = match flow {
            Some(NetResponse::Flow {
                proto,
                local,
                peer: Some(actual),
            }) if proto == protocol
                && actual == peer
                && local.0 == identity.addr()
                && local.1 != 0 =>
            {
                let scope = Scope::Flow(FlowMatch::connection(proto, local, actual));
                // The acknowledgement describes the enforced flow. This
                // local cap is its recipe. Transmit and receive are still
                // gated by the service's installed filter on this vif.
                FlowCap::mint(vif, scope, Powers::NONE)
                    .map_err(|_| errno::EIO)
                    .and_then(|cap| {
                        // SAFETY: window stays owned until the socket is
                        // dropped, even on cancellation or failed close.
                        let net = unsafe { window.attach(mac) }?;
                        let flow = RuntimeFlow { net, cap, identity };
                        storage.attach(flow, peer)
                    })
            }
            Some(NetResponse::Refused(reason)) => Err(net_errno(reason)),
            _ => Err(errno::EIO),
        };
        let attached = match attached {
            Ok(Transport::Tcp(mut tcp)) => {
                match drive(network, tcp.stream.ready(), deadline, true).await {
                    Ok(()) => Ok(Transport::Tcp(tcp)),
                    Err(errno::ECONNRESET) => Err(errno::ECONNREFUSED),
                    Err(code) => Err(code),
                }
            }
            other => other,
        };
        match attached {
            Ok(transport) => Ok(Self {
                transport,
                window,
                vif,
            }),
            Err(code) => {
                retire_vif(network, vif).await?;
                window.release();
                Err(code)
            }
        }
    }
}

enum PendingTransport {
    Udp(Box<MaybeUninit<UdpSocket<RuntimeFlow>>>),
    Tcp(Box<MaybeUninit<Tcp>>, TcpBufferOwner, u64),
}

impl PendingTransport {
    fn attach(self, flow: RuntimeFlow, peer: Endpoint) -> Result<Transport, c_int> {
        match self {
            Self::Udp(place) => {
                let udp = UdpSocket::open(flow).map_err(socket_errno)?;
                Ok(Transport::Udp(Box::write(place, udp)))
            }
            Self::Tcp(place, buffers, seed) => {
                // SAFETY: the storage is lent once, and the stream field drops
                // before its backing allocation. On failure the borrow ends
                // within open, before the owner drops.
                let storage = unsafe { buffers.lend() };
                let stream =
                    TcpStream::open_with_seed(flow, storage, peer, seed).map_err(socket_errno)?;
                Ok(Transport::Tcp(Box::write(
                    place,
                    Tcp {
                        stream,
                        _buffers: buffers,
                    },
                )))
            }
        }
    }
}

struct Tcp {
    stream: TcpStream<'static, RuntimeFlow>,
    _buffers: TcpBufferOwner,
}

const TCP_BUFFER_BYTES: usize = 16 * 1024;

struct TcpBuffers {
    rx: [u8; TCP_BUFFER_BYTES],
    tx: [u8; TCP_BUFFER_BYTES],
    scratch: [u8; SMOL_SCRATCH_BYTES],
    sockets: [SocketStorage<'static>; 1],
}

// Keep ownership in a raw pointer so moving the owner alongside its borrower
// does not retag the memory behind the stream's references.
struct TcpBufferOwner(NonNull<TcpBuffers>);

impl TcpBufferOwner {
    fn new() -> Result<Self, c_int> {
        let buffers = Box::<TcpBuffers>::try_new_zeroed().map_err(|_| errno::ENOMEM)?;
        let raw = Box::into_raw(buffers).cast::<TcpBuffers>();
        // SAFETY: the byte arrays are already zeroed. Initialize the socket
        // slot explicitly before reading or dropping the complete object.
        unsafe { (&raw mut (*raw).sockets).write([SocketStorage::EMPTY]) };
        Ok(Self(NonNull::new(raw).expect("Box allocation is non-null")))
    }

    /// # Safety
    /// Call once per owner and drop all borrowers before dropping the owner.
    unsafe fn lend(&self) -> TcpStorage<'static> {
        let buffers = unsafe { &mut *self.0.as_ptr() };
        TcpStorage {
            rx: &mut buffers.rx,
            tx: &mut buffers.tx,
            scratch: &mut buffers.scratch,
            sockets: &mut buffers.sockets,
        }
    }
}

impl Drop for TcpBufferOwner {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the original boxed allocation.
        // The stream precedes this owner in Tcp's drop order.
        unsafe {
            let raw = self.0.as_ptr();
            (&raw mut (*raw).sockets).drop_in_place();
            drop(Box::from_raw(raw.cast::<MaybeUninit<TcpBuffers>>()));
        }
    }
}

/// A published window stays live when cancellation or an ambiguous transport
/// error leaves custody unknown. Only a confirmed close permits retirement.
struct Window(Option<(MappedLeaseView, StashedLease)>);

enum WindowOpenError {
    Resource,
    Refused(NetRefusal),
    Indeterminate,
}

impl Window {
    async fn open(
        network: &mut ProcessNet<'_>,
        memory: &mut ProcessMm,
        filter: FrameFilter,
    ) -> Result<(Self, VifId, MacAddr), WindowOpenError> {
        use WindowOpenError::{Indeterminate, Refused, Resource};
        let (_, lease) = memory.client.supply(WINDOW_LEASE_BYTES as u64).await;
        let lease = lease.ok_or(Resource)?;
        let slot = lease.slot().ok_or(Resource)?;
        let mut view = verbs::map_owned_lease_rw(slot).ok_or(Resource)?;
        if view.len() < WINDOW_LEASE_BYTES {
            return Err(Resource);
        }
        // SAFETY: the lease has not been shared with the service yet.
        let bytes = unsafe { view.as_mut() };
        format_window(&mut bytes[..WINDOW_BYTES]).map_err(|_| Resource)?;
        let window = Self(Some((view, lease)));
        let (_, lease) = window.0.as_ref().expect("live window");
        match network
            .client
            .open_vif_staged(filter, |stage| {
                stage.sublease(lease, WINDOW_RIGHTS, 0, WINDOW_LEASE_BYTES as u64)
            })
            .await
        {
            Ok(NetResponse::Opened { vif, mac }) => Ok((window, vif, mac)),
            Ok(NetResponse::Refused(reason)) => {
                window.release();
                Err(Refused(reason))
            }
            Err(StagedReturnError::NotTransferred(_)) => {
                window.release();
                Err(Resource)
            }
            _ => Err(Indeterminate),
        }
    }

    /// # Safety
    /// The interface is the sole client attachment and must be dropped
    /// before retirement. The peer uses the service's SPSC window protocol.
    unsafe fn attach(&self, mac: MacAddr) -> Result<VifIfnet<'static>, c_int> {
        let (view, _) = self.0.as_ref().expect("live window");
        let bytes = unsafe { core::slice::from_raw_parts_mut(view.as_mut_ptr(), WINDOW_BYTES) };
        VifIfnet::attach(bytes, mac).map_err(|_| errno::EIO)
    }

    fn release(mut self) {
        if let Some((mut view, lease)) = self.0.take() {
            if view.try_retire_outcome().is_closed() {
                drop(view);
                drop(lease);
            } else {
                core::mem::forget((view, lease));
            }
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        if let Some(held) = self.0.take() {
            core::mem::forget(held);
        }
    }
}

async fn retire_vif(network: &mut ProcessNet<'_>, vif: VifId) -> Result<(), c_int> {
    match network
        .client
        .try_request(NetRequest::CloseVif { vif })
        .await
    {
        Some(NetResponse::Closed { vif: closed }) if closed == vif => Ok(()),
        Some(NetResponse::Refused(NetRefusal::StaleVif | NetRefusal::NoVif)) => Ok(()),
        _ => Err(errno::EIO),
    }
}

struct RuntimeFlow {
    net: VifIfnet<'static>,
    cap: FlowCap,
    identity: Identity,
}

impl Ifnet for RuntimeFlow {
    type TxSlot<'a> = <VifIfnet<'static> as Ifnet>::TxSlot<'a>;

    fn hwaddr(&self) -> MacAddr {
        self.net.hwaddr()
    }
    fn mtu(&self) -> usize {
        self.net.mtu()
    }
    fn link(&self) -> LinkState {
        self.net.link()
    }
    fn reserve(&mut self) -> Option<Self::TxSlot<'_>> {
        self.net.reserve()
    }
    fn poll<F: FnMut(&[u8], EtherType)>(&mut self, f: F) -> bool {
        self.net.poll(f)
    }
}

impl FlowFace for RuntimeFlow {
    fn cap(&self) -> &FlowCap {
        &self.cap
    }
    fn identity(&self) -> &Identity {
        &self.identity
    }
    fn now_ms(&self) -> u64 {
        now_us().unwrap_or(0) / 1_000
    }
}

/// Until the flow face carries receive readiness, drive it from a timed
/// executor wait. Its self-wake is replaced by that wait, and we kick the
/// row for transmit progress before checking the next frame.
async fn drive<T>(
    network: &ProcessNet<'_>,
    operation: impl Future<Output = Result<T, SocketRefusal>>,
    deadline_us: Option<u64>,
    kick: bool,
) -> Result<T, c_int> {
    let mut operation = pin!(operation);
    loop {
        let polled = operation
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()));
        if kick {
            network.client.kick().await;
        }
        if let Poll::Ready(result) = polled {
            return result.map_err(socket_errno);
        }
        if let Some(deadline) = deadline_us
            && now_us().ok_or(errno::EIO)? >= deadline
        {
            return Err(errno::ETIMEDOUT);
        }
        sleep_for(1_000_000).await;
    }
}

fn net_errno(reason: NetRefusal) -> c_int {
    match reason {
        NetRefusal::NotReady | NetRefusal::NoIdentity | NetRefusal::Unreachable => {
            errno::ENETUNREACH
        }
        NetRefusal::PolicyVifs
        | NetRefusal::Exhausted
        | NetRefusal::NoEphemeral
        | NetRefusal::PortsExhausted
        | NetRefusal::SocketsExhausted => errno::EAGAIN,
        NetRefusal::PolicyEtherType
        | NetRefusal::PolicyMulticast
        | NetRefusal::PolicyBroadcast
        | NetRefusal::PolicyPromiscuous
        | NetRefusal::PolicyFlowRequired
        | NetRefusal::PolicyFlowProto
        | NetRefusal::PolicyFlowLocal
        | NetRefusal::PolicyFlowPeer
        | NetRefusal::FlowPolicy
        | NetRefusal::NotOwner => errno::EACCES,
        _ => errno::EIO,
    }
}

fn socket_errno(reason: SocketRefusal) -> c_int {
    match reason {
        SocketRefusal::Output(OutputError::Unreachable) => errno::ENETUNREACH,
        SocketRefusal::Output(OutputError::TooLong) => errno::EMSGSIZE,
        SocketRefusal::OutsideFlow => errno::EACCES,
        SocketRefusal::Tcp(TcpError::Unreachable) => errno::ENETUNREACH,
        SocketRefusal::Reset => errno::ECONNRESET,
        SocketRefusal::Closed => errno::EPIPE,
        _ => errno::EIO,
    }
}

fn deadline() -> Result<u64, c_int> {
    Ok(now_us().ok_or(errno::EIO)?.saturating_add(3_000_000))
}

fn now_us() -> Option<u64> {
    clock::uptime()
        .ok()
        .map(Uptime::as_nanos)
        .filter(|ns| *ns != 0)
        .map(|ns| ns / 1_000)
}
