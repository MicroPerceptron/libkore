//! Safe, fixed-buffer ownership layer for the HostLink VirtIO console.
//!
//! A [`SplitQueue`](crate::split::SplitQueue) intentionally exposes an unsafe
//! submission seam because a raw IOVA does not prove its payload remains
//! alive.  This module closes that seam: every queue and its sole payload
//! buffer have the same owner, and a buffer is never exposed while its token
//! is in flight.  The resulting public byte API is safe and allocation-free.

use kore_driver::{Dma, DmaPool, DmaRead, DmaWrite};

use crate::console::{
    CONTROL_HEADER_LEN, CONTROL_RX_QUEUE, CONTROL_TX_QUEUE, ConsoleError, ControlAction,
    ControlCommand, HOSTLINK_BULK_RX_QUEUE, HOSTLINK_BULK_TX_QUEUE, HOSTLINK_CONTROL_RX_QUEUE,
    HOSTLINK_CONTROL_TX_QUEUE, MAX_PORT_NAME_WIRE, MultiportConsole, Plane, PortHandle,
};
use crate::pci::{
    F_ACCESS_PLATFORM, F_CONSOLE_MULTIPORT, F_VERSION_1, TransportError, VirtioTransport,
};
use crate::split::{Direction, DmaSpan, QueueError, QueueToken, SplitQueue};

/// Fixed payload-bank and split-ring depth per queue. Sixteen 4 KiB buffers
/// keep 64 KiB in flight independently in each HostLink direction.
pub const QUEUE_SIZE: usize = 16;
/// One page per data payload, matching the HostLink protocol's bounded bulk
/// record and the capability pool's allocation granularity.
pub const PAYLOAD_BYTES: usize = 4096;
/// Exact footprint drawn from `CapPool`: three page-rounded ring allocations
/// plus one page per payload, for each of six queues.
pub const DMA_POOL_BYTES: usize = 6 * (3 + QUEUE_SIZE) * PAYLOAD_BYTES;

const CONFIG_VECTOR: u16 = 0;
const QUEUE_VECTOR: u16 = 1;
const REQUIRED_FEATURES: u64 = F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT;
// EVENT_IDX is intentionally not negotiated by this one-outstanding-buffer
// owner. The split queue supports it, but renegotiating it safely across reset
// requires preserving the exact queue posture in the owning layer.
const SUPPORTED_FEATURES: u64 = REQUIRED_FEATURES;
const CONTROL_FRAME_MAX: usize = CONTROL_HEADER_LEN + MAX_PORT_NAME_WIRE;

/// Safe-console failure. `WouldBlock` and `Backpressure` are ordinary
/// nonblocking outcomes; all other variants require caller policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnedError {
    WouldBlock,
    Backpressure,
    Disconnected(Plane),
    InvalidBufferSize,
    Queue(QueueError),
    Transport(TransportError),
    Console(ConsoleError),
    ResetRequired,
}

impl From<QueueError> for OwnedError {
    fn from(value: QueueError) -> Self {
        Self::Queue(value)
    }
}

impl From<TransportError> for OwnedError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value)
    }
}

impl From<ConsoleError> for OwnedError {
    fn from(value: ConsoleError) -> Self {
        Self::Console(value)
    }
}

/// Work performed by one bounded, control-first drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrainReport {
    pub control_events: u16,
    pub control_bytes: usize,
    pub bulk_bytes: usize,
}

struct RxLane<B: Dma, const N: usize> {
    queue: SplitQueue<B, N>,
    payloads: [Option<B>; N],
    tokens: [Option<QueueToken>; N],
    ready: [ReadyPayload; N],
    ready_head: usize,
    ready_len: usize,
}

#[derive(Clone, Copy)]
struct ReadyPayload {
    slot: u16,
    offset: usize,
    len: usize,
}

impl ReadyPayload {
    const EMPTY: Self = Self {
        slot: 0,
        offset: 0,
        len: 0,
    };
}

impl<B: Dma, const N: usize> RxLane<B, N> {
    fn allocate<P: DmaPool<Buf = B>>(
        pool: &P,
        bytes: usize,
        event_idx: bool,
    ) -> Result<Self, OwnedError> {
        let queue = SplitQueue::allocate(pool, event_idx)?;
        let mut payloads = [const { None }; N];
        for payload in &mut payloads {
            let buffer = pool.alloc(bytes).ok_or(QueueError::AllocationFailed)?;
            if buffer.len() < bytes {
                return Err(OwnedError::InvalidBufferSize);
            }
            *payload = Some(buffer);
        }
        Ok(Self {
            queue,
            payloads,
            tokens: [None; N],
            ready: [ReadyPayload::EMPTY; N],
            ready_head: 0,
            ready_len: 0,
        })
    }

    fn post(&mut self, slot: usize, bytes: usize) -> Result<bool, OwnedError> {
        if slot >= N || self.tokens[slot].is_some() || self.ready_contains(slot) {
            return Err(OwnedError::Backpressure);
        }
        let payload = self.payloads[slot]
            .as_ref()
            .ok_or(OwnedError::ResetRequired)?;
        let span = DmaSpan::from_buffer(payload, 0, bytes, Direction::DeviceWrites)?;
        // SAFETY: this lane owns every payload beside `queue`. `tokens` and
        // the ready FIFO keep a slot unavailable until its exact completion
        // is consumed and reposted; reset terminates every device borrow.
        let submission = unsafe { self.queue.submit(&[span])? };
        self.tokens[slot] = Some(submission.token);
        Ok(submission.notify)
    }

    fn post_all(&mut self, bytes: usize) -> Result<(), OwnedError> {
        for slot in 0..N {
            let _ = self.post(slot, bytes)?;
        }
        Ok(())
    }

    fn poll(&mut self) -> Result<bool, OwnedError> {
        while let Some(completion) = self.queue.complete_next()? {
            let slot = self
                .tokens
                .iter()
                .position(|token| *token == Some(completion.token))
                .ok_or(OwnedError::ResetRequired)?;
            self.tokens[slot] = None;
            if self.ready_len == N {
                return Err(OwnedError::ResetRequired);
            }
            let tail = (self.ready_head + self.ready_len) & (N - 1);
            self.ready[tail] = ReadyPayload {
                slot: slot as u16,
                offset: 0,
                len: completion.written as usize,
            };
            self.ready_len += 1;
        }
        Ok(self.ready_len != 0)
    }

    fn front_len(&mut self) -> Result<Option<usize>, OwnedError> {
        if !self.poll()? {
            return Ok(None);
        }
        Ok(Some(self.ready[self.ready_head].len))
    }

    fn read(&mut self, out: &mut [u8], bytes: usize) -> Result<(usize, bool), OwnedError> {
        if !self.poll()? {
            return Err(OwnedError::WouldBlock);
        }
        let ready = &mut self.ready[self.ready_head];
        let take = out.len().min(ready.len).min(bytes);
        self.payloads[usize::from(ready.slot)]
            .as_ref()
            .ok_or(OwnedError::ResetRequired)?
            .read_bytes_at(ready.offset, &mut out[..take])
            .map_err(|_| QueueError::Memory)?;
        ready.offset += take;
        ready.len -= take;
        if ready.len != 0 {
            return Ok((take, false));
        }
        let slot = usize::from(ready.slot);
        *ready = ReadyPayload::EMPTY;
        self.ready_head = (self.ready_head + 1) & (N - 1);
        self.ready_len -= 1;
        let notify = self.post(slot, bytes)?;
        Ok((take, notify))
    }

    fn reset(&mut self) -> Result<(), OwnedError> {
        self.queue.reset()?;
        self.tokens = [None; N];
        self.ready = [ReadyPayload::EMPTY; N];
        self.ready_head = 0;
        self.ready_len = 0;
        Ok(())
    }

    fn ready_contains(&self, slot: usize) -> bool {
        (0..self.ready_len).any(|offset| {
            usize::from(self.ready[(self.ready_head + offset) & (N - 1)].slot) == slot
        })
    }

    #[cfg(test)]
    fn device_write_for_test(&mut self, bytes: &[u8]) {
        let slot = self.tokens.iter().position(Option::is_some).unwrap();
        self.payloads[slot]
            .as_ref()
            .unwrap()
            .write_bytes_at(0, bytes)
            .unwrap();
        self.queue
            .device_complete_for_test(self.tokens[slot].unwrap(), bytes.len() as u32)
            .unwrap();
    }
}

struct TxLane<B: Dma, const N: usize> {
    queue: SplitQueue<B, N>,
    payloads: [Option<B>; N],
    tokens: [Option<QueueToken>; N],
}

impl<B: Dma, const N: usize> TxLane<B, N> {
    fn allocate<P: DmaPool<Buf = B>>(
        pool: &P,
        bytes: usize,
        event_idx: bool,
    ) -> Result<Self, OwnedError> {
        let queue = SplitQueue::allocate(pool, event_idx)?;
        let mut payloads = [const { None }; N];
        for payload in &mut payloads {
            let buffer = pool.alloc(bytes).ok_or(QueueError::AllocationFailed)?;
            if buffer.len() < bytes {
                return Err(OwnedError::InvalidBufferSize);
            }
            *payload = Some(buffer);
        }
        Ok(Self {
            queue,
            payloads,
            tokens: [None; N],
        })
    }

    fn poll(&mut self) -> Result<bool, OwnedError> {
        while let Some(completion) = self.queue.complete_next()? {
            if completion.written != 0 {
                return Err(OwnedError::ResetRequired);
            }
            let slot = self
                .tokens
                .iter()
                .position(|token| *token == Some(completion.token))
                .ok_or(OwnedError::ResetRequired)?;
            self.tokens[slot] = None;
        }
        Ok(self.tokens.iter().any(Option::is_none))
    }

    fn stage(&mut self, bytes: &[u8]) -> Result<bool, OwnedError> {
        if !self.poll()? {
            return Err(OwnedError::Backpressure);
        }
        if bytes.is_empty() {
            return Ok(false);
        }
        let slot = self
            .tokens
            .iter()
            .position(Option::is_none)
            .ok_or(OwnedError::Backpressure)?;
        let payload = self.payloads[slot]
            .as_ref()
            .ok_or(OwnedError::ResetRequired)?;
        if bytes.len() > payload.len() {
            return Err(OwnedError::InvalidBufferSize);
        }
        payload
            .write_bytes_at(0, bytes)
            .map_err(|_| QueueError::Memory)?;
        let span = DmaSpan::from_buffer(payload, 0, bytes.len(), Direction::DeviceReads)?;
        // SAFETY: `tokens[slot]` reserves this owned payload until its exact
        // completion. No safe method returns or mutates the buffer meanwhile.
        let submission = unsafe { self.queue.submit(&[span])? };
        self.tokens[slot] = Some(submission.token);
        Ok(submission.notify)
    }

    fn reset(&mut self) -> Result<(), OwnedError> {
        self.queue.reset()?;
        self.tokens = [None; N];
        Ok(())
    }

    #[cfg(test)]
    fn complete_one_for_test(&mut self) {
        let token = self.tokens.iter().flatten().next().copied().unwrap();
        self.queue.device_complete_for_test(token, 0).unwrap();
    }
}

struct Queues<B: Dma, const N: usize> {
    control_rx: RxLane<B, N>,
    control_tx: TxLane<B, N>,
    host_control_rx: RxLane<B, N>,
    host_control_tx: TxLane<B, N>,
    bulk_rx: RxLane<B, N>,
    bulk_tx: TxLane<B, N>,
}

impl<B: Dma, const N: usize> Queues<B, N> {
    fn allocate<P: DmaPool<Buf = B>>(
        pool: &P,
        bytes: usize,
        event_idx: bool,
    ) -> Result<Self, OwnedError> {
        if bytes < CONTROL_FRAME_MAX {
            return Err(OwnedError::InvalidBufferSize);
        }
        Ok(Self {
            control_rx: RxLane::allocate(pool, bytes, event_idx)?,
            control_tx: TxLane::allocate(pool, bytes, event_idx)?,
            host_control_rx: RxLane::allocate(pool, bytes, event_idx)?,
            host_control_tx: TxLane::allocate(pool, bytes, event_idx)?,
            bulk_rx: RxLane::allocate(pool, bytes, event_idx)?,
            bulk_tx: TxLane::allocate(pool, bytes, event_idx)?,
        })
    }

    fn reset(&mut self) -> Result<(), OwnedError> {
        self.control_rx.reset()?;
        self.control_tx.reset()?;
        self.host_control_rx.reset()?;
        self.host_control_tx.reset()?;
        self.bulk_rx.reset()?;
        self.bulk_tx.reset()?;
        Ok(())
    }
}

struct Core<T: VirtioTransport, B: Dma, const N: usize, const BYTES: usize> {
    transport: T,
    queues: Queues<B, N>,
    state: MultiportConsole,
    pool_bytes: core::marker::PhantomData<[u8; BYTES]>,
    handles: [Option<PortHandle>; 2],
    pending_control: Option<ControlCommand>,
    faulted: bool,
}

impl<T: VirtioTransport, B: Dma, const N: usize, const BYTES: usize> Core<T, B, N, BYTES> {
    fn configure(&mut self) -> Result<(), OwnedError> {
        self.transport.set_config_vector(CONFIG_VECTOR)?;
        self.transport.setup_queue(
            CONTROL_RX_QUEUE,
            self.queues.control_rx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        self.transport.setup_queue(
            CONTROL_TX_QUEUE,
            self.queues.control_tx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        self.transport.setup_queue(
            HOSTLINK_CONTROL_RX_QUEUE,
            self.queues.host_control_rx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        self.transport.setup_queue(
            HOSTLINK_CONTROL_TX_QUEUE,
            self.queues.host_control_tx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        self.transport.setup_queue(
            HOSTLINK_BULK_RX_QUEUE,
            self.queues.bulk_rx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        self.transport.setup_queue(
            HOSTLINK_BULK_TX_QUEUE,
            self.queues.bulk_tx.queue.memory(),
            QUEUE_VECTOR,
        )?;
        Ok(())
    }

    fn post_receive_buffers(&mut self) -> Result<(), OwnedError> {
        self.queues.control_rx.post_all(BYTES)?;
        self.queues.host_control_rx.post_all(BYTES)?;
        self.queues.bulk_rx.post_all(BYTES)?;
        Ok(())
    }

    fn stage_device_ready(&mut self) -> Result<(), OwnedError> {
        let command = self.state.begin()?;
        let _ = self.queues.control_tx.stage(&command.encode())?;
        Ok(())
    }

    fn notify_initial(&self) -> Result<(), OwnedError> {
        self.transport.notify_queue(CONTROL_RX_QUEUE)?;
        self.transport.notify_queue(HOSTLINK_CONTROL_RX_QUEUE)?;
        self.transport.notify_queue(HOSTLINK_BULK_RX_QUEUE)?;
        self.transport.notify_queue(CONTROL_TX_QUEUE)?;
        Ok(())
    }

    fn reset_software(&mut self) -> Result<(), OwnedError> {
        self.queues.reset()?;
        self.state.reset();
        self.handles = [None, None];
        self.pending_control = None;
        self.faulted = false;
        Ok(())
    }

    fn handle_action(&mut self, action: ControlAction) {
        match action {
            ControlAction::SendPortReady(command) => self.pending_control = Some(command),
            ControlAction::Bound { handle, open } => {
                self.handles[plane_index(handle.plane)] = Some(handle);
                self.pending_control = Some(open);
            }
            ControlAction::Connected(handle) => {
                self.handles[plane_index(handle.plane)] = Some(handle);
            }
            ControlAction::Disconnected(handle) => {
                self.handles[plane_index(handle.plane)] = Some(handle);
            }
            ControlAction::RemovedUnbound(_) => {}
        }
    }

    fn flush_control_tx(&mut self) -> Result<bool, OwnedError> {
        if !self.queues.control_tx.poll()? {
            return Ok(false);
        }
        let Some(command) = self.pending_control else {
            return Ok(true);
        };
        let notify = self.queues.control_tx.stage(&command.encode())?;
        self.pending_control = None;
        if notify {
            self.transport.notify_queue(CONTROL_TX_QUEUE)?;
        }
        Ok(true)
    }

    fn connected(&self, plane: Plane) -> Result<(), OwnedError> {
        let handle = self.handles[plane_index(plane)].ok_or(OwnedError::Disconnected(plane))?;
        if self.state.validate(handle)? {
            Ok(())
        } else {
            Err(OwnedError::Disconnected(plane))
        }
    }
}

const fn plane_index(plane: Plane) -> usize {
    match plane {
        Plane::Control => 0,
        Plane::Bulk => 1,
    }
}

/// Queue graph prepared while the device DMA window is still inactive.
pub struct PreparedConsole<
    T: VirtioTransport,
    P: DmaPool,
    const N: usize = QUEUE_SIZE,
    const BYTES: usize = PAYLOAD_BYTES,
> {
    core: Core<T, P::Buf, N, BYTES>,
    pool: P,
}

impl<T: VirtioTransport, P: DmaPool, const N: usize, const BYTES: usize>
    PreparedConsole<T, P, N, BYTES>
{
    /// Reset/negotiate the device, allocate q2..q7, configure config vector 0
    /// and queue vector 1, and publish all three receive buffers. This method
    /// deliberately does not set `DRIVER_OK` or publish `DEVICE_READY`.
    pub fn prepare(mut transport: T, pool: P) -> Result<Self, (OwnedError, T, P)> {
        if let Err(error) = transport.reset().map_err(OwnedError::from) {
            return Err((error, transport, pool));
        }
        match transport.negotiate(SUPPORTED_FEATURES, REQUIRED_FEATURES) {
            Ok(_) => {}
            Err(error) => return Err((error.into(), transport, pool)),
        }
        let max_ports = match transport.console_max_ports() {
            Ok(max_ports) => max_ports,
            Err(error) => return Err((error.into(), transport, pool)),
        };
        let state = match MultiportConsole::new(max_ports) {
            Ok(state) => state,
            Err(error) => return Err((error.into(), transport, pool)),
        };
        let queues = match Queues::allocate(&pool, BYTES, false) {
            Ok(queues) => queues,
            Err(error) => return Err((error, transport, pool)),
        };
        let mut core = Core {
            transport,
            queues,
            state,
            pool_bytes: core::marker::PhantomData,
            handles: [None, None],
            pending_control: None,
            faulted: false,
        };
        if let Err(error) = core.configure().and_then(|()| core.post_receive_buffers()) {
            return Err((error, core.transport, pool));
        }
        Ok(Self { core, pool })
    }

    /// Complete activation after the generic host has published the DMA
    /// graph. Receive descriptors and `DEVICE_READY` are staged first;
    /// `DRIVER_OK` is the final enable edge.
    pub fn start(mut self) -> Result<FixedConsole<T, P, N, BYTES>, (OwnedError, T, P)> {
        let result = (|| {
            self.core.stage_device_ready()?;
            self.core.transport.finish_init()?;
            self.core.notify_initial()?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = self.core.transport.fail();
            let _ = self.core.transport.reset();
            return Err((error, self.core.transport, self.pool));
        }
        Ok(FixedConsole {
            core: self.core,
            pool: self.pool,
            quiesced: false,
        })
    }
}

/// Live safe console. Every method is nonblocking and every DMA payload stays
/// owned until its exact queue token completes or a device reset terminates it.
pub struct FixedConsole<
    T: VirtioTransport,
    P: DmaPool,
    const N: usize = QUEUE_SIZE,
    const BYTES: usize = PAYLOAD_BYTES,
> {
    core: Core<T, P::Buf, N, BYTES>,
    pool: P,
    quiesced: bool,
}

impl<T: VirtioTransport, P: DmaPool, const N: usize, const BYTES: usize>
    FixedConsole<T, P, N, BYTES>
{
    /// Whether the exact named port is present in this reset epoch. A bound
    /// port may still be closed; data-plane methods continue to report that
    /// state as [`OwnedError::Disconnected`].
    pub fn plane_bound(&self, plane: Plane) -> Result<bool, OwnedError> {
        self.usable()?;
        let Some(handle) = self.core.handles[plane_index(plane)] else {
            return Ok(false);
        };
        let _connected = self.core.state.validate(handle)?;
        Ok(true)
    }

    pub fn try_send(&mut self, plane: Plane, bytes: &[u8]) -> Result<usize, OwnedError> {
        self.usable()?;
        self.core.connected(plane)?;
        let count = bytes.len().min(BYTES);
        if count == 0 {
            return Ok(0);
        }
        let (lane, queue) = match plane {
            Plane::Control => (
                &mut self.core.queues.host_control_tx,
                HOSTLINK_CONTROL_TX_QUEUE,
            ),
            Plane::Bulk => (&mut self.core.queues.bulk_tx, HOSTLINK_BULK_TX_QUEUE),
        };
        let notify = lane.stage(&bytes[..count])?;
        if notify {
            self.core.transport.notify_queue(queue)?;
        }
        Ok(count)
    }

    pub fn try_recv(&mut self, plane: Plane, out: &mut [u8]) -> Result<usize, OwnedError> {
        self.usable()?;
        self.core.connected(plane)?;
        if out.is_empty() {
            return Ok(0);
        }
        let (lane, queue) = match plane {
            Plane::Control => (
                &mut self.core.queues.host_control_rx,
                HOSTLINK_CONTROL_RX_QUEUE,
            ),
            Plane::Bulk => (&mut self.core.queues.bulk_rx, HOSTLINK_BULK_RX_QUEUE),
        };
        let (count, notify) = lane.read(out, BYTES)?;
        if notify {
            self.core.transport.notify_queue(queue)?;
        }
        Ok(count)
    }

    /// Drain at most `budget` device-control records. A pending PORT_READY
    /// always gets first use of control transmitq, so name/open cannot race
    /// ahead of the mandated reply.
    pub fn poll_control_events(&mut self, budget: u16) -> Result<u16, OwnedError> {
        self.usable()?;
        let mut done = 0;
        while done < budget {
            if !self.core.flush_control_tx()? && self.core.pending_control.is_some() {
                break;
            }
            if !self.core.queues.control_rx.poll()? {
                break;
            }
            let written = self
                .core
                .queues
                .control_rx
                .front_len()?
                .ok_or(OwnedError::WouldBlock)?;
            if written == 0 || written > CONTROL_FRAME_MAX {
                self.core.faulted = true;
                let _ = self.core.transport.fail();
                return Err(OwnedError::ResetRequired);
            }
            let mut frame = [0u8; CONTROL_FRAME_MAX];
            let (read, notify) = self
                .core
                .queues
                .control_rx
                .read(&mut frame[..written], BYTES)?;
            if read != written {
                return Err(OwnedError::ResetRequired);
            }
            let action = match self.core.state.handle(&frame[..written]) {
                Ok(action) => action,
                Err(error) => {
                    self.core.faulted = true;
                    let _ = self.core.transport.fail();
                    return Err(OwnedError::Console(error));
                }
            };
            self.core.handle_action(action);
            if notify {
                self.core.transport.notify_queue(CONTROL_RX_QUEUE)?;
            }
            done += 1;
            let _ = self.core.flush_control_tx()?;
        }
        Ok(done)
    }

    /// Bounded priority primitive: device control, HostLink control bytes,
    /// then bulk bytes. `WouldBlock`/disconnected planes contribute zero;
    /// malformed input and hardware failures remain visible.
    pub fn drain(
        &mut self,
        event_budget: u16,
        control_out: &mut [u8],
        bulk_out: &mut [u8],
    ) -> Result<DrainReport, OwnedError> {
        let mut report = DrainReport {
            control_events: self.poll_control_events(event_budget)?,
            ..DrainReport::default()
        };
        match self.try_recv(Plane::Control, control_out) {
            Ok(bytes) => report.control_bytes = bytes,
            Err(OwnedError::WouldBlock | OwnedError::Disconnected(Plane::Control)) => {}
            Err(error) => return Err(error),
        }
        match self.try_recv(Plane::Bulk, bulk_out) {
            Ok(bytes) => report.bulk_bytes = bytes,
            Err(OwnedError::WouldBlock | OwnedError::Disconnected(Plane::Bulk)) => {}
            Err(error) => return Err(error),
        }
        Ok(report)
    }

    /// Device reset is the only operation that terminates every outstanding
    /// DMA borrow. Rebuild all six queues and discovery state in a new epoch.
    pub fn recover(&mut self) -> Result<(), OwnedError> {
        self.core.transport.reset()?;
        self.core.reset_software()?;
        self.core
            .transport
            .negotiate(SUPPORTED_FEATURES, REQUIRED_FEATURES)?;
        let max_ports = self.core.transport.console_max_ports()?;
        self.core.state = MultiportConsole::new(max_ports)?;
        self.core.configure()?;
        self.core.post_receive_buffers()?;
        self.core.stage_device_ready()?;
        self.core.transport.finish_init()?;
        self.core.notify_initial()?;
        self.quiesced = false;
        Ok(())
    }

    pub fn quiesce(&mut self) -> Result<(), OwnedError> {
        self.core.transport.reset()?;
        self.core.reset_software()?;
        self.quiesced = true;
        Ok(())
    }

    pub fn into_parts(self) -> (T, P) {
        (self.core.transport, self.pool)
    }

    pub fn transport(&self) -> &T {
        &self.core.transport
    }

    fn usable(&self) -> Result<(), OwnedError> {
        if self.quiesced || self.core.faulted {
            Err(OwnedError::ResetRequired)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, vec, vec::Vec};
    use core::cell::{Cell, RefCell};

    use kore_abi::device::IovaAddr;

    use super::*;
    use crate::split::QueueMemory;

    struct HostBuf {
        words: Box<[u64]>,
        iova: IovaAddr,
    }

    impl Dma for HostBuf {
        fn cpu(&self) -> *mut u8 {
            self.words.as_ptr() as *mut u8
        }

        fn iova(&self) -> IovaAddr {
            self.iova
        }

        fn len(&self) -> usize {
            self.words.len() * 8
        }
    }

    struct HostPool {
        next: Cell<u64>,
        allocated: Cell<usize>,
    }

    impl HostPool {
        fn new() -> Self {
            Self {
                next: Cell::new(0x4_0000_0000),
                allocated: Cell::new(0),
            }
        }
    }

    impl DmaPool for HostPool {
        type Buf = HostBuf;

        fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
            let bytes = bytes.max(4096).next_multiple_of(4096);
            let iova = self.next.get();
            self.next.set(iova.checked_add(bytes as u64)?);
            self.allocated.set(self.allocated.get() + bytes);
            Some(HostBuf {
                words: vec![0u64; bytes / 8].into_boxed_slice(),
                iova: IovaAddr(iova),
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Event {
        Reset,
        Negotiate,
        ConfigVector(u16),
        Setup(u16, u16),
        Finish,
        Notify(u16),
        Fail,
    }

    struct MockTransport {
        events: RefCell<Vec<Event>>,
        epoch: u32,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                events: RefCell::new(Vec::new()),
                epoch: 0,
            }
        }
    }

    impl VirtioTransport for MockTransport {
        fn reset(&mut self) -> Result<u32, TransportError> {
            self.events.borrow_mut().push(Event::Reset);
            self.epoch = self.epoch.wrapping_add(1).max(1);
            Ok(self.epoch)
        }

        fn negotiate(&mut self, supported: u64, required: u64) -> Result<u64, TransportError> {
            self.events.borrow_mut().push(Event::Negotiate);
            assert_eq!(required, REQUIRED_FEATURES);
            Ok(supported)
        }

        fn set_config_vector(&mut self, vector: u16) -> Result<(), TransportError> {
            self.events.borrow_mut().push(Event::ConfigVector(vector));
            Ok(())
        }

        fn setup_queue(
            &mut self,
            queue: u16,
            _memory: QueueMemory,
            vector: u16,
        ) -> Result<(), TransportError> {
            self.events.borrow_mut().push(Event::Setup(queue, vector));
            Ok(())
        }

        fn finish_init(&mut self) -> Result<(), TransportError> {
            self.events.borrow_mut().push(Event::Finish);
            Ok(())
        }

        fn notify_queue(&self, queue: u16) -> Result<(), TransportError> {
            self.events.borrow_mut().push(Event::Notify(queue));
            Ok(())
        }

        fn ack_interrupt(&self) -> Result<u8, TransportError> {
            Ok(3)
        }

        fn console_max_ports(&self) -> Result<u32, TransportError> {
            Ok(3)
        }

        fn fail(&mut self) -> Result<(), TransportError> {
            self.events.borrow_mut().push(Event::Fail);
            Ok(())
        }

        fn epoch(&self) -> u32 {
            self.epoch
        }
    }

    type TestPrepared = PreparedConsole<MockTransport, HostPool, QUEUE_SIZE, PAYLOAD_BYTES>;
    type TestConsole = FixedConsole<MockTransport, HostPool, QUEUE_SIZE, PAYLOAD_BYTES>;

    fn live() -> TestConsole {
        TestPrepared::prepare(MockTransport::new(), HostPool::new())
            .ok()
            .unwrap()
            .start()
            .ok()
            .unwrap()
    }

    fn complete_control_tx(console: &mut TestConsole) {
        console.core.queues.control_tx.complete_one_for_test();
    }

    fn control_frame(id: u32, event: u16, value: u16, payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![0; CONTROL_HEADER_LEN + payload.len()];
        frame[0..4].copy_from_slice(&id.to_le_bytes());
        frame[4..6].copy_from_slice(&event.to_le_bytes());
        frame[6..8].copy_from_slice(&value.to_le_bytes());
        frame[8..].copy_from_slice(payload);
        frame
    }

    fn inject_control(console: &mut TestConsole, frame: &[u8]) {
        console.core.queues.control_rx.device_write_for_test(frame);
    }

    fn establish(console: &mut TestConsole, id: u32, plane: Plane) {
        inject_control(console, &control_frame(id, 1, 1, &[]));
        assert_eq!(console.poll_control_events(1), Ok(1));
        complete_control_tx(console);
        let name = match plane {
            Plane::Control => crate::console::CONTROL_PORT_NAME,
            Plane::Bulk => crate::console::BULK_PORT_NAME,
        };
        inject_control(console, &control_frame(id, 7, 0, name));
        assert_eq!(console.poll_control_events(1), Ok(1));
        inject_control(console, &control_frame(id, 6, 1, &[]));
        assert_eq!(console.poll_control_events(1), Ok(1));
    }

    #[test]
    fn prepare_populates_rx_and_start_is_the_driver_ok_edge() {
        let prepared = TestPrepared::prepare(MockTransport::new(), HostPool::new())
            .ok()
            .unwrap();
        let events = prepared.core.transport.events.borrow();
        assert!(!events.contains(&Event::Finish));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Setup(_, 1)))
                .count(),
            6
        );
        assert!(events.contains(&Event::ConfigVector(0)));
        assert!(
            prepared
                .core
                .queues
                .control_rx
                .tokens
                .iter()
                .all(Option::is_some)
        );
        assert!(
            prepared
                .core
                .queues
                .host_control_rx
                .tokens
                .iter()
                .all(Option::is_some)
        );
        assert!(
            prepared
                .core
                .queues
                .bulk_rx
                .tokens
                .iter()
                .all(Option::is_some)
        );
        assert!(
            prepared
                .core
                .queues
                .control_tx
                .tokens
                .iter()
                .all(Option::is_none)
        );
        drop(events);

        let live = prepared.start().ok().unwrap();
        let events = live.core.transport.events.borrow();
        let finish = events
            .iter()
            .position(|event| *event == Event::Finish)
            .unwrap();
        let ready_kick = events
            .iter()
            .position(|event| *event == Event::Notify(CONTROL_TX_QUEUE))
            .unwrap();
        assert!(finish < ready_kick);
        assert_eq!(
            live.core
                .queues
                .control_tx
                .tokens
                .iter()
                .filter(|token| token.is_some())
                .count(),
            1
        );
    }

    #[test]
    fn safe_tx_is_bounded_and_owns_payload_until_completion() {
        let mut console = live();
        complete_control_tx(&mut console);
        establish(&mut console, 1, Plane::Control);
        for _ in 0..QUEUE_SIZE {
            assert_eq!(console.try_send(Plane::Control, b"hello"), Ok(5));
        }
        assert_eq!(
            console.try_send(Plane::Control, b"full"),
            Err(OwnedError::Backpressure)
        );
        console.core.queues.host_control_tx.complete_one_for_test();
        assert_eq!(console.try_send(Plane::Control, b"again"), Ok(5));
    }

    #[test]
    fn disconnect_is_applied_before_data_plane_access() {
        let mut console = live();
        assert_eq!(console.plane_bound(Plane::Control), Ok(false));
        complete_control_tx(&mut console);
        establish(&mut console, 1, Plane::Control);
        inject_control(&mut console, &control_frame(1, 6, 0, &[]));
        assert_eq!(console.poll_control_events(1), Ok(1));
        assert_eq!(console.plane_bound(Plane::Control), Ok(true));
        assert_eq!(
            console.try_send(Plane::Control, b"closed"),
            Err(OwnedError::Disconnected(Plane::Control))
        );
    }

    #[test]
    fn control_events_are_drained_before_ready_bulk_bytes() {
        let mut console = live();
        complete_control_tx(&mut console);
        establish(&mut console, 1, Plane::Control);
        establish(&mut console, 2, Plane::Bulk);

        console.core.queues.bulk_rx.device_write_for_test(b"bulk");
        inject_control(&mut console, &control_frame(1, 6, 0, &[]));

        let mut control = [0u8; 8];
        let mut bulk = [0u8; 8];
        let report = console.drain(1, &mut control, &mut bulk).unwrap();
        assert_eq!(report.control_events, 1);
        assert_eq!(report.control_bytes, 0);
        assert_eq!(report.bulk_bytes, 4);
        assert_eq!(&bulk[..4], b"bulk");
    }

    #[test]
    fn recover_terminates_old_tokens_and_starts_a_new_epoch() {
        let mut console = live();
        let old_transport_epoch = console.core.transport.epoch();
        let old_queue_epoch = console.core.queues.bulk_rx.queue.epoch();
        console.recover().unwrap();
        assert_ne!(console.core.transport.epoch(), old_transport_epoch);
        assert_ne!(console.core.queues.bulk_rx.queue.epoch(), old_queue_epoch);
        assert!(
            console
                .core
                .queues
                .bulk_rx
                .tokens
                .iter()
                .all(Option::is_some)
        );
        assert!(
            console
                .core
                .queues
                .control_tx
                .tokens
                .iter()
                .any(Option::is_some)
        );
    }

    #[test]
    fn receive_bank_replenishes_and_ready_fifo_wraps() {
        let pool = HostPool::new();
        let mut lane = RxLane::<HostBuf, 4>::allocate(&pool, 4096, false).unwrap();
        lane.post_all(4096).unwrap();
        for value in 0u8..20 {
            lane.device_write_for_test(&[value]);
            let mut byte = [0xff];
            assert_eq!(lane.read(&mut byte, 4096).unwrap().0, 1);
            assert_eq!(byte, [value]);
            assert_eq!(lane.ready_len, 0);
            assert!(lane.tokens.iter().all(Option::is_some));
        }
    }

    #[test]
    fn production_bank_geometry_has_an_exact_cap_pool_footprint() {
        assert_eq!(QUEUE_SIZE, 16);
        assert_eq!(DMA_POOL_BYTES, 466_944);

        let prepared = TestPrepared::prepare(MockTransport::new(), HostPool::new())
            .ok()
            .unwrap();
        assert_eq!(prepared.pool.allocated.get(), DMA_POOL_BYTES);
    }
}
