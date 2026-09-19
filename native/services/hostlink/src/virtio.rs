//! VirtIO-serial composition for the transport-independent HostLink service.
//!
//! This is intentionally the only layer that sees both `kore-hostlink` and
//! `kore-virtio`. The driver exposes partial nonblocking byte streams; the
//! protocol requires an all-or-nothing frame acceptance surface. This adapter
//! closes that mismatch with service-owned, caller-supplied bounded staging.

use kore_driver::host::{BindingTeardown, BoundDevice};
use kore_hostlink::{BULK_FRAME_MAX, CONTROL_FRAME_MAX, HostLinkTransport, PortError};
use kore_virtio::binding::{VirtioConsoleBindError, VirtioConsoleBinding};
use kore_virtio::console::Plane as VirtioPlane;
use kore_virtio::owned::OwnedError;
use kore_virtio::pump::{BytePortError, IrqNotice, IrqSource, NonblockingBytePorts};

pub use kore_virtio::claim::{ClaimError, HOSTLINK_VIRTIO_CONSOLE, bind_console, claim_console};
pub use kore_virtio::pump::HeldRoute;

pub const CONTROL_TX_STAGING_BYTES: usize = CONTROL_FRAME_MAX;
pub const BULK_TX_STAGING_BYTES: usize = BULK_FRAME_MAX;

/// Construction refusal that returns all affine/caller-owned inputs.
pub struct StorageError<P, C, B> {
    pub control_required: usize,
    pub control_available: usize,
    pub bulk_required: usize,
    pub bulk_available: usize,
    ports: P,
    control: C,
    bulk: B,
}

impl<P, C, B> StorageError<P, C, B> {
    pub fn into_parts(self) -> (P, C, B) {
        (self.ports, self.control, self.bulk)
    }
}

struct TxStage<S> {
    storage: S,
    start: usize,
    end: usize,
}

impl<S: AsRef<[u8]> + AsMut<[u8]>> TxStage<S> {
    fn new(storage: S) -> Self {
        Self {
            storage,
            start: 0,
            end: 0,
        }
    }

    fn pending(&self) -> &[u8] {
        &self.storage.as_ref()[self.start..self.end]
    }

    fn is_pending(&self) -> bool {
        self.start != self.end
    }

    fn stage(&mut self, frame: &[u8]) -> Result<(), PortError> {
        if self.is_pending() {
            return Err(PortError::Backpressure);
        }
        if frame.len() > self.storage.as_ref().len() {
            return Err(PortError::Fault);
        }
        self.storage.as_mut()[..frame.len()].copy_from_slice(frame);
        self.start = 0;
        self.end = frame.len();
        Ok(())
    }

    fn consume(&mut self, bytes: usize) -> Result<(), PortError> {
        if bytes > self.pending().len() {
            return Err(PortError::Fault);
        }
        self.start += bytes;
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        }
        Ok(())
    }
}

impl<S> TxStage<S> {
    fn into_storage(self) -> S {
        self.storage
    }
}

/// HostLink's all-or-nothing transport over partial VirtIO serial writes.
///
/// `C` and `B` are normally service-region slices. Requiring the full frozen
/// protocol maxima at construction prevents a lawful frame from becoming a
/// permanent `Fault` merely because a smaller implementation buffer was
/// chosen. No global allocator or driver-owned protocol storage is involved.
pub struct VirtioHostLink<P, C, B> {
    ports: P,
    control: TxStage<C>,
    bulk: TxStage<B>,
}

impl<P, C, B> VirtioHostLink<P, C, B>
where
    P: NonblockingBytePorts,
    C: AsRef<[u8]> + AsMut<[u8]>,
    B: AsRef<[u8]> + AsMut<[u8]>,
{
    pub fn new(ports: P, control: C, bulk: B) -> Result<Self, StorageError<P, C, B>> {
        let control_available = control.as_ref().len();
        let bulk_available = bulk.as_ref().len();
        if control_available < CONTROL_FRAME_MAX || bulk_available < BULK_FRAME_MAX {
            return Err(StorageError {
                control_required: CONTROL_FRAME_MAX,
                control_available,
                bulk_required: BULK_FRAME_MAX,
                bulk_available,
                ports,
                control,
                bulk,
            });
        }
        Ok(Self {
            ports,
            control: TxStage::new(control),
            bulk: TxStage::new(bulk),
        })
    }

    pub fn control_pending(&self) -> usize {
        self.control.pending().len()
    }

    pub fn bulk_pending(&self) -> usize {
        self.bulk.pending().len()
    }

    pub fn ports(&self) -> &P {
        &self.ports
    }

    pub fn ports_mut(&mut self) -> &mut P {
        &mut self.ports
    }

    /// Service device-control records and make bounded progress on already
    /// accepted TX frames. Control always precedes bulk.
    pub fn pump<I: IrqSource>(
        &mut self,
        irq: &mut I,
        budget: SerialPumpBudget,
    ) -> Result<SerialPumpReport, PortError> {
        let irq_fired = match irq.observe().map_err(map_error)? {
            IrqNotice::Quiet => false,
            IrqNotice::Fired => true,
            IrqNotice::Revoked => return Err(PortError::Disconnected),
        };
        let control_events = self
            .ports
            .service_device_control(budget.device_control_events)
            .map_err(map_error)?;
        let control_bytes = flush_bounded(
            &mut self.ports,
            VirtioPlane::Control,
            &mut self.control,
            budget.control_write_calls,
        )?;
        let bulk_bytes = flush_bounded(
            &mut self.ports,
            VirtioPlane::Bulk,
            &mut self.bulk,
            budget.bulk_write_calls,
        )?;
        Ok(SerialPumpReport {
            irq_fired,
            control_events,
            control_bytes,
            bulk_bytes,
            control_pending: self.control_pending(),
            bulk_pending: self.bulk_pending(),
        })
    }

    /// Consume the live adapter only after attempting the driver's reset edge.
    pub fn quiesce(mut self) -> Result<QuiescedVirtioHostLink<P, C, B>, QuiesceError<P, C, B>> {
        match self.ports.quiesce().map_err(map_error) {
            Ok(()) => Ok(QuiescedVirtioHostLink(self)),
            Err(error) => Err(QuiesceError {
                error,
                adapter: self,
            }),
        }
    }

    fn flush_control_once(&mut self) -> Result<usize, PortError> {
        flush_once(&mut self.ports, VirtioPlane::Control, &mut self.control)
    }

    fn flush_bulk_once(&mut self) -> Result<usize, PortError> {
        flush_once(&mut self.ports, VirtioPlane::Bulk, &mut self.bulk)
    }
}

impl<C, B> VirtioHostLink<BoundConsolePorts, C, B> {
    /// Run the native class-stop, fabric-proof and containment lifecycle.
    pub fn teardown(self) -> BindingTeardown<VirtioConsoleBindError> {
        self.ports.teardown()
    }
}

/// Affine owner adapting the generic device lifecycle to the byte-port seam.
/// The route remains inside the same owner and is only observed between
/// transport turns, so no reference to it can outlive device containment.
pub struct BoundConsolePorts {
    owner: BoundDevice<VirtioConsoleBinding>,
}

impl BoundConsolePorts {
    pub const fn new(owner: BoundDevice<VirtioConsoleBinding>) -> Self {
        Self { owner }
    }

    pub fn has_route(&self) -> bool {
        self.owner.parts().route().is_some()
    }

    pub fn planes_bound(&mut self) -> bool {
        let console = self.owner.parts_mut().console();
        console.plane_bound(VirtioPlane::Control) == Ok(true)
            && console.plane_bound(VirtioPlane::Bulk) == Ok(true)
    }

    pub fn observe_route(&self, deadline_ns: u64) -> Result<IrqNotice, BytePortError> {
        use kore_rt::cap::verbs::InterruptWait;
        let Some(route) = self.owner.parts().route() else {
            return Err(BytePortError::Fault);
        };
        match kore_rt::cap::verbs::wait_interrupt_until(route, deadline_ns) {
            Ok(InterruptWait::Fired) => Ok(IrqNotice::Fired),
            Ok(InterruptWait::Deadline) => Ok(IrqNotice::Quiet),
            Ok(InterruptWait::Revoked) => Ok(IrqNotice::Revoked),
            Err(_) => Err(BytePortError::Fault),
        }
    }

    pub fn teardown(self) -> BindingTeardown<VirtioConsoleBindError> {
        self.owner.teardown()
    }
}

impl NonblockingBytePorts for BoundConsolePorts {
    fn service_device_control(&mut self, budget: u16) -> Result<u16, BytePortError> {
        self.owner
            .parts_mut()
            .console()
            .poll_control_events(budget)
            .map_err(map_owned)
    }

    fn try_write(&mut self, plane: VirtioPlane, bytes: &[u8]) -> Result<usize, BytePortError> {
        self.owner
            .parts_mut()
            .console()
            .try_send(plane, bytes)
            .map_err(map_owned)
    }

    fn try_read(&mut self, plane: VirtioPlane, output: &mut [u8]) -> Result<usize, BytePortError> {
        match self.owner.parts_mut().console().try_recv(plane, output) {
            Ok(bytes) => Ok(bytes),
            Err(OwnedError::WouldBlock) => Ok(0),
            Err(error) => Err(map_owned(error)),
        }
    }

    fn quiesce(&mut self) -> Result<(), BytePortError> {
        self.owner
            .parts_mut()
            .console()
            .quiesce()
            .map_err(map_owned)
    }
}

fn map_owned(error: OwnedError) -> BytePortError {
    match error {
        OwnedError::WouldBlock | OwnedError::Backpressure => BytePortError::Backpressure,
        OwnedError::Disconnected(_) => BytePortError::Disconnected,
        OwnedError::InvalidBufferSize
        | OwnedError::Queue(_)
        | OwnedError::Transport(_)
        | OwnedError::Console(_)
        | OwnedError::ResetRequired => BytePortError::Fault,
    }
}

impl<P, C, B> HostLinkTransport for VirtioHostLink<P, C, B>
where
    P: NonblockingBytePorts,
    C: AsRef<[u8]> + AsMut<[u8]>,
    B: AsRef<[u8]> + AsMut<[u8]>,
{
    fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError> {
        let _ = self.flush_control_once()?;
        if self.control.is_pending() {
            return Err(PortError::Backpressure);
        }
        self.control.stage(frame)?;
        let _ = self.flush_control_once()?;
        Ok(())
    }

    fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        let _ = self.flush_control_once()?;
        self.ports
            .try_read(VirtioPlane::Control, output)
            .map_err(map_error)
    }

    fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError> {
        // Never let a bulk call jump an already accepted control frame.
        let _ = self.flush_control_once()?;
        if self.control.is_pending() {
            return Err(PortError::Backpressure);
        }
        let _ = self.flush_bulk_once()?;
        if self.bulk.is_pending() {
            return Err(PortError::Backpressure);
        }
        self.bulk.stage(frame)?;
        let _ = self.flush_bulk_once()?;
        Ok(())
    }

    fn try_recv_bulk(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        let _ = self.flush_control_once()?;
        let _ = self.flush_bulk_once()?;
        self.ports
            .try_read(VirtioPlane::Bulk, output)
            .map_err(map_error)
    }
}

fn flush_once<P, S>(
    ports: &mut P,
    plane: VirtioPlane,
    stage: &mut TxStage<S>,
) -> Result<usize, PortError>
where
    P: NonblockingBytePorts,
    S: AsRef<[u8]> + AsMut<[u8]>,
{
    if !stage.is_pending() {
        return Ok(0);
    }
    let written = match ports.try_write(plane, stage.pending()) {
        Ok(written) => written,
        Err(BytePortError::Backpressure) => return Ok(0),
        Err(error) => return Err(map_error(error)),
    };
    stage.consume(written)?;
    Ok(written)
}

fn flush_bounded<P, S>(
    ports: &mut P,
    plane: VirtioPlane,
    stage: &mut TxStage<S>,
    calls: u16,
) -> Result<usize, PortError>
where
    P: NonblockingBytePorts,
    S: AsRef<[u8]> + AsMut<[u8]>,
{
    let mut total = 0usize;
    for _ in 0..calls {
        let written = flush_once(ports, plane, stage)?;
        total = total.checked_add(written).ok_or(PortError::Fault)?;
        if written == 0 || !stage.is_pending() {
            break;
        }
    }
    Ok(total)
}

fn map_error(error: BytePortError) -> PortError {
    match error {
        BytePortError::Backpressure => PortError::Backpressure,
        BytePortError::Disconnected => PortError::Disconnected,
        BytePortError::Fault => PortError::Fault,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialPumpBudget {
    pub device_control_events: u16,
    pub control_write_calls: u16,
    pub bulk_write_calls: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SerialPumpReport {
    pub irq_fired: bool,
    pub control_events: u16,
    pub control_bytes: usize,
    pub bulk_bytes: usize,
    pub control_pending: usize,
    pub bulk_pending: usize,
}

pub struct QuiescedVirtioHostLink<P, C, B>(VirtioHostLink<P, C, B>);

impl<P, C, B> QuiescedVirtioHostLink<P, C, B> {
    /// Return the bound byte-port owner and both service-owned staging regions
    /// only after quiesce succeeded. The caller may now unbind/contain it.
    pub fn into_parts(self) -> (P, C, B) {
        (
            self.0.ports,
            self.0.control.into_storage(),
            self.0.bulk.into_storage(),
        )
    }
}

pub struct QuiesceError<P, C, B> {
    pub error: PortError,
    adapter: VirtioHostLink<P, C, B>,
}

impl<P, C, B> QuiesceError<P, C, B> {
    /// Recover ownership after the mandatory stop attempt failed, permitting
    /// the binding's unconditional containment path.
    pub fn into_parts(self) -> (P, C, B) {
        (
            self.adapter.ports,
            self.adapter.control.into_storage(),
            self.adapter.bulk.into_storage(),
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Op {
        DeviceControl,
        Write(VirtioPlane),
        Read(VirtioPlane),
        Quiesce,
        Unbind,
    }

    struct MockPorts {
        writes: Vec<u8>,
        control_rx: Vec<u8>,
        bulk_rx: Vec<u8>,
        control_at: usize,
        bulk_at: usize,
        write_chunk: usize,
        read_chunk: usize,
        control_blocked: bool,
        bulk_blocked: bool,
        disconnected: bool,
        log: Vec<Op>,
    }

    impl MockPorts {
        fn new() -> Self {
            Self {
                writes: Vec::new(),
                control_rx: Vec::new(),
                bulk_rx: Vec::new(),
                control_at: 0,
                bulk_at: 0,
                write_chunk: usize::MAX,
                read_chunk: usize::MAX,
                control_blocked: false,
                bulk_blocked: false,
                disconnected: false,
                log: Vec::new(),
            }
        }

        fn unbind(&mut self) {
            self.log.push(Op::Unbind);
        }
    }

    impl NonblockingBytePorts for MockPorts {
        fn service_device_control(&mut self, budget: u16) -> Result<u16, BytePortError> {
            self.log.push(Op::DeviceControl);
            Ok(budget.min(1))
        }

        fn try_write(&mut self, plane: VirtioPlane, bytes: &[u8]) -> Result<usize, BytePortError> {
            self.log.push(Op::Write(plane));
            if self.disconnected {
                return Err(BytePortError::Disconnected);
            }
            if (plane == VirtioPlane::Control && self.control_blocked)
                || (plane == VirtioPlane::Bulk && self.bulk_blocked)
            {
                return Err(BytePortError::Backpressure);
            }
            let take = bytes.len().min(self.write_chunk);
            self.writes.extend_from_slice(&bytes[..take]);
            Ok(take)
        }

        fn try_read(
            &mut self,
            plane: VirtioPlane,
            output: &mut [u8],
        ) -> Result<usize, BytePortError> {
            self.log.push(Op::Read(plane));
            if self.disconnected {
                return Err(BytePortError::Disconnected);
            }
            let (source, at) = match plane {
                VirtioPlane::Control => (&self.control_rx, &mut self.control_at),
                VirtioPlane::Bulk => (&self.bulk_rx, &mut self.bulk_at),
            };
            let take = output
                .len()
                .min(self.read_chunk)
                .min(source.len().saturating_sub(*at));
            output[..take].copy_from_slice(&source[*at..*at + take]);
            *at += take;
            Ok(take)
        }

        fn quiesce(&mut self) -> Result<(), BytePortError> {
            self.log.push(Op::Quiesce);
            Ok(())
        }
    }

    struct MockIrq(IrqNotice);

    impl IrqSource for MockIrq {
        fn observe(&mut self) -> Result<IrqNotice, BytePortError> {
            Ok(self.0)
        }
    }

    fn adapter(ports: MockPorts) -> VirtioHostLink<MockPorts, Vec<u8>, Vec<u8>> {
        VirtioHostLink::new(
            ports,
            vec![0; CONTROL_TX_STAGING_BYTES],
            vec![0; BULK_TX_STAGING_BYTES],
        )
        .ok()
        .unwrap()
    }

    const BUDGET: SerialPumpBudget = SerialPumpBudget {
        device_control_events: 8,
        control_write_calls: 8,
        bulk_write_calls: 8,
    };

    #[test]
    fn staged_send_is_all_or_nothing_across_partial_driver_writes() {
        let mut ports = MockPorts::new();
        ports.write_chunk = 2;
        let mut adapter = adapter(ports);
        assert_eq!(adapter.try_send_control(b"abcdef"), Ok(()));
        assert_eq!(adapter.control_pending(), 4);
        assert_eq!(
            adapter.try_send_control(b"next"),
            Err(PortError::Backpressure)
        );
        assert_eq!(adapter.control_pending(), 2);
        adapter
            .pump(&mut MockIrq(IrqNotice::Quiet), BUDGET)
            .unwrap();
        assert_eq!(adapter.try_send_control(b"next"), Ok(()));
        adapter
            .pump(&mut MockIrq(IrqNotice::Quiet), BUDGET)
            .unwrap();
        assert_eq!(&adapter.ports.writes, b"abcdefnext");
    }

    #[test]
    fn partial_and_coalesced_rx_bytes_are_not_reframed() {
        let mut ports = MockPorts::new();
        ports.control_rx.extend_from_slice(b"abcdef");
        ports.read_chunk = 2;
        let mut adapter = adapter(ports);
        let mut assembled = [0u8; 6];
        for turn in 0..3 {
            assert_eq!(adapter.try_recv_control(&mut assembled[turn * 2..]), Ok(2));
        }
        assert_eq!(&assembled, b"abcdef");

        adapter.ports.control_at = 0;
        adapter.ports.read_chunk = 6;
        let mut coalesced = [0u8; 8];
        assert_eq!(adapter.try_recv_control(&mut coalesced), Ok(6));
        assert_eq!(&coalesced[..6], b"abcdef");
    }

    #[test]
    fn saturated_bulk_never_precedes_control() {
        let mut ports = MockPorts::new();
        ports.control_blocked = true;
        ports.bulk_blocked = true;
        let mut adapter = adapter(ports);
        assert_eq!(adapter.try_send_bulk(b"bulk"), Ok(()));
        assert_eq!(adapter.try_send_control(b"control"), Ok(()));
        adapter.ports.log.clear();
        adapter.ports.control_blocked = false;
        let report = adapter
            .pump(&mut MockIrq(IrqNotice::Fired), BUDGET)
            .unwrap();
        assert!(report.irq_fired);
        assert_eq!(report.control_events, 1);
        assert_eq!(report.control_pending, 0);
        assert_ne!(report.bulk_pending, 0);
        assert_eq!(
            &adapter.ports.log[..3],
            &[
                Op::DeviceControl,
                Op::Write(VirtioPlane::Control),
                Op::Write(VirtioPlane::Bulk),
            ]
        );
    }

    #[test]
    fn revoke_disconnect_and_quiesce_order_are_preserved() {
        let mut adapter = adapter(MockPorts::new());
        assert_eq!(
            adapter.pump(&mut MockIrq(IrqNotice::Revoked), BUDGET),
            Err(PortError::Disconnected)
        );
        adapter.ports.disconnected = true;
        assert_eq!(
            adapter.try_recv_control(&mut [0; 1]),
            Err(PortError::Disconnected)
        );
        adapter.ports.disconnected = false;
        let quiesced = adapter.quiesce().ok().unwrap();
        let (mut ports, _control, _bulk) = quiesced.into_parts();
        ports.unbind();
        assert!(
            ports
                .log
                .windows(2)
                .any(|ops| ops == [Op::Quiesce, Op::Unbind])
        );
    }
}
