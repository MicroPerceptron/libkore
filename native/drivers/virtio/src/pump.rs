//! Generic service-side composition for a nonblocking two-plane byte port.
//!
//! This module deliberately knows nothing about the HostLink wire protocol.
//! It makes the serial driver's partial-stream semantics, IRQ state, strict
//! control priority, and quiesce ordering available to the service that owns
//! both the protocol and the bound device.

use kore_driver::DmaPool;

use crate::console::Plane;
use crate::owned::{FixedConsole, OwnedError};
use crate::pci::VirtioTransport;

/// Transport-neutral result vocabulary for the future protocol adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BytePortError {
    Backpressure,
    Disconnected,
    Fault,
}

/// One service-owned pair of nonblocking byte streams.
///
/// Writes may accept only a prefix. The caller retains the suffix and retries;
/// reads likewise expose arbitrary serial fragmentation or coalescing without
/// manufacturing frame boundaries.
pub trait NonblockingBytePorts {
    fn service_device_control(&mut self, budget: u16) -> Result<u16, BytePortError>;
    fn try_write(&mut self, plane: Plane, bytes: &[u8]) -> Result<usize, BytePortError>;
    fn try_read(&mut self, plane: Plane, output: &mut [u8]) -> Result<usize, BytePortError>;
    fn quiesce(&mut self) -> Result<(), BytePortError>;
}

impl<T: VirtioTransport, P: DmaPool, const N: usize, const BYTES: usize> NonblockingBytePorts
    for FixedConsole<T, P, N, BYTES>
{
    fn service_device_control(&mut self, budget: u16) -> Result<u16, BytePortError> {
        self.poll_control_events(budget).map_err(map_owned)
    }

    fn try_write(&mut self, plane: Plane, bytes: &[u8]) -> Result<usize, BytePortError> {
        self.try_send(plane, bytes).map_err(map_owned)
    }

    fn try_read(&mut self, plane: Plane, output: &mut [u8]) -> Result<usize, BytePortError> {
        match self.try_recv(plane, output) {
            Ok(bytes) => Ok(bytes),
            Err(OwnedError::WouldBlock) => Ok(0),
            Err(error) => Err(map_owned(error)),
        }
    }

    fn quiesce(&mut self) -> Result<(), BytePortError> {
        self.quiesce().map_err(map_owned)
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

/// Result of observing the service's held interrupt route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrqNotice {
    Quiet,
    Fired,
    Revoked,
}

/// The route observation seam. The service implements this with its held
/// `RouteToken` and the runtime's bounded wait/poll verb.
pub trait IrqSource {
    fn observe(&mut self) -> Result<IrqNotice, BytePortError>;
}

/// Production adapter over the binding's held route. The deadline is absolute
/// and service-owned, so every park remains bounded and may be advanced before
/// the next pump turn.
#[cfg(feature = "service")]
pub struct HeldRoute<'a> {
    route: &'a kore_rt::cap::RouteToken,
    deadline_ns: u64,
}

#[cfg(feature = "service")]
impl<'a> HeldRoute<'a> {
    pub const fn new(route: &'a kore_rt::cap::RouteToken, deadline_ns: u64) -> Self {
        Self { route, deadline_ns }
    }

    pub fn set_deadline(&mut self, deadline_ns: u64) {
        self.deadline_ns = deadline_ns;
    }
}

#[cfg(feature = "service")]
impl IrqSource for HeldRoute<'_> {
    fn observe(&mut self) -> Result<IrqNotice, BytePortError> {
        use kore_rt::cap::verbs::InterruptWait;
        match kore_rt::cap::verbs::wait_interrupt_until(self.route, self.deadline_ns) {
            Ok(InterruptWait::Fired) => Ok(IrqNotice::Fired),
            Ok(InterruptWait::Deadline) => Ok(IrqNotice::Quiet),
            Ok(InterruptWait::Revoked) => Ok(IrqNotice::Revoked),
            Err(_) => Err(BytePortError::Fault),
        }
    }
}

/// Per-step fixed work limits. Byte bounds are additionally capped by the
/// supplied slices, so no budget can create hidden storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PumpBudget {
    pub device_control_events: u16,
    pub control_bytes: usize,
    pub bulk_bytes: usize,
}

/// Borrowed input/output windows for one pump turn. Unconsumed TX suffixes
/// remain owned by the service and are named by the report's byte counts.
pub struct PumpBuffers<'a> {
    pub control_tx: &'a [u8],
    pub control_rx: &'a mut [u8],
    pub bulk_tx: &'a [u8],
    pub bulk_rx: &'a mut [u8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PumpReport {
    pub irq_fired: bool,
    pub device_control_events: u16,
    pub control_written: usize,
    pub control_read: usize,
    pub bulk_written: usize,
    pub bulk_read: usize,
    pub control_backpressured: bool,
    pub bulk_backpressured: bool,
}

/// Owning composition shell. `into_parts` exists only after `quiesce`, so a
/// service cannot intentionally dismantle its bound device without first
/// attempting the hardware stop edge.
pub struct ServicePump<P: NonblockingBytePorts, I: IrqSource> {
    ports: P,
    irq: I,
}

pub struct QuiescedPump<P, I> {
    ports: P,
    irq: I,
}

pub struct QuiesceFailure<P, I> {
    pub error: BytePortError,
    ports: P,
    irq: I,
}

impl<P: NonblockingBytePorts, I: IrqSource> ServicePump<P, I> {
    pub const fn new(ports: P, irq: I) -> Self {
        Self { ports, irq }
    }

    /// One bounded priority turn: route/control events, control RX/TX, then
    /// bulk RX/TX. Backpressure is reported per plane instead of aborting the
    /// turn, ensuring a saturated bulk bank never suppresses control work.
    pub fn step(
        &mut self,
        budget: PumpBudget,
        buffers: PumpBuffers<'_>,
    ) -> Result<PumpReport, BytePortError> {
        let irq_fired = match self.irq.observe()? {
            IrqNotice::Quiet => false,
            IrqNotice::Fired => true,
            IrqNotice::Revoked => return Err(BytePortError::Disconnected),
        };
        let mut report = PumpReport {
            irq_fired,
            device_control_events: self
                .ports
                .service_device_control(budget.device_control_events)?,
            ..PumpReport::default()
        };

        let control_rx = buffers.control_rx.len().min(budget.control_bytes);
        report.control_read = self
            .ports
            .try_read(Plane::Control, &mut buffers.control_rx[..control_rx])?;
        let control_tx = buffers.control_tx.len().min(budget.control_bytes);
        if control_tx != 0 {
            match self
                .ports
                .try_write(Plane::Control, &buffers.control_tx[..control_tx])
            {
                Ok(bytes) => report.control_written = bytes,
                Err(BytePortError::Backpressure) => report.control_backpressured = true,
                Err(error) => return Err(error),
            }
        }

        let bulk_rx = buffers.bulk_rx.len().min(budget.bulk_bytes);
        report.bulk_read = self
            .ports
            .try_read(Plane::Bulk, &mut buffers.bulk_rx[..bulk_rx])?;
        let bulk_tx = buffers.bulk_tx.len().min(budget.bulk_bytes);
        if bulk_tx != 0 {
            match self
                .ports
                .try_write(Plane::Bulk, &buffers.bulk_tx[..bulk_tx])
            {
                Ok(bytes) => report.bulk_written = bytes,
                Err(BytePortError::Backpressure) => report.bulk_backpressured = true,
                Err(error) => return Err(error),
            }
        }
        Ok(report)
    }

    pub fn quiesce(mut self) -> Result<QuiescedPump<P, I>, QuiesceFailure<P, I>> {
        match self.ports.quiesce() {
            Ok(()) => Ok(QuiescedPump {
                ports: self.ports,
                irq: self.irq,
            }),
            Err(error) => Err(QuiesceFailure {
                error,
                ports: self.ports,
                irq: self.irq,
            }),
        }
    }
}

impl<P, I> QuiescedPump<P, I> {
    pub fn into_parts(self) -> (P, I) {
        (self.ports, self.irq)
    }
}

impl<P, I> QuiesceFailure<P, I> {
    /// Recover ownership after the mandatory quiesce attempt failed. The
    /// service may now invoke its binding's unconditional containment path.
    pub fn into_parts(self) -> (P, I) {
        (self.ports, self.irq)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, vec::Vec};
    use core::cell::RefCell;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Op {
        DeviceControl,
        Read(Plane),
        Write(Plane),
        Quiesce,
        Unbind,
    }

    struct MockPorts {
        log: Rc<RefCell<Vec<Op>>>,
        control_rx: &'static [u8],
        control_at: usize,
        read_chunk: usize,
        bulk_backpressure: bool,
        disconnected: bool,
    }

    impl MockPorts {
        fn unbind(self) {
            self.log.borrow_mut().push(Op::Unbind);
        }
    }

    impl NonblockingBytePorts for MockPorts {
        fn service_device_control(&mut self, budget: u16) -> Result<u16, BytePortError> {
            self.log.borrow_mut().push(Op::DeviceControl);
            Ok(budget.min(1))
        }

        fn try_write(&mut self, plane: Plane, bytes: &[u8]) -> Result<usize, BytePortError> {
            self.log.borrow_mut().push(Op::Write(plane));
            if self.disconnected {
                return Err(BytePortError::Disconnected);
            }
            if plane == Plane::Bulk && self.bulk_backpressure {
                return Err(BytePortError::Backpressure);
            }
            Ok(bytes.len().min(3))
        }

        fn try_read(&mut self, plane: Plane, output: &mut [u8]) -> Result<usize, BytePortError> {
            self.log.borrow_mut().push(Op::Read(plane));
            if self.disconnected {
                return Err(BytePortError::Disconnected);
            }
            if plane == Plane::Bulk || self.control_at == self.control_rx.len() {
                return Ok(0);
            }
            let take = output
                .len()
                .min(self.read_chunk)
                .min(self.control_rx.len() - self.control_at);
            output[..take]
                .copy_from_slice(&self.control_rx[self.control_at..self.control_at + take]);
            self.control_at += take;
            Ok(take)
        }

        fn quiesce(&mut self) -> Result<(), BytePortError> {
            self.log.borrow_mut().push(Op::Quiesce);
            Ok(())
        }
    }

    struct MockIrq(IrqNotice);

    impl IrqSource for MockIrq {
        fn observe(&mut self) -> Result<IrqNotice, BytePortError> {
            Ok(self.0)
        }
    }

    fn ports(log: Rc<RefCell<Vec<Op>>>) -> MockPorts {
        MockPorts {
            log,
            control_rx: b"abcdef",
            control_at: 0,
            read_chunk: 2,
            bulk_backpressure: false,
            disconnected: false,
        }
    }

    fn buffers<'a>(control_rx: &'a mut [u8], bulk_rx: &'a mut [u8]) -> PumpBuffers<'a> {
        PumpBuffers {
            control_tx: b"control",
            control_rx,
            bulk_tx: b"bulk",
            bulk_rx,
        }
    }

    const BUDGET: PumpBudget = PumpBudget {
        device_control_events: 4,
        control_bytes: 16,
        bulk_bytes: 16,
    };

    #[test]
    fn partial_and_coalesced_stream_bytes_survive_adapter_turns() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut pump = ServicePump::new(ports(log), MockIrq(IrqNotice::Quiet));
        let mut assembled = [0u8; 6];
        let mut bulk = [0u8; 1];
        for turn in 0..3 {
            let start = turn * 2;
            let mut chunk = [0u8; 4];
            let report = pump.step(BUDGET, buffers(&mut chunk, &mut bulk)).unwrap();
            assert_eq!(report.control_read, 2);
            assembled[start..start + 2].copy_from_slice(&chunk[..2]);
        }
        assert_eq!(&assembled, b"abcdef");

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut coalesced = ports(log);
        coalesced.read_chunk = 6;
        let mut pump = ServicePump::new(coalesced, MockIrq(IrqNotice::Quiet));
        let mut control = [0u8; 8];
        let report = pump.step(BUDGET, buffers(&mut control, &mut bulk)).unwrap();
        assert_eq!(report.control_read, 6);
        assert_eq!(&control[..6], b"abcdef");
    }

    #[test]
    fn bulk_saturation_does_not_starve_control() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut saturated = ports(log.clone());
        saturated.bulk_backpressure = true;
        let mut pump = ServicePump::new(saturated, MockIrq(IrqNotice::Fired));
        let mut control_rx = [0u8; 8];
        let mut bulk_rx = [0u8; 8];
        let report = pump
            .step(BUDGET, buffers(&mut control_rx, &mut bulk_rx))
            .unwrap();
        assert!(report.irq_fired);
        assert_eq!(report.device_control_events, 1);
        assert_eq!(report.control_written, 3);
        assert!(report.bulk_backpressured);
        let log = log.borrow();
        assert!(
            log.iter().position(|op| *op == Op::Write(Plane::Control))
                < log.iter().position(|op| *op == Op::Write(Plane::Bulk))
        );
    }

    #[test]
    fn route_revoke_and_port_disconnect_map_to_disconnected() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut revoked = ServicePump::new(ports(log.clone()), MockIrq(IrqNotice::Revoked));
        let mut control_rx = [0u8; 1];
        let mut bulk_rx = [0u8; 1];
        assert_eq!(
            revoked.step(BUDGET, buffers(&mut control_rx, &mut bulk_rx)),
            Err(BytePortError::Disconnected)
        );

        let mut disconnected = ports(log);
        disconnected.disconnected = true;
        let mut pump = ServicePump::new(disconnected, MockIrq(IrqNotice::Quiet));
        assert_eq!(
            pump.step(BUDGET, buffers(&mut control_rx, &mut bulk_rx)),
            Err(BytePortError::Disconnected)
        );
        assert_eq!(
            map_owned(OwnedError::Disconnected(Plane::Bulk)),
            BytePortError::Disconnected
        );
    }

    #[test]
    fn parts_are_recoverable_only_after_quiesce_attempt() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let pump = ServicePump::new(ports(log.clone()), MockIrq(IrqNotice::Quiet));
        let quiesced = pump.quiesce().ok().unwrap();
        let (ports, _irq) = quiesced.into_parts();
        ports.unbind();
        assert_eq!(&*log.borrow(), &[Op::Quiesce, Op::Unbind]);
    }
}
