//! Strict VirtIO 1.3 multiport-console control state for HostLink.
//!
//! Port zero is intentionally unused.  QEMU is configured with two generic
//! ports at stable ids 1 and 2; their names are data-plane selectors only.
//! Delivery authority remains in the HostLink service's guest-held mailbox
//! table and is never derived from these strings.

use core::str;

pub const CONTROL_PORT_NAME: &[u8] = b"org.microperceptron.hostlink.control";
pub const BULK_PORT_NAME: &[u8] = b"org.microperceptron.hostlink.bulk";
pub const CONTROL_RX_QUEUE: u16 = 2;
pub const CONTROL_TX_QUEUE: u16 = 3;
pub const HOSTLINK_CONTROL_RX_QUEUE: u16 = 4;
pub const HOSTLINK_CONTROL_TX_QUEUE: u16 = 5;
pub const HOSTLINK_BULK_RX_QUEUE: u16 = 6;
pub const HOSTLINK_BULK_TX_QUEUE: u16 = 7;
pub const REQUIRED_MAX_PORTS: u32 = 3;
pub const CONTROL_HEADER_LEN: usize = 8;
pub const MAX_PORT_NAME: usize = 64;
/// QEMU currently appends one NUL byte to `PORT_NAME`, despite VirtIO 1.3
/// specifying an unterminated UTF-8 payload. Keep that compatibility byte
/// outside the normalized name bound.
pub const MAX_PORT_NAME_WIRE: usize = MAX_PORT_NAME + 1;

const EVENT_DEVICE_READY: u16 = 0;
const EVENT_PORT_ADD: u16 = 1;
const EVENT_PORT_REMOVE: u16 = 2;
const EVENT_PORT_READY: u16 = 3;
const EVENT_CONSOLE_PORT: u16 = 4;
const EVENT_RESIZE: u16 = 5;
const EVENT_PORT_OPEN: u16 = 6;
const EVENT_PORT_NAME: u16 = 7;

/// HostLink's two traffic classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    Control,
    Bulk,
}

/// The fixed RX/TX queues belonging to one generic port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortQueues {
    pub receive: u16,
    pub transmit: u16,
}

impl Plane {
    pub const fn queues(self) -> PortQueues {
        match self {
            Self::Control => PortQueues {
                receive: HOSTLINK_CONTROL_RX_QUEUE,
                transmit: HOSTLINK_CONTROL_TX_QUEUE,
            },
            Self::Bulk => PortQueues {
                receive: HOSTLINK_BULK_RX_QUEUE,
                transmit: HOSTLINK_BULK_TX_QUEUE,
            },
        }
    }
}

/// One decoded control receiveq record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlFrame<'a> {
    pub id: u32,
    pub event: u16,
    pub value: u16,
    pub payload: &'a [u8],
}

impl<'a> ControlFrame<'a> {
    pub fn decode(bytes: &'a [u8]) -> Result<Self, ConsoleError> {
        if bytes.len() < CONTROL_HEADER_LEN {
            return Err(ConsoleError::Truncated);
        }
        Ok(Self {
            id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            event: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            value: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            payload: &bytes[8..],
        })
    }
}

/// One driver-to-device control transmitq record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlCommand {
    pub id: u32,
    pub event: u16,
    pub value: u16,
}

impl ControlCommand {
    pub fn encode(self) -> [u8; CONTROL_HEADER_LEN] {
        let mut bytes = [0u8; CONTROL_HEADER_LEN];
        bytes[0..4].copy_from_slice(&self.id.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.event.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.value.to_le_bytes());
        bytes
    }
}

/// Epoch-scoped handle returned only after the exact expected port name binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortHandle {
    pub plane: Plane,
    pub id: u32,
    pub queues: PortQueues,
    epoch: u32,
    generation: u32,
}

impl PortHandle {
    pub const fn epoch(self) -> u32 {
        self.epoch
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// State-machine output for the service to enact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlAction {
    /// Publish this record on control transmitq before accepting name/open.
    SendPortReady(ControlCommand),
    /// Exact port name accepted. Publish the guest-open edge before exposing
    /// the fixed queue pair to HostLink traffic.
    Bound {
        handle: PortHandle,
        open: ControlCommand,
    },
    /// Host chardev opened this already-bound port.
    Connected(PortHandle),
    /// Host close/remove deterministically disconnects this port.
    Disconnected(PortHandle),
    /// A port vanished before its name established a HostLink binding.
    RemovedUnbound(u32),
}

/// Fail-closed multiport-console refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleError {
    Truncated,
    MaxPorts(u32),
    NotReady,
    AlreadyReady,
    WrongDirection(u16),
    UnsupportedEvent(u16),
    UnexpectedPort(u32),
    DuplicatePort(u32),
    UnknownPort(u32),
    EventOrder { id: u32, event: u16 },
    InvalidValue { event: u16, value: u16 },
    UnexpectedPayload(u16),
    InvalidName,
    UnexpectedName(u32),
    DuplicateName(Plane),
    DuplicateEvent { id: u32, event: u16 },
    StaleHandle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Absent,
    Added,
    Named(Plane),
    Open(Plane),
}

/// Bounded state for exactly the two named HostLink ports.
pub struct MultiportConsole {
    phases: [Phase; REQUIRED_MAX_PORTS as usize],
    generations: [u32; REQUIRED_MAX_PORTS as usize],
    ready: bool,
    epoch: u32,
}

impl MultiportConsole {
    pub fn new(max_ports: u32) -> Result<Self, ConsoleError> {
        if max_ports < REQUIRED_MAX_PORTS {
            return Err(ConsoleError::MaxPorts(max_ports));
        }
        Ok(Self {
            phases: [Phase::Absent; REQUIRED_MAX_PORTS as usize],
            generations: [0; REQUIRED_MAX_PORTS as usize],
            ready: false,
            epoch: 1,
        })
    }

    /// Begin multiport discovery after all control and port RX queues have
    /// been populated. The returned command is sent on control transmitq.
    pub fn begin(&mut self) -> Result<ControlCommand, ConsoleError> {
        if self.ready {
            return Err(ConsoleError::AlreadyReady);
        }
        self.ready = true;
        Ok(ControlCommand {
            id: 0,
            event: EVENT_DEVICE_READY,
            value: 1,
        })
    }

    pub const fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Consume one complete control receiveq record.
    pub fn handle(&mut self, bytes: &[u8]) -> Result<ControlAction, ConsoleError> {
        if !self.ready {
            return Err(ConsoleError::NotReady);
        }
        let frame = ControlFrame::decode(bytes)?;
        match frame.event {
            EVENT_DEVICE_READY | EVENT_PORT_READY => Err(ConsoleError::WrongDirection(frame.event)),
            EVENT_PORT_ADD => self.port_add(frame),
            EVENT_PORT_REMOVE => self.port_remove(frame),
            EVENT_PORT_OPEN => self.port_open(frame),
            EVENT_PORT_NAME => self.port_name(frame),
            EVENT_CONSOLE_PORT | EVENT_RESIZE => Err(ConsoleError::UnsupportedEvent(frame.event)),
            other => Err(ConsoleError::UnsupportedEvent(other)),
        }
    }

    /// Resolve a handle only while it belongs to this reset epoch and the
    /// named port is still present.
    pub fn validate(&self, handle: PortHandle) -> Result<bool, ConsoleError> {
        if handle.epoch != self.epoch
            || handle.id as usize >= self.phases.len()
            || self.generations[handle.id as usize] != handle.generation
        {
            return Err(ConsoleError::StaleHandle);
        }
        match self.phases[handle.id as usize] {
            Phase::Named(plane) if plane == handle.plane => Ok(false),
            Phase::Open(plane) if plane == handle.plane => Ok(true),
            _ => Err(ConsoleError::StaleHandle),
        }
    }

    /// Reset after the VirtIO device reset. All prior port handles become
    /// stale and discovery must begin again.
    pub fn reset(&mut self) {
        self.phases = [Phase::Absent; REQUIRED_MAX_PORTS as usize];
        self.generations = [0; REQUIRED_MAX_PORTS as usize];
        self.ready = false;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
    }

    fn port_add(&mut self, frame: ControlFrame<'_>) -> Result<ControlAction, ConsoleError> {
        self.check_empty(&frame)?;
        if frame.value != 1 {
            return Err(ConsoleError::InvalidValue {
                event: frame.event,
                value: frame.value,
            });
        }
        let index = self.port_index(frame.id)?;
        if self.phases[index] != Phase::Absent {
            return Err(ConsoleError::DuplicatePort(frame.id));
        }
        self.generations[index] = self.generations[index].wrapping_add(1);
        if self.generations[index] == 0 {
            self.generations[index] = 1;
        }
        self.phases[index] = Phase::Added;
        Ok(ControlAction::SendPortReady(ControlCommand {
            id: frame.id,
            event: EVENT_PORT_READY,
            value: 1,
        }))
    }

    fn port_name(&mut self, frame: ControlFrame<'_>) -> Result<ControlAction, ConsoleError> {
        let index = self.port_index(frame.id)?;
        if self.phases[index] != Phase::Added {
            return Err(ConsoleError::EventOrder {
                id: frame.id,
                event: frame.event,
            });
        }
        // `value` has no specified meaning for PORT_NAME and QEMU sets it to
        // one. QEMU also includes one terminal NUL, contrary to VirtIO 1.3.
        // Accept only that exact deployed encoding in addition to the
        // conforming unterminated form; embedded/repeated NULs remain invalid.
        let name = frame.payload.strip_suffix(&[0]).unwrap_or(frame.payload);
        if name.is_empty()
            || name.len() > MAX_PORT_NAME
            || name.contains(&0)
            || frame.payload.len() > MAX_PORT_NAME_WIRE
            || str::from_utf8(name).is_err()
        {
            return Err(ConsoleError::InvalidName);
        }
        let plane = if name == CONTROL_PORT_NAME {
            Plane::Control
        } else if name == BULK_PORT_NAME {
            Plane::Bulk
        } else {
            return Err(ConsoleError::UnexpectedName(frame.id));
        };
        if self
            .phases
            .iter()
            .enumerate()
            .any(|(at, phase)| at != index && Self::phase_plane(*phase) == Some(plane))
        {
            return Err(ConsoleError::DuplicateName(plane));
        }
        let expected = match frame.id {
            1 => Plane::Control,
            2 => Plane::Bulk,
            _ => return Err(ConsoleError::UnexpectedPort(frame.id)),
        };
        if plane != expected {
            return Err(ConsoleError::UnexpectedName(frame.id));
        }
        self.phases[index] = Phase::Named(plane);
        Ok(ControlAction::Bound {
            handle: self.make_handle(frame.id, plane),
            open: ControlCommand {
                id: frame.id,
                event: EVENT_PORT_OPEN,
                value: 1,
            },
        })
    }

    fn port_open(&mut self, frame: ControlFrame<'_>) -> Result<ControlAction, ConsoleError> {
        self.check_empty(&frame)?;
        if frame.value > 1 {
            return Err(ConsoleError::InvalidValue {
                event: frame.event,
                value: frame.value,
            });
        }
        let index = self.port_index(frame.id)?;
        match (self.phases[index], frame.value) {
            (Phase::Named(plane), 1) => {
                self.phases[index] = Phase::Open(plane);
                Ok(ControlAction::Connected(self.make_handle(frame.id, plane)))
            }
            (Phase::Open(plane), 0) => {
                self.phases[index] = Phase::Named(plane);
                Ok(ControlAction::Disconnected(
                    self.make_handle(frame.id, plane),
                ))
            }
            (Phase::Named(plane), 0) => Ok(ControlAction::Disconnected(
                self.make_handle(frame.id, plane),
            )),
            (Phase::Open(_), 1) => Err(ConsoleError::DuplicateEvent {
                id: frame.id,
                event: frame.event,
            }),
            _ => Err(ConsoleError::EventOrder {
                id: frame.id,
                event: frame.event,
            }),
        }
    }

    fn port_remove(&mut self, frame: ControlFrame<'_>) -> Result<ControlAction, ConsoleError> {
        self.check_empty(&frame)?;
        if frame.value != 1 {
            return Err(ConsoleError::InvalidValue {
                event: frame.event,
                value: frame.value,
            });
        }
        let index = self.port_index(frame.id)?;
        match self.phases[index] {
            Phase::Added => {
                self.phases[index] = Phase::Absent;
                Ok(ControlAction::RemovedUnbound(frame.id))
            }
            Phase::Named(plane) | Phase::Open(plane) => {
                let handle = self.make_handle(frame.id, plane);
                self.phases[index] = Phase::Absent;
                Ok(ControlAction::Disconnected(handle))
            }
            Phase::Absent => Err(ConsoleError::UnknownPort(frame.id)),
        }
    }

    fn port_index(&self, id: u32) -> Result<usize, ConsoleError> {
        match id {
            1 | 2 => Ok(id as usize),
            _ => Err(ConsoleError::UnexpectedPort(id)),
        }
    }

    fn check_empty(&self, frame: &ControlFrame<'_>) -> Result<(), ConsoleError> {
        if frame.payload.is_empty() {
            Ok(())
        } else {
            Err(ConsoleError::UnexpectedPayload(frame.event))
        }
    }

    const fn phase_plane(phase: Phase) -> Option<Plane> {
        match phase {
            Phase::Named(plane) | Phase::Open(plane) => Some(plane),
            Phase::Absent | Phase::Added => None,
        }
    }

    const fn make_handle(&self, id: u32, plane: Plane) -> PortHandle {
        PortHandle {
            plane,
            id,
            queues: plane.queues(),
            epoch: self.epoch,
            generation: self.generations[id as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn frame(id: u32, event: u16, value: u16, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::from(ControlCommand { id, event, value }.encode());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn ready() -> MultiportConsole {
        let mut console = MultiportConsole::new(3).unwrap();
        assert_eq!(
            console.begin().unwrap(),
            ControlCommand {
                id: 0,
                event: EVENT_DEVICE_READY,
                value: 1
            }
        );
        console
    }

    fn bind(console: &mut MultiportConsole, id: u32, name: &[u8]) -> PortHandle {
        assert_eq!(
            console.handle(&frame(id, EVENT_PORT_ADD, 1, &[])),
            Ok(ControlAction::SendPortReady(ControlCommand {
                id,
                event: EVENT_PORT_READY,
                value: 1
            }))
        );
        match console
            .handle(&frame(id, EVENT_PORT_NAME, 0, name))
            .unwrap()
        {
            ControlAction::Bound { handle, open } => {
                assert_eq!(
                    open,
                    ControlCommand {
                        id,
                        event: EVENT_PORT_OPEN,
                        value: 1,
                    }
                );
                handle
            }
            action => panic!("unexpected action: {action:?}"),
        }
    }

    #[test]
    fn exact_names_bind_only_the_fixed_hostlink_queue_pairs() {
        let mut console = ready();
        let control = bind(&mut console, 1, CONTROL_PORT_NAME);
        let bulk = bind(&mut console, 2, BULK_PORT_NAME);
        assert_eq!(control.plane, Plane::Control);
        assert_eq!(
            control.queues,
            PortQueues {
                receive: 4,
                transmit: 5
            }
        );
        assert_eq!(bulk.plane, Plane::Bulk);
        assert_eq!(
            bulk.queues,
            PortQueues {
                receive: 6,
                transmit: 7
            }
        );
        assert_ne!(control.queues.receive, CONTROL_RX_QUEUE);
    }

    #[test]
    fn qemu_port_name_encoding_binds_without_weakening_name_checks() {
        let mut console = ready();
        console.handle(&frame(1, EVENT_PORT_ADD, 1, &[])).unwrap();
        let mut qemu_name = Vec::from(CONTROL_PORT_NAME);
        qemu_name.push(0);
        assert!(matches!(
            console.handle(&frame(1, EVENT_PORT_NAME, 1, &qemu_name)),
            Ok(ControlAction::Bound {
                handle: PortHandle {
                    plane: Plane::Control,
                    ..
                },
                open: ControlCommand {
                    id: 1,
                    event: EVENT_PORT_OPEN,
                    value: 1,
                },
            })
        ));

        let mut malformed = ready();
        malformed.handle(&frame(1, EVENT_PORT_ADD, 1, &[])).unwrap();
        qemu_name.push(0);
        assert_eq!(
            malformed.handle(&frame(1, EVENT_PORT_NAME, 1, &qemu_name)),
            Err(ConsoleError::InvalidName)
        );
    }

    #[test]
    fn control_event_order_is_fail_closed() {
        let mut console = ready();
        assert!(matches!(
            console.handle(&frame(1, EVENT_PORT_NAME, 0, CONTROL_PORT_NAME)),
            Err(ConsoleError::EventOrder { .. })
        ));
        assert!(matches!(
            console.handle(&frame(1, EVENT_PORT_OPEN, 1, &[])),
            Err(ConsoleError::EventOrder { .. })
        ));
        assert_eq!(
            console.handle(&frame(0, EVENT_PORT_ADD, 1, &[])),
            Err(ConsoleError::UnexpectedPort(0))
        );
        assert_eq!(
            console.handle(&frame(1, EVENT_DEVICE_READY, 1, &[])),
            Err(ConsoleError::WrongDirection(EVENT_DEVICE_READY))
        );
    }

    #[test]
    fn duplicate_ids_names_and_wrong_names_are_rejected() {
        let mut console = ready();
        console.handle(&frame(1, EVENT_PORT_ADD, 1, &[])).unwrap();
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_ADD, 1, &[])),
            Err(ConsoleError::DuplicatePort(1))
        );
        console
            .handle(&frame(1, EVENT_PORT_NAME, 0, CONTROL_PORT_NAME))
            .unwrap();
        console.handle(&frame(2, EVENT_PORT_ADD, 1, &[])).unwrap();
        assert_eq!(
            console.handle(&frame(2, EVENT_PORT_NAME, 0, CONTROL_PORT_NAME)),
            Err(ConsoleError::DuplicateName(Plane::Control))
        );

        let mut wrong = ready();
        wrong.handle(&frame(1, EVENT_PORT_ADD, 1, &[])).unwrap();
        assert_eq!(
            wrong.handle(&frame(1, EVENT_PORT_NAME, 0, BULK_PORT_NAME)),
            Err(ConsoleError::UnexpectedName(1))
        );
        assert_eq!(
            wrong.handle(&frame(1, EVENT_PORT_NAME, 0, b"hostlink")),
            Err(ConsoleError::UnexpectedName(1))
        );
    }

    #[test]
    fn disconnect_and_remove_are_deterministic() {
        let mut console = ready();
        let bound = bind(&mut console, 1, CONTROL_PORT_NAME);
        let opened = match console.handle(&frame(1, EVENT_PORT_OPEN, 1, &[])).unwrap() {
            ControlAction::Connected(handle) => handle,
            action => panic!("unexpected action: {action:?}"),
        };
        assert_eq!(console.validate(opened), Ok(true));
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_OPEN, 0, &[])),
            Ok(ControlAction::Disconnected(bound))
        );
        assert_eq!(console.validate(bound), Ok(false));
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_REMOVE, 1, &[])),
            Ok(ControlAction::Disconnected(bound))
        );
        assert_eq!(console.validate(bound), Err(ConsoleError::StaleHandle));
    }

    #[test]
    fn initial_closed_state_and_remove_readd_advance_port_generation() {
        let mut console = ready();
        let first = bind(&mut console, 1, CONTROL_PORT_NAME);
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_OPEN, 0, &[])),
            Ok(ControlAction::Disconnected(first))
        );
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_REMOVE, 1, &[])),
            Ok(ControlAction::Disconnected(first))
        );
        let second = bind(&mut console, 1, CONTROL_PORT_NAME);
        assert_ne!(first.generation(), second.generation());
        assert_eq!(console.validate(first), Err(ConsoleError::StaleHandle));
        assert_eq!(console.validate(second), Ok(false));

        console.reset();
        console.begin().unwrap();
        assert_eq!(
            console.handle(&frame(2, EVENT_PORT_ADD, 1, &[])),
            Ok(ControlAction::SendPortReady(ControlCommand {
                id: 2,
                event: EVENT_PORT_READY,
                value: 1,
            }))
        );
        assert_eq!(
            console.handle(&frame(2, EVENT_PORT_REMOVE, 1, &[])),
            Ok(ControlAction::RemovedUnbound(2))
        );
    }

    #[test]
    fn reset_invalidates_handles_and_requires_device_ready_again() {
        let mut console = ready();
        let old = bind(&mut console, 1, CONTROL_PORT_NAME);
        let epoch = console.epoch();
        console.reset();
        assert_ne!(console.epoch(), epoch);
        assert_eq!(console.validate(old), Err(ConsoleError::StaleHandle));
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_ADD, 1, &[])),
            Err(ConsoleError::NotReady)
        );
        console.begin().unwrap();
        assert!(bind(&mut console, 1, CONTROL_PORT_NAME).epoch() > old.epoch());
    }

    #[test]
    fn malformed_names_and_payloads_are_rejected_without_state_growth() {
        let mut console = ready();
        console.handle(&frame(1, EVENT_PORT_ADD, 1, &[])).unwrap();
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_NAME, 0, &[0xff])),
            Err(ConsoleError::InvalidName)
        );
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_NAME, 0, b"bad\0name")),
            Err(ConsoleError::InvalidName)
        );
        assert_eq!(
            console.handle(&frame(1, EVENT_PORT_ADD, 1, b"extra")),
            Err(ConsoleError::UnexpectedPayload(EVENT_PORT_ADD))
        );
        assert_eq!(ControlFrame::decode(&[0; 7]), Err(ConsoleError::Truncated));
    }

    #[test]
    fn event_vocabulary_and_ids_are_total_over_small_untrusted_domain() {
        for event in 0u16..=15 {
            for id in 0u32..=4 {
                for value in 0u16..=2 {
                    let mut console = ready();
                    let _ = console.handle(&frame(id, event, value, &[]));
                    assert_eq!(console.epoch(), 1);
                }
            }
        }
    }
}
