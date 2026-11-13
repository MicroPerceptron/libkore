use super::*;
use crate::device::VirtualDevice;
use crate::mem::{Window, tests::window};
use crate::transport::Transport;
use crate::transport::tests::{ECHO_CONFIG, Echo};
use core::num::NonZeroU32;
use std::vec;
use std::vec::Vec;

const BUDGET: NonZeroU32 = NonZeroU32::new(16).unwrap();

fn layout() -> Layout {
    Layout::new(QueueSize::new(8).unwrap(), 64, 32).unwrap()
}

/// A device and the two windows a driver is given to it.
struct Bench {
    registers: Window,
    memory: Window,
    transport: Transport<Echo, Window>,
}

impl Bench {
    fn new() -> Self {
        let registers = window(regs::BYTES as usize);
        let memory = window(layout().bytes() as usize);
        let mut transport = Transport::new(registers, Echo::default()).unwrap();
        transport.lend(memory).unwrap();
        Self {
            registers,
            memory,
            transport,
        }
    }

    fn bring_up(&mut self) -> Result<(DriverRing<Window>, Slots<Window>), BringUpError> {
        self.bring_up_with(layout())
    }

    fn bring_up_with(
        &mut self,
        layout: Layout,
    ) -> Result<(DriverRing<Window>, Slots<Window>), BringUpError> {
        let transport = &mut self.transport;
        bring_up(&self.registers, self.memory, layout, || {
            transport.service(BUDGET);
        })
    }

    /// Send `request` with `room` for the answer and take the answer back.
    fn exchange(
        &mut self,
        ring: &mut DriverRing<Window>,
        slots: &mut Slots<Window>,
        request: &[u8],
        room: u32,
    ) -> Vec<u8> {
        let mut chain = slots.chain(request.len() as u32, room).expect("a slot");
        chain.write_device_readable(0, request).unwrap();
        let published = ring.publish(chain).expect("published");
        assert_eq!(published.notification(), NotificationHint::Notify);
        assert!(self.transport.service(BUDGET).interrupt);
        let used = ring.pop_used().unwrap().expect("used");
        assert_eq!(used.id(), published.id());
        let (_, len, chain) = used.into_parts();
        let mut answer = vec![0u8; len.get() as usize];
        chain.read_device_writable(0, &mut answer).unwrap();
        slots.recycle(chain);
        answer
    }
}

#[test]
fn a_layout_keeps_its_parts_apart_and_aligned() {
    for size in [2u16, 4, 8, 16, 32, 64] {
        for (request, response) in [(1u32, 1u32), (64, 32), (65_536, 65_536), (17, 4093)] {
            let layout = Layout::new(QueueSize::new(size).unwrap(), request, response).unwrap();
            let at = layout.addresses();
            let size = u64::from(size);
            assert_eq!(at.descriptors % 16, 0);
            assert_eq!(at.available % 2, 0);
            assert_eq!(at.used % 4, 0);
            assert!(at.descriptors + size * 16 <= at.available);
            assert!(at.available + 4 + size * 2 <= at.used);
            let mut end = at.used + 4 + size * 8;
            for slot in 0..layout.slots() {
                let (request_at, response_at) = layout.slot(slot);
                assert!(end <= request_at, "slot {slot} begins after what precedes");
                assert_eq!(request_at % 8, 0);
                assert_eq!(response_at % 8, 0);
                assert!(request_at + u64::from(request) <= response_at);
                end = response_at + u64::from(response);
            }
            assert!(end <= layout.bytes());
            assert_eq!(layout.slots(), (size / 2) as u16);
        }
    }
    let size = |size| QueueSize::new(size).unwrap();
    assert!(Layout::new(size(1), 8, 8).is_none(), "no room for a chain");
    assert!(
        Layout::new(size(128), 8, 8).is_none(),
        "more than a ring holds"
    );
    assert!(Layout::new(size(8), 0, 8).is_none());
    assert!(Layout::new(size(8), 8, 0).is_none());
}

#[test]
fn a_driver_identifies_brings_up_and_exchanges() {
    let mut bench = Bench::new();
    let identity = identify(&bench.registers).unwrap();
    assert_eq!(
        identity,
        Identity {
            device_id: Echo::DEVICE_ID,
            vendor_id: regs::VENDOR_KORE,
            features: F_VERSION_1,
            queue_num_max: 8,
            config_bytes: ECHO_CONFIG.len() as u32,
        }
    );
    let mut config = [0u8; ECHO_CONFIG.len()];
    read_config(&bench.registers, &mut config).unwrap();
    assert_eq!(config, ECHO_CONFIG);
    assert_eq!(
        read_config(&bench.registers, &mut [0u8; ECHO_CONFIG.len() + 1]),
        Err(BringUpError::Layout),
        "no more than the device has"
    );

    let (mut ring, mut slots) = bench.bring_up().unwrap();
    assert!(ring.state().ready());
    assert_eq!(ring.layout(), layout());
    assert_eq!(slots.free(), 4);
    assert_eq!(
        bench.transport.status(),
        STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK
    );
    assert_eq!(bench.transport.features(), F_VERSION_1);

    assert_eq!(
        bench.exchange(&mut ring, &mut slots, b"hello", 32),
        b"olleh"
    );
    assert_eq!(bench.exchange(&mut ring, &mut slots, b"device", 3), b"eci");
    // Enough exchanges to take both rings' indices round the queue.
    for round in 0..40u8 {
        let request = [round, round.wrapping_mul(7), 3];
        let answer = bench.exchange(&mut ring, &mut slots, &request, 8);
        assert_eq!(answer, [3, round.wrapping_mul(7), round]);
    }
    assert_eq!(slots.free(), 4);
    assert_eq!(bench.transport.device().commands, 42);
}

#[test]
fn a_slot_has_one_holder() {
    let mut bench = Bench::new();
    let (mut ring, mut slots) = bench.bring_up().unwrap();

    assert!(slots.chain(0, 8).is_none());
    assert!(slots.chain(8, 0).is_none());
    assert!(slots.chain(65, 8).is_none(), "more than a slot's request");
    assert!(slots.chain(8, 33).is_none(), "more than a slot's response");
    assert_eq!(slots.free(), 4, "a refusal takes nothing");

    let mut chains: Vec<_> = (0..4).map(|_| slots.chain(64, 32).unwrap()).collect();
    assert!(slots.chain(1, 1).is_none(), "every slot is taken");

    // A chain's bytes are its slot's and no other's.
    for (index, chain) in chains.iter_mut().enumerate() {
        chain.write_device_readable(0, &[index as u8; 64]).unwrap();
        assert_eq!(
            chain.write_device_readable(1, &[0; 64]),
            Err(MemoryFault::Range)
        );
        assert_eq!(
            chain.read_device_writable(30, &mut [0; 3]),
            Err(MemoryFault::Range)
        );
        assert_eq!(chain.device_readable_len(), 64);
        assert_eq!(chain.device_writable_len(), 32);
    }
    // All four in flight at once, answered in order.
    let ids: Vec<_> = chains
        .drain(..)
        .map(|chain| ring.publish(chain).unwrap().id())
        .collect();
    bench.transport.service(BUDGET);
    for (index, id) in ids.iter().enumerate() {
        let (used, len, chain) = ring.pop_used().unwrap().unwrap().into_parts();
        assert_eq!(used, *id);
        assert_eq!(len.get(), 32);
        let mut answer = [0u8; 32];
        chain.read_device_writable(0, &mut answer).unwrap();
        assert_eq!(answer, [index as u8; 32]);
        slots.recycle(chain);
    }
    assert!(ring.pop_used().unwrap().is_none());
    assert_eq!(slots.free(), 4);
}

/// VQ-017 from the driver's side: suppression in either direction loses
/// nothing.
#[test]
fn suppressed_notifications_lose_no_work() {
    let mut bench = Bench::new();
    let (mut ring, mut slots) = bench.bring_up().unwrap();
    let one = NonZeroU32::new(1).unwrap();

    // The driver asks not to be interrupted, and is not.
    ring.disable_used_notifications().unwrap();
    let mut chain = slots.chain(2, 8).unwrap();
    chain.write_device_readable(0, b"ab").unwrap();
    ring.publish(chain).unwrap();
    assert!(!bench.transport.service(BUDGET).interrupt);
    // Enabling them again finds what was used meanwhile.
    assert_eq!(
        ring.enable_used_notifications().unwrap(),
        NotificationRecheck::WorkPending
    );
    let (_, len, chain) = ring.pop_used().unwrap().unwrap().into_parts();
    assert_eq!(len.get(), 2);
    slots.recycle(chain);
    assert_eq!(
        ring.enable_used_notifications().unwrap(),
        NotificationRecheck::Idle
    );

    // The device stops listening while its budget is spent: a chain
    // published then needs no doorbell, and is served all the same.
    for _ in 0..2 {
        let mut chain = slots.chain(2, 8).unwrap();
        chain.write_device_readable(0, b"cd").unwrap();
        ring.publish(chain).unwrap();
    }
    assert!(bench.transport.service(one).pending);
    let mut chain = slots.chain(2, 8).unwrap();
    chain.write_device_readable(0, b"ef").unwrap();
    assert_eq!(
        ring.publish(chain).unwrap().notification(),
        NotificationHint::Suppressed
    );
    while bench.transport.service(one).pending {}
    let answers: Vec<_> = core::iter::from_fn(|| ring.pop_used().unwrap())
        .map(|used| {
            let (_, _, chain) = used.into_parts();
            let mut answer = [0u8; 2];
            chain.read_device_writable(0, &mut answer).unwrap();
            slots.recycle(chain);
            answer
        })
        .collect();
    assert_eq!(answers, [*b"dc", *b"dc", *b"fe"]);
}

#[test]
fn a_used_entry_is_checked_before_it_is_believed() {
    // Each forgery: the entry the device's side of the memory is made to
    // hold, as `(head, length)`.
    for (what, head, len) in [
        ("a head never published", 2u32, 1u32),
        ("a slot's second descriptor", 1, 1),
        ("no descriptor at all", 4000, 1),
        ("more than the room given", 0, 9),
    ] {
        let mut bench = Bench::new();
        let (mut ring, mut slots) = bench.bring_up().unwrap();
        let mut chain = slots.chain(2, 8).unwrap();
        chain.write_device_readable(0, b"ab").unwrap();
        ring.publish(chain).unwrap();

        let used = layout().addresses().used;
        bench.memory.store_u32(used + 4, head).unwrap();
        bench.memory.store_u32(used + 8, len).unwrap();
        bench.memory.store_u16(used + 2, 1).unwrap();
        assert_eq!(
            ring.pop_used().map(|used| used.is_some()),
            Err(QueueError::Transport(DriverRingError::UsedEntry)),
            "{what}"
        );
    }
}

#[test]
fn a_refused_bring_up_says_which_step() {
    // Not a device at all.
    let blank = window(regs::BYTES as usize);
    let memory = window(layout().bytes() as usize);
    let bring = |registers: &Window| bring_up(registers, memory, layout(), || {}).map(|_| ());
    assert_eq!(bring(&blank), Err(BringUpError::NotADevice));
    blank.store_u32(regs::MAGIC, regs::MAGIC_VALUE).unwrap();
    blank.store_u32(regs::VERSION, 2).unwrap();
    assert_eq!(bring(&blank), Err(BringUpError::Version(2)));

    // A device that does not answer a reset.
    blank.store_u32(regs::VERSION, regs::VERSION_VALUE).unwrap();
    blank
        .store_u32(regs::DEVICE_STATUS, STATUS_DRIVER_OK)
        .unwrap();
    assert_eq!(bring(&blank), Err(BringUpError::Reset));

    // One that offers nothing a driver can accept, and one that never
    // answers the features it was sent. The driver gives up on both.
    blank.store_u32(regs::DEVICE_STATUS, 0).unwrap();
    assert_eq!(bring(&blank), Err(BringUpError::Features));
    assert_eq!(blank.load_u32(regs::DRIVER_STATUS), Ok(STATUS_FAILED));
    blank.store_u64(regs::DEVICE_FEATURES, F_VERSION_1).unwrap();
    assert_eq!(bring(&blank), Err(BringUpError::Features));
    assert_eq!(blank.load_u32(regs::DRIVER_STATUS), Ok(STATUS_FAILED));

    // A queue larger than the device takes, and memory smaller than the
    // layout.
    let mut bench = Bench::new();
    let wide = Layout::new(QueueSize::new(16).unwrap(), 8, 8).unwrap();
    assert_eq!(
        bench.bring_up_with(wide).map(|_| ()),
        Err(BringUpError::Layout)
    );
    let deep = Layout::new(QueueSize::new(8).unwrap(), 4096, 4096).unwrap();
    assert_eq!(
        bench.bring_up_with(deep).map(|_| ()),
        Err(BringUpError::Layout)
    );
    assert_eq!(
        bench.registers.load_u32(regs::DRIVER_STATUS),
        Ok(STATUS_FAILED)
    );
    // The device heard the driver give up, and a new bring-up starts over.
    bench.transport.service(BUDGET);
    assert_ne!(bench.transport.status() & STATUS_FAILED, 0);
    bench.bring_up().expect("after a reset");

    // A queue the device refuses: its chains are longer than a queue of
    // two can hold.
    let mut bench = Bench::new();
    let short = Layout::new(QueueSize::new(2).unwrap(), 8, 8).unwrap();
    assert_eq!(
        bench.bring_up_with(short).map(|_| ()),
        Err(BringUpError::Queue)
    );
    assert_ne!(bench.transport.status() & STATUS_DEVICE_NEEDS_RESET, 0);
    bench.bring_up().expect("after a reset");
}

/// A second bring-up is a reset: the device forgets, both rings start from
/// nothing, and what the first one gave is dead.
#[test]
fn bringing_up_again_resets_the_device() {
    let mut bench = Bench::new();
    let (mut ring, mut slots) = bench.bring_up().unwrap();
    for _ in 0..3 {
        bench.exchange(&mut ring, &mut slots, b"before", 8);
    }
    // One left with the device across the reset.
    let mut chain = slots.chain(4, 8).unwrap();
    chain.write_device_readable(0, b"lost").unwrap();
    ring.publish(chain).unwrap();

    let (mut again, mut pool) = bench.bring_up().unwrap();
    assert_eq!(bench.transport.device().resets, 1);
    assert_eq!(
        bench.transport.device().commands,
        3,
        "the lost one never ran"
    );
    assert_eq!(pool.free(), 4);
    assert!(again.pop_used().unwrap().is_none(), "nothing from before");
    assert_eq!(bench.exchange(&mut again, &mut pool, b"after", 8), b"retfa");

    // The old ring gives its chain back and serves no more.
    let epoch = ring.state().epoch();
    assert_eq!(
        ring.reset(epoch).map(|_| ()),
        Err(QueueError::InvalidConfiguration(
            QueueConfigError::NonIncreasingEpoch
        ))
    );
    let reclaimed = ring.reset(epoch.checked_next().unwrap()).unwrap();
    assert_eq!(reclaimed.len(), 1);
    let (id, chain) = reclaimed.into_iter().next().unwrap().into_parts();
    assert_eq!(id.epoch(), epoch);
    assert_eq!(chain.device_readable_len(), 4);
    assert!(!ring.state().ready());
    assert_eq!(
        ring.pop_used().map(|used| used.is_some()),
        Err(QueueError::NotReady)
    );
    let (returned, kind) = ring.publish(chain).unwrap_err().into_parts();
    assert_eq!(kind, PublishErrorKind::NotReady);
    assert_eq!(returned.device_readable_len(), 4, "handed back unchanged");
}
