use super::*;
use core::cell::Cell;
use core::ptr::NonNull;
use kore_vdev::driver::{self, DriverChain, DriverRing, Layout, Slots};
use kore_vdev::regs::{
    STATUS_ACKNOWLEDGE, STATUS_DEVICE_NEEDS_RESET, STATUS_DRIVER, STATUS_DRIVER_OK,
    STATUS_FEATURES_OK,
};
use std::boxed::Box;
use std::rc::Rc;
use std::vec;
use std::vec::Vec;
use virtio_accel_guest::{Completion, GuestClient, GuestConfig, RequestPoll};
use virtio_accel_proto::{RequestHeader, ResponseHeader, WireDeviceInfo};
use virtio_accel_transport::QueueSize;

/// Host memory standing for a lease both sides map.
fn window(bytes: u64) -> Window {
    let words: &'static mut [u64] =
        Box::leak(vec![0u64; (bytes as usize).div_ceil(8)].into_boxed_slice());
    // SAFETY: a leaked, eight-byte aligned allocation of at least `bytes`
    // bytes that nothing else holds a reference into.
    unsafe {
        Window::new(
            NonNull::new(words.as_mut_ptr().cast()).unwrap(),
            bytes as usize,
        )
    }
}

fn page() -> Window {
    window(regs::BYTES)
}

/// What keeps lent memory mapped, in a test: it counts its own end.
#[derive(Debug)]
struct Owner(Rc<Cell<u32>>);

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

/// A count of owners dropped, and a way to make one.
fn owners() -> (Rc<Cell<u32>>, impl Fn() -> Owner) {
    let dropped = Rc::new(Cell::new(0));
    let make = {
        let dropped = dropped.clone();
        move || Owner(dropped.clone())
    };
    (dropped, make)
}

fn magic(registers: &Window) -> u32 {
    registers.load_u32(regs::MAGIC).unwrap()
}

const LIVE: u32 = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK;

#[test]
fn a_seat_is_one_faces_and_names_its_occupant() {
    let mut devices = Devices::<Owner>::new();
    let registers = page();
    registers.store_u32(regs::MAGIC, 0xdead_beef).unwrap();

    assert!(!devices.seat(MAX_SEATS, 7, registers), "no such slot");
    assert!(!devices.seat(2, 7, window(regs::BYTES - 8)), "not a page");
    assert!(!devices.seated(2));
    assert_eq!(devices.arrival(2), None);

    assert!(devices.seat(2, 7, registers));
    assert_eq!(
        magic(&registers),
        0,
        "an earlier face's device is not shown"
    );
    assert!(devices.seated(2));
    assert_eq!(devices.arrival(2), Some(7));
    assert!(!devices.seat(2, 8, page()), "the slot has its occupant");
    assert_eq!(devices.arrival(2), Some(7));
    assert!(!devices.seated(1) && !devices.seated(3));

    // The table grows to whatever slot the runtime names.
    assert!(devices.seat(MAX_SEATS - 1, 9, page()));
    assert_eq!(devices.arrival(MAX_SEATS - 1), Some(9));

    devices.unseat(2);
    assert!(!devices.seated(2));
    devices.unseat(2);
    devices.unseat(MAX_SEATS + 5);
    assert!(devices.seat(2, 8, page()), "and the slot is free again");
    assert_eq!(devices.arrival(2), Some(8));
}

#[test]
fn a_driver_opens_the_device_it_names_once() {
    let mut devices = Devices::<Owner>::new();
    let registers = page();
    assert_eq!(devices.open_device(0, DEVICE_ID), Err(Refusal::Unavailable));
    assert!(devices.seat(0, 1, registers));

    assert_eq!(devices.open_device(0, 0), Err(Refusal::NoDevice));
    assert_eq!(
        devices.open_device(0, u32::from_le_bytes(*b"echo")),
        Err(Refusal::NoDevice)
    );
    assert_eq!(magic(&registers), 0);
    assert_eq!(devices.open(), 0);

    assert_eq!(devices.open_device(0, DEVICE_ID), Ok(()));
    assert_eq!(devices.open(), 1);
    let identity = driver::identify(&registers).expect("the block is written");
    assert_eq!(identity.device_id, DEVICE_ID);
    assert_eq!(identity.vendor_id, regs::VENDOR_KORE);
    assert_eq!(devices.open_device(0, DEVICE_ID), Err(Refusal::Busy));
    assert_eq!(devices.open(), 1);

    assert_eq!(devices.close_device(0), Ok(()));
    assert_eq!(magic(&registers), 0, "no device is shown once it is closed");
    assert_eq!(devices.open(), 0);
    assert_eq!(devices.close_device(0), Err(Refusal::Busy));
    assert_eq!(devices.close_device(9), Err(Refusal::Busy));
    assert_eq!(devices.open_device(0, DEVICE_ID), Ok(()), "and opens again");
}

#[test]
fn only_so_many_devices_are_open_at_once() {
    let mut devices = Devices::<Owner>::new();
    for slot in 0..=MAX_DEVICES {
        assert!(devices.seat(slot, slot as u16, page()));
    }
    for slot in 0..MAX_DEVICES {
        assert_eq!(devices.open_device(slot, DEVICE_ID), Ok(()));
    }
    assert_eq!(devices.open(), MAX_DEVICES);
    assert_eq!(
        devices.open_device(MAX_DEVICES, DEVICE_ID),
        Err(Refusal::Limit)
    );
    // The seat is still the driver's: it asks again when another closes,
    // or when another's face is retired with its device open.
    assert_eq!(devices.close_device(1), Ok(()));
    assert_eq!(devices.open_device(MAX_DEVICES, DEVICE_ID), Ok(()));
    assert_eq!(devices.open_device(1, DEVICE_ID), Err(Refusal::Limit));
    devices.unseat(0);
    assert_eq!(devices.open(), MAX_DEVICES - 1);
    assert_eq!(devices.open_device(1, DEVICE_ID), Ok(()));
    // The heap the row is founded with is sized by this.
    assert_eq!(DEVICE_HEAP_BYTES, 8 << 20);
}

#[test]
fn lent_memory_is_kept_exactly_as_long_as_the_device_has_it() {
    let (dropped, owner) = owners();
    let mut devices = Devices::<Owner>::new();
    let registers = page();
    assert!(devices.seat(0, 1, registers));
    let memory = window(4 * PAGE_BYTES);

    // Nothing to lend to, and the memory goes straight back.
    assert_eq!(devices.lend(0, memory, owner()), Err(Refusal::Busy));
    assert_eq!(devices.lend(3, memory, owner()), Err(Refusal::Busy));
    assert_eq!(dropped.get(), 2);

    devices.open_device(0, DEVICE_ID).unwrap();
    for bad in [0, 8, PAGE_BYTES - 8, PAGE_BYTES + 8] {
        assert_eq!(
            devices.lend(0, window(bad), owner()),
            Err(Refusal::Memory),
            "{bad} bytes"
        );
    }
    assert_eq!(dropped.get(), 6);

    assert_eq!(devices.lend(0, memory, owner()), Ok(4));
    assert_eq!(dropped.get(), 6, "kept while the device has it");
    // Lent again in the first state: the old memory goes back.
    assert_eq!(devices.lend(0, window(PAGE_BYTES), owner()), Ok(1));
    assert_eq!(dropped.get(), 7);

    // Once the driver has taken a step, the device may be using it.
    registers
        .store_u32(regs::DRIVER_STATUS, STATUS_ACKNOWLEDGE)
        .unwrap();
    devices.service(|_, _| {});
    assert_eq!(devices.lend(0, memory, owner()), Err(Refusal::Busy));
    assert_eq!(dropped.get(), 8, "the refused memory, not the lent");

    // Closing the device gives it back; so does retiring the face.
    devices.close_device(0).unwrap();
    assert_eq!(dropped.get(), 9);
    devices.open_device(0, DEVICE_ID).unwrap();
    assert_eq!(devices.lend(0, memory, owner()), Ok(4));
    devices.unseat(0);
    assert_eq!(dropped.get(), 10);
}

type Client = GuestClient<DriverRing<Window>>;

/// One driver program: its two windows, and what a bring-up gave it.
struct Program {
    slot: usize,
    registers: Window,
    client: Client,
    slots: Slots<Window>,
}

const REQUEST: u32 = core::mem::size_of::<RequestHeader>() as u32;
const INFO: u32 =
    (core::mem::size_of::<ResponseHeader>() + core::mem::size_of::<WireDeviceInfo>()) as u32;

/// Seat a driver in `slot`, open the accelerator, lend it memory and bring
/// it up, with the reference guest client over the queue.
fn program(devices: &mut Devices<Owner>, slot: usize, owner: Owner, size: u16) -> Program {
    let registers = page();
    assert!(devices.seat(slot, slot as u16 + 100, registers));
    devices.open_device(slot, DEVICE_ID).unwrap();
    let layout = Layout::new(QueueSize::new(size).unwrap(), 256, 256).unwrap();
    let memory = window(layout.bytes().next_multiple_of(PAGE_BYTES));
    devices.lend(slot, memory, owner).unwrap();

    let identity = driver::identify(&registers).unwrap();
    let mut config = [0u8; 16];
    driver::read_config(&registers, &mut config).unwrap();
    let wire = virtio_accel_proto::read_exact(&config).unwrap();
    let (ring, slots) = driver::bring_up(&registers, memory, layout, || {
        devices.service(|_, _| {});
    })
    .expect("the device comes up");
    let config = GuestConfig::new(wire, identity.features & !regs::F_VERSION_1, layout.slots())
        .expect("a configuration the reference driver accepts");
    Program {
        slot,
        registers,
        client: GuestClient::new(ring, config).unwrap(),
        slots,
    }
}

impl Program {
    /// Publish one discovery command.
    fn ask(&mut self) -> virtio_accel_guest::Pending<virtio_accel_guest::GetDeviceInfo> {
        let chain: DriverChain<Window> = self.slots.chain(REQUEST, INFO).expect("a free slot");
        self.client.get_device_info(chain).ok().expect("started")
    }

    /// Take one answer, which must be there.
    fn answer(&mut self, pending: virtio_accel_guest::Pending<virtio_accel_guest::GetDeviceInfo>) {
        let RequestPoll::Ready(Completion::Success { output, chain }) = self.client.poll(pending)
        else {
            panic!("the command was answered");
        };
        assert_eq!(&output.uuid[..13], b"kore-spmd-cpu");
        self.slots.recycle(chain);
    }

    fn interrupts(&self) -> u32 {
        self.registers.load_u32(regs::INTERRUPTS).unwrap()
    }
}

/// One doorbell serves every open device, and each driver is interrupted
/// for its own work and no other's.
#[test]
fn a_doorbell_reaches_every_open_device() {
    let (_, owner) = owners();
    let mut devices = Devices::<Owner>::new();
    let mut first = program(&mut devices, 1, owner(), 16);
    let mut second = program(&mut devices, 4, owner(), 16);
    // A face with no device open is not looked at.
    assert!(devices.seat(2, 50, page()));
    let (before_first, before_second) = (first.interrupts(), second.interrupts());

    let asked_first = first.ask();
    let asked_second = [second.ask(), second.ask()];
    let mut raised = Vec::new();
    assert!(!devices.service(|slot, registers| {
        raised.push((slot, registers.load_u32(regs::INTERRUPTS).unwrap()));
    }));
    assert_eq!(
        raised,
        [(1, before_first + 1), (4, before_second + 1)],
        "each told once, its count already moved"
    );
    first.answer(asked_first);
    for asked in asked_second {
        second.answer(asked);
    }

    // Only the driver with work is interrupted.
    let asked = first.ask();
    let mut raised = Vec::new();
    devices.service(|slot, _| raised.push(slot));
    assert_eq!(raised, [first.slot]);
    first.answer(asked);
    assert_eq!(second.interrupts(), before_second + 1);

    // And an idle doorbell interrupts nobody.
    assert!(!devices.service(|slot, _| panic!("seat {slot} was interrupted")));
}

/// A queue fuller than the budget is served over as many looks as it
/// takes, and the table says to come back until it is done.
#[test]
fn a_full_queue_is_served_over_several_looks() {
    let (_, owner) = owners();
    let mut devices = Devices::<Owner>::new();
    let mut driver = program(&mut devices, 0, owner(), 64);
    let commands = SERVICE_BUDGET.get() + 4;
    let asked: Vec<_> = (0..commands).map(|_| driver.ask()).collect();

    assert!(devices.service(|_, _| {}), "the budget ran out");
    let mut looks = 1;
    while devices.service(|_, _| {}) {
        looks += 1;
        assert!(looks < 8, "the queue drains");
    }
    for asked in asked {
        driver.answer(asked);
    }
}

/// A device reset past its last namespace cannot be brought back: the
/// table drops it, the block says it needs a reset, and the driver opens a
/// new one.
#[test]
fn a_device_that_cannot_be_reset_is_dropped() {
    let (dropped, owner) = owners();
    let mut devices = Devices::<Owner>::new();
    let registers = page();
    assert!(devices.seat(0, 1, registers));
    devices.open_device(0, DEVICE_ID).unwrap();
    devices.lend(0, window(PAGE_BYTES), owner()).unwrap();

    let status = |value| registers.store_u32(regs::DRIVER_STATUS, value).unwrap();
    for _ in 1..u32::from(u16::MAX) {
        status(STATUS_ACKNOWLEDGE);
        devices.service(|_, _| {});
        status(0);
        devices.service(|_, _| {});
    }
    assert_eq!(devices.open(), 1);
    assert_eq!(registers.load_u32(regs::DEVICE_STATUS), Ok(0));

    status(STATUS_ACKNOWLEDGE);
    devices.service(|_, _| {});
    status(0);
    let mut told = 0;
    devices.service(|_, _| told += 1);
    assert_eq!(told, 1, "the driver hears of it");
    assert_eq!(
        registers.load_u32(regs::DEVICE_STATUS),
        Ok(STATUS_DEVICE_NEEDS_RESET)
    );
    assert_eq!(devices.open(), 0, "the dead device is gone");
    assert_eq!(dropped.get(), 1, "and its memory went back");

    assert_eq!(devices.close_device(0), Err(Refusal::Busy));
    devices.open_device(0, DEVICE_ID).unwrap();
    assert_eq!(registers.load_u32(regs::DEVICE_STATUS), Ok(0));
    registers
        .store_u64(regs::DRIVER_FEATURES, regs::F_VERSION_1)
        .unwrap();
    registers.store_u32(regs::QUEUE_NUM, 16).unwrap();
    registers.store_u32(regs::QUEUE_READY, 1).unwrap();
    status(LIVE);
    devices.service(|_, _| {});
    assert_ne!(
        registers.load_u32(regs::DEVICE_STATUS).unwrap() & STATUS_DEVICE_NEEDS_RESET,
        0,
        "a new device, which was lent nothing"
    );
}

/// The row's founded heap covers every device the table will open, with
/// room over for what is not buffers. The two numbers live in two files;
/// this is where they are held together.
#[test]
fn the_founded_heap_covers_every_open_device() {
    let row = include_str!("../../boot.toml");
    let value = row
        .lines()
        .find_map(|line| line.strip_prefix("heap_supply_bytes"))
        .and_then(|rest| rest.split('=').nth(1))
        .map(|value| value.split('#').next().unwrap().trim().replace('_', ""))
        .expect("the row sizes its heap");
    let heap = u64::from_str_radix(value.trim_start_matches("0x"), 16).expect("a hex size");
    // The engines' tables, the seat table and the runtime's own boxes.
    const OVERHEAD: u64 = 2 << 20;
    assert!(
        heap >= DEVICE_HEAP_BYTES + OVERHEAD,
        "{heap:#x} does not cover {MAX_DEVICES} devices of buffers and the overhead"
    );
    // One lease, mapped at one stride of the founder's heap band.
    assert!(heap <= 16 << 20);
}
