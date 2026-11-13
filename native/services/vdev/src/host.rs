//! **The devices the service hosts.**
//!
//! Every driver face has a *seat*: the register page of that face, and the
//! device the driver opened over it, if it has opened one. A seat costs a
//! few words, so there is one for every program that holds a face. A device
//! is a model with its own state on the service's heap, so only a few are
//! open at once ([`MAX_DEVICES`]).
//!
//! The table is plain data. What the runtime supplies comes in as
//! arguments: the register window when a face is admitted, the mapped
//! window and its owner when a driver lends memory, and a closure that
//! wakes a driver when its device interrupts. That is what lets the whole
//! of it run off metal.

use alloc::vec::Vec;
use core::num::NonZeroU32;

use kore_vdev::mem::{Memory, Window};
use kore_vdev::regs;
use kore_vdev::transport::Transport;
use kore_vdev_accel::device::{AccelDevice, DEVICE_ID, MAX_BUFFER_BACKING_BYTES};

/// Devices open at once.
///
/// A device may hold [`MAX_BUFFER_BACKING_BYTES`] of buffers for its driver
/// on the service's heap, and the heap is one founded lease
/// (`heap_supply_bytes` in the service's boot manifest row). Four devices
/// are 8 MiB of buffers at most, which that lease covers with room for the
/// engines' own tables; a fifth driver is told the service is full and may
/// ask again when another closes.
pub const MAX_DEVICES: usize = 4;

/// The heap the open devices can ask for between them. The boot manifest
/// row's `heap_supply_bytes` must cover it; the service's tests hold the
/// two together.
pub const DEVICE_HEAP_BYTES: u64 = MAX_DEVICES as u64 * MAX_BUFFER_BACKING_BYTES;

/// Seats there can ever be: the runtime names a peer's slot in eight bits.
pub const MAX_SEATS: usize = 1 << u8::BITS;

/// Chains one device is served per look. A driver has at most thirty-two
/// commands in flight (half the largest queue), so sixteen is two looks for
/// the fullest queue, and no driver keeps the service from the next for
/// longer than sixteen commands take.
pub const SERVICE_BUDGET: NonZeroU32 = NonZeroU32::new(16).unwrap();

/// The page size a lent window is counted in on the wire.
pub const PAGE_BYTES: u64 = 0x1000;

/// Why a driver's request was not done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No device model has that id.
    NoDevice,
    /// The device is not in a state for this: already open, not open, or
    /// holding a queue in the memory it has.
    Busy,
    /// Every device the service hosts at once is taken.
    Limit,
    /// The model would not start.
    Unavailable,
    /// What was lent is not memory a device can address.
    Memory,
}

/// One open device. The transport is declared first so that it is dropped
/// first: every view into the lent memory goes before the owner that keeps
/// that memory mapped.
#[derive(Debug)]
struct Hosted<O> {
    transport: Transport<AccelDevice, Window>,
    /// What keeps the lent memory mapped, for as long as the device has it.
    lent: Option<O>,
}

/// One driver face.
#[derive(Debug)]
struct Seat<O> {
    /// The runtime's name for this occupant of the slot: a later occupant
    /// has another, so a request from one is never taken for the other's.
    arrival: u16,
    registers: Window,
    device: Option<Hosted<O>>,
}

/// Every seat, by the runtime's slot for the face. `O` is whatever keeps a
/// lent window mapped; dropping it gives the window back.
#[derive(Debug)]
pub struct Devices<O> {
    seats: Vec<Option<Seat<O>>>,
    open: usize,
}

impl<O> Default for Devices<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O> Devices<O> {
    pub const fn new() -> Self {
        Self {
            seats: Vec::new(),
            open: 0,
        }
    }

    /// Devices open now.
    pub const fn open(&self) -> usize {
        self.open
    }

    /// Whether `slot` is seated.
    pub fn seated(&self, slot: usize) -> bool {
        self.seats.get(slot).is_some_and(Option::is_some)
    }

    /// The arrival seated in `slot`.
    pub fn arrival(&self, slot: usize) -> Option<u16> {
        self.seats.get(slot)?.as_ref().map(|seat| seat.arrival)
    }

    /// Seat the face that arrived as `arrival` in `slot`, over its register
    /// page. `false` when the slot is taken, is not one the runtime can
    /// name, or the table could not grow to hold it.
    ///
    /// # Safety of the window
    /// `registers` stays mapped until [`unseat`](Self::unseat) returns for
    /// this slot: the caller owns the mapping and asks before it unmaps.
    pub fn seat(&mut self, slot: usize, arrival: u16, registers: Window) -> bool {
        if slot >= MAX_SEATS || registers.len() < regs::BYTES || self.seated(slot) {
            return false;
        }
        if slot >= self.seats.len() {
            let more = slot + 1 - self.seats.len();
            if self.seats.try_reserve(more).is_err() {
                return false;
            }
            self.seats.resize_with(slot + 1, || None);
        }
        // A page that held another face's device says nothing of it.
        let _ = registers.store_u32(regs::MAGIC, 0);
        self.seats[slot] = Some(Seat {
            arrival,
            registers,
            device: None,
        });
        true
    }

    /// The face in `slot` is going: close its device and forget its
    /// register page. After this nothing here touches either window.
    pub fn unseat(&mut self, slot: usize) {
        if let Some(seat) = self.seats.get_mut(slot).and_then(Option::take)
            && seat.device.is_some()
        {
            self.open -= 1;
        }
    }

    fn seat_mut(&mut self, slot: usize) -> Option<&mut Seat<O>> {
        self.seats.get_mut(slot).and_then(Option::as_mut)
    }

    /// Open the device model `device` for the face in `slot`: its register
    /// block is written before this returns.
    pub fn open_device(&mut self, slot: usize, device: u32) -> Result<(), Refusal> {
        let full = self.open >= MAX_DEVICES;
        let seat = self.seat_mut(slot).ok_or(Refusal::Unavailable)?;
        if device != DEVICE_ID {
            return Err(Refusal::NoDevice);
        }
        if seat.device.is_some() {
            return Err(Refusal::Busy);
        }
        if full {
            return Err(Refusal::Limit);
        }
        let model = AccelDevice::new().map_err(|_| Refusal::Unavailable)?;
        let transport = Transport::new(seat.registers, model).map_err(|_| Refusal::Unavailable)?;
        seat.device = Some(Hosted {
            transport,
            lent: None,
        });
        self.open += 1;
        Ok(())
    }

    /// The driver in `slot` lends `memory`, which `owner` keeps mapped.
    /// Answers the pages lent. On a refusal `owner` is dropped, which gives
    /// the memory back; so is the owner of any memory this replaces.
    pub fn lend(&mut self, slot: usize, memory: Window, owner: O) -> Result<u32, Refusal> {
        let hosted = self
            .seat_mut(slot)
            .and_then(|seat| seat.device.as_mut())
            .ok_or(Refusal::Busy)?;
        let pages = memory.len() / PAGE_BYTES;
        if pages == 0 || memory.len() % PAGE_BYTES != 0 {
            return Err(Refusal::Memory);
        }
        let pages = u32::try_from(pages).map_err(|_| Refusal::Memory)?;
        // The view the device held goes inside `lend`; its owner after.
        hosted.transport.lend(memory).map_err(|_| Refusal::Busy)?;
        hosted.lent = Some(owner);
        Ok(pages)
    }

    /// The driver in `slot` gives its device up. The memory it lent goes
    /// back, and the register block stops saying a device is there.
    pub fn close_device(&mut self, slot: usize) -> Result<(), Refusal> {
        let seat = self.seat_mut(slot).ok_or(Refusal::Busy)?;
        if seat.device.take().is_none() {
            return Err(Refusal::Busy);
        }
        let _ = seat.registers.store_u32(regs::MAGIC, 0);
        self.open -= 1;
        Ok(())
    }

    /// A doorbell: look at every open device. `interrupt` is called with the
    /// slot and register window of each device that raised one, after its
    /// interrupt count moved. Answers whether any device has chains waiting
    /// that its budget did not reach, in which case the caller comes back
    /// without waiting for another doorbell.
    ///
    /// A doorbell names no device, so each open one is looked at: at most
    /// [`MAX_DEVICES`], and a look at an idle device is three loads.
    pub fn service(&mut self, mut interrupt: impl FnMut(usize, &Window)) -> bool {
        let mut pending = false;
        let mut left = self.open;
        for (slot, seat) in self.seats.iter_mut().enumerate() {
            if left == 0 {
                break;
            }
            let Some(seat) = seat else { continue };
            let Some(hosted) = seat.device.as_mut() else {
                continue;
            };
            left -= 1;
            let serviced = hosted.transport.service(SERVICE_BUDGET);
            let dead = hosted.transport.is_dead();
            if serviced.interrupt {
                interrupt(slot, &seat.registers);
            }
            pending |= serviced.pending;
            if dead {
                // A model that could not be reset is not brought back. The
                // block keeps the status that says so; the driver opens
                // again for a new device.
                seat.device = None;
                self.open -= 1;
            }
        }
        pending
    }
}

#[cfg(test)]
mod tests;
