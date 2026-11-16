//! **The transport**: one driver's path to one device model.
//!
//! It owns the three things between them: the register block
//! ([`regs`](crate::regs)), the queue in the driver's memory
//! ([`ring`](crate::ring)) and the model ([`VirtualDevice`]). The host
//! calls [`Transport::service`] when the driver rings its doorbell and
//! raises the driver's interrupt when the answer says to. That is the whole
//! of what a host does, so a host is a loop around two notices.
//!
//! The register block is there from the start, so a driver can find out
//! what the device is before it commits memory to it. The driver's memory
//! arrives when the driver lends it ([`Transport::lend`]), which it must
//! have done by the time it says the queue is ready.
//!
//! # What one service does
//!
//! 1. Reads the status the driver has written. Zero is a reset: the queue
//!    takes a new epoch, the model forgets everything, and the status the
//!    device publishes returns to zero.
//! 2. Otherwise accepts each step the driver has reached that the device
//!    has not, in Virtio's order and only in that order: `ACKNOWLEDGE`,
//!    `DRIVER`, `FEATURES_OK` if the features the driver wrote are ones the
//!    device offered, `DRIVER_OK` if the queue the driver described is one
//!    the device can serve.
//! 3. Once live, takes each chain the driver published, hands it to the
//!    model and returns it used.
//!
//! A status write that changes what the driver has announced is always
//! answered with an interrupt, whether or not the device accepted the step:
//! a driver writes its status, rings, waits, and then reads what the device
//! made of it. A refusal is the device's status not carrying the step,
//! never silence.
//!
//! Writing the status the device last saw announces nothing and is not
//! answered, zero included: a device in its reset state that is told to
//! reset has nothing to do. That matters because a look is not always the
//! driver's doing. Whoever hosts the device looks whenever anything wakes
//! it, another driver's doorbell among them, and a device that answered
//! every look in its reset state would interrupt a driver that had asked
//! for nothing.
//!
//! # What the driver cannot do to the device
//!
//! The device keeps its own copy of every decision. It never reads back a
//! register it wrote, it reads the driver's registers once per step and
//! acts on what it read, and a feature set or a queue it accepted stays
//! what it accepted whatever the driver writes afterwards. A driver that
//! describes a queue the device cannot serve, or breaks the queue once it
//! is live, gets [`STATUS_DEVICE_NEEDS_RESET`] and nothing more until it
//! resets; a chain that is merely malformed is one chain's failure, returned
//! used with a length of zero.
//!
//! One driver cannot hold the host either: a service takes at most the
//! budget of chains it is given, a second look at the queue counting as
//! one, and says whether more may be waiting.

use core::num::NonZeroU32;

use virtio_accel_transport::{
    ChainError, DeviceChain, DeviceQueue, NotificationHint, NotificationRecheck, QueueControl,
    QueuePort, QueueSize, UsedLength,
};

use crate::device::VirtualDevice;
use crate::mem::Memory;
use crate::regs::{
    self, F_VERSION_1, STATUS_ACKNOWLEDGE, STATUS_DEVICE_NEEDS_RESET, STATUS_DRIVER,
    STATUS_DRIVER_OK, STATUS_FAILED, STATUS_FEATURES_OK,
};
use crate::ring::{DeviceRing, RingAddresses, limits_fit};

/// The features every device here offers. The profile has no optional ring
/// feature (no indirect tables, no event indices, no packed ring), so the
/// offer is the one bit that says the device is not a legacy one.
pub const OFFERED_FEATURES: u64 = F_VERSION_1;

/// Why a transport could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachError {
    /// The register window is not the page the block needs.
    Registers,
    /// The model's configuration does not fit the block's configuration
    /// area.
    Config,
    /// The model's queue limits are not ones the ring can hold.
    Queue,
}

/// The driver's memory changes only while the device is in its first state:
/// a device that has accepted any step may hold a queue in the memory it
/// has, and is reset before that memory is taken from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Busy;

/// What one [`Transport::service`] left for the host to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Serviced {
    /// Raise the driver's interrupt: the device answered a status write,
    /// changed its own status, or used a chain the driver asked to hear
    /// about. [`INTERRUPTS`](regs::INTERRUPTS) has already moved; the host
    /// wakes whoever is parked on it.
    pub interrupt: bool,
    /// The budget ran out with chains possibly still waiting. The host
    /// calls again without waiting for a doorbell: the driver was told not
    /// to ring one.
    pub pending: bool,
}

/// Something under the transport stopped holding: a register that would not
/// load, the queue refusing a step it had validated, or the model using
/// more than it was given. Each ends in [`STATUS_DEVICE_NEEDS_RESET`].
struct Broken;

/// One driver's path to one device model.
#[derive(Debug)]
pub struct Transport<D, M> {
    registers: M,
    /// The queue, once the driver has lent the memory it is in.
    ring: Option<DeviceRing<M>>,
    max_queue: QueueSize,
    device: D,
    /// The status the device has accepted and published.
    status: u32,
    /// The driver's status as last read: a doorbell that finds another
    /// value is announcing a status write.
    announced: u32,
    /// The features accepted at `FEATURES_OK`.
    features: u64,
    /// A reset failed: the status stays `DEVICE_NEEDS_RESET` for good.
    dead: bool,
    /// Interrupts raised, as [`regs::INTERRUPTS`] publishes them.
    interrupts: u32,
}

impl<D: VirtualDevice, M: Memory + Clone + core::fmt::Debug> Transport<D, M> {
    /// A transport for `device`, with the register block in `registers`.
    /// Writes the device's half of the block: the driver may map and read
    /// it as soon as this returns.
    pub fn new(registers: M, device: D) -> Result<Self, AttachError> {
        if registers.len() < regs::BYTES {
            return Err(AttachError::Registers);
        }
        let config = device.config();
        if config.len() as u64 > regs::BYTES - regs::CONFIG {
            return Err(AttachError::Config);
        }
        let max_queue = QueueSize::new(D::MAX_QUEUE_SIZE)
            .filter(|size| limits_fit(*size, D::MAX_CHAIN_DESCRIPTORS))
            .ok_or(AttachError::Queue)?;

        let block = |at, value| registers.store_u32(at, value);
        // The length was checked above, and every offset is a constant of
        // the layout, aligned for its word: a fault here is a window that
        // does not keep `Memory`'s contract.
        let written = registers
            .write(regs::CONFIG, config)
            .and_then(|()| block(regs::CONFIG_BYTES, config.len() as u32))
            .and_then(|()| block(regs::CONFIG_GENERATION, 0))
            .and_then(|()| block(regs::QUEUE_NUM_MAX, u32::from(D::MAX_QUEUE_SIZE)))
            .and_then(|()| block(regs::DEVICE_STATUS, 0))
            .and_then(|()| block(regs::INTERRUPTS, 0))
            .and_then(|()| registers.store_u64(regs::DEVICE_FEATURES, OFFERED_FEATURES))
            .and_then(|()| block(regs::VENDOR_ID, regs::VENDOR_KORE))
            .and_then(|()| block(regs::DEVICE_ID, D::DEVICE_ID))
            .and_then(|()| block(regs::VERSION, regs::VERSION_VALUE))
            // Last: a driver that sees the magic sees the rest.
            .and_then(|()| block(regs::MAGIC, regs::MAGIC_VALUE));
        if written.is_err() {
            return Err(AttachError::Registers);
        }
        Ok(Self {
            registers,
            ring: None,
            max_queue,
            device,
            status: 0,
            announced: 0,
            features: 0,
            dead: false,
            interrupts: 0,
        })
    }

    /// The driver lends `memory`: device addresses are offsets into it from
    /// here on. Answers the memory it replaces, if the driver had lent some
    /// before, which the host may now unmap.
    pub fn lend(&mut self, memory: M) -> Result<Option<M>, Busy> {
        if self.status != 0 {
            return Err(Busy);
        }
        let before = self.withdraw()?;
        // The limits were checked when the transport was made.
        self.ring = DeviceRing::new(memory, self.max_queue, D::MAX_CHAIN_DESCRIPTORS);
        Ok(before)
    }

    /// Take back the memory the driver lent, leaving the device with none.
    pub fn withdraw(&mut self) -> Result<Option<M>, Busy> {
        if self.status != 0 {
            return Err(Busy);
        }
        Ok(self.ring.take().map(DeviceRing::into_memory))
    }

    /// Whether the driver has lent memory.
    pub const fn has_memory(&self) -> bool {
        self.ring.is_some()
    }

    /// The status the device has accepted (`STATUS_*`).
    pub const fn status(&self) -> u32 {
        self.status
    }

    /// The features the driver and device agreed on; zero before
    /// `FEATURES_OK`.
    pub const fn features(&self) -> u64 {
        self.features
    }

    /// A reset could not bring the device back. The host drops this
    /// transport; a driver that attaches again gets a new one.
    pub const fn is_dead(&self) -> bool {
        self.dead
    }

    /// The model.
    pub const fn device(&self) -> &D {
        &self.device
    }

    /// The driver rang its doorbell, or the last service left chains
    /// pending: look at what the driver wrote and act on it, taking at most
    /// `budget` chains.
    pub fn service(&mut self, budget: NonZeroU32) -> Serviced {
        let before = self.status;
        let mut serviced = Serviced::default();
        match self.step(budget) {
            Ok(done) => serviced = done,
            // Told to the driver below, as the status changing.
            Err(Broken) => self.status |= STATUS_DEVICE_NEEDS_RESET,
        }
        if self.status != before {
            // Nothing is left to try if the status itself will not store:
            // the driver sees the old one, and the device still refuses to
            // serve past what it accepted.
            let _ = self.registers.store_u32(regs::DEVICE_STATUS, self.status);
            serviced.interrupt = true;
        }
        if serviced.interrupt {
            // After everything the interrupt is about: a driver woken by
            // this word finds the status and the used ring already written.
            self.interrupts = self.interrupts.wrapping_add(1);
            let _ = self.registers.store_u32(regs::INTERRUPTS, self.interrupts);
        }
        serviced
    }

    fn step(&mut self, budget: NonZeroU32) -> Result<Serviced, Broken> {
        let driver = self.driver_u32(regs::DRIVER_STATUS)?;
        // Only a status the device has not seen is an announcement. A look
        // that finds the same one, a device in its reset state included,
        // answers nothing: the look may be another driver's doing.
        let answer = Serviced {
            interrupt: driver != self.announced,
            pending: false,
        };
        self.announced = driver;
        if driver == 0 {
            self.reset();
            return Ok(answer);
        }
        self.advance(driver)?;
        if !self.live() {
            return Ok(answer);
        }
        let served = self.serve(budget)?;
        Ok(Serviced {
            interrupt: answer.interrupt || served.interrupt,
            pending: served.pending,
        })
    }

    fn driver_u32(&self, at: u64) -> Result<u32, Broken> {
        self.registers.load_u32(at).map_err(|_| Broken)
    }

    fn driver_u64(&self, at: u64) -> Result<u64, Broken> {
        self.registers.load_u64(at).map_err(|_| Broken)
    }

    /// Whether the device takes chains: the driver finished bring-up and
    /// neither side has given up.
    const fn live(&self) -> bool {
        self.status & (STATUS_DRIVER_OK | STATUS_FAILED | STATUS_DEVICE_NEEDS_RESET)
            == STATUS_DRIVER_OK
    }

    /// The driver wrote zero. Nothing accepted means nothing to undo, which
    /// is also what makes a reset idempotent.
    fn reset(&mut self) {
        if self.status == 0 || self.dead {
            return;
        }
        // The queue first: from the new epoch on nothing of the old one is
        // taken, written or completed, so the model is torn down with no
        // chain in its hands and the driver may reuse its memory once it
        // reads the status as zero.
        let queue = self.ring.as_mut().is_none_or(|ring| {
            ring.state()
                .epoch()
                .checked_next()
                .is_some_and(|epoch| ring.reset(epoch).is_ok())
        });
        let model = self.device.reset().is_ok();
        self.features = 0;
        if queue && model {
            self.status = 0;
        } else {
            self.dead = true;
            self.status = STATUS_DEVICE_NEEDS_RESET;
        }
    }

    /// Accept each step `driver` has reached that the device has not, in
    /// order. A step whose predecessor is missing is not accepted, and a
    /// step once accepted is not revisited: only a reset takes it back.
    fn advance(&mut self, driver: u32) -> Result<(), Broken> {
        if self.status & (STATUS_DEVICE_NEEDS_RESET | STATUS_FAILED) != 0 {
            return Ok(());
        }
        if driver & STATUS_FAILED != 0 {
            // The driver gave up: the device stops with it.
            self.status |= STATUS_FAILED;
            return Ok(());
        }
        let reached = |status: u32, step: u32, after: u32| {
            driver & step != 0 && status & step == 0 && status & after == after
        };
        if reached(self.status, STATUS_ACKNOWLEDGE, 0) {
            self.status |= STATUS_ACKNOWLEDGE;
        }
        if reached(self.status, STATUS_DRIVER, STATUS_ACKNOWLEDGE) {
            self.status |= STATUS_DRIVER;
        }
        let named = STATUS_ACKNOWLEDGE | STATUS_DRIVER;
        if reached(self.status, STATUS_FEATURES_OK, named) {
            let accepted = self.driver_u64(regs::DRIVER_FEATURES)?;
            // A subset of the offer that keeps the one bit a driver must
            // accept. Anything else leaves `FEATURES_OK` unset, which is
            // how Virtio tells a driver its choice was refused.
            if accepted & !OFFERED_FEATURES == 0 && accepted & F_VERSION_1 != 0 {
                self.features = accepted;
                self.status |= STATUS_FEATURES_OK;
            }
        }
        if reached(self.status, STATUS_DRIVER_OK, named | STATUS_FEATURES_OK) {
            if self.bring_up_queue()? {
                self.status |= STATUS_DRIVER_OK;
            } else {
                // The driver said it was done and described a queue that
                // cannot be served. There is no later step at which it
                // could be put right.
                self.status |= STATUS_DEVICE_NEEDS_RESET;
            }
        }
        Ok(())
    }

    /// Take the queue the driver described. `false` when it is not one the
    /// device can serve: not ready, a size that is not a power of two the
    /// model takes, parts outside the driver's memory, or no memory lent
    /// for it to be in.
    fn bring_up_queue(&mut self) -> Result<bool, Broken> {
        if self.driver_u32(regs::QUEUE_READY)? == 0 {
            return Ok(false);
        }
        let Some(size) = u16::try_from(self.driver_u32(regs::QUEUE_NUM)?)
            .ok()
            .and_then(QueueSize::new)
        else {
            return Ok(false);
        };
        let addresses = RingAddresses {
            descriptors: self.driver_u64(regs::QUEUE_DESC)?,
            available: self.driver_u64(regs::QUEUE_DRIVER)?,
            used: self.driver_u64(regs::QUEUE_DEVICE)?,
        };
        let Some(ring) = self.ring.as_mut() else {
            return Ok(false);
        };
        Ok(ring.configure(size).is_ok()
            && ring.set_addresses(addresses).is_ok()
            && ring.set_ready(true).is_ok())
    }

    /// Take what the driver published, at most `budget` chains.
    fn serve(&mut self, budget: NonZeroU32) -> Result<Serviced, Broken> {
        let mut serviced = Serviced::default();
        let mut left = budget.get();
        // Live means the queue was brought up, which took memory.
        let Some(ring) = self.ring.as_mut() else {
            return Err(Broken);
        };
        // The driver need not ring again while the device is looking.
        ring.disable_available_notifications().map_err(|_| Broken)?;
        loop {
            while left != 0 {
                let Some(mut chain) = ring.pop_available().map_err(|_| Broken)? else {
                    break;
                };
                left -= 1;
                let used = match chain.io() {
                    Ok(io) => {
                        let (regions, request, response) = io.into_parts();
                        self.device.command(regions, request, response)
                    }
                    // One chain's failure: returned used, nothing written.
                    Err(ChainError::Malformed(_)) => 0,
                    // No reset runs inside a service, and the chain's bytes
                    // were checked when it was taken.
                    Err(ChainError::ResetRace { .. } | ChainError::Transport(_)) => {
                        return Err(Broken);
                    }
                };
                let hint = ring
                    .complete(chain, UsedLength::new(used))
                    .map_err(|_| Broken)?;
                serviced.interrupt |= hint == NotificationHint::Notify;
                // The command is answered before its work runs: a driver
                // that submits is told so, and then sees it complete.
                self.device.drive();
                if self.device.needs_reset() {
                    self.status |= STATUS_DEVICE_NEEDS_RESET;
                    return Ok(serviced);
                }
            }
            if left == 0 {
                // Notifications stay off: the host comes back by itself.
                serviced.pending = true;
                return Ok(serviced);
            }
            match ring.enable_available_notifications().map_err(|_| Broken)? {
                NotificationRecheck::Idle => return Ok(serviced),
                // Published while the device was not listening.
                NotificationRecheck::WorkPending => {
                    ring.disable_available_notifications().map_err(|_| Broken)?;
                    // A look costs what a chain costs. A driver that shows
                    // an index and takes it back before the device reads
                    // the entry would otherwise keep the device looking,
                    // at no cost to the budget, for as long as it liked.
                    left -= 1;
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
