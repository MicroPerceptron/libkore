//! **The driver's half**: the same registers and the same queue, from the
//! side that owns the memory.
//!
//! A driver written in C does these steps by hand against the layout in
//! [`regs`]. This is those steps in Rust, behind the driver-side queue
//! contract of `virtio-accel-transport` ([`DriverQueue`]), so the protocol's
//! reference guest client runs over a kore virtual device unchanged.
//!
//! # The memory a driver lends
//!
//! One window, laid out by [`Layout`]: the descriptor table, the available
//! ring and the used ring, then a fixed pool of command slots. A slot is two
//! descriptors and the bytes they name, room for one request the device
//! reads and one response it writes. A command is therefore always a chain
//! of two, the shortest the protocol allows, and a driver has as many
//! commands in flight as it has slots.
//!
//! A [`DriverChain`] is one slot, sized for one command. It is taken from
//! the pool ([`Slots::chain`]), filled, published on the ring, taken back
//! used and returned ([`Slots::recycle`]). It is not `Clone`: a slot has
//! one holder at a time, the driver or the device. The pool is apart from
//! the ring because the protocol's client owns the queue it is given and is
//! handed each chain by its caller.
//!
//! # Reset
//!
//! A driver that wants the device back in its first state calls
//! [`bring_up`] again and drops what the last one gave it. The reset is
//! the first step of a bring-up, and the rings are cleared after the device
//! has acknowledged it and before the device is told the queue is ready, so
//! neither side ever sees the other's indices from before.
//!
//! # What the driver trusts
//!
//! The device is the more trusted side, but a used entry is still checked
//! before it is believed: it must name a chain the driver published and has
//! not yet taken back, and a length no longer than the room that chain gave.

use core::sync::atomic::{Ordering, fence};

use virtio_accel_transport::{
    ChainId, DriverChainBuffer, DriverQueue, NotificationHint, NotificationRecheck, PublishError,
    PublishErrorKind, PublishedChain, QueueConfigError, QueueEpoch, QueueError, QueuePort,
    QueueSize, QueueState, ReclaimedChain, UsedChain, UsedLength,
};

use crate::mem::{Memory, MemoryFault};
use crate::regs::{
    self, F_VERSION_1, STATUS_ACKNOWLEDGE, STATUS_DEVICE_NEEDS_RESET, STATUS_DRIVER,
    STATUS_DRIVER_OK, STATUS_FAILED, STATUS_FEATURES_OK,
};
use crate::ring::{
    AVAIL_ALIGN, AVAIL_ENTRY, DESC_BYTES, MAX_QUEUE_SIZE, RING_FLAGS, RING_HEADER, RING_IDX,
    RingAddresses, USED_ALIGN, USED_ENTRY, USED_LEN, VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_DESC_F_NEXT,
    VIRTQ_DESC_F_WRITE, VIRTQ_USED_F_NO_NOTIFY,
};

/// Descriptors one command slot takes: its request and its response.
const SLOT_DESCRIPTORS: u16 = 2;

/// The most command slots a queue has: half the largest queue.
pub const MAX_SLOTS: usize = (MAX_QUEUE_SIZE / SLOT_DESCRIPTORS) as usize;

/// A slot's bytes start on a word, so a frame's fields are aligned for a
/// driver that reads them in place.
const SLOT_ALIGN: u64 = 8;

/// What the device's half of the register block says it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The device model ([`regs::DEVICE_ID`]).
    pub device_id: u32,
    /// Who made it ([`regs::VENDOR_ID`]).
    pub vendor_id: u32,
    /// The features it offers.
    pub features: u64,
    /// The largest queue it takes.
    pub queue_num_max: u32,
    /// Bytes of device configuration at [`regs::CONFIG`].
    pub config_bytes: u32,
}

/// Why a device could not be brought up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BringUpError {
    /// The block does not begin with [`regs::MAGIC_VALUE`].
    NotADevice,
    /// The block's layout is a version this driver does not know.
    Version(u32),
    /// The device did not return to status zero when asked to reset.
    Reset,
    /// The device does not offer [`F_VERSION_1`], or refused it.
    Features,
    /// The layout does not fit: the queue is larger than the device takes,
    /// or the memory is smaller than the layout.
    Layout,
    /// The device refused the queue.
    Queue,
    /// A register or the driver's own memory did not answer.
    Memory(MemoryFault),
}

impl From<MemoryFault> for BringUpError {
    fn from(fault: MemoryFault) -> Self {
        Self::Memory(fault)
    }
}

/// Read the device's identity, refusing a block that is not a kore virtual
/// device of a layout this driver knows.
pub fn identify<M: Memory>(registers: &M) -> Result<Identity, BringUpError> {
    if registers.load_u32(regs::MAGIC)? != regs::MAGIC_VALUE {
        return Err(BringUpError::NotADevice);
    }
    let version = registers.load_u32(regs::VERSION)?;
    if version != regs::VERSION_VALUE {
        return Err(BringUpError::Version(version));
    }
    Ok(Identity {
        device_id: registers.load_u32(regs::DEVICE_ID)?,
        vendor_id: registers.load_u32(regs::VENDOR_ID)?,
        features: registers.load_u64(regs::DEVICE_FEATURES)?,
        queue_num_max: registers.load_u32(regs::QUEUE_NUM_MAX)?,
        config_bytes: registers.load_u32(regs::CONFIG_BYTES)?,
    })
}

/// Read the first `target.len()` bytes of the device configuration.
pub fn read_config<M: Memory>(registers: &M, target: &mut [u8]) -> Result<(), BringUpError> {
    if target.len() as u64 > u64::from(registers.load_u32(regs::CONFIG_BYTES)?) {
        return Err(BringUpError::Layout);
    }
    Ok(registers.read(regs::CONFIG, target)?)
}

/// How a driver lays out the memory it lends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    size: QueueSize,
    request_bytes: u32,
    response_bytes: u32,
}

impl Layout {
    /// A queue of `size` descriptors, `size / 2` command slots, each with
    /// room for a request of `request_bytes` and a response of
    /// `response_bytes`. `None` when the queue is larger than a ring holds
    /// or a slot has no room.
    pub const fn new(size: QueueSize, request_bytes: u32, response_bytes: u32) -> Option<Self> {
        if size.get() > MAX_QUEUE_SIZE
            || size.get() < SLOT_DESCRIPTORS
            || request_bytes == 0
            || response_bytes == 0
        {
            return None;
        }
        Some(Self {
            size,
            request_bytes,
            response_bytes,
        })
    }

    /// The queue's size.
    pub const fn size(&self) -> QueueSize {
        self.size
    }

    /// Command slots: commands the driver can have in flight.
    pub const fn slots(&self) -> u16 {
        self.size.get() / SLOT_DESCRIPTORS
    }

    /// Where the queue's three parts are.
    pub const fn addresses(&self) -> RingAddresses {
        let size = self.size.get() as u64;
        let available = (size * DESC_BYTES).next_multiple_of(AVAIL_ALIGN);
        let used = (available + RING_HEADER + size * AVAIL_ENTRY).next_multiple_of(USED_ALIGN);
        RingAddresses {
            descriptors: 0,
            available,
            used,
        }
    }

    const fn slot_bytes(&self) -> u64 {
        (self.request_bytes as u64).next_multiple_of(SLOT_ALIGN)
            + (self.response_bytes as u64).next_multiple_of(SLOT_ALIGN)
    }

    const fn slots_at(&self) -> u64 {
        let size = self.size.get() as u64;
        (self.addresses().used + RING_HEADER + size * USED_ENTRY).next_multiple_of(SLOT_ALIGN)
    }

    /// Where `slot`'s request and response bytes are.
    const fn slot(&self, slot: u16) -> (u64, u64) {
        let request = self.slots_at() + slot as u64 * self.slot_bytes();
        (
            request,
            request + (self.request_bytes as u64).next_multiple_of(SLOT_ALIGN),
        )
    }

    /// Bytes of memory the layout takes.
    pub const fn bytes(&self) -> u64 {
        self.slots_at() + self.slots() as u64 * self.slot_bytes()
    }
}

/// Bring a device up: reset it, agree features, describe the queue and tell
/// it the driver is ready. `kick` rings the doorbell and waits for the
/// interrupt that answers it.
///
/// The steps are Virtio 1.3 §3.1.1's. On a refusal the driver marks the
/// device [`STATUS_FAILED`], as that section requires, and says which step
/// was refused.
pub fn bring_up<M: Memory + Clone + core::fmt::Debug>(
    registers: &M,
    memory: M,
    layout: Layout,
    mut kick: impl FnMut(),
) -> Result<(DriverRing<M>, Slots<M>), BringUpError> {
    let identity = identify(registers)?;
    let status = |value| registers.store_u32(regs::DRIVER_STATUS, value);
    let answered = || registers.load_u32(regs::DEVICE_STATUS);
    let fail = |error| {
        // Best effort: the refusal is what the caller needs to hear.
        let _ = status(STATUS_FAILED);
        Err(error)
    };

    // A device in its reset state, which the driver has announced nothing
    // to, has nothing to undo and answers no reset: there is nothing to
    // wait for. Any other is reset first, and that is answered.
    if registers.load_u32(regs::DRIVER_STATUS)? != 0 || answered()? != 0 {
        status(0)?;
        kick();
        if answered()? != 0 {
            return Err(BringUpError::Reset);
        }
    }
    let mut reached = STATUS_ACKNOWLEDGE | STATUS_DRIVER;
    status(reached)?;

    if identity.features & F_VERSION_1 == 0 {
        return fail(BringUpError::Features);
    }
    registers.store_u64(regs::DRIVER_FEATURES, F_VERSION_1)?;
    reached |= STATUS_FEATURES_OK;
    status(reached)?;
    kick();
    if answered()? & STATUS_FEATURES_OK == 0 {
        return fail(BringUpError::Features);
    }

    if u32::from(layout.size.get()) > identity.queue_num_max || memory.len() < layout.bytes() {
        return fail(BringUpError::Layout);
    }
    let ring = DriverRing::new(memory.clone(), layout)?;
    let at = layout.addresses();
    registers.store_u32(regs::QUEUE_NUM, u32::from(layout.size.get()))?;
    registers.store_u64(regs::QUEUE_DESC, at.descriptors)?;
    registers.store_u64(regs::QUEUE_DRIVER, at.available)?;
    registers.store_u64(regs::QUEUE_DEVICE, at.used)?;
    registers.store_u32(regs::QUEUE_READY, 1)?;
    reached |= STATUS_DRIVER_OK;
    status(reached)?;
    kick();
    let device = answered()?;
    if device & STATUS_DRIVER_OK == 0 || device & STATUS_DEVICE_NEEDS_RESET != 0 {
        return fail(BringUpError::Queue);
    }
    Ok((ring, Slots::new(memory, layout)))
}

/// Why the driver's ring could not go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverRingError {
    /// A used entry names no chain the driver has with the device, or a
    /// length longer than the chain's room.
    UsedEntry,
    /// The driver's own memory did not answer.
    Memory(MemoryFault),
}

impl From<MemoryFault> for QueueError<DriverRingError> {
    fn from(fault: MemoryFault) -> Self {
        Self::Transport(DriverRingError::Memory(fault))
    }
}

/// One command slot, sized for one command: `readable` bytes the driver
/// writes and the device reads, `writable` bytes the device writes.
#[derive(Debug)]
pub struct DriverChain<M> {
    memory: M,
    slot: u16,
    request: u64,
    readable: u32,
    response: u64,
    writable: u32,
}

impl<M: Memory + core::fmt::Debug> DriverChainBuffer for DriverChain<M> {
    type Error = MemoryFault;

    fn device_readable_len(&self) -> u64 {
        u64::from(self.readable)
    }

    fn device_writable_len(&self) -> u64 {
        u64::from(self.writable)
    }

    fn write_device_readable(&mut self, offset: u64, source: &[u8]) -> Result<(), MemoryFault> {
        within(offset, source.len(), self.readable)?;
        self.memory.write(self.request + offset, source)
    }

    fn read_device_writable(&self, offset: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        within(offset, target.len(), self.writable)?;
        self.memory.read(self.response + offset, target)
    }
}

fn within(offset: u64, len: usize, room: u32) -> Result<(), MemoryFault> {
    if offset
        .checked_add(len as u64)
        .is_some_and(|end| end <= u64::from(room))
    {
        Ok(())
    } else {
        Err(MemoryFault::Range)
    }
}

/// The pool of command slots in the driver's memory.
#[derive(Debug)]
pub struct Slots<M> {
    memory: M,
    layout: Layout,
    /// Slots no chain holds, one bit each.
    free: u32,
}

impl<M: Memory + Clone> Slots<M> {
    fn new(memory: M, layout: Layout) -> Self {
        let slots = u32::from(layout.slots());
        Self {
            memory,
            layout,
            free: if slots == u32::BITS {
                u32::MAX
            } else {
                (1 << slots) - 1
            },
        }
    }

    /// Slots no chain holds.
    pub const fn free(&self) -> u32 {
        self.free.count_ones()
    }

    /// A free slot sized for a request of `readable` bytes and a response of
    /// up to `writable`. `None` when every slot is taken, or the sizes are
    /// zero or more than a slot has room for.
    pub fn chain(&mut self, readable: u32, writable: u32) -> Option<DriverChain<M>> {
        if readable == 0
            || writable == 0
            || readable > self.layout.request_bytes
            || writable > self.layout.response_bytes
            || self.free == 0
        {
            return None;
        }
        let slot = self.free.trailing_zeros() as u16;
        self.free &= !(1 << slot);
        let (request, response) = self.layout.slot(slot);
        Some(DriverChain {
            memory: self.memory.clone(),
            slot,
            request,
            readable,
            response,
            writable,
        })
    }

    /// Return a chain's slot: one the device has used, or one that was
    /// never published.
    pub fn recycle(&mut self, chain: DriverChain<M>) {
        self.free |= 1 << chain.slot;
    }
}

/// The driver's half of one split virtqueue in its own memory.
#[derive(Debug)]
pub struct DriverRing<M> {
    memory: M,
    layout: Layout,
    state: QueueState,
    /// Available entries published, and used entries taken back.
    published: u16,
    taken: u16,
    /// The chains the device has, by slot.
    held: [Option<DriverChain<M>>; MAX_SLOTS],
}

impl<M: Memory + Clone + core::fmt::Debug> DriverRing<M> {
    /// A ready queue laid out by `layout` in `memory`, with both rings
    /// cleared: the memory may have held a queue before a reset.
    fn new(memory: M, layout: Layout) -> Result<Self, MemoryFault> {
        let at = layout.addresses();
        for ring in [at.available, at.used] {
            memory.store_u16(ring + RING_FLAGS, 0)?;
            memory.store_u16(ring + RING_IDX, 0)?;
        }
        Ok(Self {
            memory,
            layout,
            state: ready(layout.size, QueueEpoch::INITIAL),
            published: 0,
            taken: 0,
            held: [const { None }; MAX_SLOTS],
        })
    }

    /// The layout the queue was made with.
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    fn descriptor(
        &self,
        index: u16,
        at: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) -> Result<(), MemoryFault> {
        let mut bytes = [0u8; DESC_BYTES as usize];
        bytes[..8].copy_from_slice(&at.to_le_bytes());
        bytes[8..12].copy_from_slice(&len.to_le_bytes());
        bytes[12..14].copy_from_slice(&flags.to_le_bytes());
        bytes[14..].copy_from_slice(&next.to_le_bytes());
        self.memory.write(
            self.layout.addresses().descriptors + u64::from(index) * DESC_BYTES,
            &bytes,
        )
    }

    /// Write the chain's two descriptors and publish its head.
    fn expose(&mut self, chain: &DriverChain<M>) -> Result<NotificationHint, MemoryFault> {
        let at = self.layout.addresses();
        let head = chain.slot * SLOT_DESCRIPTORS;
        self.descriptor(
            head,
            chain.request,
            chain.readable,
            VIRTQ_DESC_F_NEXT,
            head + 1,
        )?;
        self.descriptor(
            head + 1,
            chain.response,
            chain.writable,
            VIRTQ_DESC_F_WRITE,
            0,
        )?;
        let entry = u64::from(self.published % self.layout.size.get());
        self.memory.write(
            at.available + RING_HEADER + entry * AVAIL_ENTRY,
            &head.to_le_bytes(),
        )?;
        self.published = self.published.wrapping_add(1);
        // Release: the request bytes, the descriptors and the entry are
        // visible to a device that has loaded this index.
        self.memory
            .store_u16(at.available + RING_IDX, self.published)?;
        // The index is published before the device's suppression flag is
        // read: a device that clears the flag and rechecks the index cannot
        // miss this chain.
        fence(Ordering::SeqCst);
        Ok(
            if self.memory.load_u16(at.used + RING_FLAGS)? & VIRTQ_USED_F_NO_NOTIFY == 0 {
                NotificationHint::Notify
            } else {
                NotificationHint::Suppressed
            },
        )
    }
}

fn ready(size: QueueSize, epoch: QueueEpoch) -> QueueState {
    // A configured size no larger than the maximum, which is itself: the
    // one shape `QueueState::new` refuses is not this one.
    QueueState::new(size, Some(size), true, epoch)
        .unwrap_or_else(|_| QueueState::unconfigured(size, epoch))
}

impl<M> QueuePort for DriverRing<M> {
    fn state(&self) -> QueueState {
        self.state
    }
}

impl<M: Memory + Clone + core::fmt::Debug> DriverQueue for DriverRing<M> {
    type Chain = DriverChain<M>;
    type Reclaimed = heapless::Vec<ReclaimedChain<DriverChain<M>>, MAX_SLOTS>;
    type Error = DriverRingError;

    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
        if !self.state.ready() {
            return Err(PublishError::new(chain, PublishErrorKind::NotReady));
        }
        let slot = usize::from(chain.slot);
        if self.held[slot].is_some() {
            // The slot's descriptors are the device's until it uses them.
            return Err(PublishError::new(chain, PublishErrorKind::QueueFull));
        }
        match self.expose(&chain) {
            Ok(hint) => {
                let id = ChainId::new(self.state.epoch(), u64::from(chain.slot * SLOT_DESCRIPTORS));
                self.held[slot] = Some(chain);
                Ok(PublishedChain::new(id, hint))
            }
            Err(fault) => Err(PublishError::new(
                chain,
                PublishErrorKind::Transport(DriverRingError::Memory(fault)),
            )),
        }
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
        if !self.state.ready() {
            return Err(QueueError::NotReady);
        }
        let at = self.layout.addresses();
        // Acquire: the entry and the response bytes the device wrote before
        // it published this index are visible to the reads below.
        if self.memory.load_u16(at.used + RING_IDX)? == self.taken {
            return Ok(None);
        }
        let entry =
            at.used + RING_HEADER + u64::from(self.taken % self.layout.size.get()) * USED_ENTRY;
        let head = self.memory.load_u32(entry)?;
        let len = self.memory.load_u32(entry + USED_LEN)?;
        self.taken = self.taken.wrapping_add(1);

        let slot = head / u32::from(SLOT_DESCRIPTORS);
        let chain = if head % u32::from(SLOT_DESCRIPTORS) == 0 {
            self.held
                .get_mut(slot as usize)
                .and_then(Option::take)
                .filter(|chain| len <= chain.writable)
        } else {
            None
        };
        let Some(chain) = chain else {
            return Err(QueueError::Transport(DriverRingError::UsedEntry));
        };
        Ok(Some(UsedChain::new(
            ChainId::new(self.state.epoch(), u64::from(head)),
            UsedLength::new(len),
            chain,
        )))
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.memory.store_u16(
            self.layout.addresses().available + RING_FLAGS,
            VIRTQ_AVAIL_F_NO_INTERRUPT,
        )?;
        Ok(())
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        let at = self.layout.addresses();
        self.memory.store_u16(at.available + RING_FLAGS, 0)?;
        // The flag is cleared before the index is read: a device that used
        // a chain while interrupts were off is seen here, and one that uses
        // a chain after sees the flag clear and interrupts.
        fence(Ordering::SeqCst);
        Ok(if self.memory.load_u16(at.used + RING_IDX)? == self.taken {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        if next_epoch <= self.state.epoch() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::NonIncreasingEpoch,
            ));
        }
        let epoch = self.state.epoch();
        let mut reclaimed = heapless::Vec::new();
        for (slot, held) in self.held.iter_mut().enumerate() {
            if let Some(chain) = held.take() {
                let head = slot as u64 * u64::from(SLOT_DESCRIPTORS);
                // One entry a slot at most, and the vector holds every slot.
                let _ = reclaimed.push(ReclaimedChain::new(ChainId::new(epoch, head), chain));
            }
        }
        // Unconfigured, and it stays so: the queue a driver uses after a
        // reset is the one the next `bring_up` makes.
        self.state = QueueState::unconfigured(self.layout.size, next_epoch);
        Ok(reclaimed)
    }
}

#[cfg(test)]
mod tests;
