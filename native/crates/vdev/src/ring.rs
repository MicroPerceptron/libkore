//! **The split virtqueue, in the driver's memory**: the device's half.
//!
//! A driver lays a queue out in the memory it lends: a descriptor table, an
//! available ring it writes and a used ring the device writes, in the split
//! layout of Virtio 1.3 §2.7. It tells the device where the three are by
//! device address, publishes a chain of descriptors by adding its head to the
//! available ring, and takes the chain back from the used ring with the
//! number of bytes the device wrote.
//!
//! [`DeviceRing`] is the device's half of that, behind the queue contracts of
//! `virtio-accel-transport` ([`DeviceQueue`], [`DeviceChain`]). The protocol
//! ships a queue for the same contracts, `virtio-accel-split-queue`, but it
//! is a model that owns its rings and every descriptor's bytes on the heap of
//! one task. A driver and a device in two address spaces need the rings in
//! memory both map, which is this. The model is the oracle: the tests here
//! put each chain shape through both and require the same answer.
//!
//! # What the device trusts
//!
//! Nothing the driver wrote. Every index is checked against the queue's size,
//! a chain is walked at most `size` descriptors with a visited set so a loop
//! ends, every span must lie inside the window ([`Memory`]), and a chain is
//! read **once**: the walk snapshots each descriptor, and the ports then use
//! the snapshot, so a driver that rewrites a descriptor after publishing it
//! changes nothing the device does. A chain that is malformed is not an
//! error of the queue: it is handed up like any other, its [`io`] says why,
//! and the device completes it with a used length of zero, as the protocol's
//! command-chain rules require.
//!
//! The profile is the protocol's: direct descriptors only, no indirect
//! tables, no event indices, and the two suppression flags.
//!
//! [`io`]: DeviceChain::io

use alloc::rc::Rc;
use core::sync::atomic::{AtomicU64, Ordering, fence};

use virtio_accel_transport::{
    ByteAccessError, ChainError, ChainId, ChainIo, ChainIoResult, ChainRegion, DeviceChain,
    DeviceQueue, MalformedChain, NotificationHint, NotificationRecheck, QueueConfigError,
    QueueControl, QueueEpoch, QueueError, QueuePort, QueueSize, QueueState, ReadableBytes,
    UsedLength, WritableBytes, validate_chain_layout,
};

use crate::mem::{Memory, MemoryFault};

/// The largest queue a device offers.
///
/// A chain is walked with a visited set one machine word wide, so the size
/// is bounded by it; sixty-four descriptors is four of the longest command
/// chains, or thirty-two two-descriptor commands, in flight at once.
pub const MAX_QUEUE_SIZE: u16 = u64::BITS as u16;

/// One descriptor: `addr: u64, len: u32, flags: u16, next: u16`
/// (Virtio 1.3 §2.7.5).
pub(crate) const DESC_BYTES: u64 = 16;
const DESC_LEN: usize = 8;
const DESC_FLAGS: usize = 12;
const DESC_NEXT: usize = 14;
/// The chain continues at `next`.
pub const VIRTQ_DESC_F_NEXT: u16 = 1 << 0;
/// The device writes this descriptor's bytes; otherwise it reads them.
pub const VIRTQ_DESC_F_WRITE: u16 = 1 << 1;
/// The descriptor names an indirect table: not in this profile.
pub const VIRTQ_DESC_F_INDIRECT: u16 = 1 << 2;

/// Both rings begin `flags: u16, idx: u16` (§2.7.6, §2.7.8).
pub(crate) const RING_FLAGS: u64 = 0;
pub(crate) const RING_IDX: u64 = 2;
pub(crate) const RING_HEADER: u64 = 4;
/// An available-ring entry is a descriptor index.
pub(crate) const AVAIL_ENTRY: u64 = 2;
/// A used-ring entry is `id: u32, len: u32`.
pub(crate) const USED_ENTRY: u64 = 8;
pub(crate) const USED_LEN: u64 = 4;
/// The driver asks not to be interrupted when the device uses a chain.
pub const VIRTQ_AVAIL_F_NO_INTERRUPT: u16 = 1 << 0;
/// The device asks not to be notified when the driver publishes a chain.
pub const VIRTQ_USED_F_NO_NOTIFY: u16 = 1 << 0;

/// Alignments the layout requires of each part (§2.7's table).
pub(crate) const DESC_ALIGN: u64 = 16;
pub(crate) const AVAIL_ALIGN: u64 = 2;
pub(crate) const USED_ALIGN: u64 = 4;

/// Where a driver put a queue's three parts, as device addresses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingAddresses {
    pub descriptors: u64,
    pub available: u64,
    pub used: u64,
}

/// Why the ring could not go on. Each is the driver's doing, and none is one
/// chain's: a device that sees one stops serving the queue and asks for a
/// reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    /// A part of the queue is not where the layout allows: outside the
    /// window, or not aligned for its entries.
    Layout,
    /// The queue was made ready smaller than the longest chain it must hold.
    ChainLimitExceedsQueue,
    /// The queue cannot be reconfigured while it is ready.
    Busy,
    /// The available index is further ahead than the queue is long.
    AvailableIndex,
    /// A head was published that the device already holds.
    HeadInFlight,
    /// The window stopped answering for a ring the queue validated.
    Memory(MemoryFault),
}

impl From<MemoryFault> for QueueError<RingError> {
    fn from(fault: MemoryFault) -> Self {
        Self::Transport(RingError::Memory(fault))
    }
}

/// Whether a ring can hold a queue of at most `max_size` whose chains are at
/// most `max_chain_descriptors` long: a chain is two descriptors at least,
/// and no longer than the queue.
pub const fn limits_fit(max_size: QueueSize, max_chain_descriptors: u16) -> bool {
    max_size.get() <= MAX_QUEUE_SIZE
        && max_chain_descriptors >= 2
        && max_chain_descriptors <= max_size.get()
}

/// One descriptor's bytes, snapshotted when its chain was taken.
#[derive(Clone, Copy, Debug)]
struct Span {
    at: u64,
    len: u32,
}

/// The chain's spans in one direction, as one run of bytes.
#[derive(Clone, Debug)]
struct Port<M> {
    memory: M,
    spans: heapless::Vec<Span, { MAX_QUEUE_SIZE as usize }>,
    len: u64,
    epoch: QueueEpoch,
    current: Rc<AtomicU64>,
}

impl<M: Memory> Port<M> {
    /// The spans `len` bytes at `offset` fall in, as `(address, from, count)`
    /// with `from` the offset into the caller's buffer.
    fn each(
        &self,
        offset: u64,
        len: usize,
        mut visit: impl FnMut(u64, usize, usize) -> Result<(), MemoryFault>,
    ) -> Result<(), ByteAccessError> {
        if self.current.load(Ordering::Acquire) != self.epoch.get() {
            return Err(ByteAccessError::Reset);
        }
        if !offset
            .checked_add(len as u64)
            .is_some_and(|end| end <= self.len)
        {
            return Err(ByteAccessError::OutOfBounds);
        }
        let (mut skip, mut done) = (offset, 0usize);
        for span in &self.spans {
            let span_len = u64::from(span.len);
            if skip >= span_len {
                skip -= span_len;
                continue;
            }
            if done == len {
                break;
            }
            let count = ((span_len - skip) as usize).min(len - done);
            visit(span.at + skip, done, count).map_err(|_| ByteAccessError::Access)?;
            done += count;
            skip = 0;
        }
        Ok(())
    }
}

/// The bytes a chain gives the device to read.
#[derive(Debug)]
pub struct RingSource<M>(Port<M>);

impl<M: Memory + core::fmt::Debug> ReadableBytes for RingSource<M> {
    fn len(&self) -> u64 {
        self.0.len
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
        let memory = &self.0.memory;
        let len = target.len();
        self.0.each(offset, len, |at, from, count| {
            memory.read(at, &mut target[from..from + count])
        })
    }
}

/// The bytes a chain gives the device to write.
#[derive(Debug)]
pub struct RingSink<M>(Port<M>);

impl<M: Memory + core::fmt::Debug> WritableBytes for RingSink<M> {
    fn len(&self) -> u64 {
        self.0.len
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
        let memory = &self.0.memory;
        self.0.each(offset, source.len(), |at, from, count| {
            memory.write(at, &source[from..from + count])
        })
    }
}

/// One chain the device took from the available ring and has not yet used.
///
/// Not `Clone`: [`DeviceRing::complete`] consumes it, so a chain is used
/// once.
#[derive(Debug)]
pub struct RingChain<M> {
    id: ChainId,
    head: u16,
    regions: heapless::Vec<ChainRegion, { MAX_QUEUE_SIZE as usize }>,
    malformed: Option<MalformedChain>,
    request: RingSource<M>,
    response: RingSink<M>,
}

impl<M: Memory + core::fmt::Debug> DeviceChain for RingChain<M> {
    type Request = RingSource<M>;
    type Response = RingSink<M>;
    type Error = RingError;

    fn id(&self) -> ChainId {
        self.id
    }

    fn io(&mut self) -> ChainIoResult<'_, Self::Request, Self::Response, Self::Error> {
        let current = self.request.0.current.load(Ordering::Acquire);
        if current != self.id.epoch().get() {
            return Err(ChainError::ResetRace {
                chain: self.id.epoch(),
                // An epoch is never zero: it starts at one and only grows.
                current: QueueEpoch::new(current).unwrap_or(QueueEpoch::INITIAL),
            });
        }
        if let Some(malformed) = self.malformed {
            return Err(ChainError::Malformed(malformed));
        }
        Ok(ChainIo::new(
            &self.regions,
            &self.request,
            &mut self.response,
        ))
    }
}

/// The device's half of one split virtqueue in a driver's memory.
#[derive(Debug)]
pub struct DeviceRing<M> {
    memory: M,
    state: QueueState,
    max_chain_descriptors: u16,
    addresses: RingAddresses,
    /// The epoch every chain this queue hands out carries, shared with them
    /// so a reset reaches a chain the device still holds.
    current: Rc<AtomicU64>,
    /// Available entries taken, and used entries published.
    taken: u16,
    used: u16,
    /// Heads the device holds, one bit each.
    in_flight: u64,
}

impl<M: Memory + Clone + core::fmt::Debug> DeviceRing<M> {
    /// A queue in `memory` of at most `max_size` descriptors, whose chains
    /// are at most `max_chain_descriptors` long. Unconfigured: the driver
    /// sets its size and addresses and makes it ready.
    ///
    /// `None` when the limits do not fit: a chain is two descriptors at
    /// least, and no longer than the queue.
    pub fn new(memory: M, max_size: QueueSize, max_chain_descriptors: u16) -> Option<Self> {
        if !limits_fit(max_size, max_chain_descriptors) {
            return None;
        }
        let epoch = QueueEpoch::INITIAL;
        Some(Self {
            memory,
            state: QueueState::unconfigured(max_size, epoch),
            max_chain_descriptors,
            addresses: RingAddresses::default(),
            current: Rc::new(AtomicU64::new(epoch.get())),
            taken: 0,
            used: 0,
            in_flight: 0,
        })
    }

    /// The window the queue is in, given back: the queue is over.
    pub fn into_memory(self) -> M {
        self.memory
    }

    /// Where the driver says the queue's parts are. Taken while the queue is
    /// not ready, and checked when it is made so.
    pub fn set_addresses(&mut self, addresses: RingAddresses) -> Result<(), QueueError<RingError>> {
        if self.state.ready() {
            return Err(QueueError::Transport(RingError::Busy));
        }
        self.addresses = addresses;
        Ok(())
    }

    /// The addresses the queue was given.
    pub const fn addresses(&self) -> RingAddresses {
        self.addresses
    }

    /// Chains the device holds: taken and not yet used.
    pub const fn in_flight(&self) -> u32 {
        self.in_flight.count_ones()
    }

    fn size(&self) -> Result<u16, QueueError<RingError>> {
        match (self.state.ready(), self.state.size()) {
            (true, Some(size)) => Ok(size.get()),
            _ => Err(QueueError::NotReady),
        }
    }

    /// Whether the three parts lie inside the window where the layout
    /// allows them, for a queue of `size`.
    fn layout_holds(&self, size: u16) -> bool {
        let RingAddresses {
            descriptors,
            available,
            used,
        } = self.addresses;
        let size = u64::from(size);
        descriptors % DESC_ALIGN == 0
            && available % AVAIL_ALIGN == 0
            && used % USED_ALIGN == 0
            && self.memory.contains(descriptors, size * DESC_BYTES)
            && self
                .memory
                .contains(available, RING_HEADER + size * AVAIL_ENTRY)
            && self.memory.contains(used, RING_HEADER + size * USED_ENTRY)
    }

    /// Walk the chain at `head`, snapshotting each descriptor once.
    fn take(&self, head: u16, size: u16) -> Result<RingChain<M>, QueueError<RingError>> {
        let epoch = self.state.epoch();
        let port = || Port {
            memory: self.memory.clone(),
            spans: heapless::Vec::new(),
            len: 0,
            epoch,
            current: Rc::clone(&self.current),
        };
        let mut chain = RingChain {
            id: ChainId::new(epoch, u64::from(head)),
            head,
            regions: heapless::Vec::new(),
            malformed: None,
            request: RingSource(port()),
            response: RingSink(port()),
        };
        chain.malformed = self.walk(&mut chain, head, size)?.err();
        Ok(chain)
    }

    /// The walk proper. The outer error is the queue's own memory failing;
    /// the inner one is this chain being malformed, which is the chain's
    /// business and not the queue's.
    fn walk(
        &self,
        chain: &mut RingChain<M>,
        head: u16,
        size: u16,
    ) -> Result<Result<(), MalformedChain>, QueueError<RingError>> {
        let (mut next, mut seen) = (head, 0u64);
        // At most `size` descriptors, a hostile loop included: the visited
        // set ends it on the first index met twice.
        for _ in 0..size {
            if next >= size {
                return Ok(Err(MalformedChain::DescriptorIndex));
            }
            if seen & (1 << next) != 0 {
                return Ok(Err(MalformedChain::DescriptorLoop));
            }
            seen |= 1 << next;

            let mut descriptor = [0u8; DESC_BYTES as usize];
            self.memory.read(
                self.addresses.descriptors + u64::from(next) * DESC_BYTES,
                &mut descriptor,
            )?;
            let word = |at: usize, len: usize| {
                let mut bytes = [0u8; 8];
                bytes[..len].copy_from_slice(&descriptor[at..at + len]);
                u64::from_le_bytes(bytes)
            };
            let (at, len) = (word(0, 8), word(DESC_LEN, 4) as u32);
            let flags = word(DESC_FLAGS, 2) as u16;
            next = word(DESC_NEXT, 2) as u16;

            if flags & VIRTQ_DESC_F_INDIRECT != 0 {
                return Ok(Err(MalformedChain::IndirectUnsupported));
            }
            if flags & !(VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE) != 0 {
                return Ok(Err(MalformedChain::DescriptorFlags));
            }
            let writable = flags & VIRTQ_DESC_F_WRITE != 0;
            let region = if writable {
                ChainRegion::writable(u64::from(len))
            } else {
                ChainRegion::readable(u64::from(len))
            };
            // The vectors hold `MAX_QUEUE_SIZE`, which bounds `size`.
            let _ = chain.regions.push(region);
            let port = if writable {
                &mut chain.response.0
            } else {
                &mut chain.request.0
            };
            let _ = port.spans.push(Span { at, len });
            port.len += u64::from(len);

            if flags & VIRTQ_DESC_F_NEXT == 0 {
                // Count, zero lengths, direction order and totals are the
                // protocol's own rule, by its own function.
                if let Err(error) =
                    validate_chain_layout(&chain.regions, self.max_chain_descriptors)
                {
                    return Ok(Err(error.into()));
                }
                // Every byte the engine may touch is mapped before it runs.
                let mapped = |port: &Port<M>| {
                    port.spans
                        .iter()
                        .all(|span| self.memory.contains(span.at, u64::from(span.len)))
                };
                if !mapped(&chain.request.0) || !mapped(&chain.response.0) {
                    return Ok(Err(MalformedChain::Address));
                }
                return Ok(Ok(()));
            }
        }
        // `size` descriptors and still a `next`: longer than any chain.
        Ok(Err(MalformedChain::DescriptorCount))
    }
}

impl<M> QueuePort for DeviceRing<M> {
    fn state(&self) -> QueueState {
        self.state
    }
}

impl<M: Memory + Clone + core::fmt::Debug> QueueControl for DeviceRing<M> {
    type Error = RingError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        if self.state.ready() {
            return Err(QueueError::Transport(RingError::Busy));
        }
        if size > self.state.max_size() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::SizeExceedsMaximum,
            ));
        }
        if size.get() < self.max_chain_descriptors {
            return Err(QueueError::Transport(RingError::ChainLimitExceedsQueue));
        }
        self.state = QueueState::new(self.state.max_size(), Some(size), false, self.state.epoch())
            .map_err(QueueError::InvalidConfiguration)?;
        Ok(())
    }

    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        if ready {
            let Some(size) = self.state.size() else {
                return Err(QueueError::InvalidConfiguration(
                    QueueConfigError::ReadyWithoutSize,
                ));
            };
            if !self.layout_holds(size.get()) {
                return Err(QueueError::Transport(RingError::Layout));
            }
        }
        self.state = QueueState::new(
            self.state.max_size(),
            self.state.size(),
            ready,
            self.state.epoch(),
        )
        .map_err(QueueError::InvalidConfiguration)?;
        Ok(())
    }
}

impl<M: Memory + Clone + core::fmt::Debug> DeviceQueue for DeviceRing<M> {
    type Chain = RingChain<M>;
    type Error = RingError;

    fn pop_available(&mut self) -> Result<Option<Self::Chain>, QueueError<Self::Error>> {
        let size = self.size()?;
        // Acquire: the entry and the descriptors the driver wrote before it
        // published this index are visible to the reads below.
        let published = self.memory.load_u16(self.addresses.available + RING_IDX)?;
        let pending = published.wrapping_sub(self.taken);
        if pending == 0 {
            return Ok(None);
        }
        if pending > size {
            return Err(QueueError::Transport(RingError::AvailableIndex));
        }
        let slot = u64::from(self.taken % size);
        let mut entry = [0u8; AVAIL_ENTRY as usize];
        self.memory.read(
            self.addresses.available + RING_HEADER + slot * AVAIL_ENTRY,
            &mut entry,
        )?;
        let head = u16::from_le_bytes(entry);
        self.taken = self.taken.wrapping_add(1);

        let chain = self.take(head, size)?;
        if head < size {
            // The head is a real descriptor: the device holds it until it
            // is used, and a second publication of it is the driver reusing
            // a descriptor it does not own.
            if self.in_flight & (1 << head) != 0 {
                return Err(QueueError::Transport(RingError::HeadInFlight));
            }
            self.in_flight |= 1 << head;
        }
        Ok(Some(chain))
    }

    fn complete(
        &mut self,
        chain: Self::Chain,
        used: UsedLength,
    ) -> Result<NotificationHint, QueueError<Self::Error>> {
        let current = self.state.epoch();
        if chain.id.epoch() != current {
            return Err(QueueError::ResetRace {
                operation: chain.id.epoch(),
                current,
            });
        }
        let size = self.size()?;
        let capacity = chain.response.0.len;
        if u64::from(used.get()) > capacity {
            return Err(QueueError::UsedLengthExceeded { used, capacity });
        }
        let slot = u64::from(self.used % size);
        let entry = self.addresses.used + RING_HEADER + slot * USED_ENTRY;
        self.memory.store_u32(entry, u32::from(chain.head))?;
        self.memory.store_u32(entry + USED_LEN, used.get())?;
        self.used = self.used.wrapping_add(1);
        // Release: the response bytes and the entry above are visible to a
        // driver that has loaded this index.
        self.memory
            .store_u16(self.addresses.used + RING_IDX, self.used)?;
        if u32::from(chain.head) < u64::BITS {
            self.in_flight &= !(1 << chain.head);
        }
        // The index is published before the driver's suppression flag is
        // read: a driver that clears the flag and then rechecks the index
        // cannot miss this entry, and one that set it is not interrupted.
        fence(Ordering::SeqCst);
        let flags = self
            .memory
            .load_u16(self.addresses.available + RING_FLAGS)?;
        Ok(if flags & VIRTQ_AVAIL_F_NO_INTERRUPT == 0 {
            NotificationHint::Notify
        } else {
            NotificationHint::Suppressed
        })
    }

    fn disable_available_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.size()?;
        self.memory
            .store_u16(self.addresses.used + RING_FLAGS, VIRTQ_USED_F_NO_NOTIFY)?;
        Ok(())
    }

    fn enable_available_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        self.size()?;
        self.memory.store_u16(self.addresses.used + RING_FLAGS, 0)?;
        // The flag is cleared before the index is read: a driver that
        // published while notifications were off is seen here, and one that
        // publishes after sees the flag clear and notifies.
        fence(Ordering::SeqCst);
        let published = self.memory.load_u16(self.addresses.available + RING_IDX)?;
        Ok(if published == self.taken {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(&mut self, next_epoch: QueueEpoch) -> Result<(), QueueError<Self::Error>> {
        if next_epoch <= self.state.epoch() {
            return Err(QueueError::InvalidConfiguration(
                QueueConfigError::NonIncreasingEpoch,
            ));
        }
        // Published first: a chain the device still holds refuses its bytes
        // and its completion from here on, so nothing of the old epoch is
        // written to memory the driver may now reuse.
        self.current.store(next_epoch.get(), Ordering::Release);
        self.state = QueueState::unconfigured(self.state.max_size(), next_epoch);
        self.addresses = RingAddresses::default();
        self.taken = 0;
        self.used = 0;
        self.in_flight = 0;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;
