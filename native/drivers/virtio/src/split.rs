//! Allocation-free VirtIO 1.3 split virtqueue.
//!
//! The queue owns its descriptor, driver, and device areas.  Payload buffers
//! remain owned by the device-class driver; submitting their IOVAs is unsafe
//! for exactly one reason: the caller must keep those buffers live and
//! unaliased in the declared direction until the returned token completes or
//! the device is reset.

use core::sync::atomic::{Ordering, fence};

use kore_abi::device::IovaAddr;
use kore_driver::{Dma, DmaPool, DmaWrite};

const DESC_BYTES: usize = 16;
const USED_ELEM_BYTES: usize = 8;
const NONE: u16 = u16::MAX;
const DESC_F_NEXT: u16 = 1;
const DESC_F_WRITE: u16 = 2;
const VRING_USED_F_NO_NOTIFY: u16 = 1;

/// Which participant may mutate a submitted payload span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// The device may only read bytes produced by the driver.
    DeviceReads,
    /// The device may write bytes for the driver to consume.
    DeviceWrites,
}

/// One contiguous device-visible payload span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaSpan {
    pub iova: IovaAddr,
    pub len: u32,
    pub direction: Direction,
}

impl DmaSpan {
    /// Resolve a checked subrange of a DMA buffer into a descriptor span.
    pub fn from_buffer<B: Dma + ?Sized>(
        buffer: &B,
        offset: usize,
        len: usize,
        direction: Direction,
    ) -> Result<Self, QueueError> {
        if len == 0 || len > u32::MAX as usize {
            return Err(QueueError::InvalidSpan);
        }
        let end = offset.checked_add(len).ok_or(QueueError::InvalidSpan)?;
        if end > buffer.len() {
            return Err(QueueError::InvalidSpan);
        }
        let address = buffer
            .iova()
            .0
            .checked_add(offset as u64)
            .ok_or(QueueError::InvalidSpan)?;
        address
            .checked_add(len as u64 - 1)
            .ok_or(QueueError::InvalidSpan)?;
        Ok(Self {
            iova: IovaAddr(address),
            len: len as u32,
            direction,
        })
    }
}

/// Stable device addresses programmed into the transport's queue registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueMemory {
    pub size: u16,
    pub descriptors: IovaAddr,
    pub driver_area: IovaAddr,
    pub device_area: IovaAddr,
}

/// A submission identity, scoped to one queue reset epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueToken {
    head: u16,
    epoch: u32,
    generation: u32,
}

impl QueueToken {
    pub const fn head(self) -> u16 {
        self.head
    }

    pub const fn epoch(self) -> u32 {
        self.epoch
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Successful publication of one descriptor chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Submission {
    pub token: QueueToken,
    /// Whether the transport should notify the device after this publication.
    pub notify: bool,
}

/// One validated used-ring completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub token: QueueToken,
    /// Bytes the device wrote into `DeviceWrites` spans.
    pub written: u32,
}

/// Split-ring construction, submission, or completion refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueError {
    InvalidSize,
    AllocationFailed,
    MisalignedMemory,
    InvalidSpan,
    Saturated,
    StaleToken,
    InvalidUsedIndex,
    InvalidUsedId(u32),
    DuplicateCompletion(u16),
    UsedLength { written: u32, capacity: u32 },
    Memory,
    Poisoned,
}

/// One VirtIO split queue with a compile-time bounded descriptor table.
pub struct SplitQueue<B: Dma, const N: usize> {
    descriptors: B,
    driver: B,
    device: B,
    free_next: [u16; N],
    chain_next: [u16; N],
    chain_len: [u16; N],
    writable_capacity: [u32; N],
    generations: [u32; N],
    in_flight: [bool; N],
    free_head: u16,
    free_count: u16,
    avail_idx: u16,
    last_used: u16,
    epoch: u32,
    event_idx: bool,
    poisoned: bool,
}

impl<B: Dma, const N: usize> SplitQueue<B, N> {
    /// Allocate and initialize one queue from an instance-owned DMA pool.
    pub fn allocate<P>(pool: &P, event_idx: bool) -> Result<Self, QueueError>
    where
        P: DmaPool<Buf = B>,
    {
        if N == 0 || N > 32_768 || !N.is_power_of_two() || N > u16::MAX as usize {
            return Err(QueueError::InvalidSize);
        }
        let desc_len = DESC_BYTES.checked_mul(N).ok_or(QueueError::InvalidSize)?;
        let driver_len = 6usize
            .checked_add(2usize.checked_mul(N).ok_or(QueueError::InvalidSize)?)
            .and_then(|n| n.checked_add(2))
            .ok_or(QueueError::InvalidSize)?;
        let device_len = 6usize
            .checked_add(
                USED_ELEM_BYTES
                    .checked_mul(N)
                    .ok_or(QueueError::InvalidSize)?,
            )
            .and_then(|n| n.checked_add(2))
            .ok_or(QueueError::InvalidSize)?;

        let descriptors = pool.alloc(desc_len).ok_or(QueueError::AllocationFailed)?;
        let driver = pool.alloc(driver_len).ok_or(QueueError::AllocationFailed)?;
        let device = pool.alloc(device_len).ok_or(QueueError::AllocationFailed)?;
        if descriptors.len() < desc_len || driver.len() < driver_len || device.len() < device_len {
            return Err(QueueError::Memory);
        }
        if !descriptors.iova().0.is_multiple_of(16)
            || !driver.iova().0.is_multiple_of(2)
            || !device.iova().0.is_multiple_of(4)
            || !(descriptors.cpu() as usize).is_multiple_of(16)
            || !(driver.cpu() as usize).is_multiple_of(2)
            || !(device.cpu() as usize).is_multiple_of(4)
        {
            return Err(QueueError::MisalignedMemory);
        }

        let mut queue = Self {
            descriptors,
            driver,
            device,
            free_next: [NONE; N],
            chain_next: [NONE; N],
            chain_len: [0; N],
            writable_capacity: [0; N],
            generations: [0; N],
            in_flight: [false; N],
            free_head: NONE,
            free_count: 0,
            avail_idx: 0,
            last_used: 0,
            epoch: 0,
            event_idx,
            poisoned: false,
        };
        queue.reset()?;
        Ok(queue)
    }

    pub fn memory(&self) -> QueueMemory {
        QueueMemory {
            size: N as u16,
            descriptors: self.descriptors.iova(),
            driver_area: self.driver.iova(),
            device_area: self.device.iova(),
        }
    }

    pub const fn epoch(&self) -> u32 {
        self.epoch
    }

    pub const fn free_descriptors(&self) -> u16 {
        self.free_count
    }

    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Publish a payload chain.
    ///
    /// # Safety
    ///
    /// Every span's backing DMA object must remain live until this token is
    /// completed or the device is reset.  The caller must not mutate a
    /// `DeviceReads` span or access a `DeviceWrites` span during that interval.
    pub unsafe fn submit(&mut self, spans: &[DmaSpan]) -> Result<Submission, QueueError> {
        if self.poisoned {
            return Err(QueueError::Poisoned);
        }
        if spans.is_empty() || spans.len() > N {
            return Err(QueueError::InvalidSpan);
        }
        if usize::from(self.free_count) < spans.len() {
            return Err(QueueError::Saturated);
        }

        let mut writable = 0u32;
        for span in spans {
            if span.len == 0 || span.iova.0.checked_add(u64::from(span.len) - 1).is_none() {
                return Err(QueueError::InvalidSpan);
            }
            if span.direction == Direction::DeviceWrites {
                writable = writable
                    .checked_add(span.len)
                    .ok_or(QueueError::InvalidSpan)?;
            }
        }

        let mut indices = [0u16; N];
        for index in indices.iter_mut().take(spans.len()) {
            *index = self.take_free().ok_or(QueueError::Saturated)?;
        }
        let head = indices[0];

        for (position, span) in spans.iter().enumerate() {
            let descriptor = indices[position];
            let next = if position + 1 < spans.len() {
                indices[position + 1]
            } else {
                NONE
            };
            self.chain_next[descriptor as usize] = next;
            let mut flags = if span.direction == Direction::DeviceWrites {
                DESC_F_WRITE
            } else {
                0
            };
            if next != NONE {
                flags |= DESC_F_NEXT;
            }
            if self
                .write_descriptor(descriptor, span.iova.0, span.len, flags, next)
                .is_err()
            {
                self.poisoned = true;
                return Err(QueueError::Memory);
            }
        }
        self.chain_len[head as usize] = spans.len() as u16;
        self.writable_capacity[head as usize] = writable;
        self.generations[head as usize] = self.generations[head as usize].wrapping_add(1);
        if self.generations[head as usize] == 0 {
            self.generations[head as usize] = 1;
        }
        self.in_flight[head as usize] = true;

        let old = self.avail_idx;
        let slot = usize::from(old) & (N - 1);
        self.driver
            .write_u16_at(4 + slot * 2, head)
            .map_err(|_| self.poison())?;
        fence(Ordering::Release);
        self.avail_idx = old.wrapping_add(1);
        self.driver
            .write_u16_at(2, self.avail_idx)
            .map_err(|_| self.poison())?;
        let notify = self.notification_required(old, self.avail_idx)?;
        Ok(Submission {
            token: QueueToken {
                head,
                epoch: self.epoch,
                generation: self.generations[head as usize],
            },
            notify,
        })
    }

    /// Pop and validate one used element. Malformed device input poisons the
    /// queue; only an explicit device reset plus [`reset`](Self::reset) may
    /// make it usable again.
    pub fn complete_next(&mut self) -> Result<Option<Completion>, QueueError> {
        if self.poisoned {
            return Err(QueueError::Poisoned);
        }
        let used_idx = self.read_device_u16(2)?;
        fence(Ordering::Acquire);
        let pending = used_idx.wrapping_sub(self.last_used);
        if usize::from(pending) > N {
            return Err(self.poison_as(QueueError::InvalidUsedIndex));
        }
        if pending == 0 {
            return Ok(None);
        }

        let slot = usize::from(self.last_used) & (N - 1);
        let id = self.read_device_u32(4 + slot * USED_ELEM_BYTES)?;
        let written = self.read_device_u32(8 + slot * USED_ELEM_BYTES)?;
        if id >= N as u32 {
            return Err(self.poison_as(QueueError::InvalidUsedId(id)));
        }
        let head = id as u16;
        if !self.in_flight[head as usize] {
            return Err(self.poison_as(QueueError::DuplicateCompletion(head)));
        }
        let capacity = self.writable_capacity[head as usize];
        if written > capacity {
            return Err(self.poison_as(QueueError::UsedLength { written, capacity }));
        }

        self.last_used = self.last_used.wrapping_add(1);
        self.in_flight[head as usize] = false;
        self.writable_capacity[head as usize] = 0;
        let chain_len = self.chain_len[head as usize];
        self.chain_len[head as usize] = 0;
        let mut descriptor = head;
        for _ in 0..chain_len {
            let next = self.chain_next[descriptor as usize];
            self.chain_next[descriptor as usize] = NONE;
            self.return_free(descriptor);
            descriptor = next;
        }
        Ok(Some(Completion {
            token: QueueToken {
                head,
                epoch: self.epoch,
                generation: self.generations[head as usize],
            },
            written,
        }))
    }

    /// Validate that a token still belongs to this device-reset epoch.
    pub fn validate_token(&self, token: QueueToken) -> Result<(), QueueError> {
        if token.epoch != self.epoch
            || token.head as usize >= N
            || self.generations[token.head as usize] != token.generation
        {
            return Err(QueueError::StaleToken);
        }
        Ok(())
    }

    /// Reinitialize software and ring memory after the transport has reset the
    /// device. Outstanding tokens become stale.
    pub fn reset(&mut self) -> Result<(), QueueError> {
        self.descriptors
            .fill_at(0, self.descriptors.len(), 0)
            .map_err(|_| QueueError::Memory)?;
        self.driver
            .fill_at(0, self.driver.len(), 0)
            .map_err(|_| QueueError::Memory)?;
        self.device
            .fill_at(0, self.device.len(), 0)
            .map_err(|_| QueueError::Memory)?;
        for index in 0..N {
            self.free_next[index] = if index + 1 < N {
                (index + 1) as u16
            } else {
                NONE
            };
            self.chain_next[index] = NONE;
            self.chain_len[index] = 0;
            self.writable_capacity[index] = 0;
            self.generations[index] = 0;
            self.in_flight[index] = false;
        }
        self.free_head = 0;
        self.free_count = N as u16;
        self.avail_idx = 0;
        self.last_used = 0;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
        self.poisoned = false;
        Ok(())
    }

    /// Simulate one device completion for an owning-layer unit test.
    #[cfg(test)]
    pub(crate) fn device_complete_for_test(
        &mut self,
        token: QueueToken,
        written: u32,
    ) -> Result<(), QueueError> {
        self.validate_token(token)?;
        let slot = usize::from(self.last_used) & (N - 1);
        let offset = 4 + slot * USED_ELEM_BYTES;
        self.device
            .write_u32_at(offset, u32::from(token.head))
            .and_then(|_| self.device.write_u32_at(offset + 4, written))
            .and_then(|_| self.device.write_u16_at(2, self.last_used.wrapping_add(1)))
            .map_err(|_| QueueError::Memory)
    }

    fn take_free(&mut self) -> Option<u16> {
        if self.free_head == NONE {
            return None;
        }
        let descriptor = self.free_head;
        self.free_head = self.free_next[descriptor as usize];
        self.free_next[descriptor as usize] = NONE;
        self.free_count -= 1;
        Some(descriptor)
    }

    fn return_free(&mut self, descriptor: u16) {
        self.free_next[descriptor as usize] = self.free_head;
        self.free_head = descriptor;
        self.free_count += 1;
    }

    fn write_descriptor(
        &self,
        descriptor: u16,
        address: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) -> Result<(), QueueError> {
        let offset = usize::from(descriptor) * DESC_BYTES;
        self.descriptors
            .write_u64_at(offset, address)
            .and_then(|_| self.descriptors.write_u32_at(offset + 8, len))
            .and_then(|_| self.descriptors.write_u16_at(offset + 12, flags))
            .and_then(|_| {
                self.descriptors
                    .write_u16_at(offset + 14, if next == NONE { 0 } else { next })
            })
            .map_err(|_| QueueError::Memory)
    }

    fn notification_required(&mut self, old: u16, new: u16) -> Result<bool, QueueError> {
        if self.event_idx {
            let event = self.read_device_u16(4 + N * USED_ELEM_BYTES)?;
            Ok(new.wrapping_sub(event).wrapping_sub(1) < new.wrapping_sub(old))
        } else {
            Ok(self.read_device_u16(0)? & VRING_USED_F_NO_NOTIFY == 0)
        }
    }

    fn read_device_u16(&mut self, offset: usize) -> Result<u16, QueueError> {
        let end = offset.checked_add(2).ok_or_else(|| self.poison())?;
        if end > self.device.len() || !offset.is_multiple_of(2) {
            return Err(self.poison());
        }
        // SAFETY: the DMA contract keeps the window live; bounds and natural
        // alignment were checked. Volatile observes the device's coherent write.
        Ok(u16::from_le(unsafe {
            core::ptr::read_volatile(self.device.cpu().add(offset).cast::<u16>())
        }))
    }

    fn read_device_u32(&mut self, offset: usize) -> Result<u32, QueueError> {
        let end = offset.checked_add(4).ok_or_else(|| self.poison())?;
        if end > self.device.len() || !offset.is_multiple_of(4) {
            return Err(self.poison());
        }
        // SAFETY: the DMA contract keeps the window live; bounds and natural
        // alignment were checked. Volatile observes the device's coherent write.
        Ok(u32::from_le(unsafe {
            core::ptr::read_volatile(self.device.cpu().add(offset).cast::<u32>())
        }))
    }

    fn poison(&mut self) -> QueueError {
        self.poisoned = true;
        QueueError::Memory
    }

    fn poison_as(&mut self, error: QueueError) -> QueueError {
        self.poisoned = true;
        error
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, vec, vec::Vec};
    use core::cell::{Cell, RefCell};

    use super::*;
    use kore_driver::DmaRead;

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
        allocations: RefCell<Vec<(u64, usize)>>,
    }

    impl HostPool {
        fn new() -> Self {
            Self {
                next: Cell::new(0x1_0000_0000),
                allocations: RefCell::new(Vec::new()),
            }
        }
    }

    impl DmaPool for HostPool {
        type Buf = HostBuf;

        fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
            let bytes = bytes.max(4096).next_multiple_of(4096);
            let iova = self.next.get();
            self.next.set(iova.checked_add(bytes as u64)?);
            self.allocations.borrow_mut().push((iova, bytes));
            Some(HostBuf {
                words: vec![0u64; bytes / 8].into_boxed_slice(),
                iova: IovaAddr(iova),
            })
        }
    }

    fn queue<const N: usize>() -> SplitQueue<HostBuf, N> {
        SplitQueue::allocate(&HostPool::new(), false).unwrap()
    }

    fn span(address: u64, direction: Direction) -> DmaSpan {
        DmaSpan {
            iova: IovaAddr(address),
            len: 64,
            direction,
        }
    }

    fn device_complete<const N: usize>(
        queue: &mut SplitQueue<HostBuf, N>,
        slot: u16,
        id: u32,
        len: u32,
        used_idx: u16,
    ) {
        let offset = 4 + (usize::from(slot) & (N - 1)) * USED_ELEM_BYTES;
        queue.device.write_u32_at(offset, id).unwrap();
        queue.device.write_u32_at(offset + 4, len).unwrap();
        fence(Ordering::Release);
        queue.device.write_u16_at(2, used_idx).unwrap();
    }

    #[test]
    fn descriptors_preserve_direction_and_chain_shape() {
        let mut queue = queue::<8>();
        // SAFETY: test IOVAs name fixture memory for the simulated lifetime.
        let submitted = unsafe {
            queue.submit(&[
                span(0x2000, Direction::DeviceReads),
                span(0x3000, Direction::DeviceWrites),
            ])
        }
        .unwrap();
        let head = submitted.token.head as usize;
        let second = queue.chain_next[head] as usize;
        assert_eq!(
            queue.descriptors.read_u16_at(head * 16 + 12),
            Ok(DESC_F_NEXT)
        );
        assert_eq!(
            queue.descriptors.read_u16_at(second * 16 + 12),
            Ok(DESC_F_WRITE)
        );
        assert_eq!(queue.writable_capacity[head], 64);
    }

    #[test]
    fn saturation_is_typed_and_completion_recycles_descriptors() {
        let mut queue = queue::<2>();
        // SAFETY: fixture spans remain live through reset/completion.
        let first = unsafe {
            queue.submit(&[
                span(0x2000, Direction::DeviceReads),
                span(0x3000, Direction::DeviceWrites),
            ])
        }
        .unwrap();
        assert_eq!(queue.free_descriptors(), 0);
        // SAFETY: same fixture contract; refusal publishes nothing.
        assert_eq!(
            unsafe { queue.submit(&[span(0x4000, Direction::DeviceReads)]) },
            Err(QueueError::Saturated)
        );
        device_complete(&mut queue, 0, u32::from(first.token.head), 32, 1);
        assert_eq!(queue.complete_next().unwrap().unwrap().written, 32);
        assert_eq!(queue.free_descriptors(), 2);
    }

    #[test]
    fn available_and_used_indices_wrap_without_losing_ownership() {
        let mut queue = queue::<4>();
        queue.avail_idx = u16::MAX;
        queue.last_used = u16::MAX;
        queue.driver.write_u16_at(2, u16::MAX).unwrap();
        queue.device.write_u16_at(2, u16::MAX).unwrap();
        // SAFETY: fixture span remains live through completion.
        let submission = unsafe { queue.submit(&[span(0x2000, Direction::DeviceWrites)]) }.unwrap();
        assert_eq!(queue.avail_idx, 0);
        device_complete(&mut queue, u16::MAX, u32::from(submission.token.head), 7, 0);
        let completion = queue.complete_next().unwrap().unwrap();
        assert_eq!(completion.token, submission.token);
        assert_eq!(completion.written, 7);
        assert_eq!(queue.last_used, 0);
    }

    #[test]
    fn invalid_used_id_and_length_poison_until_reset() {
        let mut queue = queue::<4>();
        // SAFETY: fixture span remains live until reset.
        unsafe { queue.submit(&[span(0x2000, Direction::DeviceWrites)]) }.unwrap();
        device_complete(&mut queue, 0, 99, 0, 1);
        assert_eq!(queue.complete_next(), Err(QueueError::InvalidUsedId(99)));
        assert_eq!(queue.complete_next(), Err(QueueError::Poisoned));
        queue.reset().unwrap();

        // SAFETY: new-epoch fixture span remains live until reset.
        let submission = unsafe { queue.submit(&[span(0x3000, Direction::DeviceWrites)]) }.unwrap();
        device_complete(&mut queue, 0, u32::from(submission.token.head), 65, 1);
        assert_eq!(
            queue.complete_next(),
            Err(QueueError::UsedLength {
                written: 65,
                capacity: 64
            })
        );
        assert!(queue.is_poisoned());
    }

    #[test]
    fn reset_invalidates_old_tokens_and_clears_poison() {
        let mut queue = queue::<4>();
        // SAFETY: fixture span is kept until reset, which terminates DMA use.
        let old = unsafe { queue.submit(&[span(0x2000, Direction::DeviceReads)]) }
            .unwrap()
            .token;
        let old_epoch = queue.epoch();
        queue.reset().unwrap();
        assert_ne!(queue.epoch(), old_epoch);
        assert_eq!(queue.validate_token(old), Err(QueueError::StaleToken));
        assert_eq!(queue.free_descriptors(), 4);
        assert!(!queue.is_poisoned());
    }

    #[test]
    fn descriptor_reuse_advances_token_generation() {
        let mut queue = queue::<1>();
        // SAFETY: the first fixture span remains live through completion.
        let first = unsafe { queue.submit(&[span(0x2000, Direction::DeviceReads)]) }
            .unwrap()
            .token;
        device_complete(&mut queue, 0, u32::from(first.head), 0, 1);
        queue.complete_next().unwrap().unwrap();
        // SAFETY: the replacement fixture span remains live for this epoch.
        let second = unsafe { queue.submit(&[span(0x3000, Direction::DeviceReads)]) }
            .unwrap()
            .token;
        assert_eq!(first.head(), second.head());
        assert_ne!(first.generation(), second.generation());
        assert_eq!(queue.validate_token(first), Err(QueueError::StaleToken));
        assert_eq!(queue.validate_token(second), Ok(()));
    }

    #[test]
    fn duplicate_completion_is_rejected() {
        let mut queue = queue::<4>();
        // SAFETY: fixture span remains live through the first completion.
        let submission = unsafe { queue.submit(&[span(0x2000, Direction::DeviceReads)]) }.unwrap();
        device_complete(&mut queue, 0, u32::from(submission.token.head), 0, 1);
        queue.complete_next().unwrap().unwrap();
        device_complete(&mut queue, 1, u32::from(submission.token.head), 0, 2);
        assert_eq!(
            queue.complete_next(),
            Err(QueueError::DuplicateCompletion(submission.token.head))
        );
    }

    #[test]
    fn event_idx_notification_math_handles_wrap() {
        let mut queue = SplitQueue::<HostBuf, 4>::allocate(&HostPool::new(), true).unwrap();
        queue.avail_idx = u16::MAX;
        queue.driver.write_u16_at(2, u16::MAX).unwrap();
        queue
            .device
            .write_u16_at(4 + 4 * USED_ELEM_BYTES, u16::MAX)
            .unwrap();
        // SAFETY: fixture span remains live for this queue epoch.
        let submission = unsafe { queue.submit(&[span(0x2000, Direction::DeviceReads)]) }.unwrap();
        assert!(submission.notify);
    }
}
