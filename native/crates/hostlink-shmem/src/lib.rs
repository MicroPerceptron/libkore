//! A bounded byte plane over a shared-memory window.
//!
//! This crate deliberately does not know about HostLink envelopes, VirtIO,
//! PCI, capabilities, or doorbells.  It formats one caller-owned window as
//! two independent SPSC lanes and moves variable-length byte records through
//! fixed-size slots.  A transport maps the window, supplies a negotiated
//! nonzero [`Epoch`], and owns notification after publishing a batch.
//!
//! ```text
//! 0                 64               128
//! + immutable header + epoch cache line + lane A -> B + lane B -> A
//!                                      | head line | tail line | slots ...
//! slot = { epoch: u64, length: u32, reserved: u32, payload ... }
//! ```
//!
//! The producer writes a slot and publishes `tail` with `Release`.  The
//! consumer observes `tail` with `Acquire` before reading the slot, then
//! releases it by storing `head` with `Release`; the producer acquires `head`
//! before reusing the slot.  Full queues return [`SendError::Full`]: retry,
//! notification, timeout, and dropping are policy owned by the caller.
//!
//! Reset is a control-plane operation.  Both endpoints must be quiescent
//! before [`reset`] is called; changing the epoch concurrently with a grant is
//! a protocol violation.  Every attach and queue operation nevertheless
//! validates the epoch and fails closed, and every slot is epoch-stamped, so
//! stale bytes are never accepted as records from a new incarnation.

#![no_std]

#[cfg(test)]
extern crate std;

use core::marker::PhantomData;
use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_endian = "big")]
compile_error!("kore-hostlink-shmem v1 requires little-endian endpoints");

/// Shared-window magic: `KSHMPLN1`, little endian.
pub const MAGIC: u64 = u64::from_le_bytes(*b"KSHMPLN1");
/// Version of the frozen memory layout.
pub const ABI: u32 = 1;
/// Required base and lane alignment.
pub const ALIGN: usize = 64;

const HEADER_BYTES: usize = 64;
const EPOCH_AT: usize = HEADER_BYTES;
const WINDOW_CONTROL_BYTES: usize = 2 * ALIGN;
const LANE_HEAD_AT: usize = 0;
const LANE_TAIL_AT: usize = ALIGN;
const LANE_SLOTS_AT: usize = 2 * ALIGN;
const SLOT_HEADER_BYTES: usize = 16;
const SLOT_ALIGN: usize = align_of::<u64>();

/// A nonzero incarnation number negotiated by the control plane.
///
/// This is intentionally not HostLink's authenticated link-epoch type.  It
/// only prevents a mapping retained across reset from consuming stale slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Epoch(u64);

impl Epoch {
    /// Construct a valid reset epoch.
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    /// The negotiated integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Which endpoint is being attached to the symmetric window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
    A,
    B,
}

/// Checked geometry for both lanes of a window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    capacity: u32,
    payload_bytes: u32,
    slot_stride: usize,
    lane_bytes: usize,
    window_bytes: usize,
}

/// Invalid requested geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryError {
    /// Capacity must be a nonzero power of two.
    BadCapacity,
    /// A slot must hold at least one payload byte.
    EmptyPayload,
    /// A multiplication, addition, or alignment round-up overflowed.
    Overflow,
}

impl Layout {
    /// Validate and compute the exact window shape.
    pub fn new(capacity: u32, payload_bytes: u32) -> Result<Self, GeometryError> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(GeometryError::BadCapacity);
        }
        if payload_bytes == 0 {
            return Err(GeometryError::EmptyPayload);
        }
        let slot_raw = SLOT_HEADER_BYTES
            .checked_add(payload_bytes as usize)
            .ok_or(GeometryError::Overflow)?;
        let slot_stride = align_up(slot_raw, SLOT_ALIGN).ok_or(GeometryError::Overflow)?;
        if slot_stride > u32::MAX as usize {
            return Err(GeometryError::Overflow);
        }
        let slots = (capacity as usize)
            .checked_mul(slot_stride)
            .ok_or(GeometryError::Overflow)?;
        let lane_raw = LANE_SLOTS_AT
            .checked_add(slots)
            .ok_or(GeometryError::Overflow)?;
        let lane_bytes = align_up(lane_raw, ALIGN).ok_or(GeometryError::Overflow)?;
        let window_bytes = WINDOW_CONTROL_BYTES
            .checked_add(lane_bytes.checked_mul(2).ok_or(GeometryError::Overflow)?)
            .ok_or(GeometryError::Overflow)?;
        Ok(Self {
            capacity,
            payload_bytes,
            slot_stride,
            lane_bytes,
            window_bytes,
        })
    }

    #[must_use]
    pub const fn capacity(self) -> u32 {
        self.capacity
    }

    #[must_use]
    pub const fn payload_bytes(self) -> u32 {
        self.payload_bytes
    }

    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        self.lane_bytes
    }

    #[must_use]
    pub const fn window_bytes(self) -> usize {
        self.window_bytes
    }
}

const fn align_up(value: usize, alignment: usize) -> Option<usize> {
    match value.checked_add(alignment - 1) {
        Some(rounded) => Some(rounded & !(alignment - 1)),
        None => None,
    }
}

#[repr(C)]
struct Header {
    magic: u64,
    abi: u32,
    header_bytes: u32,
    window_bytes: u64,
    lane_bytes: u64,
    capacity: u32,
    payload_bytes: u32,
    slot_stride: u32,
    reserved32: u32,
    reserved64: [u64; 2],
}

const _: () = assert!(size_of::<Header>() == HEADER_BYTES);

#[repr(C)]
struct SlotHeader {
    epoch: u64,
    length: u32,
    reserved: u32,
}

const _: () = assert!(size_of::<SlotHeader>() == SLOT_HEADER_BYTES);

/// Why formatting, reset, or attachment refused a window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowError {
    Misaligned,
    TooSmall,
    BadHeader,
    WrongLayout,
    ZeroEpoch,
    SameEpoch,
    EpochMismatch { expected: Epoch, observed: u64 },
}

/// Format an exclusively owned window.  Bytes after `layout.window_bytes()`
/// are untouched.
pub fn init(window: &mut [u8], layout: Layout, epoch: Epoch) -> Result<(), WindowError> {
    validate_storage(window.as_mut_ptr(), window.len(), layout)?;

    // Clear the complete active region before installing magic.  With
    // exclusive ownership there are no live atomic references yet.
    window[..layout.window_bytes].fill(0);
    let base = window.as_mut_ptr();
    let header = Header {
        magic: 0,
        abi: ABI,
        header_bytes: HEADER_BYTES as u32,
        window_bytes: layout.window_bytes as u64,
        lane_bytes: layout.lane_bytes as u64,
        capacity: layout.capacity,
        payload_bytes: layout.payload_bytes,
        slot_stride: layout.slot_stride as u32,
        reserved32: 0,
        reserved64: [0; 2],
    };
    // SAFETY: storage validation proves base alignment and header bounds.
    unsafe {
        ptr::write(base.cast::<Header>(), header);
        ptr::write(
            base.add(EPOCH_AT).cast::<AtomicU64>(),
            AtomicU64::new(epoch.0),
        );
        for lane in 0..2 {
            let lane = base.add(lane_offset(layout, lane));
            ptr::write(
                lane.add(LANE_HEAD_AT).cast::<AtomicU64>(),
                AtomicU64::new(0),
            );
            ptr::write(
                lane.add(LANE_TAIL_AT).cast::<AtomicU64>(),
                AtomicU64::new(0),
            );
        }
        // Install the recognition word last: partial initialization refuses.
        (*base.cast::<Header>()).magic = MAGIC;
    }
    Ok(())
}

/// Re-open an empty incarnation of a formatted window.
///
/// Both peers must have stopped using all endpoints and grants before this
/// call.  The new epoch is published after both lanes have been emptied.
pub fn reset(window: &mut [u8], new_epoch: Epoch) -> Result<Layout, WindowError> {
    let layout = inspect(window.as_mut_ptr(), window.len())?;
    // SAFETY: inspect validated the aligned epoch control word.
    let epoch = unsafe { epoch_atomic(window.as_mut_ptr()) };
    if epoch.load(Ordering::Acquire) == 0 {
        return Err(WindowError::ZeroEpoch);
    }
    if epoch.load(Ordering::Acquire) == new_epoch.0 {
        return Err(WindowError::SameEpoch);
    }
    for lane in 0..2 {
        // SAFETY: inspect validated both lane control blocks.
        let lane = unsafe { window.as_mut_ptr().add(lane_offset(layout, lane)) };
        // Empty the lane before the Release publication of the new epoch.
        // SAFETY: inspect validated both aligned lane control words.
        unsafe { atomic_at(lane, LANE_HEAD_AT) }.store(0, Ordering::Relaxed);
        // SAFETY: as head.
        unsafe { atomic_at(lane, LANE_TAIL_AT) }.store(0, Ordering::Relaxed);
    }
    epoch.store(new_epoch.0, Ordering::Release);
    Ok(layout)
}

/// Format and bind both sides in one address space (tests and in-process
/// transports).  Each side owns one producer and the opposite consumer.
pub fn split(
    window: &mut [u8],
    layout: Layout,
    epoch: Epoch,
) -> Result<(Endpoint<'_>, Endpoint<'_>), WindowError> {
    init(window, layout, epoch)?;
    let (base, len) = (window.as_mut_ptr(), window.len());
    // SAFETY: the exclusive borrow pins the window.  The returned endpoints
    // access disjoint SPSC roles and retain the borrow for their lifetime.
    unsafe {
        Ok((
            bind_raw(base, len, epoch, Side::A)?,
            bind_raw(base, len, epoch, Side::B)?,
        ))
    }
}

/// Attach one side to an already formatted mapping.
///
/// The peer normally supplies its own virtual mapping of the same physical
/// pages.  On this side, the exclusive borrow prevents unrelated access.
pub fn endpoint(
    window: &mut [u8],
    expected_epoch: Epoch,
    side: Side,
) -> Result<Endpoint<'_>, WindowError> {
    // SAFETY: the borrow pins this mapping and excludes other local access.
    unsafe { bind_raw(window.as_mut_ptr(), window.len(), expected_epoch, side) }
}

/// One symmetric endpoint: `tx` feeds the peer's `rx`.
pub struct Endpoint<'a> {
    tx: Producer<'a>,
    rx: Consumer<'a>,
}

impl<'window> Endpoint<'window> {
    #[must_use]
    pub fn tx(&mut self) -> &mut Producer<'window> {
        &mut self.tx
    }

    #[must_use]
    pub fn rx(&mut self) -> &mut Consumer<'window> {
        &mut self.rx
    }
}

struct RawLane {
    epoch: *const AtomicU64,
    head: *const AtomicU64,
    tail: *const AtomicU64,
    slots: *mut u8,
    expected_epoch: Epoch,
    capacity: u64,
    mask: u64,
    payload_bytes: usize,
    slot_stride: usize,
}

impl RawLane {
    #[inline]
    fn epoch(&self) -> &AtomicU64 {
        // SAFETY: bind validated this aligned address in the pinned mapping.
        unsafe { &*self.epoch }
    }

    #[inline]
    fn head(&self) -> &AtomicU64 {
        // SAFETY: as epoch.
        unsafe { &*self.head }
    }

    #[inline]
    fn tail(&self) -> &AtomicU64 {
        // SAFETY: as epoch.
        unsafe { &*self.tail }
    }

    #[inline]
    fn validate_epoch(&self) -> Result<(), EpochChanged> {
        let observed = self.epoch().load(Ordering::Acquire);
        if observed == self.expected_epoch.0 {
            Ok(())
        } else {
            Err(EpochChanged {
                expected: self.expected_epoch,
                observed,
            })
        }
    }

    #[inline]
    unsafe fn slot(&self, cursor: u64) -> *mut SlotHeader {
        // SAFETY: caller established a live cursor. Mask bounds the slot
        // index, and checked layout arithmetic bounded the slot region.
        unsafe {
            self.slots
                .add((cursor & self.mask) as usize * self.slot_stride)
                .cast::<SlotHeader>()
        }
    }
}

/// The lane's unique publisher.
pub struct Producer<'a> {
    raw: RawLane,
    _window: PhantomData<&'a [u8]>,
}

/// The lane's unique receiver.
pub struct Consumer<'a> {
    raw: RawLane,
    _window: PhantomData<&'a [u8]>,
}

/// A reset invalidated an attached endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpochChanged {
    pub expected: Epoch,
    pub observed: u64,
}

/// A publish reservation was refused without modifying the queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SendError {
    EpochChanged(EpochChanged),
    TooLarge { requested: usize, maximum: usize },
    CorruptCursor,
    Full,
}

/// A receive attempt failed without releasing a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiveError {
    EpochChanged(EpochChanged),
    Empty,
    CorruptCursor,
    StaleSlot { expected: Epoch, observed: u64 },
    CorruptLength { observed: u32, maximum: u32 },
    OutputTooSmall { required: usize, available: usize },
    InvalidOffset { requested: usize, length: usize },
}

/// Progress from copying one bounded portion of a published record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadChunk {
    pub copied: usize,
    pub total: usize,
    pub complete: bool,
}

impl From<EpochChanged> for SendError {
    fn from(value: EpochChanged) -> Self {
        Self::EpochChanged(value)
    }
}

impl From<EpochChanged> for ReceiveError {
    fn from(value: EpochChanged) -> Self {
        Self::EpochChanged(value)
    }
}

impl<'window> Producer<'window> {
    /// Reserve one slot.  Dropping the grant without committing cancels the
    /// reservation and leaves the queue unchanged.
    pub fn try_reserve<'producer>(
        &'producer mut self,
        length: usize,
    ) -> Result<WriteGrant<'producer, 'window>, SendError> {
        self.raw.validate_epoch()?;
        if length > self.raw.payload_bytes {
            return Err(SendError::TooLarge {
                requested: length,
                maximum: self.raw.payload_bytes,
            });
        }
        let tail = self.raw.tail().load(Ordering::Relaxed);
        let head = self.raw.head().load(Ordering::Acquire);
        let outstanding = tail.wrapping_sub(head);
        if outstanding > self.raw.capacity {
            return Err(SendError::CorruptCursor);
        }
        if outstanding == self.raw.capacity {
            return Err(SendError::Full);
        }
        // SAFETY: the cursor is not full, so SPSC ownership assigns this slot
        // exclusively to the producer until publication.
        let slot = unsafe { self.raw.slot(tail) };
        Ok(WriteGrant {
            producer: self,
            slot,
            tail,
            length,
        })
    }

    /// Copy and publish one byte record.
    pub fn try_push(&mut self, bytes: &[u8]) -> Result<(), SendError> {
        let mut grant = self.try_reserve(bytes.len())?;
        grant.as_mut_slice().copy_from_slice(bytes);
        grant.commit()
    }

    /// Maximum payload length of one slot.
    #[must_use]
    pub const fn payload_bytes(&self) -> usize {
        self.raw.payload_bytes
    }

    /// Fixed number of records that may be outstanding.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.raw.capacity as u32
    }
}

/// An affine unpublished producer slot.
pub struct WriteGrant<'producer, 'window> {
    producer: &'producer mut Producer<'window>,
    slot: *mut SlotHeader,
    tail: u64,
    length: usize,
}

impl WriteGrant<'_, '_> {
    /// Direct mutable access to the reserved payload bytes.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the unique grant owns this unpublished slot; payload bounds
        // were checked at reservation.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.slot.cast::<u8>().add(SLOT_HEADER_BYTES),
                self.length,
            )
        }
    }

    /// Publish this slot.  On epoch change nothing is published.
    pub fn commit(self) -> Result<(), SendError> {
        self.producer.raw.validate_epoch()?;
        // SAFETY: this grant exclusively owns the unpublished slot.  Header
        // writes become visible through the following Release tail store.
        unsafe {
            (*self.slot).epoch = self.producer.raw.expected_epoch.0;
            (*self.slot).length = self.length as u32;
            (*self.slot).reserved = 0;
        }
        self.producer
            .raw
            .tail()
            .store(self.tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl<'window> Consumer<'window> {
    /// Borrow the next published record.  Dropping without `release` leaves
    /// the record pending, which permits bounded retry by the caller.
    pub fn try_read<'consumer>(
        &'consumer mut self,
    ) -> Result<ReadGrant<'consumer, 'window>, ReceiveError> {
        self.raw.validate_epoch()?;
        let head = self.raw.head().load(Ordering::Relaxed);
        let tail = self.raw.tail().load(Ordering::Acquire);
        let outstanding = tail.wrapping_sub(head);
        if outstanding > self.raw.capacity {
            return Err(ReceiveError::CorruptCursor);
        }
        if outstanding == 0 {
            return Err(ReceiveError::Empty);
        }
        // SAFETY: Acquire observed a published tail beyond head, assigning
        // this slot exclusively to the consumer until head is released.
        let slot = unsafe { self.raw.slot(head) };
        // SAFETY: slot publication happens-before the Acquire tail load.
        let slot_header = unsafe { &*slot };
        if slot_header.epoch != self.raw.expected_epoch.0 {
            return Err(ReceiveError::StaleSlot {
                expected: self.raw.expected_epoch,
                observed: slot_header.epoch,
            });
        }
        if slot_header.length as usize > self.raw.payload_bytes {
            return Err(ReceiveError::CorruptLength {
                observed: slot_header.length,
                maximum: self.raw.payload_bytes as u32,
            });
        }
        Ok(ReadGrant {
            consumer: self,
            slot,
            head,
            length: slot_header.length as usize,
        })
    }

    /// Copy and release one byte record.
    pub fn try_pop(&mut self, output: &mut [u8]) -> Result<usize, ReceiveError> {
        let grant = self.try_read()?;
        let length = grant.len();
        if output.len() < length {
            return Err(ReceiveError::OutputTooSmall {
                required: length,
                available: output.len(),
            });
        }
        output[..length].copy_from_slice(grant.as_slice());
        grant.release()?;
        Ok(length)
    }

    /// Copy a bounded portion of the current record without releasing it
    /// until the final byte is returned. This adapts an atomic shared record
    /// to a smaller stream decoder buffer without allocating or publishing a
    /// second cursor. The caller owns `offset` and must advance it by exactly
    /// `copied`; a stale or forged offset fails closed.
    pub fn try_pop_chunk(
        &mut self,
        output: &mut [u8],
        offset: usize,
    ) -> Result<ReadChunk, ReceiveError> {
        let grant = self.try_read()?;
        let total = grant.len();
        if offset > total {
            return Err(ReceiveError::InvalidOffset {
                requested: offset,
                length: total,
            });
        }
        if output.is_empty() && offset < total {
            return Err(ReceiveError::OutputTooSmall {
                required: 1,
                available: 0,
            });
        }
        let copied = core::cmp::min(output.len(), total - offset);
        output[..copied].copy_from_slice(&grant.as_slice()[offset..offset + copied]);
        let complete = offset + copied == total;
        if complete {
            grant.release()?;
        }
        Ok(ReadChunk {
            copied,
            total,
            complete,
        })
    }

    #[must_use]
    pub const fn payload_bytes(&self) -> usize {
        self.raw.payload_bytes
    }

    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.raw.capacity as u32
    }
}

/// An affine published consumer slot.
pub struct ReadGrant<'consumer, 'window> {
    consumer: &'consumer mut Consumer<'window>,
    slot: *mut SlotHeader,
    head: u64,
    length: usize,
}

impl ReadGrant<'_, '_> {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Direct access to the published payload bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the grant owns this published slot until release, and the
        // producer cannot reuse it before the head Release.
        unsafe {
            core::slice::from_raw_parts(self.slot.cast::<u8>().add(SLOT_HEADER_BYTES), self.length)
        }
    }

    /// Release this slot for producer reuse.
    pub fn release(self) -> Result<(), ReceiveError> {
        self.consumer.raw.validate_epoch()?;
        self.consumer
            .raw
            .head()
            .store(self.head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

fn validate_storage(base: *mut u8, len: usize, layout: Layout) -> Result<(), WindowError> {
    if base.align_offset(ALIGN) != 0 {
        return Err(WindowError::Misaligned);
    }
    if len < layout.window_bytes {
        return Err(WindowError::TooSmall);
    }
    Ok(())
}

fn inspect(base: *mut u8, len: usize) -> Result<Layout, WindowError> {
    if base.align_offset(ALIGN) != 0 {
        return Err(WindowError::Misaligned);
    }
    if len < WINDOW_CONTROL_BYTES {
        return Err(WindowError::TooSmall);
    }
    // SAFETY: header is aligned and in bounds.
    let header = unsafe { &*base.cast::<Header>() };
    if header.magic != MAGIC
        || header.abi != ABI
        || header.header_bytes as usize != HEADER_BYTES
        || header.reserved32 != 0
        || header.reserved64 != [0; 2]
    {
        return Err(WindowError::BadHeader);
    }
    let layout =
        Layout::new(header.capacity, header.payload_bytes).map_err(|_| WindowError::BadHeader)?;
    if header.window_bytes != layout.window_bytes as u64
        || header.lane_bytes != layout.lane_bytes as u64
        || header.slot_stride != layout.slot_stride as u32
    {
        return Err(WindowError::BadHeader);
    }
    validate_storage(base, len, layout)?;
    Ok(layout)
}

unsafe fn bind_raw<'a>(
    base: *mut u8,
    len: usize,
    expected_epoch: Epoch,
    side: Side,
) -> Result<Endpoint<'a>, WindowError> {
    let layout = inspect(base, len)?;
    // SAFETY: inspect validated epoch address and lane geometry.
    let epoch = unsafe { base.add(EPOCH_AT).cast::<AtomicU64>() };
    let observed = unsafe { &*epoch }.load(Ordering::Acquire);
    if observed != expected_epoch.0 {
        return Err(WindowError::EpochMismatch {
            expected: expected_epoch,
            observed,
        });
    }
    let lane = |index| {
        // SAFETY: layout bounds validated by inspect.
        let lane = unsafe { base.add(lane_offset(layout, index)) };
        RawLane {
            epoch,
            // SAFETY: lane controls are aligned and within the lane.
            head: unsafe { lane.add(LANE_HEAD_AT).cast::<AtomicU64>() },
            tail: unsafe { lane.add(LANE_TAIL_AT).cast::<AtomicU64>() },
            // SAFETY: slot offset and full slot array are in bounds.
            slots: unsafe { lane.add(LANE_SLOTS_AT) },
            expected_epoch,
            capacity: layout.capacity as u64,
            mask: layout.capacity as u64 - 1,
            payload_bytes: layout.payload_bytes as usize,
            slot_stride: layout.slot_stride,
        }
    };
    let (tx, rx) = match side {
        Side::A => (lane(0), lane(1)),
        Side::B => (lane(1), lane(0)),
    };
    Ok(Endpoint {
        tx: Producer {
            raw: tx,
            _window: PhantomData,
        },
        rx: Consumer {
            raw: rx,
            _window: PhantomData,
        },
    })
}

const fn lane_offset(layout: Layout, index: usize) -> usize {
    WINDOW_CONTROL_BYTES + layout.lane_bytes * index
}

unsafe fn epoch_atomic<'window>(base: *mut u8) -> &'window AtomicU64 {
    // SAFETY: callers first validate the formatted, aligned control region.
    unsafe { &*base.add(EPOCH_AT).cast::<AtomicU64>() }
}

unsafe fn atomic_at<'window>(base: *mut u8, offset: usize) -> &'window AtomicU64 {
    // SAFETY: callers use validated aligned lane control offsets.
    unsafe { &*base.add(offset).cast::<AtomicU64>() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    const EPOCH_1: Epoch = Epoch(1);
    const EPOCH_2: Epoch = Epoch(2);
    const TEST_BYTES: usize = 8192;

    fn storage() -> Aligned<TEST_BYTES> {
        Aligned([0; TEST_BYTES])
    }

    #[test]
    fn geometry_is_checked_and_exact() {
        assert_eq!(Layout::new(0, 64), Err(GeometryError::BadCapacity));
        assert_eq!(Layout::new(3, 64), Err(GeometryError::BadCapacity));
        assert_eq!(Layout::new(4, 0), Err(GeometryError::EmptyPayload));
        assert_eq!(Layout::new(1, u32::MAX), Err(GeometryError::Overflow));
        let layout = Layout::new(8, 257).unwrap();
        assert_eq!(layout.capacity(), 8);
        assert_eq!(layout.payload_bytes(), 257);
        assert_eq!(layout.lane_bytes() % ALIGN, 0);
        assert_eq!(
            layout.window_bytes(),
            WINDOW_CONTROL_BYTES + 2 * layout.lane_bytes()
        );
    }

    #[test]
    fn storage_alignment_and_bounds_are_enforced() {
        let layout = Layout::new(4, 64).unwrap();
        let mut bytes = storage();
        assert_eq!(
            init(&mut bytes.0[1..], layout, EPOCH_1),
            Err(WindowError::Misaligned)
        );
        assert_eq!(
            init(&mut bytes.0[..layout.window_bytes() - 1], layout, EPOCH_1),
            Err(WindowError::TooSmall)
        );
    }

    #[test]
    fn both_directions_preserve_variable_length_records_and_wrap() {
        let layout = Layout::new(4, 64).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        let mut output = [0u8; 64];

        for sequence in 0..100u8 {
            let a_len = sequence as usize % 65;
            let b_len = (sequence as usize * 7) % 65;
            let mut a_payload = [0u8; 64];
            let mut b_payload = [0u8; 64];
            for (i, byte) in a_payload[..a_len].iter_mut().enumerate() {
                *byte = sequence.wrapping_add(i as u8);
            }
            for (i, byte) in b_payload[..b_len].iter_mut().enumerate() {
                *byte = sequence.wrapping_mul(3).wrapping_add(i as u8);
            }

            a.tx().try_push(&a_payload[..a_len]).unwrap();
            assert_eq!(b.rx().try_pop(&mut output).unwrap(), a_len);
            assert_eq!(&output[..a_len], &a_payload[..a_len]);

            b.tx().try_push(&b_payload[..b_len]).unwrap();
            assert_eq!(a.rx().try_pop(&mut output).unwrap(), b_len);
            assert_eq!(&output[..b_len], &b_payload[..b_len]);
        }
    }

    #[test]
    fn full_is_bounded_and_cancelled_grants_publish_nothing() {
        let layout = Layout::new(4, 8).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        {
            let mut cancelled = a.tx().try_reserve(3).unwrap();
            cancelled.as_mut_slice().copy_from_slice(b"old");
        }
        assert!(matches!(b.rx().try_read(), Err(ReceiveError::Empty)));

        for byte in 0..4 {
            a.tx().try_push(&[byte]).unwrap();
        }
        assert_eq!(a.tx().try_push(b"x"), Err(SendError::Full));
        let mut output = [0; 8];
        for byte in 0..4 {
            assert_eq!(b.rx().try_pop(&mut output).unwrap(), 1);
            assert_eq!(output[0], byte);
        }
        assert!(matches!(b.rx().try_read(), Err(ReceiveError::Empty)));
    }

    #[test]
    fn oversize_and_small_output_do_not_advance_cursors() {
        let layout = Layout::new(2, 8).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        assert_eq!(
            a.tx().try_push(&[0; 9]),
            Err(SendError::TooLarge {
                requested: 9,
                maximum: 8
            })
        );
        a.tx().try_push(b"12345678").unwrap();
        assert_eq!(
            b.rx().try_pop(&mut [0; 7]),
            Err(ReceiveError::OutputTooSmall {
                required: 8,
                available: 7
            })
        );
        let mut exact = [0; 8];
        assert_eq!(b.rx().try_pop(&mut exact).unwrap(), 8);
        assert_eq!(&exact, b"12345678");
    }

    #[test]
    fn chunked_copy_releases_only_after_the_exact_final_byte() {
        let layout = Layout::new(2, 63).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        let payload = b"a complete record larger than seven";
        a.tx().try_push(payload).unwrap();
        let mut output = [0u8; 64];
        let mut offset = 0;
        let mut queued = false;
        while offset < payload.len() {
            let chunk = b
                .rx()
                .try_pop_chunk(&mut output[offset..offset + 7], offset)
                .unwrap();
            assert_eq!(chunk.total, payload.len());
            assert_eq!(chunk.complete, offset + chunk.copied == payload.len());
            if !chunk.complete && !queued {
                assert_eq!(a.tx().try_push(b"blocked"), Ok(()));
                queued = true;
            }
            if !chunk.complete {
                assert_eq!(a.tx().try_push(b"full"), Err(SendError::Full));
            }
            offset += chunk.copied;
        }
        assert_eq!(&output[..payload.len()], payload);
        let mut next = [0u8; 8];
        assert_eq!(b.rx().try_pop(&mut next).unwrap(), 7);
        assert_eq!(&next[..7], b"blocked");
    }

    #[test]
    fn chunked_copy_rejects_forged_offset_without_releasing() {
        let layout = Layout::new(1, 8).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        a.tx().try_push(b"record").unwrap();
        assert_eq!(
            b.rx().try_pop_chunk(&mut [0; 2], 7),
            Err(ReceiveError::InvalidOffset {
                requested: 7,
                length: 6,
            })
        );
        let mut exact = [0; 6];
        assert_eq!(b.rx().try_pop(&mut exact), Ok(6));
        assert_eq!(&exact, b"record");
    }

    #[test]
    fn chunked_copy_rejects_zero_progress_without_releasing() {
        let layout = Layout::new(1, 8).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        a.tx().try_push(b"record").unwrap();

        assert_eq!(
            b.rx().try_pop_chunk(&mut [], 0),
            Err(ReceiveError::OutputTooSmall {
                required: 1,
                available: 0,
            })
        );

        let mut exact = [0; 6];
        assert_eq!(b.rx().try_pop(&mut exact), Ok(6));
        assert_eq!(&exact, b"record");
    }

    #[test]
    fn empty_record_completes_with_empty_output() {
        let layout = Layout::new(1, 8).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        a.tx().try_push(&[]).unwrap();

        assert_eq!(
            b.rx().try_pop_chunk(&mut [], 0),
            Ok(ReadChunk {
                copied: 0,
                total: 0,
                complete: true,
            })
        );
        assert!(matches!(b.rx().try_read(), Err(ReceiveError::Empty)));
    }

    #[test]
    fn reset_changes_epoch_empties_lanes_and_rejects_stale_attach() {
        let layout = Layout::new(4, 32).unwrap();
        let mut bytes = storage();
        {
            let (mut a, _b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
            a.tx().try_push(b"stale").unwrap();
        }
        assert_eq!(reset(&mut bytes.0, EPOCH_2).unwrap(), layout);
        assert_eq!(reset(&mut bytes.0, EPOCH_2), Err(WindowError::SameEpoch));
        assert!(matches!(
            endpoint(&mut bytes.0, EPOCH_1, Side::A),
            Err(WindowError::EpochMismatch {
                expected: EPOCH_1,
                observed: 2
            })
        ));
        let mut b = endpoint(&mut bytes.0, EPOCH_2, Side::B).unwrap();
        assert!(matches!(b.rx().try_read(), Err(ReceiveError::Empty)));
    }

    #[test]
    fn stale_epoch_and_corrupt_length_fail_closed() {
        let layout = Layout::new(2, 8).unwrap();
        let mut bytes = storage();
        let base = bytes.0.as_mut_ptr();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        a.tx().try_push(b"valid").unwrap();

        // Simulate an untrusted peer corrupting the published slot length.
        // SAFETY: test-only adversarial write while neither endpoint holds a
        // grant; the tail has already published slot zero.
        unsafe {
            let slot = base
                .add(lane_offset(layout, 0) + LANE_SLOTS_AT)
                .cast::<SlotHeader>();
            (*slot).length = 9;
        }
        assert!(matches!(
            b.rx().try_read(),
            Err(ReceiveError::CorruptLength {
                observed: 9,
                maximum: 8
            })
        ));

        // A remote control plane can invalidate a quiescent endpoint.  The
        // next operation observes the Acquire-loaded epoch and refuses.
        // SAFETY: the epoch atomic was initialized and is aligned.
        unsafe { &*base.add(EPOCH_AT).cast::<AtomicU64>() }.store(2, Ordering::Release);
        assert!(matches!(
            a.tx().try_push(b"x"),
            Err(SendError::EpochChanged(EpochChanged {
                expected: EPOCH_1,
                observed: 2
            }))
        ));
    }

    #[test]
    fn impossible_cursor_distance_fails_closed() {
        let layout = Layout::new(4, 8).unwrap();
        let mut bytes = storage();
        let base = bytes.0.as_mut_ptr();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();

        // Simulate corrupt device/peer metadata: tail cannot be more than one
        // capacity ahead of head in a bounded SPSC lane.
        // SAFETY: initialized, aligned lane tail atomic.
        unsafe {
            &*base
                .add(lane_offset(layout, 0) + LANE_TAIL_AT)
                .cast::<AtomicU64>()
        }
        .store(5, Ordering::Release);
        assert_eq!(a.tx().try_push(b"x"), Err(SendError::CorruptCursor));
        assert!(matches!(
            b.rx().try_read(),
            Err(ReceiveError::CorruptCursor)
        ));
    }

    #[test]
    fn deterministic_property_sweep_covers_lengths_and_queue_depths() {
        let layout = Layout::new(8, 63).unwrap();
        let mut bytes = storage();
        let (mut a, mut b) = split(&mut bytes.0, layout, EPOCH_1).unwrap();
        let mut expected = [[0u8; 63]; 8];
        let mut lengths = [0usize; 8];
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        let mut output = [0u8; 63];

        for round in 0..512usize {
            let depth = round % 9;
            for item in 0..depth {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let length = (seed as usize) % 64;
                lengths[item] = length;
                for (index, byte) in expected[item][..length].iter_mut().enumerate() {
                    *byte = (seed >> (index % 8)) as u8 ^ index as u8;
                }
                a.tx().try_push(&expected[item][..length]).unwrap();
            }
            if depth == 8 {
                assert_eq!(a.tx().try_push(b"full"), Err(SendError::Full));
            }
            for item in 0..depth {
                let length = b.rx().try_pop(&mut output).unwrap();
                assert_eq!(length, lengths[item]);
                assert_eq!(&output[..length], &expected[item][..length]);
            }
            assert!(matches!(b.rx().try_read(), Err(ReceiveError::Empty)));
        }
    }
}
