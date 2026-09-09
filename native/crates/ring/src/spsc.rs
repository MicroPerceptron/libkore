//! `spsc` — the typed single-producer/single-consumer ring over a shared
//! window (`res/USERLAND.md` §3.2).
//!
//! A ring is a **data structure in an mm-shared window**, never a kernel
//! object: fixed capacity, cache-line-separated head/tail atomics, a
//! `dropped` counter, and a magic+ABI header validated on attach
//! (fail-closed — a window that does not carry a well-formed ring of the
//! expected record type refuses).  The #299 "KINP" input queue is this
//! shape and survives as its first production instance (§5.8).
//!
//! ## Layout (frozen within `SPSC_ABI`)
//!
//! ```text
//! offset   0: Header { magic u64 ‖ abi u32 ‖ capacity u32 ‖ rec_size u32 ‖ … }
//! offset  64: head    (AtomicU32 — the consumer's cursor, own cache line)
//! offset 128: tail    (AtomicU32) ‖ dropped (AtomicU64)   (the producer's line)
//! offset 192: capacity × rec_size record slots
//! ```
//!
//! Head and tail are free-running `u32` cursors (masked on index, compared
//! by wrapping subtraction), so empty is `head == tail` and full is
//! `tail - head == capacity` with no wasted slot.  The producer writes a
//! slot **then** publishes `tail` with `Release`; the consumer reads `tail`
//! with `Acquire` **then** the slot — the classic SPSC visibility proof.
//!
//! ## The wake edge is not here (§4)
//!
//! The producer kick is a **doorbell send**, not a ring op — the caller
//! owns kicking (`push` is sync/lock-free; the kick is an async
//! `ChanTx::send` the pusher issues after its batch).  Symmetrically,
//! [`Consumer::next`] performs no waker plumbing of its own: it is
//! level-triggered — the kore executor re-polls every source-awaiting task
//! on each doorbell cycle (`res/USERLAND.md` §3.1 drain-all), so an empty
//! `next` resolves on the cycle after the producer's kick.  No capability
//! ever rides a ring (standing doctrine): records are bytes and indices.
//!
//! ## Allocation-free
//!
//! The ring lives entirely in the caller-provided window; this module
//! allocates nothing.  [`init`] formats a window; [`split`] hands back the
//! two halves over one exclusively-borrowed window (the same-space / host
//! form); [`producer`]/[`consumer`] attach one half each (the cross-space
//! form — each side attaches over its own mapping of the shared lease).

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use core::task::Poll;

use zerocopy::{FromBytes, Immutable, IntoBytes};

/// A record that may ride a shared-window ring: plain copyable bytes,
/// any-bit-pattern-valid on read (the window is shared memory — the peer
/// is untrusted, so a record must decode from any bytes), and
/// reference-free by construction ([`IntoBytes`] + [`Immutable`]).
/// Blanket-implemented; `#[derive(FromBytes, IntoBytes, Immutable)]` on a
/// `Copy` wire struct is the whole opt-in.
pub trait RingRecord: Copy + FromBytes + IntoBytes + Immutable {}

impl<T: Copy + FromBytes + IntoBytes + Immutable> RingRecord for T {}

/// Ring magic — "koreSPSC", LE.
pub const SPSC_MAGIC: u64 = 0x4353_5053_6572_6F6B;

/// The spsc layout ABI this module writes and accepts.
pub const SPSC_ABI: u32 = 1;

/// Byte offset of the consumer cursor (its own cache line).
const HEAD_AT: usize = 64;
/// Byte offset of the producer cursor + dropped counter (their line).
const TAIL_AT: usize = 128;
const DROPPED_AT: usize = 136;
/// Byte offset of the first record slot.
const SLOTS_AT: usize = 192;
/// Required window alignment (a cache line — also the header's unit).
const ALIGN: usize = 64;

/// Bytes a ring of `capacity` records of `T` occupies in its window —
/// header + cursors + slots, rounded up to the 64-byte layout unit (so
/// adjacent rings packed in one shared window each start cache-line
/// aligned; the `svc` request path packs a request + reply ring this
/// way, `res/USERLAND.md` §5.3).  Geometry only — [`init`] still
/// validates capacity/alignment when the window is formatted.
pub const fn region_bytes<T>(capacity: u32) -> usize {
    let raw = SLOTS_AT + capacity as usize * size_of::<T>();
    (raw + (ALIGN - 1)) & !(ALIGN - 1)
}

/// Why a window refused to become (or bind as) a ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    /// The window cannot hold the header + `capacity` records.
    TooSmall,
    /// The window base is not 64-byte aligned (the cache-line separation
    /// and slot alignment both rest on it).
    Misaligned,
    /// `capacity` is zero or not a power of two (the mask requirement).
    BadCapacity,
    /// The record type's alignment exceeds the layout's 64-byte unit.
    BadRecord,
    /// Attach: the header's magic/ABI is not a ring this module wrote.
    BadHeader,
    /// Attach: the header's record size does not match `T` — the window
    /// carries a ring of some *other* type; binding would misread it.
    WrongRecord,
}

/// The ring header (offset 0).  Written once by [`init`], validated by
/// every attach; never touched by push/pop.
#[repr(C)]
struct Header {
    magic: u64,
    abi: u32,
    capacity: u32,
    rec_size: u32,
}

/// The raw ring binding one side holds: validated pointers into the
/// window.  Copy-free plumbing shared by both halves.
struct Raw<T> {
    head: *const AtomicU32,
    tail: *const AtomicU32,
    dropped: *const AtomicU64,
    slots: *mut T,
    mask: u32,
    capacity: u32,
}

impl<T: RingRecord> Raw<T> {
    /// Validate `window` as a ring of `T` and bind the pointers.
    ///
    /// # Safety
    /// `base..base+len` must stay valid (and unmoved) for the binding's
    /// life, and no third party may touch the bytes except through the
    /// ring protocol (one producer, one consumer).
    unsafe fn bind(base: *mut u8, len: usize) -> Result<Self, RingError> {
        if base.align_offset(ALIGN) != 0 {
            return Err(RingError::Misaligned);
        }
        if align_of::<T>() > ALIGN {
            return Err(RingError::BadRecord);
        }
        if len < SLOTS_AT {
            return Err(RingError::TooSmall);
        }
        // SAFETY: caller contract — `base` valid for `len >= SLOTS_AT`
        // bytes, 64-aligned, so the header read is in bounds and aligned.
        let header = unsafe { &*(base as *const Header) };
        if header.magic != SPSC_MAGIC || header.abi != SPSC_ABI {
            return Err(RingError::BadHeader);
        }
        if header.rec_size as usize != size_of::<T>() {
            return Err(RingError::WrongRecord);
        }
        let capacity = header.capacity;
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(RingError::BadHeader);
        }
        let need = SLOTS_AT
            .checked_add(
                (capacity as usize)
                    .checked_mul(size_of::<T>())
                    .ok_or(RingError::TooSmall)?,
            )
            .ok_or(RingError::TooSmall)?;
        if len < need {
            return Err(RingError::TooSmall);
        }
        // SAFETY: offsets are in bounds (`len >= need > SLOTS_AT`) and
        // aligned (64-aligned base; the offsets are multiples of 8/64).
        Ok(unsafe {
            Self {
                head: base.add(HEAD_AT) as *const AtomicU32,
                tail: base.add(TAIL_AT) as *const AtomicU32,
                dropped: base.add(DROPPED_AT) as *const AtomicU64,
                slots: base.add(SLOTS_AT) as *mut T,
                mask: capacity - 1,
                capacity,
            }
        })
    }

    #[inline]
    fn head(&self) -> &AtomicU32 {
        // SAFETY: validated in-bounds pointer; the atomic is interior-
        // mutable shared state — a shared reference is exactly its API.
        unsafe { &*self.head }
    }

    #[inline]
    fn tail(&self) -> &AtomicU32 {
        // SAFETY: as `head`.
        unsafe { &*self.tail }
    }

    #[inline]
    fn dropped(&self) -> &AtomicU64 {
        // SAFETY: as `head`.
        unsafe { &*self.dropped }
    }
}

/// Format `window` as an empty ring of `capacity` records of `T` —
/// exclusive access (the window is not yet shared / not yet attached).
/// `capacity` must be a power of two; the window must be 64-byte aligned
/// and large enough (`192 + capacity × size_of::<T>()`).
pub fn init<T: RingRecord>(window: &mut [u8], capacity: u32) -> Result<(), RingError> {
    if capacity == 0 || !capacity.is_power_of_two() {
        return Err(RingError::BadCapacity);
    }
    if window.as_ptr().align_offset(ALIGN) != 0 {
        return Err(RingError::Misaligned);
    }
    if align_of::<T>() > ALIGN {
        return Err(RingError::BadRecord);
    }
    let need = SLOTS_AT
        .checked_add(
            (capacity as usize)
                .checked_mul(size_of::<T>())
                .ok_or(RingError::TooSmall)?,
        )
        .ok_or(RingError::TooSmall)?;
    if window.len() < need {
        return Err(RingError::TooSmall);
    }
    // Zero the control area (cursors + dropped start clean), then write
    // the header last — a partially-formatted window never carries the
    // magic.
    window[..SLOTS_AT].fill(0);
    let header = Header {
        magic: SPSC_MAGIC,
        abi: SPSC_ABI,
        capacity,
        rec_size: size_of::<T>() as u32,
    };
    // SAFETY: bounds + alignment validated above; Header is plain data.
    unsafe { core::ptr::write(window.as_mut_ptr() as *mut Header, header) };
    Ok(())
}

/// The producer half.  `push` is sync and lock-free; the wake edge (the
/// doorbell kick) is the **caller's**, issued after its batch (§4).
pub struct Producer<'a, T: RingRecord> {
    raw: Raw<T>,
    _window: PhantomData<&'a ()>,
}

/// The consumer half.  `pop` is sync and lock-free; [`next`](Self::next)
/// is the async register-and-await face (level-triggered — see the
/// module docs).
pub struct Consumer<'a, T: RingRecord> {
    raw: Raw<T>,
    _window: PhantomData<&'a ()>,
}

/// The ring is full — the record was **not** enqueued.  Whether to retry,
/// drop (and count — [`Producer::push_lossy`]), or backpressure is the
/// producer's policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

/// Attach the **producer** half of an already-[`init`]ialized ring in
/// `window` — the cross-space form (this side's own mapping of the shared
/// lease; the consumer attaches its own).  Exclusive borrow: nothing else
/// on this side touches the window while the half lives.
pub fn producer<T: RingRecord>(window: &mut [u8]) -> Result<Producer<'_, T>, RingError> {
    // SAFETY: the borrow pins the window for 'a; the ring protocol is the
    // only access this module performs.
    let raw = unsafe { Raw::bind(window.as_mut_ptr(), window.len()) }?;
    Ok(Producer {
        raw,
        _window: PhantomData,
    })
}

/// Attach the **consumer** half — [`producer`]'s mirror.
pub fn consumer<T: RingRecord>(window: &mut [u8]) -> Result<Consumer<'_, T>, RingError> {
    // SAFETY: as `producer`.
    let raw = unsafe { Raw::bind(window.as_mut_ptr(), window.len()) }?;
    Ok(Consumer {
        raw,
        _window: PhantomData,
    })
}

/// [`init`] + both halves over one exclusively-borrowed window — the
/// same-address-space (and host-test) form.  The two halves alias the
/// window through the ring protocol only (disjoint cursors, slot
/// hand-off by Release/Acquire), which is exactly what makes the split
/// sound.
pub fn split<T: RingRecord>(
    window: &mut [u8],
    capacity: u32,
) -> Result<(Producer<'_, T>, Consumer<'_, T>), RingError> {
    init::<T>(window, capacity)?;
    let (base, len) = (window.as_mut_ptr(), window.len());
    // SAFETY: the `&mut` borrow pins the window for 'a; both halves
    // access it only through the SPSC protocol (see the type docs).
    let p = unsafe { Raw::bind(base, len) }?;
    let c = unsafe { Raw::bind(base, len) }?;
    Ok((
        Producer {
            raw: p,
            _window: PhantomData,
        },
        Consumer {
            raw: c,
            _window: PhantomData,
        },
    ))
}

impl<T: RingRecord> Producer<'_, T> {
    /// Publish an entire contiguous batch with one release-store, or none.
    /// This preserves input-frame boundaries without locks or allocation.
    pub fn push_slice(&mut self, records: &[T]) -> Result<(), Full> {
        if records.len() > self.raw.capacity as usize {
            return Err(Full);
        }
        let tail = self.raw.tail().load(Ordering::Relaxed);
        let head = self.raw.head().load(Ordering::Acquire);
        let used = tail.wrapping_sub(head);
        if used > self.raw.capacity || records.len() > (self.raw.capacity - used) as usize {
            return Err(Full);
        }
        for (offset, record) in records.iter().enumerate() {
            // SAFETY: admission reserved every slot; the single producer owns
            // them until the one tail publication makes all writes visible.
            unsafe {
                self.raw
                    .slots
                    .add((tail.wrapping_add(offset as u32) & self.raw.mask) as usize)
                    .write(*record);
            }
        }
        self.raw
            .tail()
            .store(tail.wrapping_add(records.len() as u32), Ordering::Release);
        Ok(())
    }

    /// Atomic batch publication with loss counted in records, not batches.
    pub fn push_slice_lossy(&mut self, records: &[T]) -> bool {
        if self.push_slice(records).is_ok() {
            true
        } else {
            self.raw
                .dropped()
                .fetch_add(records.len() as u64, Ordering::Relaxed);
            false
        }
    }

    /// Enqueue one record — sync, lock-free, allocation-free.
    /// [`Full`] when `capacity` records are outstanding; the record is
    /// not enqueued and the caller owns the policy.
    pub fn push(&mut self, rec: T) -> Result<(), Full> {
        let tail = self.raw.tail().load(Ordering::Relaxed); // own cursor
        let head = self.raw.head().load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= self.raw.capacity {
            return Err(Full);
        }
        // SAFETY: `tail & mask` indexes inside the validated slot array,
        // and SPSC ownership makes this slot exclusively the producer's
        // until the tail publish below hands it over.
        unsafe {
            self.raw
                .slots
                .add((tail & self.raw.mask) as usize)
                .write(rec)
        };
        // Publish: the slot write happens-before any consumer that
        // Acquire-reads this tail.
        self.raw
            .tail()
            .store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// [`push`](Self::push), dropping (and counting) on a full ring — the
    /// lossy-telemetry idiom (the KINP input queue's posture: under a
    /// stalled consumer, drop the oldest *news*, never block the
    /// producer).  Returns whether the record was enqueued.
    pub fn push_lossy(&mut self, rec: T) -> bool {
        match self.push(rec) {
            Ok(()) => true,
            Err(Full) => {
                self.raw.dropped().fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Records dropped by [`push_lossy`](Self::push_lossy) so far.
    pub fn dropped(&self) -> u64 {
        self.raw.dropped().load(Ordering::Relaxed)
    }

    /// The ring's record capacity.
    pub fn capacity(&self) -> u32 {
        self.raw.capacity
    }

    /// Free producer slots at this instant.
    ///
    /// This is advisory under a live consumer: the consumer may publish a
    /// newer head immediately after the read. [`push`](Self::push) remains the
    /// authoritative admission check.
    pub fn capacity_left(&self) -> u32 {
        let tail = self.raw.tail().load(Ordering::Relaxed); // own cursor
        let head = self.raw.head().load(Ordering::Acquire);
        self.raw.capacity.saturating_sub(tail.wrapping_sub(head))
    }
}

impl<T: RingRecord> Consumer<'_, T> {
    /// Inspect the next published record without advancing the consumer.
    ///
    /// This is intentionally restricted to the ring's `Copy` record
    /// vocabulary.  It is used by service admission to keep a request
    /// invisible until the capability transaction that authorizes it has
    /// committed, while preserving the single consumer's ownership of the
    /// eventual dequeue.
    pub fn peek(&self) -> Option<T> {
        let head = self.raw.head().load(Ordering::Relaxed); // own cursor
        let tail = self.raw.tail().load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        // SAFETY: the Acquire observes the producer's Release publication;
        // `RingRecord: Copy`, and the consumer alone may later advance head.
        Some(unsafe { *self.raw.slots.add((head & self.raw.mask) as usize) })
    }

    /// Inspect the record `ahead` positions past the head, without
    /// advancing.  `peek_at(0)` is [`peek`](Self::peek).
    ///
    /// **What this is for.** A consumer that hands each record to hardware
    /// cannot release its slot at dequeue: the device reads the slot's
    /// payload long after the descriptor was read, so advancing the head —
    /// which is what tells the producer the slot is free — would let the
    /// producer overwrite bytes a transfer is still fetching.  Such a
    /// consumer therefore pops **on completion**, and needs to look past
    /// the records still in flight to find the next one to submit.  With
    /// only [`peek`](Self::peek) that consumer is limited to one
    /// outstanding transfer at a time.
    ///
    /// `None` once `ahead` reaches [`outstanding`](Self::outstanding) —
    /// there is nothing published there yet.  Like `outstanding`, the
    /// bound is advisory upward under a live producer (more may arrive
    /// immediately after); it is never advisory downward, because only
    /// this consumer moves the head.
    pub fn peek_at(&self, ahead: u32) -> Option<T> {
        let head = self.raw.head().load(Ordering::Relaxed); // own cursor
        let tail = self.raw.tail().load(Ordering::Acquire);
        if tail.wrapping_sub(head) <= ahead {
            return None;
        }
        let at = head.wrapping_add(ahead);
        // SAFETY: `at` is strictly inside `[head, tail)`, so the slot was
        // published before the Acquire above; `RingRecord: Copy`, and the
        // consumer alone may later advance head past it.
        Some(unsafe { *self.raw.slots.add((at & self.raw.mask) as usize) })
    }

    /// Dequeue one record — sync, lock-free, allocation-free; `None` on
    /// an empty ring.
    pub fn pop(&mut self) -> Option<T> {
        let head = self.raw.head().load(Ordering::Relaxed); // own cursor
        let tail = self.raw.tail().load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        // SAFETY: a published slot (tail Acquire pairs with the
        // producer's Release, so the record bytes are visible), owned by
        // the consumer until the head publish below returns it.
        let rec = unsafe { self.raw.slots.add((head & self.raw.mask) as usize).read() };
        self.raw
            .head()
            .store(head.wrapping_add(1), Ordering::Release);
        Some(rec)
    }

    /// Await the next record: [`pop`](Self::pop) now if one is ready,
    /// else resolve on a later poll (§4's register-and-await).
    ///
    /// **Level-triggered by design** (the module docs): the producer's
    /// kick is a doorbell notice, and the kore executor's §3.1 cycle
    /// re-polls every source-awaiting task on each doorbell delivery —
    /// that re-poll is the registration this future relies on.  Under an
    /// edge-triggered foreign executor an empty `next` would not be
    /// re-polled; this future is for the kore runtime.
    pub async fn next(&mut self) -> T {
        core::future::poll_fn(|_cx| match self.pop() {
            Some(rec) => Poll::Ready(rec),
            None => Poll::Pending,
        })
        .await
    }

    /// Records the producer dropped ([`Producer::push_lossy`]) so far —
    /// the consumer-side telemetry read.
    pub fn dropped(&self) -> u64 {
        self.raw.dropped().load(Ordering::Relaxed)
    }

    /// Records published but not yet consumed at this instant — the
    /// consumer-side occupancy read ([`Producer::capacity_left`]'s
    /// mirror).  Advisory under a live producer (a newer tail may publish
    /// immediately after); [`pop`](Self::pop) remains the authoritative
    /// dequeue.
    pub fn outstanding(&self) -> u32 {
        let head = self.raw.head().load(Ordering::Relaxed); // own cursor
        let tail = self.raw.tail().load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    /// The ring's record capacity.
    pub fn capacity(&self) -> u32 {
        self.raw.capacity
    }
}

// SAFETY: a half may move to (or be driven from) another thread — the
// cross-space reality it models: all shared state is reached through
// atomics with the SPSC Release/Acquire protocol; `T: RingRecord` is
// plain bytes (no references).
unsafe impl<T: RingRecord> Send for Producer<'_, T> {}
unsafe impl<T: RingRecord> Send for Consumer<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64-aligned test window (a stack/heap stand-in for the mm-shared
    /// lease window).
    #[repr(C, align(64))]
    struct Window<const N: usize>([u8; N]);

    impl<const N: usize> Window<N> {
        fn new() -> Self {
            Self([0u8; N])
        }
    }

    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Eq,
        zerocopy::FromBytes,
        zerocopy::IntoBytes,
        zerocopy::Immutable,
    )]
    #[repr(C)]
    struct Rec {
        seq: u64,
        val: u32,
        _pad: u32,
    }

    fn rec(seq: u64, val: u32) -> Rec {
        Rec { seq, val, _pad: 0 }
    }

    /// Round trip: push N, pop N in order; empty pops None; capacity and
    /// cursors wrap cleanly across many refills.
    #[test]
    fn push_pop_round_trips_in_order_and_wraps() {
        let mut w = Window::<{ 192 + 8 * 16 }>::new();
        let (mut p, mut c) = split::<Rec>(&mut w.0, 8).expect("split");

        assert_eq!(c.pop(), None, "fresh ring is empty");
        for round in 0..10u64 {
            for i in 0..8u64 {
                p.push(rec(round * 8 + i, i as u32)).expect("fits");
            }
            assert_eq!(p.push(rec(999, 0)), Err(Full), "capacity is a hard edge");
            for i in 0..8u64 {
                assert_eq!(c.pop(), Some(rec(round * 8 + i, i as u32)), "FIFO order");
            }
            assert_eq!(c.pop(), None, "drained");
        }
        assert_eq!(p.dropped(), 0, "plain push never counts drops");
    }

    /// `push_lossy` drops on a full ring and both sides read the count.
    #[test]
    fn lossy_push_counts_drops() {
        let mut w = Window::<{ 192 + 2 * 16 }>::new();
        let (mut p, mut c) = split::<Rec>(&mut w.0, 2).expect("split");

        assert!(p.push_lossy(rec(1, 1)));
        assert!(p.push_lossy(rec(2, 2)));
        assert!(!p.push_lossy(rec(3, 3)), "full → dropped");
        assert!(!p.push_lossy(rec(4, 4)));
        assert_eq!((p.dropped(), c.dropped()), (2, 2));
        // The ring's contents are the accepted records, unperturbed.
        assert_eq!(c.pop(), Some(rec(1, 1)));
        assert_eq!(c.pop(), Some(rec(2, 2)));
        assert_eq!(c.pop(), None);
    }

    /// Attach validates fail-closed: wrong record type, foreign bytes,
    /// short windows, bad capacities, misalignment.
    #[test]
    fn attach_validates_fail_closed() {
        #[derive(Clone, Copy, zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
        #[repr(C)]
        struct Other(u16);

        let mut w = Window::<{ 192 + 8 * 16 }>::new();
        // Unformatted bytes refuse to attach.
        assert_eq!(consumer::<Rec>(&mut w.0).err(), Some(RingError::BadHeader));

        init::<Rec>(&mut w.0, 8).expect("init");
        assert!(producer::<Rec>(&mut w.0).is_ok());
        // A ring of Rec is not a ring of Other.
        assert_eq!(
            producer::<Other>(&mut w.0).err(),
            Some(RingError::WrongRecord)
        );

        // Corrupt the magic: refused again (fail closed on attach).
        w.0[0] ^= 0xFF;
        assert_eq!(producer::<Rec>(&mut w.0).err(), Some(RingError::BadHeader));

        // Capacity/size refusals at init.
        let mut small = Window::<64>::new();
        assert_eq!(
            init::<Rec>(&mut small.0, 8).err(),
            Some(RingError::TooSmall)
        );
        let mut w2 = Window::<{ 192 + 8 * 16 }>::new();
        assert_eq!(
            init::<Rec>(&mut w2.0, 3).err(),
            Some(RingError::BadCapacity)
        );
        assert_eq!(
            init::<Rec>(&mut w2.0, 0).err(),
            Some(RingError::BadCapacity)
        );
        // A misaligned base refuses (slice offset by one byte).
        assert_eq!(
            init::<Rec>(&mut w2.0[1..], 4).err(),
            Some(RingError::Misaligned)
        );
    }

    #[test]
    fn batches_wrap_without_partial_publication_or_partial_loss() {
        let mut w = Window::<{ 192 + 8 * 16 }>::new();
        let (mut tx, mut rx) = split::<Rec>(&mut w.0, 8).unwrap();
        let batch = [rec(1, 0), rec(2, 0), rec(3, 0)];
        for _ in 0..10 {
            assert!(tx.push_slice_lossy(&batch));
            assert!(tx.push_slice_lossy(&batch));
            let dropped = tx.dropped();
            assert!(!tx.push_slice_lossy(&batch));
            assert_eq!(tx.dropped(), dropped + 3);
            for expected in batch.into_iter().cycle().take(6) {
                assert_eq!(rx.pop(), Some(expected));
            }
            assert_eq!(rx.pop(), None);
        }
        assert!(tx.push_slice(&[]).is_ok());
        assert!(tx.push_slice(&[rec(0, 0); 9]).is_err());
        assert_eq!(rx.pop(), None);
    }

    /// The cross-thread SPSC storm: a real producer thread races a real
    /// consumer thread over one shared window; every record arrives
    /// exactly once, in order — the Release/Acquire hand-off proof.
    #[test]
    fn spsc_storm_delivers_every_record_in_order() {
        const ROUNDS: u64 = 100_000;
        let mut w = std::boxed::Box::new(Window::<{ 192 + 64 * 16 }>::new());
        let (mut p, mut c) = split::<Rec>(&mut w.0, 64).expect("split");

        std::thread::scope(|s| {
            s.spawn(move || {
                for seq in 0..ROUNDS {
                    loop {
                        if p.push(rec(seq, (seq as u32) ^ 0xA5A5_A5A5)).is_ok() {
                            break;
                        }
                        std::hint::black_box(()); // spin — full ring
                    }
                }
            });
            s.spawn(move || {
                let mut expect = 0u64;
                while expect < ROUNDS {
                    if let Some(r) = c.pop() {
                        assert_eq!(r.seq, expect, "in order, exactly once");
                        assert_eq!(r.val, (expect as u32) ^ 0xA5A5_A5A5, "payload intact");
                        expect += 1;
                    }
                }
                assert_eq!(c.pop(), None, "nothing after the last record");
            });
        });
    }

    /// `next` resolves immediately over a non-empty ring and stays
    /// Pending over an empty one (the level-trigger contract — the
    /// executor's re-poll is the wake).
    #[test]
    fn next_is_pop_or_pending() {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        const NOOP: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &NOOP),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: no-op vtable over a null pointer — nothing dereferenced.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP)) };
        let mut cx = Context::from_waker(&waker);

        let mut w = Window::<{ 192 + 4 * 16 }>::new();
        let (mut p, mut c) = split::<Rec>(&mut w.0, 4).expect("split");

        {
            let mut fut = core::pin::pin!(c.next());
            assert_eq!(fut.as_mut().poll(&mut cx), Poll::Pending, "empty → Pending");
        }
        p.push(rec(7, 7)).expect("push");
        {
            let mut fut = core::pin::pin!(c.next());
            assert_eq!(fut.as_mut().poll(&mut cx), Poll::Ready(rec(7, 7)));
        }
    }

    /// `peek_at` is the pipelined consumer's window: it sees every
    /// published record past the head, refuses past the last one, and —
    /// the property the whole hardware-submit discipline rests on — does
    /// not move the head, so nothing it returns has released its slot.
    #[test]
    fn peek_at_looks_past_the_head_without_releasing_it() {
        let mut w = Window::<{ 192 + 4 * 16 }>::new();
        let (mut p, mut c) = split::<Rec>(&mut w.0, 4).expect("split");

        assert_eq!(c.peek_at(0), None, "an empty ring shows nothing");
        for i in 0..3 {
            p.push(rec(i, i as u32)).expect("push");
        }
        assert_eq!(c.peek_at(0), c.peek(), "peek_at(0) is peek");
        assert_eq!(c.peek_at(1), Some(rec(1, 1)));
        assert_eq!(c.peek_at(2), Some(rec(2, 2)));
        assert_eq!(c.peek_at(3), None, "nothing published there yet");
        assert_eq!(
            c.outstanding(),
            3,
            "peeking published nothing and freed nothing"
        );
        assert_eq!(
            p.capacity_left(),
            1,
            "the producer still sees three slots held — peek_at released none"
        );

        assert_eq!(c.pop(), Some(rec(0, 0)));
        assert_eq!(
            c.peek_at(0),
            Some(rec(1, 1)),
            "the window slid with the head"
        );
        assert_eq!(c.peek_at(2), None);
    }
}
