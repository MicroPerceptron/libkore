//! `event` — the event channel: seqlock'd **state**, an spsc ring of
//! **records**, and one **wake word** that answers "anything new?" for
//! both (`res/EVENT.md`).
//!
//! One channel per (producer → consumer) direction, in a shared window
//! the two sides already hold.  This module owns the layout and the
//! lock-free halves; the *wait* (the monitor/park ladder) and the
//! *kick* (the conditional notice) live in the runtime (`kore_rt::event`),
//! because waiting needs an executor and kicking needs a held send end —
//! neither of which this contract crate may name.
//!
//! ## Layout (frozen within `EVENT_ABI`)
//!
//! ```text
//! offset   0: Header { magic u64 ‖ abi u32 ‖ state_bytes u32 ‖ ring_at u32 ‖ … }
//! offset  64: seq    (AtomicU32 — the wake word, the producer's line)
//! offset 128: parked (AtomicU32 — the park intent, the consumer's line)
//! offset 192: state  (S, seqlock'd by `seq`; padded to 64)
//! ring_at   : an ordinary [`spsc`](crate::spsc) ring of R
//! ```
//!
//! **`seq` is both the seqlock and the wake word.**  It is odd while the
//! producer is mid-publish and advances at the end of every batch —
//! state republished, records pushed, or both — so a consumer that
//! remembers `seq == s` learns "something happened" from one word, and
//! a hardware monitor has one line to watch.
//!
//! **`parked` makes the wake conditional** (the Dekker pair, proved at
//! [`EventSink::park`]/[`EventSource::commit`]): the consumer publishes
//! its intent to wait *before* its final check; the producer advances
//! `seq` *before* collecting the intent.  The two orders cross, so
//! either the consumer sees the new `seq` and never waits, or the
//! producer sees `parked == 1` and owes the notice.  A running consumer
//! therefore costs the producer **zero** syscalls.
//!
//! State is **level** — the producer republishes the whole `S`, the
//! consumer reads a snapshot, intermediate values may be lost by
//! design.  Records are **edges** — everything that must not coalesce
//! rides the ring, with its ordinary overflow accounting.  Both sides
//! re-check in a loop; a wake is a hint, the memory is the truth.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, Ordering, fence};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::spsc::{self, RingError, RingRecord};

/// Channel magic — "koreEVNT", LE.
pub const EVENT_MAGIC: u64 = 0x544E_5645_6572_6F6B;

/// The event layout ABI this module writes and accepts.
pub const EVENT_ABI: u32 = 1;

/// Cache-line separation unit; also the required window alignment.
const ALIGN: usize = 64;

/// Byte offset of the wake word (the producer's cache line).
const SEQ_AT: usize = 64;
/// Byte offset of the park intent (the consumer's cache line).
const PARKED_AT: usize = 128;
/// Byte offset of the state block.
const STATE_AT: usize = 192;

/// The channel's level state: plain old data, republished whole.
///
/// `FromBytes` is load-bearing: a seqlock reader may observe a torn
/// value before its re-check discards it, so every bit pattern must be
/// a value of `S`.  Alignment is capped at the line so the state block
/// needs no per-type offset.
pub trait EventState: Copy + FromBytes + IntoBytes + Immutable {}
impl<T: Copy + FromBytes + IntoBytes + Immutable> EventState for T {}

#[repr(C)]
#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable)]
struct Header {
    magic: u64,
    abi: u32,
    state_bytes: u32,
    ring_at: u32,
    reserved: [u32; 3],
}

/// Bytes a channel of state `S` and `capacity` records of `R` occupies.
pub const fn region_bytes<S, R>(capacity: u32) -> usize {
    STATE_AT + pad(size_of::<S>()) + spsc::region_bytes::<R>(capacity)
}

const fn pad(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// Format `window` as an empty channel — exclusive access (not yet
/// shared / not yet attached).  The state starts as `S`'s zero bytes,
/// `seq` at 0, nobody parked, the ring empty.
pub fn init<S: EventState, R: RingRecord>(
    window: &mut [u8],
    capacity: u32,
) -> Result<(), RingError> {
    if window.as_ptr().align_offset(ALIGN) != 0 {
        return Err(RingError::Misaligned);
    }
    if align_of::<S>() > ALIGN || size_of::<S>() > u32::MAX as usize {
        return Err(RingError::BadRecord);
    }
    let ring_at = STATE_AT + pad(size_of::<S>());
    if ring_at > u32::MAX as usize || window.len() < ring_at {
        return Err(RingError::TooSmall);
    }
    // Control area and state start clean; the header lands last so a
    // partially-formatted window never carries the magic (the spsc rule).
    window[..ring_at].fill(0);
    spsc::init::<R>(&mut window[ring_at..], capacity)?;
    let header = Header {
        magic: EVENT_MAGIC,
        abi: EVENT_ABI,
        state_bytes: size_of::<S>() as u32,
        ring_at: ring_at as u32,
        reserved: [0; 3],
    };
    // SAFETY: bounds + alignment validated above; Header is plain data.
    unsafe { core::ptr::write(window.as_mut_ptr() as *mut Header, header) };
    Ok(())
}

/// The two words and the state pointer, validated against the header.
struct Raw<S> {
    seq: *const AtomicU32,
    parked: *const AtomicU32,
    state: *mut S,
    _kind: PhantomData<S>,
}

impl<S: EventState> Raw<S> {
    /// # Safety
    /// `base..base+len` stays valid and mapped for the half's lifetime;
    /// the caller side touches the window only through one half.
    unsafe fn bind(base: *mut u8, len: usize) -> Result<(Self, usize), RingError> {
        if base.align_offset(ALIGN) != 0 {
            return Err(RingError::Misaligned);
        }
        if len < STATE_AT + pad(size_of::<S>()) {
            return Err(RingError::TooSmall);
        }
        // SAFETY: header bounds checked; read_volatile because the other
        // side's mapping aliases this memory.
        let header = unsafe { core::ptr::read_volatile(base as *const Header) };
        let ring_at = header.ring_at as usize;
        if header.magic != EVENT_MAGIC
            || header.abi != EVENT_ABI
            || header.state_bytes as usize != size_of::<S>()
            || ring_at != STATE_AT + pad(size_of::<S>())
            || ring_at > len
        {
            return Err(RingError::BadHeader);
        }
        Ok((
            Self {
                // SAFETY: offsets are inside the validated window and
                // line-aligned by construction.
                seq: unsafe { base.add(SEQ_AT) } as *const AtomicU32,
                parked: unsafe { base.add(PARKED_AT) } as *const AtomicU32,
                state: unsafe { base.add(STATE_AT) } as *mut S,
                _kind: PhantomData,
            },
            ring_at,
        ))
    }

    #[inline]
    fn seq(&self) -> &AtomicU32 {
        // SAFETY: bind validated the offset; AtomicU32 is always valid.
        unsafe { &*self.seq }
    }

    #[inline]
    fn parked(&self) -> &AtomicU32 {
        // SAFETY: as `seq`.
        unsafe { &*self.parked }
    }
}

/// The producer half: sole writer of the state, the records, and `seq`.
pub struct EventSource<'a, S: EventState, R: RingRecord> {
    raw: Raw<S>,
    records: spsc::Producer<'a, R>,
}

/// The consumer half: reads state snapshots, drains records, and owns
/// the park intent.
pub struct EventSink<'a, S: EventState, R: RingRecord> {
    raw: Raw<S>,
    records: spsc::Consumer<'a, R>,
}

// SAFETY: every shared access in either half rides an atomic or a
// volatile read under the seqlock; the halves alias the window through
// this protocol only, exactly as the spsc halves do.
unsafe impl<S: EventState, R: RingRecord> Send for EventSource<'_, S, R> {}
unsafe impl<S: EventState, R: RingRecord> Send for EventSink<'_, S, R> {}

/// Attach the **producer** half of an already-[`init`]ialized channel —
/// the cross-space form (this side's own mapping of the shared lease).
/// Exclusive borrow: nothing else on this side touches the window while
/// the half lives.
pub fn source<S: EventState, R: RingRecord>(
    window: &mut [u8],
) -> Result<EventSource<'_, S, R>, RingError> {
    // SAFETY: the borrow pins the window; the protocol is the only access.
    let (raw, ring_at) = unsafe { Raw::bind(window.as_mut_ptr(), window.len()) }?;
    Ok(EventSource {
        raw,
        records: spsc::producer(&mut window[ring_at..])?,
    })
}

/// Attach the **consumer** half — [`source`]'s mirror.
pub fn sink<S: EventState, R: RingRecord>(
    window: &mut [u8],
) -> Result<EventSink<'_, S, R>, RingError> {
    // SAFETY: as `source`.
    let (raw, ring_at) = unsafe { Raw::bind(window.as_mut_ptr(), window.len()) }?;
    Ok(EventSink {
        raw,
        records: spsc::consumer(&mut window[ring_at..])?,
    })
}

/// [`init`] + both halves over one exclusively-borrowed window — the
/// same-address-space (and host-test) form, as [`spsc::split`].
pub fn split<S: EventState, R: RingRecord>(
    window: &mut [u8],
    capacity: u32,
) -> Result<(EventSource<'_, S, R>, EventSink<'_, S, R>), RingError> {
    init::<S, R>(window, capacity)?;
    let (base, len) = (window.as_mut_ptr(), window.len());
    // SAFETY: the `&mut` borrow pins the window; both halves access it
    // only through the channel protocol (disjoint roles, atomic words,
    // seqlock'd state, spsc hand-off).
    let (source_raw, ring_at) = unsafe { Raw::bind(base, len) }?;
    let (sink_raw, _) = unsafe { Raw::bind(base, len) }?;
    let ring = unsafe { core::slice::from_raw_parts_mut(base.add(ring_at), len - ring_at) };
    let ring_alias = unsafe { core::slice::from_raw_parts_mut(base.add(ring_at), len - ring_at) };
    Ok((
        EventSource {
            raw: source_raw,
            records: spsc::producer(ring)?,
        },
        EventSink {
            raw: sink_raw,
            records: spsc::consumer(ring_alias)?,
        },
    ))
}

impl<S: EventState, R: RingRecord> EventSource<'_, S, R> {
    /// Replace level state only when it changes. The sole producer can read
    /// its own state without a seqlock retry; consumers never write it.
    /// Returns whether the batch needs publishing through [`commit`](Self::commit).
    pub fn replace_state(&mut self, next: S) -> bool
    where
        S: PartialEq,
    {
        // SAFETY: bind validated the storage; this half is its sole writer.
        if unsafe { core::ptr::read(self.raw.state) } == next {
            return false;
        }
        self.state(|state| *state = next);
        true
    }

    /// Republish the state: read-modify-write under the seqlock.
    ///
    /// The write is torn from a concurrent reader's point of view; the
    /// odd `seq` makes the reader discard and retry.  Not a wake —
    /// finish the batch with [`commit`](Self::commit).
    pub fn state(&mut self, update: impl FnOnce(&mut S)) {
        let opened = self.raw.seq().load(Ordering::Relaxed).wrapping_add(1);
        debug_assert!(opened & 1 == 1, "producer batches never interleave");
        self.raw.seq().store(opened, Ordering::Relaxed);
        // The odd store is visible before any state byte changes …
        fence(Ordering::Release);
        // Read-modify-write through the raw pointer: the copy is this
        // side's (sole writer, SPSC discipline), the write-back volatile
        // so the other mapping observes every byte; concurrent readers
        // discard torn copies under the seqlock.
        // SAFETY: `bind` validated the pointer; any bit pattern is a
        // valid `S`.
        let mut state = unsafe { core::ptr::read(self.raw.state) };
        update(&mut state);
        // SAFETY: as above; sole writer.
        unsafe { core::ptr::write_volatile(self.raw.state, state) };
        // … and every state byte lands before the closing even store.
        self.raw
            .seq()
            .store(opened.wrapping_add(1), Ordering::Release);
    }

    /// Append one record — an edge; it must not coalesce.  Not a wake.
    /// `Err(Full)` leaves policy with the producer, as on the raw ring.
    pub fn push(&mut self, record: R) -> Result<(), spsc::Full> {
        self.records.push(record)
    }

    /// [`push`](Self::push), dropping and counting on a full ring.
    ///
    /// The channel wraps the same spsc ring, so it exposes the ring's own
    /// overflow policy rather than making every lossy producer reach past
    /// the wrapper for it.  A telemetry-shaped plane — input reports, where
    /// a stalled consumer must never block the producer — states its posture
    /// here, and [`dropped`](Self::dropped) is what makes the loss legible
    /// instead of silent.  Returns whether the record was enqueued.
    pub fn push_lossy(&mut self, record: R) -> bool {
        self.records.push_lossy(record)
    }

    /// [`push_lossy`](Self::push_lossy) for a batch, published atomically:
    /// either every record lands or none does and all are counted.  A
    /// protocol whose records are only meaningful together — a pointer
    /// frame's begin/motion/buttons/end — must not have its middle torn out
    /// by a ring that filled part way through.
    pub fn push_slice_lossy(&mut self, records: &[R]) -> bool {
        self.records.push_slice_lossy(records)
    }

    /// Records this side dropped on a full ring.
    pub fn dropped(&self) -> u64 {
        self.records.dropped()
    }

    /// The ring's record capacity.
    pub fn capacity(&self) -> u32 {
        self.records.capacity()
    }

    /// Records that would fit right now — capacity minus what is
    /// outstanding.  With [`capacity`](Self::capacity) this is how a
    /// producer reports its own backlog without reaching past the channel
    /// to the ring underneath.
    pub fn capacity_left(&self) -> u32 {
        self.records.capacity_left()
    }

    /// End the batch: advance the wake word and collect the park intent.
    ///
    /// Returns whether the consumer had parked — `true` means the caller
    /// owes it one notice (`kore_rt::event` sends it); `false` means the
    /// consumer is awake and the whole batch cost zero syscalls.
    ///
    /// Ordering (the producer's half of the Dekker pair): the `seq`
    /// advance happens-before the `parked` collection, both `SeqCst`.
    /// Combined with the consumer publishing `parked` *before* its final
    /// `seq` check, the two cannot both miss: a consumer that parked
    /// after this swap necessarily reads the advanced `seq` and returns
    /// instead of waiting.
    #[must_use = "true means the consumer parked: a notice is owed"]
    pub fn commit(&mut self) -> bool {
        let seq = self.raw.seq();
        // `state` may already have advanced `seq`; a records-only batch
        // advances it here.  Always move it so one word means "new".
        seq.fetch_add(2, Ordering::SeqCst);
        self.raw.parked().swap(0, Ordering::SeqCst) != 0
    }
}

impl<S: EventState, R: RingRecord> EventSink<'_, S, R> {
    /// A stable snapshot of the state — the seqlock read loop.
    ///
    /// Bounded retries: a producer republishing forever cannot livelock
    /// the reader past the cap (the window is shared with another party;
    /// fail-closed beats trust).  `None` only under that cap — treat it
    /// as "try again after the next wake".
    pub fn state(&self) -> Option<S> {
        const RETRIES: usize = 64;
        let seq = self.raw.seq();
        for _ in 0..RETRIES {
            let opened = seq.load(Ordering::Acquire);
            if opened & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            // SAFETY: any bit pattern is a valid `S` (`FromBytes`), and a
            // torn copy is discarded by the re-check below; volatile
            // because the producer's mapping aliases this memory.
            let copy = unsafe { core::ptr::read_volatile(self.raw.state) };
            // The copy must complete before the closing seq re-read.
            fence(Ordering::Acquire);
            if seq.load(Ordering::Acquire) == opened {
                return Some(copy);
            }
        }
        None
    }

    /// Drain one record, oldest first.
    pub fn pop(&mut self) -> Option<R> {
        self.records.pop()
    }

    /// Records the producer dropped on a full ring — the counter lives in
    /// the shared window, so a consumer can see it fell behind without the
    /// producer having to encode that into state.
    pub fn dropped(&self) -> u64 {
        self.records.dropped()
    }

    /// Records currently queued.
    pub fn outstanding(&self) -> u32 {
        self.records.outstanding()
    }

    /// The wake word's current value — the `seen` a later
    /// [`park`](Self::park) or `kore_rt::event::wait_past` compares.
    pub fn seq(&self) -> u32 {
        self.raw.seq().load(Ordering::SeqCst)
    }

    /// The wake word itself, for a monitored wait on its cache line.
    pub fn seq_word(&self) -> &AtomicU32 {
        self.raw.seq()
    }

    /// The park-intent word — for a wait guard that must withdraw the
    /// intent on drop without holding `self` (`kore_rt::event`'s ladder).
    pub fn parked_word(&self) -> &AtomicU32 {
        self.raw.parked()
    }

    /// Publish the intent to wait, then make the final check: `true`
    /// means "committed to waiting" (the producer will see `parked` and
    /// send its notice); `false` means `seq` already moved on — do not
    /// wait.
    ///
    /// Ordering (the consumer's half of the Dekker pair): the `parked`
    /// store happens-before the `seq` load, both `SeqCst` — see
    /// [`EventSource::commit`] for why the two halves cannot both miss.
    /// The intent may go stale (a later wake for an earlier park); a
    /// spurious notice is the harmless direction, exactly as in `futex`.
    pub fn park(&self, seen: u32) -> bool {
        self.raw.parked().store(1, Ordering::SeqCst);
        if self.raw.seq().load(Ordering::SeqCst) != seen {
            self.raw.parked().store(0, Ordering::SeqCst);
            return false;
        }
        true
    }

    /// Withdraw a published park intent (a wait that ends for another
    /// reason — deadline, shutdown — so a later batch is not charged a
    /// notice for a waiter that left).
    pub fn unpark(&self) {
        self.raw.parked().store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;
    use std::vec;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    struct TestState {
        serial: u64,
        presented: u64,
        flags: u32,
        _pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    struct TestRecord {
        kind: u32,
        value: u32,
    }

    fn window(capacity: u32) -> Box<[u8]> {
        // Over-align by allocating a u64 vec (64-aligned in practice is
        // not guaranteed by that; use a manual aligned alloc).
        let bytes = region_bytes::<TestState, TestRecord>(capacity) + ALIGN;
        vec![0u8; bytes].into_boxed_slice()
    }

    fn aligned(window: &mut [u8]) -> &mut [u8] {
        let off = window.as_ptr().align_offset(ALIGN);
        &mut window[off..]
    }

    /// A lossy producer never blocks and never tears a batch: the whole
    /// slice lands or none of it does, and what did not land is counted.
    ///
    /// The pointer plane depends on both halves — a frame is BEGIN, records,
    /// END, and a consumer may not apply a prefix, so a ring that filled part
    /// way through must drop the frame rather than publish its head.
    #[test]
    fn a_lossy_batch_is_all_or_nothing_and_counts_what_it_dropped() {
        const CAP: u32 = 4;
        let mut backing = window(CAP);
        let window = aligned(&mut backing);
        let (mut source, mut sink) = split::<TestState, TestRecord>(window, CAP).unwrap();

        let frame = [
            TestRecord { kind: 1, value: 1 },
            TestRecord { kind: 2, value: 2 },
            TestRecord { kind: 3, value: 3 },
        ];
        assert!(source.push_slice_lossy(&frame), "an empty ring takes a frame");
        assert_eq!(source.dropped(), 0);
        assert_eq!(source.capacity(), CAP);
        assert_eq!(source.capacity_left(), CAP - frame.len() as u32);

        // Only one slot left: the three-record frame cannot fit, so none of it
        // may land — a torn frame is worse than a missing one.
        assert!(!source.push_slice_lossy(&frame), "a partial frame is refused");
        assert_eq!(
            source.dropped(),
            frame.len() as u64,
            "loss is counted in records, not batches"
        );

        let mut drained = 0;
        while sink.pop().is_some() {
            drained += 1;
        }
        assert_eq!(drained, frame.len(), "only the whole frame was published");
        assert_eq!(sink.dropped(), frame.len() as u64, "the consumer sees the loss");
    }

    /// A single lossy push drops the newest record on a full ring and leaves
    /// the queued ones alone — the producer is never blocked by a stalled
    /// consumer.
    #[test]
    fn a_lossy_push_drops_the_new_record_not_the_queued_ones() {
        const CAP: u32 = 2;
        let mut backing = window(CAP);
        let window = aligned(&mut backing);
        let (mut source, mut sink) = split::<TestState, TestRecord>(window, CAP).unwrap();

        assert!(source.push_lossy(TestRecord { kind: 1, value: 1 }));
        assert!(source.push_lossy(TestRecord { kind: 2, value: 2 }));
        assert!(!source.push_lossy(TestRecord { kind: 3, value: 3 }), "full");
        assert_eq!(source.dropped(), 1);
        assert_eq!(source.capacity_left(), 0);

        assert_eq!(sink.pop(), Some(TestRecord { kind: 1, value: 1 }));
        assert_eq!(sink.pop(), Some(TestRecord { kind: 2, value: 2 }));
        assert_eq!(sink.pop(), None, "the dropped record was never enqueued");
    }

    #[test]
    fn state_and_records_round_trip_and_seq_counts_batches() {
        let mut backing = window(8);
        let window = aligned(&mut backing);
        let (mut source, mut sink) = split::<TestState, TestRecord>(window, 8).unwrap();
        assert_eq!(sink.seq(), 0);
        assert_eq!(
            sink.state().unwrap(),
            TestState {
                serial: 0,
                presented: 0,
                flags: 0,
                _pad: 0
            },
            "a fresh channel carries the zero state"
        );

        source.state(|s| {
            s.serial = 7;
            s.presented = 3;
        });
        source.push(TestRecord { kind: 1, value: 42 }).unwrap();
        assert!(!source.commit(), "nobody parked: no notice owed");

        let seen = sink.seq();
        assert_ne!(seen, 0, "the batch moved the wake word");
        let state = sink.state().unwrap();
        assert_eq!((state.serial, state.presented), (7, 3));
        assert_eq!(sink.pop(), Some(TestRecord { kind: 1, value: 42 }));
        assert_eq!(sink.pop(), None);

        // A records-only batch still moves the word.
        source.push(TestRecord { kind: 2, value: 1 }).unwrap();
        let _ = source.commit();
        assert_ne!(sink.seq(), seen);
    }

    #[test]
    fn the_park_intent_is_collected_exactly_when_it_was_committed() {
        let mut backing = window(4);
        let window = aligned(&mut backing);
        let (mut source, sink) = split::<TestState, TestRecord>(window, 4).unwrap();

        // Park before any batch: the next commit owes the notice.
        let seen = sink.seq();
        assert!(sink.park(seen), "nothing new: the wait stands");
        source.state(|s| s.serial = 1);
        assert!(source.commit(), "the parked consumer is owed a notice");
        assert!(!source.commit(), "the intent was collected once");

        // A park raced by a batch refuses instead of waiting.
        let stale = sink.seq().wrapping_sub(2);
        assert!(!sink.park(stale), "seq moved: do not wait");
        source.state(|s| s.serial = 2);
        assert!(!source.commit(), "a refused park leaves no intent behind");

        // A withdrawn park charges nothing.
        assert!(sink.park(sink.seq()));
        sink.unpark();
        source.state(|s| s.serial = 3);
        assert!(!source.commit(), "an unparked waiter is not owed a notice");
    }

    #[test]
    fn a_torn_state_read_is_discarded_never_returned() {
        // Reproduce the seqlock hazard deterministically: hold the lock
        // open (odd seq) and check the reader refuses under its cap.
        let mut backing = window(4);
        let window = aligned(&mut backing);
        let (source, sink) = split::<TestState, TestRecord>(window, 4).unwrap();
        source.raw.seq().store(1, Ordering::SeqCst);
        assert_eq!(sink.state(), None, "an open seqlock never yields a value");
        source.raw.seq().store(2, Ordering::SeqCst);
        assert!(sink.state().is_some());
    }

    #[test]
    fn attach_validates_the_header_fail_closed() {
        let mut backing = window(4);
        let window = aligned(&mut backing);
        init::<TestState, TestRecord>(window, 4).unwrap();

        // Wrong state size refuses.
        assert!(matches!(
            source::<u64, TestRecord>(window),
            Err(RingError::BadHeader)
        ));
        // A stray magic byte refuses.
        window[0] ^= 1;
        assert!(matches!(
            sink::<TestState, TestRecord>(window),
            Err(RingError::BadHeader)
        ));
        window[0] ^= 1;
        assert!(sink::<TestState, TestRecord>(window).is_ok());

        // A window too small for the declared layout refuses.
        assert!(matches!(
            init::<TestState, TestRecord>(&mut [0u8; 0], 4),
            Err(RingError::TooSmall) | Err(RingError::Misaligned)
        ));
    }

    /// The cross-thread shape: a producer thread batches state+records,
    /// a consumer thread drains through park/re-check.  No executor here
    /// — the "notice" is modeled by the returned owed flag — so this
    /// pins the memory-ordering contract, not the ladder.
    #[test]
    fn cross_thread_batches_are_never_lost() {
        use std::sync::atomic::AtomicU64;
        const BATCHES: u64 = 10_000;
        let mut backing = window(64);
        let window = aligned(&mut backing);
        let (mut source, mut sink) = split::<TestState, TestRecord>(window, 64).unwrap();
        let notices = AtomicU64::new(0);

        std::thread::scope(|scope| {
            let notices = &notices;
            scope.spawn(move || {
                for serial in 1..=BATCHES {
                    source.state(|s| s.serial = serial);
                    while source
                        .push(TestRecord {
                            kind: 0,
                            value: serial as u32,
                        })
                        .is_err()
                    {
                        std::thread::yield_now();
                    }
                    if source.commit() {
                        notices.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            scope.spawn(move || {
                let mut drained = 0u64;
                let mut last_serial = 0u64;
                while drained < BATCHES {
                    while let Some(record) = sink.pop() {
                        assert_eq!(u64::from(record.value), drained + 1, "in order");
                        drained += 1;
                    }
                    let state = match sink.state() {
                        Some(state) => state,
                        None => continue,
                    };
                    assert!(state.serial >= last_serial, "state is monotone");
                    assert!(state.serial >= drained, "state never trails the ring");
                    last_serial = state.serial;
                    if drained < BATCHES {
                        let seen = sink.seq();
                        if sink.outstanding() == 0 && sink.park(seen) {
                            // A real consumer parks here; the model spins
                            // (the ordering, not the ladder, is under test).
                            while sink.seq() == seen {
                                std::hint::spin_loop();
                            }
                            sink.unpark();
                        }
                    }
                }
                assert_eq!(last_serial, BATCHES);
            });
        });
    }
}
