//! A C program's `stdout` and `stderr`.
//!
//! The launcher's stream (the launch's `PROFILE_INTERACTIVE` window) is
//! mapped on thread 0, the host, and is bound to it: only thread 0 may publish
//! into it. Every thread writes the C streams, so their bytes reach the
//! stream through the host, by one in-process ring:
//!
//! - **A writer** holds the streams' lock for its whole C call: kore-c-shim
//!   takes it around each stream call, as POSIX's `flockfile` does, so one
//!   call's output is never split by another thread's. Under the lock the
//!   writer is the ring's one producer. It waits for room for each piece,
//!   copies it in, and kicks the host only if the host armed itself before
//!   parking, so a loop of `printf` costs no trap per call. The host can
//!   kick the main thread but holds no send into a pthread's doorbell, so
//!   every wait for room is bounded by the runtime's slice: a writer the
//!   host cannot reach re-reads the ring each slice, as the thread
//!   controller's unreached clients do.
//! - **The host** drains the ring a pass at a time — what was queued when
//!   the pass began — into the stream's output records, as many as the
//!   launcher has room for, and waits for their capacity when they are full:
//!   a slow terminal holds the program back rather than losing its output.
//!   Between passes it serves the program's end and its filesystem sessions,
//!   so a program that never stops printing still ends when it asks to.
//! - **The host's own writes** (a filesystem step's callbacks run on it)
//!   cannot wait for the host. They are kept in order beside the ring, each
//!   stamped with the ring position it follows, and drained in that order.
//!   The host never takes the streams' lock, which a writer waiting on the
//!   host may hold.
//! - **The end** drains what was queued and closes the stream before the
//!   process ends, so the launcher shows the program's output before its exit
//!   status. An end committed on the host itself (a callback's `exit` or
//!   `abort`, a panic there) cannot wait, and relays at once what the
//!   launcher has room for. A write after the end, or once the stream is
//!   gone, reports `EOF`.
//!
//! The ring and its record codec are plain data over atomics ([`Ring`]), so
//! the host suite exercises the code the target runs; the endpoint, the lock's
//! parks and the kicks are the metal half (`metal`).

use core::cell::UnsafeCell;
#[cfg(target_os = "none")]
use core::sync::atomic::AtomicBool;
use core::sync::atomic::{AtomicU32, Ordering};

/// Which C stream a record carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    Out,
    Err,
}

impl Stream {
    const fn tag(self) -> u8 {
        match self {
            Self::Out => 1,
            Self::Err => 2,
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Out),
            2 => Some(Self::Err),
            _ => None,
        }
    }
}

/// Bytes of one record's header: its stream's tag and its length.
const HEADER: usize = 2;
/// The most payload one record carries: its length is one byte.
pub(crate) const RECORD_PAYLOAD: usize = u8::MAX as usize;

/// The ring's bytes ([`RING`]). A power of two, so a position is an index by
/// a mask, and far below `u32`'s range, so the free-running positions'
/// difference never wraps past a lap.
const RING_BYTES: usize = 1 << 14;

/// The ring bytes `len` payload bytes take, in whole records.
pub(crate) const fn framed(len: usize) -> usize {
    len + HEADER * len.div_ceil(RECORD_PAYLOAD)
}

/// The largest piece the ring keeps whole: half of it, so a writer waiting
/// for that much room never waits for more room than the host can make
/// while another piece is queued ahead of it. A longer write goes in pieces
/// of this size, under the one lock its call holds.
pub(crate) const WHOLE_WRITE: usize =
    RING_BYTES / 2 - HEADER * (RING_BYTES / 2).div_ceil(RECORD_PAYLOAD);

const _: () = assert!(RING_BYTES.is_power_of_two());
const _: () = assert!(framed(WHOLE_WRITE) <= RING_BYTES / 2);

/// A cursor on a cache line of its own: the writer's `tail` and the host's
/// `head` move on different cores, and must not share a line.
#[repr(align(64))]
struct Cursor(AtomicU32);

/// One ring of framed records over free-running `u32` positions: `tail` is
/// written only by the lock holder, `head` only by the host.
pub(crate) struct Ring<const N: usize> {
    bytes: UnsafeCell<[u8; N]>,
    /// The first byte the host has not consumed.
    head: Cursor,
    /// One past the last byte a writer published.
    tail: Cursor,
}

// SAFETY: the bytes between `head` and `tail` belong to the host, the rest to
// the lock holder; each side publishes its cursor after its last access and
// reads the other's with Acquire before its first.
unsafe impl<const N: usize> Sync for Ring<N> {}

impl<const N: usize> Ring<N> {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: UnsafeCell::new([0; N]),
            head: Cursor(AtomicU32::new(0)),
            tail: Cursor(AtomicU32::new(0)),
        }
    }

    /// The host's cursor, for a writer to wait on.
    pub(crate) fn head(&self) -> &AtomicU32 {
        &self.head.0
    }

    /// Bytes published and not consumed, at most: `head` is read first, and
    /// `tail` never falls behind any `head` read before it.
    pub(crate) fn queued(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        self.tail.0.load(Ordering::Acquire).wrapping_sub(head) as usize
    }

    /// Free bytes: exact under the lock, where `tail` is this thread's;
    /// conservative outside it, where the host and another writer may both
    /// have moved between the two reads.
    pub(crate) fn room(&self) -> usize {
        N.saturating_sub(self.queued())
    }

    /// Room for `bytes` more, framed.
    pub(crate) fn fits(&self, bytes: usize) -> bool {
        self.room() >= framed(bytes)
    }

    /// The position the next record will be published at.
    pub(crate) fn tail(&self) -> u32 {
        self.tail.0.load(Ordering::Acquire)
    }

    /// Publish `payload` as records of `stream`, whole.
    ///
    /// # Safety
    /// The caller is the ring's one producer (it holds the streams' lock) and
    /// saw [`fits`](Self::fits) for `payload` under that lock.
    pub(crate) unsafe fn publish(&self, stream: Stream, payload: &[u8]) {
        let mut at = self.tail.0.load(Ordering::Relaxed);
        for record in payload.chunks(RECORD_PAYLOAD) {
            // SAFETY: the caller's contract: `[tail, tail + framed)` is free,
            // so the host reads none of these bytes until the store below.
            unsafe {
                self.put(at, stream.tag());
                self.put(at.wrapping_add(1), record.len() as u8);
                for (i, byte) in record.iter().enumerate() {
                    self.put(at.wrapping_add((HEADER + i) as u32), *byte);
                }
            }
            at = at.wrapping_add((HEADER + record.len()) as u32);
        }
        // SeqCst, not Release: the host's park recheck reads `tail` after
        // arming, and the writer reads the armed flag after this store — the
        // Dekker pair that makes a kick unnecessary unless the host armed.
        self.tail.0.store(at, Ordering::SeqCst);
    }

    /// The record at the host's cursor, if any: its stream and its payload,
    /// copied into `out`. The host is the one consumer.
    pub(crate) fn peek(&self, out: &mut [u8; RECORD_PAYLOAD]) -> Option<(Stream, usize)> {
        let head = self.head.0.load(Ordering::Relaxed);
        if self.tail.0.load(Ordering::Acquire) == head {
            return None;
        }
        // SAFETY: `[head, tail)` holds whole records the Acquire above made
        // visible, and only this consumer advances `head`.
        unsafe {
            let stream = Stream::from_tag(self.get(head))?;
            let len = self.get(head.wrapping_add(1)) as usize;
            for (i, byte) in out.iter_mut().take(len).enumerate() {
                *byte = self.get(head.wrapping_add((HEADER + i) as u32));
            }
            Some((stream, len))
        }
    }

    /// Consume the record [`peek`](Self::peek) returned, `len` payload bytes.
    pub(crate) fn consume(&self, len: usize) {
        let head = self.head.0.load(Ordering::Relaxed);
        self.head
            .0
            .store(head.wrapping_add((HEADER + len) as u32), Ordering::Release);
    }

    /// Drop every record published so far, as the one consumer.
    pub(crate) fn discard(&self) {
        let tail = self.tail.0.load(Ordering::Acquire);
        self.head.0.store(tail, Ordering::Release);
    }

    unsafe fn put(&self, at: u32, byte: u8) {
        // SAFETY: the caller owns byte `at` (see `publish`).
        unsafe { (*self.bytes.get())[at as usize & (N - 1)] = byte }
    }

    unsafe fn get(&self, at: u32) -> u8 {
        // SAFETY: the caller owns byte `at` (see `peek`).
        unsafe { (*self.bytes.get())[at as usize & (N - 1)] }
    }
}

/// The program's one ring.
#[cfg(target_os = "none")]
pub(crate) static RING: Ring<RING_BYTES> = Ring::new();

/// The host is parked, or about to, and a writer must kick it (see
/// [`Ring::publish`]).
#[cfg(target_os = "none")]
static ARMED: AtomicBool = AtomicBool::new(false);

/// The room the main thread waits for, framed, or 0: the one writer the host
/// can kick, kicked once that much is free rather than at every record.
#[cfg(target_os = "none")]
static MAIN_NEEDS: AtomicU32 = AtomicU32::new(0);

/// The launcher's stream is gone, or there is no host to drain the ring:
/// writes fail rather than wait for a drain that will not come.
#[cfg(target_os = "none")]
static LOST: AtomicBool = AtomicBool::new(false);

/// The program has ended: nothing written from here is relayed.
#[cfg(target_os = "none")]
static FINISHED: AtomicBool = AtomicBool::new(false);

/// Writes report `EOF` from here on.
#[cfg(target_os = "none")]
fn closed() -> bool {
    LOST.load(Ordering::Acquire) || FINISHED.load(Ordering::Acquire)
}

/// Whether the host has anything to drain.
#[cfg(target_os = "none")]
pub(crate) fn pending() -> bool {
    !closed() && (RING.queued() != 0 || metal::host_pending())
}

/// A write the host made while it could not drain: its bytes, and the ring
/// position it follows.
#[cfg(target_os = "none")]
pub(crate) struct Held {
    pub(crate) after: u32,
    pub(crate) stream: Stream,
    pub(crate) bytes: alloc::vec::Vec<u8>,
}

/// Whether a held write is the next item to drain: it belongs at ring
/// position `held_after`, so it goes once the host's cursor reaches that
/// point, after every record published before it and before every one
/// published after. A position the cursor has already passed (a write made
/// during a call whose first records were already relayed) goes at once.
/// Positions are free-running, and a stamp never strays half the `u32`
/// range from the cursor.
pub(crate) fn held_is_next(held_after: u32, head: u32) -> bool {
    head.wrapping_sub(held_after) as i32 >= 0
}

extern crate alloc;

#[cfg(target_os = "none")]
pub(crate) use metal::{
    Endpoint, abandon, arm, drain, finish, flush_now, install, report, write_standard,
};

#[cfg(target_os = "none")]
mod metal {
    use super::*;
    use alloc::collections::VecDeque;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};
    use core::mem::MaybeUninit;
    use core::sync::atomic::AtomicU64;
    use core::time::Duration;
    use kore_c_shim::stdio::{Standard, StandardSink, install_standard_sink};
    use kore_std::abi::process::{PROCESS_OUTPUT_CHUNK_BYTES, ProcessOutputEvent};
    use kore_std::process::{
        MappedProcessStdioChild, ProcessStdioParkStep, ProcessStdioWaitFor, ProcessStdioWake,
    };
    use kore_std::runtime::futex::{self, WaitOutcome};
    use kore_std::runtime::{executor, pace, thread};

    /// How long a writer waits before it re-reads what it waits for: the
    /// runtime's slice for a waiter its waker cannot reach (the thread
    /// controller's `UNREACHED_SLICE`). The host kicks the main thread
    /// sooner; a pthread rides the slice.
    const SLICE: Duration = Duration::from_nanos(pace::CEDE_NS);

    /// The host's half: the launcher's stream and the writes the host held.
    /// Thread 0 alone touches it (the stream is bound to that thread), and
    /// never across an await.
    pub(crate) struct Endpoint {
        stdio: RefCell<MappedProcessStdioChild>,
        held: RefCell<VecDeque<Held>>,
        /// Bytes of the next item (the front held write, or the ring's head
        /// record) already published. A drain can be dropped between two
        /// records it publishes (it races a filesystem session), so it
        /// consumes an item only once all of it is out, and resumes past
        /// what it already sent rather than send it twice.
        sent: Cell<usize>,
    }

    /// Thread 0's endpoint, in static storage: the stream is mapped before
    /// the main thread is founded, and a launch without a heap still has
    /// one.
    struct EndpointCell(UnsafeCell<MaybeUninit<Endpoint>>);
    // SAFETY: written once by the host before any other thread exists, and
    // read only on thread 0 (`crate::program::on_host`) once `MAPPED` is set.
    unsafe impl Sync for EndpointCell {}
    static ENDPOINT: EndpointCell = EndpointCell(UnsafeCell::new(MaybeUninit::uninit()));
    static MAPPED: AtomicBool = AtomicBool::new(false);

    fn host() -> Option<&'static Endpoint> {
        if !crate::program::on_host() || !MAPPED.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: on thread 0, after `install` wrote it (see `EndpointCell`).
        Some(unsafe { (*ENDPOINT.0.get()).assume_init_ref() })
    }

    /// A line on `stderr` from the host itself: what it must say when the
    /// program cannot run. Kept in order and relayed as any write made on
    /// the host is. Nothing is said when the launch carried no stream.
    pub(crate) fn report(line: &[u8]) {
        let _ = sink(Standard::Error, line);
    }

    /// `write` on a standard stream's descriptor: `bytes` to `stdout`, or to
    /// `stderr` when `error`, in order with what `<stdio.h>` writes. `false`
    /// when the launch carried no stream or it has closed.
    pub(crate) fn write_standard(error: bool, bytes: &[u8]) -> bool {
        sink(
            if error {
                Standard::Error
            } else {
                Standard::Output
            },
            bytes,
        )
    }

    pub(crate) fn host_pending() -> bool {
        host().is_some_and(|endpoint| !endpoint.held.borrow().is_empty())
    }

    /// The C streams' sink and their lock.
    static SINK: StandardSink = StandardSink {
        write: sink,
        lock: lock_streams,
        unlock: unlock_streams,
    };

    /// Map the launcher's stream and send the C streams to it. `None` when
    /// the launch carried no stream: `stdout` then reports `EOF`, as it does
    /// for every program that has none.
    pub(crate) fn install(rt: &kore_std::runtime::entry::Rt) -> Option<&'static Endpoint> {
        let stdio = kore_std::process::map_stdio(rt)?;
        // SAFETY: thread 0, before any other thread exists, and only once
        // (`__kore_program_start` founds one host).
        let endpoint = unsafe {
            (*ENDPOINT.0.get()).write(Endpoint {
                stdio: RefCell::new(stdio),
                held: RefCell::new(VecDeque::new()),
                sent: Cell::new(0),
            })
        };
        MAPPED.store(true, Ordering::Release);
        install_standard_sink(&SINK);
        Some(endpoint)
    }

    /// The host could not be founded after the stream was mapped: no one
    /// will drain it, so every write reports `EOF`.
    pub(crate) fn abandon() {
        LOST.store(true, Ordering::SeqCst);
    }

    // ── The streams' lock ───────────────────────────────────────────────

    /// The streams' lock, reentrant per thread ([`DEPTH`]): a stream call
    /// holds it across all its writes, and the sink each write reaches takes
    /// it again.
    static LOCK: crate::lock::Lock = crate::lock::Lock::new();

    /// How many times this thread holds [`LOCK`].
    #[thread_local]
    static DEPTH: Cell<u32> = Cell::new(0);

    /// The call holding [`LOCK`], if any: [`IN_CALL`] beside the ring's
    /// `tail` when it began. The host's own writes made meanwhile are
    /// stamped there, so they go before the call rather than between its
    /// pieces.
    static CALL: AtomicU64 = AtomicU64::new(0);
    const IN_CALL: u64 = 1 << u32::BITS;

    /// Take the streams' lock. The host never does: a writer that holds it
    /// may be waiting for the host to drain.
    fn lock_streams() {
        if crate::program::on_host() {
            return;
        }
        let depth = DEPTH.get();
        if depth == 0 {
            LOCK.acquire();
            // The holder is the ring's one producer: `tail` is exact here.
            CALL.store(IN_CALL | u64::from(RING.tail()), Ordering::SeqCst);
        }
        DEPTH.set(depth + 1);
    }

    fn unlock_streams() {
        if crate::program::on_host() {
            return;
        }
        match DEPTH.get() {
            // Released more often than taken: the caller's mistake, and not
            // a reason to release another thread's hold.
            0 => {}
            1 => {
                DEPTH.set(0);
                CALL.store(0, Ordering::SeqCst);
                LOCK.release();
            }
            depth => DEPTH.set(depth - 1),
        }
    }

    // ── Writers ─────────────────────────────────────────────────────────

    fn sink(standard: Standard, bytes: &[u8]) -> bool {
        let stream = match standard {
            Standard::Output => Stream::Out,
            Standard::Error => Stream::Err,
        };
        if bytes.is_empty() {
            return true;
        }
        if closed() {
            return false;
        }
        if let Some(endpoint) = host() {
            return hold(endpoint, stream, bytes);
        }
        // The call's lock, again: this thread is the ring's one producer for
        // every piece.
        lock_streams();
        let written = bytes.chunks(WHOLE_WRITE).all(|piece| write(stream, piece));
        unlock_streams();
        written
    }

    /// One piece, under the streams' lock: wait for its room, then copy it
    /// in.
    fn write(stream: Stream, piece: &[u8]) -> bool {
        loop {
            if closed() {
                return false;
            }
            // Read before the room check: a wait on this value returns at
            // once if the host has moved since.
            let head = RING.head().load(Ordering::SeqCst);
            if RING.fits(piece.len()) {
                // SAFETY: the lock makes this thread the one producer, and
                // the room was checked under it.
                unsafe { RING.publish(stream, piece) };
                if ARMED.swap(false, Ordering::SeqCst) {
                    let _ = thread::kick_parent();
                }
                return true;
            }
            // The ring is full of records the host has not drained. A host
            // parked on its wake (armed) is kicked; one that is not is
            // draining already, or waiting for the launcher's room, and
            // moves its cursor when it has drained.
            let main = crate::program::on_main();
            if main {
                MAIN_NEEDS.store(framed(piece.len()) as u32, Ordering::SeqCst);
            }
            if ARMED.swap(false, Ordering::SeqCst) {
                let _ = thread::kick_parent();
            }
            let outcome = futex::wait(RING.head(), head, Some(SLICE));
            if main {
                MAIN_NEEDS.store(0, Ordering::SeqCst);
            }
            if outcome == WaitOutcome::Unsupported {
                let _ = pace::cede();
            }
        }
    }

    /// The host's own write: kept, stamped, for its next drain. Stamped at
    /// the start of a call in progress, so it does not split that call.
    fn hold(endpoint: &Endpoint, stream: Stream, bytes: &[u8]) -> bool {
        let mut copy = Vec::new();
        if copy.try_reserve_exact(bytes.len()).is_err() {
            return false;
        }
        copy.extend_from_slice(bytes);
        let mut held = endpoint.held.borrow_mut();
        if held.try_reserve(1).is_err() {
            return false;
        }
        let call = CALL.load(Ordering::SeqCst);
        let after = if call & IN_CALL != 0 {
            call as u32
        } else {
            RING.tail()
        };
        held.push_back(Held {
            after,
            stream,
            bytes: copy,
        });
        true
    }

    // ── The host ────────────────────────────────────────────────────────

    /// How one item went out.
    enum Pumped {
        /// All of it.
        Done,
        /// The launcher has no room for the next record: wait, or stop.
        Full,
        /// The launcher's stream is gone.
        Lost,
    }

    /// Publish what is left of one item, `len` bytes that `fill` copies out
    /// from an offset, as stream records while the launcher has room, past
    /// what [`Endpoint::sent`] says is out. Never waits.
    fn pump(
        endpoint: &Endpoint,
        stream: Stream,
        len: usize,
        fill: &dyn Fn(usize, &mut [u8]) -> usize,
    ) -> Pumped {
        let mut stdio = endpoint.stdio.borrow_mut();
        while endpoint.sent.get() < len {
            if stdio.output_capacity_left() == 0 {
                return Pumped::Full;
            }
            let from = endpoint.sent.get();
            let mut chunk = [0u8; PROCESS_OUTPUT_CHUNK_BYTES];
            let want = (len - from).min(PROCESS_OUTPUT_CHUNK_BYTES);
            let filled = fill(from, &mut chunk[..want]);
            if filled == 0 {
                // The item vanished under the drain: nothing can resume it.
                return Pumped::Lost;
            }
            let event = match stream {
                Stream::Out => ProcessOutputEvent::stdout(&chunk[..filled]),
                Stream::Err => ProcessOutputEvent::stderr(&chunk[..filled]),
            };
            let Ok(event) = event else {
                return Pumped::Lost;
            };
            match stdio.send_output_now(event) {
                Ok(ProcessStdioWake::Unreached(_)) | Err(_) => return Pumped::Lost,
                Ok(_) => endpoint.sent.set(from + filled),
            }
        }
        endpoint.sent.set(0);
        Pumped::Done
    }

    /// [`pump`] one item whole, waiting for the launcher's room between its
    /// records. `false` once the stream is gone. Dropped at a wait, it has
    /// published whole records only, and `sent` says how far.
    async fn send(
        endpoint: &Endpoint,
        stream: Stream,
        len: usize,
        fill: &dyn Fn(usize, &mut [u8]) -> usize,
    ) -> bool {
        loop {
            match pump(endpoint, stream, len, fill) {
                Pumped::Done => return true,
                Pumped::Lost => return false,
                Pumped::Full => {
                    if !room(endpoint).await {
                        return false;
                    }
                }
            }
        }
    }

    /// Copy the front held write's bytes from `from` into `out`.
    fn held_bytes(endpoint: &Endpoint, from: usize, out: &mut [u8]) -> usize {
        let held = endpoint.held.borrow();
        let Some(front) = held.front() else {
            return 0;
        };
        let Some(bytes) = front.bytes.get(from..from + out.len()) else {
            return 0;
        };
        out.copy_from_slice(bytes);
        out.len()
    }

    /// The next item due, if any: the front held write when the host's
    /// cursor reached its stamp, else the ring's head record if it lies
    /// before `end`.
    enum Next {
        Held(Stream, usize),
        Record(Stream, usize),
        None,
    }

    fn next(endpoint: &Endpoint, end: u32, record: &mut [u8; RECORD_PAYLOAD]) -> Next {
        let head = RING.head().load(Ordering::Acquire);
        if let Some(front) = endpoint.held.borrow().front()
            && held_is_next(front.after, head)
        {
            return Next::Held(front.stream, front.bytes.len());
        }
        if head == end {
            return Next::None;
        }
        match RING.peek(record) {
            Some((stream, len)) => Next::Record(stream, len),
            None => Next::None,
        }
    }

    /// A record went out: consume it, and wake the main thread once the
    /// room it waits for is free. Fenced, so the cursor's store is ordered
    /// before the read of what main needs, the other half of the writer's
    /// pair.
    fn consumed(len: usize, wake_main: &dyn Fn()) {
        RING.consume(len);
        core::sync::atomic::fence(Ordering::SeqCst);
        let needs = MAIN_NEEDS.load(Ordering::SeqCst) as usize;
        if needs != 0 && RING.room() >= needs {
            wake_main();
        }
    }

    /// The launcher's stream is gone: drop everything queued, and let every
    /// writer see it.
    fn lose(endpoint: &Endpoint, wake_main: &dyn Fn()) {
        LOST.store(true, Ordering::SeqCst);
        endpoint.held.borrow_mut().clear();
        endpoint.sent.set(0);
        RING.discard();
        core::sync::atomic::fence(Ordering::SeqCst);
        if MAIN_NEEDS.load(Ordering::SeqCst) != 0 {
            wake_main();
        }
    }

    /// How long a rotating notice receiver's handoff is given before the park
    /// is prepared again: the WASI host's figure.
    const HANDOFF_NS: u64 = 100_000;

    /// Wait until the launcher has room for one more record. `false` once its
    /// stream is gone. The endpoint is borrowed to prepare and settle each
    /// park, never across the wait (the WASI host's shape).
    async fn room(endpoint: &Endpoint) -> bool {
        loop {
            let step = {
                let mut stdio = endpoint.stdio.borrow_mut();
                // The drain snapshot comes before the capacity check, so room
                // the launcher makes between them is seen by the park's
                // recheck.
                let drain = stdio.begin_drain();
                if stdio.output_capacity_left() > 0 {
                    return true;
                }
                match stdio.prepare_park(&drain, ProcessStdioWaitFor::Capacity) {
                    Ok(step) => step,
                    Err(_) => return false,
                }
            };
            match step {
                ProcessStdioParkStep::Changed => {}
                ProcessStdioParkStep::Handoff => {
                    let _ = executor::next_wake_for(HANDOFF_NS).await;
                }
                ProcessStdioParkStep::Park(ticket) => {
                    let reason = executor::next_wake_on(ticket.post()).await;
                    if endpoint
                        .stdio
                        .borrow_mut()
                        .settle_park(ticket, reason)
                        .is_err()
                    {
                        return false;
                    }
                }
            }
        }
    }

    /// One pass: what the ring held when it began, and the host's writes
    /// held by then, in order. Bounded on both, so the host serves the
    /// program between passes: a filesystem step's callbacks hold writes
    /// while a pass waits for the launcher's room, and those wait for the
    /// next pass. Cancellation-safe at every wait: an item leaves its queue
    /// only once all of it is published.
    pub(crate) async fn drain(endpoint: &Endpoint, wake_main: &dyn Fn()) {
        let end = RING.tail();
        let mut held_left = endpoint.held.borrow().len();
        let mut record = [0u8; RECORD_PAYLOAD];
        loop {
            if LOST.load(Ordering::Acquire) {
                lose(endpoint, wake_main);
                return;
            }
            let sent = match next(endpoint, end, &mut record) {
                // A write held after the pass began: the next pass's, and so
                // is everything after it.
                Next::Held(..) if held_left == 0 => return,
                Next::Held(stream, len) => {
                    let fill = |from: usize, out: &mut [u8]| held_bytes(endpoint, from, out);
                    let sent = send(endpoint, stream, len, &fill).await;
                    if sent {
                        endpoint.held.borrow_mut().pop_front();
                        held_left -= 1;
                    }
                    sent
                }
                Next::Record(stream, len) => {
                    let bytes = record;
                    let fill = |from: usize, out: &mut [u8]| {
                        out.copy_from_slice(&bytes[from..from + out.len()]);
                        out.len()
                    };
                    let sent = send(endpoint, stream, len, &fill).await;
                    if sent {
                        consumed(len, wake_main);
                    }
                    sent
                }
                Next::None => return,
            };
            if !sent {
                lose(endpoint, wake_main);
                return;
            }
        }
    }

    /// The host arms itself before it rechecks what it would park for: a
    /// writer that published before this reads it and kicks, and one that
    /// publishes after is seen by the recheck (the Dekker pair with
    /// [`Ring::publish`]'s `SeqCst` store and the writer's swap).
    pub(crate) fn arm() {
        ARMED.store(true, Ordering::SeqCst);
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// The program's end on the host's loop: drain what was queued and close
    /// the stream. A write from here on reports `EOF`.
    pub(crate) async fn finish(endpoint: &Endpoint, wake_main: &dyn Fn()) {
        FINISHED.store(true, Ordering::SeqCst);
        drain(endpoint, wake_main).await;
        if LOST.load(Ordering::Acquire) || !room(endpoint).await {
            return;
        }
        let _ = endpoint
            .stdio
            .borrow_mut()
            .send_output_now(ProcessOutputEvent::closed());
    }

    /// The program's end on the host itself, which cannot wait (a callback's
    /// `exit` or `abort`, a panic there): relay at once what the launcher has
    /// room for, and close the stream if all of it went. A write from here on
    /// reports `EOF`.
    pub(crate) fn flush_now() {
        let Some(endpoint) = host() else {
            return;
        };
        if closed() {
            return;
        }
        FINISHED.store(true, Ordering::SeqCst);
        // A borrow still held (a panic inside the drain itself) leaves the
        // stream as it is rather than panic again.
        if endpoint.stdio.try_borrow_mut().is_err() || endpoint.held.try_borrow_mut().is_err() {
            return;
        }
        let end = RING.tail();
        let mut record = [0u8; RECORD_PAYLOAD];
        loop {
            let pumped = match next(endpoint, end, &mut record) {
                Next::Held(stream, len) => {
                    let fill = |from: usize, out: &mut [u8]| held_bytes(endpoint, from, out);
                    let pumped = pump(endpoint, stream, len, &fill);
                    if matches!(pumped, Pumped::Done) {
                        endpoint.held.borrow_mut().pop_front();
                    }
                    pumped
                }
                Next::Record(stream, len) => {
                    let bytes = record;
                    let fill = |from: usize, out: &mut [u8]| {
                        out.copy_from_slice(&bytes[from..from + out.len()]);
                        out.len()
                    };
                    let pumped = pump(endpoint, stream, len, &fill);
                    if matches!(pumped, Pumped::Done) {
                        RING.consume(len);
                    }
                    pumped
                }
                Next::None => break,
            };
            if !matches!(pumped, Pumped::Done) {
                return;
            }
        }
        let mut stdio = endpoint.stdio.borrow_mut();
        if stdio.output_capacity_left() > 0 {
            let _ = stdio.send_output_now(ProcessOutputEvent::closed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain<const N: usize>(ring: &Ring<N>) -> std::vec::Vec<(Stream, std::vec::Vec<u8>)> {
        let mut out = std::vec::Vec::new();
        let mut record = [0u8; RECORD_PAYLOAD];
        while let Some((stream, len)) = ring.peek(&mut record) {
            out.push((stream, record[..len].to_vec()));
            ring.consume(len);
        }
        out
    }

    /// A write longer than a record is framed into records of at most a
    /// record's payload, in order, and the ring accounts every header.
    #[test]
    fn a_write_is_framed_into_records_and_read_back_in_order() {
        let ring = Ring::<1024>::new();
        let bytes: std::vec::Vec<u8> = (0..600u32).map(|i| i as u8).collect();
        assert!(ring.fits(bytes.len()));
        unsafe { ring.publish(Stream::Out, &bytes) };
        assert_eq!(ring.queued(), framed(600));
        assert_eq!(framed(600), 600 + 3 * HEADER);
        unsafe { ring.publish(Stream::Err, b"oops") };
        let records = drain(&ring);
        let joined: std::vec::Vec<u8> = records
            .iter()
            .filter(|(stream, _)| *stream == Stream::Out)
            .flat_map(|(_, bytes)| bytes.iter().copied())
            .collect();
        assert_eq!(joined, bytes);
        assert_eq!(records.len(), 4);
        assert_eq!(records[3], (Stream::Err, b"oops".to_vec()));
        assert!(
            records
                .iter()
                .all(|(_, bytes)| bytes.len() <= RECORD_PAYLOAD)
        );
        assert_eq!(ring.queued(), 0);
    }

    /// The positions run free across many laps and past `u32::MAX`; every
    /// record still reads back whole, and room is exactly what is unconsumed.
    #[test]
    fn the_ring_wraps_its_bytes_and_its_positions() {
        let ring = Ring::<64>::new();
        ring.head.0.store(u32::MAX - 20, Ordering::Relaxed);
        ring.tail.0.store(u32::MAX - 20, Ordering::Relaxed);
        for round in 0..200u32 {
            let payload: std::vec::Vec<u8> =
                (0..(round % 40) + 1).map(|i| (i + round) as u8).collect();
            assert!(ring.fits(payload.len()), "round {round}");
            unsafe { ring.publish(Stream::Out, &payload) };
            assert!(!ring.fits(64), "a full lap never fits");
            let records = drain(&ring);
            let joined: std::vec::Vec<u8> = records.into_iter().flat_map(|(_, b)| b).collect();
            assert_eq!(joined, payload, "round {round}");
        }
    }

    /// Room is the framed size: a write fits only while its records and
    /// their headers do.
    #[test]
    fn room_counts_headers() {
        let ring = Ring::<16>::new();
        assert!(ring.fits(14));
        assert!(!ring.fits(15));
        unsafe { ring.publish(Stream::Out, &[7; 10]) };
        assert!(ring.fits(2));
        assert!(!ring.fits(3));
    }

    /// A cursor read out of order — the host's moved past a `tail` read
    /// before it — never reads as room the ring does not have.
    #[test]
    fn a_stale_read_never_finds_room_that_is_not_there() {
        let ring = Ring::<16>::new();
        // A reader that saw `tail` = 0, then a `head` the host has advanced
        // to 10, would compute a wrapped, enormous backlog.
        ring.tail.0.store(0, Ordering::Relaxed);
        ring.head.0.store(10, Ordering::Relaxed);
        assert!(!ring.fits(1));
        // The real order, head first, is exact.
        ring.tail.0.store(14, Ordering::Relaxed);
        assert_eq!(ring.queued(), 4);
        assert!(ring.fits(10));
    }

    /// Discarding consumes every published record at once.
    #[test]
    fn discard_drops_everything_published() {
        let ring = Ring::<64>::new();
        unsafe { ring.publish(Stream::Out, b"lost") };
        unsafe { ring.publish(Stream::Err, b"too") };
        ring.discard();
        assert_eq!(ring.queued(), 0);
        assert!(drain(&ring).is_empty());
    }

    /// The largest piece kept whole leaves room for another queued ahead of
    /// it, whatever was framed in between.
    #[test]
    fn a_whole_write_fits_half_the_ring() {
        assert!(framed(WHOLE_WRITE) <= RING_BYTES / 2);
        assert!(framed(WHOLE_WRITE + RECORD_PAYLOAD) > RING_BYTES / 2);
    }

    /// The host's held writes go after the ring records that preceded them
    /// and before the ones that followed.
    #[test]
    fn a_held_write_drains_after_what_preceded_it() {
        let ring = Ring::<256>::new();
        unsafe { ring.publish(Stream::Out, b"a") };
        let stamp = ring.tail();
        unsafe { ring.publish(Stream::Out, b"b") };
        let mut record = [0u8; RECORD_PAYLOAD];
        assert!(!held_is_next(stamp, ring.head().load(Ordering::Relaxed)));
        let (_, len) = ring.peek(&mut record).unwrap();
        assert_eq!(&record[..len], b"a");
        ring.consume(len);
        assert!(held_is_next(stamp, ring.head().load(Ordering::Relaxed)));
    }

    /// A write stamped where a call began goes at once if the cursor already
    /// passed that point, and never if the point is still ahead, across the
    /// positions' wrap.
    #[test]
    fn a_held_write_behind_the_cursor_goes_at_once() {
        for base in [0u32, u32::MAX - 3] {
            let head = base.wrapping_add(10);
            assert!(held_is_next(base, head), "passed at {base}");
            assert!(held_is_next(head, head), "reached at {base}");
            assert!(!held_is_next(head.wrapping_add(1), head), "ahead at {base}");
        }
    }
}
