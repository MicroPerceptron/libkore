//! The **storage transport window** (`res/STORAGE_TRANSPORT.md` §4): the
//! frozen layout one client-funded window holds, and the halves each side
//! attaches.
//!
//! ```text
//! offset          region                                   direction
//! DEMAND_AT       event<SqState, IoUringSqe>  SQ_DEPTH      client → row
//! STREAM_AT       event<SqState, IoUringSqe>  SQ_DEPTH      client → row
//! COMMIT_AT       event<SqState, IoUringSqe>  COMMIT_SQ_DEPTH client → row
//! CQ_AT           event<CqState, IoUringCqe>  CQ_DEPTH      row → client
//! FENCE_AT        [FenceRecord; TIMELINES] on its own page  row → client
//! ```
//!
//! Every submission and completion channel is an [`event`](crate::event)
//! channel, so a waiter gets the wait ladder and a producer's kick is
//! conditional on the other side having parked. The layout is compiled into
//! both sides — nothing in it is negotiated — and the client formats the
//! window before lending it, as a vif client formats its rings
//! (`res/NETWORK_SERVICE.md` §3). The row attaches the formatted window and
//! refuses one whose channels do not validate.
//!
//! The Commit channel is always laid out; whether the row serves it is the
//! ring's policy, not the layout's.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use kore_abi::stor::{
    COMMIT_SQ_DEPTH, CQ_DEPTH, Class, CqState, FENCE_PAGE_BYTES, FenceRecord, GrantId, SQ_DEPTH,
    SqState, StorStatus, TIMELINES,
};
use kore_abi::uring::{IoUringCqe, IoUringSqe, op};
use zerocopy::FromZeros;

use crate::event::{self, EventSink, EventSource};
use crate::spsc::RingError;

const LINE: usize = 64;

const fn up(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

const fn sq_bytes(depth: u32) -> usize {
    event::region_bytes::<SqState, IoUringSqe>(depth)
}

/// Offset of the Demand submission channel.
pub const DEMAND_AT: usize = 0;
/// Offset of the Stream submission channel.
pub const STREAM_AT: usize = up(DEMAND_AT + sq_bytes(SQ_DEPTH), LINE);
/// Offset of the Commit submission channel.
pub const COMMIT_AT: usize = up(STREAM_AT + sq_bytes(SQ_DEPTH), LINE);
/// Offset of the completion channel.
pub const CQ_AT: usize = up(COMMIT_AT + sq_bytes(COMMIT_SQ_DEPTH), LINE);
/// Offset of the fence page.
pub const FENCE_AT: usize = up(
    CQ_AT + event::region_bytes::<CqState, IoUringCqe>(CQ_DEPTH),
    FENCE_PAGE_BYTES,
);
/// Bytes one window occupies — a whole number of pages.
pub const WINDOW_BYTES: usize = FENCE_AT + FENCE_PAGE_BYTES;

const _: () = assert!(WINDOW_BYTES % FENCE_PAGE_BYTES == 0);

/// Offset and depth of `class`'s submission channel.
pub const fn channel(class: Class) -> (usize, u32) {
    match class {
        Class::Demand => (DEMAND_AT, SQ_DEPTH),
        Class::Stream => (STREAM_AT, SQ_DEPTH),
        Class::Commit => (COMMIT_AT, COMMIT_SQ_DEPTH),
    }
}

const fn channel_end(class: Class) -> usize {
    match class {
        Class::Demand => STREAM_AT,
        Class::Stream => COMMIT_AT,
        Class::Commit => CQ_AT,
    }
}

/// Why a window could not be formatted or attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    /// The window does not start on a page boundary.
    Misaligned,
    /// The window is shorter than [`WINDOW_BYTES`].
    TooSmall,
    /// A channel did not validate.
    Channel(Class, RingError),
    /// The completion channel did not validate.
    Completion(RingError),
}

fn check(window: &[u8]) -> Result<(), WindowError> {
    if window.as_ptr().align_offset(FENCE_PAGE_BYTES) != 0 {
        return Err(WindowError::Misaligned);
    }
    if window.len() < WINDOW_BYTES {
        return Err(WindowError::TooSmall);
    }
    Ok(())
}

/// **Format** a fresh window: every channel empty, every fence zero. The
/// client does this before lending the window; exclusive access, not yet
/// shared.
pub fn format(window: &mut [u8]) -> Result<(), WindowError> {
    check(window)?;
    for class in Class::ALL {
        let (at, depth) = channel(class);
        event::init::<SqState, IoUringSqe>(&mut window[at..channel_end(class)], depth)
            .map_err(|error| WindowError::Channel(class, error))?;
    }
    event::init::<CqState, IoUringCqe>(&mut window[CQ_AT..FENCE_AT], CQ_DEPTH)
        .map_err(WindowError::Completion)?;
    window[FENCE_AT..WINDOW_BYTES].fill(0);
    Ok(())
}

/// One timeline's fence, as atomics over a [`FenceRecord`]'s bytes.
#[repr(C)]
pub struct FenceCell {
    done: AtomicU64,
    fault: AtomicU64,
}

const _: () = assert!(core::mem::size_of::<FenceCell>() == core::mem::size_of::<FenceRecord>());

impl FenceCell {
    /// Read the record: `done` with acquire ordering, then `fault` — so a
    /// fault at or below the `done` just read is never missed.
    pub fn load(&self) -> FenceRecord {
        let done = self.done.load(Ordering::Acquire);
        let fault = self.fault.load(Ordering::Acquire);
        FenceRecord { done, fault }
    }

    /// Publish (the row's side): `fault` first, then `done`, both release.
    pub fn publish(&self, record: FenceRecord) {
        self.fault.store(record.fault, Ordering::Release);
        self.done.store(record.done, Ordering::Release);
    }

    /// The low 32 bits of `done` — the word a monitored wait watches
    /// (kore's targets are little-endian, so the low half is the first
    /// word). Any publish that advances `done` by less than 2^32 changes it.
    pub fn done_word(&self) -> &AtomicU32 {
        // SAFETY: `done` is an 8-byte-aligned AtomicU64; its first four
        // bytes are a valid, aligned AtomicU32 on a little-endian target,
        // and every access to them is atomic.
        unsafe { &*(core::ptr::from_ref(&self.done).cast::<AtomicU32>()) }
    }
}

const _: () = assert!(cfg!(target_endian = "little"));

/// The window's fence page.
pub struct Fences<'a> {
    base: *const FenceCell,
    _window: PhantomData<&'a [u8]>,
}

// SAFETY: every access is atomic.
unsafe impl Send for Fences<'_> {}

impl<'a> Fences<'a> {
    /// Timeline `t`'s fence (`1..=TIMELINES`); `None` for 0 or past the end.
    /// The cell lives as long as the window, not this handle: it is all
    /// atomics, so a waiter may hold it while the ring is driven elsewhere.
    pub fn timeline(&self, t: u16) -> Option<&'a FenceCell> {
        let index = usize::from(t).checked_sub(1)?;
        if index >= TIMELINES {
            return None;
        }
        // SAFETY: `base` is the page-aligned fence page of a validated
        // window, holding TIMELINES cells.
        Some(unsafe { &*self.base.add(index) })
    }
}

/// # Safety
/// `base..base + WINDOW_BYTES` is a validated window pinned for `'a`.
unsafe fn fences<'a>(base: *mut u8) -> Fences<'a> {
    Fences {
        // SAFETY: FENCE_AT is inside the window and page-aligned.
        base: unsafe { base.add(FENCE_AT) }.cast::<FenceCell>(),
        _window: PhantomData,
    }
}

/// The client's halves: the submission producers, the completion consumer,
/// the fences.
pub struct ClientRing<'a> {
    /// Submission producers, indexed by [`Class::index`].
    pub sq: [EventSource<'a, SqState, IoUringSqe>; Class::COUNT],
    /// The completion consumer.
    pub cq: EventSink<'a, CqState, IoUringCqe>,
    /// The fence page.
    pub fences: Fences<'a>,
}

/// The row's halves: the submission consumers, the completion producer,
/// the fences.
pub struct ServiceRing<'a> {
    /// Submission consumers, indexed by [`Class::index`].
    pub sq: [EventSink<'a, SqState, IoUringSqe>; Class::COUNT],
    /// The completion producer.
    pub cq: EventSource<'a, CqState, IoUringCqe>,
    /// The fence page.
    pub fences: Fences<'a>,
}

/// The five disjoint regions, as sub-slices of one window.
fn regions(window: &mut [u8]) -> [&mut [u8]; 5] {
    let (demand, rest) = window[..WINDOW_BYTES].split_at_mut(STREAM_AT);
    let (stream, rest) = rest.split_at_mut(COMMIT_AT - STREAM_AT);
    let (commit, rest) = rest.split_at_mut(CQ_AT - COMMIT_AT);
    let (cq, fence) = rest.split_at_mut(FENCE_AT - CQ_AT);
    [demand, stream, commit, cq, fence]
}

fn sq_source(
    class: Class,
    bytes: &mut [u8],
) -> Result<EventSource<'_, SqState, IoUringSqe>, WindowError> {
    event::source::<SqState, IoUringSqe>(bytes).map_err(|error| WindowError::Channel(class, error))
}

fn sq_sink(class: Class, bytes: &mut [u8]) -> Result<EventSink<'_, SqState, IoUringSqe>, WindowError> {
    event::sink::<SqState, IoUringSqe>(bytes).map_err(|error| WindowError::Channel(class, error))
}

/// Attach the **client** halves of a formatted window (its own mapping).
pub fn client(window: &mut [u8]) -> Result<ClientRing<'_>, WindowError> {
    check(window)?;
    let base = window.as_mut_ptr();
    let [demand, stream, commit, cq, _] = regions(window);
    let source = sq_source;
    Ok(ClientRing {
        sq: [
            source(Class::Demand, demand)?,
            source(Class::Stream, stream)?,
            source(Class::Commit, commit)?,
        ],
        cq: event::sink::<CqState, IoUringCqe>(cq).map_err(WindowError::Completion)?,
        // SAFETY: the window was validated and stays borrowed for 'a.
        fences: unsafe { fences(base) },
    })
}

/// Attach the **row's** halves of a formatted window (its own mapping).
pub fn service(window: &mut [u8]) -> Result<ServiceRing<'_>, WindowError> {
    check(window)?;
    let base = window.as_mut_ptr();
    let [demand, stream, commit, cq, _] = regions(window);
    let sink = sq_sink;
    Ok(ServiceRing {
        sq: [
            sink(Class::Demand, demand)?,
            sink(Class::Stream, stream)?,
            sink(Class::Commit, commit)?,
        ],
        cq: event::source::<CqState, IoUringCqe>(cq).map_err(WindowError::Completion)?,
        // SAFETY: as `client`.
        fences: unsafe { fences(base) },
    })
}

/// [`format`] + both sides over one exclusively borrowed window — the
/// same-address-space (and host-test) form, as [`event::split`].
pub fn split(window: &mut [u8]) -> Result<(ClientRing<'_>, ServiceRing<'_>), WindowError> {
    format(window)?;
    let (base, len) = (window.as_mut_ptr(), window.len());
    // SAFETY: the `&mut` borrow pins the window for the returned lifetime;
    // the two sides touch it only through the channel protocols (disjoint
    // roles, atomics, seqlocks, spsc hand-off) and the atomic fences.
    let (a, b) = unsafe {
        (
            core::slice::from_raw_parts_mut(base, len),
            core::slice::from_raw_parts_mut(base, len),
        )
    };
    Ok((client(a)?, service(b)?))
}

/// One submission, as the client builds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    /// What to do.
    pub op: Op,
    /// The grant it acts inside.
    pub grant: GrantId,
    /// Byte offset within the grant (block-aligned).
    pub offset: u64,
    /// Bytes (a block multiple; 0 for a flush).
    pub len: u32,
    /// The target slot (ignored by flush and discard).
    pub target: u16,
    /// Byte offset within the target (block-aligned).
    pub target_offset: u64,
    /// Echoed back on the completion.
    pub user_data: u64,
    /// Timeline `1..=TIMELINES`, or 0 for none.
    pub timeline: u16,
    /// Absolute uptime deadline in ns, or 0 for the class default.
    pub deadline_ns: u64,
}

/// A submission's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Read from the grant into the target.
    Read,
    /// Write from the target into the grant.
    Write,
    /// Flush the device's volatile cache.
    Flush,
    /// Discard a range of the grant.
    Discard,
}

impl Request {
    /// The wire entry.
    pub fn entry(&self) -> IoUringSqe {
        let mut sqe = IoUringSqe::new_zeroed();
        sqe.opcode = match self.op {
            Op::Read => op::READ_FIXED,
            Op::Write => op::WRITE_FIXED,
            Op::Flush => op::FSYNC,
            Op::Discard => op::DISCARD,
        };
        sqe.fd = self.grant.fd();
        sqe.off = self.offset;
        sqe.len = self.len;
        sqe.buf_index = self.target;
        sqe.addr = self.target_offset;
        sqe.user_data = self.user_data;
        sqe.personality = self.timeline;
        sqe.addr3 = self.deadline_ns;
        sqe
    }
}

/// Where a submission lands on its timeline: complete when the fence's
/// `done` reaches `seq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    /// The timeline, `1..=TIMELINES`.
    pub timeline: u16,
    /// The sequence the submission took.
    pub seq: u64,
}

/// A completion, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    /// The submission's token.
    pub user_data: u64,
    /// Bytes moved, or why not.
    pub result: Result<u32, StorStatus>,
}

/// Why a submission was not queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// The class channel is full.
    Full,
    /// The timeline is outside `0..=TIMELINES`.
    Timeline,
}

/// The client side of one ring: the halves plus each timeline's sequence
/// counter, which must match the row's count exactly — every submission
/// naming a timeline takes a sequence, including one that later fails.
pub struct StorClient<'a> {
    ring: ClientRing<'a>,
    issued: [u64; TIMELINES],
}

impl<'a> StorClient<'a> {
    /// Drive `ring`. Its timelines must be fresh (a newly formatted window).
    pub fn new(ring: ClientRing<'a>) -> Self {
        Self {
            ring,
            issued: [0; TIMELINES],
        }
    }

    /// Queue `request` on `class`'s channel (not yet visible to the row —
    /// see [`publish`](Self::publish)). The ticket, when it names a
    /// timeline, is what to wait for.
    pub fn submit(&mut self, class: Class, request: &Request) -> Result<Option<Ticket>, SubmitError> {
        let t = usize::from(request.timeline);
        if t > TIMELINES {
            return Err(SubmitError::Timeline);
        }
        self.ring.sq[class.index()]
            .push(request.entry())
            .map_err(|_| SubmitError::Full)?;
        Ok((t != 0).then(|| {
            self.issued[t - 1] += 1;
            Ticket {
                timeline: request.timeline,
                seq: self.issued[t - 1],
            }
        }))
    }

    /// Make every queued submission on `class` visible to the row. `true`
    /// means the row had parked and is owed a notice (ring its doorbell).
    pub fn publish(&mut self, class: Class) -> bool {
        self.ring.sq[class.index()].commit()
    }

    /// Submissions `class`'s channel can still take.
    pub fn room(&self, class: Class) -> u32 {
        self.ring.sq[class.index()].capacity_left()
    }

    /// The last sequence issued on timeline `t` (0 before the first).
    pub fn issued(&self, t: u16) -> Option<u64> {
        let index = usize::from(t).checked_sub(1)?;
        self.issued.get(index).copied()
    }

    /// The next completion, if one is posted.
    pub fn reap(&mut self) -> Option<Completion> {
        let cqe = self.ring.cq.pop()?;
        let result = if cqe.res >= 0 {
            Ok(cqe.res as u32)
        } else {
            Err(StorStatus::from_res(cqe.res).unwrap_or(StorStatus::Invalid))
        };
        Some(Completion {
            user_data: cqe.user_data,
            result,
        })
    }

    /// Timeline `t`'s fence as it stands.
    pub fn fence(&self, t: u16) -> Option<FenceRecord> {
        self.ring.fences.timeline(t).map(FenceCell::load)
    }

    /// Timeline `t`'s fence cell itself, for the window's whole life — what
    /// a waiter on another thread watches while this client is driven
    /// under someone else's lock.
    pub fn fence_cell(&self, t: u16) -> Option<&'a FenceCell> {
        self.ring.fences.timeline(t)
    }

    /// Whether `ticket` has completed: `Ok(true)` done, `Ok(false)` not yet,
    /// `Err` when a submission at or before it on its timeline failed.
    pub fn poll_ticket(&self, ticket: Ticket) -> Result<bool, StorStatus> {
        let Some(record) = self.fence(ticket.timeline) else {
            return Err(StorStatus::Invalid);
        };
        if record.fault != 0 && record.fault <= ticket.seq {
            return Err(StorStatus::Media);
        }
        Ok(record.done >= ticket.seq)
    }

    /// The word a monitored wait watches for `ticket`'s timeline.
    pub fn fence_word(&self, t: u16) -> Option<&AtomicU32> {
        self.ring.fences.timeline(t).map(FenceCell::done_word)
    }

    /// The completion channel's level state (ring status, counts).
    pub fn state(&self) -> Option<CqState> {
        self.ring.cq.state()
    }

    /// The completion channel, for a wait that parks on it.
    pub fn completions(&self) -> &EventSink<'a, CqState, IoUringCqe> {
        &self.ring.cq
    }
}

/// A byte range widened to whole blocks: what to submit, and where the
/// caller's bytes sit inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Covering {
    /// First byte of the first block.
    pub offset: u64,
    /// Bytes of whole blocks covering the range.
    pub len: u64,
    /// Where the requested range starts inside the covering one.
    pub skew: u64,
}

/// The whole-block range covering `[offset, offset + len)`; `None` for a
/// zero block size or an overflowing end.
pub const fn covering(offset: u64, len: u64, block: u64) -> Option<Covering> {
    if block == 0 {
        return None;
    }
    let Some(end) = offset.checked_add(len) else {
        return None;
    };
    let first = offset - offset % block;
    let Some(last) = end.checked_next_multiple_of(block) else {
        return None;
    };
    Some(Covering {
        offset: first,
        len: last - first,
        skew: offset - first,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    /// A page-aligned heap window, freed on drop.
    struct Page {
        ptr: *mut u8,
        layout: Layout,
    }

    impl Page {
        fn new(bytes: usize) -> Self {
            let layout = Layout::from_size_align(bytes, FENCE_PAGE_BYTES).unwrap();
            // SAFETY: non-zero size.
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }

        fn bytes(&mut self) -> &mut [u8] {
            // SAFETY: owned allocation of `layout.size()` bytes.
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
        }
    }

    impl Drop for Page {
        fn drop(&mut self) {
            // SAFETY: allocated with this layout.
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }

    fn sqe(user_data: u64) -> IoUringSqe {
        let mut entry = IoUringSqe::new_zeroed();
        entry.opcode = op::READ_FIXED;
        entry.user_data = user_data;
        entry
    }

    #[test]
    fn the_layout_is_disjoint_line_aligned_and_page_sized() {
        assert_eq!(DEMAND_AT, 0);
        for class in Class::ALL {
            let (at, depth) = channel(class);
            assert_eq!(at % LINE, 0);
            assert!(at + event::region_bytes::<SqState, IoUringSqe>(depth) <= channel_end(class));
        }
        assert!(CQ_AT + event::region_bytes::<CqState, IoUringCqe>(CQ_DEPTH) <= FENCE_AT);
        assert_eq!(FENCE_AT % FENCE_PAGE_BYTES, 0);
        assert_eq!(WINDOW_BYTES % FENCE_PAGE_BYTES, 0);
    }

    #[test]
    fn a_window_refuses_misalignment_and_shortness() {
        let mut page = Page::new(WINDOW_BYTES + FENCE_PAGE_BYTES);
        let bytes = page.bytes();
        assert_eq!(format(&mut bytes[64..]), Err(WindowError::Misaligned));
        assert_eq!(
            format(&mut bytes[..WINDOW_BYTES - FENCE_PAGE_BYTES]),
            Err(WindowError::TooSmall)
        );
        // Unformatted memory never attaches.
        assert!(matches!(
            service(&mut bytes[..WINDOW_BYTES]),
            Err(WindowError::Channel(Class::Demand, _))
        ));
    }

    #[test]
    fn each_class_channel_carries_its_own_entries_and_the_cq_comes_back() {
        let mut page = Page::new(WINDOW_BYTES);
        let (mut client, mut service) = split(page.bytes()).unwrap();
        for class in Class::ALL {
            client.sq[class.index()]
                .push(sqe(100 + class as u64))
                .unwrap();
            let _ = client.sq[class.index()].commit();
        }
        for class in Class::ALL {
            let entry = service.sq[class.index()].pop().unwrap();
            assert_eq!(entry.user_data, 100 + class as u64);
            assert!(service.sq[class.index()].pop().is_none());
        }
        let mut cqe = IoUringCqe::new_zeroed();
        cqe.user_data = 7;
        cqe.res = 4096;
        service.cq.push(cqe).unwrap();
        service.cq.state(|state| state.completed = 1);
        let _ = service.cq.commit();
        let back = client.cq.pop().unwrap();
        assert_eq!((back.user_data, back.res), (7, 4096));
        assert_eq!(client.cq.state().unwrap().completed, 1);
    }

    #[test]
    fn a_fence_publishes_fault_before_done_and_exposes_its_low_word() {
        let mut page = Page::new(WINDOW_BYTES);
        let (client, service) = split(page.bytes()).unwrap();
        assert!(service.fences.timeline(0).is_none());
        assert!(service.fences.timeline(TIMELINES as u16 + 1).is_none());
        let cell = service.fences.timeline(3).unwrap();
        cell.publish(FenceRecord { done: 5, fault: 4 });
        let seen = client.fences.timeline(3).unwrap();
        assert_eq!(seen.load(), FenceRecord { done: 5, fault: 4 });
        assert_eq!(seen.done_word().load(Ordering::Acquire), 5);
        // Other timelines are untouched.
        assert_eq!(client.fences.timeline(2).unwrap().load(), FenceRecord::default());
    }

    #[test]
    fn covering_widens_to_whole_blocks_and_records_the_skew() {
        assert_eq!(covering(0, 512, 512), Some(Covering { offset: 0, len: 512, skew: 0 }));
        assert_eq!(covering(100, 10, 512), Some(Covering { offset: 0, len: 512, skew: 100 }));
        assert_eq!(covering(500, 20, 512), Some(Covering { offset: 0, len: 1024, skew: 500 }));
        assert_eq!(covering(1024, 0, 512), Some(Covering { offset: 1024, len: 0, skew: 0 }));
        assert_eq!(covering(1, 1, 0), None);
        assert_eq!(covering(u64::MAX, 1, 512), None);
    }

    /// The client counts sequences per timeline exactly as the row will;
    /// a ticket polls against the fence, and a fault at or before it is an
    /// error while a later one is not.
    #[test]
    fn tickets_count_per_timeline_and_poll_against_the_fence() {
        let mut page = Page::new(WINDOW_BYTES);
        let (client, mut service) = split(page.bytes()).unwrap();
        let mut client = StorClient::new(client);
        let request = |timeline, user_data| Request {
            op: Op::Read,
            grant: GrantId::new(0, 0),
            offset: 0,
            len: 512,
            target: 0,
            target_offset: 0,
            user_data,
            timeline,
            deadline_ns: 0,
        };
        let a = client.submit(Class::Demand, &request(1, 1)).unwrap().unwrap();
        let b = client.submit(Class::Stream, &request(1, 2)).unwrap().unwrap();
        let c = client.submit(Class::Demand, &request(2, 3)).unwrap().unwrap();
        assert_eq!(client.submit(Class::Demand, &request(0, 4)).unwrap(), None);
        assert_eq!((a.seq, b.seq, c.seq), (1, 2, 1));
        assert_eq!(
            client.submit(Class::Demand, &request(TIMELINES as u16 + 1, 5)),
            Err(SubmitError::Timeline)
        );
        client.publish(Class::Demand);
        client.publish(Class::Stream);
        let entry = service.sq[Class::Demand.index()].pop().unwrap();
        assert_eq!((entry.opcode, entry.personality, entry.user_data), (op::READ_FIXED, 1, 1));

        assert_eq!(client.poll_ticket(a), Ok(false));
        service.fences.timeline(1).unwrap().publish(FenceRecord { done: 1, fault: 0 });
        assert_eq!(client.poll_ticket(a), Ok(true));
        assert_eq!(client.poll_ticket(b), Ok(false));
        service.fences.timeline(1).unwrap().publish(FenceRecord { done: 2, fault: 2 });
        assert_eq!(client.poll_ticket(a), Ok(true), "the fault is after it");
        assert!(client.poll_ticket(b).is_err(), "the fault is it");

        let mut cqe = IoUringCqe::new_zeroed();
        cqe.user_data = 9;
        cqe.res = StorStatus::Revoked.res();
        service.cq.push(cqe).unwrap();
        let _ = service.cq.commit();
        assert_eq!(
            client.reap(),
            Some(Completion { user_data: 9, result: Err(StorStatus::Revoked) })
        );
        assert_eq!(client.reap(), None);
    }
}
