//! Generic foreground-process launch/result and streaming-I/O pages.

pub mod interrupt;

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;

use kore_abi::cap::Rights;
use kore_abi::monitor::TelemetrySnapshot;
#[cfg(target_os = "none")]
use kore_abi::process::{
    PROCESS_IO_BYTES, PROCESS_IO_INDEX, PROCESS_IO_ROLE, PROCESS_OUTPUT_WAKE_INDEX,
    PROCESS_OUTPUT_WAKE_ROLE, PROCESS_STDIN_WAKE_INDEX, PROCESS_STDIN_WAKE_ROLE,
    PROCESS_STDIO_BYTES, PROCESS_STDIO_INDEX, PROCESS_STDIO_ROLE,
};
use kore_abi::process::{
    PROCESS_OUTPUT_RING_OFFSET, PROCESS_OUTPUT_WAKE_COOKIE, PROCESS_STDIN_WAKE_COOKIE,
    PROCESS_STDIO_CHILD_WAITING_OFFSET, PROCESS_STDIO_PARENT_WAITING_OFFSET,
    PROCESS_STDIO_RING_BYTES, PROCESS_STDIO_RING_CAPACITY, PROCESS_STDIO_RINGS_BYTES,
    PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET, PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET,
    ProcessInputChunk, ProcessIo, ProcessIoError, ProcessOutputEvent,
};
use kore_abi::session::{Notice, SlotRef};
use kore_ring::spsc::{self, Consumer, Full, Producer, RingError};
use zerocopy::FromBytes;

use crate::cap::{ChanRecvToken, ChanSendToken};
#[cfg(target_os = "none")]
use crate::cap::{HeldChanRecvToken, HeldChanSendToken};
use crate::chan::{ChanTx, NoticeRx, RecvError, SendError};
use crate::entry::Rt;
use crate::executor::WakeReason;
use crate::transport::Transport;

/// Initialize the launcher's mapped sibling before it transfers a child
/// sublease.
pub fn initialize(window: &mut [u8], args: &[u8]) -> Result<(), ProcessIoError> {
    let (page, _) = ProcessIo::mut_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.initialize(args, TelemetrySnapshot::EMPTY)
}

/// Initialize arguments plus one bounded immutable launcher-context record.
pub fn initialize_with_context(
    window: &mut [u8],
    args: &[u8],
    context: &[u8],
) -> Result<(), ProcessIoError> {
    let (page, _) = ProcessIo::mut_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.initialize_with_context(args, context, TelemetrySnapshot::EMPTY)
}

/// Publish the launcher's filesystem snapshot, same window as the telemetry
/// snapshot and the same lifecycle point: after founding, before START.
pub fn write_filesystem(
    window: &mut [u8],
    snapshot: kore_abi::monitor::FilesystemSnapshot,
) -> Result<(), ProcessIoError> {
    let (page, _) = ProcessIo::mut_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.write_filesystem(snapshot)
}

/// Publish the launcher's frame-service snapshot — the third producer on
/// the page, same window and same lifecycle point as the filesystem's.
pub fn write_net(
    window: &mut [u8],
    snapshot: kore_abi::monitor::NetSnapshot,
) -> Result<(), ProcessIoError> {
    let (page, _) = ProcessIo::mut_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.write_net(snapshot)
}

/// Publish exec's snapshot after founding and before START.
pub fn write_snapshot(
    window: &mut [u8],
    snapshot: TelemetrySnapshot,
) -> Result<(), ProcessIoError> {
    let (page, _) = ProcessIo::mut_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.write_snapshot(snapshot)
}

/// Read the exited child's bounded result bytes.
pub fn output(window: &[u8]) -> Result<&[u8], ProcessIoError> {
    let (page, _) = ProcessIo::ref_from_prefix(window).map_err(|_| ProcessIoError::Invalid)?;
    page.output()
}

#[derive(Debug)]
enum ProcessStdioSend {
    Owned(ChanSendToken),
    #[cfg(target_os = "none")]
    Held(HeldChanSendToken),
    #[cfg(test)]
    Test(SlotRef),
}

impl ProcessStdioSend {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Owned(token) => token.slot(),
            #[cfg(target_os = "none")]
            Self::Held(token) => token.slot(),
            #[cfg(test)]
            Self::Test(slot) => *slot,
        }
    }
}

#[derive(Debug)]
enum ProcessStdioRecv {
    Owned(ChanRecvToken),
    #[cfg(target_os = "none")]
    Held(HeldChanRecvToken),
    #[cfg(test)]
    Test(SlotRef),
}

impl ProcessStdioRecv {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Owned(token) => token.slot(),
            #[cfg(target_os = "none")]
            Self::Held(token) => token.slot(),
            #[cfg(test)]
            Self::Test(slot) => *slot,
        }
    }
}

/// Result of notifying a process-stream peer after shared state changed.
///
/// All variants mean the ring/sequence publication already happened. Only a
/// [`Full`] return from a `try_send_*` method means the record was not
/// published and may be retried unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStdioWake {
    /// No notice was needed: the peer had not armed its waiting flag. The
    /// sequence publication is sufficient for its later recheck.
    PeerNotWaiting,
    /// The notice landed in the peer's standing executor post.
    Delivered,
    /// The peer's post was already consumed, so it is awake and will perform
    /// the executor's total drain. The duplicate kick is lawfully dropped.
    Coalesced,
    /// The state is visible in shared memory but no wake was proved. This is
    /// a wiring/lifecycle failure, not permission to enqueue the record again.
    Unreached(SendError),
}

/// Conditions a caller wants the wait-side recheck to recognize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStdioWaitFor {
    /// Wake for an incoming record. The caller has no record blocked on a full
    /// outgoing ring.
    Incoming,
    /// Wake for an incoming record or newly available outgoing capacity.
    IncomingOrCapacity,
    /// Wake only for newly available outgoing capacity: the caller holds
    /// output that does not fit, and records arriving meanwhile wait for it.
    /// Waiting for either would return at once while input sits unread.
    Capacity,
}

impl ProcessStdioWaitFor {
    /// Whether a side with `incoming` records unread and `capacity` free
    /// outgoing slots has nothing to wait for.
    const fn satisfied(self, incoming: bool, capacity: bool) -> bool {
        match self {
            Self::Incoming => incoming,
            Self::IncomingOrCapacity => incoming || capacity,
            Self::Capacity => capacity,
        }
    }
}

/// Why a stream-side park attempt returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStdioPark {
    /// A ring condition, peer sequence, or already-landed stream notice made
    /// another total drain necessary without an executor park.
    StateChanged,
    /// The executor resolved one of the composed park arms.
    Wake(WakeReason),
}

/// Why the dedicated alternate wake channel could not be armed or consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStdioWaitError {
    /// The affine receive/post operation refused.
    Receive(RecvError),
    /// A notice arrived on the exact channel with a non-stream cookie.
    InvalidNotice,
    /// The kernel resolved the alternate post but its owner found no notice.
    MissingNotice,
    /// A stream host has no dedicated receive: its peer's notices land on the
    /// host's own executor doorbell. Arm with `arm_shared_doorbell` and park
    /// on the executor instead.
    SharedDoorbell,
}

impl From<RecvError> for ProcessStdioWaitError {
    fn from(value: RecvError) -> Self {
        Self::Receive(value)
    }
}

/// Affine snapshot taken before a total stream drain.
///
/// Pass it to `park_after_drain` only after rechecking every condition the
/// caller owns (incoming records, outgoing capacity, and task lifecycle). The
/// endpoint compares the peer sequence again before it yields, closing the
/// publish-before-park race without cadence polling.
#[must_use]
pub struct ProcessStdioDrain {
    sequence: u32,
    source: *const AtomicU32,
}

/// Scoped publication of this endpoint's final recheck/park interval.
///
/// Cancellation is an ordinary async control-flow edge: dropping the outer
/// stream wait must retract the bit just as surely as a completed alternate
/// executor wake. Otherwise a later peer mutation spends a notice against a
/// waiter that no longer exists and can strand the replacement waiter.
struct ProcessStdioWaiting<'w> {
    word: &'w AtomicU32,
}

impl<'w> ProcessStdioWaiting<'w> {
    fn publish(word: &'w AtomicU32) -> Self {
        word.store(1, Ordering::SeqCst);
        Self { word }
    }
}

impl Drop for ProcessStdioWaiting<'_> {
    fn drop(&mut self) {
        self.word.store(0, Ordering::SeqCst);
    }
}

/// A dedicated wake receive and the notice machinery bound to its slot.
struct PeerReceive<'t, T: Transport> {
    /// Retained solely as the affine ownership witness for `notice`'s slot.
    _recv: ProcessStdioRecv,
    notice: NoticeRx<'t, T>,
}

struct ProcessStdioPeer<'w, 't, T: Transport> {
    transport: &'t T,
    send: ProcessStdioSend,
    /// `None` only for a stream host ([`map_stdio_host`]): the child's output
    /// wake is a send into the host's own executor doorbell, so the host
    /// parks with [`ProcessStdioParent::arm_shared_doorbell`] and has no
    /// receive to arm, consume, or disarm.
    receive: Option<PeerReceive<'t, T>>,
    /// This side's one-writer signal line: increment after every enqueue or
    /// dequeue that can make work/capacity visible to the peer.
    published: &'w AtomicU32,
    /// Set by the peer only across its final recheck + alternate park. This
    /// side swaps it clear before sending, so a burst costs one notice.
    peer_waiting: &'w AtomicU32,
    /// The peer's one-writer line, observed around drain → park.
    observed: &'w AtomicU32,
    /// This side's waiting publication paired with `observed`.
    waiting: &'w AtomicU32,
    send_cookie: u64,
    recv_cookie: u64,
}

impl<'w, T: Transport> ProcessStdioPeer<'w, '_, T> {
    fn begin_drain(&self) -> ProcessStdioDrain {
        ProcessStdioDrain {
            sequence: self.observed.load(Ordering::SeqCst),
            source: self.observed,
        }
    }

    async fn signal(&self) -> ProcessStdioWake {
        match self.publish() {
            Ok(notice) => Self::wake_outcome(
                ChanTx::new(self.transport, self.send.slot())
                    .send(notice)
                    .await,
            ),
            Err(wake) => wake,
        }
    }

    /// [`signal`](Self::signal), with the notice committed synchronously.
    fn signal_now(&self) -> ProcessStdioWake {
        match self.publish() {
            Ok(notice) => {
                Self::wake_outcome(ChanTx::new(self.transport, self.send.slot()).send_now(notice))
            }
            Err(wake) => wake,
        }
    }

    /// Publish this side's change, and name the notice owed to a waiting
    /// peer (`Err` when none is owed).
    fn publish(&self) -> Result<Notice, ProcessStdioWake> {
        // Ring cursor publication happens before this call. SeqCst is the
        // register/check fence shared with the address-monitor first tier and
        // makes the post-change value the notice's diagnostic payload.
        let sequence = self
            .published
            .fetch_add(1, Ordering::SeqCst)
            .wrapping_add(1);
        if self.peer_waiting.swap(0, Ordering::SeqCst) == 0 {
            return Err(ProcessStdioWake::PeerNotWaiting);
        }
        Ok(Notice {
            cookie: self.send_cookie,
            payload: u64::from(sequence),
        })
    }

    fn wake_outcome(sent: Result<(), SendError>) -> ProcessStdioWake {
        match sent {
            Ok(()) => ProcessStdioWake::Delivered,
            Err(SendError::WouldBlock) => ProcessStdioWake::Coalesced,
            Err(error) => ProcessStdioWake::Unreached(error),
        }
    }

    fn notice_is_valid(&self, notice: Notice) -> bool {
        notice.cookie == self.recv_cookie
    }

    fn close_after_recv_root_revoked(self) {
        let Self { send, receive, .. } = self;
        if let Some(PeerReceive { _recv: recv, notice }) = receive {
            notice.close_after_root_revoked();
            // Root revocation made both endpoint records dead before their
            // affine wrappers release local dynamic-arena names.
            drop(recv);
        }
        drop(send);
    }

    async fn park_after_drain_inner(
        &mut self,
        drain: ProcessStdioDrain,
        mut ready: impl FnMut() -> bool,
        deadline_ns: Option<u64>,
    ) -> Result<ProcessStdioPark, ProcessStdioWaitError> {
        loop {
            match self.prepare_park(&drain, &mut ready)? {
                ProcessStdioParkStep::Changed => return Ok(ProcessStdioPark::StateChanged),
                // A rotating notice receiver cannot reuse the preceding
                // landing.  Let that handoff finish, then post its fresh
                // landing before publishing the process-stream wait state.
                ProcessStdioParkStep::Handoff => {
                    crate::executor::next_wake_for(100_000).await;
                }
                ProcessStdioParkStep::Park(ticket) => {
                    let reason = match deadline_ns {
                        Some(delta_ns) => {
                            crate::executor::next_wake_on_for(ticket.post, delta_ns).await
                        }
                        None => crate::executor::next_wake_on(ticket.post).await,
                    };
                    return self.settle_park(ticket, reason);
                }
            }
        }
    }

    /// Everything a park does before it waits, synchronously: recheck the
    /// drained state, take one monitor turn, consume a notice that already
    /// landed, and publish `waiting` with the standing post armed. The wait
    /// itself is the caller's, so an owner that shares this endpoint can
    /// await it without holding the endpoint (see [`ProcessStdioParkStep`]).
    fn prepare_park(
        &mut self,
        drain: &ProcessStdioDrain,
        mut ready: impl FnMut() -> bool,
    ) -> Result<ProcessStdioParkStep<'w>, ProcessStdioWaitError> {
        assert!(
            core::ptr::eq(drain.source, self.observed),
            "process-stream drain token belongs to another endpoint"
        );
        if self.observed.load(Ordering::SeqCst) != drain.sequence || ready() {
            return Ok(ProcessStdioParkStep::Changed);
        }

        // Cross-process mappings can occupy different virtual addresses, so
        // `futex::wait_async`'s process-local address registry cannot name
        // this peer. A backend that can enforce this turn's cap may still use
        // its architecture monitor as a first tier; unsupported,
        // uncalibrated, or baseline-AArch64 targets execute one spin hint
        // before the real channel park.
        crate::hint::wait_on(self.observed, drain.sequence, Duration::from_micros(1));
        if self.observed.load(Ordering::SeqCst) != drain.sequence || ready() {
            return Ok(ProcessStdioParkStep::Changed);
        }

        // Arm the dedicated receive before publishing `waiting`. If a prior
        // signal already landed, consume it and re-run the caller's drain
        // without parking.
        let recv_cookie = self.recv_cookie;
        let Some(receive) = self.receive.as_mut() else {
            return Err(ProcessStdioWaitError::SharedDoorbell);
        };
        match receive.notice.try_recv() {
            Ok(Some(notice)) => {
                return if notice.cookie == recv_cookie {
                    Ok(ProcessStdioParkStep::Changed)
                } else {
                    Err(ProcessStdioWaitError::InvalidNotice)
                };
            }
            Ok(None) => {}
            Err(RecvError::Refused(kore_abi::session::Status::WouldBlock)) => {
                return Ok(ProcessStdioParkStep::Handoff);
            }
            Err(error) => return Err(ProcessStdioWaitError::Receive(error)),
        }
        let Some(post) = receive.notice.standing_post() else {
            return Err(ProcessStdioWaitError::Receive(RecvError::Refused(
                kore_abi::session::Status::Malformed,
            )));
        };

        // Register, then recheck: paired with the producer's state change →
        // epoch increment → waiting swap. Either this load sees the new epoch
        // or the producer sees `waiting` and spends the standing post.
        let waiting = ProcessStdioWaiting::publish(self.waiting);
        if self.observed.load(Ordering::SeqCst) != drain.sequence || ready() {
            return Ok(ProcessStdioParkStep::Changed);
        }
        // The task's ordinary executor doorbell remains the primary arm;
        // this affine post is the one bounded alternate. No service reply or
        // process stream can shadow the other.
        Ok(ProcessStdioParkStep::Park(ProcessStdioParkTicket {
            post,
            _waiting: waiting,
        }))
    }

    /// Everything a park does after its wait: withdraw `waiting`, then
    /// consume the notice the wake may have landed.
    fn settle_park(
        &mut self,
        ticket: ProcessStdioParkTicket<'_>,
        reason: WakeReason,
    ) -> Result<ProcessStdioPark, ProcessStdioWaitError> {
        drop(ticket);
        let Some(receive) = self.receive.as_mut() else {
            return Err(ProcessStdioWaitError::SharedDoorbell);
        };
        let delivered = if reason == WakeReason::Delivery {
            receive.notice.settle_delivery()?
        } else {
            receive.notice.try_recv()?
        };
        match (reason, delivered) {
            (_, Some(notice)) if self.notice_is_valid(notice) => Ok(ProcessStdioPark::StateChanged),
            (_, Some(_)) => Err(ProcessStdioWaitError::InvalidNotice),
            // A resolved post with no record is a spurious alternate wake:
            // the post was re-armed, and the state re-check above is truth.
            (WakeReason::Delivery, None) => Ok(ProcessStdioPark::Wake(reason)),
            // The primary doorbell won; the alternate post remains standing
            // and `try_recv` correctly left it armed for the next pass.
            (reason, None) => Ok(ProcessStdioPark::Wake(reason)),
        }
    }
}

/// One park, split at its wait.
///
/// [`ProcessStdioChild::park_after_drain`] awaits a whole park while it holds
/// the endpoint. An owner that shares the endpoint among several streams —
/// WASI's stdin, stdout and stderr resources over one process stream, which
/// `wasi:io/poll` may wait on together — must not hold it across the wait.
/// It calls [`ProcessStdioChild::prepare_park`], awaits
/// [`crate::executor::next_wake_on`] with the ticket's post *without* the
/// endpoint, and hands the ticket back to
/// [`ProcessStdioChild::settle_park`]. `park_after_drain` is exactly that
/// sequence, so the two cannot drift.
#[must_use = "a park step either reports a change or names the wait to perform"]
pub enum ProcessStdioParkStep<'w> {
    /// The drained state changed, or a notice already landed: re-drain.
    Changed,
    /// The notice receiver is mid-rotation: wait briefly
    /// (`next_wake_for`), then prepare again.
    Handoff,
    /// Await [`next_wake_on`](crate::executor::next_wake_on) on
    /// [`post`](ProcessStdioParkTicket::post), then settle.
    Park(ProcessStdioParkTicket<'w>),
}

/// A published wait: the standing post to park on, and this side's `waiting`
/// flag, withdrawn when the ticket is settled or dropped.
#[must_use = "settle the ticket after the wait, or drop it to withdraw the wait"]
pub struct ProcessStdioParkTicket<'w> {
    post: crate::transport::ParkPost,
    _waiting: ProcessStdioWaiting<'w>,
}

impl ProcessStdioParkTicket<'_> {
    /// The standing notice post to park on.
    pub fn post(&self) -> crate::transport::ParkPost {
        self.post
    }
}

/// Launcher's affine endpoint for one foreground-process stream.
///
/// The rings are private so every successful dequeue also publishes newly
/// available capacity and every enqueue publishes data before issuing the
/// directional kick. Construct this only by consuming
/// [`PreparedProcessStdio::attach_parent`] (the launcher's own end), or, in a
/// stream host, through [`map_stdio_host`] (an end the launcher delegated).
#[must_use = "retain the stream endpoint through task quiescence, then close it after both wake roots are revoked"]
pub struct ProcessStdioParent<'w, 't, T: Transport> {
    stdin: Producer<'w, ProcessInputChunk>,
    output: Consumer<'w, ProcessOutputEvent>,
    peer: ProcessStdioPeer<'w, 't, T>,
}

impl<T: Transport> ProcessStdioParent<'_, '_, T> {
    /// Consume this endpoint after both launcher-retained wake-channel roots
    /// have been successfully revoked.
    ///
    /// A deadline or primary-doorbell win can leave the output receive post
    /// standing even after `Closed` was drained. Root revocation is the proof
    /// that no late child notice can land; only then may the runtime-owned
    /// landing be disarmed and its arena reservation reused. Keep `self` live
    /// across the child's terminal wait, retry both root retirements until
    /// they return a closed outcome, then call this method before reusing the
    /// launch arena. Dropping an armed endpoint earlier fails safe by
    /// quarantining that landing, but leaks its bounded arena name.
    pub fn close_after_wake_roots_revoked(self) {
        let Self {
            stdin: _,
            output: _,
            peer,
        } = self;
        peer.close_after_recv_root_revoked();
    }

    /// Snapshot the child-to-parent sequence before a total drain.
    pub fn begin_drain(&self) -> ProcessStdioDrain {
        self.peer.begin_drain()
    }

    /// Arm a stream whose child output kick was wired to the owner's shared
    /// executor doorbell. The owner must drain every such stream on each wake.
    /// False requests another drain before parking; the sequence/flag fence
    /// covers publication racing this registration. No alternate post is used.
    pub fn arm_shared_doorbell(&self, drain: ProcessStdioDrain) -> bool {
        assert!(core::ptr::eq(drain.source, self.peer.observed));
        self.peer.waiting.store(1, Ordering::SeqCst);
        self.peer.observed.load(Ordering::SeqCst) == drain.sequence
            && self.output.outstanding() == 0
    }

    /// Enqueue one canonical-input record and notify the child. [`Full`]
    /// alone means nothing was published.
    pub async fn try_send_stdin(
        &mut self,
        record: ProcessInputChunk,
    ) -> Result<ProcessStdioWake, Full> {
        self.stdin.push(record)?;
        Ok(self.peer.signal().await)
    }

    /// Dequeue one output/control event. A successful dequeue also notifies
    /// the child that output-ring capacity became available.
    pub async fn try_recv_output(&mut self) -> Option<(ProcessOutputEvent, ProcessStdioWake)> {
        let record = self.output.pop()?;
        Some((record, self.peer.signal().await))
    }

    /// Advisory free slots in the input producer.
    pub fn stdin_capacity_left(&self) -> u32 {
        self.stdin.capacity_left()
    }

    /// Advisory input records the child has not consumed yet — how a
    /// launcher learns that the child read an interrupt it sent last.
    pub fn stdin_unconsumed(&self) -> u32 {
        self.stdin.capacity().saturating_sub(self.stdin.capacity_left())
    }

    /// Advisory published output events awaiting this parent.
    pub fn output_outstanding(&self) -> u32 {
        self.output.outstanding()
    }

    /// Recheck the rings/peer epoch, take one hardware-monitor turn when
    /// available, then compose the dedicated output notice post with the
    /// launcher's ordinary executor doorbell.
    pub async fn park_after_drain(
        &mut self,
        drain: ProcessStdioDrain,
        wait_for: ProcessStdioWaitFor,
    ) -> Result<(), ProcessStdioWaitError> {
        let stdin = &self.stdin;
        let output = &self.output;
        self.peer
            .park_after_drain_inner(
                drain,
                || wait_for.satisfied(output.outstanding() != 0, stdin.capacity_left() != 0),
                None,
            )
            .await
            .map(|_| ())
    }

    /// Deadline-bounded sibling of [`park_after_drain`](Self::park_after_drain).
    ///
    /// Stream data and capacity remain notice/epoch driven. The deadline is
    /// only a composition seam for lifecycle state that has no stream wake
    /// edge (for example, detecting a child that faults before publishing its
    /// terminal `Closed` event). Callers should use a low-frequency deadline,
    /// poll lifecycle only on return, and switch to a blocking task wait after
    /// `Closed` makes further stream output illegal.
    pub async fn park_after_drain_for(
        &mut self,
        drain: ProcessStdioDrain,
        wait_for: ProcessStdioWaitFor,
        delta_ns: u64,
    ) -> Result<ProcessStdioPark, ProcessStdioWaitError> {
        let stdin = &self.stdin;
        let output = &self.output;
        self.peer
            .park_after_drain_inner(
                drain,
                || wait_for.satisfied(output.outstanding() != 0, stdin.capacity_left() != 0),
                Some(delta_ns),
            )
            .await
    }
}

/// Child's direction-reversed affine endpoint.
#[must_use = "retain the stream endpoint until task teardown or close it after both wake roots are revoked"]
pub struct ProcessStdioChild<'w, 't, T: Transport> {
    stdin: Consumer<'w, ProcessInputChunk>,
    output: Producer<'w, ProcessOutputEvent>,
    peer: ProcessStdioPeer<'w, 't, T>,
}

impl<'w, T: Transport> ProcessStdioChild<'w, '_, T> {
    /// Gracefully consume a reusable child endpoint after its owner has proof
    /// that both wake roots were revoked. Ordinary process children may
    /// instead retain the endpoint until task teardown destroys their arena.
    pub fn close_after_wake_roots_revoked(self) {
        let Self {
            stdin: _,
            output: _,
            peer,
        } = self;
        peer.close_after_recv_root_revoked();
    }

    /// Snapshot the parent-to-child sequence before a total drain.
    pub fn begin_drain(&self) -> ProcessStdioDrain {
        self.peer.begin_drain()
    }

    /// Dequeue one canonical-input record and notify the parent that input
    /// capacity became available.
    pub async fn try_recv_stdin(&mut self) -> Option<(ProcessInputChunk, ProcessStdioWake)> {
        let record = self.stdin.pop()?;
        Some((record, self.peer.signal().await))
    }

    /// Enqueue one output/control event and notify the parent. [`Full`] alone
    /// means nothing was published.
    pub async fn try_send_output(
        &mut self,
        record: ProcessOutputEvent,
    ) -> Result<ProcessStdioWake, Full> {
        self.output.push(record)?;
        Ok(self.peer.signal().await)
    }

    /// [`try_recv_stdin`](Self::try_recv_stdin), with the parent's
    /// capacity notice committed synchronously — for a stream API whose
    /// read cannot await (WASI's `input-stream.read`).
    pub fn recv_stdin_now(&mut self) -> Option<(ProcessInputChunk, ProcessStdioWake)> {
        let record = self.stdin.pop()?;
        Some((record, self.peer.signal_now()))
    }

    /// [`try_send_output`](Self::try_send_output), with the parent's data
    /// notice committed synchronously. [`Full`] alone means nothing was
    /// published.
    pub fn send_output_now(
        &mut self,
        record: ProcessOutputEvent,
    ) -> Result<ProcessStdioWake, Full> {
        self.output.push(record)?;
        Ok(self.peer.signal_now())
    }

    /// Advisory published input records awaiting this child.
    pub fn stdin_outstanding(&self) -> u32 {
        self.stdin.outstanding()
    }

    /// Advisory free slots in the output producer.
    pub fn output_capacity_left(&self) -> u32 {
        self.output.capacity_left()
    }

    /// Recheck the rings/peer epoch, take one hardware-monitor turn when
    /// available, then compose the dedicated stdin notice post with the
    /// child's ordinary executor doorbell.
    pub async fn park_after_drain(
        &mut self,
        drain: ProcessStdioDrain,
        wait_for: ProcessStdioWaitFor,
    ) -> Result<(), ProcessStdioWaitError> {
        let (stdin, output) = (&self.stdin, &self.output);
        self.peer
            .park_after_drain_inner(drain, || stdio_ready(stdin, output, wait_for), None)
            .await
            .map(|_| ())
    }

    /// Deadline-bounded sibling of [`park_after_drain`](Self::park_after_drain)
    /// for child runtimes that must compose a non-stream lifecycle probe.
    /// Stream progress never depends on this deadline.
    pub async fn park_after_drain_for(
        &mut self,
        drain: ProcessStdioDrain,
        wait_for: ProcessStdioWaitFor,
        delta_ns: u64,
    ) -> Result<ProcessStdioPark, ProcessStdioWaitError> {
        let (stdin, output) = (&self.stdin, &self.output);
        self.peer
            .park_after_drain_inner(
                drain,
                || stdio_ready(stdin, output, wait_for),
                Some(delta_ns),
            )
            .await
    }

    /// The first half of [`park_after_drain`](Self::park_after_drain),
    /// synchronously: see [`ProcessStdioParkStep`] for the split and why an
    /// owner sharing this endpoint needs it.
    ///
    /// The drain token is borrowed: after a
    /// [`Handoff`](ProcessStdioParkStep::Handoff) the same drain is offered
    /// again.
    pub fn prepare_park(
        &mut self,
        drain: &ProcessStdioDrain,
        wait_for: ProcessStdioWaitFor,
    ) -> Result<ProcessStdioParkStep<'w>, ProcessStdioWaitError> {
        let (stdin, output) = (&self.stdin, &self.output);
        self.peer
            .prepare_park(drain, || stdio_ready(stdin, output, wait_for))
    }

    /// The second half: withdraw the wait and consume what the wake landed.
    pub fn settle_park(
        &mut self,
        ticket: ProcessStdioParkTicket<'_>,
        reason: WakeReason,
    ) -> Result<ProcessStdioPark, ProcessStdioWaitError> {
        self.peer.settle_park(ticket, reason)
    }
}

/// Whether a child's park has nothing to wait for: input arrived, or output
/// capacity opened, as `wait_for` asks.
fn stdio_ready(
    stdin: &Consumer<'_, ProcessInputChunk>,
    output: &Producer<'_, ProcessOutputEvent>,
    wait_for: ProcessStdioWaitFor,
) -> bool {
    wait_for.satisfied(stdin.outstanding() != 0, output.capacity_left() != 0)
}

/// A freshly formatted parent attachment, before its directional wake
/// capability is known.
///
/// This token is affine and holds the exclusive window borrow: safe code can
/// construct the parent producer/consumer pair exactly once. Launchers whose
/// mapping owner outlives an async spawn may create the input slice from its
/// affine mapping's raw pointer (the same unsafe lifetime proof mapping APIs
/// require), but they still cannot duplicate this token in safe code.
#[must_use]
pub struct PreparedProcessStdio<'w> {
    base: *mut u8,
    len: usize,
    _window: PhantomData<&'w mut [u8]>,
}

impl<'w> PreparedProcessStdio<'w> {
    /// Consume this one-shot attachment and bind the launcher's retained
    /// endpoints. Their two `ChannelRoot`s stay with the launcher as the
    /// lifecycle pivots; this endpoint owns only the stdin send and output
    /// receive leaves.
    pub fn attach_parent<'t, T: Transport>(
        self,
        transport: &'t T,
        stdin_send: ChanSendToken,
        output_recv: ChanRecvToken,
    ) -> Result<ProcessStdioParent<'w, 't, T>, RingError> {
        let (base, len) = (self.base, self.len);
        // SAFETY: this affine token owns the exclusive local attachment and
        // its `'w` borrow pins the mapping. Consuming it proves no second safe
        // parent binding can be constructed.
        unsafe {
            bind_parent(
                base,
                len,
                transport,
                ProcessStdioSend::Owned(stdin_send),
                Some(ProcessStdioRecv::Owned(output_recv)),
            )
        }
    }

    #[cfg(test)]
    fn into_raw(self) -> (*mut u8, usize) {
        (self.base, self.len)
    }
}

const _: () = {
    // The ABI pins geometry without depending upward on `kore-ring`; this is
    // the integration assertion that prevents an SPSC layout revision from
    // silently shifting either fixed offset.
    assert!(
        spsc::region_bytes::<ProcessInputChunk>(PROCESS_STDIO_RING_CAPACITY)
            == PROCESS_STDIO_RING_BYTES
    );
    assert!(
        spsc::region_bytes::<ProcessOutputEvent>(PROCESS_STDIO_RING_CAPACITY)
            == PROCESS_STDIO_RING_BYTES
    );
};

struct StdioRegions<'w> {
    stdin: &'w mut [u8],
    output: &'w mut [u8],
    to_child: &'w AtomicU32,
    child_waiting: &'w AtomicU32,
    to_parent: &'w AtomicU32,
    parent_waiting: &'w AtomicU32,
}

fn stdio_regions(window: &mut [u8]) -> Result<StdioRegions<'_>, RingError> {
    if window.len() < kore_abi::process::PROCESS_STDIO_BYTES {
        return Err(RingError::TooSmall);
    }
    if window.as_ptr().align_offset(64) != 0 {
        return Err(RingError::Misaligned);
    }
    let page = &mut window[..kore_abi::process::PROCESS_STDIO_BYTES];
    let (rings, signals) = page.split_at_mut(PROCESS_STDIO_RINGS_BYTES);
    let (stdin, output) = rings.split_at_mut(PROCESS_OUTPUT_RING_OFFSET);
    let output = &mut output[..PROCESS_STDIO_RING_BYTES];
    let (to_child, to_parent) = signals.split_at_mut(64);
    let to_child_ptr = to_child.as_ptr();
    let to_parent_ptr = to_parent.as_ptr();
    // SAFETY: the page is 64-byte aligned; both ABI offsets are cache-line
    // aligned and in bounds. AtomicU32 accepts the zeroed bit pattern and is
    // accessed only atomically after publication.
    let to_child = unsafe { &*(to_child_ptr as *const AtomicU32) };
    // SAFETY: the waiting flag is the next naturally aligned u32 in the same
    // dedicated cache line, at the ABI-pinned offset.
    let child_waiting = unsafe {
        &*(to_child_ptr
            .add(PROCESS_STDIO_CHILD_WAITING_OFFSET - PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET)
            .cast::<AtomicU32>())
    };
    let to_parent = unsafe { &*(to_parent_ptr as *const AtomicU32) };
    // SAFETY: as `child_waiting`, in the other directional line.
    let parent_waiting = unsafe {
        &*(to_parent_ptr
            .add(PROCESS_STDIO_PARENT_WAITING_OFFSET - PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET)
            .cast::<AtomicU32>())
    };
    Ok(StdioRegions {
        stdin,
        output,
        to_child,
        child_waiting,
        to_parent,
        parent_waiting,
    })
}

/// Format a fresh stream page before the launcher transfers the child's
/// sibling view.
///
/// The entire page is cleared, including the ABI-reserved suffix, before the
/// two fixed-depth rings and the directional sequence lines are published.
/// The returned affine token is the only safe parent attachment.
pub fn init_stdio(window: &mut [u8]) -> Result<PreparedProcessStdio<'_>, RingError> {
    if window.len() < kore_abi::process::PROCESS_STDIO_BYTES {
        return Err(RingError::TooSmall);
    }
    if window.as_ptr().align_offset(64) != 0 {
        return Err(RingError::Misaligned);
    }
    window[..kore_abi::process::PROCESS_STDIO_BYTES].fill(0);
    let regions = stdio_regions(window)?;
    spsc::init::<ProcessInputChunk>(regions.stdin, PROCESS_STDIO_RING_CAPACITY)?;
    spsc::init::<ProcessOutputEvent>(regions.output, PROCESS_STDIO_RING_CAPACITY)?;
    Ok(PreparedProcessStdio {
        base: window.as_mut_ptr(),
        len: kore_abi::process::PROCESS_STDIO_BYTES,
        _window: PhantomData,
    })
}

/// Bind the parent end. `output_recv` is `None` only for a stream host,
/// whose child output wake rings the host's own executor doorbell.
unsafe fn bind_parent<'w, 't, T: Transport>(
    base: *mut u8,
    len: usize,
    transport: &'t T,
    stdin_send: ProcessStdioSend,
    output_recv: Option<ProcessStdioRecv>,
) -> Result<ProcessStdioParent<'w, 't, T>, RingError> {
    // SAFETY: the caller owns this local mapping attachment for `'w`.
    let window = unsafe { core::slice::from_raw_parts_mut(base, len) };
    let regions = stdio_regions(window)?;
    Ok(ProcessStdioParent {
        stdin: spsc::producer::<ProcessInputChunk>(regions.stdin)?,
        output: spsc::consumer::<ProcessOutputEvent>(regions.output)?,
        peer: ProcessStdioPeer {
            transport,
            send: stdin_send,
            receive: output_recv.map(|recv| PeerReceive {
                notice: NoticeRx::new(transport, recv.slot()),
                _recv: recv,
            }),
            published: regions.to_child,
            peer_waiting: regions.child_waiting,
            observed: regions.to_parent,
            waiting: regions.parent_waiting,
            send_cookie: PROCESS_STDIN_WAKE_COOKIE,
            recv_cookie: PROCESS_OUTPUT_WAKE_COOKIE,
        },
    })
}

#[cfg(any(test, target_os = "none"))]
unsafe fn bind_child<'w, 't, T: Transport>(
    base: *mut u8,
    len: usize,
    transport: &'t T,
    output_send: ProcessStdioSend,
    stdin_recv: ProcessStdioRecv,
) -> Result<ProcessStdioChild<'w, 't, T>, RingError> {
    // SAFETY: the caller owns this local mapping attachment for `'w`.
    let window = unsafe { core::slice::from_raw_parts_mut(base, len) };
    let regions = stdio_regions(window)?;
    Ok(ProcessStdioChild {
        stdin: spsc::consumer::<ProcessInputChunk>(regions.stdin)?,
        output: spsc::producer::<ProcessOutputEvent>(regions.output)?,
        peer: ProcessStdioPeer {
            transport,
            send: output_send,
            receive: Some(PeerReceive {
                notice: NoticeRx::new(transport, stdin_recv.slot()),
                _recv: stdin_recv,
            }),
            published: regions.to_parent,
            peer_waiting: regions.parent_waiting,
            observed: regions.to_child,
            waiting: regions.child_waiting,
            send_cookie: PROCESS_OUTPUT_WAKE_COOKIE,
            recv_cookie: PROCESS_STDIN_WAKE_COOKIE,
        },
    })
}

#[cfg(test)]
fn split_stdio<'w, 'p, 'c, P: Transport, C: Transport>(
    window: &'w mut [u8],
    parent_transport: &'p P,
    child_transport: &'c C,
) -> Result<(ProcessStdioParent<'w, 'p, P>, ProcessStdioChild<'w, 'c, C>), RingError> {
    let prepared = init_stdio(window)?;
    let (base, len) = prepared.into_raw();
    // SAFETY: this is the same-address-space test form of the cross-mapping
    // protocol. Exactly one direction-specific producer and consumer are
    // constructed, and the original exclusive borrow lasts for `'w`.
    unsafe {
        Ok((
            bind_parent(
                base,
                len,
                parent_transport,
                ProcessStdioSend::Test(SlotRef::from_wire(41)),
                Some(ProcessStdioRecv::Test(SlotRef::from_wire(42))),
            )?,
            bind_child(
                base,
                len,
                child_transport,
                ProcessStdioSend::Test(SlotRef::from_wire(43)),
                ProcessStdioRecv::Test(SlotRef::from_wire(44)),
            )?,
        ))
    }
}

/// Map this process's delivered I/O view.
///
/// The launcher does not touch its retained sibling mapping between child
/// START and terminal WAIT. Before resolving the immutable boot coordinate,
/// this spends a one-shot claim in the task's runtime block, so safe code
/// cannot create a second mutable accessor by reconstructing [`Rt`]. Task
/// teardown owns the installed view afterward.
#[cfg(target_os = "none")]
pub fn map(rt: &Rt) -> Option<&'static mut ProcessIo> {
    if !rt.claim_process_io_attachment() {
        return None;
    }
    let slot = rt.bootstrap().handle(PROCESS_IO_ROLE, PROCESS_IO_INDEX)?;
    let view = crate::cap::verbs::take_view(slot)?;
    let mapped = crate::cap::verbs::map_view(view)?;
    if mapped.len() < PROCESS_IO_BYTES
        || !(mapped.cast::<u8>().as_ptr() as usize).is_multiple_of(64)
    {
        return None;
    }
    // SAFETY: `map_view` installed this task's writable, page-aligned sibling
    // view and `ProcessIo` is exactly one page with 64-byte alignment.
    let bytes =
        unsafe { core::slice::from_raw_parts_mut(mapped.cast::<u8>().as_ptr(), PROCESS_IO_BYTES) };
    ProcessIo::mut_from_bytes(bytes).ok()
}

/// Map and attach this child's delivered foreground-process stream view.
///
/// The founder retains the root and a sibling mapping; this task receives one
/// fresh sublease-backed sibling view. This spends the task-runtime attachment
/// claim before resolving its immutable boot coordinates; reconstructing
/// [`Rt`] therefore cannot build a second producer/consumer pair. The SPSC
/// direction split is the concurrency invariant: the child only consumes
/// stdin and only produces output. The shared view must carry `ACCESS|MODIFY`:
/// both halves mutate their local ring cursor even when the payload direction
/// is otherwise one-way. Each delivered wake endpoint needs only `ACCESS`
/// here; the launch contract grants exactly
/// [`kore_abi::process::PROCESS_STDIO_WAKE_RIGHTS`] (`ACCESS|REVOKE`) and
/// deliberately withholds `DERIVE`. Task teardown owns the installed mapping
/// after the returned affine endpoint is dropped.
pub type MappedProcessStdioChild =
    ProcessStdioChild<'static, 'static, crate::transport::TrapTransport>;

#[cfg(any(target_os = "none", test))]
fn process_stdio_view_is_writable(rights: Rights) -> bool {
    rights.contains(Rights::ACCESS | Rights::MODIFY)
}

#[cfg(target_os = "none")]
pub fn map_stdio(rt: &Rt) -> Option<MappedProcessStdioChild> {
    if !rt.claim_process_stdio_attachment() {
        return None;
    }
    let view_slot = rt
        .bootstrap()
        .handle(PROCESS_STDIO_ROLE, PROCESS_STDIO_INDEX)?;
    let stdin_recv_slot = rt
        .bootstrap()
        .handle(PROCESS_STDIN_WAKE_ROLE, PROCESS_STDIN_WAKE_INDEX)?;
    let output_send_slot = rt
        .bootstrap()
        .handle(PROCESS_OUTPUT_WAKE_ROLE, PROCESS_OUTPUT_WAKE_INDEX)?;
    let view = crate::cap::verbs::take_view(view_slot)?;
    // Both SPSC halves mutate their local head/tail cursor and this child is
    // the stdin consumer plus output producer. `map_view` lawfully installs a
    // read-only PTE when MODIFY is absent, so accepting that view here would
    // defer a capability-contract refusal into a first-I/O page fault.
    if !process_stdio_view_is_writable(view.rights()) {
        return None;
    }
    let stdin_recv = crate::cap::verbs::take_chan_recv(stdin_recv_slot, Rights::ACCESS)?;
    let output_send = crate::cap::verbs::take_chan_send(output_send_slot, Rights::ACCESS)?;
    let mapped = crate::cap::verbs::map_view(view)?;
    if mapped.len() < PROCESS_STDIO_BYTES
        || !(mapped.cast::<u8>().as_ptr() as usize).is_multiple_of(64)
    {
        return None;
    }
    // SAFETY: `map_view` installed this task's writable, page-aligned sibling
    // view. The mapping remains installed for the task's lifetime and the
    // spent runtime attachment claim makes this the child's sole safe local
    // attachment.
    let base = mapped.cast::<u8>().as_ptr();
    // SAFETY: this task's attachment bit was atomically spent before any
    // handle was read, so safe code cannot construct a second local child
    // producer/consumer pair. The mapped view remains installed for the
    // task's lifetime, and the two held endpoint witnesses are affine.
    unsafe {
        bind_child(
            base,
            PROCESS_STDIO_BYTES,
            crate::entry::transport(),
            ProcessStdioSend::Held(output_send),
            ProcessStdioRecv::Held(stdin_recv),
        )
        .ok()
    }
}

/// A stream host's delegated parent end: the stdin producer, the output
/// consumer, and a send-only peer (`res/SHELL_TERMINAL.md` §6).
pub type MappedProcessStdioHost =
    ProcessStdioParent<'static, 'static, crate::transport::TrapTransport>;

/// What a stream host holds at startup: the mapped stream page, not yet a
/// parent end, and the receive of its **host channel**.
///
/// The launcher formatted the page before either end was founded and keeps
/// every lifecycle pivot: the child's task handle, the channel roots, and the
/// backing the view is carved from. Over the host channel it delivers the
/// stdin-wake send, which [`HostStdioView::bind`] turns the page into a
/// parent end with, and then a delegated handle for each foreground task the
/// host may end (`res/SHELL_TERMINAL.md` §9).
pub struct HostStdio {
    pub view: HostStdioView,
    /// The host channel's receive: build a [`crate::chan::CapRx`] over its
    /// slot. Affine, so this task holds exactly one witness to it.
    pub channel: crate::cap::HeldChanRecvToken,
}

/// The mapped stream page of a stream host, awaiting its stdin-wake send.
pub struct HostStdioView {
    base: *mut u8,
}

impl HostStdioView {
    /// Bind the parent end with the stdin-wake send the launcher delivered.
    ///
    /// The host has no output receive: the child's output wake is a send
    /// into this task's own executor doorbell, so the host drains on every
    /// executor wake and arms with [`ProcessStdioParent::arm_shared_doorbell`];
    /// `park_after_drain*` answer [`ProcessStdioWaitError::SharedDoorbell`].
    /// Binding validates both ring headers and never formats: re-running
    /// [`init_stdio`] here would zero records the child may already have
    /// published.
    /// Binding refuses — with the ring's own reason, and the view spent —
    /// only when the page is not a formatted stream.
    pub fn bind(self, stdin_send: ChanSendToken) -> Result<MappedProcessStdioHost, RingError> {
        // SAFETY: the host attachment bit was atomically spent before the
        // view was taken, and consuming `self` spends this one binding, so
        // safe code cannot construct a second local parent pair. The mapped
        // view remains installed for the task's lifetime.
        unsafe {
            bind_parent(
                self.base,
                kore_abi::process::PROCESS_STDIO_BYTES,
                crate::entry::transport(),
                ProcessStdioSend::Owned(stdin_send),
                None,
            )
        }
    }
}

/// Why a stream host's delegated capabilities did not map. Each variant
/// names the one check that refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostStdioError {
    /// Neither role was staged: an ordinary launch of the same image.
    Absent,
    /// Only one of the two roles was staged.
    Partial,
    /// This task already spent its one stream-host attachment.
    AlreadyAttached,
    /// The view role does not hold a delivered view.
    ViewMissing,
    /// The view lacks `ACCESS | MODIFY`; it carries these rights.
    ViewRights(Rights),
    /// The channel role does not hold a receive with `ACCESS`.
    ChannelMissing,
    /// The kernel refused to map the view.
    MapRefused,
    /// The mapping is shorter than a stream page, or not cache-line aligned.
    Geometry { len: usize, aligned: bool },
}

/// Map the stream page and host channel a stream host's launcher delegated.
#[cfg(target_os = "none")]
pub fn map_stdio_host(rt: &Rt) -> Result<HostStdio, HostStdioError> {
    use kore_abi::process::{
        PROCESS_STDIO_HOST_CHANNEL_INDEX, PROCESS_STDIO_HOST_INDEX, PROCESS_STDIO_HOST_ROLE,
    };
    let bootstrap = rt.bootstrap();
    let (view_slot, channel_slot) = match (
        bootstrap.handle(PROCESS_STDIO_HOST_ROLE, PROCESS_STDIO_HOST_INDEX),
        bootstrap.handle(PROCESS_STDIO_HOST_ROLE, PROCESS_STDIO_HOST_CHANNEL_INDEX),
    ) {
        (Some(view), Some(channel)) => (view, channel),
        // Absent roles are the ordinary launch, and must not spend the claim.
        (None, None) => return Err(HostStdioError::Absent),
        _ => return Err(HostStdioError::Partial),
    };
    if !rt.claim_process_stdio_host_attachment() {
        return Err(HostStdioError::AlreadyAttached);
    }
    let view = crate::cap::verbs::take_view(view_slot).ok_or(HostStdioError::ViewMissing)?;
    // As for the child: both ring halves write their local cursors.
    if !process_stdio_view_is_writable(view.rights()) {
        return Err(HostStdioError::ViewRights(view.rights()));
    }
    let channel = crate::cap::verbs::take_chan_recv(channel_slot, Rights::ACCESS)
        .ok_or(HostStdioError::ChannelMissing)?;
    let mapped = crate::cap::verbs::map_view(view).ok_or(HostStdioError::MapRefused)?;
    let aligned = (mapped.cast::<u8>().as_ptr() as usize).is_multiple_of(64);
    if mapped.len() < PROCESS_STDIO_BYTES || !aligned {
        return Err(HostStdioError::Geometry {
            len: mapped.len(),
            aligned,
        });
    }
    Ok(HostStdio {
        view: HostStdioView {
            base: mapped.cast::<u8>().as_ptr(),
        },
        channel,
    })
}

/// Host builds have no mapped capability view.
#[cfg(not(target_os = "none"))]
pub fn map(_rt: &Rt) -> Option<&'static mut ProcessIo> {
    None
}

/// Host builds have no mapped capability view.
#[cfg(not(target_os = "none"))]
pub fn map_stdio(_rt: &Rt) -> Option<MappedProcessStdioChild> {
    None
}

/// Host builds have no mapped capability view.
#[cfg(not(target_os = "none"))]
pub fn map_stdio_host(_rt: &Rt) -> Result<HostStdio, HostStdioError> {
    Err(HostStdioError::Absent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use core::convert::Infallible;

    use kore_abi::process::{
        PROCESS_STDIO_LAYOUT_BYTES, ProcessInputKind, ProcessOutputKind, ProcessStreamError,
    };
    use kore_abi::session::{Committed, Status};

    use crate::session::SessionBuilder;

    struct OkTransport {
        commits: Cell<u32>,
    }

    impl OkTransport {
        const fn new() -> Self {
            Self {
                commits: Cell::new(0),
            }
        }
    }

    impl Transport for OkTransport {
        type Error = Infallible;

        async fn commit(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert_ne!(session.staged(), 0);
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }

        fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert_ne!(session.staged(), 0);
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }
    }

    struct RefusedTransport;

    impl Transport for RefusedTransport {
        type Error = Infallible;

        async fn commit(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            Ok(Committed::from_raw(Status::Revoked.encode(0)))
        }

        fn commit_now(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            Ok(Committed::from_raw(Status::Revoked.encode(0)))
        }
    }

    fn drive<F: core::future::Future>(future: F) -> F::Output {
        crate::block::try_block_on(future, 8)
            .expect("stream operation did not resolve without an external wake")
    }

    #[repr(C, align(64))]
    struct Page([u8; kore_abi::process::PROCESS_STDIO_BYTES]);

    impl Page {
        fn zeroed() -> Self {
            Self([0; kore_abi::process::PROCESS_STDIO_BYTES])
        }
    }

    #[test]
    fn cancelled_stream_wait_retracts_its_waiting_publication() {
        let word = AtomicU32::new(0);
        {
            let _waiting = ProcessStdioWaiting::publish(&word);
            assert_eq!(word.load(Ordering::SeqCst), 1);
            // Scope exit models cancellation while the alternate executor
            // future is still pending.
        }
        assert_eq!(word.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn process_stdio_mapping_requires_a_writable_view() {
        assert!(!process_stdio_view_is_writable(Rights::empty()));
        assert!(!process_stdio_view_is_writable(Rights::ACCESS));
        assert!(!process_stdio_view_is_writable(Rights::MODIFY));
        assert!(process_stdio_view_is_writable(
            Rights::ACCESS | Rights::MODIFY
        ));
        assert!(process_stdio_view_is_writable(
            Rights::ACCESS | Rights::MODIFY | Rights::REVOKE
        ));
    }

    #[test]
    fn stream_geometry_matches_the_frozen_one_page_abi() {
        assert_eq!(
            spsc::region_bytes::<ProcessInputChunk>(PROCESS_STDIO_RING_CAPACITY),
            PROCESS_STDIO_RING_BYTES
        );
        assert_eq!(
            spsc::region_bytes::<ProcessOutputEvent>(PROCESS_STDIO_RING_CAPACITY),
            PROCESS_STDIO_RING_BYTES
        );
        assert_eq!(PROCESS_OUTPUT_RING_OFFSET, PROCESS_STDIO_RING_BYTES);
        assert_eq!(PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET, 2432);
        assert_eq!(PROCESS_STDIO_CHILD_WAITING_OFFSET, 2436);
        assert_eq!(PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET, 2496);
        assert_eq!(PROCESS_STDIO_PARENT_WAITING_OFFSET, 2500);
        assert_eq!(PROCESS_STDIO_LAYOUT_BYTES, 2560);
        assert!(PROCESS_STDIO_LAYOUT_BYTES <= kore_abi::process::PROCESS_STDIO_BYTES);

        let mut too_small = [0u8; kore_abi::process::PROCESS_STDIO_BYTES - 1];
        assert!(matches!(
            init_stdio(&mut too_small),
            Err(RingError::TooSmall)
        ));

        #[repr(C, align(64))]
        struct Oversized([u8; kore_abi::process::PROCESS_STDIO_BYTES + 1]);
        let mut oversized = Oversized([0; kore_abi::process::PROCESS_STDIO_BYTES + 1]);
        assert!(matches!(
            init_stdio(&mut oversized.0[1..]),
            Err(RingError::Misaligned)
        ));
    }

    #[test]
    fn shared_doorbell_covers_both_sides_of_registration_and_coalesces() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();
        let before_publish = parent.begin_drain();
        drive(child.try_send_output(ProcessOutputEvent::stdout(b"a").unwrap())).unwrap();
        assert!(!parent.arm_shared_doorbell(before_publish));
        drive(parent.try_recv_output()).unwrap();
        assert!(parent.arm_shared_doorbell(parent.begin_drain()));
        assert_eq!(
            drive(child.try_send_output(ProcessOutputEvent::stdout(b"b").unwrap())).unwrap(),
            ProcessStdioWake::Delivered
        );
        assert_eq!(
            drive(child.try_send_output(ProcessOutputEvent::stdout(b"c").unwrap())).unwrap(),
            ProcessStdioWake::PeerNotWaiting
        );
        let drain = parent.begin_drain();
        assert!(!parent.arm_shared_doorbell(drain));
        assert_eq!(
            drive(parent.try_recv_output()).unwrap().0.bytes().unwrap(),
            b"b"
        );
        assert_eq!(
            drive(parent.try_recv_output()).unwrap().0.bytes().unwrap(),
            b"c"
        );
        assert!(parent.arm_shared_doorbell(parent.begin_drain()));
    }

    /// A stream host binds the parent end with no dedicated receive: it parks
    /// only through the shared-doorbell fence, never through a notice post it
    /// does not hold, and closes without disarming a landing it never armed.
    #[test]
    fn a_capacity_wait_ignores_unread_input() {
        use ProcessStdioWaitFor::{Capacity, Incoming, IncomingOrCapacity};
        // Output full, input waiting: only a capacity wait still parks.
        assert!(Incoming.satisfied(true, false));
        assert!(IncomingOrCapacity.satisfied(true, false));
        assert!(!Capacity.satisfied(true, false));
        assert!(Capacity.satisfied(false, true));
        assert!(!Incoming.satisfied(false, true));
    }

    #[test]
    fn a_stream_host_parks_only_through_its_shared_doorbell() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let prepared = init_stdio(&mut page.0).unwrap();
        let (base, len) = prepared.into_raw();
        let (mut host, mut child) = unsafe {
            (
                bind_parent(
                    base,
                    len,
                    &transport,
                    ProcessStdioSend::Test(SlotRef::from_wire(41)),
                    None,
                )
                .unwrap(),
                bind_child(
                    base,
                    len,
                    &transport,
                    ProcessStdioSend::Test(SlotRef::from_wire(43)),
                    ProcessStdioRecv::Test(SlotRef::from_wire(44)),
                )
                .unwrap(),
            )
        };
        assert!(host.arm_shared_doorbell(host.begin_drain()));
        assert_eq!(
            drive(child.try_send_output(ProcessOutputEvent::input_ready())).unwrap(),
            ProcessStdioWake::Delivered
        );
        assert_eq!(
            drive(host.try_recv_output()).unwrap().0.kind(),
            Ok(ProcessOutputKind::InputReady)
        );
        let drain = host.begin_drain();
        assert_eq!(
            drive(host.park_after_drain(drain, ProcessStdioWaitFor::Incoming)),
            Err(ProcessStdioWaitError::SharedDoorbell)
        );
        drive(host.try_send_stdin(ProcessInputChunk::data(b"ls\n").unwrap())).unwrap();
        assert_eq!(
            drive(child.try_recv_stdin()).unwrap().0.bytes(),
            Ok(b"ls\n".as_slice())
        );
        host.close_after_wake_roots_revoked();
    }

    #[test]
    fn affine_stream_methods_publish_both_data_and_capacity_edges() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        page.0[PROCESS_STDIO_LAYOUT_BYTES..].fill(0xa5);
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();

        for event in [
            ProcessOutputEvent::stdout(b"> 42\n").unwrap(),
            ProcessOutputEvent::input_ready(),
        ] {
            assert_eq!(
                drive(child.try_send_output(event)).unwrap(),
                ProcessStdioWake::PeerNotWaiting
            );
        }
        assert_eq!(child.peer.published.load(Ordering::SeqCst), 2);

        let (stdout, _) = drive(parent.try_recv_output()).unwrap();
        assert_eq!(stdout.kind(), Ok(ProcessOutputKind::Stdout));
        assert_eq!(stdout.bytes(), Ok(b"> 42\n".as_slice()));
        assert_eq!(
            drive(parent.try_recv_output()).unwrap().0.kind(),
            Ok(ProcessOutputKind::InputReady)
        );
        assert!(drive(parent.try_recv_output()).is_none());

        // `Closed` is the whole-stream terminal, so answer the line-level
        // InputReady before the child publishes it. This is the same ordering
        // enforced by the shell relay and required by the frozen ABI law.
        assert_eq!(
            drive(parent.try_send_stdin(ProcessInputChunk::data(b"print(6 * 7)\n").unwrap()))
                .unwrap(),
            ProcessStdioWake::PeerNotWaiting
        );
        let (input, capacity_wake) = drive(child.try_recv_stdin()).unwrap();
        assert_eq!(capacity_wake, ProcessStdioWake::PeerNotWaiting);
        assert_eq!(input.kind(), Ok(ProcessInputKind::Data));
        assert_eq!(input.bytes(), Ok(b"print(6 * 7)\n".as_slice()));
        assert_eq!(
            drive(child.try_send_output(ProcessOutputEvent::closed())).unwrap(),
            ProcessStdioWake::PeerNotWaiting
        );
        assert_eq!(
            drive(parent.try_recv_output()).unwrap().0.kind(),
            Ok(ProcessOutputKind::Closed)
        );
        assert!(drive(parent.try_recv_output()).is_none());
        assert_eq!(child.peer.published.load(Ordering::SeqCst), 4);
        assert_eq!(parent.peer.published.load(Ordering::SeqCst), 4);
        assert_eq!(transport.commits.get(), 0, "an awake peer costs no notice");
        drop(parent);
        drop(child);
        assert!(page.0[PROCESS_STDIO_LAYOUT_BYTES..].iter().all(|b| *b == 0));
    }

    #[test]
    fn producer_kicks_only_an_armed_peer_and_full_does_not_publish() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();

        for byte in 0..PROCESS_STDIO_RING_CAPACITY {
            assert_eq!(
                drive(parent.try_send_stdin(ProcessInputChunk::data(&[byte as u8]).unwrap()))
                    .unwrap(),
                ProcessStdioWake::PeerNotWaiting
            );
        }
        let published = parent.peer.published.load(Ordering::SeqCst);
        assert!(drive(parent.try_send_stdin(ProcessInputChunk::data(b"x").unwrap())).is_err());
        assert_eq!(parent.peer.published.load(Ordering::SeqCst), published);
        assert_eq!(transport.commits.get(), 0);

        // The parent's waiting bit is written only by the parent wait path.
        // Simulate that final armed interval and prove one child mutation
        // consumes it and spends exactly one notice commit.
        parent.peer.waiting.store(1, Ordering::SeqCst);
        assert_eq!(
            drive(child.try_send_output(ProcessOutputEvent::input_ready())).unwrap(),
            ProcessStdioWake::Delivered
        );
        assert_eq!(parent.peer.waiting.load(Ordering::SeqCst), 0);
        assert_eq!(transport.commits.get(), 1);
    }

    #[test]
    fn stale_drain_rechecks_state_before_trying_to_park() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();

        let drain = parent.begin_drain();
        drive(child.try_send_output(ProcessOutputEvent::input_ready())).unwrap();
        assert_eq!(
            drive(parent.park_after_drain_for(drain, ProcessStdioWaitFor::Incoming, 100_000_000,)),
            Ok(ProcessStdioPark::StateChanged)
        );
        assert_eq!(
            transport.commits.get(),
            0,
            "an epoch/ready recheck must precede notice-post construction"
        );
    }

    /// An interrupt is not input: it crosses with no InputReady outstanding,
    /// and the launcher learns the child read it from the ring alone.
    #[test]
    fn an_unsolicited_interrupt_crosses_and_its_read_is_observable() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();

        assert_eq!(parent.stdin_unconsumed(), 0);
        assert_eq!(
            drive(parent.try_send_stdin(ProcessInputChunk::interrupt())).unwrap(),
            ProcessStdioWake::PeerNotWaiting
        );
        assert_eq!(parent.stdin_unconsumed(), 1);
        assert_eq!(child.stdin_outstanding(), 1);
        let (record, _) = drive(child.try_recv_stdin()).unwrap();
        assert_eq!(record.kind(), Ok(ProcessInputKind::Interrupt));
        assert_eq!(parent.stdin_unconsumed(), 0);
        // The child may still close its stream: nothing was left unanswered.
        assert!(drive(child.try_send_output(ProcessOutputEvent::closed())).is_ok());
    }

    #[test]
    fn unreachable_kick_never_rolls_back_published_state() {
        let transport = RefusedTransport;
        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();

        child.peer.waiting.store(1, Ordering::SeqCst);
        assert_eq!(
            drive(parent.try_send_stdin(ProcessInputChunk::data(b"x").unwrap())).unwrap(),
            ProcessStdioWake::Unreached(SendError::Refused(Status::Revoked))
        );
        assert_eq!(parent.peer.published.load(Ordering::SeqCst), 1);
        assert_eq!(child.stdin_outstanding(), 1);
        assert_eq!(
            drive(child.try_recv_stdin()).unwrap().0.bytes(),
            Ok(b"x".as_slice())
        );
    }

    #[test]
    fn attachments_and_record_decoding_fail_closed() {
        let transport = OkTransport::new();
        let mut page = Page::zeroed();
        let result = unsafe {
            bind_parent(
                page.0.as_mut_ptr(),
                page.0.len(),
                &transport,
                ProcessStdioSend::Test(SlotRef::from_wire(41)),
                Some(ProcessStdioRecv::Test(SlotRef::from_wire(42))),
            )
        };
        assert!(matches!(result, Err(RingError::BadHeader)));

        let prepared = init_stdio(&mut page.0).unwrap();
        let (base, len) = prepared.into_raw();
        // Attaching either side validates both ring headers and their fixed
        // geometry at the ABI-assigned semantic offsets.
        let attached = unsafe {
            bind_parent(
                base,
                len,
                &transport,
                ProcessStdioSend::Test(SlotRef::from_wire(41)),
                Some(ProcessStdioRecv::Test(SlotRef::from_wire(42))),
            )
        };
        assert!(attached.is_ok());
        drop(attached);

        let mut page = Page::zeroed();
        let (mut parent, mut child) = split_stdio(&mut page.0, &transport, &transport).unwrap();
        let mut malformed = [0u8; core::mem::size_of::<ProcessOutputEvent>()];
        malformed[0..2].copy_from_slice(&u16::MAX.to_le_bytes());
        let malformed = ProcessOutputEvent::read_from_bytes(&malformed).unwrap();
        drive(child.try_send_output(malformed)).unwrap();
        let received = drive(parent.try_recv_output()).unwrap().0;
        assert!(!received.validate());
        assert_eq!(received.kind(), Err(ProcessStreamError::Invalid));
        assert_eq!(received.bytes(), Err(ProcessStreamError::Invalid));
    }
}
