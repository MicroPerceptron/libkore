//! Storage rings — the process side of `res/STORAGE_TRANSPORT.md` §4.
//!
//! A session attaches one **ring**: a window it funds from its own mm face,
//! formats and lends as a view ([`RingFs::stor_attach`]). From then on it
//! submits by writing its own window — io_uring records naming a grant, a
//! target and a timeline — and waits on the timeline's fence, which the row
//! stores from its completion path. The pair carries only control: the
//! attach, grants ([`RingFs::stor_grant_pack`]), targets ([`RingFs::stor_target`]),
//! and the one notice a parked row is owed ([`RingFs::stor_publish`]).
//!
//! ```text
//! client window (funded here, mapped by both)
//!   Demand SQ   ← reads a thread waits on
//!   Stream SQ   ← read-ahead, bulk
//!   CQ          → one completion per submission (drained by `StorRing`)
//!   fence page  → per-timeline {done, fault}, the word a wait sleeps on
//! ```
//!
//! **Flow control is the client's to keep.** The row pops a submission only
//! while the completion channel has room for its completion, so a ring whose
//! completions are never drained stops. [`StorRing`] drains them on every
//! poll and wait and keeps the first failure, so a caller that works in
//! tickets never has to.

use core::time::Duration;

use kore_abi::stor::FenceRecord;
use kore_ring::stor::{FenceCell, StorClient, SubmitError};
use kore_rt::svc::{ReturnCapability, ReturnLeaseSource};

use kore_rt::transport as rt_transport;

use crate::client::{LentVersion, RingFs, RingFsError};
use crate::{Fs, FsError, FsRequest, FsResponse, ProviderError};

pub use kore_abi::stor::{Class, GrantId, StorStatus};
/// What every ring call answers when it does not succeed.
pub use crate::Refusal;
pub use kore_ring::stor::{Covering, Op, Request, Ticket, WINDOW_BYTES, covering};

/// Pause between kicks once a wait has stopped watching its fence: long
/// enough that a waiting client cannot keep the row's request ring
/// permanently non-empty, short against any command's service time.
const POLL_PAUSE_NS: u64 = 50_000;

/// How long [`RingFs::stor_detach`] waits out commands still in flight.
pub const DETACH_PATIENCE_NS: u64 = 2_000_000_000;

/// A filesystem launch's own storage ring (`res/STORAGE_TRANSPORT.md` §9,
/// S5d): the window and read target exec funded and moved in
/// ([`role::STOR_RING`], [`role::STOR_TARGET`]).
///
/// The ring is attached for one [`read_file`](Fs::read_file) or large
/// [`read_at`](Fs::read_at) and detached before it returns, never left attached: a process can exit at any time,
/// and the kernel tears down a lent window when its lessor dies, under the
/// row's feet (§12). The target is registered once and kept; only its device
/// half is lent to each attachment. A setup that fails, or a read or detach
/// that may leave a command landing late, retires the ring for good, and
/// reads take the pair's window.
///
/// [`role::STOR_RING`]: kore_abi::task::role::STOR_RING
/// [`role::STOR_TARGET`]: kore_abi::task::role::STOR_TARGET
pub(crate) enum FacadeRing {
    /// The founding delivered no ring.
    Absent,
    /// Delivered, nothing set up yet.
    Delivered {
        window: kore_rt::cap::LeaseToken,
        target: kore_rt::cap::LeaseToken,
    },
    /// Window mapped and target registered; no ring attached.
    Idle(IdleRing),
    /// Unusable. Its custody is kept, never retired.
    Broken,
}

/// A facade ring between reads. Fields drop in declaration order: the
/// target's CPU view before the fabric custody that parents it.
pub(crate) struct IdleRing {
    /// This task's mapping of the ring window, `WINDOW_BYTES` long.
    window_bytes: core::ptr::NonNull<u8>,
    /// The target's CPU view, where the ring's reads land.
    cpu: core::ptr::NonNull<u8>,
    target_len: u64,
    _cpu_view: kore_rt::cap::MappedLeaseView,
    _cpu_scope: kore_rt::cap::FabricRouteToken,
    /// Parents each attachment's device child.
    route: kore_rt::cap::FabricRouteToken,
    _registration: kore_rt::cap::FabricRegistrationToken,
    _target_lease: kore_rt::cap::LeaseToken,
    window: kore_rt::cap::LeaseToken,
}

/// A facade ring wait's patience: past the engine's 30 s deadline and
/// abort, so a wedged read reports rather than hangs.
const FACADE_PATIENCE: Duration = Duration::from_secs(40);

/// How long a facade ring wait sleeps on its fence before also kicking.
const FACADE_WATCH_NS: u64 = 100_000;

/// Reads smaller than this go through the window: attaching and detaching
/// a ring costs more round trips than the window needs for them.
pub(crate) const FACADE_MIN_BYTES: u64 = 256 * 1024;

impl FacadeRing {
    pub(crate) fn delivered(
        window: Option<kore_rt::cap::LeaseToken>,
        target: Option<kore_rt::cap::LeaseToken>,
    ) -> Self {
        match (window, target) {
            (Some(window), Some(target)) => Self::Delivered { window, target },
            _ => Self::Absent,
        }
    }

    /// Map the window and register the target, once. `None` leaves nothing
    /// to use.
    fn set_up(
        window: kore_rt::cap::LeaseToken,
        target: kore_rt::cap::LeaseToken,
    ) -> Result<IdleRing, &'static str> {
        use kore_abi::session::{FabricAccess, FabricOp, FabricRouteScopeReq, FabricSync};
        use kore_rt::cap::verbs;

        let mapped = verbs::map_held_rw(window.slot()).ok_or("window map")?;
        if mapped.len() < WINDOW_BYTES {
            return Err("window size");
        }
        let (_, target_len) = verbs::lease_geometry(target.slot()).ok_or("target geometry")?;
        // Registered before any alias: one route parents this task's view of
        // the target and each attachment's device child.
        let registration =
            verbs::register_fabric_lease(target.slot()).map_err(|_| "target registration")?;
        let route = registration
            .open_self_route(target_len, FabricOp::IommuMap, FabricAccess::all(), FabricSync::Fenced)
            .map_err(|_| "target route")?;
        let cpu_scope = route
            .scope(&FabricRouteScopeReq::new(0, target_len, FabricAccess::all()))
            .map_err(|_| "target cpu scope")?;
        let cpu_view = verbs::map_owned_lease_rw(cpu_scope.slot()).ok_or("target map")?;
        Ok(IdleRing {
            window_bytes: mapped.cast::<u8>(),
            cpu: core::ptr::NonNull::new(cpu_view.as_mut_ptr()).ok_or("target map")?,
            target_len,
            _cpu_view: cpu_view,
            _cpu_scope: cpu_scope,
            route,
            _registration: registration,
            _target_lease: target,
            window,
        })
    }
}

impl Fs<'_, '_> {
    /// Stream bytes `[start, start + limit)` of `file` — clipped at its end —
    /// to `consume` through this launch's ring: attach it, pin the file's
    /// current version, read the block-aligned middle by DMA into the target a
    /// piece at a time, the unaligned head and tail (the partial last block
    /// always among them) through the pair's window, and detach. Answers how
    /// many bytes were consumed; `Ok(None)` when the ring is not used and
    /// nothing was consumed — the caller reads through the window instead.
    pub(crate) async fn read_range_by_ring<F>(
        &mut self,
        file: &crate::View,
        start: u64,
        limit: u64,
        consume: &mut F,
    ) -> Result<Option<u64>, Refusal>
    where
        F: FnMut(&[u8]) -> Result<(), Refusal>,
    {
        use kore_abi::session::{FabricAccess, FabricRouteScopeReq};

        if limit.min(file.len().saturating_sub(start)) < FACADE_MIN_BYTES {
            self.ring_note = "small read";
            return Ok(None);
        }
        if matches!(self.ring, FacadeRing::Delivered { .. }) {
            let FacadeRing::Delivered { window, target } =
                core::mem::replace(&mut self.ring, FacadeRing::Broken)
            else {
                unreachable!("matched above");
            };
            match FacadeRing::set_up(window, target) {
                Ok(idle) => self.ring = FacadeRing::Idle(idle),
                Err(step) => self.ring_note = step,
            }
        }
        let FacadeRing::Idle(idle) = &self.ring else {
            if matches!(self.ring, FacadeRing::Absent) {
                self.ring_note = "no ring delivered";
            }
            return Ok(None);
        };
        // SAFETY: the window's mapping lives in `idle` for the task's life,
        // and no other ring borrows it: the last attachment was detached
        // (and its `StorRing` consumed) before this one.
        let bytes: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(idle.window_bytes.as_ptr(), WINDOW_BYTES) };
        let (cpu, target_len) = (idle.cpu, idle.target_len);
        let device = idle
            .route
            .scope(&FabricRouteScopeReq::new(0, target_len, FabricAccess::all()));
        let Ok(device) = device else {
            self.ring_note = "target device scope";
            return Ok(None);
        };
        let window = &idle.window;
        // A session holds one ring: an app that attached its own keeps it.
        let mut ring = match self.client.stor_attach(window, bytes).await {
            Ok(ring) => ring,
            Err(refusal) => {
                self.ring_note = refusal_note("attach", &refusal);
                return Ok(None);
            }
        };
        ring.set_patience(Some(FACADE_PATIENCE));
        let target = match self.client.stor_target(device).await {
            Ok(target) => target,
            Err(refusal) => {
                self.ring_note = refusal_note("target", &refusal);
                // `Busy`: the row kept a mapping of the target past the
                // refusal, so the target may never be lent again.
                if matches!(refusal, Refusal::Fs(FsError::Busy)) {
                    self.break_ring();
                }
                if self.client.stor_detach(ring).await.is_err() {
                    self.break_ring();
                }
                return Ok(None);
            }
        };
        let Ok(version) = self.lend_version(file).await else {
            self.ring_note = "lend version";
            if self.client.stor_detach(ring).await.is_err() {
                self.break_ring();
            }
            return Ok(None);
        };
        let end = version.identity().len.min(start.saturating_add(limit));
        let block = u64::from(ring.block_size()).max(1);
        // The middle the ring can carry: whole blocks the grant covers, block
        // aligned at both ends. Nothing when the range holds no such block.
        let middle = match self.client.stor_grant_version(&version).await {
            Ok(grant) => {
                let from = start.next_multiple_of(block);
                let to = (end.min(grant.bytes) / block) * block;
                (from < to).then_some((grant, from, to))
            }
            // Nothing whole to grant: a sub-block file is all tail.
            Err(Refusal::Fs(FsError::Range)) => None,
            Err(refusal) => {
                self.ring_note = refusal_note("grant", &refusal);
                let _ = self.release_version(version).await;
                if self.client.stor_detach(ring).await.is_err() {
                    self.break_ring();
                }
                return Ok(None);
            }
        };
        let (head_end, tail_start) = match middle {
            Some((_, from, to)) => (from, to),
            None => (end, end),
        };
        // Head through the window, then the middle through the ring, then the
        // tail through the window: in order, from the same pinned version.
        let mut result = self.window_range(&version, start, head_end.max(start).min(end), consume).await;
        let mut landed_late = false;
        if let Some((grant, from, to)) = middle {
            let mut offset = from;
            while result.is_ok() && offset < to {
                let piece = (to - offset).min(target.bytes);
                let read = ReadSpec {
                    offset,
                    len: piece,
                    target,
                    target_offset: 0,
                };
                let landed = match self
                    .client
                    .stor_read_batch(&mut ring, Class::Demand, grant, 1, [read], FACADE_WATCH_NS)
                    .await
                {
                    Ok(Some(ticket)) => self
                        .client
                        .stor_wait(&mut ring, ticket, FACADE_WATCH_NS)
                        .await
                        .map(|_| ()),
                    Ok(None) => Ok(()),
                    Err(refusal) => Err(refusal),
                };
                if let Err(refusal) = landed {
                    // A read that broke off may still land in the target.
                    landed_late = true;
                    result = Err(refusal);
                    break;
                }
                // SAFETY: the fence covered every command writing these
                // bytes, and `piece <= target.bytes`, inside the mapped view.
                let bytes = unsafe { core::slice::from_raw_parts(cpu.as_ptr(), piece as usize) };
                result = consume(bytes);
                offset += piece;
            }
        }
        if result.is_ok() {
            result = self.window_range(&version, tail_start.max(start), end, consume).await;
        }
        let _ = self.release_version(version).await;
        // Detached before returning, so no exit can leave the row reading a
        // window whose lessor is gone. A detach that did not settle leaves
        // commands that may land late: the ring is never used again.
        if self.client.stor_detach(ring).await.is_err() || landed_late {
            self.break_ring();
        }
        if result.is_ok() {
            self.ring_files += 1;
            self.ring_note = "streamed";
        }
        result.map(|()| Some(end.saturating_sub(start)))
    }

    /// Bytes `[from, to)` of `version` to `consume` through the pair's window.
    async fn window_range<F>(
        &mut self,
        version: &LentVersion,
        from: u64,
        to: u64,
        consume: &mut F,
    ) -> Result<(), Refusal>
    where
        F: FnMut(&[u8]) -> Result<(), Refusal>,
    {
        let mut offset = from;
        while offset < to {
            let want = (to - offset).min(self.chunk() as u64).min(u64::from(u32::MAX)) as u32;
            match self.client.read_version(version, offset, 0, want).await {
                Ok(0) => return Err(Refusal::Fs(FsError::Range)),
                Ok(count) if count <= want => {
                    consume(&self.window[..count as usize])?;
                    offset += u64::from(count);
                }
                Ok(_) => return Err(Refusal::Pair),
                Err(RingFsError::Refused(error)) => return Err(Refusal::Fs(error)),
                Err(_) => return Err(Refusal::Pair),
            }
        }
        Ok(())
    }

    /// Why the last whole-file or large read did or did not stream
    /// through this launch's ring — a step name, for a witness to print.
    pub fn facade_ring_note(&self) -> &'static str {
        self.ring_note
    }

    /// Retire this launch's facade ring for good, keeping its custody.
    fn break_ring(&mut self) {
        if let FacadeRing::Idle(idle) = core::mem::replace(&mut self.ring, FacadeRing::Broken) {
            core::mem::forget(idle);
        }
    }
}

/// A refusal at a ring step, named for [`Fs::facade_ring_note`].
fn refusal_note(step: &'static str, refusal: &Refusal) -> &'static str {
    match (step, refusal) {
        ("attach", Refusal::Fs(FsError::AlreadyExists)) => "attach: session has a ring",
        ("attach", Refusal::Fs(FsError::OutOfHandles)) => "attach: row out of rings",
        ("attach", Refusal::Fs(_)) => "attach refused",
        ("attach", _) => "attach: pair failed",
        ("target", Refusal::Fs(FsError::Busy)) => "target: mapping outlived refusal",
        ("target", Refusal::Fs(_)) => "target refused",
        ("target", _) => "target: pair failed",
        ("grant", Refusal::Fs(_)) => "grant refused",
        _ => "grant: pair failed",
    }
}

/// A grant installed into this session's ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingGrant {
    /// The id a submission names.
    pub id: GrantId,
    /// Bytes the grant covers, from offset zero.
    pub bytes: u64,
}

/// A target adopted into this session's ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingTarget {
    /// The slot a submission names.
    pub slot: u16,
    /// The target's length.
    pub bytes: u64,
}

/// One read of a batch ([`RingFs::stor_read_batch`]): `len` bytes at `offset`
/// of the grant into `target` at `target_offset`. Any length: the batch
/// splits a read longer than [`MAX_SUBMISSION`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadSpec {
    pub offset: u64,
    pub len: u64,
    pub target: RingTarget,
    pub target_offset: u64,
}

/// Largest single submission a batch builds (block-aligned for every block
/// size up to 1 GiB); the row splits it further at the device's ceiling.
pub const MAX_SUBMISSION: u64 = 1 << 30;

/// A timeline's fence, held apart from the ring: all atomics in the
/// window, so any thread may watch it while the ring itself is driven
/// under someone else's lock.
#[derive(Clone, Copy)]
pub struct FenceWatch {
    cell: &'static FenceCell,
}

impl FenceWatch {
    /// The fence as it stands.
    pub fn load(&self) -> FenceRecord {
        self.cell.load()
    }

    /// Where the ticket numbered `seq` on this timeline stands.
    pub fn state(&self, seq: u64) -> TicketState {
        let fence = self.cell.load();
        if fence.fault != 0 && fence.fault <= seq {
            TicketState::Faulted
        } else if fence.done >= seq {
            TicketState::Done
        } else {
            TicketState::Pending
        }
    }

    /// Sleep this core on the fence's line until the row stores to it or
    /// `cap` passes, when `done` still reads `observed`. The row's store is
    /// the wake — `UMWAIT`/`MONITORX` end on a write to the line from any
    /// agent — so no wake call crosses the pair.
    pub fn sleep_while(&self, observed: u64, cap: Duration) {
        let _ = kore_rt::hint::wait_on(self.cell.done_word(), observed as u32, cap);
    }
}

/// What a full channel waits for: the timeline's next sequence to complete
/// — but only one that was issued. Once `done` has caught up with
/// everything issued there is nothing of this timeline's to wait for, and a
/// wait on `done + 1` would never end (the metal hang: a publish's kick let
/// the row drain the channel before the target was chosen).
const fn room_wait_target(done: u64, issued: u64) -> Option<u64> {
    if done < issued { Some(done + 1) } else { None }
}

/// Where a ticket stands, as its timeline's fence shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TicketState {
    /// Not yet complete.
    Pending,
    /// Complete, and nothing at or before it on its timeline failed.
    Done,
    /// A submission at or before it on its timeline failed: the bytes it
    /// named are unusable.
    Faulted,
}

/// This session's attached ring: the client halves over its own mapping of
/// the window, and the geometry every submission obeys.
pub struct StorRing {
    client: StorClient<'static>,
    block_size: u32,
    max_transfer: u32,
    /// The first failed completion drained, with its token.
    failure: Option<(u64, StorStatus)>,
    /// Completions drained.
    completed: u64,
    /// How long one wait may last before it gives up and reports.
    patience_ns: Option<u64>,
    /// The last wait that gave up, and what it saw.
    stall: Option<StallReport>,
}

/// The transport engine's counters, every ring together
/// ([`RingFs::stor_inspect`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StorCounters {
    pub submissions: u64,
    pub completions: u64,
    /// Device commands in flight when read.
    pub in_flight: u32,
    /// Ops reported past their deadline.
    pub expired: u32,
    /// Device completions naming no live command.
    pub stale: u32,
    /// Completions a timeline refused — each stops a fence for good.
    pub timeline_refusals: u32,
    /// Device commands the queue set refused to stage.
    pub stage_refusals: u32,
    /// The last stage refusal (`BlockError` discriminant plus one; 0 none).
    pub last_stage_error: u8,
    /// Abort commands sent for expired device commands.
    pub aborts: u32,
    /// Aborts the controller refused (15 reads "15 or more").
    pub aborts_refused: u8,
    /// The controller arbitrates by weighted round robin (`CAP.AMS` offered
    /// it); otherwise plain round robin.
    pub weighted: bool,
    /// Rings attached, the row's own lanes included (7 reads "7 or more").
    pub rings: u8,
}

/// What a wait saw when its patience ran out ([`StorRing::set_patience`]):
/// enough to tell a submission never popped, from one the device still
/// holds, from one that completed without its fence moving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StallReport {
    /// The ticket waited on.
    pub ticket: Ticket,
    /// Its timeline's fence.
    pub fence: FenceRecord,
    /// The row's account of this ring: status, submissions it holds,
    /// completions it posted.
    pub row: Option<kore_abi::stor::CqState>,
    /// Room left in each class's channel (Demand, Stream, Commit): below
    /// the depth means submissions the row has not popped.
    pub room: [u32; 3],
    /// Completions this client drained.
    pub drained: u64,
    /// The engine's counters, when the row answered.
    pub engine: Option<StorCounters>,
    /// How long the wait lasted.
    pub waited_ns: u64,
}

impl core::fmt::Display for StallReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ticket {}#{} after {} ms: fence done {} fault {}; room d/s/c {}/{}/{}; drained {}",
            self.ticket.timeline,
            self.ticket.seq,
            self.waited_ns / 1_000_000,
            self.fence.done,
            self.fence.fault,
            self.room[0],
            self.room[1],
            self.room[2],
            self.drained,
        )?;
        match self.row {
            Some(row) => write!(
                f,
                "; row status {} holds {} posted {}",
                row.status, row.in_flight, row.completed
            )?,
            None => f.write_str("; row state unreadable")?,
        }
        match self.engine {
            Some(e) => write!(
                f,
                "; engine popped {} posted {} in flight {} expired {} stale {} refused {} stage-refused {} (last {})",
                e.submissions,
                e.completions,
                e.in_flight,
                e.expired,
                e.stale,
                e.timeline_refusals,
                e.stage_refusals,
                e.last_stage_error
            ),
            None => f.write_str("; engine unreadable"),
        }
    }
}

impl StorRing {
    /// Bound every wait on this ring to `patience`: past it the wait gives
    /// up with `Busy` and records a [`StallReport`] ([`take_stall`]).
    ///
    /// [`take_stall`]: Self::take_stall
    pub fn set_patience(&mut self, patience: Option<core::time::Duration>) {
        self.patience_ns = patience.map(|patience| patience.as_nanos() as u64);
    }

    /// The last wait that gave up, cleared by the read.
    pub fn take_stall(&mut self) -> Option<StallReport> {
        self.stall.take()
    }

    /// Logical block size: every offset and length is a multiple.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The row's single-command ceiling. A longer submission is split by
    /// the row; this is testimony, not a limit on `len`.
    pub fn max_transfer(&self) -> u32 {
        self.max_transfer
    }

    /// Queue `request` on `class` (visible to the row only after
    /// [`RingFs::stor_publish`]). The ticket, when the request names a
    /// timeline, is what to wait for.
    pub fn submit(&mut self, class: Class, request: &Request) -> Result<Option<Ticket>, SubmitError> {
        self.client.submit(class, request)
    }

    /// Queue a read of `len` bytes at `offset` of `grant` into `target` at
    /// `target_offset`, on `timeline`.
    #[allow(clippy::too_many_arguments)]
    pub fn read(
        &mut self,
        class: Class,
        grant: RingGrant,
        offset: u64,
        len: u32,
        target: RingTarget,
        target_offset: u64,
        timeline: u16,
    ) -> Result<Ticket, SubmitError> {
        let request = Request {
            op: Op::Read,
            grant: grant.id,
            offset,
            len,
            target: target.slot,
            target_offset,
            user_data: offset,
            timeline,
            deadline_ns: 0,
        };
        self.client
            .submit(class, &request)?
            .ok_or(SubmitError::Timeline)
    }

    /// Submissions `class`'s channel can still take.
    pub fn room(&self, class: Class) -> u32 {
        self.client.room(class)
    }

    /// Timeline `t`'s fence as it stands.
    pub fn fence(&self, t: u16) -> Option<FenceRecord> {
        self.client.fence(t)
    }

    /// A handle on timeline `t`'s fence that outlives this borrow.
    pub fn watch(&self, t: u16) -> Option<FenceWatch> {
        self.client.fence_cell(t).map(|cell| FenceWatch { cell })
    }

    /// Drain every posted completion, keeping the first failure. Returns
    /// how many were drained.
    pub fn drain(&mut self) -> u32 {
        let mut drained = 0;
        while let Some(completion) = self.client.reap() {
            drained += 1;
            if let Err(status) = completion.result {
                self.failure.get_or_insert((completion.user_data, status));
            }
        }
        self.completed += u64::from(drained);
        drained
    }

    /// The row's own account of the ring — live or departed, submissions it
    /// holds, completions it posted — read without a call.
    pub fn row_state(&self) -> Option<kore_abi::stor::CqState> {
        self.client.state()
    }

    /// Completions drained since attach.
    pub fn completed(&self) -> u64 {
        self.completed
    }

    /// The first failed completion drained so far — its token (a read's
    /// grant offset) and status — cleared by the read.
    pub fn take_failure(&mut self) -> Option<(u64, StorStatus)> {
        self.failure.take()
    }

    /// Where `ticket` stands. Drains completions first, so waiting on
    /// tickets alone keeps the ring flowing.
    pub fn state(&mut self, ticket: Ticket) -> TicketState {
        self.drain();
        match self.client.poll_ticket(ticket) {
            Ok(true) => TicketState::Done,
            Ok(false) => TicketState::Pending,
            Err(_) => TicketState::Faulted,
        }
    }

    /// Sleep this core on timeline `t`'s fence line until the row stores to
    /// it or `cap` passes, when `done` still reads `observed`. The row's
    /// store is the wake — `UMWAIT`/`MONITORX` end on a write to the line
    /// from any agent — so no wake call crosses the pair.
    fn sleep_while(&self, t: u16, observed: u64, cap: Duration) {
        if let Some(watch) = self.watch(t) {
            watch.sleep_while(observed, cap);
        }
    }
}

impl<T: rt_transport::Transport> RingFs<'_, '_, T> {
    /// One control call on the pair, a service refusal folded into
    /// [`Refusal::Fs`] and anything else into [`Refusal::Pair`].
    async fn stor_call(&mut self, request: &FsRequest) -> Result<FsResponse, Refusal> {
        match self.call(request).await {
            Ok(FsResponse::Refused(error)) => Err(Refusal::Fs(error)),
            Ok(response) => Ok(response),
            Err(_) => Err(Refusal::Pair),
        }
    }

    /// Attach `window` — the caller's own mapping of `lease`, page-aligned
    /// and at least [`WINDOW_BYTES`] — as this session's ring. The window is
    /// formatted here, then lent as a view of `lease`'s first
    /// [`WINDOW_BYTES`]. The caller keeps `lease` (and the mapping) for as
    /// long as the ring lives: dropping them before
    /// [`stor_detach`](RingFs::stor_detach) tears the row's mapping out from
    /// under it.
    pub async fn stor_attach<S: ReturnLeaseSource>(
        &mut self,
        lease: &S,
        window: &'static mut [u8],
    ) -> Result<StorRing, Refusal> {
        if window.len() < WINDOW_BYTES {
            return Err(Refusal::Fs(FsError::Range));
        }
        let (window, _) = window.split_at_mut(WINDOW_BYTES);
        kore_ring::stor::format(window).map_err(|_| Refusal::Fs(FsError::Range))?;
        let (block_size, max_transfer) =
            match self.lend_ring_window(lease, WINDOW_BYTES as u32).await {
                Ok(FsResponse::StorAttached {
                    block_size,
                    max_transfer,
                }) => (block_size, max_transfer),
                Ok(FsResponse::Refused(error)) | Err(RingFsError::Refused(error)) => {
                    return Err(Refusal::Fs(error));
                }
                _ => return Err(Refusal::Pair),
            };
        let ring = kore_ring::stor::client(window).map_err(|_| Refusal::Fs(FsError::Range))?;
        Ok(StorRing {
            client: StorClient::new(ring),
            block_size,
            max_transfer,
            failure: None,
            completed: 0,
            patience_ns: None,
            stall: None,
        })
    }

    /// Install a read grant over the designated pack region into this
    /// session's ring. `Unsupported` when the machine designates none.
    pub async fn stor_grant_pack(&mut self) -> Result<RingGrant, Refusal> {
        match self.stor_call(&FsRequest::StorGrantPack).await? {
            FsResponse::StorGranted { grant, bytes } => Ok(RingGrant {
                id: GrantId::from_fd(grant as i32).ok_or(Refusal::Pair)?,
                bytes,
            }),
            _ => Err(Refusal::Pair),
        }
    }

    /// Install a read grant over `version`'s whole blocks into this
    /// session's ring. The grant lives as long as the lease: releasing the
    /// version is refused `Busy` while a read under it is in flight. The
    /// grant covers `bytes`, a whole number of blocks from offset zero; the
    /// version's partial last block, if any, is read through the lease.
    /// `Range` when the version holds no whole block.
    pub async fn stor_grant_version(&mut self, version: &LentVersion) -> Result<RingGrant, Refusal> {
        match self
            .stor_call(&FsRequest::StorGrantVersion {
                lease: version.lease(),
            })
            .await?
        {
            FsResponse::StorGranted { grant, bytes } => Ok(RingGrant {
                id: GrantId::from_fd(grant as i32).ok_or(Refusal::Pair)?,
                bytes,
            }),
            _ => Err(Refusal::Pair),
        }
    }

    /// Lend one device-mapping route over caller-funded memory as a target
    /// of this session's ring. The caller keeps its own CPU child of the
    /// same route to read what lands. A refusal leaves the row no device
    /// mapping of it, except `Busy`: then the route must be held until the
    /// ring detaches, as after a release answered `Busy`.
    pub async fn stor_target<C: ReturnCapability>(&mut self, route: C) -> Result<RingTarget, Refusal> {
        match self.lend_ring_target(route).await {
            Ok(FsResponse::StorTargetAdopted { slot, bytes }) => Ok(RingTarget { slot, bytes }),
            Ok(FsResponse::Refused(error)) => Err(Refusal::Fs(error)),
            _ => Err(Refusal::Pair),
        }
    }

    /// Hand `target` back. Success means the row holds no device mapping
    /// of it any more: the lent route may retire. `Busy` while a command
    /// uses it, or when its mapping outlived the release (then the slot is
    /// gone, and the route must be held until the ring detaches).
    pub async fn stor_release(&mut self, target: RingTarget) -> Result<(), Refusal> {
        self.stor_call(&FsRequest::StorRelease { slot: target.slot })
            .await
            .map(|_| ())
    }

    /// Make every queued submission of `class` visible to the row, and send
    /// the notice a parked row is owed. While the row is busy the publish
    /// costs no call at all.
    pub async fn stor_publish(&mut self, ring: &mut StorRing, class: Class) -> Result<(), Refusal> {
        if ring.client.publish(class) {
            self.stor_call(&FsRequest::StorKick).await?;
        }
        Ok(())
    }

    /// Queue every read of `reads` on `class` and `timeline`, then publish.
    /// Whenever the channel fills, publish and wait out the timeline's next
    /// ticket for room — so `class`'s channel must carry only this
    /// timeline's submissions while the batch runs. Tickets on one timeline
    /// complete in order: the returned last ticket covers the whole batch
    /// (`None` for an empty one).
    pub async fn stor_read_batch(
        &mut self,
        ring: &mut StorRing,
        class: Class,
        grant: RingGrant,
        timeline: u16,
        reads: impl IntoIterator<Item = ReadSpec>,
        watch_ns: u64,
    ) -> Result<Option<Ticket>, Refusal> {
        let mut last = None;
        for read in reads {
            let mut done = 0;
            while done < read.len {
                let len = (read.len - done).min(MAX_SUBMISSION) as u32;
                while ring.room(class) == 0 {
                    self.stor_publish(ring, class).await?;
                    // The publish may have kicked the row, and the row may
                    // have drained the channel before the kick answered:
                    // look again before choosing what to wait for.
                    if ring.room(class) != 0 {
                        break;
                    }
                    let done = ring.fence(timeline).map_or(0, |fence| fence.done);
                    let issued = ring.client.issued(timeline).unwrap_or(0);
                    match room_wait_target(done, issued) {
                        Some(seq) => {
                            self.stor_wait(ring, Ticket { timeline, seq }, watch_ns)
                                .await?;
                        }
                        // Everything this timeline issued has completed, yet
                        // the channel is full: it holds another timeline's
                        // submissions. Wake the row and look again, waiting
                        // on the fence line rather than an executor sleep —
                        // this runs under `block_on` too.
                        None => {
                            self.stor_kick().await?;
                            ring.sleep_while(timeline, done, Duration::from_nanos(POLL_PAUSE_NS));
                        }
                    }
                }
                let ticket = ring
                    .read(
                        class,
                        grant,
                        read.offset + done,
                        len,
                        read.target,
                        read.target_offset + done,
                        timeline,
                    )
                    .map_err(|_| Refusal::Fs(FsError::Range))?;
                last = Some(ticket);
                done += u64::from(len);
            }
        }
        self.stor_publish(ring, class).await?;
        Ok(last)
    }

    /// The transport engine's counters.
    pub async fn stor_inspect(&mut self) -> Result<StorCounters, Refusal> {
        match self.stor_call(&FsRequest::StorInspect).await? {
            FsResponse::StorInspected {
                submissions,
                completions,
                in_flight,
                expired,
                stale,
                timeline_refusals,
                stage_refusals,
                last_stage_error,
                aborts,
                aborts_refused,
                weighted,
                rings,
            } => Ok(StorCounters {
                submissions,
                completions,
                in_flight,
                expired,
                stale,
                timeline_refusals,
                stage_refusals,
                last_stage_error,
                aborts,
                aborts_refused,
                weighted,
                rings,
            }),
            _ => Err(Refusal::Pair),
        }
    }

    /// Send the notice a parked row is owed, unconditionally — a lost-wake
    /// probe: work that completes only after this was never seen.
    pub async fn stor_kick(&mut self) -> Result<(), Refusal> {
        self.stor_call(&FsRequest::StorKick).await.map(|_| ())
    }

    /// Wait for `ticket`. For the first `watch_ns` this core sleeps on the
    /// timeline's fence line, which costs no call; past that each round also
    /// kicks the row, so a lost wake costs `watch_ns` once and the wait is
    /// never slower than a pair that polls.
    ///
    /// Answers how many calls the wait made: zero when the fence alone
    /// completed it, which is the interrupt path working.
    pub async fn stor_wait(
        &mut self,
        ring: &mut StorRing,
        ticket: Ticket,
        watch_ns: u64,
    ) -> Result<u32, Refusal> {
        let mut calls = 0u32;
        let started = kore_rt::clock::uptime().map_or(0, |now| now.as_nanos());
        let deadline = kore_rt::clock::uptime()
            .ok()
            .map(|now| now.as_nanos().saturating_add(watch_ns));
        loop {
            if let Some(patience) = ring.patience_ns {
                let now = kore_rt::clock::uptime().map_or(0, |now| now.as_nanos());
                if now.saturating_sub(started) >= patience {
                    let engine = self.stor_inspect().await.ok();
                    ring.drain();
                    ring.stall = Some(StallReport {
                        ticket,
                        fence: ring.fence(ticket.timeline).unwrap_or_default(),
                        row: ring.row_state(),
                        room: [
                            ring.room(Class::Demand),
                            ring.room(Class::Stream),
                            ring.room(Class::Commit),
                        ],
                        drained: ring.completed,
                        engine,
                        waited_ns: now.saturating_sub(started),
                    });
                    return Err(Refusal::Fs(FsError::Busy));
                }
            }
            let observed = ring.fence(ticket.timeline).map_or(0, |fence| fence.done);
            match ring.state(ticket) {
                TicketState::Done => return Ok(calls),
                TicketState::Faulted => {
                    return Err(Refusal::Fs(FsError::ProviderFailure(ProviderError::Lost)));
                }
                TicketState::Pending => {}
            }
            let remaining = match (deadline, kore_rt::clock::uptime()) {
                (Some(deadline), Ok(now)) => deadline.saturating_sub(now.as_nanos()),
                _ => 0,
            };
            if remaining != 0 {
                ring.sleep_while(ticket.timeline, observed, Duration::from_nanos(remaining));
                continue;
            }
            calls = calls.saturating_add(1);
            if calls > 1 {
                // A bounded wait on the fence's own line, not an executor
                // sleep: this also runs under `block_on` from a model's read
                // callback, where no executor is polling.
                ring.sleep_while(
                    ticket.timeline,
                    observed,
                    Duration::from_nanos(POLL_PAUSE_NS),
                );
            }
            self.stor_call(&FsRequest::StorKick).await?;
        }
    }

    /// Detach this session's ring: the row stops popping at once, and this
    /// returns once nothing it issued is in flight and its targets are
    /// released. Only then may the caller unmap the window and retire the
    /// targets' routes.
    ///
    /// Bounded: a device that never completes leaves the ring `Busy` past
    /// [`DETACH_PATIENCE_NS`], and the caller must then keep the window
    /// mapped and funded — the row drains it when the session closes.
    pub async fn stor_detach(&mut self, ring: StorRing) -> Result<(), Refusal> {
        drop(ring);
        let started = kore_rt::clock::uptime().map_or(0, |now| now.as_nanos());
        loop {
            match self.stor_call(&FsRequest::StorDetach).await {
                Ok(_) => return Ok(()),
                Err(Refusal::Fs(FsError::Busy)) => {
                    let now = kore_rt::clock::uptime().map_or(u64::MAX, |now| now.as_nanos());
                    if now.saturating_sub(started) >= DETACH_PATIENCE_NS {
                        return Err(Refusal::Fs(FsError::Busy));
                    }
                    kore_rt::sleep_for(POLL_PAUSE_NS).await;
                }
                Err(other) => return Err(other),
            }
        }
    }
}

/// The same calls on the facade, over its pair.
impl Fs<'_, '_> {
    /// See [`RingFs::stor_attach`].
    pub async fn stor_attach<S: ReturnLeaseSource>(
        &mut self,
        lease: &S,
        window: &'static mut [u8],
    ) -> Result<StorRing, Refusal> {
        self.client.stor_attach(lease, window).await
    }

    /// Reads ([`read_file`](Self::read_file), large
    /// [`read_at`](Self::read_at)) streamed through this launch's own storage
    /// ring — the witness that facade reads take it.
    pub fn facade_ring_files(&self) -> u64 {
        self.ring_files
    }

    /// See [`RingFs::stor_grant_pack`].
    pub async fn stor_grant_pack(&mut self) -> Result<RingGrant, Refusal> {
        self.client.stor_grant_pack().await
    }

    /// See [`RingFs::stor_grant_version`].
    pub async fn stor_grant_version(&mut self, version: &LentVersion) -> Result<RingGrant, Refusal> {
        self.client.stor_grant_version(version).await
    }

    /// Pin `file`'s current version and read its identity: the version a
    /// ring grant ([`stor_grant_version`](Self::stor_grant_version)) and
    /// [`read_version`](Self::read_version) read, unchanged by later writes,
    /// until [`release_version`](Self::release_version).
    pub async fn lend_version(&mut self, file: &crate::View) -> Result<LentVersion, Refusal> {
        let pending = match self.client.lend_version(file.handle(), 0).await {
            Ok(pending) => pending,
            Err(RingFsError::Refused(error)) => return Err(Refusal::Fs(error)),
            Err(_) => return Err(Refusal::Pair),
        };
        match pending.identify(self.window, 0) {
            Ok(version) => Ok(version),
            Err(failure) => {
                let _ = self.client.release_pending_version(failure.version).await;
                Err(Refusal::Fs(failure.error))
            }
        }
    }

    /// Read at most one transfer window of `version` at `offset` into `dst`
    /// through the pair. `Ok(0)` at or past its end.
    pub async fn read_version(
        &mut self,
        version: &LentVersion,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, Refusal> {
        let want = dst.len().min(self.chunk()).min(u32::MAX as usize);
        if want == 0 {
            return Ok(0);
        }
        let count = match self.client.read_version(version, offset, 0, want as u32).await {
            Ok(count) => count as usize,
            Err(RingFsError::Refused(error)) => return Err(Refusal::Fs(error)),
            Err(_) => return Err(Refusal::Pair),
        };
        if count > want {
            return Err(Refusal::Pair);
        }
        dst[..count].copy_from_slice(&self.window[..count]);
        Ok(count)
    }

    /// Release a lent version. A version a ring grant still reads through is
    /// released once those reads drain; this waits for that, bounded by
    /// [`DETACH_PATIENCE_NS`], and hands the version back if it cannot.
    pub async fn release_version(&mut self, version: LentVersion) -> Result<(), (LentVersion, Refusal)> {
        let started = kore_rt::clock::uptime().map_or(0, |now| now.as_nanos());
        let mut version = version;
        loop {
            match self.client.release_version(version).await {
                Ok(()) => return Ok(()),
                Err(failure) => {
                    version = failure.version;
                    let now = kore_rt::clock::uptime().map_or(u64::MAX, |now| now.as_nanos());
                    match failure.error {
                        RingFsError::Refused(FsError::Busy)
                            if now.saturating_sub(started) < DETACH_PATIENCE_NS =>
                        {
                            kore_rt::sleep_for(POLL_PAUSE_NS).await;
                        }
                        RingFsError::Refused(error) => return Err((version, Refusal::Fs(error))),
                        _ => return Err((version, Refusal::Pair)),
                    }
                }
            }
        }
    }

    /// See [`RingFs::stor_target`].
    pub async fn stor_target<C: ReturnCapability>(&mut self, route: C) -> Result<RingTarget, Refusal> {
        self.client.stor_target(route).await
    }

    /// See [`RingFs::stor_release`].
    pub async fn stor_release(&mut self, target: RingTarget) -> Result<(), Refusal> {
        self.client.stor_release(target).await
    }

    /// See [`RingFs::stor_publish`].
    pub async fn stor_publish(&mut self, ring: &mut StorRing, class: Class) -> Result<(), Refusal> {
        self.client.stor_publish(ring, class).await
    }

    /// See [`RingFs::stor_read_batch`].
    pub async fn stor_read_batch(
        &mut self,
        ring: &mut StorRing,
        class: Class,
        grant: RingGrant,
        timeline: u16,
        reads: impl IntoIterator<Item = ReadSpec>,
        watch_ns: u64,
    ) -> Result<Option<Ticket>, Refusal> {
        self.client
            .stor_read_batch(ring, class, grant, timeline, reads, watch_ns)
            .await
    }

    /// See [`RingFs::stor_kick`].
    pub async fn stor_kick(&mut self) -> Result<(), Refusal> {
        self.client.stor_kick().await
    }

    /// See [`RingFs::stor_wait`].
    pub async fn stor_wait(
        &mut self,
        ring: &mut StorRing,
        ticket: Ticket,
        watch_ns: u64,
    ) -> Result<u32, Refusal> {
        self.client.stor_wait(ring, ticket, watch_ns).await
    }

    /// See [`RingFs::stor_detach`].
    pub async fn stor_detach(&mut self, ring: StorRing) -> Result<(), Refusal> {
        self.client.stor_detach(ring).await
    }

    /// See [`RingFs::stor_inspect`].
    pub async fn stor_inspect(&mut self) -> Result<StorCounters, Refusal> {
        self.client.stor_inspect().await
    }
}

#[cfg(test)]
mod tests {
    use super::room_wait_target;

    /// A full channel waits for the next issued sequence, never for one
    /// past the last issued — the ticket that hung the decode on metal.
    #[test]
    fn a_full_channel_never_waits_past_the_last_issued_sequence() {
        assert_eq!(room_wait_target(997_830, 997_840), Some(997_831));
        assert_eq!(room_wait_target(997_839, 997_840), Some(997_840));
        assert_eq!(room_wait_target(997_840, 997_840), None, "all done: nothing to wait for");
        assert_eq!(room_wait_target(0, 0), None);
    }
}
