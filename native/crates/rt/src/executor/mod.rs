//! # `executor` — the doorbell-muxed async executor
//! (`res/USERLAND.md` §3.1/§3.3).
//!
//! **Single-threaded per task, heapless.**  The fixed task slab holds
//! type-erased task records — never futures by value; each
//! spawned future lives in a per-task static [`TaskStorage`] (the embassy
//! pattern, the spine `#[kore_rt::task]` will wrap in Phase S), and the
//! root task is pinned in the entry caller's frame
//! ([`Executor::run_pinned`]).  Spawn
//! capacity is exactly the statically declared task set — refusal on
//! overflow (§8).
//!
//! ## The doorbell mux (§3.1)
//!
//! The executor owns **one doorbell channel**: a `ChanRecv` it keeps a
//! **standing recv post** on, whose paired `ChanSend`s peers hold.  A
//! peer with something to say commits one session delivering a
//! bind-inert notice into the standing post's slot (the §3.4.1
//! notice-mode send — `ChanTx::send`); kicks are level-triggered and
//! coalescing: the state that matters is in the rings, the notice only
//! says "look".  The idle cycle runs the **pinned order**:
//!
//! 1. **consume** the delivered notice (the cookie is a hint);
//! 2. **re-post** the doorbell recv *as its own commit* — the post now
//!    stands, so any kick during the drain is accepted, not refused
//!    (which is what makes a `WouldBlock`ed kick droppable: the
//!    producer's ring write happens-before its kick, and a consumed post
//!    means this executor is awake and about to drain);
//! 3. **drain every ready source** — the drain is **total**: every task
//!    is woken and re-polled, so a source whose kick was coalesced away
//!    loses nothing (§3.3: duplicate wakes harmless, missed edges
//!    impossible); timer wakes, by contrast, are exact ([`sleep_until`]);
//! 4. **park** on the standing post — the runtime's ONE
//!    `commit_and_park` caller ([`ParkTransport`]), with the earliest
//!    pending deadline folded in (`WAIT_DELIVERY | WAIT_SLEEP`, §3.4.4)
//!    when a sleep is pending.
//!
//! A spurious wake (a `Deadline` outcome with nothing due, a stale
//! notify) drains, finds nothing, and parks again on the **same**
//! standing post — no re-post happens and none would be legal.
//!
//! ## Host-test ceiling (§8)
//!
//! A host build cannot park: `commit_and_park` refuses typed off the
//! metal target. Host tests can still exercise protocol and wake discipline
//! with explicit outcomes; the dedicated target boot gate proves real
//! blocking.

mod task;
mod time;

use core::cell::Cell;
use core::future::Future;
use core::pin::pin;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use core::task::{Context, Poll};
use core::time::Duration;

use kore_abi::session::{SlotRef, Status};

use crate::cap::verbs;
use crate::hint;
use crate::session::SessionBuilder;
use crate::transport::{ParkSpec, ParkTransport, Transport};
use crate::wake;

pub use task::{SpawnRefused, SpawnToken, TaskStorage};
pub use time::{
    WakeReason, clock_ns, enable_irq_parks, next_wake, next_wake_for, next_wake_on,
    next_wake_on_for, next_wake_until, quiesce_asked, quiesce_deadline, retire_irq_parks,
    sleep_for, sleep_until, take_quiesce_ask,
};

use task::{TaskHeader, poll_task, waker_for};
pub(crate) use time::ExecCtx;

/// Concurrently admitted tasks per executor (excluding the root) — the
/// §3.3/§8 design bound: spawn capacity is the declared task set,
/// overflow refuses.
pub const MAX_TASKS: usize = 16;

/// How many monitored turns an idle executor spends on its wake bell before
/// it parks, and the cap on one.
///
/// A third budget beside the futex's `LOCK_SPIN` and `BARRIER_SPIN`, sized
/// the way those are: by **who the counterparty is**.  A lock's is a holder
/// releasing in hundreds of nanoseconds; a barrier's is a co-scheduled
/// sibling inside a scheduling-jitter window.  An idle executor's is a peer
/// that has *already published* and is a few instructions from ringing — so
/// the useful window is short, and past it the honest reading is not "late"
/// but "nobody is coming", which is what the park is for.
///
/// Sixteen microseconds of ceiling: comfortably more than the handful of
/// instructions between a peer's publish and its ring, and cheap against the
/// two syscalls plus wake latency it displaces.  It costs nothing when the
/// ring is prompt — the peer's own store ends the sleep — and bounds only the
/// tail, when no ring was coming at all.
const IDLE_SPIN_TURNS: u32 = 4;
/// Per-turn cap for [`IDLE_SPIN_TURNS`].
const IDLE_SPIN_CAP: Duration = Duration::from_micros(4);

/// Consecutive unenumerated park statuses tolerated before the executor calls
/// it a fault.
///
/// The catch-all used to panic on the first one, which made **any** status the
/// kernel later gains a simultaneous silent death of every service in the
/// tree. `Status::WouldBlock` already establishes that an unrecognized outcome
/// can simply re-park, so an unknown one does the same — but only so far: a
/// status that repeats forever is a livelock, and a livelocked executor is the
/// same invisible wedge by a slower route. Sixteen is chosen to be far above
/// any plausible burst of genuinely spurious wakes and far below a loop that
/// is getting nowhere.
const UNKNOWN_STATUS_TOLERANCE: u32 = 16;

/// **Every way the executor's park can end badly, as one reported fault.**
///
/// These were seven bare `panic!`/`assert!` sites. A boot-founded service
/// now reports through its founder beacon and exits through its delivered
/// `role::SELF`; the retained founder handle then lets the kernel supervisor
/// observe and reap it. Early contexts without SELF still park fail-closed.
/// The typed rungs also keep the seven protocol failures distinguishable.
///
/// Reported through the ordinary rung ladder instead, in a high band (0x60..)
/// that cannot collide with any service's own map — services own 1..~34 and
/// [`RUNG_PANIC`](crate::entry::RUNG_PANIC) is 0x7f. The offending
/// [`Status`] rides the detail bits, so `ParkRefused(Denied)` and
/// `ParkRefused(Malformed)` are distinguishable from a photograph, which on a
/// serial-less machine is the surface that exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecFault {
    /// A primary doorbell wake that delivered no notice to consume.
    NoDoorbellNotice,
    /// The kernel reported alternate delivery with no alternate post standing.
    AltDeliveryNoPost,
    // Rung 0x62 (`AltDeliveryNoDelivery`) is retired: a readable, vacant
    // alternate landing is a spurious wake the mux tolerates and counts
    // (`alt_vacant_wakes`). The number stays reserved so old photographs
    // decode; no new fault may reuse it.
    /// The doorbell re-post could not even be staged.
    RepostStage,
    /// The doorbell re-post was refused with this status.
    RepostRefused(Status),
    /// The park itself was refused with this status.
    ParkRefused(Status),
    /// [`UNKNOWN_STATUS_TOLERANCE`] consecutive unenumerated statuses. Rung
    /// `0x70 | status` — one per status, so the exit code alone names it.
    UnknownStatusStorm(Status),
    /// The park itself could not be committed — the trap did not open.
    ParkTransport,
    /// An alternate-delivery wait and an IRQ park stood at once. They do not
    /// compose, and this is executor-internal state rather than anything the
    /// kernel reported: two futures installed contradictory wake sources.
    AltDeliveryWithIrqPark,
    /// The kernel reported alternate delivery into a landing slot this
    /// thread's read-only slot alias cannot read at all — distinct from a
    /// readable slot holding no record, which is tolerated (retired rung
    /// 0x62). The detail carries the slot index.
    AltDeliveryUnreadable(u32),
}

impl ExecFault {
    /// `[7:0]` rung, `[63:8]` detail — the payload convention shared with
    /// `kore-svc-input`, `kore-svc-input-stream` and `kore-svc-compositor`,
    /// so one decoder ring reads every map in the tree.
    fn payload(self) -> u64 {
        let (rung, detail): (u64, u64) = match self {
            Self::NoDoorbellNotice => (0x60, 0),
            Self::AltDeliveryNoPost => (0x61, 0),
            Self::RepostStage => (0x63, 0),
            Self::RepostRefused(s) => (0x64, s as u16 as u64),
            Self::ParkRefused(s) => (0x65, s as u16 as u64),
            // One rung per storming status (0x70..): the exit ABI retains only
            // the rung, and the status IS the finding — a photograph of the
            // exit code must name it without the beacon's detail bits.
            Self::UnknownStatusStorm(s) => (0x70 | (s as u16 as u64 & 0xf), s as u16 as u64),
            Self::ParkTransport => (0x67, 0),
            Self::AltDeliveryWithIrqPark => (0x68, 0),
            Self::AltDeliveryUnreadable(slot) => (0x69, u64::from(slot)),
        };
        rung | (detail << 8)
    }
}

/// Report an executor fault on the founder beacon, then exit through SELF.
///
/// Rebuilds the `Rt` from this thread's anchored boot page exactly as
/// [`panic_refuse`](crate::entry::panic_refuse) does — a proven path, not a
/// new one — so the founder names the rung instead of burning its whole
/// readiness budget and reporting "no signal". Degrades to the mute park only
/// where no boot page or SELF handle is anchored, which is the early context
/// that has no complete lifecycle channel yet.
pub(crate) fn executor_fault(fault: ExecFault) -> ! {
    match crate::entry::thread_context() {
        Some(rt) => crate::entry::refuse_with(&rt, fault.payload()),
        None => crate::entry::refuse(),
    }
}

/// Cooperatively end this poll when a bounded batch left known work queued.
/// This schedules another executor pass without inventing a timer or waiting
/// for an already-coalesced external doorbell. Other ready tasks can run first.
pub async fn yield_now() {
    let mut yielded = false;
    core::future::poll_fn(|cx| {
        if yielded {
            core::task::Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    })
    .await
}

/// **May the monitored prelude run at all**, given this idle transition's
/// posture — the decline table of [`Executor::monitor_bell`], which see for
/// why each answer is what it is.
///
/// A free `const fn` rather than five inline conditions because the executor
/// has no host-side park transport to test against ([`ParkTransport`] is
/// sealed, deliberately — nothing but the trap transport may claim park
/// authority), and of everything the bell tier adds, the decline table is the
/// part that must not silently drift. Extracted, it is exhaustively testable
/// on any machine; the park cycle around it is the boot gate's to prove.
const fn may_monitor(
    has_bell: bool,
    rung_before: bool,
    deadline: Option<u64>,
    irq: bool,
    alternate: bool,
    monitor: bool,
) -> bool {
    has_bell && rung_before && deadline.is_none() && !irq && !alternate && monitor
}

/// The executor's **doorbell** — the §3.1 wake edge, named by the
/// caller-owned pieces the parking commit and the consume step touch.
/// Peers kick it through `ChanTx`s paired to `chan`.
#[derive(Clone, Copy, Debug)]
pub struct Doorbell {
    /// The doorbell channel (the standing post's board entry).
    pub chan: u32,
    /// The held `ChanRecv` end's slot — the post's authority.
    pub recv: SlotRef,
    /// The standing post's landing slot (one slot, re-donated by every
    /// re-post; notices are consumed out of it).
    pub land: SlotRef,
    /// The recv end's tag — the `ParkReq` liveness pin (§3.4.3).
    pub recv_tag: u32,
}

/// The doorbell-muxed executor.  One per task (§3.3); `run` never
/// returns on target (the root is `async fn main -> !`), and returns the
/// root's output when the root completes (which is what makes the cycle
/// host-testable end to end).
pub struct Executor<'t, T: ParkTransport> {
    transport: &'t T,
    doorbell: Doorbell,
    tasks: [Cell<Option<NonNull<TaskHeader>>>; MAX_TASKS],
    ctx: ExecCtx,
    /// Whether the doorbell post currently stands (the §3.1 re-post
    /// bookkeeping — a spurious wake re-parks without re-posting).
    posted: Cell<bool>,
    /// Consecutive unenumerated park statuses seen — the run
    /// [`UNKNOWN_STATUS_TOLERANCE`] bounds. Any recognized outcome clears it.
    unknown_statuses: Cell<u32>,
    /// Doorbell deliveries consumed — cycle telemetry the §3.1 tests
    /// assert coalescing with.
    cycles: Cell<u64>,
    /// Idle parks resolved by the armed IRQ route — the hardware twin of
    /// [`cycles`](Self::cycles) (production MSI-landed telemetry).
    irq_cycles: Cell<u64>,
    /// This executor's [`wake`](crate::wake) bell index, or `None` off the
    /// thread band (every host build, and any thread that cannot name
    /// itself).  `None` disables the bell protocol entirely and restores
    /// exactly the pre-bell behaviour — see [`arm_bell`](Self::arm_bell).
    bell: Option<usize>,
    /// The bell's generation as of the **last total drain** (see
    /// [`wake_all`](Self::wake_all)) — snapshotted once at entry, then only
    /// there.
    ///
    /// The reference the pre-park arm compares against.  The last drain is the
    /// right origin, not the sweep and not the idle path: a peer that rings
    /// while this executor holds `AWAKE` posts no notice, so the ring lives
    /// only in the generation and must stay visible until a drain services it.
    /// Advancing per sweep instead would absorb a ring that landed after a
    /// **non-draining** park return (`Deadline` / `WouldBlock`, which do not
    /// call `wake_all`): the sweep that followed would snapshot it away before
    /// any arm compared against it, and the record it announced would be
    /// stranded — a lost wake.
    bell_seen: Cell<u32>,
    /// Idle parks a bell resolved — never committed, because a peer rang
    /// first.  The syscall pair (`park` + re-post) this seam removes.
    bell_cycles: Cell<u64>,
    /// Alternate wakes whose landing read vacant — spurious by contract, not
    /// faults; a rising rate is the signal a kernel delivery race exists.
    alt_vacant_wakes: Cell<u64>,
}

impl<T: ParkTransport> Drop for Executor<'_, T> {
    fn drop(&mut self) {
        // A caller of `run_pinned` may retain its completed root future and
        // drop the executor first. Invalidate any future-local alternate-wait
        // backlink while the context is still live, so that future's later
        // Drop cannot touch freed executor storage.
        self.ctx.clear_timers();
        // Retire the bell in the safe direction: with no executor running,
        // peers must fall back to posting their notices rather than reading
        // a stale `AWAKE` and dropping them.
        if let Some(index) = self.bell {
            wake::clear_awake(index);
        }
    }
}

impl<'t, T: ParkTransport<Error: core::fmt::Debug>> Executor<'t, T> {
    /// An executor over `transport`, waking on `doorbell`.
    pub fn new(transport: &'t T, doorbell: Doorbell) -> Self {
        Self::with_post_state(transport, doorbell, false)
    }

    /// An executor that adopts a doorbell receive post the caller has already
    /// proved standing.
    ///
    /// This is the bootstrap-to-executor handoff: the caller donates the post
    /// before a child or peer can send its first notice, then temporarily (or
    /// permanently) transfers responsibility for consume/re-post/park to the
    /// executor. The executor does not try to donate the occupied landing a
    /// second time; after any delivery it follows the normal consume then
    /// re-post order and leaves a standing post behind when [`run`](Self::run)
    /// returns.
    ///
    /// The caller must not concurrently park, consume, or re-post this
    /// doorbell while the executor owns the handoff.
    pub fn from_standing(transport: &'t T, doorbell: Doorbell) -> Self {
        Self::with_post_state(transport, doorbell, true)
    }

    fn with_post_state(transport: &'t T, doorbell: Doorbell, posted: bool) -> Self {
        Self {
            transport,
            doorbell,
            tasks: [const { Cell::new(None) }; MAX_TASKS],
            ctx: ExecCtx::new(),
            posted: Cell::new(posted),
            unknown_statuses: Cell::new(0),
            cycles: Cell::new(0),
            irq_cycles: Cell::new(0),
            bell: wake::current(),
            bell_seen: Cell::new(0),
            bell_cycles: Cell::new(0),
            alt_vacant_wakes: Cell::new(0),
        }
    }

    /// Admit a spawned task ([`TaskStorage::spawn`]'s token) to the
    /// slab.  Refuses at [`MAX_TASKS`] (§8: overflow refuses).
    pub fn spawn(&self, token: SpawnToken) -> Result<(), SpawnRefused> {
        for cell in &self.tasks {
            if cell.get().is_none() {
                let hdr = token.header();
                // Admission consumes the token without un-spawning.
                core::mem::forget(token);
                // SAFETY: an admitted header is live 'static storage.
                unsafe { hdr.as_ref() }.mark_ready();
                cell.set(Some(hdr));
                return Ok(());
            }
        }
        Err(SpawnRefused)
    }

    /// Doorbell deliveries consumed so far — the §3.1 coalescing probe.
    pub fn doorbell_cycles(&self) -> u64 {
        self.cycles.get()
    }

    /// Idle parks resolved by the armed interrupt route (`Status::Irq`)
    /// so far — the doorbell probe's hardware twin.  Zero on a
    /// delivery-dead machine whose drains are purely kick-paced; the
    /// per-wake reason itself rides [`next_wake`]'s resolution
    /// ([`WakeReason`]).
    pub fn irq_cycles(&self) -> u64 {
        self.irq_cycles.get()
    }

    /// Idle parks a [`wake`](crate::wake) bell resolved before they
    /// committed — the `park` + re-post syscall pair this seam removes,
    /// counted.
    ///
    /// Rises without a monitor too: the arm alone catches any peer that rang
    /// during the sweep it raced.  Zero only off the thread band, where the
    /// bell protocol is disabled outright.
    pub fn bell_cycles(&self) -> u64 {
        self.bell_cycles.get()
    }

    /// Spurious alternate wakes tolerated so far (see the `AltDelivery` arm).
    pub fn alt_vacant_wakes(&self) -> u64 {
        self.alt_vacant_wakes.get()
    }

    /// The executor's event-driven clock (ns) — advanced by `Deadline`
    /// park outcomes and, on target, anchored to the kernel's monotonic
    /// clock by the stamp every trap return carries
    /// ([`ExecCtx::anchor_ms`](time)) — so it tracks real time to within
    /// one park round-trip.  Host-side no stamp exists and time stays
    /// purely event-driven.
    pub fn now(&self) -> u64 {
        self.ctx.now()
    }

    /// Convenience wrapper that pins `root` in this frame, then runs it
    /// alongside every admitted task.  Entry paths with a large root future
    /// must pin it in their own frame and call [`run_pinned`](Self::run_pinned)
    /// so the future is not materialized in two nested frames.
    ///
    /// On target the root never completes; under a test its output is the
    /// test's result.
    pub fn run<F: Future>(&mut self, root: F) -> F::Output {
        let mut root = pin!(root);
        self.run_pinned(root.as_mut())
    }

    /// Run an already-pinned `root` plus every admitted task
    /// to the root's completion: poll-until-quiescent, then the §3.1
    /// idle park, forever.  The future remains in the caller-owned frame;
    /// this frame holds only a pinned borrow.  On target the root never
    /// completes; under a test its output is the test's result.
    pub fn run_pinned<F: Future + ?Sized>(
        &mut self,
        mut root: core::pin::Pin<&mut F>,
    ) -> F::Output {
        // Establish the standing doorbell post FIRST — §3.1's premise:
        // peers may kick from their very first poll, and a kick is
        // droppable only against a doorbell that already stands (the
        // founding grant's contract).
        if !self.posted.get() {
            self.repost();
        }
        // Publish **awake** before the first sweep, for the same reason the
        // doorbell post is stood before it: a peer may ring from its very
        // first poll, and a bell that still reads parked costs a notice this
        // executor did not need — the safe direction, but the whole point is
        // not to pay it.
        if let Some(index) = self.bell {
            wake::set_awake(index);
            // Establish the drain reference before the first sweep.  From here
            // `bell_seen` advances ONLY at a total drain (see `wake_all`), never
            // once per sweep: a fresh per-sweep snapshot would absorb a ring
            // that landed after a non-draining park return (`Deadline` /
            // `WouldBlock`) and strand it, because those arms do not drain.
            self.bell_seen.set(wake::observe(index));
        }
        let root_hdr = TaskHeader::new();
        root_hdr.mark_ready();
        let out = 'outer: loop {
            // ── drain: poll every woken task until quiescent ─────────
            loop {
                let mut progressed = false;
                if root_hdr.take_ready() {
                    progressed = true;
                    root_hdr.exec.store(
                        &self.ctx as *const ExecCtx as *mut ExecCtx,
                        Ordering::Relaxed,
                    );
                    let waker = waker_for(NonNull::from(&root_hdr));
                    let mut cx = Context::from_waker(&waker);
                    if let Poll::Ready(v) = root.as_mut().poll(&mut cx) {
                        break 'outer v;
                    }
                }
                for cell in &self.tasks {
                    let Some(hdr) = cell.get() else { continue };
                    // SAFETY: admitted headers are live 'static storage.
                    let h = unsafe { hdr.as_ref() };
                    if h.take_ready() {
                        progressed = true;
                        h.exec.store(
                            &self.ctx as *const ExecCtx as *mut ExecCtx,
                            Ordering::Relaxed,
                        );
                        // SAFETY: admitted + armed, on this thread.
                        if unsafe { poll_task(hdr) } {
                            cell.set(None); // completed — slot reopens
                        }
                    }
                }
                if self.ctx.take_inline_doorbell() {
                    // A blocking protocol call consumed and re-posted the
                    // shared bell while one future was on-CPU. Spend that edge
                    // exactly like `idle`: every source re-polls before the
                    // task may sleep, so coalesced notices cannot hide work.
                    self.cycles.set(self.cycles.get() + 1);
                    self.wake_all(&root_hdr);
                    progressed = true;
                }
                if !progressed {
                    break;
                }
            }
            // ── idle: the §3.1 park (consume → re-post → drain → park) ─
            self.idle(&root_hdr);
        };
        // Exit hygiene: the root header is a frame local — nothing may
        // point at it past this return.
        self.ctx.clear_timers();
        // Retire the bell on every run return, not only on Drop: a finite root
        // may return while this `Executor` stays alive (and can be run again),
        // and a stale `AWAKE` in that window would make a peer's ring skip the
        // notice a later run would need — the module's core invariant is that
        // `AWAKE` is set only while an executor is running and will drain. Entry
        // ([`run_pinned`] top) republishes `AWAKE` when a new run starts.
        if let Some(index) = self.bell {
            wake::clear_awake(index);
        }
        out
    }

    /// One idle transition: park on the standing doorbell post (deadline
    /// folded), then run the §3.1 wake path for the outcome.
    fn idle(&self, root_hdr: &TaskHeader) {
        if !self.posted.get() {
            self.repost();
        }
        let deadline = self.ctx.timers_min();
        let alternate = self.ctx.delivery_post();
        if alternate.is_some() && self.ctx.irq() {
            executor_fault(ExecFault::AltDeliveryWithIrqPark);
        }
        // ── the bell tier: monitor, then arm, then park ──────────────
        //
        // Two steps, and only the first is optional.  The monitored window
        // is a *bet* that a peer is about to ring, declined on any machine
        // or in any posture where it would not pay.  The arm is the
        // protocol's commit point and runs before **every** park, bet or no
        // bet: a bell left reading `AWAKE` across a park is a peer skipping
        // the very notice that would have ended it.
        //
        // Short-circuit, deliberately — a monitored window that caught a
        // ring must NOT go on to arm, because this executor is not about to
        // park.
        //
        // A bell wake reports as [`WakeReason::Doorbell`]: to a future
        // awaiting `next_wake` it *is* the doorbell edge, delivered by
        // silicon instead of by the kernel, and the distinction lives in
        // [`bell_cycles`](Self::bell_cycles) where telemetry belongs rather
        // than in a variant every consumer would have to re-match.
        if self.monitor_bell(deadline, alternate.is_some()) || self.arm_bell() {
            self.bell_cycles.set(self.bell_cycles.get() + 1);
            self.ctx.note_wake(time::WakeReason::Doorbell);
            self.wake_all(root_hdr);
            return;
        }
        // The parking commit — wire-complete (the builder emits the park
        // block the kernel decodes) and model-complete (the spec the
        // transport consults); an EMPTY session: the re-post was its own
        // commit, per the pinned order.
        let mut park = SessionBuilder::new();
        park.park_standing(
            self.doorbell.chan,
            self.doorbell.land.index() as u16,
            self.doorbell.recv_tag,
        );
        if let Some(alt) = alternate {
            park.park_standing_alt(alt.chan, alt.land.index() as u16, alt.recv_tag);
        }
        if let Some(d) = deadline {
            park.wait_sleep(d);
        }
        if self.ctx.irq() {
            park.wait_irq();
        }
        let spec = ParkSpec {
            chan: self.doorbell.chan,
            land: self.doorbell.land,
            alternate,
            deadline_ns: deadline,
            irq: self.ctx.irq(),
        };
        let Ok(committed) = self.transport.commit_and_park(&park, &spec) else {
            executor_fault(ExecFault::ParkTransport)
        };
        // Running again, whatever the outcome was: republish `AWAKE` before
        // anything else, so a peer racing this return skips its notice
        // instead of posting one into a bell that still reads parked.
        if let Some(index) = self.bell {
            wake::set_awake(index);
        }
        // Every park return carries the kernel's clock stamp — the anchor
        // that keeps `now()` (and so every relative back-off) honest.
        self.ctx.anchor_ms(committed.clock_ms());
        // Consecutive-run capture: every recognized outcome below leaves this
        // at zero, so only an unbroken run of unrecognized statuses accumulates.
        let unknown_run = self.unknown_statuses.replace(0);
        match committed.status() {
            // Delivered: consume the notice (the cookie is a hint), then
            // re-post as its own commit, then the TOTAL drain — the §3.1
            // pinned order; the caller's poll loop is the drain.
            Status::Ok => {
                match verbs::take_notice(self.doorbell.land) {
                    None => executor_fault(ExecFault::NoDoorbellNotice),
                    // The kernel's terminal quiesce ask rides the ordinary
                    // doorbell (#809 §L); its reserved cookie is refused to
                    // userland senders, so seeing it here is the kernel.
                    Some((cookie, budget_ns))
                        if cookie == kore_abi::session::NOTICE_COOKIE_POWER_QUIESCE =>
                    {
                        self.ctx.note_quiesce_ask(budget_ns);
                    }
                    Some(_) => {}
                }
                self.cycles.set(self.cycles.get() + 1);
                self.ctx.note_wake(time::WakeReason::Doorbell);
                self.posted.set(false);
                self.repost();
                self.wake_all(root_hdr);
            }
            // The kernel resolved the second standing post.  Slot bytes are
            // deliberately only a shape check here; the future that owns the
            // affine receive/landing pair consumes them on its ready leg.
            // Alternate posts carry both capability returns and dedicated
            // notice wakes (the process-stream edge), so neither kind may be
            // consumed or rejected by the executor mux itself.
            Status::AltDelivery => {
                let Some(alt) = alternate else {
                    executor_fault(ExecFault::AltDeliveryNoPost)
                };
                if verbs::take_delivery(alt.land).is_none() {
                    // An unreadable landing is structural and stays fatal.
                    if !verbs::slot_readable(alt.land) {
                        executor_fault(ExecFault::AltDeliveryUnreadable(alt.land.index()));
                    }
                    // A readable, vacant landing is a **spurious alternate
                    // wake**: the kernel resolved the post gone while no
                    // record reached the slot (observed on metal from the
                    // NPU completion lane, 2026-09-17). The mux must not
                    // reject it — the owning future re-checks its own state
                    // on the ready leg, and every alternate-post consumer in
                    // the tree treats the notice as a wake, never as truth.
                    // Counted so the rate stays legible in telemetry.
                    self.alt_vacant_wakes
                        .set(self.alt_vacant_wakes.get().wrapping_add(1));
                }
                // Both posts may have completed before resolution.  Consume
                // and re-stand the level-triggered doorbell when present, but
                // preserve Delivery as the edge-visible reason.
                if let Some((cookie, budget_ns)) = verbs::take_notice(self.doorbell.land) {
                    if cookie == kore_abi::session::NOTICE_COOKIE_POWER_QUIESCE {
                        self.ctx.note_quiesce_ask(budget_ns);
                    }
                    self.cycles.set(self.cycles.get() + 1);
                    self.posted.set(false);
                    self.repost();
                }
                self.ctx.note_wake(time::WakeReason::Delivery);
                self.wake_all(root_hdr);
            }
            // The deadline won: the post stands (no re-post — §3.1's
            // spurious-wake rule), the clock advances, exactly the due
            // sleepers wake.
            Status::Deadline => {
                self.ctx.note_wake(time::WakeReason::Deadline);
                if let Some(d) = deadline {
                    self.ctx.advance(d);
                }
            }
            // The route fired: the post stands (no re-post — the IRQ win
            // is the deadline win's twin), and the TOTAL drain runs — the
            // woken root's device half is what answers here.
            Status::Irq => {
                self.irq_cycles.set(self.irq_cycles.get() + 1);
                self.ctx.note_wake(time::WakeReason::Irq);
                self.wake_all(root_hdr);
            }
            // A scheduler notification resumed this task while the exact
            // doorbell post still stood.  There is no notice to consume and
            // no post to replace; loop back to the same standing park.  This
            // outcome is deliberately quiet and does not wake every future:
            // no executor-visible source became ready.
            Status::WouldBlock => {}
            // The doorbell's recv end was reaped — the runtime's wake edge is
            // gone; nothing correct remains for THIS executor.
            //
            // Not a kernel-protocol violation like the arms above: a reaped
            // recv end is the ordinary shape of teardown, so the kernel is
            // reporting a lifecycle event and this task is the one being
            // retired. Report the typed rung, then take the lawful SELF exit;
            // the retained founder handle gives the parent an exact reap
            // witness.
            Status::Revoked => executor_fault(ExecFault::ParkRefused(Status::Revoked)),
            // An unenumerated status re-parks rather than dying. This used to
            // panic on the first one, which made any status the kernel later
            // gains a simultaneous silent death of every service in the tree —
            // and `WouldBlock` above already establishes that an outcome the
            // executor does not recognize can simply stand its post again.
            //
            // Bounded, because tolerance without a bound is a livelock: a
            // status that repeats forever spins this park at full rate with
            // nothing to show for it, which is the same invisible wedge by a
            // slower route. The run is CONSECUTIVE — any recognized outcome
            // clears it — so genuine sporadic noise never accumulates.
            s => {
                let seen = unknown_run.saturating_add(1);
                if seen >= UNKNOWN_STATUS_TOLERANCE {
                    executor_fault(ExecFault::UnknownStatusStorm(s));
                }
                self.unknown_statuses.set(seen);
                // With an alternate post registered, the refusal is very
                // likely *about that post* — a service retiring a client's
                // completion lane leaves the parked park naming a dead
                // channel, which the kernel answers `Malformed` on every
                // re-park (observed on metal: the accel seal on DeviceLost).
                // Re-parking blind can only storm; waking the owner as a
                // spurious alternate wake lets it settle, find the channel
                // gone, and take its own fallback. Consumers already treat
                // a recordless Delivery as a wake, never as truth.
                if alternate.is_some() {
                    self.ctx.note_wake(time::WakeReason::Delivery);
                    self.wake_all(root_hdr);
                }
            }
        }
    }

    /// **The monitored prelude** — sleep this core on its own wake bell for a
    /// bounded budget, so a sibling's ring resolves the wait with no syscall
    /// at either end.
    ///
    /// `true` when a peer rang and this executor must drain instead of
    /// parking.  This is the one place in the runtime where a core-occupying
    /// wait costs nothing: `idle` is reached only when no task progressed, so
    /// by construction there is no other work the monitored core could be
    /// doing, and the core was about to be surrendered anyway.
    ///
    /// Declined — silently, and back to exactly the pre-bell park — in five
    /// postures, each for its own reason:
    ///
    /// - **no bell**: off the thread band, so nothing can ring.
    /// - **never rung** ([`wake::ever_rung`](crate::wake)): no peer has ever
    ///   announced itself to this executor, so a budget spent watching would
    ///   be waste — and waste that withholds the core for its duration. This
    ///   is what keeps the whole tier **free until it can pay**, with no
    ///   configuration to get wrong.
    /// - **no monitor** ([`hint::available`](crate::hint::available)): the
    ///   turns would degrade to a bare spin, and a bare spin *here* is
    ///   precisely the "burn a core for nothing" this runtime has none of.
    ///   The futex can afford degraded turns because its counterparty is a
    ///   lock holder releasing in hundreds of nanoseconds; an idle executor
    ///   has no such expectation and must go to sleep properly.
    /// - **a pending deadline**: the park serves an exact deadline
    ///   (`Status::Deadline`), and this executor's clock is anchored only at
    ///   park returns — so a budget clamped against [`now`](Self::now) would
    ///   be clamped against a stale reading and could only ever make a timer
    ///   *late*.  A skipped optimisation is not a defect; a late timer is.
    ///   (The upgrade is the syscall-free calibrated read in
    ///   [`clock`](crate::clock), once the two timelines are proved to share
    ///   a base.)
    /// - **an armed IRQ route**: the executor is waiting on hardware, and
    ///   hardware does not store to a bell.  Monitoring first would add the
    ///   whole budget to interrupt latency and catch nothing — the input
    ///   path's regression, written down before it could happen.
    /// - **an armed alternate delivery**: a second standing post
    ///   (`park_standing_alt`) is an independent kernel wake source
    ///   (`next_wake_on`), not a store to this bell — so, exactly like an IRQ
    ///   route, monitoring first would add the whole budget to that delivery's
    ///   latency and catch nothing for it.
    fn monitor_bell(&self, deadline: Option<u64>, alternate_armed: bool) -> bool {
        let Some(index) = self.bell else {
            return false;
        };
        if !may_monitor(
            true,
            wake::ever_rung(index),
            deadline,
            self.ctx.irq(),
            alternate_armed,
            hint::available(),
        ) {
            return false;
        }
        let Some(word) = wake::bell(index) else {
            return false;
        };
        let seen = self.bell_seen.get();
        for _ in 0..IDLE_SPIN_TURNS {
            let current = word.load(Ordering::Acquire);
            if wake::rang(seen, current) {
                return true;
            }
            // Sleeps the core on this line until a store ends it or the cap
            // expires.  `current` rather than `seen` is the expected value:
            // this executor's own `AWAKE` publication moved the word without
            // being a ring, and monitoring for a value the word no longer
            // holds would return instantly, every turn.
            hint::wait_on(word, current, IDLE_SPIN_CAP);
        }
        wake::rang(seen, word.load(Ordering::Acquire))
    }

    /// **The arm** — publish parked, and let one RMW arbitrate the race.
    ///
    /// Runs before *every* park, monitored or not.  `true` means a peer rang
    /// since this sweep began and the park must not be entered; the bell is
    /// restored to `AWAKE` first, because this executor is about to drain
    /// rather than sleep.
    ///
    /// The comparison is against [`bell_seen`](Self::bell_seen) — the last
    /// total drain's generation, not the sweep's — because a peer that rang
    /// while this executor held `AWAKE` posted no notice.  That ring exists
    /// only in the generation, and this is the last place it can be noticed
    /// before the record it announced is stranded; catching it here drains via
    /// the [`wake_all`](Self::wake_all) that follows, which advances
    /// `bell_seen` so the same ring is not re-detected on the next turn.
    ///
    /// With no bell this is a no-op returning `false`, which is what makes
    /// the seam inert off the thread band: the park proceeds exactly as it
    /// did before the bell existed.
    fn arm_bell(&self) -> bool {
        let Some(index) = self.bell else {
            return false;
        };
        let armed = wake::arm_park(index);
        if wake::rang(self.bell_seen.get(), armed) {
            wake::set_awake(index);
            wake::note_resolved(index);
            return true;
        }
        false
    }

    /// Re-donate the doorbell's landing slot — the standing recv post,
    /// **as its own commit** (§3.1 step 2).
    fn repost(&self) {
        let mut s = SessionBuilder::new();
        if s.post_recv(self.doorbell.recv, self.doorbell.land).is_err() {
            executor_fault(ExecFault::RepostStage);
        }
        let Ok(committed) = self.transport.commit_now(&s) else {
            executor_fault(ExecFault::RepostStage)
        };
        self.ctx.anchor_ms(committed.clock_ms());
        match committed.status() {
            // Ok = posted; WouldBlock = a post already stands (a raced
            // re-post) — either way the post stands.
            Status::Ok | Status::WouldBlock => self.posted.set(true),
            s => executor_fault(ExecFault::RepostRefused(s)),
        }
    }

    /// The **total drain** (§3.1 step 3): wake every task + the root.
    /// The consumed cookie is a hint — a coalesced-away kick must lose
    /// nothing, so every source is re-checked by its own future.
    ///
    /// This is also the **only** place `bell_seen` advances after entry: a
    /// total drain re-polls every task, so every ring observed up to this
    /// point is serviced here and the reference the next arm compares against
    /// may move forward to now.  Snapshotting BEFORE the mark-ready is the
    /// conservative edge — a ring that races the drain is left out of the
    /// snapshot and re-caught by the following arm (one extra turn, never a
    /// lost wake), rather than folded in and stranded.
    fn wake_all(&self, root_hdr: &TaskHeader) {
        if let Some(index) = self.bell {
            self.bell_seen.set(wake::observe(index));
        }
        root_hdr.mark_ready();
        for cell in &self.tasks {
            if let Some(hdr) = cell.get() {
                // SAFETY: admitted headers are live 'static storage.
                unsafe { hdr.as_ref() }.mark_ready();
            }
        }
    }
}

/// Establish one task doorbell post outside [`Executor::run_pinned`].
///
/// Pre-executor runtimes (notably the dynamic linker stub) use this before a
/// direct blocking service call. `WouldBlock` is success: it proves an
/// equivalent post already stands. The executor's own synchronous repost
/// remains private because its clock anchoring and `posted` state are part of
/// the executor loop rather than this wire operation.
pub(crate) async fn stand_doorbell<T: Transport>(transport: &T, doorbell: Doorbell) -> Option<u32> {
    let mut repost = SessionBuilder::new();
    repost.post_recv(doorbell.recv, doorbell.land).ok()?;
    let committed = transport.commit(&repost).await.ok()?;
    if !matches!(committed.status(), Status::Ok | Status::WouldBlock)
        || committed.failing_index() != 0
    {
        return None;
    }
    Some(committed.clock_ms())
}

/// Settle a doorbell consumed by an in-future blocking call.
///
/// The call already returned from a kernel `commit_and_park`: consume the
/// notice, re-donate the fixed landing before another call can begin, and tell
/// the owning executor to perform its ordinary total drain. Keeping this in
/// the executor module preserves the one doorbell discipline even though the
/// park itself was fused with a protocol send.
pub(crate) async fn settle_inline_doorbell<T: Transport>(
    transport: &T,
    doorbell: Doorbell,
    stamp_ms: u32,
) -> bool {
    if verbs::take_notice(doorbell.land).is_none() {
        return false;
    }
    let Some(reposted_ms) = stand_doorbell(transport, doorbell).await else {
        return false;
    };
    let settled_ms = stamp_ms.max(reposted_ms);
    core::future::poll_fn(|cx| {
        let hdr = task::header_of(cx.waker())
            .expect("inline doorbell settlement requires the kore executor");
        // SAFETY: `header_of` authenticates the executor-owned header; the
        // owning executor installs its live context for the duration of poll.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "inline doorbell settled outside an executor pass"
        );
        // SAFETY: the assertion above and executor poll discipline prove this
        // pointer remains live through this immediate Ready result.
        unsafe { &*ctx }.note_inline_doorbell(settled_ms);
        Poll::Ready(())
    })
    .await;
    true
}

#[cfg(test)]
mod tests {
    /// Every executor fault must be told apart from a photograph: one rung
    /// each, in a band that cannot collide with a service's own ladder, with
    /// the offending status preserved rather than collapsed.
    #[test]
    fn every_executor_fault_reports_a_distinct_rung() {
        use super::{ExecFault, Status};
        let faults = [
            ExecFault::NoDoorbellNotice,
            ExecFault::AltDeliveryNoPost,
            ExecFault::RepostStage,
            ExecFault::RepostRefused(Status::Denied),
            ExecFault::ParkRefused(Status::Denied),
            ExecFault::UnknownStatusStorm(Status::Denied),
            ExecFault::UnknownStatusStorm(Status::Malformed),
            ExecFault::ParkTransport,
            ExecFault::AltDeliveryWithIrqPark,
            ExecFault::AltDeliveryUnreadable(7),
        ];
        let mut rungs = [0u64; 10];
        for (slot, fault) in rungs.iter_mut().zip(faults) {
            let payload = fault.payload();
            let rung = payload & 0xff;
            // Above every service ladder (services own 1..~34) and clear of
            // `entry::RUNG_PANIC` (0x7f), so one decoder ring reads them all.
            assert!((0x60..0x7f).contains(&rung), "rung {rung:#x} out of band");
            *slot = rung;
        }
        for (i, a) in rungs.iter().enumerate() {
            for b in &rungs[i + 1..] {
                assert_ne!(a, b, "two executor faults share a rung");
            }
        }
    }

    /// The status must survive into the detail bits — `ParkRefused(Denied)`
    /// and `ParkRefused(Malformed)` are different failures.
    #[test]
    fn a_refusing_status_rides_the_detail_bits() {
        use super::{ExecFault, Status};
        let denied = ExecFault::ParkRefused(Status::Denied).payload();
        let malformed = ExecFault::ParkRefused(Status::Malformed).payload();
        assert_eq!(denied & 0xff, malformed & 0xff, "same rung");
        assert_ne!(denied >> 8, malformed >> 8, "different detail");
        assert_eq!(denied >> 8, Status::Denied as u16 as u64);
    }

    /// The timer table must hold every header that can register one.
    ///
    /// The registrants are the admitted tasks PLUS the root future, whose
    /// header is a frame local in `run` rather than a cell in `self.tasks`.
    /// At a flat `MAX_TASKS` the seventeenth registrant overflowed into a
    /// panic — on a boot-founded service, a silent permanent park.
    #[test]
    fn the_timer_table_holds_every_registrant_including_the_root() {
        assert!(
            super::time::MAX_TIMERS >= super::MAX_TASKS + 1,
            "MAX_TIMERS must cover MAX_TASKS admitted tasks plus the root"
        );
    }

    #[test]
    fn cooperative_yield_has_one_pending_poll_and_no_clock_dependency() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        let mut future = core::pin::pin!(super::yield_now());
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(future.as_mut().poll(&mut cx), Poll::Pending);
        assert_eq!(future.as_mut().poll(&mut cx), Poll::Ready(()));
    }
    use super::*;

    /// The one posture that monitors: a bell to watch, a peer known to ring
    /// it, a monitor to watch it with, no deadline to be late for, and no
    /// route to be waiting on.
    #[test]
    fn the_prelude_runs_only_when_every_condition_holds() {
        assert!(
            may_monitor(true, true, None, false, false, true),
            "bell + a proven ringer + monitor + no deadline + no route + no alternate monitors"
        );
    }

    /// Each decline, alone, is sufficient.
    #[test]
    fn any_single_condition_declines_the_prelude() {
        assert!(
            !may_monitor(false, true, None, false, false, true),
            "off the thread band nothing can ring"
        );
        assert!(
            !may_monitor(true, false, None, false, false, true),
            "a bell no peer has ever rung is not worth watching"
        );
        assert!(
            !may_monitor(true, true, Some(1), false, false, true),
            "a pending deadline is the park's to serve exactly"
        );
        assert!(
            !may_monitor(true, true, None, true, false, true),
            "hardware does not store to a bell"
        );
        assert!(
            !may_monitor(true, true, None, false, true, true),
            "an armed alternate delivery is a kernel wake source, not a bell store"
        );
        assert!(
            !may_monitor(true, true, None, false, false, false),
            "without a monitor the turns would be a bare spin"
        );
    }

    /// Exhaustive over the whole table: exactly one of sixty-four postures
    /// monitors. Written as a sweep rather than as individual asserts so that
    /// a condition inverted by a later edit cannot hide in a case nobody
    /// enumerated.
    #[test]
    fn exactly_one_posture_of_the_whole_table_monitors() {
        let mut monitoring = 0;
        for bits in 0..64u8 {
            let has_bell = bits & 1 != 0;
            let rung_before = bits & 2 != 0;
            let deadline = (bits & 4 != 0).then_some(42u64);
            let irq = bits & 8 != 0;
            let monitor = bits & 16 != 0;
            let alternate = bits & 32 != 0;
            if may_monitor(has_bell, rung_before, deadline, irq, alternate, monitor) {
                monitoring += 1;
                assert!(
                    has_bell && rung_before && deadline.is_none() && !irq && !alternate && monitor,
                    "the monitoring posture is the only one, and this is not it"
                );
            }
        }
        assert_eq!(monitoring, 1, "exactly one posture monitors");
    }

    /// The seam is **free until it can pay**: no peer has ever rung, so no
    /// otherwise-eligible posture spends a budget.
    #[test]
    fn an_executor_no_peer_rings_never_spends_a_budget() {
        for bits in 0..32u8 {
            let has_bell = bits & 1 != 0;
            let deadline = (bits & 2 != 0).then_some(42u64);
            let irq = bits & 4 != 0;
            let monitor = bits & 8 != 0;
            let alternate = bits & 16 != 0;
            assert!(
                !may_monitor(has_bell, false, deadline, irq, alternate, monitor),
                "an unrung bell must never monitor, whatever else holds"
            );
        }
    }

    /// The budget is a bound, not a cadence: a monitored window can never
    /// outlast a park's own cost by enough to be the wrong tool.
    #[test]
    fn the_idle_budget_stays_a_microsecond_scale_bet() {
        let ceiling = IDLE_SPIN_CAP.saturating_mul(IDLE_SPIN_TURNS);
        assert!(
            ceiling <= Duration::from_micros(64),
            "an idle bet longer than a barrier's jitter window is a park in disguise"
        );
        assert!(
            !ceiling.is_zero(),
            "a zero budget would make the prelude a no-op that still reads as live"
        );
    }
}
