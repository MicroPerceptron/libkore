//! `time` — the executor's deadline table and [`sleep_until`]
//! (`res/USERLAND.md` §3.4.2/4).
//!
//! Timers are **folded into the doorbell park**: the executor's idle
//! path takes the table's minimum deadline into its one
//! `commit_and_park` (`WAIT_DELIVERY | WAIT_SLEEP`, the §3.4.4 combined
//! form) — one sleeping park per task (§8), never a park per timer.  A
//! `Deadline` outcome advances the executor's clock and expires exactly
//! the due entries (timer wakes are precise; the doorbell drain is the
//! total one).  Host-side, time is **event-driven**: the clock
//! *is* the deadlines the park resolves — no wall clock exists (§8:
//! a host cannot model kernel timing).

use core::cell::Cell;
use core::future::{Future, poll_fn};
use core::marker::PhantomPinned;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::Ordering;
use core::task::{Context, Poll};

use super::task::{TaskHeader, header_of};
use crate::transport::ParkPost;

/// Concurrent pending deadlines per executor — a design bound (§8's
/// posture: refusal on overflow, sized for the declared task set).
///
/// `MAX_TASKS + 1` retains room for one wait in every admitted task plus the
/// root. Registrations belong to individual waits, so composed futures in one
/// task share this same bounded table instead of replacing each other's alarm.
/// Callers must keep the total concurrent waits within this bound: inference's
/// NPU/GPU pair and terminal pump require three, not an unbounded task tree.
/// Exhaustion remains an invariant failure; silently losing an alarm is not a
/// valid refusal for these infallible wait APIs.
pub const MAX_TIMERS: usize = super::MAX_TASKS + 1;

#[derive(Clone, Copy)]
struct TimerEntry {
    deadline_ns: u64,
    task: NonNull<TaskHeader>,
    /// Identity and teardown backlink of the pinned registering wait.
    owner: NonNull<Cell<Option<NonNull<ExecCtx>>>>,
}

/// Why the executor's last idle park returned — what [`next_wake`]
/// resolves with, so a serve loop can tell a hardware-delivered wake
/// ([`Irq`](Self::Irq)) from a peer's kick without any register read.
/// The distinction is the production MSI-landed telemetry a serial-less
/// machine otherwise lacks: kick-paced degradation and real delivery
/// drain the same events and are indistinguishable downstream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WakeReason {
    /// The standing doorbell post consumed a peer's notice (`Status::Ok`)
    /// — also the initial value before any park resolves.
    #[default]
    Doorbell,
    /// The bounded alternate standing post delivered.
    Delivery,
    /// The armed interrupt route fired (`Status::Irq`).
    Irq,
    /// The folded sleep deadline won (`Status::Deadline`).
    Deadline,
}

/// One-way lifecycle of the executor's optional hardware-wake posture.
/// `Retired` is distinct from `Unarmed`: after an owner has quiesced its
/// device and consumed the route, no later code may accidentally resurrect a
/// WAIT_IRQ park against authority that no longer exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum IrqParkPolicy {
    #[default]
    Unarmed,
    Armed,
    Retired,
}

/// The executor's shared context — what a polled future can reach
/// through its waker ([`header_of`] → [`TaskHeader::exec`]): the
/// event-driven clock and the deadline table.
pub(crate) struct ExecCtx {
    /// The executor's monotonic now (ns). In host tests it advances from
    /// explicit `Deadline` outcomes, never from wall-clock time.
    now: Cell<u64>,
    /// Why the last idle park returned ([`WakeReason`]) — read by
    /// [`next_wake`]'s ready leg.
    last_wake: Cell<WakeReason>,
    /// Independently owned pending waits; `None` = free.
    timers: [Cell<Option<TimerEntry>>; MAX_TIMERS],
    /// At most one task may add an alternate standing post to the executor's
    /// next idle park. The registering future clears it on ready or drop.
    delivery: Cell<
        Option<(
            ParkPost,
            NonNull<TaskHeader>,
            u64,
            NonNull<Option<AlternateOwner>>,
        )>,
    >,
    /// Monotone identity for the one alternate registration. Matching the
    /// generation as well as the task prevents one cancelled future from
    /// clearing a successor owned by the same executor task.
    next_delivery: Cell<u64>,
    /// A protocol call consumed and re-posted the executor doorbell inside a
    /// blocking send-and-wait trap. The outer executor spends this edge as a
    /// total drain before it may park again.
    inline_doorbell: Cell<bool>,
    /// Whether every idle park also arms the task's held IRQ route. This is
    /// executor policy rather than future-local wait state: it remains armed
    /// until the hardware owner completes its one-way quiesce + route
    /// retirement transition.
    irq: Cell<IrqParkPolicy>,
    /// The kernel's **terminal quiesce ask** (#809 §L): the doorbell notice
    /// carried `NOTICE_COOKIE_POWER_QUIESCE`, and this is its payload — the
    /// row's remaining budget in nanoseconds.  Latched by the outcome arm,
    /// taken exactly once by whichever loop owns the service.
    quiesce_ask: Cell<Option<u64>>,
}

impl ExecCtx {
    pub(crate) const fn new() -> Self {
        Self {
            now: Cell::new(0),
            last_wake: Cell::new(WakeReason::Doorbell),
            timers: [const { Cell::new(None) }; MAX_TIMERS],
            delivery: Cell::new(None),
            next_delivery: Cell::new(0),
            inline_doorbell: Cell::new(false),
            irq: Cell::new(IrqParkPolicy::Unarmed),
            quiesce_ask: Cell::new(None),
        }
    }

    /// The doorbell notice was the kernel's terminal quiesce ask; remember
    /// its budget until the service loop takes it.  A second ask before the
    /// first was taken keeps the later (smaller) budget.
    pub(crate) fn note_quiesce_ask(&self, budget_ns: u64) {
        self.quiesce_ask.set(Some(budget_ns));
    }

    /// Take the pending quiesce ask, if any.
    pub(crate) fn take_quiesce_ask(&self) -> Option<u64> {
        self.quiesce_ask.take()
    }

    /// Enable the executor's IRQ-or-doorbell idle posture. Once the device
    /// owner retires that posture, re-arming would name consumed authority and
    /// is therefore an invariant violation.
    pub(crate) fn enable_irq(&self) {
        match self.irq.get() {
            IrqParkPolicy::Unarmed => self.irq.set(IrqParkPolicy::Armed),
            IrqParkPolicy::Armed => {}
            IrqParkPolicy::Retired => panic!("retired IRQ park policy cannot be re-armed"),
        }
    }

    /// Permanently retire IRQ-backed parks after the owning device is
    /// quiesced and its route has been consumed. Idempotent for the terminal
    /// row and also lawful from `Unarmed` during failed startup containment.
    pub(crate) fn retire_irq(&self) {
        self.irq.set(IrqParkPolicy::Retired);
    }

    /// Whether the next idle park must arm the held IRQ route.
    pub(crate) fn irq(&self) -> bool {
        self.irq.get() == IrqParkPolicy::Armed
    }

    /// Record why the idle park just returned — the executor's outcome
    /// arms call this before waking tasks.
    pub(crate) fn note_wake(&self, reason: WakeReason) {
        self.last_wake.set(reason);
    }

    /// Why the last idle park returned.
    pub(crate) fn last_wake(&self) -> WakeReason {
        self.last_wake.get()
    }

    /// The executor's current instant (ns).
    pub(crate) fn now(&self) -> u64 {
        if let Some(ns) = crate::clock::monotonic_nanos()
            && ns > self.now.get()
        {
            self.now.set(ns);
        }
        self.now.get()
    }

    /// Anchor the clock to a kernel clock stamp
    /// ([`Committed::clock_ms`](kore_abi::session::Committed::clock_ms)):
    /// advance-to-max, never backward.  Called by the executor on every
    /// trap return it owns (the idle park, the doorbell re-post), so on
    /// target `now()` tracks the kernel's monotonic clock to within one
    /// park round-trip.  Without it the event-driven clock starts at 0
    /// and advances only on `Deadline` outcomes — lagging real time by
    /// the whole boot age, which turned every relative back-off deadline
    /// into an already-past one (the input-stream convergence storm).
    /// A `0` stamp (an unstamped word — host mock transports) anchors
    /// nothing, preserving the host posture: event-driven time only.
    pub(crate) fn anchor_ms(&self, stamp_ms: u32) {
        let ns = (stamp_ms as u64).saturating_mul(1_000_000);
        if ns > self.now.get() {
            self.now.set(ns);
        }
    }

    /// Register or re-anchor one pinned wait, independently of other waits
    /// sharing its task. Only `TimerRegistration` supplies the owner pointer.
    ///
    /// # Panics
    /// When the table is full ([`MAX_TIMERS`]) — the §8 refusal made
    /// loud (a quiet drop would be a lost wake).
    fn register_timer(
        &self,
        deadline_ns: u64,
        task: NonNull<TaskHeader>,
        owner: NonNull<Cell<Option<NonNull<ExecCtx>>>>,
    ) {
        let entry = TimerEntry {
            deadline_ns,
            task,
            owner,
        };
        // Re-polling the same wait must not consume another bounded slot.
        for cell in &self.timers {
            if let Some(previous) = cell.get()
                && previous.owner == owner
            {
                assert_eq!(previous.task, task, "timer wait changed owning task");
                cell.set(Some(entry));
                return;
            }
        }
        // …else claim a free one.
        for cell in &self.timers {
            if cell.get().is_none() {
                cell.set(Some(entry));
                return;
            }
        }
        panic!("executor timer table full (MAX_TIMERS = {MAX_TIMERS})");
    }

    /// The owner has already cleared its backlink. Remove only its entry;
    /// neither a sibling wait nor a reused slot belongs to this cancellation.
    fn clear_timer(&self, owner: NonNull<Cell<Option<NonNull<ExecCtx>>>>) {
        for cell in &self.timers {
            if cell.get().is_some_and(|entry| entry.owner == owner) {
                cell.set(None);
                return;
            }
        }
    }

    /// The earliest pending deadline — what the idle park folds in
    /// (§3.4.4).
    pub(crate) fn timers_min(&self) -> Option<u64> {
        self.timers
            .iter()
            .filter_map(|c| c.get().map(|entry| entry.deadline_ns))
            .min()
    }

    fn register_delivery(
        &self,
        post: ParkPost,
        task: NonNull<TaskHeader>,
        future_owner: NonNull<Option<AlternateOwner>>,
    ) -> u64 {
        assert!(
            self.delivery.get().is_none(),
            "executor alternate delivery wait already owned"
        );
        let mut generation = self.next_delivery.get().wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        self.next_delivery.set(generation);
        self.delivery
            .set(Some((post, task, generation, future_owner)));
        generation
    }

    fn clear_delivery(&self, task: NonNull<TaskHeader>, generation: u64) {
        if self
            .delivery
            .get()
            .is_some_and(|(_, owner, registered, _)| owner == task && registered == generation)
        {
            self.delivery.set(None);
        }
    }

    pub(crate) fn delivery_post(&self) -> Option<ParkPost> {
        self.delivery.get().map(|(post, _, _, _)| post)
    }

    pub(crate) fn note_inline_doorbell(&self, stamp_ms: u32) {
        self.anchor_ms(stamp_ms);
        self.note_wake(WakeReason::Doorbell);
        self.inline_doorbell.set(true);
    }

    pub(crate) fn take_inline_doorbell(&self) -> bool {
        self.inline_doorbell.replace(false)
    }

    /// Advance the clock to `now_ns` and wake exactly the due sleepers
    /// (deadline ≤ now) — the `Deadline` outcome's arm.
    pub(crate) fn advance(&self, now_ns: u64) {
        if now_ns > self.now.get() {
            self.now.set(now_ns);
        }
        for cell in &self.timers {
            if let Some(entry) = cell.get() {
                if entry.deadline_ns <= self.now.get() {
                    cell.set(None);
                    // SAFETY: the pinned wait removes its registration before
                    // drop. Invalidate its backlink before this slot can be
                    // reused, so later cancellation cannot touch a successor.
                    unsafe { entry.owner.as_ref() }.set(None);
                    // SAFETY: registered headers are live for the run
                    // (static storages; the root header outlives every
                    // entry — `clear` runs before `run` returns).
                    unsafe { entry.task.as_ref() }.mark_ready();
                }
            }
        }
    }

    /// Drop every pending entry — `run`'s exit hygiene (the root task's
    /// header is a stack local; no entry may outlive the frame).
    pub(crate) fn clear_timers(&self) {
        for cell in &self.timers {
            if let Some(entry) = cell.replace(None) {
                // SAFETY: as in `advance`: every entry belongs to a pinned,
                // still-live wait. Its later Drop must not follow this context
                // after `Executor::run_pinned` returns or the executor drops.
                unsafe { entry.owner.as_ref() }.set(None);
            }
        }
        if let Some((_, _, _, mut future_owner)) = self.delivery.replace(None) {
            // SAFETY: alternate registration accepts this pointer only from a
            // pinned `AlternateWake`. Its Drop clears the registry before the
            // field dies; therefore a still-registered pointer names a live,
            // immovable owner field. Executor teardown invalidates the future's
            // raw context backlink before this context can itself die.
            unsafe { *future_owner.as_mut() = None };
        }
        self.inline_doorbell.set(false);
    }
}

/// A wait's affine registration in the executor's existing bounded table.
/// The pinned field address identifies it; the context invalidates that field
/// when the timer fires or execution ends, following `AlternateWake`'s teardown
/// pattern. No registration can outlive its wait or follow a destroyed context.
struct TimerRegistration {
    ctx: Cell<Option<NonNull<ExecCtx>>>,
    _pin: PhantomPinned,
}

impl TimerRegistration {
    const fn new() -> Self {
        Self {
            ctx: Cell::new(None),
            _pin: PhantomPinned,
        }
    }

    fn arm(self: Pin<&mut Self>, ctx: &ExecCtx, task: NonNull<TaskHeader>, deadline_ns: u64) {
        let this = self.as_ref().get_ref();
        let context = NonNull::from(ctx);
        if let Some(previous) = this.ctx.get() {
            assert_eq!(previous, context, "timer wait changed executor");
        }
        ctx.register_timer(deadline_ns, task, NonNull::from(&this.ctx));
        // Store only after successful registration: an exhausted table has
        // published nothing for this new owner.
        this.ctx.set(Some(context));
    }

    fn clear(self: Pin<&mut Self>) {
        self.as_ref().get_ref().clear_registration();
    }

    fn clear_registration(&self) {
        let Some(ctx) = self.ctx.take() else { return };
        // SAFETY: expiry and executor teardown invalidate this backlink while
        // the context is live. A remaining backlink therefore names the same
        // executor that owns this pinned wait, on its poll/drop thread.
        unsafe { ctx.as_ref() }.clear_timer(NonNull::from(&self.ctx));
    }
}

impl Drop for TimerRegistration {
    fn drop(&mut self) {
        self.clear_registration();
    }
}

/// Sleep until `deadline_ns` (absolute monotonic nanoseconds) — the §4
/// pinned timer verb.  Resolves on the executor's clock: the idle path
/// folds the earliest pending deadline into its combined park
/// (`WAIT_DELIVERY | WAIT_SLEEP`, §3.4.4), and a `Deadline` outcome
/// wakes exactly the due sleepers.  Already-passed deadlines resolve on
/// first poll.
///
/// # Panics
/// When polled outside the kore executor (a foreign waker): the future
/// needs its executor's timer table, reached through the cookie waker —
/// there is nothing correct to do under another runtime.
pub async fn sleep_until(deadline_ns: u64) {
    let mut timer = core::pin::pin!(TimerRegistration::new());
    poll_fn(move |cx| {
        let hdr =
            header_of(cx.waker()).expect("sleep_until requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this
        // executor's vtable; headers outlive their polls.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "sleep_until polled outside an executor pass"
        );
        // SAFETY: the polling executor set `exec` to its own live ctx
        // for the duration of the pass.
        let ctx = unsafe { &*ctx };
        if ctx.now() >= deadline_ns {
            timer.as_mut().clear();
            Poll::Ready(())
        } else {
            timer.as_mut().arm(ctx, hdr, deadline_ns);
            Poll::Pending
        }
    })
    .await
}

/// Sleep for `delta_ns` **relative** nanoseconds — the back-off verb.
///
/// The deadline is computed on the first poll as `now() + delta_ns`
/// against the executor's kernel-anchored clock
/// ([`ExecCtx::anchor_ms`]), then behaves exactly like [`sleep_until`].
/// Since the anchor refreshes at every park, `now()` is at most one
/// park-to-poll gap behind real time, so the realized sleep is
/// `delta_ns` minus that gap — for pacing purposes, `delta_ns`.
///
/// This — not `sleep_until` with a small literal — is the correct form
/// for a retry back-off: `sleep_until(k * GRACE)` names an *absolute*
/// instant in the boot's first milliseconds, forever in the past by the
/// time any service runs, which silently turns the back-off into a hot
/// retry storm (the input-stream convergence wedge).
///
/// # Panics
/// As [`sleep_until`]: polled outside the kore executor.
pub async fn sleep_for(delta_ns: u64) {
    let mut deadline: Option<u64> = None;
    let mut timer = core::pin::pin!(TimerRegistration::new());
    poll_fn(move |cx| {
        let hdr =
            header_of(cx.waker()).expect("sleep_for requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this
        // executor's vtable; headers outlive their polls.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(!ctx.is_null(), "sleep_for polled outside an executor pass");
        // SAFETY: the polling executor set `exec` to its own live ctx
        // for the duration of the pass.
        let ctx = unsafe { &*ctx };
        let d = *deadline.get_or_insert_with(|| ctx.now().saturating_add(delta_ns));
        if ctx.now() >= d {
            timer.as_mut().clear();
            Poll::Ready(())
        } else {
            timer.as_mut().arm(ctx, hdr, d);
            Poll::Pending
        }
    })
    .await
}

/// The executor's current instant (ns) — [`Executor::now`] reachable
/// from *inside* the root future, where the `&Executor` is not (it is
/// mutably borrowed by [`run`](crate::Executor::run) for the whole
/// pass).  Resolves immediately; the `.await` is only how the future
/// reaches its executor through the waker.
///
/// The clock is kernel-anchored ([`ExecCtx::anchor_ms`]) — a real
/// monotonic instant, not a boot-relative count — so this is how a
/// caller names an **absolute** deadline for
/// [`verbs::wait_interrupt_until`](crate::cap::verbs::wait_interrupt_until):
/// read it once, add the budget, park against that one instant for the
/// whole retry loop.  Do **not** re-derive `now() + delta` per
/// iteration: the anchor only refreshes on parks the *executor* owns,
/// so a loop that parks through the verb directly reads the same stale
/// `now` every time — and after the first pass every derived deadline
/// is already past (the back-off-into-hot-spin failure the `sleep_for`
/// docs name).
///
/// [`Executor::now`]: crate::Executor::now
///
/// # Panics
/// As [`sleep_until`]: polled outside the kore executor.
pub async fn clock_ns() -> u64 {
    poll_fn(|cx| {
        let hdr =
            header_of(cx.waker()).expect("clock_ns requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this
        // executor's vtable; headers outlive their polls.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(!ctx.is_null(), "clock_ns polled outside an executor pass");
        // SAFETY: the polling executor set `exec` to its own live ctx
        // for the duration of the pass.
        Poll::Ready(unsafe { &*ctx }.now())
    })
    .await
}

/// Permanently add the task's held IRQ route to this executor's idle parks.
///
/// Driver roots call this after they have established and retained their
/// remapped route. The policy is deliberately monotonic from inside futures:
/// once serving begins, briefly reverting to a doorbell-only park could lose
/// the only wake edge for an already-completing device operation. A task with
/// no live route must not call this; the kernel will refuse such an IRQ park.
///
/// # Panics
/// When polled outside the kore executor, like [`clock_ns`] and
/// [`next_wake`].
pub async fn enable_irq_parks() {
    poll_fn(|cx| {
        let hdr = header_of(cx.waker())
            .expect("enable_irq_parks requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this executor's
        // waker table, and the executor installs its live context for the
        // duration of every poll.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "enable_irq_parks polled outside an executor pass"
        );
        // SAFETY: the assertion above and executor poll discipline prove the
        // context remains live through this immediate Ready result.
        unsafe { &*ctx }.enable_irq();
        Poll::Ready(())
    })
    .await
}

/// Permanently retire this task's IRQ-backed idle posture after its hardware
/// owner has quiesced the source and consumed the held route.
///
/// The transition is deliberately one-way: a terminal driver row may keep
/// serving over its doorbell, but it can never re-arm WAIT_IRQ without fresh
/// route authority and a fresh executor epoch. The future is immediately
/// ready; the owner calls it in the same root poll that performs containment,
/// before any await that could let the executor park in the intervening state.
/// Calling it before IRQ parks were armed is valid for startup containment.
///
/// # Panics
/// When polled outside the kore executor, like [`enable_irq_parks`].
pub async fn retire_irq_parks() {
    poll_fn(|cx| {
        let hdr = header_of(cx.waker())
            .expect("retire_irq_parks requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this executor's
        // waker table, and the executor installs its live context for the
        // duration of every poll.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "retire_irq_parks polled outside an executor pass"
        );
        // SAFETY: the assertion above and executor poll discipline prove the
        // context remains live through this immediate Ready result.
        unsafe { &*ctx }.retire_irq();
        Poll::Ready(())
    })
    .await
}

/// Complete on the **next** executor wake: `Pending` once, `Ready` on the
/// re-poll — the serve-loop park idiom.  A resident entry's steady loop
/// ends each iteration with `next_wake().await`, so any doorbell kick,
/// IRQ relay, or folded deadline re-runs the whole body (drain → serve →
/// park).  A `core::future::pending().await` there is a one-way door:
/// the root future re-polls straight into the pending await forever, and
/// no later request or report is ever served — the input router's
/// first-boot trap, fixed by this primitive.
///
/// Resolves with the [`WakeReason`] the executor recorded for the park —
/// an IRQ-owning service counts real route deliveries from it (the
/// serial-less machine's only MSI-landed witness).  Callers that only
/// want the wake keep awaiting it bare; the value drops.
///
/// # Panics
/// As [`sleep_until`]: polled outside the kore executor.
pub async fn next_wake() -> WakeReason {
    let mut parked = false;
    core::future::poll_fn(move |cx| {
        if parked {
            let hdr = header_of(cx.waker())
                .expect("next_wake requires the kore executor (foreign waker)");
            // SAFETY: `header_of` vouches the header came from this
            // executor's vtable; headers outlive their polls.
            let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
            assert!(!ctx.is_null(), "next_wake polled outside an executor pass");
            // SAFETY: the polling executor set `exec` to its own live ctx
            // for the duration of the pass.
            Poll::Ready(unsafe { &*ctx }.last_wake())
        } else {
            parked = true;
            Poll::Pending
        }
    })
    .await
}

/// Did the kernel ask this row to quiesce its devices (#809 §L)?  For use
/// **inside a `poll_fn`** that owns the service's park — the generic host's
/// fan-in and the input router's — where the ask must be observed on the
/// very wake that carried it, because a wake with no ring work never leaves
/// the poll.  Takes the ask: `Some(remaining_budget_ns)` exactly once.
///
/// # Panics
/// As [`next_wake`]: polled outside the kore executor.
pub fn quiesce_asked(cx: &core::task::Context<'_>) -> Option<u64> {
    let hdr =
        header_of(cx.waker()).expect("quiesce_asked requires the kore executor (foreign waker)");
    // SAFETY: `header_of` vouches the header came from this executor's
    // vtable; headers outlive their polls.
    let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
    assert!(
        !ctx.is_null(),
        "quiesce_asked polled outside an executor pass"
    );
    // SAFETY: the polling executor set `exec` to its own live ctx for the
    // duration of the pass.
    unsafe { &*ctx }.take_quiesce_ask()
}

/// [`quiesce_asked`] for loops that park with [`next_wake`] and check between
/// wakes: completes immediately with the pending ask, if any.
pub async fn take_quiesce_ask() -> Option<u64> {
    core::future::poll_fn(|cx| Poll::Ready(quiesce_asked(cx))).await
}

/// The absolute deadline a quiesce ask's remaining budget names on this
/// task's clock — `u64::MAX` (no bound) when the clock page is unreadable,
/// which the row's own bounded waits already treat as "spin-bounded".
pub fn quiesce_deadline(budget_ns: u64) -> u64 {
    crate::clock::uptime()
        .ok()
        .map_or(u64::MAX, |now| now.as_nanos().saturating_add(budget_ns))
}

/// Complete on the next executor wake while adding `post` as a second
/// standing delivery arm beside the ordinary doorbell. The empty descriptor
/// stays inside x86-64's caller-saved-register carrier.
pub async fn next_wake_on(post: ParkPost) -> WakeReason {
    AlternateWake::new(post, None).await
}

/// Deadline-bounded sibling of [`next_wake_on`]. Either standing post or the
/// relative deadline wakes the root, with both registrations settled on the
/// ready leg or cancellation drop.
pub async fn next_wake_on_for(post: ParkPost, delta_ns: u64) -> WakeReason {
    AlternateWake::new(post, Some(delta_ns)).await
}

/// One alternate executor registration, owned by the future that installed
/// it.  The ready leg and `Drop` share the same idempotent teardown, so a
/// `select`-style caller cannot leave a stale post (or folded deadline) behind
/// when another branch cancels this wait while it is pending.
struct AlternateWake {
    post: ParkPost,
    delta_ns: Option<u64>,
    deadline_ns: Option<u64>,
    owner: Option<AlternateOwner>,
    timer: TimerRegistration,
    _pin: PhantomPinned,
}

#[derive(Clone, Copy)]
struct AlternateOwner {
    task: NonNull<TaskHeader>,
    ctx: NonNull<ExecCtx>,
    delivery_generation: Option<u64>,
}

impl AlternateWake {
    const fn new(post: ParkPost, delta_ns: Option<u64>) -> Self {
        Self {
            post,
            delta_ns,
            deadline_ns: None,
            owner: None,
            timer: TimerRegistration::new(),
            _pin: PhantomPinned,
        }
    }

    fn clear_registration(&mut self) {
        self.timer.clear_registration();
        let Some(owner) = self.owner.take() else {
            return;
        };
        // SAFETY: `owner` is installed only while this future is being polled
        // by the owning executor. A pending future is cancelled by dropping it
        // from that same task's poll/drop path, before the executor context can
        // leave `run_pinned`; target panics abort rather than unwind. Normal
        // executor exit also clears the complete registry as a backstop.
        let ctx = unsafe { owner.ctx.as_ref() };
        if let Some(generation) = owner.delivery_generation {
            ctx.clear_delivery(owner.task, generation);
        }
    }
}

impl Future for AlternateWake {
    type Output = WakeReason;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: no field is moved. Pinning makes `owner`'s address stable
        // while the executor registry holds its teardown backlink.
        let this = unsafe { self.get_unchecked_mut() };
        let hdr = header_of(cx.waker())
            .expect("alternate wake requires the kore executor (foreign waker)");
        // SAFETY: the kore waker vouches for a live executor-owned header.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "alternate wake polled outside an executor pass"
        );
        // SAFETY: the executor installs this pointer for the duration of poll.
        let ctx = unsafe { &*ctx };

        if let Some(owner) = this.owner {
            assert_eq!(owner.task, hdr, "alternate wake changed owning task");
            assert_eq!(owner.ctx.as_ptr(), ctx as *const ExecCtx as *mut ExecCtx);
            let reason = ctx.last_wake();
            this.clear_registration();
            return Poll::Ready(reason);
        }

        if let Some(delta_ns) = this.delta_ns {
            let deadline = *this
                .deadline_ns
                .get_or_insert_with(|| ctx.now().saturating_add(delta_ns));
            if ctx.now() >= deadline {
                return Poll::Ready(WakeReason::Deadline);
            }
        }

        // Publish the owner first so either registration panicking still
        // unwinds through the same idempotent cleanup.
        this.owner = Some(AlternateOwner {
            task: hdr,
            ctx: NonNull::from(ctx),
            delivery_generation: None,
        });
        let future_owner = NonNull::from(&mut this.owner);
        let generation = ctx.register_delivery(this.post, hdr, future_owner);
        let owner = this
            .owner
            .as_mut()
            .expect("alternate owner disappeared during registration");
        owner.delivery_generation = Some(generation);
        if let Some(deadline) = this.deadline_ns {
            // SAFETY: this field stays pinned with AlternateWake and is
            // cleared before that parent's Drop can release its storage.
            unsafe { Pin::new_unchecked(&mut this.timer) }.arm(ctx, hdr, deadline);
        }
        Poll::Pending
    }
}

impl Drop for AlternateWake {
    fn drop(&mut self) {
        self.clear_registration();
    }
}

/// Complete on the next executor wake, but ensure one occurs within
/// `delta_ns` by folding a relative deadline into the same idle park.
///
/// This is the resident-maintenance twin of [`next_wake`]: a service drains
/// every request, performs one bounded maintenance pass, then waits for either
/// its doorbell or the next maintenance deadline. A doorbell win cancels this
/// future's timer before returning, so repeated traffic cannot leave stale
/// deadlines that later wake an unrelated await.
///
/// # Panics
/// As [`sleep_for`]: the future must be polled by the kore executor.
pub async fn next_wake_for(delta_ns: u64) -> WakeReason {
    let mut deadline = None;
    let mut parked = false;
    let mut timer = core::pin::pin!(TimerRegistration::new());
    poll_fn(move |cx| {
        let hdr = header_of(cx.waker())
            .expect("next_wake_for requires the kore executor (foreign waker)");
        // SAFETY: `header_of` vouches the header came from this executor's
        // vtable; headers outlive their polls.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "next_wake_for polled outside an executor pass"
        );
        // SAFETY: the polling executor set `exec` to its own live context for
        // the duration of this pass.
        let ctx = unsafe { &*ctx };
        if parked {
            timer.as_mut().clear();
            return Poll::Ready(ctx.last_wake());
        }
        let deadline = *deadline.get_or_insert_with(|| ctx.now().saturating_add(delta_ns));
        if ctx.now() >= deadline {
            return Poll::Ready(WakeReason::Deadline);
        }
        timer.as_mut().arm(ctx, hdr, deadline);
        parked = true;
        Poll::Pending
    })
    .await
}

/// Complete on the next executor wake, with an **absolute monotonic** upper
/// bound.  A request kick can therefore re-run a resident service before its
/// current earliest alarm, while a quiet service wakes exactly when that alarm
/// becomes due.  The ready leg always clears the registration, so re-arming a
/// new minimum cannot leave the old deadline behind.
pub async fn next_wake_until(deadline_ns: u64) -> WakeReason {
    let mut parked = false;
    let mut timer = core::pin::pin!(TimerRegistration::new());
    poll_fn(move |cx| {
        let hdr = header_of(cx.waker())
            .expect("next_wake_until requires the kore executor (foreign waker)");
        // SAFETY: the kore waker vouches for a live executor-owned header.
        let ctx = unsafe { hdr.as_ref() }.exec.load(Ordering::Relaxed);
        assert!(
            !ctx.is_null(),
            "next_wake_until polled outside an executor pass"
        );
        // SAFETY: the executor installs this pointer for the duration of poll.
        let ctx = unsafe { &*ctx };
        if parked {
            timer.as_mut().clear();
            return Poll::Ready(ctx.last_wake());
        }
        if ctx.now() >= deadline_ns {
            return Poll::Ready(WakeReason::Deadline);
        }
        timer.as_mut().arm(ctx, hdr, deadline_ns);
        parked = true;
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::super::task::waker_for;
    use super::*;
    use kore_abi::session::SlotRef;

    fn task_waker(ctx: &ExecCtx, task: &TaskHeader) -> core::task::Waker {
        task.exec
            .store(ctx as *const ExecCtx as *mut ExecCtx, Ordering::Relaxed);
        waker_for(NonNull::from(task))
    }

    #[test]
    fn two_sleeps_in_one_task_keep_both_deadlines_in_either_poll_order() {
        for reverse in [false, true] {
            let ctx = ExecCtx::new();
            let task = TaskHeader::new();
            let waker = task_waker(&ctx, &task);
            let mut cx = Context::from_waker(&waker);
            let mut early = core::pin::pin!(sleep_until(17));
            let mut late = core::pin::pin!(sleep_until(29));
            if reverse {
                assert_eq!(late.as_mut().poll(&mut cx), Poll::Pending);
                assert_eq!(early.as_mut().poll(&mut cx), Poll::Pending);
            } else {
                assert_eq!(early.as_mut().poll(&mut cx), Poll::Pending);
                assert_eq!(late.as_mut().poll(&mut cx), Poll::Pending);
            }
            assert_eq!(ctx.timers_min(), Some(17));
            ctx.advance(16);
            assert!(!task.take_ready());
            ctx.advance(17);
            assert!(task.take_ready());
            assert_eq!(ctx.timers_min(), Some(29));
            // Re-polling the remaining sleep re-anchors its own entry; it
            // neither acquires a second slot nor restores the expired one.
            assert_eq!(late.as_mut().poll(&mut cx), Poll::Pending);
            assert_eq!(early.as_mut().poll(&mut cx), Poll::Ready(()));
            assert_eq!(
                ctx.timers
                    .iter()
                    .filter(|slot| slot.get().is_some())
                    .count(),
                1
            );
            ctx.advance(29);
            assert!(task.take_ready());
            assert_eq!(late.as_mut().poll(&mut cx), Poll::Ready(()));
            assert_eq!(ctx.timers_min(), None);
        }
    }

    #[test]
    fn cancelling_either_deadline_preserves_the_other_wait() {
        for cancelled_deadline in [17, 59] {
            let ctx = ExecCtx::new();
            let task = TaskHeader::new();
            let waker = task_waker(&ctx, &task);
            let mut cx = Context::from_waker(&waker);
            let mut retained = core::pin::pin!(sleep_until(41));
            assert_eq!(retained.as_mut().poll(&mut cx), Poll::Pending);
            {
                let mut cancelled = core::pin::pin!(sleep_until(cancelled_deadline));
                assert_eq!(cancelled.as_mut().poll(&mut cx), Poll::Pending);
            }
            assert_eq!(ctx.timers_min(), Some(41));
            ctx.advance(40);
            assert!(!task.take_ready());
            ctx.advance(41);
            assert!(task.take_ready());
            assert_eq!(retained.as_mut().poll(&mut cx), Poll::Ready(()));
        }
    }

    #[test]
    fn expired_wait_cleanup_cannot_remove_a_reused_slot() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        let waker = task_waker(&ctx, &task);
        let mut cx = Context::from_waker(&waker);
        let mut replacement = core::pin::pin!(sleep_until(29));
        {
            let mut expired = core::pin::pin!(sleep_until(17));
            assert_eq!(expired.as_mut().poll(&mut cx), Poll::Pending);
            ctx.advance(17);
            assert!(task.take_ready());
            assert_eq!(replacement.as_mut().poll(&mut cx), Poll::Pending);
            // Drop the expired but not re-polled future after its table slot
            // has been reused by another wait in the same task.
        }
        assert_eq!(ctx.timers_min(), Some(29));
        ctx.advance(29);
        assert_eq!(replacement.as_mut().poll(&mut cx), Poll::Ready(()));
    }

    #[test]
    fn a_deadline_wakes_only_its_owning_task() {
        let ctx = ExecCtx::new();
        let first = TaskHeader::new();
        let second = TaskHeader::new();
        let first_waker = task_waker(&ctx, &first);
        let second_waker = task_waker(&ctx, &second);
        let mut first_cx = Context::from_waker(&first_waker);
        let mut second_cx = Context::from_waker(&second_waker);
        let mut early = core::pin::pin!(sleep_for(17));
        let mut late = core::pin::pin!(sleep_for(29));
        assert_eq!(early.as_mut().poll(&mut first_cx), Poll::Pending);
        assert_eq!(late.as_mut().poll(&mut second_cx), Poll::Pending);
        ctx.advance(17);
        assert!(first.take_ready());
        assert!(!second.take_ready());
        assert_eq!(early.as_mut().poll(&mut first_cx), Poll::Ready(()));
        ctx.advance(29);
        assert!(!first.take_ready());
        assert!(second.take_ready());
        assert_eq!(late.as_mut().poll(&mut second_cx), Poll::Ready(()));
    }

    #[test]
    fn alternate_and_ordinary_waits_do_not_cancel_each_other() {
        for reverse in [false, true] {
            let ctx = ExecCtx::new();
            let task = TaskHeader::new();
            let waker = task_waker(&ctx, &task);
            let mut cx = Context::from_waker(&waker);
            let post = ParkPost {
                chan: 17,
                land: SlotRef::from_wire(19),
                recv_tag: 23,
            };
            let mut ordinary = core::pin::pin!(next_wake_until(41));
            {
                let mut alternate = core::pin::pin!(AlternateWake::new(post, Some(29)));
                if reverse {
                    assert_eq!(ordinary.as_mut().poll(&mut cx), Poll::Pending);
                    assert_eq!(alternate.as_mut().poll(&mut cx), Poll::Pending);
                } else {
                    assert_eq!(alternate.as_mut().poll(&mut cx), Poll::Pending);
                    assert_eq!(ordinary.as_mut().poll(&mut cx), Poll::Pending);
                }
                assert_eq!(ctx.timers_min(), Some(29));
                assert_eq!(ctx.delivery_post(), Some(post));
                // A completion on the alternate post must remove only the
                // alternate deadline, regardless of child polling order.
                ctx.note_wake(WakeReason::Delivery);
                assert_eq!(
                    alternate.as_mut().poll(&mut cx),
                    Poll::Ready(WakeReason::Delivery)
                );
            }
            assert_eq!(ctx.delivery_post(), None);
            assert_eq!(ctx.timers_min(), Some(41));
            ctx.advance(41);
            ctx.note_wake(WakeReason::Deadline);
            assert_eq!(
                ordinary.as_mut().poll(&mut cx),
                Poll::Ready(WakeReason::Deadline)
            );

            // Conversely, a cancelled ordinary wait cannot remove the
            // alternate post's later deadline.
            let mut alternate = core::pin::pin!(AlternateWake::new(post, Some(59)));
            assert_eq!(alternate.as_mut().poll(&mut cx), Poll::Pending);
            {
                let mut cancelled = core::pin::pin!(next_wake_for(17));
                assert_eq!(cancelled.as_mut().poll(&mut cx), Poll::Pending);
            }
            assert_eq!(ctx.timers_min(), Some(100));
            assert_eq!(ctx.delivery_post(), Some(post));
        }
    }

    #[test]
    fn timer_teardown_allows_future_drop_after_the_context_is_gone() {
        let mut sleep = core::pin::pin!(sleep_until(17));
        let mut wake = core::pin::pin!(next_wake_for(29));
        {
            let ctx = ExecCtx::new();
            let task = TaskHeader::new();
            let waker = task_waker(&ctx, &task);
            let mut cx = Context::from_waker(&waker);
            assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
            assert_eq!(wake.as_mut().poll(&mut cx), Poll::Pending);
            // The same cleanup Executor::run_pinned and Executor::drop use.
            ctx.clear_timers();
            assert_eq!(ctx.timers_min(), None);
        }
        // Both pinned futures now drop after their context and header. Their
        // invalidated registrations must not dereference either dead object.
    }

    #[test]
    fn cancelling_an_alternate_wait_releases_its_post_and_timer() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        task.exec
            .store(&ctx as *const ExecCtx as *mut ExecCtx, Ordering::Relaxed);
        let task = NonNull::from(&task);
        let waker = waker_for(task);
        let mut cx = Context::from_waker(&waker);
        let post = ParkPost {
            chan: 17,
            land: SlotRef::from_wire(19),
            recv_tag: 23,
        };

        {
            let wait = AlternateWake::new(post, Some(29));
            let mut wait = core::pin::pin!(wait);
            assert_eq!(wait.as_mut().poll(&mut cx), Poll::Pending);
            assert_eq!(ctx.delivery_post(), Some(post));
            assert_eq!(ctx.timers_min(), Some(29));
            // Dropping the still-pending future is the cancellation path.
        }

        assert_eq!(ctx.delivery_post(), None);
        assert_eq!(ctx.timers_min(), None);

        // The released bounded slot can be claimed immediately by another
        // wait instead of poisoning the executor's next idle park.
        {
            let replacement = AlternateWake::new(post, None);
            let mut replacement = core::pin::pin!(replacement);
            assert_eq!(replacement.as_mut().poll(&mut cx), Poll::Pending);
        }
        assert_eq!(ctx.delivery_post(), None);
    }

    #[test]
    fn stale_alternate_cleanup_cannot_clear_its_successor() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        let task = NonNull::from(&task);
        let post = ParkPost {
            chan: 31,
            land: SlotRef::from_wire(37),
            recv_tag: 41,
        };

        let mut stale_owner = None;
        let stale = ctx.register_delivery(post, task, NonNull::from(&mut stale_owner));
        ctx.clear_delivery(task, stale);
        let mut current_owner = None;
        let current = ctx.register_delivery(post, task, NonNull::from(&mut current_owner));
        assert_ne!(stale, current);

        ctx.clear_delivery(task, stale);
        assert_eq!(ctx.delivery_post(), Some(post));
        ctx.clear_delivery(task, current);
        assert_eq!(ctx.delivery_post(), None);
    }

    #[test]
    fn executor_teardown_invalidates_a_pending_futures_backlink() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        task.exec
            .store(&ctx as *const ExecCtx as *mut ExecCtx, Ordering::Relaxed);
        let task = NonNull::from(&task);
        let waker = waker_for(task);
        let mut cx = Context::from_waker(&waker);
        let post = ParkPost {
            chan: 43,
            land: SlotRef::from_wire(47),
            recv_tag: 53,
        };
        let wait = AlternateWake::new(post, Some(59));
        let mut wait = core::pin::pin!(wait);

        assert_eq!(wait.as_mut().poll(&mut cx), Poll::Pending);
        assert!(wait.as_ref().get_ref().owner.is_some());
        ctx.clear_timers();
        assert!(wait.as_ref().get_ref().owner.is_none());
        assert!(wait.as_ref().get_ref().timer.ctx.get().is_none());
        assert_eq!(ctx.delivery_post(), None);
        assert_eq!(ctx.timers_min(), None);
        // `wait` now drops without following the invalidated context backlink.
    }

    #[test]
    fn an_early_wake_cancels_its_folded_deadline() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        let waker = task_waker(&ctx, &task);
        let mut cx = Context::from_waker(&waker);
        let mut wait = core::pin::pin!(next_wake_for(17));
        assert_eq!(wait.as_mut().poll(&mut cx), Poll::Pending);
        assert_eq!(ctx.timers_min(), Some(17));
        ctx.note_wake(WakeReason::Doorbell);
        assert_eq!(
            wait.as_mut().poll(&mut cx),
            Poll::Ready(WakeReason::Doorbell)
        );
        assert_eq!(ctx.timers_min(), None);
        ctx.advance(17);
        assert!(!task.take_ready());
        assert_eq!(ctx.timers_min(), None);
    }

    #[test]
    fn inline_doorbell_is_a_one_shot_total_drain_edge() {
        let ctx = ExecCtx::new();
        ctx.note_inline_doorbell(17);
        assert_eq!(ctx.last_wake(), WakeReason::Doorbell);
        assert_eq!(ctx.now(), 17_000_000);
        assert!(ctx.take_inline_doorbell());
        assert!(!ctx.take_inline_doorbell());

        ctx.note_inline_doorbell(11);
        assert_eq!(ctx.now(), 17_000_000, "the kernel clock anchor is monotone");
        ctx.clear_timers();
        assert!(!ctx.take_inline_doorbell());
    }

    #[test]
    fn an_absolute_service_deadline_replaces_cleanly() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        let task = NonNull::from(&task);

        let mut registration = core::pin::pin!(TimerRegistration::new());
        registration.as_mut().arm(&ctx, task, 90);
        registration.as_mut().arm(&ctx, task, 40);
        assert_eq!(ctx.timers_min(), Some(40));
        assert_eq!(
            ctx.timers
                .iter()
                .filter(|slot| slot.get().is_some())
                .count(),
            1
        );
        registration.as_mut().clear();
        assert_eq!(ctx.timers_min(), None);
    }

    #[test]
    fn resident_irq_park_policy_is_executor_owned() {
        let ctx = ExecCtx::new();
        let task = TaskHeader::new();
        task.exec
            .store(&ctx as *const ExecCtx as *mut ExecCtx, Ordering::Relaxed);
        let task = NonNull::from(&task);
        let waker = waker_for(task);
        let mut cx = Context::from_waker(&waker);

        assert!(!ctx.irq());
        let arm = enable_irq_parks();
        let mut arm = core::pin::pin!(arm);
        assert_eq!(arm.as_mut().poll(&mut cx), Poll::Ready(()));
        assert!(ctx.irq());
        // Timer/doorbell cleanup does not silently disarm a resident device.
        ctx.clear_timers();
        assert!(ctx.irq());

        let retire = retire_irq_parks();
        let mut retire = core::pin::pin!(retire);
        assert_eq!(retire.as_mut().poll(&mut cx), Poll::Ready(()));
        assert!(!ctx.irq());
        ctx.clear_timers();
        assert!(!ctx.irq());

        // Terminal containment may be observed more than once; retirement is
        // idempotent even though resurrection is forbidden.
        let retire_again = retire_irq_parks();
        let mut retire_again = core::pin::pin!(retire_again);
        assert_eq!(retire_again.as_mut().poll(&mut cx), Poll::Ready(()));
        assert!(!ctx.irq());
    }

    #[test]
    #[should_panic(expected = "retired IRQ park policy cannot be re-armed")]
    fn retired_irq_policy_cannot_be_rearmed() {
        let ctx = ExecCtx::new();
        ctx.retire_irq();
        ctx.enable_irq();
    }
}
