//! The **bridge** — a synchronous engine on a worker thread, served by an
//! async host.
//!
//! Some engines cannot await. Lua's `require` must hand a module back to the
//! C frame that called it, from inside a coroutine, a metamethod or a
//! `pcall`; TinyCC opens a file it discovered mid-parse; an independent C
//! program's `main` calls the filesystem whenever it likes. What those calls
//! need is reached only through the task's async clients, on its executor. So
//! the engine runs on a kore-rt **worker thread**, the executor stays on the
//! thread that founded it (the **host**: thread 0 in every engine today), and
//! the two share one [`Bridge`] — one request cell and one word:
//!
//! ```text
//!  host (executor)                              worker (engine)
//!  ───────────────                              ───────────────
//!  spawn: first item, WORK; found ────────────▶ Engine::run(work) …
//!  park on the wake channel                      … call(): request, CALL
//!  ◀───────────────── CALL, kick_parent ──────── futex::wait on the word
//!  serve the call (async)
//!  the answer, WORK; kick ─────────────────────▶ read the answer; resume
//!  park on the wake channel                      … finish(): results, IDLE
//!  ◀───────────────── IDLE, kick_parent ──────── next(): futex::wait
//!  post(): next item, WORK; kick ──────────────▶ …
//!  quit(): QUIT; kick ─────────────────────────▶ run returns
//!  ◀───────────── GONE·Quit, kick_parent ─────── the thread ends; join
//! ```
//!
//! This module owns everything about that exchange except what crosses it.
//! An engine supplies its payload — the work, its results, and each call's
//! request and answer, as one flat structure — and the dispatch that answers a
//! call. The states, the ordering, the hand-over of the cell, and what happens
//! when the worker is gone are here, once. Lua's REPL and tool worker, TinyCC
//! and `libkore.so`'s C program host are the same exchange with different
//! payloads; before this module each of them carried its own copy of the word
//! and its discipline, and the copies had drifted apart in exactly the places
//! the discipline is subtle.
//!
//! # The word
//!
//! | Word        | Holds the cell | Entered by                                     |
//! |-------------|----------------|------------------------------------------------|
//! | `VACANT`    | nobody         | [`Bridge::new`]; a `WorkerHost::join`  |
//! | `IDLE`      | host           | the host's claim; the worker's [`Work::finish`] |
//! | `WORK`      | worker         | founding, `WorkerHost::post`, an answer |
//! | `CALL`      | host           | the worker's [`call`](Bridge::call)            |
//! | `QUIT`      | worker         | the host's `WorkerHost::quit`          |
//! | `GONE`·why  | host, for good | the worker, as its last act ([`Gone`])         |
//!
//! **The word names the cell's holder**, and only the holder touches the
//! payload. Every hand-over is a `Release` on the word after the holder's last
//! write, matched by the other side's `Acquire` before its first read, so
//! whatever the holder wrote is visible to the next one. The payload is reached
//! only through closures this module runs after it has read a state that hands
//! the cell to the caller, so no borrow outlives the state that justified it.
//!
//! **Founding is the hand-off.** `WorkerHost::spawn` writes the first work
//! item while it holds the claimed bridge, stores `WORK`, and only then founds
//! the worker: the worker is born holding its work, with nothing to wait for.
//!
//! **Two transitions are made by the side that does not hold the cell**, and
//! both touch only the word: a worker that is gone moves a host-held `IDLE` or
//! `CALL` to `GONE`. Each races one transition of the host's — a `post` out of
//! `IDLE`, an answer out of `CALL` — and both sides compare-and-swap, so
//! exactly one wins and neither overwrites the other. Every other transition
//! is made by the holder alone and could be a plain store; the races are the
//! whole reason the host's own transitions are CASes.
//!
//! # Parks, kicks, and why no wakeup is lost
//!
//! Every park is a doorbell park; nothing here spins beyond `futex::wait`'s
//! own bounded monitor.
//!
//! - **The worker** parks with [`futex::wait`] on the word, on its own
//!   doorbell. The host answers by storing the word and then kicking that
//!   doorbell (`DoorbellWorker::kick`). `futex::wait` stands its post
//!   *before* it re-reads the word, so a kick that found no post standing was
//!   refused before the worker looked — and the store it announced, sequenced
//!   before the kick, is what the worker then reads.
//! - **The host** parks on the **wake channel** the worker holds a send into
//!   (`DoorbellWorker::wake_recv`): a launched task has no self-kick for a
//!   worker to inherit, so [`futex::wake`] could not reach thread 0. The
//!   worker stores the word and then kicks the channel
//!   (`thread::kick_parent`); the host stands its post, re-reads the word,
//!   and only then parks with
//!   [`next_wake_on`](crate::executor::next_wake_on). A kick that found no
//!   post was refused before the post stood, so the re-read sees its store; a
//!   kick into a standing post resolves the park at once.
//!
//! - **A side source** (`Side`, metal only) is work other
//!   threads hand the host through the same wake channel: a C program's
//!   output, which any of its threads may write. Its producers kick only
//!   while the host is armed, and the host arms after its post stands and
//!   then re-reads the source beside the word, so the same argument holds:
//!   a kick that found no post was refused before the arm, and the re-read
//!   sees what the producer published first. The host drains the source a
//!   bounded pass at a time, between the worker's calls and beside one in
//!   flight, so neither waits for the other.
//!
//! A refused kick is therefore never an error on either side, and a wake is
//! never a promise: both sides re-read the word after every return.
//!
//! # What a call may borrow
//!
//! While the word reads `CALL`, the worker is parked inside
//! [`Bridge::call`], and `call` never returns while it does: an answer moves
//! the word to `WORK` first, and a worker that gives up a call **exits its
//! thread instead of returning**. An exited thread's stack stays mapped until
//! it is joined, and the host joins only once the word reads `GONE` — after it
//! has finished with the call, since the give-up is what it then reads. So a
//! request may borrow from the frame that made it (a path, the bytes of a
//! write) and the host may read through those borrows for the whole call.
//!
//! # A worker that is gone
//!
//! `GONE` is the worker's last word on the bridge, and it always says it,
//! whichever way it leaves ([`Gone`]): its body returns; it answers a quit; it
//! fails a work item; its probe at entry finds it cannot park, or a park is
//! refused mid-call or between items; it ends its thread itself; or it panics.
//! The last two need no help from the engine. The worker's runtime block names
//! its bridge from its first instruction, and the worker's entry registers the
//! thread's exit hook (`thread::on_exit`), which `thread::exit` runs before
//! it commits: a worker that ends its thread without having said its last
//! word has it said for it. Every kore-rt panic path — `entry::panic_refuse`,
//! `crash::panic`, `crash::report` — publishes `GONE`·[`Panicked`](Gone::Panicked)
//! and kicks the host *after* the panic record is published, so a host that
//! ends the process on it cannot cut the report short; the exit hook names a
//! panic too, for an exit taken while one is in flight. That slot is also how a
//! worker-side call proves its caller is the worker: holdings are per thread,
//! and so is the right to touch a worker's cell.
//!
//! # What an engine keeps
//!
//! Its payload — `Send`, because it crosses threads; an engine whose payload
//! carries raw pointers says why they may cross in its own `unsafe impl Send` —
//! one `static` [`Bridge`] over it, an [`Engine`] naming the two and the
//! worker's body, and the async dispatch it hands `WorkerHost::serve`. A new
//! verb is a field and an arm in the engine; never a new state, and never
//! another copy of the ordering.
//!
//! The host suite drives this state machine with real threads (and under
//! Miri, which checks every hand-over for a data race) through a park that
//! only ever returns spuriously — what `futex::wait` may always do. It proves
//! the states, the races and the hand-over; it does not, and cannot, prove the
//! kernel's half of the park discipline above, which the boot gates and the
//! engines' keytests exercise where it runs.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::barrier::Line;
use crate::futex::{self, WaitOutcome};

// ─── The word ──────────────────────────────────────────────────────────

// The phase is the word's low byte; `GONE` carries its reason above it.
const VACANT: u32 = 0;
const IDLE: u32 = 1;
const WORK: u32 = 2;
const CALL: u32 = 3;
const QUIT: u32 = 4;
const GONE: u32 = 5;
const PHASE: u32 = 0xff;
const REASON_SHIFT: u32 = 8;

/// A value the word never holds — the park probe's expectation, so the probe
/// can never actually park (`futex::wait` answers `Retry` on a thread that can
/// park, `Unsupported` on one that cannot).
#[cfg(any(test, target_os = "none"))]
const NEVER: u32 = u32::MAX;

/// Why a worker is gone: the reason its last word on the bridge carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gone {
    /// The worker's body returned, or the worker ended itself: through
    /// [`Bridge::leave`], or by ending its thread (`thread::exit`, whose exit
    /// hook says it). Results it finished before are in the payload.
    Ended,
    /// The worker answered the host's `WorkerHost::quit`.
    Quit,
    /// The worker could not do its work item and takes no other
    /// ([`Work::fail`]); the payload says why.
    Failed,
    /// The worker cannot park on its doorbell — found by the probe at entry,
    /// or by a park the kernel refused later — so it could wait for an answer
    /// only by spinning, which this bridge refuses to do.
    Unparked,
    /// The worker panicked; its panic record was published first.
    Panicked,
}

impl Gone {
    const fn word(self) -> u32 {
        let reason = match self {
            Self::Ended => 1,
            Self::Quit => 2,
            Self::Failed => 3,
            Self::Unparked => 4,
            Self::Panicked => 5,
        };
        GONE | reason << REASON_SHIFT
    }

    const fn from_reason(reason: u32) -> Self {
        match reason {
            1 => Self::Ended,
            2 => Self::Quit,
            3 => Self::Failed,
            4 => Self::Unparked,
            // 5, and any reason this module never writes: a word that cannot
            // be read is a worker that cannot be trusted, which the host must
            // treat as gone — failing closed, never as running.
            _ => Self::Panicked,
        }
    }
}

/// The word, read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Vacant,
    Idle,
    Work,
    Call,
    Quit,
    Gone(Gone),
}

const fn decode(word: u32) -> State {
    match word & PHASE {
        VACANT => State::Vacant,
        IDLE => State::Idle,
        WORK => State::Work,
        CALL => State::Call,
        QUIT => State::Quit,
        // `GONE`, and any phase this module never writes: gone, as above.
        _ => State::Gone(Gone::from_reason(word >> REASON_SHIFT)),
    }
}

/// Publish `reason` as a worker's last word, from wherever the word stands,
/// without touching the payload. `true` when this call published it; `false`
/// when the worker was already gone (or no host holds the bridge).
///
/// From `WORK` and `QUIT` the worker holds the cell and could store; from a
/// host-held `IDLE` or `CALL` it must not, because the host's `post` and
/// answer move the word too. One CAS loop serves all four, so exactly one of a
/// racing pair wins and the loser sees the winner's word. A worker that ends
/// itself (`Ended`) while asked to quit has answered the quit.
fn settle(word: &AtomicU32, reason: Gone) -> bool {
    let mut seen = word.load(Ordering::Acquire);
    loop {
        let gone = match decode(seen) {
            State::Quit if reason == Gone::Ended => Gone::Quit.word(),
            State::Idle | State::Work | State::Call | State::Quit => reason.word(),
            State::Gone(_) | State::Vacant => return false,
        };
        match word.compare_exchange_weak(seen, gone, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => seen = now,
        }
    }
}

// ─── The worker's ends ─────────────────────────────────────────────────

/// What the worker side needs from the machine: who it is, how it parks, how
/// it tells its host, and how it ends without returning.
///
/// [`Native`] is the only implementation outside this module's tests; the
/// split exists so the state machine above it is ordinary code the host suite
/// can drive, the shape `futex` and `barrier` take with their own metal halves.
trait Ends {
    /// Name this thread the worker over `word`, before anything else runs on
    /// it. `false` when the thread cannot hold the name (no runtime block).
    #[cfg(any(test, target_os = "none"))]
    fn adopt(&self, word: &AtomicU32) -> bool;
    /// This thread is the worker over `word`.
    fn is_worker(&self, word: &AtomicU32) -> bool;
    /// This thread is no longer a worker (its last word is published).
    fn release(&self, word: &AtomicU32);
    /// Park while `word` holds `seen`; `futex::wait`'s contract, spurious
    /// returns included.
    fn park(&self, word: &AtomicU32, seen: u32) -> WaitOutcome;
    /// Tell the host the word moved. A refused kick is harmless (module doc).
    fn kick_host(&self);
    /// End this thread without returning; its stack stays mapped until it is
    /// joined.
    fn exit(&self) -> !;
}

/// The worker's real ends: its runtime block, its doorbell, its wake send.
struct Native;

impl Ends for Native {
    #[cfg(any(test, target_os = "none"))]
    fn adopt(&self, word: &AtomicU32) -> bool {
        imp::adopt(word_address(word))
    }

    fn is_worker(&self, word: &AtomicU32) -> bool {
        imp::hosted() == word_address(word)
    }

    fn release(&self, word: &AtomicU32) {
        imp::release(word_address(word));
    }

    fn park(&self, word: &AtomicU32, seen: u32) -> WaitOutcome {
        futex::wait(word, seen, None)
    }

    fn kick_host(&self) {
        let _ = imp::kick_host();
    }

    fn exit(&self) -> ! {
        crate::thread::exit(crate::exit::ExitCode::INTERNAL)
    }
}

fn word_address(word: &AtomicU32) -> usize {
    word as *const AtomicU32 as usize
}

// ─── The bridge ────────────────────────────────────────────────────────

/// The request cell a worker and its host share: one word and one payload.
///
/// A `static`, named by its [`Engine`] — the worker's calls come from deep
/// inside an engine (a C callback, a stdio sink) that has no context to pass
/// down, and the cell must outlive a worker a host abandons. Its methods are
/// the **worker's** side; `WorkerHost` is the host's.
pub struct Bridge<P> {
    /// The word, alone on its destructive-interference span: the worker
    /// monitors this line in `futex::wait`'s brief spin, and a store to the
    /// payload beside it would end that monitor for nothing.
    word: Line,
    /// A worker-side payload borrow is live — one of this bridge's closures is
    /// running on the worker. Only the worker touches it.
    borrowed: AtomicBool,
    payload: UnsafeCell<P>,
}

// SAFETY: the payload is touched only by the side the word hands it to, and
// every hand-over is a `Release` on the word matched by the other side's
// `Acquire` (module doc). It crosses threads, hence `P: Send`.
unsafe impl<P: Send> Sync for Bridge<P> {}

/// Why a worker-side operation was refused. Nothing was posted or changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The caller is not this bridge's worker thread.
    NotWorker,
    /// The worker holds no work item: it is between items, asked to quit, or
    /// gone.
    NoWork,
    /// The payload is already borrowed on this thread: the call came from
    /// inside another of this bridge's closures, which would alias it.
    Nested,
}

impl<P> Bridge<P> {
    /// A vacant bridge over `payload`, ready to be a `static`.
    pub const fn new(payload: P) -> Self {
        Self {
            word: Line::new(VACANT),
            borrowed: AtomicBool::new(false),
            payload: UnsafeCell::new(payload),
        }
    }

    /// A host holds this bridge: from `WorkerHost::spawn`'s claim until its
    /// `WorkerHost::join`, including while the worker is gone.
    pub fn is_hosted(&self) -> bool {
        decode(self.word.0.load(Ordering::Acquire)) != State::Vacant
    }

    fn word(&self) -> &AtomicU32 {
        &self.word.0
    }

    fn state(&self) -> State {
        decode(self.word().load(Ordering::Acquire))
    }

    /// Run `f` over the payload for the worker, refusing a nested borrow.
    ///
    /// Caller: the worker, having read (`Acquire`) a state that hands it the
    /// cell, with no store since that hands it back.
    fn worker_borrow<R>(&self, f: impl FnOnce(&mut P) -> R) -> Result<R, Refused> {
        if self.borrowed.swap(true, Ordering::Relaxed) {
            return Err(Refused::Nested);
        }
        // SAFETY: the caller holds the cell (contract above), and `borrowed`
        // proves no other borrow of it is live on this, the only thread that
        // may hold it now.
        let value = f(unsafe { &mut *self.payload.get() });
        self.borrowed.store(false, Ordering::Relaxed);
        Ok(value)
    }

    /// The payload for the host.
    ///
    /// # Safety
    /// The host has read (`Acquire`) a state that hands it the cell — `IDLE`,
    /// `CALL` or `GONE`, or the claim it just made — and holds no other
    /// reference into the payload.
    #[cfg(any(test, target_os = "none"))]
    #[allow(clippy::mut_from_ref)]
    unsafe fn host_payload(&self) -> &mut P {
        // SAFETY: caller contract. The worker's only moves out of a host-held
        // state touch the word alone (`settle`), never the payload.
        unsafe { &mut *self.payload.get() }
    }

    // ── The worker's side ─────────────────────────────────────────────

    /// **Post a call and park until the host answers it.** `request` writes
    /// the call into the payload; `answer` reads the host's answer back out.
    ///
    /// The worker must be inside a work item. The request may borrow from the
    /// caller's frame: this returns only after the host has answered, and a
    /// worker that can no longer park for the answer gives the call up and
    /// ends its thread rather than return into a frame the host may still be
    /// reading (module doc) — the host then reads `GONE`·[`Unparked`](Gone::Unparked).
    pub fn call<R>(
        &self,
        request: impl FnOnce(&mut P),
        answer: impl FnOnce(&mut P) -> R,
    ) -> Result<R, Refused> {
        self.call_with(&Native, request, answer)
    }

    fn call_with<R>(
        &self,
        ends: &impl Ends,
        request: impl FnOnce(&mut P),
        answer: impl FnOnce(&mut P) -> R,
    ) -> Result<R, Refused> {
        if !ends.is_worker(self.word()) {
            return Err(Refused::NotWorker);
        }
        if self.state() != State::Work {
            return Err(Refused::NoWork);
        }
        self.worker_borrow(request)?;
        // The hand-over, then the kick that brings the host to read it.
        self.word().store(CALL, Ordering::Release);
        ends.kick_host();
        loop {
            match self.state() {
                State::Work => break,
                State::Call => {
                    if ends.park(self.word(), CALL) == WaitOutcome::Unsupported {
                        // The kernel refused a park it accepted at the probe:
                        // not transient, and the answer could now be waited
                        // for only by spinning. Give the call up — out of
                        // `CALL` alone, never out of an answer the host
                        // already made: that CAS fails, and the loop reads
                        // the answer's `WORK`.
                        if self
                            .word()
                            .compare_exchange(
                                CALL,
                                Gone::Unparked.word(),
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            ends.release(self.word());
                            ends.kick_host();
                            ends.exit();
                        }
                    }
                }
                // Nothing else moves the word while this worker is parked in
                // its call. If something did, the cell's holder is unknown:
                // never return into the request's frame with it unknown.
                _ => {
                    ends.release(self.word());
                    ends.kick_host();
                    ends.exit();
                }
            }
        }
        self.worker_borrow(answer)
    }

    /// The payload, for the worker inside a work item — a stdio sink, a
    /// release the engine makes between calls.
    pub fn with_work<R>(&self, f: impl FnOnce(&mut P) -> R) -> Result<R, Refused> {
        self.with_work_as(&Native, f)
    }

    fn with_work_as<R>(&self, ends: &impl Ends, f: impl FnOnce(&mut P) -> R) -> Result<R, Refused> {
        if !ends.is_worker(self.word()) {
            return Err(Refused::NotWorker);
        }
        if self.state() != State::Work {
            return Err(Refused::NoWork);
        }
        self.worker_borrow(f)
    }

    /// **Wait for the host's next word**: a work item, a quit, or — when this
    /// worker can no longer park — the end.
    ///
    /// The worker must have finished its last item ([`Work::finish`]). An item
    /// still unfinished is still the current one, and is handed back as
    /// [`Next::Work`] again.
    pub fn next(&'static self) -> Result<Next<P>, Refused> {
        self.next_with(&Native)
    }

    fn next_with(&'static self, ends: &impl Ends) -> Result<Next<P>, Refused> {
        if !ends.is_worker(self.word()) {
            return Err(Refused::NotWorker);
        }
        loop {
            match self.state() {
                State::Work => return Ok(Next::Work(Work::new(self))),
                State::Quit => return Ok(Next::Quit),
                State::Gone(gone) => return Ok(Next::Gone(gone)),
                State::Idle => {
                    if ends.park(self.word(), IDLE) == WaitOutcome::Unsupported
                        && self.leave_with(ends, Gone::Unparked)
                    {
                        return Ok(Next::Gone(Gone::Unparked));
                    }
                }
                // No call is outstanding between items, and a vacant bridge
                // has no host to wait for.
                State::Call | State::Vacant => return Err(Refused::NoWork),
            }
        }
    }

    /// **End this worker**: publish `GONE`·[`Ended`](Gone::Ended) — or
    /// [`Quit`](Gone::Quit), when the host had asked — and kick the host.
    /// For a worker that ends its thread itself instead of returning from
    /// [`Engine::run`], at a point of its choosing: a C program's main thread
    /// leaves once its exit status is posted, so the host reads both. A
    /// `thread::exit` that comes first says the same through the exit hook,
    /// and the trampoline says it for a body that returns. `false` when the
    /// caller is not the worker or the worker is already gone.
    pub fn leave(&self) -> bool {
        self.leave_with(&Native, Gone::Ended)
    }

    fn leave_with(&self, ends: &impl Ends, reason: Gone) -> bool {
        if !ends.is_worker(self.word()) {
            return false;
        }
        let published = settle(self.word(), reason);
        if published {
            ends.release(self.word());
            ends.kick_host();
        }
        published
    }
}

/// The host's side: the decisions `WorkerHost` acts on (and the tests drive).
#[cfg(any(test, target_os = "none"))]
impl<P> Bridge<P> {
    /// Claim a vacant bridge for a new worker: `VACANT` → `IDLE`. One host per
    /// bridge at a time.
    fn claim(&self) -> bool {
        self.word()
            .compare_exchange(VACANT, IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The first item is written: hand it over with the founding.
    fn hand_over_first(&self) {
        self.word().store(WORK, Ordering::Release);
    }

    /// Back to vacant: no worker reads the cell (it was never founded, or it
    /// is joined).
    fn vacate(&self) {
        self.word().store(VACANT, Ordering::Release);
    }

    /// What the host must do next.
    fn host_step(&self) -> HostStep {
        let word = self.word().load(Ordering::Acquire);
        match decode(word) {
            State::Idle => HostStep::Settled(Settled::Idle),
            State::Gone(gone) => HostStep::Settled(Settled::Gone(gone)),
            State::Call => HostStep::Serve,
            // The worker holds the cell; park until the word moves from here.
            State::Work | State::Quit => HostStep::Wait(word),
            // Unreachable while a host exists; never a state to wait in.
            State::Vacant => HostStep::Settled(Settled::Gone(Gone::Ended)),
        }
    }

    /// The call is answered: `CALL` → `WORK`. `false` when the worker gave the
    /// call up first; the answer then has no reader and nothing is kicked.
    fn answer(&self) -> bool {
        self.word()
            .compare_exchange(CALL, WORK, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Post the next item: `work` writes it while the host holds the idle
    /// cell, then `IDLE` → `WORK`.
    fn post_with(&self, work: impl FnOnce(&mut P)) -> Result<(), NotIdle> {
        match self.state() {
            State::Idle => {}
            State::Gone(gone) => return Err(NotIdle::Gone(gone)),
            _ => return Err(NotIdle::Busy),
        }
        // SAFETY: `IDLE` hands the host the cell; a worker leaving it moves
        // the word alone.
        work(unsafe { self.host_payload() });
        match self
            .word()
            .compare_exchange(IDLE, WORK, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(()),
            // The worker left while the item was written; it never saw it.
            Err(now) => Err(match decode(now) {
                State::Gone(gone) => NotIdle::Gone(gone),
                _ => NotIdle::Busy,
            }),
        }
    }

    /// Ask an idle worker to quit: `IDLE` → `QUIT`.
    fn ask_quit(&self) -> QuitAsk {
        match self
            .word()
            .compare_exchange(IDLE, QUIT, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => QuitAsk::Asked,
            Err(now) => match decode(now) {
                // Leaving already, or asked before.
                State::Gone(_) | State::Quit => QuitAsk::Leaving,
                // Mid-item: a worker inside an engine cannot be stopped from
                // here, and would never answer.
                _ => QuitAsk::Busy,
            },
        }
    }
}

/// The host's next move, as the word dictates it.
#[cfg(any(test, target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostStep {
    /// A call waits: serve it.
    Serve,
    /// The worker holds the cell; park until the word leaves this value.
    Wait(u32),
    /// The worker has yielded the cell for now (or for good).
    Settled(Settled),
}

#[cfg(any(test, target_os = "none"))]
enum QuitAsk {
    Asked,
    Leaving,
    Busy,
}

/// What the worker handed back to the host, once a `WorkerHost::serve`
/// has nothing more to answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settled {
    /// The worker finished its item and waits for the next; its results are
    /// in the payload (`WorkerHost::payload`).
    Idle,
    /// The worker is gone, and says why. Results it finished first are in the
    /// payload.
    Gone(Gone),
}

/// Why a `WorkerHost::post` was refused; the item was not handed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotIdle {
    /// The worker holds the cell: an item, a call or a quit is under way.
    Busy,
    /// The worker is gone.
    Gone(Gone),
}

/// The host's wake channel is broken: it can no longer hear the worker, so it
/// cannot serve it. Nothing here is retried; the worker is left to the task's
/// teardown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fatal;

/// How a `WorkerHost` let its worker go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retired {
    /// The worker is gone and its thread joined; the bridge is vacant again.
    Joined,
    /// The worker was left as it was — mid-item, or out of the host's hearing
    /// — for the task's teardown to reclaim. The bridge stays held.
    Abandoned,
}

// ─── The engine ────────────────────────────────────────────────────────

/// A synchronous engine hosted on a worker thread: its payload, its bridge,
/// and the worker's body.
///
/// A unit type per engine (generic, when the engine is: Lua's extension). The
/// worker's entry is monomorphized over it, so the bridge and the body are
/// bound at compile time and no entry pointer is stored anywhere.
pub trait Engine: 'static {
    /// What crosses the bridge: the work, its results, and each call's
    /// request and answer.
    type Payload: Send + 'static;

    /// The one bridge this engine's worker and host share.
    fn bridge() -> &'static Bridge<Self::Payload>;

    /// The worker's body, entered on the worker thread holding the first work
    /// item the host prepared at `WorkerHost::spawn`. Return to end
    /// the worker: the bridge then reads `GONE`·[`Ended`](Gone::Ended), or
    /// [`Quit`](Gone::Quit) after [`Next::Quit`].
    fn run(work: Work<Self::Payload>);
}

/// **A work item, held by the worker** — the proof that the cell is the
/// worker's until it [`finish`](Self::finish)es or [`fail`](Self::fail)s.
///
/// Stays on the worker thread: holdings are per thread, and so is the cell.
#[must_use = "a work item ends with finish or fail; a body that returns holding one ends the worker"]
pub struct Work<P: 'static> {
    bridge: &'static Bridge<P>,
    _thread: PhantomData<*const ()>,
}

impl<P> Work<P> {
    fn new(bridge: &'static Bridge<P>) -> Self {
        Self {
            bridge,
            _thread: PhantomData,
        }
    }

    /// The payload: the item the host wrote, and room for its results.
    ///
    /// Copy out what the engine needs and return: a closure that runs the
    /// engine would make the engine's own [`Bridge::call`]s nested borrows,
    /// which they refuse.
    ///
    /// # Panics
    /// When called from inside one of the bridge's own closures — a nested
    /// borrow of the payload, refused like `RefCell`'s — or after the worker
    /// ended itself ([`Bridge::leave`]) still holding this item: the cell is
    /// the host's then, and so is [`finish`](Self::finish)'s word.
    pub fn with<R>(&mut self, f: impl FnOnce(&mut P) -> R) -> R {
        assert!(
            self.bridge.state() == State::Work,
            "bridge: a work item used after its worker left"
        );
        match self.bridge.worker_borrow(f) {
            Ok(value) => value,
            Err(_) => {
                panic!("bridge: a work item's payload borrowed inside another of its borrows")
            }
        }
    }

    /// Write the item's results and hand the cell back: `WORK` → `IDLE`, and
    /// kick the host. The worker then waits for its next word with
    /// [`Bridge::next`], or returns from its body.
    pub fn finish(mut self, results: impl FnOnce(&mut P)) {
        self.with(results);
        self.bridge.word().store(IDLE, Ordering::Release);
        Native.kick_host();
    }

    /// The worker cannot do this item and will take no other: `why` writes
    /// what the host should know, then `WORK` → `GONE`·[`Failed`](Gone::Failed).
    /// The body should then return.
    pub fn fail(mut self, why: impl FnOnce(&mut P)) {
        self.with(why);
        self.bridge
            .word()
            .store(Gone::Failed.word(), Ordering::Release);
        Native.release(self.bridge.word());
        Native.kick_host();
    }
}

/// The host's next word to a worker, as [`Bridge::next`] reads it.
pub enum Next<P: 'static> {
    /// A work item.
    Work(Work<P>),
    /// The host asks the worker to end: return from [`Engine::run`] (after
    /// releasing whatever the engine holds).
    Quit,
    /// The worker is gone — it could no longer park — and should return.
    Gone(Gone),
}

/// The worker's entry, over `E`: adopt the bridge, probe the park, run the
/// body, settle the word.
#[cfg(any(test, target_os = "none"))]
fn enter<E: Engine>(ends: &impl Ends) {
    let bridge = E::bridge();
    let word = bridge.word();
    // First, so a panic from here on reaches the host.
    if !ends.adopt(word) {
        // A thread with no runtime block cannot park either.
        if settle(word, Gone::Unparked) {
            ends.kick_host();
        }
        return;
    }
    // Founding hands the worker `WORK`; anything else is not this worker's.
    if bridge.state() != State::Work {
        ends.release(word);
        return;
    }
    // Can this thread park? `futex::wait` decides before it compares, so a
    // wait for a value the word never holds answers `Retry` on a thread that
    // can and `Unsupported` on one that cannot — and never parks.
    if ends.park(word, NEVER) == WaitOutcome::Unsupported {
        let _ = bridge.leave_with(ends, Gone::Unparked);
        return;
    }
    E::run(Work::new(bridge));
    let _ = bridge.leave_with(ends, Gone::Ended);
}

/// A panicking thread's last word on its bridge, if it is a worker — called by
/// kore-rt's panic paths once the panic record is published (module doc).
pub(crate) fn panicked() {
    last_word(Gone::Panicked);
}

/// Say `reason` as this thread's last word on its bridge, if it is a worker
/// still on one: for the panic paths and the exit hook, which end a worker
/// that never said its own. A worker that did is released, and this is
/// nothing.
fn last_word(reason: Gone) {
    let address = imp::hosted();
    if address == 0 {
        return;
    }
    // SAFETY: a worker's runtime block names its bridge's word from the
    // worker's first instruction until its last word, and an `Engine`'s
    // bridge is `'static`.
    let word = unsafe { &*(address as *const AtomicU32) };
    if settle(word, reason) {
        imp::release(address);
        let _ = imp::kick_host();
    }
}

// ─── The host ──────────────────────────────────────────────────────────

#[cfg(target_os = "none")]
pub use host::{Side, WorkerHost};

#[cfg(target_os = "none")]
mod host {
    use core::future::poll_fn;
    use core::marker::PhantomData;
    use core::pin::{Pin, pin};
    use core::task::Poll;

    use kore_abi::session::Status;

    use super::{
        Engine, Fatal, HostStep, NotIdle, QuitAsk, Retired, Settled, State, decode, enter,
    };
    use crate::chan::{NoticeRx, RecvError};
    use crate::executor;
    use crate::thread::{self, DoorbellWorker, WorkerCap, WorkerStack};
    use crate::transport::TrapTransport;

    /// The worker's entry, as the founding calls it.
    fn trampoline<E: Engine>() {
        enter::<E>(&super::Native);
    }

    /// How soon a park whose receive was refused for want of a standing post
    /// looks again. A rotating receiver's handoff is in flight, and nothing
    /// is armed to wake the host when it lands. This is the figure the
    /// tree's other notice receivers retry a handoff at (`chan`, `process`).
    const HANDOFF_RETRY_NS: u64 = 100_000;

    /// A **side source** the host serves beside the worker's calls, kicked
    /// through the same wake channel: work other threads hand thread 0, such
    /// as a C program's output, which any of its threads may write.
    ///
    /// The host runs the source's passes between the worker's calls and
    /// beside one in flight, so neither waits for the other. A pass never
    /// ends a serve: only the serve's stop does. Producers kick the host only
    /// while it is armed, so a run of work costs no trap per item.
    pub trait Side {
        /// Whether the source has work for the host now.
        fn pending(&self) -> bool;
        /// Have the source's producers kick the wake channel from now on.
        /// The host arms after its wake post stands and before it re-reads
        /// [`pending`](Self::pending). Work published after the arm kicks a
        /// standing post, and work published before it is seen (module doc).
        fn arm(&self);
        /// One bounded pass over the source's work. `kick_worker` wakes the
        /// worker, for work it waits on beside the bridge (room it needs).
        /// A pass beside a call is dropped mid-way when the call completes
        /// first, so it must be cancellation-safe.
        fn pass(&mut self, kick_worker: &dyn Fn()) -> impl Future<Output = ()>;
    }

    /// No side source: [`WorkerHost::serve`] and
    /// [`serve_until`](WorkerHost::serve_until).
    impl Side for () {
        fn pending(&self) -> bool {
            false
        }

        fn arm(&self) {}

        async fn pass(&mut self, _: &dyn Fn()) {}
    }

    /// **The host's half of a bridge**: the worker it founded over
    /// `E`'s bridge, and the wake channel it parks on.
    ///
    /// Stays on the thread that spawned it: the worker's doorbell send and
    /// the wake channel's recv are holdings of that thread.
    #[must_use = "a hosted worker is quit or joined, or deliberately abandoned"]
    pub struct WorkerHost<'t, E: Engine> {
        worker: DoorbellWorker,
        wake: NoticeRx<'t, TrapTransport>,
        _engine: PhantomData<(fn() -> E, *const ())>,
    }

    impl<'t, E: Engine> WorkerHost<'t, E> {
        /// **Found a worker over `E`'s bridge**, holding its first work item.
        ///
        /// `first` writes that item while the host holds the claimed bridge;
        /// then the word reads `WORK` and the worker is founded on `stack` —
        /// founding is the hand-off, so the item is the worker's from its
        /// first instruction. `extra` endows it as
        /// [`spawn_with_doorbell_on`](thread::spawn_with_doorbell_on) does.
        ///
        /// `None` when another host holds the bridge, or the founding refuses
        /// (no `CHANNEL_SPACE`, no heap, a stack the window cannot place); the
        /// bridge is vacant again then, since nothing reads it.
        pub fn spawn(
            transport: &'t TrapTransport,
            stack: WorkerStack,
            extra: &[WorkerCap],
            first: impl FnOnce(&mut E::Payload),
        ) -> Option<Self> {
            let bridge = E::bridge();
            if !bridge.claim() {
                return None;
            }
            // SAFETY: the claim made the word `IDLE`, and no worker exists to
            // move it.
            first(unsafe { bridge.host_payload() });
            bridge.hand_over_first();
            let Some(worker) = thread::spawn_with_doorbell_on(trampoline::<E>, extra, stack) else {
                bridge.vacate();
                return None;
            };
            let wake = NoticeRx::new(transport, worker.wake_recv());
            Some(Self {
                worker,
                wake,
                _engine: PhantomData,
            })
        }

        /// The payload, while the host holds the cell between items or after
        /// the worker is gone — where a settled item's results are.
        pub fn payload(&mut self) -> Option<&mut E::Payload> {
            let bridge = E::bridge();
            match bridge.state() {
                // SAFETY: `IDLE` and `GONE` hand the host the cell, and the
                // `&mut self` borrow keeps every other host operation out
                // until this one ends.
                State::Idle | State::Gone(_) => Some(unsafe { bridge.host_payload() }),
                _ => None,
            }
        }

        /// **Hand the worker its next item**: `work` writes it into the idle
        /// cell, the word moves to `WORK`, and the worker is kicked.
        pub fn post(&mut self, work: impl FnOnce(&mut E::Payload)) -> Result<(), NotIdle> {
            E::bridge().post_with(work)?;
            let _ = self.worker.kick();
            Ok(())
        }

        /// **Kick the worker's doorbell**, for an engine whose worker also
        /// waits on words of its own beside the bridge (a C program's main
        /// thread, waiting for room for its output). A kick is a hint: the
        /// worker re-reads whatever it waits on, so a spurious one costs a
        /// re-read. `false` when the kernel refused it.
        pub fn kick(&self) -> bool {
            self.worker.kick()
        }

        /// **Serve the worker's calls until it yields the cell**: `answer`
        /// dispatches each call, holding the payload for the whole call.
        /// Returns at once if the worker is already idle or gone.
        pub async fn serve(
            &mut self,
            answer: impl AsyncFnMut(&mut E::Payload),
        ) -> Result<Settled, Fatal> {
            match self.drive(answer, None::<fn() -> bool>, &mut ()).await? {
                Some(settled) => Ok(settled),
                // `drive` stops only for a stop condition, and there is none.
                None => Err(Fatal),
            }
        }

        /// [`serve`](Self::serve) that also stops — `Ok(None)` — once `stop`
        /// holds: before a park, after every wake, and while a call is being
        /// served. A call cut short is left unanswered (its future dropped, as
        /// any cancellation leaves it) and its worker parked; the host may
        /// then retire the worker without waiting for it.
        ///
        /// For a host whose other threads can end it — a C program's
        /// `exit` from any thread, which posts its own word and kicks the
        /// same wake channel.
        pub async fn serve_until(
            &mut self,
            answer: impl AsyncFnMut(&mut E::Payload),
            stop: impl Fn() -> bool,
        ) -> Result<Option<Settled>, Fatal> {
            self.drive(answer, Some(stop), &mut ()).await
        }

        /// [`serve_until`](Self::serve_until) with a [`Side`] source served
        /// beside the worker: a pass whenever the source has work, between
        /// the worker's calls and beside a call in flight. The stop and the
        /// worker's next call are read between passes, so a source whose
        /// producers never pause still lets the serve stop.
        ///
        /// For a C program's host, whose every thread writes the output the
        /// host relays.
        pub async fn serve_beside(
            &mut self,
            answer: impl AsyncFnMut(&mut E::Payload),
            stop: impl Fn() -> bool,
            side: &mut impl Side,
        ) -> Result<Option<Settled>, Fatal> {
            self.drive(answer, Some(stop), side).await
        }

        async fn drive<S: Fn() -> bool>(
            &mut self,
            mut answer: impl AsyncFnMut(&mut E::Payload),
            stop: Option<S>,
            side: &mut impl Side,
        ) -> Result<Option<Settled>, Fatal> {
            let bridge = E::bridge();
            let Self { worker, wake, .. } = self;
            let worker = &*worker;
            let stopped = || stop.as_ref().is_some_and(|stop| stop());
            let kick_worker = || {
                let _ = worker.kick();
            };
            loop {
                if stopped() {
                    return Ok(None);
                }
                // One pass, then the worker: the stop and its next call are
                // read between passes, however fast the side's producers are.
                if side.pending() {
                    side.pass(&kick_worker).await;
                    if stopped() {
                        return Ok(None);
                    }
                }
                match bridge.host_step() {
                    HostStep::Settled(settled) => return Ok(Some(settled)),
                    HostStep::Wait(seen) => {
                        let moved =
                            || bridge.word().load(core::sync::atomic::Ordering::Acquire) != seen;
                        park(wake, || moved() || stopped() || armed(side)).await?;
                    }
                    HostStep::Serve => {
                        // SAFETY: `CALL` hands the host the cell until it
                        // answers, and the worker is parked inside `call`,
                        // which does not return while the word reads `CALL`
                        // (module doc): whatever the request borrows from its
                        // frame is alive for the whole call.
                        let payload = unsafe { bridge.host_payload() };
                        if stop.is_none() {
                            answer(payload).await;
                        } else {
                            let call = pin!(answer(payload));
                            let ended = pin!(until(wake, &stopped, side, &kick_worker));
                            if race(call, ended).await.is_none() {
                                return Ok(None);
                            }
                        }
                        // A worker that gave the call up already said so.
                        if bridge.answer() {
                            let _ = worker.kick();
                        }
                    }
                }
            }
        }

        /// **Ask the worker to end, and join it.** An idle worker is asked
        /// (`QUIT`) and the host parks until it is gone; a worker already
        /// leaving is waited for. A worker mid-item would never answer, so it
        /// is [`abandon`](Self::abandon)ed instead.
        pub async fn quit(self) -> Retired {
            match E::bridge().ask_quit() {
                QuitAsk::Asked => {
                    let _ = self.worker.kick();
                }
                QuitAsk::Leaving => {}
                QuitAsk::Busy => return self.abandon(),
            }
            self.join().await
        }

        /// **Wait until the worker is gone, then join its thread** and vacate
        /// the bridge. For a worker the host knows is ending — it asked, or the
        /// worker said so in the engine's own words; a worker that is not
        /// ending is waited for as long as it runs.
        pub async fn join(mut self) -> Retired {
            let bridge = E::bridge();
            loop {
                if let State::Gone(_) = bridge.state() {
                    break;
                }
                let seen = bridge.word().load(core::sync::atomic::Ordering::Acquire);
                if decode(seen) == State::Call {
                    // A call no one will serve: the worker cannot reach `GONE`.
                    return self.abandon();
                }
                let moved = || bridge.word().load(core::sync::atomic::Ordering::Acquire) != seen;
                if park(&mut self.wake, moved).await.is_err() {
                    return self.abandon();
                }
            }
            // `GONE` is the worker's last act on the bridge; joining waits for
            // the thread itself to end.
            let Self { worker, wake, .. } = self;
            drop(wake);
            match worker.join() {
                Ok(_) => {
                    bridge.vacate();
                    Retired::Joined
                }
                // The ticket is spent with the error: its band and stack stay
                // the task's until teardown, and so does the bridge.
                Err(_) => Retired::Abandoned,
            }
        }

        /// **Leave the worker as it is** for the task's teardown: no join, no
        /// retire, no syscall. The bridge stays held, since the worker may
        /// still run.
        pub fn abandon(self) -> Retired {
            core::mem::forget(self);
            Retired::Abandoned
        }
    }

    /// Park on the wake channel until a kick lands or `ready` holds — standing
    /// the post first and re-reading after (module doc).
    async fn park(
        wake: &mut NoticeRx<'_, TrapTransport>,
        ready: impl Fn() -> bool,
    ) -> Result<(), Fatal> {
        loop {
            match wake.try_recv() {
                // A kick landed; the caller re-reads the word.
                Ok(Some(_)) => return Ok(()),
                Ok(None) => {
                    if ready() {
                        return Ok(());
                    }
                    let Some(post) = wake.standing_post() else {
                        return Err(Fatal);
                    };
                    let _ = executor::next_wake_on(post).await;
                }
                Err(RecvError::Refused(Status::WouldBlock)) => {
                    // No post stands to wake on while a rotating receiver's
                    // handoff is in flight, and nothing is armed to say when
                    // it lands: look again shortly rather than wait for a wake
                    // nothing will make.
                    let _ = executor::next_wake_for(HANDOFF_RETRY_NS).await;
                }
                Err(_) => return Err(Fatal),
            }
        }
    }

    /// The side source's part of a park's re-read: arm it, then read it.
    fn armed(side: &impl Side) -> bool {
        side.arm();
        side.pending()
    }

    /// Resolve once `stopped` holds, parking on the wake channel meanwhile
    /// and running the side's passes: the call being served is not a reason
    /// to wake, only a stop or side work is. On a broken channel it never
    /// resolves, since the host can no longer hear a stop, and the call it
    /// races runs to its own end. Each pass yields, so the call is polled
    /// however fast the side's producers are; a pass in flight is dropped
    /// when the call completes first ([`Side::pass`]).
    async fn until(
        wake: &mut NoticeRx<'_, TrapTransport>,
        stopped: &impl Fn() -> bool,
        side: &mut impl Side,
        kick_worker: &dyn Fn(),
    ) {
        while !stopped() {
            if side.pending() {
                side.pass(kick_worker).await;
                executor::yield_now().await;
                continue;
            }
            if park(wake, || stopped() || armed(side)).await.is_err() {
                core::future::pending::<()>().await;
            }
        }
    }

    /// Poll `first`, then `second`, until one completes: `Some` with the
    /// first's output, `None` when the second finished first.
    async fn race<A: Future, B: Future>(
        mut first: Pin<&mut A>,
        mut second: Pin<&mut B>,
    ) -> Option<A::Output> {
        poll_fn(|cx| {
            if let Poll::Ready(output) = first.as_mut().poll(cx) {
                return Poll::Ready(Some(output));
            }
            if second.as_mut().poll(cx).is_ready() {
                return Poll::Ready(None);
            }
            Poll::Pending
        })
        .await
    }
}

// ─── The metal half ────────────────────────────────────────────────────

/// The worker's runtime block names its bridge (by its word's address), its
/// exit hook says its last word, and its wake send reaches the host.
#[cfg(target_os = "none")]
mod imp {
    use super::Gone;

    pub(super) fn adopt(word: usize) -> bool {
        crate::entry::geometry::set_hosted_bridge(word);
        // However the thread ends from here, `thread::exit` runs this first.
        // The hook is the entry's to set, as the thread controller sets its
        // children's: this entry is the worker's, so no one else sets it.
        crate::thread::on_exit(exiting, 0);
        true
    }

    /// The worker's exit hook: a worker still on its bridge as its thread
    /// ends has not said its last word, so its end says it. It runs after
    /// the thread's own code, and neither blocks nor ends the thread.
    fn exiting(_: usize) {
        let reason = if crate::entry::geometry::panicking() {
            Gone::Panicked
        } else {
            Gone::Ended
        };
        super::last_word(reason);
    }

    pub(super) fn hosted() -> usize {
        crate::entry::geometry::hosted_bridge()
    }

    pub(super) fn release(word: usize) {
        if hosted() == word {
            crate::entry::geometry::set_hosted_bridge(0);
        }
    }

    pub(super) fn kick_host() -> bool {
        crate::thread::kick_parent()
    }
}

/// Off metal no thread is a worker: no runtime block names a bridge, and there
/// is no wake send to kick — the honest refusal, never a simulated host.
#[cfg(not(target_os = "none"))]
mod imp {
    #[cfg(test)]
    pub(super) fn adopt(_word: usize) -> bool {
        false
    }

    pub(super) fn hosted() -> usize {
        0
    }

    pub(super) fn release(_word: usize) {}

    pub(super) fn kick_host() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    //! The state machine on real threads. The worker's park only ever
    //! returns spuriously (it yields), which `futex::wait` may always do; the
    //! host polls instead of parking. What this proves is the states, the
    //! races and the hand-over — every payload access below is checked for a
    //! data race under Miri — not the kernel's park discipline (module doc).

    use super::*;
    use core::sync::atomic::AtomicUsize;
    use std::boxed::Box;
    use std::thread;
    use std::vec::Vec;

    /// A worker's ends on the host. `parks` counts the parks this worker may
    /// take before the kernel "refuses" one (`usize::MAX`: never).
    struct TestEnds {
        worker: bool,
        parks: AtomicUsize,
    }

    impl TestEnds {
        const fn worker() -> Self {
            Self {
                worker: true,
                parks: AtomicUsize::new(usize::MAX),
            }
        }

        const fn outsider() -> Self {
            Self {
                worker: false,
                parks: AtomicUsize::new(usize::MAX),
            }
        }

        const fn parking_only(parks: usize) -> Self {
            Self {
                worker: true,
                parks: AtomicUsize::new(parks),
            }
        }
    }

    /// The panic a test worker's `exit` raises, so a test can tell an exit
    /// from a return.
    struct Exited;

    impl Ends for TestEnds {
        fn adopt(&self, _: &AtomicU32) -> bool {
            self.worker
        }

        fn is_worker(&self, _: &AtomicU32) -> bool {
            self.worker
        }

        fn release(&self, _: &AtomicU32) {}

        fn park(&self, word: &AtomicU32, seen: u32) -> WaitOutcome {
            let left = self.parks.load(Ordering::Relaxed);
            if left == 0 {
                return WaitOutcome::Unsupported;
            }
            if left != usize::MAX {
                self.parks.store(left - 1, Ordering::Relaxed);
            }
            if word.load(Ordering::SeqCst) != seen {
                return WaitOutcome::Retry;
            }
            thread::yield_now();
            WaitOutcome::Woken
        }

        fn kick_host(&self) {}

        fn exit(&self) -> ! {
            std::panic::resume_unwind(Box::new(Exited))
        }
    }

    /// The host's side, polled: serve calls with `answer` until the worker
    /// yields the cell.
    fn serve<P>(bridge: &Bridge<P>, mut answer: impl FnMut(&mut P)) -> Settled {
        loop {
            match bridge.host_step() {
                HostStep::Settled(settled) => return settled,
                HostStep::Serve => {
                    // SAFETY: `CALL` hands the host the cell.
                    answer(unsafe { bridge.host_payload() });
                    let _ = bridge.answer();
                }
                HostStep::Wait(_) => thread::yield_now(),
            }
        }
    }

    /// Found a "worker" over `E` on a std thread, the way `spawn` does.
    fn spawn<E: Engine>(
        ends: &'static TestEnds,
        first: impl FnOnce(&mut E::Payload),
    ) -> thread::JoinHandle<()> {
        let bridge = E::bridge();
        assert!(bridge.claim(), "a vacant bridge is claimed once");
        first(unsafe { bridge.host_payload() });
        bridge.hand_over_first();
        thread::spawn(move || enter::<E>(ends))
    }

    #[derive(Default)]
    struct Sums {
        input: u64,
        reply: u64,
        total: u64,
        calls: u32,
    }

    #[test]
    fn every_state_round_trips_through_the_word() {
        for gone in [
            Gone::Ended,
            Gone::Quit,
            Gone::Failed,
            Gone::Unparked,
            Gone::Panicked,
        ] {
            assert_eq!(decode(gone.word()), State::Gone(gone));
        }
        assert_eq!(decode(VACANT), State::Vacant);
        assert_eq!(decode(IDLE), State::Idle);
        assert_eq!(decode(WORK), State::Work);
        assert_eq!(decode(CALL), State::Call);
        assert_eq!(decode(QUIT), State::Quit);
        // The probe's value is never a state a worker could be waiting in.
        assert!(matches!(decode(NEVER), State::Gone(_)));
        // A word this module never writes reads as gone: fail closed.
        assert_eq!(decode(0x77), State::Gone(Gone::Panicked));
    }

    #[test]
    fn a_bridge_has_one_host_at_a_time() {
        static BRIDGE: Bridge<u8> = Bridge::new(0);
        assert!(!BRIDGE.is_hosted());
        assert!(BRIDGE.claim());
        assert!(BRIDGE.is_hosted());
        assert!(!BRIDGE.claim(), "a held bridge is not claimed twice");
        BRIDGE.vacate();
        assert!(BRIDGE.claim(), "a vacated bridge is claimed again");
    }

    // A compiler-shaped engine: one item, many calls, then return.
    struct Summer;
    static SUMMER: Bridge<Sums> = Bridge::new(Sums {
        input: 0,
        reply: 0,
        total: 0,
        calls: 0,
    });
    static SUMMER_ENDS: TestEnds = TestEnds::worker();

    impl Engine for Summer {
        type Payload = Sums;
        fn bridge() -> &'static Bridge<Sums> {
            &SUMMER
        }
        fn run(mut work: Work<Sums>) {
            let calls = work.with(|p| p.calls);
            let mut total = 0;
            for i in 0..u64::from(calls) {
                let reply = SUMMER
                    .call_with(&SUMMER_ENDS, |p| p.input = i, |p| p.reply)
                    .expect("inside the item, on the worker");
                total += reply;
            }
            work.finish(|p| p.total = total);
        }
    }

    #[test]
    fn calls_hand_the_cell_back_and_forth() {
        let calls = if cfg!(miri) { 24 } else { 2000 };
        let worker = spawn::<Summer>(&SUMMER_ENDS, |p| p.calls = calls);
        let first = serve(&SUMMER, |p| p.reply = p.input * 3 + 1);
        // The body finished, then returned: the host reads the item's end and
        // then, or instead, the worker's.
        assert!(matches!(first, Settled::Idle | Settled::Gone(Gone::Ended)));
        worker.join().unwrap();
        assert_eq!(
            serve(&SUMMER, |_| unreachable!()),
            Settled::Gone(Gone::Ended)
        );
        let expected: u64 = (0..u64::from(calls)).map(|i| i * 3 + 1).sum();
        assert_eq!(unsafe { SUMMER.host_payload() }.total, expected);
        SUMMER.vacate();
    }

    // A REPL-shaped engine: items posted one by one until a quit.
    struct Repl;
    static REPL: Bridge<Vec<u32>> = Bridge::new(Vec::new());
    static REPL_ENDS: TestEnds = TestEnds::worker();

    impl Engine for Repl {
        type Payload = Vec<u32>;
        fn bridge() -> &'static Bridge<Vec<u32>> {
            &REPL
        }
        fn run(first: Work<Vec<u32>>) {
            // The engine is built, and says it is ready.
            first.finish(|p| p.push(0));
            loop {
                match REPL.next_with(&REPL_ENDS) {
                    Ok(Next::Work(mut work)) => {
                        let item = work.with(|p| *p.last().unwrap());
                        // Each item makes one call before it finishes.
                        let doubled = REPL
                            .call_with(&REPL_ENDS, |p| p.push(item), |p| *p.last().unwrap())
                            .unwrap();
                        work.finish(|p| p.push(doubled + 1));
                    }
                    Ok(Next::Quit) => return,
                    other => panic!("unexpected {:?}", other.is_ok()),
                }
            }
        }
    }

    #[test]
    fn items_are_posted_served_and_quit() {
        let worker = spawn::<Repl>(&REPL_ENDS, |p| p.clear());
        // The first item (the engine's construction) settles first.
        assert_eq!(serve(&REPL, |_| unreachable!()), Settled::Idle);
        let items = if cfg!(miri) { 8 } else { 300 };
        for item in 1..=items {
            REPL.post_with(|p| p.push(item)).unwrap();
            let settled = serve(&REPL, |p| {
                let asked = p.pop().unwrap();
                p.push(asked * 2);
            });
            assert_eq!(settled, Settled::Idle);
            assert_eq!(unsafe { REPL.host_payload() }.last(), Some(&(item * 2 + 1)));
        }
        assert!(matches!(REPL.ask_quit(), QuitAsk::Asked));
        assert_eq!(serve(&REPL, |_| unreachable!()), Settled::Gone(Gone::Quit));
        worker.join().unwrap();
        assert!(matches!(REPL.ask_quit(), QuitAsk::Leaving));
        REPL.vacate();
    }

    // An engine whose park the kernel refuses mid-call.
    struct GivesUp;
    static GIVES_UP: Bridge<u64> = Bridge::new(0);
    static GIVES_UP_ENDS: TestEnds = TestEnds::parking_only(usize::MAX);

    impl Engine for GivesUp {
        type Payload = u64;
        fn bridge() -> &'static Bridge<u64> {
            &GIVES_UP
        }
        fn run(_work: Work<u64>) {
            // The probe passed; the next park is refused.
            GIVES_UP_ENDS.parks.store(0, Ordering::Relaxed);
            let _ = GIVES_UP.call_with(&GIVES_UP_ENDS, |p| *p = 7, |p| *p);
            panic!("a worker that gave its call up returned into it");
        }
    }

    #[test]
    fn a_worker_that_cannot_park_gives_its_call_up_and_exits() {
        let worker = spawn::<GivesUp>(&GIVES_UP_ENDS, |p| *p = 0);
        let error = worker.join().expect_err("the worker exits, never returns");
        assert!(error.is::<Exited>());
        // The host had not answered: its answer now has no reader.
        assert_eq!(
            GIVES_UP.host_step(),
            HostStep::Settled(Settled::Gone(Gone::Unparked))
        );
        assert!(!GIVES_UP.answer());
        GIVES_UP.vacate();
    }

    #[test]
    fn a_give_up_and_an_answer_race_and_exactly_one_wins() {
        static RACE: Bridge<u32> = Bridge::new(0);
        let rounds = if cfg!(miri) { 9 } else { 600 };
        let mut answered = 0;
        let mut abandoned = 0;
        for round in 0..rounds {
            assert!(RACE.claim());
            RACE.hand_over_first();
            // The worker posts a call, and its park is refused after a few
            // turns. One round in three forces each order; the third races.
            let (parks, order) = match round % 3 {
                0 => (0, Order::WorkerFirst),
                1 => (usize::MAX, Order::HostFirst),
                _ => ((round % 5) as usize, Order::Race(round % 7)),
            };
            let ends = TestEnds::parking_only(parks);
            let (answer, joined) = thread::scope(|scope| {
                let worker = scope.spawn(|| RACE.call_with(&ends, |p| *p = round, |p| *p).unwrap());
                order.wait(|| matches!(RACE.state(), State::Gone(_)));
                // The host answers whenever it sees the call.
                let answer = loop {
                    match RACE.host_step() {
                        HostStep::Serve => {
                            unsafe { *RACE.host_payload() = round + 1 };
                            break RACE.answer();
                        }
                        HostStep::Settled(Settled::Gone(Gone::Unparked)) => break false,
                        _ => thread::yield_now(),
                    }
                };
                (answer, worker.join())
            });
            match order {
                Order::WorkerFirst => assert!(!answer && joined.is_err()),
                Order::HostFirst => assert!(answer && joined.is_ok()),
                Order::Race(_) => {}
            }
            match joined {
                Ok(reply) => {
                    assert!(answer, "a worker reads an answer only the host made");
                    assert_eq!(reply, round + 1);
                    assert_eq!(RACE.state(), State::Work);
                    answered += 1;
                }
                Err(error) => {
                    assert!(error.is::<Exited>());
                    assert!(!answer, "an answered call is never also given up");
                    assert_eq!(RACE.state(), State::Gone(Gone::Unparked));
                    abandoned += 1;
                }
            }
            RACE.vacate();
        }
        assert_eq!(answered + abandoned, rounds);
        assert!(answered > 0 && abandoned > 0, "both orders ran");
    }

    #[test]
    fn a_post_and_a_leaving_worker_race_and_exactly_one_wins() {
        static LEAVE: Bridge<u32> = Bridge::new(0);
        let rounds = if cfg!(miri) { 9 } else { 600 };
        let (mut taken, mut refused) = (0, 0);
        for round in 0..rounds {
            assert!(LEAVE.claim());
            // The worker waits for an item, and its park is refused after a
            // few turns, so it leaves out of the host-held `IDLE`. One round
            // in three forces each order; the third races them.
            let (parks, order) = match round % 3 {
                0 => (0, Order::WorkerFirst),
                1 => (usize::MAX, Order::HostFirst),
                _ => ((round % 4) as usize, Order::Race(round % 7)),
            };
            let ends = TestEnds::parking_only(parks);
            let (posted, took) = thread::scope(|scope| {
                let worker = scope.spawn(|| match LEAVE.next_with(&ends) {
                    Ok(Next::Work(work)) => {
                        // The post won: the worker holds the item and
                        // finishes it.
                        work.finish(|p| *p += 1);
                        true
                    }
                    Ok(Next::Gone(Gone::Unparked)) => false,
                    _ => panic!("unexpected next"),
                });
                order.wait(|| matches!(LEAVE.state(), State::Gone(_)));
                let posted = LEAVE.post_with(|p| *p = round).is_ok();
                (posted, worker.join().unwrap())
            });
            match order {
                Order::WorkerFirst => assert!(!posted && !took),
                Order::HostFirst => assert!(posted && took),
                Order::Race(_) => {}
            }
            taken += usize::from(took);
            refused += usize::from(!posted);
            if took {
                assert!(posted);
                assert_eq!(LEAVE.host_step(), HostStep::Settled(Settled::Idle));
                assert_eq!(unsafe { *LEAVE.host_payload() }, round + 1);
            } else {
                // Whichever came first, the host now reads the worker gone and
                // no item is left unread in a worker-held state.
                assert_eq!(
                    LEAVE.host_step(),
                    HostStep::Settled(Settled::Gone(Gone::Unparked))
                );
            }
            LEAVE.vacate();
        }
        assert!(taken > 0 && refused > 0, "both orders ran");
    }

    /// Who moves first in a race round.
    #[derive(Clone, Copy)]
    enum Order {
        /// The worker's move is waited for before the host's.
        WorkerFirst,
        /// The worker cannot move before the host does.
        HostFirst,
        /// Neither waits; the host yields this many turns first.
        Race(u32),
    }

    impl Order {
        /// Hold the host back as the order asks; `moved` reads whether the
        /// worker has made its move.
        fn wait(self, moved: impl Fn() -> bool) {
            match self {
                Self::WorkerFirst => {
                    while !moved() {
                        thread::yield_now();
                    }
                }
                Self::HostFirst => {}
                Self::Race(turns) => {
                    for _ in 0..turns {
                        thread::yield_now();
                    }
                }
            }
        }
    }

    // A one-item engine that finishes and returns while the host quits it.
    struct OneShot;
    static ONE_SHOT: Bridge<u32> = Bridge::new(0);
    static ONE_SHOT_ENDS: TestEnds = TestEnds::worker();

    impl Engine for OneShot {
        type Payload = u32;
        fn bridge() -> &'static Bridge<u32> {
            &ONE_SHOT
        }
        fn run(work: Work<u32>) {
            work.finish(|p| *p = 42);
        }
    }

    #[test]
    fn a_quit_and_a_returning_body_race_and_the_worker_is_gone_either_way() {
        let rounds = if cfg!(miri) { 4 } else { 200 };
        for _ in 0..rounds {
            let worker = spawn::<OneShot>(&ONE_SHOT_ENDS, |p| *p = 0);
            // Wait for the item's end, then quit while the body returns.
            loop {
                if let HostStep::Settled(_) = ONE_SHOT.host_step() {
                    break;
                }
                thread::yield_now();
            }
            assert_eq!(unsafe { *ONE_SHOT.host_payload() }, 42);
            let ask = ONE_SHOT.ask_quit();
            assert!(matches!(ask, QuitAsk::Asked | QuitAsk::Leaving));
            worker.join().unwrap();
            assert!(matches!(
                ONE_SHOT.state(),
                State::Gone(Gone::Quit | Gone::Ended)
            ));
            ONE_SHOT.vacate();
        }
    }

    // An engine that cannot start.
    struct NoStart;
    static NO_START: Bridge<&'static str> = Bridge::new("");
    static NO_START_ENDS: TestEnds = TestEnds::worker();

    impl Engine for NoStart {
        type Payload = &'static str;
        fn bridge() -> &'static Bridge<&'static str> {
            &NO_START
        }
        fn run(work: Work<&'static str>) {
            work.fail(|p| *p = "no engine");
        }
    }

    #[test]
    fn a_failed_item_leaves_the_worker_gone_with_its_reason_in_the_payload() {
        let worker = spawn::<NoStart>(&NO_START_ENDS, |p| *p = "");
        worker.join().unwrap();
        assert_eq!(
            serve(&NO_START, |_| unreachable!()),
            Settled::Gone(Gone::Failed)
        );
        assert_eq!(unsafe { *NO_START.host_payload() }, "no engine");
        // A post to a gone worker is refused with its reason.
        assert_eq!(NO_START.post_with(|_| ()), Err(NotIdle::Gone(Gone::Failed)));
        NO_START.vacate();
    }

    // A thread that cannot park at all never runs the body.
    struct NeverRuns;
    static NEVER_RUNS: Bridge<bool> = Bridge::new(false);
    static NEVER_RUNS_ENDS: TestEnds = TestEnds::parking_only(0);

    impl Engine for NeverRuns {
        type Payload = bool;
        fn bridge() -> &'static Bridge<bool> {
            &NEVER_RUNS
        }
        fn run(work: Work<bool>) {
            work.finish(|p| *p = true);
        }
    }

    #[test]
    fn the_probe_keeps_an_unparkable_worker_from_running() {
        let worker = spawn::<NeverRuns>(&NEVER_RUNS_ENDS, |p| *p = false);
        worker.join().unwrap();
        assert_eq!(
            serve(&NEVER_RUNS, |_| unreachable!()),
            Settled::Gone(Gone::Unparked)
        );
        assert!(!unsafe { *NEVER_RUNS.host_payload() });
        NEVER_RUNS.vacate();
    }

    #[test]
    fn worker_side_operations_refuse_everyone_but_the_worker_inside_an_item() {
        static ROOM: Bridge<u32> = Bridge::new(0);
        static OUTSIDER: TestEnds = TestEnds::outsider();
        static WORKER: TestEnds = TestEnds::worker();
        assert!(ROOM.claim());
        // Between items (`IDLE`): the cell is the host's.
        assert_eq!(ROOM.with_work_as(&WORKER, |_| ()), Err(Refused::NoWork));
        assert_eq!(
            ROOM.call_with(&WORKER, |_| (), |_| ()),
            Err(Refused::NoWork)
        );
        ROOM.hand_over_first();
        // Inside an item, only the worker.
        assert_eq!(
            ROOM.with_work_as(&OUTSIDER, |_| ()),
            Err(Refused::NotWorker)
        );
        assert_eq!(
            ROOM.call_with(&OUTSIDER, |_| (), |_| ()),
            Err(Refused::NotWorker)
        );
        assert!(matches!(ROOM.next_with(&OUTSIDER), Err(Refused::NotWorker)));
        assert!(!ROOM.leave_with(&OUTSIDER, Gone::Ended));
        // A borrow inside a borrow aliases the payload: refused, untouched.
        let nested = ROOM.with_work_as(&WORKER, |p| {
            *p = 5;
            ROOM.call_with(&WORKER, |q| *q = 6, |_| ())
        });
        assert_eq!(nested, Ok(Err(Refused::Nested)));
        assert_eq!(ROOM.state(), State::Work, "the refused call posted nothing");
        assert_eq!(ROOM.with_work_as(&WORKER, |p| *p), Ok(5));
        // The worker ends itself, holding its item.
        let mut work = Work::new(&ROOM);
        assert_eq!(work.with(|p| *p), 5);
        assert!(ROOM.leave_with(&WORKER, Gone::Ended));
        assert!(!ROOM.leave_with(&WORKER, Gone::Ended), "once");
        assert_eq!(ROOM.state(), State::Gone(Gone::Ended));
        // The item is dead with it: the cell is the host's now, and its
        // finish may not write the word back to `IDLE`.
        let late = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            work.finish(|p| *p = 9);
        }));
        assert!(late.is_err());
        assert_eq!(ROOM.state(), State::Gone(Gone::Ended));
        assert_eq!(unsafe { *ROOM.host_payload() }, 5);
        ROOM.vacate();
    }

    #[test]
    fn a_panic_is_the_last_word_from_any_state_but_gone() {
        for (from, expected) in [
            (IDLE, Some(Gone::Panicked)),
            (WORK, Some(Gone::Panicked)),
            (CALL, Some(Gone::Panicked)),
            (QUIT, Some(Gone::Panicked)),
            (Gone::Quit.word(), None),
            (VACANT, None),
        ] {
            let word = AtomicU32::new(from);
            assert_eq!(settle(&word, Gone::Panicked), expected.is_some());
            match expected {
                Some(gone) => assert_eq!(decode(word.load(Ordering::Relaxed)), State::Gone(gone)),
                None => assert_eq!(word.load(Ordering::Relaxed), from),
            }
        }
        // A worker ending itself while asked to quit has answered the quit.
        let word = AtomicU32::new(QUIT);
        assert!(settle(&word, Gone::Ended));
        assert_eq!(
            decode(word.load(Ordering::Relaxed)),
            State::Gone(Gone::Quit)
        );
        // Off metal no thread is a worker: the panic path leaves no word.
        panicked();
    }

    #[test]
    fn a_host_serves_only_calls_and_waits_out_the_workers_states() {
        static STEPS: Bridge<u8> = Bridge::new(0);
        for (word, step) in [
            (IDLE, HostStep::Settled(Settled::Idle)),
            (WORK, HostStep::Wait(WORK)),
            (QUIT, HostStep::Wait(QUIT)),
            (CALL, HostStep::Serve),
            (
                Gone::Failed.word(),
                HostStep::Settled(Settled::Gone(Gone::Failed)),
            ),
        ] {
            STEPS.word().store(word, Ordering::Relaxed);
            assert_eq!(STEPS.host_step(), step);
        }
        // A post needs the idle cell, and quitting a busy worker is refused.
        STEPS.word().store(WORK, Ordering::Relaxed);
        assert_eq!(STEPS.post_with(|_| ()), Err(NotIdle::Busy));
        assert!(matches!(STEPS.ask_quit(), QuitAsk::Busy));
        STEPS.vacate();
    }
}
