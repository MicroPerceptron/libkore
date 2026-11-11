//! The host of an independent C program: `libkore.so`'s half of the program
//! lifecycle Kore's C library starts and ends.
//!
//! A C program on Kore links its C library's start files whatever else it
//! links. Their `_start` hands the program to the C library, which, when the
//! program also links this library, hands it on to [`__kore_program_start`].
//! From there:
//!
//! - **The program runs on its main thread**, a worker with a C stack of
//!   [`KORE_MAIN_STACK_BYTES`](crate::api::KORE_MAIN_STACK_BYTES), not on
//!   thread 0. Thread 0's stack is the
//!   founding's, sized for a runtime, and thread 0 is the process owner: the
//!   one thread that may stop the others.
//! - **Thread 0 is the host.** It keeps the task's executor and its service
//!   faces, and serves what the program's threads ask of it through one
//!   request cell and one word, the shape `kore-lua`'s VM bridge and `tcc`
//!   share: the program's filesystem session ([`kore_fs_run`](crate::api::kore_fs_run))
//!   and its end.
//! - **Any thread can end the program.** `exit` and `_Exit` reach
//!   [`__kore_program_exit`] once the C library has run its `atexit`
//!   handlers; the first request wins. The host stops every other thread,
//!   runs the finalizers when none was still running, and ends the process
//!   with the C status through the C library's own exit.
//!
//! ```text
//!  thread 0 (the host)                         main thread (C)
//!  ───────────────────                         ───────────────
//!  found the main thread ────────────────────▶ program(): main(), then exit
//!  park on the wake channel                    … kore_fs_run(step, user)
//!  ◀──────── SESSION_POSTED, kick_parent ───── futex::wait on the word
//!  run the session (its C callbacks run here)
//!  SESSION_SERVED, kick ─────────────────────▶ returns the session's status
//!                                              … exit(status): atexit handlers
//!  ◀──────── the exit word, kick_parent ────── the thread ends
//!  join it, raise the barrier, finalize
//!  end(status): the C library's exit
//! ```
//!
//! Every park is a doorbell park; nothing here spins. The ordering is
//! `kore-lua`'s: a poster stores, then kicks; the host stands its post,
//! re-reads both words, then parks, so a kick that found no post standing
//! announced a store the re-read sees. Every thread of the program can kick
//! the host: the main thread holds the wake channel's send, and each thread it
//! starts is founded holding one too (`thread_controller`).
//!
//! A launch that cannot carry the host (no channel space or heap, a stack
//! window that cannot place the main thread) runs the program on thread 0, as
//! the C library alone would: allocation, threads, time and random work, a
//! filesystem session is refused, and only thread 0's exit ends the program.

use core::cell::{Cell, UnsafeCell};
use core::ffi::{c_int, c_void};
use core::future::poll_fn;
use core::pin::pin;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use core::task::Poll;

use kore_ffi::{KoreFsError, KoreFsStep, KoreStatus, Session};

use crate::exit::{Exit, ExitWord};
use kore_std::ExitCode;
use kore_std::abi::session::Status;
use kore_std::fs::Fs;
use kore_std::runtime::chan::{NoticeRx, RecvError};
use kore_std::runtime::entry::{self, Rt};
use kore_std::runtime::executor::{self, Executor};
use kore_std::runtime::futex::{self, WaitOutcome};
use kore_std::runtime::thread::{self, DoorbellWorker, WorkerStack};
use kore_std::runtime::transport::TrapTransport;
use kore_std::runtime::{crash, startup};

/// The C library's program: runs `main`, then `exit`s with its value.
pub type Program = unsafe extern "C" fn() -> !;
/// The C library's exit: commits `status` on the calling thread's own handle.
pub type End = unsafe extern "C" fn(c_int) -> !;

/// The program, as [`__kore_program_start`] received it, for the main thread.
static PROGRAM: AtomicUsize = AtomicUsize::new(0);
/// The C library's exit, as [`__kore_program_start`] received it, for an
/// [`abort`] on thread 0.
static END: AtomicUsize = AtomicUsize::new(0);
/// Set by the first [`__kore_program_start`]; a second never founds a host.
static STARTED: AtomicBool = AtomicBool::new(false);
/// The host serves: the main thread exists and thread 0 parks for it. Set
/// before the main thread is founded, since its first call may need it.
static HOSTED: AtomicBool = AtomicBool::new(false);

/// This thread's part in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A thread the program started, or thread 0 before the program runs.
    Other,
    /// Thread 0, once it holds the program: the host, or the program itself
    /// when the launch could not carry a host.
    Host,
    /// The thread `main` runs on.
    Main,
}

#[thread_local]
static ROLE: Cell<Role> = Cell::new(Role::Other);

/// Whether the calling thread is the host: thread 0 holding the program.
pub(crate) fn on_host() -> bool {
    ROLE.get() == Role::Host
}

/// Whether the calling thread is the main thread, the one the host can kick.
pub(crate) fn on_main() -> bool {
    ROLE.get() == Role::Main
}

// ── The end ─────────────────────────────────────────────────────────────

/// The program's end, once asked for (see [`ExitWord`]).
static EXIT: ExitWord = ExitWord::new();

// ── The session cell ────────────────────────────────────────────────────

// The word's states. The main thread owns the cell at IDLE and SERVED, the
// host at POSTED; every hand-over is a Release store matched by an Acquire.
const IDLE: u32 = 0;
const POSTED: u32 = 1;
const SERVED: u32 = 2;

struct Request {
    state: AtomicU32,
    payload: UnsafeCell<Payload>,
}

struct Payload {
    // main thread → host: the session's callback program.
    step: Option<KoreFsStep>,
    user: *mut c_void,
    // host → main thread: how the session ended.
    status: KoreStatus,
    error: KoreFsError,
}

// SAFETY: `payload` is touched only by the side the word hands it to (above).
unsafe impl Sync for Request {}

static SESSION: Request = Request {
    state: AtomicU32::new(IDLE),
    payload: UnsafeCell::new(Payload {
        step: None,
        user: core::ptr::null_mut(),
        status: KoreStatus::Ok,
        error: KoreFsError::None,
    }),
};

impl Request {
    /// # Safety
    /// The caller observed (Acquire) a state that hands it the cell, and holds
    /// no other reference across the store that hands it back.
    #[allow(clippy::mut_from_ref)]
    unsafe fn payload(&self) -> &mut Payload {
        unsafe { &mut *self.payload.get() }
    }
}

/// Anything the host must answer: an end, or a session.
fn pending() -> bool {
    EXIT.posted().is_some() || SESSION.state.load(Ordering::Acquire) == POSTED
}

// ── Entry points the C library calls ────────────────────────────────────

/// Run the program. The C library's start hands it here when the program
/// links libkore: `program` runs `main` and then `exit`; `end` is the C
/// library's exit, which commits a status on the calling thread's own handle.
/// Never returns.
///
/// # Safety
/// Called once, by the C library's start on the founded thread and before the
/// program has run. Neither `program` nor `end` returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kore_program_start(program: Option<Program>, end: Option<End>) -> ! {
    let (Some(program), Some(end)) = (program, end) else {
        // A C library that hands over nothing to run has broken its half of
        // this contract; say so rather than run on.
        thread::exit(ExitCode::INTERNAL)
    };
    if STARTED.swap(true, Ordering::AcqRel) {
        // A second start founds no second host.
        unsafe { program() }
    }
    ROLE.set(Role::Host);
    PROGRAM.store(program as usize, Ordering::Release);
    END.store(end as usize, Ordering::Release);
    match Founded::found() {
        Some(host) => host.serve(end),
        None => unsafe { program() },
    }
}

/// End the program with `status`. The C library's `exit` calls this after its
/// `atexit` handlers with `finalize` nonzero, `_Exit` with zero, on whatever
/// thread called them.
///
/// Returns only when the caller must commit the exit itself: it is thread 0,
/// every other thread is stopped, and, for `exit`, the finalizers have run if
/// none was still running. On any other thread the request goes to the host
/// and the calling thread ends.
#[unsafe(no_mangle)]
pub extern "C" fn __kore_program_exit(status: c_int, finalize: c_int) {
    let finalize = finalize != 0;
    match ROLE.get() {
        // Thread 0 itself: the program when there is no host, or a callback
        // or finalizer the host is running. It cannot wait for its own loop,
        // so it relays what the launcher has room for now; the barrier is
        // this thread's.
        Role::Host => {
            crate::stdio::flush_now();
            startup::barrier(finalize)
        }
        role if HOSTED.load(Ordering::Acquire) => {
            let _ = EXIT.post(Exit {
                status,
                finalize,
                from_main: role == Role::Main,
            });
            // Refused only when no post stood, and then the host re-reads the
            // word before it parks: the store above reaches it either way.
            let _ = thread::kick_parent();
            thread::exit(ExitCode::OK)
        }
        // No host: only thread 0 owns the process. Anywhere else the barrier
        // refuses and ends this thread alone.
        _ => startup::barrier(finalize),
    }
}

/// `abort` (C11 7.22.4.1): the program's abnormal end. It ends with the
/// aborted status, 134 — what a shell reports for a program `SIGABRT` ended —
/// and, like `_Exit`, runs no `atexit` handler or finalizer. No stream
/// buffers, so what the program wrote before has already left it: the host's
/// loop relays all of it before the end when another thread aborts, and a
/// callback's `abort` on the host itself relays what the launcher has room
/// for at once (`stdio::flush_now`).
///
/// The C library's `_Exit` lives in the program, so this takes `_Exit`'s
/// path itself: any other thread hands the status to the host and ends, and
/// thread 0, the barrier raised, commits the end the C library handed over.
#[unsafe(no_mangle)]
pub extern "C" fn abort() -> ! {
    let status = ExitCode::ABORTED.get();
    __kore_program_exit(status, 0);
    match END.load(Ordering::Acquire) {
        // A program the C library never started: thread 0 ends the task.
        0 => thread::exit(ExitCode::ABORTED),
        end => {
            // SAFETY: `__kore_program_start` stored an `End`, which never
            // returns; every other thread is stopped.
            let end: End = unsafe { core::mem::transmute(end) };
            unsafe { end(status) }
        }
    }
}

/// The main thread's half of [`kore_fs_run`](crate::api::kore_fs_run): post
/// the session to the host and wait for its answer.
///
/// # Safety
/// `kore_fs_run`'s contract for `step` and `user`.
pub(crate) unsafe fn session(step: KoreFsStep, user: *mut c_void) -> (KoreStatus, KoreFsError) {
    if ROLE.get() != Role::Main || !HOSTED.load(Ordering::Acquire) {
        return (KoreStatus::Refused, KoreFsError::None);
    }
    // The main thread is the only poster, and it waits out each request: the
    // cell is idle whenever control is here.
    {
        // SAFETY: IDLE hands the cell to the main thread.
        let payload = unsafe { SESSION.payload() };
        payload.step = Some(step);
        payload.user = user;
    }
    SESSION.state.store(POSTED, Ordering::Release);
    // Refused only when no post stood, and then the host re-reads the word
    // before it parks.
    let _ = thread::kick_parent();
    loop {
        match SESSION.state.load(Ordering::Acquire) {
            SERVED => break,
            state => {
                if futex::wait(&SESSION.state, state, None) == WaitOutcome::Unsupported {
                    // The main thread is founded with its own doorbell, so it
                    // can always park; if it cannot, spinning is the one
                    // thing left, and this host will not. End the program.
                    fatal()
                }
            }
        }
    }
    let answer = {
        // SAFETY: SERVED hands the cell back to the main thread.
        let payload = unsafe { SESSION.payload() };
        payload.step = None;
        payload.user = core::ptr::null_mut();
        (payload.status, payload.error)
    };
    SESSION.state.store(IDLE, Ordering::Release);
    answer
}

/// This library's panic: the record first, then the end. A panic on thread 0
/// stops the program there; anywhere else it asks the host to, so the process
/// ends with the panic rather than running on without the panicking thread.
pub(crate) fn panicked(info: &core::panic::PanicInfo<'_>) -> ! {
    crash::report(info);
    match ROLE.get() {
        Role::Host => {
            crate::stdio::flush_now();
            startup::barrier(false)
        }
        role if HOSTED.load(Ordering::Acquire) => {
            let _ = EXIT.post(Exit {
                status: ExitCode::PANIC.get(),
                finalize: false,
                from_main: role == Role::Main,
            });
            let _ = thread::kick_parent();
        }
        _ => {}
    }
    thread::exit(ExitCode::PANIC)
}

/// A host invariant broke on a program thread: end the program with
/// `INTERNAL`, never by spinning.
fn fatal() -> ! {
    let _ = EXIT.post(Exit {
        status: ExitCode::INTERNAL.get(),
        finalize: false,
        from_main: ROLE.get() == Role::Main,
    });
    let _ = thread::kick_parent();
    thread::exit(ExitCode::INTERNAL)
}

// ── The host ────────────────────────────────────────────────────────────

/// The host's wake channel broke or a park was refused: it cannot hear the
/// program, so it ends it.
struct Fatal;

/// Thread 0 with its program founded: the runtime context, the executor's
/// doorbell, the launcher's stream when the launch carried one, and the main
/// thread.
struct Founded {
    rt: Rt,
    doorbell: executor::Doorbell,
    output: Option<&'static crate::stdio::Endpoint>,
    worker: DoorbellWorker,
}

impl Founded {
    /// Found the main thread. `None` when the launch cannot carry the host.
    fn found() -> Option<Self> {
        let rt = entry::thread_context()?;
        let doorbell = rt.doorbell()?;
        // The C streams reach the launcher through this thread: map its
        // stream and install the sink before any thread can write.
        let output = crate::stdio::install(&rt);
        HOSTED.store(true, Ordering::Release);
        // A mint grant of its own: the first `pthread_create` on the main
        // thread starts kore-rt's thread controller there, which mints the
        // doorbells its threads park on (`thread_controller`). Without one the
        // controller still runs, taking the deadline park.
        let grant = thread::channel_space_endowment();
        match thread::spawn_with_doorbell_on(
            main_thread,
            grant.as_slice(),
            WorkerStack::Bytes(crate::api::KORE_MAIN_STACK_BYTES),
        ) {
            Some(worker) => Some(Self {
                rt,
                doorbell,
                output,
                worker,
            }),
            None => {
                HOSTED.store(false, Ordering::Release);
                // No host will drain the stream: writes report `EOF`, as
                // they do for a launch that carried none.
                if output.is_some() {
                    crate::stdio::abandon();
                }
                None
            }
        }
    }

    /// Serve the program until it ends, then end the process.
    fn serve(self, end: End) -> ! {
        let Self {
            rt,
            doorbell,
            output,
            worker,
        } = self;
        let transport = TrapTransport::new();
        let exit = {
            let mut executor = Executor::new(entry::transport(), doorbell);
            executor.run(async {
                let exit = host(&rt, &transport, &worker, output).await;
                // Whatever ended the program, what it wrote reaches the
                // launcher, and then the stream's close, before the process
                // ends: the launcher shows all of it before the status.
                if let Some(output) = output {
                    crate::stdio::finish(output, &|| {
                        let _ = worker.kick();
                    })
                    .await;
                }
                exit
            })
        };
        let exit = exit.unwrap_or(Exit {
            status: ExitCode::INTERNAL.get(),
            finalize: false,
            from_main: false,
        });
        if exit.from_main {
            // The main thread asked and is ending itself: wait for it, so a
            // program that ran on it alone is stopped cleanly.
            let _ = worker.join();
        } else {
            // Still running: the barrier stops it, and its channels end with
            // the process. Its join ticket must not try to reclaim it first.
            core::mem::forget(worker);
        }
        startup::barrier(exit.finalize);
        // SAFETY: the C library's exit, as `__kore_program_start` received it.
        unsafe { end(exit.status) }
    }
}

/// The main thread's entry: the program, as the C library handed it over.
fn main_thread() {
    ROLE.set(Role::Main);
    // SAFETY: `__kore_program_start` stored a `Program` before founding this
    // thread, and the founding's session ordered that store before this load.
    let program: Program = unsafe { core::mem::transmute(PROGRAM.load(Ordering::Acquire)) };
    unsafe { program() }
}

/// The host's loop: answer each session the main thread posts, until any
/// thread asks for the end.
async fn host(
    rt: &Rt,
    transport: &TrapTransport,
    worker: &DoorbellWorker,
    output: Option<&'static crate::stdio::Endpoint>,
) -> Result<Exit, Fatal> {
    let mut wake = NoticeRx::new(transport, worker.wake_recv());
    let mut filesystem = Filesystem::new(rt, transport);
    // The one writer the host can kick, woken when its room is made.
    let wake_main = || {
        let _ = worker.kick();
    };
    loop {
        if let Some(exit) = EXIT.posted() {
            return Ok(exit);
        }
        if let Some(output) = output {
            // One bounded pass: the end and the next session are served
            // between passes, however fast the program writes.
            crate::stdio::drain(output, &wake_main).await;
            if let Some(exit) = EXIT.posted() {
                return Ok(exit);
            }
        }
        if SESSION.state.load(Ordering::Acquire) == POSTED {
            // SAFETY: POSTED hands the cell to the host until it stores SERVED.
            let (step, user) = {
                let payload = unsafe { SESSION.payload() };
                (payload.step, payload.user)
            };
            let Some(step) = step else {
                return Err(Fatal);
            };
            // An end requested mid-session ends the program without waiting
            // for the session: the barrier stops the waiting main thread, and
            // the abandoned session's owner is quarantined, as cancellation
            // leaves it.
            let session = pin!(filesystem.run(step, user));
            let ended = pin!(until_exit(&mut wake, output, &wake_main));
            let outcome = race(session, ended).await;
            let Some((status, error)) = outcome else {
                continue;
            };
            {
                // SAFETY: still POSTED; the host owns the cell.
                let payload = unsafe { SESSION.payload() };
                payload.status = status;
                payload.error = error;
            }
            SESSION.state.store(SERVED, Ordering::Release);
            let _ = worker.kick();
            continue;
        }
        // Armed before the recheck `park` makes, so a writer that published
        // meanwhile either kicks or is seen.
        crate::stdio::arm();
        park(&mut wake, || pending() || crate::stdio::pending()).await?;
    }
}

/// Poll `first` and then `second` until one completes: `Some` with the first's
/// output, or `None` when the second finished first.
async fn race<A: Future, B: Future>(
    mut first: core::pin::Pin<&mut A>,
    mut second: core::pin::Pin<&mut B>,
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

/// How soon a receive refused for want of a standing post looks again: the
/// tree's figure for a rotating notice receiver (the WASI host's handoff).
const WOULD_BLOCK_RETRY_NS: u64 = 100_000;

/// Park until a kick, a spurious wake, or `ready` holds, having stood the
/// wake post before re-reading what `ready` reads.
async fn park(
    wake: &mut NoticeRx<'_, TrapTransport>,
    ready: impl Fn() -> bool,
) -> Result<(), Fatal> {
    loop {
        match wake.try_recv() {
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
                // No post stands to wake on: look again shortly, as the
                // tree's other receivers do, rather than wait for a wake
                // nothing armed.
                let _ = executor::next_wake_for(WOULD_BLOCK_RETRY_NS).await;
            }
            Err(_) => return Err(Fatal),
        }
    }
}

/// Resolve once an end is posted, parking on the wake channel meanwhile and
/// draining what the program writes: the session being served is not a
/// reason to wake, only an end or output is. On a broken channel it never
/// resolves, since the host can no longer hear an end; the session it races
/// then runs to its own completion. Dropped when the session ends first,
/// which the drain survives (`stdio::drain`). Each pass yields, so the
/// session it races is polled however fast the program writes.
async fn until_exit(
    wake: &mut NoticeRx<'_, TrapTransport>,
    output: Option<&'static crate::stdio::Endpoint>,
    wake_main: &dyn Fn(),
) {
    let ended = || EXIT.posted().is_some();
    while !ended() {
        if let Some(output) = output {
            crate::stdio::drain(output, wake_main).await;
            executor::yield_now().await;
        }
        crate::stdio::arm();
        if park(wake, || ended() || crate::stdio::pending())
            .await
            .is_err()
        {
            core::future::pending::<()>().await;
        }
    }
}

/// The program's filesystem: attached on first use, one native owner for the
/// program's life.
struct Filesystem<'r, 't> {
    rt: &'r Rt,
    transport: &'t TrapTransport,
    session: Option<Session<'static, 't>>,
    /// The launch carried no face, or the owner could not be formed: every
    /// later session is refused the same way without asking again.
    refused: Option<(KoreStatus, KoreFsError)>,
}

impl<'r, 't> Filesystem<'r, 't> {
    fn new(rt: &'r Rt, transport: &'t TrapTransport) -> Self {
        Self {
            rt,
            transport,
            session: None,
            refused: None,
        }
    }

    async fn run(&mut self, step: KoreFsStep, user: *mut c_void) -> (KoreStatus, KoreFsError) {
        if let Some(refused) = self.refused {
            return refused;
        }
        if self.session.is_none() {
            let Some(fs) = Fs::attach(self.rt, self.transport).await else {
                let refused = (KoreStatus::Refused, KoreFsError::Unsupported);
                self.refused = Some(refused);
                return refused;
            };
            match Session::new(fs) {
                Ok(session) => self.session = Some(session),
                Err(_) => {
                    let refused = (KoreStatus::Capacity, KoreFsError::None);
                    self.refused = Some(refused);
                    return refused;
                }
            }
        }
        let Some(session) = self.session.as_mut() else {
            return (KoreStatus::Refused, KoreFsError::None);
        };
        // SAFETY: `kore_fs_run`'s caller took on `Session::run`'s contract for
        // `step` and `user`, and waits until this future completes or is
        // dropped with the program's end.
        match unsafe { session.run(step, user) }.await {
            Ok(()) => (KoreStatus::Ok, KoreFsError::None),
            Err(error) => error.foreign(),
        }
    }
}
