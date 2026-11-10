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
//!   faces, and serves the program's filesystem session
//!   ([`kore_fs_run`](crate::api::kore_fs_run)) across `kore-rt`'s bridge —
//!   the exchange `kore-lua`'s VM and `tcc` speak too: the main thread is the
//!   bridge's worker, and its one work item is the program.
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
//!  serve: park on the wake channel             … kore_fs_run(step, user)
//!  ◀──────────────────────────── the call ──── parked on its doorbell
//!  run the session (its C callbacks run here)
//!  the answer ───────────────────────────────▶ returns the session's status
//!                                              … exit(status): atexit handlers
//!  ◀─────────── the exit word; the bridge's ── the thread ends
//!  join it, raise the barrier, finalize
//!  end(status): the C library's exit
//! ```
//!
//! The bridge owns the word, the park-and-kick ordering, the session cell's
//! hand-over, and the main thread's end: a main thread that panics or loses
//! its park is reported gone, never waited for. What stays here is the
//! program's lifecycle, which no other engine shares: the **exit word**, which
//! any thread of the program may post — every thread it starts is founded
//! holding a send into the host's wake channel (`thread_controller`), so its
//! kick reaches the host's park — and the decision, once the program ends,
//! whether the main thread is ending too (then the host joins it, so the
//! barrier is clean and the finalizers run) or still running (then the
//! barrier stops it).
//!
//! A launch that cannot carry the host (no channel space or heap, a stack
//! window that cannot place the main thread) runs the program on thread 0, as
//! the C library alone would: allocation, threads, time and random work, a
//! filesystem session is refused, and only thread 0's exit ends the program.

use core::cell::Cell;
use core::ffi::{c_int, c_void};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use kore_ffi::{KoreFsError, KoreFsStep, KoreStatus, Session};

use crate::exit::{Exit, ExitWord};
use kore_std::ExitCode;
use kore_std::fs::Fs;
use kore_std::runtime::bridge::{Bridge, Engine, Gone, Settled, Work, WorkerHost};
use kore_std::runtime::entry::{self, Rt};
use kore_std::runtime::executor::{self, Executor};
use kore_std::runtime::thread::{self, WorkerStack};
use kore_std::runtime::transport::TrapTransport;
use kore_std::runtime::{crash, startup};

/// The C library's program: runs `main`, then `exit`s with its value.
pub type Program = unsafe extern "C" fn() -> !;
/// The C library's exit: commits `status` on the calling thread's own handle.
pub type End = unsafe extern "C" fn(c_int) -> !;

/// The program, as [`__kore_program_start`] received it, for the main thread.
static PROGRAM: AtomicUsize = AtomicUsize::new(0);
/// Set by the first [`__kore_program_start`]; a second never founds a host.
static STARTED: AtomicBool = AtomicBool::new(false);

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

// ── The end ─────────────────────────────────────────────────────────────

/// The program's end, once asked for (see [`ExitWord`]).
static EXIT: ExitWord = ExitWord::new();

/// The end the host reaches when it cannot read one the program asked for.
const INTERNAL: Exit = Exit {
    status: ExitCode::INTERNAL.get(),
    finalize: false,
    from_main: false,
};

// ── The session ─────────────────────────────────────────────────────────

/// What crosses the bridge: the session the main thread asks for, and how it
/// ended.
struct Payload {
    // main thread → host: the session's callback program.
    step: Option<KoreFsStep>,
    user: *mut c_void,
    // host → main thread: how the session ended.
    status: KoreStatus,
    error: KoreFsError,
}

// SAFETY: `user` crosses under the bridge's hand-over: `kore_fs_run`'s caller
// keeps whatever it names alive for the session, and the main thread is
// parked inside the call until the host answers it.
unsafe impl Send for Payload {}

static SESSION: Bridge<Payload> = Bridge::new(Payload {
    step: None,
    user: core::ptr::null_mut(),
    status: KoreStatus::Ok,
    error: KoreFsError::None,
});

/// The main thread as the bridge's worker: its one work item is the program,
/// which ends the process rather than finishing.
struct Main;

impl Engine for Main {
    type Payload = Payload;

    fn bridge() -> &'static Bridge<Payload> {
        &SESSION
    }

    fn run(_program: Work<Payload>) {
        ROLE.set(Role::Main);
        // SAFETY: `__kore_program_start` stored a `Program` before founding
        // this thread, and the founding's session ordered that store before
        // this load.
        let program: Program = unsafe { core::mem::transmute(PROGRAM.load(Ordering::Acquire)) };
        unsafe { program() }
    }
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
    // Lives as long as this frame, which never returns.
    let transport = TrapTransport::new();
    match Founded::found(&transport) {
        Some(host) => host.serve(&transport, end),
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
        // or finalizer the host is running. The barrier is this thread's.
        Role::Host => startup::barrier(finalize),
        role if SESSION.is_hosted() => {
            let _ = EXIT.post(Exit {
                status,
                finalize,
                from_main: role == Role::Main,
            });
            // The main thread ends as the bridge's worker, which kicks the
            // host and lets it join this thread; any other thread kicks. A
            // refused kick found no post standing, and the host re-reads both
            // words before it parks: the stores above reach it either way.
            if !SESSION.leave() {
                let _ = thread::kick_parent();
            }
            thread::exit(ExitCode::OK)
        }
        // No host: only thread 0 owns the process. Anywhere else the barrier
        // refuses and ends this thread alone.
        _ => startup::barrier(finalize),
    }
}

/// The main thread's half of [`kore_fs_run`](crate::api::kore_fs_run): post
/// the session to the host and wait for its answer. Refused anywhere but the
/// main thread of a hosted program.
///
/// # Safety
/// `kore_fs_run`'s contract for `step` and `user`.
pub(crate) unsafe fn session(step: KoreFsStep, user: *mut c_void) -> (KoreStatus, KoreFsError) {
    SESSION
        .call(
            |cell| {
                cell.step = Some(step);
                cell.user = user;
            },
            |cell| {
                cell.step = None;
                cell.user = core::ptr::null_mut();
                (cell.status, cell.error)
            },
        )
        .unwrap_or((KoreStatus::Refused, KoreFsError::None))
}

/// This library's panic: the record first, then the end. A panic on thread 0
/// stops the program there; anywhere else it asks the host to, so the process
/// ends with the panic rather than running on without the panicking thread.
/// The main thread's panic also reaches the host as the bridge's word, which
/// `crash::report` publishes once the record is out.
pub(crate) fn panicked(info: &core::panic::PanicInfo<'_>) -> ! {
    crash::report(info);
    match ROLE.get() {
        Role::Host => startup::barrier(false),
        role if SESSION.is_hosted() => {
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

// ── The host ────────────────────────────────────────────────────────────

/// Thread 0 with its program founded: the runtime context, the executor's
/// doorbell, and the main thread.
struct Founded<'t> {
    rt: Rt,
    doorbell: executor::Doorbell,
    main: WorkerHost<'t, Main>,
}

impl<'t> Founded<'t> {
    /// Found the main thread. `None` when the launch cannot carry the host.
    fn found(transport: &'t TrapTransport) -> Option<Self> {
        let rt = entry::thread_context()?;
        let doorbell = rt.doorbell()?;
        let main = WorkerHost::<Main>::spawn(
            transport,
            WorkerStack::Bytes(crate::api::KORE_MAIN_STACK_BYTES),
            &[],
            |_| {},
        )?;
        Some(Self { rt, doorbell, main })
    }

    /// Serve the program until it ends, then end the process.
    fn serve(self, transport: &TrapTransport, end: End) -> ! {
        let Self { rt, doorbell, main } = self;
        let exit = {
            let mut executor = Executor::new(entry::transport(), doorbell);
            executor.run(host(&rt, transport, main))
        };
        startup::barrier(exit.finalize);
        // SAFETY: the C library's exit, as `__kore_program_start` received it.
        unsafe { end(exit.status) }
    }
}

/// The host's loop: answer each session the main thread asks for, until any
/// thread asks for the end; then join the main thread if it is ending too, so
/// the barrier is clean, or leave it to the barrier if it is not.
async fn host(rt: &Rt, transport: &TrapTransport, mut main: WorkerHost<'_, Main>) -> Exit {
    let mut filesystem = Filesystem::new(rt, transport);
    // An end requested mid-session ends the program without waiting for the
    // session: the barrier stops the waiting main thread, and the abandoned
    // session's owner is quarantined, as cancellation leaves it.
    let served = main
        .serve_until(
            async |cell: &mut Payload| {
                let (status, error) = match cell.step {
                    Some(step) => filesystem.run(step, cell.user).await,
                    // A session posted without its program: nothing to run.
                    None => (KoreStatus::Refused, KoreFsError::None),
                };
                cell.status = status;
                cell.error = error;
            },
            || EXIT.posted().is_some(),
        )
        .await;
    let (exit, joined) = match served {
        // A thread asked for the end. If it was the main thread, that thread
        // is ending: the bridge reads it gone once it has.
        Ok(None) => match EXIT.posted() {
            Some(exit) => (exit, exit.from_main),
            None => (INTERNAL, false),
        },
        // The main thread is gone: it asked for the end first, or it panicked
        // (its record is published), or it lost its park.
        Ok(Some(Settled::Gone(gone))) => {
            let status = match gone {
                Gone::Panicked => ExitCode::PANIC.get(),
                _ => ExitCode::INTERNAL.get(),
            };
            let exit = EXIT.posted().unwrap_or(Exit {
                status,
                finalize: false,
                from_main: true,
            });
            (exit, true)
        }
        // The program never finishes its item; and a host that cannot hear
        // the main thread cannot tell whether it is ending.
        Ok(Some(Settled::Idle)) | Err(_) => (INTERNAL, false),
    };
    if joined {
        // The main thread asked and is ending itself: wait for it, so a
        // program that ran on it alone is stopped cleanly.
        let _ = main.join().await;
    } else {
        // Still running: the barrier stops it, and its channels end with the
        // process. Its join ticket must not try to reclaim it first.
        let _ = main.abandon();
    }
    exit
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
