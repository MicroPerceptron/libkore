//! The VM's thread and the host-call bridge (`res/DAGFS.md` §4).
//!
//! Lua is a synchronous C engine: `require` must return a module to the
//! C frame that called it, and that frame may sit inside a coroutine, a
//! metamethod, or a `pcall`.  The store, however, is reached only through
//! the task's async filesystem client on its executor.  So the VM runs on a
//! kore-rt **worker thread** and the executor stays on thread 0; the two
//! share one request cell and one word:
//!
//! ```text
//!  thread 0 (executor)                       worker (VM)
//!  ───────────────────                       ───────────
//!  write eval → EVAL_POSTED, kick ──────────▶ wait(EVAL_DONE) returns
//!  park on the wake channel                   runs the chunk …
//!                                             … C `require` → write op
//!  ◀──────── CALL_POSTED, kick_parent ─────── wait(CALL_POSTED)
//!  serve the op (async)
//!  write reply → CALL_SERVED, kick ─────────▶ returns; EVAL_POSTED
//!  park on the wake channel
//!                                             … chunk ends
//!  ◀──────── EVAL_DONE, kick_parent ────────  wait(EVAL_DONE)
//!  take stdout/stderr
//! ```
//!
//! **One transport, many ops.**  `require`, `kore.time.sleep`, and every
//! later verb that needs a service — this crate's own or an embedder's,
//! through [`Extension`] — are the same round trip with a different
//! payload, so they share one word-state pair
//! (`CALL_POSTED` / `CALL_SERVED`), one worker-side helper ([`host_call`]),
//! and one thread-0 dispatcher ([`serve_call`]).  Adding a verb is a new op
//! constant and a new arm — never a new state, and never another copy of
//! the ordering discipline below.  Verbs that need *no* service do not come
//! here at all: `kore.time.now` reads the clock page and `kore.random`
//! draws from the CPU, both directly on the VM thread.
//!
//! Every park is a doorbell park; nothing spins.  The worker parks with
//! `futex::wait` on the word (its own doorbell); thread 0 parks on the
//! **wake channel** the worker holds a send to (`DoorbellWorker::wake_recv`,
//! `executor::next_wake_on`) — a launched task has no self-kick for a
//! worker to inherit, so `futex::wake` could not reach it.  Lost wakeups
//! are excluded by discipline, not luck: the worker stores the word, then
//! kicks; thread 0 stands its post, then re-loads the word, then parks.  A
//! kick that found no standing post was refused, and that ordering means
//! the store it announced is visible to the re-load.  The
//! word's holder owns the cell: the worker while it reads `EVAL_POSTED` /
//! `CALL_SERVED` / `QUIT`, thread 0 while it reads `EVAL_DONE` /
//! `CALL_POSTED` / `WORKER_GONE`.  Buffers cross the boundary as owned
//! `Vec`s — the heap frees a block from any thread onto its owning shard's
//! remote list — or as pointers into a frame the owner keeps alive across
//! the whole phase (the eval's script).
//!
//! Resolution order (`SEARCH`): the task's own tree, then `libs/`; a source
//! module `<name>.lua` wins over a native `<name>.so`.  Native modules are
//! store objects stamped `EXECUTABLE_IMAGE` that `kore-late` checks, places
//! in the JIT arena, and links against this executable's exported C API.

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::ffi::{CStr, c_char, c_int};
use core::sync::atomic::{AtomicU32, Ordering};

use kore::abi::clock::TimerError;
use kore::abi::session::Status;
use kore::abi::svc::FS_NAME_BYTES;
use kore::fs::Fs;
use kore::runtime::{
    chan::{NoticeRx, RecvError},
    entry::Rt,
    futex::{self, WaitOutcome},
    thread::{self, DoorbellWorker},
    transport::TrapTransport,
};
use kore_late::{LoadError, Loader, Search};

use crate::vm::{EvalResult, ReplResult, Vm};
use crate::{Answer, Extension, OP_EMBEDDER_BASE};

/// The bound on a source module (`<name>.lua`).  Sized like a generous
/// script, not like a library: the whole buffer is staged in the task's
/// 2 MiB launch heap beside the VM, and a module is read whole before the
/// engine sees it.  Larger sources belong in the store as several modules.
pub const MAX_SOURCE_MODULE_BYTES: usize = 64 * 1024;

/// Where `require` looks, in order: the task's own tree (its cwd view),
/// then the READ-only `libs/` view every task carries.
const SEARCH: &[&[u8]] = &[b"", b"libs"];

/// The prefix a native module's opener carries: `luaopen_<name>`.
const OPENER_PREFIX: &[u8] = b"luaopen_";

// The word's states.  See the module doc for who owns the cell in each.
const EVAL_DONE: u32 = 0;
const EVAL_POSTED: u32 = 1;
/// The worker posted a host call — a `require`, a sleep, an embedder's
/// library call, any operation that needs thread 0's executor.  Thread 0
/// owns the cell until it stores [`CALL_SERVED`].  One pair serves every
/// op: the VM is a synchronous C engine and every service face lives on
/// thread 0, so the transport is identical and only the payload differs.
const CALL_POSTED: u32 = 2;
const CALL_SERVED: u32 = 3;
const QUIT: u32 = 4;
const WORKER_GONE: u32 = 5;

/// The host-call op vocabulary, as the C side names them (`kore_lua.c`).
/// One flat namespace over one transport; a new standard-library verb is a
/// new constant and a new `serve_call` arm, never a new word state.  An
/// embedder's numbers start at [`OP_EMBEDDER_BASE`].
///
/// `require "<name>"` — the module name is in `name`, not `args`, and the
/// reply is a chunk or an opener rather than words.
const OP_REQUIRE: u32 = 7;
/// `kore.time.sleep(seconds)` — arg0 is the delay in nanoseconds.
const OP_SLEEP: u32 = 16;
/// `kore.proc.args` — the launch argument bytes, answered as text.
const OP_PROC_ARGS: u32 = 17;

// `kore.fs`, and so `io` and the filesystem half of `os` — one verb each,
// all over the one transport.  `arg0` is a handle wherever a verb takes
// one; `input` carries a path or the bytes of a write.  Every verb is
// positional: the stream position and its read-ahead belong to the file
// object a script holds, which is in C because `io.open` and
// `kore.fs.open` answer the same object (`crate::files`).
const OP_FS_OPEN: u32 = 18;
const OP_FS_CLOSE: u32 = 19;
const OP_FS_READ: u32 = 20;
const OP_FS_WRITE: u32 = 21;
const OP_FS_SYNC: u32 = 23;
const OP_FS_RESIZE: u32 = 24;
const OP_FS_STAT: u32 = 25;
const OP_FS_REMOVE: u32 = 26;
const OP_FS_RENAME: u32 = 27;
const OP_FS_LIST: u32 = 28;
/// `os.exit(code)` — records the code, which the embedder reads once the
/// evaluation has unwound.  Recording rather than exiting is the point: the
/// VM still has C frames below it and `<close>` handlers above it, and
/// leaving from here would run neither.
const OP_PROC_EXIT: u32 = 29;
/// `kore.fs.read_file(path)` — a whole file, opened, streamed and closed
/// here.  The one filesystem verb that is *not* positional, and
/// deliberately so: only a read the host performs end to end can hold one
/// pinned version for the whole file, or hand the facade a destination
/// large enough to stream through the storage ring.  Numbered past
/// `OP_PROC_EXIT` rather than filling the retired 22, because a recycled op
/// number is one a stale half of the bridge would answer wrongly.
const OP_FS_READ_FILE: u32 = 30;

/// Host-call outcomes: the value(s) are good; the service refused (the
/// text names the refusal); the call could not be made at all (the text
/// says why — no face, no lease, a broken ring).
///
/// Every op answers one of these three, so a script that learned one
/// module's error handling has learned them all.
const STATUS_OK: c_int = 0;
const STATUS_REFUSED: c_int = 1;
const STATUS_UNAVAILABLE: c_int = 2;

const KIND_CHUNK: u32 = 0;
const KIND_REPL: u32 = 1;

const REPLY_ERROR: c_int = 0;
const REPLY_SOURCE: c_int = 1;
const REPLY_NATIVE: c_int = 2;

/// `lua_CFunction`.
type LuaCFunction = unsafe extern "C" fn(*mut c_void) -> c_int;
use core::ffi::c_void;

/// How one evaluation ended, as the VM reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Failed,
    /// The chunk ran `os.exit`. The sentinel it raised unwound every
    /// `<close>` handler on the way out, so this is a completed evaluation
    /// and not a failure; the code is in
    /// [`Services::exit_request`]. A batch run leaves with it and a REPL
    /// stops reading.
    Exited,
    /// REPL only: the chunk needs a continuation line.
    Incomplete,
    /// REPL only: the expression buffer could not be allocated.
    ResourceRefused,
    /// The worker could not construct a VM.
    Startup,
}

/// The bridge is broken: the worker is gone, a kick was refused, or a park
/// is impossible.  The host exits; nothing here is retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fatal;

struct Payload {
    // thread 0 → worker: the evaluation.
    kind: u32,
    script: *const u8,
    script_len: usize,
    chunk_name: *const u8,
    chunk_name_len: usize,
    // worker → thread 0: its result.
    outcome: Outcome,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    // worker → thread 0: the host call.  `op` selects the arm; `args` and
    // `name` carry its arguments (a `require` names a module, the value
    // ops pass words).  One shape because one transport serves them all.
    op: u32,
    args: [u64; 3],
    name: [u8; FS_NAME_BYTES],
    name_len: usize,
    // worker → thread 0: the call's byte argument — a path, the bytes a
    // write is handing over.  A *borrow* into the Lua frame, not a copy:
    // the worker is parked for the whole call, so the string the argument
    // names cannot move or be collected, and a `Vec` here would copy every
    // written buffer twice for nothing (`res/SYSTEM_LIBRARY.md`).  Null
    // and zero when the op takes no bytes.
    input: *const u8,
    input_len: usize,
    // thread 0 → worker: a `require`'s answer.  Its reply is a function
    // pointer or owned bytes, not words, so it keeps its own fields — that
    // shape is the native-module ABI (`res/IMAGE.md` §8), not a detail.
    reply: c_int,
    source: Vec<u8>,
    opener: Option<LuaCFunction>,
    error: *const c_char,
    // thread 0 → worker: every other op's outcome.  `text` is the refusal,
    // reason, or byte answer, alive until the next call (C copies it into
    // a Lua string before returning).
    status: c_int,
    results: [u64; 2],
    text: Vec<u8>,
}

struct Cell {
    state: AtomicU32,
    payload: UnsafeCell<Payload>,
}

// SAFETY: `payload` is touched only by the thread the word's current state
// hands the cell to (module doc); every hand-over is a Release store on
// `state` matched by the other side's Acquire load.
unsafe impl Sync for Cell {}

static CELL: Cell = Cell {
    state: AtomicU32::new(EVAL_DONE),
    payload: UnsafeCell::new(Payload {
        kind: KIND_CHUNK,
        script: core::ptr::null(),
        script_len: 0,
        chunk_name: core::ptr::null(),
        chunk_name_len: 0,
        outcome: Outcome::Ok,
        stdout: Vec::new(),
        stderr: Vec::new(),
        op: 0,
        args: [0; 3],
        name: [0; FS_NAME_BYTES],
        name_len: 0,
        input: core::ptr::null(),
        input_len: 0,
        reply: REPLY_ERROR,
        source: Vec::new(),
        opener: None,
        error: core::ptr::null(),
        status: STATUS_UNAVAILABLE,
        results: [0; 2],
        text: Vec::new(),
    }),
};

/// The cell, for the side the word currently hands it to.
///
/// # Safety
/// The caller has observed (Acquire) a state that makes it the owner, and
/// holds no other reference across the store that hands the cell back.
unsafe fn payload<'a>() -> &'a mut Payload {
    unsafe { &mut *CELL.payload.get() }
}

// ---------------------------------------------------------------- worker

/// The worker's entry: build the VM, then serve evaluations until `QUIT`.
fn vm_thread<E: Extension>() {
    let Some(mut vm) = Vm::new(E::LIBRARIES) else {
        // SAFETY: the worker owns the cell while the word reads EVAL_DONE
        // and thread 0 has not yet posted (it posts only after spawn).
        unsafe { payload().outcome = Outcome::Startup };
        CELL.state.store(WORKER_GONE, Ordering::Release);
        let _ = thread::kick_parent();
        return;
    };
    loop {
        match CELL.state.load(Ordering::Acquire) {
            EVAL_POSTED => {
                run_eval(&mut vm);
                CELL.state.store(EVAL_DONE, Ordering::Release);
                let _ = thread::kick_parent();
            }
            QUIT => {
                drop(vm);
                CELL.state.store(WORKER_GONE, Ordering::Release);
                let _ = thread::kick_parent();
                return;
            }
            WORKER_GONE => return,
            other => {
                if futex::wait(&CELL.state, other, None) == WaitOutcome::Unsupported {
                    // No doorbell to park on: a spin would be the only
                    // alternative, and that is the one thing this bridge
                    // refuses to be.  Leave; thread 0 reads WORKER_GONE.
                    drop(vm);
                    CELL.state.store(WORKER_GONE, Ordering::Release);
                    let _ = thread::kick_parent();
                    return;
                }
            }
        }
    }
}

fn run_eval(vm: &mut Vm) {
    // SAFETY: EVAL_POSTED — the worker owns the cell; thread 0 keeps the
    // script and chunk name alive until it reads EVAL_DONE.
    let cell = unsafe { payload() };
    let script = if cell.script.is_null() {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(cell.script, cell.script_len) }
    };
    cell.outcome = match cell.kind {
        KIND_REPL => match vm.eval_repl(script) {
            ReplResult::Complete(true) => Outcome::Ok,
            ReplResult::Complete(false) => Outcome::Failed,
            ReplResult::Exited => Outcome::Exited,
            ReplResult::Incomplete => Outcome::Incomplete,
            ReplResult::ResourceRefused => Outcome::ResourceRefused,
        },
        _ => {
            // SAFETY: thread 0 posts a non-empty `&'static [u8]` chunk
            // name with every KIND_CHUNK evaluation, so this is never the
            // dangling-null case `from_raw_parts` forbids.
            let name: &'static [u8] =
                unsafe { core::slice::from_raw_parts(cell.chunk_name, cell.chunk_name_len) };
            match vm.eval(script, name) {
                EvalResult::Ok => Outcome::Ok,
                EvalResult::Failed => Outcome::Failed,
                EvalResult::Exited => Outcome::Exited,
            }
        }
    };
    cell.stdout = vm.take_stdout();
    cell.stderr = vm.take_stderr();
}

/// The transport every host call shares: write the request, hand the cell
/// to thread 0, park on the doorbell, read the response back, and resume
/// the evaluation.  One function owns the whole ordering so no caller can
/// get it wrong, and a new standard-library verb is a `write`/`read` pair
/// rather than another copy of this discipline.
///
/// `Err` means the bridge is gone and nothing was served; the cell is not
/// handed back and the evaluation is not resumed.
///
/// # Safety
/// Worker thread only, inside an evaluation (the word reads `EVAL_POSTED`).
unsafe fn host_call<T>(
    write: impl FnOnce(&mut Payload),
    read: impl FnOnce(&Payload) -> T,
) -> Result<T, &'static CStr> {
    if CELL.state.load(Ordering::Acquire) != EVAL_POSTED {
        return Err(c"host call outside an evaluation");
    }
    // SAFETY: EVAL_POSTED — the worker owns the cell.
    write(unsafe { payload() });
    CELL.state.store(CALL_POSTED, Ordering::Release);
    let _ = thread::kick_parent();
    loop {
        match CELL.state.load(Ordering::Acquire) {
            CALL_SERVED => break,
            WORKER_GONE | QUIT => return Err(c"host is shutting down"),
            other => {
                if futex::wait(&CELL.state, other, None) == WaitOutcome::Unsupported {
                    return Err(c"worker cannot park");
                }
            }
        }
    }
    // SAFETY: CALL_SERVED — the worker owns the cell again.
    let value = read(unsafe { payload() });
    // Back inside the evaluation.  No kick: thread 0 does not wait on this
    // word — it parks on its wake doorbell (`park_until_kicked`) and reads
    // `CALL_SERVED` and `EVAL_POSTED` alike as "the worker is busy".  It
    // has nothing to do until the worker's next post, and both of those
    // (`CALL_POSTED`, `EVAL_DONE`) kick.
    CELL.state.store(EVAL_POSTED, Ordering::Release);
    Ok(value)
}

/// C `require`'s half (`kore_lua.c`): post the name, park until served,
/// hand back what thread 0 stored.  Called on the worker, inside an
/// evaluation, so the word reads `EVAL_POSTED` on entry.
///
/// # Safety
/// `name` points at `len` bytes; the out-pointers are valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_host_require(
    name: *const c_char,
    len: usize,
    bytes: *mut *const c_char,
    bytes_len: *mut usize,
    opener: *mut Option<LuaCFunction>,
) -> c_int {
    let fail = |message: &'static CStr| -> c_int {
        unsafe {
            *bytes = message.as_ptr();
            *bytes_len = message.count_bytes();
            *opener = None;
        }
        REPLY_ERROR
    };
    if name.is_null() || len == 0 || len > FS_NAME_BYTES {
        return fail(c"module name is not a store name");
    }
    let requested = unsafe { core::slice::from_raw_parts(name.cast::<u8>(), len) };
    let served = unsafe {
        host_call(
            |cell| {
                cell.op = OP_REQUIRE;
                cell.name[..len].copy_from_slice(requested);
                cell.name_len = len;
                cell.reply = REPLY_ERROR;
                cell.opener = None;
                cell.error = core::ptr::null();
            },
            |cell| match cell.reply {
                REPLY_SOURCE => (
                    REPLY_SOURCE,
                    cell.source.as_ptr().cast(),
                    cell.source.len(),
                    None,
                ),
                REPLY_NATIVE => (REPLY_NATIVE, core::ptr::null(), 0, cell.opener),
                other => (other, cell.error, 0, None),
            },
        )
    };
    let (reply, reply_bytes, reply_len, reply_opener) = match served {
        Ok(served) => served,
        Err(message) => return fail(message),
    };
    unsafe {
        *bytes = reply_bytes;
        *bytes_len = reply_len;
        *opener = reply_opener;
    }
    reply
}

// ------------------------------------------------ synchronous leaf calls
//
// Verbs that need no service at all: a clock-page projection and a CPU
// entropy draw.  These run straight on the VM thread.  Crossing the bridge
// for them would buy a park, a kick and a scheduling round trip in exchange
// for nothing — the answer is already readable from here.

/// Why a clock read could not be answered.  An uncalibrated clock is a
/// refusal, never a zero timestamp: `kore.time` would otherwise report the
/// monotonic epoch as a successful reading.
fn clock_text(error: TimerError) -> &'static CStr {
    match error {
        TimerError::Unsupported => c"this machine exposes no user-readable counter",
        TimerError::Unconfigured => c"the clock is not configured",
        TimerError::Uncalibrated => c"the clock is not calibrated yet",
        TimerError::WrongCpu => c"the clock belongs to another CPU",
        TimerError::ControllerUnavailable => c"the timer controller is unavailable",
        TimerError::OutOfRange => c"the reading does not fit the counter width",
    }
}

/// Answer one `TimerResult` into the C out-parameters.
///
/// # Safety
/// Both out-pointers are valid for writes.
unsafe fn answer_clock(
    reading: Result<u64, TimerError>,
    out: *mut u64,
    text: *mut *const c_char,
) -> c_int {
    match reading {
        Ok(nanoseconds) => {
            unsafe {
                *out = nanoseconds;
                *text = core::ptr::null();
            }
            STATUS_OK
        }
        Err(error) => {
            let message = clock_text(error);
            unsafe {
                *out = 0;
                *text = message.as_ptr();
            }
            STATUS_UNAVAILABLE
        }
    }
}

/// `kore.time.now` / `kore.time.mono_ns`: nanoseconds since the monotonic
/// epoch, projected from this thread's calibration page.
///
/// # Safety
/// Both out-pointers are valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_host_uptime_ns(out: *mut u64, text: *mut *const c_char) -> c_int {
    unsafe {
        answer_clock(
            kore::runtime::clock::uptime().map(|at| at.as_nanos()),
            out,
            text,
        )
    }
}

/// `kore.time.wall`: nanoseconds since the UNIX epoch.  Separate from the
/// monotonic read because the two fail independently — a machine can have a
/// good counter and no wall anchor.
///
/// # Safety
/// Both out-pointers are valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_host_wall_ns(out: *mut u64, text: *mut *const c_char) -> c_int {
    unsafe {
        answer_clock(
            kore::runtime::clock::wall().map(|at| at.as_nanos()),
            out,
            text,
        )
    }
}

/// `kore.random.bytes`: fill `len` bytes from the CPU's entropy source.
///
/// # Safety
/// `dst` points at `len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_host_random(
    dst: *mut u8,
    len: usize,
    text: *mut *const c_char,
) -> c_int {
    // Answer the empty draw before touching `dst`: a zero-length Lua buffer
    // may hand us a null pointer, and `from_raw_parts_mut` demands a
    // non-null one even for an empty slice.  Zero bytes drawn is a success.
    if len == 0 {
        unsafe { *text = core::ptr::null() };
        return STATUS_OK;
    }
    if dst.is_null() {
        unsafe { *text = c"no destination".as_ptr() };
        return STATUS_UNAVAILABLE;
    }
    let buffer = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    match kore::runtime::random::fill(buffer) {
        Ok(()) => {
            unsafe { *text = core::ptr::null() };
            STATUS_OK
        }
        Err(_) => {
            // Leave nothing half-drawn for a caller that ignores the status.
            buffer.fill(0);
            unsafe { *text = c"this machine exposes no entropy source".as_ptr() };
            STATUS_UNAVAILABLE
        }
    }
}

/// `kore.random.available`: whether a draw can succeed at all, so a caller
/// can decide before doing work whose result it would have to throw away.
#[unsafe(no_mangle)]
pub extern "C" fn kore_lua_host_random_available() -> c_int {
    c_int::from(kore::runtime::random::available())
}

/// The VM's seed (`luai_makeseed`, and `lua_newstate`'s own argument).
///
/// It seeds string hashing, table iteration order, and — through
/// `luaL_makeseed` — `math.random`'s initial state.  Drawing it from
/// hardware is what makes stock `math.random` behave the way every Lua
/// program assumes; the fixed constant it replaces produced one identical
/// sequence on every boot.  A machine with no entropy source keeps that
/// constant, because a predictable seed is still a working VM.
#[unsafe(no_mangle)]
pub extern "C" fn kore_lua_host_seed() -> u32 {
    seed()
}

/// The seed a machine with no entropy source gets: ASCII `"KORE"`.
///
/// This was the *only* seed before hardware entropy was wired in, so every
/// boot produced one identical `math.random` sequence and one identical
/// table iteration order.  It survives as the fallback because a predictable
/// VM is still a working VM — a REPL should not fail to start because the
/// CPU exposes no generator.  `kore_lua.c`'s weak `kore_lua_host_seed` keeps
/// the same value so a host smoke build reproduces target behaviour.
const UNSEEDED: u32 = 0x4b4f_5245;

/// [`kore_lua_host_seed`] for the Rust side — `lua_newstate` takes the same
/// word as an argument, so both spellings draw from one place.
pub fn seed() -> u32 {
    kore::runtime::random::u64().map_or(UNSEEDED, |word| word as u32)
}

/// Release the source buffer a `REPLY_SOURCE` handed to C, once the chunk
/// is compiled.  Worker-side, inside the evaluation.
#[unsafe(no_mangle)]
pub extern "C" fn kore_lua_host_release() {
    if CELL.state.load(Ordering::Acquire) == EVAL_POSTED {
        // SAFETY: EVAL_POSTED — the worker owns the cell.
        let cell = unsafe { payload() };
        cell.source = Vec::new();
    }
}

/// Every value-returning host call's half (`kore_lua.c`): post the op and
/// its words, park until served, hand back the outcome.  Called on the
/// worker, inside an evaluation.
///
/// Answers one of [`STATUS_OK`], [`STATUS_REFUSED`], [`STATUS_UNAVAILABLE`],
/// up to two result words, and a text that lives until the next call (C
/// copies it into a Lua string before returning).
///
/// # Safety
/// The out-pointers are valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_host_call(
    op: u32,
    arg0: u64,
    arg1: u64,
    arg2: u64,
    input: *const u8,
    input_len: usize,
    result: *mut u64,
    result2: *mut u64,
    text: *mut *const c_char,
    text_len: *mut usize,
) -> c_int {
    let fail = |message: &'static CStr| -> c_int {
        unsafe {
            *result = 0;
            *result2 = 0;
            *text = message.as_ptr();
            *text_len = message.count_bytes();
        }
        STATUS_UNAVAILABLE
    };
    let served = unsafe {
        host_call(
            |cell| {
                cell.op = op;
                cell.args = [arg0, arg1, arg2];
                cell.input = input;
                cell.input_len = if input.is_null() { 0 } else { input_len };
                cell.status = STATUS_UNAVAILABLE;
                cell.results = [0; 2];
                cell.text = Vec::new();
            },
            |cell| {
                (
                    cell.status,
                    cell.results,
                    cell.text.as_ptr().cast::<c_char>(),
                    cell.text.len(),
                )
            },
        )
    };
    let (status, results, reply_text, reply_len) = match served {
        Ok(served) => served,
        Err(message) => return fail(message),
    };
    unsafe {
        *result = results[0];
        *result2 = results[1];
        *text = reply_text;
        *text_len = reply_len;
    }
    status
}

// -------------------------------------------------------------- thread 0

/// What thread 0 reaches the store with, attached lazily on the first
/// `require` (a chunk that never requires never opens the filesystem).
pub struct Services<'r, 't> {
    rt: &'r Rt,
    transport: &'t TrapTransport,
    /// The launch arguments, as `main` read them from ProcessIo. Owned
    /// because the VM may ask for them at any point in a long REPL, long
    /// after the caller's own borrow would have ended.
    args: Vec<u8>,
    fs: Option<Fs<'r, 't>>,
    loader: Option<Loader>,
    /// The exit code `os.exit` asked for, if a script asked.  Read by the
    /// embedder after the evaluation returns — `os.exit` cannot leave from
    /// inside the VM without skipping every `<close>` handler above it.
    exit_request: Option<i32>,
    /// The VM's open files.  Beside the session rather than inside it:
    /// `Fs` is the store's client and knows nothing of streams, and a
    /// handle must not outlive the session that could close it.
    files: crate::files::Files,
}

impl<'r, 't> Services<'r, 't> {
    pub fn new(rt: &'r Rt, transport: &'t TrapTransport, args: Vec<u8>) -> Self {
        Self {
            rt,
            transport,
            args,
            fs: None,
            loader: None,
            exit_request: None,
            files: crate::files::Files::default(),
        }
    }

    /// The exit code a script asked for with `os.exit`, if it did.
    ///
    /// The embedder checks this after each evaluation, which
    /// [`Outcome::Exited`] tells it to do: a REPL stops reading and a batch
    /// run leaves with it.
    pub fn exit_request(&self) -> Option<i32> {
        self.exit_request
    }

    /// Close every file the VM still holds.
    ///
    /// The REPL's teardown calls this: a script that ended without
    /// `:close()` has left session handles open, and the store must get
    /// them back even though the `__gc` that would have returned them will
    /// never run once the VM is gone.
    pub async fn close_open_files(&mut self) {
        if self.files.is_empty() {
            return;
        }
        let views = self.files.drain();
        let Some(fs) = self.fs.as_mut() else {
            return;
        };
        for view in views {
            fs.close(view).await;
        }
    }

    /// The filesystem client, attached on first use.
    pub async fn fs(&mut self) -> Option<&mut Fs<'r, 't>> {
        if self.fs.is_none() {
            self.fs = Some(Fs::attach(self.rt, self.transport).await?);
        }
        self.fs.as_mut()
    }
}

/// The running VM: the worker, its doorbell, and the wake channel thread 0
/// parks on.
pub struct Host<'t> {
    worker: DoorbellWorker,
    wake: NoticeRx<'t, TrapTransport>,
}

impl<'t> Host<'t> {
    /// Spawn the VM's worker.  `None` when the task holds no channel space
    /// (its profile lacks `channels`) or the thread cannot be founded.
    pub fn start<E: Extension>(transport: &'t TrapTransport) -> Option<Self> {
        let worker = thread::spawn_with_doorbell(vm_thread::<E>, &[])?;
        let wake = NoticeRx::new(transport, worker.wake_recv());
        Some(Self { worker, wake })
    }

    /// Park until the worker kicks (or a spurious executor wake), having
    /// first stood the wake post and re-read the word — the module doc's
    /// discipline.  `seen` is the state thread 0 last observed.
    async fn park_until_kicked(&mut self, seen: u32) -> Result<(), Fatal> {
        loop {
            match self.wake.try_recv() {
                // A kick landed; the caller re-reads the word.
                Ok(Some(_)) => return Ok(()),
                Ok(None) => {
                    if CELL.state.load(Ordering::Acquire) != seen {
                        return Ok(());
                    }
                    let Some(post) = self.wake.standing_post() else {
                        return Err(Fatal);
                    };
                    let _ = kore::runtime::executor::next_wake_on(post).await;
                }
                Err(RecvError::Refused(Status::WouldBlock)) => {
                    let _ = kore::runtime::executor::next_wake().await;
                }
                Err(_) => return Err(Fatal),
            }
        }
    }

    /// Run `script` on the VM and collect its outcome and output, serving
    /// every `require` it makes on the way.  `chunk_name` is Lua's chunk
    /// name (NUL-terminated, static); `repl` selects the REPL's
    /// expression-then-statement grammar.
    pub async fn eval<E: Extension>(
        &mut self,
        services: &mut Services<'_, '_>,
        extension: &mut E,
        script: &[u8],
        chunk_name: &'static [u8],
        repl: bool,
    ) -> Result<(Outcome, Vec<u8>, Vec<u8>), Fatal> {
        match CELL.state.load(Ordering::Acquire) {
            EVAL_DONE => {}
            WORKER_GONE => {
                // SAFETY: WORKER_GONE — thread 0 owns the cell.
                let outcome = unsafe { payload().outcome };
                return Ok((outcome, Vec::new(), Vec::new()));
            }
            _ => return Err(Fatal),
        }
        {
            // SAFETY: EVAL_DONE — thread 0 owns the cell.
            let cell = unsafe { payload() };
            cell.kind = if repl { KIND_REPL } else { KIND_CHUNK };
            cell.script = script.as_ptr();
            cell.script_len = script.len();
            cell.chunk_name = chunk_name.as_ptr();
            cell.chunk_name_len = chunk_name.len();
        }
        CELL.state.store(EVAL_POSTED, Ordering::Release);
        // A refused kick is not a failure: it means no post stood, and the
        // worker re-tests the word after it arms one (`futex::wait`'s
        // contract), so the store above reaches it regardless.
        let _ = self.worker.kick();
        loop {
            match CELL.state.load(Ordering::Acquire) {
                state @ (EVAL_POSTED | CALL_SERVED) => {
                    self.park_until_kicked(state).await?;
                }
                CALL_POSTED => {
                    serve_call(services, extension).await;
                    CELL.state.store(CALL_SERVED, Ordering::Release);
                    let _ = self.worker.kick();
                }
                EVAL_DONE => {
                    // SAFETY: EVAL_DONE — thread 0 owns the cell; the
                    // script pointer is dead from here (its frame is ours).
                    let cell = unsafe { payload() };
                    cell.script = core::ptr::null();
                    cell.script_len = 0;
                    return Ok((
                        cell.outcome,
                        core::mem::take(&mut cell.stdout),
                        core::mem::take(&mut cell.stderr),
                    ));
                }
                _ => return Err(Fatal),
            }
        }
    }

    /// Stop the worker and join it, so the task's last thread is thread 0.
    /// Joins only a worker that acknowledged `QUIT` (or is already gone):
    /// a worker mid-evaluation is parked on the bridge and would never
    /// return, so it is left to task teardown rather than waited on.
    pub async fn quit(mut self) {
        match CELL.state.load(Ordering::Acquire) {
            EVAL_DONE => {
                CELL.state.store(QUIT, Ordering::Release);
                let _ = self.worker.kick();
                loop {
                    match CELL.state.load(Ordering::Acquire) {
                        WORKER_GONE => break,
                        state => {
                            if self.park_until_kicked(state).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
            WORKER_GONE => {}
            _ => return,
        }
        let _ = self.worker.join();
    }
}

/// Answer the host call the cell holds.  CALL_POSTED — thread 0 owns the
/// cell for the whole call.  One dispatcher: a new standard-library verb
/// is an arm here and nothing about the transport changes.
async fn serve_call<E: Extension>(services: &mut Services<'_, '_>, extension: &mut E) {
    // SAFETY: CALL_POSTED.
    let op = unsafe { payload() }.op;
    if op == OP_REQUIRE {
        serve_require(services).await;
    } else {
        serve_value(services, extension, op).await;
    }
}

/// Answer the `require` the cell holds: read the source, or link the
/// native module and look its opener up.
async fn serve_require(services: &mut Services<'_, '_>) {
    let (name, len) = {
        // SAFETY: CALL_POSTED.
        let cell = unsafe { payload() };
        (cell.name, cell.name_len)
    };
    let reply = resolve(services, &name[..len.min(FS_NAME_BYTES)]).await;
    // SAFETY: CALL_POSTED.
    let cell = unsafe { payload() };
    match reply {
        Reply::Source(source) => {
            cell.reply = REPLY_SOURCE;
            cell.source = source;
            cell.opener = None;
            cell.error = core::ptr::null();
        }
        Reply::Native(opener) => {
            cell.reply = REPLY_NATIVE;
            cell.source = Vec::new();
            cell.opener = Some(opener);
            cell.error = core::ptr::null();
        }
        Reply::Error(message) => {
            cell.reply = REPLY_ERROR;
            cell.source = Vec::new();
            cell.opener = None;
            cell.error = message.as_ptr();
        }
    }
}

/// Answer every value-returning op.  The outcome is written back as a
/// status, up to two words, and a text; nothing here panics on a refusal.
async fn serve_value<E: Extension>(services: &mut Services<'_, '_>, extension: &mut E, op: u32) {
    let args = {
        // SAFETY: CALL_POSTED.
        unsafe { payload() }.args
    };
    // The call's byte argument, borrowed from the Lua frame the parked
    // worker is holding open.  Read here, once, so no arm below repeats
    // the reasoning about why this pointer is live.
    //
    // SAFETY: CALL_POSTED — thread 0 owns the cell, and the worker set
    // this window from a Lua value on its own stack before parking.  It
    // stays parked, and so the value stays rooted, until CALL_SERVED.
    let input: &[u8] = unsafe {
        let cell = payload();
        if cell.input.is_null() || cell.input_len == 0 {
            &[]
        } else {
            core::slice::from_raw_parts(cell.input, cell.input_len)
        }
    };
    // The contiguous block, plus the one verb numbered past `os.exit`.
    if (OP_FS_OPEN..=OP_FS_LIST).contains(&op) || op == OP_FS_READ_FILE {
        let answer = serve_fs(services, op, args, input).await;
        return finish_answer(answer);
    }
    let answer = match op {
        // The executor's own timer — the one verb `kore.time` cannot serve
        // from the VM thread, because parking is what it is for.
        OP_SLEEP => {
            kore::runtime::executor::sleep_for(args[0]).await;
            Answer::Value(0, 0)
        }
        // The launch arguments, answered as text.  A REPL was launched
        // with none, so an empty answer is the honest one, not a refusal.
        OP_PROC_ARGS => {
            let launch = services.args.clone();
            return finish(STATUS_OK, [0; 2], launch);
        }
        // Recorded, not acted on: the raise that follows this call is what
        // unwinds the evaluation, running every `<close>` on the way out.
        OP_PROC_EXIT => {
            services.exit_request = Some(args[0] as i32);
            Answer::Value(0, 0)
        }
        // Anything this crate does not own belongs to the embedding
        // program, which was launched holding the face it needs.
        op if op >= OP_EMBEDDER_BASE => extension.serve(op, args).await,
        _ => Answer::Unavailable(alloc::string::String::from("unknown host call")),
    };
    finish_answer(answer)
}

/// Write one [`Answer`] back into the cell.  Every op ends here, so the
/// three shapes a script sees are decided in exactly one place.
fn finish_answer(answer: Answer) {
    match answer {
        Answer::Value(result, result2) => finish(STATUS_OK, [result, result2], Vec::new()),
        Answer::Bytes(count, bytes) => finish(STATUS_OK, [count, bytes.len() as u64], bytes),
        Answer::Refused(text) => finish(STATUS_REFUSED, [0; 2], text.into_bytes()),
        Answer::Unavailable(text) => finish(STATUS_UNAVAILABLE, [0; 2], text.into_bytes()),
    }
}

/// The `kore.fs` arms.
///
/// Every one needs the filesystem session, so it is attached once here
/// rather than in eleven places, and a task whose launch conveyed no
/// filesystem face answers "unavailable" — rule 6's honest absence, not a
/// refusal that would read as though the file were the problem.
async fn serve_fs(
    services: &mut Services<'_, '_>,
    op: u32,
    args: [u64; 3],
    input: &[u8],
) -> Answer {
    // Split the borrow: `files` and `fs` are distinct fields, and every
    // verb below needs both at once.
    let files = &mut services.files;
    let fs = match services.fs.as_mut() {
        Some(fs) => fs,
        None => {
            let Some(attached) = Fs::attach(services.rt, services.transport).await else {
                return Answer::Unavailable(alloc::string::String::from(
                    "this task holds no filesystem face",
                ));
            };
            services.fs.insert(attached)
        }
    };
    match op {
        OP_FS_OPEN => crate::files::open(fs, files, input, args[0]).await,
        OP_FS_CLOSE => crate::files::close(fs, files, args[0]).await,
        OP_FS_READ => crate::files::read(fs, files, args[0], args[1], args[2]).await,
        OP_FS_READ_FILE => crate::files::read_file(fs, input).await,
        OP_FS_WRITE => crate::files::write(fs, files, args[0], args[1], input).await,
        OP_FS_SYNC => crate::files::sync(fs, files, args[0]).await,
        OP_FS_RESIZE => crate::files::resize(fs, files, args[0], args[1]).await,
        OP_FS_STAT => crate::files::stat(fs, files, args[0], input).await,
        OP_FS_REMOVE => crate::files::remove(fs, input).await,
        // `rename` carries two paths in one window, separated by a NUL —
        // a name may not contain one, so the split is unambiguous.
        OP_FS_RENAME => match input.iter().position(|byte| *byte == 0) {
            Some(cut) => crate::files::rename(fs, &input[..cut], &input[cut + 1..]).await,
            None => Answer::Refused(alloc::string::String::from("invalid-name")),
        },
        OP_FS_LIST => crate::files::list(fs, input).await,
        _ => Answer::Unavailable(alloc::string::String::from("unknown filesystem call")),
    }
}

/// Write one value-op outcome back into the cell.  Every op answers the
/// same three things, so every arm ends here.
fn finish(status: c_int, results: [u64; 2], text: Vec<u8>) {
    // SAFETY: CALL_POSTED.
    let cell = unsafe { payload() };
    cell.status = status;
    cell.results = results;
    cell.text = text;
}

enum Reply {
    Source(Vec<u8>),
    Native(LuaCFunction),
    Error(&'static CStr),
}

async fn resolve(services: &mut Services<'_, '_>, stem: &[u8]) -> Reply {
    let search = Search { prefixes: SEARCH };
    let Some(fs) = services.fs().await else {
        return Reply::Error(c"filesystem face is unavailable");
    };
    let mut buf = Vec::new();
    if buf.try_reserve_exact(MAX_SOURCE_MODULE_BYTES).is_err() {
        return Reply::Error(c"source buffer allocation refused");
    }
    buf.resize(MAX_SOURCE_MODULE_BYTES, 0);
    match Loader::read_source(fs, search, stem, &mut buf).await {
        Ok(len) => {
            buf.truncate(len);
            return Reply::Source(buf);
        }
        Err(LoadError::NotFound) => {}
        Err(error) => return Reply::Error(describe(error)),
    }
    drop(buf);
    if services.loader.is_none() {
        match Loader::new(services.rt) {
            Ok(loader) => services.loader = Some(loader),
            Err(error) => return Reply::Error(describe(error)),
        }
    }
    let (Some(fs), Some(loader)) = (services.fs.as_mut(), services.loader.as_mut()) else {
        return Reply::Error(c"filesystem face is unavailable");
    };
    let id = match loader.load_native(fs, search, stem).await {
        Ok(id) => id,
        Err(error) => return Reply::Error(describe(error)),
    };
    let mut symbol = [0u8; OPENER_PREFIX.len() + FS_NAME_BYTES];
    symbol[..OPENER_PREFIX.len()].copy_from_slice(OPENER_PREFIX);
    symbol[OPENER_PREFIX.len()..OPENER_PREFIX.len() + stem.len()].copy_from_slice(stem);
    match loader.lookup_executable(id, &symbol[..OPENER_PREFIX.len() + stem.len()]) {
        // SAFETY: the address of an exported function in a module this
        // task linked and published executable; the C ABI is the host's.
        Some(address) => {
            Reply::Native(unsafe { core::mem::transmute::<*const u8, LuaCFunction>(address) })
        }
        None => Reply::Error(c"module exports no executable luaopen_<name>"),
    }
}

/// One line per refusal class — what the Lua error says after
/// `module '<name>' not found:`.
fn describe(error: LoadError) -> &'static CStr {
    match error {
        LoadError::Name => c"module name is not a store name",
        LoadError::NotFound => c"no <name>.lua in the task tree and no libs/<name>.so",
        LoadError::NotExecutableImage => c"libs/ member is not stamped EXECUTABLE_IMAGE",
        LoadError::TooLarge => c"module exceeds the bounded loader",
        LoadError::TooMany => c"module table is full",
        LoadError::Fs(_) | LoadError::Pair => c"filesystem refused",
        LoadError::Check(_) => c"module failed the image check",
        LoadError::Plan(_) => c"module layout is not late-linkable (TLS, or data below code)",
        LoadError::HostTls => c"host carries TLS; a late scope cannot reproduce it",
        LoadError::NoRecord(1) => c"host has no .kore.ld block",
        LoadError::NoRecord(2) => c"host has no passive module page at the fixed address",
        LoadError::NoRecord(3) => c"passive page's first record is not this executable",
        LoadError::NoRecord(4) => c"host geometry has no LINK window",
        LoadError::NoRecord(5) => c"passive page address overflowed",
        LoadError::NoRecord(6) => c"passive page records no live segment",
        LoadError::NoRecord(_) => c"host segment could not be viewed",
        LoadError::NoWindow | LoadError::NoSupply | LoadError::Map | LoadError::Publish => {
            c"arena refused the module"
        }
        LoadError::Link(_) => c"module did not link against the host",
        LoadError::InitializerArray => c"module initializer array is malformed",
        LoadError::Seal => c"module RELRO protection was refused",
        LoadError::Initializer(kore_late::FunctionError::Shape) => {
            c"module initializer array is malformed"
        }
        LoadError::Initializer(kore_late::FunctionError::Target(_)) => {
            c"module initializer is outside loaded executable text"
        }
        LoadError::TooManyInitializers => c"module has too many initializers",
    }
}
