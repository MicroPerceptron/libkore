//! The VM's thread and the host-call bridge (`res/DAGFS.md` §4).
//!
//! Lua is a synchronous C engine: `require` must return a module to the
//! C frame that called it, and that frame may sit inside a coroutine, a
//! metamethod, or a `pcall`.  The store, however, is reached only through
//! the task's async filesystem client on its executor.  So the VM runs on a
//! kore-rt **worker thread** and the executor stays on thread 0, across
//! `kore::runtime::bridge`: the `Interpreter` engine, whose work items are
//! evaluations and whose calls are every verb that needs a service.
//!
//! ```text
//!  thread 0 (executor)                       worker (VM)
//!  ───────────────────                       ───────────
//!  found the worker: libraries, policy ─────▶ build the VM, finish: ready
//!  post the eval ───────────────────────────▶ runs the chunk …
//!                                             … C `require` → call
//!  ◀──────────────────────────── the call ─── parked on its doorbell
//!  serve the op (async)
//!  the answer ──────────────────────────────▶ returns; the chunk resumes
//!                                             … chunk ends
//!  ◀───────────────────── finish(outcome) ─── waits for the next item
//!  take stdout/stderr
//! ```
//!
//! The bridge owns the word, the park-and-kick ordering, the cell's
//! hand-over and the worker's end: a VM thread that panics, cannot park, or
//! cannot be built is reported gone, never waited for.  This module keeps
//! what crosses it — an evaluation and its output, a call and its answer —
//! and the dispatch that answers a call ([`serve_call`]).
//!
//! **One transport, many ops.**  `require`, `kore.time.sleep`, and every
//! later verb that needs a service — this crate's own or an embedder's,
//! through [`Extension`] — are the same round trip with a different
//! payload, so they share one call shape ([`kore_lua_host_call`]) and one
//! thread-0 dispatcher ([`serve_call`]).  Adding a verb is a new op
//! constant and a new arm — never a new state, and never a copy of the
//! bridge's ordering.  Verbs that need *no* service do not come here at
//! all: `kore.time.now` reads the clock page and `kore.random` draws from
//! the CPU, both directly on the VM thread.  Buffers cross the boundary as
//! owned `Vec`s — the heap frees a block from any thread onto its owning
//! shard's remote list — or as borrows the owner keeps alive across the
//! whole item (the eval's script) or call (a write's bytes).
//!
//! Resolution: a source module `<name>.lua` from `SOURCE_SEARCH` — the
//! task's own tree, then `libs/` — wins over a native `<name>.so`, which
//! comes from the READ-only `libs/` view alone.  Source runs inside the VM
//! with the VM's authority; native code runs with the task's whole
//! authority, so it is never taken from a tree the task can write (nor can
//! a squatting `<name>.so` there shadow the library).  `kore-late` holds
//! that rule: it resolves the library edge itself, refuses a member not
//! stamped `EXECUTABLE_IMAGE` (a stamp only the install authority can
//! make), checks it, places it in the JIT arena, and links it against this
//! executable's exported C API.

use alloc::vec::Vec;
use core::ffi::{CStr, c_char, c_int};

use kore::abi::clock::TimerError;
use kore::abi::svc::FS_NAME_BYTES;
use kore::fs::Fs;
use kore::runtime::{
    bridge::{Bridge, Engine, Gone, Next, Refused, Settled, Work, WorkerHost},
    entry::Rt,
    thread::WorkerStack,
    transport::TrapTransport,
};
use kore_late::{LoadError, Loader, Search};

use crate::vm::{EvalResult, ReplResult, Vm};
use crate::{Answer, Extension, Library, OP_EMBEDDER_BASE, Policy};

/// The bound on a source module (`<name>.lua`).  Sized like a generous
/// script, not like a library: the whole buffer is staged in the task's
/// 2 MiB launch heap beside the VM, and a module is read whole before the
/// engine sees it.  Larger sources belong in the store as several modules.
pub const MAX_SOURCE_MODULE_BYTES: usize = 64 * 1024;

/// Where `require` looks for a *source* module, in order: the task's own
/// tree (its cwd view), then the READ-only `libs/` view every task carries.
/// A native module takes no search list (`Loader::load_native`).
const SOURCE_SEARCH: &[&[u8]] = &[b"", kore_late::LIBS_VIEW.as_bytes()];

/// The prefix a native module's opener carries: `luaopen_<name>`.
const OPENER_PREFIX: &[u8] = b"luaopen_";

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
/// `kore.fs.write_file`'s second half — replace an open file's whole
/// contents (`arg0` the handle, `input` the bytes).  Positional
/// [`OP_FS_WRITE`] commits once per window; this is the store's own
/// replacement under one held commit, which is the difference between one
/// image/flush/root/flush chain for a save and one per window.
const OP_FS_SAVE: u32 = 31;
/// `kore.fs.entries(path)` — a directory's edges with each target's class
/// and subtype, where [`OP_FS_LIST`] answers only the names.
const OP_FS_ENTRIES: u32 = 32;
/// `kore.fs.rights(path)` — the rights this task holds there, in the
/// store's own `rwns` spelling.  Reports authority; confers none.
const OP_FS_RIGHTS: u32 = 33;

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
    /// Lua source failed to compile; stderr includes the chunk and line.
    CompileFailed,
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

/// What crosses the bridge.
pub struct Payload {
    // thread 0 → worker, with the founding: what the VM is built with —
    // the embedder's monomorphized libraries and policy.
    libraries: &'static [Library],
    policy: Policy,
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

// SAFETY: the pointers cross under the bridge's hand-over and stay alive
// while the other side reads them: the script and chunk name in thread 0's
// frame until the evaluation settles, a call's input in the Lua frame of the
// parked worker until the call is answered, and `error` names a `'static`
// string.
unsafe impl Send for Payload {}

static BRIDGE: Bridge<Payload> = Bridge::new(Payload {
    libraries: &[],
    policy: Policy::INTERACTIVE,
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
});

// ---------------------------------------------------------------- worker

/// The VM as a bridge engine: build it with the embedder's libraries and
/// policy, then evaluate each item until thread 0 asks it to quit.
///
/// Not generic over the [`Extension`]: the extension may borrow what its
/// `main` owns, and an engine outlives no such borrow.  What the worker needs
/// of it is `'static` data, which arrives in the first work item.
struct Interpreter;

impl Engine for Interpreter {
    type Payload = Payload;

    fn bridge() -> &'static Bridge<Payload> {
        &BRIDGE
    }

    fn run(mut first: Work<Payload>) {
        let (libraries, policy) = first.with(|cell| (cell.libraries, cell.policy));
        let Some(mut vm) = Vm::new_with_policy(libraries, policy) else {
            first.fail(|cell| cell.outcome = Outcome::Startup);
            return;
        };
        // The VM is built: the first item, done.
        first.finish(|_| ());
        while let Ok(Next::Work(mut work)) = BRIDGE.next() {
            let (outcome, stdout, stderr) = run_eval(&mut vm, &mut work);
            work.finish(|cell| {
                cell.outcome = outcome;
                cell.stdout = stdout;
                cell.stderr = stderr;
            });
        }
        // Asked to quit (or unable to wait any longer): the VM is dropped
        // before the bridge reads the worker gone, on the body's return.
        drop(vm);
    }
}

fn run_eval(vm: &mut Vm, work: &mut Work<Payload>) -> (Outcome, Vec<u8>, Vec<u8>) {
    let (kind, script, script_len, chunk_name, chunk_name_len) = work.with(|cell| {
        (
            cell.kind,
            cell.script,
            cell.script_len,
            cell.chunk_name,
            cell.chunk_name_len,
        )
    });
    // SAFETY: thread 0 keeps the script and chunk name alive until the
    // evaluation settles.
    let script = if script.is_null() {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(script, script_len) }
    };
    let outcome = match kind {
        KIND_REPL => match vm.eval_repl(script) {
            ReplResult::Complete(true) => Outcome::Ok,
            ReplResult::Complete(false) => Outcome::Failed,
            ReplResult::CompileFailed => Outcome::CompileFailed,
            ReplResult::Exited => Outcome::Exited,
            ReplResult::Incomplete => Outcome::Incomplete,
            ReplResult::ResourceRefused => Outcome::ResourceRefused,
        },
        _ => {
            // SAFETY: thread 0 posts a non-empty `&'static [u8]` chunk
            // name with every KIND_CHUNK evaluation, so this is never the
            // dangling-null case `from_raw_parts` forbids.
            let name: &'static [u8] =
                unsafe { core::slice::from_raw_parts(chunk_name, chunk_name_len) };
            match vm.eval(script, name) {
                EvalResult::Ok => Outcome::Ok,
                EvalResult::CompileFailed => Outcome::CompileFailed,
                EvalResult::Failed => Outcome::Failed,
                EvalResult::BudgetExceeded => Outcome::ResourceRefused,
                EvalResult::Exited => Outcome::Exited,
            }
        }
    };
    (outcome, vm.take_stdout(), vm.take_stderr())
}

/// What a refused host call tells the script.
fn refused(refusal: Refused) -> &'static CStr {
    match refusal {
        Refused::NotWorker | Refused::NoWork => c"host call outside an evaluation",
        Refused::Nested => c"host call made inside another",
    }
}

/// C `require`'s half (`kore_lua.c`): post the name, park until served,
/// hand back what thread 0 stored.  Called on the worker, inside an
/// evaluation.
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
    let served = BRIDGE.call(
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
    );
    let (reply, reply_bytes, reply_len, reply_opener) = match served {
        Ok(served) => served,
        Err(refusal) => return fail(refused(refusal)),
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
    let _ = BRIDGE.with_work(|cell| cell.source = Vec::new());
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
/// The out-pointers are valid for writes; `input` points at `input_len`
/// bytes (or is null), alive until this returns.
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
    let served = BRIDGE.call(
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
            cell.input = core::ptr::null();
            cell.input_len = 0;
            (
                cell.status,
                cell.results,
                cell.text.as_ptr().cast::<c_char>(),
                cell.text.len(),
            )
        },
    );
    let (status, results, reply_text, reply_len) = match served {
        Ok(served) => served,
        Err(refusal) => return fail(refused(refusal)),
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

/// The running VM: its worker, as thread 0 hosts it.
pub struct Host<'t> {
    host: WorkerHost<'t, Interpreter>,
}

impl<'t> Host<'t> {
    /// Spawn the VM's worker, built with `E`'s libraries and policy.  `None`
    /// when the task holds no channel space (its profile lacks `channels`)
    /// or the thread cannot be founded.
    pub fn start<E: Extension>(transport: &'t TrapTransport) -> Option<Self> {
        let host = WorkerHost::<Interpreter>::spawn(transport, WorkerStack::Band, &[], |cell| {
            cell.libraries = E::LIBRARIES;
            cell.policy = E::POLICY;
            cell.outcome = Outcome::Ok;
        })?;
        Some(Self { host })
    }

    /// Run `script` on the VM and collect its outcome and output, serving
    /// every call it makes on the way.  `chunk_name` is Lua's chunk
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
        // The previous item has settled first — the first time, the VM's
        // construction, which makes no calls.
        let settled = self
            .host
            .serve(async |cell: &mut Payload| serve_call(services, extension, cell).await)
            .await
            .map_err(|_| Fatal)?;
        if let Settled::Gone(gone) = settled {
            return self.gone(gone);
        }
        self.host
            .post(|cell| {
                cell.kind = if repl { KIND_REPL } else { KIND_CHUNK };
                cell.script = script.as_ptr();
                cell.script_len = script.len();
                cell.chunk_name = chunk_name.as_ptr();
                cell.chunk_name_len = chunk_name.len();
            })
            .map_err(|_| Fatal)?;
        let settled = self
            .host
            .serve(async |cell: &mut Payload| serve_call(services, extension, cell).await)
            .await
            .map_err(|_| Fatal)?;
        match settled {
            Settled::Idle => {
                let cell = self.host.payload().ok_or(Fatal)?;
                // The script pointer is dead from here: its frame is ours.
                cell.script = core::ptr::null();
                cell.script_len = 0;
                Ok((
                    cell.outcome,
                    core::mem::take(&mut cell.stdout),
                    core::mem::take(&mut cell.stderr),
                ))
            }
            Settled::Gone(gone) => self.gone(gone),
        }
    }

    /// What an evaluation answers once the worker is gone: a VM that could
    /// not be built says so in its outcome ([`Outcome::Startup`]); any other
    /// loss is the bridge's.
    fn gone(&mut self, gone: Gone) -> Result<(Outcome, Vec<u8>, Vec<u8>), Fatal> {
        match (gone, self.host.payload()) {
            (Gone::Failed, Some(cell)) => Ok((cell.outcome, Vec::new(), Vec::new())),
            _ => Err(Fatal),
        }
    }

    /// Stop the worker and join it, so the task's last thread is thread 0.
    /// Joins only a worker that acknowledged the quit (or is already gone):
    /// a worker mid-evaluation is parked on the bridge and would never
    /// return, so it is left to task teardown rather than waited on.
    pub async fn quit(self) {
        let _ = self.host.quit().await;
    }
}

/// Answer the host call the cell holds; thread 0 holds the cell for the
/// whole call.  One dispatcher: a new standard-library verb is an arm here
/// and nothing about the transport changes.
async fn serve_call<E: Extension>(
    services: &mut Services<'_, '_>,
    extension: &mut E,
    cell: &mut Payload,
) {
    if cell.op == OP_REQUIRE {
        serve_require(services, cell, E::POLICY).await;
    } else {
        serve_value(services, extension, cell, E::POLICY).await;
    }
}

/// Answer a `require`: read the source, or link the native module and
/// look its opener up.
async fn serve_require(services: &mut Services<'_, '_>, cell: &mut Payload, policy: Policy) {
    let len = cell.name_len.min(FS_NAME_BYTES);
    match resolve(services, &cell.name[..len], policy).await {
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
async fn serve_value<E: Extension>(
    services: &mut Services<'_, '_>,
    extension: &mut E,
    cell: &mut Payload,
    policy: Policy,
) {
    let op = cell.op;
    if policy.is_restricted()
        && !matches!(
            op,
            OP_FS_READ_FILE | OP_FS_STAT | OP_FS_LIST | OP_FS_ENTRIES | OP_FS_RIGHTS
        )
    {
        return finish_answer(
            cell,
            Answer::Unavailable(alloc::string::String::from(
                "host operation unavailable to read-only worker",
            )),
        );
    }
    let args = cell.args;
    // The call's byte argument, borrowed from the Lua frame the parked
    // worker is holding open.  Read here, once, so no arm below repeats
    // the reasoning about why this pointer is live.
    //
    // SAFETY: the worker set this window from a Lua value on its own stack
    // before it called, and it stays parked inside the call, so the value
    // stays rooted, until the call is answered.
    let input: &[u8] = if cell.input.is_null() || cell.input_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(cell.input, cell.input_len) }
    };
    // The contiguous block, plus the one verb numbered past `os.exit`.
    if (OP_FS_OPEN..=OP_FS_LIST).contains(&op)
        || matches!(
            op,
            OP_FS_READ_FILE | OP_FS_SAVE | OP_FS_ENTRIES | OP_FS_RIGHTS
        )
    {
        let answer = serve_fs(services, op, args, input, policy).await;
        return finish_answer(cell, answer);
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
            return finish(cell, STATUS_OK, [0; 2], launch);
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
    finish_answer(cell, answer)
}

/// Write one [`Answer`] back into the cell.  Every op ends here, so the
/// three shapes a script sees are decided in exactly one place.
fn finish_answer(cell: &mut Payload, answer: Answer) {
    match answer {
        Answer::Value(result, result2) => finish(cell, STATUS_OK, [result, result2], Vec::new()),
        Answer::Bytes(count, bytes) => finish(cell, STATUS_OK, [count, bytes.len() as u64], bytes),
        Answer::Refused(text) => finish(cell, STATUS_REFUSED, [0; 2], text.into_bytes()),
        Answer::Unavailable(text) => finish(cell, STATUS_UNAVAILABLE, [0; 2], text.into_bytes()),
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
    policy: Policy,
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
        // `arg0` is the handle's permissions and the open's behaviour;
        // `arg1` is the route's rights, as packed ASCII the far side parses
        // with the store's own `rights_parse`.
        OP_FS_OPEN => crate::files::open(fs, files, input, args[0], args[1]).await,
        OP_FS_CLOSE => crate::files::close(fs, files, args[0]).await,
        OP_FS_READ => crate::files::read(fs, files, args[0], args[1], args[2]).await,
        OP_FS_READ_FILE => {
            crate::files::read_file_with_limit(fs, input, policy.fs_result_bytes()).await
        }
        OP_FS_SAVE => crate::files::save(fs, files, args[0], input).await,
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
        OP_FS_LIST => crate::files::list_with_limit(fs, input, policy.fs_result_bytes()).await,
        OP_FS_ENTRIES => {
            crate::files::entries_with_limit(fs, input, policy.fs_result_bytes()).await
        }
        OP_FS_RIGHTS => crate::files::rights(fs, input).await,
        _ => Answer::Unavailable(alloc::string::String::from("unknown filesystem call")),
    }
}

/// Write one value-op outcome back into the cell.  Every op answers the
/// same three things, so every arm ends here.
fn finish(cell: &mut Payload, status: c_int, results: [u64; 2], text: Vec<u8>) {
    cell.status = status;
    cell.results = results;
    cell.text = text;
}

enum Reply {
    Source(Vec<u8>),
    Native(LuaCFunction),
    Error(&'static CStr),
}

async fn resolve(services: &mut Services<'_, '_>, stem: &[u8], policy: Policy) -> Reply {
    let search = Search {
        prefixes: SOURCE_SEARCH,
    };
    let Some(fs) = services.fs().await else {
        return Reply::Error(c"filesystem face is unavailable");
    };
    let mut buf = Vec::new();
    let source_limit = MAX_SOURCE_MODULE_BYTES.min(policy.fs_result_bytes());
    if buf.try_reserve_exact(source_limit).is_err() {
        return Reply::Error(c"source buffer allocation refused");
    }
    buf.resize(source_limit, 0);
    match Loader::read_source(fs, search, stem, &mut buf).await {
        Ok(len) => {
            buf.truncate(len);
            return Reply::Source(buf);
        }
        Err(LoadError::NotFound) => {}
        Err(error) => return Reply::Error(describe(error)),
    }
    drop(buf);
    if policy.is_restricted() {
        return Reply::Error(c"source-only worker cannot load native modules");
    }
    if services.loader.is_none() {
        match Loader::new(services.rt) {
            Ok(loader) => services.loader = Some(loader),
            Err(error) => return Reply::Error(describe(error)),
        }
    }
    let (Some(fs), Some(loader)) = (services.fs.as_mut(), services.loader.as_mut()) else {
        return Reply::Error(c"filesystem face is unavailable");
    };
    let id = match loader.load_native(fs, stem).await {
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
        LoadError::NotFound => {
            c"no <name>.lua in the task tree or libs/, and no libs/<name>.so (native modules load only from libs/)"
        }
        LoadError::LibsWritable => c"libs/ is writable by this task; native modules load only from the READ-only library view",
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
