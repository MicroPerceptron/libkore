//! Lua 5.5 on Kore.
//!
//! The pinned engine (`vendor/lua`, `KORE-VENDOR.md`), its freestanding
//! host, and the `kore.*` standard library a script sees. An embedding
//! program supplies the process around it — arguments, streams, a REPL or a
//! batch run — and, through [`Extension`], any library this crate does not
//! own.
//!
//! # Two surfaces, one implementation
//!
//! A script reaches the same code two ways, because the two audiences want
//! different things and neither should be made to use the other's.
//!
//! ```lua
//! local kore = require "kore"          -- also a global
//! kore.time.now()                      -- monotonic seconds, float
//! kore.time.sleep(0.25)
//! kore.random.bytes(32)
//! kore.proc.args                       -- a 1-based array, read once
//! kore.fs.read_file "/etc/hosts"       -- value, err
//! kore.fs.stat "/etc/hosts"            -- size_bytes, kind, directory
//! kore.has "fs"  kore.caps()           -- honest provisioning
//! ```
//!
//! `kore.fs` adds only what the stock surface has no spelling for —
//! `read_file`, `write_file`, `mkdir`, `resize`, `stat`, `exists`, `list`
//! — and never changes a stock member.
//!
//! ```lua
//! local f = assert(io.open("/tmp/log", "a"))   -- the SAME object
//! f:write("hello\n"); f:close()
//! for line in io.lines "/etc/hosts" do print(line) end
//! print(os.date("!%Y-%m-%dT%H:%M:%SZ"))
//! ```
//!
//! `io.open` and [`files::open`]'s `kore.fs.open` answer one userdata;
//! only the failure spelling differs — `nil, message, errno` on the stock
//! surface, `nil, {code = "not-found"}` on the kore one, both derived from
//! the same [`kore::fs::FsError`]. A stock member either conforms fully or
//! is absent, never in between.
//!
//! Plus Lua's own base, coroutine, math, string, table and UTF-8 libraries,
//! and `require` over the task's store (`<name>.lua`, then `<name>.so`).
//!
//! # What is absent, and why
//!
//! - `os.execute`, `io.popen` — both launch a process, which is a service
//!   face the embedding program holds and this crate deliberately does not.
//!   An embedder adds them through [`Extension`].
//! - `string.format` — kore has no hosted `printf`, and a hand-written
//!   subset would accept format strings it silently did not conform to
//!   (`KORE-VENDOR.md`).
//! - `load`, `loadfile`, `dofile`, `string.dump` — this build accepts text
//!   chunks only; upstream Lua has no bytecode verifier.
//!
//! `os.getenv` is *present* and answers `nil` for every name: a kore task
//! has no environment, authority arrives as capabilities in its launch,
//! and an unset variable is exactly what `nil` means.
//!
//! # The six rules
//!
//! Every `kore.*` module obeys all six, so one module teaches the rest:
//! one result convention (`value, err`, never a raise for an expected
//! failure); one error vocabulary; one options convention (a trailing table
//! with named keys); one handle protocol; one unit convention (seconds are
//! floats, anything else carries its unit in the name); one availability
//! convention (`kore.has`/`kore.caps`, and an unheld module is `nil`).
//!
//! # Threads
//!
//! Lua is a synchronous C engine, so the VM runs on a [`kore::thread`]
//! worker and the executor stays on thread 0. The `bridge` module
//! documents the one transport between them. Verbs needing no service —
//! `kore.time.now`, `kore.random`, and the whole calendar — never cross
//! it.
//!
//! `bridge`, and with it `Host`, `Services` and `Outcome`, exist only
//! where the engine does. They are named in prose rather than linked
//! throughout these docs: a host build compiles no engine, so the items
//! are absent and an intra-doc link to one would not resolve.
//!
//! # Time
//!
//! kore keeps one clock and it reads UTC. `os.date` therefore answers the
//! same thing with and without the `!` prefix, reports `isdst = false`, and
//! `os.time` reads a table as UTC. The calendar itself is computed, not
//! tabulated: there is no `gmtime` below this and no timezone database.
//! A clock that cannot be read answers `nil, why` and never a zero
//! timestamp, which a caller would read as 1970 rather than as "no
//! reading".

#![no_std]
// `shim`'s allocator seam, which only the engine's build has.
#![cfg_attr(target_os = "none", feature(allocator_api))]

extern crate alloc;

// The engine and its transport exist only where the engine does: the C
// amalgamation, the worker thread and the doorbell are all `target_os =
// "none"` surfaces, and `kore::runtime`'s thread verbs are themselves
// gated that way. Gating here rather than leaving the crate target-only is
// what lets `files`'s logic — path splitting, handle encoding, the error
// vocabulary — be checked and unit-tested by the host workspace.
#[cfg(target_os = "none")]
pub mod bridge;
// Documented by its own `//!` header. Keeping the module's docs in one
// file is what lets rustdoc resolve their intra-doc links against that
// file's imports.
pub mod files;
#[cfg(target_os = "none")]
mod numeric;
#[cfg(target_os = "none")]
mod shim;
#[cfg(target_os = "none")]
mod vm;

use alloc::string::String;
use core::ffi::{CStr, c_int, c_void};

#[cfg(target_os = "none")]
pub use bridge::{Fatal, Host, Outcome, Services};

/// A `lua_CFunction`: the C signature every Lua-callable function has.
///
/// An [`Extension`] writes these in ordinary Rust — `extern "C" fn` over
/// the exported `lua_*` / `luaL_*` API — so an embedder adds a library
/// without compiling C of its own.
pub type LuaOpener = unsafe extern "C" fn(*mut c_void) -> c_int;

/// One library an embedder adds beneath the `kore` table.
///
/// `name` is the field it lands on (`kore.<name>`); `opener` is called once
/// at VM construction and must leave exactly one value — the module table —
/// on the stack, exactly as `luaL_newlib` does.
///
/// `#[repr(C)]` with a thin name pointer so a `&'static [Library]` *is* the
/// array the C amalgamation walks; nothing is marshalled at startup.
#[repr(C)]
pub struct Library {
    name: *const core::ffi::c_char,
    opener: LuaOpener,
}

impl Library {
    /// Name a library and the opener that builds it.
    pub const fn new(name: &'static CStr, opener: LuaOpener) -> Self {
        Self {
            name: name.as_ptr(),
            opener,
        }
    }
}

// SAFETY: both fields are `'static` — a string literal's bytes and a
// function pointer. The table is read, never written, and the VM thread is
// the only reader.
unsafe impl Sync for Library {}

/// What an embedder-served op answers, in the three shapes every `kore.*`
/// call already uses: a value, a refusal the service named, or a statement
/// that the call could not be made at all.
pub enum Answer {
    /// Up to two result words. Lua sees whatever the opener pushes.
    Value(u64, u64),
    /// A byte answer — a read's bytes, a directory's names — with one
    /// result word beside it (how many bytes, how many entries). Bytes and
    /// a refusal ride the same reply field, so a verb that sometimes
    /// answers data and sometimes a reason needs no second channel.
    Bytes(u64, alloc::vec::Vec<u8>),
    /// The service refused. Lua sees `nil, "refused: <text>"`.
    Refused(String),
    /// No face, no lease, a broken ring. Lua sees `nil, "<text>"`.
    Unavailable(String),
}

/// The first op number an embedder may use.
///
/// Below this the vocabulary is this crate's (`require`, `kore.time.sleep`,
/// `kore.proc.args`); at or above it [`Extension::serve`] is asked. Keeping
/// one numbering over one transport is what lets an embedder add a library
/// without this crate learning that the library exists.
pub const OP_EMBEDDER_BASE: u32 = 64;

/// Libraries and ops the embedding program adds to the VM.
///
/// The split is deliberate: this crate owns the engine, the transport and
/// everything reachable through libkore's own facade; anything needing a
/// Kore *service* — an accelerator face, a device — belongs to the program
/// that was launched holding it. Kore's `lua` binary implements this for
/// `kore.accel`; the shape mirrors `kore-wasi`'s embedder-composed
/// interfaces.
///
/// Selection is by type instantiation, not runtime dispatch: `Host::start`
/// is generic over `E`, so `LIBRARIES` reaches the worker thread as
/// monomorphized static data with no global to publish it.
///
/// There is deliberately **no `'static` bound**. The worker thread reads
/// only [`Extension::LIBRARIES`], which is `&'static` in its own right;
/// [`Extension::serve`] runs on thread 0 against a `&mut E` that never
/// leaves that thread. Requiring `'static` of the whole type would forbid
/// the one shape an embedder actually has — a service face borrowed from
/// the `Rt` and the transport its `main` owns.
pub trait Extension {
    /// Registered beneath `kore` at VM construction, in this order.
    const LIBRARIES: &'static [Library] = &[];

    /// Serve one op at or above [`OP_EMBEDDER_BASE`], on the executor
    /// thread, with the cell's three argument words.
    fn serve(&mut self, op: u32, args: [u64; 3]) -> impl Future<Output = Answer>;
}

/// An embedder that adds nothing: Lua plus `kore.*` and no more.
pub struct NoExtension;

impl Extension for NoExtension {
    async fn serve(&mut self, _op: u32, _args: [u64; 3]) -> Answer {
        Answer::Unavailable(String::from("this program registered no such library"))
    }
}

/// The Lua C API an [`Extension`]'s openers call.
///
/// Deliberately the subset an ordinary library binding needs — arguments
/// in, values out, a table built, an error raised — rather than the whole
/// engine. Every one of these is exported from this crate's image
/// (`--export-dynamic-symbol=lua_*` / `luaL_*`), which is also the ABI a
/// native `require`d module binds against.
#[cfg(target_os = "none")]
pub mod api {
    pub use crate::vm::api::*;
}
