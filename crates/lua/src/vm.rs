//! Thin ownership wrapper over the pinned Lua C engine.

use alloc::{boxed::Box, vec::Vec};
use core::{
    ffi::{c_char, c_int, c_void},
    ptr::{self, NonNull},
    slice,
};
use kore::abi::process::PROCESS_OUTPUT_BYTES;

use crate::Library;
use crate::shim;

const LUA_OK: c_int = 0;
const LUA_MULTRET: c_int = -1;

#[repr(C)]
struct LuaState {
    _private: [u8; 0],
}

type LuaAlloc = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void;
type LuaAppend = unsafe extern "C" fn(*mut c_void, *const c_char, usize, c_int) -> c_int;

unsafe extern "C" {
    fn kore_lua_new(
        alloc: LuaAlloc,
        ud: *mut c_void,
        seed: u32,
        libraries: *const Library,
        library_count: usize,
    ) -> *mut LuaState;
    fn kore_lua_install_print(state: *mut LuaState, append: LuaAppend, ud: *mut c_void) -> c_int;
    fn kore_lua_load(
        state: *mut LuaState,
        bytes: *const c_char,
        len: usize,
        name: *const c_char,
    ) -> c_int;
    fn kore_lua_call(state: *mut LuaState, arguments: c_int, results: c_int) -> c_int;
    fn kore_lua_print_results(state: *mut LuaState) -> c_int;
    fn kore_lua_top(state: *mut LuaState) -> c_int;
    fn kore_lua_set_top(state: *mut LuaState, top: c_int);
    fn kore_lua_error_string(state: *mut LuaState, len: *mut usize) -> *const c_char;
    fn kore_lua_close(state: *mut LuaState);
}

#[derive(Default)]
struct Capture {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

unsafe extern "C" fn append(
    ud: *mut c_void,
    bytes: *const c_char,
    len: usize,
    stderr: c_int,
) -> c_int {
    let capture = ud.cast::<Capture>();
    if capture.is_null() || bytes.is_null() {
        return 0;
    }
    let bytes = unsafe { slice::from_raw_parts(bytes.cast::<u8>(), len) };
    let used = unsafe {
        (*capture)
            .stdout
            .len()
            .saturating_add((*capture).stderr.len())
    };
    if bytes.len() > PROCESS_OUTPUT_BYTES.saturating_sub(used) {
        return 0;
    }
    let output = if stderr == 0 {
        unsafe { &mut (*capture).stdout }
    } else {
        unsafe { &mut (*capture).stderr }
    };
    if output.try_reserve(bytes.len()).is_err() {
        return 0;
    }
    output.extend_from_slice(bytes);
    1
}

pub enum ReplResult {
    Complete(bool),
    Incomplete,
    ResourceRefused,
}

pub struct Vm {
    state: NonNull<LuaState>,
    capture: Box<Capture>,
}

impl Vm {
    /// Build a VM, registering `libraries` beneath the `kore` table
    /// after this crate's own modules.
    pub fn new(libraries: &'static [Library]) -> Option<Self> {
        let mut capture = Box::try_new(Capture::default()).ok()?;
        // Seed from the CPU, not from a constant: this word decides string
        // hashing and table iteration order, and a fixed one made every
        // boot's ordering identical.  `math.random`'s own seed comes from
        // the same source, through `luai_makeseed` (`kore_lua.c`).
        let state = NonNull::new(unsafe {
            kore_lua_new(
                shim::allocate,
                ptr::null_mut(),
                crate::bridge::seed(),
                libraries.as_ptr(),
                libraries.len(),
            )
        })?;
        let status = unsafe {
            kore_lua_install_print(
                state.as_ptr(),
                append,
                (&mut *capture as *mut Capture).cast(),
            )
        };
        if status != LUA_OK {
            // Installation runs below a C recovery frame, so closing here
            // cannot double-own the state or unwind through this Rust frame.
            unsafe { kore_lua_close(state.as_ptr()) };
            return None;
        }
        Some(Self { state, capture })
    }

    fn clear_stack(&mut self) {
        unsafe { kore_lua_set_top(self.state.as_ptr(), 0) };
    }

    fn load(&mut self, bytes: &[u8], name: &'static [u8]) -> c_int {
        unsafe {
            kore_lua_load(
                self.state.as_ptr(),
                bytes.as_ptr().cast(),
                bytes.len(),
                name.as_ptr().cast(),
            )
        }
    }

    fn error_parts(&self) -> (*const u8, usize) {
        let mut len = 0;
        // This wrapper only borrows an existing Lua string. It deliberately
        // never invokes __tostring outside lua_pcall's recovery frame.
        let text = unsafe { kore_lua_error_string(self.state.as_ptr(), &mut len) };
        if text.is_null() {
            (b"Lua error".as_ptr(), b"Lua error".len())
        } else {
            (text.cast(), len)
        }
    }

    fn publish_error(&mut self) {
        let (error, len) = self.error_parts();
        let used = self
            .capture
            .stdout
            .len()
            .saturating_add(self.capture.stderr.len());
        let remaining = PROCESS_OUTPUT_BYTES.saturating_sub(used);
        if remaining != 0 {
            // Always reserve the final byte for one deterministic line
            // terminator. An attacker-controlled Lua error therefore cannot
            // bypass the aggregate ProcessIo output bound or trigger a large
            // temporary allocation while it is being reported.
            let error = unsafe { slice::from_raw_parts(error, len) };
            let body = error.strip_suffix(b"\n").unwrap_or(error);
            let copied = body.len().min(remaining - 1);
            if self.capture.stderr.try_reserve(copied + 1).is_ok() {
                self.capture.stderr.extend_from_slice(&body[..copied]);
                self.capture.stderr.push(b'\n');
            }
        }
        self.clear_stack();
    }

    fn execute_loaded(&mut self, print_results: bool) -> bool {
        if unsafe { kore_lua_call(self.state.as_ptr(), 0, LUA_MULTRET) } != LUA_OK {
            self.publish_error();
            return false;
        }
        if print_results
            && unsafe { kore_lua_top(self.state.as_ptr()) } > 0
            && unsafe { kore_lua_print_results(self.state.as_ptr()) } != LUA_OK
        {
            self.publish_error();
            return false;
        }
        self.clear_stack();
        true
    }

    pub fn eval(&mut self, bytes: &[u8], name: &'static [u8]) -> bool {
        self.clear_stack();
        if self.load(bytes, name) != LUA_OK {
            self.publish_error();
            false
        } else {
            self.execute_loaded(false)
        }
    }

    /// Stock Lua REPL semantics: expression first, then statement, with an
    /// `<eof>` syntax error requesting a continuation line.
    pub fn eval_repl(&mut self, bytes: &[u8]) -> ReplResult {
        self.clear_stack();
        let Some(capacity) = bytes.len().checked_add(8) else {
            return ReplResult::ResourceRefused;
        };
        let mut expression = Vec::new();
        if expression.try_reserve(capacity).is_err() {
            return ReplResult::ResourceRefused;
        }
        expression.extend_from_slice(b"return ");
        expression.extend_from_slice(bytes);
        if self.load(&expression, b"=stdin\0") == LUA_OK {
            drop(expression);
            return ReplResult::Complete(self.execute_loaded(true));
        }
        self.clear_stack();
        drop(expression);
        if self.load(bytes, b"=stdin\0") == LUA_OK {
            return ReplResult::Complete(self.execute_loaded(true));
        }
        let (error, len) = self.error_parts();
        let error = unsafe { slice::from_raw_parts(error, len) };
        if error.windows(5).any(|part| part == b"<eof>") {
            self.clear_stack();
            ReplResult::Incomplete
        } else {
            self.publish_error();
            ReplResult::Complete(false)
        }
    }

    pub fn take_stdout(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.capture.stdout)
    }

    pub fn take_stderr(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.capture.stderr)
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe { kore_lua_close(self.state.as_ptr()) };
    }
}

/// The Lua C API an [`Extension`](crate::Extension) opener calls.
///
/// Deliberately the subset an ordinary library binding needs — arguments
/// in, values out, a table built, a refusal raised — rather than the whole
/// engine. Every symbol here is one the image already exports as the
/// native-module ABI (`--export-dynamic-symbol=lua_*` / `luaL_*`), so an
/// embedder writes its library in Rust and compiles no C of its own.
///
/// # Safety
///
/// Every function takes the raw `lua_State` an opener was handed, and is
/// valid only on the VM thread inside a call from the engine. Index and
/// type discipline is the caller's, exactly as it is in C: these are thin
/// declarations, not a checked binding.
pub mod api {
    use core::ffi::{CStr, c_char, c_int, c_void};

    /// The engine's `lua_State`, opaque by design.
    pub type State = c_void;
    /// Lua's integer type on every Kore target.
    pub type Integer = i64;

    unsafe extern "C" {
        pub fn lua_settop(state: *mut State, index: c_int);
        pub fn lua_gettop(state: *mut State) -> c_int;
        pub fn lua_type(state: *mut State, index: c_int) -> c_int;
        pub fn lua_createtable(state: *mut State, array: c_int, record: c_int);
        pub fn lua_setfield(state: *mut State, index: c_int, name: *const c_char);
        pub fn lua_pushnil(state: *mut State);
        pub fn lua_pushboolean(state: *mut State, value: c_int);
        pub fn lua_pushinteger(state: *mut State, value: Integer);
        pub fn lua_pushlstring(
            state: *mut State,
            bytes: *const c_char,
            len: usize,
        ) -> *const c_char;
        pub fn lua_pushcclosure(state: *mut State, function: crate::LuaOpener, upvalues: c_int);
        pub fn luaL_checkinteger(state: *mut State, index: c_int) -> Integer;
        pub fn luaL_checklstring(state: *mut State, index: c_int, len: *mut usize)
        -> *const c_char;
        pub fn luaL_argerror(state: *mut State, index: c_int, message: *const c_char) -> c_int;
    }

    /// Push a string.
    ///
    /// # Safety
    /// `state` is a live Lua state on this thread.
    pub unsafe fn push_bytes(state: *mut State, bytes: &[u8]) {
        unsafe { lua_pushlstring(state, bytes.as_ptr().cast::<c_char>(), bytes.len()) };
    }

    /// Set `table[name]` from the value on top, `table` being at `index`.
    ///
    /// # Safety
    /// `state` is live; `index` names a table.
    pub unsafe fn set_field(state: *mut State, index: c_int, name: &CStr) {
        unsafe { lua_setfield(state, index, name.as_ptr()) };
    }

    /// Build a module table from `(name, function)` pairs — the Rust
    /// spelling of `luaL_newlib`. Leaves exactly one value on the stack,
    /// which is what an opener must return.
    ///
    /// # Safety
    /// `state` is a live Lua state on this thread.
    pub unsafe fn new_lib(state: *mut State, entries: &[(&CStr, crate::LuaOpener)]) {
        unsafe {
            lua_createtable(state, 0, entries.len() as c_int);
            for (name, function) in entries {
                lua_pushcclosure(state, *function, 0);
                lua_setfield(state, -2, name.as_ptr());
            }
        }
    }

    /// Read a string argument.
    ///
    /// # Safety
    /// `state` is live and inside a call. Raises if the argument is not a
    /// string, unwinding through the engine's own protected frame.
    pub unsafe fn check_bytes<'a>(state: *mut State, index: c_int) -> &'a [u8] {
        let mut len = 0usize;
        let bytes = unsafe { luaL_checklstring(state, index, &raw mut len) };
        if bytes.is_null() {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(bytes.cast::<u8>(), len) }
    }
}
