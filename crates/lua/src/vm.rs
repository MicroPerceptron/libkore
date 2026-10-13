//! Thin ownership wrapper over the pinned Lua C engine.

use alloc::{boxed::Box, vec::Vec};
use core::{
    ffi::{c_char, c_int, c_void},
    ptr::{self, NonNull},
    slice,
};
use kore::abi::process::PROCESS_OUTPUT_BYTES;

use crate::output::Capture;
use crate::shim;
use crate::{Library, Policy};

const LUA_OK: c_int = 0;
const LUA_MULTRET: c_int = -1;

/// The error `os.exit` raises to unwind the evaluation, as `kore_lua.c`'s
/// `KORE_LUA_EXIT_SENTINEL` spells it.
///
/// Matched as a *suffix*, because `luaL_error` puts the chunk name and line
/// in front of it. `os.exit` cannot leave from inside the VM — there are C
/// frames below it and `<close>` handlers above it — so it records the code
/// with the host and raises this; the raise is the unwind, not a failure,
/// and the code is read afterwards through
/// [`Services::exit_request`](crate::Services::exit_request).
const EXIT_SENTINEL: &[u8] = b"kore:exit";

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
    fn kore_lua_new_with_policy(
        alloc: LuaAlloc,
        ud: *mut c_void,
        seed: u32,
        libraries: *const Library,
        library_count: usize,
        restricted: c_int,
    ) -> *mut LuaState;
    fn kore_lua_set_instruction_budget(state: *mut LuaState, budget: *mut InstructionBudget);
    fn kore_lua_install_print(state: *mut LuaState, append: LuaAppend, ud: *mut c_void) -> c_int;
    fn kore_lua_load(
        state: *mut LuaState,
        bytes: *const c_char,
        len: usize,
        name: *const c_char,
    ) -> c_int;
    fn kore_lua_call(state: *mut LuaState, arguments: c_int, results: c_int) -> c_int;
    fn kore_lua_call_traceback(state: *mut LuaState, arguments: c_int, results: c_int) -> c_int;
    fn kore_lua_print_results(state: *mut LuaState) -> c_int;
    fn kore_lua_top(state: *mut LuaState) -> c_int;
    fn kore_lua_set_top(state: *mut LuaState, top: c_int);
    fn kore_lua_error_string(state: *mut LuaState, len: *mut usize) -> *const c_char;
    fn kore_lua_close(state: *mut LuaState);
}

#[repr(C)]
struct InstructionBudget {
    remaining: u64,
    exhausted: c_int,
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
    i32::from(unsafe { &mut *capture }.append(bytes, stderr != 0))
}

pub enum ReplResult {
    Complete(bool),
    CompileFailed,
    /// The chunk ran `os.exit`.
    Exited,
    Incomplete,
    ResourceRefused,
}

/// How one chunk ended.
pub enum EvalResult {
    Ok,
    CompileFailed,
    Failed,
    BudgetExceeded,
    /// The chunk ran `os.exit`: the sentinel unwound every `<close>` on the
    /// way out and the recorded code is the answer, not the raise.
    Exited,
}

pub struct Vm {
    state: NonNull<LuaState>,
    capture: Box<Capture>,
    budget: Option<Box<shim::Budget>>,
    instructions: Option<Box<InstructionBudget>>,
    policy: Policy,
}

impl Vm {
    /// Build a VM, registering `libraries` beneath the `kore` table
    /// after this crate's own modules.
    pub fn new_with_policy(libraries: &'static [Library], policy: Policy) -> Option<Self> {
        let mut capture = Box::try_new(Capture {
            stdout: Vec::new(),
            stderr: Vec::new(),
            limit: PROCESS_OUTPUT_BYTES.min(policy.output_bytes()),
            exhausted: false,
        })
        .ok()?;
        let mut budget = if policy.is_restricted() {
            Some(Box::try_new(shim::Budget::new(policy.heap_bytes())).ok()?)
        } else {
            None
        };
        let instructions = if policy.is_restricted() {
            Some(
                Box::try_new(InstructionBudget {
                    remaining: policy.instructions(),
                    exhausted: 0,
                })
                .ok()?,
            )
        } else {
            None
        };
        // Seed from the CPU, not from a constant: this word decides string
        // hashing and table iteration order, and a fixed one made every
        // boot's ordering identical.  `math.random`'s own seed comes from
        // the same source, through `luai_makeseed` (`kore_lua.c`).
        let state = NonNull::new(unsafe {
            if let Some(ref mut budget) = budget {
                kore_lua_new_with_policy(
                    shim::allocate_bounded,
                    (&mut **budget as *mut shim::Budget).cast(),
                    crate::bridge::seed(),
                    libraries.as_ptr(),
                    libraries.len(),
                    1,
                )
            } else {
                kore_lua_new(
                    shim::allocate,
                    ptr::null_mut(),
                    crate::bridge::seed(),
                    libraries.as_ptr(),
                    libraries.len(),
                )
            }
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
        Some(Self {
            state,
            capture,
            budget,
            instructions,
            policy,
        })
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

    /// Whether the pending error is `os.exit`'s unwind rather than a
    /// failure. `luaL_error` prefixes the chunk and line, so this is a
    /// suffix match on [`EXIT_SENTINEL`].
    fn error_is_exit(&self) -> bool {
        let (error, len) = self.error_parts();
        // SAFETY: `error_parts` answers a live Lua string or a static.
        let error = unsafe { slice::from_raw_parts(error, len) };
        error
            .strip_suffix(b"\n")
            .unwrap_or(error)
            .ends_with(EXIT_SENTINEL)
    }

    fn publish_error(&mut self) {
        let (error, len) = self.error_parts();
        // The Lua string stays live until clear_stack below.
        self.capture
            .append_error(unsafe { slice::from_raw_parts(error, len) });
        self.clear_stack();
    }

    fn execute_loaded(&mut self, print_results: bool) -> EvalResult {
        self.capture.exhausted = false;
        if let Some(ref mut budget) = self.budget {
            budget.exhausted = false;
        }
        if let Some(ref mut remaining) = self.instructions {
            remaining.remaining = self.policy.instructions();
            remaining.exhausted = 0;
            unsafe { kore_lua_set_instruction_budget(self.state.as_ptr(), &mut **remaining) };
        }
        let call = if self.policy.is_restricted() {
            unsafe { kore_lua_call_traceback(self.state.as_ptr(), 0, LUA_MULTRET) }
        } else {
            unsafe { kore_lua_call(self.state.as_ptr(), 0, LUA_MULTRET) }
        };
        if call != LUA_OK {
            // `os.exit`'s unwind reaches here as an error, because that is
            // how it ran the `<close>` handlers on its way out. It is not
            // one, so it is not reported as one.
            if self.error_is_exit() {
                self.clear_stack();
                return EvalResult::Exited;
            }
            let budget_hit = self
                .instructions
                .as_ref()
                .is_some_and(|budget| budget.exhausted != 0)
                || self.capture.exhausted
                || self.budget.as_ref().is_some_and(|budget| budget.exhausted);
            self.publish_error();
            return if budget_hit {
                EvalResult::BudgetExceeded
            } else {
                EvalResult::Failed
            };
        }
        if print_results
            && unsafe { kore_lua_top(self.state.as_ptr()) } > 0
            && unsafe { kore_lua_print_results(self.state.as_ptr()) } != LUA_OK
        {
            if self.error_is_exit() {
                self.clear_stack();
                return EvalResult::Exited;
            }
            self.publish_error();
            return EvalResult::Failed;
        }
        self.clear_stack();
        EvalResult::Ok
    }

    pub fn eval(&mut self, bytes: &[u8], name: &'static [u8]) -> EvalResult {
        self.clear_stack();
        if self.load(bytes, name) != LUA_OK {
            self.publish_error();
            EvalResult::CompileFailed
        } else {
            self.execute_loaded(false)
        }
    }

    fn repl_result(result: EvalResult) -> ReplResult {
        match result {
            EvalResult::Ok => ReplResult::Complete(true),
            EvalResult::CompileFailed => ReplResult::CompileFailed,
            EvalResult::Failed => ReplResult::Complete(false),
            EvalResult::BudgetExceeded => ReplResult::ResourceRefused,
            EvalResult::Exited => ReplResult::Exited,
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
            return Self::repl_result(self.execute_loaded(true));
        }
        self.clear_stack();
        drop(expression);
        if self.load(bytes, b"=stdin\0") == LUA_OK {
            return Self::repl_result(self.execute_loaded(true));
        }
        let (error, len) = self.error_parts();
        let error = unsafe { slice::from_raw_parts(error, len) };
        if error.windows(5).any(|part| part == b"<eof>") {
            self.clear_stack();
            ReplResult::Incomplete
        } else {
            self.publish_error();
            ReplResult::CompileFailed
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
