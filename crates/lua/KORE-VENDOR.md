# Lua source provenance

`vendor/lua` contains the Lua 5.5.1 core, auxiliary library, and the selected
base/coroutine/math/string/table/UTF-8 libraries copied from
`MicroPerceptron/lua` commit
`7579fc9d7ed90240487251dfb69168f8e64e9294`, with ONE kore edit:
`luaconf.h` guards its `#define LUA_API extern` with `#if !defined(LUA_API)`
so `src/kore_lua.c` can predefine `LUA_API` as default-visibility `extern`.
The engine is compiled `-fvisibility=hidden`; that predefinition plus the
linker's `--export-dynamic-symbol=lua_*` / `luaL_*` is what puts the C API
in the host's `.dynsym`, where a native module loaded by `kore-late`
(`require "lua_demo"`) binds to it.  The exported set is the module ABI.

The copied source is intentionally pinned in-tree: a kore image build must not
fetch source or execute a moving upstream build. `src/kore_lua.c` is the
kore-owned amalgamation/configuration layer; it excludes Lua's POSIX I/O, OS,
package loader, and debug libraries — `liolib.c`, `loslib.c`, `loadlib.c` and
`ldblib.c` are not compiled at all.

`io` and `os` are nonetheless *present*, written by kore over its own
filesystem face rather than over a libc that does not exist here. They are a
veneer, not a second implementation: `io.open` and `kore.fs.open` answer the
same file object, and one failure is rendered two ways — `nil, message, errno`
on the stock surface and `nil, {code = "not-found"}` on the kore one, both
derived from `crates/dagfs/src/error.rs`'s `FsError` rather than invented.

The same rule that keeps `string.format` out governs what those two tables
contain: **a member either conforms fully or is absent.** Measured against
the Lua manual §6.8 and §6.9, twenty of the twenty-two members are present,
along with all seven `file:` methods:

- `io.close`, `io.flush`, `io.input`, `io.lines`, `io.open`, `io.output`,
  `io.read`, `io.tmpfile`, `io.type`, `io.write`
- `file:close`, `file:flush`, `file:lines`, `file:read`, `file:seek`,
  `file:setvbuf`, `file:write`
- `os.clock`, `os.date`, `os.difftime`, `os.exit`, `os.getenv`, `os.remove`,
  `os.rename`, `os.setlocale`, `os.time`, `os.tmpname`

The file object carries a `closed` method beside the manual's seven, and
`kore.fs` adds the verbs the stock surface has no spelling for —
`read_file`, `write_file`, `mkdir`, `resize`, `stat`, `exists`, `list`.
Those are additions on kore's own surface, never changes to a stock member:
a file `kore.fs.open` answers is the same object `io.open` answers, and its
seven methods render failure the same way on both.

Where the manual leaves a detail to the system, this build chooses and says
so:

- `os.date` computes civil time directly (Howard Hinnant's
  `civil_from_days`). There is no `gmtime` and no timezone database below
  it, so kore's one clock reads UTC: `os.date` answers the same thing with
  and without the `!` prefix and reports `isdst = false`, and `os.time`
  reads a table as UTC.
- `os.clock` reports monotonic time since the VM started. kore accounts a
  task no separate CPU time, and the manual asks only for "an
  approximation" — which this is exactly, for the single-threaded script
  that is the only thing measuring it.
- `os.tmpname` *creates* its file exclusively before answering the name,
  which is the POSIX behaviour the manual describes, rather than handing
  back a name another task could already hold. `io.tmpfile` opens the same
  draw in update mode and removes the file when the handle is retired —
  sooner than "when the program ends", and by the same path, since the
  VM's teardown retires every handle.
- `os.getenv` answers `nil` for every name: a kore task has no environment,
  authority arrives as capabilities in its launch, and an unset variable is
  exactly what `nil` means.
- `os.date` and `os.time` answer for years 0000 through 9999 and refuse
  anything outside that with the manual's own
  `date result cannot be represented in this installation`. The civil
  conversion is exact for every day the proleptic Gregorian calendar
  defines, but the broken-down year is an `int` and `%Y` spells four
  digits, so the range is stated rather than left to truncate a huge year
  into a plausible-looking date. The bound is computed from the same
  `days_from_civil` the conversion uses, so the two cannot disagree.
- `io.stdin` is a real file object that is always at end of file, the
  conforming shape for a stream with nothing behind it. `io.stdin`,
  `io.stdout` and `io.stderr` refuse `:close()` with
  `cannot close standard file` — "the I/O library never closes these
  files".
- `io.input`, `io.output` and `io.lines` raise on a failed open
  (`cannot open file 'NAME' (REASON)`) rather than answering a code, as the
  manual specifies for those three alone.

`os.execute` and `io.popen` are the two absent members. Both launch a
process, which is a service face the embedding program holds and this crate
deliberately does not; Kore's own `lua` binary supplies them through
`Extension`, the same seam `kore.accel` arrives on.

The selected string library deliberately omits `string.format`. kore has no
hosted `printf`, and exposing a hand-written subset would make accepted format
strings silently non-conforming. Core number-to-string conversion remains
available: the Rust shim implements Lua's `%.15g`/`%.17g` two-pass contract,
while integer and pointer formatting are restricted to Lua core's audited
internal formats. This build accepts text chunks only and omits base
`load`/`loadfile`/`dofile` plus `string.dump`: upstream Lua deliberately has no
bytecode verifier, so an untrusted binary chunk is not a safe capability-task
input. The base library also omits `warn`: the selected freestanding host has
no warning sink, so retaining an API that silently discards every message
would be misleading.

Lua is Copyright (C) 1994-2026 Lua.org, PUC-Rio and distributed under the MIT
license. The full upstream copyright and license notice is preserved at the
end of `vendor/lua/lua.h`.

`vendor/musl-setjmp` contains only musl's six architecture-specific
`setjmp`/`longjmp` assembly files for x86_64, AArch64, and RV64, copied from
the musl source bundled by Zig `0.16.0-dev.2193+fc517bd01`. The build accepts
the Zig `0.16.x` family, including stable `0.16.0`. The C amalgamation asserts
that the toolchain's musl `jmp_buf` still begins with the exact 8-, 22-, or
26-word register-storage prefix consumed by the x86_64, AArch64, or RV64
assembly, so compatibility follows the consumed ABI rather than one Zig build
identifier.
The six vendored routines and those three header dimensions were verified
unchanged in stable Zig `0.16.0`.
Lua requires non-local error
recovery and Clang's restricted builtins are not implemented on all three
targets. The corresponding musl MIT notice is preserved in
`vendor/musl-setjmp/COPYRIGHT`; no other musl runtime is linked.

The exact AArch64 and RV64 assembly is runtime-tested under full-system QEMU
without a hosted libc:

```sh
python3 crates/lua/tests/run_setjmp_targets.py
```

That probe crosses a deeper stack frame and checks both a nonzero long-jump
value and C's required `longjmp(..., 0) -> 1` normalization.

## What the build tests

`build.rs` compiles and runs `tests/host_smoke.c` on every build that has a
host C compiler, and `make cross` therefore runs it for all three targets.
It is skipped, with a `cargo:warning`, where no C compiler is present: an
ordinary `cargo check`/`cargo test` of this workspace must not require a C
or Zig toolchain, and the Zig steps that build the freestanding object run
only for `target_os = "none"`.

The harness has two passes over the same engine:

- The **selftest** runs with `src/kore_lua.c`'s weak host stubs, which is
  what a build with no Rust host links. It covers the engine, the
  calendar, the number formatting, the selected and omitted library
  members, and the shapes every `io`/`os` member answers with nothing
  behind them.
- The **filesystem suite** runs against `tests/host_fs.c`, an in-memory
  store behind the *real* host transport, and covers `io`, `os` and
  `kore.fs` end to end: open modes including append and update, seeks,
  every read format, multi-page reads, `lines` and its to-be-closed
  variable, the default input and output, directories and listings,
  `resize`, `os.remove`/`rename`/`tmpname`, `io.tmpfile`'s self-removal,
  `os.exit`'s recorded code, the `__gc` backstop, and both failure
  spellings.

Two properties of that store are deliberate and load-bearing. Its reply is
**length-delimited with no terminator and a poisoned byte after it**, which
is the bridge's actual contract (`src/bridge.rs` answers a Rust `Vec<u8>`),
so anything that reaches for `strlen` over a reply fails a test rather than
passing by luck. And it injects faults — a refused read after *n* reads, a
refused close, a refused sleep — which is the only way to reach the paths
where a failure lands after a partial read, with `luaL_Buffer` already
grown onto the stack as a box.
