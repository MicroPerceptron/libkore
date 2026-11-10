# C and Zig: one ABI over native ownership

`kore-ffi` provides a Rust-founded embedding ABI. cbindgen 0.29.4 generates
`extern/c/include/kore.h` from its actual `extern "C"` functions and `repr(C)` /
integer-representation types. The namespace generator groups those exported
names into `extern/zig/src/root.zig`. Zig 0.16.0 translates the C header for the
consumer's target at build time; generated host-specific layouts are not checked
in. `build.zig` exports the `kore` module, with filesystem, graph, time, random, task, thread and SPMD namespaces, and the
complete declarations/constants under `kore.abi`.

Generation is inexpensive once the boundary exists. Rust lifetimes, affine
`View` ownership, futures, slices and native error enums are not themselves a C
ABI. The adapter below establishes those semantics; neither cbindgen nor Zig's
translator invents them. See [cbindgen's supported API shapes](https://github.com/mozilla/cbindgen/blob/v0.29.4/docs.md#writing-your-c-api)
and [Zig C translation](https://ziglang.org/documentation/0.16.0/#C-Translation).

## Native scheduling owns the request

A Rust host attaches `Fs` using its real founding face, then moves it into
`kore_ffi::Session`, after closing any external views opened before handoff.
The host awaits `Session::run(step, user)` on the existing
Kore executor. No second executor, global allocator or global filesystem root is
installed. The C/Zig callback is linked into that same application's image. A C
program with no Rust host gets one from `libkore.so` ([below](#an-independent-c-program-libkoreso)).
Neither is a registry-published static SDK or a POSIX syscall veneer.

Each callback receives an opaque context and the preceding completion. Initially
that completion is `KORE_OP_START`. Queue one operation and return. Submission
returning `KORE_STATUS_QUEUED` means **accepted by the adapter**, not successful
I/O. The next callback reports the native completion. Returning without queuing
anything ends the program and starts acknowledged cleanup of its views.

Implemented operations are open, create file/directory, read, write, refreshed
stat, resize, sync, checked close, compatibility unlink, and paged directory
listing. Native `kore_dag_link` and `kore_dag_detach` preserve DAG projection
and nonempty-directory edge removal separately from compatibility unlink.
Open resolves only within the task's endowed namespace; an empty path
names its root. Names and rights follow DagFS, including rejection of `..`,
unknown right bits and the native name/depth limits. A parent handle is required
for creation and unlink; no ambient POSIX descriptor table is manufactured.

The adapter retains at most eight native views, leaving room for bounded native
path walks within the service's handle ceiling. A foreign handle carries owner,
slot and generation. Copying its bits does not clone a `View`. Successful close
invalidates all copies; refused close retains the original view. Foreign handles
cannot be reused in another owner, and exhausted generations are never recycled.
Native service authority is still revalidated on every operation.

The owner copies write input before returning from the submission call. Reads
copy acknowledged bytes into its bounded scratch buffer, then `kore_fs_read_data`
copies them into foreign storage during the following callback. Input and output
buffers must be disjoint from opaque context storage. No foreign buffer is held
across an await. One transfer is capped at 4096 bytes and the native window may
complete less; `completion.count` is the acknowledged count. Loops must inspect
it. Writes neither truncate nor sync implicitly. After a new request is queued,
previous result-buffer access is refused rather than observing overwritten data.

The fixed owner storage is an intentional copy/custody tradeoff for this ABI;
Rust's existing native window operations remain available for zero-copy use.
There is no hot-path allocation or new lock; each turn makes one foreign
function-pointer call followed by a statically typed native operation. Every eight
completed turns, the driver yields through the existing native executor, even
if requests (such as zero-byte reads) complete immediately. C/Zig callbacks must
return promptly: the adapter does not preempt arbitrary foreign code inside a
callback.

## An independent C program: `libkore.so`

A C program a compiler writes on Kore (`tcc` at Kore's shell) has no Rust
host. `kore-c` is its runtime. A distribution builds `libkore.so` from a
one-line `cdylib` named `kore` over it, linked with `--soname=libkore.so
--no-undefined` (Kore's `user/runtime/libkore`). It ships the library beside
the headers, which `kore-c` publishes to that crate's build script as
`DEP_KORE_INCLUDE`, so they come from the same revision. A program links it with
`-lkore` and gets:

- every `kore.h` function, and the C names `kore/stdlib.h` and
  `kore/pthread.h` alias as real symbols (`c-stdlib`), over the process heap
  `kore-c` installs on the launch's `HEAP` supply;
- a host on thread 0, entered from the C library's start;
- `kore_fs_run(step, user, &error)` (`kore/program.h`), which runs a
  `KoreFsStep` program as the program's filesystem session.

The host is the shape `kore-lua`'s VM bridge has. `main` runs on a main
thread, a `DoorbellWorker` with a 1 MiB guarded stack
(`KORE_MAIN_STACK_BYTES`). Thread 0 is the process owner, and keeps the
executor and the launch's service faces. The two share one request cell and
one word. Every park is a doorbell park; nothing spins.

- **Filesystem.** `kore_fs_run` from the main thread posts the session and
  waits. The host attaches the launch's filesystem face on first use, rooted
  at the directory the program was launched in, and runs `kore-ffi`'s
  `Session` on its executor. The callbacks run there, as on any native
  executor, while the caller waits: they must not depend on its
  thread-local state. A thread that cannot park (any but the main thread)
  is refused, as is a launch with no face. A quarantined session can never
  run again.
- **The end.** The C library's `exit` and `_Exit` call
  `__kore_program_exit(status, finalize)` after their `atexit` handlers, on
  whatever thread called them. The first request wins and reaches the host,
  since threads the program starts inherit a send into its wake channel.
  - The host waits for a main thread that asked to end itself.
  - It then raises kore-rt's process barrier, stopping every other thread
    and running the finalizers only if none was still running. Once a
    program has started a thread, `kore-ffi`'s thread controller is one of
    them, so such a program's finalizers do not run; `atexit` handlers
    always do.
  - It commits the C status, the wire's `i32`, through the C library's own
    exit.
  - An end requested mid-session does not wait for the session.
  - A panic in the library ends the program the same way, its record
    published first.
- **Without a host.** A launch that cannot carry one (no channel space or
  heap) runs the program on thread 0, as its C library alone would. Then the
  filesystem session is refused, and only thread 0's exit ends the program.

The C library's half is two weak references. Its start calls
`__kore_program_start(program, end)` when the program links libkore, and its
exit calls `__kore_program_exit`. A program without libkore is unchanged. On
Kore those are the start files Kore ships with TCC. `tools/check-program.py`
(`make cross`, and CI per ISA) links the shared object as a distribution does
and checks it:

- a library with no entry point;
- its soname, nothing needed or undefined;
- static initial-exec TLS only;
- exactly the declared ABI exported: `kore.h`, `kore/program.h`, the
  `c-stdlib` names and the two lifecycle entry points.

Kore's `make keytest-tcc-x86` compiles a C program against these headers at
Kore's shell, links `libkore.so`, and runs it through every surface above
(Kore `res/TCC.md` §10). That is boot evidence; this repository's checks are
compiler and link evidence. Task preparation (`kore_task_*`) still needs a
Rust host to admit its batch: the independent host does not admit one.

## Failure and lifetime

Filesystem and provider errors have explicit generated foreign enums. Monotonic
and wall time retain distinct functions and explicit calibration/domain errors;
an unavailable clock is never reported as a successful zero timestamp.

An indeterminate transport outcome may follow a committed write. Its final
callback receives `KORE_STATUS_INDETERMINATE`; the context rejects further work,
and the driver returns `RunError::Quarantined`. Nothing is automatically replayed.
Dropping a started driver while it is pending likewise prevents owner reuse
(an unpolled future has not changed the owner). The native ring/data
mappings are task-lifetime mappings; dropping the adapter's scratch buffers does
not free memory still visible to the service. Retire the enclosing task to
reclaim a quarantined owner. Cancellation is not successful close or DMA-style
completion. A normal exit closes each retained view; a refused cleanup close
returns the native error while retaining that view in the Rust owner.

The context and completion pointers exist only for one callback. They may not be
retained, accessed by another thread, used for re-entry, or crossed by
unwinding/longjmp. C/Zig callers must supply valid, aligned, non-overlapping
buffers. This is the usual memory-unsafe FFI boundary, not an intra-process
sandbox; the process's Kore capabilities remain the security boundary.

`kore-lua` already owns this lifecycle rather than exposing these pointers to a
script: the VM runs on a worker thread and reaches every service through one
request cell served on thread 0, so a script holds a handle — an index and a
generation — and never a native view. A Python adapter can do the same.

## Build and verify

Normal Rust consumers need neither generation tool. To develop this repository:

```sh
cargo install cbindgen --version 0.29.4 --locked
# Or use an existing official Zig 0.16.0 installation.
python3 tools/install-zig.py /tmp/libkore-tools
export ZIG=/path/printed/by/the/installer
make bindings
make fmt test check docs owners cross miri bindings-check
```

The installer supports the CI Linux x86_64 and local macOS arm64 hosts and verifies
pinned official archive checksums. Other hosts can supply an official Zig 0.16.0
binary through `ZIG`. `CBINDGEN` and `CLANG` also accept explicit tool paths.

`test-foreign` on `kore-ffi` links compiled C and translated Zig consumers into
host tests. It is disabled for native target checks and Miri; Rust pointer,
handle, buffer and scheduling tests still run under Miri. `bindings-check` checks
drift, compiles C and Zig consumers for x86_64, aarch64 and riscv64, and exercises
the Zig build-system module. Native Rust target checks use Kore's actual JSON
user targets, not merely stock bare-metal targets.

Local macOS package validation with Zig 0.16.0 uses the host-selected SDK;
no global SDK selection change or per-invocation SDK override is needed.

These checks prove declarations, layouts, linked foreign calls, and admission
contracts. They do not claim that a C/Zig-founded standalone process has booted
on hardware. The concrete Rust-owned driver calls the existing native `Fs`
implementation; its founding/entry integration remains explicit in the host.

## Runtime and random extensions

The generated API also includes [native sleep, async task preparation and joined
worker scopes](runtime.md), plus [automatically seeded random values](random.md).
Async callbacks return to the existing executor; synchronous worker scopes retain
the native join owner. No foreign macro translation, second scheduler, TLS RNG,
or global entropy lock is required.
