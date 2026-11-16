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
`-lkore` — Kore's TCC links it into every program, its `-lc` reading a script
that joins `libkore.so` to the program's own `libc.a` — and gets:

- every `kore.h` function, and the C names `kore/stdlib.h` and
  `kore/pthread.h` alias as real symbols (`c-stdlib`), over the process heap
  `kore-c` installs on the launch's `HEAP` supply;
- the rest of the C library the program's headers declare (Kore's hosted
  `<stdio.h>`, `<stdlib.h>`, `<string.h>` and `<errno.h>`, beside TCC):
  `kore-c-shim`'s `printf` family, every conversion C has but the wide ones and
  every float digit exact, `stdout` and `stderr`, the `strto*` parsers,
  `qsort`, `strerror`, the string routines and a per-thread `errno`, and
  `kore-c`'s own `abort`;
- descriptors over the capabilities it holds (`<fcntl.h>`, `<unistd.h>`,
  `<sys/mman.h>`: `open`, `close`, `mmap`, `munmap`), and
  `kore_fd_describe` (`kore/program.h`), described below;
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
  thread-local state, nor wait for another program thread — while one runs
  the host serves nothing else, the streams included. A thread that cannot park (any but the main thread)
  is refused, as is a launch with no face. A quarantined session can never
  run again.
- **The streams.** `stdout` and `stderr` reach the launch's interactive
  stream (`PROFILE_INTERACTIVE`), the terminal the program was run from,
  which only thread 0 may publish into. Any thread's write goes whole into
  one in-process ring under a futex lock, which a stream call holds across
  all its writes (POSIX's `flockfile`): the holder is the ring's one
  producer, so one call's output is never split by another thread's, and it
  kicks the host only if the host armed itself before parking. Every wait
  for room is bounded by the runtime's slice: the host kicks the main thread
  when its room is free, and a pthread, which the host cannot reach,
  re-reads the ring each slice. The host drains the
  ring into the stream's records as the launcher has room, parks on the
  stream's capacity when it has none — a slow terminal holds the program
  back rather than losing its output — and wakes the writers waiting for
  space, a pass at a time, serving the program's end and its sessions
  between passes. The host's own writes (a filesystem step's callbacks) are
  kept in order beside the ring, before any call another thread has in
  progress. A launch without the stream has no sink, and a
  write to either stream returns `EOF`.
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
  - `abort` takes `_Exit`'s path with the aborted status, 134, what a shell
    reports for a program `SIGABRT` ended: no `atexit` handler runs.
  - An end the host's loop serves drains the ring and closes the stream
    before the status, so the launcher shows all of the program's output
    first. An end committed on the host itself (a callback's `exit` or
    `abort`, a panic there) cannot wait: it relays what the launcher has room
    for, and closes the stream if that was all of it.
- **Without a host.** A launch that cannot carry one (no channel space or
  heap) runs the program on thread 0, as its C library alone would. Then the
  filesystem session is refused, and only thread 0's exit ends the program.

### Descriptors: C's spelling for a held capability

A program's authority is the capabilities it was founded holding. Each sits
at a fixed place: the boot block names it by a role and an index, and the
capability is a slot of the thread's arena. `open` is the readable way to say
which one is meant, and nothing more:

```c
int fd = open("/kore/memory", O_RDWR);
char *region = mmap(NULL, 1 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
...
munmap(region, 1 << 20);
close(fd);
```

- **The path is not a file.** `open` looks the string up in one static table
  (`crates/c/src/fd.rs`) whose rows are a name and the bootstrap coordinate it
  stands for. No filesystem is asked, and no name outside the table exists:
  `ENOENT`. The first row is `/kore/memory`, `role::HEAP@0`, the memory supply
  the allocator grows on.
- **It grants nothing.** `open` succeeds only for a capability the calling
  thread holds (`ENXIO` otherwise), with access its rights cover (`EACCES`).
  The only flags are the access modes. `close` forgets the number; the
  capability stays the program's, and mappings made through it stand.
- **A descriptor stores the coordinate, never the capability.** Capabilities
  are per thread on Kore, and `main` and every `pthread` have their own boot
  block. Each use resolves the coordinate in the *calling* thread's. Numbers
  are the process's, as C expects, from 3 (0 to 2 are the streams'), and no
  authority crosses a thread through one.
- **`kore_fd_describe(fd, &info)`** reads what the calling thread holds
  there: its kind (`KORE_FD_MEMORY` or `KORE_FD_DEVICE`), what its rights
  permit, and for a memory supply its whole length.

`mmap` on the memory supply answers fresh, zeroed memory, as mapping
`/dev/zero` does elsewhere. It is three of kore-rt's verbs, the sequence a
Wasmtime engine's guest memory takes: a lease carved from the supply's top,
clear of the allocator, which grows from its bottom; an address from the
free part of the task's link window (`Rt::link_free_span`); the whole lease
mapped there. The rules follow from that:

- **The caller never picks the address.** A hint is ignored, and `MAP_FIXED`
  is `ENOTSUP`.
- **Read, or read and write.** `PROT_EXEC` is `ENOTSUP`: memory a program
  wrote is never executable. `MAP_SHARED` and `MAP_PRIVATE` mean the same,
  since the region is this program's alone. There is no offset, and no
  `MAP_ANONYMOUS`: memory comes from a capability, so a descriptor names it.
- **Thread 0 maps, for whichever thread asked.** Every thread that may
  allocate is founded holding a share of the supply, and `open` and `mmap`
  hold the caller's access to that share's rights. A share carries no
  `REVOKE`, though, and a region that can be unmapped again is a carve its
  holder may revoke, so the caller posts its request to thread 0, which
  holds the supply's root, and waits as a filesystem session does. A mapping
  is therefore the process's: any thread may `munmap` it, whole, by its base
  and its length.
- **`munmap` parks the region.** The supply's cursors and the window's only
  advance, so carved bytes and placed addresses never return to them. The
  mapping is removed and a stale access faults, but the program keeps the
  lease and its address, and the next `mmap` of the same length takes it,
  zeroed again, before anything new is carved. One size mapped and unmapped
  in a loop uses one region. Ever-different sizes spend supply each time and
  end with `ENOMEM`. A program holds sixteen regions, mapped and parked
  together.

### A virtual device

A program launched with Kore's virtual-device profile has a face with the
virtual-device service: a device with no hardware behind it, for a driver to
be written against (Kore's `res/VIRTUAL_DEVICE.md`). Its name is a second row
of the table, and its descriptor has the shape Linux's UIO gives a userspace
driver (`crates/c/src/device.rs`):

```c
int fd = open("/kore/device/virtual/accel", O_RDWR);
volatile KoreVdevRegisters *regs =
    mmap(NULL, KORE_VDEV_REGISTERS_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED,
         fd, KORE_VDEV_MAP_REGISTERS);
uint8_t *lent = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, KORE_VDEV_MAP_MEMORY);
...
uint32_t word = 1;
write(fd, &word, sizeof word);          /* ring the doorbell */
read(fd, &word, sizeof word);           /* wait for the interrupt */
```

- **`open` starts the device.** The service starts the model the name
  stands for and writes its register block. A program has one device open at
  a time (`EBUSY`), and `close` gives it up.
- **Offset 0 is the register block**, one page the program and the device
  both write. `KoreVdevRegisters` (`<kore/program.h>`) is its layout, held to
  `kore-vdev`'s register constants by this crate's build.
- **Offset one page is memory to lend.** It is fresh and zeroed, the
  program's own, and a device address is an offset into it. The device takes
  a loan only in its reset state, so a driver maps it before its first status
  write. One region at a time, and it cannot be unmapped while the device is
  open (`EBUSY`), since the device may be using it.
- **`read` waits for an interrupt** and answers the device's interrupt count,
  one 32-bit word. Each interrupt is read once. **`write`** of one word rings
  the doorbell. **`poll`** waits for the interrupt with a timeout.
- **The interrupt and the doorbell never cross a thread.** They are
  capabilities, and capabilities are per thread, so the host founds the main
  thread holding its own share of the face and the thread controller hands
  the share on to every `pthread`. `read` parks the calling thread in the
  kernel on the interrupt count, and `write` posts its notice directly.
  `open`, the loan and `close` are requests to the service, which thread 0's
  face carries: they are posted to it and waited for, as a mapping of the
  memory supply is.

`write` on descriptors 1 and 2 writes to `stdout` and `stderr`, in order
with what `<stdio.h>` writes.

**`main` gets its arguments.** The launcher's page carries the name it
resolved the program by and the rest of the command line. The host reads it
once, on thread 0 before the main thread exists, and splits the rest as every
Kore program that wants words does: a POSIX shell's quoting, without
expansion. `argv[0]` is the name, or the empty string when the launcher gave
none, `argv[argc]` is null, and the strings are the program's to change. A
line that does not split (an unclosed quote, a lone trailing backslash) is
not handed to `main`: the host says why on `stderr` and the program ends with
status 5 before it runs. `envp` is the empty vector, since a Kore task has no
environment. A launch that carried no page runs with `argc` zero.

The C library's half is three weak references. Its start calls
`__kore_program_start(program, end)` when the program links libkore, asks
`__kore_program_arguments(&argv)` for the vector before it calls `main`, and
its exit calls `__kore_program_exit`. A program without libkore is unchanged. On
Kore those are the start files Kore ships with TCC. `tools/check-program.py`
(`make cross`, and CI per ISA) links the shared object as a distribution does
and checks it:

- a library with no entry point;
- its soname, nothing needed or undefined;
- static initial-exec TLS only;
- exactly the declared ABI exported: `kore.h`, `kore/program.h`, the
  `c-stdlib` names, the two lifecycle entry points, and the C library names
  Kore's hosted headers declare (`C_LIBRARY`).

Kore's `make keytest-tcc-x86` compiles a C program against these headers at
Kore's shell, links `libkore.so`, and runs it through every surface above. It
also compiles Kore's `hello.c`, written against the standard headers alone,
reads every line it prints off the terminal, and ends a program by `abort`
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
