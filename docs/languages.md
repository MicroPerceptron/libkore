# C and Zig: one ABI over native ownership

`kore-ffi` provides a Rust-founded embedding ABI. cbindgen 0.29.4 generates
`extern/c/include/kore.h` from its actual `extern "C"` functions and `repr(C)` /
integer-representation types. The namespace generator groups those exported
names into `extern/zig/src/root.zig`. Zig 0.15.2 translates the C header for the
consumer's target at build time; generated host-specific layouts are not checked
in. `build.zig` exports the `kore` module, with `kore.fs`, `kore.dag`, `kore.time`, and the
complete declarations/constants under `kore.abi`.

Generation is inexpensive once the boundary exists. Rust lifetimes, affine
`View` ownership, futures, slices and native error enums are not themselves a C
ABI. The adapter below establishes those semantics; neither cbindgen nor Zig's
translator invents them. See [cbindgen's supported API shapes](https://github.com/mozilla/cbindgen/blob/v0.29.4/docs.md#writing-your-c-api)
and [Zig C translation](https://ziglang.org/documentation/0.15.2/#C-Translation).

## Native scheduling owns the request

A Rust host attaches `Fs` using its real founding face, then moves it into
`kore_ffi::Session`, after closing any external views opened before handoff.
The host awaits `Session::run(step, user)` on the existing
Kore executor. No second executor, global allocator or global filesystem root is
installed. The C/Zig callback is linked into that same application's image.
This is not yet an independent C `main`/CRT, registry-published static SDK, or a
POSIX syscall veneer.

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
sandbox; the process's Kore capabilities remain the security boundary. Lua and
Python adapters can later own this lifecycle instead of exposing these pointers
to scripts.

## Build and verify

Normal Rust consumers need neither generation tool. To develop this repository:

```sh
cargo install cbindgen --version 0.29.4 --locked
# Or use an existing official Zig 0.15.2 installation.
python3 tools/install-zig.py /tmp/libkore-tools
export ZIG=/path/printed/by/the/installer
make bindings
make fmt test check docs owners cross miri bindings-check
```

The installer supports the CI Linux x86_64 and local macOS arm64 hosts and verifies
pinned official archive checksums. Other hosts can supply an official Zig 0.15.2
binary through `ZIG`. `CBINDGEN` and `CLANG` also accept explicit tool paths.

`test-foreign` on `kore-ffi` links compiled C and translated Zig consumers into
host tests. It is disabled for native target checks and Miri; Rust pointer,
handle, buffer and scheduling tests still run under Miri. `bindings-check` checks
drift, compiles C and Zig consumers for x86_64, aarch64 and riscv64, and exercises
the Zig build-system module. Native Rust target checks use Kore's actual JSON
user targets, not merely stock bare-metal targets.

Local macOS package validation used the installed macOS 15.4 SDK selected for
that invocation only. This host's selected SDK lacks the `arm64-macos` libSystem
exports that Zig 0.15.2's build runner needs; C translation and freestanding
compilation do not have that dependency. No global SDK selection was changed.

These checks prove declarations, layouts, linked foreign calls, and admission
contracts. They do not claim that a C/Zig-founded standalone process has booted
on hardware. The concrete Rust-owned driver calls the existing native `Fs`
implementation; its founding/entry integration remains explicit in the host.
