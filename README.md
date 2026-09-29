# libkore — the Kore application standard library

`0.1.0-alpha.1` is the first **Rust foundation**, replacing the aspirational
scaffold on `alpha`. It provides a real filesystem API, common prelude, native
runtime macros, typed sleep, process facilities, automatically seeded random
values, hash collections and an opt-in driver substrate.
The `kore-ffi` package adds generated C bindings and a translated Zig module for
filesystem requests, clocks/sleep, random values, async task preparation,
joined thread/SPMD scopes and independent native create/join/detach, driven by
a Rust-founded host. It also exposes
opaque hash tables, C allocation functions and a Zig `std.mem.Allocator`
adapter over the application's installed heap.
It is `no_std`; all authority comes from the application's actual endowments.

```toml
[dependencies]
kore = { package = "kore-std", git = "https://github.com/MicroPerceptron/libkore", branch = "alpha" }
```

That branch is the integration destination; the alpha implementation is available
there after its PR merges. Pin a reviewed commit for reproducible applications.
The package name avoids collision with Kore's kernel package, also named `kore`.

```rust
# use kore_std as kore;
use kore::prelude::*;
use kore::fs::{self, TransferError};

// The caller supplies an attached client and an affine view it actually owns.
async fn store(fs: &mut Fs<'_, '_>, file: &View, bytes: &[u8])
    -> Result<(), TransferError>
{
    fs::write_all_at(fs, file, 0, bytes).await
}
```

Writes do not implicitly truncate, sync, or atomically replace a file. Use the
native `Fs::resize` and sync operations explicitly. Ordinary operations build on
DagFS; graph/revision/grant facilities remain under `kore::fs::dag`. See
[the filesystem contract](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/filesystem.md) for partial writes and cleanup.

| Feature | Effect |
| --- | --- |
| `fs` (default) | Native DagFS client and allocation-free transfer helpers |
| `heap` | Existing runtime heap engine; installs no allocator |
| `crt0` | Native process entry stub on Kore targets |
| `driver` | Existing native device host, register/DMA and ring APIs |

Returning executables can use ordinary Rust `fn main()` through the executable-only
[`kore-entry` adapter](docs/entry.md), with typed return codes, the native heap,
and a kernel-backed worker shutdown barrier.

`use kore::prelude::*` brings common types into scope. `#[kore::main]`,
`#[kore::task]`, `#[kore::spmd]` and `#[derive(kore::Bundle)]` work even when the
dependency is renamed; no hidden direct runtime/ABI dependency is required.
The driver feature does not turn arbitrary register sequences into safe hardware
experiments. Device-specific lifecycle, register semantics and DMA completion
ownership still belong to the driver.

`kore::ExitCode` supplies closed standard outcomes and 16 application-reserved
slots. `kore::exit_codes!` names those slots without accepting arbitrary integers;
native exit/join preserves the full signed 32-bit status. See
[native exit outcomes](docs/runtime.md#native-exit-outcomes).

See [C and Zig allocation](docs/allocators.md), [hash collections](docs/collections.md), [time and concurrency](docs/runtime.md)
and [random values](docs/random.md) for API examples, admission and failure contracts.

## Build and integration

The toolchain, dependency locks and native source revision are pinned. A normal
clone needs no sibling checkout, private cross-repository token or submodule setup:

```sh
cargo fetch --locked
make fmt test check docs owners cross
make miri bindings-check
```

For the full workspace checks, install the pinned C/Zig generation tools as
described in [language integration](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/languages.md). Ordinary Rust dependency builds do not need them.

The roughly 5 MiB `native/` export contains 14 required packages, including optional
and test dependencies, with unchanged source paths and manifests. It contains no
kernel or boot images. `make native-check` detects source drift. For work beside
a local Kore tree, follow [local integration](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/integration.md).

## Alpha boundary

| Language | This alpha |
| --- | --- |
| Rust | Working foundation with native ownership types |
| C | Generated filesystem, time, random, task and joined worker APIs |
| Zig | Target-specific C translation and generated module namespaces |
| Lua | Lua 5.5 on a worker thread, with `io`/`os` over the task's own filesystem |
| Python | Planned handwritten ownership/scheduling adapter |

Filesystem support remains a requirement for **every initial language surface**.
A clocks-only foreign binding would not satisfy that requirement. The C/Zig owner uses scoped generational handles and bounded copied buffers;
callbacks return to the native executor for each request. The `kore-lua` package
carries the pinned Lua 5.5 engine, its freestanding host and the `kore.*`
standard library a script sees; `io` and `os` are written over the task's own
filesystem rather than vendored, because both of Lua's are libc over POSIX and
this target has neither. `os.execute` and `io.popen` are absent, and arrive
through `Extension` like `kore.accel`: both launch a process, which is a service
face the embedding program holds. See
[`crates/lua/KORE-VENDOR.md`](crates/lua/KORE-VENDOR.md) for the provenance, the
selected libraries and every place the manual leaves a detail to the system.
Standalone foreign entry/CRT, allocator adapters and the bounded Lua driver
experiment interface remain follow-on work. See [the language contract](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/languages.md) for ownership, cancellation and validation boundaries.

The previous sketches and address/collection experiments remain in Git history
at `8a6381c`; their parallel runtime, allocator placeholder and empty language
headers are no longer part of the active library. See [provenance](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/provenance.md)
for upstream dependencies, licensing records and validation boundaries.


Pthread-shaped C/Zig lifecycle calls use a process-lifetime native controller;
join may originate on another worker and detached threads are reaped after exit.
`kore/pthread.h` provides lifecycle, TLS-key, cleanup and deferred-cancellation
aliases with default attributes; `c-stdlib` supplies their freestanding linker
symbols and compiler TLS destructor registration. Managed threads run TLS
destructors before native reclamation. Asynchronous cancellation is refused. See
[thread lifecycle and TLS](docs/runtime.md#independent-foreign-threads) for the
ownership contract and the remaining POSIX boundaries.

Atomic-word synchronization is available through `kore::futex` (including Rust
async waits), `kore_futex_*` in C, and `kore.futex.*` in Zig. Foreign waits block
native threads; see [runtime contracts](docs/runtime.md#atomic-word-waits).

### Fatal process diagnostics

The matching Kore #1004 kernel contains fatal application faults and Rust panics
within their process. `kore::crash` exposes bounded evidence collection, text
rendering, and explicit parent-side persistence. Reports carry a bounded task
name and opaque 128-bit public identity; IDs convey no authority. See
[crash reports](docs/crash.md) for lifecycle ordering and validation limits.
