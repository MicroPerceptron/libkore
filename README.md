# libkore — application libraries for Kore

libkore `0.1.0-alpha.1` provides `no_std` libraries for programs running under
the [Kore capability microkernel](https://github.com/MicroPerceptron/kore).
It is an application facade, not a kernel package or a hosted operating-system
runtime. Filesystem, clocks, process operations and devices use capabilities
actually endowed to the process; selecting a Cargo feature grants none of them.
The libraries do not install a global allocator or start a second executor.

| Package | Use |
| --- | --- |
| `kore-std` | Rust facade: filesystem, prelude, time, random, collections, process and optional driver APIs |
| `kore-entry` | Executable-only adapter for a returning Rust `fn main()` on Kore |
| `kore-ffi` | C headers and Zig module over a Rust-founded host, including filesystem, time, random, allocation and worker APIs; no standalone C/Zig CRT |
| `kore-lua` | Embeddable Lua 5.5 engine and bridge; interactive and opt-in restricted worker policies |
| `kore-wasi` | WASI 0.2 component host composed by an embedder with its Wasmtime engine and native owners |

All five package manifests set `publish = false`; consume a reviewed Git
revision or a deliberately composed local checkout, not a registry version.
The WASI host supports the `wasi:cli/command` world and native-backed I/O,
filesystem, clocks and random imports; unsupported imports such as sockets
fail at instantiation. Its embedder configures the Wasmtime engine.
For this PR's source revision, name the `kore-std` dependency `kore` to avoid
confusion with Kore's distinct kernel package:

```toml
[dependencies]
kore = { package = "kore-std", git = "https://github.com/MicroPerceptron/libkore", rev = "06cddb199eee59739841887bca843d8ce493de7b" }
```

The pinned revision is a source dependency, not an ABI stability promise. Choose
a newer reviewed commit when updating. See [local integration](docs/integration.md)
before combining this Git source with packages from a local Kore checkout:
Cargo treats otherwise identical path and Git dependencies as different owners.

## First use

The collection facade can be exercised in a host test. A Kore process needs
its founded HEAP supply before an allocating collection can do the same:

```rust
# use kore_std as kore;
use kore::collections::HashMap;
let mut values = HashMap::new();
values.insert("ready", 1);
assert_eq!(values["ready"], 1);
```

Native service calls require an attached client, an executor and the relevant
endowed face; they are not host emulations. For example, given an owned DagFS
view and an attached filesystem client, this helper writes its complete input
starting at offset zero:

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
native `Fs::resize` and sync operations explicitly. An error reports the
acknowledged prefix, not a safe replay offset. Ordinary operations build on
DagFS; graph/revision/grant facilities remain under `kore::fs::dag`. See the
[filesystem contract](docs/filesystem.md) for partial writes and cleanup.

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

See [C and Zig allocation](docs/allocators.md), [hash collections](docs/collections.md),
[time and concurrency](docs/runtime.md), [random values](docs/random.md),
[Lua modes](docs/lua.md), and the [WASI host](crates/wasi/src/lib.rs) for API examples,
admission and failure contracts.

## Build and integration

The pinned Rust toolchain is `nightly-2026-07-13`. Kore executables use the
architecture-specific JSON user targets in `targets/` (x86_64, aarch64 and
riscv64gc), with `-Z build-std=core,alloc,compiler_builtins` and
`-Z build-std-features=compiler-builtins-mem`. A normal clone needs no sibling
checkout or private cross-repository token. Start with host-facing checks:

```sh
cargo fetch --locked
cargo test --locked -p kore-std
cargo test --locked -p kore-lua
```

For the full source/target matrix, install **official Zig 0.16.0 release** and
`cbindgen 0.29.4` using [language integration](docs/languages.md), then run:

```sh
make fmt test check docs owners cross
make miri bindings-check
```

The host tests exercise Rust logic and the linked foreign/Lua harnesses; they
do not prove founded service behavior or physical-device operation. Ordinary
Rust dependency builds do not need the C/Zig generation tools.

The roughly 5 MiB `native/` export contains 14 required packages, including optional
and test dependencies, with unchanged source paths and manifests. It contains no
kernel or boot images. `make native-check` detects source drift. For work beside
a local Kore tree, follow [local integration](docs/integration.md).

## Alpha boundary

| Language | This alpha |
| --- | --- |
| Rust | Working foundation with native ownership types |
| C | Generated filesystem, time, random, task and joined worker APIs |
| Zig | Target-specific C translation and generated module namespaces |
| Lua | Embeddable Lua 5.5: interactive `io`/`os` over the task's filesystem, or an opt-in restricted read-only worker |
| WASI | Embedder-composed WASI 0.2 component host; no standalone runner |
| Python | Planned handwritten ownership/scheduling adapter |

Filesystem support remains a requirement for **every initial language surface**.
A clocks-only foreign binding would not satisfy that requirement. The C/Zig owner uses scoped generational handles and bounded copied buffers;
callbacks return to the native executor for each request. The `kore-lua` package
carries the pinned Lua 5.5 engine, its freestanding host and the `kore.*`
standard library a script sees. Interactive `io` and `os` use the task's own
filesystem; `os.execute` and `io.popen` are absent. The restricted worker
opens only pure libraries and read-only `kore.fs` queries, with local quotas;
its embedder must provide a read-only filesystem face and enforce Stop/deadlines
by terminating the whole task. See [Lua modes](docs/lua.md) and
[`crates/lua/KORE-VENDOR.md`](crates/lua/KORE-VENDOR.md) for the provenance, the
selected libraries and every place the manual leaves a detail to the system.
Standalone foreign entry/CRT and a bounded Lua driver experiment interface
remain follow-on work. See [the language contract](docs/languages.md) for
ownership, cancellation and validation boundaries.

The previous sketches and address/collection experiments remain in Git history
at `8a6381c`; their parallel runtime, allocator placeholder and empty language
headers are no longer part of the active library. See [provenance](docs/provenance.md)
for upstream dependencies and validation boundaries.


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

## Contributing and licenses

Read [AGENTS.md](AGENTS.md) for ownership and verification rules before changing
code, and [integration](docs/integration.md) for the single native owner graph.
The repository retains its [MPL-2.0 license](./LICENSE); the exported native
sources retain their [upstream notice](native/LICENSE), and adapted Kore files
retain the [source notice](docs/KORE-LICENSE). The Lua vendor record is
[here](crates/lua/KORE-VENDOR.md). See [provenance](docs/provenance.md) for
the precise source and validation boundaries.
