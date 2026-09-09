# libkore — the Kore application standard library

`0.1.0-alpha.1` is the first **Rust foundation**, replacing the aspirational
scaffold on `alpha`. It provides a real filesystem API, common prelude, native
runtime macros, clocks, process/random facilities and an opt-in driver substrate.
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

`use kore::prelude::*` brings common types into scope. `#[kore::main]`,
`#[kore::task]`, `#[kore::spmd]` and `#[derive(kore::Bundle)]` work even when the
dependency is renamed; no hidden direct runtime/ABI dependency is required.
The driver feature does not turn arbitrary register sequences into safe hardware
experiments. Device-specific lifecycle, register semantics and DMA completion
ownership still belong to the driver.

## Build and integration

The toolchain, dependency locks and native source revision are pinned. A normal
clone needs no sibling checkout, private cross-repository token or submodule setup:

```sh
cargo fetch --locked
make fmt test check docs owners cross
make miri
```

The roughly 5 MiB `native/` export contains 14 required packages, including optional
and test dependencies, with unchanged source paths and manifests. It contains no
kernel or boot images. `make native-check` detects source drift. For work beside
a local Kore tree, follow [local integration](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/integration.md).

## Alpha boundary

| Language | This alpha |
| --- | --- |
| Rust | Working foundation with native ownership types |
| C | Planned generated ABI; no exported C functions yet |
| Zig | Planned translated C surface and allocator adapter |
| Lua / Python | Planned handwritten ownership/scheduling adapters |

Filesystem support remains a requirement for **every initial language surface**.
A clocks-only foreign binding would not satisfy that requirement. Managed foreign
handles, declarative bindings and the bounded Lua driver experiment interface
are follow-on work, not features claimed by this version.

The previous sketches and address/collection experiments remain in Git history
at `8a6381c`; their parallel runtime, allocator placeholder and empty language
headers are no longer part of the active library. See [provenance](https://github.com/MicroPerceptron/libkore/blob/alpha/docs/provenance.md)
for upstream dependencies, licensing records and validation boundaries.
