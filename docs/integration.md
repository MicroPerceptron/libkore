# One native owner graph

`native/` contains a mechanically exported, committed dependency closure from
Kore. `native/provenance.json` records the exact revision, every source file hash
and the package/workspace mapping. The facade imports these packages by path;
they all refer to the same ABI/runtime owners. `make owners` checks the resolved
graph, including all optional features, for duplicate Kore packages and any
accidental dependency on the kernel.

## Updating the native export

Make native behavior changes in the owning Kore crate and validate them there.
Then export a committed revision:

```sh
python3 tools/sync-native.py /absolute/path/to/kore --revision COMMIT
cargo fetch
cargo fetch --manifest-path native/Cargo.toml
make fmt test check docs owners cross miri bindings-check
```

The importer follows normal, optional, target-specific, build and test path
dependencies. It copies committed Git bytes, never working-tree changes, and
preserves relative paths. Existing exports must pass their drift check before
replacement. Commit both lockfiles and the provenance record. A manually edited
snapshot must be resolved deliberately upstream; regenerating may not silently
erase the edit. The snapshot is one source per owner, not a parallel allocator,
runtime or handle implementation to evolve independently.

## Composing with a local Kore checkout

Cargo treats a path dependency and a Git dependency as different packages even
when their source and version match. An `Fs`, capability or ABI type from one is
not interchangeable with the other. An application depending on **libkore via
Git** can patch its native roots to a local Kore checkout:

```sh
python3 /path/to/libkore/tools/local-kore.py /path/to/kore > /tmp/libkore-local.toml
cd /path/to/application
cargo --config /tmp/libkore-local.toml metadata --all-features --format-version 1 > /tmp/libkore-local.json
python3 /path/to/libkore/tools/check-owners.py < /tmp/libkore-local.json
cargo --config /tmp/libkore-local.toml test
```

The generated file uses Cargo's supported [`patch` configuration](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#the-patch-section)
and targets libkore's Git source. Use `--source URL` if the application's Git
source differs. It validates package names and paths and changes no manifests.
These Git patches cannot override path dependencies when building libkore itself
as a path package; doing so would leave the exported owners in place. The
negative-control check below explicitly proves that mixed owners are rejected.

For an uncommitted libkore tree, this helper builds a temporary local Git
snapshot and tests the actual downstream composition without changing either
checkout or the distribution locks:

```sh
python3 tools/check-local.py native
python3 tools/check-local.py /path/to/kore
```

It first requires the owner checker to reject an unpatched mixed graph. Then
it applies the complete override, checks uniqueness again and compiles functions
that pass native ABI, runtime, filesystem and affine view types across the
facade boundary, including consumption of the same native `Fs` by
`kore-ffi::Session`. A stale checkout must fail compilation, not be bypassed with
casts. Downstream applications with additional independent pins must reconcile
those as well. Local overrides change the consumer's lock; do not substitute it
for either distribution lock.

## Bootstrap and future ownership

The initial export includes Kore #962 plus its runtime task-provenance repair and bounded SFNT metadata owner;
the exact source commit is in `native/provenance.json`. #962 is still open and
provides required offset I/O, checked-close and macro behavior that current Kore
main alone does not yet provide.

Kore #962's facade is the migration source, not a second implementation to keep
evolving forever. This establishes the external home without silently retargeting
existing Kore applications. A later subtree/package integration should make Kore
consume the same native packages and deliberately transfer their source ownership
if desired. Recheck the full consumer graph, user images, QEMU gates and hardware
when making that transition.
