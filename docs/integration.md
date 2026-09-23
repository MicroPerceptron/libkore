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

### Optional supervisors and explicit endowments

An application can implement agent orchestration using `kore::runtime`,
`kore::process`, `kore::exec` and explicit filesystem capabilities. Enabling
these library features does not start a broker, mint a project view, or grant
installer authority. The optional supervisor implementation lives in Kore's
`user/apps/agent-supervisor` and `crates/agent-supervisor`; its policy is not
part of this SDK.

The parent retains the native task and cleanup owners. It transfers a counted,
versioned configuration package through a privately founded channel: selected
executable capabilities, a delegated exec connection, HostLink connections,
an optional project view, and parent/control endpoints. Executable delegation
uses `ExecutableCapability::transfer_derived` in the native exec client;
the endpoint's actual derive rights bound the transfer. A configured child
connection may mint empty child connections, but cannot create filesystem
homes or administer unrelated connections. Connection-tree retirement is
enforced by the native service against the authenticated ingress.

A directory view refers to the same underlying objects as other grants to that
directory. Starting a process neither duplicates the project nor makes it
empty. Omitting the project grant leaves no project inventory; there is no
fallback to the launcher's administrative filesystem. A child's exit or a
missing reply is not a cleanup acknowledgment. Reap descendants, retire the
connection tree, confirm transport closure, and drain retained rings before
issuing successor generations. Never replay an operation just because its
response was lost.

The current supervisor still uses Kore's transport-specific HostLink/MM
adapters and the broker/runner support crates. The SDK facade reuses the same
native owner types; it does not provide a second transport or supervisor
implementation. Native export, application composition, and guest boot gates
are distinct checks. The export revision in `native/provenance.json` is the
authority for which native APIs this checkout currently distributes.

Allocation-free executables can use `default-features = false` for the native
clock, process, task and random surfaces without linking hash collections.
The `collections` feature enables hashbrown and is on by default. It requires
the application's real allocator; enabling it never installs one. Filesystem
consumers can select `features = ["fs"]` independently. Verify these modes in
an actual `no_std` executable: checking a library alone does not detect an
unwanted global-allocator requirement at the final executable boundary.

The export includes Kore #962, its runtime task-provenance repair and bounded
SFNT metadata owner, resynchronized with current mainline. The exact source
commit is in `native/provenance.json`. #962 is still open and
provides required offset I/O, checked-close and macro behavior that current Kore
main alone does not yet provide.

Kore #962's facade is the migration source, not a second implementation to keep
evolving forever. This establishes the external home without silently retargeting
existing Kore applications. A later subtree/package integration should make Kore
consume the same native packages and deliberately transfer their source ownership
if desired. Recheck the full consumer graph, user images, QEMU gates and hardware
when making that transition.


### Fresh MM telemetry and recovery

The native export at Kore `52493011e3491f955f478f92e895731af9fa9604`
lets `RingMm::stats_observed` distinguish an ordinary fresh reply wait (`None`)
from a caller-selected observation deadline (`Some(deadline)`). Exec's 5 ms
abandoned-operation recovery slice must not be reused as the lifetime of every
fresh telemetry request: repeated timeout/recovery otherwise discards healthy
late replies and prevents downstream monitor observations. Cancellation retains
the exact original request, and bounded recovery does not replay it. This is an
upstream owner change; the export contains no independent implementation.
