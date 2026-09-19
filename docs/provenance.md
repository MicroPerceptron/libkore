# Provenance and acceptance

The starting scaffold is libkore `8a6381c` on `alpha`. The active Rust filesystem
and prelude source, renamed-consumer fixture and three user target descriptions
are adapted from Kore `98289c37c4f51ddcb247f7f0c5f30171399dc3a6` (PR #962):

- `crates/std/src/{lib,fs}.rs` → `src/`
- `tools/std-consumer/` → `tests/consumer/`
- `targets/*-unknown-kore-user.json` → `targets/`

Native owner implementations are exported from the exact later revision in
`native/provenance.json`, including the task-provenance fix exposed by this alpha's
Miri consumer. `make native-check` verifies every exported byte and rejects extra
implementation files. Their licensing is unchanged; `native/LICENSE` retains the
upstream notice. This repository retains its existing `LICENSE`; the
source project's copyright notice is preserved in `docs/KORE-LICENSE` for the
adapted files above. This alpha is not published to a package registry.

The broader approved proposals remain available at their pinned source:
[system library and language surfaces](https://github.com/MicroPerceptron/kore/blob/98289c37c4f51ddcb247f7f0c5f30171399dc3a6/res/SYSTEM_LIBRARY.md)
and [driver development](https://github.com/MicroPerceptron/kore/blob/98289c37c4f51ddcb247f7f0c5f30171399dc3a6/res/DRIVER_DEVELOPMENT.md).
Their proposed foreign APIs and driver recipes are not implied to be implemented
by the native re-exports in this alpha.

The previous placeholder APIs and the alternate address/collection sketches
remain available in Git history. Removing them from the active API avoids
advertising unimplemented allocation or creating a second source of runtime
ownership types.

## Evidence boundaries

This repository checks its resolved package graph, real renamed consumer,
allocation-free feature selection, all-feature build, documentation, Miri, and
the three actual Kore user targets. These are host/compiler checks, not a boot
or device witness. The optional driver feature exposes existing native contracts;
it does not certify arbitrary MMIO recipes or cancellation of active DMA.

At the original facade revision, #962 passed the full host suite, architecture checks,
kernel QEMU boottest and a founded filesystem command gate (write 16 bytes,
shorten to 2, remove). Its production gate twice lacked `net-load: PASS` despite
the other required witnesses completing. That unresolved acceptance result is
not promoted into a full pass by repackaging the library here. No new physical
hardware evidence or foreign-language ABI stability is claimed.

The alpha's Miri check exposed two task-erasure issues in that original runtime:
integer storage lost callable-pointer provenance, and a header-only pointer could
not recover the sibling future storage. The exported revision fixes these with
pointer-typed atomics and a storage-wide pointer derivation, preserving the same
atomic ordering, capacity and allocation behavior. A native regression covers
unadmitted drop, pending/ready polls, wake, and storage reuse; both it and the
facade consumer are included in `make miri`.

For this exported runtime revision, the alpha's host tests, no-default-feature
check, Clippy, generated documentation, owner-graph checks, all three custom
user-target checks and Miri checks passed locally. The local/Git composition
fixture passed its duplicate-owner negative control and native type-identity
checks. The export was also compared directly with the recorded upstream Git
commit using `sync-native.py --check`. Dependency documentation retains existing
upstream warnings; the facade and consumer documentation pass with warnings
denied. The advisory audit is configured in CI but was not run locally.

Kore was rebuilt and revalidated after the runtime repair: the full host suite,
all three architecture checks, separate production/test image builds and kernel
QEMU boottest passed. The production gate again completed the shell and application
seeding witnesses but failed solely on the missing `net-load: PASS` marker. No
clean-main A/B result establishes that failure as pre-existing, and the gate
remains unchanged.

The rebuilt founded filesystem gate also passed after the repair: guest keyboard
input produced the acknowledged 16-byte write, explicit shortening to length 2
and absence after removal, with the final prompt decoded from the real display.
Boot manifests, both architecture payload digests and all shipped lockfiles were
verified again after image generation.

## C/Zig and extraction follow-up

The first GitHub advisory audit blocked the full native lockfile on
[RUSTSEC-2026-0192](https://rustsec.org/advisories/RUSTSEC-2026-0192):
`ttf-parser 0.25.1` is unmaintained, with no patched version. Its owner was the
extraction worker reachable through DagFS's optional `index` feature. The warning
was not a reported exploit, but unchanged vendoring would not address maintenance.

Kore `9836aa3c68308d2b3f03de72289f5afdc5765537` replaces that worker dependency
with a bounded, allocation-free SFNT metadata derivative of the upstream checks
it actually used. The MIT notice and exact source provenance accompany it. This
is not a replacement outline renderer or whole-font validator. Its parser identity
is version 2, so the changed validation contract cannot reuse version-1 results.
Both distribution lockfiles now omit `ttf-parser`; CI still denies advisory
warnings. Axiom's independent rendering dependency remains in Kore and is outside
this export's dependency closure.

Native worker tests and Miri passed. A differential run over 370 installed
TTF/OTF/TTC files matched all glyph counts with no stricter refusals. Comparable
single-host benchmark samples were 1,434 ns/op before and 1,413 ns/op after, both
with a 400-byte output; this is not a statistically established speedup. No system
fonts were copied into either repository.

After this source change, Kore passed the full host suite, all architecture
checks, separate production/test builds, kernel QEMU boottest, the founded
filesystem command gate, manifests, both payload digests and lockfile checks.
The production gate again failed solely on the missing `net-load: PASS` witness;
there is still no clean-main A/B proof or new physical hardware acceptance.

The foreign adapter passed host tests including linked C and translated Zig calls,
Clippy, documentation, owner uniqueness, all three actual Kore user targets and
Rust Miri checks. Generated declarations and ABI assertions compile for all three
freestanding ISAs; the Zig build-system module also passed. The local/Git
composition fixture proves that `kore-ffi::Session` consumes the same native
`Fs` as the Rust facade after owner overrides. These are not a standalone
C/Zig-founded process boot witness; see [language contracts](languages.md).

## Runtime and random surfaces

The next alpha slice adds checked Duration/monotonic sleep, direct native task
and thread module paths, automatically seeded random values, and generated
C/Zig sleep, task preparation and joined worker APIs. The native export and
its source revision are unchanged. Foreign task batches produce actual native
SpawnTokens; the composition test also checks their type identity after Git/path
overrides. No new executor, raw capability owner, TLS layout or native worker
implementation is introduced.

The full local host/Clippy/docs/owner/custom-target/Miri/generated-binding matrix
passed, as did the negative-control local/Git composition fixture. Linked C/Zig
fixtures execute callback action returns and verify real native token cancellation
before admission. Random tests cover a ChaCha known answer, typed draws, bounded
unbiased sampling and unchanged output after initial/reseed failures. Host
parallel tests establish pre-launch refusal without callbacks or output writes.
These do not constitute a new native guest or hardware witness for foreign
worker execution; the prior Kore production-gate limitation remains unchanged.

## Mainline resynchronization

That resynchronization exported Kore
`75b8256ea60d4fc4a63bb17fc1379c21a98ce06d`, the #962 merge of main through
`75ca434a9`. It contains 14 native packages and 221 source files. This brings
in the memory/status, power/event, DagFS negotiated-transfer and driver changes
without importing the kernel, services or firmware payloads into libkore.

The offset-I/O merge preserves the service-advertised transfer limit instead of
capping every Rust operation at the 4 KiB protocol floor; requests still fit the
actual window, wire byte count and checked offset range. The foreign callback
adapter retains its explicit 4 KiB copied-buffer contract. The driver export now
contains the native accelerator extensions over published `virtio-accel-core`
0.3.7; its optional dependency is recorded in the native workspace lockfile.

For this export, `make fmt test check docs owners cross miri bindings-check`
passed with Zig 0.16.0. Both local/Git composition fixtures passed: one against
the exported workspace and one against the merged Kore checkout, including the
mixed-owner negative controls. The root dependency lock remains unchanged;
the native workspace lock adds only the newly reachable optional accelerator
packages. These checks do not establish physical-device acceptance.

Kore's resynchronized source also passed its full host suite, all three kernel
architecture checks, separate production/test builds, serial QEMU boottest,
production QEMU boot gate and founded filesystem command gate. The production
gate includes `net-load: PASS`, clearing the earlier missing-witness blocker on
this merged tree. No physical-hardware acceptance is inferred from these runs.


## Native TLS and independent thread lifecycle

The current export records Kore `2a45f8bce5db65e7713f11c89d7eb5d2d9a04d3f`
from PR #962: 14 native packages and 223 exact source files. It adds native TLS
provisioning, owner-checked native tickets and a process-lifetime controller for
cross-worker create/join/detach. Loader publication and the founded guest witness
remain in Kore; libkore imports their runtime contracts without copying kernel
or service implementations. Both workspace dependency locks are unchanged.

The C/Zig adapter generates default-attribute pthread-shaped lifecycle calls,
source aliases and optional freestanding linker symbols. Its callback context is
retained until native reclamation, and its shared result uses a pointer-typed
atomic. Host/Clippy/docs/owner/custom-target/Miri/generated-binding checks passed,
including real C/Zig host-refusal consumers and the returned-pointer provenance
check. Object inspection verifies all three thread linker symbols on every Kore
ISA. These checks do not establish physical-hardware acceptance or a standalone
foreign-founded process boot witness.

The export revision also contains the loader fix for pure zero-initialized TLS:
zero `.tdata` bytes require no source lease, while TLS allocation geometry remains
present. The runtime also releases the child TLS scratch-slot reservation on success
and refusal, and exposes its existing deadline-park helper for foreign join
without requiring doorbell authority. The loader itself remains upstream. Application targets select initial-exec TLS.

Managed foreign callbacks now have key/compiler TLS destructors, cleanup handlers,
explicit exit and deferred cancellation. Host/Miri tests cover key-generation
reuse, destructor repopulation/order, registration during teardown and cancellation
state. The checked-in guest harness exercises the actual adapter in founded
`hello`, with C-ABI Rust callbacks; separate C/Zig consumers verify generated
language bindings. It does not claim a C/Zig CRT, automatic stack unwinding,
initial-process exit hooks, asynchronous cancellation or physical acceptance.

## Compatibility tuple — current-Kore integration (libkore #19)

Recorded together, as the issue requires, so the graph can be reproduced rather
than reconstructed:

| component | value |
|---|---|
| Kore commit | `5108c7732e4383109bf86adb55f199fb9447bafb` |
| native export source revision | same — `native/provenance.json` |
| exported packages / files | 29 / 397 |
| libkore commit | this one |
| distribution lockfiles | `Cargo.lock` and `native/Cargo.lock`, regenerated together with the export |
| host commit | vkore `codex/vkore-40-current-kore` |
| Kore ISO SHA-256 | recorded in vkore's `config/real-stack.lock.mk` alongside the guest commit it was built from |

| Kore ISO SHA-256 (verified) | `6303e89f44e513b16a0b164140f464cb01f9d467e2cca10364af5e157399c2b5` |

The Kore revision is the head of `codex/vkore-994-hostlink`
(MicroPerceptron/kore#1015), which is **in review and not merged**. This export
is therefore a candidate, not a release baseline: re-run
`tools/sync-native.py <kore-checkout> --revision <reviewed-commit>` and update
this table when that PR lands. Nothing here should be read as an approved
distribution.

`kore-exec-client` joins the exported owners with this revision. It is the exec
protocol and caller face split out of `kore-svc-exec`, whose crate cites about
twenty sibling services in its founder manifest — exporting the client used to
mean exporting all of them. The service that answers exec requests stays in
Kore; libkore exports only what a caller links.
