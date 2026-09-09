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
