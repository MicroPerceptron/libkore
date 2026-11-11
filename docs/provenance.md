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

The TLS lifecycle export recorded Kore `2a45f8bce5db65e7713f11c89d7eb5d2d9a04d3f`
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


## Closed native exit vocabulary

The current export records Kore `b4113c30a8f2ae95a02e511664798eb85505a7c6`:
14 native packages and 223 exact source files. It includes the intervening #962
owner updates and the closed exit vocabulary. Public integer constructors are
removed; the session builder requires the same `ExitCode` as the thread API.
The ABI layout remains signed 32-bit, with canonical report validation shared by
all consumers. The facade also re-exports the typed alias macro as `exit_codes!`.

That Kore revision passed host tests, all three architecture checks, user and
service verification, fresh boot/test builds, kernel and production QEMU gates,
20 repeated shell launches plus exact 256/257/271/-1 probes, and a screen-decoded
store launch returning 256. Digests and boot manifests verified. These are
emulator results, not physical-hardware acceptance. The known lifetime
capability-tag budget is unchanged.

For this export, `make fmt test check docs owners cross miri bindings-check`
passed, including the renamed-consumer exit alias and C/Zig declarations and ABI
assertions on all three Kore architectures. The complete local-owner override
passed its type-identity checks and duplicate-owner negative control. Both
workspace dependency locks remain unchanged and pass locked resolution; the
exporter verifies byte equality against the recorded Kore commit.

## Host volume access: `kore-host` over the exported `kore-vol`

The export now records Kore `ab6e7b157659cc4158ca30bb70a14bb64b0c6820`: Kore main
`e5e0b96a` (Cargo version 0.40.0, 2026-11-01) plus four commits on Kore's
`ft/amazing-cray-qf1tpm`. Three make `tools/kore-vol` a library with its binary
printing over it (`beda0947`, `6f56689d`, `1b6cb322`); the fourth (`ab6e7b15`)
makes a mount release every engine handle it opens on every path, so a
long-lived reader (a volume browser) cannot exhaust its session's 32 handles
on missing files, directories read as files, or reads it stops early. The library
returns `format`/`put`/`rm` results as data, mounts a volume read-only for
`list`/`read_to`/`walk`, names the DagFS `FORMAT_VERSION`, reports
`VolError::is_in_use`, and documents that its image lock is a per-process
POSIX record lock. The binary's output, refusals and exit statuses are
unchanged.

`tools/kore-vol` is a new owner root. Following its path dependencies adds the
filesystem service closure it drives — `kore-svc-dagfs`, `kore-svc-block`,
`kore-svc-device`, `kore-svc-mm`, `kore-svc-swap`, `kore-nvme` and
`kore-svc-extract-worker-c` — for 26 packages and 381 exact files. Publishing
that closure was an explicit decision: host programs (vKore first) must read and
write Kore volumes with the filesystem row's own engine rather than a second
implementation of the DagFS format. `crates/host` (`kore-host`) is the facade
they depend on; it adds no behavior of its own.

Locally, with the pinned nightly: `cargo test` for the workspace (every feature
except `test-foreign`), the native `kore-rt`/`kore-dagfs` suites and the
exported `kore-vol` suites (9 CLI, 5 library, 1 unit, at `1b6cb322`) against
this distribution's lock, the tool tests, the no-default-feature check, Clippy with
warnings denied, documentation with warnings denied, all three Kore user-target
checks for `kore-std`/`kore-ffi`/`kore-entry`, `check-entry`, Miri, the owner
graph, the native drift check and the local-composition fixture all passed.
`kore-host`'s own tests format, install, walk, read and remove through the
exported closure. Not run locally: everything needing Zig 0.16.0 (`zig fmt`,
`test-foreign`, `bindings-check`, `kore-lua`'s Kore-target build), because this
environment could not execute a Zig toolchain; CI's host job runs them. The
upstream device suite (loop-device GPT disks, as root) passed in Kore's tree.

`ab6e7b15` changes two files, both in `tools/kore-vol`, and nothing the
workspace builds besides. For it, the native drift check, the tool tests, the
exported `kore-vol` suites (9 CLI, 6 library, 1 unit; the sixth library test
drives a hundred of each refusal through one mount and then lists and reads
whole) and `kore-host`'s tests were rerun and passed; the rest of the gate set
above was not, since no other exported byte or workspace source changed.

A production x86 ISO built from Kore `1b6cb322` (`ab6e7b15` changes only the
host-side tool, which the ISO does not carry; `make iso-sealed-boot-x86
BOOT_FEATURES=boot PAGING_PROFILE=4k48 MODE=release`, SHA-256
`029ddd7e1e1cc1819c7e3ecf7ec22ffc3f0f6e01105972cd4d968313465aab23`) booted under
QEMU 8.2 TCG on the gate's topology: 19/19 services and beacons, `first-shell:
ready`, `app-store: seeded 44/44`, `home-skeleton: homes: 1/1`. After the machine
was stopped, the host-side `kore-vol` read the volume that guest had formatted
and populated. These are emulator results, not physical-hardware acceptance.

## Kore main, with the host volume procedure merged

The export now records Kore `7074bda28c416ffe34562857ee43cbc883482782`, main
once MicroPerceptron/kore#1475 landed. That pull request was merged by rebase,
so the four `tools/kore-vol` commits the previous export named are on main as
`e39d7308`, `99ef4b86`, `3445d543` and `9d616ce0`, and the exported
`tools/kore-vol` is byte-identical to `ab6e7b15`'s. `7074bda2` itself makes the
kernel access the ACPI PM1 registers at their width; it changes nothing in the
closure.

Main took 52 commits between `e5e0b96a` and the merge. Eleven reach the
closure, which keeps its 26 packages and now holds 382 files (`kore-window`
gains `desktop.rs`):

- DagFS. A cwd anchor, and `FsRequest::MatchView` to confirm that a remembered
  route still names the launcher's view (`45e7d02b`, across `kore-dagfs`,
  `kore-svc-dagfs` and `kore-abi`); `.` adds no path step, and `path::resolve`
  folds `..` and rooted paths (`87fa74e5`); the service reaps a dead view by
  probing its lineage in the kernel instead of trusting its record
  (`77f00903`, with `kore-rt`'s `probe_chan_send_lineage`).
- `kore-abi`: a launch context may fill the payload it borrows;
  `PROCESS_CONTEXT_BYTES` is `PROCESS_OUTPUT_BYTES`, and `PROCESS_IO_ABI`
  stays 13 (`44abeeca`).
- `kore-rt`: a mapped view's registry entry is retired exactly once
  (`602ab362`).
- `kore-driver`: `IntakeError::failed_before_device_publish` tells a DMA
  import refused at its CPU view from one refused at device publication
  (`8eacc00e`).
- `kore-window` and `kore-app`: System Settings (`047f82f2`, `751fa23c`,
  `387b9a08`), and the dock's capacity without the window runtime feature
  (`afa21902`).
- The services' `boot.toml` digests, resealed twice: over Wasmtime 48.0.8
  (`89a85e45`) and over the POSIX shims (`0ed97d12`).

No exported manifest changed, so `native/Cargo.lock` did not. The DagFS
on-disk format stays at version 5 (`DIGEST_FORMAT_VERSION`).

Locally, with the pinned nightly, the gate set the previous section lists passed
again on this export: `cargo test` for the workspace (every feature except
`test-foreign`), the native `kore-rt`/`kore-dagfs` suites, the exported
`kore-vol` suites (1 unit, 9 CLI, 6 library) and, as root over loop devices,
its device suite (4), `kore-host`'s volume tests (2), the tool tests, the
no-default-feature check, Clippy and documentation with warnings denied, the
three Kore user-target checks for `kore-std`/`kore-ffi`/`kore-entry`,
`check-alloc-symbols`, `check-entry`, Miri, the owner graph, the native drift
check and the local-composition fixture. The declaration half of
`bindings-check` ran without Zig: cbindgen 0.29.4 regenerates
`extern/c/include/kore.h` byte-identical, and the C consumers compile for all
three ISAs. The Zig namespace and the `stdlib.h`/`pthread.h` aliases derive from
that header's function list alone, so they are current as well. Not run locally:
`zig fmt`, `test-foreign`, the rest of `bindings-check` and `kore-lua`'s
Kore-target build; CI's host job runs them.

## A C program's runtime, over Kore #1494's kore-rt

The export records Kore `0018956a3aca19dc6acea8d28fb8ec9fadefcb09`, a commit
on MicroPerceptron/kore#1494's branch, which brings TinyCC to Kore's shell.
Eleven of the 26 commits since `7074bda2` reach the closure, which keeps its
26 packages and 382 files:

- `kore-rt` carries what `kore-c`'s host needs:
  - `startup::barrier`, the process barrier without the exit;
  - managed threads that inherit their founder's wake send;
  - `crash::report`, a panic's report without the thread's end;
  - also a per-thread C `errno` word, guarded worker stacks sized in bytes,
    and the verbs and bundle call a shell's private install takes.
- `kore-rt`'s thread controller, which `kore-ffi`'s pthreads run on, parks
  on the futex and never polls (Kore `0018956a`). Its threads get their own
  doorbells and the controller relays the wakes they cannot deliver to each
  other. A thread's end reaches it through the exit hook `thread::exit`
  runs, and its reap parks on a join arm. Here `kore-ffi`'s join parks on
  its retirement word, which the controller wakes. `kore-c`'s main thread
  holds a channel-space grant to mint the controller's doorbells from.
- `kore-abi` carries the install request and its replies, and the claimant
  band sized by driver rows (`MAX_CLAIMANT_FACES`), which the device service
  uses.
- The services' `boot.toml` digests were resealed three times.

No exported manifest changed, so `native/Cargo.lock` did not.

`kore-c` is new: the runtime of `libkore.so`, and the host of an independent
C program (`docs/languages.md`). It is the first libkore package a C program
on Kore links without a Rust host. Its shared object is a distribution's
one-line `cdylib` over it. `tools/check-program.py` links that object for the
three Kore ISAs and checks its shape and its exact exported ABI.

## A cancellation is a wake

The export records Kore `a4ab7234e2fe60b43a1df42f12be8f636255fc7a`, one commit
past MicroPerceptron/kore#1494's branch head `76fddb9f`. Two of the six
commits since `0018956a` reach the closure, which keeps its 26 packages and
382 files:

- `kore-rt`'s futex gains a wait another thread can end, and the wake that
  ends it (Kore `a4ab7234`). `futex::wait_unless` is `wait` with a caller
  condition read once the waiter is registered, as the word is, and always
  parks on the doorbell. `futex::wake_thread` kicks one thread's doorbell
  rather than a word's waiters, directly or relayed through the caller's hub,
  and reads the target's registration before it kicks. A condition stored
  with `SeqCst` before that wake is therefore never missed, and no slice is
  needed to notice it. Arm, register and check are one function that every
  doorbell wait shares, and the host suite drives it with the arm injected to
  pin that order. `futex::current_thread` is public: the band a thread
  records for its wakers.
- Its thread controller exposes the rule its own joins used:
  `Controller::slice` is unbounded where the controller can wake the calling
  thread, and `UNREACHED_SLICE` where nothing could. `Controller::wakes_threads`
  says whether a wake the calling thread sends reaches the threads it founded.
- The services' `boot.toml` digests are #1494's reseal over the parking
  controller (`76fddb9f`).

No exported manifest changed, so `native/Cargo.lock` did not.

`kore-ffi`'s cancellation points no longer poll. A managed thread records its
band when it starts; `kore_thread_cancel` stores the request and wakes that
band; `kore_thread_join` and a managed thread's `kore_futex_wait` wait with
`wait_unless`, with no timeout of their own. An idle cancellable waiter now
sleeps until something happens rather than waking every millisecond. Only a
thread the controller cannot wake keeps the bounded slice, now the
controller's `UNREACHED_SLICE`. A canceller whose wake cannot reach the
controller's threads is refused with EAGAIN, as create refuses it, and records
nothing. A host test drives `request_cancel` to that refusal. The metal
evidence is Kore's `keytest-tcc-x86`, whose libkore tour cancels a thread
parked in `pthread_join` and one parked in `kore_futex_wait` with no timeout.

The export then moved to Kore `40de25825eb1ea7816f7e7df86e116b3ad20d1fc`,
which closes a child's founding window. The controller sets a child's reach
bit only after `spawn_with_arg` returns, and the child ran its callback at
once. Its first wait could therefore start before the controller had
recorded that it can kick the child. A cancellable `kore_futex_wait` in that
window took the unreached slice and kept it until its first real wake.
`child_entry` now settles on its record until the founding is published,
which follows the reach bit, so a thread the controller founds with a
doorbell is reached from its first wait. A host test holds a founding back
and checks that the callback waits for it. The closure keeps its 26 packages
and 382 files. Besides `thread_controller.rs`, it carries the
`services/swap/boot.toml` digest of Kore's reseal over the cancellation wake
(`bfecfd17`).

## A relayed wake reaches its waiter alone

The export records Kore `7b9bd960374649a731132b3047e6e84190653b6f`, on
`ft/admiring-lovelace-69eiok`, which is stacked on `ft/brave-heisenberg-zgoaux`
and merges its head (`36b99426`). Two changes since `40de2582` reach the
closure, which keeps its 26 packages and 382 files:

- `kore-rt`'s thread controller relays a wake to the threads it was for and no
  others (Kore `b67f198`). Managed threads hold no send into each other's
  doorbells, so a wake between two of them goes through the controller's hub.
  The hub used to count relays, and the controller swept: it kicked every
  thread with a doorbell waiter, whatever word that thread waited on, so every
  relayed wake woke every parked thread in the program. Now a waker owes the
  hub the bands of the waiters it found and could not kick before it moves the
  hub's word: `wake` owes its first, `wake_all` all of them and `wake_thread`
  its target. The controller takes the owed bands after it reads that word and
  kicks each band that still waits. The registry's Dekker argument makes this
  lossless: a waiter that registers after the waker's scan reads the new word
  and never parks.
- The services' `boot.toml` digests are the reseal over that change
  (`920526a`). The founding wait leaves every service's image as it was: its
  own reseal (`36b99426`) rebuilt only the shell and the image archive.

No exported manifest changed, so `native/Cargo.lock` did not.

Nothing in `kore-ffi` or `kore-c` changes. A C program's futex wakes take the
controller's relay as before and reach fewer threads; `docs/runtime.md` now
says so. The metal evidence is Kore's `keytest-tcc-x86`. The libkore tour has
two threads take 400 turns through a mutex and a condition variable on the
futex while `main` waits parked on a word nobody stores. The controller admits
two managed threads at once on this stack, so `main` is the only parked
thread; Kore #1494's later `2a0730d`, which gives the controller's children
their founder's arena, lifts that bound, and the step then parks up to eight
more. On the image built from `4f913c4`, the relay before the founding wait
was merged, no wake reached it. On the same image with kore-rt's futex and
controller from `a4ab7234`, which count and sweep, it was woken 208 times and
the tour failed that step.

Kore's reseal over this export repeats that measurement on the merged tree
(`e0717eb`, built from `d64b7a5`, which pins it): every step ran to `exit 42`
and no wake reached the parked thread. The same tree with the futex and
controller of `36b99426`, the founding wait's own, which count and sweep, woke
it 206 times, and the tour failed that step.
