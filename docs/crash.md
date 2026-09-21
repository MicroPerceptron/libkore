# Fatal process diagnostics

This snapshot matches Kore commit `7785f129e661305f19076618c42ee5589a0b7a06`
([Kore #1004](https://github.com/MicroPerceptron/kore/pull/1004)). Updating the
library alone does not add the required kernel session effects.

`kore::crash` (also `std::crash` through `kore-entry`) reexports the native
`CrashBuffer`, `CrashRecord`, registration, collection and rendering interfaces.
A fatal application CPU exception or Rust panic terminates its process cohort.
It does not unwind or run user finalizers; a worker panic also terminates the
process. Kernel/system faults retain their fatal handling.

## Parent ownership and ordering

The parent may register a writable, exclusive RAM lease before starting the
child with `crash::register(&task, &lease, offset, length)`. Offsets and lengths
must be page aligned; the registration granule is currently 4096 bytes and the
record occupies 1024 bytes. Aliased, sealed, executable, overlapping or revoked
storage is refused. The buffer remains unmapped during the child's execution.
The child receives no authority over it.

After terminal notification, collect through the held owning Task capability
before reaping. Crash collection requires ACCESS and REVOKE; join-only observers
cannot read panic text or registers. Reap the stopped cohort before rendering or
awaiting file I/O. Optional diagnostic failure must not bypass process cleanup.
The first fault publishes immutable evidence before notifying waiters.

With the default `fs` feature, `PreparedDump::prepare` borrows a caller-owned
filesystem View, invalidates the file's record extent, and syncs it before launch.
After collection and reaping, consume `PreparedDump` with `persist` to write and
sync the record body, then publish and sync its magic. Errors distinguish body
write/sync from publication write/sync; transport failures are not automatically
replayed. The caller must exclude other writers. RAM capture is not a durability
acknowledgement. This is explicit file I/O, not file mmap or fault-time disk I/O.

## Names and public identity

Before START, an owning Task handle can call
`set_name(kore::abi::task::TaskName::new("my-app").unwrap())`. The copied UTF-8
label has a 63-byte bound, is assigned once, and conveys no authority. Worker
labels inherit their process name.

`TaskToken::identity()` returns `Result<TaskIdentity, Status>`;
`TaskToken::id()` returns `Result<TaskId, Status>`. The ID is an opaque 128-bit
value formatted as 32 hexadecimal digits. It is stable through exit until reap,
and distinct from scheduler keys or capability routing IDs. Querying requires
Task ACCESS; knowing an ID grants no task operation. Entropy provenance records
hardware/firmware, timing jitter, degenerate jitter or unverified initialization;
weak seeds do not promise unpredictability. IDs are scoped to one boot.

## Format and validation limits

The version-2 record includes observed registers, decoded fault data, partial
frame-pointer traces, bounded panic text, task name and public ID. Missing
observations are explicit. `crash::render` supports plain text and ANSI accents,
escaping untrusted controls. Kore's `tools/crash_dump.py` can inspect extracted
records on Linux/macOS and optionally use an explicitly selected ELF and source
root for LLVM symbol lookup. The format does not yet authenticate module identity
or provide an ELF core file or Linux DagFS partition reader.

The companion Kore PR has AMD/Intel QEMU fault and subsequent-launch evidence.
Physical-machine fault testing and end-to-end native crash-file persistence
remain outstanding; AArch64/RISC-V have compile/decoder coverage. The companion
also records an editor-input gate failure reproduced on its unchanged base.
Libkore's host, Miri and cross-link checks do not substitute for those tests.
