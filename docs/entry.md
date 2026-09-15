# Returning Rust programs

`kore-entry` composes the Kore library with the native compiler entry adapter.
It belongs only in the final executable. Reusable libraries keep depending on
`kore-std`, and resident services keep their existing service entry points.
The adapter requires matching Kore process-drain and crash-report session support
(Kore #1004, including #1002); updating
only libkore on an older kernel is insufficient.

Name the executable dependency `std` so rustc selects this facade and its
edition-specific prelude:

```toml
[dependencies]
std = { package = "kore-entry", git = "https://github.com/MicroPerceptron/libkore", branch = "alpha" }
```

```rust
fn main() -> Result<std::ExitCode, std::ExitCode> {
    let values = vec![1, 2, 3];
    assert_eq!(values.iter().sum::<i32>(), 6);
    Ok(std::ExitCode::OK)
}
```

There is no `no_main`, handwritten `_start`, entry attribute, or application
language-item feature. `fn main() {}` also works. The alias does not claim full
compatibility with upstream Rust std: Kore's implemented APIs and authority
requirements still apply. Arguments remain on the process page, accessible
through `std::runtime::entry::thread_context()` and `std::process::map`.

Build with the pinned Kore nightly, its architecture-specific user target,
`-Z build-std=core,alloc,compiler_builtins`, and
`-Z build-std-features=compiler-builtins-mem`. The existing loader/PIE and
initial-exec TLS requirements remain. The distributed target files retain frame
pointers for bounded crash-stack capture. Renaming the dependency is a manifest
choice; it does not replace the installed Rust sysroot or affect other projects.

The executable adapter installs the native process heap. Allocation still draws
from its founded HEAP supply, and an absent or exhausted supply cannot create
memory authority. This same global allocator serves native Rust and the C/Zig
default allocation adapters when they are linked into the executable.

## Return and teardown

Accepted return types are `()`, the closed `ExitCode` vocabulary,
`Result<T, ExitCode>` for a supported `T`, `!`, and `Infallible`. Implementing
`Termination` for an application type can select a named exit code but cannot
construct arbitrary integers. An integer return from main fails to compile.
The signed 32-bit exit report preserves 256, 257, 271 and negative outcomes.

After the main frame returns and its locals drop, the runtime uses the initial
thread's authenticated SELF handle to request the process shutdown barrier.
The kernel closes task admission, stops every peer in that exact generational
address space, then reaps the stopped tasks. This includes controller children,
native detached workers, and workers whose handles were transferred. A task
image retains its process membership until physical teardown, even after its
scheduler key disappears; a pending reaper cannot produce an early success.
Workers cannot invoke this barrier for their parent or for another process.

If every peer had already exited, the runtime runs the loader's snapshotted
finalizers in reverse dependency and reverse array order, once, while modules
remain mapped. A forced worker stop skips user finalizers: interrupted code
may leave application locks or shared state incomplete. Worker TLS destructors
and stack unwinding are not promised on hard termination. Explicitly join
workers whose orderly cleanup is required. An idle controller is still a live
task and therefore makes this initial implementation take the forced path.

A fatal Rust panic reports the typed PANIC outcome without unwinding and
terminates the entire process, including a panic in a worker. Fatal paths skip
user finalizers and TLS destructors. The parent can collect bounded crash
evidence before reaping; the launcher's root reaper enforces the physical
worker-stop barrier before shared mappings are reclaimed. See [crash reports](crash.md)
for the parent-owned buffer and persistence contracts.
Reentrant finalization cannot invoke the finalizer list twice. Refused or deferred
physical teardown never authorizes reuse of the process's shared storage.
Such failures remain pending; this is not a bounded recovery guarantee for a
CPU or kernel that cannot complete the scheduler rendezvous.

This does not add the upstream `thread_local!` macro, C `main`, Zig `main`,
process signals, or C++ exception unwinding. Existing managed foreign-thread
TLS destructors and cooperative cancellation keep their existing contracts.
