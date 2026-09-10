# Time, tasks and joined workers

Rust applications can use `kore::time::sleep(Duration).await?` and
`kore::time::sleep_until(Uptime).await`. Duration conversion refuses values
outside u64 nanoseconds. The executor owns clock anchoring, timers and parking;
these futures require Kore's executor. `sleep_until` now takes the typed
monotonic instant; callers with raw nanoseconds use `sleep_until_ns`.
`sleep_for(u64)` remains the existing relative-nanoseconds compatibility API.

`#[kore::task]` and `#[kore::main]` remain the native macros; `kore::task`
also names the native executor module. `kore::thread` names the existing native
thread API. `kore::spmd` and its macro retain the native grid, capacity, founding,
join and refusal semantics. Features do not grant worker capabilities.

## Foreign sleep and async tasks

A filesystem callback queues `kore_time_sleep_for`, `kore_time_sleep_until`, or
`kore_time_yield` and returns. The same native task suspends, then invokes the
callback with the operation's completion. The one-command admission rule remains
in force: another submission after Queued returns Busy. Cancellation retains the
filesystem session's existing quarantine rule.

Tasks that need no filesystem use `KoreTaskStep`: return `KoreTaskAction` with
Done, Yield, SleepFor or SleepUntil. Sleep arguments are nanoseconds; SleepUntil
is absolute monotonic time, never wall time. A task that repeatedly requests
already-expired deadlines still yields every eight callback turns. Invalid
actions end the task with an explicit InvalidAction completion.

During Rust founding, create `KoreTaskBatch`, pass its exclusive pointer to the
C/Zig registration function, and let that function call `kore_task_prepare`.
Four native `#[kore::task]` declarations provide the bounded process-wide pool.
Queued means a native SpawnToken was prepared, not that the executor admitted it.

The host can `take(handle.slot as usize)` and pass the resulting native token to
`runtime::entry::run_main_with_tasks`, or `admit(&executor)` before running a
custom founded executor. Admission is per token: the returned count reports how
many were admitted; a refused token drops and reports Cancelled. This follows
the runtime's predeclared task-set admission rather than adding dynamic spawning
inside a callback or introducing a second executor. An example registration:

```c
static KoreTaskAction work(void *user) {
    unsigned *turn = user;
    if ((*turn)++ == 0)
        return (KoreTaskAction){ .operation = KORE_TASK_SLEEP_FOR,
                                 .nanoseconds = 1000000 };
    return (KoreTaskAction){ .operation = KORE_TASK_DONE };
}
/* batch is Rust-founded; turn outlives this prepared task. */
KoreStatus start(KoreTaskBatch *batch, unsigned *turn, KoreTaskHandle *ticket) {
    return kore_task_prepare(batch, work, turn, ticket);
}
```

Observe `kore_task_poll(ticket)` with cooperative waits, not an executor-blocking
spin. Complete, Cancelled and InvalidAction are acquire witnesses that the
callback no longer accesses user storage. Pending is not permission to reclaim
it. Release the terminal record with `kore_task_release`; copied old handles
cannot release a new generation even during concurrent reuse. Exhausted
generations retire permanently. A terminal record occupies a pool slot until
released. Rust dropping an **unadmitted** batch cancels its native tokens even
before their first poll. Forgetting tokens or abandoning an admitted executor
is not cancellation: the host must drive admitted tasks to completion or retire
the enclosing native task. The adapter exposes no forced task cancellation.

## Foreign threads and SPMD

`kore_thread_run(step, user)` runs one native worker and joins it before returning.
`kore_spmd_dispatch(tiles, step, user, out)` partitions the tile span across native
workers using the existing width/founding policy. Each callback receives its
disjoint half-open tile range; that range is not a physical CPU number.

Both are **synchronous scopes**. Call them from a suitable worker or synchronous
entry, not an executor callback that must keep other futures responsive. Their
native owner retains borrowed callbacks and data until join completes. Refused
means no callback launched; after release, an uncertain join never returns for
serial replay or lets the borrowed frame die. This preserves native fail-stop
behavior and means a post-launch lifecycle failure can keep the scope blocked.
No detached ticket table, arbitrary per-worker capability grant, nested dispatch,
or extra device authority is introduced by scoped dispatch.

Foreign callbacks must obey native thread restrictions, synchronize shared data,
return promptly where cooperative scheduling requires it, and never unwind or
longjmp across Rust. These are native-code FFI contracts, not script isolation.
Zig exposes the generated names under `task`, `thread`, `spmd` and `time`.


## Independent foreign threads

`kore_thread_create(&thread, NULL, start, argument)` starts a native callback;
`kore_thread_join(thread, &result)` joins it from any native worker and
`kore_thread_detach(thread)` gives up joining while retaining native reclamation.
Zig generates the same `thread.create`, `thread.join` and `thread.detach` faces.
Include `<kore/pthread.h>` for `pthread_t` and source aliases for lifecycle,
TLS keys, deferred cancellation and cleanup handlers. `kore-ffi/c-stdlib` additionally emits their actual freestanding linker
names on Kore targets; it never interposes host pthreads.

The first create starts one process-lifetime controller. It owns all native join
capabilities; portable handles contain identifiers, never arena slot authority.
Calls use bounded shared records and paced parks, including a 1 ms controller
scan while idle. They must run on native threads, not executor callbacks that
owe other futures progress. Capacity depends on native bands and the controller's
capability arena; ordinary slot exhaustion is refused before spending a band.

Create uses the installed process allocator for its handle and callback context.
On error the output is unchanged and the argument is not retained. On success,
the callback may already be running: synchronize argument access accordingly.
Join success consumes the handle and publishes the callback's pointer result;
Explicit exit joins successfully with its supplied pointer; cancellation joins
successfully with `PTHREAD_CANCELED`. ECANCELED consumes the handle only for
abrupt native termination that bypassed the managed exit path.
Other errors retain the handle for retry. Detach success consumes the handle but
keeps callback context alive until native reclamation. A per-result `Arc` and
pointer-typed atomic preserve the result's provenance and lifetime across that
handoff. Caller-owned arguments and returned pointees retain their own lifetime
requirements; a pointer into an exited thread's stack is not a valid result.

Return values are positive `KORE_THREAD_E*` constants (zero means success), not
writes to `errno`. Only null/default attributes are accepted. Native workers
receive a derived process heap, not arbitrary capabilities held by their creator.
Do not concurrently join/detach or reuse a consumed foreign handle. This is a
pthread-shaped lifecycle subset, not a complete POSIX implementation: custom
stacks/attributes, asynchronous cancellation, `pthread_self`, automatic stack
unwinding, and inheritance of the requesting thread's signal/floating-point
environment remain unsupported. Borrowing cancel calls must finish before a
consuming join/detach frees the same handle.

The native loader now retains pristine `.tdata`/`.tbss` templates and provisions
each new native thread's own architecture-correct TLS/TCB. Async tasks still see
the TLS of whichever native worker polls them; task-local state is separate.
Native guest witnesses cover TLS isolation, cross-worker joins, self-join refusal,
repeated detached cleanup and recovery after refused creates. Host adapter tests
prove refusal and pointer/lifetime behavior; they do not simulate a native thread.


The call shapes follow the Open Group specifications for
[`pthread_create`](https://pubs.opengroup.org/onlinepubs/000095399/functions/pthread_create.html),
[`pthread_join`](https://pubs.opengroup.org/onlinepubs/009695399/functions/pthread_join.html)
and [`pthread_detach`](https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_detach.html);
the supported subset above is deliberately narrower than those full contracts.

## Managed TLS teardown and deferred cancellation

`kore_thread_key_create/delete`, `setspecific` and `getspecific` provide 128
process-local keys with per-native-thread values. Deletion does not call a
destructor. Generation checks prevent a recycled key from observing an older
key's value. Exit clears each non-null value before calling its destructor and
allows up to four repopulation passes (`KORE_THREAD_DESTRUCTOR_ITERATIONS`).

`kore_thread_atexit` registers up to 128 outstanding compiler TLS destructors,
run in reverse registration order. Registration during teardown is supported.
With `c-stdlib`, `__cxa_thread_atexit` and `__cxa_thread_atexit_impl` forward to
this registry. The supplied application targets select initial-exec TLS, matching
Kore's static TLS loader. Custom build targets must do the same (or pass
`-Z tls-model=initial-exec`); dynamic `__tls_get_addr` is not supplied. Destructor code and its DSO must remain loaded until teardown;
this does not implement dynamic DSO unloading or Rust/C++ stack unwinding.

`kore_thread_cancel` requests cancellation; enabled targets observe it at
`kore_thread_testcancel`, foreign join and foreign futex waits. Disabled targets
retain pending requests. `setcancelstate` enables/disables delivery;
`setcanceltype` accepts deferred and refuses asynchronous with ENOTSUP, preserving
output on failure. Cancellation-aware waits use bounded 1 ms native parks and
one monotonic deadline. A canceled joiner leaves its target joinable.

On explicit `kore_thread_exit` or cancellation, caller-owned cleanup nodes run
LIFO, then compiler TLS destructors, then pthread key destructors. Normal callback
return runs TLS destructors without invoking unpopped cleanup handlers.
`kore_thread_cleanup_push/pop` operate on stable caller-owned nodes; generated
pthread macros provide lexical C scopes. Zig uses the generated functions and
must keep each node alive and unmoved until pop or exit. Destructors and cleanup
handlers must return normally, without exiting, unwinding or longjmp. Raw native
termination, faults and process destruction cannot promise application cleanup.

These lifecycle operations apply to callbacks created by `kore_thread_create`.
The controller keeps context storage alive until native reclamation, including
explicit exit, cancellation and detach. Registration on an unmanaged native
thread returns EAGAIN; this does not install an initial-process exit hook or add
teardown to raw native `thread::spawn`/scoped worker callbacks. Async tasks still
share their worker's TLS. Cancel/exit must not cross live Rust RAII guards or
executor callbacks: no unwinder runs to release those guards. Use explicit
cleanup handlers for foreign stack-owned resources.

The ordering and bounded key passes follow the [Open Group thread cancellation
and termination contracts](https://pubs.opengroup.org/onlinepubs/9799919799/functions/V2_chap02.html).
The restrictions above remain narrower than full POSIX.

`tools/check-thread-guest.py /path/to/isolated/kore-worktree` injects the checked-in
`tests/guest/thread_lifecycle.rs` as the founded `hello` workload (separate from its native stress witness) and
runs `make runtest-x86`. It uses a temporary Git export and complete local owner
patches, grants the fixture its heap in both the authenticated archive and
founding profiles, and uses a fresh test store before restoring tracked files. Its cases cover normal/explicit/canceled
exit, compiler and key destructor order, disabled pending cancellation, canceled
joiner ownership, and 40 canceled detached workers. This is actual foreign-adapter
execution in a QEMU process; C/Zig host consumers and all-ISA declaration/symbol
checks separately verify language linkage. It is not physical-hardware evidence.

## Atomic-word waits

Rust applications can use `kore::futex::{wait, wait_async, wake, wake_all,
WaitOutcome}` directly. These are reexports of the existing runtime owner;
there is no second wait registry. Use `wait_async` in Rust executor tasks.

C exposes `kore_futex_available`, `kore_futex_wait`, `kore_futex_wake` and
`kore_futex_wake_all` in `<kore.h>`; Zig exposes `kore.futex.*`. The foreign
wait blocks a native thread and must not run inside an executor callback.
For managed threads it is also a deferred cancellation point. Independent
controller-created threads are not endowed with doorbells, so their native futex
waits report `Unsupported`; this adapter does not create a peer wake mesh.
Foreign join uses the controller's deadline-park helper and works on these
threads. Cancellation is checked even when a foreign futex wait is unsupported.
Timeouts are relative nanoseconds; zero is a deadline of zero and
`KORE_FUTEX_INFINITE` requests an unbounded wait. Retry, spurious wake, timeout,
unsupported context and invalid pointer shape remain distinct outcomes.
Unsupported is a refusal, not a reason to spin or a fabricated timeout.

Words are caller-owned, writable 32-bit storage aligned to at least four bytes,
kept at a stable address until all waits and wakes finish. All concurrent accesses
must be compatible 32-bit atomics. For C, use atomic builtins on the `uint32_t`
word; for Zig, use `@atomicLoad`/`@atomicStore`. Publish the changed condition with
a sequentially consistent store **before** wake, and re-check the condition in a
loop after retry/wake. The pointer signature does not authorize ordinary C loads
and stores during concurrent use. Null and misalignment are rejected; other
pointer validity and lifetime requirements belong to the caller.

Wake-one returns whether a direct or relay notice was posted. Wake-all returns
the posted-notice count, potentially including one relay notice. These results
are telemetry, not waiter counts or proof that the condition holds. Both return
`KORE_FUTEX_INVALID_ARGUMENT` for invalid pointer shape. This API is process-local;
it is neither a Linux futex syscall ABI nor a cross-process synchronization API.
Foreign async wait submission is not provided by this blocking adapter.
