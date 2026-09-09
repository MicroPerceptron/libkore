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
Include `<kore/pthread.h>` for `pthread_t` and source aliases for those three
calls. `kore-ffi/c-stdlib` additionally emits their actual freestanding linker
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
ECANCELED also consumes it, reporting native termination without callback return.
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
stacks/attributes, pthread-specific keys/destructors, compiler TLS destructors,
cancellation, cleanup stacks, `pthread_self`, `pthread_exit`, and inheritance of
the requesting thread's signal/floating-point environment remain unsupported.

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
