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
or pthread-style ambient authority is introduced.

Foreign callbacks must obey native thread restrictions, synchronize shared data,
return promptly where cooperative scheduling requires it, and never unwind or
longjmp across Rust. These are native-code FFI contracts, not script isolation.
Zig exposes the generated names under `task`, `thread`, `spmd` and `time`.
