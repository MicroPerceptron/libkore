# ax-exec

The allocator-owned, `no_std` native execution core shared by Axiom's host
heterogeneous adapter and freestanding runtimes. It depends on `ax`, `axnn`,
`axnn-cpu`, and the provider-independent `virtio-accel-core` byte sink contract.
There is no provider compiler, session manager, thread runtime, environment,
clock, or global allocator here.

This extracts the production native path. `axnn-vaccel::AxHetero` uses this same
partition compiler, fusion planner, typed storage, and instruction executor for
its native regions, including regions inside mixed CPU/provider schedules.
Its public host APIs retain their error types and calling conventions.

## Entry points

- `NativeProgram::compile_auto_in` retains a validated `.ax` graph and selects
  available native kernels and exact fusions using compiled CPU capabilities.
  `compile_in` accepts an explicit admission policy and a fusion toggle.
  `from_graph_in` adopts a `GraphBuf` without duplicating its encoded storage;
  `KernelPolicy::into()` preserves runtime capability, fidelity and scratch limits.
- `begin_in` / `begin_with_resources_in` borrow input bytes and resource views.
  Aligned, supported little-endian inputs are borrowed; other supported layouts
  use the same explicit decoding path as the host adapter.
- `NativeInvocation::advance(budget)` runs at most that many whole instructions.
  There are no hidden workers or waits. **One kernel is not preemptible**; use
  appropriately sized graph regions/tiles when latency needs a bound.
- `advance_into` validates all destinations before advancing and writes outputs
  only after execution succeeds. `AxOutput::pitched` supports offsets and padded
  rows. The output mapping is borrowed only for that call.
- `take_retained_in` transfers completed tensor owners to an in-process consumer
  without a pixel copy on the supported native storage path. It grants no
  cross-process or device access to that backing.
  `AxTensor::into_native_in` moves existing typed allocations into `NativeTensor`;
  borrowed/encoded results are materialized only when owned typed storage is requested.
- `prepare_native_in` explicitly creates the existing reusable arena. Ordinary
  execution uses automatic fusion but still allocates invocation metadata and
  intermediate storage. The prepared arena supports both whole-run execution
  and `begin_with_executor` for cooperative execution over retained storage.

## Cooperative prepared execution

`PreparedNative::begin_with_executor` borrows the arena, typed ingress, resources
and a runtime-owned `PreparedKernelExecutor` for one invocation. `advance(budget)`
executes at most that many instructions using the same native engine. Pure,
single-F32-output kernels may be delegated under the caller's explicit accuracy
policy. Unsupported kernels use their existing CPU plans; prepared fusions keep
their native execution and ordering.

A MoE instruction routes every token once, sorts the routes by (expert, token),
and runs them in waves of up to `MAX_WAVE_EXPERTS` distinct experts (bounded by
the source's `max_picks`). A wave's experts load together in one `load`, and the
wave is offered whole to `advance_moe_experts`: every route adds
`mixture * down_scale * down(sqrelu(up_scale * up(hidden[token])))` to its
token's output row, in route order, onto what earlier waves left there. So each
token accumulates its experts in ascending order on either side. The loaded
weights stay readable and unmoved until the wave completes; no load happens
while it is pending. An executor that declines a wave (Unsupported before any
Pending) leaves it, and every later wave of the instruction, to the native
kernel. This is an execution interface, not automatic GPU admission or a
measured acceleration claim.

The prepared arena accounts for the route records, one token's contribution and
the native kernel's scratch through `moe_prepared_scratch_elements`. There are no
per-dispatch allocations or model-specific batch sizes. `(instruction, wave)`
identifies an admission across waits, `wave` being the index of the wave's
first route. Routing hints follow the unique expert order. Declining after
Pending is an error, including when the runtime abandons a run with device work
outstanding.

`Pending` retains the instruction, output slot and input use counts. The caller
then awaits `PreparedInvocation::wait`, which lets the executor drive its own
retained IPC state and completion notification. CPU budget exhaustion is a
separate `Progress` result: it must not wait for a device notification. The
executor may refuse a kernel only before admission; returning Unsupported after
Pending is an error. Failed or abandoned runs expose no graph outputs.

An executor that can hold several pure instructions at once returns an
`admission_window` above one (clamped to `MAX_ADMISSION_WINDOW`). The engine then
offers the following pure, single-F32-output instructions through
`advance_chained` while earlier ones are still Pending, up to that window, so a
device can run a dependent run of them as one submission. An operand that a
pending admission produces arrives as `ChainedInput::Pending` (producer, offset,
length), never as stale host bytes; every other operand is final host data. While
anything is admitted the executor answers Pending or Unsupported, never Complete.
The engine retries admissions strictly in order with the same operand
descriptions, so outputs publish in program order. Unsupported ends the
lookahead: the engine drains what is admitted, then offers that instruction again
with host operands or runs it natively. Instructions that cannot join a window
(resource effects, MoE routing/loading, several results) run only when nothing
is admitted. A window of one, the default, is the strict path above, unchanged.

After all consumers of an intermediate have been admitted, the engine can call
`retire_unobserved` instead of requesting host publication. The executor may
retire that result without copying it back, while retaining storage needed by
pending consumers. Graph outputs, CPU-observed values, reference-check runs,
and resource updates still publish normally. Last-use information is prepared
once and shared by forks; it follows physical slot reuse rather than assuming
that a slot has only one writer. The default hook requests ordinary copy-back.

A MoE invocation may expose its final delegated Pending wave through
`moe_output_pending`. The window then admits following GPU consumers using
`ChainedInput::Pending`, while retaining that invocation and its expert-weight
leases. Earlier waves still complete before the source may load another wave;
no native operation or second MoE invocation runs across the retained admission.
When all output consumers have been admitted, the output is not a graph output,
and reference checking is off, `retire_moe_unobserved` can retire the final wave
without host copy-back. Otherwise normal completion publishes the output. Both
paths account for completed waves/routes and release source borrows only after
confirmed completion. Executors which do not implement these hooks retain the
serial behavior.

### Runtime-owned KV caches

`ResidentKvCache` identifies an external runtime's allocation and session epoch,
with the graph's validated cache geometry. It is inert metadata, never a host
pointer or device-memory authority. Bind it with
`NativeResourceBinding::resident_kv_cache`; the runtime retains the allocation,
mapping and in-flight work independently of graph futures. It must validate
both identities at every admission and keep identities unique while any old
binding remains usable.

`PreparedKernelExecutor::advance_kv` receives append and attention operations
against that allocation. Append writes the specified new rows; attention reads
only each query's causal prefix. These resource effects drain the pure admission
window before execution. Only the append dependency token and attention output
are returned to the graph. Every tensor borrow is call-scoped, including Pending;
the cache itself stays in the runtime. Plain CPU execution, reference checking
and an Unsupported response fail explicitly: there is no shadow CPU cache.

`axmodel::nemotron_h::Decoder::new_with_resident_kv_in` asks the runtime's factory
for each attention layer's exact cache geometry instead of allocating host KV
bytes. Reset and partial speculative acceptance change the next logical position;
later appends overwrite rejected suffix rows before they are visible to attention.
The runtime still owns allocations after decoder construction failure, cancelled
work or decoder drop, and releases them only after device work has retired.

The contract tests compare persistent storage with the CPU numerical reference,
including suffix overwrite, reset, geometry refusal and abandoned admission.
This API does not implement a GPU kernel, automatic placement, persistent Mamba
state, or a hardware throughput improvement. The Kore consumer is tracked in
[Kore PR 1134](https://github.com/MicroPerceptron/kore/pull/1134).

Within a window the engine may also offer a fused run through `advance_fused`:
a projection with no epilogue followed by `RELU` and `MUL(r, r)`
(`FusedRun::SquaredRelu`), or followed by an `ADD` with another value
(`FusedRun::Accumulate`, that value passed as `base`). A run is offered only when
each intermediate is read by the next instruction of the run and by nothing else,
and is not a graph output, so only the run's last output is ever published. It
occupies one admission, retires every instruction in it, and follows the same
ordering rules. An executor that declines gets the instructions one by one.

`FusedRun::RmsNorm` additionally combines a full-shape element-wise ADD or MUL
with weighted RMS normalization. `base` carries the normalization weight and
`width` identifies each reduction group. A full intermediate reshape is legal;
a partial view or another observer prevents fusion. The binary result retains
F32 rounding before reduction, and the executor must still honor the requested
accuracy tier.

Call-scoped tensor references never authorize DMA after `advance` returns. The
runtime must independently own imported buffers, admission and event retirement
state, including when a wait future is cancelled. An abandoned invocation cannot
reuse an unresolved executor: its `begin` must refuse until that work has been
drained or contained. Shared-memory ownership must be established by the runtime;
this API does not fabricate a route from a slice address.

Nemotron's `verify_with_executor` and `prefill_with_executor` use this path with
the decoder's configured accuracy. They share resource binding and successful
state publication with ordinary execution. Verification leaves state uncommitted;
prefill commits the complete chunk only after success. Their CPU-yield callback
and the executor's device wait remain runtime supplied. The existing synchronous
projection delegate remains available on the synchronous path.

### CPU placement and timing

`PreparedNative::set_profiler(clock)` enables `OpProfile`. Its original `ns`
and `runs` describe instruction lifetimes, including device waits. Concurrent
admissions overlap, so summing these times does not give turn time or CPU time.

The `cpu`, `fallback`, and `reference` fields count actual synchronous CPU
instruction attempts by operation class. They distinguish direct execution,
execution after a final unsupported device offer, and explicit recomputation
for `set_delegate_check`. The interval includes synchronous worker dispatch and
joining; it is wall time, not CPU busy cycles. A rejected fusion followed by
successful individual device execution, an offer retried after draining a
window, and a Pending observation do not count as CPU execution. Failed CPU
attempts do count. No clock reads are added when profiling is disabled.

For prepared MoE, `moe.delegated_waves/routes` and `moe.native_waves/routes`
count completed work. `moe.mixed_invocations` identifies an instruction that
completed device waves before switching to native execution. These counters
and the existing host phases are retained as execution progresses, including
when a later wave fails. Native expert time remains in `moe.ns` rather than
being counted again in `fallback`. Explicit expert reference checks go into
`reference`. MoE routing and source loading remain separate host phases.

These are engine execution boundaries. CPU work inside a runtime callback,
device driver, or the opaque legacy synchronous `ProjectionExecutor` is not
attributed by them. Runtime copy, IPC, cache maintenance, and worker scheduling
measurements are still needed to explain a complete turn.

Standalone `NativeProgram` admits one native CPU partition or a producer-free
input/constant passthrough graph, not arbitrary provider work. Unsupported kernels
are refused. Unconsumed node outputs are reclaimed immediately, including graphs
whose only public outputs are passthroughs.
It shares kernel selection with the heterogeneous path but does not replace
its provider cost model. No source compiler or implicit precision conversion
is introduced here.

`native` and `storage` are hidden implementation bridges for the in-tree host
adapter and its adversarial tests. Downstream adapters should use the root API,
not depend on mutable plan records inside those modules.

## Kore boundary

The final runtime supplies an allocator, CPU admission, region sizing, input
lease owners, event-loop policy, and output lifetime. Cancellation between native
calls or during a cooperative wait does not cancel independently owned provider
work or establish device quiescence.

`axmodel::nemotron_h::Speculative` exposes `prefill_with_executor` and
`continuation_with_executor` for a cooperative target beside an asynchronous
drafter. Both use the ordinary speculation transaction: accepted-prefix
selection, capture feeding and state commit are shared. The decoder's accuracy
policy is preserved. Dropping either future during a device wait leaves the
session interrupted and the runtime responsible for its outstanding work.
One-token prompt tails and plain decode steps keep their native path.

The complete `AxHetero` provider/session layer is **not yet `no_std`**. Separating
provider artifact preparation from portable asynchronous transitions is future
work. Do not block Kore's executor on host-style polling.

See [the two-window experiment and Kore contract](../../tools/ax-surface-lab/KORE_PREP.md).

## Checks

```sh
cargo test -p ax-exec -p axnn-vaccel
cargo check -p ax-exec --target aarch64-unknown-none
cargo check -p ax-exec --target x86_64-unknown-none
cargo check -p ax-exec --target riscv64gc-unknown-none-elf
```

Target checks establish compilation, not native execution or ISA support on
those machines. Execution still needs the runtime's admitted CPU contract.
