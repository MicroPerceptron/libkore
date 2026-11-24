# Backend adapter boundary

An adapter owns runtime policy; `ax` owns graph structure. Backend IDs are
assigned for one scheduling invocation and never appear in `.ax` bytes.

## Placement

Implement `CostModel` with:

- an integer backend count;
- exact operation/dtype/shape capability checks;
- estimated execution cost for one node; and
- transfer cost between an external tensor domain and backend domains.

The model may capture concrete dynamic shapes, power policy, queue depth,
device generation, and measured calibration data. Those values remain outside
the serialized language. Unsupported combinations return `supports == false`;
the adapter must not silently narrow or approximate them.

`Schedule::place` tracks tensor residency per backend with a bit table. A
constant or input imported once can therefore be reused by later nodes without
charging another transfer. `Schedule::build_plan` then returns ordered
partitions and the imports, cross-backend copies, and exports needed to execute
them.

An implementation ID is separate from a backend ID. An adapter can expose
multiple implementations on the same residency domain—such as dense and BSR
CPU `MATMUL`—with `CostModel::implementation_count`,
`supports_implementation`, and `execute_cost_implementation`.
`Partition::implementation` splits executable regions when necessary, but it
never creates a transfer by itself. The legacy low-level `place` and
`partition` functions remain dense-only; allocator-owned `Schedule` records
the selected variants. Override `partition_cost_implementation` when those
implementation families have different fixed setup or dispatch costs; its
default delegates to the existing backend-only `partition_cost` hook.

Adapters that enable `ax`'s `kernel-lowering` feature should use
`ax::node_kernel_request` as the canonical `.ax`-to-`axnn` bridge, then ask an
`axnn::lowering::KernelCatalog` to select among admissible candidates. This
applies to typed operations such as
`MATMUL`, not only elementwise nodes. Expose one scheduler implementation ID
for a native kernel family and keep scalar, SIMD, ISA-specific, packed, and
other tuned plans private to the compiled partition. A TOSA or provider recipe
can be another candidate when its primitive operations, precision policy,
scratch requirements, and provider capabilities are all admitted. This keeps
`.ax` lean while making every registered kernel reachable through automatic
lowering.

Adding a tuned kernel without registering a catalog candidate is incomplete:
the kernel is not part of automatic execution. New semantic families should
first gain a typed `KernelRequest` constructor, then candidates and invocation
support in their implementation crates. Placement compares those selected
native plans with provider and sparse implementation families using calibrated
partition, compute, and transfer costs. Catalog migration is incremental; the
current native proof covers the graph elementwise families and dense F32
`MATMUL`, while the typed request contract already represents reductions,
normalization, RoPE, convolution, and recurrent convolution for subsequent
catalog adoption.

After refinement, `Schedule::aggregate_cost` exposes the same complete-graph
score used to accept or reject region moves: partition launches, compute,
unique residency transfers, and output egress. Those units are deliberately
adapter-owned. They become dispatch durations only after calibration against
warm provider events.

## Cross-graph admission

Placement does not imply that a backend is idle when an invocation arrives.
`AdmissionPlanner` performs the next policy layer over caller-owned
`DispatchLane` state and already-compiled `DispatchCandidate` variants. A lane
is one independently reservable queue/SPMD capacity unit; several lanes may
name the same backend. Each request supplies an absolute ready tick, deadline,
and minimum fidelity.

The planner first preserves the highest-fidelity choice that can still finish
on time. When every option is late it chooses earliest completion before
fidelity, allowing a UI to select an interaction-quality CPU graph while a
resident inference graph owns an accelerator. A throughput-only inference job
uses `u64::MAX` as its deadline and equal fidelity for equivalent variants.

This API neither reads a host clock nor submits work. An adapter derives warm
duration estimates from provider telemetry, calls `admit`, submits the selected
executable, then calls `reconcile` with the event's observed completion tick.
Kore can do this from its event loop and SPMD capacity model. A host capacity
probe can do the same around provider events; no thread primitive belongs in
`ax`, and no queue/deadline metadata enters canonical `.ax` bytes.

`axnn-vaccel::Session::submit_many` is the provider seam for that loop. It
returns an opaque invocation token immediately after admission;
`poll_many_in` keeps events, programs, and fixed I/O buffers session-owned
until terminal completion and materializes outputs only then. The same program
key cannot reuse those buffers while its invocation is live, while distinct
cached programs may be in flight together.

`AxHetero::begin_in` and `advance_in` lift those tokens through complete placed
graphs. One advance performs at most one provider submission, one bounded
event poll, or one local sparse-partition transition. Terminal partition
outputs and boundary precision transforms remain in the invocation's explicit
transient allocator until the next partition becomes eligible. `Progress`
reports submitted, waiting, and completed partition phases; graph completion
is the observation an adapter pairs with its clock before calling
`AdmissionPlanner::reconcile`. Mixed CPU/accelerator schedules therefore need
no blocking admission, hidden executor, or thread per graph.

## Compilation and execution

For each `Partition`, iterate this range in `Schedule::order()`:

```text
order_start .. order_start + node_count
```

Compile those nodes with the adapter selected by both `Partition::backend` and
`Partition::implementation`. The adapter owns executable objects, device
buffers, fences/events, queues, and error translation. It should cache
compilation using the rules in
`CACHING.md`.

Allocation policy crosses this boundary explicitly. Host adapters may call
`AxHetero::compile_in(..., Global)`, but `AxExecutable<A>`, its schedule, and
partition metadata retain the caller's persistent
`A: core::alloc::Allocator`. `AxHetero::run_in(..., R)` independently assigns
blocking-run transients to `R`; `begin_in(..., R)` assigns the same storage to
a caller-owned invocation that survives across `advance_in` calls. Graph input
planes stay borrowed, so admission does not add a host-side full-input copy. A
kore adapter can therefore use different arenas or lifetimes for compilation
and dispatch. `begin_owned_in` is the explicit alternative for tiny dynamic
control planes that are easier to copy than pin across polls; it should not be
used to hide model- or frame-sized staging. Library code must not hide `Global`
behind a convenience constructor and accidentally turn a host choice into part
of the graph model.

The intended crate direction is:

```text
ax                         no_std language + scheduler
↑
axtosa                     TOSA frontend
↑
axnn-vaccel / kore service OpenVINO, CPU, NPU, or virtio adapter policy
```

OpenVINO and other host runtimes may depend on `ax`; `ax` must never depend on
them. A kore-native adapter can use the same interfaces with its allocator and
device transport, without filesystem, process, thread, or host-library APIs.

## First integration target

The Planet + Moon example is the first complete adapter slice. Planet's fused
FP16 `MATMUL -> TANH -> SIGMOID` graph is placed as one OpenVINO partition.
Moon's B8 (OCP E4M3) graph uses the same `.ax` path: capability-aware placement widens
F8 MATMUL inputs and constants for FP16 accelerator regions, while explicit
F8 narrowing casts remain CPU partitions so graph-visible precision is
preserved. Both paths cache provider programs by concrete batch signature.

`axtosa::lower_ax_in` remains the frontend bridge; neither TOSA nor `.ax` names
OpenVINO, an NPU, or a CPU. Precision adaptation and device placement stay in
the adapter and its cost policy.
