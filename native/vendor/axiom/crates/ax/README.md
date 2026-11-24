# ax

`ax` is axiom's backend-agnostic tensor graph language and scheduler core.
Serialized graphs use the `.ax` extension.

## Dependency boundary

The core crate is `no_std` and depends only on `axnn`'s tensor vocabulary. It
does **not** depend on OpenVINO, Core ML, operating-system APIs, filesystem APIs,
threads, clocks, or `virtio-accel` providers.

There are two storage modes:

- borrowed graphs plus caller-provided scheduler slices allocate nothing;
- `GraphBuf<A>` and `Schedule<A>` own their storage with an explicit nightly
  `core::alloc::Allocator` supplied by the caller.

No API assumes a global allocator. A kore allocator can therefore be passed to
`Vec::new_in`-style APIs without introducing a host runtime.

Provider and kernel crates sit outside this boundary:

```text
frontends ──> .ax / ax::Graph ──> ax scheduler ──> placement plan
                                                     │
                         ┌───────────────────────────┼──────────────┐
                         ▼                           ▼              ▼
                    CPU adapter               Accelerator      other adapter
                    executable                 programs
```

Adapters supply capability and cost information through `ax::CostModel`, then
compile placed regions using their own runtime. The `.ax` language contains no
provider identities or host handles.

A placement also carries an adapter-owned implementation ID. This lets one CPU
residency domain choose, for example, a dense or BSR-packed kernel without
inventing a second backend and charging a fictitious tensor transfer. The core
uses dense implementation `0` by default; adapters opt into additional IDs
through `CostModel` when they can compile and execute them. Aggregate placement
charges `partition_cost_implementation` at every backend/implementation run;
the default delegates to `partition_cost`, so existing backend-only cost
models retain their behavior.

Adapters that need automatic kernel selection opt into `ax`'s
`kernel-lowering` feature. Semantic nodes then reach tuned kernels through one
allocation-free lowering seam:

```text
.ax NodeView
    -> ax::node_kernel_request
    -> axnn::lowering::KernelCatalog
    -> selected backend-private plan (scalar / SIMD / ISA / accelerator recipe)
```

`.ax` therefore records operations such as `ATAN2`, `ROUND`, and `SELECT`, not
an ever-growing set of machine-specific kernel variants. `axnn` owns candidate
admission and selection using shape, dtype, accuracy, scratch, cost, and
runtime-supplied capability constraints. The scheduler sees only the adapter's
implementation family; it does not encode scalar, SIMD, ISA, TOSA, or provider
details in the graph format.

The hot structural operations use the same bit-packed approach as axiom's
existing `bitmat` kernels: a dynamic `u64` adjacency matrix, sparse set-bit
edge walks, a packed ready frontier for placement-aware clustering, cached
backend support rows, and portable SIMD row algebra. Clustering therefore
walks ready nodes and retired edges instead of rescanning the complete order
for every emitted node. The x86_64 build uses four `u64` lanes (one AVX2
register under `target-cpu=native`); graph capacity is no longer limited by the
older fixed 1024-node heapless buffer.

Execution and transfer costs are non-negative, but optimized placement must
compare aggregate scores whose implicit deltas can be negative. It also uses
`u64::MAX` as the unavailable-transfer sentinel. `Schedule` caches the best
regional weights once per backend/node, but retains exact `u64` values for the
authoritative comparison: UF8 E5M3 maps nearby costs such as 1000 and 1001 to
the same byte, and E4M4 has only 248 as its largest finite value. UF formats
remain suitable for optional coarse ranking or telemetry, never for accepting
or rejecting a placement without an exact fallback. On a 4K-node/two-backend
graph, exact cached regional weights occupy roughly 64 KiB, small beside the
dependency matrix while avoiding repeated model probes during fixed-point
refinement.

## `.ax` version 1

The canonical container is little-endian and consists of:

1. a 64-byte header;
2. fixed-size tensor records;
3. fixed-size node records;
4. flat tensor-reference and dimension tables;
5. opaque per-node attribute bytes;
6. constant tensor bytes.

Parsing borrows the source bytes. Encoding writes into caller-provided storage,
or `GraphBuf::encode_in` sizes an owned buffer using an explicit allocator.
Scheduler adjacency matrices and work queues can likewise be caller-provided or
created with `Schedule::analyze_in`. Graph capacity and allocation policy remain
deployment choices rather than heap or host assumptions.

Cross-graph admission is also allocation-free. `AdmissionPlanner` selects among
precompiled backend/quality candidates using caller-owned lane availability,
absolute deadlines, and measured warm durations. This keeps interactive frame
policy usable from a kore event loop without putting clocks, queues, threads,
or provider identities into the graph format.

See [FORMAT.md](FORMAT.md) for the byte-level contract,
[ADAPTERS.md](ADAPTERS.md) for provider integration, and
[CACHING.md](CACHING.md) for repeated-graph and intermediate-value reuse rules.

## Scheduling pipeline

```text
Graph::parse
    -> analyze       dependency bit matrix + topological order
    -> lower         semantic nodes -> automatically selected kernel plans
    -> place         capability/cost model -> backend + implementation per node
    -> partition     executable implementation regions + backend transfers
    -> admit         deadline/fidelity -> compiled variant + available lane
    -> adapter       compile/cache regions and submit device work
```

`axtosa::lower_ax_in` is the first frontend bridge. It validates TOSA outside
this crate and emits canonical `.ax` for the attribute-free fused neural path.
That dependency direction is intentional: `axtosa -> ax`, never `ax -> TOSA`.

## Neural lowering and streaming adapters

The [multi-target streaming guide](../../docs/ax-streaming-inference.md) covers
packed quantized graph execution, explicit GGML block import, target-specific
capability admission, kernel explanations, and bounded independent request
streaming. It also states the boundary between ingress accounting and physical
memory-fabric working-set guarantees.
