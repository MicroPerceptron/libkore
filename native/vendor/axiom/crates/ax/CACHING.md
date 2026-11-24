# Repeated graph and value caching

Caching belongs at three separate layers. Keeping them distinct avoids treating
a changing frame as though the compiled program changed.

1. **Structural cache** — parse, analyze, place, and partition once for exact
   canonical `Graph::as_bytes()`. A cache entry must also include cost-policy and
   backend-generation revisions.
2. **Compiled-region cache** — compile each partition once per backend, concrete
   dynamic-shape signature, device generation, and relevant precision policy.
   Expanded FP8-to-FP16/BF16 constants belong here, so immutable weights are
   widened once rather than once per dispatch. BSR-packed sparse constants
   belong here too; include the selected implementation and packing policy
   (such as block width) in the cache key so they are packed once rather than
   rescanned on every dispatch.
3. **Intermediate-value cache** — memoize a pure subgraph only when every input
   buffer version and concrete shape matches. Nodes marked `ORDERED` or
   `MUTATING` are never cacheable. `NodeView::cacheable()` exposes that rule.

For animation, frame coordinates normally change every invocation, so caching
the final frame rarely helps. Structural and compiled-region caching should
still remove parsing, placement, graph construction, and weight-expansion work.
Intermediate caching helps when a branch depends only on constants or on a
slower-changing input (for example, static terrain plus a moving moon).

The `axrd` boot study follows this split: light centers and strengths are nine
scalar graph inputs, while the multi-megabyte scene graph, schedule, provider
program, glyph coverage, and provider I/O buffers remain resident. Canonical
`current + (target - current) * weight` regions stay ordinary SUB/MUL/ADD nodes;
the portable TOSA CPU provider recognizes their SSA use graph and executes each
single-consumer region as one strict SIMD instruction. Other providers remain
free to perform the equivalent fusion without an `axrd`-specific shader or op.

Cache lookup, eviction, clocks, synchronization, and device handles are adapter
policy. They do not belong in `.ax` or in the scheduler core. A kore service may
implement them with its own allocator and event primitives; a host adapter may
use its runtime's facilities. Cache hits must compare the exact graph bytes or
otherwise use a collision-safe key before executing cached device code.
`AxHetero::release_executable` is the provider-lifetime counterpart to adapter
eviction: consuming an entry releases its resident programs and fixed I/O,
while provider-rejected handles are quarantined rather than made reusable.

An optimized or unusual kernel does not need a UI-specific dispatch escape
hatch. A semantics-preserving provider fusion can remain the provider's warm
implementation. If several implementations have materially different cost or
fidelity, compile them as explicit executable variants and give their measured
durations and fidelity to `AdmissionPlanner`; automatic dispatch then makes the
same deadline/capacity decision used for CPU and accelerator alternatives.

The `axnn-vaccel` Planet + Moon example demonstrates the host form of the first
two layers: each concrete batch signature owns one `AxExecutable<Global>`, while
provider programs, I/O buffers, schedules, and precision-expanded constants are
retained across frames. `Global` is explicit example policy; the executable and
dispatch APIs continue to accept caller-selected persistent and transient
allocators.

Geometry introduces a distinct result-cache layer above compiled programs.
For a static camera and metric, an `axgeom` ray trace can cache the completed
ray-to-event map (escape direction, disk hit, capture, and redshift inputs)
while procedural sky or disk shading continues to animate. Its key must cover
camera/tetrad state, metric and disk parameters, ray sampling, integrator
settings, graph bytes, concrete tile shape, and backend generation. This cache
belongs to the scene/runtime adapter: neither the `.ax` tensor language nor the
geometry kernel assumes storage lifetime, eviction, clocks, or `Global`.
