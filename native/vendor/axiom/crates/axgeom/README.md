# axgeom

`axgeom` is Axiom's backend-neutral geometry and multidimensional field-math
substrate. It owns reusable equations and geometric data semantics, but not
the runtime that schedules providers, caches work, or presents a scene.

The default `no_std` core depends only on `axmath` and does not allocate. The
optional `graph` feature adds allocator-owned `.ax` emission; every graph
builder receives a caller-provided `core::alloc::Allocator`. There is no
implicit `Global`, provider, scheduler policy, cache, window, clock, thread,
environment, or filesystem dependency.

The production surface is organized around reusable kernels:

- dimension-generic vector, matrix, and nonlinear scalar algebra;
- continuous fields and sampled multidimensional grids;
- finite-difference gradients, Jacobians, divergence, curl, Laplacians, and
  Hessians with explicit boundary and halo semantics;
- explicit and symplectic integration primitives;
- checked field layouts and tile decomposition;
- geometric error and conditioning estimates used to form dynamic-precision
  batches;
- projection geometry for Gaussian splats, raster primitives, voxels, and
  rays.

CPU-specific AVX2 implementations remain in `axnn-cpu`. They may consume the
core definitions and reference kernels without introducing an `ax` or
accelerator dependency. Provider-specific compilation and placement remain in
the runtime crates.

Issue #33's Dneg wormhole and Kerr geodesics are intentionally demanding
clients of these kernels. They validate nonlinear field evaluation,
differentiation, integration, event handling, graph lowering, and heterogeneous
placement; they do not define the crate's abstraction boundary.

The wormhole `.ax` builder emits a fixed SSA transition graph. Its stable
planes are `ell`, `phi`, `p_ell`, conserved impact parameter, and an active
mask. A host can compile one specialization per ray-tile size and number of
steps, then reuse it for every tile or feed its outputs back for another step
block. This keeps graph placement backend-agnostic while allowing the host to
choose latency/compile-time tradeoffs.

Work quality has separate controls:

- field/ray/image sampling density;
- integrator step size and maximum step count;
- fixed steps per compiled transition graph;
- tile dimensions, halo width, and boundary policy;
- projected error budgets and precision class.

Dynamic precision is a tile or compacted-batch policy because a graph tensor
has one dtype. A cheap geometric estimator classifies sensitive, near, and far
work; `compact_precision_indices` writes stable per-class index lists into
caller storage, and `AxProjectionPrecisionPolicy::batches` pairs them with the
F32, FP16, or FP8-to-FP16 specialization used by the existing mixed-precision
graph and CPU kernels. Geometry records stay in place and there is no hot-path
allocation. Camera
subtraction happens before narrowing, FP8 is storage that widens once at graph
ingress, nonlinear work uses FP16/BF16 where supported, and sensitive
determinants, reductions, event decisions, and accumulation stay in F32.

Intermediate geometry—such as a ray-to-event map or projected splat
footprint—can be cached independently of shading whenever its geometric inputs
are unchanged. Raster dimensions therefore remain distinct from integration,
field, or projection quality.
