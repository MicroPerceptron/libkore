# axshader

`axshader` is the host/build-time shader compiler for Axiom. It parses and
validates WGSL with standalone [Naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga),
checks the deliberately constrained `ax-shader-v0` profile, and emits a
canonical allocator-owned `.ax` graph. It does **not** depend on `wgpu` or any
platform graphics API.

Naga currently uses ordinary global allocation internally. That is an explicit
asset-compiler boundary: parsing, diagnostics, and temporary lowering state run
on the host, while every byte that crosses the boundary in `GraphBuf<A>` uses
the caller-supplied `A: Allocator`. A `kore` image should normally ship the
compiled `.ax` asset rather than Naga.

## `ax-shader-v0` ABI

The default output layout accepts one fragment entry point with this shape:

```wgsl
@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // pure expression graph; &&/|| become eager semantic logical nodes
}
```

Every emitted graph begins with two F32 input planes, in order:

1. `position.x`
2. `position.y`

To avoid materializing two full coordinate fields, `position.x` has shape
`[1, width]` and `position.y` has shape `[height, 1]`; normal tensor broadcasting
produces the full viewport. `position.z` is defined as `0` and `position.w` as
`1` by this 2D profile. Four F32 output planes follow in RGBA order with shape
`[height, width]`. Callers must use
`CompileOptions::for_viewport("main", width, height)` so the emitted graph is
directly executable by current providers. A future graph-specialization API may
allow one unresolved asset to be rebound to multiple viewport sizes; the v0
compiler does not claim that facility yet.

Optional inputs follow those axes:

- Up to 16 F32 field lanes from scalar/vector entry arguments at consecutive
  `@location(0)`, `@location(1)`, etc. Each component has `[height, width]`
  shape. These are tensor fields, **not** GPU-interpolated vertex varyings.
- Up to 32 F32 uniform lanes from bound, read-only scalar/vector globals,
  sorted by `(group, binding, component)`, each with `[1, 1]` shape. The ABI
  passes independent scalar tensors, not packed WGSL uniform-buffer bytes.

`ShaderInterface` reports both lane counts. Input order is stable even if
declaration/use order changes; unused declared uniforms still occupy slots.
Aggregate inputs, arrays and dynamic indexing remain unsupported. Ordinary
result structs have a separate bounded interface described below. For example:

```wgsl
@group(0) @binding(0) var<uniform> viewport: vec2<f32>;
@group(0) @binding(1) var<uniform> pointer: vec2<f32>;
@fragment
fn main(@builtin(position) p: vec4<f32>,
        @location(0) coverage: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(p.xy / viewport + pointer, coverage.x, coverage.y);
}
```

The production `axboot` asset uses this ABI to retain one 128×32 graph across
all physical display sizes and pointer/clock changes. Tile coordinates remain
global physical pixel centers; the viewport itself is supplied as uniforms.

Reachable return expressions may contain:

- F32, I32 and bool literals, constants, scalar/vector constructors and splats;
- constant vector indexing, swizzles, and immutable `let` expressions;
- pure helper calls with F32/I32/bool scalar or vector parameters and results;
- unary `-` and `!`, plus I32 bitwise `~`;
- F32/I32 `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, and `>=`;
- I32 `&`, `|`, `^`, `<<` and arithmetic `>>`; shift counts must be constant
  U32 scalars/vectors in `0..32`, with matching vector width;
- explicit numeric F32/I32 scalar/vector conversions;
- bool `&&` and `||`, plus branchless `select` with F32/I32 results. Naga lowers
  WGSL's short-circuit operators through anonymous bool temporaries and
  structured branches; the profile recognizes exactly that pure
  compiler-generated pattern and emits semantic `LOGICAL_AND`/`LOGICAL_OR`
  nodes;
- F32 `abs`, `min`, `max`, `clamp`, `saturate`, `cos`, `sin`, `tan`, `tanh`,
  `acos`, `asin`, `atan`, `atan2`, `ceil`, `floor`, `round`, `trunc`, `fract`,
  `exp`, `exp2`, `log`, `log2`, `pow`, `sign`, `fma`, `mix`, `step`,
  `smoothstep`, `sqrt`, `inverseSqrt`, `radians`, and `degrees`;
- `dot`, `length`, `distance`, and `normalize`, expanded into scalar/lane `.ax`
  operations so the normal optimizer and heterogeneous scheduler see them.

Scalar/vector operations are lane-wise, and scalar operands broadcast across
vectors where WGSL permits it. The semantic `.ax` operators retain WGSL's
requirements: notably, `round` is ties-to-even and F32 `%` is floating remainder
with a truncating quotient. Providers must not substitute host-language
rounding or modulo semantics. The canonical `1.0 / x` form lowers directly to
`RECIPROCAL`, while `inverseSqrt(x)` lowers to `RSQRT`.

I32 addition, subtraction and multiplication wrap at 32 bits; integer division
truncates toward zero. Runtime division by zero and `i32::MIN / -1` return the
left operand; the corresponding remainder cases return zero. Right shifts
preserve the sign. Naga still rejects invalid constant expressions, including
constant zero divisors and shift counts outside `0..32`. U32 runtime values,
unsigned arithmetic, dynamic shift counts and bitcasts remain unsupported;
constant U32 shift counts become exact I32 count tensors, not a U32 dtype alias.

F32-to-I32 conversion emits an explicit clamp to `[-2147483648, 2147483520]`
before canonical `CAST`, retaining WGSL's representable F32 endpoint bound.
I32-to-F32 conversion is explicit and may round values above 2²⁴; IDs remain
exact until the shader asks for that conversion. NaN conversion is not a
portable numeric value guarantee. No implicit conversion of IDs or byte
loads through F32 is used. The field and uniform input ABI remains F32-only.

## Typed fragment result structs

The ordinary `compile_wgsl_in` API also accepts user-named fragment result
structs. Each member must be F32 scalar/vector or scalar I32, with unique,
consecutive `@location` values starting at zero:

```wgsl
struct Sample {
    @location(1) @interpolate(flat) code: i32,
    @location(0) value: vec3<f32>,
}

@fragment
fn main(@builtin(position) p: vec4<f32>) -> Sample {
    let result = Sample(16777217i, vec3<f32>(p.xy, 0.5));
    let shared_result = result;
    return shared_result;
}
```

Outputs flatten in ascending location order, then x/y/z/w component order,
independently of member declaration order. A vector occupies one WGSL location
and emits one tensor plane per component. This example therefore emits
`[F32, F32, F32, I32]`, each with `[height, width]` shape. At most 16 flattened
planes are admitted. `ShaderOutputKind::FragmentStruct` and
`ShaderInterface::output_lane(index)` expose each plane's location, component
and dtype; `output_count()` and `output_dtype(index)` work for every layout.
Existing `RgbaF32` and `VoxelHit` layouts retain their original plane order.
Runtime execution only needs the typed output tensors serialized in `.ax`.

A result must be a direct constructor or an immutable `let` alias of one.
Numeric members can use existing pure helpers and expressions. This does not
admit general struct operations: nested records, arrays, matrices, struct
helper parameters/results, member loads from ordinary records, record
selection/conditional returns, and authored mutation are rejected. Builtin,
dual-source, per-primitive, and custom interpolation/sampling fields are also
rejected. These outputs are tensor planes, not GPU attachment or interpolation
operations; ordinary default qualifiers and `@interpolate(flat)` are accepted.

I32 members stay exact, including values above 2²⁴. Literal/global-constant I32
members are expanded into exact constant output planes, with a 64 MiB aggregate
payload bound checked before expansion. Full-field values such as queried voxel IDs use typed
identity without this expansion. Computed scalar and axis I32 values broadcast
through typed selection with identical arms and a full-field mask; only graph
nodes are emitted, without large constant payloads or conversion through F32.

## Pure helpers and structured branches

Helpers are inlined into the same graph without changing the input ABI. Each
invocation has its own argument/expression cache; uniforms and constant tensors
remain shared. Nested calls, lexical blocks, and structured `if`/`else` with
early value returns are supported:

```wgsl
fn shade(albedo: vec3<f32>, irradiance: vec3<f32>, emissive: bool) -> vec3<f32> {
    if emissive {
        return vec3<f32>(2.5, 1.5, 0.5);
    }
    return albedo * (vec3<f32>(0.24, 0.29, 0.34) + irradiance);
}
```

Every path must return a value or continue to a subsequent value return.
Branches become eager value selection over the admitted pure arithmetic;
boolean helper returns use logical mask operations. Continuation returns are
shared rather than copied into each branch. As with branchless `select`, both
sides may execute: WGSL builtin domain and accuracy requirements still apply.
This does not authorize speculative resource access or memory effects.

Authored mutable locals, loops, recursion, void helpers, non-uniform globals,
textures, derivatives, general pointers, atomics, barriers, discard,
matrices, U32 runtime values and integer math builtins remain unsupported. I32
values have the explicit operations described above. The only
local stores admitted are Naga's exact compiler-generated `&&`/`||` temporaries.
All called helper bodies are checked for unsupported effects even if their
results are discarded; unused resource declarations are also checked.
Unsupported dead expressions that do not contribute to a returned value may
be removed.

Compilation is bounded to 32 nested helper calls, 1,024 expanded calls and
16,384 emitted nodes. Structured nesting is limited to 64 levels, with a
256-level lowering limit for expression/return nesting. Exceeding these limits
produces a `ProfileRule::Budget` diagnostic rather than unbounded inlining.
These limits apply to compiler expansion, independently of viewport dimensions.

`compile_wgsl_in` is the authored-source entry point. The shared
`lower_naga_module_in` boundary accepts an already parsed Naga module, allowing
future SPIR-V input to reuse exactly the same profile and `.ax` lowerer.

## Explicit voxel-query extension

`compile_wgsl_with_resources_in(source, options, resources, allocator)` adds a
small opt-in profile. A standalone leading `enable ax_voxel_query;` directive
and an explicit resource compiler API are required: either this voxel-only
API or `compile_wgsl_with_resource_specs_in`, described below. Ordinary
`compile_wgsl_in` continues to reject the extension; arbitrary ray-marching
WGSL is not automatically recognized or rewritten into voxel kernels.

The resource slice contains `ax::voxel::VoxelBrickTraceParams` values with a
`ResourceKey { group, binding }` and fixed voxel `dims: [x, y, z]`. Dimensions
must be positive multiples of four, with a cell count that fits I32. Each
storage declaration must have one matching specialization. The only admitted
declaration is `var<storage, read> occupancy: array<vec2<u32>>`, which names a
borrowed array of 4×4×4 occupancy words. The two U32 components denote the low
and high halves of each U64 word; the CPU runtime binds a typed `&[u64]`, not
an unaligned WGSL byte buffer. Occupancy is neither an input tensor nor an
asset constant, so edits can use the same compiled graph.

For example, this source authors the camera rays as well as the query:

```wgsl
enable ax_voxel_query;
@group(1) @binding(0)
var<storage, read> occupancy: array<vec2<u32>>;

@group(0) @binding(0) var<uniform> viewport: vec2<f32>;
@group(0) @binding(1) var<uniform> camera_origin: vec3<f32>;
@group(0) @binding(2) var<uniform> forward: vec3<f32>;
@group(0) @binding(3) var<uniform> right: vec3<f32>;
@group(0) @binding(4) var<uniform> up: vec3<f32>;
@group(0) @binding(5) var<uniform> grid_scale: f32;
@group(0) @binding(6) var<uniform> ray_limit: f32;

fn unit(v: vec3<f32>) -> vec3<f32> {
    return v * (1.0 / max(sqrt(dot(v, v)), 1e-8));
}

struct RayResult {
    @location(0) distance: f32,
    @location(1) @interpolate(flat) cell_id: i32,
    @location(2) direction: vec3<f32>,
}

@fragment
fn main(@builtin(position) p: vec4<f32>,
        @location(0) pixel: vec2<f32>) -> RayResult {
    let sx = (pixel.x - viewport.x * 0.5) / viewport.y;
    let sy = (viewport.y * 0.5 - pixel.y) / viewport.y;
    let direction = unit(forward + (right * sx + up * sy));
    let hit = ax_brick_trace(&occupancy, camera_origin * grid_scale,
                            direction * grid_scale, ray_limit);
    return RayResult(hit.t, hit.cell_id, direction);
}
```

The existing position axes still occupy the first two input slots. This
example adds two full field lanes for actual screen pixel centers, allowing
one tile to cross a scanline, followed by 16 uniform scalar lanes in binding
order. Scaling both origin and direction preserves the ray parameter as a
world distance when the unscaled direction is unit length. The query itself
does not normalize its direction.
The ordinary result struct returns distance, exact cell ID and the unscaled
WGSL direction as five planes `[F32, I32, F32, F32, F32]`. Consumers can reuse
that direction without regenerating camera rays on the host. The query and
the expressions constructing its ray are shared with the result fields.

The compiler supplies the reserved `AxVoxelHit` type. Its members are `t:
f32` and `cell_id: i32`; a direct entry return emits two `[height, width]`
planes in that order. `ShaderInterface::output_count()` and
`output_dtype(index)` describe the layout. IDs are exact x-major cell indices
`x + dims[0] * (y + dims[1] * z)`, with `-1` for a miss; miss distance is
positive infinity. No cell ID passes through F32, including IDs above 2²⁴.
Alternatively, a shader can read `hit.t`, compute with `hit.cell_id` using the
admitted I32 operations, and return F32 RGBA or a bounded ordinary result struct.

Calls require the direct address of a root occupancy global, F32 origin and
direction vectors, and one uniform scalar F32 limit. Query calls must be
unconditional in the selected entry body. Pure numeric helpers can construct
rays; queries inside helpers, branches, short-circuit operands, or after a
possible early return are rejected. Pointer aliases, storage loads/indexing,
hit construction/selection and general resource operations remain unsupported.
A hit may be returned directly or
accessed by its two named members; those values can also be collected in an
ordinary result constructor. The reserved `AxVoxelHit` itself cannot be
constructed, selected, or used as a general mutable record.

Naga validates all ordinary ray arguments. A private normalization registers
the exact intrinsic and binds its resource before removing the storage-pointer
parameter that Naga's validator does not support. The generated declaration
is parser scaffolding only: its body cannot execute or be inlined, authored
name collisions are rejected, and the ordinary Naga entry point cannot
register it. Original source byte offsets are retained for diagnostics.
Lowering emits two `INTERLEAVE3_F32` nodes and one `VOXEL_BRICK_TRACE` node
with explicit resource metadata and paired F32/I32 results. The query has a
native CPU implementation; this does not claim NPU execution of traversal.

## Explicit RGB-volume sampling extension

`compile_wgsl_with_resource_specs_in(source, options, resources, allocator)`
accepts `ShaderResource::VoxelBrickGrid(VoxelBrickTraceParams)` and
`ShaderResource::RgbVolume(VolumeRgbParams)` specializations. RGB sampling
requires a standalone leading `enable ax_volume_sampling;` directive. Both
resource directives may appear in the same source; the existing voxel-only
API remains compatible and does not enable volume sampling.

The compiler supplies an opaque `AxRgbVolume` handle and the intrinsic
`ax_sample_volume_rgb(&volume, point) -> vec3<f32>`:

```wgsl
enable ax_volume_sampling;
@group(1) @binding(1)
var<storage, read> irradiance: AxRgbVolume;

@fragment
fn main(@builtin(position) p: vec4<f32>,
        @location(0) point: vec3<f32>) -> @location(0) vec4<f32> {
    let light = ax_sample_volume_rgb(&irradiance, point);
    return vec4<f32>(light, 1.0);
}
```

For example, the matching specialization can select UF16 storage:

```rust
use std::alloc::Global;
use ax::sampling::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};
use axshader::{CompileOptions, ShaderResource, compile_wgsl_with_resource_specs_in};

let resources = [ShaderResource::RgbVolume(VolumeRgbParams {
    resource: ResourceKey { group: 1, binding: 1 },
    dims: [24, 16, 24],
    storage: VolumeRgbStorage::Uf16E5M11,
})];
let compiled = compile_wgsl_with_resource_specs_in(
    source,
    CompileOptions::for_viewport("main", 128, 1),
    &resources,
    Global,
)?;
```

The resource is an x-major, interleaved RGB point grid. `F32` storage uses
12 bytes per point; `Uf16E5M11` uses 6, with unsigned E5M11 values promoted to
F32 before interpolation. This is storage conversion, not a tensor dtype,
IEEE F16/BF16 storage, or native UF16 arithmetic. `AxRgbVolume` deliberately
does not expose an `array<vec3<f32>>` layout. Its private parser scaffold is
not a host buffer layout, and its generated intrinsic body cannot execute.
The runtime validates exact storage length and finite, nonnegative energy;
F32 negative zero is allowed. Invalid UF16 encodings are rejected, not
saturated. Dimensions must be positive and at most 16,777,217 per axis, with
checked addressable storage size.

Coordinates are F32 **grid indices**, not normalized texture coordinates:
integer coordinates select grid points, without a half-texel offset. Each
axis clamps to `[0, dimension - 1]`; infinities clamp to edges and any NaN
coordinate produces three NaNs. Singleton axes use their sole point. The
operation emits three full `[height, width]` F32 planes even when coordinates
are scalar, and accepts existing scalar, axis, uniform and field broadcasts.
Each channel accumulates all eight corners from positive zero in z/y/x
order, with x fastest. Weights use `(wx * wy) * wz`, followed by separate F32
multiply and add operations; nested lerps or fused contraction would change
that contract.

Every storage declaration needs one specialization with the same binding and
resource kind. Duplicate keys, wrong kinds, invalid dimensions and specs
without declarations are rejected. A declared resource may be unused; this
lets shader variants keep a stable declared interface. Resource contents are
borrowed at execution, never serialized as constants or passed as ordinary
tensor inputs. A sampling node serializes its resource key, dimensions and
storage format in 24-byte attributes in a canonical `.ax` 1.1 asset. Updating
the bound contents does not require recompilation; changing dimensions or
storage format does.

The same resource-call restrictions apply to every extension: calls must be
unconditional in the selected fragment body and take the direct address of
the matching root storage global. Helpers, branches, short-circuit operands
and calls after a possible early return cannot contain resource operations.
Pure numeric helpers may construct sampling coordinates or consume the F32
result. Queries can feed hit distances into sampling coordinates, and sampled
RGB can feed ordinary shading or bounded fragment result structs. Authored
handle declarations, type aliases, constructors, pointer aliases, member
access, loads, indexing and mutation are rejected even when unused. Ordinary
WGSL compilation also rejects the reserved types and intrinsic names.
Naga still validates ordinary arguments after the resource pointer is bound;
normalization preserves authored source offsets.

The graph records `VOLUME_TRILINEAR_RGB` as a resource-reading operation with
F32 computation and results. Its native CPU implementation does not imply
provider or NPU support for sampling, or for the chosen resource storage
format.

## Explicit byte-buffer extension

`compile_wgsl_with_resource_specs_in` also accepts
`ShaderResource::U8Buffer(ax::buffer::U8BufferParams { resource, elements })`.
The resource is a tightly packed immutable byte slice with a checked length
from zero through `i32::MAX`. No scan of byte contents is necessary: all 256
values are valid. A standalone leading `enable ax_buffer_access;` directive
registers one bounded load:

```wgsl
enable ax_buffer_access;
@group(1) @binding(2) var<storage, read> materials: AxU8Buffer;

@fragment
fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let material = ax_load_u8(&materials, i32(p.x));
    return vec4<f32>(select(vec3<f32>(0.5), vec3<f32>(0.61, 0.235, 0.13), material == 4i), 1.0);
}
```

`ax_load_u8(&buffer, index: i32)` widens the selected byte exactly to I32.
Negative indices and indices at or beyond `elements` produce zero, including
every index of an empty buffer. Scalar, axis and full-field indices are
admitted; the result is always an I32 `[height, width]` plane. Callers that
flatten multidimensional coordinates must check each coordinate first:
an invalid x must not accidentally address a neighboring row.

The graph emits `BUFFER_LOAD_U8` with `READS_RESOURCE` and 16-byte attributes
containing the binding and element count. Contents are borrowed per invocation,
never serialized into the graph or added to its ordinary tensor input ABI.
The opaque `AxU8Buffer` scaffold does not assert WGSL storage alignment or
expose an array layout. All three resource directives may coexist; voxel IDs
can index materials, whose integer values can feed pure palette helpers and
RGB sampling coordinates. Resource calls themselves remain unconditional in
the selected fragment and must use the direct root address. General loads,
indexing, aliases, helper/conditional resource calls and mutation remain
rejected, even when unused.
