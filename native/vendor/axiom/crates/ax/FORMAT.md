# `.ax` binary format 1.0 – 1.8

An `.ax` artifact is a canonical, little-endian tensor graph. It contains no
paths, pointers, host handles, provider names, queues, or device identities.
The complete artifact must be present in one byte slice.

Ordinary graphs retain their exact 1.0 encoding. Graphs containing opcodes
`51` through `59` or the resource-read flag below require minor version 1.1.
This includes the new attribute-free integer bit operations, not only resource
queries. Raster opcodes `60` through `65` and the resource-write flag require
1.2, neural opcodes `66` through `69` require 1.3, and the key/value cache
opcodes `70` and `71` require 1.4. The decoder operations `72` through `75`
and the axis-aware `RMS_NORM` record require 1.5, the recurrent
operations `76` and `77` require 1.6, and expert dispatch `78` requires
1.7. A fused-epilogue record on `MATMUL_TRANSPOSED`, and a
`CAUSAL_CONV1D` record carrying an activation, require 1.8. Sections for 1.2 and 1.3 are not yet
written here; `ax::raster` and `docs/ax-streaming-inference.md` describe them.
The section layout and the assignments of legacy opcodes are unchanged.

## Header

The header is exactly 64 bytes:

| Offset |      Type | Field                    |
| -----: | --------: | ------------------------ |
|      0 | `[u8; 4]` | magic `AXIR`             |
|      4 |     `u16` | major version (`1`)      |
|      6 |     `u16` | minor version (`0` or `1`) |
|      8 |     `u16` | header size (`64`)       |
|     10 |     `u16` | reserved, zero           |
|     12 |     `u32` | reserved, zero           |
|     16 |     `u32` | tensor count             |
|     20 |     `u32` | node count               |
|     24 |     `u32` | tensor-reference count   |
|     28 |     `u32` | dimension count          |
|     32 |     `u32` | attribute byte count     |
|     36 |     `u32` | constant byte count      |
|     40 |     `u32` | tensor-table offset      |
|     44 |     `u32` | node-table offset        |
|     48 |     `u32` | reference-table offset   |
|     52 |     `u32` | dimension-table offset   |
|     56 |     `u32` | attribute-section offset |
|     60 |     `u32` | constant-section offset  |

Sections are packed in that order with no gaps, overlaps, reordering, or
trailing bytes. Record subranges must likewise be monotonically packed. This
makes the byte representation deterministic and safe to compare for caching.

## Tensor record

Each tensor record is 16 bytes:

| Offset |  Type | Field                                |
| -----: | ----: | ------------------------------------ |
|      0 |  `u8` | `axnn::DType` tag                    |
|      1 |  `u8` | rank, at most `axnn::MAX_RANK`       |
|      2 | `u16` | flags: input=1, output=2, constant=4 |
|      4 | `u32` | first dimension index                |
|      8 | `u32` | first constant byte                  |
|     12 | `u32` | constant byte length                 |

Dimensions are `u32`. A frontend may use `ax::DYNAMIC_DIM` (`u32::MAX`) for a runtime-bound
dimension; its cost model and backend adapter are then responsible for the
concrete specialization. Constant bytes are stored exactly as the tensor dtype
requires. A nonempty constant range requires the constant flag.

## Node record

Each node record is 24 bytes:

| Offset |  Type | Field                        |
| -----: | ----: | ---------------------------- |
|      0 | `u16` | operation code               |
|      2 | `u16` | flags: ordered=1, mutating=2, reads-resource=4, writes-resource=8 |
|      4 | `u32` | first input reference        |
|      8 | `u16` | input count                  |
|     10 | `u16` | output count                 |
|     12 | `u32` | first output reference       |
|     16 | `u32` | first attribute byte         |
|     20 | `u32` | attribute byte length        |

References are `u32` tensor IDs. Inputs immediately precede outputs for each
node. The analyzer requires single-assignment tensors, rejects missing or
multiple producers, and adds serialized edges between ordered/mutating nodes.
Resource reads do not require serialization against other reads of the same
immutable snapshot. They are not memoizable from tensor input versions alone:
resource contents may change between executions.

Version 1 defines the built-in operation identifiers in `ax::OpCode`. Numeric
identifiers `0` through `15` retain their original assignments:

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 0 | `IDENTITY` | `1 -> 1` |
| 1 | `CAST` | `1 -> 1` |
| 2 | `ADD` | `2 -> 1` |
| 3 | `SUB` | `2 -> 1` |
| 4 | `MUL` | `2 -> 1` |
| 5 | `DIV` | `2 -> 1` |
| 6 | `MATMUL` | `2 -> 1` |
| 7 | `CONV1D` | schema-dependent |
| 8 | `REDUCE_SUM` | schema-dependent |
| 9 | `REDUCE_MAX` | schema-dependent |
| 10 | `TANH` | `1 -> 1` |
| 11 | `SIGMOID` | `1 -> 1` |
| 12 | `RELU` | `1 -> 1` |
| 13 | `RESHAPE` | `2 -> 1` (`data`, `shape`) |
| 14 | `TRANSPOSE` | schema-dependent |
| 15 | `COPY` | `1 -> 1` |

The backend-neutral, attribute-free operation set is appended without changing
the container encoding or the format version:

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 16 | `POW` | `2 -> 1` |
| 17 | `MAXIMUM` | `2 -> 1` |
| 18 | `MINIMUM` | `2 -> 1` |
| 19 | `ABS` | `1 -> 1` |
| 20 | `COS` | `1 -> 1` |
| 21 | `EXP` | `1 -> 1` |
| 22 | `LOG` | `1 -> 1` |
| 23 | `LOGICAL_NOT` | `1 -> 1` |
| 24 | `NEGATE` | `1 -> 1` |
| 25 | `RECIPROCAL` | `1 -> 1` |
| 26 | `RSQRT` | `1 -> 1` |
| 27 | `SIN` | `1 -> 1` |
| 28 | `SELECT` | `3 -> 1` (`condition`, `then`, `else`) |
| 29 | `EQUAL` | `2 -> 1` |
| 30 | `GREATER` | `2 -> 1` |
| 31 | `GREATER_EQUAL` | `2 -> 1` |
| 32 | `LOGICAL_AND` | `2 -> 1` |
| 33 | `LOGICAL_OR` | `2 -> 1` |
| 34 | `LOGICAL_XOR` | `2 -> 1` |
| 35 | `SQRT` | `1 -> 1` |
| 36 | `FLOOR` | `1 -> 1` |
| 37 | `CEIL` | `1 -> 1` |
| 38 | `ROUND` | `1 -> 1` |
| 39 | `TRUNC` | `1 -> 1` |
| 40 | `REMAINDER` | `2 -> 1` (`lhs`, `rhs`) |
| 41 | `ATAN` | `1 -> 1` |
| 42 | `ATAN2` | `2 -> 1` (`y`, `x`) |
| 43 | `ACOS` | `1 -> 1` |
| 44 | `ASIN` | `1 -> 1` |
| 45 | `TAN` | `1 -> 1` |
| 46 | `EXP2` | `1 -> 1` |
| 47 | `LOG2` | `1 -> 1` |
| 48 | `SIGN` | `1 -> 1` |
| 49 | `CLAMP` | `3 -> 1` (`value`, `low`, `high`) |
| 50 | `FMA` | `3 -> 1` (`a`, `b`, `c`) |

Codes `16` through `50` have no attribute schema; conforming frontends emit an
empty attribute slice for them. Attribute schemas for reductions, convolution,
and transpose must be standardized before frontends emit those operations;
adapters must reject unsupported schemas, not guess them.

The appended codes name elementwise semantics, not individual kernels.
`ROUND` rounds halfway cases to the nearest even integer. For floating operands,
`REMAINDER(lhs, rhs)` is `lhs - rhs * TRUNC(lhs / rhs)`; I32 semantics are
specified below. `FMA(a, b, c)` computes `a * b + c`; a
backend may contract it to a fused instruction or use separate multiply and add
rounding, matching the WGSL builtin's permitted implementations. Composite
WGSL operations remain frontend rewrites where that preserves intent: for
example, `fract(x)` is `SUB(x, FLOOR(x))`, and `saturate(x)` is a `CLAMP` with
zero and one bounds. Backend implementation IDs select tuned or fused kernels
without allocating additional operation codes.

## Typed voxel operations (1.1)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 51 | `INTERLEAVE3_F32` | `3 -> 1` (`x`, `y`, `z`) |
| 52 | `VOXEL_BRICK_TRACE` | `3 -> 2` (`origins`, `directions`, `limit` -> `t`, `cell_id`) |

`INTERLEAVE3_F32` has no attributes. Its authored result must be F32
`[H, W, 3]`; each input must be F32 and broadcast to `[H, W]`. An explicit
result extent allows three uniform camera coordinates to expand into a full
ray tile. The last axis stores x, y, and z in that order.

`VOXEL_BRICK_TRACE` consumes matching F32 `[H, W, 3]` origins and directions
plus one F32 limit element (any checked one-element shape). Its result ABI is
exactly two separate `[H, W]` tensors: F32 distance `t`, then I32 `cell_id`.
Neither result may alias an operand or the other result. Valid IDs are the
x-major voxel index `x + nx * (y + ny * z)`; a miss is `(+infinity, -1)`.
IDs never pass through F32, including indices above 2^24.

Nonfinite origins or directions and NaN/negative limits produce misses;
positive infinity is a valid limit. Finite rays preserve the scalar brick
traversal's clipping, rounding, and tie behavior. Direction components with
magnitude below `1e-12` are clamped to signed `1e-12`, preserving the sign of
zero. An all-zero direction is therefore not categorically a miss: an origin
inside an occupied cell can report a hit at `t = 0`.

The query requires `reads-resource` and forbids `mutating`. Its attributes are
exactly 24 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `u8` | schema version: `1` |
| 1 | `u8` | resource kind: `1` = `Brick4x4x4U64` |
| 2 | `u16` | reserved: zero |
| 4 | `u32` | binding group |
| 8 | `u32` | binding number |
| 12 | `u32` | voxel dimension x |
| 16 | `u32` | voxel dimension y |
| 20 | `u32` | voxel dimension z |

All integers are little-endian. Voxel dimensions must be positive multiples
of four and their checked product must be at most `i32::MAX`. The resource
contains one u64 word per 4x4x4 brick, with x-major brick ordering and bit
`x_local + 4 * (y_local + 4 * z_local)` inside each word. The checked brick
count is the voxel count divided by 64. This describes occupancy only; it
does not describe F16 or UF16 density storage.

The graph stores no occupancy bytes or runtime resource pointer. The adapter
borrows a matching typed brick resource for an execution and validates its
binding, dimensions, and length before running the node. Content changes
between runs must be visible without rebuilding the graph or treating the
resource as a tensor input or constant.

`ax::voxel` provides the checked attribute codec and `GraphBuilder` helpers.
The shared semantic bridge validates this specific schema and both result
descriptors before constructing `KernelRequest::voxel_brick_trace`.
`KernelResult::TensorPair` preserves the result order, `result_tensor(index)`
exposes each descriptor, and candidate output dtype/rank/layout admission
checks both results. The legacy primary `output_dtype` and `shape` fields
continue to describe result zero. Resource reads carry
`KernelEffects::READS_RESOURCE`; ordinary attributed operations remain
unsupported unless their own schema has been standardized.

## Typed RGB volume sampling (1.1)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 53 | `VOLUME_TRILINEAR_RGB` | `3 -> 3` (`x`, `y`, `z` -> `r`, `g`, `b`) |

The inputs are F32 volume-index coordinates. Each broadcasts to the explicit
rank-two output field `[H, W]`; three scalar coordinates therefore still
produce three full output planes. Results are distinct F32 tensors in red,
green, blue order, with identical `[H, W]` dimensions. No result may alias
another result or an operand. Zero-sized output fields are allowed. This
operation samples a resource; it does not include camera, lighting, shading,
or world-to-volume coordinate transformations.

Each coordinate is clamped to `[0, dimension - 1]`. Negative and positive
infinity clamp to the respective edge. If any coordinate is NaN, all three
results at that sample are NaN, with unspecified payload. Otherwise let `q`
be the clamped coordinate, `lo = floor(q)`, `hi = min(lo + 1, dimension - 1)`,
and `fraction = q - F32(lo)`. A dimension of one selects index zero with a
zero fraction. For each channel, start at positive F32 zero and visit all
eight corners in z/y/x order, x changing fastest. Each axis weight is
`1 - fraction` for its lower corner and `fraction` for its upper corner.
Compute `weight = (wx * wy) * wz`, then add `F32(value) * weight` to the
accumulator using separate F32 operations. Implementations must not replace
this order with nested lerps, reassociate or fuse operations, or skip
zero-weight corners. Storage promotion precedes interpolation.

The operation requires `reads-resource` and forbids `mutating`. Attributes
are exactly 24 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `u8` | schema version: `1` |
| 1 | `u8` | resource kind: `2` = x-major interleaved RGB volume |
| 2 | `u8` | storage: `0` = F32, `1` = unsigned E5M11 |
| 3 | `u8` | reserved: zero |
| 4 | `u32` | binding group |
| 8 | `u32` | binding number |
| 12 | `u32` | volume dimension x |
| 16 | `u32` | volume dimension y |
| 20 | `u32` | volume dimension z |

All multibyte integers are little-endian. Dimensions are in
`1..=16_777_217`, so every integer sampling endpoint through `dimension - 1`
is exactly representable in F32. Checked voxel, component, and byte products
must fit `usize`, and the resource byte length must not exceed `isize::MAX`.
The resource has exactly `nx * ny * nz` tightly packed RGB triples in
x-major order: triple index `x + nx * (y + ny * z)`, channel offsets 0, 1,
and 2. F32 triples occupy 12 bytes and unsigned E5M11 triples occupy 6 bytes;
there is no padding, arbitrary stride, extra channel, or implicit alpha.

F32 resource components must be finite and nonnegative; negative zero is
allowed. Unsigned E5M11 has exponent bias 15, eleven fraction bits,
subnormal spacing `2^-25`, and finite raw words `0x0000..=0xf7ff`, with
maximum value 65520. Raw infinity/NaN words `0xf800..=0xffff` are invalid
resource contents. This encoding is neither IEEE F16 nor BF16 and does not
introduce or reinterpret any tensor dtype. Resource content validation must
not silently clamp, quantize, or substitute invalid values.

The graph stores metadata only. A validated immutable borrowed view checks
the exact shape, encoding, length, and content range before execution, so
individual queries do not rescan the whole resource. The view cannot outlive
its storage or overlap mutable access. Prepared plans retain only the
metadata and resolve a fresh borrowed binding on each run; changed contents
must be visible without recompilation, tensor uploads, or a retained F32
mirror of UF16 storage.

`ax::sampling` provides the checked attribute codec and
`GraphBuilder::volume_trilinear_rgb`, returning `RgbSample { r, g, b }`.
The shared bridge constructs `KernelRequest::volume_trilinear_rgb` with an
explicit output `Shape`, checks every authored output, and preserves
`KernelEffects::READS_RESOURCE`. `KernelResult::Tensors(TensorResults)`
provides an allocation-free list of one to four distinct results with private
storage; request/plan `result_count()` and `result_tensor(index)` handle it
alongside the existing `Tensor` and `TensorPair` variants. Checked extents
and candidate dtype/rank/layout constraints apply to every output. The
legacy `output_dtype` and `shape` projections still describe result zero.

## Typed byte-buffer loads (1.1)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 54 | `BUFFER_LOAD_U8` | `1 -> 1` (`index` -> `value`) |

The input is an I32 index tensor that broadcasts to the explicit rank-two I32
output field `[H, W]`. A scalar index therefore still produces a full output
plane. Zero-sized output fields are allowed; checked element and byte counts
must fit `usize`, with output byte length at most `isize::MAX`. The result may
not alias the index tensor.

For `0 <= index < elements`, the selected unsigned byte is zero-extended to
I32, exactly preserving every value from 0 through 255. Negative indices and
indices at or beyond `elements` return zero. An empty resource is valid and
returns zero for every index. Neither indices nor loaded values pass through
F32. This is a linear load: callers flattening multidimensional coordinates
must check each coordinate before flattening, or an invalid coordinate can
produce an otherwise valid linear index.

The operation requires `reads-resource` and forbids `mutating`. Its attributes
are exactly 16 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `u8` | schema version: `1` |
| 1 | `u8` | resource kind: `3` = tightly packed unsigned bytes |
| 2 | `u16` | reserved: zero |
| 4 | `u32` | binding group |
| 8 | `u32` | binding number |
| 12 | `u32` | element count, also the exact byte length |

Multibyte integers are little-endian. The element count must be at most
`i32::MAX` and `isize::MAX`; zero is admitted. There is no stride, padding,
signed-byte interpretation, or floating storage encoding. All byte payloads
are valid, so resource validation does not need a content scan.

Only metadata is serialized. The native CPU adapter resolves a matching
immutable borrowed `&[u8]` on each execution and validates binding identity,
resource kind, and exact length before any node executes. The borrowed view
cannot outlive its storage or overlap mutable access. Prepared native plans
retain metadata, not the borrowed pointer or a copied resource; changed
contents must be observed on a later run without recompilation or adding the
resource to the graph's ordinary tensor input ABI.

`ax::buffer::U8BufferParams { resource, elements }` and the checked
`encode_u8_buffer_attributes` / `decode_u8_buffer_attributes` codec describe
this schema. `GraphBuilder::buffer_load_u8(index, params, [H, W])` emits the
node. The shared bridge validates its input, output, attributes, and flags
before constructing `KernelRequest::buffer_load_u8` with an explicit output
`Shape` and `KernelEffects::READS_RESOURCE`.

## Integer word operations and casts

The following new opcodes require version 1.1, have no attributes, and must
not carry `reads-resource`:

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 55 | `BITWISE_AND` | `2 -> 1` (`lhs`, `rhs`) |
| 56 | `BITWISE_OR` | `2 -> 1` (`lhs`, `rhs`) |
| 57 | `BITWISE_XOR` | `2 -> 1` (`lhs`, `rhs`) |
| 58 | `SHIFT_LEFT` | `2 -> 1` (`value`, `count`) |
| 59 | `SHIFT_RIGHT` | `2 -> 1` (`value`, `count`) |

Both operands and the result are I32. Shapes use ordinary elementwise
broadcasting, and the authored output must match that broadcast shape.
Bitwise operations act on the 32-bit representations. Both shifts use only
the low five bits of the I32 count, including negative counts. `SHIFT_LEFT`
discards bits above bit 31; `SHIFT_RIGHT` is arithmetic and fills with the
value's sign bit. For example, shifting by 32 is shifting by zero, and
`SHIFT_RIGHT(-8, 1)` is `-4`. There is no new U32 tensor ABI.

Existing arithmetic opcodes also have exact I32 specializations. Their
numeric assignments and minimum format version remain unchanged. Operands
and results stay I32, without implicit conversion through F32; binary and
ternary operands broadcast normally.

| Operation | I32 semantics |
| --- | --- |
| `ADD`, `SUB`, `MUL` | Wrap modulo `2^32`. |
| `NEGATE`, `ABS` | Wrap on overflow; both leave `i32::MIN` as `i32::MIN`. |
| `SIGN` | Return `-1`, `0`, or `1`. |
| `DIV(lhs, rhs)` | Truncate the quotient toward zero. If `rhs == 0`, return `lhs`; `DIV(i32::MIN, -1)` returns `i32::MIN`. |
| `REMAINDER(lhs, rhs)` | Use the truncating integer quotient; a nonzero remainder has the numerator's sign. If `rhs == 0`, or for `REMAINDER(i32::MIN, -1)`, return zero. |
| `MINIMUM`, `MAXIMUM` | Compare signed I32 values. |
| `CLAMP(value, low, high)` | Compute signed `MINIMUM(MAXIMUM(value, low), high)` in that order. |

`EQUAL`, `GREATER`, and `GREATER_EQUAL` compare I32 operands directly and
produce U8 masks containing zero or one. `SELECT(condition, then, else)`
accepts I32 value operands and a U8 condition: zero selects `else`; any
nonzero byte selects `then`. Selection preserves the complete integer word.

The F32/I32 specialization of existing opcode `CAST` is attribute-free and
preserves shape. The result tensor's dtype selects the destination:

| Conversion | Semantics |
| --- | --- |
| F32 to I32 | Truncate toward zero and saturate to `[-2147483648, 2147483647]`. Negative/positive infinity maps to the respective endpoint; NaN maps to zero. |
| I32 to F32 | Round to the nearest representable F32, with ties to even; values above `2^24` in magnitude may lose low bits. |
| F32 to F32, I32 to I32 | Storage identity, preserving all bits, including F32 NaN payloads and signed zero. |

For example, canonical `CAST` from F32 `2147483648.0` to I32 produces
`2147483647`. The WGSL frontend instead emits an explicit F32 `CLAMP` to
`[-2147483648, 2147483520]` before `CAST`: `2147483520` is the greatest F32
value within the I32 range. This frontend normalization does not change
canonical `CAST`, and does not promise a portable WGSL NaN conversion result.

The shared bridge uses `KernelRequest::integer_binary` for codes 55–59 and
`KernelRequest::cast` for F32/I32 conversions; same-type casts become typed
identity requests. These are canonical semantics, not a claim that every
frontend accepts every operation or every provider implements every dtype
pair. In particular, the current WGSL profile only admits constant U32 shift
counts in `0..32`, although canonical I32 shift counts may be runtime tensors.

## Attention key/value cache (1.4)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 70 | `KV_APPEND` | `3 -> 1` (`position`, `keys`, `values` -> `token`) |
| 71 | `ATTENTION_DECODE` | `3 -> 1` (`token`, `position`, `queries` -> `output`) |

A key/value cache is one attention layer's bound resource. The graph stores
its binding and geometry only, never its contents, and the cache stores no
length. `position` is a rank-0 I32 input naming the first slot this
invocation writes. The caller that owns decode state advances it only after
an invocation completes. A failed or repeated step therefore rewrites the
same slots, and the cache needs no commit protocol of its own.

`KV_APPEND` writes F32 `keys` and `values`, each `[n, kv_heads, head_dim]`,
into slots `position .. position + n` and returns a one-word I32 `[1]`
token. It never reads cache contents. `ATTENTION_DECODE` consumes that token,
so ordinary data dependency orders the read after the write. It attends F32
`queries` `[n, query_heads, head_dim]`, with query row `i` reading slots
`0 ..= position + i`, and returns F32 `[n, query_heads, head_dim]`. Query
head `h` reads cache head `h / (query_heads / kv_heads)`, and logits are
`logit_scale * dot(query, key)` before the softmax. A node rejects
`position < 0` or `position + n > capacity` before touching the cache; slots
at or beyond `position + n` are never read. Because the cache stores no
length, slots that earlier nodes of a failed invocation wrote are simply
rewritten when the step is repeated.

A read is admitted only directly after its own write: its token must be
produced by a `KV_APPEND` with the identical cache record, from the same
`position` tensor, with as many token rows as the read has query rows.

`KV_APPEND` requires exactly `ordered | writes-resource`, and
`ATTENTION_DECODE` exactly `ordered | reads-resource`; neither may be
`mutating`. `writes-resource` is legal only on raster nodes and `KV_APPEND`.
Because both nodes are ordered, the scheduler keeps them in node order with
every other ordered effect, so a later append to the same cache cannot
overtake an earlier read.

`KV_APPEND` attributes are exactly 28 bytes and `ATTENTION_DECODE`
attributes exactly 36:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `u8` | schema version: `1` |
| 1 | `u8` | record kind: `5` = append, `6` = attention |
| 2 | `u8` | storage: `0` = E4M3 elements with per-token E8M0 scales |
| 3 | `u8` | reserved: zero |
| 4 | `u32` | binding group |
| 8 | `u32` | binding number |
| 12 | `u32` | `kv_heads`, positive |
| 16 | `u32` | `head_dim`, positive |
| 20 | `u32` | `capacity` in token slots: a positive multiple of `page_tokens`, at most `i32::MAX` |
| 24 | `u32` | `page_tokens`, a power of two |
| 28 | `u32` | attention only: `query_heads`, a positive multiple of `kv_heads` |
| 32 | `u32` | attention only: `logit_scale` F32 bits, finite and positive |

The resource is `capacity / page_tokens` packed pages. Each page is laid out
head-major, keys before values, each head's rows contiguous:

```text
keys         [kv_heads][page_tokens][head_dim]  E4M3 bytes
values       [kv_heads][page_tokens][head_dim]  E4M3 bytes
key scales   [kv_heads][page_tokens]            E8M0 bytes
value scales [kv_heads][page_tokens]            E8M0 bytes
```

A page therefore holds `2 * kv_heads * page_tokens * (head_dim + 1)` bytes.
Checked page and resource byte lengths must not exceed `isize::MAX`. A cached
element decodes to `scale * e4m3(byte)` in F32, where the scale belongs to
that element's (token, head) row of keys or of values. Scale byte `e` is the
OCP MX E8M0 value `2^(e - 127)` for `e` in `0..=254` and NaN for `0xff`;
E8M0 has no zero or infinity, and an all-zero row decodes to zeros under any
finite scale. Multiplying by a power of two only moves the element's
exponent, so the product is exact in F32 whenever it is finite, including
F32 subnormals, and every executor decodes a page bit-identically with or
without fused multiply-add. Only scale bytes above 246 can overflow F32;
such products decode to infinity of the element's sign. Every byte decodes: E4M3 `S.1111.111` is NaN and
E4M3 has no infinities. Contents are not scanned or validated at binding. Storage encodings are not tensor dtypes, and
no F16/BF16 tensor may be substituted for the cache. The page is the unit a
data plan persists, reloads or leases.

Appends are `Strict`. Each (token, head) row of keys, and of values, takes
the smallest E8M0 scale with `max |x| <= 448 * scale`, so no element
saturates, and stores `RNE_e4m3(x / scale)`: dividing by a power of two is
exact, so each element rounds once. An all-zero row stores scale `0` and
keeps signed zeros. A row with any non-finite value stores NaN codes (`0x7f`)
with scale `0xff`, so it decodes to NaN. Every conforming writer therefore
produces bit-identical pages.

Attention is `Faithful`: logits and the softmax accumulate in F32 in an
unspecified order. A `-inf` logit contributes nothing; a NaN or `+inf` logit,
or a row whose logits are all `-inf`, yields NaN, as a direct F32 softmax
would.

`ax::kv` provides the checked attribute codecs and
`GraphBuilder::{kv_append, attention_decode}`. The native CPU executor runs
both nodes against a borrowed `KvCacheRef` binding; other executors reject
them rather than approximating.

## Decoder operations (1.5)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 72 | `SILU` | `1 -> 1` |
| 73 | `GATHER` | `2 -> 1` (`table`, `indices`) |
| 74 | `VIEW` | `1 -> 1` |
| 75 | `TOP_K` | `1 -> 2` (`input` -> `values`, `indices`) |

These are pure: resource flags are illegal on them.

`SILU` is `x * sigmoid(x)` over F32, with no attributes.

`GATHER` copies rows of a rank-2 `table` selected by I32 `indices` of any
shape of rank at most 7. The table is F32, or Q8 with whole 32-element blocks
per row, which decode exactly. The F32 result has shape
`indices.shape ++ [width]`. Every index must be in `0..rows`; an invocation
checks all of them before writing and fails otherwise. No attributes.

`VIEW` reinterprets its input's bytes: data never moves. Its output tensor
declares the view's shape, which reads `elements(output)` consecutive elements
of the input's row-major storage from an element `offset`. That covers
reshapes (offset zero, equal element counts) and contiguous ranges, such as
one part of a fused projection. Input and output share a dtype; for packed
dtypes the offset and the output's last dimension are multiples of the block
size. Attributes are exactly 8 bytes: LE u64 `offset`, with `offset +
elements(output) <= elements(input)`.

An executor aliases the input's storage. Every read of a view is a read of
its source, which therefore stays live until the view's last reader. A view
has no storage of its own, so the native CPU executor refuses views that
are graph outputs or leave their region, and nodes that read two different
ranges of one source. Strided views, such as one column range of several
rows, are not yet expressible.

`TOP_K` returns the `k` largest values of each last-axis row of an F32 input
as F32 `[.., k]`, then their I32 indices `[.., k]`. The order is descending
`f32::total_cmp`, with ties to the lower index: `+NaN` ranks above `+inf`
and `+0` above `-0`. Attributes are exactly 4 bytes: LE u32 `k`, with
`1 <= k <= width`.

`RMS_NORM` gains an 8-byte record: LE f32 `epsilon` (finite, positive), then
LE i32 `axis`. It normalizes over the trailing axes `axis..rank`, treating
each leading index as one row, and multiplies by a `weight` that is a
trailing slice of the input shape covering at least the normalized axes,
broadcast over the rest. A grouped norm, for example, normalizes
`[groups, width]` over axis 1 with a `[groups, width]` weight. The original
4-byte record keeps its exact 1.3 meaning and encoding: the whole tensor with
a same-shape weight, which is axis 0.

`ax::tensor_ops` provides the checked attribute codecs. The native CPU
executor runs `GATHER` and `TOP_K` exactly (`Strict`); `SILU` and `RMS_NORM`
keep the accuracy of their existing kernels.

## Recurrent state (1.6)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 76 | `CAUSAL_CONV1D` | `3 -> 1` (`input`, `weight`, `bias`) |
| 77 | `SSM_SCAN` | `3 -> 1` (`xbc`, `dt`, `head_params`) |

Each node carries state between invocations in two caller-owned F32
buffers. It reads the state before its first token from a **source**
resource and writes the state after its last token to a distinct **target**
resource; it never writes its source. The caller swaps the two buffers only
after an invocation completes, so a failed or repeated step recomputes from
an unchanged source, and rolling back one step rebinds the previous buffer.
The graph keeps both keys; only the bound buffers alternate. Several tokens
in one invocation run the same per-token recurrence in order, so a chunk
equals the same tokens stepped one at a time.

Both nodes require exactly `ordered | reads-resource | writes-resource` and
forbid `mutating`. Their attributes start with `[1, kind, 0, 0]` and the LE
u32 source group, source binding, target group and target binding; the two
keys must differ.

`CAUSAL_CONV1D` (kind `7`, 28 bytes) adds LE u32 `channels` and `kernel >= 2`.
`input` is F32 `[tokens, channels]`, `weight` F32 `[channels, kernel]` with the
oldest tap first, and `bias` F32 `[channels]`. The state is `[channels,
kernel - 1]`: each channel's previous inputs, oldest first. Output row `t` of
channel `c` is `(sum_i weight[c, i] * previous[i]) + weight[c, kernel - 1] *
input[t, c] + bias[c]`, summed in that order (`Strict`), with no activation.

`SSM_SCAN` (kind `8`, 44 bytes) adds LE u32 `heads`, `head_dim`, `groups`
(dividing `heads`) and `state_size`, then the F32 bits of `dt_min` (finite,
nonnegative) and `dt_max` (greater, possibly `+inf`). `xbc` is F32 `[tokens,
conv_dim]` where each row is `x` (`heads * head_dim`), then every group's `B`,
then every group's `C` (`groups * state_size` each). `dt` is F32 `[tokens,
heads]` before its bias, and `head_params` is F32 `[3, heads]`: `A_log`,
`dt_bias`, `D`. The state is `[heads, head_dim, state_size]`. For each token and
head `h` in group `g = h / (heads / groups)`:

```text
step  = clamp(softplus(dt + dt_bias[h]), dt_min, dt_max)
state = state * exp(step * -exp(A_log[h])) + (step * B[g]) * x[h]
y[h]  = state . C[g] + D[h] * x[h]
```

`softplus(v)` is `v` above 20 and `log1p(exp(v))` otherwise, as in PyTorch.
The clamp keeps NaN. The result is F32 `[tokens, heads * head_dim]`; the
readout sum is `Faithful`.

`ax::recurrent` provides the checked codecs and
`GraphBuilder::{causal_conv1d, ssm_scan}`. The native CPU executor binds
sources as `StateSourceRef` and targets as `StateTargetRef`, and checks both
kinds and exact lengths before any node runs.

## Mixture-of-experts dispatch (1.7)

| Code | Operation | Fixed arity |
| ---: | --- | --- |
| 78 | `MOE_EXPERTS` | `3 -> 1` (`hidden`, `scores`, `correction_bias`) |

`MOE_EXPERTS` routes each token to `top_k` of `experts` experts whose weights
live in a bound, read-only **expert store**, and returns the weighted sum of
their outputs. It requires exactly `reads-resource`. The store is never
serialized: it is borrowed for each invocation as a table of per-expert
projections, so it can point into memory-mapped checkpoint tensors as stored,
or into whichever experts a residency cache holds.

`hidden` is F32 `[tokens, hidden]`, `scores` F32 `[tokens, experts]` (router
probabilities, such as a sigmoid of the router logits), and `correction_bias`
F32 `[experts]`. For each token:

1. Keep the `top_k` largest `scores + correction_bias`, in descending
   `f32::total_cmp` order with ties to the lower index. The bias only selects.
2. Weight each pick by its unbiased score. With `normalize`, divide by the
   picks' sum plus `1e-20`. Multiply by `routed_scale`.
3. In ascending expert order, add `weight * expert(x)`, where
   `expert(x) = down_scale * down(act(up_scale * up(x)))`.

Each expert's `up` projection is `[intermediate, hidden]` and its `down`
projection `[hidden, intermediate]`, stored `[out, in]` in the store's dtype
with one F32 scale each (1 for BF16). The only activation so far is
`relu(x)^2`, with no gate projection. The result is F32 `[tokens, hidden]`;
sums are `Faithful`.

Attributes are exactly 36 bytes: `[1, 9, weights, activation]` (weights `0` =
E4M3, `1` = BF16; activation `0` = squared ReLU), then LE u32 group, binding,
`experts`, `top_k` (`1..=min(16, experts)`), `hidden`, `intermediate`, flags
(bit 0 `normalize`, others zero) and the F32 bits of `routed_scale` (finite,
positive).

`ax::moe` provides the checked codec and `GraphBuilder::moe_experts`. The
native CPU executor binds the store as an `ExpertStoreRef` and checks the
expert count, projection sizes and scales before any node runs.

## Fused epilogues (1.8)

An elementwise operation that consumes only its producer's result can be
computed in that producer's write-out instead of a separate node. Two
records carry one.

### `MATMUL_TRANSPOSED` epilogue

Attributes are either absent, keeping the plain 1.3 product, or exactly 8
bytes: the LE F32 bits of `scale`, then an LE u32 activation code (`0` none,
`1` SiLU, `2` sigmoid). The node keeps its `2 -> 1` arity and stays pure.
The result is

```
out = activation((lhs x transpose(rhs)) * scale)
```

`scale` must be finite; a negative scale is legal. An unknown activation
code is rejected, so a reader never silently drops an epilogue it does not
implement. This folds an FP8 checkpoint's per-tensor weight scale, and a
router's sigmoid, into the matmul.

The kernels also admit a residual accumulate, `input2 += activation(product
* scale)`, which folds a layer's closing `ADD` into its output projection.
It has deliberately **no** record here: it aliases its result onto an input,
and every `.ax` tensor has exactly one producer, so expressing it needs
alias semantics this format does not have. It is reachable only through the
kernel API.

### `CAUSAL_CONV1D` activation

The 1.6 record stays exactly 28 bytes and applies no activation. A 32-byte
record appends one further LE u32 activation code, using the same codes, and
applies it to each element after its bias. Taps read the input rows and the
source window, never the output, so the activation cannot affect a later
token's sum. The carried successor window is unchanged. An activated
convolution is `Faithful` rather than `Strict`, because the activation adds
transcendental rounding to an otherwise exact tap order.

`ax::tensor_ops` and `ax::recurrent` provide the checked codecs. A fused
result equals the unfused sequence through the same `axmath` kernel the
standalone node would have used.

## Validation and forward compatibility

Readers reject unknown flag bits, malformed ranges, unknown dtype tags,
noncanonical packing, and newer minor versions. Unknown operation codes remain
representable so an extension-aware adapter can report capability accurately;
ordinary cost models should return `supports == false`.

The 1.1 reader also rejects malformed voxel, RGB-volume, and byte-buffer
attribute records, nonempty attributes on codes 51 and 55–59, missing or
illegal resource flags, incorrect fixed arities, and 1.1 operations or flags
mislabeled as 1.0. The resource-read flag is required on codes 52–54 and
illegal on built-in codes 0–51 and 55–59; an unknown extension operation may
declare it, which requires 1.1 and disables input-version-only memoization.
Encoders choose the lowest required minor version, preserving ordinary 1.0
bytes rather than rewriting every artifact.
