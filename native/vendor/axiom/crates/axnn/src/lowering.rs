//! Allocation-free kernel candidate planning.
//!
//! The semantic operation is deliberately separate from its executable
//! implementation. A provider exposes candidates through a visitor, and the
//! planner selects the cheapest applicable candidate without a registry,
//! global constructor, or heap allocation. CPU, TOSA, and future device
//! providers can consequently share policy while keeping their implementation
//! identifiers private to the provider crate.
//!
//! This module is the required discovery path for tuned kernels. `.ax` names
//! semantic operations; `axnn` validates their typed geometry and selects an
//! implementation; an adapter invokes and caches the returned plan. Keeping
//! candidate discovery here prevents the graph format from growing one opcode
//! per microarchitecture and prevents optimized kernels from becoming
//! unreachable private entry points.

use crate::{
    ArgReduceOp, BinaryOp, CompareOp, DType, ReduceOp, RopeMode, Shape, TensorDesc, TernaryOp,
    UnaryOp,
};

/// Boolean operation over U8 masks.
///
/// Masks use `0` for false and any non-zero input for true. Implementations
/// normalize every result to `0` or `1` so the representation can safely cross
/// backend boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalOp {
    Not,
    And,
    Or,
    Xor,
}

/// Bit operations on I32 payloads. Shift counts use their low five bits;
/// right shift sign-extends and left shift discards high bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerBinaryOp {
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}

/// Backend-independent operation identity used during lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticOp {
    /// Shape- and dtype-preserving tensor copy.
    Identity,
    /// F32/I32 conversion. Float-to-integer truncates and saturates, NaN to
    /// zero; integer-to-float rounds to nearest, ties to even. A shader
    /// frontend must apply any narrower source-language conversion bounds.
    Cast,
    Unary(UnaryOp),
    Binary(BinaryOp),
    Ternary(TernaryOp),
    Compare(CompareOp),
    /// Boolean algebra over U8 masks.
    Logical(LogicalOp),
    IntegerBinary(IntegerBinaryOp),
    /// Select per element from the second or third input using a U8 mask in
    /// the first input.
    Select,
    Scale,
    Reduce(ReduceOp),
    ArgReduce(ArgReduceOp),
    MatMul,
    /// Row-wise matrix product: [m,k] times transpose([n,k]).
    MatMulTransposed,
    /// F32 rows times NVFP4 checkpoint rows without expanding the weights.
    /// The inputs are `[m,k]` F32 activations, `[n,k/2]` packed E2M1 bytes,
    /// and `[n,k/16]` E4M3 block scales.
    MatMulNvfp4Transposed,
    /// F32 rows times I8 weight rows with a separate F32 scale per output row.
    MatMulI8PerChannel,
    VecDot,
    Softmax,
    RmsNorm,
    LayerNorm,
    Rope(RopeMode),
    Conv1d,
    SsmConv,
    /// Broadcast three F32 planes into an xyz-interleaved `[H, W, 3]` tensor.
    Interleave3F32,
    /// Read a bound 4x4x4 brick occupancy grid and return distance and I32 cell ID.
    VoxelBrickTrace,
    /// Sample a bound interleaved RGB volume into three F32 planes.
    VolumeTrilinearRgb,
    /// Load a byte using an I32 index; widen to I32, or return zero outside
    /// the registered immutable buffer. Scalar indices may broadcast.
    BufferLoadU8,
    /// Paint one raster node onto a bound packed-XRGB target from its row of
    /// the parameter table and the band's dirty columns; the result is a
    /// one-word token.
    Raster(RasterOp),
    /// Encode F32 key and value rows into a bound key/value cache at an I32
    /// slot position; the result is a one-word token.
    KvAppend,
    /// Grouped-query attention of F32 query rows over a bound key/value
    /// cache, after the append whose token it consumes.
    AttentionDecode,
    /// Rows of a rank-2 F32 or packed table selected by I32 indices.
    Gather,
    /// A reinterpretation of its input's bytes: a reshape or a contiguous
    /// range. Executors alias the input; no data moves.
    View,
    /// The `k` largest last-axis values and their indices.
    TopK,
    /// Depthwise causal convolution continuing a carried input window.
    CausalConv1d,
    /// Mamba-2 selective state-space recurrence over carried state.
    SsmScan,
    /// Route each token to its top-k experts and sum their gated outputs,
    /// reading expert weights from a bound store.
    MoeExperts,
    /// Gated delta rule (Gated DeltaNet linear attention) over a carried
    /// key-by-value state.
    GatedDeltaRule,
    /// Each token's experts from its router logits ([`MoeRouteParams`]):
    /// I32 `(expert, weight bits)` slots.
    MoeRoute,
    /// The routed SwiGLU of an NVFP4 expert table: each slot's expert's
    /// `silu(gate(x)) * up(x)` ([`RoutedExpertsParams`]).
    MoeGateUpNvfp4,
    /// The routed down projection of an NVFP4 expert table, each slot's
    /// product weighted and summed in slot order ([`RoutedExpertsParams`]).
    MoeDownNvfp4,
}

/// The raster node family: what a compositor's scene lowers to, one node
/// per scene node, painting a bound packed-XRGB target in graph order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RasterOp {
    /// A rounded-rectangle fill with one premultiplied pixel.
    Fill,
    /// An inner ring along a rounded rectangle's edge.
    Ring,
    /// A soft shadow under a rounded rectangle.
    Shadow,
    /// A bound surface blitted 1:1 (source-over below full opacity) through
    /// an optional rounded aperture.
    SurfaceBlit,
    /// A rect of a bound surface scaled nearest-neighbour into the node.
    SurfaceSample,
    /// A run of atlas glyph coverage tinted with one colour.
    Glyphs,
}

/// Static parameters of one raster node: the rows of the frame program's
/// parameter table it paints in order (`row .. row + rows`, one node per
/// run of same-kind scene nodes), the target it paints, the resource
/// group its surfaces are bound in by slot, and the atlas and glyph table
/// a glyph run samples.  A row's contents are the per-frame
/// [`RasterRow`](crate::RasterRow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RasterNodeParams {
    pub row: u32,
    /// Rows painted, in order, from `row`; at least one.
    pub rows: u32,
    pub target: ResourceKey,
    pub surfaces: u32,
    pub atlas: ResourceKey,
    pub glyphs: ResourceKey,
    /// Width in pixels of the tile columns the row's tile mask and the
    /// band's dirty vector count in.
    pub tile_width: u32,
}

impl RasterNodeParams {
    /// Rows a parameter table may hold.
    pub const MAX_ROWS: u32 = 1 << 16;
    /// Tile columns a row's mask addresses (`RasterRow::tiles`).
    pub const MAX_TILE_COLUMNS: usize = 128;

    pub fn new(
        row: u32,
        rows: u32,
        target: ResourceKey,
        surfaces: u32,
        atlas: ResourceKey,
        glyphs: ResourceKey,
        tile_width: u32,
    ) -> Option<Self> {
        let params = Self {
            row,
            rows,
            target,
            surfaces,
            atlas,
            glyphs,
            tile_width,
        };
        params.is_valid().then_some(params)
    }

    pub const fn is_valid(self) -> bool {
        self.rows != 0
            && self.row < Self::MAX_ROWS
            && self.row as u64 + self.rows as u64 <= Self::MAX_ROWS as u64
            && self.tile_width != 0
            && self.tile_width <= u16::MAX as u32
    }

    /// The last row this node paints, exclusive.
    pub const fn end_row(self) -> u32 {
        self.row + self.rows
    }
}

impl SemanticOp {
    /// Number of tensor operands consumed by this semantic operation.
    pub const fn arity(self) -> u8 {
        match self {
            Self::Identity
            | Self::Cast
            | Self::BufferLoadU8
            | Self::Unary(_)
            | Self::Logical(LogicalOp::Not)
            | Self::Scale
            | Self::Reduce(_)
            | Self::ArgReduce(_)
            | Self::Softmax
            | Self::View
            | Self::TopK
            | Self::MoeRoute => 1,
            Self::Binary(_)
            | Self::IntegerBinary(_)
            | Self::Compare(_)
            | Self::Logical(LogicalOp::And | LogicalOp::Or | LogicalOp::Xor)
            | Self::MatMul
            | Self::MatMulTransposed
            | Self::VecDot
            | Self::RmsNorm
            | Self::Conv1d
            | Self::SsmConv
            | Self::Raster(_)
            | Self::Gather => 2,
            Self::Ternary(_)
            | Self::Select
            | Self::MatMulNvfp4Transposed
            | Self::MatMulI8PerChannel
            | Self::LayerNorm
            | Self::Rope(_)
            | Self::Interleave3F32
            | Self::VoxelBrickTrace
            | Self::VolumeTrilinearRgb
            | Self::KvAppend
            | Self::AttentionDecode
            | Self::CausalConv1d
            | Self::SsmScan
            | Self::MoeExperts
            | Self::GatedDeltaRule
            | Self::MoeGateUpNvfp4
            | Self::MoeDownNvfp4 => 3,
        }
    }
}

/// Provider-defined capability bits.
///
/// `axnn` assigns no meaning to individual bits. A provider publishes named
/// constants and the final runtime supplies the mask it is willing to admit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CapabilityMask(u64);

impl CapabilityMask {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(u64::MAX);

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::ops::BitOr for CapabilityMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for CapabilityMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Compact set of [`DType`] values for static candidate constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DTypeSet(u32);

impl DTypeSet {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(u32::MAX);

    pub const fn one(dtype: DType) -> Self {
        Self(1u32 << dtype as u8)
    }

    pub const fn contains(self, dtype: DType) -> bool {
        self.0 & (1u32 << dtype as u8) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::ops::BitOr for DTypeSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

/// Bit-preserving F32 parameter storage.
///
/// Keeping the IEEE-754 payload as an integer gives requests structural
/// equality (including distinct NaN payloads and signed zero) without making
/// lowering depend on floating-point comparison semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Bits(u32);

impl F32Bits {
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub fn from_f32(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Rank-bounded reduction-axis set. Bit `n` names logical axis `n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ReductionAxes(u8);

impl ReductionAxes {
    pub const NONE: Self = Self(0);

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, axis: u8) -> bool {
        axis < crate::MAX_RANK as u8 && self.0 & (1 << axis) != 0
    }

    pub fn all(rank: usize) -> Option<Self> {
        if rank > crate::MAX_RANK {
            return None;
        }
        let bits = if rank == 0 { 0 } else { u8::MAX >> (8 - rank) };
        Some(Self(bits))
    }
}

/// Provider-neutral 1D convolution geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv1dParams {
    pub stride: u32,
    pub dilation: u32,
    pub padding_before: u32,
    pub padding_after: u32,
    pub groups: u32,
}

impl Conv1dParams {
    /// Dense, valid convolution used by the current `ComputeBackend` API.
    pub const VALID: Self = Self {
        stride: 1,
        dilation: 1,
        padding_before: 0,
        padding_after: 0,
        groups: 1,
    };

    pub const fn is_valid(self) -> bool {
        self.stride != 0 && self.dilation != 0 && self.groups != 0
    }
}

/// Stable graph resource binding identity, independent of a runtime pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceKey {
    pub group: u32,
    pub binding: u32,
}

/// Length and binding identity of immutable, tightly packed unsigned bytes.
/// Empty buffers are valid; all loads then return zero. Length is bounded by
/// I32 so every valid index is representable without a floating conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U8BufferParams {
    pub resource: ResourceKey,
    pub elements: u32,
}

impl U8BufferParams {
    pub fn new(resource: ResourceKey, elements: u32) -> Option<Self> {
        let params = Self { resource, elements };
        params.is_valid().then_some(params)
    }

    pub const fn is_valid(self) -> bool {
        self.elements <= i32::MAX as u32 && self.elements as u64 <= isize::MAX as u64
    }
}

/// Geometry of a borrowed `Brick4x4x4U64` occupancy resource.
///
/// Each word stores one 4x4x4 brick. Voxel coordinates and cell IDs are
/// x-major: `x + dims[0] * (y + dims[1] * z)`. Dimensions must be positive
/// multiples of four and the voxel count must fit I32. The resource itself
/// is supplied only when executing a plan; requests never own its contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelBrickTraceParams {
    pub resource: ResourceKey,
    pub dims: [u32; 3],
}

impl VoxelBrickTraceParams {
    pub fn new(resource: ResourceKey, dims: [u32; 3]) -> Option<Self> {
        let params = Self { resource, dims };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.brick_count().is_some()
    }

    /// Checked resource length in u64 brick words, not bytes or voxel cells.
    pub fn brick_count(self) -> Option<usize> {
        let mut cells = 1_u32;
        for dimension in self.dims {
            if dimension == 0 || !dimension.is_multiple_of(4) {
                return None;
            }
            cells = cells.checked_mul(dimension)?;
        }
        if cells > i32::MAX as u32 {
            return None;
        }
        usize::try_from(cells / 64).ok()
    }
}

/// Storage encoding of a bound, x-major interleaved RGB volume.
///
/// Storage encodings are not tensor dtypes. In particular, unsigned E5M11
/// with exponent bias 15 is neither IEEE F16 nor BF16. Sampling promotes each
/// corner value to F32 before interpolation and accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolumeRgbStorage {
    /// Finite, nonnegative F32 components; negative zero is allowed.
    F32,
    /// Unsigned E5M11 words in `0x0000..=0xf7ff` (finite, at most 65520).
    Uf16E5M11,
    /// Unsigned E5M3 bytes in `0x00..=0xf7`, promoted exactly before F32 sampling.
    Uf8E5M3,
}

impl VolumeRgbStorage {
    pub const fn bytes_per_component(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Uf16E5M11 => 2,
            Self::Uf8E5M3 => 1,
        }
    }
}

/// Geometry and storage encoding of a borrowed RGB volume.
///
/// Values are tightly packed RGB triples at x-major index
/// `x + dims[0] * (y + dims[1] * z)`. Dimensions must be positive and at most
/// `2^24 + 1`, so every integer sampling endpoint `0..=dimension - 1` is
/// exactly representable in F32. Checked component and byte counts must fit
/// `usize`, and the byte length must not exceed `isize::MAX`.
///
/// Requests retain only this metadata. The execution adapter borrows a typed
/// immutable view, checks its exact length and finite nonnegative contents,
/// and never substitutes an F16/BF16 tensor or retains a storage pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VolumeRgbParams {
    pub resource: ResourceKey,
    pub dims: [u32; 3],
    pub storage: VolumeRgbStorage,
}

impl VolumeRgbParams {
    pub const MAX_DIMENSION: u32 = (1 << 24) + 1;

    pub fn new(resource: ResourceKey, dims: [u32; 3], storage: VolumeRgbStorage) -> Option<Self> {
        let params = Self {
            resource,
            dims,
            storage,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.checked_lengths().is_some()
    }

    /// Checked number of RGB triples, not individual channel components.
    pub fn voxel_count(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.0)
    }

    /// Checked number of tightly packed channel components (`3 * voxels`).
    pub fn element_count(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.1)
    }

    /// Checked byte length of the selected storage encoding.
    pub fn storage_bytes(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.2)
    }

    fn checked_lengths(self) -> Option<(usize, usize, usize)> {
        let mut voxels = 1_usize;
        for dimension in self.dims {
            if dimension == 0 || dimension > Self::MAX_DIMENSION {
                return None;
            }
            voxels = voxels.checked_mul(usize::try_from(dimension).ok()?)?;
        }
        let elements = voxels.checked_mul(3)?;
        let bytes = elements.checked_mul(self.storage.bytes_per_component())?;
        (bytes <= isize::MAX as usize).then_some((voxels, elements, bytes))
    }
}

/// Element and scale encoding of a bound attention key/value cache.
///
/// Storage encodings are not tensor dtypes: a cached element decodes to
/// `scale * element`, where the scale belongs to one (token, head) row of
/// keys or of values. Attention promotes both factors to F32; it never
/// substitutes an F16/BF16 tensor for the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheStorage {
    /// OCP E4M3 bytes (`DType::B8`) with one OCP MX E8M0 scale byte per
    /// (token, head) row of keys and one per row of values.
    ///
    /// Scale byte `e` is `2^(e - 127)` for `e` in `0..=254`, and NaN for
    /// `0xff`; E8M0 has no zero or infinity. A power-of-two scale only moves
    /// an E4M3 value's exponent, so `scale * element` is exact in F32
    /// whenever the result is finite, and every executor decodes a page
    /// identically, fused multiply-add or not.
    E4M3TokenScaleE8M0,
    /// Keys as [`E4M3TokenScaleE8M0`](Self::E4M3TokenScaleE8M0); values packed
    /// two 4-bit float (E2M1) per byte, one E8M0 scale per (token, head) row
    /// of values. The value head dimension must be even. Attention weights are
    /// far more tolerant of value error than key error, so halving the value
    /// plane saves a quarter of the cache for little quality cost.
    E4M3kE2M1vTokenScaleE8M0,
}

impl KvCacheStorage {
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::E4M3TokenScaleE8M0 | Self::E4M3kE2M1vTokenScaleE8M0 => 1,
        }
    }

    pub const fn scale_bytes(self) -> usize {
        match self {
            Self::E4M3TokenScaleE8M0 | Self::E4M3kE2M1vTokenScaleE8M0 => 1,
        }
    }
}

/// Resource-independent geometry of one layer's key/value cache.
///
/// This is the planning form of [`KvCacheParams`]: admission controllers and
/// storage planners can calculate page and capacity footprints without
/// inventing a resource key or allocating the cache. Byte arithmetic is
/// reported as `u64`, independently of the host pointer width. Constructing a
/// bound [`KvCacheParams`] still additionally proves that its complete byte
/// span fits `usize`/`isize` on the executing target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheLayout {
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Token slots per packed compute page; a power of two.
    pub page_tokens: u32,
    pub storage: KvCacheStorage,
}

impl KvCacheLayout {
    /// Largest logical position representable by the graph's I32 position
    /// input. This is an execution-format bound, not a recommended context.
    pub const MAX_CAPACITY_TOKENS: u32 = i32::MAX as u32;

    pub fn new(
        kv_heads: u32,
        head_dim: u32,
        page_tokens: u32,
        storage: KvCacheStorage,
    ) -> Option<Self> {
        let layout = Self {
            kv_heads,
            head_dim,
            page_tokens,
            storage,
        };
        layout.is_valid().then_some(layout)
    }

    pub fn is_valid(self) -> bool {
        let dim_even = match self.storage {
            KvCacheStorage::E4M3TokenScaleE8M0 => true,
            KvCacheStorage::E4M3kE2M1vTokenScaleE8M0 => self.head_dim.is_multiple_of(2),
        };
        dim_even
            && self.kv_heads != 0
            && self.head_dim != 0
            && self.page_tokens.is_power_of_two()
            && self.bytes_per_token().is_some()
            && self.page_bytes().is_some()
    }

    /// Key plus value bytes contributed by one token in one layer, including
    /// each row's storage scale. The value plane packs two E2M1 values per
    /// byte when the storage says so.
    pub fn bytes_per_token(self) -> Option<u64> {
        let dim = u64::from(self.head_dim);
        let scale = u64::try_from(self.storage.scale_bytes()).ok()?;
        let element = u64::try_from(self.storage.element_bytes()).ok()?;
        // Keys: `dim` elements with one row scale. Values: the same, or
        // `dim / 2` packed bytes when the value plane holds two E2M1 each.
        let key_row = dim.checked_mul(element)?.checked_add(scale)?;
        let value_bytes = match self.storage {
            KvCacheStorage::E4M3TokenScaleE8M0 => dim,
            KvCacheStorage::E4M3kE2M1vTokenScaleE8M0 => dim / 2,
        };
        let value_row = value_bytes.checked_add(scale)?;
        u64::from(self.kv_heads)
            .checked_mul(key_row.checked_add(value_row)?)
    }

    /// Checked byte length of one packed compute page.
    pub fn page_bytes(self) -> Option<u64> {
        self.bytes_per_token()?
            .checked_mul(u64::from(self.page_tokens))
    }

    /// Number of compute pages needed by an exact capacity.
    ///
    /// Capacities are positive multiples of `page_tokens` and remain within
    /// the graph's signed-I32 position vocabulary.
    pub fn page_count(self, capacity: u32) -> Option<u32> {
        if !self.is_valid()
            || capacity == 0
            || capacity > Self::MAX_CAPACITY_TOKENS
            || !capacity.is_multiple_of(self.page_tokens)
        {
            return None;
        }
        Some(capacity / self.page_tokens)
    }

    /// Planned bytes for one layer at `capacity`, without allocating or
    /// narrowing the result to the current host pointer width.
    pub fn capacity_bytes(self, capacity: u32) -> Option<u64> {
        self.page_bytes()?
            .checked_mul(u64::from(self.page_count(capacity)?))
    }

    /// Bytes for the same K/V geometry held as dense F32 rows. This is a
    /// residency comparison and migration aid; execution remains governed by
    /// `storage` and [`Self::capacity_bytes`].
    pub fn dense_f32_capacity_bytes(self, capacity: u32) -> Option<u64> {
        self.page_count(capacity)?;
        u64::from(self.kv_heads)
            .checked_mul(u64::from(self.head_dim))?
            .checked_mul(2)?
            .checked_mul(4)?
            .checked_mul(u64::from(capacity))
    }
}

/// Geometry and storage encoding of one layer's borrowed key/value cache.
///
/// The cache holds `capacity` token slots in `capacity / page_tokens` packed
/// pages. Each page is laid out head-major, keys before values, with each
/// head's rows contiguous so attention walks one run per head:
///
/// ```text
/// keys   [kv_heads][page_tokens][head_dim]   elements
/// values [kv_heads][page_tokens][head_dim]   elements
/// key scales   [kv_heads][page_tokens]       scales
/// value scales [kv_heads][page_tokens]       scales
/// ```
///
/// A page is the unit a data plan persists, reloads or leases. Requests
/// retain only this metadata: the execution adapter borrows the storage for
/// one invocation and never retains its pointer. The cache carries no length;
/// which slots are valid is the invocation's explicit position input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheParams {
    pub resource: ResourceKey,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Token slots; a positive multiple of `page_tokens`, at most `i32::MAX`
    /// so every slot is addressable by an I32 position.
    pub capacity: u32,
    /// Token slots per page; a power of two.
    pub page_tokens: u32,
    pub storage: KvCacheStorage,
}

impl KvCacheParams {
    pub fn new(
        resource: ResourceKey,
        kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        page_tokens: u32,
        storage: KvCacheStorage,
    ) -> Option<Self> {
        let params = Self {
            resource,
            kv_heads,
            head_dim,
            capacity,
            page_tokens,
            storage,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.checked_lengths().is_some()
    }

    /// The resource-independent layout used for admission and residency
    /// planning.
    pub const fn layout(self) -> KvCacheLayout {
        KvCacheLayout {
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            page_tokens: self.page_tokens,
            storage: self.storage,
        }
    }

    /// Planned byte length as `u64`, before target pointer-width checks.
    pub fn planned_storage_bytes(self) -> Option<u64> {
        self.layout().capacity_bytes(self.capacity)
    }

    pub fn page_count(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.0)
    }

    /// Checked byte length of one packed page.
    pub fn page_bytes(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.1)
    }

    /// Checked byte length of the whole bound resource.
    pub fn storage_bytes(self) -> Option<usize> {
        self.checked_lengths().map(|lengths| lengths.2)
    }

    fn checked_lengths(self) -> Option<(usize, usize, usize)> {
        let layout = self.layout();
        let pages = usize::try_from(layout.page_count(self.capacity)?).ok()?;
        let page_bytes = usize::try_from(layout.page_bytes()?).ok()?;
        let bytes = usize::try_from(layout.capacity_bytes(self.capacity)?).ok()?;
        (bytes <= isize::MAX as usize).then_some((pages, page_bytes, bytes))
    }
}

/// A grouped-query attention read of one layer's bound key/value cache.
///
/// `query_heads` is a positive multiple of the cache's `kv_heads`; query head
/// `h` reads cache head `h / (query_heads / kv_heads)`. Logits are
/// `logit_scale * dot(query, key)` over valid slots, before the softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionDecodeParams {
    pub cache: KvCacheParams,
    pub query_heads: u32,
    pub logit_scale: F32Bits,
}

impl AttentionDecodeParams {
    pub fn new(cache: KvCacheParams, query_heads: u32, logit_scale: f32) -> Option<Self> {
        let params = Self {
            cache,
            query_heads,
            logit_scale: F32Bits::from_f32(logit_scale),
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        let scale = self.logit_scale.to_f32();
        self.cache.is_valid()
            && self.query_heads != 0
            && self.query_heads.is_multiple_of(self.cache.kv_heads)
            && scale.is_finite()
            && scale > 0.0
    }
}

/// A bounded tree of token inputs in parent-before-child order. Node zero
/// is the common anchor; every other node descends from an earlier node.
/// This borrowed control view grants no storage authority. Linear operations
/// retain their existing semantics when no tree is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenTree<'a> {
    parents: &'a [u32],
}

impl<'a> TokenTree<'a> {
    /// Largest speculative verification batch: sixteen total input nodes.
    /// This bounds path scratch and retained controls, including the eight-row
    /// comparison configuration; increasing it requires auditing those owners.
    pub const MAX_NODES: usize = 16;
    pub const ROOT: u32 = u32::MAX;

    pub fn new(parents: &'a [u32]) -> Option<Self> {
        if parents.is_empty()
            || parents.len() > Self::MAX_NODES
            || parents[0] != Self::ROOT
            || parents
                .iter()
                .enumerate()
                .skip(1)
                .any(|(node, &parent)| parent as usize >= node)
        {
            return None;
        }
        Some(Self { parents })
    }

    pub const fn parents(self) -> &'a [u32] {
        self.parents
    }
    pub const fn len(self) -> usize {
        self.parents.len()
    }
    pub const fn is_empty(self) -> bool {
        false
    }

    /// Write one root-to-node path; strict parent ordering bounds the walk.
    pub fn path(self, node: usize, out: &mut [u32; TokenTree::MAX_NODES]) -> Option<usize> {
        if node >= self.len() {
            return None;
        }
        let mut next = node as u32;
        let mut count = 0;
        while next != Self::ROOT {
            out[count] = next;
            count += 1;
            next = self.parents[next as usize];
        }
        out[..count].reverse();
        Some(count)
    }

    pub fn contains_path(self, path: &[u32]) -> bool {
        path.first() == Some(&0)
            && path.len() <= self.len()
            && path.iter().all(|&node| (node as usize) < self.len())
            && path
                .windows(2)
                .all(|pair| self.parents[pair[1] as usize] == pair[0])
    }
}

/// Recurrent state carried between invocations in two caller-owned F32
/// buffers: `source` holds the state before a step and `target` receives the
/// state after it.
///
/// A step never writes its source. The caller swaps the two buffers only
/// after an invocation completes, so a failed or repeated step recomputes
/// from an unchanged source, and rolling back one step rebinds the previous
/// buffer. The graph keeps both keys; only the bound buffers alternate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateBinding {
    pub source: ResourceKey,
    pub target: ResourceKey,
}

impl StateBinding {
    pub fn is_valid(self) -> bool {
        self.source != self.target
    }
}

/// An elementwise activation folded into a kernel's write-out (#107).
///
/// `None` leaves the result as computed. The others are applied after any
/// scale, so a fused epilogue produces `activation(product · scale)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EpilogueActivation {
    #[default]
    None,
    Silu,
    Sigmoid,
}

impl EpilogueActivation {
    /// Wire code used by the `.ax` attribute records.
    pub const fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Silu => 1,
            Self::Sigmoid => 2,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::None),
            1 => Some(Self::Silu),
            2 => Some(Self::Sigmoid),
            _ => None,
        }
    }
}

/// A depthwise causal convolution over `channels` whose carried state is the
/// previous `kernel - 1` inputs of each channel, oldest first: a `[channels,
/// kernel - 1]` F32 buffer.
///
/// `activation` is folded into the same kernel, after the bias add (#107).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalConv1dParams {
    pub state: StateBinding,
    pub channels: u32,
    pub kernel: u32,
    pub activation: EpilogueActivation,
}

impl CausalConv1dParams {
    pub fn new(state: StateBinding, channels: u32, kernel: u32) -> Option<Self> {
        Self::new_activated(state, channels, kernel, EpilogueActivation::None)
    }

    /// A convolution whose bias add is followed by `activation`, computed in
    /// the same kernel rather than a separate elementwise node.
    pub fn new_activated(
        state: StateBinding,
        channels: u32,
        kernel: u32,
        activation: EpilogueActivation,
    ) -> Option<Self> {
        let params = Self {
            state,
            channels,
            kernel,
            activation,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.state.is_valid()
            && self.channels != 0
            && self.kernel >= 2
            && self.state_elements().is_some()
    }

    /// Checked F32 elements of each state buffer.
    pub fn state_elements(self) -> Option<usize> {
        let elements = usize::try_from(self.channels)
            .ok()?
            .checked_mul(usize::try_from(self.kernel.checked_sub(1)?).ok()?)?;
        (elements.checked_mul(4)? <= isize::MAX as usize).then_some(elements)
    }
}

/// A Mamba-2 selective state-space recurrence.
///
/// `heads` heads of width `head_dim` share `groups` groups of `B` and `C`
/// vectors of length `state_size`; head `h` uses group
/// `h / (heads / groups)`. The carried state is a `[heads, head_dim,
/// state_size]` F32 buffer. The step size is
/// `clamp(softplus(dt + dt_bias), dt_min, dt_max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsmScanParams {
    pub state: StateBinding,
    pub heads: u32,
    pub head_dim: u32,
    pub groups: u32,
    pub state_size: u32,
    pub dt_min: F32Bits,
    pub dt_max: F32Bits,
}

impl SsmScanParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: StateBinding,
        heads: u32,
        head_dim: u32,
        groups: u32,
        state_size: u32,
        dt_min: f32,
        dt_max: f32,
    ) -> Option<Self> {
        let params = Self {
            state,
            heads,
            head_dim,
            groups,
            state_size,
            dt_min: F32Bits::from_f32(dt_min),
            dt_max: F32Bits::from_f32(dt_max),
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        let (min, max) = (self.dt_min.to_f32(), self.dt_max.to_f32());
        self.state.is_valid()
            && self.head_dim != 0
            && self.state_size != 0
            && self.groups != 0
            && self.heads.is_multiple_of(self.groups)
            && self.heads != 0
            && min.is_finite()
            && min >= 0.0
            && max > min
            && self.state_elements().is_some()
            && self.conv_dim().is_some()
    }

    /// Checked F32 elements of each state buffer.
    pub fn state_elements(self) -> Option<usize> {
        let elements = usize::try_from(self.heads)
            .ok()?
            .checked_mul(usize::try_from(self.head_dim).ok()?)?
            .checked_mul(usize::try_from(self.state_size).ok()?)?;
        (elements.checked_mul(4)? <= isize::MAX as usize).then_some(elements)
    }

    /// Width of each input row: `x`, then every group's `B`, then `C`.
    pub fn conv_dim(self) -> Option<usize> {
        let x = usize::try_from(self.heads)
            .ok()?
            .checked_mul(usize::try_from(self.head_dim).ok()?)?;
        let bc = usize::try_from(self.groups)
            .ok()?
            .checked_mul(usize::try_from(self.state_size).ok()?)?
            .checked_mul(2)?;
        x.checked_add(bc)
    }
}

/// A gated delta rule (Gated DeltaNet) recurrence.
///
/// `value_heads` heads of width `value_dim` share `key_heads` query/key
/// heads of width `key_dim`: value head `h` reads key head
/// `h / (value_heads / key_heads)`. The carried state is a `[value_heads,
/// key_dim, value_dim]` F32 buffer, the layout FLA and Transformers keep.
/// With `qk_l2norm` the query and key are each scaled by
/// `1 / sqrt(sum(x * x) + 1e-6)` before the query takes `1 / sqrt(key_dim)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GatedDeltaRuleParams {
    pub state: StateBinding,
    pub key_heads: u32,
    pub key_dim: u32,
    pub value_heads: u32,
    pub value_dim: u32,
    pub qk_l2norm: bool,
}

impl GatedDeltaRuleParams {
    /// The epsilon the L2 normalization adds under its root (FLA's).
    pub const L2_EPSILON: f32 = 1e-6;

    pub fn new(
        state: StateBinding,
        (key_heads, key_dim): (u32, u32),
        (value_heads, value_dim): (u32, u32),
        qk_l2norm: bool,
    ) -> Option<Self> {
        let params = Self {
            state,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            qk_l2norm,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.state.is_valid()
            && self.key_heads != 0
            && self.key_dim != 0
            && self.value_dim != 0
            && self.value_heads != 0
            && self.value_heads.is_multiple_of(self.key_heads)
            && self.state_elements().is_some()
            && self.qkv_dim().is_some()
    }

    /// Checked F32 elements of each state buffer.
    pub fn state_elements(self) -> Option<usize> {
        let elements = usize::try_from(self.value_heads)
            .ok()?
            .checked_mul(usize::try_from(self.key_dim).ok()?)?
            .checked_mul(usize::try_from(self.value_dim).ok()?)?;
        (elements.checked_mul(4)? <= isize::MAX as usize).then_some(elements)
    }

    /// Width of each input row: every query head, then every key head,
    /// then every value head.
    pub fn qkv_dim(self) -> Option<usize> {
        let keys = usize::try_from(self.key_heads)
            .ok()?
            .checked_mul(usize::try_from(self.key_dim).ok()?)?;
        keys.checked_mul(2)?.checked_add(self.value_width()?)
    }

    /// Width of each output row: every value head.
    pub fn value_width(self) -> Option<usize> {
        usize::try_from(self.value_heads)
            .ok()?
            .checked_mul(usize::try_from(self.value_dim).ok()?)
    }
}

/// The most experts one token may select.
pub const MAX_MOE_TOP_K: u32 = 16;

/// The nonlinearity between an expert's up and down projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertActivation {
    /// `relu(x)^2`, with no gate projection (Nemotron-H).
    SquaredRelu,
}

/// A mixture-of-experts layer over a bound expert store.
///
/// Each expert is an up projection `[intermediate, hidden]` and a down
/// projection `[hidden, intermediate]` in the store's `weights` dtype (FP8
/// E4M3 or BF16), stored `[out, in]` as a checkpoint keeps them, each with
/// its own F32 scale. Tokens pick `top_k` of `experts` by
/// `scores + correction_bias`, then weight each pick by its unbiased score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeParams {
    pub resource: ResourceKey,
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub weights: DType,
    pub activation: ExpertActivation,
    /// Divide the picked weights by their sum plus `1e-20`.
    pub normalize: bool,
    /// Multiplies every picked weight after normalization.
    pub routed_scale: F32Bits,
}

impl MoeParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        resource: ResourceKey,
        experts: u32,
        top_k: u32,
        hidden: u32,
        intermediate: u32,
        weights: DType,
        activation: ExpertActivation,
        normalize: bool,
        routed_scale: f32,
    ) -> Option<Self> {
        let params = Self {
            resource,
            experts,
            top_k,
            hidden,
            intermediate,
            weights,
            activation,
            normalize,
            routed_scale: F32Bits::from_f32(routed_scale),
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        let scale = self.routed_scale.to_f32();
        matches!(self.weights, DType::U8 | DType::B8 | DType::B16)
            && (1..=MAX_MOE_TOP_K.min(self.experts)).contains(&self.top_k)
            && self.hidden != 0
            && self.intermediate != 0
            && scale.is_finite()
            && scale > 0.0
            && self.projection_bytes().is_some()
            && self.experts <= i32::MAX as u32
    }

    /// Checked bytes of one expert's up (or, equally, down) projection.
    pub fn projection_bytes(self) -> Option<usize> {
        let elements = self.projection_elements()?;
        let bytes = if self.weights == DType::U8 {
            elements.checked_add(1)?.checked_div(2)?
        } else {
            elements.checked_mul(self.weights.storage_bytes(1))?
        };
        (bytes <= isize::MAX as usize).then_some(bytes)
    }

    /// Checked logical elements in one expert projection. Packed formats
    /// must use this for work estimates rather than their storage byte count.
    pub fn projection_elements(self) -> Option<usize> {
        usize::try_from(self.hidden)
            .ok()?
            .checked_mul(usize::try_from(self.intermediate).ok()?)
    }

    /// E4M3 block-scale bytes for one NVFP4 projection.
    pub fn projection_scale_bytes(self) -> Option<usize> {
        if self.weights != DType::U8 {
            return Some(0);
        }
        usize::try_from(self.hidden)
            .ok()?
            .checked_mul(usize::try_from(self.intermediate).ok()?)?
            .checked_div(16)
    }
}

/// The most always-on (shared) experts a route appends after its picks.
pub const MAX_MOE_SHARED: u32 = 4;

/// The routing of an expert table ([`RoutedExpertsParams`]): each token's
/// `top_k` largest of its first `experts` router logits, in descending
/// order (ties to the lower index), weighted by a softmax over those
/// `top_k` (the renormalized softmax probabilities of all `experts`), then
/// its `shared` always-on experts, indices `experts..experts + shared`,
/// each weighted by the sigmoid of its own gate logit (the logits' last
/// `shared` columns). The result is I32 `[tokens, top_k + shared, 2]`:
/// each slot's expert index, then its F32 weight's bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeRouteParams {
    pub experts: u32,
    pub top_k: u32,
    pub shared: u32,
}

impl MoeRouteParams {
    pub fn new(experts: u32, top_k: u32, shared: u32) -> Option<Self> {
        let params = Self {
            experts,
            top_k,
            shared,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        (1..=MAX_MOE_TOP_K.min(self.experts)).contains(&self.top_k)
            && self.shared <= MAX_MOE_SHARED
            && self
                .experts
                .checked_add(self.shared)
                .is_some_and(|logits| logits <= i32::MAX as u32)
    }

    /// Router logits a token carries: every routed expert's, then each
    /// shared expert's gate.
    pub fn logits(self) -> u32 {
        self.experts + self.shared
    }

    /// Slots a token's route holds: its picks, then its shared experts.
    pub fn slots(self) -> u32 {
        self.top_k + self.shared
    }
}

/// One NVFP4 expert projection's record in an expert table, `[out, in]` as
/// a checkpoint stores it: [`ExpertRecord::HEADER`] bytes holding `scales`
/// F32 tensor scales (multipliers, one per equal run of rows: a fused
/// gate/up projection's gate rows, then its up rows), then `rows` rows of
/// `cols / 2` E2M1 code bytes (the even input low), then their E4M3 block
/// scales (`cols / 16` a row), the whole padded to a multiple of
/// [`ExpertRecord::ALIGN`]. A table is `[experts, record bytes]` U8, so an
/// expert's record is one stride away from the next, and its codes start
/// sixteen-byte aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertRecord {
    pub rows: u32,
    pub cols: u32,
    pub scales: u32,
}

impl ExpertRecord {
    /// Bytes before the codes: the tensor scales, zero-padded.
    pub const HEADER: usize = 16;
    /// A record's stride is a multiple of this.
    pub const ALIGN: usize = 64;

    fn is_valid(self) -> bool {
        self.rows != 0
            && self.cols.is_multiple_of(64)
            && self.cols != 0
            && (1..=4).contains(&self.scales)
            && self.rows.is_multiple_of(self.scales)
    }

    /// Bytes of E2M1 codes.
    pub fn code_bytes(self) -> Option<usize> {
        usize::try_from(self.rows)
            .ok()?
            .checked_mul(usize::try_from(self.cols).ok()? / 2)
    }

    /// Offset of the E4M3 block scales.
    pub fn block_scale_offset(self) -> Option<usize> {
        Self::HEADER.checked_add(self.code_bytes()?)
    }

    /// Bytes of one record, padding included.
    pub fn bytes(self) -> Option<usize> {
        if !self.is_valid() {
            return None;
        }
        let scales = usize::try_from(self.rows)
            .ok()?
            .checked_mul(usize::try_from(self.cols).ok()? / 16)?;
        let used = self.block_scale_offset()?.checked_add(scales)?;
        let bytes = used.checked_next_multiple_of(Self::ALIGN)?;
        (bytes <= isize::MAX as usize).then_some(bytes)
    }
}

/// A routed SwiGLU or down projection over an NVFP4 expert table of
/// `experts` records (routed and shared experts alike), `slots` routes a
/// token ([`MoeRouteParams::slots`]). The gate/up record is
/// [`Self::gate_up_record`] (`2 * intermediate` rows of `hidden`, the gate
/// rows first, two tensor scales), the down record
/// [`Self::down_record`] (`hidden` rows of `intermediate`, one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoutedExpertsParams {
    pub experts: u32,
    pub slots: u32,
    pub hidden: u32,
    pub intermediate: u32,
}

impl RoutedExpertsParams {
    pub fn new(experts: u32, slots: u32, hidden: u32, intermediate: u32) -> Option<Self> {
        let params = Self {
            experts,
            slots,
            hidden,
            intermediate,
        };
        params.is_valid().then_some(params)
    }

    pub fn is_valid(self) -> bool {
        self.experts != 0
            && self.experts <= i32::MAX as u32
            && (1..=MAX_MOE_TOP_K + MAX_MOE_SHARED).contains(&self.slots)
            && self
                .intermediate
                .checked_mul(2)
                .is_some_and(|rows| rows <= i32::MAX as u32)
            && self.gate_up_record().bytes().is_some()
            && self.down_record().bytes().is_some()
    }

    pub fn gate_up_record(self) -> ExpertRecord {
        ExpertRecord {
            rows: self.intermediate.saturating_mul(2),
            cols: self.hidden,
            scales: 2,
        }
    }

    pub fn down_record(self) -> ExpertRecord {
        ExpertRecord {
            rows: self.hidden,
            cols: self.intermediate,
            scales: 1,
        }
    }
}

/// I32 `[tokens, slots, 2]` routes.
fn is_routes(desc: &TensorDesc, tokens: usize, slots: u32) -> bool {
    desc.dtype == DType::I32 && desc.shape == Shape::from([tokens, slots as usize, 2])
}

/// A checked axis in `0..rank`; negative axes count from the end.
fn normalized_axis(rank: usize, axis: isize) -> Option<u8> {
    let rank = isize::try_from(rank).ok()?;
    let axis = if axis < 0 {
        rank.checked_add(axis)?
    } else {
        axis
    };
    (0..rank).contains(&axis).then(|| axis as u8)
}

/// A rank-0 I32 cache slot index.
fn is_slot_position(desc: &TensorDesc) -> bool {
    desc.dtype == DType::I32 && desc.shape.rank() == 0
}

/// F32 `[tokens, heads, head_dim]` rows with `1 <= tokens <= capacity`.
fn is_cache_rows(desc: &TensorDesc, heads: u32, cache: KvCacheParams) -> bool {
    desc.dtype == DType::F32
        && desc.shape.rank() == 3
        && (1..=cache.capacity as usize).contains(&desc.shape.dim(0))
        && desc.shape.dim(1) == heads as usize
        && desc.shape.dim(2) == cache.head_dim as usize
}

/// Operation-specific, allocation-free lowering parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelParams {
    None,
    /// Scalar multiplier, represented by exact F32 bits.
    Scalar(F32Bits),
    /// A fused matmul epilogue (#107): scale the product, then apply
    /// `activation`, as each output element is written. A plain
    /// [`Self::Scalar`] is the same thing with no activation.
    MatmulEpilogue {
        scale: F32Bits,
        activation: EpilogueActivation,
    },
    /// Normalization over the trailing axes `axis..rank`, with a
    /// numerical stability epsilon. Axis 0 normalizes the whole tensor.
    Normalization {
        epsilon: F32Bits,
        axis: u8,
    },
    /// A normalized, non-negative logical axis.
    Axis(u8),
    Reduction {
        axes: ReductionAxes,
        keep_dims: bool,
    },
    /// `rotary_dims` leading values of each head rotate (all of them for a
    /// full rotation); the rest pass through unchanged.
    Rope {
        mode: RopeMode,
        rotary_dims: u32,
    },
    Conv1d(Conv1dParams),
    VoxelBrickTrace(VoxelBrickTraceParams),
    VolumeTrilinearRgb(VolumeRgbParams),
    BufferLoadU8(U8BufferParams),
    Raster(RasterNodeParams),
    KvAppend(KvCacheParams),
    AttentionDecode(AttentionDecodeParams),
    /// The view starts `offset` elements into its input.
    View {
        offset: u64,
    },
    /// The `k` largest last-axis values.
    TopK {
        k: u32,
    },
    CausalConv1d(CausalConv1dParams),
    SsmScan(SsmScanParams),
    MoeExperts(MoeParams),
    GatedDeltaRule(GatedDeltaRuleParams),
    MoeRoute(MoeRouteParams),
    RoutedExperts(RoutedExpertsParams),
}

/// Host scalar result type. Scalars do not pretend to be tensor buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelScalar {
    F32,
    Index,
}

/// Maximum number of distinct tensor results in one semantic request.
pub const MAX_TENSOR_RESULTS: usize = 4;

/// Nonempty, allocation-free tensor result list in semantic ABI order.
///
/// Construction enforces the one-to-four result bound. Request construction
/// separately validates the shape, layout, and storage extent of every
/// descriptor; candidate constraints likewise apply to every result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorResults {
    tensors: [Option<TensorDesc>; MAX_TENSOR_RESULTS],
    len: u8,
}

impl TensorResults {
    pub fn new(tensors: &[TensorDesc]) -> Option<Self> {
        if tensors.is_empty() || tensors.len() > MAX_TENSOR_RESULTS {
            return None;
        }
        let mut result = Self {
            tensors: [None; MAX_TENSOR_RESULTS],
            len: tensors.len() as u8,
        };
        for (slot, tensor) in result.tensors.iter_mut().zip(tensors) {
            *slot = Some(*tensor);
        }
        Some(result)
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn get(&self, index: usize) -> Option<&TensorDesc> {
        if index >= self.len() {
            return None;
        }
        self.tensors[index].as_ref()
    }
}

/// Typed result contract presented to executable kernel providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::large_enum_variant,
    reason = "Bounded result descriptors stay Copy and allocation-free in the no_std contract"
)]
pub enum KernelResult {
    Tensor(TensorDesc),
    /// Two distinct tensor results in semantic ABI order.
    TensorPair([TensorDesc; 2]),
    /// One to four distinct tensor results in semantic ABI order.
    Tensors(TensorResults),
    Scalar(KernelScalar),
    /// The result aliases and replaces the named input.
    InPlace {
        input: u8,
    },
    /// The named input is both read and accumulated into.
    Accumulate {
        input: u8,
    },
}

impl KernelResult {
    const fn count(&self) -> usize {
        match self {
            Self::TensorPair(_) => 2,
            Self::Tensors(tensors) => tensors.len(),
            _ => 1,
        }
    }

    const fn tensor(&self, index: usize) -> Option<&TensorDesc> {
        match self {
            Self::Tensor(desc) if index == 0 => Some(desc),
            Self::TensorPair(descs) if index < 2 => Some(&descs[index]),
            Self::Tensors(tensors) => tensors.get(index),
            _ => None,
        }
    }
}

/// Side effects relevant to scheduling and alias analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KernelEffects(u8);

impl KernelEffects {
    pub const PURE: Self = Self(0);
    pub const WRITES_INPUT: Self = Self(1 << 0);
    pub const ACCUMULATES: Self = Self((1 << 0) | (1 << 1));
    /// Reads caller-bound resource contents, which may change between runs.
    pub const READS_RESOURCE: Self = Self(1 << 2);
    /// Writes a caller-bound resource (a raster node painting its target).
    pub const WRITES_RESOURCE: Self = Self(1 << 3);
    /// A raster node: reads its sources and writes its target.
    pub const RASTER: Self = Self((1 << 2) | (1 << 3));
    /// A recurrent step: reads the carried state and writes its successor.
    pub const UPDATES_RESOURCE: Self = Self((1 << 2) | (1 << 3));

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
}

/// Semantic request presented to a provider's candidate catalog.
///
/// The compatibility projections remain public while current providers move
/// to descriptor-aware selection. The authoritative input descriptors,
/// result, parameters, effects, and checked element count are private, so a
/// request can only be formed through typed constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelRequest {
    pub semantic: SemanticOp,
    /// Compatibility projection; use [`KernelRequest::input`] for layout and
    /// view information.
    pub input_dtypes: [Option<DType>; 3],
    /// Compatibility projection; use [`KernelRequest::input`] for the full
    /// descriptor.
    pub input_shapes: [Option<Shape>; 3],
    pub output_dtype: DType,
    /// Logical result shape after broadcasting.
    pub shape: Shape,
    pub contiguous: bool,
    inputs: [Option<TensorDesc>; 3],
    result: KernelResult,
    params: KernelParams,
    effects: KernelEffects,
    elements: usize,
    work_items: u64,
}

impl KernelRequest {
    fn from_parts(
        semantic: SemanticOp,
        inputs: [Option<TensorDesc>; 3],
        result: KernelResult,
        params: KernelParams,
        effects: KernelEffects,
    ) -> Option<Self> {
        for input in inputs.iter().flatten() {
            input.shape.checked_num_elements()?;
            input
                .layout
                .offset()
                .checked_add(input.checked_storage_bytes()?)?;
        }
        let result_desc = match result {
            KernelResult::Tensor(desc) => desc,
            KernelResult::TensorPair(descs) => descs[0],
            KernelResult::Tensors(descs) => *descs.get(0)?,
            KernelResult::Scalar(KernelScalar::F32) => {
                TensorDesc::try_new(Shape::SCALAR, DType::F32)?
            }
            KernelResult::Scalar(KernelScalar::Index) => {
                TensorDesc::try_new(Shape::SCALAR, DType::I64)?
            }
            KernelResult::InPlace { input } | KernelResult::Accumulate { input } => {
                inputs.get(input as usize).copied().flatten()?
            }
        };
        let elements = result_desc.shape.checked_num_elements()?;
        let input_elements = |index: usize| {
            inputs
                .get(index)
                .copied()
                .flatten()?
                .shape
                .checked_num_elements()
        };
        let work_items = match semantic {
            SemanticOp::MatMul => {
                let lhs = inputs[0]?;
                let rhs = inputs[1]?;
                lhs.shape
                    .dim(0)
                    .checked_mul(lhs.shape.dim(1))?
                    .checked_mul(rhs.shape.dim(1))?
            }
            SemanticOp::MatMulTransposed
            | SemanticOp::MatMulNvfp4Transposed
            | SemanticOp::MatMulI8PerChannel => {
                let lhs = inputs[0]?;
                let rhs = inputs[1]?;
                lhs.shape
                    .dim(0)
                    .checked_mul(lhs.shape.dim(1))?
                    .checked_mul(rhs.shape.dim(0))?
            }
            SemanticOp::VecDot
            | SemanticOp::Reduce(_)
            | SemanticOp::ArgReduce(_)
            | SemanticOp::Softmax
            | SemanticOp::RmsNorm
            | SemanticOp::LayerNorm
            | SemanticOp::Rope(_) => input_elements(0)?,
            SemanticOp::Conv1d | SemanticOp::SsmConv => {
                let kernel = inputs[1]?;
                elements.checked_mul(kernel.shape.dim(-1))?
            }
            // Keys and values are each encoded once.
            SemanticOp::KvAppend => input_elements(1)?.checked_mul(2)?,
            SemanticOp::CausalConv1d => match params {
                KernelParams::CausalConv1d(conv) => {
                    elements.checked_mul(usize::try_from(conv.kernel).ok()?)?
                }
                _ => return None,
            },
            // Two projections per pick: up, then down.
            SemanticOp::MoeExperts => match params {
                KernelParams::MoeExperts(moe) => usize::try_from(moe.top_k)
                    .ok()?
                    .checked_mul(moe.projection_elements()?)?
                    .checked_mul(2)?
                    .checked_mul(input_elements(0)? / usize::try_from(moe.hidden).ok()?)?,
                _ => return None,
            },
            // A multiply-add to update, and one to read, each state element.
            SemanticOp::SsmScan => match params {
                KernelParams::SsmScan(scan) => elements
                    .checked_mul(usize::try_from(scan.state_size).ok()?)?
                    .checked_mul(2)?,
                _ => return None,
            },
            // A gate and an up product (or one down product) per slot.
            SemanticOp::MoeGateUpNvfp4 | SemanticOp::MoeDownNvfp4 => match params {
                KernelParams::RoutedExperts(routed) => {
                    let record = if semantic == SemanticOp::MoeGateUpNvfp4 {
                        routed.gate_up_record()
                    } else {
                        routed.down_record()
                    };
                    let tokens = input_elements(2)? / usize::try_from(routed.slots).ok()? / 2;
                    tokens
                        .checked_mul(usize::try_from(routed.slots).ok()?)?
                        .checked_mul(usize::try_from(record.rows).ok()?)?
                        .checked_mul(usize::try_from(record.cols).ok()?)?
                }
                _ => return None,
            },
            // Per state element: the decayed key read, the delta update and
            // the query read-out.
            SemanticOp::GatedDeltaRule => match params {
                KernelParams::GatedDeltaRule(rule) => elements
                    .checked_mul(usize::try_from(rule.key_dim).ok()?)?
                    .checked_mul(3)?,
                _ => return None,
            },
            // Every query reads at most every slot twice (key, then value);
            // the executed count depends on the runtime position.
            SemanticOp::AttentionDecode => match params {
                KernelParams::AttentionDecode(attention) => elements
                    .checked_mul(usize::try_from(attention.cache.capacity).ok()?)?
                    .checked_mul(2)?,
                _ => return None,
            },
            _ => elements,
        };
        let work_items = u64::try_from(work_items).ok()?;
        result_desc
            .layout
            .offset()
            .checked_add(result_desc.checked_storage_bytes()?)?;
        for index in 1..result.count() {
            let desc = result.tensor(index)?;
            desc.shape.checked_num_elements()?;
            desc.layout
                .offset()
                .checked_add(desc.checked_storage_bytes()?)?;
        }
        let input_dtypes = inputs.map(|input| input.map(|desc| desc.dtype));
        let input_shapes = inputs.map(|input| input.map(|desc| desc.shape));
        let contiguous = result_desc.is_contiguous()
            && (1..result.count())
                .all(|index| result.tensor(index).is_some_and(TensorDesc::is_contiguous))
            && inputs.iter().flatten().all(TensorDesc::is_contiguous);
        Some(Self {
            semantic,
            input_dtypes,
            input_shapes,
            output_dtype: result_desc.dtype,
            shape: result_desc.shape,
            contiguous,
            inputs,
            result,
            params,
            effects,
            elements,
            work_items,
        })
    }

    fn tensor_result(shape: Shape, dtype: DType) -> Option<KernelResult> {
        Some(KernelResult::Tensor(TensorDesc::try_new(shape, dtype)?))
    }

    /// Compatibility constructor for shape-preserving requests.
    ///
    /// New semantic families should use the typed constructor bearing the
    /// operation's name so parameters and aliasing cannot be omitted.
    pub fn new(semantic: SemanticOp, desc: &TensorDesc) -> Self {
        assert!(
            matches!(
                semantic,
                SemanticOp::Identity
                    | SemanticOp::Unary(_)
                    | SemanticOp::Binary(_)
                    | SemanticOp::Ternary(_)
            ),
            "parameterized and typed-result operations require their typed KernelRequest constructor"
        );
        let inputs = match semantic.arity() {
            1 => [Some(*desc), None, None],
            2 => [Some(*desc), Some(*desc), None],
            _ => [Some(*desc), Some(*desc), Some(*desc)],
        };
        Self::from_parts(
            semantic,
            inputs,
            Self::tensor_result(desc.shape, desc.dtype).expect("kernel result descriptor overflow"),
            KernelParams::None,
            KernelEffects::PURE,
        )
        .expect("kernel input descriptor overflow")
    }

    /// Typed shape- and dtype-preserving unary request.
    pub fn unary(op: UnaryOp, input: &TensorDesc) -> Self {
        Self::new(SemanticOp::Unary(op), input)
    }

    /// Typed shape- and dtype-preserving identity request.
    pub fn identity(desc: &TensorDesc) -> Self {
        Self::new(SemanticOp::Identity, desc)
    }

    /// Explicit shape-preserving F32/I32 conversion or whole-row packed
    /// quantization/dequantization through F32. Quantization is never implicit.
    pub fn cast(input: &TensorDesc, dtype: DType) -> Option<Self> {
        let packed = matches!(
            (input.dtype, dtype),
            (DType::F32, DType::Q4 | DType::Q6 | DType::Q8)
                | (DType::Q4 | DType::Q6 | DType::Q8, DType::F32)
        );
        if packed {
            if input.shape.rank() == 0
                || !input
                    .shape
                    .dim(-1)
                    .is_multiple_of(input.dtype.block_size().max(dtype.block_size()))
            {
                return None;
            }
        } else if !matches!(input.dtype, DType::F32 | DType::I32)
            || !matches!(dtype, DType::F32 | DType::I32)
        {
            return None;
        }
        if input.dtype == dtype {
            return Some(Self::identity(input));
        }
        Self::from_parts(
            SemanticOp::Cast,
            [Some(*input), None, None],
            Self::tensor_result(input.shape, dtype)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    pub fn integer_binary(op: IntegerBinaryOp, lhs: &TensorDesc, rhs: &TensorDesc) -> Option<Self> {
        if lhs.dtype != DType::I32 || rhs.dtype != DType::I32 {
            return None;
        }
        let shape = lhs.shape.broadcast_with(&rhs.shape)?;
        Self::from_parts(
            SemanticOp::IntegerBinary(op),
            [Some(*lhs), Some(*rhs), None],
            Self::tensor_result(shape, DType::I32)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Checked resource load into a rank-two field. Index broadcasting must
    /// produce exactly `field`; no input or output aliases resource storage.
    pub fn buffer_load_u8(
        index: &TensorDesc,
        params: U8BufferParams,
        field: Shape,
    ) -> Option<Self> {
        if !params.is_valid()
            || index.dtype != DType::I32
            || field.rank() != 2
            || index.shape.broadcast_with(&field)? != field
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::BufferLoadU8,
            [Some(*index), None, None],
            Self::tensor_result(field, DType::I32)?,
            KernelParams::BufferLoadU8(params),
            KernelEffects::READS_RESOURCE,
        )
    }

    /// Typed raster node request: `params` is the I32 `[rows,
    /// RASTER_ROW_WORDS]` parameter table, `dirty` the U8 per-tile-column
    /// damage of the band, and the result a one-word I32 token — the node's
    /// effect is on the target resource it paints.
    pub fn raster(
        op: RasterOp,
        params: &TensorDesc,
        dirty: &TensorDesc,
        node: RasterNodeParams,
    ) -> Option<Self> {
        if !node.is_valid()
            || params.dtype != DType::I32
            || params.shape.rank() != 2
            || params.shape.dim(1) != crate::raster::RASTER_ROW_WORDS
            || params.shape.dim(0) < node.end_row() as usize
            || dirty.dtype != DType::U8
            || dirty.shape.rank() != 1
            || dirty.shape.dim(0) == 0
            || dirty.shape.dim(0) > RasterNodeParams::MAX_TILE_COLUMNS
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::Raster(op),
            [Some(*params), Some(*dirty), None],
            Self::tensor_result(Shape::from([1]), DType::I32)?,
            KernelParams::Raster(node),
            KernelEffects::RASTER,
        )
    }

    /// Append key and value rows to a bound key/value cache.
    ///
    /// `position` is a rank-0 I32 slot index; `keys` and `values` are F32
    /// `[tokens, kv_heads, head_dim]` with `1 <= tokens <= capacity`. The
    /// result is a one-word I32 token; the node's effect is the write to
    /// slots `position .. position + tokens`, which the executor checks
    /// against the capacity before any work runs.
    pub fn kv_append(
        position: &TensorDesc,
        keys: &TensorDesc,
        values: &TensorDesc,
        params: KvCacheParams,
    ) -> Option<Self> {
        if !params.is_valid()
            || !is_slot_position(position)
            || !is_cache_rows(keys, params.kv_heads, params)
            || values.dtype != DType::F32
            || values.shape != keys.shape
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::KvAppend,
            [Some(*position), Some(*keys), Some(*values)],
            Self::tensor_result(Shape::from([1]), DType::I32)?,
            KernelParams::KvAppend(params),
            KernelEffects::WRITES_RESOURCE,
        )
    }

    /// Grouped-query attention over a bound key/value cache.
    ///
    /// `token` is the one-word I32 result of the append this read follows,
    /// `position` the same rank-0 I32 slot index, and `queries` F32
    /// `[tokens, query_heads, head_dim]`. Query row `i` attends slots
    /// `0 ..= position + i`; the result has the query shape. Logits are
    /// `logit_scale * dot(query, key)`, followed by a softmax with F32
    /// accumulation. Summation order is unspecified (`Faithful`).
    pub fn attention_decode(
        token: &TensorDesc,
        position: &TensorDesc,
        queries: &TensorDesc,
        params: AttentionDecodeParams,
    ) -> Option<Self> {
        if !params.is_valid()
            || token.dtype != DType::I32
            || token.shape != Shape::from([1])
            || !is_slot_position(position)
            || !is_cache_rows(queries, params.query_heads, params.cache)
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::AttentionDecode,
            [Some(*token), Some(*position), Some(*queries)],
            Self::tensor_result(queries.shape, DType::F32)?,
            KernelParams::AttentionDecode(params),
            KernelEffects::READS_RESOURCE,
        )
    }

    /// Rows of `table` selected by `indices`.
    ///
    /// `table` is F32, packed Q8 (whole blocks per row) or BF16
    /// `[rows, width]`; `indices` is I32 of any shape below `MAX_RANK`. The F32 result is
    /// `indices.shape ++ [width]`. Every index must be in `0..rows`; the
    /// executor checks all of them before writing.
    pub fn gather(table: &TensorDesc, indices: &TensorDesc) -> Option<Self> {
        let width = table.shape.dim(1);
        if table.shape.rank() != 2
            || !matches!(table.dtype, DType::F32 | DType::Q8 | DType::B16)
            || !width.is_multiple_of(table.dtype.block_size())
            || indices.dtype != DType::I32
            || indices.shape.rank() >= crate::MAX_RANK
        {
            return None;
        }
        let mut dims = [0_usize; crate::MAX_RANK];
        let rank = indices.shape.rank();
        dims[..rank].copy_from_slice(indices.shape.dims());
        dims[rank] = width;
        Self::from_parts(
            SemanticOp::Gather,
            [Some(*table), Some(*indices), None],
            Self::tensor_result(Shape::try_from(&dims[..=rank]).ok()?, DType::F32)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// A view of `input`'s bytes, shaped like `output`: `output`'s elements
    /// start `offset` elements into `input`'s row-major storage. Nothing is
    /// copied; an executor aliases the input's storage.
    ///
    /// Covers reshapes (`offset` 0, equal element counts) and contiguous
    /// ranges such as one part of a fused projection. Packed dtypes must keep
    /// whole blocks: `offset` and the output's last dimension are multiples
    /// of the block size.
    pub fn view(input: &TensorDesc, output: &TensorDesc, offset: u64) -> Option<Self> {
        let elements = output.shape.checked_num_elements()?;
        let block = input.dtype.block_size();
        let start = usize::try_from(offset).ok()?;
        if output.dtype != input.dtype
            || !input.is_contiguous()
            || !output.is_contiguous()
            || start.checked_add(elements)? > input.shape.checked_num_elements()?
            || !start.is_multiple_of(block)
            || (output.shape.rank() > 0 && !output.shape.dim(-1).is_multiple_of(block))
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::View,
            [Some(*input), None, None],
            KernelResult::Tensor(*output),
            KernelParams::View { offset },
            KernelEffects::PURE,
        )
    }

    /// The `k` largest values of each last-axis row, in descending
    /// `f32::total_cmp` order with ties to the lower index, and their I32
    /// indices: F32 and I32 results `[.., k]` with `1 <= k <= width`.
    pub fn top_k(input: &TensorDesc, k: u32) -> Option<Self> {
        let rank = input.shape.rank();
        if input.dtype != DType::F32
            || rank == 0
            || k == 0
            || k as usize > input.shape.dim(-1)
            || input.shape.dim(-1) > i32::MAX as usize
        {
            return None;
        }
        let mut dims = [0_usize; crate::MAX_RANK];
        dims[..rank].copy_from_slice(input.shape.dims());
        dims[rank - 1] = k as usize;
        let shape = Shape::try_from(&dims[..rank]).ok()?;
        Self::from_parts(
            SemanticOp::TopK,
            [Some(*input), None, None],
            KernelResult::TensorPair([
                TensorDesc::try_new(shape, DType::F32)?,
                TensorDesc::try_new(shape, DType::I32)?,
            ]),
            KernelParams::TopK { k },
            KernelEffects::PURE,
        )
    }

    /// One or more steps of a depthwise causal convolution.
    ///
    /// `input` is F32 `[tokens, channels]`, `weight` F32 `[channels,
    /// kernel]` (oldest tap first) and `bias` F32 `[channels]`. Output row
    /// `t` of channel `c` is the dot product of the weights with the window
    /// ending at input `t`, plus the bias, with no activation. The node
    /// reads the carried window from the source buffer and writes the window
    /// after the last token to the target.
    pub fn causal_conv1d(
        input: &TensorDesc,
        weight: &TensorDesc,
        bias: &TensorDesc,
        params: CausalConv1dParams,
    ) -> Option<Self> {
        let channels = params.channels as usize;
        if !params.is_valid()
            || [input, weight, bias].iter().any(|d| d.dtype != DType::F32)
            || input.shape.rank() != 2
            || input.shape.dim(0) == 0
            || input.shape.dim(1) != channels
            || weight.shape != Shape::from([channels, params.kernel as usize])
            || bias.shape != Shape::from([channels])
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::CausalConv1d,
            [Some(*input), Some(*weight), Some(*bias)],
            Self::tensor_result(input.shape, DType::F32)?,
            KernelParams::CausalConv1d(params),
            KernelEffects::UPDATES_RESOURCE,
        )
    }

    /// One or more Mamba-2 recurrence steps.
    ///
    /// `xbc` is F32 `[tokens, conv_dim]` (each row `x`, then `B`, then `C`),
    /// `dt` F32 `[tokens, heads]` before its bias, and `head_params` F32
    /// `[3, heads]` holding `A_log`, `dt_bias` and `D`. For each token and
    /// head, with `A = -exp(A_log)`:
    ///
    /// ```text
    /// state = state * exp(dt * A) + (dt * B) x
    /// y     = state . C + D * x
    /// ```
    ///
    /// The F32 result is `[tokens, heads * head_dim]`. The node reads the
    /// carried state from the source buffer and writes the state after the
    /// last token to the target.
    pub fn ssm_scan(
        xbc: &TensorDesc,
        dt: &TensorDesc,
        head_params: &TensorDesc,
        params: SsmScanParams,
    ) -> Option<Self> {
        let heads = params.heads as usize;
        let tokens = xbc.shape.dim(0);
        if !params.is_valid()
            || [xbc, dt, head_params].iter().any(|d| d.dtype != DType::F32)
            || xbc.shape.rank() != 2
            || tokens == 0
            || Some(xbc.shape.dim(1)) != params.conv_dim()
            || dt.shape != Shape::from([tokens, heads])
            || head_params.shape != Shape::from([3, heads])
        {
            return None;
        }
        let width = heads.checked_mul(params.head_dim as usize)?;
        Self::from_parts(
            SemanticOp::SsmScan,
            [Some(*xbc), Some(*dt), Some(*head_params)],
            Self::tensor_result(Shape::from([tokens, width]), DType::F32)?,
            KernelParams::SsmScan(params),
            KernelEffects::UPDATES_RESOURCE,
        )
    }

    /// One or more gated delta rule steps.
    ///
    /// `qkv` is F32 `[tokens, qkv_dim]` (each row the query heads, then the
    /// key heads, then the value heads), `decay` F32 `[tokens,
    /// value_heads]` the log-space gate `g <= 0`, and `beta` F32 `[tokens,
    /// value_heads]` the write strength. For each token and value head,
    /// with `q` and `k` its key head's vectors (normalized per
    /// [`GatedDeltaRuleParams`]) and `S` its `[key_dim, value_dim]` state:
    ///
    /// ```text
    /// S     = S * exp(g)
    /// delta = (v - S^T k) * beta
    /// S     = S + k delta^T
    /// y     = S^T q
    /// ```
    ///
    /// The F32 result is `[tokens, value_heads * value_dim]`. The node reads
    /// the carried state from the source buffer and writes the state after
    /// the last token to the target.
    pub fn gated_delta_rule(
        qkv: &TensorDesc,
        decay: &TensorDesc,
        beta: &TensorDesc,
        params: GatedDeltaRuleParams,
    ) -> Option<Self> {
        let heads = params.value_heads as usize;
        let tokens = qkv.shape.dim(0);
        if !params.is_valid()
            || [qkv, decay, beta].iter().any(|d| d.dtype != DType::F32)
            || qkv.shape.rank() != 2
            || tokens == 0
            || Some(qkv.shape.dim(1)) != params.qkv_dim()
            || decay.shape != Shape::from([tokens, heads])
            || beta.shape != Shape::from([tokens, heads])
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::GatedDeltaRule,
            [Some(*qkv), Some(*decay), Some(*beta)],
            Self::tensor_result(Shape::from([tokens, params.value_width()?]), DType::F32)?,
            KernelParams::GatedDeltaRule(params),
            KernelEffects::UPDATES_RESOURCE,
        )
    }

    /// Route tokens to experts and sum their gated outputs.
    ///
    /// `hidden` is F32 `[tokens, hidden]`, `scores` F32 `[tokens, experts]`
    /// (router probabilities, for example a sigmoid), and `correction_bias`
    /// F32 `[experts]`, which only affects selection. Each token keeps the
    /// `top_k` largest `scores + correction_bias` in `f32::total_cmp` order,
    /// ties to the lower index, weights each by its unbiased score
    /// (normalized to their sum plus `1e-20` when requested, then scaled),
    /// and returns F32 `[tokens, hidden]`: the weighted sum of
    /// `down(act(up(x)))` over its picks in ascending expert order.
    pub fn moe_experts(
        hidden: &TensorDesc,
        scores: &TensorDesc,
        correction_bias: &TensorDesc,
        params: MoeParams,
    ) -> Option<Self> {
        let tokens = hidden.shape.dim(0);
        if !params.is_valid()
            || [hidden, scores, correction_bias]
                .iter()
                .any(|d| d.dtype != DType::F32)
            || hidden.shape.rank() != 2
            || tokens == 0
            || hidden.shape.dim(1) != params.hidden as usize
            || scores.shape != Shape::from([tokens, params.experts as usize])
            || correction_bias.shape != Shape::from([params.experts as usize])
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MoeExperts,
            [Some(*hidden), Some(*scores), Some(*correction_bias)],
            Self::tensor_result(hidden.shape, DType::F32)?,
            KernelParams::MoeExperts(params),
            KernelEffects::READS_RESOURCE,
        )
    }

    /// Route each token by its F32 router logits `[tokens, experts +
    /// shared]` ([`MoeRouteParams`]): I32 `[tokens, top_k + shared, 2]`.
    pub fn moe_route(logits: &TensorDesc, params: MoeRouteParams) -> Option<Self> {
        let tokens = logits.shape.dim(0);
        if !params.is_valid()
            || logits.dtype != DType::F32
            || logits.shape.rank() != 2
            || tokens == 0
            || logits.shape.dim(1) != params.logits() as usize
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MoeRoute,
            [Some(*logits), None, None],
            Self::tensor_result(
                Shape::from([tokens, params.slots() as usize, 2]),
                DType::I32,
            )?,
            KernelParams::MoeRoute(params),
            KernelEffects::PURE,
        )
    }

    /// The routed SwiGLU of F32 `x` `[tokens, hidden]` through the NVFP4
    /// gate/up records `[experts, gate_up_record bytes]` its `routes` pick
    /// (I32 `[tokens, slots, 2]`): F32 `[tokens, slots, intermediate]`.
    pub fn moe_gate_up_nvfp4(
        x: &TensorDesc,
        experts: &TensorDesc,
        routes: &TensorDesc,
        params: RoutedExpertsParams,
    ) -> Option<Self> {
        let tokens = x.shape.dim(0);
        if !params.is_valid()
            || x.dtype != DType::F32
            || x.shape.rank() != 2
            || tokens == 0
            || x.shape.dim(1) != params.hidden as usize
            || experts.dtype != DType::U8
            || experts.shape
                != Shape::from([params.experts as usize, params.gate_up_record().bytes()?])
            || !is_routes(routes, tokens, params.slots)
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MoeGateUpNvfp4,
            [Some(*x), Some(*experts), Some(*routes)],
            Self::tensor_result(
                Shape::from([tokens, params.slots as usize, params.intermediate as usize]),
                DType::F32,
            )?,
            KernelParams::RoutedExperts(params),
            KernelEffects::PURE,
        )
    }

    /// The routed down projection of F32 `h` `[tokens, slots,
    /// intermediate]` through the NVFP4 down records `[experts, down_record
    /// bytes]` its `routes` pick, each slot weighted by its route and summed
    /// in slot order: F32 `[tokens, hidden]`.
    pub fn moe_down_nvfp4(
        h: &TensorDesc,
        experts: &TensorDesc,
        routes: &TensorDesc,
        params: RoutedExpertsParams,
    ) -> Option<Self> {
        let tokens = h.shape.dim(0);
        if !params.is_valid()
            || h.dtype != DType::F32
            || h.shape
                != Shape::from([tokens, params.slots as usize, params.intermediate as usize])
            || tokens == 0
            || experts.dtype != DType::U8
            || experts.shape
                != Shape::from([params.experts as usize, params.down_record().bytes()?])
            || !is_routes(routes, tokens, params.slots)
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MoeDownNvfp4,
            [Some(*h), Some(*experts), Some(*routes)],
            Self::tensor_result(
                Shape::from([tokens, params.hidden as usize]),
                DType::F32,
            )?,
            KernelParams::RoutedExperts(params),
            KernelEffects::PURE,
        )
    }

    /// Typed boolean negation request over a U8 mask.
    pub fn logical_not(input: &TensorDesc) -> Option<Self> {
        if input.dtype != DType::U8 {
            return None;
        }
        Self::from_parts(
            SemanticOp::Logical(LogicalOp::Not),
            [Some(*input), None, None],
            Self::tensor_result(input.shape, DType::U8)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed binary boolean request with NumPy-compatible broadcasting.
    pub fn logical_binary(op: LogicalOp, lhs: &TensorDesc, rhs: &TensorDesc) -> Option<Self> {
        if op == LogicalOp::Not || lhs.dtype != DType::U8 || rhs.dtype != DType::U8 {
            return None;
        }
        let shape = lhs.shape.broadcast_with(&rhs.shape)?;
        Self::from_parts(
            SemanticOp::Logical(op),
            [Some(*lhs), Some(*rhs), None],
            Self::tensor_result(shape, DType::U8)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed binary request with NumPy/WGSL-compatible broadcasting.
    pub fn binary(op: BinaryOp, lhs: &TensorDesc, rhs: &TensorDesc) -> Option<Self> {
        if lhs.dtype != rhs.dtype {
            return None;
        }
        let shape = lhs.shape.broadcast_with(&rhs.shape)?;
        Self::from_parts(
            SemanticOp::Binary(op),
            [Some(*lhs), Some(*rhs), None],
            Self::tensor_result(shape, lhs.dtype)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed ternary request with broadcasting.
    pub fn ternary(
        op: TernaryOp,
        first: &TensorDesc,
        second: &TensorDesc,
        third: &TensorDesc,
    ) -> Option<Self> {
        if first.dtype != second.dtype || first.dtype != third.dtype {
            return None;
        }
        let shape = first
            .shape
            .broadcast_with(&second.shape)?
            .broadcast_with(&third.shape)?;
        Self::from_parts(
            SemanticOp::Ternary(op),
            [Some(*first), Some(*second), Some(*third)],
            Self::tensor_result(shape, first.dtype)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed comparison request producing a U8 boolean mask.
    pub fn compare(op: CompareOp, lhs: &TensorDesc, rhs: &TensorDesc) -> Option<Self> {
        if lhs.dtype != rhs.dtype {
            return None;
        }
        let shape = lhs.shape.broadcast_with(&rhs.shape)?;
        Self::from_parts(
            SemanticOp::Compare(op),
            [Some(*lhs), Some(*rhs), None],
            Self::tensor_result(shape, DType::U8)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed mask selection request: `U8 × T × T → T`.
    pub fn select(mask: &TensorDesc, on_true: &TensorDesc, on_false: &TensorDesc) -> Option<Self> {
        if mask.dtype != DType::U8 || on_true.dtype != on_false.dtype {
            return None;
        }
        let shape = mask
            .shape
            .broadcast_with(&on_true.shape)?
            .broadcast_with(&on_false.shape)?;
        Self::from_parts(
            SemanticOp::Select,
            [Some(*mask), Some(*on_true), Some(*on_false)],
            Self::tensor_result(shape, on_true.dtype)?,
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Typed scalar multiplication request.
    pub fn scale(input: &TensorDesc, scalar: f32) -> Self {
        Self::from_parts(
            SemanticOp::Scale,
            [Some(*input), None, None],
            Self::tensor_result(input.shape, input.dtype)
                .expect("kernel result descriptor overflow"),
            KernelParams::Scalar(F32Bits::from_f32(scalar)),
            KernelEffects::PURE,
        )
        .expect("kernel input descriptor overflow")
    }

    /// Typed full-tensor reduction returning an F32 host scalar.
    pub fn reduce(op: ReduceOp, input: &TensorDesc) -> Self {
        Self::from_parts(
            SemanticOp::Reduce(op),
            [Some(*input), None, None],
            KernelResult::Scalar(KernelScalar::F32),
            KernelParams::Reduction {
                axes: ReductionAxes::all(input.shape.rank()).expect("rank is bounded by Shape"),
                keep_dims: false,
            },
            KernelEffects::PURE,
        )
        .expect("kernel input descriptor overflow")
    }

    /// Typed full-tensor index reduction returning a host index scalar.
    pub fn arg_reduce(op: ArgReduceOp, input: &TensorDesc) -> Self {
        Self::from_parts(
            SemanticOp::ArgReduce(op),
            [Some(*input), None, None],
            KernelResult::Scalar(KernelScalar::Index),
            KernelParams::Reduction {
                axes: ReductionAxes::all(input.shape.rank()).expect("rank is bounded by Shape"),
                keep_dims: false,
            },
            KernelEffects::PURE,
        )
        .expect("kernel input descriptor overflow")
    }

    /// Pure rank-2 matrix multiplication into a fresh result tensor.
    pub fn matmul(lhs: &TensorDesc, rhs: &TensorDesc, output: &TensorDesc) -> Option<Self> {
        if !matmul_shape_matches(lhs.shape, rhs.shape, output.shape) {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMul,
            [Some(*lhs), Some(*rhs), None],
            KernelResult::Tensor(*output),
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Pure row-wise matrix product without materializing a transpose.
    /// Packed rows must contain whole blocks; partial row padding is not data.
    pub fn matmul_transposed(
        lhs: &TensorDesc,
        rhs: &TensorDesc,
        output: &TensorDesc,
    ) -> Option<Self> {
        if lhs.shape.rank() != 2
            || rhs.shape.rank() != 2
            || output.shape.rank() != 2
            || lhs.shape.dim(1) != rhs.shape.dim(1)
            || output.shape != Shape::from([lhs.shape.dim(0), rhs.shape.dim(0)])
            || !lhs.shape.dim(1).is_multiple_of(lhs.dtype.block_size())
            || !rhs.shape.dim(1).is_multiple_of(rhs.dtype.block_size())
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulTransposed,
            [Some(*lhs), Some(*rhs), None],
            KernelResult::Tensor(*output),
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Like [`Self::matmul_transposed`], but with the per-tensor weight
    /// scale folded into the result as a fused epilogue instead of a
    /// separate `Scale` node: `out = (lhs × rhsᵀ) · scale`. `scale` must be
    /// finite. Shapes are validated exactly as the unscaled form.
    pub fn matmul_transposed_scaled(
        lhs: &TensorDesc,
        rhs: &TensorDesc,
        output: &TensorDesc,
        scale: f32,
    ) -> Option<Self> {
        Self::matmul_transposed(lhs, rhs, output)?;
        if !scale.is_finite() {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulTransposed,
            [Some(*lhs), Some(*rhs), None],
            KernelResult::Tensor(*output),
            KernelParams::Scalar(F32Bits::from_f32(scale)),
            KernelEffects::PURE,
        )
    }

    /// F32 activations times native NVFP4 checkpoint rows. `tensor_scale`
    /// is the checkpoint's `weight_scale_2` and is fused into write-out.
    pub fn matmul_nvfp4_transposed(
        activations: &TensorDesc,
        packed: &TensorDesc,
        block_scales: &TensorDesc,
        output: &TensorDesc,
        tensor_scale: f32,
        activation: EpilogueActivation,
    ) -> Option<Self> {
        if activations.dtype != DType::F32
            || packed.dtype != DType::U8
            || block_scales.dtype != DType::B8
            || output.dtype != DType::F32
            || activations.shape.rank() != 2
            || packed.shape.rank() != 2
            || block_scales.shape.rank() != 2
            || output.shape.rank() != 2
            || !activations.shape.dim(1).is_multiple_of(16)
            || packed.shape.dim(1) != activations.shape.dim(1) / 2
            || block_scales.shape.dim(0) != packed.shape.dim(0)
            || block_scales.shape.dim(1) != activations.shape.dim(1) / 16
            || output.shape != Shape::from([activations.shape.dim(0), packed.shape.dim(0)])
            || !tensor_scale.is_finite()
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulNvfp4Transposed,
            [Some(*activations), Some(*packed), Some(*block_scales)],
            KernelResult::Tensor(*output),
            KernelParams::MatmulEpilogue {
                scale: F32Bits::from_f32(tensor_scale),
                activation,
            },
            KernelEffects::PURE,
        )
    }

    /// F32 rows times I8 weights, with one finite F32 scale per output row.
    /// The mathematical result is `(activations @ weights.T) * row_scales`.
    /// Providers may use reduced-precision arithmetic under an explicit
    /// approximate accuracy policy, but must preserve this port contract.
    pub fn matmul_i8_per_channel(
        activations: &TensorDesc,
        weights: &TensorDesc,
        row_scales: &TensorDesc,
        output: &TensorDesc,
    ) -> Option<Self> {
        if activations.dtype != DType::F32
            || weights.dtype != DType::I8
            || row_scales.dtype != DType::F32
            || output.dtype != DType::F32
            || activations.shape.rank() != 2
            || weights.shape.rank() != 2
            || row_scales.shape.rank() != 2
            || output.shape.rank() != 2
            || row_scales.shape.dim(0) != 1
            || row_scales.shape.dim(1) != weights.shape.dim(0)
            || activations.shape.dim(1) != weights.shape.dim(1)
            || output.shape != Shape::from([activations.shape.dim(0), weights.shape.dim(0)])
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulI8PerChannel,
            [Some(*activations), Some(*weights), Some(*row_scales)],
            KernelResult::Tensor(*output),
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Like [`Self::matmul_transposed_scaled`], with an elementwise
    /// `activation` applied after the scale in the same write-out:
    /// `out = activation((lhs × rhsᵀ) · scale)`. This folds a router's
    /// `sigmoid` into its matmul (#107).
    pub fn matmul_transposed_activated(
        lhs: &TensorDesc,
        rhs: &TensorDesc,
        output: &TensorDesc,
        scale: f32,
        activation: EpilogueActivation,
    ) -> Option<Self> {
        Self::matmul_transposed(lhs, rhs, output)?;
        if !scale.is_finite() {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulTransposed,
            [Some(*lhs), Some(*rhs), None],
            KernelResult::Tensor(*output),
            KernelParams::MatmulEpilogue {
                scale: F32Bits::from_f32(scale),
                activation,
            },
            KernelEffects::PURE,
        )
    }

    /// Like [`Self::matmul_transposed_scaled`], but accumulating into the
    /// residual stream held in input 2: `residual += (lhs × rhsᵀ) · scale`.
    /// This folds a layer's closing `Add` into the matmul's write-out, so a
    /// mixer's output projection and its residual are one kernel (#107).
    pub fn matmul_transposed_scaled_accumulate(
        lhs: &TensorDesc,
        rhs: &TensorDesc,
        residual: &TensorDesc,
        scale: f32,
    ) -> Option<Self> {
        Self::matmul_transposed(lhs, rhs, residual)?;
        if !scale.is_finite() || residual.dtype != DType::F32 {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMulTransposed,
            [Some(*lhs), Some(*rhs), Some(*residual)],
            KernelResult::Accumulate { input: 2 },
            KernelParams::Scalar(F32Bits::from_f32(scale)),
            KernelEffects::ACCUMULATES,
        )
    }

    /// Current `ComputeBackend::gemm` contract: `accumulator += lhs × rhs`.
    pub fn gemm_accumulate(
        lhs: &TensorDesc,
        rhs: &TensorDesc,
        accumulator: &TensorDesc,
    ) -> Option<Self> {
        if !matmul_shape_matches(lhs.shape, rhs.shape, accumulator.shape) {
            return None;
        }
        Self::from_parts(
            SemanticOp::MatMul,
            [Some(*lhs), Some(*rhs), Some(*accumulator)],
            KernelResult::Accumulate { input: 2 },
            KernelParams::None,
            KernelEffects::ACCUMULATES,
        )
    }

    /// Dot product returning an F32 host scalar.
    pub fn vec_dot(lhs: &TensorDesc, rhs: &TensorDesc) -> Option<Self> {
        if lhs.shape.checked_num_elements()? != rhs.shape.checked_num_elements()? {
            return None;
        }
        Self::from_parts(
            SemanticOp::VecDot,
            [Some(*lhs), Some(*rhs), None],
            KernelResult::Scalar(KernelScalar::F32),
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Softmax along a checked, normalized axis.
    pub fn softmax(input: &TensorDesc, axis: isize) -> Option<Self> {
        let rank = isize::try_from(input.shape.rank()).ok()?;
        let normalized = if axis < 0 {
            rank.checked_add(axis)?
        } else {
            axis
        };
        if normalized < 0 || normalized >= rank {
            return None;
        }
        Self::from_parts(
            SemanticOp::Softmax,
            [Some(*input), None, None],
            Self::tensor_result(input.shape, input.dtype)?,
            KernelParams::Axis(u8::try_from(normalized).ok()?),
            KernelEffects::PURE,
        )
    }

    /// RMS normalization of the whole tensor, with a same-shape weight.
    pub fn rms_norm(input: &TensorDesc, weight: &TensorDesc, epsilon: f32) -> Option<Self> {
        Self::rms_norm_over(input, weight, epsilon, 0)
    }

    /// RMS normalization over the trailing axes `axis..rank`: each leading
    /// index is one row, scaled by `weight`.
    ///
    /// `weight` is a trailing slice of the input shape covering at least the
    /// normalized axes, broadcast over the remaining leading ones. A grouped
    /// norm, for example, normalizes `[groups, width]` over axis 1 with a
    /// `[groups, width]` weight.
    pub fn rms_norm_over(
        input: &TensorDesc,
        weight: &TensorDesc,
        epsilon: f32,
        axis: isize,
    ) -> Option<Self> {
        let axis = normalized_axis(input.shape.rank(), axis)?;
        let rank = input.shape.rank();
        let weight_rank = weight.shape.rank();
        if input.dtype != weight.dtype
            || weight_rank > rank
            || weight_rank < rank - axis as usize
            || weight.shape.dims() != &input.shape.dims()[rank - weight_rank..]
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::RmsNorm,
            [Some(*input), Some(*weight), None],
            Self::tensor_result(input.shape, input.dtype)?,
            KernelParams::Normalization {
                epsilon: F32Bits::from_f32(epsilon),
                axis,
            },
            KernelEffects::PURE,
        )
    }

    /// Layer normalization over tensors with matching logical storage.
    pub fn layer_norm(
        input: &TensorDesc,
        weight: &TensorDesc,
        bias: &TensorDesc,
        epsilon: f32,
    ) -> Option<Self> {
        if input.dtype != weight.dtype
            || input.dtype != bias.dtype
            || input.shape != weight.shape
            || input.shape != bias.shape
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::LayerNorm,
            [Some(*input), Some(*weight), Some(*bias)],
            Self::tensor_result(input.shape, input.dtype)?,
            KernelParams::Normalization {
                epsilon: F32Bits::from_f32(epsilon),
                axis: 0,
            },
            KernelEffects::PURE,
        )
    }

    /// Rotary positional embedding of each head's whole width:
    /// [`Self::rope_partial`] with `rotary_dims` the head width.
    pub fn rope(
        input: &TensorDesc,
        cosine: &TensorDesc,
        sine: &TensorDesc,
        mode: RopeMode,
    ) -> Option<Self> {
        let width = u32::try_from(input.shape.dims().last().copied()?).ok()?;
        Self::rope_partial(input, cosine, sine, mode, width)
    }

    /// Rotary positional embedding of the leading `rotary_dims` values of
    /// each head. `input` is `[tokens, heads, head_dim]` or `[tokens,
    /// head_dim]`; `cosine` and `sine` are `[tokens, rotary_dims / 2]`.
    /// `mode` pairs values within the rotated span (NeoX: `i` with
    /// `i + rotary_dims / 2`), and the other `head_dim - rotary_dims` values
    /// of each head pass through. The result is a fresh tensor.
    pub fn rope_partial(
        input: &TensorDesc,
        cosine: &TensorDesc,
        sine: &TensorDesc,
        mode: RopeMode,
        rotary_dims: u32,
    ) -> Option<Self> {
        let rotary = usize::try_from(rotary_dims).ok()?;
        let dims = input.shape.dims();
        if !matches!(dims.len(), 2 | 3)
            || cosine.shape != sine.shape
            || cosine.dtype != sine.dtype
            || rotary == 0
            || !rotary.is_multiple_of(2)
            || rotary > dims[dims.len() - 1]
            || cosine.shape != Shape::from([dims[0], rotary / 2])
        {
            return None;
        }
        Self::from_parts(
            SemanticOp::Rope(mode),
            [Some(*input), Some(*cosine), Some(*sine)],
            Self::tensor_result(input.shape, input.dtype)?,
            KernelParams::Rope { mode, rotary_dims },
            KernelEffects::PURE,
        )
    }

    /// 1D convolution whose frontend has already inferred the output tensor.
    pub fn conv1d(
        input: &TensorDesc,
        kernel: &TensorDesc,
        output: &TensorDesc,
        params: Conv1dParams,
    ) -> Option<Self> {
        if !params.is_valid()
            || input.shape.rank() != 2
            || kernel.shape.rank() != 2
            || output.shape.rank() != 2
            || input.shape.dim(0) != kernel.shape.dim(0)
            || input.shape.dim(0) != output.shape.dim(0)
        {
            return None;
        }
        let kernel_extent = usize::try_from(params.dilation)
            .ok()?
            .checked_mul(kernel.shape.dim(1).checked_sub(1)?)?
            .checked_add(1)?;
        let padded_input = input
            .shape
            .dim(1)
            .checked_add(usize::try_from(params.padding_before).ok()?)?
            .checked_add(usize::try_from(params.padding_after).ok()?)?;
        let covered = padded_input.checked_sub(kernel_extent)?;
        let output_len = covered
            .checked_div(usize::try_from(params.stride).ok()?)?
            .checked_add(1)?;
        if output.shape.dim(1) != output_len {
            return None;
        }
        Self::from_parts(
            SemanticOp::Conv1d,
            [Some(*input), Some(*kernel), None],
            KernelResult::Tensor(*output),
            KernelParams::Conv1d(params),
            KernelEffects::PURE,
        )
    }

    /// SSM-style convolution preserving the input shape and dtype.
    pub fn ssm_conv(input: &TensorDesc, kernel: &TensorDesc) -> Option<Self> {
        if input.shape.rank() == 0 || kernel.shape.rank() == 0 {
            return None;
        }
        Self::from_parts(
            SemanticOp::SsmConv,
            [Some(*input), Some(*kernel), None],
            Self::tensor_result(input.shape, input.dtype)?,
            KernelParams::Conv1d(Conv1dParams::VALID),
            KernelEffects::PURE,
        )
    }

    /// Interleave broadcast F32 components into an explicit `[H, W, 3]` result.
    ///
    /// The explicit result shape lets three uniform camera coordinates expand
    /// to a full ray tile. Components must each broadcast into `[H, W]`;
    /// layouts remain part of the request for candidate-specific admission.
    pub fn interleave3_f32(
        x: &TensorDesc,
        y: &TensorDesc,
        z: &TensorDesc,
        output: &TensorDesc,
    ) -> Option<Self> {
        if output.dtype != DType::F32 || output.shape.rank() != 3 || output.shape.dim(2) != 3 {
            return None;
        }
        let plane = Shape::try_from(&output.shape.dims()[..2]).ok()?;
        for input in [x, y, z] {
            if input.dtype != DType::F32 || input.shape.broadcast_with(&plane) != Some(plane) {
                return None;
            }
        }
        Self::from_parts(
            SemanticOp::Interleave3F32,
            [Some(*x), Some(*y), Some(*z)],
            KernelResult::Tensor(*output),
            KernelParams::None,
            KernelEffects::PURE,
        )
    }

    /// Trace packed F32 rays against a borrowed brick occupancy resource.
    ///
    /// Origins and directions have matching `[H, W, 3]` shapes. The F32 limit
    /// has exactly one element. Results are `[H, W]` tensors in order: F32
    /// distance, then exact I32 cell ID. Misses are `(+infinity, -1)`; no ID
    /// passes through a floating-point representation. Nonfinite origins or
    /// directions and NaN/negative limits produce misses; +infinity is a valid
    /// limit. Finite rays retain scalar traversal rounding and tie semantics,
    /// including clamping direction magnitudes below `1e-12` to signed `1e-12`.
    pub fn voxel_brick_trace(
        origins: &TensorDesc,
        directions: &TensorDesc,
        limit: &TensorDesc,
        params: VoxelBrickTraceParams,
    ) -> Option<Self> {
        if !params.is_valid()
            || origins.dtype != DType::F32
            || directions.dtype != DType::F32
            || limit.dtype != DType::F32
            || origins.shape != directions.shape
            || origins.shape.rank() != 3
            || origins.shape.dim(2) != 3
            || limit.shape.checked_num_elements()? != 1
        {
            return None;
        }
        let plane = Shape::try_from(&origins.shape.dims()[..2]).ok()?;
        Self::from_parts(
            SemanticOp::VoxelBrickTrace,
            [Some(*origins), Some(*directions), Some(*limit)],
            KernelResult::TensorPair([
                TensorDesc::try_new(plane, DType::F32)?,
                TensorDesc::try_new(plane, DType::I32)?,
            ]),
            KernelParams::VoxelBrickTrace(params),
            KernelEffects::READS_RESOURCE,
        )
    }

    /// Sample a borrowed RGB volume at broadcast F32 index coordinates.
    ///
    /// `field` is an explicit rank-two `[H, W]` result shape, including when
    /// all coordinates are scalar. Three independent F32 planes are returned
    /// in red, green, blue order. No result aliases an operand or another
    /// result. Resource contents must satisfy [`VolumeRgbParams`].
    ///
    /// Coordinates are clamped to `[0, dimension - 1]`; infinities clamp to
    /// the respective edge, and any NaN coordinate produces three NaNs
    /// (payload unspecified). A dimension of one always selects its sole
    /// index, with a zero interpolation fraction. For each channel, start at
    /// positive zero and visit the eight corners in z/y/x order, x fastest.
    /// Corner weight is `(wx * wy) * wz`; accumulate `value * weight` with
    /// separate F32 multiplication and addition. Do not reassociate to nested
    /// lerps, skip zero-weight corners, or contract into fused operations.
    pub fn volume_trilinear_rgb(
        x: &TensorDesc,
        y: &TensorDesc,
        z: &TensorDesc,
        params: VolumeRgbParams,
        field: Shape,
    ) -> Option<Self> {
        if !params.is_valid() || field.rank() != 2 {
            return None;
        }
        for input in [x, y, z] {
            if input.dtype != DType::F32 || input.shape.broadcast_with(&field) != Some(field) {
                return None;
            }
        }
        let output = TensorDesc::try_new(field, DType::F32)?;
        if output.checked_storage_bytes()? > isize::MAX as usize {
            return None;
        }
        Self::from_parts(
            SemanticOp::VolumeTrilinearRgb,
            [Some(*x), Some(*y), Some(*z)],
            KernelResult::Tensors(TensorResults::new(&[output; 3])?),
            KernelParams::VolumeTrilinearRgb(params),
            KernelEffects::READS_RESOURCE,
        )
    }

    pub const fn primary_input_dtype(self) -> DType {
        match self.input_dtypes[0] {
            Some(dtype) => dtype,
            None => self.output_dtype,
        }
    }

    pub const fn input(&self, index: usize) -> Option<&TensorDesc> {
        if index >= self.inputs.len() {
            return None;
        }
        self.inputs[index].as_ref()
    }

    pub const fn inputs(&self) -> &[Option<TensorDesc>; 3] {
        &self.inputs
    }

    pub const fn result(&self) -> KernelResult {
        self.result
    }

    /// Number of semantic results, including a host scalar or aliased result.
    pub const fn result_count(&self) -> usize {
        self.result.count()
    }

    /// Tensor descriptor in result ABI order; host scalars have no tensor.
    pub const fn result_tensor(&self, index: usize) -> Option<&TensorDesc> {
        match &self.result {
            KernelResult::InPlace { input } | KernelResult::Accumulate { input } if index == 0 => {
                self.input(*input as usize)
            }
            _ => self.result.tensor(index),
        }
    }

    pub const fn params(&self) -> KernelParams {
        self.params
    }

    pub const fn effects(&self) -> KernelEffects {
        self.effects
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Operation-aware unit count used by deterministic cost recipes.
    /// Elementwise operations use result elements; matrix and convolution
    /// requests include their reduction extent.
    pub const fn work_items(self) -> u64 {
        self.work_items
    }
}

fn matmul_shape_matches(lhs: Shape, rhs: Shape, output: Shape) -> bool {
    lhs.rank() == 2
        && rhs.rank() == 2
        && output.rank() == 2
        && lhs.dim(1) == rhs.dim(0)
        && output.dim(0) == lhs.dim(0)
        && output.dim(1) == rhs.dim(1)
}

/// Numerical contract offered by a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum KernelAccuracy {
    /// Approximation whose error is operation/provider-specific.
    Approximate = 0,
    /// Faithful implementation suitable for ordinary shader/NN evaluation.
    Faithful = 1,
    /// Strictly preserves the semantic operation's specified rounding and
    /// exceptional-value behavior.
    Strict = 2,
}

/// Shape, dtype, and storage conditions attached to a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelShapePolicy {
    /// Every tensor input must have exactly the result shape.
    Exact,
    /// Inputs may right-align and broadcast into the result shape.
    Broadcast,
    /// The typed request constructor owns operation-specific shape checking.
    /// Suitable for reductions, matrix operations, and scalar results.
    Typed,
}

/// Shape, dtype, and storage conditions attached to a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConstraints {
    /// Admitted dtype at each semantic input position. Unused request inputs
    /// ignore the corresponding set.
    pub input_dtypes: [DTypeSet; 3],
    /// Every tensor result must belong to this set, including secondary results.
    pub output_dtypes: DTypeSet,
    pub min_elements: usize,
    /// Inclusive maximum; `usize::MAX` means unbounded.
    pub max_elements: usize,
    /// Element count must be divisible by this value. Zero and one both mean
    /// no divisibility constraint.
    pub element_multiple: usize,
    /// Inclusive minimum operation-aware work count. Elementwise requests use
    /// result elements, while typed requests include their reduction extent.
    pub min_work_items: u64,
    /// Inclusive maximum operation-aware work count; `u64::MAX` is unbounded.
    pub max_work_items: u64,
    /// Work count must be divisible by this value. Zero and one both mean no
    /// divisibility constraint.
    pub work_item_multiple: u64,
    /// Maximum rank of every tensor input and tensor result.
    pub max_rank: u8,
    pub contiguous_only: bool,
    pub shape_policy: KernelShapePolicy,
}

impl KernelConstraints {
    pub const fn elementwise(dtype: DType) -> Self {
        Self {
            input_dtypes: [DTypeSet::one(dtype); 3],
            output_dtypes: DTypeSet::one(dtype),
            min_elements: 0,
            max_elements: usize::MAX,
            element_multiple: 1,
            min_work_items: 0,
            max_work_items: u64::MAX,
            work_item_multiple: 1,
            max_rank: crate::MAX_RANK as u8,
            contiguous_only: true,
            shape_policy: KernelShapePolicy::Exact,
        }
    }

    pub const fn compare(input_dtype: DType) -> Self {
        Self {
            input_dtypes: [
                DTypeSet::one(input_dtype),
                DTypeSet::one(input_dtype),
                DTypeSet::NONE,
            ],
            output_dtypes: DTypeSet::one(DType::U8),
            min_elements: 0,
            max_elements: usize::MAX,
            element_multiple: 1,
            min_work_items: 0,
            max_work_items: u64::MAX,
            work_item_multiple: 1,
            max_rank: crate::MAX_RANK as u8,
            contiguous_only: true,
            shape_policy: KernelShapePolicy::Exact,
        }
    }

    pub const fn select(value_dtype: DType) -> Self {
        Self {
            input_dtypes: [
                DTypeSet::one(DType::U8),
                DTypeSet::one(value_dtype),
                DTypeSet::one(value_dtype),
            ],
            output_dtypes: DTypeSet::one(value_dtype),
            min_elements: 0,
            max_elements: usize::MAX,
            element_multiple: 1,
            min_work_items: 0,
            max_work_items: u64::MAX,
            work_item_multiple: 1,
            max_rank: crate::MAX_RANK as u8,
            contiguous_only: true,
            shape_policy: KernelShapePolicy::Exact,
        }
    }

    pub fn admits(self, request: &KernelRequest) -> bool {
        let elements = request.elements();
        let work_items = request.work_items();
        request
            .input_dtypes
            .iter()
            .enumerate()
            .all(|(index, dtype)| {
                dtype.is_none_or(|dtype| self.input_dtypes[index].contains(dtype))
            })
            && self.output_dtypes.contains(request.output_dtype)
            && (0..request.result_count()).all(|index| {
                request.result_tensor(index).is_none_or(|desc| {
                    self.output_dtypes.contains(desc.dtype)
                        && desc.shape.rank() <= self.max_rank as usize
                        && (!self.contiguous_only || desc.is_contiguous())
                })
            })
            && request.input_shapes.iter().all(|shape| {
                shape.is_none_or(|shape| match self.shape_policy {
                    KernelShapePolicy::Exact => shape == request.shape,
                    KernelShapePolicy::Broadcast => {
                        shape.broadcast_with(&request.shape) == Some(request.shape)
                    }
                    KernelShapePolicy::Typed => true,
                })
            })
            && elements >= self.min_elements
            && elements <= self.max_elements
            && (self.element_multiple <= 1 || elements.is_multiple_of(self.element_multiple))
            && work_items >= self.min_work_items
            && work_items <= self.max_work_items
            && (self.work_item_multiple <= 1 || work_items.is_multiple_of(self.work_item_multiple))
            && request.shape.rank() <= self.max_rank as usize
            && request
                .input_shapes
                .iter()
                .flatten()
                .all(|shape| shape.rank() <= self.max_rank as usize)
            && (!self.contiguous_only || request.contiguous)
    }
}

/// Deterministic cost recipe. Units are provider-relative; only candidates
/// from the same catalog should be compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCost {
    pub fixed: u64,
    pub per_work_item: u64,
    pub temporary_bytes: usize,
}

impl KernelCost {
    pub const fn new(fixed: u64, per_work_item: u64) -> Self {
        Self {
            fixed,
            per_work_item,
            temporary_bytes: 0,
        }
    }

    pub const fn with_temporary_bytes(mut self, bytes: usize) -> Self {
        self.temporary_bytes = bytes;
        self
    }

    fn estimate(self, work_items: u64, temporary_byte_cost: u64) -> u64 {
        self.fixed
            .saturating_add(self.per_work_item.saturating_mul(work_items))
            .saturating_add(temporary_byte_cost.saturating_mul(self.temporary_bytes as u64))
    }
}

/// One executable implementation of a semantic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCandidate<I> {
    pub semantic: SemanticOp,
    pub implementation: I,
    pub required_capabilities: CapabilityMask,
    pub constraints: KernelConstraints,
    pub accuracy: KernelAccuracy,
    pub cost: KernelCost,
}

/// Runtime-owned admission and selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelPolicy {
    pub capabilities: CapabilityMask,
    pub minimum_accuracy: KernelAccuracy,
    pub max_temporary_bytes: usize,
    pub temporary_byte_cost: u64,
}

impl KernelPolicy {
    pub const fn new(capabilities: CapabilityMask) -> Self {
        Self {
            capabilities,
            minimum_accuracy: KernelAccuracy::Approximate,
            max_temporary_bytes: usize::MAX,
            temporary_byte_cost: 0,
        }
    }

    pub const fn with_minimum_accuracy(mut self, accuracy: KernelAccuracy) -> Self {
        self.minimum_accuracy = accuracy;
        self
    }

    pub const fn with_max_temporary_bytes(mut self, bytes: usize) -> Self {
        self.max_temporary_bytes = bytes;
        self
    }

    pub const fn with_temporary_byte_cost(mut self, cost: u64) -> Self {
        self.temporary_byte_cost = cost;
        self
    }
}

/// Allocation-free sink used by static candidate catalogs.
pub trait CandidateVisitor<I> {
    fn visit(&mut self, candidate: KernelCandidate<I>);
}

/// Provider-owned static candidate enumeration.
pub trait KernelCatalog {
    type Implementation: Copy;

    fn visit_candidates<V: CandidateVisitor<Self::Implementation>>(
        &self,
        request: &KernelRequest,
        visitor: &mut V,
    );
}

/// Selected executable implementation and its deterministic estimated cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelPlan<I> {
    pub semantic: SemanticOp,
    pub implementation: I,
    pub input_dtypes: [Option<DType>; 3],
    pub input_shapes: [Option<Shape>; 3],
    pub output_dtype: DType,
    pub shape: Shape,
    pub accuracy: KernelAccuracy,
    pub estimated_cost: u64,
    pub work_items: u64,
    pub temporary_bytes: usize,
    inputs: [Option<TensorDesc>; 3],
    result: KernelResult,
    params: KernelParams,
    effects: KernelEffects,
}

impl<I> KernelPlan<I> {
    pub const fn input(&self, index: usize) -> Option<&TensorDesc> {
        if index >= self.inputs.len() {
            return None;
        }
        self.inputs[index].as_ref()
    }

    pub const fn inputs(&self) -> &[Option<TensorDesc>; 3] {
        &self.inputs
    }

    pub const fn result(&self) -> KernelResult {
        self.result
    }

    /// Number of semantic results, including a host scalar or aliased result.
    pub const fn result_count(&self) -> usize {
        self.result.count()
    }

    /// Tensor descriptor in result ABI order; host scalars have no tensor.
    pub const fn result_tensor(&self, index: usize) -> Option<&TensorDesc> {
        match &self.result {
            KernelResult::InPlace { input } | KernelResult::Accumulate { input } if index == 0 => {
                self.input(*input as usize)
            }
            _ => self.result.tensor(index),
        }
    }

    pub const fn params(&self) -> KernelParams {
        self.params
    }

    pub const fn effects(&self) -> KernelEffects {
        self.effects
    }
}

/// Select the cheapest applicable candidate without allocating.
pub fn select_kernel<C: KernelCatalog>(
    catalog: &C,
    request: &KernelRequest,
    policy: KernelPolicy,
) -> Option<KernelPlan<C::Implementation>> {
    struct Best<'a, I> {
        request: &'a KernelRequest,
        policy: KernelPolicy,
        plan: Option<KernelPlan<I>>,
    }

    impl<I: Copy> CandidateVisitor<I> for Best<'_, I> {
        fn visit(&mut self, candidate: KernelCandidate<I>) {
            if candidate.semantic != self.request.semantic
                || !self
                    .policy
                    .capabilities
                    .contains(candidate.required_capabilities)
                || candidate.accuracy < self.policy.minimum_accuracy
                || candidate.cost.temporary_bytes > self.policy.max_temporary_bytes
                || !candidate.constraints.admits(self.request)
            {
                return;
            }

            let estimated_cost = candidate
                .cost
                .estimate(self.request.work_items(), self.policy.temporary_byte_cost);
            let replace = self.plan.is_none_or(|current| {
                estimated_cost < current.estimated_cost
                    || (estimated_cost == current.estimated_cost
                        && candidate.accuracy > current.accuracy)
            });
            if replace {
                self.plan = Some(KernelPlan {
                    semantic: candidate.semantic,
                    implementation: candidate.implementation,
                    input_dtypes: self.request.input_dtypes,
                    input_shapes: self.request.input_shapes,
                    output_dtype: self.request.output_dtype,
                    shape: self.request.shape,
                    accuracy: candidate.accuracy,
                    estimated_cost,
                    work_items: self.request.work_items(),
                    temporary_bytes: candidate.cost.temporary_bytes,
                    inputs: *self.request.inputs(),
                    result: self.request.result(),
                    params: self.request.params(),
                    effects: self.request.effects(),
                });
            }
        }
    }

    let mut best = Best {
        request,
        policy,
        plan: None,
    };
    catalog.visit_candidates(request, &mut best);
    best.plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    const SIMD: CapabilityMask = CapabilityMask::from_bits(1 << 0);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Implementation {
        Scalar,
        Simd,
    }

    struct Catalog;

    impl KernelCatalog for Catalog {
        type Implementation = Implementation;

        fn visit_candidates<V: CandidateVisitor<Self::Implementation>>(
            &self,
            request: &KernelRequest,
            visitor: &mut V,
        ) {
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: Implementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints: KernelConstraints::elementwise(DType::F32),
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(0, 12),
            });
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: Implementation::Simd,
                required_capabilities: SIMD,
                constraints: KernelConstraints {
                    min_elements: 8,
                    ..KernelConstraints::elementwise(DType::F32)
                },
                accuracy: KernelAccuracy::Faithful,
                cost: KernelCost::new(24, 1),
            });
        }
    }

    #[test]
    fn runtime_capabilities_select_a_static_candidate() {
        let desc = TensorDesc::new(shape![128], DType::F32);
        let request = KernelRequest::new(SemanticOp::Unary(UnaryOp::Sqrt), &desc);
        assert_eq!(request.work_items(), 128);

        let scalar =
            select_kernel(&Catalog, &request, KernelPolicy::new(CapabilityMask::NONE)).unwrap();
        assert_eq!(scalar.implementation, Implementation::Scalar);

        let simd = select_kernel(&Catalog, &request, KernelPolicy::new(SIMD)).unwrap();
        assert_eq!(simd.implementation, Implementation::Simd);
        assert!(simd.estimated_cost < scalar.estimated_cost);
    }

    #[test]
    fn shape_dtype_and_accuracy_constraints_are_admitted_before_cost() {
        let small = TensorDesc::new(shape![4], DType::F32);
        let request = KernelRequest::new(SemanticOp::Binary(BinaryOp::Add), &small);
        let plan = select_kernel(&Catalog, &request, KernelPolicy::new(SIMD)).unwrap();
        assert_eq!(plan.implementation, Implementation::Scalar);

        let wrong_dtype = TensorDesc::new(shape![128], DType::F16);
        let request = KernelRequest::new(SemanticOp::Binary(BinaryOp::Add), &wrong_dtype);
        assert!(select_kernel(&Catalog, &request, KernelPolicy::new(SIMD)).is_none());

        let large = TensorDesc::new(shape![128], DType::F32);
        let request = KernelRequest::new(SemanticOp::Unary(UnaryOp::Sqrt), &large);
        let strict = select_kernel(
            &Catalog,
            &request,
            KernelPolicy::new(SIMD).with_minimum_accuracy(KernelAccuracy::Strict),
        )
        .unwrap();
        assert_eq!(strict.implementation, Implementation::Scalar);
    }

    #[test]
    fn typed_mask_requests_preserve_positional_dtypes_and_scalar_shapes() {
        let field = TensorDesc::new(shape![16], DType::F32);
        let scalar = TensorDesc::new(shape![1], DType::F32);
        let comparison = KernelRequest::compare(CompareOp::Greater, &field, &scalar).unwrap();
        assert_eq!(
            comparison.input_dtypes,
            [Some(DType::F32), Some(DType::F32), None]
        );
        assert_eq!(comparison.output_dtype, DType::U8);
        assert_eq!(comparison.shape, field.shape);

        let mut compare_constraints = KernelConstraints::compare(DType::F32);
        assert!(!compare_constraints.admits(&comparison));
        compare_constraints.shape_policy = KernelShapePolicy::Broadcast;
        assert!(compare_constraints.admits(&comparison));

        let mask = TensorDesc::new(shape![16], DType::U8);
        let selection = KernelRequest::select(&mask, &scalar, &field).unwrap();
        assert_eq!(
            selection.input_dtypes,
            [Some(DType::U8), Some(DType::F32), Some(DType::F32)]
        );
        assert_eq!(selection.output_dtype, DType::F32);
        assert_eq!(selection.shape, field.shape);
    }

    #[test]
    fn identity_and_logical_requests_preserve_typed_semantics() {
        let float = TensorDesc::new(shape![2, 3], DType::F32);
        let identity = KernelRequest::identity(&float);
        assert_eq!(identity.semantic, SemanticOp::Identity);
        assert_eq!(identity.input_dtypes, [Some(DType::F32), None, None]);
        assert_eq!(identity.output_dtype, DType::F32);
        assert_eq!(identity.shape, float.shape);

        let column = TensorDesc::new(shape![2, 1], DType::U8);
        let row = TensorDesc::new(shape![1, 3], DType::U8);
        let not = KernelRequest::logical_not(&column).unwrap();
        assert_eq!(not.semantic, SemanticOp::Logical(LogicalOp::Not));
        assert_eq!(not.output_dtype, DType::U8);

        let and = KernelRequest::logical_binary(LogicalOp::And, &column, &row).unwrap();
        assert_eq!(and.semantic, SemanticOp::Logical(LogicalOp::And));
        assert_eq!(and.input_dtypes, [Some(DType::U8), Some(DType::U8), None]);
        assert_eq!(and.shape, shape![2, 3]);

        assert!(KernelRequest::logical_not(&float).is_none());
        assert!(KernelRequest::logical_binary(LogicalOp::Not, &column, &row).is_none());
        assert!(KernelRequest::logical_binary(LogicalOp::Or, &column, &float).is_none());
    }

    #[test]
    fn requests_and_plans_retain_full_view_descriptors() {
        let base = TensorDesc::new(shape![4, 8], DType::F32);
        let view = base.slice(0, 2, 1).unwrap();
        let request = KernelRequest::identity(&view);
        assert_eq!(request.input(0), Some(&view));
        assert_eq!(request.input(0).unwrap().layout.offset(), 64);
        assert_eq!(
            request.input(0).unwrap().contiguous_byte_range(),
            Some(64..96)
        );

        let plan =
            select_kernel(&Catalog, &request, KernelPolicy::new(CapabilityMask::NONE)).unwrap();
        assert_eq!(plan.input(0), Some(&view));
        assert_eq!(plan.result(), request.result());
    }

    #[test]
    fn contiguous_admission_checks_the_result_layout_too() {
        let input = TensorDesc::new(shape![2, 3], DType::F32);
        let strided_result = input.transpose(0, 1).unwrap();
        assert!(!strided_result.is_contiguous());

        let request = KernelRequest::from_parts(
            SemanticOp::Identity,
            [Some(input), None, None],
            KernelResult::Tensor(strided_result),
            KernelParams::None,
            KernelEffects::PURE,
        )
        .unwrap();
        assert!(!request.contiguous);
        assert!(!KernelConstraints::elementwise(DType::F32).admits(&request));
    }

    #[test]
    fn typed_admission_checks_input_ranks_for_scalar_results() {
        let input = TensorDesc::new(shape![1, 1, 1, 1, 1, 1, 1, 1], DType::F32);
        let request = KernelRequest::reduce(ReduceOp::Sum, &input);
        assert_eq!(request.shape.rank(), 0);

        let constraints = KernelConstraints {
            max_rank: 2,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        };
        assert!(!constraints.admits(&request));
    }

    #[test]
    fn typed_admission_constrains_work_independently_of_result_elements() {
        let input = TensorDesc::new(shape![16], DType::F32);
        let request = KernelRequest::reduce(ReduceOp::Sum, &input);
        assert_eq!(request.elements(), 1);
        assert_eq!(request.work_items(), 16);

        let work_constrained = KernelConstraints {
            min_work_items: 8,
            max_work_items: 32,
            work_item_multiple: 8,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        };
        assert!(work_constrained.admits(&request));

        let result_constrained = KernelConstraints {
            min_elements: 8,
            ..work_constrained
        };
        assert!(!result_constrained.admits(&request));

        let odd = KernelRequest::reduce(ReduceOp::Sum, &TensorDesc::new(shape![10], DType::F32));
        assert!(!work_constrained.admits(&odd));
    }

    #[test]
    fn typed_parameters_results_and_effects_cover_compute_families() {
        let vector = TensorDesc::new(shape![8], DType::F32);
        let scale = KernelRequest::scale(&vector, -0.0);
        assert_eq!(
            scale.params(),
            KernelParams::Scalar(F32Bits::from_bits((-0.0f32).to_bits()))
        );
        assert_eq!(scale.result(), KernelResult::Tensor(vector));

        let reduce = KernelRequest::reduce(ReduceOp::Sum, &vector);
        assert_eq!(reduce.result(), KernelResult::Scalar(KernelScalar::F32));
        assert_eq!(
            reduce.params(),
            KernelParams::Reduction {
                axes: ReductionAxes::from_bits(1),
                keep_dims: false,
            }
        );
        let exact = KernelConstraints::elementwise(DType::F32);
        assert!(!exact.admits(&reduce));
        let typed_shapes = KernelConstraints {
            shape_policy: KernelShapePolicy::Typed,
            ..exact
        };
        assert!(typed_shapes.admits(&reduce));

        let softmax = KernelRequest::softmax(&vector, -1).unwrap();
        assert_eq!(softmax.params(), KernelParams::Axis(0));
        assert!(KernelRequest::softmax(&vector, 1).is_none());

        let lhs = TensorDesc::new(shape![2, 3], DType::F32);
        let rhs = TensorDesc::new(shape![3, 4], DType::F32);
        let accumulator = TensorDesc::new(shape![2, 4], DType::F32);
        let gemm = KernelRequest::gemm_accumulate(&lhs, &rhs, &accumulator).unwrap();
        assert_eq!(gemm.input(2), Some(&accumulator));
        assert_eq!(gemm.result(), KernelResult::Accumulate { input: 2 });
        assert!(gemm.effects().contains(KernelEffects::ACCUMULATES));
        assert_eq!(gemm.work_items(), 24);
        assert!(KernelRequest::matmul(&lhs, &rhs, &vector).is_none());

        let heads = TensorDesc::new(shape![2, 3, 8], DType::F32);
        let phase = TensorDesc::new(shape![2, 4], DType::F32);
        let rope = KernelRequest::rope(&heads, &phase, &phase, RopeMode::NeoX).unwrap();
        assert_eq!(
            rope.params(),
            KernelParams::Rope {
                mode: RopeMode::NeoX,
                rotary_dims: 8,
            }
        );
        assert!(
            matches!(rope.result(), KernelResult::Tensor(_)),
            "rope writes a fresh output tensor"
        );
        assert!(!rope.effects().contains(KernelEffects::WRITES_INPUT));
        // A partial rotation names its span; the phase rows must match it,
        // and the span must be even and within the head.
        let quarter = TensorDesc::new(shape![2, 1], DType::F32);
        let partial =
            KernelRequest::rope_partial(&heads, &quarter, &quarter, RopeMode::NeoX, 2).unwrap();
        assert_eq!(
            partial.params(),
            KernelParams::Rope {
                mode: RopeMode::NeoX,
                rotary_dims: 2,
            }
        );
        assert!(KernelRequest::rope_partial(&heads, &phase, &phase, RopeMode::NeoX, 2).is_none());
        assert!(
            KernelRequest::rope_partial(&heads, &quarter, &quarter, RopeMode::NeoX, 3).is_none()
        );
        assert!(KernelRequest::rope_partial(&heads, &phase, &phase, RopeMode::NeoX, 10).is_none());
        assert!(KernelRequest::rope(&heads, &quarter, &quarter, RopeMode::NeoX).is_none());
        assert!(KernelRequest::rope(&lhs, &vector, &vector, RopeMode::NeoX).is_none());

        let normalized = KernelRequest::rms_norm(&vector, &vector, 1.0e-5).unwrap();
        assert_eq!(
            normalized.params(),
            KernelParams::Normalization {
                epsilon: F32Bits::from_f32(1.0e-5),
                axis: 0
            }
        );
        let conv_output = TensorDesc::new(shape![2, 6], DType::F32);
        let kernel = TensorDesc::new(shape![2, 3], DType::F32);
        let conv = KernelRequest::conv1d(
            &TensorDesc::new(shape![2, 8], DType::F32),
            &kernel,
            &conv_output,
            Conv1dParams::VALID,
        )
        .unwrap();
        assert_eq!(conv.params(), KernelParams::Conv1d(Conv1dParams::VALID));
        assert_eq!(conv.work_items(), 36);
    }

    #[test]
    fn fp4_value_storage_halves_the_value_plane_and_rejects_odd_dim() {
        let fp4 = KvCacheLayout::new(2, 128, 16, KvCacheStorage::E4M3kE2M1vTokenScaleE8M0).unwrap();
        // Keys 129 bytes, values 65 bytes, per head; two heads.
        assert_eq!(fp4.bytes_per_token(), Some(2 * (128 + 1 + 64 + 1)));
        assert!(
            KvCacheLayout::new(2, 127, 16, KvCacheStorage::E4M3kE2M1vTokenScaleE8M0).is_none(),
            "an odd head dimension does not pack E2M1 pairs"
        );
        assert!(
            KvCacheLayout::new(2, 128, 16, KvCacheStorage::E4M3TokenScaleE8M0).is_some()
        );
    }

    #[test]
    fn kv_layout_plans_one_million_tokens_without_allocating() {
        let layout = KvCacheLayout::new(2, 128, 16, KvCacheStorage::E4M3TokenScaleE8M0)
            .expect("published FP8 geometry");
        assert_eq!(layout.bytes_per_token(), Some(516));
        assert_eq!(layout.page_bytes(), Some(8_256));
        assert_eq!(layout.page_count(1 << 20), Some(65_536));
        assert_eq!(layout.capacity_bytes(1 << 20), Some(541_065_216));

        let params = KvCacheParams::new(
            ResourceKey {
                group: 3,
                binding: 9,
            },
            2,
            128,
            1 << 20,
            16,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        assert_eq!(params.layout(), layout);
        assert_eq!(params.planned_storage_bytes(), Some(541_065_216));
        assert_eq!(params.storage_bytes(), Some(541_065_216));
    }

    #[test]
    fn kv_layout_refuses_unaddressable_or_partial_capacities() {
        let layout = KvCacheLayout::new(2, 128, 16, KvCacheStorage::E4M3TokenScaleE8M0)
            .expect("valid geometry");
        for capacity in [0, 15, 17, u32::MAX] {
            assert_eq!(layout.page_count(capacity), None);
            assert_eq!(layout.capacity_bytes(capacity), None);
        }
        assert!(KvCacheLayout::new(0, 128, 16, layout.storage).is_none());
        assert!(KvCacheLayout::new(2, 0, 16, layout.storage).is_none());
        assert!(KvCacheLayout::new(2, 128, 24, layout.storage).is_none());
    }

    fn voxel_params() -> VoxelBrickTraceParams {
        VoxelBrickTraceParams::new(
            ResourceKey {
                group: 1,
                binding: 7,
            },
            [32, 16, 8],
        )
        .unwrap()
    }

    #[test]
    fn brick_resource_geometry_is_checked_without_occupancy_storage() {
        let params = voxel_params();
        assert_eq!(params.brick_count(), Some(64));
        for dims in [
            [0, 4, 4],
            [4, 6, 4],
            [4, 4, 1],
            [134_217_728, 4, 4], // exactly 2^31 cells is outside the I32 bound
            [u32::MAX - 3, u32::MAX - 3, 4],
        ] {
            assert!(VoxelBrickTraceParams::new(params.resource, dims).is_none());
            assert!(!VoxelBrickTraceParams { dims, ..params }.is_valid());
        }
        let largest = VoxelBrickTraceParams::new(params.resource, [134_217_724, 4, 4]).unwrap();
        assert_eq!(largest.brick_count(), Some(33_554_431));
    }

    fn volume_params(storage: VolumeRgbStorage) -> VolumeRgbParams {
        VolumeRgbParams::new(voxel_params().resource, [24, 12, 24], storage).unwrap()
    }

    #[test]
    fn rgb_resource_geometry_checks_endpoints_products_and_storage_bytes() {
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let params = volume_params(storage);
            assert_eq!(params.voxel_count(), Some(6_912));
            assert_eq!(params.element_count(), Some(20_736));
            assert_eq!(
                params.storage_bytes(),
                Some(20_736 * storage.bytes_per_component())
            );
            for dims in [[1, 1, 1], [VolumeRgbParams::MAX_DIMENSION, 1, 1]] {
                let boundary = VolumeRgbParams::new(params.resource, dims, storage).unwrap();
                assert_eq!(boundary.voxel_count(), Some(dims[0] as usize));
                assert_eq!((dims[0] - 1) as f32 as u32, dims[0] - 1);
            }
            for dims in [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [VolumeRgbParams::MAX_DIMENSION + 1, 1, 1],
                [1, VolumeRgbParams::MAX_DIMENSION + 1, 1],
                [1, 1, VolumeRgbParams::MAX_DIMENSION + 1],
                [VolumeRgbParams::MAX_DIMENSION; 3],
                [u32::MAX; 3],
            ] {
                assert!(VolumeRgbParams::new(params.resource, dims, storage).is_none());
                let invalid = VolumeRgbParams { dims, ..params };
                assert!(!invalid.is_valid());
                assert_eq!(invalid.voxel_count(), None);
                assert_eq!(invalid.element_count(), None);
                assert_eq!(invalid.storage_bytes(), None);
            }
        }
        // The voxel/component products fit on 64-bit, while the selected F32
        // bytes exceed isize::MAX. UF16 remains a distinct, smaller encoding.
        #[cfg(target_pointer_width = "64")]
        {
            let dims = [1 << 24, 1 << 24, 4096];
            assert!(
                VolumeRgbParams::new(voxel_params().resource, dims, VolumeRgbStorage::F32)
                    .is_none()
            );
            let compact =
                VolumeRgbParams::new(voxel_params().resource, dims, VolumeRgbStorage::Uf16E5M11)
                    .unwrap();
            assert_eq!(compact.storage_bytes(), Some(6_usize << 60));
            // One further z doubling overflows bytes for F32 and exceeds the
            // addressable-slice bound for UF16, before any allocation occurs.
            for storage in [VolumeRgbStorage::F32, VolumeRgbStorage::Uf16E5M11] {
                assert!(
                    VolumeRgbParams::new(
                        voxel_params().resource,
                        [1 << 24, 1 << 24, 8192],
                        storage
                    )
                    .is_none()
                );
            }
        }
    }

    #[test]
    fn volume_sampling_expands_scalar_coordinates_into_three_explicit_planes() {
        let scalar = TensorDesc::new(Shape::SCALAR, DType::F32);
        let row = TensorDesc::new(shape![3], DType::F32);
        let column = TensorDesc::new(shape![2, 1], DType::F32);
        let dense = TensorDesc::new(shape![2, 3], DType::F32);
        for storage in [VolumeRgbStorage::F32, VolumeRgbStorage::Uf16E5M11] {
            for inputs in [[&scalar; 3], [&column, &row, &dense]] {
                let request = KernelRequest::volume_trilinear_rgb(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    volume_params(storage),
                    dense.shape,
                )
                .unwrap();
                assert_eq!(request.semantic, SemanticOp::VolumeTrilinearRgb);
                assert_eq!(request.semantic.arity(), 3);
                assert_eq!(request.result_count(), 3);
                for index in 0..3 {
                    assert_eq!(request.result_tensor(index), Some(&dense));
                }
                assert_eq!(request.result_tensor(3), None);
                assert_eq!(request.result_tensor(usize::MAX), None);
                assert_eq!(request.output_dtype, DType::F32);
                assert_eq!(request.shape, dense.shape);
                assert_eq!(request.elements(), 6);
                assert_eq!(request.work_items(), 6);
                assert_eq!(
                    request.params(),
                    KernelParams::VolumeTrilinearRgb(volume_params(storage))
                );
                assert_eq!(request.effects(), KernelEffects::READS_RESOURCE);
                assert!(
                    matches!(request.result(), KernelResult::Tensors(results) if results.len() == 3)
                );
            }
        }
        let params = volume_params(VolumeRgbStorage::F32);
        for invalid in [
            TensorDesc::new(shape![4], DType::F32),
            TensorDesc::new(shape![1, 1, 1], DType::F32),
            TensorDesc::new(Shape::SCALAR, DType::I32),
            TensorDesc::new(Shape::SCALAR, DType::F16),
            TensorDesc::new(Shape::SCALAR, DType::B16),
        ] {
            for index in 0..3 {
                let mut inputs = [&scalar; 3];
                inputs[index] = &invalid;
                assert!(
                    KernelRequest::volume_trilinear_rgb(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        params,
                        dense.shape
                    )
                    .is_none()
                );
            }
        }
        for field in [
            Shape::SCALAR,
            shape![6],
            shape![1, 2, 3],
            shape![usize::MAX, 2],
            shape![isize::MAX as usize, 1],
        ] {
            assert!(
                KernelRequest::volume_trilinear_rgb(&scalar, &scalar, &scalar, params, field)
                    .is_none()
            );
        }
        assert!(
            KernelRequest::volume_trilinear_rgb(
                &scalar,
                &scalar,
                &scalar,
                VolumeRgbParams {
                    dims: [0; 3],
                    ..params
                },
                dense.shape
            )
            .is_none()
        );
        let empty =
            KernelRequest::volume_trilinear_rgb(&scalar, &scalar, &scalar, params, shape![0, 3])
                .unwrap();
        assert_eq!(empty.elements(), 0);
        for index in 0..3 {
            assert_eq!(empty.result_tensor(index).unwrap().shape, shape![0, 3]);
        }
    }

    #[test]
    fn bounded_tensor_results_validate_and_admit_every_descriptor() {
        let primary = TensorDesc::new(shape![2, 3], DType::F32);
        assert!(TensorResults::new(&[]).is_none());
        assert!(TensorResults::new(&[primary; MAX_TENSOR_RESULTS + 1]).is_none());
        let constraints = KernelConstraints {
            max_rank: 2,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        };
        let with_results = |descs: &[TensorDesc]| {
            KernelRequest::from_parts(
                SemanticOp::Identity,
                [Some(primary), None, None],
                KernelResult::Tensors(TensorResults::new(descs)?),
                KernelParams::None,
                KernelEffects::PURE,
            )
        };
        for count in 1..=MAX_TENSOR_RESULTS {
            let descriptors = [primary; MAX_TENSOR_RESULTS];
            let results = TensorResults::new(&descriptors[..count]).unwrap();
            assert_eq!(results.len(), count);
            assert!(!results.is_empty());
            assert_eq!(results.get(count), None);
            assert_eq!(results.get(usize::MAX), None);
            let request = with_results(&descriptors[..count]).unwrap();
            assert_eq!(request.result_count(), count);
            assert!(constraints.admits(&request));
            for index in 0..count {
                assert_eq!(results.get(index), Some(&primary));
                assert_eq!(request.result_tensor(index), Some(&primary));
                for changed in [
                    TensorDesc::new(primary.shape, DType::I32),
                    TensorDesc::new(shape![1, 2, 3], DType::F32),
                    primary.transpose(0, 1).unwrap(),
                ] {
                    let mut changed_descriptors = descriptors;
                    changed_descriptors[index] = changed;
                    let changed_request = with_results(&changed_descriptors[..count]).unwrap();
                    assert!(
                        !constraints.admits(&changed_request),
                        "output {index}/{count}: {changed:?}"
                    );
                    if !changed.is_contiguous() {
                        assert!(!changed_request.contiguous);
                    }
                }
                for invalid in [
                    TensorDesc {
                        shape: shape![usize::MAX, 2],
                        ..primary
                    },
                    TensorDesc {
                        layout: crate::Layout::from_raw(&[12, 4], usize::MAX),
                        ..primary
                    },
                ] {
                    let mut invalid_descriptors = descriptors;
                    invalid_descriptors[index] = invalid;
                    assert!(with_results(&invalid_descriptors[..count]).is_none());
                }
            }
        }

        struct ResultCatalog(KernelConstraints);
        impl KernelCatalog for ResultCatalog {
            type Implementation = ();
            fn visit_candidates<V: CandidateVisitor<()>>(
                &self,
                request: &KernelRequest,
                visitor: &mut V,
            ) {
                visitor.visit(KernelCandidate {
                    semantic: request.semantic,
                    implementation: (),
                    required_capabilities: CapabilityMask::NONE,
                    constraints: self.0,
                    accuracy: KernelAccuracy::Strict,
                    cost: KernelCost::new(1, 1),
                });
            }
        }
        for count in 1..=MAX_TENSOR_RESULTS {
            let request = with_results(&[primary; MAX_TENSOR_RESULTS][..count]).unwrap();
            let plan = select_kernel(
                &ResultCatalog(constraints),
                &request,
                KernelPolicy::new(CapabilityMask::NONE),
            )
            .unwrap();
            assert_eq!(plan.result_count(), count);
            for index in 0..=count {
                assert_eq!(plan.result_tensor(index), request.result_tensor(index));
            }
            assert_eq!(plan.result_tensor(usize::MAX), None);
            assert_eq!(plan.result(), request.result());
        }
    }

    #[test]
    fn interleave_expands_uniform_and_broadcast_components_to_explicit_plane() {
        let scalar = TensorDesc::new(Shape::SCALAR, DType::F32);
        let column = TensorDesc::new(shape![2, 1], DType::F32);
        let row = TensorDesc::new(shape![3], DType::F32);
        let packed = TensorDesc::new(shape![2, 3, 3], DType::F32);
        for components in [[&scalar; 3], [&column, &row, &scalar]] {
            let request = KernelRequest::interleave3_f32(
                components[0],
                components[1],
                components[2],
                &packed,
            )
            .unwrap();
            assert_eq!(request.semantic, SemanticOp::Interleave3F32);
            assert_eq!(request.result_count(), 1);
            assert_eq!(request.result_tensor(0), Some(&packed));
            assert_eq!(request.result_tensor(1), None);
            assert_eq!(request.elements(), 18);
            assert_eq!(request.effects(), KernelEffects::PURE);
        }
        for invalid in [
            TensorDesc::new(shape![2, 3], DType::F32),
            TensorDesc::new(shape![2, 3, 4], DType::F32),
            TensorDesc::new(shape![2, 3, 3], DType::F16),
        ] {
            assert!(KernelRequest::interleave3_f32(&scalar, &scalar, &scalar, &invalid).is_none());
        }
        for invalid in [
            TensorDesc::new(shape![4], DType::F32),
            TensorDesc::new(shape![1, 1, 1], DType::F32),
            TensorDesc::new(Shape::SCALAR, DType::I32),
        ] {
            assert!(KernelRequest::interleave3_f32(&invalid, &scalar, &scalar, &packed).is_none());
        }
    }

    #[test]
    fn voxel_query_retains_separate_distance_id_and_resource_contracts() {
        let packed = TensorDesc::new(shape![2, 3, 3], DType::F32);
        for limit_shape in [Shape::SCALAR, shape![1], shape![1, 1, 1, 1]] {
            let limit = TensorDesc::new(limit_shape, DType::F32);
            let request =
                KernelRequest::voxel_brick_trace(&packed, &packed, &limit, voxel_params()).unwrap();
            assert_eq!(request.semantic.arity(), 3);
            assert_eq!(request.result_count(), 2);
            assert_eq!(
                request.result_tensor(0),
                Some(&TensorDesc::new(shape![2, 3], DType::F32))
            );
            assert_eq!(
                request.result_tensor(1),
                Some(&TensorDesc::new(shape![2, 3], DType::I32))
            );
            assert_eq!(request.result_tensor(2), None);
            assert_eq!(request.output_dtype, DType::F32);
            assert_eq!(request.shape, shape![2, 3]);
            assert_eq!(request.elements(), 6);
            assert_eq!(request.work_items(), 6);
            assert_eq!(
                request.params(),
                KernelParams::VoxelBrickTrace(voxel_params())
            );
            assert_eq!(request.effects(), KernelEffects::READS_RESOURCE);
            assert!(!request.effects().contains(KernelEffects::WRITES_INPUT));
        }
        let limit = TensorDesc::new(Shape::SCALAR, DType::F32);
        for invalid in [
            TensorDesc::new(shape![2, 3], DType::F32),
            TensorDesc::new(shape![2, 3, 4], DType::F32),
            TensorDesc::new(shape![2, 3, 3], DType::F16),
        ] {
            assert!(
                KernelRequest::voxel_brick_trace(&invalid, &invalid, &limit, voxel_params())
                    .is_none()
            );
            assert!(
                KernelRequest::voxel_brick_trace(&packed, &invalid, &limit, voxel_params())
                    .is_none()
            );
        }
        for invalid in [
            TensorDesc::new(shape![2], DType::F32),
            TensorDesc::new(shape![0], DType::F32),
            TensorDesc::new(Shape::SCALAR, DType::F16),
        ] {
            assert!(
                KernelRequest::voxel_brick_trace(&packed, &packed, &invalid, voxel_params())
                    .is_none()
            );
        }
        assert!(
            KernelRequest::voxel_brick_trace(
                &packed,
                &packed,
                &limit,
                VoxelBrickTraceParams {
                    dims: [1, 1, 1],
                    ..voxel_params()
                },
            )
            .is_none()
        );
        let empty = TensorDesc::new(shape![0, 3, 3], DType::F32);
        let request =
            KernelRequest::voxel_brick_trace(&empty, &empty, &limit, voxel_params()).unwrap();
        assert_eq!(request.elements(), 0);
        assert_eq!(request.result_tensor(1).unwrap().shape, shape![0, 3]);
    }

    #[test]
    fn pair_admission_checks_secondary_dtype_rank_layout_and_storage() {
        let packed = TensorDesc::new(shape![2, 3, 3], DType::F32);
        let limit = TensorDesc::new(Shape::SCALAR, DType::F32);
        let request =
            KernelRequest::voxel_brick_trace(&packed, &packed, &limit, voxel_params()).unwrap();
        let f32_only = KernelConstraints {
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        };
        assert!(!f32_only.admits(&request));
        let typed = KernelConstraints {
            output_dtypes: DTypeSet::one(DType::F32).union(DTypeSet::one(DType::I32)),
            max_rank: 3,
            ..f32_only
        };
        assert!(typed.admits(&request));
        let primary = *request.result_tensor(0).unwrap();
        let secondary = *request.result_tensor(1).unwrap();
        let with_secondary = |second| {
            KernelRequest::from_parts(
                request.semantic,
                *request.inputs(),
                KernelResult::TensorPair([primary, second]),
                request.params(),
                request.effects(),
            )
        };
        let strided = with_secondary(secondary.transpose(0, 1).unwrap()).unwrap();
        assert!(!strided.contiguous);
        assert!(!typed.admits(&strided));
        let high_rank = with_secondary(TensorDesc::new(shape![1, 1, 2, 3], DType::I32)).unwrap();
        assert!(!typed.admits(&high_rank));
        let offset_overflow = TensorDesc {
            layout: crate::Layout::from_raw(&[12, 4], usize::MAX),
            ..secondary
        };
        assert!(with_secondary(offset_overflow).is_none());
        let shape_overflow = TensorDesc {
            shape: shape![usize::MAX, 2],
            ..secondary
        };
        assert!(with_secondary(shape_overflow).is_none());

        struct PairCatalog(KernelConstraints);
        impl KernelCatalog for PairCatalog {
            type Implementation = ();
            fn visit_candidates<V: CandidateVisitor<()>>(
                &self,
                request: &KernelRequest,
                visitor: &mut V,
            ) {
                visitor.visit(KernelCandidate {
                    semantic: request.semantic,
                    implementation: (),
                    required_capabilities: CapabilityMask::NONE,
                    constraints: self.0,
                    accuracy: KernelAccuracy::Strict,
                    cost: KernelCost::new(1, 1),
                });
            }
        }
        let policy = KernelPolicy::new(CapabilityMask::NONE);
        assert!(select_kernel(&PairCatalog(f32_only), &request, policy).is_none());
        let plan = select_kernel(&PairCatalog(typed), &request, policy).unwrap();
        assert_eq!(plan.result_count(), 2);
        for index in 0..=2 {
            assert_eq!(plan.result_tensor(index), request.result_tensor(index));
        }
        assert_eq!(plan.params(), request.params());
        assert_eq!(plan.effects(), request.effects());
    }

    #[test]
    fn result_accessors_do_not_invent_tensor_storage_for_scalars_or_aliases() {
        let vector = TensorDesc::new(shape![4], DType::F32);
        let reduce = KernelRequest::reduce(ReduceOp::Sum, &vector);
        assert_eq!(reduce.result_count(), 1);
        assert_eq!(reduce.result_tensor(0), None);
        let matrix = TensorDesc::new(shape![2, 2], DType::F32);
        let gemm = KernelRequest::gemm_accumulate(&matrix, &matrix, &matrix).unwrap();
        assert_eq!(gemm.result_count(), 1);
        assert_eq!(gemm.result_tensor(0), gemm.input(2));
        assert_eq!(gemm.result_tensor(1), None);
    }
}

#[cfg(test)]
mod token_tree_tests {
    use super::TokenTree;
    #[test]
    fn token_tree_rejects_cycles_forests_and_oversize_before_traversal() {
        for parents in [
            &[][..],
            &[0],
            &[TokenTree::ROOT, 1],
            &[TokenTree::ROOT, TokenTree::ROOT],
            &[TokenTree::ROOT, 2, 0],
        ] {
            assert!(TokenTree::new(parents).is_none());
        }
        assert!(TokenTree::new(&[TokenTree::ROOT; TokenTree::MAX_NODES + 1]).is_none());
        let tree = TokenTree::new(&[TokenTree::ROOT, 0, 0, 1, 2, 4]).unwrap();
        let mut path = [0; TokenTree::MAX_NODES];
        assert_eq!(tree.path(5, &mut path), Some(4));
        assert_eq!(&path[..4], &[0, 2, 4, 5]);
        assert!(tree.contains_path(&path[..4]));
        assert!(!tree.contains_path(&[0, 1, 4]));
        assert!(!tree.contains_path(&[1]));
        assert!(!tree.contains_path(&[0, 6]));
        assert_eq!(tree.path(6, &mut path), None);
    }
}
