//! Paged E4M3 key/value cache: E8M0 row scales, appends and grouped-query
//! attention over a borrowed, caller-owned cache resource.
//!
//! The byte layout is `axnn::lowering::KvCacheParams`: pages of keys,
//! values, key scales and value scales, head-major. One (token, head) row of
//! `head_dim` E4M3 bytes decodes to `2^(e - 127) * e4m3(byte)` with its E8M0
//! scale byte `e`; a power-of-two scale only moves the exponent, so decoding
//! is exact whenever finite.

use crate::floats::Float;
use crate::fork_join::{ForkJoin, for_rows};
use crate::fp8::{dequant_e4m3, f32_to_e4m3_rne};
use crate::planner::{typed_input_matches, typed_output_matches};
use crate::{CpuError, CpuImplementation, CpuKernelPlan};
use axnn::DType;
use axnn::lowering::{
    AttentionDecodeParams, CandidateVisitor, CapabilityMask, DTypeSet, KernelAccuracy,
    KernelCandidate, KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelRequest,
    KernelShapePolicy, KvCacheParams, KvCacheStorage, SemanticOp, TokenTree,
};
use core::marker::PhantomData;
use core::simd::Select;
use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::f32x8;
use core::simd::num::{SimdFloat, SimdInt};

/// E8M0 scale byte that marks a row as NaN.
pub const E8M0_NAN: u8 = 0xff;

/// Decode one OCP MX E8M0 scale: `2^(e - 127)`, or NaN for `0xff`.
#[inline]
pub fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        E8M0_NAN => f32::NAN,
        // 2^-127 is the F32 subnormal with only the top mantissa bit set.
        0 => f32::from_bits(0x0040_0000),
        e => f32::from_bits(u32::from(e) << 23),
    }
}

/// The smallest E8M0 scale `2^(e - 127)` with `amax <= 448 * scale`, for a
/// finite, positive `amax`, so no element of the row saturates.
///
/// `amax = m * 2^E` with `1 <= m < 2`: `448 * 2^(E - 8)` (`1.75 * 2^E`)
/// suffices when `m <= 1.75`, otherwise `2^(E - 7)` does. The result never
/// exceeds 247, because `amax < 2^128`.
#[inline]
fn row_scale_exponent(amax: f32) -> u8 {
    let bits = amax.to_bits();
    let field = (bits >> 23) as i32;
    if field == 0 {
        // Below 2^-126, and 448 * 2^-127 exceeds that.
        return 0;
    }
    let exponent = field - 127;
    let step = if bits & 0x7f_ffff <= 0x60_0000 { 8 } else { 7 };
    (exponent - step + 127).clamp(0, 254) as u8
}

/// Encode one row into `codes` and return its E8M0 scale byte.
///
/// Elements are `RNE_e4m3(x / scale)`; dividing by a power of two is exact,
/// so each element rounds once. An all-zero row takes scale byte 0 and keeps
/// its signed zeros. A row with any non-finite value is stored as NaN codes
/// with the NaN scale, so the whole row decodes to NaN.
fn encode_row(row: &[f32], codes: &mut [u8]) -> u8 {
    let mut amax = 0.0_f32;
    let mut finite = true;
    for &value in row {
        finite &= value.is_finite();
        amax = amax.max(value.abs());
    }
    if !finite {
        codes.fill(0x7f);
        return E8M0_NAN;
    }
    let scale = if amax == 0.0 {
        0
    } else {
        row_scale_exponent(amax)
    };
    // `2^(127 - scale)`, a normal F32 for every scale in 0..=247.
    let inverse = f32::from_bits((254 - u32::from(scale)) << 23);
    for (code, &value) in codes.iter_mut().zip(row) {
        *code = f32_to_e4m3_rne(value * inverse);
    }
    scale
}

/// A caller-owned key/value cache borrowed for one invocation.
///
/// Like the raster target, this is a copyable view of an exclusive borrow:
/// the executor hands it to one node at a time, and no kernel keeps a slice
/// of it past its return.
#[derive(Clone, Copy, Debug)]
pub struct KvCacheRef<'a> {
    base: *mut u8,
    params: KvCacheParams,
    tree: Option<TokenTree<'a>>,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl<'a> KvCacheRef<'a> {
    /// A cache over exactly `params.storage_bytes()` bytes.
    pub fn new(bytes: &'a mut [u8], params: KvCacheParams) -> Result<Self, CpuError> {
        if params.storage_bytes() != Some(bytes.len()) {
            return Err(CpuError::ShapeMismatch {
                op: "borrowed key/value cache length",
            });
        }
        Ok(Self {
            base: bytes.as_mut_ptr(),
            params,
            tree: None,
            _borrow: PhantomData,
        })
    }

    /// Node rows occupy consecutive temporary slots; reads follow logical
    /// ancestor positions, never a sibling's physical slot.
    pub const fn with_tree(mut self, tree: TokenTree<'a>) -> Self {
        self.tree = Some(tree);
        self
    }

    pub const fn tree(self) -> Option<TokenTree<'a>> {
        self.tree
    }

    /// Commit one validated path by moving its encoded rows and scale bytes
    /// into the linear prefix. No requantization or whole-cache copy occurs.
    /// Every range is checked before the first write; increasing node indices
    /// ensure that forward compaction cannot overwrite a later source.
    pub fn compact_tree_path(self, start: usize, path: &[u32]) -> Result<(), CpuError> {
        let invalid = || CpuError::ShapeMismatch {
            op: "key/value tree commit",
        };
        let tree = self.tree.ok_or_else(invalid)?;
        let layout = Layout::new(self.params).ok_or_else(invalid)?;
        if !tree.contains_path(path)
            || start
                .checked_add(tree.len())
                .is_none_or(|end| end > self.params.capacity as usize)
        {
            return Err(invalid());
        }
        // SAFETY: this is the sole exclusive cache view, after all users retire.
        let bytes = unsafe { self.bytes_mut() };
        for (destination, &source) in path.iter().enumerate() {
            for head in 0..layout.kv_heads {
                let from = layout.row(start + source as usize, head);
                let to = layout.row(start + destination, head);
                bytes.copy_within(from.key..from.key + layout.head_dim, to.key);
                bytes.copy_within(from.value..from.value + layout.head_dim, to.value);
                bytes[to.key_scale] = bytes[from.key_scale];
                bytes[to.value_scale] = bytes[from.value_scale];
            }
        }
        Ok(())
    }

    pub const fn params(&self) -> KvCacheParams {
        self.params
    }

    fn len(self) -> usize {
        self.params
            .storage_bytes()
            .expect("validated at construction")
    }

    /// The whole cache, for an append.
    ///
    /// # Safety
    /// No other view of this cache is live while the result is. The executor
    /// runs one node at a time, and kernels drop the view before returning.
    unsafe fn bytes_mut(self) -> &'a mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.base, self.len()) }
    }

    /// The whole cache, for a read.
    ///
    /// # Safety
    /// No mutable view of this cache is live while the result is.
    unsafe fn bytes(self) -> &'a [u8] {
        unsafe { core::slice::from_raw_parts(self.base, self.len()) }
    }
}

/// Byte offsets of one (slot, head) row within the cache.
#[derive(Clone, Copy)]
struct RowAt {
    key: usize,
    value: usize,
    key_scale: usize,
    value_scale: usize,
}

#[derive(Clone, Copy)]
struct Layout {
    kv_heads: usize,
    head_dim: usize,
    page_tokens: usize,
    /// `log2(page_tokens)`: valid parameters page by a power of two.
    page_shift: u32,
    page_bytes: usize,
}

impl Layout {
    fn new(params: KvCacheParams) -> Option<Self> {
        Some(Self {
            kv_heads: params.kv_heads as usize,
            head_dim: params.head_dim as usize,
            page_tokens: params.page_tokens as usize,
            page_shift: params.page_tokens.trailing_zeros(),
            page_bytes: params.page_bytes()?,
        })
    }

    #[inline]
    fn row(self, slot: usize, head: usize) -> RowAt {
        let page = slot >> self.page_shift;
        let row = head * self.page_tokens + (slot & (self.page_tokens - 1));
        let rows = self.kv_heads * self.page_tokens;
        let base = page * self.page_bytes;
        let scales = base + 2 * rows * self.head_dim;
        RowAt {
            key: base + row * self.head_dim,
            value: base + (rows + row) * self.head_dim,
            key_scale: scales + row,
            value_scale: scales + rows + row,
        }
    }
}

/// The rows of slots `0..slots` of one head, in slot order.
///
/// Within a page, a head's consecutive slots are consecutive rows, so each
/// step advances every offset by one row; only a page boundary recomputes
/// them.
struct HeadRows {
    layout: Layout,
    head: usize,
    slot: usize,
    slots: usize,
    at: RowAt,
}

impl HeadRows {
    /// Cache head `head`'s rows for slots `first..end`.
    fn new(layout: Layout, head: usize, first: usize, end: usize) -> Self {
        Self {
            layout,
            head,
            slot: first,
            slots: end,
            at: layout.row(first, head),
        }
    }
}

impl Iterator for HeadRows {
    type Item = RowAt;

    #[inline]
    fn next(&mut self) -> Option<RowAt> {
        if self.slot == self.slots {
            return None;
        }
        let at = self.at;
        self.slot += 1;
        let dim = self.layout.head_dim;
        self.at = if self.slot & (self.layout.page_tokens - 1) == 0 {
            self.layout.row(self.slot, self.head)
        } else {
            RowAt {
                key: at.key + dim,
                value: at.value + dim,
                key_scale: at.key_scale + 1,
                value_scale: at.value_scale + 1,
            }
        };
        Some(at)
    }
}

/// Slots every chunk holds, at absolute boundaries: a whole number of
/// blocks, enough work to outweigh its merge. The same for every cache,
/// every fill and every thread count, so every query of a run, and a single
/// step at the same position, divides its slots at the same points, and a
/// query's result never depends on which rows it was computed beside.
const CHUNK_SLOTS: usize = 256;
/// Chunks per cache head computed at once before they merge: a bound on
/// scratch, not on the context. Waves merge in chunk order, so their size
/// never changes a result.
const WAVE_CHUNKS: usize = 32;
/// Query rows that share one pass over the cache.
const QUERY_TILE: usize = 8;
/// Slots decoded together and folded into the softmax at once: one max,
/// one rescale and one vector `exp` per row per block, not per slot.
const SLOT_BLOCK: usize = 16;
/// Query rows per vector: the rows of a tile lie along lanes.
const LANES: usize = 8;
type Lanes = f32x8;

/// Query rows a tile holds, padded to whole vectors.
fn tile_rows(group: usize) -> usize {
    (QUERY_TILE * group).next_multiple_of(LANES)
}

/// F32 elements of one chunk's partial: every tile row's unnormalized
/// output, running maximum and sum, then the chunk's working buffers — a
/// block of decoded keys and of values, the block's weights for every row,
/// and each row's rescale factor.
fn partial_elements(group: usize, dim: usize) -> usize {
    let rows = tile_rows(group);
    rows * dim + 2 * rows + 2 * SLOT_BLOCK * dim + SLOT_BLOCK * rows + rows
}

/// F32 elements of one cache head's queries, transposed so a tile's rows
/// lie along lanes.
fn transposed_elements(group: usize, dim: usize) -> usize {
    dim * tile_rows(group)
}

/// Scratch F32 elements `ATTENTION_DECODE` needs, per cache head: its
/// transposed queries, every tile row's running maximum and sum across
/// waves, and one wave of chunk partials. None of it grows with the
/// context.
pub fn attention_scratch_elements(params: AttentionDecodeParams) -> usize {
    let group = (params.query_heads / params.cache.kv_heads) as usize;
    let dim = params.cache.head_dim as usize;
    let heads = params.cache.kv_heads as usize;
    heads
        * (transposed_elements(group, dim)
            + 2 * tile_rows(group)
            + WAVE_CHUNKS * partial_elements(group, dim))
}

/// Offer the KV cache kernels for their requests.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    let (inputs, output, accuracy, cost) = match (request.semantic, request.params()) {
        (SemanticOp::KvAppend, KernelParams::KvAppend(params))
            if params.storage == KvCacheStorage::E4M3TokenScaleE8M0 =>
        {
            (
                [DType::I32, DType::F32, DType::F32],
                DType::I32,
                // Encoding is integer arithmetic on F32 bits: every executor
                // writes identical pages.
                KernelAccuracy::Strict,
                KernelCost::new(0, 4),
            )
        }
        (SemanticOp::AttentionDecode, KernelParams::AttentionDecode(params))
            if params.cache.storage == KvCacheStorage::E4M3TokenScaleE8M0 =>
        {
            let Some(bytes) = attention_scratch_elements(params).checked_mul(4) else {
                return true;
            };
            (
                [DType::I32, DType::I32, DType::F32],
                DType::F32,
                // Dot products and the online softmax reassociate sums.
                KernelAccuracy::Faithful,
                KernelCost::new(0, 2).with_temporary_bytes(bytes),
            )
        }
        (SemanticOp::KvAppend | SemanticOp::AttentionDecode, _) => return true,
        _ => return false,
    };
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints: KernelConstraints {
            input_dtypes: inputs.map(DTypeSet::one),
            output_dtypes: DTypeSet::one(output),
            max_rank: 3,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(output)
        },
        accuracy,
        cost,
    });
    true
}

/// Check the planned descriptors every cache kernel shares: the scalar
/// implementation, a rank-0 I32 position at input `position_input`, and
/// zero-offset rows.
fn position_and_rows(
    plan: &CpuKernelPlan,
    position_input: usize,
    position: &[i32],
    rows: usize,
) -> Option<(usize, usize)> {
    if plan.implementation != CpuImplementation::Scalar
        || plan.result_count() != 1
        || !typed_input_matches(plan, position_input, DType::I32, position.len())
        || (0..3).any(|index| plan.input(index).is_none_or(|d| d.layout.offset() != 0))
        || plan.result_tensor(0).is_none_or(|d| d.layout.offset() != 0)
    {
        return None;
    }
    let [position] = position else {
        return None;
    };
    Some((usize::try_from(*position).ok()?, rows))
}

/// Encode `keys` and `values`, F32 `[tokens, kv_heads, head_dim]`, into
/// slots `position .. position + tokens` and write the one-word token.
///
/// Every descriptor and the slot range are checked before the cache is
/// touched. The cache stores no length, so the caller advances its position
/// only after the invocation completes.
pub fn invoke_kv_append(
    plan: &CpuKernelPlan,
    cache: KvCacheRef<'_>,
    position: &[i32],
    keys: &[f32],
    values: &[f32],
    token: &mut [i32],
) -> Result<(), CpuError> {
    let mismatch = || CpuError::ShapeMismatch {
        op: "planned key/value cache append",
    };
    let KernelParams::KvAppend(params) = plan.params() else {
        return Err(mismatch());
    };
    let layout = Layout::new(params).ok_or_else(mismatch)?;
    let rows = plan.input_shapes[1].map_or(0, |shape| shape.dim(0));
    let row_elements = layout.kv_heads * layout.head_dim;
    if plan.semantic != SemanticOp::KvAppend
        || cache.params() != params
        || plan.effects() != KernelEffects::WRITES_RESOURCE
        || plan.temporary_bytes != 0
        || plan.output_dtype != DType::I32
        || !typed_input_matches(plan, 1, DType::F32, keys.len())
        || !typed_input_matches(plan, 2, DType::F32, values.len())
        || plan.input_shapes[1] != plan.input_shapes[2]
        || keys.len() != rows * row_elements
        || !typed_output_matches(plan, 0, DType::I32, plan.shape, token.len())
    {
        return Err(mismatch());
    }
    let (start, rows) = position_and_rows(plan, 0, position, rows).ok_or_else(mismatch)?;
    if cache.tree.is_some_and(|tree| tree.len() != rows) {
        return Err(mismatch());
    }
    if start
        .checked_add(rows)
        .is_none_or(|end| end > params.capacity as usize)
    {
        return Err(CpuError::ShapeMismatch {
            op: "key/value cache position out of range",
        });
    }
    // SAFETY: `cache` is the node's exclusive resource for this call.
    let bytes = unsafe { cache.bytes_mut() };
    let dim = layout.head_dim;
    for token_row in 0..rows {
        for head in 0..layout.kv_heads {
            let at = layout.row(start + token_row, head);
            let offset = (token_row * layout.kv_heads + head) * dim;
            bytes[at.key_scale] = encode_row(
                &keys[offset..offset + dim],
                &mut bytes[at.key..at.key + dim],
            );
            bytes[at.value_scale] = encode_row(
                &values[offset..offset + dim],
                &mut bytes[at.value..at.value + dim],
            );
        }
    }
    token[0] = 0;
    Ok(())
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    let (left_chunks, left_tail) = left.as_chunks::<8>();
    let (right_chunks, right_tail) = right.as_chunks::<8>();
    let mut sum = f32x8::splat(0.0);
    for (a, b) in left_chunks.iter().zip(right_chunks) {
        sum += f32x8::from_array(*a) * f32x8::from_array(*b);
    }
    let mut total = sum.reduce_sum();
    for (a, b) in left_tail.iter().zip(right_tail) {
        total += a * b;
    }
    total
}

/// `acc = acc * keep + weight * row`.
#[inline]
fn rescale_accumulate(acc: &mut [f32], keep: f32, weight: f32, row: &[f32]) {
    let (acc_chunks, acc_tail) = acc.as_chunks_mut::<8>();
    let (row_chunks, row_tail) = row.as_chunks::<8>();
    let (keep_lanes, weight_lanes) = (f32x8::splat(keep), f32x8::splat(weight));
    for (a, r) in acc_chunks.iter_mut().zip(row_chunks) {
        *a = (f32x8::from_array(*a) * keep_lanes + weight_lanes * f32x8::from_array(*r)).to_array();
    }
    for (a, r) in acc_tail.iter_mut().zip(row_tail) {
        *a = *a * keep + weight * r;
    }
}

/// `exp` of every lane of a softmax argument: `x <= 0`, `-inf`, or NaN.
///
/// Range-reduced by `ln 2` (split in two for accuracy) and a degree-six
/// polynomial, fused where the target has FMA. The softmax needs no more
/// of the scalar contract than this: `-inf` — a slot nobody can see, or an
/// empty row's first rescale — and anything below `-87` give exactly zero,
/// where a general `exp` would round-trip those lanes through scalar code;
/// NaN stays NaN, and so poisons its row.
#[inline(always)]
fn exp_lanes(x: Lanes) -> Lanes {
    const LOG2E: f32 = core::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    const MAGIC: f32 = 12_582_912.0; // 2^23 + 2^22: rounds to an integer
    const C: [f32; 5] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5];
    let below = x.simd_lt(Lanes::splat(-87.0));
    let n = (x * Lanes::splat(LOG2E) + Lanes::splat(MAGIC)) - Lanes::splat(MAGIC);
    // A NaN or `-inf` lane casts to something finite; `below` clears the
    // latter, and the former's polynomial is NaN.
    let power = Lanes::from_bits(
        (n.cast::<i32>() + core::simd::Simd::splat(127)).cast::<u32>()
            << core::simd::Simd::splat(23),
    );
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on AArch64.
    let p = unsafe {
        use core::arch::aarch64::*;
        let halves = |v: Lanes| {
            let a = v.to_array();
            (vld1q_f32(a.as_ptr()), vld1q_f32(a.as_ptr().add(4)))
        };
        let ((xl, xh), (nl, nh)) = (halves(x), halves(n));
        let poly = |x: float32x4_t, n: float32x4_t| {
            let r = vfmsq_n_f32(x, n, LN2_HI);
            let r = vfmsq_n_f32(r, n, LN2_LO);
            let mut p = vdupq_n_f32(C[0]);
            for &c in &C[1..] {
                p = vfmaq_f32(vdupq_n_f32(c), p, r);
            }
            p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
            vfmaq_f32(vdupq_n_f32(1.0), p, r)
        };
        let mut out = [0.0_f32; LANES];
        vst1q_f32(out.as_mut_ptr(), poly(xl, nl));
        vst1q_f32(out.as_mut_ptr().add(4), poly(xh, nh));
        Lanes::from_array(out)
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // SAFETY: the target guarantees AVX2 and FMA.
    let p = unsafe {
        use core::arch::x86_64::*;
        let load = |v: Lanes| _mm256_loadu_ps(v.to_array().as_ptr());
        let (x, n) = (load(x), load(n));
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
        let mut p = _mm256_set1_ps(C[0]);
        for &c in &C[1..] {
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
        }
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        let mut out = [0.0_f32; LANES];
        _mm256_storeu_ps(out.as_mut_ptr(), p);
        Lanes::from_array(out)
    };
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    let p = {
        let r = x - n * Lanes::splat(LN2_HI) - n * Lanes::splat(LN2_LO);
        let mut p = Lanes::splat(C[0]);
        for &c in &C[1..] {
            p = p * r + Lanes::splat(c);
        }
        p = p * r + Lanes::splat(1.0);
        p * r + Lanes::splat(1.0)
    };
    below.select(Lanes::splat(0.0), p * power)
}

/// One vector of rows' logits against `count` decoded keys: `logits[j]`
/// lane `l` is `Σ_d queries[d][l] · keys[j][d]`, summed over `d` in order.
/// Four keys at a time share each query load, as four independent chains.
#[inline(always)]
fn block_logits_portable(
    queries: &[f32],
    stride: usize,
    keys: &[f32],
    dim: usize,
    count: usize,
    logits: &mut [Lanes; SLOT_BLOCK],
) {
    for first in (0..count).step_by(4) {
        let width = 4.min(count - first);
        let mut sums = [Lanes::splat(0.0); 4];
        for d in 0..dim {
            let query = Lanes::from_slice(&queries[d * stride..][..LANES]);
            for (i, sum) in sums.iter_mut().enumerate().take(width) {
                *sum += query * Lanes::splat(keys[(first + i) * dim + d]);
            }
        }
        logits[first..first + width].copy_from_slice(&sums[..width]);
    }
}

/// One row's output: scaled by `correction`, then each of its `visible`
/// slots' values weighted in slot order. Four vectors of the row stay in
/// registers across the block.
#[inline(always)]
fn block_values_portable(
    output: &mut [f32],
    correction: f32,
    weights: &[f32],
    stride: usize,
    values: &[f32],
    visible: usize,
) {
    let dim = output.len();
    let scale = Lanes::splat(correction);
    let (chunks, tail) = output.as_chunks_mut::<LANES>();
    for (group, quad) in chunks.chunks_mut(4).enumerate() {
        let mut acc = [Lanes::splat(0.0); 4];
        for (a, chunk) in acc.iter_mut().zip(quad.iter()) {
            *a = Lanes::from_array(*chunk) * scale;
        }
        for j in 0..visible {
            let weight = Lanes::splat(weights[j * stride]);
            let row = &values[j * dim + group * 4 * LANES..];
            for (i, a) in acc.iter_mut().enumerate().take(quad.len()) {
                *a += weight * Lanes::from_slice(&row[i * LANES..][..LANES]);
            }
        }
        for (a, chunk) in acc.iter().zip(quad.iter_mut()) {
            *chunk = a.to_array();
        }
    }
    let done = dim - tail.len();
    for (d, value) in tail.iter_mut().enumerate() {
        let mut v = *value * correction;
        for j in 0..visible {
            v += weights[j * stride] * values[j * dim + done + d];
        }
        *value = v;
    }
}

/// [`block_logits_portable`] with fused multiply-adds where the target has
/// them: `core::simd` cannot fuse without `std`, and an unfused multiply and
/// add is half the throughput.
#[inline(always)]
fn block_logits(
    queries: &[f32],
    stride: usize,
    keys: &[f32],
    dim: usize,
    count: usize,
    logits: &mut [Lanes; SLOT_BLOCK],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on AArch64; every load is within the slices'
    // asserted extents (a vector of `stride`-spaced rows, `count` keys).
    unsafe {
        use core::arch::aarch64::*;
        debug_assert!(queries.len() >= (dim - 1) * stride + LANES && keys.len() >= count * dim);
        // Eight keys at a time: sixteen independent chains, what four FMA
        // pipes at a four-cycle latency need in flight. The width is a
        // constant so the chains unroll into registers; a block's ragged
        // end takes the same loop one key at a time.
        //
        // Four dimensions of a key load as one vector and multiply by lane
        // (`fmla v, v, v[l]`): the broadcast is free inside the FMA, where a
        // load-and-replicate spends a SIMD op per key and starves the FMA
        // pipes. Each lane still sums over `d` in order.
        #[inline(always)]
        unsafe fn tile<const W: usize>(
            queries: &[f32],
            stride: usize,
            keys: &[f32],
            dim: usize,
            first: usize,
            logits: &mut [Lanes; SLOT_BLOCK],
        ) {
            // SAFETY: as the caller: NEON present, loads in bounds.
            unsafe {
                let mut low = [vdupq_n_f32(0.0); W];
                let mut high = [vdupq_n_f32(0.0); W];
                let mut d = 0;
                while d + 4 <= dim {
                    let q = queries.as_ptr().add(d * stride);
                    let ql = [
                        vld1q_f32(q),
                        vld1q_f32(q.add(stride)),
                        vld1q_f32(q.add(2 * stride)),
                        vld1q_f32(q.add(3 * stride)),
                    ];
                    let qh = [
                        vld1q_f32(q.add(4)),
                        vld1q_f32(q.add(stride + 4)),
                        vld1q_f32(q.add(2 * stride + 4)),
                        vld1q_f32(q.add(3 * stride + 4)),
                    ];
                    for i in 0..W {
                        let k = vld1q_f32(keys.as_ptr().add((first + i) * dim + d));
                        low[i] = vfmaq_laneq_f32::<0>(low[i], ql[0], k);
                        high[i] = vfmaq_laneq_f32::<0>(high[i], qh[0], k);
                        low[i] = vfmaq_laneq_f32::<1>(low[i], ql[1], k);
                        high[i] = vfmaq_laneq_f32::<1>(high[i], qh[1], k);
                        low[i] = vfmaq_laneq_f32::<2>(low[i], ql[2], k);
                        high[i] = vfmaq_laneq_f32::<2>(high[i], qh[2], k);
                        low[i] = vfmaq_laneq_f32::<3>(low[i], ql[3], k);
                        high[i] = vfmaq_laneq_f32::<3>(high[i], qh[3], k);
                    }
                    d += 4;
                }
                while d < dim {
                    let q = queries.as_ptr().add(d * stride);
                    let (q_low, q_high) = (vld1q_f32(q), vld1q_f32(q.add(4)));
                    for i in 0..W {
                        let k = *keys.get_unchecked((first + i) * dim + d);
                        low[i] = vfmaq_n_f32(low[i], q_low, k);
                        high[i] = vfmaq_n_f32(high[i], q_high, k);
                    }
                    d += 1;
                }
                for i in 0..W {
                    let mut lanes = [0.0_f32; LANES];
                    vst1q_f32(lanes.as_mut_ptr(), low[i]);
                    vst1q_f32(lanes.as_mut_ptr().add(4), high[i]);
                    logits[first + i] = Lanes::from_array(lanes);
                }
            }
        }
        let mut first = 0;
        while first + 8 <= count {
            tile::<8>(queries, stride, keys, dim, first, logits);
            first += 8;
        }
        while first < count {
            tile::<1>(queries, stride, keys, dim, first, logits);
            first += 1;
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // SAFETY: the target guarantees AVX2 and FMA; loads as above.
    unsafe {
        use core::arch::x86_64::*;
        debug_assert!(queries.len() >= (dim - 1) * stride + LANES && keys.len() >= count * dim);
        // Eight keys at a time: eight chains, what two FMA ports at a
        // four-cycle latency need, within sixteen registers; a constant width
        // so they unroll into registers.
        #[inline(always)]
        unsafe fn tile<const W: usize>(
            queries: &[f32],
            stride: usize,
            keys: &[f32],
            dim: usize,
            first: usize,
            logits: &mut [Lanes; SLOT_BLOCK],
        ) {
            // SAFETY: as the caller: NEON/AVX2 present, loads in bounds.
            unsafe {
                let mut sums = [_mm256_setzero_ps(); W];
                for d in 0..dim {
                    let q = _mm256_loadu_ps(queries.as_ptr().add(d * stride));
                    for i in 0..W {
                        let k = _mm256_broadcast_ss(keys.get_unchecked((first + i) * dim + d));
                        sums[i] = _mm256_fmadd_ps(q, k, sums[i]);
                    }
                }
                for i in 0..W {
                    let mut lanes = [0.0_f32; LANES];
                    _mm256_storeu_ps(lanes.as_mut_ptr(), sums[i]);
                    logits[first + i] = Lanes::from_array(lanes);
                }
            }
        }
        let mut first = 0;
        while first + 8 <= count {
            tile::<8>(queries, stride, keys, dim, first, logits);
            first += 8;
        }
        while first < count {
            tile::<1>(queries, stride, keys, dim, first, logits);
            first += 1;
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    block_logits_portable(queries, stride, keys, dim, count, logits);
}

/// [`block_values_portable`] with fused multiply-adds where the target has
/// them.
#[inline(always)]
fn block_values(
    output: &mut [f32],
    correction: f32,
    weights: &[f32],
    stride: usize,
    values: &[f32],
    visible: usize,
) {
    let dim = output.len();
    #[cfg(target_arch = "aarch64")]
    if dim.is_multiple_of(32) {
        // SAFETY: NEON is baseline on AArch64; the output holds `dim`
        // elements in whole 32-element quads, and `values` holds `visible`
        // rows of `dim`.
        unsafe {
            use core::arch::aarch64::*;
            debug_assert!(values.len() >= visible * dim);
            for base in (0..dim).step_by(32) {
                let out = output.as_mut_ptr().add(base);
                let mut acc = [vdupq_n_f32(0.0); 8];
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = vmulq_n_f32(vld1q_f32(out.add(4 * i)), correction);
                }
                for j in 0..visible {
                    let weight = *weights.get_unchecked(j * stride);
                    let row = values.as_ptr().add(j * dim + base);
                    for (i, a) in acc.iter_mut().enumerate() {
                        *a = vfmaq_n_f32(*a, vld1q_f32(row.add(4 * i)), weight);
                    }
                }
                for (i, a) in acc.iter().enumerate() {
                    vst1q_f32(out.add(4 * i), *a);
                }
            }
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if dim.is_multiple_of(32) {
        // SAFETY: the AVX-512 feature is a compile-time CPU contract;
        // `values` has every visible row and each output quad is complete.
        unsafe {
            crate::arch::avx512_attention::block_values(
                output, correction, weights, stride, values, visible,
            );
        }
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    if dim.is_multiple_of(32) {
        // SAFETY: the target guarantees AVX2 and FMA; extents as above.
        unsafe {
            use core::arch::x86_64::*;
            debug_assert!(values.len() >= visible * dim);
            let scale = _mm256_set1_ps(correction);
            for base in (0..dim).step_by(32) {
                let out = output.as_mut_ptr().add(base);
                let mut acc = [_mm256_setzero_ps(); 4];
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_mul_ps(_mm256_loadu_ps(out.add(8 * i)), scale);
                }
                for j in 0..visible {
                    let weight = _mm256_broadcast_ss(weights.get_unchecked(j * stride));
                    let row = values.as_ptr().add(j * dim + base);
                    for (i, a) in acc.iter_mut().enumerate() {
                        *a = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(8 * i)), weight, *a);
                    }
                }
                for (i, a) in acc.iter().enumerate() {
                    _mm256_storeu_ps(out.add(8 * i), *a);
                }
            }
        }
        return;
    }
    block_values_portable(output, correction, weights, stride, values, visible);
}

/// [`block_values`] for two rows that see the same slots: they share every
/// load of a value row, and each keeps its own chains, so each row's sums
/// are exactly what [`block_values`] gives it alone.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block_values_pair(
    first: &mut [f32],
    second: &mut [f32],
    corrections: [f32; 2],
    first_weights: &[f32],
    second_weights: &[f32],
    stride: usize,
    values: &[f32],
    visible: usize,
) {
    let dim = first.len();
    #[cfg(target_arch = "aarch64")]
    if dim.is_multiple_of(32) {
        // SAFETY: NEON is baseline on AArch64; both outputs hold `dim`
        // elements in whole quads, and `values` holds `visible` rows.
        unsafe {
            use core::arch::aarch64::*;
            debug_assert!(second.len() == dim && values.len() >= visible * dim);
            for base in (0..dim).step_by(32) {
                let (a_out, b_out) = (first.as_mut_ptr().add(base), second.as_mut_ptr().add(base));
                let mut a = [vdupq_n_f32(0.0); 8];
                let mut b = [vdupq_n_f32(0.0); 8];
                for i in 0..8 {
                    a[i] = vmulq_n_f32(vld1q_f32(a_out.add(4 * i)), corrections[0]);
                    b[i] = vmulq_n_f32(vld1q_f32(b_out.add(4 * i)), corrections[1]);
                }
                for j in 0..visible {
                    let (wa, wb) = (
                        *first_weights.get_unchecked(j * stride),
                        *second_weights.get_unchecked(j * stride),
                    );
                    let row = values.as_ptr().add(j * dim + base);
                    for i in 0..8 {
                        let v = vld1q_f32(row.add(4 * i));
                        a[i] = vfmaq_n_f32(a[i], v, wa);
                        b[i] = vfmaq_n_f32(b[i], v, wb);
                    }
                }
                for i in 0..8 {
                    vst1q_f32(a_out.add(4 * i), a[i]);
                    vst1q_f32(b_out.add(4 * i), b[i]);
                }
            }
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if dim.is_multiple_of(32) {
        // SAFETY: the AVX-512 feature is a compile-time CPU contract;
        // the two outputs and every visible value row have `dim` elements.
        unsafe {
            crate::arch::avx512_attention::block_values_pair(
                first,
                second,
                corrections,
                first_weights,
                second_weights,
                stride,
                values,
                visible,
            );
        }
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    if dim.is_multiple_of(32) {
        // SAFETY: the target guarantees AVX2 and FMA; extents as above.
        unsafe {
            use core::arch::x86_64::*;
            debug_assert!(second.len() == dim && values.len() >= visible * dim);
            let (sa, sb) = (
                _mm256_set1_ps(corrections[0]),
                _mm256_set1_ps(corrections[1]),
            );
            for base in (0..dim).step_by(32) {
                let (a_out, b_out) = (first.as_mut_ptr().add(base), second.as_mut_ptr().add(base));
                let mut a = [_mm256_setzero_ps(); 4];
                let mut b = [_mm256_setzero_ps(); 4];
                for i in 0..4 {
                    a[i] = _mm256_mul_ps(_mm256_loadu_ps(a_out.add(8 * i)), sa);
                    b[i] = _mm256_mul_ps(_mm256_loadu_ps(b_out.add(8 * i)), sb);
                }
                for j in 0..visible {
                    let wa = _mm256_broadcast_ss(first_weights.get_unchecked(j * stride));
                    let wb = _mm256_broadcast_ss(second_weights.get_unchecked(j * stride));
                    let row = values.as_ptr().add(j * dim + base);
                    for i in 0..4 {
                        let v = _mm256_loadu_ps(row.add(8 * i));
                        a[i] = _mm256_fmadd_ps(v, wa, a[i]);
                        b[i] = _mm256_fmadd_ps(v, wb, b[i]);
                    }
                }
                for i in 0..4 {
                    _mm256_storeu_ps(a_out.add(8 * i), a[i]);
                    _mm256_storeu_ps(b_out.add(8 * i), b[i]);
                }
            }
        }
        return;
    }
    block_values(
        first,
        corrections[0],
        first_weights,
        stride,
        values,
        visible,
    );
    block_values(
        second,
        corrections[1],
        second_weights,
        stride,
        values,
        visible,
    );
}

/// One cache head's chunk for a tile of query rows, into `partial`: each
/// row's unnormalized output, running maximum and sum over the chunk's
/// slots it can see.
///
/// Slots go in blocks of [`SLOT_BLOCK`]. A block's keys and values are
/// decoded once; its logits come for a vector of rows at a time, with the
/// rows along lanes, so the block's maximum, its rescale and its `exp`s are
/// per-lane vector work; then each row's output takes the block's weighted
/// values in slot order. Row `r` sees slot `s` when `s < base + r / group +
/// 1`, which within a block is always a prefix — so each row's arithmetic,
/// lane by lane and slot by slot, is exactly what it would be alone.
#[allow(clippy::too_many_arguments)]
fn attend_chunk(
    layout: Layout,
    bytes: &[u8],
    head: usize,
    span: core::ops::Range<usize>,
    tree_tail: Option<(usize, &[u32])>,
    base: usize,
    group: usize,
    real_rows: usize,
    padded: usize,
    row_capacity: usize,
    queries: &[f32],
    partial: &mut [f32],
) {
    let dim = layout.head_dim;
    let (acc, rest) = partial.split_at_mut(row_capacity * dim);
    let (maxima, rest) = rest.split_at_mut(row_capacity);
    let (sums, rest) = rest.split_at_mut(row_capacity);
    let (keys, rest) = rest.split_at_mut(SLOT_BLOCK * dim);
    let (values, rest) = rest.split_at_mut(SLOT_BLOCK * dim);
    let (weights, rest) = rest.split_at_mut(SLOT_BLOCK * row_capacity);
    let corrections = &mut rest[..row_capacity];
    acc[..real_rows * dim].fill(0.0);
    maxima[..padded].fill(f32::NEG_INFINITY);
    sums[..padded].fill(0.0);
    // The first slot a row cannot see; padding rows see none.
    let limit = |row: usize| {
        if row < real_rows {
            base + row / group + 1
        } else {
            0
        }
    };
    let none = Lanes::splat(f32::NEG_INFINITY);

    let mut first = span.start;
    while first < span.end {
        let count = SLOT_BLOCK.min(span.end - first);
        // Rows decode unscaled. Their E8M0 scales are powers of two, which
        // commute exactly with every rounding: a key's scales its logit and
        // a value's scales its weight, once per slot instead of per element.
        let mut key_scales = [0.0_f32; SLOT_BLOCK];
        let mut value_scales = [0.0_f32; SLOT_BLOCK];
        for (j, at) in HeadRows::new(layout, head, first, first + count).enumerate() {
            // Map only the short speculative tail. Logical slot ordering and
            // chunk boundaries stay identical to independent serial decoding;
            // nonancestor values (including NaNs) are never decoded at all.
            let at = match tree_tail {
                Some((prefix, path)) if first + j >= prefix => {
                    layout.row(prefix + path[first + j - prefix] as usize, head)
                }
                _ => at,
            };
            dequant_e4m3(
                &bytes[at.key..at.key + dim],
                &mut keys[j * dim..(j + 1) * dim],
                dim,
            );
            dequant_e4m3(
                &bytes[at.value..at.value + dim],
                &mut values[j * dim..(j + 1) * dim],
                dim,
            );
            key_scales[j] = e8m0_to_f32(bytes[at.key_scale]);
            value_scales[j] = e8m0_to_f32(bytes[at.value_scale]);
        }
        for vector in 0..padded / LANES {
            let lanes = vector * LANES..(vector + 1) * LANES;
            let limits = core::simd::Simd::<u32, LANES>::from_array(core::array::from_fn(|l| {
                limit(vector * LANES + l) as u32
            }));
            let mut logits = [none; SLOT_BLOCK];
            block_logits(
                &queries[vector * LANES..],
                padded,
                keys,
                dim,
                count,
                &mut logits,
            );
            // A slot a row cannot see, like a `-inf` logit, contributes
            // nothing; a NaN or `+inf` logit stays, and poisons its row.
            let mut block_max = none;
            for (j, logit) in logits.iter_mut().enumerate().take(count) {
                let slot = first + j;
                let seen = limits.simd_gt(core::simd::Simd::splat(slot as u32));
                let scaled = *logit * Lanes::splat(key_scales[j]);
                *logit = (seen & scaled.simd_ne(none)).select(scaled, none);
                block_max = block_max.simd_max(*logit);
            }
            let old = Lanes::from_slice(&maxima[lanes.clone()]);
            let new = old.simd_max(block_max);
            let active = new.simd_ne(none);
            // `exp(-inf)` is zero: a row's first block rescales empty state.
            let correction = active.select(exp_lanes(old - new), Lanes::splat(1.0));
            let mut total = Lanes::from_slice(&sums[lanes.clone()]) * correction;
            for (j, logit) in logits.iter().enumerate().take(count) {
                let weight = active.select(exp_lanes(*logit - new), Lanes::splat(0.0));
                total += weight;
                weights[j * row_capacity + vector * LANES..][..LANES]
                    .copy_from_slice(&(weight * Lanes::splat(value_scales[j])).to_array());
            }
            sums[lanes.clone()].copy_from_slice(&total.to_array());
            maxima[lanes.clone()].copy_from_slice(&active.select(new, old).to_array());
            corrections[lanes].copy_from_slice(&correction.to_array());
        }
        let mut row = 0;
        while row < real_rows {
            let visible = limit(row).saturating_sub(first).min(count);
            // Rows of one token see the same slots: two at a time share
            // every value load.
            if row + 1 < real_rows && limit(row + 1) == limit(row) {
                let (a, b) = acc[row * dim..(row + 2) * dim].split_at_mut(dim);
                block_values_pair(
                    a,
                    b,
                    [corrections[row], corrections[row + 1]],
                    &weights[row..],
                    &weights[row + 1..],
                    row_capacity,
                    values,
                    visible,
                );
                row += 2;
            } else {
                block_values(
                    &mut acc[row * dim..(row + 1) * dim],
                    corrections[row],
                    &weights[row..],
                    row_capacity,
                    values,
                    visible,
                );
                row += 1;
            }
        }
        first += count;
    }
}

/// Grouped-query attention of `queries`, F32 `[tokens, query_heads,
/// head_dim]`, over the cache: query row `i` attends slots
/// `0 ..= position + i`.
///
/// Each cache head's key and value rows are decoded once per slot and shared
/// by every query head in its group. The softmax is online, with F32 running
/// max and sum. A `-inf` logit contributes nothing; a NaN or `+inf` logit,
/// or a row whose logits are all `-inf`, yields NaN, as a direct F32 softmax
/// would.
///
/// Long contexts split, as flash-decoding does: each cache head's slots
/// divide into chunks of [`CHUNK_SLOTS`] at absolute boundaries, each chunk
/// runs the online softmax on its own, on `parallel`'s tasks, a wave of
/// them at a time, and the chunks' maxima, sums and outputs merge in chunk
/// order. One chunk is the plain online softmax.
///
/// Rows share passes, as flash attention's query tiles do: up to eight
/// query rows walk the cache together, so a row of keys and values is read
/// and decoded once for all of them, not once per row. Each row's own
/// arithmetic — which slots, in what order, which chunks merged — is exactly
/// a single step's at its position, so a run of rows gives each row the bits
/// a step would.
#[allow(clippy::too_many_arguments)]
pub fn invoke_attention_decode(
    plan: &CpuKernelPlan,
    cache: KvCacheRef<'_>,
    token: &[i32],
    position: &[i32],
    queries: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    let mismatch = || CpuError::ShapeMismatch {
        op: "planned key/value cache attention",
    };
    let KernelParams::AttentionDecode(params) = plan.params() else {
        return Err(mismatch());
    };
    let layout = Layout::new(params.cache).ok_or_else(mismatch)?;
    let query_heads = params.query_heads as usize;
    let group = query_heads / layout.kv_heads;
    let dim = layout.head_dim;
    let rows = plan.input_shapes[2].map_or(0, |shape| shape.dim(0));
    let scratch_elements = attention_scratch_elements(params);
    if plan.semantic != SemanticOp::AttentionDecode
        || cache.params() != params.cache
        || plan.effects() != KernelEffects::READS_RESOURCE
        || plan.temporary_bytes != scratch_elements * 4
        || scratch.len() < scratch_elements
        || plan.output_dtype != DType::F32
        || !typed_input_matches(plan, 0, DType::I32, token.len())
        || !typed_input_matches(plan, 2, DType::F32, queries.len())
        || queries.len() != rows * query_heads * dim
        || !typed_output_matches(plan, 0, DType::F32, plan.shape, output.len())
    {
        return Err(mismatch());
    }
    let (start, rows) = position_and_rows(plan, 1, position, rows).ok_or_else(mismatch)?;
    if cache.tree.is_some_and(|tree| tree.len() != rows) {
        return Err(mismatch());
    }
    if start
        .checked_add(rows)
        .is_none_or(|end| end > params.cache.capacity as usize)
    {
        return Err(CpuError::ShapeMismatch {
            op: "key/value cache position out of range",
        });
    }
    let logit_scale = params.logit_scale.to_f32();
    // SAFETY: `cache` is the node's exclusive resource for this call, and
    // this kernel only reads it.
    let bytes = unsafe { cache.bytes() };
    let heads = layout.kv_heads;
    let partial = partial_elements(group, dim);
    let row_capacity = tile_rows(group);
    let per_head = transposed_elements(group, dim);
    let width = CHUNK_SLOTS;
    let (transposed, rest) = scratch.split_at_mut(heads * per_head);
    let (running, scratch) = rest.split_at_mut(heads * 2 * row_capacity);

    // A tile of query rows shares each pass over the cache: every key and
    // value row is decoded once for all of them, where a row at a time would
    // stream the whole prefix once per query.
    // A tree tile initially contains one token's query heads. Prefix and
    // tail still use the existing arithmetic; this preserves exact serial
    // chunk/reduction order even when node indices cross a chunk boundary.
    let query_tile = if cache.tree.is_some() { 1 } else { QUERY_TILE };
    for first_row in (0..rows).step_by(query_tile) {
        let tile = query_tile.min(rows - first_row);
        let mut path = [0; TokenTree::MAX_NODES];
        let path_len = match cache.tree {
            Some(tree) => tree.path(first_row, &mut path).ok_or_else(mismatch)?,
            None => 0,
        };
        let tree_tail = cache.tree.map(|_| (start, &path[..path_len]));
        // Query `t` of the tile sees logical slots `0 .. base + t + 1`.
        let base = start
            + if cache.tree.is_some() {
                path_len - 1
            } else {
                first_row
            };
        let last = base + tile;
        let total = last.div_ceil(width);
        // Row `t * group + q` is token `t`'s query head `q` of a cache head.
        let real_rows = tile * group;
        let padded = real_rows.next_multiple_of(LANES);
        // Each cache head's rows onto lanes, with the logit scale folded in.
        for head in 0..heads {
            let lanes = &mut transposed[head * per_head..][..dim * padded];
            lanes.fill(0.0);
            for t in 0..tile {
                for q in 0..group {
                    let row = t * group + q;
                    let source =
                        &queries[((first_row + t) * query_heads + head * group + q) * dim..][..dim];
                    for (d, &value) in source.iter().enumerate() {
                        lanes[d * padded + row] = value * logit_scale;
                    }
                }
            }
        }
        let transposed = &*transposed;
        // Every row's merge state: its output accumulates in place, its
        // running maximum and sum here, across waves.
        for head in 0..heads {
            let state = &mut running[head * 2 * row_capacity..][..2 * row_capacity];
            state[..row_capacity].fill(f32::NEG_INFINITY);
            state[row_capacity..].fill(0.0);
        }
        output[first_row * query_heads * dim..][..tile * query_heads * dim].fill(0.0);

        for wave_first in (0..total).step_by(WAVE_CHUNKS) {
            let wave = WAVE_CHUNKS.min(total - wave_first);
            let partials = &mut scratch[..heads * wave * partial];
            let work = width * (2 * dim + 2 * real_rows * dim);
            for_rows(parallel, partials, partial, work, |first, partials| {
                for (index, partial) in (first..).zip(partials.chunks_exact_mut(partial)) {
                    let (head, chunk) = (index / wave, wave_first + index % wave);
                    let span = chunk * width..((chunk + 1) * width).min(last);
                    attend_chunk(
                        layout,
                        bytes,
                        head,
                        span,
                        tree_tail,
                        base,
                        group,
                        real_rows,
                        padded,
                        row_capacity,
                        &transposed[head * per_head..][..dim * padded],
                        partial,
                    );
                }
            });
            // Merge the wave's chunks into each query's state, in chunk
            // order — only the chunks its own slots reach, as a single step
            // would.
            for t in 0..tile {
                let own = (base + t + 1).div_ceil(width);
                let output =
                    &mut output[(first_row + t) * query_heads * dim..][..query_heads * dim];
                for (head, output) in output.chunks_exact_mut(group * dim).enumerate() {
                    let partials = &partials[head * wave * partial..][..wave * partial];
                    let state = &mut running[head * 2 * row_capacity..][..2 * row_capacity];
                    for (q, output) in output.chunks_exact_mut(dim).enumerate() {
                        let row = t * group + q;
                        let (mut max, mut sum) = (state[row], state[row_capacity + row]);
                        for (index, partial) in partials.chunks_exact(partial).enumerate() {
                            if wave_first + index >= own {
                                break;
                            }
                            let acc = &partial[row * dim..(row + 1) * dim];
                            let chunk_max = partial[row_capacity * dim + row];
                            let chunk_sum = partial[row_capacity * dim + row_capacity + row];
                            if own == 1 {
                                // The online softmax as is.
                                sum = chunk_sum;
                                output.copy_from_slice(acc);
                                break;
                            }
                            if chunk_max == f32::NEG_INFINITY {
                                // Nothing but `-inf` logits contributes
                                // nothing; a NaN among them left the sum
                                // NaN, and so the row.
                                if chunk_sum != 0.0 {
                                    sum = f32::NAN;
                                }
                                continue;
                            }
                            let mut keep = 1.0;
                            if chunk_max > max {
                                keep = (max - chunk_max).exp();
                                sum *= keep;
                                max = chunk_max;
                            }
                            let weight = (chunk_max - max).exp();
                            sum += chunk_sum * weight;
                            rescale_accumulate(output, keep, weight, acc);
                        }
                        state[row] = max;
                        state[row_capacity + row] = sum;
                    }
                }
            }
        }
        // Normalize every row by its sum.
        for t in 0..tile {
            let output = &mut output[(first_row + t) * query_heads * dim..][..query_heads * dim];
            for (head, output) in output.chunks_exact_mut(group * dim).enumerate() {
                let state = &running[head * 2 * row_capacity..][..2 * row_capacity];
                for (q, output) in output.chunks_exact_mut(dim).enumerate() {
                    let inverse = 1.0 / state[row_capacity + t * group + q];
                    for value in output.iter_mut() {
                        *value *= inverse;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::e4m3_to_f32;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use alloc::vec;
    use alloc::vec::Vec;
    use axnn::lowering::{KernelRequest, ResourceKey};
    use axnn::{Shape, TensorDesc};

    const KEY: ResourceKey = ResourceKey {
        group: 0,
        binding: 7,
    };

    fn cache(heads: u32, dim: u32, capacity: u32, page: u32) -> KvCacheParams {
        KvCacheParams::new(
            KEY,
            heads,
            dim,
            capacity,
            page,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap()
    }

    fn desc(dims: &[usize], dtype: DType) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), dtype)
    }

    fn append_plan(params: KvCacheParams, rows: usize) -> CpuKernelPlan {
        let heads = params.kv_heads as usize;
        let dim = params.head_dim as usize;
        let request = KernelRequest::kv_append(
            &desc(&[], DType::I32),
            &desc(&[rows, heads, dim], DType::F32),
            &desc(&[rows, heads, dim], DType::F32),
            params,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    fn attention_plan(params: AttentionDecodeParams, rows: usize) -> CpuKernelPlan {
        let request = KernelRequest::attention_decode(
            &desc(&[1], DType::I32),
            &desc(&[], DType::I32),
            &desc(
                &[
                    rows,
                    params.query_heads as usize,
                    params.cache.head_dim as usize,
                ],
                DType::F32,
            ),
            params,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    /// Deterministic values spanning signs and several binades.
    fn values(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let unit = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                unit * 4.0 * f32::from_bits((state % 7 + 124) << 23)
            })
            .collect()
    }

    /// The page layout is a persisted format: pin it with literal offsets
    /// rather than the kernel's own arithmetic.
    #[test]
    fn row_offsets_follow_the_documented_page_layout() {
        let params = cache(2, 12, 16, 4);
        // 2 heads * 4 slots * (12 key + 12 value + 1 + 1 scale bytes).
        assert_eq!(params.page_bytes(), Some(208));
        let layout = Layout::new(params).unwrap();
        // Slot 5 is row 1 of page 1; head 1 makes it row 5 of 8 per block.
        let at = layout.row(5, 1);
        assert_eq!(
            (at.key, at.value, at.key_scale, at.value_scale),
            (
                208 + 5 * 12,
                208 + 13 * 12,
                208 + 192 + 5,
                208 + 192 + 8 + 5
            )
        );
        let at = layout.row(0, 0);
        assert_eq!(
            (at.key, at.value, at.key_scale, at.value_scale),
            (0, 96, 192, 200)
        );
        // Walking a head's slots across pages lands on the same rows.
        for head in 0..2 {
            let walked: Vec<_> = HeadRows::new(layout, head, 0, 11)
                .map(|at| (at.key, at.value, at.key_scale, at.value_scale))
                .collect();
            let direct: Vec<_> = (0..11)
                .map(|slot| layout.row(slot, head))
                .map(|at| (at.key, at.value, at.key_scale, at.value_scale))
                .collect();
            assert_eq!(walked, direct, "head {head}");
        }
    }

    #[test]
    fn e8m0_scales_decode_every_byte_as_a_power_of_two() {
        for e in 0..=254_u8 {
            assert_eq!(e8m0_to_f32(e) as f64, 2.0_f64.powi(i32::from(e) - 127));
        }
        assert!(e8m0_to_f32(E8M0_NAN).is_nan());
    }

    #[test]
    fn row_scales_are_the_smallest_that_never_saturate() {
        for amax in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0e-30,
            0.4,
            1.0,
            1.75,
            1.750_000_1,
            448.0,
            449.0,
            1.0e30,
            f32::MAX,
        ] {
            let e = row_scale_exponent(amax);
            let limit = |e: u8| 448.0_f64 * 2.0_f64.powi(i32::from(e) - 127);
            assert!(f64::from(amax) <= limit(e), "{amax:e} saturates at {e}");
            assert!(
                e == 0 || f64::from(amax) > limit(e - 1),
                "{amax:e} admits a smaller scale than {e}"
            );
        }
    }

    #[test]
    fn rows_encode_once_within_half_an_e4m3_step() {
        let mut codes = [0_u8; 64];
        for seed in 0..64 {
            let row = values(64, seed);
            let scale = encode_row(&row, &mut codes);
            let factor = e8m0_to_f32(scale);
            for (&value, &code) in row.iter().zip(&codes) {
                let exact = f32_to_e4m3_rne(value / factor);
                assert_eq!(code, exact, "row {seed}");
                let decoded = e4m3_to_f32(code) * factor;
                // E4M3 keeps 4 significant bits (relative step 2^-3); the
                // smallest values may land in its subnormals.
                let bound = value.abs() / 16.0 + factor * 2.0_f32.powi(-10);
                assert!((decoded - value).abs() <= bound, "{value} -> {decoded}");
            }
        }
        let mut zeros = [0.0, -0.0, 0.0];
        assert_eq!(encode_row(&zeros, &mut codes[..3]), 0);
        assert_eq!(codes[..3], [0x00, 0x80, 0x00]);
        zeros[1] = f32::INFINITY;
        assert_eq!(encode_row(&zeros, &mut codes[..3]), E8M0_NAN);
    }

    #[test]
    fn tree_attention_keeps_logical_chunks_and_excludes_poisoned_siblings() {
        let nodes = TokenTree::MAX_NODES;
        let prefix = CHUNK_SLOTS - 3;
        let params = cache(2, 12, (CHUNK_SLOTS * 2) as u32, 16);
        let attention = AttentionDecodeParams::new(params, 6, 0.25).unwrap();
        let mut storage = vec![0; params.storage_bytes().unwrap()];
        let mut token = [0];
        let initial_keys = values(prefix * 24, 71);
        let initial_values = values(prefix * 24, 72);
        invoke_kv_append(
            &append_plan(params, prefix),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[0],
            &initial_keys,
            &initial_values,
            &mut token,
        )
        .unwrap();
        let mut serial = storage.clone();
        let keys = values(nodes * 24, 73);
        let mut node_values = values(nodes * 24, 74);
        node_values[24..48].fill(f32::NAN); // Sibling one must not poison node fifteen.
        let mut parents = [0; TokenTree::MAX_NODES];
        parents[0] = TokenTree::ROOT;
        let tree = TokenTree::new(&parents).unwrap();
        invoke_kv_append(
            &append_plan(params, nodes),
            KvCacheRef::new(&mut storage, params)
                .unwrap()
                .with_tree(tree),
            &[prefix as i32],
            &keys,
            &node_values,
            &mut token,
        )
        .unwrap();
        for (depth, node) in [0, nodes - 1].into_iter().enumerate() {
            invoke_kv_append(
                &append_plan(params, 1),
                KvCacheRef::new(&mut serial, params).unwrap(),
                &[(prefix + depth) as i32],
                &keys[node * 24..][..24],
                &node_values[node * 24..][..24],
                &mut token,
            )
            .unwrap();
        }
        let queries = values(nodes * 72, 75);
        let mut output = vec![0.0; queries.len()];
        let mut expected = vec![0.0; 72];
        let mut scratch = vec![0.0; attention_scratch_elements(attention)];
        invoke_attention_decode(
            &attention_plan(attention, nodes),
            KvCacheRef::new(&mut storage, params)
                .unwrap()
                .with_tree(tree),
            &token,
            &[prefix as i32],
            &queries,
            &mut output,
            &mut scratch,
            &crate::Serial,
        )
        .unwrap();
        invoke_attention_decode(
            &attention_plan(attention, 1),
            KvCacheRef::new(&mut serial, params).unwrap(),
            &token,
            &[(prefix + 1) as i32],
            &queries[(nodes - 1) * 72..],
            &mut expected,
            &mut scratch,
            &crate::Serial,
        )
        .unwrap();
        assert!(expected.iter().all(|v| v.is_finite()));
        assert_eq!(&output[(nodes - 1) * 72..], expected.as_slice());
        let before = storage.clone();
        assert!(
            KvCacheRef::new(&mut storage, params)
                .unwrap()
                .with_tree(tree)
                .compact_tree_path(prefix, &[0, 1, 15])
                .is_err()
        );
        assert_eq!(storage, before, "invalid path must not partially compact");
        KvCacheRef::new(&mut storage, params)
            .unwrap()
            .with_tree(tree)
            .compact_tree_path(prefix, &[0, 15])
            .unwrap();
        let layout = Layout::new(params).unwrap();
        for slot in 0..prefix + 2 {
            for head in 0..2 {
                let at = layout.row(slot, head);
                assert_eq!(&storage[at.key..at.key + 12], &serial[at.key..at.key + 12]);
                assert_eq!(
                    &storage[at.value..at.value + 12],
                    &serial[at.value..at.value + 12]
                );
                assert_eq!(storage[at.key_scale], serial[at.key_scale]);
                assert_eq!(storage[at.value_scale], serial[at.value_scale]);
            }
        }
    }

    /// Append rows in two steps, then attend: compare with an F64 softmax
    /// over the decoded cache, and check the pages hold exactly the
    /// row-encoded bytes at their documented offsets.
    #[test]
    fn append_then_attend_matches_a_direct_softmax_over_decoded_rows() {
        let params = cache(2, 12, 16, 4);
        let attention = AttentionDecodeParams::new(params, 6, 0.25).unwrap();
        let mut storage = vec![0xa5_u8; params.storage_bytes().unwrap()];
        let (heads, dim, group) = (2, 12, 3);
        let first_keys = values(5 * heads * dim, 1);
        let first_values = values(5 * heads * dim, 2);
        let mut token = [9];
        invoke_kv_append(
            &append_plan(params, 5),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[0],
            &first_keys,
            &first_values,
            &mut token,
        )
        .unwrap();
        assert_eq!(token, [0]);
        let keys = values(2 * heads * dim, 3);
        let values_2 = values(2 * heads * dim, 4);
        invoke_kv_append(
            &append_plan(params, 2),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[5],
            &keys,
            &values_2,
            &mut token,
        )
        .unwrap();

        let queries = values(2 * 6 * dim, 5);
        let mut output = vec![f32::NAN; queries.len()];
        let mut scratch = vec![0.0; attention_scratch_elements(attention)];
        invoke_attention_decode(
            &attention_plan(attention, 2),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &token,
            &[5],
            &queries,
            &mut output,
            &mut scratch,
            &crate::Serial,
        )
        .unwrap();

        let all_keys: Vec<f32> = first_keys.iter().chain(&keys).copied().collect();
        let all_values: Vec<f32> = first_values.iter().chain(&values_2).copied().collect();
        let layout = Layout::new(params).unwrap();
        let decoded = |slot: usize, head: usize, value: bool| -> Vec<f64> {
            let at = layout.row(slot, head);
            let (row, scale) = if value {
                (at.value, at.value_scale)
            } else {
                (at.key, at.key_scale)
            };
            let factor = f64::from(e8m0_to_f32(storage[scale]));
            storage[row..row + dim]
                .iter()
                .map(|&code| f64::from(e4m3_to_f32(code)) * factor)
                .collect()
        };
        let mut codes = vec![0; dim];
        for slot in 0..7 {
            for head in 0..heads {
                let offset = (slot * heads + head) * dim;
                let at = layout.row(slot, head);
                let scale = encode_row(&all_keys[offset..offset + dim], &mut codes);
                assert_eq!(storage[at.key..at.key + dim], codes[..]);
                assert_eq!(storage[at.key_scale], scale);
                let scale = encode_row(&all_values[offset..offset + dim], &mut codes);
                assert_eq!(storage[at.value..at.value + dim], codes[..]);
                assert_eq!(storage[at.value_scale], scale);
            }
        }

        for row in 0..2 {
            for query_head in 0..6 {
                let head = query_head / group;
                let offset = (row * 6 + query_head) * dim;
                let query = &queries[offset..offset + dim];
                let logits: Vec<f64> = (0..=5 + row)
                    .map(|slot| {
                        let key = decoded(slot, head, false);
                        0.25 * query
                            .iter()
                            .zip(&key)
                            .map(|(q, k)| f64::from(*q) * k)
                            .sum::<f64>()
                    })
                    .collect();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..dim {
                    let expected: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(slot, w)| w * decoded(slot, head, true)[d])
                        .sum::<f64>()
                        / total;
                    let actual = f64::from(output[offset + d]);
                    // The reference already uses the decoded cache, so
                    // only F32 dot and softmax rounding remain.
                    assert!(
                        (actual - expected).abs() <= 5e-4 * (1.0 + expected.abs()),
                        "row {row} head {query_head} dim {d}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    /// A context long enough to split: the merged chunks match an F64
    /// softmax, every split gives the same bits, and a NaN row in one chunk
    /// poisons only its own cache head's queries.
    /// A run of query rows shares its passes over the cache, yet every row
    /// gets exactly the bits a single step at its position gives: across a
    /// chunk boundary that falls inside a tile, and across tiles.
    #[test]
    fn runs_of_rows_match_single_steps_bit_for_bit() {
        let (heads, dim, group) = (2, 16, 4);
        let params = cache(heads as u32, dim as u32, 1200, 16);
        let attention = AttentionDecodeParams::new(params, (heads * group) as u32, 0.3).unwrap();
        // Eleven rows: two tiles, and slots 250 ..= 260 straddle the
        // 256-slot chunk boundary inside the first.
        let (prefix, run) = (250, 11);
        let total = prefix + run;
        let mut storage = vec![0_u8; params.storage_bytes().unwrap()];
        let keys = values(total * heads * dim, 31);
        let rows = values(total * heads * dim, 32);
        let mut token = [0];
        invoke_kv_append(
            &append_plan(params, total),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[0],
            &keys,
            &rows,
            &mut token,
        )
        .unwrap();
        let queries = values(run * heads * group * dim, 33);
        let per_row = heads * group * dim;
        let attend = |storage: &mut [u8], at: usize, count: usize, queries: &[f32]| {
            let mut output = vec![0.0_f32; count * per_row];
            let mut scratch = vec![0.0; attention_scratch_elements(attention)];
            invoke_attention_decode(
                &attention_plan(attention, count),
                KvCacheRef::new(storage, params).unwrap(),
                &token,
                &[at as i32],
                queries,
                &mut output,
                &mut scratch,
                &crate::fork_join::Reversed(3),
            )
            .unwrap();
            output
        };
        let together = attend(&mut storage, prefix, run, &queries);
        for r in 0..run {
            let alone = attend(
                &mut storage,
                prefix + r,
                1,
                &queries[r * per_row..(r + 1) * per_row],
            );
            assert_eq!(
                &together[r * per_row..(r + 1) * per_row],
                &alone[..],
                "row {r}"
            );
        }
    }

    /// A context past one wave of chunks: the waves merge into the same
    /// softmax a direct F64 one gives, every split gives the same bits, and
    /// a run of rows straddling the waves still matches single steps.
    #[test]
    fn contexts_past_a_wave_merge_across_waves() {
        let (heads, dim, group) = (2, 16, 4);
        let slots = (WAVE_CHUNKS + 2) * CHUNK_SLOTS + 100;
        let params = cache(
            heads as u32,
            dim as u32,
            (slots as u32 + 16).next_multiple_of(16),
            16,
        );
        let attention = AttentionDecodeParams::new(params, (heads * group) as u32, 0.3).unwrap();
        let mut storage = vec![0_u8; params.storage_bytes().unwrap()];
        let keys = values(slots * heads * dim, 41);
        let rows = values(slots * heads * dim, 42);
        let mut token = [0];
        invoke_kv_append(
            &append_plan(params, slots),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[0],
            &keys,
            &rows,
            &mut token,
        )
        .unwrap();
        let per_row = heads * group * dim;
        let run = 5;
        let queries = values(run * per_row, 43);
        let at = slots - run;
        let attend = |storage: &mut [u8],
                      position: usize,
                      count: usize,
                      queries: &[f32],
                      parallel: &dyn ForkJoin| {
            let mut output = vec![0.0_f32; count * per_row];
            let mut scratch = vec![0.0; attention_scratch_elements(attention)];
            invoke_attention_decode(
                &attention_plan(attention, count),
                KvCacheRef::new(storage, params).unwrap(),
                &token,
                &[position as i32],
                queries,
                &mut output,
                &mut scratch,
                parallel,
            )
            .unwrap();
            output
        };
        let together = attend(&mut storage, at, run, &queries, &crate::Serial);
        for width in [3, 8] {
            let split = attend(
                &mut storage,
                at,
                run,
                &queries,
                &crate::fork_join::Reversed(width),
            );
            assert_eq!(split, together, "width {width}");
        }
        for r in 0..run {
            let alone = attend(
                &mut storage,
                at + r,
                1,
                &queries[r * per_row..(r + 1) * per_row],
                &crate::Serial,
            );
            assert_eq!(
                &together[r * per_row..(r + 1) * per_row],
                &alone[..],
                "row {r}"
            );
        }
        // The last row against a direct softmax over the decoded cache.
        let layout = Layout::new(params).unwrap();
        let decoded = |slot: usize, head: usize, value: bool| -> Vec<f64> {
            let at = layout.row(slot, head);
            let (row, scale) = if value {
                (at.value, at.value_scale)
            } else {
                (at.key, at.key_scale)
            };
            let factor = f64::from(e8m0_to_f32(storage[scale]));
            storage[row..row + dim]
                .iter()
                .map(|&code| f64::from(e4m3_to_f32(code)) * factor)
                .collect()
        };
        let r = run - 1;
        for query_head in 0..heads * group {
            let head = query_head / group;
            let query = &queries[r * per_row + query_head * dim..][..dim];
            let logits: Vec<f64> = (0..at + r + 1)
                .map(|slot| {
                    0.3 * query
                        .iter()
                        .zip(decoded(slot, head, false))
                        .map(|(q, k)| f64::from(*q) * k)
                        .sum::<f64>()
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut expected = vec![0.0_f64; dim];
            for (slot, w) in weights.iter().enumerate() {
                for (e, v) in expected.iter_mut().zip(decoded(slot, head, true)) {
                    *e += w * v / total;
                }
            }
            for (d, e) in expected.iter().enumerate() {
                let actual = f64::from(together[r * per_row + query_head * dim + d]);
                assert!(
                    (actual - e).abs() <= 5e-4 * (1.0 + e.abs()),
                    "head {query_head} dim {d}: {actual} vs {e}"
                );
            }
        }
    }

    #[test]
    fn long_contexts_split_into_chunks_that_merge_exactly() {
        let (heads, dim, group, slots) = (2, 16, 4, 1100);
        let params = cache(heads as u32, dim as u32, 1200, 16);
        let attention = AttentionDecodeParams::new(params, (heads * group) as u32, 0.3).unwrap();
        // Chunks of 256 at absolute boundaries: 1100 slots reach five.
        assert_eq!(usize::div_ceil(slots, CHUNK_SLOTS), 5);
        let mut storage = vec![0_u8; params.storage_bytes().unwrap()];
        let keys = values(slots * heads * dim, 21);
        let rows = values(slots * heads * dim, 22);
        let mut token = [0];
        invoke_kv_append(
            &append_plan(params, slots),
            KvCacheRef::new(&mut storage, params).unwrap(),
            &[0],
            &keys,
            &rows,
            &mut token,
        )
        .unwrap();
        let queries = values(heads * group * dim, 23);
        let attend = |storage: &mut [u8], parallel: &dyn ForkJoin| {
            let mut output = vec![0.0_f32; queries.len()];
            let mut scratch = vec![0.0; attention_scratch_elements(attention)];
            invoke_attention_decode(
                &attention_plan(attention, 1),
                KvCacheRef::new(storage, params).unwrap(),
                &token,
                &[slots as i32 - 1],
                &queries,
                &mut output,
                &mut scratch,
                parallel,
            )
            .unwrap();
            output
        };
        let serial = attend(&mut storage, &crate::Serial);
        for width in [2, 5, 8] {
            let split = attend(&mut storage, &crate::fork_join::Reversed(width));
            assert_eq!(split, serial, "width {width}");
        }

        let layout = Layout::new(params).unwrap();
        let decoded = |storage: &[u8], slot: usize, head: usize, value: bool| -> Vec<f64> {
            let at = layout.row(slot, head);
            let (row, scale) = if value {
                (at.value, at.value_scale)
            } else {
                (at.key, at.key_scale)
            };
            let factor = f64::from(e8m0_to_f32(storage[scale]));
            storage[row..row + dim]
                .iter()
                .map(|&c| f64::from(e4m3_to_f32(c)) * factor)
                .collect()
        };
        for query_head in 0..heads * group {
            let head = query_head / group;
            let query = &queries[query_head * dim..(query_head + 1) * dim];
            let logits: Vec<f64> = (0..slots)
                .map(|slot| {
                    let key = decoded(&storage, slot, head, false);
                    0.3 * query
                        .iter()
                        .zip(&key)
                        .map(|(q, k)| f64::from(*q) * k)
                        .sum::<f64>()
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut expected = vec![0.0_f64; dim];
            for (slot, w) in weights.iter().enumerate() {
                for (e, v) in expected.iter_mut().zip(decoded(&storage, slot, head, true)) {
                    *e += w * v / total;
                }
            }
            for (d, e) in expected.iter().enumerate() {
                let actual = f64::from(serial[query_head * dim + d]);
                assert!(
                    (actual - e).abs() <= 5e-4 * (1.0 + e.abs()),
                    "head {query_head} dim {d}"
                );
            }
        }

        // A NaN key scale in the third chunk of cache head 1.
        let at = layout.row(600, 1);
        storage[at.key_scale] = E8M0_NAN;
        let poisoned = attend(&mut storage, &crate::fork_join::Reversed(8));
        for (i, value) in poisoned.iter().enumerate() {
            assert_eq!(value.is_nan(), i / (group * dim) == 1, "element {i}");
        }
    }

    #[test]
    fn out_of_range_positions_fail_before_touching_the_cache() {
        let params = cache(1, 8, 8, 4);
        let mut storage = vec![0x5a_u8; params.storage_bytes().unwrap()];
        let rows = values(3 * 8, 9);
        let mut token = [7];
        for position in [-1, 6, i32::MAX] {
            assert!(
                invoke_kv_append(
                    &append_plan(params, 3),
                    KvCacheRef::new(&mut storage, params).unwrap(),
                    &[position],
                    &rows,
                    &rows,
                    &mut token,
                )
                .is_err()
            );
        }
        assert!(storage.iter().all(|&byte| byte == 0x5a));
        assert_eq!(token, [7]);
        assert!(KvCacheRef::new(&mut storage[1..], params).is_err());
    }

    #[test]
    fn non_finite_logits_follow_a_direct_softmax() {
        let params = cache(1, 8, 4, 4);
        let attention = AttentionDecodeParams::new(params, 1, 1.0).unwrap();
        let mut storage = vec![0_u8; params.storage_bytes().unwrap()];
        let mut keys = vec![1.0_f32; 3 * 8];
        let values = values(3 * 8, 11);
        let mut token = [0];
        let mut run = |keys: &[f32], storage: &mut [u8]| {
            invoke_kv_append(
                &append_plan(params, 3),
                KvCacheRef::new(storage, params).unwrap(),
                &[0],
                keys,
                &values,
                &mut token,
            )
            .unwrap();
            let mut output = [0.0_f32; 8];
            let mut scratch = vec![0.0; attention_scratch_elements(attention)];
            invoke_attention_decode(
                &attention_plan(attention, 1),
                KvCacheRef::new(storage, params).unwrap(),
                &token,
                &[2],
                &[1.0; 8],
                &mut output,
                &mut scratch,
                &crate::Serial,
            )
            .unwrap();
            output
        };
        assert!(run(&keys, &mut storage).iter().all(|v| v.is_finite()));
        // A NaN key row decodes to NaN and poisons the output.
        keys[8] = f32::NAN;
        assert!(run(&keys, &mut storage).iter().all(|v| v.is_nan()));
    }
}
