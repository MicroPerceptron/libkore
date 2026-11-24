//! Typed attention key/value cache nodes and their portable attribute schema.
//!
//! A cache is one layer's bound resource: graphs store its binding key,
//! geometry and storage encoding only, never its contents. The cache has no
//! length of its own. Both nodes take the slot position as an explicit I32
//! input, so the caller that owns decode state advances it only after an
//! invocation completes, and re-running a failed step overwrites the same
//! slots.
//!
//! `KV_APPEND` writes `n` tokens at slots `position .. position + n` and
//! returns a one-word token. `ATTENTION_DECODE` consumes that token, which
//! orders the read after the write, and attends query row `i` over slots
//! `0 ..= position + i`. Both carry `ORDERED`, so the scheduler also keeps
//! every other ordered effect, including a later append to the same cache,
//! in node order.

use core::alloc::Allocator;

pub use axnn::lowering::{
    AttentionDecodeParams, F32Bits, KvCacheLayout, KvCacheParams, KvCacheStorage, ResourceKey,
};

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};

/// Version 1 attribute record of a `KV_APPEND` node.
pub const KV_APPEND_ATTRIBUTE_BYTES: usize = 28;
/// Version 1 attribute record of an `ATTENTION_DECODE` node.
pub const ATTENTION_DECODE_ATTRIBUTE_BYTES: usize = 36;

/// The exact node flags of `KV_APPEND`: an ordered write that never reads
/// cache contents.
pub const KV_APPEND_NODE_FLAGS: NodeFlags =
    NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::WRITES_RESOURCE.0);
/// The exact node flags of `ATTENTION_DECODE`: an ordered cache read.
pub const ATTENTION_DECODE_NODE_FLAGS: NodeFlags =
    NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::READS_RESOURCE.0);

/// End of the header and the five cache fields both records share.
const CACHE_FIELDS_END: usize = 24;
const KV_CACHE_KIND: u8 = 5;
const ATTENTION_DECODE_KIND: u8 = 6;

/// Model-neutral packing of a lifecycle group of layer caches into backing
/// transfer segments. Execution pages and persistent frames remain separate
/// so Kore can use 1 GiB mappings while DAGFS transfers NVMe-sized batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvBackingLayout {
    pub cache: KvCacheLayout,
    pub layers: u32,
    pub segment_tokens: u32,
    pub frame_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCapacityPlan {
    pub compute_pages_per_layer: u32,
    pub residency_segments: u32,
    pub cache_bytes: u64,
    pub packed_frames: u64,
    pub transfer_frames: u64,
    pub transfer_padding_bytes: u64,
    pub dense_f32_resident_bytes: u64,
}

impl KvBackingLayout {
    pub fn new(
        cache: KvCacheLayout,
        layers: u32,
        segment_tokens: u32,
        frame_bytes: u64,
    ) -> Option<Self> {
        let layout = Self {
            cache,
            layers,
            segment_tokens,
            frame_bytes,
        };
        layout.is_valid().then_some(layout)
    }

    pub fn is_valid(self) -> bool {
        self.cache.is_valid()
            && self.layers != 0
            && self.segment_tokens != 0
            && self.segment_tokens <= KvCacheLayout::MAX_CAPACITY_TOKENS
            && self.segment_tokens.is_multiple_of(self.cache.page_tokens)
            && self.frame_bytes.is_power_of_two()
            && self.bytes_per_token().is_some()
            && self.segment_bytes().is_some()
    }

    pub fn bytes_per_token(self) -> Option<u64> {
        self.cache
            .bytes_per_token()?
            .checked_mul(u64::from(self.layers))
    }

    pub fn page_group_bytes(self) -> Option<u64> {
        self.cache.page_bytes()?.checked_mul(u64::from(self.layers))
    }

    pub fn segment_bytes(self) -> Option<u64> {
        self.bytes_per_token()?
            .checked_mul(u64::from(self.segment_tokens))
    }

    pub fn segment_frames(self) -> Option<u64> {
        Some(self.segment_bytes()?.div_ceil(self.frame_bytes))
    }

    pub fn plan(self, capacity: u32) -> Option<KvCapacityPlan> {
        if !self.is_valid() {
            return None;
        }
        let compute_pages_per_layer = self.cache.page_count(capacity)?;
        let per_layer_bytes = self.cache.capacity_bytes(capacity)?;
        let cache_bytes = per_layer_bytes.checked_mul(u64::from(self.layers))?;
        let packed_frames = cache_bytes.div_ceil(self.frame_bytes);
        let full_segments = capacity / self.segment_tokens;
        let tail_tokens = capacity % self.segment_tokens;
        let tail_bytes = self
            .bytes_per_token()?
            .checked_mul(u64::from(tail_tokens))?;
        let transfer_frames = u64::from(full_segments)
            .checked_mul(self.segment_frames()?)?
            .checked_add(if tail_bytes == 0 {
                0
            } else {
                tail_bytes.div_ceil(self.frame_bytes)
            })?;
        let dense_f32_resident_bytes = self
            .cache
            .dense_f32_capacity_bytes(capacity)?
            .checked_mul(u64::from(self.layers))?;
        Some(KvCapacityPlan {
            compute_pages_per_layer,
            residency_segments: capacity.div_ceil(self.segment_tokens),
            cache_bytes,
            packed_frames,
            transfer_frames,
            transfer_padding_bytes: transfer_frames
                .checked_mul(self.frame_bytes)?
                .checked_sub(cache_bytes)?,
            dense_f32_resident_bytes,
        })
    }
}

const fn storage_tag(storage: KvCacheStorage) -> u8 {
    match storage {
        KvCacheStorage::E4M3TokenScaleE8M0 => 0,
        KvCacheStorage::E4M3kE2M1vTokenScaleE8M0 => 1,
    }
}

const fn storage_from_tag(tag: u8) -> Option<KvCacheStorage> {
    match tag {
        0 => Some(KvCacheStorage::E4M3TokenScaleE8M0),
        1 => Some(KvCacheStorage::E4M3kE2M1vTokenScaleE8M0),
        _ => None,
    }
}

/// Write the shared header and cache fields, then `page_tokens` after them.
fn encode_cache(params: KvCacheParams, kind: u8, bytes: &mut [u8]) {
    bytes[0] = 1; // schema version
    bytes[1] = kind;
    bytes[2] = storage_tag(params.storage); // byte 3 is reserved zero
    for (slot, value) in bytes[4..CACHE_FIELDS_END + 4].chunks_exact_mut(4).zip([
        params.resource.group,
        params.resource.binding,
        params.kv_heads,
        params.head_dim,
        params.capacity,
        params.page_tokens,
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Read the shared header and cache fields written by [`encode_cache`].
fn decode_cache(bytes: &[u8], kind: u8) -> Option<KvCacheParams> {
    if bytes[0] != 1 || bytes[1] != kind || bytes[3] != 0 {
        return None;
    }
    KvCacheParams::new(
        ResourceKey {
            group: read_u32(bytes, 4)?,
            binding: read_u32(bytes, 8)?,
        },
        read_u32(bytes, 12)?,
        read_u32(bytes, 16)?,
        read_u32(bytes, 20)?,
        read_u32(bytes, CACHE_FIELDS_END)?,
        storage_from_tag(bytes[2])?,
    )
}

/// Encode a checked cache identity, geometry and storage encoding.
///
/// Layout: `[1, 5, storage, 0]`, then LE u32 group, binding, kv_heads,
/// head_dim, capacity and page_tokens.
pub fn encode_kv_append_attributes(
    params: KvCacheParams,
) -> Option<[u8; KV_APPEND_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0_u8; KV_APPEND_ATTRIBUTE_BYTES];
    encode_cache(params, KV_CACHE_KIND, &mut bytes);
    Some(bytes)
}

/// Decode only the exact version 1 `KV_APPEND` record.
///
/// Unknown versions, kinds or storage tags, nonzero reserved bytes, invalid
/// geometry, and truncated or extended records are rejected.
pub fn decode_kv_append_attributes(bytes: &[u8]) -> Option<KvCacheParams> {
    if bytes.len() != KV_APPEND_ATTRIBUTE_BYTES {
        return None;
    }
    decode_cache(bytes, KV_CACHE_KIND)
}

/// Encode a checked grouped-query cache read.
///
/// Layout: `[1, 6, storage, 0]`, then LE u32 group, binding, kv_heads,
/// head_dim, capacity, page_tokens, query_heads and the logit scale's F32
/// bits.
pub fn encode_attention_decode_attributes(
    params: AttentionDecodeParams,
) -> Option<[u8; ATTENTION_DECODE_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0_u8; ATTENTION_DECODE_ATTRIBUTE_BYTES];
    encode_cache(params.cache, ATTENTION_DECODE_KIND, &mut bytes);
    bytes[28..32].copy_from_slice(&params.query_heads.to_le_bytes());
    bytes[32..36].copy_from_slice(&params.logit_scale.bits().to_le_bytes());
    Some(bytes)
}

/// Decode only the exact version 1 `ATTENTION_DECODE` record.
pub fn decode_attention_decode_attributes(bytes: &[u8]) -> Option<AttentionDecodeParams> {
    if bytes.len() != ATTENTION_DECODE_ATTRIBUTE_BYTES {
        return None;
    }
    let cache = decode_cache(bytes, ATTENTION_DECODE_KIND)?;
    AttentionDecodeParams::new(
        cache,
        read_u32(bytes, 28)?,
        F32Bits::from_bits(read_u32(bytes, 32)?).to_f32(),
    )
}

fn validate_rows(tokens: u32, heads: u32, head_dim: u32) -> Result<[u32; 3], AxError> {
    let dimensions = [tokens, heads, head_dim];
    if tokens == 0 || dimensions.contains(&DYNAMIC_DIM) {
        return Err(AxError::InvalidTensorRange);
    }
    dimensions
        .iter()
        .try_fold(core::mem::size_of::<f32>(), |bytes, &dimension| {
            bytes.checked_mul(usize::try_from(dimension).ok()?)
        })
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(AxError::CountOverflow)?;
    Ok(dimensions)
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Append `keys` and `values`, both F32 `[tokens, kv_heads, head_dim]`,
    /// at slots `position .. position + tokens` of the bound cache.
    ///
    /// `position` is a rank-0 I32 input. Returns the append's one-word I32
    /// token, which the layer's [`GraphBuilder::attention_decode`] consumes.
    /// The semantic bridge validates every input descriptor.
    pub fn kv_append(
        &mut self,
        position: TensorId,
        keys: TensorId,
        values: TensorId,
        params: KvCacheParams,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_kv_append_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::KV_APPEND))?;
        let token = self.tensor(DType::I32, TensorFlags::default(), &[1], &[])?;
        self.node(
            OpCode::KV_APPEND,
            KV_APPEND_NODE_FLAGS,
            &[position, keys, values],
            &[token],
            &attributes,
        )?;
        Ok(token)
    }

    /// Attend F32 `queries` `[tokens, query_heads, head_dim]` over the bound
    /// cache after the append that produced `token`.
    ///
    /// Query row `i` reads slots `0 ..= position + i`. Returns the F32
    /// `[tokens, query_heads, head_dim]` attention output, created with
    /// `output_flags` (mark it `OUTPUT` when the graph returns it).
    pub fn attention_decode(
        &mut self,
        token: TensorId,
        position: TensorId,
        queries: TensorId,
        tokens: u32,
        params: AttentionDecodeParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let dimensions = validate_rows(tokens, params.query_heads, params.cache.head_dim)?;
        let attributes = encode_attention_decode_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::ATTENTION_DECODE))?;
        let output = self.tensor(DType::F32, output_flags, &dimensions, &[])?;
        self.node(
            OpCode::ATTENTION_DECODE,
            ATTENTION_DECODE_NODE_FLAGS,
            &[token, position, queries],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScheduleScratch, analyze};
    use alloc::alloc::Global;
    use alloc::vec;

    const KEY: ResourceKey = ResourceKey {
        group: 3,
        binding: 0x0102_0304,
    };

    fn cache() -> KvCacheParams {
        KvCacheParams::new(KEY, 2, 128, 64, 16, KvCacheStorage::E4M3TokenScaleE8M0).unwrap()
    }

    fn attention() -> AttentionDecodeParams {
        AttentionDecodeParams::new(cache(), 8, 0.125).unwrap()
    }

    #[test]
    fn fp4_value_storage_round_trips_through_its_tag() {
        let fp4 = KvCacheParams::new(
            KEY,
            2,
            128,
            64,
            16,
            KvCacheStorage::E4M3kE2M1vTokenScaleE8M0,
        )
        .unwrap();
        let append = encode_kv_append_attributes(fp4).unwrap();
        assert_eq!(append[2], 1, "fp4 value storage tag");
        assert_eq!(decode_kv_append_attributes(&append), Some(fp4));
        assert!(decode_kv_append_attributes(&[1, 5, 2, 0]).is_none(), "unknown tag");
    }

    #[test]
    fn cache_geometry_is_bounded_and_pages_are_packed() {
        let new = |heads, dim, capacity, page| {
            KvCacheParams::new(
                KEY,
                heads,
                dim,
                capacity,
                page,
                KvCacheStorage::E4M3TokenScaleE8M0,
            )
        };
        // Per head and token: 128 E4M3 bytes plus one E8M0 scale byte, for
        // keys and for values.
        assert_eq!(cache().page_bytes(), Some(2 * 16 * (128 + 1) * 2));
        assert_eq!(cache().page_count(), Some(4));
        assert_eq!(cache().storage_bytes(), Some(4 * 2 * 16 * (128 + 1) * 2));
        assert!(new(0, 128, 64, 16).is_none());
        assert!(new(2, 0, 64, 16).is_none());
        assert!(new(2, 128, 0, 16).is_none());
        assert!(new(2, 128, 64, 0).is_none());
        assert!(new(2, 128, 64, 12).is_none(), "pages are powers of two");
        assert!(new(2, 128, 72, 16).is_none(), "whole pages only");
        assert!(new(2, 128, 1, 1).is_some());
        assert!(
            new(1, 1, 1 << 31, 1 << 31).is_none(),
            "I32-addressable slots"
        );
        assert!(
            new(u32::MAX, u32::MAX, 1 << 30, 1 << 30).is_none(),
            "checked bytes"
        );
    }

    #[test]
    fn attention_requires_grouped_heads_and_a_positive_finite_scale() {
        let new = |heads, scale| AttentionDecodeParams::new(cache(), heads, scale);
        assert!(new(2, 1.0).is_some());
        assert!(new(8, 1.0).is_some());
        assert!(new(0, 1.0).is_none());
        assert!(
            new(3, 1.0).is_none(),
            "query heads group evenly over cache heads"
        );
        for scale in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(new(8, scale).is_none());
        }
    }

    #[test]
    fn kv_schemas_are_canonical_and_reject_every_malformed_record() {
        let append = encode_kv_append_attributes(cache()).unwrap();
        assert_eq!(
            append,
            [
                1, 5, 0, 0, 3, 0, 0, 0, 4, 3, 2, 1, 2, 0, 0, 0, 128, 0, 0, 0, 64, 0, 0, 0, 16, 0,
                0, 0
            ]
        );
        assert_eq!(decode_kv_append_attributes(&append), Some(cache()));
        let read = encode_attention_decode_attributes(attention()).unwrap();
        assert_eq!(read[..4], [1, 6, 0, 0]);
        assert_eq!(read[4..28], append[4..]);
        assert_eq!(read[28..32], 8_u32.to_le_bytes());
        assert_eq!(read[32..], 0.125_f32.to_bits().to_le_bytes());
        assert_eq!(decode_attention_decode_attributes(&read), Some(attention()));

        // The two records never decode as each other.
        assert!(decode_kv_append_attributes(&read).is_none());
        assert!(decode_attention_decode_attributes(&append).is_none());
        for length in 0..append.len() {
            assert!(decode_kv_append_attributes(&append[..length]).is_none());
        }
        for length in 0..read.len() {
            assert!(decode_attention_decode_attributes(&read[..length]).is_none());
        }
        let mut extended = vec![0; append.len() + 1];
        extended[..append.len()].copy_from_slice(&append);
        assert!(decode_kv_append_attributes(&extended).is_none());
        // Version, kind and the reserved byte never decode when corrupted.
        for offset in [0, 1, 3] {
            let mut invalid = append;
            invalid[offset] ^= 1;
            assert!(decode_kv_append_attributes(&invalid).is_none());
            let mut invalid = read;
            invalid[offset] ^= 1;
            assert!(decode_attention_decode_attributes(&invalid).is_none());
        }
        // The storage tag has a second live value: flipping it names the
        // fp4-value storage, never the original; a still-unknown tag stays
        // rejected.
        let mut fp4 = append;
        fp4[2] = 1;
        assert_eq!(
            decode_kv_append_attributes(&fp4).map(|p| p.storage),
            Some(KvCacheStorage::E4M3kE2M1vTokenScaleE8M0)
        );
        let mut unknown = append;
        unknown[2] = 2;
        assert!(decode_kv_append_attributes(&unknown).is_none());
        // Geometry the params constructor rejects stays rejected on decode.
        let mut invalid = append;
        invalid[24] = 12;
        assert!(decode_kv_append_attributes(&invalid).is_none());
        let mut invalid = read;
        invalid[28] = 3;
        assert!(decode_attention_decode_attributes(&invalid).is_none());
        let mut invalid = read;
        invalid[32..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(decode_attention_decode_attributes(&invalid).is_none());
    }

    /// A read is admitted only directly after its own append: same cache,
    /// same position tensor, as many rows as it wrote.
    #[test]
    #[cfg(feature = "kernel-lowering")]
    fn attention_is_admitted_only_after_its_own_append() {
        let build = |append_rows: u32, forged_token: bool, other_cache: bool| {
            let mut builder = GraphBuilder::new_in(Global);
            let position = builder
                .tensor(DType::I32, TensorFlags::INPUT, &[], &[])
                .unwrap();
            let rows = [append_rows, 2, 128];
            let keys = builder
                .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
                .unwrap();
            let values = builder
                .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
                .unwrap();
            let queries = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[1, 8, 128], &[])
                .unwrap();
            let append_cache = if other_cache {
                KvCacheParams {
                    resource: ResourceKey {
                        group: 3,
                        binding: 9,
                    },
                    ..cache()
                }
            } else {
                cache()
            };
            let mut token = builder
                .kv_append(position, keys, values, append_cache)
                .unwrap();
            if forged_token {
                token = builder
                    .tensor(DType::I32, TensorFlags::INPUT, &[1], &[])
                    .unwrap();
            }
            builder
                .attention_decode(
                    token,
                    position,
                    queries,
                    1,
                    attention(),
                    TensorFlags::OUTPUT,
                )
                .unwrap();
            builder.finish().unwrap()
        };
        let admitted = |bytes: crate::GraphBuf<Global>| {
            let graph = bytes.parse().unwrap();
            let read = (0..graph.node_count())
                .map(|index| graph.node(index).unwrap())
                .find(|node| node.op == OpCode::ATTENTION_DECODE)
                .unwrap();
            crate::node_kernel_request(&graph, read).is_some()
        };
        assert!(admitted(build(1, false, false)));
        assert!(!admitted(build(2, false, false)), "rows differ");
        assert!(!admitted(build(1, true, false)), "token not from an append");
        assert!(
            !admitted(build(1, false, true)),
            "append wrote another cache"
        );
    }

    #[test]
    fn a_layer_serializes_ordered_cache_effects_at_minor_four() {
        let mut builder = GraphBuilder::new_in(Global);
        let position = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[], &[])
            .unwrap();
        let rows = [1, 2, 128];
        let keys = builder
            .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
            .unwrap();
        let values = builder
            .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
            .unwrap();
        let queries = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 8, 128], &[])
            .unwrap();
        let token = builder.kv_append(position, keys, values, cache()).unwrap();
        let output = builder
            .attention_decode(
                token,
                position,
                queries,
                1,
                attention(),
                TensorFlags::OUTPUT,
            )
            .unwrap();
        assert!(
            builder
                .attention_decode(
                    token,
                    position,
                    queries,
                    0,
                    attention(),
                    TensorFlags::default()
                )
                .is_err()
        );
        let bytes = builder.finish().unwrap();
        assert_eq!(&bytes.as_bytes()[4..8], &[1, 0, 4, 0]);
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.node_count(), 2);

        let append = graph.node(0).unwrap();
        assert_eq!(append.op, OpCode::KV_APPEND);
        assert_eq!(append.flags, KV_APPEND_NODE_FLAGS);
        assert_eq!(
            decode_kv_append_attributes(append.attributes()),
            Some(cache())
        );
        let read = graph.node(1).unwrap();
        assert_eq!(read.op, OpCode::ATTENTION_DECODE);
        assert_eq!(read.flags, ATTENTION_DECODE_NODE_FLAGS);
        assert!(read.inputs().eq([token, position, queries]));
        assert_eq!(
            decode_attention_decode_attributes(read.attributes()),
            Some(attention())
        );
        assert!(!append.cacheable() && !read.cacheable());
        let result = graph.tensor(output).unwrap();
        assert_eq!(result.dtype, DType::F32);
        assert!(result.dimensions().eq([1, 8, 128]));

        let mut scratch = ScheduleScratch {
            adjacency: &mut [0; 4],
            producers: &mut [0; 8],
            indegree: &mut [0; 2],
            queue: &mut [0; 2],
            order: &mut [0; 2],
        };
        analyze(&graph, &mut scratch).unwrap();
        assert_eq!(scratch.order, &[0, 1], "the read follows the append");

        #[cfg(feature = "kernel-lowering")]
        {
            use axnn::lowering::{KernelEffects, KernelParams, SemanticOp};
            let request = crate::node_kernel_request(&graph, append).unwrap();
            assert_eq!(request.semantic, SemanticOp::KvAppend);
            assert_eq!(request.effects(), KernelEffects::WRITES_RESOURCE);
            assert_eq!(request.params(), KernelParams::KvAppend(cache()));
            let request = crate::node_kernel_request(&graph, read).unwrap();
            assert_eq!(request.semantic, SemanticOp::AttentionDecode);
            assert_eq!(request.effects(), KernelEffects::READS_RESOURCE);
            assert_eq!(request.params(), KernelParams::AttentionDecode(attention()));
        }
    }
}
