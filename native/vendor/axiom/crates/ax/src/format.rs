//! Canonical `.ax` binary container.
//!
//! Version 1 is deliberately simple: fixed-size tensor/node tables followed
//! by flat reference, dimension, attribute, and constant sections. All
//! integers are little-endian and all indexes are bounds-checked.

use axnn::DType;

pub const MAGIC: [u8; 4] = *b"AXIR";
pub const VERSION_MAJOR: u16 = 1;
/// Latest minor version accepted by this reader. Encoders use the oldest
/// version required by a graph, preserving ordinary 1.0 artifacts unchanged.
pub const VERSION_MINOR: u16 = 23;
pub const HEADER_BYTES: usize = 64;
/// Dimension value whose concrete extent is supplied at specialization time.
pub const DYNAMIC_DIM: u32 = u32::MAX;
const TENSOR_BYTES: usize = 16;
const NODE_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCode(pub u16);

impl OpCode {
    pub const IDENTITY: Self = Self(0);
    pub const CAST: Self = Self(1);
    pub const ADD: Self = Self(2);
    pub const SUB: Self = Self(3);
    pub const MUL: Self = Self(4);
    pub const DIV: Self = Self(5);
    pub const MATMUL: Self = Self(6);
    pub const CONV1D: Self = Self(7);
    pub const REDUCE_SUM: Self = Self(8);
    pub const REDUCE_MAX: Self = Self(9);
    pub const TANH: Self = Self(10);
    pub const SIGMOID: Self = Self(11);
    pub const RELU: Self = Self(12);
    pub const RESHAPE: Self = Self(13);
    pub const TRANSPOSE: Self = Self(14);
    pub const COPY: Self = Self(15);
    pub const POW: Self = Self(16);
    pub const MAXIMUM: Self = Self(17);
    pub const MINIMUM: Self = Self(18);
    pub const ABS: Self = Self(19);
    pub const COS: Self = Self(20);
    pub const EXP: Self = Self(21);
    pub const LOG: Self = Self(22);
    pub const LOGICAL_NOT: Self = Self(23);
    pub const NEGATE: Self = Self(24);
    pub const RECIPROCAL: Self = Self(25);
    pub const RSQRT: Self = Self(26);
    pub const SIN: Self = Self(27);
    pub const SELECT: Self = Self(28);
    pub const EQUAL: Self = Self(29);
    pub const GREATER: Self = Self(30);
    pub const GREATER_EQUAL: Self = Self(31);
    pub const LOGICAL_AND: Self = Self(32);
    pub const LOGICAL_OR: Self = Self(33);
    pub const LOGICAL_XOR: Self = Self(34);
    pub const SQRT: Self = Self(35);
    pub const FLOOR: Self = Self(36);
    pub const CEIL: Self = Self(37);
    pub const ROUND: Self = Self(38);
    pub const TRUNC: Self = Self(39);
    pub const REMAINDER: Self = Self(40);
    pub const ATAN: Self = Self(41);
    pub const ATAN2: Self = Self(42);
    pub const ACOS: Self = Self(43);
    pub const ASIN: Self = Self(44);
    pub const TAN: Self = Self(45);
    pub const EXP2: Self = Self(46);
    pub const LOG2: Self = Self(47);
    pub const SIGN: Self = Self(48);
    pub const CLAMP: Self = Self(49);
    pub const FMA: Self = Self(50);
    pub const INTERLEAVE3_F32: Self = Self(51);
    pub const VOXEL_BRICK_TRACE: Self = Self(52);
    pub const VOLUME_TRILINEAR_RGB: Self = Self(53);
    pub const BUFFER_LOAD_U8: Self = Self(54);
    pub const BITWISE_AND: Self = Self(55);
    pub const BITWISE_OR: Self = Self(56);
    pub const BITWISE_XOR: Self = Self(57);
    pub const SHIFT_LEFT: Self = Self(58);
    pub const SHIFT_RIGHT: Self = Self(59);
    /// The raster node family (`crate::raster`): a rounded fill, an inner
    /// ring, a soft shadow, a surface blit, a scaled surface sample and a
    /// glyph run, each painting a bound packed-XRGB target from one row of
    /// a parameter table.
    pub const RASTER_FILL: Self = Self(60);
    pub const RASTER_RING: Self = Self(61);
    pub const RASTER_SHADOW: Self = Self(62);
    pub const SURFACE_BLIT: Self = Self(63);
    pub const SURFACE_SAMPLE: Self = Self(64);
    pub const RASTER_GLYPHS: Self = Self(65);

    /// Row-wise matrix product [m,k] x transpose([n,k]).
    pub const MATMUL_TRANSPOSED: Self = Self(66);
    pub const SOFTMAX: Self = Self(67);
    pub const RMS_NORM: Self = Self(68);
    pub const LAYER_NORM: Self = Self(69);

    /// The key/value cache pair (`crate::kv`): an ordered write of token
    /// rows into a bound cache, and an ordered grouped-query read of it.
    pub const KV_APPEND: Self = Self(70);
    pub const ATTENTION_DECODE: Self = Self(71);

    /// Decoder data movement and selection (`crate::tensor_ops`).
    /// `x * sigmoid(x)`.
    pub const SILU: Self = Self(72);
    /// Rows of a rank-2 table selected by I32 indices.
    pub const GATHER: Self = Self(73);
    /// A static range of one axis.
    /// A reshape or contiguous range of its input's bytes. Executors alias
    /// the input; no data moves.
    pub const VIEW: Self = Self(74);
    /// The `k` largest last-axis values, then their I32 indices.
    pub const TOP_K: Self = Self(75);

    /// Recurrences over carried state (`crate::recurrent`): a depthwise
    /// causal convolution window and the Mamba-2 state-space step.
    pub const CAUSAL_CONV1D: Self = Self(76);
    pub const SSM_SCAN: Self = Self(77);

    /// Mixture-of-experts dispatch over a bound expert store (`crate::moe`).
    pub const MOE_EXPERTS: Self = Self(78);
    /// Native checkpoint NVFP4 projection: F32 activations, packed E2M1
    /// weights, and E4M3 block scales. Attributes carry the fused epilogue.
    pub const MATMUL_NVFP4_TRANSPOSED: Self = Self(79);
    /// F32 activations times signed I8 checkpoint rows with one F32 scale
    /// per output row. The scale is applied to the dot product.
    pub const MATMUL_I8_PER_CHANNEL: Self = Self(80);
    /// Fixed eight-row sink grouped-query attention over a physical F32 ring.
    /// Its five inputs are Q, ring K/V, block K/V, sinks, and I32 controls.
    pub const SINK_GQA_BLOCK: Self = Self(81);
    /// Fixed eight-row DSpark-style Q/K/V preparation. Dynamic checkpoint
    /// weights and norm vectors yield attention-ready Q and block K/V ports.
    pub const QKV_PREP_BLOCK: Self = Self(82);
    /// Eight-row gated MLP over two dynamic per-output-channel I8 matrices.
    /// The input is RMS-normalized; output is SiLU(gate) times up.
    pub const MLP_GATE_UP_I8_BLOCK: Self = Self(83);
    /// Three distinct target residual captures and one dynamic I8 matrix
    /// produce a partial DSpark context projection, without a host concat.
    pub const CONTEXT_PARTIAL_I8_BLOCK: Self = Self(84);
    /// Eight-row I8-weight down projection fused with its residual addition.
    pub const MLP_DOWN_RESIDUAL_I8_BLOCK: Self = Self(85);
    /// Sum two packed context partials, apply hidden RMSNorm, then prepare
    /// one drafter layer's position-rotated key and unrotated value blocks.
    pub const CONTEXT_KV_BLOCK: Self = Self(86);
    /// Eight-row sink attention over a packed physical ring/block view,
    /// followed by I8 O projection and residual addition.
    pub const SINK_GQA_O_RESIDUAL_BLOCK: Self = Self(87);
    /// The target's shared LM head with one-time converted I8 rows.
    pub const DSPARK_LM_HEAD_I8: Self = Self(88);
    /// Eight serial Markov corrections and deterministic token choices.
    pub const DSPARK_MARKOV_FEEDBACK: Self = Self(89);
    /// Eight-row Q/K/V preparation with output-major resident I8 weights.
    pub const QKV_PREP_I8_BLOCK: Self = Self(90);
    /// Context K/V producer with signed-I8 output rows and per-row scales.
    /// The packed partial and ring output semantics match opcode 86.
    pub const CONTEXT_KV_I8_BLOCK: Self = Self(91);
    /// Rotary position embedding producing a fresh rotated query or key tensor:
    /// F32 `[tokens, heads, head_dim]` input, per-token cosine and sine
    /// vectors, and a same-shaped output. Attributes tag the layout,
    /// and from 1.22 may name a leading rotated span (partial rotation).
    pub const ROPE: Self = Self(92);
    /// A gated delta rule (Gated DeltaNet) step over a carried key-by-value
    /// state (`crate::recurrent`): F32 query/key/value rows, log-space
    /// decay and write strength, read from a source state and written to a
    /// distinct target like the other recurrences.
    pub const GATED_DELTA_RULE: Self = Self(93);
    /// `log(1 + exp(x))`, and `x` above 20: PyTorch's `softplus`.
    pub const SOFTPLUS: Self = Self(94);
    /// Mixture-of-experts over an expert table of graph weights
    /// (`crate::routed`): each token's route from its router logits (I32
    /// `(expert, weight bits)` slots), then the routed NVFP4 SwiGLU and
    /// down projection, each record of the table one expert.
    pub const MOE_ROUTE: Self = Self(95);
    pub const MOE_GATE_UP_NVFP4: Self = Self(96);
    pub const MOE_DOWN_NVFP4: Self = Self(97);

    const fn fixed_arity(self) -> Option<(u16, u16)> {
        match self.0 {
            0 | 1 | 8..=12 | 15 | 19..=27 | 35..=39 | 41 | 43..=48 => Some((1, 1)),
            2..=6 | 13 | 16..=18 | 29..=34 | 40 | 42 => Some((2, 1)),
            28 | 49 | 50 | 51 => Some((3, 1)),
            52 => Some((3, 2)),
            53 => Some((3, 3)),
            54 => Some((1, 1)),
            55..=59 => Some((2, 1)),
            60..=65 | 66 | 68 => Some((2, 1)),
            67 => Some((1, 1)),
            69 => Some((3, 1)),
            79 | 80 => Some((3, 1)),
            81 => Some((5, 1)),
            82 => Some((4, 2)),
            83 => Some((5, 1)),
            84 => Some((5, 1)),
            85 => Some((4, 1)),
            86 | 91 => Some((4, 2)),
            87 => Some((5, 1)),
            88 => Some((3, 1)),
            89 => Some((5, 1)),
            90 => Some((4, 2)),
            92 => Some((3, 1)),
            93 => Some((3, 1)),
            94 | 95 => Some((1, 1)),
            96 | 97 => Some((3, 1)),
            70 | 71 => Some((3, 1)),
            72 | 74 => Some((1, 1)),
            73 => Some((2, 1)),
            75 => Some((1, 2)),
            76..=78 => Some((3, 1)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorFlags(pub u16);

impl TensorFlags {
    pub const INPUT: Self = Self(1 << 0);
    pub const OUTPUT: Self = Self(1 << 1);
    pub const CONSTANT: Self = Self(1 << 2);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl core::ops::BitOr for TensorFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeFlags(pub u16);

impl NodeFlags {
    /// The operation has externally observable ordering requirements.
    pub const ORDERED: Self = Self(1 << 0);
    /// The operation may update an aliased tensor.
    pub const MUTATING: Self = Self(1 << 1);
    /// Reads a caller-bound resource whose contents may change between runs.
    pub const READS_RESOURCE: Self = Self(1 << 2);
    /// Writes a caller-bound resource: a raster node painting its target.
    pub const WRITES_RESOURCE: Self = Self(1 << 3);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxError {
    Truncated,
    BadMagic,
    UnsupportedVersion {
        major: u16,
        minor: u16,
    },
    NonCanonicalLayout,
    CountOverflow,
    InvalidDType(u8),
    InvalidTensorFlags(u16),
    InvalidNodeFlags(u16),
    InvalidNodeAttributes(OpCode),
    InvalidVersionForNode {
        op: OpCode,
        minor: u16,
    },
    RankTooLarge(u8),
    InvalidTensorReference(u32),
    InvalidTensorRange,
    InvalidNodeRange,
    InvalidArity {
        op: OpCode,
        inputs: u16,
        outputs: u16,
    },
    ConstantFlagMissing,
    OutputBufferTooSmall {
        needed: usize,
        actual: usize,
    },
}

#[derive(Clone, Copy)]
struct Header {
    tensor_count: u32,
    node_count: u32,
    reference_count: u32,
    dimension_count: u32,
    attribute_bytes: u32,
    constant_bytes: u32,
    tensors_offset: u32,
    nodes_offset: u32,
    references_offset: u32,
    dimensions_offset: u32,
    attributes_offset: u32,
    constants_offset: u32,
}

#[derive(Clone, Copy)]
pub struct Graph<'a> {
    bytes: &'a [u8],
    header: Header,
}

impl<'a> Graph<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, AxError> {
        if bytes.len() < HEADER_BYTES {
            return Err(AxError::Truncated);
        }
        if bytes[..4] != MAGIC {
            return Err(AxError::BadMagic);
        }
        let major = read_u16(bytes, 4)?;
        let minor = read_u16(bytes, 6)?;
        if major != VERSION_MAJOR || minor > VERSION_MINOR {
            return Err(AxError::UnsupportedVersion { major, minor });
        }
        if read_u16(bytes, 8)? as usize != HEADER_BYTES || read_u16(bytes, 10)? != 0 {
            return Err(AxError::NonCanonicalLayout);
        }
        if read_u32(bytes, 12)? != 0 {
            return Err(AxError::NonCanonicalLayout);
        }
        let header = Header {
            tensor_count: read_u32(bytes, 16)?,
            node_count: read_u32(bytes, 20)?,
            reference_count: read_u32(bytes, 24)?,
            dimension_count: read_u32(bytes, 28)?,
            attribute_bytes: read_u32(bytes, 32)?,
            constant_bytes: read_u32(bytes, 36)?,
            tensors_offset: read_u32(bytes, 40)?,
            nodes_offset: read_u32(bytes, 44)?,
            references_offset: read_u32(bytes, 48)?,
            dimensions_offset: read_u32(bytes, 52)?,
            attributes_offset: read_u32(bytes, 56)?,
            constants_offset: read_u32(bytes, 60)?,
        };
        validate_layout(bytes.len(), header)?;
        let graph = Self { bytes, header };
        graph.validate_records()?;
        Ok(graph)
    }

    pub const fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }

    pub const fn node_count(&self) -> usize {
        self.header.node_count as usize
    }

    pub const fn reference_count(&self) -> usize {
        self.header.reference_count as usize
    }

    /// Exact canonical bytes backing this graph.
    ///
    /// Cache implementations should compare these bytes (plus concrete
    /// dynamic shapes and backend generation) before reusing compiled code.
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn tensor(&self, id: TensorId) -> Result<TensorView<'a>, AxError> {
        let index = id.0 as usize;
        if index >= self.tensor_count() {
            return Err(AxError::InvalidTensorReference(id.0));
        }
        let offset = self.header.tensors_offset as usize + index * TENSOR_BYTES;
        let dtype_tag = self.bytes[offset];
        let rank = self.bytes[offset + 1];
        let flags = TensorFlags(read_u16(self.bytes, offset + 2)?);
        let dimensions_start = read_u32(self.bytes, offset + 4)?;
        let constant_start = read_u32(self.bytes, offset + 8)?;
        let constant_len = read_u32(self.bytes, offset + 12)?;
        let dtype = decode_dtype(dtype_tag).ok_or(AxError::InvalidDType(dtype_tag))?;
        Ok(TensorView {
            graph: *self,
            id,
            dtype,
            rank,
            flags,
            dimensions_start,
            constant_start,
            constant_len,
        })
    }

    pub fn node(&self, index: usize) -> Result<NodeView<'a>, AxError> {
        if index >= self.node_count() {
            return Err(AxError::InvalidNodeRange);
        }
        let offset = self.header.nodes_offset as usize + index * NODE_BYTES;
        Ok(NodeView {
            graph: *self,
            index,
            op: OpCode(read_u16(self.bytes, offset)?),
            flags: NodeFlags(read_u16(self.bytes, offset + 2)?),
            inputs_start: read_u32(self.bytes, offset + 4)?,
            input_count: read_u16(self.bytes, offset + 8)?,
            output_count: read_u16(self.bytes, offset + 10)?,
            outputs_start: read_u32(self.bytes, offset + 12)?,
            attributes_start: read_u32(self.bytes, offset + 16)?,
            attributes_len: read_u32(self.bytes, offset + 20)?,
        })
    }

    fn reference(&self, index: usize) -> TensorId {
        let offset = self.header.references_offset as usize + index * 4;
        TensorId(u32::from_le_bytes(
            self.bytes[offset..offset + 4].try_into().unwrap(),
        ))
    }

    fn validate_records(&self) -> Result<(), AxError> {
        let minor = read_u16(self.bytes, 6)?;
        let mut expected_dimensions = 0_u32;
        let mut expected_constants = 0_u32;
        for index in 0..self.tensor_count() {
            let tensor = self.tensor(TensorId(index as u32))?;
            if tensor.flags.0
                & !(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0 | TensorFlags::CONSTANT.0)
                != 0
            {
                return Err(AxError::InvalidTensorFlags(tensor.flags.0));
            }
            if tensor.rank as usize > axnn::MAX_RANK {
                return Err(AxError::RankTooLarge(tensor.rank));
            }
            if tensor.dimensions_start != expected_dimensions
                || tensor.constant_start != expected_constants
            {
                return Err(AxError::NonCanonicalLayout);
            }
            let dims_end = tensor
                .dimensions_start
                .checked_add(tensor.rank as u32)
                .ok_or(AxError::CountOverflow)?;
            if dims_end > self.header.dimension_count {
                return Err(AxError::InvalidTensorRange);
            }
            let constant_end = tensor
                .constant_start
                .checked_add(tensor.constant_len)
                .ok_or(AxError::CountOverflow)?;
            if constant_end > self.header.constant_bytes {
                return Err(AxError::InvalidTensorRange);
            }
            if tensor.constant_len != 0 && !tensor.flags.contains(TensorFlags::CONSTANT) {
                return Err(AxError::ConstantFlagMissing);
            }
            expected_dimensions = dims_end;
            expected_constants = constant_end;
        }
        let mut expected_references = 0_u32;
        let mut expected_attributes = 0_u32;
        for index in 0..self.node_count() {
            let node = self.node(index)?;
            validate_node_flags(node.op, node.flags)?;
            if minor < node_minimum_minor(node.op, node.flags, node.attributes()) {
                return Err(AxError::InvalidVersionForNode { op: node.op, minor });
            }
            if let Some((inputs, outputs)) = node.op.fixed_arity()
                && (node.input_count != inputs || node.output_count != outputs)
            {
                return Err(AxError::InvalidArity {
                    op: node.op,
                    inputs: node.input_count,
                    outputs: node.output_count,
                });
            }
            if node.inputs_start != expected_references
                || node.outputs_start
                    != node
                        .inputs_start
                        .checked_add(node.input_count as u32)
                        .ok_or(AxError::CountOverflow)?
                || node.attributes_start != expected_attributes
            {
                return Err(AxError::NonCanonicalLayout);
            }
            let input_end = node
                .inputs_start
                .checked_add(node.input_count as u32)
                .ok_or(AxError::CountOverflow)?;
            let output_end = node
                .outputs_start
                .checked_add(node.output_count as u32)
                .ok_or(AxError::CountOverflow)?;
            let attributes_end = node
                .attributes_start
                .checked_add(node.attributes_len)
                .ok_or(AxError::CountOverflow)?;
            if input_end > self.header.reference_count
                || output_end > self.header.reference_count
                || attributes_end > self.header.attribute_bytes
            {
                return Err(AxError::InvalidNodeRange);
            }
            validate_node_attributes(node.op, node.attributes())?;
            for tensor in node.inputs().chain(node.outputs()) {
                if tensor.0 >= self.header.tensor_count {
                    return Err(AxError::InvalidTensorReference(tensor.0));
                }
            }
            expected_references = output_end;
            expected_attributes = attributes_end;
        }
        if expected_dimensions != self.header.dimension_count
            || expected_constants != self.header.constant_bytes
            || expected_references != self.header.reference_count
            || expected_attributes != self.header.attribute_bytes
        {
            return Err(AxError::NonCanonicalLayout);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    graph: Graph<'a>,
    pub id: TensorId,
    pub dtype: DType,
    rank: u8,
    pub flags: TensorFlags,
    dimensions_start: u32,
    constant_start: u32,
    constant_len: u32,
}

impl<'a> TensorView<'a> {
    pub const fn rank(&self) -> usize {
        self.rank as usize
    }

    pub fn dimensions(&self) -> Dimensions<'a> {
        Dimensions {
            bytes: self.graph.bytes,
            offset: self.graph.header.dimensions_offset as usize
                + self.dimensions_start as usize * 4,
            remaining: self.rank as usize,
        }
    }

    pub fn constant(&self) -> &'a [u8] {
        let start = self.graph.header.constants_offset as usize + self.constant_start as usize;
        &self.graph.bytes[start..start + self.constant_len as usize]
    }
}

pub struct Dimensions<'a> {
    bytes: &'a [u8],
    offset: usize,
    remaining: usize,
}

impl Iterator for Dimensions<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let value =
            u32::from_le_bytes(self.bytes[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        self.remaining -= 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Dimensions<'_> {}

#[derive(Clone, Copy)]
pub struct NodeView<'a> {
    graph: Graph<'a>,
    pub index: usize,
    pub op: OpCode,
    pub flags: NodeFlags,
    inputs_start: u32,
    input_count: u16,
    output_count: u16,
    outputs_start: u32,
    attributes_start: u32,
    attributes_len: u32,
}

impl<'a> NodeView<'a> {
    pub const fn input_count(&self) -> usize {
        self.input_count as usize
    }

    pub const fn output_count(&self) -> usize {
        self.output_count as usize
    }

    /// Whether this node may be memoized when all input versions match.
    pub const fn cacheable(&self) -> bool {
        self.flags.0
            & (NodeFlags::ORDERED.0
                | NodeFlags::MUTATING.0
                | NodeFlags::READS_RESOURCE.0
                | NodeFlags::WRITES_RESOURCE.0)
            == 0
    }

    pub fn inputs(&self) -> References<'a> {
        References {
            graph: self.graph,
            index: self.inputs_start as usize,
            remaining: self.input_count as usize,
        }
    }

    pub fn outputs(&self) -> References<'a> {
        References {
            graph: self.graph,
            index: self.outputs_start as usize,
            remaining: self.output_count as usize,
        }
    }

    pub fn attributes(&self) -> &'a [u8] {
        let start = self.graph.header.attributes_offset as usize + self.attributes_start as usize;
        &self.graph.bytes[start..start + self.attributes_len as usize]
    }
}

pub struct References<'a> {
    graph: Graph<'a>,
    index: usize,
    remaining: usize,
}

impl Iterator for References<'_> {
    type Item = TensorId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let value = self.graph.reference(self.index);
        self.index += 1;
        self.remaining -= 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for References<'_> {}

#[derive(Clone, Copy)]
pub struct TensorSpec<'a> {
    pub dtype: DType,
    pub flags: TensorFlags,
    pub dimensions: &'a [u32],
    pub constant: &'a [u8],
}

#[derive(Clone, Copy)]
pub struct NodeSpec<'a> {
    pub op: OpCode,
    pub flags: NodeFlags,
    pub inputs: &'a [TensorId],
    pub outputs: &'a [TensorId],
    pub attributes: &'a [u8],
}

#[derive(Clone, Copy)]
struct EncodingLayout {
    minor: u16,
    tensor_count: u32,
    node_count: u32,
    reference_count: u32,
    dimension_count: u32,
    attribute_bytes: u32,
    constant_bytes: u32,
    tensors_offset: usize,
    nodes_offset: usize,
    references_offset: usize,
    dimensions_offset: usize,
    attributes_offset: usize,
    constants_offset: usize,
    total: usize,
}

/// Returns the exact number of bytes required to encode a canonical graph.
pub fn encoded_len(tensors: &[TensorSpec<'_>], nodes: &[NodeSpec<'_>]) -> Result<usize, AxError> {
    Ok(encoding_layout(tensors, nodes)?.total)
}

pub fn encode(
    dst: &mut [u8],
    tensors: &[TensorSpec<'_>],
    nodes: &[NodeSpec<'_>],
) -> Result<usize, AxError> {
    let layout = encoding_layout(tensors, nodes)?;
    let EncodingLayout {
        minor,
        tensor_count,
        node_count,
        reference_count,
        dimension_count,
        attribute_bytes,
        constant_bytes,
        tensors_offset,
        nodes_offset,
        references_offset,
        dimensions_offset,
        attributes_offset,
        constants_offset,
        total,
    } = layout;
    if dst.len() < total {
        return Err(AxError::OutputBufferTooSmall {
            needed: total,
            actual: dst.len(),
        });
    }
    dst[..total].fill(0);
    dst[..4].copy_from_slice(&MAGIC);
    write_u16(dst, 4, VERSION_MAJOR);
    write_u16(dst, 6, minor);
    write_u16(dst, 8, HEADER_BYTES as u16);
    write_u32(dst, 16, tensor_count);
    write_u32(dst, 20, node_count);
    write_u32(dst, 24, reference_count);
    write_u32(dst, 28, dimension_count);
    write_u32(dst, 32, attribute_bytes);
    write_u32(dst, 36, constant_bytes);
    write_u32(dst, 40, as_u32(tensors_offset)?);
    write_u32(dst, 44, as_u32(nodes_offset)?);
    write_u32(dst, 48, as_u32(references_offset)?);
    write_u32(dst, 52, as_u32(dimensions_offset)?);
    write_u32(dst, 56, as_u32(attributes_offset)?);
    write_u32(dst, 60, as_u32(constants_offset)?);

    let (mut dimension_cursor, mut constant_cursor) = (0_u32, 0_u32);
    for (index, tensor) in tensors.iter().enumerate() {
        if tensor.dimensions.len() > axnn::MAX_RANK {
            return Err(AxError::RankTooLarge(tensor.dimensions.len() as u8));
        }
        if !tensor.constant.is_empty() && !tensor.flags.contains(TensorFlags::CONSTANT) {
            return Err(AxError::ConstantFlagMissing);
        }
        let offset = tensors_offset + index * TENSOR_BYTES;
        dst[offset] = encode_dtype(tensor.dtype);
        dst[offset + 1] = tensor.dimensions.len() as u8;
        write_u16(dst, offset + 2, tensor.flags.0);
        write_u32(dst, offset + 4, dimension_cursor);
        write_u32(dst, offset + 8, constant_cursor);
        write_u32(dst, offset + 12, as_u32(tensor.constant.len())?);
        for &dimension in tensor.dimensions {
            let out = dimensions_offset + dimension_cursor as usize * 4;
            write_u32(dst, out, dimension);
            dimension_cursor += 1;
        }
        let out = constants_offset + constant_cursor as usize;
        dst[out..out + tensor.constant.len()].copy_from_slice(tensor.constant);
        constant_cursor += tensor.constant.len() as u32;
    }

    let (mut reference_cursor, mut attribute_cursor) = (0_u32, 0_u32);
    for (index, node) in nodes.iter().enumerate() {
        let input_count = u16::try_from(node.inputs.len()).map_err(|_| AxError::CountOverflow)?;
        let output_count = u16::try_from(node.outputs.len()).map_err(|_| AxError::CountOverflow)?;
        let offset = nodes_offset + index * NODE_BYTES;
        write_u16(dst, offset, node.op.0);
        write_u16(dst, offset + 2, node.flags.0);
        write_u32(dst, offset + 4, reference_cursor);
        write_u16(dst, offset + 8, input_count);
        write_u16(dst, offset + 10, output_count);
        write_u32(dst, offset + 12, reference_cursor + input_count as u32);
        write_u32(dst, offset + 16, attribute_cursor);
        write_u32(dst, offset + 20, as_u32(node.attributes.len())?);
        for &tensor in node.inputs.iter().chain(node.outputs) {
            if tensor.0 >= tensor_count {
                return Err(AxError::InvalidTensorReference(tensor.0));
            }
            let out = references_offset + reference_cursor as usize * 4;
            write_u32(dst, out, tensor.0);
            reference_cursor += 1;
        }
        let out = attributes_offset + attribute_cursor as usize;
        dst[out..out + node.attributes.len()].copy_from_slice(node.attributes);
        attribute_cursor += node.attributes.len() as u32;
    }
    Ok(total)
}

fn encoding_layout(
    tensors: &[TensorSpec<'_>],
    nodes: &[NodeSpec<'_>],
) -> Result<EncodingLayout, AxError> {
    let tensor_count = u32::try_from(tensors.len()).map_err(|_| AxError::CountOverflow)?;
    let node_count = u32::try_from(nodes.len()).map_err(|_| AxError::CountOverflow)?;
    for tensor in tensors {
        if tensor.flags.0
            & !(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0 | TensorFlags::CONSTANT.0)
            != 0
        {
            return Err(AxError::InvalidTensorFlags(tensor.flags.0));
        }
        if tensor.dimensions.len() > axnn::MAX_RANK {
            return Err(AxError::RankTooLarge(
                u8::try_from(tensor.dimensions.len()).unwrap_or(u8::MAX),
            ));
        }
        if !tensor.constant.is_empty() && !tensor.flags.contains(TensorFlags::CONSTANT) {
            return Err(AxError::ConstantFlagMissing);
        }
    }
    let dimension_count = sum_u32(tensors.iter().map(|tensor| tensor.dimensions.len()))?;
    let mut reference_count = 0_u32;
    let mut minor = 0;
    for node in nodes {
        validate_node_flags(node.op, node.flags)?;
        let inputs = u16::try_from(node.inputs.len()).map_err(|_| AxError::CountOverflow)?;
        let outputs = u16::try_from(node.outputs.len()).map_err(|_| AxError::CountOverflow)?;
        if let Some((expected_inputs, expected_outputs)) = node.op.fixed_arity()
            && (inputs != expected_inputs || outputs != expected_outputs)
        {
            return Err(AxError::InvalidArity {
                op: node.op,
                inputs,
                outputs,
            });
        }
        validate_node_attributes(node.op, node.attributes)?;
        minor = minor.max(node_minimum_minor(node.op, node.flags, node.attributes));
        reference_count = reference_count
            .checked_add(inputs as u32)
            .and_then(|count| count.checked_add(outputs as u32))
            .ok_or(AxError::CountOverflow)?;
    }
    let attribute_bytes = sum_u32(nodes.iter().map(|node| node.attributes.len()))?;
    let constant_bytes = sum_u32(tensors.iter().map(|tensor| tensor.constant.len()))?;
    let tensors_offset = HEADER_BYTES;
    let nodes_offset = add_mul(tensors_offset, tensors.len(), TENSOR_BYTES)?;
    let references_offset = add_mul(nodes_offset, nodes.len(), NODE_BYTES)?;
    let dimensions_offset = add_mul(references_offset, reference_count as usize, 4)?;
    let attributes_offset = add_mul(dimensions_offset, dimension_count as usize, 4)?;
    let constants_offset = attributes_offset
        .checked_add(attribute_bytes as usize)
        .ok_or(AxError::CountOverflow)?;
    let total = constants_offset
        .checked_add(constant_bytes as usize)
        .ok_or(AxError::CountOverflow)?;
    Ok(EncodingLayout {
        minor,
        tensor_count,
        node_count,
        reference_count,
        dimension_count,
        attribute_bytes,
        constant_bytes,
        tensors_offset,
        nodes_offset,
        references_offset,
        dimensions_offset,
        attributes_offset,
        constants_offset,
        total,
    })
}

const fn node_minimum_minor(op: OpCode, flags: NodeFlags, attributes: &[u8]) -> u16 {
    // 1.8 fused epilogues (#107): a matmul that carries a scale, activation
    // or residual accumulate, and a convolution carrying an activation.
    // 1.22: the gated delta rule, softplus, and a partially rotated ROPE.
    if matches!(op.0, 95..=97) {
        23
    } else if matches!(op.0, 93 | 94)
        || (op.0 == 92 && attributes.len() == crate::tensor_ops::ROPE_PARTIAL_ATTRIBUTE_BYTES)
    {
        22
    } else if op.0 == 92 {
        21
    } else if op.0 == 91 {
        20
    } else if op.0 == 90 {
        19
    } else if op.0 == 89 {
        18
    } else if op.0 == 88 {
        17
    } else if op.0 == 87 {
        16
    } else if op.0 == 86 {
        15
    } else if op.0 == 85 {
        14
    } else if op.0 == 84 {
        13
    } else if op.0 == 83 {
        12
    } else if op.0 == 82 {
        11
    } else if matches!(op.0, 80 | 81) {
        10
    } else if op.0 == OpCode::MATMUL_NVFP4_TRANSPOSED.0 {
        9
    } else if (op.0 == OpCode::MATMUL_TRANSPOSED.0
        && attributes.len() == crate::tensor_ops::MATMUL_EPILOGUE_ATTRIBUTE_BYTES)
        || (op.0 == 76
            && attributes.len() == crate::recurrent::CAUSAL_CONV1D_ACTIVATED_ATTRIBUTE_BYTES)
    {
        8
    } else if op.0 == 78 {
        7
    } else if matches!(op.0, 76 | 77) {
        6
    } else if matches!(op.0, 72..=75)
        || (op.0 == OpCode::RMS_NORM.0
            && attributes.len() == crate::tensor_ops::RMS_NORM_AXIS_ATTRIBUTE_BYTES)
    {
        5
    } else if matches!(op.0, 70 | 71) {
        4
    } else if matches!(op.0, 66..=69) {
        3
    } else if matches!(op.0, 60..=65) || flags.contains(NodeFlags::WRITES_RESOURCE) {
        2
    } else if matches!(op.0, 51..=59) || flags.contains(NodeFlags::READS_RESOURCE) {
        1
    } else {
        0
    }
}

const fn is_raster(op: OpCode) -> bool {
    matches!(op.0, 60..=65)
}

fn validate_node_flags(op: OpCode, flags: NodeFlags) -> Result<(), AxError> {
    let known = NodeFlags::ORDERED.0
        | NodeFlags::MUTATING.0
        | NodeFlags::READS_RESOURCE.0
        | NodeFlags::WRITES_RESOURCE.0;
    if flags.0 & !known != 0
        || (matches!(
            op,
            OpCode::VOXEL_BRICK_TRACE | OpCode::VOLUME_TRILINEAR_RGB | OpCode::BUFFER_LOAD_U8
        ) && (!flags.contains(NodeFlags::READS_RESOURCE)
            || flags.contains(NodeFlags::MUTATING)))
        || (op.0 <= OpCode::INTERLEAVE3_F32.0 && flags.contains(NodeFlags::READS_RESOURCE))
        || (matches!(op.0, 55..=59 | 72..=75 | 94) && flags.contains(NodeFlags::READS_RESOURCE))
        // Routing and the routed projections are pure: their experts are
        // graph weights.
        || (matches!(op.0, 95..=97) && flags != NodeFlags::default())
        || (op == OpCode::MATMUL_I8_PER_CHANNEL && flags != NodeFlags::default())
        // Only raster nodes, cache appends and recurrent steps write a
        // resource. A raster node's effects are exactly an ordered
        // read+write; an append is an ordered write that never reads, and a
        // cache read is ordered so a later append to the same cache cannot
        // overtake it. A recurrent step reads its source state and writes
        // its target, in order.
        || ((is_raster(op) || matches!(op.0, 70 | 76 | 77 | 93))
            != flags.contains(NodeFlags::WRITES_RESOURCE))
        || (matches!(op.0, 76 | 77 | 93) && flags.0 != crate::recurrent::RECURRENT_NODE_FLAGS.0)
        // Expert dispatch only reads its immutable store.
        || (op.0 == 78 && flags.0 != NodeFlags::READS_RESOURCE.0)
        || (is_raster(op)
            && (!flags.contains(NodeFlags::ORDERED)
                || !flags.contains(NodeFlags::READS_RESOURCE)
                || flags.contains(NodeFlags::MUTATING)))
        || (op.0 == OpCode::KV_APPEND.0 && flags.0 != crate::kv::KV_APPEND_NODE_FLAGS.0)
        || (op.0 == OpCode::ATTENTION_DECODE.0
            && flags.0 != crate::kv::ATTENTION_DECODE_NODE_FLAGS.0)
        || (matches!(
            op,
            OpCode::SINK_GQA_BLOCK
                | OpCode::QKV_PREP_BLOCK
                | OpCode::MLP_GATE_UP_I8_BLOCK
                | OpCode::CONTEXT_PARTIAL_I8_BLOCK
                | OpCode::MLP_DOWN_RESIDUAL_I8_BLOCK
                | OpCode::CONTEXT_KV_BLOCK
                | OpCode::CONTEXT_KV_I8_BLOCK
                | OpCode::SINK_GQA_O_RESIDUAL_BLOCK
                | OpCode::DSPARK_LM_HEAD_I8
                | OpCode::DSPARK_MARKOV_FEEDBACK
                | OpCode::QKV_PREP_I8_BLOCK
        )
            && flags != NodeFlags::default())
    {
        return Err(AxError::InvalidNodeFlags(flags.0));
    }
    Ok(())
}

fn validate_node_attributes(op: OpCode, attributes: &[u8]) -> Result<(), AxError> {
    if (op == OpCode::INTERLEAVE3_F32 && !attributes.is_empty())
        || (matches!(op.0, 55..=59) && !attributes.is_empty())
        || (op == OpCode::BUFFER_LOAD_U8
            && crate::buffer::decode_u8_buffer_attributes(attributes).is_none())
        || (op == OpCode::VOXEL_BRICK_TRACE
            && crate::voxel::decode_brick_trace_attributes(attributes).is_none())
        || (op == OpCode::VOLUME_TRILINEAR_RGB
            && crate::sampling::decode_volume_rgb_attributes(attributes).is_none())
        || (is_raster(op) && crate::raster::decode_raster_attributes(attributes).is_none())
        || (matches!(op.0, 72 | 73 | 94) && !attributes.is_empty())
        || (op == OpCode::VIEW && crate::tensor_ops::decode_view_attributes(attributes).is_none())
        || (op == OpCode::TOP_K && crate::tensor_ops::decode_top_k_attributes(attributes).is_none())
        || (op == OpCode::KV_APPEND && crate::kv::decode_kv_append_attributes(attributes).is_none())
        || (op == OpCode::CAUSAL_CONV1D
            && crate::recurrent::decode_causal_conv1d_attributes(attributes).is_none())
        || (op == OpCode::SSM_SCAN
            && crate::recurrent::decode_ssm_scan_attributes(attributes).is_none())
        || (op == OpCode::GATED_DELTA_RULE
            && crate::recurrent::decode_gated_delta_rule_attributes(attributes).is_none())
        || (op == OpCode::MOE_EXPERTS && crate::moe::decode_moe_attributes(attributes).is_none())
        || (op == OpCode::MOE_ROUTE && crate::routed::decode_route_attributes(attributes).is_none())
        || (matches!(op, OpCode::MOE_GATE_UP_NVFP4 | OpCode::MOE_DOWN_NVFP4)
            && crate::routed::decode_routed_attributes(attributes).is_none())
        || (op == OpCode::MATMUL_NVFP4_TRANSPOSED
            && crate::tensor_ops::decode_matmul_epilogue_attributes(attributes).is_none())
        || (op == OpCode::MATMUL_I8_PER_CHANNEL && !attributes.is_empty())
        || (op == OpCode::ROPE
            && crate::tensor_ops::decode_rope_attributes(attributes).is_none())
        || (op == OpCode::ATTENTION_DECODE
            && crate::kv::decode_attention_decode_attributes(attributes).is_none())
        || (matches!(
            op,
            OpCode::SINK_GQA_BLOCK
                | OpCode::QKV_PREP_BLOCK
                | OpCode::MLP_GATE_UP_I8_BLOCK
                | OpCode::CONTEXT_PARTIAL_I8_BLOCK
                | OpCode::MLP_DOWN_RESIDUAL_I8_BLOCK
                | OpCode::CONTEXT_KV_BLOCK
                | OpCode::CONTEXT_KV_I8_BLOCK
                | OpCode::SINK_GQA_O_RESIDUAL_BLOCK
                | OpCode::DSPARK_LM_HEAD_I8
                | OpCode::DSPARK_MARKOV_FEEDBACK
                | OpCode::QKV_PREP_I8_BLOCK
        ) && !attributes.is_empty())
    {
        return Err(AxError::InvalidNodeAttributes(op));
    }
    Ok(())
}

fn validate_layout(bytes: usize, header: Header) -> Result<(), AxError> {
    let tensors = HEADER_BYTES;
    let nodes = add_mul(tensors, header.tensor_count as usize, TENSOR_BYTES)?;
    let references = add_mul(nodes, header.node_count as usize, NODE_BYTES)?;
    let dimensions = add_mul(references, header.reference_count as usize, 4)?;
    let attributes = add_mul(dimensions, header.dimension_count as usize, 4)?;
    let constants = attributes
        .checked_add(header.attribute_bytes as usize)
        .ok_or(AxError::CountOverflow)?;
    let total = constants
        .checked_add(header.constant_bytes as usize)
        .ok_or(AxError::CountOverflow)?;
    if header.tensors_offset as usize != tensors
        || header.nodes_offset as usize != nodes
        || header.references_offset as usize != references
        || header.dimensions_offset as usize != dimensions
        || header.attributes_offset as usize != attributes
        || header.constants_offset as usize != constants
        || bytes != total
    {
        return Err(AxError::NonCanonicalLayout);
    }
    Ok(())
}

fn decode_dtype(value: u8) -> Option<DType> {
    Some(match value {
        0 => DType::F64,
        1 => DType::F32,
        2 => DType::F16,
        3 => DType::B16,
        4 => DType::I64,
        5 => DType::I32,
        6 => DType::I16,
        7 => DType::I8,
        8 => DType::U8,
        9 => DType::Q8,
        10 => DType::Q6,
        11 => DType::Q4,
        12 => DType::Q1,
        13 => DType::B8,
        14 => DType::F8,
        _ => return None,
    })
}

fn encode_dtype(value: DType) -> u8 {
    value as u8
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, AxError> {
    let value = bytes.get(offset..offset + 2).ok_or(AxError::Truncated)?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AxError> {
    let value = bytes.get(offset..offset + 4).ok_or(AxError::Truncated)?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn as_u32(value: usize) -> Result<u32, AxError> {
    u32::try_from(value).map_err(|_| AxError::CountOverflow)
}

fn add_mul(base: usize, count: usize, width: usize) -> Result<usize, AxError> {
    count
        .checked_mul(width)
        .and_then(|bytes| base.checked_add(bytes))
        .ok_or(AxError::CountOverflow)
}

fn sum_u32(values: impl Iterator<Item = usize>) -> Result<u32, AxError> {
    let mut sum = 0_usize;
    for value in values {
        sum = sum.checked_add(value).ok_or(AxError::CountOverflow)?;
    }
    as_u32(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_float_tags_remain_wire_compatible() {
        assert_eq!(encode_dtype(DType::B8), 13);
        assert_eq!(encode_dtype(DType::F8), 14);
        assert_eq!(decode_dtype(13), Some(DType::B8));
        assert_eq!(decode_dtype(14), Some(DType::F8));
    }

    fn arity_tensors() -> [TensorSpec<'static>; 5] {
        [TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::default(),
            dimensions: &[1],
            constant: &[],
        }; 5]
    }

    fn assert_fixed_arity(op: OpCode, expected_inputs: usize) {
        let tensors = arity_tensors();
        let references = [
            TensorId(0),
            TensorId(1),
            TensorId(2),
            TensorId(3),
            TensorId(4),
        ];
        let output = [TensorId(4)];
        let node = NodeSpec {
            op,
            flags: NodeFlags::default(),
            inputs: &references[..expected_inputs],
            outputs: &output,
            attributes: &[],
        };
        assert!(encoded_len(&tensors, &[node]).is_ok());

        let too_few_inputs = NodeSpec {
            inputs: &references[..expected_inputs - 1],
            ..node
        };
        assert_eq!(
            encoded_len(&tensors, &[too_few_inputs]),
            Err(AxError::InvalidArity {
                op,
                inputs: (expected_inputs - 1) as u16,
                outputs: 1,
            })
        );

        let too_many_inputs = NodeSpec {
            inputs: &references[..expected_inputs + 1],
            ..node
        };
        assert_eq!(
            encoded_len(&tensors, &[too_many_inputs]),
            Err(AxError::InvalidArity {
                op,
                inputs: (expected_inputs + 1) as u16,
                outputs: 1,
            })
        );

        let two_outputs = [TensorId(3), TensorId(4)];
        let too_many_outputs = NodeSpec {
            outputs: &two_outputs,
            ..node
        };
        assert_eq!(
            encoded_len(&tensors, &[too_many_outputs]),
            Err(AxError::InvalidArity {
                op,
                inputs: expected_inputs as u16,
                outputs: 2,
            })
        );
    }

    #[test]
    fn opcode_numbers_preserve_the_v1_wire_contract() {
        let legacy = [
            OpCode::IDENTITY,
            OpCode::CAST,
            OpCode::ADD,
            OpCode::SUB,
            OpCode::MUL,
            OpCode::DIV,
            OpCode::MATMUL,
            OpCode::CONV1D,
            OpCode::REDUCE_SUM,
            OpCode::REDUCE_MAX,
            OpCode::TANH,
            OpCode::SIGMOID,
            OpCode::RELU,
            OpCode::RESHAPE,
            OpCode::TRANSPOSE,
            OpCode::COPY,
        ];
        let extension = [
            OpCode::POW,
            OpCode::MAXIMUM,
            OpCode::MINIMUM,
            OpCode::ABS,
            OpCode::COS,
            OpCode::EXP,
            OpCode::LOG,
            OpCode::LOGICAL_NOT,
            OpCode::NEGATE,
            OpCode::RECIPROCAL,
            OpCode::RSQRT,
            OpCode::SIN,
            OpCode::SELECT,
            OpCode::EQUAL,
            OpCode::GREATER,
            OpCode::GREATER_EQUAL,
            OpCode::LOGICAL_AND,
            OpCode::LOGICAL_OR,
            OpCode::LOGICAL_XOR,
            OpCode::SQRT,
            OpCode::FLOOR,
            OpCode::CEIL,
            OpCode::ROUND,
            OpCode::TRUNC,
            OpCode::REMAINDER,
            OpCode::ATAN,
            OpCode::ATAN2,
            OpCode::ACOS,
            OpCode::ASIN,
            OpCode::TAN,
            OpCode::EXP2,
            OpCode::LOG2,
            OpCode::SIGN,
            OpCode::CLAMP,
            OpCode::FMA,
        ];

        for (value, op) in legacy.into_iter().enumerate() {
            assert_eq!(op.0, value as u16);
        }
        for (value, op) in extension.into_iter().enumerate() {
            assert_eq!(op.0, (value + 16) as u16);
        }
    }

    #[test]
    fn new_unary_operations_enforce_fixed_arity() {
        for op in [
            OpCode::ABS,
            OpCode::COS,
            OpCode::EXP,
            OpCode::LOG,
            OpCode::LOGICAL_NOT,
            OpCode::NEGATE,
            OpCode::RECIPROCAL,
            OpCode::RSQRT,
            OpCode::SIN,
            OpCode::SQRT,
            OpCode::FLOOR,
            OpCode::CEIL,
            OpCode::ROUND,
            OpCode::TRUNC,
            OpCode::ATAN,
            OpCode::ACOS,
            OpCode::ASIN,
            OpCode::TAN,
            OpCode::EXP2,
            OpCode::LOG2,
            OpCode::SIGN,
        ] {
            assert_fixed_arity(op, 1);
        }
    }

    #[test]
    fn legacy_reductions_enforce_one_input_and_one_output() {
        for op in [OpCode::REDUCE_SUM, OpCode::REDUCE_MAX] {
            assert_fixed_arity(op, 1);
        }
    }

    #[test]
    fn new_binary_operations_enforce_fixed_arity() {
        for op in [
            OpCode::POW,
            OpCode::MAXIMUM,
            OpCode::MINIMUM,
            OpCode::EQUAL,
            OpCode::GREATER,
            OpCode::GREATER_EQUAL,
            OpCode::LOGICAL_AND,
            OpCode::LOGICAL_OR,
            OpCode::LOGICAL_XOR,
            OpCode::REMAINDER,
            OpCode::ATAN2,
        ] {
            assert_fixed_arity(op, 2);
        }
    }

    #[test]
    fn ternary_operations_enforce_fixed_arity() {
        for op in [OpCode::SELECT, OpCode::CLAMP, OpCode::FMA] {
            assert_fixed_arity(op, 3);
        }
    }

    #[test]
    fn wgsl_numeric_operations_roundtrip_without_attributes() {
        let tensors = arity_tensors();
        let inputs = [TensorId(0), TensorId(1), TensorId(2)];
        let output = [TensorId(4)];
        let operations = [
            (OpCode::SQRT, 1),
            (OpCode::FLOOR, 1),
            (OpCode::CEIL, 1),
            (OpCode::ROUND, 1),
            (OpCode::TRUNC, 1),
            (OpCode::REMAINDER, 2),
            (OpCode::ATAN, 1),
            (OpCode::ATAN2, 2),
            (OpCode::ACOS, 1),
            (OpCode::ASIN, 1),
            (OpCode::TAN, 1),
            (OpCode::EXP2, 1),
            (OpCode::LOG2, 1),
            (OpCode::SIGN, 1),
            (OpCode::CLAMP, 3),
            (OpCode::FMA, 3),
        ];

        for (op, input_count) in operations {
            let node = NodeSpec {
                op,
                flags: NodeFlags::default(),
                inputs: &inputs[..input_count],
                outputs: &output,
                attributes: &[],
            };
            let mut bytes = [0_u8; 512];
            let len = encode(&mut bytes, &tensors, &[node]).unwrap();
            let graph = Graph::parse(&bytes[..len]).unwrap();
            let parsed = graph.node(0).unwrap();

            assert_eq!(parsed.op, op);
            assert_eq!(parsed.input_count(), input_count);
            assert_eq!(parsed.output_count(), 1);
            assert!(parsed.inputs().eq(inputs[..input_count].iter().copied()));
            assert!(parsed.outputs().eq(output));
            assert!(parsed.attributes().is_empty());
        }
    }

    #[test]
    fn parser_rejects_noncanonical_wgsl_operation_arity() {
        let tensors = arity_tensors();
        let node = NodeSpec {
            op: OpCode::CLAMP,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0), TensorId(1), TensorId(2)],
            outputs: &[TensorId(4)],
            attributes: &[],
        };
        let mut bytes = [0_u8; 512];
        let len = encode(&mut bytes, &tensors, &[node]).unwrap();
        let node_offset = read_u32(&bytes[..len], 44).unwrap() as usize;
        write_u16(&mut bytes, node_offset + 8, 2);

        match Graph::parse(&bytes[..len]) {
            Err(error) => assert_eq!(
                error,
                AxError::InvalidArity {
                    op: OpCode::CLAMP,
                    inputs: 2,
                    outputs: 1,
                }
            ),
            Ok(_) => panic!("invalid CLAMP arity was accepted"),
        }
    }

    #[test]
    fn canonical_roundtrip_is_borrowed() {
        let tensors = [
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::INPUT,
                dimensions: &[1, 8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::CONSTANT,
                dimensions: &[8, 3],
                constant: &[1; 48],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1, 3],
                constant: &[],
            },
        ];
        let nodes = [NodeSpec {
            op: OpCode::MATMUL,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0), TensorId(1)],
            outputs: &[TensorId(2)],
            attributes: &[7, 8],
        }];
        let mut bytes = [0_u8; 512];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        assert_eq!(graph.tensor_count(), 3);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.tensor(TensorId(1)).unwrap().constant(), &[1; 48]);
        assert_eq!(
            graph
                .node(0)
                .unwrap()
                .inputs()
                .collect::<std::vec::Vec<_>>(),
            [TensorId(0), TensorId(1)]
        );
        assert_eq!(graph.node(0).unwrap().attributes(), &[7, 8]);
    }

    #[test]
    fn malformed_reference_is_rejected() {
        let tensors = [TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::INPUT,
            dimensions: &[1],
            constant: &[],
        }];
        let nodes = [NodeSpec {
            op: OpCode::COPY,
            flags: NodeFlags::default(),
            inputs: &[TensorId(9)],
            outputs: &[TensorId(0)],
            attributes: &[],
        }];
        let mut bytes = [0_u8; 256];
        assert_eq!(
            encode(&mut bytes, &tensors, &nodes),
            Err(AxError::InvalidTensorReference(9))
        );
    }

    #[test]
    fn noncanonical_record_ranges_are_rejected() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[2],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[2],
                constant: &[],
            },
        ];
        let mut bytes = [0_u8; 256];
        let len = encode(&mut bytes, &tensors, &[]).unwrap();
        // Tensor 1 must begin at dimension index 1, not overlap tensor 0.
        write_u32(&mut bytes, HEADER_BYTES + TENSOR_BYTES + 4, 0);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::NonCanonicalLayout)
        ));
    }

    fn trace_attributes() -> [u8; crate::voxel::VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES] {
        crate::voxel::encode_brick_trace_attributes(crate::voxel::VoxelBrickTraceParams {
            resource: crate::voxel::ResourceKey {
                group: 1,
                binding: 2,
            },
            dims: [16, 8, 4],
        })
        .unwrap()
    }

    #[test]
    fn ordinary_graphs_preserve_exact_v1_0_bytes() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
        ];
        let node = NodeSpec {
            op: OpCode::IDENTITY,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(1)],
            attributes: &[],
        };
        // Original 1.0 header, tensor records, identity, references, dimensions.
        let golden = concat!(
            "41584952010000004000000000000000",
            "02000000010000000200000002000000",
            "00000000000000004000000060000000",
            "78000000800000008800000088000000",
            "01010100000000000000000000000000",
            "01010200010000000000000000000000",
            "000000000000000001000100010000000000000000000000",
            "00000000010000000100000001000000",
        );
        let expected: alloc::vec::Vec<u8> = golden
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(core::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect();
        let mut encoded = [0_u8; 136];
        assert_eq!(
            encode(&mut encoded, &tensors, &[node]).unwrap(),
            expected.len()
        );
        assert_eq!(encoded.as_slice(), expected.as_slice());
        assert!(
            Graph::parse(&expected)
                .unwrap()
                .node(0)
                .unwrap()
                .cacheable()
        );
    }

    #[test]
    fn voxel_query_arity_flags_and_attributes_are_enforced_before_encoding() {
        let tensors = arity_tensors();
        let attributes = trace_attributes();
        let inputs = [TensorId(0), TensorId(1), TensorId(2)];
        let outputs = [TensorId(3), TensorId(4)];
        let node = NodeSpec {
            op: OpCode::VOXEL_BRICK_TRACE,
            flags: NodeFlags::READS_RESOURCE,
            inputs: &inputs,
            outputs: &outputs,
            attributes: &attributes,
        };
        assert_eq!(OpCode::INTERLEAVE3_F32.0, 51);
        assert_eq!(OpCode::VOXEL_BRICK_TRACE.0, 52);
        assert!(encoded_len(&tensors, &[node]).is_ok());
        for (input_count, output_count) in [(2, 2), (3, 1)] {
            let invalid = NodeSpec {
                inputs: &inputs[..input_count],
                outputs: &outputs[..output_count],
                ..node
            };
            assert_eq!(
                encoded_len(&tensors, &[invalid]),
                Err(AxError::InvalidArity {
                    op: node.op,
                    inputs: input_count as u16,
                    outputs: output_count as u16,
                })
            );
        }
        for flags in [
            NodeFlags::default(),
            NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::MUTATING.0),
        ] {
            assert_eq!(
                encoded_len(&tensors, &[NodeSpec { flags, ..node }]),
                Err(AxError::InvalidNodeFlags(flags.0))
            );
        }
        assert_eq!(
            encoded_len(
                &tensors,
                &[NodeSpec {
                    attributes: &[],
                    ..node
                }]
            ),
            Err(AxError::InvalidNodeAttributes(node.op))
        );
        let interleave = NodeSpec {
            op: OpCode::INTERLEAVE3_F32,
            flags: NodeFlags::default(),
            outputs: &outputs[..1],
            ..node
        };
        assert_eq!(
            encoded_len(&tensors, &[interleave]),
            Err(AxError::InvalidNodeAttributes(interleave.op))
        );
        assert_fixed_arity(OpCode::INTERLEAVE3_F32, 3);
    }

    #[test]
    fn raw_voxel_records_cannot_hide_resource_effects_or_version_requirements() {
        let tensors = arity_tensors();
        let attributes = trace_attributes();
        let node = NodeSpec {
            op: OpCode::VOXEL_BRICK_TRACE,
            flags: NodeFlags::READS_RESOURCE,
            inputs: &[TensorId(0), TensorId(1), TensorId(2)],
            outputs: &[TensorId(3), TensorId(4)],
            attributes: &attributes,
        };
        let mut bytes = [0_u8; 512];
        let len = encode(&mut bytes, &tensors, &[node]).unwrap();
        assert_eq!(read_u16(&bytes, 6).unwrap(), 1);
        assert!(
            !Graph::parse(&bytes[..len])
                .unwrap()
                .node(0)
                .unwrap()
                .cacheable()
        );
        let node_offset = read_u32(&bytes, 44).unwrap() as usize;
        for flags in [
            0,
            NodeFlags::READS_RESOURCE.0 | NodeFlags::MUTATING.0,
            1 << 15,
        ] {
            let mut invalid = bytes;
            write_u16(&mut invalid, node_offset + 2, flags);
            assert!(
                matches!(Graph::parse(&invalid[..len]), Err(AxError::InvalidNodeFlags(found)) if found == flags)
            );
        }
        let mut invalid = bytes;
        write_u16(&mut invalid, 6, 0);
        assert!(matches!(
            Graph::parse(&invalid[..len]),
            Err(AxError::InvalidVersionForNode {
                op: OpCode::VOXEL_BRICK_TRACE,
                minor: 0
            })
        ));
        let mut invalid = bytes;
        write_u16(&mut invalid, node_offset + 10, 1);
        assert!(matches!(
            Graph::parse(&invalid[..len]),
            Err(AxError::InvalidArity {
                op: OpCode::VOXEL_BRICK_TRACE,
                inputs: 3,
                outputs: 1
            })
        ));
        let attributes_offset = read_u32(&bytes, 56).unwrap() as usize;
        for (offset, value) in [(0, 2), (1, 2), (2, 1), (3, 1), (12, 0), (16, 3)] {
            let mut invalid = bytes;
            invalid[attributes_offset + offset] = value;
            assert!(matches!(
                Graph::parse(&invalid[..len]),
                Err(AxError::InvalidNodeAttributes(OpCode::VOXEL_BRICK_TRACE))
            ));
        }
    }

    #[test]
    fn rgb_sampling_enforces_arity_flags_attributes_and_minor_version() {
        use crate::sampling::{
            ResourceKey, VolumeRgbParams, VolumeRgbStorage, encode_volume_rgb_attributes,
        };

        let tensors = [arity_tensors()[0]; 6];
        let inputs = [TensorId(0), TensorId(1), TensorId(2), TensorId(3)];
        let outputs = [TensorId(3), TensorId(4), TensorId(5), TensorId(2)];
        assert_eq!(OpCode::VOLUME_TRILINEAR_RGB.0, 53);
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let attributes = encode_volume_rgb_attributes(
                VolumeRgbParams::new(
                    ResourceKey {
                        group: 1,
                        binding: 2,
                    },
                    [24, 12, 24],
                    storage,
                )
                .unwrap(),
            )
            .unwrap();
            let node = NodeSpec {
                op: OpCode::VOLUME_TRILINEAR_RGB,
                flags: NodeFlags::READS_RESOURCE,
                inputs: &inputs[..3],
                outputs: &outputs[..3],
                attributes: &attributes,
            };
            for (input_count, output_count) in [(0, 3), (2, 3), (4, 3), (3, 0), (3, 2), (3, 4)] {
                assert_eq!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            inputs: &inputs[..input_count],
                            outputs: &outputs[..output_count],
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidArity {
                        op: node.op,
                        inputs: input_count as u16,
                        outputs: output_count as u16
                    })
                );
            }
            for flags in [
                NodeFlags::default(),
                NodeFlags::MUTATING,
                NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::MUTATING.0),
            ] {
                assert_eq!(
                    encoded_len(&tensors, &[NodeSpec { flags, ..node }]),
                    Err(AxError::InvalidNodeFlags(flags.0))
                );
            }
            for attributes in [&[][..], &attributes[..23]] {
                assert_eq!(
                    encoded_len(&tensors, &[NodeSpec { attributes, ..node }]),
                    Err(AxError::InvalidNodeAttributes(node.op))
                );
            }
            let mut bytes = [0_u8; 512];
            let len = encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), 1);
            assert!(
                !Graph::parse(&bytes[..len])
                    .unwrap()
                    .node(0)
                    .unwrap()
                    .cacheable()
            );
            let node_offset = read_u32(&bytes, 44).unwrap() as usize;
            for flags in [
                0,
                NodeFlags::MUTATING.0,
                NodeFlags::READS_RESOURCE.0 | NodeFlags::MUTATING.0,
                1 << 15,
            ] {
                let mut invalid = bytes;
                write_u16(&mut invalid, node_offset + 2, flags);
                assert!(
                    matches!(Graph::parse(&invalid[..len]), Err(AxError::InvalidNodeFlags(found)) if found == flags)
                );
            }
            let mut invalid = bytes;
            write_u16(&mut invalid, 6, 0);
            assert!(matches!(
                Graph::parse(&invalid[..len]),
                Err(AxError::InvalidVersionForNode {
                    op: OpCode::VOLUME_TRILINEAR_RGB,
                    minor: 0
                })
            ));
            for (offset, count) in [(8, 2), (10, 2)] {
                let mut invalid = bytes;
                write_u16(&mut invalid, node_offset + offset, count);
                assert!(matches!(
                    Graph::parse(&invalid[..len]),
                    Err(AxError::InvalidArity {
                        op: OpCode::VOLUME_TRILINEAR_RGB,
                        ..
                    })
                ));
            }
            let attributes_offset = read_u32(&bytes, 56).unwrap() as usize;
            for (offset, value) in [(0, 2), (1, 1), (2, 3), (3, 1), (12, 0)] {
                let mut invalid = bytes;
                invalid[attributes_offset + offset] = value;
                assert!(matches!(
                    Graph::parse(&invalid[..len]),
                    Err(AxError::InvalidNodeAttributes(OpCode::VOLUME_TRILINEAR_RGB))
                ));
            }
            let mut invalid = bytes;
            write_u32(
                &mut invalid,
                attributes_offset + 20,
                VolumeRgbParams::MAX_DIMENSION + 1,
            );
            assert!(matches!(
                Graph::parse(&invalid[..len]),
                Err(AxError::InvalidNodeAttributes(OpCode::VOLUME_TRILINEAR_RGB))
            ));
        }
    }

    #[test]
    fn kv_cache_nodes_enforce_exact_effects_arity_attributes_and_minor_version() {
        use crate::kv::{
            ATTENTION_DECODE_NODE_FLAGS, AttentionDecodeParams, KV_APPEND_NODE_FLAGS,
            KvCacheParams, KvCacheStorage, ResourceKey, encode_attention_decode_attributes,
            encode_kv_append_attributes,
        };

        let cache = KvCacheParams::new(
            ResourceKey {
                group: 0,
                binding: 1,
            },
            1,
            4,
            8,
            4,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        let append_attributes = encode_kv_append_attributes(cache).unwrap();
        let read_attributes =
            encode_attention_decode_attributes(AttentionDecodeParams::new(cache, 2, 0.5).unwrap())
                .unwrap();
        assert_eq!((OpCode::KV_APPEND.0, OpCode::ATTENTION_DECODE.0), (70, 71));
        let tensors = [arity_tensors()[0]; 6];
        let references = [
            TensorId(0),
            TensorId(1),
            TensorId(2),
            TensorId(3),
            TensorId(4),
        ];
        for (op, flags, attributes) in [
            (
                OpCode::KV_APPEND,
                KV_APPEND_NODE_FLAGS,
                &append_attributes[..],
            ),
            (
                OpCode::ATTENTION_DECODE,
                ATTENTION_DECODE_NODE_FLAGS,
                &read_attributes[..],
            ),
        ] {
            let node = NodeSpec {
                op,
                flags,
                inputs: &references[..3],
                outputs: &[TensorId(5)],
                attributes,
            };
            for (inputs, outputs) in [(2, 1), (4, 1), (3, 0), (3, 2)] {
                assert_eq!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            inputs: &references[..inputs],
                            outputs: &references[3..3 + outputs],
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidArity {
                        op,
                        inputs: inputs as u16,
                        outputs: outputs as u16
                    })
                );
            }
            // Effects are exact: never unordered, mutating, or the other
            // cache node's read/write direction.
            for wrong in [
                NodeFlags::default(),
                NodeFlags::ORDERED,
                NodeFlags(flags.0 & !NodeFlags::ORDERED.0),
                NodeFlags(flags.0 | NodeFlags::MUTATING.0),
                NodeFlags(
                    NodeFlags::ORDERED.0
                        | NodeFlags::READS_RESOURCE.0
                        | NodeFlags::WRITES_RESOURCE.0,
                ),
                if op == OpCode::KV_APPEND {
                    ATTENTION_DECODE_NODE_FLAGS
                } else {
                    KV_APPEND_NODE_FLAGS
                },
            ] {
                assert_eq!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            flags: wrong,
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidNodeFlags(wrong.0))
                );
            }
            let other = if op == OpCode::KV_APPEND {
                &read_attributes[..]
            } else {
                &append_attributes[..]
            };
            for attributes in [&[][..], &attributes[..attributes.len() - 1], other] {
                assert_eq!(
                    encoded_len(&tensors, &[NodeSpec { attributes, ..node }]),
                    Err(AxError::InvalidNodeAttributes(op))
                );
            }

            let mut bytes = [0_u8; 512];
            let len = encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), 4);
            assert!(
                !Graph::parse(&bytes[..len])
                    .unwrap()
                    .node(0)
                    .unwrap()
                    .cacheable()
            );
            for minor in 0..4 {
                let mut invalid = bytes;
                write_u16(&mut invalid, 6, minor);
                assert!(matches!(
                    Graph::parse(&invalid[..len]),
                    Err(AxError::InvalidVersionForNode { op: found, minor: m })
                        if found == op && m == minor
                ));
            }
            let node_offset = read_u32(&bytes, 44).unwrap() as usize;
            let mut invalid = bytes;
            write_u16(&mut invalid, node_offset + 2, NodeFlags::ORDERED.0);
            assert!(matches!(
                Graph::parse(&invalid[..len]),
                Err(AxError::InvalidNodeFlags(found)) if found == NodeFlags::ORDERED.0
            ));
        }

        // Only raster nodes and cache appends may write a resource.
        let pure = NodeSpec {
            op: OpCode::IDENTITY,
            flags: NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::WRITES_RESOURCE.0),
            inputs: &references[..1],
            outputs: &[TensorId(5)],
            attributes: &[],
        };
        assert_eq!(
            encoded_len(&tensors, &[pure]),
            Err(AxError::InvalidNodeFlags(pure.flags.0))
        );
    }

    /// Only a fused-epilogue record requires 1.8 (#107). A matmul without
    /// one keeps its 1.3 encoding, and the 1.6 convolution record keeps its
    /// own, byte for byte.
    #[test]
    fn fused_epilogue_records_require_minor_eight() {
        use crate::recurrent::{
            CausalConv1dParams, EpilogueActivation, RECURRENT_NODE_FLAGS, ResourceKey,
            StateBinding, encode_causal_conv1d_activated_attributes,
            encode_causal_conv1d_attributes,
        };
        use crate::tensor_ops::encode_matmul_epilogue_attributes;

        let tensors = [arity_tensors()[0]; 6];
        let references = [TensorId(0), TensorId(1), TensorId(2)];
        let epilogue = encode_matmul_epilogue_attributes(0.5, EpilogueActivation::Sigmoid).unwrap();
        for (attributes, minor) in [(&epilogue[..], 8), (&[][..], 3)] {
            let node = NodeSpec {
                op: OpCode::MATMUL_TRANSPOSED,
                flags: NodeFlags::default(),
                inputs: &references[..2],
                outputs: &[TensorId(4)],
                attributes,
            };
            let mut bytes = [0_u8; 512];
            encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), minor, "matmul epilogue");
        }

        let state = StateBinding {
            source: ResourceKey {
                group: 4,
                binding: 0,
            },
            target: ResourceKey {
                group: 4,
                binding: 1,
            },
        };
        let plain =
            encode_causal_conv1d_attributes(CausalConv1dParams::new(state, 1, 2).unwrap()).unwrap();
        let activated = encode_causal_conv1d_activated_attributes(
            CausalConv1dParams::new_activated(state, 1, 2, EpilogueActivation::Silu).unwrap(),
        )
        .unwrap();
        for (attributes, minor) in [(&activated[..], 8), (&plain[..], 6)] {
            let node = NodeSpec {
                op: OpCode::CAUSAL_CONV1D,
                flags: RECURRENT_NODE_FLAGS,
                inputs: &references[..3],
                outputs: &[TensorId(4)],
                attributes,
            };
            let mut bytes = [0_u8; 512];
            encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), minor, "conv1d activation");
        }
    }

    #[test]
    fn decoder_operations_enforce_arity_attributes_effects_and_minor_version() {
        use crate::tensor_ops::{
            encode_rms_norm_attributes, encode_top_k_attributes, encode_view_attributes,
        };

        let view = encode_view_attributes(0);
        let top_k = encode_top_k_attributes(1).unwrap();
        let tensors = [arity_tensors()[0]; 6];
        let references = [
            TensorId(0),
            TensorId(1),
            TensorId(2),
            TensorId(3),
            TensorId(4),
        ];
        assert_eq!(
            (
                OpCode::SILU.0,
                OpCode::GATHER.0,
                OpCode::VIEW.0,
                OpCode::TOP_K.0
            ),
            (72, 73, 74, 75)
        );
        for (op, inputs, outputs, attributes, wrong) in [
            (OpCode::SILU, 1, 1, &[][..], &[0_u8][..]),
            (OpCode::GATHER, 2, 1, &[][..], &[0_u8][..]),
            (OpCode::VIEW, 1, 1, &view[..], &view[..7]),
            (OpCode::TOP_K, 1, 2, &top_k[..], &[0_u8; 4][..]),
        ] {
            let node = NodeSpec {
                op,
                flags: NodeFlags::default(),
                inputs: &references[..inputs],
                outputs: &[TensorId(4), TensorId(5)][..outputs],
                attributes,
            };
            for (inputs, outputs) in [(inputs + 1, outputs), (inputs, outputs + 1)] {
                assert!(matches!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            inputs: &references[..inputs],
                            outputs: &references[..outputs],
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidArity { .. })
                ));
            }
            assert_eq!(
                encoded_len(
                    &tensors,
                    &[NodeSpec {
                        attributes: wrong,
                        ..node
                    }]
                ),
                Err(AxError::InvalidNodeAttributes(op))
            );
            for flags in [NodeFlags::READS_RESOURCE, NodeFlags::WRITES_RESOURCE] {
                assert_eq!(
                    encoded_len(&tensors, &[NodeSpec { flags, ..node }]),
                    Err(AxError::InvalidNodeFlags(flags.0))
                );
            }
            let mut bytes = [0_u8; 512];
            let len = encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), 5);
            write_u16(&mut bytes, 6, 4);
            assert!(matches!(
                Graph::parse(&bytes[..len]),
                Err(AxError::InvalidVersionForNode { op: found, minor: 4 }) if found == op
            ));
        }

        // Only the axis-aware RMS_NORM record requires 1.5; the whole-tensor
        // record keeps its 1.3 encoding byte for byte.
        let axis = encode_rms_norm_attributes(1.0e-5, -1).unwrap();
        for (attributes, minor) in [(&axis[..], 5), (&axis[..4], 3)] {
            let node = NodeSpec {
                op: OpCode::RMS_NORM,
                flags: NodeFlags::default(),
                inputs: &references[..2],
                outputs: &[TensorId(4)],
                attributes,
            };
            let mut bytes = [0_u8; 512];
            encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), minor);
        }
    }

    /// SOFTPLUS is a pure 1.22 unary: one input, one output, no record, no
    /// resource effect.
    #[test]
    fn softplus_is_a_pure_unary_of_minor_twenty_two() {
        let tensors = [arity_tensors()[0]; 3];
        let node = NodeSpec {
            op: OpCode::SOFTPLUS,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(1)],
            attributes: &[],
        };
        assert_eq!(OpCode::SOFTPLUS.0, 94);
        assert!(matches!(
            encoded_len(
                &tensors,
                &[NodeSpec {
                    inputs: &[TensorId(0), TensorId(2)],
                    ..node
                }]
            ),
            Err(AxError::InvalidArity { .. })
        ));
        assert_eq!(
            encoded_len(
                &tensors,
                &[NodeSpec {
                    attributes: &[0],
                    ..node
                }]
            ),
            Err(AxError::InvalidNodeAttributes(OpCode::SOFTPLUS))
        );
        for flags in [NodeFlags::READS_RESOURCE, NodeFlags::WRITES_RESOURCE] {
            assert_eq!(
                encoded_len(&tensors, &[NodeSpec { flags, ..node }]),
                Err(AxError::InvalidNodeFlags(flags.0))
            );
        }
        let mut bytes = [0_u8; 512];
        let len = encode(&mut bytes, &tensors, &[node]).unwrap();
        assert_eq!(read_u16(&bytes, 6).unwrap(), 22);
        write_u16(&mut bytes, 6, 21);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::InvalidVersionForNode {
                op: OpCode::SOFTPLUS,
                minor: 21
            })
        ));
    }

    #[test]
    fn recurrent_nodes_enforce_exact_effects_arity_attributes_and_minor_version() {
        use crate::recurrent::{
            CausalConv1dParams, GatedDeltaRuleParams, RECURRENT_NODE_FLAGS, ResourceKey,
            SsmScanParams, StateBinding, encode_causal_conv1d_attributes,
            encode_gated_delta_rule_attributes, encode_ssm_scan_attributes,
        };

        let state = StateBinding {
            source: ResourceKey {
                group: 0,
                binding: 0,
            },
            target: ResourceKey {
                group: 0,
                binding: 1,
            },
        };
        let conv =
            encode_causal_conv1d_attributes(CausalConv1dParams::new(state, 1, 2).unwrap()).unwrap();
        let scan = encode_ssm_scan_attributes(
            SsmScanParams::new(state, 1, 1, 1, 1, 0.0, f32::INFINITY).unwrap(),
        )
        .unwrap();
        let rule = encode_gated_delta_rule_attributes(
            GatedDeltaRuleParams::new(state, (1, 2), (2, 2), true).unwrap(),
        )
        .unwrap();
        assert_eq!(
            (
                OpCode::CAUSAL_CONV1D.0,
                OpCode::SSM_SCAN.0,
                OpCode::GATED_DELTA_RULE.0
            ),
            (76, 77, 93)
        );
        let tensors = [arity_tensors()[0]; 6];
        let references = [TensorId(0), TensorId(1), TensorId(2), TensorId(3)];
        for (op, attributes, other, minor) in [
            (OpCode::CAUSAL_CONV1D, &conv[..], &scan[..], 6),
            (OpCode::SSM_SCAN, &scan[..], &conv[..], 6),
            (OpCode::GATED_DELTA_RULE, &rule[..], &scan[..], 22),
        ] {
            let node = NodeSpec {
                op,
                flags: RECURRENT_NODE_FLAGS,
                inputs: &references[..3],
                outputs: &[TensorId(5)],
                attributes,
            };
            for inputs in [2, 4] {
                assert!(matches!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            inputs: &references[..inputs],
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidArity { .. })
                ));
            }
            for wrong in [
                NodeFlags::default(),
                NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::READS_RESOURCE.0),
                NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::WRITES_RESOURCE.0),
                NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::WRITES_RESOURCE.0),
                NodeFlags(RECURRENT_NODE_FLAGS.0 | NodeFlags::MUTATING.0),
            ] {
                assert_eq!(
                    encoded_len(
                        &tensors,
                        &[NodeSpec {
                            flags: wrong,
                            ..node
                        }]
                    ),
                    Err(AxError::InvalidNodeFlags(wrong.0))
                );
            }
            for attributes in [&[][..], &attributes[..attributes.len() - 1], other] {
                assert_eq!(
                    encoded_len(&tensors, &[NodeSpec { attributes, ..node }]),
                    Err(AxError::InvalidNodeAttributes(op))
                );
            }
            let mut bytes = [0_u8; 512];
            let len = encode(&mut bytes, &tensors, &[node]).unwrap();
            assert_eq!(read_u16(&bytes, 6).unwrap(), minor);
            assert!(
                !Graph::parse(&bytes[..len])
                    .unwrap()
                    .node(0)
                    .unwrap()
                    .cacheable()
            );
            write_u16(&mut bytes, 6, minor - 1);
            assert!(matches!(
                Graph::parse(&bytes[..len]),
                Err(AxError::InvalidVersionForNode { op: found, minor: m })
                    if found == op && m == minor - 1
            ));
        }
    }

    #[test]
    fn expert_dispatch_enforces_its_read_effect_arity_attributes_and_minor_version() {
        use crate::moe::{ExpertActivation, MoeParams, ResourceKey, encode_moe_attributes};

        let attributes = encode_moe_attributes(
            MoeParams::new(
                ResourceKey {
                    group: 0,
                    binding: 0,
                },
                4,
                2,
                1,
                1,
                DType::B8,
                ExpertActivation::SquaredRelu,
                true,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(OpCode::MOE_EXPERTS.0, 78);
        let tensors = [arity_tensors()[0]; 6];
        let references = [TensorId(0), TensorId(1), TensorId(2), TensorId(3)];
        let node = NodeSpec {
            op: OpCode::MOE_EXPERTS,
            flags: NodeFlags::READS_RESOURCE,
            inputs: &references[..3],
            outputs: &[TensorId(5)],
            attributes: &attributes,
        };
        for inputs in [2, 4] {
            assert!(matches!(
                encoded_len(
                    &tensors,
                    &[NodeSpec {
                        inputs: &references[..inputs],
                        ..node
                    }]
                ),
                Err(AxError::InvalidArity { .. })
            ));
        }
        for wrong in [
            NodeFlags::default(),
            NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::ORDERED.0),
            NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::WRITES_RESOURCE.0),
            NodeFlags(NodeFlags::READS_RESOURCE.0 | NodeFlags::MUTATING.0),
        ] {
            assert_eq!(
                encoded_len(
                    &tensors,
                    &[NodeSpec {
                        flags: wrong,
                        ..node
                    }]
                ),
                Err(AxError::InvalidNodeFlags(wrong.0))
            );
        }
        for attributes in [&[][..], &attributes[..35]] {
            assert_eq!(
                encoded_len(&tensors, &[NodeSpec { attributes, ..node }]),
                Err(AxError::InvalidNodeAttributes(OpCode::MOE_EXPERTS))
            );
        }
        let mut bytes = [0_u8; 512];
        let len = encode(&mut bytes, &tensors, &[node]).unwrap();
        assert_eq!(read_u16(&bytes, 6).unwrap(), 7);
        write_u16(&mut bytes, 6, 6);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::InvalidVersionForNode {
                op: OpCode::MOE_EXPERTS,
                minor: 6
            })
        ));
    }

    #[test]
    fn reads_flag_is_reserved_for_resource_operations_and_extensions() {
        let tensors = arity_tensors();
        let mut bytes = [0_u8; 512];
        let pure = NodeSpec {
            op: OpCode::IDENTITY,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(4)],
            attributes: &[],
        };
        assert!(matches!(
            encoded_len(
                &tensors,
                &[NodeSpec {
                    flags: NodeFlags::READS_RESOURCE,
                    ..pure
                }]
            ),
            Err(AxError::InvalidNodeFlags(_))
        ));
        let len = encode(&mut bytes, &tensors, &[pure]).unwrap();
        let node_offset = read_u32(&bytes, 44).unwrap() as usize;
        write_u16(&mut bytes, 6, 1);
        write_u16(&mut bytes, node_offset + 2, NodeFlags::READS_RESOURCE.0);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::InvalidNodeFlags(_))
        ));

        let extension = NodeSpec {
            op: OpCode(60_000),
            flags: NodeFlags::READS_RESOURCE,
            attributes: &[9],
            ..pure
        };
        let len = encode(&mut bytes, &tensors, &[extension]).unwrap();
        assert_eq!(read_u16(&bytes, 6).unwrap(), 1);
        assert!(
            !Graph::parse(&bytes[..len])
                .unwrap()
                .node(0)
                .unwrap()
                .cacheable()
        );
        write_u16(&mut bytes, 6, 0);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::InvalidVersionForNode {
                op: OpCode(60_000),
                minor: 0
            })
        ));

        let interleave = NodeSpec {
            op: OpCode::INTERLEAVE3_F32,
            inputs: &[TensorId(0), TensorId(1), TensorId(2)],
            ..pure
        };
        let len = encode(&mut bytes, &tensors, &[interleave]).unwrap();
        assert_eq!(read_u16(&bytes, 6).unwrap(), 1);
        write_u16(&mut bytes, 6, 0);
        assert!(matches!(
            Graph::parse(&bytes[..len]),
            Err(AxError::InvalidVersionForNode {
                op: OpCode::INTERLEAVE3_F32,
                minor: 0
            })
        ));
    }
}
