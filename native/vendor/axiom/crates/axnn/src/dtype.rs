/// Data type of a tensor.
///
/// Naming convention: `{category}{bits}` — short, unambiguous, greppable.
///
/// | Prefix | Meaning                          |
/// |--------|----------------------------------|
/// | `F`    | IEEE 754 floating-point           |
/// | `B`    | Brain float (truncated mantissa)  |
/// | `I`    | Signed integer                    |
/// | `U`    | Unsigned integer                  |
/// | `Q`    | Block-quantized                   |
///
/// Scalar types (F/B/I/U) are block_size=1, zero scale overhead.
/// Quantized types (Q) have multi-element blocks with packed values
/// and scale metadata. The unified block API (`block_size()`,
/// `block_byte_size()`, etc.) works identically for all variants —
/// scalar types are just degenerate blocks.
///
/// ## Quantized format details
///
/// Only one format per bit-width, chosen for the best quality-to-size
/// ratio and hardware compatibility:
///
/// - **Q8**: Q8_0 — symmetric INT8 with FP16 per-block scale (block=32)
/// - **Q6**: Q6_K — 6-bit K-quant with super-block structure (block=256)
/// - **Q4**: Q4_K_M — 4-bit K-quant medium with min offsets (block=256)
///
/// The enum is 1 byte (`#[repr(u8)]`). Block parameters are computed
/// by `const fn` methods, not stored as fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    // ── Floating-point ──────────────────────────────────────────
    /// 64-bit IEEE 754 double-precision float.
    F64,
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit brain float (8-bit exponent, 7-bit mantissa).
    B16,

    // ── Integer ─────────────────────────────────────────────────
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// 16-bit signed integer.
    I16,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,

    // ── Block-quantized ─────────────────────────────────────────
    /// Q8_0: 8-bit symmetric, FP16 per-block scale.
    /// Block: 32 × i8 (32 B) + 1 × f16 scale (2 B) = 34 B.
    /// Effective: 8.5 bits/element.
    Q8,
    /// Axiom Q6: 6-bit K-quant with interleaved lane packing.
    /// Canonical GGML Q6_K blocks require packed lane conversion.
    /// Block: 256 elements.
    ///   ql: 128 B (low nibbles) + qh: 64 B (high 2-bit) +
    ///   scales: 16 × i8 + d: 1 × f16 = 210 B.
    /// Effective: ~6.56 bits/element.
    Q6,
    /// Axiom Q4: 4-bit K-quant with interleaved lanes and scale packing.
    /// Canonical GGML Q4_K blocks require packed lane and scale conversion.
    /// Block: 256 elements.
    ///   d: f16 + dmin: f16 + scales: 12 B (packed 6-bit) +
    ///   qs: 128 B (packed nibbles) = 144 B.
    /// Effective: 4.5 bits/element.
    Q4,
    /// Q1: 1-bit packed binary. Block: 256 elements = 32 bytes.
    /// No scale/offset — pure bit field. Intended for adjacency
    /// matrices, masks, and binary linear algebra (AND-OR semiring).
    /// Effective: 1.0 bit/element.
    Q1,

    // ── 8-bit floating-point (external per-tensor scale) ────────
    /// 8-bit brain float (B8; OCP E4M3, `torch.float8_e4m3fn`): sign
    /// + 4-bit exponent (bias 7)
    /// + 3-bit mantissa. No infinities; `S.1111.111` is the sole NaN;
    /// max magnitude ±448. A one-byte narrow float (like F16); the
    /// quantization scale is external (per-tensor), not embedded the way
    /// a `Q`-block's scale is.
    B8,
    /// 8-bit IEEE float (F8; OCP E5M2): sign + 5-bit exponent (bias 15)
    /// + 2-bit mantissa, with IEEE-style infinities and NaNs. Exponent width
    /// and bias match F16, so E5M2 → F16 is an exact `<< 8`. One byte;
    /// external scale.
    F8,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::B16 => "B16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Q8 => "Q8",
            Self::Q6 => "Q6",
            Self::Q4 => "Q4",
            Self::Q1 => "Q1",
            Self::B8 => "B8",
            Self::F8 => "F8",
        };
        f.write_str(name)
    }
}

impl DType {
    /// Elements per quantization block.
    ///
    /// For scalar types this is 1: each element is its own "block."
    /// This is the divisor for converting a logical element count
    /// to a physical block count along the quantization axis.
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F64 | Self::F32 | Self::F16 | Self::B16 => 1,
            Self::B8 | Self::F8 => 1,
            Self::I64 | Self::I32 | Self::I16 | Self::I8 | Self::U8 => 1,
            Self::Q8 => 32,
            Self::Q6 | Self::Q4 | Self::Q1 => 256,
        }
    }

    /// Total bytes per block, including packed values and any scale overhead.
    ///
    /// For scalar types, this is just the element byte size.
    /// For quantized types, this includes packed values + scales + metadata.
    pub const fn block_byte_size(&self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::B16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
            Self::B8 | Self::F8 => 1,
            Self::Q8 => 34,  // 32 × i8 + 1 × f16
            Self::Q6 => 210, // 128 + 64 + 16 + 2
            Self::Q4 => 144, // 2 + 2 + 12 + 128
            Self::Q1 => 32,  // 256 bits packed, no scale
        }
    }

    /// Bytes consumed by scale/metadata per block (zero for scalar types).
    pub const fn scale_overhead_bytes(&self) -> usize {
        match self {
            Self::F64 | Self::F32 | Self::F16 | Self::B16 => 0,
            Self::B8 | Self::F8 => 0,
            Self::I64 | Self::I32 | Self::I16 | Self::I8 | Self::U8 => 0,
            Self::Q8 => 2,  // 1 × f16 scale
            Self::Q6 => 18, // 16 × i8 sub-scales + 1 × f16 d
            Self::Q4 => 16, // f16 d + f16 dmin + 12 packed scale/min
            Self::Q1 => 0,  // no scale metadata
        }
    }

    // ── Category predicates ─────────────────────────────────────

    /// Any floating-point type (IEEE or brain float).
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F64 | Self::F32 | Self::F16 | Self::B16 | Self::B8 | Self::F8
        )
    }

    /// 8-bit floating-point types (B8 / F8; OCP E4M3 / E5M2).
    ///
    /// A subset of [`is_float`](Self::is_float): narrow floats decoded
    /// per-element with an external (per-tensor) scale — distinct from the
    /// block-embedded scale of the `Q` types.
    pub const fn is_fp8(&self) -> bool {
        matches!(self, Self::B8 | Self::F8)
    }

    /// Any integer type (signed or unsigned).
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::I64 | Self::I32 | Self::I16 | Self::I8 | Self::U8
        )
    }

    /// Any block-quantized type (including binary).
    pub const fn is_quantized(&self) -> bool {
        matches!(self, Self::Q8 | Self::Q6 | Self::Q4 | Self::Q1)
    }

    /// Scaled block-quantized types (Q8/Q6/Q4) — supports dequant↔F32.
    /// Excludes Q1 which is a raw bit field with no scale.
    pub const fn is_scaled_quant(&self) -> bool {
        matches!(self, Self::Q8 | Self::Q6 | Self::Q4)
    }

    /// Binary packed type (Q1). Pure bit field, no scale metadata.
    pub const fn is_binary(&self) -> bool {
        matches!(self, Self::Q1)
    }

    // ── Promotion / accumulation ────────────────────────────────

    /// The natural accumulator type for reductions and dot products.
    ///
    /// This is the type that intermediate results widen into during
    /// compute — not a device property, but a format property:
    ///
    /// - Floats accumulate in F32 (half/brain are too narrow).
    /// - Integers and quantized types accumulate in I32
    ///   (VNNI `vpdpbusd`, NPU MAC units, GPU shader ints — all I32).
    /// - F32 and I32 are already full-width; they accumulate as themselves.
    pub const fn accumulator(&self) -> DType {
        match self {
            Self::F64 => Self::F64,
            Self::F32 => Self::F32,
            Self::F16 | Self::B16 | Self::B8 | Self::F8 => Self::F32,

            Self::I64 => Self::I64,
            Self::I32 => Self::I32,
            Self::I16 | Self::I8 | Self::U8 => Self::I32,

            Self::Q8 | Self::Q6 | Self::Q4 => Self::I32,
            Self::Q1 => Self::U8, // binary ops, no arithmetic accumulation
        }
    }

    /// Number of physical blocks needed to store `num_elements` along
    /// the quantization axis. Rounds up (partial blocks are padded).
    pub const fn num_blocks(&self, num_elements: usize) -> usize {
        let bs = self.block_size();
        (num_elements + bs - 1) / bs
    }

    /// Total storage bytes for `num_elements` along the quantization axis.
    pub const fn storage_bytes(&self, num_elements: usize) -> usize {
        self.num_blocks(num_elements) * self.block_byte_size()
    }

    /// Convert a logical element shape to a physical block shape.
    ///
    /// The innermost dimension is divided by `block_size` (rounded up).
    /// All other dimensions are unchanged. For scalar types (block_size=1),
    /// the shape is returned unmodified.
    ///
    /// This is the shape that the `Layout` operates over.
    pub fn block_shape(&self, shape: &crate::shape::Shape) -> crate::shape::Shape {
        let bs = self.block_size();
        if bs == 1 || shape.rank() == 0 {
            return *shape;
        }
        let rank = shape.rank();
        let mut dims = [0usize; crate::MAX_RANK];
        dims[..rank].copy_from_slice(shape.dims());
        dims[rank - 1] = self.num_blocks(dims[rank - 1]);
        crate::shape::Shape::from_parts(dims, rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    // ── Block parameters ────────────────────────────────────────

    #[test]
    fn scalar_types_are_single_element_blocks() {
        for dt in [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::B16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U8,
            DType::B8,
            DType::F8,
        ] {
            assert_eq!(dt.block_size(), 1, "{dt} should have block_size=1");
            assert_eq!(
                dt.scale_overhead_bytes(),
                0,
                "{dt} should have 0 scale overhead"
            );
        }
    }

    #[test]
    fn quantized_block_sizes() {
        assert_eq!(DType::Q8.block_size(), 32);
        assert_eq!(DType::Q6.block_size(), 256);
        assert_eq!(DType::Q4.block_size(), 256);
        assert_eq!(DType::Q1.block_size(), 256);
    }

    #[test]
    fn quantized_block_byte_sizes() {
        assert_eq!(DType::Q8.block_byte_size(), 34);
        assert_eq!(DType::Q6.block_byte_size(), 210);
        assert_eq!(DType::Q4.block_byte_size(), 144);
        assert_eq!(DType::Q1.block_byte_size(), 32);
    }

    #[test]
    fn scalar_byte_sizes() {
        assert_eq!(DType::F64.block_byte_size(), 8);
        assert_eq!(DType::F32.block_byte_size(), 4);
        assert_eq!(DType::F16.block_byte_size(), 2);
        assert_eq!(DType::I8.block_byte_size(), 1);
    }

    // ── Category predicates ─────────────────────────────────────

    #[test]
    fn categories_are_mutually_exclusive() {
        let all = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::B16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U8,
            DType::Q8,
            DType::Q6,
            DType::Q4,
            DType::Q1,
            DType::B8,
            DType::F8,
        ];
        for dt in all {
            let count = dt.is_float() as u8 + dt.is_integer() as u8 + dt.is_quantized() as u8;
            assert_eq!(count, 1, "{dt} should belong to exactly one category");
        }
    }

    // ── Accumulator ─────────────────────────────────────────────

    #[test]
    fn narrow_floats_accumulate_to_f32() {
        assert_eq!(DType::F16.accumulator(), DType::F32);
        assert_eq!(DType::B16.accumulator(), DType::F32);
    }

    #[test]
    fn wide_types_accumulate_as_themselves() {
        assert_eq!(DType::F64.accumulator(), DType::F64);
        assert_eq!(DType::F32.accumulator(), DType::F32);
        assert_eq!(DType::I64.accumulator(), DType::I64);
        assert_eq!(DType::I32.accumulator(), DType::I32);
    }

    #[test]
    fn quantized_accumulates_to_i32() {
        assert_eq!(DType::Q8.accumulator(), DType::I32);
        assert_eq!(DType::Q6.accumulator(), DType::I32);
        assert_eq!(DType::Q4.accumulator(), DType::I32);
    }

    #[test]
    fn q1_binary_properties() {
        assert!(DType::Q1.is_quantized());
        assert!(DType::Q1.is_binary());
        assert!(!DType::Q1.is_scaled_quant());
        assert_eq!(DType::Q1.block_size(), 256);
        assert_eq!(DType::Q1.block_byte_size(), 32);
        assert_eq!(DType::Q1.scale_overhead_bytes(), 0);
        assert_eq!(DType::Q1.accumulator(), DType::U8);
        // 256 elements = 1 block = 32 bytes
        assert_eq!(DType::Q1.storage_bytes(256), 32);
        // 512 elements = 2 blocks = 64 bytes
        assert_eq!(DType::Q1.storage_bytes(512), 64);
    }

    #[test]
    fn f8_properties() {
        for dt in [DType::B8, DType::F8] {
            assert!(dt.is_float(), "{dt} is a float");
            assert!(dt.is_fp8(), "{dt} is an 8-bit float");
            assert!(!dt.is_integer());
            assert!(!dt.is_quantized());
            assert_eq!(dt.block_size(), 1);
            assert_eq!(dt.block_byte_size(), 1, "{dt} is one byte");
            assert_eq!(dt.scale_overhead_bytes(), 0);
            assert_eq!(dt.accumulator(), DType::F32);
            assert_eq!(dt.storage_bytes(100), 100);
        }
    }

    // ── Block count & storage ───────────────────────────────────

    #[test]
    fn num_blocks_rounds_up() {
        assert_eq!(DType::Q8.num_blocks(32), 1); // exact
        assert_eq!(DType::Q8.num_blocks(33), 2); // one over → 2 blocks
        assert_eq!(DType::Q8.num_blocks(1), 1); // partial → still 1
    }

    #[test]
    fn storage_bytes_scalar() {
        assert_eq!(DType::F32.storage_bytes(100), 400); // 100 × 4B
    }

    #[test]
    fn storage_bytes_quantized() {
        // 256 elements = 1 block of Q4 = 144 bytes
        assert_eq!(DType::Q4.storage_bytes(256), 144);
        // 512 elements = 2 blocks = 288 bytes
        assert_eq!(DType::Q4.storage_bytes(512), 288);
    }

    // ── Block shape ─────────────────────────────────────────────

    #[test]
    fn block_shape_scalar_unchanged() {
        let s = shape![2, 3, 4];
        assert_eq!(DType::F32.block_shape(&s), s);
    }

    #[test]
    fn block_shape_quantized_divides_innermost() {
        let s = shape![2, 3, 256];
        let bs = DType::Q4.block_shape(&s);
        assert_eq!(bs.dims(), &[2, 3, 1]); // 256/256 = 1 block
    }

    #[test]
    fn block_shape_quantized_rounds_up() {
        let s = shape![2, 3, 300];
        let bs = DType::Q4.block_shape(&s);
        assert_eq!(bs.dims(), &[2, 3, 2]); // ceil(300/256) = 2
    }

    // ── Display ─────────────────────────────────────────────────

    #[test]
    fn display_names() {
        use alloc::format;
        assert_eq!(format!("{}", DType::Q4), "Q4");
        assert_eq!(format!("{}", DType::B16), "B16");
        assert_eq!(format!("{}", DType::F64), "F64");
        assert_eq!(format!("{}", DType::B8), "B8");
        assert_eq!(format!("{}", DType::F8), "F8");
    }
}
