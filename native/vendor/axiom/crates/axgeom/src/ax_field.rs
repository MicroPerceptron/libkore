//! Typed, allocator-owned field-plane construction for `.ax` graphs.
//!
//! A plane has one statically ranked multidimensional field shape.  Every
//! tensor handle records whether it is a broadcast scalar or a full field,
//! along with its data type and graph role.  Elementwise operations derive
//! their output shape from their operands; callers cannot accidentally ask a
//! scalar-scalar operation or copy to manufacture a field-shaped result.

use core::alloc::Allocator;

use ax::{
    AllocationError, DType, GraphBuf, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId,
};

use crate::algebra::RealOps;

/// Logical extent carried by a tensor in a field plane.
///
/// Scalars are encoded with rank `D` and unit extents so TOSA's rank-equal
/// elementwise broadcasting remains explicit and portable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueShape {
    Scalar,
    Field,
}

/// How a tensor participates in the graph ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRole {
    Input,
    Output,
    Constant,
    Temporary,
}

/// A real-valued scalar or field tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RealValue {
    tensor: TensorId,
    dtype: DType,
    shape: ValueShape,
    role: ValueRole,
}

impl RealValue {
    pub const fn tensor(self) -> TensorId {
        self.tensor
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn shape(self) -> ValueShape {
        self.shape
    }

    pub const fn role(self) -> ValueRole {
        self.role
    }
}

/// A boolean (`U8`) scalar or field tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskValue {
    tensor: TensorId,
    shape: ValueShape,
    role: ValueRole,
}

impl MaskValue {
    pub const fn tensor(self) -> TensorId {
        self.tensor
    }

    pub const fn shape(self) -> ValueShape {
        self.shape
    }

    pub const fn role(self) -> ValueRole {
        self.role
    }
}

/// Errors detected by the typed field layer or the underlying `.ax` builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxFieldError {
    Allocation(AllocationError),
    UnsupportedRealDType(DType),
    InvalidExtent {
        axis: usize,
    },
    ElementCountOverflow,
    InvalidConstantBytes {
        expected: usize,
        actual: usize,
    },
    DTypeMismatch {
        left: DType,
        right: DType,
    },
    ShapeMismatch {
        expected: ValueShape,
        actual: ValueShape,
    },
    ExpectedScalar,
    ExpectedField,
    ExpectedOutput,
}

impl From<AllocationError> for AxFieldError {
    fn from(value: AllocationError) -> Self {
        Self::Allocation(value)
    }
}

/// Builds one statically ranked, multidimensional `.ax` field plane.
///
/// The supplied allocator is moved into [`GraphBuilder`].  Shape metadata is
/// stored inline, and no production path uses `Global`, host services, or a
/// provider runtime.  `compute_dtype` is the type used for constants emitted
/// through [`RealOps`]; explicit [`cast`](Self::cast) nodes allow a caller to
/// form mixed-precision regions without weakening the type checks.
pub struct AxFieldPlane<A: Allocator, const D: usize> {
    builder: GraphBuilder<A>,
    dimensions: [u32; D],
    scalar_dimensions: [u32; D],
    compute_dtype: DType,
}

impl<A: Allocator + Clone, const D: usize> AxFieldPlane<A, D> {
    /// Creates an empty field plane without allocating.
    pub fn new_in(
        dimensions: [u32; D],
        compute_dtype: DType,
        allocator: A,
    ) -> Result<Self, AxFieldError> {
        validate_real_dtype(compute_dtype)?;
        for (axis, &extent) in dimensions.iter().enumerate() {
            if extent == 0 {
                return Err(AxFieldError::InvalidExtent { axis });
            }
        }
        Ok(Self {
            builder: GraphBuilder::new_in(allocator),
            dimensions,
            scalar_dimensions: [1; D],
            compute_dtype,
        })
    }

    pub const fn dimensions(&self) -> &[u32; D] {
        &self.dimensions
    }

    pub const fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    /// Adds a full-field input using the plane's compute type.
    pub fn input(&mut self) -> Result<RealValue, AxFieldError> {
        self.input_as(self.compute_dtype)
    }

    /// Adds a full-field input with an explicit real type.
    pub fn input_as(&mut self, dtype: DType) -> Result<RealValue, AxFieldError> {
        self.real_tensor(dtype, ValueShape::Field, ValueRole::Input, &[])
    }

    /// Adds a broadcast-scalar input with an explicit real type.
    pub fn scalar_input_as(&mut self, dtype: DType) -> Result<RealValue, AxFieldError> {
        self.real_tensor(dtype, ValueShape::Scalar, ValueRole::Input, &[])
    }

    /// Adds a full-field output using the plane's compute type.
    pub fn output(&mut self) -> Result<RealValue, AxFieldError> {
        self.output_as(self.compute_dtype)
    }

    /// Adds a full-field output with an explicit real type.
    pub fn output_as(&mut self, dtype: DType) -> Result<RealValue, AxFieldError> {
        self.real_tensor(dtype, ValueShape::Field, ValueRole::Output, &[])
    }

    /// Adds a broadcast-scalar output with an explicit real type.
    pub fn scalar_output_as(&mut self, dtype: DType) -> Result<RealValue, AxFieldError> {
        self.real_tensor(dtype, ValueShape::Scalar, ValueRole::Output, &[])
    }

    /// Adds an unproduced real temporary.  Most callers should let arithmetic
    /// methods create temporaries so their shapes cannot be chosen manually.
    pub fn temporary(
        &mut self,
        dtype: DType,
        shape: ValueShape,
    ) -> Result<RealValue, AxFieldError> {
        self.real_tensor(dtype, shape, ValueRole::Temporary, &[])
    }

    /// Adds a scalar constant in the plane's compute type.
    pub fn scalar(&mut self, value: f32) -> Result<RealValue, AxFieldError> {
        let (bytes, len) = encode_real_constant(value, self.compute_dtype)?;
        self.real_tensor(
            self.compute_dtype,
            ValueShape::Scalar,
            ValueRole::Constant,
            &bytes[..len],
        )
    }

    /// Adds an exactly encoded scalar constant.
    pub fn scalar_constant_raw(
        &mut self,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<RealValue, AxFieldError> {
        validate_real_dtype(dtype)?;
        validate_constant_len(dtype, 1, bytes.len())?;
        self.real_tensor(dtype, ValueShape::Scalar, ValueRole::Constant, bytes)
    }

    /// Adds an exactly encoded full-field constant.
    ///
    /// This API accepts caller-owned bytes so a large field is never silently
    /// materialized or allocated by the geometry layer.
    pub fn field_constant_raw(
        &mut self,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<RealValue, AxFieldError> {
        validate_real_dtype(dtype)?;
        validate_constant_len(dtype, self.element_count()?, bytes.len())?;
        self.real_tensor(dtype, ValueShape::Field, ValueRole::Constant, bytes)
    }

    pub fn mask_input(&mut self) -> Result<MaskValue, AxFieldError> {
        self.mask_tensor(ValueShape::Field, ValueRole::Input)
    }

    pub fn scalar_mask_input(&mut self) -> Result<MaskValue, AxFieldError> {
        self.mask_tensor(ValueShape::Scalar, ValueRole::Input)
    }

    pub fn mask_output(&mut self) -> Result<MaskValue, AxFieldError> {
        self.mask_tensor(ValueShape::Field, ValueRole::Output)
    }

    pub fn scalar_mask_output(&mut self) -> Result<MaskValue, AxFieldError> {
        self.mask_tensor(ValueShape::Scalar, ValueRole::Output)
    }

    pub fn mask_temporary(&mut self, shape: ValueShape) -> Result<MaskValue, AxFieldError> {
        self.mask_tensor(shape, ValueRole::Temporary)
    }

    #[inline]
    pub fn add(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::ADD, left, right)
    }

    #[inline]
    pub fn sub(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::SUB, left, right)
    }

    #[inline]
    pub fn mul(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::MUL, left, right)
    }

    #[inline]
    pub fn minimum(
        &mut self,
        left: RealValue,
        right: RealValue,
    ) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::MINIMUM, left, right)
    }

    #[inline]
    pub fn maximum(
        &mut self,
        left: RealValue,
        right: RealValue,
    ) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::MAXIMUM, left, right)
    }

    #[inline]
    pub fn pow(&mut self, base: RealValue, exponent: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_binary(OpCode::POW, base, exponent)
    }

    #[inline]
    pub fn negate(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::NEGATE, value)
    }

    #[inline]
    pub fn abs(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::ABS, value)
    }

    #[inline]
    pub fn reciprocal(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::RECIPROCAL, value)
    }

    #[inline]
    pub fn rsqrt(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::RSQRT, value)
    }

    #[inline]
    pub fn sin(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::SIN, value)
    }

    #[inline]
    pub fn cos(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::COS, value)
    }

    #[inline]
    pub fn exp(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::EXP, value)
    }

    #[inline]
    pub fn log(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        self.real_unary(OpCode::LOG, value)
    }

    /// Inserts an explicit precision conversion while preserving shape.
    pub fn cast(&mut self, value: RealValue, dtype: DType) -> Result<RealValue, AxFieldError> {
        validate_real_dtype(dtype)?;
        let output = self.real_tensor(dtype, value.shape, ValueRole::Temporary, &[])?;
        self.builder.node(
            OpCode::CAST,
            NodeFlags::default(),
            &[value.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    #[inline]
    pub fn equal(&mut self, left: RealValue, right: RealValue) -> Result<MaskValue, AxFieldError> {
        self.compare(OpCode::EQUAL, left, right)
    }

    #[inline]
    pub fn greater(
        &mut self,
        left: RealValue,
        right: RealValue,
    ) -> Result<MaskValue, AxFieldError> {
        self.compare(OpCode::GREATER, left, right)
    }

    #[inline]
    pub fn greater_equal(
        &mut self,
        left: RealValue,
        right: RealValue,
    ) -> Result<MaskValue, AxFieldError> {
        self.compare(OpCode::GREATER_EQUAL, left, right)
    }

    #[inline]
    pub fn logical_not(&mut self, value: MaskValue) -> Result<MaskValue, AxFieldError> {
        let output = self.mask_tensor(value.shape, ValueRole::Temporary)?;
        self.builder.node(
            OpCode::LOGICAL_NOT,
            NodeFlags::default(),
            &[value.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    #[inline]
    pub fn logical_and(
        &mut self,
        left: MaskValue,
        right: MaskValue,
    ) -> Result<MaskValue, AxFieldError> {
        self.mask_binary(OpCode::LOGICAL_AND, left, right)
    }

    #[inline]
    pub fn logical_or(
        &mut self,
        left: MaskValue,
        right: MaskValue,
    ) -> Result<MaskValue, AxFieldError> {
        self.mask_binary(OpCode::LOGICAL_OR, left, right)
    }

    #[inline]
    pub fn logical_xor(
        &mut self,
        left: MaskValue,
        right: MaskValue,
    ) -> Result<MaskValue, AxFieldError> {
        self.mask_binary(OpCode::LOGICAL_XOR, left, right)
    }

    /// Selects values with scalar-to-field broadcasting as required by any
    /// one of the three operands.
    pub fn select(
        &mut self,
        condition: MaskValue,
        when_true: RealValue,
        when_false: RealValue,
    ) -> Result<RealValue, AxFieldError> {
        require_same_dtype(when_true, when_false)?;
        let arms_shape = broadcast_shape(when_true.shape, when_false.shape);
        let shape = broadcast_shape(condition.shape, arms_shape);
        let output = self.real_tensor(when_true.dtype, shape, ValueRole::Temporary, &[])?;
        self.builder.node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[condition.tensor, when_true.tensor, when_false.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    /// Explicitly broadcasts a scalar into a field using `like` only to carry
    /// the field domain.  Both select arms are the scalar, so the output is
    /// independent of the values (including NaNs) in `like`.
    pub fn splat_like(
        &mut self,
        scalar: RealValue,
        like: RealValue,
    ) -> Result<RealValue, AxFieldError> {
        if scalar.shape != ValueShape::Scalar {
            return Err(AxFieldError::ExpectedScalar);
        }
        if like.shape != ValueShape::Field {
            return Err(AxFieldError::ExpectedField);
        }
        require_same_dtype(scalar, like)?;
        let domain_mask = self.equal(like, like)?;
        self.select(domain_mask, scalar, scalar)
    }

    /// Copies a computed real value into an ABI output.
    ///
    /// Shape equality is strict.  A scalar cannot be copied to a field output;
    /// callers must make the broadcast visible with [`splat_like`](Self::splat_like).
    pub fn write_output(
        &mut self,
        output: RealValue,
        value: RealValue,
    ) -> Result<(), AxFieldError> {
        if output.role != ValueRole::Output {
            return Err(AxFieldError::ExpectedOutput);
        }
        require_same_dtype(output, value)?;
        require_same_shape(output.shape, value.shape)?;
        self.builder.node(
            OpCode::COPY,
            NodeFlags::default(),
            &[value.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(())
    }

    /// Copies a computed mask into an ABI mask output with strict shape
    /// equality.  Logical OR with itself preserves `U8` boolean semantics.
    pub fn write_mask_output(
        &mut self,
        output: MaskValue,
        value: MaskValue,
    ) -> Result<(), AxFieldError> {
        if output.role != ValueRole::Output {
            return Err(AxFieldError::ExpectedOutput);
        }
        require_same_shape(output.shape, value.shape)?;
        self.builder.node(
            OpCode::LOGICAL_OR,
            NodeFlags::default(),
            &[value.tensor, value.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(())
    }

    /// Encodes and validates the graph, preserving allocator ownership.
    pub fn finish(self) -> Result<GraphBuf<A>, AxFieldError> {
        Ok(self.builder.finish()?)
    }

    fn real_tensor(
        &mut self,
        dtype: DType,
        shape: ValueShape,
        role: ValueRole,
        constant: &[u8],
    ) -> Result<RealValue, AxFieldError> {
        validate_real_dtype(dtype)?;
        let flags = tensor_flags(role);
        let dimensions = match shape {
            ValueShape::Scalar => &self.scalar_dimensions,
            ValueShape::Field => &self.dimensions,
        };
        let tensor = self.builder.tensor(dtype, flags, dimensions, constant)?;
        Ok(RealValue {
            tensor,
            dtype,
            shape,
            role,
        })
    }

    fn mask_tensor(
        &mut self,
        shape: ValueShape,
        role: ValueRole,
    ) -> Result<MaskValue, AxFieldError> {
        let flags = tensor_flags(role);
        let dimensions = match shape {
            ValueShape::Scalar => &self.scalar_dimensions,
            ValueShape::Field => &self.dimensions,
        };
        let tensor = self.builder.tensor(DType::U8, flags, dimensions, &[])?;
        Ok(MaskValue {
            tensor,
            shape,
            role,
        })
    }

    fn real_unary(&mut self, op: OpCode, value: RealValue) -> Result<RealValue, AxFieldError> {
        let output = self.real_tensor(value.dtype, value.shape, ValueRole::Temporary, &[])?;
        self.builder.node(
            op,
            NodeFlags::default(),
            &[value.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    fn real_binary(
        &mut self,
        op: OpCode,
        left: RealValue,
        right: RealValue,
    ) -> Result<RealValue, AxFieldError> {
        require_same_dtype(left, right)?;
        let shape = broadcast_shape(left.shape, right.shape);
        let output = self.real_tensor(left.dtype, shape, ValueRole::Temporary, &[])?;
        self.builder.node(
            op,
            NodeFlags::default(),
            &[left.tensor, right.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    fn compare(
        &mut self,
        op: OpCode,
        left: RealValue,
        right: RealValue,
    ) -> Result<MaskValue, AxFieldError> {
        require_same_dtype(left, right)?;
        let shape = broadcast_shape(left.shape, right.shape);
        let output = self.mask_tensor(shape, ValueRole::Temporary)?;
        self.builder.node(
            op,
            NodeFlags::default(),
            &[left.tensor, right.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    fn mask_binary(
        &mut self,
        op: OpCode,
        left: MaskValue,
        right: MaskValue,
    ) -> Result<MaskValue, AxFieldError> {
        let shape = broadcast_shape(left.shape, right.shape);
        let output = self.mask_tensor(shape, ValueRole::Temporary)?;
        self.builder.node(
            op,
            NodeFlags::default(),
            &[left.tensor, right.tensor],
            &[output.tensor],
            &[],
        )?;
        Ok(output)
    }

    fn element_count(&self) -> Result<usize, AxFieldError> {
        self.dimensions.iter().try_fold(1_usize, |count, &extent| {
            count
                .checked_mul(extent as usize)
                .ok_or(AxFieldError::ElementCountOverflow)
        })
    }
}

impl<A: Allocator + Clone, const D: usize> RealOps for AxFieldPlane<A, D> {
    type Value = RealValue;
    type Error = AxFieldError;

    #[inline]
    fn constant(&mut self, value: f32) -> Result<RealValue, AxFieldError> {
        self.scalar(value)
    }

    #[inline]
    fn add(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::add(self, left, right)
    }

    #[inline]
    fn sub(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::sub(self, left, right)
    }

    #[inline]
    fn mul(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::mul(self, left, right)
    }

    #[inline]
    fn negate(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::negate(self, value)
    }

    #[inline]
    fn reciprocal(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::reciprocal(self, value)
    }

    #[inline]
    fn rsqrt(&mut self, value: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::rsqrt(self, value)
    }

    #[inline]
    fn minimum(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::minimum(self, left, right)
    }

    #[inline]
    fn maximum(&mut self, left: RealValue, right: RealValue) -> Result<RealValue, AxFieldError> {
        AxFieldPlane::maximum(self, left, right)
    }
}

const fn tensor_flags(role: ValueRole) -> TensorFlags {
    match role {
        ValueRole::Input => TensorFlags::INPUT,
        ValueRole::Output => TensorFlags::OUTPUT,
        ValueRole::Constant => TensorFlags::CONSTANT,
        ValueRole::Temporary => TensorFlags(0),
    }
}

const fn broadcast_shape(left: ValueShape, right: ValueShape) -> ValueShape {
    if matches!(left, ValueShape::Field) || matches!(right, ValueShape::Field) {
        ValueShape::Field
    } else {
        ValueShape::Scalar
    }
}

fn require_same_shape(expected: ValueShape, actual: ValueShape) -> Result<(), AxFieldError> {
    if expected == actual {
        Ok(())
    } else {
        Err(AxFieldError::ShapeMismatch { expected, actual })
    }
}

fn require_same_dtype(left: RealValue, right: RealValue) -> Result<(), AxFieldError> {
    if left.dtype == right.dtype {
        Ok(())
    } else {
        Err(AxFieldError::DTypeMismatch {
            left: left.dtype,
            right: right.dtype,
        })
    }
}

fn validate_real_dtype(dtype: DType) -> Result<(), AxFieldError> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(AxFieldError::UnsupportedRealDType(dtype))
    }
}

fn validate_constant_len(dtype: DType, elements: usize, actual: usize) -> Result<(), AxFieldError> {
    let expected = elements
        .checked_mul(dtype.block_byte_size())
        .ok_or(AxFieldError::ElementCountOverflow)?;
    if expected == actual {
        Ok(())
    } else {
        Err(AxFieldError::InvalidConstantBytes { expected, actual })
    }
}

fn encode_real_constant(value: f32, dtype: DType) -> Result<([u8; 8], usize), AxFieldError> {
    let mut bytes = [0_u8; 8];
    let len = match dtype {
        DType::F64 => {
            bytes.copy_from_slice(&(value as f64).to_le_bytes());
            8
        }
        DType::F32 => {
            bytes[..4].copy_from_slice(&value.to_le_bytes());
            4
        }
        DType::F16 => {
            bytes[..2].copy_from_slice(&f32_to_f16(value).to_le_bytes());
            2
        }
        DType::B16 => {
            bytes[..2].copy_from_slice(&f32_to_b16(value).to_le_bytes());
            2
        }
        DType::B8 => {
            bytes[0] = nearest_fp8(value, DType::B8);
            1
        }
        DType::F8 => {
            bytes[0] = nearest_fp8(value, DType::F8);
            1
        }
        _ => return Err(AxFieldError::UnsupportedRealDType(dtype)),
    };
    Ok((bytes, len))
}

// Offline graph constants favor a small, dependency-free reference encoder
// over a host conversion dependency.  Every finite encoding is considered;
// ties retain the lower canonical bit pattern.
fn nearest_fp8(value: f32, dtype: DType) -> u8 {
    if value.is_nan() {
        return if dtype == DType::B8 { 0x7f } else { 0x7e };
    }
    if value == f32::INFINITY {
        return if dtype == DType::B8 { 0x7e } else { 0x7c };
    }
    if value == f32::NEG_INFINITY {
        return if dtype == DType::B8 { 0xfe } else { 0xfc };
    }

    let mut best_error = f32::INFINITY;
    let mut best = 0_u8;
    let mut bits = 0_u16;
    while bits <= u8::MAX as u16 {
        let candidate = match dtype {
            DType::B8 => e4m3_to_f32(bits as u8),
            DType::F8 => e5m2_to_f32(bits as u8),
            _ => unreachable!(),
        };
        if !candidate.is_nan() {
            let error = (candidate - value).abs();
            if error < best_error {
                best_error = error;
                best = bits as u8;
            }
        }
        bits += 1;
    }
    best
}

// Compact scalar encoders/decoders are kept private: precision policy belongs
// to callers, while this layer only needs exact bytes for graph constants.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return (sign | 0x7c00 | payload) as u16;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return (sign | 0x7c00) as u16;
    }
    if half_exponent <= 0 {
        return sign as u16;
    }
    (sign | ((half_exponent as u32) << 10) | (mantissa >> 13)) as u16
}

fn f32_to_b16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        ((bits >> 16) as u16) | 0x0040
    } else {
        (bits >> 16) as u16
    }
}

const fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) as u32) << 31;
    let exponent = (bits >> 3) & 0x0f;
    let mut mantissa = (bits & 0x07) as u32;
    let encoded = if exponent == 0x0f && mantissa == 0x07 {
        sign | 0x7fc0_0000
    } else if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut shift = 0_u32;
            while mantissa & 0x08 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x07;
            sign | ((127 + 1 - 7 - shift) << 23) | (mantissa << 20)
        }
    } else {
        sign | (((exponent as u32) + 127 - 7) << 23) | (mantissa << 20)
    };
    f32::from_bits(encoded)
}

const fn e5m2_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) as u32) << 31;
    let exponent = (bits >> 2) & 0x1f;
    let mut mantissa = (bits & 0x03) as u32;
    let encoded = if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 21)
    } else if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut shift = 0_u32;
            while mantissa & 0x04 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x03;
            sign | ((127 + 1 - 15 - shift) << 23) | (mantissa << 21)
        }
    } else {
        sign | (((exponent as u32) + 127 - 15) << 23) | (mantissa << 21)
    };
    f32::from_bits(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::central_gradient;
    use core::ptr::NonNull;
    use std::alloc::{AllocError, Layout, System};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct AllocationCounts {
        live: AtomicUsize,
        total: AtomicUsize,
    }

    #[derive(Clone)]
    struct TrackingAllocator(Arc<AllocationCounts>);

    // SAFETY: System owns every allocation.  This wrapper observes successful
    // ownership changes and delegates with the original pointer/layout pair.
    unsafe impl Allocator for TrackingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let allocation = System.allocate(layout)?;
            self.0.live.fetch_add(1, Ordering::Relaxed);
            self.0.total.fetch_add(1, Ordering::Relaxed);
            Ok(allocation)
        }

        unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
            self.0.live.fetch_sub(1, Ordering::Relaxed);
            // SAFETY: `pointer` and `layout` came from System above.
            unsafe { System.deallocate(pointer, layout) };
        }
    }

    struct OneBackend;

    impl ax::CostModel for OneBackend {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, backend: u16, _graph: &ax::Graph<'_>, _node: ax::NodeView<'_>) -> bool {
            backend == 0
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &ax::Graph<'_>,
            _node: ax::NodeView<'_>,
        ) -> u64 {
            1
        }

        fn transfer_cost(
            &self,
            _tensor: ax::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }
    }

    #[test]
    fn shape_and_dtype_rules_make_broadcasts_explicit() {
        let mut graph = AxFieldPlane::new_in([7, 5, 3], DType::F32, &System).unwrap();
        let field = graph.input().unwrap();
        let scalar = graph.scalar(2.0).unwrap();
        let sum = graph.add(field, scalar).unwrap();
        assert_eq!(sum.shape(), ValueShape::Field);
        assert_eq!(sum.dtype(), DType::F32);

        let scalar_sum = graph.add(scalar, scalar).unwrap();
        assert_eq!(scalar_sum.shape(), ValueShape::Scalar);
        let output = graph.output().unwrap();
        assert_eq!(
            graph.write_output(output, scalar_sum),
            Err(AxFieldError::ShapeMismatch {
                expected: ValueShape::Field,
                actual: ValueShape::Scalar,
            })
        );
        let splat = graph.splat_like(scalar_sum, field).unwrap();
        assert_eq!(splat.shape(), ValueShape::Field);
        graph.write_output(output, splat).unwrap();
    }

    #[test]
    fn raw_constants_and_casts_preserve_precision_metadata() {
        let mut graph = AxFieldPlane::new_in([32, 16], DType::F16, &System).unwrap();
        let narrow = graph.input_as(DType::B8).unwrap();
        let expanded = graph.cast(narrow, DType::F16).unwrap();
        assert_eq!(expanded.dtype(), DType::F16);
        assert_eq!(expanded.shape(), ValueShape::Field);
        let one = graph.scalar(1.0).unwrap();
        assert_eq!(one.dtype(), DType::F16);
        assert_eq!(
            graph.scalar_constant_raw(DType::F32, &[0, 0]),
            Err(AxFieldError::InvalidConstantBytes {
                expected: 4,
                actual: 2,
            })
        );
    }

    #[test]
    fn generic_gradient_emits_a_schedulable_allocator_owned_graph() {
        let counts = Arc::new(AllocationCounts::default());
        let allocator = TrackingAllocator(counts.clone());
        {
            let mut graph = AxFieldPlane::new_in([16, 8, 4], DType::F32, &allocator).unwrap();
            let point = [
                graph.input().unwrap(),
                graph.input().unwrap(),
                graph.input().unwrap(),
            ];
            let step = graph.scalar(0.125).unwrap();
            let mut quadratic = |ops: &mut AxFieldPlane<&TrackingAllocator, 3>,
                                 p: [RealValue; 3]| {
                let xx = ops.mul(p[0], p[0])?;
                let yy = ops.mul(p[1], p[1])?;
                let zz = ops.mul(p[2], p[2])?;
                let xy = ops.add(xx, yy)?;
                ops.add(xy, zz)
            };
            let gradient = central_gradient(&mut graph, &mut quadratic, point, [step; 3]).unwrap();
            for component in gradient {
                let output = graph.output().unwrap();
                graph.write_output(output, component).unwrap();
            }

            let owned = graph.finish().unwrap();
            let parsed = owned.parse().unwrap();
            assert_eq!(parsed.node_count(), 51);
            assert!(parsed.tensor_count() > parsed.node_count());
            for tensor_index in 0..parsed.tensor_count() {
                let tensor = parsed.tensor(TensorId(tensor_index as u32)).unwrap();
                let dimensions: std::vec::Vec<_> = tensor.dimensions().collect();
                assert!(dimensions == [16, 8, 4] || dimensions == [1, 1, 1]);
            }

            let mut schedule = ax::Schedule::analyze_in(&parsed, &allocator).unwrap();
            schedule.place(&parsed, &OneBackend).unwrap();
            let plan = schedule.build_plan(&parsed).unwrap();
            assert_eq!(plan.partitions.len(), 1);
            assert_eq!(plan.partitions[0].node_count as usize, parsed.node_count());
            assert!(counts.total.load(Ordering::Relaxed) > 0);
        }
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn fp8_constant_reference_encodings_cover_exact_geometry_values() {
        assert_eq!(nearest_fp8(0.0, DType::B8), 0x00);
        assert_eq!(nearest_fp8(1.0, DType::B8), 0x38);
        assert_eq!(nearest_fp8(-1.0, DType::F8), 0xbc);
        assert_eq!(nearest_fp8(f32::INFINITY, DType::F8), 0x7c);
    }
}
