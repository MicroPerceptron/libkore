use crate::DType;

/// A single typed value matching a [`DType`].
///
/// Used for element-wise get/set on tensors. Each variant maps
/// to exactly one `DType`:
///
/// | Variant | DType |
/// |---------|-------|
/// | `Bool`  | Q1    |
/// | `U8`    | U8    |
/// | `I8`    | I8    |
/// | `I16`   | I16   |
/// | `I32`   | I32   |
/// | `I64`   | I64   |
/// | `F16`   | F16   |
/// | `B16`   | B16   |
/// | `F32`   | F32   |
/// | `F64`   | F64   |
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    U8(u8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(f16),
    B16(u16), // bfloat16 has no native Rust type yet
    F32(f32),
    F64(f64),
}

impl Scalar {
    /// The `DType` this scalar corresponds to.
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Q1,
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F16(_) => DType::F16,
            Self::B16(_) => DType::B16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    /// Convert to `f64` (lossy for integers > 2^53).
    pub fn to_f64(&self) -> f64 {
        match *self {
            Self::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
            Self::U8(v) => v as f64,
            Self::I8(v) => v as f64,
            Self::I16(v) => v as f64,
            Self::I32(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::F16(v) => v as f64,
            Self::B16(bits) => {
                // bfloat16 → f32: shift left 16 bits into f32 format
                let f = f32::from_bits((bits as u32) << 16);
                f as f64
            }
            Self::F32(v) => v as f64,
            Self::F64(v) => v,
        }
    }

    /// Whether the value is logically "truthy" (non-zero / true).
    pub fn is_truthy(&self) -> bool {
        match *self {
            Self::Bool(v) => v,
            Self::U8(v) => v != 0,
            Self::I8(v) => v != 0,
            Self::I16(v) => v != 0,
            Self::I32(v) => v != 0,
            Self::I64(v) => v != 0,
            Self::F16(v) => v != 0.0,
            Self::B16(v) => v != 0,
            Self::F32(v) => v != 0.0,
            Self::F64(v) => v != 0.0,
        }
    }
}

// Convenience From impls for common types.
impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<u8> for Scalar {
    fn from(v: u8) -> Self {
        Self::U8(v)
    }
}
impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Self::I32(v)
    }
}
impl From<f16> for Scalar {
    fn from(v: f16) -> Self {
        Self::F16(v)
    }
}
impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}
impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}
