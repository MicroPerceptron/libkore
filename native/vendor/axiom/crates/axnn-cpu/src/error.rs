use axnn::DType;

/// Errors from the CPU backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuError {
    /// Byte buffer length does not match the expected storage size.
    SizeMismatch { expected: usize, actual: usize },

    /// Fill pattern length does not equal `dtype.block_byte_size()`.
    PatternSizeMismatch { expected: usize, actual: usize },

    /// The requested dtype cast is not supported.
    /// Covers quantized↔anything casts that are not yet implemented.
    UnsupportedCast { from: DType, to: DType },

    /// Arena has insufficient space for the requested allocation.
    OutOfMemory,

    /// The requested operation is not supported for the given dtype(s).
    UnsupportedOp { op: &'static str, dtype: DType },

    /// Shape mismatch between operands.
    ShapeMismatch { op: &'static str },

    /// A safe in-place operation targeted immutable external storage.
    ReadOnlyStorage { op: &'static str },

    /// A typed tensor view does not meet its element alignment requirement.
    MisalignedStorage { required: usize },

    /// An in-place output overlaps another live operand.
    AliasedStorage { op: &'static str },

    /// A bound resource could not supply its data, such as an expert
    /// source whose read failed.
    ResourceUnavailable { op: &'static str },
}

impl core::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} bytes, got {actual}")
            }
            Self::PatternSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "fill pattern size mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::UnsupportedCast { from, to } => {
                write!(f, "unsupported cast: {from} -> {to}")
            }
            Self::OutOfMemory => {
                write!(f, "arena out of memory")
            }
            Self::UnsupportedOp { op, dtype } => {
                write!(f, "unsupported op '{op}' for dtype {dtype}")
            }
            Self::ShapeMismatch { op } => {
                write!(f, "shape mismatch in op '{op}'")
            }
            Self::ReadOnlyStorage { op } => {
                write!(f, "operation '{op}' requires writable tensor storage")
            }
            Self::MisalignedStorage { required } => {
                write!(f, "tensor storage requires {required}-byte alignment")
            }
            Self::ResourceUnavailable { op } => {
                write!(f, "'{op}' could not supply its data")
            }
            Self::AliasedStorage { op } => {
                write!(
                    f,
                    "operation '{op}' does not support overlapping tensor storage"
                )
            }
        }
    }
}
