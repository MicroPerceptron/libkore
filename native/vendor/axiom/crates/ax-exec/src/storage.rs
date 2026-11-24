//! Host tensor backing. Typed allocations retain their original allocator and
//! alignment; byte views never turn a `Vec<T>` into a mismatched `Vec<u8>`.
use crate::native::{
    NativeValue, allocate_native_value_in, copy_bytes_in, decode_prepared_native,
    encode_native_value_in,
};
use crate::{Error as AxCompileError, PlanError as AxArtifactError};
use alloc::vec::Vec;
use ax::DType;
use core::alloc::Allocator;
use core::mem::{align_of, size_of_val};
use virtio_accel_core::{BackendError, ByteSink};

pub enum RuntimeValue<'a, A: Allocator> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8, A>),
    Native(NativeValue<A>),
}

impl<A: Allocator> RuntimeValue<'_, A> {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
            Self::Native(value) => value.as_native_bytes(),
        }
    }
    pub fn into_bytes(self, allocator: A) -> Result<Vec<u8, A>, AxCompileError> {
        match self {
            Self::Borrowed(bytes) => copy_bytes_in(bytes, allocator),
            Self::Owned(bytes) => Ok(bytes),
            Self::Native(value) => encode_native_value_in(value, allocator),
        }
    }
}

/// A completed graph tensor retaining its backing, without an obligatory byte
/// vector allocation. Input passthroughs can still borrow invocation inputs.
/// On little-endian hosts native F32/I32 results expose their aligned allocation
/// as bytes. Provider transfers and precision conversions remain explicit.
/// Drop this value only after all consumers of its borrowed view have finished.
pub struct AxTensor<'a, A: Allocator> {
    pub(crate) dtype: DType,
    pub(crate) value: RuntimeValue<'a, A>,
}
impl<A: Allocator> AxTensor<'_, A> {
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn as_bytes(&self) -> &[u8] {
        self.value.as_slice()
    }
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
    /// Materialize the legacy byte-vector ABI, moving existing byte storage
    /// when possible. Typed F32/I32 or borrowed backing requires one copy here.
    /// `allocator` is used only for new storage; moved storage keeps its allocator.
    pub fn into_bytes_in(self, allocator: A) -> Result<Vec<u8, A>, AxCompileError> {
        self.value.into_bytes(allocator)
    }

    /// Transfer typed ownership without copying an existing native allocation.
    /// Borrowed or encoded F32/I32 storage is explicitly materialized once.
    /// Existing allocations retain their original allocator and alignment.
    pub fn into_native_in(self, allocator: A) -> Result<NativeValue<A>, AxCompileError> {
        match self.value {
            RuntimeValue::Native(value) => Ok(value),
            RuntimeValue::Owned(bytes) if self.dtype == DType::U8 => Ok(NativeValue::U8(bytes)),
            value => {
                let bytes = value.as_slice();
                let element_bytes = self.dtype.block_byte_size();
                if element_bytes == 0 || !bytes.len().is_multiple_of(element_bytes) {
                    return Err(AxArtifactError::InvalidPartition.into());
                }
                let mut native = allocate_native_value_in(
                    self.dtype,
                    bytes.len() / element_bytes * self.dtype.block_size(),
                    allocator,
                )?;
                decode_prepared_native(&mut native, bytes);
                Ok(native)
            }
        }
    }
}
impl<A: Allocator> core::fmt::Debug for AxTensor<'_, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AxTensor")
            .field("dtype", &self.dtype)
            .field("bytes", &self.len())
            .finish_non_exhaustive()
    }
}

// Only numeric types with every bit pattern valid are passed to these helpers.
fn bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: the concrete callers use initialized, padding-free f32/i32/u8;
    // the view borrows exactly the original allocation and never owns it.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}
fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: as above, and the exclusive borrow prevents aliasing. Every
    // possible byte pattern is a valid value of the concrete numeric types.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), size_of_val(values)) }
}
impl<A: Allocator> NativeValue<A> {
    pub fn as_native_bytes(&self) -> &[u8] {
        // Native F32/I32 runtime backing is retained on little-endian hosts.
        // Big-endian hosts encode it before crossing a boundary; U8 is unchanged.
        match self {
            Self::F32(v) => bytes(v),
            Self::I32(v) => bytes(v),
            Self::U8(v) | Self::Packed(_, v) => v,
        }
    }
    pub fn as_native_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::F32(v) => bytes_mut(v),
            Self::I32(v) => bytes_mut(v),
            Self::U8(v) | Self::Packed(_, v) => v,
        }
    }
}

#[derive(Clone, Copy)]
pub enum NativeRef<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    U8(&'a [u8]),
    Packed(DType, &'a [u8]),
}
impl<'a> NativeRef<'a> {
    pub fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::Packed(dtype, values) => {
                values.len() / dtype.block_byte_size() * dtype.block_size()
            }
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    /// `elements` elements from `offset`, in place. Packed ranges keep whole
    /// blocks.
    pub fn range(self, offset: usize, elements: usize) -> Option<Self> {
        let end = offset.checked_add(elements)?;
        Some(match self {
            Self::F32(values) => Self::F32(values.get(offset..end)?),
            Self::I32(values) => Self::I32(values.get(offset..end)?),
            Self::U8(values) => Self::U8(values.get(offset..end)?),
            Self::Packed(dtype, values) => {
                let block = dtype.block_size();
                if !offset.is_multiple_of(block) || !elements.is_multiple_of(block) {
                    return None;
                }
                let bytes = |count: usize| count / block * dtype.block_byte_size();
                Self::Packed(dtype, values.get(bytes(offset)..bytes(end))?)
            }
        })
    }
    pub fn dtype(self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::U8(_) => DType::U8,
            Self::Packed(dtype, _) => dtype,
        }
    }
    pub fn as_bytes(self) -> &'a [u8] {
        match self {
            Self::F32(values) => bytes(values),
            Self::I32(values) => bytes(values),
            Self::U8(values) | Self::Packed(_, values) => values,
        }
    }
    pub fn from_bytes(dtype: DType, source: &'a [u8]) -> Option<Self> {
        if matches!(
            dtype,
            DType::Q4 | DType::Q6 | DType::Q8 | DType::B8 | DType::B16
        ) {
            return source
                .len()
                .is_multiple_of(dtype.block_byte_size())
                .then_some(Self::Packed(dtype, source));
        }
        if dtype == DType::U8 {
            return Some(Self::U8(source));
        }
        if !cfg!(target_endian = "little") || !source.len().is_multiple_of(4) {
            return None;
        }
        match dtype {
            DType::F32 if source.as_ptr().align_offset(align_of::<f32>()) == 0 => {
                // SAFETY: alignment and length checked, f32 accepts every bit
                // pattern, and the returned view cannot outlive the input.
                Some(Self::F32(unsafe {
                    core::slice::from_raw_parts(source.as_ptr().cast(), source.len() / 4)
                }))
            }
            DType::I32 if source.as_ptr().align_offset(align_of::<i32>()) == 0 => {
                // SAFETY: same checks as the f32 arm; all i32 patterns valid.
                Some(Self::I32(unsafe {
                    core::slice::from_raw_parts(source.as_ptr().cast(), source.len() / 4)
                }))
            }
            _ => None,
        }
    }
}
pub enum NativeSlot<'a, A: Allocator> {
    Borrowed(NativeRef<'a>),
    Owned(NativeValue<A>),
}
impl<A: Allocator> NativeValue<A> {
    pub(crate) fn view(&self) -> NativeRef<'_> {
        match self {
            Self::F32(v) => NativeRef::F32(v),
            Self::I32(v) => NativeRef::I32(v),
            Self::U8(v) => NativeRef::U8(v),
            Self::Packed(dtype, v) => NativeRef::Packed(*dtype, v),
        }
    }
}
pub trait NativeInputs {
    fn view(&self, slot: usize) -> Option<NativeRef<'_>>;
    fn packed(&self, slot: usize, dtype: DType) -> Result<&[u8], AxCompileError> {
        match self.view(slot) {
            Some(NativeRef::Packed(actual, bytes)) if actual == dtype => Ok(bytes),
            _ => Err(AxArtifactError::InvalidPartition.into()),
        }
    }
    fn f32(&self, slot: usize) -> Result<&[f32], AxCompileError> {
        match self.view(slot) {
            Some(NativeRef::F32(v)) => Ok(v),
            _ => Err(AxArtifactError::InvalidPartition.into()),
        }
    }
    fn i32(&self, slot: usize) -> Result<&[i32], AxCompileError> {
        match self.view(slot) {
            Some(NativeRef::I32(v)) => Ok(v),
            _ => Err(AxArtifactError::InvalidPartition.into()),
        }
    }
    fn u8(&self, slot: usize) -> Result<&[u8], AxCompileError> {
        match self.view(slot) {
            Some(NativeRef::U8(v)) => Ok(v),
            _ => Err(AxArtifactError::InvalidPartition.into()),
        }
    }
}
impl<A: Allocator> NativeInputs for [Option<NativeValue<A>>] {
    fn view(&self, slot: usize) -> Option<NativeRef<'_>> {
        self.get(slot)?.as_ref().map(NativeValue::view)
    }
}
impl<A: Allocator> NativeInputs for [Option<NativeSlot<'_, A>>] {
    fn view(&self, slot: usize) -> Option<NativeRef<'_>> {
        match self.get(slot)?.as_ref()? {
            NativeSlot::Borrowed(v) => Some(*v),
            NativeSlot::Owned(v) => Some(v.view()),
        }
    }
}

/// Call-scoped writable graph output. `pitched` describes packed logical rows
/// in caller-owned backing; prefix, row padding and suffix are never written.
/// This is a host destination, not a device import or a synchronization fence.
pub struct AxOutput<'a> {
    backing: &'a mut [u8],
    offset: usize,
    row_bytes: usize,
    stride: usize,
    len: usize,
}
impl core::fmt::Debug for AxOutput<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AxOutput")
            .field("offset", &self.offset)
            .field("row_bytes", &self.row_bytes)
            .field("stride", &self.stride)
            .field("logical_bytes", &self.len)
            .finish_non_exhaustive()
    }
}
impl<'a> AxOutput<'a> {
    pub fn packed(backing: &'a mut [u8]) -> Self {
        let len = backing.len();
        Self {
            backing,
            offset: 0,
            row_bytes: len,
            stride: len,
            len,
        }
    }
    pub fn pitched(
        backing: &'a mut [u8],
        offset: usize,
        row_bytes: usize,
        rows: usize,
        stride: usize,
    ) -> Result<Self, BackendError> {
        if row_bytes == 0 || rows == 0 || stride < row_bytes {
            return Err(BackendError::InvalidArgument);
        }
        let len = row_bytes
            .checked_mul(rows)
            .ok_or(BackendError::InvalidArgument)?;
        let end = (rows - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(offset))
            .and_then(|v| v.checked_add(row_bytes))
            .ok_or(BackendError::InvalidArgument)?;
        if end > backing.len() {
            return Err(BackendError::InvalidArgument);
        }
        Ok(Self {
            backing,
            offset,
            row_bytes,
            stride,
            len,
        })
    }
}
impl ByteSink for AxOutput<'_> {
    fn len(&self) -> u64 {
        self.len as u64
    }
    fn write_at(&mut self, offset: u64, mut source: &[u8]) -> Result<(), BackendError> {
        let mut pos = usize::try_from(offset).map_err(|_| BackendError::InvalidArgument)?;
        if pos
            .checked_add(source.len())
            .is_none_or(|end| end > self.len)
        {
            return Err(BackendError::InvalidArgument);
        }
        while !source.is_empty() {
            let col = pos % self.row_bytes;
            let count = source.len().min(self.row_bytes - col);
            let dst = self.offset + (pos / self.row_bytes) * self.stride + col;
            self.backing[dst..dst + count].copy_from_slice(&source[..count]);
            source = &source[count..];
            pos += count;
        }
        Ok(())
    }
    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        (self.stride == self.row_bytes || self.len == self.row_bytes)
            .then(|| &mut self.backing[self.offset..self.offset + self.len])
    }
}
