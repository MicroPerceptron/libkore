use axnn::lowering::{KernelRequest, SemanticOp};
use axnn::{
    ArgReduceOp, Backend, BinaryOp, ComputeBackend, DType, RasterBackend, ReduceOp, RopeMode,
    RoundedShape, Scalar, Tensor, TensorDesc, TernaryOp, UnaryOp,
};

use crate::arch;
use crate::cast;
use crate::error::CpuError;
use crate::planner::{self, CpuCapabilities};
use crate::pool::BumpArena;
use crate::shapes;
use crate::storage::{CpuStorage, CpuStorageMut};
use crate::stride::StrideIter;

#[inline]
const fn dtype_alignment(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 => 8,
        DType::F32 | DType::I32 => 4,
        DType::F16 | DType::B16 | DType::I16 | DType::Q8 | DType::Q6 | DType::Q4 => 2,
        // Q1 raster/bitmat paths reinterpret packed words as u32.
        DType::Q1 => 4,
        DType::I8 | DType::U8 | DType::B8 | DType::F8 => 1,
    }
}

fn validate_external_storage(bytes: &[u8], desc: &TensorDesc) -> Result<(), CpuError> {
    let span = desc.checked_storage_bytes().ok_or(CpuError::SizeMismatch {
        expected: usize::MAX,
        actual: bytes.len(),
    })?;
    let expected = desc
        .layout
        .offset()
        .checked_add(span)
        .ok_or(CpuError::SizeMismatch {
            expected: usize::MAX,
            actual: bytes.len(),
        })?;
    if bytes.len() < expected {
        return Err(CpuError::SizeMismatch {
            expected,
            actual: bytes.len(),
        });
    }

    let required = dtype_alignment(desc.dtype);
    // `expected <= bytes.len()` above proves `offset` is in-bounds (or the
    // one-past position for an empty view), so this address calculation is
    // valid even when the logical tensor is empty.
    let start = unsafe { bytes.as_ptr().add(desc.layout.offset()) };
    if start.addr() % required != 0 {
        return Err(CpuError::MisalignedStorage { required });
    }
    Ok(())
}

/// Default arena capacity: 16 MB.
pub const DEFAULT_ARENA_CAP: usize = 16 * 1024 * 1024;

/// Reference CPU backend backed by a bump arena.
///
/// The arena owns all tensor storage. Allocating methods bump the
/// cursor; the lifetime on `CpuStorage<'_>` ensures all tensors are
/// dropped before the backend can be reset or dropped.
///
/// With the `alloc` feature, allocations that exceed the arena capacity
/// fall back to the global allocator. These overflow allocations are
/// individually heap-allocated and freed when the backend is reset or
/// dropped.
/// 32-byte-aligned unit for heap overflow allocations. `vec![0u8; n]`
/// requests only align-1, and a conforming global allocator (e.g. a
/// custom kernel allocator) may honor exactly that — leaving the
/// f32/u32/f64 views taken over tensor storage undefined behavior.
/// Building overflow buffers from these chunks gives them the same
/// 32-byte guarantee as the arena.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy)]
#[repr(align(32))]
struct OverflowChunk([u8; 32]);

pub struct CpuBackend<const N: usize = DEFAULT_ARENA_CAP> {
    arena: BumpArena<N>,
    kernel_capabilities: CpuCapabilities,
    #[cfg(feature = "alloc")]
    overflow: core::cell::RefCell<alloc::vec::Vec<alloc::boxed::Box<[OverflowChunk]>>>,
}

impl<const N: usize> CpuBackend<N> {
    pub const fn new() -> Self {
        Self::with_kernel_capabilities(CpuCapabilities::compiled())
    }

    /// Construct a backend with an explicit kernel-admission policy.
    ///
    /// The caller (host runtime or `kore`) owns feature discovery for
    /// catalogued implementation tiers. This policy cannot downgrade the
    /// artifact's compile-target baseline: the runtime must still load the
    /// artifact only on a CPU satisfying every feature used by `arch::*`.
    /// Families migrate behind this catalog boundary incrementally; until a
    /// family does, it retains that target-wide ISA contract.
    pub const fn with_kernel_capabilities(kernel_capabilities: CpuCapabilities) -> Self {
        CpuBackend {
            arena: BumpArena::new(),
            kernel_capabilities,
            #[cfg(feature = "alloc")]
            overflow: core::cell::RefCell::new(alloc::vec::Vec::new()),
        }
    }

    pub const fn kernel_capabilities(&self) -> CpuCapabilities {
        self.kernel_capabilities
    }

    /// Wrap an external immutable byte slice as a zero-copy tensor.
    ///
    /// The external lifetime remains attached through zero-copy operations;
    /// it cannot be extended to the backend's lifetime:
    ///
    /// ```compile_fail
    /// use axnn::{Backend, DType, TensorDesc, shape};
    /// use axnn_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::<64>::new();
    /// let escaped;
    /// {
    ///     let bytes = [0u8; 4];
    ///     let wrapped = backend.wrap(&bytes, &TensorDesc::new(shape![1], DType::F32)).unwrap();
    ///     escaped = backend.contiguous(&wrapped).unwrap();
    /// }
    /// drop(escaped);
    /// ```
    pub fn wrap<'a>(
        &self,
        bytes: &'a [u8],
        desc: &TensorDesc,
    ) -> Result<Tensor<CpuStorage<'a>>, CpuError> {
        validate_external_storage(bytes, desc)?;
        Ok(Tensor::new(*desc, CpuStorage::from_bytes(bytes)))
    }

    /// Wrap an external mutable byte slice as a zero-copy tensor.
    ///
    /// This is the intended path for caller-owned output buffers such
    /// as framebuffers that backend kernels should write in place.
    pub fn wrap_mut<'a>(
        &self,
        bytes: &'a mut [u8],
        desc: &TensorDesc,
    ) -> Result<Tensor<CpuStorage<'a>>, CpuError> {
        validate_external_storage(bytes, desc)?;
        Ok(Tensor::new(*desc, CpuStorage::from_bytes_mut(bytes)))
    }

    /// Heap-allocate the backend, avoiding multi-MB stack temporaries.
    ///
    /// `CpuBackend` stores its arena inline as `[u8; N]`.  With large
    /// `N` values, `new()` can overflow the stack in debug builds (no
    /// guaranteed copy elision).  This constructor uses
    /// `Box::new_zeroed()` to allocate directly on the heap.
    ///
    /// # Safety justification
    /// `BumpArena` fields are valid when zero-initialized:
    /// - `BumpArena.buf`: `AlignedBuf([0u8; N])` — same as `new()`.
    /// - `BumpArena.cursor`: `Cell<usize>` at 0 — same as `new()`.
    #[cfg(feature = "std")]
    pub fn new_boxed() -> alloc::boxed::Box<Self> {
        // SAFETY: BumpArena<N> is valid when all bytes are zero.
        let mut be = unsafe { alloc::boxed::Box::<Self>::new_zeroed().assume_init() };
        // Vec is NOT valid when zeroed (null pointer), so reinitialize.
        #[cfg(feature = "alloc")]
        {
            be.overflow = core::cell::RefCell::new(alloc::vec::Vec::new());
        }
        be.kernel_capabilities = CpuCapabilities::compiled();
        be
    }

    /// Reset the arena, reclaiming all tensor storage.
    pub fn reset(&mut self) {
        self.arena.reset();
        #[cfg(feature = "alloc")]
        self.overflow.get_mut().clear();
    }

    pub fn used(&self) -> usize {
        self.arena.used()
    }

    pub fn remaining(&self) -> usize {
        self.arena.remaining()
    }

    /// Number of heap overflow allocations (only with `alloc` feature).
    #[cfg(feature = "alloc")]
    pub fn overflow_count(&self) -> usize {
        self.overflow.borrow().len()
    }

    fn arena_alloc_zeroed(&self, size: usize) -> Result<CpuStorageMut<'_>, CpuError> {
        if let Some(bytes) = self.arena.alloc_zeroed(size) {
            return Ok(CpuStorageMut::new(bytes));
        }
        #[cfg(feature = "alloc")]
        {
            return self.heap_overflow_alloc(size);
        }
        #[cfg(not(feature = "alloc"))]
        Err(CpuError::OutOfMemory)
    }

    /// Allocate from the global allocator when the arena is full.
    ///
    /// The buffer is built from [`OverflowChunk`]s so its address is
    /// 32-byte aligned like the arena's (see the type's docs), then
    /// stashed in `self.overflow` so its heap data lives as long as
    /// the backend.  We return a `&mut [u8]` pointing into the box —
    /// this is safe because:
    /// 1. Each box occupies a unique, non-overlapping heap region.
    /// 2. Moving the `Box` into the `Vec` does not move its heap data.
    /// 3. `Vec` reallocation moves `Box` values (fat pointers), not the
    ///    heap data they point to.
    #[cfg(feature = "alloc")]
    fn heap_overflow_alloc(&self, size: usize) -> Result<CpuStorageMut<'_>, CpuError> {
        let chunks = size.div_ceil(size_of::<OverflowChunk>());
        let boxed = alloc::vec![OverflowChunk([0u8; 32]); chunks].into_boxed_slice();
        let ptr = boxed.as_ptr() as *mut u8;
        self.overflow.borrow_mut().push(boxed);
        // SAFETY: see doc comment above; `size` ≤ the chunked length,
        // and an empty box's dangling pointer is aligned and non-null.
        let slice = unsafe { core::slice::from_raw_parts_mut(ptr, size) };
        Ok(CpuStorageMut::new(slice))
    }
}

impl<const N: usize> Backend for CpuBackend<N> {
    type Storage<'a>
        = CpuStorage<'a>
    where
        Self: 'a;
    type Error = CpuError;

    fn alloc(&self, desc: &TensorDesc) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let fresh = TensorDesc::new(desc.shape, desc.dtype);
        let storage = self.arena_alloc_zeroed(fresh.storage_bytes())?;
        Ok(Tensor::new(fresh, storage.freeze()))
    }

    fn zeros(&self, desc: &TensorDesc) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        self.alloc(desc)
    }

    fn fill(&self, desc: &TensorDesc, pattern: &[u8]) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let bbs = desc.dtype.block_byte_size();
        if pattern.len() != bbs {
            return Err(CpuError::PatternSizeMismatch {
                expected: bbs,
                actual: pattern.len(),
            });
        }
        let fresh = TensorDesc::new(desc.shape, desc.dtype);
        let n = fresh.storage_bytes();
        let mut storage = self.arena_alloc_zeroed(n)?;
        arch::mem::fill_pattern(storage.as_bytes_mut(), pattern, n / bbs);
        Ok(Tensor::new(fresh, storage.freeze()))
    }

    fn wrap<'a>(
        &self,
        bytes: &'a [u8],
        desc: &TensorDesc,
    ) -> Result<Tensor<CpuStorage<'a>>, CpuError>
    where
        Self: 'a,
    {
        CpuBackend::wrap(self, bytes, desc)
    }

    fn upload(&self, src: &[u8], desc: &TensorDesc) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let fresh = TensorDesc::new(desc.shape, desc.dtype);
        let n = fresh.storage_bytes();
        if src.len() != n {
            return Err(CpuError::SizeMismatch {
                expected: n,
                actual: src.len(),
            });
        }
        let mut storage = self.arena_alloc_zeroed(n)?;
        storage.as_bytes_mut().copy_from_slice(src);
        Ok(Tensor::new(fresh, storage.freeze()))
    }

    fn download(&self, tensor: &Tensor<CpuStorage<'_>>, dst: &mut [u8]) -> Result<(), CpuError> {
        ensure_tensor_storage("download", tensor)?;
        let desc = tensor.desc();
        let out_desc = TensorDesc::new(desc.shape, desc.dtype);
        let expected = out_desc.storage_bytes();
        if dst.len() < expected {
            return Err(CpuError::SizeMismatch {
                expected,
                actual: dst.len(),
            });
        }
        let src = tensor.storage().as_bytes();
        let bbs = desc.dtype.block_byte_size();
        if desc.is_contiguous() {
            let offset = desc.layout.offset();
            dst[..expected].copy_from_slice(&src[offset..offset + expected]);
        } else {
            let block_dims = desc.dtype.block_shape(&desc.shape);
            let mut dst_offset = 0;
            for src_offset in StrideIter::new(&block_dims, &desc.layout) {
                dst[dst_offset..dst_offset + bbs]
                    .copy_from_slice(&src[src_offset..src_offset + bbs]);
                dst_offset += bbs;
            }
        }
        Ok(())
    }

    fn contiguous<'a>(
        &'a self,
        tensor: &Tensor<CpuStorage<'a>>,
    ) -> Result<Tensor<CpuStorage<'a>>, CpuError> {
        ensure_tensor_storage("contiguous", tensor)?;
        if tensor.is_contiguous() {
            // The trait ties the output to the same lifetime carried by the
            // input storage. Returning the handle directly is therefore both
            // zero-copy and provenance-preserving, with no raw lifetime
            // reconstruction.
            return Ok(*tensor);
        }
        let desc = tensor.desc();
        let fresh = TensorDesc::new(desc.shape, desc.dtype);
        let n = fresh.storage_bytes();
        let bbs = desc.dtype.block_byte_size();
        let src = tensor.storage().as_bytes();

        let mut storage = self.arena_alloc_zeroed(n)?;
        let dst = storage.as_bytes_mut();
        let block_dims = desc.dtype.block_shape(&desc.shape);
        let mut dst_offset = 0;
        for src_offset in StrideIter::new(&block_dims, &desc.layout) {
            dst[dst_offset..dst_offset + bbs].copy_from_slice(&src[src_offset..src_offset + bbs]);
            dst_offset += bbs;
        }
        Ok(Tensor::new(fresh, storage.freeze()))
    }

    fn cast<'a>(
        &'a self,
        tensor: &Tensor<CpuStorage<'a>>,
        target: DType,
    ) -> Result<Tensor<CpuStorage<'a>>, CpuError> {
        let src_dtype = tensor.dtype();

        if src_dtype == target {
            return self.contiguous(tensor);
        }

        let tensor = self.contiguous(tensor)?;
        let src_bytes = tensor.storage().as_bytes();
        let offset = tensor.desc().layout.offset();
        let shape = tensor.desc().shape;
        let src_storage_bytes = tensor.desc().storage_bytes();
        let num_elements = tensor.count();

        // ── Dequantize: Qx → F32 ──
        if src_dtype.is_scaled_quant() && target == DType::F32 {
            let bs = src_dtype.block_size();
            if num_elements % bs != 0 {
                return Err(CpuError::UnsupportedCast {
                    from: src_dtype,
                    to: target,
                });
            }
            let out_desc = TensorDesc::new(shape, target);
            let mut out_storage = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
            let dst_f32 = unsafe { as_f32_mut(out_storage.as_bytes_mut()) };
            let src_slice = &src_bytes[offset..offset + src_storage_bytes];
            match src_dtype {
                DType::Q8 => arch::cast::dequant_q8_f32(src_slice, dst_f32, num_elements),
                DType::Q6 => arch::cast::dequant_q6_f32(src_slice, dst_f32, num_elements),
                DType::Q4 => arch::cast::dequant_q4_f32(src_slice, dst_f32, num_elements),
                _ => unreachable!(),
            }
            return Ok(Tensor::new(out_desc, out_storage.freeze()));
        }

        // ── Quantize: F32 → Qx ──
        if src_dtype == DType::F32 && target.is_scaled_quant() {
            let bs = target.block_size();
            if num_elements % bs != 0 {
                return Err(CpuError::UnsupportedCast {
                    from: src_dtype,
                    to: target,
                });
            }
            let out_desc = TensorDesc::new(shape, target);
            let src_f32 = unsafe { as_f32(&src_bytes[offset..offset + num_elements * 4]) };
            let mut out_storage = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
            match target {
                DType::Q8 => {
                    arch::cast::quant_f32_q8(src_f32, out_storage.as_bytes_mut(), num_elements)
                }
                DType::Q6 => {
                    arch::cast::quant_f32_q6(src_f32, out_storage.as_bytes_mut(), num_elements)
                }
                DType::Q4 => {
                    arch::cast::quant_f32_q4(src_f32, out_storage.as_bytes_mut(), num_elements)
                }
                _ => unreachable!(),
            }
            return Ok(Tensor::new(out_desc, out_storage.freeze()));
        }

        // ── B8/F8 dequantize (E4M3/E5M2 → F32; fast SIMD widen) ──
        if src_dtype.is_fp8() && target == DType::F32 {
            let out_desc = TensorDesc::new(shape, target);
            let mut out_storage = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
            let dst_f32 = unsafe { as_f32_mut(out_storage.as_bytes_mut()) };
            let src_slice = &src_bytes[offset..offset + num_elements];
            match src_dtype {
                DType::B8 => arch::f8::e4m3_to_f32(src_slice, dst_f32, num_elements),
                DType::F8 => arch::f8::e5m2_to_f32(src_slice, dst_f32, num_elements),
                _ => unreachable!(),
            }
            return Ok(Tensor::new(out_desc, out_storage.freeze()));
        }

        // F8 as a cast *target* (F32 → F8 quantize) is a follow-up; this
        // pass lands the decode/dequant + dot path.
        if target.is_fp8() {
            return Err(CpuError::UnsupportedCast {
                from: src_dtype,
                to: target,
            });
        }

        if src_dtype.is_quantized() || target.is_quantized() {
            return Err(CpuError::UnsupportedCast {
                from: src_dtype,
                to: target,
            });
        }

        // ── Scalar-to-scalar ──
        let src_bbs = src_dtype.block_byte_size();
        let dst_bbs = target.block_byte_size();
        let out_desc = TensorDesc::new(shape, target);
        let mut out_storage = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
        let dst = out_storage.as_bytes_mut();
        for i in 0..num_elements {
            let s = offset + i * src_bbs;
            let d = i * dst_bbs;
            cast::cast_element(
                &src_bytes[s..s + src_bbs],
                src_dtype,
                &mut dst[d..d + dst_bbs],
                target,
            );
        }
        Ok(Tensor::new(out_desc, out_storage.freeze()))
    }
}

// ── ComputeBackend ──────────────────────────────────────────────────────
//
// Dtype dispatch: each method matches on the input tensor's dtype and
// calls the correct arch kernel. The arch module re-exports handle
// compile-time ISA selection (avx2 vs portable vs neon).

/// Enforce the alignment contract of a typed reinterpret in every build.
/// External storage is rejected by `wrap`/`wrap_mut`; this assertion remains
/// as a final safety boundary for descriptors and views formed internally.
#[inline(always)]
fn assert_reinterpret_aligned(ptr: *const u8, align: usize) {
    assert!(
        (ptr as usize) % align == 0,
        "tensor storage reinterpret requires {align}-byte alignment"
    );
}

/// Reinterpret a `&[u8]` tensor body as `&[f32]`.
///
/// # Safety
/// Caller must ensure the slice is properly aligned and the dtype is F32.
#[inline]
pub(crate) unsafe fn as_f32(bytes: &[u8]) -> &[f32] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<f32>());
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

/// Reinterpret a `&mut [u8]` tensor body as `&mut [f32]`.
#[inline]
pub(crate) unsafe fn as_f32_mut(bytes: &mut [u8]) -> &mut [f32] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<f32>());
    unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4) }
}

/// Reinterpret a `&[u8]` tensor body as `&[u16]`.
#[inline]
pub(crate) unsafe fn as_u16(bytes: &[u8]) -> &[u16] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<u16>());
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) }
}

/// Reinterpret a `&[u8]` tensor body as `&[u32]`.
#[inline]
pub(crate) unsafe fn as_u32(bytes: &[u8]) -> &[u32] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<u32>());
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
}

/// Reinterpret a `&mut [u8]` tensor body as `&mut [u32]`.
#[inline]
pub(crate) unsafe fn as_u32_mut(bytes: &mut [u8]) -> &mut [u32] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<u32>());
    unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut u32, bytes.len() / 4) }
}

/// Reinterpret a `&[u8]` tensor body as `&[f64]`.
#[inline]
pub(crate) unsafe fn as_f64(bytes: &[u8]) -> &[f64] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<f64>());
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f64, bytes.len() / 8) }
}

/// Reinterpret a `&mut [u8]` tensor body as `&mut [f64]`.
#[inline]
pub(crate) unsafe fn as_f64_mut(bytes: &mut [u8]) -> &mut [f64] {
    assert_reinterpret_aligned(bytes.as_ptr(), align_of::<f64>());
    unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f64, bytes.len() / 8) }
}

/// Borrow exactly the bytes occupied by a contiguous tensor view.
///
/// `CpuStorage` is deliberately allowed to describe a larger shared backing
/// allocation. Kernels must never reinterpret that whole allocation when the
/// descriptor carries a non-zero view offset.
fn contiguous_bytes<'a>(
    tensor: &'a Tensor<CpuStorage<'_>>,
    op: &'static str,
) -> Result<&'a [u8], CpuError> {
    ensure_tensor_storage(op, tensor)?;
    let range = tensor
        .desc()
        .contiguous_byte_range()
        .ok_or(CpuError::ShapeMismatch { op })?;
    tensor
        .storage()
        .as_bytes()
        .get(range.clone())
        .ok_or(CpuError::SizeMismatch {
            expected: range.end,
            actual: tensor.storage().len(),
        })
}

/// Mutably borrow exactly the bytes occupied by a contiguous tensor view.
///
/// The mutable tensor borrow guarantees that no other operation can access
/// the selected logical view for the duration of the returned slice.
fn contiguous_bytes_mut<'a>(
    tensor: &'a mut Tensor<CpuStorage<'_>>,
    op: &'static str,
) -> Result<&'a mut [u8], CpuError> {
    ensure_tensor_storage(op, tensor)?;
    let range = tensor
        .desc()
        .contiguous_byte_range()
        .ok_or(CpuError::ShapeMismatch { op })?;
    let storage = tensor.storage();
    if !storage.is_writable() {
        return Err(CpuError::ReadOnlyStorage { op });
    }
    if range.end > storage.len() {
        return Err(CpuError::SizeMismatch {
            expected: range.end,
            actual: storage.len(),
        });
    }
    // SAFETY: every in-place backend entry point rejects overlapping operands
    // before reaching this helper. The range was checked inside explicitly
    // writable backing storage.
    let bytes = unsafe { storage.as_bytes_mut() }.ok_or(CpuError::ReadOnlyStorage { op })?;
    Ok(&mut bytes[range])
}

fn ensure_disjoint(
    op: &'static str,
    output: &CpuStorage<'_>,
    input: &CpuStorage<'_>,
) -> Result<(), CpuError> {
    if output.overlaps(input) {
        return Err(CpuError::AliasedStorage { op });
    }
    Ok(())
}

fn ensure_tensor_storage(
    op: &'static str,
    tensor: &Tensor<CpuStorage<'_>>,
) -> Result<(), CpuError> {
    let span = tensor
        .desc()
        .checked_storage_bytes()
        .ok_or(CpuError::ShapeMismatch { op })?;
    let end = tensor
        .desc()
        .layout
        .offset()
        .checked_add(span)
        .ok_or(CpuError::ShapeMismatch { op })?;
    if end > tensor.storage().len() {
        return Err(CpuError::SizeMismatch {
            expected: end,
            actual: tensor.storage().len(),
        });
    }
    let required = dtype_alignment(tensor.dtype());
    let start = tensor
        .storage()
        .as_ptr()
        .addr()
        .checked_add(tensor.desc().layout.offset())
        .ok_or(CpuError::ShapeMismatch { op })?;
    if start % required != 0 {
        return Err(CpuError::MisalignedStorage { required });
    }
    Ok(())
}

fn ensure_dtype(
    op: &'static str,
    tensor: &Tensor<CpuStorage<'_>>,
    expected: DType,
) -> Result<(), CpuError> {
    ensure_tensor_storage(op, tensor)?;
    if tensor.dtype() != expected {
        return Err(CpuError::UnsupportedOp {
            op,
            dtype: tensor.dtype(),
        });
    }
    Ok(())
}

fn ensure_writable(op: &'static str, tensor: &Tensor<CpuStorage<'_>>) -> Result<(), CpuError> {
    ensure_tensor_storage(op, tensor)?;
    if !tensor.storage().is_writable() {
        return Err(CpuError::ReadOnlyStorage { op });
    }
    Ok(())
}

fn ensure_same_shape(
    op: &'static str,
    first: &Tensor<CpuStorage<'_>>,
    second: &Tensor<CpuStorage<'_>>,
) -> Result<(), CpuError> {
    if first.shape() != second.shape() {
        return Err(CpuError::ShapeMismatch { op });
    }
    Ok(())
}

fn ensure_packed_surface(
    op: &'static str,
    tensor: &Tensor<CpuStorage<'_>>,
) -> Result<(), CpuError> {
    ensure_pitched_plane(op, tensor, DType::I32).map(|_| ())
}

/// Validate a rank-2 `[H, W]` view of `dtype` whose elements are packed
/// along the row and whose rows are `pitch` elements apart. Returns
/// `(pitch, width, height)` in elements.
fn ensure_pitched_plane(
    op: &'static str,
    tensor: &Tensor<CpuStorage<'_>>,
    dtype: DType,
) -> Result<(usize, usize, usize), CpuError> {
    ensure_dtype(op, tensor, dtype)?;
    let elem = dtype.block_byte_size();
    if tensor.shape().rank() != 2
        || tensor.desc().layout.rank() != 2
        || tensor.desc().layout.strides()[1] != elem
    {
        return Err(CpuError::ShapeMismatch { op });
    }
    let (height, width) = (tensor.shape().dim(0), tensor.shape().dim(1));
    let row_bytes = tensor.desc().layout.strides()[0];
    let min_row_bytes = width
        .checked_mul(elem)
        .ok_or(CpuError::ShapeMismatch { op })?;
    if !row_bytes.is_multiple_of(elem) || row_bytes < min_row_bytes {
        return Err(CpuError::ShapeMismatch { op });
    }
    Ok((row_bytes / elem, width, height))
}

/// Elements a `[height, width]` pitched plane spans from its origin.
fn plane_span(pitch: usize, width: usize, height: usize) -> Option<usize> {
    if width == 0 || height == 0 {
        return Some(0);
    }
    (height - 1).checked_mul(pitch)?.checked_add(width)
}

/// Borrow a read-only pitched plane: the bytes from the view origin, with
/// `(pitch, width, height)` in elements. Alignment and extent are checked
/// so the caller's reinterpretation cannot fault.
fn plane_bytes<'a>(
    op: &'static str,
    tensor: &'a Tensor<CpuStorage<'_>>,
    dtype: DType,
) -> Result<(&'a [u8], usize, usize, usize), CpuError> {
    let (pitch, width, height) = ensure_pitched_plane(op, tensor, dtype)?;
    ensure_tensor_storage(op, tensor)?;
    let elem = dtype.block_byte_size();
    let offset = tensor.desc().layout.offset();
    let bytes = &tensor.storage().as_bytes()[offset..];
    if !(bytes.as_ptr() as usize).is_multiple_of(elem) {
        return Err(CpuError::MisalignedStorage { required: elem });
    }
    let span = plane_span(pitch, width, height).ok_or(CpuError::ShapeMismatch { op })?;
    if span * elem > bytes.len() {
        return Err(CpuError::SizeMismatch {
            expected: offset + span * elem,
            actual: tensor.storage().len(),
        });
    }
    Ok((bytes, pitch, width, height))
}

/// Borrow a writable packed-XRGB `[H, W]` view as a pitched word surface.
fn surface_mut<'a>(
    op: &'static str,
    dst: &'a mut Tensor<CpuStorage<'_>>,
) -> Result<shapes::Surface<'a>, CpuError> {
    let (pitch, width, height) = ensure_pitched_plane(op, dst, DType::I32)?;
    ensure_writable(op, dst)?;
    let offset = dst.desc().layout.offset();
    let total = dst.storage().len();
    let bytes = &mut mut_bytes(dst, op)?[offset..];
    if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<u32>()) {
        return Err(CpuError::MisalignedStorage { required: 4 });
    }
    let span = plane_span(pitch, width, height).ok_or(CpuError::ShapeMismatch { op })?;
    if span * 4 > bytes.len() {
        return Err(CpuError::SizeMismatch {
            expected: offset + span * 4,
            actual: total,
        });
    }
    Ok(shapes::Surface {
        words: unsafe { as_u32_mut(bytes) },
        pitch,
        width,
        height,
    })
}

fn ensure_surface_region(
    op: &'static str,
    tensor: &Tensor<CpuStorage<'_>>,
    width: usize,
    height: usize,
) -> Result<(), CpuError> {
    ensure_packed_surface(op, tensor)?;
    if width > tensor.shape().dim(1) || height > tensor.shape().dim(0) {
        return Err(CpuError::ShapeMismatch { op });
    }
    Ok(())
}

/// Validate the public GEMM contract before any typed slice reaches a raw
/// microkernel. Dense kernels consume `[m,k] x [k,n]`; packed inference
/// kernels consume weight rows `[m,k]` and activation rows `[n,k]`.
fn gemm_geometry(
    lhs: &TensorDesc,
    rhs: &TensorDesc,
    accumulator: &TensorDesc,
) -> Result<(usize, usize, usize), CpuError> {
    if lhs.shape.rank() != 2 || rhs.shape.rank() != 2 || accumulator.shape.rank() != 2 {
        return Err(CpuError::ShapeMismatch { op: "gemm" });
    }

    let (output_dtype, packed_rhs) = match (lhs.dtype, rhs.dtype) {
        (DType::F32, DType::F32) => (DType::F32, false),
        (DType::F64, DType::F64) => (DType::F64, false),
        (DType::F16, DType::F16) | (DType::B16, DType::B16) => (DType::F32, false),
        (DType::Q4, DType::Q8)
        | (DType::Q6, DType::Q8)
        | (DType::Q8, DType::Q8)
        | (DType::B8, DType::F32)
        | (DType::F8, DType::F32) => (DType::F32, true),
        _ => {
            return Err(CpuError::UnsupportedOp {
                op: "gemm",
                dtype: lhs.dtype,
            });
        }
    };
    if accumulator.dtype != output_dtype || !accumulator.is_contiguous() {
        return Err(CpuError::ShapeMismatch { op: "gemm" });
    }

    let m = lhs.shape.dim(0);
    let k = lhs.shape.dim(1);
    let (rhs_k, n) = if packed_rhs {
        (rhs.shape.dim(1), rhs.shape.dim(0))
    } else {
        (rhs.shape.dim(0), rhs.shape.dim(1))
    };
    if rhs_k != k || accumulator.shape != axnn::Shape::from([m, n]) {
        return Err(CpuError::ShapeMismatch { op: "gemm" });
    }
    if packed_rhs
        && (!k.is_multiple_of(lhs.dtype.block_size()) || !k.is_multiple_of(rhs.dtype.block_size()))
    {
        return Err(CpuError::ShapeMismatch { op: "gemm" });
    }
    Ok((m, n, k))
}

fn ensure_vec_dot_geometry(lhs: &TensorDesc, rhs: &TensorDesc) -> Result<usize, CpuError> {
    let elements = lhs
        .shape
        .checked_num_elements()
        .ok_or(CpuError::ShapeMismatch { op: "vec_dot" })?;
    if rhs.shape.checked_num_elements() != Some(elements) {
        return Err(CpuError::ShapeMismatch { op: "vec_dot" });
    }
    let block = match (lhs.dtype, rhs.dtype) {
        (DType::Q8, DType::Q8) => DType::Q8.block_size(),
        (DType::Q4, DType::Q8) => DType::Q4.block_size(),
        (DType::Q6, DType::Q8) => DType::Q6.block_size(),
        (DType::B8, DType::F32) | (DType::F8, DType::F32) => 1,
        _ => {
            return Err(CpuError::UnsupportedOp {
                op: "vec_dot",
                dtype: lhs.dtype,
            });
        }
    };
    if !elements.is_multiple_of(block) {
        return Err(CpuError::ShapeMismatch { op: "vec_dot" });
    }
    Ok(elements)
}

fn binary_broadcast_non_f32(
    op: BinaryOp,
    first_desc: &TensorDesc,
    first: &[u8],
    second_desc: &TensorDesc,
    second: &[u8],
    output_desc: &TensorDesc,
    output: &mut [u8],
) -> Result<(), CpuError> {
    for output_index in 0..output_desc.num_elements() {
        let first_index =
            planner::broadcast_index(first_desc.shape, output_desc.shape, output_index)
                .ok_or(CpuError::ShapeMismatch { op: "binary" })?;
        let second_index =
            planner::broadcast_index(second_desc.shape, output_desc.shape, output_index)
                .ok_or(CpuError::ShapeMismatch { op: "binary" })?;
        match output_desc.dtype {
            DType::F64 => {
                (unsafe { as_f64_mut(output) })[output_index] = planner::binary_value_f64(
                    op,
                    unsafe { as_f64(first) }[first_index],
                    unsafe { as_f64(second) }[second_index],
                );
            }
            DType::F16 => {
                let first = &first[first_index * 2..first_index * 2 + 2];
                let second = &second[second_index * 2..second_index * 2 + 2];
                cast::write_f16(
                    &mut output[output_index * 2..output_index * 2 + 2],
                    planner::binary_value_f32(op, cast::read_f16(first), cast::read_f16(second)),
                );
            }
            DType::B16 => {
                let first = &first[first_index * 2..first_index * 2 + 2];
                let second = &second[second_index * 2..second_index * 2 + 2];
                cast::write_bf16(
                    &mut output[output_index * 2..output_index * 2 + 2],
                    planner::binary_value_f32(op, cast::read_bf16(first), cast::read_bf16(second)),
                );
            }
            dtype => {
                return Err(CpuError::UnsupportedOp {
                    op: "binary",
                    dtype,
                });
            }
        }
    }
    Ok(())
}

fn ternary_broadcast_non_f32(
    op: TernaryOp,
    first_desc: &TensorDesc,
    first: &[u8],
    second_desc: &TensorDesc,
    second: &[u8],
    third_desc: &TensorDesc,
    third: &[u8],
    output_desc: &TensorDesc,
    output: &mut [u8],
) -> Result<(), CpuError> {
    for output_index in 0..output_desc.num_elements() {
        let index = |desc: &TensorDesc| {
            planner::broadcast_index(desc.shape, output_desc.shape, output_index)
                .ok_or(CpuError::ShapeMismatch { op: "ternary" })
        };
        let first_index = index(first_desc)?;
        let second_index = index(second_desc)?;
        let third_index = index(third_desc)?;
        match output_desc.dtype {
            DType::F64 => {
                (unsafe { as_f64_mut(output) })[output_index] = planner::ternary_value_f64(
                    op,
                    unsafe { as_f64(first) }[first_index],
                    unsafe { as_f64(second) }[second_index],
                    unsafe { as_f64(third) }[third_index],
                );
            }
            DType::F16 | DType::B16 => {
                let read = |bytes: &[u8], index: usize| {
                    let bytes = &bytes[index * 2..index * 2 + 2];
                    if output_desc.dtype == DType::F16 {
                        cast::read_f16(bytes)
                    } else {
                        cast::read_bf16(bytes)
                    }
                };
                let value = planner::ternary_value_f32(
                    op,
                    read(first, first_index),
                    read(second, second_index),
                    read(third, third_index),
                );
                let output = &mut output[output_index * 2..output_index * 2 + 2];
                if output_desc.dtype == DType::F16 {
                    cast::write_f16(output, value);
                } else {
                    cast::write_bf16(output, value);
                }
            }
            dtype => {
                return Err(CpuError::UnsupportedOp {
                    op: "ternary",
                    dtype,
                });
            }
        }
    }
    Ok(())
}

impl<const N: usize> ComputeBackend for CpuBackend<N> {
    fn gemm(
        &self,
        a: &Tensor<CpuStorage<'_>>,
        b: &Tensor<CpuStorage<'_>>,
        c: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_tensor_storage("gemm", a)?;
        ensure_tensor_storage("gemm", b)?;
        ensure_writable("gemm", c)?;
        ensure_disjoint("gemm", c.storage(), a.storage())?;
        ensure_disjoint("gemm", c.storage(), b.storage())?;
        let (m, n, k) = gemm_geometry(a.desc(), b.desc(), c.desc())?;
        let a = self.contiguous(a)?;
        let b = self.contiguous(b)?;
        let ad = a.desc();
        let bd = b.desc();
        let cd = *c.desc();

        let a_bytes = contiguous_bytes(&a, "gemm")?;
        let b_bytes = contiguous_bytes(&b, "gemm")?;
        let c_bytes = contiguous_bytes_mut(c, "gemm")?;

        match (ad.dtype, bd.dtype) {
            (DType::F32, DType::F32) => {
                let a_f32 = unsafe { as_f32(a_bytes) };
                let b_f32 = unsafe { as_f32(b_bytes) };
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                let request = KernelRequest::gemm_accumulate(ad, bd, &cd)
                    .ok_or(CpuError::ShapeMismatch { op: "gemm" })?;
                let plan = planner::plan_cpu_kernel(&request, self.kernel_capabilities).ok_or(
                    CpuError::UnsupportedOp {
                        op: "gemm lowering",
                        dtype: ad.dtype,
                    },
                )?;
                planner::invoke_gemm_f32(&plan, a_f32, b_f32, c_f32)?;
            }
            (DType::F64, DType::F64) => {
                let a_f64 = unsafe { as_f64(a_bytes) };
                let b_f64 = unsafe { as_f64(b_bytes) };
                let c_f64 = unsafe { as_f64_mut(c_bytes) };
                arch::gemm::gemm_f64(a_f64, b_f64, c_f64, m, n, k);
            }
            (DType::F16, DType::F16) | (DType::B16, DType::B16) => {
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                if ad.dtype == DType::F16 {
                    arch::gemm::gemm_f16(a_bytes, b_bytes, c_f32, m, n, k);
                } else {
                    arch::gemm::gemm_bf16(a_bytes, b_bytes, c_f32, m, n, k);
                }
            }
            (DType::Q4, DType::Q8) => {
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                arch::qdot::gemm_q4_q8(a_bytes, b_bytes, c_f32, m, n, k);
            }
            (DType::Q6, DType::Q8) => {
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                arch::qdot::gemm_q6_q8(a_bytes, b_bytes, c_f32, m, n, k);
            }
            (DType::Q8, DType::Q8) => {
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                arch::qdot::gemm_q8_q8(a_bytes, b_bytes, c_f32, m, n, k);
            }
            // F8 weights × F32 activations → F32 (structurally the q-gemm
            // pattern: c[i,j] += Σ_k decode(w[i,k]) · x[j,k]).
            (DType::B8, DType::F32) => {
                let x = unsafe { as_f32(b_bytes) };
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                arch::f8::gemm_e4m3_f32(a_bytes, x, c_f32, m, n, k);
            }
            (DType::F8, DType::F32) => {
                let x = unsafe { as_f32(b_bytes) };
                let c_f32 = unsafe { as_f32_mut(c_bytes) };
                arch::f8::gemm_e5m2_f32(a_bytes, x, c_f32, m, n, k);
            }
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "gemm",
                    dtype: ad.dtype,
                });
            }
        }
        Ok(())
    }

    fn vec_dot(
        &self,
        a: &Tensor<CpuStorage<'_>>,
        b: &Tensor<CpuStorage<'_>>,
    ) -> Result<f32, CpuError> {
        ensure_tensor_storage("vec_dot", a)?;
        ensure_tensor_storage("vec_dot", b)?;
        let n = ensure_vec_dot_geometry(a.desc(), b.desc())?;
        let a = self.contiguous(a)?;
        let b = self.contiguous(b)?;
        let a_bytes = contiguous_bytes(&a, "vec_dot")?;
        let b_bytes = contiguous_bytes(&b, "vec_dot")?;

        match (a.dtype(), b.dtype()) {
            (DType::Q8, DType::Q8) => Ok(arch::qdot::vec_dot_q8_q8(a_bytes, b_bytes, n)),
            (DType::Q4, DType::Q8) => Ok(arch::qdot::vec_dot_q4_q8(a_bytes, b_bytes, n)),
            (DType::Q6, DType::Q8) => Ok(arch::qdot::vec_dot_q6_q8(a_bytes, b_bytes, n)),
            (DType::B8, DType::F32) => Ok(arch::f8::vec_dot_e4m3_f32(
                a_bytes,
                unsafe { as_f32(b_bytes) },
                n,
            )),
            (DType::F8, DType::F32) => Ok(arch::f8::vec_dot_e5m2_f32(
                a_bytes,
                unsafe { as_f32(b_bytes) },
                n,
            )),
            _ => Err(CpuError::UnsupportedOp {
                op: "vec_dot",
                dtype: a.dtype(),
            }),
        }
    }

    fn unary(
        &self,
        op: UnaryOp,
        src: &Tensor<CpuStorage<'_>>,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let src = self.contiguous(src)?;
        let input_desc = *src.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let s = contiguous_bytes(&src, "unary")?;
        let d = out.as_bytes_mut();

        if desc.dtype == DType::F32 {
            let request = KernelRequest::new(SemanticOp::Unary(op), &input_desc);
            let plan = planner::plan_cpu_kernel(&request, self.kernel_capabilities).ok_or(
                CpuError::UnsupportedOp {
                    op: "unary lowering",
                    dtype: desc.dtype,
                },
            )?;
            planner::invoke_f32(&plan, unsafe { as_f32(s) }, None, None, unsafe {
                as_f32_mut(d)
            })?;
            return Ok(Tensor::new(desc, out.freeze()));
        }

        macro_rules! dispatch_unary {
            ($f32_fn:expr, $f64_fn:expr, $f16_fn:expr, $bf16_fn:expr) => {
                match desc.dtype {
                    DType::F32 => $f32_fn(unsafe { as_f32(s) }, unsafe { as_f32_mut(d) }),
                    DType::F64 => $f64_fn(unsafe { as_f64(s) }, unsafe { as_f64_mut(d) }),
                    DType::F16 => $f16_fn(s, d),
                    DType::B16 => $bf16_fn(s, d),
                    _ => {
                        return Err(CpuError::UnsupportedOp {
                            op: "unary",
                            dtype: desc.dtype,
                        })
                    }
                }
            };
        }

        match op {
            UnaryOp::Relu => dispatch_unary!(
                arch::vec::relu_f32,
                arch::vec::relu_f64,
                arch::vec::relu_f16,
                arch::vec::relu_bf16
            ),
            UnaryOp::Neg => dispatch_unary!(
                arch::vec::neg_f32,
                arch::vec::neg_f64,
                arch::vec::neg_f16,
                arch::vec::neg_bf16
            ),
            UnaryOp::Sqrt => dispatch_unary!(
                arch::vec::sqrt_f32,
                arch::vec::sqrt_f64,
                arch::vec::sqrt_f16,
                arch::vec::sqrt_bf16
            ),
            UnaryOp::Exp => dispatch_unary!(
                arch::vec::exp_f32,
                arch::vec::exp_f64,
                arch::vec::exp_f16,
                arch::vec::exp_bf16
            ),
            UnaryOp::Tanh => dispatch_unary!(
                arch::vec::tanh_f32,
                arch::vec::tanh_f64,
                arch::vec::tanh_f16,
                arch::vec::tanh_bf16
            ),
            UnaryOp::Sigmoid => dispatch_unary!(
                arch::vec::sigmoid_f32,
                arch::vec::sigmoid_f64,
                arch::vec::sigmoid_f16,
                arch::vec::sigmoid_bf16
            ),
            UnaryOp::Silu => dispatch_unary!(
                arch::vec::silu_f32,
                arch::vec::silu_f64,
                arch::vec::silu_f16,
                arch::vec::silu_bf16
            ),
            UnaryOp::Gelu => dispatch_unary!(
                arch::vec::gelu_f32,
                arch::vec::gelu_f64,
                arch::vec::gelu_f16,
                arch::vec::gelu_bf16
            ),
            UnaryOp::Abs
            | UnaryOp::Acos
            | UnaryOp::Asin
            | UnaryOp::Atan
            | UnaryOp::Ceil
            | UnaryOp::Cos
            | UnaryOp::Exp2
            | UnaryOp::Floor
            | UnaryOp::Fract
            | UnaryOp::InverseSqrt
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Reciprocal
            | UnaryOp::Round
            | UnaryOp::Sign
            | UnaryOp::Sin
            | UnaryOp::Tan
            | UnaryOp::Trunc
            | UnaryOp::Softplus => match desc.dtype {
                DType::F64 => {
                    for (out, &value) in unsafe { as_f64_mut(d) }
                        .iter_mut()
                        .zip(unsafe { as_f64(s) })
                    {
                        *out = planner::unary_value_f64(op, value);
                    }
                }
                DType::F16 => {
                    for (input, output) in s.chunks_exact(2).zip(d.chunks_exact_mut(2)) {
                        let value = cast::read_f16(input);
                        cast::write_f16(output, planner::unary_value_f32(op, value));
                    }
                }
                DType::B16 => {
                    for (input, output) in s.chunks_exact(2).zip(d.chunks_exact_mut(2)) {
                        let value = cast::read_bf16(input);
                        cast::write_bf16(output, planner::unary_value_f32(op, value));
                    }
                }
                _ => {
                    return Err(CpuError::UnsupportedOp {
                        op: "unary",
                        dtype: desc.dtype,
                    });
                }
            },
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: &Tensor<CpuStorage<'_>>,
        b: &Tensor<CpuStorage<'_>>,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let a = self.contiguous(a)?;
        let b = self.contiguous(b)?;
        let request = KernelRequest::binary(op, a.desc(), b.desc())
            .ok_or(CpuError::ShapeMismatch { op: "binary" })?;
        let desc = TensorDesc::new(request.shape, request.output_dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let sa = contiguous_bytes(&a, "binary")?;
        let sb = contiguous_bytes(&b, "binary")?;
        let d = out.as_bytes_mut();

        if desc.dtype == DType::F32 {
            let plan = planner::plan_cpu_kernel(&request, self.kernel_capabilities).ok_or(
                CpuError::UnsupportedOp {
                    op: "binary lowering",
                    dtype: desc.dtype,
                },
            )?;
            planner::invoke_f32(
                &plan,
                unsafe { as_f32(sa) },
                Some(unsafe { as_f32(sb) }),
                None,
                unsafe { as_f32_mut(d) },
            )?;
            return Ok(Tensor::new(desc, out.freeze()));
        }

        if a.shape() != &desc.shape || b.shape() != &desc.shape {
            binary_broadcast_non_f32(op, a.desc(), sa, b.desc(), sb, &desc, d)?;
            return Ok(Tensor::new(desc, out.freeze()));
        }

        macro_rules! dispatch_binary {
            ($f32_fn:expr, $f64_fn:expr, $f16_fn:expr, $bf16_fn:expr) => {
                match desc.dtype {
                    DType::F32 => $f32_fn(unsafe { as_f32(sa) }, unsafe { as_f32(sb) }, unsafe {
                        as_f32_mut(d)
                    }),
                    DType::F64 => $f64_fn(unsafe { as_f64(sa) }, unsafe { as_f64(sb) }, unsafe {
                        as_f64_mut(d)
                    }),
                    DType::F16 => $f16_fn(sa, sb, d),
                    DType::B16 => $bf16_fn(sa, sb, d),
                    _ => {
                        return Err(CpuError::UnsupportedOp {
                            op: "binary",
                            dtype: desc.dtype,
                        })
                    }
                }
            };
        }

        match op {
            BinaryOp::Add => dispatch_binary!(
                arch::vec::add_f32,
                arch::vec::add_f64,
                arch::vec::add_f16,
                arch::vec::add_bf16
            ),
            BinaryOp::Sub => dispatch_binary!(
                arch::vec::sub_f32,
                arch::vec::sub_f64,
                arch::vec::sub_f16,
                arch::vec::sub_bf16
            ),
            BinaryOp::Mul => dispatch_binary!(
                arch::vec::mul_f32,
                arch::vec::mul_f64,
                arch::vec::mul_f16,
                arch::vec::mul_bf16
            ),
            BinaryOp::Div => dispatch_binary!(
                arch::vec::div_f32,
                arch::vec::div_f64,
                arch::vec::div_f16,
                arch::vec::div_bf16
            ),
            BinaryOp::Min | BinaryOp::Max | BinaryOp::Atan2 | BinaryOp::Pow | BinaryOp::Rem => {
                match desc.dtype {
                    DType::F64 => {
                        for ((out, &lhs), &rhs) in unsafe { as_f64_mut(d) }
                            .iter_mut()
                            .zip(unsafe { as_f64(sa) })
                            .zip(unsafe { as_f64(sb) })
                        {
                            *out = planner::binary_value_f64(op, lhs, rhs);
                        }
                    }
                    DType::F16 => {
                        for ((lhs, rhs), output) in sa
                            .chunks_exact(2)
                            .zip(sb.chunks_exact(2))
                            .zip(d.chunks_exact_mut(2))
                        {
                            cast::write_f16(
                                output,
                                planner::binary_value_f32(
                                    op,
                                    cast::read_f16(lhs),
                                    cast::read_f16(rhs),
                                ),
                            );
                        }
                    }
                    DType::B16 => {
                        for ((lhs, rhs), output) in sa
                            .chunks_exact(2)
                            .zip(sb.chunks_exact(2))
                            .zip(d.chunks_exact_mut(2))
                        {
                            cast::write_bf16(
                                output,
                                planner::binary_value_f32(
                                    op,
                                    cast::read_bf16(lhs),
                                    cast::read_bf16(rhs),
                                ),
                            );
                        }
                    }
                    _ => {
                        return Err(CpuError::UnsupportedOp {
                            op: "binary",
                            dtype: desc.dtype,
                        });
                    }
                }
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn ternary(
        &self,
        op: TernaryOp,
        a: &Tensor<CpuStorage<'_>>,
        b: &Tensor<CpuStorage<'_>>,
        c: &Tensor<CpuStorage<'_>>,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let a = self.contiguous(a)?;
        let b = self.contiguous(b)?;
        let c = self.contiguous(c)?;
        let request = KernelRequest::ternary(op, a.desc(), b.desc(), c.desc())
            .ok_or(CpuError::ShapeMismatch { op: "ternary" })?;
        let desc = TensorDesc::new(request.shape, request.output_dtype);
        let mut out = self.arena_alloc_zeroed(desc.storage_bytes())?;
        let first = contiguous_bytes(&a, "ternary")?;
        let second = contiguous_bytes(&b, "ternary")?;
        let third = contiguous_bytes(&c, "ternary")?;
        let dst = out.as_bytes_mut();

        match desc.dtype {
            DType::F32 => {
                let plan = planner::plan_cpu_kernel(&request, self.kernel_capabilities).ok_or(
                    CpuError::UnsupportedOp {
                        op: "ternary lowering",
                        dtype: desc.dtype,
                    },
                )?;
                planner::invoke_f32(
                    &plan,
                    unsafe { as_f32(first) },
                    Some(unsafe { as_f32(second) }),
                    Some(unsafe { as_f32(third) }),
                    unsafe { as_f32_mut(dst) },
                )?;
            }
            DType::F64 | DType::F16 | DType::B16
                if a.shape() != &desc.shape
                    || b.shape() != &desc.shape
                    || c.shape() != &desc.shape =>
            {
                ternary_broadcast_non_f32(
                    op,
                    a.desc(),
                    first,
                    b.desc(),
                    second,
                    c.desc(),
                    third,
                    &desc,
                    dst,
                )?;
            }
            DType::F64 => {
                for (((output, &x), &y), &z) in unsafe { as_f64_mut(dst) }
                    .iter_mut()
                    .zip(unsafe { as_f64(first) })
                    .zip(unsafe { as_f64(second) })
                    .zip(unsafe { as_f64(third) })
                {
                    *output = planner::ternary_value_f64(op, x, y, z);
                }
            }
            DType::F16 => {
                for (((x, y), z), output) in first
                    .chunks_exact(2)
                    .zip(second.chunks_exact(2))
                    .zip(third.chunks_exact(2))
                    .zip(dst.chunks_exact_mut(2))
                {
                    cast::write_f16(
                        output,
                        planner::ternary_value_f32(
                            op,
                            cast::read_f16(x),
                            cast::read_f16(y),
                            cast::read_f16(z),
                        ),
                    );
                }
            }
            DType::B16 => {
                for (((x, y), z), output) in first
                    .chunks_exact(2)
                    .zip(second.chunks_exact(2))
                    .zip(third.chunks_exact(2))
                    .zip(dst.chunks_exact_mut(2))
                {
                    cast::write_bf16(
                        output,
                        planner::ternary_value_f32(
                            op,
                            cast::read_bf16(x),
                            cast::read_bf16(y),
                            cast::read_bf16(z),
                        ),
                    );
                }
            }
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "ternary",
                    dtype: desc.dtype,
                });
            }
        }

        Ok(Tensor::new(desc, out.freeze()))
    }

    fn scale(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        s: f32,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let src = self.contiguous(src)?;
        let input_desc = *src.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let src_b = contiguous_bytes(&src, "scale")?;
        let dst_b = out.as_bytes_mut();

        match desc.dtype {
            DType::F32 => {
                arch::vec::scale_f32(unsafe { as_f32(src_b) }, unsafe { as_f32_mut(dst_b) }, s)
            }
            DType::F64 => arch::vec::scale_f64(
                unsafe { as_f64(src_b) },
                unsafe { as_f64_mut(dst_b) },
                s as f64,
            ),
            DType::F16 => arch::vec::scale_f16(src_b, dst_b, s),
            DType::B16 => arch::vec::scale_bf16(src_b, dst_b, s),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "scale",
                    dtype: desc.dtype,
                });
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn reduce(&self, op: ReduceOp, src: &Tensor<CpuStorage<'_>>) -> Result<f32, CpuError> {
        let src = self.contiguous(src)?;
        let s = contiguous_bytes(&src, "reduce")?;
        match (op, src.dtype()) {
            (ReduceOp::Sum, DType::F32) => Ok(arch::reduce::sum_f32(unsafe { as_f32(s) })),
            (ReduceOp::Sum, DType::F64) => Ok(arch::reduce::sum_f64(unsafe { as_f64(s) }) as f32),
            (ReduceOp::Sum, DType::F16) => Ok(arch::reduce::sum_f16(s)),
            (ReduceOp::Sum, DType::B16) => Ok(arch::reduce::sum_bf16(s)),
            (ReduceOp::Max, DType::F32) => Ok(arch::reduce::max_f32(unsafe { as_f32(s) })),
            (ReduceOp::Max, DType::F64) => Ok(arch::reduce::max_f64(unsafe { as_f64(s) }) as f32),
            (ReduceOp::Max, DType::F16) => Ok(arch::reduce::max_f16(s)),
            (ReduceOp::Max, DType::B16) => Ok(arch::reduce::max_bf16(s)),
            _ => Err(CpuError::UnsupportedOp {
                op: "reduce",
                dtype: src.dtype(),
            }),
        }
    }

    fn arg_reduce(&self, op: ArgReduceOp, src: &Tensor<CpuStorage<'_>>) -> Result<usize, CpuError> {
        let src = self.contiguous(src)?;
        let s = contiguous_bytes(&src, "arg_reduce")?;
        match (op, src.dtype()) {
            (ArgReduceOp::Min, DType::F32) => Ok(arch::reduce::argmin_f32(unsafe { as_f32(s) })),
            (ArgReduceOp::Min, DType::F64) => Ok(arch::reduce::argmin_f64(unsafe { as_f64(s) })),
            (ArgReduceOp::Min, DType::F16) => Ok(arch::reduce::argmin_f16(s)),
            (ArgReduceOp::Min, DType::B16) => Ok(arch::reduce::argmin_bf16(s)),
            (ArgReduceOp::Max, DType::F32) => Ok(arch::reduce::argmax_f32(unsafe { as_f32(s) })),
            (ArgReduceOp::Max, DType::F64) => Ok(arch::reduce::argmax_f64(unsafe { as_f64(s) })),
            (ArgReduceOp::Max, DType::F16) => Ok(arch::reduce::argmax_f16(s)),
            (ArgReduceOp::Max, DType::B16) => Ok(arch::reduce::argmax_bf16(s)),
            _ => Err(CpuError::UnsupportedOp {
                op: "arg_reduce",
                dtype: src.dtype(),
            }),
        }
    }

    fn softmax(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        _dim: isize,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let src = self.contiguous(src)?;
        let input_desc = *src.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let s = contiguous_bytes(&src, "softmax")?;
        let d = out.as_bytes_mut();

        match desc.dtype {
            DType::F32 => arch::reduce::softmax_f32(unsafe { as_f32(s) }, unsafe { as_f32_mut(d) }),
            DType::F64 => arch::reduce::softmax_f64(unsafe { as_f64(s) }, unsafe { as_f64_mut(d) }),
            DType::F16 => arch::reduce::softmax_f16(s, d),
            DType::B16 => arch::reduce::softmax_bf16(s, d),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "softmax",
                    dtype: desc.dtype,
                });
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn rms_norm(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        weight: &Tensor<CpuStorage<'_>>,
        eps: f32,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let src = self.contiguous(src)?;
        let weight = self.contiguous(weight)?;
        let input_desc = *src.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let s = contiguous_bytes(&src, "rms_norm")?;
        let w = contiguous_bytes(&weight, "rms_norm")?;
        let d = out.as_bytes_mut();

        match desc.dtype {
            DType::F32 => arch::reduce::rms_norm_f32(
                unsafe { as_f32(s) },
                unsafe { as_f32_mut(d) },
                unsafe { as_f32(w) },
                eps,
            ),
            DType::F64 => arch::reduce::rms_norm_f64(
                unsafe { as_f64(s) },
                unsafe { as_f64_mut(d) },
                unsafe { as_f64(w) },
                eps as f64,
            ),
            DType::F16 => arch::reduce::rms_norm_f16(s, d, w, eps),
            DType::B16 => arch::reduce::rms_norm_bf16(s, d, w, eps),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "rms_norm",
                    dtype: desc.dtype,
                });
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn layer_norm(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        weight: &Tensor<CpuStorage<'_>>,
        bias: &Tensor<CpuStorage<'_>>,
        eps: f32,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let src = self.contiguous(src)?;
        let weight = self.contiguous(weight)?;
        let bias = self.contiguous(bias)?;
        let input_desc = *src.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let n = desc.storage_bytes();
        let mut out = self.arena_alloc_zeroed(n)?;
        let s = contiguous_bytes(&src, "layer_norm")?;
        let w = contiguous_bytes(&weight, "layer_norm")?;
        let bi = contiguous_bytes(&bias, "layer_norm")?;
        let d = out.as_bytes_mut();

        match desc.dtype {
            DType::F32 => arch::reduce::layer_norm_f32(
                unsafe { as_f32(s) },
                unsafe { as_f32_mut(d) },
                unsafe { as_f32(w) },
                unsafe { as_f32(bi) },
                eps,
            ),
            DType::F64 => arch::reduce::layer_norm_f64(
                unsafe { as_f64(s) },
                unsafe { as_f64_mut(d) },
                unsafe { as_f64(w) },
                unsafe { as_f64(bi) },
                eps as f64,
            ),
            DType::F16 => arch::reduce::layer_norm_f16(s, d, w, bi, eps),
            DType::B16 => arch::reduce::layer_norm_bf16(s, d, w, bi, eps),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "layer_norm",
                    dtype: desc.dtype,
                });
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn rope(
        &self,
        qk: &mut Tensor<CpuStorage<'_>>,
        cos: &Tensor<CpuStorage<'_>>,
        sin: &Tensor<CpuStorage<'_>>,
        mode: RopeMode,
    ) -> Result<(), CpuError> {
        ensure_tensor_storage("rope", cos)?;
        ensure_tensor_storage("rope", sin)?;
        ensure_writable("rope", qk)?;
        ensure_disjoint("rope", qk.storage(), cos.storage())?;
        ensure_disjoint("rope", qk.storage(), sin.storage())?;
        let cos = self.contiguous(cos)?;
        let sin = self.contiguous(sin)?;
        let desc = qk.desc();
        let rank = desc.shape.rank();
        // Expect shape [n_tokens, n_heads, head_dim] or [n_tokens, head_dim].
        let (n_tokens, n_heads, head_dim) = if rank == 3 {
            (
                desc.shape.dim(0) as usize,
                desc.shape.dim(1) as usize,
                desc.shape.dim(2) as usize,
            )
        } else if rank == 2 {
            (desc.shape.dim(0) as usize, 1, desc.shape.dim(1) as usize)
        } else {
            return Err(CpuError::ShapeMismatch { op: "rope" });
        };

        let dtype = desc.dtype;
        let cos_b = contiguous_bytes(&cos, "rope")?;
        let sin_b = contiguous_bytes(&sin, "rope")?;
        let qk_bytes = contiguous_bytes_mut(qk, "rope")?;

        match (mode, dtype) {
            (RopeMode::Interleaved, DType::F32) => arch::rope::rope_interleaved_f32(
                unsafe { as_f32_mut(qk_bytes) },
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::NeoX, DType::F32) => arch::rope::rope_neox_f32(
                unsafe { as_f32_mut(qk_bytes) },
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::Interleaved, DType::F64) => arch::rope::rope_interleaved_f64(
                unsafe { as_f64_mut(qk_bytes) },
                unsafe { as_f64(cos_b) },
                unsafe { as_f64(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::NeoX, DType::F64) => arch::rope::rope_neox_f64(
                unsafe { as_f64_mut(qk_bytes) },
                unsafe { as_f64(cos_b) },
                unsafe { as_f64(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::Interleaved, DType::F16) => arch::rope::rope_interleaved_f16(
                qk_bytes,
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::NeoX, DType::F16) => arch::rope::rope_neox_f16(
                qk_bytes,
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::Interleaved, DType::B16) => arch::rope::rope_interleaved_bf16(
                qk_bytes,
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            (RopeMode::NeoX, DType::B16) => arch::rope::rope_neox_bf16(
                qk_bytes,
                unsafe { as_f32(cos_b) },
                unsafe { as_f32(sin_b) },
                n_tokens,
                n_heads,
                head_dim,
            ),
            _ => {
                return Err(CpuError::UnsupportedOp { op: "rope", dtype });
            }
        }
        Ok(())
    }

    fn conv_1d(
        &self,
        input: &Tensor<CpuStorage<'_>>,
        kernel: &Tensor<CpuStorage<'_>>,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let input = self.contiguous(input)?;
        let kernel = self.contiguous(kernel)?;
        let id = input.desc();
        let kd = kernel.desc();
        let channels = id.shape.dim(0) as usize;
        let in_len = id.shape.dim(-1) as usize;
        let kernel_len = kd.shape.dim(-1) as usize;
        let out_len = in_len - kernel_len + 1;

        let out_shape = axnn::Shape::from([channels, out_len]);
        let out_desc = TensorDesc::new(out_shape, id.dtype);
        let mut out = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
        let si = contiguous_bytes(&input, "conv_1d")?;
        let sk = contiguous_bytes(&kernel, "conv_1d")?;
        let d = out.as_bytes_mut();

        match id.dtype {
            DType::F32 => arch::conv::conv_1d_f32(
                unsafe { as_f32(si) },
                unsafe { as_f32(sk) },
                unsafe { as_f32_mut(d) },
                channels,
                in_len,
                kernel_len,
            ),
            DType::F64 => arch::conv::conv_1d_f64(
                unsafe { as_f64(si) },
                unsafe { as_f64(sk) },
                unsafe { as_f64_mut(d) },
                channels,
                in_len,
                kernel_len,
            ),
            DType::F16 => arch::conv::conv_1d_f16(si, sk, d, channels, in_len, kernel_len),
            DType::B16 => arch::conv::conv_1d_bf16(si, sk, d, channels, in_len, kernel_len),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "conv_1d",
                    dtype: id.dtype,
                });
            }
        }
        Ok(Tensor::new(out_desc, out.freeze()))
    }

    fn ssm_conv(
        &self,
        input: &Tensor<CpuStorage<'_>>,
        kernel: &Tensor<CpuStorage<'_>>,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        let input = self.contiguous(input)?;
        let kernel = self.contiguous(kernel)?;
        let id = input.desc();
        let kd = kernel.desc();
        let d_model = id.shape.dim(0) as usize;
        let seq_len = id.shape.dim(-1) as usize;
        let d_conv = kd.shape.dim(-1) as usize;

        let out_desc = TensorDesc::new(id.shape, id.dtype);
        let mut out = self.arena_alloc_zeroed(out_desc.storage_bytes())?;
        let si = contiguous_bytes(&input, "ssm_conv")?;
        let sk = contiguous_bytes(&kernel, "ssm_conv")?;
        let d = out.as_bytes_mut();

        match id.dtype {
            DType::F32 => arch::conv::ssm_conv_f32(
                unsafe { as_f32(si) },
                unsafe { as_f32(sk) },
                unsafe { as_f32_mut(d) },
                d_model,
                seq_len,
                d_conv,
            ),
            DType::F64 => arch::conv::ssm_conv_f64(
                unsafe { as_f64(si) },
                unsafe { as_f64(sk) },
                unsafe { as_f64_mut(d) },
                d_model,
                seq_len,
                d_conv,
            ),
            DType::F16 => arch::conv::ssm_conv_f16(si, sk, d, d_model, seq_len, d_conv),
            DType::B16 => arch::conv::ssm_conv_bf16(si, sk, d, d_model, seq_len, d_conv),
            _ => {
                return Err(CpuError::UnsupportedOp {
                    op: "ssm_conv",
                    dtype: id.dtype,
                });
            }
        }
        Ok(Tensor::new(out_desc, out.freeze()))
    }

    // ── Q1 (binary) graph operations ────────────────────────────

    fn set_element(
        &self,
        tensor: &Tensor<CpuStorage<'_>>,
        indices: &[usize],
        value: Scalar,
    ) -> Result<(), CpuError> {
        set_cpu_tensor_element(tensor, indices, value)
    }

    fn get_element(
        &self,
        tensor: &Tensor<CpuStorage<'_>>,
        indices: &[usize],
    ) -> Result<Scalar, CpuError> {
        get_cpu_tensor_element(tensor, indices)
    }

    fn topo_sort(&self, adj: &Tensor<CpuStorage<'_>>, out: &mut [u16]) -> Result<usize, CpuError> {
        let desc = adj.desc();
        if desc.dtype != DType::Q1 || desc.shape.rank() != 2 {
            return Err(CpuError::UnsupportedOp {
                op: "topo_sort",
                dtype: desc.dtype,
            });
        }
        let n = desc.shape.dim(0) as usize;
        if n != desc.shape.dim(1) as usize {
            return Err(CpuError::ShapeMismatch { op: "topo_sort" });
        }
        let row_bytes = desc.layout.strides()[0];
        let bytes = contiguous_bytes(adj, "topo_sort")?;
        let order =
            arch::bitmat::topo_sort(bytes, row_bytes, n).ok_or(CpuError::ShapeMismatch {
                op: "topo_sort (cycle detected)",
            })?;
        let count = order.len().min(out.len());
        out[..count].copy_from_slice(&order[..count]);
        Ok(count)
    }

    fn indegree(&self, adj: &Tensor<CpuStorage<'_>>, out: &mut [u32]) -> Result<usize, CpuError> {
        let desc = adj.desc();
        if desc.dtype != DType::Q1 || desc.shape.rank() != 2 {
            return Err(CpuError::UnsupportedOp {
                op: "indegree",
                dtype: desc.dtype,
            });
        }
        let n = desc.shape.dim(0) as usize;
        let row_bytes = desc.layout.strides()[0];
        let bytes = contiguous_bytes(adj, "indegree")?;
        let counts = arch::bitmat::col_popcounts(bytes, row_bytes, n);
        let count = counts.len().min(out.len());
        out[..count].copy_from_slice(&counts[..count]);
        Ok(count)
    }
}

pub(crate) fn set_cpu_tensor_element(
    tensor: &Tensor<CpuStorage<'_>>,
    indices: &[usize],
    value: Scalar,
) -> Result<(), CpuError> {
    ensure_tensor_storage("set_element", tensor)?;
    let desc = tensor.desc();
    // SAFETY: this operation takes no other tensor operand and returns no byte
    // reference. Writable provenance is checked before the interior write.
    let bytes = unsafe { tensor.storage().as_bytes_mut() }
        .ok_or(CpuError::ReadOnlyStorage { op: "set_element" })?;
    let ptr = bytes.as_mut_ptr();

    match desc.dtype {
        DType::Q1 => {
            // Q1 storage is block-padded: each row occupies
            // ceil(cols/256)*32 bytes, encoded in layout.strides()[0].
            // Use row stride for addressing, not a flat bit index.
            let (row, col) = q1_row_col(desc, indices)?;
            let row_bytes = desc.layout.strides()[0];
            let byte_off = desc
                .layout
                .offset()
                .checked_add(
                    row.checked_mul(row_bytes)
                        .and_then(|offset| offset.checked_add(col / 8))
                        .ok_or(CpuError::ShapeMismatch { op: "set_element" })?,
                )
                .ok_or(CpuError::ShapeMismatch { op: "set_element" })?;
            if byte_off >= bytes.len() {
                return Err(CpuError::SizeMismatch {
                    expected: byte_off.saturating_add(1),
                    actual: bytes.len(),
                });
            }
            let bit = col % 8;
            unsafe {
                let byte = ptr.add(byte_off);
                if value.is_truthy() {
                    *byte |= 1 << bit;
                } else {
                    *byte &= !(1 << bit);
                }
            }
        }
        DType::F32 => {
            let offset = element_byte_offset(desc, indices)?;
            if offset.checked_add(4).is_none_or(|end| end > bytes.len()) {
                return Err(CpuError::SizeMismatch {
                    expected: offset.saturating_add(4),
                    actual: bytes.len(),
                });
            }
            let v = match value {
                Scalar::F32(f) => f,
                other => other.to_f64() as f32,
            };
            unsafe {
                core::ptr::write_unaligned(ptr.add(offset) as *mut f32, v);
            }
        }
        DType::F64 => {
            let offset = element_byte_offset(desc, indices)?;
            if offset.checked_add(8).is_none_or(|end| end > bytes.len()) {
                return Err(CpuError::SizeMismatch {
                    expected: offset.saturating_add(8),
                    actual: bytes.len(),
                });
            }
            let v = value.to_f64();
            unsafe {
                core::ptr::write_unaligned(ptr.add(offset) as *mut f64, v);
            }
        }
        DType::I32 => {
            let offset = element_byte_offset(desc, indices)?;
            if offset.checked_add(4).is_none_or(|end| end > bytes.len()) {
                return Err(CpuError::SizeMismatch {
                    expected: offset.saturating_add(4),
                    actual: bytes.len(),
                });
            }
            let v = value.to_f64() as i32;
            unsafe {
                core::ptr::write_unaligned(ptr.add(offset) as *mut i32, v);
            }
        }
        DType::U8 => {
            let offset = element_byte_offset(desc, indices)?;
            if offset >= bytes.len() {
                return Err(CpuError::SizeMismatch {
                    expected: offset.saturating_add(1),
                    actual: bytes.len(),
                });
            }
            let v = match value {
                Scalar::U8(b) => b,
                other => other.to_f64() as u8,
            };
            unsafe {
                *ptr.add(offset) = v;
            }
        }
        dt => {
            return Err(CpuError::UnsupportedOp {
                op: "set_element",
                dtype: dt,
            });
        }
    }
    Ok(())
}

pub(crate) fn get_cpu_tensor_element(
    tensor: &Tensor<CpuStorage<'_>>,
    indices: &[usize],
) -> Result<Scalar, CpuError> {
    ensure_tensor_storage("get_element", tensor)?;
    let desc = tensor.desc();
    let bytes = tensor.storage().as_bytes();

    match desc.dtype {
        DType::Q1 => {
            let (row, col) = q1_row_col(desc, indices)?;
            let row_bytes = desc.layout.strides()[0];
            let byte_off = desc
                .layout
                .offset()
                .checked_add(
                    row.checked_mul(row_bytes)
                        .and_then(|offset| offset.checked_add(col / 8))
                        .ok_or(CpuError::ShapeMismatch { op: "get_element" })?,
                )
                .ok_or(CpuError::ShapeMismatch { op: "get_element" })?;
            let bit = col % 8;
            let byte = *bytes.get(byte_off).ok_or(CpuError::SizeMismatch {
                expected: byte_off.saturating_add(1),
                actual: bytes.len(),
            })?;
            Ok(Scalar::Bool((byte >> bit) & 1 != 0))
        }
        DType::F32 => {
            let off = element_byte_offset(desc, indices)?;
            let body = element_bytes(bytes, off, 4)?;
            let v = f32::from_le_bytes(body.try_into().unwrap());
            Ok(Scalar::F32(v))
        }
        DType::F64 => {
            let off = element_byte_offset(desc, indices)?;
            let body = element_bytes(bytes, off, 8)?;
            let v = f64::from_le_bytes(body.try_into().unwrap());
            Ok(Scalar::F64(v))
        }
        DType::I32 => {
            let off = element_byte_offset(desc, indices)?;
            let body = element_bytes(bytes, off, 4)?;
            let v = i32::from_le_bytes(body.try_into().unwrap());
            Ok(Scalar::I32(v))
        }
        DType::U8 => {
            let off = element_byte_offset(desc, indices)?;
            let byte = *bytes.get(off).ok_or(CpuError::SizeMismatch {
                expected: off.saturating_add(1),
                actual: bytes.len(),
            })?;
            Ok(Scalar::U8(byte))
        }
        dt => Err(CpuError::UnsupportedOp {
            op: "get_element",
            dtype: dt,
        }),
    }
}

fn element_bytes(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8], CpuError> {
    let end = offset
        .checked_add(len)
        .ok_or(CpuError::ShapeMismatch { op: "get_element" })?;
    bytes.get(offset..end).ok_or(CpuError::SizeMismatch {
        expected: end,
        actual: bytes.len(),
    })
}

// ── RasterBackend ───────────────────────────────────────────────────
//
// Dtype dispatch: currently only F32 edge functions and U32/I32
// framebuffers. The match arms are ready to grow when F16/Q8
// raster kernels land.

impl<const N: usize> RasterBackend for CpuBackend<N> {
    fn edge_coverage(
        &self,
        e0: &Tensor<CpuStorage<'_>>,
        e1: &Tensor<CpuStorage<'_>>,
        e2: &Tensor<CpuStorage<'_>>,
        thresh: f32,
        color: &Tensor<CpuStorage<'_>>,
        fb: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_dtype("edge_coverage", e0, DType::F32)?;
        ensure_dtype("edge_coverage", e1, DType::F32)?;
        ensure_dtype("edge_coverage", e2, DType::F32)?;
        ensure_dtype("edge_coverage", color, DType::I32)?;
        ensure_dtype("edge_coverage", fb, DType::I32)?;
        ensure_writable("edge_coverage", fb)?;
        ensure_same_shape("edge_coverage", e0, e1)?;
        ensure_same_shape("edge_coverage", e0, e2)?;
        ensure_same_shape("edge_coverage", e0, color)?;
        ensure_same_shape("edge_coverage", e0, fb)?;
        ensure_disjoint("edge_coverage", fb.storage(), e0.storage())?;
        ensure_disjoint("edge_coverage", fb.storage(), e1.storage())?;
        ensure_disjoint("edge_coverage", fb.storage(), e2.storage())?;
        ensure_disjoint("edge_coverage", fb.storage(), color.storage())?;
        let e0 = self.contiguous(e0)?;
        let e1 = self.contiguous(e1)?;
        let e2 = self.contiguous(e2)?;
        let color = self.contiguous(color)?;
        match e0.dtype() {
            DType::F32 => {
                arch::raster::fused_edge_coverage(
                    unsafe { as_f32(contiguous_bytes(&e0, "edge_coverage")?) },
                    unsafe { as_f32(contiguous_bytes(&e1, "edge_coverage")?) },
                    unsafe { as_f32(contiguous_bytes(&e2, "edge_coverage")?) },
                    thresh,
                    unsafe { as_u32(contiguous_bytes(&color, "edge_coverage")?) },
                    unsafe { as_u32_mut(contiguous_bytes_mut(fb, "edge_coverage")?) },
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "edge_coverage",
                dtype: dt,
            }),
        }
    }

    fn depth_test(
        &self,
        coverage: &Tensor<CpuStorage<'_>>,
        frag_z: &Tensor<CpuStorage<'_>>,
        zbuf: &mut Tensor<CpuStorage<'_>>,
        out_mask: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_dtype("depth_test", coverage, DType::I32)?;
        ensure_dtype("depth_test", frag_z, DType::F32)?;
        ensure_dtype("depth_test", zbuf, DType::F32)?;
        ensure_dtype("depth_test", out_mask, DType::I32)?;
        ensure_writable("depth_test", zbuf)?;
        ensure_writable("depth_test", out_mask)?;
        ensure_same_shape("depth_test", coverage, frag_z)?;
        ensure_same_shape("depth_test", coverage, zbuf)?;
        ensure_same_shape("depth_test", coverage, out_mask)?;
        ensure_disjoint("depth_test", zbuf.storage(), coverage.storage())?;
        ensure_disjoint("depth_test", zbuf.storage(), frag_z.storage())?;
        ensure_disjoint("depth_test", out_mask.storage(), coverage.storage())?;
        ensure_disjoint("depth_test", out_mask.storage(), frag_z.storage())?;
        ensure_disjoint("depth_test", zbuf.storage(), out_mask.storage())?;
        let coverage = self.contiguous(coverage)?;
        let frag_z = self.contiguous(frag_z)?;
        match frag_z.dtype() {
            DType::F32 => {
                arch::raster::depth_test(
                    unsafe { as_u32(contiguous_bytes(&coverage, "depth_test")?) },
                    unsafe { as_f32(contiguous_bytes(&frag_z, "depth_test")?) },
                    unsafe { as_f32_mut(contiguous_bytes_mut(zbuf, "depth_test")?) },
                    unsafe { as_u32_mut(contiguous_bytes_mut(out_mask, "depth_test")?) },
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "depth_test",
                dtype: dt,
            }),
        }
    }

    fn bary_interp(
        &self,
        w0: &Tensor<CpuStorage<'_>>,
        w1: &Tensor<CpuStorage<'_>>,
        w2: &Tensor<CpuStorage<'_>>,
        a0: f32,
        a1: f32,
        a2: f32,
    ) -> Result<Tensor<CpuStorage<'_>>, CpuError> {
        ensure_dtype("bary_interp", w0, DType::F32)?;
        ensure_dtype("bary_interp", w1, DType::F32)?;
        ensure_dtype("bary_interp", w2, DType::F32)?;
        ensure_same_shape("bary_interp", w0, w1)?;
        ensure_same_shape("bary_interp", w0, w2)?;
        let w0 = self.contiguous(w0)?;
        let w1 = self.contiguous(w1)?;
        let w2 = self.contiguous(w2)?;
        let input_desc = *w0.desc();
        let desc = TensorDesc::new(input_desc.shape, input_desc.dtype);
        let mut out = self.arena_alloc_zeroed(desc.storage_bytes())?;
        match desc.dtype {
            DType::F32 => {
                arch::raster::bary_interp(
                    unsafe { as_f32(contiguous_bytes(&w0, "bary_interp")?) },
                    unsafe { as_f32(contiguous_bytes(&w1, "bary_interp")?) },
                    unsafe { as_f32(contiguous_bytes(&w2, "bary_interp")?) },
                    a0,
                    a1,
                    a2,
                    unsafe { as_f32_mut(out.as_bytes_mut()) },
                );
            }
            dt => {
                return Err(CpuError::UnsupportedOp {
                    op: "bary_interp",
                    dtype: dt,
                });
            }
        }
        Ok(Tensor::new(desc, out.freeze()))
    }

    fn alpha_blend(
        &self,
        mask: &Tensor<CpuStorage<'_>>,
        src: &Tensor<CpuStorage<'_>>,
        dst: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_dtype("alpha_blend", mask, DType::I32)?;
        ensure_dtype("alpha_blend", src, DType::I32)?;
        ensure_dtype("alpha_blend", dst, DType::I32)?;
        ensure_writable("alpha_blend", dst)?;
        ensure_same_shape("alpha_blend", mask, src)?;
        ensure_same_shape("alpha_blend", mask, dst)?;
        ensure_disjoint("alpha_blend", dst.storage(), mask.storage())?;
        ensure_disjoint("alpha_blend", dst.storage(), src.storage())?;
        let mask = self.contiguous(mask)?;
        let src = self.contiguous(src)?;
        match dst.dtype() {
            DType::I32 => {
                arch::raster::alpha_blend(
                    unsafe { as_u32(contiguous_bytes(&mask, "alpha_blend")?) },
                    unsafe { as_u32(contiguous_bytes(&src, "alpha_blend")?) },
                    unsafe { as_u32_mut(contiguous_bytes_mut(dst, "alpha_blend")?) },
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "alpha_blend",
                dtype: dt,
            }),
        }
    }

    fn xrgb_unpack(
        &self,
        packed: &Tensor<CpuStorage<'_>>,
        r: &mut Tensor<CpuStorage<'_>>,
        g: &mut Tensor<CpuStorage<'_>>,
        b: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_dtype("xrgb_unpack", packed, DType::I32)?;
        ensure_dtype("xrgb_unpack", r, DType::F32)?;
        ensure_dtype("xrgb_unpack", g, DType::F32)?;
        ensure_dtype("xrgb_unpack", b, DType::F32)?;
        ensure_writable("xrgb_unpack", r)?;
        ensure_writable("xrgb_unpack", g)?;
        ensure_writable("xrgb_unpack", b)?;
        ensure_same_shape("xrgb_unpack", packed, r)?;
        ensure_same_shape("xrgb_unpack", packed, g)?;
        ensure_same_shape("xrgb_unpack", packed, b)?;
        ensure_disjoint("xrgb_unpack", r.storage(), packed.storage())?;
        ensure_disjoint("xrgb_unpack", g.storage(), packed.storage())?;
        ensure_disjoint("xrgb_unpack", b.storage(), packed.storage())?;
        ensure_disjoint("xrgb_unpack", r.storage(), g.storage())?;
        ensure_disjoint("xrgb_unpack", r.storage(), b.storage())?;
        ensure_disjoint("xrgb_unpack", g.storage(), b.storage())?;
        let packed = self.contiguous(packed)?;
        match r.dtype() {
            DType::F32 => {
                arch::raster::xrgb_unpack(
                    unsafe { as_u32(contiguous_bytes(&packed, "xrgb_unpack")?) },
                    unsafe { as_f32_mut(contiguous_bytes_mut(r, "xrgb_unpack")?) },
                    unsafe { as_f32_mut(contiguous_bytes_mut(g, "xrgb_unpack")?) },
                    unsafe { as_f32_mut(contiguous_bytes_mut(b, "xrgb_unpack")?) },
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "xrgb_unpack",
                dtype: dt,
            }),
        }
    }

    fn xrgb_pack(
        &self,
        r: &Tensor<CpuStorage<'_>>,
        g: &Tensor<CpuStorage<'_>>,
        b: &Tensor<CpuStorage<'_>>,
        packed: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        ensure_dtype("xrgb_pack", r, DType::F32)?;
        ensure_dtype("xrgb_pack", g, DType::F32)?;
        ensure_dtype("xrgb_pack", b, DType::F32)?;
        ensure_dtype("xrgb_pack", packed, DType::I32)?;
        ensure_writable("xrgb_pack", packed)?;
        ensure_same_shape("xrgb_pack", r, g)?;
        ensure_same_shape("xrgb_pack", r, b)?;
        ensure_same_shape("xrgb_pack", r, packed)?;
        ensure_disjoint("xrgb_pack", packed.storage(), r.storage())?;
        ensure_disjoint("xrgb_pack", packed.storage(), g.storage())?;
        ensure_disjoint("xrgb_pack", packed.storage(), b.storage())?;
        let r = self.contiguous(r)?;
        let g = self.contiguous(g)?;
        let b = self.contiguous(b)?;
        match r.dtype() {
            DType::F32 => {
                arch::raster::xrgb_pack(
                    unsafe { as_f32(contiguous_bytes(&r, "xrgb_pack")?) },
                    unsafe { as_f32(contiguous_bytes(&g, "xrgb_pack")?) },
                    unsafe { as_f32(contiguous_bytes(&b, "xrgb_pack")?) },
                    unsafe { as_u32_mut(contiguous_bytes_mut(packed, "xrgb_pack")?) },
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "xrgb_pack",
                dtype: dt,
            }),
        }
    }

    fn fill_rect(&self, dst: &mut Tensor<CpuStorage<'_>>, pixel: u32) -> Result<(), CpuError> {
        ensure_packed_surface("fill_rect", dst)?;
        ensure_writable("fill_rect", dst)?;
        let elem = dst.dtype().block_byte_size();
        let pitch = dst.desc().layout.strides()[0] / elem;
        let (h, w) = (dst.shape().dim(0), dst.shape().dim(1));
        match dst.dtype() {
            DType::I32 => {
                let off = dst.desc().layout.offset();
                arch::raster::fill_rect_u32(
                    unsafe { as_u32_mut(&mut mut_bytes(dst, "fill_rect")?[off..]) },
                    pitch,
                    pixel,
                    w,
                    h,
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "fill_rect",
                dtype: dt,
            }),
        }
    }

    fn fill_blend(&self, dst: &mut Tensor<CpuStorage<'_>>, pixel: u32) -> Result<(), CpuError> {
        ensure_packed_surface("fill_blend", dst)?;
        ensure_writable("fill_blend", dst)?;
        let elem = dst.dtype().block_byte_size();
        let pitch = dst.desc().layout.strides()[0] / elem;
        let (h, w) = (dst.shape().dim(0), dst.shape().dim(1));
        match dst.dtype() {
            DType::I32 => {
                let off = dst.desc().layout.offset();
                arch::raster::fill_blend_u32(
                    unsafe { as_u32_mut(&mut mut_bytes(dst, "fill_blend")?[off..]) },
                    pitch,
                    pixel,
                    w,
                    h,
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "fill_blend",
                dtype: dt,
            }),
        }
    }

    fn blit_opaque(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        dst: &mut Tensor<CpuStorage<'_>>,
        width: usize,
        height: usize,
    ) -> Result<(), CpuError> {
        ensure_surface_region("blit_opaque", src, width, height)?;
        ensure_surface_region("blit_opaque", dst, width, height)?;
        ensure_writable("blit_opaque", dst)?;
        ensure_disjoint("blit_opaque", dst.storage(), src.storage())?;
        // Pitch = row stride in elements (strides are in bytes for I32).
        // For a sliced view, strides reflect the *original* allocation's
        // row width, not the view's shape.dim(-1).
        let elem = src.dtype().block_byte_size();
        let src_pitch = src.desc().layout.strides()[0] / elem;
        let dst_pitch = dst.desc().layout.strides()[0] / elem;
        match src.dtype() {
            DType::I32 => {
                // Offset byte pointers by layout.offset() to respect
                // sliced views (slice sets the offset without copying).
                let src_off = src.desc().layout.offset();
                let dst_off = dst.desc().layout.offset();
                arch::raster::blit_opaque(
                    unsafe { as_u32(&src.storage().as_bytes()[src_off..]) },
                    src_pitch,
                    unsafe { as_u32_mut(&mut mut_bytes(dst, "blit_opaque")?[dst_off..]) },
                    dst_pitch,
                    width,
                    height,
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "blit_opaque",
                dtype: dt,
            }),
        }
    }

    fn blit_blend(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        dst: &mut Tensor<CpuStorage<'_>>,
        width: usize,
        height: usize,
    ) -> Result<(), CpuError> {
        ensure_surface_region("blit_blend", src, width, height)?;
        ensure_surface_region("blit_blend", dst, width, height)?;
        ensure_writable("blit_blend", dst)?;
        ensure_disjoint("blit_blend", dst.storage(), src.storage())?;
        let elem = src.dtype().block_byte_size();
        let src_pitch = src.desc().layout.strides()[0] / elem;
        let dst_pitch = dst.desc().layout.strides()[0] / elem;
        match src.dtype() {
            DType::I32 => {
                let src_off = src.desc().layout.offset();
                let dst_off = dst.desc().layout.offset();
                arch::raster::blit_blend(
                    unsafe { as_u32(&src.storage().as_bytes()[src_off..]) },
                    src_pitch,
                    unsafe { as_u32_mut(&mut mut_bytes(dst, "blit_blend")?[dst_off..]) },
                    dst_pitch,
                    width,
                    height,
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "blit_blend",
                dtype: dt,
            }),
        }
    }

    fn blit_nearest(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        dst: &mut Tensor<CpuStorage<'_>>,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<(), CpuError> {
        ensure_surface_region("blit_nearest", src, src_w, src_h)?;
        ensure_surface_region("blit_nearest", dst, dst_w, dst_h)?;
        ensure_writable("blit_nearest", dst)?;
        ensure_disjoint("blit_nearest", dst.storage(), src.storage())?;
        let elem = src.dtype().block_byte_size();
        let src_pitch = src.desc().layout.strides()[0] / elem;
        let dst_pitch = dst.desc().layout.strides()[0] / elem;
        match src.dtype() {
            DType::I32 => {
                let src_off = src.desc().layout.offset();
                let dst_off = dst.desc().layout.offset();
                arch::raster::blit_nearest(
                    unsafe { as_u32(&src.storage().as_bytes()[src_off..]) },
                    src_pitch,
                    src_w,
                    src_h,
                    unsafe { as_u32_mut(&mut mut_bytes(dst, "blit_nearest")?[dst_off..]) },
                    dst_pitch,
                    dst_w,
                    dst_h,
                );
                Ok(())
            }
            dt => Err(CpuError::UnsupportedOp {
                op: "blit_nearest",
                dtype: dt,
            }),
        }
    }

    fn surface_mut<'a>(
        &self,
        dst: &'a mut Tensor<CpuStorage<'_>>,
    ) -> Result<axnn::SurfaceMut<'a>, CpuError> {
        let surface = surface_mut("surface_mut", dst)?;
        Ok(axnn::SurfaceMut {
            words: surface.words,
            pitch: surface.pitch,
            width: surface.width,
            height: surface.height,
        })
    }

    fn rounded_fill(
        &self,
        dst: &mut Tensor<CpuStorage<'_>>,
        pixel: u32,
        shape: RoundedShape,
    ) -> Result<(), CpuError> {
        let mut surface = surface_mut("rounded_fill", dst)?;
        shapes::rounded_fill(&mut surface, pixel, shape);
        Ok(())
    }

    fn border_ring(
        &self,
        dst: &mut Tensor<CpuStorage<'_>>,
        pixel: u32,
        shape: RoundedShape,
        width: u16,
    ) -> Result<(), CpuError> {
        let mut surface = surface_mut("border_ring", dst)?;
        shapes::border_ring(&mut surface, pixel, shape, width);
        Ok(())
    }

    fn shadow_fill(
        &self,
        dst: &mut Tensor<CpuStorage<'_>>,
        pixel: u32,
        shape: RoundedShape,
        falloff: &[u8],
    ) -> Result<(), CpuError> {
        const OP: &str = "shadow_fill";
        if falloff.is_empty() || falloff.len() > axnn::MAX_FALLOFF {
            return Err(CpuError::ShapeMismatch { op: OP });
        }
        let mut surface = surface_mut(OP, dst)?;
        shapes::shadow_fill(&mut surface, pixel, shape, falloff);
        Ok(())
    }

    fn coverage_blend(
        &self,
        coverage: &Tensor<CpuStorage<'_>>,
        color: u32,
        opacity: u8,
        dst: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        const OP: &str = "coverage_blend";
        let dtype = coverage.dtype();
        if !matches!(dtype, DType::U8 | DType::I16 | DType::I32) {
            return Err(CpuError::UnsupportedOp { op: OP, dtype });
        }
        ensure_same_shape(OP, coverage, dst)?;
        ensure_disjoint(OP, dst.storage(), coverage.storage())?;
        let (bytes, pitch, width, height) = plane_bytes(OP, coverage, dtype)?;
        let mut surface = surface_mut(OP, dst)?;
        for y in 0..height {
            let row = surface.row_mut(y, 0, width);
            let start = y * pitch;
            match dtype {
                DType::U8 => arch::raster::coverage_blend_u8(
                    &bytes[start..start + width],
                    color,
                    opacity,
                    row,
                ),
                DType::I16 => arch::raster::coverage_blend_u16(
                    &unsafe { as_u16(bytes) }[start..start + width],
                    color,
                    opacity,
                    row,
                ),
                _ => arch::raster::coverage_blend_a32(
                    &unsafe { as_u32(bytes) }[start..start + width],
                    color,
                    opacity,
                    row,
                ),
            }
        }
        Ok(())
    }

    fn blit_masked(
        &self,
        src: &Tensor<CpuStorage<'_>>,
        mask: &Tensor<CpuStorage<'_>>,
        dst: &mut Tensor<CpuStorage<'_>>,
    ) -> Result<(), CpuError> {
        const OP: &str = "blit_masked";
        ensure_same_shape(OP, src, dst)?;
        ensure_same_shape(OP, mask, dst)?;
        ensure_disjoint(OP, dst.storage(), src.storage())?;
        ensure_disjoint(OP, dst.storage(), mask.storage())?;
        let (src_bytes, src_pitch, width, height) = plane_bytes(OP, src, DType::I32)?;
        let (mask_bytes, mask_pitch, _, _) = plane_bytes(OP, mask, DType::U8)?;
        let src_words = unsafe { as_u32(src_bytes) };
        let mut surface = surface_mut(OP, dst)?;
        for y in 0..height {
            let row = surface.row_mut(y, 0, width);
            arch::raster::blend_row_masked(
                &src_words[y * src_pitch..y * src_pitch + width],
                &mask_bytes[y * mask_pitch..y * mask_pitch + width],
                row,
            );
        }
        Ok(())
    }
}

/// Compute the byte offset of an element in a contiguous tensor.
pub(crate) fn element_byte_offset(desc: &TensorDesc, indices: &[usize]) -> Result<usize, CpuError> {
    if indices.len() != desc.shape.rank() {
        return Err(CpuError::ShapeMismatch {
            op: "element_access",
        });
    }
    let strides = desc.layout.strides();
    let mut off = desc.layout.offset();
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= desc.shape.dim(i as isize) as usize {
            return Err(CpuError::ShapeMismatch {
                op: "element_access (out of bounds)",
            });
        }
        off = off
            .checked_add(idx.checked_mul(strides[i]).ok_or(CpuError::ShapeMismatch {
                op: "element_access",
            })?)
            .ok_or(CpuError::ShapeMismatch {
                op: "element_access",
            })?;
    }
    Ok(off)
}

/// Extract and validate row/col indices for a rank-2 Q1 tensor.
///
/// Q1 storage is block-padded: each row occupies `layout.strides()[0]`
/// bytes (= `ceil(cols/256) * 32`), so bit addressing must use the
/// row stride, not a flat element index.
pub(crate) fn q1_row_col(desc: &TensorDesc, indices: &[usize]) -> Result<(usize, usize), CpuError> {
    if desc.shape.rank() != 2 || indices.len() != 2 {
        return Err(CpuError::ShapeMismatch {
            op: "q1_element_access",
        });
    }
    let (row, col) = (indices[0], indices[1]);
    let rows = desc.shape.dim(0) as usize;
    let cols = desc.shape.dim(1) as usize;
    if row >= rows || col >= cols {
        return Err(CpuError::ShapeMismatch {
            op: "q1_element_access (out of bounds)",
        });
    }
    Ok((row, col))
}

/// Reborrow a tensor's explicitly writable storage as a mutable byte slice.
#[inline]
fn mut_bytes<'a>(
    t: &'a mut Tensor<CpuStorage<'_>>,
    op: &'static str,
) -> Result<&'a mut [u8], CpuError> {
    // SAFETY: callers either have no other operand or reject overlapping
    // storage before this helper. Provenance is checked by `as_bytes_mut`.
    unsafe { t.storage().as_bytes_mut() }.ok_or(CpuError::ReadOnlyStorage { op })
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::{vec, vec::Vec};

    use super::*;
    use axnn::{
        Backend, ComputeBackend, DType, F32, F64, I32, Layout, Scalar, argmax, argmin, shape,
        tensor,
    };

    /// 256 KB arena — fits on test stack.
    const TEST_ARENA: usize = 256 * 1024;

    fn cpu() -> CpuBackend<TEST_ARENA> {
        CpuBackend::new()
    }

    fn encode_f16(values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0_u8; values.len() * 2];
        for (value, output) in values.iter().zip(bytes.chunks_exact_mut(2)) {
            crate::cast::write_f16(output, *value);
        }
        bytes
    }

    fn decode_f16(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2).map(crate::cast::read_f16).collect()
    }

    #[test]
    fn alloc_returns_correct_size() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 3, 4], DType::F32);
        let t = be.alloc(&desc).unwrap();
        assert_eq!(t.size(), 96);
        assert!(t.is_contiguous());
    }

    // Overflow allocations must carry the arena's 32-byte alignment —
    // the f32/u32/f64 views over tensor storage are UB without it, and
    // `vec![0u8; n]` alone guarantees only align-1.
    #[cfg(feature = "alloc")]
    #[test]
    fn overflow_allocations_are_aligned() {
        // 64-byte arena: anything past the first few bytes overflows.
        let be = CpuBackend::<64>::new();
        for size in [1usize, 7, 33, 100, 4096] {
            let mut storage = be.arena_alloc_zeroed(size).unwrap();
            let bytes = storage.as_bytes_mut();
            assert_eq!(bytes.len(), size);
            assert!(bytes.iter().all(|&b| b == 0));
            assert_eq!(
                bytes.as_ptr() as usize % 32,
                0,
                "overflow allocation of {size} B not 32-byte aligned"
            );
        }
        // 1 and 7 fit the 64 B arena (offsets 0 and 32); the rest overflow.
        assert_eq!(be.overflow_count(), 3, "expected 3 heap overflows");
    }

    #[test]
    fn zeros_all_zero() {
        let be = cpu();
        let desc = TensorDesc::new(shape![16], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let mut buf = vec![0xFF; 64];
        be.download(&t, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn fill_f32_pattern() {
        let be = cpu();
        let desc = TensorDesc::new(shape![3], DType::F32);
        let pattern = 3.14f32.to_le_bytes();
        let t = be.fill(&desc, &pattern).unwrap();
        let mut buf = vec![0u8; 12];
        be.download(&t, &mut buf).unwrap();
        for chunk in buf.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!((val - 3.14).abs() < 1e-6);
        }
    }

    #[test]
    fn fill_wrong_pattern_size() {
        let be = cpu();
        let desc = TensorDesc::new(shape![3], DType::F32);
        let result = be.fill(&desc, &[0, 0]);
        assert_eq!(
            result.unwrap_err(),
            CpuError::PatternSizeMismatch {
                expected: 4,
                actual: 2
            }
        );
    }

    #[test]
    fn upload_download_f32() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let mut buf = vec![0u8; 16];
        be.download(&t, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    #[test]
    fn upload_wrong_size_errors() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let result = be.upload(&[0; 8], &desc);
        assert_eq!(
            result.unwrap_err(),
            CpuError::SizeMismatch {
                expected: 16,
                actual: 8
            }
        );
    }

    #[test]
    fn download_buffer_too_small() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let mut buf = vec![0u8; 4];
        assert!(be.download(&t, &mut buf).is_err());
    }

    #[test]
    fn transpose_download() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 3], DType::F32);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let t = t.transpose(0, 1).unwrap();
        let mut buf = vec![0u8; 24];
        be.download(&t, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn slice_download() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4, 3], DType::F32);
        let data: Vec<u8> = (0..12)
            .map(|i| i as f32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let view = t.slice(0, 1, 2).unwrap();
        let mut buf = vec![0u8; 24];
        be.download(&view, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn gemm_rejects_accumulator_dtype_and_geometry_before_dispatch() {
        let be = cpu();
        let lhs = be
            .zeros(&TensorDesc::new(shape![1, 1], DType::F64))
            .unwrap();
        let rhs = be
            .zeros(&TensorDesc::new(shape![1, 4], DType::F64))
            .unwrap();
        let mut too_small = be
            .zeros(&TensorDesc::new(shape![1, 1], DType::F64))
            .unwrap();
        assert_eq!(
            be.gemm(&lhs, &rhs, &mut too_small),
            Err(CpuError::ShapeMismatch { op: "gemm" })
        );

        let rhs = be
            .zeros(&TensorDesc::new(shape![1, 1], DType::F64))
            .unwrap();
        let mut wrong_dtype = be
            .zeros(&TensorDesc::new(shape![1, 1], DType::F32))
            .unwrap();
        assert_eq!(
            be.gemm(&lhs, &rhs, &mut wrong_dtype),
            Err(CpuError::ShapeMismatch { op: "gemm" })
        );
    }

    #[test]
    fn packed_gemm_uses_rhs_rows_and_rejects_partial_blocks() {
        let be = cpu();
        let weights = be
            .zeros(&TensorDesc::new(shape![2, 32], DType::Q8))
            .unwrap();
        let activations = be
            .zeros(&TensorDesc::new(shape![3, 32], DType::Q8))
            .unwrap();
        let mut output = be
            .zeros(&TensorDesc::new(shape![2, 3], DType::F32))
            .unwrap();
        be.gemm(&weights, &activations, &mut output).unwrap();

        let partial_weights = be
            .zeros(&TensorDesc::new(shape![1, 31], DType::Q8))
            .unwrap();
        let partial_activations = be
            .zeros(&TensorDesc::new(shape![1, 31], DType::Q8))
            .unwrap();
        let mut partial_output = be
            .zeros(&TensorDesc::new(shape![1, 1], DType::F32))
            .unwrap();
        assert_eq!(
            be.gemm(&partial_weights, &partial_activations, &mut partial_output),
            Err(CpuError::ShapeMismatch { op: "gemm" })
        );
    }

    #[test]
    fn vec_dot_rejects_mismatched_lengths_and_partial_blocks() {
        let be = cpu();
        let lhs = be.zeros(&TensorDesc::new(shape![32], DType::Q8)).unwrap();
        let longer = be.zeros(&TensorDesc::new(shape![64], DType::Q8)).unwrap();
        assert_eq!(
            be.vec_dot(&lhs, &longer),
            Err(CpuError::ShapeMismatch { op: "vec_dot" })
        );

        let partial_lhs = be.zeros(&TensorDesc::new(shape![31], DType::Q8)).unwrap();
        let partial_rhs = be.zeros(&TensorDesc::new(shape![31], DType::Q8)).unwrap();
        assert_eq!(
            be.vec_dot(&partial_lhs, &partial_rhs),
            Err(CpuError::ShapeMismatch { op: "vec_dot" })
        );
    }

    #[test]
    fn unary_uses_only_the_contiguous_view_window() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4, 3], DType::F32);
        let data: Vec<u8> = (0..12)
            .map(|i| i as f32)
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensor = be.upload(&data, &desc).unwrap();
        let view = tensor.slice(0, 1, 2).unwrap();

        let output = be.unary(UnaryOp::Neg, &view).unwrap();

        assert_eq!(output.desc().layout.offset(), 0);
        let mut bytes = vec![0_u8; output.size()];
        be.download(&output, &mut bytes).unwrap();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![-3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    }

    #[test]
    fn f16_binary_broadcast_is_right_aligned() {
        let be = cpu();
        let lhs = be
            .upload(
                &encode_f16(&[1.0, 2.0, 3.0]),
                &TensorDesc::new(shape![3], DType::F16),
            )
            .unwrap();
        let rhs = be
            .upload(
                &encode_f16(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                &TensorDesc::new(shape![2, 3], DType::F16),
            )
            .unwrap();

        let output = be.binary(BinaryOp::Add, &lhs, &rhs).unwrap();
        let mut bytes = vec![0_u8; output.size()];
        be.download(&output, &mut bytes).unwrap();

        assert_eq!(output.shape(), &shape![2, 3]);
        assert_eq!(decode_f16(&bytes), vec![11.0, 22.0, 33.0, 41.0, 52.0, 63.0]);
    }

    #[test]
    fn f16_ternary_broadcast_is_right_aligned() {
        let be = cpu();
        let values = be
            .upload(
                &encode_f16(&[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]),
                &TensorDesc::new(shape![2, 3], DType::F16),
            )
            .unwrap();
        let low = be
            .upload(&encode_f16(&[1.0]), &TensorDesc::new(shape![1], DType::F16))
            .unwrap();
        let high = be
            .upload(
                &encode_f16(&[3.0, 5.0, 7.0]),
                &TensorDesc::new(shape![3], DType::F16),
            )
            .unwrap();

        let output = be.ternary(TernaryOp::Clamp, &values, &low, &high).unwrap();
        let mut bytes = vec![0_u8; output.size()];
        be.download(&output, &mut bytes).unwrap();

        assert_eq!(decode_f16(&bytes), vec![1.0, 1.0, 2.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn contiguous_noop_shares_storage() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let c = be.contiguous(&t).unwrap();
        assert!(t.storage().ptr_eq(c.storage()));
    }

    #[test]
    fn contiguous_materializes_transpose() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 3], DType::F32);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap().transpose(0, 1).unwrap();
        assert!(!t.is_contiguous());
        let c = be.contiguous(&t).unwrap();
        assert!(c.is_contiguous());
        let mut buf = vec![0u8; 24];
        be.download(&c, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn cast_f32_to_f64() {
        let be = cpu();
        let desc = TensorDesc::new(shape![3], DType::F32);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let c = be.cast(&t, DType::F64).unwrap();
        assert_eq!(c.dtype(), DType::F64);
        let mut buf = vec![0u8; 24];
        be.download(&c, &mut buf).unwrap();
        let values: Vec<f64> = buf
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cast_identity_noop() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let c = be.cast(&t, DType::F32).unwrap();
        assert!(t.storage().ptr_eq(c.storage()));
    }

    #[test]
    fn cast_f32_to_q8_roundtrip() {
        let be = cpu();
        let desc = TensorDesc::new(shape![32], DType::F32);
        let data: Vec<u8> = (0..32)
            .map(|i| (i as f32 - 16.0) * 0.1)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let q = be.cast(&t, DType::Q8).unwrap();
        let back = be.cast(&q, DType::F32).unwrap();
        let mut buf = vec![0u8; 128];
        be.download(&back, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for i in 0..32 {
            let src = (i as f32 - 16.0) * 0.1;
            assert!((src - values[i]).abs() < 0.02, "q8 elem {i}");
        }
    }

    #[test]
    fn cast_f32_to_q6_roundtrip() {
        let be = cpu();
        let desc = TensorDesc::new(shape![256], DType::F32);
        let data: Vec<u8> = (0..256)
            .map(|i| (i as f32 / 128.0 - 1.0) * 2.0)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let q = be.cast(&t, DType::Q6).unwrap();
        let back = be.cast(&q, DType::F32).unwrap();
        let mut buf = vec![0u8; 1024];
        be.download(&back, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let max_err = (0..256)
            .map(|i| ((i as f32 / 128.0 - 1.0) * 2.0 - values[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.15, "q6 max_err={max_err}");
    }

    #[test]
    fn cast_f32_to_q4_roundtrip() {
        let be = cpu();
        let desc = TensorDesc::new(shape![256], DType::F32);
        let data: Vec<u8> = (0..256)
            .map(|i| i as f32 / 128.0 - 1.0)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = be.upload(&data, &desc).unwrap();
        let q = be.cast(&t, DType::Q4).unwrap();
        let back = be.cast(&q, DType::F32).unwrap();
        let mut buf = vec![0u8; 1024];
        be.download(&back, &mut buf).unwrap();
        let values: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let max_err = (0..256)
            .map(|i| ((i as f32 / 128.0 - 1.0) - values[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.3, "q4 max_err={max_err}");
    }

    #[test]
    fn cast_quant_to_quant_unsupported() {
        let be = cpu();
        let desc = TensorDesc::new(shape![256], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let q = be.cast(&t, DType::Q8).unwrap();
        assert_eq!(
            be.cast(&q, DType::Q4).unwrap_err(),
            CpuError::UnsupportedCast {
                from: DType::Q8,
                to: DType::Q4
            }
        );
    }

    #[test]
    fn cast_quant_to_non_f32_unsupported() {
        let be = cpu();
        let desc = TensorDesc::new(shape![32], DType::F32);
        let t = be.zeros(&desc).unwrap();
        let q = be.cast(&t, DType::Q8).unwrap();
        assert_eq!(
            be.cast(&q, DType::I32).unwrap_err(),
            CpuError::UnsupportedCast {
                from: DType::Q8,
                to: DType::I32
            }
        );
    }

    #[test]
    fn cast_unaligned_elements_errors() {
        let be = cpu();
        let desc = TensorDesc::new(shape![33], DType::F32);
        let t = be.zeros(&desc).unwrap();
        assert_eq!(
            be.cast(&t, DType::Q8).unwrap_err(),
            CpuError::UnsupportedCast {
                from: DType::F32,
                to: DType::Q8
            }
        );
    }

    #[test]
    fn arg_reduce_f32_first_tie() {
        let be = cpu();
        let t = tensor!(&be, [3.0, -2.0, 5.0, -2.0, 5.0]).unwrap();
        assert_eq!(argmin(&t, &be).unwrap(), 1);
        assert_eq!(argmax(&t, &be).unwrap(), 2);
    }

    #[test]
    fn reset_reclaims_memory() {
        let mut be = cpu();
        let desc = TensorDesc::new(shape![1024], DType::F32);
        let _ = be.zeros(&desc).unwrap();
        assert!(be.used() > 0);
        be.reset();
        assert_eq!(be.used(), 0);
    }

    // ── tensor! macro ──────────────────────────────────────────

    #[test]
    fn tensor_macro_f32_literal() {
        let be = cpu();
        let t = tensor!(&be, [1.0, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::F32);
        let mut buf = [0u8; 12];
        be.download(&t, &mut buf).unwrap();
        let vals: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_macro_zeros() {
        let be = cpu();
        let t = tensor!(&be, F32, shape![2, 4]).unwrap();
        assert_eq!(t.shape().dims(), &[2, 4]);
        assert_eq!(t.dtype(), DType::F32);
        let mut buf = [0u8; 32];
        be.download(&t, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn tensor_macro_f64_zeros() {
        let be = cpu();
        let t = tensor!(&be, F64, shape![2]).unwrap();
        assert_eq!(t.shape().dims(), &[2]);
        assert_eq!(t.dtype(), DType::F64);
    }

    #[test]
    fn tensor_macro_upload_bytes() {
        let be = cpu();
        let data = 42.0f32.to_le_bytes();
        let t = tensor!(&be, F32, shape![1], &data).unwrap();
        assert_eq!(t.shape().dims(), &[1]);
        let mut buf = [0u8; 4];
        be.download(&t, &mut buf).unwrap();
        assert_eq!(f32::from_le_bytes(buf), 42.0);
    }

    #[test]
    fn tensor_macro_i32_upload() {
        let be = cpu();
        let values: [i32; 3] = [10, 20, 30];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = tensor!(&be, I32, shape![3], &bytes).unwrap();
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::I32);
        let mut buf = [0u8; 12];
        be.download(&t, &mut buf).unwrap();
        let vals: Vec<i32> = buf
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn wrap_reuses_external_bytes_without_copy() {
        let be = cpu();
        let bytes = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let desc = TensorDesc::new(shape![4], DType::F32);

        let t = be.wrap(&bytes, &desc).unwrap();

        assert_eq!(t.storage().as_bytes().as_ptr(), bytes.as_ptr());
        assert_eq!(be.get_element(&t, &[2]).unwrap(), Scalar::F32(3.0));
    }

    #[test]
    fn wrap_mut_updates_external_buffer_in_place() {
        let be = cpu();
        let desc = TensorDesc::new(shape![4], DType::F32);
        let mut bytes = vec![0u8; desc.storage_bytes()];

        let t = be.wrap_mut(&mut bytes, &desc).unwrap();
        be.set_element(&t, &[1], 42.0f32.into()).unwrap();

        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![0.0, 42.0, 0.0, 0.0]);
    }

    #[test]
    fn immutable_wrap_rejects_safe_mutation() {
        let be = cpu();
        let desc = TensorDesc::new(shape![1], DType::F32);
        let bytes = 1.0f32.to_le_bytes();
        let t = be.wrap(&bytes, &desc).unwrap();

        assert_eq!(
            be.set_element(&t, &[0], 2.0f32.into()),
            Err(CpuError::ReadOnlyStorage { op: "set_element" })
        );
        assert_eq!(f32::from_le_bytes(bytes), 1.0);
    }

    #[test]
    fn wrap_rejects_misaligned_typed_storage() {
        let be = cpu();
        let desc = TensorDesc::new(shape![1], DType::F32);
        let bytes = [0u8; 8];
        let offset = (0..4)
            .find(|offset| unsafe { bytes.as_ptr().add(*offset) }.addr() % 4 != 0)
            .unwrap();

        assert_eq!(
            be.wrap(&bytes[offset..offset + 4], &desc).unwrap_err(),
            CpuError::MisalignedStorage { required: 4 }
        );
    }

    #[test]
    fn wrap_rejects_misaligned_q1_word_storage() {
        let be = cpu();
        let desc = TensorDesc::new(shape![256], DType::Q1);
        let bytes = [0u8; 36];
        let offset = (0..4)
            .find(|offset| unsafe { bytes.as_ptr().add(*offset) }.addr() % 4 != 0)
            .unwrap();

        assert_eq!(
            be.wrap(&bytes[offset..offset + 32], &desc).unwrap_err(),
            CpuError::MisalignedStorage { required: 4 }
        );
    }

    #[test]
    fn contiguous_fast_path_preserves_writable_provenance() {
        #[repr(align(4))]
        struct Aligned([u8; 4]);

        let be = cpu();
        let desc = TensorDesc::new(shape![1], DType::F32);
        let mut bytes = Aligned([0; 4]);
        let wrapped = be.wrap_mut(&mut bytes.0, &desc).unwrap();
        let contiguous = be.contiguous(&wrapped).unwrap();

        assert!(contiguous.storage().is_writable());
        assert!(contiguous.storage().ptr_eq(wrapped.storage()));
        be.set_element(&contiguous, &[0], 3.0f32.into()).unwrap();
        assert_eq!(f32::from_le_bytes(bytes.0), 3.0);
    }

    #[test]
    fn raster_rejects_aliased_input_and_output_storage() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 2], DType::I32);
        let fb = be.zeros(&desc).unwrap();
        let src = fb;
        let mut dst = fb;

        assert_eq!(
            be.blit_opaque(&src, &mut dst, 2, 2),
            Err(CpuError::AliasedStorage { op: "blit_opaque" })
        );
    }

    #[test]
    fn raster_rejects_u8_as_packed_u32_storage() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 2], DType::U8);
        let mut surface = be.zeros(&desc).unwrap();

        assert_eq!(
            be.fill_rect(&mut surface, 0xff00_0000),
            Err(CpuError::UnsupportedOp {
                op: "fill_rect",
                dtype: DType::U8,
            })
        );
    }

    #[test]
    fn raster_rejects_mismatched_shapes_before_simd() {
        let be = cpu();
        let e0 = be.zeros(&TensorDesc::new(shape![4], DType::F32)).unwrap();
        let e1 = be.zeros(&TensorDesc::new(shape![3], DType::F32)).unwrap();
        let e2 = be.zeros(&TensorDesc::new(shape![4], DType::F32)).unwrap();
        let color = be.zeros(&TensorDesc::new(shape![4], DType::I32)).unwrap();
        let mut fb = be.zeros(&TensorDesc::new(shape![4], DType::I32)).unwrap();

        assert_eq!(
            be.edge_coverage(&e0, &e1, &e2, 0.0, &color, &mut fb),
            Err(CpuError::ShapeMismatch {
                op: "edge_coverage"
            })
        );
    }

    #[test]
    fn raster_rejects_out_of_bounds_blit_region() {
        let be = cpu();
        let desc = TensorDesc::new(shape![2, 2], DType::I32);
        let src = be.zeros(&desc).unwrap();
        let mut dst = be.zeros(&desc).unwrap();

        assert_eq!(
            be.blit_opaque(&src, &mut dst, 3, 2),
            Err(CpuError::ShapeMismatch { op: "blit_opaque" })
        );
    }

    #[test]
    fn raster_rejects_non_pixel_aligned_row_pitch() {
        #[repr(align(4))]
        struct Aligned([u8; 16]);

        let be = cpu();
        let mut bytes = Aligned([0; 16]);
        let desc = TensorDesc {
            shape: shape![2, 2],
            dtype: DType::I32,
            layout: Layout::from_raw(&[5, 4], 0),
        };
        let mut surface = Tensor::new(desc, CpuStorage::from_bytes_mut(&mut bytes.0));
        assert_eq!(
            be.fill_rect(&mut surface, 0),
            Err(CpuError::ShapeMismatch { op: "fill_rect" })
        );
    }

    // ── Element access ──────────────────────────────────────────

    #[test]
    fn set_get_q1() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![8, 256]).unwrap();
        assert_eq!(be.get_element(&adj, &[0, 1]).unwrap(), Scalar::Bool(false));
        be.set_element(&adj, &[0, 1], true.into()).unwrap();
        assert_eq!(be.get_element(&adj, &[0, 1]).unwrap(), Scalar::Bool(true));
        be.set_element(&adj, &[0, 1], false.into()).unwrap();
        assert_eq!(be.get_element(&adj, &[0, 1]).unwrap(), Scalar::Bool(false));
    }

    #[test]
    fn set_get_f32() {
        let be = cpu();
        let t = tensor!(&be, F32, shape![3]).unwrap();
        be.set_element(&t, &[1], 42.0f32.into()).unwrap();
        assert_eq!(be.get_element(&t, &[1]).unwrap(), Scalar::F32(42.0));
    }

    #[test]
    fn get_element_out_of_bounds() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();
        assert!(be.get_element(&adj, &[300, 0]).is_err());
    }

    #[test]
    fn malformed_descriptor_is_rejected_at_public_storage_boundaries() {
        let be = cpu();
        let desc = TensorDesc {
            shape: shape![2, 2],
            dtype: DType::F32,
            layout: Layout::from_raw(&[], 0),
        };

        let immutable = [0_u8; 16];
        let malformed = Tensor::new(desc, CpuStorage::from_bytes(&immutable));
        assert_eq!(
            be.get_element(&malformed, &[0, 0]),
            Err(CpuError::ShapeMismatch { op: "get_element" })
        );
        assert_eq!(
            be.download(&malformed, &mut [0_u8; 16]),
            Err(CpuError::ShapeMismatch { op: "download" })
        );

        let mut mutable = [0_u8; 16];
        let malformed = Tensor::new(desc, CpuStorage::from_bytes_mut(&mut mutable));
        assert_eq!(
            be.set_element(&malformed, &[0, 0], 1.0_f32.into()),
            Err(CpuError::ShapeMismatch { op: "set_element" })
        );
    }

    // ── Q1 graph operations ────────────────────────────────────

    /// Helper: set edge (i, j) in a Q1 adjacency tensor.
    fn set_edge(be: &CpuBackend<TEST_ARENA>, adj: &Tensor<CpuStorage<'_>>, i: usize, j: usize) {
        be.set_element(adj, &[i, j], true.into()).unwrap();
    }

    fn get_edge(
        be: &CpuBackend<TEST_ARENA>,
        adj: &Tensor<CpuStorage<'_>>,
        i: usize,
        j: usize,
    ) -> bool {
        match be.get_element(adj, &[i, j]).unwrap() {
            Scalar::Bool(v) => v,
            _ => panic!("expected Bool"),
        }
    }

    #[test]
    fn q1_topo_sort_diamond() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();
        set_edge(&be, &adj, 0, 1);
        set_edge(&be, &adj, 0, 2);
        set_edge(&be, &adj, 1, 3);
        set_edge(&be, &adj, 2, 3);

        let mut order = [0u16; 256];
        let count = be.topo_sort(&adj, &mut order).unwrap();
        assert_eq!(count, 256);
        let pos = |node: u16| order.iter().position(|&n| n == node).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn q1_topo_sort_cycle_fails() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();
        set_edge(&be, &adj, 0, 1);
        set_edge(&be, &adj, 1, 2);
        set_edge(&be, &adj, 2, 0);
        let mut order = [0u16; 256];
        assert!(be.topo_sort(&adj, &mut order).is_err());
    }

    #[test]
    fn q1_indegree() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();
        set_edge(&be, &adj, 0, 1);
        set_edge(&be, &adj, 0, 2);
        set_edge(&be, &adj, 1, 2);
        set_edge(&be, &adj, 2, 3);

        let mut deg = [0u32; 256];
        be.indegree(&adj, &mut deg).unwrap();
        assert_eq!(deg[0], 0);
        assert_eq!(deg[1], 1);
        assert_eq!(deg[2], 2);
        assert_eq!(deg[3], 1);
    }

    /// Regression: Q1 set/get must use row-stride addressing, not
    /// flat bit index. A 256×256 Q1 tensor has 32-byte padded rows.
    /// Setting bit (7, 6) must write to row 7's storage, not byte 28
    /// of a flat bit array.
    #[test]
    fn q1_set_get_row_stride_regression() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();

        // Set several bits across different rows.
        set_edge(&be, &adj, 7, 6);
        set_edge(&be, &adj, 7, 200);
        set_edge(&be, &adj, 0, 255);
        set_edge(&be, &adj, 100, 50);

        // Verify round-trip.
        assert!(get_edge(&be, &adj, 7, 6));
        assert!(get_edge(&be, &adj, 7, 200));
        assert!(get_edge(&be, &adj, 0, 255));
        assert!(get_edge(&be, &adj, 100, 50));

        // Verify no cross-row contamination.
        assert!(!get_edge(&be, &adj, 6, 6)); // row before 7
        assert!(!get_edge(&be, &adj, 8, 6)); // row after 7
        assert!(!get_edge(&be, &adj, 7, 5)); // adjacent col
        assert!(!get_edge(&be, &adj, 7, 7)); // adjacent col
    }

    /// Regression: topo_sort must see edges set via set_element.
    /// Verifies that set_element and bitmat::topo_sort agree on
    /// the Q1 row-stride layout.
    #[test]
    fn q1_topo_sort_agrees_with_set_element() {
        let be = cpu();
        let adj = tensor!(&be, axnn::Q1, shape![256, 256]).unwrap();

        // Chain: 0→1→2→3
        set_edge(&be, &adj, 0, 1);
        set_edge(&be, &adj, 1, 2);
        set_edge(&be, &adj, 2, 3);

        let mut order = [0u16; 256];
        let count = be.topo_sort(&adj, &mut order).unwrap();
        assert_eq!(count, 256);

        let pos = |n: u16| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
    }
}
