use core::cell::UnsafeCell;

/// CPU tensor storage: a borrowed view into a slab-allocated buffer.
///
/// `Clone` is `Copy` — just duplicates the slice pointer. Multiple
/// `Tensor<CpuStorage<'p>>` values can share the same underlying
/// buffer with different descriptors (offset, strides), enabling
/// zero-copy views for transpose, slice, etc.
///
/// The lifetime `'p` ties the storage to the backend's pool, so
/// the borrow checker prevents use-after-free at compile time.
#[derive(Debug, Clone, Copy)]
enum CpuStorageBacking<'p> {
    /// Memory borrowed from [`CpuBackend::wrap`](crate::CpuBackend::wrap).
    ReadOnly(&'p [u8]),
    /// Backend-owned memory or memory borrowed from
    /// [`CpuBackend::wrap_mut`](crate::CpuBackend::wrap_mut).
    ///
    /// `UnsafeCell` is the Rust-level marker that this backing may be mutated
    /// through a shared tensor handle (for example `ComputeBackend::set_element`).
    Writable(&'p UnsafeCell<[u8]>),
}

#[derive(Debug, Clone, Copy)]
pub struct CpuStorage<'p> {
    backing: CpuStorageBacking<'p>,
}

impl<'p> CpuStorage<'p> {
    /// Wrap a borrowed immutable byte slice as storage.
    #[inline]
    pub fn from_bytes(bytes: &'p [u8]) -> Self {
        CpuStorage {
            backing: CpuStorageBacking::ReadOnly(bytes),
        }
    }

    /// Wrap a borrowed mutable byte slice as storage.
    ///
    /// Taking `&mut [u8]` preserves the exclusive borrow for the
    /// returned storage lifetime, which is what backend in-place paths
    /// rely on when they later reborrow the bytes mutably.
    #[inline]
    pub fn from_bytes_mut(bytes: &'p mut [u8]) -> Self {
        CpuStorage {
            backing: CpuStorageBacking::Writable(UnsafeCell::from_mut(bytes)),
        }
    }

    /// Borrow the underlying byte slice for a backend-internal read.
    ///
    /// This remains crate-private so safe callers cannot retain a shared byte
    /// reference while another aliased tensor handle performs an interior
    /// write. Public reads go through backend operations such as `download`
    /// and `TensorElementAccess::get`, which copy their result.
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self.backing {
            CpuStorageBacking::ReadOnly(bytes) => bytes,
            // SAFETY: backend read borrows are scoped to one synchronous
            // operation. In-place operations reject overlapping read/write
            // storage before obtaining their slices.
            CpuStorageBacking::Writable(bytes) => unsafe { &*bytes.get() },
        }
    }

    /// Copy readable storage bytes into `dst`, returning the number copied.
    ///
    /// This is the public raw-byte inspection path. It deliberately does not
    /// return a borrowed slice: writable storage supports interior mutation,
    /// so allowing a byte reference to escape would make aliased tensor
    /// handles unsound. Use [`Backend::download`](axnn::Backend::download)
    /// when a descriptor-aware contiguous copy is desired.
    pub fn copy_bytes_to(&self, dst: &mut [u8]) -> usize {
        let bytes = self.as_bytes();
        let copied = bytes.len().min(dst.len());
        dst[..copied].copy_from_slice(&bytes[..copied]);
        copied
    }

    /// Borrow writable storage after checking its provenance.
    ///
    /// # Safety
    ///
    /// The caller must ensure no overlapping byte reference is alive for the
    /// returned slice. Backend in-place entry points establish that by
    /// rejecting aliased input/output storage first.
    #[inline]
    pub(crate) unsafe fn as_bytes_mut(&self) -> Option<&mut [u8]> {
        match self.backing {
            CpuStorageBacking::ReadOnly(_) => None,
            // SAFETY: guaranteed by the caller; `UnsafeCell` is the storage's
            // explicit interior-mutation provenance.
            CpuStorageBacking::Writable(bytes) => Some(unsafe { &mut *bytes.get() }),
        }
    }

    /// Whether this handle carries writable storage provenance.
    #[inline]
    pub const fn is_writable(&self) -> bool {
        matches!(self.backing, CpuStorageBacking::Writable(_))
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        match self.backing {
            CpuStorageBacking::ReadOnly(bytes) => bytes.len(),
            CpuStorageBacking::Writable(bytes) => unsafe { (&*bytes.get()).len() },
        }
    }

    /// Whether two `CpuStorage` values point to the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        match self.backing {
            CpuStorageBacking::ReadOnly(bytes) => bytes.as_ptr(),
            CpuStorageBacking::Writable(bytes) => bytes.get().cast::<u8>(),
        }
    }

    /// Conservatively test whether two complete backing allocations overlap.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        let lhs_start = self.as_ptr().addr();
        let rhs_start = other.as_ptr().addr();
        let lhs_end = lhs_start.saturating_add(self.len());
        let rhs_end = rhs_start.saturating_add(other.len());
        lhs_start < rhs_end && rhs_start < lhs_end
    }
}

/// Mutable handle for writing into pool-owned memory.
///
/// Only the backend can create this (via `&mut` pool access).
/// Converted to an immutable `CpuStorage` after initialization.
pub(crate) struct CpuStorageMut<'p> {
    bytes: &'p mut [u8],
}

impl<'p> CpuStorageMut<'p> {
    pub(crate) fn new(bytes: &'p mut [u8]) -> Self {
        CpuStorageMut { bytes }
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.bytes
    }

    /// Freeze into an immutable storage handle.
    pub(crate) fn freeze(self) -> CpuStorage<'p> {
        CpuStorage::from_bytes_mut(self.bytes)
    }
}
