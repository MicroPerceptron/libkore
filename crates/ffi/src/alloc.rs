//! Foreign allocation through the application's installed Rust allocator.
//! Private allocation metadata preserves size/alignment for unsized C free.
extern crate alloc;
use alloc::alloc::Global;
use core::{
    alloc::{Allocator, Layout},
    ffi::c_void,
    ptr::{self, NonNull},
};

/// Minimum alignment of malloc/calloc results on all supported Kore targets.
pub const KORE_ALLOC_MIN_ALIGNMENT: usize = 16;

#[derive(Clone, Copy)]
struct Header {
    layout: Layout,
    offset: usize,
    size: usize,
}

fn allocate_in<A: Allocator>(allocator: &A, size: usize, alignment: usize) -> *mut u8 {
    if !alignment.is_power_of_two() {
        return ptr::null_mut();
    }
    let alignment = alignment
        .max(KORE_ALLOC_MIN_ALIGNMENT)
        .max(align_of::<Header>());
    let Some(offset) = size_of::<Header>()
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
    else {
        return ptr::null_mut();
    };
    let Some(total) = offset.checked_add(size.max(1)) else {
        return ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(total, alignment) else {
        return ptr::null_mut();
    };
    let Ok(block) = allocator.allocate(layout) else {
        return ptr::null_mut();
    };
    // SAFETY: checked layout covers the aligned prefix, header and payload.
    unsafe {
        let data = block.cast::<u8>().as_ptr().add(offset);
        data.sub(size_of::<Header>())
            .cast::<Header>()
            .write(Header {
                layout,
                offset,
                size,
            });
        data
    }
}

unsafe fn release_in<A: Allocator>(allocator: &A, data: *mut u8) {
    if data.is_null() {
        return;
    }
    // SAFETY: caller provides a live allocation returned by allocate_in with
    // this allocator; the header and base are within the same allocation.
    unsafe {
        let header = data.sub(size_of::<Header>()).cast::<Header>().read();
        allocator.deallocate(
            NonNull::new_unchecked(data.sub(header.offset)),
            header.layout,
        );
    }
}

unsafe fn reallocate_in<A: Allocator>(allocator: &A, data: *mut u8, size: usize) -> *mut u8 {
    if data.is_null() {
        return allocate_in(allocator, size, KORE_ALLOC_MIN_ALIGNMENT);
    }
    if size == 0 {
        unsafe { release_in(allocator, data) };
        return ptr::null_mut();
    }
    // SAFETY: caller guarantees a live, exclusively owned allocation.
    let header = unsafe { data.sub(size_of::<Header>()).cast::<Header>().read() };
    let result = allocate_in(allocator, size, header.layout.align());
    if !result.is_null() {
        // SAFETY: disjoint live allocations; copy fits both payloads. Only
        // release the original after successful allocation and copying.
        unsafe {
            ptr::copy_nonoverlapping(data, result, header.size.min(size));
            release_in(allocator, data);
        }
    }
    result
}

/// Allocate size bytes using the application's installed global allocator.
/// Returns null on allocation/layout failure. Zero size requests allocate a
/// unique freeable block when successful. No allocator is installed by libkore.
#[unsafe(no_mangle)]
pub extern "C" fn kore_alloc_malloc(size: usize) -> *mut c_void {
    allocate_in(&Global, size, KORE_ALLOC_MIN_ALIGNMENT).cast()
}

/// Allocate count * size zeroed bytes. Multiplication overflow returns null.
/// Zero size follows kore_alloc_malloc. This API does not set errno.
#[unsafe(no_mangle)]
pub extern "C" fn kore_alloc_calloc(count: usize, size: usize) -> *mut c_void {
    let Some(bytes) = count.checked_mul(size) else {
        return ptr::null_mut();
    };
    let result = kore_alloc_malloc(bytes);
    if !result.is_null() {
        // SAFETY: successful allocation covers bytes writable payload bytes.
        unsafe { result.cast::<u8>().write_bytes(0, bytes) };
    }
    result
}

/// Allocate size bytes aligned to a nonzero power of two. Size need not be an
/// alignment multiple. Null means invalid layout or allocation failure.
/// Free with kore_alloc_free. Reallocation preserves the alignment.
#[unsafe(no_mangle)]
pub extern "C" fn kore_alloc_aligned(size: usize, alignment: usize) -> *mut c_void {
    allocate_in(&Global, size, alignment).cast()
}

/// Resize an allocation, preserving min(old size, new size) bytes and alignment.
/// Null input behaves as malloc. Non-null input with zero size is freed and
/// returns null. Other failures leave the original allocation live and intact.
/// # Safety
/// Non-null data is a live, exclusively owned pointer from kore_alloc_*.
/// Never pass a Rust allocation, another allocator's pointer, or an interior
/// pointer. The installed allocator must remain compatible for its lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_alloc_realloc(data: *mut c_void, size: usize) -> *mut c_void {
    unsafe { reallocate_in(&Global, data.cast(), size).cast() }
}

/// Release a kore_alloc_* allocation. Null is a no-op.
/// # Safety
/// Same allocation provenance as kore_alloc_realloc. Release exactly once;
/// callers must stop accessing the allocation before it is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_alloc_free(data: *mut c_void) {
    unsafe { release_in(&Global, data.cast()) };
}

/// Callback adapter using the application's installed global allocator.
/// Borrows no context. Explicit table allocators can still be supplied.
#[unsafe(no_mangle)]
pub extern "C" fn kore_alloc_hash_allocator() -> crate::KoreHashAllocator {
    unsafe extern "C" fn allocate(_: *mut c_void, size: usize, align: usize) -> *mut c_void {
        kore_alloc_aligned(size, align)
    }
    unsafe extern "C" fn deallocate(_: *mut c_void, p: *mut c_void, _: usize, _: usize) {
        unsafe { kore_alloc_free(p) };
    }
    crate::KoreHashAllocator {
        context: ptr::null_mut(),
        allocate: Some(allocate),
        deallocate: Some(deallocate),
    }
}

// Host allocators can themselves call libc malloc: never interpose these
// symbols on a hosted test binary. A Kore binary installs its native heap.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
mod c_symbols {
    use super::*;
    #[unsafe(no_mangle)]
    pub extern "C" fn malloc(size: usize) -> *mut c_void {
        kore_alloc_malloc(size)
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn calloc(count: usize, size: usize) -> *mut c_void {
        kore_alloc_calloc(count, size)
    }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn realloc(data: *mut c_void, size: usize) -> *mut c_void {
        unsafe { kore_alloc_realloc(data, size) }
    }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn free(data: *mut c_void) {
        unsafe { kore_alloc_free(data) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use kore::runtime::heap::{Heap, StaticMemSource};

    #[test]
    fn native_heap_alignment_resize_and_exhaustion() {
        // Actual Kore heap engine with host-owned backing, not a fake allocator.
        let mut backing = vec![0u8; 2 * 1024 * 1024];
        let heap =
            Heap::new_in(unsafe { StaticMemSource::over(backing.as_mut_ptr(), backing.len()) });
        unsafe {
            for alignment in [1, 16, 64, 4096] {
                let p = allocate_in(&heap, 37, alignment);
                assert!(!p.is_null());
                assert_eq!(p.addr() % alignment, 0);
                p.write_bytes(0xab, 37);
                // Valid layout, but larger than the source: actual OOM.
                assert!(reallocate_in(&heap, p, 4 * 1024 * 1024).is_null());
                assert_eq!(*p.add(36), 0xab);
                let q = reallocate_in(&heap, p, 300);
                assert!(!q.is_null());
                assert_eq!(q.addr() % alignment, 0);
                assert_eq!(*q.add(36), 0xab);
                let r = reallocate_in(&heap, q, 11);
                assert!(!r.is_null());
                assert_eq!(*r.add(10), 0xab);
                release_in(&heap, r);
            }
        }
    }

    #[test]
    fn public_api_zero_overflow_and_reallocation() {
        unsafe {
            assert!(kore_alloc_calloc(usize::MAX, 2).is_null());
            assert!(kore_alloc_malloc(usize::MAX).is_null());
            assert!(kore_alloc_aligned(8, 0).is_null());
            assert!(kore_alloc_aligned(8, 3).is_null());
            kore_alloc_free(ptr::null_mut());
            let zero = kore_alloc_malloc(0);
            assert!(!zero.is_null());
            kore_alloc_free(zero);
            let p = kore_alloc_calloc(5, 13).cast::<u8>();
            assert!(!p.is_null());
            assert!((0..65).all(|i| *p.add(i) == 0));
            p.write_bytes(7, 65);
            assert!(kore_alloc_realloc(p.cast(), usize::MAX).is_null());
            assert_eq!(*p.add(64), 7);
            assert!(kore_alloc_realloc(p.cast(), 0).is_null());
            let p = kore_alloc_realloc(ptr::null_mut(), 23);
            assert!(!p.is_null());
            kore_alloc_free(p);
        }
    }
}

#[cfg(all(test, feature = "test-foreign", not(miri), not(target_os = "none")))]
#[test]
fn compiled_c_and_zig_allocators() {
    unsafe extern "C" {
        fn c_check_alloc() -> u32;
        fn zig_check_alloc() -> u32;
    }
    // SAFETY: fixtures retain allocation provenance and obey the foreign API.
    unsafe {
        assert_eq!(c_check_alloc(), 0);
        assert_eq!(zig_check_alloc(), 0);
    }
}
