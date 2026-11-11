//! `malloc`, `calloc`, `realloc` and `free` over the task's global heap.
//!
//! C frees without a size and Rust deallocates with one, so every block
//! carries its payload size in a header just below the pointer C sees. The
//! header is one `max_align_t` wide, so the payload keeps C's alignment
//! guarantee. Overflow and exhaustion return null, as C expects; a size of
//! zero returns null too, which `free` and `realloc` accept.

extern crate alloc;

use core::alloc::Layout;
use core::ffi::c_void;
use core::ptr;

/// C's `max_align_t` on every kore target: the payload alignment `malloc`
/// promises, and the header's width.
const ALIGN: usize = 16;

fn layout(payload: usize) -> Option<Layout> {
    Layout::from_size_align(payload.checked_add(ALIGN)?, ALIGN).ok()
}

/// The payload size recorded below `payload`.
///
/// # Safety
/// `payload` came from this module and is live.
unsafe fn payload_size(payload: *mut c_void) -> usize {
    // SAFETY: the caller's contract; the header sits `ALIGN` bytes below.
    unsafe { payload.cast::<u8>().sub(ALIGN).cast::<usize>().read() }
}

/// Write `size` into the block's header and return its payload.
///
/// # Safety
/// `block` is a live allocation of `layout(size)`.
unsafe fn finish(block: *mut u8, size: usize) -> *mut c_void {
    if block.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the block is at least `ALIGN` bytes and `ALIGN`-aligned.
    unsafe {
        block.cast::<usize>().write(size);
        block.add(ALIGN).cast()
    }
}

#[unsafe(no_mangle)]
extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        return ptr::null_mut();
    }
    let Some(layout) = layout(size) else {
        return ptr::null_mut();
    };
    // SAFETY: the layout is non-zero; the header is written before return.
    unsafe { finish(alloc::alloc::alloc(layout), size) }
}

#[unsafe(no_mangle)]
extern "C" fn calloc(count: usize, size: usize) -> *mut c_void {
    let Some(total) = count.checked_mul(size).filter(|&total| total != 0) else {
        return ptr::null_mut();
    };
    let Some(layout) = layout(total) else {
        return ptr::null_mut();
    };
    // SAFETY: as `malloc`; the zeroed allocation covers the payload.
    unsafe { finish(alloc::alloc::alloc_zeroed(layout), total) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn realloc(payload: *mut c_void, size: usize) -> *mut c_void {
    if payload.is_null() {
        return malloc(size);
    }
    if size == 0 {
        // SAFETY: the caller's pointer came from this module.
        unsafe { free(payload) };
        return ptr::null_mut();
    }
    // SAFETY: a non-null pointer C passes back came from this module.
    let old = unsafe { payload_size(payload) };
    let (Some(old_layout), Some(_)) = (layout(old), layout(size)) else {
        return ptr::null_mut();
    };
    // SAFETY: the block and its layout are the live allocation; the new size
    // fits a layout (checked above).
    unsafe {
        let block = payload.cast::<u8>().sub(ALIGN);
        finish(alloc::alloc::realloc(block, old_layout, size + ALIGN), size)
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn free(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    // SAFETY: a non-null pointer C passes back came from this module, so
    // its header names the layout it was allocated with.
    unsafe {
        let size = payload_size(payload);
        if let Some(layout) = layout(size) {
            alloc::alloc::dealloc(payload.cast::<u8>().sub(ALIGN), layout);
        }
    }
}
