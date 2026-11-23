//! AAPCS64: a `long double` is a quad-precision floating-point type, passed in
//! the next SIMD and floating-point register while any remains, its save slot
//! in the register save area, and on the stack aligned to sixteen after that
//! (AAPCS64 §B.4, `va_arg` for a floating-point type).

use super::{BYTES, Float, VaList, align_up, read};

/// `va_list` as AAPCS64 declares it (`__va_list`).
#[repr(C)]
struct List {
    stack: *mut u8,
    gr_top: *mut u8,
    vr_top: *mut u8,
    gr_offs: i32,
    vr_offs: i32,
}

const _: () = assert!(core::mem::size_of::<VaList<'static>>() == core::mem::size_of::<List>());

/// One vector register's save slot.
const VECTOR_SLOT: i32 = 16;

/// The next argument, a binary128 `long double`.
///
/// # Safety
/// The next argument `list` holds is a `long double`.
pub(crate) unsafe fn next(list: &mut VaList<'_>) -> Float {
    // SAFETY: `VaList` is the platform's `va_list` on this target (the size
    // is asserted), and both areas are the caller's.
    let list = unsafe { &mut *(list as *mut VaList<'_>).cast::<List>() };
    let offset = list.vr_offs;
    let at = if offset < 0 && offset + VECTOR_SLOT <= 0 {
        list.vr_offs = offset + VECTOR_SLOT;
        list.vr_top.wrapping_offset(offset as isize)
    } else {
        if offset < 0 {
            // The registers ran out under this argument: the standard still
            // counts it against them before it reads the stack.
            list.vr_offs = offset + VECTOR_SLOT;
        }
        let at = align_up(list.stack);
        list.stack = at.wrapping_add(BYTES);
        at
    };
    // SAFETY: the caller's contract puts the argument here.
    Float::from_quad(unsafe { read(at) })
}
