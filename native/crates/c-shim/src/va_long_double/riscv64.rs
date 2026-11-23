//! RISC-V LP64D: a `long double` is a 2×XLEN scalar with 2×XLEN alignment, so a
//! variadic one takes an aligned (even, odd) pair of argument registers, or
//! the stack aligned to sixteen (psABI, "Integer Calling Convention" and
//! "va_list, va_start, and va_arg"). The prologue spills the argument
//! registers just below the stack arguments, a pair to a 16-aligned slot, so
//! one pointer walks both and `va_arg` aligns it.

use super::{BYTES, Float, VaList, align_up, read};

const _: () = assert!(core::mem::size_of::<VaList<'static>>() == core::mem::size_of::<*mut u8>());

/// The next argument, a binary128 `long double`.
///
/// # Safety
/// The next argument `list` holds is a `long double`.
pub(crate) unsafe fn next(list: &mut VaList<'_>) -> Float {
    // SAFETY: `VaList` is the platform's `va_list`, one pointer, on this
    // target (the size is asserted).
    let cursor = unsafe { &mut *(list as *mut VaList<'_>).cast::<*mut u8>() };
    let at = align_up(*cursor);
    *cursor = at.wrapping_add(BYTES);
    // SAFETY: the caller's contract puts the argument here.
    Float::from_quad(unsafe { read(at) })
}
