//! System V AMD64: a `long double` is class X87, which the caller always
//! passes in memory, in the overflow area, aligned to sixteen (psABI §3.5.7,
//! the `va_arg` algorithm's step for a type that is not passed in
//! registers).

use super::{BYTES, Float, VaList, align_up, read};

/// `va_list`'s one element, as the psABI declares it (`__va_list_tag`).
#[repr(C)]
struct Tag {
    gp_offset: u32,
    fp_offset: u32,
    overflow_arg_area: *mut u8,
    reg_save_area: *mut u8,
}

const _: () = assert!(core::mem::size_of::<VaList<'static>>() == core::mem::size_of::<Tag>());

/// The next argument, an x87 `long double`.
///
/// # Safety
/// The next argument `list` holds is a `long double`.
pub(crate) unsafe fn next(list: &mut VaList<'_>) -> Float {
    // SAFETY: `VaList` is the platform's `va_list` element on this target
    // (the size is asserted), and the overflow area is the caller's frame.
    let tag = unsafe { &mut *(list as *mut VaList<'_>).cast::<Tag>() };
    let at = align_up(tag.overflow_arg_area);
    // SAFETY: the caller's contract puts the argument here.
    let bits = unsafe { read(at) };
    tag.overflow_arg_area = at.wrapping_add(BYTES);
    Float::from_x87(bits)
}
