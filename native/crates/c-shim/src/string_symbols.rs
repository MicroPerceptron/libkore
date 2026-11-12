//! The ISO string routines a C library imports beyond `compiler_builtins`'
//! `memcpy`/`memmove`/`memset`/`memcmp`/`strlen`.

use core::ffi::{CStr, c_char, c_int, c_void};

/// The bytes of a C string, without its NUL.
///
/// # Safety
/// `text` is NUL-terminated.
unsafe fn bytes<'a>(text: *const c_char) -> &'a [u8] {
    // SAFETY: the caller's contract.
    unsafe { CStr::from_ptr(text) }.to_bytes()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strchr(text: *const c_char, value: c_int) -> *mut c_char {
    let needle = value as u8;
    // SAFETY: C promises a NUL-terminated string.
    let body = unsafe { bytes(text) };
    // The terminator itself is part of the searchable string: `strchr(s, 0)`
    // answers the NUL's address.
    let found = if needle == 0 {
        Some(body.len())
    } else {
        body.iter().position(|&byte| byte == needle)
    };
    match found {
        // SAFETY: the offset lies within the string or on its NUL.
        Some(offset) => unsafe { text.add(offset) }.cast_mut(),
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strrchr(text: *const c_char, value: c_int) -> *mut c_char {
    let needle = value as u8;
    // SAFETY: C promises a NUL-terminated string.
    let body = unsafe { bytes(text) };
    let found = if needle == 0 {
        Some(body.len())
    } else {
        body.iter().rposition(|&byte| byte == needle)
    };
    match found {
        // SAFETY: as `strchr`.
        Some(offset) => unsafe { text.add(offset) }.cast_mut(),
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strcpy(target: *mut c_char, source: *const c_char) -> *mut c_char {
    // SAFETY: C promises `source` NUL-terminated and `target` large enough
    // for it and its NUL, not overlapping.
    unsafe {
        let len = bytes(source).len();
        core::ptr::copy_nonoverlapping(source, target, len + 1);
    }
    target
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strstr(haystack: *const c_char, needle: *const c_char) -> *mut c_char {
    // SAFETY: C promises both NUL-terminated.
    let (hay, pin) = unsafe { (bytes(haystack), bytes(needle)) };
    let Some((&first, rest)) = pin.split_first() else {
        return haystack.cast_mut();
    };
    if pin.len() > hay.len() {
        return core::ptr::null_mut();
    }
    // Scan for the first byte, then compare the rest in place.
    let last_start = hay.len() - pin.len();
    let mut at = 0;
    while let Some(skip) = hay[at..=last_start].iter().position(|&byte| byte == first) {
        let start = at + skip;
        if &hay[start + 1..start + pin.len()] == rest {
            // SAFETY: `start` lies within the haystack.
            return unsafe { haystack.add(start) }.cast_mut();
        }
        at = start + 1;
        if at > last_start {
            break;
        }
    }
    core::ptr::null_mut()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn memchr(source: *const c_void, value: c_int, len: usize) -> *mut c_void {
    let bytes = source.cast::<u8>();
    let needle = value as u8;
    for index in 0..len {
        // SAFETY: C promises `len` readable bytes at `source`.
        if unsafe { *bytes.add(index) } == needle {
            return unsafe { bytes.add(index) }.cast_mut().cast();
        }
    }
    core::ptr::null_mut()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strcmp(left: *const c_char, right: *const c_char) -> c_int {
    // SAFETY: both are NUL-terminated, so the bounded compare stops at a NUL.
    unsafe { strncmp(left, right, usize::MAX) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strncmp(left: *const c_char, right: *const c_char, len: usize) -> c_int {
    for index in 0..len {
        // SAFETY: C promises both strings readable up to their NUL or `len`.
        let (a, b) = unsafe { (*left.add(index) as u8, *right.add(index) as u8) };
        if a != b {
            return c_int::from(a) - c_int::from(b);
        }
        if a == 0 {
            break;
        }
    }
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strncpy(target: *mut c_char, source: *const c_char, len: usize) -> *mut c_char {
    let mut index = 0;
    // SAFETY: C promises `len` writable bytes at `target` and `source`
    // readable up to its NUL or `len`; the tail is NUL-padded, as ISO says.
    unsafe {
        while index < len && *source.add(index) != 0 {
            *target.add(index) = *source.add(index);
            index += 1;
        }
        while index < len {
            *target.add(index) = 0;
            index += 1;
        }
    }
    target
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strncat(target: *mut c_char, source: *const c_char, len: usize) -> *mut c_char {
    // SAFETY: C promises `target` NUL-terminated with room for at most `len`
    // more bytes and a NUL, and `source` readable up to its NUL or `len`.
    unsafe {
        let mut end = 0;
        while *target.add(end) != 0 {
            end += 1;
        }
        let mut index = 0;
        while index < len && *source.add(index) != 0 {
            *target.add(end + index) = *source.add(index);
            index += 1;
        }
        *target.add(end + index) = 0;
    }
    target
}
