//! The deliberately small hosted-C surface retained by the selected Lua core.

use core::{
    alloc::Layout,
    ffi::{c_char, c_double, c_int, c_uchar, c_void},
    ptr, slice,
};

const LUA_ALIGNMENT: usize = 16;

/// Lua's per-state allocator, backed by this task's capability-grown heap.
pub unsafe extern "C" fn allocate(
    _ud: *mut c_void,
    ptr: *mut c_void,
    old_size: usize,
    new_size: usize,
) -> *mut c_void {
    if new_size == 0 {
        if !ptr.is_null()
            && let Ok(layout) = Layout::from_size_align(old_size, LUA_ALIGNMENT)
        {
            // SAFETY: every pointer reaching this branch was returned below
            // with the same size/alignment, and Lua transfers it exactly once.
            unsafe { alloc::alloc::dealloc(ptr.cast(), layout) };
        }
        return ptr::null_mut();
    }
    if ptr.is_null() {
        let Ok(layout) = Layout::from_size_align(new_size, LUA_ALIGNMENT) else {
            return ptr::null_mut();
        };
        // SAFETY: the global allocator is kore-rt's task-local Heap.
        return unsafe { alloc::alloc::alloc(layout) }.cast();
    }
    let Ok(_new_layout) = Layout::from_size_align(new_size, LUA_ALIGNMENT) else {
        return ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(old_size, LUA_ALIGNMENT) else {
        return ptr::null_mut();
    };
    // SAFETY: `ptr` and `layout` describe the live allocation supplied by the
    // preceding callback; realloc preserves its contents to the shorter size.
    unsafe { alloc::alloc::realloc(ptr.cast(), layout, new_size) }.cast()
}

#[unsafe(no_mangle)]
extern "C" fn kore_lua_abort() -> ! {
    kore::runtime::thread::exit(kore::runtime::ExitCode::ABORTED)
}

macro_rules! unary_math {
    ($($name:ident),* $(,)?) => {$(
        #[unsafe(no_mangle)]
        extern "C" fn $name(value: c_double) -> c_double { libm::$name(value) }
    )*};
}

unary_math!(
    acos, asin, ceil, cos, cosh, exp, fabs, floor, log, log10, log2, sin, sinh, sqrt, tan, tanh,
    trunc
);

#[unsafe(no_mangle)]
extern "C" fn atan2(y: c_double, x: c_double) -> c_double {
    libm::atan2(y, x)
}
#[unsafe(no_mangle)]
extern "C" fn fmod(x: c_double, y: c_double) -> c_double {
    libm::fmod(x, y)
}
#[unsafe(no_mangle)]
extern "C" fn pow(x: c_double, y: c_double) -> c_double {
    libm::pow(x, y)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn frexp(value: c_double, exponent: *mut c_int) -> c_double {
    let (fraction, exp) = libm::frexp(value);
    if !exponent.is_null() {
        unsafe { *exponent = exp };
    }
    fraction
}
#[unsafe(no_mangle)]
extern "C" fn ldexp(value: c_double, exponent: c_int) -> c_double {
    libm::ldexp(value, exponent)
}

#[unsafe(no_mangle)]
extern "C" fn abs(value: c_int) -> c_int {
    value.wrapping_abs()
}

unsafe fn c_len(mut text: *const c_char) -> usize {
    let mut len = 0;
    while unsafe { *text } != 0 {
        len += 1;
        text = unsafe { text.add(1) };
    }
    len
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strnlen(text: *const c_char, maximum: usize) -> usize {
    let mut len = 0;
    while len < maximum && unsafe { *text.add(len) } != 0 {
        len += 1;
    }
    len
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strcmp(left: *const c_char, right: *const c_char) -> c_int {
    unsafe { strncmp(left, right, usize::MAX) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strncmp(left: *const c_char, right: *const c_char, maximum: usize) -> c_int {
    for index in 0..maximum {
        let a = unsafe { *left.add(index) } as c_uchar;
        let b = unsafe { *right.add(index) } as c_uchar;
        if a != b {
            return c_int::from(a) - c_int::from(b);
        }
        if a == 0 {
            return 0;
        }
    }
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strcpy(dst: *mut c_char, src: *const c_char) -> *mut c_char {
    let len = unsafe { c_len(src) };
    unsafe { ptr::copy_nonoverlapping(src, dst, len + 1) };
    dst
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strchr(text: *const c_char, needle: c_int) -> *mut c_char {
    let needle = needle as c_char;
    let mut at = text;
    loop {
        let byte = unsafe { *at };
        if byte == needle {
            return at.cast_mut();
        }
        if byte == 0 {
            return ptr::null_mut();
        }
        at = unsafe { at.add(1) };
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strspn(text: *const c_char, accept: *const c_char) -> usize {
    let mut len = 0;
    while unsafe { *text.add(len) } != 0 {
        if unsafe { strchr(accept, c_int::from(*text.add(len))) }.is_null() {
            break;
        }
        len += 1;
    }
    len
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strcspn(text: *const c_char, reject: *const c_char) -> usize {
    let mut len = 0;
    while unsafe { *text.add(len) } != 0 {
        if !unsafe { strchr(reject, c_int::from(*text.add(len))) }.is_null() {
            break;
        }
        len += 1;
    }
    len
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strpbrk(text: *const c_char, accept: *const c_char) -> *mut c_char {
    let offset = unsafe { strcspn(text, accept) };
    let result = unsafe { text.add(offset) };
    if unsafe { *result } == 0 {
        ptr::null_mut()
    } else {
        result.cast_mut()
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strstr(haystack: *const c_char, needle: *const c_char) -> *mut c_char {
    let needle_len = unsafe { c_len(needle) };
    if needle_len == 0 {
        return haystack.cast_mut();
    }
    let haystack_len = unsafe { c_len(haystack) };
    if needle_len > haystack_len {
        return ptr::null_mut();
    }
    for offset in 0..=haystack_len - needle_len {
        if unsafe { strncmp(haystack.add(offset), needle, needle_len) } == 0 {
            return unsafe { haystack.add(offset) }.cast_mut();
        }
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn memchr(memory: *const c_void, needle: c_int, len: usize) -> *mut c_void {
    let bytes = unsafe { slice::from_raw_parts(memory.cast::<u8>(), len) };
    match bytes.iter().position(|byte| *byte == needle as u8) {
        Some(offset) => unsafe { memory.cast::<u8>().add(offset) }.cast_mut().cast(),
        None => ptr::null_mut(),
    }
}

fn ascii(value: c_int) -> Option<u8> {
    u8::try_from(value).ok().filter(u8::is_ascii)
}

macro_rules! ctype {
    ($name:ident, $predicate:ident) => {
        #[unsafe(no_mangle)]
        extern "C" fn $name(value: c_int) -> c_int {
            ascii(value).is_some_and(|byte| byte.$predicate()) as c_int
        }
    };
}

ctype!(isalnum, is_ascii_alphanumeric);
ctype!(isalpha, is_ascii_alphabetic);
ctype!(iscntrl, is_ascii_control);
ctype!(isdigit, is_ascii_digit);
ctype!(isgraph, is_ascii_graphic);
ctype!(islower, is_ascii_lowercase);
ctype!(ispunct, is_ascii_punctuation);
ctype!(isupper, is_ascii_uppercase);
ctype!(isxdigit, is_ascii_hexdigit);

#[unsafe(no_mangle)]
extern "C" fn isspace(value: c_int) -> c_int {
    ascii(value).is_some_and(crate::numeric::c_locale_whitespace) as c_int
}

#[unsafe(no_mangle)]
extern "C" fn isprint(value: c_int) -> c_int {
    ascii(value).is_some_and(|byte| byte == b' ' || byte.is_ascii_graphic()) as c_int
}

#[unsafe(no_mangle)]
extern "C" fn tolower(value: c_int) -> c_int {
    ascii(value).map_or(value, |byte| c_int::from(byte.to_ascii_lowercase()))
}

#[unsafe(no_mangle)]
extern "C" fn toupper(value: c_int) -> c_int {
    ascii(value).map_or(value, |byte| c_int::from(byte.to_ascii_uppercase()))
}
