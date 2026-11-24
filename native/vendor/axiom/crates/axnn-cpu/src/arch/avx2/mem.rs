//! AVX2 memory utility kernels.
//!
//! These delegate to slice primitives — the compiler and libc already
//! use the widest available instructions for bulk copy and fill.
//! No benefit from hand-rolling intrinsics here.

pub(crate) fn copy_blocks(src: &[u8], dst: &mut [u8], block_bytes: usize, count: usize) {
    let total = block_bytes * count;
    dst[..total].copy_from_slice(&src[..total]);
}

pub(crate) fn fill_pattern(dst: &mut [u8], pattern: &[u8], count: usize) {
    let plen = pattern.len();
    for i in 0..count {
        let start = i * plen;
        dst[start..start + plen].copy_from_slice(pattern);
    }
}
