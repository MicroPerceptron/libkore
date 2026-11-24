//! Memory utility kernels: block copy, pattern fill, and friends.
//!
//! Plain byte loops — no SIMD. This is the portable baseline.

/// Copy `count` contiguous blocks of `block_bytes` each.
pub(crate) fn copy_blocks(src: &[u8], dst: &mut [u8], block_bytes: usize, count: usize) {
    let total = block_bytes * count;
    dst[..total].copy_from_slice(&src[..total]);
}

/// Fill `dst` by repeating `pattern` for `count` blocks.
pub(crate) fn fill_pattern(dst: &mut [u8], pattern: &[u8], count: usize) {
    let plen = pattern.len();
    for i in 0..count {
        let start = i * plen;
        dst[start..start + plen].copy_from_slice(pattern);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_blocks_basic() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut dst = [0u8; 6];
        copy_blocks(&src, &mut dst, 3, 2);
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn fill_pattern_basic() {
        let pattern = [0xDE, 0xAD];
        let mut dst = [0u8; 6];
        fill_pattern(&mut dst, &pattern, 3);
        assert_eq!(dst, [0xDE, 0xAD, 0xDE, 0xAD, 0xDE, 0xAD]);
    }
}
