//! `qsort`'s algorithm: heapsort over raw elements of a C-supplied size.
//!
//! Heapsort because it is O(n log n) in the worst case, iterative (no
//! recursion to exhaust a 64 KiB worker stack), and in place. C's `qsort`
//! promises no stability, and this keeps none.

/// Sort `count` elements of `size` bytes at `base` by `before(a, b)`, which
/// answers whether `a` sorts strictly after `b` (a C comparator's `> 0`).
///
/// # Safety
/// `base` points to `count * size` writable bytes, and `after` reads only
/// the `size` bytes at each pointer it is given.
pub unsafe fn heapsort(
    base: *mut u8,
    count: usize,
    size: usize,
    after: &mut impl FnMut(*const u8, *const u8) -> bool,
) {
    if count < 2 || size == 0 {
        return;
    }
    let at = |index: usize| -> *mut u8 {
        // SAFETY: every index the algorithm forms is below `count`.
        unsafe { base.add(index * size) }
    };
    // SAFETY (the swaps): distinct indices name disjoint elements.
    let sift =
        |mut root: usize, end: usize, after: &mut dyn FnMut(*const u8, *const u8) -> bool| {
            loop {
                let left = 2 * root + 1;
                if left >= end {
                    return;
                }
                let right = left + 1;
                let child = if right < end && after(at(right), at(left)) {
                    right
                } else {
                    left
                };
                if !after(at(child), at(root)) {
                    return;
                }
                unsafe { core::ptr::swap_nonoverlapping(at(root), at(child), size) };
                root = child;
            }
        };
    let mut start = count / 2;
    while start > 0 {
        start -= 1;
        sift(start, count, after);
    }
    let mut end = count;
    while end > 1 {
        end -= 1;
        unsafe { core::ptr::swap_nonoverlapping(at(0), at(end), size) };
        sift(0, end, after);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    fn sort_u32(values: &mut [u32]) {
        let len = values.len();
        // SAFETY: the slice's bytes, elements of four bytes.
        unsafe {
            heapsort(values.as_mut_ptr().cast(), len, 4, &mut |a, b| {
                a.cast::<u32>().read_unaligned() > b.cast::<u32>().read_unaligned()
            })
        };
    }

    #[test]
    fn sorts_every_shape() {
        for len in 0..40 {
            let mut ascending: Vec<u32> = (0..len).collect();
            let mut descending: Vec<u32> = (0..len).rev().collect();
            let mut repeated: Vec<u32> = (0..len).map(|i| i % 3).collect();
            let mut scrambled: Vec<u32> =
                (0..len).map(|i| i.wrapping_mul(2654435761) % 97).collect();
            for values in [
                &mut ascending,
                &mut descending,
                &mut repeated,
                &mut scrambled,
            ] {
                let mut expected = values.clone();
                expected.sort_unstable();
                sort_u32(values);
                assert_eq!(*values, expected);
            }
        }
    }

    #[test]
    fn sorts_wide_elements_by_their_key() {
        // 24-byte records keyed on their first word, as C sorts structs.
        let mut records: Vec<[u64; 3]> = (0..17u64).map(|i| [(i * 7) % 17, i, !i]).collect();
        let len = records.len();
        // SAFETY: the vector's bytes, elements of 24 bytes.
        unsafe {
            heapsort(records.as_mut_ptr().cast(), len, 24, &mut |a, b| {
                a.cast::<u64>().read_unaligned() > b.cast::<u64>().read_unaligned()
            })
        };
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record[0], index as u64);
            // Each record moved whole.
            assert_eq!(record[2], !record[1]);
        }
    }
}
