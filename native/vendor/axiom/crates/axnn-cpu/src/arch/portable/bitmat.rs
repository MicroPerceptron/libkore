//! Portable bit-packed matrix operations for Q1 tensors.
//!
//! Operates on flat Q1 row-major data where each row is
//! `ceil(N/256) × 32` bytes. Bit (i, j) lives at byte
//! `i * row_bytes + j/8`, bit position `j % 8`.
//!
//! Row-level bulk operations use explicit `core::simd` for guaranteed
//! vectorization. SIMD lane width matches the target register width
//! via the `WD` constant from the portable module.
//!
//! All collections use `heapless::Vec` with a fixed capacity of
//! [`MAX_NODES`] — no heap allocation. Callers must ensure `n ≤ MAX_NODES`.
//!
//! Primary use case: topological ordering of compute graphs (DAG
//! scheduling), where `bit[i][j] = 1` means "node i precedes node j."

use core::simd::prelude::*;

use heapless::Vec as HVec;

use super::W;

/// Maximum supported node count for matrix-level graph operations
/// (topo_sort, col_popcounts). Row-level ops have no cap.
pub const MAX_NODES: usize = 1024;

/// Byte-width SIMD lane count: W f32 lanes × 4 bytes = register width in bytes.
/// NEON: W=4 → WB=16 (128-bit). AVX2: W=8 → WB=32 (256-bit).
const WB: usize = W * 4;

/// OR row `src` into row `dst` in-place: `dst[b] |= src[b]`.
///
/// Uses `Simd<u8, WB>` — one register-width load/op/store per chunk,
/// 2× unrolled for pipeline fill.
/// Load WB bytes from `ptr` as a SIMD vector. No bounds check.
#[inline(always)]
unsafe fn load_wb(ptr: *const u8) -> Simd<u8, WB> {
    let arr = unsafe { core::ptr::read(ptr as *const [u8; WB]) };
    Simd::from_array(arr)
}

/// Store WB bytes from a SIMD vector to `ptr`. No bounds check.
#[inline(always)]
unsafe fn store_wb(ptr: *mut u8, v: Simd<u8, WB>) {
    unsafe { core::ptr::write(ptr as *mut [u8; WB], *v.as_array()) };
}

/// OR row `src` into row `dst` in-place: `dst[b] |= src[b]`.
pub fn or_row_into(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let chunks = n / WB;
    let quads = chunks / 4;
    unsafe {
        for i in 0..quads {
            let base = i * 4 * WB;
            let d0 = load_wb(dp.add(base));
            let d1 = load_wb(dp.add(base + WB));
            let d2 = load_wb(dp.add(base + 2 * WB));
            let d3 = load_wb(dp.add(base + 3 * WB));
            let s0 = load_wb(sp.add(base));
            let s1 = load_wb(sp.add(base + WB));
            let s2 = load_wb(sp.add(base + 2 * WB));
            let s3 = load_wb(sp.add(base + 3 * WB));
            store_wb(dp.add(base), d0 | s0);
            store_wb(dp.add(base + WB), d1 | s1);
            store_wb(dp.add(base + 2 * WB), d2 | s2);
            store_wb(dp.add(base + 3 * WB), d3 | s3);
        }
        let mut off = quads * 4 * WB;
        while off + WB <= n {
            store_wb(dp.add(off), load_wb(dp.add(off)) | load_wb(sp.add(off)));
            off += WB;
        }
    }
    for j in chunks * WB..n {
        dst[j] |= src[j];
    }
}

/// AND row `src` into row `dst` in-place: `dst[b] &= src[b]`.
pub fn and_row_into(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let chunks = n / WB;
    let quads = chunks / 4;
    unsafe {
        for i in 0..quads {
            let base = i * 4 * WB;
            let d0 = load_wb(dp.add(base));
            let d1 = load_wb(dp.add(base + WB));
            let d2 = load_wb(dp.add(base + 2 * WB));
            let d3 = load_wb(dp.add(base + 3 * WB));
            let s0 = load_wb(sp.add(base));
            let s1 = load_wb(sp.add(base + WB));
            let s2 = load_wb(sp.add(base + 2 * WB));
            let s3 = load_wb(sp.add(base + 3 * WB));
            store_wb(dp.add(base), d0 & s0);
            store_wb(dp.add(base + WB), d1 & s1);
            store_wb(dp.add(base + 2 * WB), d2 & s2);
            store_wb(dp.add(base + 3 * WB), d3 & s3);
        }
        let mut off = quads * 4 * WB;
        while off + WB <= n {
            store_wb(dp.add(off), load_wb(dp.add(off)) & load_wb(sp.add(off)));
            off += WB;
        }
    }
    for j in chunks * WB..n {
        dst[j] &= src[j];
    }
}

/// AND-NOT: `dst[b] = ~mask[b] & dst[b]`.
pub fn andn_row(dst: &mut [u8], mask: &[u8]) {
    debug_assert_eq!(dst.len(), mask.len());
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let mp = mask.as_ptr();
    let chunks = n / WB;
    let quads = chunks / 4;
    unsafe {
        for i in 0..quads {
            let base = i * 4 * WB;
            let d0 = load_wb(dp.add(base));
            let d1 = load_wb(dp.add(base + WB));
            let d2 = load_wb(dp.add(base + 2 * WB));
            let d3 = load_wb(dp.add(base + 3 * WB));
            let m0 = load_wb(mp.add(base));
            let m1 = load_wb(mp.add(base + WB));
            let m2 = load_wb(mp.add(base + 2 * WB));
            let m3 = load_wb(mp.add(base + 3 * WB));
            store_wb(dp.add(base), d0 & !m0);
            store_wb(dp.add(base + WB), d1 & !m1);
            store_wb(dp.add(base + 2 * WB), d2 & !m2);
            store_wb(dp.add(base + 3 * WB), d3 & !m3);
        }
        let mut off = quads * 4 * WB;
        while off + WB <= n {
            store_wb(dp.add(off), load_wb(dp.add(off)) & !load_wb(mp.add(off)));
            off += WB;
        }
    }
    for j in chunks * WB..n {
        dst[j] &= !mask[j];
    }
}

/// Test if an entire row is zero (no set bits).
pub fn row_is_zero(row: &[u8]) -> bool {
    let zero = Simd::<u8, WB>::splat(0);
    let chunks = row.len() / WB;
    for i in 0..chunks {
        let v = Simd::<u8, WB>::from_slice(&row[i * WB..]);
        if v != zero {
            return false;
        }
    }
    for &b in &row[chunks * WB..] {
        if b != 0 {
            return false;
        }
    }
    true
}

/// Count the number of set bits in a row (out-degree of a node).
///
/// Uses u64 `count_ones()` which maps to hardware `popcnt` / `cnt`.
pub fn row_popcount(row: &[u8]) -> u32 {
    let mut count = 0u32;
    for chunk in row.chunks_exact(8) {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        count += word.count_ones();
    }
    for &b in row.chunks_exact(8).remainder() {
        count += b.count_ones();
    }
    count
}

/// Compute column popcounts (in-degree) for all N columns across M rows.
///
/// `data` is the flat row-major Q1 matrix, `row_bytes` is bytes per row,
/// `n` is the number of columns (and rows for a square matrix).
/// Requires `n ≤ MAX_NODES`.
///
/// Returns `indeg[j]` = number of rows with bit j set.
///
/// Uses sparse iteration via `trailing_zeros` — cost is O(total_edges)
/// instead of O(N × M). For typical DAGs with ~10 edges/node this is
/// 10-25× faster than the naive per-bit scan.
pub fn col_popcounts(data: &[u8], row_bytes: usize, n: usize) -> HVec<u32, MAX_NODES> {
    debug_assert!(n <= MAX_NODES);
    let m = data.len() / row_bytes;
    let mut counts = HVec::new();
    counts.resize(n, 0).unwrap();
    for i in 0..m {
        let row = &data[i * row_bytes..(i + 1) * row_bytes];
        for_each_set_bit(row, n, |j| {
            counts[j] += 1;
        });
    }
    counts
}

/// Clear column `j` across all rows of a flat Q1 matrix.
///
/// Removes all incoming edges to node `j`.
pub fn clear_column(data: &mut [u8], row_bytes: usize, n: usize, j: usize) {
    debug_assert!(j < n);
    let byte_idx = j / 8;
    let bit_mask = !(1u8 << (j % 8));
    let m = data.len() / row_bytes;
    for i in 0..m {
        data[i * row_bytes + byte_idx] &= bit_mask;
    }
}

/// Iterate over set bit positions in `row`, calling `f(j)` for each.
///
/// Uses `trailing_zeros` to jump directly to set bits — O(popcount)
/// instead of O(row_len × 8). The `n` parameter caps iteration so
/// padding bits beyond the logical column count are never visited.
#[inline]
pub fn for_each_set_bit(row: &[u8], n: usize, mut f: impl FnMut(usize)) {
    let mut base = 0usize;
    for chunk in row.chunks_exact(8) {
        let mut word = u64::from_le_bytes(chunk.try_into().unwrap());
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let j = base + bit;
            if j >= n {
                return;
            }
            f(j);
            word &= word - 1; // clear lowest set bit
        }
        base += 64;
        if base >= n {
            return;
        }
    }
}

/// Topological sort of an N×N Q1 adjacency matrix via Kahn's algorithm.
///
/// `data` is the flat row-major bit matrix, `row_bytes` = bytes per row,
/// `n` = number of nodes. Bit (i,j) set means "i must precede j."
/// Requires `n ≤ MAX_NODES`.
///
/// Returns node indices in topological order, or `None` if a cycle exists.
pub fn topo_sort(data: &[u8], row_bytes: usize, n: usize) -> Option<HVec<u16, MAX_NODES>> {
    debug_assert!(n <= MAX_NODES);
    let mut indeg = col_popcounts(data, row_bytes, n);

    // Seed queue with zero-indegree nodes.
    let mut queue = HVec::<u16, MAX_NODES>::new();
    for i in 0..n {
        if indeg[i] == 0 {
            queue.push(i as u16).unwrap();
        }
    }

    let mut order = HVec::<u16, MAX_NODES>::new();
    let mut head = 0;

    while head < queue.len() {
        let node = queue[head] as usize;
        head += 1;
        order.push(node as u16).unwrap();

        // Sparse successor scan — only visits set bits.
        let row = &data[node * row_bytes..(node + 1) * row_bytes];
        for_each_set_bit(row, n, |j| {
            indeg[j] -= 1;
            if indeg[j] == 0 {
                queue.push(j as u16).unwrap();
            }
        });
    }

    if order.len() == n {
        Some(order)
    } else {
        None // cycle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 32;

    fn make_mat(n: usize) -> (alloc::vec::Vec<u8>, usize) {
        let row_bytes = ((n + 255) / 256) * CHUNK;
        let data = alloc::vec![0u8; n * row_bytes];
        (data, row_bytes)
    }

    fn set_bit(data: &mut [u8], row_bytes: usize, i: usize, j: usize) {
        data[i * row_bytes + j / 8] |= 1 << (j % 8);
    }

    fn get_bit(data: &[u8], row_bytes: usize, i: usize, j: usize) -> bool {
        (data[i * row_bytes + j / 8] >> (j % 8)) & 1 != 0
    }

    #[test]
    fn or_row_merges_bits() {
        let (mut data, rb) = make_mat(64);
        set_bit(&mut data, rb, 0, 10);
        set_bit(&mut data, rb, 0, 20);
        set_bit(&mut data, rb, 1, 30);
        set_bit(&mut data, rb, 1, 40);
        let (dst, src) = data.split_at_mut(rb);
        or_row_into(dst, &src[..rb]);
        // Re-borrow after split
        assert!(get_bit(&data, rb, 0, 10));
        assert!(get_bit(&data, rb, 0, 20));
        assert!(get_bit(&data, rb, 0, 30));
        assert!(get_bit(&data, rb, 0, 40));
    }

    #[test]
    fn and_row_intersects() {
        let (mut data, rb) = make_mat(64);
        set_bit(&mut data, rb, 0, 5);
        set_bit(&mut data, rb, 0, 10);
        set_bit(&mut data, rb, 1, 10);
        set_bit(&mut data, rb, 1, 15);
        let (dst, src) = data.split_at_mut(rb);
        and_row_into(dst, &src[..rb]);
        assert!(!get_bit(&data, rb, 0, 5)); // only in row 0
        assert!(get_bit(&data, rb, 0, 10)); // in both
        assert!(!get_bit(&data, rb, 0, 15)); // only in row 1
    }

    #[test]
    fn andn_clears_masked_bits() {
        let (mut data, rb) = make_mat(64);
        set_bit(&mut data, rb, 0, 5);
        set_bit(&mut data, rb, 0, 10);
        set_bit(&mut data, rb, 0, 20);
        set_bit(&mut data, rb, 1, 5);
        set_bit(&mut data, rb, 1, 20);
        let (dst, mask) = data.split_at_mut(rb);
        andn_row(dst, &mask[..rb]);
        assert!(!get_bit(&data, rb, 0, 5)); // cleared
        assert!(get_bit(&data, rb, 0, 10)); // untouched
        assert!(!get_bit(&data, rb, 0, 20)); // cleared
    }

    #[test]
    fn row_is_zero_check() {
        let (mut data, rb) = make_mat(64);
        assert!(row_is_zero(&data[..rb]));
        set_bit(&mut data, rb, 0, 5);
        assert!(!row_is_zero(&data[..rb]));
    }

    #[test]
    fn popcount_counts_bits() {
        let (mut data, rb) = make_mat(64);
        set_bit(&mut data, rb, 0, 1);
        set_bit(&mut data, rb, 0, 5);
        set_bit(&mut data, rb, 0, 63);
        assert_eq!(row_popcount(&data[..rb]), 3);
    }

    #[test]
    fn col_popcounts_basic() {
        let (mut data, rb) = make_mat(4);
        // 0→1, 0→2, 1→2, 2→3
        set_bit(&mut data, rb, 0, 1);
        set_bit(&mut data, rb, 0, 2);
        set_bit(&mut data, rb, 1, 2);
        set_bit(&mut data, rb, 2, 3);
        let deg = col_popcounts(&data, rb, 4);
        assert_eq!(deg.as_slice(), &[0, 1, 2, 1]);
    }

    #[test]
    fn clear_column_removes_edges() {
        let (mut data, rb) = make_mat(8);
        set_bit(&mut data, rb, 0, 3);
        set_bit(&mut data, rb, 1, 3);
        set_bit(&mut data, rb, 2, 3);
        clear_column(&mut data, rb, 8, 3);
        assert!(!get_bit(&data, rb, 0, 3));
        assert!(!get_bit(&data, rb, 1, 3));
        assert!(!get_bit(&data, rb, 2, 3));
    }

    // ── Topological sort ────────────────────────────────────────

    #[test]
    fn topo_sort_diamond() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let (mut data, rb) = make_mat(4);
        set_bit(&mut data, rb, 0, 1);
        set_bit(&mut data, rb, 0, 2);
        set_bit(&mut data, rb, 1, 3);
        set_bit(&mut data, rb, 2, 3);
        let order = topo_sort(&data, rb, 4).unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0);
        assert_eq!(order[3], 3);
    }

    #[test]
    fn topo_sort_linear() {
        let (mut data, rb) = make_mat(4);
        set_bit(&mut data, rb, 0, 1);
        set_bit(&mut data, rb, 1, 2);
        set_bit(&mut data, rb, 2, 3);
        assert_eq!(
            topo_sort(&data, rb, 4).unwrap().as_slice(),
            &[0u16, 1, 2, 3],
        );
    }

    #[test]
    fn topo_sort_cycle_returns_none() {
        let (mut data, rb) = make_mat(3);
        set_bit(&mut data, rb, 0, 1);
        set_bit(&mut data, rb, 1, 2);
        set_bit(&mut data, rb, 2, 0);
        assert!(topo_sort(&data, rb, 3).is_none());
    }

    #[test]
    fn topo_sort_disconnected() {
        let (data, rb) = make_mat(4);
        let order = topo_sort(&data, rb, 4).unwrap();
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn topo_sort_256_node_chain() {
        let (mut data, rb) = make_mat(256);
        for i in 0..255 {
            set_bit(&mut data, rb, i, i + 1);
        }
        let order = topo_sort(&data, rb, 256).unwrap();
        assert_eq!(order.len(), 256);
        assert_eq!(order[0], 0);
        assert_eq!(order[255], 255);
    }
}
