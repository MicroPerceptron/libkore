//! Voxel ray tracing — uniform-grid DDA and sparse voxel octree (SVO).
//!
//! Two occupancy representations over the same bit-packed grid (one
//! bit per voxel, x-major: `idx = x + nx·(y + ny·z)`, u64 words):
//!
//! * **DDA** ([`dda_trace`]) — Amanatides–Woo traversal of the dense
//!   grid: one occupancy test plus one comparison-and-add per crossed
//!   voxel. Optimal when rays hit quickly or grids are small/dense;
//!   cost grows linearly with empty distance crossed.
//! * **SVO** ([`svo_build`] / [`svo_trace`]) — a pointerless-style
//!   octree (8-bit child mask + index of the contiguous child block)
//!   traversed front-to-back with an explicit stack. Empty space is
//!   skipped a subtree at a time, so cost scales with occupied surface
//!   rather than volume — the classic win for sparse scenes.
//!
//! Rays live in grid space: voxel `(x,y,z)` spans `[x,x+1)³`; callers
//! scale world coordinates. Both tracers return the parametric entry
//! `t` into the first occupied voxel (clamped to 0 for origins inside
//! it) and its linear index, or ([`f32::INFINITY`], `u32::MAX`) on a
//! miss — bit-comparable across the two structures, which the tests
//! exploit.
//!
//! The single-ray traversals and volume marchers are scalar and
//! allocation-free (caller-provided node buffer, fixed traversal stack).
//! [`brick_trace_packet_n`] is an opt-in four-ray portable-SIMD traversal,
//! including occupied bricks when multiple lanes need a fine walk. Its
//! batching does not change the default scalar entry points.

use core::simd::prelude::*;

#[path = "voxel_hierarchy.rs"]
mod hierarchy;
pub use hierarchy::BrickHierarchy;

/// Miss sentinel: `(f32::INFINITY, u32::MAX)`.
pub const MISS: (f32, u32) = (f32::INFINITY, u32::MAX);

/// Deepest supported octree (2^10 = 1024³ voxels).
pub const SVO_MAX_LEVELS: u32 = 10;

/// Replace near-zero direction components so slab and step math stays
/// finite; 1e-12 never advances within any representable grid span.
#[inline(always)]
fn sanitize(d: f32) -> f32 {
    if d.abs() < 1e-12 {
        1e-12f32.copysign(d)
    } else {
        d
    }
}

#[inline(always)]
fn bit_get(bits: &[u64], idx: usize) -> bool {
    (bits[idx / 64] >> (idx % 64)) & 1 != 0
}

/// Any set bit in `[base, base + len)`? Word-at-a-time scan.
#[inline(always)]
fn row_any(bits: &[u64], base: usize, len: usize) -> bool {
    let (mut w, mut b) = (base / 64, base % 64);
    let mut remaining = len;
    while remaining > 0 {
        let take = (64 - b).min(remaining);
        let mask = (!0u64 >> (64 - take)) << b;
        if bits[w] & mask != 0 {
            return true;
        }
        remaining -= take;
        w += 1;
        b = 0;
    }
    false
}

// ── DDA: Amanatides–Woo dense-grid traversal ───────────────────────

/// Trace one ray through a dense occupancy grid.
///
/// `bits`: x-major bit grid covering `dims` voxels. Returns the entry
/// `t` (≥ 0, ≤ `t_max`) and linear index of the first occupied voxel,
/// or [`MISS`].
pub fn dda_trace(
    bits: &[u64],
    dims: [u32; 3],
    origin: [f32; 3],
    dir: [f32; 3],
    t_max: f32,
) -> (f32, u32) {
    debug_assert!(bits.len() * 64 >= (dims[0] * dims[1] * dims[2]) as usize);
    let d = [sanitize(dir[0]), sanitize(dir[1]), sanitize(dir[2])];

    // Clip to the grid AABB [0, dims]³.
    let (mut t0, mut t1) = (0.0f32, t_max);
    for a in 0..3 {
        let inv = 1.0 / d[a];
        let ta = (0.0 - origin[a]) * inv;
        let tb = (dims[a] as f32 - origin[a]) * inv;
        let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
        t0 = t0.max(lo);
        t1 = t1.min(hi);
    }
    if t0 > t1 {
        return MISS;
    }

    // Per-axis stepping state at the entry point.
    let mut cell = [0i32; 3];
    let mut step = [0i32; 3];
    let mut t_next = [0.0f32; 3];
    let mut t_delta = [0.0f32; 3];
    for a in 0..3 {
        let p = origin[a] + t0 * d[a];
        // `p` is clipped into [0, dims], so the `as` truncation is
        // floor (`f32::floor` itself is std-only).
        let c = (p as i32).clamp(0, dims[a] as i32 - 1);
        cell[a] = c;
        step[a] = if d[a] > 0.0 { 1 } else { -1 };
        t_delta[a] = 1.0 / d[a].abs();
        let boundary = if d[a] > 0.0 { c as f32 + 1.0 } else { c as f32 };
        t_next[a] = t0 + (boundary - p) / d[a];
    }

    let (nx, ny) = (dims[0] as i32, dims[1] as i32);
    // Maintain the linear index by stride increments instead of two
    // multiplies per step.
    let strides = [1i32, nx, nx * ny];
    let mut idx = cell[0] + nx * (cell[1] + ny * cell[2]);
    let mut t = t0;
    loop {
        if bit_get(bits, idx as usize) {
            return (t, idx as u32);
        }
        // Advance across the nearest boundary.
        let a = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
            0
        } else if t_next[1] <= t_next[2] {
            1
        } else {
            2
        };
        t = t_next[a];
        if t > t1 {
            return MISS;
        }
        cell[a] += step[a];
        if cell[a] < 0 || cell[a] >= dims[a] as i32 {
            return MISS;
        }
        idx += step[a] * strides[a];
        t_next[a] += t_delta[a];
    }
}

/// Batch [`dda_trace`]: `origins`/`dirs` are xyz-interleaved, one hit
/// `(out_t[i], out_idx[i])` per ray.
pub fn dda_trace_n(
    bits: &[u64],
    dims: [u32; 3],
    origins: &[f32],
    dirs: &[f32],
    n: usize,
    t_max: f32,
    out_t: &mut [f32],
    out_idx: &mut [u32],
) {
    debug_assert!(origins.len() >= n * 3 && dirs.len() >= n * 3);
    debug_assert!(out_t.len() >= n && out_idx.len() >= n);
    for i in 0..n {
        let o = [origins[i * 3], origins[i * 3 + 1], origins[i * 3 + 2]];
        let d = [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]];
        let (t, idx) = dda_trace(bits, dims, o, d, t_max);
        out_t[i] = t;
        out_idx[i] = idx;
    }
}

// ── SVO: build ─────────────────────────────────────────────────────

/// One octree node. Octant bit `i` = `x_hi | y_hi·2 | z_hi·4`; set
/// bits own a node each, stored contiguously from `first_child` in
/// ascending octant order. Nodes over 2³ cubes are leaf parents: the
/// mask holds the 8 voxel bits directly and `first_child` is unused.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SvoNode {
    pub child_mask: u8,
    pub first_child: u32,
}

#[inline(always)]
fn child_org(org: [u32; 3], half: u32, oct: u8) -> [u32; 3] {
    [
        org[0] + (oct as u32 & 1) * half,
        org[1] + ((oct as u32 >> 1) & 1) * half,
        org[2] + ((oct as u32 >> 2) & 1) * half,
    ]
}

/// Any occupied voxel in the `size`³ cube at `org`? (`n` = grid edge.)
fn region_occupied(bits: &[u64], n: u32, org: [u32; 3], size: u32) -> bool {
    for z in org[2]..org[2] + size {
        for y in org[1]..org[1] + size {
            let base = (org[0] + n * (y + n * z)) as usize;
            if row_any(bits, base, size as usize) {
                return true;
            }
        }
    }
    false
}

fn build_rec(
    bits: &[u64],
    n: u32,
    org: [u32; 3],
    size: u32,
    my: usize,
    nodes: &mut [SvoNode],
    cursor: &mut usize,
) -> Option<()> {
    let half = size / 2;
    let mut mask = 0u8;
    if size == 2 {
        for oct in 0..8u8 {
            let v = child_org(org, 1, oct);
            if bit_get(bits, (v[0] + n * (v[1] + n * v[2])) as usize) {
                mask |= 1 << oct;
            }
        }
        nodes[my] = SvoNode {
            child_mask: mask,
            first_child: 0,
        };
        return Some(());
    }
    for oct in 0..8u8 {
        if region_occupied(bits, n, child_org(org, half, oct), half) {
            mask |= 1 << oct;
        }
    }
    let first = *cursor;
    let count = mask.count_ones() as usize;
    if first + count > nodes.len() {
        return None;
    }
    *cursor = first + count;
    nodes[my] = SvoNode {
        child_mask: mask,
        first_child: first as u32,
    };
    let mut ci = first;
    for oct in 0..8u8 {
        if mask & (1 << oct) != 0 {
            build_rec(bits, n, child_org(org, half, oct), half, ci, nodes, cursor)?;
            ci += 1;
        }
    }
    Some(())
}

/// Build an SVO over a `2^levels`-edge cubic bit grid into `nodes`
/// (preorder; root at index 0). Returns the node count, or `None` if
/// the buffer is too small or `levels` is out of `1..=SVO_MAX_LEVELS`.
pub fn svo_build(bits: &[u64], levels: u32, nodes: &mut [SvoNode]) -> Option<usize> {
    if levels == 0 || levels > SVO_MAX_LEVELS || nodes.is_empty() {
        return None;
    }
    let n = 1u32 << levels;
    debug_assert!(bits.len() * 64 >= (n * n * n) as usize);
    let mut cursor = 1usize;
    build_rec(bits, n, [0, 0, 0], n, 0, nodes, &mut cursor)?;
    Some(cursor)
}

// ── SVO: front-to-back traversal ───────────────────────────────────

/// Slab test of the ray against the cube `[org, org + size]³`.
/// Returns `(t_enter, t_exit)` — a hit iff `t_enter ≤ t_exit`.
#[inline(always)]
fn cube_span(origin: [f32; 3], inv: [f32; 3], org: [u32; 3], size: u32) -> (f32, f32) {
    let (mut e, mut x) = (f32::NEG_INFINITY, f32::INFINITY);
    for a in 0..3 {
        let ta = (org[a] as f32 - origin[a]) * inv[a];
        let tb = (org[a] as f32 + size as f32 - origin[a]) * inv[a];
        let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
        e = e.max(lo);
        x = x.min(hi);
    }
    (e, x)
}

/// Trace one ray through an SVO built by [`svo_build`]. Same ray
/// semantics and hit encoding as [`dda_trace`] over the same grid.
pub fn svo_trace(
    nodes: &[SvoNode],
    levels: u32,
    origin: [f32; 3],
    dir: [f32; 3],
    t_max: f32,
) -> (f32, u32) {
    if levels == 0 || levels > SVO_MAX_LEVELS || nodes.is_empty() {
        return MISS;
    }
    let n = 1u32 << levels;
    let d = [sanitize(dir[0]), sanitize(dir[1]), sanitize(dir[2])];
    let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];

    let (re, rx) = cube_span(origin, inv, [0, 0, 0], n);
    if re > rx || rx < 0.0 || re > t_max {
        return MISS;
    }

    // DFS stack: nearest child pushed last (popped first), so subtrees
    // are visited in strict front-to-back order and the first occupied
    // voxel found is the global nearest.
    let mut stack = [(0u32, [0u32; 3], 0u32); (8 * SVO_MAX_LEVELS) as usize];
    let mut top = 0usize;
    stack[top] = (0, [0, 0, 0], n);
    top += 1;

    while top > 0 {
        top -= 1;
        let (idx, org, size) = stack[top];
        let node = nodes[idx as usize];
        if node.child_mask == 0 {
            continue;
        }
        let half = size / 2;

        // Order this node's present children front-to-back.
        let mut order = [(0.0f32, 0u8); 8];
        let mut cnt = 0usize;
        for oct in 0..8u8 {
            if node.child_mask & (1 << oct) == 0 {
                continue;
            }
            let corg = child_org(org, half, oct);
            let (e, x) = cube_span(origin, inv, corg, half);
            if e > x || x < 0.0 || e > t_max {
                continue;
            }
            let key = e.max(0.0);
            let mut j = cnt;
            while j > 0 && order[j - 1].0 > key {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = (key, oct);
            cnt += 1;
        }

        if size == 2 {
            // Children are voxels; nearest valid one is the hit.
            if cnt > 0 {
                let (t, oct) = order[0];
                let v = child_org(org, 1, oct);
                return (t, v[0] + n * (v[1] + n * v[2]));
            }
            continue;
        }

        // Push far-to-near so the nearest pops first.
        for k in (0..cnt).rev() {
            let oct = order[k].1;
            let rank = (node.child_mask & ((1 << oct) - 1)).count_ones();
            let corg = child_org(org, half, oct);
            stack[top] = (node.first_child + rank, corg, half);
            top += 1;
        }
    }
    MISS
}

/// Batch [`svo_trace`], mirroring [`dda_trace_n`].
pub fn svo_trace_n(
    nodes: &[SvoNode],
    levels: u32,
    origins: &[f32],
    dirs: &[f32],
    n: usize,
    t_max: f32,
    out_t: &mut [f32],
    out_idx: &mut [u32],
) {
    debug_assert!(origins.len() >= n * 3 && dirs.len() >= n * 3);
    debug_assert!(out_t.len() >= n && out_idx.len() >= n);
    for i in 0..n {
        let o = [origins[i * 3], origins[i * 3 + 1], origins[i * 3 + 2]];
        let d = [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]];
        let (t, idx) = svo_trace(nodes, levels, o, d, t_max);
        out_t[i] = t;
        out_idx[i] = idx;
    }
}

// ── Brick grid: 4×4×4 voxels = one u64 word ────────────────────────
//
// The `bitmat` idea applied to occupancy: a brick is a 64-bit matrix
// held in a register during traversal. The coarse walk is a DDA over
// bricks — one u64 load per step, and a zero word rejects 64 voxels at
// once — while the fine walk inside a non-empty brick tests bits of
// the already-loaded word: no memory traffic at all. Unlike the SVO
// there is no stack, no sorting and no pointer chasing, which is why
// brickmaps are the practical CPU sweet spot between dense DDA (cost ∝
// distance) and deep octrees (cost ∝ log N per step, pointer-bound).

/// Brick edge in voxels; `BRICK³ = 64` bits = one word.
pub const BRICK: u32 = 4;

/// Pack an x-major bit grid into brick words (bit `vx + 4·(vy + 4·vz)`
/// of brick `bx + bdx·(by + bdy·bz)`). `dims` must be multiples of 4.
/// Nibble-at-a-time: 16 shift/or steps per brick, never bit-by-bit.
pub fn brick_pack(bits: &[u64], dims: [u32; 3], bricks: &mut [u64]) {
    brick_pack_range(bits, dims, bricks, 0, dims[2] / BRICK)
}

/// [`brick_pack`] restricted to brick-z slabs `bz0 .. bz1` — the unit
/// dynamic scenes repack incrementally, and a disjoint split for
/// threading a full repack. `bricks` is *slab-relative*: it receives
/// exactly bricks `[bz0·bdx·bdy, bz1·bdx·bdy)`, so threaded callers
/// hand each slab its own `chunks_mut` of the global brick array (the
/// full-grid [`brick_pack`] passes `bz0 = 0`, making the two views
/// coincide).
pub fn brick_pack_range(bits: &[u64], dims: [u32; 3], bricks: &mut [u64], bz0: u32, bz1: u32) {
    debug_assert!(dims.iter().all(|d| d % BRICK == 0));
    let (nx, ny) = (dims[0], dims[1]);
    let (bdx, bdy, bdz) = (dims[0] / BRICK, dims[1] / BRICK, dims[2] / BRICK);
    debug_assert!(bricks.len() >= ((bz1 - bz0) * bdx * bdy) as usize && bz0 <= bz1 && bz1 <= bdz);
    let out_base = (bz0 * bdx * bdy) as usize;
    for bz in bz0..bz1 {
        for by in 0..bdy {
            for bx in 0..bdx {
                let mut word = 0u64;
                for vz in 0..BRICK {
                    for vy in 0..BRICK {
                        // x-nibbles start 4-aligned (nx % 4 == 0), so a
                        // nibble never straddles a u64.
                        let base =
                            (bx * BRICK + nx * (by * BRICK + vy + ny * (bz * BRICK + vz))) as usize;
                        let nib = (bits[base / 64] >> (base % 64)) & 0xF;
                        word |= nib << (4 * (vy + 4 * vz));
                    }
                }
                bricks[(bx + bdx * (by + bdy * bz)) as usize - out_base] = word;
            }
        }
    }
}

/// Trace one ray through a brick grid built by [`brick_pack`]. Same
/// ray semantics and hit encoding as [`dda_trace`] on the source grid.
pub fn brick_trace(
    bricks: &[u64],
    dims: [u32; 3],
    origin: [f32; 3],
    dir: [f32; 3],
    t_max: f32,
) -> (f32, u32) {
    debug_assert!(dims.iter().all(|d| d % BRICK == 0));
    let d = [sanitize(dir[0]), sanitize(dir[1]), sanitize(dir[2])];

    let (mut t0, mut t1) = (0.0f32, t_max);
    for a in 0..3 {
        let inv = 1.0 / d[a];
        let ta = (0.0 - origin[a]) * inv;
        let tb = (dims[a] as f32 - origin[a]) * inv;
        let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
        t0 = t0.max(lo);
        t1 = t1.min(hi);
    }
    if t0 > t1 {
        return MISS;
    }

    let bdims = [dims[0] / BRICK, dims[1] / BRICK, dims[2] / BRICK];
    let bf = BRICK as f32;

    // Coarse DDA over bricks: boundaries at multiples of BRICK voxels,
    // t stays in voxel-space units.
    let mut cell = [0i32; 3];
    let mut step = [0i32; 3];
    let mut t_next = [0.0f32; 3];
    let mut t_delta = [0.0f32; 3];
    for a in 0..3 {
        let p = origin[a] + t0 * d[a];
        let c = ((p / bf) as i32).clamp(0, bdims[a] as i32 - 1);
        cell[a] = c;
        step[a] = if d[a] > 0.0 { 1 } else { -1 };
        t_delta[a] = bf / d[a].abs();
        let boundary = if d[a] > 0.0 {
            (c + 1) as f32 * bf
        } else {
            c as f32 * bf
        };
        t_next[a] = t0 + (boundary - p) / d[a];
    }

    let (nx, ny) = (dims[0] as i32, dims[1] as i32);
    let (bnx, bny) = (bdims[0] as i32, bdims[1] as i32);
    let mut t = t0;
    loop {
        let word = bricks[(cell[0] + bnx * (cell[1] + bny * cell[2])) as usize];
        if word != 0 {
            // Fine DDA inside the brick against the register word.
            let base = [
                cell[0] * BRICK as i32,
                cell[1] * BRICK as i32,
                cell[2] * BRICK as i32,
            ];
            let mut v = [0i32; 3];
            let mut vt_next = [0.0f32; 3];
            for a in 0..3 {
                let p = origin[a] + t * d[a];
                let l = ((p - base[a] as f32) as i32).clamp(0, BRICK as i32 - 1);
                v[a] = l;
                let boundary = base[a] as f32 + if d[a] > 0.0 { (l + 1) as f32 } else { l as f32 };
                vt_next[a] = t + (boundary - p) / d[a];
            }
            let vt_delta = [1.0 / d[0].abs(), 1.0 / d[1].abs(), 1.0 / d[2].abs()];
            let mut vt = t;
            loop {
                if word >> (v[0] + 4 * (v[1] + 4 * v[2])) & 1 != 0 {
                    let g = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
                    return (vt, (g[0] + nx * (g[1] + ny * g[2])) as u32);
                }
                let a = if vt_next[0] <= vt_next[1] && vt_next[0] <= vt_next[2] {
                    0
                } else if vt_next[1] <= vt_next[2] {
                    1
                } else {
                    2
                };
                vt = vt_next[a];
                if vt > t1 {
                    return MISS;
                }
                v[a] += step[a];
                if v[a] < 0 || v[a] >= BRICK as i32 {
                    break; // left the brick — resume the coarse walk
                }
                vt_next[a] += vt_delta[a];
            }
        }
        // Coarse step to the next brick.
        let a = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
            0
        } else if t_next[1] <= t_next[2] {
            1
        } else {
            2
        };
        t = t_next[a];
        if t > t1 {
            return MISS;
        }
        cell[a] += step[a];
        if cell[a] < 0 || cell[a] >= bdims[a] as i32 {
            return MISS;
        }
        t_next[a] += t_delta[a];
    }
}

/// Batch [`brick_trace`], mirroring [`dda_trace_n`].
pub fn brick_trace_n(
    bricks: &[u64],
    dims: [u32; 3],
    origins: &[f32],
    dirs: &[f32],
    n: usize,
    t_max: f32,
    out_t: &mut [f32],
    out_idx: &mut [u32],
) {
    debug_assert!(origins.len() >= n * 3 && dirs.len() >= n * 3);
    debug_assert!(out_t.len() >= n && out_idx.len() >= n);
    for i in 0..n {
        let o = [origins[i * 3], origins[i * 3 + 1], origins[i * 3 + 2]];
        let d = [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]];
        let (t, idx) = brick_trace(bricks, dims, o, d, t_max);
        out_t[i] = t;
        out_idx[i] = idx;
    }
}

/// Trace four-ray packets through the same brick grid as [`brick_trace_n`].
///
/// Slab setup and coarse/fine traversal use portable SIMD; single occupied
/// lanes retain the scalar fine walk. Both keep the x-before-y-before-z
/// boundary tie rule. Each
/// lane keeps the scalar operation order, including division and repeated
/// boundary addition. This is not a reciprocal approximation or a ray packet
/// that assumes every lane intersects the same voxels.
///
/// Inputs are xyz-interleaved, and only the first `n` output elements are
/// written. Partial packets never read or gather from inactive lanes. No heap
/// storage is allocated. `dims` must be positive multiples of four and their
/// product must fit the scalar tracer's signed linear-index representation;
/// every slice must cover its declared dimensions/count. Invalid slice/grid
/// descriptors panic. Nonfinite origins/directions and negative or NaN
/// `t_max` produce `(f32::INFINITY, u32::MAX)`; positive infinite `t_max`
/// remains supported.
/// Finite rays whose traversal arithmetic overflows use the scalar reference.
pub fn brick_trace_packet_n(
    bricks: &[u64],
    dims: [u32; 3],
    origins: &[f32],
    dirs: &[f32],
    n: usize,
    t_max: f32,
    out_t: &mut [f32],
    out_idx: &mut [u32],
) {
    let coordinates = n.checked_mul(3).expect("voxel packet ray count overflow");
    assert!(origins.len() >= coordinates && dirs.len() >= coordinates);
    assert!(out_t.len() >= n && out_idx.len() >= n);
    if n == 0 {
        return;
    }
    let cells = dims
        .iter()
        .try_fold(1_u32, |count, &extent| count.checked_mul(extent));
    assert!(dims.iter().all(|&extent| extent > 0 && extent % BRICK == 0));
    let cells = cells
        .filter(|&count| count <= i32::MAX as u32)
        .expect("voxel packet grid index overflow");
    assert!(bricks.len() >= cells as usize / 64);
    out_t[..n].fill(MISS.0);
    out_idx[..n].fill(MISS.1);
    if t_max.is_nan() || t_max < 0.0 {
        return;
    }
    for start in (0..n).step_by(4) {
        brick_trace_packet4::<false>(
            bricks,
            dims,
            &origins[start * 3..],
            &dirs[start * 3..],
            (n - start).min(4),
            t_max,
            &mut out_t[start..],
            &mut out_idx[start..],
            None,
        );
    }
}

/// `fine_delta` must be abs(1 / direction) from the same sanitized, finite
/// direction. A normal reciprocal scales exactly by four on the same F32
/// rounding grid as 4 / abs(direction); sanitization rules out overflow.
/// Subnormal reciprocals may already have lost bits, so replay the original
/// division for the whole axis if any lane is subnormal. Use a branch rather
/// than a select so normal packets do not evaluate that division at all.
#[inline]
fn brick_packet_coarse_delta(direction: Simd<f32, 4>, fine_delta: Simd<f32, 4>) -> Simd<f32, 4> {
    let edge = Simd::splat(BRICK as f32);
    if fine_delta.is_normal().all() {
        fine_delta * edge
    } else {
        edge / direction.abs()
    }
}

#[inline]
fn brick_trace_packet4<const USE_HIERARCHY: bool>(
    bricks: &[u64],
    dims: [u32; 3],
    origins: &[f32],
    dirs: &[f32],
    lanes: usize,
    t_max: f32,
    out_t: &mut [f32],
    out_idx: &mut [u32],
    hierarchy: Option<&BrickHierarchy<'_>>,
) {
    type F = Simd<f32, 4>;
    type I = Simd<i32, 4>;
    let mut scalar_origins = [[0.0_f32; 3]; 4];
    let mut scalar_dirs = [[1.0_f32; 3]; 4];
    let mut valid = [false; 4];
    for lane in 0..lanes {
        let offset = lane * 3;
        let origin = [origins[offset], origins[offset + 1], origins[offset + 2]];
        let direction = [dirs[offset], dirs[offset + 1], dirs[offset + 2]];
        if origin
            .iter()
            .chain(&direction)
            .all(|value| value.is_finite())
        {
            scalar_origins[lane] = origin;
            scalar_dirs[lane] = direction.map(sanitize);
            valid[lane] = true;
        }
    }
    let mut active = Mask::<i32, 4>::from_array(valid);
    if !active.any() {
        return;
    }
    let origin: [F; 3] = core::array::from_fn(|axis| {
        F::from_array(core::array::from_fn(|lane| scalar_origins[lane][axis]))
    });
    let direction: [F; 3] = core::array::from_fn(|axis| {
        F::from_array(core::array::from_fn(|lane| scalar_dirs[lane][axis]))
    });
    let zero = F::splat(0.0);
    let one = F::splat(1.0);
    let edge = F::splat(BRICK as f32);
    let mut t0 = zero;
    let mut t1 = F::splat(t_max);
    let mut fine_delta = [zero; 3];
    for axis in 0..3 {
        let inverse = one / direction[axis];
        // For finite sanitized directions, abs(1 / d) is bit-identical to
        // 1 / abs(d), including subnormal results. Keep the correctly rounded
        // division already needed by the slab test for every fine brick walk.
        fine_delta[axis] = inverse.abs();
        let ta = (zero - origin[axis]) * inverse;
        let tb = (F::splat(dims[axis] as f32) - origin[axis]) * inverse;
        let ordered = ta.simd_lt(tb);
        let lo = ordered.select(ta, tb);
        let hi = ordered.select(tb, ta);
        t0 = t0.simd_max(lo);
        t1 = t1.simd_min(hi);
    }
    let overflow = active & !t0.is_finite();
    if overflow.any() {
        for lane in 0..lanes {
            if overflow.test(lane) {
                (out_t[lane], out_idx[lane]) =
                    brick_trace(bricks, dims, scalar_origins[lane], scalar_dirs[lane], t_max);
            }
        }
    }
    active &= t0.simd_le(t1) & !overflow;
    if !active.any() {
        return;
    }
    // Give dead lanes finite setup values. They never participate in a gather
    // or a result, even if their independent state later leaves the grid.
    t0 = active.select(t0, zero);
    let bdims = dims.map(|extent| (extent / BRICK) as i32);
    let mut cell = [I::splat(0); 3];
    let mut step = [I::splat(0); 3];
    let mut t_next = [zero; 3];
    let mut t_delta = [zero; 3];
    for axis in 0..3 {
        let p = origin[axis] + t0 * direction[axis];
        let c = (p / edge)
            .cast::<i32>()
            .simd_max(I::splat(0))
            .simd_min(I::splat(bdims[axis] - 1));
        let positive = direction[axis].simd_gt(zero);
        cell[axis] = c;
        step[axis] = positive.select(I::splat(1), I::splat(-1));
        t_delta[axis] = brick_packet_coarse_delta(direction[axis], fine_delta[axis]);
        let boundary = positive.select(c + I::splat(1), c).cast::<f32>() * edge;
        t_next[axis] = t0 + (boundary - p) / direction[axis];
    }
    let mut t = t0;
    let brick_strides = [1, bdims[0], bdims[0] * bdims[1]];
    'packet: while active.any() {
        let indices = cell[0] + I::splat(bdims[0]) * (cell[1] + I::splat(bdims[1]) * cell[2]);
        // All four lanes must still be active and in the same brick before
        // sharing this checked load. Active cells are in bounds, so equal
        // linear indices also prove equality of all three brick coordinates.
        let shared = (active & indices.simd_eq(I::splat(indices[0]))).all();
        let shared_word = if shared {
            bricks[indices[0] as usize]
        } else {
            0
        };
        if shared && shared_word == 0 {
            // The flat specialization has neither this lookup nor the
            // parent-boundary checks below. A shared empty leaf can skip its
            // 64³ parent only when the borrowed parent word proves it empty.
            if USE_HIERARCHY
                && let Some(hierarchy) = hierarchy
                && let Some(fallback) = hierarchy.skip_packet64(
                    &mut cell,
                    step,
                    &mut t_next,
                    t_delta,
                    &mut t,
                    t1,
                    &mut active,
                )
            {
                for lane in 0..lanes {
                    if fallback.test(lane) {
                        (out_t[lane], out_idx[lane]) = brick_trace(
                            bricks,
                            dims,
                            scalar_origins[lane],
                            scalar_dirs[lane],
                            t_max,
                        );
                    }
                }
                // Each successful skip already performed the exact coarse
                // updates through the parent exit. Do not add another step.
                continue 'packet;
            }
            let mut shared_cell = [cell[0][0], cell[1][0], cell[2][0]];
            let mut shared_index = indices[0];
            // Keep each axis statically indexed, so the common loop need not
            // spill the SIMD state for an indirect array read/write each step.
            macro_rules! advance_shared {
                ($axis:literal) => {{
                    let offset = step[$axis][0];
                    let next = t_next[$axis];
                    if !step[$axis].simd_eq(I::splat(offset)).all()
                        || !(next.is_finite() & next.simd_le(t1)).all()
                    {
                        // The existing generic step handles differing signs,
                        // partial clipping, and the finite-overflow fallback.
                        break;
                    }
                    let coordinate = shared_cell[$axis] + offset;
                    if coordinate < 0 || coordinate >= bdims[$axis] {
                        // Every active ray leaves the grid across this face.
                        return;
                    }
                    let crossed_parent =
                        USE_HIERARCHY && (coordinate ^ shared_cell[$axis]) & !15 != 0;
                    shared_cell[$axis] = coordinate;
                    cell[$axis] = I::splat(coordinate);
                    t = next;
                    // Exactly one original F32 addition per lane and step.
                    t_next[$axis] = next + t_delta[$axis];
                    shared_index += offset * brick_strides[$axis];
                    if crossed_parent {
                        // Long coherent runs otherwise stay inside this
                        // inner loop and never inspect their next parent.
                        continue 'packet;
                    }
                    if bricks[shared_index as usize] != 0 {
                        // Re-enter the fine walk at the new brick.
                        continue 'packet;
                    }
                }};
            }
            loop {
                // This is the same x-before-y-before-z selector as below.
                // Divergent masks leave the empty brick to the generic step;
                // no lane has advanced yet in that case.
                let x = t_next[0].simd_le(t_next[1]) & t_next[0].simd_le(t_next[2]);
                match x.to_bitmask() {
                    0b1111 => advance_shared!(0),
                    0 => match t_next[1].simd_le(t_next[2]).to_bitmask() {
                        0b1111 => advance_shared!(1),
                        0 => advance_shared!(2),
                        _ => break,
                    },
                    _ => break,
                }
            }
        } else {
            let mut words = [shared_word; 4];
            let mut occupied = if shared { 0b1111_u64 } else { 0 };
            if !shared {
                // Checked scalar gathers never form pointers for inactive or
                // escaped lanes. Their words stay zero, including packet tails.
                for lane in 0..lanes {
                    if active.test(lane) {
                        words[lane] = bricks[indices[lane] as usize];
                        if words[lane] != 0 {
                            occupied |= 1 << lane;
                        }
                    }
                }
            }
            // Clearing the lowest set bit leaves a nonzero value exactly
            // when at least two occupied lanes can share SIMD fine setup.
            if occupied & occupied.wrapping_sub(1) != 0 {
                let fine = brick_trace_packet_fine4(
                    words,
                    dims,
                    cell,
                    origin,
                    direction,
                    fine_delta,
                    [t, t1],
                    Mask::from_bitmask(occupied),
                );
                let mut retired = fine.retired.to_bitmask();
                while retired != 0 {
                    let lane = retired.trailing_zeros() as usize;
                    out_t[lane] = fine.distance[lane];
                    out_idx[lane] = fine.id[lane];
                    retired &= retired - 1;
                }
                active &= !fine.retired;
                occupied = fine.fallback.to_bitmask();
            }
            // A single occupied lane avoids SIMD fine setup. Rare overflowing
            // lanes also replay the original fine walk from its entry state.
            while occupied != 0 {
                let lane = occupied.trailing_zeros() as usize;
                occupied &= occupied - 1;
                let brick = [cell[0][lane], cell[1][lane], cell[2][lane]];
                if let Some((distance, id)) = brick_trace_packet_fine(
                    words[lane],
                    dims,
                    brick,
                    scalar_origins[lane],
                    scalar_dirs[lane],
                    [
                        fine_delta[0][lane],
                        fine_delta[1][lane],
                        fine_delta[2][lane],
                    ],
                    [t[lane], t1[lane]],
                ) {
                    out_t[lane] = distance;
                    out_idx[lane] = id;
                    active.set(lane, false);
                }
            }
            if !active.any() {
                break;
            }
        }
        let x = t_next[0].simd_le(t_next[1]) & t_next[0].simd_le(t_next[2]);
        let y = !x & t_next[1].simd_le(t_next[2]);
        let axes = [x, y, !(x | y)];
        let next = x.select(t_next[0], y.select(t_next[1], t_next[2]));
        let overflow = active & !next.is_finite();
        if overflow.any() {
            for lane in 0..lanes {
                if overflow.test(lane) {
                    (out_t[lane], out_idx[lane]) =
                        brick_trace(bricks, dims, scalar_origins[lane], scalar_dirs[lane], t_max);
                }
            }
        }
        active &= next.simd_le(t1) & !overflow;
        t = active.select(next, t);
        for axis in 0..3 {
            let advance = active & axes[axis];
            cell[axis] = advance.select(cell[axis] + step[axis], cell[axis]);
            t_next[axis] = advance.select(t_next[axis] + t_delta[axis], t_next[axis]);
        }
        for axis in 0..3 {
            active &= cell[axis].simd_ge(I::splat(0)) & cell[axis].simd_lt(I::splat(bdims[axis]));
        }
    }
}

/// Results for a masked occupied-brick walk. Lanes that leave their brick
/// without a hit are absent from both masks and resume the coarse traversal.
struct BrickFinePacket {
    retired: Mask<i32, 4>,
    fallback: Mask<i32, 4>,
    distance: Simd<f32, 4>,
    id: Simd<u32, 4>,
}

#[inline]
fn brick_packet_occupied(
    words: [Simd<u32, 4>; 2],
    voxel: [Simd<i32, 4>; 3],
    active: Mask<i32, 4>,
) -> Mask<i32, 4> {
    type U = Simd<u32, 4>;
    let bit = active
        .select(
            voxel[0] + Simd::splat(4) * (voxel[1] + Simd::splat(4) * voxel[2]),
            Simd::splat(0),
        )
        .cast::<u32>();
    // Two four-lane halves avoid widening the packet to four u64 lanes. All
    // shifts stay in 0..32, including inactive/escaped lanes; no memory gather
    // occurs during the fine loop.
    let word = bit.simd_lt(U::splat(32)).select(words[0], words[1]);
    active & ((word >> (bit & U::splat(31))) & U::splat(1)).simd_ne(U::splat(0))
}

/// Masked equivalent of the scalar fine walk below. A lane's position,
/// division, tie selector, and repeated additions retain their scalar order.
/// The caller replays exceptional lanes with the scalar helper, from the same
/// coarse entry state, rather than inventing a different overflow policy.
#[inline]
fn brick_trace_packet_fine4(
    words: [u64; 4],
    dims: [u32; 3],
    brick: [Simd<i32, 4>; 3],
    origin: [Simd<f32, 4>; 3],
    direction: [Simd<f32, 4>; 3],
    delta: [Simd<f32, 4>; 3],
    span: [Simd<f32, 4>; 2],
    occupied: Mask<i32, 4>,
) -> BrickFinePacket {
    type F = Simd<f32, 4>;
    type I = Simd<i32, 4>;
    type U = Simd<u32, 4>;
    let zero = F::splat(0.0);
    let base = brick.map(|coordinate| coordinate * I::splat(BRICK as i32));
    let words = [
        U::from_array(words.map(|word| word as u32)),
        U::from_array(words.map(|word| (word >> 32) as u32)),
    ];
    let mut position = [zero; 3];
    let mut voxel = [I::splat(0); 3];
    let mut fallback = occupied & !span[0].is_finite();
    for axis in 0..3 {
        position[axis] = origin[axis] + span[0] * direction[axis];
        voxel[axis] = (position[axis] - base[axis].cast::<f32>())
            .cast::<i32>()
            .simd_max(I::splat(0))
            .simd_min(I::splat(BRICK as i32 - 1));
        fallback |= occupied & !position[axis].is_finite();
    }
    let mut active = occupied & !fallback;
    // Occupied entry cells require no boundary divisions, and their original
    // entry t is returned without alteration, as in the scalar helper.
    let mut hits = brick_packet_occupied(words, voxel, active);
    let mut retired = hits;
    let mut distance = hits.select(span[0], F::splat(MISS.0));
    active &= !hits;
    if active.any() {
        let mut next = [zero; 3];
        let mut step = [I::splat(0); 3];
        for axis in 0..3 {
            let positive = direction[axis].simd_gt(zero);
            step[axis] = positive.select(I::splat(1), I::splat(-1));
            let boundary = base[axis].cast::<f32>()
                + positive
                    .select(voxel[axis] + I::splat(1), voxel[axis])
                    .cast::<f32>();
            next[axis] = span[0] + (boundary - position[axis]) / direction[axis];
            fallback |= active & !next[axis].is_finite();
        }
        active &= !fallback;
        while active.any() {
            let x = next[0].simd_le(next[1]) & next[0].simd_le(next[2]);
            let y = !x & next[1].simd_le(next[2]);
            let axes = [x, y, !(x | y)];
            let crossing = x.select(next[0], y.select(next[1], next[2]));
            let exceptional = active & !crossing.is_finite();
            fallback |= exceptional;
            active &= !exceptional;
            // The scalar checks clipping before advancing or testing bounds.
            let clipped = active & crossing.simd_gt(span[1]);
            retired |= clipped;
            active &= !clipped;
            for axis in 0..3 {
                let advance = active & axes[axis];
                voxel[axis] = advance.select(voxel[axis] + step[axis], voxel[axis]);
            }
            for coordinate in voxel {
                active &=
                    coordinate.simd_ge(I::splat(0)) & coordinate.simd_lt(I::splat(BRICK as i32));
            }
            for axis in 0..3 {
                let advance = active & axes[axis];
                next[axis] = advance.select(next[axis] + delta[axis], next[axis]);
            }
            let hit = brick_packet_occupied(words, voxel, active);
            distance = hit.select(crossing, distance);
            hits |= hit;
            retired |= hit;
            active &= !hit;
        }
    }
    // Hit lanes no longer advance, so their final local coordinates suffice
    // to form exact integer IDs once. Mask other lanes before global-index
    // arithmetic, including cells that just escaped a brick.
    let global: [I; 3] =
        core::array::from_fn(|axis| hits.select(base[axis] + voxel[axis], I::splat(0)));
    let id = (global[0]
        + I::splat(dims[0] as i32) * (global[1] + I::splat(dims[1] as i32) * global[2]))
        .cast::<u32>();
    BrickFinePacket {
        retired,
        fallback,
        distance,
        id: hits.select(id, U::splat(MISS.1)),
    }
}

/// The fine walk is intentionally the same arithmetic as `brick_trace`.
/// `None` resumes the packet's coarse walk; `Some(MISS)` retires a clipped ray.
#[inline]
fn brick_trace_packet_fine(
    word: u64,
    dims: [u32; 3],
    brick: [i32; 3],
    origin: [f32; 3],
    direction: [f32; 3],
    delta: [f32; 3],
    span: [f32; 2],
) -> Option<(f32, u32)> {
    let [t, t1] = span;
    let base = brick.map(|coordinate| coordinate * BRICK as i32);
    let mut position = [0.0_f32; 3];
    let mut v = [0_i32; 3];
    for axis in 0..3 {
        let p = origin[axis] + t * direction[axis];
        position[axis] = p;
        v[axis] = ((p - base[axis] as f32) as i32).clamp(0, BRICK as i32 - 1);
    }
    // An occupied entry voxel needs no fine boundary setup. Its bit and
    // distance are already known, so return before doing those divisions.
    if word >> (v[0] + 4 * (v[1] + 4 * v[2])) & 1 != 0 {
        let g = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
        return Some((
            t,
            (g[0] + dims[0] as i32 * (g[1] + dims[1] as i32 * g[2])) as u32,
        ));
    }
    let mut next = [0.0_f32; 3];
    let mut step = [0_i32; 3];
    for axis in 0..3 {
        step[axis] = if direction[axis] > 0.0 { 1 } else { -1 };
        let boundary = base[axis] as f32
            + if direction[axis] > 0.0 {
                (v[axis] + 1) as f32
            } else {
                v[axis] as f32
            };
        next[axis] = t + (boundary - position[axis]) / direction[axis];
    }
    loop {
        let axis = if next[0] <= next[1] && next[0] <= next[2] {
            0
        } else if next[1] <= next[2] {
            1
        } else {
            2
        };
        let distance = next[axis];
        if distance > t1 {
            return Some(MISS);
        }
        v[axis] += step[axis];
        if v[axis] < 0 || v[axis] >= BRICK as i32 {
            return None;
        }
        next[axis] += delta[axis];
        if word >> (v[0] + 4 * (v[1] + 4 * v[2])) & 1 != 0 {
            let g = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
            return Some((
                distance,
                (g[0] + dims[0] as i32 * (g[1] + dims[1] as i32 * g[2])) as u32,
            ));
        }
    }
}

// ── Density marching: mixed-precision volume integration ───────────
//
// The mixed-precision voxel payload: a density grid marched by the
// same DDA, accumulating optical depth `τ = Σ σ·Δt` per crossed voxel.
// No per-step exp — the caller applies `T = exp(-τ)` once — and the
// march stops early when `τ ≥ tau_max` (opacity cutoff), returning the
// exact crossing `t` by linear interpolation inside the final voxel.
//
// Storing σ as E4M3/E5M2/F16 quarters/halves the volume footprint at
// identical traversal cost per step: at 256³ that is 16/32 MB versus
// 64 MB f32 — the difference between a volume that lives in L2 and one
// that streams from DRAM, which is where the quantized variants win
// wall-clock, not just memory. The scalar decode per step (the same
// `cast` decoders the f8 kernels tail into) hides under the grid
// fetch. A per-volume scale belongs on `tau_max`'s side: store
// normalized σ and fold the scale into `tau_max` and the final `T`.

macro_rules! march_density_kernel {
    ($name:ident, $ty:ty, $get:expr) => {
        /// March the ray, integrating `τ += σ·Δt`; returns `(τ, t_end)`
        /// where `t_end` is the opacity crossing (τ hits `tau_max`), or
        /// the grid/`t_max` exit with the final τ.
        ///
        /// `density` is x-major and must cover `dims[0] * dims[1] * dims[2]`
        /// voxels. The suffix defines its storage: one `f32`, two packed
        /// little-endian F16 bytes, or one signed/unsigned FP8 byte per voxel.
        /// An undersized payload panics if the ray reaches the missing voxel.
        pub fn $name(
            density: &[$ty],
            dims: [u32; 3],
            origin: [f32; 3],
            dir: [f32; 3],
            t_max: f32,
            tau_max: f32,
        ) -> (f32, f32) {
            if dims.contains(&0) {
                return (0.0, 0.0);
            }
            let d = [sanitize(dir[0]), sanitize(dir[1]), sanitize(dir[2])];
            let (mut t0, mut t1) = (0.0f32, t_max);
            for a in 0..3 {
                let inv = 1.0 / d[a];
                let ta = (0.0 - origin[a]) * inv;
                let tb = (dims[a] as f32 - origin[a]) * inv;
                let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
                t0 = t0.max(lo);
                t1 = t1.min(hi);
            }
            if t0 > t1 {
                return (0.0, 0.0);
            }
            if tau_max <= 0.0 {
                return (tau_max, t0);
            }

            let mut cell = [0i32; 3];
            let mut step = [0i32; 3];
            let mut t_next = [0.0f32; 3];
            let mut t_delta = [0.0f32; 3];
            for a in 0..3 {
                let p = origin[a] + t0 * d[a];
                let c = (p as i32).clamp(0, dims[a] as i32 - 1);
                cell[a] = c;
                step[a] = if d[a] > 0.0 { 1 } else { -1 };
                t_delta[a] = 1.0 / d[a].abs();
                let boundary = if d[a] > 0.0 { c as f32 + 1.0 } else { c as f32 };
                t_next[a] = t0 + (boundary - p) / d[a];
            }

            let (nx, ny) = (dims[0] as i32, dims[1] as i32);
            let strides = [1i32, nx, nx * ny];
            let mut idx = cell[0] + nx * (cell[1] + ny * cell[2]);
            let mut tau = 0.0f32;
            let mut t = t0;
            loop {
                let a = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
                    0
                } else if t_next[1] <= t_next[2] {
                    1
                } else {
                    2
                };
                let t_exit = t_next[a].min(t1);
                let sigma = $get(density, idx as usize);
                let new_tau = sigma * (t_exit - t) + tau;
                if new_tau >= tau_max {
                    // Exact opacity depth inside this voxel.
                    return (tau_max, t + (tau_max - tau) / sigma);
                }
                tau = new_tau;
                t = t_next[a];
                if t > t1 {
                    return (tau, t1);
                }
                cell[a] += step[a];
                if cell[a] < 0 || cell[a] >= dims[a] as i32 {
                    return (tau, t_exit);
                }
                idx += step[a] * strides[a];
                t_next[a] += t_delta[a];
            }
        }
    };
}

march_density_kernel!(march_density_f32, f32, |d: &[f32], i: usize| d[i]);
march_density_kernel!(march_density_f16, u8, |d: &[u8], i: usize| {
    crate::cast::read_f16(&d[i * 2..])
});
march_density_kernel!(march_density_e4m3, u8, |d: &[u8], i: usize| {
    crate::cast::e4m3_to_f32(d[i])
});
march_density_kernel!(march_density_e5m2, u8, |d: &[u8], i: usize| {
    crate::cast::e5m2_to_f32(d[i])
});
march_density_kernel!(march_density_uf8e4m4, u8, |d: &[u8], i: usize| {
    crate::cast::uf8e4m4_to_f32(d[i])
});
march_density_kernel!(march_density_uf8e5m3, u8, |d: &[u8], i: usize| {
    crate::cast::uf8e5m3_to_f32(d[i])
});

// ── Emissive volume transfer ───────────────────────────────────────

/// Front-to-back solution of the scalar emission/absorption equation
/// through a voxel volume. The caller composites a background as
/// `radiance + transmittance * background`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissionTransfer {
    pub radiance: [f32; 3],
    pub transmittance: f32,
    /// Grid-space parameter at the opacity cutoff or volume exit.
    pub t_end: f32,
}

impl EmissionTransfer {
    const EMPTY: Self = Self {
        radiance: [0.0; 3],
        transmittance: 1.0,
        t_end: 0.0,
    };
}

// Each voxel is piecewise constant. For extinction `sigma`, source
// radiance `S`, and segment length `ds`, the exact unpolarized transfer is
//
//   I_out = I_in exp(-sigma ds) + S (1 - exp(-sigma ds)).
//
// Front-to-back accumulation therefore needs only the three radiance
// accumulators plus one transmittance value. `expm1` preserves precision for
// optically thin cells, and early termination bounds opaque-ray work.
macro_rules! march_emission_kernel {
    ($name:ident, $ty:ty, $get_extinction:expr, $get_source:expr) => {
        /// March a non-negative emission/absorption volume front-to-back.
        ///
        /// `extinction` is x-major with one encoded value per voxel and
        /// `source_rgb` has three interleaved values per voxel. Both payloads
        /// must cover the product of `dims`; undersized inputs panic. Values
        /// must use the format named by this function's suffix.
        pub fn $name(
            extinction: &[$ty],
            source_rgb: &[$ty],
            dims: [u32; 3],
            origin: [f32; 3],
            dir: [f32; 3],
            t_max: f32,
            extinction_scale: f32,
            emission_scale: f32,
            transmittance_cutoff: f32,
        ) -> EmissionTransfer {
            let voxel_count = dims[0] as usize * dims[1] as usize * dims[2] as usize;
            debug_assert!(extinction.len() >= voxel_count);
            debug_assert!(source_rgb.len() >= voxel_count * 3);
            debug_assert!(t_max.is_finite() && t_max >= 0.0);
            debug_assert!(extinction_scale.is_finite() && extinction_scale >= 0.0);
            debug_assert!(emission_scale.is_finite() && emission_scale >= 0.0);
            debug_assert!((0.0..=1.0).contains(&transmittance_cutoff));
            if voxel_count == 0 || t_max <= 0.0 {
                return EmissionTransfer::EMPTY;
            }

            let d = [sanitize(dir[0]), sanitize(dir[1]), sanitize(dir[2])];
            let (mut t0, mut t1) = (0.0f32, t_max);
            for axis in 0..3 {
                let inverse = 1.0 / d[axis];
                let ta = -origin[axis] * inverse;
                let tb = (dims[axis] as f32 - origin[axis]) * inverse;
                let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
                t0 = t0.max(lo);
                t1 = t1.min(hi);
            }
            if t0 > t1 {
                return EmissionTransfer::EMPTY;
            }

            let mut cell = [0i32; 3];
            let mut step = [0i32; 3];
            let mut t_next = [0.0f32; 3];
            let mut t_delta = [0.0f32; 3];
            for axis in 0..3 {
                let position = origin[axis] + t0 * d[axis];
                let coordinate = (position as i32).clamp(0, dims[axis] as i32 - 1);
                cell[axis] = coordinate;
                step[axis] = if d[axis] > 0.0 { 1 } else { -1 };
                t_delta[axis] = 1.0 / d[axis].abs();
                let boundary = if d[axis] > 0.0 {
                    coordinate as f32 + 1.0
                } else {
                    coordinate as f32
                };
                t_next[axis] = t0 + (boundary - position) / d[axis];
            }

            let (nx, ny) = (dims[0] as i32, dims[1] as i32);
            let strides = [1i32, nx, nx * ny];
            let mut index = cell[0] + nx * (cell[1] + ny * cell[2]);
            let mut transfer = EmissionTransfer {
                t_end: t0,
                ..EmissionTransfer::EMPTY
            };
            let mut t = t0;
            loop {
                let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
                    0
                } else if t_next[1] <= t_next[2] {
                    1
                } else {
                    2
                };
                let t_exit = t_next[axis].min(t1);
                let segment = (t_exit - t).max(0.0);
                let extinction_value = $get_extinction(extinction, index as usize).max(0.0);
                if extinction_value > 0.0 && segment > 0.0 {
                    let optical_depth = extinction_value * extinction_scale * segment;
                    let absorbed = -axmath::expm1_f32(-optical_depth);
                    let transmittance_before = transfer.transmittance;
                    let transmittance_after = transmittance_before * (1.0 - absorbed);
                    let reached_cutoff = transmittance_after <= transmittance_cutoff;
                    let applied_absorbed = if reached_cutoff {
                        1.0 - transmittance_cutoff / transmittance_before
                    } else {
                        absorbed
                    };
                    let weight = transmittance_before * applied_absorbed;
                    for channel in 0..3 {
                        let source = $get_source(source_rgb, index as usize, channel).max(0.0);
                        transfer.radiance[channel] += weight * source * emission_scale;
                    }
                    transfer.transmittance = if reached_cutoff {
                        transmittance_cutoff
                    } else {
                        transmittance_after
                    };
                    if reached_cutoff {
                        let crossing_depth =
                            axmath::log_f32(transmittance_before / transmittance_cutoff);
                        transfer.t_end = if crossing_depth.is_finite() {
                            (t + crossing_depth / (extinction_value * extinction_scale))
                                .clamp(t, t_exit)
                        } else {
                            t_exit
                        };
                        return transfer;
                    }
                }
                transfer.t_end = t_exit;
                t = t_next[axis];
                if t > t1 {
                    return transfer;
                }
                cell[axis] += step[axis];
                if cell[axis] < 0 || cell[axis] >= dims[axis] as i32 {
                    return transfer;
                }
                index += step[axis] * strides[axis];
                t_next[axis] += t_delta[axis];
            }
        }
    };
}

march_emission_kernel!(
    march_emission_f32,
    f32,
    |values: &[f32], index: usize| values[index],
    |values: &[f32], index: usize, channel: usize| values[index * 3 + channel]
);
march_emission_kernel!(
    march_emission_e4m3,
    u8,
    |values: &[u8], index: usize| crate::cast::e4m3_to_f32(values[index]),
    |values: &[u8], index: usize, channel: usize| crate::cast::e4m3_to_f32(
        values[index * 3 + channel]
    )
);
march_emission_kernel!(
    march_emission_uf8e4m4,
    u8,
    |values: &[u8], index: usize| crate::cast::uf8e4m4_to_f32(values[index]),
    |values: &[u8], index: usize, channel: usize| crate::cast::uf8e4m4_to_f32(
        values[index * 3 + channel]
    )
);
march_emission_kernel!(
    march_emission_uf8e5m3,
    u8,
    |values: &[u8], index: usize| crate::cast::uf8e5m3_to_f32(values[index]),
    |values: &[u8], index: usize, channel: usize| crate::cast::uf8e5m3_to_f32(
        values[index * 3 + channel]
    )
);

/// Shared trilinear sampler for one-byte floating-point voxel fields.
///
/// Integer voxel `i` is centered at `i + 0.5`, matching the cell convention
/// used by the DDA marchers above. Coordinates within half a voxel of the
/// volume boundary clamp to the nearest center; coordinates outside the
/// volume return zero. Keeping interpolation here lets renderers reuse the
/// mixed-precision storage contract without expanding the field to FP32.
#[inline]
fn sample_trilinear_u8(
    values: &[u8],
    dims: [u32; 3],
    point: [f32; 3],
    decode: impl Fn(u8) -> f32,
) -> f32 {
    let voxel_count = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    debug_assert!(values.len() >= voxel_count);
    if voxel_count == 0
        || point.iter().zip(dims).any(|(coordinate, extent)| {
            !coordinate.is_finite() || *coordinate < 0.0 || *coordinate > extent as f32
        })
    {
        return 0.0;
    }

    let mut low = [0_u32; 3];
    let mut high = [0_u32; 3];
    let mut fraction = [0.0_f32; 3];
    for axis in 0..3 {
        let centered = (point[axis] - 0.5).clamp(0.0, dims[axis] as f32 - 1.0);
        low[axis] = axmath::floor_f32(centered) as u32;
        high[axis] = (low[axis] + 1).min(dims[axis] - 1);
        fraction[axis] = centered - low[axis] as f32;
    }

    let index = |x: u32, y: u32, z: u32| (x + dims[0] * (y + dims[1] * z)) as usize;
    let lerp = |a: f32, b: f32, amount: f32| a + amount * (b - a);
    let sample = |x: u32, y: u32, z: u32| decode(values[index(x, y, z)]);
    let x00 = lerp(
        sample(low[0], low[1], low[2]),
        sample(high[0], low[1], low[2]),
        fraction[0],
    );
    let x10 = lerp(
        sample(low[0], high[1], low[2]),
        sample(high[0], high[1], low[2]),
        fraction[0],
    );
    let x01 = lerp(
        sample(low[0], low[1], high[2]),
        sample(high[0], low[1], high[2]),
        fraction[0],
    );
    let x11 = lerp(
        sample(low[0], high[1], high[2]),
        sample(high[0], high[1], high[2]),
        fraction[0],
    );
    lerp(
        lerp(x00, x10, fraction[1]),
        lerp(x01, x11, fraction[1]),
        fraction[2],
    )
}

pub fn sample_trilinear_e4m3(values: &[u8], dims: [u32; 3], point: [f32; 3]) -> f32 {
    sample_trilinear_u8(values, dims, point, crate::cast::e4m3_to_f32)
}

/// Trilinearly sample a physically non-negative unsigned E4M4 field.
///
/// `values` is an x-major one-byte-per-voxel payload covering the product of
/// `dims`. Empty dimensions and points outside the volume return zero;
/// undersized non-empty payloads panic.
pub fn sample_trilinear_uf8e4m4(values: &[u8], dims: [u32; 3], point: [f32; 3]) -> f32 {
    sample_trilinear_u8(values, dims, point, crate::cast::uf8e4m4_to_f32)
}

/// Trilinearly sample a physically non-negative unsigned E5M3 field.
///
/// `values` is an x-major one-byte-per-voxel payload covering the product of
/// `dims`. Empty dimensions and points outside the volume return zero;
/// undersized non-empty payloads panic.
pub fn sample_trilinear_uf8e5m3(values: &[u8], dims: [u32; 3], point: [f32; 3]) -> f32 {
    sample_trilinear_u8(values, dims, point, crate::cast::uf8e5m3_to_f32)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    fn grid(n: u32) -> Vec<u64> {
        vec![0u64; ((n * n * n) as usize).div_ceil(64)]
    }

    fn set(bits: &mut [u64], n: u32, x: u32, y: u32, z: u32) {
        let idx = (x + n * (y + n * z)) as usize;
        bits[idx / 64] |= 1 << (idx % 64);
    }

    /// Sphere shell centered mid-grid — smooth, sparse, non-convex
    /// from inside.
    fn shell(n: u32) -> Vec<u64> {
        let mut bits = grid(n);
        let c = n as f32 * 0.5;
        let (r0, r1) = (n as f32 * 0.30, n as f32 * 0.38);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy, dz) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c, z as f32 + 0.5 - c);
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r >= r0 && r <= r1 {
                        set(&mut bits, n, x, y, z);
                    }
                }
            }
        }
        bits
    }

    #[test]
    fn dda_axis_corridor() {
        let n = 8;
        let mut bits = grid(n);
        // Wall at z = 5.
        for y in 0..n {
            for x in 0..n {
                set(&mut bits, n, x, y, 5);
            }
        }
        let (t, idx) = dda_trace(&bits, [n; 3], [4.5, 4.5, 0.5], [0.0, 0.0, 1.0], 100.0);
        assert_eq!(idx, 4 + n * (4 + n * 5));
        assert!((t - 4.5).abs() < 1e-5, "t = {t}");

        // Away from the wall: miss.
        let (t, idx) = dda_trace(&bits, [n; 3], [4.5, 4.5, 0.5], [0.0, 0.0, -1.0], 100.0);
        assert_eq!((t, idx), MISS);
        // t_max short of the wall: miss.
        let (t, _) = dda_trace(&bits, [n; 3], [4.5, 4.5, 0.5], [0.0, 0.0, 1.0], 4.0);
        assert!(t.is_infinite());
    }

    #[test]
    fn dda_inside_occupied_start() {
        let n = 8;
        let mut bits = grid(n);
        set(&mut bits, n, 3, 3, 3);
        let (t, idx) = dda_trace(&bits, [n; 3], [3.5, 3.5, 3.5], [1.0, 0.3, -0.2], 100.0);
        assert_eq!(idx, 3 + n * (3 + n * 3));
        assert_eq!(t, 0.0);
    }

    #[test]
    fn svo_build_shapes() {
        let n = 8u32; // levels = 3
        let mut bits = grid(n);
        set(&mut bits, n, 0, 0, 0);
        set(&mut bits, n, 7, 7, 7);
        let mut nodes = [SvoNode::default(); 64];
        let count = svo_build(&bits, 3, &mut nodes).unwrap();
        // Two opposite corners: root + one chain per corner, 2 levels each.
        assert_eq!(count, 5);
        assert_eq!(nodes[0].child_mask, 0b1000_0001);
        // Too-small buffer refuses.
        assert!(svo_build(&bits, 3, &mut nodes[..2]).is_none());
    }

    #[test]
    fn svo_matches_dda_on_shell() {
        let n = 32u32;
        let levels = 5;
        let bits = shell(n);
        let mut nodes = vec![SvoNode::default(); 200_000];
        let count = svo_build(&bits, levels, &mut nodes).unwrap();
        assert!(count > 8);

        let mut hits = 0usize;
        let mut rays = 0usize;
        for iy in 0..12 {
            for ix in 0..12 {
                for &(ox, oy, oz, tilt) in &[
                    (-2.0f32, 0.0f32, 0.0f32, 0.06f32), // side, tilted
                    (16.0, 16.0, -4.0, 0.02),           // below, near-axial
                    (16.0, 16.0, 16.0, 0.9),            // inside the shell
                ] {
                    let dir = [
                        1.0 + tilt * ix as f32 * if ox < 0.0 { 1.0 } else { 0.0 }
                            + if ox < 0.0 { 0.0 } else { tilt * ix as f32 },
                        0.1 + 0.17 * iy as f32 * tilt,
                        0.23 + 0.11 * (ix + iy) as f32 * tilt,
                    ];
                    let o = [ox, oy + 0.37 * iy as f32, oz + 0.29 * ix as f32];
                    let (td, id) = dda_trace(&bits, [n; 3], o, dir, 500.0);
                    let (ts, is_) = svo_trace(&nodes[..count], levels, o, dir, 500.0);
                    rays += 1;
                    if id != u32::MAX {
                        hits += 1;
                    }
                    assert_eq!(id, is_, "voxel mismatch at ray {rays} (o={o:?} d={dir:?})");
                    if id != u32::MAX {
                        assert!((td - ts).abs() < 1e-3, "t mismatch: {td} vs {ts}");
                    }
                }
            }
        }
        assert!(hits > 50, "scene should be hit often (got {hits}/{rays})");
    }

    #[test]
    fn svo_respects_t_max() {
        let n = 8;
        let mut bits = grid(n);
        for y in 0..n {
            for x in 0..n {
                set(&mut bits, n, x, y, 5);
            }
        }
        let mut nodes = [SvoNode::default(); 128];
        let count = svo_build(&bits, 3, &mut nodes).unwrap();
        let nodes = &nodes[..count];
        let (t, idx) = svo_trace(nodes, 3, [4.5, 4.5, 0.5], [0.0, 0.0, 1.0], 100.0);
        assert_eq!(idx, 4 + n * (4 + n * 5));
        assert!((t - 4.5).abs() < 1e-5);
        let (t, _) = svo_trace(nodes, 3, [4.5, 4.5, 0.5], [0.0, 0.0, 1.0], 4.0);
        assert!(t.is_infinite());
    }

    #[test]
    fn brick_matches_dda_on_shell() {
        let n = 32u32;
        let bits = shell(n);
        let mut bricks = vec![0u64; ((n / BRICK).pow(3)) as usize];
        brick_pack(&bits, [n; 3], &mut bricks);

        let mut hits = 0usize;
        for iy in 0..16 {
            for ix in 0..16 {
                for &(ox, oy, oz) in &[
                    (-2.0f32, 0.0f32, 0.0f32),
                    (16.0, 16.0, -4.0),
                    (16.0, 16.0, 16.0), // inside the hollow center
                ] {
                    let o = [ox, oy + 0.41 * iy as f32, oz + 0.23 * ix as f32];
                    let dir = [
                        1.0 + 0.05 * ix as f32,
                        0.1 + 0.13 * iy as f32,
                        0.27 + 0.09 * (ix + iy) as f32,
                    ];
                    let (td, id) = dda_trace(&bits, [n; 3], o, dir, 500.0);
                    let (tb, ib) = brick_trace(&bricks, [n; 3], o, dir, 500.0);
                    assert_eq!(id, ib, "voxel mismatch (o={o:?} d={dir:?})");
                    if id != u32::MAX {
                        hits += 1;
                        assert!((td - tb).abs() < 1e-3, "t mismatch: {td} vs {tb}");
                    }
                }
            }
        }
        assert!(hits > 80, "scene should be hit often (got {hits})");
    }

    #[test]
    fn brick_pack_bit_layout() {
        let n = 8u32;
        let mut bits = grid(n);
        set(&mut bits, n, 5, 6, 7); // brick (1,1,1), local (1,2,3)
        let mut bricks = vec![0u64; 8];
        brick_pack(&bits, [n; 3], &mut bricks);
        let word = bricks[1 + 2 * (1 + 2 * 1)];
        assert_eq!(word, 1u64 << (1 + 4 * (2 + 4 * 3)));
        assert_eq!(bricks.iter().filter(|&&w| w != 0).count(), 1);
    }

    fn assert_brick_packets_match(
        bricks: &[u64],
        dims: [u32; 3],
        rays: &[([f32; 3], [f32; 3])],
        t_max: f32,
    ) {
        let origins: Vec<f32> = rays.iter().flat_map(|ray| ray.0).collect();
        let directions: Vec<f32> = rays.iter().flat_map(|ray| ray.1).collect();
        let mut expected_t = vec![-123.0_f32; rays.len() + 3];
        let mut expected_id = vec![123_u32; rays.len() + 3];
        let mut actual_t = expected_t.clone();
        let mut actual_id = expected_id.clone();
        brick_trace_n(
            bricks,
            dims,
            &origins,
            &directions,
            rays.len(),
            t_max,
            &mut expected_t,
            &mut expected_id,
        );
        brick_trace_packet_n(
            bricks,
            dims,
            &origins,
            &directions,
            rays.len(),
            t_max,
            &mut actual_t,
            &mut actual_id,
        );
        for index in 0..rays.len() {
            assert_eq!(
                actual_id[index], expected_id[index],
                "packet hit ID at ray {index}, {:?}, clip {t_max}",
                rays[index]
            );
            assert_eq!(
                actual_t[index].to_bits(),
                expected_t[index].to_bits(),
                "packet distance at ray {index}, {:?}, clip {t_max}",
                rays[index]
            );
        }
        assert_eq!(&actual_t[rays.len()..], &[-123.0; 3]);
        assert_eq!(&actual_id[rays.len()..], &[123; 3]);
    }

    fn packet_random(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    #[derive(Clone, Copy, Debug)]
    struct FineLane {
        word: u64,
        brick: [i32; 3],
        origin: [f32; 3],
        direction: [f32; 3],
        span: [f32; 2],
    }

    fn assert_brick_fine_packets_match(dims: [u32; 3], lanes: [FineLane; 4], mask: u64) -> u64 {
        let direction = lanes.map(|lane| lane.direction.map(sanitize));
        let delta = direction.map(|direction| direction.map(|d| 1.0 / d.abs()));
        let actual = brick_trace_packet_fine4(
            lanes.map(|lane| lane.word),
            dims,
            core::array::from_fn(|axis| Simd::from_array(lanes.map(|lane| lane.brick[axis]))),
            core::array::from_fn(|axis| Simd::from_array(lanes.map(|lane| lane.origin[axis]))),
            core::array::from_fn(|axis| Simd::from_array(direction.map(|dir| dir[axis]))),
            core::array::from_fn(|axis| Simd::from_array(delta.map(|d| d[axis]))),
            core::array::from_fn(|end| Simd::from_array(lanes.map(|lane| lane.span[end]))),
            Mask::from_bitmask(mask),
        );
        assert_eq!((actual.retired & actual.fallback).to_bitmask(), 0);
        assert_eq!((actual.retired | actual.fallback).to_bitmask() & !mask, 0);
        for (index, lane) in lanes.into_iter().enumerate() {
            if mask & (1 << index) == 0 || actual.fallback.test(index) {
                assert_eq!(actual.distance[index].to_bits(), MISS.0.to_bits());
                assert_eq!(actual.id[index], MISS.1);
                continue;
            }
            let expected = brick_trace_packet_fine(
                lane.word,
                dims,
                lane.brick,
                lane.origin,
                direction[index],
                delta[index],
                lane.span,
            );
            let result = actual
                .retired
                .test(index)
                .then_some((actual.distance[index].to_bits(), actual.id[index]));
            assert_eq!(
                result,
                expected.map(|(distance, id)| (distance.to_bits(), id)),
                "fine lane {index}, mask {mask:04b}: {lane:?}",
            );
        }
        actual.fallback.to_bitmask()
    }

    #[test]
    fn brick_fine_packets_preserve_mixed_outcomes_and_masks() {
        let lanes = [
            FineLane {
                word: 1,
                brick: [0; 3],
                origin: [0.25; 3],
                direction: [1.0, 0.0, -0.0],
                span: [0.0, f32::INFINITY],
            },
            FineLane {
                word: 1 << 3,
                brick: [1; 3],
                origin: [4.25; 3],
                direction: [1.0, 0.0, 0.0],
                span: [0.0, f32::INFINITY],
            },
            FineLane {
                word: 1 << 63,
                brick: [4, 2, 6],
                origin: [16.25, 8.25, 24.25],
                direction: [1.0, 0.0, 0.0],
                span: [0.0, f32::INFINITY],
            },
            FineLane {
                word: 1 << 3,
                brick: [2, 0, 1],
                origin: [8.25, 0.25, 4.25],
                direction: [1.0, 0.0, 0.0],
                span: [0.0, 0.5],
            },
        ];
        for mask in 0..16 {
            // Inactive slots may contain arbitrary state, even though the
            // public packet entry point initializes its tails benignly.
            let masked = core::array::from_fn(|index| {
                if mask & (1 << index) != 0 {
                    lanes[index]
                } else {
                    FineLane {
                        word: u64::MAX,
                        brick: [i32::MIN; 3],
                        origin: [f32::NAN; 3],
                        direction: [f32::INFINITY; 3],
                        span: [f32::INFINITY; 2],
                    }
                }
            });
            assert_eq!(
                assert_brick_fine_packets_match([20, 12, 28], masked, mask),
                0
            );
        }
        // One occupied word contains entry/later-hit cells for three rows,
        // and no hit for the fourth. Shared storage cannot merge lane states.
        let word = (1_u64 << 33) | (1_u64 << 39) | (1_u64 << 40);
        let coherent = core::array::from_fn(|index| FineLane {
            word,
            brick: [1; 3],
            origin: [4.25, 4.25 + index as f32, 6.25],
            direction: [1.0, 0.0, -0.0],
            span: [0.0, f32::INFINITY],
        });
        for mask in 0..16 {
            assert_eq!(assert_brick_fine_packets_match([8; 3], coherent, mask), 0);
        }
    }

    #[test]
    fn brick_fine_packets_preserve_ties_entry_bits_and_large_ids() {
        for sign in [-1.0_f32, 1.0] {
            for limit in [f32::from_bits(0.5_f32.to_bits() - 1), 0.5, f32::INFINITY] {
                let start = if sign > 0.0 { 1 } else { 2 };
                let lanes = core::array::from_fn(|lane| {
                    let mut target = [start; 3];
                    for coordinate in &mut target[..=lane.min(2)] {
                        *coordinate += sign as i32;
                    }
                    // Intermediate cells have the same crossing time.
                    // Advancing tied axes together would skip the first two.
                    FineLane {
                        word: 1_u64 << (target[0] + 4 * (target[1] + 4 * target[2])),
                        brick: [1; 3],
                        origin: [4.5 + start as f32; 3],
                        direction: [sign * if lane == 3 { 2.0 } else { 1.0 }; 3],
                        span: [0.0, limit],
                    }
                });
                for mask in 0..16 {
                    assert_eq!(assert_brick_fine_packets_match([8; 3], lanes, mask), 0);
                }
            }
        }
        for bit in 0..64 {
            let local = [bit % 4, bit / 4 % 4, bit / 16];
            let lanes = core::array::from_fn(|index| {
                let brick = [64 + index as i32, 65, 66];
                FineLane {
                    word: 1_u64 << bit,
                    brick,
                    origin: core::array::from_fn(|axis| {
                        (brick[axis] * 4 + local[axis]) as f32 + 0.5
                    }),
                    direction: [
                        [1.0, 0.0, -0.0],
                        [-1.0, -0.0, 0.0],
                        [0.25, 2.0, -0.5],
                        [-0.0; 3],
                    ][index],
                    span: [if index % 2 == 0 { -0.0 } else { 0.0 }, f32::INFINITY],
                }
            });
            // IDs exceed 2^24, including odd values not representable in F32.
            // This fine-only check needs no allocated grid storage.
            for mask in 0..16 {
                assert_eq!(assert_brick_fine_packets_match([512; 3], lanes, mask), 0);
            }
        }
    }

    #[test]
    fn brick_fine_packets_match_random_finite_entries_and_edits() {
        let dims = [20, 12, 28];
        let mut state = 0x471a_03c5;
        for case in 0..2048 {
            let mut lanes = core::array::from_fn(|index| {
                let brick = core::array::from_fn(|axis| {
                    (packet_random(&mut state) % (dims[axis] / 4)) as i32
                });
                let mut direction = core::array::from_fn(|_| {
                    (packet_random(&mut state) & 0xffff) as f32 / 16384.0 - 2.0
                });
                if case % 5 == 0 {
                    direction[case % 3] = if index % 2 == 0 { -0.0 } else { 0.0 };
                }
                let entry = (packet_random(&mut state) & 0xff) as f32 / 64.0;
                let position: [f32; 3] = core::array::from_fn(|axis| {
                    (brick[axis] * 4) as f32 + (packet_random(&mut state) & 0x3ff) as f32 / 256.0
                });
                let word = match case % 4 {
                    0 => 1_u64 << (packet_random(&mut state) % 64),
                    1 => {
                        u64::from(packet_random(&mut state)) << 32
                            | u64::from(packet_random(&mut state))
                    }
                    2 => 0,
                    _ => u64::MAX,
                };
                FineLane {
                    word,
                    brick,
                    origin: core::array::from_fn(|axis| position[axis] - entry * direction[axis]),
                    direction,
                    span: [
                        entry,
                        if case % 7 == 0 {
                            f32::INFINITY
                        } else {
                            entry + (packet_random(&mut state) & 0xff) as f32 / 64.0
                        },
                    ],
                }
            });
            // Decouple lane masks from the four occupancy distributions.
            let mask = ((case / 4) % 16) as u64;
            assert_eq!(assert_brick_fine_packets_match(dims, lanes, mask), 0);
            for lane in &mut lanes {
                lane.word ^= 1_u64 << (packet_random(&mut state) % 64);
            }
            assert_eq!(assert_brick_fine_packets_match(dims, lanes, mask), 0);
        }
    }

    #[test]
    fn brick_fine_packets_separate_exceptional_and_stalled_lanes() {
        let lanes = [
            FineLane {
                word: u64::MAX,
                brick: [0; 3],
                origin: [f32::MAX; 3],
                direction: [f32::MAX; 3],
                span: [2.0, f32::INFINITY],
            },
            FineLane {
                word: 1_u64 << 63,
                brick: [1; 3],
                origin: [-f32::MAX; 3],
                direction: [1e-12; 3],
                span: [0.0, f32::INFINITY],
            },
            FineLane {
                word: 1_u64 << 3,
                brick: [0; 3],
                origin: [-f32::MAX; 3],
                direction: [f32::MAX; 3],
                span: [1.0, f32::INFINITY],
            },
            FineLane {
                word: 1_u64 << 16,
                brick: [0; 3],
                origin: [0.25; 3],
                direction: [0.0, -0.0, 1.0],
                span: [0.0, f32::INFINITY],
            },
        ];
        for mask in 0..16 {
            // Lane 0 overflows its entry position; lane 1 its boundary setup.
            // Lane 2 has subnormal deltas that stall at t=1, yet still follows
            // scalar integer-cell progress and must hit, not be discarded.
            assert_eq!(
                assert_brick_fine_packets_match([8; 3], lanes, mask),
                mask & 3
            );
        }
    }

    #[test]
    fn brick_packets_replay_exceptional_fine_lane_beside_hits_and_resume() {
        let dims = [8; 3];
        let huge = f32::from_bits(0x7f72_f911);
        let entry = f32::MAX * (1.0 / huge);
        // The reciprocal is subnormal: the slab entry stays finite, while
        // recomputing its position overflows. This reaches fine fallback,
        // not the packet's earlier nonfinite-slab fallback.
        assert_eq!(entry.to_bits(), 0x3f86_dcdf);
        assert!((-f32::MAX + entry * huge).is_infinite());
        let word = 1_u64 << 63;
        let lanes = [
            FineLane {
                word,
                brick: [1; 3],
                origin: [-f32::MAX; 3],
                direction: [huge; 3],
                span: [entry, entry],
            },
            FineLane {
                word,
                brick: [1; 3],
                origin: [7.25; 3],
                direction: [1.0, 0.0, 0.0],
                span: [0.0, 0.75],
            },
            FineLane {
                word,
                brick: [1; 3],
                origin: [4.25, 7.25, 7.25],
                direction: [1.0, 0.0, 0.0],
                span: [0.0, 3.75],
            },
            FineLane {
                word,
                brick: [1; 3],
                origin: [4.25; 3],
                direction: [-1.0, 0.0, 0.0],
                span: [0.0, 4.25],
            },
        ];
        // These are the actual first occupied-brick states of the rays
        // below. All four share word 7, so the integrated path enters fine4.
        // Only lane 0 falls back; lanes 1/2 hit and lane 3 resumes coarse DDA.
        assert_eq!(assert_brick_fine_packets_match(dims, lanes, 0b1111), 0b0001);
        let mut bricks = [0_u64; 8];
        bricks[7] = word;
        bricks[6] = 1 << 3;
        let rays = lanes.map(|lane| (lane.origin, lane.direction));
        let expected = [(entry, 511), (0.0, 511), (2.75, 511), (0.25, 291)];
        for (index, ((origin, direction), (distance, id))) in
            rays.into_iter().zip(expected).enumerate()
        {
            let actual = brick_trace(&bricks, dims, origin, direction, f32::INFINITY);
            assert_eq!(
                actual.0.to_bits(),
                distance.to_bits(),
                "scalar lane {index}"
            );
            assert_eq!(actual.1, id, "scalar lane {index}");
        }
        // Check the real caller's scalar replay, output scatter, and resumed
        // lane against those scalar results, including exact distance bits.
        assert_brick_packets_match(&bricks, dims, &rays, f32::INFINITY);
    }

    #[test]
    fn brick_packets_match_coherent_and_random_rays_after_edits() {
        let dims = [20_u32, 12, 28];
        let count = (dims[0] * dims[1] * dims[2] / 64) as usize;
        let mut state = 0x5a17_c3e9;
        let mut sparse = vec![0_u64; count];
        for word in &mut sparse {
            if packet_random(&mut state) % 5 == 0 {
                *word = u64::from(packet_random(&mut state)) << 32
                    | u64::from(packet_random(&mut state));
            }
        }
        let mut rays = Vec::new();
        for y in 0..9 {
            for x in 0..29 {
                rays.push((
                    [-5.0, 6.0, 14.0],
                    [1.0, (y as f32 - 4.0) * 0.0625, (x as f32 - 14.0) * 0.0625],
                ));
            }
        }
        for _ in 0..513 {
            let origin = core::array::from_fn(|axis| {
                (packet_random(&mut state) & 0xffff) as f32 / 65536.0 * (dims[axis] as f32 + 16.0)
                    - 8.0
            });
            let direction = core::array::from_fn(|_| {
                (packet_random(&mut state) & 0xffff) as f32 / 16384.0 - 2.0
            });
            rays.push((origin, direction));
        }
        for bricks in [vec![0; count], vec![u64::MAX; count], sparse] {
            for clip in [0.0, 0.125, 4.0, 100.0, f32::INFINITY] {
                assert_brick_packets_match(&bricks, dims, &rays, clip);
            }
            let mut edited = bricks;
            for (index, word) in edited.iter_mut().enumerate() {
                if index % 3 == 0 {
                    *word = 0;
                } else {
                    *word ^= 1_u64 << (packet_random(&mut state) % 64);
                }
            }
            assert_brick_packets_match(&edited, dims, &rays, 100.0);
        }
    }

    #[test]
    fn brick_packets_preserve_shared_empty_steps_and_partial_activity() {
        let dims = [32, 8, 8];
        let mut bricks = [0_u64; 32];
        let near = 3 + 8 * (1 + 2);
        let far = 6 + 8 * (1 + 2);
        let near_word = (1_u64 << 16) | (1_u64 << 24);
        bricks[near] = near_word;
        bricks[far] = u64::MAX;
        // Four lanes share several empty bricks. At x=12, lanes 0 and 2
        // retire on different occupied entry bits; the remaining lanes must
        // keep walking to x=24 without gathering the retired lanes again.
        let coherent = [4.25, 5.25, 6.25, 7.25].map(|y| ([-4.0, y, 5.5], [1.0, 0.0, 0.0]));
        for (lane, &(origin, direction)) in coherent.iter().enumerate() {
            let (distance, id) = brick_trace(&bricks, dims, origin, direction, f32::INFINITY);
            assert_eq!(distance, if lane % 2 == 0 { 16.0 } else { 28.0 });
            assert_ne!(id, u32::MAX);
        }
        let mut rays = coherent.to_vec();
        // These lanes begin together, then cross different brick faces or
        // leave the grid. The final three rays exercise an incomplete packet.
        rays.extend_from_slice(&[
            ([-4.0, 5.5, 5.5], [1.0, 0.0, 0.0]),
            ([-4.0, 5.5, 5.5], [1.0, 0.0625, 0.0]),
            ([-4.0, 5.5, 5.5], [1.0, -0.125, 0.0]),
            ([-4.0, 5.5, 5.5], [1.0, 0.0, 0.25]),
        ]);
        rays.extend_from_slice(&coherent[..3]);
        for (near_word, far_word) in [
            (near_word, u64::MAX),
            (0, u64::MAX),
            (0, 0),
            (1_u64 << 28, u64::MAX),
        ] {
            // Destruction and a newly occupied voxel must be visible on the
            // next call; the shared empty-brick result is never cached.
            bricks[near] = near_word;
            bricks[far] = far_word;
            for count in 0..=rays.len() {
                for clip in [0.0, 15.999, 16.0, 27.999, 28.0, f32::INFINITY] {
                    assert_brick_packets_match(&bricks, dims, &rays[..count], clip);
                }
            }
        }
    }

    #[test]
    fn brick_packets_preserve_common_axis_transitions_in_both_directions() {
        let dims = [32; 3];
        let index = |cell: [usize; 3]| cell[0] + 8 * (cell[1] + 8 * cell[2]);
        for axis in 0..3 {
            let side = (axis + 1) % 3;
            for sign in [-1.0_f32, 1.0] {
                let mut near = [1; 3];
                let mut far = [1; 3];
                near[axis] = if sign > 0.0 { 3 } else { 4 };
                far[axis] = if sign > 0.0 { 6 } else { 1 };
                let mut local = [1; 3];
                local[axis] = if sign > 0.0 { 0 } else { 3 };
                let mut near_word = 0;
                for lane in [0, 2] {
                    local[side] = lane;
                    near_word |= 1_u64 << (local[0] + 4 * (local[1] + 4 * local[2]));
                }
                let mut bricks = [0_u64; 512];
                bricks[index(near)] = near_word;
                bricks[index(far)] = u64::MAX;
                let mut rays = Vec::new();
                for lane in 0..4 {
                    let mut origin = [5.5; 3];
                    origin[axis] = if sign > 0.0 { -4.0 } else { 36.0 };
                    origin[side] = 4.25 + lane as f32;
                    let mut direction = [0.0; 3];
                    direction[axis] = sign;
                    rays.push((origin, direction));
                    let (distance, id) =
                        brick_trace(&bricks, dims, origin, direction, f32::INFINITY);
                    assert_eq!(distance, if lane % 2 == 0 { 16.0 } else { 28.0 });
                    assert_ne!(id, u32::MAX);
                }
                for lane in 0..4 {
                    let (origin, mut direction) = rays[lane];
                    // Shared cells/axis, but different times: a clip can retire
                    // only some lanes immediately after a common empty step.
                    direction[axis] *= 1.0 + lane as f32 * 0.25;
                    rays.push((origin, direction));
                }
                for slope in [0.0, 0.0625, -0.125, 0.25] {
                    let mut origin = [5.5; 3];
                    origin[axis] = if sign > 0.0 { -4.0 } else { 36.0 };
                    let mut direction = [0.0; 3];
                    direction[axis] = sign;
                    direction[side] = slope;
                    // These start together and later choose different axes.
                    rays.push((origin, direction));
                }
                for direction_sign in [1.0, -1.0, 1.0, -1.0] {
                    let mut origin = [5.5; 3];
                    origin[axis] = ((near[axis] + far[axis]) / 2 * 4) as f32 + 1.5;
                    let mut direction = [0.0; 3];
                    direction[axis] = direction_sign;
                    // Shared empty brick and selected axis are insufficient
                    // when the signs differ: the scalar cells must split.
                    rays.push((origin, direction));
                }
                for (near_word, far_word) in [
                    (near_word, u64::MAX),
                    (0, u64::MAX),
                    (0, 0),
                    (u64::MAX, u64::MAX),
                ] {
                    bricks[index(near)] = near_word;
                    bricks[index(far)] = far_word;
                    for clip in [0.0, 6.0, 13.0, 15.999, 16.0, 27.999, 28.0, f32::INFINITY] {
                        for count in 0..=rays.len() {
                            assert_brick_packets_match(&bricks, dims, &rays[..count], clip);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn brick_packets_preserve_common_axis_boundary_ties() {
        for sign in [-1.0, 1.0] {
            let mut bricks = [0_u64; 512];
            // The first tied crossing must choose x. Choosing y or z first
            // would bypass this occupied brick entirely.
            let x = if sign > 0.0 { 2 } else { 0 };
            bricks[x + 8 * (1 + 8)] = u64::MAX;
            let ray = ([5.5; 3], [sign; 3]);
            assert_ne!(
                brick_trace(&bricks, [32; 3], ray.0, ray.1, f32::INFINITY).1,
                u32::MAX
            );
            assert_brick_packets_match(&bricks, [32; 3], &[ray; 4], f32::INFINITY);
        }
    }

    #[test]
    fn brick_packets_hierarchy_crosses_long_shared_runs_and_partial_parent_tails() {
        let dims = [196, 4, 4];
        let mut bricks = [0_u64; 49];
        let mut rays = Vec::new();
        for sign in [1.0_f32, -1.0] {
            for scaled in [false, true] {
                for lane in 0..4 {
                    rays.push((
                        [
                            if sign > 0.0 { -4.0 } else { 200.0 },
                            lane as f32 + 0.25,
                            1.5,
                        ],
                        [
                            sign * if scaled {
                                1.0 + lane as f32 * 0.25
                            } else {
                                1.0
                            },
                            -0.0,
                            0.0,
                        ],
                    ));
                }
            }
        }
        rays.extend_from_slice(&[
            ([64.0, 1.5, 1.5], [1.0, 0.0, 0.0]),
            ([128.0, 1.5, 1.5], [-1.0, -0.0, 0.0]),
            ([192.0, 1.5, 1.5], [1.0, 0.0, -0.0]),
        ]);
        let origins: Vec<_> = rays.iter().flat_map(|ray| ray.0).collect();
        let directions: Vec<_> = rays.iter().flat_map(|ray| ray.1).collect();
        for (first, middle, last) in [
            (1_u64 << 63, 0, (1_u64 << 16) | (1_u64 << 24)),
            (0, 0, u64::MAX),
            (1_u64 << 63, u64::MAX, u64::MAX),
            (0, 0, 0),
        ] {
            bricks[0] = first;
            bricks[16] = middle;
            bricks[48] = last;
            // The occupied first word can make parent zero nonempty without
            // intersecting any ray. Its fine walk resumes a long shared loop
            // that crosses into the empty middle64 parents. Rebuild after
            // edits so adding occupancy cannot leave a cached empty parent.
            let mut level16 = [0; 13];
            let mut level64 = [0; 4];
            for (leaf, &word) in bricks.iter().enumerate() {
                if word != 0 {
                    level16[leaf / 4] |= 1_u64 << (leaf % 4);
                    level64[leaf / 16] |= 1_u64 << (leaf / 4 % 4);
                }
            }
            let hierarchy = BrickHierarchy::new(&bricks, dims, &level16, &level64);
            for count in 0..=rays.len() {
                for limit in [0.0, 63.999, 64.0, 68.0, 128.0, 192.0, 196.0, f32::INFINITY] {
                    let mut flat_t = [-123.0_f32; 22];
                    let mut flat_id = [123; 22];
                    let mut actual_t = flat_t;
                    let mut actual_id = flat_id;
                    brick_trace_packet_n(
                        &bricks,
                        dims,
                        &origins,
                        &directions,
                        count,
                        limit,
                        &mut flat_t,
                        &mut flat_id,
                    );
                    hierarchy.trace_packet_n(
                        &origins,
                        &directions,
                        count,
                        limit,
                        &mut actual_t,
                        &mut actual_id,
                    );
                    assert_eq!(actual_t.map(f32::to_bits), flat_t.map(f32::to_bits));
                    assert_eq!(actual_id, flat_id);
                    for (lane, &(origin, direction)) in rays[..count].iter().enumerate() {
                        let expected = brick_trace(&bricks, dims, origin, direction, limit);
                        assert_eq!(actual_t[lane].to_bits(), expected.0.to_bits());
                        assert_eq!(actual_id[lane], expected.1);
                    }
                    assert!(actual_t[count..].iter().all(|&value| value == -123.0));
                    assert!(actual_id[count..].iter().all(|&value| value == 123));
                }
            }
        }
    }

    #[test]
    fn brick_packets_preserve_boundaries_signed_zero_clipping_and_tails() {
        let dims = [8_u32; 3];
        let origins = [
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [8.0, 8.0, 8.0],
            [-0.0, 4.0, 0.0],
            [7.5, 7.5, 7.5],
            [-4.0, 4.0, 4.0],
            [12.0, 4.0, 4.0],
            [4.0, -4.0, 4.0],
            [4.0, 4.0, -4.0],
        ];
        let directions = [
            [0.0, 0.0, 1.0],
            [-0.0, 0.0, -1.0],
            [0.0; 3],
            [-0.0; 3],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 0.0],
            [0.5, 0.0, 0.0],
            [5e-13, -5e-13, 1.0],
            [-1e-12, 1e-12, -0.0],
        ];
        let rays: Vec<_> = origins
            .iter()
            .flat_map(|&origin| directions.iter().map(move |&direction| (origin, direction)))
            .collect();
        for bricks in [[0_u64; 8], [u64::MAX; 8], [0x8100_0081_8000_0001; 8]] {
            for clip in [-1.0, -0.0, 0.0, 0.25, 4.0, 100.0, f32::INFINITY] {
                assert_brick_packets_match(&bricks, dims, &rays, clip);
            }
            for count in 0..=17 {
                assert_brick_packets_match(&bricks, dims, &rays[..count], 100.0);
            }
        }
    }

    #[test]
    fn brick_packets_preserve_immediate_entry_hits() {
        let dims = [8; 3];
        let mut bricks = [0_u64; 8];
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, -0.0, 0.0],
            [0.0; 3],
            [-0.0; 3],
            [0.25, 2.0, -0.5],
        ];
        for bit in 0..64 {
            bricks[7] = 1_u64 << bit;
            let cell = [4 + bit % 4, 4 + bit / 4 % 4, 4 + bit / 16];
            let id = cell[0] + 8 * (cell[1] + 8 * cell[2]);
            let lower = cell.map(|v| v as f32);
            let center = lower.map(|v| v + 0.5);
            let rays: Vec<_> = [lower, center]
                .into_iter()
                .flat_map(|origin| directions.iter().map(move |&direction| (origin, direction)))
                .collect();
            for &(origin, direction) in &rays {
                assert_eq!(
                    brick_trace(&bricks, dims, origin, direction, 0.0),
                    (0.0, id)
                );
            }
            assert_brick_packets_match(&bricks, dims, &rays, 0.0);
            assert_brick_packets_match(&bricks, dims, &rays, f32::INFINITY);
        }

        // Entry from outside the grid and after empty coarse bricks, using
        // non-unit directions. The positive rays hit at t=4; negative at t=2.
        bricks[7] = u64::MAX;
        let rays = [
            ([-4.0, 5.5, 5.5], [2.0, 0.0, 0.0]),
            ([12.0, 5.5, 5.5], [-2.0, -0.0, 0.0]),
            ([5.5, -4.0, 5.5], [0.0, 2.0, 0.0]),
            ([5.5, 12.0, 5.5], [0.0, -2.0, -0.0]),
            ([5.5, 5.5, -4.0], [0.0, 0.0, 2.0]),
            ([5.5, 5.5, 12.0], [-0.0, 0.0, -2.0]),
        ];
        for clip in [2.0, 3.999, 4.0, f32::INFINITY] {
            assert_brick_packets_match(&bricks, dims, &rays, clip);
        }
    }

    #[test]
    fn brick_packets_coarse_delta_reuses_only_normal_reciprocals() {
        type F = Simd<f32, 4>;
        let boundary = 0x7e80_0000_u32; // 2^126; its reciprocal is MIN_POSITIVE.
        let packets = [
            (
                [
                    f32::from_bits(boundary - 2),
                    f32::from_bits(boundary - 1),
                    f32::from_bits(boundary),
                    3.0,
                ],
                0b1111_u64,
            ),
            (
                [
                    f32::from_bits(boundary + 1),
                    f32::from_bits(boundary + 2),
                    f32::MAX,
                    f32::from_bits(f32::MAX.to_bits() - 1),
                ],
                0,
            ),
            (
                [
                    f32::from_bits(boundary - 1),
                    f32::from_bits(boundary + 1),
                    f32::from_bits(boundary),
                    f32::MAX,
                ],
                0b0101,
            ),
            ([0.125, 1.75, 1e-12, 1e12], 0b1111),
        ];
        for sign in [-1.0_f32, 1.0] {
            for &(magnitudes, normal_mask) in &packets {
                let direction = F::from_array(magnitudes.map(|value| sign * value));
                let fine = (F::splat(1.0) / direction).abs();
                assert_eq!(fine.is_normal().to_bitmask(), normal_mask);
                let actual = brick_packet_coarse_delta(direction, fine);
                let expected = F::splat(4.0) / direction.abs();
                assert_eq!(
                    actual.to_array().map(f32::to_bits),
                    expected.to_array().map(f32::to_bits),
                    "coarse delta for {direction:?}"
                );
            }
            // Reusing a subnormal reciprocal is observably wrong, despite
            // the scaled result being normal. This must take the division.
            let direction = F::splat(sign * f32::MAX);
            let fine = (F::splat(1.0) / direction).abs();
            assert_eq!((fine * F::splat(4.0))[0].to_bits(), 0x0080_0000);
            assert_eq!(
                brick_packet_coarse_delta(direction, fine)[0].to_bits(),
                0x0080_0001
            );
        }

        // Exercise all three setup axes, both signs and several coarse
        // additions. The far leaf also traverses empty 64-voxel parents in
        // the hybrid path, with a partial final parent at voxel 128.
        for axis in 0..3 {
            let mut dims = [4_u32; 3];
            dims[axis] = 132;
            for destination in [0_usize, 16, 32] {
                let mut bricks = [0_u64; 33];
                bricks[destination] = u64::MAX;
                let mut level16 = [0_u64; 9];
                let mut level64 = [0_u64; 3];
                level16[destination / 4] = 1_u64 << ((destination % 4) << (axis * 2));
                level64[destination / 16] = 1_u64 << ((destination / 4 % 4) << (axis * 2));
                let hierarchy = BrickHierarchy::new(&bricks, dims, &level16, &level64);
                for sign in [-1.0_f32, 1.0] {
                    for &(magnitudes, _) in &packets {
                        let rays: [_; 4] = core::array::from_fn(|lane| {
                            let mut origin = [1.5_f32; 3];
                            origin[axis] = if sign > 0.0 { -4.0 } else { 136.0 };
                            let mut direction = [-0.0_f32; 3];
                            direction[axis] = sign * magnitudes[lane];
                            (origin, direction)
                        });
                        let origins: Vec<_> = rays.iter().flat_map(|ray| ray.0).collect();
                        let directions: Vec<_> = rays.iter().flat_map(|ray| ray.1).collect();
                        for limit in [0.0, 1000.0, f32::INFINITY] {
                            assert_brick_packets_match(&bricks, dims, &rays, limit);
                            let mut actual_t = [-123.0_f32; 7];
                            let mut actual_id = [123_u32; 7];
                            hierarchy.trace_packet_n(
                                &origins,
                                &directions,
                                rays.len(),
                                limit,
                                &mut actual_t,
                                &mut actual_id,
                            );
                            for (lane, &(origin, direction)) in rays.iter().enumerate() {
                                let expected = brick_trace(&bricks, dims, origin, direction, limit);
                                assert_eq!(actual_t[lane].to_bits(), expected.0.to_bits());
                                assert_eq!(actual_id[lane], expected.1);
                            }
                            assert_eq!(actual_t[4..], [-123.0; 3]);
                            assert_eq!(actual_id[4..], [123; 3]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn brick_packets_preserve_scalar_finite_overflow_results() {
        let origins = [
            [-f32::MAX; 3],
            [f32::MAX; 3],
            [-f32::MAX, 2.0, 2.0],
            [f32::MAX, 2.0, 2.0],
            [-1e30, -1e30, -1e30],
            [2.0; 3],
        ];
        let directions = [
            [1e-12; 3],
            [-1e-12; 3],
            [f32::MAX; 3],
            [-f32::MAX; 3],
            [0.0, 0.0, 1.0],
            [1.0; 3],
        ];
        let mut rays: Vec<_> = origins
            .iter()
            .flat_map(|&origin| directions.iter().map(move |&direction| (origin, direction)))
            .collect();
        let mut state = 0x9753_1ace;
        for _ in 0..513 {
            let mut coordinate = || {
                let value = f32::from_bits(packet_random(&mut state));
                if value.is_finite() { value } else { 0.0 }
            };
            rays.push((
                core::array::from_fn(|_| coordinate()),
                core::array::from_fn(|_| coordinate()),
            ));
        }
        for bricks in [[0_u64; 8], [u64::MAX; 8], [0x8100_0081_8000_0001; 8]] {
            for clip in [0.0, 100.0, f32::MAX, f32::INFINITY] {
                assert_brick_packets_match(&bricks, [8; 3], &rays, clip);
            }
        }
    }

    #[test]
    fn brick_packets_retire_nonfinite_lanes_without_gathering_them() {
        let origins = [
            f32::NAN,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            -1.0,
            2.5,
            2.5,
            2.5,
            f32::NEG_INFINITY,
            2.0,
            2.0,
        ];
        let directions = [
            0.0,
            0.0,
            1.0,
            f32::INFINITY,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            -0.0,
            0.0,
            -0.0,
            1.0,
            0.0,
            0.0,
        ];
        let mut distances = [-123.0; 7];
        let mut ids = [123; 7];
        brick_trace_packet_n(
            &[u64::MAX; 8],
            [8; 3],
            &origins,
            &directions,
            5,
            100.0,
            &mut distances,
            &mut ids,
        );
        assert_eq!(ids, [u32::MAX, u32::MAX, 18, 146, u32::MAX, 123, 123]);
        assert_eq!(
            distances,
            [
                f32::INFINITY,
                f32::INFINITY,
                1.0,
                0.0,
                f32::INFINITY,
                -123.0,
                -123.0
            ]
        );
        for clip in [f32::NAN, f32::NEG_INFINITY] {
            brick_trace_packet_n(
                &[u64::MAX; 8],
                [8; 3],
                &origins,
                &directions,
                5,
                clip,
                &mut distances,
                &mut ids,
            );
            assert_eq!(&ids[..5], &[u32::MAX; 5]);
            assert!(distances[..5].iter().all(|value| *value == f32::INFINITY));
        }
        brick_trace_packet_n(&[], [0; 3], &[], &[], 0, 0.0, &mut [], &mut []);
    }

    #[test]
    #[should_panic]
    fn brick_packets_reject_short_ray_buffers_before_reading() {
        brick_trace_packet_n(
            &[0; 8],
            [8; 3],
            &[0.0; 2],
            &[1.0; 3],
            1,
            10.0,
            &mut [0.0],
            &mut [0],
        );
    }

    #[test]
    fn march_uniform_corridor() {
        // 8³ of σ = 1.0 everywhere: τ is exactly the in-grid path length.
        let n = 8u32;
        let density = vec![1.0f32; (n * n * n) as usize];
        let (tau, t_end) = march_density_f32(
            &density,
            [n; 3],
            [4.5, 4.5, 0.5],
            [0.0, 0.0, 1.0],
            100.0,
            f32::MAX,
        );
        assert!((tau - 7.5).abs() < 1e-4, "tau = {tau}");
        assert!((t_end - 7.5).abs() < 1e-4, "t_end = {t_end}");

        // Opacity cutoff at τ = 3 lands exactly 3 units in.
        let (tau, t_end) = march_density_f32(
            &density,
            [n; 3],
            [4.5, 4.5, 0.5],
            [0.0, 0.0, 1.0],
            100.0,
            3.0,
        );
        assert_eq!(tau, 3.0);
        assert!((t_end - 3.0).abs() < 1e-4, "t_end = {t_end}");
    }

    #[test]
    fn march_variants_match_f32_on_decoded() {
        let n = 16u32;
        let count = (n * n * n) as usize;
        // Densities born quantized; the f32 kernel gets the decoded
        // values — identical stepping ⇒ bit-identical results.
        let e4: Vec<u8> = (0..count).map(|i| (i % 60) as u8).collect();
        let e5: Vec<u8> = (0..count).map(|i| ((i * 7) % 60) as u8).collect();
        let uf4: Vec<u8> = (0..count)
            .map(|i| crate::cast::f32_to_uf8e4m4_rne((i % 40) as f32 * 0.025))
            .collect();
        let uf5: Vec<u8> = (0..count)
            .map(|i| crate::cast::f32_to_uf8e5m3_rne((i % 40) as f32 * 0.025))
            .collect();
        let f16: Vec<u8> = (0..count)
            .flat_map(|i| crate::cast::f32_to_f16((i % 40) as f32 * 0.01).to_le_bytes())
            .collect();
        let e4_f32: Vec<f32> = e4.iter().map(|&b| crate::cast::e4m3_to_f32(b)).collect();
        let e5_f32: Vec<f32> = e5.iter().map(|&b| crate::cast::e5m2_to_f32(b)).collect();
        let uf4_f32: Vec<f32> = uf4
            .iter()
            .map(|&b| crate::cast::uf8e4m4_to_f32(b))
            .collect();
        let uf5_f32: Vec<f32> = uf5
            .iter()
            .map(|&b| crate::cast::uf8e5m3_to_f32(b))
            .collect();
        let f16_f32: Vec<f32> = (0..count)
            .map(|i| crate::cast::read_f16(&f16[i * 2..]))
            .collect();

        for ray in 0..24 {
            let o = [-1.0, 0.3 + 0.6 * ray as f32, 0.9 + 0.4 * ray as f32];
            let d = [1.0, 0.07 * (ray % 5) as f32, 0.05 * (ray % 3) as f32];
            let tm = 3.0 + 0.4 * ray as f32; // exercise the cutoff path too
            assert_eq!(
                march_density_e4m3(&e4, [n; 3], o, d, 200.0, tm),
                march_density_f32(&e4_f32, [n; 3], o, d, 200.0, tm),
                "e4m3 ray {ray}"
            );
            assert_eq!(
                march_density_e5m2(&e5, [n; 3], o, d, 200.0, tm),
                march_density_f32(&e5_f32, [n; 3], o, d, 200.0, tm),
                "e5m2 ray {ray}"
            );
            assert_eq!(
                march_density_f16(&f16, [n; 3], o, d, 200.0, tm),
                march_density_f32(&f16_f32, [n; 3], o, d, 200.0, tm),
                "f16 ray {ray}"
            );
            assert_eq!(
                march_density_uf8e4m4(&uf4, [n; 3], o, d, 200.0, tm),
                march_density_f32(&uf4_f32, [n; 3], o, d, 200.0, tm),
                "UF8 E4M4 ray {ray}"
            );
            assert_eq!(
                march_density_uf8e5m3(&uf5, [n; 3], o, d, 200.0, tm),
                march_density_f32(&uf5_f32, [n; 3], o, d, 200.0, tm),
                "UF8 E5M3 ray {ray}"
            );
        }
    }

    #[test]
    fn unsigned_fraction_bit_reduces_whole_ray_error() {
        let n = 16u32;
        let count = (n * n * n) as usize;
        let density: Vec<f32> = (0..count)
            .map(|i| 0.003 + 0.847 * ((i * 73 % 997) as f32 / 997.0))
            .collect();
        let signed: Vec<u8> = density
            .iter()
            .map(|&v| crate::fp8::f32_to_e4m3_rne(v))
            .collect();
        let unsigned: Vec<u8> = density
            .iter()
            .map(|&v| crate::cast::f32_to_uf8e4m4_rne(v))
            .collect();

        let (mut signed_error, mut unsigned_error) = (0.0f64, 0.0f64);
        let (mut signed_max, mut unsigned_max) = (0.0f64, 0.0f64);
        for ray in 0..96 {
            let origin = [
                -1.0,
                0.21 + 0.157 * (ray % 91) as f32,
                0.37 + 0.131 * (ray * 7 % 89) as f32,
            ];
            let direction = [1.0, 0.013 * (ray % 7) as f32, 0.011 * (ray % 5) as f32];
            // Alternate full traversal with cutoff crossings, so the gate
            // covers both integrated optical depth and the cutoff position.
            let cutoff = if ray & 1 == 0 {
                f32::MAX
            } else {
                2.1 + 0.07 * (ray % 9) as f32
            };
            let reference = march_density_f32(&density, [n; 3], origin, direction, 100.0, cutoff);
            let signed_result =
                march_density_e4m3(&signed, [n; 3], origin, direction, 100.0, cutoff);
            let unsigned_result =
                march_density_uf8e4m4(&unsigned, [n; 3], origin, direction, 100.0, cutoff);
            let signed_tau = f64::from(signed_result.0 - reference.0);
            let signed_end = f64::from(signed_result.1 - reference.1);
            let unsigned_tau = f64::from(unsigned_result.0 - reference.0);
            let unsigned_end = f64::from(unsigned_result.1 - reference.1);
            signed_error += signed_tau * signed_tau + signed_end * signed_end;
            unsigned_error += unsigned_tau * unsigned_tau + unsigned_end * unsigned_end;
            signed_max = signed_max.max(signed_tau.abs()).max(signed_end.abs());
            unsigned_max = unsigned_max.max(unsigned_tau.abs()).max(unsigned_end.abs());
        }
        assert!(
            unsigned_error < signed_error * 0.35,
            "whole-ray squared error: UF8={unsigned_error}, E4M3={signed_error}",
        );
        assert!(
            unsigned_max < signed_max,
            "whole-ray maximum error: UF8={unsigned_max}, E4M3={signed_max}",
        );
    }

    #[test]
    fn emissive_march_matches_homogeneous_slab_solution() {
        let dims = [1, 1, 4];
        let extinction = [0.5; 4];
        let source = [0.25, 0.5, 1.0].repeat(4);
        let transfer = march_emission_f32(
            &extinction,
            &source,
            dims,
            [0.5, 0.5, -1.0],
            [0.0, 0.0, 1.0],
            10.0,
            0.75,
            2.0,
            0.0,
        );
        let expected_transmittance = axmath::exp_f32(-1.5);
        assert!((transfer.transmittance - expected_transmittance).abs() < 2.0e-6);
        for (actual, source) in transfer.radiance.into_iter().zip([0.25, 0.5, 1.0]) {
            let expected = source * 2.0 * (1.0 - expected_transmittance);
            assert!((actual - expected).abs() < 2.0e-6, "{actual} vs {expected}");
        }
        assert!((transfer.t_end - 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn emissive_e4m3_matches_f32_on_decoded_payload() {
        let dims = [4, 3, 2];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let extinction: Vec<u8> = (0..count)
            .map(|index| crate::fp8::f32_to_e4m3_rne((index % 7) as f32 * 0.125))
            .collect();
        let source: Vec<u8> = (0..count * 3)
            .map(|index| crate::fp8::f32_to_e4m3_rne((index % 11) as f32 * 0.0625))
            .collect();
        let extinction_f32: Vec<_> = extinction
            .iter()
            .map(|&value| crate::cast::e4m3_to_f32(value))
            .collect();
        let source_f32: Vec<_> = source
            .iter()
            .map(|&value| crate::cast::e4m3_to_f32(value))
            .collect();
        let args = ([0.2, -1.0, 0.6], [0.35, 1.0, 0.13], 20.0, 0.8, 1.7, 0.01);
        assert_eq!(
            march_emission_e4m3(
                &extinction,
                &source,
                dims,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
            ),
            march_emission_f32(
                &extinction_f32,
                &source_f32,
                dims,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
            )
        );
    }

    #[test]
    fn emissive_unsigned_variants_match_f32_on_decoded_payload() {
        let dims = [4, 3, 2];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let args = ([0.2, -1.0, 0.6], [0.35, 1.0, 0.13], 20.0, 0.8, 1.7, 0.01);
        for layout in 0..2 {
            let encode = |value| {
                if layout == 0 {
                    crate::cast::f32_to_uf8e4m4_rne(value)
                } else {
                    crate::cast::f32_to_uf8e5m3_rne(value)
                }
            };
            let decode = |bits| {
                if layout == 0 {
                    crate::cast::uf8e4m4_to_f32(bits)
                } else {
                    crate::cast::uf8e5m3_to_f32(bits)
                }
            };
            let extinction: Vec<u8> = (0..count).map(|i| encode((i % 7) as f32 * 0.125)).collect();
            let source: Vec<u8> = (0..count * 3)
                .map(|i| encode((i % 11) as f32 * 0.0625))
                .collect();
            let extinction_f32: Vec<f32> = extinction.iter().map(|&v| decode(v)).collect();
            let source_f32: Vec<f32> = source.iter().map(|&v| decode(v)).collect();
            let got = if layout == 0 {
                march_emission_uf8e4m4(
                    &extinction,
                    &source,
                    dims,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                )
            } else {
                march_emission_uf8e5m3(
                    &extinction,
                    &source,
                    dims,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                )
            };
            let want = march_emission_f32(
                &extinction_f32,
                &source_f32,
                dims,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
            );
            assert_eq!(got, want, "UF8 layout {layout}");
        }
    }

    #[test]
    fn unsigned_fraction_bit_reduces_emissive_transfer_error() {
        let dims = [8, 7, 6];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let extinction: Vec<f32> = (0..count)
            .map(|i| 0.002 + 0.848 * ((i * 43 % 337) as f32 / 337.0))
            .collect();
        let source: Vec<f32> = (0..count * 3)
            .map(|i| 0.01 + 15.0 * ((i * 101 % 1_009) as f32 / 1_009.0))
            .collect();
        let signed_ext: Vec<u8> = extinction
            .iter()
            .map(|&v| crate::fp8::f32_to_e4m3_rne(v))
            .collect();
        let unsigned_ext: Vec<u8> = extinction
            .iter()
            .map(|&v| crate::cast::f32_to_uf8e4m4_rne(v))
            .collect();
        let signed_src: Vec<u8> = source
            .iter()
            .map(|&v| crate::fp8::f32_to_e4m3_rne(v))
            .collect();
        let unsigned_src: Vec<u8> = source
            .iter()
            .map(|&v| crate::cast::f32_to_uf8e4m4_rne(v))
            .collect();

        let (mut signed_error, mut unsigned_error) = (0.0f64, 0.0f64);
        let (mut signed_max, mut unsigned_max) = (0.0f64, 0.0f64);
        for ray in 0..48 {
            let origin = [
                -0.4,
                0.15 + 0.12 * (ray % 41) as f32,
                0.2 + 0.1 * (ray % 37) as f32,
            ];
            let direction = [1.0, 0.017 * (ray % 5) as f32, 0.013 * (ray % 3) as f32];
            let cutoff = if ray & 1 == 0 { 0.0 } else { 0.08 };
            let reference = march_emission_f32(
                &extinction,
                &source,
                dims,
                origin,
                direction,
                20.0,
                0.7,
                1.3,
                cutoff,
            );
            let signed_result = march_emission_e4m3(
                &signed_ext,
                &signed_src,
                dims,
                origin,
                direction,
                20.0,
                0.7,
                1.3,
                cutoff,
            );
            let unsigned_result = march_emission_uf8e4m4(
                &unsigned_ext,
                &unsigned_src,
                dims,
                origin,
                direction,
                20.0,
                0.7,
                1.3,
                cutoff,
            );
            for channel in 0..3 {
                let signed_delta =
                    f64::from(signed_result.radiance[channel] - reference.radiance[channel]);
                let unsigned_delta =
                    f64::from(unsigned_result.radiance[channel] - reference.radiance[channel]);
                signed_error += signed_delta * signed_delta;
                unsigned_error += unsigned_delta * unsigned_delta;
                signed_max = signed_max.max(signed_delta.abs());
                unsigned_max = unsigned_max.max(unsigned_delta.abs());
            }
            for (signed_value, unsigned_value, reference_value) in [
                (
                    signed_result.transmittance,
                    unsigned_result.transmittance,
                    reference.transmittance,
                ),
                (signed_result.t_end, unsigned_result.t_end, reference.t_end),
            ] {
                let signed_delta = f64::from(signed_value - reference_value);
                let unsigned_delta = f64::from(unsigned_value - reference_value);
                signed_error += signed_delta * signed_delta;
                unsigned_error += unsigned_delta * unsigned_delta;
                signed_max = signed_max.max(signed_delta.abs());
                unsigned_max = unsigned_max.max(unsigned_delta.abs());
            }
        }
        assert!(
            unsigned_error < signed_error * 0.8,
            "emissive squared error: UF8={unsigned_error}, E4M3={signed_error}",
        );
        assert!(
            unsigned_max < signed_max,
            "emissive maximum error: UF8={unsigned_max}, E4M3={signed_max}",
        );
    }

    #[test]
    fn emissive_march_stops_at_requested_opacity() {
        let cutoff = axmath::exp_f32(-2.0);
        let transfer = march_emission_f32(
            &[1.0; 4],
            &[1.0; 12],
            [1, 1, 4],
            [0.5, 0.5, -1.0],
            [0.0, 0.0, 1.0],
            10.0,
            1.0,
            1.0,
            cutoff,
        );
        assert_eq!(transfer.transmittance, cutoff);
        assert!((transfer.t_end - 3.0).abs() < 2.0e-5, "{}", transfer.t_end);
        for channel in transfer.radiance {
            assert!((channel - (1.0 - cutoff)).abs() < 2.0e-6);
        }
    }

    #[test]
    fn trilinear_e4m3_sample_uses_voxel_centers_and_interpolates() {
        let dims = [2, 2, 2];
        let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(crate::fp8::f32_to_e4m3_rne);
        assert_eq!(sample_trilinear_e4m3(&values, dims, [0.5, 0.5, 0.5]), 0.0);
        assert_eq!(sample_trilinear_e4m3(&values, dims, [1.5, 1.5, 1.5]), 7.0);
        assert_eq!(sample_trilinear_e4m3(&values, dims, [1.0, 1.0, 1.0]), 3.5);
        assert_eq!(sample_trilinear_e4m3(&values, dims, [-0.01, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn trilinear_unsigned_samples_use_voxel_centers_and_interpolate() {
        let dims = [2, 2, 2];
        let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let e4 = values.map(crate::cast::f32_to_uf8e4m4_rne);
        let e5 = values.map(crate::cast::f32_to_uf8e5m3_rne);
        for (sample, field) in [
            (
                sample_trilinear_uf8e4m4 as fn(&[u8], [u32; 3], [f32; 3]) -> f32,
                e4.as_slice(),
            ),
            (sample_trilinear_uf8e5m3, e5.as_slice()),
        ] {
            assert_eq!(sample(field, dims, [0.5, 0.5, 0.5]), 0.0);
            assert_eq!(sample(field, dims, [1.5, 1.5, 1.5]), 7.0);
            assert_eq!(sample(field, dims, [1.0, 1.0, 1.0]), 3.5);
            assert_eq!(sample(field, dims, [-0.01, 1.0, 1.0]), 0.0);
        }
    }

    #[test]
    fn unsigned_fraction_bit_reduces_trilinear_field_error() {
        fn reference(values: &[f32], dims: [u32; 3], point: [f32; 3]) -> f32 {
            let mut low = [0_u32; 3];
            let mut high = [0_u32; 3];
            let mut fraction = [0.0_f32; 3];
            for axis in 0..3 {
                let centered = (point[axis] - 0.5).clamp(0.0, dims[axis] as f32 - 1.0);
                low[axis] = axmath::floor_f32(centered) as u32;
                high[axis] = (low[axis] + 1).min(dims[axis] - 1);
                fraction[axis] = centered - low[axis] as f32;
            }
            let index = |x: u32, y: u32, z: u32| (x + dims[0] * (y + dims[1] * z)) as usize;
            let lerp = |a: f32, b: f32, amount: f32| a + amount * (b - a);
            let sample = |x: u32, y: u32, z: u32| values[index(x, y, z)];
            let x00 = lerp(
                sample(low[0], low[1], low[2]),
                sample(high[0], low[1], low[2]),
                fraction[0],
            );
            let x10 = lerp(
                sample(low[0], high[1], low[2]),
                sample(high[0], high[1], low[2]),
                fraction[0],
            );
            let x01 = lerp(
                sample(low[0], low[1], high[2]),
                sample(high[0], low[1], high[2]),
                fraction[0],
            );
            let x11 = lerp(
                sample(low[0], high[1], high[2]),
                sample(high[0], high[1], high[2]),
                fraction[0],
            );
            lerp(
                lerp(x00, x10, fraction[1]),
                lerp(x01, x11, fraction[1]),
                fraction[2],
            )
        }

        let dims = [8, 7, 6];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let field: Vec<f32> = (0..count)
            .map(|i| 0.003 + 149.0 * ((i * 101 % 337) as f32 / 337.0))
            .collect();
        let signed: Vec<u8> = field
            .iter()
            .map(|&v| crate::fp8::f32_to_e4m3_rne(v))
            .collect();
        let unsigned: Vec<u8> = field
            .iter()
            .map(|&v| crate::cast::f32_to_uf8e4m4_rne(v))
            .collect();
        let (mut signed_error, mut unsigned_error) = (0.0f64, 0.0f64);
        let (mut signed_max, mut unsigned_max) = (0.0f64, 0.0f64);
        for i in 0..2_048 {
            let point = [
                0.5 + (i * 73 % 699) as f32 / 100.0,
                0.5 + (i * 43 % 599) as f32 / 100.0,
                0.5 + (i * 29 % 499) as f32 / 100.0,
            ];
            let want = reference(&field, dims, point);
            let signed_delta = f64::from(sample_trilinear_e4m3(&signed, dims, point) - want);
            let unsigned_delta = f64::from(sample_trilinear_uf8e4m4(&unsigned, dims, point) - want);
            signed_error += signed_delta * signed_delta;
            unsigned_error += unsigned_delta * unsigned_delta;
            signed_max = signed_max.max(signed_delta.abs());
            unsigned_max = unsigned_max.max(unsigned_delta.abs());
        }
        assert!(
            unsigned_error < signed_error * 0.35,
            "trilinear squared error: UF8={unsigned_error}, E4M3={signed_error}",
        );
        assert!(
            unsigned_max < signed_max,
            "trilinear maximum error: UF8={unsigned_max}, E4M3={signed_max}",
        );
    }
}
