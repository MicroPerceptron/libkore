//! AMX qdot — tile-based INT8 matrix multiply for Q8xQ8 GEMM.
//!
//! vec_dot: re-exported from portable (AMX tiles have prohibitive setup
//! cost for single-vector operations).
//!
//! gemm_q8_q8: AMX tile implementation following GGML's mmq.cpp pattern.
//! gemm_q4_q8, gemm_q6_q8: delegate to portable (VNNI weight packing deferred).
//!
//! ## Tile layout (2-2-4 pattern)
//!
//!            B TMM0   B TMM1
//!   A TMM2   C TMM4   C TMM6
//!   A TMM3   C TMM5   C TMM7
//!
//! Per K-block iteration the kernel uses a single A-B-C triple
//! (TMM2, TMM0, TMM4). The full 2-2-4 pattern (double-buffered across
//! both tile halves) is reserved for a future double-buffered version
//! that overlaps AVX-512 scale accumulation with the next tile multiply.
//!
//! ## Per-block scale limitation
//!
//! Q8_0 carries an f16 scale per 32-element block, so the i32 tile
//! accumulator must be drained and scaled after every K-block — the
//! tile cannot accumulate across blocks. This limits the AMX advantage
//! to the 16x16x32 multiply itself; future double-buffering will
//! overlap the scale post-processing with the next tile multiply.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE,
    Q8_QS_OFFSET,
};

// ── Constants ────────────────────────────────────────────────────────

const TILE_M: usize = 16;
const TILE_N: usize = 16;
const TILE_K: usize = 32; // = Q8_BLOCK_SIZE
const VNNI_BLK: usize = 4;

// TMM register assignments (2-2-4 pattern).  Only the first triple
// (TMM0, TMM2, TMM4) is used in the current single-buffered kernel.
const TMM_B0: i32 = 0;
#[allow(dead_code)]
const TMM_B1: i32 = 1;
const TMM_A0: i32 = 2;
#[allow(dead_code)]
const TMM_A1: i32 = 3;
const TMM_C0: i32 = 4;
#[allow(dead_code)]
const TMM_C1: i32 = 5;
#[allow(dead_code)]
const TMM_C2: i32 = 6;
#[allow(dead_code)]
const TMM_C3: i32 = 7;

// ── Tile configuration ──────────────────────────────────────────────

/// 64-byte tile configuration descriptor loaded via `_tile_loadconfig`.
///
/// See Intel AMX Architecture Specification, section "PALETTE_TABLE".
#[repr(C, align(64))]
struct TileConfig {
    palette_id: u8,
    start_row: u8,
    _reserved: [u8; 14],
    colsb: [u16; 8],
    _reserved2: [u8; 16],
    rows: [u8; 8],
    _reserved3: [u8; 8],
}

impl TileConfig {
    const fn new() -> Self {
        let mut tc = TileConfig {
            palette_id: 1,
            start_row: 0,
            _reserved: [0; 14],
            colsb: [0; 8],
            _reserved2: [0; 16],
            rows: [0; 8],
            _reserved3: [0; 8],
        };
        // B tiles: 8 rows x 64 bytes (TILE_K/VNNI_BLK rows, TILE_N*VNNI_BLK cols)
        tc.rows[TMM_B0 as usize] = 8;
        tc.colsb[TMM_B0 as usize] = 64;
        tc.rows[TMM_B1 as usize] = 8;
        tc.colsb[TMM_B1 as usize] = 64;
        // A tiles: 16 rows x 32 bytes (TILE_M rows, TILE_K cols)
        tc.rows[TMM_A0 as usize] = 16;
        tc.colsb[TMM_A0 as usize] = 32;
        tc.rows[TMM_A1 as usize] = 16;
        tc.colsb[TMM_A1 as usize] = 32;
        // C tiles: 16 rows x 64 bytes (TILE_M rows, TILE_N * 4B i32)
        tc.rows[TMM_C0 as usize] = 16;
        tc.colsb[TMM_C0 as usize] = 64;
        tc.rows[TMM_C1 as usize] = 16;
        tc.colsb[TMM_C1 as usize] = 64;
        tc.rows[TMM_C2 as usize] = 16;
        tc.colsb[TMM_C2 as usize] = 64;
        tc.rows[TMM_C3 as usize] = 16;
        tc.colsb[TMM_C3 as usize] = 64;
        tc
    }
}

static TILE_CFG: TileConfig = TileConfig::new();

// ── vec_dot: re-export from portable ────────────────────────────────

pub(crate) use crate::arch::portable::qdot::{vec_dot_q4_q8, vec_dot_q6_q8, vec_dot_q8_q8};

// ── GEMM: Q4 and Q6 delegate to portable ───────────────────────────
//
// VNNI packing for sub-byte quant types (Q4_K, Q6_K) requires nibble/
// 6-bit unpack and repack — deferred to a future change.

pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q4_BLOCK_SIZE, 0);
    let w_row_bytes = k / Q4_BLOCK_SIZE * Q4_BLOCK_BYTES;
    let x_row_bytes = k / Q8_BLOCK_SIZE * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q4_q8(w_row, x_row, k);
        }
    }
}

pub(crate) fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q6_BLOCK_SIZE, 0);
    let w_row_bytes = k / Q6_BLOCK_SIZE * Q6_BLOCK_BYTES;
    let x_row_bytes = k / Q8_BLOCK_SIZE * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q6_q8(w_row, x_row, k);
        }
    }
}

// ── GEMM: Q8 x Q8 — AMX tile implementation ────────────────────────

/// Quantized Q8_0 x Q8_0 matrix multiply using AMX `_tile_dpbssd`.
///
/// Layout conventions:
///   - `w`: weight matrix, M rows of quantized Q8_0 blocks (row-major)
///   - `x`: activation matrix, N rows of quantized Q8_0 blocks (row-major)
///   - `c`: output M x N f32, row-major, **accumulated** (caller zeros)
///   - Dimensions: M output rows, N output cols, K inner dimension
///
/// Falls back to portable vec_dot for:
///   - Small N (<=4): setup overhead dominates
///   - Edge tiles that don't fill a full TILE_M x TILE_N
pub(crate) fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q8_BLOCK_SIZE, 0);
    let nb = k / Q8_BLOCK_SIZE;
    let row_bytes = nb * Q8_BLOCK_BYTES;

    // Small-N fast path: vec_dot per (i,j) avoids tile setup overhead.
    if n <= 4 || m < TILE_M || k < TILE_K {
        for i in 0..m {
            let w_row = &w[i * row_bytes..][..row_bytes];
            for j in 0..n {
                let x_row = &x[j * row_bytes..][..row_bytes];
                c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
            }
        }
        return;
    }

    unsafe { gemm_q8_q8_amx(w, x, c, m, n, k, nb, row_bytes) }
}

/// Inner AMX kernel, factored out for `unsafe` scoping clarity.
///
/// # Safety
///
/// Requires AMX-INT8 + AMX-TILE target features. Caller must verify
/// dimensions are valid and slices are large enough.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "amx-int8,amx-tile")]
unsafe fn gemm_q8_q8_amx(
    w: &[u8],
    x: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    row_bytes: usize,
) {
    unsafe {
        // Configure tile registers.
        _tile_loadconfig(core::ptr::addr_of!(TILE_CFG) as *const u8);

        // Scratch buffer for VNNI-transposed B quants (one tile's worth).
        // Layout: [TILE_K/VNNI_BLK][TILE_N][VNNI_BLK] = [8][16][4] = 512 bytes,
        // stored row-major with stride = TILE_N * VNNI_BLK = 64 bytes.
        let mut packed_b = [0u8; TILE_N * TILE_K];

        // Scratch for storing the i32 tile result.
        // Layout: TILE_M rows x TILE_N cols of i32.
        #[repr(align(64))]
        struct TileBuf([i32; TILE_M * TILE_N]);
        let mut tile_buf = TileBuf([0i32; TILE_M * TILE_N]);

        // Tile over M (rows of W / rows of output).
        for i0 in (0..m).step_by(TILE_M) {
            let i_end = (i0 + TILE_M).min(m);
            let i_count = i_end - i0;

            // Tile over N (rows of X / cols of output).
            for j0 in (0..n).step_by(TILE_N) {
                let j_end = (j0 + TILE_N).min(n);
                let j_count = j_end - j0;

                // Edge tiles: fall back to vec_dot.
                if i_count < TILE_M || j_count < TILE_N {
                    for i in i0..i_end {
                        let w_row = &w[i * row_bytes..][..row_bytes];
                        for j in j0..j_end {
                            let x_row = &x[j * row_bytes..][..row_bytes];
                            c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
                        }
                    }
                    continue;
                }

                // Full tile: AMX path.  Accumulate across K-blocks in f32.
                for kb in 0..nb {
                    // ── Pack B: transpose TILE_N Q8 blocks into VNNI format ──
                    //
                    // Source: x[(j0+n_idx) * row_bytes + kb * Q8_BLOCK_BYTES + Q8_QS_OFFSET]
                    //   for n_idx in 0..TILE_N, quant index k_idx in 0..TILE_K
                    //
                    // Target: packed_b[(k_idx/4) * TILE_N * 4 + n_idx * 4 + k_idx%4]
                    pack_q8_vnni(&mut packed_b, x, row_bytes, j0, kb);

                    // ── Zero C accumulator tile ──
                    _tile_zero::<TMM_C0>();

                    // ── Load A tile: TILE_M rows from W ──
                    //
                    // A[row][col] = w[(i0+row)*row_bytes + kb*Q8_BLOCK_BYTES + Q8_QS_OFFSET + col]
                    // Stride between rows = row_bytes (one full Q8 row of blocks).
                    let a_ptr = w
                        .as_ptr()
                        .add(i0 * row_bytes + kb * Q8_BLOCK_BYTES + Q8_QS_OFFSET);
                    _tile_loadd::<TMM_A0>(a_ptr, row_bytes);

                    // ── Load B tile: VNNI-packed buffer ──
                    //
                    // 8 rows x 64 bytes, stride = 64 (contiguous).
                    _tile_loadd::<TMM_B0>(packed_b.as_ptr(), TILE_N * VNNI_BLK);

                    // ── Tile multiply: C += A * B (signed x signed) ──
                    _tile_dpbssd::<TMM_C0, TMM_A0, TMM_B0>();

                    // ── Store tile result to scratch buffer ──
                    _tile_stored::<TMM_C0>(tile_buf.0.as_mut_ptr() as *mut u8, TILE_N * 4);

                    // ── Scale accumulation ──
                    //
                    // For each output element (ti, tj):
                    //   c[i0+ti][j0+tj] += w_scale[i0+ti, kb] * x_scale[j0+tj, kb] * tile_buf[ti][tj]
                    //
                    // The per-block f16 scale sits at the start of each Q8 block.
                    for ti in 0..TILE_M {
                        let w_block_ptr =
                            w.as_ptr().add((i0 + ti) * row_bytes + kb * Q8_BLOCK_BYTES);
                        let wd =
                            f16_to_f32(u16::from_le_bytes([*w_block_ptr, *w_block_ptr.add(1)]));

                        let c_row = (i0 + ti) * n + j0;
                        let t_row = ti * TILE_N;

                        for tj in 0..TILE_N {
                            let x_block_ptr =
                                x.as_ptr().add((j0 + tj) * row_bytes + kb * Q8_BLOCK_BYTES);
                            let xd =
                                f16_to_f32(u16::from_le_bytes([*x_block_ptr, *x_block_ptr.add(1)]));

                            *c.as_mut_ptr().add(c_row + tj) +=
                                wd * xd * tile_buf.0[t_row + tj] as f32;
                        }
                    }
                }
            }
        }

        _tile_release();
    }
}

// ── B-tile VNNI packing ─────────────────────────────────────────────

/// Transpose TILE_N Q8 quant vectors into VNNI layout for B tile load.
///
/// Source layout (per block): 32 contiguous i8 quants at
///   `x[(j0+n)*row_bytes + kb*Q8_BLOCK_BYTES + Q8_QS_OFFSET + k]`
///
/// Target VNNI layout: `packed[k/4][n][k%4]`
///   = `packed[(k/4) * TILE_N * 4  +  n * 4  +  k%4]`
///
/// Total: 8 x 16 x 4 = 512 bytes with row stride 64.
///
/// This is a scalar implementation.  A future version can vectorize with
/// AVX-512 byte shuffles (cf. GGML `pack_B<block_q8_0>`).
#[inline(always)]
fn pack_q8_vnni(
    packed: &mut [u8; TILE_N * TILE_K],
    x: &[u8],
    row_bytes: usize,
    j0: usize,
    kb: usize,
) {
    for n_idx in 0..TILE_N {
        let qs_base = (j0 + n_idx) * row_bytes + kb * Q8_BLOCK_BYTES + Q8_QS_OFFSET;
        for k_idx in 0..TILE_K {
            let vnni_idx =
                (k_idx / VNNI_BLK) * (TILE_N * VNNI_BLK) + n_idx * VNNI_BLK + (k_idx % VNNI_BLK);
            packed[vnni_idx] = x[qs_base + k_idx];
        }
    }
}
