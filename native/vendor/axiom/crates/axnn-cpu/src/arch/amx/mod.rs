//! AMX (Advanced Matrix Extensions) qdot override.
//!
//! AMX accelerates GEMM via tile-based INT8 matrix multiplication
//! (`_tile_dpbssd`).  Single vec_dot operations re-export from the
//! portable tier — AMX tiles have high setup overhead that makes
//! them counterproductive for single-vector dot products.
//!
//! ## Tile layout (2-2-4 pattern, per GGML/oneDNN)
//!
//!            B TMM0   B TMM1
//!   A TMM2   C TMM4   C TMM6
//!   A TMM3   C TMM5   C TMM7
//!
//!   A tiles: 16 rows × 32 cols (TILE_M × TILE_K)
//!   B tiles: 8 rows × 64 cols  (TILE_K/VNNI_BLK × TILE_N × VNNI_BLK)
//!   C tiles: 16 rows × 16 cols i32 (TILE_M × TILE_N)
//!
//! ## Requires
//!
//! `target_feature = "amx-int8,amx-tile"` + AVX-512 for scale
//! accumulation.  Weight B must be pre-packed to VNNI format.

#[allow(dead_code)]
pub(crate) mod qdot;
