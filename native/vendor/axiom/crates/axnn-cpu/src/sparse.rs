//! Packed block-sparse matrix kernels.
//!
//! This module keeps sparsity an execution detail: callers retain their
//! logical dense matrix while immutable weights are packed once into BSR
//! (block-compressed row) storage.  Blocks span the reduction axis, so every
//! stored block is contiguous in both the weight and activation planes.  That
//! keeps SIMD dot kernels useful and avoids the per-element index traffic of
//! CSR/COO.
//!
//! The matrix is stored as `W[output, input]`; [`gemm_f32`] computes
//! `C[batch, output] += X[batch, input] × Wᵀ`.  This is the natural packed
//! form for an ordinary dense `X × B` operation when its immutable right-hand
//! operand `B[input, output]` is packed through [`pack_f32_transposed`].

use core::fmt;
use core::simd::Simd;

/// Default reduction-axis tile width.  It matches the Q8 quantization block
/// and is large enough to amortize one column index over SIMD work.
pub const DEFAULT_BLOCK_K: usize = 32;

// Match the dense portable tier's native F32 register width.  BSR blocks are
// contiguous along K, so the same load/FMA shape applies without gathers.
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
const F32_LANES: usize = 8;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
const F32_LANES: usize = 4;

/// Static profile of a BSR packing opportunity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockSparseProfile {
    /// Logical matrix rows (`output` features in [`BsrMatrix`]).
    pub rows: usize,
    /// Logical matrix columns (`input` features in [`BsrMatrix`]).
    pub columns: usize,
    /// Logical elements stored by one BSR block along the reduction axis.
    pub block_k: usize,
    /// Number of candidate blocks in the dense matrix.
    pub total_blocks: usize,
    /// Number of nonzero blocks retained by the packed matrix.
    pub nonzero_blocks: usize,
    /// Logical values in retained blocks (the tail block is not padded here).
    pub stored_values: usize,
}

impl BlockSparseProfile {
    /// Fraction of retained blocks in parts per million.  Integer form keeps
    /// cost-model decisions deterministic and `no_std` friendly.
    pub const fn block_density_ppm(self) -> u32 {
        if self.total_blocks == 0 {
            return 0;
        }
        ((self.nonzero_blocks as u128 * 1_000_000) / self.total_blocks as u128) as u32
    }

    /// Number of zero blocks removed by BSR packing.
    pub const fn zero_blocks(self) -> usize {
        self.total_blocks.saturating_sub(self.nonzero_blocks)
    }

    /// Number of value slots required by the physical packed representation.
    /// The last block in each row is zero-padded to preserve fixed-stride
    /// block loads.
    pub const fn packed_values(self) -> Option<usize> {
        self.nonzero_blocks.checked_mul(self.block_k)
    }
}

/// Errors returned while profiling, packing, validating, or executing BSR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseError {
    ZeroBlockWidth,
    ShapeOverflow,
    SourceLength { expected: usize, actual: usize },
    RowOffsetsLength { expected: usize, actual: usize },
    ColumnIndicesLength { expected: usize, actual: usize },
    ValuesLength { expected: usize, actual: usize },
    InvalidRowOffsets,
    InvalidColumnIndex { row: usize, column: usize },
    ActivationLength { expected: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
    IndexOverflow,
}

impl fmt::Display for SparseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// A borrowed BSR matrix with one logical output row per BSR row.
///
/// `row_offsets` has `rows + 1` entries.  Every value block has exactly
/// `block_k` elements; the final reduction-axis block is zero-padded when the
/// logical column count is not a multiple of `block_k`.
#[derive(Clone, Copy, Debug)]
pub struct BsrMatrix<'a, T> {
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a [u32],
    column_indices: &'a [u32],
    values: &'a [T],
}

impl<'a, T> BsrMatrix<'a, T> {
    /// Validates and borrows a packed BSR matrix.
    pub fn new(
        rows: usize,
        columns: usize,
        block_k: usize,
        row_offsets: &'a [u32],
        column_indices: &'a [u32],
        values: &'a [T],
    ) -> Result<Self, SparseError> {
        if block_k == 0 {
            return Err(SparseError::ZeroBlockWidth);
        }
        let offset_len = rows.checked_add(1).ok_or(SparseError::ShapeOverflow)?;
        if row_offsets.len() != offset_len {
            return Err(SparseError::RowOffsetsLength {
                expected: offset_len,
                actual: row_offsets.len(),
            });
        }
        let expected_values = column_indices
            .len()
            .checked_mul(block_k)
            .ok_or(SparseError::ShapeOverflow)?;
        if values.len() != expected_values {
            return Err(SparseError::ValuesLength {
                expected: expected_values,
                actual: values.len(),
            });
        }
        let column_count =
            u32::try_from(column_indices.len()).map_err(|_| SparseError::IndexOverflow)?;
        if row_offsets.first().copied() != Some(0)
            || row_offsets.last().copied() != Some(column_count)
        {
            return Err(SparseError::InvalidRowOffsets);
        }
        let blocks_per_row = blocks_per_row(columns, block_k)?;
        let mut previous = 0usize;
        for row in 0..rows {
            let end = row_offsets[row + 1] as usize;
            if end < previous || end > column_indices.len() {
                return Err(SparseError::InvalidRowOffsets);
            }
            for &column in &column_indices[previous..end] {
                if column as usize >= blocks_per_row {
                    return Err(SparseError::InvalidColumnIndex {
                        row,
                        column: column as usize,
                    });
                }
            }
            previous = end;
        }
        Ok(Self {
            rows,
            columns,
            block_k,
            row_offsets,
            column_indices,
            values,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn block_k(&self) -> usize {
        self.block_k
    }

    pub const fn nonzero_blocks(&self) -> usize {
        self.column_indices.len()
    }

    pub const fn row_offsets(&self) -> &'a [u32] {
        self.row_offsets
    }

    pub const fn column_indices(&self) -> &'a [u32] {
        self.column_indices
    }

    pub const fn values(&self) -> &'a [T] {
        self.values
    }
}

/// Profile a row-major F32 matrix without allocating.
pub fn profile_f32(
    weights: &[f32],
    rows: usize,
    columns: usize,
    block_k: usize,
) -> Result<BlockSparseProfile, SparseError> {
    profile(
        rows,
        columns,
        block_k,
        |row, column, width| {
            weights[row * columns + column..][..width]
                .iter()
                .any(|&value| value != 0.0)
        },
        weights.len(),
    )
}

/// Profile a row-major FP8 matrix.  Both positive and negative zero are
/// treated as sparse zeros; every other encoding, including NaN, is retained.
pub fn profile_fp8(
    weights: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
) -> Result<BlockSparseProfile, SparseError> {
    profile(
        rows,
        columns,
        block_k,
        |row, column, width| {
            weights[row * columns + column..][..width]
                .iter()
                .any(|&value| value & 0x7f != 0)
        },
        weights.len(),
    )
}

/// Profile a standard dense right-hand operand `B[columns, rows]` as the
/// transposed packed matrix `W = Bᵀ[rows, columns]`.
pub fn profile_f32_transposed(
    source: &[f32],
    rows: usize,
    columns: usize,
    block_k: usize,
) -> Result<BlockSparseProfile, SparseError> {
    profile(
        rows,
        columns,
        block_k,
        |row, column, width| {
            (column..column + width).any(|input| source[input * rows + row] != 0.0)
        },
        source.len(),
    )
}

/// Profile little-endian F32 bytes holding a standard dense right-hand
/// operand `B[columns, rows]` as the transposed packed matrix
/// `W = Bᵀ[rows, columns]`.  This is useful at graph-compilation time, where
/// constants are represented as canonical bytes and may not be F32-aligned.
pub fn profile_f32_le_transposed(
    source: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
) -> Result<BlockSparseProfile, SparseError> {
    validate_f32_bytes(source, rows, columns)?;
    profile(
        rows,
        columns,
        block_k,
        |row, column, width| {
            (column..column + width).any(|input| read_f32_le(source, input * rows + row) != 0.0)
        },
        rows.checked_mul(columns)
            .ok_or(SparseError::ShapeOverflow)?,
    )
}

/// Profile a standard dense FP8 right-hand operand `B[columns, rows]` as
/// `W = Bᵀ[rows, columns]`.
pub fn profile_fp8_transposed(
    source: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
) -> Result<BlockSparseProfile, SparseError> {
    profile(
        rows,
        columns,
        block_k,
        |row, column, width| {
            (column..column + width).any(|input| source[input * rows + row] & 0x7f != 0)
        },
        source.len(),
    )
}

/// Pack a row-major F32 `W[rows, columns]` into caller-owned BSR storage.
pub fn pack_f32<'a>(
    weights: &[f32],
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [f32],
) -> Result<BsrMatrix<'a, f32>, SparseError> {
    pack(
        rows,
        columns,
        block_k,
        |row, column| weights[row * columns + column],
        |value| value == 0.0,
        row_offsets,
        column_indices,
        values,
        weights.len(),
    )
}

/// Pack a row-major FP8 `W[rows, columns]` into caller-owned BSR storage.
pub fn pack_fp8<'a>(
    weights: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [u8],
) -> Result<BsrMatrix<'a, u8>, SparseError> {
    pack(
        rows,
        columns,
        block_k,
        |row, column| weights[row * columns + column],
        |value| value & 0x7f == 0,
        row_offsets,
        column_indices,
        values,
        weights.len(),
    )
}

/// Pack the transpose of a standard F32 right-hand operand
/// `B[columns, rows]` into BSR `W = Bᵀ[rows, columns]`.
pub fn pack_f32_transposed<'a>(
    source: &[f32],
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [f32],
) -> Result<BsrMatrix<'a, f32>, SparseError> {
    pack(
        rows,
        columns,
        block_k,
        |row, column| source[column * rows + row],
        |value| value == 0.0,
        row_offsets,
        column_indices,
        values,
        source.len(),
    )
}

/// Pack little-endian F32 bytes holding a standard dense right-hand operand
/// `B[columns, rows]` into BSR `W = Bᵀ[rows, columns]` without requiring an
/// aligned temporary F32 plane.
pub fn pack_f32_le_transposed<'a>(
    source: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [f32],
) -> Result<BsrMatrix<'a, f32>, SparseError> {
    validate_f32_bytes(source, rows, columns)?;
    pack(
        rows,
        columns,
        block_k,
        |row, column| read_f32_le(source, column * rows + row),
        |value| value == 0.0,
        row_offsets,
        column_indices,
        values,
        rows.checked_mul(columns)
            .ok_or(SparseError::ShapeOverflow)?,
    )
}

/// Pack the transpose of a standard FP8 right-hand operand
/// `B[columns, rows]` into BSR `W = Bᵀ[rows, columns]`.
pub fn pack_fp8_transposed<'a>(
    source: &[u8],
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [u8],
) -> Result<BsrMatrix<'a, u8>, SparseError> {
    pack(
        rows,
        columns,
        block_k,
        |row, column| source[column * rows + row],
        |value| value & 0x7f == 0,
        row_offsets,
        column_indices,
        values,
        source.len(),
    )
}

/// `C[batch, rows] += X[batch, columns] × Wᵀ` for packed F32 weights.
///
/// The hot reduction loop has contiguous block loads from both operands.  It
/// uses the same ordinary multiply-add form as the dense portable kernels, so
/// LLVM can select the target's legal fused instruction where appropriate.
pub fn gemm_f32(
    weights: BsrMatrix<'_, f32>,
    activations: &[f32],
    output: &mut [f32],
    batch: usize,
) -> Result<(), SparseError> {
    validate_gemm(&weights, activations.len(), output.len(), batch)?;
    for row in 0..weights.rows {
        let start = weights.row_offsets[row] as usize;
        let end = weights.row_offsets[row + 1] as usize;
        for sample in 0..batch {
            let input = &activations[sample * weights.columns..][..weights.columns];
            let mut sum = output[sample * weights.rows + row];
            for block in start..end {
                let column = weights.column_indices[block] as usize * weights.block_k;
                let width = (weights.columns - column).min(weights.block_k);
                let values = &weights.values[block * weights.block_k..][..width];
                sum += dot_f32(values, &input[column..][..width]);
            }
            output[sample * weights.rows + row] = sum;
        }
    }
    Ok(())
}

/// `C[batch, rows] += X[batch, columns] × Wᵀ` for E4M3 BSR weights and
/// F32 activations.  Each retained block uses axnn-cpu's ISA-dispatched FP8
/// dot kernel rather than scalar per-element decoding.
pub fn gemm_e4m3_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[f32],
    output: &mut [f32],
    batch: usize,
) -> Result<(), SparseError> {
    gemm_fp8_f32(
        weights,
        activations,
        output,
        batch,
        crate::fp8::vec_dot_e4m3_f32,
    )
}

/// `C[batch, rows] += X[batch, columns] × Wᵀ` for E5M2 BSR weights and
/// F32 activations.
pub fn gemm_e5m2_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[f32],
    output: &mut [f32],
    batch: usize,
) -> Result<(), SparseError> {
    gemm_fp8_f32(
        weights,
        activations,
        output,
        batch,
        crate::fp8::vec_dot_e5m2_f32,
    )
}

/// `C[batch, rows] += decode(X) × decode(W)ᵀ` for E4M3 BSR weights and
/// E4M3 activations, with F32 accumulation.
///
/// This keeps both dynamic activations and immutable blocks in FP8 storage;
/// the ISA-specific dot kernel widens them only inside its vector reduction.
pub fn gemm_e4m3_e4m3_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[u8],
    output: &mut [f32],
    batch: usize,
) -> Result<(), SparseError> {
    gemm_fp8_fp8_f32(
        weights,
        activations,
        output,
        batch,
        crate::fp8::dot_e4m3_e4m3,
    )
}

/// `C[batch, rows] += decode(X) × decode(W)ᵀ` for E5M2 BSR weights and
/// E5M2 activations, with F32 accumulation.
pub fn gemm_e5m2_e5m2_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[u8],
    output: &mut [f32],
    batch: usize,
) -> Result<(), SparseError> {
    gemm_fp8_fp8_f32(
        weights,
        activations,
        output,
        batch,
        crate::fp8::dot_e5m2_e5m2,
    )
}

fn gemm_fp8_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[f32],
    output: &mut [f32],
    batch: usize,
    dot: fn(&[u8], &[f32], usize) -> f32,
) -> Result<(), SparseError> {
    validate_gemm(&weights, activations.len(), output.len(), batch)?;
    for row in 0..weights.rows {
        let start = weights.row_offsets[row] as usize;
        let end = weights.row_offsets[row + 1] as usize;
        for sample in 0..batch {
            let input = &activations[sample * weights.columns..][..weights.columns];
            let mut sum = output[sample * weights.rows + row];
            for block in start..end {
                let column = weights.column_indices[block] as usize * weights.block_k;
                let width = (weights.columns - column).min(weights.block_k);
                let values = &weights.values[block * weights.block_k..][..width];
                sum += dot(values, &input[column..][..width], width);
            }
            output[sample * weights.rows + row] = sum;
        }
    }
    Ok(())
}

fn gemm_fp8_fp8_f32(
    weights: BsrMatrix<'_, u8>,
    activations: &[u8],
    output: &mut [f32],
    batch: usize,
    dot: fn(&[u8], &[u8], usize) -> f32,
) -> Result<(), SparseError> {
    validate_gemm(&weights, activations.len(), output.len(), batch)?;
    for row in 0..weights.rows {
        let start = weights.row_offsets[row] as usize;
        let end = weights.row_offsets[row + 1] as usize;
        for sample in 0..batch {
            let input = &activations[sample * weights.columns..][..weights.columns];
            let mut sum = output[sample * weights.rows + row];
            for block in start..end {
                let column = weights.column_indices[block] as usize * weights.block_k;
                let width = (weights.columns - column).min(weights.block_k);
                let values = &weights.values[block * weights.block_k..][..width];
                sum += dot(values, &input[column..][..width], width);
            }
            output[sample * weights.rows + row] = sum;
        }
    }
    Ok(())
}

/// Contiguous SIMD dot used for F32 BSR tiles.  Two independent accumulator
/// chains keep FMA latency from serializing the common 32-element block.
#[inline(always)]
fn dot_f32(weights: &[f32], activations: &[f32]) -> f32 {
    debug_assert_eq!(weights.len(), activations.len());
    let pairs = weights.len() / (2 * F32_LANES);
    let mut even = Simd::<f32, F32_LANES>::splat(0.0);
    let mut odd = Simd::<f32, F32_LANES>::splat(0.0);
    for pair in 0..pairs {
        let first = pair * 2 * F32_LANES;
        let second = first + F32_LANES;
        even += Simd::from_slice(&weights[first..]) * Simd::from_slice(&activations[first..]);
        odd += Simd::from_slice(&weights[second..]) * Simd::from_slice(&activations[second..]);
    }
    let mut sum: f32 = (even + odd).to_array().into_iter().sum();
    for index in pairs * 2 * F32_LANES..weights.len() {
        sum += weights[index] * activations[index];
    }
    sum
}

fn profile(
    rows: usize,
    columns: usize,
    block_k: usize,
    mut is_nonzero: impl FnMut(usize, usize, usize) -> bool,
    source_len: usize,
) -> Result<BlockSparseProfile, SparseError> {
    validate_source_len(source_len, rows, columns)?;
    let blocks_per_row = blocks_per_row(columns, block_k)?;
    let total_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or(SparseError::ShapeOverflow)?;
    let mut nonzero_blocks = 0usize;
    let mut stored_values = 0usize;
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let column = block * block_k;
            let width = (columns - column).min(block_k);
            if is_nonzero(row, column, width) {
                nonzero_blocks = nonzero_blocks
                    .checked_add(1)
                    .ok_or(SparseError::ShapeOverflow)?;
                stored_values = stored_values
                    .checked_add(width)
                    .ok_or(SparseError::ShapeOverflow)?;
            }
        }
    }
    Ok(BlockSparseProfile {
        rows,
        columns,
        block_k,
        total_blocks,
        nonzero_blocks,
        stored_values,
    })
}

#[allow(clippy::too_many_arguments)]
fn pack<'a, T: Copy + Default>(
    rows: usize,
    columns: usize,
    block_k: usize,
    value_at: impl Fn(usize, usize) -> T,
    is_zero: impl Fn(T) -> bool,
    row_offsets: &'a mut [u32],
    column_indices: &'a mut [u32],
    values: &'a mut [T],
    source_len: usize,
) -> Result<BsrMatrix<'a, T>, SparseError> {
    validate_source_len(source_len, rows, columns)?;
    let blocks_per_row = blocks_per_row(columns, block_k)?;
    let needed_offsets = rows.checked_add(1).ok_or(SparseError::ShapeOverflow)?;
    if row_offsets.len() < needed_offsets {
        return Err(SparseError::RowOffsetsLength {
            expected: needed_offsets,
            actual: row_offsets.len(),
        });
    }

    let mut retained = 0usize;
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let column = block * block_k;
            let width = (columns - column).min(block_k);
            if (0..width).any(|offset| !is_zero(value_at(row, column + offset))) {
                retained = retained.checked_add(1).ok_or(SparseError::ShapeOverflow)?;
            }
        }
    }
    if column_indices.len() < retained {
        return Err(SparseError::ColumnIndicesLength {
            expected: retained,
            actual: column_indices.len(),
        });
    }
    let needed_values = retained
        .checked_mul(block_k)
        .ok_or(SparseError::ShapeOverflow)?;
    if values.len() < needed_values {
        return Err(SparseError::ValuesLength {
            expected: needed_values,
            actual: values.len(),
        });
    }

    let mut cursor = 0usize;
    for row in 0..rows {
        row_offsets[row] = u32::try_from(cursor).map_err(|_| SparseError::IndexOverflow)?;
        for block in 0..blocks_per_row {
            let column = block * block_k;
            let width = (columns - column).min(block_k);
            if !(0..width).any(|offset| !is_zero(value_at(row, column + offset))) {
                continue;
            }
            column_indices[cursor] =
                u32::try_from(block).map_err(|_| SparseError::IndexOverflow)?;
            let destination = &mut values[cursor * block_k..][..block_k];
            destination.fill(T::default());
            for offset in 0..width {
                destination[offset] = value_at(row, column + offset);
            }
            cursor += 1;
        }
    }
    row_offsets[rows] = u32::try_from(cursor).map_err(|_| SparseError::IndexOverflow)?;
    BsrMatrix::new(
        rows,
        columns,
        block_k,
        &row_offsets[..needed_offsets],
        &column_indices[..cursor],
        &values[..needed_values],
    )
}

fn validate_gemm<T>(
    weights: &BsrMatrix<'_, T>,
    activation_len: usize,
    output_len: usize,
    batch: usize,
) -> Result<(), SparseError> {
    let needed_activations = batch
        .checked_mul(weights.columns)
        .ok_or(SparseError::ShapeOverflow)?;
    if activation_len < needed_activations {
        return Err(SparseError::ActivationLength {
            expected: needed_activations,
            actual: activation_len,
        });
    }
    let needed_output = batch
        .checked_mul(weights.rows)
        .ok_or(SparseError::ShapeOverflow)?;
    if output_len < needed_output {
        return Err(SparseError::OutputLength {
            expected: needed_output,
            actual: output_len,
        });
    }
    Ok(())
}

fn validate_source_len(source_len: usize, rows: usize, columns: usize) -> Result<(), SparseError> {
    let expected = rows
        .checked_mul(columns)
        .ok_or(SparseError::ShapeOverflow)?;
    if source_len != expected {
        return Err(SparseError::SourceLength {
            expected,
            actual: source_len,
        });
    }
    Ok(())
}

fn validate_f32_bytes(source: &[u8], rows: usize, columns: usize) -> Result<(), SparseError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or(SparseError::ShapeOverflow)?;
    let expected = elements
        .checked_mul(size_of::<f32>())
        .ok_or(SparseError::ShapeOverflow)?;
    if source.len() != expected {
        return Err(SparseError::SourceLength {
            expected,
            actual: source.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn read_f32_le(source: &[u8], index: usize) -> f32 {
    let offset = index * size_of::<f32>();
    f32::from_le_bytes(
        source[offset..offset + size_of::<f32>()]
            .try_into()
            .unwrap(),
    )
}

fn blocks_per_row(columns: usize, block_k: usize) -> Result<usize, SparseError> {
    if block_k == 0 {
        return Err(SparseError::ZeroBlockWidth);
    }
    columns
        .checked_add(block_k - 1)
        .map(|total| total / block_k)
        .ok_or(SparseError::ShapeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_tracks_block_profile_and_pads_the_tail() {
        let weights = [
            1.0, 0.0, 0.0, 0.0, // row 0, block 0
            0.0, 0.0, 0.0, 0.0, // row 0, block 1
            2.0, // row 0, tail block 2
            0.0, 0.0, 0.0, 0.0, // row 1, block 0
            3.0, 0.0, 0.0, 0.0, // row 1, block 1
            0.0, // row 1, tail block 2
        ];
        let profile = profile_f32(&weights, 2, 9, 4).unwrap();
        assert_eq!(profile.total_blocks, 6);
        assert_eq!(profile.nonzero_blocks, 3);
        assert_eq!(profile.stored_values, 9);
        assert_eq!(profile.block_density_ppm(), 500_000);

        let mut offsets = [0_u32; 3];
        let mut columns = [0_u32; 3];
        let mut values = [0.0_f32; 12];
        let matrix = pack_f32(&weights, 2, 9, 4, &mut offsets, &mut columns, &mut values).unwrap();
        assert_eq!(matrix.row_offsets(), &[0, 2, 3]);
        assert_eq!(matrix.column_indices(), &[0, 2, 1]);
        assert_eq!(
            matrix.values(),
            &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn f32_bsr_gemm_matches_dense_transposed_rhs() {
        // W[output, input], sparse in 4-wide input blocks.
        let weights = [
            1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 2.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -2.0,
        ];
        let activations = [
            0.5, 1.0, -2.0, 3.0, 0.25, -1.0, 4.0, -1.0, 2.0, 0.5, 0.25, -3.0, 1.0, 0.75,
        ];
        let mut offsets = [0_u32; 4];
        let mut columns = [0_u32; 6];
        let mut packed = [0.0_f32; 24];
        let matrix = pack_f32(&weights, 3, 7, 4, &mut offsets, &mut columns, &mut packed).unwrap();
        let mut got = [0.0_f32; 6];
        gemm_f32(matrix, &activations, &mut got, 2).unwrap();

        let mut expected = [0.0_f32; 6];
        for batch in 0..2 {
            for output in 0..3 {
                for input in 0..7 {
                    expected[batch * 3 + output] +=
                        activations[batch * 7 + input] * weights[output * 7 + input];
                }
            }
        }
        for (actual, expected) in got.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_bsr_gemm_uses_zero_encodings_as_sparse_blocks() {
        let values = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let weights: [u8; 8] = values.map(crate::fp8::f32_to_e4m3_rne);
        let mut offsets = [0_u32; 2];
        let mut columns = [0_u32; 2];
        let mut packed = [0_u8; 8];
        let matrix = pack_fp8(&weights, 1, 8, 4, &mut offsets, &mut columns, &mut packed).unwrap();
        assert_eq!(matrix.nonzero_blocks(), 1);
        let activations = [0.5, 1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut got = [0.0_f32; 1];
        gemm_e4m3_f32(matrix, &activations, &mut got, 1).unwrap();
        let expected: f32 = weights[..4]
            .iter()
            .zip(&activations[..4])
            .map(|(&weight, &input)| crate::fp8::e4m3_to_f32(weight) * input)
            .sum();
        assert!((got[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn fp8_bsr_gemm_keeps_both_operands_packed() {
        let weights = [
            crate::fp8::f32_to_e4m3_rne(1.0),
            crate::fp8::f32_to_e4m3_rne(-2.0),
            crate::fp8::f32_to_e4m3_rne(0.5),
            0,
            0,
            0,
            0,
            0,
        ];
        let activations = [
            crate::fp8::f32_to_e4m3_rne(0.5),
            crate::fp8::f32_to_e4m3_rne(1.0),
            crate::fp8::f32_to_e4m3_rne(-2.0),
            crate::fp8::f32_to_e4m3_rne(3.0),
            crate::fp8::f32_to_e4m3_rne(4.0),
            crate::fp8::f32_to_e4m3_rne(5.0),
            crate::fp8::f32_to_e4m3_rne(6.0),
            crate::fp8::f32_to_e4m3_rne(7.0),
        ];
        let mut offsets = [0_u32; 2];
        let mut columns = [0_u32; 2];
        let mut packed = [0_u8; 8];
        let matrix = pack_fp8(&weights, 1, 8, 4, &mut offsets, &mut columns, &mut packed).unwrap();
        let mut got = [0.0_f32; 1];
        gemm_e4m3_e4m3_f32(matrix, &activations, &mut got, 1).unwrap();
        let expected: f32 = weights[..4]
            .iter()
            .zip(&activations[..4])
            .map(|(&weight, &input)| {
                crate::fp8::e4m3_to_f32(weight) * crate::fp8::e4m3_to_f32(input)
            })
            .sum();
        assert!((got[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn e5m2_bsr_gemm_keeps_both_operands_packed() {
        let weights = [
            crate::fp8::f32_to_e5m2_rne(1.0),
            crate::fp8::f32_to_e5m2_rne(-2.0),
            crate::fp8::f32_to_e5m2_rne(0.5),
            0,
            0,
            0,
            0,
            0,
        ];
        let activations = [
            crate::fp8::f32_to_e5m2_rne(0.5),
            crate::fp8::f32_to_e5m2_rne(1.0),
            crate::fp8::f32_to_e5m2_rne(-2.0),
            crate::fp8::f32_to_e5m2_rne(3.0),
            crate::fp8::f32_to_e5m2_rne(4.0),
            crate::fp8::f32_to_e5m2_rne(5.0),
            crate::fp8::f32_to_e5m2_rne(6.0),
            crate::fp8::f32_to_e5m2_rne(7.0),
        ];
        let mut offsets = [0_u32; 2];
        let mut columns = [0_u32; 2];
        let mut packed = [0_u8; 8];
        let matrix = pack_fp8(&weights, 1, 8, 4, &mut offsets, &mut columns, &mut packed).unwrap();
        let mut got = [0.0_f32; 1];
        gemm_e5m2_e5m2_f32(matrix, &activations, &mut got, 1).unwrap();
        let expected: f32 = weights[..4]
            .iter()
            .zip(&activations[..4])
            .map(|(&weight, &input)| {
                crate::fp8::e5m2_to_f32(weight) * crate::fp8::e5m2_to_f32(input)
            })
            .sum();
        assert!((got[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn little_endian_rhs_packing_needs_no_aligned_scratch() {
        // Dense B[input, output] = [[1, 0], [0, 2], [3, 0], [0, 0]].
        let source = [1.0_f32, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let bytes: [u8; 32] =
            core::array::from_fn(|index| source[index / 4].to_le_bytes()[index % 4]);
        let profile = profile_f32_le_transposed(&bytes, 2, 4, 2).unwrap();
        assert_eq!(profile.nonzero_blocks, 3);
        let mut offsets = [0_u32; 3];
        let mut columns = [0_u32; 3];
        let mut values = [0.0_f32; 6];
        let packed =
            pack_f32_le_transposed(&bytes, 2, 4, 2, &mut offsets, &mut columns, &mut values)
                .unwrap();
        assert_eq!(packed.row_offsets(), &[0, 2, 3]);
        assert_eq!(packed.column_indices(), &[0, 1, 0]);
        assert_eq!(packed.values(), &[1.0, 0.0, 3.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn malformed_bsr_is_rejected() {
        assert!(matches!(
            BsrMatrix::<f32>::new(1, 4, 4, &[0, 2], &[0], &[0.0; 4]),
            Err(SparseError::InvalidRowOffsets)
        ));
    }
}
