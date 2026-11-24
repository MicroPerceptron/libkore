//! Block-sparse GEMM micro-benchmarks.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p axnn-cpu --features bench -- sparse
//! ```
//!
//! The weights are packed outside the timed region, matching compiled graph
//! execution where immutable constants live in the compiled-region cache.
//! Compare `dense_f32_25pct_blocks` with `bsr_f32_25pct_blocks` on each target
//! before changing an adapter's BSR cost policy.

use core::mem::size_of;

use axnn_cpu::{linalg, sparse};
use divan::counter::BytesCount;

const OUTPUTS: usize = 128;
const INPUTS: usize = 1024;
const BATCH: usize = 16;
const BLOCK_K: usize = 32;

fn main() {
    divan::main();
}

struct PackedF32 {
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    values: Vec<f32>,
}

impl PackedF32 {
    fn matrix(&self) -> sparse::BsrMatrix<'_, f32> {
        sparse::BsrMatrix::new(
            self.rows,
            self.columns,
            self.block_k,
            &self.row_offsets,
            &self.column_indices,
            &self.values,
        )
        .unwrap()
    }
}

struct PackedFp8 {
    rows: usize,
    columns: usize,
    block_k: usize,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
    values: Vec<u8>,
}

impl PackedFp8 {
    fn matrix(&self) -> sparse::BsrMatrix<'_, u8> {
        sparse::BsrMatrix::new(
            self.rows,
            self.columns,
            self.block_k,
            &self.row_offsets,
            &self.column_indices,
            &self.values,
        )
        .unwrap()
    }
}

fn dense_weights() -> Vec<f32> {
    let mut weights = vec![0.0; OUTPUTS * INPUTS];
    for row in 0..OUTPUTS {
        for block in 0..INPUTS / BLOCK_K {
            // 25% retained blocks; values are deterministic and nonzero.
            if (row * 5 + block * 3) % 4 != 0 {
                continue;
            }
            for offset in 0..BLOCK_K {
                weights[row * INPUTS + block * BLOCK_K + offset] =
                    ((row + block + offset) % 23) as f32 * 0.03125 + 0.015625;
            }
        }
    }
    weights
}

fn pack_f32(weights: &[f32]) -> PackedF32 {
    let profile = sparse::profile_f32(weights, OUTPUTS, INPUTS, BLOCK_K).unwrap();
    let mut row_offsets = vec![0; OUTPUTS + 1];
    let mut column_indices = vec![0; profile.nonzero_blocks];
    let mut values = vec![0.0; profile.packed_values().unwrap()];
    sparse::pack_f32(
        weights,
        OUTPUTS,
        INPUTS,
        BLOCK_K,
        &mut row_offsets,
        &mut column_indices,
        &mut values,
    )
    .unwrap();
    PackedF32 {
        rows: OUTPUTS,
        columns: INPUTS,
        block_k: BLOCK_K,
        row_offsets,
        column_indices,
        values,
    }
}

fn pack_e4m3(weights: &[f32]) -> PackedFp8 {
    let fp8: Vec<u8> = weights
        .iter()
        .copied()
        .map(axnn_cpu::fp8::f32_to_e4m3_rne)
        .collect();
    let profile = sparse::profile_fp8(&fp8, OUTPUTS, INPUTS, BLOCK_K).unwrap();
    let mut row_offsets = vec![0; OUTPUTS + 1];
    let mut column_indices = vec![0; profile.nonzero_blocks];
    let mut values = vec![0; profile.packed_values().unwrap()];
    sparse::pack_fp8(
        &fp8,
        OUTPUTS,
        INPUTS,
        BLOCK_K,
        &mut row_offsets,
        &mut column_indices,
        &mut values,
    )
    .unwrap();
    PackedFp8 {
        rows: OUTPUTS,
        columns: INPUTS,
        block_k: BLOCK_K,
        row_offsets,
        column_indices,
        values,
    }
}

fn activations() -> Vec<f32> {
    (0..BATCH * INPUTS)
        .map(|index| index as f32 * (1.0 / 2048.0) - 0.5)
        .collect()
}

fn e4m3_activations() -> Vec<u8> {
    activations()
        .into_iter()
        .map(axnn_cpu::fp8::f32_to_e4m3_rne)
        .collect()
}

#[divan::bench]
fn dense_f32_25pct_blocks(bencher: divan::Bencher) {
    let weights = dense_weights();
    // Dense GEMM consumes the conventional right-hand `[input, output]`
    // layout, so transpose the BSR source once outside the measurement.
    let mut rhs = vec![0.0; INPUTS * OUTPUTS];
    for input in 0..INPUTS {
        for output in 0..OUTPUTS {
            rhs[input * OUTPUTS + output] = weights[output * INPUTS + input];
        }
    }
    let activations = activations();
    bencher
        .counter(BytesCount::new(
            weights.len() * size_of::<f32>() + activations.len() * size_of::<f32>(),
        ))
        .with_inputs(|| vec![0.0; BATCH * OUTPUTS])
        .bench_values(|mut output| {
            linalg::gemm_f32(
                divan::black_box(&activations),
                divan::black_box(&rhs),
                &mut output,
                BATCH,
                OUTPUTS,
                INPUTS,
            );
            output
        });
}

#[divan::bench]
fn bsr_f32_25pct_blocks(bencher: divan::Bencher) {
    let packed = pack_f32(&dense_weights());
    let activations = activations();
    bencher
        .counter(BytesCount::new(
            packed.values.len() * size_of::<f32>()
                + packed.column_indices.len() * size_of::<u32>()
                + activations.len() * size_of::<f32>(),
        ))
        .with_inputs(|| vec![0.0; BATCH * OUTPUTS])
        .bench_values(|mut output| {
            sparse::gemm_f32(
                divan::black_box(packed.matrix()),
                divan::black_box(&activations),
                &mut output,
                BATCH,
            )
            .unwrap();
            output
        });
}

#[divan::bench]
fn bsr_e4m3_f32_25pct_blocks(bencher: divan::Bencher) {
    let packed = pack_e4m3(&dense_weights());
    let activations = activations();
    bencher
        .counter(BytesCount::new(
            packed.values.len()
                + packed.column_indices.len() * size_of::<u32>()
                + activations.len() * size_of::<f32>(),
        ))
        .with_inputs(|| vec![0.0; BATCH * OUTPUTS])
        .bench_values(|mut output| {
            sparse::gemm_e4m3_f32(
                divan::black_box(packed.matrix()),
                divan::black_box(&activations),
                &mut output,
                BATCH,
            )
            .unwrap();
            output
        });
}

#[divan::bench]
fn bsr_e4m3_e4m3_25pct_blocks(bencher: divan::Bencher) {
    let packed = pack_e4m3(&dense_weights());
    let activations = e4m3_activations();
    bencher
        .counter(BytesCount::new(
            packed.values.len()
                + packed.column_indices.len() * size_of::<u32>()
                + activations.len(),
        ))
        .with_inputs(|| vec![0.0; BATCH * OUTPUTS])
        .bench_values(|mut output| {
            sparse::gemm_e4m3_e4m3_f32(
                divan::black_box(packed.matrix()),
                divan::black_box(&activations),
                &mut output,
                BATCH,
            )
            .unwrap();
            output
        });
}
