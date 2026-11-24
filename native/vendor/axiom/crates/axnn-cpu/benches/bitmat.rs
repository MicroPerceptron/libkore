//! Q1 bit-matrix micro-benchmarks.
//!
//! Run:  cargo bench -p axnn-cpu --features bench -- bitmat
//!
//! Measures graph algorithm performance at sizes typical for inference
//! compute graphs (64–1024 nodes).

use axnn_cpu::_bench::*;
use divan::counter::BytesCount;

fn main() {
    divan::main();
}

// ── Helpers ─────────────────────────────────────────────────────────────

const CHUNK: usize = 32;

/// Allocate an N×N zero bit matrix, return (data, row_bytes).
fn make_mat(n: usize) -> (Vec<u8>, usize) {
    let row_bytes = ((n + 255) / 256) * CHUNK;
    let data = vec![0u8; n * row_bytes];
    (data, row_bytes)
}

fn set_bit(data: &mut [u8], rb: usize, i: usize, j: usize) {
    data[i * rb + j / 8] |= 1 << (j % 8);
}

/// Build a "wide diamond" DAG: layer 0 → layer 1 → … → layer L.
/// Each layer has `width` nodes, fully connected to the next layer.
/// Total nodes = layers × width.
fn make_layered_dag(layers: usize, width: usize) -> (Vec<u8>, usize, usize) {
    let n = layers * width;
    let (mut data, rb) = make_mat(n);
    for layer in 0..layers - 1 {
        for src in 0..width {
            for dst in 0..width {
                set_bit(
                    &mut data,
                    rb,
                    layer * width + src,
                    (layer + 1) * width + dst,
                );
            }
        }
    }
    (data, rb, n)
}

/// Build a linear chain: 0 → 1 → 2 → … → n-1.
fn make_chain(n: usize) -> (Vec<u8>, usize) {
    let (mut data, rb) = make_mat(n);
    for i in 0..n - 1 {
        set_bit(&mut data, rb, i, i + 1);
    }
    (data, rb)
}

// ── Topo sort benchmarks ────────────────────────────────────────────────

const TOPO_CHAIN_SIZES: &[usize] = &[64, 256, 512, 1024];

#[divan::bench(args = TOPO_CHAIN_SIZES)]
fn bench_topo_sort_chain(bencher: divan::Bencher, n: usize) {
    let (data, rb) = make_chain(n);
    let total_bytes = data.len();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench(|| bitmat_topo_sort(divan::black_box(&data), rb, n));
}

/// Layered DAG: 8 layers × (n/8) width. More realistic for inference graphs:
/// each layer fans out to the next.
#[divan::bench(args = TOPO_CHAIN_SIZES)]
fn bench_topo_sort_layered(bencher: divan::Bencher, n: usize) {
    let width = n / 8;
    let (data, rb, nn) = make_layered_dag(8, width);
    let total_bytes = data.len();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench(|| bitmat_topo_sort(divan::black_box(&data), rb, nn));
}

// ── Row-level operation benchmarks ──────────────────────────────────────

const ROW_SIZES: &[usize] = &[256, 1024];

#[divan::bench(args = ROW_SIZES)]
fn bench_or_row_into(bencher: divan::Bencher, n: usize) {
    let rb = ((n + 255) / 256) * CHUNK;
    let src = vec![0xAAu8; rb];
    bencher
        .counter(BytesCount::new(rb * 2))
        .with_inputs(|| vec![0x55u8; rb])
        .bench_values(|mut dst| {
            bitmat_or_row_into(&mut dst, divan::black_box(&src));
            dst
        });
}

#[divan::bench(args = ROW_SIZES)]
fn bench_row_popcount(bencher: divan::Bencher, n: usize) {
    let rb = ((n + 255) / 256) * CHUNK;
    let row = vec![0xAAu8; rb]; // half the bits set
    bencher
        .counter(BytesCount::new(rb))
        .bench(|| bitmat_row_popcount(divan::black_box(&row)));
}

// ── Column popcount benchmark ───────────────────────────────────────────

#[divan::bench(args = TOPO_CHAIN_SIZES)]
fn bench_col_popcounts(bencher: divan::Bencher, n: usize) {
    let (data, rb) = make_chain(n);
    let total_bytes = data.len();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench(|| bitmat_col_popcounts(divan::black_box(&data), rb, n));
}
