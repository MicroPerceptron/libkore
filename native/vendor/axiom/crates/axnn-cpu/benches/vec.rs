//! Elementwise vec kernel benchmarks.
//!
//! Run:  cargo bench -p axnn-cpu --features bench -- vec

use axnn_cpu::_bench::*;
use divan::counter::BytesCount;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[256, 4096, 65536];

fn make_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 4.0).collect()
}

// ── Unary ops ───────────────────────────────────────────────────────

#[divan::bench(args = SIZES)]
fn bench_relu_f32(bencher: divan::Bencher, n: usize) {
    let src = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            relu_f32(divan::black_box(&src), &mut dst);
            dst
        });
}

#[divan::bench(args = SIZES)]
fn bench_neg_f32(bencher: divan::Bencher, n: usize) {
    let src = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            neg_f32(divan::black_box(&src), &mut dst);
            dst
        });
}

#[divan::bench(args = SIZES)]
fn bench_silu_f32(bencher: divan::Bencher, n: usize) {
    let src = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            silu_f32(divan::black_box(&src), &mut dst);
            dst
        });
}

#[divan::bench(args = SIZES)]
fn bench_gelu_f32(bencher: divan::Bencher, n: usize) {
    let src = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            gelu_f32(divan::black_box(&src), &mut dst);
            dst
        });
}

// ── Binary ops ──────────────────────────────────────────────────────

#[divan::bench(args = SIZES)]
fn bench_add_f32(bencher: divan::Bencher, n: usize) {
    let a = make_f32(n);
    let b = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 3))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            add_f32(divan::black_box(&a), divan::black_box(&b), &mut dst);
            dst
        });
}

#[divan::bench(args = SIZES)]
fn bench_mul_f32(bencher: divan::Bencher, n: usize) {
    let a = make_f32(n);
    let b = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 3))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            mul_f32(divan::black_box(&a), divan::black_box(&b), &mut dst);
            dst
        });
}

#[divan::bench(args = SIZES)]
fn bench_scale_f32(bencher: divan::Bencher, n: usize) {
    let src = make_f32(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            scale_f32(divan::black_box(&src), &mut dst, 2.5);
            dst
        });
}
