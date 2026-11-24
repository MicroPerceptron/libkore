//! Quantized kernel micro-benchmarks.
//!
//! Run:  cargo bench -p axnn-cpu --features bench
//! Filter: cargo bench -p axnn-cpu --features bench -- vec_dot
//! NEON:  cargo bench -p axnn-cpu --features bench,neon
//! AVX2:  cargo bench -p axnn-cpu --features bench,avx2
//!
//! The `active` benchmarks measure the dispatched tier (NEON, AVX2, etc.).
//! The `baseline` benchmarks always measure the portable scalar fallback.
//! Compare them side-by-side to see the SIMD uplift.

use axnn_cpu::_bench::baseline;
use axnn_cpu::_bench::*;
use divan::counter::BytesCount;

fn main() {
    divan::main();
}

// ── Input factories ──────────────────────────────────────────────────
//
// Pre-quantize buffers so allocation is excluded from measurement.
// Values are deterministic (not random) — sufficient for throughput
// benchmarks where we care about instruction mix, not data-dependent
// branch prediction.

fn make_q8_pair(n: usize) -> (Vec<u8>, Vec<u8>) {
    let src: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 2.0).collect();
    let nb = n / Q8_BLOCK_SIZE;
    let mut x = vec![0u8; nb * Q8_BLOCK_BYTES];
    let mut y = vec![0u8; nb * Q8_BLOCK_BYTES];
    quantize_q8(&src, &mut x);
    // Shift y slightly so it's not identical to x.
    let src_y: Vec<f32> = (0..n)
        .map(|i| ((i + 7) as f32 / n as f32 - 0.5) * 1.5)
        .collect();
    quantize_q8(&src_y, &mut y);
    (x, y)
}

fn make_q4_q8_pair(n: usize) -> (Vec<u8>, Vec<u8>) {
    let w: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 2.0).collect();
    let a: Vec<f32> = (0..n)
        .map(|i| ((i + 3) as f32 / n as f32 - 0.5) * 1.5)
        .collect();
    let nb4 = n / Q4_BLOCK_SIZE;
    let nb8 = n / Q8_BLOCK_SIZE;
    let mut qw = vec![0u8; nb4 * Q4_BLOCK_BYTES];
    let mut qa = vec![0u8; nb8 * Q8_BLOCK_BYTES];
    quantize_q4(&w, &mut qw);
    quantize_q8(&a, &mut qa);
    (qw, qa)
}

fn make_q6_q8_pair(n: usize) -> (Vec<u8>, Vec<u8>) {
    let w: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 2.0).collect();
    let a: Vec<f32> = (0..n)
        .map(|i| ((i + 5) as f32 / n as f32 - 0.5) * 1.5)
        .collect();
    let nb6 = n / Q6_BLOCK_SIZE;
    let nb8 = n / Q8_BLOCK_SIZE;
    let mut qw = vec![0u8; nb6 * Q6_BLOCK_BYTES];
    let mut qa = vec![0u8; nb8 * Q8_BLOCK_BYTES];
    quantize_q6(&w, &mut qw);
    quantize_q8(&a, &mut qa);
    (qw, qa)
}

// ── Vec dot benchmarks ───────────────────────────────────────────────

const DOT_SIZES: &[usize] = &[256, 1024, 4096];

#[divan::bench(args = DOT_SIZES)]
fn bench_vec_dot_q8_q8(bencher: divan::Bencher, n: usize) {
    let (x, y) = make_q8_pair(n);
    bencher
        .counter(BytesCount::new(x.len() + y.len()))
        .bench(|| vec_dot_q8_q8(divan::black_box(&x), divan::black_box(&y), n));
}

#[divan::bench(args = DOT_SIZES)]
fn bench_vec_dot_q4_q8(bencher: divan::Bencher, n: usize) {
    let (w, a) = make_q4_q8_pair(n);
    bencher
        .counter(BytesCount::new(w.len() + a.len()))
        .bench(|| vec_dot_q4_q8(divan::black_box(&w), divan::black_box(&a), n));
}

#[divan::bench(args = DOT_SIZES)]
fn bench_vec_dot_q6_q8(bencher: divan::Bencher, n: usize) {
    let (w, a) = make_q6_q8_pair(n);
    bencher
        .counter(BytesCount::new(w.len() + a.len()))
        .bench(|| vec_dot_q6_q8(divan::black_box(&w), divan::black_box(&a), n));
}

// ── Vec dot baselines (portable scalar) ─────────────────────────────

#[divan::bench(args = DOT_SIZES)]
fn baseline_vec_dot_q8_q8(bencher: divan::Bencher, n: usize) {
    let (x, y) = make_q8_pair(n);
    bencher
        .counter(BytesCount::new(x.len() + y.len()))
        .bench(|| baseline::vec_dot_q8_q8(divan::black_box(&x), divan::black_box(&y), n));
}

#[divan::bench(args = DOT_SIZES)]
fn baseline_vec_dot_q4_q8(bencher: divan::Bencher, n: usize) {
    let (w, a) = make_q4_q8_pair(n);
    bencher
        .counter(BytesCount::new(w.len() + a.len()))
        .bench(|| baseline::vec_dot_q4_q8(divan::black_box(&w), divan::black_box(&a), n));
}

#[divan::bench(args = DOT_SIZES)]
fn baseline_vec_dot_q6_q8(bencher: divan::Bencher, n: usize) {
    let (w, a) = make_q6_q8_pair(n);
    bencher
        .counter(BytesCount::new(w.len() + a.len()))
        .bench(|| baseline::vec_dot_q6_q8(divan::black_box(&w), divan::black_box(&a), n));
}

// ── GEMM benchmarks ──────────────────────────────────────────────────
//
// Typical inference shapes: (m=output_features, n=batch, k=inner_dim).
// We benchmark single-batch (n=1) and batched (n=32).

#[divan::bench(args = [(64, 1, 4096), (64, 32, 4096)])]
fn bench_gemm_q4_q8(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let nb4 = k / Q4_BLOCK_SIZE;
    let nb8 = k / Q8_BLOCK_SIZE;

    // Weight matrix: m rows of k elements in Q4.
    let w_row: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32 - 0.5) * 2.0).collect();
    let mut qw = vec![0u8; m * nb4 * Q4_BLOCK_BYTES];
    for row in 0..m {
        let off = row * nb4 * Q4_BLOCK_BYTES;
        quantize_q4(&w_row, &mut qw[off..off + nb4 * Q4_BLOCK_BYTES]);
    }

    // Activation matrix: n rows of k elements in Q8.
    let a_row: Vec<f32> = (0..k)
        .map(|i| ((i + 3) as f32 / k as f32 - 0.5) * 1.5)
        .collect();
    let mut qa = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];
    for row in 0..n {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&a_row, &mut qa[off..off + nb8 * Q8_BLOCK_BYTES]);
    }

    let total_bytes = qw.len() + qa.len();
    bencher
        .counter(BytesCount::new(total_bytes))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            gemm_q4_q8(
                divan::black_box(&qw),
                divan::black_box(&qa),
                &mut c,
                m,
                n,
                k,
            );
            c
        });
}

#[divan::bench(args = [(64, 1, 4096), (64, 32, 4096)])]
fn bench_gemm_q8_q8(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let nb8 = k / Q8_BLOCK_SIZE;
    let w_row: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32 - 0.5) * 2.0).collect();
    let mut qw = vec![0u8; m * nb8 * Q8_BLOCK_BYTES];
    for row in 0..m {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&w_row, &mut qw[off..off + nb8 * Q8_BLOCK_BYTES]);
    }
    let a_row: Vec<f32> = (0..k)
        .map(|i| ((i + 3) as f32 / k as f32 - 0.5) * 1.5)
        .collect();
    let mut qa = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];
    for row in 0..n {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&a_row, &mut qa[off..off + nb8 * Q8_BLOCK_BYTES]);
    }
    bencher
        .counter(BytesCount::new(qw.len() + qa.len()))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            gemm_q8_q8(
                divan::black_box(&qw),
                divan::black_box(&qa),
                &mut c,
                m,
                n,
                k,
            );
            c
        });
}

#[divan::bench(args = [(64, 32, 4096)])]
fn baseline_gemm_q8_q8(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let nb8 = k / Q8_BLOCK_SIZE;
    let w_row: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32 - 0.5) * 2.0).collect();
    let mut qw = vec![0u8; m * nb8 * Q8_BLOCK_BYTES];
    for row in 0..m {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&w_row, &mut qw[off..off + nb8 * Q8_BLOCK_BYTES]);
    }
    let a_row: Vec<f32> = (0..k)
        .map(|i| ((i + 3) as f32 / k as f32 - 0.5) * 1.5)
        .collect();
    let mut qa = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];
    for row in 0..n {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&a_row, &mut qa[off..off + nb8 * Q8_BLOCK_BYTES]);
    }
    bencher
        .counter(BytesCount::new(qw.len() + qa.len()))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            baseline::gemm_q8_q8(
                divan::black_box(&qw),
                divan::black_box(&qa),
                &mut c,
                m,
                n,
                k,
            );
            c
        });
}

#[divan::bench(args = [(64, 32, 4096)])]
fn baseline_gemm_q4_q8(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let nb4 = k / Q4_BLOCK_SIZE;
    let nb8 = k / Q8_BLOCK_SIZE;
    let w_row: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32 - 0.5) * 2.0).collect();
    let mut qw = vec![0u8; m * nb4 * Q4_BLOCK_BYTES];
    for row in 0..m {
        let off = row * nb4 * Q4_BLOCK_BYTES;
        quantize_q4(&w_row, &mut qw[off..off + nb4 * Q4_BLOCK_BYTES]);
    }
    let a_row: Vec<f32> = (0..k)
        .map(|i| ((i + 3) as f32 / k as f32 - 0.5) * 1.5)
        .collect();
    let mut qa = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];
    for row in 0..n {
        let off = row * nb8 * Q8_BLOCK_BYTES;
        quantize_q8(&a_row, &mut qa[off..off + nb8 * Q8_BLOCK_BYTES]);
    }
    bencher
        .counter(BytesCount::new(qw.len() + qa.len()))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            baseline::gemm_q4_q8(
                divan::black_box(&qw),
                divan::black_box(&qa),
                &mut c,
                m,
                n,
                k,
            );
            c
        });
}

// ── Cast benchmarks ──────────────────────────────────────────────────

const CAST_N: usize = 4096;

#[divan::bench]
fn bench_quant_f32_q8(bencher: divan::Bencher) {
    let src: Vec<f32> = (0..CAST_N)
        .map(|i| (i as f32 / CAST_N as f32 - 0.5) * 4.0)
        .collect();
    let nb = CAST_N / Q8_BLOCK_SIZE;
    bencher
        .counter(BytesCount::new(CAST_N * 4))
        .with_inputs(|| vec![0u8; nb * Q8_BLOCK_BYTES])
        .bench_values(|mut dst| {
            quant_f32_q8(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn bench_dequant_q8_f32(bencher: divan::Bencher) {
    let src_f32: Vec<f32> = (0..CAST_N)
        .map(|i| (i as f32 / CAST_N as f32 - 0.5) * 4.0)
        .collect();
    let nb = CAST_N / Q8_BLOCK_SIZE;
    let mut q8 = vec![0u8; nb * Q8_BLOCK_BYTES];
    quantize_q8(&src_f32, &mut q8);
    bencher
        .counter(BytesCount::new(q8.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            dequant_q8_f32(divan::black_box(&q8), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn bench_quant_f32_q4(bencher: divan::Bencher) {
    let src: Vec<f32> = (0..CAST_N)
        .map(|i| (i as f32 / CAST_N as f32 - 0.5) * 4.0)
        .collect();
    let nb = CAST_N / Q4_BLOCK_SIZE;
    bencher
        .counter(BytesCount::new(CAST_N * 4))
        .with_inputs(|| vec![0u8; nb * Q4_BLOCK_BYTES])
        .bench_values(|mut dst| {
            quant_f32_q4(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn bench_dequant_q4_f32(bencher: divan::Bencher) {
    let src_f32: Vec<f32> = (0..CAST_N)
        .map(|i| (i as f32 / CAST_N as f32 - 0.5) * 4.0)
        .collect();
    let nb = CAST_N / Q4_BLOCK_SIZE;
    let mut q4 = vec![0u8; nb * Q4_BLOCK_BYTES];
    quantize_q4(&src_f32, &mut q4);
    bencher
        .counter(BytesCount::new(q4.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            dequant_q4_f32(divan::black_box(&q4), &mut dst, CAST_N);
            dst
        });
}
