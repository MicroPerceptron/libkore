#![feature(f16)]

//! FP8 (E4M3 / E5M2) kernel micro-benchmarks.
//!
//! Run:  cargo bench -p axnn-cpu --features bench
//! AVX2: cargo bench -p axnn-cpu --features bench,avx2
//!
//! Measures the dispatched tier — F16C on x86 (`vcvtph2ps`), `fcvtl` on
//! AArch64, or the software `core::simd` f16 path elsewhere. Dequant is
//! one byte in / four out (bandwidth-bound); the fused dot folds in an
//! f32 activation per element (compute-bound), which is where the
//! hardware half-precision widen pays off over a decode table.

use axnn_cpu::_bench::*;
use axnn_cpu::uf;
use divan::counter::{BytesCount, ItemsCount};

fn main() {
    println!("signed FP8 ISA: {}", axnn_cpu::build_info::signed_f8_isa());
    println!("unsigned-float ISA: {}", axnn_cpu::build_info::uf_isa());
    divan::main();
}

// Deterministic finite FP8 encodings — throughput depends on instruction
// mix, not data, and the kernels are branchless.
fn fp8_weights(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 127) as u8).collect()
}
fn f32_acts(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 2.0).collect()
}

// ── Dequantize: FP8 → F32 (bandwidth-bound) ──────────────────────────

const CAST_N: usize = 4096;

#[divan::bench]
fn dequant_e4m3(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            dequant_e4m3_f32(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn dequant_e5m2(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            dequant_e5m2_f32(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn dequant_uf8e4m4(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            uf::dequant_uf8e4m4(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn dequant_uf8e5m3(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            uf::dequant_uf8e5m3(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn promote_f16_to_f32(bencher: divan::Bencher) {
    let src: Vec<f16> = (0..CAST_N)
        .map(|i| f16::from_bits((i % 0x7C00) as u16))
        .collect();
    bencher
        .counter(BytesCount::new(src.len() * 2))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            for (out, &value) in dst.iter_mut().zip(divan::black_box(&src)) {
                *out = value as f32;
            }
            dst
        });
}

#[divan::bench]
fn promote_uf16e5m11_to_f32(bencher: divan::Bencher) {
    let src: Vec<u16> = (0..CAST_N).map(|i| (i % 0xF800) as u16).collect();
    bencher
        .counter(BytesCount::new(src.len() * 2))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            uf::promote_uf16e5m11_to_f32(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn promote_uf16e5m11_le_to_f32(bencher: divan::Bencher) {
    let src: Vec<u8> = (0..CAST_N)
        .flat_map(|i| ((i % 0xF800) as u16).to_le_bytes())
        .collect();
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0.0f32; CAST_N])
        .bench_values(|mut dst| {
            uf::promote_uf16e5m11_le_to_f32(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn dequant_e4m3_to_f16(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0u8; CAST_N * 2])
        .bench_values(|mut dst| {
            dequant_e4m3_f16(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn dequant_e5m2_to_f16(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0u8; CAST_N * 2])
        .bench_values(|mut dst| {
            dequant_e5m2_f16(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn promote_uf8e4m4_to_f16(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0u8; CAST_N * 2])
        .bench_values(|mut dst| {
            uf::promote_uf8e4m4_to_f16(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

#[divan::bench]
fn promote_uf8e5m3_to_f16(bencher: divan::Bencher) {
    let src = fp8_weights(CAST_N);
    bencher
        .counter(BytesCount::new(src.len()))
        .with_inputs(|| vec![0u8; CAST_N * 2])
        .bench_values(|mut dst| {
            uf::promote_uf8e5m3_to_f16(divan::black_box(&src), &mut dst, CAST_N);
            dst
        });
}

// ── Fused dot: Σ decode(w) · x (compute-bound) ───────────────────────

const DOT_SIZES: &[usize] = &[256, 1024, 4096];

#[divan::bench(args = DOT_SIZES)]
fn dot_e4m3_f32(bencher: divan::Bencher, n: usize) {
    let w = fp8_weights(n);
    let x = f32_acts(n);
    bencher
        .counter(BytesCount::new(w.len() + x.len() * 4))
        .bench(|| vec_dot_e4m3_f32(divan::black_box(&w), divan::black_box(&x), n));
}

#[divan::bench(args = DOT_SIZES)]
fn dot_e5m2_f32(bencher: divan::Bencher, n: usize) {
    let w = fp8_weights(n);
    let x = f32_acts(n);
    bencher
        .counter(BytesCount::new(w.len() + x.len() * 4))
        .bench(|| vec_dot_e5m2_f32(divan::black_box(&w), divan::black_box(&x), n));
}

// ── Mixed-precision MAC: FP8 · FP8 + {F16, BF16} → F32 ────────────────
//
// The addend `c` is a 16-bit (F16 / BF16) buffer — deterministic finite
// patterns; throughput is data-independent (branchless kernels).

fn f16_bytes(n: usize) -> Vec<u8> {
    (0..2 * n).map(|i| ((i * 3) % 200) as u8).collect()
}

const MAC_N: usize = 4096;

// Elementwise MAD — F16 vs BF16 addend (same kernel, different widen).
#[divan::bench]
fn mad_e4m3_f16(bencher: divan::Bencher) {
    let a = fp8_weights(MAC_N);
    let b = fp8_weights(MAC_N);
    let c = f16_bytes(MAC_N);
    bencher
        .counter(BytesCount::new(a.len() + b.len() + c.len()))
        .with_inputs(|| vec![0.0f32; MAC_N])
        .bench_values(|mut d| {
            mad_e4m3_e4m3_f16(
                divan::black_box(&a),
                divan::black_box(&b),
                divan::black_box(&c),
                &mut d,
                MAC_N,
            );
            d
        });
}

#[divan::bench]
fn mad_e4m3_bf16(bencher: divan::Bencher) {
    let a = fp8_weights(MAC_N);
    let b = fp8_weights(MAC_N);
    let c = f16_bytes(MAC_N);
    bencher
        .counter(BytesCount::new(a.len() + b.len() + c.len()))
        .with_inputs(|| vec![0.0f32; MAC_N])
        .bench_values(|mut d| {
            mad_e4m3_e4m3_bf16(
                divan::black_box(&a),
                divan::black_box(&b),
                divan::black_box(&c),
                &mut d,
                MAC_N,
            );
            d
        });
}

// Elementwise FP8 × F32.
#[divan::bench]
fn mul_e4m3_f32_(bencher: divan::Bencher) {
    let a = fp8_weights(MAC_N);
    let b = f32_acts(MAC_N);
    bencher
        .counter(BytesCount::new(a.len() + b.len() * 4))
        .with_inputs(|| vec![0.0f32; MAC_N])
        .bench_values(|mut d| {
            mul_e4m3_f32(divan::black_box(&a), divan::black_box(&b), &mut d, MAC_N);
            d
        });
}

// Reduction dot — FP8 × FP8.
#[divan::bench(args = DOT_SIZES)]
fn dot_e4m3_e4m3_(bencher: divan::Bencher, n: usize) {
    let a = fp8_weights(n);
    let b = fp8_weights(n);
    bencher
        .counter(BytesCount::new(a.len() + b.len()))
        .bench(|| dot_e4m3_e4m3(divan::black_box(&a), divan::black_box(&b), n));
}

// GEMM with F16 vs BF16 bias — (m=64, n=32, k=4096).
#[divan::bench(args = [(64usize, 32usize, 4096usize)])]
fn gemm_e4m3_f16(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let a = fp8_weights(m * k);
    let b = fp8_weights(n * k);
    let c = f16_bytes(m * n);
    bencher
        .counter(BytesCount::new(a.len() + b.len() + c.len()))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut d| {
            gemm_e4m3_e4m3_f16(
                divan::black_box(&a),
                divan::black_box(&b),
                divan::black_box(&c),
                &mut d,
                m,
                n,
                k,
            );
            d
        });
}

#[divan::bench(args = [(64usize, 32usize, 4096usize)])]
fn gemm_e4m3_bf16(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let a = fp8_weights(m * k);
    let b = fp8_weights(n * k);
    let c = f16_bytes(m * n);
    bencher
        .counter(BytesCount::new(a.len() + b.len() + c.len()))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut d| {
            gemm_e4m3_e4m3_bf16(
                divan::black_box(&a),
                divan::black_box(&b),
                divan::black_box(&c),
                &mut d,
                m,
                n,
                k,
            );
            d
        });
}

// ── Mixed GEMM: FP8 weight × F32 activation → F32 (single-widen) ──────
//
// Same (m,n,k) as the double-FP8 GEMM above, so wall-time is the honest
// comparator (identical MAC count). Only the weight widens; the F32
// activation streams. No bias — the reduction accumulates straight to F32.
#[divan::bench(args = [(64usize, 32usize, 4096usize)])]
fn gemm_e4m3_f32_(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let w = fp8_weights(m * k);
    let x = f32_acts(n * k);
    bencher
        .counter(BytesCount::new(w.len() + x.len() * 4))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            gemm_e4m3_f32(divan::black_box(&w), divan::black_box(&x), &mut c, m, n, k);
            c
        });
}

#[divan::bench(args = [(64usize, 32usize, 4096usize)])]
fn gemm_e5m2_f32_(bencher: divan::Bencher, &(m, n, k): &(usize, usize, usize)) {
    let w = fp8_weights(m * k);
    let x = f32_acts(n * k);
    bencher
        .counter(BytesCount::new(w.len() + x.len() * 4))
        .with_inputs(|| vec![0.0f32; m * n])
        .bench_values(|mut c| {
            gemm_e5m2_f32(divan::black_box(&w), divan::black_box(&x), &mut c, m, n, k);
            c
        });
}

// ── SH → RGB eval: FP8 coeffs × f32 basis (3DGS "Kernel A") ──────────
//
// N Gaussians, degree 3 (16 coeffs × 3 channels). Reports Gaussians/s.
// FP8 coeffs are ~4× smaller than f32 (48 vs 192 B/Gaussian), so at this
// scale (coeffs spill cache) the footprint shows up as shading throughput.
const SH_N: usize = 1 << 17;
const SH_DEG: usize = 3;
const SH_NC: usize = 16;

// Deterministic unit directions (branchless; normalized).
fn unit_dirs(n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n * 3];
    for i in 0..n {
        let mut x = ((i * 13 % 200) as f32) / 100.0 - 1.0;
        let mut y = ((i * 29 % 200) as f32) / 100.0 - 1.0;
        let mut z = ((i * 47 % 200) as f32) / 100.0 - 1.0;
        let inv = 1.0 / (x * x + y * y + z * z + 1e-6).sqrt();
        x *= inv;
        y *= inv;
        z *= inv;
        v[i * 3] = x;
        v[i * 3 + 1] = y;
        v[i * 3 + 2] = z;
    }
    v
}

#[divan::bench]
fn sh_eval_e4m3_(bencher: divan::Bencher) {
    let dirs = unit_dirs(SH_N);
    let coeffs = fp8_weights(SH_N * 3 * SH_NC);
    bencher
        .counter(ItemsCount::new(SH_N))
        .with_inputs(|| vec![0.0f32; SH_N * 3])
        .bench_values(|mut out| {
            sh_eval_e4m3(
                divan::black_box(&dirs),
                divan::black_box(&coeffs),
                &mut out,
                SH_N,
                SH_DEG,
            );
            out
        });
}

#[divan::bench]
fn sh_eval_e5m2_(bencher: divan::Bencher) {
    let dirs = unit_dirs(SH_N);
    let coeffs = fp8_weights(SH_N * 3 * SH_NC);
    bencher
        .counter(ItemsCount::new(SH_N))
        .with_inputs(|| vec![0.0f32; SH_N * 3])
        .bench_values(|mut out| {
            sh_eval_e5m2(
                divan::black_box(&dirs),
                divan::black_box(&coeffs),
                &mut out,
                SH_N,
                SH_DEG,
            );
            out
        });
}

#[divan::bench]
fn sh_eval_f32_(bencher: divan::Bencher) {
    let dirs = unit_dirs(SH_N);
    let coeffs: Vec<f32> = (0..SH_N * 3 * SH_NC)
        .map(|i| (i % 100) as f32 * 0.01 - 0.5)
        .collect();
    bencher
        .counter(ItemsCount::new(SH_N))
        .with_inputs(|| vec![0.0f32; SH_N * 3])
        .bench_values(|mut out| {
            sh_eval_f32(
                divan::black_box(&dirs),
                divan::black_box(&coeffs),
                &mut out,
                SH_N,
                SH_DEG,
            );
            out
        });
}

#[divan::bench]
fn sh_eval_f16_(bencher: divan::Bencher) {
    let dirs = unit_dirs(SH_N);
    let coeffs: Vec<f16> = (0..SH_N * 3 * SH_NC)
        .map(|i| ((i % 100) as f32 * 0.01 - 0.5) as f16)
        .collect();
    bencher
        .counter(ItemsCount::new(SH_N))
        .with_inputs(|| vec![0.0f32; SH_N * 3])
        .bench_values(|mut out| {
            sh_eval_f16(
                divan::black_box(&dirs),
                divan::black_box(&coeffs),
                &mut out,
                SH_N,
                SH_DEG,
            );
            out
        });
}
