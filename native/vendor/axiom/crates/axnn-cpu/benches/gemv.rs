#![feature(portable_simd)]
//! GEMV bandwidth: FP8 (E4M3) and BF16 weight rows against one F32
//! activation row, the shape of every decode projection.
//!
//! Run: cargo bench -p axnn-cpu --features bench --bench gemv [-- threads,..]
//!
//! Two regimes per kernel and shape:
//! * **cached**: one small matrix, reread — the kernel's compute ceiling
//!   per core, with no memory traffic in the way;
//! * **streamed**: consecutive windows of a region far larger than the
//!   caches, claimed by every thread in turn — what decode sees, where
//!   the sum over threads meets the memory bus.
//!
//! Each kernel runs against the row-at-a-time dot it replaces. Reports
//! weight bytes per second, summed over threads.

use axnn_cpu::_bench::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Each measurement's length, unless `GEMV_RUN_MS` says otherwise (a
/// long run shows the sustained clock rather than the boost).
fn run() -> Duration {
    let ms = std::env::var("GEMV_RUN_MS")
        .ok()
        .and_then(|v| v.parse().ok());
    Duration::from_millis(ms.unwrap_or(600))
}
/// The streamed region: well past any last-level cache.
const REGION: usize = 512 << 20;

type Kernel = fn(&[u8], &[f32], &mut [f32]);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn measure_axpy(
    weights: &[u8],
    k: usize,
    threads: usize,
    kernel: fn(&[u8], f32, &mut [f32]),
) -> f64 {
    let window = 64 * k;
    let windows = weights.len() / window;
    let next = AtomicUsize::new(0);
    let run = run();
    let bytes: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut out = vec![0.0_f32; k];
                    let mut done = 0;
                    let start = Instant::now();
                    while start.elapsed() < run {
                        let at = next.fetch_add(1, Ordering::Relaxed) % windows;
                        for row in weights[at * window..(at + 1) * window].chunks_exact(k) {
                            kernel(row, 0.125, &mut out);
                        }
                        std::hint::black_box(&mut out);
                        done += window;
                    }
                    (done, start.elapsed())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                let (bytes, elapsed) = w.join().unwrap();
                (bytes as f64 * run.as_secs_f64() / elapsed.as_secs_f64()) as usize
            })
            .sum()
    });
    bytes as f64 / run.as_secs_f64() / 1e9
}

fn rowwise_e4m3(w: &[u8], x: &[f32], out: &mut [f32]) {
    for (row, y) in w.chunks_exact(x.len()).zip(out) {
        *y = vec_dot_e4m3_f32(row, x, x.len());
    }
}

/// Only reads the weights: the memory bus's ceiling for these kernels.
fn read(w: &[u8], _: &[f32], out: &mut [f32]) {
    let words = w.as_chunks::<32>().0;
    let or = words.iter().fold([0u8; 32], |mut acc, word| {
        for (a, b) in acc.iter_mut().zip(word) {
            *a |= b;
        }
        acc
    });
    out[0] = f32::from(or.iter().fold(0, |a, &b| a | b));
}

/// NVFP4 rows as the repacked file stores them: `k / 2` bytes of packed
/// E2M1 weights followed by the row's `k / 16` E4M3 block scales.
fn rowwise_nvfp4(w: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    let row_bytes = k / 2 + k / 16;
    for (row, y) in w.chunks_exact(row_bytes).zip(out) {
        let (packed, scales) = row.split_at(k / 2);
        *y = vec_dot_nvfp4_f32(packed, scales, x, k);
    }
}

/// The same rows through the NEON tier.
#[cfg(target_arch = "aarch64")]
fn rowwise_nvfp4_neon(w: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    let row_bytes = k / 2 + k / 16;
    for (row, y) in w.chunks_exact(row_bytes).zip(out) {
        let (packed, scales) = row.split_at(k / 2);
        *y = vec_dot_nvfp4_neon_f32(packed, scales, x, k);
    }
}

/// The same rows through the AVX2 tier, which blocks four rows at a time.
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
fn rowwise_nvfp4_avx2(w: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    let (row_bytes, row_scales) = (k / 2, k / 16);
    // The streamed region interleaves each row's weights and scales; the
    // kernel wants them apart, so walk one row at a time.
    for (row, y) in w.chunks_exact(row_bytes + row_scales).zip(out) {
        let (packed, scales) = row.split_at(row_bytes);
        gemv_nvfp4_avx2_f32(packed, scales, x, core::array::from_mut(y));
    }
}

/// The same packed rows through the AVX-512 float tier.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn rowwise_nvfp4_avx512(w: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    let (row_bytes, row_scales) = (k / 2, k / 16);
    for (row, y) in w.chunks_exact(row_bytes + row_scales).zip(out) {
        let (packed, scales) = row.split_at(row_bytes);
        gemv_nvfp4_avx512_f32(packed, scales, x, core::array::from_mut(y));
    }
}

/// The same rows with activations quantized to U8, through AVX-VNNI. The
/// quantization runs once per window of rows, as it would once per token.
#[cfg(all(target_arch = "x86_64", feature = "nvfp4_vnni"))]
fn rowwise_nvfp4_vnni(w: &[u8], x: &[f32], out: &mut [f32]) {
    rowwise_nvfp4_vnni_with(w, x, out, axnn_cpu::_bench::nvfp4_q8::nvfp4_q8_f32)
}

#[cfg(all(target_arch = "x86_64", feature = "avx_vnni"))]
fn rowwise_nvfp4_vnni_256(w: &[u8], x: &[f32], out: &mut [f32]) {
    rowwise_nvfp4_vnni_with(w, x, out, axnn_cpu::_bench::nvfp4_q8_avx_vnni::nvfp4_q8_f32)
}

#[cfg(all(target_arch = "x86_64", feature = "nvfp4_vnni"))]
fn rowwise_nvfp4_vnni_with(
    w: &[u8],
    x: &[f32],
    out: &mut [f32],
    kernel: impl Fn(&[u8], &[u8], &axnn_cpu::_bench::nvfp4_q8::Quantized<'_>, &mut [f32]),
) {
    use axnn_cpu::_bench::nvfp4_q8::*;
    let k = x.len();
    let (row_bytes, row_scales) = (k / 2, k / 16);
    let mut biased = [0u8; 4096];
    let mut scales = [0.0_f32; 256];
    quantize(x, &mut biased[..k], &mut scales[..row_scales]);
    let quantized = Quantized {
        biased: &biased[..k],
        scales: &scales[..row_scales],
    };
    for (row, y) in w.chunks_exact(row_bytes + row_scales).zip(out) {
        let (packed, block_scales) = row.split_at(row_bytes);
        kernel(packed, block_scales, &quantized, core::array::from_mut(y));
    }
}

/// The same rows through the table-free decoder.
fn rowwise_nvfp4_bits(w: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    let row_bytes = k / 2 + k / 16;
    for (row, y) in w.chunks_exact(row_bytes).zip(out) {
        let (packed, scales) = row.split_at(k / 2);
        *y = vec_dot_nvfp4_bits_f32(packed, scales, x, k);
    }
}

/// Weight elements per second when `T` distinct tokens share one widen.
/// Each token gets its own activation row, so nothing the kernel does for
/// one token can be reused for another except the dequantized weights.
fn fp4_tokens<const T: usize, const BITS: bool>(weights: &[u8], k: usize, threads: usize) -> f64 {
    let row_bytes = k / 2 + k / 16;
    let rows = 64;
    let window = rows * row_bytes;
    let windows = weights.len() / window;
    let next = AtomicUsize::new(0);
    let xs: Vec<Vec<f32>> = (0..T)
        .map(|t| {
            (0..k)
                .map(|i| ((i * (37 + 5 * t) % 101) as f32 - 50.0) / 50.0)
                .collect()
        })
        .collect();
    let run = run();
    let elems: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (xs, next) = (&xs, &next);
                scope.spawn(move || {
                    let mut refs: [&[f32]; T] = [&[]; T];
                    for (r, x) in refs.iter_mut().zip(xs) {
                        *r = x.as_slice();
                    }
                    let mut sums = [0.0_f32; T];
                    let mut done = 0;
                    let start = Instant::now();
                    while start.elapsed() < run {
                        let at = next.fetch_add(1, Ordering::Relaxed) % windows;
                        let w = &weights[at * window..(at + 1) * window];
                        for row in w.chunks_exact(row_bytes) {
                            let (packed, scales) = row.split_at(k / 2);
                            nvfp4_tokens::<T, BITS>(packed, scales, &refs, k, &mut sums);
                            std::hint::black_box(&mut sums);
                        }
                        done += rows * k;
                    }
                    (done, start.elapsed())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                let (elems, elapsed) = w.join().unwrap();
                (elems as f64 * run.as_secs_f64() / elapsed.as_secs_f64()) as usize
            })
            .sum()
    });
    elems as f64 / run.as_secs_f64() / 1e9
}

fn rowwise_bf16(w: &[u8], x: &[f32], out: &mut [f32]) {
    for (row, y) in w.chunks_exact(2 * x.len()).zip(out) {
        *y = vec_dot_bf16_f32(row, x);
    }
}

/// Multiply-accumulates per second on data that never leaves L1: the
/// ceiling a decode-free kernel of this shape could reach. Eight chains,
/// as the row-blocked kernels run.
fn fma_peak(k: usize, threads: usize) -> f64 {
    let run = run();
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let done: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let x = &x;
                scope.spawn(move || {
                    use std::simd::{Simd, StdFloat};
                    // Eight independent vectors in flight, so the measure
                    // is FMA throughput and not its latency.
                    let mut acc = [Simd::<f32, 8>::splat(1.0); 8];
                    let mut done = 0;
                    let start = Instant::now();
                    while start.elapsed() < run {
                        for window in x.chunks_exact(64) {
                            for (acc, lanes) in acc.iter_mut().zip(window.chunks_exact(8)) {
                                let v = Simd::<f32, 8>::from_slice(lanes);
                                *acc = acc.mul_add(v, v);
                            }
                        }
                        std::hint::black_box(&mut acc);
                        done += k / 64 * 64;
                    }
                    (done, start.elapsed())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                let (done, elapsed) = w.join().unwrap();
                (done as f64 * run.as_secs_f64() / elapsed.as_secs_f64()) as usize
            })
            .sum()
    });
    done as f64 / run.as_secs_f64() / 1e9
}

/// Weight bytes per second over `threads` threads, each running `kernel`
/// on `rows × k` windows of `weights`: the same window when `stream` is
/// false, successive shared windows otherwise.
fn measure(
    kernel: Kernel,
    weights: &[u8],
    row_bytes: usize,
    rows: usize,
    k: usize,
    threads: usize,
    stream: bool,
) -> f64 {
    let window = rows * row_bytes;
    let windows = weights.len() / window;
    let next = AtomicUsize::new(0);
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let run = run();
    let bytes: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                let (x, next) = (&x, &next);
                scope.spawn(move || {
                    let mut out = vec![0.0_f32; rows];
                    let mut done = 0;
                    let start = Instant::now();
                    let own = t % windows;
                    while start.elapsed() < run {
                        let at = if stream {
                            next.fetch_add(1, Ordering::Relaxed) % windows
                        } else {
                            own
                        };
                        kernel(&weights[at * window..(at + 1) * window], x, &mut out);
                        std::hint::black_box(&mut out);
                        done += window;
                    }
                    (done, start.elapsed())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                let (bytes, elapsed) = w.join().unwrap();
                (bytes as f64 * run.as_secs_f64() / elapsed.as_secs_f64()) as usize
            })
            .sum()
    });
    bytes as f64 / run.as_secs_f64() / 1e9
}

/// Tokens per multi-token product in the chunked comparison.
const TOKENS: usize = 5;
/// Weight bytes a chunk holds (as `neural.rs` walks them).
const CHUNK_BYTES: usize = 128 << 10;

/// `TOKENS` activation rows against the same weights: every token's full
/// pass in turn (`chunked` false), or each cache-sized chunk of rows run
/// for every token before the next chunk. Reports weight bytes per second
/// per token-equivalent, so the two are comparable with one-token runs.
fn tokens_measure(
    kernel: Kernel,
    weights: &[u8],
    row_bytes: usize,
    rows: usize,
    k: usize,
    threads: usize,
    chunked: bool,
) -> f64 {
    let window = rows * row_bytes;
    let windows = weights.len() / window;
    let next = AtomicUsize::new(0);
    let xs: Vec<f32> = (0..TOKENS * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let run = run();
    let bytes: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (xs, next) = (&xs, &next);
                scope.spawn(move || {
                    let mut out = vec![0.0_f32; TOKENS * rows];
                    let mut done = 0;
                    let start = Instant::now();
                    let chunk = (CHUNK_BYTES / row_bytes).max(1);
                    while start.elapsed() < run {
                        let at = next.fetch_add(1, Ordering::Relaxed) % windows;
                        let w = &weights[at * window..(at + 1) * window];
                        if chunked {
                            for first in (0..rows).step_by(chunk) {
                                let len = chunk.min(rows - first);
                                for (t, x) in xs.chunks_exact(k).enumerate() {
                                    let y = &mut out[t * rows + first..t * rows + first + len];
                                    kernel(&w[first * row_bytes..(first + len) * row_bytes], x, y);
                                }
                            }
                        } else {
                            for (x, y) in xs.chunks_exact(k).zip(out.chunks_exact_mut(rows)) {
                                kernel(w, x, y);
                            }
                        }
                        std::hint::black_box(&mut out);
                        done += window * TOKENS;
                    }
                    (done, start.elapsed())
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                let (bytes, elapsed) = w.join().unwrap();
                (bytes as f64 * run.as_secs_f64() / elapsed.as_secs_f64()) as usize
            })
            .sum()
    });
    bytes as f64 / run.as_secs_f64() / 1e9
}

fn main() {
    let threads: Vec<usize> = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|list| list.split(',').map(|t| t.parse().unwrap()).collect())
        .unwrap_or_else(|| {
            let n = std::thread::available_parallelism().map_or(1, |n| n.get());
            let mut list = vec![1, n / 2, n];
            list.dedup();
            list
        });
    println!("signed FP8 ISA: {}", axnn_cpu::build_info::signed_f8_isa());
    // Weight-like values, as trained weights are: E4M3 over every finite
    // encoding but the NaN (the kernels' clean path), BF16 of |w| < 0.25
    // (no subnormals, which would cost microcode assists no real
    // checkpoint pays).
    let region = std::env::var("GEMV_REGION_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .and_then(|mb| mb.checked_mul(1 << 20))
        .unwrap_or(REGION);
    let e4m3: Vec<u8> = (0..region)
        .map(|i| ((i * 131 + i / 7) % 127) as u8)
        .collect();
    let bf16: Vec<u8> = (0..region / 2)
        .flat_map(|i| {
            let v = (((i * 2654435761) >> 7) % 20001) as f32 / 20000.0 - 0.5;
            ((((v * 0.5).to_bits()) >> 16) as u16).to_le_bytes()
        })
        .collect();
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if std::env::var("GEMV_AXPY").is_ok() {
        for (name, kernel) in [
            (
                "e4m3 axpy avx2",
                axpy_e4m3_avx2_f32 as fn(&[u8], f32, &mut [f32]),
            ),
            (
                "e4m3 axpy selected",
                axpy_e4m3_selected_f32 as fn(&[u8], f32, &mut [f32]),
            ),
        ] {
            for &t in &threads {
                println!(
                    "{name:<22} {t:>3} {:>8.1} GB/s",
                    measure_axpy(&e4m3, 2688, t, kernel)
                );
            }
        }
        return;
    }

    let cases: [(&str, usize, usize, Kernel, Kernel); 5] = [
        ("read      2688x2688", 2688, 2688, read, read),
        (
            "e4m3 up   1856x2688",
            1856,
            2688,
            rowwise_e4m3,
            gemv_e4m3_f32,
        ),
        (
            "e4m3 down 2688x1856",
            2688,
            1856,
            rowwise_e4m3,
            gemv_e4m3_f32,
        ),
        (
            "bf16      1024x2688",
            1024,
            2688,
            rowwise_bf16,
            gemv_bf16_f32,
        ),
        (
            "bf16      2688x4096",
            2688,
            4096,
            rowwise_bf16,
            gemv_bf16_f32,
        ),
    ];
    if std::env::var("GEMV_FP4_TOKENS").is_ok() {
        println!(
            "{:<20} {:>7} {:>16} {:>14} {:>10}",
            "nvfp4, k = 2688", "threads", "weight G elem/s", "G products/s", "speedup"
        );
        let k = 2688;
        for &t in &threads {
            for bits in [false, true] {
                let name = if bits { "bits" } else { "table" };
                let one = if bits {
                    fp4_tokens::<1, true>(&e4m3, k, t)
                } else {
                    fp4_tokens::<1, false>(&e4m3, k, t)
                };
                let rates = if bits {
                    [
                        one,
                        fp4_tokens::<2, true>(&e4m3, k, t),
                        fp4_tokens::<4, true>(&e4m3, k, t),
                        fp4_tokens::<8, true>(&e4m3, k, t),
                    ]
                } else {
                    [
                        one,
                        fp4_tokens::<2, false>(&e4m3, k, t),
                        fp4_tokens::<4, false>(&e4m3, k, t),
                        fp4_tokens::<8, false>(&e4m3, k, t),
                    ]
                };
                for (tokens, rate) in [1, 2, 4, 8].into_iter().zip(rates) {
                    println!(
                        "{:<20} {t:>7} {rate:>16.1} {:>14.1} {:>9.2}x",
                        format!("{name}, T = {tokens}"),
                        rate * tokens as f64,
                        rate * tokens as f64 / one
                    );
                }
            }
        }
        return;
    }
    if std::env::var("GEMV_ROOFLINE").is_ok() {
        // What the machine can do at all, in the same harness: the bus
        // reading weights nobody decodes, and FMA units fed from L1.
        println!(
            "{:<22} {:>7} {:>12} {:>14}",
            "roofline, k = 2688", "threads", "GB/s", "G elem/s"
        );
        let k = 2688;
        for &t in &threads {
            let bus = measure(read as Kernel, &e4m3, k, 64, k, t, true);
            println!("{:<22} {t:>7} {bus:>12.1} {:>14}", "bus, read only", "-");
            let fma = fma_peak(k, t);
            println!("{:<22} {t:>7} {:>12} {fma:>14.1}", "fma, L1 resident", "-");
        }
        return;
    }
    if std::env::var("GEMV_FP4").is_ok() {
        // Elements per second is what decides these kernels: the widen
        // bounds them well before the bus does.
        println!(
            "{:<20} {:>7} {:>10} {:>12} {:>10}",
            "k = 2688", "threads", "GB/s", "G elem/s", "bytes/elem"
        );
        let k = 2688;
        for (name, kernel, row_bytes, region) in [
            ("e4m3", rowwise_e4m3 as Kernel, k, &e4m3),
            ("e4m3 (selected)", gemv_e4m3_f32 as Kernel, k, &e4m3),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            ("e4m3 (avx2)", gemv_e4m3_avx2_f32 as Kernel, k, &e4m3),
            (
                "nvfp4 (table)",
                rowwise_nvfp4 as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            (
                "nvfp4 (bits)",
                rowwise_nvfp4_bits as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            #[cfg(target_arch = "aarch64")]
            (
                "nvfp4 (neon fma)",
                rowwise_nvfp4_neon as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
            (
                "nvfp4 (avx2)",
                rowwise_nvfp4_avx2 as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            (
                "nvfp4 (avx512)",
                rowwise_nvfp4_avx512 as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            #[cfg(all(target_arch = "x86_64", feature = "nvfp4_vnni"))]
            (
                "nvfp4 (w4a8 selected)",
                rowwise_nvfp4_vnni as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "avx_vnni"))]
            (
                "nvfp4 (w4a8 256)",
                rowwise_nvfp4_vnni_256 as Kernel,
                k / 2 + k / 16,
                &e4m3,
            ),
            ("bf16 (rowwise)", rowwise_bf16 as Kernel, 2 * k, &bf16),
            ("bf16 (selected)", gemv_bf16_f32 as Kernel, 2 * k, &bf16),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            ("bf16 (avx2)", gemv_bf16_avx2_f32 as Kernel, 2 * k, &bf16),
        ] {
            for &t in &threads {
                let gbs = measure(kernel, region, row_bytes, 64, k, t, true);
                let per = row_bytes as f64 / k as f64;
                println!(
                    "{name:<20} {t:>7} {gbs:>10.1} {:>12.1} {per:>10.3}",
                    gbs / per
                );
            }
        }
        return;
    }
    if std::env::var("GEMV_TOKENS").is_ok() {
        println!("{TOKENS} tokens per product, GB/s per token-equivalent");
        println!(
            "{:<22} {:>7} {:>11} {:>11}",
            "", "threads", "passes", "chunked"
        );
        for (name, rows, k, gemv, bf) in [
            (
                "e4m3 up   1856x2688",
                1856,
                2688,
                gemv_e4m3_f32 as Kernel,
                false,
            ),
            (
                "bf16      2688x4096",
                2688,
                4096,
                gemv_bf16_f32 as Kernel,
                true,
            ),
        ] {
            let (region, row_bytes) = if bf { (&bf16, 2 * k) } else { (&e4m3, k) };
            for &t in &threads {
                println!(
                    "{name:<22} {t:>7} {:>11.1} {:>11.1}",
                    tokens_measure(gemv, region, row_bytes, rows, k, t, false),
                    tokens_measure(gemv, region, row_bytes, rows, k, t, true)
                );
            }
        }
        return;
    }
    println!(
        "{:<22} {:>7} {:>11} {:>11} {:>11} {:>11}",
        "GB/s", "threads", "cached row", "cached gemv", "stream row", "stream gemv"
    );
    for (name, rows, k, rowwise, gemv) in cases {
        let (region, row_bytes) = if name.starts_with("bf16") {
            (&bf16, 2 * k)
        } else {
            (&e4m3, k)
        };
        // A 64-row slice stays in a core's L2 for the cached regime.
        let cached_rows = 64.min(rows);
        for &t in &threads {
            let cells = [
                measure(rowwise, region, row_bytes, cached_rows, k, t, false),
                measure(gemv, region, row_bytes, cached_rows, k, t, false),
                measure(rowwise, region, row_bytes, rows, k, t, true),
                measure(gemv, region, row_bytes, rows, k, t, true),
            ];
            println!(
                "{name:<22} {t:>7} {:>11.1} {:>11.1} {:>11.1} {:>11.1}",
                cells[0], cells[1], cells[2], cells[3]
            );
        }
    }
}
