//! Decode attention over a filled FP8 cache: one invocation of `T` query
//! rows at the end of the context, as a verified draft or a prefill chunk
//! runs it, in Nemotron-H's shape (2 cache heads of 128, 32 query heads).
//!
//! Run: cargo bench -p axnn-cpu --features bench --bench attention
//!
//! Single-threaded by default, so the kernel shows without a pool's noise;
//! `ATTENTION_THREADS=N` runs on N scoped threads instead (spawned per
//! fork-join, so short calls carry that cost). The last case fills 4K of a
//! 128K cache, as a long-context model does early on. The cache rate is the
//! bytes a single step at the last row's position would read, per second of
//! the whole invocation.

use std::time::{Duration, Instant};

use axnn::lowering::{
    AttentionDecodeParams, KernelRequest, KvCacheParams, KvCacheStorage, ResourceKey,
};
use axnn::{DType, Shape, TensorDesc};
use axnn_cpu::{
    CpuCapabilities, ForkJoin, KvCacheRef, Serial, attention_scratch_elements,
    invoke_attention_decode, invoke_kv_append, plan_cpu_kernel,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scoped threads per fork-join, the caller included.
struct Scoped(usize);

// SAFETY: every index below `tasks` is taken exactly once, and the scope
// joins every thread before `run` returns.
unsafe impl ForkJoin for Scoped {
    fn width(&self) -> usize {
        self.0
    }

    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        let next = AtomicUsize::new(0);
        let work = || loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= tasks {
                break;
            }
            task(index);
        };
        std::thread::scope(|scope| {
            for _ in 1..self.0.min(tasks) {
                scope.spawn(work);
            }
            work();
        });
    }
}

const KEY: ResourceKey = ResourceKey {
    group: 0,
    binding: 0,
};
const HEADS: usize = 2;
const DIM: usize = 128;
const QUERY_HEADS: usize = 32;
const PAGE: u32 = 16;

fn desc(dims: &[usize], dtype: DType) -> TensorDesc {
    TensorDesc::new(Shape::try_from(dims).unwrap(), dtype)
}

fn values(count: usize, seed: u32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let v = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8;
            (v % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn main() {
    let run = Duration::from_millis(
        std::env::var("ATTENTION_RUN_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400),
    );
    let threads: usize = std::env::var("ATTENTION_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let scoped = Scoped(threads);
    let parallel: &dyn ForkJoin = if threads > 1 { &scoped } else { &Serial };
    println!(
        "{:>9} {:>9} {:>7} {:>12} {:>11} {:>11}",
        "context", "capacity", "rows", "ms / call", "ms / row", "cache GB/s"
    );
    for (context, capacity) in [
        (4096usize, 4096usize),
        (32768, 32768),
        (131072, 131072),
        (4096, 131072),
    ] {
        let params = KvCacheParams::new(
            KEY,
            HEADS as u32,
            DIM as u32,
            capacity as u32,
            PAGE,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        let attention = AttentionDecodeParams::new(params, QUERY_HEADS as u32, 0.088).unwrap();
        let mut storage = vec![0_u8; params.storage_bytes().unwrap()];
        // Fill all but the last eight slots' worth of rows, then append the
        // run's own rows as the model does before attending.
        let fill = context - 8;
        let append = |storage: &mut [u8], at: usize, rows: usize, seed: u32| {
            let request = KernelRequest::kv_append(
                &desc(&[], DType::I32),
                &desc(&[rows, HEADS, DIM], DType::F32),
                &desc(&[rows, HEADS, DIM], DType::F32),
                params,
            )
            .unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
            let mut token = [0];
            invoke_kv_append(
                &plan,
                KvCacheRef::new(storage, params).unwrap(),
                &[at as i32],
                &values(rows * HEADS * DIM, seed),
                &values(rows * HEADS * DIM, seed + 1),
                &mut token,
            )
            .unwrap();
        };
        append(&mut storage, 0, fill + 8, 7);
        let mut scratch = vec![0.0_f32; attention_scratch_elements(attention)];
        for rows in [1usize, 4, 8] {
            let request = KernelRequest::attention_decode(
                &desc(&[1], DType::I32),
                &desc(&[], DType::I32),
                &desc(&[rows, QUERY_HEADS, DIM], DType::F32),
                attention,
            )
            .unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
            let queries = values(rows * QUERY_HEADS * DIM, 11);
            let mut output = vec![0.0_f32; queries.len()];
            let mut calls = 0u32;
            let start = Instant::now();
            while start.elapsed() < run || calls < 3 {
                invoke_attention_decode(
                    &plan,
                    KvCacheRef::new(&mut storage, params).unwrap(),
                    &[0],
                    &[fill as i32],
                    &queries,
                    &mut output,
                    &mut scratch,
                    parallel,
                )
                .unwrap();
                std::hint::black_box(&mut output);
                calls += 1;
            }
            let per_call = start.elapsed().as_secs_f64() / f64::from(calls);
            // A step at the last row reads every slot's key, value and two
            // scale bytes, for each cache head.
            let bytes = (fill + rows) * HEADS * (2 * DIM + 2);
            println!(
                "{context:>9} {capacity:>9} {rows:>7} {:>12.3} {:>11.3} {:>11.2}",
                per_call * 1e3,
                per_call * 1e3 / rows as f64,
                bytes as f64 / per_call / 1e9
            );
        }
    }
}
