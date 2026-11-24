//! Gaussian-splat tile compositing micro-benchmarks (3DGS Kernel B).
//!
//! Run:  cargo bench -p axnn-cpu --features bench --bench splat
//!
//! One 16×16 tile (32 strips of `SPLAT_PW`), a front-to-back list of N
//! splats scattered over it. Opacities are kept moderate so early
//! termination stays rare and the rows measure raw blend throughput;
//! the counter is nominal splat·pixel blends. The three color formats
//! share identical geometry — the delta is the color-table decode and
//! its footprint (12 B/splat f32, 6 B f16, 3 B FP8/UF8). This benchmark
//! repeatedly walks one hot shared table, so it measures decode/compositing
//! throughput; it does not claim a cache-pressure result for a particular
//! renderer's binning layout.

use axnn_cpu::_bench::*;
use axnn_cpu::splat::{composite_strip_uf8e4m4, composite_strip_uf8e5m3};
use axnn_cpu::uf;
use divan::counter::ItemsCount;

fn main() {
    println!("splat ISA: {}", axnn_cpu::build_info::splat_isa());
    divan::main();
}

const TILE: usize = 16;
const STRIPS_X: usize = TILE / SPLAT_PW;

fn scene(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut center = Vec::new();
    let mut conic = Vec::new();
    let mut opacity = Vec::new();
    let mut color = Vec::new();
    for i in 0..n {
        center.push((i * 7 % (TILE * 4)) as f32 * 0.25);
        center.push((i * 11 % (TILE * 4)) as f32 * 0.25);
        let inv_s2 = 1.0 / (2.0 + (i % 7) as f32); // σ² in 2..9 px²
        conic.push(inv_s2);
        conic.push(0.15 * inv_s2 * if i % 2 == 0 { 1.0 } else { -1.0 });
        conic.push(1.2 * inv_s2);
        opacity.push(0.15 + 0.1 * (i % 3) as f32);
        for ch in 0..3 {
            color.push(((i * 5 + ch * 3) % 100) as f32 * 0.01);
        }
    }
    (center, conic, opacity, color, (0..n as u32).collect())
}

const SPLATS: usize = 1024;

macro_rules! bench_composite {
    ($name:ident, $kernel:ident, $colors:expr) => {
        #[divan::bench]
        fn $name(bencher: divan::Bencher) {
            let (center, conic, opacity, color_f32, ids) = scene(SPLATS);
            let color = $colors(&color_f32);
            bencher
                .counter(ItemsCount::new(SPLATS * TILE * TILE))
                .with_inputs(|| (vec![0.0f32; 3 * SPLAT_PW], vec![1.0f32; SPLAT_PW]))
                .bench_values(|(mut rgb, mut t)| {
                    let mut consumed = 0usize;
                    for sy in 0..TILE {
                        for sx in 0..STRIPS_X {
                            rgb.fill(0.0);
                            t.fill(1.0);
                            consumed += $kernel(
                                divan::black_box(&center),
                                divan::black_box(&conic),
                                divan::black_box(&opacity),
                                divan::black_box(&color),
                                divan::black_box(&ids),
                                (sx * SPLAT_PW) as f32 + 0.5,
                                sy as f32 + 0.5,
                                &mut rgb,
                                &mut t,
                            );
                        }
                    }
                    (rgb, t, consumed)
                });
        }
    };
}

bench_composite!(composite_f32, composite_strip_f32, |c: &[f32]| c.to_vec());
bench_composite!(composite_f16, composite_strip_f16, |c: &[f32]| c
    .iter()
    .flat_map(|&v| pack_f16(v))
    .collect::<Vec<u8>>());
bench_composite!(composite_e4m3, composite_strip_e4m3, |c: &[f32]| c
    .iter()
    .map(|&v| quantize_e4m3(v))
    .collect::<Vec<u8>>());
bench_composite!(composite_uf8e4m4, composite_strip_uf8e4m4, |c: &[f32]| c
    .iter()
    .map(|&v| uf::f32_to_uf8e4m4_rne(v))
    .collect::<Vec<u8>>());
bench_composite!(composite_uf8e5m3, composite_strip_uf8e5m3, |c: &[f32]| c
    .iter()
    .map(|&v| uf::f32_to_uf8e5m3_rne(v))
    .collect::<Vec<u8>>());

/// Nearest-value E4M3 encode for bench setup: pick the closest of the
/// 256 encodings (exhaustive — setup cost only).
fn quantize_e4m3(v: f32) -> u8 {
    let mut best = (f32::INFINITY, 0u8);
    for b in 0..=255u8 {
        let d = axnn_cpu::_bench::decode_e4m3(b);
        if d.is_nan() {
            continue;
        }
        let err = (d - v).abs();
        if err < best.0 {
            best = (err, b);
        }
    }
    best.1
}
