//! Voxel ray tracing micro-benchmarks — dense-grid DDA vs SVO.
//!
//! Run:  cargo bench -p axnn-cpu --features bench --bench voxel
//!
//! Scene: a sphere shell (30%..38% of the grid edge) — sparse
//! occupancy with real empty interior, so the two traversals split:
//! DDA pays per voxel crossed, the SVO skips empty space a subtree at
//! a time. Rays are an orthographic bundle from outside (all hit the
//! shell) plus a divergent bundle from the hollow center (mixed path
//! lengths). Rates are rays/s; build rows report voxels/s consumed.

use axnn_cpu::_bench::*;
use axnn_cpu::voxel::{
    march_density_uf8e4m4, march_density_uf8e5m3, march_emission_e4m3, march_emission_f32,
    march_emission_uf8e4m4, march_emission_uf8e5m3, sample_trilinear_e4m3,
    sample_trilinear_uf8e4m4, sample_trilinear_uf8e5m3,
};
use axnn_cpu::{fp8, uf};
use divan::counter::ItemsCount;

fn main() {
    println!("voxel ISA: {}", axnn_cpu::build_info::voxel_isa());
    divan::main();
}

fn shell(n: u32) -> Vec<u64> {
    let mut bits = vec![0u64; ((n * n * n) as usize).div_ceil(64)];
    let c = n as f32 * 0.5;
    let (r0, r1) = (n as f32 * 0.30, n as f32 * 0.38);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (dx, dy, dz) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c, z as f32 + 0.5 - c);
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r >= r0 && r <= r1 {
                    let idx = (x + n * (y + n * z)) as usize;
                    bits[idx / 64] |= 1 << (idx % 64);
                }
            }
        }
    }
    bits
}

/// 64×64 orthographic bundle from below (+z, slight tilt) and a
/// divergent bundle from the shell's hollow center.
fn rays(n: u32) -> (Vec<f32>, Vec<f32>, usize) {
    let (mut origins, mut dirs) = (Vec::new(), Vec::new());
    let side = 64usize;
    let scale = n as f32 / side as f32;
    for iy in 0..side {
        for ix in 0..side {
            origins.extend_from_slice(&[
                (ix as f32 + 0.5) * scale,
                (iy as f32 + 0.5) * scale,
                -1.0,
            ]);
            dirs.extend_from_slice(&[0.011 * (ix % 7) as f32, 0.013 * (iy % 5) as f32, 1.0]);
        }
    }
    let c = n as f32 * 0.5;
    for i in 0..side * side {
        origins.extend_from_slice(&[c, c, c]);
        let a = i as f32 * 0.61803;
        let b = i as f32 * 0.31831;
        dirs.extend_from_slice(&[a.cos() * b.sin(), a.sin() * b.sin(), b.cos()]);
    }
    let count = origins.len() / 3;
    (origins, dirs, count)
}

const SIZES: &[u32] = &[64, 256];

#[divan::bench(args = SIZES)]
fn dda(bencher: divan::Bencher, n: u32) {
    let bits = shell(n);
    let (origins, dirs, count) = rays(n);
    bencher
        .counter(ItemsCount::new(count))
        .with_inputs(|| (vec![0.0f32; count], vec![0u32; count]))
        .bench_values(|(mut t, mut idx)| {
            dda_trace_n(
                divan::black_box(&bits),
                [n; 3],
                divan::black_box(&origins),
                divan::black_box(&dirs),
                count,
                4.0 * n as f32,
                &mut t,
                &mut idx,
            );
            (t, idx)
        });
}

#[divan::bench(args = SIZES)]
fn svo(bencher: divan::Bencher, n: u32) {
    let levels = n.trailing_zeros();
    let bits = shell(n);
    let mut nodes = vec![SvoNode::default(); 4_000_000];
    let cnt = svo_build(&bits, levels, &mut nodes).unwrap();
    nodes.truncate(cnt);
    let (origins, dirs, count) = rays(n);
    bencher
        .counter(ItemsCount::new(count))
        .with_inputs(|| (vec![0.0f32; count], vec![0u32; count]))
        .bench_values(|(mut t, mut idx)| {
            svo_trace_n(
                divan::black_box(&nodes),
                levels,
                divan::black_box(&origins),
                divan::black_box(&dirs),
                count,
                4.0 * n as f32,
                &mut t,
                &mut idx,
            );
            (t, idx)
        });
}

#[divan::bench(args = SIZES)]
fn brick(bencher: divan::Bencher, n: u32) {
    let bits = shell(n);
    let mut bricks = vec![0u64; ((n / VOXEL_BRICK).pow(3)) as usize];
    brick_pack(&bits, [n; 3], &mut bricks);
    let (origins, dirs, count) = rays(n);
    bencher
        .counter(ItemsCount::new(count))
        .with_inputs(|| (vec![0.0f32; count], vec![0u32; count]))
        .bench_values(|(mut t, mut idx)| {
            brick_trace_n(
                divan::black_box(&bricks),
                [n; 3],
                divan::black_box(&origins),
                divan::black_box(&dirs),
                count,
                4.0 * n as f32,
                &mut t,
                &mut idx,
            );
            (t, idx)
        });
}

/// Density volume: the shell band filled with a varying σ (0 outside),
/// scaled so rays reach the opacity cutoff mid-shell — both the
/// integration and the early exit run. Formats share the traversal;
/// the delta is per-step decode plus volume footprint (f32 64 MB vs
/// f16 32 MB vs FP8 16 MB at 256³) — the mixed-precision story.
fn density_f32(n: u32) -> Vec<f32> {
    let bits = shell(n);
    (0..(n * n * n) as usize)
        .map(|i| {
            if bits[i / 64] >> (i % 64) & 1 != 0 {
                0.4 + 0.05 * (i % 8) as f32
            } else {
                0.0
            }
        })
        .collect()
}

const TAU_MAX: f32 = 9.21; // -ln(1e-4)

macro_rules! bench_march {
    ($name:ident, $kernel:ident, $conv:expr) => {
        #[divan::bench(args = SIZES)]
        fn $name(bencher: divan::Bencher, n: u32) {
            let density = $conv(&density_f32(n));
            let (origins, dirs, count) = rays(n);
            bencher
                .counter(ItemsCount::new(count))
                .with_inputs(|| vec![(0.0f32, 0.0f32); count])
                .bench_values(|mut out| {
                    for i in 0..count {
                        out[i] = $kernel(
                            divan::black_box(&density),
                            [n; 3],
                            [origins[i * 3], origins[i * 3 + 1], origins[i * 3 + 2]],
                            [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]],
                            4.0 * n as f32,
                            TAU_MAX,
                        );
                    }
                    out
                });
        }
    };
}

bench_march!(march_f32, march_density_f32, |d: &[f32]| d.to_vec());
bench_march!(march_f16, march_density_f16, |d: &[f32]| d
    .iter()
    .flat_map(|&v| pack_f16(v))
    .collect::<Vec<u8>>());
bench_march!(march_e4m3, march_density_e4m3, |d: &[f32]| d
    .iter()
    .map(|&v| fp8::f32_to_e4m3_rne(v))
    .collect::<Vec<u8>>());
bench_march!(march_uf8e4m4, march_density_uf8e4m4, |d: &[f32]| d
    .iter()
    .map(|&v| uf::f32_to_uf8e4m4_rne(v))
    .collect::<Vec<u8>>());
bench_march!(march_uf8e5m3, march_density_uf8e5m3, |d: &[f32]| d
    .iter()
    .map(|&v| uf::f32_to_uf8e5m3_rne(v))
    .collect::<Vec<u8>>());

fn source_f32(n: u32) -> Vec<f32> {
    let density = density_f32(n);
    density
        .into_iter()
        .enumerate()
        .flat_map(|(i, value)| {
            let phase = (i % 97) as f32 / 97.0;
            [
                value * (0.4 + 0.6 * phase),
                value * (0.7 + 0.3 * (1.0 - phase)),
                value * (0.25 + 0.75 * phase * phase),
            ]
        })
        .collect()
}

// Emissive transfer is substantially more expensive than density-only DDA;
// cap the largest volume so the benchmark remains practical while still
// crossing the last-level-cache boundary.
const TRANSFER_SIZES: &[u32] = &[64, 128];

macro_rules! bench_emission {
    ($name:ident, $kernel:ident, $conv:expr) => {
        #[divan::bench(args = TRANSFER_SIZES)]
        fn $name(bencher: divan::Bencher, n: u32) {
            let extinction = $conv(&density_f32(n));
            let source = $conv(&source_f32(n));
            let (origins, dirs, count) = rays(n);
            bencher.counter(ItemsCount::new(count)).bench(|| {
                let mut checksum = 0.0f32;
                for i in 0..count {
                    let transfer = $kernel(
                        divan::black_box(&extinction),
                        divan::black_box(&source),
                        [n; 3],
                        [origins[i * 3], origins[i * 3 + 1], origins[i * 3 + 2]],
                        [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]],
                        4.0 * n as f32,
                        0.7,
                        1.3,
                        0.01,
                    );
                    checksum += transfer.radiance[0]
                        + transfer.radiance[1]
                        + transfer.radiance[2]
                        + transfer.transmittance
                        + transfer.t_end;
                }
                divan::black_box(checksum)
            });
        }
    };
}

bench_emission!(emission_f32, march_emission_f32, |v: &[f32]| v.to_vec());
bench_emission!(emission_e4m3, march_emission_e4m3, |v: &[f32]| v
    .iter()
    .map(|&value| fp8::f32_to_e4m3_rne(value))
    .collect::<Vec<u8>>());
bench_emission!(emission_uf8e4m4, march_emission_uf8e4m4, |v: &[f32]| v
    .iter()
    .map(|&value| uf::f32_to_uf8e4m4_rne(value))
    .collect::<Vec<u8>>());
bench_emission!(emission_uf8e5m3, march_emission_uf8e5m3, |v: &[f32]| v
    .iter()
    .map(|&value| uf::f32_to_uf8e5m3_rne(value))
    .collect::<Vec<u8>>());

const SAMPLE_COUNT: usize = 65_536;

fn sample_points(n: u32) -> Vec<[f32; 3]> {
    (0..SAMPLE_COUNT)
        .map(|i| {
            let extent = n as f32;
            [
                0.5 + ((i * 73 % (n as usize * 997)) as f32 / 997.0) % (extent - 1.0),
                0.5 + ((i * 43 % (n as usize * 991)) as f32 / 991.0) % (extent - 1.0),
                0.5 + ((i * 29 % (n as usize * 983)) as f32 / 983.0) % (extent - 1.0),
            ]
        })
        .collect()
}

macro_rules! bench_trilinear {
    ($name:ident, $kernel:ident, $encode:path) => {
        #[divan::bench(args = SIZES)]
        fn $name(bencher: divan::Bencher, n: u32) {
            let field: Vec<u8> = density_f32(n).into_iter().map($encode).collect();
            let points = sample_points(n);
            bencher.counter(ItemsCount::new(points.len())).bench(|| {
                let mut checksum = 0.0f32;
                for &point in &points {
                    checksum += $kernel(divan::black_box(&field), [n; 3], point);
                }
                divan::black_box(checksum)
            });
        }
    };
}

bench_trilinear!(trilinear_e4m3, sample_trilinear_e4m3, fp8::f32_to_e4m3_rne);
bench_trilinear!(
    trilinear_uf8e4m4,
    sample_trilinear_uf8e4m4,
    uf::f32_to_uf8e4m4_rne
);
bench_trilinear!(
    trilinear_uf8e5m3,
    sample_trilinear_uf8e5m3,
    uf::f32_to_uf8e5m3_rne
);

#[divan::bench(args = SIZES)]
fn svo_build_(bencher: divan::Bencher, n: u32) {
    let levels = n.trailing_zeros();
    let bits = shell(n);
    bencher
        .counter(ItemsCount::new((n as usize).pow(3)))
        .with_inputs(|| vec![SvoNode::default(); 4_000_000])
        .bench_values(|mut nodes| {
            let cnt = svo_build(divan::black_box(&bits), levels, &mut nodes).unwrap();
            (nodes, cnt)
        });
}
