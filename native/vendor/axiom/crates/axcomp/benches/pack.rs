//! Staged-materializer micro-benchmarks — planar provider output → XRGB.
//!
//! Run:  cargo bench -p axcomp --bench pack
//!
//! Bytes/s counts packed output pixels (4 bytes each) per materialization.

use axcomp::{LinearXrgbWords, PlanarFormat, PlanarOutput, pack_planar_xrgb_words};
use axgfx::{Damage, Extent2d, PixelRect};
use divan::counter::BytesCount;

fn main() {
    divan::main();
}

const W: u32 = 1280;
const H: u32 = 720;

fn f32_plane_bytes(seed: u32) -> Vec<u8> {
    (0..W * H)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0x3FF) as f32 / 1023.0)
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn f32_plane(seed: u32) -> Vec<f32> {
    (0..W * H)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0x3FF) as f32 / 1023.0)
        .collect()
}

fn biased_plane(seed: u32) -> Vec<u8> {
    (0..W * H)
        .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 12) as u8)
        .collect()
}

fn bench_words(bencher: divan::Bencher, output: PlanarOutput<'_>, damage: Damage<'_>, pixels: u32) {
    let extent = output.extent();
    let stride = extent.width() as usize + 16;
    let mut words = vec![0u32; stride * extent.height() as usize];
    bencher
        .counter(BytesCount::new(pixels as usize * 4))
        .bench_local(|| {
            let mut target = LinearXrgbWords::new(&mut words, extent, stride).unwrap();
            pack_planar_xrgb_words(divan::black_box(output), &mut target, damage).unwrap();
        });
}

#[divan::bench]
fn pack_f32_1280x720_full(bencher: divan::Bencher) {
    let extent = Extent2d::new(W, H).unwrap();
    let (r, g, b) = (f32_plane(1), f32_plane(2), f32_plane(3));
    let output = PlanarOutput::f32([&r, &g, &b], extent).unwrap();
    bench_words(bencher, output, Damage::Full, W * H);
}

#[divan::bench]
fn pack_f32_damage_64x64(bencher: divan::Bencher) {
    let extent = Extent2d::new(W, H).unwrap();
    let (r, g, b) = (f32_plane(1), f32_plane(2), f32_plane(3));
    let output = PlanarOutput::f32([&r, &g, &b], extent).unwrap();
    let rect = [PixelRect::new(600, 300, 64, 64).unwrap()];
    bench_words(bencher, output, Damage::Rects(&rect), 64 * 64);
}

#[divan::bench]
fn pack_f32_bytes_1280x720_full(bencher: divan::Bencher) {
    let extent = Extent2d::new(W, H).unwrap();
    let (r, g, b) = (f32_plane_bytes(1), f32_plane_bytes(2), f32_plane_bytes(3));
    let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::F32, extent).unwrap();
    bench_words(bencher, output, Damage::Full, W * H);
}

#[divan::bench]
fn pack_f32_bytes_damage_64x64(bencher: divan::Bencher) {
    let extent = Extent2d::new(W, H).unwrap();
    let (r, g, b) = (f32_plane_bytes(1), f32_plane_bytes(2), f32_plane_bytes(3));
    let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::F32, extent).unwrap();
    let rect = [PixelRect::new(600, 300, 64, 64).unwrap()];
    bench_words(bencher, output, Damage::Rects(&rect), 64 * 64);
}

#[divan::bench]
fn pack_u8_biased_1280x720(bencher: divan::Bencher) {
    let extent = Extent2d::new(W, H).unwrap();
    let (r, g, b) = (biased_plane(1), biased_plane(2), biased_plane(3));
    let output = PlanarOutput::new([&r, &g, &b], PlanarFormat::U8Biased, extent).unwrap();
    bench_words(bencher, output, Damage::Full, W * H);
}
