//! Raster primitive micro-benchmarks.
//!
//! Run:  cargo bench -p axnn-cpu --features bench -- raster
//! NEON: cargo bench -p axnn-cpu --features bench,neon -- raster
//!
//! `bench_*` measures the active tier (portable core::simd).
//! `baseline_*` measures the scalar tier (no SIMD).

use axnn::{DType, RasterBackend, RoundedShape, TensorDesc, shape};
use axnn_cpu::_bench::baseline;
use axnn_cpu::_bench::*;
use axnn_cpu::{CpuBackend, raster, uf};
use divan::counter::BytesCount;

// Re-import for fused bench (same crate, same wildcard, just documenting the dep)
// fused_edge_coverage is exported from _bench alongside the other primitives.

fn main() {
    println!("raster ISA: {}", axnn_cpu::build_info::raster_isa());
    divan::main();
}

// ── Tile sizes ────────────────────────────────────────────────────────
//
// Typical software rasterizer tile: 8×8 (64), 16×16 (256), 32×32 (1024).
// 4096 represents a large tile or a full scanline stripe.

const TILE_SIZES: &[usize] = &[64, 256, 1024, 4096];

// ── Input factories ───────────────────────────────────────────────────

/// Edge function values for n pixels along a scanline row.
/// Deterministic: half the pixels pass (positive), half fail (negative).
fn make_edge_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn make_masks(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 0xFFFF_FFFF } else { 0 })
        .collect()
}

fn make_color(n: usize) -> Vec<u32> {
    vec![0xFF_40_80_C0u32; n] // packed RGBA
}

// ── Individual primitive benches ──────────────────────────────────────

#[divan::bench(args = TILE_SIZES)]
fn bench_sub_f32(bencher: divan::Bencher, n: usize) {
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            sub_f32(divan::black_box(&a), divan::black_box(&b), &mut dst);
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_cmpge_f32(bencher: divan::Bencher, n: usize) {
    let a = make_edge_values(n);
    let zero = vec![0.0f32; n];
    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut dst| {
            cmpge_f32(divan::black_box(&a), divan::black_box(&zero), &mut dst);
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_and_masks(bencher: divan::Bencher, n: usize) {
    let a = make_masks(n);
    let b = make_masks(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut dst| {
            and_masks(divan::black_box(&a), divan::black_box(&b), &mut dst);
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_maskstore_u32(bencher: divan::Bencher, n: usize) {
    let mask = make_masks(n);
    let color = make_color(n);
    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut fb| {
            maskstore_u32(divan::black_box(&mask), divan::black_box(&color), &mut fb);
            fb
        });
}

// ── Scalar baselines ────────────────────────────────────────────────

#[divan::bench(args = TILE_SIZES)]
fn baseline_sub_f32(bencher: divan::Bencher, n: usize) {
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            baseline::sub_f32(divan::black_box(&a), divan::black_box(&b), &mut dst);
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_cmpge_f32(bencher: divan::Bencher, n: usize) {
    let a = make_edge_values(n);
    let zero = vec![0.0f32; n];
    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut dst| {
            baseline::cmpge_f32(divan::black_box(&a), divan::black_box(&zero), &mut dst);
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_fused_edge_coverage(bencher: divan::Bencher, n: usize) {
    let e0: Vec<f32> = vec![1.0f32; n];
    let e1 = make_edge_values(n);
    let e2: Vec<f32> = vec![1.0f32; n];
    let color = make_color(n);
    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut fb| {
            baseline::fused_edge_coverage(
                divan::black_box(&e0),
                divan::black_box(&e1),
                divan::black_box(&e2),
                0.0,
                divan::black_box(&color),
                &mut fb,
            );
            fb
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_edge_coverage_pipeline(bencher: divan::Bencher, n: usize) {
    let e0: Vec<f32> = vec![1.0f32; n];
    let e1 = make_edge_values(n);
    let e2: Vec<f32> = vec![1.0f32; n];
    let zero = vec![0.0f32; n];
    let color = make_color(n);
    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| {
            (
                vec![0u32; n],
                vec![0u32; n],
                vec![0u32; n],
                vec![0u32; n],
                vec![0u32; n],
            )
        })
        .bench_values(|(mut m0, mut m1, mut m2, mut coverage, mut fb)| {
            baseline::cmpge_f32(divan::black_box(&e0), &zero, &mut m0);
            baseline::cmpge_f32(divan::black_box(&e1), &zero, &mut m1);
            baseline::cmpge_f32(divan::black_box(&e2), &zero, &mut m2);
            baseline::and_masks(&m0, &m1, &mut coverage);
            baseline::and_masks(&coverage, &m2, &mut m0);
            baseline::maskstore_u32(&m0, divan::black_box(&color), &mut fb);
            fb
        });
}

// ── Fused single-pass coverage bench ─────────────────────────────────
//
// All 3 comparisons + 2 mask ANDs + maskstore in one loop, zero
// intermediate mask buffers materialised.  Direct comparison against
// bench_edge_coverage_pipeline below quantifies the bandwidth saving.

#[divan::bench(args = TILE_SIZES)]
fn bench_fused_edge_coverage(bencher: divan::Bencher, n: usize) {
    let e0: Vec<f32> = vec![1.0f32; n];
    let e1 = make_edge_values(n);
    let e2: Vec<f32> = vec![1.0f32; n];
    let color = make_color(n);

    bencher
        .counter(BytesCount::new(n * 4)) // same output footprint as pipeline
        .with_inputs(|| vec![0u32; n])
        .bench_values(|mut fb| {
            fused_edge_coverage(
                divan::black_box(&e0),
                divan::black_box(&e1),
                divan::black_box(&e2),
                0.0,
                divan::black_box(&color),
                &mut fb,
            );
            fb
        });
}

// ── Full coverage pipeline bench ──────────────────────────────────────
//
// Models the actual inner loop: evaluate three edge functions for a row
// of pixels, AND the three coverage masks, write covered pixels.
//
// This is the number that matters for the SIMD vs scalar comparison —
// individual primitives are cheap enough that only the fused pipeline
// reflects real rasterizer cost.

#[divan::bench(args = TILE_SIZES)]
fn bench_edge_coverage_pipeline(bencher: divan::Bencher, n: usize) {
    // Simulate edge function values for three triangle edges.
    // Edge A: all pass; edge B: alternating; edge C: all pass.
    // ~50% pixel coverage — realistic for a triangle covering half a tile.
    let e0: Vec<f32> = vec![1.0f32; n];
    let e1 = make_edge_values(n);
    let e2: Vec<f32> = vec![1.0f32; n];
    let zero = vec![0.0f32; n];
    let color = make_color(n);

    bencher
        .counter(BytesCount::new(n * 4))
        .with_inputs(|| {
            let m0 = vec![0u32; n];
            let m1 = vec![0u32; n];
            let m2 = vec![0u32; n];
            let coverage = vec![0u32; n];
            let fb = vec![0u32; n];
            (m0, m1, m2, coverage, fb)
        })
        .bench_values(|(mut m0, mut m1, mut m2, mut coverage, mut fb)| {
            cmpge_f32(divan::black_box(&e0), &zero, &mut m0);
            cmpge_f32(divan::black_box(&e1), &zero, &mut m1);
            cmpge_f32(divan::black_box(&e2), &zero, &mut m2);
            and_masks(&m0, &m1, &mut coverage);
            // final coverage = m0 & m1 & m2; use m0 as scratch to avoid aliasing
            and_masks(&coverage, &m2, &mut m0);
            maskstore_u32(&m0, divan::black_box(&color), &mut fb);
            fb
        });
}

// ═══════════════════════════════════════════════════════════════════
//  Fragment processing benchmarks
// ═══════════════════════════════════════════════════════════════════

fn make_zbuf(n: usize) -> Vec<f32> {
    vec![1.0f32; n] // far plane
}

fn make_frag_z(n: usize) -> Vec<f32> {
    (0..n).map(|i| if i % 3 == 0 { 0.5 } else { 1.5 }).collect() // 1/3 pass
}

fn make_weights(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Simulate barycentric weights that sum to 1.0
    let w0: Vec<f32> = (0..n).map(|i| 0.5 + 0.3 * (i as f32 / n as f32)).collect();
    let w1: Vec<f32> = (0..n).map(|i| 0.3 - 0.2 * (i as f32 / n as f32)).collect();
    let w2: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| 1.0 - a - b).collect();
    (w0, w1, w2)
}

fn make_rgba_src(n: usize) -> Vec<u32> {
    // Semi-transparent red
    vec![0x80_00_00_FFu32; n]
}

#[divan::bench(args = TILE_SIZES)]
fn bench_depth_test(bencher: divan::Bencher, n: usize) {
    let coverage = make_masks(n);
    let frag_z = make_frag_z(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2)) // read frag_z + zbuf
        .with_inputs(|| (make_zbuf(n), vec![0u32; n]))
        .bench_values(|(mut zbuf, mut out)| {
            depth_test(
                divan::black_box(&coverage),
                divan::black_box(&frag_z),
                &mut zbuf,
                &mut out,
            );
            (zbuf, out)
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_depth_test_uf16e5m11(bencher: divan::Bencher, n: usize) {
    let coverage = make_masks(n);
    let frag_z: Vec<u16> = make_frag_z(n)
        .into_iter()
        .map(uf::f32_to_uf16e5m11_rne)
        .collect();
    let far = uf::f32_to_uf16e5m11_rne(1.0);
    bencher
        .counter(BytesCount::new(n * 2 * 2)) // fragment + z-buffer planes
        .with_inputs(|| (vec![far; n], vec![0u32; n]))
        .bench_values(|(mut zbuf, mut out)| {
            raster::depth_test_uf16e5m11(
                divan::black_box(&coverage),
                divan::black_box(&frag_z),
                &mut zbuf,
                &mut out,
            );
            (zbuf, out)
        });
}

/// Honest unfused pipeline comparator: quantize a freshly interpolated F32
/// depth plane, then run the compact UF16 depth test. If this loses to F32,
/// the production win requires the fused barycentric/encode/compare kernel.
#[divan::bench(args = TILE_SIZES)]
fn bench_depth_test_uf16e5m11_with_encode(bencher: divan::Bencher, n: usize) {
    let coverage = make_masks(n);
    let frag_f32 = make_frag_z(n);
    let far = uf::f32_to_uf16e5m11_rne(1.0);
    bencher
        .counter(BytesCount::new(n * (4 + 2 + 2)))
        .with_inputs(|| (vec![0u16; n], vec![far; n], vec![0u32; n]))
        .bench_values(|(mut frag_z, mut zbuf, mut out)| {
            for (dst, &value) in frag_z.iter_mut().zip(divan::black_box(&frag_f32)) {
                *dst = uf::f32_to_uf16e5m11_rne(value);
            }
            raster::depth_test_uf16e5m11(divan::black_box(&coverage), &frag_z, &mut zbuf, &mut out);
            (frag_z, zbuf, out)
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_depth_test(bencher: divan::Bencher, n: usize) {
    let coverage = make_masks(n);
    let frag_z = make_frag_z(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| (make_zbuf(n), vec![0u32; n]))
        .bench_values(|(mut zbuf, mut out)| {
            baseline::depth_test(
                divan::black_box(&coverage),
                divan::black_box(&frag_z),
                &mut zbuf,
                &mut out,
            );
            (zbuf, out)
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_bary_interp(bencher: divan::Bencher, n: usize) {
    let (w0, w1, w2) = make_weights(n);
    bencher
        .counter(BytesCount::new(n * 4 * 3)) // read 3 weight arrays
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            bary_interp(
                divan::black_box(&w0),
                divan::black_box(&w1),
                divan::black_box(&w2),
                1.0,
                0.5,
                0.0,
                &mut dst,
            );
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_bary_interp(bencher: divan::Bencher, n: usize) {
    let (w0, w1, w2) = make_weights(n);
    bencher
        .counter(BytesCount::new(n * 4 * 3))
        .with_inputs(|| vec![0.0f32; n])
        .bench_values(|mut dst| {
            baseline::bary_interp(
                divan::black_box(&w0),
                divan::black_box(&w1),
                divan::black_box(&w2),
                1.0,
                0.5,
                0.0,
                &mut dst,
            );
            dst
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_alpha_blend(bencher: divan::Bencher, n: usize) {
    let mask = make_masks(n);
    let src = make_rgba_src(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2)) // read src + dst
        .with_inputs(|| make_color(n))
        .bench_values(|mut fb| {
            alpha_blend(divan::black_box(&mask), divan::black_box(&src), &mut fb);
            fb
        });
}

#[divan::bench(args = TILE_SIZES)]
fn baseline_alpha_blend(bencher: divan::Bencher, n: usize) {
    let mask = make_masks(n);
    let src = make_rgba_src(n);
    bencher
        .counter(BytesCount::new(n * 4 * 2))
        .with_inputs(|| make_color(n))
        .bench_values(|mut fb| {
            baseline::alpha_blend(divan::black_box(&mask), divan::black_box(&src), &mut fb);
            fb
        });
}

#[divan::bench(args = TILE_SIZES)]
fn bench_xrgb_pack_unpack(bencher: divan::Bencher, n: usize) {
    let packed = make_color(n);
    bencher
        .counter(BytesCount::new(n * 4 * 4)) // read packed, write 3 channels
        .with_inputs(|| {
            (
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0u32; n],
            )
        })
        .bench_values(|(mut r, mut g, mut b, mut out)| {
            xrgb_unpack(divan::black_box(&packed), &mut r, &mut g, &mut b);
            xrgb_pack(&r, &g, &b, &mut out);
            out
        });
}

// ═══════════════════════════════════════════════════════════════════
//  Blit benchmarks
// ═══════════════════════════════════════════════════════════════════

// Simulate rectangular blits at common tile sizes.
// Pitch = width (tightly packed rows).
const BLIT_SIZES: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256)];

fn make_src_surface(w: usize, h: usize) -> Vec<u32> {
    // Semi-transparent source for blend, opaque for opaque blit
    (0..w * h)
        .map(|i| 0x80_40_80_C0u32.wrapping_add(i as u32))
        .collect()
}

#[divan::bench(args = BLIT_SIZES)]
fn bench_blit_opaque(bencher: divan::Bencher, &(w, h): &(usize, usize)) {
    let src = make_src_surface(w, h);
    bencher
        .counter(BytesCount::new(w * h * 4))
        .with_inputs(|| vec![0u32; w * h])
        .bench_values(|mut dst| {
            blit_opaque(divan::black_box(&src), w, &mut dst, w, w, h);
            dst
        });
}

#[divan::bench(args = BLIT_SIZES)]
fn baseline_blit_opaque(bencher: divan::Bencher, &(w, h): &(usize, usize)) {
    let src = make_src_surface(w, h);
    bencher
        .counter(BytesCount::new(w * h * 4))
        .with_inputs(|| vec![0u32; w * h])
        .bench_values(|mut dst| {
            baseline::blit_opaque(divan::black_box(&src), w, &mut dst, w, w, h);
            dst
        });
}

#[divan::bench(args = BLIT_SIZES)]
fn bench_blit_blend(bencher: divan::Bencher, &(w, h): &(usize, usize)) {
    let src = make_src_surface(w, h);
    bencher
        .counter(BytesCount::new(w * h * 4 * 2)) // read src + dst
        .with_inputs(|| vec![0xFF_20_40_60u32; w * h])
        .bench_values(|mut dst| {
            blit_blend(divan::black_box(&src), w, &mut dst, w, w, h);
            dst
        });
}

#[divan::bench(args = BLIT_SIZES)]
fn baseline_blit_blend(bencher: divan::Bencher, &(w, h): &(usize, usize)) {
    let src = make_src_surface(w, h);
    bencher
        .counter(BytesCount::new(w * h * 4 * 2))
        .with_inputs(|| vec![0xFF_20_40_60u32; w * h])
        .bench_values(|mut dst| {
            baseline::blit_blend(divan::black_box(&src), w, &mut dst, w, w, h);
            dst
        });
}

// ═══════════════════════════════════════════════════════════════════
//  Shape primitives — through `RasterBackend` on a wrapped framebuffer
// ═══════════════════════════════════════════════════════════════════

type ShapeBe = CpuBackend<4096>;

/// Bytes/s counts the shape's bounding box, the pixels the op is asked
/// to produce; `rounded_fill` of a square shape must match `fill_rect`.
fn bench_shape(
    bencher: divan::Bencher,
    fb_w: usize,
    fb_h: usize,
    pixels: usize,
    mut op: impl FnMut(&ShapeBe, &mut axnn::Tensor<axnn_cpu::CpuStorage<'_>>),
) {
    let be = ShapeBe::new();
    let mut fb = vec![0u8; fb_w * fb_h * 4];
    let desc = TensorDesc::new(shape![fb_h, fb_w], DType::I32);
    bencher
        .counter(BytesCount::new(pixels * 4))
        .bench_local(|| {
            let mut view = be.wrap_mut(&mut fb, &desc).unwrap();
            op(&be, divan::black_box(&mut view));
        });
}

#[divan::bench]
fn rounded_fill_12x12_r6(bencher: divan::Bencher) {
    let shape = RoundedShape::uniform(2, 2, 12, 12, 6);
    bench_shape(bencher, 16, 16, 12 * 12, |be, fb| {
        be.rounded_fill(fb, 0xff40_80c0, shape).unwrap()
    });
}

#[divan::bench]
fn rounded_fill_158x200_r6(bencher: divan::Bencher) {
    let shape = RoundedShape::uniform(4, 4, 158, 200, 6);
    bench_shape(bencher, 166, 208, 158 * 200, |be, fb| {
        be.rounded_fill(fb, 0xff40_80c0, shape).unwrap()
    });
}

#[divan::bench]
fn rounded_fill_158x200_r6_translucent(bencher: divan::Bencher) {
    let shape = RoundedShape::uniform(4, 4, 158, 200, 6);
    bench_shape(bencher, 166, 208, 158 * 200, |be, fb| {
        be.rounded_fill(fb, 0x8020_4060, shape).unwrap()
    });
}

#[divan::bench]
fn rounded_fill_1280x720_r0(bencher: divan::Bencher) {
    let shape = RoundedShape::rect(0, 0, 1280, 720);
    bench_shape(bencher, 1280, 720, 1280 * 720, |be, fb| {
        be.rounded_fill(fb, 0xff40_80c0, shape).unwrap()
    });
}

#[divan::bench]
fn fill_rect_1280x720(bencher: divan::Bencher) {
    bench_shape(bencher, 1280, 720, 1280 * 720, |be, fb| {
        be.fill_rect(fb, 0xff40_80c0).unwrap()
    });
}

#[divan::bench]
fn border_ring_158x200_w1(bencher: divan::Bencher) {
    let shape = RoundedShape::uniform(4, 4, 158, 200, 6);
    bench_shape(bencher, 166, 208, 158 * 200, |be, fb| {
        be.border_ring(fb, 0xffe0_e0e0, shape, 1).unwrap()
    });
}

#[divan::bench]
fn shadow_fill_158x200_r12_blur16(bencher: divan::Bencher) {
    let shape = RoundedShape::uniform(20, 20, 158, 200, 12);
    let falloff = axnn::gaussian_falloff::<32>(16);
    bench_shape(bencher, 200, 240, (158 + 32) * (200 + 32), move |be, fb| {
        be.shadow_fill(fb, 0xa000_0000, shape, &falloff).unwrap()
    });
}

fn coverage_bytes(n: usize, bytes_per: usize) -> Vec<u8> {
    (0..n * bytes_per)
        .map(|i| {
            let lane = i / bytes_per;
            match lane % 4 {
                0 => 0,
                1 => 0xff,
                _ => (lane * 41) as u8,
            }
        })
        .collect()
}

fn bench_coverage(bencher: divan::Bencher, dtype: DType) {
    const N: usize = 64;
    let be = ShapeBe::new();
    let mask_bytes = coverage_bytes(N * N, dtype.block_byte_size());
    let mask = be
        .wrap(&mask_bytes, &TensorDesc::new(shape![N, N], dtype))
        .unwrap();
    let mut fb = vec![0u8; N * N * 4];
    let desc = TensorDesc::new(shape![N, N], DType::I32);
    bencher.counter(BytesCount::new(N * N * 4)).bench_local(|| {
        let mut view = be.wrap_mut(&mut fb, &desc).unwrap();
        be.coverage_blend(divan::black_box(&mask), 0x00e0_e8f0, 255, &mut view)
            .unwrap();
    });
}

#[divan::bench]
fn coverage_blend_64x64_u8(bencher: divan::Bencher) {
    bench_coverage(bencher, DType::U8);
}

#[divan::bench]
fn coverage_blend_64x64_u16(bencher: divan::Bencher) {
    bench_coverage(bencher, DType::I16);
}
