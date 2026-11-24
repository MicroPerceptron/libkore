//! Test-only candidate for the final two coverage blends in compose.wgsl.
//! This is NOT graph auto-lowering or a whole-welcome performance claim.
use super::*;
use core::simd::prelude::*;
use kore_rt::{Heap, heap::StaticMemSource};
use std::{hint::black_box, time::Instant};

// The caller proves coverage is finite and in [0, 1] before encoding. Retain
// exact zero/subnormal decoding: font coverage includes zero routinely.
#[inline(always)]
fn unit_e5m11(raw: Simd<u16, 8>) -> Simd<f32, 8> {
    let bits = raw.cast::<u32>();
    let normal = Simd::<f32, 8>::from_bits((bits << Simd::splat(12)) + Simd::splat(112 << 23));
    let subnormal = bits.cast::<f32>() * Simd::splat(2.0_f32.powi(-25));
    bits.simd_lt(Simd::splat(0x800)).select(subnormal, normal)
}

// Const selection gives separately compiled F32 and E5M11 consumers. Neither
// materializes widened planes or intermediate RGB planes. Signed differences
// and the two multiply/add pairs remain F32, matching the native FMA catalog's
// current non-contracted arithmetic. The same packing is used for both paths.
#[inline(never)]
fn blend<const COMPACT: bool>(
    rgb: [&[f32]; 3],
    coverage: [&[f32]; 2],
    words: [&[u16]; 2],
    out: &mut [u32],
) {
    let n = out.len();
    assert!(rgb.iter().chain(coverage.iter()).all(|p| p.len() == n));
    assert!(words.iter().all(|p| p.len() == n));
    let end = n / 8 * 8;
    for i in (0..end).step_by(8) {
        let weights: [Simd<f32, 8>; 2] = core::array::from_fn(|c| {
            if COMPACT {
                unit_e5m11(Simd::from_slice(&words[c][i..]))
            } else {
                Simd::from_slice(&coverage[c][i..])
            }
        });
        let mut packed = Simd::splat(0_u32);
        for (c, (foreground, button)) in [(0.245, 0.99), (0.12, 0.97), (0.055, 0.91)]
            .into_iter()
            .enumerate()
        {
            let base = Simd::from_slice(&rgb[c][i..]);
            let first = (Simd::splat(foreground) - base) * weights[0] + base;
            let second = (Simd::splat(button) - first) * weights[1] + first;
            let channel = (second.simd_clamp(Simd::splat(0.0), Simd::splat(1.0))
                * Simd::splat(255.0)
                + Simd::splat(0.5))
            .cast::<u32>();
            packed |= channel << Simd::splat((2 - c) as u32 * 8);
        }
        packed.copy_to_slice(&mut out[i..i + 8]);
    }
    for (i, pixel) in out.iter_mut().enumerate().skip(end) {
        let weights = core::array::from_fn(|c| {
            if COMPACT {
                axnn_cpu::uf::uf16e5m11_to_f32(words[c][i])
            } else {
                coverage[c][i]
            }
        });
        *pixel = reference(rgb.map(|p| p[i]), weights);
    }
}

fn reference(rgb: [f32; 3], weights: [f32; 2]) -> u32 {
    let mut packed = 0;
    for (c, (foreground, button)) in [(0.245, 0.99), (0.12, 0.97), (0.055, 0.91)]
        .into_iter()
        .enumerate()
    {
        let first = (foreground - rgb[c]) * weights[0] + rgb[c];
        let second = (button - first) * weights[1] + first;
        packed |= ((second.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) << ((2 - c) * 8);
    }
    packed
}

#[test]
fn fused_coverage_decode_and_tail_match_scalar() {
    // Every E5M11 encoding in [0,1], including every subnormal and zero.
    for first in (0..=0x7800_u16).step_by(8) {
        let raw = Simd::from_array(core::array::from_fn(|i| (first + i as u16).min(0x7800)));
        let actual = unit_e5m11(raw).to_array();
        for (bits, value) in raw.to_array().into_iter().zip(actual) {
            assert_eq!(
                value.to_bits(),
                axnn_cpu::uf::uf16e5m11_to_f32(bits).to_bits()
            );
        }
    }
    let colors = [0.73_f32; 23];
    let weights: [f32; 23] = core::array::from_fn(|i| i as f32 / 22.0);
    let encoded = weights.map(axnn_cpu::uf::f32_to_uf16e5m11_rne);
    for n in 0..=23 {
        let mut exact = [0_u32; 23];
        let mut compact = exact;
        blend::<false>(
            [&colors[..n]; 3],
            [&weights[..n]; 2],
            [&encoded[..n]; 2],
            &mut exact[..n],
        );
        blend::<true>(
            [&colors[..n]; 3],
            [&weights[..n]; 2],
            [&encoded[..n]; 2],
            &mut compact[..n],
        );
        for i in 0..n {
            assert_eq!(exact[i], reference([colors[i]; 3], [weights[i]; 2]));
            assert_eq!(
                compact[i],
                reference(
                    [colors[i]; 3],
                    [axnn_cpu::uf::uf16e5m11_to_f32(encoded[i]); 2]
                )
            );
        }
    }
}

#[test]
#[ignore = "release-mode fused coverage experiment; prints measurements"]
fn welcome_fused_coverage_probe() {
    #[repr(align(4096))]
    struct Storage([u8; 96 * 1024 * 1024]);
    std::thread::Builder::new().stack_size(256 * 1024 * 1024).spawn(|| {
        let mut storage = Storage([0; 96 * 1024 * 1024]);
        // SAFETY: all allocations die before this named backing is released.
        let heap = Heap::new_in(unsafe { StaticMemSource::over(storage.0.as_mut_ptr(), storage.0.len()) });
        const TILE: usize = TILE_WIDTH * TILE_HEIGHT;
        // All 570 padded tiles of 1920x1200, starting near the central card.
        // Smaller prefixes study resident working sets, not smaller displays.
        const N: usize = 570 * TILE;
        let mut rgb: [Vec<f32, _>; 3] = core::array::from_fn(|_| Vec::new_in(&heap));
        let mut coverage: [Vec<f32, _>; 2] = core::array::from_fn(|_| Vec::new_in(&heap));
        let mut words: [Vec<u16, _>; 2] = core::array::from_fn(|_| Vec::new_in(&heap));
        let mut output: [Vec<u32, _>; 2] = core::array::from_fn(|_| Vec::new_in(&heap));
        for (c, plane) in rgb.iter_mut().enumerate() {
            plane.resize(N, 0.0);
            for (i, value) in plane.iter_mut().enumerate() { *value = ((i * (c + 1) % 1021) as f32) / 1020.0; }
        }
        for plane in &mut coverage { plane.resize(N, 0.0); }
        for plane in &mut words { plane.resize(N, 0); }
        for plane in &mut output { plane.resize(N, 0); }
        let mut input = axboot::ComposeInputs::new_in(&heap).unwrap();
        let viewport = Viewport::new(1920, 1200).unwrap();
        // The coverage under study is the overlay's; the hole under it is
        // irrelevant here, so a black hole frame stands in.
        let mut hole = Vec::new_in(&heap);
        hole.resize(1920 * 1200, 0_u32);
        let image = HoleImage { pixels: &hole, width: 1920, height: 1200 };
        for tile in 0..570 {
            let position = (tile + 277) % 570;
            input.prepare(viewport, [(position % 15 * TILE_WIDTH) as u32, (position / 15 * TILE_HEIGHT) as u32],
                Clock::from_unix_seconds(Some(76740)), Hole::Image(image)).unwrap();
            for c in 0..2 { coverage[c][tile*TILE..(tile+1)*TILE].copy_from_slice(&input.ink[c]); }
        }
        assert!(coverage.iter().flatten().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        // Coverage is real, base RGB is a synthetic bounded sweep, not the
        // actual c7 frontier. This isolates this region's cost/error only.
        for n in [TILE, 16*TILE, N] {
            let mut samples = [[0_u128; 11]; 4];
            for sample in 0..12 {
                for offset in 0..4 {
                    let mode = (sample + offset) % 4;
                    // Consumer-only modes use pre-encoded storage. Encoding
                    // happens outside their timer; the producer-inclusive
                    // control below includes it and its memory writes.
                    for c in 0..2 { for i in 0..n { words[c][i] = axnn_cpu::uf::f32_to_uf16e5m11_rne(coverage[c][i]); } }
                    let started = Instant::now();
                    for _ in 0..8 {
                        if mode >= 2 {
                            for c in 0..2 { for i in 0..n { words[c][i] = axnn_cpu::uf::f32_to_uf16e5m11_rne(black_box(coverage[c][i])); } }
                        }
                        let repeats = if mode == 3 { 4 } else { 1 };
                        for _ in 0..repeats {
                            let src = rgb.each_ref().map(|p| black_box(&p[..n]));
                            let cov = coverage.each_ref().map(|p| black_box(&p[..n]));
                            let enc = words.each_ref().map(|p| black_box(&p[..n]));
                            if mode == 0 { blend::<false>(src, cov, enc, &mut output[0][..n]); }
                            else { blend::<true>(src, cov, enc, &mut output[1][..n]); }
                            black_box(&output[usize::from(mode != 0)][..n]);
                        }
                    }
                    if sample > 0 { samples[mode][sample-1] = started.elapsed().as_nanos() / if mode == 3 { 32 } else { 8 }; }
                }
            }
            let mut max_error = 0;
            let mut changed = 0;
            for i in 0..n {
                assert_eq!(output[0][i], reference(rgb.each_ref().map(|p| p[i]), coverage.each_ref().map(|p| p[i])));
                assert_eq!(output[1][i], reference(rgb.each_ref().map(|p| p[i]), words.each_ref().map(|p| axnn_cpu::uf::uf16e5m11_to_f32(p[i]))));
                changed += usize::from(output[0][i] != output[1][i]);
                for shift in [0,8,16] { max_error = max_error.max(((output[0][i] >> shift)&255).abs_diff((output[1][i] >> shift)&255)); }
            }
            assert!(max_error <= 1);
            for v in &mut samples { v.sort_unstable(); }
            std::eprintln!("fused coverage arch={} pixels={n} median_ns [F32, E5M11 consumer, encode+consumer, encode amortized/4]={:?} min={:?} max={:?} RGB_error={max_error} changed={changed}",
                std::env::consts::ARCH, samples.map(|v| v[5]), samples.map(|v| v[0]), samples.map(|v| v[10]));
        }
    }).unwrap().join().unwrap();
}
