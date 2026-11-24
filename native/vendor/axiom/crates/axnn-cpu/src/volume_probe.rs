//! Matched Aftershock-sized volume experiment; no performance CI thresholds.
extern crate std;
use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "release-mode matched small/large UF8 working-set benchmark"]
fn uf8_volume_working_set_probe() {
    // Fixed, named stack owns the entire corpus; no global-allocator volume
    // buffers. Construction and thread setup are excluded from measurements.
    std::thread::Builder::new()
        .name("uf8-volume-probe".into())
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            working_set::<6912>([24, 12, 24]);
            working_set::<2097152>([128; 3]);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[inline(never)]
fn working_set<const N: usize>(dims: [u32; 3]) {
    let bytes: [[u8; 3]; N] =
        core::array::from_fn(|i| core::array::from_fn(|c| ((i * 37 + c * 11) % 248) as u8));
    let floats: [[f32; 3]; N] = core::array::from_fn(|i| bytes[i].map(crate::cast::uf8e5m3_to_f32));
    let words: [[u16; 3]; N] =
        core::array::from_fn(|i| floats[i].map(crate::cast::f32_to_uf16e5m11_rne));
    let geometry = SampleGeometry::new(dims);
    for scattered in [false, true] {
        let mut times = [[0_u128; 51]; 3];
        let mut outputs = [[[0.0_f32; 3]; 4096]; 3];
        for frame in 0..52 {
            // Move through the backing each frame so the large case does not
            // merely warm a small fixed subset of a nominally large resource.
            let points: [[f32; 3]; 4096] = core::array::from_fn(|i| {
                let linear = if scattered {
                    i.wrapping_mul(2654435761)
                } else {
                    i
                };
                let p = (linear + frame * 7919) % N;
                let nx = dims[0] as usize;
                let ny = dims[1] as usize;
                [
                    (p % nx) as f32 + 0.13,
                    ((p / nx) % ny) as f32 + 0.37,
                    (p / (nx * ny)) as f32 + 0.79,
                ]
            });
            for offset in 0..3 {
                let mode = (frame + offset) % 3;
                let started = Instant::now();
                for (out, &point) in outputs[mode].iter_mut().zip(black_box(&points)) {
                    *out = match mode {
                        0 => black_box(floats.as_slice()).sample::<true>(geometry, point),
                        1 => black_box(words.as_slice()).sample::<true>(geometry, point),
                        _ => TierUf8(black_box(bytes.as_slice())).sample::<true>(geometry, point),
                    };
                }
                black_box(&outputs[mode]);
                if frame > 0 {
                    times[mode][frame - 1] = started.elapsed().as_nanos();
                }
            }
            for i in 0..4096 {
                for mode in 1..3 {
                    assert_eq!(
                        outputs[0][i].map(f32::to_bits),
                        outputs[mode][i].map(f32::to_bits)
                    );
                }
            }
        }
        for t in &mut times {
            t.sort_unstable();
        }
        std::eprintln!(
            "UF8 volume dims={dims:?} scattered={scattered} queries=4096 ns [F32,E5M11,E5M3] bytes={:?} p50={:?} p95={:?} p99={:?} max={:?}",
            [N * 12, N * 6, N * 3],
            times.map(|t| t[25]),
            times.map(|t| t[48]),
            times.map(|t| t[50]),
            times.map(|t| t[50])
        );
    }
}

#[test]
fn direct_volume_candidate_preserves_finite_words_and_corner_order() {
    let geometry = SampleGeometry::new([2, 2, 2]);
    for first in (0..=0xf7ff_u16).step_by(24) {
        let rgb: [[u16; 3]; 8] = core::array::from_fn(|i| {
            core::array::from_fn(|c| (first as usize + i * 3 + c).min(0xf7ff) as u16)
        });
        let floats = rgb.map(|p| p.map(crate::uf::uf16e5m11_to_f32));
        for point in [
            [0.0; 3],
            [1.0; 3],
            [0.13, 0.79, 0.37],
            [f32::NEG_INFINITY, f32::INFINITY, 0.5],
        ] {
            let expected = floats
                .as_slice()
                .sample::<false>(geometry, point)
                .map(f32::to_bits);
            assert_eq!(
                sample_uf16_direct::<true>(&rgb, geometry, point).map(f32::to_bits),
                expected
            );
            assert_eq!(
                sample_uf16_direct::<false>(&rgb, geometry, point).map(f32::to_bits),
                expected
            );
        }
    }
    assert!(
        sample_uf16_direct::<true>(&[[0; 3]; 8], geometry, [f32::NAN; 3])
            .iter()
            .all(|v| v.is_nan())
    );
}

#[test]
#[ignore = "release-mode volume sampling benchmark"]
fn aftershock_volume_sampling_probe() {
    let words: [[u16; 3]; 6912] = core::array::from_fn(|i| {
        core::array::from_fn(|c| {
            crate::uf::f32_to_uf16e5m11_rne(((i * 37 + c * 13) % 1021) as f32 / 127.0)
        })
    });
    // Quantized F32 mirror isolates storage/conversion cost; both interpolate
    // exactly the same representable values. No encoding occurs in this timer.
    let floats = words.map(|p| p.map(crate::uf::uf16e5m11_to_f32));
    let geometry = SampleGeometry::new([24, 12, 24]);
    for scattered in [false, true] {
        let points: [[f32; 3]; 4096] = core::array::from_fn(|i| {
            let i = if scattered {
                i.wrapping_mul(2654435761) % 6912
            } else {
                i % 6912
            };
            [
                (i % 24) as f32 + 0.13,
                ((i / 24) % 12) as f32 + 0.37,
                (i / 288) as f32 + 0.79,
            ]
        });
        let mut outputs = [[[0_f32; 3]; 4096]; 3];
        let mut times = [[0_u128; 101]; 3];
        for frame in 0..102 {
            for offset in 0..3 {
                let mode = (frame + offset) % 3;
                let started = Instant::now();
                for (out, &point) in outputs[mode].iter_mut().zip(black_box(&points)) {
                    *out = match mode {
                        0 => black_box(floats.as_slice()).sample::<true>(geometry, point),
                        1 => sample_uf16_gathered::<true>(black_box(&words), geometry, point),
                        _ => sample_uf16_direct::<true>(black_box(&words), geometry, point),
                    };
                }
                black_box(&outputs[mode]);
                if frame > 0 {
                    times[mode][frame - 1] = started.elapsed().as_nanos();
                }
            }
        }
        for i in 0..4096 {
            for mode in 1..3 {
                assert_eq!(
                    outputs[0][i].map(f32::to_bits),
                    outputs[mode][i].map(f32::to_bits)
                );
            }
        }
        for t in &mut times {
            t.sort_unstable();
        }
        std::eprintln!(
            "volume sampler scattered={scattered} queries=4096 [F32, E5M11 gather, E5M11 direct] ns p50={:?} p95={:?} p99={:?} max={:?}",
            times.map(|t| t[50]),
            times.map(|t| t[95]),
            times.map(|t| t[99]),
            times.map(|t| t[100])
        );
    }
}
