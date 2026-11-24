//! The NVFP4 kernels against a real checkpoint row.
//!
//! Synthetic weights cannot tell whether this crate reads NVIDIA's packing
//! the way NVIDIA wrote it — which nibble is the even element, how the
//! E4M3 block scale and the F32 tensor scale compose. This test takes one
//! expert's up projection out of a published checkpoint, with the expected
//! dot computed independently in F64, and holds the kernels to it.
//!
//! The fixture is built by `tools/nvfp4_fixture.py` and named by
//! `AXNN_NVFP4_FIXTURE`; without it the test reports that it was skipped.
//! Layout, little-endian: `u32` elements, `u32` block, `f32` tensor scale,
//! `f64` expected dot, then the row's packed weights, its E4M3 block
//! scales, and the F32 activations.

use axnn_cpu::_bench::*;

struct Fixture {
    k: usize,
    tensor_scale: f32,
    expected: f64,
    weights: Vec<u8>,
    scales: Vec<u8>,
    x: Vec<f32>,
}

fn fixture() -> Option<Fixture> {
    let path = std::env::var("AXNN_NVFP4_FIXTURE").ok()?;
    let raw = std::fs::read(path).expect("the named fixture is readable");
    let word = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    let (k, block) = (word(0) as usize, word(4) as usize);
    assert_eq!(block, 16, "the kernels scale per sixteen elements");
    let tensor_scale = f32::from_le_bytes(raw[8..12].try_into().unwrap());
    let expected = f64::from_le_bytes(raw[12..20].try_into().unwrap());
    let (weights, rest) = raw[20..].split_at(k / 2);
    let (scales, activations) = rest.split_at(k / block);
    Some(Fixture {
        k,
        tensor_scale,
        expected,
        weights: weights.to_vec(),
        scales: scales.to_vec(),
        x: activations
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    })
}

#[test]
fn kernels_match_a_checkpoint_row_decoded_independently() {
    let Some(f) = fixture() else {
        eprintln!("skipped: set AXNN_NVFP4_FIXTURE to a fixture from tools/nvfp4_fixture.py");
        return;
    };
    assert_eq!(f.x.len(), f.k);

    // The tensor scale is the caller's, as it is for a routed expert.
    let check = |name: &str, dot: f32| {
        let scaled = f64::from(dot) * f64::from(f.tensor_scale);
        let tolerance = 1e-4 * (1.0 + f.expected.abs());
        assert!(
            (scaled - f.expected).abs() <= tolerance,
            "{name}: {scaled} vs {} expected",
            f.expected
        );
    };

    check(
        "portable, table",
        vec_dot_nvfp4_f32(&f.weights, &f.scales, &f.x, f.k),
    );
    check(
        "portable, table-free",
        vec_dot_nvfp4_bits_f32(&f.weights, &f.scales, &f.x, f.k),
    );

    #[cfg(target_arch = "aarch64")]
    check(
        "neon",
        vec_dot_nvfp4_neon_f32(&f.weights, &f.scales, &f.x, f.k),
    );

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        let mut out = [0.0_f32; 1];
        gemv_nvfp4_avx2_f32(&f.weights, &f.scales, &f.x, &mut out);
        check("avx2", out[0]);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        let mut out = [0.0_f32; 1];
        gemv_nvfp4_avx512_f32(&f.weights, &f.scales, &f.x, &mut out);
        check("avx512", out[0]);
    }

    // Quantized activations are an approximation, held to the grid they
    // impose rather than to the exact dot.
    #[cfg(all(target_arch = "x86_64", feature = "nvfp4_vnni"))]
    {
        use axnn_cpu::_bench::nvfp4_q8::*;
        let (bytes, blocks) = quantized_len(f.k);
        let mut biased = vec![0u8; bytes];
        let mut scales = vec![0.0_f32; blocks];
        quantize(&f.x, &mut biased, &mut scales);
        let quantized = Quantized {
            biased: &biased,
            scales: &scales,
        };
        let mut out = [0.0_f32; 1];
        nvfp4_q8_f32(&f.weights, &f.scales, &quantized, &mut out);
        let scaled = f64::from(out[0]) * f64::from(f.tensor_scale);
        assert!(
            (scaled - f.expected).abs() <= 0.02 * (1.0 + f.expected.abs()),
            "vnni w4a8: {scaled} vs {} expected",
            f.expected
        );
    }
}
