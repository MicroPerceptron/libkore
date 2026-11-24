//! 1D convolution kernels — portable `core::simd`.
//!
//! Vectorizes across output positions: for each group of W consecutive
//! outputs, broadcast each kernel tap and FMA with the corresponding
//! input window.  This gives kernel_len FMA ops per W outputs.
//!
//! Hot case: SSM short convolution (d_conv=4, LFM2 ShortConv layers).

use core::simd::prelude::*;

use super::W;
const WD: usize = super::WD;

// ── F32 ──────────────────────────────────────────────────────────────

pub(crate) fn conv_1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    let out_len = in_len - kernel_len + 1;
    debug_assert_eq!(input.len(), channels * in_len);
    debug_assert_eq!(kernel.len(), channels * kernel_len);
    debug_assert_eq!(output.len(), channels * out_len);

    let chunks = out_len / W;
    let tail = out_len % W;

    for ch in 0..channels {
        let inp = &input[ch * in_len..][..in_len];
        let ker = &kernel[ch * kernel_len..][..kernel_len];
        let out = &mut output[ch * out_len..][..out_len];

        for ci in 0..chunks {
            let i = ci * W;
            let mut acc = Simd::<f32, W>::splat(0.0);
            for j in 0..kernel_len {
                let kv = Simd::splat(ker[j]);
                let iv = Simd::from_slice(&inp[i + j..]);
                acc = kv * iv + acc;
            }
            acc.copy_to_slice(&mut out[i..]);
        }

        let i_start = chunks * W;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f32;
            for j in 0..kernel_len {
                acc += inp[i + j] * ker[j];
            }
            out[i] = acc;
        }
    }
}

pub(crate) fn ssm_conv_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    conv_1d_f32(input, kernel, output, d_model, seq_len, d_conv);
}

// ── F64 ──────────────────────────────────────────────────────────────

pub(crate) fn conv_1d_f64(
    input: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    let out_len = in_len - kernel_len + 1;
    debug_assert_eq!(input.len(), channels * in_len);
    debug_assert_eq!(kernel.len(), channels * kernel_len);
    debug_assert_eq!(output.len(), channels * out_len);

    let chunks = out_len / WD;
    let tail = out_len % WD;

    for ch in 0..channels {
        let inp = &input[ch * in_len..][..in_len];
        let ker = &kernel[ch * kernel_len..][..kernel_len];
        let out = &mut output[ch * out_len..][..out_len];

        for ci in 0..chunks {
            let i = ci * WD;
            let mut acc = Simd::<f64, WD>::splat(0.0);
            for j in 0..kernel_len {
                let kv = Simd::splat(ker[j]);
                let iv = Simd::from_slice(&inp[i + j..]);
                acc = kv * iv + acc;
            }
            acc.copy_to_slice(&mut out[i..]);
        }

        let i_start = chunks * WD;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f64;
            for j in 0..kernel_len {
                acc += inp[i + j] * ker[j];
            }
            out[i] = acc;
        }
    }
}

pub(crate) fn ssm_conv_f64(
    input: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    conv_1d_f64(input, kernel, output, d_model, seq_len, d_conv);
}

// ── F16 (packed &[u8], compute in f32) ───────────────────────────────

pub(crate) fn conv_1d_f16(
    input: &[u8],
    kernel: &[u8],
    output: &mut [u8],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    let out_len = in_len - kernel_len + 1;
    debug_assert_eq!(input.len(), channels * in_len * 2);
    debug_assert_eq!(kernel.len(), channels * kernel_len * 2);
    debug_assert_eq!(output.len(), channels * out_len * 2);

    let chunks = out_len / W;
    let tail = out_len % W;

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;

        for ci in 0..chunks {
            let i = ci * W;
            let mut acc = Simd::<f32, W>::splat(0.0);
            for j in 0..kernel_len {
                let kv = Simd::splat(crate::cast::read_f16(&kernel[ker_base + j * 2..]));
                let iv = Simd::from_array(core::array::from_fn(|lane| {
                    crate::cast::read_f16(&input[inp_base + (i + j + lane) * 2..])
                }));
                acc = kv * iv + acc;
            }
            let a = acc.as_array();
            for lane in 0..W {
                crate::cast::write_f16(&mut output[out_base + (i + lane) * 2..], a[lane]);
            }
        }

        let i_start = chunks * W;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f32;
            for j in 0..kernel_len {
                acc += crate::cast::read_f16(&input[inp_base + (i + j) * 2..])
                    * crate::cast::read_f16(&kernel[ker_base + j * 2..]);
            }
            crate::cast::write_f16(&mut output[out_base + i * 2..], acc);
        }
    }
}

pub(crate) fn ssm_conv_f16(
    input: &[u8],
    kernel: &[u8],
    output: &mut [u8],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    conv_1d_f16(input, kernel, output, d_model, seq_len, d_conv);
}

// ── BF16 (packed &[u8], compute in f32) ──────────────────────────────

pub(crate) fn conv_1d_bf16(
    input: &[u8],
    kernel: &[u8],
    output: &mut [u8],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    let out_len = in_len - kernel_len + 1;
    debug_assert_eq!(input.len(), channels * in_len * 2);
    debug_assert_eq!(kernel.len(), channels * kernel_len * 2);
    debug_assert_eq!(output.len(), channels * out_len * 2);

    let chunks = out_len / W;
    let tail = out_len % W;

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;

        for ci in 0..chunks {
            let i = ci * W;
            let mut acc = Simd::<f32, W>::splat(0.0);
            for j in 0..kernel_len {
                let kv = Simd::splat(crate::cast::read_bf16(&kernel[ker_base + j * 2..]));
                let iv = Simd::from_array(core::array::from_fn(|lane| {
                    crate::cast::read_bf16(&input[inp_base + (i + j + lane) * 2..])
                }));
                acc = kv * iv + acc;
            }
            let a = acc.as_array();
            for lane in 0..W {
                crate::cast::write_bf16(&mut output[out_base + (i + lane) * 2..], a[lane]);
            }
        }

        let i_start = chunks * W;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f32;
            for j in 0..kernel_len {
                acc += crate::cast::read_bf16(&input[inp_base + (i + j) * 2..])
                    * crate::cast::read_bf16(&kernel[ker_base + j * 2..]);
            }
            crate::cast::write_bf16(&mut output[out_base + i * 2..], acc);
        }
    }
}

pub(crate) fn ssm_conv_bf16(
    input: &[u8],
    kernel: &[u8],
    output: &mut [u8],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    conv_1d_bf16(input, kernel, output, d_model, seq_len, d_conv);
}
