//! 1D convolution kernels — AVX2 + FMA.
//!
//! Vectorizes across output positions: for each group of 8 consecutive
//! outputs, broadcast each kernel tap and FMA with the corresponding
//! input window. This gives kernel_len FMA ops per 8 outputs.
//!
//! Hot case: SSM short convolution (d_conv=4, LFM2 ShortConv layers).

use core::arch::x86_64::*;

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

    let chunks = out_len / 8;
    let tail = out_len % 8;

    for ch in 0..channels {
        let inp = &input[ch * in_len..][..in_len];
        let ker = &kernel[ch * kernel_len..][..kernel_len];
        let out = &mut output[ch * out_len..][..out_len];

        unsafe {
            for ci in 0..chunks {
                let i = ci * 8;
                let mut acc = _mm256_setzero_ps();
                for j in 0..kernel_len {
                    let kv = _mm256_set1_ps(ker[j]);
                    let iv = _mm256_loadu_ps(inp.as_ptr().add(i + j));
                    acc = _mm256_fmadd_ps(kv, iv, acc);
                }
                _mm256_storeu_ps(out.as_mut_ptr().add(i), acc);
            }
        }

        // Scalar tail
        let i_start = chunks * 8;
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

    let chunks = out_len / 4;
    let tail = out_len % 4;

    for ch in 0..channels {
        let inp = &input[ch * in_len..][..in_len];
        let ker = &kernel[ch * kernel_len..][..kernel_len];
        let out = &mut output[ch * out_len..][..out_len];

        unsafe {
            for ci in 0..chunks {
                let i = ci * 4;
                let mut acc = _mm256_setzero_pd();
                for j in 0..kernel_len {
                    let kv = _mm256_set1_pd(ker[j]);
                    let iv = _mm256_loadu_pd(inp.as_ptr().add(i + j));
                    acc = _mm256_fmadd_pd(kv, iv, acc);
                }
                _mm256_storeu_pd(out.as_mut_ptr().add(i), acc);
            }
        }

        let i_start = chunks * 4;
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

    let chunks = out_len / 8;
    let tail = out_len % 8;

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;

        unsafe {
            for ci in 0..chunks {
                let i = ci * 8;
                let mut acc = _mm256_setzero_ps();
                for j in 0..kernel_len {
                    let kv = _mm256_set1_ps(crate::cast::read_f16(&kernel[ker_base + j * 2..]));
                    let iv = _mm256_cvtph_ps(_mm_loadu_si128(
                        input.as_ptr().add(inp_base + (i + j) * 2) as *const __m128i,
                    ));
                    acc = _mm256_fmadd_ps(kv, iv, acc);
                }
                let out_ptr = output.as_mut_ptr().add(out_base + i * 2);
                _mm_storeu_si128(
                    out_ptr as *mut __m128i,
                    _mm256_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT),
                );
            }
        }

        let i_start = chunks * 8;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f32;
            for j in 0..kernel_len {
                let iv = crate::cast::read_f16(&input[inp_base + (i + j) * 2..]);
                let kv = crate::cast::read_f16(&kernel[ker_base + j * 2..]);
                acc += iv * kv;
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

    let chunks = out_len / 8;
    let tail = out_len % 8;

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;

        unsafe {
            for ci in 0..chunks {
                let i = ci * 8;
                let mut acc = _mm256_setzero_ps();
                for j in 0..kernel_len {
                    let kv = _mm256_set1_ps(crate::cast::read_bf16(&kernel[ker_base + j * 2..]));
                    // bf16 → f32: zero-extend u16→i32, shift left 16
                    let bits16 = _mm_loadu_si128(
                        input.as_ptr().add(inp_base + (i + j) * 2) as *const __m128i
                    );
                    let iv =
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits16), 16));
                    acc = _mm256_fmadd_ps(kv, iv, acc);
                }
                // f32 → bf16: truncate (shift right 16, pack)
                let out_ptr = output.as_mut_ptr().add(out_base + i * 2);
                let i32s = _mm256_srli_epi32(_mm256_castps_si256(acc), 16);
                let lo128 = _mm256_castsi256_si128(i32s);
                let hi128 = _mm256_extracti128_si256(i32s, 1);
                _mm_storeu_si128(out_ptr as *mut __m128i, _mm_packus_epi32(lo128, hi128));
            }
        }

        let i_start = chunks * 8;
        for i in i_start..i_start + tail {
            let mut acc = 0.0f32;
            for j in 0..kernel_len {
                let iv = crate::cast::read_bf16(&input[inp_base + (i + j) * 2..]);
                let kv = crate::cast::read_bf16(&kernel[ker_base + j * 2..]);
                acc += iv * kv;
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
