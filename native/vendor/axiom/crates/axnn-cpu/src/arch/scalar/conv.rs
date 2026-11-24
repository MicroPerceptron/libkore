//! 1D convolution kernels — scalar fallback.

use crate::floats::Float;

/// General 1D convolution (per-channel, no padding).
///
/// input:  [channels, in_len]
/// kernel: [channels, kernel_len]
/// output: [channels, out_len]   where out_len = in_len - kernel_len + 1
fn conv_1d<T: Float>(
    input: &[T],
    kernel: &[T],
    output: &mut [T],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    let out_len = in_len - kernel_len + 1;
    debug_assert_eq!(input.len(), channels * in_len);
    debug_assert_eq!(kernel.len(), channels * kernel_len);
    debug_assert_eq!(output.len(), channels * out_len);

    for ch in 0..channels {
        let inp = &input[ch * in_len..][..in_len];
        let ker = &kernel[ch * kernel_len..][..kernel_len];
        let out = &mut output[ch * out_len..][..out_len];
        for i in 0..out_len {
            let mut acc = T::ZERO;
            for j in 0..kernel_len {
                acc += inp[i + j] * ker[j];
            }
            out[i] = acc;
        }
    }
}

/// Short depthwise SSM convolution.
///
/// Optimised path for small kernel sizes (typically d_conv=4).
/// Same math as `conv_1d` but the short, fixed kernel width
/// is the hot case for LFM2 ShortConv layers.
///
/// input:  [d_model, seq_len]
/// kernel: [d_model, d_conv]
/// output: [d_model, seq_len - d_conv + 1]
fn ssm_conv<T: Float>(
    input: &[T],
    kernel: &[T],
    output: &mut [T],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    conv_1d(input, kernel, output, d_model, seq_len, d_conv);
}

// ── F32 / F64 wrappers ───────────────────────────────────────────────

pub(crate) fn conv_1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    conv_1d(input, kernel, output, channels, in_len, kernel_len)
}
pub(crate) fn ssm_conv_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    ssm_conv(input, kernel, output, d_model, seq_len, d_conv)
}

pub(crate) fn conv_1d_f64(
    input: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    channels: usize,
    in_len: usize,
    kernel_len: usize,
) {
    conv_1d(input, kernel, output, channels, in_len, kernel_len)
}
pub(crate) fn ssm_conv_f64(
    input: &[f64],
    kernel: &[f64],
    output: &mut [f64],
    d_model: usize,
    seq_len: usize,
    d_conv: usize,
) {
    ssm_conv(input, kernel, output, d_model, seq_len, d_conv)
}

// ── F16 (packed &[u8], compute through f32) ──────────────────────────

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

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;
        for i in 0..out_len {
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

// ── BF16 (packed &[u8], compute through f32) ─────────────────────────

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

    for ch in 0..channels {
        let inp_base = ch * in_len * 2;
        let ker_base = ch * kernel_len * 2;
        let out_base = ch * out_len * 2;
        for i in 0..out_len {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_1d_single_channel() {
        // input: [1, 2, 3, 4, 5], kernel: [1, 0, -1]
        // out[0] = 1*1 + 2*0 + 3*(-1) = -2
        // out[1] = 2*1 + 3*0 + 4*(-1) = -2
        // out[2] = 3*1 + 4*0 + 5*(-1) = -2
        let input = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let kernel = [1.0, 0.0, -1.0f32];
        let mut output = [0.0f32; 3];
        conv_1d_f32(&input, &kernel, &mut output, 1, 5, 3);
        assert_eq!(output, [-2.0, -2.0, -2.0]);
    }

    #[test]
    fn conv_1d_multi_channel() {
        // 2 channels, in_len=4, kernel_len=2 → out_len=3
        // ch0: [1,2,3,4] * [1,1] → [3,5,7]
        // ch1: [5,6,7,8] * [1,-1] → [-1,-1,-1]
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let kernel = [1.0, 1.0, 1.0, -1.0f32];
        let mut output = [0.0f32; 6];
        conv_1d_f32(&input, &kernel, &mut output, 2, 4, 2);
        assert_eq!(output, [3.0, 5.0, 7.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn conv_1d_identity_kernel() {
        // kernel=[1] → output = input
        let input = [1.0, 2.0, 3.0f32];
        let kernel = [1.0f32];
        let mut output = [0.0f32; 3];
        conv_1d_f32(&input, &kernel, &mut output, 1, 3, 1);
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn ssm_conv_d4() {
        // LFM2-typical: d_model=2, d_conv=4, seq_len=6 → out_len=3
        // ch0: [1,2,3,4,5,6] * [1,0,0,0] → [1,2,3]
        // ch1: [7,8,9,10,11,12] * [0,0,0,1] → [10,11,12]
        let input = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32,
        ];
        let kernel = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0f32];
        let mut output = [0.0f32; 6];
        ssm_conv_f32(&input, &kernel, &mut output, 2, 6, 4);
        assert_eq!(output, [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }
}
