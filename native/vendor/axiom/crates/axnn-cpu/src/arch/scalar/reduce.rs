//! Reduction operations — scalar fallback.
//!
//! sum, max, softmax, rms_norm, layer_norm
//!
//! Uses 4-wide accumulators where beneficial to break serial
//! dependency chains and let the CPU pipeline independent adds.

use crate::floats::Float;

// ── Generic implementations (private) ─────────────────────────────────

fn sum<T: Float>(src: &[T]) -> T {
    let mut s0 = T::ZERO;
    let mut s1 = T::ZERO;
    let mut s2 = T::ZERO;
    let mut s3 = T::ZERO;
    let chunks = src.chunks_exact(4);
    let tail = chunks.remainder();
    for c in chunks {
        s0 += c[0];
        s1 += c[1];
        s2 += c[2];
        s3 += c[3];
    }
    for &v in tail {
        s0 += v;
    }
    (s0 + s1) + (s2 + s3)
}

fn max<T: Float>(src: &[T]) -> T {
    debug_assert!(!src.is_empty());
    let mut m = src[0];
    for &v in &src[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

fn argmin<T: Float>(src: &[T]) -> usize {
    debug_assert!(!src.is_empty());
    let mut idx = 0;
    let mut best = src[0];
    for (i, &v) in src.iter().enumerate().skip(1) {
        if v < best {
            best = v;
            idx = i;
        }
    }
    idx
}

fn argmax<T: Float>(src: &[T]) -> usize {
    debug_assert!(!src.is_empty());
    let mut idx = 0;
    let mut best = src[0];
    for (i, &v) in src.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

fn softmax<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    let m = max(src);
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = (s - m).exp();
    }
    let inv = T::ONE / sum(dst);
    for d in dst.iter_mut() {
        *d *= inv;
    }
}

fn rms_norm<T: Float>(src: &[T], dst: &mut [T], weight: &[T], eps: T) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());

    let mut ss = T::ZERO;
    for &v in src {
        ss += v * v;
    }
    let rms_inv = T::ONE / (ss / T::from_usize(n) + eps).sqrt();
    for ((d, &s), &w) in dst.iter_mut().zip(src).zip(weight) {
        *d = s * rms_inv * w;
    }
}

fn layer_norm<T: Float>(src: &[T], dst: &mut [T], weight: &[T], bias: &[T], eps: T) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, bias.len());

    let mean = sum(src) / T::from_usize(n);
    let mut var = T::ZERO;
    for &v in src {
        let d = v - mean;
        var += d * d;
    }
    let inv = T::ONE / (var / T::from_usize(n) + eps).sqrt();
    for (((d, &s), &w), &b) in dst.iter_mut().zip(src).zip(weight).zip(bias) {
        *d = (s - mean) * inv * w + b;
    }
}

// ── F32 wrappers ──────────────────────────────────────────────────────

pub(crate) fn sum_f32(src: &[f32]) -> f32 {
    sum(src)
}
pub(crate) fn max_f32(src: &[f32]) -> f32 {
    max(src)
}
pub(crate) fn argmin_f32(src: &[f32]) -> usize {
    argmin(src)
}
pub(crate) fn argmax_f32(src: &[f32]) -> usize {
    argmax(src)
}
pub(crate) fn softmax_f32(src: &[f32], dst: &mut [f32]) {
    softmax(src, dst)
}
pub(crate) fn rms_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], eps: f32) {
    rms_norm(src, dst, weight, eps)
}
pub(crate) fn layer_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    layer_norm(src, dst, weight, bias, eps)
}

// ── F64 wrappers ──────────────────────────────────────────────────────

pub(crate) fn sum_f64(src: &[f64]) -> f64 {
    sum(src)
}
pub(crate) fn max_f64(src: &[f64]) -> f64 {
    max(src)
}
pub(crate) fn argmin_f64(src: &[f64]) -> usize {
    argmin(src)
}
pub(crate) fn argmax_f64(src: &[f64]) -> usize {
    argmax(src)
}
pub(crate) fn softmax_f64(src: &[f64], dst: &mut [f64]) {
    softmax(src, dst)
}
pub(crate) fn rms_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], eps: f64) {
    rms_norm(src, dst, weight, eps)
}
pub(crate) fn layer_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], bias: &[f64], eps: f64) {
    layer_norm(src, dst, weight, bias, eps)
}

// ── F16 (packed &[u8], accumulate in f32) ─────────────────────────────
//
// Scalar-producing reductions return f32.
// Vector-producing reductions write packed f16 output.

pub(crate) fn sum_f16(src: &[u8]) -> f32 {
    debug_assert_eq!(src.len() % 2, 0);
    let mut acc = 0.0f32;
    for chunk in src.chunks_exact(2) {
        acc += crate::cast::read_f16(chunk);
    }
    acc
}

pub(crate) fn max_f16(src: &[u8]) -> f32 {
    debug_assert!(src.len() >= 2);
    let mut m = crate::cast::read_f16(&src[..2]);
    for chunk in src[2..].chunks_exact(2) {
        let v = crate::cast::read_f16(chunk);
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_f16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    let mut idx = 0;
    let mut best = crate::cast::read_f16(&src[..2]);
    for (i, chunk) in src[2..].chunks_exact(2).enumerate() {
        let v = crate::cast::read_f16(chunk);
        if v < best {
            best = v;
            idx = i + 1;
        }
    }
    idx
}

pub(crate) fn argmax_f16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    let mut idx = 0;
    let mut best = crate::cast::read_f16(&src[..2]);
    for (i, chunk) in src[2..].chunks_exact(2).enumerate() {
        let v = crate::cast::read_f16(chunk);
        if v > best {
            best = v;
            idx = i + 1;
        }
    }
    idx
}

pub(crate) fn softmax_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let m = max_f16(src);
    // Pass 1: write exp(x - max), accumulate sum in f32.
    let mut sum = 0.0f32;
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let e = axmath::exp_f32(crate::cast::read_f16(s) - m);
        crate::cast::write_f16(d, e);
        sum += e;
    }
    // Pass 2: normalize.
    let inv = 1.0 / sum;
    for d in dst.chunks_exact_mut(2) {
        crate::cast::write_f16(d, crate::cast::read_f16(d) * inv);
    }
}

pub(crate) fn rms_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;

    let mut ss = 0.0f32;
    for chunk in src.chunks_exact(2) {
        let v = crate::cast::read_f16(chunk);
        ss += v * v;
    }
    let rms_inv = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
    for ((d, s), w) in dst
        .chunks_exact_mut(2)
        .zip(src.chunks_exact(2))
        .zip(weight.chunks_exact(2))
    {
        crate::cast::write_f16(
            d,
            crate::cast::read_f16(s) * rms_inv * crate::cast::read_f16(w),
        );
    }
}

pub(crate) fn layer_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;

    let mean = sum_f16(src) / n as f32;
    let mut var = 0.0f32;
    for chunk in src.chunks_exact(2) {
        let d = crate::cast::read_f16(chunk) - mean;
        var += d * d;
    }
    let inv = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
    for (((d, s), w), b) in dst
        .chunks_exact_mut(2)
        .zip(src.chunks_exact(2))
        .zip(weight.chunks_exact(2))
        .zip(bias.chunks_exact(2))
    {
        let val = (crate::cast::read_f16(s) - mean) * inv * crate::cast::read_f16(w)
            + crate::cast::read_f16(b);
        crate::cast::write_f16(d, val);
    }
}

// ── BF16 (packed &[u8], accumulate in f32) ────────────────────────────

pub(crate) fn sum_bf16(src: &[u8]) -> f32 {
    debug_assert_eq!(src.len() % 2, 0);
    let mut acc = 0.0f32;
    for chunk in src.chunks_exact(2) {
        acc += crate::cast::read_bf16(chunk);
    }
    acc
}

pub(crate) fn max_bf16(src: &[u8]) -> f32 {
    debug_assert!(src.len() >= 2);
    let mut m = crate::cast::read_bf16(&src[..2]);
    for chunk in src[2..].chunks_exact(2) {
        let v = crate::cast::read_bf16(chunk);
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_bf16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    let mut idx = 0;
    let mut best = crate::cast::read_bf16(&src[..2]);
    for (i, chunk) in src[2..].chunks_exact(2).enumerate() {
        let v = crate::cast::read_bf16(chunk);
        if v < best {
            best = v;
            idx = i + 1;
        }
    }
    idx
}

pub(crate) fn argmax_bf16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    let mut idx = 0;
    let mut best = crate::cast::read_bf16(&src[..2]);
    for (i, chunk) in src[2..].chunks_exact(2).enumerate() {
        let v = crate::cast::read_bf16(chunk);
        if v > best {
            best = v;
            idx = i + 1;
        }
    }
    idx
}

pub(crate) fn softmax_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let m = max_bf16(src);
    let mut sum = 0.0f32;
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let e = axmath::exp_f32(crate::cast::read_bf16(s) - m);
        crate::cast::write_bf16(d, e);
        sum += e;
    }
    let inv = 1.0 / sum;
    for d in dst.chunks_exact_mut(2) {
        crate::cast::write_bf16(d, crate::cast::read_bf16(d) * inv);
    }
}

pub(crate) fn rms_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;

    let mut ss = 0.0f32;
    for chunk in src.chunks_exact(2) {
        let v = crate::cast::read_bf16(chunk);
        ss += v * v;
    }
    let rms_inv = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
    for ((d, s), w) in dst
        .chunks_exact_mut(2)
        .zip(src.chunks_exact(2))
        .zip(weight.chunks_exact(2))
    {
        crate::cast::write_bf16(
            d,
            crate::cast::read_bf16(s) * rms_inv * crate::cast::read_bf16(w),
        );
    }
}

pub(crate) fn layer_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;

    let mean = sum_bf16(src) / n as f32;
    let mut var = 0.0f32;
    for chunk in src.chunks_exact(2) {
        let d = crate::cast::read_bf16(chunk) - mean;
        var += d * d;
    }
    let inv = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
    for (((d, s), w), b) in dst
        .chunks_exact_mut(2)
        .zip(src.chunks_exact(2))
        .zip(weight.chunks_exact(2))
        .zip(bias.chunks_exact(2))
    {
        let val = (crate::cast::read_bf16(s) - mean) * inv * crate::cast::read_bf16(w)
            + crate::cast::read_bf16(b);
        crate::cast::write_bf16(d, val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_sum() {
        assert!(close(sum_f32(&[1.0, 2.0, 3.0, 4.0]), 10.0));
        // Non-chunk-aligned length (remainder path).
        assert!(close(sum_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]), 15.0));
        assert!(close(sum_f32(&[1.0]), 1.0));
    }

    #[test]
    fn test_max() {
        assert_eq!(max_f32(&[1.0, 5.0, 3.0, 2.0]), 5.0);
        assert_eq!(max_f32(&[-3.0, -1.0, -2.0]), -1.0);
        assert_eq!(max_f32(&[42.0]), 42.0);
    }

    #[test]
    fn test_arg_reduce() {
        assert_eq!(argmin_f32(&[3.0, -1.0, -1.0, 2.0]), 1);
        assert_eq!(argmax_f32(&[3.0, 5.0, 4.0, 5.0]), 1);
        assert_eq!(argmin_f64(&[2.0, -4.0, 1.0]), 1);
        assert_eq!(argmax_f64(&[-7.0, -2.0, -3.0]), 1);
        assert_eq!(argmax_f32(&[42.0]), 0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        softmax_f32(&src, &mut dst);
        let total: f32 = dst.iter().sum();
        assert!(close(total, 1.0));
    }

    #[test]
    fn test_softmax_monotonic() {
        let src = [1.0, 2.0, 3.0];
        let mut dst = [0.0f32; 3];
        softmax_f32(&src, &mut dst);
        assert!(dst[0] < dst[1]);
        assert!(dst[1] < dst[2]);
    }

    #[test]
    fn test_softmax_uniform() {
        // Equal inputs → uniform distribution.
        let src = [0.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0f32; 4];
        softmax_f32(&src, &mut dst);
        for &v in &dst {
            assert!(close(v, 0.25));
        }
    }

    #[test]
    fn test_rms_norm_unit_weight() {
        // rms_norm with weight=1 just normalises by RMS.
        let src = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut dst = [0.0f32; 4];
        rms_norm_f32(&src, &mut dst, &weight, 1e-6);

        // RMS = sqrt(mean(1+4+9+16)) = sqrt(30/4) = sqrt(7.5)
        let rms = axmath::sqrt_f32(30.0 / 4.0 + 1e-6);
        for i in 0..4 {
            assert!(close(dst[i], src[i] / rms));
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let src = [3.0, 4.0];
        let weight = [2.0, 0.5];
        let mut dst = [0.0f32; 2];
        rms_norm_f32(&src, &mut dst, &weight, 1e-6);

        let rms = axmath::sqrt_f32((9.0 + 16.0) / 2.0 + 1e-6);
        assert!(close(dst[0], 3.0 / rms * 2.0));
        assert!(close(dst[1], 4.0 / rms * 0.5));
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_var() {
        // For [−1, 1]: mean=0, var=1, so output ≈ weight*src + bias.
        let src = [-1.0, 1.0];
        let weight = [1.0, 1.0];
        let bias = [0.0, 0.0];
        let mut dst = [0.0f32; 2];
        layer_norm_f32(&src, &mut dst, &weight, &bias, 1e-6);

        let var = 1.0f32; // mean of [1, 1] = 1
        let inv = 1.0 / axmath::sqrt_f32(var + 1e-6);
        assert!(close(dst[0], -1.0 * inv));
        assert!(close(dst[1], 1.0 * inv));
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let src = [2.0, 4.0, 6.0, 8.0];
        let weight = [1.0; 4];
        let bias = [10.0; 4];
        let mut dst = [0.0f32; 4];
        layer_norm_f32(&src, &mut dst, &weight, &bias, 1e-6);

        // Mean-centered output shifted by bias=10.
        let mean: f32 = 5.0;
        let centered: Vec<f32> = src.iter().map(|&v| v - mean).collect();
        let var: f32 = centered.iter().map(|&c| c * c).sum::<f32>() / 4.0;
        let inv = 1.0 / axmath::sqrt_f32(var + 1e-6);
        for i in 0..4 {
            assert!(close(dst[i], centered[i] * inv + 10.0));
        }
    }
}
