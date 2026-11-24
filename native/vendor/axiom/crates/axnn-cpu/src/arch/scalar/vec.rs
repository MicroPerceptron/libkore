//! Element-wise vector operations — scalar fallback.
//!
//! Unary: silu, gelu, relu, tanh, sigmoid, exp, neg, sqrt
//! Binary: add, sub, mul, div, scale
//! Raster: cmpge_f32, and_masks, maskstore_u32

use crate::floats::Float;

// ── Generic implementations (private) ─────────────────────────────────

fn silu<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x / (T::ONE + (-x).exp());
    }
}

fn gelu<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    let c = T::from_f64(0.797_884_560_802_865_4); // sqrt(2/π)
    let k = T::from_f64(0.044715);
    let half = T::from_f64(0.5);
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = half * x * (T::ONE + (c * (x + k * x * x * x)).tanh());
    }
}

fn relu<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = if s > T::ZERO { s } else { T::ZERO };
    }
}

fn tanh_op<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x.tanh();
    }
}

fn sigmoid<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = T::ONE / (T::ONE + (-x).exp());
    }
}

fn exp_op<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x.exp();
    }
}

fn neg<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = -s;
    }
}

fn sqrt_op<T: Float>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x.sqrt();
    }
}

fn add<T: Float>(a: &[T], b: &[T], dst: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av + bv;
    }
}

fn mul<T: Float>(a: &[T], b: &[T], dst: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av * bv;
    }
}

fn sub<T: Float>(a: &[T], b: &[T], dst: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av - bv;
    }
}

fn div<T: Float>(a: &[T], b: &[T], dst: &mut [T]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av / bv;
    }
}

fn scale<T: Float>(src: &[T], dst: &mut [T], s: T) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = v * s;
    }
}

// ── F32 wrappers ──────────────────────────────────────────────────────

pub(crate) fn silu_f32(src: &[f32], dst: &mut [f32]) {
    silu(src, dst)
}
pub(crate) fn gelu_f32(src: &[f32], dst: &mut [f32]) {
    gelu(src, dst)
}
pub(crate) fn relu_f32(src: &[f32], dst: &mut [f32]) {
    relu(src, dst)
}
pub(crate) fn tanh_f32(src: &[f32], dst: &mut [f32]) {
    tanh_op(src, dst)
}
pub(crate) fn sigmoid_f32(src: &[f32], dst: &mut [f32]) {
    sigmoid(src, dst)
}
pub(crate) fn exp_f32(src: &[f32], dst: &mut [f32]) {
    exp_op(src, dst)
}
pub(crate) fn neg_f32(src: &[f32], dst: &mut [f32]) {
    neg(src, dst)
}
pub(crate) fn sqrt_f32(src: &[f32], dst: &mut [f32]) {
    sqrt_op(src, dst)
}
pub(crate) fn add_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    add(a, b, dst)
}
pub(crate) fn sub_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    sub(a, b, dst)
}
pub(crate) fn mul_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    mul(a, b, dst)
}
pub(crate) fn div_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    div(a, b, dst)
}
pub(crate) fn scale_f32(src: &[f32], dst: &mut [f32], s: f32) {
    scale(src, dst, s)
}

// ── F64 wrappers ──────────────────────────────────────────────────────

pub(crate) fn silu_f64(src: &[f64], dst: &mut [f64]) {
    silu(src, dst)
}
pub(crate) fn gelu_f64(src: &[f64], dst: &mut [f64]) {
    gelu(src, dst)
}
pub(crate) fn relu_f64(src: &[f64], dst: &mut [f64]) {
    relu(src, dst)
}
pub(crate) fn tanh_f64(src: &[f64], dst: &mut [f64]) {
    tanh_op(src, dst)
}
pub(crate) fn sigmoid_f64(src: &[f64], dst: &mut [f64]) {
    sigmoid(src, dst)
}
pub(crate) fn exp_f64(src: &[f64], dst: &mut [f64]) {
    exp_op(src, dst)
}
pub(crate) fn neg_f64(src: &[f64], dst: &mut [f64]) {
    neg(src, dst)
}
pub(crate) fn sqrt_f64(src: &[f64], dst: &mut [f64]) {
    sqrt_op(src, dst)
}
pub(crate) fn add_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    add(a, b, dst)
}
pub(crate) fn sub_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    sub(a, b, dst)
}
pub(crate) fn mul_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    mul(a, b, dst)
}
pub(crate) fn div_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    div(a, b, dst)
}
pub(crate) fn scale_f64(src: &[f64], dst: &mut [f64], s: f64) {
    scale(src, dst, s)
}

// ── Raster primitives ─────────────────────────────────────────────────
//
// These three functions form the inner-loop of a Pineda edge-function
// rasterizer.  They are deliberately typed (f32 in, u32 mask) rather
// than polymorphic.
//
// Typical call sequence per 8-pixel tile column:
//   cmpge_f32(e0, zero, m0);
//   cmpge_f32(e1, zero, m1);
//   cmpge_f32(e2, zero, m2);
//   and_masks(m0, m1, m01);
//   and_masks(m01, m2, coverage);
//   maskstore_u32(coverage, color, framebuffer_tile);

/// For each lane: dst[i] = if a[i] >= b[i] { 0xFFFF_FFFF } else { 0 }
pub(crate) fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = if av >= bv { 0xFFFF_FFFF } else { 0 };
    }
}

/// Bitwise AND of two u32 mask slices.
pub(crate) fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av & bv;
    }
}

/// Conditionally write src[i] → dst[i] where mask[i] != 0.
pub(crate) fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(mask.len(), src.len());
    debug_assert_eq!(mask.len(), dst.len());
    for ((d, &m), &s) in dst.iter_mut().zip(mask).zip(src) {
        if m != 0 {
            *d = s;
        }
    }
}

/// Fused 3-edge coverage test + conditional framebuffer write.
///
/// Equivalent to:
///   cmpge_f32(e0, thresh, m0); cmpge_f32(e1, thresh, m1); cmpge_f32(e2, thresh, m2);
///   and_masks(m0, m1, t); and_masks(t, m2, coverage);
///   maskstore_u32(coverage, color, fb);
///
/// but without materialising the intermediate mask buffers.
pub(crate) fn fused_edge_coverage(
    e0: &[f32],
    e1: &[f32],
    e2: &[f32],
    thresh: f32,
    color: &[u32],
    fb: &mut [u32],
) {
    let n = fb.len();
    debug_assert_eq!(e0.len(), n);
    debug_assert_eq!(e1.len(), n);
    debug_assert_eq!(e2.len(), n);
    debug_assert_eq!(color.len(), n);
    for i in 0..n {
        if e0[i] >= thresh && e1[i] >= thresh && e2[i] >= thresh {
            fb[i] = color[i];
        }
    }
}

// ── F16 (packed &[u8], compute through f32) ───────────────────────────

pub(crate) fn silu_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_f16(s);
        crate::cast::write_f16(d, axmath::silu_f32(x));
    }
}

pub(crate) fn gelu_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_f16(s);
        crate::cast::write_f16(d, axmath::gelu_f32(x));
    }
}

pub(crate) fn relu_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_f16(s);
        crate::cast::write_f16(d, if x > 0.0 { x } else { 0.0 });
    }
}

pub(crate) fn tanh_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_f16(d, axmath::tanh_f32(crate::cast::read_f16(s)));
    }
}

pub(crate) fn sigmoid_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_f16(s);
        crate::cast::write_f16(d, axmath::sigmoid_f32(x));
    }
}

pub(crate) fn exp_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_f16(d, axmath::exp_f32(crate::cast::read_f16(s)));
    }
}

pub(crate) fn neg_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_f16(d, -crate::cast::read_f16(s));
    }
}

pub(crate) fn sqrt_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_f16(d, axmath::sqrt_f32(crate::cast::read_f16(s)));
    }
}

pub(crate) fn add_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_f16(d, crate::cast::read_f16(sa) + crate::cast::read_f16(sb));
    }
}

pub(crate) fn mul_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_f16(d, crate::cast::read_f16(sa) * crate::cast::read_f16(sb));
    }
}

pub(crate) fn div_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_f16(d, crate::cast::read_f16(sa) / crate::cast::read_f16(sb));
    }
}

pub(crate) fn scale_f16(src: &[u8], dst: &mut [u8], s: f32) {
    debug_assert_eq!(src.len(), dst.len());
    for (sc, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_f16(d, crate::cast::read_f16(sc) * s);
    }
}

// ── BF16 (packed &[u8], compute through f32) ─────────────────────────

pub(crate) fn silu_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_bf16(s);
        crate::cast::write_bf16(d, axmath::silu_f32(x));
    }
}

pub(crate) fn gelu_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_bf16(s);
        crate::cast::write_bf16(d, axmath::gelu_f32(x));
    }
}

pub(crate) fn relu_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_bf16(s);
        crate::cast::write_bf16(d, if x > 0.0 { x } else { 0.0 });
    }
}

pub(crate) fn tanh_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_bf16(d, axmath::tanh_f32(crate::cast::read_bf16(s)));
    }
}

pub(crate) fn sigmoid_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        let x = crate::cast::read_bf16(s);
        crate::cast::write_bf16(d, axmath::sigmoid_f32(x));
    }
}

pub(crate) fn exp_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_bf16(d, axmath::exp_f32(crate::cast::read_bf16(s)));
    }
}

pub(crate) fn neg_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_bf16(d, -crate::cast::read_bf16(s));
    }
}

pub(crate) fn sqrt_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_bf16(d, axmath::sqrt_f32(crate::cast::read_bf16(s)));
    }
}

pub(crate) fn add_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_bf16(d, crate::cast::read_bf16(sa) + crate::cast::read_bf16(sb));
    }
}

pub(crate) fn mul_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_bf16(d, crate::cast::read_bf16(sa) * crate::cast::read_bf16(sb));
    }
}

pub(crate) fn div_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, sa), sb) in dst
        .chunks_exact_mut(2)
        .zip(a.chunks_exact(2))
        .zip(b.chunks_exact(2))
    {
        crate::cast::write_bf16(d, crate::cast::read_bf16(sa) / crate::cast::read_bf16(sb));
    }
}

pub(crate) fn scale_bf16(src: &[u8], dst: &mut [u8], s: f32) {
    debug_assert_eq!(src.len(), dst.len());
    for (sc, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
        crate::cast::write_bf16(d, crate::cast::read_bf16(sc) * s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── Unary ────────────────────────────────────────────────

    #[test]
    fn test_silu() {
        let src = [0.0, 1.0, -1.0, 5.0, -5.0];
        let mut dst = [0.0f32; 5];
        silu_f32(&src, &mut dst);
        assert!(close(dst[0], 0.0));
        assert!(close(dst[1], axmath::sigmoid_f32(1.0)));
        assert!(dst[3] > 4.9);
        assert!(dst[4].abs() < 0.04);
    }

    #[test]
    fn test_gelu() {
        let src = [0.0, 3.0, -3.0];
        let mut dst = [0.0f32; 3];
        gelu_f32(&src, &mut dst);
        assert!(close(dst[0], 0.0));
        assert!(dst[1] > 2.99);
        assert!(dst[2].abs() < 0.01);
    }

    #[test]
    fn test_relu() {
        let src = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let mut dst = [0.0f32; 5];
        relu_f32(&src, &mut dst);
        assert_eq!(dst, [0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn test_tanh() {
        let src = [0.0, 1.0, -1.0];
        let mut dst = [0.0f32; 3];
        tanh_f32(&src, &mut dst);
        assert!(close(dst[0], 0.0));
        assert!(close(dst[1], axmath::tanh_f32(1.0)));
        assert!(close(dst[2], axmath::tanh_f32(-1.0)));
    }

    #[test]
    fn test_sigmoid() {
        let src = [0.0, 10.0, -10.0];
        let mut dst = [0.0f32; 3];
        sigmoid_f32(&src, &mut dst);
        assert!(close(dst[0], 0.5));
        assert!(dst[1] > 0.9999);
        assert!(dst[2] < 0.0001);
    }

    #[test]
    fn test_exp() {
        let src = [0.0, 1.0, -1.0];
        let mut dst = [0.0f32; 3];
        exp_f32(&src, &mut dst);
        assert!(close(dst[0], 1.0));
        assert!(close(dst[1], core::f32::consts::E));
        assert!(close(dst[2], 1.0 / core::f32::consts::E));
    }

    #[test]
    fn test_neg() {
        let src = [1.0, -2.0, 0.0, 3.5];
        let mut dst = [0.0f32; 4];
        neg_f32(&src, &mut dst);
        assert_eq!(dst, [-1.0, 2.0, 0.0, -3.5]);
    }

    #[test]
    fn test_sqrt() {
        let src = [0.0, 1.0, 4.0, 9.0, 16.0];
        let mut dst = [0.0f32; 5];
        sqrt_f32(&src, &mut dst);
        assert!(close(dst[0], 0.0));
        assert!(close(dst[1], 1.0));
        assert!(close(dst[2], 2.0));
        assert!(close(dst[3], 3.0));
        assert!(close(dst[4], 4.0));
    }

    // ── Binary ───────────────────────────────────────────────

    #[test]
    fn test_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut dst = [0.0f32; 3];
        add_f32(&a, &b, &mut dst);
        assert_eq!(dst, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut dst = [0.0f32; 3];
        mul_f32(&a, &b, &mut dst);
        assert_eq!(dst, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_div() {
        let a = [10.0, 20.0, 30.0];
        let b = [2.0, 5.0, 10.0];
        let mut dst = [0.0f32; 3];
        div_f32(&a, &b, &mut dst);
        assert_eq!(dst, [5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let src = [1.0, 2.0, 3.0];
        let mut dst = [0.0f32; 3];
        scale_f32(&src, &mut dst, 2.5);
        assert_eq!(dst, [2.5, 5.0, 7.5]);
    }

    // ── Raster primitives ─────────────────────────────────────

    #[test]
    fn test_sub_f32() {
        let a = [3.0f32, 1.0, 0.0, -1.0];
        let b = [1.0f32, 1.0, 2.0, -3.0];
        let mut dst = [0.0f32; 4];
        sub_f32(&a, &b, &mut dst);
        assert_eq!(dst, [2.0, 0.0, -2.0, 2.0]);
    }

    #[test]
    fn test_cmpge_f32_basic() {
        // gt, eq, lt — check exact bit pattern
        let a = [2.0f32, 1.0, 0.0];
        let b = [1.0f32, 1.0, 1.0];
        let mut dst = [0u32; 3];
        cmpge_f32(&a, &b, &mut dst);
        assert_eq!(dst[0], 0xFFFF_FFFF, "gt lane should be all-ones");
        assert_eq!(dst[1], 0xFFFF_FFFF, "eq lane should be all-ones");
        assert_eq!(dst[2], 0x0000_0000, "lt lane should be zero");
    }

    #[test]
    fn test_cmpge_f32_negatives() {
        let a = [-1.0f32, -2.0, -2.0];
        let b = [-2.0f32, -1.0, -2.0];
        let mut dst = [0u32; 3];
        cmpge_f32(&a, &b, &mut dst);
        assert_eq!(dst[0], 0xFFFF_FFFF);
        assert_eq!(dst[1], 0x0000_0000);
        assert_eq!(dst[2], 0xFFFF_FFFF);
    }

    #[test]
    fn test_and_masks() {
        let a = [0xFFFF_FFFFu32, 0xFFFF_FFFF, 0x0000_0000, 0x0000_0000];
        let b = [0xFFFF_FFFFu32, 0x0000_0000, 0xFFFF_FFFF, 0x0000_0000];
        let mut dst = [0u32; 4];
        and_masks(&a, &b, &mut dst);
        assert_eq!(dst, [0xFFFF_FFFF, 0x0000_0000, 0x0000_0000, 0x0000_0000]);
    }

    #[test]
    fn test_maskstore_u32_writes_on_nonzero() {
        let mask = [0xFFFF_FFFFu32, 0x0000_0000, 0xFFFF_FFFF, 0x0000_0000];
        let src = [0xDEAD_BEEFu32; 4];
        let mut dst = [0u32; 4];
        maskstore_u32(&mask, &src, &mut dst);
        assert_eq!(dst[0], 0xDEAD_BEEF, "nonzero mask should write");
        assert_eq!(dst[1], 0x0000_0000, "zero mask should not write");
        assert_eq!(dst[2], 0xDEAD_BEEF, "nonzero mask should write");
        assert_eq!(dst[3], 0x0000_0000, "zero mask should not write");
    }

    #[test]
    fn test_maskstore_u32_preserves_existing() {
        // dst starts with sentinel values; zero-mask lanes must not be clobbered
        let mask = [0u32, 0xFFFF_FFFF];
        let src = [0xAAAA_AAAAu32, 0xBBBB_BBBB];
        let mut dst = [0xCCCC_CCCCu32, 0xCCCC_CCCC];
        maskstore_u32(&mask, &src, &mut dst);
        assert_eq!(dst[0], 0xCCCC_CCCC, "existing value must be preserved");
        assert_eq!(dst[1], 0xBBBB_BBBB, "new value must be written");
    }
}
