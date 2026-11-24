//! Tensor operations — user-facing free functions.
//!
//! Each function delegates to the corresponding [`ComputeBackend`] method.
//! The backend is passed as the last argument (allocator/context pattern).
//!
//! ```ignore
//! let y = axnn::relu(&x, &be)?;
//! axnn::matmul(&a, &b, &mut c, &be)?;
//! ```

use crate::compute::{
    ArgReduceOp, BinaryOp, ComputeBackend, ReduceOp, RopeMode, TernaryOp, UnaryOp,
};
use crate::raster::RasterBackend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

// ── Linear algebra ──────────────────────────────────────────────────

/// General matrix multiply: `C += A × B`.
pub fn matmul<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    c: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.gemm(a, b, c)
}

/// Quantized dot product: `a · b → f32`.
pub fn vec_dot<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<f32, B::Error> {
    be.vec_dot(a, b)
}

// ── Activations ─────────────────────────────────────────────────────

pub fn relu<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Relu, x)
}

macro_rules! unary_op {
    ($name:ident, $variant:ident) => {
        pub fn $name<'p, B: ComputeBackend>(
            x: &Tensor<B::Storage<'p>>,
            be: &'p B,
        ) -> Result<Tensor<B::Storage<'p>>, B::Error> {
            be.unary(UnaryOp::$variant, x)
        }
    };
}

unary_op!(abs, Abs);
unary_op!(acos, Acos);
unary_op!(asin, Asin);
unary_op!(atan, Atan);
unary_op!(ceil, Ceil);
unary_op!(cos, Cos);
unary_op!(exp2, Exp2);
unary_op!(floor, Floor);
unary_op!(fract, Fract);
unary_op!(inverse_sqrt, InverseSqrt);
unary_op!(log, Log);
unary_op!(log2, Log2);
unary_op!(reciprocal, Reciprocal);
unary_op!(round, Round);
unary_op!(sign, Sign);
unary_op!(sin, Sin);
unary_op!(tan, Tan);
unary_op!(trunc, Trunc);

pub fn neg<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Neg, x)
}

pub fn sqrt<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Sqrt, x)
}

pub fn exp<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Exp, x)
}

pub fn tanh<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Tanh, x)
}

pub fn sigmoid<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Sigmoid, x)
}

pub fn silu<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Silu, x)
}

pub fn gelu<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Gelu, x)
}

pub fn softplus<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.unary(UnaryOp::Softplus, x)
}

// ── Elementwise binary ──────────────────────────────────────────────

pub fn add<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.binary(BinaryOp::Add, a, b)
}

pub fn sub<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.binary(BinaryOp::Sub, a, b)
}

pub fn mul<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.binary(BinaryOp::Mul, a, b)
}

pub fn div<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.binary(BinaryOp::Div, a, b)
}

macro_rules! binary_op {
    ($name:ident, $variant:ident) => {
        pub fn $name<'p, B: ComputeBackend>(
            a: &Tensor<B::Storage<'p>>,
            b: &Tensor<B::Storage<'p>>,
            be: &'p B,
        ) -> Result<Tensor<B::Storage<'p>>, B::Error> {
            be.binary(BinaryOp::$variant, a, b)
        }
    };
}

binary_op!(minimum, Min);
binary_op!(maximum, Max);
binary_op!(atan2, Atan2);
binary_op!(pow, Pow);
binary_op!(remainder, Rem);

/// Component-wise clamp to the inclusive interval `[low, high]`.
pub fn clamp<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    low: &Tensor<B::Storage<'p>>,
    high: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.ternary(TernaryOp::Clamp, x, low, high)
}

/// Component-wise multiply-add, `a * b + c`.
pub fn fma<'p, B: ComputeBackend>(
    a: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    c: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.ternary(TernaryOp::Fma, a, b, c)
}

pub fn scale<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    s: f32,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.scale(x, s)
}

// ── Reductions ──────────────────────────────────────────────────────

pub fn sum<'p, B: ComputeBackend>(x: &Tensor<B::Storage<'p>>, be: &'p B) -> Result<f32, B::Error> {
    be.reduce(ReduceOp::Sum, x)
}

pub fn max<'p, B: ComputeBackend>(x: &Tensor<B::Storage<'p>>, be: &'p B) -> Result<f32, B::Error> {
    be.reduce(ReduceOp::Max, x)
}

pub fn argmin<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<usize, B::Error> {
    be.arg_reduce(ArgReduceOp::Min, x)
}

pub fn argmax<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<usize, B::Error> {
    be.arg_reduce(ArgReduceOp::Max, x)
}

pub fn softmax<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    dim: isize,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.softmax(x, dim)
}

// ── Normalization ───────────────────────────────────────────────────

pub fn rms_norm<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    weight: &Tensor<B::Storage<'p>>,
    eps: f32,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.rms_norm(x, weight, eps)
}

pub fn layer_norm<'p, B: ComputeBackend>(
    x: &Tensor<B::Storage<'p>>,
    weight: &Tensor<B::Storage<'p>>,
    bias: &Tensor<B::Storage<'p>>,
    eps: f32,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.layer_norm(x, weight, bias, eps)
}

// ── Positional ──────────────────────────────────────────────────────

pub fn rope<'p, B: ComputeBackend>(
    qk: &mut Tensor<B::Storage<'p>>,
    cos: &Tensor<B::Storage<'p>>,
    sin: &Tensor<B::Storage<'p>>,
    mode: RopeMode,
    be: &'p B,
) -> Result<(), B::Error> {
    be.rope(qk, cos, sin, mode)
}

// ── Convolution ─────────────────────────────────────────────────────

pub fn conv_1d<'p, B: ComputeBackend>(
    input: &Tensor<B::Storage<'p>>,
    kernel: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.conv_1d(input, kernel)
}

pub fn ssm_conv<'p, B: ComputeBackend>(
    input: &Tensor<B::Storage<'p>>,
    kernel: &Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.ssm_conv(input, kernel)
}

// ── Element access ──────────────────────────────────────────────────

/// Set a single element by multi-dimensional index.
///
/// ```ignore
/// axnn::set(&adj, &[0, 1], true.into(), &be)?;   // Q1: set bit
/// axnn::set(&weights, &[0, 0], 1.5f32.into(), &be)?; // F32: write float
/// ```
pub fn set<'p, B: ComputeBackend>(
    tensor: &Tensor<B::Storage<'p>>,
    indices: &[usize],
    value: Scalar,
    be: &'p B,
) -> Result<(), B::Error> {
    be.set_element(tensor, indices, value)
}

/// Get a single element by multi-dimensional index.
///
/// Returns a [`Scalar`] matching the tensor's dtype.
pub fn get<'p, B: ComputeBackend>(
    tensor: &Tensor<B::Storage<'p>>,
    indices: &[usize],
    be: &'p B,
) -> Result<Scalar, B::Error> {
    be.get_element(tensor, indices)
}

// ── Graph operations (Q1) ───────────────────────────────────────────

/// Topological sort of a square Q1 adjacency matrix:
///
/// Writes node indices in dependency order into `out`.
/// Returns the number of nodes written.
pub fn topo_sort<'p, B: ComputeBackend>(
    adj: &Tensor<B::Storage<'p>>,
    out: &mut [u16],
    be: &'p B,
) -> Result<usize, B::Error> {
    be.topo_sort(adj, out)
}

/// Compute in-degree for every node in a Q1 adjacency matrix.
///
/// Writes in-degree counts into `out`. Returns number written.
pub fn indegree<'p, B: ComputeBackend>(
    adj: &Tensor<B::Storage<'p>>,
    out: &mut [u32],
    be: &'p B,
) -> Result<usize, B::Error> {
    be.indegree(adj, out)
}

// ── Raster pipeline ─────────────────────────────────────────────────

/// Fused 3-edge coverage test + conditional framebuffer write.
pub fn edge_coverage<'p, B: RasterBackend>(
    e0: &Tensor<B::Storage<'p>>,
    e1: &Tensor<B::Storage<'p>>,
    e2: &Tensor<B::Storage<'p>>,
    thresh: f32,
    color: &Tensor<B::Storage<'p>>,
    fb: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.edge_coverage(e0, e1, e2, thresh, color, fb)
}

/// Depth test with Z-buffer update.
pub fn depth_test<'p, B: RasterBackend>(
    coverage: &Tensor<B::Storage<'p>>,
    frag_z: &Tensor<B::Storage<'p>>,
    zbuf: &mut Tensor<B::Storage<'p>>,
    out_mask: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.depth_test(coverage, frag_z, zbuf, out_mask)
}

/// Barycentric interpolation of a single attribute.
pub fn bary_interp<'p, B: RasterBackend>(
    w0: &Tensor<B::Storage<'p>>,
    w1: &Tensor<B::Storage<'p>>,
    w2: &Tensor<B::Storage<'p>>,
    a0: f32,
    a1: f32,
    a2: f32,
    be: &'p B,
) -> Result<Tensor<B::Storage<'p>>, B::Error> {
    be.bary_interp(w0, w1, w2, a0, a1, a2)
}

/// Alpha blend: composite premultiplied-alpha src over dst.
pub fn alpha_blend<'p, B: RasterBackend>(
    mask: &Tensor<B::Storage<'p>>,
    src: &Tensor<B::Storage<'p>>,
    dst: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.alpha_blend(mask, src, dst)
}

/// Unpack XRGB pixels into separate R, G, B channel tensors.
pub fn xrgb_unpack<'p, B: RasterBackend>(
    packed: &Tensor<B::Storage<'p>>,
    r: &mut Tensor<B::Storage<'p>>,
    g: &mut Tensor<B::Storage<'p>>,
    b: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.xrgb_unpack(packed, r, g, b)
}

/// Pack R, G, B channel tensors into XRGB pixels.
pub fn xrgb_pack<'p, B: RasterBackend>(
    r: &Tensor<B::Storage<'p>>,
    g: &Tensor<B::Storage<'p>>,
    b: &Tensor<B::Storage<'p>>,
    packed: &mut Tensor<B::Storage<'p>>,
    be: &'p B,
) -> Result<(), B::Error> {
    be.xrgb_pack(r, g, b, packed)
}

/// Opaque rectangular blit.
pub fn blit_opaque<'p, B: RasterBackend>(
    src: &Tensor<B::Storage<'p>>,
    dst: &mut Tensor<B::Storage<'p>>,
    width: usize,
    height: usize,
    be: &'p B,
) -> Result<(), B::Error> {
    be.blit_opaque(src, dst, width, height)
}

/// Alpha-blended rectangular blit.
pub fn blit_blend<'p, B: RasterBackend>(
    src: &Tensor<B::Storage<'p>>,
    dst: &mut Tensor<B::Storage<'p>>,
    width: usize,
    height: usize,
    be: &'p B,
) -> Result<(), B::Error> {
    be.blit_blend(src, dst, width, height)
}
