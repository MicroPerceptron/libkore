use crate::backend::Backend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;

// ── Op enums ────────────────────────────────────────────────────────────

/// Elementwise unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Absolute value.
    Abs,
    /// Arc cosine, in radians.
    Acos,
    /// Arc sine, in radians.
    Asin,
    /// Arc tangent, in radians.
    Atan,
    /// Round toward positive infinity.
    Ceil,
    /// Cosine, with the input in radians.
    Cos,
    Relu,
    Neg,
    Sqrt,
    Exp,
    /// Base-2 exponential.
    Exp2,
    /// Round toward negative infinity.
    Floor,
    /// Fractional part, defined as `x - floor(x)`.
    ///
    /// This is a frontend convenience semantic. Static graph formats may
    /// canonicalize it to `SUB(x, FLOOR(x))` instead of assigning an opcode.
    Fract,
    /// Reciprocal square root.
    InverseSqrt,
    /// Natural logarithm.
    Log,
    /// Base-2 logarithm.
    Log2,
    /// Multiplicative inverse, `1 / x`.
    Reciprocal,
    /// Round to the nearest integer, with ties to even (WGSL semantics).
    Round,
    /// Return -1, 0, or 1 according to the sign of the input.
    Sign,
    /// Sine, with the input in radians.
    Sin,
    /// Tangent, with the input in radians.
    Tan,
    Tanh,
    /// Round toward zero.
    Trunc,
    Sigmoid,
    Silu,
    Gelu,
    /// `log(1 + exp(x))`, and `x` itself above 20: PyTorch's `softplus`
    /// with its default threshold, as gated recurrences discretize a step.
    Softplus,
}

/// Elementwise binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Component-wise minimum.
    Min,
    /// Component-wise maximum.
    Max,
    /// Two-argument arc tangent, `atan2(y, x)`.
    Atan2,
    /// Component-wise floating-point exponentiation.
    Pow,
    /// Floating-point remainder, `lhs - rhs * trunc(lhs / rhs)`.
    Rem,
}

/// Elementwise ternary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryOp {
    /// Clamp `x` to the inclusive interval `[low, high]`.
    Clamp,
    /// `a * b + c`; backends may use either a fused instruction or a strict
    /// multiply followed by add where the frontend semantics permit it.
    Fma,
}

/// Elementwise comparisons producing a boolean mask tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Equal,
    Greater,
    GreaterEqual,
}

/// Scalar reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
}

/// Index-producing scalar reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgReduceOp {
    Min,
    Max,
}

/// RoPE embedding layout variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeMode {
    /// Pairs (x0,x1), (x2,x3), … rotated together.
    Interleaved,
    /// First half / second half split (GPT-NeoX style).
    NeoX,
}

// ── ComputeBackend trait ────────────────────────────────────────────────

/// Compute operations on tensors.
///
/// Extends [`Backend`] (which handles memory, transfer, cast) with
/// the actual math. Each method is dtype-agnostic — the backend
/// dispatches to the correct kernel based on the input tensor's dtype.
///
/// This trait is the **backend implementor's** interface. Users call
/// the free functions in [`ops`](crate::ops) instead.
pub trait ComputeBackend: Backend {
    // ── Linear algebra ──────────────────────────────────────────

    /// General matrix multiply: `C += A × B`.
    ///
    /// Supports float×float (f32, f64, f16, bf16) and quantized
    /// weight × activation (q4×q8, q6×q8, q8×q8). Output C is
    /// always the accumulator type (f32 for half/quant, same type
    /// for f32/f64).
    ///
    /// Dense operands use `[m, k] × [k, n]`. Packed weight/activation
    /// kernels keep their reduction axis innermost and therefore consume
    /// `[m, k] × [n, k]`, producing `[m, n]`.
    ///
    /// `c` must be pre-allocated with the correct shape. The
    /// operation accumulates into `c` (does not zero it first).
    fn gemm(
        &self,
        a: &Tensor<Self::Storage<'_>>,
        b: &Tensor<Self::Storage<'_>>,
        c: &mut Tensor<Self::Storage<'_>>,
    ) -> Result<(), Self::Error>;

    /// Single quantized dot product: `a · b → f32`.
    ///
    /// Fast path for q8·q8, q4·q8, q6·q8 similarity.
    /// For float dot products, use `gemm` with m=n=1.
    fn vec_dot(
        &self,
        a: &Tensor<Self::Storage<'_>>,
        b: &Tensor<Self::Storage<'_>>,
    ) -> Result<f32, Self::Error>;

    // ── Elementwise ─────────────────────────────────────────────

    /// Unary elementwise: `dst = op(src)`. Same shape, same dtype.
    fn unary(
        &self,
        op: UnaryOp,
        src: &Tensor<Self::Storage<'_>>,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Binary elementwise: `dst = a ⊕ b`. Broadcast rules apply.
    fn binary(
        &self,
        op: BinaryOp,
        a: &Tensor<Self::Storage<'_>>,
        b: &Tensor<Self::Storage<'_>>,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Ternary elementwise operation. Inputs share a dtype and follow the
    /// backend's elementwise broadcasting rules.
    fn ternary(
        &self,
        op: TernaryOp,
        a: &Tensor<Self::Storage<'_>>,
        b: &Tensor<Self::Storage<'_>>,
        c: &Tensor<Self::Storage<'_>>,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Scalar multiply: `dst = src × s`.
    fn scale(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        s: f32,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    // ── Reductions ──────────────────────────────────────────────

    /// Scalar reduction over the entire tensor.
    fn reduce(&self, op: ReduceOp, src: &Tensor<Self::Storage<'_>>) -> Result<f32, Self::Error>;

    /// Index-producing scalar reduction over the entire tensor.
    ///
    /// Ties resolve to the first matching element in logical storage order.
    fn arg_reduce(
        &self,
        op: ArgReduceOp,
        src: &Tensor<Self::Storage<'_>>,
    ) -> Result<usize, Self::Error>;

    /// Softmax along a dimension. Output same shape/dtype as input.
    fn softmax(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        dim: isize,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    // ── Normalization ───────────────────────────────────────────

    /// RMS normalization: `dst = (src / rms(src)) × weight`.
    fn rms_norm(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        weight: &Tensor<Self::Storage<'_>>,
        eps: f32,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// Layer normalization:
    /// `dst = ((src - mean) / std) × weight + bias`.
    fn layer_norm(
        &self,
        src: &Tensor<Self::Storage<'_>>,
        weight: &Tensor<Self::Storage<'_>>,
        bias: &Tensor<Self::Storage<'_>>,
        eps: f32,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    // ── Positional ──────────────────────────────────────────────

    /// Rotary position embedding. Modifies `qk` in-place.
    fn rope(
        &self,
        qk: &mut Tensor<Self::Storage<'_>>,
        cos: &Tensor<Self::Storage<'_>>,
        sin: &Tensor<Self::Storage<'_>>,
        mode: RopeMode,
    ) -> Result<(), Self::Error>;

    // ── Convolution ─────────────────────────────────────────────

    /// 1D convolution across channels.
    fn conv_1d(
        &self,
        input: &Tensor<Self::Storage<'_>>,
        kernel: &Tensor<Self::Storage<'_>>,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    /// SSM-style depthwise convolution (Mamba).
    fn ssm_conv(
        &self,
        input: &Tensor<Self::Storage<'_>>,
        kernel: &Tensor<Self::Storage<'_>>,
    ) -> Result<Tensor<Self::Storage<'_>>, Self::Error>;

    // ── Element access ────────────────────────────────────────────

    /// Set a single element by multi-dimensional index.
    ///
    /// The [`Scalar`] variant must match the tensor's dtype.
    /// For Q1, `Scalar::Bool(true)` sets the bit, `Bool(false)` clears.
    fn set_element(
        &self,
        tensor: &Tensor<Self::Storage<'_>>,
        indices: &[usize],
        value: Scalar,
    ) -> Result<(), Self::Error>;

    /// Get a single element by multi-dimensional index.
    ///
    /// Returns a [`Scalar`] matching the tensor's dtype.
    fn get_element(
        &self,
        tensor: &Tensor<Self::Storage<'_>>,
        indices: &[usize],
    ) -> Result<Scalar, Self::Error>;

    // ── Graph operations (Q1) ───────────────────────────────────

    /// Topological sort of a square Q1 adjacency matrix.
    ///
    /// Writes node indices in dependency order into `out`.
    /// Returns the number of nodes written (== N for acyclic graphs).
    fn topo_sort(
        &self,
        adj: &Tensor<Self::Storage<'_>>,
        out: &mut [u16],
    ) -> Result<usize, Self::Error>;

    /// Compute in-degree (column popcount) for every node in a Q1
    /// adjacency matrix.
    ///
    /// Writes in-degree counts into `out` (one per column).
    /// Returns the number of columns written.
    fn indegree(
        &self,
        adj: &Tensor<Self::Storage<'_>>,
        out: &mut [u32],
    ) -> Result<usize, Self::Error>;
}
