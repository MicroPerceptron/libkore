//! Floating-point trait for generic scalar kernels.
//!
//! Only f32 and f64 implement this — F16 kernels operate on packed
//! `&[u8]` via the `crate::cast` helpers instead.

pub(crate) trait Float:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::MulAssign
    + PartialOrd
    + core::fmt::Debug
{
    const ZERO: Self;
    const ONE: Self;

    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;

    /// Narrow a compile-time constant from f64.
    fn from_f64(v: f64) -> Self;

    /// Convert a runtime count (e.g., array length).
    fn from_usize(n: usize) -> Self;
}

impl Float for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;

    #[inline(always)]
    fn exp(self) -> f32 {
        axmath::exp_f32(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f32 {
        axmath::sqrt_f32(self)
    }
    #[inline(always)]
    fn tanh(self) -> f32 {
        axmath::tanh_f32(self)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    #[inline(always)]
    fn from_usize(n: usize) -> f32 {
        n as f32
    }
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    #[inline(always)]
    fn exp(self) -> f64 {
        axmath::exp_f64(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        axmath::sqrt_f64(self)
    }
    #[inline(always)]
    fn tanh(self) -> f64 {
        axmath::tanh_f64(self)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn from_usize(n: usize) -> f64 {
        n as f64
    }
}
