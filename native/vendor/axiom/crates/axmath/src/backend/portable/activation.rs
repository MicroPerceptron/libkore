//! Portable composite scalar functions built on elementary ops.

#[inline(always)]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + super::exp_f32(-x))
}

#[inline(always)]
pub(crate) fn silu_f32(x: f32) -> f32 {
    x / (1.0 + super::exp_f32(-x))
}

#[inline(always)]
pub(crate) fn gelu_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + super::tanh_f32(0.797_884_56 * (x + 0.044715 * x * x * x)))
}

#[inline(always)]
pub(crate) fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + super::exp_f64(-x))
}

#[inline(always)]
pub(crate) fn silu_f64(x: f64) -> f64 {
    x / (1.0 + super::exp_f64(-x))
}

#[inline(always)]
pub(crate) fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + super::tanh_f64(0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x)))
}
