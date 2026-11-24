//! AVX-512 P·V update for Nemotron's blocked decode attention.
//!
//! Query tiles have only eight lanes, so their logits stay on the AVX2
//! implementation. The 128-wide value rows have enough contiguous work
//! for 512-bit loads and FMAs. A paired update shares each value load while
//! preserving each row's slot order and FMA association.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// `output[d] = output[d] * correction + sum_j(weights[j] * values[j][d])`.
/// Each `d` is accumulated in increasing slot order.
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn block_values(
    output: &mut [f32],
    correction: f32,
    weights: &[f32],
    stride: usize,
    values: &[f32],
    visible: usize,
) {
    let dim = output.len();
    debug_assert!(dim.is_multiple_of(32));
    debug_assert!(values.len() >= visible * dim);
    debug_assert!(weights.len() >= visible.saturating_sub(1) * stride + usize::from(visible > 0));
    let scale = _mm512_set1_ps(correction);
    for base in (0..dim).step_by(32) {
        let out = output.as_mut_ptr().add(base);
        let mut low = _mm512_mul_ps(_mm512_loadu_ps(out), scale);
        let mut high = _mm512_mul_ps(_mm512_loadu_ps(out.add(16)), scale);
        for j in 0..visible {
            let weight = _mm512_set1_ps(*weights.get_unchecked(j * stride));
            let row = values.as_ptr().add(j * dim + base);
            low = _mm512_fmadd_ps(_mm512_loadu_ps(row), weight, low);
            high = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16)), weight, high);
        }
        _mm512_storeu_ps(out, low);
        _mm512_storeu_ps(out.add(16), high);
    }
}

/// Two query rows over the same visible slots. The value vectors are loaded
/// once and used for both rows; each output has the single-row bit pattern.
#[target_feature(enable = "avx512f,fma")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn block_values_pair(
    first: &mut [f32],
    second: &mut [f32],
    corrections: [f32; 2],
    first_weights: &[f32],
    second_weights: &[f32],
    stride: usize,
    values: &[f32],
    visible: usize,
) {
    let dim = first.len();
    debug_assert_eq!(second.len(), dim);
    debug_assert!(dim.is_multiple_of(32));
    debug_assert!(values.len() >= visible * dim);
    let (sa, sb) = (
        _mm512_set1_ps(corrections[0]),
        _mm512_set1_ps(corrections[1]),
    );
    for base in (0..dim).step_by(32) {
        let (a_out, b_out) = (first.as_mut_ptr().add(base), second.as_mut_ptr().add(base));
        let mut a0 = _mm512_mul_ps(_mm512_loadu_ps(a_out), sa);
        let mut a1 = _mm512_mul_ps(_mm512_loadu_ps(a_out.add(16)), sa);
        let mut b0 = _mm512_mul_ps(_mm512_loadu_ps(b_out), sb);
        let mut b1 = _mm512_mul_ps(_mm512_loadu_ps(b_out.add(16)), sb);
        for j in 0..visible {
            let wa = _mm512_set1_ps(*first_weights.get_unchecked(j * stride));
            let wb = _mm512_set1_ps(*second_weights.get_unchecked(j * stride));
            let row = values.as_ptr().add(j * dim + base);
            let v0 = _mm512_loadu_ps(row);
            let v1 = _mm512_loadu_ps(row.add(16));
            a0 = _mm512_fmadd_ps(v0, wa, a0);
            a1 = _mm512_fmadd_ps(v1, wa, a1);
            b0 = _mm512_fmadd_ps(v0, wb, b0);
            b1 = _mm512_fmadd_ps(v1, wb, b1);
        }
        _mm512_storeu_ps(a_out, a0);
        _mm512_storeu_ps(a_out.add(16), a1);
        _mm512_storeu_ps(b_out, b0);
        _mm512_storeu_ps(b_out.add(16), b1);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn paired_values_equal_independent_updates_and_scalar_fma() {
        let dim = 128;
        let visible = 13;
        let stride = 8;
        let values: Vec<f32> = (0..visible * dim)
            .map(|i| ((i * 37 % 97) as f32 - 48.0) / 49.0)
            .collect();
        let weights: Vec<f32> = (0..visible * stride)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 37.0)
            .collect();
        let first: Vec<f32> = (0..dim).map(|i| i as f32 / 103.0).collect();
        let second: Vec<f32> = (0..dim).map(|i| -(i as f32) / 91.0).collect();
        let mut pair_a = first.clone();
        let mut pair_b = second.clone();
        let mut alone_a = first.clone();
        let mut alone_b = second.clone();
        // SAFETY: this module is compiled only for the AVX-512 CPU tier.
        unsafe {
            super::block_values_pair(
                &mut pair_a,
                &mut pair_b,
                [0.75, -0.25],
                &weights,
                &weights[stride..],
                stride,
                &values,
                visible - 1,
            );
            super::block_values(&mut alone_a, 0.75, &weights, stride, &values, visible - 1);
            super::block_values(
                &mut alone_b,
                -0.25,
                &weights[stride..],
                stride,
                &values,
                visible - 1,
            );
        }
        assert_eq!(pair_a, alone_a);
        assert_eq!(pair_b, alone_b);
        for (base, (actual_a, actual_b)) in pair_a.iter().zip(&pair_b).enumerate() {
            let mut want_a = first[base] * 0.75;
            let mut want_b = second[base] * -0.25;
            for j in 0..visible - 1 {
                want_a = values[j * dim + base].mul_add(weights[j * stride], want_a);
                want_b = values[j * dim + base].mul_add(weights[(j + 1) * stride], want_b);
            }
            assert_eq!(actual_a.to_bits(), want_a.to_bits());
            assert_eq!(actual_b.to_bits(), want_b.to_bits());
        }
        assert_ne!(pair_a, vec![0.0; dim]);
    }
}
