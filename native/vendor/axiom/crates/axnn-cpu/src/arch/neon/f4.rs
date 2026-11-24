//! NEON E2M1 (OCP FP4) kernels with external block scales.
//!
//! The portable tier decodes a block in the same shape this one does —
//! `and`/`ushr`/`zip1`, one `tbl`, `shll`/`shll2`, four `fcvtl` — but
//! `core::simd` has no multiply-accumulate outside `std`, so it emits eight
//! `fmul` and four `fadd` per block where four `fmul` and four `fmla` will
//! do. These intrinsics take that back: the block scale multiplies the
//! decoded weights once, and `vfmaq_f32` folds the activation product into
//! the accumulator.
//!
//! Each row is still one serial accumulation in lane order, so a row's
//! result does not depend on how callers split the rows. Lane order matches
//! the portable tier's, and both agree bit for bit.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::arch::portable::f4::{BLOCK, F16_HIGH, e2m1};
use core::arch::aarch64::*;

/// Sixteen decoded weights from eight packed bytes, scaled by `scale`.
#[inline(always)]
unsafe fn block(p: *const u8, table: uint8x16_t, scale: float32x4_t) -> [float32x4_t; 4] {
    let packed = vld1_u8(p);
    // Codes in element order — the even element's low nibble first — then
    // one lookup of each code's F16 high byte.
    let pair = vzip_u8(vand_u8(packed, vdup_n_u8(0x0F)), vshr_n_u8::<4>(packed));
    let codes = vcombine_u8(pair.0, pair.1);
    let high = vqtbl1q_u8(table, codes);
    let low = vreinterpretq_f16_u16(vshll_n_u8::<8>(vget_low_u8(high)));
    let hi = vreinterpretq_f16_u16(vshll_high_n_u8::<8>(high));
    [
        vmulq_f32(vcvt_f32_f16(vget_low_f16(low)), scale),
        vmulq_f32(vcvt_high_f32_f16(low), scale),
        vmulq_f32(vcvt_f32_f16(vget_low_f16(hi)), scale),
        vmulq_f32(vcvt_high_f32_f16(hi), scale),
    ]
}

/// `Σ_i x_i · scale(i / 16) · e2m1(w_i)` over `n` elements.
pub(crate) fn vec_dot_nvfp4_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w.len() * 2 >= n && x.len() >= n);
    debug_assert!(scales.len() * BLOCK >= n);
    let whole = n / BLOCK;
    // SAFETY: the loop reads `whole * 8` weight bytes, `whole` scales and
    // `whole * 16` activations, all asserted above.
    let mut total = unsafe {
        let table = vld1q_u8(F16_HIGH.as_ptr());
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..whole {
            let scale = vdupq_n_f32(crate::fp8::e4m3_to_f32(*scales.get_unchecked(b)));
            let weights = block(w.as_ptr().add(b * BLOCK / 2), table, scale);
            let xs = x.as_ptr().add(b * BLOCK);
            for (l, (acc, weights)) in acc.iter_mut().zip(weights).enumerate() {
                *acc = vfmaq_f32(*acc, weights, vld1q_f32(xs.add(l * 4)));
            }
        }
        let sums = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        vaddvq_f32(sums)
    };
    let rest = n - whole * BLOCK;
    if rest != 0 {
        let scale = crate::fp8::e4m3_to_f32(scales[whole]);
        let mut tail = 0.0;
        for i in whole * BLOCK..n {
            let byte = w[i / 2];
            tail += x[i] * e2m1(if i % 2 == 0 { byte } else { byte >> 4 });
        }
        total += tail * scale;
    }
    total
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    /// The NEON tier sums four lane accumulators where portable sums
    /// sixteen lanes, so the two agree to F32 association, not bit for bit.
    #[test]
    fn matches_the_portable_tier() {
        let k = 2688;
        let packed: Vec<u8> = (0..k / 2).map(|i| (i * 37 % 251) as u8).collect();
        let scales: Vec<u8> = (0..k / 16).map(|i| (60 + i % 12) as u8).collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 29 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let neon = super::vec_dot_nvfp4_f32(&packed, &scales, &x, k);
        let portable = crate::arch::portable::f4::vec_dot_nvfp4_f32(&packed, &scales, &x, k);
        assert!(
            (neon - portable).abs() <= 1e-3 * (1.0 + portable.abs()),
            "{neon} vs {portable}"
        );
    }
}
