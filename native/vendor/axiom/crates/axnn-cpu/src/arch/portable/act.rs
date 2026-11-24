//! Elementwise activation kernels — portable `core::simd`, in place.
//!
//! `tanh` is the classic 13/6-degree minimax rational (float-ulp
//! accuracy on the fitted range, saturating cleanly at the clamp
//! bound); `sigmoid` rides it via `σ(x) = ½ + ½·tanh(x/2)`. The scalar
//! tail evaluates the identical polynomial, so results don't depend on
//! where the SIMD chunking falls.

use core::simd::prelude::*;

use super::W;

const ALPHA: [f32; 7] = [
    4.89352455891786e-03,
    6.37261928875436e-04,
    1.48572235717979e-05,
    5.12229709037114e-08,
    -8.60467152213735e-11,
    2.00018790482477e-13,
    -2.76076847742355e-16,
];
const BETA: [f32; 4] = [
    4.89352518554385e-03,
    2.26843463243900e-03,
    1.18534705686654e-04,
    1.19825839466702e-06,
];
/// Fitted range bound; the rational is ±1 to f32 precision beyond it.
const CLAMP: f32 = 7.90531110763549805;

#[inline(always)]
fn tanh_lanes(x: Simd<f32, W>) -> Simd<f32, W> {
    let x = x.simd_clamp(Simd::splat(-CLAMP), Simd::splat(CLAMP));
    let x2 = x * x;
    let mut p = Simd::splat(ALPHA[6]);
    for a in ALPHA[..6].iter().rev() {
        p = p * x2 + Simd::splat(*a);
    }
    let p = p * x;
    let mut q = Simd::splat(BETA[3]);
    for b in BETA[..3].iter().rev() {
        q = q * x2 + Simd::splat(*b);
    }
    p / q
}

#[inline(always)]
fn tanh_one(x: f32) -> f32 {
    let x = x.clamp(-CLAMP, CLAMP);
    let x2 = x * x;
    let mut p = ALPHA[6];
    for a in ALPHA[..6].iter().rev() {
        p = p * x2 + a;
    }
    let p = p * x;
    let mut q = BETA[3];
    for b in BETA[..3].iter().rev() {
        q = q * x2 + b;
    }
    p / q
}

pub(crate) fn tanh_f32_inplace(v: &mut [f32]) {
    let mut chunks = v.chunks_exact_mut(W);
    for ch in &mut chunks {
        tanh_lanes(Simd::from_slice(ch)).copy_to_slice(ch);
    }
    for x in chunks.into_remainder() {
        *x = tanh_one(*x);
    }
}

pub(crate) fn sigmoid_f32_inplace(v: &mut [f32]) {
    let half = Simd::<f32, W>::splat(0.5);
    let mut chunks = v.chunks_exact_mut(W);
    for ch in &mut chunks {
        let x = Simd::from_slice(ch);
        (half + half * tanh_lanes(x * half)).copy_to_slice(ch);
    }
    for x in chunks.into_remainder() {
        *x = 0.5 + 0.5 * tanh_one(*x * 0.5);
    }
}

#[cfg(test)]
mod tests {
    /// Reference points from libm tanh.
    const CASES: [(f32, f32); 13] = [
        (-8.0, -0.9999997749296758),
        (-3.5, -0.9981778976111987),
        (-1.75, -0.9413755384972874),
        (-0.9, -0.7162978701990245),
        (-0.31, -0.3004370971476541),
        (-0.0078125, -0.007812341058161014),
        (0.0, 0.0),
        (0.0078125, 0.007812341058161014),
        (0.31, 0.3004370971476541),
        (0.9, 0.7162978701990245),
        (1.75, 0.9413755384972874),
        (3.5, 0.9981778976111987),
        (8.0, 0.9999997749296758),
    ];

    #[test]
    fn tanh_matches_libm_reference() {
        // Vector-path lanes and the scalar tail through one call.
        let mut v: [f32; 13] = core::array::from_fn(|i| CASES[i].0);
        super::tanh_f32_inplace(&mut v);
        for (i, &(x, want)) in CASES.iter().enumerate() {
            // ±3e-7: saturation to exactly ±1 at the clamp bound.
            assert!(
                (v[i] - want).abs() <= 3e-7,
                "tanh({x}): got {}, want {want}",
                v[i]
            );
        }
    }

    #[test]
    fn sigmoid_matches_identity() {
        let s = |x: f32| {
            let mut a = [x];
            super::sigmoid_f32_inplace(&mut a);
            a[0]
        };
        // Defining identities plus one libm-derived point.
        for &(x, _) in &CASES {
            assert!((s(x) + s(-x) - 1.0).abs() <= 2e-7, "σ({x})+σ(−{x}) ≠ 1");
        }
        assert!((s(0.0) - 0.5).abs() <= 1e-7);
        assert!((s(0.9) - 0.710949502625004).abs() <= 2e-7);
        let mut v: [f32; 13] = core::array::from_fn(|i| CASES[i].0);
        super::sigmoid_f32_inplace(&mut v);
        for i in 1..v.len() {
            assert!(v[i] > v[i - 1] - 1e-7, "sigmoid must be nondecreasing");
        }
    }
}
