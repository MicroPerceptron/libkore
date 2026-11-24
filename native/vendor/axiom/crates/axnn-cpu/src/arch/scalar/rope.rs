//! Rotary position embedding kernels — scalar fallback.

use crate::floats::Float;

/// RoPE with interleaved pairs (mode 0 — Llama, Mistral).
///
/// Rotates consecutive (even, odd) pairs in each head:
///   q[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
///   q[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
///
/// qk:    mutable [n_tokens, n_heads, head_dim]
/// cos:   [n_tokens, head_dim / 2]
/// sin:   [n_tokens, head_dim / 2]
fn rope_interleaved<T: Float>(
    qk: &mut [T],
    cos: &[T],
    sin: &[T],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            let head = &mut qk[base..base + head_dim];
            for (i, pair) in head.chunks_exact_mut(2).enumerate() {
                let e = pair[0];
                let o = pair[1];
                let c = freq_row[i];
                let s = sin_row[i];
                pair[0] = e * c - o * s;
                pair[1] = e * s + o * c;
            }
        }
    }
}

/// RoPE with split halves (mode 2 — Qwen, GPT-NeoX).
///
/// Rotates first-half against second-half of each head:
///   q[i]        = q[i]        * cos(θ) - q[i + half] * sin(θ)
///   q[i + half] = q[i + half] * cos(θ) + q[i]        * sin(θ)
///
/// qk:    mutable [n_tokens, n_heads, head_dim]
/// cos:   [n_tokens, head_dim / 2]
/// sin:   [n_tokens, head_dim / 2]
fn rope_neox<T: Float>(
    qk: &mut [T],
    cos: &[T],
    sin: &[T],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            let (lo, hi) = qk[base..base + head_dim].split_at_mut(half);
            for ((l, h), (&c, &s)) in lo
                .iter_mut()
                .zip(hi.iter_mut())
                .zip(freq_row.iter().zip(sin_row))
            {
                let lv = *l;
                let hv = *h;
                *l = lv * c - hv * s;
                *h = hv * c + lv * s;
            }
        }
    }
}

// ── F32 / F64 wrappers ───────────────────────────────────────────────

pub(crate) fn rope_interleaved_f32(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    rope_interleaved(qk, cos, sin, n_tokens, n_heads, head_dim)
}
pub(crate) fn rope_neox_f32(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    rope_neox(qk, cos, sin, n_tokens, n_heads, head_dim)
}

pub(crate) fn rope_interleaved_f64(
    qk: &mut [f64],
    cos: &[f64],
    sin: &[f64],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    rope_interleaved(qk, cos, sin, n_tokens, n_heads, head_dim)
}
pub(crate) fn rope_neox_f64(
    qk: &mut [f64],
    cos: &[f64],
    sin: &[f64],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    rope_neox(qk, cos, sin, n_tokens, n_heads, head_dim)
}

// ── F16 (packed &[u8], cos/sin always f32) ───────────────────────────

pub(crate) fn rope_interleaved_f16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim * 2; // byte offset
            let head = &mut qk[base..base + head_dim * 2];
            for (i, pair) in head.chunks_exact_mut(4).enumerate() {
                let e = crate::cast::read_f16(&pair[0..2]);
                let o = crate::cast::read_f16(&pair[2..4]);
                let c = freq_row[i];
                let s = sin_row[i];
                crate::cast::write_f16(&mut pair[0..2], e * c - o * s);
                crate::cast::write_f16(&mut pair[2..4], e * s + o * c);
            }
        }
    }
}

pub(crate) fn rope_neox_f16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim * 2; // byte offset
            let head = &mut qk[base..base + head_dim * 2];
            let (lo, hi) = head.split_at_mut(half * 2);
            for ((lc, hc), (&c, &s)) in lo
                .chunks_exact_mut(2)
                .zip(hi.chunks_exact_mut(2))
                .zip(freq_row.iter().zip(sin_row))
            {
                let lv = crate::cast::read_f16(lc);
                let hv = crate::cast::read_f16(hc);
                crate::cast::write_f16(lc, lv * c - hv * s);
                crate::cast::write_f16(hc, hv * c + lv * s);
            }
        }
    }
}

// ── BF16 (packed &[u8], cos/sin always f32) ──────────────────────────

pub(crate) fn rope_interleaved_bf16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim * 2;
            let head = &mut qk[base..base + head_dim * 2];
            for (i, pair) in head.chunks_exact_mut(4).enumerate() {
                let e = crate::cast::read_bf16(&pair[0..2]);
                let o = crate::cast::read_bf16(&pair[2..4]);
                let c = freq_row[i];
                let s = sin_row[i];
                crate::cast::write_bf16(&mut pair[0..2], e * c - o * s);
                crate::cast::write_bf16(&mut pair[2..4], e * s + o * c);
            }
        }
    }
}

pub(crate) fn rope_neox_bf16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_row = &cos[t * half..(t + 1) * half];
        let sin_row = &sin[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim * 2;
            let head = &mut qk[base..base + head_dim * 2];
            let (lo, hi) = head.split_at_mut(half * 2);
            for ((lc, hc), (&c, &s)) in lo
                .chunks_exact_mut(2)
                .zip(hi.chunks_exact_mut(2))
                .zip(freq_row.iter().zip(sin_row))
            {
                let lv = crate::cast::read_bf16(lc);
                let hv = crate::cast::read_bf16(hc);
                crate::cast::write_bf16(lc, lv * c - hv * s);
                crate::cast::write_bf16(hc, hv * c + lv * s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn interleaved_zero_freq_is_noop() {
        // cos=1, sin=0 → no rotation.
        let mut qk = [1.0, 2.0, 3.0, 4.0f32]; // 1 token, 1 head, dim=4
        let cos = [1.0, 1.0f32];
        let sin = [0.0, 0.0f32];
        rope_interleaved_f32(&mut qk, &cos, &sin, 1, 1, 4);
        assert_eq!(qk, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn interleaved_90_degree_rotation() {
        // cos=0, sin=1 → (e, o) → (-o, e)
        let mut qk = [3.0, 5.0, 7.0, 11.0f32];
        let cos = [0.0, 0.0f32];
        let sin = [1.0, 1.0f32];
        rope_interleaved_f32(&mut qk, &cos, &sin, 1, 1, 4);
        assert!(close(qk[0], -5.0));
        assert!(close(qk[1], 3.0));
        assert!(close(qk[2], -11.0));
        assert!(close(qk[3], 7.0));
    }

    #[test]
    fn interleaved_multi_head() {
        // 1 token, 2 heads, dim=2. Each head gets the same freqs.
        let mut qk = [1.0, 0.0, 0.0, 1.0f32];
        let cos = [0.0f32]; // head_dim/2 = 1
        let sin = [1.0f32];
        rope_interleaved_f32(&mut qk, &cos, &sin, 1, 2, 2);
        // head 0: (1,0) → (1*0 - 0*1, 1*1 + 0*0) = (0, 1)
        // head 1: (0,1) → (0*0 - 1*1, 0*1 + 1*0) = (-1, 0)
        assert!(close(qk[0], 0.0));
        assert!(close(qk[1], 1.0));
        assert!(close(qk[2], -1.0));
        assert!(close(qk[3], 0.0));
    }

    #[test]
    fn neox_zero_freq_is_noop() {
        let mut qk = [1.0, 2.0, 3.0, 4.0f32];
        let cos = [1.0, 1.0f32];
        let sin = [0.0, 0.0f32];
        rope_neox_f32(&mut qk, &cos, &sin, 1, 1, 4);
        assert_eq!(qk, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn neox_90_degree_rotation() {
        // cos=0, sin=1 → (lo, hi) → (-hi, lo)
        // Split: lo=[3,5], hi=[7,11]
        let mut qk = [3.0, 5.0, 7.0, 11.0f32];
        let cos = [0.0, 0.0f32];
        let sin = [1.0, 1.0f32];
        rope_neox_f32(&mut qk, &cos, &sin, 1, 1, 4);
        assert!(close(qk[0], -7.0)); // lo[0]*0 - hi[0]*1
        assert!(close(qk[1], -11.0)); // lo[1]*0 - hi[1]*1
        assert!(close(qk[2], 3.0)); // hi[0]*0 + lo[0]*1
        assert!(close(qk[3], 5.0)); // hi[1]*0 + lo[1]*1
    }

    #[test]
    fn neox_multi_token() {
        // 2 tokens, 1 head, dim=2 — each token has its own freq.
        let mut qk = [1.0, 2.0, 3.0, 4.0f32]; // [2, 1, 2]
        // Token 0: cos=1,sin=0 (no rotation)
        // Token 1: cos=0,sin=1 (90° rotation)
        let cos = [1.0, 0.0f32]; // [2, 1]
        let sin = [0.0, 1.0f32]; // [2, 1]
        rope_neox_f32(&mut qk, &cos, &sin, 2, 1, 2);
        // Token 0: (1,2) → (1,2) (no change)
        // Token 1: lo=3, hi=4 → (-4, 3)
        assert_eq!(qk[0], 1.0);
        assert_eq!(qk[1], 2.0);
        assert!(close(qk[2], -4.0));
        assert!(close(qk[3], 3.0));
    }
}
