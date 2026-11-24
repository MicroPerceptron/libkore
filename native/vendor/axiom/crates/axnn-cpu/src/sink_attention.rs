//! Attention of one query row over F32 keys and values with a per-head sink:
//! a learned logit in each softmax's denominator that carries no value, so a
//! head may attend to "nothing" in part.
//!
//! `out_h = Σ_j e^(a_j − m) v_j / (e^(sink_h − m) + Σ_j e^(a_j − m))` with
//! `a_j = scale · q_h · k_j`. The heads sharing one key/value head are done
//! together, so each key and value row is read once for all of them. The
//! softmax is online: blocks of keys update a running maximum, rescaling what
//! has accumulated, so no logit row is stored. The sink starts the running
//! maximum, which also makes an empty key set well defined (zero output).

use crate::arch::portable::vec::exp_f32_n;
use core::simd::prelude::*;

/// Keys and values of consecutive slots, `[slots, kv_heads * dim]` each.
#[derive(Clone, Copy, Debug)]
pub struct KeyValues<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
}

/// The most query heads one key/value head may serve.
pub const MAX_GROUP: usize = 64;

/// Keys per online-softmax block: one vector of logits per head.
const BLOCK: usize = 8;
type Lanes = Simd<f32, BLOCK>;

/// Attend `query` (`[group * dim]`: the `group` heads sharing key/value head
/// `kv_head` of `kv_heads`) over every slot of `spans`, writing `[group *
/// dim]` to `output`. `sinks` holds each head's unscaled sink logit, or
/// −∞ for none (then at least one key must be given).
///
/// Panics when the shapes disagree or `group` exceeds [`MAX_GROUP`].
pub fn sink_attention_f32(
    query: &[f32],
    spans: &[KeyValues<'_>],
    (kv_head, kv_heads, dim): (usize, usize, usize),
    scale: f32,
    sinks: &[f32],
    output: &mut [f32],
) {
    let group = sinks.len();
    assert!(group <= MAX_GROUP && dim > 0 && kv_head < kv_heads);
    assert_eq!(query.len(), group * dim);
    assert_eq!(output.len(), group * dim);
    let row = kv_heads * dim;
    let mut peak = [0.0_f32; MAX_GROUP];
    let mut total = [0.0_f32; MAX_GROUP];
    for h in 0..group {
        // The sink as a key with no value: e^(sink − peak) = 1. A head
        // without one (−∞) starts from an empty sum.
        peak[h] = sinks[h];
        total[h] = if sinks[h] == f32::NEG_INFINITY {
            0.0
        } else {
            1.0
        };
    }
    output.fill(0.0);
    let mut logits = [[0.0_f32; BLOCK]; MAX_GROUP];
    for span in spans {
        assert_eq!(span.keys.len(), span.values.len());
        assert!(span.keys.len().is_multiple_of(row));
        let slots = span.keys.len() / row;
        for start in (0..slots).step_by(BLOCK) {
            let count = BLOCK.min(slots - start);
            for b in 0..count {
                let key = &span.keys[(start + b) * row + kv_head * dim..][..dim];
                for (row, q) in logits.iter_mut().zip(query.chunks_exact(dim)) {
                    row[b] = scale * dot(q, key);
                }
            }
            for h in 0..group {
                let block = Lanes::from_array(logits[h]);
                let valid = Mask::<i32, BLOCK>::from_array(core::array::from_fn(|b| b < count));
                let block = valid.select(block, Lanes::splat(f32::NEG_INFINITY));
                let top = block.reduce_max();
                let acc = &mut output[h * dim..][..dim];
                if top > peak[h] {
                    let correction = axmath::exp_f32(peak[h] - top);
                    total[h] *= correction;
                    for a in acc.iter_mut() {
                        *a *= correction;
                    }
                    peak[h] = top;
                }
                let weights = valid
                    .select(exp_f32_n(block - Lanes::splat(peak[h])), Lanes::splat(0.0))
                    .to_array();
                total[h] += weights.iter().sum::<f32>();
                for (b, &weight) in weights.iter().enumerate().take(count) {
                    let value = &span.values[(start + b) * row + kv_head * dim..][..dim];
                    axpy(weight, value, acc);
                }
            }
        }
    }
    for h in 0..group {
        let inverse = 1.0 / total[h];
        for a in &mut output[h * dim..][..dim] {
            *a *= inverse;
        }
    }
}

/// `a · b` in eight lanes, then a lane sum.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a_body, a_tail) = a.as_chunks::<8>();
    let (b_body, b_tail) = b.as_chunks::<8>();
    let mut sum = f32x8::splat(0.0);
    for (a, b) in a_body.iter().zip(b_body) {
        sum += f32x8::from_array(*a) * f32x8::from_array(*b);
    }
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(a, b)| a * b).sum();
    sum.reduce_sum() + tail
}

/// `y += w · x`.
#[inline(always)]
fn axpy(w: f32, x: &[f32], y: &mut [f32]) {
    let (x_body, x_tail) = x.as_chunks::<8>();
    let (y_body, y_tail) = y.as_chunks_mut::<8>();
    let weight = f32x8::splat(w);
    for (x, y) in x_body.iter().zip(y_body) {
        *y = (f32x8::from_array(*y) + weight * f32x8::from_array(*x)).to_array();
    }
    for (x, y) in x_tail.iter().zip(y_tail) {
        *y += w * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn values(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Against a direct F64 softmax with the sink in its denominator, over
    /// keys split into spans of uneven lengths, including an empty one.
    #[test]
    fn matches_a_direct_softmax_with_sinks() {
        let (kv_heads, dim, group) = (2, 12, 3);
        let row = kv_heads * dim;
        let slots = [5, 0, 17, 1];
        let keys: Vec<Vec<f32>> = slots
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                values(n * row, 10 + i as u32)
                    .iter()
                    .map(|v| 3.0 * v)
                    .collect()
            })
            .collect();
        let vals: Vec<Vec<f32>> = slots
            .iter()
            .enumerate()
            .map(|(i, &n)| values(n * row, 20 + i as u32))
            .collect();
        let spans: Vec<KeyValues<'_>> = keys
            .iter()
            .zip(&vals)
            .map(|(k, v)| KeyValues { keys: k, values: v })
            .collect();
        let query = values(group * dim, 3);
        let sinks = [0.5, -2.0, 4.0];
        let scale = 0.3;
        for kv_head in 0..kv_heads {
            let mut output = alloc::vec![f32::NAN; group * dim];
            sink_attention_f32(
                &query,
                &spans,
                (kv_head, kv_heads, dim),
                scale,
                &sinks,
                &mut output,
            );
            for h in 0..group {
                let q = &query[h * dim..][..dim];
                let mut logits = Vec::new();
                let mut rows = Vec::new();
                for span in &spans {
                    for s in 0..span.keys.len() / row {
                        let k = &span.keys[s * row + kv_head * dim..][..dim];
                        let dot: f64 = q.iter().zip(k).map(|(a, b)| f64::from(a * b)).sum();
                        logits.push(f64::from(scale) * dot);
                        rows.push(&span.values[s * row + kv_head * dim..][..dim]);
                    }
                }
                let top = logits.iter().fold(f64::from(sinks[h]), |m, &l| m.max(l));
                let mut denominator = (f64::from(sinks[h]) - top).exp();
                let mut expected = alloc::vec![0.0_f64; dim];
                for (l, v) in logits.iter().zip(&rows) {
                    let w = (l - top).exp();
                    denominator += w;
                    for (e, &x) in expected.iter_mut().zip(v.iter()) {
                        *e += w * f64::from(x);
                    }
                }
                for (d, e) in expected.iter().enumerate() {
                    let actual = f64::from(output[h * dim + d]);
                    let e = e / denominator;
                    assert!(
                        (actual - e).abs() < 1e-5,
                        "head {h}, dim {d}: {actual} vs {e}"
                    );
                }
            }
        }
        // No keys: the sink takes all the weight and the output is zero.
        let mut output = alloc::vec![1.0; group * dim];
        sink_attention_f32(&query, &[], (0, kv_heads, dim), scale, &sinks, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }
}
