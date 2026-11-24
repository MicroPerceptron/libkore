#![feature(allocator_api)]
//! A Qwen3.5-MoE block's routed experts through the native engine: the
//! route from the router logits (softmax top-k, renormalized, and the
//! shared expert's sigmoid gate), the routed SwiGLU and the weighted down
//! projection over NVFP4 expert tables, against an F64 reference of
//! Transformers' `Qwen3_5MoeSparseMoeBlock` that dequantizes the records
//! itself.
use ax::routed::{ExpertRecord, MoeRouteParams, RoutedExpertsParams};
use ax::{DType, GraphBuilder, TensorFlags, TensorId};
use ax_exec::{AxCpuKernelPolicy, NativeProgram};
use std::alloc::Global;

const TOKENS: usize = 3;
const HIDDEN: usize = 128;
const INNER: usize = 64;
const EXPERTS: usize = 6;
const TOP_K: usize = 2;
const SHARED: usize = 1;
const TABLE: usize = EXPERTS + SHARED;
const SLOTS: usize = TOP_K + SHARED;

/// A deterministic generator: the same stream on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

/// E2M1 code values, the sign in bit 3.
fn e2m1(code: u8) -> f64 {
    const MAGNITUDE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAGNITUDE[(code & 7) as usize];
    if code & 8 != 0 { -value } else { value }
}

/// E4M3FN values (no NaN codes are generated).
fn e4m3(bits: u8) -> f64 {
    let (sign, exponent, mantissa) = (bits >> 7, (bits >> 3) & 15, bits & 7);
    let value = if exponent == 0 {
        f64::from(mantissa) / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7)
    };
    if sign != 0 { -value } else { value }
}

/// One table of `TABLE` records of `layout`, random codes, block scales
/// over a modest range (exponents 3..=8, both signs), and positive tensor
/// scales; plus each record dequantized `[rows, cols]` for the reference.
fn table(layout: ExpertRecord, rng: &mut Rng) -> (Vec<u8>, Vec<Vec<f64>>) {
    let (rows, cols) = (layout.rows as usize, layout.cols as usize);
    let stride = layout.bytes().unwrap();
    let mut bytes = vec![0u8; TABLE * stride];
    let mut dense = Vec::new();
    for expert in 0..TABLE {
        let record = &mut bytes[expert * stride..(expert + 1) * stride];
        let scales: Vec<f32> = (0..layout.scales).map(|_| 0.25 + (rng.next() % 1000) as f32 / 1000.0).collect();
        for (index, scale) in scales.iter().enumerate() {
            record[index * 4..index * 4 + 4].copy_from_slice(&scale.to_le_bytes());
        }
        let codes_at = ExpertRecord::HEADER;
        let scales_at = layout.block_scale_offset().unwrap();
        for byte in &mut record[codes_at..scales_at] {
            *byte = rng.next() as u8;
        }
        for byte in &mut record[scales_at..scales_at + rows * cols / 16] {
            let exponent = 3 + rng.next() % 6;
            *byte = ((rng.next() % 2) << 7) as u8 | (exponent << 3) as u8 | (rng.next() % 8) as u8;
        }
        let run = rows / layout.scales as usize;
        let mut weights = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let byte = record[codes_at + row * cols / 2 + col / 2];
                let code = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                let block = record[scales_at + row * cols / 16 + col / 16];
                weights[row * cols + col] =
                    e2m1(code) * e4m3(block) * f64::from(scales[row / run]);
            }
        }
        dense.push(weights);
    }
    (bytes, dense)
}

fn bytes(values: &[f32]) -> &[u8] {
    // SAFETY: initialized, padding-free f32 words, borrowed immutably.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn constant(b: &mut GraphBuilder<Global>, dims: &[u32], data: &[u8]) -> TensorId {
    b.tensor(DType::U8, TensorFlags::CONSTANT, dims, data).unwrap()
}

/// The block: inputs the router logits `[tokens, experts + shared]` and the
/// hidden rows; outputs the routes, then the block's output.
fn block(gate_up: &[u8], down: &[u8]) -> ax::GraphBuf<Global> {
    let route = MoeRouteParams::new(EXPERTS as u32, TOP_K as u32, SHARED as u32).unwrap();
    let routed =
        RoutedExpertsParams::new(TABLE as u32, SLOTS as u32, HIDDEN as u32, INNER as u32).unwrap();
    let mut b = GraphBuilder::new_in(Global);
    let t = TOKENS as u32;
    let logits = b
        .tensor(DType::F32, TensorFlags::INPUT, &[t, TABLE as u32], &[])
        .unwrap();
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[t, HIDDEN as u32], &[])
        .unwrap();
    let gate_up_bytes = routed.gate_up_record().bytes().unwrap() as u32;
    let down_bytes = routed.down_record().bytes().unwrap() as u32;
    let gate_up = constant(&mut b, &[TABLE as u32, gate_up_bytes], gate_up);
    let down = constant(&mut b, &[TABLE as u32, down_bytes], down);
    let routes = b.moe_route(logits, t, route, TensorFlags::OUTPUT).unwrap();
    let h = b
        .moe_gate_up_nvfp4(x, gate_up, routes, t, routed, TensorFlags::default())
        .unwrap();
    b.moe_down_nvfp4(h, down, routes, t, routed, TensorFlags::OUTPUT)
        .unwrap();
    b.finish().unwrap()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Transformers' block in F64: softmax over every routed logit, the top-k
/// renormalized, each picked expert's SwiGLU then down projection, plus the
/// shared expert gated by the sigmoid of its logit.
fn reference(
    logits: &[f32],
    x: &[f32],
    gate_up: &[Vec<f64>],
    down: &[Vec<f64>],
) -> (Vec<Vec<usize>>, Vec<f64>, Vec<f64>) {
    let mut picks = Vec::new();
    let mut output = vec![0.0; TOKENS * HIDDEN];
    let mut magnitude = vec![0.0; TOKENS * HIDDEN];
    for token in 0..TOKENS {
        let row = &logits[token * TABLE..(token + 1) * TABLE];
        let largest = row[..EXPERTS].iter().fold(f64::MIN, |m, &v| m.max(f64::from(v)));
        let probabilities: Vec<f64> =
            row[..EXPERTS].iter().map(|&v| (f64::from(v) - largest).exp()).collect();
        let total: f64 = probabilities.iter().sum();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        // Descending, ties to the lower index.
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        order.truncate(TOP_K);
        let picked: f64 = order.iter().map(|&e| probabilities[e] / total).sum();
        let mut weighted: Vec<(usize, f64)> =
            order.iter().map(|&e| (e, probabilities[e] / total / picked)).collect();
        let gate = 1.0 / (1.0 + (-f64::from(row[EXPERTS])).exp());
        weighted.push((EXPERTS, gate));
        picks.push(order);
        let input = &x[token * HIDDEN..(token + 1) * HIDDEN];
        for (expert, weight) in weighted {
            let w = &gate_up[expert];
            let h: Vec<f64> = (0..INNER)
                .map(|i| {
                    let dot = |row: usize| -> f64 {
                        (0..HIDDEN).map(|k| w[row * HIDDEN + k] * f64::from(input[k])).sum()
                    };
                    silu(dot(i)) * dot(INNER + i)
                })
                .collect();
            for j in 0..HIDDEN {
                let terms = (0..INNER).map(|i| down[expert][j * INNER + i] * h[i]);
                let (sum, size) = terms.fold((0.0, 0.0), |(s, m), t| (s + t, m + t.abs()));
                output[token * HIDDEN + j] += weight * sum;
                magnitude[token * HIDDEN + j] += weight.abs() * size;
            }
        }
    }
    (picks, output, magnitude)
}

#[test]
fn routed_experts_match_the_transformers_block() {
    let mut rng = Rng(7);
    let params =
        RoutedExpertsParams::new(TABLE as u32, SLOTS as u32, HIDDEN as u32, INNER as u32).unwrap();
    let (gate_up, gate_up_dense) = table(params.gate_up_record(), &mut rng);
    let (down, down_dense) = table(params.down_record(), &mut rng);
    let mut logits: Vec<f32> = (0..TOKENS * TABLE).map(|_| 2.0 * rng.unit()).collect();
    // A tie at the top on the last token: the lower index ranks first.
    logits[2 * TABLE + 1] = 3.0;
    logits[2 * TABLE + 4] = 3.0;
    let x: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| rng.unit()).collect();

    let program =
        NativeProgram::from_graph_in(block(&gate_up, &down), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut run = program.prepare_native_in(Global).unwrap();
    run.run(&[bytes(&logits), bytes(&x)]).unwrap();
    let routes = run.output_i32(0).unwrap().to_vec();
    let output = run.output_f32(1).unwrap().to_vec();

    let (picks, expected, magnitude) = reference(&logits, &x, &gate_up_dense, &down_dense);
    assert_eq!(&picks[2], &[1, 4], "the tie goes to the lower index first");
    for token in 0..TOKENS {
        let route = &routes[token * SLOTS * 2..(token + 1) * SLOTS * 2];
        let experts: Vec<usize> = route.chunks(2).map(|slot| slot[0] as usize).collect();
        assert_eq!(&experts[..TOP_K], &picks[token][..], "token {token}");
        assert_eq!(experts[TOP_K], EXPERTS, "the shared expert closes the route");
        let weights: Vec<f32> = route.chunks(2).map(|slot| f32::from_bits(slot[1] as u32)).collect();
        let picked: f32 = weights[..TOP_K].iter().sum();
        assert!((picked - 1.0).abs() < 1e-6, "renormalized: {picked}");
    }
    for (index, (&got, &want)) in output.iter().zip(&expected).enumerate() {
        let error = (f64::from(got) - want).abs();
        assert!(
            error <= 2e-6 * magnitude[index] + 1e-9,
            "output {index}: {got} against {want} (terms {})",
            magnitude[index]
        );
    }
}

#[test]
fn a_route_outside_the_table_is_refused() {
    let params =
        RoutedExpertsParams::new(TABLE as u32, SLOTS as u32, HIDDEN as u32, INNER as u32).unwrap();
    let mut rng = Rng(9);
    let (gate_up, _) = table(params.gate_up_record(), &mut rng);
    let mut b = GraphBuilder::new_in(Global);
    let t = 1;
    let x = b.tensor(DType::F32, TensorFlags::INPUT, &[t, HIDDEN as u32], &[]).unwrap();
    let routes = b.tensor(DType::I32, TensorFlags::INPUT, &[t, SLOTS as u32, 2], &[]).unwrap();
    let bytes_per = params.gate_up_record().bytes().unwrap() as u32;
    let table = constant(&mut b, &[TABLE as u32, bytes_per], &gate_up);
    b.moe_gate_up_nvfp4(x, table, routes, t, params, TensorFlags::OUTPUT).unwrap();
    let program =
        NativeProgram::from_graph_in(b.finish().unwrap(), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut run = program.prepare_native_in(Global).unwrap();
    let input = vec![0.5_f32; HIDDEN];
    let one = 1.0_f32.to_bits() as i32;
    let inside: Vec<i32> = vec![0, one, 1, one, TABLE as i32 - 1, one];
    let outside: Vec<i32> = vec![0, one, TABLE as i32, one, 1, one];
    let words = |route: &[i32]| -> Vec<u8> { route.iter().flat_map(|w| w.to_le_bytes()).collect() };
    run.run(&[bytes(&input), &words(&inside)]).unwrap();
    assert!(run.run(&[bytes(&input), &words(&outside)]).is_err());
}
