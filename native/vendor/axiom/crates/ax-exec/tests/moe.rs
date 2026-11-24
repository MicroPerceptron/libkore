#![feature(allocator_api)]
use ax::moe::{ExpertActivation, MoeParams, ResourceKey};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeResourceBinding};
use ax_exec::{KernelProgress, NativeRef, PreparedKernelExecutor, PreparedProgress, Serial};
use axnn::lowering::{KernelAccuracy, SemanticOp};
use axnn::{BinaryOp, UnaryOp};
use axnn_cpu::fp8::{e4m3_to_f32, f32_to_e4m3_rne};
use axnn_cpu::{CpuError, ExpertSource, ExpertStoreRef, ExpertWeights, WeightResidency};
use std::alloc::Global;
use std::sync::Mutex;

const STORE: ResourceKey = ResourceKey {
    group: 7,
    binding: 1,
};
const EXPERTS: usize = 8;
const TOP_K: usize = 3;
const HIDDEN: usize = 32;
const INNER: usize = 16;
const SHARED: usize = 24;
const SHARED_SCALE: f32 = 0.5;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 41 + seed * 19) % 53) as f32 - 26.0) / 20.0)
        .collect()
}

fn fp8(values: &[f32]) -> Vec<u8> {
    values.iter().map(|&v| f32_to_e4m3_rne(v)).collect()
}

fn decode(bytes: &[u8], scale: f32) -> Vec<f64> {
    bytes
        .iter()
        .map(|&b| f64::from(e4m3_to_f32(b)) * f64::from(scale))
        .collect()
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn params() -> MoeParams {
    MoeParams::new(
        STORE,
        EXPERTS as u32,
        TOP_K as u32,
        HIDDEN as u32,
        INNER as u32,
        DType::B8,
        ExpertActivation::SquaredRelu,
        true,
        2.5,
    )
    .unwrap()
}

struct Weights {
    router: Vec<u8>,
    bias: Vec<f32>,
    experts: Vec<(Vec<u8>, Vec<u8>, f32, f32)>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
}

fn weights() -> Weights {
    Weights {
        router: bf16(&values(EXPERTS * HIDDEN, 1)),
        bias: (0..EXPERTS)
            .map(|e| if e == 6 { 0.4 } else { 0.0 })
            .collect(),
        experts: (0..EXPERTS)
            .map(|e| {
                (
                    fp8(&values(INNER * HIDDEN, 10 + 2 * e)),
                    fp8(&values(HIDDEN * INNER, 11 + 2 * e)),
                    0.25 + e as f32 / 16.0,
                    0.5 + e as f32 / 32.0,
                )
            })
            .collect(),
        shared_up: fp8(&values(SHARED * HIDDEN, 2)),
        shared_down: fp8(&values(HIDDEN * SHARED, 3)),
    }
}

/// A Nemotron-H MoE block: F32 router over BF16 weights, sigmoid scores,
/// routed experts from the store, plus a squared-ReLU shared expert.
fn block(w: &Weights) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let none = NodeFlags::default();
    let hidden = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN as u32], &[])
        .unwrap();
    let constant = |b: &mut GraphBuilder<Global>, dtype, dims: &[u32], bytes: &[u8]| {
        b.tensor(dtype, TensorFlags::CONSTANT, dims, bytes).unwrap()
    };
    let f32_bytes =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let router = constant(
        &mut b,
        DType::B16,
        &[EXPERTS as u32, HIDDEN as u32],
        &w.router,
    );
    let bias = constant(&mut b, DType::F32, &[EXPERTS as u32], &f32_bytes(&w.bias));
    let up = constant(
        &mut b,
        DType::B8,
        &[SHARED as u32, HIDDEN as u32],
        &w.shared_up,
    );
    let down = constant(
        &mut b,
        DType::B8,
        &[HIDDEN as u32, SHARED as u32],
        &w.shared_down,
    );
    let scale = constant(&mut b, DType::F32, &[1], &SHARED_SCALE.to_le_bytes());
    let tensor = |b: &mut GraphBuilder<Global>, width: usize, flags| -> TensorId {
        b.tensor(DType::F32, flags, &[1, width as u32], &[])
            .unwrap()
    };
    let logits = tensor(&mut b, EXPERTS, TensorFlags::default());
    let scores = tensor(&mut b, EXPERTS, TensorFlags::default());
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        none,
        &[hidden, router],
        &[logits],
        &[],
    )
    .unwrap();
    b.node(OpCode::SIGMOID, none, &[logits], &[scores], &[])
        .unwrap();
    let routed = b
        .moe_experts(hidden, scores, bias, 1, params(), TensorFlags::default())
        .unwrap();

    let chain = |b: &mut GraphBuilder<Global>, op, inputs: &[TensorId], width| {
        let output = tensor(b, width, TensorFlags::default());
        b.node(op, none, inputs, &[output], &[]).unwrap();
        output
    };
    let up_raw = chain(&mut b, OpCode::MATMUL_TRANSPOSED, &[hidden, up], SHARED);
    let up_scaled = chain(&mut b, OpCode::MUL, &[up_raw, scale], SHARED);
    let rectified = chain(&mut b, OpCode::RELU, &[up_scaled], SHARED);
    let squared = chain(&mut b, OpCode::MUL, &[rectified, rectified], SHARED);
    let down_raw = chain(&mut b, OpCode::MATMUL_TRANSPOSED, &[squared, down], HIDDEN);
    let shared = chain(&mut b, OpCode::MUL, &[down_raw, scale], HIDDEN);
    let output = tensor(&mut b, HIDDEN, TensorFlags::OUTPUT);
    b.node(OpCode::ADD, none, &[routed, shared], &[output], &[])
        .unwrap();
    b.finish().unwrap()
}

fn reference(w: &Weights, x: &[f32]) -> Vec<f64> {
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let router: Vec<f64> = w
        .router
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&p| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(p)) << 16)))
        .collect();
    let matvec = |weights: &[f64], rows: usize, input: &[f64]| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                (0..input.len())
                    .map(|i| weights[r * input.len() + i] * input[i])
                    .sum()
            })
            .collect()
    };
    let scores: Vec<f64> = matvec(&router, EXPERTS, &x64)
        .iter()
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    let mut ranked: Vec<usize> = (0..EXPERTS).collect();
    let choice = |e: usize| scores[e] + f64::from(w.bias[e]);
    ranked.sort_by(|&a, &b| choice(b).total_cmp(&choice(a)).then(a.cmp(&b)));
    let picked = &ranked[..TOP_K];
    let total: f64 = picked.iter().map(|&e| scores[e]).sum::<f64>() + 1e-20;
    let mut out = vec![0.0; HIDDEN];
    for &e in picked {
        let (up, down, up_scale, down_scale) = &w.experts[e];
        let act: Vec<f64> = matvec(&decode(up, *up_scale), INNER, &x64)
            .iter()
            .map(|v| v.max(0.0).powi(2))
            .collect();
        let expert = matvec(&decode(down, *down_scale), HIDDEN, &act);
        for (o, v) in out.iter_mut().zip(expert) {
            *o += scores[e] / total * 2.5 * v;
        }
    }
    let shared_act: Vec<f64> = matvec(&decode(&w.shared_up, SHARED_SCALE), SHARED, &x64)
        .iter()
        .map(|v| v.max(0.0).powi(2))
        .collect();
    let shared = matvec(&decode(&w.shared_down, SHARED_SCALE), HIDDEN, &shared_act);
    out.iter().zip(shared).map(|(a, b)| a + b).collect()
}

#[test]
fn a_moe_block_routes_to_its_bound_experts() {
    let w = weights();
    let table: Vec<ExpertWeights<'_>> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let store = ExpertStoreRef::new(&table, params()).unwrap();
    let program =
        NativeProgram::from_graph_in(block(&w), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let bindings = [NativeResourceBinding::expert_store(STORE, store)];
    for seed in [30, 31, 32] {
        let x = values(HIDDEN, seed);
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        arena.run_with_resources(&[&bytes], &bindings).unwrap();
        let expected = reference(&w, &x);
        for (actual, expected) in arena.output_f32(0).unwrap().iter().zip(&expected) {
            assert!(
                (f64::from(*actual) - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                "seed {seed}: {actual} vs {expected}"
            );
        }
    }

    // A missing store, or one with the wrong number of experts, fails before
    // any node runs.
    let x: Vec<u8> = values(HIDDEN, 30)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert!(matches!(
        arena.run_with_resources(&[&x], &[]),
        Err(Error::MissingResource(STORE))
    ));
    // A store that is valid for a seven-expert layer is refused here.
    let seven = MoeParams {
        experts: EXPERTS as u32 - 1,
        ..params()
    };
    let short = ExpertStoreRef::new(&table[..EXPERTS - 1], seven).unwrap();
    assert!(matches!(
        arena.run_with_resources(&[&x], &[NativeResourceBinding::expert_store(STORE, short)]),
        Err(Error::Cpu(_))
    ));
    assert!(arena.output_f32(0).is_none());
}

/// Records which expert each hinted range belongs to.
struct Hints<'a> {
    table: &'a [ExpertWeights<'a>],
    seen: Mutex<Vec<(usize, &'static str)>>,
}

impl WeightResidency for Hints<'_> {
    fn will_read(&self, bytes: &[u8]) {
        let found = self.table.iter().enumerate().find_map(|(e, w)| {
            if core::ptr::eq(w.up, bytes) {
                Some((e, "up"))
            } else if core::ptr::eq(w.down, bytes) {
                Some((e, "down"))
            } else {
                None
            }
        });
        self.seen
            .lock()
            .unwrap()
            .push(found.expect("a hint names a whole projection"));
    }
}

/// The kernel names each pick's projections before reading any, in the
/// order it reads them, and the hint changes no result.
#[test]
fn routed_experts_are_named_before_they_are_read() {
    let w = weights();
    let table: Vec<ExpertWeights<'_>> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let hints = Hints {
        table: &table,
        seen: Mutex::new(Vec::new()),
    };
    let program =
        NativeProgram::from_graph_in(block(&w), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let x: Vec<u8> = values(HIDDEN, 31)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let plain = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::new(&table, params()).unwrap(),
    )];
    arena.run_with_resources(&[&x], &plain).unwrap();
    let expected: Vec<u32> = arena
        .output_f32(0)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let hinted = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::new(&table, params())
            .unwrap()
            .with_residency(&hints),
    )];
    arena.run_with_resources(&[&x], &hinted).unwrap();
    let actual: Vec<u32> = arena
        .output_f32(0)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(actual, expected);

    let seen = hints.seen.into_inner().unwrap();
    assert_eq!(seen.len(), 2 * TOP_K);
    let picks: Vec<usize> = seen.chunks(2).map(|pair| pair[0].0).collect();
    for (pair, &expert) in seen.chunks(2).zip(&picks) {
        assert_eq!(pair, [(expert, "up"), (expert, "down")]);
    }
    assert!(
        picks.windows(2).all(|p| p[0] < p[1]),
        "ascending: {picks:?}"
    );
}

/// Supplies a table's experts on demand, recording each load, or fails.
struct OnDemand<'a> {
    table: &'a [ExpertWeights<'a>],
    loads: Mutex<Vec<(Vec<u32>, usize)>>,
    fail: bool,
}

unsafe impl ExpertSource for OnDemand<'_> {
    fn experts(&self) -> usize {
        self.table.len()
    }
    fn load<'s>(
        &'s self,
        picks: &[u32],
        hidden: &[f32],
        out: &mut [ExpertWeights<'s>],
    ) -> Result<(), CpuError> {
        if self.fail {
            return Err(CpuError::ResourceUnavailable { op: "test source" });
        }
        self.loads
            .lock()
            .unwrap()
            .push((picks.to_vec(), hidden.len()));
        for (weights, &expert) in out.iter_mut().zip(picks) {
            *weights = self.table[expert as usize];
        }
        Ok(())
    }
}

/// A source-backed store computes exactly what the table does, loading
/// each token's picks once, ascending, with the token's hidden row.
#[test]
fn a_source_supplies_each_tokens_picks_on_demand() {
    let w = weights();
    let table: Vec<ExpertWeights<'_>> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let program =
        NativeProgram::from_graph_in(block(&w), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let x: Vec<u8> = values(HIDDEN, 32)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bits = |arena: &ax_exec::PreparedNative<Global>| -> Vec<u32> {
        arena
            .output_f32(0)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let table_store = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::new(&table, params()).unwrap(),
    )];
    arena.run_with_resources(&[&x], &table_store).unwrap();
    let expected = bits(&arena);

    let source = OnDemand {
        table: &table,
        loads: Mutex::new(Vec::new()),
        fail: false,
    };
    let sourced = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::from_source(&source, params()).unwrap(),
    )];
    arena.run_with_resources(&[&x], &sourced).unwrap();
    assert_eq!(bits(&arena), expected);
    let loads = source.loads.into_inner().unwrap();
    assert_eq!(loads.len(), 1);
    let (picks, hidden) = &loads[0];
    assert_eq!((picks.len(), *hidden), (TOP_K, HIDDEN));
    assert!(
        picks.windows(2).all(|p| p[0] < p[1]),
        "ascending: {picks:?}"
    );

    // A failed load fails the run and publishes nothing.
    let failing = OnDemand {
        table: &table,
        loads: Mutex::new(Vec::new()),
        fail: true,
    };
    let broken = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::from_source(&failing, params()).unwrap(),
    )];
    assert!(matches!(
        arena.run_with_resources(&[&x], &broken),
        Err(Error::Cpu(CpuError::ResourceUnavailable { .. }))
    ));
    assert!(arena.output_f32(0).is_none());
}

/// Takes each wave on its first offer, as a device would, and finishes it
/// after `wait`: every route's squared-ReLU expert, weighted, added to its
/// token's row in route order.
#[derive(Default)]
struct DeferredExperts {
    capacity: Option<usize>,
    pending: Option<(usize, usize)>,
    ready: bool,
    completed: usize,
    take: bool,
    decline_pending_after_completed: Option<usize>,
    decline_after_completed: Option<usize>,
    fail_wait: bool,
}

impl PreparedKernelExecutor for DeferredExperts {
    fn moe_wave_capacity(&self, _: MoeParams, _: usize, _: &[axnn_cpu::MoeBatchRow]) -> usize {
        assert!(self.pending.is_none(), "never resize a live wave");
        self.capacity.unwrap_or(axnn_cpu::DEFAULT_WAVE_EXPERTS)
    }

    fn advance_moe_experts(
        &mut self,
        instruction: usize,
        wave: usize,
        _: &axnn_cpu::CpuKernelPlan,
        p: MoeParams,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        assert_eq!(p, params());
        assert_eq!(minimum, KernelAccuracy::Faithful);
        if !self.take {
            return Ok(KernelProgress::Unsupported);
        }
        if self.pending.is_none() && self.decline_after_completed == Some(self.completed) {
            return Ok(KernelProgress::Unsupported);
        }
        if let Some(identity) = self.pending {
            assert_eq!(identity, (instruction, wave));
            assert!(self.ready);
            if self.decline_pending_after_completed == Some(self.completed) {
                return Ok(KernelProgress::Unsupported);
            }
        } else {
            self.pending = Some((instruction, wave));
            self.ready = false;
            return Ok(KernelProgress::Pending);
        }
        assert!(routes.windows(2).all(|pair| {
            (pair[0].expert(), pair[0].token()) < (pair[1].expert(), pair[1].token())
        }));
        let dot = |weights: &[u8], input: &[f32]| -> f32 {
            weights
                .iter()
                .zip(input)
                .fold(0.0, |sum, (&w, &x)| sum + e4m3_to_f32(w) * x)
        };
        let mut pick = 0;
        for (index, route) in routes.iter().enumerate() {
            if index > 0 && route.expert() != routes[index - 1].expert() {
                pick += 1;
            }
            let expert = &experts[pick];
            let token = route.token();
            let input = &hidden[token * HIDDEN..(token + 1) * HIDDEN];
            let act: Vec<f32> = expert
                .up
                .chunks_exact(HIDDEN)
                .map(|row| (expert.up_scale * dot(row, input)).max(0.0).powi(2))
                .collect();
            for (out, row) in output[token * HIDDEN..(token + 1) * HIDDEN]
                .iter_mut()
                .zip(expert.down.chunks_exact(INNER))
            {
                *out += route.mixture() * expert.down_scale * dot(row, &act);
            }
        }
        assert_eq!(pick + 1, experts.len());
        self.completed += 1;
        self.pending = None;
        Ok(KernelProgress::Complete)
    }

    fn begin(&mut self) -> Result<(), Error> {
        if self.pending.is_some() {
            return Err(Error::InvocationFinished);
        }
        self.completed = 0;
        Ok(())
    }

    async fn wait(&mut self) -> Result<(), Error> {
        assert!(self.pending.is_some());
        if self.fail_wait {
            return Err(Error::InvocationFinished);
        }
        self.ready = true;
        Ok(())
    }

    fn advance(
        &mut self,
        _: usize,
        _: &axnn_cpu::CpuKernelPlan,
        _: &[Option<NativeRef<'_>>; 3],
        _: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        assert!(
            self.pending.is_none(),
            "successor must not run across an external wait"
        );
        Ok(KernelProgress::Unsupported)
    }
}

fn wait_ready(future: impl core::future::Future<Output = Result<(), Error>>) -> Result<(), Error> {
    use core::task::{Context, Poll, Waker};
    let mut future = core::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("the test completion is already ready"),
    }
}

#[test]
fn prepared_expert_waits_keep_loads_and_successors_ordered() {
    let w = weights();
    let table: Vec<_> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let tokens = 3usize;
    let program = moe_followed_by_negate(&w, tokens);
    let mut arena = program.prepare_native_in(Global).unwrap();
    let x = values(tokens * HIDDEN, 29);
    let scores: Vec<_> = values(tokens * EXPERTS, 30)
        .into_iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    let inputs = [NativeRef::F32(&x), NativeRef::F32(&scores)];
    let baseline = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::new(&table, params()).unwrap(),
    )];
    arena
        .run_with_native_inputs_and_resources(&inputs, &baseline)
        .unwrap();
    let expected: Vec<_> = arena
        .output_f32(0)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();

    // The source loads at most `TOP_K` experts at once, so the routes of
    // these three tokens take several waves.
    let mut selected = Vec::new();
    for scores in scores.chunks_exact(EXPERTS) {
        let mut ranked: Vec<_> = (0..EXPERTS).collect();
        ranked.sort_by(|&a, &b| {
            (scores[b] + w.bias[b])
                .total_cmp(&(scores[a] + w.bias[a]))
                .then(a.cmp(&b))
        });
        selected.extend_from_slice(&ranked[..TOP_K]);
    }
    selected.sort_unstable();
    selected.dedup();
    let waves = selected.len().div_ceil(TOP_K);
    assert!(waves > 1);
    let output_bits = |arena: &ax_exec::PreparedNative<Global>| -> Option<Vec<u32>> {
        arena
            .output_f32(0)
            .map(|values| values.iter().map(|v| v.to_bits()).collect())
    };

    // The last run checks every wave the device completes against the
    // native kernel, whose result then stands.
    for (take, failure, check, capacity) in [(false, 0, false)]
        .into_iter()
        .chain((0..6).map(|f| (true, f, false)))
        .chain([(true, 0, true)])
        .map(|(take, failure, check)| (take, failure, check, None))
        .chain([(true, 0, false, Some(1))])
    {
        let limit = capacity.unwrap_or(TOP_K).min(TOP_K);
        let waves = selected.len().div_ceil(limit);
        arena.set_delegate_check(check);
        arena.set_profiler(|| 0);
        let source = OnDemand {
            table: &table,
            loads: Mutex::new(Vec::new()),
            fail: false,
        };
        let resources = [NativeResourceBinding::expert_store(
            STORE,
            ExpertStoreRef::from_source(&source, params()).unwrap(),
        )];
        let mut executor = DeferredExperts {
            capacity,
            take,
            decline_pending_after_completed: match failure {
                1 => Some(0),
                5 => Some(1),
                _ => None,
            },
            decline_after_completed: (failure == 4).then_some(1),
            fail_wait: failure == 2,
            ..Default::default()
        };
        let mut run = arena
            .begin_with_executor(
                &inputs,
                &resources,
                &Serial,
                &mut executor,
                KernelAccuracy::Faithful,
            )
            .unwrap();
        if !take {
            // A declined wave runs natively, and so does every later one.
            assert_eq!(run.advance(8).unwrap(), PreparedProgress::Complete);
            drop(run);
            assert_eq!(executor.completed, 0);
            assert_eq!(source.loads.lock().unwrap().len(), waves);
            assert_eq!(output_bits(&arena).unwrap(), expected);
            let profile = arena.profile();
            assert_eq!(profile.moe.native_waves, waves as u64);
            assert_eq!(profile.moe.native_routes, (tokens * TOP_K) as u64);
            assert_eq!(profile.moe.delegated_waves, 0);
            assert_eq!(profile.moe.mixed_invocations, 0);
            continue;
        }
        assert_eq!(run.advance(8).unwrap(), PreparedProgress::Pending);
        assert_eq!(source.loads.lock().unwrap().len(), 1);
        if failure == 3 {
            drop(run);
            assert!(
                executor.begin().is_err(),
                "abandonment does not cancel runtime work"
            );
        } else if failure == 2 {
            assert!(wait_ready(run.wait()).is_err());
            assert!(run.advance(8).is_err());
            drop(run);
        } else if failure == 1 {
            wait_ready(run.wait()).unwrap();
            assert!(
                run.advance(8).is_err(),
                "cannot fallback after pending admission"
            );
            assert!(run.advance(8).is_err());
            drop(run);
        } else if failure == 5 {
            wait_ready(run.wait()).unwrap();
            assert_eq!(run.advance(8).unwrap(), PreparedProgress::Pending);
            wait_ready(run.wait()).unwrap();
            assert!(
                run.advance(8).is_err(),
                "uncertain second wave cannot fall back"
            );
            drop(run);
            assert_eq!(executor.completed, 1);
            let profile = arena.profile();
            assert_eq!(
                profile.moe.delegated_waves, 1,
                "failure must not erase earlier completed work"
            );
            assert_eq!(profile.moe.native_waves, 0);
            assert_eq!(profile.moe.mixed_invocations, 0);
            assert!(arena.output_f32(0).is_none());
            continue;
        } else if failure == 4 {
            // One device wave succeeds; the next declines before admission.
            // The engine's existing sticky fallback executes the remainder
            // natively, and the profile must expose this mixed operation.
            wait_ready(run.wait()).unwrap();
            assert_eq!(run.advance(8).unwrap(), PreparedProgress::Complete);
            drop(run);
            assert_eq!(executor.completed, 1);
            let profile = arena.profile();
            assert_eq!(profile.moe.delegated_waves, 1);
            assert_eq!(profile.moe.native_waves, waves as u64 - 1);
            assert_eq!(profile.moe.mixed_invocations, 1);
            assert_eq!(
                profile.moe.native_routes + profile.moe.delegated_routes,
                (tokens * TOP_K) as u64
            );
            for (actual, expected) in arena.output_f32(0).unwrap().iter().zip(&expected) {
                let expected = f32::from_bits(*expected);
                assert!((actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()));
            }
            continue;
        } else {
            for wave in 0..waves {
                assert_eq!(source.loads.lock().unwrap().len(), wave + 1);
                wait_ready(run.wait()).unwrap();
                assert_eq!(
                    run.advance(8).unwrap(),
                    if wave + 1 == waves {
                        PreparedProgress::Complete
                    } else {
                        PreparedProgress::Pending
                    }
                );
            }
            drop(run);
            assert_eq!(executor.completed, waves);
            let profile = arena.profile();
            assert_eq!(profile.moe.delegated_waves, waves as u64);
            assert_eq!(profile.moe.delegated_routes, (tokens * TOP_K) as u64);
            assert_eq!(profile.moe.native_waves, 0);
            assert_eq!(profile.moe.mixed_invocations, 0);
            assert_eq!(
                profile.reference.runs[0],
                if check { waves as u64 } else { 0 }
            );
            let loads = source.loads.lock().unwrap();
            for ((picks, len), wave) in loads.iter().zip(selected.chunks(limit)) {
                let wave: Vec<u32> = wave.iter().map(|&e| e as u32).collect();
                assert_eq!(picks, &wave);
                assert_eq!(*len, HIDDEN);
            }
            if check {
                let report = arena.delegate_check();
                assert_eq!(report.checked[0], waves as u64);
                assert!(report.worst[0] < 1e-4, "{report}");
                assert_eq!(output_bits(&arena).unwrap(), expected);
                continue;
            }
            // The device's own reduction order: close to, not bitwise, the
            // native kernel's.
            for (actual, expected) in arena.output_f32(0).unwrap().iter().zip(&expected) {
                let expected = f32::from_bits(*expected);
                assert!(
                    (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "{actual} vs {expected}"
                );
            }
        }
        if failure != 0 {
            assert_eq!(source.loads.lock().unwrap().len(), 1);
            assert!(arena.output_f32(0).is_none());
        }
    }
}

/// A window executor over the MoE block: it admits the router's sigmoid and
/// the elementwise steps the schedule interleaves with it (the shared
/// expert's), then the whole routed-expert instruction with those scores
/// still pending, and publishes each admission once a `wait` has made it
/// visible. Projections it declines, draining what is pending. The
/// experts run through the native routed-expert kernel on the scores the
/// executor itself produced, as a device reads its own results in place.
#[derive(Default)]
struct RoutedWindow {
    take: bool,
    /// Results by instruction, as the device holds them.
    results: std::collections::BTreeMap<usize, Vec<f32>>,
    /// Admitted and not yet completed, in admission order.
    queue: Vec<usize>,
    flushed: usize,
    waits: usize,
    /// Routed-expert offers, those whose scores were pending, and declines.
    offers: usize,
    pending_scores: usize,
    declined: usize,
}

impl RoutedWindow {
    /// A retry of an admission: Pending until a wait made it visible, then
    /// its result.
    fn retry(&mut self, instruction: usize, output: &mut [f32]) -> Option<KernelProgress> {
        let position = self
            .queue
            .iter()
            .position(|&queued| queued == instruction)?;
        assert_eq!(position, 0, "admissions retire in admission order");
        if self.flushed == 0 {
            return Some(KernelProgress::Pending);
        }
        output.copy_from_slice(&self.results[&instruction]);
        self.queue.remove(0);
        self.flushed -= 1;
        Some(KernelProgress::Complete)
    }

    fn operand(&self, input: Option<ax_exec::ChainedInput<'_>>) -> (Vec<f32>, bool) {
        match input.expect("operand") {
            ax_exec::ChainedInput::Host(NativeRef::F32(values)) => (values.to_vec(), false),
            ax_exec::ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => {
                assert!(
                    self.queue.contains(&instruction),
                    "pending operand of a live admission"
                );
                (
                    self.results[&instruction][offset..offset + elements].to_vec(),
                    true,
                )
            }
            ax_exec::ChainedInput::Host(_) => panic!("F32 operands only"),
        }
    }
}

impl PreparedKernelExecutor for RoutedWindow {
    async fn wait(&mut self) -> Result<(), Error> {
        self.waits += 1;
        self.flushed = self.queue.len();
        Ok(())
    }

    fn begin(&mut self) -> Result<(), Error> {
        assert!(self.queue.is_empty());
        self.results.clear();
        Ok(())
    }

    fn admission_window(&self) -> usize {
        8
    }

    fn advance(
        &mut self,
        _: usize,
        _: &axnn_cpu::CpuKernelPlan,
        _: &[Option<NativeRef<'_>>; 3],
        _: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        Ok(KernelProgress::Unsupported)
    }

    fn advance_chained(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if let Some(progress) = self.retry(instruction, output) {
            return Ok(progress);
        }
        let operand = |input| self.operand(input).0;
        let result: Vec<f32> = match plan.semantic {
            SemanticOp::Unary(UnaryOp::Sigmoid) => operand(inputs[0])
                .iter()
                .map(|&v| 1.0 / (1.0 + (-v).exp()))
                .collect(),
            SemanticOp::Unary(UnaryOp::Relu) => {
                operand(inputs[0]).iter().map(|&v| v.max(0.0)).collect()
            }
            // Row-major trailing-axis broadcast, as `Binary` lowers it.
            SemanticOp::Binary(BinaryOp::Mul) => {
                let (a, b) = (operand(inputs[0]), operand(inputs[1]));
                (0..a.len().max(b.len()))
                    .map(|i| a[i % a.len()] * b[i % b.len()])
                    .collect()
            }
            _ => return Ok(KernelProgress::Unsupported),
        };
        assert_eq!(result.len(), output.len());
        self.results.insert(instruction, result);
        self.queue.push(instruction);
        Ok(KernelProgress::Pending)
    }

    fn advance_moe_routed(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        store: ExpertStoreRef<'_>,
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if let Some(progress) = self.retry(instruction, output) {
            return Ok(progress);
        }
        self.offers += 1;
        if !self.take {
            self.declined += 1;
            return Ok(KernelProgress::Unsupported);
        }
        let mut resident = vec![ExpertWeights::EMPTY; EXPERTS];
        assert!(
            store.resident(&mut resident),
            "a table holds every expert in place"
        );
        let (hidden, _) = self.operand(inputs[0]);
        let (scores, pending) = self.operand(inputs[1]);
        let (bias, _) = self.operand(inputs[2]);
        self.pending_scores += usize::from(pending);
        let tokens = hidden.len() / HIDDEN;
        let mut routed = vec![f32::NAN; output.len()];
        let mut scratch = vec![0.0; axnn_cpu::moe_scratch_elements(params()) * tokens + 4096];
        axnn_cpu::invoke_moe_experts(
            plan,
            store,
            &hidden,
            &scores,
            &bias,
            &mut routed,
            &mut scratch,
            &Serial,
        )
        .map_err(Error::Cpu)?;
        self.results.insert(instruction, routed);
        self.queue.push(instruction);
        Ok(KernelProgress::Pending)
    }
}

/// A routed-expert instruction rides the admission window whole: offered
/// right after the router, with its scores still pending, it
/// needs no drain between the router and the experts, and the block's
/// result is the reference's. A declining executor gets the drained,
/// host-routed path it had before.
#[test]
fn a_routed_expert_instruction_rides_the_window_with_its_scores_pending() {
    let w = weights();
    let table: Vec<ExpertWeights<'_>> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let program =
        NativeProgram::from_graph_in(block(&w), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let resources = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::new(&table, params()).unwrap(),
    )];
    for take in [true, false] {
        for seed in [30, 31] {
            let x = values(HIDDEN, seed);
            let inputs = [NativeRef::F32(&x)];
            let mut executor = RoutedWindow {
                take,
                ..Default::default()
            };
            let mut run = arena
                .begin_with_executor(
                    &inputs,
                    &resources,
                    &Serial,
                    &mut executor,
                    KernelAccuracy::Faithful,
                )
                .unwrap();
            let mut rounds = 0;
            loop {
                match run.advance(64).unwrap() {
                    PreparedProgress::Complete => break,
                    PreparedProgress::Pending => wait_ready(run.wait()).unwrap(),
                    PreparedProgress::Progress { .. } => {}
                }
                rounds += 1;
                assert!(rounds < 64, "the block finishes");
            }
            drop(run);
            if take {
                assert_eq!(executor.offers, 1, "offered once, then retried in place");
                assert_eq!(
                    executor.pending_scores, 1,
                    "the scores never reached the host"
                );
                assert_eq!(executor.declined, 0);
            } else {
                // Offered behind the pending router, declined, drained, then
                // offered again with host scores and declined into the
                // strict path: the router's result had to reach the host.
                assert_eq!(executor.offers, 2);
                assert_eq!(executor.declined, 2);
                assert_eq!(executor.pending_scores, 0);
            }
            assert!(executor.queue.is_empty());
            let expected = reference(&w, &x);
            for (actual, expected) in arena.output_f32(0).unwrap().iter().zip(&expected) {
                assert!(
                    (f64::from(*actual) - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "take {take}, seed {seed}: {actual} vs {expected}"
                );
            }
        }
    }
}


fn moe_followed_by_negate(w: &Weights, tokens: usize) -> NativeProgram<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let hidden = b
        .tensor(
            DType::F32,
            TensorFlags::INPUT,
            &[tokens as u32, HIDDEN as u32],
            &[],
        )
        .unwrap();
    let scores = b
        .tensor(
            DType::F32,
            TensorFlags::INPUT,
            &[tokens as u32, EXPERTS as u32],
            &[],
        )
        .unwrap();
    let bias_bytes: Vec<_> = w.bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bias = b
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[EXPERTS as u32],
            &bias_bytes,
        )
        .unwrap();
    let routed = b
        .moe_experts(
            hidden,
            scores,
            bias,
            tokens as u32,
            params(),
            TensorFlags::default(),
        )
        .unwrap();
    let output = b
        .tensor(
            DType::F32,
            TensorFlags::OUTPUT,
            &[tokens as u32, HIDDEN as u32],
            &[],
        )
        .unwrap();
    b.node(
        OpCode::NEGATE,
        NodeFlags::default(),
        &[routed],
        &[output],
        &[],
    )
    .unwrap();
    NativeProgram::from_graph_in(
        b.finish().unwrap(),
        AxCpuKernelPolicy {
            capabilities: axnn_cpu::CpuCapabilities::SCALAR,
            ..Default::default()
        },
        false,
        Global,
    )
    .unwrap()
}

/// Keeps simulated device output separate from host output. A successor must
/// consume that device value before wait/retirement to exercise the handoff.
struct ChainedExperts {
    capacity: usize,
    expose: bool,
    pending: Option<(usize, usize, Vec<f32>)>,
    follower: Option<(usize, Vec<f32>)>,
    ready: bool,
    chained: usize,
    completed: usize,
    discarded: usize,
    fail_wait: bool,
}

impl PreparedKernelExecutor for ChainedExperts {
    fn begin(&mut self) -> Result<(), Error> {
        if self.pending.is_some() || self.follower.is_some() {
            return Err(Error::InvocationFinished);
        }
        Ok(())
    }
    fn admission_window(&self) -> usize {
        8
    }
    fn moe_wave_capacity(&self, _: MoeParams, _: usize, _: &[axnn_cpu::MoeBatchRow]) -> usize {
        assert!(self.pending.is_none());
        self.capacity
    }
    fn moe_output_pending(&self, instruction: usize, wave: usize) -> bool {
        self.expose
            && self
                .pending
                .as_ref()
                .is_some_and(|&(i, w, _)| (i, w) == (instruction, wave))
    }
    fn retire_moe_unobserved(
        &mut self,
        instruction: usize,
        wave: usize,
        _: usize,
    ) -> Result<KernelProgress, Error> {
        let (i, w, _) = self.pending.as_ref().expect("retained final wave");
        assert_eq!((*i, *w), (instruction, wave));
        assert!(
            self.follower.is_some(),
            "consumer was admitted before skipping publication"
        );
        if !self.ready {
            return Ok(KernelProgress::Pending);
        }
        self.pending = None;
        self.completed += 1;
        self.discarded += 1;
        Ok(KernelProgress::Complete)
    }
    fn advance_moe_experts(
        &mut self,
        instruction: usize,
        wave: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        p: MoeParams,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if let Some((i, w, values)) = &self.pending {
            assert_eq!((*i, *w), (instruction, wave));
            if !self.ready {
                return Ok(KernelProgress::Pending);
            }
            output.copy_from_slice(values);
            self.pending = None;
            self.completed += 1;
            return Ok(KernelProgress::Complete);
        }
        let mut values = output.to_vec();
        axnn_cpu::native_wave(
            plan,
            p,
            hidden,
            routes,
            experts,
            &mut values,
            &mut vec![0.0; p.hidden as usize],
            &mut vec![0.0; axnn_cpu::moe_scratch_elements(p)],
            &Serial,
        )?;
        self.pending = Some((instruction, wave, values));
        self.ready = false;
        Ok(KernelProgress::Pending)
    }
    fn advance_chained(
        &mut self,
        instruction: usize,
        _: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if let Some((i, values)) = &self.follower {
            assert_eq!(*i, instruction);
            assert!(self.pending.is_none(), "producer retires before consumer");
            if !self.ready {
                return Ok(KernelProgress::Pending);
            }
            output.copy_from_slice(values);
            self.follower = None;
            return Ok(KernelProgress::Complete);
        }
        match inputs[0] {
            Some(ax_exec::ChainedInput::Pending {
                instruction: producer,
                offset,
                elements,
            }) => {
                let (i, _, values) = self.pending.as_ref().expect("retained expert wave");
                assert_eq!(*i, producer);
                assert!(!self.ready, "consumer admitted before device completion");
                self.follower = Some((
                    instruction,
                    values[offset..offset + elements]
                        .iter()
                        .map(|v| -v)
                        .collect(),
                ));
                self.chained += 1;
                Ok(KernelProgress::Pending)
            }
            _ => {
                assert!(self.pending.is_none(), "native consumer must wait");
                Ok(KernelProgress::Unsupported)
            }
        }
    }
    fn advance(
        &mut self,
        _: usize,
        _: &axnn_cpu::CpuKernelPlan,
        _: &[Option<NativeRef<'_>>; 3],
        _: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        Ok(KernelProgress::Unsupported)
    }
    async fn wait(&mut self) -> Result<(), Error> {
        if self.fail_wait {
            return Err(Error::InvocationFinished);
        }
        self.ready = true;
        Ok(())
    }
}

#[test]
fn final_expert_wave_admits_device_consumer_before_completion() {
    let w = weights();
    let table: Vec<_> = w
        .experts
        .iter()
        .map(|(up, down, up_scale, down_scale)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: *up_scale,
            down_scale: *down_scale,
        })
        .collect();
    let tokens = 3;
    let program = moe_followed_by_negate(&w, tokens);
    let mut arena = program.prepare_native_in(Global).unwrap();
    let x = values(tokens * HIDDEN, 29);
    let scores = vec![0.5; tokens * EXPERTS];
    let inputs = [NativeRef::F32(&x), NativeRef::F32(&scores)];
    let source = OnDemand {
        table: &table,
        loads: Mutex::new(Vec::new()),
        fail: false,
    };
    let resources = [NativeResourceBinding::expert_store(
        STORE,
        ExpertStoreRef::from_source(&source, params()).unwrap(),
    )];
    arena
        .run_with_native_inputs_and_resources(&inputs, &resources)
        .unwrap();
    let expected = arena.output_f32(0).unwrap().to_vec();
    for capacity in [1, axnn_cpu::MAX_WAVE_EXPERTS] {
        for (expose, check) in [(false, false), (true, false), (true, true)] {
            arena.set_delegate_check(check);
            for fail_wait in [false, true] {
                let mut executor = ChainedExperts {
                    capacity,
                    expose,
                    pending: None,
                    follower: None,
                    ready: false,
                    chained: 0,
                    completed: 0,
                    discarded: 0,
                    fail_wait,
                };
                let mut run = arena
                    .begin_with_executor(
                        &inputs,
                        &resources,
                        &Serial,
                        &mut executor,
                        KernelAccuracy::Faithful,
                    )
                    .unwrap();
                let mut finished = false;
                for _ in 0..32 {
                    match run.advance(32).unwrap() {
                        PreparedProgress::Complete => {
                            finished = true;
                            break;
                        }
                        PreparedProgress::Pending => {
                            if fail_wait {
                                assert!(wait_ready(run.wait()).is_err());
                                break;
                            }
                            wait_ready(run.wait()).unwrap();
                        }
                        PreparedProgress::Progress { .. } => {}
                    }
                }
                drop(run);
                if fail_wait {
                    assert!(!finished);
                    assert!(
                        executor.pending.is_some(),
                        "failed wait retains device ownership"
                    );
                    assert!(executor.begin().is_err());
                } else {
                    assert!(finished);
                    assert_eq!(executor.chained, usize::from(expose));
                    assert_eq!(executor.discarded, usize::from(expose && !check));
                    assert!(executor.pending.is_none() && executor.follower.is_none());
                    assert_eq!(arena.output_f32(0).unwrap(), expected);
                    assert_eq!(executor.completed, if capacity == 1 { TOP_K } else { 1 });
                }
            }
        }
    }
}
