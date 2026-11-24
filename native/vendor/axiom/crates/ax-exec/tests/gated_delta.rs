#![feature(allocator_api)]
//! The recurrent core of a Gated DeltaNet mixer through the native engine:
//! the depthwise convolution over the query/key/value projection (SiLU, no
//! bias), the decay `-exp(A_log) * softplus(a + dt_bias)` and the write
//! strength `sigmoid(b)`, then the gated delta rule.
use ax::recurrent::{
    CausalConv1dParams, EpilogueActivation, GatedDeltaRuleParams, ResourceKey, StateBinding,
};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeResourceBinding};
use axnn_cpu::{StateSourceRef, StateTargetRef};
use std::alloc::Global;

const KEY_HEADS: usize = 2;
const KEY_DIM: usize = 4;
const VALUE_HEADS: usize = 6;
const VALUE_DIM: usize = 3;
const QKV: usize = 2 * KEY_HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM;
const KERNEL: usize = 4;
const CONV_STATE: usize = QKV * (KERNEL - 1);
const RULE_STATE: usize = VALUE_HEADS * KEY_DIM * VALUE_DIM;
const OUTPUT: usize = VALUE_HEADS * VALUE_DIM;

fn key(binding: u32) -> ResourceKey {
    ResourceKey { group: 5, binding }
}

const CONV: [u32; 2] = [0, 1];
const RULE: [u32; 2] = [2, 3];

fn binding(pair: [u32; 2]) -> StateBinding {
    StateBinding {
        source: key(pair[0]),
        target: key(pair[1]),
    }
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 31 + seed * 17) % 23) as f32 - 11.0) / 16.0)
        .collect()
}

fn constant(b: &mut GraphBuilder<Global>, dims: &[u32], values: &[f32]) -> TensorId {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.tensor(DType::F32, TensorFlags::CONSTANT, dims, &bytes)
        .unwrap()
}

fn unary(b: &mut GraphBuilder<Global>, op: OpCode, input: TensorId, dims: &[u32]) -> TensorId {
    let output = b
        .tensor(DType::F32, TensorFlags::default(), dims, &[])
        .unwrap();
    b.node(op, NodeFlags::default(), &[input], &[output], &[])
        .unwrap();
    output
}

fn binary(
    b: &mut GraphBuilder<Global>,
    op: OpCode,
    inputs: [TensorId; 2],
    dims: &[u32],
) -> TensorId {
    let output = b
        .tensor(DType::F32, TensorFlags::default(), dims, &[])
        .unwrap();
    b.node(op, NodeFlags::default(), &inputs, &[output], &[])
        .unwrap();
    output
}

/// The mixer core over `tokens` tokens: inputs are the projected `qkv`,
/// `a` and `b` rows; the output is the rule's `[tokens, heads * dim]`.
fn mixer(tokens: u32) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let heads = VALUE_HEADS as u32;
    let rows = [tokens, heads];
    let qkv = b
        .tensor(DType::F32, TensorFlags::INPUT, &[tokens, QKV as u32], &[])
        .unwrap();
    let a = b
        .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
        .unwrap();
    let write = b
        .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
        .unwrap();
    let taps = constant(
        &mut b,
        &[QKV as u32, KERNEL as u32],
        &values(QKV * KERNEL, 1),
    );
    let bias = constant(&mut b, &[QKV as u32], &[0.0; QKV]);
    let conv = CausalConv1dParams::new_activated(
        binding(CONV),
        QKV as u32,
        KERNEL as u32,
        EpilogueActivation::Silu,
    )
    .unwrap();
    let mixed = b
        .causal_conv1d(qkv, taps, bias, tokens, conv, TensorFlags::default())
        .unwrap();
    // Per-head parameters, one row per token so each gate op is
    // shape-preserving.
    let dt_bias: Vec<f32> = values(VALUE_HEADS, 2).repeat(tokens as usize);
    let rate: Vec<f32> = values(VALUE_HEADS, 3)
        .iter()
        .map(|a_log| -a_log.exp())
        .collect::<Vec<_>>()
        .repeat(tokens as usize);
    let dt_bias = constant(&mut b, &rows, &dt_bias);
    let rate = constant(&mut b, &rows, &rate);
    let shifted = binary(&mut b, OpCode::ADD, [a, dt_bias], &rows);
    let step = unary(&mut b, OpCode::SOFTPLUS, shifted, &rows);
    let decay = binary(&mut b, OpCode::MUL, [step, rate], &rows);
    let beta = unary(&mut b, OpCode::SIGMOID, write, &rows);
    let rule = GatedDeltaRuleParams::new(
        binding(RULE),
        (KEY_HEADS as u32, KEY_DIM as u32),
        (heads, VALUE_DIM as u32),
        true,
    )
    .unwrap();
    b.gated_delta_rule(mixed, decay, beta, tokens, rule, TensorFlags::OUTPUT)
        .unwrap();
    b.finish().unwrap()
}

fn bytes(values: &[f32]) -> &[u8] {
    // SAFETY: initialized, padding-free f32 words, borrowed immutably.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

/// Two alternating buffers per recurrence, as a decode loop keeps them.
struct States {
    conv: [Vec<f32>; 2],
    rule: [Vec<f32>; 2],
}

impl States {
    fn new() -> Self {
        Self {
            conv: [values(CONV_STATE, 4), vec![0.0; CONV_STATE]],
            rule: [values(RULE_STATE, 5), vec![0.0; RULE_STATE]],
        }
    }

    fn bindings(&mut self) -> [NativeResourceBinding<'_>; 4] {
        let [conv_source, conv_target] = &mut self.conv;
        let [rule_source, rule_target] = &mut self.rule;
        [
            NativeResourceBinding::state_source(key(CONV[0]), StateSourceRef::new(conv_source)),
            NativeResourceBinding::state_target(key(CONV[1]), StateTargetRef::new(conv_target)),
            NativeResourceBinding::state_source(key(RULE[0]), StateSourceRef::new(rule_source)),
            NativeResourceBinding::state_target(key(RULE[1]), StateTargetRef::new(rule_target)),
        ]
    }

    fn swap(&mut self) {
        self.conv.swap(0, 1);
        self.rule.swap(0, 1);
    }
}

/// Inputs for `tokens` tokens: qkv rows, then the decay and write gates.
fn inputs(tokens: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        values(tokens * QKV, 6),
        values(tokens * VALUE_HEADS, 7),
        values(tokens * VALUE_HEADS, 8),
    )
}

#[test]
fn stepping_with_alternating_buffers_equals_one_chunk() {
    let tokens = 3;
    let (qkv, a, write) = inputs(tokens);
    let chunked =
        NativeProgram::from_graph_in(mixer(3), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut chunk = chunked.prepare_native_in(Global).unwrap();
    let mut chunk_states = States::new();
    chunk
        .run_with_resources(
            &[bytes(&qkv), bytes(&a), bytes(&write)],
            &chunk_states.bindings(),
        )
        .unwrap();
    let chunk_output = chunk.output_f32(0).unwrap().to_vec();
    chunk_states.swap();

    let single =
        NativeProgram::from_graph_in(mixer(1), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut step = single.prepare_native_in(Global).unwrap();
    let mut states = States::new();
    let mut stepped = Vec::new();
    for t in 0..tokens {
        let step_inputs = [
            bytes(&qkv[t * QKV..(t + 1) * QKV]),
            bytes(&a[t * VALUE_HEADS..(t + 1) * VALUE_HEADS]),
            bytes(&write[t * VALUE_HEADS..(t + 1) * VALUE_HEADS]),
        ];
        step.run_with_resources(&step_inputs, &states.bindings())
            .unwrap();
        stepped.extend_from_slice(step.output_f32(0).unwrap());
        states.swap();
    }
    assert_eq!(stepped, chunk_output);
    assert_eq!(states.conv[0], chunk_states.conv[0]);
    assert_eq!(states.rule[0], chunk_states.rule[0]);
    assert!(stepped.iter().all(|v| v.is_finite()));
    assert_eq!(stepped.len(), tokens * OUTPUT);
}

#[test]
fn a_failed_step_leaves_its_source_and_a_retry_repeats_it() {
    let program =
        NativeProgram::from_graph_in(mixer(1), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut step = program.prepare_native_in(Global).unwrap();
    let (qkv, a, write) = inputs(1);
    let step_inputs = [bytes(&qkv), bytes(&a), bytes(&write)];
    let mut states = States::new();
    let before = states.rule[0].clone();
    let mut short = vec![0.0; RULE_STATE - 1];
    {
        let [conv_source, conv_target, rule_source, _] = states.bindings();
        assert!(matches!(
            step.run_with_resources(
                &step_inputs,
                &[
                    conv_source,
                    conv_target,
                    rule_source,
                    NativeResourceBinding::state_target(
                        key(RULE[1]),
                        StateTargetRef::new(&mut short)
                    ),
                ],
            ),
            Err(Error::ResourceLength {
                expected: RULE_STATE,
                ..
            })
        ));
    }
    assert!(step.output_f32(0).is_none());
    assert_eq!(states.rule[0], before);
    step.run_with_resources(&step_inputs, &states.bindings())
        .unwrap();
    let first = (step.output_f32(0).unwrap().to_vec(), states.rule[1].clone());
    step.run_with_resources(&step_inputs, &states.bindings())
        .unwrap();
    assert_eq!(step.output_f32(0).unwrap(), first.0);
    assert_eq!(states.rule[1], first.1);
    assert_eq!(states.rule[0], before);
}

/// A delegate that takes only recurrences, computing each at its offer
/// with the CPU kernels: every pure node stays native.
#[derive(Default)]
struct RecurrentLane {
    offered: Vec<axnn::lowering::SemanticOp>,
}

impl ax_exec::PreparedKernelExecutor for RecurrentLane {
    async fn wait(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn begin(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn admission_window(&self) -> usize {
        8
    }

    fn advance(
        &mut self,
        _: usize,
        _: &axnn_cpu::CpuKernelPlan,
        _: &[Option<ax_exec::NativeRef<'_>>; 3],
        _: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        Ok(ax_exec::KernelProgress::Unsupported)
    }

    fn advance_chained(
        &mut self,
        _: usize,
        _: &axnn_cpu::CpuKernelPlan,
        _: &[Option<ax_exec::ChainedInput<'_>>; 3],
        _: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        Ok(ax_exec::KernelProgress::Unsupported)
    }

    fn advance_recurrent(
        &mut self,
        _: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        state: ax_exec::RecurrentState<'_>,
        output: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        let operand = |index: usize| match inputs[index] {
            Some(ax_exec::ChainedInput::Host(ax_exec::NativeRef::F32(values))) => values,
            _ => panic!("pure nodes ran natively, so every operand is on the host"),
        };
        let source = StateSourceRef::new(state.source);
        match plan.semantic {
            axnn::lowering::SemanticOp::CausalConv1d => axnn_cpu::invoke_causal_conv1d(
                plan,
                source,
                state.target,
                operand(0),
                operand(1),
                operand(2),
                output,
                &ax_exec::Serial,
            )?,
            axnn::lowering::SemanticOp::GatedDeltaRule => axnn_cpu::invoke_gated_delta_rule(
                plan,
                source,
                state.target,
                operand(0),
                operand(1),
                operand(2),
                output,
                &ax_exec::Serial,
            )?,
            other => panic!("{other:?} is not this mixer's recurrence"),
        }
        self.offered.push(plan.semantic);
        Ok(ax_exec::KernelProgress::Complete)
    }
}

/// The engine offers the gated delta rule to a delegate with its source and
/// target, the seam a device lane takes it through, and the delegate's
/// results match the native run bit for bit.
#[test]
fn a_delegate_is_offered_the_rule_with_its_state() {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let tokens = 2;
    let (qkv, a, write) = inputs(tokens);
    let program =
        NativeProgram::from_graph_in(mixer(2), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut native = program.prepare_native_in(Global).unwrap();
    let mut expected = States::new();
    native
        .run_with_resources(
            &[bytes(&qkv), bytes(&a), bytes(&write)],
            &expected.bindings(),
        )
        .unwrap();
    let expected_output = native.output_f32(0).unwrap().to_vec();

    let mut arena = program.prepare_native_in(Global).unwrap();
    let mut lane = RecurrentLane::default();
    let mut states = States::new();
    {
        let refs = [
            ax_exec::NativeRef::F32(&qkv),
            ax_exec::NativeRef::F32(&a),
            ax_exec::NativeRef::F32(&write),
        ];
        let bindings = states.bindings();
        let mut run = arena
            .begin_with_executor(
                &refs,
                &bindings,
                &ax_exec::Serial,
                &mut lane,
                axnn::lowering::KernelAccuracy::Faithful,
            )
            .unwrap();
        loop {
            match run.advance(64).unwrap() {
                ax_exec::PreparedProgress::Complete => break,
                ax_exec::PreparedProgress::Progress { .. } => {}
                ax_exec::PreparedProgress::Pending => {
                    let mut wait = pin!(run.wait());
                    assert!(matches!(
                        wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Ready(Ok(()))
                    ));
                }
            }
        }
    }
    assert_eq!(
        lane.offered,
        [
            axnn::lowering::SemanticOp::CausalConv1d,
            axnn::lowering::SemanticOp::GatedDeltaRule
        ]
    );
    assert_eq!(arena.output_f32(0).unwrap(), expected_output);
    assert_eq!(states.conv[1], expected.conv[1]);
    assert_eq!(states.rule[1], expected.rule[1]);
}
