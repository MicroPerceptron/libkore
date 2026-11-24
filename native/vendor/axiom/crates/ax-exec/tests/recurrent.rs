#![feature(allocator_api)]
use ax::recurrent::{CausalConv1dParams, ResourceKey, SsmScanParams, StateBinding};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeResourceBinding};
use axnn_cpu::{StateSourceRef, StateTargetRef};
use std::alloc::Global;

const HEADS: usize = 4;
const HEAD_DIM: usize = 3;
const GROUPS: usize = 2;
const STATE_SIZE: usize = 5;
const CONV_DIM: usize = HEADS * HEAD_DIM + 2 * GROUPS * STATE_SIZE;
const KERNEL: usize = 4;
const CONV_STATE: usize = CONV_DIM * (KERNEL - 1);
const SCAN_STATE: usize = HEADS * HEAD_DIM * STATE_SIZE;

fn key(binding: u32) -> ResourceKey {
    ResourceKey { group: 3, binding }
}

const CONV: [u32; 2] = [0, 1];
const SCAN: [u32; 2] = [2, 3];

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

fn constant(b: &mut GraphBuilder<Global>, dims: &[u32], seed: usize) -> ax::TensorId {
    let count = dims.iter().product::<u32>() as usize;
    let bytes: Vec<u8> = values(count, seed)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    b.tensor(DType::F32, TensorFlags::CONSTANT, dims, &bytes)
        .unwrap()
}

/// The recurrent core of a Mamba-2 mixer: causal convolution over the
/// `x|B|C` projection, SiLU, then the state-space scan.
fn mixer(tokens: u32) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let xbc = b
        .tensor(
            DType::F32,
            TensorFlags::INPUT,
            &[tokens, CONV_DIM as u32],
            &[],
        )
        .unwrap();
    let dt = b
        .tensor(DType::F32, TensorFlags::INPUT, &[tokens, HEADS as u32], &[])
        .unwrap();
    let weight = constant(&mut b, &[CONV_DIM as u32, KERNEL as u32], 1);
    let bias = constant(&mut b, &[CONV_DIM as u32], 2);
    let head_params = constant(&mut b, &[3, HEADS as u32], 3);
    let conv = CausalConv1dParams::new(binding(CONV), CONV_DIM as u32, KERNEL as u32).unwrap();
    let convolved = b
        .causal_conv1d(xbc, weight, bias, tokens, conv, TensorFlags::default())
        .unwrap();
    let activated = b
        .tensor(
            DType::F32,
            TensorFlags::default(),
            &[tokens, CONV_DIM as u32],
            &[],
        )
        .unwrap();
    b.node(
        OpCode::SILU,
        NodeFlags::default(),
        &[convolved],
        &[activated],
        &[],
    )
    .unwrap();
    let scan = SsmScanParams::new(
        binding(SCAN),
        HEADS as u32,
        HEAD_DIM as u32,
        GROUPS as u32,
        STATE_SIZE as u32,
        0.0,
        f32::INFINITY,
    )
    .unwrap();
    b.ssm_scan(
        activated,
        dt,
        head_params,
        tokens,
        scan,
        TensorFlags::OUTPUT,
    )
    .unwrap();
    b.finish().unwrap()
}

fn bytes(values: &[f32]) -> &[u8] {
    // SAFETY: initialized, padding-free f32 words, borrowed immutably.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

/// Two alternating buffers per recurrence, as a decode loop would keep them.
struct States {
    conv: [Vec<f32>; 2],
    scan: [Vec<f32>; 2],
}

impl States {
    fn new() -> Self {
        Self {
            conv: [values(CONV_STATE, 4), vec![0.0; CONV_STATE]],
            scan: [values(SCAN_STATE, 5), vec![0.0; SCAN_STATE]],
        }
    }

    /// Bind buffer 0 as each source and buffer 1 as each target.
    fn bindings(&mut self) -> [NativeResourceBinding<'_>; 4] {
        self.tree_bindings(None)
    }

    fn tree_bindings<'a>(
        &'a mut self,
        tree: Option<axnn::lowering::TokenTree<'a>>,
    ) -> [NativeResourceBinding<'a>; 4] {
        let [conv_source, conv_target] = &mut self.conv;
        let [scan_source, scan_target] = &mut self.scan;
        [
            NativeResourceBinding::state_source(
                key(CONV[0]),
                tree.map_or(StateSourceRef::new(conv_source), |tree| {
                    StateSourceRef::new(conv_source).with_tree(tree)
                }),
            ),
            NativeResourceBinding::state_target(key(CONV[1]), StateTargetRef::new(conv_target)),
            NativeResourceBinding::state_source(
                key(SCAN[0]),
                tree.map_or(StateSourceRef::new(scan_source), |tree| {
                    StateSourceRef::new(scan_source).with_tree(tree)
                }),
            ),
            NativeResourceBinding::state_target(key(SCAN[1]), StateTargetRef::new(scan_target)),
        ]
    }

    /// Commit a completed step: the target becomes the next source.
    fn swap(&mut self) {
        self.conv.swap(0, 1);
        self.scan.swap(0, 1);
    }
}

#[test]
fn stepping_with_alternating_buffers_equals_one_chunk() {
    let tokens = 3;
    let xbc = values(tokens * CONV_DIM, 6);
    let dt = values(tokens * HEADS, 7);

    let chunked =
        NativeProgram::from_graph_in(mixer(3), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut chunk = chunked.prepare_native_in(Global).unwrap();
    let mut chunk_states = States::new();
    chunk
        .run_with_resources(&[bytes(&xbc), bytes(&dt)], &chunk_states.bindings())
        .unwrap();
    let chunk_output = chunk.output_f32(0).unwrap().to_vec();
    chunk_states.swap();

    let single =
        NativeProgram::from_graph_in(mixer(1), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut step = single.prepare_native_in(Global).unwrap();
    let mut states = States::new();
    let mut stepped = Vec::new();
    for t in 0..tokens {
        let inputs = [
            bytes(&xbc[t * CONV_DIM..(t + 1) * CONV_DIM]),
            bytes(&dt[t * HEADS..(t + 1) * HEADS]),
        ];
        step.run_with_resources(&inputs, &states.bindings())
            .unwrap();
        stepped.extend_from_slice(step.output_f32(0).unwrap());
        states.swap();
    }
    assert_eq!(stepped, chunk_output);
    assert_eq!(states.conv[0], chunk_states.conv[0]);
    assert_eq!(states.scan[0], chunk_states.scan[0]);
    assert!(stepped.iter().all(|v| v.is_finite()));
}

#[test]
fn a_failed_step_leaves_its_source_and_can_be_retried() {
    let program =
        NativeProgram::from_graph_in(mixer(1), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut step = program.prepare_native_in(Global).unwrap();
    let xbc = values(CONV_DIM, 8);
    let dt = values(HEADS, 9);
    let inputs = [bytes(&xbc), bytes(&dt)];
    let mut states = States::new();
    let (conv_before, scan_before) = (states.conv[0].clone(), states.scan[0].clone());

    // A missing target, a source bound as a target, and a short buffer all
    // fail before any node runs.
    let [conv_source, conv_target, scan_source, scan_target] = states.bindings();
    assert!(matches!(
        step.run_with_resources(&inputs, &[conv_source, conv_target, scan_source]),
        Err(Error::MissingResource(k)) if k == key(SCAN[1])
    ));
    let mut elsewhere = vec![0.0; CONV_STATE];
    let wrong_kind =
        NativeResourceBinding::state_target(key(CONV[0]), StateTargetRef::new(&mut elsewhere));
    assert!(matches!(
        step.run_with_resources(
            &inputs,
            &[wrong_kind, conv_target, scan_source, scan_target]
        ),
        Err(Error::ResourceKind { .. })
    ));
    let mut short = vec![0.0; SCAN_STATE - 1];
    assert!(matches!(
        step.run_with_resources(
            &inputs,
            &[
                conv_source,
                conv_target,
                scan_source,
                NativeResourceBinding::state_target(key(SCAN[1]), StateTargetRef::new(&mut short)),
            ],
        ),
        Err(Error::ResourceLength {
            expected: SCAN_STATE,
            ..
        })
    ));
    assert!(step.output_f32(0).is_none());
    assert_eq!(states.conv[0], conv_before);
    assert_eq!(states.scan[0], scan_before);

    // Running the same step twice from the same source gives the same
    // successor: nothing was committed.
    step.run_with_resources(&inputs, &states.bindings())
        .unwrap();
    let first = (step.output_f32(0).unwrap().to_vec(), states.scan[1].clone());
    step.run_with_resources(&inputs, &states.bindings())
        .unwrap();
    assert_eq!(step.output_f32(0).unwrap(), first.0);
    assert_eq!(states.scan[1], first.1);
    assert_eq!(states.scan[0], scan_before);
}

/// A windowed executor that takes every instruction: pure ones through
/// `advance_chained`, the recurrences with their state, computing each with
/// the CPU kernels at its offer and publishing it only once the engine
/// waits, as a device lane does.
#[derive(Default)]
struct Lane {
    /// Admitted and not yet completed, in admission order.
    queue: Vec<usize>,
    results: std::collections::BTreeMap<usize, Vec<f32>>,
    /// The successor states each admitted recurrence produced.
    states: std::collections::BTreeMap<usize, Vec<f32>>,
    /// Admissions a `wait` has made visible.
    flushed: usize,
    waits: usize,
    /// Recurrences offered with an operand a pending admission produces.
    chained_recurrent: usize,
    /// Decline the recurrences (the default), so the engine runs them
    /// natively around the window.
    decline_recurrent: bool,
    /// The fixture's storage owner registers these stable targets before the
    /// graph borrows them. A direct lane may write only these ranges.
    direct: bool,
    retained: [Option<(*mut f32, usize)>; 2],
    destinations: std::collections::BTreeMap<usize, (*mut f32, usize)>,
    tree_parents: Option<Vec<u32>>,
    tree_offers: usize,
}

impl Lane {
    fn operand(&self, input: ax_exec::ChainedInput<'_>) -> Vec<f32> {
        match input {
            ax_exec::ChainedInput::Host(ax_exec::NativeRef::F32(values)) => values.to_vec(),
            ax_exec::ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => self.results[&instruction][offset..offset + elements].to_vec(),
            ax_exec::ChainedInput::Host(_) => panic!("F32 operands only"),
        }
    }

    /// A retry of an admission: Pending until a wait, then its results.
    fn retry(
        &mut self,
        instruction: usize,
        output: &mut [f32],
        target: Option<StateTargetRef<'_>>,
    ) -> Option<ax_exec::KernelProgress> {
        let position = self
            .queue
            .iter()
            .position(|&queued| queued == instruction)?;
        assert_eq!(position, 0, "admissions retire in admission order");
        if self.flushed == 0 {
            return Some(ax_exec::KernelProgress::Pending);
        }
        output.copy_from_slice(&self.results[&instruction]);
        if let Some(target) = target {
            if self.direct {
                assert_eq!(
                    self.destinations[&instruction],
                    (target.as_mut_ptr(), target.len())
                );
            } else {
                // SAFETY: the staged lane has completed; this fixture has
                // exclusive access and retains no CPU reference afterwards.
                unsafe { target.values() }.copy_from_slice(&self.states[&instruction]);
            }
        }
        self.queue.remove(0);
        self.flushed -= 1;
        Some(ax_exec::KernelProgress::Complete)
    }
}

impl ax_exec::PreparedKernelExecutor for Lane {
    async fn wait(&mut self) -> Result<(), Error> {
        self.waits += 1;
        if self.direct {
            for (&instruction, &(target, len)) in &self.destinations {
                let values = &self.states[&instruction];
                assert_eq!(values.len(), len);
                // SAFETY: run_lane owns the registered targets for the entire
                // invocation and does not access them before completion. This
                // models a device write between offers, without a CPU slice.
                unsafe {
                    core::ptr::copy_nonoverlapping(values.as_ptr(), target, len);
                }
            }
        }
        self.flushed = self.queue.len();
        Ok(())
    }

    fn begin(&mut self) -> Result<(), Error> {
        assert!(self.queue.is_empty());
        self.results.clear();
        self.states.clear();
        self.destinations.clear();
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
        panic!("a windowed executor is offered through advance_chained")
    }

    fn advance_chained(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        output: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        if let Some(progress) = self.retry(instruction, output, None) {
            return Ok(progress);
        }
        let operands: Vec<Vec<f32>> = inputs.iter().flatten().map(|&i| self.operand(i)).collect();
        let result = match plan.semantic {
            axnn::lowering::SemanticOp::Unary(axnn::UnaryOp::Silu) => operands[0]
                .iter()
                .map(|&x| x / (1.0 + (-x).exp()))
                .collect(),
            other => panic!("the mixer offers no {other:?} as a pure kernel"),
        };
        self.results.insert(instruction, result);
        self.queue.push(instruction);
        Ok(ax_exec::KernelProgress::Pending)
    }

    fn advance_recurrent(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        state: ax_exec::RecurrentState<'_>,
        output: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        assert_eq!(
            state.tree.map(|tree| tree.parents()),
            self.tree_parents.as_deref()
        );
        if state.tree.is_some() {
            self.tree_offers += 1;
        }
        let source = state
            .tree
            .map_or(StateSourceRef::new(state.source), |tree| {
                StateSourceRef::new(state.source).with_tree(tree)
            });
        if let Some(progress) = self.retry(instruction, output, Some(state.target)) {
            return Ok(progress);
        }
        if self.decline_recurrent {
            return Ok(ax_exec::KernelProgress::Unsupported);
        }
        self.chained_recurrent += inputs
            .iter()
            .flatten()
            .filter(|input| matches!(input, ax_exec::ChainedInput::Pending { .. }))
            .count();
        let operands: Vec<Vec<f32>> = inputs.iter().flatten().map(|&i| self.operand(i)).collect();
        let mut result = vec![0.0; output.len()];
        let mut successor = vec![0.0; state.target.len()];
        match plan.semantic {
            axnn::lowering::SemanticOp::CausalConv1d => axnn_cpu::invoke_causal_conv1d(
                plan,
                source,
                StateTargetRef::new(&mut successor),
                &operands[0],
                &operands[1],
                &operands[2],
                &mut result,
                &ax_exec::Serial,
            )
            .unwrap(),
            axnn::lowering::SemanticOp::SsmScan => axnn_cpu::invoke_ssm_scan(
                plan,
                source,
                StateTargetRef::new(&mut successor),
                &operands[0],
                &operands[1],
                &operands[2],
                &mut result,
                &ax_exec::Serial,
            )
            .unwrap(),
            other => panic!("{other:?} is not recurrent"),
        }
        if self.direct {
            let destination = (state.target.as_mut_ptr(), state.target.len());
            assert!(self.retained.contains(&Some(destination)), "unowned target");
            self.destinations.insert(instruction, destination);
        }
        self.results.insert(instruction, result);
        self.states.insert(instruction, successor);
        self.queue.push(instruction);
        Ok(ax_exec::KernelProgress::Pending)
    }
}

/// The mixer through a windowed executor: the outputs, and the successor
/// states the run left in the targets.
fn run_lane(lane: &mut Lane, tokens: usize, xbc: &[f32], dt: &[f32]) -> (Vec<f32>, States) {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let program = NativeProgram::from_graph_in(
        mixer(tokens as u32),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap();
    let mut arena = program.prepare_native_in(Global).unwrap();
    let parents = lane.tree_parents.clone();
    let tree = parents
        .as_deref()
        .map(|parents| axnn::lowering::TokenTree::new(parents).unwrap());
    let mut states = States::new();
    if tree.is_some() {
        states.conv[1].resize(tokens * CONV_STATE, 0.0);
        states.scan[1].resize(tokens * SCAN_STATE, 0.0);
    }
    lane.retained = [
        Some((states.conv[1].as_mut_ptr(), states.conv[1].len())),
        Some((states.scan[1].as_mut_ptr(), states.scan[1].len())),
    ];
    let inputs = [ax_exec::NativeRef::F32(xbc), ax_exec::NativeRef::F32(dt)];
    {
        let bindings = states.tree_bindings(tree);
        let mut run = arena
            .begin_with_executor(
                &inputs,
                &bindings,
                &ax_exec::Serial,
                lane,
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
    (arena.output_f32(0).unwrap().to_vec(), states)
}

/// An admission window carries the convolution and the scan with their
/// states: the scan reads the SiLU's pending output on the device, one wait
/// covers the whole mixer, and the targets hold the successor states.
#[test]
fn an_admission_window_carries_the_recurrences_with_their_state() {
    let tokens = 3;
    let xbc = values(tokens * CONV_DIM, 6);
    let dt = values(tokens * HEADS, 7);
    let native =
        NativeProgram::from_graph_in(mixer(3), AxCpuKernelPolicy::default(), true, Global).unwrap();
    let mut native = native.prepare_native_in(Global).unwrap();
    let mut expected = States::new();
    native
        .run_with_resources(&[bytes(&xbc), bytes(&dt)], &expected.bindings())
        .unwrap();
    let expected_output = native.output_f32(0).unwrap().to_vec();

    // The lane's SiLU is not the native kernel's to the last ulp, and the
    // scan carries that ulp into its states.
    let close = |actual: &[f32], expected: &[f32]| {
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, e)| (a - e).abs() <= 1e-6 * (1.0 + e.abs()))
    };

    let mut lane = Lane::default();
    let (output, states) = run_lane(&mut lane, tokens, &xbc, &dt);
    assert!(close(&output, &expected_output));
    assert_eq!(
        states.conv[1], expected.conv[1],
        "the convolution ran the native kernel"
    );
    assert!(close(&states.scan[1], &expected.scan[1]));
    assert_eq!(lane.waits, 1, "the mixer waited more than once");
    assert_eq!(
        lane.chained_recurrent, 1,
        "the scan read the SiLU as a pending operand"
    );
    assert!(lane.queue.is_empty());

    // Independently retained targets can be written between offers; retries
    // carry metadata only and publish no CPU slice before the wait completes.
    let mut direct = Lane {
        direct: true,
        ..Lane::default()
    };
    let (output, states) = run_lane(&mut direct, tokens, &xbc, &dt);
    assert!(close(&output, &expected_output));
    assert_eq!(states.conv[1], expected.conv[1]);
    assert!(close(&states.scan[1], &expected.scan[1]));
    assert_eq!(direct.waits, 1);
    assert_eq!(direct.destinations.len(), 2);
    assert!(direct.queue.is_empty());

    // Declined, the recurrences run natively around the window, to the
    // same results.
    let mut declining = Lane {
        decline_recurrent: true,
        ..Lane::default()
    };
    let (output, states) = run_lane(&mut declining, tokens, &xbc, &dt);
    assert!(close(&output, &expected_output));
    assert_eq!(states.conv[1], expected.conv[1]);
    assert!(close(&states.scan[1], &expected.scan[1]));
    assert_eq!(declining.chained_recurrent, 0);
    assert!(declining.queue.is_empty());
}

#[test]
fn tree_topology_survives_admission_window_retries_and_cpu_declines() {
    use axnn::lowering::TokenTree;
    let tokens = TokenTree::MAX_NODES;
    let parents: Vec<u32> = (0..tokens)
        .map(|i| {
            if i == 0 {
                TokenTree::ROOT
            } else {
                ((i - 1) / 2) as u32
            }
        })
        .collect();
    let tree = TokenTree::new(&parents).unwrap();
    let xbc = values(tokens * CONV_DIM, 6);
    let dt = values(tokens * HEADS, 7);
    let native = NativeProgram::from_graph_in(
        mixer(tokens as u32),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap();
    let mut native = native.prepare_native_in(Global).unwrap();
    let mut expected = States::new();
    expected.conv[1].resize(tokens * CONV_STATE, 0.0);
    expected.scan[1].resize(tokens * SCAN_STATE, 0.0);
    native
        .run_with_resources(
            &[bytes(&xbc), bytes(&dt)],
            &expected.tree_bindings(Some(tree)),
        )
        .unwrap();
    let close = |actual: &[f32], expected: &[f32]| {
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, e)| (a - e).abs() <= 1e-6 * (1.0 + e.abs()))
    };
    for (direct, decline_recurrent) in [(false, false), (true, false), (false, true)] {
        let mut lane = Lane {
            direct,
            decline_recurrent,
            tree_parents: Some(parents.clone()),
            ..Default::default()
        };
        let (output, states) = run_lane(&mut lane, tokens, &xbc, &dt);
        assert!(close(&output, native.output_f32(0).unwrap()));
        assert_eq!(states.conv[1], expected.conv[1]);
        assert!(close(&states.scan[1], &expected.scan[1]));
        assert!(lane.tree_offers >= 2);
        if !decline_recurrent {
            assert!(
                lane.tree_offers > 2,
                "retirement retries must preserve the tree view"
            );
            assert_eq!(lane.waits, 1);
        }
        assert!(lane.queue.is_empty());
    }
}
