#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{
    AxCpuKernelPolicy, ChainedInput, Error, FusedRun, KernelProgress, NativeProgram, NativeRef,
    PreparedKernelExecutor, PreparedNative, PreparedProgress, Serial,
};
use axnn::lowering::{KernelAccuracy, SemanticOp};
use axnn_cpu::CpuKernelPlan;
use std::alloc::Global;

fn arena() -> PreparedNative<Global> {
    let mut graph = GraphBuilder::new_in(Global);
    let x = graph
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let y = graph
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let z = graph
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    graph
        .node(OpCode::NEGATE, NodeFlags::default(), &[x], &[y], &[])
        .unwrap();
    graph
        .node(OpCode::MUL, NodeFlags::default(), &[x, y], &[z], &[])
        .unwrap();
    let graph = graph.finish().unwrap();
    NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        false,
        Global,
    )
    .unwrap()
    .prepare_borrowing_inputs_in(Global)
    .unwrap()
}

#[derive(Default)]
struct Deferred {
    calls: usize,
    original: Option<usize>,
    refuse_after_pending: bool,
    fail_after_pending: bool,
    fail_wait: bool,
}

impl PreparedKernelExecutor for Deferred {
    async fn wait(&mut self) -> Result<(), Error> {
        if self.fail_wait {
            Err(Error::InvocationFinished)
        } else {
            Ok(())
        }
    }

    fn begin(&mut self) -> Result<(), Error> {
        if self.calls != 0 && self.calls != 3 {
            return Err(Error::InvocationFinished);
        }
        self.calls = 0;
        self.original = None;
        Ok(())
    }
    fn advance(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<NativeRef<'_>>; 3],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        assert_eq!(minimum, KernelAccuracy::Faithful);
        if instruction != 0 {
            assert_eq!(self.calls, 3, "successor ran before producer completed");
            return Ok(KernelProgress::Unsupported);
        }
        assert_eq!(plan.semantic, SemanticOp::Unary(axnn::UnaryOp::Neg));
        let Some(NativeRef::F32(input)) = inputs[0] else {
            panic!("typed F32 ingress")
        };
        assert!(inputs[1..].iter().all(Option::is_none));
        let address = input.as_ptr() as usize;
        assert_eq!(
            *self.original.get_or_insert(address),
            address,
            "ingress was replaced"
        );
        self.calls += 1;
        if self.calls == 1 {
            return Ok(KernelProgress::Pending);
        }
        if self.refuse_after_pending {
            return Ok(KernelProgress::Unsupported);
        }
        if self.fail_after_pending {
            return Err(Error::InvocationFinished);
        }
        if self.calls == 2 {
            return Ok(KernelProgress::Pending);
        }
        for (dst, src) in output.iter_mut().zip(input) {
            *dst = -*src;
        }
        Ok(KernelProgress::Complete)
    }
}

#[test]
fn suspension_preserves_borrowed_ingress_and_native_successor_order() {
    let mut arena = arena();
    let input = [1.0, -2.0, 3.0, -4.0];
    let inputs = [NativeRef::F32(&input)];
    let mut executor = Deferred::default();
    {
        let mut run = arena
            .begin_with_executor(
                &inputs,
                &[],
                &Serial,
                &mut executor,
                KernelAccuracy::Faithful,
            )
            .unwrap();
        assert!(matches!(run.advance(0), Err(Error::InvalidStepBudget)));
        assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
        assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
        assert_eq!(
            run.advance(1).unwrap(),
            PreparedProgress::Progress { instructions: 1 }
        );
        assert_eq!(run.advance(1).unwrap(), PreparedProgress::Complete);
        assert_eq!(run.advance(1).unwrap(), PreparedProgress::Complete);
    }
    assert_eq!(executor.original, Some(input.as_ptr() as usize));
    assert_eq!(arena.output_f32(0).unwrap(), &[-1.0, -4.0, -9.0, -16.0]);
}

#[test]
fn uncertain_admission_cannot_fall_back_and_rerun_the_kernel() {
    let mut arena = arena();
    arena.set_profiler(|| 0);
    let input = [1.0; 4];
    let inputs = [NativeRef::F32(&input)];
    let mut executor = Deferred {
        refuse_after_pending: true,
        ..Default::default()
    };
    {
        let mut run = arena
            .begin_with_executor(
                &inputs,
                &[],
                &Serial,
                &mut executor,
                KernelAccuracy::Faithful,
            )
            .unwrap();
        assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
        assert!(run.advance(2).is_err());
        assert!(matches!(run.advance(2), Err(Error::InvocationFinished)));
    }
    assert!(arena.output_f32(0).is_none());
    assert_eq!(arena.profile().fallback.runs, [0; ax_exec::OP_CLASSES]);
    assert_eq!(arena.profile().cpu.runs, [0; ax_exec::OP_CLASSES]);
}

fn profile_run(
    arena: &mut PreparedNative<Global>,
    executor: &mut impl PreparedKernelExecutor,
    input: &[f32],
) {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let inputs = [NativeRef::F32(input)];
    let mut run = arena
        .begin_with_executor(&inputs, &[], &Serial, executor, KernelAccuracy::Faithful)
        .unwrap();
    for _ in 0..128 {
        match run.advance(64).unwrap() {
            PreparedProgress::Complete => return,
            PreparedProgress::Progress { .. } => {}
            PreparedProgress::Pending => {
                let mut wait = pin!(run.wait());
                assert!(matches!(
                    wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                    Poll::Ready(Ok(()))
                ));
            }
        }
    }
    panic!("fixture did not complete");
}

#[test]
fn cpu_placement_counts_execution_not_offers_or_device_waits() {
    let input = [1.0, -2.0, 3.0, -4.0];
    let mut direct = chain_arena();
    direct.set_profiler(|| 0);
    direct
        .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
        .unwrap();
    assert_eq!(direct.profile().cpu.runs.iter().sum::<u64>(), 6);
    assert_eq!(direct.profile().fallback.runs.iter().sum::<u64>(), 0);

    let mut deferred = arena();
    deferred.set_profiler(|| 0);
    profile_run(&mut deferred, &mut Deferred::default(), &input);
    assert_eq!(deferred.profile().cpu.runs.iter().sum::<u64>(), 0);
    assert_eq!(deferred.profile().fallback.runs.iter().sum::<u64>(), 1);

    for check in [false, true] {
        let mut arena = chain_arena();
        arena.set_profiler(|| 0);
        arena.set_delegate_check(check);
        let mut executor = Windowed {
            window: 4,
            decline: vec![2],
            ..Default::default()
        };
        profile_run(&mut arena, &mut executor, &input);
        assert_eq!(arena.output_f32(0).unwrap(), native_chain(&input));
        let profile = arena.profile();
        assert_eq!(profile.cpu.runs.iter().sum::<u64>(), 0);
        assert_eq!(
            profile.fallback.runs.iter().sum::<u64>(),
            1,
            "the declined offer behind pending work was counted twice"
        );
        assert_eq!(
            profile.reference.runs.iter().sum::<u64>(),
            if check { 5 } else { 0 }
        );
    }
}

#[test]
fn unfused_device_retry_is_not_cpu_fallback_and_disabled_profile_is_empty() {
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 1.0];
    for enabled in [false, true] {
        let mut arena = fusable_arena(false);
        if enabled {
            arena.set_profiler(|| 0);
        }
        let mut executor = Windowed {
            window: 8,
            fuse: false,
            ..Default::default()
        };
        profile_run(&mut arena, &mut executor, &input);
        assert!(executor.fused.is_empty());
        let profile = arena.profile();
        assert_eq!(profile.cpu.runs.iter().sum::<u64>(), 0);
        assert_eq!(profile.fallback.runs.iter().sum::<u64>(), 0);
        assert_eq!(profile.reference.runs.iter().sum::<u64>(), 0);
        assert_eq!(
            profile.runs.iter().sum::<u64>(),
            if enabled { 6 } else { 0 }
        );
    }
}

#[test]
fn failed_or_abandoned_runs_restore_slots_without_publishing_outputs() {
    for fail in [false, true] {
        let mut arena = arena();
        let input = [2.0; 4];
        let inputs = [NativeRef::F32(&input)];
        let mut executor = Deferred {
            fail_after_pending: fail,
            ..Default::default()
        };
        {
            let mut run = arena
                .begin_with_executor(
                    &inputs,
                    &[],
                    &Serial,
                    &mut executor,
                    KernelAccuracy::Faithful,
                )
                .unwrap();
            assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
            if fail {
                assert!(run.advance(2).is_err());
            }
        }
        assert!(arena.output_f32(0).is_none());
        assert!(
            arena
                .begin_with_executor(
                    &inputs,
                    &[],
                    &Serial,
                    &mut executor,
                    KernelAccuracy::Faithful
                )
                .is_err(),
            "an abandoned event was reused"
        );
        arena.run_native_on(&inputs, &[], &Serial).unwrap();
        assert_eq!(arena.output_f32(0).unwrap(), &[-4.0; 4]);
    }
}

#[test]
fn malformed_ingress_is_refused_before_executor_admission() {
    let mut arena = arena();
    let mut executor = Deferred::default();
    let inputs = [NativeRef::F32(&[0.0; 3])];
    assert!(
        arena
            .begin_with_executor(
                &inputs,
                &[],
                &Serial,
                &mut executor,
                KernelAccuracy::Faithful
            )
            .is_err()
    );
    assert_eq!(executor.calls, 0);
}

#[test]
fn failed_runtime_wait_seals_the_graph_without_publishing_outputs() {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let mut arena = arena();
    let inputs = [NativeRef::F32(&[1.0; 4])];
    let mut executor = Deferred {
        fail_wait: true,
        ..Default::default()
    };
    {
        let mut run = arena
            .begin_with_executor(
                &inputs,
                &[],
                &Serial,
                &mut executor,
                KernelAccuracy::Faithful,
            )
            .unwrap();
        assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
        {
            let mut wait = pin!(run.wait());
            assert!(matches!(
                wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(Error::InvocationFinished))
            ));
        }
        assert!(matches!(run.advance(2), Err(Error::InvocationFinished)));
    }
    assert!(arena.output_f32(0).is_none());
    assert_eq!(executor.calls, 1, "failed wait reran the instruction");
}

/// A chain long enough that the compiler reuses slots: every value but the
/// output dies within two instructions of its producer.
fn chain_arena() -> PreparedNative<Global> {
    let mut graph = GraphBuilder::new_in(Global);
    let tensor = |graph: &mut GraphBuilder<Global>, flags| {
        graph.tensor(DType::F32, flags, &[4], &[]).unwrap()
    };
    let x = tensor(&mut graph, TensorFlags::INPUT);
    let a = tensor(&mut graph, TensorFlags::default());
    let b = tensor(&mut graph, TensorFlags::default());
    let c = tensor(&mut graph, TensorFlags::default());
    let d = tensor(&mut graph, TensorFlags::default());
    let e = tensor(&mut graph, TensorFlags::default());
    let f = tensor(&mut graph, TensorFlags::OUTPUT);
    let flags = NodeFlags::default();
    graph.node(OpCode::NEGATE, flags, &[x], &[a], &[]).unwrap();
    graph.node(OpCode::MUL, flags, &[x, a], &[b], &[]).unwrap();
    graph.node(OpCode::ADD, flags, &[b, a], &[c], &[]).unwrap();
    graph.node(OpCode::NEGATE, flags, &[c], &[d], &[]).unwrap();
    graph.node(OpCode::MUL, flags, &[d, b], &[e], &[]).unwrap();
    graph.node(OpCode::ADD, flags, &[e, x], &[f], &[]).unwrap();
    let graph = graph.finish().unwrap();
    NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        false,
        Global,
    )
    .unwrap()
    .prepare_borrowing_inputs_in(Global)
    .unwrap()
}

/// A device double for the admission window: it computes each admission when
/// offered (from host operands, or from its own results for pending ones),
/// publishes only after a `wait`, and checks the engine's ordering promises.
#[derive(Default)]
struct Windowed {
    window: usize,
    /// Instructions to decline (they run natively).
    decline: Vec<usize>,
    /// Results by instruction, as the device holds them.
    results: std::collections::BTreeMap<usize, Vec<f32>>,
    /// Admitted and not yet completed, in admission order.
    queue: Vec<usize>,
    /// Admissions a `wait` has made visible.
    flushed: usize,
    waits: usize,
    chained: usize,
    /// Take fused runs (otherwise decline them), and the runs taken.
    fuse: bool,
    fused: Vec<FusedRun>,
    /// Instructions whose device result is wrong (every value off by 10).
    corrupt: Vec<usize>,
    discard: bool,
    discarded: Vec<usize>,
}

impl Windowed {
    /// `x [m, k]` times the transposed `w [n, k]`, summed in order.
    fn project(plan: &CpuKernelPlan, operands: &[Vec<f32>]) -> Vec<f32> {
        let (m, k) = (
            plan.input(0).unwrap().shape.dim(0),
            plan.input(0).unwrap().shape.dim(1),
        );
        let n = plan.input(1).unwrap().shape.dim(0);
        let (x, w) = (&operands[0], &operands[1]);
        (0..m * n)
            .map(|at| {
                let (row, column) = (at / n, at % n);
                (0..k).fold(0.0, |sum, j| sum + x[row * k + j] * w[column * k + j])
            })
            .collect()
    }

    fn operand(&self, input: ChainedInput<'_>) -> Vec<f32> {
        match input {
            ChainedInput::Host(NativeRef::F32(values)) => values.to_vec(),
            ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => {
                assert!(
                    self.queue.contains(&instruction),
                    "a pending operand names an admission that is not pending"
                );
                self.results[&instruction][offset..offset + elements].to_vec()
            }
            ChainedInput::Host(_) => panic!("F32 operands only"),
        }
    }
}

impl PreparedKernelExecutor for Windowed {
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
        self.window
    }

    fn retire_unobserved(&mut self, instruction: usize) -> Result<KernelProgress, Error> {
        if !self.discard {
            return Ok(KernelProgress::Unsupported);
        }
        assert_eq!(self.queue.first(), Some(&instruction));
        if self.flushed == 0 {
            return Ok(KernelProgress::Pending);
        }
        self.queue.remove(0);
        self.flushed -= 1;
        self.discarded.push(instruction);
        Ok(KernelProgress::Complete)
    }

    fn advance(
        &mut self,
        _: usize,
        _: &CpuKernelPlan,
        _: &[Option<NativeRef<'_>>; 3],
        _: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        panic!("a windowed executor is offered through advance_chained")
    }

    fn advance_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if let Some(position) = self.queue.iter().position(|&queued| queued == instruction) {
            assert_eq!(position, 0, "admissions retire in admission order");
            if self.flushed == 0 {
                return Ok(KernelProgress::Pending);
            }
            output.copy_from_slice(&self.results[&instruction]);
            self.queue.remove(0);
            self.flushed -= 1;
            return Ok(KernelProgress::Complete);
        }
        if self.decline.contains(&instruction) {
            return Ok(KernelProgress::Unsupported);
        }
        self.chained += inputs
            .iter()
            .flatten()
            .filter(|input| matches!(input, ChainedInput::Pending { .. }))
            .count();
        let operands: Vec<Vec<f32>> = inputs.iter().flatten().map(|&i| self.operand(i)).collect();
        let mut result: Vec<f32> = match plan.semantic {
            SemanticOp::Unary(axnn::UnaryOp::Neg) => operands[0].iter().map(|v| -v).collect(),
            SemanticOp::Unary(axnn::UnaryOp::Relu) => {
                operands[0].iter().map(|v| v.max(0.0)).collect()
            }
            SemanticOp::MatMulTransposed => Self::project(plan, &operands),
            SemanticOp::RmsNorm => {
                let axnn::lowering::KernelParams::Normalization { epsilon, axis } = plan.params()
                else {
                    panic!("normalization parameters");
                };
                let width = plan.shape.dims()[axis as usize..].iter().product();
                rms(&operands[0], &operands[1], width, epsilon.to_f32())
            }
            // Row-major trailing-axis broadcast, as `Binary` lowers it.
            SemanticOp::Binary(op @ (axnn::BinaryOp::Mul | axnn::BinaryOp::Add)) => {
                let (a, b) = (&operands[0], &operands[1]);
                (0..a.len().max(b.len()))
                    .map(|i| {
                        let (a, b) = (a[i % a.len()], b[i % b.len()]);
                        if op == axnn::BinaryOp::Mul {
                            a * b
                        } else {
                            a + b
                        }
                    })
                    .collect()
            }
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(result.len(), output.len());
        if self.corrupt.contains(&instruction) {
            result.iter_mut().for_each(|v| *v += 10.0);
        }
        self.results.insert(instruction, result);
        self.queue.push(instruction);
        Ok(KernelProgress::Pending)
    }

    fn advance_fused(
        &mut self,
        last: usize,
        run: FusedRun,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        base: Option<ChainedInput<'_>>,
        output: &mut [f32],
        _: KernelAccuracy,
    ) -> Result<KernelProgress, Error> {
        if self.queue.contains(&last) {
            // A retry: the same bookkeeping as a single admission.
            return self.advance_chained(last, plan, inputs, output, KernelAccuracy::Faithful);
        }
        if !self.fuse {
            return Ok(KernelProgress::Unsupported);
        }
        let operands: Vec<Vec<f32>> = inputs.iter().flatten().map(|&i| self.operand(i)).collect();
        let projected = match plan.semantic {
            SemanticOp::Binary(op) => operands[0]
                .iter()
                .zip(&operands[1])
                .map(|(a, b)| {
                    if op == axnn::BinaryOp::Mul {
                        a * b
                    } else {
                        a + b
                    }
                })
                .collect(),
            _ => Self::project(plan, &operands),
        };
        let result: Vec<f32> = match (run, base) {
            (FusedRun::SquaredRelu, None) => projected
                .iter()
                .map(|v| {
                    let r = v.max(0.0);
                    r * r
                })
                .collect(),
            (FusedRun::Accumulate, Some(base)) => self
                .operand(base)
                .iter()
                .zip(&projected)
                .map(|(b, p)| b + p)
                .collect(),
            (FusedRun::RmsNorm { width, epsilon }, Some(weight)) => {
                rms(&projected, &self.operand(weight), width, epsilon.to_f32())
            }
            other => panic!("malformed fused offer {other:?}", other = other.0),
        };
        assert_eq!(result.len(), output.len());
        self.fused.push(run);
        self.results.insert(last, result);
        self.queue.push(last);
        Ok(KernelProgress::Pending)
    }
}

fn run_windowed(executor: &mut Windowed, input: &[f32; 4]) -> Vec<f32> {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let mut arena = chain_arena();
    let inputs = [NativeRef::F32(input)];
    {
        let mut run = arena
            .begin_with_executor(&inputs, &[], &Serial, executor, KernelAccuracy::Faithful)
            .unwrap();
        loop {
            match run.advance(64).unwrap() {
                PreparedProgress::Complete => break,
                PreparedProgress::Progress { .. } => {}
                PreparedProgress::Pending => {
                    let mut wait = pin!(run.wait());
                    assert!(matches!(
                        wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Ready(Ok(()))
                    ));
                }
            }
        }
    }
    arena.output_f32(0).unwrap().to_vec()
}

fn run_windowed_checked(
    executor: &mut Windowed,
    input: &[f32; 4],
) -> (Vec<f32>, ax_exec::CheckReport) {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let mut arena = chain_arena();
    arena.set_delegate_check(true);
    let inputs = [NativeRef::F32(input)];
    {
        let mut run = arena
            .begin_with_executor(&inputs, &[], &Serial, executor, KernelAccuracy::Faithful)
            .unwrap();
        loop {
            match run.advance(64).unwrap() {
                PreparedProgress::Complete => break,
                PreparedProgress::Progress { .. } => {}
                PreparedProgress::Pending => {
                    let mut wait = pin!(run.wait());
                    assert!(matches!(
                        wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Ready(Ok(()))
                    ));
                }
            }
        }
    }
    (
        arena.output_f32(0).unwrap().to_vec(),
        arena.delegate_check(),
    )
}

/// A delegate check recomputes every delegated result natively from the
/// same host operands and reports how far each class was, and where the
/// first large error showed; the native result stands, so the output is
/// exactly the native one however wrong the device is.
#[test]
fn a_delegate_check_names_a_wrong_kernel_and_keeps_native_results() {
    let input = [1.0, -2.0, 3.0, -4.0];
    let other = ax_exec::OP_CLASS_NAMES
        .iter()
        .position(|&name| name == "other")
        .unwrap();
    let mut honest = Windowed {
        window: 4,
        ..Default::default()
    };
    let (output, report) = run_windowed_checked(&mut honest, &input);
    assert_eq!(output, native_chain(&input));
    assert_eq!(report.checked[other], 6);
    assert_eq!(report.worst[other], 0.0);
    assert_eq!(report.first, None);
    // The device's fourth result is wrong; its successors, chained on the
    // device, inherit the error, but the first names the culprit.
    let mut wrong = Windowed {
        window: 4,
        corrupt: vec![3],
        ..Default::default()
    };
    let (output, report) = run_windowed_checked(&mut wrong, &input);
    assert_eq!(output, native_chain(&input));
    assert_eq!(report.first, Some((3, other)));
    assert!(report.worst[other] > 0.5, "{report}");
    assert!(std::format!("{report}").contains("first over 1e-2: #3 other"));
    // Every result here has one shape, so the culprit is the one offender,
    // with the value it got wrong.
    let offenders: Vec<_> = report.offenders().collect();
    assert_eq!(offenders.len(), 1);
    let offender = offenders[0];
    assert_eq!((offender.instruction, offender.class), (3, other));
    assert_eq!(offender.len, input.len());
    assert_ne!(offender.delegated, offender.native);
    assert!(std::format!("{offender}").starts_with("#3 other of 4: error"));
    let mut merged = ax_exec::CheckReport::default();
    merged.merge(&report);
    merged.merge(&report);
    assert_eq!(merged.offenders().count(), 1, "one per shape");
}

fn native_chain(input: &[f32; 4]) -> Vec<f32> {
    let mut arena = chain_arena();
    arena
        .run_native_on(&[NativeRef::F32(input)], &[], &Serial)
        .unwrap();
    arena.output_f32(0).unwrap().to_vec()
}

/// A whole dependent chain admits before one wait, every operand an earlier
/// admission produces arrives as a reference to it, and the result is the
/// native result exactly.
#[test]
fn an_admission_window_runs_a_dependent_chain_on_one_wait() {
    let input = [1.5, -2.0, 0.25, 3.0];
    let mut executor = Windowed {
        window: 8,
        ..Default::default()
    };
    assert_eq!(run_windowed(&mut executor, &input), native_chain(&input));
    assert_eq!(executor.waits, 1, "the chain waited more than once");
    // Pending operands: b reads a; c reads b, a; d reads c; e reads d, b;
    // f reads e. Seven references, and x always arrives as host data.
    assert_eq!(executor.chained, 7);
}

#[test]
fn unobserved_retirement_preserves_outputs_cpu_boundaries_and_reused_slots() {
    let input = [1.5, -2.0, 0.25, 3.0];
    // Small windows and declined consumers must materialize values that are
    // still needed on the host; the whole chain can omit every intermediate.
    for (window, decline) in [(2, vec![]), (3, vec![2]), (8, vec![]), (8, vec![5])] {
        let mut executor = Windowed {
            window,
            decline,
            discard: true,
            ..Default::default()
        };
        assert_eq!(run_windowed(&mut executor, &input), native_chain(&input));
        assert!(
            !executor.discarded.contains(&5),
            "graph output must be published"
        );
        if window == 8 && executor.decline.is_empty() {
            assert_eq!(executor.discarded, [0, 1, 2, 3, 4]);
        }
    }
    let mut checked = Windowed {
        window: 8,
        discard: true,
        ..Default::default()
    };
    let (actual, report) = run_windowed_checked(&mut checked, &input);
    assert_eq!(actual, native_chain(&input));
    assert!(
        checked.discarded.is_empty(),
        "reference checks observe every result"
    );
    assert!(report.checked.iter().sum::<u64>() > 0);
}

/// A narrow window drains before admitting more, and a declined instruction
/// in the middle runs natively only after everything before it completed.
#[test]
fn declined_and_window_bounded_instructions_keep_program_order() {
    let input = [0.5, -1.0, 4.0, -3.5];
    for (window, decline) in [(2, vec![]), (3, vec![2]), (8, vec![0, 3]), (8, vec![5])] {
        let mut executor = Windowed {
            window,
            decline: decline.clone(),
            ..Default::default()
        };
        assert_eq!(
            run_windowed(&mut executor, &input),
            native_chain(&input),
            "window {window}, declining {decline:?}"
        );
        assert!(executor.queue.is_empty());
    }
}

/// `x -> matmul -> relu -> mul(r, r) -> matmul -> add(x, d) -> neg`, plus an
/// optional second reader of the relu output (which must stop that run from
/// fusing).
fn fusable_arena(escape: bool) -> PreparedNative<Global> {
    fusable_arena_options(escape, false, true, false)
}

fn fusable_arena_options(
    escape: bool,
    interleave: bool,
    borrow: bool,
    optimize: bool,
) -> PreparedNative<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let (m, k, n) = (2u32, 4u32, 3u32);
    let f32s = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[m, k], &[])
        .unwrap();
    let w1: Vec<f32> = (0..n * k).map(|i| (i as f32 - 5.0) * 0.25).collect();
    let w2: Vec<f32> = (0..k * n).map(|i| (3.0 - i as f32) * 0.125).collect();
    let w1 = b
        .tensor(DType::F32, TensorFlags::CONSTANT, &[n, k], &f32s(&w1))
        .unwrap();
    let w2 = b
        .tensor(DType::F32, TensorFlags::CONSTANT, &[k, n], &f32s(&w2))
        .unwrap();
    let value = |b: &mut GraphBuilder<Global>, cols, flags| {
        b.tensor(DType::F32, flags, &[m, cols], &[]).unwrap()
    };
    let p = value(&mut b, n, TensorFlags::default());
    let r = value(&mut b, n, TensorFlags::default());
    let s = value(&mut b, n, TensorFlags::default());
    let d = value(&mut b, k, TensorFlags::default());
    let y = value(&mut b, k, TensorFlags::default());
    let out = value(&mut b, k, TensorFlags::OUTPUT);
    let none = NodeFlags::default();
    b.node(OpCode::MATMUL_TRANSPOSED, none, &[x, w1], &[p], &[])
        .unwrap();
    if interleave {
        let side = value(&mut b, k, TensorFlags::OUTPUT);
        b.node(OpCode::NEGATE, none, &[x], &[side], &[]).unwrap();
    }
    b.node(OpCode::RELU, none, &[p], &[r], &[]).unwrap();
    b.node(OpCode::MUL, none, &[r, r], &[s], &[]).unwrap();
    b.node(OpCode::MATMUL_TRANSPOSED, none, &[s, w2], &[d], &[])
        .unwrap();
    b.node(OpCode::ADD, none, &[x, d], &[y], &[]).unwrap();
    b.node(OpCode::NEGATE, none, &[y], &[out], &[]).unwrap();
    if escape {
        let seen = value(&mut b, n, TensorFlags::OUTPUT);
        b.node(OpCode::NEGATE, none, &[r], &[seen], &[]).unwrap();
    }
    let graph = b.finish().unwrap();
    let program = NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        optimize,
        Global,
    )
    .unwrap();
    if borrow {
        program.prepare_borrowing_inputs_in(Global).unwrap()
    } else {
        program.prepare_native_in(Global).unwrap()
    }
}

fn run_fusable(executor: &mut Windowed, escape: bool, input: &[f32; 8]) -> Vec<f32> {
    run_prepared(fusable_arena(escape), executor, input)
}

fn run_prepared(
    mut arena: PreparedNative<Global>,
    executor: &mut Windowed,
    input: &[f32; 8],
) -> Vec<f32> {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let inputs = [NativeRef::F32(input)];
    {
        let mut run = arena
            .begin_with_executor(&inputs, &[], &Serial, executor, KernelAccuracy::Faithful)
            .unwrap();
        loop {
            match run.advance(64).unwrap() {
                PreparedProgress::Complete => break,
                PreparedProgress::Progress { .. } => {}
                PreparedProgress::Pending => {
                    let mut wait = pin!(run.wait());
                    assert!(matches!(
                        wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Ready(Ok(()))
                    ));
                }
            }
        }
    }
    arena.output_f32(0).unwrap().to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-5 * (1.0 + e.abs()),
            "{actual:?} vs {expected:?}"
        );
    }
}

/// Both runs fuse: `matmul -> relu -> mul(r, r)` and `matmul -> add`, each
/// one admission, the whole graph on one wait, matching the native result.
#[test]
fn fused_runs_admit_as_one_and_match_the_native_result() {
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 1.0];
    let mut native = fusable_arena(false);
    native
        .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
        .unwrap();
    let expected = native.output_f32(0).unwrap().to_vec();
    let mut executor = Windowed {
        window: 8,
        fuse: true,
        ..Default::default()
    };
    assert_close(&run_fusable(&mut executor, false, &input), &expected);
    assert_eq!(
        executor.fused,
        [FusedRun::SquaredRelu, FusedRun::Accumulate]
    );
    assert_eq!(executor.waits, 1);
    let mut resident = Windowed {
        window: 8,
        fuse: true,
        discard: true,
        ..Default::default()
    };
    assert_close(&run_fusable(&mut resident, false, &input), &expected);
    assert_eq!(resident.fused, executor.fused);
    assert!(!resident.discarded.is_empty());
    // A device that declines fusion gets the same result one by one.
    let mut unfused = Windowed {
        window: 8,
        ..Default::default()
    };
    assert_close(&run_fusable(&mut unfused, false, &input), &expected);
    assert!(unfused.fused.is_empty());
}

/// `x [2,4]` through a projection, then an ADD; `square` makes the
/// projection `[4,4]` over an intermediate that dies there and adds `x` back
/// (a residual), otherwise it is `[3,4]` plus a broadcast bias `[3]`.
fn projection_then_add(square: bool) -> PreparedNative<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let (m, k) = (2u32, 4u32);
    let n = if square { k } else { 3 };
    let f32s = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let none = NodeFlags::default();
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[m, k], &[])
        .unwrap();
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 6.0) * 0.25).collect();
    let w = b
        .tensor(DType::F32, TensorFlags::CONSTANT, &[n, k], &f32s(&w))
        .unwrap();
    let p = b
        .tensor(DType::F32, TensorFlags::default(), &[m, n], &[])
        .unwrap();
    let y = b
        .tensor(DType::F32, TensorFlags::default(), &[m, n], &[])
        .unwrap();
    let out = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[m, n], &[])
        .unwrap();
    if square {
        let h = b
            .tensor(DType::F32, TensorFlags::default(), &[m, k], &[])
            .unwrap();
        b.node(OpCode::NEGATE, none, &[x], &[h], &[]).unwrap();
        b.node(OpCode::MATMUL_TRANSPOSED, none, &[h, w], &[p], &[])
            .unwrap();
        b.node(OpCode::ADD, none, &[p, x], &[y], &[]).unwrap();
    } else {
        let bias = b
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[n],
                &f32s(&[0.5, -1.0, 2.0]),
            )
            .unwrap();
        b.node(OpCode::MATMUL_TRANSPOSED, none, &[x, w], &[p], &[])
            .unwrap();
        b.node(OpCode::ADD, none, &[p, bias], &[y], &[]).unwrap();
    }
    b.node(OpCode::NEGATE, none, &[y], &[out], &[]).unwrap();
    let graph = b.finish().unwrap();
    NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        false,
        Global,
    )
    .unwrap()
    .prepare_borrowing_inputs_in(Global)
    .unwrap()
}

/// A broadcast bias add is not an accumulate: it runs one instruction at a
/// time and matches the native result. A square projection's residual add
/// matches too, fused or not. (Whether its output takes the dying input's
/// slot is the allocator's choice; `fused_run` refuses that case either way.)
#[test]
fn only_element_wise_adds_with_free_outputs_accumulate() {
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 1.0];
    for square in [false, true] {
        let mut native = projection_then_add(square);
        native
            .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
            .unwrap();
        let expected = native.output_f32(0).unwrap().to_vec();
        let mut executor = Windowed {
            window: 8,
            fuse: true,
            ..Default::default()
        };
        assert_close(
            &run_prepared(projection_then_add(square), &mut executor, &input),
            &expected,
        );
        if !square {
            assert!(executor.fused.is_empty(), "a broadcast add is not fused");
        }
    }
}

/// A relu output that another node also reads is published, so the
/// squared-ReLU run is not offered; the accumulate run still is.
#[test]
fn a_run_whose_intermediate_escapes_is_not_offered() {
    let input = [0.5, 1.0, -2.0, 1.5, 2.0, -0.5, 1.25, -1.0];
    let mut native = fusable_arena(true);
    native
        .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
        .unwrap();
    let expected = native.output_f32(0).unwrap().to_vec();
    let mut executor = Windowed {
        window: 8,
        fuse: true,
        ..Default::default()
    };
    assert_close(&run_fusable(&mut executor, true, &input), &expected);
    assert_eq!(executor.fused, [FusedRun::Accumulate]);
}

/// An independent branch used to split the projection/RELU/square run.
/// Compaction must preserve CPU bits, owned/borrowed ingress, and the rule
/// that an escaping intermediate cannot be hidden by device fusion.
#[test]
fn interleaved_projection_tails_fuse_without_changing_values() {
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 1.0];
    for borrow in [false, true] {
        for escape in [false, true] {
            let mut reference = fusable_arena_options(escape, true, borrow, false);
            reference
                .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
                .unwrap();
            let expected = reference.output_f32(0).unwrap().to_vec();
            let mut native = fusable_arena_options(escape, true, borrow, true);
            native
                .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
                .unwrap();
            for output in 0..reference.output_count() {
                let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(native.output_f32(output).unwrap()),
                    bits(reference.output_f32(output).unwrap())
                );
            }
            let mut executor = Windowed {
                window: 8,
                fuse: true,
                ..Default::default()
            };
            assert_close(
                &run_prepared(
                    fusable_arena_options(escape, true, borrow, true),
                    &mut executor,
                    &input,
                ),
                &expected,
            );
            let expected_runs = if escape {
                vec![FusedRun::Accumulate]
            } else {
                vec![FusedRun::SquaredRelu, FusedRun::Accumulate]
            };
            assert_eq!(executor.fused, expected_runs);
        }
    }
}

fn rms(input: &[f32], weights: &[f32], width: usize, epsilon: f32) -> Vec<f32> {
    input
        .chunks_exact(width)
        .enumerate()
        .flat_map(|(row, values)| {
            let inverse = (values.iter().map(|x| x * x).sum::<f32>() / width as f32 + epsilon)
                .sqrt()
                .recip();
            values.iter().enumerate().map(move |(column, value)| {
                value * inverse * weights[(row * width + column) % weights.len()]
            })
        })
        .collect()
}

fn binary_norm_arena(multiply: bool, escape: bool, partial: bool) -> PreparedNative<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[2, 4], &[])
        .unwrap();
    let bytes: Vec<u8> = [0.5_f32, -2.0, 3.0, 0.0, -1.0, 0.25, 1.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let gate = b
        .tensor(DType::F32, TensorFlags::CONSTANT, &[2, 4], &bytes)
        .unwrap();
    let binary = b
        .tensor(
            DType::F32,
            if escape {
                TensorFlags::OUTPUT
            } else {
                TensorFlags::default()
            },
            &[2, 4],
            &[],
        )
        .unwrap();
    b.node(
        if multiply { OpCode::MUL } else { OpCode::ADD },
        NodeFlags::default(),
        &[x, gate],
        &[binary],
        &[],
    )
    .unwrap();
    let dims = [if partial { 1 } else { 2 }, 2, 2];
    let grouped = b
        .tensor(DType::F32, TensorFlags::default(), &dims, &[])
        .unwrap();
    b.node(
        OpCode::VIEW,
        NodeFlags::default(),
        &[binary],
        &[grouped],
        &ax::tensor_ops::encode_view_attributes(if partial { 2 } else { 0 }),
    )
    .unwrap();
    let weights: Vec<u8> = [0.75_f32, 2.0, -1.0, 0.125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let weight = b
        .tensor(DType::F32, TensorFlags::CONSTANT, &[2, 2], &weights)
        .unwrap();
    let norm = b
        .tensor(DType::F32, TensorFlags::default(), &dims, &[])
        .unwrap();
    b.node(
        OpCode::RMS_NORM,
        NodeFlags::default(),
        &[grouped, weight],
        &[norm],
        &ax::tensor_ops::encode_rms_norm_attributes(1e-5, 2).unwrap(),
    )
    .unwrap();
    let out = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &dims, &[])
        .unwrap();
    b.node(OpCode::NEGATE, NodeFlags::default(), &[norm], &[out], &[])
        .unwrap();
    let graph = b.finish().unwrap();
    NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        false,
        Global,
    )
    .unwrap()
    .prepare_borrowing_inputs_in(Global)
    .unwrap()
}

#[test]
fn binary_rms_fusion_preserves_group_weights_and_declines_observable_or_partial_inputs() {
    let input = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 1.0];
    for multiply in [false, true] {
        for (escape, partial) in [(false, false), (true, false), (false, true)] {
            let mut reference = binary_norm_arena(multiply, escape, partial);
            reference
                .run_native_on(&[NativeRef::F32(&input)], &[], &Serial)
                .unwrap();
            for fuse in [false, true] {
                let mut arena = binary_norm_arena(multiply, escape, partial);
                let mut executor = Windowed {
                    window: 8,
                    fuse,
                    discard: true,
                    ..Default::default()
                };
                // Preserve every graph output, including an escaping binary value.
                let inputs = [NativeRef::F32(&input)];
                {
                    let mut run = arena
                        .begin_with_executor(
                            &inputs,
                            &[],
                            &Serial,
                            &mut executor,
                            KernelAccuracy::Faithful,
                        )
                        .unwrap();
                    loop {
                        match run.advance(64).unwrap() {
                            PreparedProgress::Complete => break,
                            PreparedProgress::Progress { .. } => {}
                            PreparedProgress::Pending => {
                                use core::{
                                    future::Future,
                                    pin::pin,
                                    task::{Context, Poll, Waker},
                                };
                                let mut wait = pin!(run.wait());
                                assert!(matches!(
                                    wait.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                                    Poll::Ready(Ok(()))
                                ));
                            }
                        }
                    }
                }
                for index in 0..reference.output_count() {
                    assert_close(
                        arena.output_f32(index).unwrap(),
                        reference.output_f32(index).unwrap(),
                    );
                }
                assert_eq!(!executor.fused.is_empty(), fuse && !escape && !partial);
            }
        }
    }
}
