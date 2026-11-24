#![feature(allocator_api)]
use ax::kv::{AttentionDecodeParams, KvCacheParams, KvCacheStorage, ResourceKey};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeProgress, NativeResourceBinding};
use axnn::lowering::SemanticOp;
use axnn_cpu::{CpuError, KvCacheRef};
use std::alloc::Global;

const CACHE: ResourceKey = ResourceKey {
    group: 1,
    binding: 0,
};
const HEADS: usize = 2;
const QUERY_HEADS: usize = 4;
const DIM: usize = 8;

fn cache(capacity: u32) -> KvCacheParams {
    KvCacheParams::new(
        CACHE,
        HEADS as u32,
        DIM as u32,
        capacity,
        4,
        KvCacheStorage::E4M3TokenScaleE8M0,
    )
    .unwrap()
}

fn attention() -> AttentionDecodeParams {
    AttentionDecodeParams::new(cache(8), QUERY_HEADS as u32, 0.353_553_4).unwrap()
}

/// One decode layer: append a token's keys and values, then attend. With
/// `split_position`, the read takes its own position input instead of the
/// append's.
fn layer(split_position: bool) -> ax::GraphBuf<Global> {
    layer_rows(split_position, 1)
}

fn layer_rows(split_position: bool, tokens: u32) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let position = b.tensor(DType::I32, TensorFlags::INPUT, &[], &[]).unwrap();
    let rows = [tokens, HEADS as u32, DIM as u32];
    let keys = b
        .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
        .unwrap();
    let values = b
        .tensor(DType::F32, TensorFlags::INPUT, &rows, &[])
        .unwrap();
    let queries = b
        .tensor(
            DType::F32,
            TensorFlags::INPUT,
            &[tokens, QUERY_HEADS as u32, DIM as u32],
            &[],
        )
        .unwrap();
    let read_position = if split_position {
        b.tensor(DType::I32, TensorFlags::INPUT, &[], &[]).unwrap()
    } else {
        position
    };
    let token = b.kv_append(position, keys, values, cache(8)).unwrap();
    b.attention_decode(
        token,
        read_position,
        queries,
        tokens,
        attention(),
        TensorFlags::OUTPUT,
    )
    .unwrap();
    b.finish().unwrap()
}

fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: initialized, padding-free f32/i32 words, borrowed immutably.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((index * 37 + seed * 101) % 29) as f32 - 14.0) / 5.0)
        .collect()
}

#[test]
fn decode_steps_append_then_attend_like_the_direct_kernels() {
    let program =
        NativeProgram::from_graph_in(layer(false), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let plans: Vec<_> = program.kernel_plans().copied().collect();
    assert_eq!(
        plans.iter().map(|plan| plan.semantic).collect::<Vec<_>>(),
        [SemanticOp::KvAppend, SemanticOp::AttentionDecode]
    );
    let mut prepared = program.prepare_native_in(Global).unwrap();
    let storage_bytes = cache(8).storage_bytes().unwrap();
    let mut storage = vec![0_u8; storage_bytes];
    let mut direct = vec![0_u8; storage_bytes];
    for step in 0..3_usize {
        let position = [step as i32];
        let keys = values(HEADS * DIM, 3 * step);
        let values_ = values(HEADS * DIM, 3 * step + 1);
        let queries = values(QUERY_HEADS * DIM, 3 * step + 2);
        let inputs = [
            bytes(&position),
            bytes(&keys),
            bytes(&values_),
            bytes(&queries),
        ];
        prepared
            .run_with_resources(
                &inputs,
                &[NativeResourceBinding::kv_cache(
                    CACHE,
                    KvCacheRef::new(&mut storage, cache(8)).unwrap(),
                )],
            )
            .unwrap();

        let mut token = [0_i32];
        axnn_cpu::invoke_kv_append(
            &plans[0],
            KvCacheRef::new(&mut direct, cache(8)).unwrap(),
            &position,
            &keys,
            &values_,
            &mut token,
        )
        .unwrap();
        let mut output = vec![0.0_f32; QUERY_HEADS * DIM];
        let mut scratch = vec![0.0; axnn_cpu::attention_scratch_elements(attention())];
        axnn_cpu::invoke_attention_decode(
            &plans[1],
            KvCacheRef::new(&mut direct, cache(8)).unwrap(),
            &token,
            &position,
            &queries,
            &mut output,
            &mut scratch,
            &axnn_cpu::Serial,
        )
        .unwrap();
        assert_eq!(storage, direct, "step {step} pages");
        let actual = prepared.output_f32(0).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {step} attention"
        );
        assert!(actual.iter().all(|v| v.is_finite()));
    }

    // The unprepared invocation path executes the same instructions.
    let position = [3_i32];
    let keys = values(HEADS * DIM, 9);
    let values_ = values(HEADS * DIM, 10);
    let queries = values(QUERY_HEADS * DIM, 11);
    let inputs = [
        bytes(&position),
        bytes(&keys),
        bytes(&values_),
        bytes(&queries),
    ];
    let mut fresh = storage.clone();
    prepared
        .run_with_resources(
            &inputs,
            &[NativeResourceBinding::kv_cache(
                CACHE,
                KvCacheRef::new(&mut storage, cache(8)).unwrap(),
            )],
        )
        .unwrap();
    let expected = prepared.output_f32(0).unwrap().to_vec();
    let bindings = [NativeResourceBinding::kv_cache(
        CACHE,
        KvCacheRef::new(&mut fresh, cache(8)).unwrap(),
    )];
    let mut run = program
        .begin_with_resources_in(&inputs, &bindings, Global)
        .unwrap();
    assert_eq!(run.advance(usize::MAX).unwrap(), NativeProgress::Ready);
    let outputs = run.take_retained_in().unwrap();
    assert_eq!(outputs[0].as_bytes(), bytes(&expected));
    assert!(prepared.scratch_bytes() >= 4 * (2 * DIM + 2 * (QUERY_HEADS / HEADS)));
}

#[test]
fn bindings_and_positions_are_checked_before_the_cache_is_touched() {
    let program =
        NativeProgram::from_graph_in(layer(false), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut prepared = program.prepare_native_in(Global).unwrap();
    let keys = values(HEADS * DIM, 0);
    let queries = values(QUERY_HEADS * DIM, 1);
    let mut storage = vec![0x5a_u8; cache(8).storage_bytes().unwrap()];

    for position in [-1_i32, 8] {
        let position = [position];
        let inputs = [
            bytes(&position),
            bytes(&keys),
            bytes(&keys),
            bytes(&queries),
        ];
        let result = prepared.run_with_resources(
            &inputs,
            &[NativeResourceBinding::kv_cache(
                CACHE,
                KvCacheRef::new(&mut storage, cache(8)).unwrap(),
            )],
        );
        assert!(matches!(
            result,
            Err(Error::Cpu(CpuError::ShapeMismatch { .. }))
        ));
        assert!(
            prepared.output_f32(0).is_none(),
            "a failed run publishes nothing"
        );
    }
    assert!(storage.iter().all(|&byte| byte == 0x5a));

    let position = [0_i32];
    let inputs = [
        bytes(&position),
        bytes(&keys),
        bytes(&keys),
        bytes(&queries),
    ];
    assert!(matches!(
        prepared.run_with_resources(&inputs, &[]),
        Err(Error::MissingResource(CACHE))
    ));
    let mut larger = vec![0_u8; cache(16).storage_bytes().unwrap()];
    assert!(matches!(
        prepared.run_with_resources(
            &inputs,
            &[NativeResourceBinding::kv_cache(
                CACHE,
                KvCacheRef::new(&mut larger, cache(16)).unwrap(),
            )],
        ),
        Err(Error::KvCacheGeometry {
            resource: CACHE,
            ..
        })
    ));
}

#[test]
fn a_read_whose_position_differs_from_its_append_is_not_admitted() {
    assert!(
        NativeProgram::from_graph_in(layer(true), AxCpuKernelPolicy::default(), true, Global)
            .is_err()
    );
}

fn resident(params: KvCacheParams) -> ax_exec::ResidentKvCache {
    ax_exec::ResidentKvCache::new(
        core::num::NonZeroU64::new(7).unwrap(),
        core::num::NonZeroU64::new(11).unwrap(),
        params,
    )
    .unwrap()
}

/// Contract test backend: storage belongs to the executor, and the existing
/// CPU kernels provide a numerical oracle. This is not a GPU timing witness.
struct ResidentExecutor {
    bytes: Vec<u8>,
    pending: Option<usize>,
    window: usize,
    decline: bool,
    complete: usize,
}

impl ax_exec::PreparedKernelExecutor for ResidentExecutor {
    fn begin(&mut self) -> Result<(), Error> {
        if self.pending.is_some() {
            Err(Error::InvocationFinished)
        } else {
            Ok(())
        }
    }
    async fn wait(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn admission_window(&self) -> usize {
        self.window
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
    fn advance_kv(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<ax_exec::NativeRef<'_>>; 3],
        output: ax_exec::KvOutput<'_>,
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        use ax_exec::{KernelProgress, KvOutput, NativeRef};
        if cache != resident(cache.params()) {
            return Err(Error::ResidentResourceRequiresExecutor(
                cache.params().resource,
            ));
        }
        if self.decline {
            return Ok(KernelProgress::Unsupported);
        }
        if self.pending.is_none() {
            self.pending = Some(instruction);
            return Ok(KernelProgress::Pending);
        }
        assert_eq!(self.pending, Some(instruction));
        let cache = KvCacheRef::new(&mut self.bytes, cache.params()).unwrap();
        match (output, inputs) {
            (
                KvOutput::Appended(token),
                [
                    Some(NativeRef::I32(pos)),
                    Some(NativeRef::F32(k)),
                    Some(NativeRef::F32(v)),
                ],
            ) => {
                axnn_cpu::invoke_kv_append(plan, cache, pos, k, v, token)?;
            }
            (
                KvOutput::Attention(out),
                [
                    Some(NativeRef::I32(token)),
                    Some(NativeRef::I32(pos)),
                    Some(NativeRef::F32(q)),
                ],
            ) => {
                let mut scratch = vec![0.; axnn_cpu::attention_scratch_elements(attention())];
                axnn_cpu::invoke_attention_decode(
                    plan,
                    cache,
                    token,
                    pos,
                    q,
                    out,
                    &mut scratch,
                    &axnn_cpu::Serial,
                )?;
            }
            _ => panic!("wrong resident KV invocation"),
        }
        self.pending = None;
        self.complete += 1;
        Ok(KernelProgress::Complete)
    }
}

fn executor(window: usize) -> ResidentExecutor {
    ResidentExecutor {
        bytes: vec![0; cache(8).storage_bytes().unwrap()],
        pending: None,
        window,
        decline: false,
        complete: 0,
    }
}

#[test]
fn resident_kv_persists_across_invocations_and_rejected_suffix_is_overwritten() {
    use ax_exec::{NativeRef, PreparedProgress};
    for window in [1, 8] {
        let program = NativeProgram::from_graph_in(
            layer_rows(false, 2),
            AxCpuKernelPolicy::default(),
            true,
            Global,
        )
        .unwrap();
        let mut prepared = program.prepare_native_in(Global).unwrap();
        let mut reference = program.prepare_native_in(Global).unwrap();
        let bindings = [NativeResourceBinding::resident_kv_cache(resident(cache(8)))];
        let mut backend = executor(window);
        let mut host = vec![0_u8; cache(8).storage_bytes().unwrap()];
        // Advance, commit only the first row, overwrite the rejected row,
        // then reset to a new sequence. Stale suffix bytes differ on purpose.
        for (seed, pos) in [0, 1, 3, 0].into_iter().enumerate() {
            let position = [pos];
            let keys = values(2 * HEADS * DIM, seed * 3);
            let vals = values(2 * HEADS * DIM, seed * 3 + 1);
            let query = values(2 * QUERY_HEADS * DIM, seed * 3 + 2);
            let inputs = [
                NativeRef::I32(&position),
                NativeRef::F32(&keys),
                NativeRef::F32(&vals),
                NativeRef::F32(&query),
            ];
            let mut run = prepared
                .begin_with_executor(
                    &inputs,
                    &bindings,
                    &axnn_cpu::Serial,
                    &mut backend,
                    axnn::lowering::KernelAccuracy::Faithful,
                )
                .unwrap();
            for _ in 0..8 {
                if run.advance(2).unwrap() == PreparedProgress::Complete {
                    break;
                }
            }
            drop(run);
            reference
                .run_with_native_inputs_and_resources(
                    &inputs,
                    &[NativeResourceBinding::kv_cache(
                        CACHE,
                        KvCacheRef::new(&mut host, cache(8)).unwrap(),
                    )],
                )
                .unwrap();
            assert_eq!(
                prepared.output_f32(0).unwrap(),
                reference.output_f32(0).unwrap()
            );
            // Poison positions outside the committed next prefix. Attention
            // must never read these before append overwrites them.
            if seed == 0 {
                backend.bytes[cache(8).page_bytes().unwrap()..].fill(0xff);
            }
        }
        assert_eq!(backend.complete, 8);
    }
}

#[test]
fn resident_kv_refuses_cpu_fallback_geometry_mismatch_and_abandoned_admission() {
    use ax_exec::{NativeRef, PreparedProgress};
    let program =
        NativeProgram::from_graph_in(layer(false), AxCpuKernelPolicy::default(), true, Global)
            .unwrap();
    let mut prepared = program.prepare_native_in(Global).unwrap();
    let position = [0];
    let keys = values(HEADS * DIM, 0);
    let query = values(QUERY_HEADS * DIM, 1);
    let inputs = [
        NativeRef::I32(&position),
        NativeRef::F32(&keys),
        NativeRef::F32(&keys),
        NativeRef::F32(&query),
    ];
    let bindings = [NativeResourceBinding::resident_kv_cache(resident(cache(8)))];
    assert!(matches!(
        prepared.run_with_native_inputs_and_resources(&inputs, &bindings),
        Err(Error::ResidentResourceRequiresExecutor(CACHE))
    ));
    let mut backend = executor(8);
    // An identity from another runtime epoch must not access this allocation.
    let stale = [NativeResourceBinding::resident_kv_cache(
        ax_exec::ResidentKvCache::new(
            core::num::NonZeroU64::new(8).unwrap(),
            resident(cache(8)).allocation(),
            cache(8),
        )
        .unwrap(),
    )];
    let mut run = prepared
        .begin_with_executor(
            &inputs,
            &stale,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful,
        )
        .unwrap();
    assert!(matches!(
        run.advance(2),
        Err(Error::ResidentResourceRequiresExecutor(CACHE))
    ));
    drop(run);
    assert!(backend.bytes.iter().all(|b| *b == 0));
    prepared.set_delegate_check(true);
    let mut run = prepared
        .begin_with_executor(
            &inputs,
            &bindings,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful,
        )
        .unwrap();
    assert!(matches!(
        run.advance(2),
        Err(Error::ResidentResourceRequiresExecutor(CACHE))
    ));
    drop(run);
    assert!(backend.pending.is_none());
    assert!(backend.bytes.iter().all(|b| *b == 0));
    prepared.set_delegate_check(false);
    let wrong = [NativeResourceBinding::resident_kv_cache(resident(cache(
        16,
    )))];
    assert!(matches!(
        prepared.begin_with_executor(
            &inputs,
            &wrong,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful
        ),
        Err(Error::KvCacheGeometry { .. })
    ));
    backend.decline = true;
    let mut run = prepared
        .begin_with_executor(
            &inputs,
            &bindings,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful,
        )
        .unwrap();
    assert!(matches!(
        run.advance(2),
        Err(Error::ResidentResourceRequiresExecutor(CACHE))
    ));
    drop(run);
    assert!(prepared.output_f32(0).is_none());
    backend.decline = false;
    let mut run = prepared
        .begin_with_executor(
            &inputs,
            &bindings,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful,
        )
        .unwrap();
    assert_eq!(run.advance(2).unwrap(), PreparedProgress::Pending);
    drop(run);
    assert!(matches!(
        prepared.begin_with_executor(
            &inputs,
            &bindings,
            &axnn_cpu::Serial,
            &mut backend,
            axnn::lowering::KernelAccuracy::Faithful
        ),
        Err(Error::InvocationFinished)
    ));
    assert!(
        backend.pending.is_some(),
        "dropping graph work must not drop owned device work"
    );
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

/// A decode layer whose keys, values and queries come from pure kernels
/// (negations of its inputs) that a window admits, as projections would.
fn projected_layer(tokens: u32) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let position = b.tensor(DType::I32, TensorFlags::INPUT, &[], &[]).unwrap();
    let rows = [tokens, HEADS as u32, DIM as u32];
    let query_rows = [tokens, QUERY_HEADS as u32, DIM as u32];
    let mut negated = |b: &mut GraphBuilder<Global>, dims: &[u32]| -> TensorId {
        let x = b.tensor(DType::F32, TensorFlags::INPUT, dims, &[]).unwrap();
        let y = b
            .tensor(DType::F32, TensorFlags::default(), dims, &[])
            .unwrap();
        b.node(OpCode::NEGATE, NodeFlags::default(), &[x], &[y], &[])
            .unwrap();
        y
    };
    let keys = negated(&mut b, &rows);
    let values = negated(&mut b, &rows);
    let queries = negated(&mut b, &query_rows);
    let token = b.kv_append(position, keys, values, cache(8)).unwrap();
    b.attention_decode(
        token,
        position,
        queries,
        tokens,
        attention(),
        TensorFlags::OUTPUT,
    )
    .unwrap();
    b.finish().unwrap()
}

/// A window executor that owns a resident cache: it admits the negations
/// and, when `take`, the KV append and read with their operands still
/// pending, computing each admission in order with the CPU kernels (as a
/// device would) and publishing it once a `wait` made it visible. Declining,
/// it serves the strict path's `advance_kv` directly.
#[derive(Default)]
struct ChainedKv {
    take: bool,
    bytes: Vec<u8>,
    /// Admissions' results, as the device holds them.
    rows: std::collections::BTreeMap<usize, Vec<f32>>,
    tokens: std::collections::BTreeMap<usize, Vec<i32>>,
    /// Admitted and not yet completed, in admission order.
    queue: Vec<usize>,
    flushed: usize,
    waits: usize,
    /// KV offers, their operands that were pending, declines, direct calls.
    kv_offers: usize,
    pending_operands: usize,
    declined: usize,
    direct: usize,
}

impl ChainedKv {
    fn new(take: bool) -> Self {
        Self {
            take,
            bytes: vec![0; cache(8).storage_bytes().unwrap()],
            ..Default::default()
        }
    }

    /// Pending until a wait made the admission visible, then retired.
    fn retry(&mut self, instruction: usize) -> Option<bool> {
        let position = self
            .queue
            .iter()
            .position(|&queued| queued == instruction)?;
        assert_eq!(position, 0, "admissions retire in admission order");
        if self.flushed == 0 {
            return Some(false);
        }
        self.queue.remove(0);
        self.flushed -= 1;
        Some(true)
    }

    fn f32s(&mut self, input: Option<ax_exec::ChainedInput<'_>>) -> Vec<f32> {
        match input.expect("operand") {
            ax_exec::ChainedInput::Host(ax_exec::NativeRef::F32(values)) => values.to_vec(),
            ax_exec::ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => {
                self.pending_operands += 1;
                self.rows[&instruction][offset..offset + elements].to_vec()
            }
            ax_exec::ChainedInput::Host(_) => panic!("an F32 operand"),
        }
    }

    fn i32s(&mut self, input: Option<ax_exec::ChainedInput<'_>>) -> Vec<i32> {
        match input.expect("operand") {
            ax_exec::ChainedInput::Host(ax_exec::NativeRef::I32(values)) => values.to_vec(),
            ax_exec::ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => {
                self.pending_operands += 1;
                self.tokens[&instruction][offset..offset + elements].to_vec()
            }
            ax_exec::ChainedInput::Host(_) => panic!("an I32 operand"),
        }
    }
}

impl ax_exec::PreparedKernelExecutor for ChainedKv {
    fn begin(&mut self) -> Result<(), Error> {
        assert!(self.queue.is_empty());
        self.rows.clear();
        self.tokens.clear();
        Ok(())
    }
    async fn wait(&mut self) -> Result<(), Error> {
        self.waits += 1;
        self.flushed = self.queue.len();
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
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        output: &mut [f32],
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        use ax_exec::KernelProgress;
        match self.retry(instruction) {
            Some(false) => return Ok(KernelProgress::Pending),
            Some(true) => {
                output.copy_from_slice(&self.rows[&instruction]);
                return Ok(KernelProgress::Complete);
            }
            None => {}
        }
        if plan.semantic != SemanticOp::Unary(axnn::UnaryOp::Neg) {
            return Ok(KernelProgress::Unsupported);
        }
        let negated = self.f32s(inputs[0]).iter().map(|v| -v).collect();
        self.rows.insert(instruction, negated);
        self.queue.push(instruction);
        Ok(KernelProgress::Pending)
    }
    fn advance_kv_chained(
        &mut self,
        instruction: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<ax_exec::ChainedInput<'_>>; 3],
        output: ax_exec::KvOutput<'_>,
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        use ax_exec::{KernelProgress, KvOutput};
        assert_eq!(cache, resident(cache.params()));
        match self.retry(instruction) {
            Some(false) => return Ok(KernelProgress::Pending),
            Some(true) => {
                match output {
                    KvOutput::Appended(token) => token.copy_from_slice(&self.tokens[&instruction]),
                    KvOutput::Attention(rows) => rows.copy_from_slice(&self.rows[&instruction]),
                }
                return Ok(KernelProgress::Complete);
            }
            None => {}
        }
        self.kv_offers += 1;
        if !self.take {
            self.declined += 1;
            return Ok(KernelProgress::Unsupported);
        }
        let params = cache.params();
        match output {
            KvOutput::Appended(token) => {
                let position = self.i32s(inputs[0]);
                let (keys, values) = (self.f32s(inputs[1]), self.f32s(inputs[2]));
                let mut appended = vec![0; token.len()];
                let cache = KvCacheRef::new(&mut self.bytes, params).unwrap();
                axnn_cpu::invoke_kv_append(plan, cache, &position, &keys, &values, &mut appended)?;
                self.tokens.insert(instruction, appended);
            }
            KvOutput::Attention(rows) => {
                let token = self.i32s(inputs[0]);
                let position = self.i32s(inputs[1]);
                let queries = self.f32s(inputs[2]);
                let mut out = vec![0.0; rows.len()];
                let mut scratch = vec![0.; axnn_cpu::attention_scratch_elements(attention())];
                let cache = KvCacheRef::new(&mut self.bytes, params).unwrap();
                axnn_cpu::invoke_attention_decode(
                    plan,
                    cache,
                    &token,
                    &position,
                    &queries,
                    &mut out,
                    &mut scratch,
                    &axnn_cpu::Serial,
                )?;
                self.rows.insert(instruction, out);
            }
        }
        self.queue.push(instruction);
        Ok(KernelProgress::Pending)
    }
    fn advance_kv(
        &mut self,
        _: usize,
        plan: &axnn_cpu::CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<ax_exec::NativeRef<'_>>; 3],
        output: ax_exec::KvOutput<'_>,
        _: axnn::lowering::KernelAccuracy,
    ) -> Result<ax_exec::KernelProgress, Error> {
        use ax_exec::{KernelProgress, KvOutput, NativeRef};
        self.direct += 1;
        let kv = KvCacheRef::new(&mut self.bytes, cache.params()).unwrap();
        match (output, inputs) {
            (
                KvOutput::Appended(token),
                [
                    Some(NativeRef::I32(position)),
                    Some(NativeRef::F32(keys)),
                    Some(NativeRef::F32(values)),
                ],
            ) => axnn_cpu::invoke_kv_append(plan, kv, position, keys, values, token)?,
            (
                KvOutput::Attention(out),
                [
                    Some(NativeRef::I32(token)),
                    Some(NativeRef::I32(position)),
                    Some(NativeRef::F32(queries)),
                ],
            ) => {
                let mut scratch = vec![0.; axnn_cpu::attention_scratch_elements(attention())];
                axnn_cpu::invoke_attention_decode(
                    plan,
                    kv,
                    token,
                    position,
                    queries,
                    out,
                    &mut scratch,
                    &axnn_cpu::Serial,
                )?
            }
            _ => panic!("wrong resident KV invocation"),
        }
        Ok(KernelProgress::Complete)
    }
}

/// A resident KV append and read ride the admission window behind the
/// kernels producing their keys, values and queries: offered with those
/// operands and the append's token still pending, the whole layer needs a
/// single wait, and its output is the host cache's. A declining executor
/// gets the drained path it had before, `advance_kv` over host operands.
#[test]
fn resident_kv_operations_ride_the_window_with_their_operands_pending() {
    use ax_exec::{NativeRef, PreparedProgress};
    let program = NativeProgram::from_graph_in(
        projected_layer(2),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap();
    let mut prepared = program.prepare_native_in(Global).unwrap();
    let mut reference = program.prepare_native_in(Global).unwrap();
    let bindings = [NativeResourceBinding::resident_kv_cache(resident(cache(8)))];
    for take in [true, false] {
        let mut backend = ChainedKv::new(take);
        let mut host = vec![0_u8; cache(8).storage_bytes().unwrap()];
        for (seed, pos) in [0, 2].into_iter().enumerate() {
            let position = [pos];
            let keys = values(2 * HEADS * DIM, seed * 3 + 40);
            let vals = values(2 * HEADS * DIM, seed * 3 + 41);
            let query = values(2 * QUERY_HEADS * DIM, seed * 3 + 42);
            let inputs = [
                NativeRef::I32(&position),
                NativeRef::F32(&keys),
                NativeRef::F32(&vals),
                NativeRef::F32(&query),
            ];
            let mut run = prepared
                .begin_with_executor(
                    &inputs,
                    &bindings,
                    &axnn_cpu::Serial,
                    &mut backend,
                    axnn::lowering::KernelAccuracy::Faithful,
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
                assert!(rounds < 64, "the layer finishes");
            }
            drop(run);
            reference
                .run_with_native_inputs_and_resources(
                    &inputs,
                    &[NativeResourceBinding::kv_cache(
                        CACHE,
                        KvCacheRef::new(&mut host, cache(8)).unwrap(),
                    )],
                )
                .unwrap();
            assert_eq!(
                prepared.output_f32(0).unwrap(),
                reference.output_f32(0).unwrap(),
                "take {take}, step {seed}"
            );
        }
        if take {
            assert_eq!(backend.kv_offers, 4, "append and read, offered once a step");
            assert_eq!(backend.pending_operands, 8, "keys, values, token, queries");
            assert_eq!(backend.waits, 2, "one wait a step");
            assert_eq!((backend.declined, backend.direct), (0, 0));
        } else {
            // Every offer declined (behind pending work, then again after the
            // drain): each operation was served once on the strict path.
            assert_eq!(backend.declined, backend.kv_offers);
            assert_eq!(backend.direct, 4);
            assert_eq!(backend.pending_operands, 0);
        }
        assert!(backend.queue.is_empty());
    }
}
