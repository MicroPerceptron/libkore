#![feature(allocator_api)]
//! Kernels split across a caller's threads compute every element as a
//! serial run does: outputs are bit-identical.
use ax::moe::{ExpertActivation, MoeParams, ResourceKey};
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{
    AxCpuKernelPolicy, ForkJoin, NativeProgram, NativeRef, NativeResourceBinding, Serial,
};
use axnn_cpu::fp8::f32_to_e4m3_rne;
use axnn_cpu::{ExpertStoreRef, ExpertWeights};
use std::alloc::Global;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One scoped thread per task, counting how many tasks were asked for.
#[derive(Default)]
struct Threads {
    largest: AtomicUsize,
}

unsafe impl ForkJoin for Threads {
    fn width(&self) -> usize {
        4
    }
    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        self.largest.fetch_max(tasks, Ordering::Relaxed);
        std::thread::scope(|scope| {
            for index in 0..tasks {
                scope.spawn(move || task(index));
            }
        });
    }
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 29 + seed * 13) % 37) as f32 - 18.0) / 17.0)
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

const HIDDEN: usize = 256;
const OUT: usize = 640;

/// `x [1, HIDDEN] · W^T` for a weight input of each float dtype.
#[test]
fn float_weight_products_split_across_threads_bit_identically() {
    let x = values(HIDDEN, 1);
    let w = values(OUT * HIDDEN, 2);
    let fp8: Vec<u8> = w.iter().map(|&v| f32_to_e4m3_rne(v)).collect();
    let bf16: Vec<u8> = w
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let cases: [(DType, NativeRef<'_>); 3] = [
        (DType::B8, NativeRef::Packed(DType::B8, &fp8)),
        (DType::B16, NativeRef::Packed(DType::B16, &bf16)),
        (DType::F32, NativeRef::F32(&w)),
    ];
    for (dtype, weight) in cases {
        let mut b = GraphBuilder::new_in(Global);
        let input = b
            .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN as u32], &[])
            .unwrap();
        let matrix = b
            .tensor(dtype, TensorFlags::INPUT, &[OUT as u32, HIDDEN as u32], &[])
            .unwrap();
        let out = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, OUT as u32], &[])
            .unwrap();
        b.node(
            OpCode::MATMUL_TRANSPOSED,
            NodeFlags::default(),
            &[input, matrix],
            &[out],
            &[],
        )
        .unwrap();
        let program = NativeProgram::from_graph_in(
            b.finish().unwrap(),
            AxCpuKernelPolicy::default(),
            true,
            Global,
        )
        .unwrap();
        let mut arena = program.prepare_borrowing_inputs_in(Global).unwrap();
        let inputs = [NativeRef::F32(&x), weight];
        arena.run_native_on(&inputs, &[], &Serial).unwrap();
        let serial = bits(arena.output_f32(0).unwrap());
        let threads = Threads::default();
        arena.run_native_on(&inputs, &[], &threads).unwrap();
        assert!(threads.largest.into_inner() > 1, "{dtype:?} was not split");
        assert_eq!(bits(arena.output_f32(0).unwrap()), serial, "{dtype:?}");
    }
}

/// A routed expert block whose projections are large enough to split.
#[test]
fn expert_projections_split_across_threads_bit_identically() {
    const EXPERTS: usize = 6;
    const INNER: usize = 384;
    const STORE: ResourceKey = ResourceKey {
        group: 1,
        binding: 5,
    };
    let params = MoeParams::new(
        STORE,
        EXPERTS as u32,
        2,
        HIDDEN as u32,
        INNER as u32,
        DType::B8,
        ExpertActivation::SquaredRelu,
        true,
        2.5,
    )
    .unwrap();
    let fp8 = |seed| -> Vec<u8> {
        values(INNER * HIDDEN, seed)
            .iter()
            .map(|&v| f32_to_e4m3_rne(v))
            .collect()
    };
    let weights: Vec<(Vec<u8>, Vec<u8>)> =
        (0..EXPERTS).map(|e| (fp8(10 + e), fp8(20 + e))).collect();
    let table: Vec<ExpertWeights<'_>> = weights
        .iter()
        .map(|(up, down)| ExpertWeights {
            up,
            down,
            up_block_scales: &[],
            down_block_scales: &[],
            up_scale: 0.25,
            down_scale: 0.125,
        })
        .collect();

    let mut b = GraphBuilder::new_in(Global);
    let hidden = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN as u32], &[])
        .unwrap();
    let scores = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, EXPERTS as u32], &[])
        .unwrap();
    let bias = b
        .tensor(DType::F32, TensorFlags::INPUT, &[EXPERTS as u32], &[])
        .unwrap();
    b.moe_experts(hidden, scores, bias, 1, params, TensorFlags::OUTPUT)
        .unwrap();
    let program = NativeProgram::from_graph_in(
        b.finish().unwrap(),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap();
    let mut arena = program.prepare_borrowing_inputs_in(Global).unwrap();
    let (x, s) = (values(HIDDEN, 3), [0.9, 0.2, 0.7, 0.4, 0.8, 0.1]);
    let inputs = [
        NativeRef::F32(&x),
        NativeRef::F32(&s),
        NativeRef::F32(&[0.0; EXPERTS]),
    ];
    let store = |_| {
        [NativeResourceBinding::expert_store(
            STORE,
            ExpertStoreRef::new(&table, params).unwrap(),
        )]
    };
    arena.run_native_on(&inputs, &store(()), &Serial).unwrap();
    let serial = bits(arena.output_f32(0).unwrap());
    let threads = Threads::default();
    arena.run_native_on(&inputs, &store(()), &threads).unwrap();
    assert!(threads.largest.into_inner() > 1, "experts were not split");
    assert_eq!(bits(arena.output_f32(0).unwrap()), serial);
}
