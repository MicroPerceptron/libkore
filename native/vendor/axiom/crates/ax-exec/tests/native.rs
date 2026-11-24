#![feature(allocator_api)]
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, AxOutput, Error, NativeProgram, NativeProgress, NativeTensor};
use std::alloc::Global;

fn graph(passthrough: bool) -> ax::GraphBuf<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let x = b
        .tensor(
            DType::F32,
            if passthrough {
                TensorFlags(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0)
            } else {
                TensorFlags::INPUT
            },
            &[4],
            &[],
        )
        .unwrap();
    let scale = b.tensor(DType::F32, TensorFlags::INPUT, &[1], &[]).unwrap();
    let offset = b.tensor(DType::F32, TensorFlags::INPUT, &[1], &[]).unwrap();
    let product = b
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let output = b
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    b.node(
        OpCode::MUL,
        NodeFlags::default(),
        &[x, scale],
        &[product],
        &[],
    )
    .unwrap();
    b.node(
        OpCode::ADD,
        NodeFlags::default(),
        &[product, offset],
        &[output],
        &[],
    )
    .unwrap();
    b.finish().unwrap()
}

#[test]
fn automatic_fusion_retains_passthrough_and_exact_results() {
    let graph = graph(true);
    let graph_pointer = graph.as_bytes().as_ptr();
    let program =
        NativeProgram::from_graph_in(graph, AxCpuKernelPolicy::default(), true, Global).unwrap();
    assert_eq!(program.as_bytes().as_ptr(), graph_pointer);
    assert_eq!(program.native_plan_stats().mul_add_fusions, 1);
    assert_eq!(
        program.schedule().plan().partitions[0].implementation,
        ax_exec::CPU_KERNEL_IMPLEMENTATION
    );
    let x = [1.0_f32, -0.0, 2.0, 8.0];
    // SAFETY: initialized padding-free f32 words, immutably borrowed for the run.
    let bytes = unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), 16) };
    let scale = 2.0_f32.to_le_bytes();
    let offset = 3.0_f32.to_le_bytes();
    let mut run = program.begin_in(&[bytes, &scale, &offset], Global).unwrap();
    assert_eq!(run.advance(1).unwrap(), NativeProgress::Ready);
    let mut output = run.take_retained_in().unwrap();
    assert_eq!(output[0].as_bytes().as_ptr(), bytes.as_ptr());
    let expected: Vec<u8> = [5.0_f32, 3.0, 7.0, 19.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(output[1].as_bytes(), expected);
    assert_eq!(output[1].dtype(), DType::F32);
    let pointer = output[1].as_bytes().as_ptr();
    let NativeTensor::F32(values) = output.pop().unwrap().into_native_in(Global).unwrap() else {
        panic!("typed ownership changed dtype");
    };
    assert_eq!(values.as_ptr().cast::<u8>(), pointer);
    assert_eq!(&values, &[5.0_f32, 3.0, 7.0, 19.0]);
    assert_eq!(run.host_copy_stats().output_materialization_bytes, 0);
    assert!(matches!(run.advance(1), Err(Error::InvocationFinished)));
}

#[test]
fn bounded_progress_never_publishes_partial_output_and_cancellation_is_terminal() {
    let graph = graph(false);
    let program = NativeProgram::compile_in(
        &graph.parse().unwrap(),
        AxCpuKernelPolicy::default(),
        false,
        Global,
    )
    .unwrap();
    let x: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let scale = 2.0_f32.to_le_bytes();
    let offset = 3.0_f32.to_le_bytes();
    let mut run = program.begin_in(&[&x, &scale, &offset], Global).unwrap();
    let mut backing = [0xa5; 64];
    // Bad geometry does not consume the first instruction.
    assert!(
        run.advance_into(1, &mut [AxOutput::packed(&mut backing[..15])])
            .is_err()
    );
    assert_eq!(
        run.advance_into(
            1,
            &mut [AxOutput::pitched(&mut backing, 8, 8, 2, 24).unwrap()]
        )
        .unwrap(),
        NativeProgress::Progress { instructions: 1 }
    );
    assert_eq!(backing, [0xa5; 64]);
    assert_eq!(
        run.advance_into(
            1,
            &mut [AxOutput::pitched(&mut backing, 8, 8, 2, 24).unwrap()]
        )
        .unwrap(),
        NativeProgress::Ready
    );
    for (at, value) in [(8, 5.0_f32), (12, 7.0), (32, 9.0), (36, 11.0)] {
        assert_eq!(&backing[at..at + 4], &value.to_le_bytes());
    }
    assert!(
        backing[..8]
            .iter()
            .chain(&backing[16..32])
            .chain(&backing[40..])
            .all(|&v| v == 0xa5)
    );
    assert_eq!(run.host_copy_stats().output_materialization_bytes, 16);
    let mut cancelled = program.begin_in(&[&x, &scale, &offset], Global).unwrap();
    assert!(matches!(
        cancelled.advance(0),
        Err(Error::InvalidStepBudget)
    ));
    cancelled.advance(1).unwrap();
    cancelled.cancel().unwrap();
    assert!(matches!(
        cancelled.take_retained_in(),
        Err(Error::InvocationFinished)
    ));
}

#[test]
fn reduction_stays_inside_the_native_partition_and_returns_one_scalar() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[8], &[])
        .unwrap();
    let scale = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[], &[])
        .unwrap();
    let product = builder
        .tensor(DType::F32, TensorFlags::default(), &[8], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[], &[])
        .unwrap();
    builder
        .node(
            OpCode::MUL,
            NodeFlags::default(),
            &[input, scale],
            &[product],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::REDUCE_SUM,
            NodeFlags::default(),
            &[product],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let program =
        NativeProgram::from_graph_in(graph, AxCpuKernelPolicy::default(), true, Global).unwrap();

    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // SAFETY: initialized padding-free f32 words, immutably borrowed for the run.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    let scale = 2.0_f32.to_le_bytes();
    let mut run = program.begin_in(&[bytes, &scale], Global).unwrap();
    assert_eq!(run.advance(2).unwrap(), NativeProgress::Ready);
    let output = run.take_retained_in().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].as_bytes(), 72.0_f32.to_le_bytes());
    assert_eq!(run.host_copy_stats().output_materialization_bytes, 0);
}

/// The MoE phases ride with the profile: merged, subtracted, and printed
/// after the `moe` class with the uncovered remainder as its device wait.
#[test]
fn op_profile_carries_moe_phases() {
    let mut before = ax_exec::OpProfile::default();
    before.add(0, 2_000_000);
    before.moe.ns = [1_000_000, 0, 0, 0];
    let mut now = before;
    now.add(0, 10_000_000);
    let mut turn = ax_exec::OpProfile::default();
    turn.moe.ns = [1_000_000, 2_000_000, 1_000_000, 3_000_000];
    now.merge(&turn);
    let delta = now.since(&before);
    assert_eq!(delta.ns[0], 10_000_000);
    assert_eq!(delta.moe.ns, turn.moe.ns);
    assert_eq!(
        std::format!("{delta}"),
        "moe 10 ms/1 (route 1 load 2 device 1 native 3 wait 3 ms); CPU direct [none]; CPU fallback [none]; CPU reference [none]"
    );
    assert_eq!(
        std::format!("{}", ax_exec::OpProfile::default()),
        "none; CPU direct [none]; CPU fallback [none]; CPU reference [none]"
    );
}

#[test]
fn placement_profiles_merge_and_subtract_without_losing_mixed_expert_work() {
    let mut sample = ax_exec::OpProfile::default();
    sample.cpu.runs[9] = 3;
    sample.cpu.ns[9] = 12_000;
    sample.fallback.runs[1] = 1;
    sample.fallback.ns[1] = 20_000;
    sample.reference.runs[0] = 2;
    sample.reference.ns[0] = 1_000;
    sample.moe.delegated_waves = 2;
    sample.moe.delegated_routes = 8;
    sample.moe.native_waves = 1;
    sample.moe.native_routes = 4;
    sample.moe.mixed_invocations = 1;
    let mut merged = sample;
    merged.merge(&sample);
    assert_eq!(merged.since(&sample), sample);
    assert_eq!(sample.since(&merged), ax_exec::OpProfile::default());
    let text = std::format!("{sample}");
    assert!(text.contains("CPU direct [other 12 us/3]"));
    assert!(text.contains("CPU fallback [nvfp4 20 us/1]"));
    assert!(text.contains("CPU reference [moe 1 us/2]"));
    assert!(
        text.contains(
            "expert placement: delegated 2/8 waves/routes, CPU 1/4 waves/routes, mixed 1"
        )
    );
}
