use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use axnn::lowering::{KernelAccuracy, KernelPolicy};
use axnn_cpu::CpuCapabilities;
use kore_axiom::program::{Input, Program};
use kore_rt::{Heap, heap::StaticMemSource};

#[repr(align(4096))]
struct Backing([u8; 256 * 1024]);

fn policy() -> KernelPolicy {
    KernelPolicy::new(CpuCapabilities::compiled().mask())
        .with_minimum_accuracy(KernelAccuracy::Faithful)
}

#[test]
fn selected_math_masks_and_broadcasts_execute_without_a_host_runtime() {
    let mut backing = Backing([0; 256 * 1024]);
    // SAFETY: backing outlives heap and every graph/program/value; no other
    // accessor uses it until the named heap and its allocations are dropped.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut graph = GraphBuilder::new_in(&heap);
    let x = graph
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 4], &[])
        .unwrap();
    let threshold = graph
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, 1],
            &4f32.to_le_bytes(),
        )
        .unwrap();
    let square = graph
        .tensor(DType::F32, TensorFlags::default(), &[1, 4], &[])
        .unwrap();
    let mask = graph
        .tensor(DType::U8, TensorFlags::default(), &[1, 4], &[])
        .unwrap();
    let root = graph
        .tensor(DType::F32, TensorFlags::default(), &[1, 4], &[])
        .unwrap();
    let output = graph
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 4], &[])
        .unwrap();
    graph
        .node(OpCode::MUL, NodeFlags::default(), &[x, x], &[square], &[])
        .unwrap();
    graph
        .node(
            OpCode::GREATER,
            NodeFlags::default(),
            &[square, threshold],
            &[mask],
            &[],
        )
        .unwrap();
    graph
        .node(OpCode::SQRT, NodeFlags::default(), &[square], &[root], &[])
        .unwrap();
    graph
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[mask, root, threshold],
            &[output],
            &[],
        )
        .unwrap();
    let graph = graph.finish().unwrap();
    let program = Program::compile_in(graph.as_bytes(), policy(), &heap).unwrap();
    drop(graph); // executable owns its topology, constants, and selected plans
    assert_eq!(program.placement().nodes, 4);
    assert_eq!(program.placement().partitions, 1);
    assert_eq!(program.kernel_plans().count(), 4);
    let result = program
        .run_in(&[Input::F32(&[-3., -2., 0., 5.])], &heap)
        .unwrap();
    assert_eq!(result[0].f32().unwrap(), &[3., 4., 4., 5.]);
    let result = program
        .run_in(&[Input::F32(&[1., 2., 3., 4.])], &heap)
        .unwrap();
    assert_eq!(result[0].f32().unwrap(), &[4., 4., 3., 4.]);
    assert!(matches!(
        program.run_in(&[Input::F32(&[1.])], &heap),
        Err(kore_axiom::Error::InvalidBinding)
    ));
    assert!(matches!(
        program.run_in(&[Input::U8(&[0; 4])], &heap),
        Err(kore_axiom::Error::InvalidBinding)
    ));
    assert!(matches!(
        program.run_in(&[], &heap),
        Err(kore_axiom::Error::InvalidBinding)
    ));
}

#[test]
fn constant_and_input_outputs_do_not_need_a_producer_node() {
    let mut backing = Backing([0; 256 * 1024]);
    // SAFETY: all heap users are local and drop before the backing.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut graph = GraphBuilder::new_in(&heap);
    graph
        .tensor(
            DType::F32,
            TensorFlags(TensorFlags::OUTPUT.0 | TensorFlags::CONSTANT.0),
            &[1],
            &7f32.to_le_bytes(),
        )
        .unwrap();
    graph
        .tensor(
            DType::U8,
            TensorFlags(TensorFlags::OUTPUT.0 | TensorFlags::INPUT.0),
            &[2],
            &[],
        )
        .unwrap();
    graph
        .tensor(
            DType::I32,
            TensorFlags(TensorFlags::OUTPUT.0 | TensorFlags::INPUT.0),
            &[2],
            &[],
        )
        .unwrap();
    let graph = graph.finish().unwrap();
    let program = Program::compile_in(graph.as_bytes(), policy(), &heap).unwrap();
    let integers = [0x0012_3456, 0x0078_9abc];
    let result = program
        .run_in(&[Input::U8(&[0, 7]), Input::I32(&integers)], &heap)
        .unwrap();
    assert_eq!(result[0].f32().unwrap(), &[7.]);
    match result[1].as_input() {
        Input::U8(v) => assert_eq!(v, &[0, 7]),
        _ => panic!("dtype changed"),
    }
    assert_eq!(result[2].i32().unwrap(), &integers);
    assert!(matches!(result[2].as_input(), Input::I32(v) if v == integers));
    // The same producer-free topology also works through the shared core's
    // explicit arena path; retaining it must not invent a producer partition.
    let native = ax_exec::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap).unwrap();
    assert_eq!(native.schedule().plan().partitions.len(), 0);
    let mut prepared = native.prepare_native_in(&heap).unwrap();
    prepared
        .run(&[&[0, 7], integers.map(i32::to_le_bytes).as_flattened()])
        .unwrap();
    assert_eq!(prepared.output_f32(0), Some(&[7.0][..]));
    assert_eq!(prepared.output_u8(1), Some(&[0, 7][..]));
    assert_eq!(prepared.output_i32(2), Some(&integers[..]));
}

#[test]
fn unsupported_semantics_refuse_during_admission() {
    let mut backing = Backing([0; 256 * 1024]);
    // SAFETY: all heap users are local and drop before the backing.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut graph = GraphBuilder::new_in(&heap);
    let input = graph
        .tensor(DType::F16, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let output = graph
        .tensor(DType::F16, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    graph
        .node(OpCode::SQRT, NodeFlags::default(), &[input], &[output], &[])
        .unwrap();
    let graph = graph.finish().unwrap();
    assert!(Program::compile_in(graph.as_bytes(), policy(), &heap).is_err());
}

#[test]
fn unconsumed_node_outputs_do_not_accumulate_in_the_transient_arena() {
    let mut backing = Backing([0; 256 * 1024]);
    // SAFETY: all persistent allocations stay within this backing's lifetime.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut graph = GraphBuilder::new_in(&heap);
    let input = graph
        .tensor(
            DType::F32,
            TensorFlags(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0),
            &[512],
            &[],
        )
        .unwrap();
    for _ in 0..48 {
        let dead = graph
            .tensor(DType::F32, TensorFlags::default(), &[512], &[])
            .unwrap();
        graph
            .node(
                OpCode::MUL,
                NodeFlags::default(),
                &[input, input],
                &[dead],
                &[],
            )
            .unwrap();
    }
    let program = Program::from_graph_in(graph.finish().unwrap(), policy(), &heap).unwrap();
    #[repr(align(4096))]
    // One 64 KiB Kore heap extent, smaller than the 96 KiB of dead planes.
    struct Transient([u8; 64 * 1024]);
    let mut storage = Transient([0; 64 * 1024]);
    // SAFETY: only the transient heap borrows storage, and both invocation
    // outputs and the heap drop before storage does.
    let transient =
        Heap::new_in(unsafe { StaticMemSource::over(storage.0.as_mut_ptr(), storage.0.len()) });
    let values = [2.0; 512];
    let output = program.run_in(&[Input::F32(&values)], &transient).unwrap();
    assert_eq!(output[0].f32().unwrap(), &values);
    assert!(transient.stats().mapped_bytes <= 64 * 1024);
}

#[test]
fn adopted_graph_automatically_fuses_without_changing_rounding_or_admission() {
    let mut backing = Backing([0; 256 * 1024]);
    // SAFETY: the named heap exclusively owns this backing until all values drop.
    let heap =
        Heap::new_in(unsafe { StaticMemSource::over(backing.0.as_mut_ptr(), backing.0.len()) });
    let mut graph = GraphBuilder::new_in(&heap);
    let x = graph
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let scale = graph
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let offset = graph
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let product = graph
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let output = graph
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    graph
        .node(
            OpCode::MUL,
            NodeFlags::default(),
            &[x, scale],
            &[product],
            &[],
        )
        .unwrap();
    graph
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[product, offset],
            &[output],
            &[],
        )
        .unwrap();
    let graph = graph.finish().unwrap();
    let encoded = graph.as_bytes().as_ptr();
    let admitted = KernelPolicy::new(CpuCapabilities::SCALAR.mask())
        .with_minimum_accuracy(KernelAccuracy::Strict)
        .with_max_temporary_bytes(0)
        .with_temporary_byte_cost(17);
    let program = Program::from_graph_in(graph, admitted, &heap).unwrap();
    assert_eq!(program.as_bytes().as_ptr(), encoded);
    assert_eq!(program.native_plan_stats().mul_add_fusions, 1);
    assert_eq!(program.native_plan_stats().instructions, 1);
    assert!(
        program
            .kernel_plans()
            .all(|p| p.implementation == axnn_cpu::CpuImplementation::Scalar
                && p.accuracy == KernelAccuracy::Strict
                && p.temporary_bytes == 0)
    );
    let x = [f32::from_bits(0x3f80_0001), -0.0, 2.0, 8.0];
    let scale = [f32::from_bits(0x3f7f_fffe)];
    let offset = [-1.0];
    let result = program
        .run_in(
            &[Input::F32(&x), Input::F32(&scale), Input::F32(&offset)],
            &heap,
        )
        .unwrap();
    for (actual, x) in result[0].f32().unwrap().iter().zip(x) {
        // The shared fusion is a single traversal, not an FMA contraction.
        let product = x * scale[0];
        assert_eq!(actual.to_bits(), (product + offset[0]).to_bits());
    }
    // Equal byte length is insufficient: the typed binding must also match.
    assert!(matches!(
        program.run_in(
            &[Input::F32(&x), Input::U8(&[0; 4]), Input::F32(&offset)],
            &heap,
        ),
        Err(kore_axiom::Error::InvalidBinding)
    ));
}
