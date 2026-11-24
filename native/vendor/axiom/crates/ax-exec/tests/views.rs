#![feature(allocator_api)]
//! `VIEW` aliases its input's storage: no bytes move, and every read of a
//! view counts as a read of its source.
use ax::tensor_ops::encode_view_attributes;
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};
use ax_exec::{AxCpuKernelPolicy, NativeProgram, NativeProgress, NativeRef};
use std::alloc::Global;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

struct Graph {
    b: GraphBuilder<Global>,
}

impl Graph {
    fn new() -> Self {
        Self {
            b: GraphBuilder::new_in(Global),
        }
    }
    fn tensor(&mut self, dims: &[u32], flags: TensorFlags) -> TensorId {
        self.b.tensor(DType::F32, flags, dims, &[]).unwrap()
    }
    fn constant(&mut self, dims: &[u32], values: &[f32]) -> TensorId {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.b
            .tensor(DType::F32, TensorFlags::CONSTANT, dims, &bytes)
            .unwrap()
    }
    fn node(&mut self, op: OpCode, inputs: &[TensorId], output: TensorId) {
        self.b
            .node(op, NodeFlags::default(), inputs, &[output], &[])
            .unwrap();
    }
    fn view(&mut self, input: TensorId, offset: u64, dims: &[u32], flags: TensorFlags) -> TensorId {
        let output = self.tensor(dims, flags);
        self.b
            .node(
                OpCode::VIEW,
                NodeFlags::default(),
                &[input],
                &[output],
                &encode_view_attributes(offset),
            )
            .unwrap();
        output
    }
    fn program(self) -> Result<NativeProgram<Global>, ax_exec::Error> {
        NativeProgram::from_graph_in(
            self.b.finish().unwrap(),
            AxCpuKernelPolicy::default(),
            true,
            Global,
        )
    }
}

/// Runs a single-input program both ways and checks they agree.
fn run(program: &NativeProgram<Global>, x: &[f32]) -> Vec<Vec<f32>> {
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut run = program.begin_in(&[&bytes], Global).unwrap();
    assert_eq!(run.advance(usize::MAX).unwrap(), NativeProgress::Ready);
    let unprepared: Vec<Vec<f32>> = run
        .take_retained_in()
        .unwrap()
        .iter()
        .map(|output| f32s(output.as_bytes()))
        .collect();
    let mut arena = program.prepare_native_in(Global).unwrap();
    arena
        .run_with_native_inputs_and_resources(&[NativeRef::F32(x)], &[])
        .unwrap();
    for (index, expected) in unprepared.iter().enumerate() {
        assert_eq!(arena.output_f32(index).unwrap(), expected, "output {index}");
    }
    unprepared
}

/// `a` is read only through views, after a same-shaped result is born: if
/// view reads did not keep `a` alive, that result would take its slot.
#[test]
fn a_source_read_only_through_views_stays_live() {
    let mut g = Graph::new();
    let x = g.tensor(&[1, 8], TensorFlags::INPUT);
    let a = g.tensor(&[1, 8], TensorFlags::default());
    g.node(OpCode::MUL, &[x, x], a);
    let negated = g.tensor(&[1, 8], TensorFlags::OUTPUT);
    g.node(OpCode::NEGATE, &[x], negated);
    let grid = g.view(a, 0, &[2, 4], TensorFlags::default());
    let offset = g.constant(&[2, 4], &[100.0; 8]);
    let shifted = g.tensor(&[2, 4], TensorFlags::OUTPUT);
    g.node(OpCode::ADD, &[grid, offset], shifted);
    let tail = g.view(a, 4, &[1, 4], TensorFlags::default());
    let squared = g.tensor(&[1, 4], TensorFlags::OUTPUT);
    g.node(OpCode::MUL, &[tail, tail], squared);

    let x: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let outputs = run(&g.program().unwrap(), &x);
    assert_eq!(outputs[0], x.iter().map(|v| -v).collect::<Vec<_>>());
    assert_eq!(
        outputs[1],
        x.iter().map(|v| v * v + 100.0).collect::<Vec<_>>()
    );
    assert_eq!(
        outputs[2],
        x[4..].iter().map(|v| (v * v) * (v * v)).collect::<Vec<_>>()
    );
}

/// A view of a view composes offsets, and a view of a graph input reads the
/// caller's borrowed storage.
#[test]
fn views_compose_and_read_borrowed_inputs() {
    let mut g = Graph::new();
    let x = g.tensor(&[1, 12], TensorFlags::INPUT);
    let middle = g.view(x, 2, &[1, 8], TensorFlags::default());
    let inner = g.view(middle, 4, &[2, 2], TensorFlags::default());
    let zero = g.constant(&[2, 2], &[0.0; 4]);
    let out = g.tensor(&[2, 2], TensorFlags::OUTPUT);
    g.node(OpCode::ADD, &[inner, zero], out);
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
    assert_eq!(run(&g.program().unwrap(), &x)[0], [6.0, 7.0, 8.0, 9.0]);
}

/// An ADD reading a view of a product is never fused with that MUL: fused
/// instructions carry whole slots, not ranges.
#[test]
fn nodes_reading_views_are_not_fused() {
    let mut g = Graph::new();
    let x = g.tensor(&[1, 4], TensorFlags::INPUT);
    let product = g.tensor(&[1, 4], TensorFlags::default());
    g.node(OpCode::MUL, &[x, x], product);
    let square = g.view(product, 0, &[2, 2], TensorFlags::default());
    let offset = g.constant(&[2, 2], &[0.5; 4]);
    let out = g.tensor(&[2, 2], TensorFlags::OUTPUT);
    g.node(OpCode::ADD, &[square, offset], out);
    let program = g.program().unwrap();
    assert_eq!(program.native_plan_stats().mul_add_fusions, 0);
    assert_eq!(
        run(&program, &[1.0, 2.0, 3.0, 4.0])[0],
        [1.5, 4.5, 9.5, 16.5]
    );
}

#[test]
fn exported_or_conflicting_views_are_refused() {
    // A graph output needs its own storage.
    let mut g = Graph::new();
    let x = g.tensor(&[1, 4], TensorFlags::INPUT);
    let doubled = g.tensor(&[1, 4], TensorFlags::default());
    g.node(OpCode::ADD, &[x, x], doubled);
    g.view(doubled, 0, &[2, 2], TensorFlags::OUTPUT);
    assert!(g.program().is_err());

    // One node reading two different ranges of one source.
    let mut g = Graph::new();
    let x = g.tensor(&[1, 8], TensorFlags::INPUT);
    let doubled = g.tensor(&[1, 8], TensorFlags::default());
    g.node(OpCode::ADD, &[x, x], doubled);
    let head = g.view(doubled, 0, &[1, 4], TensorFlags::default());
    let tail = g.view(doubled, 4, &[1, 4], TensorFlags::default());
    let sum = g.tensor(&[1, 4], TensorFlags::OUTPUT);
    g.node(OpCode::ADD, &[head, tail], sum);
    assert!(g.program().is_err());
}
