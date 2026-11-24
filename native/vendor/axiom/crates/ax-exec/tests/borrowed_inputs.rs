#![feature(allocator_api)]
//! An arena prepared to borrow its inputs owns no storage for them: weights
//! stay where the caller mapped them, and every run reads them in place.
use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
use ax_exec::{AxCpuKernelPolicy, Error, NativeProgram, NativeProgress, NativeRef};
use axnn_cpu::fp8::f32_to_e4m3_rne;
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global, counting the bytes it has live.
#[derive(Default)]
struct Counting(AtomicUsize);

unsafe impl Allocator for Counting {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.fetch_add(layout.size(), Ordering::Relaxed);
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { Global.deallocate(ptr, layout) }
    }
}

const HIDDEN: usize = 256;
const OUT: usize = 512;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 29 + seed * 13) % 37) as f32 - 18.0) / 9.0)
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `y = (-x · W^T)^2` over an FP8 weight input, then `x^2` for a second
/// output. `x` retires after its first use and `-x` has `x`'s layout, so an
/// ordinary arena recycles the input slot for it.
fn program() -> NativeProgram<Global> {
    let mut b = GraphBuilder::new_in(Global);
    let x = b
        .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN as u32], &[])
        .unwrap();
    let weight = b
        .tensor(
            DType::B8,
            TensorFlags::INPUT,
            &[OUT as u32, HIDDEN as u32],
            &[],
        )
        .unwrap();
    let f32_tensor = |b: &mut GraphBuilder<Global>, dims: &[u32], flags| {
        b.tensor(DType::F32, flags, dims, &[]).unwrap()
    };
    let none = NodeFlags::default();
    let negated = f32_tensor(&mut b, &[1, HIDDEN as u32], TensorFlags::default());
    b.node(OpCode::NEGATE, none, &[x], &[negated], &[]).unwrap();
    let squared = f32_tensor(&mut b, &[1, HIDDEN as u32], TensorFlags::OUTPUT);
    b.node(OpCode::MUL, none, &[negated, negated], &[squared], &[])
        .unwrap();
    let product = f32_tensor(&mut b, &[1, OUT as u32], TensorFlags::default());
    b.node(
        OpCode::MATMUL_TRANSPOSED,
        none,
        &[negated, weight],
        &[product],
        &[],
    )
    .unwrap();
    let out = f32_tensor(&mut b, &[1, OUT as u32], TensorFlags::OUTPUT);
    b.node(OpCode::MUL, none, &[product, product], &[out], &[])
        .unwrap();
    NativeProgram::from_graph_in(
        b.finish().unwrap(),
        AxCpuKernelPolicy::default(),
        true,
        Global,
    )
    .unwrap()
}

/// The unprepared invocation's outputs, for comparison.
fn transient(program: &NativeProgram<Global>, x: &[f32], weight: &[u8]) -> Vec<Vec<u8>> {
    let x = f32_bytes(x);
    let mut run = program.begin_in(&[&x, weight], Global).unwrap();
    assert_eq!(run.advance(usize::MAX).unwrap(), NativeProgress::Ready);
    run.take_retained_in()
        .unwrap()
        .iter()
        .map(|output| output.as_bytes().to_vec())
        .collect()
}

#[test]
fn a_borrowing_arena_reads_inputs_in_place_and_allocates_none() {
    let program = program();
    let (owning_heap, borrowing_heap) = (Counting::default(), Counting::default());
    let owning = program.prepare_native_in(&owning_heap).unwrap();
    let mut arena = program
        .prepare_borrowing_inputs_in(&borrowing_heap)
        .unwrap();
    // Measured, not just reported: the owning arena holds the weight, the
    // borrowing one only an extra slot for `-x`.
    let held = |heap: &Counting| heap.0.load(Ordering::Relaxed);
    assert!(held(&owning_heap) + 4 * HIDDEN > held(&borrowing_heap) + OUT * HIDDEN);
    // Only the four F32 results own storage. `-x` gets a slot of its own,
    // where the owning arena reuses `x`'s, so the saving is the weight.
    assert_eq!(arena.buffer_bytes(), 4 * (2 * HIDDEN + 2 * OUT));
    assert_eq!(owning.buffer_bytes() - arena.buffer_bytes(), OUT * HIDDEN);

    for seed in 0..3 {
        let x = values(HIDDEN, seed);
        let weight: Vec<u8> = values(OUT * HIDDEN, seed + 7)
            .iter()
            .map(|&v| f32_to_e4m3_rne(v))
            .collect();
        let expected = transient(&program, &x, &weight);
        arena
            .run_with_native_inputs_and_resources(
                &[NativeRef::F32(&x), NativeRef::Packed(DType::B8, &weight)],
                &[],
            )
            .unwrap();
        assert_eq!(f32_bytes(arena.output_f32(0).unwrap()), expected[0]);
        assert_eq!(f32_bytes(arena.output_f32(1).unwrap()), expected[1]);
    }

    // A fork borrows too.
    let mut fork = arena.fork_in(&borrowing_heap).unwrap();
    let x = values(HIDDEN, 9);
    let weight = vec![f32_to_e4m3_rne(0.5); OUT * HIDDEN];
    fork.run_with_native_inputs_and_resources(
        &[NativeRef::F32(&x), NativeRef::Packed(DType::B8, &weight)],
        &[],
    )
    .unwrap();
    assert_eq!(
        f32_bytes(fork.output_f32(1).unwrap()),
        transient(&program, &x, &weight)[1]
    );

    // Byte ingress would decode into storage the arena never allocated.
    let bytes = f32_bytes(&x);
    assert!(matches!(
        arena.run(&[&bytes, &weight]),
        Err(Error::UnsupportedImplementation { .. })
    ));
    assert!(arena.output_f32(0).is_none());
}
