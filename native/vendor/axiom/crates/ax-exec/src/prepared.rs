//! Stable arena facade; implementation records stay behind the native adapter module.
use crate::{Error, NativeRef, NativeResourceBinding};
use core::alloc::Allocator;

/// Reusable native arena. Preparation is explicit; ordinary invocations retain
/// borrowed ingress and typed results without silently choosing this arena.
pub struct PreparedNative<A: Allocator> {
    pub(crate) inner: crate::native::PreparedNative<A>,
}
impl<A: Allocator> PreparedNative<A> {
    /// Share immutable instructions while allocating independent tensor slots,
    /// constants and scratch from `allocator`. The fork can run concurrently
    /// with, and outlive, its source. No previous result is published in a fork.
    /// Both allocators have the same type but may name different bounded heaps.
    pub fn fork_in(&self, allocator: A) -> Result<Self, Error>
    where
        A: Clone,
    {
        Ok(Self {
            inner: self.inner.fork_in(allocator)?,
        })
    }

    /// Decode byte inputs into the arena and run. An arena prepared to borrow
    /// its inputs has no storage to decode into and refuses.
    pub fn run(&mut self, inputs: &[&[u8]]) -> Result<(), Error> {
        self.inner.run(inputs)
    }
    pub fn run_with_resources(
        &mut self,
        inputs: &[&[u8]],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), Error> {
        self.inner.run_with_resources(inputs, resources)
    }
    /// Execute over call-scoped, already typed inputs. The arena continues to
    /// own outputs and scratch; ingress remains borrowed only until return.
    pub fn run_with_native_inputs_and_resources(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), Error> {
        self.inner
            .run_with_native_inputs_and_resources(inputs, resources)
    }
    /// [`Self::run_with_native_inputs_and_resources`], splitting large
    /// kernels (float-weight products, expert projections) across the
    /// caller's workers. Every output element is computed as in a serial
    /// run, so results are identical at any width.
    pub fn run_native_on(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
        parallel: &dyn crate::ForkJoin,
    ) -> Result<(), Error> {
        self.inner.run_native_on(inputs, resources, parallel)
    }

    /// Time every instruction from now on by kernel class — see
    /// [`OpProfile`](crate::native::OpProfile).
    pub fn set_profiler(&mut self, clock: fn() -> u64) {
        self.inner.set_profiler(clock);
    }

    /// The kernel-class tally so far; zero without a profiler.
    pub fn profile(&self) -> crate::native::OpProfile {
        self.inner.profile()
    }

    /// Recompute delegated results natively and compare — see
    /// [`CheckReport`](crate::native::CheckReport). The native results stand.
    pub fn set_delegate_check(&mut self, enabled: bool) {
        self.inner.set_delegate_check(enabled);
    }

    /// The comparison so far; empty unless checking.
    pub fn delegate_check(&self) -> crate::native::CheckReport {
        self.inner.delegate_check()
    }
    /// Run with a host projection dispatcher. All other instructions and
    /// recurrent resources continue through the native executor.
    pub fn run_native_with_projections(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
        parallel: &dyn crate::ForkJoin,
        projections: Option<&mut dyn crate::ProjectionExecutor>,
    ) -> Result<(), Error> {
        self.inner
            .run_native_with_projections(inputs, resources, parallel, projections)
    }
    pub fn output_f32(&self, index: usize) -> Option<&[f32]> {
        self.inner.output_f32(index)
    }
    /// Compact increasing selected rows of a completed F32 output into its
    /// existing allocation, retaining the original shape and capture geometry.
    pub fn compact_output_f32_rows(
        &mut self,
        index: usize,
        rows: usize,
        path: &[u32],
    ) -> Option<()> {
        self.inner.compact_output_f32_rows(index, rows, path)
    }
    pub fn output_i32(&self, index: usize) -> Option<&[i32]> {
        self.inner.output_i32(index)
    }
    pub fn output_u8(&self, index: usize) -> Option<&[u8]> {
        self.inner.output_u8(index)
    }
    pub fn input_count(&self) -> usize {
        self.inner.input_count()
    }
    pub fn output_count(&self) -> usize {
        self.inner.output_count()
    }
    pub fn instruction_count(&self) -> usize {
        self.inner.instruction_count()
    }
    pub fn sub_mul_add_fusion_count(&self) -> usize {
        self.inner
            .nodes
            .iter()
            .filter(|node| matches!(node, crate::native::PreparedNativeInstruction::SubMulAdd(_)))
            .count()
    }
    pub fn mul_add_fusion_count(&self) -> usize {
        self.inner.mul_add_fusion_count()
    }
    pub fn simd_mul_add_fusion_count(&self) -> usize {
        self.inner.simd_mul_add_fusion_count()
    }
    pub fn scale_interleave3_fusion_count(&self) -> usize {
        self.inner.scale_interleave3_fusion_count()
    }
    pub fn simd_scale_interleave3_fusion_count(&self) -> usize {
        self.inner.simd_scale_interleave3_fusion_count()
    }
    pub fn cast_shift_or3_fusion_count(&self) -> usize {
        self.inner.cast_shift_or3_fusion_count()
    }
    pub fn buffer_count(&self) -> usize {
        self.inner.buffer_count()
    }
    pub fn buffer_bytes(&self) -> usize {
        self.inner.buffer_bytes()
    }
    pub fn logical_buffer_count(&self) -> usize {
        self.inner.logical_buffer_count()
    }
    pub fn logical_buffer_bytes(&self) -> usize {
        self.inner.logical_buffer_bytes()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.inner.scratch_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax::{DType, GraphBuilder, NodeFlags, OpCode, TensorFlags};
    use core::{
        alloc::{AllocError, Layout},
        cell::UnsafeCell,
        ptr::NonNull,
        sync::atomic::{AtomicUsize, Ordering},
    };

    /// A named, bounded test heap; no graph/arena storage uses Global. This
    /// monotonic test allocator only releases its backing at the end of a test.
    #[repr(align(4096))]
    struct ForkTestHeap {
        memory: UnsafeCell<[u8; 128 * 1024]>,
        cursor: AtomicUsize,
    }

    impl ForkTestHeap {
        fn new() -> Self {
            Self {
                memory: UnsafeCell::new([0; 128 * 1024]),
                cursor: AtomicUsize::new(0),
            }
        }
    }

    // SAFETY: atomic reservation returns disjoint extents; no allocator method
    // reads or writes their payload, and callers retain the backing borrow.
    unsafe impl Sync for ForkTestHeap {}
    // SAFETY: each extent is correctly aligned, within the live backing, and
    // never reused. Allocation lifetimes cannot exceed the borrowed heap.
    unsafe impl Allocator for &ForkTestHeap {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let base = self.memory.get().cast::<u8>();
            let mut cursor = self.cursor.load(Ordering::Relaxed);
            loop {
                let address = (base as usize)
                    .checked_add(cursor)
                    .and_then(|p| p.checked_add(layout.align() - 1))
                    .ok_or(AllocError)?
                    & !(layout.align() - 1);
                let offset = address.checked_sub(base as usize).ok_or(AllocError)?;
                let end = offset.checked_add(layout.size()).ok_or(AllocError)?;
                if end > 128 * 1024 {
                    return Err(AllocError);
                }
                match self.cursor.compare_exchange_weak(
                    cursor,
                    end,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: the reserved extent is within our live backing.
                        let pointer = unsafe { NonNull::new_unchecked(base.add(offset)) };
                        return Ok(NonNull::slice_from_raw_parts(pointer, layout.size()));
                    }
                    Err(actual) => cursor = actual,
                }
            }
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    fn template(heap: &ForkTestHeap) -> PreparedNative<&ForkTestHeap> {
        let mut graph = GraphBuilder::new_in(heap);
        let x = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let scale = graph
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1],
                &2.0_f32.to_le_bytes(),
            )
            .unwrap();
        let bias = graph
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1],
                &3.0_f32.to_le_bytes(),
            )
            .unwrap();
        let product = graph
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let result = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        let passthrough = TensorFlags(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0);
        graph.tensor(DType::I32, passthrough, &[4], &[]).unwrap();
        graph.tensor(DType::U8, passthrough, &[4], &[]).unwrap();
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
                &[product, bias],
                &[result],
                &[],
            )
            .unwrap();
        let graph = graph.finish().unwrap();
        crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), heap)
            .unwrap()
            .prepare_native_in(heap)
            .unwrap()
    }

    #[test]
    fn forks_share_instructions_but_own_slots_and_survive_their_source() {
        let plan_heap = ForkTestHeap::new();
        let worker_heap = ForkTestHeap::new();
        let mut original = template(&plan_heap);
        let x = [1.0_f32, 2.0, 3.0, 4.0].map(f32::to_le_bytes);
        let integers = [11_i32, 12, 13, 14].map(i32::to_le_bytes);
        let bytes = [21_u8, 22, 23, 24];
        original
            .run(&[x.as_flattened(), integers.as_flattened(), &bytes])
            .unwrap();
        let mut fork = original.fork_in(&worker_heap).unwrap();
        assert_eq!(original.inner.nodes.as_ptr(), fork.inner.nodes.as_ptr());
        assert_eq!(
            original.inner.borrowed_input_remaining,
            fork.inner.borrowed_input_remaining
        );
        assert_eq!(
            original.inner.borrowed_input_by_slot,
            fork.inner.borrowed_input_by_slot
        );
        assert!(fork.output_f32(0).is_none());
        for (source, copy) in original.inner.values.iter().zip(&fork.inner.values) {
            use crate::native::NativeValue;
            match (source, copy) {
                (Some(NativeValue::F32(a)), Some(NativeValue::F32(b))) => {
                    assert_eq!(a.as_slice(), b.as_slice());
                    assert_ne!(a.as_ptr(), b.as_ptr());
                }
                (Some(NativeValue::I32(a)), Some(NativeValue::I32(b))) => {
                    assert_eq!(a.as_slice(), b.as_slice());
                    assert_ne!(a.as_ptr(), b.as_ptr());
                }
                (Some(NativeValue::U8(a)), Some(NativeValue::U8(b))) => {
                    assert_eq!(a.as_slice(), b.as_slice());
                    assert_ne!(a.as_ptr(), b.as_ptr());
                }
                _ => panic!("a fork changed a prepared slot's storage"),
            }
        }
        drop(original);
        fork.run(&[x.as_flattened(), integers.as_flattened(), &bytes])
            .unwrap();
        assert_eq!(fork.output_f32(0), Some([5.0, 7.0, 9.0, 11.0].as_slice()));
        assert_eq!(fork.output_i32(1), Some([11, 12, 13, 14].as_slice()));
        assert_eq!(fork.output_u8(2), Some(bytes.as_slice()));
    }

    #[test]
    fn typed_borrowed_inputs_match_owned_ingress_across_slot_reuse() {
        let owned_heap = ForkTestHeap::new();
        let borrowed_heap = ForkTestHeap::new();
        let mut owned = template(&owned_heap);
        let mut borrowed = template(&borrowed_heap);
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let x_bytes = x.map(f32::to_le_bytes);
        let integers = [11_i32, 12, 13, 14];
        let integer_bytes = integers.map(i32::to_le_bytes);
        let bytes = [21_u8, 22, 23, 24];
        owned
            .run(&[x_bytes.as_flattened(), integer_bytes.as_flattened(), &bytes])
            .unwrap();
        borrowed
            .run_with_native_inputs_and_resources(
                &[
                    NativeRef::F32(&x),
                    NativeRef::I32(&integers),
                    NativeRef::U8(&bytes),
                ],
                &[],
            )
            .unwrap();
        assert_eq!(borrowed.output_f32(0), owned.output_f32(0));
        assert_eq!(borrowed.output_i32(1), owned.output_i32(1));
        assert_eq!(borrowed.output_u8(2), owned.output_u8(2));

        assert!(
            borrowed
                .run_with_native_inputs_and_resources(
                    &[
                        NativeRef::I32(&integers),
                        NativeRef::I32(&integers),
                        NativeRef::U8(&bytes),
                    ],
                    &[],
                )
                .is_err()
        );
        assert!(borrowed.output_f32(0).is_none());
    }

    #[test]
    fn borrowed_ingress_retires_before_a_recycled_slot_is_read() {
        let heap = ForkTestHeap::new();
        let mut graph = GraphBuilder::new_in(&heap);
        let x = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let y = graph
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let z = graph
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let out = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        graph
            .node(OpCode::ADD, NodeFlags::default(), &[x, x], &[y], &[])
            .unwrap();
        graph
            .node(OpCode::NEGATE, NodeFlags::default(), &[y], &[z], &[])
            .unwrap();
        graph
            .node(OpCode::ABS, NodeFlags::default(), &[z], &[out], &[])
            .unwrap();
        let graph = graph.finish().unwrap();
        let mut prepared = crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap)
            .unwrap()
            .prepare_native_in(&heap)
            .unwrap();
        let input_slot = prepared.inner.inputs[0].slot as u32;
        assert!(
            prepared
                .inner
                .nodes
                .iter()
                .any(|node| node.output_slots().contains(&input_slot))
        );
        for tick in 1..4 {
            let x = [tick as f32, -2.0, 3.0, -4.0];
            let bytes = x.map(f32::to_le_bytes);
            let expected = x.map(|v| (v + v).abs());
            prepared.run(&[bytes.as_flattened()]).unwrap();
            assert_eq!(prepared.output_f32(0), Some(expected.as_slice()));
            prepared
                .run_with_native_inputs_and_resources(&[NativeRef::F32(&x)], &[])
                .unwrap();
            assert_eq!(prepared.output_f32(0), Some(expected.as_slice()));
            assert!(
                prepared
                    .run_with_native_inputs_and_resources(&[], &[])
                    .is_err()
            );
            assert!(prepared.output_f32(0).is_none());
        }
    }

    #[test]
    fn boundary_slots_preserve_abi_order_for_unused_and_passthrough_values() {
        let heap = ForkTestHeap::new();
        let mut graph = GraphBuilder::new_in(&heap);
        let constant = [3.0_f32; 4].map(f32::to_le_bytes);
        graph
            .tensor(
                DType::F32,
                TensorFlags(TensorFlags::CONSTANT.0 | TensorFlags::OUTPUT.0),
                &[4],
                constant.as_flattened(),
            )
            .unwrap();
        graph
            .tensor(
                DType::F32,
                TensorFlags(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0),
                &[4],
                &[],
            )
            .unwrap();
        graph
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let x = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let out = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        graph
            .node(OpCode::ADD, NodeFlags::default(), &[x, x], &[out], &[])
            .unwrap();
        let graph = graph.finish().unwrap();
        let mut prepared = crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap)
            .unwrap()
            .prepare_native_in(&heap)
            .unwrap();
        assert_eq!(prepared.inner.borrowed_input_takes.as_slice(), &[0, 0, 2]);
        let passthrough = [7.0_f32; 4];
        let unused = [99.0_f32; 4];
        let x = [-1.0_f32, 2.0, -3.0, 4.0];
        let input_bytes = [passthrough, unused, x].map(|values| values.map(f32::to_le_bytes));
        for borrowed in [false, true, false, true] {
            if borrowed {
                prepared
                    .run_with_native_inputs_and_resources(
                        &[
                            NativeRef::F32(&passthrough),
                            NativeRef::F32(&unused),
                            NativeRef::F32(&x),
                        ],
                        &[],
                    )
                    .unwrap();
            } else {
                prepared
                    .run(&input_bytes.each_ref().map(|bytes| bytes.as_flattened()))
                    .unwrap();
            }
            assert_eq!(prepared.output_f32(0), Some([3.0; 4].as_slice()));
            assert_eq!(prepared.output_f32(1), Some(passthrough.as_slice()));
            assert_eq!(
                prepared.output_f32(2),
                Some([-2.0, 4.0, -6.0, 8.0].as_slice())
            );
        }
    }

    #[test]
    fn sub_mul_add_matches_unfused_for_every_scalar_dense_pattern_and_policy() {
        for vector in [false, true] {
            for mask in 0..16 {
                let heap = ForkTestHeap::new();
                let mut b = GraphBuilder::new_in(&heap);
                let lengths =
                    core::array::from_fn::<_, 4, _>(
                        |i| if mask & (1 << i) == 0 { 1_u32 } else { 19 },
                    );
                let inputs =
                    lengths.map(|n| b.tensor(DType::F32, TensorFlags::INPUT, &[n], &[]).unwrap());
                let difference = b
                    .tensor(
                        DType::F32,
                        TensorFlags::default(),
                        &[lengths[0].max(lengths[1])],
                        &[],
                    )
                    .unwrap();
                let out = b
                    .tensor(
                        DType::F32,
                        TensorFlags::OUTPUT,
                        &[*lengths.iter().max().unwrap()],
                        &[],
                    )
                    .unwrap();
                b.node(
                    OpCode::SUB,
                    NodeFlags::default(),
                    &inputs[..2],
                    &[difference],
                    &[],
                )
                .unwrap();
                b.node(
                    OpCode::FMA,
                    NodeFlags::default(),
                    &[difference, inputs[2], inputs[3]],
                    &[out],
                    &[],
                )
                .unwrap();
                let bytes = b.finish().unwrap();
                let graph = bytes.parse().unwrap();
                let policy = crate::AxCpuKernelPolicy {
                    capabilities: if vector {
                        axnn_cpu::CpuCapabilities::PORTABLE_SIMD
                    } else {
                        axnn_cpu::CpuCapabilities::SCALAR
                    },
                    ..Default::default()
                };
                let mut reference = crate::NativeProgram::compile_in(&graph, policy, false, &heap)
                    .unwrap()
                    .prepare_native_in(&heap)
                    .unwrap();
                let mut fused = crate::NativeProgram::compile_in(&graph, policy, true, &heap)
                    .unwrap()
                    .prepare_native_in(&heap)
                    .unwrap();
                assert_eq!(fused.sub_mul_add_fusion_count(), 1);
                assert_eq!(fused.instruction_count(), reference.instruction_count() - 1);
                let values: [[f32; 19]; 4] = core::array::from_fn(|c| {
                    core::array::from_fn(|i| {
                        [
                            0.0,
                            -0.0,
                            1.0,
                            -1.0,
                            f32::MIN_POSITIVE,
                            f32::from_bits(1),
                            f32::MAX,
                            -f32::MAX,
                            f32::INFINITY,
                            f32::NEG_INFINITY,
                            f32::NAN,
                            0.13,
                        ][(i + c * 3) % 12]
                    })
                });
                let native = core::array::from_fn::<_, 4, _>(|c| {
                    NativeRef::F32(&values[c][..lengths[c] as usize])
                });
                let raw = values.map(|p| p.map(f32::to_le_bytes));
                let bindings = core::array::from_fn::<_, 4, _>(|c| {
                    raw[c][..lengths[c] as usize].as_flattened()
                });
                for borrowed in [false, true, false] {
                    reference.run(&bindings).unwrap();
                    if borrowed {
                        fused
                            .run_with_native_inputs_and_resources(&native, &[])
                            .unwrap();
                    } else {
                        fused.run(&bindings).unwrap();
                    }
                    for (&a, &b) in reference
                        .output_f32(0)
                        .unwrap()
                        .iter()
                        .zip(fused.output_f32(0).unwrap())
                    {
                        if a.is_nan() {
                            assert!(b.is_nan());
                        } else {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sub_mul_add_does_not_elide_pinned_or_shared_differences() {
        for pinned in [false, true] {
            let heap = ForkTestHeap::new();
            let mut b = GraphBuilder::new_in(&heap);
            let x = b.tensor(DType::F32, TensorFlags::INPUT, &[4], &[]).unwrap();
            let delta = b
                .tensor(
                    DType::F32,
                    if pinned {
                        TensorFlags::OUTPUT
                    } else {
                        TensorFlags::default()
                    },
                    &[4],
                    &[],
                )
                .unwrap();
            let out = b
                .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
                .unwrap();
            b.node(OpCode::SUB, NodeFlags::default(), &[x, x], &[delta], &[])
                .unwrap();
            b.node(
                OpCode::FMA,
                NodeFlags::default(),
                &[delta, x, if pinned { x } else { delta }],
                &[out],
                &[],
            )
            .unwrap();
            let graph = b.finish().unwrap();
            let program =
                crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap).unwrap();
            assert_eq!(
                program
                    .prepare_native_in(&heap)
                    .unwrap()
                    .sub_mul_add_fusion_count(),
                0
            );
        }
    }

    #[test]
    fn sub_mul_add_repeated_base_retires_before_recycled_slot_reads() {
        let heap = ForkTestHeap::new();
        let mut b = GraphBuilder::new_in(&heap);
        let x = b
            .tensor(DType::F32, TensorFlags::INPUT, &[19], &[])
            .unwrap();
        let color = b
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1],
                &0.75_f32.to_le_bytes(),
            )
            .unwrap();
        let weight = b
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1],
                &0.3_f32.to_le_bytes(),
            )
            .unwrap();
        let delta = b
            .tensor(DType::F32, TensorFlags::default(), &[19], &[])
            .unwrap();
        let mixed = b
            .tensor(DType::F32, TensorFlags::default(), &[19], &[])
            .unwrap();
        let negative = b
            .tensor(DType::F32, TensorFlags::default(), &[19], &[])
            .unwrap();
        let out = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[19], &[])
            .unwrap();
        b.node(
            OpCode::SUB,
            NodeFlags::default(),
            &[color, x],
            &[delta],
            &[],
        )
        .unwrap();
        b.node(
            OpCode::FMA,
            NodeFlags::default(),
            &[delta, weight, x],
            &[mixed],
            &[],
        )
        .unwrap();
        b.node(
            OpCode::NEGATE,
            NodeFlags::default(),
            &[mixed],
            &[negative],
            &[],
        )
        .unwrap();
        b.node(OpCode::ABS, NodeFlags::default(), &[negative], &[out], &[])
            .unwrap();
        let graph = b.finish().unwrap();
        let mut prepared = crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap)
            .unwrap()
            .prepare_native_in(&heap)
            .unwrap();
        assert_eq!(prepared.sub_mul_add_fusion_count(), 1);
        assert_eq!(prepared.inner.borrowed_input_takes.as_slice(), &[2]);
        let slot = prepared.inner.inputs[0].slot as u32;
        assert!(
            prepared
                .inner
                .nodes
                .iter()
                .any(|n| n.output_slots().contains(&slot))
        );
        for tick in 0..3 {
            let values: [f32; 19] = core::array::from_fn(|i| (i as f32 + tick as f32) / 23.0);
            prepared
                .run_with_native_inputs_and_resources(&[NativeRef::F32(&values)], &[])
                .unwrap();
            let expected = values.map(|x| (-((0.75 - x) * 0.3 + x)).abs());
            for (&actual, expected) in prepared.output_f32(0).unwrap().iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn sub_mul_add_keeps_partial_broadcasts_on_checked_nodes() {
        let heap = ForkTestHeap::new();
        let mut b = GraphBuilder::new_in(&heap);
        let x = b
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 1], &[])
            .unwrap();
        let y = b
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 2], &[])
            .unwrap();
        let delta = b
            .tensor(DType::F32, TensorFlags::default(), &[2, 2], &[])
            .unwrap();
        let out = b
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        b.node(OpCode::SUB, NodeFlags::default(), &[x, y], &[delta], &[])
            .unwrap();
        b.node(
            OpCode::FMA,
            NodeFlags::default(),
            &[delta, x, y],
            &[out],
            &[],
        )
        .unwrap();
        let graph = b.finish().unwrap();
        let mut prepared = crate::NativeProgram::compile_auto_in(&graph.parse().unwrap(), &heap)
            .unwrap()
            .prepare_native_in(&heap)
            .unwrap();
        assert_eq!(prepared.sub_mul_add_fusion_count(), 0);
        prepared
            .run_with_native_inputs_and_resources(
                &[NativeRef::F32(&[1.0, 2.0]), NativeRef::F32(&[3.0, 4.0])],
                &[],
            )
            .unwrap();
        assert_eq!(
            prepared.output_f32(0),
            Some([1.0, 1.0, 1.0, 0.0].as_slice())
        );
    }

    #[test]
    fn forked_workers_accept_independent_inputs_and_publish_concurrently() {
        let plan_heap = ForkTestHeap::new();
        let first_heap = ForkTestHeap::new();
        let second_heap = ForkTestHeap::new();
        let original = template(&plan_heap);
        let mut first = original.fork_in(&first_heap).unwrap();
        let mut second = original.fork_in(&second_heap).unwrap();
        drop(original);
        let barrier = std::sync::Barrier::new(2);
        let run = |worker: &mut PreparedNative<&ForkTestHeap>, offset: i32| {
            for tick in 0..32 {
                let value = tick + offset;
                let x = [value as f32; 4].map(f32::to_le_bytes);
                let integers = [value; 4].map(i32::to_le_bytes);
                let bytes = [value as u8; 4];
                barrier.wait();
                worker
                    .run(&[x.as_flattened(), integers.as_flattened(), &bytes])
                    .unwrap();
                barrier.wait();
                assert_eq!(
                    worker.output_f32(0),
                    Some([value as f32 * 2.0 + 3.0; 4].as_slice())
                );
                assert_eq!(worker.output_i32(1), Some([value; 4].as_slice()));
                assert_eq!(worker.output_u8(2), Some(bytes.as_slice()));
            }
        };
        std::thread::scope(|scope| {
            scope.spawn(|| run(&mut first, 0));
            scope.spawn(|| run(&mut second, 100));
        });
        assert!(first.run(&[]).is_err());
        assert!(first.output_f32(0).is_none());
        assert_eq!(second.output_f32(0), Some([265.0; 4].as_slice()));
    }
}
