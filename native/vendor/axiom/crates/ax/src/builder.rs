//! Allocator-owned construction of canonical `.ax` graphs.
//!
//! The builder owns flat tensor, node, dimension, reference, attribute, and
//! constant tables. Every allocation uses the allocator supplied to
//! [`GraphBuilder::new_in`]; callers that cannot clone their allocator may pass
//! it by reference.

use alloc::vec::Vec;
use core::alloc::Allocator;

use crate::{
    AllocationError, AxError, DType, GraphBuf, NodeFlags, NodeSpec, OpCode, TensorFlags, TensorId,
    TensorSpec,
};

#[derive(Clone, Copy)]
struct TensorRecord {
    dtype: DType,
    flags: TensorFlags,
    dimensions_start: usize,
    dimensions_len: usize,
    constant_start: usize,
    constant_len: usize,
}

#[derive(Clone, Copy)]
struct NodeRecord {
    op: OpCode,
    flags: NodeFlags,
    references_start: usize,
    input_count: usize,
    output_count: usize,
    attributes_start: usize,
    attributes_len: usize,
}

/// Incrementally constructs a canonical `.ax` graph using caller-owned
/// allocation policy.
///
/// Tensor and node payloads are copied into allocator-backed flat tables when
/// they are added, so their source slices need not outlive the corresponding
/// method call. [`finish`](Self::finish) performs the format's canonical
/// validation and returns graph bytes owned by the same allocator.
pub struct GraphBuilder<A: Allocator> {
    allocator: A,
    tensors: Vec<TensorRecord, A>,
    nodes: Vec<NodeRecord, A>,
    dimensions: Vec<u32, A>,
    references: Vec<TensorId, A>,
    attributes: Vec<u8, A>,
    constants: Vec<u8, A>,
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Creates an empty builder without allocating.
    pub fn new_in(allocator: A) -> Self {
        Self {
            tensors: Vec::new_in(allocator.clone()),
            nodes: Vec::new_in(allocator.clone()),
            dimensions: Vec::new_in(allocator.clone()),
            references: Vec::new_in(allocator.clone()),
            attributes: Vec::new_in(allocator.clone()),
            constants: Vec::new_in(allocator.clone()),
            allocator,
        }
    }

    /// Adds a tensor and returns its stable graph-local identifier.
    ///
    /// Dimensions and constant bytes are copied into builder-owned storage.
    /// Canonical flag, rank, and constant validation is performed immediately;
    /// whole-graph validation is repeated by [`finish`](Self::finish).
    pub fn tensor(
        &mut self,
        dtype: DType,
        flags: TensorFlags,
        dimensions: &[u32],
        constant: &[u8],
    ) -> Result<TensorId, AllocationError> {
        validate_tensor(flags, dimensions, constant)?;

        let tensor_count = self
            .tensors
            .len()
            .checked_add(1)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(AxError::CountOverflow)?;
        checked_u32_end(self.dimensions.len(), dimensions.len())?;
        checked_u32_end(self.constants.len(), constant.len())?;

        reserve(&mut self.tensors, 1)?;
        reserve(&mut self.dimensions, dimensions.len())?;
        reserve(&mut self.constants, constant.len())?;

        let record = TensorRecord {
            dtype,
            flags,
            dimensions_start: self.dimensions.len(),
            dimensions_len: dimensions.len(),
            constant_start: self.constants.len(),
            constant_len: constant.len(),
        };
        self.dimensions.extend_from_slice(dimensions);
        self.constants.extend_from_slice(constant);
        self.tensors.push(record);
        Ok(TensorId(tensor_count - 1))
    }

    /// Adds an operator whose tensor references use IDs returned by
    /// [`tensor`](Self::tensor).
    ///
    /// Input, output, and attribute slices are copied into builder-owned
    /// storage. Operation-specific arity and complete graph structure are
    /// validated when [`finish`](Self::finish) encodes the graph.
    pub fn node(
        &mut self,
        op: OpCode,
        flags: NodeFlags,
        inputs: &[TensorId],
        outputs: &[TensorId],
        attributes: &[u8],
    ) -> Result<(), AllocationError> {
        validate_node_header(flags, inputs, outputs, self.tensors.len())?;

        self.nodes
            .len()
            .checked_add(1)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(AxError::CountOverflow)?;
        let reference_count = inputs
            .len()
            .checked_add(outputs.len())
            .ok_or(AxError::CountOverflow)?;
        checked_u32_end(self.references.len(), reference_count)?;
        checked_u32_end(self.attributes.len(), attributes.len())?;

        reserve(&mut self.nodes, 1)?;
        reserve(&mut self.references, reference_count)?;
        reserve(&mut self.attributes, attributes.len())?;

        let record = NodeRecord {
            op,
            flags,
            references_start: self.references.len(),
            input_count: inputs.len(),
            output_count: outputs.len(),
            attributes_start: self.attributes.len(),
            attributes_len: attributes.len(),
        };
        self.references.extend_from_slice(inputs);
        self.references.extend_from_slice(outputs);
        self.attributes.extend_from_slice(attributes);
        self.nodes.push(record);
        Ok(())
    }

    /// Encodes and validates the accumulated graph into canonical `.ax` bytes.
    pub fn finish(self) -> Result<GraphBuf<A>, AllocationError> {
        let mut tensors = Vec::new_in(self.allocator.clone());
        let mut nodes = Vec::new_in(self.allocator.clone());
        reserve(&mut tensors, self.tensors.len())?;
        reserve(&mut nodes, self.nodes.len())?;

        for tensor in &self.tensors {
            tensors.push(TensorSpec {
                dtype: tensor.dtype,
                flags: tensor.flags,
                dimensions: subslice(
                    &self.dimensions,
                    tensor.dimensions_start,
                    tensor.dimensions_len,
                ),
                constant: subslice(&self.constants, tensor.constant_start, tensor.constant_len),
            });
        }
        for node in &self.nodes {
            let outputs_start = node.references_start + node.input_count;
            nodes.push(NodeSpec {
                op: node.op,
                flags: node.flags,
                inputs: subslice(&self.references, node.references_start, node.input_count),
                outputs: subslice(&self.references, outputs_start, node.output_count),
                attributes: subslice(&self.attributes, node.attributes_start, node.attributes_len),
            });
        }

        let graph = GraphBuf::encode_in(&tensors, &nodes, self.allocator.clone())?;
        graph.parse()?;
        Ok(graph)
    }
}

fn validate_tensor(
    flags: TensorFlags,
    dimensions: &[u32],
    constant: &[u8],
) -> Result<(), AllocationError> {
    if flags.0 & !(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0 | TensorFlags::CONSTANT.0) != 0 {
        return Err(AxError::InvalidTensorFlags(flags.0).into());
    }
    if dimensions.len() > axnn::MAX_RANK {
        return Err(
            AxError::RankTooLarge(u8::try_from(dimensions.len()).unwrap_or(u8::MAX)).into(),
        );
    }
    if !constant.is_empty() && !flags.contains(TensorFlags::CONSTANT) {
        return Err(AxError::ConstantFlagMissing.into());
    }
    Ok(())
}

fn validate_node_header(
    flags: NodeFlags,
    inputs: &[TensorId],
    outputs: &[TensorId],
    tensor_count: usize,
) -> Result<(), AllocationError> {
    if flags.0
        & !(NodeFlags::ORDERED.0
            | NodeFlags::MUTATING.0
            | NodeFlags::READS_RESOURCE.0
            | NodeFlags::WRITES_RESOURCE.0)
        != 0
    {
        return Err(AxError::InvalidNodeFlags(flags.0).into());
    }
    u16::try_from(inputs.len()).map_err(|_| AxError::CountOverflow)?;
    u16::try_from(outputs.len()).map_err(|_| AxError::CountOverflow)?;
    for tensor in inputs.iter().chain(outputs) {
        if tensor.0 as usize >= tensor_count {
            return Err(AxError::InvalidTensorReference(tensor.0).into());
        }
    }
    Ok(())
}

fn checked_u32_end(start: usize, len: usize) -> Result<(), AllocationError> {
    start
        .checked_add(len)
        .and_then(|end| u32::try_from(end).ok())
        .ok_or(AxError::CountOverflow)?;
    Ok(())
}

/// Amortized, never exact: the builder appends one record per tensor and
/// node, so an exact reservation would reallocate on every push — a
/// 52-layer decoder graph did so 3,785 times, 1,269 of them past 8 KiB,
/// which on an allocator that serves each large block from a fresh extent
/// (kore's userland heap) mapped 282 MiB for a graph whose live size never
/// passed 7 MiB, and then failed.
fn reserve<T, A: Allocator>(
    values: &mut Vec<T, A>,
    additional: usize,
) -> Result<(), AllocationError> {
    values
        .try_reserve(additional)
        .map_err(|_| AllocationError::AllocationFailed)
}

fn subslice<T>(values: &[T], start: usize, len: usize) -> &[T] {
    &values[start..start + len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr::NonNull;
    use std::alloc::{AllocError, Global, Layout};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct AllocationCounts {
        live: AtomicUsize,
        total: AtomicUsize,
    }

    #[derive(Clone)]
    struct TrackingAllocator(Arc<AllocationCounts>);

    // SAFETY: allocation and deallocation are delegated unchanged to `Global`;
    // the wrapper only records successful ownership transitions for this test.
    unsafe impl Allocator for TrackingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let allocation = Global.allocate(layout)?;
            self.0.live.fetch_add(1, Ordering::Relaxed);
            self.0.total.fetch_add(1, Ordering::Relaxed);
            Ok(allocation)
        }

        unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
            self.0.live.fetch_sub(1, Ordering::Relaxed);
            // SAFETY: `pointer` and `layout` are the pair returned by the
            // delegated allocation or an Allocator default grow operation.
            unsafe { Global.deallocate(pointer, layout) };
        }
    }

    #[test]
    fn supplied_allocator_owns_builder_and_finished_graph() {
        let counts = Arc::new(AllocationCounts::default());
        {
            let mut builder = GraphBuilder::new_in(TrackingAllocator(Arc::clone(&counts)));
            let input = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[1, 2], &[])
                .unwrap();
            let weights = builder
                .tensor(
                    DType::F32,
                    TensorFlags::CONSTANT,
                    &[2, 1],
                    &[0, 0, 128, 63, 0, 0, 0, 64],
                )
                .unwrap();
            let output = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1], &[])
                .unwrap();
            builder
                .node(
                    OpCode::MATMUL,
                    NodeFlags::default(),
                    &[input, weights],
                    &[output],
                    &[],
                )
                .unwrap();
            assert!(counts.live.load(Ordering::Relaxed) > 0);

            let graph = builder.finish().unwrap();
            assert!(counts.live.load(Ordering::Relaxed) > 0);
            let parsed = graph.parse().unwrap();
            assert_eq!(parsed.tensor_count(), 3);
            assert_eq!(parsed.node_count(), 1);
            assert_eq!(parsed.tensor(weights).unwrap().constant().len(), 8);
        }
        assert!(counts.total.load(Ordering::Relaxed) > 0);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn finish_rejects_noncanonical_arity() {
        let mut builder = GraphBuilder::new_in(&Global);
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1], &[])
            .unwrap();
        builder
            .node(OpCode::ADD, NodeFlags::default(), &[input], &[output], &[])
            .unwrap();
        match builder.finish() {
            Err(AllocationError::Format(AxError::InvalidArity {
                op,
                inputs,
                outputs,
            })) => {
                assert_eq!(op, OpCode::ADD);
                assert_eq!(inputs, 1);
                assert_eq!(outputs, 1);
            }
            _ => panic!("invalid arity was not rejected"),
        }
    }
}
