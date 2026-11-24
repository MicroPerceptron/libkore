//! Allocator-threaded execution of precompiled `.ax` assets on Kore's CPU lane.
//!
//! This is a runtime adapter, not a rendering kernel implementation. `ax`
//! validates/orders/places the DAG; `ax-exec` owns native execution, including
//! automatic exact fusions and last-use reclamation. No host
//! runtime, WGSL parser, GPU API, implicit allocator or CPU feature probe is
//! linked. The final runtime supplies both allocation and kernel policy.

use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::{CostModel, DType, Graph, GraphBuf, NodeView, Schedule, TensorFlags, TensorView};
use ax_exec::{NativeProgram, NativeTensor};
use axnn::lowering::KernelPolicy;
use axnn_cpu::CpuKernelPlan;

use crate::{CPU_LANE, Error, Placement};

/// Typed input planes, borrowed for one invocation. No byte-slice alignment
/// assumption crosses the runtime boundary.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    U8(&'a [u8]),
}

impl Input<'_> {
    fn dtype(self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::U8(_) => DType::U8,
        }
    }
    fn len(self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }
}

pub enum Output<A: Allocator> {
    F32(Vec<f32, A>),
    I32(Vec<i32, A>),
    U8(Vec<u8, A>),
}

impl<A: Allocator> Output<A> {
    pub fn as_input(&self) -> Input<'_> {
        match self {
            Self::F32(v) => Input::F32(v),
            Self::I32(v) => Input::I32(v),
            Self::U8(v) => Input::U8(v),
        }
    }
    pub fn f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }
    pub fn i32(&self) -> Option<&[i32]> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }
}

/// The exact kernel policy participates in placement cost and admission.
/// It cannot admit an op only to discover a missing implementation at draw.
pub(crate) struct CatalogModel {
    pub(crate) policy: KernelPolicy,
}

impl CatalogModel {
    fn plan(&self, graph: &Graph<'_>, node: NodeView<'_>) -> Option<CpuKernelPlan> {
        axnn_cpu::select_cpu_kernel(&ax::node_kernel_request(graph, node)?, self.policy)
    }
}

impl CostModel for CatalogModel {
    fn backend_count(&self) -> u16 {
        1
    }
    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        backend == CPU_LANE && self.plan(graph, node).is_some()
    }
    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        if backend != CPU_LANE {
            return u64::MAX;
        }
        self.plan(graph, node)
            .map_or(u64::MAX, |p| p.estimated_cost)
    }
    fn transfer_cost(&self, _: TensorView<'_>, source: Option<u16>, destination: u16) -> u64 {
        if destination == CPU_LANE && source.is_none_or(|s| s == CPU_LANE) {
            0
        } else {
            u64::MAX
        }
    }
    fn external_output_backend(&self) -> Option<u16> {
        Some(CPU_LANE)
    }
}

/// Kore's typed facade over the shared portable executor. Graph storage and
/// tensor allocations retain the caller's allocator. Native results move into
/// `Output` without encoding to bytes and decoding back into typed planes.
pub struct Program<A: Allocator> {
    native: NativeProgram<A>,
}

impl<A: Allocator> Program<A> {
    pub fn compile_in(bytes: &[u8], policy: KernelPolicy, allocator: A) -> Result<Self, Error>
    where
        A: Clone,
    {
        let bytes = GraphBuf::copy_from_in(bytes, allocator.clone()).map_err(|_| Error::Graph)?;
        Self::from_graph_in(bytes, policy, allocator)
    }

    /// Move an axrd-produced graph into the executable without copying it.
    /// Exact native fusions are automatic; capability/fidelity/scratch admission
    /// still comes entirely from Kore's supplied policy.
    pub fn from_graph_in(
        bytes: GraphBuf<A>,
        policy: KernelPolicy,
        allocator: A,
    ) -> Result<Self, Error>
    where
        A: Clone,
    {
        let graph = bytes.parse().map_err(|_| Error::Graph)?;
        // Preserve the public typed-plane ABI's admission checks, including
        // unused inputs/constants. The executor does not guess caller dtypes
        // from equal byte lengths (e.g. four U8 elements versus one F32).
        for index in 0..graph.tensor_count() {
            let tensor = graph
                .tensor(ax::TensorId(index as u32))
                .map_err(|_| Error::Graph)?;
            if !matches!(tensor.dtype, DType::F32 | DType::I32 | DType::U8) {
                return Err(Error::UnsupportedDType);
            }
            let count = elements(tensor)?;
            if tensor.flags.contains(TensorFlags::CONSTANT)
                && tensor.constant().len()
                    != count
                        .checked_mul(tensor.dtype.storage_bytes(1))
                        .ok_or(Error::InvalidBinding)?
            {
                return Err(Error::InvalidBinding);
            }
        }
        Ok(Self {
            native: NativeProgram::from_graph_in(bytes, policy.into(), true, allocator)
                .map_err(execution_error)?,
        })
    }

    pub fn schedule(&self) -> &Schedule<A> {
        self.native.schedule()
    }
    pub fn graph(&self) -> Graph<'_> {
        self.native.graph()
    }
    pub fn as_bytes(&self) -> &[u8] {
        self.native.as_bytes()
    }
    pub fn placement(&self) -> Placement {
        Placement {
            nodes: self.graph().node_count(),
            partitions: self.schedule().plan().partitions.len(),
            aggregate_cost: self.native.placement().aggregate_cost,
        }
    }
    pub fn kernel_plans(&self) -> impl Iterator<Item = &CpuKernelPlan> {
        self.native.kernel_plans()
    }
    pub fn native_plan_stats(&self) -> ax_exec::AxNativePlanStats {
        self.native.native_plan_stats()
    }

    /// Prepare reusable storage for a caller-owned repeated invocation lane.
    /// The admitted plan and constants are retained; execution then allocates
    /// no per-node or per-output tensors. The arena may outlive this program.
    pub fn prepare_native_in<R: Allocator + Clone>(
        &self,
        allocator: R,
    ) -> Result<ax_exec::PreparedNative<R>, Error> {
        self.native
            .prepare_native_in(allocator)
            .map_err(execution_error)
    }

    /// Inputs/outputs are ordered by tensor ID, as in the WGSL asset ABI.
    /// Transient allocation is independent of persistent program allocation.
    /// The shared executor bounds the live set and borrows aligned inputs.
    pub fn run_in<R: Allocator + Clone>(
        &self,
        inputs: &[Input<'_>],
        allocator: R,
    ) -> Result<Vec<Output<R>, R>, Error> {
        self.run_with_resources_in(inputs, &[], allocator)
    }

    /// [`Self::run_in`] with borrowed resources the graph reads (an RGB
    /// volume, a byte buffer), validated against the graph before it runs.
    pub fn run_with_resources_in<R: Allocator + Clone>(
        &self,
        inputs: &[Input<'_>],
        resources: &[ax_exec::NativeResourceBinding<'_>],
        allocator: R,
    ) -> Result<Vec<Output<R>, R>, Error> {
        if inputs.len() != self.native.input_ids().len() {
            return Err(Error::InvalidBinding);
        }
        let graph = self.graph();
        let mut bindings = Vec::new_in(allocator.clone());
        bindings
            .try_reserve_exact(inputs.len())
            .map_err(|_| Error::Allocation)?;
        for (&input, &id) in inputs.iter().zip(self.native.input_ids()) {
            let tensor = graph.tensor(id).map_err(|_| Error::Graph)?;
            if input.dtype() != tensor.dtype || input.len() != elements(tensor)? {
                return Err(Error::InvalidBinding);
            }
            bindings.push(match input {
                Input::U8(values) => values,
                Input::F32(values) => typed_bytes(values),
                Input::I32(values) => typed_bytes(values),
            });
        }
        let mut run = self
            .native
            .begin_with_resources_in(&bindings, resources, allocator.clone())
            .map_err(execution_error)?;
        run.advance(usize::MAX).map_err(execution_error)?;
        let retained = run.take_retained_in().map_err(execution_error)?;
        let mut outputs = Vec::new_in(allocator.clone());
        outputs
            .try_reserve_exact(retained.len())
            .map_err(|_| Error::Allocation)?;
        for tensor in retained {
            outputs.push(
                match tensor
                    .into_native_in(allocator.clone())
                    .map_err(execution_error)?
                {
                    NativeTensor::F32(values) => Output::F32(values),
                    NativeTensor::I32(values) => Output::I32(values),
                    NativeTensor::U8(values) => Output::U8(values),
                    // This facade admits only F32/I32/U8 tensors above;
                    // preserve its typed-plane contract if the executor grows.
                    NativeTensor::Packed(_, _) => return Err(Error::UnsupportedDType),
                },
            );
        }
        Ok(outputs)
    }
}

fn typed_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: all supported Kore targets are little endian. F32 and I32 have no
    // padding, and this view neither mutates nor changes allocation ownership.
    #[cfg(not(target_endian = "little"))]
    compile_error!("Kore's typed graph input ABI requires a little-endian target");
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}

fn elements(tensor: TensorView<'_>) -> Result<usize, Error> {
    tensor.dimensions().try_fold(1usize, |n, d| {
        if d == ax::DYNAMIC_DIM {
            return Err(Error::UnsupportedShape);
        }
        n.checked_mul(d as usize).ok_or(Error::UnsupportedShape)
    })
}

fn execution_error(error: ax_exec::Error) -> Error {
    match error {
        ax_exec::Error::Ax(ax::AllocationError::AllocationFailed) => Error::Allocation,
        ax_exec::Error::Ax(_) | ax_exec::Error::NativeRequiresSinglePartition => Error::Schedule,
        ax_exec::Error::Plan(_) => Error::Graph,
        ax_exec::Error::MissingTensor(id) => Error::MissingValue(id.0),
        _ => Error::InvalidBinding,
    }
}
