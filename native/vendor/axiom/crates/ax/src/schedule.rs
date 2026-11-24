//! Structural DAG analysis and backend-neutral greedy placement.

use crate::bitmat::{BitMatrixError, BitMatrixMut, BitMatrixRef, BitTableMut};
use crate::format::{Graph, NodeView, TensorFlags, TensorId, TensorView};

pub const NO_PRODUCER: u32 = u32::MAX;
pub const UNPLACED: u16 = u16::MAX;

/// Adapter-owned implementation selected for a placed node.
///
/// Backend IDs describe a tensor residency domain; implementations describe
/// how that backend realizes one logical operation.  Keeping the two separate
/// lets a CPU select dense versus packed sparse code without inventing a fake
/// device and paying nonexistent transfer costs.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Implementation(pub u16);

/// The baseline dense implementation.  Every existing [`CostModel`] supports
/// this implementation by default, preserving the pre-variant API behavior.
pub const DENSE_IMPLEMENTATION: Implementation = Implementation(0);
/// Sentinel used only while a schedule has no valid selected implementation.
pub const UNSELECTED_IMPLEMENTATION: Implementation = Implementation(u16::MAX);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    ScratchTooSmall,
    GraphTooLarge,
    Matrix(BitMatrixError),
    MultipleProducers(TensorId),
    ProducedInputOrConstant(TensorId),
    MissingProducer(TensorId),
    Cycle,
    InvalidGraph,
}

impl From<BitMatrixError> for ScheduleError {
    fn from(value: BitMatrixError) -> Self {
        Self::Matrix(value)
    }
}

pub struct ScheduleScratch<'a> {
    pub adjacency: &'a mut [u64],
    pub producers: &'a mut [u32],
    pub indegree: &'a mut [u32],
    pub queue: &'a mut [u32],
    pub order: &'a mut [u32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleInfo {
    pub nodes: usize,
    pub tensors: usize,
    pub edges: usize,
}

pub fn analyze(
    graph: &Graph<'_>,
    scratch: &mut ScheduleScratch<'_>,
) -> Result<ScheduleInfo, ScheduleError> {
    let nodes = graph.node_count();
    let tensors = graph.tensor_count();
    if nodes > u32::MAX as usize || tensors > u32::MAX as usize {
        return Err(ScheduleError::GraphTooLarge);
    }
    if scratch.producers.len() < tensors
        || scratch.indegree.len() < nodes
        || scratch.queue.len() < nodes
        || scratch.order.len() < nodes
    {
        return Err(ScheduleError::ScratchTooSmall);
    }
    scratch.producers[..tensors].fill(NO_PRODUCER);
    scratch.indegree[..nodes].fill(0);
    let mut matrix = BitMatrixMut::new(scratch.adjacency, nodes)?;

    for node_index in 0..nodes {
        let node = graph
            .node(node_index)
            .map_err(|_| ScheduleError::InvalidGraph)?;
        for output in node.outputs() {
            let tensor = graph
                .tensor(output)
                .map_err(|_| ScheduleError::InvalidGraph)?;
            if tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::CONSTANT)
            {
                return Err(ScheduleError::ProducedInputOrConstant(output));
            }
            let producer = &mut scratch.producers[output.0 as usize];
            if *producer != NO_PRODUCER {
                return Err(ScheduleError::MultipleProducers(output));
            }
            *producer = node_index as u32;
        }
    }

    let mut edges = 0usize;
    let mut last_effect = None;
    for node_index in 0..nodes {
        let node = graph
            .node(node_index)
            .map_err(|_| ScheduleError::InvalidGraph)?;
        for input in node.inputs() {
            let producer = scratch.producers[input.0 as usize];
            if producer == NO_PRODUCER {
                let tensor = graph
                    .tensor(input)
                    .map_err(|_| ScheduleError::InvalidGraph)?;
                if !tensor.flags.contains(TensorFlags::INPUT)
                    && !tensor.flags.contains(TensorFlags::CONSTANT)
                {
                    return Err(ScheduleError::MissingProducer(input));
                }
                continue;
            }
            if matrix.set(producer as usize, node_index)? {
                scratch.indegree[node_index] += 1;
                edges += 1;
            }
        }
        if node.flags.0
            & (crate::format::NodeFlags::ORDERED.0 | crate::format::NodeFlags::MUTATING.0)
            != 0
        {
            if let Some(predecessor) = last_effect
                && matrix.set(predecessor, node_index)?
            {
                scratch.indegree[node_index] += 1;
                edges += 1;
            }
            last_effect = Some(node_index);
        }
    }

    for tensor_index in 0..tensors {
        let tensor = graph
            .tensor(TensorId(tensor_index as u32))
            .map_err(|_| ScheduleError::InvalidGraph)?;
        if tensor.flags.contains(TensorFlags::OUTPUT)
            && scratch.producers[tensor_index] == NO_PRODUCER
            && !tensor.flags.contains(TensorFlags::INPUT)
            && !tensor.flags.contains(TensorFlags::CONSTANT)
        {
            return Err(ScheduleError::MissingProducer(tensor.id));
        }
    }

    let mut tail = 0usize;
    for (node, &degree) in scratch.indegree[..nodes].iter().enumerate() {
        if degree == 0 {
            scratch.queue[tail] = node as u32;
            tail += 1;
        }
    }
    let mut head = 0usize;
    let mut written = 0usize;
    let matrix = matrix.as_ref();
    while head < tail {
        let node = scratch.queue[head] as usize;
        head += 1;
        scratch.order[written] = node as u32;
        written += 1;
        matrix.for_each_set_bit(node, |successor| {
            scratch.indegree[successor] -= 1;
            if scratch.indegree[successor] == 0 {
                scratch.queue[tail] = successor as u32;
                tail += 1;
            }
        })?;
    }
    if written != nodes {
        return Err(ScheduleError::Cycle);
    }
    Ok(ScheduleInfo {
        nodes,
        tensors,
        edges,
    })
}

/// Backend capability and dynamic cost surface consumed by placement.
///
/// Implementations may incorporate queue occupancy, power policy, cache
/// state, transfer domains, or measured shape specializations. The scheduler
/// itself sees only integer backend IDs and costs. Return `u64::MAX` from
/// [`transfer_cost`](Self::transfer_cost) when that transfer is unavailable;
/// the scheduler will discard that backend candidate and continue looking for
/// a backend that can execute the node.
pub trait CostModel {
    fn backend_count(&self) -> u16;
    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool;
    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64;
    fn transfer_cost(&self, tensor: TensorView<'_>, source: Option<u16>, destination: u16) -> u64;

    /// Fixed cost paid once for each contiguous executable partition.
    ///
    /// Node-greedy placement deliberately uses [`execute_cost`](Self::execute_cost),
    /// which may conservatively include this cost per candidate. Aggregate
    /// region refinement uses this value once and
    /// [`region_execute_cost_implementation`](Self::region_execute_cost_implementation)
    /// for the nodes inside the partition.
    fn partition_cost(&self, _backend: u16) -> u64 {
        0
    }

    /// Fixed cost paid once for a contiguous `(backend, implementation)`
    /// executable partition.
    ///
    /// Existing models that price only a backend launch can keep overriding
    /// [`partition_cost`](Self::partition_cost). Adapters whose implementation
    /// families have different setup, packing, or dispatch overheads may
    /// override this finer-grained hook instead.
    fn partition_cost_implementation(&self, backend: u16, _implementation: Implementation) -> u64 {
        self.partition_cost(backend)
    }

    /// Backend domain that can satisfy graph outputs without a separate
    /// external transfer. `None` leaves output egress outside placement cost.
    fn external_output_backend(&self) -> Option<u16> {
        None
    }

    /// Number of implementation candidates for this node on `backend`.
    ///
    /// The old placement entry point intentionally evaluates only the dense
    /// candidate; callers that retain implementation storage (such as
    /// [`crate::Schedule`]) receive the full candidate set.
    fn implementation_count(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u16 {
        1
    }

    /// Whether one adapter-owned implementation is usable.  The default maps
    /// candidate zero to the existing backend capability contract.
    fn supports_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> bool {
        implementation == DENSE_IMPLEMENTATION && self.supports(backend, graph, node)
    }

    /// Estimated execution cost for one implementation.  Transfer costs stay
    /// keyed solely by backend domain.
    fn execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        if implementation == DENSE_IMPLEMENTATION {
            self.execute_cost(backend, graph, node)
        } else {
            u64::MAX
        }
    }

    /// Per-node compute cost used while evaluating a complete backend region.
    ///
    /// Models whose greedy execution cost includes a conservative launch
    /// penalty should override this with compute-only cost.
    fn region_execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        self.execute_cost_implementation(backend, graph, node, implementation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementError {
    ScratchTooSmall,
    InvalidOrder,
    InvalidProducer,
    NoBackend { node: u32 },
}

pub fn place(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &mut [u16],
    residency_words: &mut [u64],
    model: &impl CostModel,
) -> Result<u64, PlacementError> {
    place_with_implementations(
        graph,
        order,
        producers,
        placements,
        None,
        residency_words,
        model,
    )
}

/// Places nodes and, when caller-owned storage is supplied, records the
/// selected implementation for every node.  The legacy [`place`] function
/// remains dense-only so existing external executors cannot accidentally be
/// assigned an implementation they do not yet understand.
pub fn place_with_implementations(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &mut [u16],
    implementations: Option<&mut [Implementation]>,
    residency_words: &mut [u64],
    model: &impl CostModel,
) -> Result<u64, PlacementError> {
    let nodes = graph.node_count();
    if order.len() < nodes
        || producers.len() < graph.tensor_count()
        || placements.len() < nodes
        || implementations
            .as_ref()
            .is_some_and(|implementations| implementations.len() < nodes)
    {
        return Err(PlacementError::ScratchTooSmall);
    }
    placements[..nodes].fill(UNPLACED);
    let mut implementations = implementations;
    if let Some(values) = implementations.as_deref_mut() {
        values[..nodes].fill(UNSELECTED_IMPLEMENTATION);
    }
    let backends = model.backend_count();
    let mut residency = BitTableMut::new(residency_words, graph.tensor_count(), backends as usize)
        .map_err(|_| PlacementError::ScratchTooSmall)?;
    let mut total = 0_u64;
    for &node_index in &order[..nodes] {
        let node_index = node_index as usize;
        if node_index >= nodes || placements[node_index] != UNPLACED {
            return Err(PlacementError::InvalidOrder);
        }
        let node = graph
            .node(node_index)
            .map_err(|_| PlacementError::InvalidOrder)?;
        let mut best: Option<(u64, u16, Implementation)> = None;
        'backend: for backend in 0..backends {
            let implementation_count = if implementations.is_some() {
                model.implementation_count(backend, graph, node)
            } else {
                1
            };
            for raw_implementation in 0..implementation_count {
                let implementation = Implementation(raw_implementation);
                if !model.supports_implementation(backend, graph, node, implementation) {
                    continue;
                }
                let mut cost =
                    model.execute_cost_implementation(backend, graph, node, implementation);
                for (input_index, input) in node.inputs().enumerate() {
                    if node.inputs().take(input_index).any(|seen| seen == input) {
                        continue;
                    }
                    let producer = producers[input.0 as usize];
                    let external = producer == NO_PRODUCER;
                    if !external {
                        let producer = producer as usize;
                        if producer >= nodes || placements[producer] == UNPLACED {
                            return Err(PlacementError::InvalidProducer);
                        }
                    }
                    if !residency
                        .get(input.0 as usize, backend as usize)
                        .map_err(|_| PlacementError::InvalidProducer)?
                    {
                        let tensor = graph
                            .tensor(input)
                            .map_err(|_| PlacementError::InvalidProducer)?;
                        let mut transfer = if external {
                            model.transfer_cost(tensor, None, backend)
                        } else {
                            u64::MAX
                        };
                        for source in 0..backends {
                            if residency
                                .get(input.0 as usize, source as usize)
                                .map_err(|_| PlacementError::InvalidProducer)?
                            {
                                transfer = transfer.min(model.transfer_cost(
                                    tensor,
                                    Some(source),
                                    backend,
                                ));
                            }
                        }
                        if transfer == u64::MAX {
                            // A backend can implement an operation without
                            // being able to materialize every input domain.
                            // That makes this candidate infeasible, not the
                            // graph invalid.
                            continue 'backend;
                        }
                        cost = cost.saturating_add(transfer);
                    }
                }
                if best.is_none_or(|candidate| (cost, backend, implementation) < candidate) {
                    best = Some((cost, backend, implementation));
                }
            }
        }
        let Some((cost, backend, implementation)) = best else {
            return Err(PlacementError::NoBackend {
                node: node_index as u32,
            });
        };
        placements[node_index] = backend;
        if let Some(values) = implementations.as_deref_mut() {
            values[node_index] = implementation;
        }
        for (input_index, input) in node.inputs().enumerate() {
            if !node.inputs().take(input_index).any(|seen| seen == input) {
                residency
                    .set(input.0 as usize, backend as usize)
                    .map_err(|_| PlacementError::InvalidProducer)?;
            }
        }
        for output in node.outputs() {
            residency
                .set(output.0 as usize, backend as usize)
                .map_err(|_| PlacementError::InvalidProducer)?;
        }
        total = total.saturating_add(cost);
    }
    Ok(total)
}

pub fn adjacency<'a>(words: &'a [u64], nodes: usize) -> Result<BitMatrixRef<'a>, ScheduleError> {
    Ok(BitMatrixRef::new(words, nodes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmat::matrix_words;
    use crate::format::{NodeFlags, NodeSpec, OpCode, TensorSpec, encode};
    use axnn::DType;
    use std::vec;

    fn diamond() -> std::vec::Vec<u8> {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags(0),
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags(0),
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags(0),
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
        ];
        let nodes = [
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(1)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::RELU,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::SIGMOID,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::ADD,
                flags: NodeFlags::default(),
                inputs: &[TensorId(2), TensorId(3)],
                outputs: &[TensorId(4)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0_u8; 1024];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        bytes.truncate(len);
        bytes
    }

    #[test]
    fn schedules_dependency_diamond() {
        let bytes = diamond();
        let graph = Graph::parse(&bytes).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        let mut scratch = ScheduleScratch {
            adjacency: &mut adjacency_words,
            producers: &mut producers,
            indegree: &mut indegree,
            queue: &mut queue,
            order: &mut order,
        };
        let info = analyze(&graph, &mut scratch).unwrap();
        assert_eq!(info.edges, 4);
        let position = |node| order.iter().position(|value| *value == node).unwrap();
        assert!(position(0) < position(1));
        assert!(position(0) < position(2));
        assert!(position(1) < position(3));
        assert!(position(2) < position(3));
    }

    struct TwoBackends;

    impl CostModel for TwoBackends {
        fn backend_count(&self) -> u16 {
            2
        }
        fn supports(&self, backend: u16, _graph: &Graph<'_>, node: NodeView<'_>) -> bool {
            backend == 0 || node.op != OpCode::RELU
        }
        fn execute_cost(&self, backend: u16, _graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
            if backend == 1 && node.op == OpCode::SIGMOID {
                1
            } else if backend == 1 {
                4
            } else {
                10
            }
        }
        fn transfer_cost(
            &self,
            _tensor: TensorView<'_>,
            source: Option<u16>,
            destination: u16,
        ) -> u64 {
            if source.is_none() || source == Some(destination) {
                0
            } else {
                2
            }
        }
    }

    struct NearlyEqualBackends;

    impl CostModel for NearlyEqualBackends {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(&self, backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u64 {
            if backend == 0 { 1_001 } else { 1_000 }
        }

        fn transfer_cost(
            &self,
            _tensor: TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }
    }

    #[test]
    fn placement_keeps_exact_cost_order_inside_one_uf8_bucket() {
        use unsigned_float::Uf8E5M3;

        assert_eq!(
            Uf8E5M3::from_f32(1_000.0).to_bits(),
            Uf8E5M3::from_f32(1_001.0).to_bits(),
            "the regression requires costs that UF8 cannot distinguish"
        );
        let bytes = diamond();
        let graph = Graph::parse(&bytes).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();
        let mut placements = vec![UNPLACED; graph.node_count()];
        let mut residency = vec![0; crate::bitmat::table_words(graph.tensor_count(), 2).unwrap()];
        place(
            &graph,
            &order,
            &producers,
            &mut placements,
            &mut residency,
            &NearlyEqualBackends,
        )
        .unwrap();
        assert_eq!(placements, vec![1; graph.node_count()]);
    }

    #[test]
    fn placement_obeys_capability_and_transfer_cost() {
        let bytes = diamond();
        let graph = Graph::parse(&bytes).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();
        let mut placements = vec![UNPLACED; graph.node_count()];
        let mut residency = vec![0; crate::bitmat::table_words(graph.tensor_count(), 2).unwrap()];
        place(
            &graph,
            &order,
            &producers,
            &mut placements,
            &mut residency,
            &TwoBackends,
        )
        .unwrap();
        assert_eq!(placements[1], 0, "relu unsupported on backend 1");
        assert_eq!(placements[2], 1, "cheap sigmoid chooses backend 1");
    }

    struct VariantModel;

    impl CostModel for VariantModel {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u64 {
            10
        }

        fn transfer_cost(
            &self,
            _tensor: TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: NodeView<'_>,
        ) -> u16 {
            2
        }

        fn supports_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            implementation == DENSE_IMPLEMENTATION || implementation == Implementation(1)
        }

        fn execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            if implementation == Implementation(1) {
                1
            } else {
                10
            }
        }
    }

    #[test]
    fn implementation_selection_does_not_create_a_new_transfer_domain() {
        let bytes = diamond();
        let graph = Graph::parse(&bytes).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();
        let mut placements = vec![UNPLACED; graph.node_count()];
        let mut implementations = vec![UNSELECTED_IMPLEMENTATION; graph.node_count()];
        let mut residency = vec![0; crate::bitmat::table_words(graph.tensor_count(), 1).unwrap()];
        let cost = place_with_implementations(
            &graph,
            &order,
            &producers,
            &mut placements,
            Some(&mut implementations),
            &mut residency,
            &VariantModel,
        )
        .unwrap();
        assert_eq!(cost, graph.node_count() as u64);
        assert!(placements.iter().all(|backend| *backend == 0));
        assert!(
            implementations
                .iter()
                .all(|implementation| *implementation == Implementation(1))
        );

        // Legacy callers receive the dense implementation only, preserving
        // behavior until they opt into implementation storage.
        let mut dense_placements = vec![UNPLACED; graph.node_count()];
        let mut dense_residency =
            vec![0; crate::bitmat::table_words(graph.tensor_count(), 1).unwrap()];
        assert_eq!(
            place(
                &graph,
                &order,
                &producers,
                &mut dense_placements,
                &mut dense_residency,
                &VariantModel,
            )
            .unwrap(),
            graph.node_count() as u64 * 10
        );
    }

    #[test]
    fn cycles_are_rejected() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags(0),
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags(0),
                dimensions: &[1],
                constant: &[],
            },
        ];
        let nodes = [
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(0)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(1)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0; 256];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        let result = analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        );
        assert_eq!(result, Err(ScheduleError::Cycle));
    }

    #[test]
    fn effectful_nodes_keep_serialized_order() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
        ];
        let nodes = [
            NodeSpec {
                op: OpCode::COPY,
                flags: NodeFlags::ORDERED,
                inputs: &[TensorId(0)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::COPY,
                flags: NodeFlags::ORDERED,
                inputs: &[TensorId(1)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0; 320];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        let info = analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();
        assert_eq!(info.edges, 1);
        assert!(adjacency(&adjacency_words, 2).unwrap().get(0, 1).unwrap());
    }

    struct ReuseCost;

    impl CostModel for ReuseCost {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u64 {
            1
        }

        fn transfer_cost(
            &self,
            _tensor: TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            100
        }
    }

    #[test]
    fn repeated_input_transfer_is_charged_once_per_backend() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1],
                constant: &[],
            },
        ];
        let nodes = [
            NodeSpec {
                op: OpCode::RELU,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(1)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::SIGMOID,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0; 320];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();
        let mut placements = vec![UNPLACED; graph.node_count()];
        let mut residency = vec![0; crate::bitmat::table_words(graph.tensor_count(), 1).unwrap()];
        let cost = place(
            &graph,
            &order,
            &producers,
            &mut placements,
            &mut residency,
            &ReuseCost,
        )
        .unwrap();
        assert_eq!(cost, 102);
    }

    struct DomainLimitedBackends;

    impl CostModel for DomainLimitedBackends {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(&self, _backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u64 {
            1
        }

        fn transfer_cost(
            &self,
            _tensor: TensorView<'_>,
            source: Option<u16>,
            destination: u16,
        ) -> u64 {
            match (source, destination) {
                // Backend zero can execute the op but cannot import the
                // graph's external input domain.
                (None, 0) => u64::MAX,
                _ => 1,
            }
        }
    }

    #[test]
    fn placement_skips_backends_that_cannot_import_an_input() {
        let bytes = diamond();
        let graph = Graph::parse(&bytes).unwrap();
        let mut adjacency_words = vec![0; matrix_words(graph.node_count()).unwrap()];
        let mut producers = vec![0; graph.tensor_count()];
        let mut indegree = vec![0; graph.node_count()];
        let mut queue = vec![0; graph.node_count()];
        let mut order = vec![0; graph.node_count()];
        analyze(
            &graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency_words,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )
        .unwrap();

        let mut placements = vec![UNPLACED; graph.node_count()];
        let mut residency = vec![0; crate::bitmat::table_words(graph.tensor_count(), 2).unwrap()];
        let cost = place(
            &graph,
            &order,
            &producers,
            &mut placements,
            &mut residency,
            &DomainLimitedBackends,
        )
        .unwrap();

        assert_eq!(placements, [1, 1, 1, 1]);
        assert_eq!(cost, 5);
    }
}
