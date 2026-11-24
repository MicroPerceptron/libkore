//! Allocator-owned graph and schedule conveniences.
//!
//! Every allocation is made through the caller's `Allocator`. These types do
//! not require a global allocator and remain usable in `no_std` environments.

use alloc::vec::Vec;
use core::alloc::Allocator;

use crate::bitmat::{
    BitMatrixRef, BitSetMut, BitTableMut, BitTableRef, matrix_words, table_words, words_for,
};
use crate::format::{
    AxError, Graph, NodeSpec, TensorFlags, TensorId, TensorSpec, encode, encoded_len,
};
use crate::partition::{
    Partition, PartitionError, PartitionInfo, PlanView, Transfer, partition_with_implementations,
};
use crate::schedule::{
    CostModel, DENSE_IMPLEMENTATION, Implementation, NO_PRODUCER, PlacementError, ScheduleError,
    ScheduleInfo, ScheduleScratch, UNPLACED, analyze, place_with_implementations,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    AllocationFailed,
    Format(AxError),
    Analysis(ScheduleError),
    Placement(PlacementError),
    Partition(PartitionError),
}

/// Telemetry from the complete backend-neutral placement pipeline.
///
/// `greedy_cost` records the conservative node-at-a-time seed. The remaining
/// fields describe the final placement after dependency-safe clustering and
/// fixed-point aggregate region refinement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlacementReport {
    pub greedy_cost: u64,
    /// Backend/implementation runs removed by topological clustering.
    pub regions_eliminated: usize,
    /// Nodes moved while amortizing launch and unique transfer costs.
    pub nodes_reassigned: usize,
    /// Final partition-aware cost, including output egress.
    pub aggregate_cost: u64,
}

impl From<AxError> for AllocationError {
    fn from(value: AxError) -> Self {
        Self::Format(value)
    }
}

impl From<ScheduleError> for AllocationError {
    fn from(value: ScheduleError) -> Self {
        Self::Analysis(value)
    }
}

impl From<PlacementError> for AllocationError {
    fn from(value: PlacementError) -> Self {
        Self::Placement(value)
    }
}

impl From<PartitionError> for AllocationError {
    fn from(value: PartitionError) -> Self {
        Self::Partition(value)
    }
}

/// Canonical `.ax` bytes owned by an allocator selected by the caller.
pub struct GraphBuf<A: Allocator> {
    bytes: Vec<u8, A>,
}

impl<A: Allocator> GraphBuf<A> {
    pub fn copy_from_in(bytes: &[u8], allocator: A) -> Result<Self, AllocationError> {
        Graph::parse(bytes)?;
        let mut owned = filled_in(bytes.len(), 0_u8, allocator)?;
        owned.copy_from_slice(bytes);
        Ok(Self { bytes: owned })
    }

    pub fn encode_in(
        tensors: &[TensorSpec<'_>],
        nodes: &[NodeSpec<'_>],
        allocator: A,
    ) -> Result<Self, AllocationError> {
        let len = encoded_len(tensors, nodes)?;
        let mut bytes = filled_in(len, 0_u8, allocator)?;
        encode(&mut bytes, tensors, nodes)?;
        Ok(Self { bytes })
    }

    pub fn parse(&self) -> Result<Graph<'_>, AxError> {
        Graph::parse(&self.bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8, A> {
        self.bytes
    }
}

/// An analyzed graph schedule whose storage belongs to a supplied allocator.
///
/// `A: Clone` is needed only while constructing the independent vectors. Pass
/// `&allocator` when the allocator itself is not cheaply cloneable.
pub struct Schedule<A: Allocator> {
    info: ScheduleInfo,
    adjacency: Vec<u64, A>,
    producers: Vec<u32, A>,
    order: Vec<u32, A>,
    placements: Vec<u16, A>,
    implementations: Vec<Implementation, A>,
    residency: Vec<u64, A>,
    /// Backend-major support mask for aggregate region refinement.
    region_support: Vec<u64, A>,
    /// Exact best regional implementation and weight for every backend/node.
    region_implementations: Vec<Implementation, A>,
    region_costs: Vec<u64, A>,
    /// Maximum implementation-candidate count observed by the most recent
    /// placement model. The following two vectors are pre-sized by `place` so
    /// partition-aware implementation DP remains allocation-free during
    /// refinement.
    implementation_width: usize,
    implementation_dp_costs: Vec<u64, A>,
    implementation_dp_predecessors: Vec<Implementation, A>,
    /// Scratch order used by the placement-aware topological clustering pass.
    /// Kept with the schedule so the optimization remains allocation-free
    /// after analysis, including in `no_std` callers.
    clustered_order: Vec<u32, A>,
    /// Original topological position of each node during clustering.
    order_positions: Vec<u32, A>,
    /// Packed ready positions during clustering.
    ready: Vec<u64, A>,
    node_partitions: Vec<u32, A>,
    partitions: Vec<Partition, A>,
    transfers: Vec<Transfer, A>,
    plan: PartitionInfo,
}

impl<A: Allocator + Clone> Schedule<A> {
    pub fn analyze_in(graph: &Graph<'_>, allocator: A) -> Result<Self, AllocationError> {
        let nodes = graph.node_count();
        let tensors = graph.tensor_count();
        let adjacency_len = matrix_words(nodes).ok_or(ScheduleError::GraphTooLarge)?;
        let mut adjacency = filled_in(adjacency_len, 0_u64, allocator.clone())?;
        let mut producers = filled_in(tensors, 0_u32, allocator.clone())?;
        let mut indegree = filled_in(nodes, 0_u32, allocator.clone())?;
        let mut queue = filled_in(nodes, 0_u32, allocator.clone())?;
        let mut order = filled_in(nodes, 0_u32, allocator.clone())?;
        let placements = filled_in(nodes, UNPLACED, allocator.clone())?;
        let clustered_order = filled_in(nodes, 0_u32, allocator.clone())?;
        let order_positions = filled_in(nodes, 0_u32, allocator.clone())?;
        let ready = filled_in(words_for(nodes), 0_u64, allocator.clone())?;
        let implementations = filled_in(nodes, DENSE_IMPLEMENTATION, allocator.clone())?;
        let node_partitions = filled_in(nodes, u32::MAX, allocator.clone())?;
        let partitions = filled_in(nodes, Partition::default(), allocator.clone())?;
        let transfer_capacity = graph
            .reference_count()
            .checked_add(tensors)
            .ok_or(ScheduleError::GraphTooLarge)?;
        let transfers = filled_in(transfer_capacity, Transfer::default(), allocator.clone())?;
        let residency = Vec::new_in(allocator.clone());
        let region_support = Vec::new_in(allocator.clone());
        let region_implementations = Vec::new_in(allocator.clone());
        let region_costs = Vec::new_in(allocator.clone());
        let implementation_dp_costs = Vec::new_in(allocator.clone());
        let implementation_dp_predecessors = Vec::new_in(allocator);
        let info = analyze(
            graph,
            &mut ScheduleScratch {
                adjacency: &mut adjacency,
                producers: &mut producers,
                indegree: &mut indegree,
                queue: &mut queue,
                order: &mut order,
            },
        )?;
        Ok(Self {
            info,
            adjacency,
            producers,
            order,
            placements,
            implementations,
            residency,
            region_support,
            region_implementations,
            region_costs,
            implementation_width: 0,
            implementation_dp_costs,
            implementation_dp_predecessors,
            clustered_order,
            order_positions,
            ready,
            node_partitions,
            partitions,
            transfers,
            plan: PartitionInfo::default(),
        })
    }
}

impl<A: Allocator> Schedule<A> {
    pub const fn info(&self) -> ScheduleInfo {
        self.info
    }

    pub fn adjacency(&self) -> BitMatrixRef<'_> {
        // Construction validated this exact allocation and node count.
        BitMatrixRef::new(&self.adjacency, self.info.nodes).unwrap()
    }

    pub fn producers(&self) -> &[u32] {
        &self.producers
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    pub fn placements(&self) -> &[u16] {
        &self.placements
    }

    /// Adapter-owned implementations selected alongside [`placements`](Self::placements).
    /// They do not describe a transfer domain and are never serialized into
    /// the `.ax` graph.
    pub fn implementations(&self) -> &[Implementation] {
        &self.implementations
    }

    /// Reassigns one already-placed node for a post-placement optimization.
    ///
    /// The caller is responsible for checking that `backend` supports the
    /// node. Any existing partition plan is invalidated and must be rebuilt
    /// with [`build_plan`](Self::build_plan).
    pub fn reassign_node(&mut self, node: usize, backend: u16) -> Result<(), AllocationError> {
        self.reassign_node_with_implementation(node, backend, DENSE_IMPLEMENTATION)
    }

    /// Reassigns one node with an explicit adapter-owned implementation.
    ///
    /// This is intended for post-placement refinement that moves a node into
    /// a different backend domain.  Callers selecting a non-dense CPU kernel
    /// should generally provide it through [`CostModel`] so cost accounting
    /// remains coherent.
    pub fn reassign_node_with_implementation(
        &mut self,
        node: usize,
        backend: u16,
        implementation: Implementation,
    ) -> Result<(), AllocationError> {
        let placement = self
            .placements
            .get_mut(node)
            .ok_or(PlacementError::InvalidOrder)?;
        if *placement == UNPLACED || backend == UNPLACED || implementation.0 == u16::MAX {
            return Err(PlacementError::InvalidOrder.into());
        }
        *placement = backend;
        *self
            .implementations
            .get_mut(node)
            .ok_or(PlacementError::InvalidOrder)? = implementation;
        self.plan = PartitionInfo::default();
        Ok(())
    }

    pub fn place(
        &mut self,
        graph: &Graph<'_>,
        model: &impl CostModel,
    ) -> Result<u64, AllocationError> {
        if graph.node_count() != self.info.nodes || graph.tensor_count() != self.info.tensors {
            return Err(PlacementError::InvalidOrder.into());
        }
        let residency_len = table_words(graph.tensor_count(), model.backend_count() as usize)
            .ok_or(AllocationError::AllocationFailed)?;
        resize_filled(&mut self.residency, residency_len, 0_u64)?;
        let backends = model.backend_count() as usize;
        let region_len = graph
            .node_count()
            .checked_mul(backends)
            .ok_or(AllocationError::AllocationFailed)?;
        let support_len =
            table_words(backends, graph.node_count()).ok_or(AllocationError::AllocationFailed)?;
        resize_filled(&mut self.region_support, support_len, 0_u64)?;
        resize_filled(
            &mut self.region_implementations,
            region_len,
            crate::schedule::UNSELECTED_IMPLEMENTATION,
        )?;
        resize_filled(&mut self.region_costs, region_len, u64::MAX)?;
        let mut implementation_width = 0_usize;
        for backend in 0..model.backend_count() {
            for node_index in 0..graph.node_count() {
                let node = graph
                    .node(node_index)
                    .map_err(|_| PlacementError::InvalidOrder)?;
                implementation_width = implementation_width.max(usize::from(
                    model.implementation_count(backend, graph, node),
                ));
            }
        }
        let implementation_states = graph
            .node_count()
            .checked_mul(implementation_width)
            .ok_or(AllocationError::AllocationFailed)?;
        let implementation_costs = implementation_width
            .checked_mul(2)
            .ok_or(AllocationError::AllocationFailed)?;
        resize_filled(
            &mut self.implementation_dp_predecessors,
            implementation_states,
            crate::schedule::UNSELECTED_IMPLEMENTATION,
        )?;
        resize_filled(
            &mut self.implementation_dp_costs,
            implementation_costs,
            u64::MAX,
        )?;
        self.implementation_width = implementation_width;
        let cost = place_with_implementations(
            graph,
            &self.order,
            &self.producers,
            &mut self.placements,
            Some(&mut self.implementations),
            &mut self.residency,
            model,
        )?;
        self.plan = PartitionInfo::default();
        Ok(cost)
    }

    /// Place and optimize a graph through the complete generic pipeline.
    ///
    /// This is the normal adapter entry point. It prevents each provider
    /// integration from having to reproduce the ordering-sensitive
    /// `place → cluster → refine → cluster` sequence that lets realistic GR,
    /// rendering, and inference regions amortize launch and transfer costs.
    /// The operation remains allocation-free after schedule analysis.
    pub fn place_optimized(
        &mut self,
        graph: &Graph<'_>,
        model: &impl CostModel,
    ) -> Result<PlacementReport, AllocationError> {
        let greedy_cost = self.place(graph, model)?;
        let mut regions_eliminated = self.cluster_by_placement()?;
        let nodes_reassigned = self.refine_regions(graph, model)?;
        regions_eliminated = regions_eliminated.saturating_add(self.cluster_by_placement()?);
        let aggregate_cost = self.aggregate_cost(graph, model)?;
        Ok(PlacementReport {
            greedy_cost,
            regions_eliminated,
            nodes_reassigned,
            aggregate_cost,
        })
    }

    /// Reconsider complete contiguous regions using partition launch costs,
    /// compute-only implementation costs, and unique tensor transfers.
    ///
    /// Greedy placement remains a conservative valid baseline. This pass
    /// moves a maximal same-placement run only when the aggregate graph cost
    /// strictly decreases, and repeats to a fixed point so accepted moves can
    /// fuse neighboring partitions. It is backend-neutral and allocation-free
    /// after [`place`](Self::place) has sized the schedule's residency scratch.
    ///
    /// Returns the number of node backend or implementation assignments
    /// changed by accepted aggregate-cost improvements.
    pub fn refine_regions(
        &mut self,
        graph: &Graph<'_>,
        model: &impl CostModel,
    ) -> Result<usize, AllocationError> {
        let nodes = self.info.nodes;
        let backends = model.backend_count();
        if graph.node_count() != nodes
            || graph.tensor_count() != self.info.tensors
            || self.order.len() < nodes
            || self.placements.len() < nodes
            || self.implementations.len() < nodes
            || self.placements[..nodes].contains(&UNPLACED)
        {
            return Err(PlacementError::InvalidOrder.into());
        }
        if nodes == 0 {
            return Ok(0);
        }
        let residency_len = table_words(graph.tensor_count(), backends as usize)
            .ok_or(AllocationError::AllocationFailed)?;
        resize_filled(&mut self.residency, residency_len, 0_u64)?;
        self.prepare_region_weights(graph, model)?;
        let region_weights = RegionWeights {
            support: BitTableRef::new(&self.region_support, backends as usize, nodes)
                .map_err(|_| PlacementError::ScratchTooSmall)?,
            implementations: &self.region_implementations,
            costs: &self.region_costs,
            nodes,
            backends: backends as usize,
        };
        let mut baseline = placed_graph_cost(
            graph,
            &self.order,
            &self.producers,
            &self.placements,
            &self.implementations,
            &mut self.residency,
            model,
            Some(region_weights),
        )?;
        let mut reassigned = 0_usize;

        loop {
            let mut changed = false;

            // For a fixed residency-domain run, implementation selection is a
            // shortest path over candidate IDs. A state retains its current
            // implementation for free or starts a new implementation-specific
            // partition and pays that candidate's launch cost. Backpointers
            // recover the exact minimum for the complete run, so a common
            // implementation can replace heterogeneous greedy choices when
            // launch amortization makes the joint change profitable.
            let mut start = 0_usize;
            while start < nodes {
                let first = self.order[start] as usize;
                if first >= nodes {
                    return Err(PlacementError::InvalidOrder.into());
                }
                let backend = self.placements[first];
                let mut end = start + 1;
                while end < nodes {
                    let node = self.order[end] as usize;
                    if node >= nodes || self.placements[node] != backend {
                        break;
                    }
                    end += 1;
                }
                save_assignment_range(
                    &self.order,
                    &self.placements,
                    &self.implementations,
                    &mut self.order_positions,
                    start,
                    end,
                )?;
                if let Err(error) = select_implementation_range(
                    graph,
                    &self.order,
                    &self.placements,
                    &mut self.implementations,
                    backend,
                    start,
                    end,
                    self.implementation_width,
                    &mut self.implementation_dp_costs,
                    &mut self.implementation_dp_predecessors,
                    model,
                ) {
                    restore_assignment_range(
                        &self.order,
                        &mut self.placements,
                        &mut self.implementations,
                        &self.order_positions,
                        start,
                        end,
                    )?;
                    return Err(error);
                }
                let candidate = placed_graph_cost(
                    graph,
                    &self.order,
                    &self.producers,
                    &self.placements,
                    &self.implementations,
                    &mut self.residency,
                    model,
                    Some(region_weights),
                );
                match candidate {
                    Ok(candidate) if candidate < baseline => {
                        baseline = candidate;
                        reassigned = reassigned.saturating_add(changed_assignments(
                            &self.order,
                            &self.placements,
                            &self.implementations,
                            &self.order_positions,
                            start,
                            end,
                        )?);
                        changed = true;
                        self.plan = PartitionInfo::default();
                    }
                    result => {
                        restore_assignment_range(
                            &self.order,
                            &mut self.placements,
                            &mut self.implementations,
                            &self.order_positions,
                            start,
                            end,
                        )?;
                        result?;
                    }
                }
                start = end;
            }

            for target in 0..backends {
                let mut start = 0_usize;
                while start < nodes {
                    let first = self.order[start] as usize;
                    if first >= nodes {
                        return Err(PlacementError::InvalidOrder.into());
                    }
                    let source = self.placements[first];
                    if source == target || !region_weights.supports(target as usize, first) {
                        start += 1;
                        continue;
                    }

                    let mut end = start + 1;
                    while end < nodes {
                        let node_index = self.order[end] as usize;
                        if node_index >= nodes || self.placements[node_index] != source {
                            break;
                        }
                        if !region_weights.supports(target as usize, node_index) {
                            break;
                        }
                        end += 1;
                    }

                    // Include already-target neighbors in implementation DP.
                    // Their implementations may need to change jointly with
                    // the moved run before a newly fused partition becomes
                    // profitable.
                    let mut optimize_start = start;
                    while optimize_start != 0 {
                        let node = self.order[optimize_start - 1] as usize;
                        if node >= nodes || self.placements[node] != target {
                            break;
                        }
                        optimize_start -= 1;
                    }
                    let mut optimize_end = end;
                    while optimize_end < nodes {
                        let node = self.order[optimize_end] as usize;
                        if node >= nodes || self.placements[node] != target {
                            break;
                        }
                        optimize_end += 1;
                    }
                    save_assignment_range(
                        &self.order,
                        &self.placements,
                        &self.implementations,
                        &mut self.order_positions,
                        optimize_start,
                        optimize_end,
                    )?;
                    for position in start..end {
                        let node_index = self.order[position] as usize;
                        self.placements[node_index] = target;
                    }
                    if let Err(error) = select_implementation_range(
                        graph,
                        &self.order,
                        &self.placements,
                        &mut self.implementations,
                        target,
                        optimize_start,
                        optimize_end,
                        self.implementation_width,
                        &mut self.implementation_dp_costs,
                        &mut self.implementation_dp_predecessors,
                        model,
                    ) {
                        restore_assignment_range(
                            &self.order,
                            &mut self.placements,
                            &mut self.implementations,
                            &self.order_positions,
                            optimize_start,
                            optimize_end,
                        )?;
                        if matches!(
                            error,
                            AllocationError::Placement(PlacementError::NoBackend { .. })
                        ) {
                            start = end;
                            continue;
                        }
                        return Err(error);
                    }
                    let candidate = placed_graph_cost(
                        graph,
                        &self.order,
                        &self.producers,
                        &self.placements,
                        &self.implementations,
                        &mut self.residency,
                        model,
                        Some(region_weights),
                    );
                    match candidate {
                        Ok(candidate) if candidate < baseline => {
                            baseline = candidate;
                            reassigned = reassigned.saturating_add(changed_assignments(
                                &self.order,
                                &self.placements,
                                &self.implementations,
                                &self.order_positions,
                                optimize_start,
                                optimize_end,
                            )?);
                            changed = true;
                            self.plan = PartitionInfo::default();
                        }
                        result => {
                            restore_assignment_range(
                                &self.order,
                                &mut self.placements,
                                &mut self.implementations,
                                &self.order_positions,
                                optimize_start,
                                optimize_end,
                            )?;
                            result?;
                        }
                    }
                    start = end;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(reassigned)
    }

    fn prepare_region_weights(
        &mut self,
        graph: &Graph<'_>,
        model: &impl CostModel,
    ) -> Result<(), AllocationError> {
        let nodes = self.info.nodes;
        let backends = model.backend_count() as usize;
        let expected = nodes
            .checked_mul(backends)
            .ok_or(AllocationError::AllocationFailed)?;
        if self.region_implementations.len() < expected || self.region_costs.len() < expected {
            return Err(PlacementError::ScratchTooSmall.into());
        }
        self.region_implementations[..expected].fill(crate::schedule::UNSELECTED_IMPLEMENTATION);
        self.region_costs[..expected].fill(u64::MAX);
        let mut support = BitTableMut::new(&mut self.region_support, backends, nodes)
            .map_err(|_| PlacementError::ScratchTooSmall)?;
        for backend in 0..backends {
            for node_index in 0..nodes {
                let node = graph
                    .node(node_index)
                    .map_err(|_| PlacementError::InvalidOrder)?;
                let Some((cost, implementation)) =
                    best_region_weight(model, backend as u16, graph, node)
                else {
                    continue;
                };
                let index = backend * nodes + node_index;
                self.region_implementations[index] = implementation;
                self.region_costs[index] = cost;
                support
                    .set(backend, node_index)
                    .map_err(|_| PlacementError::ScratchTooSmall)?;
            }
        }
        Ok(())
    }

    /// Aggregate cost of the current complete placement.
    ///
    /// This uses the same one-per-partition launch, compute-only region,
    /// unique residency transfer, and output-egress accounting as
    /// [`refine_regions`](Self::refine_regions). Cost units remain those of
    /// `model`; adapters may calibrate them into warm execution ticks before
    /// constructing cross-graph dispatch candidates.
    pub fn aggregate_cost(
        &mut self,
        graph: &Graph<'_>,
        model: &impl CostModel,
    ) -> Result<u64, AllocationError> {
        let nodes = self.info.nodes;
        let backends = model.backend_count();
        if graph.node_count() != nodes
            || graph.tensor_count() != self.info.tensors
            || self.order.len() < nodes
            || self.placements.len() < nodes
            || self.implementations.len() < nodes
            || self.placements[..nodes].contains(&UNPLACED)
        {
            return Err(PlacementError::InvalidOrder.into());
        }
        let residency_len = table_words(graph.tensor_count(), backends as usize)
            .ok_or(AllocationError::AllocationFailed)?;
        resize_filled(&mut self.residency, residency_len, 0_u64)?;
        placed_graph_cost(
            graph,
            &self.order,
            &self.producers,
            &self.placements,
            &self.implementations,
            &mut self.residency,
            model,
            None,
        )
    }

    /// Reorders independent nodes to cluster work already assigned to the
    /// same backend and implementation, reducing executable regions without
    /// changing placement.
    ///
    /// Every emitted order remains topological because a node is selected
    /// only after all incoming dependency edges have retired. `ORDERED` and
    /// `MUTATING` nodes are therefore kept behind the serialization edges
    /// installed by analysis. The original order is retained unless the new
    /// order strictly reduces the number of backend runs.
    ///
    /// Returns the number of executable regions eliminated. The pass uses
    /// schedule-owned scratch and performs no allocation.
    pub fn cluster_by_placement(&mut self) -> Result<usize, AllocationError> {
        let nodes = self.info.nodes;
        if self.order.len() < nodes
            || self.placements.len() < nodes
            || self.implementations.len() < nodes
            || self.clustered_order.len() < nodes
            || self.order_positions.len() < nodes
            || self.ready.len() < words_for(nodes)
            || self.node_partitions.len() < nodes
        {
            return Err(PlacementError::ScratchTooSmall.into());
        }
        if self.placements[..nodes].contains(&UNPLACED) {
            return Err(PlacementError::InvalidOrder.into());
        }
        if nodes == 0 {
            return Ok(0);
        }

        // `node_partitions` is scratch until build_plan() populates it. Use it
        // as a fresh indegree table rather than allocating a second worklist.
        let indegree = &mut self.node_partitions[..nodes];
        indegree.fill(0);
        let adjacency =
            BitMatrixRef::new(&self.adjacency, nodes).map_err(|_| PlacementError::InvalidOrder)?;
        for predecessor in 0..nodes {
            let mut overflow = false;
            adjacency
                .for_each_set_bit(predecessor, |successor| {
                    let Some(next) = indegree[successor].checked_add(1) else {
                        overflow = true;
                        return;
                    };
                    indegree[successor] = next;
                })
                .map_err(|_| PlacementError::InvalidOrder)?;
            if overflow {
                return Err(PlacementError::InvalidOrder.into());
            }
        }

        let original_runs = placement_runs(
            &self.order[..nodes],
            &self.placements,
            &self.implementations,
        )?;
        for (position, &node) in self.order[..nodes].iter().enumerate() {
            let node = node as usize;
            if node >= nodes {
                return Err(PlacementError::InvalidOrder.into());
            }
            self.order_positions[node] = position as u32;
        }
        let mut ready =
            BitSetMut::new(&mut self.ready, nodes).map_err(|_| PlacementError::ScratchTooSmall)?;
        for &node in &self.order[..nodes] {
            let node = node as usize;
            if indegree[node] == 0 {
                ready
                    .set(self.order_positions[node] as usize)
                    .map_err(|_| PlacementError::InvalidOrder)?;
            }
        }

        let mut active_placement = None;
        for destination in &mut self.clustered_order[..nodes] {
            let selected_position = active_placement
                .and_then(|active| {
                    ready.first_where(|position| {
                        let candidate = self.order[position] as usize;
                        active == (self.placements[candidate], self.implementations[candidate])
                    })
                })
                .or_else(|| ready.first())
                .ok_or(PlacementError::InvalidOrder)?;
            ready
                .clear(selected_position)
                .map_err(|_| PlacementError::InvalidOrder)?;
            let selected = self.order[selected_position] as usize;
            *destination = selected as u32;
            indegree[selected] = u32::MAX;
            active_placement = Some((self.placements[selected], self.implementations[selected]));
            let mut invalid = false;
            adjacency
                .for_each_set_bit(selected, |successor| {
                    let Some(next) = indegree[successor].checked_sub(1) else {
                        invalid = true;
                        return;
                    };
                    indegree[successor] = next;
                    if next == 0 && ready.set(self.order_positions[successor] as usize).is_err() {
                        invalid = true;
                    }
                })
                .map_err(|_| PlacementError::InvalidOrder)?;
            if invalid {
                return Err(PlacementError::InvalidOrder.into());
            }
        }

        let clustered_runs = placement_runs(
            &self.clustered_order[..nodes],
            &self.placements,
            &self.implementations,
        )?;
        if clustered_runs >= original_runs {
            return Ok(0);
        }
        self.order[..nodes].copy_from_slice(&self.clustered_order[..nodes]);
        self.plan = PartitionInfo::default();
        Ok(original_runs - clustered_runs)
    }

    pub fn build_plan(&mut self, graph: &Graph<'_>) -> Result<PlanView<'_>, PartitionError> {
        if graph.node_count() != self.info.nodes || graph.tensor_count() != self.info.tensors {
            return Err(PartitionError::InvalidOrder);
        }
        self.plan = partition_with_implementations(
            graph,
            &self.order,
            &self.producers,
            &self.placements,
            &self.implementations,
            &mut self.node_partitions,
            &mut self.partitions,
            &mut self.transfers,
        )?;
        Ok(self.plan())
    }

    pub fn plan(&self) -> PlanView<'_> {
        PlanView {
            partitions: &self.partitions[..self.plan.partitions],
            transfers: &self.transfers[..self.plan.transfers],
        }
    }
}

fn placement_runs(
    order: &[u32],
    placements: &[u16],
    implementations: &[Implementation],
) -> Result<usize, PlacementError> {
    let mut previous = None;
    let mut runs = 0_usize;
    for &node in order {
        let backend = placements
            .get(node as usize)
            .copied()
            .filter(|&backend| backend != UNPLACED)
            .ok_or(PlacementError::InvalidOrder)?;
        let implementation = implementations
            .get(node as usize)
            .copied()
            .ok_or(PlacementError::InvalidOrder)?;
        if previous != Some((backend, implementation)) {
            runs += 1;
            previous = Some((backend, implementation));
        }
    }
    Ok(runs)
}

fn best_region_weight(
    model: &impl CostModel,
    backend: u16,
    graph: &Graph<'_>,
    node: crate::format::NodeView<'_>,
) -> Option<(u64, Implementation)> {
    let mut best = None;
    for raw in 0..model.implementation_count(backend, graph, node) {
        let implementation = Implementation(raw);
        if !model.supports_implementation(backend, graph, node, implementation) {
            continue;
        }
        let cost = model.region_execute_cost_implementation(backend, graph, node, implementation);
        if best.is_none_or(|candidate: (u64, Implementation)| (cost, implementation) < candidate) {
            best = Some((cost, implementation));
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn select_implementation_range(
    graph: &Graph<'_>,
    order: &[u32],
    placements: &[u16],
    implementations: &mut [Implementation],
    backend: u16,
    start: usize,
    end: usize,
    width: usize,
    costs: &mut [u64],
    predecessors: &mut [Implementation],
    model: &impl CostModel,
) -> Result<(), AllocationError> {
    let nodes = graph.node_count();
    let state_len = nodes
        .checked_mul(width)
        .ok_or(AllocationError::AllocationFailed)?;
    if start >= end
        || end > nodes
        || order.len() < nodes
        || placements.len() < nodes
        || implementations.len() < nodes
        || width == 0
        || costs.len() < width.saturating_mul(2)
        || predecessors.len() < state_len
    {
        return Err(PlacementError::ScratchTooSmall.into());
    }

    let previous_pair = if start == 0 {
        None
    } else {
        let node = order[start - 1] as usize;
        Some((
            *placements.get(node).ok_or(PlacementError::InvalidOrder)?,
            *implementations
                .get(node)
                .ok_or(PlacementError::InvalidOrder)?,
        ))
    };
    let next_pair = if end == nodes {
        None
    } else {
        let node = order[end] as usize;
        Some((
            *placements.get(node).ok_or(PlacementError::InvalidOrder)?,
            *implementations
                .get(node)
                .ok_or(PlacementError::InvalidOrder)?,
        ))
    };
    let predecessor_start = start
        .checked_mul(width)
        .ok_or(AllocationError::AllocationFailed)?;
    let predecessor_end = end
        .checked_mul(width)
        .ok_or(AllocationError::AllocationFailed)?;
    predecessors[predecessor_start..predecessor_end]
        .fill(crate::schedule::UNSELECTED_IMPLEMENTATION);

    let (mut previous, mut current) = costs.split_at_mut(width);
    previous.fill(u64::MAX);
    current[..width].fill(u64::MAX);

    let first_index = order[start] as usize;
    if first_index >= nodes || placements[first_index] != backend {
        return Err(PlacementError::InvalidOrder.into());
    }
    let first = graph
        .node(first_index)
        .map_err(|_| PlacementError::InvalidOrder)?;
    let first_count = usize::from(model.implementation_count(backend, graph, first));
    if first_count > width {
        return Err(PlacementError::ScratchTooSmall.into());
    }
    let mut reachable = false;
    for (raw, cost) in previous.iter_mut().enumerate().take(first_count) {
        let implementation = Implementation(raw as u16);
        if !model.supports_implementation(backend, graph, first, implementation) {
            continue;
        }
        let launch = if previous_pair == Some((backend, implementation)) {
            0
        } else {
            model.partition_cost_implementation(backend, implementation)
        };
        *cost = launch.saturating_add(model.region_execute_cost_implementation(
            backend,
            graph,
            first,
            implementation,
        ));
        reachable = true;
    }
    if !reachable {
        return Err(PlacementError::NoBackend {
            node: first_index as u32,
        }
        .into());
    }

    for position in start + 1..end {
        current[..width].fill(u64::MAX);
        let previous_index = order[position - 1] as usize;
        let previous_node = graph
            .node(previous_index)
            .map_err(|_| PlacementError::InvalidOrder)?;
        let previous_count = usize::from(model.implementation_count(backend, graph, previous_node));
        if previous_count > width {
            return Err(PlacementError::ScratchTooSmall.into());
        }
        let mut best = None;
        let mut second = None;
        for (raw, &cost) in previous.iter().enumerate().take(previous_count) {
            let implementation = Implementation(raw as u16);
            if !model.supports_implementation(backend, graph, previous_node, implementation) {
                continue;
            }
            let candidate = (cost, implementation);
            if best.is_none_or(|value| candidate < value) {
                second = best;
                best = Some(candidate);
            } else if second.is_none_or(|value| candidate < value) {
                second = Some(candidate);
            }
        }
        let Some(best) = best else {
            return Err(PlacementError::NoBackend {
                node: previous_index as u32,
            }
            .into());
        };

        let node_index = order[position] as usize;
        if node_index >= nodes || placements[node_index] != backend {
            return Err(PlacementError::InvalidOrder.into());
        }
        let node = graph
            .node(node_index)
            .map_err(|_| PlacementError::InvalidOrder)?;
        let implementation_count = usize::from(model.implementation_count(backend, graph, node));
        if implementation_count > width {
            return Err(PlacementError::ScratchTooSmall.into());
        }
        reachable = false;
        for raw in 0..implementation_count {
            let implementation = Implementation(raw as u16);
            if !model.supports_implementation(backend, graph, node, implementation) {
                continue;
            }

            let stay = (raw < previous_count
                && model.supports_implementation(backend, graph, previous_node, implementation))
            .then_some((previous[raw], implementation));
            let switch_predecessor = if best.1 != implementation {
                Some(best)
            } else {
                second
            };
            let switch = switch_predecessor.map(|(cost, predecessor)| {
                (
                    cost.saturating_add(
                        model.partition_cost_implementation(backend, implementation),
                    ),
                    predecessor,
                )
            });
            let selected = match (stay, switch) {
                (Some(stay), Some(switch)) => {
                    if switch.0 < stay.0 {
                        switch
                    } else {
                        stay
                    }
                }
                (Some(stay), None) => stay,
                (None, Some(switch)) => switch,
                (None, None) => continue,
            };
            current[raw] = selected
                .0
                .saturating_add(model.region_execute_cost_implementation(
                    backend,
                    graph,
                    node,
                    implementation,
                ));
            predecessors[position * width + raw] = selected.1;
            reachable = true;
        }
        if !reachable {
            return Err(PlacementError::NoBackend {
                node: node_index as u32,
            }
            .into());
        }
        core::mem::swap(&mut previous, &mut current);
    }

    let last_index = order[end - 1] as usize;
    let last = graph
        .node(last_index)
        .map_err(|_| PlacementError::InvalidOrder)?;
    let last_count = usize::from(model.implementation_count(backend, graph, last));
    if last_count > width {
        return Err(PlacementError::ScratchTooSmall.into());
    }
    let mut selected = None;
    for (raw, &path_cost) in previous.iter().enumerate().take(last_count) {
        let implementation = Implementation(raw as u16);
        if !model.supports_implementation(backend, graph, last, implementation) {
            continue;
        }
        let boundary_mismatch = next_pair.is_some_and(|(next_backend, next_implementation)| {
            next_backend == backend && next_implementation != implementation
        });
        let mut cost = path_cost;
        if boundary_mismatch {
            let (_, next_implementation) = next_pair.unwrap();
            cost = cost
                .saturating_add(model.partition_cost_implementation(backend, next_implementation));
        }
        let candidate = (cost, boundary_mismatch, implementation);
        if selected.is_none_or(|value| candidate < value) {
            selected = Some(candidate);
        }
    }
    let Some((_, _, mut implementation)) = selected else {
        return Err(PlacementError::NoBackend {
            node: last_index as u32,
        }
        .into());
    };
    for position in (start..end).rev() {
        let node_index = order[position] as usize;
        implementations[node_index] = implementation;
        if position != start {
            implementation = predecessors[position * width + implementation.0 as usize];
            if implementation == crate::schedule::UNSELECTED_IMPLEMENTATION {
                return Err(PlacementError::InvalidOrder.into());
            }
        }
    }
    Ok(())
}

fn save_assignment_range(
    order: &[u32],
    placements: &[u16],
    implementations: &[Implementation],
    saved: &mut [u32],
    start: usize,
    end: usize,
) -> Result<(), PlacementError> {
    if end > order.len() || end > saved.len() || start > end {
        return Err(PlacementError::ScratchTooSmall);
    }
    for position in start..end {
        let node = order[position] as usize;
        let backend = *placements.get(node).ok_or(PlacementError::InvalidOrder)?;
        let implementation = *implementations
            .get(node)
            .ok_or(PlacementError::InvalidOrder)?;
        saved[position] = (u32::from(backend) << 16) | u32::from(implementation.0);
    }
    Ok(())
}

fn restore_assignment_range(
    order: &[u32],
    placements: &mut [u16],
    implementations: &mut [Implementation],
    saved: &[u32],
    start: usize,
    end: usize,
) -> Result<(), PlacementError> {
    if end > order.len() || end > saved.len() || start > end {
        return Err(PlacementError::ScratchTooSmall);
    }
    for position in start..end {
        let node = order[position] as usize;
        let packed = saved[position];
        *placements
            .get_mut(node)
            .ok_or(PlacementError::InvalidOrder)? = (packed >> 16) as u16;
        *implementations
            .get_mut(node)
            .ok_or(PlacementError::InvalidOrder)? = Implementation(packed as u16);
    }
    Ok(())
}

fn changed_assignments(
    order: &[u32],
    placements: &[u16],
    implementations: &[Implementation],
    saved: &[u32],
    start: usize,
    end: usize,
) -> Result<usize, PlacementError> {
    if end > order.len() || end > saved.len() || start > end {
        return Err(PlacementError::ScratchTooSmall);
    }
    let mut changed = 0_usize;
    for position in start..end {
        let node = order[position] as usize;
        let current = (u32::from(*placements.get(node).ok_or(PlacementError::InvalidOrder)?) << 16)
            | u32::from(
                implementations
                    .get(node)
                    .ok_or(PlacementError::InvalidOrder)?
                    .0,
            );
        changed += usize::from(current != saved[position]);
    }
    Ok(changed)
}

#[derive(Clone, Copy)]
struct RegionWeights<'a> {
    support: BitTableRef<'a>,
    implementations: &'a [Implementation],
    costs: &'a [u64],
    nodes: usize,
    backends: usize,
}

impl RegionWeights<'_> {
    fn index(self, backend: usize, node: usize) -> Option<usize> {
        if backend >= self.backends || node >= self.nodes {
            return None;
        }
        backend.checked_mul(self.nodes)?.checked_add(node)
    }

    fn supports(self, backend: usize, node: usize) -> bool {
        self.support.get(backend, node).unwrap_or(false)
    }

    fn cost(self, backend: usize, node: usize, implementation: Implementation) -> Option<u64> {
        let index = self.index(backend, node)?;
        (self.implementations[index] == implementation).then_some(self.costs[index])
    }
}

#[allow(clippy::too_many_arguments)]
fn placed_graph_cost(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &[u16],
    implementations: &[Implementation],
    residency_words: &mut [u64],
    model: &impl CostModel,
    region_weights: Option<RegionWeights<'_>>,
) -> Result<u64, AllocationError> {
    let nodes = graph.node_count();
    let tensors = graph.tensor_count();
    let backends = model.backend_count();
    if order.len() < nodes
        || producers.len() < tensors
        || placements.len() < nodes
        || implementations.len() < nodes
    {
        return Err(PlacementError::ScratchTooSmall.into());
    }
    residency_words.fill(0);
    let mut residency = BitTableMut::new(residency_words, tensors, backends as usize)
        .map_err(|_| PlacementError::ScratchTooSmall)?;
    let mut total = 0_u64;
    let mut previous_partition = None;
    for &node_index in &order[..nodes] {
        let node_index = node_index as usize;
        if node_index >= nodes {
            return Err(PlacementError::InvalidOrder.into());
        }
        let backend = placements[node_index];
        let implementation = implementations[node_index];
        if backend >= backends || backend == UNPLACED || implementation.0 == u16::MAX {
            return Err(PlacementError::InvalidOrder.into());
        }
        if previous_partition != Some((backend, implementation)) {
            total =
                total.saturating_add(model.partition_cost_implementation(backend, implementation));
            previous_partition = Some((backend, implementation));
        }
        let node = graph
            .node(node_index)
            .map_err(|_| PlacementError::InvalidOrder)?;
        let region_cost = region_weights
            .and_then(|weights| weights.cost(backend as usize, node_index, implementation))
            .unwrap_or_else(|| {
                model.region_execute_cost_implementation(backend, graph, node, implementation)
            });
        total = total.saturating_add(region_cost);

        for input in node.inputs() {
            let tensor_index = input.0 as usize;
            if tensor_index >= tensors {
                return Err(PlacementError::InvalidProducer.into());
            }
            if residency
                .get(tensor_index, backend as usize)
                .map_err(|_| PlacementError::InvalidProducer)?
            {
                continue;
            }
            let producer = producers[tensor_index];
            if producer != NO_PRODUCER {
                let producer = producer as usize;
                if producer >= nodes || placements[producer] == UNPLACED {
                    return Err(PlacementError::InvalidProducer.into());
                }
            }
            let tensor = graph
                .tensor(input)
                .map_err(|_| PlacementError::InvalidProducer)?;
            let mut transfer = if producer == NO_PRODUCER {
                model.transfer_cost(tensor, None, backend)
            } else {
                u64::MAX
            };
            for source in 0..backends {
                if residency
                    .get(tensor_index, source as usize)
                    .map_err(|_| PlacementError::InvalidProducer)?
                {
                    transfer = transfer.min(model.transfer_cost(tensor, Some(source), backend));
                }
            }
            total = total.saturating_add(transfer);
            residency
                .set(tensor_index, backend as usize)
                .map_err(|_| PlacementError::InvalidProducer)?;
        }
        for output in node.outputs() {
            residency
                .set(output.0 as usize, backend as usize)
                .map_err(|_| PlacementError::InvalidProducer)?;
        }
    }

    if let Some(host) = model.external_output_backend() {
        if host >= backends {
            return Err(PlacementError::InvalidProducer.into());
        }
        for tensor_index in 0..tensors {
            let tensor = graph
                .tensor(TensorId(tensor_index as u32))
                .map_err(|_| PlacementError::InvalidProducer)?;
            if !tensor.flags.contains(TensorFlags::OUTPUT)
                || residency
                    .get(tensor_index, host as usize)
                    .map_err(|_| PlacementError::InvalidProducer)?
            {
                continue;
            }
            let producer = producers[tensor_index];
            if producer == NO_PRODUCER {
                continue;
            }
            let producer = producer as usize;
            if producer >= nodes {
                return Err(PlacementError::InvalidProducer.into());
            }
            let mut transfer = u64::MAX;
            for source in 0..backends {
                if residency
                    .get(tensor_index, source as usize)
                    .map_err(|_| PlacementError::InvalidProducer)?
                {
                    transfer = transfer.min(model.transfer_cost(tensor, Some(source), host));
                }
            }
            total = total.saturating_add(transfer);
        }
    }
    Ok(total)
}

fn resize_filled<T: Clone, A: Allocator>(
    values: &mut Vec<T, A>,
    len: usize,
    value: T,
) -> Result<(), AllocationError> {
    values.clear();
    if values.capacity() < len {
        values
            .try_reserve_exact(len)
            .map_err(|_| AllocationError::AllocationFailed)?;
    }
    values.resize(len, value);
    Ok(())
}

fn filled_in<T: Clone, A: Allocator>(
    len: usize,
    value: T,
    allocator: A,
) -> Result<Vec<T, A>, AllocationError> {
    let mut values = Vec::new_in(allocator);
    values
        .try_reserve_exact(len)
        .map_err(|_| AllocationError::AllocationFailed)?;
    values.resize(len, value);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{NodeFlags, OpCode, TensorFlags, TensorId};
    use axnn::DType;
    use std::alloc::Global;

    fn two_node_chain() -> GraphBuf<Global> {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
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
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
        ];
        GraphBuf::encode_in(&tensors, &nodes, Global).unwrap()
    }

    fn one_node_graph() -> GraphBuf<Global> {
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
        ];
        let nodes = [NodeSpec {
            op: OpCode::RELU,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(1)],
            attributes: &[],
        }];
        GraphBuf::encode_in(&tensors, &nodes, Global).unwrap()
    }

    struct OneBackend;

    impl CostModel for OneBackend {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            1
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }
    }

    struct OpBackends;

    impl CostModel for OpBackends {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(&self, backend: u16, _graph: &Graph<'_>, node: crate::NodeView<'_>) -> u64 {
            match (backend, node.op) {
                (0, OpCode::RELU) | (1, OpCode::SIGMOID) => 0,
                _ => 100,
            }
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }
    }

    struct LaunchAmortized;

    impl CostModel for LaunchAmortized {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            if backend == 0 { 10 } else { 101 }
        }

        fn region_execute_cost_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            _implementation: Implementation,
        ) -> u64 {
            if backend == 0 { 10 } else { 1 }
        }

        fn partition_cost(&self, backend: u16) -> u64 {
            if backend == 1 { 15 } else { 0 }
        }

        fn external_output_backend(&self) -> Option<u16> {
            Some(0)
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            source: Option<u16>,
            destination: u16,
        ) -> u64 {
            if source == Some(destination) || (source.is_none() && destination == 0) {
                0
            } else {
                2
            }
        }
    }

    struct ImplementationLaunchCosts;

    impl CostModel for ImplementationLaunchCosts {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            100
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u16 {
            2
        }

        fn supports_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            implementation.0 < 2
        }

        fn execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            match (node.op, implementation) {
                (OpCode::RELU, DENSE_IMPLEMENTATION) | (OpCode::SIGMOID, Implementation(1)) => 1,
                _ => 100,
            }
        }

        fn partition_cost_implementation(
            &self,
            _backend: u16,
            implementation: Implementation,
        ) -> u64 {
            if implementation == DENSE_IMPLEMENTATION {
                3
            } else {
                11
            }
        }
    }

    struct OneBackendImplementationRefinement;

    impl CostModel for OneBackendImplementationRefinement {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            2
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u16 {
            2
        }

        fn supports_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            implementation.0 < 2
        }

        fn execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            if implementation == DENSE_IMPLEMENTATION {
                2
            } else {
                1
            }
        }

        fn region_execute_cost_implementation(
            &self,
            backend: u16,
            graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            self.execute_cost_implementation(backend, graph, node, implementation)
        }

        fn partition_cost_implementation(
            &self,
            _backend: u16,
            implementation: Implementation,
        ) -> u64 {
            if implementation == DENSE_IMPLEMENTATION {
                0
            } else {
                100
            }
        }
    }

    struct HeterogeneousImplementationFusion;

    impl CostModel for HeterogeneousImplementationFusion {
        fn backend_count(&self) -> u16 {
            1
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            200
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u16 {
            3
        }

        fn supports_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            implementation == DENSE_IMPLEMENTATION
                || matches!(
                    (node.op, implementation),
                    (OpCode::RELU, Implementation(1)) | (OpCode::SIGMOID, Implementation(2))
                )
        }

        fn execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            if matches!(
                (node.op, implementation),
                (OpCode::RELU, Implementation(1)) | (OpCode::SIGMOID, Implementation(2))
            ) {
                0
            } else {
                200
            }
        }

        fn region_execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            _implementation: Implementation,
        ) -> u64 {
            0
        }

        fn partition_cost_implementation(
            &self,
            _backend: u16,
            implementation: Implementation,
        ) -> u64 {
            if implementation == DENSE_IMPLEMENTATION {
                100
            } else {
                60
            }
        }
    }

    struct HeterogeneousBackendMove;

    impl CostModel for HeterogeneousBackendMove {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            200
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u16 {
            if backend == 0 { 3 } else { 1 }
        }

        fn supports_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            match backend {
                0 => matches!(
                    (node.op, implementation),
                    (OpCode::RELU, Implementation(1)) | (OpCode::SIGMOID, Implementation(2))
                ),
                1 => implementation == DENSE_IMPLEMENTATION,
                _ => false,
            }
        }

        fn execute_cost_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            if backend == 0
                && matches!(
                    (node.op, implementation),
                    (OpCode::RELU, Implementation(1)) | (OpCode::SIGMOID, Implementation(2))
                )
            {
                0
            } else {
                200
            }
        }

        fn region_execute_cost_implementation(
            &self,
            _backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            _implementation: Implementation,
        ) -> u64 {
            0
        }

        fn partition_cost_implementation(
            &self,
            backend: u16,
            _implementation: Implementation,
        ) -> u64 {
            if backend == 0 { 60 } else { 100 }
        }
    }

    struct PartitionAwareTargetImplementation;

    impl CostModel for PartitionAwareTargetImplementation {
        fn backend_count(&self) -> u16 {
            2
        }

        fn supports(&self, _backend: u16, _graph: &Graph<'_>, _node: crate::NodeView<'_>) -> bool {
            true
        }

        fn execute_cost(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u64 {
            if backend == 0 { 0 } else { 200 }
        }

        fn transfer_cost(
            &self,
            _tensor: crate::TensorView<'_>,
            _source: Option<u16>,
            _destination: u16,
        ) -> u64 {
            0
        }

        fn implementation_count(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
        ) -> u16 {
            if backend == 0 { 1 } else { 2 }
        }

        fn supports_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> bool {
            match backend {
                0 => implementation == DENSE_IMPLEMENTATION,
                1 => implementation.0 < 2,
                _ => false,
            }
        }

        fn execute_cost_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            _implementation: Implementation,
        ) -> u64 {
            if backend == 0 { 0 } else { 200 }
        }

        fn region_execute_cost_implementation(
            &self,
            backend: u16,
            _graph: &Graph<'_>,
            _node: crate::NodeView<'_>,
            implementation: Implementation,
        ) -> u64 {
            match (backend, implementation) {
                (0, DENSE_IMPLEMENTATION) => 100,
                (1, DENSE_IMPLEMENTATION) => 0,
                (1, Implementation(1)) => 20,
                _ => u64::MAX,
            }
        }

        fn partition_cost_implementation(
            &self,
            backend: u16,
            implementation: Implementation,
        ) -> u64 {
            if backend == 1 && implementation == DENSE_IMPLEMENTATION {
                200
            } else {
                0
            }
        }
    }

    #[test]
    fn aggregate_cost_prices_each_implementation_boundary() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
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
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        assert_eq!(
            schedule.place(&graph, &ImplementationLaunchCosts).unwrap(),
            2
        );
        assert_eq!(
            schedule.implementations(),
            &[DENSE_IMPLEMENTATION, Implementation(1)]
        );
        // Dense launch (3) + RELU (1) + specialized launch (11) + SIGMOID (1).
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &ImplementationLaunchCosts)
                .unwrap(),
            16
        );
        let plan = schedule.build_plan(&graph).unwrap();
        assert_eq!(plan.partitions.len(), 2);
        assert_eq!(plan.partitions[0].backend, plan.partitions[1].backend);
        assert_eq!(plan.partitions[0].implementation, DENSE_IMPLEMENTATION);
        assert_eq!(plan.partitions[1].implementation, Implementation(1));
    }

    #[test]
    fn implementation_partition_cost_defaults_to_legacy_backend_cost() {
        assert_eq!(
            LaunchAmortized.partition_cost_implementation(1, Implementation(7)),
            LaunchAmortized.partition_cost(1)
        );
    }

    #[test]
    fn refinement_can_change_implementations_inside_one_backend() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
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
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        schedule
            .place(&graph, &OneBackendImplementationRefinement)
            .unwrap();
        assert_eq!(schedule.implementations(), &[Implementation(1); 2]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &OneBackendImplementationRefinement)
                .unwrap(),
            102,
        );

        assert_eq!(
            schedule
                .refine_regions(&graph, &OneBackendImplementationRefinement)
                .unwrap(),
            2,
        );
        assert_eq!(schedule.implementations(), &[DENSE_IMPLEMENTATION; 2]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &OneBackendImplementationRefinement)
                .unwrap(),
            4,
        );
    }

    #[test]
    fn refinement_fuses_heterogeneous_implementations_when_joint_launch_is_cheaper() {
        let bytes = two_node_chain();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        schedule
            .place(&graph, &HeterogeneousImplementationFusion)
            .unwrap();
        assert_eq!(
            schedule.implementations(),
            &[Implementation(1), Implementation(2)]
        );
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &HeterogeneousImplementationFusion)
                .unwrap(),
            120,
        );

        assert_eq!(
            schedule
                .refine_regions(&graph, &HeterogeneousImplementationFusion)
                .unwrap(),
            2,
        );
        assert_eq!(schedule.implementations(), &[DENSE_IMPLEMENTATION; 2]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &HeterogeneousImplementationFusion)
                .unwrap(),
            100,
        );
    }

    #[test]
    fn refinement_moves_a_heterogeneous_source_backend_run_atomically() {
        let bytes = two_node_chain();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        schedule.place(&graph, &HeterogeneousBackendMove).unwrap();
        assert_eq!(schedule.placements(), &[0, 0]);
        assert_eq!(
            schedule.implementations(),
            &[Implementation(1), Implementation(2)]
        );
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &HeterogeneousBackendMove)
                .unwrap(),
            120,
        );

        assert_eq!(
            schedule
                .refine_regions(&graph, &HeterogeneousBackendMove)
                .unwrap(),
            2,
        );
        assert_eq!(schedule.placements(), &[1, 1]);
        assert_eq!(schedule.implementations(), &[DENSE_IMPLEMENTATION; 2]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &HeterogeneousBackendMove)
                .unwrap(),
            100,
        );
    }

    #[test]
    fn backend_refinement_prices_non_minimum_compute_implementations() {
        let bytes = one_node_graph();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        schedule
            .place(&graph, &PartitionAwareTargetImplementation)
            .unwrap();
        assert_eq!(schedule.placements(), &[0]);
        assert_eq!(schedule.implementations(), &[DENSE_IMPLEMENTATION]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &PartitionAwareTargetImplementation)
                .unwrap(),
            100,
        );

        assert_eq!(
            schedule
                .refine_regions(&graph, &PartitionAwareTargetImplementation)
                .unwrap(),
            1,
        );
        assert_eq!(schedule.placements(), &[1]);
        assert_eq!(schedule.implementations(), &[Implementation(1)]);
        assert_eq!(
            schedule
                .aggregate_cost(&graph, &PartitionAwareTargetImplementation)
                .unwrap(),
            20,
        );
    }

    #[test]
    fn supplied_allocator_owns_graph_and_schedule() {
        let tensors = [
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::INPUT,
                dimensions: &[1, 8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[1, 8],
                constant: &[],
            },
        ];
        let nodes = [NodeSpec {
            op: OpCode::IDENTITY,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(1)],
            attributes: &[],
        }];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
        assert_eq!(schedule.info().nodes, 1);
        assert_eq!(schedule.order(), &[0]);
        assert!(
            schedule
                .adjacency()
                .row(0)
                .unwrap()
                .iter()
                .all(|word| *word == 0)
        );
        assert_eq!(schedule.place(&graph, &OneBackend).unwrap(), 1);
        let plan = schedule.build_plan(&graph).unwrap();
        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.transfers.len(), 2);
        assert_eq!(plan.transfers[0].source, None);
        assert_eq!(plan.transfers[1].destination, None);
    }

    #[test]
    fn reassignment_invalidates_and_rebuilds_the_partition_plan() {
        let tensors = [
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::INPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
                constant: &[],
            },
        ];
        let nodes = [NodeSpec {
            op: OpCode::IDENTITY,
            flags: NodeFlags::default(),
            inputs: &[TensorId(0)],
            outputs: &[TensorId(1)],
            attributes: &[],
        }];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        assert_eq!(
            schedule.reassign_node(0, 1),
            Err(AllocationError::Placement(PlacementError::InvalidOrder))
        );
        schedule.place(&graph, &OneBackend).unwrap();
        assert_eq!(schedule.build_plan(&graph).unwrap().partitions.len(), 1);
        schedule.reassign_node(0, 1).unwrap();
        assert!(schedule.plan().partitions.is_empty());
        assert!(schedule.plan().transfers.is_empty());
        let plan = schedule.build_plan(&graph).unwrap();
        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.partitions[0].backend, 1);
        assert_eq!(
            schedule.reassign_node(1, 0),
            Err(AllocationError::Placement(PlacementError::InvalidOrder))
        );
        assert_eq!(
            schedule.reassign_node(0, UNPLACED),
            Err(AllocationError::Placement(PlacementError::InvalidOrder))
        );
    }

    #[test]
    fn aggregate_refinement_amortizes_one_launch_across_a_region() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[64],
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
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(2)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(3)],
                outputs: &[TensorId(4)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        schedule.place(&graph, &LaunchAmortized).unwrap();
        assert_eq!(schedule.placements(), &[0, 0, 0, 0]);
        assert_eq!(
            schedule.refine_regions(&graph, &LaunchAmortized).unwrap(),
            4
        );
        assert_eq!(schedule.placements(), &[1, 1, 1, 1]);
        // One accelerator launch + four compute units + input/output
        // transfers. This is the same aggregate score refinement optimized.
        assert_eq!(
            schedule.aggregate_cost(&graph, &LaunchAmortized).unwrap(),
            23
        );
        let plan = schedule.build_plan(&graph).unwrap();
        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.partitions[0].backend, 1);
    }

    #[test]
    fn optimized_placement_owns_the_complete_region_pipeline() {
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::default(),
                dimensions: &[64],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[64],
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
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(2)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(3)],
                outputs: &[TensorId(4)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

        let report = schedule.place_optimized(&graph, &LaunchAmortized).unwrap();
        assert_eq!(report.greedy_cost, 40);
        assert_eq!(report.regions_eliminated, 0);
        assert_eq!(report.nodes_reassigned, 4);
        assert_eq!(report.aggregate_cost, 23);
        assert_eq!(schedule.placements(), &[1, 1, 1, 1]);
        assert_eq!(schedule.build_plan(&graph).unwrap().partitions.len(), 1);
    }

    #[test]
    fn placement_clustering_coalesces_independent_backend_runs() {
        let tensors = [
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::INPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
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
            NodeSpec {
                op: OpCode::RELU,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::SIGMOID,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0)],
                outputs: &[TensorId(4)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
        schedule.place(&graph, &OpBackends).unwrap();
        assert_eq!(schedule.placements(), &[0, 1, 0, 1]);
        assert_eq!(schedule.build_plan(&graph).unwrap().partitions.len(), 4);

        assert_eq!(schedule.cluster_by_placement().unwrap(), 2);
        assert_eq!(schedule.order(), &[0, 2, 1, 3]);
        let plan = schedule.build_plan(&graph).unwrap();
        assert_eq!(plan.partitions.len(), 2);
        assert_eq!(plan.partitions[0].backend, 0);
        assert_eq!(plan.partitions[0].node_count, 2);
        assert_eq!(plan.partitions[1].backend, 1);
        assert_eq!(plan.partitions[1].node_count, 2);
    }

    #[test]
    fn placement_clustering_preserves_effect_serialization_edges() {
        let tensors = [
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::INPUT,
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags(0),
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags(0),
                dimensions: &[8],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F16,
                flags: TensorFlags::OUTPUT,
                dimensions: &[8],
                constant: &[],
            },
        ];
        let nodes = [
            NodeSpec {
                op: OpCode::RELU,
                flags: NodeFlags::ORDERED,
                inputs: &[TensorId(0)],
                outputs: &[TensorId(1)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::SIGMOID,
                flags: NodeFlags::ORDERED,
                inputs: &[TensorId(0)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
            NodeSpec {
                op: OpCode::RELU,
                flags: NodeFlags::ORDERED,
                inputs: &[TensorId(0)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
        ];
        let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
        schedule.place(&graph, &OpBackends).unwrap();
        assert_eq!(schedule.placements(), &[0, 1, 0]);

        assert_eq!(schedule.cluster_by_placement().unwrap(), 0);
        assert_eq!(schedule.order(), &[0, 1, 2]);
    }
}
