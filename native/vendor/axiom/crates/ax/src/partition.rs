//! Convert node placement into backend-sized executable regions and transfers.

use crate::format::{Graph, TensorFlags, TensorId};
use crate::schedule::{DENSE_IMPLEMENTATION, Implementation, NO_PRODUCER, UNPLACED};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Partition {
    pub backend: u16,
    /// Adapter-owned implementation for every node in this contiguous region.
    /// This does not affect tensor residency or transfer accounting.
    pub implementation: Implementation,
    /// Start in the topological order array, not the serialized node table.
    pub order_start: u32,
    pub node_count: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Transfer {
    pub tensor: TensorId,
    /// `None` denotes the external tensor domain.
    pub source: Option<u16>,
    /// `None` denotes the external tensor domain.
    pub destination: Option<u16>,
    /// Transfer completes before this partition. `partition_count` means
    /// graph egress after the final partition.
    pub before_partition: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartitionInfo {
    pub partitions: usize,
    pub transfers: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionError {
    ScratchTooSmall,
    InvalidOrder,
    InvalidProducer,
    UnplacedNode(u32),
    CountOverflow,
}

#[derive(Clone, Copy)]
pub struct PlanView<'a> {
    pub partitions: &'a [Partition],
    pub transfers: &'a [Transfer],
}

/// Groups consecutive, equally placed nodes in topological order.
///
/// Contiguous grouping deliberately preserves an acyclic partition order.
/// Allocator-owned schedules may first use
/// [`Schedule::cluster_by_placement`](crate::Schedule::cluster_by_placement)
/// to topologically cluster independent same-backend work; this function then
/// keeps the resulting provider regions contiguous and dependency-safe.
pub fn partition(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &[u16],
    node_partitions: &mut [u32],
    partitions: &mut [Partition],
    transfers: &mut [Transfer],
) -> Result<PartitionInfo, PartitionError> {
    partition_impl(
        graph,
        order,
        producers,
        placements,
        None,
        node_partitions,
        partitions,
        transfers,
    )
}

/// Groups placed nodes while retaining an adapter-owned implementation ID.
///
/// The legacy [`partition`] entry point remains dense-only for executors that
/// do not know about implementation variants yet.
pub fn partition_with_implementations(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &[u16],
    implementations: &[Implementation],
    node_partitions: &mut [u32],
    partitions: &mut [Partition],
    transfers: &mut [Transfer],
) -> Result<PartitionInfo, PartitionError> {
    partition_impl(
        graph,
        order,
        producers,
        placements,
        Some(implementations),
        node_partitions,
        partitions,
        transfers,
    )
}

#[allow(clippy::too_many_arguments)]
fn partition_impl(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    placements: &[u16],
    implementations: Option<&[Implementation]>,
    node_partitions: &mut [u32],
    partitions: &mut [Partition],
    transfers: &mut [Transfer],
) -> Result<PartitionInfo, PartitionError> {
    let nodes = graph.node_count();
    if order.len() < nodes
        || placements.len() < nodes
        || implementations.is_some_and(|implementations| implementations.len() < nodes)
        || producers.len() < graph.tensor_count()
        || node_partitions.len() < nodes
        || partitions.len() < nodes
    {
        return Err(PartitionError::ScratchTooSmall);
    }
    node_partitions[..nodes].fill(u32::MAX);

    let mut partition_count = 0_usize;
    for (position, &node) in order[..nodes].iter().enumerate() {
        let node = node as usize;
        if node >= nodes {
            return Err(PartitionError::InvalidOrder);
        }
        if node_partitions[node] != u32::MAX {
            return Err(PartitionError::InvalidOrder);
        }
        let backend = placements[node];
        let implementation = implementations
            .map(|implementations| implementations[node])
            .unwrap_or(DENSE_IMPLEMENTATION);
        if backend == UNPLACED || implementation.0 == u16::MAX {
            return Err(PartitionError::UnplacedNode(node as u32));
        }
        let extend = partition_count != 0
            && partitions[partition_count - 1].backend == backend
            && partitions[partition_count - 1].implementation == implementation
            && partitions[partition_count - 1].order_start as usize
                + partitions[partition_count - 1].node_count as usize
                == position;
        if extend {
            partitions[partition_count - 1].node_count += 1;
        } else {
            partitions[partition_count] = Partition {
                backend,
                implementation,
                order_start: u32::try_from(position).map_err(|_| PartitionError::CountOverflow)?,
                node_count: 1,
            };
            partition_count += 1;
        }
        node_partitions[node] = (partition_count - 1) as u32;
    }

    let mut transfer_count = 0_usize;
    for (partition_index, region) in partitions[..partition_count].iter().enumerate() {
        let start = region.order_start as usize;
        let end = start + region.node_count as usize;
        for &node_index in &order[start..end] {
            let node = graph
                .node(node_index as usize)
                .map_err(|_| PartitionError::InvalidOrder)?;
            for input in node.inputs() {
                let producer = producers[input.0 as usize];
                let source = if producer == NO_PRODUCER {
                    None
                } else {
                    let producer = producer as usize;
                    if producer >= nodes || placements[producer] == UNPLACED {
                        return Err(PartitionError::InvalidProducer);
                    }
                    Some(placements[producer])
                };
                if source == Some(region.backend)
                    || transfers[..transfer_count].iter().any(|transfer| {
                        transfer.tensor == input && transfer.destination == Some(region.backend)
                    })
                {
                    continue;
                }
                push_transfer(
                    transfers,
                    &mut transfer_count,
                    Transfer {
                        tensor: input,
                        source,
                        destination: Some(region.backend),
                        before_partition: partition_index as u32,
                    },
                )?;
            }
        }
    }

    for (tensor_index, &producer) in producers[..graph.tensor_count()].iter().enumerate() {
        let tensor = graph
            .tensor(TensorId(tensor_index as u32))
            .map_err(|_| PartitionError::InvalidProducer)?;
        if !tensor.flags.contains(TensorFlags::OUTPUT) {
            continue;
        }
        if producer == NO_PRODUCER {
            // External inputs and constants that are also outputs already live
            // in the external domain and need no egress transfer.
            continue;
        }
        let producer = producer as usize;
        if producer >= nodes || placements[producer] == UNPLACED {
            return Err(PartitionError::InvalidProducer);
        }
        push_transfer(
            transfers,
            &mut transfer_count,
            Transfer {
                tensor: tensor.id,
                source: Some(placements[producer]),
                destination: None,
                before_partition: partition_count as u32,
            },
        )?;
    }

    Ok(PartitionInfo {
        partitions: partition_count,
        transfers: transfer_count,
    })
}

fn push_transfer(
    transfers: &mut [Transfer],
    count: &mut usize,
    transfer: Transfer,
) -> Result<(), PartitionError> {
    let slot = transfers
        .get_mut(*count)
        .ok_or(PartitionError::ScratchTooSmall)?;
    *slot = transfer;
    *count += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{NodeFlags, NodeSpec, OpCode, TensorSpec, encode};
    use crate::schedule::{DENSE_IMPLEMENTATION, Implementation};
    use axnn::DType;
    use std::vec;

    #[test]
    fn placed_chain_becomes_regions_and_boundary_transfers() {
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
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(2)],
                outputs: &[TensorId(3)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0; 512];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        let mut node_partitions = [0; 3];
        let mut regions = [Partition::default(); 3];
        let mut transfers = [Transfer::default(); 8];
        let info = partition(
            &graph,
            &[0, 1, 2],
            &[NO_PRODUCER, 0, 1, 2],
            &[0, 1, 1],
            &mut node_partitions,
            &mut regions,
            &mut transfers,
        )
        .unwrap();
        assert_eq!(
            info,
            PartitionInfo {
                partitions: 2,
                transfers: 3
            }
        );
        assert_eq!(regions[0].node_count, 1);
        assert_eq!(regions[1].node_count, 2);
        assert_eq!(transfers[0].source, None);
        assert_eq!(transfers[0].destination, Some(0));
        assert_eq!(transfers[1].source, Some(0));
        assert_eq!(transfers[1].destination, Some(1));
        assert_eq!(transfers[2].source, Some(1));
        assert_eq!(transfers[2].destination, None);
    }

    #[test]
    fn implementations_split_partitions_without_creating_transfers() {
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
                op: OpCode::IDENTITY,
                flags: NodeFlags::default(),
                inputs: &[TensorId(1)],
                outputs: &[TensorId(2)],
                attributes: &[],
            },
        ];
        let mut bytes = vec![0; 512];
        let len = encode(&mut bytes, &tensors, &nodes).unwrap();
        let graph = Graph::parse(&bytes[..len]).unwrap();
        let mut node_partitions = [0; 2];
        let mut regions = [Partition::default(); 2];
        let mut transfers = [Transfer::default(); 4];
        let info = partition_with_implementations(
            &graph,
            &[0, 1],
            &[NO_PRODUCER, 0, 1],
            &[0, 0],
            &[DENSE_IMPLEMENTATION, Implementation(1)],
            &mut node_partitions,
            &mut regions,
            &mut transfers,
        )
        .unwrap();
        assert_eq!(info.partitions, 2);
        assert_eq!(info.transfers, 2, "only graph ingress and egress remain");
        assert_eq!(regions[0].implementation, DENSE_IMPLEMENTATION);
        assert_eq!(regions[1].implementation, Implementation(1));
    }
}
