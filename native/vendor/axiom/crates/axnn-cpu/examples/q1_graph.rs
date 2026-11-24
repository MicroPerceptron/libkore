//! Build and schedule a small graph with a Q1 bit-packed adjacency matrix.
//!
//! Run:
//! `cargo run -p axnn-cpu --example q1_graph`

use axnn::{Q1, TensorElementAccess, indegree, shape, tensor, topo_sort};
use axnn_cpu::CpuBackend;

fn main() -> Result<(), axnn_cpu::CpuError> {
    // This graph is tiny, so a 4 KiB arena is plenty. Larger workloads can
    // either raise CAP or use CpuBackend::new_boxed() with the `std` feature.
    let be = CpuBackend::<4096>::new();

    // Q1 is bit-packed, but the shape is still logical: this is a 4x4 graph.
    let mut adj = tensor!(&be, Q1, shape![4, 4])?;

    // Diamond DAG: 0 -> {1, 2} -> 3.
    adj.set(&[0, 1], true.into())?;
    adj.set(&[0, 2], true.into())?;
    adj.set(&[1, 3], true.into())?;
    adj.set(&[2, 3], true.into())?;

    let mut ideg = [0u32; 4];
    let written = indegree(&adj, &mut ideg, &be)?;
    println!("indegree: {:?}", &ideg[..written]);

    let mut order = [0u16; 4];
    let written = topo_sort(&adj, &mut order, &be)?;
    println!("topological order: {:?}", &order[..written]);

    Ok(())
}
