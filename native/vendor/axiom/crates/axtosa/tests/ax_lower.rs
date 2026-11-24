#![feature(allocator_api)]

use ax::{OpCode, Schedule};
use std::alloc::Global;
use virtio_accel_conformance::numerics;

#[test]
fn lowers_fused_classifier_shape_into_ax() {
    let bytes =
        axtosa::lower_ax_in(numerics::MOCK_LINEAR_CLASSIFIER_FP16.artifact, Global).unwrap();
    let graph = bytes.parse().unwrap();
    assert!(graph.node_count() >= 1);
    assert!(
        (0..graph.node_count())
            .map(|index| graph.node(index).unwrap().op)
            .any(|op| op == OpCode::MATMUL)
    );
    let schedule = Schedule::analyze_in(&graph, Global).unwrap();
    assert_eq!(schedule.info().nodes, graph.node_count());
}

#[test]
fn lowers_attribute_free_conformance_graphs() {
    for artifact in [
        numerics::IDENTITY_EDGES_FP16.artifact,
        numerics::ADD_FP16.artifact,
        numerics::MATMUL_FP16.artifact,
        unary_case("ceil-fp16"),
        unary_case("floor-fp16"),
    ] {
        let bytes = axtosa::lower_ax_in(artifact, Global).unwrap();
        let graph = bytes.parse().unwrap();
        let schedule = Schedule::analyze_in(&graph, Global).unwrap();
        assert_eq!(schedule.info().nodes, graph.node_count());
    }
}

#[test]
fn lowers_tosa_rounding_ops_to_canonical_ax_ops() {
    for (name, expected) in [("ceil-fp16", OpCode::CEIL), ("floor-fp16", OpCode::FLOOR)] {
        let bytes = axtosa::lower_ax_in(unary_case(name), Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.node_count(), 1, "{name}");
        assert_eq!(graph.node(0).unwrap().op, expected, "{name}");
    }
}

fn unary_case(name: &str) -> &'static [u8] {
    numerics::HEXAGON_UNARY_FP16_CASES
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing conformance case {name}"))
        .artifact
}
