//! `ax` — the backend-agnostic tensor graph language and scheduler core.
//!
//! `.ax` files contain tensor metadata, constants, operators, and value
//! references in a canonical little-endian container. Parsing is borrowed and
//! the lowest-level scheduler uses caller-provided storage. Allocator-owned
//! conveniences take an explicit nightly [`core::alloc::Allocator`]; `ax`
//! never assumes a global allocator. The crate has no host, OS, provider, or
//! device-runtime dependency.
//!
//! Device integrations live above this crate. They implement the placement
//! cost model, compile placed regions, and manage their own buffers/events.

#![no_std]
#![feature(allocator_api)]
#![feature(portable_simd)]

extern crate alloc;

pub use axnn::DType;

#[cfg(test)]
extern crate std;

pub mod bitmat;
pub mod buffer;
pub mod builder;
pub mod context_kv;
pub mod context_partial_i8;
pub mod dispatch;
pub mod dspark_lm_head_i8;
pub mod dspark_markov_feedback;
pub mod format;
pub mod graphics;
pub mod kv;
#[cfg(feature = "kernel-lowering")]
pub mod lowering;
pub mod mlp_down_residual_i8;
pub mod mlp_gate_up_i8;
pub mod moe;
pub mod owned;
pub mod partition;
pub mod qkv_prep;
pub mod qkv_prep_i8;
pub mod raster;
pub mod recurrent;
pub mod routed;
pub mod sampling;
pub mod schedule;
pub mod sink_gqa;
pub mod sink_gqa_o_residual;
pub mod tensor_ops;
pub mod voxel;

pub use builder::GraphBuilder;
pub use dispatch::{
    AdmissionError, AdmissionPlanner, DispatchCandidate, DispatchDecision, DispatchLane,
    DispatchRequest,
};
pub use format::{
    AxError, DYNAMIC_DIM, Graph, NodeFlags, NodeSpec, NodeView, OpCode, TensorFlags, TensorId,
    TensorSpec, TensorView, encode, encoded_len,
};
pub use graphics::{LinearizedField2d, PremultipliedRgba};
#[cfg(feature = "kernel-lowering")]
pub use lowering::node_kernel_request;
pub use owned::{AllocationError, GraphBuf, PlacementReport, Schedule};
pub use partition::{
    Partition, PartitionError, PartitionInfo, PlanView, Transfer, partition,
    partition_with_implementations,
};
pub use raster::{RASTER_NODE_FLAGS, RasterNodeParams, RasterOp, raster_op, raster_opcode};
pub use sampling::RgbSample;
pub use schedule::{
    CostModel, DENSE_IMPLEMENTATION, Implementation, PlacementError, ScheduleError, ScheduleInfo,
    ScheduleScratch, UNSELECTED_IMPLEMENTATION, analyze, place, place_with_implementations,
};
pub use voxel::VoxelHit;
