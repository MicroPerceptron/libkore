#![feature(allocator_api)]
#![feature(f16)]
//! axnn-vaccel — axiom ⇄ virtio-accel adapter, both directions.
//!
//! virtio-accel proved that heterogeneous NPUs (Core ML/ANE, OpenVINO,
//! Hexagon) can share one input form — TOSA — behind one provider
//! contract, [`virtio_accel_core::Accelerator`]. This crate plugs axiom
//! into that seam from both sides:
//!
//! - **Provider** — [`AxTosaAccelerator`] runs TOSA artifacts on the
//!   axnn-cpu SIMD kernels via the `axtosa` executor. It is the backend
//!   that *accepts* the F8 extension tier every current NPU provider
//!   rejects, and it gives any `Accelerator` consumer a fully-executing
//!   CPU device on every host (no NPU required).
//! - **Consumer** — [`artifact`] builds TOSA 1.0 artifacts through the
//!   typed, validating `virtio-accel-tosa-build` authoring surface,
//!   [`Session`] drives any provider through the load/bind/submit/read
//!   cycle with a shape-keyed program cache, and [`Hetero`] routes each
//!   call between a CPU session and an accelerator session by
//!   capability first (F8 → CPU the moment the NPU rejects it) and
//!   problem size second.
//! - **Graph scheduler adapter** — [`AxHetero::compile_auto_in`] discovers
//!   provider capabilities, selects placement and fuses exact native patterns
//!   without a backend-specific model. [`AxHetero::compile_in`] retains explicit
//!   cost and policy controls. Both compile placed `.ax` partitions, keeping executable
//!   metadata in the caller's persistent allocator; [`AxHetero::run_in`]
//!   accepts a separate allocator for transient tensors and outputs, while
//!   [`AxHetero::begin_in`] and [`AxHetero::advance_in`] expose the same mixed
//!   schedule as bounded event-loop transitions without a worker thread.
//!   [`AxHetero::advance_retained_in`] returns [`AxTensor`] handles without
//!   forcing typed results into byte vectors. [`AxHetero::advance_into_in`]
//!   and [`AxHetero::run_into_in`] accept borrowed [`AxOutput`] destinations,
//!   including pitched surfaces. Terminal provider results can read directly
//!   into these sinks; native results perform one final host write. Neither
//!   path implies external device-memory import or removes provider transfers.
//!   [`AxHetero::run_repeated_auto_in`] selects provider-resident feedback when
//!   possible, otherwise retaining host tensor backing across native/mixed runs.
//!   [`AxDispatchVariant`] binds calibrated admission metadata to that final
//!   placement, and [`admit_variants`] returns the exact executable selected by
//!   the caller-owned capacity planner.
//!
//! The same `Session` code path that dispatches to [`AxTosaAccelerator`]
//! dispatches to Core ML's Neural Engine with `coreml`, Intel NPU/GPU
//! devices through OpenVINO with `openvino`, or any Vulkan 1.3 compute
//! device (its FP32 and FP16 TOSA tiers) with `vulkan`. [`blob`] records
//! what such a provider compiled, hash-bound, for runtimes that only load
//! offline-compiled programs. That symmetry is the point:
//! kore's accelerator framework can slot axiom in as one more provider
//! row, and axiom can burn work down to NPUs through the identical
//! contract.

pub mod artifact;
pub mod ax_graph;
pub mod blob;
mod device_set;
pub use device_set::{AxDeviceSet, AxTransferCost};
mod lowering;
#[cfg(feature = "metal")]
pub mod metal;
pub mod provider;
pub mod session;
mod session_erased;
pub mod tensor_parallel;
#[cfg(all(feature = "vulkan", axnn_vaccel_vulkan_runtime))]
pub mod vulkan;
#[cfg(feature = "xdna")]
pub mod xdna;

pub use artifact::MatMulDType;
pub use ax_graph::{
    ACCEL_BACKEND, AxCompileError, AxCpuKernelPolicy, AxDeviceModel, AxDispatchChoice,
    AxDispatchVariant, AxExecutable, AxFeedback, AxHetero, AxHostCopyStats, AxInvocation,
    AxInvocationStatus, AxNativePlanStats, AxOutput, AxPartitionPhase, AxPartitionProgress,
    AxSparseBsrPolicy, AxStream, AxStreamProgress, AxStreamSubmitError, AxTensor, CPU_BACKEND,
    CPU_BSR_B8_IMPLEMENTATION, CPU_BSR_F8_IMPLEMENTATION, CPU_BSR_F32_IMPLEMENTATION,
    CPU_KERNEL_IMPLEMENTATION, NativeResourceBinding, NativeResourceKind, PreparedNative,
    admit_variants,
};
pub use axnn_cpu::{BrickGridRef, RgbVolumeRef, U8BufferRef};
pub use provider::AxTosaAccelerator;
pub use session::{
    ExecutionTimings, FeedbackBinding, Hetero, Invocation, InvocationStatus, InvocationSubmitError,
    ProfiledExecution, RepeatedExecution, Route, Session,
};

// The contract vocabulary consumers need to hold sessions and interpret
// errors, re-exported so downstream code needs no direct core dep.
pub use virtio_accel_core::{Accelerator, BackendError};
pub use virtio_accel_tosa::Target;
