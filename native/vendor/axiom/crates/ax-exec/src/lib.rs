//! Portable execution shared by host provider adapters and freestanding runtimes.
//!
//! No device discovery, thread, clock, provider compiler, or default allocator
//! lives here. The caller supplies allocation and the admitted CPU policy.
//! Runtimes may execute a prepared partition directly or place it behind a
//! provider queue; tenancy does not create a second graph compiler. Worker and
//! tile scheduling remain caller responsibilities.
#![no_std]
#![feature(allocator_api)]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod continuation;
#[doc(hidden)]
pub mod native;
mod prepared;
mod program;
#[doc(hidden)]
pub mod storage;
pub use axnn_cpu::{ForkJoin, Serial};
pub use continuation::{
    ChainedInput, FusedRun, KernelProgress, KvOutput, MAX_ADMISSION_WINDOW, PreparedInvocation,
    PreparedKernelExecutor, PreparedProgress, RecurrentState, ResidentKvCache,
};
pub use prepared::PreparedNative;

pub use program::{NativeInvocation, NativeProgram, NativeProgress};

/// Owned typed tensor payload; moving it preserves allocation and alignment.
pub use native::NativeValue as NativeTensor;
pub use native::{
    AxCpuKernelPolicy, AxHostCopyStats, AxNativePlanStats, CPU_KERNEL_IMPLEMENTATION,
    NativeResourceBinding, NativeResourceKind, ProjectionExecutor,
};
pub use native::{
    CHECK_OFFENDERS, CHECK_THRESHOLD, CheckOffender, CheckReport, CpuProfile, OP_CLASS_NAMES,
    OP_CLASSES, OpProfile,
};
pub use storage::{AxOutput, AxTensor, NativeRef};

/// Structural native-plan error, independent of provider artifact authoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    InvalidGraph,
    InvalidPartition,
    NoPartitionOutputs,
}

/// Portable execution failure. Host adapters preserve their existing error ABI
/// by mapping these variants, without depending on host artifact/compiler types.
#[derive(Debug)]
pub enum Error {
    Ax(ax::AllocationError),
    Plan(PlanError),
    Backend(virtio_accel_core::BackendError),
    NativeRequiresSinglePartition,
    InvalidInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidInputSize {
        tensor: ax::TensorId,
        expected: usize,
        actual: usize,
    },
    MissingResource(axnn::lowering::ResourceKey),
    /// Resident state has no CPU mirror. An executor must accept the operation.
    ResidentResourceRequiresExecutor(axnn::lowering::ResourceKey),
    DuplicateResource(axnn::lowering::ResourceKey),
    ResourceKind {
        resource: axnn::lowering::ResourceKey,
        expected: NativeResourceKind,
        actual: NativeResourceKind,
    },
    ResourceDimensions {
        resource: axnn::lowering::ResourceKey,
        expected: [u32; 3],
        actual: [u32; 3],
    },
    ResourceStorage {
        resource: axnn::lowering::ResourceKey,
        expected: axnn::lowering::VolumeRgbStorage,
        actual: axnn::lowering::VolumeRgbStorage,
    },
    ResourceLength {
        resource: axnn::lowering::ResourceKey,
        expected: usize,
        actual: usize,
    },
    /// A bound key/value cache's geometry or storage differs from the node's.
    KvCacheGeometry {
        resource: axnn::lowering::ResourceKey,
        expected: axnn::lowering::KvCacheParams,
        actual: axnn::lowering::KvCacheParams,
    },
    MissingTensor(ax::TensorId),
    UnsupportedImplementation {
        backend: u16,
        implementation: ax::Implementation,
    },
    Cpu(axnn_cpu::CpuError),
    InvocationFinished,
    InvalidStepBudget,
}

impl From<ax::AllocationError> for Error {
    fn from(value: ax::AllocationError) -> Self {
        Self::Ax(value)
    }
}
impl From<PlanError> for Error {
    fn from(value: PlanError) -> Self {
        Self::Plan(value)
    }
}
impl From<virtio_accel_core::BackendError> for Error {
    fn from(value: virtio_accel_core::BackendError) -> Self {
        Self::Backend(value)
    }
}
impl From<axnn_cpu::CpuError> for Error {
    fn from(value: axnn_cpu::CpuError) -> Self {
        Self::Cpu(value)
    }
}
