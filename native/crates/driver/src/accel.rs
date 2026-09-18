//! Kore-native execution extensions over the published virtio-accel lifecycle.
//! Upstream owns graph artifacts, resource handles, and acceptance semantics.
//! This module owns the deployed Kore workgroup dialect and provider-wide
//! fault probe. Neither extends upstream capabilities or changes its wire ABI.

pub use virtio_accel_core::*;

/// Deployed Kore Query capability bit, confined to the native pair-ring wire.
/// Never insert this bit into upstream `DeviceInfo::capabilities`.
pub const EXPLICIT_WORKGROUPS: u64 = 1 << 6;

/// Kore's sealed Intel Xe native-kernel package.
///
/// Format 1 is the deployed VPUX ELF envelope. Format 2 deliberately carries
/// no Vulkan/SPIR-V promise: the payload is provider-validated Xe machine code
/// plus its execution ABI and generation identity.
pub const FORMAT_XE_NATIVE: ArtifactFormat = match ArtifactFormat::new(2) {
    Some(format) => format,
    None => unreachable!(),
};

// A future upstream assignment must trigger an explicit wire-compatibility review.
const _: () = assert!(Capabilities::all().bits() & EXPLICIT_WORKGROUPS == 0);

/// Legacy kore `(grid, group, SIMD)` submission geometry.
///
/// This type remains solely so deployed `OP_SUBMIT` peers keep their exact
/// source and wire contract. New portable code must use [`Workgroups`]: SIMD
/// width is a property of the compiled program, not a client-selected launch
/// parameter. A backend accepting a non-scalar legacy dispatch must prove the
/// requested width against that program; the default [`Accelerator`] method
/// rejects it rather than discarding or reinterpreting it.
#[deprecated(note = "legacy OP_SUBMIT compatibility only; use Workgroups and submit_with_launch")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    grid: [core::num::NonZeroU32; 3],
    group: [core::num::NonZeroU32; 3],
    simd_width: core::num::NonZeroU32,
}

#[allow(deprecated)]
impl Dispatch {
    /// A legacy dispatch of `grid` thread groups, each `group` work items, in
    /// SIMD lockstep of `simd_width` lanes.
    pub fn new(grid: [u32; 3], group: [u32; 3], simd_width: u32) -> Result<Self, BackendError> {
        let lift = |axes: [u32; 3]| -> Result<[core::num::NonZeroU32; 3], BackendError> {
            let mut out = [core::num::NonZeroU32::MIN; 3];
            for (slot, axis) in out.iter_mut().zip(axes) {
                *slot = core::num::NonZeroU32::new(axis).ok_or(BackendError::InvalidArgument)?;
            }
            Ok(out)
        };
        let simd = core::num::NonZeroU32::new(simd_width).ok_or(BackendError::InvalidArgument)?;
        if !simd.get().is_power_of_two() {
            return Err(BackendError::InvalidArgument);
        }
        Ok(Self {
            grid: lift(grid)?,
            group: lift(group)?,
            simd_width: simd,
        })
    }

    /// The legacy artifact-defined sentinel used by `OP_SUBMIT`.
    pub fn scalar() -> Self {
        Self {
            grid: [core::num::NonZeroU32::MIN; 3],
            group: [core::num::NonZeroU32::MIN; 3],
            simd_width: core::num::NonZeroU32::MIN,
        }
    }

    /// Whether this is the deployed artifact-defined sentinel.
    pub fn is_scalar(self) -> bool {
        self == Self::scalar()
    }

    /// The legacy thread-group grid `[x, y, z]`.
    pub fn grid(self) -> [u32; 3] {
        self.grid.map(core::num::NonZeroU32::get)
    }

    /// The legacy intra-group work-item shape `[x, y, z]`.
    pub fn group(self) -> [u32; 3] {
        self.group.map(core::num::NonZeroU32::get)
    }

    /// The client-requested legacy SIMD/lane width.
    pub fn simd_width(self) -> u32 {
        self.simd_width.get()
    }
}

/// Explicit workgroup geometry for a data-parallel launch.
///
/// Kore native extension, separate from the upstream graph contract. This names only the portable facts a
/// client may lawfully choose: the number of workgroups and each workgroup's
/// logical size. SIMD/subgroup width is deliberately absent because it is a
/// provider compilation and scheduling decision, not portable client input.
/// Device-specific ceilings belong to the backend and are checked at admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workgroups {
    group_count: [core::num::NonZeroU32; 3],
    local_size: [core::num::NonZeroU32; 3],
}

impl Workgroups {
    /// Construct an explicit workgroup launch. Every axis must be nonzero.
    pub fn new(group_count: [u32; 3], local_size: [u32; 3]) -> Result<Self, BackendError> {
        let lift = |axes: [u32; 3]| -> Result<[core::num::NonZeroU32; 3], BackendError> {
            let mut out = [core::num::NonZeroU32::MIN; 3];
            for (slot, axis) in out.iter_mut().zip(axes) {
                *slot = core::num::NonZeroU32::new(axis).ok_or(BackendError::InvalidArgument)?;
            }
            Ok(out)
        };
        Ok(Self {
            group_count: lift(group_count)?,
            local_size: lift(local_size)?,
        })
    }

    /// Number of workgroups in each dimension.
    pub fn group_count(self) -> [u32; 3] {
        self.group_count.map(core::num::NonZeroU32::get)
    }

    /// Logical work-item shape of each workgroup.
    pub fn local_size(self) -> [u32; 3] {
        self.local_size.map(core::num::NonZeroU32::get)
    }
}

/// How a submitted artifact chooses its logical launch shape.
///
/// Artifact-defined execution is not encoded as a magic `1x1x1` launch: an
/// explicit one-work-item launch remains observably distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchShape {
    /// The loaded artifact/provider owns all launch-shape decisions.
    ArtifactDefined,
    /// The client requests portable logical workgroup geometry.
    Workgroups(Workgroups),
}

/// Native execution hooks alongside the unmodified upstream lifecycle.
/// The host probes health at each resource-admission boundary. DeviceLost
/// retires the complete provider epoch; other errors refuse that admission.
/// Geometry defaults reject requests they cannot honor. Handles and failures
/// are the exact upstream types, so conformance and DeviceState stay shared.
pub trait NativeAccelerator: Accelerator {
    /// Immutable support for the Kore explicit-workgroup opcode.
    const EXPLICIT_WORKGROUPS: bool = false;

    fn validate_launch_shape(&self, launch: LaunchShape) -> Result<(), BackendError> {
        match launch {
            LaunchShape::ArtifactDefined => Ok(()),
            LaunchShape::Workgroups(_) if Self::EXPLICIT_WORKGROUPS => Ok(()),
            LaunchShape::Workgroups(_) => Err(BackendError::Unsupported),
        }
    }

    /// Bounded, nonblocking provider-wide fault probe; never drives work.
    fn poll_health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// The packed device-plane witness of a reported `DeviceLost` (the
    /// provider's own layout; `0` when none was retained). A sealed epoch's
    /// refusals may carry it so a serial-less client can name the fault
    /// instead of only the loss. Providers with no device witness keep the
    /// default.
    fn loss_evidence(&self) -> u64 {
        0
    }

    /// Copy the newest bytes of the device firmware's own log into `out`,
    /// returning how many were written (the tail is right-aligned to what
    /// fits). A provider without a readable firmware log keeps the default
    /// and writes nothing.
    fn firmware_log(&self, out: &mut [u8]) -> usize {
        let _ = out;
        0
    }

    /// Preserve the deployed scalar sentinel; reject unproved grid/group/SIMD.
    #[allow(deprecated)]
    #[deprecated(note = "legacy OP_SUBMIT compatibility only; use submit_with_launch")]
    fn submit_dispatch(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        dispatch: Dispatch,
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        if dispatch.is_scalar() {
            self.submit(queue, program, bindings, timeout)
        } else {
            Err(SubmitFailure::Rejected(BackendError::Unsupported))
        }
    }

    /// Honor the requested geometry or reject it before acceptance.
    fn submit_with_launch(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        launch: LaunchShape,
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        match launch {
            LaunchShape::ArtifactDefined => self.submit(queue, program, bindings, timeout),
            LaunchShape::Workgroups(_) => Err(SubmitFailure::Rejected(BackendError::Unsupported)),
        }
    }
}

/// Explicitly host an upstream graph provider with artifact-defined launches.
/// Faults remain observable through its ordinary upstream operations. Native
/// hardware with out-of-band faults implements `NativeAccelerator` directly.
#[derive(Default)]
pub struct ArtifactBackend<A>(pub A);

impl<A> core::ops::Deref for ArtifactBackend<A> {
    type Target = A;
    fn deref(&self) -> &A {
        &self.0
    }
}

impl<A: Accelerator> NativeAccelerator for ArtifactBackend<A> {}

impl<A: Accelerator> Accelerator for ArtifactBackend<A> {
    type Context = A::Context;
    type Buffer = A::Buffer;
    type Program = A::Program;
    type Queue = A::Queue;
    type Event = A::Event;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        self.0.device_info()
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.0.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.0.destroy_context(context)
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.0.allocate_buffer(context, desc)
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        self.0.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        self.0.read_buffer(buffer, offset, data)
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.0.free_buffer(buffer)
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        self.0.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.0.unload_program(program)
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.0.create_queue(context, desc)
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.0.destroy_queue(queue)
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.0.submit(queue, program, bindings, timeout)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        self.0.poll_event(event)
    }

    fn cancel_event(&self, _event: &Self::Event) -> Result<(), BackendError> {
        self.0.cancel_event(_event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.0.destroy_event(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(deprecated)]
    fn launch_geometries_keep_legacy_and_portable_contracts_distinct() {
        let legacy = Dispatch::new([16, 8, 1], [64, 1, 1], 16).unwrap();
        assert_eq!(legacy.grid(), [16, 8, 1]);
        assert_eq!(legacy.group(), [64, 1, 1]);
        assert_eq!(legacy.simd_width(), 16);
        assert!(!legacy.is_scalar());
        assert!(Dispatch::scalar().is_scalar());
        assert_eq!(
            Dispatch::new([0, 1, 1], [1, 1, 1], 8),
            Err(BackendError::InvalidArgument)
        );
        assert_eq!(
            Dispatch::new([1, 1, 1], [1, 1, 1], 24),
            Err(BackendError::InvalidArgument)
        );

        let groups = Workgroups::new([16, 8, 1], [64, 1, 1]).unwrap();
        assert_eq!(groups.group_count(), [16, 8, 1]);
        assert_eq!(groups.local_size(), [64, 1, 1]);
        assert_eq!(
            Workgroups::new([0, 1, 1], [1, 1, 1]),
            Err(BackendError::InvalidArgument)
        );
        assert_eq!(
            Workgroups::new([1, 1, 1], [1, 0, 1]),
            Err(BackendError::InvalidArgument)
        );
        assert_ne!(
            LaunchShape::ArtifactDefined,
            LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap())
        );
    }
}
