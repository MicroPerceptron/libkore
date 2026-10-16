//! Kore-native execution extensions over the published virtio-accel lifecycle.
//! Upstream owns graph artifacts, resource handles, and acceptance semantics.
//! This module owns the deployed Kore workgroup dialect and provider-wide
//! fault probe. Neither extends upstream capabilities or changes its wire ABI.

pub use virtio_accel_core::*;

/// Opaque, provider-defined evidence for a sealed device epoch.
/// The host forwards both fields through the existing External refusal without
/// interpreting or masking the code. An i64 carries all 64 diagnostic bits;
/// signedness is part of the provider's layout. This is data, not authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LossEvidence {
    pub domain: u32,
    pub code: i64,
}

/// One member of a provider-ordered execution plan.
///
/// Programs remain independent lifecycle objects. The event returned for the
/// complete plan retains every program and bound buffer until retirement.
/// Backends that can encode an ordered hardware command list execute the slice
/// without a host completion round trip between members.
///
/// Each member carries its own [`LaunchShape`], admitted exactly as a single
/// [`NativeAccelerator::submit_with_launch`] of that program would admit it,
/// and one ordering edge to the members before it:
///
/// * `barrier == true` — the member starts only after every earlier member's
///   memory writes are complete and visible to it (a data dependency);
/// * `barrier == false` — the client declares the member independent of the
///   members since the most recent barrier, so a backend may overlap it with
///   them. It still never starts ahead of a barrier that precedes it.
///
/// The first member's flag is ignored: nothing in the plan precedes it. A
/// backend that executes every member strictly in order honors both values;
/// the flag grants permission to overlap, never an obligation to.
pub struct OrderedExecution<'a, P, B> {
    pub program: &'a P,
    pub bindings: &'a [BindingRef<'a, B>],
    pub launch: LaunchShape,
    pub barrier: bool,
    /// Before the member, wait until a host-published word reaches a
    /// value (a gate, [`SyncRef`]). A backend with no command-stream wait
    /// refuses a list that names one.
    pub gate: Option<SyncRef<'a, B>>,
    /// Once the member retires and its writes are visible to the host,
    /// write a value to a word (a signal, [`SyncRef`]). A backend with no
    /// such completion write refuses a list that names one.
    pub signal: Option<SyncRef<'a, B>>,
}

/// A gate's or signal's word: the 8-byte-aligned quadword at `offset` in a
/// buffer the queue's context holds (`res/QWEN_EXPERT_STREAMING.md` §3-4).
/// A gate holds its member until the word's low dword is at least `value`
/// (unsigned); a signal writes `value`, zero-extended, after its member's
/// writes are visible to the host. The host and the device meet on the
/// word alone: nothing else of the buffer is bound.
pub struct SyncRef<'a, B> {
    pub buffer: &'a B,
    pub offset: u64,
    pub value: u64,
}

impl<B> Clone for SyncRef<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for SyncRef<'_, B> {}

/// One accelerator's DMA demand, stated per frame class
/// (`res/ACCEL_MEMORY_SCALE.md` §2.2).
///
/// A profile is data the backend declares once, and it replaces the
/// constants that used to sit in each service: the founder stages the
/// `structural` and `firmware` demands as the row's DMA pool (from the
/// kernel's 2 MiB class), and the service derives its retained-byte budgets
/// (`virtio_accel_device::ResourcePolicy`) from `buffers` and `programs`,
/// which arrive as client-owned leases and are never founder-staged.
///
/// Classes are named by the lease alignment vocabulary the kernel already
/// routes on (`kore_abi::session::LEASE_ALIGN_*`); nothing new crosses a
/// wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaProfile {
    /// Rings, CTBs, MMU/GuC tables, IPC, event arenas: small, fixed,
    /// founder-staged.
    pub structural: ClassDemand,
    /// Resident firmware and its runtime: founder-staged.
    pub firmware: ClassDemand,
    /// The client working set: grows on demand up to `ceiling`.
    pub buffers: GrowDemand,
    /// Resident programs: grows on demand up to `ceiling`.
    pub programs: GrowDemand,
}

/// A fixed demand in one class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassDemand {
    /// `kore_abi::session::LEASE_ALIGN_4K` / `_2M` / `_1G`.
    pub align_log2: u8,
    pub bytes: u64,
}

/// A demand that is funded incrementally by client supplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowDemand {
    /// The class a funding supply should be leased in; the kernel falls to
    /// the next finer class when it is exhausted, so this is a preference
    /// with a measured miss, never a refusal.
    pub align_log2: u8,
    /// Bytes the service may keep resident across every client.
    pub ceiling: u64,
}

impl DmaProfile {
    /// The pool the founder must deliver for this backend to found at all:
    /// the two fixed demands.  A backend checks its delivered pool against
    /// this at claim time so a manifest that under-funds the row refuses by
    /// name instead of dying at bring-up rung N with "pool exhausted".
    pub const fn founded_bytes(&self) -> u64 {
        self.structural.bytes.saturating_add(self.firmware.bytes)
    }
}

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

    /// The first retained device-plane witness of a reported DeviceLost.
    /// None means no witness; Some with a zero code is valid evidence.
    /// Providers own the domain and bit layout; hosts must preserve both.
    fn loss_evidence(&self) -> Option<LossEvidence> {
        None
    }

    /// Format retained failure metadata before the host hands off the lost
    /// epoch. Must not allocate, touch registers, wait, or submit device work.
    /// Device-domain addresses are permitted; host physical addresses and
    /// buffer contents are not. Truncation is controlled by the host's writer.
    fn failure_report(&self, _out: &mut dyn core::fmt::Write) {}

    /// Bounded operational telemetry for a diagnostic reader — clock, device
    /// time, completion wakes — formatted from retained counters: no MMIO,
    /// waits or DMA access. The default reports nothing.
    fn telemetry_report(&self, _out: &mut dyn core::fmt::Write) {}

    /// Configure optional execution timing for one owned, quiescent context.
    /// Enabling resets its measurement interval; disabling retains the result.
    /// This must not change submission, completion or visibility semantics.
    fn configure_profiling(
        &self,
        _context: &Self::Context,
        _enabled: bool,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported)
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

    /// Submit a non-empty, dependency-ordered program list as one event.
    ///
    /// The default rejects the native extension. A backend accepting it must
    /// preserve slice order, honor every member's
    /// [`barrier`](OrderedExecution::barrier), and signal completion only
    /// after every member is complete. A member whose launch shape the
    /// backend cannot honor rejects the whole list. Failure is atomic with
    /// respect to acceptance: a rejected call publishes no member, while an
    /// indeterminate call returns the one event retaining the entire list.
    fn submit_ordered(
        &self,
        queue: &Self::Queue,
        executions: &[OrderedExecution<'_, Self::Program, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let _ = (queue, executions, timeout);
        Err(SubmitFailure::Rejected(BackendError::Unsupported))
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
