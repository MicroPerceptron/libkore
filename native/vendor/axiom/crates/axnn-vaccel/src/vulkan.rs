//! Session construction for the virtio-accel Vulkan provider.
//!
//! The provider measures buffer alignment from the driver's mapped pointer
//! and device address rather than assuming it (its own suite requests page
//! alignment), so a session must ask for what a Vulkan allocation can prove.
//! Everything else — placement, artifact authoring, host staging — is the
//! generic `Session`/`AxHetero` path; this module exists so every consumer
//! opens the provider under one alignment contract instead of repeating it.

use virtio_accel_core::BackendError;
use virtio_accel_vulkan::{InitError, VULKAN_TOSA_TARGET, VulkanAccelerator};

use crate::session::Session;

/// Buffer alignment requested from Vulkan sessions: one page, the value the
/// provider's conformance run measures on every verified driver stack.
pub const BUFFER_ALIGN: u64 = 4 * 1024;

/// Why a Vulkan session could not be opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenError {
    Init(InitError),
    Session(BackendError),
}

impl core::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Init(error) => write!(formatter, "Vulkan initialization: {error}"),
            Self::Session(error) => write!(formatter, "Vulkan session: {error:?}"),
        }
    }
}

impl std::error::Error for OpenError {}

/// Whether an initialization error means "no device here" rather than a
/// fault: the condition under which examples and tests skip.
pub fn is_absent(error: InitError) -> bool {
    matches!(
        error,
        InitError::RuntimeUnavailable | InitError::DeviceUnavailable
    )
}

/// Open the named device, or the provider's preferred one when `None`.
pub fn open(device: Option<&str>) -> Result<VulkanAccelerator, InitError> {
    match device {
        Some(name) => VulkanAccelerator::with_device(name),
        None => VulkanAccelerator::new(),
    }
}

/// A session over `provider` at the Vulkan float target and alignment contract.
pub fn session(provider: VulkanAccelerator) -> Result<Session<VulkanAccelerator>, BackendError> {
    session_for(provider, VULKAN_TOSA_TARGET)
}

/// A session over `provider` at `target`, keeping the alignment contract.
///
/// The FP8 tier is a distinct target rather than a dtype narrowing of the float one, because
/// TOSA gates FP8 on an extension the float target does not carry. The alignment is the part
/// that must not vary: the adapter's 16 KiB default is refused by the Vulkan driver, which
/// measures alignment, so every session goes through [`BUFFER_ALIGN`].
pub fn session_for(
    provider: VulkanAccelerator,
    target: virtio_accel_tosa::Target,
) -> Result<Session<VulkanAccelerator>, BackendError> {
    Session::new_with_buffer_alignment(provider, target, BUFFER_ALIGN)
}

/// Open a device and stand up its session in one step.
pub fn open_session(device: Option<&str>) -> Result<Session<VulkanAccelerator>, OpenError> {
    let provider = open(device).map_err(OpenError::Init)?;
    session(provider).map_err(OpenError::Session)
}
