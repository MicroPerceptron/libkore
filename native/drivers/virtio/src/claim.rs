//! Exact device-broker claim contract for the HostLink VirtIO console.

use kore_driver::host::{BoundDevice, IntakeError};
use kore_rt::cap::{DeviceToken, SupplyToken};
use kore_rt::service::{DeviceSelector, PciSelector};
use kore_svc_device::client::{ClaimedDevice, DeviceClient};
use kore_svc_device::service::{DeviceRefusal, DeviceRequest};

use crate::binding::VirtioConsoleBinding;

/// QEMU's modern transitional-free VirtIO console PCI identity and the exact
/// authority its guest driver consumes.
pub const HOSTLINK_VIRTIO_CONSOLE: PciSelector =
    kore_driver::host::pci_selector_for_binding::<VirtioConsoleBinding>(
        0x07,
        0x80,
        0x00,
        kore_rt::service::PciIdentitySelector::exact(0x1af4, 0x1043),
        kore_driver::host::PciClaimPosture::RouteCapable,
    );

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimError {
    Broker(DeviceRefusal),
    WrongSpecies,
}

/// Claim the first exact HostLink VirtIO console from a device-broker face.
/// The returned affine root remains unbound; the service chooses when to
/// consume its DMA supply into [`bind_console`].
pub async fn claim_console<C: DeviceClient>(client: &mut C) -> Result<DeviceToken, ClaimError> {
    let claimed = client
        .claim(DeviceRequest::Claim(DeviceSelector::Pci(
            HOSTLINK_VIRTIO_CONSOLE,
        )))
        .await
        .map_err(ClaimError::Broker)?;
    match claimed.device {
        ClaimedDevice::Pci(device) => Ok(device),
        ClaimedDevice::Platform(_) => Err(ClaimError::WrongSpecies),
    }
}

/// Consume the exact claimed function and founded DMA supply through the
/// generic host lifecycle. This is the sole production construction path for
/// [`VirtioConsoleBinding`].
pub fn bind_console(
    device: DeviceToken,
    pool: SupplyToken,
) -> Result<BoundDevice<VirtioConsoleBinding>, IntakeError> {
    BoundDevice::claim(device, pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_is_identity_exact_and_requires_bar_dma_and_msi() {
        assert_eq!(
            (
                HOSTLINK_VIRTIO_CONSOLE.class,
                HOSTLINK_VIRTIO_CONSOLE.subclass,
                HOSTLINK_VIRTIO_CONSOLE.progif,
                HOSTLINK_VIRTIO_CONSOLE.identity.exact_pair(),
                HOSTLINK_VIRTIO_CONSOLE.bars.bits(),
                HOSTLINK_VIRTIO_CONSOLE.dma,
            ),
            (0x07, 0x80, 0x00, Some((0x1af4, 0x1043)), 1 << 4, true)
        );
        assert_eq!(HOSTLINK_VIRTIO_CONSOLE.interrupts,
            kore_rt::service::PciInterruptRequirement::Msix(kore_rt::service::PciMsixBlock::ENTRIES_0_1));
    }
}
