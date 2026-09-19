//! Runtime-selected ivshmem bulk composition for HostLink.
//!
//! Control frames and every wake edge remain on the exact VirtIO-console
//! transport. Only complete bulk frames enter the typed coherent BAR2 window.
//! The adapter re-attaches to the checked shared layout for each operation;
//! queue cursors live in the window, so this avoids a self-referential owner
//! and adds no second wire format or allocation.

use kore_hostlink::{
    ENVELOPE_LEN, HostLinkTransport, LinkEpoch, PortError, RequestId, RouteId,
    SHMEM_QUEUE_CAPACITY, SHMEM_SLOT_PAYLOAD_BYTES, SHMEM_WAKE, SHMEM_WAKE_LEN, ShmemWakeV1,
    derive_shmem_epoch, link_envelope,
};
use kore_hostlink_shmem::{Epoch, ReceiveError, SendError, Side, WindowError, endpoint};
use kore_rt::cap::{DeviceToken, SharedRamWindow};
use kore_svc_device::client::{ClaimedDevice, DeviceClient};
use kore_svc_device::service::{DeviceRefusal, DeviceRequest};
use zerocopy::IntoBytes;

/// Active bytes formatted within the power-of-two QEMU BAR backing.
/// Kept as a frozen integration constant and checked against `Layout` in tests.
pub const HOSTLINK_SHMEM_ACTIVE_BYTES: usize = 16_779_264;
/// BAR/backing size used by the QEMU A/B topology.
pub const HOSTLINK_SHMEM_BAR_BYTES: usize = 32 * 1024 * 1024;

/// Why an optional exact-device probe was not a lawful serial fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShmemClaimError {
    Broker(DeviceRefusal),
    WrongSpecies,
    MapRefused,
}

/// Claim and map ivshmem when present. Absence alone selects serial bulk;
/// a taken, malformed, or unmappable matching function fails closed so an A/B
/// shmem run cannot silently measure serial.
pub async fn claim_optional_shared_ram<C: DeviceClient>(
    client: &mut C,
) -> Result<Option<MappedSharedRam>, ShmemClaimError> {
    let claimed = match client
        .claim(DeviceRequest::Claim(
            crate::front::HOSTLINK_IVSHMEM_PLAIN.selector,
        ))
        .await
    {
        Ok(claimed) => claimed,
        Err(DeviceRefusal::NoMatch) => return Ok(None),
        Err(error) => return Err(ShmemClaimError::Broker(error)),
    };
    let device = match claimed.device {
        ClaimedDevice::Pci(device) => device,
        ClaimedDevice::Platform(_) => return Err(ShmemClaimError::WrongSpecies),
    };
    MappedSharedRam::new(device)
        .map(Some)
        .map_err(|_| ShmemClaimError::MapRefused)
}

/// The mapped affine child followed by its exact device parent. Field order
/// makes normal drop retire the mapping before its derivation root.
#[derive(Debug)]
pub struct MappedSharedRam {
    window: SharedRamWindow,
    _device: DeviceToken,
}

impl MappedSharedRam {
    pub fn new(device: DeviceToken) -> Result<Self, DeviceToken> {
        let Some(window) = kore_rt::cap::verbs::map_shared_ram(&device) else {
            return Err(device);
        };
        Ok(Self {
            window,
            _device: device,
        })
    }
}

/// Stable, exclusively borrowed mapped bytes. Static dispatch keeps the metal
/// owner affine while allowing aligned test storage to exercise the same code.
pub trait SharedWindow {
    fn with_bytes<R>(&mut self, use_window: impl FnOnce(&mut [u8]) -> R) -> R;
}

impl SharedWindow for MappedSharedRam {
    fn with_bytes<R>(&mut self, use_window: impl FnOnce(&mut [u8]) -> R) -> R {
        self.window.with_bytes(use_window)
    }
}

/// Serial control plus a shared-memory bulk endpoint.
pub struct ShmemBulk<S, W> {
    serial: S,
    window: W,
    link_epoch: Option<LinkEpoch>,
    memory_epoch: Option<Epoch>,
    publication: u64,
    wake_pending: Option<u64>,
    rx_offset: usize,
}

impl<S, W> ShmemBulk<S, W> {
    pub const fn new(serial: S, window: W) -> Self {
        Self {
            serial,
            window,
            link_epoch: None,
            memory_epoch: None,
            publication: 0,
            wake_pending: None,
            rx_offset: 0,
        }
    }

    fn set_link_epoch(&mut self, epoch: Option<LinkEpoch>) {
        if self.link_epoch == epoch {
            return;
        }
        self.link_epoch = epoch;
        self.memory_epoch = epoch.and_then(derive_shmem_epoch).and_then(Epoch::new);
        self.publication = 0;
        self.wake_pending = None;
        self.rx_offset = 0;
    }

    pub fn serial(&self) -> &S {
        &self.serial
    }

    pub fn serial_mut(&mut self) -> &mut S {
        &mut self.serial
    }
}

impl<S: HostLinkTransport, W> ShmemBulk<S, W> {
    /// Attempt the one coalesced wake. Serial backpressure retains the latest
    /// publication instead of reporting the already-published bulk as unsent.
    pub fn flush_wake(&mut self) -> Result<bool, PortError> {
        let Some(publication) = self.wake_pending else {
            return Ok(true);
        };
        let epoch = self.link_epoch.ok_or(PortError::Disconnected)?;
        let wake = ShmemWakeV1::new(publication).ok_or(PortError::Fault)?;
        let envelope = link_envelope(
            SHMEM_WAKE,
            epoch,
            RouteId::ZERO,
            0,
            0,
            RequestId::ZERO,
            SHMEM_WAKE_LEN,
        );
        let mut frame = [0u8; ENVELOPE_LEN + SHMEM_WAKE_LEN];
        frame[..ENVELOPE_LEN].copy_from_slice(envelope.as_bytes());
        frame[ENVELOPE_LEN..].copy_from_slice(wake.as_bytes());
        match self.serial.try_send_control(&frame) {
            Ok(()) => {
                // A later publication cannot occur while `&mut self` is held.
                self.wake_pending = None;
                Ok(true)
            }
            Err(PortError::Backpressure) => Ok(false),
            Err(error) => Err(error),
        }
    }
}

impl<S: HostLinkTransport, W: SharedWindow> HostLinkTransport for ShmemBulk<S, W> {
    fn synchronize_link_epoch(&mut self, epoch: Option<LinkEpoch>) -> Result<(), PortError> {
        self.set_link_epoch(epoch);
        self.serial.synchronize_link_epoch(epoch)
    }

    fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError> {
        // Protocol control owns priority. A pending hint follows only after
        // the caller's complete control frame has been accepted.
        self.serial.try_send_control(frame)?;
        let _ = self.flush_wake()?;
        Ok(())
    }

    fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        let _ = self.flush_wake()?;
        self.serial.try_recv_control(output)
    }

    fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError> {
        let epoch = self.memory_epoch.ok_or(PortError::Disconnected)?;
        let result = self.window.with_bytes(|bytes| {
            let mut queue = endpoint(bytes, epoch, Side::A).map_err(map_window)?;
            queue.tx().try_push(frame).map_err(map_send)
        });
        result?;
        self.publication = self.publication.checked_add(1).ok_or(PortError::Fault)?;
        self.wake_pending = Some(self.publication);
        let _ = self.flush_wake()?;
        Ok(())
    }

    fn try_recv_bulk(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        let epoch = self.memory_epoch.ok_or(PortError::Disconnected)?;
        let offset = self.rx_offset;
        let progress = self.window.with_bytes(|bytes| {
            let mut queue = endpoint(bytes, epoch, Side::A).map_err(map_window)?;
            match queue.rx().try_pop_chunk(output, offset) {
                Ok(progress) => Ok(Some(progress)),
                Err(ReceiveError::Empty) => Ok(None),
                Err(error) => Err(map_receive(error)),
            }
        })?;
        let Some(progress) = progress else {
            return Ok(0);
        };
        self.rx_offset = if progress.complete {
            0
        } else {
            self.rx_offset
                .checked_add(progress.copied)
                .ok_or(PortError::Fault)?
        };
        Ok(progress.copied)
    }
}

/// One always-built transport selected from actual hardware inventory.
pub enum RuntimeBulk<S, W> {
    Serial(S),
    Shared(ShmemBulk<S, W>),
}

impl<S, W> RuntimeBulk<S, W> {
    pub const fn serial(serial: S) -> Self {
        Self::Serial(serial)
    }

    pub const fn shared(serial: S, window: W) -> Self {
        Self::Shared(ShmemBulk::new(serial, window))
    }

    #[must_use]
    pub const fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    pub fn serial_mut(&mut self) -> &mut S {
        match self {
            Self::Serial(serial) => serial,
            Self::Shared(shared) => shared.serial_mut(),
        }
    }

    /// Release the optional shared window and return the control device owner.
    pub fn into_serial(self) -> S {
        match self {
            Self::Serial(serial) => serial,
            Self::Shared(shared) => shared.serial,
        }
    }

    /// Synchronize transport reset state after the control-plane handshake.
    pub fn set_link_epoch(&mut self, epoch: Option<LinkEpoch>) {
        if let Self::Shared(shared) = self {
            shared.set_link_epoch(epoch);
        }
    }
}

impl<S: HostLinkTransport, W> RuntimeBulk<S, W> {
    pub fn flush_wake(&mut self) -> Result<bool, PortError> {
        match self {
            Self::Serial(_) => Ok(true),
            Self::Shared(shared) => shared.flush_wake(),
        }
    }
}

impl<S: HostLinkTransport, W: SharedWindow> HostLinkTransport for RuntimeBulk<S, W> {
    fn synchronize_link_epoch(&mut self, epoch: Option<LinkEpoch>) -> Result<(), PortError> {
        match self {
            Self::Serial(serial) => serial.synchronize_link_epoch(epoch),
            Self::Shared(shared) => shared.synchronize_link_epoch(epoch),
        }
    }

    fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError> {
        match self {
            Self::Serial(serial) => serial.try_send_control(frame),
            Self::Shared(shared) => shared.try_send_control(frame),
        }
    }

    fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        match self {
            Self::Serial(serial) => serial.try_recv_control(output),
            Self::Shared(shared) => shared.try_recv_control(output),
        }
    }

    fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError> {
        match self {
            Self::Serial(serial) => serial.try_send_bulk(frame),
            Self::Shared(shared) => shared.try_send_bulk(frame),
        }
    }

    fn try_recv_bulk(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
        match self {
            Self::Serial(serial) => serial.try_recv_bulk(output),
            Self::Shared(shared) => shared.try_recv_bulk(output),
        }
    }
}

fn map_window(error: WindowError) -> PortError {
    match error {
        WindowError::EpochMismatch { .. } | WindowError::ZeroEpoch => PortError::Disconnected,
        WindowError::Misaligned
        | WindowError::TooSmall
        | WindowError::BadHeader
        | WindowError::WrongLayout
        | WindowError::SameEpoch => PortError::Fault,
    }
}

fn map_send(error: SendError) -> PortError {
    match error {
        SendError::Full => PortError::Backpressure,
        SendError::EpochChanged(_) => PortError::Disconnected,
        SendError::TooLarge { .. } | SendError::CorruptCursor => PortError::Fault,
    }
}

fn map_receive(error: ReceiveError) -> PortError {
    match error {
        ReceiveError::Empty => PortError::Backpressure,
        ReceiveError::EpochChanged(_) => PortError::Disconnected,
        ReceiveError::CorruptCursor
        | ReceiveError::StaleSlot { .. }
        | ReceiveError::CorruptLength { .. }
        | ReceiveError::OutputTooSmall { .. }
        | ReceiveError::InvalidOffset { .. } => PortError::Fault,
    }
}

/// A full queue can be drained in this many bounded service passes. One extra
/// pass is the empty recheck that closes the coalesced-wake race before park.
pub const SHMEM_DRAIN_RECHECK_PASSES: usize = SHMEM_QUEUE_CAPACITY as usize + 1;

/// A saturated bounded drain means the caller must recheck rather than park.
#[must_use]
pub const fn needs_bulk_recheck(shared: bool, drained: u8, budget: usize) -> bool {
    shared && drained as usize == budget
}

const _: () = assert!(SHMEM_SLOT_PAYLOAD_BYTES as usize >= kore_hostlink::BULK_FRAME_MAX);
const _: () = {
    assert!(HOSTLINK_SHMEM_ACTIVE_BYTES <= HOSTLINK_SHMEM_BAR_BYTES);
    assert!(HOSTLINK_SHMEM_BAR_BYTES.is_power_of_two());
};

#[cfg(test)]
mod tests {
    extern crate std;

    use std::collections::VecDeque;
    use std::vec;
    use std::vec::Vec;

    use kore_hostlink::{EnvelopeV1, shmem_wake_envelope_is_valid};
    use kore_hostlink_shmem::{Layout, init};
    use kore_rt::service::{DeviceSelector, PciSelector};
    use zerocopy::FromBytes;

    use super::*;

    struct TestWindow {
        storage: Vec<u8>,
        offset: usize,
    }

    impl TestWindow {
        fn new() -> Self {
            let storage = vec![0; HOSTLINK_SHMEM_BAR_BYTES + kore_hostlink_shmem::ALIGN - 1];
            let offset = storage.as_ptr().align_offset(kore_hostlink_shmem::ALIGN);
            assert!(offset < kore_hostlink_shmem::ALIGN);
            Self { storage, offset }
        }
    }

    impl SharedWindow for TestWindow {
        fn with_bytes<R>(&mut self, use_window: impl FnOnce(&mut [u8]) -> R) -> R {
            use_window(&mut self.storage[self.offset..self.offset + HOSTLINK_SHMEM_BAR_BYTES])
        }
    }

    #[derive(Default)]
    struct MockSerial {
        control: Vec<Vec<u8>>,
        bulk: Vec<Vec<u8>>,
        control_rx: VecDeque<Vec<u8>>,
        reject_control: bool,
    }

    impl HostLinkTransport for MockSerial {
        fn try_send_control(&mut self, frame: &[u8]) -> Result<(), PortError> {
            if self.reject_control {
                return Err(PortError::Backpressure);
            }
            self.control.push(frame.to_vec());
            Ok(())
        }

        fn try_recv_control(&mut self, output: &mut [u8]) -> Result<usize, PortError> {
            let Some(frame) = self.control_rx.pop_front() else {
                return Ok(0);
            };
            if output.len() < frame.len() {
                return Err(PortError::Fault);
            }
            output[..frame.len()].copy_from_slice(&frame);
            Ok(frame.len())
        }

        fn try_send_bulk(&mut self, frame: &[u8]) -> Result<(), PortError> {
            self.bulk.push(frame.to_vec());
            Ok(())
        }

        fn try_recv_bulk(&mut self, _output: &mut [u8]) -> Result<usize, PortError> {
            Ok(0)
        }
    }

    fn epoch() -> LinkEpoch {
        LinkEpoch([7; 16])
    }

    fn adapter(reject_control: bool) -> ShmemBulk<MockSerial, TestWindow> {
        let memory_epoch = Epoch::new(derive_shmem_epoch(epoch()).unwrap()).unwrap();
        let layout = Layout::new(SHMEM_QUEUE_CAPACITY, SHMEM_SLOT_PAYLOAD_BYTES).unwrap();
        assert_eq!(layout.window_bytes(), HOSTLINK_SHMEM_ACTIVE_BYTES);
        let mut window = TestWindow::new();
        window.with_bytes(|bytes| init(bytes, layout, memory_epoch).unwrap());
        let mut adapter = ShmemBulk::new(
            MockSerial {
                reject_control,
                ..MockSerial::default()
            },
            window,
        );
        adapter.set_link_epoch(Some(epoch()));
        adapter
    }

    fn wake_publication(frame: &[u8]) -> u64 {
        let envelope = EnvelopeV1::read_from_bytes(&frame[..ENVELOPE_LEN]).unwrap();
        assert!(shmem_wake_envelope_is_valid(&envelope, epoch()));
        ShmemWakeV1::read_from_bytes(&frame[ENVELOPE_LEN..])
            .unwrap()
            .publication()
            .unwrap()
    }

    #[test]
    fn selector_and_geometry_are_exact_and_non_dma() {
        assert_eq!(
            crate::front::HOSTLINK_IVSHMEM_PLAIN,
            kore_rt::service::DeviceRequest {
                selector: DeviceSelector::Pci(PciSelector {
                    class: 0x05,
                    subclass: 0,
                    progif: 0,
                    identity: kore_rt::service::PciIdentitySelector::exact(0x1af4, 0x1110),
                    bars: kore_rt::service::PciBarMask::BAR2,
                    dma: false,
                    interrupts: kore_rt::service::PciInterruptRequirement::None,
                }),
                required: false,
            },
        );
        let layout = Layout::new(SHMEM_QUEUE_CAPACITY, SHMEM_SLOT_PAYLOAD_BYTES).unwrap();
        assert_eq!(layout.window_bytes(), HOSTLINK_SHMEM_ACTIVE_BYTES);
    }

    #[test]
    fn serial_mode_delegates_both_planes_unchanged() {
        let mut adapter = RuntimeBulk::<_, TestWindow>::serial(MockSerial::default());
        adapter.try_send_control(b"control").unwrap();
        adapter.try_send_bulk(b"bulk").unwrap();
        let serial = adapter.serial_mut();
        assert_eq!(serial.control, [b"control".to_vec()]);
        assert_eq!(serial.bulk, [b"bulk".to_vec()]);
    }

    #[test]
    fn shared_bulk_never_enters_the_serial_bulk_plane() {
        let mut adapter = adapter(false);
        adapter.try_send_bulk(b"bulk-one").unwrap();
        assert!(adapter.serial.bulk.is_empty());
        assert_eq!(adapter.serial.control.len(), 1);
        assert_eq!(wake_publication(&adapter.serial.control[0]), 1);

        let memory_epoch = Epoch::new(derive_shmem_epoch(epoch()).unwrap()).unwrap();
        let mut output = [0u8; 32];
        let bytes = adapter.window.with_bytes(|window| {
            endpoint(window, memory_epoch, Side::B)
                .unwrap()
                .rx()
                .try_pop(&mut output)
                .unwrap()
        });
        assert_eq!(&output[..bytes], b"bulk-one");
    }

    #[test]
    fn wake_backpressure_coalesces_without_replaying_published_bulk() {
        let mut adapter = adapter(true);
        adapter.try_send_bulk(b"one").unwrap();
        adapter.try_send_bulk(b"two").unwrap();
        assert_eq!(adapter.wake_pending, Some(2));
        assert!(adapter.serial.control.is_empty());
        assert!(adapter.serial.bulk.is_empty());

        adapter.serial.reject_control = false;
        assert_eq!(adapter.flush_wake(), Ok(true));
        assert_eq!(adapter.serial.control.len(), 1);
        assert_eq!(wake_publication(&adapter.serial.control[0]), 2);

        let memory_epoch = Epoch::new(derive_shmem_epoch(epoch()).unwrap()).unwrap();
        let mut output = [0u8; 8];
        adapter.window.with_bytes(|window| {
            let mut host = endpoint(window, memory_epoch, Side::B).unwrap();
            assert_eq!(host.rx().try_pop(&mut output).unwrap(), 3);
            assert_eq!(&output[..3], b"one");
            assert_eq!(host.rx().try_pop(&mut output).unwrap(), 3);
            assert_eq!(&output[..3], b"two");
            assert!(matches!(
                host.rx().try_pop(&mut output),
                Err(ReceiveError::Empty)
            ));
        });
    }

    #[test]
    fn inbound_shared_records_are_bounded_and_empty_is_not_an_error() {
        let mut adapter = adapter(false);
        let memory_epoch = Epoch::new(derive_shmem_epoch(epoch()).unwrap()).unwrap();
        adapter.window.with_bytes(|window| {
            endpoint(window, memory_epoch, Side::B)
                .unwrap()
                .tx()
                .try_push(b"from-host")
                .unwrap();
        });
        let mut output = [0u8; 16];
        assert_eq!(adapter.try_recv_bulk(&mut output), Ok(9));
        assert_eq!(&output[..9], b"from-host");
        assert_eq!(adapter.try_recv_bulk(&mut output), Ok(0));
        assert_eq!(adapter.try_recv_bulk(&mut [0u8; 2]), Ok(0));
    }

    #[test]
    fn deterministic_frame_sweep_streams_records_across_decoder_sized_reads() {
        let mut adapter = adapter(false);
        let memory_epoch = Epoch::new(derive_shmem_epoch(epoch()).unwrap()).unwrap();
        for length in [1usize, 87, 88, 4095, 4096, 4097, 8193, 65_537] {
            let payload: Vec<u8> = (0..length)
                .map(|index| (index as u8).wrapping_mul(29).wrapping_add(length as u8))
                .collect();
            adapter.window.with_bytes(|window| {
                endpoint(window, memory_epoch, Side::B)
                    .unwrap()
                    .tx()
                    .try_push(&payload)
                    .unwrap();
            });
            let mut observed = Vec::new();
            let mut chunk = [0u8; 4096];
            while observed.len() < payload.len() {
                let copied = adapter.try_recv_bulk(&mut chunk).unwrap();
                assert!(copied != 0);
                observed.extend_from_slice(&chunk[..copied]);
            }
            assert_eq!(observed, payload, "length {length}");
            assert_eq!(adapter.try_recv_bulk(&mut chunk), Ok(0));
        }
    }

    #[test]
    fn epoch_change_fails_stale_window_closed_until_host_reset() {
        let mut adapter = adapter(false);
        let next = LinkEpoch([8; 16]);
        adapter.set_link_epoch(Some(next));
        assert_eq!(
            adapter.try_send_bulk(b"stale"),
            Err(PortError::Disconnected)
        );
        assert_eq!(adapter.publication, 0);
        assert_eq!(adapter.wake_pending, None);
    }

    #[test]
    fn saturated_drain_requires_recheck_and_empty_observation_allows_park() {
        assert!(needs_bulk_recheck(true, 2, 2));
        assert!(!needs_bulk_recheck(true, 1, 2));
        assert!(!needs_bulk_recheck(false, 2, 2));
        assert_eq!(
            SHMEM_DRAIN_RECHECK_PASSES,
            SHMEM_QUEUE_CAPACITY as usize + 1
        );
    }
}
