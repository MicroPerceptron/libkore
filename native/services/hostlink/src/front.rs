//! Manifest-visible HostLink front and its founded DMA endowment.
//!
//! The production entry replaces the disconnected posture with the concrete
//! VirtIO-owned core before advertising readiness. Keeping the manifest type
//! concrete lets the founder cite one protocol and one capability bundle;
//! transport selection remains below this front.

use kore_abi::task::role;
use kore_rt::cap::SupplyToken;
use kore_rt::service::{DeviceRequest, DeviceSelector, PciSelector, Serve, SystemService};
use kore_rt::svc::SvcProtocol;

use crate::{ClientRequest, ClientResponse, HostLinkWire, RepWire, ReqWire};

/// The mandatory control transport.  The exact modern VirtIO-console
/// function owns the control plane in every build and remains the serial
/// bulk fallback when no shared window is present.
pub const HOSTLINK_VIRTIO_CONSOLE: DeviceRequest = DeviceRequest {
    selector: DeviceSelector::Pci(kore_virtio::claim::HOSTLINK_VIRTIO_CONSOLE),
    required: true,
};

/// Optional QEMU `ivshmem-plain` bulk transport.
///
/// Its exact identity and BAR are part of the capability contract.  It is
/// coherent CPU-shared RAM, never a DMA or interrupting device, so asking the
/// broker for a binding, bus-master activation, or MSI would reject the only
/// lawful implementation.
pub const HOSTLINK_IVSHMEM_PLAIN: DeviceRequest = DeviceRequest {
    selector: DeviceSelector::Pci(PciSelector {
        class: 0x05,
        subclass: 0x00,
        progif: 0x00,
        identity: kore_rt::service::PciIdentitySelector::exact(0x1af4, 0x1110),
        bars: kore_rt::service::PciBarMask::BAR2,
        dma: false,
        interrupts: kore_rt::service::PciInterruptRequirement::None,
    }),
    required: false,
};

/// One measured service image, selected solely by available hardware:
/// production machines and the serial baseline omit the optional row, while
/// the Linux ivshmem arm supplies it.  Keeping this one binary means the A/B
/// comparison changes QEMU topology, not code or measured-boot identity.
pub const HOSTLINK_DEVICES: &[DeviceRequest] = &[HOSTLINK_VIRTIO_CONSOLE, HOSTLINK_IVSHMEM_PLAIN];

#[derive(kore_rt::Bundle)]
#[bundle(name = "hostlink")]
pub struct HostLinkEndowment {
    #[bundle(role = role::DMA_POOL)]
    pub pool: SupplyToken,
}

/// The manifest's concrete service type.
///
/// Its generic-host posture is deliberately disconnected. The production
/// entry consumes the pool, claims the exact console through the broker, and
/// installs the device-backed router before it beacons.
pub struct HostLinkFront {
    pool: Option<SupplyToken>,
}

impl HostLinkFront {
    pub fn take_pool(&mut self) -> Option<SupplyToken> {
        self.pool.take()
    }
}

impl SvcProtocol for HostLinkFront {
    type Request = ClientRequest;
    type Response = ClientResponse;
    type ReqWire = ReqWire;
    type RepWire = RepWire;

    const REPLY_GRANTS: usize = HostLinkWire::REPLY_GRANTS;
    const REQUEST_CAPACITY: u32 = HostLinkWire::REQUEST_CAPACITY;
    const RESPONSE_CAPACITY: u32 = HostLinkWire::RESPONSE_CAPACITY;
    const SERVE_PAIRS: usize = HostLinkWire::SERVE_PAIRS;

    fn encode_req(request: &Self::Request) -> Self::ReqWire {
        HostLinkWire::encode_req(request)
    }

    fn decode_req(wire: &Self::ReqWire) -> Option<Self::Request> {
        HostLinkWire::decode_req(wire)
    }

    fn encode_rep(response: &Self::Response, grants: u8) -> Self::RepWire {
        HostLinkWire::encode_rep(response, grants)
    }

    fn decode_rep(wire: &Self::RepWire) -> Option<(Self::Response, u8)> {
        HostLinkWire::decode_rep(wire)
    }

    fn reply_matches(request: &Self::ReqWire, response: &Self::Response, grants: u8) -> bool {
        HostLinkWire::reply_matches(request, response, grants)
    }
}

impl SystemService for HostLinkFront {
    const NAME: &'static str = "hostlink";
    type Endowment = HostLinkEndowment;

    const DEVICES: &'static [DeviceRequest] = HOSTLINK_DEVICES;
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] = &[
        kore_driver::host::device_class::<kore_virtio::binding::VirtioConsoleBinding>(),
        kore_rt::service::DeviceClass::NoDma,
    ];

    fn bind(endowment: Self::Endowment) -> Self {
        Self {
            pool: Some(endowment.pool),
        }
    }

    fn serve(&mut self, request: Self::Request, reply: &mut Serve<'_, Self::Response>) {
        // The custom production host owns the live core and never enters this
        // disconnected fallback. Keeping it lawful makes accidental use of
        // the generic entry fail closed instead of silently accepting work.
        reply.reply(ClientResponse::Refused {
            request: request.request_id(),
            code: kore_hostlink::RefusalCode::RouteClosed,
        });
    }
}

const _: () = {
    assert!(HostLinkFront::DEVICES.len() == 2);
    let selector = match HOSTLINK_VIRTIO_CONSOLE.selector {
        DeviceSelector::Pci(selector) => selector,
        _ => panic!("HostLink must claim PCI"),
    };
    match selector.identity.exact_pair() {
        Some((0x1af4, 0x1043)) => {}
        _ => panic!("HostLink PCI identity drifted"),
    }
    assert!(selector.bars.bits() == 1 << 4);
    assert!(
        selector.dma
            && matches!(
                selector.interrupts,
                kore_rt::service::PciInterruptRequirement::Msix(_)
            )
    );
    assert!(HOSTLINK_VIRTIO_CONSOLE.required);

    let selector = match HOSTLINK_IVSHMEM_PLAIN.selector {
        DeviceSelector::Pci(selector) => selector,
        _ => panic!("HostLink shared RAM must be PCI"),
    };
    match selector.identity.exact_pair() {
        Some((0x1af4, 0x1110)) => {}
        _ => panic!("HostLink ivshmem identity drifted"),
    }
    assert!(selector.class == 0x05 && selector.subclass == 0 && selector.progif == 0);
    assert!(selector.bars.bits() == 1 << 2);
    assert!(
        !selector.dma
            && matches!(
                selector.interrupts,
                kore_rt::service::PciInterruptRequirement::None
            )
    );
    assert!(!HOSTLINK_IVSHMEM_PLAIN.required);
};

#[cfg(test)]
mod tests {
    use super::*;
    use kore_rt::svc::{SvcProtocol, client_rings, init_window};

    #[test]
    fn founder_client_uses_the_hostlink_protocol_geometry() {
        // The production founder gives this pair a page-rounded 0x3000-byte
        // view. Keep that headroom here so the obsolete generic 8/8 attach
        // fails because its ring header disagrees, not merely because a
        // minimally-sized test allocation ends too early.
        let mut window = std::vec![0u8; 0x3000];
        init_window::<HostLinkFront>(
            &mut window,
            HostLinkFront::REQUEST_CAPACITY,
            HostLinkFront::RESPONSE_CAPACITY,
        )
        .expect("founder formats HostLink's declared 2/2 geometry");
        assert!(
            client_rings::<HostLinkFront>(
                &mut window,
                HostLinkFront::REQUEST_CAPACITY,
                HostLinkFront::RESPONSE_CAPACITY,
            )
            .is_ok(),
            "the founder-held client must attach with HostLink's capacities"
        );
        assert!(
            client_rings::<HostLinkFront>(
                &mut window,
                kore_rt::svc::REQ_CAP,
                kore_rt::svc::REP_CAP,
            )
            .is_err(),
            "generic 8/8 capacities must fail closed against HostLink's 2/2 header"
        );
    }
}
