//! Founded block-channel control plane.

use kore_abi::task::role;
use kore_driver::host::{
    InterruptBlock, InterruptPolicy, PciBindingContract, PciClaimPosture, pci_selector_for_contract,
};
use kore_rt::cap::{DeviceToken, SupplyToken, verbs};
use kore_rt::service::{DeviceSelector, GrantArg, Serve, SystemService};

use crate::ProvisionRequest;

/// The one NVMe PCI contract shared by the block provisioner's claim ladder
/// and the filesystem binding that eventually realizes the controller.
pub struct NvmeIoPci;

impl PciBindingContract for NvmeIoPci {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR0;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Optional(InterruptBlock::SINGLE);
}

/// The founder gives the control plane one ordinary DMA-capable physical
/// supply. The NVMe controller itself is claimed from the device broker.
#[derive(kore_rt::Bundle)]
#[bundle(name = "block")]
pub struct BlockEndowment {
    #[bundle(role = role::DMA_POOL)]
    pub pool: SupplyToken,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRequest {
    Open(ProvisionRequest),
    Status,
    /// Revoke the sole live channel lineage after the client has quiesced it.
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRefusal {
    Unavailable,
    Policy,
    Exhausted,
    Lost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockResponse {
    /// Exactly two grants accompany this reply: device then DMA supply.
    Opened {
        channels: u16,
        limit: u16,
    },
    Status {
        available: bool,
        channels: u16,
        limit: u16,
    },
    Closed,
    Refused(BlockRefusal),
}

/// A single-controller provisioner. It retains the ordinary device and
/// physical-region ancestors and grants attenuated children; filesystem I/O
/// never re-enters this object.
pub struct BlockService {
    device: Option<DeviceToken>,
    pool: Option<SupplyToken>,
    live: bool,
    granted: u64,
    revoked: u64,
    note: u64,
}

impl BlockService {
    pub const LIMIT: u16 = 1;

    pub fn note(&self) -> u64 {
        self.note
    }

    fn available(&self) -> bool {
        self.device.is_some() && self.pool.is_some()
    }

    pub fn adopt_device(&mut self, device: DeviceToken) {
        self.device = Some(device);
    }
}

impl SystemService for BlockService {
    const NAME: &'static str = "block";
    /// An I/O worker: parked on completions for most of its life, its work
    /// done in microseconds between parks.  Keep it off the performance
    /// cores; a completion wait cannot tell the difference.
    const CORE_PREFERENCE: kore_abi::task::CorePreference =
        kore_abi::task::CorePreference::Efficient;
    type Endowment = BlockEndowment;
    const DEVICES: &'static [kore_rt::service::DeviceRequest] =
        &[kore_rt::service::DeviceRequest {
            selector: DeviceSelector::Pci(pci_selector_for_contract::<NvmeIoPci>(
                0x01,
                0x08,
                0x02,
                kore_rt::service::PciIdentitySelector::ANY,
                // The declared FLOOR of the claim ladder, deliberately not a
                // match constraint: the channel's owner realizes its own
                // MSI-family route from the granted device
                // (`InterruptPolicy::Optional`), so a controller on a
                // machine without remapped-MSI delivery still matches and
                // serves at the polled posture.  The entry's claim asks for
                // the interrupt-capable carrier FIRST
                // ([`preferred_selector`]) and only then falls back to this
                // row, so a mixed machine never pairs the filesystem with a
                // polled controller while a remapped one sits idle.
                PciClaimPosture::PollingFallback,
            )),
            required: false,
        }];

    /// The NVMe function this row claims is provisioned to the filesystem,
    /// which binds and — on the way down — stops it (#809 §L.5).
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] =
        &[kore_rt::service::DeviceClass::Delegated {
            to: "kore-svc-dagfs",
        }];

    fn bind(endowment: BlockEndowment) -> Self {
        Self {
            device: None,
            pool: Some(endowment.pool),
            live: false,
            granted: 0,
            revoked: 0,
            note: 0,
        }
    }

    fn serve(&mut self, request: BlockRequest, sv: &mut Serve<'_, BlockResponse>) {
        match request {
            BlockRequest::Status => sv.reply(BlockResponse::Status {
                available: self.available(),
                channels: u16::from(self.live),
                limit: Self::LIMIT,
            }),
            BlockRequest::Open(request) => {
                if !self.available() {
                    sv.reply(BlockResponse::Refused(BlockRefusal::Unavailable));
                    return;
                }
                if self.live {
                    sv.reply(BlockResponse::Refused(BlockRefusal::Exhausted));
                    return;
                }
                // Both completion postures are lawful policy: the channel's
                // owner builds the data plane over the granted device, and
                // whether its binding realizes an interrupt route is that
                // owner's declaration, not this control plane's.
                if request.requested_queue_depth == 0 || request.requested_buffer_bytes == 0 {
                    sv.reply(BlockResponse::Refused(BlockRefusal::Policy));
                    return;
                }
                let device = self.device.as_ref().expect("availability checked");
                let pool = self.pool.as_ref().expect("availability checked");
                sv.grant(
                    device.slot(),
                    device.rights(),
                    GrantArg::Attenuate {
                        kind: kore_abi::cap::ResourceKind::PciDevice,
                    },
                );
                sv.grant(
                    pool.slot(),
                    pool.rights(),
                    GrantArg::Sublease {
                        offset: 0,
                        len: pool.len(),
                    },
                );
                self.live = true;
                self.granted = self.granted.saturating_add(1);
                sv.reply(BlockResponse::Opened {
                    channels: 1,
                    limit: Self::LIMIT,
                });
            }
            BlockRequest::Close => {
                if !self.live {
                    sv.reply(BlockResponse::Refused(BlockRefusal::Lost));
                    return;
                }
                // One root per kind and one channel maximum: retiring both
                // ancestors cuts the exact provisioned bundle and all BAR,
                // VM and IOMMU descendants beneath it. Reopening requires a
                // fresh broker claim and supply, so this instance becomes
                // honestly unavailable.
                let device = self.device.take();
                let pool = self.pool.take();
                if let Some(device) = device {
                    let _ = verbs::revoke_slot(device.slot());
                }
                if let Some(pool) = pool {
                    let _ = verbs::revoke_slot(pool.slot());
                }
                self.live = false;
                self.revoked = self.revoked.saturating_add(1);
                sv.reply(BlockResponse::Closed);
            }
        }
    }

    fn grants_dropped(&mut self, response: &mut BlockResponse, emitted: usize) {
        if matches!(response, BlockResponse::Opened { .. }) {
            debug_assert_eq!(emitted, 2);
            self.live = false;
            *response = BlockResponse::Refused(BlockRefusal::Unavailable);
        }
    }
}

/// Run the founded provisioner's broker claim and serving loop on the
/// executor owned by `#[kore_rt::main]`.  Startup is ordinary async work: this
/// helper must never construct another executor over the task's sole
/// doorbell.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    use kore_rt::entry::{
        RUNG_SERVICE_ENDOWMENT, RUNG_SERVICE_SERVE_SETUP, RUNG_SERVICE_START, refuse_with,
    };
    use kore_rt::service::{Bundle, SystemService};

    let Ok(endowment) = BlockEndowment::resolve(rt.bootstrap()) else {
        refuse_with(&rt, RUNG_SERVICE_ENDOWMENT);
    };
    let mut service = BlockService::bind(endowment);
    let transport = kore_rt::entry::transport();
    let note = match claim_controller(transport, rt.bootstrap()).await {
        Ok(device) => {
            service.adopt_device(device);
            0
        }
        // A diskless machine still founds and serves the honest unavailable
        // posture; the readiness note preserves the broker's reason.
        Err(ClaimFailure::Unavailable(note)) => note,
        Err(ClaimFailure::Wiring) => refuse_with(&rt, RUNG_SERVICE_START),
    };
    service.note = note;

    // Pair 0 serves BlockService; pair 1 is this task's client face onto the
    // device broker. Observability is supplied through the ordinary monitor
    // provider seam, not a second service-specific status ring.
    kore_rt::entry::serve_forever(&rt, service, None, note).await;
    refuse_with(&rt, RUNG_SERVICE_SERVE_SETUP)
}

/// The claim ladder's first rung: the declared class row with the
/// remapped-MSI match constraint added.
///
/// The filesystem paces its completions by the route it realizes from the
/// granted device, so when a machine offers several matching controllers
/// the one whose MSI path the platform already remapped is strictly
/// preferable — the swap row's constraint, tried first.  The declared
/// selector remains the fallback floor, so an unremapped machine still
/// gets storage at the polled posture rather than none at all.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
fn preferred_selector(selector: DeviceSelector) -> DeviceSelector {
    match selector {
        DeviceSelector::Pci(pci) => DeviceSelector::Pci(pci_selector_for_contract::<NvmeIoPci>(
            pci.class,
            pci.subclass,
            pci.progif,
            pci.identity,
            PciClaimPosture::RouteCapable,
        )),
        other => other,
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClaimFailure {
    Wiring,
    Unavailable(u64),
}

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
async fn claim_controller(
    transport: &kore_rt::transport::TrapTransport,
    boot: &kore_abi::task::Bootstrap<'_>,
) -> Result<DeviceToken, ClaimFailure> {
    use kore_svc_device::DeviceRefusal;
    use kore_svc_device::client::{Claimed, ClaimedDevice, FaceLayout, MultiBroker};

    // Every broker face the founder stamped — one per live root complex
    // (#572).  @0/@1 are this service's own two outgoing grant channels;
    // the broker-client receive endpoints stride from @2, deliberately
    // disjoint — derived from the protocol's own pair and grant counts,
    // exactly as the founder stamps them.
    let mut broker = MultiBroker::open(
        transport,
        boot,
        FaceLayout::for_row(
            <BlockService as kore_rt::svc::SvcProtocol>::SERVE_PAIRS,
            <BlockService as kore_rt::svc::SvcProtocol>::BOOT_REPLY_GRANTS,
        ),
    )
    .ok_or(ClaimFailure::Wiring)?;
    let selector = BlockService::DEVICES[0].selector;
    // Rung one: the interrupt-capable carrier, sought across EVERY live
    // root before descending.  `NoMatch` and `Taken` are both honest
    // reasons to fall to the declared floor — either no remapped
    // controller exists anywhere, or another claimant already carries it
    // and a polled sibling is still better than no storage.
    match broker.claim_first(preferred_selector(selector)).await {
        Ok(Claimed {
            device: ClaimedDevice::Pci(device),
            ..
        }) => return Ok(device),
        // A non-PCI shape from a PCI ask is the same broker-contract
        // anomaly the floor claim refuses; never abandon a claimed bundle
        // to double-claim below it.
        Ok(_) => return Err(ClaimFailure::Unavailable(3)),
        Err(DeviceRefusal::NoMatch) | Err(DeviceRefusal::Taken) => {}
        Err(_) => return Err(ClaimFailure::Wiring),
    }
    match broker.claim_first(selector).await {
        Ok(Claimed {
            device: ClaimedDevice::Pci(device),
            ..
        }) => Ok(device),
        Ok(_) => Err(ClaimFailure::Unavailable(3)),
        Err(DeviceRefusal::NoMatch) => Err(ClaimFailure::Unavailable(1)),
        Err(DeviceRefusal::Taken) => Err(ClaimFailure::Unavailable(2)),
        Err(_) => Err(ClaimFailure::Wiring),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_rt::service::PciInterruptRequirement;

    #[test]
    fn nvme_selector_demands_dma_but_not_an_interrupt_route() {
        let DeviceSelector::Pci(selector) = BlockService::DEVICES[0].selector else {
            panic!("block controller must be PCI")
        };
        assert_eq!(
            (selector.class, selector.subclass, selector.progif),
            (1, 8, 2)
        );
        assert!(selector.dma);
        // The declared floor: matchable on machines without remapped MSI.
        assert_eq!(selector.interrupts, PciInterruptRequirement::None);
    }

    #[test]
    fn the_claim_ladder_prefers_the_remapped_carrier_over_the_same_floor() {
        let floor = BlockService::DEVICES[0].selector;
        let DeviceSelector::Pci(preferred) = preferred_selector(floor) else {
            panic!("a PCI floor must prefer a PCI carrier")
        };
        let DeviceSelector::Pci(declared) = floor else {
            panic!("block controller must be PCI")
        };
        // Rung one adds exactly the remapped-MSI constraint and nothing else,
        // so descending to the floor can only ever widen the match.
        assert_eq!(preferred.interrupts, PciInterruptRequirement::MsiFamily);
        assert_eq!(
            (
                preferred.class,
                preferred.subclass,
                preferred.progif,
                preferred.dma,
                preferred.bars,
                preferred.identity,
            ),
            (
                declared.class,
                declared.subclass,
                declared.progif,
                declared.dma,
                declared.bars,
                declared.identity,
            )
        );
    }
}
